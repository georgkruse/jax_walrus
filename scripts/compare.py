"""
Pretrained weight equivalence test: verify that the JAX model with converted
weights produces the same output as the PyTorch model with original weights.

Modified to work with deterministic synthetic inputs instead of a dataset.

Usage:
    python compare.py
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

jax.config.update("jax_platform_name", "cpu")

# ── Resolve paths ──

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_WALRUS_PT = PROJECT_ROOT / "models" / "walrus.pt"
DEFAULT_JAX_WALRUS_MSGPACK = PROJECT_ROOT / "models" / "walrus.msgpack"

# ── Mock the_well ──

import types

_well_mod = types.ModuleType("the_well")
_data_mod = types.ModuleType("the_well.data")
_ds_mod = types.ModuleType("the_well.data.datasets")


class _BoundaryConditionEnum:
    _map = {
        "PERIODIC": type("BC", (), {"value": 2})(),
        "OPEN": type("BC", (), {"value": 0})(),
    }

    def __getitem__(self, key):
        return self._map[key]


_ds_mod.BoundaryCondition = _BoundaryConditionEnum()
_well_mod.data = _data_mod
_data_mod.datasets = _ds_mod
sys.modules.setdefault("the_well", _well_mod)
sys.modules.setdefault("the_well.data", _data_mod)
sys.modules.setdefault("the_well.data.datasets", _ds_mod)

np.random.seed(0)
torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare local Walrus PyTorch and JAX checkpoints on synthetic inputs"
    )
    parser.add_argument(
        "--walrus-root",
        type=Path,
        required=True,
        help="Path to the local Walrus repository",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_WALRUS_PT,
        help="Path to walrus.pt",
    )
    parser.add_argument(
        "--msgpack-path",
        type=Path,
        default=DEFAULT_JAX_WALRUS_MSGPACK,
        help="Path to walrus.msgpack",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for deterministic synthetic input generation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Batch size for synthetic inputs",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=2,
        help="Number of timesteps for synthetic inputs",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Synthetic input height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Synthetic input width",
    )
    parser.add_argument(
        "--target-depth",
        type=int,
        default=32,
        help="Depth dimension used after tiling 2D synthetic data to 3D",
    )
    return parser.parse_args()


def import_walrus_modules(walrus_root: Path):
    sys.path.insert(0, str(walrus_root))

    from walrus.models.encoders.vstride_encoder import (
        SpaceBagAdaptiveDVstrideEncoder as TorchSpaceBagEncoder,
    )
    from walrus.models.decoders.vstride_decoder import (
        AdaptiveDVstrideDecoder as TorchAdaptiveDecoder,
    )
    from walrus.models.spatiotemporal_blocks.space_time_split import (
        SpaceTimeSplitBlock as TorchBlock,
    )
    from walrus.models.spatial_blocks.full_attention import (
        FullAttention as TorchFullAttn,
    )
    from walrus.models.temporal_blocks.axial_time_attention import (
        AxialTimeAttention as TorchAxialTime,
    )
    from walrus.models.shared_utils.normalization import (
        RMSGroupNorm as TorchRMSGroupNorm,
    )

    return {
        "TorchSpaceBagEncoder": TorchSpaceBagEncoder,
        "TorchAdaptiveDecoder": TorchAdaptiveDecoder,
        "TorchBlock": TorchBlock,
        "TorchFullAttn": TorchFullAttn,
        "TorchAxialTime": TorchAxialTime,
        "TorchRMSGroupNorm": TorchRMSGroupNorm,
    }


def import_jax_walrus_modules(project_root: Path):
    sys.path.insert(0, str(project_root))

    from jax_walrus.model import IsotropicModel as JaxIsotropicModel
    from jax_walrus.convert_weights import (
        convert_pytorch_to_jax_params,
        load_pytorch_state_dict,
        torch_to_numpy,
    )

    return {
        "JaxIsotropicModel": JaxIsotropicModel,
        "convert_pytorch_to_jax_params": convert_pytorch_to_jax_params,
        "load_pytorch_state_dict": load_pytorch_state_dict,
        "torch_to_numpy": torch_to_numpy,
    }


# ═══════════════════════════════════════════════════════════════════════════════

# Synthetic Input Generation

# ═══════════════════════════════════════════════════════════════════════════════


def generate_synthetic_input(
    num_samples=1,
    num_timesteps=2,
    num_channels=5,
    height=128,
    width=128,
    seed=0,
):
    """
    Generate deterministic smooth 2D synthetic fields.

    Args:
        num_samples: Batch size
        num_timesteps: Number of timesteps
        num_channels: Number of physical channels
        height: Spatial height
        width: Spatial width
        seed: Random seed

    Returns:
        data: numpy array of shape (T, B, C, H, W)
        metadata: dict with synthetic input info
    """
    print("\nGenerating synthetic smooth input fields...")
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, 2.0 * np.pi, width, dtype=np.float32)
    y = np.linspace(0.0, 2.0 * np.pi, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    data = np.zeros(
        (num_timesteps, num_samples, num_channels, height, width), dtype=np.float32
    )

    for t in range(num_timesteps):
        time_phase = 2.0 * np.pi * t / max(num_timesteps, 1)
        for b in range(num_samples):
            for c in range(num_channels):
                field = np.zeros((height, width), dtype=np.float32)
                for _ in range(4):
                    kx = rng.integers(1, 5)
                    ky = rng.integers(1, 5)
                    amplitude = rng.uniform(0.5, 1.5)
                    phase = rng.uniform(0.0, 2.0 * np.pi)
                    field += amplitude * np.sin(kx * xx + phase + time_phase)
                    field += 0.5 * amplitude * np.cos(ky * yy - phase + 0.5 * time_phase)
                data[t, b, c] = field

    mean = data.mean(axis=(0, 1, 3, 4), keepdims=True)
    std = data.std(axis=(0, 1, 3, 4), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    data_normalized = (data - mean) / std

    metadata = {
        "kind": "synthetic",
        "num_samples": num_samples,
        "num_timesteps": num_timesteps,
        "original_shape": data.shape,
        "channels": ["density", "u_x", "u_y", "pressure", "tracer"],
        "mean": mean.squeeze(),
        "std": std.squeeze(),
        "domain": "unit square [0,1]^2",
        "spatial_resolution": (height, width),
        "seed": seed,
    }

    print(f"  Selected data shape: {data.shape} (T, B, C, H, W)")
    print(f"  Channels: {metadata['channels']}")
    print(f"  Per-channel mean: {metadata['mean']}")
    print(f"  Per-channel std:  {metadata['std']}")
    print(f"  Raw data range: [{data.min():.4f}, {data.max():.4f}]")
    print(
        f"  Normalized range: [{data_normalized.min():.4f}, {data_normalized.max():.4f}]"
    )

    return data_normalized.astype(np.float32), metadata


def prepare_2d_to_3d(data_2d, target_depth=32):
    """
    Convert 2D data to 3D by tiling along depth dimension.

    The CE-RM dataset is 2D (128x128), but the WALRUS model expects 3D input.
    We tile the 2D slices to create a pseudo-3D volume.

    Args:
        data_2d: Shape (T, B, C, H, W)
        target_depth: Desired depth dimension

    Returns:
        data_3d: Shape (T, B, C, H, W, D)
    """
    T, B, C, H, W = data_2d.shape
    # Tile along new depth axis
    data_3d = np.tile(data_2d[:, :, :, :, :, np.newaxis], (1, 1, 1, 1, 1, target_depth))
    print(f"  Converted 2D {data_2d.shape} -> 3D {data_3d.shape}")
    return data_3d


def add_coordinate_channels(data_3d):
    """
    Add dx, dy, dz coordinate channels required by SpaceBag encoder.

    The SpaceBag encoder expects field_indices that include extra dims
    [2, 0, 1] corresponding to dx/dy/dz coordinate information.

    Args:
        data_3d: Shape (T, B, C, H, W, D)

    Returns:
        data_with_coords: Shape (T, B, C+3, H, W, D)
    """
    T, B, C, H, W, D = data_3d.shape

    # Create normalized coordinate grids [0, 1]
    x_coords = np.linspace(0, 1, H, dtype=np.float32)
    y_coords = np.linspace(0, 1, W, dtype=np.float32)
    z_coords = np.linspace(0, 1, D, dtype=np.float32)

    # Broadcast to full shape (T, B, 1, H, W, D)
    dx = np.broadcast_to(
        x_coords[None, None, None, :, None, None], (T, B, 1, H, W, D)
    ).copy()
    dy = np.broadcast_to(
        y_coords[None, None, None, None, :, None], (T, B, 1, H, W, D)
    ).copy()
    dz = np.broadcast_to(
        z_coords[None, None, None, None, None, :], (T, B, 1, H, W, D)
    ).copy()

    # Concatenate: [physical channels, dx, dy, dz]
    data_with_coords = np.concatenate([data_3d, dx, dy, dz], axis=2)

    print(f"  Added coordinate channels: {data_3d.shape} -> {data_with_coords.shape}")
    return data_with_coords


def pytorch_to_jax_layout(x):
    """Convert (T, B, C, H, W, D) to JAX input layout (B, T, H, W, D, C)."""
    x = np.swapaxes(x, 0, 1)
    return np.moveaxis(x, 2, -1)


def jax_to_pytorch_layout(x):
    """Convert JAX output layout (B, T, H, W, D, C) to (T, B, C, H, W, D)."""
    x = np.moveaxis(x, -1, 2)
    return np.swapaxes(x, 0, 1)


# ═══════════════════════════════════════════════════════════════════════════════

# Comparison Utilities

# ═══════════════════════════════════════════════════════════════════════════════


def assert_close(name, torch_out, jax_out, atol=5e-3, rtol=5e-3):
    from jax_walrus.convert_weights import torch_to_numpy

    t_np = torch_to_numpy(torch_out)
    j_np = np.asarray(jax_out)
    max_diff = np.max(np.abs(t_np - j_np))
    mean_diff = np.mean(np.abs(t_np - j_np))
    print(f"  [{name}] shape: torch={t_np.shape} jax={j_np.shape}")
    print(f"           max_diff={max_diff:.2e} mean_diff={mean_diff:.2e}")
    if not np.allclose(t_np, j_np, atol=atol, rtol=rtol):
        print(f"  [FAIL] MISMATCH (atol={atol}, rtol={rtol})")
        return False
    print(f"  [PASS] MATCH")
    return True


# ═══════════════════════════════════════════════════════════════════════════════

# Visualization Functions

# ═══════════════════════════════════════════════════════════════════════════════


def visualize_input(data, metadata, save_path="synthetic_input.png"):
    """
    Visualize the synthetic input data.
    """
    # Handle 3D data by taking middle slice
    if len(data.shape) == 6:
        T, B, C, H, W, D = data.shape
        data_2d = data[:, :, :, :, :, D // 2]
    else:
        T, B, C, H, W = data.shape
        data_2d = data

    channels = metadata["channels"]
    n_channels = min(len(channels), C)  # Don't show coordinate channels
    n_timesteps = min(T, 4)

    fig, axes = plt.subplots(
        n_channels, n_timesteps, figsize=(4 * n_timesteps, 3.5 * n_channels)
    )
    if n_channels == 1:
        axes = axes[np.newaxis, :]
    if n_timesteps == 1:
        axes = axes[:, np.newaxis]

    for c_idx in range(n_channels):
        for t_idx in range(n_timesteps):
            ax = axes[c_idx, t_idx]
            img = data_2d[t_idx, 0, c_idx]
            im = ax.imshow(img, cmap="viridis", origin="lower", aspect="equal")
            ax.set_title(f"{channels[c_idx]}, t={t_idx}", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if t_idx == 0:
                ax.set_ylabel("y")
            if c_idx == n_channels - 1:
                ax.set_xlabel("x")

    fig.suptitle(
        f"Synthetic Input\n"
        f"Shape: {data.shape}, Domain: {metadata['domain']}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Input visualization saved to: {save_path}")


def visualize_comparison(
    torch_out, jax_out, channel_names=None, save_path="output_comparison.png"
):
    """
    Visualize PyTorch vs JAX output comparison.
    """
    from jax_walrus.convert_weights import torch_to_numpy

    t_np = torch_to_numpy(torch_out)
    j_np = np.asarray(jax_out)
    diff = t_np - j_np

    # Handle 2D and 3D
    if len(t_np.shape) == 6:
        T, B, C, H, W, D = t_np.shape
        d_slice = D // 2
        is_3d = True
    else:
        T, B, C, H, W = t_np.shape
        d_slice = None
        is_3d = False

    n_channels = min(C, 5)
    n_timesteps = min(T, 2)

    if channel_names is None:
        channel_names = [f"Ch {i}" for i in range(C)]

    fig = plt.figure(figsize=(16, 4 * n_timesteps * n_channels + 2))

    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    rel_diff = np.mean(np.abs(diff) / (np.abs(t_np) + 1e-8))

    fig.suptitle(
        f"PyTorch vs JAX Output Comparison\n"
        f"Max Diff: {max_diff:.2e} | Mean Diff: {mean_diff:.2e} | "
        f"Rel Diff: {rel_diff:.2e}",
        fontsize=14,
        fontweight="bold",
    )

    gs = GridSpec(
        n_timesteps * n_channels + 1,
        4,
        figure=fig,
        height_ratios=[1] * (n_timesteps * n_channels) + [0.3],
        hspace=0.4,
        wspace=0.3,
    )

    row = 0
    for t in range(n_timesteps):
        for c in range(n_channels):
            if is_3d:
                pt_slice = t_np[t, 0, c, :, :, d_slice]
                jax_slice = j_np[t, 0, c, :, :, d_slice]
                diff_slice = diff[t, 0, c, :, :, d_slice]
            else:
                pt_slice = t_np[t, 0, c, :, :]
                jax_slice = j_np[t, 0, c, :, :]
                diff_slice = diff[t, 0, c, :, :]

            vmin = min(pt_slice.min(), jax_slice.min())
            vmax = max(pt_slice.max(), jax_slice.max())

            # PyTorch
            ax1 = fig.add_subplot(gs[row, 0])
            im1 = ax1.imshow(
                pt_slice, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower"
            )
            ax1.set_title(f"PyTorch: {channel_names[c]}, t={t}", fontsize=9)
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # JAX
            ax2 = fig.add_subplot(gs[row, 1])
            im2 = ax2.imshow(
                jax_slice, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower"
            )
            ax2.set_title(f"JAX: {channel_names[c]}, t={t}", fontsize=9)
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # Difference
            ax3 = fig.add_subplot(gs[row, 2])
            diff_max = max(abs(diff_slice.min()), abs(diff_slice.max()), 1e-10)
            im3 = ax3.imshow(
                diff_slice, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max, origin="lower"
            )
            ax3.set_title(f"Diff: max={np.abs(diff_slice).max():.2e}", fontsize=9)
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # Scatter
            ax4 = fig.add_subplot(gs[row, 3])
            ax4.scatter(pt_slice.flatten(), jax_slice.flatten(), alpha=0.3, s=1)
            lims = [vmin, vmax]
            ax4.plot(lims, lims, "r--", linewidth=1, label="y=x")
            ax4.set_xlim(lims)
            ax4.set_ylim(lims)
            ax4.set_xlabel("PyTorch", fontsize=8)
            ax4.set_ylabel("JAX", fontsize=8)
            ax4.set_aspect("equal")
            corr = np.corrcoef(pt_slice.flatten(), jax_slice.flatten())[0, 1]
            ax4.set_title(f"Corr: r={corr:.6f}", fontsize=9)

            row += 1

    # Histogram
    ax_hist = fig.add_subplot(gs[-1, :])
    all_diff = diff.flatten()
    ax_hist.hist(all_diff, bins=100, color="steelblue", edgecolor="black", alpha=0.7)
    ax_hist.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax_hist.set_xlabel("Difference (PyTorch - JAX)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(
        f"Difference Distribution: μ={np.mean(all_diff):.2e}, σ={np.std(all_diff):.2e}"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison visualization saved to: {save_path}")

    return max_diff, mean_diff, rel_diff


def visualize_3d_slices(torch_out, jax_out, save_path="output_3d_slices.png"):
    """
    Visualize 3D output with slices along all axes.
    """
    from jax_walrus.convert_weights import torch_to_numpy

    t_np = torch_to_numpy(torch_out)
    j_np = np.asarray(jax_out)

    if len(t_np.shape) != 6:
        print("  Skipping 3D slices (data is 2D)")
        return

    diff = t_np - j_np
    T, B, C, H, W, D = t_np.shape

    t, b, c = 0, 0, 0
    pt_vol = t_np[t, b, c]
    jax_vol = j_np[t, b, c]
    diff_vol = diff[t, b, c]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    slices = [
        (
            "XY (z=mid)",
            pt_vol[:, :, D // 2],
            jax_vol[:, :, D // 2],
            diff_vol[:, :, D // 2],
        ),
        (
            "XZ (y=mid)",
            pt_vol[:, W // 2, :],
            jax_vol[:, W // 2, :],
            diff_vol[:, W // 2, :],
        ),
        (
            "YZ (x=mid)",
            pt_vol[H // 2, :, :],
            jax_vol[H // 2, :, :],
            diff_vol[H // 2, :, :],
        ),
    ]

    for row, (name, pt_s, jax_s, diff_s) in enumerate(slices):
        vmin = min(pt_s.min(), jax_s.min())
        vmax = max(pt_s.max(), jax_s.max())

        im1 = axes[row, 0].imshow(
            pt_s, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower"
        )
        axes[row, 0].set_title(f"PyTorch - {name}")
        plt.colorbar(im1, ax=axes[row, 0])

        im2 = axes[row, 1].imshow(
            jax_s, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower"
        )
        axes[row, 1].set_title(f"JAX - {name}")
        plt.colorbar(im2, ax=axes[row, 1])

        diff_max = max(abs(diff_s.min()), abs(diff_s.max()), 1e-10)
        im3 = axes[row, 2].imshow(
            diff_s, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max, origin="lower"
        )
        axes[row, 2].set_title(f"Diff - {name}\nmax={np.abs(diff_s).max():.2e}")
        plt.colorbar(im3, ax=axes[row, 2])

        axes[row, 3].scatter(pt_s.flatten(), jax_s.flatten(), alpha=0.5, s=2)
        lims = [vmin, vmax]
        axes[row, 3].plot(lims, lims, "r--", linewidth=1)
        axes[row, 3].set_xlabel("PyTorch")
        axes[row, 3].set_ylabel("JAX")
        corr = np.corrcoef(pt_s.flatten(), jax_s.flatten())[0, 1]
        axes[row, 3].set_title(f"Correlation: r={corr:.6f}")

    fig.suptitle(
        f"3D Volume Comparison (t=0, ch=0)\nShape: {t_np.shape}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  3D slices saved to: {save_path}")


def visualize_channel_statistics(
    torch_out, jax_out, channel_names=None, save_path="channel_stats.png"
):
    """
    Per-channel statistics comparison.
    """
    from jax_walrus.convert_weights import torch_to_numpy

    t_np = torch_to_numpy(torch_out)
    j_np = np.asarray(jax_out)

    if len(t_np.shape) == 6:
        axes_reduce = (0, 1, 3, 4, 5)
    else:
        axes_reduce = (0, 1, 3, 4)

    C = t_np.shape[2]
    if channel_names is None:
        channel_names = [f"Ch {i}" for i in range(C)]

    pt_means = t_np.mean(axis=axes_reduce)
    pt_stds = t_np.std(axis=axes_reduce)
    jax_means = j_np.mean(axis=axes_reduce)
    jax_stds = j_np.std(axis=axes_reduce)

    diff = t_np - j_np
    diff_means = diff.mean(axis=axes_reduce)
    diff_max_abs = np.abs(diff).max(axis=axes_reduce)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(C)
    width = 0.35

    # Means
    axes[0, 0].bar(x - width / 2, pt_means, width, label="PyTorch", alpha=0.8)
    axes[0, 0].bar(x + width / 2, jax_means, width, label="JAX", alpha=0.8)
    axes[0, 0].set_ylabel("Mean")
    axes[0, 0].set_title("Per-Channel Mean")
    axes[0, 0].legend()
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(channel_names, rotation=45, ha="right")

    # Stds
    axes[0, 1].bar(x - width / 2, pt_stds, width, label="PyTorch", alpha=0.8)
    axes[0, 1].bar(x + width / 2, jax_stds, width, label="JAX", alpha=0.8)
    axes[0, 1].set_ylabel("Std Dev")
    axes[0, 1].set_title("Per-Channel Std Dev")
    axes[0, 1].legend()
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(channel_names, rotation=45, ha="right")

    # Mean difference
    axes[1, 0].bar(x, diff_means, width, color="green", alpha=0.8)
    axes[1, 0].axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    axes[1, 0].set_ylabel("Mean Difference")
    axes[1, 0].set_title("Per-Channel Mean Difference (PT - JAX)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(channel_names, rotation=45, ha="right")

    # Max absolute difference
    axes[1, 1].bar(x, diff_max_abs, width, color="red", alpha=0.8)
    axes[1, 1].set_ylabel("Max |Difference|")
    axes[1, 1].set_title("Per-Channel Max Absolute Difference")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(channel_names, rotation=45, ha="right")

    fig.suptitle("Channel-wise Statistics Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Channel statistics saved to: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════

# Main

# ═══════════════════════════════════════════════════════════════════════════════


def main():
    args = parse_args()

    if not args.walrus_root.exists():
        raise FileNotFoundError(f"Walrus repository not found: {args.walrus_root}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Walrus checkpoint not found: {args.checkpoint_path}")
    if not args.msgpack_path.exists():
        raise FileNotFoundError(f"JAX msgpack not found: {args.msgpack_path}")
    walrus_modules = import_walrus_modules(args.walrus_root)
    jax_walrus_modules = import_jax_walrus_modules(PROJECT_ROOT)
    TorchSpaceBagEncoder = walrus_modules["TorchSpaceBagEncoder"]
    TorchAdaptiveDecoder = walrus_modules["TorchAdaptiveDecoder"]
    TorchBlock = walrus_modules["TorchBlock"]
    TorchFullAttn = walrus_modules["TorchFullAttn"]
    TorchAxialTime = walrus_modules["TorchAxialTime"]
    TorchRMSGroupNorm = walrus_modules["TorchRMSGroupNorm"]
    JaxIsotropicModel = jax_walrus_modules["JaxIsotropicModel"]
    convert_pytorch_to_jax_params = jax_walrus_modules["convert_pytorch_to_jax_params"]
    load_pytorch_state_dict = jax_walrus_modules["load_pytorch_state_dict"]

    print("=" * 70)
    print("Synthetic Input: PyTorch vs JAX Model Comparison")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════
    # Configuration
    # ═══════════════════════════════════════════════════════════════════

    # Model config
    hidden_dim = 1408
    intermediate_dim = 352
    n_states = 67
    processor_blocks = 40
    num_heads = 16
    groups = 16
    bks_3d = ((8, 4), (8, 4), (8, 4))
    causal = True
    bias_type = "rel"
    extra_dims = 3

    # Dataset config
    NUM_SAMPLES = args.num_samples
    NUM_TIMESTEPS = args.num_timesteps
    TARGET_DEPTH = args.target_depth

    stride1 = (8, 8, 8)
    stride2 = (4, 4, 4)
    random_kernel = ((8, 4), (8, 4), (8, 4))
    bcs = [[2, 2], [2, 2], [2, 2]]  # periodic

    # ═══════════════════════════════════════════════════════════════════
    # Step 1: Load CE-RM Dataset
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 1: Generating Synthetic Input")
    print("=" * 70)

    print(f"  Walrus repo: {args.walrus_root}")
    print(f"  PyTorch checkpoint: {args.checkpoint_path}")
    print(f"  JAX msgpack: {args.msgpack_path}")
    print(f"  Seed: {args.seed}")

    data_2d, metadata = generate_synthetic_input(
        num_samples=NUM_SAMPLES,
        num_timesteps=NUM_TIMESTEPS,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    # Convert to 3D
    print("\nConverting 2D -> 3D...")
    data_3d = prepare_2d_to_3d(data_2d, target_depth=TARGET_DEPTH)

    # Add coordinate channels
    print("\nAdding coordinate channels...")
    x_np = add_coordinate_channels(data_3d)

    # Visualize input
    visualize_input(data_3d, metadata, save_path="synthetic_input.png")

    # Setup field indices
    # CE-RM has 5 physical channels
    n_out_states = 5
    state_labels_np = np.arange(n_out_states, dtype=np.int64)
    field_indices_np = np.concatenate(
        [state_labels_np, np.array([2, 0, 1], dtype=np.int64)]
    )

    print(f"\n  Final input shape: {x_np.shape}")
    print(f"  Output states: {n_out_states}")
    print(f"  State labels: {state_labels_np}")
    print(f"  Field indices: {field_indices_np}")

    x_jax = pytorch_to_jax_layout(x_np)
    print(f"  JAX input shape: {x_jax.shape}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 2: Load PyTorch Checkpoint
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 2: Loading PyTorch Checkpoint")
    print("=" * 70)

    t0 = time.time()
    sd = load_pytorch_state_dict(str(args.checkpoint_path))
    print(f"  Loaded {len(sd)} parameters in {time.time() - t0:.1f}s")

    encoder_dummy_val = sd["encoder_dummy"].numpy()
    print(f"  encoder_dummy = {encoder_dummy_val}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 3: Build PyTorch Model
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 3: Building PyTorch Model")
    print("=" * 70)

    # Encoder
    torch_enc = TorchSpaceBagEncoder(
        kernel_scales_seq=((4, 4),),
        base_kernel_size3d=bks_3d,
        input_dim=n_states,
        inner_dim=intermediate_dim,
        output_dim=hidden_dim,
        spatial_dims=3,
        groups=groups,
        extra_dims=extra_dims,
        variable_downsample=True,
        variable_deterministic_ds=True,
        learned_pad=True,
        norm_layer=TorchRMSGroupNorm,
        activation=nn.SiLU,
    )
    torch_enc.eval()
    torch_enc.load_state_dict(
        {
            "proj1.weight": sd["embed.3.proj1.weight"],
            "norm1.weight": sd["embed.3.norm1.weight"],
            "proj2.weight": sd["embed.3.proj2.weight"],
            "norm2.weight": sd["embed.3.norm2.weight"],
        }
    )

    # Decoder
    torch_dec = TorchAdaptiveDecoder(
        base_kernel_size3d=bks_3d,
        input_dim=hidden_dim,
        inner_dim=intermediate_dim,
        output_dim=n_states,
        spatial_dims=3,
        groups=groups,
        learned_pad=True,
        norm_layer=TorchRMSGroupNorm,
        activation=nn.SiLU,
    )
    torch_dec.eval()
    torch_dec.load_state_dict(
        {
            "proj1.weight": sd["debed.3.proj1.weight"],
            "norm1.weight": sd["debed.3.norm1.weight"],
            "proj2.weight": sd["debed.3.proj2.weight"],
            "proj2.bias": sd["debed.3.proj2.bias"],
        }
    )

    # Processor blocks
    torch_blocks = nn.ModuleList()
    for i in range(processor_blocks):
        blk = TorchBlock(
            space_mixing=partial(TorchFullAttn, num_heads=num_heads, mlp_dim=None),
            time_mixing=partial(
                TorchAxialTime, num_heads=num_heads, bias_type=bias_type
            ),
            channel_mixing=partial(nn.Identity),
            hidden_dim=hidden_dim,
            drop_path=0.0,
            causal_in_time=causal,
            norm_layer=TorchRMSGroupNorm,
        )
        blk.eval()

        blk_sd = {
            k[len(f"blocks.{i}.") :]: v
            for k, v in sd.items()
            if k.startswith(f"blocks.{i}.")
        }
        blk.load_state_dict(blk_sd)
        torch_blocks.append(blk)

    print(f"  Built encoder + {processor_blocks} blocks + decoder")

    # ═══════════════════════════════════════════════════════════════════
    # Step 4: Load JAX Checkpoint
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 4: Loading JAX Checkpoint")
    print("=" * 70)

    t0 = time.time()
    from flax.serialization import from_bytes

    target = convert_pytorch_to_jax_params(sd, processor_blocks=40, dim_keys=[2, 3])

    def to_jax_arrays(d):
        if isinstance(d, dict):
            return {k: to_jax_arrays(v) for k, v in d.items()}
        return jnp.array(d)

    target_jnp = {"params": to_jax_arrays(target["params"])}

    with open(args.msgpack_path, "rb") as f:
        jax_params = from_bytes(target_jnp, f.read())
    print(f"  Loaded JAX params in {time.time() - t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # Step 5: Weight Sanity Check
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 5: Weight Sanity Check")
    print("=" * 70)

    checks = [
        (
            "encoder_dummy",
            sd["encoder_dummy"].numpy(),
            np.array(jax_params["params"]["encoder_dummy"]),
        ),
        (
            "embed_3.proj1_weight",
            sd["embed.3.proj1.weight"].numpy(),
            np.array(jax_params["params"]["embed_3"]["proj1_weight"]),
        ),
        (
            "blocks_0.space_mixing.fused_ff_qkv.kernel",
            sd["blocks.0.space_mixing.fused_ff_qkv.weight"].numpy().T,
            np.array(
                jax_params["params"]["blocks_0"]["space_mixing"]["fused_ff_qkv"][
                    "kernel"
                ]
            ),
        ),
    ]

    all_match = True
    for name, pt_w, jax_w in checks:
        diff = np.max(np.abs(pt_w - jax_w))
        status = "✓" if diff < 1e-6 else "✗"
        print(f"  {status} {name}: max_diff={diff:.2e}")
        if diff >= 1e-6:
            all_match = False

    if not all_match:
        print("\n  [FAIL] Weight mismatch!")
        sys.exit(1)
    print("  All weights match!")

    # ═══════════════════════════════════════════════════════════════════
    # Step 6: PyTorch Forward Pass
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 6: PyTorch Forward Pass")
    print("=" * 70)

    t0 = time.time()
    with torch.no_grad():
        x_pt = torch.from_numpy(x_np)
        x_pt = x_pt * torch.from_numpy(encoder_dummy_val)

        # Encode
        x_pt, _ = torch_enc(
            x_pt, torch.from_numpy(field_indices_np), random_kernel=random_kernel
        )
        print(f"  Encoded: {x_pt.shape}")

        # Process
        for ii, blk in enumerate(torch_blocks):
            x_pt, _ = blk(x_pt, (bcs,), return_att=False)
            if (ii + 1) % 10 == 0:
                print(f"  Block {ii + 1}/{processor_blocks}")

        if not causal:
            x_pt = x_pt[-1:]

        # Decode
        torch_out = torch_dec(
            x_pt,
            torch.from_numpy(state_labels_np),
            bcs,
            stage_info={"random_kernel": random_kernel},
        )
        print(f"  Decoded: {torch_out.shape}")

    pt_time = time.time() - t0
    print(f"  PyTorch time: {pt_time:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # Step 7: JAX Forward Pass
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 7: JAX Forward Pass")
    print("=" * 70)

    t0 = time.time()

    jax_model = JaxIsotropicModel(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        n_states=n_states,
        processor_blocks=processor_blocks,
        groups=groups,
        num_heads=num_heads,
        mlp_dim=0,
        max_d=3,
        causal_in_time=causal,
        drop_path=0.0,
        bias_type=bias_type,
        base_kernel_size=bks_3d,
        use_spacebag=True,
        use_silu=True,
        include_d=(2, 3),
        encoder_groups=groups,
        learned_pad=False,
        jitter_patches=False,
    )

    jax_out = jax_model.apply(
        jax_params,
        jnp.array(x_jax),
        jnp.array(state_labels_np),
        bcs,
        stride1=stride1,
        stride2=stride2,
        field_indices=jnp.array(field_indices_np),
        dim_key=3,
    )

    jax_out = jax_to_pytorch_layout(np.asarray(jax_out))

    jax_time = time.time() - t0
    print(f"  JAX output: {jax_out.shape}")
    print(f"  JAX time: {jax_time:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # Step 8: Numerical Comparison
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 8: Numerical Comparison")
    print("=" * 70)

    passed = assert_close("Full Model Output", torch_out, jax_out, atol=5e-2, rtol=5e-2)

    # ═══════════════════════════════════════════════════════════════════
    # Step 9: Visualizations
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 9: Generating Visualizations")
    print("=" * 70)

    channel_names = metadata["channels"]

    max_diff, mean_diff, rel_diff = visualize_comparison(
        torch_out,
        jax_out,
        channel_names=channel_names,
        save_path="output_comparison.png",
    )

    visualize_3d_slices(torch_out, jax_out, save_path="output_3d_slices.png")

    visualize_channel_statistics(
        torch_out, jax_out, channel_names=channel_names, save_path="channel_stats.png"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Dataset:              Synthetic smooth fields")
    print(f"  Input shape:          {x_np.shape}")
    print(f"  Output shape:         {torch_out.shape}")
    print(f"  Channels:             {metadata['channels']}")
    print(f"  Seed:                 {metadata['seed']}")
    print(f"  Max absolute diff:    {max_diff:.2e}")
    print(f"  Mean absolute diff:   {mean_diff:.2e}")
    print(f"  Mean relative diff:   {rel_diff:.2e}")
    print(f"  PyTorch time:         {pt_time:.1f}s")
    print(f"  JAX time:             {jax_time:.1f}s")
    print("=" * 70)

    if passed:
        print("\n✓ SUCCESS: JAX model matches PyTorch model!")
        print("\nVisualization files saved:")
        print("  - synthetic_input.png")
        print("  - output_comparison.png")
        print("  - output_3d_slices.png")
        print("  - channel_stats.png")
    else:
        print("\n✗ FAIL: Models diverge. Check visualizations.")
        sys.exit(1)


if __name__ == "__main__":
    main()
