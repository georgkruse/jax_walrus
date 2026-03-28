import argparse
import importlib.util
import sys
from pathlib import Path
from urllib.request import urlretrieve

import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes, to_bytes


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_DIR = PROJECT_ROOT / "jax_walrus"

for path in (PROJECT_ROOT, SCRIPT_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

CONVERT_WEIGHTS_PATH = PACKAGE_DIR / "convert_weights.py"
CONVERT_WEIGHTS_SPEC = importlib.util.spec_from_file_location(
    "jax_walrus_convert_weights", CONVERT_WEIGHTS_PATH
)
if CONVERT_WEIGHTS_SPEC is None or CONVERT_WEIGHTS_SPEC.loader is None:
    raise ImportError(f"Unable to load {CONVERT_WEIGHTS_PATH}")
CONVERT_WEIGHTS_MODULE = importlib.util.module_from_spec(CONVERT_WEIGHTS_SPEC)
CONVERT_WEIGHTS_SPEC.loader.exec_module(CONVERT_WEIGHTS_MODULE)

convert_pytorch_to_jax_params = CONVERT_WEIGHTS_MODULE.convert_pytorch_to_jax_params
load_pytorch_state_dict = CONVERT_WEIGHTS_MODULE.load_pytorch_state_dict


WALRUS_URL = "https://huggingface.co/polymathic-ai/walrus/resolve/main/walrus.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, convert, and test the pretrained Walrus checkpoint"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory used to store downloaded and converted model files",
    )
    parser.add_argument(
        "--processor-blocks",
        type=int,
        default=40,
        help="Number of processor blocks in the model",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download walrus.pt even if it already exists",
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Recreate walrus.msgpack even if it already exists",
    )
    return parser.parse_args()


def ensure_models_dir(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def ensure_downloaded(models_dir: Path, force: bool = False) -> Path:
    checkpoint_path = models_dir / "walrus.pt"
    if checkpoint_path.exists() and not force:
        print(f"Reusing existing PyTorch checkpoint: {checkpoint_path}")
        return checkpoint_path

    print("Downloading Walrus checkpoint from Hugging Face...")
    urlretrieve(WALRUS_URL, checkpoint_path)
    print(f"Saved PyTorch checkpoint to: {checkpoint_path}")
    return checkpoint_path


def to_jax_arrays(tree):
    if isinstance(tree, dict):
        return {key: to_jax_arrays(value) for key, value in tree.items()}
    return jnp.array(tree)


def flatten_params(tree, prefix=""):
    result = []
    for key, value in sorted(tree.items()):
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.extend(flatten_params(value, path))
        else:
            result.append((path, value))
    return result


def convert_checkpoint(
    checkpoint_path: Path,
    msgpack_path: Path,
    processor_blocks: int,
    force: bool = False,
):
    print(f"Loading PyTorch checkpoint: {checkpoint_path}")
    state_dict = load_pytorch_state_dict(str(checkpoint_path))
    print(f"  PyTorch state_dict: {len(state_dict)} parameters")

    print("Converting to JAX parameter tree...")
    jax_params = convert_pytorch_to_jax_params(
        state_dict,
        processor_blocks=processor_blocks,
        dim_keys=[2, 3],
    )
    jax_params_jnp = {"params": to_jax_arrays(jax_params["params"])}

    if msgpack_path.exists() and not force:
        print(f"Reusing existing JAX msgpack: {msgpack_path}")
        return jax_params_jnp, msgpack_path

    print(f"Saving JAX msgpack to: {msgpack_path}")
    with msgpack_path.open("wb") as f:
        f.write(to_bytes(jax_params_jnp))
    return jax_params_jnp, msgpack_path


def compare_weights(jax_params_jnp, msgpack_path: Path) -> dict:
    with msgpack_path.open("rb") as f:
        msgpack_bytes = f.read()

    loaded_params = from_bytes(jax_params_jnp, msgpack_bytes)
    flat_converted = flatten_params(jax_params_jnp["params"])
    flat_loaded = flatten_params(loaded_params["params"])

    sum_sq = 0.0
    max_abs = 0.0
    exact = 0
    total_elements = 0

    for (orig_path, orig_arr), (load_path, load_arr) in zip(flat_converted, flat_loaded):
        if orig_path != load_path:
            raise ValueError(f"Path mismatch: {orig_path} vs {load_path}")

        diff = np.asarray(orig_arr) - np.asarray(load_arr)
        abs_diff = np.abs(diff)
        total_elements += int(np.asarray(orig_arr).size)
        sum_sq += float(np.sum(diff * diff))
        if abs_diff.size:
            max_abs = max(max_abs, float(abs_diff.max()))
        if np.array_equal(np.asarray(orig_arr), np.asarray(load_arr)):
            exact += 1

    return {
        "keys_converted": len(flat_converted),
        "keys_msgpack": len(flat_loaded),
        "exact_tensors": exact,
        "global_l2": float(np.sqrt(sum_sq)),
        "global_max_abs": max_abs,
        "total_elements": total_elements,
    }


def print_summary(checkpoint_path: Path, msgpack_path: Path, metrics: dict) -> None:
    print("\n=== SUMMARY ===")
    print(f"pytorch_checkpoint: {checkpoint_path}")
    print(f"jax_msgpack: {msgpack_path}")
    print(f"checkpoint_size_mb: {checkpoint_path.stat().st_size / (1024 * 1024):.1f}")
    print(f"msgpack_size_mb: {msgpack_path.stat().st_size / (1024 * 1024):.1f}")
    print(f"keys_converted: {metrics['keys_converted']}")
    print(f"keys_msgpack: {metrics['keys_msgpack']}")
    print(
        f"exact_tensors: {metrics['exact_tensors']}/{metrics['keys_converted']}"
    )
    print(f"total_elements: {metrics['total_elements']:,}")
    print(f"global_l2: {metrics['global_l2']:.6e}")
    print(f"global_max_abs: {metrics['global_max_abs']:.6e}")


def main() -> None:
    args = parse_args()
    models_dir = ensure_models_dir(args.models_dir)
    checkpoint_path = ensure_downloaded(models_dir, force=args.force_download)
    msgpack_path = models_dir / "walrus.msgpack"
    jax_params_jnp, msgpack_path = convert_checkpoint(
        checkpoint_path,
        msgpack_path,
        processor_blocks=args.processor_blocks,
        force=args.force_convert,
    )
    metrics = compare_weights(jax_params_jnp, msgpack_path)
    print_summary(checkpoint_path, msgpack_path, metrics)


if __name__ == "__main__":
    main()