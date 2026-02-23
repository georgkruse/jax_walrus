"""
IsotropicModel: JAX/Flax translation of walrus.models.isotropic_model.IsotropicModel

Top-level model that composes encoder -> processor blocks -> decoder.
Maintains exact 1-to-1 weight compatibility with the PyTorch model, including
both 2D and 3D encoder/decoder variants (embed_2/embed_3, debed_2/debed_3)
and the encoder_dummy parameter.

Supports both training and inference:
- ``deterministic=True``  (default): no dropout, no jitter, no periodic rolling
- ``deterministic=False``: enables drop_path, input field dropout, patch
  jittering, and periodic rolling (matching PyTorch ``model.train()`` mode)

RNG keys used during training (``deterministic=False``):
- ``"dropout"``: input field dropout (Bonferroni-corrected dropout3d)
- ``"drop_path"``: stochastic depth in processor blocks
- ``"jitter"``: patch jittering and periodic rolling
"""

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

from jax_walrus.encoder import AdaptiveDVstrideEncoder, SpaceBagAdaptiveDVstrideEncoder
from jax_walrus.decoder import AdaptiveDVstrideDecoder
from jax_walrus.processor import SpaceTimeSplitBlock


# ── Deterministic stride selection ──
# Mirrors walrus.models.shared_utils.flexi_utils.choose_kernel_size_deterministic

_PATCH_DICT = {
    0: (1, 1),
    1: (1, 1),
    4: (2, 2),
    8: (4, 2),
    12: (6, 2),
    16: (4, 4),
    24: (6, 4),
    32: (8, 4),
}


def choose_kernel_size_deterministic(
    x_shape: Tuple[int, ...],
) -> Tuple[Tuple[int, int], ...]:
    """Choose kernel size deterministically from spatial shape."""
    dims = len(x_shape)
    non_singleton = sum(int(s != 1) for s in x_shape)

    if dims == 1:
        per_axis = 512 // 16
        return (_PATCH_DICT[x_shape[0] // per_axis],)
    elif dims == 2:
        per_axis = 512 // 16
        H, W = x_shape
        return (_PATCH_DICT[H // per_axis], _PATCH_DICT[W // per_axis])
    elif dims == 3:
        per_axis = 256 // 16 if non_singleton >= 3 else 512 // 16
        H, W, D = x_shape
        h_p = H // per_axis if H != 1 else 0
        w_p = W // per_axis if W != 1 else 0
        d_p = D // per_axis if D != 1 else 0
        return (_PATCH_DICT[h_p], _PATCH_DICT[w_p], _PATCH_DICT[d_p])
    else:
        raise ValueError(f"Spatial dims must be 1-3, got {dims}")


# ── Patch jittering ──
# Mirrors walrus.models.shared_utils.patch_jitterers.FixedPatchJittererBoundaryPad

BC_PERIODIC = 2  # the_well BoundaryCondition.PERIODIC.value


def _compute_padding(
    shape: Tuple[int, ...],
    bcs: list,
    n_dims: int,
    max_d: int,
    base_kernel: Tuple[Tuple[int, int], ...],
    random_kernel: Tuple[Tuple[int, int], ...],
    jitter_patches: bool,
) -> Tuple[List[int], List[int], int, int]:
    """Compute constant and periodic padding amounts.

    Returns ``(constant_paddings, periodic_paddings, effective_ps, effective_stride)``
    in PyTorch reversed-dim ordering: ``[last_start, last_end, ..., first_start, first_end]``.
    """
    constant_paddings: List[int] = []
    periodic_paddings: List[int] = []
    effective_ps = 0
    effective_stride = 0

    for i in range(max_d):
        if i >= n_dims or shape[i] == 1:
            periodic_paddings = [0, 0] + periodic_paddings
            constant_paddings = [0, 0] + constant_paddings
            continue

        base_k1 = base_kernel[i][0]
        base_k2 = base_kernel[i][1]
        s1 = random_kernel[i][0]
        s2 = random_kernel[i][1]
        effective_ps = base_k1 + s1 * (base_k2 - 1)
        effective_stride = s1 * s2
        extra_padding = (effective_ps - effective_stride) // 2

        is_periodic = bcs[i][0] == BC_PERIODIC
        if is_periodic:
            jitter_pad = [0, 0]
        else:
            jitter_pad = (
                [effective_stride // 2, effective_stride // 2]
                if jitter_patches
                else [0, 0]
            )

        axis_pad = [p + extra_padding for p in jitter_pad]

        if is_periodic:
            periodic_paddings = axis_pad + periodic_paddings
            constant_paddings = [0, 0] + constant_paddings
        else:
            constant_paddings = axis_pad + constant_paddings
            periodic_paddings = [0, 0] + periodic_paddings

    return constant_paddings, periodic_paddings, effective_ps, effective_stride


def _pad_nd(x, paddings, mode="constant"):
    """Apply N-D padding.  ``paddings`` uses PyTorch reversed order."""
    if sum(abs(p) for p in paddings) == 0:
        return x
    n_spatial = len(paddings) // 2
    pad_per_axis = []
    for i in range(n_spatial):
        idx = len(paddings) - 2 * (i + 1)
        pad_per_axis.append((paddings[idx], paddings[idx + 1]))

    full_pad = [(0, 0)] * (x.ndim - n_spatial) + pad_per_axis

    if mode == "constant":
        return jnp.pad(x, full_pad, mode="constant", constant_values=0)
    elif mode == "circular":
        return jnp.pad(x, full_pad, mode="wrap")
    else:
        raise ValueError(f"Unsupported pad mode: {mode}")


def _slice_padding(x, paddings, n_leading_dims):
    """Remove padding by slicing.  ``paddings`` uses PyTorch reversed order."""
    if sum(abs(p) for p in paddings) == 0:
        return x
    n_spatial = len(paddings) // 2
    slices = [slice(None)] * n_leading_dims
    for i in range(n_spatial):
        idx = len(paddings) - 2 * (i + 1)
        ps, pe = paddings[idx], paddings[idx + 1]
        dim_size = x.shape[n_leading_dims + i]
        if ps + pe > 0:
            slices.append(slice(ps, dim_size - pe if pe > 0 else None))
        else:
            slices.append(slice(None))
    return x[tuple(slices)]


def _jitter_forward(
    x,
    bcs,
    n_dims,
    max_d,
    base_kernel,
    random_kernel,
    jitter_patches,
    rng_key,
):
    """Apply patch jittering: pad, create BC flags, optionally roll.

    Mirrors ``FixedPatchJittererBoundaryPad.forward``.

    Args:
        x: (T, B, C, H, W, D)
        bcs: boundary conditions, list of [bc_left, bc_right] per dim
        n_dims: number of real spatial dims
        rng_key: JAX PRNG key (None when jitter_patches=False)

    Returns:
        x_out: (T, B, C+3, H', W', D')  with BC flags appended
        jitter_info: dict
    """
    T = x.shape[0]
    shape = x.shape[3:]

    const_pad, periodic_pad, _, _ = _compute_padding(
        shape,
        bcs,
        n_dims,
        max_d,
        base_kernel,
        random_kernel,
        jitter_patches,
    )

    # Constant padding
    x_flat = rearrange(x, "t b c h w d -> (t b) c h w d")
    x_flat = _pad_nd(x_flat, const_pad, mode="constant")
    x = rearrange(x_flat, "(t b) c h w d -> t b c h w d", t=T)

    # BC flags: 3 channels (bias_correction, open, closed)
    bc_flag_shape = list(x.shape)
    bc_flag_shape[2] = 3
    bc_flags = jnp.zeros(bc_flag_shape, dtype=x.dtype)
    bc_flags = bc_flags.at[:, :, 0].set(1.0)

    dim_offset = 3  # T, B, C
    roll_quantities = []
    roll_dims = []

    for i in range(max_d):
        if i >= n_dims or shape[i] == 1:
            continue

        is_periodic = bcs[i][0] == BC_PERIODIC

        if not is_periodic:
            pad_idx = len(const_pad) - 2 * (i + 1)
            pad_start = const_pad[pad_idx]
            pad_end = const_pad[pad_idx + 1]

            if pad_start > 0:
                sl = [slice(None)] * len(bc_flags.shape)
                sl[i + dim_offset] = slice(None, pad_start)
                sl[2] = 1 + int(bcs[i][0])
                bc_flags = bc_flags.at[tuple(sl)].add(1.0)
                sl[2] = 0
                bc_flags = bc_flags.at[tuple(sl)].set(0.0)

            if pad_end > 0:
                sl = [slice(None)] * len(bc_flags.shape)
                sl[i + dim_offset] = slice(-pad_end, None)
                sl[2] = 1 + int(bcs[i][1])
                bc_flags = bc_flags.at[tuple(sl)].add(1.0)
                sl[2] = 0
                bc_flags = bc_flags.at[tuple(sl)].set(0.0)

        if jitter_patches and rng_key is not None:
            total_pad_idx = len(const_pad) - 2 * (i + 1)
            total_pad = const_pad[total_pad_idx + 1] + periodic_pad[total_pad_idx + 1]
            half_patch = total_pad if not is_periodic else x.shape[i + dim_offset] // 2

            rng_key, subkey = jax.random.split(rng_key)
            if half_patch > 1:
                roll_rate = int(
                    jax.random.randint(subkey, (), -(half_patch - 1), half_patch)
                )
            else:
                roll_rate = 0
            roll_quantities.append(roll_rate)
            roll_dims.append(i + dim_offset)

    x = jnp.concatenate([x, bc_flags], axis=2)

    if jitter_patches and len(roll_quantities) > 0:
        for rq, rd in zip(roll_quantities, roll_dims):
            x = jnp.roll(x, rq, axis=rd)

    # Periodic padding (FixedPatchJitterer: applied AFTER roll)
    if sum(periodic_pad) > 0:
        x_flat = rearrange(x, "t b c h w d -> (t b) c h w d")
        x_flat = _pad_nd(x_flat, periodic_pad, mode="circular")
        x = rearrange(x_flat, "(t b) c h w d -> t b c h w d", t=T)

    jitter_info = {
        "constant_paddings": const_pad,
        "periodic_paddings": periodic_pad,
        "rolls": (roll_quantities, roll_dims),
    }
    return x, jitter_info


def _unjitter(x, jitter_info, jitter_patches):
    """Inverse of _jitter_forward (FixedPatchJitterer ordering).

    Order: remove periodic padding -> unroll -> remove constant padding.
    """
    const_pad = jitter_info["constant_paddings"]
    periodic_pad = jitter_info["periodic_paddings"]
    roll_quantities, roll_dims = jitter_info["rolls"]

    # 1. Remove periodic padding
    x = _slice_padding(x, periodic_pad, n_leading_dims=x.ndim - len(periodic_pad) // 2)

    # 2. Reverse rolls
    if jitter_patches and len(roll_quantities) > 0:
        for rq, rd in zip(roll_quantities, roll_dims):
            x = jnp.roll(x, -rq, axis=rd)

    # 3. Remove constant padding
    x = _slice_padding(x, const_pad, n_leading_dims=x.ndim - len(const_pad) // 2)

    return x


class IsotropicModel(nn.Module):
    """
    Isotropic model: encoder -> N processor blocks -> decoder.

    Weight layout mirrors the PyTorch model exactly:
    - ``embed_2`` / ``embed_3``: encoder variants for 2D / 3D
    - ``debed_2`` / ``debed_3``: decoder variants for 2D / 3D
    - ``encoder_dummy``: scalar for gradient-checkpointing compatibility
    - ``blocks_0`` .. ``blocks_{N-1}``: processor blocks

    Training features (active when ``deterministic=False``):
    - **Drop path** (stochastic depth) linearly increasing across blocks
    - **Input field dropout** (dropout3d with Bonferroni correction)
    - **Patch jittering** with boundary-aware padding and random rolling
    - **Periodic rolling** of latent tokens between processor blocks
    - **Auto field_indices**: ``state_labels`` + ``[2, 0, 1]`` appended
    """

    hidden_dim: int = 768
    intermediate_dim: int = 192
    n_states: int = 4
    processor_blocks: int = 12
    groups: int = 16
    num_heads: int = 12
    mlp_dim: int = 0
    max_d: int = 3
    causal_in_time: bool = False
    drop_path: float = 0.05
    input_field_drop: float = 0.1
    bias_type: str = "rel"
    base_kernel_size: Tuple[Tuple[int, int], ...] = ((8, 4), (8, 4), (8, 4))
    use_spacebag: bool = True
    use_silu: bool = True
    include_d: Tuple[int, ...] = (2, 3)
    encoder_groups: int = 16
    jitter_patches: bool = True
    learned_pad: bool = True
    remat: bool = True

    def _make_encoder(self, name, field_indices, x_flat, stride1, stride2):
        """Instantiate and call the encoder."""
        if self.use_spacebag and field_indices is not None:
            return SpaceBagAdaptiveDVstrideEncoder(
                input_dim=self.n_states,
                inner_dim=self.intermediate_dim,
                output_dim=self.hidden_dim,
                base_kernel_size=self.base_kernel_size,
                groups=self.encoder_groups,
                spatial_dims=self.max_d,
                extra_dims=3,
                use_silu=self.use_silu,
                name=name,
            )(x_flat, field_indices, stride1, stride2)
        else:
            return AdaptiveDVstrideEncoder(
                input_dim=self.n_states,
                inner_dim=self.intermediate_dim,
                output_dim=self.hidden_dim,
                base_kernel_size=self.base_kernel_size,
                groups=self.encoder_groups,
                spatial_dims=self.max_d,
                use_silu=self.use_silu,
                name=name,
            )(x_flat, stride1, stride2)

    def _make_decoder(self, name, x_flat, state_labels, bcs_flat, stride1, stride2):
        """Instantiate and call the decoder."""
        return AdaptiveDVstrideDecoder(
            input_dim=self.hidden_dim,
            inner_dim=self.intermediate_dim,
            output_dim=self.n_states,
            base_kernel_size=self.base_kernel_size,
            groups=self.encoder_groups,
            spatial_dims=self.max_d,
            use_silu=self.use_silu,
            name=name,
        )(x_flat, state_labels, bcs_flat, stride1, stride2)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        state_labels: jnp.ndarray,
        bcs: list,
        stride1: Optional[Tuple[int, ...]] = None,
        stride2: Optional[Tuple[int, ...]] = None,
        field_indices: Optional[jnp.ndarray] = None,
        dim_key: Optional[int] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass.

        Accepts **channels-last** input with batch-first layout::

            x: (B, T, H, [W, [D]], C)

        Internally converts to the native ``(T, B, C, H, W, D)`` layout,
        runs the model, and converts the output back to channels-last::

            output: (B, T_out, H, [W, [D]], C_out)

        Args:
            x: Input fields in channels-last format ``(B, T, H, [W, [D]], C)``.
            state_labels: ``(C_out,)`` — output channel indices for the decoder.
            bcs: Boundary conditions per spatial dim, each ``[bc_left, bc_right]``.
                Use ``2`` for periodic, ``0`` for open.
            stride1: First encoder conv stride.  Auto-computed if ``None``.
            stride2: Second encoder conv stride.  Auto-computed if ``None``.
            field_indices: SpaceBag channel indices.  Auto-built if ``None``.
            dim_key: Encoder/decoder variant (2 or 3).  Auto-detected if ``None``.
            deterministic: ``True`` for inference, ``False`` for training.

        Returns:
            ``(B, T_out, ..., C_out)`` channels-last output.
        """
        # ── Convert (B, T, *spatial, C) → (T, B, C, *spatial) ──
        # x arrives as channels-last: (B, T, H, [W, [D]], C)
        n_spatial = x.ndim - 3  # total dims minus B, T, C
        # Move B,T to front and C from last to third position
        x = jnp.moveaxis(x, -1, 2)  # (B, T, C, *spatial)
        x = jnp.swapaxes(x, 0, 1)  # (T, B, C, *spatial)

        # ── Pad to max_d spatial dims ──
        squeeze_out = 0
        while x.ndim - 3 < self.max_d:
            x = jnp.expand_dims(x, axis=-1)
            squeeze_out += 1

        # Ensure state_labels is a proper array (callers may pass a tuple
        # to avoid the equinox "JAX array set as static" warning).
        state_labels = jnp.asarray(state_labels, dtype=jnp.int32)

        T, B, C = x.shape[:3]
        x_shape = x.shape[3:]

        # ── encoder_dummy ──
        encoder_dummy = self.param("encoder_dummy", nn.initializers.ones, (1,))

        # ── Determine variant ──
        if dim_key is None:
            dim_key = sum(int(s != 1) for s in x_shape)
        enc_name = f"embed_{dim_key}"
        dec_name = f"debed_{dim_key}"

        # ── Compute strides ──
        if stride1 is None or stride2 is None:
            dynamic_ks = choose_kernel_size_deterministic(x_shape)
            stride1 = stride1 or tuple(k[0] for k in dynamic_ks)
            stride2 = stride2 or tuple(k[1] for k in dynamic_ks)
            random_kernel = dynamic_ks
        else:
            random_kernel = tuple(zip(stride1, stride2))

        # ── Auto-build field_indices ──
        if self.use_spacebag and field_indices is None:
            field_indices = jnp.concatenate(
                [
                    state_labels,
                    jnp.array([2, 0, 1], dtype=state_labels.dtype),
                ]
            )

        # ── Input field dropout (training only) ──
        if not deterministic and self.input_field_drop > 0.0:
            x = rearrange(x, "t b c h w d -> b c (t h) w d")
            drop_rate = self.input_field_drop / x.shape[1]
            keep = 1.0 - drop_rate
            rng_field = self.make_rng("dropout")
            mask_shape = (x.shape[0], x.shape[1], 1, 1, 1)
            mask = jax.random.bernoulli(rng_field, keep, shape=mask_shape)
            x = x * mask.astype(x.dtype) / keep
            x = rearrange(x, "b c (t h) w d -> t b c h w d", t=T)

        # ── Encoder dummy multiply ──
        x = x * encoder_dummy

        # ── Patch jittering ──
        jitter_active = self.jitter_patches and not deterministic
        should_jitter = jitter_active or self.learned_pad
        rng_jitter = self.make_rng("jitter") if jitter_active else None

        if should_jitter:
            bcs_flat = bcs[0] if isinstance(bcs, tuple) else bcs
            x, jitter_info = _jitter_forward(
                x,
                bcs=bcs_flat,
                n_dims=dim_key,
                max_d=self.max_d,
                base_kernel=self.base_kernel_size,
                random_kernel=random_kernel,
                jitter_patches=jitter_active,
                rng_key=rng_jitter,
            )
        else:
            jitter_info = None

        # ── Encode ──
        x_flat = rearrange(x, "T B ... -> (T B) ...")
        x_enc = self._make_encoder(enc_name, field_indices, x_flat, stride1, stride2)
        x_enc = rearrange(x_enc, "(T B) ... -> T B ...", T=T)

        # ── Process ──
        # Compute drop-path rates as plain Python floats to avoid
        # ConcretizationTypeError when called inside a JIT context.
        dp = [i * self.drop_path / max(self.processor_blocks - 1, 1)
              for i in range(self.processor_blocks)]
        x_proc = x_enc

        bcs_flat = bcs[0] if isinstance(bcs, tuple) else bcs
        periodic_dims = []
        for dim_i in range(len(bcs_flat)):
            if bcs_flat[dim_i][0] == BC_PERIODIC:
                periodic_dims.append(dim_i + 3)
        periodic_dim_shapes = [x_proc.shape[d] for d in periodic_dims]
        roll_total = [0] * len(periodic_dims)

        for i in range(self.processor_blocks):
            # Random periodic rolling between blocks (training only)
            if not deterministic and len(periodic_dims) > 0 and self.jitter_patches:
                rng_roll = self.make_rng("jitter")
                for dim_idx in range(len(periodic_dims)):
                    rng_roll, subkey = jax.random.split(rng_roll)
                    rq = int(
                        jax.random.randint(
                            subkey,
                            (),
                            0,
                            periodic_dim_shapes[dim_idx],
                        )
                    )
                    roll_total[dim_idx] += rq
                    x_proc = jnp.roll(x_proc, rq, axis=periodic_dims[dim_idx])

            # Optionally wrap each processor block with gradient
            # checkpointing (nn.remat) to trade compute for memory.
            # static_argnums=(2,3,4) marks bcs, return_att, deterministic
            # as static (Flax offsets user indices by +1 for self).
            BlockCls = (
                nn.remat(SpaceTimeSplitBlock, static_argnums=(2, 3, 4))
                if self.remat
                else SpaceTimeSplitBlock
            )
            x_proc, _ = BlockCls(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                drop_path=dp[i],
                causal_in_time=self.causal_in_time,
                bias_type=self.bias_type,
                name=f"blocks_{i}",
            )(x_proc, bcs, False, deterministic)

        # Undo accumulated periodic rolling
        if not deterministic and sum(roll_total) > 0 and self.jitter_patches:
            for dim_idx, pdim in enumerate(periodic_dims):
                x_proc = jnp.roll(x_proc, -roll_total[dim_idx], axis=pdim)

        # Non-causal: decode only last time step
        if not self.causal_in_time:
            x_proc = x_proc[-1:]

        # ── Decode ──
        T_out = x_proc.shape[0]
        x_dec = rearrange(x_proc, "T B ... -> (T B) ...")
        x_dec = self._make_decoder(
            dec_name,
            x_dec,
            state_labels,
            bcs_flat,
            stride2,
            stride1,
        )
        x_dec = rearrange(x_dec, "(T B) ... -> T B ...", T=T_out)

        # ── Unjitter ──
        if should_jitter and jitter_info is not None:
            x_dec = _unjitter(x_dec, jitter_info, jitter_active)

        # Remove padded singleton spatial dims
        for _ in range(squeeze_out):
            x_dec = x_dec[..., 0]

        # ── Convert (T, B, C, *spatial) → (B, T, *spatial, C) ──
        x_dec = jnp.swapaxes(x_dec, 0, 1)  # (B, T, C, *spatial)
        x_dec = jnp.moveaxis(x_dec, 2, -1)  # (B, T, *spatial, C)

        return x_dec
