"""Minimal forward-pass tests for jax_walrus (no checkpoint required)."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_platform_name", "cpu")

# SpaceBag encoder auto-builds field_indices = [*state_labels, 2, 0, 1].
# This references proj1_weight[:, idx], so needs idx < n_states.
# Minimum safe n_states = 3 (index 2 must exist).
# Input channels must be n_states + 3 (physical fields + 3 coord channels).
_N_STATES = 4
_C_INPUT = _N_STATES + 3  # 7
_H = _W = 32


@pytest.fixture(scope="module")
def tiny_model():
    from jax_walrus import IsotropicModel

    return IsotropicModel(
        hidden_dim=48,
        intermediate_dim=12,
        n_states=_N_STATES,
        processor_blocks=1,
        groups=4,
        num_heads=4,
        mlp_dim=48,
        max_d=3,  # must be 3 – encoder always expects 5-D (N,C,H,W,D) tensors;
        # 2-D data is zero-padded to match (D=1 added automatically)
        encoder_groups=4,
        remat=False,  # no gradient checkpointing in tests
        jitter_patches=False,
        learned_pad=False,
        input_field_drop=0.0,
        drop_path=0.0,
        include_d=(2, 3),  # build both 2-D and 3-D encoder/decoder variants
        base_kernel_size=((4, 2), (4, 2), (1, 1)),
    )


@pytest.fixture(scope="module")
def tiny_inputs():
    # x: (B, T, H, W, C) – channels-last.
    # Channels: n_states physical fields + 3 spatial-coord channels (dx, dy, dz).
    # The model auto-builds field_indices = [state_labels..., 2, 0, 1]
    # where [2, 0, 1] are coordinate-channel indices within the physical field space.
    x = jnp.ones((1, 2, _H, _W, _C_INPUT), dtype=jnp.float32)
    state_labels = jnp.arange(_N_STATES, dtype=jnp.int32)
    bcs = [[0, 0], [0, 0]]  # 2 spatial axes, open (non-periodic) boundaries
    return x, state_labels, bcs


@pytest.fixture(scope="module")
def tiny_params(tiny_model, tiny_inputs):
    x, state_labels, bcs = tiny_inputs
    rng = jax.random.PRNGKey(0)
    return tiny_model.init(
        {"params": rng},
        x,
        state_labels,
        bcs,
        deterministic=True,
    )


def test_import():
    from jax_walrus import IsotropicModel

    assert IsotropicModel is not None


def test_init(tiny_params):
    leaves = jax.tree_util.tree_leaves(tiny_params)
    assert len(leaves) > 0


def test_param_count(tiny_params):
    n = sum(v.size for v in jax.tree_util.tree_leaves(tiny_params))
    assert n > 0


def test_forward_shape(tiny_model, tiny_inputs, tiny_params):
    x, state_labels, bcs = tiny_inputs
    B = x.shape[0]
    y = tiny_model.apply(tiny_params, x, state_labels, bcs, deterministic=True)
    # Returns (B, T_out=1, H, W, C_out=n_states) – last timestep only
    assert y.ndim == 5
    assert y.shape[0] == B
    assert y.shape[-1] == _N_STATES


def test_forward_finite(tiny_model, tiny_inputs, tiny_params):
    x, state_labels, bcs = tiny_inputs
    y = tiny_model.apply(tiny_params, x, state_labels, bcs, deterministic=True)
    assert jnp.all(jnp.isfinite(y))
