# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import jax
import jax.numpy as jnp

from pytest import skip
from transformers.models.qwen3_next.modular_qwen3_next import (
    torch_chunk_gated_delta_rule as hf_chunk,
    torch_recurrent_gated_delta_rule as hf_recur,
)

from levanter.layers.gated_deltanet import chunk_gated_delta_rule_jax, recurrent_gated_delta_rule_jax
from tests.test_utils import skip_if_no_torch


def _to_np(x):
    return np.array(x.detach().cpu().numpy())


@skip_if_no_torch
def test_recurrent_kernel_matches_hf():
    import torch

    def _to_torch(x):
        return torch.from_numpy(np.array(x))

    key = jax.random.PRNGKey(1)
    B, H, L, dk, dv = 1, 2, 17, 8, 8

    q = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    k = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    v = jax.random.normal(key, (B, H, L, dv), dtype=jnp.float32)
    g = jax.random.normal(key, (B, H, L), dtype=jnp.float32) * -0.1
    beta = jax.random.uniform(key, (B, H, L), dtype=jnp.float32)

    out_j, _ = recurrent_gated_delta_rule_jax(q, k, v, g, beta, output_final_state=False)

    q_t = _to_torch(np.moveaxis(np.array(q), 1, 2))
    k_t = _to_torch(np.moveaxis(np.array(k), 1, 2))
    v_t = _to_torch(np.moveaxis(np.array(v), 1, 2))
    g_t = _to_torch(np.moveaxis(np.array(g), 1, 2))
    b_t = _to_torch(np.moveaxis(np.array(beta), 1, 2))

    out_t, _ = hf_recur(
        q_t, k_t, v_t, g_t, b_t, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True
    )
    out_hf = np.moveaxis(_to_np(out_t), 1, 2)

    np.testing.assert_allclose(np.array(out_j), out_hf, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_chunk_kernel_matches_hf():
    import torch

    def _to_torch(x):
        return torch.from_numpy(np.array(x))

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 4, 64, 8, 16

    q = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    k = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    v = jax.random.normal(key, (B, H, L, dv), dtype=jnp.float32)
    g = jax.random.normal(key, (B, H, L), dtype=jnp.float32) * -0.1  # slightly negative gate
    beta = jax.random.uniform(key, (B, H, L), dtype=jnp.float32)

    out_j, _ = chunk_gated_delta_rule_jax(q, k, v, g, beta, chunk_size=32, output_final_state=False)

    # HF expects tensors as (B,L,H,dim) on input and transposes internally.
    q_t = _to_torch(np.moveaxis(np.array(q), 1, 2))
    k_t = _to_torch(np.moveaxis(np.array(k), 1, 2))
    v_t = _to_torch(np.moveaxis(np.array(v), 1, 2))
    g_t = _to_torch(np.moveaxis(np.array(g), 1, 2))
    b_t = _to_torch(np.moveaxis(np.array(beta), 1, 2))

    out_t, _ = hf_chunk(
        q_t,
        k_t,
        v_t,
        g_t,
        b_t,
        chunk_size=32,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_hf = np.moveaxis(_to_np(out_t), 1, 2)  # back to (B,H,L,dv)

    np.testing.assert_allclose(np.array(out_j), out_hf, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_chunk_kernel_matches_hf_non_divisible():
    """L not divisible by chunk_size should still match HF fallback (padding path)."""
    import torch

    key = jax.random.PRNGKey(42)
    B, H, L, dk, dv = 2, 3, 61, 8, 16  # L not divisible by 32
    chunk_size = 32

    q = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    k = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    v = jax.random.normal(key, (B, H, L, dv), dtype=jnp.float32)
    g = jax.random.normal(key, (B, H, L), dtype=jnp.float32) * -0.1
    beta = jax.random.uniform(key, (B, H, L), dtype=jnp.float32)

    out_j, _ = chunk_gated_delta_rule_jax(q, k, v, g, beta, chunk_size=chunk_size, output_final_state=False)

    q_t = torch.from_numpy(np.moveaxis(np.array(q), 1, 2))
    k_t = torch.from_numpy(np.moveaxis(np.array(k), 1, 2))
    v_t = torch.from_numpy(np.moveaxis(np.array(v), 1, 2))
    g_t = torch.from_numpy(np.moveaxis(np.array(g), 1, 2))
    b_t = torch.from_numpy(np.moveaxis(np.array(beta), 1, 2))

    out_t, _ = hf_chunk(
        q_t,
        k_t,
        v_t,
        g_t,
        b_t,
        chunk_size=chunk_size,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_hf = np.moveaxis(_to_np(out_t), 1, 2)

    np.testing.assert_allclose(np.array(out_j), out_hf, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_chunk_size_one_matches_hf_recurrent():
    """chunk_size=1 should degenerate to the recurrent rule."""
    import torch

    key = jax.random.PRNGKey(7)
    B, H, L, dk, dv = 2, 2, 29, 8, 8

    q = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    k = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    v = jax.random.normal(key, (B, H, L, dv), dtype=jnp.float32)
    g = jax.random.normal(key, (B, H, L), dtype=jnp.float32) * -0.1
    beta = jax.random.uniform(key, (B, H, L), dtype=jnp.float32)

    # JAX
    out_chunk, _ = chunk_gated_delta_rule_jax(q, k, v, g, beta, chunk_size=1, output_final_state=False)
    out_recur, _ = recurrent_gated_delta_rule_jax(q, k, v, g, beta, output_final_state=False)
    np.testing.assert_allclose(np.array(out_chunk), np.array(out_recur), rtol=1e-5, atol=1e-5)

    # HF (reference)
    def to_t(x):
        return torch.from_numpy(np.moveaxis(np.array(x), 1, 2))

    out_chunk_t, _ = hf_chunk(
        to_t(q),
        to_t(k),
        to_t(v),
        to_t(g),
        to_t(beta),
        chunk_size=1,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_recur_t, _ = hf_recur(
        to_t(q),
        to_t(k),
        to_t(v),
        to_t(g),
        to_t(beta),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    np.testing.assert_allclose(_to_np(out_chunk_t), _to_np(out_recur_t), rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_chunk_kernel_with_initial_state_matches_recurrent_continuation():
    """
    Provide an initial S0 and check chunk kernel == recurrent kernel on the same sequence.
    This exercises the cross-chunk carry and the 'initial_state' plumbing.
    """
    import torch

    key = jax.random.PRNGKey(123)
    B, H, L, dk, dv = 1, 3, 47, 8, 8
    chunk_size = 16

    q = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    k = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    v = jax.random.normal(key, (B, H, L, dv), dtype=jnp.float32)
    g = jax.random.normal(key, (B, H, L), dtype=jnp.float32) * -0.2
    beta = jax.random.uniform(key, (B, H, L), dtype=jnp.float32)

    S0 = jax.random.normal(key, (B, H, dk, dv), dtype=jnp.float32) * 0.1

    out_chunk, _ = chunk_gated_delta_rule_jax(
        q, k, v, g, beta, chunk_size=chunk_size, initial_state=S0, output_final_state=False
    )
    out_recur, _ = recurrent_gated_delta_rule_jax(q, k, v, g, beta, initial_state=S0, output_final_state=False)
    np.testing.assert_allclose(np.array(out_chunk), np.array(out_recur), rtol=1e-5, atol=1e-5)

    # HF reference
    def to_t(x):
        return torch.from_numpy(np.moveaxis(np.array(x), 1, 2))

    S0_t = torch.from_numpy(np.array(S0))
    out_chunk_t, _ = hf_chunk(
        to_t(q),
        to_t(k),
        to_t(v),
        to_t(g),
        to_t(beta),
        chunk_size=chunk_size,
        initial_state=S0_t,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_recur_t, _ = hf_recur(
        to_t(q),
        to_t(k),
        to_t(v),
        to_t(g),
        to_t(beta),
        initial_state=S0_t,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    np.testing.assert_allclose(_to_np(out_chunk_t), _to_np(out_recur_t), rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_short_sequences_edge_cases():
    """Short L vs chunk_size and kernel-size behaviors."""
    import torch

    key = jax.random.PRNGKey(321)
    B, H, dk, dv = 2, 2, 8, 8

    for L in [1, 2, 3, 5, 7]:
        q = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
        k = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
        v = jax.random.normal(key, (B, H, L, dv), dtype=jnp.float32)
        g = jax.random.normal(key, (B, H, L), dtype=jnp.float32) * -0.3
        beta = jax.random.uniform(key, (B, H, L), dtype=jnp.float32)

        out_j, _ = chunk_gated_delta_rule_jax(q, k, v, g, beta, chunk_size=64, output_final_state=False)

        def to_t(x):
            return torch.from_numpy(np.moveaxis(np.array(x), 1, 2))

        out_t, _ = hf_chunk(
            to_t(q),
            to_t(k),
            to_t(v),
            to_t(g),
            to_t(beta),
            chunk_size=64,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        out_hf = np.moveaxis(_to_np(out_t), 1, 2)
        np.testing.assert_allclose(np.array(out_j), out_hf, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_extreme_gates_no_nans_and_parity():
    """Stress α≈exp(g) close to 0 (very negative g) and β near 0/1."""
    import torch

    key = jax.random.PRNGKey(999)
    B, H, L, dk, dv = 1, 2, 37, 16, 8

    q = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    k = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    v = jax.random.normal(key, (B, H, L, dv), dtype=jnp.float32)
    g = -jax.random.uniform(key, (B, H, L), minval=2.0, maxval=8.0, dtype=jnp.float32)  # strong decay
    beta_small = jnp.full((B, H, L), 1e-5, dtype=jnp.float32)
    beta_big = jnp.full((B, H, L), 1.0 - 1e-6, dtype=jnp.float32)

    for beta in [beta_small, beta_big]:
        out_j, _ = chunk_gated_delta_rule_jax(q, k, v, g, beta, chunk_size=32, output_final_state=False)
        assert np.isfinite(np.array(out_j)).all()

        def to_t(x):
            return torch.from_numpy(np.moveaxis(np.array(x), 1, 2))

        out_t, _ = hf_chunk(
            to_t(q),
            to_t(k),
            to_t(v),
            to_t(g),
            to_t(beta),
            chunk_size=32,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        out_hf = np.moveaxis(_to_np(out_t), 1, 2)
        np.testing.assert_allclose(np.array(out_j), out_hf, rtol=1e-5, atol=1e-5)


def test_kernels_match_hf_without_l2norm():
    # TODO: fix this
    skip("not matching HF implementation")

    import torch

    key = jax.random.PRNGKey(202)
    B, H, L, dk, dv = 2, 3, 57, 16, 8
    q = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    k = jax.random.normal(key, (B, H, L, dk), dtype=jnp.float32)
    v = jax.random.normal(key, (B, H, L, dv), dtype=jnp.float32)
    g = jax.random.normal(key, (B, H, L), dtype=jnp.float32) * -0.1
    beta = jax.random.uniform(key, (B, H, L), dtype=jnp.float32)

    out_chunk_j, _ = chunk_gated_delta_rule_jax(
        q, k, v, g, beta, chunk_size=32, output_final_state=False, use_qk_l2norm_in_kernel=False
    )
    out_recur_j, _ = recurrent_gated_delta_rule_jax(
        q, k, v, g, beta, output_final_state=False, use_qk_l2norm_in_kernel=False
    )

    def to_t(x):
        return torch.from_numpy(np.moveaxis(np.array(x), 1, 2))

    out_chunk_t, _ = hf_chunk(
        to_t(q),
        to_t(k),
        to_t(v),
        to_t(g),
        to_t(beta),
        chunk_size=32,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    )
    out_recur_t, _ = hf_recur(
        to_t(q),
        to_t(k),
        to_t(v),
        to_t(g),
        to_t(beta),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    )

    np.testing.assert_allclose(np.array(out_chunk_j), np.moveaxis(_to_np(out_chunk_t), 1, 2), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(out_recur_j), np.moveaxis(_to_np(out_recur_t), 1, 2), rtol=1e-5, atol=1e-5)
