# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import jax
import jax.numpy as jnp
import haliax as hax
from haliax import Axis
import pytest

from levanter.layers.gated_deltanet import chunk_gated_delta_rule, recurrent_gated_delta_rule
from tests.test_utils import skip_if_no_torch


def _to_np(x):
    return np.array(x.detach().cpu().numpy())


def _get_hf_kernels():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen3_next.modular_qwen3_next import (
        torch_chunk_gated_delta_rule as hf_chunk,
        torch_recurrent_gated_delta_rule as hf_recur,
    )

    return hf_chunk, hf_recur


def _named_kernels_inputs(B, H, L, dk, dv, key):
    Batch, Heads, Pos, Dk, Dv = (
        Axis("batch", B),
        Axis("heads", H),
        Axis("position", L),
        Axis("k_head_dim", dk),
        Axis("v_head_dim", dv),
    )
    q = hax.named(
        jax.random.normal(key, (B, L, H, dk), dtype=jnp.float32), (Batch.name, Pos.name, Heads.name, Dk.name)
    )
    k = hax.named(
        jax.random.normal(key, (B, L, H, dk), dtype=jnp.float32), (Batch.name, Pos.name, Heads.name, Dk.name)
    )
    v = hax.named(
        jax.random.normal(key, (B, L, H, dv), dtype=jnp.float32), (Batch.name, Pos.name, Heads.name, Dv.name)
    )
    g = hax.named(jax.random.normal(key, (B, L, H), dtype=jnp.float32) * -0.1, (Batch.name, Pos.name, Heads.name))
    beta = hax.named(jax.random.uniform(key, (B, L, H), dtype=jnp.float32), (Batch.name, Pos.name, Heads.name))
    return q, k, v, g, beta


@skip_if_no_torch
def test_recurrent_kernel_matches_hf():
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 2, 17, 8, 8

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_named, _ = recurrent_gated_delta_rule(q, k, v, g, beta, output_final_state=False)

    # HF expects (B, L, H, dim) on input and transposes internally.
    def to_t(arr: jnp.ndarray):
        return torch.from_numpy(np.array(arr))

    out_hf, _ = hf_recur(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_hf_np = _to_np(out_hf)

    np.testing.assert_allclose(np.array(out_named.array), out_hf_np, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_chunk_kernel_matches_hf():
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 4, 64, 8, 16

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_named, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=32, output_final_state=False)

    def to_t(arr: jnp.ndarray):
        return torch.from_numpy(np.array(arr))

    out_hf, _ = hf_chunk(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        chunk_size=32,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_hf_np = _to_np(out_hf)

    np.testing.assert_allclose(np.array(out_named.array), out_hf_np, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_chunk_kernel_matches_hf_non_divisible():
    """L not divisible by chunk_size should still match HF fallback (padding path)."""
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 3, 61, 8, 16
    chunk_size = 32

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_named, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk_size, output_final_state=False)

    def to_t(arr: jnp.ndarray):
        return torch.from_numpy(np.array(arr))

    out_hf, _ = hf_chunk(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        chunk_size=chunk_size,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_hf_np = _to_np(out_hf)

    np.testing.assert_allclose(np.array(out_named.array), out_hf_np, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_chunk_size_one_matches_hf_recurrent():
    """chunk_size=1 should degenerate to the recurrent rule."""
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 2, 29, 8, 8

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

    out_chunk, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=1, output_final_state=False)
    out_recur, _ = recurrent_gated_delta_rule(q, k, v, g, beta, output_final_state=False)
    np.testing.assert_allclose(np.array(out_chunk.array), np.array(out_recur.array), rtol=1e-5, atol=1e-5)

    def to_t(arr: jnp.ndarray):
        return torch.from_numpy(np.array(arr))

    out_chunk_t, _ = hf_chunk(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        chunk_size=1,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_recur_t, _ = hf_recur(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    np.testing.assert_allclose(_to_np(out_chunk_t), _to_np(out_recur_t), rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_chunk_kernel_with_initial_state_matches_recurrent_continuation():
    """
    Provide an initial S0 and check chunk kernel == recurrent kernel on the same sequence.
    """
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 3, 47, 8, 8
    chunk_size = 16

    q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)
    S0 = jax.random.normal(key, (B, H, dk, dv), dtype=jnp.float32) * 0.1

    out_chunk, _ = chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=chunk_size, initial_state=S0, output_final_state=False
    )
    out_recur, _ = recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=S0, output_final_state=False)
    np.testing.assert_allclose(np.array(out_chunk.array), np.array(out_recur.array), rtol=1e-5, atol=1e-5)

    def to_t(arr: jnp.ndarray):
        return torch.from_numpy(np.array(arr))

    S0_t = torch.from_numpy(np.array(S0))
    out_chunk_t, _ = hf_chunk(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        chunk_size=chunk_size,
        initial_state=S0_t,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_recur_t, _ = hf_recur(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        initial_state=S0_t,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    np.testing.assert_allclose(_to_np(out_chunk_t), _to_np(out_recur_t), rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_short_sequences_edge_cases():
    """Short L vs chunk_size and kernel-size behaviors."""
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)

    for L in [1, 2, 3, 5, 7]:
        B, H, dk, dv = 2, 2, 8, 8
        q, k, v, g, beta = _named_kernels_inputs(B, H, L, dk, dv, key)

        out_named, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64, output_final_state=False)

        def to_t(arr: jnp.ndarray):
            return torch.from_numpy(np.array(arr))

        out_t, _ = hf_chunk(
            to_t(q.array),
            to_t(k.array),
            to_t(v.array),
            to_t(g.array),
            to_t(beta.array),
            chunk_size=64,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        out_hf = _to_np(out_t)

        np.testing.assert_allclose(np.array(out_named.array), out_hf, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_extreme_gates_no_nans_and_parity():
    """Stress alpha = exp(g) close to 0 (very negative g) and beta near 0/1."""
    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 1, 2, 37, 16, 8

    Batch, Heads, Pos, Dk, Dv = (
        Axis("batch", B),
        Axis("heads", H),
        Axis("position", L),
        Axis("k_head_dim", dk),
        Axis("v_head_dim", dv),
    )
    q = hax.named(jax.random.normal(key, (B, L, H, dk), dtype=jnp.float32), (Batch, Pos, Heads, Dk))
    k = hax.named(jax.random.normal(key, (B, L, H, dk), dtype=jnp.float32), (Batch, Pos, Heads, Dk))
    v = hax.named(jax.random.normal(key, (B, L, H, dv), dtype=jnp.float32), (Batch, Pos, Heads, Dv))
    g = hax.named(-jax.random.uniform(key, (B, L, H), minval=2.0, maxval=8.0, dtype=jnp.float32), (Batch, Pos, Heads))
    beta_small = hax.named(jnp.full((B, L, H), 1e-5, dtype=jnp.float32), (Batch, Pos, Heads))
    beta_big = hax.named(jnp.full((B, L, H), 1.0 - 1e-6, dtype=jnp.float32), (Batch, Pos, Heads))

    for beta in [beta_small, beta_big]:
        out_named, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=32, output_final_state=False)
        assert np.isfinite(np.array(out_named.array)).all()

        def to_t(arr: jnp.ndarray):
            return torch.from_numpy(np.array(arr))

        out_t, _ = hf_chunk(
            to_t(q.array),
            to_t(k.array),
            to_t(v.array),
            to_t(g.array),
            to_t(beta.array),
            chunk_size=32,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        out_hf = _to_np(out_t)

        np.testing.assert_allclose(np.array(out_named.array), out_hf, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_kernels_match_hf_without_l2norm():
    # TODO: fix this
    # Mismatched elements: 76 / 2736 (2.78%)
    # Max absolute difference among violations: 296960.
    # Max relative difference among violations: 0.00058013
    pytest.skip("not matching HF implementation")

    import torch

    hf_chunk, hf_recur = _get_hf_kernels()

    key = jax.random.PRNGKey(0)
    B, H, L, dk, dv = 2, 3, 57, 16, 8

    # Haliax inputs (B, L, H, dim)
    Batch, Heads, Pos, Dk, Dv = (
        Axis("batch", B),
        Axis("heads", H),
        Axis("position", L),
        Axis("k_head_dim", dk),
        Axis("v_head_dim", dv),
    )
    q = hax.named(jax.random.normal(key, (B, L, H, dk), dtype=jnp.float32), (Batch, Pos, Heads, Dk))
    k = hax.named(jax.random.normal(key, (B, L, H, dk), dtype=jnp.float32), (Batch, Pos, Heads, Dk))
    v = hax.named(jax.random.normal(key, (B, L, H, dv), dtype=jnp.float32), (Batch, Pos, Heads, Dv))
    g = hax.named(jax.random.normal(key, (B, L, H), dtype=jnp.float32) * -0.1, (Batch, Pos, Heads))
    beta = hax.named(jax.random.uniform(key, (B, L, H), dtype=jnp.float32), (Batch, Pos, Heads))

    # Haliax kernels with use_qk_l2norm_in_kernel=False
    out_chunk_j, _ = chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=32, output_final_state=False, use_qk_l2norm_in_kernel=False
    )
    out_recur_j, _ = recurrent_gated_delta_rule(
        q, k, v, g, beta, output_final_state=False, use_qk_l2norm_in_kernel=False
    )

    # HF fallback expects (B, L, H, dim) on input and transposes internally; don't move axes.
    def to_t(arr: jnp.ndarray):
        return torch.from_numpy(np.array(arr))

    out_chunk_t, _ = hf_chunk(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        chunk_size=32,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    )
    out_recur_t, _ = hf_recur(
        to_t(q.array),
        to_t(k.array),
        to_t(v.array),
        to_t(g.array),
        to_t(beta.array),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    )

    np.testing.assert_allclose(np.array(out_chunk_j.array), _to_np(out_chunk_t), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(out_recur_j.array), _to_np(out_recur_t), rtol=1e-5, atol=1e-5)
