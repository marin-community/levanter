# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import haliax as hax
from haliax import Axis
import pytest

from levanter.layers.gated_deltanet import (
    GatedDeltaNet,
    GatedDeltaNetConfig,
    _causal_depthwise_conv1d_full,
    _causal_depthwise_conv1d_update,
)
from tests.test_utils import skip_if_no_torch

jax.config.update("jax_default_matmul_precision", "float32")


def _np(x):
    return np.array(x.detach().cpu().numpy())


def _init_small_hf_layer(hidden_size=128, nk=4, nv=8, dk=8, dv=8, ksz=4):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modular_qwen3_next import Qwen3NextGatedDeltaNet

    cfg = Qwen3NextConfig(
        hidden_size=hidden_size,
        linear_num_key_heads=nk,
        linear_num_value_heads=nv,
        linear_key_head_dim=dk,
        linear_value_head_dim=dv,
        linear_conv_kernel_dim=ksz,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
    )
    layer = Qwen3NextGatedDeltaNet(cfg, layer_idx=0)
    return cfg, layer


def _init_small_hf_layer_with_linear_only(hidden_size=128, nk=4, nv=8, dk=8, dv=8, ksz=4):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modular_qwen3_next import Qwen3NextGatedDeltaNet

    cfg = Qwen3NextConfig(
        hidden_size=hidden_size,
        linear_num_key_heads=nk,
        linear_num_value_heads=nv,
        linear_key_head_dim=dk,
        linear_value_head_dim=dv,
        linear_conv_kernel_dim=ksz,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
    )
    # Ensure the dynamic cache recognizes at least one linear_attention layer
    cfg.layer_types = ["linear_attention"]
    layer = Qwen3NextGatedDeltaNet(cfg, layer_idx=0)
    return cfg, layer


def _init_small_lev_layer(hidden_size=128, nk=4, nv=8, dk=8, dv=8, ksz=4, key=jax.random.PRNGKey(0)):
    Embed = Axis("embed", hidden_size)
    cfg = GatedDeltaNetConfig(
        Embed=Embed, num_k_heads=nk, num_v_heads=nv, head_k_dim=dk, head_v_dim=dv, conv_kernel_size=ksz
    )
    layer = GatedDeltaNet.init(cfg, key=key)
    return cfg, layer


def _assign_linear_weight(named_linear, np_weight, out_axis, in_axis):
    """
    Haliax Linear stores weight as NamedArray with axes (Out, In).
    """
    w_named = hax.named(np_weight, (out_axis.name, in_axis.name))
    return dataclasses.replace(named_linear, weight=w_named)


def _load_hf_weights_into_lev(lev_layer: GatedDeltaNet, hf_layer) -> GatedDeltaNet:
    """
    Load weights from HF layer into Levanter layer (avoid module-level transformers imports).
    """
    cfg = lev_layer.config
    # in_proj_qkvz
    w_qkvz = hf_layer.in_proj_qkvz.weight.detach().cpu().numpy()  # [proj, hidden]
    lev_layer = dataclasses.replace(
        lev_layer,
        in_proj_qkvz=_assign_linear_weight(lev_layer.in_proj_qkvz, w_qkvz, cfg.mix_qkvz_axis, cfg.Embed),
    )
    # in_proj_ba
    w_ba = hf_layer.in_proj_ba.weight.detach().cpu().numpy()
    lev_layer = dataclasses.replace(
        lev_layer,
        in_proj_ba=_assign_linear_weight(lev_layer.in_proj_ba, w_ba, cfg.ba_axis, cfg.Embed),
    )
    # conv weight (C,1,K) in HF -> (C,K) here
    conv_w = hf_layer.conv1d.weight.detach().cpu().numpy().squeeze(1)  # (C, K)
    lev_layer = dataclasses.replace(lev_layer, conv_weight=conv_w.astype(np.float32))
    # A_log, dt_bias
    lev_layer = dataclasses.replace(
        lev_layer,
        A_log=hf_layer.A_log.detach().cpu().numpy().astype(np.float32),
        dt_bias=hf_layer.dt_bias.detach().cpu().numpy().astype(np.float32),
    )
    # o_norm weight
    w_norm = hf_layer.norm.weight.detach().cpu().numpy()  # [v_head_dim]
    lev_layer = dataclasses.replace(
        lev_layer, o_norm=dataclasses.replace(lev_layer.o_norm, weight=hax.named(w_norm, (cfg.VHeadDim.name,)))
    )
    # out_proj
    w_out = hf_layer.out_proj.weight.detach().cpu().numpy().astype(np.float32)  # [hidden, value_dim]
    hidden = cfg.Embed.size
    value_dim = cfg.value_dim
    assert w_out.shape == (hidden, value_dim)
    w_out_3d = w_out.reshape(hidden, cfg.num_v_heads, cfg.head_v_dim)  # [embed, v_heads, v_head_dim]

    lev_layer = dataclasses.replace(
        lev_layer,
        out_proj=dataclasses.replace(
            lev_layer.out_proj,
            weight=hax.named(w_out_3d, (cfg.Embed.name, cfg.VHeads.name, cfg.VHeadDim.name)),
        ),
    )
    return lev_layer


# -------------------------------
# JAX-only layer-level tests
# -------------------------------


def test_layer_streaming_decode_matches_one_shot_prefill():
    """Streaming (per-token) with carried (conv_state, S_state) must match one-shot prefill."""
    key = jax.random.PRNGKey(0)
    B, L = 2, 20
    cfg = GatedDeltaNetConfig(
        Embed=Axis("embed", 32),
        num_k_heads=2,
        num_v_heads=4,  # ratio > 1 exercises Q/K repetition across V groups
        head_k_dim=8,
        head_v_dim=8,
        conv_kernel_size=4,
        rms_norm_eps=1e-6,
    )
    layer = GatedDeltaNet.init(cfg, key=key)

    Batch, Pos, Embed = Axis("batch", B), Axis("position", L), cfg.Embed
    xkey = jax.random.PRNGKey(0)
    x = hax.named(jax.random.normal(xkey, (B, L, Embed.size), dtype=jnp.float32), (Batch, Pos, Embed))

    # One-shot prefill (returns final state as well since inference=True)
    y_full, state_full = layer(x, inference=True, chunk_size=8, attention_mask=None, decode_state=None)
    assert state_full is not None

    # Streaming decode: step through one token at a time with carried state
    Channels = cfg.key_dim * 2 + cfg.value_dim
    K = cfg.conv_kernel_size
    conv_state = jnp.zeros((B, Channels, K), dtype=jnp.float32)
    S_state = jnp.zeros((B, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim), dtype=jnp.float32)

    ys = []
    for t in range(L):
        xt = x["position", hax.ds(t, Axis("pos1", 1))]
        y_t, (conv_state, S_state) = layer(xt, inference=True, chunk_size=8, decode_state=(conv_state, S_state))
        ys.append(y_t.array)

    y_stream = np.concatenate(ys, axis=1)  # (B, L, Embed)
    np.testing.assert_allclose(y_stream, y_full.array, rtol=1e-5, atol=1e-5)


def test_layer_masking_trailing_zeros_equivalence():
    """Zeroing trailing positions via attention_mask should not change earlier outputs (causality)."""
    key = jax.random.PRNGKey(0)
    B, L = 2, 19
    cfg = GatedDeltaNetConfig(
        Embed=Axis("embed", 48),
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=8,
        head_v_dim=8,
        conv_kernel_size=4,
        rms_norm_eps=1e-6,
    )
    layer = GatedDeltaNet.init(cfg, key=key)

    Batch, Pos, Embed = Axis("batch", B), Axis("position", L), cfg.Embed
    x = hax.named(jax.random.normal(key, (B, L, Embed.size), dtype=jnp.float32), (Batch, Pos, Embed))

    # Keep first L0 tokens
    L0 = 11
    mask = hax.named(
        jnp.concatenate([jnp.ones((B, L0)), jnp.zeros((B, L - L0))], axis=1).astype(jnp.float32),
        (Batch.name, Pos.name),
    )

    y_masked, _ = layer(x, inference=False, attention_mask=mask)
    y_trunc, _ = layer(x["position", hax.ds(0, Axis("pos1", L0))], inference=False)

    np.testing.assert_allclose(y_masked.array[:, :L0, :], y_trunc.array, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("csize_a,csize_b", [(8, 16), (16, 32)])
def test_layer_chunk_size_invariance(csize_a, csize_b):
    """Prefill outputs should be invariant to chunk size."""
    key = jax.random.PRNGKey(0)
    B, L = 2, 37
    cfg = GatedDeltaNetConfig(
        Embed=Axis("embed", 40),
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=8,
        head_v_dim=8,
        conv_kernel_size=4,
        rms_norm_eps=1e-6,
    )
    layer = GatedDeltaNet.init(cfg, key=key)

    Batch, Pos, Embed = Axis("batch", B), Axis("position", L), cfg.Embed
    x = hax.named(jax.random.normal(key, (B, L, Embed.size), dtype=jnp.float32), (Batch, Pos, Embed))

    y_a, _ = layer(x, inference=False, chunk_size=csize_a)
    y_b, _ = layer(x, inference=False, chunk_size=csize_b)
    np.testing.assert_allclose(y_a.array, y_b.array, rtol=1e-5, atol=1e-5)


def test_layer_gradients_exist():
    """End-to-end differentiability: grads w.r.t. inputs exist and are finite."""
    key = jax.random.PRNGKey(0)
    B, L = 1, 12
    cfg = GatedDeltaNetConfig(
        Embed=Axis("embed", 32),
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=8,
        head_v_dim=8,
        conv_kernel_size=4,
        rms_norm_eps=1e-6,
    )
    layer = GatedDeltaNet.init(cfg, key=key)

    Batch, Pos, Embed = Axis("batch", B), Axis("position", L), cfg.Embed
    x0 = hax.named(jax.random.normal(key, (B, L, Embed.size), dtype=jnp.float32), (Batch, Pos, Embed))

    def loss_fn(x_arr):
        x = hax.named(x_arr, x0.axes)
        y, _ = layer(x, inference=False, chunk_size=8)
        return jnp.sum(y.array)

    grads = jax.grad(loss_fn)(x0.array)
    assert jnp.all(jnp.isfinite(grads))


@skip_if_no_torch
def test_gdn_layer_matches_hf_prefill():
    import torch  # local import for environments without torch

    # helper imports happen inside helper functions

    def _to_torch(x):
        return torch.from_numpy(np.array(x))

    hidden_size, nk, nv, dk, dv, ksz = 128, 4, 8, 8, 8, 4
    hf_cfg, hf_layer = _init_small_hf_layer(hidden_size, nk, nv, dk, dv, ksz)
    lev_cfg, lev_layer = _init_small_lev_layer(hidden_size, nk, nv, dk, dv, ksz)

    # copy HF weights into Levanter layer
    lev_layer = _load_hf_weights_into_lev(lev_layer, hf_layer)

    # random input
    B, L = 2, 64
    x_j = jax.random.normal(jax.random.PRNGKey(0), (B, L, hidden_size), dtype=jnp.float32)
    x_named = hax.named(x_j, ("batch", "position", "embed"))

    # 1) Check b/a regrouping parity right after projections
    with torch.no_grad():
        x_t = _to_torch(x_j)
        qkvz = hf_layer.in_proj_qkvz(x_t)  # [B,L,qkvz]
        ba = hf_layer.in_proj_ba(x_t)  # [B,L,2*nv]
        q_t, k_t, v_t, z_t, b_t, a_t = hf_layer.fix_query_key_value_ordering(qkvz, ba)
        # bring to numpy
        q_hf, k_hf, v_hf = map(_np, (q_t, k_t, v_t))
        b_hf, a_hf = map(_np, (b_t, a_t))
        z_hf = _np(z_t)

    q_lev, k_lev, v_lev, z_lev, b_lev, a_lev = lev_layer._fix_qkvz_ordering(
        hax.named(qkvz.numpy(), ("batch", "position", "qkvz")),
        hax.named(ba.numpy(), ("batch", "position", "ba")),
    )

    np.testing.assert_allclose(q_lev.array, q_hf, atol=0, rtol=0)
    np.testing.assert_allclose(k_lev.array, k_hf, atol=0, rtol=0)
    np.testing.assert_allclose(v_lev.array, v_hf, atol=0, rtol=0)
    np.testing.assert_allclose(z_lev.array, z_hf, atol=0, rtol=0)
    np.testing.assert_allclose(b_lev.array, b_hf, atol=0, rtol=0)
    np.testing.assert_allclose(a_lev.array, a_hf, atol=0, rtol=0)

    # Levanter forward (prefill)
    y_lev, _ = lev_layer(x_named, inference=True, chunk_size=32)

    # HF forward (prefill)
    # Use torch CPU fallback path: by default `is_fast_path_available` will be False on CPU-only test envs.
    x_t = _to_torch(x_j)
    with torch.no_grad():
        y_hf = hf_layer(
            hidden_states=x_t,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
        )
        # hf_layer returns [B,L,H] torch tensor
        y_hf = y_hf.detach().cpu().numpy()

    np.testing.assert_allclose(np.array(y_lev.array), y_hf, rtol=1e-4, atol=1e-4)


@skip_if_no_torch
def test_gdn_layer_decode_matches_hf_one_step():
    """
    Prefill to build state, then decode one token using the recurrent path in both Levanter and HF.
    Ensures conv-state length K and S-state handoff are correct and parity holds.
    """
    import torch
    from transformers.models.qwen3_next.modular_qwen3_next import Qwen3NextDynamicCache

    hidden_size, nk, nv, dk, dv, ksz = 128, 4, 8, 8, 8, 4
    hf_cfg, hf_layer = _init_small_hf_layer_with_linear_only(hidden_size, nk, nv, dk, dv, ksz)
    lev_cfg = GatedDeltaNetConfig(
        Embed=Axis("embed", hidden_size),
        num_k_heads=nk,
        num_v_heads=nv,
        head_k_dim=dk,
        head_v_dim=dv,
        conv_kernel_size=ksz,
    )
    lev_layer = GatedDeltaNet.init(lev_cfg, key=jax.random.PRNGKey(0))

    # copy HF weights into Levanter layer
    lev_layer = _load_hf_weights_into_lev(lev_layer, hf_layer)

    B, L = 2, 37
    x_full = jax.random.normal(jax.random.PRNGKey(1), (B, L, hidden_size), dtype=jnp.float32)
    x_next = jax.random.normal(jax.random.PRNGKey(2), (B, 1, hidden_size), dtype=jnp.float32)

    # ---------- Levanter prefill (to get states) ----------
    x_named = hax.named(np.array(x_full), ("batch", "position", "embed"))
    y_lev_prefill, new_state = lev_layer(x_named, inference=True, chunk_size=32)
    assert new_state is not None, "expected state tuple when inference=True"
    conv_state, S_state = new_state

    # sanity: conv_state should have length K (NOT K-1)
    K = ksz
    C = lev_cfg.key_dim * 2 + lev_cfg.value_dim
    assert conv_state.shape == (B, C, K)

    # ---------- Levanter decode one token ----------
    x_next_named = hax.named(np.array(x_next), ("batch", "position", "embed"))
    y_lev_step, new_state2 = lev_layer(x_next_named, inference=True, decode_state=(conv_state, S_state))
    assert new_state2 is not None, "expected state tuple for decode step"
    conv_state2, S_state2 = new_state2

    # ---------- HF prefill with cache ----------
    cache = Qwen3NextDynamicCache(hf_cfg)
    with torch.no_grad():
        x_t = torch.from_numpy(np.array(x_full))
        # cache_position not strictly used in the layer prefill; pass something sensible
        _ = hf_layer(hidden_states=x_t, cache_params=cache, cache_position=torch.arange(L))
        # After prefill, cache must contain conv and recurrent states
        assert cache.conv_states[0] is not None and cache.recurrent_states[0] is not None

    # ---------- HF decode one token ----------
    with torch.no_grad():
        x_next_t = torch.from_numpy(np.array(x_next))
        y_hf_step = hf_layer(hidden_states=x_next_t, cache_params=cache, cache_position=torch.arange(L, L + 1))

    # ---------- Compare one-step decode outputs ----------
    y_lev_step_np = np.array(y_lev_step.array)
    y_hf_step_np = _np(y_hf_step)
    np.testing.assert_allclose(y_lev_step_np, y_hf_step_np, rtol=1e-4, atol=1e-4)


def test_depthwise_conv_update_equivalence():
    """
    Pure conv test: the incremental conv update should equal the full causal conv output, step by step,
    after warmup. Does not require HF.
    """
    key = jax.random.PRNGKey(0)
    B, C, L, K = 2, 48, 35, 7  # C = (2*key_dim + value_dim) in a typical layer
    w = jax.random.normal(key, (C, K), dtype=jnp.float32)
    bias = None

    # random "qkv" channels sequence
    x = jax.random.normal(key, (B, C, L), dtype=jnp.float32)

    # full conv once
    y_full = _causal_depthwise_conv1d_full(x, w, bias)  # (B,C,L)

    # incremental conv: start with zero state of shape (B,C,K)
    state = jnp.zeros((B, C, K), dtype=jnp.float32)
    outs = []
    for t in range(L):
        y_t, state = _causal_depthwise_conv1d_update(x[..., t : t + 1], w, bias, state)  # (B,C,1), (B,C,K)
        outs.append(y_t[..., 0])  # (B,C)

    y_update = jnp.stack(outs, axis=-1)  # (B,C,L)

    np.testing.assert_allclose(np.array(y_update), np.array(y_full), rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_ratio_equal_one_and_greater_than_one():
    """
    Exercise both ratio paths: nv == nk and nv > nk (repeat-interleave of Q/K).
    Run prefill parity vs HF in both cases.
    """
    import torch

    for nk, nv in [(4, 4), (4, 8)]:
        hidden_size, dk, dv, ksz = 96, 8, 8, 4
        hf_cfg, hf_layer = _init_small_hf_layer_with_linear_only(hidden_size, nk, nv, dk, dv, ksz)
        lev_cfg = GatedDeltaNetConfig(
            Embed=Axis("embed", hidden_size),
            num_k_heads=nk,
            num_v_heads=nv,
            head_k_dim=dk,
            head_v_dim=dv,
            conv_kernel_size=ksz,
        )
        lev_layer = GatedDeltaNet.init(lev_cfg, key=jax.random.PRNGKey(0))
        lev_layer = _load_hf_weights_into_lev(lev_layer, hf_layer)

        B, L = 2, 64
        x_j = jax.random.normal(jax.random.PRNGKey(0), (B, L, hidden_size), dtype=jnp.float32)
        x_named = hax.named(np.array(x_j), ("batch", "position", "embed"))

        # Levanter prefill
        y_lev, _ = lev_layer(x_named, inference=True, chunk_size=32)

        # HF prefill
        with torch.no_grad():
            x_t = torch.from_numpy(np.array(x_j))
            y_hf = hf_layer(hidden_states=x_t, cache_params=None, cache_position=None)

        np.testing.assert_allclose(np.array(y_lev.array), _np(y_hf), rtol=1e-4, atol=1e-4)


@skip_if_no_torch
def test_linear_mask_zeroes_padded_tokens_prefill():
    hidden_size, nk, nv, dk, dv, ksz = 96, 4, 8, 8, 8, 4
    hf_cfg, hf_layer = _init_small_hf_layer_with_linear_only(hidden_size, nk, nv, dk, dv, ksz)
    lev_cfg = GatedDeltaNetConfig(
        Embed=Axis("embed", hidden_size),
        num_k_heads=nk,
        num_v_heads=nv,
        head_k_dim=dk,
        head_v_dim=dv,
        conv_kernel_size=ksz,
    )
    lev_layer = GatedDeltaNet.init(lev_cfg, key=jax.random.PRNGKey(0))
    lev_layer = _load_hf_weights_into_lev(lev_layer, hf_layer)

    B, L_core, L_pad = 2, 16, 8
    x_core = jax.random.normal(jax.random.PRNGKey(0), (B, L_core, hidden_size), dtype=jnp.float32)
    x_full = jnp.concatenate(
        [jax.random.normal(jax.random.PRNGKey(1), (B, L_pad, hidden_size), dtype=jnp.float32), x_core], axis=1
    )

    # mask: left padding zeros, then ones
    mask = jnp.concatenate(
        [jnp.zeros((B, L_pad), dtype=jnp.float32), jnp.ones((B, L_core), dtype=jnp.float32)], axis=1
    )

    x_named = hax.named(np.array(x_full), ("batch", "position", "embed"))
    mask_named = hax.named(np.array(mask), ("batch", "position"))

    # Levanter with mask on full sequence
    y_full_masked, _ = lev_layer(x_named, inference=True, chunk_size=32, attention_mask=mask_named)

    # Levanter on just the unpadded core (no mask)
    x_core_named = hax.named(np.array(x_core), ("batch", "position", "embed"))
    y_core, _ = lev_layer(x_core_named, inference=True, chunk_size=32)

    np.testing.assert_allclose(
        np.array(y_full_masked.array)[:, L_pad:, :], np.array(y_core.array), rtol=1e-4, atol=1e-4
    )
