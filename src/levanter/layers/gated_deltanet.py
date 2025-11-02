# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# based on:
# - https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py
# - the JAX implementation by Yu Sun and Leo Lee

"""
This module implements the Gated DeltaNet: https://arxiv.org/abs/2412.06464.
It exposes:
  - recurrent_gated_delta_rule: the sequential (decode) rule
  - chunk_gated_delta_rule:    the chunkwise-parallel (prefill / train) rule
  - GatedDeltaNet:             a full layer that wraps projections, a small
                               depthwise causal conv over [Q|K|V], the kernels,
                               and the gated RMSNorm + output projection.

Core update (rectangular state S ∈ R^{d_k × d_v}):
  S_t = α_t S_{t-1} + β_t (v_t - S_{t-1} k_t) k_t^T
  o_t = S_t^T q_t

where:
  α_t = exp(g_t) ∈ (0,1)   (forget/decay gate, log-parameterized by g_t ≤ 0)
  β_t = σ(b_t) ∈ (0,1)     (learning-rate gate)

Follows the GDN implementation for Qwen3-Next. Notably most math is performed in fp32.
"""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import functools
import os
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax._src.state.indexing import dslice

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray

_GDN_DBG = bool(int(os.environ.get("GDN_DEBUG_SHARDING", "0")))


def _dbg(tag: str, arr):
    if not _GDN_DBG:
        return
    try:
        jax.debug.inspect_array_sharding(arr, callback=lambda s: print(f"[GDN][internal] {tag}: {s}"))
    except Exception:
        pass


def _should_interpret_pallas() -> bool:
    try:
        platform = jax.devices()[0].platform
    except RuntimeError:
        platform = "cpu"
    return platform == "cpu"


# ---------- small utilities ----------


def _l2norm(x: NamedArray, axis: hax.AxisSelector, eps: float = 1e-6) -> NamedArray:
    """L2-normalize x along a named axis.

    Args:
        x: NamedArray of any shape.
        axis: the single axis to normalize along (e.g., the head dimension Dk).
    """
    x32 = x.astype(jnp.float32)
    inv = hax.rsqrt(hax.sum(hax.square(x32), axis=axis) + jnp.asarray(eps, dtype=jnp.float32))
    return (x32 * inv).astype(x.dtype)


def _rmsnorm_gated_reference(
    x_2d: jnp.ndarray,
    gate_2d: jnp.ndarray,
    weight: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    """Fallback RMSNorm + SiLU gate."""

    x32 = x_2d.astype(jnp.float32)
    gate32 = gate_2d.astype(jnp.float32)
    weight32 = weight.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + jnp.asarray(eps, dtype=jnp.float32))
    y32 = x32 * inv * weight32[None, :]
    gated32 = y32 * jax.nn.silu(gate32)
    return gated32.astype(x_2d.dtype)


def _fused_rmsnorm_gated_pallas(
    x_2d: jnp.ndarray,
    gate_2d: jnp.ndarray,
    weight: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    n_rows, hidden_size = x_2d.shape

    def kernel(x_ref, gate_ref, weight_ref, out_ref, *, eps):
        x = x_ref[0, :].astype(jnp.float32)
        gate = gate_ref[0, :].astype(jnp.float32)
        weight = weight_ref[:].astype(jnp.float32)
        eps32 = jnp.asarray(eps, dtype=jnp.float32)
        inv = jax.lax.rsqrt(jnp.mean(x * x) + eps32)
        y = x * inv * weight
        gated = y * jax.nn.silu(gate)
        out_ref[0, :] = gated.astype(out_ref.dtype)

    kernel_partial = functools.partial(kernel, eps=float(eps))

    out = pl.pallas_call(
        kernel_partial,
        out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
        grid=(n_rows,),
        in_specs=[
            pl.BlockSpec((1, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((1, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((hidden_size,), lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((1, hidden_size), lambda i: (i, 0)),
        interpret=_should_interpret_pallas(),
    )(x_2d, gate_2d, weight)
    return out


# ---------- depthwise conv: positional (lax) helpers with named wrappers ----------


def _causal_depthwise_conv1d_full(
    x_ncl: jnp.ndarray, w_ck: jnp.ndarray, bias_c: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Depthwise 1D convolution with *causal* semantics (left padding).

    Shapes:
      x_ncl: (N, C, L)  - batch, channels, length
      w_ck:  (C, K)     - per-channel (depthwise) filter of length K
      bias:  (C,)       - optional per-channel bias
      return: (N, C, L)

    DimensionNumbers ("NCH","OIH","NCH") means:
    - lhs (x):    N=0, C=1, H=2
    - rhs (w):    O=0, I=1, H=2  (we inject a singleton I=1 for depthwise)
    - out:        N=0, C=1, H=2
    """
    in_dtype = x_ncl.dtype
    N, C, L = x_ncl.shape
    K = w_ck.shape[-1]
    # pad x on the left with K-1 zeros so that output length == L ("causal")
    x_pad = jnp.pad(x_ncl, ((0, 0), (0, 0), (K - 1, 0)))

    # Upcast both sides to float32 for conv
    x32 = x_pad.astype(jnp.float32)
    w32 = w_ck.astype(jnp.float32)
    w_oik = w32[:, None, :]  # (C, 1, K)

    y32 = lax.conv_general_dilated(
        lhs=x32,
        rhs=w_oik,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=C,  # depthwise
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    _dbg("conv/full/y32", y32)

    if bias_c is not None:
        y32 = y32 + bias_c.astype(jnp.float32)[:, None]

    y32 = jax.nn.silu(y32)
    return y32.astype(in_dtype)


def _causal_depthwise_conv1d_update(
    x_ncl_1: jnp.ndarray,  # (N, C, 1)
    w_ck: jnp.ndarray,  # (C, K)
    bias_c: Optional[jnp.ndarray],
    prev_state_nck: jnp.ndarray,  # (N, C, K)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-step streaming update for the causal depthwise conv.

    Args:
      x_ncl_1: (N, C, 1) current step input
      w_ck:    (C, K)    depthwise kernel
      bias_c:  (C,)      optional bias
      prev_state_nck: (N, C, K) left context state (the last K inputs)

    Returns:
      y: (N, C, 1)   the latest convolved sample
      new_state: (N, C, K) with the newest x appended on the right

    Used during decode to avoid re-convolving the entire history.
    """
    in_dtype = x_ncl_1.dtype

    x_hist = jnp.concatenate([prev_state_nck, x_ncl_1], axis=-1)
    x32 = x_hist.astype(jnp.float32)
    w32 = w_ck.astype(jnp.float32)

    y32_all = lax.conv_general_dilated(
        lhs=x32,
        rhs=w32[:, None, :],
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=w32.shape[0],
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    y32 = y32_all[..., -1:]
    _dbg("conv/update/y32", y32)

    if bias_c is not None:
        y32 = y32 + bias_c.astype(jnp.float32)[:, None]

    y32 = jax.nn.silu(y32)
    new_state = jnp.concatenate([prev_state_nck[..., 1:], x_ncl_1], axis=-1)

    return y32.astype(in_dtype), new_state.astype(in_dtype)


# ---------- Fused Gated RMSNorm ----------


class FusedRMSNormGated(eqx.Module):
    """RMSNorm(x) * SiLU(gate) using an optional fused Pallas kernel."""

    axis: Axis
    weight: NamedArray  # [axis]
    eps: float = eqx.field(default=1e-6, static=True)
    use_flash: bool = eqx.field(default=True, static=True)

    @staticmethod
    def init(axis: Axis, eps: float = 1e-6, *, use_flash: bool = True) -> "FusedRMSNormGated":
        return FusedRMSNormGated(axis=axis, weight=hax.ones(axis), eps=eps, use_flash=use_flash)

    def __call__(self, x: NamedArray, gate: NamedArray) -> NamedArray:
        if x.resolve_axis(self.axis.name) != gate.resolve_axis(self.axis.name):
            raise ValueError("x and gate must share the normalization axis")

        # Move target axis to the end to make the flattened 2D view contiguous
        other_axes = tuple(ax for ax in x.axes if ax.name != self.axis.name)
        permuted_axes = other_axes + (self.axis,)
        x_perm = hax.rearrange(x, permuted_axes)
        gate_perm = hax.rearrange(gate, permuted_axes)

        x_arr = x_perm.array.reshape(-1, self.axis.size)
        gate_arr = gate_perm.array.reshape(-1, self.axis.size)
        weight_arr = self.weight.array

        if self.use_flash:
            try:
                out_arr = _fused_rmsnorm_gated_pallas(x_arr, gate_arr, weight_arr, self.eps)
            except Exception:
                if self.use_flash:
                    raise
                out_arr = _rmsnorm_gated_reference(x_arr, gate_arr, weight_arr, self.eps)
        else:
            out_arr = _rmsnorm_gated_reference(x_arr, gate_arr, weight_arr, self.eps)

        out_perm = out_arr.reshape(x_perm.array.shape)
        out_named = hax.named(out_perm, permuted_axes)
        return hax.rearrange(out_named, x.axes)


# ---------- Config ----------


@dataclass(frozen=True)
class GatedDeltaNetConfig:
    """Configuration for a GDN block (per layer).

    Head layout:
      - num_k_heads * head_k_dim = key_dim
      - num_v_heads * head_v_dim = value_dim
      - Keys/queries may have different head count/dim from values (rectangular S).

    Conv:
      - Small depthwise causal conv over concatenated channels [Q|K|V].
    """

    Embed: Axis
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int = 4
    rms_norm_eps: float = 1e-6

    @property
    def KHeads(self) -> Axis:
        return Axis("k_heads", self.num_k_heads)

    @property
    def VHeads(self) -> Axis:
        return Axis("v_heads", self.num_v_heads)

    @property
    def Heads(self) -> Axis:
        # expose VHeads as heads for tensor-parallel sharding
        return Axis("heads", self.num_v_heads)

    @property
    def KHeadDim(self) -> Axis:
        return Axis("k_head_dim", self.head_k_dim)

    @property
    def VHeadDim(self) -> Axis:
        return Axis("v_head_dim", self.head_v_dim)

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def mix_qkvz_axis(self) -> Axis:
        # [Q | K | V | Z]; the layer projects all at once
        return Axis("qkvz", self.key_dim * 2 + self.value_dim * 2)

    @property
    def ba_axis(self) -> Axis:
        # [b | a]; per value head: β = σ(b), g uses a via Mamba2-style discretization
        return Axis("ba", self.num_v_heads * 2)


# ---------- Triangular masks ----------


def _tri_upper_eq_mask(Ci: Axis, Cj: Axis) -> NamedArray:
    """Mask for i <= j (upper-triangular incl. diagonal) in (Ci, Cj) coordinates.

    Used to zero-out invalid contributions when building strictly lower-triangular
    in-chunk operators for the UT forward substitution.
    """
    ii = hax.arange(Ci)
    jj = hax.arange(Cj)
    I = ii.broadcast_axis(Cj)
    J = jj.broadcast_axis(Ci)
    return I <= J


def _diag_mask(Ci: Axis, Cj: Axis) -> NamedArray:
    ii = hax.arange(Ci)
    jj = hax.arange(Cj)
    I = ii.broadcast_axis(Cj)
    J = jj.broadcast_axis(Ci)
    return I == J


# ---------- Kernels ----------


def _recurrent_gated_delta_rule_reference(
    query: NamedArray,  # [batch, position, heads, k_head_dim]
    key: NamedArray,  # [batch, position, heads, k_head_dim]
    value: NamedArray,  # [batch, position, heads, v_head_dim]
    g: NamedArray,  # [batch, position, heads] (log-decay; α = exp(g))
    beta: NamedArray,  # [batch, position, heads] (β ∈ (0,1))
    *,
    initial_state: Optional[jnp.ndarray] = None,  # (B, H, dk, dv)
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> Tuple[NamedArray, Optional[jnp.ndarray]]:
    """Sequential (decode) GDN kernel

    For each t:
      α_t = exp(g_t)
      kv_t = S_{t-1}^T k_t             # shape: [B, H, d_v]
      δ_t  = β_t * (v_t - kv_t)        # [B, H, d_v]
      S_t  = α_t S_{t-1} + k_t δ_t^T   # [B, H, d_k, d_v]
      o_t  = S_t^T q_t                 # [B, H, d_v] (readout)

    Args:
      query, key, value: NamedArray tensors with explicit [batch, position, heads, dim]
      g:   log-decay; α = exp(g) is the forget gate in (0,1)
      beta: learning-rate gate β in (0,1)
      initial_state: optional S_0 (B, H, d_k, d_v)
      output_final_state: whether to return S_T
      use_qk_l2norm_in_kernel: if True, L2-normalize Q,K and scale Q by 1/sqrt(d_k)

    Returns:
      outputs: [batch, position, heads, v_head_dim]
      final_state (optional): (B, H, d_k, d_v)
    """
    # ---- axes ----
    Batch = query.resolve_axis("batch")
    Pos = query.resolve_axis("position")
    Heads = query.resolve_axis("heads")
    Dk = query.resolve_axis("k_head_dim")
    Dv = value.resolve_axis("v_head_dim")

    # ---- promote & normalize ----
    q = query.astype(jnp.float32)
    k = key.astype(jnp.float32)
    v = value.astype(jnp.float32)
    b = beta.astype(jnp.float32)
    gg = g.astype(jnp.float32)

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q, axis=Dk)
        k = _l2norm(k, axis=Dk)
    q = q * (Dk.size**-0.5)  # 1/sqrt(d_k) scaling

    # Prepare initial S
    B_, H_, L_, dk_, dv_ = Batch.size, Heads.size, Pos.size, Dk.size, Dv.size
    S0 = jnp.zeros((B_, H_, dk_, dv_), dtype=v.dtype) if initial_state is None else initial_state.astype(v.dtype)
    _dbg("recurrent/S0", S0)

    # Re-layout to positional major for lax.scan
    q_bhld = hax.rearrange(q, (Batch, Heads, Pos, Dk)).array  # (B,H,L,d_k)
    k_bhld = hax.rearrange(k, (Batch, Heads, Pos, Dk)).array
    v_bhld = hax.rearrange(v, (Batch, Heads, Pos, Dv)).array
    g_bhl = hax.rearrange(gg, (Batch, Heads, Pos)).array  # (B,H,L)
    b_bhl = hax.rearrange(b, (Batch, Heads, Pos)).array

    def step(S_prev_arr, xs_arr):
        # Unwrap per-step slices as NamedArrays for axis-safe math
        q_t_arr, k_t_arr, v_t_arr, g_t_arr, b_t_arr = xs_arr
        S_prev = hax.named(S_prev_arr, (Batch, Heads, Dk, Dv))
        q_t = hax.named(q_t_arr, (Batch, Heads, Dk))
        k_t = hax.named(k_t_arr, (Batch, Heads, Dk))
        v_t = hax.named(v_t_arr, (Batch, Heads, Dv))
        g_t = hax.named(g_t_arr, (Batch, Heads))
        b_t = hax.named(b_t_arr, (Batch, Heads))

        # Decay: S ← α_t S  (α_t = exp(g_t))
        decay = hax.exp(g_t).broadcast_axis(Dk).broadcast_axis(Dv)
        S_prev = S_prev * decay

        # Prediction kv_t = S^T k_t  (i.e., along Dk)
        kv = hax.dot(S_prev * k_t.broadcast_axis(Dv), axis=Dk)  # [B,H,Dv]

        # Rank-1 delta update and state write
        delta = (v_t - kv) * b_t.broadcast_axis(Dv)  # [B,H,Dv]
        S_new = S_prev + k_t.broadcast_axis(Dv) * delta.broadcast_axis(Dk)

        # Readout: o_t = S^T q_t
        y_t = hax.dot(S_new * q_t.broadcast_axis(Dv), axis=Dk)  # [B,H,Dv]
        return S_new.array, y_t.array

    S_final, out_seq = jax.lax.scan(
        step,
        S0,
        (
            jnp.moveaxis(q_bhld, 2, 0),  # time-major
            jnp.moveaxis(k_bhld, 2, 0),
            jnp.moveaxis(v_bhld, 2, 0),
            jnp.moveaxis(g_bhl, 2, 0),
            jnp.moveaxis(b_bhl, 2, 0),
        ),
        length=L_,
    )

    # Back to [B, Pos, H, Dv]
    out_bhlv = jnp.moveaxis(out_seq, 0, 2)  # (B,H,L,Dv)
    out_bhlv = hax.named(out_bhlv, (Batch, Heads, Pos, Dv))
    out_final = hax.rearrange(out_bhlv, (Batch, Pos, Heads, Dv))
    _dbg("recurrent/out", out_final.array)

    if output_final_state:
        return out_final, S_final
    else:
        return out_final, None


def _recurrent_gated_delta_rule_flash(
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    g: NamedArray,
    beta: NamedArray,
    *,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> Tuple[NamedArray, Optional[jnp.ndarray]]:
    Batch = query.resolve_axis("batch")
    Pos = query.resolve_axis("position")
    Heads = query.resolve_axis("heads")
    Dk = query.resolve_axis("k_head_dim")
    Dv = value.resolve_axis("v_head_dim")

    q = query.astype(jnp.float32)
    k = key.astype(jnp.float32)
    v = value.astype(jnp.float32)
    gg = g.astype(jnp.float32)
    b = beta.astype(jnp.float32)

    q_arr = hax.rearrange(q, (Batch, Heads, Pos, Dk)).array
    k_arr = hax.rearrange(k, (Batch, Heads, Pos, Dk)).array
    v_arr = hax.rearrange(v, (Batch, Heads, Pos, Dv)).array
    g_arr = hax.rearrange(gg, (Batch, Heads, Pos)).array

    beta_axis_names = tuple(ax.name for ax in beta.axes)
    if Dv.name in beta_axis_names:
        beta_arr = hax.rearrange(b, (Batch, Heads, Pos, Dv)).array
        is_beta_headwise = False
    else:
        beta_arr = hax.rearrange(b, (Batch, Heads, Pos)).array
        is_beta_headwise = True

    B_, H_, T_, K_ = q_arr.shape
    V_ = v_arr.shape[-1]
    NH = B_ * H_

    q_flat = q_arr.reshape(NH, T_, K_)
    k_flat = k_arr.reshape(NH, T_, K_)
    v_flat = v_arr.reshape(NH, T_, V_)
    g_flat = g_arr.reshape(NH, T_)
    if is_beta_headwise:
        beta_flat = beta_arr.reshape(NH, T_)
    else:
        beta_flat = beta_arr.reshape(NH, T_, V_)

    if initial_state is None:
        init_state = jnp.zeros((NH, K_, V_), dtype=jnp.float32)
    else:
        init_state = initial_state.astype(jnp.float32).reshape(NH, K_, V_)

    def kernel(
        q_ref,
        k_ref,
        v_ref,
        g_ref,
        beta_ref,
        init_ref,
        out_ref,
        final_ref,
        *,
        T,
        K,
        V,
        use_qk_l2norm,
        store_final_state,
        has_initial_state,
        is_beta_headwise,
        scale,
    ):
        head = pl.program_id(0)
        q_view = q_ref[dslice(head, 1), dslice(0, T), dslice(0, K)][0]
        k_view = k_ref[dslice(head, 1), dslice(0, T), dslice(0, K)][0]
        v_view = v_ref[dslice(head, 1), dslice(0, T), dslice(0, V)][0]
        g_view = g_ref[dslice(head, 1), dslice(0, T)][0]
        beta_view = (
            beta_ref[dslice(head, 1), dslice(0, T)][0]
            if is_beta_headwise
            else beta_ref[dslice(head, 1), dslice(0, T), dslice(0, V)][0]
        )
        if has_initial_state:
            state = init_ref[dslice(head, 1), dslice(0, K), dslice(0, V)][0].astype(jnp.float32)
        else:
            state = jnp.zeros((K, V), dtype=jnp.float32)

        scale32 = jnp.asarray(scale, dtype=jnp.float32)

        out_tile = jnp.zeros((T, V), dtype=out_ref.dtype)
        for t in range(T):
            q_t = q_view[t].astype(jnp.float32)
            k_t = k_view[t].astype(jnp.float32)
            if use_qk_l2norm:
                q_t = q_t / jnp.sqrt(jnp.sum(q_t * q_t) + 1e-6)
                k_t = k_t / jnp.sqrt(jnp.sum(k_t * k_t) + 1e-6)
            q_t = q_t * scale32
            v_t = v_view[t].astype(jnp.float32)
            g_t = g_view[t].astype(jnp.float32)
            state = state * jnp.exp(g_t)

            kv = jnp.sum(state * k_t[:, None], axis=0)
            if is_beta_headwise:
                beta_t = beta_view[t].astype(jnp.float32)
                delta = (v_t - kv) * beta_t
            else:
                beta_t = beta_view[t].astype(jnp.float32)
                delta = (v_t - kv) * beta_t
            state = state + k_t[:, None] * delta[None, :]

            out_tile = out_tile.at[t].set(jnp.sum(state * q_t[:, None], axis=0).astype(out_ref.dtype))

        out_ref[dslice(head, 1), dslice(0, T), dslice(0, V)] = out_tile[None, :, :]

        if store_final_state:
            final_ref[dslice(head, 1), dslice(0, K), dslice(0, V)] = state.astype(final_ref.dtype)[None, :, :]

    out_struct = jax.ShapeDtypeStruct((NH, T_, V_), v_flat.dtype)
    final_struct = jax.ShapeDtypeStruct((NH, K_, V_), jnp.float32)

    kernel_partial = functools.partial(
        kernel,
        T=T_,
        K=K_,
        V=V_,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        store_final_state=output_final_state,
        has_initial_state=initial_state is not None,
        is_beta_headwise=is_beta_headwise,
        scale=Dk.size**-0.5,
    )

    beta_spec: pl.BlockSpec
    if is_beta_headwise:
        beta_spec = pl.BlockSpec((1, T_), lambda bid_nh: (bid_nh, 0))
    else:
        beta_spec = pl.BlockSpec((1, T_, V_), lambda bid_nh: (bid_nh, 0, 0))

    result = pl.pallas_call(
        kernel_partial,
        out_shape=(out_struct, final_struct),
        grid=(NH,),
        in_specs=(
            pl.BlockSpec((1, T_, K_), lambda bid_nh: (bid_nh, 0, 0)),
            pl.BlockSpec((1, T_, K_), lambda bid_nh: (bid_nh, 0, 0)),
            pl.BlockSpec((1, T_, V_), lambda bid_nh: (bid_nh, 0, 0)),
            pl.BlockSpec((1, T_), lambda bid_nh: (bid_nh, 0)),
            beta_spec,
            pl.BlockSpec((1, K_, V_), lambda bid_nh: (bid_nh, 0, 0)),
        ),
        out_specs=(
            pl.BlockSpec((1, T_, V_), lambda bid_nh: (bid_nh, 0, 0)),
            pl.BlockSpec((1, K_, V_), lambda bid_nh: (bid_nh, 0, 0)),
        ),
        interpret=_should_interpret_pallas(),
    )(q_flat, k_flat, v_flat, g_flat, beta_flat, init_state)

    out_flat, final_flat = result
    out_arr = out_flat.reshape(B_, H_, T_, V_)
    out_named = hax.named(out_arr, (Batch, Heads, Pos, Dv))
    out_final = hax.rearrange(out_named, (Batch, Pos, Heads, Dv))

    if output_final_state:
        final_arr = final_flat.reshape(B_, H_, K_, V_)
        return out_final, final_arr
    else:
        return out_final, None


def recurrent_gated_delta_rule(
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    g: NamedArray,
    beta: NamedArray,
    *,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_flash: bool = True,
) -> Tuple[NamedArray, Optional[jnp.ndarray]]:
    if use_flash:
        try:
            return _recurrent_gated_delta_rule_flash(
                query,
                key,
                value,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        except Exception:
            if use_flash:
                raise
    return _recurrent_gated_delta_rule_reference(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


recurrent_gated_delta_rule.__doc__ = _recurrent_gated_delta_rule_reference.__doc__


def _prepare_chunk_inputs(
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    g: NamedArray,
    beta: NamedArray,
    *,
    chunk_size: int,
    use_qk_l2norm_in_kernel: bool,
):
    Batch = query.resolve_axis("batch")
    Pos = query.resolve_axis("position")
    Heads = query.resolve_axis("heads")
    Dk = query.resolve_axis("k_head_dim")
    Dv = value.resolve_axis("v_head_dim")

    q = query.astype(jnp.float32)
    k = key.astype(jnp.float32)
    v = value.astype(jnp.float32)
    gg = g.astype(jnp.float32)
    b = beta.astype(jnp.float32)

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q, axis=Dk)
        k = _l2norm(k, axis=Dk)
    q = q * (Dk.size**-0.5)

    L = Pos.size
    pad = (chunk_size - (L % chunk_size)) % chunk_size
    if pad > 0:
        q = hax.pad(q, {Pos: (0, pad)})
        k = hax.pad(k, {Pos: (0, pad)})
        v = hax.pad(v, {Pos: (0, pad)})
        b = hax.pad(b, {Pos: (0, pad)})
        gg = hax.pad(gg, {Pos: (0, pad)})

    PosPad = q.resolve_axis("position")
    Lt = PosPad.size
    Nc = Lt // chunk_size
    Chunks = Axis("chunks", Nc)
    C = Axis("chunk", chunk_size)

    def _chunk(x: NamedArray) -> NamedArray:
        return x.unflatten_axis(PosPad, (Chunks, C))

    q_c = _chunk(q)
    k_c = _chunk(k)
    v_c = _chunk(v)
    b_c = _chunk(b)
    g_c = _chunk(gg)

    return {
        "q_c": q_c,
        "k_c": k_c,
        "v_c": v_c,
        "b_c": b_c,
        "g_c": g_c,
        "Batch": Batch,
        "Pos": Pos,
        "PosPad": PosPad,
        "Heads": Heads,
        "Dk": Dk,
        "Dv": Dv,
        "Chunks": Chunks,
        "Chunk": C,
        "pad": pad,
    }


def _chunk_gated_delta_rule_reference(
    query: NamedArray,  # [batch, position, heads, k_head_dim]
    key: NamedArray,  # [batch, position, heads, k_head_dim]
    value: NamedArray,  # [batch, position, heads, v_head_dim]
    g: NamedArray,  # [batch, position, heads]  (log-decay; α=exp(g))
    beta: NamedArray,  # [batch, position, heads]  (β)
    *,
    chunk_size: int = 64,
    initial_state: Optional[jnp.ndarray] = None,  # (B,H,dk,dv)
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[NamedArray, Optional[jnp.ndarray]]:
    """Chunkwise-parallel GDN (DeltaNet UT/WY extended with decay).

    High-level sketch (per head):
      1) Split the length-L sequence into Nc = ceil(L/C) chunks of size C.
      2) Inside each chunk, form a strictly lower-triangular operator encoding
         the rank-1 updates and the *relative decays* between positions.
      3) Compute T = (I - A)^{-1} via *forward substitution*.
      4) Obtain "pseudo values" U = T (β V) and a decayed key summary K̂ = T (β K ⊙ exp(g)).
      5) Bridge chunks with the cross-chunk state S (decayed carry and innovation).
      6) Produce outputs by combining inter-chunk (from S) and intra-chunk terms.
    """

    # ---- axes ----
    prepared = _prepare_chunk_inputs(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=chunk_size,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    q_c = prepared["q_c"]
    k_c = prepared["k_c"]
    v_c = prepared["v_c"]
    b_c = prepared["b_c"]
    g_c = prepared["g_c"]
    Batch = prepared["Batch"]
    Pos = prepared["Pos"]
    PosPad = prepared["PosPad"]
    Heads = prepared["Heads"]
    Dk = prepared["Dk"]
    Dv = prepared["Dv"]
    Chunks = prepared["Chunks"]
    C = prepared["Chunk"]
    # pad = prepared["pad"]

    Nc = Chunks.size
    # Lt = PosPad.size
    L = Pos.size

    v_beta = v_c * b_c.broadcast_axis(Dv)  # βV per position
    k_beta = k_c * b_c.broadcast_axis(Dk)  # βK per position

    # cumulative g in chunk (for relative decays)
    g_cum = hax.cumsum(g_c, axis=C)  # [B, Nc, C, H]

    # --- Build strictly lower-triangular A in (Ci, Cj) coordinates ---
    Ci = Axis("Ci", C.size)
    Cj = Axis("Cj", C.size)

    kb_ci = k_beta.rename({C.name: Ci.name})  # [B,Nc,Ci,H,Dk]
    k_cj = k_c.rename({C.name: Cj.name})  # [B,Nc,Cj,H,Dk]

    # Raw interactions scaled by β: -(βK) @ K^T  (per head)
    A_raw = -hax.dot(kb_ci, k_cj, axis=Dk)  # [B,Nc,Ci,Cj,H]

    # Relative decay between positions i and j inside the chunk:
    #   exp( g_cum[i] - g_cum[j] )  for i >= j, else 0
    gi = g_cum.rename({C.name: Ci.name})
    gj = g_cum.rename({C.name: Cj.name})
    diff = gi.broadcast_axis(Cj) - gj.broadcast_axis(Ci)
    # Avoid overflow/NaNs in the strict upper triangle by setting exp argument to -inf
    neg_inf = jnp.asarray(-jnp.inf, dtype=diff.dtype)
    diff = hax.where(_diag_mask(Ci, Cj), jnp.asarray(0.0, dtype=diff.dtype), diff)
    diff = hax.where(_tri_upper_eq_mask(Ci, Cj), neg_inf, diff)
    decay = hax.exp(diff)  # [B,Nc,Ci,Cj,H]

    # Zero out diagonal and strict upper triangle
    A = A_raw * decay
    A = hax.where(_tri_upper_eq_mask(Ci, Cj), jnp.asarray(0.0, dtype=A.dtype), A)

    # --- Forward substitution (UT transform) to get T = (I - A)^{-1} ---
    A_bhcc = hax.rearrange(A, (Batch, Heads, Chunks, Ci, Cj)).array
    _dbg("chunk/A_bhcc", A_bhcc)

    eyeC = jnp.eye(C.size, dtype=A_bhcc.dtype)

    def body(i, attn):
        """Perform y[i] ← y[i] + sum_{j<i} y[i,j] * y[j,:]  (forward-subst)

        This loop computes the implicit lower-triangular transform so that
        'attn + I' acts like T above
        """
        row_i = lax.dynamic_slice_in_dim(attn, i, 1, axis=-2)  # (...,1,C)
        row_i = jnp.squeeze(row_i, axis=-2)  # (...,C)

        # Masks for the strict lower sub-block up to row i
        ar = jnp.arange(C.size, dtype=attn.dtype)
        m1 = (ar < i).astype(attn.dtype)  # vector mask
        m2 = ((ar[:, None] < i) & (ar[None, :] < i)).astype(attn.dtype)  # matrix mask

        row_pref = row_i * m1
        sub_pref = attn * m2
        incr = jnp.sum(row_pref[..., None] * sub_pref, axis=-2)
        new_row = jnp.expand_dims(row_i + incr, axis=-2)

        return lax.dynamic_update_slice_in_dim(attn, new_row, i, axis=-2)

    attn_low = lax.fori_loop(1, C.size, body, A_bhcc)
    T = attn_low + eyeC  # lower-triangular with ones on diagonal; acts like (I - A)^-1
    _dbg("chunk/T", T)

    # --- Pseudo values and decayed key summaries (intra-chunk) ---
    # v_pseudo = T @ (β V)
    vbeta_bhccd = hax.rearrange(v_beta.rename({C.name: Cj.name}), (Batch, Heads, Chunks, Cj, Dv)).array
    v_pseudo = jnp.einsum("bhnij,bhnjd->bhnid", T, vbeta_bhccd)  # (B,H,Nc,C,Dv)

    # k_cumdecay = T @ (β K ⊙ exp(g_cum))
    kbeta_bhccd = hax.rearrange(k_beta.rename({C.name: Cj.name}), (Batch, Heads, Chunks, Cj, Dk)).array
    exp_g_bhcc = hax.rearrange(hax.exp(g_cum).rename({C.name: Cj.name}), (Batch, Heads, Chunks, Cj)).array
    k_cumdecay = jnp.einsum("bhnij,bhnjd->bhnid", T, kbeta_bhccd * exp_g_bhcc[..., None])  # (B,H,Nc,C,d_k)
    _dbg("chunk/v_pseudo", v_pseudo)
    _dbg("chunk/k_cumdecay", k_cumdecay)

    # --- Scan over chunks: bridge with cross-chunk S ---
    q_bhccd = hax.rearrange(q_c, (Batch, Heads, Chunks, C, Dk)).array
    k_bhccd = hax.rearrange(k_c, (Batch, Heads, Chunks, C, Dk)).array
    g_bhcc = hax.rearrange(g_cum, (Batch, Heads, Chunks, C)).array

    B_, H_, dk_, dv_ = Batch.size, Heads.size, Dk.size, Dv.size
    v_dtype = v_c.dtype
    S = jnp.zeros((B_, H_, dk_, dv_), dtype=v_dtype) if initial_state is None else initial_state.astype(v_dtype)

    # Strict upper mask (i<j) to zero invalid future positions within a chunk
    mask_strict_upper = jnp.triu(jnp.ones((C.size, C.size), dtype=bool), k=1)

    def chunk_step(S_prev, inps):
        """Process one chunk i with in-chunk triangular ops + cross-chunk state S."""
        q_i, k_i, v_i, gcum_i, kcum_i = inps  # shapes: (B,H,C,dk/dv)
        # In-chunk relative decay mask for attention-like term with q
        diff = gcum_i[..., None] - gcum_i[..., None, :]  # (B,H,C,C)
        decay_i = jnp.exp(jnp.tril(diff))
        attn_i = jnp.einsum("bhid,bhjd->bhij", q_i, k_i) * decay_i
        attn_i = jnp.where(mask_strict_upper, 0.0, attn_i)  # strictly lower

        # Contribution predicted by previous cross-chunk state (remove it)
        v_prime = jnp.einsum("bhid,bhdm->bhim", kcum_i, S_prev)  # (B,H,C,dv)
        v_new = v_i - v_prime  # "innovation" within the chunk

        # Output: inter-chunk term (from decayed S) + in-chunk triangular mix
        qexp = q_i * jnp.exp(gcum_i)[..., None]
        inter = jnp.einsum("bhid,bhdm->bhim", qexp, S_prev)
        out_i = inter + jnp.einsum("bhij,bhjm->bhim", attn_i, v_new)

        # Update cross-chunk state S with the *tail* decay and innovations
        g_tail = gcum_i[..., -1]  # last position's cumulative g
        decay_tail = jnp.exp(g_tail)[..., None, None]  # α at the chunk tail
        decay_weights = jnp.exp((g_tail[..., None] - gcum_i))[..., None]  # exp(g_tail - g_pos)

        add = jnp.einsum("bhid,bhim->bhdm", k_i * decay_weights, v_new)
        S_new = S_prev * decay_tail + add
        return S_new, out_i

    S, out_chunks = jax.lax.scan(
        chunk_step,
        S,
        (
            jnp.moveaxis(q_bhccd, 2, 0),  # time-major over chunks
            jnp.moveaxis(k_bhccd, 2, 0),
            jnp.moveaxis(v_pseudo, 2, 0),
            jnp.moveaxis(g_bhcc, 2, 0),
            jnp.moveaxis(k_cumdecay, 2, 0),
        ),
        length=Nc,
    )

    # Back to [B, Pos, H, Dv], trimming padding if any
    out_bhcd = jnp.moveaxis(out_chunks, 0, 2)  # (B,H,Nc,C,Dv)
    out_bhcd = hax.named(out_bhcd, (Batch, Heads, Chunks, C, Dv))
    out_flat_bhPd = out_bhcd.flatten_axes((Chunks, C), PosPad)
    out_bhLd = out_flat_bhPd["position", hax.ds(0, L)]
    out_final = hax.rearrange(out_bhLd, (Batch, PosPad.name, Heads, Dv))
    _dbg("chunk/out", out_final.array)

    return (out_final, S) if output_final_state else (out_final, None)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


@dataclass(frozen=True)
class _Tiles:
    BK: int = 64
    BV: int = 64


def _pick_tiles(dk: int, dv: int) -> _Tiles:
    bk = 64 if dk >= 128 else 32
    bv = 64 if dv >= 128 else 32
    return _Tiles(BK=bk, BV=bv)


def _pad_L_axis(x: jnp.ndarray, width: int) -> jnp.ndarray:
    if width <= 0:
        return x
    if x.ndim == 2:  # [NH, L]
        return jnp.pad(x, ((0, 0), (0, width)))
    elif x.ndim == 3:  # [NH, L, D]
        return jnp.pad(x, ((0, 0), (0, width), (0, 0)))
    elif x.ndim == 4:  # [B, H, L, D]
        return jnp.pad(x, ((0, 0), (0, 0), (0, width), (0, 0)))
    else:
        raise ValueError(f"Unexpected rank for length-padding: {x.ndim}")


def _gdn_chunk_fwd_kernel(
    q_ref,  # [NH, T_pad, K]
    k_ref,  # [NH, T_pad, K]
    v_ref,  # [NH, T_pad, V]
    g_ref,  # [NH, T_pad]
    beta_ref,  # [NH, T_pad]
    init_ref,  # [NH, K, V]
    lengths_ref,  # [NH]  (real lengths, but we rely on zero padding)
    out_ref,  # [NH, T_pad, V]
    final_ref,  # [NH, K, V]
    *,
    T_pad: int,
    K: int,
    V: int,
    chunk_len: int,
    head_first: bool,
    store_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    BK: int,
    BV: int,
):
    nh = pl.program_id(axis=0)

    # Per-(seq,head) views (depend on refs → no capture)
    q_view = q_ref[dslice(nh, 1), dslice(0, T_pad), dslice(0, K)][0]
    k_view = k_ref[dslice(nh, 1), dslice(0, T_pad), dslice(0, K)][0]
    v_view = v_ref[dslice(nh, 1), dslice(0, T_pad), dslice(0, V)][0]
    g_view = g_ref[dslice(nh, 1), dslice(0, T_pad)][0]
    b_view = beta_ref[dslice(nh, 1), dslice(0, T_pad)][0]

    # FP32 state S
    S = init_ref[dslice(nh, 1), dslice(0, K), dslice(0, V)][0].astype(jnp.float32)

    # “zero” scratch derived from inputs (no captured zero)
    out_tile = v_view * (g_view[:, None] - g_view[:, None])  # [T_pad, V], all zeros

    # 1/sqrt(K) passed as kwarg would be nice, but this constant is tiny and stable
    # Make it an input-like traced scalar: multiply sum-by-sum to get 1 then divide by sqrt(K)
    # However, to keep things simple and JAX-friendly, do this:
    scale = jnp.asarray(1.0 / jnp.sqrt(K), dtype=jnp.float32)

    nchunks_max = T_pad // chunk_len

    def chunk_body(c_idx, carry):
        S_carry, out_tile_carry = carry
        # dynamic indices only: derive zero from c_start
        c_start = c_idx * chunk_len
        zero_col_dyn = c_start - c_start

        # Fixed-size chunk slices
        q_c = lax.dynamic_slice(q_view, (c_start, zero_col_dyn), (chunk_len, K)).astype(jnp.float32)
        k_c = lax.dynamic_slice(k_view, (c_start, zero_col_dyn), (chunk_len, K)).astype(jnp.float32)
        v_c = lax.dynamic_slice(v_view, (c_start, zero_col_dyn), (chunk_len, V)).astype(jnp.float32)
        g_c = lax.dynamic_slice(g_view, (c_start,), (chunk_len,)).astype(jnp.float32)
        b_c = lax.dynamic_slice(b_view, (c_start,), (chunk_len,)).astype(jnp.float32)

        # Optional L2-norm + scale
        if use_qk_l2norm_in_kernel:
            qn = jnp.sqrt(jnp.sum(q_c * q_c, axis=1, keepdims=True) + 1e-6)
            kn = jnp.sqrt(jnp.sum(k_c * k_c, axis=1, keepdims=True) + 1e-6)
            q_c = q_c / jnp.where(qn > 0, qn, 1.0)
            k_c = k_c / jnp.where(kn > 0, kn, 1.0)
        q_c = q_c * scale

        # Chunkwise cumsum/exp
        g_cum = jnp.cumsum(g_c, axis=0)  # (C,)
        eg_cum = jnp.exp(g_cum)  # (C,)

        v_beta = v_c * b_c[:, None]  # (C,V)
        k_beta = k_c * b_c[:, None]  # (C,K)

        # UT buffers as "zeros" from inputs
        yv = v_c * (g_c[:, None] - g_c[:, None])  # (C,V)
        yk = k_c * (g_c[:, None] - g_c[:, None])  # (C,K)

        def ut_row_update(i, carry_ut):
            yv_cur, yk_cur = carry_ut

            # accumulate over j < i
            def acc_j(j, acc):
                acc_v, acc_k = acc
                dot_ij = jnp.dot(k_beta[i], k_c[j])
                coeff = -dot_ij * jnp.exp(g_cum[i] - g_cum[j])
                acc_v = acc_v + coeff * yv_cur[j]
                acc_k = acc_k + coeff * yk_cur[j]
                return (acc_v, acc_k)

            zero_v = v_c[0] * (g_c[0] - g_c[0])
            zero_k = k_c[0] * (g_c[0] - g_c[0])
            acc_v, acc_k = lax.fori_loop(0, i, acc_j, (zero_v, zero_k))
            yv_cur = yv_cur.at[i].set(v_beta[i] + acc_v)
            yk_cur = yk_cur.at[i].set(k_beta[i] * eg_cum[i] + acc_k)
            return (yv_cur, yk_cur)

        # device loop over rows
        yv, yk = lax.fori_loop(0, chunk_len, ut_row_update, (yv, yk))

        # ---- bridge with cross-chunk S and compute outputs ----
        qexp = q_c * eg_cum[:, None]
        v_prime = yk @ S_carry
        inter = qexp @ S_carry

        # lower-triangular (incl. diagonal) attention-like term (vectorized)
        attn_raw = (q_c @ k_c.T) * jnp.exp(g_cum[:, None] - g_cum[None, :])
        idx_i = lax.broadcasted_iota(jnp.int32, (chunk_len, chunk_len), 0)
        idx_j = lax.broadcasted_iota(jnp.int32, (chunk_len, chunk_len), 1)
        zero_scalar = jnp.sum(g_c[:1] - g_c[:1])
        attn = jnp.where(idx_i >= idx_j, attn_raw, zero_scalar)
        v_new = yv - v_prime
        out_chunk = inter + attn @ v_new

        # write chunk result
        out_tile_next = lax.dynamic_update_slice(
            out_tile_carry, out_chunk.astype(out_ref.dtype), (c_start, zero_col_dyn)
        )

        # ---- Update S ----
        g_tail = jnp.sum(g_c, axis=0)
        decay_tail = jnp.exp(g_tail)
        S_carry = S_carry * decay_tail
        dw_full = jnp.exp(g_tail - g_cum)
        k_scaled = k_c * dw_full[:, None]
        S_carry = S_carry + k_scaled.T @ v_new

        return (S_carry, out_tile_next)

    S, out_tile = lax.fori_loop(0, nchunks_max, chunk_body, (S, out_tile))

    # final writes (add singleton to match NDIndexer slice)
    out_ref[dslice(nh, 1), dslice(0, T_pad), dslice(0, V)] = out_tile[None, :, :]
    if store_final_state:
        final_ref[dslice(nh, 1), dslice(0, K), dslice(0, V)] = S.astype(final_ref.dtype)[None, :, :]


def _chunk_gated_delta_rule_flash_tiled(
    query,
    key,
    value,
    g,
    beta,
    *,
    chunk_size: int = 64,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    head_first: bool = False,
    offsets: Optional[jnp.ndarray] = None,
):
    if not head_first:
        Batch = query.resolve_axis("batch")
        Pos = query.resolve_axis("position")
        Heads = query.resolve_axis("heads")
        Dk = query.resolve_axis("k_head_dim")
        Dv = value.resolve_axis("v_head_dim")
        B_, L_, H_, K_ = Batch.size, Pos.size, Heads.size, Dk.size
        V_ = Dv.size
        NH = B_ * H_

        q_bhlk = hax.rearrange(query, (Batch, Heads, Pos, Dk)).array
        k_bhlk = hax.rearrange(key, (Batch, Heads, Pos, Dk)).array
        v_bhlv = hax.rearrange(value, (Batch, Heads, Pos, Dv)).array
        g_bhl = hax.rearrange(g, (Batch, Heads, Pos)).array
        b_bhl = hax.rearrange(beta, (Batch, Heads, Pos)).array
    else:
        Batch = query.resolve_axis("batch")
        Heads = query.resolve_axis("heads")
        Pos = query.resolve_axis("position")
        Dk = query.resolve_axis("k_head_dim")
        Dv = value.resolve_axis("v_head_dim")
        B_, H_, L_, K_ = Batch.size, Heads.size, Pos.size, Dk.size
        V_ = Dv.size
        NH = B_ * H_
        q_bhlk = query.array
        k_bhlk = key.array
        v_bhlv = value.array
        g_bhl = g.array
        b_bhl = beta.array

    q_flat = q_bhlk.reshape(NH, L_, K_)
    k_flat = k_bhlk.reshape(NH, L_, K_)
    v_flat = v_bhlv.reshape(NH, L_, V_)
    g_flat = g_bhl.reshape(NH, L_)
    b_flat = b_bhl.reshape(NH, L_)

    lengths = (
        jnp.full((NH,), L_, dtype=jnp.int32)
        if offsets is None
        else (offsets.astype(jnp.int32)[1:] - offsets.astype(jnp.int32)[:-1])
    )

    T_pad = int(_ceil_div(L_, chunk_size) * chunk_size)
    pad_amt = T_pad - L_
    q_flat = _pad_L_axis(q_flat, pad_amt)
    k_flat = _pad_L_axis(k_flat, pad_amt)
    v_flat = _pad_L_axis(v_flat, pad_amt)
    g_flat = _pad_L_axis(g_flat, pad_amt)
    b_flat = _pad_L_axis(b_flat, pad_amt)

    init_flat = (
        jnp.zeros((NH, K_, V_), dtype=jnp.float32)
        if initial_state is None
        else initial_state.reshape(NH, K_, V_).astype(jnp.float32)
    )

    tiles = _pick_tiles(K_, V_)
    BK, BV = int(tiles.BK), int(tiles.BV)

    out_struct = jax.ShapeDtypeStruct((NH, T_pad, V_), value.dtype)
    final_struct = jax.ShapeDtypeStruct((NH, K_, V_), jnp.float32)

    kernel_partial = functools.partial(
        _gdn_chunk_fwd_kernel,
        T_pad=T_pad,
        K=K_,
        V=V_,
        chunk_len=int(chunk_size),
        head_first=head_first,
        store_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        BK=BK,
        BV=BV,
    )

    out_flat, final_flat = pl.pallas_call(
        kernel_partial,
        out_shape=(out_struct, final_struct),
        grid=(NH,),
        in_specs=(
            pl.BlockSpec((1, T_pad, K_), lambda bid: (bid, 0, 0)),
            pl.BlockSpec((1, T_pad, K_), lambda bid: (bid, 0, 0)),
            pl.BlockSpec((1, T_pad, V_), lambda bid: (bid, 0, 0)),
            pl.BlockSpec((1, T_pad), lambda bid: (bid, 0)),
            pl.BlockSpec((1, T_pad), lambda bid: (bid, 0)),
            pl.BlockSpec((1, K_, V_), lambda bid: (bid, 0, 0)),
            pl.BlockSpec((1,), lambda bid: (bid,)),  # lengths
        ),
        out_specs=(
            pl.BlockSpec((1, T_pad, V_), lambda bid: (bid, 0, 0)),
            pl.BlockSpec((1, K_, V_), lambda bid: (bid, 0, 0)),
        ),
        interpret=_should_interpret_pallas(),
    )(q_flat, k_flat, v_flat, g_flat, b_flat, init_flat, lengths)

    out_trim = out_flat[:, :L_, :]
    if not head_first:
        out_bhLv = out_trim.reshape(B_, H_, L_, V_)
        out_named = hax.named(out_bhLv, (Batch, Heads, Pos, Dv))
        out_final = hax.rearrange(out_named, (Batch, Pos, Heads, Dv))
        final = final_flat.reshape(B_, H_, K_, V_) if output_final_state else None
        return out_final, final
    else:
        out_bHLv = out_trim.reshape(B_, H_, L_, V_)
        out_named = hax.named(out_bHLv, (Batch, Heads, Pos, Dv))
        final = final_flat.reshape(B_, H_, K_, V_) if output_final_state else None
        return out_named, final


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10, 11))
def _chunk_gdn_flash_array(
    q_arr: jnp.ndarray,  # [B,L,H,K] or [B,H,L,K]
    k_arr: jnp.ndarray,
    v_arr: jnp.ndarray,
    g_arr: jnp.ndarray,  # [B,L,H] or [B,H,L]
    beta_arr: jnp.ndarray,  # [B,L,H] or [B,H,L]
    initial_state: Optional[jnp.ndarray],  # [B,H,K,V]
    chunk_size: int,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    head_first: bool,
    use_varlen: bool,
    offsets: Optional[jnp.ndarray],
):
    if not head_first:
        Batch = Axis("batch", q_arr.shape[0])
        Pos = Axis("position", q_arr.shape[1])
        Heads = Axis("heads", q_arr.shape[2])
        Dk = Axis("k_head_dim", q_arr.shape[3])
        Dv = Axis("v_head_dim", v_arr.shape[3])
        q = hax.named(q_arr, (Batch, Pos, Heads, Dk))
        k = hax.named(k_arr, (Batch, Pos, Heads, Dk))
        v = hax.named(v_arr, (Batch, Pos, Heads, Dv))
        gg = hax.named(g_arr, (Batch, Pos, Heads))
        bb = hax.named(beta_arr, (Batch, Pos, Heads))
    else:
        Batch = Axis("batch", q_arr.shape[0])
        Heads = Axis("heads", q_arr.shape[1])
        Pos = Axis("position", q_arr.shape[2])
        Dk = Axis("k_head_dim", q_arr.shape[3])
        Dv = Axis("v_head_dim", v_arr.shape[3])
        q = hax.named(q_arr, (Batch, Heads, Pos, Dk))
        k = hax.named(k_arr, (Batch, Heads, Pos, Dk))
        v = hax.named(v_arr, (Batch, Heads, Pos, Dv))
        gg = hax.named(g_arr, (Batch, Heads, Pos))
        bb = hax.named(beta_arr, (Batch, Heads, Pos))

    out_named, final = _chunk_gated_delta_rule_flash_tiled(
        q,
        k,
        v,
        gg,
        bb,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        head_first=head_first,
        offsets=offsets if use_varlen else None,
    )
    return out_named.array, final


def _chunk_gdn_flash_array_fwd(
    q_arr,
    k_arr,
    v_arr,
    g_arr,
    beta_arr,
    initial_state,
    chunk_size,
    output_final_state,
    use_qk_l2norm_in_kernel,
    head_first,
    use_varlen,
    offsets,
):
    out_arr, final = _chunk_gdn_flash_array(
        q_arr,
        k_arr,
        v_arr,
        g_arr,
        beta_arr,
        initial_state,
        chunk_size,
        output_final_state,
        use_qk_l2norm_in_kernel,
        head_first,
        use_varlen,
        offsets,
    )
    residual = (
        q_arr,
        k_arr,
        v_arr,
        g_arr,
        beta_arr,
        initial_state,
        chunk_size,
        use_qk_l2norm_in_kernel,
        head_first,
        use_varlen,
        offsets,
    )
    return (out_arr, final), residual


def _chunk_gdn_flash_array_bwd(
    chunk_size,
    output_final_state,
    use_qk_l2norm_in_kernel,
    head_first,
    use_varlen,
    offsets,
    residual,
    tangents,
):
    (
        q_arr,
        k_arr,
        v_arr,
        g_arr,
        beta_arr,
        init_arr,
        chunk_size_res,
        use_norm_res,
        head_first_res,
        use_varlen_res,
        offsets_res,
    ) = residual
    dout, dfinal = tangents

    # Rematerialize via reference kernel for VJP (memory friendly and already parity-checked)
    Batch = Axis("batch", q_arr.shape[0])
    if not head_first_res:
        Pos = Axis("position", q_arr.shape[1])
        Heads = Axis("heads", q_arr.shape[2])
        Dk = Axis("k_head_dim", q_arr.shape[3])
        Dv = Axis("v_head_dim", v_arr.shape[3])
        qn = hax.named(q_arr, (Batch, Pos, Heads, Dk))
        kn = hax.named(k_arr, (Batch, Pos, Heads, Dk))
        vn = hax.named(v_arr, (Batch, Pos, Heads, Dv))
        gn = hax.named(g_arr, (Batch, Pos, Heads))
        bn = hax.named(beta_arr, (Batch, Pos, Heads))
    else:
        Heads = Axis("heads", q_arr.shape[1])
        Pos = Axis("position", q_arr.shape[2])
        Dk = Axis("k_head_dim", q_arr.shape[3])
        Dv = Axis("v_head_dim", v_arr.shape[3])
        qn = hax.named(q_arr, (Batch, Heads, Pos, Dk))
        kn = hax.named(k_arr, (Batch, Heads, Pos, Dk))
        vn = hax.named(v_arr, (Batch, Heads, Pos, Dv))
        gn = hax.named(g_arr, (Batch, Heads, Pos))
        bn = hax.named(beta_arr, (Batch, Heads, Pos))

    def _ref_fun(q_in, k_in, v_in, g_in, beta_in, init_in):
        out_ref, fin_ref = _chunk_gated_delta_rule_reference(
            q_in,
            k_in,
            v_in,
            g_in,
            beta_in,
            chunk_size=chunk_size_res,
            initial_state=init_in,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_norm_res,
        )
        return out_ref.array, fin_ref

    init_default = (
        init_arr
        if init_arr is not None
        else jnp.zeros((q_arr.shape[0], q_arr.shape[2], q_arr.shape[3], v_arr.shape[3]), dtype=jnp.float32)
    )
    _, vjp_fun = jax.vjp(_ref_fun, qn, kn, vn, gn, bn, init_default)
    dout_arr = dout if not isinstance(dout, NamedArray) else dout.array
    dfinal_arr = jnp.zeros_like(init_default) if dfinal is None else dfinal
    gq, gk, gv, gg, gb, ginit = vjp_fun((dout_arr, dfinal_arr))

    return (
        gq.array if isinstance(gq, NamedArray) else gq,
        gk.array if isinstance(gk, NamedArray) else gk,
        gv.array if isinstance(gv, NamedArray) else gv,
        gg.array if isinstance(gg, NamedArray) else gg,
        gb.array if isinstance(gb, NamedArray) else gb,
        (None if init_arr is None else ginit),
    )


_chunk_gdn_flash_array.defvjp(_chunk_gdn_flash_array_fwd, _chunk_gdn_flash_array_bwd)


def chunk_gated_delta_rule(
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    g: NamedArray,
    beta: NamedArray,
    chunk_size: int = 64,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_flash: bool = True,
    *,
    head_first: bool = False,
    offsets: Optional[jnp.ndarray] = None,
    use_varlen: bool = False,
) -> tuple[NamedArray, Optional[jnp.ndarray]]:
    if use_flash:
        out_arr, fin = _chunk_gdn_flash_array(
            query.array,
            key.array,
            value.array,
            g.array,
            beta.array,
            initial_state,
            chunk_size,
            output_final_state,
            use_qk_l2norm_in_kernel,
            head_first,
            use_varlen,
            offsets,
        )
        out_named = hax.named(
            out_arr,
            (
                value.axes
                if not head_first
                else (
                    query.resolve_axis("batch"),
                    query.resolve_axis("heads"),
                    query.resolve_axis("position"),
                    value.resolve_axis("v_head_dim"),
                )
            ),
        )
        return out_named, fin

    # reference fallback
    return _chunk_gated_delta_rule_reference(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


# ---------- Layer ----------


class GatedDeltaNet(eqx.Module):
    """Complete Gated DeltaNet layer (projections + conv + kernels + norm + out proj).

    Block structure (per token t):
      1) Linear projections → [Q | K | V | Z] and [b | a]
      2) Short depthwise causal Conv1D over concatenated [Q|K|V] channels
      3) Compute gates:
           β_t = σ(b_t)              (per V-head)
           g_t = -exp(A)·softplus(a_t + dt_bias)   (per V-head)
           α_t = exp(g_t)
      4) Core kernel:
           - prefill/train:  chunk_gated_delta_rule (chunkwise parallel, returns S_T)
           - decode:         recurrent_gated_delta_rule (sequential, updates S)
      5) Gated RMSNorm with Z:  RMSNorm(o) * SiLU(Z)
      6) Output projection back to model dim.

    Caching (inference):
      - conv_state: (N, Channels, K) running window for the causal depthwise conv
      - S_state:    (B, H, d_k, d_v) cross-chunk recurrent state for the delta rule

    Head layout:
      - If num_v_heads > num_k_heads, Q/K are repeated across V-head groups so each V-head
        has a corresponding Q,K.
    """

    config: GatedDeltaNetConfig = eqx.field(static=True)

    # projections
    in_proj_qkvz: hnn.Linear  # [Embed] -> [Q|K|V|Z]
    in_proj_ba: hnn.Linear  # [Embed] -> [b|a]

    # depthwise conv parameters over concatenated [Q|K|V] channels
    conv_weight: NamedArray  # [channels, conv_kernel]
    conv_bias: Optional[NamedArray]  # [channels] or None

    # discretization params per V head (Mamba2-style)
    A_log: NamedArray  # [Heads]
    dt_bias: NamedArray  # [Heads]

    # gated RMSNorm and output projection
    o_norm: FusedRMSNormGated
    out_proj: hnn.Linear  # [Heads, VHeadDim] -> [Embed]

    @staticmethod
    def init(config: GatedDeltaNetConfig, *, key) -> "GatedDeltaNet":
        """Initializer mirrors the HF defaults: no biases in projections/out_proj;
        A_log ~ log U(0,16), dt_bias = 1, small conv kernel."""
        k_qkvz, k_ba, k_conv, k_out = jax.random.split(key, 4)
        in_proj_qkvz = hnn.Linear.init(
            In=config.Embed,
            Out=config.mix_qkvz_axis,
            out_first=True,
            use_bias=False,
            key=k_qkvz,
        )
        in_proj_ba = hnn.Linear.init(
            In=config.Embed,
            Out=config.ba_axis,
            out_first=True,
            use_bias=False,
            key=k_ba,
        )

        # Depthwise conv over channels = 2*key_dim + value_dim
        C = config.key_dim * 2 + config.value_dim
        K = config.conv_kernel_size
        ConvChannels = Axis("channels", C)
        ConvKernel = Axis("conv_kernel", K)

        conv_w = jax.random.normal(k_conv, (C, K), dtype=jnp.float32) * (1.0 / jnp.sqrt(C * K))
        conv_weight = hax.named(conv_w, (ConvChannels, ConvKernel))
        conv_bias = None

        # GDN discretization parameters (per V-head)
        A_log = hax.named(
            jnp.log(jax.random.uniform(k_out, (config.Heads.size,), minval=1e-6, maxval=16.0, dtype=jnp.float32)),
            (config.Heads.name,),
        )
        dt_bias = hax.named(jnp.ones((config.Heads.size,), dtype=jnp.float32), (config.Heads.name,))

        o_norm = FusedRMSNormGated.init(config.VHeadDim, eps=config.rms_norm_eps)
        out_proj = hnn.Linear.init(
            In=(config.Heads, config.VHeadDim), Out=config.Embed, out_first=True, use_bias=False, key=k_out
        )
        return GatedDeltaNet(
            config=config,
            in_proj_qkvz=in_proj_qkvz,
            in_proj_ba=in_proj_ba,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            A_log=A_log,
            dt_bias=dt_bias,
            o_norm=o_norm,
            out_proj=out_proj,
        )

    def _fix_qkvz_ordering(
        self,
        mixed_qkvz: NamedArray,  # [B, Pos, qkvz=2*key_dim + 2*value_dim]
        mixed_ba: NamedArray,  # [B, Pos, 2*num_v_heads]
    ) -> Tuple[NamedArray, NamedArray, NamedArray, NamedArray, NamedArray, NamedArray]:
        """Split packed projections into per-head tensors and align head layout. (match HF version)

        Input shapes:
          mixed_qkvz: [B, Pos, 2*key_dim + 2*value_dim]  (Q|K|V|Z concatenated)
          mixed_ba:   [B, Pos, 2*num_v_heads]            (b|a per V-head)

        Returns:
          q: [B, Pos, KHeads, KHeadDim]
          k: [B, Pos, KHeads, KHeadDim]
          v: [B, Pos, VHeads, VHeadDim]
          z: [B, Pos, VHeads, VHeadDim]
          b: [B, Pos, VHeads]        (→ β via sigmoid)
          a: [B, Pos, VHeads]        (→ g via Mamba2-style discretization)
        """
        cfg = self.config
        ratio = cfg.num_v_heads // cfg.num_k_heads

        per_head = Axis("per_head", 2 * cfg.head_k_dim + 2 * ratio * cfg.head_v_dim)
        x = mixed_qkvz.unflatten_axis("qkvz", (cfg.KHeads, per_head))

        def sl(start, size):
            return hax.ds(start, size)

        # per-head order: [Q (dk)] [K (dk)] [V-chunk (ratio*dv)] [Z-chunk (ratio*dv)]
        q = x["per_head", sl(0, cfg.head_k_dim)].rename({"per_head": cfg.KHeadDim.name})
        k = x["per_head", sl(cfg.head_k_dim, cfg.head_k_dim)].rename({"per_head": cfg.KHeadDim.name})
        v_chunk = x["per_head", sl(2 * cfg.head_k_dim, ratio * cfg.head_v_dim)]
        z_chunk = x["per_head", sl(2 * cfg.head_k_dim + ratio * cfg.head_v_dim, ratio * cfg.head_v_dim)]

        # (KHeads, ratio*dv) → (VHeads, VHeadDim)
        v = v_chunk.unflatten_axis(
            v_chunk.resolve_axis("per_head"), (Axis("v_group", ratio), cfg.VHeadDim)
        ).flatten_axes(("k_heads", "v_group"), cfg.VHeads)
        z = z_chunk.unflatten_axis(
            z_chunk.resolve_axis("per_head"), (Axis("v_group", ratio), cfg.VHeadDim)
        ).flatten_axes(("k_heads", "v_group"), cfg.VHeads)

        # b | a are per V-head; shape path mirrors HF:
        per_ba = Axis("per_ba", 2 * ratio)
        ba = mixed_ba.unflatten_axis("ba", (cfg.KHeads, per_ba))
        b_chunk = ba["per_ba", hax.ds(0, ratio)]
        a_chunk = ba["per_ba", hax.ds(ratio, ratio)]
        b = b_chunk.flatten_axes(("k_heads", "per_ba"), cfg.VHeads)
        a = a_chunk.flatten_axes(("k_heads", "per_ba"), cfg.VHeads)

        return q, k, v, z, b, a

    def __call__(
        self,
        x: NamedArray,
        *,
        inference: bool = True,
        chunk_size: int = 64,
        attention_mask: Optional[NamedArray] = None,
        decode_state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,  # (conv_state, S_state)
    ) -> Tuple[NamedArray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """Run the full GDN token mixer.

        Args:
          x: [B, Pos, Embed]
          inference: if True, returns and expects state for streaming decode.
          chunk_size: chunk length for the parallel kernel (prefill/train).
          attention_mask: optional [B, Pos] (1 for real tokens, 0 for pad).
          decode_state: optional tuple (conv_state, S_state) for streaming decode:
              conv_state: (N, Channels, K)
              S_state:    (B, VHeads, d_k, d_v)

        Returns:
          y_out: [B, Pos, Embed]
          new_state (optional): (conv_state, S_state) if inference=True
        """
        cfg = self.config

        # Zero out padding tokens early so they don't affect conv or states.
        if attention_mask is not None:
            m3 = attention_mask.astype(x.dtype).broadcast_axis(cfg.Embed)
            x = x * m3

        _dbg("layer/in_x", x.array if hasattr(x, "array") else x)

        # 1) Project to [Q|K|V|Z] and [b|a]
        mixed_qkvz = self.in_proj_qkvz(x)  # [B, Pos, qkvz=2*key_dim + 2*value_dim]
        mixed_ba = self.in_proj_ba(x)  # [B, Pos, ba=2*num_v_heads]
        _dbg("layer/mixed_qkvz", mixed_qkvz.array if hasattr(mixed_qkvz, "array") else mixed_qkvz)
        _dbg("layer/mixed_ba", mixed_ba.array if hasattr(mixed_ba, "array") else mixed_ba)

        # 1b) Re-group like HF for parity (also used for conv channel ordering)
        q, k, v, z, b, a = self._fix_qkvz_ordering(mixed_qkvz, mixed_ba)

        # 2) Depthwise causal conv over concatenated [Q|K|V] channels
        #    HF orders channels as: [Q_flat | K_flat | V_flat] (no Z).
        q_ch = q.flatten_axes((cfg.KHeads, cfg.KHeadDim), Axis("channels", cfg.key_dim))
        k_ch = k.flatten_axes((cfg.KHeads, cfg.KHeadDim), Axis("channels", cfg.key_dim))
        v_ch = v.flatten_axes((cfg.VHeads, cfg.VHeadDim), Axis("channels", cfg.value_dim))
        qkv_ch = hax.concatenate("channels", [q_ch, k_ch, v_ch])  # [B, Pos, channels]
        qkv_ncl = hax.rearrange(qkv_ch, ("batch", "channels", "position")).array  # (N, C, L)
        _dbg("conv/in_ncl", qkv_ncl)

        S_state: Optional[jnp.ndarray] = None
        if decode_state is not None and x.axis_size("position") == 1:
            # Streaming decode: cheap single-step conv update + carry conv_state
            conv_state, S_state = decode_state
            K = self.conv_weight.resolve_axis("conv_kernel").size
            assert conv_state.shape[-1] == K
            _dbg("conv/state_in_decode", conv_state)
            y_ncl, new_conv_state = _causal_depthwise_conv1d_update(
                qkv_ncl,
                self.conv_weight.array,
                self.conv_bias.array if self.conv_bias is not None else None,
                conv_state,
            )
        else:
            # Prefill/train: full causal conv over the sequence
            y_ncl = _causal_depthwise_conv1d_full(
                qkv_ncl, self.conv_weight.array, self.conv_bias.array if self.conv_bias is not None else None
            )
            if inference:
                # cache the rightmost K samples of channels as the next conv_state
                K = self.conv_weight.resolve_axis("conv_kernel").size
                Lpos = x.axis_size("position")
                if Lpos >= K:
                    new_conv_state = qkv_ncl[..., -K:]
                else:
                    new_conv_state = jnp.pad(qkv_ncl, ((0, 0), (0, 0), (K - Lpos, 0)))
            else:
                new_conv_state = None
                S_state = None

        _dbg("conv/out_ncl", y_ncl)

        # Unpack [Q|K|V] after conv back to per-head tensors (mirror the same channel order)
        y_bpc = hax.rearrange(hax.named(y_ncl, ("batch", "channels", "position")), ("batch", "position", "channels"))
        q_y = y_bpc["channels", hax.ds(0, cfg.key_dim)]
        k_y = y_bpc["channels", hax.ds(cfg.key_dim, cfg.key_dim)]
        v_y = y_bpc["channels", hax.ds(2 * cfg.key_dim, cfg.value_dim)]
        q = q_y.unflatten_axis("channels", (cfg.KHeads, cfg.KHeadDim))
        k = k_y.unflatten_axis("channels", (cfg.KHeads, cfg.KHeadDim))
        v = v_y.unflatten_axis("channels", (cfg.VHeads, cfg.VHeadDim))

        # 3) Gates: β via sigmoid(b); α via g = -exp(A) * softplus(a + dt_bias), α=exp(g)
        # Map a, b to Heads axis to line up with TP and kernels.
        ratio = cfg.num_v_heads // cfg.num_k_heads
        if ratio > 1:
            VGroup = Axis("v_group", ratio)
            # Repeat Q,K to Heads (num_v_heads)
            q = q.broadcast_axis(VGroup).flatten_axes((cfg.KHeads, VGroup), cfg.Heads)
            k = k.broadcast_axis(VGroup).flatten_axes((cfg.KHeads, VGroup), cfg.Heads)
            # Map V/Z/B/A to Heads as well
            v_h = v.rename({cfg.VHeads.name: cfg.Heads.name})
            z_h = z.rename({cfg.VHeads.name: cfg.Heads.name})
            b_hparam = b.rename({cfg.VHeads.name: cfg.Heads.name})
            a_hparam = a.rename({cfg.VHeads.name: cfg.Heads.name})
        else:
            # 1:1 map KHeads -> Heads; and VHeads -> Heads for v/z/a/b
            q = q.rename({cfg.KHeads.name: cfg.Heads.name})
            k = k.rename({cfg.KHeads.name: cfg.Heads.name})
            v_h = v.rename({cfg.VHeads.name: cfg.Heads.name})
            z_h = z.rename({cfg.VHeads.name: cfg.Heads.name})
            b_hparam = b.rename({cfg.VHeads.name: cfg.Heads.name})
            a_hparam = a.rename({cfg.VHeads.name: cfg.Heads.name})

        beta = hnn.sigmoid(b_hparam)
        a32 = a_hparam.astype(jnp.float32)
        dt_bias_na = self.dt_bias.astype(jnp.float32)
        A_exp = hax.exp(self.A_log.astype(jnp.float32))
        g = -(A_exp * hnn.softplus(a32 + dt_bias_na)).astype(x.dtype)  # log-decay on Heads

        # 4) Kernels expect [batch, position, heads, dim] (axis name "heads")
        q_h = q.rename({cfg.Heads.name: "heads"})
        k_h = k.rename({cfg.Heads.name: "heads"})
        v_kern = v_h.rename({cfg.Heads.name: "heads"})
        g_h = g.rename({cfg.Heads.name: "heads"})
        b_h = beta.rename({cfg.Heads.name: "heads"})

        q_bphd = hax.rearrange(q_h, ("batch", "position", "heads", cfg.KHeadDim.name))
        k_bphd = hax.rearrange(k_h, ("batch", "position", "heads", cfg.KHeadDim.name))
        v_bphd = hax.rearrange(v_kern, ("batch", "position", "heads", cfg.VHeadDim.name))
        _dbg("kernel/q_bphd", q_bphd.array)
        _dbg("kernel/k_bphd", k_bphd.array)
        _dbg("kernel/v_bphd", v_bphd.array)

        # Choose the kernel:
        if decode_state is not None and x.axis_size("position") == 1 and S_state is not None:
            out_bphd, S_new = recurrent_gated_delta_rule(
                q_bphd,
                k_bphd,
                v_bphd,
                g_h,
                b_h,
                initial_state=S_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            out_bphd, S_new = chunk_gated_delta_rule(
                q_bphd,
                k_bphd,
                v_bphd,
                g_h,
                b_h,
                chunk_size=chunk_size,
                initial_state=None,
                output_final_state=inference,
                use_qk_l2norm_in_kernel=True,
            )

        # Keep the kernel output on "heads" so TP can shard the out-projection.
        out = out_bphd  # [B, Pos, heads, VHeadDim]
        _dbg("kernel/out_bphd", out.array)

        # 5) Gated RMSNorm with Z (rename Z to "heads" to match)
        z_gate = z_h.rename({cfg.Heads.name: "heads"})
        y_norm = self.o_norm(out, gate=z_gate)

        # 6) Output projection back to model dimension (In=(Heads, VHeadDim) -> Out=Embed)
        y_out = self.out_proj(y_norm.astype(x.dtype))
        _dbg("layer/y_out", y_out.array)

        # State packing for streaming
        new_state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
        if inference and (new_conv_state is not None) and (S_new is not None):
            new_state = (new_conv_state, S_new)
        return y_out, new_state

    def to_state_dict(self) -> dict[str, jnp.ndarray]:
        return {
            "in_proj_qkvz.weight": jnp.array(self.in_proj_qkvz.weight.array),
            "in_proj_ba.weight": jnp.array(self.in_proj_ba.weight.array),
            "conv_weight": jnp.array(self.conv_weight.array),
            "A_log": jnp.array(self.A_log.array),
            "dt_bias": jnp.array(self.dt_bias.array),
            "o_norm.weight": jnp.array(self.o_norm.weight.array),
            "out_proj.weight": jnp.array(self.out_proj.weight.array),
        }

    def load_state_dict(self, state: dict[str, jnp.ndarray]) -> "GatedDeltaNet":
        cfg = self.config

        def _assign_linear_weight(named_linear: hnn.Linear, np_weight: jnp.ndarray, out_axis: Axis, in_axis: Axis):
            w_named = hax.named(jnp.asarray(np_weight, dtype=jnp.float32), (out_axis.name, in_axis.name))
            return dataclasses.replace(named_linear, weight=w_named)

        new_in_proj_qkvz = _assign_linear_weight(
            self.in_proj_qkvz, state["in_proj_qkvz.weight"], cfg.mix_qkvz_axis, cfg.Embed
        )
        new_in_proj_ba = _assign_linear_weight(self.in_proj_ba, state["in_proj_ba.weight"], cfg.ba_axis, cfg.Embed)

        # Rebuild named conv axes
        ConvChannels = Axis("channels", cfg.key_dim * 2 + cfg.value_dim)
        ConvKernel = Axis("conv_kernel", cfg.conv_kernel_size)
        new_conv_weight = hax.named(jnp.asarray(state["conv_weight"], dtype=jnp.float32), (ConvChannels, ConvKernel))

        # Heads-based params
        new_A_log = hax.named(jnp.asarray(state["A_log"], dtype=jnp.float32), (cfg.Heads.name,))
        new_dt_bias = hax.named(jnp.asarray(state["dt_bias"], dtype=jnp.float32), (cfg.Heads.name,))
        new_o_norm = dataclasses.replace(
            self.o_norm, weight=hax.named(jnp.asarray(state["o_norm.weight"], dtype=jnp.float32), (cfg.VHeadDim.name,))
        )

        # out_proj.weight is (Embed, Heads, VHeadDim)
        out_w = jnp.asarray(state["out_proj.weight"], dtype=jnp.float32)
        new_out_proj = dataclasses.replace(
            self.out_proj, weight=hax.named(out_w, (cfg.Embed.name, cfg.Heads.name, cfg.VHeadDim.name))
        )

        return dataclasses.replace(
            self,
            in_proj_qkvz=new_in_proj_qkvz,
            in_proj_ba=new_in_proj_ba,
            conv_weight=new_conv_weight,
            A_log=new_A_log,
            dt_bias=new_dt_bias,
            o_norm=new_o_norm,
            out_proj=new_out_proj,
        )

    @classmethod
    def from_state_dict(cls, config: GatedDeltaNetConfig, state: dict[str, jnp.ndarray], *, key) -> "GatedDeltaNet":
        layer = cls.init(config, key=key)
        return layer.load_state_dict(state)
