# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# based on:
# - https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py
# - the JAX implementation by Yu Sun and Leo Lee
# - Flash Linear Attention's Triton implementation: https://github.com/fla-org/flash-linear-attention

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
import contextlib

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


@contextlib.contextmanager
def _fp32_mm():
    with jax.default_matmul_precision("float32"):
        yield


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


@jax.custom_vjp
def _rmsnorm_gated_flash(x_2d: jnp.ndarray, gate_2d: jnp.ndarray, weight: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Forward: call the Pallas fused kernel (fallback to reference).
    Backward: analytic grads in pure JAX (no Pallas autodiff)."""
    try:
        return _fused_rmsnorm_gated_pallas(x_2d, gate_2d, weight, eps)
    except Exception:
        return _rmsnorm_gated_reference(x_2d, gate_2d, weight, eps)


def _rmsnorm_gated_flash_fwd(x_2d, gate_2d, weight, eps):
    y = _rmsnorm_gated_flash(x_2d, gate_2d, weight, eps)
    # Keep inputs as residuals; recompute inv etc. in bwd
    return y, (x_2d, gate_2d, weight, jnp.asarray(eps, dtype=jnp.float32))


def _rmsnorm_gated_flash_bwd(res, dY):
    x_2d, gate_2d, weight, eps32 = res
    # Promote to fp32
    x = x_2d.astype(jnp.float32)
    gate = gate_2d.astype(jnp.float32)
    w = weight.astype(jnp.float32)
    dY = dY.astype(jnp.float32)

    # Row-wise RMSNorm stats
    # inv = 1 / sqrt(mean(x^2) + eps)
    D = x.shape[-1]
    inv = lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps32)  # [N,1]
    y_no_gate = (x * inv) * w  # [N,D]

    # SiLU and derivative
    sigma = jax.nn.sigmoid(gate)  # [N,D]
    silu = gate * sigma
    silu_prime = sigma * (1.0 + gate * (1.0 - sigma))  # [N,D]

    # Chain rule
    dy = dY * silu  # [N,D]
    # y_no_gate = (x*inv) * w = u * w, with u = x*inv
    du = dy * w  # [N,D]

    # dweight: sum over rows of dy * (x * inv)
    dW = jnp.sum(dy * (x * inv), axis=0)  # [D]

    # dx: u = x*inv  with inv = (mean(x^2)+eps)^(-1/2)
    # dL/dx = inv * du + (-inv^3 / D) * x * dot(du, x), where dot is along last dim
    dot = jnp.sum(du * x, axis=-1, keepdims=True)  # [N,1]
    dx = inv * du + (-(inv**3) / float(D)) * x * dot  # [N,D]

    # dgate: elementwise (no reduction)
    dGate = dY * y_no_gate * silu_prime  # [N,D]

    # Cast back to input dtypes
    return (dx.astype(x_2d.dtype), dGate.astype(gate_2d.dtype), dW.astype(weight.dtype), None)  # no grad for eps


_rmsnorm_gated_flash.defvjp(_rmsnorm_gated_flash_fwd, _rmsnorm_gated_flash_bwd)


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
                out_arr = _rmsnorm_gated_flash(x_arr, gate_arr, weight_arr, self.eps)
            except Exception:
                # If Pallas fails at runtime, fall back to reference (grad still correct via custom VJP bwd)
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


def _pick_bk_tile_for_decode(dk: int) -> int:
    # Simple heuristic; autotune later if desired.
    if dk >= 256:
        return 64
    elif dk >= 128:
        return 64
    else:
        return 32


def _pick_bv_tile_for_decode(dv: int) -> int:
    if dv >= 512:
        return 128
    elif dv >= 256:
        return 64
    else:
        return 32


def _pad_last_axis(arr: jnp.ndarray, new_width: int) -> jnp.ndarray:
    """Right-pad the *last* axis to new_width."""
    cur = arr.shape[-1]
    if cur == new_width:
        return arr
    pad = new_width - cur
    assert pad >= 0
    pad_spec = [(0, 0)] * arr.ndim
    pad_spec[-1] = (0, pad)
    return jnp.pad(arr, tuple(pad_spec))


def _pad_k_for_decode(
    q_like_TK: jnp.ndarray,  # [..., T, K]
    k_like_TK: jnp.ndarray,  # [..., T, K]
    init_KV: jnp.ndarray,  # [..., K, V]
    K_pad: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    q_pad = _pad_last_axis(q_like_TK, K_pad)
    k_pad = _pad_last_axis(k_like_TK, K_pad)
    pad_K = K_pad - init_KV.shape[-2]
    if pad_K > 0:
        init_pad = jnp.pad(init_KV, ((0, 0),) * (init_KV.ndim - 2) + ((0, pad_K), (0, 0)))
    else:
        init_pad = init_KV
    return q_pad, k_pad, init_pad


def _nh_to_bh(nh_i32: jnp.int32, H: int) -> tuple[jnp.int32, jnp.int32]:
    """Map flattened nh ∈ [0, B·H) → (b, h)."""
    b = nh_i32 // jnp.int32(H)
    h = nh_i32 - b * jnp.int32(H)
    return b, h


def _in_specs_head_first(B, H, T, K_pad, BV, is_beta_headwise):
    # Inputs (HEAD_FIRST):
    #   q,k:  [B,H,T,K_pad]
    #   v:    [B,H,T,V_pad]
    #   g:    [B,H,T]
    #   β:    [B,H,T] or [B,H,T,V_pad]
    # State:
    #   init, final: [B,H,K_pad,V_pad]
    def _bh(nh, vb):
        return _nh_to_bh(nh, H)

    in_specs = (
        pl.BlockSpec((1, 1, T, K_pad), lambda nh, vb: (*_bh(nh, vb), 0, 0)),  # q
        pl.BlockSpec((1, 1, T, K_pad), lambda nh, vb: (*_bh(nh, vb), 0, 0)),  # k
        pl.BlockSpec((1, 1, T, BV), lambda nh, vb: (*_bh(nh, vb), 0, vb * BV)),  # v
        pl.BlockSpec((1, 1, T), lambda nh, vb: (*_bh(nh, vb), 0)),  # g
        (
            pl.BlockSpec((1, 1, T), lambda nh, vb: (*_bh(nh, vb), 0))  # β headwise
            if is_beta_headwise
            else pl.BlockSpec((1, 1, T, BV), lambda nh, vb: (*_bh(nh, vb), 0, vb * BV))
        ),  # β per‑V
        pl.BlockSpec((1, 1, K_pad, BV), lambda nh, vb: (*_bh(nh, vb), 0, vb * BV)),  # init
        pl.BlockSpec((1,), lambda nh, vb: (nh,)),  # lengths [NH]
    )
    # Outputs:
    #   out:   [NH, T, V_pad]   (3‑D)
    #   final: [B,H,K_pad,V_pad]
    out_specs = (
        pl.BlockSpec((1, T, BV), lambda nh, vb: (nh, 0, vb * BV)),  # out (NH-major)
        pl.BlockSpec((1, 1, K_pad, BV), lambda nh, vb: (*_bh(nh, vb), 0, vb * BV)),  # final
    )
    return in_specs, out_specs


def _in_specs_bth(B, H, T, K_pad, BV, is_beta_headwise):
    # Inputs (BTH):
    #   q,k:  [B, T, H, K_pad]
    #   v:    [B, T, H, V_pad]
    #   g:    [B, T, H]
    #   β:    [B, T, H] (headwise) or [B, T, H, V_pad] (per-V)
    # State:
    #   init, final: [B, H, K_pad, V_pad]

    def _bth(nh, vb):
        b, h = _nh_to_bh(nh, H)
        return (b, 0, h)  # (B, T_start, H)

    in_specs = (
        # q, k: 4-D tiles (1, T, 1, K_pad)
        pl.BlockSpec((1, T, 1, K_pad), lambda nh, vb: (*_bth(nh, vb), 0)),
        pl.BlockSpec((1, T, 1, K_pad), lambda nh, vb: (*_bth(nh, vb), 0)),
        # v: 4-D tile (1, T, 1, BV)
        pl.BlockSpec((1, T, 1, BV), lambda nh, vb: (*_bth(nh, vb), vb * BV)),
        # g: 3-D tile (1, T, 1)  → must return exactly 3 indices (b, 0, h)
        pl.BlockSpec((1, T, 1), lambda nh, vb: _bth(nh, vb)),
        # β: headwise uses 3-D (1, T, 1); per-V uses 4-D (1, T, 1, BV)
        (
            pl.BlockSpec((1, T, 1), lambda nh, vb: _bth(nh, vb))  # headwise β
            if is_beta_headwise
            else pl.BlockSpec((1, T, 1, BV), lambda nh, vb: (*_bth(nh, vb), vb * BV))
        ),  # per-V β
        # init state: 4-D (1, 1, K_pad, BV) into [B,H,K_pad,V_pad]
        pl.BlockSpec((1, 1, K_pad, BV), lambda nh, vb: (_nh_to_bh(nh, H)[0], _nh_to_bh(nh, H)[1], 0, vb * BV)),
        # lengths: 1-D (1,) over NH
        pl.BlockSpec((1,), lambda nh, vb: (nh,)),
    )

    out_specs = (
        # out: NH-major 3-D (1, T, BV)
        pl.BlockSpec((1, T, BV), lambda nh, vb: (nh, 0, vb * BV)),
        # final state: 4-D (1, 1, K_pad, BV) into [B,H,K_pad,V_pad]
        pl.BlockSpec((1, 1, K_pad, BV), lambda nh, vb: (_nh_to_bh(nh, H)[0], _nh_to_bh(nh, H)[1], 0, vb * BV)),
    )
    return in_specs, out_specs


def _gdn_recurrent_fwd_kernel_tiled_2d(
    q_ref,
    k_ref,
    v_ref,
    g_ref,
    beta_ref,
    init_ref,
    lengths_ref,
    out_ref,
    final_ref,
    *,
    T,
    K_pad,
    BK,
    BV,
    use_qk_l2norm,
    has_initial_state,
    is_beta_headwise,
    scale,
    head_first_layout: bool,
):
    # ---- Local (tile) views; squeeze the 1-sized dims to get 2-D/1-D arrays ----
    if head_first_layout:
        # q,k: [1,1,T,K] → (T,K); v: [1,1,T,BV] → (T,BV); g: [1,1,T] → (T,)
        q_view = q_ref[dslice(0, 1), dslice(0, 1), dslice(0, T), dslice(0, K_pad)][0, 0]
        k_view = k_ref[dslice(0, 1), dslice(0, 1), dslice(0, T), dslice(0, K_pad)][0, 0]
        v_view = v_ref[dslice(0, 1), dslice(0, 1), dslice(0, T), dslice(0, BV)][0, 0]
        g_view = g_ref[dslice(0, 1), dslice(0, 1), dslice(0, T)][0, 0]
        if is_beta_headwise:
            beta_h = beta_ref[dslice(0, 1), dslice(0, 1), dslice(0, T)][0, 0]  # (T,)
        else:
            beta_h = beta_ref[dslice(0, 1), dslice(0, 1), dslice(0, T), dslice(0, BV)][0, 0]  # (T,BV)

        # State tiles: [1,1,K,BV] → (K,BV)
        def _read_state(k0, n):
            return final_ref[dslice(0, 1), dslice(0, 1), dslice(k0, n), dslice(0, BV)][0, 0]

        def _write_state(k0, block):
            final_ref[dslice(0, 1), dslice(0, 1), dslice(k0, block.shape[0]), dslice(0, BV)] = block[None, None, :, :]

        def _read_init(k0, n):
            return init_ref[dslice(0, 1), dslice(0, 1), dslice(k0, n), dslice(0, BV)][0, 0]

    else:
        # q,k: [1,T,1,K] → (T,K); v: [1,T,1,BV] → (T,BV); g: [1,T,1] → (T,)
        q_view = q_ref[dslice(0, 1), dslice(0, T), dslice(0, 1), dslice(0, K_pad)][0, :, 0, :]
        k_view = k_ref[dslice(0, 1), dslice(0, T), dslice(0, 1), dslice(0, K_pad)][0, :, 0, :]
        v_view = v_ref[dslice(0, 1), dslice(0, T), dslice(0, 1), dslice(0, BV)][0, :, 0, :]
        g_view = g_ref[dslice(0, 1), dslice(0, T), dslice(0, 1)][0, :, 0]
        if is_beta_headwise:
            beta_h = beta_ref[dslice(0, 1), dslice(0, T), dslice(0, 1)][0, :, 0]  # (T,)
        else:
            beta_h = beta_ref[dslice(0, 1), dslice(0, T), dslice(0, 1), dslice(0, BV)][0, :, 0, :]  # (T,BV)

        # State tiles: [1,1,K,BV] → (K,BV)
        def _read_state(k0, n):
            return final_ref[dslice(0, 1), dslice(0, 1), dslice(k0, n), dslice(0, BV)][0, 0]

        def _write_state(k0, block):
            final_ref[dslice(0, 1), dslice(0, 1), dslice(k0, block.shape[0]), dslice(0, BV)] = block[None, None, :, :]

        def _read_init(k0, n):
            return init_ref[dslice(0, 1), dslice(0, 1), dslice(k0, n), dslice(0, BV)][0, 0]

    # ---- Initialize per-tile state buffer from init_ref once ----
    n_ktiles = K_pad // BK

    def _copy_body(kb, _):
        k0 = kb * BK
        S_blk = _read_init(k0, BK).astype(jnp.float32) if has_initial_state else jnp.zeros((BK, BV), dtype=jnp.float32)
        _write_state(k0, S_blk.astype(final_ref.dtype))
        return ()

    _ = lax.fori_loop(0, n_ktiles, _copy_body, ())

    L_i32 = lengths_ref[dslice(0, 1)][0].astype(jnp.int32)
    T_i32 = jnp.int32(T)
    scale32 = jnp.asarray(scale, dtype=jnp.float32)

    out_tile = jnp.zeros((T, BV), dtype=out_ref.dtype)

    def time_step(t, out_cur):
        do_step = t < L_i32

        q_t = q_view[t].astype(jnp.float32)  # (K_pad,)
        k_t = k_view[t].astype(jnp.float32)  # (K_pad,)
        v_t = v_view[t].astype(jnp.float32)  # (BV,)
        g_t = g_view[t].astype(jnp.float32)
        alpha = jnp.exp(g_t)

        if use_qk_l2norm:
            q_norm = jnp.sqrt(jnp.sum(q_t * q_t) + 1e-6)
            k_norm = jnp.sqrt(jnp.sum(k_t * k_t) + 1e-6)
            q_t = q_t / jnp.where(q_norm > 0.0, q_norm, 1.0)
            k_t = k_t / jnp.where(k_norm > 0.0, k_norm, 1.0)
        q_t = q_t * scale32

        def _do_step(_):
            # ---- Pass 1: accumulate kv, y_alpha, kq (no writes) ----
            kv = jnp.zeros((BV,), dtype=jnp.float32)
            y_alpha = jnp.zeros((BV,), dtype=jnp.float32)
            kq = jnp.array(0.0, dtype=jnp.float32)

            def pass1_body(kb, acc):
                kv_acc, yA_acc, kq_acc = acc
                k0 = kb * BK
                k_chunk = lax.dynamic_slice_in_dim(k_t, start_index=k0, slice_size=BK, axis=0)
                q_chunk = lax.dynamic_slice_in_dim(q_t, start_index=k0, slice_size=BK, axis=0)
                S_blk = _read_state(k0, BK).astype(jnp.float32)

                kv_acc = kv_acc + jnp.sum(S_blk * (alpha * k_chunk)[:, None], axis=0)
                yA_acc = yA_acc + jnp.sum(S_blk * (alpha * q_chunk)[:, None], axis=0)
                kq_acc = kq_acc + jnp.sum(k_chunk * q_chunk)
                return (kv_acc, yA_acc, kq_acc)

            kv, y_alpha, kq = lax.fori_loop(0, n_ktiles, pass1_body, (kv, y_alpha, kq))

            # δ = (v - kv) * β
            if is_beta_headwise:
                delta = (v_t - kv) * beta_h[t].astype(jnp.float32)
            else:
                delta = (v_t - kv) * beta_h[t].astype(jnp.float32)

            # y = y_alpha + δ * (k^T q)
            y_t = (y_alpha + delta * kq).astype(out_ref.dtype)

            # ---- Pass 2: S_new = α S_blk + k⊗δ (single write) ----
            def pass2_body(kb, _):
                k0 = kb * BK
                k_chunk = lax.dynamic_slice_in_dim(k_t, start_index=k0, slice_size=BK, axis=0)
                S_blk = _read_state(k0, BK).astype(jnp.float32)
                S_new = S_blk * alpha + k_chunk[:, None] * delta[None, :]
                _write_state(k0, S_new.astype(final_ref.dtype))
                return ()

            _ = lax.fori_loop(0, n_ktiles, pass2_body, ())
            return y_t

        def _skip_step(_):
            return jnp.zeros((BV,), dtype=out_ref.dtype)

        y_t = lax.cond(do_step, _do_step, _skip_step, operand=None)
        out_next = out_cur.at[t].set(y_t)
        return out_next

    out_tile = lax.fori_loop(0, T_i32, time_step, out_tile)

    # Local writes (out is NH-major 3‑D)
    out_ref[dslice(0, 1), dslice(0, T), dslice(0, BV)] = out_tile[None, :, :]


def _recurrent_gated_delta_rule_flash(
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    g: NamedArray,
    beta: NamedArray,
    *,
    initial_state: Optional[jnp.ndarray] = None,  # [B,H,dk,dv]
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    head_first: bool = False,  # True: inputs are [B,H,T,*]; False: [B,T,H,*]
    lengths: Optional[jnp.ndarray] = None,  # [B*H] or [B,H]
) -> Tuple[NamedArray, Optional[jnp.ndarray]]:
    Batch = query.resolve_axis("batch")
    Pos = query.resolve_axis("position")
    Heads = query.resolve_axis("heads")
    Dk = query.resolve_axis("k_head_dim")
    Dv = value.resolve_axis("v_head_dim")

    B_, T_, H_, K_ = Batch.size, Pos.size, Heads.size, Dk.size
    V_ = Dv.size
    NH = B_ * H_

    # Ensure arrays match the declared layout (no-op if already matching)
    def _ensure_layout(x_named: NamedArray, layout: tuple[str, ...]) -> jnp.ndarray:
        have = tuple(ax.name for ax in x_named.axes)
        if have == layout:
            return x_named.array
        return hax.rearrange(x_named, layout).array

    if head_first:
        q_arr = _ensure_layout(query.astype(jnp.float32), (Batch.name, Heads.name, Pos.name, Dk.name))  # [B,H,T,K]
        k_arr = _ensure_layout(key.astype(jnp.float32), (Batch.name, Heads.name, Pos.name, Dk.name))
        v_arr = _ensure_layout(value.astype(jnp.float32), (Batch.name, Heads.name, Pos.name, Dv.name))
        g_arr = _ensure_layout(g.astype(jnp.float32), (Batch.name, Heads.name, Pos.name))
        beta_axis_names = tuple(ax.name for ax in beta.axes)
        is_beta_headwise = Dv.name not in beta_axis_names
        if is_beta_headwise:
            beta_arr = _ensure_layout(beta.astype(jnp.float32), (Batch.name, Heads.name, Pos.name))  # [B,H,T]
        else:
            beta_arr = _ensure_layout(
                beta.astype(jnp.float32), (Batch.name, Heads.name, Pos.name, Dv.name)
            )  # [B,H,T,V]
    else:
        q_arr = _ensure_layout(query.astype(jnp.float32), (Batch.name, Pos.name, Heads.name, Dk.name))  # [B,T,H,K]
        k_arr = _ensure_layout(key.astype(jnp.float32), (Batch.name, Pos.name, Heads.name, Dk.name))
        v_arr = _ensure_layout(value.astype(jnp.float32), (Batch.name, Pos.name, Heads.name, Dv.name))
        g_arr = _ensure_layout(g.astype(jnp.float32), (Batch.name, Pos.name, Heads.name))
        beta_axis_names = tuple(ax.name for ax in beta.axes)
        is_beta_headwise = Dv.name not in beta_axis_names
        if is_beta_headwise:
            beta_arr = _ensure_layout(beta.astype(jnp.float32), (Batch.name, Pos.name, Heads.name))  # [B,T,H]
        else:
            beta_arr = _ensure_layout(
                beta.astype(jnp.float32), (Batch.name, Pos.name, Heads.name, Dv.name)
            )  # [B,T,H,V]

    # Initial state: [B,H,K,V]
    if initial_state is None:
        init_arr = jnp.zeros((B_, H_, K_, V_), dtype=jnp.float32)
        has_initial = False
    else:
        init_in = initial_state.astype(jnp.float32)
        if init_in.shape == (B_, H_, K_, V_):
            init_arr = init_in
        elif init_in.ndim == 3 and init_in.shape[0] == NH:
            init_arr = init_in.reshape(B_, H_, K_, V_)
        else:
            init_arr = hax.rearrange(
                hax.named(init_in, (Batch, Heads, Dk, Dv)),
                (Batch.name, Heads.name, Dk.name, Dv.name),
            ).array
        has_initial = True

    # Varlen
    if lengths is None:
        lengths_flat = jnp.full((NH,), T_, dtype=jnp.int32)
    else:
        lf = lengths
        if lf.ndim == 2 and lf.shape == (B_, H_):
            lf = lf.reshape(NH)
        lengths_flat = lf.astype(jnp.int32)

    # 2D tiling & padding
    BK = _pick_bk_tile_for_decode(Dk.size)
    BV = _pick_bv_tile_for_decode(Dv.size)
    K_pad = int(((K_ + BK - 1) // BK) * BK)
    V_pad = int(((V_ + BV - 1) // BV) * BV)

    if head_first:
        # Pad K with BH→NH reshape and back; pad V in place
        q_pad, k_pad, init_kpad = _pad_k_for_decode(
            q_arr.reshape(B_ * H_, T_, K_),
            k_arr.reshape(B_ * H_, T_, K_),
            init_arr.reshape(B_ * H_, K_, V_),
            K_pad,
        )
        q_pad = q_pad.reshape(B_, H_, T_, K_pad)
        k_pad = k_pad.reshape(B_, H_, T_, K_pad)
        init_kpad = init_kpad.reshape(B_, H_, K_pad, V_)
        v_pad = _pad_last_axis(v_arr, V_pad)  # [B,H,T,V_pad]
        beta_pad = beta_arr if is_beta_headwise else _pad_last_axis(beta_arr, V_pad)
    else:
        q_pad, k_pad, init_kpad = _pad_k_for_decode(
            q_arr.reshape(B_ * H_, T_, K_),
            k_arr.reshape(B_ * H_, T_, K_),
            init_arr.reshape(B_ * H_, K_, V_),
            K_pad,
        )
        q_pad = q_pad.reshape(B_, T_, H_, K_pad)
        k_pad = k_pad.reshape(B_, T_, H_, K_pad)
        init_kpad = init_kpad.reshape(B_, H_, K_pad, V_)
        v_pad = _pad_last_axis(v_arr, V_pad)  # [B,T,H,V_pad]
        beta_pad = beta_arr if is_beta_headwise else _pad_last_axis(beta_arr, V_pad)

    # Pallas shapes
    out_struct = jax.ShapeDtypeStruct((NH, T_, V_pad), value.dtype)  # NH-major for output
    final_struct = jax.ShapeDtypeStruct((B_, H_, K_pad, V_pad), jnp.float32)

    kernel_partial = functools.partial(
        _gdn_recurrent_fwd_kernel_tiled_2d,
        T=T_,
        K_pad=K_pad,
        BK=int(BK),
        BV=int(BV),
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        has_initial_state=has_initial,
        is_beta_headwise=is_beta_headwise,
        scale=Dk.size**-0.5,
        head_first_layout=head_first,  # pass layout flag to kernel
    )

    n_vtiles = V_pad // BV
    grid = (NH, n_vtiles)

    in_specs, out_specs = (
        _in_specs_head_first(B_, H_, T_, K_pad, BV, is_beta_headwise)
        if head_first
        else _in_specs_bth(B_, H_, T_, K_pad, BV, is_beta_headwise)
    )

    out_pad, final_pad = pl.pallas_call(
        kernel_partial,
        out_shape=(out_struct, final_struct),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=_should_interpret_pallas(),
    )(q_pad, k_pad, v_pad, g_arr, beta_pad, init_kpad, lengths_flat)

    # Trim and wrap
    out_trim = out_pad[:, :, :V_]
    if output_final_state:
        final_trim = final_pad[:, :, :K_, :V_]  # [B,H,K,V]

    out_bhTv = out_trim.reshape(B_, H_, T_, V_)
    out_named = hax.named(out_bhTv, (Batch, Heads, Pos, Dv))
    out_final_named = hax.rearrange(out_named, (Batch, Pos, Heads, Dv))

    final_ret = None if not output_final_state else final_trim.reshape(B_, H_, Dk.size, Dv.size)
    return out_final_named, final_ret


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
                # optional toggles:
                head_first=False,
                lengths=None,
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


def _gdn_chunk_fwd_kernel_fused(
    q_ref,  # [NH, T_pad, K]
    k_ref,  # [NH, T_pad, K]
    v_ref,  # [NH, T_pad, V_pad]
    g_ref,  # [NH, T_pad]
    beta_ref,  # [NH, T_pad]
    s_prev_ref,  # [NH, K, BV]
    ci_ref,  # [1]  int32 (chunk index)
    lengths_ref,  # [NH] int32 (per sequence length)
    out_tile_ref,  # [NH, 1, C, BV]
    s_tile_ref,  # [NH, 1, K, BV]
    a0_tile_ref,  # [NH, 1, C, C]
    *,
    T_pad: int,
    K: int,
    V_pad: int,
    chunk_len: int,
    BK: int,
    BV: int,
):
    nh = pl.program_id(0)
    vt = pl.program_id(1)

    # --- chunk index / active rows
    ci = ci_ref[dslice(0, 1)][0].astype(jnp.int32)
    c0 = ci * jnp.int32(chunk_len)
    Lnh = lengths_ref[dslice(nh, 1)][0].astype(jnp.int32)
    active = jnp.clip(Lnh - c0, 0, jnp.int32(chunk_len))
    ar = lax.iota(jnp.int32, chunk_len)
    m = (ar < active).astype(jnp.float32)  # (C,)

    # --- read chunk views
    q_c = q_ref[dslice(nh, 1), dslice(c0, chunk_len), dslice(0, K)][0].astype(jnp.float32)
    k_c = k_ref[dslice(nh, 1), dslice(c0, chunk_len), dslice(0, K)][0].astype(jnp.float32)
    v_c = v_ref[dslice(nh, 1), dslice(c0, chunk_len), dslice(v0 := vt * jnp.int32(BV), BV)][0].astype(jnp.float32)
    g_c = g_ref[dslice(nh, 1), dslice(c0, chunk_len)][0].astype(jnp.float32)
    b_c = beta_ref[dslice(nh, 1), dslice(c0, chunk_len)][0].astype(jnp.float32)
    S_prev_v = s_prev_ref[dslice(nh, 1), dslice(0, K), dslice(v0, BV)][0].astype(jnp.float32)

    # mask inactive rows
    q_c = q_c * m[:, None]
    k_c = k_c * m[:, None]
    v_c = v_c * m[:, None]
    b_c = b_c * m

    # cumulative g, tail, exp(g)
    g_cum = jnp.cumsum(g_c, axis=0)
    eg = jnp.exp(g_cum)
    g_tail = lax.cond(
        active > 0,
        lambda _: lax.dynamic_slice_in_dim(g_cum, active - 1, 1, axis=0)[0],
        lambda _: jnp.array(0.0, jnp.float32),
        operand=None,
    )
    alpha_tail = jnp.exp(g_tail)

    # ---------- Gram and A0 (strictly lower) ----------
    G = jnp.zeros((chunk_len, chunk_len), dtype=jnp.float32)
    n_kb = (K + BK - 1) // BK

    def gram_body(kb, G_cur):
        k0 = kb * BK
        k_tile = lax.dynamic_slice(k_c, (0, k0), (chunk_len, BK))
        kb_tile = k_tile * b_c[:, None]  # βK
        return G_cur + kb_tile @ jnp.transpose(k_tile)

    G = lax.fori_loop(0, n_kb, gram_body, G)
    A0 = -jnp.tril(G, k=-1)  # (C, C)

    # write A0 once per (NH, chunk) – only vt==0 writes
    def _write_a0(_):
        a0_tile_ref[dslice(nh, 1), dslice(0, 1), dslice(0, chunk_len), dslice(0, chunk_len)] = A0[None, None, :, :]
        return ()

    _ = lax.cond(vt == 0, _write_a0, lambda _: (), operand=None)

    # ---------- forward-sub yv (C×BV) ----------
    yv_tile = jnp.zeros((chunk_len, BV), dtype=jnp.float32)

    def fs_v(i, Y):
        rhs_i = b_c[i] * lax.dynamic_slice(v_c, (i, 0), (1, BV))[0]

        def acc_v(j, acc):
            aij = lax.dynamic_slice(A0, (i, j), (1, 1))[0, 0]
            decay = jnp.exp(g_cum[i] - g_cum[j])
            yj = lax.dynamic_slice(Y, (j, 0), (1, BV))[0]
            return acc + (aij * decay) * yj

        contrib = lax.fori_loop(0, i, acc_v, jnp.zeros((BV,), dtype=jnp.float32))
        yi = rhs_i + contrib
        return lax.dynamic_update_slice(Y, yi[None, :], (i, 0))

    yv_tile = lax.fori_loop(0, active, fs_v, yv_tile)

    # ---------- forward-sub yk (C×K) in BK tiles ----------
    yk_tile = jnp.zeros((chunk_len, K), dtype=jnp.float32)

    def fs_k(kb, Y):
        k0 = kb * BK
        k_t = lax.dynamic_slice(k_c, (0, k0), (chunk_len, BK))
        rhs_tile = (eg[:, None] * (b_c[:, None] * k_t)).astype(jnp.float32)

        def fs_row(i, Ycur):
            rhs_i = lax.dynamic_slice(rhs_tile, (i, 0), (1, BK))[0]

            def acc_k(j, acc):
                aij = lax.dynamic_slice(A0, (i, j), (1, 1))[0, 0]
                decay = jnp.exp(g_cum[i] - g_cum[j])
                yj = lax.dynamic_slice(Ycur, (j, k0), (1, BK))[0]
                return acc + (aij * decay) * yj

            contrib = lax.fori_loop(0, i, acc_k, jnp.zeros((BK,), dtype=jnp.float32))
            yi = rhs_i + contrib
            return lax.dynamic_update_slice(Ycur, yi[None, :], (i, k0))

        return lax.fori_loop(0, active, fs_row, Y)

    yk_tile = lax.fori_loop(0, n_kb, fs_k, yk_tile)

    # ---------- pass 1 accumulators ----------
    v_prime = jnp.zeros((chunk_len, BV), dtype=jnp.float32)
    inter = jnp.zeros((chunk_len, BV), dtype=jnp.float32)
    qk_raw = jnp.zeros((chunk_len, chunk_len), dtype=jnp.float32)

    def pass1_body(kb, acc):
        vpr_acc, inter_acc, qk_acc = acc
        k0 = kb * BK
        q_tile = lax.dynamic_slice(q_c, (0, k0), (chunk_len, BK))
        yk_t = lax.dynamic_slice(yk_tile, (0, k0), (chunk_len, BK))
        S_blk = lax.dynamic_slice(S_prev_v, (k0, 0), (BK, BV))
        vpr_acc = vpr_acc + yk_t @ S_blk
        inter_acc = inter_acc + (q_tile * eg[:, None]) @ S_blk
        qk_acc = qk_acc + q_tile @ jnp.transpose(lax.dynamic_slice(k_c, (0, k0), (chunk_len, BK)))
        return (vpr_acc, inter_acc, qk_acc)

    v_prime, inter, qk_raw = lax.fori_loop(0, n_kb, pass1_body, (v_prime, inter, qk_raw))

    decay = jnp.exp(g_cum[:, None] - g_cum[None, :])
    attn = jnp.tril(decay * qk_raw)

    v_new = yv_tile - v_prime
    out = inter + attn @ v_new

    # ---------- state update ----------
    dw = jnp.exp(g_tail - g_cum) * m
    v_weighted = v_new * dw[:, None]
    S_add = jnp.zeros((K, BV), dtype=jnp.float32)

    def body_kb(kb, S_acc):
        k0 = kb * BK
        k_tile = lax.dynamic_slice(k_c, (0, k0), (chunk_len, BK))
        S_add_tile = jnp.transpose(k_tile) @ v_weighted
        return lax.dynamic_update_slice(S_acc, S_add_tile, (k0, 0))

    S_add = lax.fori_loop(0, n_kb, body_kb, S_add)

    def write_s(kb, _):
        k0 = kb * BK
        S_blk = lax.dynamic_slice(S_prev_v, (k0, 0), (BK, BV))
        S_new_blk = alpha_tail * S_blk + lax.dynamic_slice(S_add, (k0, 0), (BK, BV))
        s_tile_ref[dslice(nh, 1), dslice(0, 1), dslice(k0, BK), dslice(0, BV)] = S_new_blk[None, None, :, :]
        return ()

    _ = lax.fori_loop(0, n_kb, write_s, ())

    # write output tile
    out_tile_ref[dslice(nh, 1), dslice(0, 1), dslice(0, chunk_len), dslice(0, BV)] = out.astype(out_tile_ref.dtype)[
        None, None, :, :
    ]


def _chunk_gated_delta_rule_flash_fused(
    query,
    key,
    value,
    g,
    beta,
    *,
    chunk_size: int = 64,
    initial_state: Optional[jnp.ndarray] = None,  # [B,H,K,V] or [NH,K,V]
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    head_first: bool = False,
    lengths: Optional[jnp.ndarray] = None,
    offsets: Optional[jnp.ndarray] = None,
    use_varlen: bool = False,
    save_for_backward: bool = False,
):
    # ----- (unchanged) layout/flatten/pad/QK-norm -----
    if head_first:
        B, H, L, K = query.shape[:4]
        V = value.shape[-1]
        q_bhlk, k_bhlk, v_bhlv = query, key, value
        g_bhl, b_bhl = g, beta
    else:
        B, L, H, K = query.shape[:4]
        V = value.shape[-1]
        q_bhlk = jnp.transpose(query, (0, 2, 1, 3))
        k_bhlk = jnp.transpose(key, (0, 2, 1, 3))
        v_bhlv = jnp.transpose(value, (0, 2, 1, 3))
        g_bhl = jnp.transpose(g, (0, 2, 1))
        b_bhl = jnp.transpose(beta, (0, 2, 1))

    NH = B * H
    q_flat = q_bhlk.reshape(NH, L, K).astype(jnp.float32)
    k_flat = k_bhlk.reshape(NH, L, K).astype(jnp.float32)
    v_flat = v_bhlv.reshape(NH, L, V).astype(jnp.float32)
    g_flat = g_bhl.reshape(NH, L).astype(jnp.float32)
    b_flat = b_bhl.reshape(NH, L).astype(jnp.float32)

    if use_varlen:
        if lengths is not None:
            lengths_flat = lengths.reshape(-1).astype(jnp.int32)
        elif offsets is not None:
            off = offsets.reshape(-1).astype(jnp.int32)
            lengths_flat = off[1:] - off[:-1] if off.shape[0] == NH + 1 else off
        else:
            raise ValueError("use_varlen=True requires `lengths` or `offsets`.")
        lengths_flat = jnp.clip(lengths_flat, 0, jnp.int32(L))
    else:
        lengths_flat = jnp.full((NH,), L, dtype=jnp.int32)

    if use_qk_l2norm_in_kernel:
        eps = 1e-6
        qn = jnp.sqrt(jnp.sum(q_flat * q_flat, axis=-1, keepdims=True) + eps)
        kn = jnp.sqrt(jnp.sum(k_flat * k_flat, axis=-1, keepdims=True) + eps)
        q_flat = (q_flat / jnp.where(qn > 0.0, qn, 1.0)) * (K**-0.5)
        k_flat = k_flat / jnp.where(kn > 0.0, kn, 1.0)
    else:
        q_flat = q_flat * (K**-0.5)

    C = int(chunk_size)
    Nc = (L + C - 1) // C
    T_pad = Nc * C

    def _pad_last(x, w):
        pad = T_pad - w
        if pad == 0:
            return x
        return jnp.pad(x, ((0, 0), (0, pad))) if x.ndim == 2 else jnp.pad(x, ((0, 0), (0, pad), (0, 0)))

    q_flat = _pad_last(q_flat, L)
    k_flat = _pad_last(k_flat, L)
    v_flat = _pad_last(v_flat, L)
    g_flat = _pad_last(g_flat, L)
    b_flat = _pad_last(b_flat, L)

    BK = min(64 if K >= 128 else 32, K)
    BV = min(64 if V >= 128 else 32, V)
    V_pad = ((V + BV - 1) // BV) * BV
    if V_pad != V:
        v_flat = jnp.pad(v_flat, ((0, 0), (0, 0), (0, V_pad - V)))

    # Initial S (NH,K,V_pad)
    if initial_state is None:
        S = jnp.zeros((NH, K, V_pad), dtype=jnp.float32)
    else:
        init = initial_state.astype(jnp.float32)
        S = init.reshape(NH, K, init.shape[-1]) if init.ndim == 4 else init
        if S.shape[-1] != V_pad:
            S = jnp.pad(S, ((0, 0), (0, 0), (0, V_pad - S.shape[-1])))

    out_pad = jnp.zeros((NH, T_pad, V_pad), dtype=value.dtype)
    n_vtiles = V_pad // BV

    # Buffers for backward
    A0_buf = jnp.zeros((NH, Nc, C, C), dtype=jnp.float32) if save_for_backward else None
    S_prev_buf = jnp.zeros((NH, Nc, K, V_pad), dtype=jnp.float32) if save_for_backward else None

    kernel = functools.partial(
        _gdn_chunk_fwd_kernel_fused,
        T_pad=T_pad,
        K=int(K),
        V_pad=int(V_pad),
        chunk_len=int(C),
        BK=int(BK),
        BV=int(BV),
    )

    in_specs = (
        pl.BlockSpec((1, T_pad, K), lambda nh, vt: (nh, 0, 0)),  # q
        pl.BlockSpec((1, T_pad, K), lambda nh, vt: (nh, 0, 0)),  # k
        pl.BlockSpec((1, T_pad, V_pad), lambda nh, vt: (nh, 0, 0)),  # v
        pl.BlockSpec((1, T_pad), lambda nh, vt: (nh, 0)),  # g
        pl.BlockSpec((1, T_pad), lambda nh, vt: (nh, 0)),  # beta
        pl.BlockSpec((1, K, BV), lambda nh, vt: (nh, 0, vt * BV)),  # S_prev tile
        pl.BlockSpec((1,), lambda nh, vt: (0,)),  # ci (scalar)
        pl.BlockSpec((1,), lambda nh, vt: (nh,)),  # lengths[nh]
    )
    out_specs = (
        pl.BlockSpec((1, 1, C, BV), lambda nh, vt: (nh, vt, 0, 0)),  # out tile
        pl.BlockSpec((1, 1, K, BV), lambda nh, vt: (nh, vt, 0, 0)),  # S tile
        pl.BlockSpec((1, 1, C, C), lambda nh, vt: (nh, 0, 0, 0)),  # A0 tile (vt ignored; only vt==0 writes)
    )

    def one_chunk(carry, ci):
        S_in, out_buf, A0_b, Sprev_b = carry
        ci_ary = jnp.asarray([ci], dtype=jnp.int32)

        if save_for_backward:
            # Save S_prev (whole V_pad) for this chunk (outside kernel; no tracers in index maps)
            Sprev_b = Sprev_b.at[:, ci, :, :].set(S_in)

        with _fp32_mm():
            out_tiles, s_tiles, a0_tiles = pl.pallas_call(
                kernel,
                out_shape=(
                    jax.ShapeDtypeStruct((NH, n_vtiles, C, BV), value.dtype),
                    jax.ShapeDtypeStruct((NH, n_vtiles, K, BV), jnp.float32),
                    jax.ShapeDtypeStruct((NH, n_vtiles, C, C), jnp.float32),
                ),
                grid=(NH, n_vtiles),
                in_specs=in_specs,
                out_specs=out_specs,
                interpret=_should_interpret_pallas(),
            )(q_flat, k_flat, v_flat, g_flat, b_flat, S_in, ci_ary, lengths_flat)

        out_chunk = out_tiles.reshape(NH, C, n_vtiles * BV)[:, :, :V_pad]
        s_chunk = s_tiles.reshape(NH, K, n_vtiles * BV)[:, :, :V_pad]

        if save_for_backward:
            # Store A0 for this chunk (written only in vt==0)
            A0_b = A0_b.at[:, ci, :, :].set(a0_tiles[:, 0, :, :])

        c0 = ci * C
        out_buf = lax.dynamic_update_slice(out_buf, out_chunk.astype(out_buf.dtype), (0, c0, 0))
        S_out = s_chunk
        return (S_out, out_buf, A0_b, Sprev_b), None

    (S_final, out_all, A0_buf, S_prev_buf), _ = lax.scan(
        one_chunk,
        (S, out_pad, A0_buf, S_prev_buf),
        jnp.arange(Nc, dtype=jnp.int32),
    )

    out_trim = out_all[:, :L, :V]
    out_bHLv = out_trim.reshape(B, H, L, V)
    out_named = out_bHLv if head_first else jnp.transpose(out_bHLv, (0, 2, 1, 3))
    S_ret = None if not output_final_state else S_final[:, :, :V].reshape(B, H, K, V)

    # Return aux for backward
    aux = (
        {
            "A0_buf": A0_buf,
            "S_prev_buf": S_prev_buf,
            "T_pad": T_pad,
            "K": K,
            "V_pad": V_pad,
            "C": C,
            "NH": NH,
            "Nc": Nc,
        }
        if save_for_backward
        else None
    )

    return out_named, S_ret, aux


def _gdn_chunk_bwd_kernel_fused(
    q_chunk_ref,  # [NH, C, K]
    k_chunk_ref,  # [NH, C, K]
    v_tile_ref,  # [NH, C, BV]
    g_chunk_ref,  # [NH, C]
    beta_chunk_ref,  # [NH, C]
    a0_chunk_ref,  # [NH, C, C]
    s_prev_tile_ref,  # [NH, K, BV]
    dout_tile_ref,  # [NH, C, BV]
    dS_in_tile_ref,  # [NH, K, BV]
    lengths_chunk_ref,  # [NH]
    dQ_chunk_ref,  # [NH, 1, C, K]
    dK_chunk_ref,  # [NH, 1, C, K]
    dV_tile_ref,  # [NH, 1, C, BV]
    dG_chunk_ref,  # [NH, 1, C]
    dB_chunk_ref,  # [NH, 1, C]
    dS_out_tile_ref,  # [NH, K, BV]
    *,
    C: int,
    K: int,
    BV: int,
):
    nh = pl.program_id(0)

    # ---- local views (all dynamic slicing, static sizes) ----
    q_c = q_chunk_ref[dslice(nh, 1), dslice(0, C), dslice(0, K)][0].astype(jnp.float32)
    k_c = k_chunk_ref[dslice(nh, 1), dslice(0, C), dslice(0, K)][0].astype(jnp.float32)
    v_c = v_tile_ref[dslice(nh, 1), dslice(0, C), dslice(0, BV)][0].astype(jnp.float32)
    g_c = g_chunk_ref[dslice(nh, 1), dslice(0, C)][0].astype(jnp.float32)
    b_c = beta_chunk_ref[dslice(nh, 1), dslice(0, C)][0].astype(jnp.float32)
    A0 = a0_chunk_ref[dslice(nh, 1), dslice(0, C), dslice(0, C)][0].astype(jnp.float32)
    S_prev = s_prev_tile_ref[dslice(nh, 1), dslice(0, K), dslice(0, BV)][0].astype(jnp.float32)
    dY = dout_tile_ref[dslice(nh, 1), dslice(0, C), dslice(0, BV)][0].astype(jnp.float32)
    dS_in = dS_in_tile_ref[dslice(nh, 1), dslice(0, K), dslice(0, BV)][0].astype(jnp.float32)
    active = jnp.clip(lengths_chunk_ref[dslice(nh, 1)][0].astype(jnp.int32), 0, jnp.int32(C))

    # row mask for varlen chunk
    m = (lax.iota(jnp.int32, C) < active).astype(jnp.float32)
    q_c = q_c * m[:, None]
    k_c = k_c * m[:, None]
    v_c = v_c * m[:, None]
    b_c = b_c * m

    g_cum = jnp.cumsum(g_c, axis=0)
    eg = jnp.exp(g_cum)
    g_tail = lax.cond(
        active > 0,
        lambda _: lax.dynamic_slice_in_dim(g_cum, active - 1, 1, axis=0)[0],
        lambda _: jnp.array(0.0, jnp.float32),
        operand=None,
    )
    alpha_tail = jnp.exp(g_tail)

    # ---------- Re-solve yv, yk using saved A0 (forward-sub, bounded by active) ----------
    def fs_yv():
        Y = jnp.zeros((C, BV), jnp.float32)

        def row(i, Ycur):
            rhs_i = b_c[i] * lax.dynamic_slice(v_c, (i, 0), (1, BV))[0]

            def acc_j(j, acc):
                aij = lax.dynamic_slice(A0, (i, j), (1, 1))[0, 0]
                decay = jnp.exp(g_cum[i] - g_cum[j])
                yj = lax.dynamic_slice(Ycur, (j, 0), (1, BV))[0]
                return acc + (aij * decay) * yj

            contrib = lax.fori_loop(0, i, acc_j, jnp.zeros((BV,), jnp.float32))
            yi = rhs_i + contrib
            return lax.dynamic_update_slice(Ycur, yi[None, :], (i, 0))

        return lax.fori_loop(0, active, row, Y)

    yv = fs_yv()

    def fs_yk():
        Y = jnp.zeros((C, K), jnp.float32)

        def row(i, Ycur):
            rhs_i = (eg[i] * b_c[i]) * lax.dynamic_slice(k_c, (i, 0), (1, K))[0]

            def acc_j(j, acc):
                aij = lax.dynamic_slice(A0, (i, j), (1, 1))[0, 0]
                decay = jnp.exp(g_cum[i] - g_cum[j])
                yj = lax.dynamic_slice(Ycur, (j, 0), (1, K))[0]
                return acc + (aij * decay) * yj

            contrib = lax.fori_loop(0, i, acc_j, jnp.zeros((K,), jnp.float32))
            yi = rhs_i + contrib
            return lax.dynamic_update_slice(Ycur, yi[None, :], (i, 0))

        return lax.fori_loop(0, active, row, Y)

    yk = fs_yk()

    v_prime = yk @ S_prev
    v_new = yv - v_prime

    qk = q_c @ jnp.transpose(k_c)
    decay = jnp.exp(g_cum[:, None] - g_cum[None, :])
    tril_mask = jnp.tril(jnp.ones((C, C), jnp.float32))
    M = decay * tril_mask
    attn = M * qk
    # inter = (q_c * eg[:, None]) @ S_prev

    # ---------- Backprop ----------
    # out = inter + attn @ v_new
    dQ_inter = (dY @ jnp.transpose(S_prev)) * eg[:, None]
    dS_prev_local = jnp.transpose(q_c * eg[:, None]) @ dY
    inner = jnp.sum((dY @ jnp.transpose(S_prev)) * q_c, axis=-1)
    dgc_inter = inner * eg

    dAttn = dY @ jnp.transpose(v_new)
    dv_new = jnp.transpose(attn) @ dY
    d_qk = M * dAttn
    dQ_attn = d_qk @ k_c
    dK_attn = jnp.transpose(d_qk) @ q_c
    W = (qk * dAttn) * tril_mask
    dgc_attn = jnp.sum(W, axis=1) - jnp.sum(W, axis=0)

    # v_new = yv - yk @ S_prev
    dyv = dv_new
    dyk = -(dv_new @ jnp.transpose(S_prev))
    dS_prev_local = dS_prev_local - jnp.transpose(yk) @ dv_new

    # cross-chunk carry: S_out = alpha_tail * S_prev + sum_i (exp(g_tail - g_cum[i]) k_i v_new[i]^T)
    w = jnp.exp(g_tail - g_cum) * m  # (C,)
    kv_dS = k_c @ dS_in  # (C, BV)
    dv_new = dv_new + kv_dS * w[:, None]  # (C, BV)

    # dK from carry: for each row i, dK_i += w_i * (dS_in @ v_new_i)
    dK_add = ((dS_in @ jnp.transpose(v_new)).T) * w[:, None]  # (C, K)

    dgt_from_alpha = alpha_tail * jnp.sum(S_prev * dS_in)
    s_i = jnp.sum(kv_dS * v_new, axis=-1)  # (C,)
    dgt_from_w = jnp.sum(w * s_i)
    dgc_from_w = -w * s_i

    # SAFE update at tail index (active-1) without Python if:
    def _add_tail(_):
        idx = active - jnp.int32(1)
        base = dgc_from_w
        cur = lax.dynamic_slice_in_dim(base, idx, 1, axis=0)[0]
        upd = cur + (dgt_from_alpha + dgt_from_w)
        return lax.dynamic_update_slice_in_dim(base, upd[None], idx, axis=0)

    dgc_carry = lax.cond(active > 0, _add_tail, lambda _: dgc_from_w, operand=None)
    dS_prev_carry = alpha_tail * dS_in

    # backward triangular solved contributions (Ā) for yv/yk
    def bwd_sub_yv():
        d_rhs = jnp.zeros((C, BV), jnp.float32)
        dAbar = jnp.zeros((C, C), jnp.float32)

        def row_rev(i_rev, state):
            d_rhs_cur, dA = state
            i = active - 1 - i_rev
            dyi = lax.dynamic_slice(dyv, (i, 0), (1, BV))[0]
            d_rhs_cur = lax.dynamic_update_slice(d_rhs_cur, dyi[None, :], (i, 0))

            def upd_j(j, carry):
                dA_cur, d_rhs_inner = carry
                aij0 = lax.dynamic_slice(A0, (i, j), (1, 1))[0, 0]
                decay = jnp.exp(g_cum[i] - g_cum[j])
                yj = lax.dynamic_slice(yv, (j, 0), (1, BV))[0]
                # dAbar[i,j] += <dyi, yj>
                val = lax.dynamic_slice(dA_cur, (i, j), (1, 1))[0, 0] + jnp.sum(dyi * yj)
                dA_cur = lax.dynamic_update_slice(dA_cur, jnp.asarray([[val]], jnp.float32), (i, j))
                d_rhs_inner = lax.dynamic_update_slice(
                    d_rhs_inner,
                    (lax.dynamic_slice(d_rhs_inner, (j, 0), (1, BV))[0] + (aij0 * decay) * dyi)[None, :],
                    (j, 0),
                )
                return (dA_cur, d_rhs_inner)

            dA, d_rhs_cur = lax.fori_loop(0, i, upd_j, (dA, d_rhs_cur))
            return (d_rhs_cur, dA)

        return lax.fori_loop(0, active, row_rev, (d_rhs, dAbar))

    d_rhs_v, dAbar_v = bwd_sub_yv()

    def bwd_sub_yk():
        d_rhs = jnp.zeros((C, K), jnp.float32)
        dAbar = jnp.zeros((C, C), jnp.float32)

        def row_rev(i_rev, state):
            d_rhs_cur, dA = state
            i = active - 1 - i_rev
            dyi = lax.dynamic_slice(dyk, (i, 0), (1, K))[0]
            d_rhs_cur = lax.dynamic_update_slice(d_rhs_cur, dyi[None, :], (i, 0))

            def upd_j(j, carry):
                dA_cur, d_rhs_inner = carry
                aij0 = lax.dynamic_slice(A0, (i, j), (1, 1))[0, 0]
                decay = jnp.exp(g_cum[i] - g_cum[j])
                yj = lax.dynamic_slice(yk, (j, 0), (1, K))[0]
                val = lax.dynamic_slice(dA_cur, (i, j), (1, 1))[0, 0] + jnp.sum(dyi * yj)
                dA_cur = lax.dynamic_update_slice(dA_cur, jnp.asarray([[val]], jnp.float32), (i, j))
                d_rhs_inner = lax.dynamic_update_slice(
                    d_rhs_inner,
                    (lax.dynamic_slice(d_rhs_inner, (j, 0), (1, K))[0] + (aij0 * decay) * dyi)[None, :],
                    (j, 0),
                )
                return (dA_cur, d_rhs_inner)

            dA, d_rhs_cur = lax.fori_loop(0, i, upd_j, (dA, d_rhs_cur))
            return (d_rhs_cur, dA)

        return lax.fori_loop(0, active, row_rev, (d_rhs, dAbar))

    d_rhs_k, dAbar_k = bwd_sub_yk()

    dAbar = dAbar_v + dAbar_k

    # map dAbar → dA0 and d g_cum
    dA0 = jnp.zeros_like(A0)
    dgc_from_A = jnp.zeros((C,), jnp.float32)

    def tri_map(i, carry):
        dA0_cur, dgc_cur = carry

        def col(j, inner):
            dA0_i, dgc_i = inner
            aij0 = lax.dynamic_slice(A0, (i, j), (1, 1))[0, 0]
            decay = jnp.exp(g_cum[i] - g_cum[j])
            dAij = lax.dynamic_slice(dAbar, (i, j), (1, 1))[0, 0]
            # dA0 += dAbar * decay
            val = lax.dynamic_slice(dA0_i, (i, j), (1, 1))[0, 0] + dAij * decay
            dA0_i = lax.dynamic_update_slice(dA0_i, jnp.asarray([[val]], jnp.float32), (i, j))
            # g_cum: + on i, - on j
            dgc_i = dgc_i.at[i].add(dAij * aij0 * decay)
            dgc_i = dgc_i.at[j].add(-dAij * aij0 * decay)
            return (dA0_i, dgc_i)

        return lax.fori_loop(0, i, col, (dA0_cur, dgc_cur))

    dA0, dgc_from_A = lax.fori_loop(1, active, tri_map, (dA0, dgc_from_A))

    # RHS contributions
    dV_tile = b_c[:, None] * d_rhs_v
    dB_from_v = jnp.sum(v_c * d_rhs_v, axis=-1)
    dK_rhs = (eg[:, None] * b_c[:, None]) * d_rhs_k
    dB_from_k = jnp.sum((eg[:, None] * k_c) * d_rhs_k, axis=-1)
    dgc_from_rhs_k = eg * jnp.sum((b_c[:, None] * k_c) * d_rhs_k, axis=-1)

    # dA0 → dK and dβ, via A0[i,j] = -β_i <k_i, k_j>
    dK_from_A0 = jnp.zeros_like(k_c)
    dB_from_A0 = jnp.zeros((C,), jnp.float32)

    def gram_bwd(i, state):
        dK_acc, dB_acc = state
        ki = lax.dynamic_slice(k_c, (i, 0), (1, K))[0]

        def body(j, carry):
            dK_a, dB_a = carry
            kj = lax.dynamic_slice(k_c, (j, 0), (1, K))[0]
            coeff = -(dA0[i, j] * b_c[i])
            dK_a = dK_a.at[i, :].add(coeff * kj)
            dK_a = dK_a.at[j, :].add(coeff * ki)
            dB_a = dB_a + (-dA0[i, j]) * jnp.sum(ki * kj)
            return (dK_a, dB_a)

        return lax.fori_loop(0, i, body, (dK_acc, dB_acc))

    dK_from_A0, dB_from_A0 = lax.fori_loop(1, active, gram_bwd, (dK_from_A0, dB_from_A0))

    # collect per-chunk grads
    dQ_tile = dQ_inter + dQ_attn
    dK_tile = dK_attn + dK_add + dK_rhs + dK_from_A0
    dB_tile = (dB_from_v + dB_from_k + dB_from_A0) * m
    dgc_total = (dgc_inter + dgc_attn + dgc_from_A + dgc_from_rhs_k + dgc_carry) * m

    # reverse-cumsum d g_cum → d g, avoiding dynamic jnp.pad
    def rev_cumsum_full(x):
        # returns length-C vector; only positions < active are written
        def body(i_rev, carry):
            acc, out_vec = carry
            i = active - jnp.int32(1) - i_rev
            xi = lax.dynamic_slice_in_dim(x, i, 1, axis=0)[0]
            acc = acc + xi
            out_vec = lax.dynamic_update_slice_in_dim(out_vec, acc[None], i, axis=0)
            return (acc, out_vec)

        init = (jnp.array(0.0, jnp.float32), jnp.zeros((C,), jnp.float32))
        _, out_vec = lax.fori_loop(0, active, body, init)
        return out_vec

    dG_tile = rev_cumsum_full(dgc_total)

    dS_out = dS_prev_local + dS_prev_carry

    # writes
    dQ_chunk_ref[dslice(nh, 1), dslice(0, 1), dslice(0, C), dslice(0, K)] = dQ_tile[None, None, :, :]
    dK_chunk_ref[dslice(nh, 1), dslice(0, 1), dslice(0, C), dslice(0, K)] = dK_tile[None, None, :, :]
    dV_tile_ref[dslice(nh, 1), dslice(0, 1), dslice(0, BV)] = dV_tile[None, None, :, :]
    dG_chunk_ref[dslice(nh, 1), dslice(0, 1)] = dG_tile[None, None, :]
    dB_chunk_ref[dslice(nh, 1), dslice(0, 1)] = dB_tile[None, None, :]
    dS_out_tile_ref[dslice(nh, 1), dslice(0, K), dslice(0, BV)] = dS_out[None, :, :]


def _chunk_gated_delta_rule_flash_bwd_fused(
    q_arr,
    k_arr,
    v_arr,
    g_arr,
    beta_arr,
    d_out_arr,
    *,
    head_first: bool,
    lengths,
    aux,
    initial_state_was_none: bool,
    use_qk_l2norm_in_kernel: bool,
):
    A0_buf = aux["A0_buf"]
    S_prev_buf = aux["S_prev_buf"]
    T_pad = int(aux["T_pad"])  # padded time length used in fwd
    K = int(aux["K"])
    V_pad = int(aux["V_pad"])  # padded value dim used in fwd
    C = int(aux["C"])
    NH = int(aux["NH"])
    Nc = int(aux["Nc"])

    # ---------- helpers: pad along time and value ----------
    def _pad_time(x, t_pad):
        """Pad axis=1 (time) of an NH-major array to t_pad."""
        t = x.shape[1]
        if t == t_pad:
            return x
        return jnp.pad(x, ((0, 0), (0, t_pad - t)) + ((0, 0),) * (x.ndim - 2))

    def _pad_time_and_feat(x, t_pad, v_pad):
        """Pad axis=1 (time) to t_pad and last axis (value) to v_pad."""
        t, v = x.shape[1], x.shape[-1]
        if t != t_pad:
            x = jnp.pad(x, ((0, 0), (0, t_pad - t)) + ((0, 0),) * (x.ndim - 2))
        if v != v_pad:
            x = jnp.pad(x, ((0, 0), (0, 0)) + ((0, v_pad - v),))
        return x

    # ---------- layout-normalize inputs to NH-major and pad to T_pad/V_pad ----------
    if head_first:
        # q,k: [B,H,L,K]  → [NH, L, K]
        B, H, L, _K = q_arr.shape
        _ = _K  # silence lints
        q_LK = jnp.transpose(q_arr, (0, 2, 1, 3)).reshape(NH, L, K)
        k_LK = jnp.transpose(k_arr, (0, 2, 1, 3)).reshape(NH, L, K)
        # v: [B,H,L,V]   → [NH, L, V]
        V = v_arr.shape[-1]
        v_LV = jnp.transpose(v_arr, (0, 2, 1, 3)).reshape(NH, L, V)
        # g,b: [B,H,L]   → [NH, L]
        g_L = jnp.transpose(g_arr, (0, 2, 1)).reshape(NH, L)
        b_L = jnp.transpose(beta_arr, (0, 2, 1)).reshape(NH, L)
        # dY: [B,H,L,V]  → [NH, L, V]
        dY_LV = jnp.transpose(d_out_arr, (0, 2, 1, 3)).reshape(NH, L, V)
    else:
        # q,k: [B,L,H,K] → [NH, L, K]
        B, L, H, _K = q_arr.shape
        _ = _K
        q_LK = q_arr.reshape(NH, L, K)
        k_LK = k_arr.reshape(NH, L, K)
        # v: [B,L,H,V]   → [NH, L, V]
        V = v_arr.shape[-1]
        v_LV = v_arr.reshape(NH, L, V)
        # g,b: [B,L,H]   → [NH, L]
        g_L = g_arr.reshape(NH, L)
        b_L = beta_arr.reshape(NH, L)
        # dY: [B,L,H,V]  → [NH, L, V]
        dY_LV = d_out_arr.reshape(NH, L, V)

    # === PAD to T_pad / V_pad so chunk slicing works for the last chunk ===
    q_flat = _pad_time(q_LK, T_pad).astype(jnp.float32)  # [NH, T_pad, K]
    k_flat = _pad_time(k_LK, T_pad).astype(jnp.float32)  # [NH, T_pad, K]
    g_flat = _pad_time(g_L, T_pad).astype(jnp.float32)  # [NH, T_pad]
    b_flat = _pad_time(b_L, T_pad).astype(jnp.float32)  # [NH, T_pad]
    v_flat = _pad_time_and_feat(v_LV, T_pad, V_pad).astype(jnp.float32)  # [NH, T_pad, V_pad]
    dY_full = _pad_time_and_feat(dY_LV, T_pad, V_pad).astype(jnp.float32)  # [NH, T_pad, V_pad]

    # ---------- init grads ----------
    dQ_flat = jnp.zeros_like(q_flat, dtype=jnp.float32)
    dK_flat = jnp.zeros_like(k_flat, dtype=jnp.float32)
    dV_flat = jnp.zeros_like(v_flat, dtype=jnp.float32)
    dG_flat = jnp.zeros_like(g_flat, dtype=jnp.float32)
    dB_flat = jnp.zeros_like(b_flat, dtype=jnp.float32)

    # lengths per NH
    if lengths is None:
        lengths_flat = jnp.full((NH,), T_pad, dtype=jnp.int32)
    else:
        lf = lengths.astype(jnp.int32)
        lf = lf.reshape(-1) if lf.ndim == 2 else lf
        lengths_flat = jnp.clip(lf, 0, jnp.int32(T_pad))

    BV = min(64 if V_pad >= 128 else 32, V_pad)
    n_vtiles = V_pad // BV

    # reverse carry dS (NH, K, V_pad)
    dS_carry = jnp.zeros((NH, K, V_pad), dtype=jnp.float32)

    kernel = functools.partial(
        _gdn_chunk_bwd_kernel_fused,
        C=int(C),
        K=int(K),
        BV=int(BV),
    )

    # ---------- reverse over chunks ----------
    def one_chunk(ci, dS_in):
        c0 = ci * C
        lengths_chunk = jnp.clip(lengths_flat - jnp.int32(c0), 0, jnp.int32(C))  # [NH]

        # slice per-chunk tensors OUTSIDE the kernel (static sizes inside)
        q_chunk = jax.lax.dynamic_slice(q_flat, (0, c0, 0), (NH, C, K))
        k_chunk = jax.lax.dynamic_slice(k_flat, (0, c0, 0), (NH, C, K))
        g_chunk = jax.lax.dynamic_slice(g_flat, (0, c0), (NH, C))
        b_chunk = jax.lax.dynamic_slice(b_flat, (0, c0), (NH, C))
        A0_chunk = jax.lax.dynamic_slice(A0_buf, (0, ci, 0, 0), (NH, 1, C, C)).squeeze(1)
        dY_chunk = jax.lax.dynamic_slice(dY_full, (0, c0, 0), (NH, C, V_pad))
        S_prev_chunk = jax.lax.dynamic_slice(S_prev_buf, (0, ci, 0, 0), (NH, 1, K, V_pad)).squeeze(1)

        # per-chunk accumulators
        dQ_c = jnp.zeros((NH, C, K), jnp.float32)
        dK_c = jnp.zeros((NH, C, K), jnp.float32)
        dV_c = jnp.zeros((NH, C, V_pad), jnp.float32)
        dG_c = jnp.zeros((NH, C), jnp.float32)
        dB_c = jnp.zeros((NH, C), jnp.float32)

        def one_vt(vt, acc):
            dQ_acc, dK_acc, dV_acc, dG_acc, dB_acc, dS_in_prev = acc

            v_tile = jax.lax.dynamic_slice(v_flat, (0, c0, vt * BV), (NH, C, BV))
            dY_tile = jax.lax.dynamic_slice(dY_chunk, (0, 0, vt * BV), (NH, C, BV))
            S_prev_t = jax.lax.dynamic_slice(S_prev_chunk, (0, 0, vt * BV), (NH, K, BV))
            dS_in_t = jax.lax.dynamic_slice(dS_in, (0, 0, vt * BV), (NH, K, BV))

            dQ_t, dK_t, dV_t, dG_t, dB_t, dS_out_t = pl.pallas_call(
                kernel,
                out_shape=(
                    jax.ShapeDtypeStruct((NH, 1, C, K), jnp.float32),
                    jax.ShapeDtypeStruct((NH, 1, C, K), jnp.float32),
                    jax.ShapeDtypeStruct((NH, 1, C, BV), jnp.float32),
                    jax.ShapeDtypeStruct((NH, 1, C), jnp.float32),
                    jax.ShapeDtypeStruct((NH, 1, C), jnp.float32),
                    jax.ShapeDtypeStruct((NH, K, BV), jnp.float32),
                ),
                grid=(NH,),
                in_specs=(
                    pl.BlockSpec((1, C, K), lambda nh: (nh, 0, 0)),  # q_chunk
                    pl.BlockSpec((1, C, K), lambda nh: (nh, 0, 0)),  # k_chunk
                    pl.BlockSpec((1, C, BV), lambda nh: (nh, 0, 0)),  # v_tile
                    pl.BlockSpec((1, C), lambda nh: (nh, 0)),  # g_chunk
                    pl.BlockSpec((1, C), lambda nh: (nh, 0)),  # b_chunk
                    pl.BlockSpec((1, C, C), lambda nh: (nh, 0, 0)),  # A0_chunk
                    pl.BlockSpec((1, K, BV), lambda nh: (nh, 0, 0)),  # S_prev_tile
                    pl.BlockSpec((1, C, BV), lambda nh: (nh, 0, 0)),  # dY_tile
                    pl.BlockSpec((1, K, BV), lambda nh: (nh, 0, 0)),  # dS_in_tile
                    pl.BlockSpec((1,), lambda nh: (nh,)),  # lengths_chunk
                ),
                out_specs=(
                    pl.BlockSpec((1, 1, C, K), lambda nh: (nh, 0, 0, 0)),
                    pl.BlockSpec((1, 1, C, K), lambda nh: (nh, 0, 0, 0)),
                    pl.BlockSpec((1, 1, C, BV), lambda nh: (nh, 0, 0, 0)),
                    pl.BlockSpec((1, 1, C), lambda nh: (nh, 0, 0)),
                    pl.BlockSpec((1, 1, C), lambda nh: (nh, 0, 0)),
                    pl.BlockSpec((1, K, BV), lambda nh: (nh, 0, 0)),
                ),
                interpret=_should_interpret_pallas(),
            )(q_chunk, k_chunk, v_tile, g_chunk, b_chunk, A0_chunk, S_prev_t, dY_tile, dS_in_t, lengths_chunk)

            dQ_acc = dQ_acc + dQ_t[:, 0, :, :]
            dK_acc = dK_acc + dK_t[:, 0, :, :]

            start = vt * jnp.int32(BV)

            dv_old = lax.dynamic_slice(dV_acc, (0, 0, start), (NH, C, BV))
            dv_new = dv_old + dV_t[:, 0, :, :]
            dV_acc = lax.dynamic_update_slice(dV_acc, dv_new, (0, 0, start))

            dG_acc = dG_acc + dG_t[:, 0, :]
            dB_acc = dB_acc + dB_t[:, 0, :]
            dS_in_prev = lax.dynamic_update_slice(dS_in_prev, dS_out_t, (0, 0, start))
            return (dQ_acc, dK_acc, dV_acc, dG_acc, dB_acc, dS_in_prev)

        dQ_c, dK_c, dV_c, dG_c, dB_c, dS_out = lax.fori_loop(
            0, n_vtiles, one_vt, (dQ_c, dK_c, dV_c, dG_c, dB_c, jnp.zeros_like(dS_in))
        )

        # scatter per-chunk grads
        old = lax.dynamic_slice(dQ_flat, (0, c0, 0), (NH, C, K))
        dQ_sc = lax.dynamic_update_slice(dQ_flat, old + dQ_c, (0, c0, 0))

        old = lax.dynamic_slice(dK_flat, (0, c0, 0), (NH, C, K))
        dK_sc = lax.dynamic_update_slice(dK_flat, old + dK_c, (0, c0, 0))

        old = lax.dynamic_slice(dV_flat, (0, c0, 0), (NH, C, V_pad))
        dV_sc = lax.dynamic_update_slice(dV_flat, old + dV_c, (0, c0, 0))

        old = lax.dynamic_slice(dG_flat, (0, c0), (NH, C))
        dG_sc = lax.dynamic_update_slice(dG_flat, old + dG_c, (0, c0))

        old = lax.dynamic_slice(dB_flat, (0, c0), (NH, C))
        dB_sc = lax.dynamic_update_slice(dB_flat, old + dB_c, (0, c0))
        return (dQ_sc, dK_sc, dV_sc, dG_sc, dB_sc, dS_out)

    # fold over chunks in reverse: Nc-1 .. 0
    def rev_chunks(carry, i):
        dQ_c, dK_c, dV_c, dG_c, dB_c, dS_in = carry
        ci = jnp.int32(Nc - 1 - i)
        dQ_c, dK_c, dV_c, dG_c, dB_c, dS_in = one_chunk(ci, dS_in)
        return (dQ_c, dK_c, dV_c, dG_c, dB_c, dS_in), None

    (dQ_flat, dK_flat, dV_flat, dG_flat, dB_flat, dS0), _ = lax.scan(
        rev_chunks,
        (dQ_flat, dK_flat, dV_flat, dG_flat, dB_flat, dS_carry),
        jnp.arange(Nc, dtype=jnp.int32),
    )

    # ---------- pack back to original layout & trim to L,V ----------
    # Reconstruct NH-major raw q,k (unpadded), then pad to T_pad to match dQ_flat/dK_flat
    if head_first:
        B, H = v_arr.shape[0], v_arr.shape[1] if v_arr.ndim == 4 else (q_arr.shape[0], k_arr.shape[2])
        # From earlier layout branch, we already have B, L, H, K above
        q_LK_raw = (jnp.transpose(q_arr, (0, 2, 1, 3))).reshape(aux["NH"], -1, aux["K"]).astype(jnp.float32)
        k_LK_raw = (jnp.transpose(k_arr, (0, 2, 1, 3))).reshape(aux["NH"], -1, aux["K"]).astype(jnp.float32)
    else:
        # q_arr: [B, L, H, K] -> [NH, L, K]
        NH = aux["NH"]
        K = aux["K"]
        L = q_arr.shape[1]
        q_LK_raw = q_arr.reshape(NH, L, K).astype(jnp.float32)
        k_LK_raw = k_arr.reshape(NH, L, K).astype(jnp.float32)

    q_raw_flat = _pad_time(q_LK_raw, aux["T_pad"])
    k_raw_flat = _pad_time(k_LK_raw, aux["T_pad"])

    if use_qk_l2norm_in_kernel:
        eps = jnp.asarray(1e-6, jnp.float32)

        def _l2norm_bwd(dY, X, scale):
            # d/ dX ( (X / ||X||) * scale ) applied to row-vectors along last dim
            inv = lax.rsqrt(jnp.sum(X * X, axis=-1, keepdims=True) + eps)
            dot = jnp.sum(dY * X, axis=-1, keepdims=True)
            return scale * (inv * dY + (-(inv**3.0)) * X * dot)

        scale_q = jnp.asarray(aux["K"] ** -0.5, jnp.float32)
        dQ_flat_raw = _l2norm_bwd(dQ_flat, q_raw_flat, scale_q)
        dK_flat_raw = _l2norm_bwd(dK_flat, k_raw_flat, jnp.asarray(1.0, jnp.float32))
    else:
        # Only the 1/sqrt(K) scaling was applied to q in forward
        scale_q = jnp.asarray(aux["K"] ** -0.5, jnp.float32)
        dQ_flat_raw = dQ_flat * scale_q
        dK_flat_raw = dK_flat

    # From here on, use *raw* grads
    dQ_flat = dQ_flat_raw
    dK_flat = dK_flat_raw

    if head_first:
        dq = jnp.transpose(dQ_flat.reshape(B, H, T_pad, K)[:, :, :L, :], (0, 2, 1, 3))
        dk = jnp.transpose(dK_flat.reshape(B, H, T_pad, K)[:, :, :L, :], (0, 2, 1, 3))
        dv = jnp.transpose(dV_flat[:, :L, : v_arr.shape[-1]].reshape(B, H, L, v_arr.shape[-1]), (0, 2, 1, 3))
        dg = jnp.transpose(dG_flat[:, :L].reshape(B, H, L), (0, 2, 1))
        db = jnp.transpose(dB_flat[:, :L].reshape(B, H, L), (0, 2, 1))
    else:
        dq = dQ_flat[:, :L, :].reshape(B, L, H, K)
        dk = dK_flat[:, :L, :].reshape(B, L, H, K)
        dv = dV_flat[:, :L, : v_arr.shape[-1]].reshape(B, L, H, v_arr.shape[-1])
        dg = dG_flat[:, :L].reshape(B, L, H)
        db = dB_flat[:, :L].reshape(B, L, H)

    d_initial_state = None if initial_state_was_none else dS0.reshape(B, -1, K, V_pad)[:, :H, :, : v_arr.shape[-1]]
    return dq, dk, dv, dg, db, d_initial_state


# ---------------------------------------------------------------------------
# Rematerialized backward for the flash chunk kernel (custom VJP wrapper)
# ---------------------------------------------------------------------------

# We wrap a *pure arrays* front-end around `_chunk_gated_delta_rule_flash_fused`
# and give it a custom VJP whose backward recomputes the reference forward and
# uses JAX's VJP to obtain gradients. This avoids saving large per-token
# intermediates from the flash forward (T, W/U, etc.).
#
# Notes:
# - All non-differentiable flags are marked static via `nondiff_argnums`.
# - We ignore d(final_state) in backward (it's not used in training prefill);
#   if you want gradients wrt initial_state in the future, wire a second VJP
#   with a reverse scan over the chunk bridge.
#


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=(
        8,  # chunk_size
        9,  # output_final_state
        10,  # use_qk_l2norm_in_kernel
        11,  # head_first
        12,  # use_varlen
    ),
)
def _chunk_gated_delta_rule_flash(
    q_arr,
    k_arr,
    v_arr,
    g_arr,
    beta_arr,
    lengths,
    offsets,
    initial_state,
    chunk_size,
    output_final_state,
    use_qk_l2norm_in_kernel,
    head_first,
    use_varlen,
):
    out_arr, final_state, _ = _chunk_gated_delta_rule_flash_fused(
        q_arr,
        k_arr,
        v_arr,
        g_arr,
        beta_arr,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        head_first=head_first,
        lengths=lengths,
        offsets=offsets,
        use_varlen=use_varlen,
        save_for_backward=False,  # plain call (used by export or when no grads)
    )
    return (out_arr, final_state)


def _chunk_gated_delta_rule_flash_call_fwd(
    q_arr,
    k_arr,
    v_arr,
    g_arr,
    beta_arr,
    lengths,
    offsets,
    initial_state,
    chunk_size,
    output_final_state,
    use_qk_l2norm_in_kernel,
    head_first,
    use_varlen,
):
    out_arr, final_state, aux = _chunk_gated_delta_rule_flash_fused(
        q_arr,
        k_arr,
        v_arr,
        g_arr,
        beta_arr,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        head_first=head_first,
        lengths=lengths,
        offsets=offsets,
        use_varlen=use_varlen,
        save_for_backward=True,  # <-- save A0/S_prev for bwd
    )
    # Residuals for bwd (arrays only; no Python captures)
    initial_state_was_none = initial_state is None
    res = (q_arr, k_arr, v_arr, g_arr, beta_arr, lengths, head_first, initial_state_was_none, aux)
    return (out_arr, final_state), res


def _chunk_gated_delta_rule_flash_call_bwd(
    chunk_size,
    output_final_state,
    use_qk_l2norm_in_kernel,
    head_first,
    use_varlen,
    res,
    cotangents,
):
    q_arr, k_arr, v_arr, g_arr, beta_arr, lengths, head_first_res, initial_state_was_none, aux = res
    d_out_arr, _d_final_state = cotangents  # we ignore d(final_state)

    dq, dk, dv, dg, db, dS0 = _chunk_gated_delta_rule_flash_bwd_fused(
        q_arr,
        k_arr,
        v_arr,
        g_arr,
        beta_arr,
        d_out_arr,
        head_first=head_first_res,
        lengths=lengths,
        aux=aux,
        initial_state_was_none=initial_state_was_none,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # Match primal differentiable args: (q, k, v, g, beta, lengths, offsets, initial_state)
    d_lengths = None
    d_offsets = None
    d_initial_state = None if initial_state_was_none else dS0
    return (dq, dk, dv, dg, db, d_lengths, d_offsets, d_initial_state)


_chunk_gated_delta_rule_flash.defvjp(
    _chunk_gated_delta_rule_flash_call_fwd,
    _chunk_gated_delta_rule_flash_call_bwd,
)


def _wrap_chunk_out_as_named(out_arr, query, value, *, head_first: bool):
    """Return the chunk output as NamedArray with the same axes as the reference path."""
    if not head_first:
        # out_arr is (B, L, H, V) and value.axes is (Batch, Position, Heads, VHeadDim)
        return hax.named(out_arr, value.axes)
    else:
        # out_arr is (B, H, L, V)
        return hax.named(
            out_arr,
            (
                query.resolve_axis("batch"),
                query.resolve_axis("heads"),
                query.resolve_axis("position"),
                value.resolve_axis("v_head_dim"),
            ),
        )


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
    lengths: Optional[jnp.ndarray] = None,
) -> tuple[NamedArray, Optional[jnp.ndarray]]:
    if use_flash:
        try:
            out_arr, fin = _chunk_gated_delta_rule_flash(
                query.array,
                key.array,
                value.array,
                g.array,
                beta.array,
                lengths=lengths,
                offsets=offsets,
                initial_state=initial_state,
                chunk_size=chunk_size,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                head_first=head_first,
                use_varlen=use_varlen,
            )
            out_named = _wrap_chunk_out_as_named(out_arr, query, value, head_first=head_first)
            return out_named, fin
        except Exception:
            if use_flash:
                raise
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
