# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray


# ---------- small utilities ----------


def _l2norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """L2-normalize the last dimension of a positional tensor."""
    inv = jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
    return x * inv


def _causal_depthwise_conv1d_full(
    x_ncl: jnp.ndarray, w_ck: jnp.ndarray, bias_c: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Depthwise 1D conv with causal semantics (left padding). Matches HF fallback behavior:

      - Input:  x: (N, C, L)
      - Filter: w: (C, K) interpreted as depthwise (out_channels=C, in_channels/group=1, kernel=K)
      - Output: y: (N, C, L)

    We implement causality by explicit left padding of the LHS with (K-1) zeros and then a VALID conv.
    """
    N, C, L = x_ncl.shape
    K = w_ck.shape[-1]
    # LHS with left-pad (K-1, 0)
    x_pad = jnp.pad(x_ncl, ((0, 0), (0, 0), (K - 1, 0)))
    # RHS filter shape (O, I, K) with I=1 for depthwise; feature_group_count=C
    w_oik = w_ck[:, None, :]
    y = lax.conv_general_dilated(
        lhs=x_pad,
        rhs=w_oik,
        window_strides=(1,),
        padding="VALID",  # already applied left pad
        dimension_numbers=("NCH", "OIH", "NCH"),  # treat time as H in 1d
        feature_group_count=C,
    )
    if bias_c is not None:
        y = y + bias_c[:, None]
    # SiLU activation (as in HF)
    y = jax.nn.silu(y)
    return y


def _causal_depthwise_conv1d_update(
    x_ncl_1: jnp.ndarray,  # (N, C, 1)
    w_ck: jnp.ndarray,  # (C, K)
    bias_c: Optional[jnp.ndarray],
    prev_state_nck: jnp.ndarray,  # (N, C, K)   <-- K not K-1
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Depthwise conv update for a single timestep with state carry (N, C, K)."""
    # concat to window of K+1
    x_hist = jnp.concatenate([prev_state_nck, x_ncl_1], axis=-1)  # (N, C, K+1)

    # run conv: output length = 2, take the last one
    y2 = lax.conv_general_dilated(
        lhs=x_hist,
        rhs=w_ck[:, None, :],  # (O=C, I=1, K)
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=x_hist.shape[1],
    )  # (N, C, 2)
    y = y2[..., -1:]  # (N, C, 1)
    if bias_c is not None:
        y = y + bias_c[:, None]
    y = jax.nn.silu(y)

    # keep last K inputs as new state
    new_state = jnp.concatenate([prev_state_nck[..., 1:], x_ncl_1], axis=-1)  # (N, C, K)
    return y, new_state


# ---------- Gated RMSNorm with external gate ----------


class GatedRmsNorm(eqx.Module):
    """RMSNorm(x) * SiLU(gate) with learnable weight over a single axis (Haliax-native)."""

    axis: Axis
    weight: NamedArray  # [axis]
    eps: float = eqx.field(default=1e-6, static=True)

    @staticmethod
    def init(axis: Axis, eps: float = 1e-6) -> "GatedRmsNorm":
        return GatedRmsNorm(axis=axis, weight=hax.ones(axis), eps=eps)

    def __call__(self, x: NamedArray, gate: NamedArray) -> NamedArray:
        """
        Normalize x along `axis`, then multiply by learned weight and SiLU(gate).
        Shapes:
          x:    [..., axis]
          gate: [..., axis]
        """
        in_dtype = x.dtype
        x32 = x.astype(jnp.float32)
        var = hax.mean(hax.square(x32), axis=self.axis)
        inv = hax.rsqrt(var + self.eps)
        y = (x32 * inv).astype(in_dtype)
        y = self.weight * y
        gated = y * hnn.silu(gate)
        return gated.astype(in_dtype)


# ---------- Config ----------


@dataclass(frozen=True)
class GatedDeltaNetConfig:
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
        # [Q|K|V|Z] concatenated dimension
        return Axis("qkvz", self.key_dim * 2 + self.value_dim * 2)

    @property
    def ba_axis(self) -> Axis:
        # [b|a] concatenated per V head
        return Axis("ba", self.num_v_heads * 2)


# ---------- Core kernels (chunkwise & recurrent) ----------


def chunk_gated_delta_rule_jax(
    query: jnp.ndarray,  # (B, H, L, d_k)
    key: jnp.ndarray,  # (B, H, L, d_k)
    value: jnp.ndarray,  # (B, H, L, d_v)
    g: jnp.ndarray,  # (B, H, L)
    beta: jnp.ndarray,  # (B, H, L)
    *,
    chunk_size: int = 64,
    initial_state: Optional[jnp.ndarray] = None,  # (B, H, d_k, d_v)
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """HF-compatible chunkwise-parallel gated delta rule (pure JAX fallback)."""
    # ---- dtypes & optional L2 norm ----
    q = query.astype(jnp.float32)
    k = key.astype(jnp.float32)
    v = value.astype(jnp.float32)
    b = beta.astype(jnp.float32)
    gg = g.astype(jnp.float32)

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)

    B, H, L, dk = k.shape
    dv = v.shape[-1]
    q = q * (dk**-0.5)

    # ---- pad to multiple of chunk_size ----
    pad = (chunk_size - (L % chunk_size)) % chunk_size

    def _pad_time(x, pad_last: int):
        if pad_last == 0:
            return x
        return jnp.pad(x, ((0, 0), (0, 0), (0, pad_last)) + ((0, 0),) * (x.ndim - 3))

    q = _pad_time(q, pad)
    k = _pad_time(k, pad)
    v = _pad_time(v, pad)
    b = _pad_time(b, pad)
    gg = _pad_time(gg, pad)

    Lt = L + pad
    nc = Lt // chunk_size

    v_beta = v * b[..., None]
    k_beta = k * b[..., None]

    def _chunk(x):
        return x.reshape((B, H, nc, chunk_size) + x.shape[3:])

    q_c = _chunk(q)  # (B, H, nc, C, dk)
    k_c = _chunk(k)  # (B, H, nc, C, dk)
    # v_c = _chunk(v)  # (B, H, nc, C, dv)
    kbeta_c = _chunk(k_beta)  # (B, H, nc, C, dk)
    vbeta_c = _chunk(v_beta)  # (B, H, nc, C, dv)
    g_c = _chunk(gg)  # (B, H, nc, C)

    # ---- strictly-lower interaction A and unit-lower precursor ----
    g_cum = jnp.cumsum(g_c, axis=-1)  # (B, H, nc, C)
    diff = g_cum[..., None] - g_cum[..., None, :]  # (B, H, nc, C, C)
    decay = jnp.tril(jnp.exp(jnp.tril(diff)))  # (B, H, nc, C, C)

    A = -jnp.einsum("bhnid,bhnjd->bhnij", kbeta_c, k_c) * decay  # (B,H,nc,C,C)
    mask_upper_eq = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)
    A = jnp.where(mask_upper_eq, 0.0, A)  # strictly lower
    I = jnp.eye(chunk_size, dtype=A.dtype)

    # Forward substitution (HF arithmetic order), no dynamic slice lengths
    attn_low = A
    ar = jnp.arange(chunk_size)  # static for lax.fori_loop

    def body(i, attn):
        # row i (length C), then compute row_i += row_i @ attn[:i,:i]
        row_i = lax.dynamic_slice_in_dim(attn, i, 1, axis=-2)  # (...,1,C)
        row_i = jnp.squeeze(row_i, axis=-2)  # (...,C)

        m1 = (ar < i).astype(attn.dtype)  # (C,)
        m2 = ((ar[:, None] < i) & (ar[None, :] < i)).astype(attn.dtype)  # (C,C)

        row_pref = row_i * m1
        sub_pref = attn * m2

        prod = row_pref[..., None] * sub_pref  # (...,C,C)
        incr = jnp.sum(prod, axis=-2)  # (...,C)

        new_row = jnp.expand_dims(row_i + incr, axis=-2)  # (...,1,C)
        attn = lax.dynamic_update_slice_in_dim(attn, new_row, i, axis=-2)
        return attn

    attn_low = lax.fori_loop(1, chunk_size, body, attn_low)
    T = attn_low + I  # (B, H, nc, C, C)

    # v_pseudo = T @ v_beta
    v_pseudo = jnp.einsum("bhnij,bhnjd->bhnid", T, vbeta_c)  # (B,H,nc,C,dv)
    # k_cumdecay = T @ (k_beta * exp(g_cum))
    k_cumdecay = jnp.einsum("bhnij,bhnjd->bhnid", T, kbeta_c * jnp.exp(g_cum)[..., None])

    # ---- scan over chunks ----
    q_s = jnp.moveaxis(q_c, 2, 0)  # (nc, B, H, C, dk)
    k_s = jnp.moveaxis(k_c, 2, 0)  # (nc, B, H, C, dk)
    v_s = jnp.moveaxis(v_pseudo, 2, 0)  # (nc, B, H, C, dv)
    g_s = jnp.moveaxis(g_cum, 2, 0)  # (nc, B, H, C)
    kc_s = jnp.moveaxis(k_cumdecay, 2, 0)  # (nc, B, H, C, dk)

    S = jnp.zeros((B, H, dk, dv), dtype=v.dtype) if initial_state is None else initial_state.astype(v.dtype)
    mask_strict_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)

    def chunk_step(S_prev, inputs):
        q_i, k_i, v_i, gcum_i, kcum_i = inputs  # q_i: (B,H,C,dk), etc.

        # in-chunk decay & attention (keep diagonal, zero strictly upper)
        diff_i = gcum_i[..., None] - gcum_i[..., None, :]  # (B,H,C,C)
        decay_i = jnp.tril(jnp.exp(jnp.tril(diff_i)))  # (B,H,C,C)
        attn_i = jnp.einsum("bhid,bhjd->bhij", q_i, k_i) * decay_i
        attn_i = jnp.where(mask_strict_upper, 0.0, attn_i)

        v_prime = jnp.einsum("bhid,bhdm->bhim", kcum_i, S_prev)  # (B,H,C,dv)
        v_new = v_i - v_prime

        qexp = q_i * jnp.exp(gcum_i)[..., None]  # (B,H,C,dk)
        inter = jnp.einsum("bhid,bhdm->bhim", qexp, S_prev)  # (B,H,C,dv)
        out_i = inter + jnp.einsum("bhij,bhjm->bhim", attn_i, v_new)

        g_tail = gcum_i[..., -1]  # (B,H)
        decay_tail = jnp.exp(g_tail)[..., None, None]  # (B,H,1,1)
        decay_weights = jnp.exp((g_tail[..., None] - gcum_i))[..., None]  # (B,H,C,1)

        add = jnp.einsum("bhid,bhim->bhdm", k_i * decay_weights, v_new)
        S_new = S_prev * decay_tail + add
        return S_new, out_i

    S, out_chunks = jax.lax.scan(
        chunk_step,
        S,
        (q_s, k_s, v_s, g_s, kc_s),
        length=nc,
    )

    out = jnp.moveaxis(out_chunks, 0, 2).reshape(B, H, Lt, dv)[:, :, :L, :]  # (B,H,L,dv)
    return (out, S) if output_final_state else (out, None)


def recurrent_gated_delta_rule_jax(
    query: jnp.ndarray,  # (B, H, L, d_k)
    key: jnp.ndarray,  # (B, H, L, d_k)
    value: jnp.ndarray,  # (B, H, L, d_v)
    g: jnp.ndarray,  # (B, H, L)
    beta: jnp.ndarray,  # (B, H, L)
    *,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    HF-compatible recurrent (decode) gated delta rule (pure JAX fallback).
    """
    q = query.astype(jnp.float32)
    k = key.astype(jnp.float32)
    v = value.astype(jnp.float32)
    b = beta.astype(jnp.float32)
    gg = g.astype(jnp.float32)

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)

    B, H, L, dk = k.shape
    dv = v.shape[-1]

    q = q * (dk**-0.5)
    # o = jnp.zeros((B, H, L, dv), dtype=v.dtype)
    S = jnp.zeros((B, H, dk, dv), dtype=v.dtype) if initial_state is None else initial_state.astype(v.dtype)

    def body(carry, xs):
        S_prev = carry
        q_t, k_t, v_t, g_t, b_t = xs
        # decay
        S_prev = S_prev * jnp.exp(g_t)[..., None, None]
        # delta = (v - (S k)) * beta
        kv_mem = jnp.sum(S_prev * k_t[..., None], axis=-2)  # (B,H,dv)
        delta = (v_t - kv_mem) * b_t[..., None]
        S_new = S_prev + k_t[..., None] * delta[..., None, :]
        out_t = jnp.sum(S_new * q_t[..., None], axis=-2)
        return S_new, out_t

    S, out_seq = jax.lax.scan(
        body,
        S,
        (
            jnp.moveaxis(q, 2, 0),
            jnp.moveaxis(k, 2, 0),
            jnp.moveaxis(v, 2, 0),
            jnp.moveaxis(gg, 2, 0),
            jnp.moveaxis(b, 2, 0),
        ),
        length=L,
    )
    out = jnp.moveaxis(out_seq, 0, 2)  # back to (B,H,L,dv)

    if output_final_state:
        return out, S
    else:
        return out, None


# ---------- NamedArray wrappers for kernels ----------


def chunk_gated_delta_rule(
    query: NamedArray,  # [batch, position, heads, k_head_dim]
    key: NamedArray,  # [batch, position, heads, k_head_dim]
    value: NamedArray,  # [batch, position, heads, v_head_dim]
    g: NamedArray,  # [batch, position, heads]
    beta: NamedArray,  # [batch, position, heads]
    *,
    chunk_size: int = 64,
    initial_state: Optional[jnp.ndarray] = None,  # (B, H, d_k, d_v)
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[NamedArray, Optional[jnp.ndarray]]:
    """
    Haliax-friendly wrapper around `chunk_gated_delta_rule_jax`.

    Expected axes:
        query/key:  ("batch", "position", "heads", "k_head_dim")
        value:      ("batch", "position", "heads", "v_head_dim")
        g/beta:     ("batch", "position", "heads")

    Returns:
        out:  [batch, position, heads, v_head_dim]
        S:    optional jnp.ndarray with shape (B, H, dk, dv)
    """
    q_bhl = hax.rearrange(query, ("batch", "heads", "position", "k_head_dim")).array
    k_bhl = hax.rearrange(key, ("batch", "heads", "position", "k_head_dim")).array
    v_bhl = hax.rearrange(value, ("batch", "heads", "position", "v_head_dim")).array
    g_bhl = hax.rearrange(g, ("batch", "heads", "position")).array
    b_bhl = hax.rearrange(beta, ("batch", "heads", "position")).array

    out_bhl, S = chunk_gated_delta_rule_jax(
        q_bhl,
        k_bhl,
        v_bhl,
        g_bhl,
        b_bhl,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    # to [B, L, H, Dv] named
    out_blh = jnp.moveaxis(out_bhl, 2, 1)
    out_named = hax.named(out_blh, ("batch", "position", "heads", "v_head_dim"))
    return out_named, S


def recurrent_gated_delta_rule(
    query: NamedArray,  # [batch, position, heads, k_head_dim]
    key: NamedArray,  # [batch, position, heads, k_head_dim]
    value: NamedArray,  # [batch, position, heads, v_head_dim]
    g: NamedArray,  # [batch, position, heads]
    beta: NamedArray,  # [batch, position, heads]
    *,
    initial_state: Optional[jnp.ndarray] = None,  # (B, H, dk, dv)
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[NamedArray, Optional[jnp.ndarray]]:
    """
    Haliax-friendly wrapper around `recurrent_gated_delta_rule_jax`.

    Expected axes:
        query/key:  ("batch", "position", "heads", "k_head_dim")
        value:      ("batch", "position", "heads", "v_head_dim")
        g/beta:     ("batch", "position", "heads")

    Returns:
        out:  [batch, position, heads, v_head_dim]
        S:    optional jnp.ndarray with shape (B, H, dk, dv)
    """
    q_bhl = hax.rearrange(query, ("batch", "heads", "position", "k_head_dim")).array
    k_bhl = hax.rearrange(key, ("batch", "heads", "position", "k_head_dim")).array
    v_bhl = hax.rearrange(value, ("batch", "heads", "position", "v_head_dim")).array
    g_bhl = hax.rearrange(g, ("batch", "heads", "position")).array
    b_bhl = hax.rearrange(beta, ("batch", "heads", "position")).array

    out_bhl, S = recurrent_gated_delta_rule_jax(
        q_bhl,
        k_bhl,
        v_bhl,
        g_bhl,
        b_bhl,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    out_blh = jnp.moveaxis(out_bhl, 2, 1)
    out_named = hax.named(out_blh, ("batch", "position", "heads", "v_head_dim"))
    return out_named, S


# ---------- Layer ----------


class GatedDeltaNet(eqx.Module):
    """Gated DeltaNet token mixer implemented with Haliax-friendly plumbing around HF-equivalent kernels."""

    config: GatedDeltaNetConfig = eqx.field(static=True)

    # projections
    in_proj_qkvz: hnn.Linear  # [Embed] -> [qkvz]
    in_proj_ba: hnn.Linear  # [Embed] -> [ba]

    # depthwise conv parameters over concatenated [Q|K|V] channels
    conv_weight: jnp.ndarray  # (C, K)
    conv_bias: Optional[jnp.ndarray]  # (C,) or None

    # discretization params per V head
    A_log: jnp.ndarray  # (num_v_heads,)
    dt_bias: jnp.ndarray  # (num_v_heads,)

    # gated RMSNorm and output projection
    o_norm: GatedRmsNorm
    out_proj: hnn.Linear  # [VHeads, VHeadDim] -> [Embed]

    @staticmethod
    def init(config: GatedDeltaNetConfig, *, key) -> "GatedDeltaNet":
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

        # conv parameters (depthwise over concatenated [Q|K|V] only)
        C = config.key_dim * 2 + config.value_dim  # channels through conv (Q|K|V)
        K = config.conv_kernel_size
        conv_weight = jax.random.normal(k_conv, (C, K), dtype=jnp.float32) * (1.0 / jnp.sqrt(C * K))
        conv_bias = None  # HF sets bias=False

        # Mamba-style discretization parameters per value head
        A_log = jnp.log(jax.random.uniform(k_out, (config.num_v_heads,), minval=0.0, maxval=16.0, dtype=jnp.float32))
        # HF uses dt_bias initialized to 1.0
        dt_bias = jnp.ones((config.num_v_heads,), dtype=jnp.float32)

        # gated RMSNorm and output projection
        o_norm = GatedRmsNorm.init(config.VHeadDim, eps=config.rms_norm_eps)
        out_proj = hnn.Linear.init(
            In=(config.VHeads, config.VHeadDim), Out=config.Embed, out_first=True, use_bias=False, key=k_out
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

    # ---- helpers to reshape/split like HF ----

    def _fix_qkvz_ordering(
        self, mixed_qkvz: NamedArray, mixed_ba: NamedArray
    ) -> Tuple[NamedArray, NamedArray, NamedArray, NamedArray, NamedArray, NamedArray]:
        """Unpack [Q|K|V|Z] and [b|a] and lay them out with named per-head axes.

        Returns:
            q: [B, Pos, KHeads, KHeadDim]
            k: [B, Pos, KHeads, KHeadDim]
            v: [B, Pos, VHeads, VHeadDim]
            z: [B, Pos, VHeads, VHeadDim]
            b: [B, Pos, VHeads]
            a: [B, Pos, VHeads]
        """
        cfg = self.config
        ratio = cfg.num_v_heads // cfg.num_k_heads

        # ---- qkvz ----
        per_head = Axis("per_head", 2 * cfg.head_k_dim + 2 * ratio * cfg.head_v_dim)
        x = mixed_qkvz.unflatten_axis(cfg.mix_qkvz_axis, (cfg.KHeads, per_head))

        def sl(start, size, ax=per_head):
            return hax.ds(start, Axis(ax.name, size))

        q = x["per_head", sl(0, cfg.head_k_dim)].rename({"per_head": cfg.KHeadDim.name})
        k = x["per_head", sl(cfg.head_k_dim, cfg.head_k_dim)].rename({"per_head": cfg.KHeadDim.name})
        v_chunk = x["per_head", sl(2 * cfg.head_k_dim, ratio * cfg.head_v_dim)]
        z_chunk = x["per_head", sl(2 * cfg.head_k_dim + ratio * cfg.head_v_dim, ratio * cfg.head_v_dim)]

        v = v_chunk.unflatten_axis(
            v_chunk.resolve_axis("per_head"), (Axis("v_group", ratio), cfg.VHeadDim)
        ).flatten_axes(("k_heads", "v_group"), cfg.VHeads)

        z = z_chunk.unflatten_axis(
            z_chunk.resolve_axis("per_head"), (Axis("v_group", ratio), cfg.VHeadDim)
        ).flatten_axes(("k_heads", "v_group"), cfg.VHeads)

        # ---- b|a ----
        per_ba = Axis("per_ba", 2 * ratio)
        ba = mixed_ba.unflatten_axis(cfg.ba_axis, (cfg.KHeads, per_ba))

        b_chunk = ba["per_ba", hax.ds(0, ratio)]
        a_chunk = ba["per_ba", hax.ds(ratio, ratio)]

        b = b_chunk.flatten_axes(("k_heads", "per_ba"), cfg.VHeads)
        a = a_chunk.flatten_axes(("k_heads", "per_ba"), cfg.VHeads)

        return q, k, v, z, b, a

    # ---- forward ----

    def __call__(
        self,
        x: NamedArray,
        *,
        inference: bool = True,
        chunk_size: int = 64,
        attention_mask: Optional[NamedArray] = None,
        decode_state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,  # (conv_state, S_state)
    ) -> Tuple[NamedArray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """
        x: [Batch, Position, Embed]
        Returns: (y, new_state)
            y: [Batch, Position, Embed]
            new_state: (conv_state, recurrent_S) if decode_state is not None or inference==True
        """
        cfg = self.config

        # 0) Mask padded tokens (if provided): ensure mask has Embed axis and broadcast by name.
        if attention_mask is not None:
            m3 = attention_mask.astype(x.dtype).broadcast_axis(cfg.Embed)
            x = x * m3

        # 1) Projections (NamedArray)
        mixed_qkvz = self.in_proj_qkvz(x)  # [B,Pos,qkvz]
        mixed_ba = self.in_proj_ba(x)  # [B,Pos,ba]
        q, k, v, z, b, a = self._fix_qkvz_ordering(mixed_qkvz, mixed_ba)

        # 2) Depthwise causal conv over concatenated [Q|K|V] channels
        q_ch = q.flatten_axes((cfg.KHeads, cfg.KHeadDim), Axis("channels", cfg.key_dim))
        k_ch = k.flatten_axes((cfg.KHeads, cfg.KHeadDim), Axis("channels", cfg.key_dim))
        v_ch = v.flatten_axes((cfg.VHeads, cfg.VHeadDim), Axis("channels", cfg.value_dim))
        Channels = Axis("channels", cfg.key_dim * 2 + cfg.value_dim)
        qkv_ch = hax.concatenate(Channels, [q_ch, k_ch, v_ch])  # [B,Pos,Channels]

        # to (N, C, L) by name
        qkv_ncl = hax.rearrange(qkv_ch, ("batch", Channels, "position")).array  # [N, C, L]

        if decode_state is not None and x.axis_size("position") == 1:
            # decode update path
            conv_state, S_state = decode_state
            K = self.conv_weight.shape[-1]
            assert conv_state.shape[-1] == K, "conv_state must have shape (N, C, K)"
            y_ncl, new_conv_state = _causal_depthwise_conv1d_update(
                qkv_ncl, self.conv_weight, self.conv_bias, conv_state
            )
        else:
            # prefill path
            y_ncl = _causal_depthwise_conv1d_full(qkv_ncl, self.conv_weight, self.conv_bias)
            if inference:
                # cache conv state for future decode (last K inputs)
                K = self.conv_weight.shape[-1]
                L = x.axis_size("position")
                if L >= K:
                    new_conv_state = qkv_ncl[..., -K:]
                else:
                    new_conv_state = jnp.pad(qkv_ncl, ((0, 0), (0, 0), (K - L, 0)))
            else:
                new_conv_state = None
                S_state = None

        # 3) Split conv output back to Q,K,V
        y_bpc = hax.rearrange(hax.named(y_ncl, ("batch", "channels", "position")), ("batch", "position", "channels"))
        q_y = y_bpc["channels", hax.ds(0, Axis("chan_k2", cfg.key_dim))].rename({"channels": "chan_k2"})
        k_y = y_bpc["channels", hax.ds(cfg.key_dim, Axis("chan_k2", cfg.key_dim))].rename({"channels": "chan_k2"})
        v_y = y_bpc["channels", hax.ds(2 * cfg.key_dim, Axis("chan_v2", cfg.value_dim))].rename(
            {"channels": "chan_v2"}
        )

        q = q_y.unflatten_axis(q_y.resolve_axis("chan_k2"), (cfg.KHeads, cfg.KHeadDim))
        k = k_y.unflatten_axis(k_y.resolve_axis("chan_k2"), (cfg.KHeads, cfg.KHeadDim))
        v = v_y.unflatten_axis(v_y.resolve_axis("chan_v2"), (cfg.VHeads, cfg.VHeadDim))

        # 4) Parameters for decay and delta, done with NamedArrays
        beta = hnn.sigmoid(b)  # [B,Pos,VHeads]

        # Mamba-style discretization: g = -exp(A) * softplus(a + dt_bias); α = exp(g)
        a32 = a.astype(jnp.float32)
        dt_bias_na = hax.named(self.dt_bias.astype(jnp.float32), cfg.VHeads)
        A_exp = hax.exp(hax.named(self.A_log.astype(jnp.float32), cfg.VHeads))
        g = -(A_exp * hnn.softplus(a32 + dt_bias_na)).astype(x.dtype)  # [B,Pos,VHeads] (log-decay)

        # 5) Match heads: if VHeads > KHeads, repeat Q,K by broadcasting a new axis, then flatten
        ratio = cfg.num_v_heads // cfg.num_k_heads
        if ratio > 1:
            VGroup = Axis("v_group", ratio)
            q = q.broadcast_axis(VGroup).flatten_axes((cfg.KHeads, VGroup), cfg.VHeads)
            k = k.broadcast_axis(VGroup).flatten_axes((cfg.KHeads, VGroup), cfg.VHeads)
        else:
            q = q.rename({cfg.KHeads.name: cfg.VHeads.name})
            k = k.rename({cfg.KHeads.name: cfg.VHeads.name})

        # 6) Core kernels (prefill=chunkwise, decode=recurrent)
        def _to_bhl(t: NamedArray, head_axis: Axis, dim_axis: Axis) -> jnp.ndarray:
            # [B,Pos,H,Dim] -> [B,H,L,Dim]
            t_bphd = t.rearrange(("batch", "position", head_axis.name, dim_axis.name)).array
            return jnp.moveaxis(t_bphd, 1, 2)

        q_bhl = _to_bhl(q, cfg.VHeads, cfg.KHeadDim)
        k_bhl = _to_bhl(k, cfg.VHeads, cfg.KHeadDim)
        v_bhl = _to_bhl(v, cfg.VHeads, cfg.VHeadDim)
        g_bhl = jnp.moveaxis(g.rename({cfg.VHeads.name: "heads"}).array, 1, 2)
        b_bhl = jnp.moveaxis(beta.rename({cfg.VHeads.name: "heads"}).array, 1, 2)

        if decode_state is not None and x.axis_size("position") == 1 and S_state is not None:
            out_bhl, S_new = recurrent_gated_delta_rule_jax(
                q_bhl,
                k_bhl,
                v_bhl,
                g_bhl,
                b_bhl,
                initial_state=S_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            out_bhl, S_new = chunk_gated_delta_rule_jax(
                q_bhl,
                k_bhl,
                v_bhl,
                g_bhl,
                b_bhl,
                chunk_size=chunk_size,
                initial_state=None,
                output_final_state=inference,
                use_qk_l2norm_in_kernel=True,
            )

        # 7) Back to [B,Pos,VHeads,VHeadDim]
        out_blhd = jnp.moveaxis(out_bhl, 2, 1)
        out_named = hax.named(out_blhd, ("batch", "position", "v_heads", "v_head_dim"))

        # 8) Gated RMSNorm with Z gate
        y_norm = self.o_norm(out_named, gate=z)

        # 9) Project out to Embed
        y_out = self.out_proj(y_norm.astype(x.dtype))  # [B,Pos,Embed]

        new_state = None
        if inference:
            new_state = (new_conv_state, S_new)
        return y_out, new_state
