# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# based on:
# - https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py
# - the JAX implementation by Yu Sun and Leo Lee

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


def _l2norm(x: NamedArray, axis: hax.AxisSelector, eps: float = 1e-6) -> NamedArray:
    """L2-normalize x along a named axis."""
    inv = hax.rsqrt(hax.sum(hax.square(x), axis=axis) + jnp.asarray(eps, dtype=jnp.float32))
    return (x * inv).astype(x.dtype)


# ---------- depthwise conv: positional (lax) helpers with named wrappers ----------


def _causal_depthwise_conv1d_full(
    x_ncl: jnp.ndarray, w_ck: jnp.ndarray, bias_c: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Depthwise 1D conv with causal semantics (left padding).

    - Input:  x: (N, C, L)
    - Filter: w: (C, K) depthwise
    - Output: y: (N, C, L)
    """
    N, C, L = x_ncl.shape
    K = w_ck.shape[-1]
    x_pad = jnp.pad(x_ncl, ((0, 0), (0, 0), (K - 1, 0)))
    w_oik = w_ck[:, None, :]
    y = lax.conv_general_dilated(
        lhs=x_pad,
        rhs=w_oik,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=C,
    )
    if bias_c is not None:
        y = y + bias_c[:, None]
    y = jax.nn.silu(y)
    return y


def _causal_depthwise_conv1d_update(
    x_ncl_1: jnp.ndarray,  # (N, C, 1)
    w_ck: jnp.ndarray,  # (C, K)
    bias_c: Optional[jnp.ndarray],
    prev_state_nck: jnp.ndarray,  # (N, C, K)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Depthwise conv update for a single timestep with state carry (N, C, K)."""
    x_hist = jnp.concatenate([prev_state_nck, x_ncl_1], axis=-1)  # (N, C, K+1)
    y2 = lax.conv_general_dilated(
        lhs=x_hist,
        rhs=w_ck[:, None, :],
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=x_hist.shape[1],
    )
    y = y2[..., -1:]  # (N, C, 1)
    if bias_c is not None:
        y = y + bias_c[:, None]
    y = jax.nn.silu(y)
    new_state = jnp.concatenate([prev_state_nck[..., 1:], x_ncl_1], axis=-1)
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
        in_dtype = x.dtype
        var = hax.mean(hax.square(x), axis=self.axis)
        inv = hax.rsqrt(var + jnp.asarray(self.eps, dtype=jnp.float32))
        y = (x * inv).astype(in_dtype)
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
        return Axis("qkvz", self.key_dim * 2 + self.value_dim * 2)

    @property
    def ba_axis(self) -> Axis:
        return Axis("ba", self.num_v_heads * 2)


# ---------- Triangular masks ----------


def _tri_upper_eq_mask(Ci: Axis, Cj: Axis) -> NamedArray:
    ii = hax.arange(Ci)
    jj = hax.arange(Cj)
    I = ii.broadcast_axis(Cj)
    J = jj.broadcast_axis(Ci)
    return I <= J


# ---------- Kernels ----------


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
) -> Tuple[NamedArray, Optional[jnp.ndarray]]:
    """
    Haliax-native recurrent (decode) gated delta rule, keeping lax.scan outer structure
    and performing inner math with NamedArrays (wrapping/unwrap tracers).
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
    q = q * (Dk.size**-0.5)

    B_, H_, L_, dk_, dv_ = Batch.size, Heads.size, Pos.size, Dk.size, Dv.size
    S0 = jnp.zeros((B_, H_, dk_, dv_), dtype=v.dtype) if initial_state is None else initial_state.astype(v.dtype)

    # positional for scan
    q_bhld = hax.rearrange(q, (Batch, Heads, Pos, Dk)).array
    k_bhld = hax.rearrange(k, (Batch, Heads, Pos, Dk)).array
    v_bhld = hax.rearrange(v, (Batch, Heads, Pos, Dv)).array
    g_bhl = hax.rearrange(gg, (Batch, Heads, Pos)).array
    b_bhl = hax.rearrange(b, (Batch, Heads, Pos)).array

    def step(S_prev_arr, xs_arr):
        q_t_arr, k_t_arr, v_t_arr, g_t_arr, b_t_arr = xs_arr
        S_prev = hax.named(S_prev_arr, (Batch, Heads, Dk, Dv))
        q_t = hax.named(q_t_arr, (Batch, Heads, Dk))
        k_t = hax.named(k_t_arr, (Batch, Heads, Dk))
        v_t = hax.named(v_t_arr, (Batch, Heads, Dv))
        g_t = hax.named(g_t_arr, (Batch, Heads))
        b_t = hax.named(b_t_arr, (Batch, Heads))

        # Decay
        decay = hax.exp(g_t).broadcast_axis(Dk).broadcast_axis(Dv)
        S_prev = S_prev * decay

        # kv = (S_prev · k_t) over Dk
        kv = hax.dot(S_prev * k_t.broadcast_axis(Dv), axis=Dk)

        # delta and S update
        delta = (v_t - kv) * b_t.broadcast_axis(Dv)
        S_new = S_prev + k_t.broadcast_axis(Dv) * delta.broadcast_axis(Dk)

        # output
        y_t = hax.dot(S_new * q_t.broadcast_axis(Dv), axis=Dk)

        return S_new.array, y_t.array

    S_final, out_seq = jax.lax.scan(
        step,
        S0,
        (
            jnp.moveaxis(q_bhld, 2, 0),
            jnp.moveaxis(k_bhld, 2, 0),
            jnp.moveaxis(v_bhld, 2, 0),
            jnp.moveaxis(g_bhl, 2, 0),
            jnp.moveaxis(b_bhl, 2, 0),
        ),
        length=L_,
    )

    out_bhlv = jnp.moveaxis(out_seq, 0, 2)  # (B,H,L,Dv)
    out_named_bhlv = hax.named(out_bhlv, (Batch, Heads, Pos, Dv))
    out_final = hax.rearrange(out_named_bhlv, (Batch, Pos, Heads, Dv))

    if output_final_state:
        return out_final, S_final
    else:
        return out_final, None


def chunk_gated_delta_rule(
    query: NamedArray,  # [batch, position, heads, k_head_dim]
    key: NamedArray,  # [batch, position, heads, k_head_dim]
    value: NamedArray,  # [batch, position, heads, v_head_dim]
    g: NamedArray,  # [batch, position, heads]
    beta: NamedArray,  # [batch, position, heads]
    *,
    chunk_size: int = 64,
    initial_state: Optional[jnp.ndarray] = None,  # (B,H,dk,dv)
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[NamedArray, Optional[jnp.ndarray]]:
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
    gg = g.astype(jnp.float32)
    b = beta.astype(jnp.float32)

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q, axis=Dk)
        k = _l2norm(k, axis=Dk)
    q = q * (Dk.size**-0.5)

    # ---- pad to multiple of chunk_size ----
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

    v_beta = v_c * b_c.broadcast_axis(Dv)
    k_beta = k_c * b_c.broadcast_axis(Dk)

    # cumulative g in chunk
    g_cum = hax.cumsum(g_c, axis=C)

    # build A = -(k_beta @ k^T) * decay, lower-triangular in (Ci,Cj)
    Ci = Axis("Ci", C.size)
    Cj = Axis("Cj", C.size)

    kb_ci = k_beta.rename({C.name: Ci.name})  # [B,Chunks,Ci,H,Dk]
    k_cj = k_c.rename({C.name: Cj.name})  # [B,Chunks,Cj,H,Dk]
    A_raw = -hax.dot(kb_ci, k_cj, axis=Dk)  # [B,Chunks,Ci,Cj,H]

    gi = g_cum.rename({C.name: Ci.name})
    gj = g_cum.rename({C.name: Cj.name})
    decay = hax.exp(gi.broadcast_axis(Cj) - gj.broadcast_axis(Ci))  # [B,Chunks,Ci,Cj,H]
    A = A_raw * decay
    A = hax.where(_tri_upper_eq_mask(Ci, Cj), jnp.asarray(0.0, dtype=A.dtype), A)

    # forward substitution on positional view with layout (B,H,nc,C,C)
    A_bhcc = hax.rearrange(A, (Batch, Heads, Chunks, Ci, Cj)).array
    eyeC = jnp.eye(C.size, dtype=A_bhcc.dtype)

    def body(i, attn):
        row_i = lax.dynamic_slice_in_dim(attn, i, 1, axis=-2)  # (...,1,C)
        row_i = jnp.squeeze(row_i, axis=-2)  # (...,C)
        ar = jnp.arange(C.size, dtype=attn.dtype)
        m1 = (ar < i).astype(attn.dtype)
        m2 = ((ar[:, None] < i) & (ar[None, :] < i)).astype(attn.dtype)
        row_pref = row_i * m1
        sub_pref = attn * m2
        incr = jnp.sum(row_pref[..., None] * sub_pref, axis=-2)
        new_row = jnp.expand_dims(row_i + incr, axis=-2)
        return lax.dynamic_update_slice_in_dim(attn, new_row, i, axis=-2)

    attn_low = lax.fori_loop(1, C.size, body, A_bhcc)
    T = attn_low + eyeC  # (B,H,nc,C,C)

    # v_pseudo = T @ v_beta
    vbeta_bhccd = hax.rearrange(v_beta.rename({C.name: Cj.name}), (Batch, Heads, Chunks, Cj, Dv)).array
    v_pseudo = jnp.einsum("bhnij,bhnjd->bhnid", T, vbeta_bhccd)  # (B,H,nc,C,Dv)

    # k_cumdecay = T @ (k_beta * exp(g_cum))
    kbeta_bhccd = hax.rearrange(k_beta.rename({C.name: Cj.name}), (Batch, Heads, Chunks, Cj, Dk)).array
    exp_g_bhcc = hax.rearrange(hax.exp(g_cum).rename({C.name: Cj.name}), (Batch, Heads, Chunks, Cj)).array
    k_cumdecay = jnp.einsum("bhnij,bhnjd->bhnid", T, kbeta_bhccd * exp_g_bhcc[..., None])  # (B,H,nc,C,dk)

    # scan chunks
    q_bhccd = hax.rearrange(q_c, (Batch, Heads, Chunks, C, Dk)).array
    k_bhccd = hax.rearrange(k_c, (Batch, Heads, Chunks, C, Dk)).array
    g_bhcc = hax.rearrange(g_cum, (Batch, Heads, Chunks, C)).array

    B_, H_, dk_, dv_ = Batch.size, Heads.size, Dk.size, Dv.size
    S = jnp.zeros((B_, H_, dk_, dv_), dtype=v.dtype) if initial_state is None else initial_state.astype(v.dtype)
    mask_strict_upper = jnp.triu(jnp.ones((C.size, C.size), dtype=bool), k=1)

    def chunk_step(S_prev, inps):
        q_i, k_i, v_i, gcum_i, kcum_i = inps  # (B,H,C,dk/dv)
        diff = gcum_i[..., None] - gcum_i[..., None, :]
        decay_i = jnp.exp(jnp.tril(diff))
        attn_i = jnp.einsum("bhid,bhjd->bhij", q_i, k_i) * decay_i
        attn_i = jnp.where(mask_strict_upper, 0.0, attn_i)

        v_prime = jnp.einsum("bhid,bhdm->bhim", kcum_i, S_prev)  # (B,H,C,dv)
        v_new = v_i - v_prime

        qexp = q_i * jnp.exp(gcum_i)[..., None]
        inter = jnp.einsum("bhid,bhdm->bhim", qexp, S_prev)
        out_i = inter + jnp.einsum("bhij,bhjm->bhim", attn_i, v_new)

        g_tail = gcum_i[..., -1]
        decay_tail = jnp.exp(g_tail)[..., None, None]
        decay_weights = jnp.exp((g_tail[..., None] - gcum_i))[..., None]

        add = jnp.einsum("bhid,bhim->bhdm", k_i * decay_weights, v_new)
        S_new = S_prev * decay_tail + add
        return S_new, out_i

    S, out_chunks = jax.lax.scan(
        chunk_step,
        S,
        (
            jnp.moveaxis(q_bhccd, 2, 0),
            jnp.moveaxis(k_bhccd, 2, 0),
            jnp.moveaxis(v_pseudo, 2, 0),
            jnp.moveaxis(g_bhcc, 2, 0),
            jnp.moveaxis(k_cumdecay, 2, 0),
        ),
        length=Nc,
    )

    out_bhcd = jnp.moveaxis(out_chunks, 0, 2)  # (B,H,nc,C,Dv)
    out_named_bhcd = hax.named(out_bhcd, (Batch, Heads, Chunks, C, Dv))
    out_flat_bhPd = out_named_bhcd.flatten_axes((Chunks, C), PosPad)
    out_bhLd = out_flat_bhPd["position", hax.ds(0, L)]
    out_final = hax.rearrange(out_bhLd, (Batch, PosPad.name, Heads, Dv))

    return (out_final, S) if output_final_state else (out_final, None)


# ---------- Layer ----------


class GatedDeltaNet(eqx.Module):
    """Gated DeltaNet token mixer implemented with Haliax-friendly plumbing around kernels."""

    config: GatedDeltaNetConfig = eqx.field(static=True)

    # projections
    in_proj_qkvz: hnn.Linear  # [Embed] -> [qkvz]
    in_proj_ba: hnn.Linear  # [Embed] -> [ba]

    # depthwise conv parameters over concatenated [Q|K|V] channels
    conv_weight: jnp.ndarray
    conv_bias: Optional[jnp.ndarray]

    # discretization params per V head
    A_log: jnp.ndarray
    dt_bias: jnp.ndarray

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

        C = config.key_dim * 2 + config.value_dim
        K = config.conv_kernel_size
        conv_weight = jax.random.normal(k_conv, (C, K), dtype=jnp.float32) * (1.0 / jnp.sqrt(C * K))
        conv_bias = None

        A_log = jnp.log(jax.random.uniform(k_out, (config.num_v_heads,), minval=0.0, maxval=16.0, dtype=jnp.float32))
        dt_bias = jnp.ones((config.num_v_heads,), dtype=jnp.float32)

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

    def _fix_qkvz_ordering(
        self, mixed_qkvz: NamedArray, mixed_ba: NamedArray
    ) -> Tuple[NamedArray, NamedArray, NamedArray, NamedArray, NamedArray, NamedArray]:
        cfg = self.config
        ratio = cfg.num_v_heads // cfg.num_k_heads

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

        per_ba = Axis("per_ba", 2 * ratio)
        ba = mixed_ba.unflatten_axis(cfg.ba_axis, (cfg.KHeads, per_ba))

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
        cfg = self.config
        Batch = x.resolve_axis("batch")
        Pos = x.resolve_axis("position")

        if attention_mask is not None:
            m3 = attention_mask.astype(x.dtype).broadcast_axis(cfg.Embed)
            x = x * m3

        mixed_qkvz = self.in_proj_qkvz(x)
        mixed_ba = self.in_proj_ba(x)
        q, k, v, z, b, a = self._fix_qkvz_ordering(mixed_qkvz, mixed_ba)

        # Depthwise causal conv over concatenated [Q|K|V]
        Chan = "channels"
        q_ch = q.flatten_axes((cfg.KHeads, cfg.KHeadDim), Axis(Chan, cfg.key_dim))
        k_ch = k.flatten_axes((cfg.KHeads, cfg.KHeadDim), Axis(Chan, cfg.key_dim))
        v_ch = v.flatten_axes((cfg.VHeads, cfg.VHeadDim), Axis(Chan, cfg.value_dim))
        Channels = Axis(Chan, cfg.key_dim * 2 + cfg.value_dim)
        qkv_ch = hax.concatenate(Chan, [q_ch, k_ch, v_ch])  # [B,Pos,Channels]
        qkv_ncl = hax.rearrange(qkv_ch, (Batch, Channels, Pos)).array  # (N,C,L)

        S_state: Optional[jnp.ndarray] = None
        if decode_state is not None and x.axis_size("position") == 1:
            conv_state, S_state = decode_state
            K = self.conv_weight.shape[-1]
            assert conv_state.shape[-1] == K
            y_ncl, new_conv_state = _causal_depthwise_conv1d_update(
                qkv_ncl, self.conv_weight, self.conv_bias, conv_state
            )
        else:
            y_ncl = _causal_depthwise_conv1d_full(qkv_ncl, self.conv_weight, self.conv_bias)
            if inference:
                K = self.conv_weight.shape[-1]
                L = x.axis_size("position")
                if L >= K:
                    new_conv_state = qkv_ncl[..., -K:]
                else:
                    new_conv_state = jnp.pad(qkv_ncl, ((0, 0), (0, 0), (K - L, 0)))
            else:
                new_conv_state = None
                S_state = None

        y_bpc = hax.rearrange(hax.named(y_ncl, (Batch.name, Channels.name, Pos.name)), (Batch, Pos, Channels))
        q_y = y_bpc[Channels.name, hax.ds(0, Axis("chan_k2", cfg.key_dim))].rename({Channels.name: "chan_k2"})
        k_y = y_bpc[Channels.name, hax.ds(cfg.key_dim, Axis("chan_k2", cfg.key_dim))].rename(
            {Channels.name: "chan_k2"}
        )
        v_y = y_bpc[Channels.name, hax.ds(2 * cfg.key_dim, Axis("chan_v2", cfg.value_dim))].rename(
            {Channels.name: "chan_v2"}
        )

        q = q_y.unflatten_axis(q_y.resolve_axis("chan_k2"), (cfg.KHeads, cfg.KHeadDim))
        k = k_y.unflatten_axis(k_y.resolve_axis("chan_k2"), (cfg.KHeads, cfg.KHeadDim))
        v = v_y.unflatten_axis(v_y.resolve_axis("chan_v2"), (cfg.VHeads, cfg.VHeadDim))

        beta = hnn.sigmoid(b)

        a32 = a.astype(jnp.float32)
        dt_bias_na = hax.named(jnp.asarray(self.dt_bias, dtype=jnp.float32), cfg.VHeads)
        A_exp = hax.exp(hax.named(jnp.asarray(self.A_log, dtype=jnp.float32), cfg.VHeads))
        g_named = -(A_exp * hnn.softplus(a32 + dt_bias_na)).astype(x.dtype)

        ratio = cfg.num_v_heads // cfg.num_k_heads
        if ratio > 1:
            VGroup = Axis("v_group", ratio)
            q = q.broadcast_axis(VGroup).flatten_axes((cfg.KHeads, VGroup), cfg.VHeads)
            k = k.broadcast_axis(VGroup).flatten_axes((cfg.KHeads, VGroup), cfg.VHeads)
        else:
            q = q.rename({cfg.KHeads.name: cfg.VHeads.name})
            k = k.rename({cfg.KHeads.name: cfg.VHeads.name})

        # ---- Core kernels expect [batch, position, heads, dim] with head axis named "heads"
        q_h = q.rename({cfg.VHeads.name: "heads"})
        k_h = k.rename({cfg.VHeads.name: "heads"})
        v_h = v.rename({cfg.VHeads.name: "heads"})
        g_h = g_named.rename({cfg.VHeads.name: "heads"})
        b_h = beta.rename({cfg.VHeads.name: "heads"})

        q_bphd = hax.rearrange(q_h, ("batch", "position", "heads", cfg.KHeadDim.name))
        k_bphd = hax.rearrange(k_h, ("batch", "position", "heads", cfg.KHeadDim.name))
        v_bphd = hax.rearrange(v_h, ("batch", "position", "heads", cfg.VHeadDim.name))

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

        # Back to [B,Pos,VHeads,VHeadDim]
        out_named = out_bphd.rename({"heads": cfg.VHeads.name})

        # Gated RMSNorm with Z
        y_norm = self.o_norm(out_named, gate=z)

        # Out proj
        y_out = self.out_proj(y_norm.astype(x.dtype))

        new_state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
        if inference and (new_conv_state is not None) and (S_new is not None):
            new_state = (new_conv_state, S_new)
        return y_out, new_state
