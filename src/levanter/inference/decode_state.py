# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
from typing import Generic, Iterable, Iterator, TypeVar

import equinox as eqx
import haliax as hax
import haliax.haxtyping as ht
import jax
import jax.numpy as jnp
import jaxtyping
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call
from jax import lax

from levanter.inference.utils import INVALID, is_valid


@named_call
def _interleave_kv(new_k, new_v):
    # [T, H, D] x2 -> [T, 2H, D] with (k0,v0,k1,v1,...) along heads
    T, H, D = new_k.shape
    return jnp.stack([new_k, new_v], axis=2).reshape(T, 2 * H, D)


@named_call
@functools.partial(jax.jit, donate_argnums=(0,))
def kv_update_unified_prefix(kv_pages, t_pages, t_slots, new_k, new_v, K):
    """
    Update interleaved key/value pages with new tokens.

    kv_pages: [P, S, 2H, D]  (unified K/V buffer, donated)
    t_pages, t_slots: [T] int32  (only first K are valid)
    new_k, new_v: [T, H, D]
    K: int32 scalar = number of valid updates (num_new_tokens)
    """
    kv_ev = _interleave_kv(new_k.astype(kv_pages.dtype), new_v.astype(kv_pages.dtype))  # [T, 2H, D]

    def body(i, buf):
        p = t_pages[i]
        s = t_slots[i]
        ins = kv_ev[i][None, None, :, :]
        return lax.dynamic_update_slice(buf, ins, (p, s, 0, 0))

    return lax.fori_loop(0, K, body, kv_pages)


class SeqDecodingParams(eqx.Module):
    """Per-sequence decoding parameters."""

    max_num_tokens: jnp.ndarray
    stop_tokens: ht.i32[NamedArray, "stop_seq position"] | None
    temperature: jnp.ndarray
    key: jaxtyping.PRNGKeyArray

    @staticmethod
    def default() -> "SeqDecodingParams":
        """
        Returns a default SeqDecodingParams with the given number of stop sequences and maximum stop tokens.
        """
        max_int_jnp = jnp.iinfo(jnp.int32).max
        return SeqDecodingParams(
            max_num_tokens=jnp.array(max_int_jnp - 100000, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(0),
        )


@dataclasses.dataclass(frozen=True)
class PageTableSpec:
    """Lightweight description of the layout required for allocating paged KV caches."""

    num_pages: int
    page_size: int
    max_seqs: int

    @property
    def pages_per_seq(self) -> int:
        return self.num_pages // self.max_seqs

    @property
    def tokens_per_seq(self) -> int:
        return self.pages_per_seq * self.page_size

    def pages_needed_for_prompt(self, prompt_len: int) -> int:
        size = int(self.page_size)
        return (int(prompt_len) + size - 1) // size


class PageCache(eqx.Module):
    """Abstract base for paged attention caches."""


PageCacheT = TypeVar("PageCacheT", bound=PageCache)


class ListCache(PageCache, Generic[PageCacheT]):
    """Container cache that delegates operations to a sequence of caches."""

    caches: tuple[PageCacheT, ...]

    def __post_init__(self):
        object.__setattr__(self, "caches", tuple(self.caches))

    @staticmethod
    def from_iterable(caches: Iterable[PageCacheT]) -> "ListCache[PageCacheT]":
        return ListCache(tuple(caches))

    def __len__(self) -> int:
        return len(self.caches)

    def __iter__(self) -> Iterator[PageCacheT]:
        return iter(self.caches)

    def __getitem__(self, idx: int) -> PageCacheT:
        return self.caches[idx]

    def replace(self, idx: int, value: PageCacheT) -> "ListCache[PageCacheT]":
        caches = list(self.caches)
        caches[idx] = value
        return ListCache(tuple(caches))


class BatchInfo(eqx.Module):
    """Information for where to store paged KV values."""

    kv_cache: PageCache
    page_size: int = eqx.field(static=True)

    num_seqs: jnp.ndarray

    seq_lens: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    """The length of each sequence."""

    tokens: ht.i32[NamedArray, " seq position"]  # type: ignore[name-defined]
    """The token IDs for each sequence in the batch."""

    pos_ids: ht.i32[NamedArray, " seq position"]  # type: ignore[name-defined]
    """The position IDs for each sequence in the batch."""

    cu_q_lens: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    """..."""

    new_token_dests: NamedArray
    """(num_seq, seq_len) array containing target locations for KV cache updates.

    page = v // page_size,  offset_in_page = v % page_size
    """

    page_indices: NamedArray

    def pages_and_slots(self):
        token_dests = self.new_token_dests

        t_pages = hax.where(is_valid(token_dests), token_dests // self.page_size, INVALID)
        t_slots = hax.where(is_valid(token_dests), token_dests % self.page_size, INVALID)

        return t_pages, t_slots


class KvPageCache(PageCache):
    """Concrete KV cache storing interleaved key/value pages for paged attention."""

    kv_pages: NamedArray  # [Page, Slot, 2 * KVHeads, Embed]

    @staticmethod
    def init(spec: PageTableSpec, kv_heads: Axis, head_size: Axis, dtype=jnp.float32) -> "KvPageCache":
        """
        Initialize a KvPageCache with the given page table specification and dimensions.

        Args:
            spec: The layout specification for KV pages.
            kv_heads: Axis for key/value heads.
            head_size: Axis for head size.
            dtype: Data type for the cache.
        """
        kv_pages = hax.zeros(
            {
                "page": spec.num_pages,
                "slot": spec.page_size,
                "kv_head": 2 * kv_heads.size,
                head_size.name: head_size.size,
            },
            dtype=dtype,
        )
        return KvPageCache(kv_pages)

    @named_call
    def update(
        self,
        new_k: NamedArray,  # [Tok, KvHeads, HeadDim]
        new_v: NamedArray,  # [Tok, KvHeads, HeadDim]
        batch_info: BatchInfo,
    ) -> "KvPageCache":
        """Append keys and values to the cache based on *batch_info*."""
        # K should be the number of tokens, not the number of sequences
        K = jnp.asarray(new_k.array.shape[0], jnp.int32)
        t_pages, t_slots = batch_info.pages_and_slots()  # [T] int32 (first K valid)

        jax.debug.print(
            "[KV_UPDATE] K={k} new_token_dests={d} t_pages={p} t_slots={s}",
            k=K,
            d=batch_info.new_token_dests.array,
            p=t_pages.array,
            s=t_slots.array,
        )

        updated = kv_update_unified_prefix(
            self.kv_pages.array,
            t_pages.astype(jnp.int32).array,
            t_slots.astype(jnp.int32).array,
            new_k.array,
            new_v.array,
            K,
        )
        updated = NamedArray(updated, self.kv_pages.axes)
        return dataclasses.replace(self, kv_pages=updated)


class DecodeState(eqx.Module):
    """Decoding state for a batch of sequences."""

    page_spec: PageTableSpec = eqx.field(static=True)

    seq_lens: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    """The length of each sequence."""

    cu_q_lens: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    """The cumulative lengths for the sequences, including new tokens."""

    tokens: ht.i32[NamedArray, "position"]  # type: ignore[name-defined]
    """The token IDs for each sequence."""

    logprobs: ht.f32[NamedArray, "position"]  # type: ignore[name-defined]
    """The log probabilities for each sequence."""

    pos_ids: ht.i32[NamedArray, "position"]  # type: ignore[name-defined]
    """The position of each token in `tokens` in the sequence."""

    offset: jnp.ndarray
    """Offset in the iteration space for storing KV pages."""

    finished: ht.bool_[NamedArray, " seq"]  # type: ignore[name-defined]
    """Whether each sequence has completed generation."""

    def update_tokens(self, step: jnp.ndarray, new_tokens: NamedArray, new_logprobs: NamedArray):
        jax.debug.print(
            "[UPDATE_TOKENS] step={s} new_tokens={t} self.tokens_before={tb} self.pos_ids_before={pb}",
            s=step,
            t=new_tokens.array,
            tb=self.tokens.array,
            pb=self.pos_ids.array,
        )

        tokens = new_tokens
        new_logprobs = new_logprobs
        seq_lens = self.seq_lens + 1
        pos_ids = self.pos_ids + 1
        # cu_q_lens is static for decode: always [0, 1, 2, ..., N] for N sequences
        # It represents offsets into the current batch's token array, not global positions
        cu_q_lens = hax.named(jnp.arange(self.num_seqs + 1, dtype=jnp.int32), self.cu_q_lens.axes)

        jax.debug.print(
            "[UPDATE_TOKENS] tokens_after={ta} pos_ids_after={pa} seq_lens_after={sla}",
            ta=tokens.array,
            pa=pos_ids.array,
            sla=seq_lens.array,
        )

        return dataclasses.replace(
            self,
            tokens=tokens,
            logprobs=new_logprobs,
            pos_ids=pos_ids,
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            finished=self.finished,
        )

    @property
    def num_seqs(self):
        return self.seq_lens.shape["seq"]

    def batch_info(self, inner_iteration: jnp.ndarray, kv_cache: KvPageCache):
        iteration = inner_iteration + self.offset

        token_dests = jnp.arange(self.num_seqs * self.page_spec.tokens_per_seq, dtype=jnp.int32).reshape(
            self.page_spec.tokens_per_seq, self.num_seqs
        )

        page_indices = hax.NamedArray(
            (token_dests // self.page_spec.page_size).reshape(self.num_seqs, -1),
            {"seq": self.num_seqs, "page": self.page_spec.tokens_per_seq},
        )

        jax.debug.print(
            "[BATCH_INFO_BUILD] inner_iter={i} offset={o} iteration={it}",
            i=inner_iteration,
            o=self.offset,
            it=iteration,
        )
        jax.debug.print(
            "[BATCH_INFO_BUILD] self.tokens={t} self.pos_ids={p} self.seq_lens={sl}",
            t=self.tokens.array,
            p=self.pos_ids.array,
            sl=self.seq_lens.array,
        )

        return BatchInfo(
            kv_cache=kv_cache,
            page_size=self.page_spec.page_size,
            cu_q_lens=self.cu_q_lens,
            page_indices=page_indices,
            num_seqs=self.seq_lens.shape["seq"],
            seq_lens=self.seq_lens,
            tokens=self.tokens,
            pos_ids=self.pos_ids,
            new_token_dests=hax.NamedArray(token_dests[iteration], {"position": self.num_seqs}),
        )
