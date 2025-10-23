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

    # the true number of (non-padded) sequences in the batch
    num_seqs: jnp.ndarray

    @property
    def max_seqs(self) -> int:
        return self.seq_lens.shape["seq"]

    seq_lens: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    """The length of each sequence, padded with zeros."""

    tokens: ht.i32[NamedArray, " seq position"]  # type: ignore[name-defined]
    """The token IDs for each sequence in the batch, padded with INVALID."""

    pos_ids: ht.i32[NamedArray, " seq position"]  # type: ignore[name-defined]
    """The position IDs for each sequence in the batch, padded with zeros."""

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
        # K should be the number of VALID tokens (excluding padding)
        # Count non-INVALID entries in new_token_dests
        K = jnp.sum(is_valid(batch_info.new_token_dests).astype(jnp.int32).array)
        t_pages, t_slots = batch_info.pages_and_slots()  # [T] int32 (first K valid)
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

    num_seqs: jnp.ndarray
    """The number of sequences in the batch."""

    page_spec: PageTableSpec = eqx.field(static=True)

    token_dests: ht.i32[NamedArray, " seq position"]  # type: ignore[name-defined]
    """The target locations for each token in the KV cache."""

    seq_lens: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    """The length of each sequence."""

    # N.B. The _query_ vector is recomputed each step. So cu_q_lens does _not_ refer
    # to the sequence length across the entire decode, but only the current step.
    cu_q_lens: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    """The cumulative lengths for the sequences."""

    tokens: ht.i32[NamedArray, "position"]  # type: ignore[name-defined]
    """The token IDs for each sequence."""

    logprobs: ht.f32[NamedArray, "position"]  # type: ignore[name-defined]
    """The log probabilities for each sequence."""

    pos_ids: ht.i32[NamedArray, "position"]  # type: ignore[name-defined]
    """The position of each token in `tokens` in the sequence."""

    finished: ht.bool_[NamedArray, " seq"]  # type: ignore[name-defined]
    """Whether each sequence has completed generation."""

    @property
    def max_seqs(self) -> int:
        return self.page_spec.max_seqs

    def __post_init__(self):
        assert self.token_dests.shape["seq"] == self.page_spec.max_seqs, (self.token_dests, self.page_spec)
        assert self.tokens.shape["position"] <= self.page_spec.tokens_per_seq * self.page_spec.max_seqs, (self.tokens.shape, self.page_spec)
        assert self.tokens.shape["position"] >= self.max_seqs, (self.tokens.shape, self.page_spec)


    def update_tokens(self, step: jnp.ndarray, new_tokens: NamedArray, new_logprobs: NamedArray):
        seq_lens = jnp.where(jnp.arange(self.max_seqs) < self.num_seqs, self.seq_lens.array + 1, self.seq_lens.array) # type: ignore
        pos_ids = jnp.where(jnp.arange(self.max_seqs) < self.num_seqs, self.pos_ids.array + 1, self.pos_ids.array) # type: ignore

        # jax.debug.print(
        #     "[UPDATE_TOKENS] tokens_after={ta} pos_ids_after={pa} seq_lens_after={sla}",
        #     ta=new_tokens.array,
        #     pa=pos_ids,
        #     sla=seq_lens,
        # )

        return dataclasses.replace(
            self,
            tokens=new_tokens,
            logprobs=new_logprobs,
            pos_ids=hax.named(pos_ids, self.pos_ids.axes),
            seq_lens=hax.named(seq_lens, self.seq_lens.axes),
            finished=self.finished,
        )

    def batch_info(self, kv_cache: KvPageCache, prefill: bool = False) -> BatchInfo:
        page_indices = self.token_dests // self.page_spec.page_size
        # page_indices are per-page, and pages are assumed to be fully filled
        page_indices = page_indices.array[:, :: self.page_spec.page_size]
        page_indices = hax.NamedArray(
            page_indices,
            {"seq": page_indices.shape[0], "page": page_indices.shape[1]},
        )

        # generate an array of shape [tokens] with the appropriate target locations for each KV update
        # during decode, this is just a lookup, during prefill, we have to scatter from the token_dests
        # static array into the correct positions. we do this with a where and using 0 as a sentinel.
        num_tokens = self.tokens.shape["position"]
        tokens_per_seq = self.page_spec.tokens_per_seq
        pad_len = max(num_tokens - tokens_per_seq, 0)

        def fill_seq(i, dests):
            # where we are writing into `dests`
            slice_start = self.cu_q_lens[{"seq": i}].array
            slice_end = self.cu_q_lens[{"seq": i + 1}].array
            seq_len = self.seq_lens[{"seq": i}].array
            
            # we start writing from the beginning of the sequence's slice in token_dests
            # backing out by the number of tokens in our current sequence.
            seq_start = 0 if prefill else seq_len
            source_values = self.token_dests[{"seq": i}].array
            # roll our source values to start from 0
            source_values = jnp.roll(source_values, -seq_start)

            # we need to pad or slice source_values to be the same size as `dests` (#tokens)
            if pad_len > 0:
                source_values = jnp.pad(source_values, (0, pad_len), constant_values=0)
            else:
                source_values = source_values[:num_tokens]

            # now we need to roll our targets to align with the current sequence's slice
            source_values = jnp.roll(source_values, slice_start)

            # jax.debug.print("Copying {seq_start}:{seq_len} tokens into dests[{slice_start}:{slice_end}]",
            #     seq_start=seq_start,
            #     seq_len=seq_start + seq_len,
            #     slice_start=slice_start,
            #     slice_end=slice_end,
            # )
            dest_valid = jnp.arange(num_tokens) >= slice_start
            dest_valid = dest_valid & (jnp.arange(num_tokens) < slice_end)
            return jnp.where(dest_valid, source_values, dests)

        new_token_dests_raw = jnp.full(num_tokens, INVALID, dtype=jnp.int32)
        new_token_dests_raw = lax.fori_loop(0, self.page_spec.max_seqs, fill_seq, new_token_dests_raw)
        new_token_dests = hax.NamedArray(new_token_dests_raw, self.tokens.axes)

        # jax.debug.print(
        #     "[BATCH_INFO_BUILD]  tokens={t} pos_ids={p} seq_lens={sl} new_token_dests={ntd}, cu_q_lens={cu}",
        #     t=self.tokens.array,
        #     p=self.pos_ids.array,
        #     sl=self.seq_lens.array,
        #     ntd=new_token_dests_raw,
        #     cu=self.cu_q_lens.array,
        # )

        return BatchInfo(
            num_seqs=self.num_seqs,
            kv_cache=kv_cache,
            page_size=self.page_spec.page_size,
            cu_q_lens=self.cu_q_lens,
            page_indices=page_indices,
            seq_lens=self.seq_lens,
            tokens=self.tokens,
            pos_ids=self.pos_ids,
            new_token_dests=new_token_dests,
        )
