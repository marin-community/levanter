# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
from typing import Generic, Iterable, Iterator, Self, Type, TypeVar

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
        return (self.num_pages + self.page_size - 1) // self.page_size

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

    new_token_dests: NamedArray
    """(num_seq, seq_len) array containing target locations for KV cache updates.

    page = v // page_size,  offset_in_page = v % page_size
    """

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
        K = jnp.asarray(batch_info.num_seqs, jnp.int32)
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

    kv_cache: KvPageCache

    seq_lens: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    """The length of each sequence."""

    tokens: ht.i32[NamedArray, " seq position"]  # type: ignore[name-defined]
    """The token IDs for each sequence in the batch."""

    pos_ids: ht.i32[NamedArray, " seq position"]  # type: ignore[name-defined]
    """The position IDs for each sequence in the batch."""

    logprobs: ht.f32[NamedArray, " seq position"]  # type: ignore[name-defined]
    """The log probabilities for each token in the sequences."""

    cu_q_lens: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    """The cumulative lengths for the sequences, including new tokens."""

    iteration: jnp.ndarray
    """The current iteration of the generation loop, used to index token locations."""

    page_size: int = eqx.field(static=True)

    batch_info: BatchInfo

    @property
    def num_seqs(self):
        return self.seq_lens.shape[0]

    @classmethod
    def init(
        cls: Type[Self],
        kv_cache: KvPageCache,
        page_table: PageTableSpec,
        seq_lens: jnp.ndarray,
        tokens: jnp.ndarray,
        pos_ids: jnp.ndarray,
        cu_q_lens: jnp.ndarray,
        batch_info: BatchInfo,
    ) -> Self:
        Seq = hax.Axis("seq", size=len(seq_lens))
        Position = hax.Axis("position", size=tokens.shape[-1])
        return cls(
            iteration=0,
            kv_cache=kv_cache,
            seq_lens=hax.NamedArray(seq_lens, (Seq,)),
            tokens=hax.NamedArray(tokens, (Seq, Position)),
            pos_ids=hax.NamedArray(pos_ids, (Seq, Position)),
            cu_q_lens=hax.NamedArray(cu_q_lens, (Seq)),
            logprobs=hax.zeros((Seq, Position)),
            page_size=page_table.page_size,
            batch_info=batch_info,
        )

    @named_call
    def decode_sequences(self) -> NamedArray:
        """Decode all sequences in the batch from their token ID and length information."""
        pass


class GenState(eqx.Module):
    """Container for generation state used during decoding."""

    cache: PageCache
    decode_state: DecodeState
