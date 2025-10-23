# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# tests/test_ragged_paged_attention.py
import math

import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.random as jrandom
import numpy as np
import pytest
from chex import assert_trees_all_close
from haliax import Axis, NamedArray

from levanter.inference.utils import INVALID
from levanter.layers.attention import AttentionMask, ragged_paged_attention, simple_attention_with_dropout

SLOT = hax.Axis("slot", 4)  # page size
NUM_SLOTS = SLOT.size
KV_HEADS = hax.Axis("kv_head", 1)
QH = hax.Axis("q_heads_per_group", 1)
D = hax.Axis("head_size", 128)

KV_BS = 32  # must match constant inside kernel
SM_SCALE = 1 / math.sqrt(D.size)


def _build_random_case(rng, seq_lens):
    """Seq‑by‑seq contiguous paging with padding to KV_BS."""
    num_seqs = len(seq_lens)

    # token → sequence mapping and cumulative Q lengths
    tok_offsets = np.cumsum([0] + seq_lens)
    cu_q_lens = jnp.asarray(tok_offsets, dtype=jnp.int32)
    cu_q_lens = hax.named(cu_q_lens, "seq")

    # allocate enough pages for the longest sequence, but give each sequence its own slice
    pages_per_seq_raw = [(length + NUM_SLOTS - 1) // NUM_SLOTS for length in seq_lens]
    max_pages = KV_BS * ((max(pages_per_seq_raw) + KV_BS - 1) // KV_BS)  # pad to mult of 32
    page_indices = -jnp.ones((num_seqs, max_pages), dtype=jnp.int32)

    # generate actual page contents
    next_free_page = 0
    kv_data = []
    for sid, (qlen, p_raw) in enumerate(zip(seq_lens, pages_per_seq_raw)):
        pages = jnp.arange(next_free_page, next_free_page + p_raw, dtype=jnp.int32)
        page_indices = page_indices.at[sid, :p_raw].set(pages)
        next_free_page += p_raw

        # fill those pages with random k/v
        kv_pages_for_seq = hax.random.normal(
            jr.fold_in(rng, sid),
            # (hax.Axis("page", p_raw), SLOT, KV_HEADS.resize(2 * KV_HEADS.size), D),
            {"page": p_raw, "slot": NUM_SLOTS, "kv_head": 2 * KV_HEADS.size, "head_size": 128},
        )
        kv_data.append(kv_pages_for_seq)

    # stack pages, pad out to PAGE axis
    # ensure we have enough pages
    # kv_pages = hax.zeros(({PAGE, SLOT, KV_HEADS.resize(2 * KV_HEADS.size), D))
    num_pages_needed = next_free_page + 4
    kv_pages = hax.zeros({"page": num_pages_needed, "slot": 4, "kv_head": 2, "head_size": 128})
    kv_pages = kv_pages.at["page", :next_free_page].set(hax.concatenate("page", kv_data))

    # random queries for whole token axis
    total_tokens = tok_offsets[-1]
    this_position = hax.Axis("position", int(total_tokens))
    q = hax.random.normal(rng, (this_position, KV_HEADS, QH, D))

    kv_lens = jnp.asarray(seq_lens, dtype=jnp.int32)
    kv_lens = hax.named(kv_lens, "seq")

    page_indices = hax.named(page_indices, ("seq", "page"))

    return q, kv_pages, kv_lens, page_indices, cu_q_lens, jnp.array(num_seqs, dtype=jnp.int32)


def _build_incremental_case(rng, seq_lens, k_lens):
    """Like ``_build_random_case`` but query only contains the last ``k`` tokens.

    ``seq_lens`` gives the total tokens already in the KV cache for each
    sequence. ``k_lens`` is how many query tokens each sequence has. The KV
    cache still contains ``seq_lens`` tokens for every sequence.
    """
    q_full, kv_pages, kv_lens, page_indices, full_cu_q_lens, num_seqs = _build_random_case(rng, seq_lens)

    assert len(seq_lens) == len(k_lens)

    chunks = []
    new_offsets = [0]
    for sid, (total_len, k) in enumerate(zip(seq_lens, k_lens)):
        start = int(full_cu_q_lens["seq", sid]) + total_len - k
        chunks.append(q_full["position", hax.ds(start, k)])
        new_offsets.append(new_offsets[-1] + k)

    q = hax.concatenate("position", chunks)
    cu_q_lens = hax.named(jnp.asarray(new_offsets, dtype=jnp.int32), "seq")

    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, seq_lens):
    """Naïve per‑sequence causal soft‑max (slow but tiny)."""
    out_chunks = []
    for sid, qlen in enumerate(seq_lens):
        # slice query tokens for this sequence
        start = int(cu_q_lens["seq", sid])
        q_seq = q["position", hax.ds(start, qlen)]
        TOK_S = hax.Axis("position", qlen)

        # gather kv for the sequence
        n_pages = (kv_lens["seq", sid] + SLOT.size - 1) // SLOT.size
        pages = page_indices["seq", sid, "page", : n_pages.scalar()]
        kv_flat = kv_pages["page", pages, "slot", :].flatten_axes(("page", "slot"), "kv_position")
        kv_flat = kv_flat["kv_position", hax.ds(0, kv_lens["seq", sid].scalar())]

        k_seq = kv_flat["kv_head", 0::2]
        v_seq = kv_flat["kv_head", 1::2]

        # rename axes so they line up with dot_product_attention sig
        q_seq = q_seq.rename({"position": TOK_S.name})

        offset = kv_lens["seq", sid].scalar() - qlen
        mask = AttentionMask.causal(offset=offset)
        ref = simple_attention_with_dropout(
            "position", "kv_position", D, q_seq, k_seq, v_seq, mask=mask, scaling_factor=SM_SCALE
        )

        out_chunks.append(ref)

    return hax.concatenate("position", out_chunks)


# very loose tolerance. JAX uses a very loose tolerance for their ragged_attention tests.
def _rpa_tol() -> float:
    devices = jax.devices()
    # 2%?!?!
    return 2e-2 if any(device.platform == "tpu" for device in devices) else 1e-4


def test_ragged_paged_attention_single_seq():
    with jax.make_mesh((len(jax.devices()),), ("dp",)):
        rng = jr.PRNGKey(0)
        seq_lens = [1]  # one sequence
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_random_case(rng, seq_lens)

    ragged = ragged_paged_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, seq_lens)

    assert ragged.axes == ref.axes
    tol = _rpa_tol()
    for i in range(len(ragged.array)):
        assert_trees_all_close(ragged.array[i], ref.array[i], atol=tol, rtol=tol, custom_message=f" at index {i}")
    # assert_trees_all_close(ragged.array[:-1], ref.array[:-1], atol=_rpa_tol(), rtol=_rpa_tol())
    # assert_trees_all_close(ragged.array[-1], ref.array[-1], atol=_rpa_tol(), rtol=_rpa_tol())
    # assert_trees_all_close(ragged.array, ref.array, atol=_rpa_tol(), rtol=_rpa_tol())


jit_rpa = jax.jit(ragged_paged_attention)


@pytest.mark.parametrize(
    "seq_lens", [[8], [8, 32, 16], [10, 37, 64], [34, 17], [9, 10, 34, 17], [64, 10, 37], [5, 15, 25, 35, 45]]
)
def test_ragged_paged_attention_multi_seq(seq_lens):
    rng = jr.PRNGKey(hash(tuple(seq_lens)))
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_random_case(rng, seq_lens)

    ragged = jit_rpa(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, seq_lens)

    assert ragged.axes == ref.axes
    tol = _rpa_tol()
    assert_trees_all_close(ragged.array, ref.array, atol=tol, rtol=tol)


def test_ragged_paged_attention_incremental_single_seq():
    rng = jr.PRNGKey(2)
    seq_lens = [47]
    k_lens = [5]
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_incremental_case(rng, seq_lens, k_lens)

    ragged = jit_rpa(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, k_lens)

    assert ragged.axes == ref.axes
    tol = _rpa_tol()
    assert_trees_all_close(ragged.array, ref.array, atol=tol, rtol=tol)


def test_ragged_paged_attention_incremental_multi_seq():
    rng = jr.PRNGKey(3)
    seq_lens = [10, 37, 64]
    k_lens = [1, 3, 9]
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_incremental_case(rng, seq_lens, k_lens)

    ragged = jit_rpa(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, k_lens)

    assert ragged.axes == ref.axes
    tol = _rpa_tol()
    assert_trees_all_close(ragged.array, ref.array, atol=tol, rtol=tol)
