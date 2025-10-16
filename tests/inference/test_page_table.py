# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

import haliax as hax

from levanter.inference.utils import INVALID
from levanter.inference.page_table import PageBatchInfo, PageTable
from levanter.inference.jit_scheduler import SequenceTable


def _make_table(pages=8, seqs=4, page_size=2, pages_per_seq=2):
    return PageTable.init(pages, seqs, page_size, pages_per_seq)


def test_page_table_max_len_per_seq():
    pt = _make_table(page_size=2, pages_per_seq=3)
    assert pt.max_len_per_seq == 6


def test_sequence_table_reserve_and_release_slot():
    pt = _make_table()
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)

    sequences, slot_arr = sequences.reserve_slot()
    slot = int(slot_arr)
    assert slot == 0
    assert bool(sequences.used_mask.array[slot])

    sequences = sequences.release_slot(slot)
    assert not bool(sequences.used_mask.array[0])


def test_page_batch_info_shapes():
    seq = hax.Axis("seq", 2)
    page = hax.Axis("page", 3)
    pb = PageBatchInfo(
        slot_ids=hax.arange(seq),
        page_indices=hax.full((seq, page), INVALID, dtype=jnp.int32),
        seq_lens=hax.full((seq,), INVALID, dtype=jnp.int32),
        cu_q_lens=hax.named(jnp.array([0, 1, 2], dtype=jnp.int32), hax.Axis("seq_plus_one", 3)),
        num_seqs=jnp.array(2, dtype=jnp.int32),
        new_token_dests=hax.full((hax.Axis("position", 2),), INVALID, dtype=jnp.int32),
        page_size=2,
    )

    assert pb.page_indices.axes == (seq, page)
    assert pb.seq_lens.axes == (seq,)
    assert pb.cu_q_lens.array.shape[0] == pb.num_seqs + 1


# NOTE: test_sequence_table_allocate_and_free_pages removed - device-side allocation removed
# Allocation now happens on CPU via PageAllocatorCPU + set_page_assignments()
# See test_checkpoint3_device_integration.py for CPU allocation tests
