# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import haliax as hax
import pytest

from levanter.inference.jit_scheduler import SequenceTable, TokenQueue
from levanter.inference.page_table import PageTable


def test_pack_next_sequence_single_seq_boundary_at_last_token():
    # Queue entirely filled with a single sequence id; pack exactly all tokens.
    capacity = 8
    tq = TokenQueue.init(capacity)

    tokens = hax.named(jnp.arange(capacity, dtype=jnp.int32), axis=("position",))
    slot_ids = hax.named(jnp.full((capacity,), 0, dtype=jnp.int32), axis=("position",))

    # Absolute pos_ids for a single sequence: 0..capacity-1
    pos_ids = hax.named(jnp.arange(capacity, dtype=jnp.int32), axis=("position",))
    tq = tq.enqueue_tokens(tokens, slot_ids, pos_ids, capacity)

    tq2, packed = tq.pack_next_sequence(capacity)

    # Queue should be empty now
    assert tq2.num_queued_tokens == 0

    # Determine boundaries based on PageTable seq_lens after allocation
    assert int(packed.num_tokens) == capacity
    pt = PageTable.init(max_pages=16, max_seqs=4, page_size=8, max_pages_per_seq=4)
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)
    sequences, _ = sequences.reserve_slot(0)
    sequences, pt, binfo = sequences.allocate_for_seq(pt, packed.slot_ids, packed.pos_ids)
    seq_lens_after = binfo.seq_lens["seq", packed.slot_ids]
    boundary_mask = packed.pos_ids == (seq_lens_after - 1)
    # Expect exactly one boundary at the last token
    bm = boundary_mask.array
    assert bm.dtype == jnp.bool_
    assert bool(bm[-1]) is True
    assert int(bm.sum()) == 1


@pytest.mark.parametrize("seq_ids", [[0, 1], [1, 0]])
def test_pack_next_sequence_boundaries_between_sequences(seq_ids):
    # Two sequences back-to-back; boundaries at the last token of each sequence in the packed slice.
    capacity = 7
    tq = TokenQueue.init(capacity)
    seq1, seq2 = seq_ids

    tokens = hax.named(jnp.array([10, 11, 12, 20, 21, 22, 23], dtype=jnp.int32), axis=("position",))
    slot_ids = hax.named(jnp.array([seq1, seq1, seq1, seq2, seq2, seq2, seq2], dtype=jnp.int32), axis=("position",))

    # Absolute pos_ids are per-sequence; start fresh at 0 for each sequence in this test
    pos_ids = hax.named(jnp.array([0, 1, 2, 0, 1, 2, 3], dtype=jnp.int32), axis=("position",))
    tq = tq.enqueue_tokens(tokens, slot_ids, pos_ids, capacity)

    tq2, packed = tq.pack_next_sequence(capacity)

    assert int(packed.num_tokens) == capacity
    pt = PageTable.init(max_pages=16, max_seqs=4, page_size=8, max_pages_per_seq=4)
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)
    sequences, assigned1 = sequences.reserve_slot(seq1)
    assert int(assigned1) == seq1
    sequences, assigned2 = sequences.reserve_slot(seq2)
    assert int(assigned2) == seq2
    sequences, pt, binfo = sequences.allocate_for_seq(pt, packed.slot_ids, packed.pos_ids)
    seq_lens_after = binfo.seq_lens["seq", packed.slot_ids]

    boundary_mask = packed.pos_ids == (seq_lens_after - 1)
    bm = boundary_mask.array
    if seq1 == 0:
        # Boundaries at positions 2 and 6
        assert bool(bm[2]) is True
        assert bool(bm[6]) is True
    else:
        # Boundaries at positions 2 and 5
        assert bool(bm[3]) is True
        assert bool(bm[6]) is True

    assert int(bm.sum()) == 2


# NOTE: Tests for device-side allocation removed - allocation now happens on CPU
# via set_page_assignments() before calling allocate_for_seq()
# See test_checkpoint3_device_integration.py and test_checkpoint4_allocation_bypass.py
# for tests of the new CPU-side allocation flow
