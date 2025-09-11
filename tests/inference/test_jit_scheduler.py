import jax.numpy as jnp
import haliax as hax

from levanter.inference.jit_scheduler import TokenQueue
from levanter.inference.page_table import PageTable


def test_pack_next_sequence_single_seq_boundary_at_last_token():
    # Queue entirely filled with a single sequence id; pack exactly all tokens.
    capacity = 8
    tq = TokenQueue.init(capacity)

    tokens = hax.named(jnp.arange(capacity, dtype=jnp.int32), axis=("position",))
    seq_ids = hax.named(jnp.full((capacity,), 0, dtype=jnp.int32), axis=("position",))

    # Absolute pos_ids for a single sequence: 0..capacity-1
    pos_ids = hax.named(jnp.arange(capacity, dtype=jnp.int32), axis=("position",))
    tq = tq.enqueue_tokens(tokens, seq_ids, pos_ids, capacity)

    tq2, packed = tq.pack_next_sequence(capacity)

    # Queue should be empty now
    assert tq2.num_queued_tokens == 0

    # Determine boundaries based on PageTable seq_lens after allocation
    assert int(packed.num_tokens) == capacity
    pt = PageTable.init(max_pages=16, max_seqs=4, page_size=8, max_pages_per_seq=4)
    # activate seq 0
    pt, _ = pt.assign_seq_id_to_seq()
    pt, binfo = pt.allocate_for_seq(packed.seq_ids)
    seq_lens_after = binfo.seq_lens["seq", packed.seq_ids]
    boundary_mask = packed.pos_ids == (seq_lens_after - 1)
    # Expect exactly one boundary at the last token
    bm = boundary_mask.array
    assert bm.dtype == jnp.bool_
    assert bool(bm[-1]) is True
    assert int(bm.sum()) == 1


def test_pack_next_sequence_boundaries_between_sequences():
    # Two sequences back-to-back; boundaries at the last token of each sequence in the packed slice.
    capacity = 6
    tq = TokenQueue.init(capacity)

    tokens = hax.named(jnp.array([10, 11, 12, 20, 21, 22], dtype=jnp.int32), axis=("position",))
    seq_ids = hax.named(jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32), axis=("position",))

    # Absolute pos_ids are per-sequence; start fresh at 0 for each sequence in this test
    pos_ids = hax.named(jnp.array([0, 1, 2, 0, 1, 2], dtype=jnp.int32), axis=("position",))
    tq = tq.enqueue_tokens(tokens, seq_ids, pos_ids, capacity)

    tq2, packed = tq.pack_next_sequence(capacity)

    assert int(packed.num_tokens) == capacity
    pt = PageTable.init(max_pages=16, max_seqs=4, page_size=8, max_pages_per_seq=4)
    # activate seq 0 and 1
    pt, _ = pt.assign_seq_id_to_seq()
    pt, _ = pt.assign_seq_id_to_seq()
    pt, binfo = pt.allocate_for_seq(packed.seq_ids)
    seq_lens_after = binfo.seq_lens["seq", packed.seq_ids]
    boundary_mask = packed.pos_ids == (seq_lens_after - 1)
    bm = boundary_mask.array
    # Boundaries at positions 2 and 5
    assert bool(bm[2]) is True
    assert bool(bm[5]) is True
    assert int(bm.sum()) == 2
