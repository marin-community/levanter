# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
CHECKPOINT 4: Verify device-side allocation is bypassed when CPU pre-allocates.

This test verifies that:
1. allocate_for_seq does NOT allocate new pages when CPU has already allocated
2. The device allocation loop is effectively skipped (zero iterations)
3. PageBatchInfo is still correctly created from existing page assignments
4. Free page count remains unchanged after allocate_for_seq call
"""

import jax
import jax.numpy as jnp
import numpy as np

from levanter.inference.page_allocator import PageAllocatorCPU
from levanter.inference.utils import INVALID


def test_checkpoint4_device_skips_allocation():
    """Test that device-side allocate_for_seq skips allocation when CPU pre-allocated."""
    print("\n" + "=" * 80)
    print("CHECKPOINT 4: Device Allocation Bypass Test")
    print("=" * 80)

    # Configuration
    num_pages = 64
    page_size = 128
    max_seqs = 8
    pages_per_seq = 8

    # Initialize CPU page allocator
    cpu_allocator = PageAllocatorCPU(
        num_pages=num_pages,
        max_seqs=max_seqs,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
    )

    # Allocate pages for 3 sequences on CPU
    slot_ids = [0, 1, 2]
    prompt_lengths = [200, 350, 150]  # Different lengths requiring different page counts

    print("\n" + "-" * 80)
    print("Phase 1: CPU Pre-Allocation")
    print("-" * 80)

    for slot_id, prompt_len in zip(slot_ids, prompt_lengths):
        positions = list(range(prompt_len))
        pages = cpu_allocator.allocate_pages_for_tokens(slot_id, positions)
        pages_needed = (prompt_len + page_size - 1) // page_size
        print(f"[CPU] Slot {slot_id}: allocated {len(pages)} pages for {prompt_len} tokens (needed {pages_needed})")

    cpu_stats = cpu_allocator.get_stats()
    print(f"\n[CPU Stats] Used: {cpu_stats['used_pages']}/{cpu_stats['total_pages']}, Free: {cpu_stats['free_pages']}")

    cpu_page_assignments = cpu_allocator.get_page_assignments()

    # Initialize device structures
    print("\n" + "-" * 80)
    print("Phase 2: Device Initialization")
    print("-" * 80)

    from levanter.inference.jit_scheduler import DecodeState, SequenceTable
    from levanter.inference.page_table import PageTable

    page_table = PageTable.init(num_pages, max_seqs, page_size, pages_per_seq)
    sequence_table = SequenceTable.init(max_seqs, pages_per_seq, page_size)
    decode_state = DecodeState.init(
        page_table,
        max_stop_seqs=0,
        max_stop_tokens=0,
        max_queued_tokens=512,
        enable_logprobs=False,
    )

    import haliax as hax

    initial_free_pages = int(hax.sum(decode_state.page_table.page_ref_counts == 0).item())
    print(f"[Device] Initial free pages: {initial_free_pages}")

    # Reserve slots and set CPU page assignments
    print("\n" + "-" * 80)
    print("Phase 3: Transfer CPU Allocations to Device")
    print("-" * 80)

    for slot_id in slot_ids:
        decode_state, assigned = decode_state.reserve_slot(slot_id)
        print(f"[Device] Reserved slot {slot_id}")

        pages_for_slot = cpu_page_assignments[slot_id, :]
        valid_pages = pages_for_slot[pages_for_slot != INVALID]
        print(f"[Device] Setting {len(valid_pages)} CPU-allocated pages for slot {slot_id}")
        decode_state = decode_state.set_page_assignments(slot_id, pages_for_slot)

    free_pages_after_set = int(hax.sum(decode_state.page_table.page_ref_counts == 0).item())
    print(f"\n[Device] Free pages after set_page_assignments: {free_pages_after_set}")
    print(f"[Device] Pages consumed by set_page_assignments: {initial_free_pages - free_pages_after_set}")

    # Verify device state matches CPU
    device_page_indices = jax.device_get(decode_state.sequences.page_indices.array)
    for slot_id in slot_ids:
        cpu_pages = cpu_page_assignments[slot_id, :]
        cpu_valid = cpu_pages[cpu_pages != INVALID]
        device_pages = device_page_indices[slot_id, :]
        device_valid = device_pages[device_pages != INVALID]
        assert np.array_equal(cpu_valid, device_valid), f"Page mismatch for slot {slot_id}!"
    print("✓ Device page assignments match CPU allocations")

    # Now call allocate_for_seq - this should NOT allocate any new pages
    print("\n" + "-" * 80)
    print("Phase 4: Call allocate_for_seq (Should Skip Allocation)")
    print("-" * 80)

    # Prepare token arrays as if we're doing prefill
    # Need to create NamedArrays for allocate_for_seq
    token_slot_ids_raw = jnp.array(
        [slot_ids[0]] * prompt_lengths[0] + [slot_ids[1]] * prompt_lengths[1] + [slot_ids[2]] * prompt_lengths[2],
        dtype=jnp.int32,
    )

    # Create position arrays
    token_pos_ids_raw = jnp.concatenate(
        [
            jnp.arange(prompt_lengths[0], dtype=jnp.int32),
            jnp.arange(prompt_lengths[1], dtype=jnp.int32),
            jnp.arange(prompt_lengths[2], dtype=jnp.int32),
        ]
    )

    # Convert to NamedArrays
    from haliax import Axis

    Position = Axis("position", len(token_slot_ids_raw))
    token_slot_ids = hax.named(token_slot_ids_raw, Position)
    token_pos_ids = hax.named(token_pos_ids_raw, Position)

    print(f"[Test] Calling allocate_for_seq with {Position.size} tokens across {len(slot_ids)} sequences")
    print(f"[Test] Sequence lengths: {prompt_lengths}")

    # Call allocate_for_seq - THE KEY TEST
    new_decode_state, batch_info = decode_state.allocate_for_seq(
        token_slot_ids=token_slot_ids, token_pos_ids=token_pos_ids
    )

    free_pages_after_allocate = int(hax.sum(new_decode_state.page_table.page_ref_counts == 0).item())
    print(f"\n[Device] Free pages after allocate_for_seq: {free_pages_after_allocate}")

    # Verify critical property: NO NEW PAGES ALLOCATED
    print("\n" + "-" * 80)
    print("Phase 5: Verification")
    print("-" * 80)

    pages_allocated_by_device = free_pages_after_set - free_pages_after_allocate
    print(f"[Verify] Pages allocated by allocate_for_seq: {pages_allocated_by_device}")

    if pages_allocated_by_device == 0:
        print("✓ SUCCESS: allocate_for_seq did NOT allocate any new pages!")
        print("  This confirms the device allocation loop was bypassed.")
    else:
        print(f"✗ FAILURE: allocate_for_seq allocated {pages_allocated_by_device} pages!")
        print("  This means the device is still doing redundant allocation.")
        assert False, f"Expected 0 new pages, but {pages_allocated_by_device} were allocated"

    # Verify PageBatchInfo was still created correctly
    print("\n[Verify] Checking PageBatchInfo correctness...")
    print(f"  batch_info.slot_ids shape: {batch_info.slot_ids.array.shape}")
    print(f"  batch_info.page_indices shape: {batch_info.page_indices.array.shape}")
    print(f"  batch_info.seq_lens shape: {batch_info.seq_lens.array.shape}")
    print(f"  batch_info.num_seqs: {batch_info.num_seqs}")

    # Check that num_seqs is correct
    expected_num_seqs = len(slot_ids)
    actual_num_seqs = int(batch_info.num_seqs)
    print(f"  Expected num_seqs: {expected_num_seqs}, actual: {actual_num_seqs}")
    assert actual_num_seqs == expected_num_seqs, f"num_seqs mismatch: {actual_num_seqs} != {expected_num_seqs}"

    # Check that seq_lens reflect the updated lengths
    batch_seq_lens = jax.device_get(batch_info.seq_lens.array)
    batch_slot_ids = jax.device_get(batch_info.slot_ids.array)
    for i in range(actual_num_seqs):
        slot_id = batch_slot_ids[i]
        expected_len = prompt_lengths[slot_ids.index(slot_id)]
        actual_len = batch_seq_lens[i]
        print(f"  Batch slot {i} (device slot {slot_id}): seq_len = {actual_len} (expected {expected_len})")
        assert actual_len == expected_len, f"seq_len mismatch for slot {slot_id}"

    print("✓ PageBatchInfo is correctly constructed")

    # Verify page assignments haven't changed
    new_device_page_indices = jax.device_get(new_decode_state.sequences.page_indices.array)
    for slot_id in slot_ids:
        old_pages = device_page_indices[slot_id, :]
        new_pages = new_device_page_indices[slot_id, :]
        assert np.array_equal(old_pages, new_pages), f"Page assignments changed for slot {slot_id}!"
    print("✓ Page assignments unchanged by allocate_for_seq")

    # Verify reference counts haven't changed
    old_ref_counts = jax.device_get(decode_state.page_table.page_ref_counts.array)
    new_ref_counts = jax.device_get(new_decode_state.page_table.page_ref_counts.array)
    assert np.array_equal(old_ref_counts, new_ref_counts), "Reference counts changed!"
    print("✓ Reference counts unchanged by allocate_for_seq")

    print("\n" + "=" * 80)
    print("CHECKPOINT 4 PASSED: Device allocation successfully bypassed!")
    print("Key findings:")
    print("  - allocate_for_seq recognized pre-allocated pages")
    print("  - Zero new pages were allocated on device")
    print("  - PageBatchInfo correctly constructed from existing assignments")
    print("  - All page metadata preserved")
    print("=" * 80)


if __name__ == "__main__":
    test_checkpoint4_device_skips_allocation()
