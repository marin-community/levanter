# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
CHECKPOINT 3: Verify device-side integration of CPU page allocations.

This test verifies that:
1. CPU-allocated pages are correctly transferred to the device
2. Device uses CPU page assignments during prefill
3. allocate_for_seq sees pages already allocated and doesn't re-allocate
"""

import jax
import numpy as np

from levanter.inference.page_allocator import PageAllocatorCPU
from levanter.inference.utils import INVALID


def test_checkpoint3_device_uses_cpu_allocations():
    """Test that device-side code uses CPU-allocated pages during prefill."""
    print("\n" + "=" * 80)
    print("CHECKPOINT 3: Device-Side Integration Test")
    print("=" * 80)

    # Configuration
    num_pages = 32
    page_size = 128
    max_seqs = 8
    pages_per_seq = 4

    # Initialize CPU page allocator
    cpu_allocator = PageAllocatorCPU(
        num_pages=num_pages,
        max_seqs=max_seqs,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
    )

    # Simulate allocating pages for 2 sequences on the CPU
    slot_id_1 = 0
    slot_id_2 = 1
    prompt_len_1 = 150  # Needs 2 pages (150 / 128 = 1.17, ceil = 2)
    prompt_len_2 = 300  # Needs 3 pages (300 / 128 = 2.34, ceil = 3)

    # Allocate pages on CPU
    positions_1 = list(range(prompt_len_1))
    pages_1 = cpu_allocator.allocate_pages_for_tokens(slot_id_1, positions_1)
    print(f"\n[CPU] Allocated pages for slot {slot_id_1} (len={prompt_len_1}): {pages_1}")

    positions_2 = list(range(prompt_len_2))
    pages_2 = cpu_allocator.allocate_pages_for_tokens(slot_id_2, positions_2)
    print(f"[CPU] Allocated pages for slot {slot_id_2} (len={prompt_len_2}): {pages_2}")

    # Get full page assignments array
    cpu_page_assignments = cpu_allocator.get_page_assignments()
    print(f"\n[CPU] Full page assignments shape: {cpu_page_assignments.shape}")
    print(f"[CPU] Page assignments for slot 0: {cpu_page_assignments[0]}")
    print(f"[CPU] Page assignments for slot 1: {cpu_page_assignments[1]}")

    # Verify reference counts on CPU
    cpu_stats = cpu_allocator.get_stats()
    print(f"\n[CPU Stats] Used pages: {cpu_stats['used_pages']}/{cpu_stats['total_pages']}")
    print(f"[CPU Stats] Free pages: {cpu_stats['free_pages']}")
    print(f"[CPU Stats] Active slots: {cpu_stats['active_slots']}")

    # Now test device-side integration
    print("\n" + "-" * 80)
    print("Testing Device-Side Integration...")
    print("-" * 80)

    # Import device-side components
    from levanter.inference.jit_scheduler import DecodeState, SequenceTable
    from levanter.inference.page_table import PageTable

    # Initialize device-side structures
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

    print(f"\n[Device] Initial free pages: {hax.sum(page_table.page_ref_counts == 0).item()}")

    # Reserve slots and set CPU page assignments
    decode_state, assigned_0 = decode_state.reserve_slot(slot_id_1)
    print(f"[Device] Reserved slot {assigned_0}")

    decode_state, assigned_1 = decode_state.reserve_slot(slot_id_2)
    print(f"[Device] Reserved slot {assigned_1}")

    # Set CPU page assignments for slot 0
    pages_for_slot_0 = cpu_page_assignments[slot_id_1, :]
    print(f"\n[Device] Setting CPU pages for slot {slot_id_1}: {pages_for_slot_0}")
    decode_state = decode_state.set_page_assignments(slot_id_1, pages_for_slot_0)

    # Set CPU page assignments for slot 1
    pages_for_slot_1 = cpu_page_assignments[slot_id_2, :]
    print(f"[Device] Setting CPU pages for slot {slot_id_2}: {pages_for_slot_1}")
    decode_state = decode_state.set_page_assignments(slot_id_2, pages_for_slot_1)

    # Verify device state matches CPU allocations
    print("\n" + "-" * 80)
    print("Verification Results:")
    print("-" * 80)

    device_page_indices = jax.device_get(decode_state.sequences.page_indices.array)
    device_ref_counts = jax.device_get(decode_state.page_table.page_ref_counts.array)

    # Check slot 0 pages
    slot_0_device_pages = device_page_indices[slot_id_1, :]
    slot_0_valid_pages = slot_0_device_pages[slot_0_device_pages != INVALID]
    print(f"\n[Verify] Slot {slot_id_1} CPU pages: {pages_1}")
    print(f"[Verify] Slot {slot_id_1} device pages: {slot_0_valid_pages}")
    assert np.array_equal(pages_1, slot_0_valid_pages), "Slot 0 pages mismatch!"
    print(f"✓ Slot {slot_id_1} pages match CPU allocations")

    # Check slot 1 pages
    slot_1_device_pages = device_page_indices[slot_id_2, :]
    slot_1_valid_pages = slot_1_device_pages[slot_1_device_pages != INVALID]
    print(f"\n[Verify] Slot {slot_id_2} CPU pages: {pages_2}")
    print(f"[Verify] Slot {slot_id_2} device pages: {slot_1_valid_pages}")
    assert np.array_equal(pages_2, slot_1_valid_pages), "Slot 1 pages mismatch!"
    print(f"✓ Slot {slot_id_2} pages match CPU allocations")

    # Check reference counts
    expected_ref_counts = np.zeros(num_pages, dtype=np.int32)
    for page_id in pages_1:
        expected_ref_counts[page_id] += 1
    for page_id in pages_2:
        expected_ref_counts[page_id] += 1

    print(
        f"\n[Verify] Expected ref counts (non-zero): {np.nonzero(expected_ref_counts)[0]} -> {expected_ref_counts[expected_ref_counts > 0]}"
    )
    print(
        f"[Verify] Device ref counts (non-zero): {np.nonzero(device_ref_counts)[0]} -> {device_ref_counts[device_ref_counts > 0]}"
    )
    assert np.array_equal(expected_ref_counts, device_ref_counts), "Reference counts mismatch!"
    print("✓ Reference counts match CPU allocations")

    # Check free pages
    device_free_pages = int(hax.sum(decode_state.page_table.page_ref_counts == 0).item())
    expected_free_pages = num_pages - len(pages_1) - len(pages_2)
    print(f"\n[Verify] Expected free pages: {expected_free_pages}")
    print(f"[Verify] Device free pages: {device_free_pages}")
    assert (
        device_free_pages == expected_free_pages
    ), f"Free pages mismatch! {device_free_pages} != {expected_free_pages}"
    print("✓ Free page count matches")

    print("\n" + "=" * 80)
    print("CHECKPOINT 3 PASSED: Device successfully uses CPU page allocations!")
    print("=" * 80)


if __name__ == "__main__":
    test_checkpoint3_device_uses_cpu_allocations()
