# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from levanter.inference.page_allocator import PageAllocatorCPU
from levanter.inference.utils import INVALID


def test_page_allocator_init():
    """Test basic initialization"""
    allocator = PageAllocatorCPU(num_pages=10, max_seqs=4, page_size=128, pages_per_seq=3)

    assert allocator.num_pages == 10
    assert allocator.page_size == 128
    assert allocator.max_seqs == 4
    assert allocator.pages_per_seq == 3
    assert allocator.get_free_page_count() == 10
    assert np.all(allocator.page_ref_counts == 0)


def test_allocate_single_page():
    """Test allocating a single page for a sequence"""
    allocator = PageAllocatorCPU(num_pages=10, max_seqs=4, page_size=128, pages_per_seq=3)

    # Allocate for positions 0-63 (should need 1 page)
    pages = allocator.allocate_pages_for_tokens(slot_id=0, token_positions=list(range(64)))

    assert len(pages) == 1
    assert pages[0] == 0  # Should get first page
    assert allocator.page_ref_counts[0] == 1
    assert allocator.get_free_page_count() == 9


def test_allocate_multiple_pages():
    """Test allocating multiple pages for a sequence"""
    allocator = PageAllocatorCPU(num_pages=10, max_seqs=4, page_size=128, pages_per_seq=3)

    # Allocate for positions 0-255 (should need 2 pages)
    pages = allocator.allocate_pages_for_tokens(slot_id=0, token_positions=list(range(256)))

    assert len(pages) == 2
    assert pages[0] == 0
    assert pages[1] == 1
    assert allocator.page_ref_counts[0] == 1
    assert allocator.page_ref_counts[1] == 1
    assert allocator.get_free_page_count() == 8


def test_allocate_incremental():
    """Test incremental allocation (adding more tokens to an existing sequence)"""
    allocator = PageAllocatorCPU(num_pages=10, max_seqs=4, page_size=128, pages_per_seq=3)

    # First allocation: 0-63 (1 page)
    pages1 = allocator.allocate_pages_for_tokens(slot_id=0, token_positions=list(range(64)))
    assert len(pages1) == 1

    # Second allocation: extend to 0-191 (2 pages total)
    pages2 = allocator.allocate_pages_for_tokens(slot_id=0, token_positions=list(range(192)))
    assert len(pages2) == 2
    assert pages2[0] == pages1[0]  # First page should be the same
    assert allocator.get_free_page_count() == 8


def test_allocate_multiple_sequences():
    """Test allocating pages for multiple sequences"""
    allocator = PageAllocatorCPU(num_pages=10, max_seqs=4, page_size=128, pages_per_seq=3)

    pages0 = allocator.allocate_pages_for_tokens(slot_id=0, token_positions=list(range(128)))
    pages1 = allocator.allocate_pages_for_tokens(slot_id=1, token_positions=list(range(64)))

    assert len(pages0) == 1
    assert len(pages1) == 1
    assert pages0[0] != pages1[0]  # Different pages
    assert allocator.get_free_page_count() == 8


def test_clone_pages():
    """Test cloning page assignments from one slot to another"""
    allocator = PageAllocatorCPU(num_pages=10, max_seqs=4, page_size=128, pages_per_seq=3)

    # Allocate for slot 0
    pages0 = allocator.allocate_pages_for_tokens(slot_id=0, token_positions=list(range(256)))
    assert len(pages0) == 2

    # Clone to slot 1
    pages1 = allocator.clone_pages(src_slot=0, dst_slot=1)

    assert len(pages1) == 2
    assert pages1 == pages0  # Same page IDs
    assert allocator.page_ref_counts[pages0[0]] == 2  # Ref count increased
    assert allocator.page_ref_counts[pages0[1]] == 2
    assert allocator.get_free_page_count() == 8  # No new pages allocated


def test_free_pages():
    """Test freeing pages for a sequence"""
    allocator = PageAllocatorCPU(num_pages=10, max_seqs=4, page_size=128, pages_per_seq=3)

    # Allocate and then free
    pages = allocator.allocate_pages_for_tokens(slot_id=0, token_positions=list(range(256)))
    assert len(pages) == 2
    assert allocator.get_free_page_count() == 8

    allocator.free_pages_for_slot(slot_id=0)

    assert len(allocator.slot_to_pages[0]) == 0
    assert allocator.page_ref_counts[pages[0]] == 0
    assert allocator.page_ref_counts[pages[1]] == 0
    assert allocator.get_free_page_count() == 10  # All pages free again


def test_free_shared_pages():
    """Test freeing shared pages (from clone)"""
    allocator = PageAllocatorCPU(num_pages=10, max_seqs=4, page_size=128, pages_per_seq=3)

    # Allocate for slot 0 and clone to slot 1
    pages0 = allocator.allocate_pages_for_tokens(slot_id=0, token_positions=list(range(128)))
    pages1 = allocator.clone_pages(src_slot=0, dst_slot=1)

    assert allocator.page_ref_counts[pages0[0]] == 2

    # Free slot 0
    allocator.free_pages_for_slot(slot_id=0)

    # Page should still be in use (ref_count=1) because slot 1 is using it
    assert allocator.page_ref_counts[pages0[0]] == 1
    assert allocator.get_free_page_count() == 9

    # Free slot 1
    allocator.free_pages_for_slot(slot_id=1)

    # Now page should be free
    assert allocator.page_ref_counts[pages0[0]] == 0
    assert allocator.get_free_page_count() == 10


def test_out_of_pages():
    """Test behavior when running out of pages"""
    allocator = PageAllocatorCPU(num_pages=2, max_seqs=4, page_size=128, pages_per_seq=3)

    # Allocate 2 pages for slot 0
    pages0 = allocator.allocate_pages_for_tokens(slot_id=0, token_positions=list(range(256)))
    assert len(pages0) == 2

    # Try to allocate for slot 1 (should fail - no free pages)
    with pytest.raises(RuntimeError, match="Out of free pages"):
        allocator.allocate_pages_for_tokens(slot_id=1, token_positions=list(range(128)))


def test_get_page_assignments():
    """Test getting snapshot of all page assignments"""
    allocator = PageAllocatorCPU(num_pages=10, max_seqs=4, page_size=128, pages_per_seq=3)

    allocator.allocate_pages_for_tokens(slot_id=0, token_positions=list(range(256)))
    allocator.allocate_pages_for_tokens(slot_id=2, token_positions=list(range(128)))

    assignments = allocator.get_page_assignments()

    assert assignments.shape == (4, 3)  # max_seqs x pages_per_seq
    assert assignments[0, 0] == 0
    assert assignments[0, 1] == 1
    assert assignments[0, 2] == INVALID
    assert assignments[2, 0] == 2
    assert assignments[2, 1] == INVALID
    assert assignments[1, 0] == INVALID  # Slot 1 not used


def test_ensure_pages():
    """Test ensure_pages method"""
    allocator = PageAllocatorCPU(num_pages=10, max_seqs=4, page_size=128, pages_per_seq=3)

    # Ensure 2 pages for slot 0
    allocator.ensure_pages(slot_id=0, num_pages=2)
    assert len(allocator.slot_to_pages[0]) == 2

    # Calling again with same num_pages should be a no-op
    allocator.ensure_pages(slot_id=0, num_pages=2)
    assert len(allocator.slot_to_pages[0]) == 2

    # Calling with higher num_pages should allocate more
    allocator.ensure_pages(slot_id=0, num_pages=3)
    assert len(allocator.slot_to_pages[0]) == 3


def test_get_stats():
    """Test statistics reporting"""
    allocator = PageAllocatorCPU(num_pages=10, max_seqs=4, page_size=128, pages_per_seq=3)

    stats1 = allocator.get_stats()
    assert stats1["total_pages"] == 10
    assert stats1["free_pages"] == 10
    assert stats1["used_pages"] == 0
    assert stats1["active_slots"] == 0

    # Allocate some pages
    allocator.allocate_pages_for_tokens(slot_id=0, token_positions=list(range(256)))
    stats2 = allocator.get_stats()
    assert stats2["free_pages"] == 8
    assert stats2["used_pages"] == 2
    assert stats2["active_slots"] == 1

    # Clone (creates shared pages)
    allocator.clone_pages(src_slot=0, dst_slot=1)
    stats3 = allocator.get_stats()
    assert stats3["shared_pages"] == 2
    assert stats3["active_slots"] == 2


def test_reset():
    """Test resetting the allocator"""
    allocator = PageAllocatorCPU(num_pages=10, max_seqs=4, page_size=128, pages_per_seq=3)

    # Allocate some pages
    allocator.allocate_pages_for_tokens(slot_id=0, token_positions=list(range(256)))
    assert allocator.get_free_page_count() == 8

    # Reset
    allocator.reset()

    assert allocator.get_free_page_count() == 10
    assert all(len(pages) == 0 for pages in allocator.slot_to_pages)
    assert np.all(allocator.page_ref_counts == 0)
