# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
CPU-side page allocator for KV cache management.

This module provides a pure Python/NumPy implementation of page allocation,
moving allocation logic from the device (TPU/GPU) to the host (CPU).
This simplifies the JIT-compiled code and provides better control over
memory management.
"""

import logging

import numpy as np

from levanter.inference.utils import INVALID

logger = logging.getLogger(__name__)


class PageAllocatorCPU:
    """
    CPU-side page allocator for managing KV cache pages.

    This allocator uses reference counting to support copy-on-write semantics
    for sequence cloning. Pages can be shared between sequences (ref_count > 1)
    and are only freed when ref_count reaches 0.

    Attributes:
        num_pages: Total number of pages available
        page_size: Number of tokens per page (typically 128)
        max_seqs: Maximum number of concurrent sequences
        pages_per_seq: Maximum pages per sequence
        page_ref_counts: Reference count per page (0 = free)
        slot_to_pages: List of page IDs for each sequence slot
    """

    def __init__(self, num_pages: int, max_seqs: int, page_size: int, pages_per_seq: int):
        """
        Initialize the page allocator.

        Args:
            num_pages: Total number of pages in the KV cache
            max_seqs: Maximum number of concurrent sequences
            page_size: Number of tokens per page
            pages_per_seq: Maximum pages per sequence
        """
        self.num_pages = num_pages
        self.page_size = page_size
        self.max_seqs = max_seqs
        self.pages_per_seq = pages_per_seq

        # Track reference counts: 0 = free, >0 = in use
        self.page_ref_counts = np.zeros(num_pages, dtype=np.int32)

        # Track which pages each slot is using (list of lists for variable length)
        self.slot_to_pages: list[list[int]] = [[] for _ in range(max_seqs)]

        logger.info(
            f"Initialized PageAllocatorCPU: {num_pages} pages, "
            f"{page_size} tokens/page, {max_seqs} max seqs, "
            f"{pages_per_seq} pages/seq"
        )

    def allocate_pages_for_tokens(self, slot_id: int, token_positions: list[int]) -> list[int]:
        """
        Allocate pages for a sequence based on token positions.

        This method determines how many pages are needed based on the maximum
        token position and allocates any additional pages required.

        Args:
            slot_id: Sequence slot ID
            token_positions: List of token positions to allocate pages for

        Returns:
            List of page IDs assigned to this slot (including existing pages)

        Raises:
            RuntimeError: If no free pages are available
        """
        if not token_positions:
            return self.slot_to_pages[slot_id]

        # Calculate how many pages we need based on max position
        max_pos = max(token_positions)
        pages_needed = (max_pos + self.page_size) // self.page_size
        pages_have = len(self.slot_to_pages[slot_id])

        # Allocate additional pages if needed
        for _ in range(pages_needed - pages_have):
            page_id = self._allocate_one_page()
            if page_id == INVALID:
                raise RuntimeError(
                    f"Out of free pages! Slot {slot_id} needs {pages_needed} pages, "
                    f"has {pages_have}, but no free pages available. "
                    f"Free pages: {np.sum(self.page_ref_counts == 0)}/{self.num_pages}"
                )
            self.slot_to_pages[slot_id].append(page_id)
            self.page_ref_counts[page_id] += 1

        return self.slot_to_pages[slot_id]

    def _allocate_one_page(self) -> int:
        """
        Allocate a single free page.

        Returns:
            Page ID of allocated page, or INVALID if none available
        """
        # Find first free page (ref_count == 0)
        free_pages = np.where(self.page_ref_counts == 0)[0]
        if len(free_pages) == 0:
            return INVALID
        return int(free_pages[0])

    def clone_pages(self, src_slot: int, dst_slot: int) -> list[int]:
        """
        Clone page assignments from source slot to destination slot.

        This implements copy-on-write semantics by incrementing reference counts
        on shared pages. For the last partial page (if any), we allocate a fresh
        page instead of sharing.

        Args:
            src_slot: Source slot to clone from
            dst_slot: Destination slot to clone to

        Returns:
            List of page IDs assigned to destination slot

        Raises:
            RuntimeError: If no free pages available for partial page copy
        """
        src_pages = self.slot_to_pages[src_slot]

        if not src_pages:
            # Nothing to clone
            self.slot_to_pages[dst_slot] = []
            return []

        # Clone all pages by incrementing ref counts
        # For simplicity in this initial version, we share all pages
        # A more sophisticated version could detect partial pages and copy them
        dst_pages = src_pages.copy()
        self.slot_to_pages[dst_slot] = dst_pages

        # Increment ref counts for all shared pages
        for page_id in dst_pages:
            self.page_ref_counts[page_id] += 1

        logger.debug(f"Cloned slot {src_slot} -> {dst_slot}: {len(dst_pages)} pages shared")

        return dst_pages

    def free_pages_for_slot(self, slot_id: int):
        """
        Free all pages for a sequence slot.

        Decrements reference counts for all pages. Pages with ref_count=0
        become available for reallocation.

        Args:
            slot_id: Sequence slot ID to free
        """
        pages = self.slot_to_pages[slot_id]

        if not pages:
            return

        # Decrement ref counts
        for page_id in pages:
            self.page_ref_counts[page_id] -= 1
            if self.page_ref_counts[page_id] < 0:
                logger.warning(f"Page {page_id} ref count went negative! " f"This indicates a bug in page management.")
                self.page_ref_counts[page_id] = 0

        # Clear slot's page list
        self.slot_to_pages[slot_id] = []

        logger.debug(f"Freed {len(pages)} pages for slot {slot_id}")

    def get_page_assignments(self) -> np.ndarray:
        """
        Get a snapshot of all page assignments.

        Returns:
            Array of shape [max_seqs, pages_per_seq] with page IDs,
            padded with INVALID for unused entries
        """
        assignments = np.full((self.max_seqs, self.pages_per_seq), INVALID, dtype=np.int32)

        for slot_id, pages in enumerate(self.slot_to_pages):
            if pages:
                assignments[slot_id, : len(pages)] = pages

        return assignments

    def get_free_page_count(self) -> int:
        """Get the number of free pages (ref_count == 0)."""
        return int(np.sum(self.page_ref_counts == 0))

    def ensure_pages(self, slot_id: int, num_pages: int):
        """
        Ensure a slot has at least num_pages allocated.

        This is used for pre-allocation before decode rounds.

        Args:
            slot_id: Sequence slot ID
            num_pages: Minimum number of pages to have allocated

        Raises:
            RuntimeError: If allocation fails
        """
        current_pages = len(self.slot_to_pages[slot_id])
        if current_pages >= num_pages:
            return

        # Allocate additional pages
        for _ in range(num_pages - current_pages):
            page_id = self._allocate_one_page()
            if page_id == INVALID:
                raise RuntimeError(
                    f"Failed to ensure {num_pages} pages for slot {slot_id}. "
                    f"Currently have {current_pages}, free pages: {self.get_free_page_count()}"
                )
            self.slot_to_pages[slot_id].append(page_id)
            self.page_ref_counts[page_id] += 1

    def get_stats(self) -> dict:
        """
        Get allocation statistics for debugging/monitoring.

        Returns:
            Dictionary with statistics about page usage
        """
        free_pages = self.get_free_page_count()
        used_pages = self.num_pages - free_pages
        shared_pages = np.sum(self.page_ref_counts > 1)
        active_slots = sum(1 for pages in self.slot_to_pages if pages)

        return {
            "total_pages": self.num_pages,
            "free_pages": free_pages,
            "used_pages": used_pages,
            "shared_pages": shared_pages,
            "active_slots": active_slots,
            "utilization": used_pages / self.num_pages if self.num_pages > 0 else 0.0,
        }

    def reset(self):
        """Reset the allocator to initial state (all pages free)."""
        self.page_ref_counts.fill(0)
        self.slot_to_pages = [[] for _ in range(self.max_seqs)]
        logger.info("PageAllocatorCPU reset")
