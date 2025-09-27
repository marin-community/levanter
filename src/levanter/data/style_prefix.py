# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for style/source prefix tokens used during training."""

from __future__ import annotations

from typing import Sequence

from levanter.utils.hf_utils import HfTokenizer


STYLE_PREFIX_TOKEN = "<style>"
STYLE_SUFFIX_TOKEN = "</style>"


def ensure_style_tokens(tokenizer: HfTokenizer, *, required_tokens: Sequence[str] | None = None) -> int:
    """Ensure that the tokenizer contains the reserved style tokens.

    Args:
        tokenizer: The tokenizer to update.
        required_tokens: Optional override for which tokens should be ensured. Defaults to
            ``("<style>", "</style>")``.

    Returns:
        The number of new tokens added to the tokenizer vocabulary.
    """

    tokens = list(required_tokens if required_tokens is not None else (STYLE_PREFIX_TOKEN, STYLE_SUFFIX_TOKEN))

    # ``add_special_tokens`` tolerates duplicates, so we only need to add tokens that are not already
    # declared as special tokens. ``additional_special_tokens`` preserves insertion order, so we append
    # any missing tokens at the end of that list.
    missing = [tok for tok in tokens if tok not in tokenizer.additional_special_tokens]
    added = 0
    if missing:
        added = tokenizer.add_special_tokens({"additional_special_tokens": missing})

    return added

