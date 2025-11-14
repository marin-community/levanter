# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# test_cross_entropy.py
import math

import equinox
import jax.numpy as jnp
import jax.random
import pytest

import haliax as hax
from haliax import NamedArray

# Import the functions from your module
# Replace 'your_module' with the actual module name where your functions are defined
from levanter.models.loss import _blockwise_cross_entropy_loss, cross_entropy_loss_and_log_normalizers
from levanter.utils.jax_utils import key_iterator


Batch = hax.Axis("batch", size=2)
Seq = hax.Axis("seq", size=3)
Embed = hax.Axis("embed", size=8)
Vocab = hax.Axis("vocab", size=16)


@pytest.fixture
def test_data():
    """
    Create synthetic test data for cross-entropy loss computation.
    """

    key = key_iterator(jax.random.PRNGKey(0))

    # Initialize pred_embeddings with ones
    pred_embeddings = hax.random.normal(next(key), (Batch, Seq, Embed), dtype=jnp.float32) / math.sqrt(Embed.size)

    # Initialize lm_head with ones
    lm_head = hax.random.normal(next(key), (Vocab, Embed), dtype=jnp.float32) / math.sqrt(Embed.size)

    # Define true_ids such that the target is always the first token in vocab
    true_ids = hax.random.randint(next(key), (Batch, Seq), 0, Vocab.size)

    return pred_embeddings, lm_head, true_ids


def test_basic_equivalence(test_data):
    """
    Test that block-wise loss equals full loss when block_size perfectly divides vocab_size.
    """
    pred_embeddings, lm_head, true_ids = test_data

    # Compute full loss
    logits_full = hax.dot(pred_embeddings, lm_head, axis="embed")
    target_y_full = hax.nn.one_hot(true_ids, Vocab, dtype=pred_embeddings.dtype)
    loss_full, norm_full = cross_entropy_loss_and_log_normalizers(logits_full, Vocab, target_y_full)

    loss_block, norm_this = _blockwise_cross_entropy_loss(
        (pred_embeddings, lm_head),
        Contract=Embed,
        Label=Vocab,
        Pos=Seq,
        labels_y=true_ids,
        vocab_block_size=8,
        dtype=pred_embeddings.dtype,
    )

    # Assert that the losses are close
    assert hax.all(
        hax.isclose(loss_full, loss_block, atol=1e-3, rtol=1e-3)
    ), "Block-wise loss does not match full loss."


def test_single_block(test_data):
    """
    Test behavior when vocab_size equals block_size.
    """
    pred_embeddings, lm_head, true_ids = test_data

    # Compute full loss
    loss_full, sumexp_full = _compute_full(Vocab, pred_embeddings, lm_head, true_ids)

    # Compute block-wise loss with vocab_block_size=4 (vocab_size=4)
    loss_block, sumexp_block = _blockwise_cross_entropy_loss(
        (pred_embeddings, lm_head),
        Contract=Embed,
        Label=Vocab,
        Pos=Seq,
        labels_y=true_ids,
        vocab_block_size=Vocab.size,
        dtype=pred_embeddings.dtype,
    )

    # Assert that the losses are close
    assert hax.all(
        hax.isclose(sumexp_full, sumexp_block, atol=1e-3, rtol=1e-3)
    ), "Single block-wise sumexp does not match full sumexp."
    assert hax.all(
        hax.isclose(loss_full, loss_block, atol=1e-3, rtol=1e-3)
    ), "Single block-wise loss does not match full loss."


def _compute_full(Vocab, pred_embeddings, lm_head, true_ids):
    logits_full = hax.dot(pred_embeddings, lm_head, axis="embed")
    target_y_full = hax.nn.one_hot(true_ids, Vocab, dtype=pred_embeddings.dtype)
    loss_full, sumexp_full = cross_entropy_loss_and_log_normalizers(logits_full, Vocab, target_y_full)
    return loss_full, sumexp_full


def test_multiple_blocks(test_data):
    """
    Test block-wise loss with multiple blocks.
    """
    pred_embeddings, lm_head, true_ids = test_data

    # Compute full loss
    loss_full, logz_full = _compute_full(Vocab, pred_embeddings, lm_head, true_ids)

    # Compute block-wise loss with vocab_block_size=1 (vocab_size=4)
    loss_block, logz_block = _blockwise_cross_entropy_loss(
        (pred_embeddings, lm_head),
        Contract=Embed,
        Label=Vocab,
        Pos=Seq,
        labels_y=true_ids,
        vocab_block_size=1,
        dtype=pred_embeddings.dtype,
    )

    # Assert that the losses are close
    assert hax.all(
        hax.isclose(logz_full, logz_block, atol=1e-3, rtol=1e-3)
    ), "Multiple block-wise logz does not match full logz."
    assert hax.all(
        hax.isclose(loss_full, loss_block, atol=1e-3, rtol=1e-3)
    ), "Multiple block-wise loss does not match full loss."


def test_block_size_not_dividing_vocab(test_data):
    pred_embeddings, lm_head, true_ids = test_data

    block_size = 3
    assert Vocab.size % block_size != 0

    loss_block, logz_block = _blockwise_cross_entropy_loss(
        (pred_embeddings, lm_head),
        Contract=Embed,
        Label=Vocab,
        Pos=Seq,
        labels_y=true_ids,
        vocab_block_size=block_size,
        dtype=pred_embeddings.dtype,
    )

    # Compute full loss
    loss_full, logz_full = cross_entropy_loss_and_log_normalizers(
        pred_y=hax.dot(pred_embeddings, lm_head, axis="embed"),
        Label=Vocab,
        target_y=hax.nn.one_hot(true_ids, Vocab, dtype=pred_embeddings.dtype),
    )

    # Assert that the losses are close
    assert hax.all(
        hax.isclose(loss_full, loss_block, atol=1e-3, rtol=1e-3)
    ), "Block-wise loss does not match full loss."
    assert hax.all(
        hax.isclose(logz_full, logz_block, atol=1e-3, rtol=1e-3)
    ), "Block-wise logz does not match full logz."


def test_vocab_size_less_than_block_size(test_data):
    """
    Test behavior when vocab_size is less than block_size.
    """
    pred_embeddings, lm_head, true_ids = test_data

    # Set block_size greater than vocab_size
    block_size = 5  # vocab_size=4

    # should be fine now
    loss_block, logz_block = _blockwise_cross_entropy_loss(
        (pred_embeddings, lm_head),
        Contract=Embed,
        Label=Vocab,
        Pos=Seq,
        labels_y=true_ids,
        vocab_block_size=block_size,
        dtype=pred_embeddings.dtype,
    )

    # Compute full loss
    loss_full, logz_full = cross_entropy_loss_and_log_normalizers(
        pred_y=hax.dot(pred_embeddings, lm_head, axis="embed"),
        Label=Vocab,
        target_y=hax.nn.one_hot(true_ids, Vocab, dtype=pred_embeddings.dtype),
    )

    # Assert that the losses are close
    assert hax.all(hax.isclose(loss_full, loss_block, atol=1e-3, rtol=1e-3)), "loss does not match full loss."
    assert hax.all(hax.isclose(logz_full, logz_block, atol=1e-3, rtol=1e-3)), "logz does not match full logz."


def test_large_vocab():
    """
    Test block-wise loss with a larger vocabulary.
    """
    Batch = hax.Axis("batch", size=4)
    Seq = hax.Axis("seq", size=5)
    Embed = hax.Axis("embed", size=6)
    Vocab = hax.Axis("vocab", size=12)

    pred_embeddings = NamedArray(
        jnp.ones((Batch.size, Seq.size, Embed.size)),
        axes=(Batch, Seq, Embed),
    )
    lm_head = NamedArray(
        jnp.ones((Embed.size, Vocab.size)),
        axes=(Embed, Vocab),
    )
    true_ids = NamedArray(
        jnp.zeros((Batch.size, Seq.size), dtype=jnp.int32),
        axes=(Batch, Seq),
    )

    # Compute full loss
    loss_full, logz_full = cross_entropy_loss_and_log_normalizers(
        pred_y=hax.dot(pred_embeddings, lm_head, axis="embed"),
        Label=Vocab,
        target_y=hax.nn.one_hot(true_ids, Vocab, dtype=pred_embeddings.dtype),
    )

    # Compute block-wise loss with vocab_block_size=3 (vocab_size=12 is divisible by 3)
    loss_block, logz_block = _blockwise_cross_entropy_loss(
        (pred_embeddings, lm_head),
        Contract=Embed,
        Label=Vocab,
        Pos=Seq,
        labels_y=true_ids,
        vocab_block_size=3,
        dtype=pred_embeddings.dtype,
    )

    # Assert that the losses are close
    assert hax.all(
        hax.isclose(loss_full, loss_block, atol=1e-3, rtol=1e-3)
    ), "Large vocab block-wise loss does not match full loss."
    assert hax.all(
        hax.isclose(logz_full, logz_block, atol=1e-3, rtol=1e-3)
    ), "Large vocab block-wise logz does not match full logz."


@pytest.mark.parametrize("vocab_block_size", [1, 2, 3, 4, 5])
def test_gradient_block_cross_entropy(vocab_block_size, test_data):
    """
    Test the gradient of block-wise cross-entropy loss.
    """
    pred_embeddings, lm_head, true_ids = test_data

    # Compute block-wise loss
    def custom_fn(pred):
        pred_embeddings, lm_head = pred
        a, b = _blockwise_cross_entropy_loss(
            (pred_embeddings, lm_head),
            Contract=Embed,
            Label=Vocab,
            Pos=Seq,
            labels_y=true_ids,
            vocab_block_size=vocab_block_size,
            dtype=pred_embeddings.dtype,
        )

        return (a.mean() + b.mean()).scalar()

    (
        g_embed,
        g_head,
    ) = equinox.filter_grad(
        custom_fn
    )((pred_embeddings, lm_head))

    # compute directly

    def direct_fn(pred):
        pred_embeddings, lm_head = pred
        logits = hax.dot(pred_embeddings, lm_head, axis="embed")
        target_y = hax.nn.one_hot(true_ids, Vocab, dtype=pred_embeddings.dtype)
        loss, logz = cross_entropy_loss_and_log_normalizers(logits, Vocab, target_y)
        return (loss.mean() + logz.mean()).scalar()

    g_embed_direct, g_head_direct = equinox.filter_grad(direct_fn)((pred_embeddings, lm_head))

    assert hax.all(
        hax.isclose(g_embed, g_embed_direct, atol=1e-3, rtol=1e-3)
    ), "Gradient of embeddings does not match."
    assert hax.all(hax.isclose(g_head, g_head_direct, atol=1e-3, rtol=1e-3)), "Gradient of lm_head does not match."


def test_grad_loss_without_logz(test_data):
    """
    Test the gradient of block-wise cross-entropy loss without logz.
    """
    pred_embeddings, lm_head, true_ids = test_data

    # Compute block-wise loss
    def custom_fn(pred):
        pred_embeddings, lm_head = pred
        a, b = _blockwise_cross_entropy_loss(
            (pred_embeddings, lm_head),
            Contract=Embed,
            Label=Vocab,
            Pos=Seq,
            labels_y=true_ids,
            vocab_block_size=2,
            dtype=pred_embeddings.dtype,
        )

        return a.mean().scalar()

    (
        g_embed,
        g_head,
    ) = equinox.filter_grad(
        custom_fn
    )((pred_embeddings, lm_head))

    # compute directly

    def direct_fn(pred):
        pred_embeddings, lm_head = pred
        logits = hax.dot(pred_embeddings, lm_head, axis="embed", preferred_element_type=pred_embeddings.dtype)
        target_y = hax.nn.one_hot(true_ids, Vocab, dtype=pred_embeddings.dtype)
        loss, _ = cross_entropy_loss_and_log_normalizers(logits, Vocab, target_y)
        return loss.mean().scalar()

    g_embed_direct, g_head_direct = equinox.filter_grad(direct_fn)((pred_embeddings, lm_head))

    assert hax.all(
        hax.isclose(g_embed, g_embed_direct, atol=1e-3, rtol=1e-3)
    ), "Gradient of embeddings does not match."
    assert hax.all(hax.isclose(g_head, g_head_direct, atol=1e-3, rtol=1e-3)), "Gradient of lm_head does not match."


# add a test case for logit_soft_cap
def test_grad_loss_with_logit_soft_cap(test_data):
    """
    Test the gradient of block-wise cross-entropy loss with logit_soft_cap.
    """
    pred_embeddings, lm_head, true_ids = test_data

    logit_soft_cap = 0.5

    # Compute block-wise loss
    def custom_fn(pred):
        pred_embeddings, lm_head = pred
        a, b = _blockwise_cross_entropy_loss(
            (pred_embeddings, lm_head),
            Contract=Embed,
            Label=Vocab,
            Pos=Seq,
            labels_y=true_ids,
            vocab_block_size=2,
            dtype=pred_embeddings.dtype,
            logit_soft_cap=logit_soft_cap,
        )

        return a.mean().scalar()

    (
        g_embed,
        g_head,
    ) = equinox.filter_grad(
        custom_fn
    )((pred_embeddings, lm_head))

    # compute directly with logit_soft_cap applied
    def direct_fn(pred):
        pred_embeddings, lm_head = pred
        logits = hax.dot(pred_embeddings, lm_head, axis="embed", preferred_element_type=pred_embeddings.dtype)
        # Apply logit soft cap
        logits = logit_soft_cap * hax.tanh(logits / logit_soft_cap)
        target_y = hax.nn.one_hot(true_ids, Vocab, dtype=pred_embeddings.dtype)
        loss, _ = cross_entropy_loss_and_log_normalizers(logits, Vocab, target_y)
        return loss.mean().scalar()

    g_embed_direct, g_head_direct = equinox.filter_grad(direct_fn)((pred_embeddings, lm_head))

    assert hax.all(
        hax.isclose(g_embed, g_embed_direct, atol=1e-3, rtol=1e-3)
    ), "Gradient of embeddings does not match."
    assert hax.all(hax.isclose(g_head, g_head_direct, atol=1e-3, rtol=1e-3)), "Gradient of lm_head does not match."


# -----------------------------------------------------------------------------
# New tests: batch/seq parallelism permutations
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("batch_block_size", [None, 1, Batch.size, 2 * Batch.size])
@pytest.mark.parametrize("seq_block_size", [None, 1, 2, Seq.size, 2 * Seq.size])
@pytest.mark.parametrize("vocab_block_size", [1, 3, 8, Vocab.size, 2 * Vocab.size])
def test_blockwise_batch_seq_parallelism_equivalence(test_data, batch_block_size, seq_block_size, vocab_block_size):
    pred_embeddings, lm_head, true_ids = test_data

    # Full reference
    loss_full, logz_full = cross_entropy_loss_and_log_normalizers(
        pred_y=hax.dot(pred_embeddings, lm_head, axis="embed"),
        Label=Vocab,
        target_y=hax.nn.one_hot(true_ids, Vocab, dtype=pred_embeddings.dtype),
    )

    # Blockwise with batch/seq splitting
    loss_block, logz_block = _blockwise_cross_entropy_loss(
        (pred_embeddings, lm_head),
        Contract=Embed,
        Label=Vocab,
        Pos=Seq,
        labels_y=true_ids,
        vocab_block_size=vocab_block_size,
        dtype=pred_embeddings.dtype,
        batch_block_size=batch_block_size,
        seq_block_size=seq_block_size,
    )

    assert hax.all(hax.isclose(loss_full, loss_block, atol=1e-3, rtol=1e-3))
    assert hax.all(hax.isclose(logz_full, logz_block, atol=1e-3, rtol=1e-3))


@pytest.mark.parametrize("batch_block_size", [1, Batch.size, 2 * Batch.size])
@pytest.mark.parametrize("seq_block_size", [1, 2, Seq.size, 2 * Seq.size])
@pytest.mark.parametrize("vocab_block_size", [1, 3, 8, Vocab.size, 2 * Vocab.size])
@pytest.mark.parametrize("logit_soft_cap", [None, 0.5])
def test_gradient_equivalence_with_batch_seq_vocab_block(
    test_data, batch_block_size, seq_block_size, vocab_block_size, logit_soft_cap
):
    pred_embeddings, lm_head, true_ids = test_data

    def custom_fn(pred):
        pred_embeddings, lm_head = pred
        loss, logz = _blockwise_cross_entropy_loss(
            (pred_embeddings, lm_head),
            Contract=Embed,
            Label=Vocab,
            Pos=Seq,
            labels_y=true_ids,
            vocab_block_size=vocab_block_size,
            dtype=pred_embeddings.dtype,
            batch_block_size=batch_block_size,
            seq_block_size=seq_block_size,
            logit_soft_cap=logit_soft_cap,
        )
        return (loss.mean() + logz.mean()).scalar()

    g_embed, g_lm_head = equinox.filter_grad(custom_fn)((pred_embeddings, lm_head))

    def direct_fn(pred):
        pred_embeddings, lm_head = pred
        logits = hax.dot(pred_embeddings, lm_head, axis="embed")
        if logit_soft_cap is not None:
            logits = logit_soft_cap * hax.tanh(logits / logit_soft_cap)
        target_y = hax.nn.one_hot(true_ids, Vocab, dtype=pred_embeddings.dtype)
        loss, logz = cross_entropy_loss_and_log_normalizers(logits, Vocab, target_y)
        return (loss.mean() + logz.mean()).scalar()

    g_embed_direct, g_lm_head_direct = equinox.filter_grad(direct_fn)((pred_embeddings, lm_head))

    assert hax.all(hax.isclose(g_embed, g_embed_direct, atol=1e-3, rtol=1e-3))
    assert hax.all(hax.isclose(g_lm_head, g_lm_head_direct, atol=1e-3, rtol=1e-3))
