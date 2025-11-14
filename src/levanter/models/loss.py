# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import math
from typing import Optional
import jax.experimental.pallas as pl

import equinox
import jax
import jax.numpy as jnp

import haliax as hax
from haliax import NamedArray
from haliax.nn import cross_entropy_loss_and_log_normalizers

use_interpret = jax.default_backend() == "cpu"


def maybe_fused_next_token_loss(
    Pos: hax.AxisSelector,
    Embed: hax.AxisSelector,
    Vocab: hax.AxisSelector,
    pred_embeddings: NamedArray,
    lm_head: NamedArray,
    true_ids: NamedArray,
    loss_mask: Optional[NamedArray] = None,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    logsumexp_weight: Optional[float] = None,
    batch_block_size: Optional[int] = None,
    seq_block_size: Optional[int] = None,
    vocab_block_size: Optional[int] = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
) -> NamedArray:
    """
    Compute the next token loss with optional block-wise processing.

    Args:
        Pos (hax.AxisSelector): Position axis selector.
        Vocab (hax.AxisSelector): Vocabulary axis selector.
        pred_embeddings (NamedArray): Predicted embeddings.
        lm_head (NamedArray): Language model head weights.
        true_ids (NamedArray): True token IDs.
        loss_mask (Optional[NamedArray]): Mask to apply to the loss.
        reduction (Optional[hax.ReductionFunction]): Reduction function.
        reduction_axis (Optional[hax.AxisSelection]): Axis to apply reduction.
        logsumexp_weight (Optional[float]): Weight for logsumexp penalty.
        batch_block_size (Optional[int]): Size of each batch block for processing.
        seq_block_size (Optional[int]): Size of each sequence block for processing.
        vocab_block_size (Optional[int]): Size of each block for processing.
        dtype (Optional[jnp.dtype]): Data type for the loss.
        logit_soft_cap (Optional[float]): Optional soft cap for logits
    Returns:
        NamedArray: Computed loss.
    """
    # Resolve axes
    Pos = pred_embeddings.resolve_axis(Pos.name)
    Vocab = lm_head.resolve_axis(Vocab)

    if vocab_block_size is None:
        # Full softmax computation
        logits = hax.dot(pred_embeddings, lm_head, axis=Embed)
        if dtype is not None:
            logits = logits.astype(dtype)

        if logit_soft_cap is not None:
            logits = hax.tanh(logits / logit_soft_cap) * logit_soft_cap

        # Shift target tokens to predict the next token
        return next_token_loss(Pos, Vocab, logits, true_ids, loss_mask, reduction, reduction_axis, logsumexp_weight)

    # Shift target tokens to predict the next token
    target_y = hax.roll(true_ids, -1, Pos)

    # Create a mask that excludes the last token
    not_last_loss_mask = hax.logical_not(hax.nn.one_hot(-1, Pos, dtype=jnp.bool_))  # type: ignore
    if loss_mask is not None:
        loss_mask = loss_mask * not_last_loss_mask
    else:
        loss_mask = not_last_loss_mask

    # Compute the loss with optional block-wise processing
    return fused_cross_entropy_loss_and_logsumexp_penalty(
        pred_embeddings,
        lm_head,
        Pos=Pos,
        Contract=Embed,
        Label=Vocab,
        target_y=target_y,
        reduction=reduction,
        reduction_axis=reduction_axis,
        where=loss_mask,
        logsumexp_weight=logsumexp_weight,
        batch_block_size=batch_block_size,
        seq_block_size=seq_block_size,
        vocab_block_size=vocab_block_size,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
    )


def next_token_loss(
    Pos: hax.AxisSelector,
    Vocab: hax.AxisSelector,
    logits: NamedArray,
    true_ids: NamedArray,
    loss_mask: Optional[NamedArray] = None,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    logsumexp_weight: Optional[float] = None,
):
    """
    Compute the next token loss with optional logsumexp penalty.

    Args:
        Pos: axis selector for the position axis
        Vocab: axis selector for the vocabulary axis
        logits: predicted logits
        true_ids: true token IDs (not shifted)
        loss_mask: mask to apply to the loss
        reduction: reduction function or None to disable reduction
        reduction_axis: axis to apply reduction. None means all axes
        logsumexp_weight: weight for the logsumexp penalty
        logit_soft_cap: optional soft cap for logits
    Returns:
        NamedArray: computed loss
    """
    Pos = logits.resolve_axis(hax.axis_name(Pos))

    target_y = hax.roll(true_ids, -1, Pos)
    target_y_full = hax.nn.one_hot(target_y, Vocab, dtype=logits.dtype)

    # Create a mask that excludes the last token
    not_last_loss_mask = hax.logical_not(hax.nn.one_hot(-1, Pos, dtype=jnp.bool_))
    if loss_mask is not None:
        loss_mask = hax.logical_and(loss_mask, not_last_loss_mask)
    else:
        loss_mask = not_last_loss_mask

    return cross_entropy_and_logsumexp_penalty(
        Vocab=Vocab,
        pred_y=logits,
        target_y=target_y_full,
        reduction=reduction,
        reduction_axis=reduction_axis,
        where=loss_mask,
        logsumexp_weight=logsumexp_weight,
    )


def cross_entropy_and_logsumexp_penalty(
    Vocab: hax.Axis,
    pred_y: NamedArray,
    target_y: NamedArray,
    *,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    where: Optional[NamedArray] = None,
    logsumexp_weight=0.0,
) -> NamedArray:
    """A loss function that combines cross entropy loss with a logsumexp penalty."""

    loss, log_normalizers = cross_entropy_loss_and_log_normalizers(pred_y, Vocab, target_y)

    if logsumexp_weight is not None and logsumexp_weight != 0.0:
        loss = loss + logsumexp_weight * (log_normalizers**2)

    return hax.nn.loss.maybe_reduce_loss(loss, reduction, reduction_axis, where)


def fused_cross_entropy_loss_and_logsumexp_penalty(
    pred_embeddings: NamedArray,
    lm_head: NamedArray,
    Contract: hax.AxisSelector,
    Label: hax.AxisSelector,
    Pos: hax.AxisSelector,
    target_y: NamedArray,
    *,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    where: Optional[NamedArray] = None,
    logsumexp_weight: float | None = 0.0,
    batch_block_size: Optional[int] = None,
    seq_block_size: Optional[int] = None,
    vocab_block_size: int,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
) -> NamedArray:
    """
    Compute the cross-entropy loss and logsumexp penalty using embeddings and lm_head,
    with optional block-wise processing.

    Args:
        pred_embeddings (NamedArray): Predicted embeddings.
        lm_head (NamedArray): Language model head weights.
        Contract (hax.AxisSelector): Axis to contract over.
        Label (hax.AxisSelector): Label (Vocab) axis.
        Pos (hax.AxisSelector): Position axis.
        target_y (NamedArray): One-hot encoded target tokens.
        reduction (Optional[hax.ReductionFunction]): Reduction function.
        reduction_axis (Optional[hax.AxisSelection]): Axis to apply reduction.
        where (Optional[NamedArray]): Mask to apply to the loss.
        logsumexp_weight (float): Weight for logsumexp penalty.
        batch_block_size (Optional[int]): Size of each batch block for processing.
        seq_block_size (Optional[int]): Size of each sequence block for processing.
        vocab_block_size (int): Size of each block for processing.
        dtype (Optional[jnp.dtype]): Data type for the loss.

    Returns:
        NamedArray: Computed loss.
    """

    # Block-wise softmax computation
    loss, log_normalizers = _blockwise_cross_entropy_loss(
        (pred_embeddings, lm_head),
        Contract,
        Label,
        Pos,
        target_y,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        batch_block_size=batch_block_size,
        seq_block_size=seq_block_size,
        vocab_block_size=vocab_block_size,
    )

    if logsumexp_weight is not None and (not isinstance(logsumexp_weight, (int, float)) or logsumexp_weight != 0.0):
        loss = loss + logsumexp_weight * (log_normalizers**2)

    return hax.nn.loss.maybe_reduce_loss(loss, reduction, reduction_axis, where)


@equinox.filter_custom_vjp
def _blockwise_cross_entropy_loss(
    # pred_embeddings: NamedArray,
    # lm_head: NamedArray,
    pred: tuple[NamedArray, NamedArray],
    Contract: hax.Axis,
    Label: hax.Axis,
    Pos: hax.Axis,
    labels_y: NamedArray,
    *,
    batch_block_size: Optional[int] = None,
    seq_block_size: Optional[int] = None,
    vocab_block_size: int,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float] = None,
) -> tuple[NamedArray, NamedArray]:
    """
    Compute cross-entropy loss and log normalizers in a block-wise manner without materializing the full logits.

    Args:
        pred_embeddings (NamedArray): Predicted embeddings.
        lm_head (NamedArray): Language model head weights.
        Contract (hax.Axis): Axis to contract over.
        Label (hax.AxisSelector): Label (Vocab) axis.
        Pos (hax.AxisSelector): Position axis.
        labels_y (NamedArray): label tensor.
        batch_block_size (Optional[int]): Size of each batch block for processing.
        seq_block_size (Optional[int]): Size of each sequence block for processing.
        vocab_block_size (int): Size of each block for processing.
        dtype (Optional[jnp.dtype]): Data type for the loss.

    Notes:
        labels_y being anything other than the label tensor would remove any benefits

        TODO: but if XLA smart enough to optimize it out?

    Returns:
        tuple[NamedArray, NamedArray]: tuple of loss and log_normalizers.
    """

    return _block_cross_entropy_forward(
        None,
        pred,
        Contract,
        Label,
        Pos,
        labels_y,
        batch_block_size=batch_block_size,
        seq_block_size=seq_block_size,
        vocab_block_size=vocab_block_size,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
    )[0]


def _block_to_one_hot(labels: NamedArray, VocabSlice: hax.Axis, block_start: int, dtype: jnp.dtype):
    end = block_start + VocabSlice.size
    label_is_in_block = hax.logical_and(labels >= block_start, labels < end)
    one_hot = hax.nn.one_hot(labels - block_start, class_axis=VocabSlice, dtype=dtype)
    return one_hot * label_is_in_block


def _make_tile_mask(axis: hax.Axis, axis_full: hax.Axis, start: int):
    axis_pos = hax.arange(axis) + (start * axis.size)
    return axis_pos < axis_full.size


def _block_cross_entropy_forward_kernel(
    lm_head_ref,  # [Embed x Vocab]
    pred_embeddings_ref,  # [Batch x Pos x Embed]
    labels_ref,  # [Batch x Pos]
    dot_ref,
    log_sum_exp_ref,
    max_logit_ref,
    *,
    Vocab: hax.Axis,
    Pos: hax.Axis,
    PosFull: hax.Axis,
    Batch: hax.Axis,
    BatchFull: hax.Axis,
    Embed: hax.Axis,
    Label: hax.Axis,
    dtype: jnp.dtype,
    logit_soft_cap: Optional[float] = None,
):
    pid_vocab = pl.program_id(0)
    pid_batch = pl.program_id(1)
    pid_seq = pl.program_id(2)

    vocab_start = pid_vocab * Vocab.size

    batch_mask = _make_tile_mask(Batch, BatchFull, pid_batch)
    pos_mask = _make_tile_mask(Pos, PosFull, pid_seq)

    vocab_mask = _make_tile_mask(Vocab, Label, pid_vocab)
    batch_pos_mask = batch_mask.broadcast_axis((Batch, Pos)) * pos_mask.broadcast_axis((Batch, Pos))

    pred_embeddings = hax.NamedArray(
        array=pl.load(
            pred_embeddings_ref,
            ...,
            mask=batch_pos_mask.array[..., None],
            other=0,
        ),
        axes=(Batch, Pos, Embed),
    )

    lm_head = hax.NamedArray(
        array=pl.load(
            lm_head_ref,
            ...,
            mask=vocab_mask.array[..., None],
            other=0,
        ),
        axes=(Vocab, Embed),
    )

    labels = hax.NamedArray(
        array=pl.load(
            labels_ref,
            ...,
            mask=batch_pos_mask.array,
            other=0,
        ),
        axes=(Batch, Pos),
    )

    logits = hax.dot(pred_embeddings, lm_head, axis=Embed)  # [Batch x Pos x Vocab]
    if logit_soft_cap is not None:
        logits = hax.tanh(logits / logit_soft_cap) * logit_soft_cap

    # Compute max only over valid vocab columns
    masked_for_max = hax.NamedArray(array=jnp.where(vocab_mask.array, logits.array, -jnp.inf), axes=logits.axes)
    max_logit = hax.max(masked_for_max, axis=Vocab)
    targets = _block_to_one_hot(labels, Vocab, vocab_start, dtype) * pos_mask * batch_mask

    # Mask out logits which aren't in the block. Must happen after max_logit but before dot.
    logits = logits * vocab_mask * pos_mask * batch_mask
    dot = hax.dot(logits, targets, axis=Vocab)  # [Batch x Pos]
    log_sum_exp = hax.log(hax.sum(hax.exp(logits - max_logit) * vocab_mask, axis=Vocab))

    # Zero out invalid rows explicitly in outputs
    dot = dot * pos_mask * batch_mask
    max_logit = max_logit * pos_mask * batch_mask
    log_sum_exp = log_sum_exp * pos_mask * batch_mask

    dot_ref[...] = dot.array[..., None]
    max_logit_ref[...] = max_logit.array[..., None]
    log_sum_exp_ref[...] = log_sum_exp.array[..., None]


def _block_cross_entropy_forward(
    ignore,
    pred: tuple[NamedArray, NamedArray],
    Contract: hax.Axis,
    Label: hax.Axis,
    Pos: hax.Axis,
    labels_y: NamedArray,
    *,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float] = None,
    batch_block_size: Optional[int] = None,
    seq_block_size: Optional[int] = None,
    vocab_block_size: int,
) -> tuple[tuple[NamedArray, NamedArray], tuple[NamedArray]]:
    """
    Forward pass for block-wise cross-entropy loss.

    This function computes the cross-entropy loss and log-sum-exp (`log_z`) in a block-wise manner
    to maintain memory efficiency by processing subsets of the vocabulary at a time.

    Args:
        ignore: Placeholder argument (unused).
        pred (Tuple[NamedArray, NamedArray]): Tuple containing predicted embeddings and language model head weights.
        Contract (hax.Axis): Axis to contract over (e.g., embedding axis).
        Label (hax.Axis): Label axis (e.g., vocabulary axis).
        Pos (hax.Axis): Position axis.
        labels_y (NamedArray): True target labels [Batch, Pos].
        dtype (Optional[jnp.dtype]): Data type for the computations.
        logit_soft_cap (Optional[float]): Optional soft cap for logits
        batch_block_size (Optional[int]): Size of each batch block for processing.
        seq_block_size (Optional[int]): Size of each sequence block for processing.
        vocab_block_size (int): Size of each block for processing.

    Returns:
        Tuple:
            - Tuple[NamedArray, NamedArray]: Computed loss and logsumexp.
            - Tuple[NamedArray]: Residuals needed for the backward pass.
    """

    num_vocab_blocks = math.ceil(Label.size / vocab_block_size)

    pred_embeddings, lm_head = pred
    Batch = pred_embeddings.axes[0]

    if batch_block_size is None:
        batch_block_size = Batch.size
    if seq_block_size is None:
        seq_block_size = Pos.size

    num_batch_blocks = math.ceil(Batch.size / batch_block_size)
    num_seq_blocks = math.ceil(Pos.size / seq_block_size)

    VocabSlice = Label.resize(vocab_block_size)
    BatchSlice = Batch.resize(batch_block_size)
    PosSlice = Pos.resize(seq_block_size)

    VocabBlock = Label.resize(num_vocab_blocks)

    block_dots, block_max_logits, block_logsumexps = pl.pallas_call(
        functools.partial(
            _block_cross_entropy_forward_kernel,
            logit_soft_cap=logit_soft_cap,
            Pos=PosSlice,
            PosFull=Pos,
            Batch=BatchSlice,
            BatchFull=Batch,
            Vocab=VocabSlice,
            Embed=Contract,
            Label=Label,
            dtype=dtype,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((Batch.size, Pos.size, VocabBlock.size), dtype=dtype),  # dot
            jax.ShapeDtypeStruct((Batch.size, Pos.size, VocabBlock.size), dtype=dtype),  # max_logit
            jax.ShapeDtypeStruct((Batch.size, Pos.size, VocabBlock.size), dtype=dtype),  # logsumexp
        ],
        grid=(num_vocab_blocks, num_batch_blocks, num_seq_blocks),
        in_specs=[
            pl.BlockSpec([VocabSlice.size, Contract.size], index_map=lambda v, b, s: (v, 0)),  # lm_head
            pl.BlockSpec(
                [BatchSlice.size, PosSlice.size, Contract.size],
                index_map=lambda v, b, s: (b, s, 0),
            ),  # embeddings
            pl.BlockSpec([BatchSlice.size, PosSlice.size], index_map=lambda v, b, s: (b, s)),  # labels
        ],
        out_specs=[
            pl.BlockSpec(
                [BatchSlice.size, PosSlice.size, 1],
                index_map=lambda v, b, s: (b, s, v),
            ),  # dot
            pl.BlockSpec(
                [BatchSlice.size, PosSlice.size, 1],
                index_map=lambda v, b, s: (b, s, v),
            ),  # max_logit
            pl.BlockSpec(
                [BatchSlice.size, PosSlice.size, 1],
                index_map=lambda v, b, s: (b, s, v),
            ),  # logsumexp
        ],
        interpret=use_interpret,
    )(lm_head.array, pred_embeddings.array, labels_y.array)

    block_max_logits = hax.NamedArray(array=block_max_logits, axes=(Batch, Pos, VocabBlock))
    block_logsumexps = hax.NamedArray(array=block_logsumexps, axes=(Batch, Pos, VocabBlock))
    block_dots = hax.NamedArray(array=block_dots, axes=(Batch, Pos, VocabBlock))

    max_logit = hax.max(block_max_logits, axis=VocabBlock)
    logsumexp = max_logit + hax.log(hax.sum(hax.exp(block_logsumexps + block_max_logits - max_logit), axis=VocabBlock))
    dot = hax.sum(block_dots, axis=VocabBlock)
    loss = logsumexp - dot

    return (loss, logsumexp), (logsumexp,)


def _block_cross_entropy_backward_kernel(
    lm_head_ref: jax.Array,
    pred_embeddings_ref: jax.Array,
    labels_y_ref: jax.Array,
    log_z_ref: jax.Array,
    grad_loss_ref: jax.Array,
    grad_log_z_ref: jax.Array,
    grad_embeddings_ref: jax.Array,
    grad_lm_head_ref: jax.Array,
    *,
    logit_soft_cap: Optional[float],
    Pos: hax.Axis,
    PosFull: hax.Axis,
    Batch: hax.Axis,
    BatchFull: hax.Axis,
    Vocab: hax.Axis,
    Embed: hax.Axis,
    Label: hax.Axis,
    dtype: jnp.dtype,
):
    """
    Pallas kernel for computing gradients in block-wise cross-entropy loss.
    """
    pid_vocab = pl.program_id(0)
    pid_batch = pl.program_id(1)
    pid_seq = pl.program_id(2)
    vocab_start = pid_vocab * Vocab.size

    batch_mask = _make_tile_mask(Batch, BatchFull, pid_batch)
    pos_mask = _make_tile_mask(Pos, PosFull, pid_seq)
    vocab_mask = _make_tile_mask(Vocab, Label, pid_vocab)
    batch_pos_mask = batch_mask.broadcast_axis((Batch, Pos)) * pos_mask.broadcast_axis((Batch, Pos))

    lm_head_block = hax.NamedArray(
        array=pl.load(lm_head_ref, ..., mask=vocab_mask.array[..., None], other=0),
        axes=(Vocab, Embed),
    )
    embeddings = hax.NamedArray(
        array=pl.load(pred_embeddings_ref, ..., mask=batch_pos_mask.array[..., None], other=0),
        axes=(Batch, Pos, Embed),
    )
    labels = hax.NamedArray(
        array=pl.load(labels_y_ref, ..., mask=batch_pos_mask.array, other=0),
        axes=(Batch, Pos),
    )
    log_z = hax.NamedArray(
        array=pl.load(log_z_ref, ..., mask=batch_pos_mask.array, other=0),
        axes=(Batch, Pos),
    )
    grad_loss = hax.NamedArray(
        array=pl.load(grad_loss_ref, ..., mask=batch_pos_mask.array, other=0),
        axes=(Batch, Pos),
    )
    grad_log_z = hax.NamedArray(
        array=pl.load(grad_log_z_ref, ..., mask=batch_pos_mask.array, other=0),
        axes=(Batch, Pos),
    )

    logits = hax.dot(embeddings, lm_head_block, axis=Embed)  # [Batch, Pos, Vocab]
    if logit_soft_cap is not None:
        logits = hax.tanh(logits / logit_soft_cap) * logit_soft_cap

    probs = hax.exp(logits - log_z) * vocab_mask

    targets = _block_to_one_hot(labels, Vocab, vocab_start, dtype) * pos_mask * batch_mask

    grad_logits = grad_loss * (probs - targets) + grad_log_z * probs  # [Batch, Pos, Vocab]
    grad_logits = grad_logits * vocab_mask
    if logit_soft_cap is not None:
        jac = 1.0 - (logits / logit_soft_cap) ** 2
        grad_logits = grad_logits * jac

    grad_logits = grad_logits * pos_mask * batch_mask

    grad_embeddings_block = hax.dot(grad_logits, lm_head_block, axis=Vocab)  # [Batch, Pos, Embed]
    grad_lm_head_block = hax.sum(hax.dot(grad_logits, embeddings, axis=Pos), axis=Batch)  # [Vocab, Embed]

    pl.store(grad_embeddings_ref, ..., grad_embeddings_block.array[..., None])  # last dim is Block=1 slice
    pl.store(grad_lm_head_ref, ..., grad_lm_head_block.array[None, None, ...])


def _block_cross_entropy_backward(
    residuals: tuple[NamedArray,],
    grad_in: tuple[NamedArray, NamedArray],
    ignore,
    pred: tuple[NamedArray, NamedArray],
    Contract: hax.Axis,
    Label: hax.Axis,
    Pos: hax.Axis,
    labels_y: NamedArray,
    *,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float] = None,
    batch_block_size: Optional[int] = None,
    seq_block_size: Optional[int] = None,
    vocab_block_size: int,
) -> tuple[NamedArray, NamedArray]:
    """
    Compute the gradients of the block-wise cross-entropy loss using Pallas.

    Args:
        residuals (tuple[NamedArray, NamedArray]): Residuals from the forward pass.
        grad_in (tuple[NamedArray, NamedArray]): Incoming gradients.
        pred (tuple[NamedArray, NamedArray]): Predictions.
        Contract (hax.Axis): Axis to contract over.
        Label (hax.Axis): Label axis.
        Pos (hax.Axis): Position axis.
        labels_y (NamedArray): Target labels.
        dtype (Optional[jnp.dtype]): Data type for the loss.
        logit_soft_cap (Optional[float]): Optional soft cap for logits.
        batch_block_size (Optional[int]): Size of each batch block for processing.
        seq_block_size (Optional[int]): Size of each sequence block for processing.
        vocab_block_size (int): Size of each block for processing.

    Returns:
        tuple[NamedArray, NamedArray]: Gradients.
    """
    (log_z,) = residuals
    grad_loss, grad_log_z = grad_in

    if vocab_block_size > Label.size:
        vocab_block_size = Label.size

    num_vocab_blocks = math.ceil(Label.size / vocab_block_size)
    pred_embeddings, lm_head = pred

    VocabSlice = Label.resize(vocab_block_size)
    VocabBlock = Label.resize(num_vocab_blocks)
    Batch = pred_embeddings.axes[0]

    if batch_block_size is None or batch_block_size > Batch.size:
        batch_block_size = Batch.size
    if seq_block_size is None or seq_block_size > Pos.size:
        seq_block_size = Pos.size

    num_batch_blocks = math.ceil(Batch.size / batch_block_size)
    num_pos_blocks = math.ceil(Pos.size / seq_block_size)

    BatchBlock = hax.Axis("batch_block", num_batch_blocks)
    PosBlock = hax.Axis("pos_block", num_pos_blocks)
    BatchSlice = Batch.resize(batch_block_size)
    PosSlice = Pos.resize(seq_block_size)

    # Grads may be None if corresponding output wasn't used
    if grad_loss.array is None:
        grad_loss = hax.zeros((Batch, Pos), dtype=pred_embeddings.dtype)
    if grad_log_z.array is None:
        grad_log_z = hax.zeros((Batch, Pos), dtype=pred_embeddings.dtype)

    grad_embedding_out_shape = (Batch, Pos, Contract, VocabBlock)
    grad_lm_head_out_shape = (BatchBlock, PosBlock, Label, Contract)

    grad_embeddings_blocks, grad_lm_head_blocks = pl.pallas_call(
        functools.partial(
            _block_cross_entropy_backward_kernel,
            logit_soft_cap=logit_soft_cap,
            Pos=PosSlice,
            PosFull=Pos,
            Batch=BatchSlice,
            BatchFull=Batch,
            Vocab=VocabSlice,
            Embed=Contract,
            Label=Label,
            dtype=dtype,
        ),
        out_shape=[
            # grad_embeddings - aggregated over vocab
            jax.ShapeDtypeStruct([ax.size for ax in grad_embedding_out_shape], dtype=pred_embeddings.dtype),
            # grad_lm_head - aggregated over batch and pos
            jax.ShapeDtypeStruct([ax.size for ax in grad_lm_head_out_shape], dtype=lm_head.dtype),
        ],
        grid=(num_vocab_blocks, num_batch_blocks, num_pos_blocks),
        in_specs=[
            pl.BlockSpec([VocabSlice.size, Contract.size], index_map=lambda v, b, s: (v, 0)),  # lm_head
            pl.BlockSpec(
                [BatchSlice.size, PosSlice.size, Contract.size],
                index_map=lambda v, b, s: (b, s, 0),
            ),  # embeddings
            pl.BlockSpec([BatchSlice.size, PosSlice.size], index_map=lambda v, b, s: (b, s)),  # labels
            pl.BlockSpec([BatchSlice.size, PosSlice.size], index_map=lambda v, b, s: (b, s)),  # log_z
            pl.BlockSpec([BatchSlice.size, PosSlice.size], index_map=lambda v, b, s: (b, s)),  # grad_loss
            pl.BlockSpec([BatchSlice.size, PosSlice.size], index_map=lambda v, b, s: (b, s)),  # grad_log_z
        ],
        out_specs=[
            pl.BlockSpec(
                [BatchSlice.size, PosSlice.size, Contract.size, 1],
                index_map=lambda v, b, s: (b, s, 0, v),
            ),  # grad_embeddings - aggregated over vocab
            pl.BlockSpec(
                [1, 1, VocabSlice.size, Contract.size],
                index_map=lambda v, b, s: (b, s, v, 0),
            ),  # grad_lm_head - aggregated over batch and pos
        ],
        interpret=use_interpret,
    )(
        lm_head.array,
        pred_embeddings.array,
        labels_y.array,
        log_z.array,
        grad_loss.array,
        grad_log_z.array,
    )

    grad_embeddings_blocks = hax.NamedArray(array=grad_embeddings_blocks, axes=grad_embedding_out_shape)
    grad_embeddings = hax.sum(grad_embeddings_blocks, axis=VocabBlock)

    grad_lm_head = hax.NamedArray(array=grad_lm_head_blocks, axes=grad_lm_head_out_shape)
    grad_lm_head = hax.sum(grad_lm_head, axis=(BatchBlock, PosBlock))

    return (grad_embeddings, grad_lm_head)


_blockwise_cross_entropy_loss.def_fwd(_block_cross_entropy_forward)
_blockwise_cross_entropy_loss.def_bwd(_block_cross_entropy_backward)
