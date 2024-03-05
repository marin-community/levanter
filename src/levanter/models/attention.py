import math
from typing import Optional, Union, overload

import equinox as eqx
import jax.numpy as jnp
from jax.lib import xla_bridge
from jaxtyping import PRNGKeyArray

import haliax
from haliax import Axis, AxisSelection, AxisSelector, NamedArray
from haliax.nn.attention import causal_mask, combine_masks_and, combine_masks_or
from haliax.types import PrecisionLike


def dot_product_attention(
    QPos: AxisSelector,
    KPos: AxisSelection,
    Key: AxisSelector,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[Union["AttentionMask", NamedArray]] = None,
    bias: Optional[NamedArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    use_flash: bool = False,
    flash_block_size: Optional[int] = None,
    dropout: float = 0.0,
    *,
    inference: bool = True,
    prng: Optional[PRNGKeyArray] = None,
):
    """
    This method is similar to [haliax.nn.attention.dot_product_attention][] but uses the [AttentionMask][] class,
    which we might move to haliax.nn.attention in the future.

    Unlike the Haliax version, it requires that the Q and K already be different.

    Args:
        Key: Size of key dimension
        QPos: Axis of query sequence length. Can be an AxisSpec to attend along more than one axis.
        KPos: Axis of key sequence length. Can be an AxisSpec to attend along more than one axis.
        query: shape at least {QPos, KeySize}
        key: shape at least {KPos, KeySize}
        value: shape at least {KPos, ValueSize}
        mask: attention mask
        bias: Optional[NamedArray] broadcast compatible with (KeySize, QPos, KPos). Should be float
        attention_dtype: Optional dtype to use for attention
        precision: PrecisionLike for dot product. See precision argument to jax.lax.dot_general
        use_flash: whether to use flash attention
        flash_block_size: block size for flash attention. If None, will use an appropriate default
        dropout: dropout rate
        inference: whether to use inference mode
        prng: PRNGKeyArray for dropout
    Returns:
        NamedArray of shape (value.axes - KPos + QPos)
    """
    if QPos == KPos:
        raise ValueError("QPos and KPos must be different")

    if use_flash:
        accelerator_type = xla_bridge.get_backend().platform.lower()

        if accelerator_type == "tpu" or accelerator_type == "cpu":
            from levanter.models.flash_attention import flash_attention

            return flash_attention(
                QPos,
                KPos,
                Key,
                query,
                key,
                value,
                block_size=flash_block_size,
                mask=mask,
                bias=bias,
                dropout=dropout,
                inference=inference,
                key=prng,
                dtype=attention_dtype,
                precision=precision,
            )

        elif accelerator_type == "gpu":

            try:
                # TODO: Find better way than hard coding dimension variable names
                query.resolve_axis("heads")
                key.resolve_axis("heads")
                num_q_heads_dim = "heads"
                num_kv_heads_dim = "heads"

            except ValueError:
                from levanter.models.flash_attention import flash_attention

                # Using MQA / GQA, calling jax flash attention for now
                # TODO: Figure out re-shaping of query and key/value dims for
                # GQA so TransformerEngine cross_fused_attn can be called instead
                return flash_attention(
                    QPos,
                    KPos,
                    Key,
                    query,
                    key,
                    value,
                    block_size=flash_block_size,
                    mask=mask,
                    bias=bias,
                    dropout=dropout,
                    inference=inference,
                    key=prng,
                    dtype=attention_dtype,
                    precision=precision,
                )

            from transformer_engine.jax.fused_attn import AttnBiasType, AttnMaskType, cross_fused_attn, self_fused_attn

            try:
                # TODO: Change hard coded 'batch' variable name
                query.resolve_axis("batch")
                batch_dim = "batch"
                batch_size = query.shape[batch_dim]
                no_batch_dim = False

                # TransformerEngine self_fused_attn looks for qkv shape:
                # batch_shape, max_seqlen, nqkv, num_heads, head_dim; where nqkv = 3
                q_ = haliax.rearrange(query, (batch_dim, QPos, num_q_heads_dim, Key)).array
                k_ = haliax.rearrange(key, (batch_dim, KPos, num_kv_heads_dim, Key)).array
                v_ = haliax.rearrange(value, (batch_dim, KPos, num_kv_heads_dim, Key)).array

            except ValueError:
                no_batch_dim = True
                batch_size = 1

                # TransformerEngine self_fused_attn looks for qkv shape:
                # batch_shape, max_seqlen, nqkv, num_heads, head_dim; where nqkv = 3
                q_ = haliax.rearrange(query, (QPos, num_q_heads_dim, Key)).array
                q_ = jnp.expand_dims(q_, 0)

                k_ = haliax.rearrange(key, (KPos, num_kv_heads_dim, Key)).array
                k_ = jnp.expand_dims(k_, 0)

                v_ = haliax.rearrange(value, (KPos, num_kv_heads_dim, Key)).array
                v_ = jnp.expand_dims(v_, 0)

            # Mask is generated by transformer engine based on AttnMaskType
            if type(mask) is NamedArray:
                raise NotImplementedError(
                    "Custom NamedArray masks are not implemented for flash attention. Please pass an AttentionMask"
                    " object"
                )
            elif type(mask) is AttentionMask:
                if mask.causal():
                    attn_mask_type = AttnMaskType.CAUSAL_MASK

                    QPos = query.resolve_axis(QPos)
                    KPos = key.resolve_axis(KPos)
                    fused_attn_mask = mask.materialize(QPos, KPos)

                    if fused_attn_mask is not None:
                        fused_attn_mask = fused_attn_mask.array
                        fused_attn_mask = jnp.dstack([fused_attn_mask] * batch_size)
                    else:
                        raise ValueError(
                            "If AttentionMask is causal, the materialized array should never be None. Something is"
                            " wrong."
                        )
                else:
                    raise NotImplementedError(
                        "Non-Causal masks are not implemented for flash attention. Please pass an AttentionMask object"
                    )
            else:
                attn_mask_type = AttnMaskType.NO_MASK
                fused_attn_mask = jnp.ones((batch_size, query.axis_size(QPos), key.axis_size(KPos)))

            scaling_factor = 1 / math.sqrt(float(query.axis_size(Key)))
            is_training = not inference

            # TODO: bias type is probably also configurable
            attn_bias_type = AttnBiasType.NO_BIAS
            fused_attn_bias = None
            if bias:
                raise NotImplementedError(
                    "Using bias with flash attention on GPU is not currently implemented. Try setting"
                    " model.use_bias=False"
                )

            # Vanilla multi-head self-attention case
            # Head mismatch implies MultiQueryAttn or GroupQueryAttn
            # Seq len mismatch implies cross-attn
            if q_.shape == k_.shape:
                assert (
                    q_.shape == k_.shape == v_.shape
                ), "Q, K, and V projections must have the same dimensions for multi-head self-attention with flash"

                qkv = jnp.stack((q_, k_, v_), axis=2)

                attn_output = self_fused_attn(
                    qkv=qkv,  # jnp.ndarray,
                    bias=fused_attn_bias,  # jnp.ndarray,
                    mask=fused_attn_mask,  # jnp.ndarray,
                    seed=prng,  # jnp.ndarray,
                    attn_bias_type=attn_bias_type,  # AttnBiasType,
                    attn_mask_type=attn_mask_type,  # AttnMaskType,
                    scaling_factor=scaling_factor,  # float,
                    dropout_probability=dropout,  # float,
                    is_training=is_training,  # bool,
                )

            else:  # Cross-Attn, GQA, or MQA Case
                assert (
                    k_.shape == v_.shape
                ), "K and V must have the same dimensions for cross attention, GQA, or MQA with flash attention."

                q = q_
                kv = jnp.stack((k_, v_), axis=2)

                attn_output = cross_fused_attn(
                    q=q,
                    kv=kv,
                    bias=fused_attn_bias,
                    mask=fused_attn_mask,
                    seed=prng,
                    attn_bias_type=attn_bias_type,
                    attn_mask_type=attn_mask_type,
                    scaling_factor=scaling_factor,
                    dropout_probability=dropout,
                    is_training=is_training,
                )

            if no_batch_dim:
                attn_output = jnp.squeeze(attn_output)
                attn_output = haliax.named(attn_output, (QPos, num_q_heads_dim, Key))
            else:
                attn_output = haliax.named(attn_output, (batch_dim, QPos, num_q_heads_dim, Key))

            return attn_output

    else:
        return simple_attention_with_dropout(
            QPos, KPos, Key, query, key, value, mask, bias, inference, dropout, attention_dtype, precision, prng=prng
        )


def simple_attention_with_dropout(
    QPos: Axis,
    KPos: Axis,
    Key: Axis,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[Union[NamedArray, "AttentionMask"]] = None,
    bias: Optional[NamedArray] = None,
    inference: bool = False,
    dropout: float = 0.0,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    *,
    prng: Optional[PRNGKeyArray] = None,
):
    QPos = query.resolve_axis(QPos)
    KPos = key.resolve_axis(KPos)
    m = materialize_mask(mask, QPos, KPos)
    weights = haliax.nn.attention.dot_product_attention_weights(
        Key, KPos, query, key, mask=m, bias=bias, attention_dtype=attention_dtype, precision=precision
    )
    weights = haliax.nn.dropout(weights, dropout, key=prng, inference=inference)
    return haliax.dot(KPos, weights, value)


class AttentionMask(eqx.Module):
    """

    !!! warning
        This class is still experimental. I'm not super happy with it yet.

    Represents an attention mask in a structured way to make it easier to optimize attention for particular use cases
    (causal, prefix, etc.). It is anticipated that this will be extended with new types of masks as needed.

    In general, it should be safe to batch Attention Masks, but it is important that *all members of a batch have the
    same sequence of combined masks*. Otherwise, the batching will not work and you'll get weird errors

    The interface exposed by this class is designed to work well with the attention functions in this module as
    well as something like flash attention.

    A mask can be materialized, in which case it returns the mask as a NamedArray.
    """

    is_causal: bool = eqx.static_field()
    explicit_mask: Optional[NamedArray] = None
    # TODO: add sequence packing
    # TODO: add prefixlm
    # cf https://github.com/google-research/t5x/blob/51a99bff8696c373cc03918707ada1e98cbca407/t5x/examples/decoder_only/layers.py#L978

    def materialize(
        self, QPos: Axis, KPos: Axis, q_slice: Optional[haliax.dslice] = None, k_slice: Optional[haliax.dslice] = None
    ) -> Optional[NamedArray]:
        """
        Materialize the mask as a NamedArray. This is useful for attention functions that don't support masks,
        or for the inner loop
        """
        if q_slice is None:
            q_slice = haliax.dslice(0, QPos.size)
        if k_slice is None:
            k_slice = haliax.dslice(0, KPos.size)

        if self.is_causal:
            causal = causal_mask(QPos.resize(q_slice.size), KPos.resize(k_slice.size), q_slice.start, k_slice.start)
        else:
            causal = None

        if self.explicit_mask is not None:
            explicit = self.explicit_mask[QPos, q_slice, KPos, k_slice]
        else:
            explicit = None

        return combine_masks_and(causal, explicit)

    @staticmethod
    def causal() -> "AttentionMask":
        return AttentionMask(is_causal=True)

    @staticmethod
    def explicit(mask: NamedArray) -> "AttentionMask":
        return AttentionMask(is_causal=False, explicit_mask=mask)

    def __and__(self, other) -> "AttentionMask":
        is_causal = self.is_causal and other.is_causal
        explicit_mask = combine_masks_and(self.explicit_mask, other.explicit_mask)
        return AttentionMask(is_causal=is_causal, explicit_mask=explicit_mask)

    def __or__(self, other) -> "AttentionMask":
        is_causal = self.is_causal or other.is_causal
        explicit_mask = combine_masks_or(self.explicit_mask, other.explicit_mask)
        return AttentionMask(is_causal=is_causal, explicit_mask=explicit_mask)


@overload
def materialize_mask(
    mask: NamedArray | AttentionMask,
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> NamedArray:
    ...


@overload
def materialize_mask(
    mask: Optional[NamedArray | AttentionMask],
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> Optional[NamedArray]:
    ...


def materialize_mask(
    mask: Optional[NamedArray | AttentionMask],
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> Optional[NamedArray]:
    """
    Materialize an attention mask if it is an AttentionMask. Otherwise, just return it.
    """
    if isinstance(mask, AttentionMask):
        mask = mask.materialize(QPos, KPos, q_slice=q_slice, k_slice=k_slice)
        return mask
    elif isinstance(mask, NamedArray):
        if q_slice is not None or k_slice is not None:
            if q_slice is None:
                q_slice = haliax.dslice(0, QPos.size)
            if k_slice is None:
                k_slice = haliax.dslice(0, KPos.size)
            mask = mask[QPos, q_slice, KPos, k_slice]

        return mask
    else:
        assert mask is None
        return None


# TODO: padding mask
# TODO: FCM mask?
# TODO: sequence packing mask
