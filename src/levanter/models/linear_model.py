import dataclasses
from dataclasses import dataclass
from typing import Optional, Type
import math

import jax.random as jrandom
from jaxtyping import PRNGKeyArray
import jax.debug as debug

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call

from levanter.models.lm_model import LmConfig, LmHeadModel


@LmConfig.register_subclass("linear")
@dataclass(frozen=True)
class LinearConfig(LmConfig):
    dim: int = 512
    seq_len: int = 2048
    initializer_range: float = 0.02

    # Axes
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.dim))

    @property
    def model_type(self) -> Type["LinearModel"]:
        return LinearModel


class LinearModel(LmHeadModel[LinearConfig]):
    _config: LinearConfig = dataclasses.field(init=False)
    embedding: hnn.Embedding
    lm_head: hnn.Linear

    @property
    def config(self) -> LinearConfig:
        return self._config

    @property
    def Vocab(self) -> Axis:
        return self.lm_head.Out

    @classmethod
    def init(cls, Vocab: Axis, config: LinearConfig, *, key: PRNGKeyArray) -> "LinearModel":
        k_emb, k_head = jrandom.split(key, 2)
        embedding = hnn.Embedding.init(
            Vocab, config.Embed, key=k_emb,
            initializer_range=math.sqrt(config.Embed.size ** 2.0 / Vocab.size)
        )
        lm_head = hnn.Linear.init(
            In=config.Embed, Out=Vocab, key=k_head, use_bias=False,
        )
        return LinearModel(config, embedding, lm_head)

    def __init__(self, config: LinearConfig, embedding: hnn.Embedding, lm_head: hnn.Linear):
        super().__init__()
        self._config = config
        self.embedding = embedding
        self.lm_head = lm_head

    @named_call
    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask=None,
        *,
        key=None,
        pos_ids: Optional[NamedArray] = None,
    ) -> NamedArray:
        x = self.embedding(input_ids)
        logits = self.lm_head(x)
        #debug.print("LinearModel.__call__ > input_ids: {}", input_ids)
        #debug.print("LinearModel.__call__ > x: {}", x)
        #debug.print("LinearModel.__call__ > logits: {}", logits)
        return logits

    def activations(self, input_ids: NamedArray, *args, **kwargs) -> NamedArray:
        #debug.print("LinearModel.activations > input_ids: {}", input_ids)
        #debug.print("LinearModel.activations > embedding: {}", self.embedding.weight)
        #debug.print("LinearModel.activations > lm_head: {}", self.lm_head.weight.array.T)
        x = self.embedding(input_ids)
        #debug.print("LinearModel.activations > x: {}", x)
        logits = self.lm_head(x)
        #debug.print("LinearModel.activations > logits: {}", logits)
        return x

    def get_lm_head(self) -> NamedArray:
        return self.lm_head.weight

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "LinearModel":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = hax.jax_utils.maybe_rng_split(key, 2)

        new_embedding = self.embedding.resize_embeddings(new_size, key=k1)
        new_lm_head_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
        new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_head_matrix)

        return dataclasses.replace(self, embedding=new_embedding, lm_head=new_lm_head)