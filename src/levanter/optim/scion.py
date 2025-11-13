# Revised by Wanyun Xie (wanyun.xie@epfl.ch)
# 1. Changed the scaling factor calculation in the `transform_linear_layer` function to use sqrt(d_out/d_in), removing the conditional maximum with 1. 
# 2. Added the scaling factor 1/d_in in `scale_by_sign` with controllable `sign_normalized` parameter.
# 3. Added layer-specific `sign_normalized` control: Embedding layers use sign_normalized=False, lm_head layers use sign_normalized=True.
# 4. Added scaling factors `spectral_radius` and `sign_radius` to allow for additional scaling after orthogonalization.
# 5. Removed weight decay, since Scion does a convex combination with learning rate directly.
# 6. Removed scion_to_signum_lr as its role is replaced by the scaling factors `spectral_radius` and `sign_radius`.
# 7. Added BiasRMS_transform that applies RMS normalization to bias parameters.

# Other tips:
# - Note that we usually set `sign_radius` larger than `spectral_radius`. This is opposite to the original setting `signum_lr = learning_rate * 0.25`.
# - Tuning lr and spectral_radius/sign_radius is important and can be transferred from a small proxy model. `spectral_radius=50` and `sign_radius=3000` is what we set for modded-nanogpt, not for this one.
# - Could choose constrained or unconstrained version of Scion by setting `unconstrained` flag.
# - Not sure how necessary to set different momentum for Spectral and Sign.
# - Not sure how necessary to set `max_grad_norm` for Sign.
# - It's okay to still keep Sign for bias parameters, but without 1/d scaling as we suggested in Table 6 of Scion paper https://arxiv.org/abs/2502.07529.


import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import haliax
from haliax.nn import Linear

from levanter.optim.config import OptimizerConfig
from levanter.optim.util import map_flattened_linear_layers
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("scion")
@dataclass(frozen=True)
class ScionConfig(OptimizerConfig):
    """
    Scion optimizer configuration
    cf:
    Original Paper: https://arxiv.org/abs/2502.07529
    """

    lr: float = 0.02
    # scion_to_signum_lr: float = 0.25  
    momentum: float = 0.95
    backend_steps: int = 10  # Number of steps for Newton-Schulz orthogonalization
    beta1: float = 0.9
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    unconstrained: bool = False
    spectral_radius: float = 50
    sign_radius : float = 3000

    def build(self, num_train_steps):
        """
        Creates the optimizer.
        """
        learning_rate_schedule = self.lr_scheduler(num_train_steps)

        def optimizer(learning_rate):
            # signum_lr = learning_rate * self.scion_to_signum_lr
            def Spectral_transform():
                components = []
                components.append(scale_with_spectral(self.momentum, self.backend_steps, self.epsilon, self.spectral_radius))
                if not self.unconstrained:
                    components.append(optax.add_decayed_weights(1, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            def Sign_lmhead_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_sign(self.beta1, self.sign_radius, sign_normalized=True))
                if not self.unconstrained:
                    components.append(optax.add_decayed_weights(1, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            def Sign_embedding_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_sign(self.beta1, self.sign_radius, sign_normalized=False))
                if not self.unconstrained:
                    components.append(optax.add_decayed_weights(1, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            def BiasRMS_transform():
                components = []
                components.append(scale_by_biasrms(momentum=self.momentum, eps=self.epsilon))
                if not self.unconstrained:
                    components.append(optax.add_decayed_weights(1, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            transformations = {
                "Spectral": Spectral_transform(),
                "Sign_lmhead": Sign_lmhead_transform(),
                "Sign_embedding": Sign_embedding_transform(),
                "BiasRMS": BiasRMS_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule)

    def create_mask(self, params):
        """
        Creates a mask that labels parameters as 'Spectral' or 'Sign' based on their
        dimensionality and module path, using different Sign transforms for Embedding and lm_head parameters.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str:
                return "Sign_embedding"  # For Embedding layers: sign_normalized=False
            elif "lm_head" in path_str:
                return "Sign_lmhead"  # For lm_head layers: sign_normalized=True
            elif isinstance(param, Linear):
                # scion for linear layers
                return dataclasses.replace(param, weight="Spectral", bias="BiasRMS" if param.bias is not None else None)
            else:
                return "Sign_embedding"

        return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


class ScaleByScionState(NamedTuple):
    """State for the Scion algorithm."""

    momentum_buffer: optax.Updates


def scale_by_sign(momentum=0.95, sign_radius=3000, sign_normalized=True):
    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)  # First moment
        return ScaleByScionState(momentum_buffer=momentum_buffer)

    def update_fn(updates, state, params=None):
        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + (1 - momentum) * g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )

        def apply_sign_scaling(u):
            if u is None:
                return None
            if sign_normalized and u.ndim >= 2:
                scale = sign_radius / u.shape[1]
            else:
                scale = sign_radius
            return scale * jnp.sign(u)

        updates = jax.tree_map(apply_sign_scaling, buf, is_leaf=lambda x: x is None)

        return updates, ScaleByScionState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_with_spectral(momentum=0.95, steps=5, eps=1e-8, spectral_radius=50):
    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)  # First moment
        return ScaleByScionState(momentum_buffer=momentum_buffer)

    def update_fn(updates, state, params=None):
        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + (1 - momentum) * g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )
        updates = buf

        def transform_linear_layer(layer: haliax.nn.Linear):
            assert layer.weight.ndim == 2

            updated_weight_array = zeropower_via_newtonschulz5(layer.weight.array, steps=steps, eps=eps)

            # scale = jnp.sqrt(jnp.maximum(1, updated_weight_array.shape[0] / updated_weight_array.shape[1]))
            scale = jnp.sqrt(updated_weight_array.shape[0] / updated_weight_array.shape[1])
            updated_weight_array *= scale * spectral_radius

            updated_weight = dataclasses.replace(layer.weight, array=updated_weight_array)

            return dataclasses.replace(layer, weight=updated_weight)  # type: ignore

        updates = map_flattened_linear_layers(transform_linear_layer, updates)

        return updates, ScaleByScionState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_biasrms(momentum=0.95, eps=1e-8):
    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)  # First moment
        return ScaleByScionState(momentum_buffer=momentum_buffer)

    def update_fn(updates, state, params=None):
        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + (1 - momentum) * g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )

        def rms_normalize_and_sign(u):
            if u is None:
                return None
            rms_values = jnp.sqrt(jnp.mean(u ** 2, axis=0, keepdims=True))
            u_normalized = u / (rms_values + eps)
            return u_normalized
        
        updates = jax.tree_map(rms_normalize_and_sign, buf, is_leaf=lambda x: x is None)

        return updates, ScaleByScionState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def zeropower_via_newtonschulz5(X, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    chex.assert_rank(X, 2)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X /= jnp.linalg.norm(X) + eps  # Ensure top singular value <= 1
    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    # https://x.com/leloykun/status/1874358290093924849

    return X
