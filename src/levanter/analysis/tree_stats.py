# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import jax
import jax.numpy as jnp
import optax

import haliax
import haliax.nn
from haliax import NamedArray, is_named_array
from haliax.jax_utils import is_jax_array_like

from levanter.tracker.histogram import Histogram
from levanter.utils import jax_utils


def summary_statistics_for_tree(
    prefix: str,
    tree: Any,
    split_scan_layers: bool,
    *,
    include_histogram: bool = False,
    include_norms: bool = True,
    include_per_parameter_norms: bool = True,
) -> dict[str, jax.Array | Histogram]:
    """
    Computes the summary statistics for a tree of (named) arrays.

    This function is designed to allow you to emulate the behavior of Wandb's PyTorch-only built-in gradient logging,
    but also works for any PyTree. It computes the Froebinius norm of each array,
    and optionally the histogram as well.

    Args:
        prefix: The prefix to use for logging.
        tree: The tree of arrays to compute the summary statistics for.
        split_scan_layers: Whether to split the scan layers into separate histograms/norms. Recommended.
        include_norms: Whether to include norms of the gradients. This increases overhead significantly.
        include_histogram: Whether to include histograms of the gradients. This increases overhead significantly.
        include_per_parameter_norms: Whether to include per-parameter norms.

    Returns:
        A dictionary of summary statistics.

    """
    if split_scan_layers:
        is_leaf = lambda n: isinstance(n, haliax.nn.Stacked) or is_named_array(n)  # noqa: E731
    else:
        is_leaf = is_named_array

    def _rec_log_magnitudes(norms, hists, path_prefix, tree):
        leaf_key_paths = jax_utils.leaf_key_paths(tree, prefix=path_prefix, is_leaf=is_leaf)
        del path_prefix
        for key_path, g in zip(
            jax.tree.leaves(leaf_key_paths, is_leaf=is_leaf),
            jax.tree.leaves(tree, is_leaf=is_leaf),
            strict=True,
        ):
            if split_scan_layers and isinstance(g, haliax.nn.Stacked):
                vmapped_norms, vmapped_hists = haliax.vmap(_rec_log_magnitudes, g.Block)({}, {}, "", g.stacked)

                for k, v in vmapped_norms.items():
                    for i in range(g.Block.size):
                        norms[f"{key_path}.{i}.{k}"] = v[i]

                for k, v in vmapped_hists.items():
                    for i in range(g.Block.size):
                        hists[f"{key_path}.{i}.{k}"] = jax.tree.map(lambda x: x[i] if is_jax_array_like(x) else x, v)

            elif isinstance(g, NamedArray):
                if include_norms:
                    norms[key_path] = optax.global_norm(g)
                if include_histogram:
                    hist = Histogram.from_named_array(g)
                    hists[key_path] = hist
            elif is_jax_array_like(g):
                if include_norms:
                    norms[key_path] = optax.global_norm(g)

                if include_histogram:
                    with jax.named_scope(f"histogram({prefix}/{key_path})"):
                        hist = Histogram.from_array(g)
                        hists[key_path] = hist

        return norms, hists

    norms_to_log: dict[str, jax.Array] = {}
    hists_to_log: dict[str, Histogram] = {}

    _rec_log_magnitudes(norms_to_log, hists_to_log, None, tree)

    to_log: dict[str, jax.Array | Histogram] = {}

    for key, value in norms_to_log.items():
        if include_per_parameter_norms:
            to_log[f"{prefix}/norm/{key}"] = value

    to_log[f"{prefix}/norm/total"] = optax.global_norm(tree)

    for key, hist in hists_to_log.items():
        to_log[f"{prefix}/hist/{key}"] = hist

    return to_log


def nu_dead_neuron_tracking(
    prefix: str,
    tree: Any,
    *,
    include_histograms: bool,
    split_scan_layers: bool = True,
) -> dict[str, Histogram | jnp.array]:
    """
    Compute stats and histograms of per-row and per-column gradient flow for Linear layers.

    This looks at the optimizer second-moment (``nu``) values and sums over the
    input and output dimensions of each Linear layer. If either sum is nearly
    zero, the corresponding row or column is effectively a dead neuron.

    Args:
        prefix: Prefix to use when constructing log keys.
        tree: PyTree containing ``nu`` values structured like the model.
        split_scan_layers: Whether to split ``Stacked`` layers into individual layers


    Returns:
        A mapping from key names to histograms.
    """

    hists: dict[str, Histogram | jnp.ndarray] = {}

    def _rec(path_prefix: str | None, subtree: Any):
        is_leaf = lambda n: isinstance(n, haliax.nn.Linear) or isinstance(n, haliax.nn.Stacked)
        leaf_key_paths = jax_utils.leaf_key_paths(subtree, prefix=path_prefix, is_leaf=is_leaf)

        for key_path, node in zip(
            jax.tree.leaves(leaf_key_paths, is_leaf=is_leaf),
            jax.tree.leaves(subtree, is_leaf=is_leaf),
            strict=True,
        ):
            if isinstance(node, haliax.nn.Stacked) and split_scan_layers:
                for idx, sub in enumerate(node.unstacked()):
                    _rec(f"{key_path}.{idx}", sub)
            elif isinstance(node, haliax.nn.Linear):
                if node.weight is None:
                    continue
                row_flow = haliax.ones(node.Out).dot(node.weight, axis=node.Out)
                col_flow = node.weight.dot(haliax.ones(node.In), axis=node.In)
                if include_histograms:
                    hists[f"{key_path}/row"] = Histogram.from_named_array(row_flow)
                    hists[f"{key_path}/col"] = Histogram.from_named_array(col_flow)
                # compute z-scores of log transformed flows and log outliers
                log_row = haliax.log(row_flow + 1e-14)  # this should be very small because row_flow is
                log_col = haliax.log(col_flow + 1e-14)
                mean_row = haliax.mean(log_row)
                std_row = haliax.std(log_row)
                mean_col = haliax.mean(log_col)
                std_col = haliax.std(log_col)
                z_row = (log_row - mean_row) / (std_row + 1e-14)
                z_col = (log_col - mean_col) / (std_col + 1e-14)
                hists[f"{key_path}/row/dead"] = haliax.sum(z_row < -3).scalar() / z_row.size
                hists[f"{key_path}/col/dead"] = haliax.sum(z_col < -3).scalar() / z_col.size

    _rec(None, tree)

    return {f"{prefix}/hist/{k}": v for k, v in hists.items()}
