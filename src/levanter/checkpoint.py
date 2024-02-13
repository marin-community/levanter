import dataclasses
import datetime
import json
import logging
import os
import pathlib
import urllib.parse
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, List, Optional, Sequence, TypeVar, Union

import equinox
import fsspec
import jax
import jax.numpy as jnp
from draccus import field
from fsspec import AbstractFileSystem
from jax.experimental.multihost_utils import broadcast_one_to_all, sync_global_devices
from jaxtyping import PyTree

import haliax.partitioning

from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore, tree_serialize_leaves_tensorstore
from levanter.types import FilterSpec


logger = logging.getLogger(__name__)

PathLike = Union[str, pathlib.Path]

M = TypeVar("M", bound=PyTree)


@dataclass(frozen=True)
class CheckpointInterval:
    every: int  # how often to checkpoint
    until: Optional[int] = None  # until what step to save checkpoints with this policy, None means forever


class Checkpointer:
    """
    A checkpointer class that saves checkpoints with two different, but overlapping policies: time and step.

    Note that this class is stateful: it keeps track of the last time a checkpoint was saved, and the last step
    a checkpoint was saved at.
    """

    base_path: str
    save_interval: Optional[datetime.timedelta]  # we save at least this frequently
    step_policies: Sequence[CheckpointInterval] = dataclasses.field(
        default_factory=lambda: [CheckpointInterval(every=1000)]
    )

    _last_temporary_checkpoint: Optional[str] = None

    def __init__(
        self,
        base_path: PathLike,
        save_interval: Optional[datetime.timedelta],
        step_policies: Sequence[CheckpointInterval],
        *,
        keep_params: PyTree[FilterSpec] = True,
        dt_now_injection: Optional[Callable[[], datetime.datetime]] = None,
    ):
        """
        Class for managing checkpoints. Saves checkpoints according to two policies: time and step.

        Time policy: we save a checkpoint at least every `save_interval` seconds.
        Step policy: we save a checkpoint every `every` steps, until `until` steps have been reached.

        Time checkpoints are deleted after the next checkpoint is saved. Step checkpoints are never deleted.

        Args:
            base_path: the base path to save checkpoints to. may be gcs, local, or anything that tensorstore supports
            save_interval: the minimum amount of time between checkpoints (for time)
            step_policies: the step policies to use
            keep_params: a PyTree of FilterSpecs that specifies which parameters to keep in the checkpoint
            dt_now_injection: a function that returns the current time. useful for testing
        """
        self.base_path = str(base_path)
        self.save_interval = save_interval
        self.step_policies = list(step_policies)
        self.keep_params = keep_params
        self._dt_now_injection = dt_now_injection or datetime.datetime.now
        self._last_save_time = self._dt_now_injection()
        self._last_save_step = 0
        self._last_temporary_checkpoint = None

        # ensure that the step_policies are sorted. We could sort, but instead we'll just insist that they are sorted
        # since it's probably a typo if they aren't
        for i in range(1, len(step_policies)):
            # factor these out so mypy can figure it out
            prev_until = step_policies[i - 1].until
            until = step_policies[i].until
            if prev_until is None:
                raise ValueError("Only the last step policy can have an 'until' value of None")
            if until is None:
                continue
            if prev_until >= until:
                raise ValueError("Step policies must be sorted by 'until' value")

    def load_checkpoint(
        self,
        state: M,
        path: Optional[PathLike] = None,
        *,
        discover_latest: bool = True,
        axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
        mesh: Optional[haliax.partitioning.Mesh] = None,
    ) -> Optional[M]:
        if path is None:
            path = self.base_path
        return load_checkpoint(state, path, discover_latest=discover_latest, axis_mapping=axis_mapping, mesh=mesh)

    def load_model(
        self,
        model: M,
        path: Optional[str] = None,
        *,
        discover_latest: bool = True,
        axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
        mesh: Optional[haliax.partitioning.Mesh] = None,
    ) -> Optional[M]:
        """
        Convenience method/holdover from  previous API for loading checkpoints.
        Loads just the model assuming the model is in the `model` subdir of the discovered checkpoint.
        """
        ret_dict = self.load_checkpoint(
            {"model": model}, path, discover_latest=discover_latest, axis_mapping=axis_mapping, mesh=mesh
        )
        if ret_dict is None:
            return None
        return ret_dict["model"]

    def on_step(self, info, force: bool = False):
        step = info.step

        if step == 0:
            self._last_save_time = self._dt_now_injection()
            if not force:
                return  # don't save checkpoint at step 0 unless forced

        if step == self._last_save_step:
            # we've already saved a checkpoint at this step
            return

        # two reasons we can save: time or step
        # they have different behaviors for retention.
        # if the previous checkpoint was a temporary checkpoint (i.e. saved b/c of time), we can delete it

        # there's a potential clock skew issue here: if we save by time, and the clock is skewed across processes,
        # then we could end up with a situation where one process saves a checkpoint, and then another process
        # saves a checkpoint for the next step, etc. This leads to partial checkpoints, no good.
        # we fix by having process 0 make the decision
        my_should_save = force
        my_save_permanent_ckpt = force

        current_every = self._get_current_step_save_interval(step)
        last_save_time = self._dt_now_injection() - self._last_save_time
        if current_every is not None and step % current_every == 0:
            my_should_save = True
            my_save_permanent_ckpt = True
        elif self.save_interval and last_save_time >= self.save_interval:
            my_should_save = True
            my_save_permanent_ckpt = False

        should_save, save_permanent_ckpt = broadcast_one_to_all(
            jnp.array([my_should_save, my_save_permanent_ckpt], dtype=jnp.bool_)
        )

        # log the decision
        if should_save:
            if save_permanent_ckpt:
                logger.info(f"Saving checkpoint at step {step}.")
            else:
                logger.info(f"Saving temporary checkpoint at step {step}.")

        if should_save:
            last_checkpoint = self._last_temporary_checkpoint
            destination = f"step-{step}"

            self.save_checkpoint(info, destination)

            if not save_permanent_ckpt:
                self._last_temporary_checkpoint = destination
            else:
                self._last_temporary_checkpoint = None

            # TODO: we should consider writing to disk whether it's a temporary checkpoint or not
            # so that we can delete it properly if we recover
            if last_checkpoint is not None:
                self._rm_checkpoint(last_checkpoint)

    def _get_current_step_save_interval(self, step):
        # binary search for the correct interval
        # we assume that the intervals are sorted by until
        current_policy = next(filter(lambda p: p.until is None or p.until >= step, self.step_policies), None)
        if current_policy is None:
            return None
        return current_policy.every

    def _rm_checkpoint(self, checkpoint):
        if jax.process_index() != 0:
            return

        fs, plain_path = _get_fs_and_plain_path(self.base_path)
        # have to strip protocol from path because fsspec filesystems don't like them
        try:
            cp_path = os.path.join(plain_path, checkpoint)
            logger.info(f"Deleting checkpoint {checkpoint} from {cp_path}")
            fs.rm(cp_path, recursive=True)
        # don't let this take down a run
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to delete checkpoint", exc_info=True)

    def save_checkpoint(self, info, destination: str):
        path = os.path.join(self.base_path, destination)
        logger.info(f"Saving checkpoint at step {info.step} to {path}")
        state = saveable_state(info.state)
        save_checkpoint(
            state,
            step=info.step,
            checkpoint_path=path,
        )
        self._last_save_step = info.step
        self._last_save_time = self._dt_now_injection()
        logger.info(f"Saved checkpoint at step {info.step} to {path}. Save time is {self._last_save_time}")


def saveable_state(state):
    to_keep = jax.tree_util.tree_map(lambda _: True, state)
    to_keep = dataclasses.replace(to_keep, model=state.is_trainable)
    state = equinox.filter(state, to_keep)
    return state


def save_checkpoint(tree: M, step: int, checkpoint_path: PathLike):
    """
    Save a checkpoint to a given path using TensorStore. If exist_ok is True, the checkpoint
    will be saved even if a checkpoint already exists at the given path.

    If the path does not exist, it will be created.

    If training_state is None, no training state will be saved.

    This method is jax.Array-aware and will save shards in a way that can be restored
    """
    checkpoint_path = str(checkpoint_path)
    logger.info(f"Saving checkpoint to {checkpoint_path} for step {step}")

    fs: AbstractFileSystem
    fs, plain_path = _get_fs_and_plain_path(checkpoint_path)
    fs.makedirs(plain_path, exist_ok=True)

    tree_serialize_leaves_tensorstore(checkpoint_path, tree)
    save_metadata(checkpoint_path, fs, step)

    logger.info(f"Saved checkpoint for step {step}")

    # make sure that all processes agree on the checkpoint path and also synchronize hosts
    sync_global_devices(checkpoint_path)

    return checkpoint_path


def save_metadata(checkpoint_path, fs, step):
    metadata = {"step": step, "timestamp": datetime.datetime.now().isoformat()}
    if jax.process_index() == 0:
        with fs.open(os.path.join(checkpoint_path, "metadata.json"), "w") as json_out:
            json.dump(metadata, json_out)


def load_checkpoint(
    tree: M,
    checkpoint_path: PathLike,
    *,
    subpath: Optional[str] = None,
    discover_latest=True,
    axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
    mesh: Optional[jax.sharding.Mesh] = None,
) -> M:
    """
    Load a checkpoint from a given path. If discover_latest is True, then the latest checkpoint
    in a subdirectory of the given path will be loaded. If subpath is not None, then the checkpoint
    loads only that subpath of the checkpoint. This is useful for loading, e.g., just the model and not
    the entire training state.

    Args:
        tree: an exemplar of the tree to load. Can be a PyTree[ShapeDTypeStruct] instead of a PyTree[Any]
        checkpoint_path: the path to load the checkpoint from
        subpath: the subpath to load from the checkpoint
        discover_latest: whether to discover the latest checkpoint in the given path
        axis_mapping: the axis mapping to use for loading the checkpoint
        mesh: the mesh to use for loading the checkpoint
    Returns:
        the loaded checkpoint, with the same structure as the exemplar tree

    """
    fs: AbstractFileSystem
    fs, _ = _get_fs_and_plain_path(checkpoint_path)

    checkpoint_path = str(checkpoint_path)

    if discover_latest:
        checkpoint_path = discover_latest_checkpoint(checkpoint_path)  # type: ignore

    if checkpoint_path is None or not fs.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    metadata = load_metadata(checkpoint_path, fs)

    if subpath:
        checkpoint_path = os.path.join(checkpoint_path, subpath)

    try:
        tree = tree_deserialize_leaves_tensorstore(checkpoint_path, tree, axis_mapping=axis_mapping, mesh=mesh)
        return tree
    except:  # noqa
        from levanter.trainer import TrainerState

        if not isinstance(tree, TrainerState):
            raise
        else:
            logger.warning("Attempting to load old-style checkpoint")
            model, training_state = tree.model, (tree.opt_state, tree.training_key)

            model = tree_deserialize_leaves_tensorstore(
                os.path.join(checkpoint_path, "model"), model, axis_mapping=axis_mapping, mesh=mesh
            )

            if training_state is None:
                opt_state = None
                key = None
            else:
                training_state = tree_deserialize_leaves_tensorstore(
                    os.path.join(checkpoint_path, "training_state"),
                    training_state,
                    axis_mapping=axis_mapping,
                    mesh=mesh,
                )
                opt_state, key = training_state

            # TODO: pretty sure this is right, but should verify
            step = metadata["step"]
            new_state = dataclasses.replace(
                tree, _step=step + 1, model=model, opt_state=opt_state, training_key=key  # type: ignore
            )
            return new_state


def load_metadata(checkpoint_path, fs=None):
    if fs is None:
        fs, _, _ = fsspec.get_fs_token_paths(str(checkpoint_path))
    with fs.open(os.path.join(checkpoint_path, "metadata.json")) as metadata_in:
        metadata = json.load(metadata_in)
    return metadata


def discover_latest_checkpoint(checkpoint_path: PathLike) -> Optional[str]:
    """
    Discover the latest checkpoint in a given path.
    """
    checkpoint_path = str(checkpoint_path)
    # need to use fsspec for this, as glob.glob doesn't work on gs://
    fs: AbstractFileSystem
    fs, _ = _get_fs_and_plain_path(checkpoint_path)

    def is_checkpoint_dir(path: str):
        return fs.exists(os.path.join(path, "metadata.json"))

    def maybe_unstrip_protocol(path: str):
        base_path_protocol = urllib.parse.urlparse(str(checkpoint_path)).scheme
        if base_path_protocol != "" and not urllib.parse.urlparse(path).scheme != "":
            return f"{base_path_protocol}://{path}"
        return path

    ckpt_dirs = [maybe_unstrip_protocol(d) for d in fs.glob(os.path.join(checkpoint_path, "*")) if fs.isdir(d)]
    ckpt_dirs.append(checkpoint_path)
    ckpt_dirs = [d for d in ckpt_dirs if is_checkpoint_dir(d)]

    def checkpoint_sort_key(ckpt_dir):
        metadata = json.load(fs.open(os.path.join(ckpt_dir, "metadata.json")))
        return (datetime.datetime.fromisoformat(metadata["timestamp"]), metadata["step"])

    if len(ckpt_dirs) > 0:
        out = max(ckpt_dirs, key=checkpoint_sort_key)
        logger.info(f"Discovered latest checkpoint from {checkpoint_path} at {out}")
        return out
    else:
        logger.warning(f"No checkpoints found in {checkpoint_path}")
        return None


def _get_fs_and_plain_path(path, fs=None):
    if fs is None:
        fs, _, (path_to_open,) = fsspec.get_fs_token_paths(str(path))
    else:
        path_to_open = path
    return fs, path_to_open


@dataclass
class CheckpointerConfig:
    base_path: str = "checkpoints/"
    save_interval: timedelta = timedelta(minutes=15)
    # TODO: I'd like to write this, but it's not supported by draccus
    # keep: List[CheckpointInterval] = field(default_factory=lambda: [CheckpointInterval(every=1000)])
    keep: List[dict] = field(
        default_factory=lambda: [dict(every=10000)]
    )  # list of dicts with two keys: every and until

    def expanded_path(self, run_id):
        return os.path.expanduser(os.path.join(self.base_path, run_id))

    def create(self, run_id) -> Checkpointer:
        keeps = [CheckpointInterval(**k) for k in self.keep]
        return Checkpointer(
            base_path=self.expanded_path(run_id),
            save_interval=self.save_interval,
            step_policies=keeps,
        )

    def __post_init__(self):
        self.base_path = os.path.expanduser(self.base_path)

        # validate the checkpoint intervals.
        # we want to make sure that the intervals are monotonic. only the last one can be None
        prev_interval = None
        for interval in self.keep:
            if prev_interval is not None:
                assert prev_interval["until"] is not None, "Only the last checkpoint interval can be None"
                assert (
                    interval["until"] is None or interval["until"] > prev_interval["until"]
                ), "Checkpoint intervals must be monotonic"
            prev_interval = interval
