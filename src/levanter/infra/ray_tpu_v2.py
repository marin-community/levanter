"""Tree-of-actors implementation for multi-slice TPU scheduling.

"""

import asyncio

from typing import Any, Dict, List, Optional
import ray
from ray._private.accelerators import TPUAcceleratorManager
from ray.actor import ActorHandle
from ray.remote_function import RemoteFunction
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.dag import FunctionNode


class TPUActorNotReadyException(Exception):
    pass

# TODO: Set max_restarts
@ray.remote
class TPUWorkerActor:
    def __init__(self):
        # TODO: Need locking
        self._waitable: Optional[ray.ObjectRef] = None

    def _get_remote_fn_resources(self) -> Dict[str, int]:
        # DO_NOT_MERGE: set this to real resources
        return {}
        num_tpus_per_host = TPUAcceleratorManager.get_current_node_num_accelerators()
        return {
            "CPU": 8,
            "TPU": num_tpus_per_host,
            "memory": 20e9,
        }

    async def setup(self) -> None:
        # TODO: call _hacky_remove_tpu_lockfile() here
        pass

    async def run(self, remote_fn: RemoteFunction) -> Any:
        if self._waitable:
            ray.cancel(self._waitable, force=True, recursive=True)
        node_id = ray.get_runtime_context().get_node_id()
        self._waitable = remote_fn.options(
            resources=self._get_remote_fn_resources(),
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
            # TODO: Can we just set max_calls here?
        ).remote()
        return await self._waitable

    async def teardown(self) -> None:
        if self._waitable:
            ray.cancel(self._waitable, force=True, recursive=True)


# TODO: Set max_restarts
@ray.remote
class TPUSliceActor:
    def __init__(self):
        # TODO: Need locking
        self._worker_actors: List[ActorHandle] = []

    def _get_worker_actor_resources(self) -> Dict[str, int]:
        # DO_NOT_MERGE: set this to real resources
        return {}
        pod_name = ray.util.accelerators.tpu.get_current_pod_name()
        return {pod_name: 1}
    
    async def setup(self) -> None:
        # DO_NOT_MERGE: set this to real number of hosts
        # num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
        # TODO: Add a timeout for scheduling a worker
        # TODO: Maybe kill workers that fail? Or just throw away the whole slice?
        # NOTE: We can use kill(no_restart=False) to tell Ray to respawn the a child actor
        # NOTE: We can also kill the actor itself to make setup respawn this actor
        num_hosts = 4
        while len(self._worker_actors) < num_hosts:
            self._worker_actors.append(TPUWorkerActor.options(resources=self._get_worker_actor_resources()).remote())
        return await asyncio.gather(*[worker_actor.setup.remote() for worker_actor in self._worker_actors])

    async def run(self, remote_fn: RemoteFunction) -> List[Any]:
        if not self._worker_actors:
            raise TPUActorNotReadyException("setup() must be called before run()")
        return await asyncio.gather(*[worker_actor.run.remote(remote_fn) for worker_actor in self._worker_actors])

    async def teardown(self) -> None:
        await asyncio.gather(*[worker_actor.teardown.remote() for worker_actor in self._worker_actors])
        for worker_actor in self._worker_actors:
            ray.kill(worker_actor)
        

@ray.remote(num_cpus=0.01)
class TPUMultiSliceActor:
    def __init__(self, tpu_type: str, num_slices: int = 1):
        self._tpu_type = tpu_type
        self._num_slices = num_slices
        # TODO: Need locking
        self._slice_actors: List[ActorHandle] = []

    def _get_slice_actor_resources(self) -> Dict[str, int]:
        # DO_NOT_MERGE: set this to real resources
        return {}
        # return {f"TPU-{self._tpu_type}-head": 1}
    
    async def setup(self) -> None:
        # TODO: Add a timeout for scheduling a slice
        # TODO: Kill slices that fail
        # NOTE: We can use no_restart=False to tell Ray to respawn the slice
        # TODO: Handle mxla_env related things
        while len(self._slice_actors) < self._num_slices:
            self._slice_actors.append(
                TPUSliceActor.options(resources=self._get_slice_actor_resources()).remote()
            )
        return await asyncio.gather(*[slice_actor.setup.remote() for slice_actor in self._slice_actors])
    
    async def run(self, remote_fn: RemoteFunction) -> List[ray.ObjectRef]:
        if not self._slice_actors:
            raise TPUActorNotReadyException("setup() must be called before run()")
        return await asyncio.gather(*[slice_actor.run.remote(remote_fn) for slice_actor in self._slice_actors])
    
    async def teardown(self) -> None:
        await asyncio.gather(*[slice_actor.teardown.remote() for slice_actor in self._slice_actors])
        for slice_actor in self._slice_actors:
            ray.kill(slice_actor)


@ray.remote(max_calls=1)
def remote_fn():
    task_id = ray.get_runtime_context().get_task_id()
    node_id = ray.get_runtime_context().get_node_id()
    print(f"{task_id} hello from {node_id}")
    import time
    time.sleep(15)
    print(f"{task_id} goodbye from {node_id}")
    return task_id

@ray.remote(num_cpus=0.01)
def run_on_pod_ray(
    remote_fn: RemoteFunction,
    tpu_type: str,
    num_slices: int = 1,
    max_retries_preemption: int = 10000,
    max_retries_failure: int = 10,
):
    
    if isinstance(remote_fn, FunctionNode):
        raise ValueError(
            "Remote function must be a Ray remote function or a plain function, not a FunctionNode. Don't use bind."
        )
    elif not isinstance(remote_fn, RemoteFunction):
        remote_fn = ray.remote(max_calls=1)(remote_fn)
    elif remote_fn._default_options.get("max_calls") is None:
        raise ValueError("Remote function must have max_calls set to 1 for TPU workloads.")
    # TODO: Do we actually want max_calls=1?

    actor = TPUMultiSliceActor.remote(tpu_type, num_slices)
    success = False
    num_retries_preemption = 0
    num_retries_failure = 0
    while not success and num_retries_preemption <= max_retries_preemption and num_retries_failure < max_retries_failure:
        try:
            ray.get(actor.setup.remote())
        except Exception as e:
            num_retries_preemption += 1
            continue

        try:
            result = ray.get(actor.run.remote(remote_fn))
        except Exception as e:
            # TODO: Distinguish between preemption and failure
            num_retries_preemption += 1
            continue
        print(result)
        success = True
        ray.get(actor.teardown.remote())
        return result


def main():
    ray.init()
    ray.get(run_on_pod_ray.remote(remote_fn, "fake-tpu-type", num_slices=2))

# def bad_main():
#     # Early terminating a task before it finishes
#     ray.init()
#     actor = TPUSliceActor.remote()
#     ray.get(actor.setup.remote())
#     wait = actor.run.remote(remote_fn)
#     import time
#     time.sleep(5)
#     # terminate before run completes
#     ray.get(actor.teardown.remote())


if __name__ == "__main__":
    main()
