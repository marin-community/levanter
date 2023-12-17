from typing import Optional

import jax
import numpy as np
from jax.sharding import Mesh


def local_device_grid_positions(mesh, process_index: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns a tuple of nd arrays, one for each axis, indicating the position of each device on the grid.
    Analogous to what np.where would return."""
    pi = process_index or jax.process_index()
    # our device indices are [process_index * num_devices_per_node, (process_index + 1) * num_devices_per_node)
    # we could be clever here and do math to figure out where we are in the grid, but it's simpler and less
    # fragile to just search the grid for our devices
    my_device_pos = np.vectorize(lambda dev: dev.process_index == pi)(mesh.devices)
    return my_device_pos.nonzero()


def process_mesh_position(mesh, process_index: Optional[int] = None) -> tuple[int, ...]:
    """
    If we envision each process as a subgrid of the mesh for its devices, this is the position of the process
    in the coarsened process-level mesh
    """
    upper_left_position = np.array([np.min(axis) for axis in local_device_grid_positions(mesh, process_index)])
    local_mesh_size = mesh.local_mesh.devices.shape
    pos = upper_left_position // local_mesh_size
    return pos


def process_mesh_size(mesh: Mesh) -> tuple[int, ...]:
    """
    If we envision each process as a subgrid of the mesh for its devices, then there is a process grid that
    is a coarsened version of the mesh. This is the size of the process grid.
    """
    local_mesh_size = mesh.local_mesh.devices.shape
    return tuple(mesh.devices.shape[i] // local_mesh_size[i] for i in range(len(local_mesh_size)))
