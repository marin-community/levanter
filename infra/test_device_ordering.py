#!/usr/bin/env python3
"""Test device ordering across JAX versions."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

print("\n" + "="*60)
print("JAX VERSION:", jax.__version__)
print("="*60)

print("\n=== jax.devices() ===")
devices = jax.devices()
print(f"Number of devices: {len(devices)}")
print(f"Device IDs: {[d.id for d in devices]}")
print(f"Device platform: {devices[0].platform if devices else None}")

if hasattr(devices[0], "coords"):
    print("\nDevice coordinates (physical topology):")
    for i, d in enumerate(devices):
        print(f"  Device {d.id}: coords={d.coords}")

print("\n=== Default array sharding (no mesh) ===")
x = jnp.ones((8, 8))
print(f"Array sharding: {x.sharding}")
if hasattr(x.sharding, "device_set"):
    dev_ids = sorted([d.id for d in x.sharding.device_set])
    print(f"Array on device IDs: {dev_ids}")

print("\n=== Mesh created from jax.devices() ===")
mesh_devices = np.array(devices).reshape(4, 1)
mesh = Mesh(mesh_devices, ("data", "model"))
print(f"Mesh shape: {mesh.shape}")
print(f"Mesh device_ids: {mesh.device_ids}")
print(f"Mesh devices flattened: {[d.id for d in mesh.devices.flat]}")

print("\n=== Array created inside mesh context ===")
with mesh:
    y = jnp.ones((8, 8))
    print(f"Array sharding: {y.sharding}")
    if hasattr(y.sharding, "device_set"):
        dev_ids = sorted([d.id for d in y.sharding.device_set])
        print(f"Array on device IDs: {dev_ids}")

print("\n=== Haliax array without mesh ===")
try:
    import haliax as hax
    from jax import random as jrandom
    Pos = hax.Axis("Pos", 16)
    Embed = hax.Axis("Embed", 32)
    z = hax.random.normal(jrandom.PRNGKey(0), (Pos, Embed))
    print(f"Haliax array sharding: {z.array.sharding}")
    if hasattr(z.array.sharding, "device_set"):
        dev_ids = sorted([d.id for d in z.array.sharding.device_set])
        print(f"Haliax array on device IDs: {dev_ids}")
except Exception as e:
    print(f"Haliax test failed: {e}")

print("\n" + "="*60)
