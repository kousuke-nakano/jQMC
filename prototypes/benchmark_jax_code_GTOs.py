import os
import time

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
from jax import jit, vmap

jax.config.update("jax_enable_x64", True)
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"


# Define GTO for single input
@jit
def gto_single(c, l, m, r, R, Z):
    diff = R - r
    r_squared = jnp.dot(diff, diff)
    radial = c * jnp.exp(-Z * r_squared)
    # Placeholder for spherical harmonics (Y_l,m)
    spherical_harmonics = 1.0  # Replace with actual computation
    return radial * spherical_harmonics


# vmap approach
@jit
def vmap_gto(c, l, m, r, R, Z):
    batched_gto = vmap(gto_single, in_axes=(0, 0, 0, None, 0, 0))
    return vmap(lambda r_single: batched_gto(c, l, m, r_single, R, Z))(r).T


# Batch computation approach
@jit
def batch_gto(c, l, m, r, R, Z):
    diff = R[:, None, :] - r[None, :, :]
    r_squared = jnp.sum(diff**2, axis=-1)
    radial = c[:, None] * jnp.exp(-Z[:, None] * r_squared)
    spherical_harmonics = 1.0  # Placeholder for actual spherical harmonics computation
    return radial * spherical_harmonics


trial = 5000
N = 103  # Number of GTO parameters (water)
M = 8  # Number of r vectors (water)
# N = 30000  # Number of GTO parameters (water)
# M = 30  # Number of r vectors (water)

c = np.linspace(1.0, 2.0, N)
l = np.arange(N)
m = np.arange(-N // 2, N // 2)
R = np.random.uniform(0, 1, (N, 3))
Z = np.linspace(1.0, 1.5, N)
r = np.random.uniform(0, 1, (M, 3))

# it turns out that np -> jnp transformation inside functions
# is a rate-determinant step! Pre transformation is needed.
# """
c = jnp.array(c)
l = jnp.array(l)
m = jnp.array(m)
R = jnp.array(R)
Z = jnp.array(Z)
r = jnp.array(r)
# """

# JIT compilation
vmap_gto_jit = jit(vmap_gto)
batch_gto_jit = jit(batch_gto)

# Batch benchmark
result_batch = batch_gto_jit(c, l, m, r, R, Z)
result_batch.block_until_ready()
start = time.perf_counter()
for _ in range(trial):
    result_batch = batch_gto_jit(c, l, m, r, R, Z)
    result_batch.block_until_ready()
end = time.perf_counter()
batch_time = (end - start) / trial

# vmap benchmark
result_vmap = vmap_gto_jit(c, l, m, r, R, Z)
result_vmap.block_until_ready()
start = time.perf_counter()
for _ in range(trial):
    result_vmap = vmap_gto_jit(c, l, m, r, R, Z)
    result_vmap.block_until_ready()
end = time.perf_counter()
vmap_time = (end - start) / trial

print(f"vmap computation time: {vmap_time * 10e3:.3f} msec.")
print(f"Batch computation time: {batch_time * 10e3:.3f} msec.")

print(result_batch.shape)
print(result_vmap.shape)

print(jnp.max(result_vmap - result_batch))
