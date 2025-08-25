import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")  # insures we use the CPU

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.environ["NUM_INTER_THREADS"] = "1"
os.environ["NUM_INTRA_THREADS"] = "1"

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# taskset -c 0,1,2,3 mpirun -np 2 python benchmark.py works as expected on a linux machine (i.e. 2 MPI processes and 2 threads per MPI process) to limit the maximum thread numbers per task.
# mpirun -np 4 python benchmark.py simply works :-)


N = int(1e8)
num_gfmc_warmup_steps = 3
num_gfmc_bin_collect = 10
num_gfmc_bin_blocks = 5
e_L_averaged = np.random.rand(N)
w_L_averaged = np.random.rand(N)
e_L_eq = e_L_averaged[num_gfmc_warmup_steps + num_gfmc_bin_collect :]
w_L_eq = w_L_averaged[num_gfmc_warmup_steps:]

# """ list and numpy
print("  Progress: Computing G_eq and G_e_L_eq.")
start = time.perf_counter()
G_eq = [np.prod([w_L_eq[n - j] for j in range(1, num_gfmc_bin_collect + 1)]) for n in range(num_gfmc_bin_collect, len(w_L_eq))]
G_e_L_eq = e_L_eq * G_eq
end = time.perf_counter()
print(f"  Elapsed Time = {(end - start):.5f} sec.")

print(f"  Progress: Computing binned G_e_L_eq and G_eq with # binned blocks = {num_gfmc_bin_blocks}.")
start = time.perf_counter()
G_e_L_split = np.array_split(G_e_L_eq, num_gfmc_bin_blocks)
G_e_L_binned = np.array([np.average(G_e_L_list) for G_e_L_list in G_e_L_split])
G_split = np.array_split(G_eq, num_gfmc_bin_blocks)
G_binned = np.array([np.average(G_list) for G_list in G_split])
end = time.perf_counter()
print(f"  Elapsed Time = {(end - start):.5f} sec.")

print(f"  Progress: Computing jackknife samples with # binned blocks = {num_gfmc_bin_blocks}.")
start = time.perf_counter()
G_e_L_binned_sum = np.sum(G_e_L_binned)
G_binned_sum = np.sum(G_binned)
e_L_jackknife = [(G_e_L_binned_sum - G_e_L_binned[m]) / (G_binned_sum - G_binned[m]) for m in range(num_gfmc_bin_blocks)]
end = time.perf_counter()
print(f"  Elapsed Time = {(end - start):.5f} sec.")

print("  Progress: Computing jackknife mean and std.")
start = time.perf_counter()
e_L_mean = np.average(e_L_jackknife)
e_L_std = np.sqrt(num_gfmc_bin_blocks - 1) * np.std(e_L_jackknife)
end = time.perf_counter()
print(f"  Elapsed Time = {(end - start):.5f} sec.")
# """

#''' jax and numpy
print("  Progress: Computing G_eq and G_e_L_eq.")
start = time.perf_counter()


@partial(jit, static_argnums=2)
def _compute_G_eq_and_G_e_L_eq_jax(w_L_eq, e_L_eq, num_gfmc_bin_collect):
    def get_slice(n):
        return jax.lax.dynamic_slice(w_L_eq, (n - num_gfmc_bin_collect,), (num_gfmc_bin_collect,))

    indices = jnp.arange(num_gfmc_bin_collect, len(w_L_eq))
    G_eq_matrix = vmap(get_slice)(indices)
    G_eq = jnp.prod(G_eq_matrix, axis=1)
    G_e_L_eq = e_L_eq * G_eq
    return G_eq, G_e_L_eq


w_L_eq = jnp.array(w_L_eq)
e_L_eq = jnp.array(e_L_eq)
G_eq, G_e_L_eq = _compute_G_eq_and_G_e_L_eq_jax(w_L_eq, e_L_eq, num_gfmc_bin_collect)
G_eq.block_until_ready()
G_e_L_eq.block_until_ready()
G_eq = np.array(G_eq)
G_e_L_eq = np.array(G_e_L_eq)
end = time.perf_counter()
print(f"  Elapsed Time = {(end - start):.5f} sec.")

print(f"  Progress: Computing binned G_e_L_eq and G_eq with # binned blocks = {num_gfmc_bin_blocks}.")
start = time.perf_counter()
G_e_L_split = np.array_split(G_e_L_eq, num_gfmc_bin_blocks)
G_e_L_binned = np.array([np.average(G_e_L_list) for G_e_L_list in G_e_L_split])
G_split = np.array_split(G_eq, num_gfmc_bin_blocks)
G_binned = np.array([np.average(G_list) for G_list in G_split])
end = time.perf_counter()
print(f"  Elapsed Time = {(end - start):.5f} sec.")

print(f"  Progress: Computing jackknife samples with # binned blocks = {num_gfmc_bin_blocks}.")
start = time.perf_counter()
G_e_L_binned_sum = np.sum(G_e_L_binned)
G_binned_sum = np.sum(G_binned)
e_L_jackknife = [(G_e_L_binned_sum - G_e_L_binned[m]) / (G_binned_sum - G_binned[m]) for m in range(num_gfmc_bin_blocks)]
end = time.perf_counter()
print(f"  Elapsed Time = {(end - start):.5f} sec.")

print("  Progress: Computing jackknife mean and std.")
start = time.perf_counter()
e_L_mean = np.average(e_L_jackknife)
e_L_std = np.sqrt(num_gfmc_bin_blocks - 1) * np.std(e_L_jackknife)
end = time.perf_counter()
print(f"  Elapsed Time = {(end - start):.5f} sec.")
#'''
#'''
