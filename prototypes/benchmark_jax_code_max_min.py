import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from jqmc.atomic_orbital import compute_AOs_api, compute_AOs_grad_api, compute_AOs_laplacian_api
from jqmc.trexio_wrapper import read_trexio_file

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")  # insures we use the CPU

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.environ["NUM_INTER_THREADS"] = "1"
os.environ["NUM_INTRA_THREADS"] = "1"

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# taskset -c 0,1,2,3 mpirun -np 2 python benchmark.py works as expected on a linux machine (i.e. 2 MPI processes and 2 threads per MPI process) to limit the maximum thread numbers per task.
# mpirun -np 4 python benchmark.py simply works :-)

A = 1e-3
B = 1e-5
trial = 500000

start = time.perf_counter()
for _ in range(trial):
    _ = np.exp(A) * np.exp(B)
end = time.perf_counter()
print(f"np.exp * np.exp elapsed Time = {(end - start) / trial * 1e3:.5f} msec.")

time.sleep(3)

start = time.perf_counter()
for _ in range(trial):
    _ = np.maximum(A, B)
end = time.perf_counter()
print(f"np.maximum elapsed Time = {(end - start) / trial * 1e3:.5f} msec.")

time.sleep(3)

start = time.perf_counter()
for _ in range(trial):
    _ = np.max((A, B))
end = time.perf_counter()
print(f"np.max elapsed Time = {(end - start) / trial * 1e3:.5f} msec.")

time.sleep(3)

start = time.perf_counter()
for _ in range(trial):
    _ = max(A, B)
end = time.perf_counter()
print(f"max elapsed Time = {(end - start) / trial * 1e3:.5f} msec.")

time.sleep(3)

start = time.perf_counter()
for _ in range(trial):
    _ = min(A, B)
end = time.perf_counter()
print(f"min elapsed Time = {(end - start) / trial * 1e3:.5f} msec.")

time.sleep(3)
