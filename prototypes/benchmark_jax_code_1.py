import time, os, jax, numpy as np, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")  # insures we use the CPU

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.environ["NUM_INTER_THREADS"] = "1"
os.environ["NUM_INTRA_THREADS"] = "1"

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
)

# taskset -c 0,1,2,3 mpirun -np 2 python benchmark.py works as expected on a linux machine (i.e. 2 MPI processes and 2 threads per MPI process) to limit the maximum thread numbers per task.
# mpirun -np 4 python benchmark.py simply works :-)


def timer(name, f, x, shouldBlock=True):
    # warmup
    y = f(x).block_until_ready() if shouldBlock else f(x)
    # running the code
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    y = f(x).block_until_ready() if shouldBlock else f(x)
    end_wall = time.perf_counter()
    end_cpu = time.process_time()
    # computing the metric and displaying it
    wall_time = end_wall - start_wall
    cpu_time = end_cpu - start_cpu
    cpu_count = os.cpu_count()
    print(
        f"{name}: cpu usage {cpu_time/wall_time:.1f}/{cpu_count} wall_time:{wall_time:.1f}s"
    )


key = jax.random.PRNGKey(0)
x = jax.random.normal(key, shape=(500000000,), dtype=jnp.float64)
x_mat = jax.random.normal(key, shape=(10000, 10000), dtype=jnp.float64)
f_numpy = np.cos
f_vmap = jax.jit(jax.vmap(jnp.cos))
# f_pmap = jax.jit(jax.pmap(jnp.cos)) # error
# f_dot = jax.jit(lambda x: jnp.dot(x, x.T))

timer("numpy", f_numpy, x, shouldBlock=False)
timer("vmap", f_vmap, x)
# timer("pmap", f_pmap, x)
# timer("dot", f_dot, x_mat)
