import os
import time

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
from jax import jit

from jqmc.atomic_orbital import (
    _compute_AOs_jax,
    _compute_normalization_fator_jax,
    _compute_primitive_AOs_jax,
    _compute_R_n_jax,
    _compute_S_l_m_jax,
    compute_AOs_api,
    compute_AOs_grad_api,
    compute_AOs_laplacian_api,
)
from jqmc.trexio_wrapper import read_trexio_file

os.environ["NUM_INTER_THREADS"] = "1"
os.environ["NUM_INTRA_THREADS"] = "1"

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# taskset -c 0,1,2,3 mpirun -np 2 python benchmark.py works as expected on a linux machine (i.e. 2 MPI processes and 2 threads per MPI process) to limit the maximum thread numbers per task.
# mpirun -np 4 python benchmark.py simply works :-)

(
    structure_data,
    aos_data,
    mos_data_up,
    mos_data_dn,
    geminal_mo_data,
    coulomb_potential_data,
) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "water_ccpvtz_trexio.hdf5"))

num_ele_up = geminal_mo_data.num_electron_up
num_ele_dn = geminal_mo_data.num_electron_dn
r_cart_min, r_cart_max = -3.0, +3.0
r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

"""
_ = compute_AOs_api(aos_data=aos_data, r_carts=r_up_carts)
_ = compute_AOs_grad_api(aos_data=aos_data, r_carts=r_up_carts)
_ = compute_AOs_laplacian_api(aos_data=aos_data, r_carts=r_up_carts)

# tensorboard --logdir /tmp/tensorboard
jax.profiler.start_trace("/tmp/tensorboard", create_perfetto_link=True)
print("AO comput. starts.")
start = time.perf_counter()
aos_up = compute_AOs_api(aos_data=aos_data, r_carts=r_up_carts)
# aos_up_grad = compute_AOs_grad_api(aos_data=aos_data, r_carts=r_up_carts)
# aos_up_laplacian = compute_AOs_laplacian_api(aos_data=aos_data, r_carts=r_up_carts)

aos_up.block_until_ready()
# aos_up_grad[0].block_until_ready()
# aos_up_grad[1].block_until_ready()
# aos_up_grad[2].block_until_ready()
# aos_up_laplacian.block_until_ready()

end = time.perf_counter()
print("AO comput. ends.")
print(f"Elapsed Time = {end-start:.2f} sec.")
jax.profiler.stop_trace()
"""

"""
num_ele = 4
r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele, 3) + r_cart_min

_ = _compute_AOs_jax(aos_data=aos_data, r_carts=r_carts)
start = time.perf_counter()
AOs = _compute_AOs_jax(aos_data=aos_data, r_carts=r_carts)
AOs.block_until_ready()
end = time.perf_counter()
print(f"Comput. elapsed Time = {(end-start)*1e3:.3f} msec.")
"""

trial = 10000
R_cart = np.array([0, 0, 0])
r_cart = np.array([1.0, 1.0, 1.0])
coefficient = 1.0
exponent = 15.0
l = 5
m = 3

ao = _compute_primitive_AOs_jax(coefficient=coefficient, exponent=exponent, l=l, m=m, R_cart=R_cart, r_cart=r_cart)
ao.block_until_ready()
start = time.perf_counter()
for _ in range(trial):
    ao = _compute_primitive_AOs_jax(coefficient=coefficient, exponent=exponent, l=l, m=m, R_cart=R_cart, r_cart=r_cart)
    ao.block_until_ready()
end = time.perf_counter()
print(f"Comput. elapsed Time = {(end - start) / trial * 1e3:.3f} msec.")
time.sleep(3)

N_n_dup = _compute_normalization_fator_jax(l, exponent)
R_n_dup = _compute_R_n_jax(coefficient, exponent, R_cart, r_cart)
S_l_m_dup = _compute_S_l_m_jax(l, m, R_cart, r_cart)

start = time.perf_counter()
for _ in range(trial):
    N_n_dup = _compute_normalization_fator_jax(l, exponent)
    N_n_dup.block_until_ready()
end = time.perf_counter()
print(f"N_n_dup Comput. elapsed Time = {(end - start) / trial * 1e3:.3f} msec.")
time.sleep(3)

start = time.perf_counter()
for _ in range(trial):
    R_n_dup = _compute_R_n_jax(coefficient, exponent, R_cart, r_cart)
    R_n_dup.block_until_ready()
end = time.perf_counter()
print(f"R_n_dup Comput. elapsed Time = {(end - start) / trial * 1e3:.3f} msec.")
time.sleep(3)

start = time.perf_counter()
for _ in range(trial):
    S_l_m_dup = _compute_S_l_m_jax(l, m, R_cart, r_cart)
    S_l_m_dup.block_until_ready()
end = time.perf_counter()
print(f"S_l_m_dup Comput. elapsed Time = {(end - start) / trial * 1e3:.3f} msec.")
time.sleep(3)
