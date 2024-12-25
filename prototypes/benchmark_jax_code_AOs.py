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

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"

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
