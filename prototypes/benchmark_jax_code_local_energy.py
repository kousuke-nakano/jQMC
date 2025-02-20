import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy_api
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data, Jastrow_two_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")  # insures we use the CPU

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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

jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5)
jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)

# define data
jastrow_data = Jastrow_data(
    jastrow_two_body_data=jastrow_twobody_data,
    jastrow_two_body_flag=True,
    jastrow_three_body_data=jastrow_threebody_data,
    jastrow_three_body_flag=True,
)

wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
hamiltonian_data = Hamiltonian_data(
    structure_data=structure_data, wavefunction_data=wavefunction_data, coulomb_potential_data=coulomb_potential_data
)

# tensorboard --logdir /tmp/tensorboard
jax.profiler.start_trace("/tmp/tensorboard", create_perfetto_link=True)
print("E_L comput. starts.")
start = time.perf_counter()
_ = compute_local_energy_api(hamiltonian_data=hamiltonian_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
end = time.perf_counter()
print("E_L comput. ends.")
print(f"Elapsed Time = {end - start:.2f} sec.")
jax.profiler.stop_trace()
