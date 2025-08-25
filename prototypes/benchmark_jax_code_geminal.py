import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from jqmc.determinant import _compute_ratio_determinant_part_jax
from jqmc.jastrow_factor import _compute_ratio_Jastrow_part_jax
from jqmc.molecular_orbital import _compute_MOs_jax

jax.config.update("jax_enable_x64", True)

# jax.config.update("jax_platform_name", "cpu")  # insures we use the CPU
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUM_INTER_THREADS"] = "1"
# os.environ["NUM_INTRA_THREADS"] = "1"
# os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"

# ratio
hamiltonian_chk = "hamiltonian_data_water.chk"
with open(hamiltonian_chk, "rb") as f:
    hamiltonian_data = pickle.load(f)
geminal_data = hamiltonian_data.wavefunction_data.geminal_data
jastrow_data = hamiltonian_data.wavefunction_data.jastrow_data

# print
print(geminal_data)

# test MOs
num_electron_up = 4
num_electron_dn = 4

# Initialization
r_carts_up = []
r_carts_dn = []

total_electrons = 0

if hamiltonian_data.coulomb_potential_data.ecp_flag:
    charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
        hamiltonian_data.coulomb_potential_data.z_cores
    )
else:
    charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

coords = hamiltonian_data.structure_data.positions_cart

# Place electrons around each nucleus
for i in range(len(coords)):
    charge = charges[i]
    num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

    # Retrieve the position coordinates
    x, y, z = coords[i]

    # Place electrons
    for _ in range(num_electrons):
        # Calculate distance range
        distance = np.random.uniform(0.1, 2.0)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        # Convert spherical to Cartesian coordinates
        dx = distance * np.sin(theta) * np.cos(phi)
        dy = distance * np.sin(theta) * np.sin(phi)
        dz = distance * np.cos(theta)

        # Position of the electron
        electron_position = np.array([x + dx, y + dy, z + dz])

        # Assign spin
        if len(r_carts_up) < num_electron_up:
            r_carts_up.append(electron_position)
        else:
            r_carts_dn.append(electron_position)

    total_electrons += num_electrons

# Handle surplus electrons
remaining_up = num_electron_up - len(r_carts_up)
remaining_dn = num_electron_dn - len(r_carts_dn)

# Randomly place any remaining electrons
for _ in range(remaining_up):
    r_carts_up.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))
for _ in range(remaining_dn):
    r_carts_dn.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))

r_up_carts = jnp.array(r_carts_up)
r_dn_carts = jnp.array(r_carts_dn)


@partial(jit, static_argnums=1)
def split_lambda_matrix(lambda_matrix, orb_num_dn):
    return jnp.hsplit(lambda_matrix, [orb_num_dn])


lambda_matrix_paired, lambda_matrix_unpaired = split_lambda_matrix(geminal_data.lambda_matrix, geminal_data.orb_num_dn)
lambda_matrix_paired.block_until_ready()
lambda_matrix_unpaired.block_until_ready()
start = time.perf_counter()
lambda_matrix_paired, lambda_matrix_unpaired = split_lambda_matrix(geminal_data.lambda_matrix, geminal_data.orb_num_dn)
lambda_matrix_paired.block_until_ready()
lambda_matrix_unpaired.block_until_ready()
end = time.perf_counter()
print(f"Split elapsed Time = {(end - start) * 1e3:.3f} msec.")

d_up = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_carts)
d_up.block_until_ready()
d_dn = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_carts)
d_dn.block_until_ready()

start = time.perf_counter()
orb_matrix_up = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_carts)
orb_matrix_dn = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_carts)
orb_matrix_up.block_until_ready()
orb_matrix_dn.block_until_ready()
end = time.perf_counter()
print(f"Comput. MOs elapsed Time = {(end - start) * 1e3:.3f} msec.")

"""
d_up = _compute_MOs_jax(mos_data=geminal_data.orb_data_up_spin, r_carts=r_up_carts)
d_up.block_until_ready()
d_dn = _compute_MOs_jax(mos_data=geminal_data.orb_data_dn_spin, r_carts=r_dn_carts)
d_dn.block_until_ready()

start = time.perf_counter()
orb_matrix_up_d = _compute_MOs_jax(mos_data=geminal_data.orb_data_up_spin, r_carts=r_up_carts)
orb_matrix_dn_d = _compute_MOs_jax(mos_data=geminal_data.orb_data_dn_spin, r_carts=r_dn_carts)
orb_matrix_up_d.block_until_ready()
orb_matrix_dn_d.block_until_ready()
end = time.perf_counter()
print(f"Comput. MOs elapsed Time = {(end-start)*1e3:.3f} msec.")
"""


@jit
def construct_geminal(orb_matrix_up, orb_matrix_dn, lambda_matrix_paired, lambda_matrix_unpaired):
    geminal_paired = jnp.dot(orb_matrix_up.T, jnp.dot(lambda_matrix_paired, orb_matrix_dn))
    geminal_unpaired = jnp.dot(orb_matrix_up.T, lambda_matrix_unpaired)
    geminal = jnp.hstack([geminal_paired, geminal_unpaired])
    return geminal


_ = construct_geminal(orb_matrix_up, orb_matrix_dn, lambda_matrix_paired, lambda_matrix_unpaired)
start = time.perf_counter()
geminal = construct_geminal(orb_matrix_up, orb_matrix_dn, lambda_matrix_paired, lambda_matrix_unpaired)
geminal.block_until_ready()
end = time.perf_counter()
print(f"Construct elapsed Time = {(end - start) * 1e3:.3f} msec.")
