import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from jqmc.determinant import _compute_det_geminal_all_elements_jax, compute_det_geminal_all_elements_api
from jqmc.jastrow_factor import compute_Jastrow_part_api

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

alat = 0.10


@jit
def generate_mesh(alat, r_up_carts, r_dn_carts):
    RT = np.eye(3)
    # Define the shifts to apply (+/- alat in each coordinate direction)
    shifts = alat * jnp.array(
        [
            [1, 0, 0],  # x+
            [-1, 0, 0],  # x-
            [0, 1, 0],  # y+
            [0, -1, 0],  # y-
            [0, 0, 1],  # z+
            [0, 0, -1],  # z-
        ]
    )  # Shape: (6, 3)

    shifts = shifts @ RT  # Shape: (6, 3)

    # num shift
    num_shifts = shifts.shape[0]

    # Process up-spin electrons
    num_up_electrons = r_up_carts.shape[0]
    num_up_configs = num_up_electrons * num_shifts

    # Create base positions repeated for each configuration
    base_positions_up = jnp.repeat(r_up_carts[None, :, :], num_up_configs, axis=0)  # Shape: (num_up_configs, N_up, 3)

    # Initialize shifts_to_apply_up
    shifts_to_apply_up = jnp.zeros_like(base_positions_up)

    # Create indices for configurations
    config_indices_up = jnp.arange(num_up_configs)
    electron_indices_up = jnp.repeat(jnp.arange(num_up_electrons), num_shifts)
    shift_indices_up = jnp.tile(jnp.arange(num_shifts), num_up_electrons)

    # Apply shifts to the appropriate electron in each configuration
    shifts_to_apply_up = shifts_to_apply_up.at[config_indices_up, electron_indices_up, :].set(shifts[shift_indices_up])

    # Apply shifts to base positions
    r_up_carts_shifted = base_positions_up + shifts_to_apply_up  # Shape: (num_up_configs, N_up, 3)

    # Repeat down-spin electrons for up-spin configurations
    r_dn_carts_repeated_up = jnp.repeat(r_dn_carts[None, :, :], num_up_configs, axis=0)  # Shape: (num_up_configs, N_dn, 3)

    # Process down-spin electrons
    num_dn_electrons = r_dn_carts.shape[0]
    num_dn_configs = num_dn_electrons * num_shifts

    base_positions_dn = jnp.repeat(r_dn_carts[None, :, :], num_dn_configs, axis=0)  # Shape: (num_dn_configs, N_dn, 3)
    shifts_to_apply_dn = jnp.zeros_like(base_positions_dn)

    config_indices_dn = jnp.arange(num_dn_configs)
    electron_indices_dn = jnp.repeat(jnp.arange(num_dn_electrons), num_shifts)
    shift_indices_dn = jnp.tile(jnp.arange(num_shifts), num_dn_electrons)

    # Apply shifts to the appropriate electron in each configuration
    shifts_to_apply_dn = shifts_to_apply_dn.at[config_indices_dn, electron_indices_dn, :].set(shifts[shift_indices_dn])

    r_dn_carts_shifted = base_positions_dn + shifts_to_apply_dn  # Shape: (num_dn_configs, N_dn, 3)

    # Repeat up-spin electrons for down-spin configurations
    r_up_carts_repeated_dn = jnp.repeat(r_up_carts[None, :, :], num_dn_configs, axis=0)  # Shape: (num_dn_configs, N_up, 3)

    # Combine configurations
    r_up_carts_combined = jnp.concatenate([r_up_carts_shifted, r_up_carts_repeated_dn], axis=0)  # Shape: (N_configs, N_up, 3)
    r_dn_carts_combined = jnp.concatenate([r_dn_carts_repeated_up, r_dn_carts_shifted], axis=0)  # Shape: (N_configs, N_dn, 3)

    return r_up_carts_combined, r_dn_carts_combined


_, _ = generate_mesh(alat=alat, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

start = time.perf_counter()
r_up_carts_combined, r_dn_carts_combined = generate_mesh(alat=alat, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
r_up_carts_combined.block_until_ready()
r_dn_carts_combined.block_until_ready()
end = time.perf_counter()
print(f"Init elapsed Time = {(end - start) * 1e3:.3f} msec.")

""" fast update
_ = _compute_ratio_determinant_part_jax(
    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
    old_r_up_carts=r_up_carts,
    old_r_dn_carts=r_dn_carts,
    new_r_up_carts_arr=r_up_carts_combined,
    new_r_dn_carts_arr=r_dn_carts_combined,
) * _compute_ratio_Jastrow_part_jax(
    jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
    old_r_up_carts=r_up_carts,
    old_r_dn_carts=r_dn_carts,
    new_r_up_carts_arr=r_up_carts_combined,
    new_r_dn_carts_arr=r_dn_carts_combined,
)

start = time.perf_counter()
# Evaluate the ratios of wavefunctions between the shifted positions and the original position
wf_ratio = _compute_ratio_determinant_part_jax(
    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
    old_r_up_carts=r_up_carts,
    old_r_dn_carts=r_dn_carts,
    new_r_up_carts_arr=r_up_carts_combined,
    new_r_dn_carts_arr=r_dn_carts_combined,
) * _compute_ratio_Jastrow_part_jax(
    jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
    old_r_up_carts=r_up_carts,
    old_r_dn_carts=r_dn_carts,
    new_r_up_carts_arr=r_up_carts_combined,
    new_r_dn_carts_arr=r_dn_carts_combined,
)
wf_ratio.block_until_ready()
end = time.perf_counter()
print(f"Comput. elapsed Time = {(end-start)*1e3:.3f} msec.")
"""

psi_j = compute_Jastrow_part_api(hamiltonian_data.wavefunction_data.jastrow_data, r_up_carts, r_dn_carts)
# Evaluate the wavefunction at the shifted positions using vectorization
psi_jp = vmap(compute_Jastrow_part_api, in_axes=(None, 0, 0))(
    hamiltonian_data.wavefunction_data.jastrow_data, r_up_carts_combined, r_dn_carts_combined
)
psi_j.block_until_ready()
psi_jp.block_until_ready()

psi_d = _compute_det_geminal_all_elements_jax(hamiltonian_data.wavefunction_data.geminal_data, r_up_carts, r_dn_carts)
# Evaluate the wavefunction at the shifted positions using vectorization
psi_dp = vmap(_compute_det_geminal_all_elements_jax, in_axes=(None, 0, 0))(
    hamiltonian_data.wavefunction_data.geminal_data, r_up_carts_combined, r_dn_carts_combined
)
psi_d.block_until_ready()
psi_dp.block_until_ready()

start = time.perf_counter()
psi_j = compute_Jastrow_part_api(hamiltonian_data.wavefunction_data.jastrow_data, r_up_carts, r_dn_carts)
psi_j.block_until_ready()
end = time.perf_counter()
print(f"Comput. elapsed Time (jas) = {(end - start) * 1e3:.3f} msec.")

start = time.perf_counter()
psi_jp = vmap(compute_Jastrow_part_api, in_axes=(None, 0, 0))(
    hamiltonian_data.wavefunction_data.jastrow_data, r_up_carts_combined, r_dn_carts_combined
)
psi_jp.block_until_ready()
end = time.perf_counter()
print(f"Comput. elapsed Time (jas-vmap) = {(end - start) * 1e3:.3f} msec.")

start = time.perf_counter()
psi_d = _compute_det_geminal_all_elements_jax(hamiltonian_data.wavefunction_data.geminal_data, r_up_carts, r_dn_carts)
psi_d.block_until_ready()
end = time.perf_counter()
print(f"Comput. elapsed Time (det) = {(end - start) * 1e3:.3f} msec.")

start = time.perf_counter()
psi_dp = vmap(_compute_det_geminal_all_elements_jax, in_axes=(None, 0, 0))(
    hamiltonian_data.wavefunction_data.geminal_data, r_up_carts_combined, r_dn_carts_combined
)
psi_dp.block_until_ready()
end = time.perf_counter()
print(f"Comput. elapsed Time (det-vmap) = {(end - start) * 1e3:.3f} msec.")


wf_ratio = psi_jp * psi_dp / (psi_j * psi_d)


# Compute the kinetic part elements
elements_kinetic_part = -1.0 / (2.0 * alat**2) * wf_ratio

r_up_carts_combined.block_until_ready()
r_dn_carts_combined.block_until_ready()
elements_kinetic_part.block_until_ready()
