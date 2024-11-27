"""Wavefunction module"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# python modules
# from dataclasses import dataclass
from logging import Formatter, StreamHandler, getLogger

# import jax
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import jit

from .determinant import Geminal_data, compute_det_geminal_all_elements_api, compute_grads_and_laplacian_ln_Det_api
from .jastrow_factor import Jastrow_data, compute_grads_and_laplacian_Jastrow_part_api, compute_Jastrow_part_api

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


@struct.dataclass
class Wavefunction_data:
    """
    The class contains data for wavefunction

    Args:
        jastrow_data (Jastrow_data)
        geminal_data (Geminal_data)
    """

    jastrow_data: Jastrow_data = struct.field(pytree_node=True)
    geminal_data: Geminal_data = struct.field(pytree_node=True)

    def __post_init__(self) -> None:
        pass


def evaluate_ln_wavefunction_api(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """
    The method is for evaluate the logarithm of |wavefunction| (ln |Psi|) at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns
    -------
        The value of the given wavefunction (float)
    """
    return jnp.log(
        jnp.abs(
            evaluate_wavefunction_api(
                wavefunction_data=wavefunction_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )
        )
    )


def evaluate_wavefunction_api(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """
    The method is for evaluate wavefunction (Psi) at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns
    -------
        The value of the given wavefunction (float)
    """
    Jastrow_part = compute_Jastrow_part_api(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    Determinant_part = compute_det_geminal_all_elements_api(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return jnp.exp(Jastrow_part) * Determinant_part


def evaluate_jastrow_api(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """
    The method is for evaluate the Jastrow part of the wavefunction (Psi) at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns
    -------
        The value of the given Jastrow (float)
    """
    Jastrow_part = compute_Jastrow_part_api(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return Jastrow_part


def evaluate_determinant_api(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """
    The method is for evaluate the determinant part of the wavefunction (Psi) at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns
    -------
        The value of the given determinant (float)
    """
    Determinant_part = compute_det_geminal_all_elements_api(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return Determinant_part


def compute_kinetic_energy_api(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """
    The method is for computing kinetic energy of the given WF at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns
    -------
        The value of laplacian the given wavefunction (float | complex)
    """
    grad_J_up, grad_J_dn, sum_laplacian_J = compute_grads_and_laplacian_Jastrow_part_api(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    grad_ln_D_up, grad_ln_D_dn, sum_laplacian_ln_D = compute_grads_and_laplacian_ln_Det_api(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # compute kinetic energy
    L = (
        1.0
        / 2.0
        * (
            -(sum_laplacian_J + sum_laplacian_ln_D)
            - (
                jnp.sum((grad_J_up + grad_ln_D_up) * (grad_J_up + grad_ln_D_up))
                + jnp.sum((grad_J_dn + grad_ln_D_dn) * (grad_J_dn + grad_ln_D_dn))
            )
        )
    )

    return L


def compute_discretized_kinetic_energy_debug(
    alat: float, wavefunction_data: Wavefunction_data, r_up_carts: npt.NDArray, r_dn_carts: npt.NDArray
) -> list[tuple[npt.NDArray, npt.NDArray]]:
    r"""_summary.

    Args:
        alat (float): Hamiltonian discretization (bohr), which will be replaced with LRDMC_data.
        wavefunction_data (Wavefunction_data): an instance of Qavefunction_data, which will be replaced with LRDMC_data.
        r_carts_up (npt.NDArray): up electron position (N_e,3).
        r_carts_dn (npt.NDArray): down electron position (N_e,3).

    Returns:
        list[tuple[npt.NDArray, npt.NDArray]], list[npt.NDArray]:
            return mesh for the LRDMC kinetic part, a list containing tuples containing (r_carts_up, r_carts_dn),
            and a list containing values of the \Psi(x')/\Psi(x) corresponding to the grid.
    """
    mesh_kinetic_part = []

    # up electron
    for r_up_i in range(len(r_up_carts)):
        # x, plus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 0] += alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # x, minus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 0] -= alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # y, plus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 1] += alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # y, minus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 1] -= alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # z, plus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 2] += alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # z, minus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 2] -= alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))

    # dn electron
    for r_dn_i in range(len(r_dn_carts)):
        # x, plus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 0] += alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # x, minus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 0] -= alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # y, plus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 1] += alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # y, minus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 1] -= alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # z, plus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 2] += alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # z, minus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 2] -= alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))

    elements_kinetic_part = [
        float(
            -1.0
            / (2.0 * alat**2)
            * evaluate_wavefunction_api(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_, r_dn_carts=r_dn_carts_)
            / evaluate_wavefunction_api(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
        )
        for r_up_carts_, r_dn_carts_ in mesh_kinetic_part
    ]

    return mesh_kinetic_part, elements_kinetic_part


@jit
def compute_discretized_kinetic_energy_jax(
    alat: float, wavefunction_data, r_up_carts: jnp.ndarray, r_dn_carts: jnp.ndarray, jax_PRNG_key: jax.Array = None
) -> tuple[list[tuple[npt.NDArray, npt.NDArray]], list[npt.NDArray], jax.Array]:
    r"""_Summary.

    Args:
        alat (float): Hamiltonian discretization (bohr), which will be replaced with LRDMC_data.
        wavefunction_data (Wavefunction_data): an instance of Qavefunction_data, which will be replaced with LRDMC_data.
        r_carts_up (npt.NDArray): up electron position (N_e,3).
        r_carts_dn (npt.NDArray): down electron position (N_e,3).
        jax_PRNG_key (jax.random.PRNGKey): A pseudo-random number generator key used by JAX for generating random numbers

    Returns:
        list[tuple[npt.NDArray, npt.NDArray]], list[npt.NDArray], jax.Array:
            return mesh for the LRDMC kinetic part, a list containing tuples containing (r_carts_up, r_carts_dn),
            a list containing values of the \Psi(x')/\Psi(x) corresponding to the grid, and the new jax_PRNG_key
            that should be used in the next call of this @jitted function.
    """
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

    if jax_PRNG_key is not None:
        # Specify rotation angles in radians
        jax_PRNG_key, jax_PRNG_subkey = jax.random.split(jax_PRNG_key)
        alpha, beta, gamma = jax.random.uniform(
            jax_PRNG_subkey, shape=(3,), minval=-2 * jnp.pi, maxval=2 * jnp.pi
        )  # Rotation angle around the x,y,z-axis (in radians)

        # Define the rotation matrix for rotation around the x-axis
        def rotation_matrix_x(alpha):
            cos_a = jnp.cos(alpha)
            sin_a = jnp.sin(alpha)
            R_x = jnp.array(
                [
                    [1, 0, 0],
                    [0, cos_a, -sin_a],
                    [0, sin_a, cos_a],
                ]
            )
            return R_x

        # Define the rotation matrix for rotation around the y-axis
        def rotation_matrix_y(beta):
            cos_b = jnp.cos(beta)
            sin_b = jnp.sin(beta)
            R_y = jnp.array(
                [
                    [cos_b, 0, sin_b],
                    [0, 1, 0],
                    [-sin_b, 0, cos_b],
                ]
            )
            return R_y

        # Define the rotation matrix for rotation around the z-axis
        def rotation_matrix_z(gamma):
            cos_g = jnp.cos(gamma)
            sin_g = jnp.sin(gamma)
            R_z = jnp.array(
                [
                    [cos_g, -sin_g, 0],
                    [sin_g, cos_g, 0],
                    [0, 0, 1],
                ]
            )
            return R_z

        # Compute individual rotation matrices
        R_x = rotation_matrix_x(alpha)
        R_y = rotation_matrix_y(beta)
        R_z = rotation_matrix_z(gamma)

        # Compute the combined rotation matrix (order matters)
        R = R_z @ R_y @ R_x  # Rotate in the order x -> y -> z

        # Apply the rotation to the shift vectors
        shifts = shifts @ R.T  # Shape: (6, 3)

    else:
        jax_PRNG_key = None

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

    # Evaluate the wavefunction at the original positions
    psi_x = evaluate_wavefunction_api(wavefunction_data, r_up_carts, r_dn_carts)

    # Evaluate the wavefunction at the shifted positions using vectorization
    def eval_psi(r_up, r_dn):
        return evaluate_wavefunction_api(wavefunction_data, r_up, r_dn)

    psi_xp = jax.vmap(eval_psi)(r_up_carts_combined, r_dn_carts_combined)

    # Compute the kinetic part elements
    elements_kinetic_part = -1.0 / (2.0 * alat**2) * psi_xp / psi_x

    # Determine the maximum number of electrons
    N_up = r_up_carts_combined.shape[1]
    N_dn = r_dn_carts_combined.shape[1]
    N_max = max(N_up, N_dn)

    # Pad the arrays to have the same number of electrons
    r_up_padded = jnp.pad(r_up_carts_combined, ((0, 0), (0, N_max - N_up), (0, 0)), mode="constant")
    r_dn_padded = jnp.pad(r_dn_carts_combined, ((0, 0), (0, N_max - N_dn), (0, 0)), mode="constant")

    # Stack along the spin axis to get shape (N_configs, 2, N_max, 3)
    mesh_kinetic_part = jnp.stack([r_up_padded, r_dn_padded], axis=1)

    # Return the combined configurations and the kinetic elements
    return mesh_kinetic_part, elements_kinetic_part, jax_PRNG_key


def compute_quantum_force(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    The method is for computing quantum forces at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns
    -------
        The value of quantum forces of the given wavefunction -> return tuple[(N_e^{up}, 3), (N_e^{dn}, 3)]
    """
    grad_J_up, grad_J_dn, _ = 0, 0, 0  # tentative

    grad_ln_D_up, grad_ln_D_dn, _ = compute_grads_and_laplacian_ln_Det_api(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    grad_ln_WF_up = grad_J_up + grad_ln_D_up
    grad_ln_WF_dn = grad_J_dn + grad_ln_D_dn

    return 2.0 * grad_ln_WF_up, 2.0 * grad_ln_WF_dn


if __name__ == "__main__":
    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    import os

    from jax import grad

    from .hamiltonians import Hamiltonian_data
    from .jastrow_factor import Jastrow_data, Jastrow_two_body_data
    from .trexio_wrapper import read_trexio_file

    # """
    # water cc-pVTZ with Mitas ccECP (8 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "water_ccpvtz_trexio.hdf5"))
    # """

    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    # WF data
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_two_body_pade_flag=True,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wavefunction_data
    )

    # Initialization
    r_up_carts = []
    r_dn_carts = []

    total_electrons = 0

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data.positions_cart

    # Place electrons around each nucleus
    for i in range(len(coords)):
        charge = charges[i]
        num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

        # Retrieve the position coordinates
        x, y, z = coords[i]

        # Place electrons
        for _ in range(num_electrons):
            # Calculate distance range
            distance = np.random.uniform(1.0 / charge, 2.0 / charge)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            # Convert spherical to Cartesian coordinates
            dx = distance * np.sin(theta) * np.cos(phi)
            dy = distance * np.sin(theta) * np.sin(phi)
            dz = distance * np.cos(theta)

            # Position of the electron
            electron_position = np.array([x + dx, y + dy, z + dz])

            # Assign spin
            if len(r_up_carts) < num_electron_up:
                r_up_carts.append(electron_position)
            else:
                r_dn_carts.append(electron_position)

        total_electrons += num_electrons

    # Handle surplus electrons
    remaining_up = num_electron_up - len(r_up_carts)
    remaining_dn = num_electron_dn - len(r_dn_carts)

    # Randomly place any remaining electrons
    for _ in range(remaining_up):
        r_up_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))
    for _ in range(remaining_dn):
        r_dn_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))

    r_up_carts = np.array(r_up_carts)
    r_dn_carts = np.array(r_dn_carts)

    """ test discritized kinetic mesh
    alat = 0.05
    mesh_kinetic_part_debug, elements_kinetic_part_debug = compute_discretized_kinetic_energy_debug(
        alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    mesh_kinetic_part_jax, elements_kinetic_part_jax, _ = compute_discretized_kinetic_energy_jax(
        alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    np.testing.assert_array_almost_equal(mesh_kinetic_part_jax, mesh_kinetic_part_debug, decimal=10)
    np.testing.assert_array_almost_equal(elements_kinetic_part_jax, elements_kinetic_part_debug, decimal=10)

    jax_PRNG_key = jax.random.PRNGKey(42)

    mesh_kinetic_part_jax_pp1, elements_kinetic_part_jax_pp1, new_jax_PRNG_key = compute_discretized_kinetic_energy_jax(
        alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts, jax_PRNG_key=jax_PRNG_key
    )

    mesh_kinetic_part_jax_pp2, elements_kinetic_part_jax_pp2, new_jax_PRNG_key = compute_discretized_kinetic_energy_jax(
        alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts, jax_PRNG_key=jax_PRNG_key
    )

    np.testing.assert_array_almost_equal(elements_kinetic_part_jax_pp1, elements_kinetic_part_jax_pp2, decimal=10)

    mesh_kinetic_part_jax_pp3, elements_kinetic_part_jax_pp3, new_jax_PRNG_key = compute_discretized_kinetic_energy_jax(
        alat=alat,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        jax_PRNG_key=new_jax_PRNG_key,
    )

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_almost_equal(elements_kinetic_part_jax_pp1, elements_kinetic_part_jax_pp3, decimal=10)
    """

    # test jax grad
    grad_ln_Psi_h = grad(evaluate_ln_wavefunction_api, argnums=(0))(
        hamiltonian_data.wavefunction_data,
        r_up_carts,
        r_dn_carts,
    )

    grad_ln_Psi_jastrow2b_param_jax = grad_ln_Psi_h.jastrow_data.jastrow_two_body_data.jastrow_2b_param

    d_jastrow2b_param = 1.0e-5

    # WF data
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0 + d_jastrow2b_param)

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_two_body_pade_flag=True,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wavefunction_data
    )

    ln_Psi_h_p = evaluate_ln_wavefunction_api(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

    # WF data
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0 - d_jastrow2b_param)

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_two_body_pade_flag=True,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wavefunction_data
    )

    ln_Psi_h_m = evaluate_ln_wavefunction_api(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

    grad_ln_Psi_jastrow2b_param_fdm = (ln_Psi_h_p - ln_Psi_h_m) / (2.0 * d_jastrow2b_param)

    np.testing.assert_almost_equal(grad_ln_Psi_jastrow2b_param_fdm, grad_ln_Psi_jastrow2b_param_jax, decimal=6)
