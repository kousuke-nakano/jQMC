"""LRDMC module.

Todo:
    The bottleneck of the LRDMC projection is that numbers of projection
    are different among OpenMP/MPI distributed walkers, which makes many
    idle walkers. Can we use different numbers of projection for each
    walker??
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
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

import logging
import os
import sys
import time
from functools import partial
from logging import Formatter, StreamHandler, getLogger

import jax
import numpy as np
import numpy.typing as npt
from jax import jit, vmap
from jax import numpy as jnp
from mpi4py import MPI

from .coulomb_potential import (
    _compute_bare_coulomb_potential_jax,
    _compute_ecp_local_parts_full_NN_jax,
    _compute_ecp_non_local_parts_NN_jax,
)
from .hamiltonians import Hamiltonian_data, compute_kinetic_energy_api
from .jastrow_factor import compute_ratio_Jastrow_part_api
from .wavefunction import compute_discretized_kinetic_energy_api

# MPI related
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# create new logger level for development
DEVEL_LEVEL = 5
logging.addLevelName(DEVEL_LEVEL, "DEVEL")


# a new method to create a new logger
def _loglevel_devel(self, message, *args, **kwargs):
    if self.isEnabledFor(DEVEL_LEVEL):
        self._log(DEVEL_LEVEL, message, args, **kwargs)


logging.Logger.devel = _loglevel_devel

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


class GFMC_multiple_walkers:
    """GFMC class.

    GFMC class. Runing GFMC.

    Args:
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
        num_walkers (int): the number of walkers
        mcmc_seed (int): seed for the MCMC chain.
        tau (float): projection time (bohr^-1)
        alat (float): discretized grid length (bohr)
        non_local_move (str):
            treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
            Valid only for ECP calculations. All-electron calculations, do not specify this value.
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data = None,
        num_walkers: int = 40,
        mcmc_seed: int = 34467,
        E_scf: float = 0.0,
        gamma: float = 10.0,
        alat: float = 0.1,
        non_local_move: str = "tmove",
    ) -> None:
        """Init.

        Initialize a MCMC class, creating list holding results, etc...

        """
        self.__hamiltonian_data = hamiltonian_data
        self.__num_walkers = num_walkers
        self.__mcmc_seed = mcmc_seed
        self.__E_scf = E_scf
        self.__gamma = gamma
        self.__alat = alat
        self.__non_local_move = non_local_move

        self.__num_survived_walkers = 0
        self.__num_killed_walkers = 0
        self.__e_L_averaged_list = []
        self.__w_L_averaged_list = []

        # timer
        self.__timer_gmfc_init = 0.0
        self.__timer_gmfc_total = 0.0
        self.__timer_projection_init = 0.0
        self.__timer_projection_total = 0.0
        self.__timer_branching = 0.0
        self.__timer_observable = 0.0

        # gfmc branching counter
        self.__gfmc_branching_counter = 0

        start = time.perf_counter()
        # Initialization
        self.__mpi_seed = mcmc_seed * (mpi_rank + 1)
        self.__jax_PRNG_key = jax.random.PRNGKey(self.__mpi_seed)
        self.__jax_PRNG_key_list_init = jnp.array(
            [jax.random.fold_in(self.__jax_PRNG_key, nw) for nw in range(self.__num_walkers)]
        )
        self.__jax_PRNG_key_list = self.__jax_PRNG_key_list_init

        # Place electrons around each nucleus
        num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn

        r_carts_up_list = []
        r_carts_dn_list = []

        np.random.seed(self.__mpi_seed)
        for _ in range(self.__num_walkers):
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

            r_carts_up_list.append(r_carts_up)
            r_carts_dn_list.append(r_carts_dn)

        self.__latest_r_up_carts = jnp.array(r_carts_up_list)
        self.__latest_r_dn_carts = jnp.array(r_carts_dn_list)

        logger.info(f"The number of MPI process = {mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self.__num_walkers}.")

        logger.debug(f"initial r_up_carts= {self.__latest_r_up_carts}")
        logger.debug(f"initial r_dn_carts = {self.__latest_r_dn_carts}")
        logger.debug(f"initial r_up_carts.shape = {self.__latest_r_up_carts.shape}")
        logger.debug(f"initial r_dn_carts.shape = {self.__latest_r_dn_carts.shape}")
        logger.info("")

        # print out structure info
        logger.info("Structure information:")
        self.__hamiltonian_data.structure_data.logger_info()
        logger.info("")

        logger.info("Compilation of fundamental functions starts.")

        logger.info("  Compilation e_L starts.")
        _ = compute_kinetic_energy_api(
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts[0],
            r_dn_carts=self.__latest_r_dn_carts[0],
        )
        _, _, _ = compute_discretized_kinetic_energy_api(
            alat=self.__alat,
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts[0],
            r_dn_carts=self.__latest_r_dn_carts[0],
            RT=jnp.eye(3, 3),
        )
        _ = _compute_bare_coulomb_potential_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            r_up_carts=self.__latest_r_up_carts[0],
            r_dn_carts=self.__latest_r_dn_carts[0],
        )
        _ = _compute_ecp_local_parts_full_NN_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            r_up_carts=self.__latest_r_up_carts[0],
            r_dn_carts=self.__latest_r_dn_carts[0],
        )
        if self.__non_local_move == "tmove":
            _, _, _, _ = _compute_ecp_non_local_parts_NN_jax(
                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                r_up_carts=self.__latest_r_up_carts[0],
                r_dn_carts=self.__latest_r_dn_carts[0],
                flag_determinant_only=False,
            )
        elif self.__non_local_move == "dltmove":
            _, _, _, _ = _compute_ecp_non_local_parts_NN_jax(
                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                r_up_carts=self.__latest_r_up_carts[0],
                r_dn_carts=self.__latest_r_dn_carts[0],
                flag_determinant_only=True,
            )
        else:
            logger.error(f"non_local_move = {self.__non_local_move} is not yet implemented.")
            raise NotImplementedError

        end = time.perf_counter()
        self.__timer_gmfc_init += end - start
        logger.info("  Compilation e_L is done.")

        logger.info("Compilation of fundamental functions is done.")
        logger.info(f"Elapsed Time = {self.__timer_gmfc_init:.2f} sec.")
        logger.info("")

    def run(self, num_branching: int = 50, num_projection: int = 20, max_time: int = 86400) -> None:
        """Run LRDMC with multiple walkers.

        Args:
            num_branching (int): number of branching (reconfiguration of walkers).
            max_time (int): maximum time in sec.
        """
        # set timer
        timer_projection_init = 0.0
        timer_projection_total = 0.0
        timer_observable = 0.0
        timer_reconfiguration = 0.0
        gmfc_total_start = time.perf_counter()

        # projection function.
        start_init = time.perf_counter()
        logger.info("Start compilation of the GMFC projection funciton.")

        @jit
        def generate_rotation_matrix(alpha, beta, gamma):
            # Precompute all necessary cosines and sines
            cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
            cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
            cos_g, sin_g = jnp.cos(gamma), jnp.sin(gamma)

            # Combine the rotations directly
            R = jnp.array(
                [
                    [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
                    [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
                    [-sin_b, cos_b * sin_a, cos_a * cos_b],
                ]
            )
            return R

        @partial(jit, static_argnums=6)
        def _projection(
            init_w_L: float,
            init_r_up_carts: jax.Array,
            init_r_dn_carts: jax.Array,
            init_jax_PRNG_key: jax.Array,
            E_scf: float,
            num_projection: int,
            non_local_move: bool,
        ):
            """Do projection, compatible with vmap.

            Do projection for a set of (r_up_cart, r_dn_cart).

            Args:
                E(float): trial total energy
                init_w_L (float): weight before projection
                init_r_up_carts (N_e^up, 3) before projection
                init_r_dn_carts (N_e^dn, 3) before projection
            Returns:
                latest_w_L (float): weight after the final projection
                latest_r_up_carts (N_e^up, 3) after the final projection
                latest_r_dn_carts (N_e^dn, 3) after the final projection
            """
            logger.debug(f"init_jax_PRNG_key={init_jax_PRNG_key}")

            @jit
            def body_fun(_, carry):
                w_L, r_up_carts, r_dn_carts, jax_PRNG_key = carry
                # compute non-diagonal grids and elements (kinetic)

                # generate a random rotation matrix
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                alpha, beta, gamma = jax.random.uniform(
                    subkey, shape=(3,), minval=-2 * jnp.pi, maxval=2 * jnp.pi
                )  # Rotation angle around the x,y,z-axis (in radians)

                R = generate_rotation_matrix(alpha, beta, gamma)  # Rotate in the order x -> y -> z

                # compute discretized kinetic energy and mesh (with a random rotation)
                mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                    compute_discretized_kinetic_energy_api(
                        alat=self.__alat,
                        wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                        RT=R.T,
                    )
                )
                elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
                diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
                non_diagonal_sum_hamiltonian_kinetic = jnp.sum(elements_non_diagonal_kinetic_part_FN)

                # compute diagonal elements, kinetic part
                diagonal_kinetic_continuum = compute_kinetic_energy_api(
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
                diagonal_kinetic_discretized = -1.0 * jnp.sum(elements_non_diagonal_kinetic_part)

                # compute diagonal elements, bare couloumb
                diagonal_bare_coulomb_part = _compute_bare_coulomb_potential_jax(
                    coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

                # """ if-else for all-ele, ecp with tmove, and ecp with dltmove
                # with ECP
                if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
                    # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                    # ecp local
                    diagonal_ecp_local_part = _compute_ecp_local_parts_full_NN_jax(
                        coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    if non_local_move == "tmove":
                        # ecp non-local (t-move)
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            _compute_ecp_non_local_parts_NN_jax(
                                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=False,
                            )
                        )

                        V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                        diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                        non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                        non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                        diagonal_sum_hamiltonian = (
                            diagonal_kinetic_continuum
                            + diagonal_kinetic_discretized
                            + diagonal_bare_coulomb_part
                            + diagonal_ecp_local_part
                            + diagonal_kinetic_part_SP
                            + diagonal_ecp_part_SP
                        )

                    elif non_local_move == "dltmove":
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            _compute_ecp_non_local_parts_NN_jax(
                                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=True,
                            )
                        )

                        V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                        diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))

                        Jastrow_ratio = compute_ratio_Jastrow_part_api(
                            jastrow_data=self.__hamiltonian_data.wavefunction_data.jastrow_data,
                            old_r_up_carts=r_up_carts,
                            old_r_dn_carts=r_dn_carts,
                            new_r_up_carts_arr=mesh_non_local_ecp_part_r_up_carts,
                            new_r_dn_carts_arr=mesh_non_local_ecp_part_r_dn_carts,
                        )
                        V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

                        non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                        non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                        diagonal_sum_hamiltonian = (
                            diagonal_kinetic_continuum
                            + diagonal_kinetic_discretized
                            + diagonal_bare_coulomb_part
                            + diagonal_ecp_local_part
                            + diagonal_kinetic_part_SP
                            + diagonal_ecp_part_SP
                        )

                    else:
                        logger.error(f"non_local_move = {self.__non_local_move} is not yet implemented.")
                        raise NotImplementedError

                    # probability
                    p_list = jnp.concatenate([jnp.ravel(elements_non_diagonal_kinetic_part_FN), jnp.ravel(V_nonlocal_FN)])
                    non_diagonal_move_probabilities = p_list / p_list.sum()
                    non_diagonal_move_mesh_r_up_carts = jnp.concatenate(
                        [mesh_kinetic_part_r_up_carts, mesh_non_local_ecp_part_r_up_carts], axis=0
                    )
                    non_diagonal_move_mesh_r_dn_carts = jnp.concatenate(
                        [mesh_kinetic_part_r_dn_carts, mesh_non_local_ecp_part_r_dn_carts], axis=0
                    )

                # with all electrons
                else:
                    # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                    p_list = jnp.ravel(elements_non_diagonal_kinetic_part_FN)
                    non_diagonal_move_probabilities = p_list / p_list.sum()
                    non_diagonal_move_mesh_r_up_carts = mesh_kinetic_part_r_up_carts
                    non_diagonal_move_mesh_r_dn_carts = mesh_kinetic_part_r_dn_carts

                    diagonal_sum_hamiltonian = (
                        diagonal_kinetic_continuum
                        + diagonal_kinetic_discretized
                        + diagonal_bare_coulomb_part
                        + diagonal_ecp_local_part
                        + diagonal_kinetic_part_SP
                        + diagonal_ecp_part_SP
                    )

                # compute b_L_bar
                b_x_bar = -1.0 * non_diagonal_sum_hamiltonian
                logger.debug(f"  b_x_bar={b_x_bar}")

                # compute bar_b_L
                logger.debug(f"  diagonal_sum_hamiltonian={diagonal_sum_hamiltonian}")
                logger.debug(f"  E_scf={E_scf}")
                b_x = 1.0 / (diagonal_sum_hamiltonian - E_scf) ** (1.0 + self.__gamma * self.__alat**2) * b_x_bar
                logger.debug(f"  b_x={b_x}")

                # update weight
                logger.debug(f"  old: w_L={w_L}")
                w_L = w_L * b_x
                logger.debug(f"  new: w_L={w_L}")

                # electron position update
                # random choice
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                cdf = jnp.cumsum(non_diagonal_move_probabilities)
                random_value = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
                k = jnp.searchsorted(cdf, random_value)
                logger.debug(f"len(non_diagonal_move_probabilities) = {len(non_diagonal_move_probabilities)}.")
                logger.debug(f"chosen update electron index, k = {k}.")
                logger.debug(f"old: r_up_carts = {r_up_carts}")
                logger.debug(f"old: r_dn_carts = {r_dn_carts}")
                r_up_carts = non_diagonal_move_mesh_r_up_carts[k]
                r_dn_carts = non_diagonal_move_mesh_r_dn_carts[k]
                logger.debug(f"new: r_up_carts={r_up_carts}.")
                logger.debug(f"new: r_dn_carts={r_dn_carts}.")

                carry = (w_L, r_up_carts, r_dn_carts, jax_PRNG_key)
                return carry

            latest_w_L, latest_r_up_carts, latest_r_dn_carts, latest_jax_PRNG_key = jax.lax.fori_loop(
                0, num_projection, body_fun, (init_w_L, init_r_up_carts, init_r_dn_carts, init_jax_PRNG_key)
            )

            return (latest_w_L, latest_r_up_carts, latest_r_dn_carts, latest_jax_PRNG_key)

        @partial(jit, static_argnums=3)
        def _compute_observable(
            r_up_carts: jax.Array,
            r_dn_carts: jax.Array,
            jax_PRNG_key: jax.Array,
            non_local_move: bool,
        ):
            # compute non-diagonal grids and elements (kinetic)

            # generate a random rotation matrix
            jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
            alpha, beta, gamma = jax.random.uniform(
                subkey, shape=(3,), minval=-2 * jnp.pi, maxval=2 * jnp.pi
            )  # Rotation angle around the x,y,z-axis (in radians)

            R = generate_rotation_matrix(alpha, beta, gamma)  # Rotate in the order x -> y -> z

            # compute discretized kinetic energy and mesh (with a random rotation)
            mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                compute_discretized_kinetic_energy_api(
                    alat=self.__alat,
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                    RT=R.T,
                )
            )
            elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
            diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
            non_diagonal_sum_hamiltonian_kinetic = jnp.sum(elements_non_diagonal_kinetic_part_FN)

            # compute diagonal elements, kinetic part
            diagonal_kinetic_continuum = compute_kinetic_energy_api(
                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )
            diagonal_kinetic_discretized = -1.0 * jnp.sum(elements_non_diagonal_kinetic_part)

            # compute diagonal elements, bare couloumb
            diagonal_bare_coulomb_part = _compute_bare_coulomb_potential_jax(
                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

            # """ if-else for all-ele, ecp with tmove, and ecp with dltmove
            # with ECP
            if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
                # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                # ecp local
                diagonal_ecp_local_part = _compute_ecp_local_parts_full_NN_jax(
                    coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

                if non_local_move == "tmove":
                    # ecp non-local (t-move)
                    mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                        _compute_ecp_non_local_parts_NN_jax(
                            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                            flag_determinant_only=False,
                        )
                    )

                    V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                    diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                    non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                    diagonal_sum_hamiltonian = (
                        diagonal_kinetic_continuum
                        + diagonal_kinetic_discretized
                        + diagonal_bare_coulomb_part
                        + diagonal_ecp_local_part
                        + diagonal_kinetic_part_SP
                        + diagonal_ecp_part_SP
                    )

                elif non_local_move == "dltmove":
                    mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                        _compute_ecp_non_local_parts_NN_jax(
                            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                            flag_determinant_only=True,
                        )
                    )

                    V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                    diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))

                    Jastrow_ratio = compute_ratio_Jastrow_part_api(
                        jastrow_data=self.__hamiltonian_data.wavefunction_data.jastrow_data,
                        old_r_up_carts=r_up_carts,
                        old_r_dn_carts=r_dn_carts,
                        new_r_up_carts_arr=mesh_non_local_ecp_part_r_up_carts,
                        new_r_dn_carts_arr=mesh_non_local_ecp_part_r_dn_carts,
                    )
                    V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

                    non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                    diagonal_sum_hamiltonian = (
                        diagonal_kinetic_continuum
                        + diagonal_kinetic_discretized
                        + diagonal_bare_coulomb_part
                        + diagonal_ecp_local_part
                        + diagonal_kinetic_part_SP
                        + diagonal_ecp_part_SP
                    )

                else:
                    logger.error(f"non_local_move = {self.__non_local_move} is not yet implemented.")
                    raise NotImplementedError

                # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                e_L = (
                    diagonal_kinetic_continuum
                    + diagonal_kinetic_discretized
                    + diagonal_bare_coulomb_part
                    + diagonal_ecp_local_part
                    + diagonal_kinetic_part_SP
                    + diagonal_ecp_part_SP
                    + non_diagonal_sum_hamiltonian
                )

            # with all electrons
            else:
                # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                e_L = (
                    diagonal_kinetic_continuum
                    + diagonal_kinetic_discretized
                    + diagonal_bare_coulomb_part
                    + diagonal_kinetic_part_SP
                    + non_diagonal_sum_hamiltonian
                )

                diagonal_sum_hamiltonian = (
                    diagonal_kinetic_continuum
                    + diagonal_kinetic_discretized
                    + diagonal_bare_coulomb_part
                    + diagonal_ecp_local_part
                    + diagonal_kinetic_part_SP
                    + diagonal_ecp_part_SP
                )

            V_diag = diagonal_sum_hamiltonian

            logger.debug(f"  e_L={e_L}")
            logger.debug(f"  V_diag={V_diag}")
            # """

            return (e_L, V_diag)

        # projection compilation.
        logger.info("  Compilation is in progress...")
        w_L_list = jnp.array([1.0 for _ in range(self.__num_walkers)])
        (
            _,
            _,
            _,
            _,
        ) = vmap(_projection, in_axes=(0, 0, 0, 0, None, None, None))(
            w_L_list,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
            self.__jax_PRNG_key_list,
            self.__E_scf,
            num_projection,
            self.__non_local_move,
        )

        end_init = time.perf_counter()
        timer_projection_init += end_init - start_init
        logger.info("End compilation of the GMFC projection funciton.")
        logger.info(f"Elapsed Time = {timer_projection_init:.2f} sec.")
        logger.info("")

        # Main branching loop.
        gfmc_interval = int(np.maximum(num_branching / 100, 1))  # gfmc_projection set print-interval

        logger.info("-Start branching-")
        progress = (self.__gfmc_branching_counter) / (num_branching + self.__gfmc_branching_counter) * 100.0
        gmfc_total_current = time.perf_counter()
        logger.info(
            f"  branching step = {self.__gfmc_branching_counter}/{num_branching + self.__gfmc_branching_counter}: {progress:.1f} %. Elapsed time = {(gmfc_total_current - gmfc_total_start):.1f} sec."
        )

        num_branching_done = 0
        for i_branching in range(num_branching):
            if (i_branching + 1) % gfmc_interval == 0:
                progress = (
                    (i_branching + self.__gfmc_branching_counter + 1) / (num_branching + self.__gfmc_branching_counter) * 100.0
                )
                gmfc_total_current = time.perf_counter()
                logger.info(
                    f"  branching step = {i_branching + self.__gfmc_branching_counter + 1}/{num_branching + self.__gfmc_branching_counter}: {progress:.1f} %. Elapsed time = {(gmfc_total_current - gmfc_total_start):.1f} sec."
                )

            # Always set the initial weight list to 1.0
            w_L_list = jnp.array([1.0 for _ in range(self.__num_walkers)])

            logger.debug("  Projection is on going....")

            start_projection = time.perf_counter()

            # projection loop
            logger.debug(f"  in: w_L_list = {w_L_list}.")
            (
                w_L_list,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                self.__jax_PRNG_key_list,
            ) = vmap(_projection, in_axes=(0, 0, 0, 0, None, None, None))(
                w_L_list,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                self.__jax_PRNG_key_list,
                self.__E_scf,
                num_projection,
                self.__non_local_move,
            )

            # sync. jax arrays computations.
            w_L_list.block_until_ready()
            self.__latest_r_up_carts.block_until_ready()
            self.__latest_r_dn_carts.block_until_ready()
            self.__jax_PRNG_key_list.block_until_ready()

            end_projection = time.perf_counter()
            timer_projection_total += end_projection - start_projection

            # projection ends
            logger.debug("  Projection ends.")

            # evaluate observables
            start_observable = time.perf_counter()
            # e_L and V_diag
            e_L_list, V_diag_list = vmap(_compute_observable, in_axes=(0, 0, 0, None))(
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                self.__jax_PRNG_key_list,
                self.__non_local_move,
            )
            # to be implemented other observables, such as derivatives.
            end_observable = time.perf_counter()
            timer_observable += end_observable - start_observable

            # Barrier before MPI operation
            # mpi_comm.Barrier()

            # Branching starts
            start_reconfiguration = time.perf_counter()

            # jnp.array -> np.array
            w_L_latest = np.array(w_L_list)
            e_L_latest = np.array(e_L_list)
            V_diag_E_latest = np.array(V_diag_list) - self.__E_scf

            # jnp.array -> np.array
            self.__latest_r_up_carts = np.array(self.__latest_r_up_carts)
            self.__latest_r_dn_carts = np.array(self.__latest_r_dn_carts)

            # MPI reduce
            r_up_carts_shape = self.__latest_r_up_carts.shape
            r_up_carts_gathered_dyad = (mpi_rank, self.__latest_r_up_carts)
            r_up_carts_gathered_dyad = mpi_comm.gather(r_up_carts_gathered_dyad, root=0)

            r_dn_carts_shape = self.__latest_r_dn_carts.shape
            r_dn_carts_gathered_dyad = (mpi_rank, self.__latest_r_dn_carts)
            r_dn_carts_gathered_dyad = mpi_comm.gather(r_dn_carts_gathered_dyad, root=0)

            e_L_gathered_dyad = (mpi_rank, e_L_latest)
            e_L_gathered_dyad = mpi_comm.gather(e_L_gathered_dyad, root=0)
            w_L_gathered_dyad = (mpi_rank, w_L_latest)
            w_L_gathered_dyad = mpi_comm.gather(w_L_gathered_dyad, root=0)
            V_diag_E_gathered_dyad = (mpi_rank, V_diag_E_latest)
            V_diag_E_gathered_dyad = mpi_comm.gather(V_diag_E_gathered_dyad, root=0)

            if mpi_rank == 0:
                logger.debug(f"e_L_gathered_dyad={e_L_gathered_dyad}")
                logger.debug(f"w_L_gathered_dyad={w_L_gathered_dyad}")
                logger.debug(f"V_diag_E_gathered_dyad={V_diag_E_gathered_dyad}")
                r_up_carts_gathered_dict = dict(r_up_carts_gathered_dyad)
                r_dn_carts_gathered_dict = dict(r_dn_carts_gathered_dyad)
                e_L_gathered_dict = dict(e_L_gathered_dyad)
                w_L_gathered_dict = dict(w_L_gathered_dyad)
                V_diag_E_gathered_dict = dict(V_diag_E_gathered_dyad)
                logger.debug(f"  e_L_gathered_dict = {e_L_gathered_dict} Ha")
                logger.debug(f"  w_L_gathered_dict = {w_L_gathered_dict}")
                logger.debug(f"  V_diag_E_gathered_dict = {V_diag_E_gathered_dict} Ha^-1")
                r_up_carts_gathered = np.concatenate([r_up_carts_gathered_dict[i] for i in range(mpi_size)])
                r_dn_carts_gathered = np.concatenate([r_dn_carts_gathered_dict[i] for i in range(mpi_size)])
                e_L_gathered = np.concatenate([e_L_gathered_dict[i] for i in range(mpi_size)])
                w_L_gathered = np.concatenate([w_L_gathered_dict[i] for i in range(mpi_size)])
                V_diag_E_gathered = np.concatenate([V_diag_E_gathered_dict[i] for i in range(mpi_size)])
                logger.debug(f"  e_L_gathered = {e_L_gathered} Ha")
                logger.debug(f"  w_L_gathered = {w_L_gathered}")
                logger.debug(f"  V_diag_E_gathered = {V_diag_E_gathered} Ha^-1")
                e_L_averaged = np.sum(
                    w_L_gathered / V_diag_E_gathered ** (1.0 + self.__gamma * self.__alat**2) * e_L_gathered
                ) / np.sum(w_L_gathered / V_diag_E_gathered ** (1.0 + self.__gamma * self.__alat**2))
                w_L_averaged = np.average(w_L_gathered)
                logger.debug(f"  e_L_averaged = {e_L_averaged} Ha")
                logger.debug(f"  w_L_averaged = {w_L_averaged}")
                self.__e_L_averaged_list.append(e_L_averaged)
                self.__w_L_averaged_list.append(w_L_averaged)
                w_L_list = w_L_gathered
                logger.debug(f"w_L_list = {w_L_list}")
                probabilities = w_L_list / w_L_list.sum()
                logger.debug(f"probabilities = {probabilities}")

                # correlated choice (see Sandro's textbook, page 182)
                self.__jax_PRNG_key, subkey = jax.random.split(self.__jax_PRNG_key)
                zeta = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
                z_list = [(alpha + zeta) / len(probabilities) for alpha in range(len(probabilities))]
                logger.debug(f"z_list = {z_list}")
                cumulative_prob = np.cumsum(probabilities)
                chosen_walker_indices = np.array(
                    [next(idx for idx, prob in enumerate(cumulative_prob) if z <= prob) for z in z_list]
                )
                logger.debug(f"The chosen walker indices = {chosen_walker_indices}")
                logger.debug(f"The chosen walker indices.shape = {chosen_walker_indices.shape}")
                logger.debug(f"r_up_carts_gathered.shape = {r_up_carts_gathered.shape}")
                logger.debug(f"r_dn_carts_gathered.shape = {r_dn_carts_gathered.shape}")

                proposed_r_up_carts = r_up_carts_gathered[chosen_walker_indices]
                proposed_r_dn_carts = r_dn_carts_gathered[chosen_walker_indices]

                self.__num_survived_walkers += len(set(chosen_walker_indices))
                self.__num_killed_walkers += len(w_L_list) - len(set(chosen_walker_indices))
                logger.debug(f"num_survived_walkers={self.__num_survived_walkers}")
                logger.debug(f"num_killed_walkers={self.__num_killed_walkers}")
            else:
                self.__e_L_averaged_list = None
                self.__w_L_averaged_list = None
                self.__num_survived_walkers = None
                self.__num_killed_walkers = None
                proposed_r_up_carts = None
                proposed_r_dn_carts = None

            self.__e_L_averaged_list = mpi_comm.bcast(self.__e_L_averaged_list, root=0)
            self.__w_L_averaged_list = mpi_comm.bcast(self.__w_L_averaged_list, root=0)

            logger.debug(f"Before branching: rank={mpi_rank}:gfmc.r_up_carts = {self.__latest_r_up_carts}")
            logger.debug(f"Before branching: rank={mpi_rank}:gfmc.r_dn_carts = {self.__latest_r_dn_carts}")

            self.__num_survived_walkers = mpi_comm.bcast(self.__num_survived_walkers, root=0)
            self.__num_killed_walkers = mpi_comm.bcast(self.__num_killed_walkers, root=0)

            proposed_r_up_carts = mpi_comm.bcast(proposed_r_up_carts, root=0)
            proposed_r_dn_carts = mpi_comm.bcast(proposed_r_dn_carts, root=0)

            proposed_r_up_carts = proposed_r_up_carts.reshape(
                mpi_size, r_up_carts_shape[0], r_up_carts_shape[1], r_up_carts_shape[2]
            )
            proposed_r_dn_carts = proposed_r_dn_carts.reshape(
                mpi_size, r_dn_carts_shape[0], r_dn_carts_shape[1], r_dn_carts_shape[2]
            )

            # set new r_up_carts and r_dn_carts, and, np.array -> jnp.array
            self.__latest_r_up_carts = proposed_r_up_carts[mpi_rank, :, :, :]
            self.__latest_r_dn_carts = proposed_r_dn_carts[mpi_rank, :, :, :]

            # np.array -> jnp.array
            self.__latest_r_up_carts = jnp.array(self.__latest_r_up_carts)
            self.__latest_r_dn_carts = jnp.array(self.__latest_r_dn_carts)

            logger.debug(f"*After branching: rank={mpi_rank}:gfmc.r_up_carts = {self.__latest_r_up_carts}")
            logger.debug(f"*After branching: rank={mpi_rank}:gfmc.r_dn_carts = {self.__latest_r_dn_carts}")

            end_reconfiguration = time.perf_counter()
            timer_reconfiguration += end_reconfiguration - start_reconfiguration

            # update E_scf
            if (i_branching + 1) % gfmc_interval == 0:
                if i_branching >= 20:
                    E_scf, E_scf_std = self.get_e_L(num_gfmc_warmup_steps=10, num_gfmc_bin_blocks=5, num_gfmc_bin_collect=3)
                    logger.info(f"    Updated E_scf = {E_scf:.5f} +- {E_scf_std:.5f} Ha.")
                    self.__E_scf = E_scf
                else:
                    logger.info(f"    Init E_scf = {self.__E_scf:.5f} Ha. Being equilibrated.")

            num_branching_done += 1
            gmfc_current = time.perf_counter()
            if max_time < gmfc_current - gmfc_total_start:
                logger.info(f"  Max_time = {max_time} sec. exceeds.")
                logger.info("  Break the branching loop.")
                break

        logger.info("-End branching-")
        logger.info("")

        # count up
        self.__gfmc_branching_counter += i_branching + 1

        gmfc_total_end = time.perf_counter()
        timer_gmfc_total = gmfc_total_end - gmfc_total_start

        logger.info(f"Total GFMC time for {num_branching_done} branching steps = {timer_gmfc_total: .3f} sec.")
        logger.info(f"Pre-compilation time for GFMC = {timer_projection_init: .3f} sec.")
        logger.info(f"Net GFMC time without pre-compilations = {timer_gmfc_total - timer_projection_init: .3f} sec.")
        logger.info(f"Elapsed times per branching, averaged over {num_branching_done} branching steps.")
        logger.info(f"  Projection time per branching = {timer_projection_total / num_branching_done * 10**3: .3f} msec.")
        logger.info(f"  Observable measurement time per branching = {timer_observable / num_branching_done * 10**3: .3f} msec.")
        logger.info(
            f"  Walker reconfiguration time per branching = {timer_reconfiguration / num_branching_done * 10**3: .3f} msec."
        )
        logger.debug(f"Survived walkers = {self.__num_survived_walkers}")
        logger.debug(f"killed walkers = {self.__num_killed_walkers}")
        logger.info(
            f"Survived walkers ratio = {self.__num_survived_walkers / (self.__num_survived_walkers + self.__num_killed_walkers) * 100:.2f} %"
        )
        logger.debug(f"self.__e_L_averaged_list = {self.__e_L_averaged_list}.")
        logger.debug(f"self.__w_L_averaged_list = {self.__w_L_averaged_list}.")
        logger.debug(f"len(self.__e_L_averaged_list) = {len(self.__e_L_averaged_list)}.")
        logger.debug(f"len(self.__w_L_averaged_list) = {len(self.__w_L_averaged_list)}.")
        logger.info("")

        self.__timer_gmfc_total += timer_gmfc_total
        self.__timer_projection_init += timer_projection_init
        self.__timer_projection_total += timer_projection_total
        self.__timer_branching += timer_reconfiguration
        self.__timer_observable += timer_observable

    def get_e_L(self, num_gfmc_warmup_steps: int = 3, num_gfmc_bin_blocks: int = 10, num_gfmc_bin_collect: int = 2) -> float:
        """Get e_L."""
        logger.debug("- Comput. e_L -")
        if mpi_rank == 0:
            e_L_eq = self.__e_L_averaged_list[num_gfmc_warmup_steps + num_gfmc_bin_collect :]
            w_L_eq = self.__w_L_averaged_list[num_gfmc_warmup_steps:]
            logger.debug("  Progress: Computing G_eq and G_e_L_eq.")

            @partial(jit, static_argnums=2)
            def compute_G_eq_and_G_e_L_eq_jax(w_L_eq, e_L_eq, num_gfmc_bin_collect):
                def get_slice(n):
                    return jax.lax.dynamic_slice(w_L_eq, (n - num_gfmc_bin_collect,), (num_gfmc_bin_collect,))

                indices = jnp.arange(num_gfmc_bin_collect, len(w_L_eq))
                G_eq_matrix = vmap(get_slice)(indices)
                G_eq = jnp.prod(G_eq_matrix, axis=1)
                G_e_L_eq = e_L_eq * G_eq
                return G_eq, G_e_L_eq

            w_L_eq = jnp.array(w_L_eq)
            e_L_eq = jnp.array(e_L_eq)
            G_eq, G_e_L_eq = compute_G_eq_and_G_e_L_eq_jax(w_L_eq, e_L_eq, num_gfmc_bin_collect)
            G_eq = np.array(G_eq)
            G_e_L_eq = np.array(G_e_L_eq)

            logger.debug(f"  Progress: Computing binned G_e_L_eq and G_eq with # binned blocks = {num_gfmc_bin_blocks}.")
            G_e_L_split = np.array_split(G_e_L_eq, num_gfmc_bin_blocks)
            G_e_L_binned = np.array([np.average(G_e_L_list) for G_e_L_list in G_e_L_split])
            G_split = np.array_split(G_eq, num_gfmc_bin_blocks)
            G_binned = np.array([np.average(G_list) for G_list in G_split])

            logger.debug(f"  Progress: Computing jackknife samples with # binned blocks = {num_gfmc_bin_blocks}.")

            G_e_L_binned_sum = np.sum(G_e_L_binned)
            G_binned_sum = np.sum(G_binned)

            e_L_jackknife = [
                (G_e_L_binned_sum - G_e_L_binned[m]) / (G_binned_sum - G_binned[m]) for m in range(num_gfmc_bin_blocks)
            ]

            logger.debug("  Progress: Computing jackknife mean and std.")
            e_L_mean = np.average(e_L_jackknife)
            e_L_std = np.sqrt(num_gfmc_bin_blocks - 1) * np.std(e_L_jackknife)

            logger.debug(f"  e_L = {e_L_mean} +- {e_L_std} Ha")
        else:
            e_L_mean = None
            e_L_std = None

        e_L_mean = mpi_comm.bcast(e_L_mean, root=0)
        e_L_std = mpi_comm.bcast(e_L_std, root=0)

        return e_L_mean, e_L_std

    @property
    def hamiltonian_data(self):
        """Return hamiltonian_data."""
        return self.__hamiltonian_data

    @property
    def e_L(self) -> npt.NDArray:
        """Return the stored e_L array."""
        return self.__latest_e_L

    @property
    def w_L(self) -> npt.NDArray:
        """Return the stored w_L array."""
        return self.__latest_w_L

    @property
    def num_walkers(self) -> int:
        """The number of walkers."""
        return self.__num_walkers

    @property
    def mcmc_seed(self) -> int:
        """Return the mcmc_seed used for generating initial electron positions. This is used only for generating mpi_seed which differs among MPI processes."""
        return self.__mcmc_seed

    @property
    def mpi_seed(self) -> int:
        """Return the mpi_seed used for generating initial electron positions."""
        return self.__mpi_seed

    @property
    def jax_PRNG_key(self) -> int:
        """Return jax_PRNG_key used for generating jax_PRNG_key. This is used only for generating jax_PRNG_key_list containing different jax_PRNG_key among walkers."""
        return self.__jax_PRNG_key

    @property
    def jax_PRNG_key_list(self) -> int:
        """Return the initial jax_PRNG_key_list used for controlling random numbers in vectorized mcmc update."""
        return self.__jax_PRNG_key_list_init

    @property
    def latest_r_up_carts(self) -> npt.NDArray:
        """Latest updated electron position for up-spin."""
        return self.__latest_r_up_carts

    @property
    def latest_r_dn_carts(self) -> npt.NDArray:
        """Latest updated electron position for down-spin."""
        return self.__latest_r_dn_carts

    @property
    def timer_gmfc_init(self) -> float:
        """Return the measured elapsed time for initialization."""
        return self.__timer_gmfc_init

    @property
    def timer_gmfc_total(self) -> float:
        """Return the measured elapsed time for total GFMC."""
        return self.__timer_gmfc_total

    @property
    def timer_projection_total(self) -> float:
        """Return the measured elapsed time for GFMC projection."""
        return self.__timer_projection_total

    @property
    def timer_branching(self) -> float:
        """Return the measured elapsed time for GFMC branching."""
        return self.__timer_branching

    @property
    def timer_observable(self) -> float:
        """Return the measured elapsed time for computing other observables."""
        return self.__timer_observable


if __name__ == "__main__":
    import pickle

    # import os
    # from .jastrow_factor import Jastrow_data, Jastrow_three_body_data, Jastrow_two_body_data
    # from .trexio_wrapper import read_trexio_file
    # from .wavefunction import Wavefunction_data

    logger_level = "MPI-INFO"

    log = getLogger("jqmc")

    if logger_level == "MPI-INFO":
        if mpi_rank == 0:
            log.setLevel("INFO")
            stream_handler = StreamHandler()
            stream_handler.setLevel("INFO")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
        else:
            log.setLevel("WARNING")
            stream_handler = StreamHandler()
            stream_handler.setLevel("WARNING")
            handler_format = Formatter(f"MPI-rank={mpi_rank}: %(name)s - %(levelname)s - %(lineno)d - %(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
    else:
        log.setLevel(logger_level)
        stream_handler = StreamHandler()
        stream_handler.setLevel(logger_level)
        handler_format = Formatter(f"MPI-rank={mpi_rank}: %(name)s - %(levelname)s - %(lineno)d - %(message)s")
        stream_handler.setFormatter(handler_format)
        log.addHandler(stream_handler)

    # jax-MPI related
    try:
        jax.distributed.initialize()
    except ValueError:
        pass

    # print recognized XLA devices
    logger.info("*** XLA devices recognized by JAX***")
    logger.info(jax.devices())
    logger.info("")

    """
    # water cc-pVTZ with Mitas ccECP (8 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "water_ccpvtz_trexio.hdf5"))

    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_two_body_pade_flag=True,
        jastrow_three_body_data=jastrow_threebody_data,
        jastrow_three_body_flag=True,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    """

    hamiltonian_chk = "hamiltonian_data_water.chk"
    # hamiltonian_chk = "hamiltonian_data_AcOH.chk"
    # hamiltonian_chk = "hamiltonian_data_benzene.chk"
    # hamiltonian_chk = "hamiltonian_data_C60.chk"
    with open(hamiltonian_chk, "rb") as f:
        hamiltonian_data = pickle.load(f)

    # run branching
    num_walkers = 100
    mcmc_seed = 3446
    max_time = 3600
    E_scf = -16.90
    gamma = 1.0e-2
    alat = 0.30
    num_branching = 1000
    num_projection = 30
    non_local_move = "tmove"

    num_gfmc_warmup_steps = 5
    num_gfmc_bin_blocks = 5
    num_gfmc_bin_collect = 3

    # run GFMC with multiple walkers
    gfmc = GFMC_multiple_walkers(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        mcmc_seed=mcmc_seed,
        E_scf=E_scf,
        gamma=gamma,
        alat=alat,
        non_local_move=non_local_move,
    )
    gfmc.run(num_branching=num_branching, num_projection=num_projection, max_time=max_time)

    # """
    e_L_mean, e_L_std = gfmc.get_e_L(
        num_gfmc_warmup_steps=num_gfmc_warmup_steps,
        num_gfmc_bin_blocks=num_gfmc_bin_blocks,
        num_gfmc_bin_collect=num_gfmc_bin_collect,
    )
    logger.info(f"e_L = {e_L_mean} +- {e_L_std} Ha")
    # """
