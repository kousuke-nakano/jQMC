"""LRDMC module."""

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
import time
from collections import Counter
from logging import Formatter, StreamHandler, getLogger

import jax
import numpy as np
import numpy.typing as npt
from jax import jit
from jax import numpy as jnp
from mpi4py import MPI

from .coulomb_potential import (
    _compute_bare_coulomb_potential_jax,
    _compute_ecp_local_parts_full_NN_jax,
    _compute_ecp_non_local_parts_full_NN_jax,
    _compute_ecp_non_local_parts_NN_jax,
)
from .determinant import compute_geminal_all_elements_api
from .hamiltonians import Hamiltonian_data, compute_kinetic_energy_api
from .jastrow_factor import compute_ratio_Jastrow_part_api
from .wavefunction import compute_discretized_kinetic_energy_api, compute_discretized_kinetic_energy_api_fast_update

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


class GFMC:
    """GFMC class.

    GFMC class. Runing GFMC.

    Args:
        mcmc_seed (int): seed for the MCMC chain.
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
        tau (float): projection time (bohr^-1)
        alat (float): discretized grid length (bohr)
        non_local_move (str):
            treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
            Valid only for ECP calculations. All-electron calculations, do not specify this value.
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data = None,
        mcmc_seed: int = 34467,
        tau: float = 0.1,
        alat: float = 0.1,
        non_local_move: str = "tmove",
    ) -> None:
        """Init.

        Initialize a MCMC class, creating list holding results, etc...

        """
        self.__hamiltonian_data = hamiltonian_data
        self.__tau = tau
        self.__alat = alat
        self.__non_local_move = non_local_move

        self.__num_survived_walkers = 0
        self.__num_killed_walkers = 0
        self.__e_L_averaged_list = []
        self.__w_L_averaged_list = []

        # gfmc branching counter
        self.__gfmc_branching_counter = 0

        # Initialization
        init_r_up_carts = []
        init_r_dn_carts = []

        total_electrons = 0

        if hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
                hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

        coords = hamiltonian_data.structure_data.positions_cart

        # set random seeds
        self.__mpi_seed = mcmc_seed * (mpi_rank + 1) ** 2
        np.random.seed(self.__mpi_seed)
        self.__jax_PRNG_key = jax.random.PRNGKey(self.__mpi_seed)

        # Place electrons around each nucleus
        num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn

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
                if len(init_r_up_carts) < num_electron_up:
                    init_r_up_carts.append(electron_position)
                else:
                    init_r_dn_carts.append(electron_position)

            total_electrons += num_electrons

        # Handle surplus electrons
        remaining_up = num_electron_up - len(init_r_up_carts)
        remaining_dn = num_electron_dn - len(init_r_dn_carts)

        # Randomly place any remaining electrons
        for _ in range(remaining_up):
            init_r_up_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))
        for _ in range(remaining_dn):
            init_r_dn_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))

        init_r_up_carts = np.array(init_r_up_carts)
        init_r_dn_carts = np.array(init_r_dn_carts)

        logger.info(f"The number of MPI process = {mpi_size}.")
        logger.debug(f"initial r_up_carts = {init_r_up_carts}")
        logger.debug(f"initial r_dn_carts = {init_r_dn_carts}")

        self.__latest_r_up_carts = init_r_up_carts
        self.__latest_r_dn_carts = init_r_dn_carts

        # print out structure info
        logger.info("Structure information:")
        self.__hamiltonian_data.structure_data.logger_info()
        logger.info("")

        # """
        # compiling methods
        # jax.profiler.start_trace("/tmp/tensorboard", create_perfetto_link=True)
        # open the generated URL (UI with perfetto)
        # tensorboard --logdir /tmp/tensorboard
        # tensorborad does not work with safari. use google chrome

        logger.info("Compilation starts.")

        logger.info("Compilation e_L starts.")
        start = time.perf_counter()
        _ = compute_kinetic_energy_api(
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        # """old
        _, _, _ = compute_discretized_kinetic_energy_api(
            alat=self.__alat,
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
            RT=jnp.eye(3, 3),
        )
        # """
        """ fast update with given A_old_inv, WIP
        # tentative for fast update, A_old, A_old_inv
        A_old = compute_geminal_all_elements_api(
            geminal_data=self.__hamiltonian_data.wavefunction_data.geminal_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        A_old_inv = jnp.linalg.inv(A_old)
        _, _, _ = compute_discretized_kinetic_energy_api_fast_update(
            alat=self.__alat,
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            A_old_inv=A_old_inv,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
            RT=jnp.eye(3, 3),
        )
        """
        _ = _compute_bare_coulomb_potential_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        _ = _compute_ecp_local_parts_full_NN_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        if self.__non_local_move == "tmove":
            _, _, _, _ = _compute_ecp_non_local_parts_NN_jax(
                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
                flag_determinant_only=False,
            )
        elif self.__non_local_move == "dltmove":
            _, _, _, _ = _compute_ecp_non_local_parts_NN_jax(
                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
                flag_determinant_only=True,
            )
        else:
            logger.error(f"non_local_move = {self.__non_local_move} is not yet implemented.")
            raise NotImplementedError

        end = time.perf_counter()
        logger.info("Compilation e_L is done.")
        logger.info(f"Elapsed Time = {end - start:.2f} sec.")

        logger.info("Compilation is done.")

        # jax.profiler.stop_trace()
        # """

    def run(self, num_branching: int = 50, max_time: int = 86400) -> None:
        """Run LRDMC.

        Args:
            num_branching (int): number of branching (reconfiguration of walkers).
            max_time (int): maximum time in sec.
        """
        # set timer
        timer_projection_total = 0.0
        timer_projection_non_diagonal_kinetic_part_init = 0.0
        timer_projection_non_diagonal_kinetic_part_comput = 0.0
        timer_projection_non_diagonal_kinetic_part_post = 0.0
        timer_projection_diag_kinetic_part_comput = 0.0
        timer_projection_diag_kinetic_part_post = 0.0
        timer_projection_diag_bare_couloumb_part_comput = 0.0
        timer_projection_diag_ecp_part_comput = 0.0
        timer_projection_non_diagonal_ecp_part_comput = 0.0
        timer_projection_non_diagonal_ecp_part_post = 0.0
        timer_projection_non_diagonal_probablity = 0.0
        timer_projection_comput_tau_update = 0.0
        timer_projection_update_weights_and_positions = 0.0
        timer_observable = 0.0
        timer_reconfiguration = 0.0
        timer_mpi_reduce = 0.0
        timer_mpi_comput = 0.0
        timer_mpi_bcast = 0.0
        counter_projection_times = 0
        gmfc_total_start = time.perf_counter()

        # initialize numpy random seed
        np.random.seed(self.__mpi_seed)

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

        # Main branching loop.
        logger.info("-Start branching-")
        gfmc_interval = int(np.maximum(num_branching / 100, 1))
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

            # MAIN project loop.
            logger.debug(f"  Projection time {self.__tau} a.u.^{-1}.")

            tau_left = self.__tau
            logger.debug(f"  Left projection time = {tau_left}/{self.__tau}: {0.0:.0f} %.")

            # Always set the initial weight to 1.0
            w_L = 1.0

            logger.debug("  Projection is on going....")

            start_projection = time.perf_counter()
            # projection loop
            projection_times = 0
            while True:
                projection_times += 1
                progress = (tau_left) / (self.__tau) * 100.0
                logger.debug(f"  Left projection time = {tau_left}/{self.__tau}: {progress:.1f} %.")

                # compute non-diagonal grids and elements (kinetic)
                start_projection_non_diagonal_kinetic_part_init = time.perf_counter()
                # generate a random rotation matrix
                self.__jax_PRNG_key, subkey = jax.random.split(self.__jax_PRNG_key)
                alpha, beta, gamma = jax.random.uniform(
                    subkey, shape=(3,), minval=-2 * jnp.pi, maxval=2 * jnp.pi
                )  # Rotation angle around the x,y,z-axis (in radians)

                # Compute individual rotation matrices
                R = generate_rotation_matrix(alpha, beta, gamma)
                R.block_until_ready()
                end_projection_non_diagonal_kinetic_part_init = time.perf_counter()
                timer_projection_non_diagonal_kinetic_part_init += (
                    end_projection_non_diagonal_kinetic_part_init - start_projection_non_diagonal_kinetic_part_init
                )

                start_projection_non_diagonal_kinetic_part_comput = time.perf_counter()
                # """ old
                mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                    compute_discretized_kinetic_energy_api(
                        alat=self.__alat,
                        wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                        r_up_carts=self.__latest_r_up_carts,
                        r_dn_carts=self.__latest_r_dn_carts,
                        RT=R.T,
                    )
                )
                # """
                """ fast update with given A_old_inv, WIP
                # tentative for fast update, A_old, A_old_inv
                A_old = compute_geminal_all_elements_api(
                    geminal_data=self.__hamiltonian_data.wavefunction_data.geminal_data,
                    r_up_carts=self.__latest_r_up_carts,
                    r_dn_carts=self.__latest_r_dn_carts,
                )
                A_old_inv = jnp.linalg.inv(A_old)
                A_old_inv.block_until_ready()
                start_projection_non_diagonal_kinetic_part_comput = time.perf_counter()
                mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                    compute_discretized_kinetic_energy_api_fast_update(
                        alat=self.__alat,
                        wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                        A_old_inv=A_old_inv,
                        r_up_carts=self.__latest_r_up_carts,
                        r_dn_carts=self.__latest_r_dn_carts,
                        RT=R.T,
                    )
                )
                """
                mesh_kinetic_part_r_up_carts.block_until_ready()
                mesh_kinetic_part_r_dn_carts.block_until_ready()
                elements_non_diagonal_kinetic_part.block_until_ready()
                end_projection_non_diagonal_kinetic_part_comput = time.perf_counter()
                timer_projection_non_diagonal_kinetic_part_comput += (
                    end_projection_non_diagonal_kinetic_part_comput - start_projection_non_diagonal_kinetic_part_comput
                )

                start_projection_non_diagonal_kinetic_part_post = time.perf_counter()
                elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
                diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
                non_diagonal_sum_hamiltonian = jnp.sum(elements_non_diagonal_kinetic_part_FN)
                elements_non_diagonal_kinetic_part_FN.block_until_ready()
                diagonal_kinetic_part_SP.block_until_ready()
                non_diagonal_sum_hamiltonian.block_until_ready()
                end_projection_non_diagonal_kinetic_part_post = time.perf_counter()
                timer_projection_non_diagonal_kinetic_part_post += (
                    end_projection_non_diagonal_kinetic_part_post - start_projection_non_diagonal_kinetic_part_post
                )

                # compute diagonal elements, kinetic part
                start_projection_diag_kinetic_part_comput = time.perf_counter()
                diagonal_kinetic_continuum = compute_kinetic_energy_api(
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=self.__latest_r_up_carts,
                    r_dn_carts=self.__latest_r_dn_carts,
                )
                diagonal_kinetic_continuum.block_until_ready()
                end_projection_diag_kinetic_part_comput = time.perf_counter()
                timer_projection_diag_kinetic_part_comput += (
                    end_projection_diag_kinetic_part_comput - start_projection_diag_kinetic_part_comput
                )

                start_projection_diag_kinetic_part_post = time.perf_counter()
                diagonal_kinetic_discretized = -1.0 * jnp.sum(elements_non_diagonal_kinetic_part)
                diagonal_kinetic_discretized.block_until_ready()
                end_projection_diag_kinetic_part_post = time.perf_counter()
                timer_projection_diag_kinetic_part_post += (
                    end_projection_diag_kinetic_part_post - start_projection_diag_kinetic_part_post
                )

                # compute diagonal elements, bare couloumb
                start_projection_diag_bare_couloumb_comput = time.perf_counter()
                diagonal_bare_coulomb_part = _compute_bare_coulomb_potential_jax(
                    coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                    r_up_carts=self.__latest_r_up_carts,
                    r_dn_carts=self.__latest_r_dn_carts,
                )
                diagonal_bare_coulomb_part.block_until_ready()
                end_projection_diag_bare_couloumb_comput = time.perf_counter()
                timer_projection_diag_bare_couloumb_part_comput += (
                    end_projection_diag_bare_couloumb_comput - start_projection_diag_bare_couloumb_comput
                )

                # with ECP
                if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
                    # ecp local
                    start_projection_diag_ecp_comput = time.perf_counter()
                    diagonal_ecp_local_part_comput = _compute_ecp_local_parts_full_NN_jax(
                        coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                        r_up_carts=self.__latest_r_up_carts,
                        r_dn_carts=self.__latest_r_dn_carts,
                    )
                    diagonal_ecp_local_part_comput.block_until_ready()
                    end_projection_diag_ecp_comput = time.perf_counter()
                    timer_projection_diag_ecp_part_comput += end_projection_diag_ecp_comput - start_projection_diag_ecp_comput

                    # ecp non-local
                    if self.__non_local_move == "tmove":
                        start_projection_non_diagonal_ecp_part_comput = time.perf_counter()
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            _compute_ecp_non_local_parts_NN_jax(
                                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                                r_up_carts=self.__latest_r_up_carts,
                                r_dn_carts=self.__latest_r_dn_carts,
                                flag_determinant_only=False,
                            )
                        )
                        mesh_non_local_ecp_part_r_up_carts.block_until_ready()
                        mesh_non_local_ecp_part_r_dn_carts.block_until_ready()
                        V_nonlocal.block_until_ready()
                        end_projection_non_diagonal_ecp_part_comput = time.perf_counter()
                        timer_projection_non_diagonal_ecp_part_comput += (
                            end_projection_non_diagonal_ecp_part_comput - start_projection_non_diagonal_ecp_part_comput
                        )

                        start_projection_non_diagonal_ecp_part_post = time.perf_counter()
                        V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                        diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                        non_diagonal_sum_hamiltonian += jnp.sum(V_nonlocal_FN)
                        V_nonlocal_FN.block_until_ready()
                        diagonal_ecp_part_SP.block_until_ready()
                        non_diagonal_sum_hamiltonian.block_until_ready()
                        end_projection_non_diagonal_ecp_part_post = time.perf_counter()
                        timer_projection_non_diagonal_ecp_part_post += (
                            end_projection_non_diagonal_ecp_part_post - start_projection_non_diagonal_ecp_part_post
                        )

                    elif self.__non_local_move == "dltmove":
                        start_projection_non_diagonal_ecp_part_comput = time.perf_counter()
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            _compute_ecp_non_local_parts_NN_jax(
                                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                                r_up_carts=self.__latest_r_up_carts,
                                r_dn_carts=self.__latest_r_dn_carts,
                                flag_determinant_only=True,
                            )
                        )
                        mesh_non_local_ecp_part_r_up_carts.block_until_ready()
                        mesh_non_local_ecp_part_r_dn_carts.block_until_ready()
                        V_nonlocal.block_until_ready()
                        end_projection_non_diagonal_ecp_part_comput = time.perf_counter()
                        timer_projection_non_diagonal_ecp_part_comput += (
                            end_projection_non_diagonal_ecp_part_comput - start_projection_non_diagonal_ecp_part_comput
                        )

                        start_projection_non_diagonal_ecp_part_post = time.perf_counter()
                        V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                        diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))

                        Jastrow_ratio = compute_ratio_Jastrow_part_api(
                            jastrow_data=self.__hamiltonian_data.wavefunction_data.jastrow_data,
                            old_r_up_carts=self.__latest_r_up_carts,
                            old_r_dn_carts=self.__latest_r_dn_carts,
                            new_r_up_carts_arr=mesh_non_local_ecp_part_r_up_carts,
                            new_r_dn_carts_arr=mesh_non_local_ecp_part_r_dn_carts,
                        )
                        V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

                        non_diagonal_sum_hamiltonian += jnp.sum(V_nonlocal_FN)

                        V_nonlocal_FN.block_until_ready()
                        diagonal_ecp_part_SP.block_until_ready()
                        non_diagonal_sum_hamiltonian.block_until_ready()
                        end_projection_non_diagonal_ecp_part_post = time.perf_counter()
                        timer_projection_non_diagonal_ecp_part_post += (
                            end_projection_non_diagonal_ecp_part_post - start_projection_non_diagonal_ecp_part_post
                        )

                    else:
                        logger.error(f"non_local_move = {self.__non_local_move} is not yet implemented.")
                        raise NotImplementedError

                    # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                    start_projection_non_diagonal_probablity = time.perf_counter()
                    e_L = (
                        diagonal_kinetic_continuum
                        + diagonal_kinetic_discretized
                        + diagonal_bare_coulomb_part
                        + diagonal_ecp_local_part_comput
                        + diagonal_kinetic_part_SP
                        + diagonal_ecp_part_SP
                        + non_diagonal_sum_hamiltonian
                    )

                    p_list = jnp.concatenate([jnp.ravel(elements_non_diagonal_kinetic_part_FN), jnp.ravel(V_nonlocal_FN)])
                    non_diagonal_move_probabilities = p_list / p_list.sum()
                    non_diagonal_move_mesh_r_up_carts = jnp.concatenate(
                        [mesh_kinetic_part_r_up_carts, mesh_non_local_ecp_part_r_up_carts], axis=0
                    )
                    non_diagonal_move_mesh_r_dn_carts = jnp.concatenate(
                        [mesh_kinetic_part_r_dn_carts, mesh_non_local_ecp_part_r_dn_carts], axis=0
                    )
                    e_L.block_until_ready()
                    p_list.block_until_ready()
                    non_diagonal_move_probabilities.block_until_ready()
                    non_diagonal_move_mesh_r_up_carts.block_until_ready()
                    non_diagonal_move_mesh_r_dn_carts.block_until_ready()
                    end_projection_non_diagonal_probablity = time.perf_counter()
                    timer_projection_non_diagonal_probablity += (
                        end_projection_non_diagonal_probablity - start_projection_non_diagonal_probablity
                    )

                # with all electrons
                else:
                    start_projection_non_diagonal_probablity = time.perf_counter()
                    # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                    e_L = (
                        diagonal_kinetic_continuum
                        + diagonal_kinetic_discretized
                        + diagonal_bare_coulomb_part
                        + diagonal_kinetic_part_SP
                        + non_diagonal_sum_hamiltonian
                    )

                    p_list = jnp.ravel(elements_non_diagonal_kinetic_part_FN)
                    non_diagonal_move_probabilities = p_list / p_list.sum()
                    non_diagonal_move_mesh_r_up_carts = mesh_kinetic_part_r_up_carts
                    non_diagonal_move_mesh_r_dn_carts = mesh_kinetic_part_r_dn_carts
                    e_L.block_until_ready()
                    p_list.block_until_ready()
                    non_diagonal_move_probabilities.block_until_ready()
                    non_diagonal_move_mesh_r_up_carts.block_until_ready()
                    non_diagonal_move_mesh_r_dn_carts.block_until_ready()
                    end_projection_non_diagonal_probablity = time.perf_counter()
                    timer_projection_non_diagonal_probablity += (
                        end_projection_non_diagonal_probablity - start_projection_non_diagonal_probablity
                    )

                # update weight and positions
                start_projection_comput_tau_update = time.perf_counter()
                xi = np.random.random()
                # compute the time the walker remaining in the same configuration
                tau_update = jnp.minimum(tau_left, jnp.log(1 - xi) / non_diagonal_sum_hamiltonian)
                tau_update.block_until_ready()
                end_projection_comput_tau_update = time.perf_counter()
                timer_projection_comput_tau_update += end_projection_comput_tau_update - start_projection_comput_tau_update

                # update weight and positions
                start_projection_update_weights_and_positions = time.perf_counter()
                w_L = w_L * jnp.exp(-tau_update * e_L)
                logger.debug(f"  w_L={w_L}")

                # update tau_left
                tau_left = tau_left - tau_update

                # random choice and update electron positions
                if tau_left > 0.0:
                    k = np.random.choice(len(p_list), p=non_diagonal_move_probabilities)
                    self.__latest_r_up_carts = non_diagonal_move_mesh_r_up_carts[k]
                    self.__latest_r_dn_carts = non_diagonal_move_mesh_r_dn_carts[k]
                else:
                    self.__latest_r_up_carts = self.__latest_r_up_carts
                    self.__latest_r_dn_carts = self.__latest_r_dn_carts

                w_L.block_until_ready()
                tau_left.block_until_ready()
                self.__latest_r_up_carts.block_until_ready()
                self.__latest_r_dn_carts.block_until_ready()
                end_projection_update_weights_and_positions = time.perf_counter()
                timer_projection_update_weights_and_positions += (
                    end_projection_update_weights_and_positions - start_projection_update_weights_and_positions
                )

                # if tau_left becomes < 0, break the loop (i.e., proceed with the branching step.)
                if tau_left <= 0.0:
                    logger.debug("tau_left = {tau_left} <= 0.0. Exit the projection loop.")
                    break

            end_projection = time.perf_counter()
            timer_projection_total += end_projection - start_projection
            logger.debug(f"  #projection times = x {projection_times}")
            counter_projection_times += projection_times

            # projection ends
            logger.debug("  Projection ends.")

            # evaluate observables
            start_observable = time.perf_counter()
            # e_L evaluation is not necesarily repeated here.
            # to be implemented other observables, such as derivatives.
            end_observable = time.perf_counter()
            timer_observable += end_observable - start_observable

            # Reconfigurations!
            start_reconfiguration = time.perf_counter()

            # jnp.float -> np.array
            w_L_latest = float(w_L)
            e_L_latest = float(e_L)

            # jnp.array -> np.array
            self.__latest_r_up_carts = np.array(self.__latest_r_up_carts)
            self.__latest_r_dn_carts = np.array(self.__latest_r_dn_carts)

            # MPI reduce
            start_reduce = time.perf_counter()
            e_L_gathered_dyad = (mpi_rank, e_L_latest)
            e_L_gathered_dyad = mpi_comm.gather(e_L_gathered_dyad, root=0)
            w_L_gathered_dyad = (mpi_rank, w_L_latest)
            w_L_gathered_dyad = mpi_comm.gather(w_L_gathered_dyad, root=0)
            end_reduce = time.perf_counter()
            timer_mpi_reduce += end_reduce - start_reduce

            if mpi_rank == 0:
                start_comput = time.perf_counter()
                logger.debug(f"e_L_gathered_dyad={e_L_gathered_dyad}")
                logger.debug(f"w_L_gathered_dyad={w_L_gathered_dyad}")
                e_L_gathered = np.array([e_L for _, e_L in e_L_gathered_dyad])
                w_L_gathered = np.array([w_L for _, w_L in w_L_gathered_dyad])
                e_L_averaged = np.sum(w_L_gathered * e_L_gathered) / np.sum(w_L_gathered)
                w_L_averaged = np.average(w_L_gathered)
                logger.debug(f"  e_L_averaged = {e_L_averaged} Ha")
                logger.debug(f"  w_L_averaged(before branching) = {w_L_averaged}")
                self.__e_L_averaged_list.append(e_L_averaged)
                self.__w_L_averaged_list.append(w_L_averaged)
                mpi_rank_list = [r for r, _ in w_L_gathered_dyad]
                w_L_list = np.array([w_L for _, w_L in w_L_gathered_dyad])
                logger.debug(f"w_L_list = {w_L_list}")
                probabilities = w_L_list / w_L_list.sum()
                logger.debug(f"probabilities = {probabilities}")

                # correlated choice (see Sandro's textbook, page 182)
                zeta = float(np.random.random())
                z_list = [(alpha + zeta) / len(probabilities) for alpha in range(len(probabilities))]
                cumulative_prob = np.cumsum(probabilities)
                k_list = np.array([next(idx for idx, prob in enumerate(cumulative_prob) if z <= prob) for z in z_list])
                logger.debug(f"The chosen walker indices = {k_list}")

                chosen_rank_list = [w_L_gathered_dyad[k][0] for k in k_list]
                chosen_rank_list.sort()
                logger.debug(f"chosen_rank_list = {chosen_rank_list}")
                counter = Counter(chosen_rank_list)
                self.__num_survived_walkers += len(set(chosen_rank_list))
                self.__num_killed_walkers += len(mpi_rank_list) - len(set(chosen_rank_list))
                logger.debug(f"num_survived_walkers={self.__num_survived_walkers}")
                logger.debug(f"num_killed_walkers={self.__num_killed_walkers}")
                mpi_send_rank = [item for item, count in counter.items() for _ in range(count - 1) if count > 1]
                mpi_recv_rank = list(set(mpi_rank_list) - set(chosen_rank_list))
                logger.debug(f"mpi_send_rank={mpi_send_rank}")
                logger.debug(f"mpi_recv_rank={mpi_recv_rank}")
                end_comput = time.perf_counter()
                timer_mpi_comput += end_comput - start_comput
            else:
                mpi_send_rank = None
                mpi_recv_rank = None
                self.__e_L_averaged_list = None
                self.__w_L_averaged_list = None

            start_bcast = time.perf_counter()
            mpi_send_rank = mpi_comm.bcast(mpi_send_rank, root=0)
            mpi_recv_rank = mpi_comm.bcast(mpi_recv_rank, root=0)
            self.__e_L_averaged_list = mpi_comm.bcast(self.__e_L_averaged_list, root=0)
            self.__w_L_averaged_list = mpi_comm.bcast(self.__w_L_averaged_list, root=0)

            # logger.debug(f"Before branching: rank={mpi_rank}:gfmc.r_up_carts = {self.__latest_r_up_carts}")
            # logger.debug(f"Before branching: rank={mpi_rank}:gfmc.r_dn_carts = {self.__latest_r_dn_carts}")
            # mpi_comm.barrier()
            self.__num_survived_walkers = mpi_comm.bcast(self.__num_survived_walkers, root=0)
            self.__num_killed_walkers = mpi_comm.bcast(self.__num_killed_walkers, root=0)
            for ii, (send_rank, recv_rank) in enumerate(zip(mpi_send_rank, mpi_recv_rank)):
                if mpi_rank == send_rank:
                    mpi_comm.send(self.__latest_r_up_carts, dest=recv_rank, tag=100 + 2 * ii)
                    mpi_comm.send(self.__latest_r_dn_carts, dest=recv_rank, tag=100 + 2 * ii + 1)
                if mpi_rank == recv_rank:
                    self.__latest_r_up_carts = mpi_comm.recv(source=send_rank, tag=100 + 2 * ii)
                    self.__latest_r_dn_carts = mpi_comm.recv(source=send_rank, tag=100 + 2 * ii + 1)
            end_bcast = time.perf_counter()
            timer_mpi_bcast += end_bcast - start_bcast
            # mpi_comm.barrier()
            # logger.debug(f"*After branching: rank={mpi_rank}:gfmc.r_up_carts = {self.__latest_r_up_carts}")
            # logger.debug(f"*After branching: rank={mpi_rank}:gfmc.r_dn_carts = {self.__latest_r_dn_carts}")
            # mpi_comm.barrier()

            # np.array -> jnp.array
            self.__latest_r_up_carts = jnp.array(self.__latest_r_up_carts)
            self.__latest_r_dn_carts = jnp.array(self.__latest_r_dn_carts)

            num_branching_done += 1

            end_reconfiguration = time.perf_counter()
            timer_reconfiguration += end_reconfiguration - start_reconfiguration

            gmfc_current = time.perf_counter()
            if max_time < gmfc_current - gmfc_total_start:
                logger.info(f"  Max_time = {max_time} sec. exceeds.")
                logger.info("  Break the branching loop.")
                break

        logger.info("-End branching-")

        # count up
        self.__gfmc_branching_counter += i_branching + 1

        gmfc_total_end = time.perf_counter()
        timer_gmfc_total = gmfc_total_end - gmfc_total_start

        logger.info(f"Total GFMC time for {num_branching_done} branching steps = {timer_gmfc_total: .3f} sec.")
        logger.info(f"Net GFMC time for {num_branching_done} branching steps = {timer_gmfc_total: .3f} sec.")
        logger.info(f"Elapsed times per branching, averaged over {num_branching_done} branching steps.")
        logger.info(f"  Projection time per branching = {timer_projection_total / num_branching_done * 10**3: .3f} msec.")
        logger.info(f"  Projection iterations per branching ={counter_projection_times / num_branching_done: .3f} times.")
        logger.info(
            f"    - Non_diagonal kinetic part (init) = {timer_projection_non_diagonal_kinetic_part_init / num_branching_done * 10**3: .3f} msec."
        )
        logger.info(
            f"    - Non_diagonal kinetic part (comput) = {timer_projection_non_diagonal_kinetic_part_comput / num_branching_done * 10**3: .3f} msec."
        )
        logger.info(
            f"    - Non_diagonal kinetic part (post) = {timer_projection_non_diagonal_kinetic_part_post / num_branching_done * 10**3: .3f} msec."
        )
        logger.info(
            f"    - Diagonal kinetic part (comput) = {timer_projection_diag_kinetic_part_comput / num_branching_done * 10**3: .3f} msec."
        )
        logger.info(
            f"    - Diagonal kinetic part (post) = {timer_projection_diag_kinetic_part_post / num_branching_done * 10**3: .3f} msec."
        )
        logger.info(
            f"    - Diagonal ecp part (comput) = {timer_projection_diag_ecp_part_comput / num_branching_done * 10**3: .3f} msec."
        )
        logger.info(
            f"    - Diagonal bare coulomb part (comput) = {timer_projection_diag_bare_couloumb_part_comput / num_branching_done * 10**3: .3f} msec."
        )
        logger.info(
            f"    - Non_diagonal ecp part (comput) = {timer_projection_non_diagonal_ecp_part_comput / num_branching_done * 10**3: .3f} msec."
        )
        logger.info(
            f"    - Non_diagonal ecp part (post) = {timer_projection_non_diagonal_ecp_part_post / num_branching_done * 10**3: .3f} msec."
        )
        logger.info(
            f"    - Non_diagonal probablity part (post) = {timer_projection_non_diagonal_probablity / num_branching_done * 10**3: .3f} msec."
        )
        logger.info(f"    - Comput. tau_update = {timer_projection_comput_tau_update / num_branching_done * 10**3: .3f} msec.")
        logger.info(
            f"    - Update weights and positions = {timer_projection_update_weights_and_positions / num_branching_done * 10**3: .3f} msec."
        )
        logger.info(f"  Observable measurement time per branching = {timer_observable / num_branching_done * 10**3: .3f} msec.")
        logger.info(
            f"  Walker reconfiguration time per branching = {timer_reconfiguration / num_branching_done * 10**3: .3f} msec."
        )
        logger.info(f"    - MPI reduce time per reconfiguration = {timer_mpi_reduce / num_branching_done * 10**3: .3f} msec.")
        logger.info(f"    - Comput time per reconfiguration = {timer_mpi_comput / num_branching_done * 10**3: .3f} msec.")
        logger.info(f"    - MPI bcast time per reconfiguration = {timer_mpi_bcast / num_branching_done * 10**3: .3f} msec.")
        logger.info(
            f"Survived walkers ratio = {self.__num_survived_walkers / (self.__num_survived_walkers + self.__num_killed_walkers) * 100:.2f} %"
        )

    def get_e_L(self, num_gfmc_warmup_steps: int = 3, num_gfmc_bin_blocks: int = 10, num_gfmc_bin_collect: int = 2) -> float:
        """Get e_L."""
        logger.info("- Comput. e_L -")
        if mpi_rank == 0:
            e_L_eq = self.__e_L_averaged_list[num_gfmc_warmup_steps + num_gfmc_bin_collect :]
            w_L_eq = self.__w_L_averaged_list[num_gfmc_warmup_steps:]
            logger.debug(f"e_L_eq = {e_L_eq}")
            logger.debug(f"w_L_eq = {w_L_eq}")
            logger.info("  Progress: Computing G_eq and G_e_L_eq.")
            G_eq = [
                np.prod([w_L_eq[n - j] for j in range(1, num_gfmc_bin_collect + 1)])
                for n in range(num_gfmc_bin_collect, len(w_L_eq))
            ]
            logger.debug(f"G_eq = {G_eq}")
            logger.debug(f"len(e_L_eq) = {len(e_L_eq)}")
            logger.debug(f"len(G_eq) = {len(G_eq)}")

            e_L_eq = np.array(e_L_eq)
            G_eq = np.array(G_eq)

            logger.info("  Comput. G_e_L_eq.")
            G_e_L_eq = e_L_eq * G_eq

            logger.info(f"  Progress: Computing binned G_e_L_eq and G_eq with # binned blocks = {num_gfmc_bin_blocks}.")
            G_e_L_split = np.array_split(G_e_L_eq, num_gfmc_bin_blocks)
            G_e_L_binned = np.array([np.average(G_e_L_list) for G_e_L_list in G_e_L_split])
            G_split = np.array_split(G_eq, num_gfmc_bin_blocks)
            G_binned = np.array([np.average(G_list) for G_list in G_split])

            logger.info(f"  Progress: Computing jackknife samples with # binned blocks = {num_gfmc_bin_blocks}.")

            G_e_L_binned_sum = np.sum(G_e_L_binned)
            G_binned_sum = np.sum(G_binned)

            e_L_jackknife = [
                (G_e_L_binned_sum - G_e_L_binned[m]) / (G_binned_sum - G_binned[m]) for m in range(num_gfmc_bin_blocks)
            ]

            logger.info("  Progress: Computing jackknife mean and std.")
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
    def latest_r_up_carts(self) -> npt.NDArray:
        """Latest updated electron position for up-spin."""
        return self.__latest_r_up_carts

    @property
    def latest_r_dn_carts(self) -> npt.NDArray:
        """Latest updated electron position for down-spin."""
        return self.__latest_r_dn_carts


if __name__ == "__main__":
    import pickle

    # import os
    # from .jastrow_factor import Jastrow_data, Jastrow_three_body_data, Jastrow_two_body_data
    # from .trexio_wrapper import read_trexio_file
    # from .wavefunction import Wavefunction_data, compute_discretized_kinetic_energy_api, evaluate_jastrow_api

    logger_level = "MPI-INFO"
    # logger_level = "INFO"

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

    # global JAX device
    global_device_info = jax.devices()
    # local JAX device
    num_devices = jax.local_devices()
    device_info_str = f"Rank {mpi_rank}: {num_devices}"
    local_device_info = mpi_comm.allgather(device_info_str)
    # print recognized XLA devices
    logger.info("*** XLA Global devices recognized by JAX***")
    logger.info(global_device_info)
    logger.info("*** XLA Local devices recognized by JAX***")
    logger.info(local_device_info)
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
    mcmc_seed = 3446
    max_time = 86400
    tau = 0.10
    alat = 0.30
    num_branching = 20
    non_local_move = "tmove"

    num_gfmc_warmup_steps = 5
    num_gfmc_bin_blocks = 5
    num_gfmc_bin_collect = 3

    # run GFMC
    gfmc = GFMC(hamiltonian_data=hamiltonian_data, mcmc_seed=mcmc_seed, tau=tau, alat=alat, non_local_move=non_local_move)
    gfmc.run(num_branching=num_branching, max_time=max_time)
    """
    e_L_mean, e_L_std = gfmc.get_e_L(
        num_gfmc_warmup_steps=num_gfmc_warmup_steps,
        num_gfmc_bin_blocks=num_gfmc_bin_blocks,
        num_gfmc_bin_collect=num_gfmc_bin_collect,
    )
    logger.info(f"e_L = {e_L_mean} +- {e_L_std} Ha")
    """
