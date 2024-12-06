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

# python modules
import os
import random
import time
from collections import Counter
from logging import Formatter, StreamHandler, getLogger

# import mpi4jax
# JAX
import jax
import numpy as np
import numpy.typing as npt

# MPI
from mpi4py import MPI

# jQMC module
from .coulomb_potential import (
    _compute_bare_coulomb_potential_jax,
    _compute_ecp_local_parts_jax,
    _compute_ecp_non_local_parts_jax,
)
from .hamiltonians import Hamiltonian_data, compute_kinetic_energy_api, compute_local_energy
from .jastrow_factor import Jastrow_data, Jastrow_three_body_data, Jastrow_two_body_data
from .trexio_wrapper import read_trexio_file
from .wavefunction import Wavefunction_data, compute_discretized_kinetic_energy_jax, evaluate_jastrow_api

# MPI related
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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
        self.__mcmc_seed = mcmc_seed
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
        mpi_seed = mcmc_seed * (rank + 1)
        random.seed(mpi_seed)
        np.random.seed(mpi_seed)

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

        logger.debug(f"initial r_up_carts = {init_r_up_carts}")
        logger.debug(f"initial r_dn_carts = {init_r_dn_carts}")

        self.__latest_r_up_carts = init_r_up_carts
        self.__latest_r_dn_carts = init_r_dn_carts

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
        jax_PRNG_key = jax.random.PRNGKey(self.__mcmc_seed)
        _, _, _ = compute_discretized_kinetic_energy_jax(
            alat=self.__alat,
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
            jax_PRNG_key=jax_PRNG_key,
        )
        _ = _compute_bare_coulomb_potential_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        _ = _compute_ecp_local_parts_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        _, _, _ = _compute_ecp_non_local_parts_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        end = time.perf_counter()
        logger.info("Compilation e_L is done.")
        logger.info(f"Elapsed Time = {end-start:.2f} sec.")

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
        timer_projection_non_diagonal_kinetic_part = 0.0
        timer_projection_non_diagonal_ecp_part = 0.0
        timer_projection_diag_kinetic_part = 0.0
        timer_projection_diag_bare_couloumb_part = 0.0
        timer_projection_diag_ecp_part = 0.0
        timer_projection_update_weights_and_positions = 0.0
        timer_observable = 0.0
        timer_branching = 0.0
        gmfc_total_start = time.perf_counter()

        # set jax PRNG key for pseudo random number generators of JAX.
        jax_PRNG_key = jax.random.PRNGKey(self.__mcmc_seed)

        # Main branching loop.
        progress = (self.__gfmc_branching_counter) / (num_branching + self.__gfmc_branching_counter) * 100.0
        logger.info(
            f"Current branching step = {self.__gfmc_branching_counter}/{num_branching+self.__gfmc_branching_counter}: {progress:.0f} %."
        )
        logger.info("-Start branching-")
        gfmc_interval = int(num_branching / 10)  # %
        for i_branching in range(num_branching):
            if (i_branching + 1) % gfmc_interval == 0:
                progress = (
                    (i_branching + self.__gfmc_branching_counter + 1) / (num_branching + self.__gfmc_branching_counter) * 100.0
                )
                logger.info(
                    f"  Progress: branching step = {i_branching + self.__gfmc_branching_counter + 1}/{num_branching+self.__gfmc_branching_counter}: {progress:.1f} %."
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
            while True:
                progress = (tau_left) / (self.__tau) * 100.0
                logger.debug(f"  Left projection time = {tau_left}/{self.__tau}: {progress:.1f} %.")

                # compute non-diagonal grids and elements (kinetic)
                start_projection_non_diagonal_kinetic_part = time.perf_counter()
                mesh_kinetic_part, elements_non_diagonal_kinetic_part, jax_PRNG_key = compute_discretized_kinetic_energy_jax(
                    alat=self.__alat,
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=self.__latest_r_up_carts,
                    r_dn_carts=self.__latest_r_dn_carts,
                    jax_PRNG_key=jax_PRNG_key,
                )
                elements_non_diagonal_kinetic_part_FN = list(
                    map(lambda K: K if K < 0 else 0.0, elements_non_diagonal_kinetic_part)
                )
                diagonal_kinetic_part_SP = np.sum(list(map(lambda K: K if K >= 0 else 0.0, elements_non_diagonal_kinetic_part)))
                non_diagonal_sum_hamiltonian = np.sum(elements_non_diagonal_kinetic_part_FN)
                end_projection_non_diagonal_kinetic_part = time.perf_counter()
                timer_projection_non_diagonal_kinetic_part += (
                    end_projection_non_diagonal_kinetic_part - start_projection_non_diagonal_kinetic_part
                )

                # compute diagonal elements, kinetic part
                start_projection_diag_kinetic_part = time.perf_counter()
                diagonal_kinetic_continuum = compute_kinetic_energy_api(
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=self.__latest_r_up_carts,
                    r_dn_carts=self.__latest_r_dn_carts,
                )
                diagonal_kinetic_discretized = -1.0 * np.sum(elements_non_diagonal_kinetic_part)
                end_projection_diag_kinetic_part = time.perf_counter()
                timer_projection_diag_kinetic_part += end_projection_diag_kinetic_part - start_projection_diag_kinetic_part

                # compute diagonal elements, bare couloumb
                start_projection_diag_bare_couloumb = time.perf_counter()
                diagonal_bare_coulomb_part = _compute_bare_coulomb_potential_jax(
                    coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                    r_up_carts=self.__latest_r_up_carts,
                    r_dn_carts=self.__latest_r_dn_carts,
                )
                end_projection_diag_bare_couloumb = time.perf_counter()
                timer_projection_diag_bare_couloumb_part += (
                    end_projection_diag_bare_couloumb - start_projection_diag_bare_couloumb
                )

                # with ECP
                if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
                    # ecp local
                    start_projection_diag_ecp = time.perf_counter()
                    diagonal_ecp_local_part = _compute_ecp_local_parts_jax(
                        coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                        r_up_carts=self.__latest_r_up_carts,
                        r_dn_carts=self.__latest_r_dn_carts,
                    )
                    end_projection_diag_ecp = time.perf_counter()
                    timer_projection_diag_ecp_part += end_projection_diag_ecp - start_projection_diag_ecp

                    # ecp non-local
                    start_projection_non_diagonal_ecp_part = time.perf_counter()
                    if self.__non_local_move == "tmove":
                        mesh_non_local_ecp_part, V_nonlocal, _ = _compute_ecp_non_local_parts_jax(
                            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                            r_up_carts=self.__latest_r_up_carts,
                            r_dn_carts=self.__latest_r_dn_carts,
                            flag_determinant_only=False,
                        )

                        V_nonlocal_FN = list(map(lambda V: V if V < 0.0 else 0.0, V_nonlocal))
                        diagonal_ecp_part_SP = np.sum(list(map(lambda V: V if V >= 0.0 else 0.0, V_nonlocal)))
                        non_diagonal_sum_hamiltonian += np.sum(V_nonlocal_FN)

                    elif self.__non_local_move == "dltmove":
                        mesh_non_local_ecp_part, V_nonlocal, _ = _compute_ecp_non_local_parts_jax(
                            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                            r_up_carts=self.__latest_r_up_carts,
                            r_dn_carts=self.__latest_r_dn_carts,
                            flag_determinant_only=True,
                        )

                        V_nonlocal_FN = list(map(lambda V: V if V < 0.0 else 0.0, V_nonlocal))
                        diagonal_ecp_part_SP = np.sum(list(map(lambda V: V if V >= 0.0 else 0.0, V_nonlocal)))

                        Jastrow_ref = evaluate_jastrow_api(
                            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                            r_up_carts=self.__latest_r_up_carts,
                            r_dn_carts=self.__latest_r_dn_carts,
                        )
                        V_nonlocal_FN = [
                            V
                            * evaluate_jastrow_api(
                                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                                r_up_carts=mesh_non_local_ecp_part[i][0],
                                r_dn_carts=mesh_non_local_ecp_part[i][1],
                            )
                            / Jastrow_ref
                            if V < 0.0
                            else 0.0
                            for i, V in enumerate(V_nonlocal_FN)
                        ]
                        non_diagonal_sum_hamiltonian += np.sum(V_nonlocal_FN)

                        """ for debug / to be deleted soon.
                        _, V_nonlocal_debug, _ = compute_ecp_non_local_parts_jax(
                            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                            r_up_carts=self.__latest_r_up_carts,
                            r_dn_carts=self.__latest_r_dn_carts,
                            flag_determinant_only=False,
                        )

                        V_nonlocal_FN_debug = list(map(lambda V: V if V < 0.0 else 0.0, V_nonlocal_debug))

                        logger.info(
                            f"np.max(np.array(V_nonlocal_FN_debug) - np.array(V_nonlocal_FN)) = {np.max(np.array(V_nonlocal_FN_debug) - np.array(V_nonlocal_FN))}"
                        )
                        """

                    else:
                        logger.error(f"non_local_move = {self.__non_local_move} is not yet implemented.")
                        raise NotImplementedError

                    end_projection_non_diagonal_ecp_part = time.perf_counter()
                    timer_projection_non_diagonal_ecp_part += (
                        end_projection_non_diagonal_ecp_part - start_projection_non_diagonal_ecp_part
                    )

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

                logger.debug(f"  e_L={e_L}")

                # compute the time the walker remaining in the same configuration
                start_projection_update_weights_and_positions = time.perf_counter()
                xi = random.random()
                tau_update = np.min((tau_left, np.log(1 - xi) / non_diagonal_sum_hamiltonian))
                logger.debug(f"  tau_update={tau_update}")

                # update weight
                w_L = w_L * np.exp(-tau_update * e_L)
                logger.debug(f"  w_L={w_L}")

                # update tau_left
                tau_left = tau_left - tau_update

                # if tau_left becomes < 0, break the loop (i.e., proceed with the branching step.)
                if tau_left <= 0.0:
                    logger.debug("tau_left = {tau_left} <= 0.0. Exit the projection loop.")
                    break

                # choose a non-diagonal move destination
                if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
                    p_list = np.array(elements_non_diagonal_kinetic_part_FN + V_nonlocal_FN)
                else:
                    p_list = np.array(elements_non_diagonal_kinetic_part_FN)
                probabilities = p_list / p_list.sum()
                logger.debug(f"len(probabilities) = {len(probabilities)}")

                # random choice
                logger.debug(f"self.__latest_r_up_carts = {self.__latest_r_up_carts}")
                logger.debug(f"self.__latest_r_dn_carts = {self.__latest_r_dn_carts}")
                k = np.random.choice(len(p_list), p=probabilities)
                logger.debug(f"chosen update electron index = {k}.")
                # update electron position
                if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
                    self.__latest_r_up_carts, self.__latest_r_dn_carts = (
                        list(mesh_kinetic_part) + list(mesh_non_local_ecp_part)
                    )[k]
                else:
                    self.__latest_r_up_carts, self.__latest_r_dn_carts = (list(mesh_kinetic_part))[k]
                logger.debug(f"self.__latest_r_up_carts = {self.__latest_r_up_carts}")
                logger.debug(f"self.__latest_r_dn_carts = {self.__latest_r_dn_carts}")
                end_projection_update_weights_and_positions = time.perf_counter()
                timer_projection_update_weights_and_positions += (
                    end_projection_update_weights_and_positions - start_projection_update_weights_and_positions
                )

            end_projection = time.perf_counter()
            timer_projection_total += end_projection - start_projection

            # projection ends
            logger.debug("  Projection ends.")

            # evaluate observables
            start_observable = time.perf_counter()
            # e_L evaluation is not necesarily repeated here.
            # to be implemented other observables, such as derivatives.
            end_observable = time.perf_counter()
            timer_observable += end_observable - start_observable

            logger.devel(f"  e_L = {e_L}")
            logger.devel(f"  w_L = {w_L}")
            w_L_latest = float(w_L)
            e_L_latest = float(e_L)

            # Branching!
            start_branching = time.perf_counter()

            logger.debug(f"e_L={e_L_latest} for rank={rank}")
            logger.debug(f"w_L={w_L_latest} for rank={rank}")

            e_L_gathered_dyad = (rank, e_L_latest)
            e_L_gathered_dyad = comm.gather(e_L_gathered_dyad, root=0)
            w_L_gathered_dyad = (rank, w_L_latest)
            w_L_gathered_dyad = comm.gather(w_L_gathered_dyad, root=0)

            if rank == 0:
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

                """
                # random choice
                k_list = np.random.choice(len(w_L_list), size=len(w_L_list), p=probabilities, replace=True)
                """

                # correlated choice (see Sandro's textbook, page 182)
                zeta = np.random.random()
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
            else:
                mpi_send_rank = None
                mpi_recv_rank = None
                self.__e_L_averaged_list = None
                self.__w_L_averaged_list = None
            mpi_send_rank = comm.bcast(mpi_send_rank, root=0)
            mpi_recv_rank = comm.bcast(mpi_recv_rank, root=0)
            self.__e_L_averaged_list = comm.bcast(self.__e_L_averaged_list, root=0)
            self.__w_L_averaged_list = comm.bcast(self.__w_L_averaged_list, root=0)

            logger.debug(f"Before branching: rank={rank}:gfmc.r_up_carts = {self.__latest_r_up_carts}")
            logger.debug(f"Before branching: rank={rank}:gfmc.r_dn_carts = {self.__latest_r_dn_carts}")

            comm.barrier()
            self.__num_survived_walkers = comm.bcast(self.__num_survived_walkers, root=0)
            self.__num_killed_walkers = comm.bcast(self.__num_killed_walkers, root=0)
            for ii, (send_rank, recv_rank) in enumerate(zip(mpi_send_rank, mpi_recv_rank)):
                if rank == send_rank:
                    comm.send(self.__latest_r_up_carts, dest=recv_rank, tag=100 + 2 * ii)
                    comm.send(self.__latest_r_dn_carts, dest=recv_rank, tag=100 + 2 * ii + 1)
                if rank == recv_rank:
                    self.__latest_r_up_carts = comm.recv(source=send_rank, tag=100 + 2 * ii)
                    self.__latest_r_dn_carts = comm.recv(source=send_rank, tag=100 + 2 * ii + 1)
            comm.barrier()

            logger.debug(f"*After branching: rank={rank}:gfmc.r_up_carts = {self.__latest_r_up_carts}")
            logger.debug(f"*After branching: rank={rank}:gfmc.r_dn_carts = {self.__latest_r_dn_carts}")

            comm.barrier()

            end_branching = time.perf_counter()
            timer_branching += end_branching - start_branching

            gmfc_current = time.perf_counter()
            if max_time < gmfc_current - gmfc_total_start:
                logger.info(f"max_time = {max_time} sec. exceeds.")
                logger.info("break the branching loop.")
                break

        logger.info("-End branching-")

        # count up
        self.__gfmc_branching_counter += i_branching + 1

        gmfc_total_end = time.perf_counter()
        timer_gmfc_total = gmfc_total_end - gmfc_total_start

        logger.info(f"Total GFMC time = {timer_gmfc_total: .3f} sec.")
        logger.info(f"  Projection time per branching = {timer_projection_total/num_branching: .3f} sec.")
        logger.info(f"    Non_diagonal kinetic part = {timer_projection_non_diagonal_kinetic_part/num_branching: .3f} sec.")
        logger.info(f"    Diagonal kinetic part = {timer_projection_diag_kinetic_part/num_branching: 3f} sec.")
        logger.info(f"    Diagonal ecp part = {timer_projection_diag_ecp_part/num_branching: 3f} sec.")
        logger.info(f"    Diagonal bare coulomb part = {timer_projection_diag_bare_couloumb_part/num_branching: 3f} sec.")
        logger.info(f"    Non_diagonal ecp part = {timer_projection_non_diagonal_ecp_part/num_branching: .3f} sec.")
        logger.info(
            f"    Update weights and positions = {timer_projection_update_weights_and_positions/num_branching: .3f} sec."
        )
        logger.info(f"  Observable time per branching = {timer_observable/num_branching: .3f} sec.")
        logger.info(f"  Branching time per branching = {timer_branching/num_branching: .3f} sec.")
        logger.debug(f"Survived walkers = {self.__num_survived_walkers}")
        logger.debug(f"killed walkers = {self.__num_killed_walkers}")
        logger.info(
            f"Survived walkers ratio = {self.__num_survived_walkers/(self.__num_survived_walkers + self.__num_killed_walkers) * 100:.2f} %"
        )

    def get_e_L(self, num_gfmc_warmup_steps: int = 3, num_gfmc_bin_blocks: int = 10, num_gfmc_bin_collect: int = 2) -> float:
        """Get e_L."""
        if rank == 0:
            e_L_eq = self.__e_L_averaged_list[num_gfmc_warmup_steps + num_gfmc_bin_collect :]
            w_L_eq = self.__w_L_averaged_list[num_gfmc_warmup_steps:]
            logger.debug(f"e_L_eq = {e_L_eq}")
            logger.debug(f"w_L_eq = {w_L_eq}")
            G_eq = [
                np.prod([w_L_eq[n - j] for j in range(1, num_gfmc_bin_collect + 1)])
                for n in range(num_gfmc_bin_collect, len(w_L_eq))
            ]
            logger.debug(f"G_eq = {G_eq}")
            logger.debug(f"len(e_L_eq) = {len(e_L_eq)}")
            logger.debug(f"len(G_eq) = {len(G_eq)}")

            e_L_eq = np.array(e_L_eq)
            G_eq = np.array(G_eq)

            G_e_L_eq = e_L_eq * G_eq

            G_e_L_split = np.array_split(G_e_L_eq, num_gfmc_bin_blocks)
            G_e_L_binned = np.array([np.average(G_e_L_list) for G_e_L_list in G_e_L_split])
            G_split = np.array_split(G_eq, num_gfmc_bin_blocks)
            G_binned = np.array([np.average(G_list) for G_list in G_split])

            M = num_gfmc_bin_blocks

            logger.info(f"The number of binned blocks = {num_gfmc_bin_blocks}.")

            e_L_jackknife = [(np.sum(G_e_L_binned) - G_e_L_binned[m]) / (np.sum(G_binned) - G_binned[m]) for m in range(M)]

            e_L_mean = np.average(e_L_jackknife)
            e_L_std = np.sqrt(M - 1) * np.std(e_L_jackknife)

            logger.debug(f"e_L = {e_L_mean} +- {e_L_std} Ha")
        else:
            e_L_mean = None
            e_L_std = None

        e_L_mean = comm.bcast(e_L_mean, root=0)
        e_L_std = comm.bcast(e_L_std, root=0)

        return e_L_mean, e_L_std

    @property
    def hamiltonian_data(self):
        return self.__hamiltonian_data

    @property
    def e_L(self):
        return self.__latest_e_L

    @property
    def w_L(self):
        return self.__latest_w_L

    @property
    def latest_r_up_carts(self):
        return self.__latest_r_up_carts

    @property
    def latest_r_dn_carts(self):
        return self.__latest_r_dn_carts


if __name__ == "__main__":
    logger_level = "DEBUG"

    log = getLogger("jqmc")

    if logger_level == "MPI-INFO":
        if rank == 0:
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
            handler_format = Formatter(f"MPI-rank={rank}: %(name)s - %(levelname)s - %(lineno)d - %(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
    else:
        log.setLevel(logger_level)
        stream_handler = StreamHandler()
        stream_handler.setLevel(logger_level)
        handler_format = Formatter(f"MPI-rank={rank}: %(name)s - %(levelname)s - %(lineno)d - %(message)s")
        stream_handler.setFormatter(handler_format)
        log.addHandler(stream_handler)

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

    # run branching
    mcmc_seed = 3446
    tau = 0.01
    alat = 0.3
    num_branching = 10
    non_local_move = "tmove"

    num_gfmc_warmup_steps = 2
    num_gfmc_bin_blocks = 2
    num_gfmc_bin_collect = 2

    # run GFMC
    gfmc = GFMC(hamiltonian_data=hamiltonian_data, mcmc_seed=mcmc_seed, tau=tau, alat=alat, non_local_move=non_local_move)
    gfmc.run(num_branching=num_branching)
    e_L_mean, e_L_std = gfmc.get_e_L(
        num_gfmc_warmup_steps=num_gfmc_warmup_steps,
        num_gfmc_bin_blocks=num_gfmc_bin_blocks,
        num_gfmc_bin_collect=num_gfmc_bin_collect,
    )
    logger.info(f"e_L = {e_L_mean} +- {e_L_std} Ha")
