"""LRDMC module."""

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
from .coulomb_potential import compute_bare_coulomb_potential_jax, compute_ecp_local_parts_jax, compute_ecp_non_local_parts_jax
from .hamiltonians import Hamiltonian_data, compute_kinetic_energy_api, compute_local_energy
from .jastrow_factor import Jastrow_data, Jastrow_three_body_data, Jastrow_two_body_data
from .trexio_wrapper import read_trexio_file
from .wavefunction import Wavefunction_data, compute_discretized_kinetic_energy_jax

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
        init_r_up_carts (npt.NDArray): starting electron positions for up electrons
        init_r_dn_carts (npt.NDArray): starting electron positions for dn electrons
        tau (float): projection time (bohr^-1)
        alat (float): discretized grid length (bohr)
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data = None,
        init_r_up_carts: npt.NDArray[np.float64] = None,
        init_r_dn_carts: npt.NDArray[np.float64] = None,
        mcmc_seed: int = 34467,
        tau: float = 0.1,
        alat: float = 0.1,
    ) -> None:
        """Init.

        Initialize a MCMC class, creating list holding results, etc...

        """
        self.__hamiltonian_data = hamiltonian_data
        self.__mcmc_seed = mcmc_seed
        self.__tau = tau
        self.__alat = alat

        # latest electron positions
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
        _, _, _ = compute_discretized_kinetic_energy_jax(
            alat=self.__alat,
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        _ = compute_bare_coulomb_potential_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        _ = compute_ecp_local_parts_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        _, _, _ = compute_ecp_non_local_parts_jax(
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

        # init attributes
        self.init_attributes()

    def init_attributes(self):
        # init attributes
        # stored local energy (e_L) and weight (w_L)
        self.__latest_e_L = 0.0
        self.__latest_w_L = 1.0

    def run(self) -> None:
        """Run LRDMC."""
        cpu_count = os.cpu_count()
        logger.debug(f"cpu count = {cpu_count}")

        # MAIN MCMC loop from here !!!
        tau_left = self.__tau
        logger.info(f"  Left projection time = {tau_left}/{self.__tau}: {0.0:.0f} %.")

        # timer_counter
        timer_mcmc_updated = 0.0
        timer_e_L = 0.0

        mcmc_total_start = time.perf_counter()

        w_L = self.__latest_w_L
        jax_PRNG_key = jax.random.PRNGKey(self.__mcmc_seed)
        while tau_left > 0.0:
            progress = (tau_left) / (self.__tau) * 100.0
            logger.info(f"  Left projection time = {tau_left}/{self.__tau}: {progress:.1f} %.")

            # compute non-diagonal grids and elements
            mesh_kinetic_part, elements_kinetic_part, jax_PRNG_key = compute_discretized_kinetic_energy_jax(
                alat=self.__alat,
                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
                jax_PRNG_key=jax_PRNG_key,
            )
            mesh_non_local_ecp_part, V_nonlocal, sum_V_nonlocal = compute_ecp_non_local_parts_jax(
                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
            )

            # Fixed-node.
            elements_kinetic_part_FN = list(map(lambda x: x if x < 0 else 0.0, elements_kinetic_part))
            V_nonlocal_FN = list(map(lambda x: x if x < 0 else 0.0, V_nonlocal))

            sum_non_diagonal_hamiltonian = np.sum(elements_kinetic_part_FN) + np.sum(V_nonlocal_FN)

            # compute diagonal element
            kinetic_continuum = compute_kinetic_energy_api(
                wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
            )
            kinetic_discretized = -1.0 * np.sum(elements_kinetic_part)
            bare_coulomb_part = compute_bare_coulomb_potential_jax(
                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
            )
            ecp_local_part = compute_ecp_local_parts_jax(
                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
            )
            elements_kinetic_part_SP = np.sum(list(map(lambda x: x if x >= 0 else 0.0, elements_kinetic_part)))
            V_nonlocal_SP = np.sum(list(map(lambda x: x if x >= 0 else 0.0, V_nonlocal)))

            # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
            e_L = (
                kinetic_continuum
                + kinetic_discretized
                + bare_coulomb_part
                + ecp_local_part
                + elements_kinetic_part_SP
                + V_nonlocal_SP
                + sum_non_diagonal_hamiltonian
            )
            logger.debug(f"  sum_non_diagonal_hamiltonian = {sum_non_diagonal_hamiltonian}")
            logger.info(f"  e_L={e_L}")

            """
            logger.info(self.__latest_r_up_carts)
            logger.info(self.__latest_r_dn_carts)
            e_L_debug = compute_local_energy(
                hamiltonian_data=self.__hamiltonian_data,
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
            )
            logger.info(f"  e_L_debug={e_L_debug}")
            """

            # compute the time the walker remaining in the same configuration
            xi = random.random()
            tau_update = np.min((tau_left, np.log(1 - xi) / sum_non_diagonal_hamiltonian))
            logger.info(f"  tau_update={tau_update}")

            # update weight
            w_L = w_L * np.exp(-tau_update * e_L)
            logger.info(f"  w_L={w_L}")

            # update tau_left
            tau_left = tau_left - tau_update

            # choose a non-diagonal move destination
            p_list = np.array(elements_kinetic_part_FN + V_nonlocal_FN)
            probabilities = p_list / p_list.sum()
            k = np.random.choice(len(p_list), p=probabilities)

            # update electron position
            self.__latest_r_up_carts, self.__latest_r_dn_carts = (list(mesh_kinetic_part) + list(mesh_non_local_ecp_part))[k]

            end = time.perf_counter()

        # projection ends
        logger.info("  Projection ends.")

        # evaluate observables
        start = time.perf_counter()
        # compute non-diagonal grids and elements
        _, elements_kinetic_part, jax_PRNG_key = compute_discretized_kinetic_energy_jax(
            alat=self.__alat,
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
            jax_PRNG_key=jax_PRNG_key,
        )
        _, V_nonlocal, _ = compute_ecp_non_local_parts_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )

        # Fixed-node.
        elements_kinetic_part_FN = list(map(lambda x: x if x < 0 else 0.0, elements_kinetic_part))
        V_nonlocal_FN = list(map(lambda x: x if x < 0 else 0.0, V_nonlocal))

        sum_non_diagonal_hamiltonian = np.sum(elements_kinetic_part_FN) + np.sum(V_nonlocal_FN)

        # compute diagonal element
        kinetic_continuum = compute_kinetic_energy_api(
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        kinetic_discretized = -1.0 * np.sum(elements_kinetic_part)
        bare_coulomb_part = compute_bare_coulomb_potential_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        ecp_local_part = compute_ecp_local_parts_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        elements_kinetic_part_SP = np.sum(list(map(lambda x: x if x >= 0 else 0.0, elements_kinetic_part)))
        V_nonlocal_SP = np.sum(list(map(lambda x: x if x >= 0 else 0.0, V_nonlocal)))

        # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
        e_L = (
            kinetic_continuum
            + kinetic_discretized
            + bare_coulomb_part
            + ecp_local_part
            + elements_kinetic_part_SP
            + V_nonlocal_SP
            + sum_non_diagonal_hamiltonian
        )

        end = time.perf_counter()
        timer_e_L += end - start

        logger.devel(f"  e_L = {e_L}")
        logger.devel(f"  w_L = {w_L}")
        self.__latest_w_L = float(w_L)
        self.__latest_e_L = float(e_L)

        mcmc_total_end = time.perf_counter()
        timer_mcmc_total = mcmc_total_end - mcmc_total_start
        timer_others = timer_mcmc_total - (timer_mcmc_updated + timer_e_L)

        logger.info(f"Total Elapsed time for MCMC = {timer_mcmc_total*10**3:.2f} msec.")
        logger.info(f"Time for MCMC updated = {timer_mcmc_updated*10**3:.2f} msec.")
        logger.info(f"Time for computing e_L = {timer_e_L*10**3:.2f} msec.")
        logger.info(f"Time for misc. (others) = {timer_others*10**3:.2f} msec.")

    @property
    def hamiltonian_data(self):
        return self.__hamiltonian_data

    @property
    def alat(self):
        return self.__alat

    @property
    def tau(self):
        return self.__tau

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

    @property
    def mcmc_seed(self):
        return self.__mcmc_seed

    @mcmc_seed.setter
    def mcmc_seed(self, mcmc_seed):
        self.__mcmc_seed = mcmc_seed


if __name__ == "__main__":
    logger_level = "INFO"

    log = getLogger("jqmc")

    if logger_level == "MPI-INFO":
        if rank == 0:
            log.setLevel("INFO")
            stream_handler = StreamHandler()
            stream_handler.setLevel("INFO")
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

    # LRDMC parameters
    alat = 0.1
    tau = 0.1
    mcmc_seed = 34356

    # Initialization
    r_up_carts = []
    r_dn_carts = []

    total_electrons = 0

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data.positions_cart

    # set random seeds
    mpi_seed = mcmc_seed * (rank + 1)
    random.seed(mpi_seed)
    np.random.seed(mpi_seed)

    # Place electrons around each nucleus
    num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
    num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn

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

    init_r_up_carts = np.array(r_carts_up)
    init_r_dn_carts = np.array(r_carts_dn)

    logger.info(f"initial r_up_carts = {init_r_up_carts}")
    logger.info(f"initial r_dn_carts = {init_r_dn_carts}")

    # run branching
    num_branching = 50
    num_survived_walkers = 0
    num_killed_walkers = 0
    e_L_averaged_list = []
    w_L_averaged_list = []

    # run GFMC
    gfmc = GFMC(
        hamiltonian_data=hamiltonian_data,
        mcmc_seed=mpi_seed,
        tau=tau,
        alat=alat,
        init_r_up_carts=init_r_up_carts,
        init_r_dn_carts=init_r_dn_carts,
    )

    for i_branching in range(num_branching):
        logger.info(f"i_branching = {i_branching}/{num_branching}")
        gfmc.run()

        w_L = gfmc.w_L
        e_L = gfmc.e_L
        logger.info(f"e_L={e_L} for rank={rank}")
        logger.info(f"w_L={w_L} for rank={rank}")

        e_L_gathered_dyad = (rank, e_L)
        e_L_gathered_dyad = comm.gather(e_L_gathered_dyad, root=0)
        w_L_gathered_dyad = (rank, w_L)
        w_L_gathered_dyad = comm.gather(w_L_gathered_dyad, root=0)
        if rank == 0:
            logger.info(e_L_gathered_dyad)
            logger.info(w_L_gathered_dyad)
            e_L_gathered = np.array([e_L for _, e_L in e_L_gathered_dyad])
            w_L_gathered = np.array([w_L for _, w_L in w_L_gathered_dyad])
            e_L_averaged = np.sum(w_L_gathered * e_L_gathered) / np.sum(w_L_gathered)
            w_L_averaged = np.average(w_L_gathered)
            logger.info(f"e_L_averaged={e_L_averaged} for rank={rank}")
            logger.info(f"w_L_averaged={w_L_averaged} for rank={rank}")
            e_L_averaged_list.append(e_L_averaged)
            w_L_averaged_list.append(w_L_averaged)
            mpi_rank_list = [r for r, _ in w_L_gathered_dyad]
            w_L_list = np.array([w_L for _, w_L in w_L_gathered_dyad])
            logger.info(f"w_L_list = {w_L_list}")
            probabilities = w_L_list / w_L_list.sum()
            logger.info(f"probabilities = {probabilities}")
            k_list = np.random.choice(len(w_L_list), size=len(w_L_list), p=probabilities, replace=True)
            chosen_rank_list = [w_L_gathered_dyad[k][0] for k in k_list]
            chosen_rank_list.sort()
            logger.info(f"chosen_rank_list = {chosen_rank_list}")
            counter = Counter(chosen_rank_list)
            num_survived_walkers += len(set(chosen_rank_list))
            num_killed_walkers += len(mpi_rank_list) - len(set(chosen_rank_list))
            logger.info(f"num_survived_walkers={num_survived_walkers}")
            logger.info(f"num_killed_walkers={num_killed_walkers}")
            mpi_send_rank = [item for item, count in counter.items() for _ in range(count - 1) if count > 1]
            mpi_recv_rank = list(set(mpi_rank_list) - set(chosen_rank_list))
            logger.info(f"mpi_send_rank={mpi_send_rank}")
            logger.info(f"mpi_recv_rank={mpi_recv_rank}")
        else:
            mpi_send_rank = None
            mpi_recv_rank = None
        mpi_send_rank = comm.bcast(mpi_send_rank, root=0)
        mpi_recv_rank = comm.bcast(mpi_recv_rank, root=0)

        if rank == 0:
            logger.info("-------------------------------------")
            logger.info("*before branching")
        logger.info(f"rank={rank}:gfmc.e_L = {gfmc.e_L}")
        comm.barrier()
        if rank == 0:
            logger.info("-------------------------------------")
        comm.barrier()

        for ii, (send_rank, recv_rank) in enumerate(zip(mpi_send_rank, mpi_recv_rank)):
            if rank == send_rank:
                comm.send(gfmc, dest=recv_rank, tag=100 + ii)
            if rank == recv_rank:
                gfmc = comm.recv(source=send_rank, tag=100 + ii)

        if rank == 0:
            logger.info("-------------------------------------")
            logger.info("*after branching")
        logger.info(f"rank={rank}:gfmc.e_L = {gfmc.e_L}")
        comm.barrier()
        if rank == 0:
            logger.info("-------------------------------------")
        comm.barrier()

        # run GFMC
        gfmc = GFMC(
            hamiltonian_data=hamiltonian_data,
            mcmc_seed=mpi_seed + i_branching * rank,
            tau=tau,
            alat=alat,
            init_r_up_carts=gfmc.latest_r_up_carts,
            init_r_dn_carts=gfmc.latest_r_dn_carts,
        )

        """
        # reset weight -> 0
        gfmc.init_attributes()
        logger.info(f"gfmc.w_L={gfmc.w_L}")

        # reset random seeds
        gfmc.mcmc_seed = mpi_seed
        logger.info(f"gfmc.mcmc_seed={gfmc.mcmc_seed}")
        """

    logger.info("Branching is over.")
    if rank == 0:
        logger.info(e_L_averaged_list)
        logger.info(w_L_averaged_list)
        num_gfmc_warmup_steps = 3
        num_gfmc_bin_blocks = 10
        num_gfmc_bin_collect = 2
        e_L_averaged_eq = e_L_averaged_list[num_gfmc_warmup_steps:]
        e_L_split = np.array_split(e_L_averaged_eq, num_gfmc_bin_blocks)
        e_L_binned = np.array([np.average(e_list) for e_list in e_L_split])
        w_L_averaged_eq = w_L_averaged_list[num_gfmc_warmup_steps:]
        w_L_split = np.array_split(w_L_averaged_eq, num_gfmc_bin_blocks)
        w_L_binned = np.array([np.average(w_list) for w_list in w_L_split])

        logger.info(e_L_binned)
        logger.info(w_L_binned)

        e_L_binned_c = e_L_binned[num_gfmc_bin_collect:]
        G_binned_c = np.array(
            [
                np.sum([w_L_binned[i - k] for k in range(num_gfmc_bin_collect)])
                for i in range(num_gfmc_bin_collect, len(w_L_binned))
            ]
        )

        logger.info(e_L_binned_c)
        logger.info(G_binned_c)

        M = len(e_L_binned_c)
        e_L_jackknife = [
            (np.sum(G_binned_c * e_L_binned_c) - G_binned_c[m] * e_L_binned_c[m]) / (np.sum(G_binned_c) - G_binned_c[m])
            for m in range(M)
        ]

        logger.info(e_L_jackknife)

        e_L_mean = np.average(e_L_jackknife)
        e_L_std = np.sqrt(M - 1) * np.std(e_L_jackknife)

        logger.info(f"e_L = {e_L_mean} +- {e_L_std} Ha")
        logger.info(f"Survived walkers = {num_survived_walkers}")
        logger.info(f"killed walkers = {num_killed_walkers}")
        logger.info(f"Survived walkers ratio = {num_survived_walkers/(num_survived_walkers + num_killed_walkers) * 100} :.3f %")
