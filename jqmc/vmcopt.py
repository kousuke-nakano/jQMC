"""VMC module"""

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
import os
import random
from logging import Formatter, StreamHandler, getLogger

# import mpi4jax
# JAX
import jax
import numpy as np
import numpy.typing as npt

# MPI
from mpi4py import MPI

from .hamiltonians import Hamiltonian_data
from .jastrow_factor import Jastrow_data
from .trexio_wrapper import read_trexio_file
from .vmc import MCMC
from .wavefunction import Wavefunction_data

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


class VMCopt:
    """VMC class.

    Runing VMC using MCMC.

    Args:
        mcmc_seed (int): seed for the MCMC chain.
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
        mcmc_seed (int): random seed for MCMC
        num_mcmc_warmup_steps (int): number of equilibration steps.
        num_mcmc_bin_blocks (int): number of blocks for reblocking.
        Dt (float): electron move step (bohr)
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data = None,
        mcmc_seed: int = 34467,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
        Dt: float = 2.0,
    ) -> None:
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        log = getLogger("jqmc")
        log.setLevel("INFO")
        stream_handler = StreamHandler()
        stream_handler.setLevel("INFO")
        handler_format = Formatter(
            f"MPI-rank={self.rank}: %(name)s - %(levelname)s - %(lineno)d - %(message)s"
        )
        stream_handler.setFormatter(handler_format)
        log.addHandler(stream_handler)

        self.mpi_seed = mcmc_seed * (self.rank + 1)

        logger.info(f"mcmc_seed for MPI-rank={self.rank} is {self.mpi_seed}.")

        # set random seeds
        random.seed(self.mpi_seed)
        np.random.seed(self.mpi_seed)

        self.__num_mcmc_warmup_steps = num_mcmc_warmup_steps
        self.__num_mcmc_bin_blocks = num_mcmc_bin_blocks

        # set the initial electron configurations
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
            num_electrons = int(
                np.round(charge)
            )  # Number of electrons to place based on the charge

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

        self.__mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            init_r_up_carts=init_r_up_carts,
            init_r_dn_carts=init_r_dn_carts,
            mcmc_seed=self.mpi_seed,
            Dt=Dt,
        )

    def run(self, num_mcmc_steps=0):
        if self.rank == 0:
            logger.info(f"num_mcmc_warmup_steps={self.__num_mcmc_warmup_steps}.")
            logger.info(f"num_mcmc_bin_blocks={self.__num_mcmc_bin_blocks}.")

        # run VMC
        self.__mcmc.run(num_mcmc_steps=num_mcmc_steps)

        latest_r_up_carts = self.__mcmc.latest_r_up_carts
        latest_r_dn_carts = self.__mcmc.latest_r_dn_carts
        Dt = 2.0

        # update WF
        self.__mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            init_r_up_carts=latest_r_up_carts,
            init_r_dn_carts=latest_r_dn_carts,
            mcmc_seed=self.mpi_seed,
            Dt=Dt,
        )

        # run VMC
        self.__mcmc.run(num_mcmc_steps=num_mcmc_steps)

    def get_e_L(self):
        # analysis VMC
        e_L = self.__mcmc.e_L[self.__num_mcmc_warmup_steps :]
        e_L_split = np.array_split(e_L, self.__num_mcmc_bin_blocks)
        e_L_binned = [np.average(e_list) for e_list in e_L_split]

        logger.debug(f"e_L_binned for MPI-rank={self.rank} is {e_L_binned}.")

        e_L_binned = self.comm.reduce(e_L_binned, op=MPI.SUM, root=0)

        if self.rank == 0:
            logger.debug(f"e_L_binned = {e_L_binned}.")
            # jackknife implementation
            # https://www2.yukawa.kyoto-u.ac.jp/~etsuko.itou/old-HP/Notes/Jackknife-method.pdf
            e_L_jackknife_binned = [
                np.average(np.delete(e_L_binned, i)) for i in range(len(e_L_binned))
            ]

            logger.debug(f"e_L_jackknife_binned  = {e_L_jackknife_binned}.")

            e_L_mean = np.average(e_L_jackknife_binned)
            e_L_std = np.sqrt(len(e_L_binned) - 1) * np.std(e_L_jackknife_binned)

            logger.info(f"e_L = {e_L_mean} +- {e_L_std} Ha.")
        else:
            e_L_mean = 0.0
            e_L_std = 0.0

        e_L_mean = self.comm.bcast(e_L_mean, root=0)
        e_L_std = self.comm.bcast(e_L_std, root=0)

        return (e_L_mean, e_L_std)

    def get_atomic_forces(self):
        # analysis VMC
        e_L_mean, _ = self.get_e_L()

        e_L_list = self.__mcmc.e_L[self.__num_mcmc_warmup_steps :]
        de_L_dR_list = self.__mcmc.de_L_dR[self.__num_mcmc_warmup_steps :]
        de_L_dr_up_list = self.__mcmc.de_L_dr_up[self.__num_mcmc_warmup_steps :]
        de_L_dr_dn_list = self.__mcmc.de_L_dr_dn[self.__num_mcmc_warmup_steps :]
        dln_Psi_dr_up_list = self.__mcmc.dln_Psi_dr_up[self.__num_mcmc_warmup_steps :]
        dln_Psi_dr_dn_list = self.__mcmc.dln_Psi_dr_dn[self.__num_mcmc_warmup_steps :]
        dln_Psi_dR_list = self.__mcmc.dln_Psi_dR[self.__num_mcmc_warmup_steps :]
        omega_up_list = self.__mcmc.omega_up[self.__num_mcmc_warmup_steps :]
        omega_dn_list = self.__mcmc.omega_dn[self.__num_mcmc_warmup_steps :]
        domega_dr_up_list = self.__mcmc.domega_dr_up[self.__num_mcmc_warmup_steps :]
        domega_dr_dn_list = self.__mcmc.domega_dr_dn[self.__num_mcmc_warmup_steps :]

        force = []

        for (
            e_L,
            de_L_dR,
            de_L_dr_up,
            de_L_dr_dn,
            dln_Psi_dr_up,
            dln_Psi_dr_dn,
            dln_Psi_dR,
            omega_up,
            omega_dn,
            domega_dr_up,
            domega_dr_dn,
        ) in zip(
            e_L_list,
            de_L_dR_list,
            de_L_dr_up_list,
            de_L_dr_dn_list,
            dln_Psi_dr_up_list,
            dln_Psi_dr_dn_list,
            dln_Psi_dR_list,
            omega_up_list,
            omega_dn_list,
            domega_dr_up_list,
            domega_dr_dn_list,
        ):
            # logger.info(f"e_L.shape = {e_L.shape}")
            # logger.info(f"de_L_dR.shape = {de_L_dR.shape}")
            # logger.info(f"de_L_dr_up.shape = {de_L_dr_up.shape}")
            # logger.info(f"de_L_dr_dn.shape = {de_L_dr_dn.shape}")
            # logger.info(f"dln_Psi_dr_up.shape = {dln_Psi_dr_up.shape}")
            # logger.info(f"dln_Psi_dr_dn.shape = {dln_Psi_dr_dn.shape}")
            # logger.info(f"de_L_dR.shape = {de_L_dR.shape}")
            # logger.info(f"omega_up.shape = {omega_up.shape}")
            # logger.info(f"omega_dn.shape = {omega_dn.shape}")
            # logger.info(f"domega_dr_up.shape = {domega_dr_up.shape}")
            # logger.info(f"domega_dr_dn.shape = {domega_dr_dn.shape}")

            force.append(
                -(de_L_dR + omega_up @ de_L_dr_up + omega_dn @ de_L_dr_dn)
                - 2
                * (
                    (e_L - e_L_mean)
                    * (
                        dln_Psi_dR
                        + omega_up @ dln_Psi_dr_up
                        + omega_dn @ dln_Psi_dr_dn
                        + 1.0 / 2.0 * (domega_dr_up + domega_dr_dn)
                    )
                )
            )

        force = np.array(force)
        logger.info(f"force.shape = {force.shape}")

        force_split = np.array_split(force, self.__num_mcmc_bin_blocks)
        force_binned = np.array([np.average(force_list, axis=0) for force_list in force_split])
        logger.info(f"force_binned.shape = {force_binned.shape}")

        logger.info(f"force_binned for MPI-rank={self.rank} is {force_binned}.")

        force_binned = self.comm.reduce(force_binned, op=MPI.SUM, root=0)

        if self.rank == 0:
            # logger.info(f"force_binned = {force_binned }.")
            # jackknife implementation
            # https://www2.yukawa.kyoto-u.ac.jp/~etsuko.itou/old-HP/Notes/Jackknife-method.pdf
            force_jackknife_binned = np.array(
                [
                    np.average(np.delete(force_binned, i, axis=0), axis=0)
                    for i in range(len(force_binned))
                ]
            )

            logger.info(f"force_jackknife_binned.shape = {force_jackknife_binned.shape}.")

            force_mean = np.average(force_jackknife_binned, axis=0)
            force_std = np.sqrt(len(force_jackknife_binned) - 1) * np.std(
                force_jackknife_binned, axis=0
            )

            logger.info(f"force_mean.shape  = {force_mean.shape}.")
            logger.info(f"force_std.shape  = {force_std.shape}.")

            logger.info(f"force = {force_mean} +- {force_std} Ha.")

        else:
            force_mean = np.array([])
            force_std = np.array([])

        force_mean = self.comm.bcast(force_mean, root=0)
        force_std = self.comm.bcast(force_std, root=0)

        return (force_mean, force_std)


if __name__ == "__main__":
    """
    # water cc-pVTZ with Mitas ccECP (8 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "water_trexio.hdf5")
    )
    """

    # """
    # H2 dimer cc-pV5Z with Mitas ccECP (2 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "H2_dimer_trexio.hdf5")
    )
    # """

    """ Error!! To be fixed.
    # Ne atom cc-pV5Z with Mitas ccECP (10 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), 'trexio_files', "Ne_trexio.hdf5"))
    """

    """
    # benzene cc-pVDZ with Mitas ccECP (30 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), 'trexio_files', "benzene_trexio.hdf5"))
    """

    """
    # C60 cc-pVTZ with Mitas ccECP (240 electrons, not feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "C60_trexio.hdf5"))
    """

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=None,
        jastrow_two_body_type="off",
        jastrow_three_body_data=None,
        jastrow_three_body_type="off",
    )  # no jastrow for the time-being.

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    # VMC parameters
    num_mcmc_warmup_steps = 20
    num_mcmc_bin_blocks = 5
    mcmc_seed = 34356

    # run VMC
    vmcopt = VMCopt(
        hamiltonian_data=hamiltonian_data,
        mcmc_seed=mcmc_seed,
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )
    vmcopt.run(num_mcmc_steps=100)
    vmcopt.get_e_L()
    vmcopt.get_atomic_forces()
