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
from jax import grad

# MPI
from mpi4py import MPI

from .hamiltonians import Hamiltonian_data, compute_local_energy
from .jastrow_factor import Jastrow_data
from .structure import find_nearest_index
from .swct import SWCT_data, evaluate_swct_domega_api, evaluate_swct_omega_api
from .trexio_wrapper import read_trexio_file
from .wavefunction import Wavefunction_data, evaluate_ln_wavefunction_api, evaluate_wavefunction_api

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


class MCMC:
    """MCMC class.

    MCMC class. Runing MCMC.

    Args:
        mcmc_seed (int): seed for the MCMC chain.
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
        swct_data (SWCT_data): an instance of SWCT_data
        mcmc_seed (int): random seed for MCMC
        Dt (float): electron move step (bohr)
        flag_energy_deriv (bool): compute derivatives of local energy.
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data = None,
        swct_data: SWCT_data = None,
        mcmc_seed: int = 34467,
        Dt: float = 2.0,
        flag_energy_deriv: bool = True,
    ) -> None:
        """Init.

        Initialize a MCMC class, creating list holding results, etc...

        Args:
            hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
            swct_data (SWCT_data)

        """
        self.__hamiltonian_data = hamiltonian_data
        self.__swct_data = swct_data
        self.__mcmc_seed = mcmc_seed
        self.__Dt = Dt
        self.__flag_energy_deriv = flag_energy_deriv

        # mcmc counter
        self.__mcmc_counter = 0

        # latest electron positions
        self.__latest_r_up_carts = None
        self.__latest_r_dn_carts = None

        # stored electron positions
        self.__stored_r_up_carts = None
        self.__stored_r_dn_carts = None

        # stored local energy (e_L)
        self.__stored_e_L = []

        # stored de_L / dR
        self.__stored_grad_e_L_dR = []

        # stored de_L / dr_up
        self.__stored_grad_e_L_r_up = []

        # stored de_L / dr_dn
        self.__stored_grad_e_L_r_dn = []

        # stored ln_Psi
        self.__stored_ln_Psi = []

        # stored dln_Psi / dr_up
        self.__stored_grad_ln_Psi_r_up = []

        # stored dln_Psi / dr_dn
        self.__stored_grad_ln_Psi_r_dn = []

        # stored dln_Psi / dR
        self.__stored_grad_ln_Psi_dR = []

        # intialize all attributes
        self.__init__attributes()

    def __init__attributes(self):
        # set random seeds
        random.seed(self.__mcmc_seed)
        np.random.seed(self.__mcmc_seed)

        # set the initial electron configurations
        num_electron_up = self.__hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        num_electron_dn = self.__hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn

        # Initialization
        r_carts_up = []
        r_carts_dn = []

        total_electrons = 0

        if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = np.array(self.__hamiltonian_data.structure_data.atomic_numbers) - np.array(
                self.__hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = np.array(self.__hamiltonian_data.structure_data.atomic_numbers)

        coords = self.__hamiltonian_data.structure_data.positions_cart

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

        self.__latest_r_up_carts = np.array(r_carts_up)
        self.__latest_r_dn_carts = np.array(r_carts_dn)

        logger.info(f"initial r_up_carts = { self.__latest_r_up_carts}")
        logger.info(f"initial r_dn_carts = { self.__latest_r_dn_carts}")

        # reset mcmc counter
        self.__mcmc_counter = 0

        # stored local energy (e_L)
        self.__stored_e_L = []

        # stored de_L / dR
        self.__stored_grad_e_L_dR = []

        # stored de_L / dr_up
        self.__stored_grad_e_L_r_up = []

        # stored de_L / dr_dn
        self.__stored_grad_e_L_r_dn = []

        # stored ln_Psi
        self.__stored_ln_Psi = []

        # stored dln_Psi / dr_up
        self.__stored_grad_ln_Psi_r_up = []

        # stored dln_Psi / dr_dn
        self.__stored_grad_ln_Psi_r_dn = []

        # stored dln_Psi / dR
        self.__stored_grad_ln_Psi_dR = []

        # stored Omega_up (SWCT)
        self.__stored_omega_up = []

        # stored Omega_dn (SWCT)
        self.__stored_omega_dn = []

        # stored sum_i d omega/d r_i for up spins (SWCT)
        self.__stored_grad_omega_r_up = []

        # stored sum_i d omega/d r_i for dn spins (SWCT)
        self.__stored_grad_omega_r_dn = []

        # """
        # compiling methods
        # jax.profiler.start_trace("/tmp/tensorboard")
        # tensorboard --logdir /tmp/tensorboard
        # tensorborad does not work with safari. use google chrome

        logger.info("Compilation starts.")

        logger.info("  Compilation e_L starts.")
        _ = compute_local_energy(
            hamiltonian_data=self.__hamiltonian_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        logger.info("  Compilation e_L is done.")

        if self.__flag_energy_deriv:
            logger.info("  Compilation de_L starts.")
            _, _, _ = grad(compute_local_energy, argnums=(0, 1, 2))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            logger.info("  Compilation de_L is done.")

            logger.info("  Compilation dln_Psi starts.")
            _, _, _ = grad(evaluate_ln_wavefunction_api, argnums=(0, 1, 2))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            logger.info("  Compilation dln_Psi is done.")

            logger.info("  Compilation domega starts.")
            _ = evaluate_swct_domega_api(
                self.__swct_data,
                self.__latest_r_up_carts,
            )
            logger.info("  Compilation domega is done.")

        logger.info("Compilation is done.")

        # jax.profiler.stop_trace()
        # """

    def run(self, num_mcmc_steps: int = 0) -> None:
        """
        Args:
            num_mcmc_steps (int): the number of total mcmc steps
            continuation (int): 1 = VMC run from sctach, 0 = VMC run continuataion
        Returns:
            None
        """
        cpu_count = os.cpu_count()
        logger.info(f"cpu count = {cpu_count}")

        # Set the random seed. Use the Mersenne Twister generator
        accepted_moves = 0
        nbra = 16

        # MAIN MCMC loop from here !!!
        for i_mcmc_step in range(num_mcmc_steps):
            logger.info(
                f"Current MCMC step = {i_mcmc_step+1+self.__mcmc_counter}/{num_mcmc_steps+self.__mcmc_counter}."
            )

            # Determine the total number of electrons
            total_electrons = len(self.__latest_r_up_carts) + len(self.__latest_r_dn_carts)

            if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
                charges = np.array(
                    self.__hamiltonian_data.structure_data.atomic_numbers
                ) - np.array(self.__hamiltonian_data.coulomb_potential_data.z_cores)
            else:
                charges = np.array(self.__hamiltonian_data.structure_data.atomic_numbers)

            coords = self.__hamiltonian_data.structure_data.positions_cart

            for _ in range(nbra):
                # Choose randomly if the electron comes from up or dn
                if random.randint(0, total_electrons - 1) < len(self.__latest_r_up_carts):
                    selected_electron_spin = "up"
                    # Randomly select an electron from r_carts_up
                    selected_electron_index = random.randint(0, len(self.__latest_r_up_carts) - 1)

                    old_r_cart = self.__latest_r_up_carts[selected_electron_index]
                else:
                    selected_electron_spin = "dn"
                    # Randomly select an electron from r_carts_dn
                    selected_electron_index = random.randint(0, len(self.__latest_r_dn_carts) - 1)
                    old_r_cart = self.__latest_r_dn_carts[selected_electron_index]

                nearest_atom_index = find_nearest_index(
                    self.__hamiltonian_data.structure_data, old_r_cart
                )

                R_cart = coords[nearest_atom_index]
                Z = charges[nearest_atom_index]
                norm_r_R = np.linalg.norm(old_r_cart - R_cart)
                f_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

                logger.debug(f"nearest_atom_index = {nearest_atom_index}")
                logger.debug(f"norm_r_R = {norm_r_R}")
                logger.debug(f"f_l  = {f_l }")

                sigma = f_l * self.__Dt
                g = float(np.random.normal(loc=0, scale=sigma))
                g_vector = np.zeros(3)
                random_index = np.random.randint(0, 3)
                g_vector[random_index] = g
                logger.debug(f"jn = {random_index}, g \equiv dstep  = {g_vector}")
                new_r_cart = old_r_cart + g_vector

                if selected_electron_spin == "up":
                    proposed_r_up_carts = self.__latest_r_up_carts.copy()
                    proposed_r_dn_carts = self.__latest_r_dn_carts.copy()
                    proposed_r_up_carts[selected_electron_index] = new_r_cart
                else:
                    proposed_r_up_carts = self.__latest_r_up_carts.copy()
                    proposed_r_dn_carts = self.__latest_r_dn_carts.copy()
                    proposed_r_dn_carts[selected_electron_index] = new_r_cart

                nearest_atom_index = find_nearest_index(
                    self.__hamiltonian_data.structure_data, new_r_cart
                )

                R_cart = coords[nearest_atom_index]
                Z = charges[nearest_atom_index]
                norm_r_R = np.linalg.norm(new_r_cart - R_cart)
                f_prime_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)
                logger.debug(f"nearest_atom_index = {nearest_atom_index}")
                logger.debug(f"norm_r_R = {norm_r_R}")
                logger.debug(f"f_prime_l  = {f_prime_l }")

                logger.debug(
                    f"The selected electron is {selected_electron_index+1}-th {selected_electron_spin} electron."
                )
                logger.debug(f"The selected electron position is {old_r_cart}.")
                logger.debug(f"The proposed electron position is {new_r_cart}.")

                T_ratio = (f_l / f_prime_l) * np.exp(
                    -(np.linalg.norm(new_r_cart - old_r_cart) ** 2)
                    * (
                        1.0 / (2.0 * f_prime_l**2 * self.__Dt**2)
                        - 1.0 / (2.0 * f_l**2 * self.__Dt**2)
                    )
                )

                R_ratio = (
                    evaluate_wavefunction_api(
                        wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                        r_up_carts=proposed_r_up_carts,
                        r_dn_carts=proposed_r_dn_carts,
                    )
                    / evaluate_wavefunction_api(
                        wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                        r_up_carts=self.__latest_r_up_carts,
                        r_dn_carts=self.__latest_r_dn_carts,
                    )
                ) ** 2.0

                logger.debug(f"R_ratio, T_ratio = {R_ratio}, {T_ratio}")
                acceptance_ratio = np.min([1.0, R_ratio * T_ratio])
                logger.debug(f"acceptance_ratio = {acceptance_ratio}")

                b = np.random.uniform(0, 1)

                if b < acceptance_ratio:
                    logger.debug("The proposed move is accepted!")
                    accepted_moves += 1
                    self.__latest_r_up_carts = proposed_r_up_carts
                    self.__latest_r_dn_carts = proposed_r_dn_carts
                else:
                    logger.debug("The proposed move is rejected!")

            # evaluate observables
            e_L = compute_local_energy(
                hamiltonian_data=self.__hamiltonian_data,
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
            )
            logger.info(f"e_L = {e_L}")
            self.__stored_e_L.append(e_L)

            if self.__flag_energy_deriv:
                # """
                grad_e_L_h, grad_e_L_r_up, grad_e_L_r_dn = grad(
                    compute_local_energy, argnums=(0, 1, 2)
                )(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                self.__stored_grad_e_L_r_up.append(grad_e_L_r_up)
                self.__stored_grad_e_L_r_dn.append(grad_e_L_r_dn)

                grad_e_L_R = (
                    grad_e_L_h.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.structure_data.positions
                    + grad_e_L_h.wavefunction_data.geminal_data.orb_data_dn_spin.aos_data.structure_data.positions
                    + grad_e_L_h.coulomb_potential_data.structure_data.positions
                )
                self.__stored_grad_e_L_dR.append(grad_e_L_R)
                # """

                # """
                logger.debug(
                    f"de_L_dR(AOs_data_up) = {grad_e_L_h.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.structure_data.positions}"
                )
                logger.debug(
                    f"de_L_dR(AOs_data_dn) = {grad_e_L_h.wavefunction_data.geminal_data.orb_data_dn_spin.aos_data.structure_data.positions}"
                )
                logger.debug(
                    f"de_L_dR(coulomb_potential_data) = {grad_e_L_h.coulomb_potential_data.structure_data.positions}"
                )
                logger.info(f"de_L_dR = {grad_e_L_R}")
                logger.info(f"de_L_dr_up = {grad_e_L_r_up}")
                logger.info(f"de_L_dr_dn= {grad_e_L_r_dn}")
                # """

                ln_Psi = evaluate_ln_wavefunction_api(
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=self.__latest_r_up_carts,
                    r_dn_carts=self.__latest_r_dn_carts,
                )
                logger.info(f"ln_Psi = {ln_Psi}")
                self.__stored_ln_Psi.append(ln_Psi)

                # """
                grad_ln_Psi_h, grad_ln_Psi_r_up, grad_ln_Psi_r_dn = grad(
                    evaluate_ln_wavefunction_api, argnums=(0, 1, 2)
                )(
                    self.__hamiltonian_data.wavefunction_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                logger.info(f"dln_Psi_dr_up = {grad_ln_Psi_r_up}")
                logger.info(f"dln_Psi_dr_dn = {grad_ln_Psi_r_dn}")
                self.__stored_grad_ln_Psi_r_up.append(grad_ln_Psi_r_up)
                self.__stored_grad_ln_Psi_r_dn.append(grad_ln_Psi_r_dn)

                grad_ln_Psi_dR = (
                    grad_ln_Psi_h.geminal_data.orb_data_up_spin.aos_data.structure_data.positions
                    + grad_ln_Psi_h.geminal_data.orb_data_dn_spin.aos_data.structure_data.positions
                )

                # stored dln_Psi / dR
                logger.info(f"dln_Psi_dR = {grad_ln_Psi_dR}")
                self.__stored_grad_ln_Psi_dR.append(grad_ln_Psi_dR)
                # """

                omega_up = evaluate_swct_omega_api(
                    swct_data=self.__swct_data,
                    r_carts=self.__latest_r_up_carts,
                )

                omega_dn = evaluate_swct_omega_api(
                    swct_data=self.__swct_data,
                    r_carts=self.__latest_r_dn_carts,
                )

                logger.info(f"omega_up = {omega_up}")
                logger.info(f"omega_dn = {omega_dn}")

                self.__stored_omega_up.append(omega_up)
                self.__stored_omega_dn.append(omega_dn)

                grad_omega_dr_up = evaluate_swct_domega_api(
                    self.__swct_data,
                    self.__latest_r_up_carts,
                )

                grad_omega_dr_dn = evaluate_swct_domega_api(
                    self.__swct_data,
                    self.__latest_r_dn_carts,
                )

                logger.info(f"grad_omega_dr_up = {grad_omega_dr_up}")
                logger.info(f"grad_omega_dr_dn = {grad_omega_dr_dn}")

                self.__stored_grad_omega_r_up.append(grad_omega_dr_up)
                self.__stored_grad_omega_r_dn.append(grad_omega_dr_dn)

        self.__mcmc_counter += num_mcmc_steps
        logger.info(f"acceptance ratio is {accepted_moves/num_mcmc_steps/nbra*100} %")

    @property
    def e_L(self):
        return self.__stored_e_L

    @property
    def de_L_dR(self):
        return self.__stored_grad_e_L_dR

    @property
    def de_L_dr_up(self):
        return self.__stored_grad_e_L_r_up

    @property
    def de_L_dr_dn(self):
        return self.__stored_grad_e_L_r_dn

    @property
    def dln_Psi_dr_up(self):
        return self.__stored_grad_ln_Psi_r_up

    @property
    def dln_Psi_dr_dn(self):
        return self.__stored_grad_ln_Psi_r_dn

    @property
    def dln_Psi_dR(self):
        return self.__stored_grad_ln_Psi_dR

    @property
    def omega_up(self):
        return self.__stored_omega_up

    @property
    def omega_dn(self):
        return self.__stored_omega_dn

    @property
    def domega_dr_up(self):
        return self.__stored_grad_omega_r_up

    @property
    def domega_dr_dn(self):
        return self.__stored_grad_omega_r_dn


class VMC:
    """VMC class.

    MCMC class. Runing MCMC.

    Args:
        mcmc_seed (int): seed for the MCMC chain.
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
        swct_data (SWCT_data): an instance of SWCT_data
        mcmc_seed (int): random seed for MCMC
        num_mcmc_warmup_steps (int): number of equilibration steps.
        num_mcmc_bin_blocks (int): number of blocks for reblocking.
        Dt (float): electron move step (bohr)
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data = None,
        swct_data: SWCT_data = None,
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

        self.__num_mcmc_warmup_steps = num_mcmc_warmup_steps
        self.__num_mcmc_bin_blocks = num_mcmc_bin_blocks

        self.__mcmc = MCMC(
            hamiltonian_data=hamiltonian_data, swct_data=swct_data, mcmc_seed=self.mpi_seed, Dt=Dt
        )

    def run(self, num_mcmc_steps=0):
        if self.rank == 0:
            logger.info(f"num_mcmc_warmup_steps={self.__num_mcmc_warmup_steps}.")
            logger.info(f"num_mcmc_bin_blocks={self.__num_mcmc_bin_blocks}.")

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
    # """
    # water cc-pVTZ with Mitas ccECP (8 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "water_trexio.hdf5"))
    # """

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
        trexio_file=os.path.join(os.path.dirname(__file__), "H2_dimer_trexio.hdf5")
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
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "Ne_trexio.hdf5"))
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
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "benzene_trexio.hdf5"))
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

    swct_data = SWCT_data(structure=structure_data)

    # VMC parameters
    num_mcmc_warmup_steps = 20
    num_mcmc_bin_blocks = 5
    mcmc_seed = 34356

    # run VMC
    vmc = VMC(
        hamiltonian_data=hamiltonian_data,
        swct_data=swct_data,
        mcmc_seed=mcmc_seed,
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )
    vmc.run(num_mcmc_steps=100)
    vmc.get_e_L()
    vmc.get_atomic_forces()
