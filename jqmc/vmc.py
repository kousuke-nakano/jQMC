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
import time
from logging import Formatter, StreamHandler, getLogger

# import mpi4jax
# JAX
import jax
import numpy as np
import numpy.typing as npt
import scipy
from jax import grad

# MPI
from mpi4py import MPI
from scipy.linalg import cho_factor, cho_solve

from .hamiltonians import Hamiltonian_data, compute_local_energy
from .jastrow_factor import Jastrow_data, Jastrow_three_body_data, Jastrow_two_body_data
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
        init_r_up_carts (npt.NDArray): starting electron positions for up electrons
        init_r_dn_carts (npt.NDArray): starting electron positions for dn electrons
        Dt (float): electron move step (bohr)
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data = None,
        init_r_up_carts: npt.NDArray[np.float64] = None,
        init_r_dn_carts: npt.NDArray[np.float64] = None,
        mcmc_seed: int = 34467,
        Dt: float = 2.0,
        comput_jas_param_deriv: bool = False,
        comput_position_deriv: bool = False,
    ) -> None:
        """Init.

        Initialize a MCMC class, creating list holding results, etc...

        """
        self.__hamiltonian_data = hamiltonian_data
        self.__mcmc_seed = mcmc_seed
        self.__Dt = Dt

        self.__comput_jas_param_deriv = comput_jas_param_deriv
        self.__comput_position_deriv = comput_position_deriv

        # set random seeds
        random.seed(self.__mcmc_seed)
        np.random.seed(self.__mcmc_seed)

        # mcmc counter
        self.__mcmc_counter = 0

        # latest electron positions
        self.__latest_r_up_carts = init_r_up_carts
        self.__latest_r_dn_carts = init_r_dn_carts

        # stored electron positions
        self.__stored_r_up_carts = None
        self.__stored_r_dn_carts = None

        # stored local energy (e_L)
        self.__stored_e_L = []

        self.__swct_data = SWCT_data(structure=self.__hamiltonian_data.structure_data)

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

        # stored dln_Psi / dr_up
        self.__stored_grad_ln_Psi_r_up = []

        # stored dln_Psi / dc_jas2b
        self.__stored_grad_ln_Psi_jas2b = []

        # stored dln_Psi / dc_jas1b3b
        self.__stored_grad_ln_Psi_jas1b3b_j_matrix_up_up = []
        self.__stored_grad_ln_Psi_jas1b3b_j_matrix_dn_dn = []
        self.__stored_grad_ln_Psi_jas1b3b_j_matrix_up_dn = []

        # """
        # compiling methods
        # jax.profiler.start_trace("/tmp/tensorboard", create_perfetto_link=True)
        # open the generated URL (UI with perfetto)
        # tensorboard --logdir /tmp/tensorboard
        # tensorborad does not work with safari. use google chrome

        logger.info("Compilation starts.")

        logger.info("Compilation e_L starts.")
        start = time.perf_counter()
        _ = compute_local_energy(
            hamiltonian_data=self.__hamiltonian_data,
            r_up_carts=self.__latest_r_up_carts,
            r_dn_carts=self.__latest_r_dn_carts,
        )
        end = time.perf_counter()
        logger.info("Compilation e_L is done.")
        logger.info(f"Elapsed Time = {end-start:.2f} sec.")

        if self.__comput_position_deriv:
            logger.info("Compilation de_L starts.")
            start = time.perf_counter()
            _, _, _ = grad(compute_local_energy, argnums=(0, 1, 2))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            end = time.perf_counter()
            logger.info("Compilation de_L is done.")
            logger.info(f"Elapsed Time = {end-start:.2f} sec.")

            logger.info("Compilation dln_Psi starts.")
            start = time.perf_counter()
            _, _, _ = grad(evaluate_ln_wavefunction_api, argnums=(0, 1, 2))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            end = time.perf_counter()
            logger.info("Compilation dln_Psi is done.")
            logger.info(f"Elapsed Time = {end-start:.2f} sec.")

            logger.info("Compilation domega starts.")
            start = time.perf_counter()
            _ = evaluate_swct_domega_api(
                self.__swct_data,
                self.__latest_r_up_carts,
            )
            end = time.perf_counter()
            logger.info("Compilation domega is done.")
            logger.info(f"Elapsed Time = {end-start:.2f} sec.")

        if self.__comput_jas_param_deriv:
            logger.info("Compilation dln_Psi starts.")
            start = time.perf_counter()
            _ = grad(evaluate_ln_wavefunction_api, argnums=(0))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            end = time.perf_counter()
            logger.info("Compilation dln_Psi is done.")
            logger.info(f"Elapsed Time = {end-start:.2f} sec.")

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
                f"  Current MCMC step = {i_mcmc_step+1+self.__mcmc_counter}/{num_mcmc_steps+self.__mcmc_counter}."
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
            logger.info(f"  e_L = {e_L}")
            self.__stored_e_L.append(e_L)

            if self.__comput_position_deriv:
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

                if jastrow_data.jastrow_three_body_flag:
                    grad_e_L_R += grad_e_L_h.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data_up_spin.structure_data.positions
                    grad_e_L_R += grad_e_L_h.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data_dn_spin.structure_data.positions

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
                logger.debug(f"de_L_dR = {grad_e_L_R}")
                logger.debug(f"de_L_dr_up = {grad_e_L_r_up}")
                logger.debug(f"de_L_dr_dn= {grad_e_L_r_dn}")
                # """

                ln_Psi = evaluate_ln_wavefunction_api(
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=self.__latest_r_up_carts,
                    r_dn_carts=self.__latest_r_dn_carts,
                )
                logger.debug(f"ln_Psi = {ln_Psi}")
                self.__stored_ln_Psi.append(ln_Psi)

                # """
                grad_ln_Psi_h, grad_ln_Psi_r_up, grad_ln_Psi_r_dn = grad(
                    evaluate_ln_wavefunction_api, argnums=(0, 1, 2)
                )(
                    self.__hamiltonian_data.wavefunction_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                logger.debug(f"dln_Psi_dr_up = {grad_ln_Psi_r_up}")
                logger.debug(f"dln_Psi_dr_dn = {grad_ln_Psi_r_dn}")
                self.__stored_grad_ln_Psi_r_up.append(grad_ln_Psi_r_up)
                self.__stored_grad_ln_Psi_r_dn.append(grad_ln_Psi_r_dn)

                grad_ln_Psi_dR = (
                    grad_ln_Psi_h.geminal_data.orb_data_up_spin.aos_data.structure_data.positions
                    + grad_ln_Psi_h.geminal_data.orb_data_dn_spin.aos_data.structure_data.positions
                )

                if jastrow_data.jastrow_three_body_flag:
                    grad_ln_Psi_dR += grad_ln_Psi_h.jastrow_data.jastrow_three_body_data.orb_data_up_spin.structure_data.positions
                    grad_ln_Psi_dR += grad_ln_Psi_h.jastrow_data.jastrow_three_body_data.orb_data_dn_spin.structure_data.positions

                # stored dln_Psi / dR
                logger.debug(f"dln_Psi_dR = {grad_ln_Psi_dR}")
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

                logger.debug(f"omega_up = {omega_up}")
                logger.debug(f"omega_dn = {omega_dn}")

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

                logger.debug(f"grad_omega_dr_up = {grad_omega_dr_up}")
                logger.debug(f"grad_omega_dr_dn = {grad_omega_dr_dn}")

                self.__stored_grad_omega_r_up.append(grad_omega_dr_up)
                self.__stored_grad_omega_r_dn.append(grad_omega_dr_dn)

            if self.__comput_jas_param_deriv:
                grad_ln_Psi_h = grad(evaluate_ln_wavefunction_api, argnums=(0))(
                    self.__hamiltonian_data.wavefunction_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_pade_flag:
                    grad_ln_Psi_jas2b = (
                        grad_ln_Psi_h.jastrow_data.jastrow_two_body_data.jastrow_2b_param
                    )
                    logger.debug(f"grad_ln_Psi_jas2b = {grad_ln_Psi_jas2b}")
                    self.__stored_grad_ln_Psi_jas2b.append(grad_ln_Psi_jas2b)

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_flag:
                    grad_ln_Psi_jas1b3b_j_matrix_up_up = (
                        grad_ln_Psi_h.jastrow_data.jastrow_three_body_data.j_matrix_up_up
                    )
                    logger.debug(
                        f"grad_ln_Psi_jas1b3b_j_matrix_up_up = {grad_ln_Psi_jas1b3b_j_matrix_up_up}"
                    )
                    self.__stored_grad_ln_Psi_jas1b3b_j_matrix_up_up.append(
                        grad_ln_Psi_jas1b3b_j_matrix_up_up
                    )
                    grad_ln_Psi_jas1b3b_j_matrix_dn_dn = (
                        grad_ln_Psi_h.jastrow_data.jastrow_three_body_data.j_matrix_dn_dn
                    )
                    logger.debug(
                        f"grad_ln_Psi_jas1b3b_j_matrix_dn_dn = {grad_ln_Psi_jas1b3b_j_matrix_dn_dn}"
                    )
                    self.__stored_grad_ln_Psi_jas1b3b_j_matrix_dn_dn.append(
                        grad_ln_Psi_jas1b3b_j_matrix_dn_dn
                    )
                    grad_ln_Psi_jas1b3b_j_matrix_up_dn = (
                        grad_ln_Psi_h.jastrow_data.jastrow_three_body_data.j_matrix_up_dn
                    )
                    logger.debug(
                        f"grad_ln_Psi_jas1b3b_j_matrix_up_dn = {grad_ln_Psi_jas1b3b_j_matrix_up_dn}"
                    )
                    self.__stored_grad_ln_Psi_jas1b3b_j_matrix_up_dn.append(
                        grad_ln_Psi_jas1b3b_j_matrix_up_dn
                    )

        self.__mcmc_counter += num_mcmc_steps
        logger.info(f"acceptance ratio is {accepted_moves/num_mcmc_steps/nbra*100} %")

    @property
    def hamiltonian_data(self):
        return self.__hamiltonian_data

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

    @property
    def dln_Psi_dc_jas_2b(self):
        return self.__stored_grad_ln_Psi_jas2b

    @property
    def dln_Psi_dc_jas_1b3b_up_up(self):
        return self.__stored_grad_ln_Psi_jas1b3b_j_matrix_up_up

    @property
    def dln_Psi_dc_jas_1b3b_up_dn(self):
        return self.__stored_grad_ln_Psi_jas1b3b_j_matrix_up_dn

    @property
    def dln_Psi_dc_jas_1b3b_dn_dn(self):
        return self.__stored_grad_ln_Psi_jas1b3b_j_matrix_dn_dn

    @property
    def domega_dr_dn(self):
        return self.__stored_grad_omega_r_dn

    @property
    def latest_r_up_carts(self):
        return self.__latest_r_up_carts

    @property
    def latest_r_dn_carts(self):
        return self.__latest_r_dn_carts

    @property
    def Dt(self):
        return self.__Dt

    @property
    def mcmc_seed(self):
        return self.__mcmc_seed

    @property
    def mcmc_counter(self):
        return self.__mcmc_counter

    @property
    def opt_param_dict(self):
        """Return a dictionary containing information about variational parameters to be optimized.

        Return:
            opt_param_list (list): instances of the parameters to be optimized.
            dln_Psi_dc_list (list): dln_Psi_dc instances computed by JAX-grad.
            dln_Psi_dc_size_list (list): sizes of dln_Psi_dc instances
            dln_Psi_dc_shape_list (list): shapes of dln_Psi_dc instances
            dln_Psi_dc_flattened_index_list (list): indices of dln_Psi_dc instances for the flattened parameter
        #
        """
        opt_param_list = []
        dln_Psi_dc_list = []
        dln_Psi_dc_size_list = []
        dln_Psi_dc_shape_list = []
        dln_Psi_dc_flattened_index_list = []

        if self.__comput_jas_param_deriv:
            if self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_pade_flag:
                opt_param = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
                dln_Psi_dc = self.dln_Psi_dc_jas_2b
                dln_Psi_dc_size = 1
                dln_Psi_dc_shape = (1,)
                dln_Psi_dc_flattened_index = [len(opt_param_list)] * dln_Psi_dc_size

                opt_param_list.append(opt_param)
                dln_Psi_dc_list.append(dln_Psi_dc)
                dln_Psi_dc_size_list.append(dln_Psi_dc_size)
                dln_Psi_dc_shape_list.append(dln_Psi_dc_shape)
                dln_Psi_dc_flattened_index_list += dln_Psi_dc_flattened_index

            if self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_flag:
                opt_param = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix_up_up
                dln_Psi_dc = self.dln_Psi_dc_jas_1b3b_up_up
                dln_Psi_dc_size = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix_up_up.size
                dln_Psi_dc_shape = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix_up_up.shape
                dln_Psi_dc_flattened_index = [len(opt_param_list)] * dln_Psi_dc_size

                opt_param_list.append(opt_param)
                dln_Psi_dc_list.append(dln_Psi_dc)
                dln_Psi_dc_size_list.append(dln_Psi_dc_size)
                dln_Psi_dc_shape_list.append(dln_Psi_dc_shape)
                dln_Psi_dc_flattened_index_list += dln_Psi_dc_flattened_index

                opt_param = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix_up_dn
                dln_Psi_dc = self.dln_Psi_dc_jas_1b3b_up_dn
                dln_Psi_dc_size = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix_up_dn.size
                dln_Psi_dc_shape = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix_up_dn.shape
                dln_Psi_dc_flattened_index = [len(opt_param_list)] * dln_Psi_dc_size

                opt_param_list.append(opt_param)
                dln_Psi_dc_list.append(dln_Psi_dc)
                dln_Psi_dc_size_list.append(dln_Psi_dc_size)
                dln_Psi_dc_shape_list.append(dln_Psi_dc_shape)
                dln_Psi_dc_flattened_index_list += dln_Psi_dc_flattened_index

                opt_param = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix_dn_dn
                dln_Psi_dc = self.dln_Psi_dc_jas_1b3b_dn_dn
                dln_Psi_dc_size = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix_dn_dn.size
                dln_Psi_dc_shape = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix_dn_dn.shape
                dln_Psi_dc_flattened_index = [len(opt_param_list)] * dln_Psi_dc_size

                opt_param_list.append(opt_param)
                dln_Psi_dc_list.append(dln_Psi_dc)
                dln_Psi_dc_size_list.append(dln_Psi_dc_size)
                dln_Psi_dc_shape_list.append(dln_Psi_dc_shape)
                dln_Psi_dc_flattened_index_list += dln_Psi_dc_flattened_index

        return {
            "opt_param_list": opt_param_list,
            "dln_Psi_dc_list": dln_Psi_dc_list,
            "dln_Psi_dc_size_list": dln_Psi_dc_size_list,
            "dln_Psi_dc_shape_list": dln_Psi_dc_shape_list,
            "dln_Psi_dc_flattened_index_list": dln_Psi_dc_flattened_index_list,
        }


class VMC:
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
        Dt_init: float = 2.0,
        comput_jas_param_deriv=False,
        comput_position_deriv=False,
    ) -> None:
        self.__comm = MPI.COMM_WORLD
        self.__rank = self.__comm.Get_rank()

        log = getLogger("jqmc")
        log.setLevel("INFO")
        stream_handler = StreamHandler()
        stream_handler.setLevel("INFO")
        handler_format = Formatter(
            f"MPI-rank={self.__rank}: %(name)s - %(levelname)s - %(lineno)d - %(message)s"
        )
        stream_handler.setFormatter(handler_format)
        log.addHandler(stream_handler)

        self.__mpi_seed = mcmc_seed * (self.__rank + 1)
        self.__num_mcmc_warmup_steps = num_mcmc_warmup_steps
        self.__num_mcmc_bin_blocks = num_mcmc_bin_blocks
        self.__comput_jas_param_deriv = comput_jas_param_deriv
        self.__comput_position_deriv = comput_position_deriv

        logger.info(f"mcmc_seed for MPI-rank={self.__rank} is {self.__mpi_seed}.")

        # set random seeds
        random.seed(self.__mpi_seed)
        np.random.seed(self.__mpi_seed)

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
            mcmc_seed=self.__mpi_seed,
            Dt=Dt_init,
            comput_jas_param_deriv=self.__comput_jas_param_deriv,
            comput_position_deriv=self.__comput_position_deriv,
        )

    def run_single_shot(self, num_mcmc_steps=0):
        if self.__rank == 0:
            logger.info(f"num_mcmc_warmup_steps={self.__num_mcmc_warmup_steps}.")
            logger.info(f"num_mcmc_bin_blocks={self.__num_mcmc_bin_blocks}.")

        # run VMC
        self.__mcmc.run(num_mcmc_steps=num_mcmc_steps)

    def run_optimize(
        self,
        num_mcmc_steps=100,
        num_opt_steps=1,
    ):
        if self.__rank == 0:
            logger.info(f"num_mcmc_warmup_steps={self.__num_mcmc_warmup_steps}.")
            logger.info(f"num_mcmc_bin_blocks={self.__num_mcmc_bin_blocks}.")
            logger.info(f"num_mcmc_steps={num_mcmc_steps}.")
            logger.info(f"num_opt_steps={num_opt_steps}.")

            logger.info(f"Optimize Jastrow 1b2b3b={self.__comput_jas_param_deriv}")

        logger.warning(f"twobody param before opt.")
        logger.warning(
            self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
        )
        self.__mcmc.run(num_mcmc_steps=num_mcmc_steps)

        f, _ = self.get_generalized_forces(mpi_broadcast=False)
        S, _ = self.get_stochastic_matrix(mpi_broadcast=False)

        if self.__rank == 0:
            var_epsilon = 1.0e-1
            I = np.eye(S.shape[0])
            S_prime = S + var_epsilon * I

            logger.info(S_prime)

            # logger.info(
            #    f"The matrix S_prime is symmetric? = {np.allclose(S_prime, S_prime.T, atol=1.0e-10)}"
            # )
            # logger.info(f"The condition number of the matrix S is {np.linalg.cond(S)}")
            # logger.info(f"The condition number of the matrix S_prime is {np.linalg.cond(S_prime)}")

            # solve Sx=f
            # X = scipy.linalg.solve(S_prime, f, assume_a="sym")
            c, lower = cho_factor(S_prime)
            X = cho_solve((c, lower), f)

        else:
            X = None

        X = self.__comm.bcast(X, root=0)
        logger.info(f"X for MPI-rank={self.__rank} is {X}")
        logger.info(f"X.shape for MPI-rank={self.__rank} is {X.shape}")

        opt_param_list = self.__mcmc.opt_param_dict["opt_param_list"]
        dln_Psi_dc_size_list = self.__mcmc.opt_param_dict["dln_Psi_dc_size_list"]
        dln_Psi_dc_shape_list = self.__mcmc.opt_param_dict["dln_Psi_dc_shape_list"]
        dln_Psi_dc_flattened_index_list = self.__mcmc.opt_param_dict[
            "dln_Psi_dc_flattened_index_list"
        ]

        logger.info(f"dln_Psi_dc_flattened_index_list={dln_Psi_dc_flattened_index_list}")

        delta = 0.01

        for ii, opt_param in enumerate(opt_param_list):
            param_shape = dln_Psi_dc_shape_list[ii]
            param_index = [i for i, v in enumerate(dln_Psi_dc_flattened_index_list) if v == ii]
            dX = X[param_index].reshape(param_shape)
            logger.info(f"dX.shape for MPI-rank={self.__rank} is {dX.shape}")

            logger.info(opt_param)
            opt_param = opt_param + delta * dX
            logger.info(opt_param)

        logger.warning(f"twobody param before opt.")
        logger.warning(
            self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
        )

        """ WIP
        for i_opt_steps in range(num_opt_steps):
            if self.__rank == 0:
                logger.info(f"i opt_steps={i_opt_steps + 1}/{num_opt_steps}.")

            # run VMC
            self.__mcmc.run(num_mcmc_steps=num_mcmc_steps)

            if self.__rank == 0:
                M = 200
                var_epsilon = 1.0e-5
                S_matrix = np.cov(O_matrix, bias=True)
                I_matrix = np.one(S_matrix.shape)
				S_prime_matrix = S_matrix + var_epsilon * I_matrix



                X_matrix_2b = X[0:size_jas_2b].reshape(shape_jas_2b)
                X_matrix_3b_up_up = xxx

                # update parameter
                delta = 1.0e-4
                jas_3b_up_up = jas_3b_up_up + delta * X_matrix_3b_up_up

                # Create new WF and Hamiltonian

            # Broadcast the updated Hamiltonian

            # Other variables
            latest_r_up_carts = self.__mcmc.latest_r_up_carts
            latest_r_dn_carts = self.__mcmc.latest_r_dn_carts
            mpi_seed = self.__mcmc.mcmc_seed
            Dt = self.__mcmc.Dt

            time.sleep(1)

            # Generate a new MCMC instance
            # No, let's
            self.__mcmc = MCMC(
                hamiltonian_data=self.hamiltonian_data,
                init_r_up_carts=latest_r_up_carts,
                init_r_dn_carts=latest_r_dn_carts,
                mcmc_seed=mpi_seed,
                Dt=Dt,
            )
        """

    def get_deriv_ln_WF(self):
        opt_param_dict = self.__mcmc.opt_param_dict

        # opt_param_list = opt_param_dict["opt_param_list"]
        dln_Psi_dc_list = opt_param_dict["dln_Psi_dc_list"]
        # dln_Psi_dc_size_list = opt_param_dict["dln_Psi_dc_size_list"]
        # dln_Psi_dc_shape_list = opt_param_dict["dln_Psi_dc_shape_list"]
        # dln_Psi_dc_flattened_index_list = opt_param_dict["dln_Psi_dc_flattened_index_list"]

        O_matrix = np.empty((self.__mcmc.mcmc_counter, 0))

        for dln_Psi_dc in dln_Psi_dc_list:
            dln_Psi_dc_flat = np.stack([arr.flatten() for arr in dln_Psi_dc], axis=0)
            O_matrix = np.hstack([O_matrix, dln_Psi_dc_flat])

        return O_matrix  # O.... (x....) M * L matrix

    def get_generalized_forces(self, mpi_broadcast=True):
        e_L = self.__mcmc.e_L[self.__num_mcmc_warmup_steps :]
        e_L_split = np.array_split(e_L, self.__num_mcmc_bin_blocks)
        e_L_binned = [np.average(e_list) for e_list in e_L_split]

        logger.info(
            f"[before reduce] len(e_L_binned) for MPI-rank={self.__rank} is {len(e_L_binned)}"
        )

        e_L_binned = self.__comm.reduce(e_L_binned, op=MPI.SUM, root=0)

        if self.__rank == 0:
            logger.info(
                f"[before reduce] len(e_L_binned) for MPI-rank={self.__rank} is {len(e_L_binned)}"
            )

        O_matrix = self.get_deriv_ln_WF()
        O_matrix_split = np.array_split(O_matrix, self.__num_mcmc_bin_blocks)
        O_matrix_binned = [np.average(O_matrix_list, axis=0) for O_matrix_list in O_matrix_split]

        logger.info(f"[before reduce] O_matrix_binned.shape = {np.array(O_matrix_binned).shape}")

        O_matrix_binned = self.__comm.reduce(O_matrix_binned, op=MPI.SUM, root=0)

        if self.__rank == 0:
            logger.info(f"[after reduce] O_matrix_binned.shape = {np.array(O_matrix_binned).shape}")

            e_L_binned = np.array(e_L_binned)
            O_matrix_binned = np.array(O_matrix_binned)

            eL_O_matrix_binned = np.einsum("i,ij->ij", e_L_binned, O_matrix_binned)

            logger.info(f"eL_O_matrix_binned.shape = {eL_O_matrix_binned.shape}")

            M = self.__num_mcmc_bin_blocks * self.__comm.size

            eL_O_jn = (
                1.0
                / (M - 1)
                * np.array(
                    [np.sum(eL_O_matrix_binned, axis=0) - eL_O_matrix_binned[j] for j in range(M)]
                )
            )

            logger.info(f"eL_O_jn.shape = {eL_O_jn.shape}")

            eL_jn = np.array([np.sum(e_L_binned, axis=0) - e_L_binned[j] for j in range(M)])

            logger.info(f"eL_jn.shape = {eL_jn.shape}")

            O_jn = (
                1.0
                / (M - 1)
                * np.array([np.sum(O_matrix_binned, axis=0) - O_matrix_binned[j] for j in range(M)])
            )

            logger.info(f"O_jn.shape = {O_jn.shape}")

            eL_barO_jn = 1.0 / (M - 1) * np.einsum("i,ij->ij", eL_jn, O_jn)

            logger.info(f"eL_barO_jn.shape = {eL_barO_jn.shape}")

            generalized_force_mean = np.average(-2.0 * (eL_O_jn - eL_barO_jn), axis=0)
            generalized_force_std = np.sqrt(M - 1) * np.std(-2.0 * (eL_O_jn - eL_barO_jn), axis=0)

            logger.info(f"generalized_force_mean = {generalized_force_mean}")
            logger.info(f"generalized_force_std = {generalized_force_std}")

            logger.info(f"generalized_force_mean.shape = {generalized_force_mean.shape}")
            logger.info(f"generalized_force_std.shape = {generalized_force_std.shape}")

        else:
            generalized_force_mean = None
            generalized_force_std = None

        if mpi_broadcast:
            # self.__comm.Bcast(generalized_force_mean, root=0)
            # self.__comm.Bcast(generalized_force_std, root=0)
            generalized_force_mean = self.__comm.bcast(generalized_force_mean, root=0)
            generalized_force_std = self.__comm.bcast(generalized_force_std, root=0)

        return (generalized_force_mean, generalized_force_std)  # (L vector, L vector)

    def get_stochastic_matrix(self, mpi_broadcast=False):
        O_matrix = self.get_deriv_ln_WF()
        O_matrix_split = np.array_split(O_matrix, self.__num_mcmc_bin_blocks)
        O_matrix_binned = [np.average(O_matrix_list, axis=0) for O_matrix_list in O_matrix_split]
        O_matrix_binned = self.__comm.reduce(O_matrix_binned, op=MPI.SUM, root=0)

        if self.__rank == 0:
            O_matrix_binned = np.array(O_matrix_binned)
            logger.info(f"O_matrix_binned.shape = {O_matrix_binned.shape}")
            S_mean = np.array(np.cov(O_matrix_binned, bias=True, rowvar=False))
            S_std = np.zeros(S_mean.size)
            logger.info(f"S_mean = {S_mean}")
            logger.info(f"S_mean.is_nan for MPI-rank={self.__rank} is {np.isnan(S_mean).any()}")
            logger.info(f"S_mean.shape for MPI-rank={self.__rank} is {S_mean.shape}")
            logger.info(f"S_mean.type for MPI-rank={self.__rank} is {type(S_mean)}")
        else:
            S_mean = None
            S_std = None

        if mpi_broadcast:
            # self.__comm.Bcast(S_mean, root=0)
            # self.__comm.Bcast(S_std, root=0)
            S_mean = self.__comm.bcast(S_mean, root=0)
            S_std = self.__comm.bcast(S_std, root=0)

        return (S_mean, S_std)  # (S_mu,nu ...., var(S)_mu,nu....) (L*L matrix, L*L matrix)

    def get_e_L(self):
        # analysis VMC
        e_L = self.__mcmc.e_L[self.__num_mcmc_warmup_steps :]
        e_L_split = np.array_split(e_L, self.__num_mcmc_bin_blocks)
        e_L_binned = [np.average(e_list) for e_list in e_L_split]

        logger.info(
            f"[before reduce] len(e_L_binned) for MPI-rank={self.__rank} is {len(e_L_binned)}."
        )

        e_L_binned = self.__comm.reduce(e_L_binned, op=MPI.SUM, root=0)

        if self.__rank == 0:
            logger.info(
                f"[after reduce] len(e_L_binned) for MPI-rank={self.__rank} is {len(e_L_binned)}."
            )
            logger.debug(f"e_L_binned = {e_L_binned}.")
            # jackknife implementation
            # https://www2.yukawa.kyoto-u.ac.jp/~etsuko.itou/old-HP/Notes/Jackknife-method.pdf
            e_L_jackknife_binned = [
                np.average(np.delete(e_L_binned, i)) for i in range(len(e_L_binned))
            ]

            logger.info(f"len(e_L_jackknife_binned)  = {len(e_L_jackknife_binned)}.")

            e_L_mean = np.average(e_L_jackknife_binned)
            e_L_std = np.sqrt(len(e_L_binned) - 1) * np.std(e_L_jackknife_binned)

            logger.info(f"e_L = {e_L_mean} +- {e_L_std} Ha.")
        else:
            e_L_mean = 0.0
            e_L_std = 0.0

        e_L_mean = self.__comm.bcast(e_L_mean, root=0)
        e_L_std = self.__comm.bcast(e_L_std, root=0)

        return (e_L_mean, e_L_std)

    def get_atomic_forces(self):
        if not self.__comput_position_deriv:
            force_mean = np.array([])
            force_std = np.array([])
            return (force_mean, force_std)

        else:
            e_L = np.array(self.__mcmc.e_L[self.__num_mcmc_warmup_steps :])
            de_L_dR = np.array(self.__mcmc.de_L_dR[self.__num_mcmc_warmup_steps :])
            de_L_dr_up = np.array(self.__mcmc.de_L_dr_up[self.__num_mcmc_warmup_steps :])
            de_L_dr_dn = np.array(self.__mcmc.de_L_dr_dn[self.__num_mcmc_warmup_steps :])
            dln_Psi_dr_up = np.array(self.__mcmc.dln_Psi_dr_up[self.__num_mcmc_warmup_steps :])
            dln_Psi_dr_dn = np.array(self.__mcmc.dln_Psi_dr_dn[self.__num_mcmc_warmup_steps :])
            dln_Psi_dR = np.array(self.__mcmc.dln_Psi_dR[self.__num_mcmc_warmup_steps :])
            omega_up = np.array(self.__mcmc.omega_up[self.__num_mcmc_warmup_steps :])
            omega_dn = np.array(self.__mcmc.omega_dn[self.__num_mcmc_warmup_steps :])
            domega_dr_up = np.array(self.__mcmc.domega_dr_up[self.__num_mcmc_warmup_steps :])
            domega_dr_dn = np.array(self.__mcmc.domega_dr_dn[self.__num_mcmc_warmup_steps :])

            force_HF = (
                de_L_dR
                + np.einsum("ijk,ikl->ijl", omega_up, de_L_dr_up)
                + np.einsum("ijk,ikl->ijl", omega_dn, de_L_dr_dn)
            )

            force_PP = (
                dln_Psi_dR
                + np.einsum("ijk,ikl->ijl", omega_up, dln_Psi_dr_up)
                + np.einsum("ijk,ikl->ijl", omega_dn, dln_Psi_dr_dn)
                + 1.0 / 2.0 * (domega_dr_up + domega_dr_dn)
            )

            E_L_force_PP = np.einsum("i,ijk->ijk", e_L, force_PP)

            logger.info(f"e_L.shape for MPI-rank={self.__rank} is {e_L.shape}")
            logger.info(f"force_HF.shape for MPI-rank={self.__rank} is {force_HF.shape}")
            logger.info(f"force_PP.shape for MPI-rank={self.__rank} is {force_PP.shape}")
            logger.info(f"E_L_force_PP.shape for MPI-rank={self.__rank} is {E_L_force_PP.shape}")

            e_L_split = np.array_split(e_L, self.__num_mcmc_bin_blocks)
            force_HF_split = np.array_split(force_HF, self.__num_mcmc_bin_blocks)
            force_PP_split = np.array_split(force_PP, self.__num_mcmc_bin_blocks)
            E_L_force_PP_split = np.array_split(E_L_force_PP, self.__num_mcmc_bin_blocks)

            e_L_binned = [np.average(A, axis=0) for A in e_L_split]
            force_HF_binned = [np.average(A, axis=0) for A in force_HF_split]
            force_PP_binned = [np.average(A, axis=0) for A in force_PP_split]
            E_L_force_PP_binned = [np.average(A, axis=0) for A in E_L_force_PP_split]

            e_L_binned = self.__comm.reduce(e_L_binned, op=MPI.SUM, root=0)
            force_HF_binned = self.__comm.reduce(force_HF_binned, op=MPI.SUM, root=0)
            force_PP_binned = self.__comm.reduce(force_PP_binned, op=MPI.SUM, root=0)
            E_L_force_PP_binned = self.__comm.reduce(E_L_force_PP_binned, op=MPI.SUM, root=0)

            if self.__rank == 0:
                e_L_binned = np.array(e_L_binned)
                force_HF_binned = np.array(force_HF_binned)
                force_PP_binned = np.array(force_PP_binned)
                E_L_force_PP_binned = np.array(E_L_force_PP_binned)

                logger.info(f"e_L_binned.shape for MPI-rank={self.__rank} is {e_L_binned.shape}")
                logger.info(
                    f"force_HF_binned.shape for MPI-rank={self.__rank} is {force_HF_binned.shape}"
                )
                logger.info(
                    f"force_PP_binned.shape for MPI-rank={self.__rank} is {force_PP_binned.shape}"
                )
                logger.info(
                    f"E_L_force_PP_binned.shape for MPI-rank={self.__rank} is {E_L_force_PP_binned.shape}"
                )

                M = self.__num_mcmc_bin_blocks * self.__comm.size

                force_HF_jn = np.array(
                    [
                        -1.0 / (M - 1) * (np.sum(force_HF_binned, axis=0) - force_HF_binned[j])
                        for j in range(M)
                    ]
                )

                force_Pulay_jn = np.array(
                    [
                        -2.0
                        / (M - 1)
                        * (
                            (np.sum(E_L_force_PP_binned, axis=0) - E_L_force_PP_binned[j])
                            - (
                                1.0
                                / (M - 1)
                                * (np.sum(e_L_binned) - e_L_binned[j])
                                * (np.sum(force_PP_binned, axis=0) - force_PP_binned[j])
                            )
                        )
                        for j in range(M)
                    ]
                )

                logger.info(f"force_HF_jn.shape for MPI-rank={self.__rank} is {force_HF_jn.shape}")
                logger.info(
                    f"force_Pulay_jn.shape for MPI-rank={self.__rank} is {force_Pulay_jn.shape}"
                )

                force_jn = force_HF_jn + force_Pulay_jn

                force_mean = np.average(force_jn, axis=0)
                force_std = np.sqrt(M - 1) * np.std(force_jn, axis=0)

                logger.info(f"force_mean.shape  = {force_mean.shape}.")
                logger.info(f"force_std.shape  = {force_std.shape}.")

                logger.info(f"force = {force_mean} +- {force_std} Ha.")

            else:
                force_mean = np.array([])
                force_std = np.array([])

            force_mean = self.__comm.bcast(force_mean, root=0)
            force_std = self.__comm.bcast(force_std, root=0)

            return (force_mean, force_std)

        """
        else:
            # analysis VMC

            # todo!! I do not think it's true. e_L_mean should be computed differently
            # for each bin in the jackknife sampling.
            # no? because jackknife mean = true mean, as shown in the O_k calc.
            # Let's think about it tomorrow.
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
            logger.info(f"force.shape for MPI-rank={self.__rank} is {force.shape}")

            force_split = np.array_split(force, self.__num_mcmc_bin_blocks)
            # force_binned = np.array([np.average(force_list, axis=0) for force_list in force_split]) # ?
            force_binned = [np.average(force_list, axis=0) for force_list in force_split]
            logger.info(
                f"[before reduce] force_binned.shape for MPI-rank={self.__rank} is {np.array(force_binned).shape}"
            )

            force_binned = self.__comm.reduce(force_binned, op=MPI.SUM, root=0)

            if self.__rank == 0:
                logger.info(
                    f"[after reduce] force_binned.shape for MPI-rank={self.__rank} is {np.array(force_binned).shape}"
                )
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

            if mpi_broadcast:
                force_mean = self.__comm.bcast(force_mean, root=0)
                force_std = self.__comm.bcast(force_std, root=0)

            return (force_mean, force_std)
        """


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
    ) = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(__file__), "trexio_files", "water_ccpvtz_trexio.hdf5"
        )
    )
    # """

    """
    # H2 dimer cc-pV5Z with Mitas ccECP (2 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(__file__), "trexio_files", "H2_dimer_ccpv5z_trexio.hdf5"
        )
    )
    """

    """
    # Ne atom cc-pV5Z with Mitas ccECP (10 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "Ne_ccpv5z_trexio.hdf5")
    )
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
    ) = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(__file__), "trexio_files", "benzene_ccpvdz_trexio.hdf5"
        )
    )
    """

    """
    # benzene cc-pV6Z with Mitas ccECP (30 electrons, slow, but feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(__file__), "trexio_files", "benzene_ccpv6z_trexio.hdf5"
        )
    )
    """

    """
    # AcOH-AcOH dimer aug-cc-pV6Z with Mitas ccECP (48 electrons, slow, but feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(__file__), "trexio_files", "AcOH_dimer_augccpv6z.hdf5"
        )
    )
    """

    """
    # benzene dimer cc-pV6Z with Mitas ccECP (60 electrons, not feasible, why?).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(__file__), "trexio_files", "benzene_dimer_ccpv6z_trexio.hdf5"
        )
    )
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
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "C60_ccpvtz_trexio.hdf5")
    )
    """

    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data_up_spin=aos_data, orb_data_dn_spin=aos_data
    )

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

    # VMC parameters
    num_mcmc_warmup_steps = 20
    num_mcmc_bin_blocks = 5
    mcmc_seed = 34356

    # run VMC
    vmc = VMC(
        hamiltonian_data=hamiltonian_data,
        mcmc_seed=mcmc_seed,
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
        comput_position_deriv=False,
        comput_jas_param_deriv=True,
    )
    # vmc.run_single_shot(num_mcmc_steps=50)
    # vmc.get_e_L()
    # vmc.get_atomic_forces()
    vmc.run_optimize(num_mcmc_steps=100, num_opt_steps=1)
    # vmc.get_generalized_forces(mpi_broadcast=False)
    # vmc.get_stochastic_matrix(mpi_broadcast=False)
