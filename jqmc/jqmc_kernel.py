"""QMC module."""

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
from functools import partial
from logging import getLogger

import jax
import numpy as np
import numpy.typing as npt
import scipy
from jax import grad, jit, lax, vmap
from jax import numpy as jnp
from mpi4py import MPI

from .coulomb_potential import (
    compute_bare_coulomb_potential_el_el_jax,
    compute_bare_coulomb_potential_el_ion_element_wise_jax,
    compute_bare_coulomb_potential_ion_ion_jax,
    compute_bare_coulomb_potential_jax,
    compute_discretized_bare_coulomb_potential_el_ion_element_wise_jax,
    compute_ecp_local_parts_all_pairs_jax,
    compute_ecp_non_local_parts_nearest_neighbors_jax,
)
from .determinant import Geminal_data, compute_AS_regularization_factor_jax, compute_det_geminal_all_elements_jax
from .hamiltonians import (
    Hamiltonian_data,  # Hamiltonian_data_deriv_R,
    Hamiltonian_data_deriv_params,
    compute_kinetic_energy_jax,
    compute_local_energy_jax,
)
from .jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    compute_Jastrow_part_jax,
)
from .structure import find_nearest_index_jax
from .swct import SWCT_data, evaluate_swct_domega_jax, evaluate_swct_omega_jax
from .wavefunction import (
    Wavefunction_data,
    compute_discretized_kinetic_energy_jax,
    compute_kinetic_energy_all_elements_jax,
    evaluate_ln_wavefunction_jax,
)

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

# separator
num_sep_line = 66


class MCMC:
    """MCMC with multiple walker class.

    MCMC class. Runing MCMC with multiple walkers. The independent 'num_walkers' MCMCs are
    vectrized via the jax-vmap function.

    Args:
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data.
        mcmc_seed (int): seed for the MCMC chain.
        num_walkers (int): the number of walkers.
        num_mcmc_per_measurement (int): the number of MCMC steps between a value (e.g., local energy) measurement.
        Dt (float): electron move step (bohr)
        epsilon_AS (float): the exponent of the AS regularization
        # adjust_epsilon_AS (bool): if True, adjust epsilon_AS to keep the average of weights close to target_weight (~0.8).
        comput_param_deriv (bool): if True, compute the derivatives of E wrt. variational parameters.
        comput_position_deriv (bool): if True, compute the derivatives of E wrt. atomic positions.
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data = None,
        mcmc_seed: int = 34467,
        num_walkers: int = 40,
        num_mcmc_per_measurement: int = 16,
        Dt: float = 2.0,
        epsilon_AS: float = 1e-1,
        # adjust_epsilon_AS: bool = False,
        comput_param_deriv: bool = False,
        comput_position_deriv: bool = False,
    ) -> None:
        """Initialize a MCMC class, creating list holding results."""
        self.__mcmc_seed = mcmc_seed
        self.__num_walkers = num_walkers
        self.__num_mcmc_per_measurement = num_mcmc_per_measurement
        self.__Dt = Dt
        self.__epsilon_AS = epsilon_AS
        # self.__adjust_epsilon_AS = adjust_epsilon_AS
        self.__comput_param_deriv = comput_param_deriv
        self.__comput_position_deriv = comput_position_deriv

        # check sanity of hamiltonian_data
        hamiltonian_data.sanity_check()

        # set hamiltonian_data
        self.__hamiltonian_data = hamiltonian_data

        # seeds
        self.__mpi_seed = self.__mcmc_seed * (mpi_rank + 1)
        self.__jax_PRNG_key = jax.random.PRNGKey(self.__mpi_seed)
        self.__jax_PRNG_key_list = jnp.array([jax.random.fold_in(self.__jax_PRNG_key, nw) for nw in range(self.__num_walkers)])

        # timer
        self.__timer_mcmc_total = 0.0
        self.__timer_mcmc_init = 0.0
        self.__timer_mcmc_update_init = 0.0
        self.__timer_mcmc_update = 0.0
        self.__timer_e_L = 0.0
        self.__timer_de_L_dR_dr = 0.0
        self.__timer_dln_Psi_dR_dr = 0.0
        self.__timer_dln_Psi_dc = 0.0
        self.__timer_de_L_dc = 0.0
        self.__timer_misc = 0.0

        # Place electrons around each nucleus with improved spin assignment

        tot_num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        tot_num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn

        r_carts_up_list = []
        r_carts_dn_list = []

        np.random.seed(self.__mpi_seed)

        for _ in range(self.__num_walkers):
            # Initialization
            r_carts_up = []
            r_carts_dn = []
            total_assigned_up = 0
            total_assigned_dn = 0

            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
                    hamiltonian_data.coulomb_potential_data.z_cores
                )
            else:
                charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

            logger.devel(f"charges = {charges}.")
            coords = hamiltonian_data.structure_data.positions_cart_jnp

            # Place electrons for each atom
            for i in range(len(coords)):
                charge = charges[i]
                n_elec = int(np.round(charge))  # Number of electrons to place for this atom
                logger.devel(f"charge = {charge}.")
                logger.devel(f"num_electrons = {n_elec}.")

                x, y, z = coords[i]

                # Determine the number of up and dn electrons for this atom
                if n_elec % 2 == 0:
                    num_up = n_elec // 2
                    num_dn = n_elec // 2
                else:
                    # For odd numbers, decide which spin gets the extra electron based on the overall shortage
                    remaining_up = tot_num_electron_up - total_assigned_up
                    remaining_dn = tot_num_electron_dn - total_assigned_dn
                    if remaining_up >= remaining_dn:
                        num_up = n_elec // 2 + 1
                        num_dn = n_elec // 2
                    else:
                        num_up = n_elec // 2
                        num_dn = n_elec // 2 + 1

                # Place up electrons
                for _ in range(num_up):
                    distance = np.random.uniform(0.1, 1.0)
                    theta = np.random.uniform(0, np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    dx = distance * np.sin(theta) * np.cos(phi)
                    dy = distance * np.sin(theta) * np.sin(phi)
                    dz = distance * np.cos(theta)
                    electron_position = np.array([x + dx, y + dy, z + dz])
                    r_carts_up.append(electron_position)
                total_assigned_up += num_up

                # Place dn electrons
                for _ in range(num_dn):
                    distance = np.random.uniform(0.1, 1.0)
                    theta = np.random.uniform(0, np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    dx = distance * np.sin(theta) * np.cos(phi)
                    dy = distance * np.sin(theta) * np.sin(phi)
                    dz = distance * np.cos(theta)
                    electron_position = np.array([x + dx, y + dy, z + dz])
                    r_carts_dn.append(electron_position)
                total_assigned_dn += num_dn

            # Electron assignment for all atoms is complete
            logger.devel(f"Total assigned up electrons: {total_assigned_up} (target {tot_num_electron_up}).")
            logger.devel(f"Total assigned dn electrons: {total_assigned_dn} (target {tot_num_electron_dn}).")

            # If necessary, include a check/adjustment step to ensure the overall assignment matches the targets
            # (Here it is assumed that sum(round(charge)) equals tot_num_electron_up + tot_num_electron_dn)

            r_carts_up_list.append(r_carts_up)
            r_carts_dn_list.append(r_carts_dn)

        self.__latest_r_up_carts = jnp.array(r_carts_up_list)
        self.__latest_r_dn_carts = jnp.array(r_carts_dn_list)

        logger.info(f"The number of MPI process = {mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self.__num_walkers}.")

        logger.devel(f"initial r_up_carts= {self.__latest_r_up_carts}")
        logger.devel(f"initial r_dn_carts = {self.__latest_r_dn_carts}")
        logger.devel(f"initial r_up_carts.shape = {self.__latest_r_up_carts.shape}")
        logger.devel(f"initial r_dn_carts.shape = {self.__latest_r_dn_carts.shape}")
        logger.info("")

        # print out hamiltonian info
        logger.info("Printing out information in hamitonian_data instance.")
        self.__hamiltonian_data.logger_info()
        logger.info("")

        # SWCT data
        self.__swct_data = SWCT_data(structure=self.__hamiltonian_data.structure_data)

        # compiling methods
        logger.info("Compilation of fundamental functions starts.")

        logger.info("  Compilation e_L starts.")
        start = time.perf_counter()
        _ = compute_local_energy_jax(
            hamiltonian_data=self.__hamiltonian_data,
            r_up_carts=self.__latest_r_up_carts[0],
            r_dn_carts=self.__latest_r_dn_carts[0],
        )
        end = time.perf_counter()
        logger.info("  Compilation e_L is done.")
        logger.info(f"  Elapsed Time = {end - start:.2f} sec.")
        self.__timer_mcmc_init += end - start

        if self.__comput_position_deriv:
            logger.info("  Compilation de_L/dR starts.")
            start = time.perf_counter()
            _, _, _ = grad(compute_local_energy_jax, argnums=(0, 1, 2))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts[0],
                self.__latest_r_dn_carts[0],
            )
            end = time.perf_counter()
            logger.info("  Compilation de_L/dR is done.")
            logger.info(f"  Elapsed Time = {end - start:.2f} sec.")
            self.__timer_mcmc_init += end - start

            logger.info("  Compilation dln_Psi/dR starts.")
            start = time.perf_counter()
            _, _, _ = grad(evaluate_ln_wavefunction_jax, argnums=(0, 1, 2))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts[0],
                self.__latest_r_dn_carts[0],
            )
            end = time.perf_counter()
            logger.info("  Compilation dln_Psi/dR is done.")
            logger.info(f"  Elapsed Time = {end - start:.2f} sec.")
            self.__timer_mcmc_init += end - start

            logger.info("  Compilation domega/dR starts.")
            start = time.perf_counter()
            _ = evaluate_swct_domega_jax(
                self.__swct_data,
                self.__latest_r_up_carts[0],
            )
            end = time.perf_counter()
            logger.info("  Compilation domega/dR is done.")
            logger.info(f"  Elapsed Time = {end - start:.2f} sec.")
            self.__timer_mcmc_init += end - start

        if self.__comput_param_deriv:
            logger.info("  Compilation dln_Psi/dc starts.")
            start = time.perf_counter()
            _ = grad(evaluate_ln_wavefunction_jax, argnums=(0))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts[0],
                self.__latest_r_dn_carts[0],
            )
            end = time.perf_counter()
            logger.info("  Compilation dln_Psi/dc is done.")
            logger.info(f"  Elapsed Time = {end - start:.2f} sec.")
            self.__timer_mcmc_init += end - start

            """ for linear method
            logger.info("  Compilation de_L/dc starts.")
            start = time.perf_counter()
            _ = grad(compute_local_energy_api, argnums=0)(
                self.__hamiltonian_data,
                self.__latest_r_up_carts[0],
                self.__latest_r_dn_carts[0],
            )
            end = time.perf_counter()
            logger.info("  Compilation de_L/dc is done.")
            logger.info(f"  Elapsed Time = {end - start:.2f} sec.")
            self.__timer_mcmc_init += end - start
            """

        logger.info("Compilation of fundamental functions is done.")
        logger.info(f"Elapsed Time = {self.__timer_mcmc_init:.2f} sec.")
        logger.info("")

        # init_attributes
        self.hamiltonian_data = self.__hamiltonian_data
        self.__init_attributes()

    def __init_attributes(self):
        # mcmc counter
        self.__mcmc_counter = 0

        # mcmc accepted/rejected moves
        self.__accepted_moves = 0
        self.__rejected_moves = 0

        # stored weight (w_L)
        self.__stored_w_L = []

        # stored local energy (e_L)
        self.__stored_e_L = []

        # stored local energy (e_L2)
        self.__stored_e_L2 = []

        # stored de_L / dR
        self.__stored_grad_e_L_dR = []

        # stored de_L / dr_up
        self.__stored_grad_e_L_r_up = []

        # stored de_L / dr_dn
        self.__stored_grad_e_L_r_dn = []

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

        # stored dln_Psi / dc_jas1b
        self.__stored_grad_ln_Psi_jas1b = []

        # stored dln_Psi / dc_jas2b
        self.__stored_grad_ln_Psi_jas2b = []

        # stored dln_Psi / dc_jas1b3b
        self.__stored_grad_ln_Psi_jas1b3b_j_matrix = []

        """ linear method
        # stored de_L / dc_jas2b
        self.__stored_grad_e_L_jas2b = []

        # stored de_L / dc_jas1b3b
        self.__stored_grad_e_L_jas1b3b_j_matrix = []
        """

        # stored dln_Psi / dc_lambda_matrix
        self.__stored_grad_ln_Psi_lambda_matrix = []

        """ linear method
        # stored de_L / dc_lambda_matrix
        self.__stored_grad_e_L_lambda_matrix = []
        """

    def run(self, num_mcmc_steps: int = 0, max_time=86400) -> None:
        """Launch MCMCs with the set multiple walkers.

        Args:
            num_mcmc_steps (int): the number of total mcmc steps per walker.
            max_time(int):
                Max elapsed time (sec.). If the elapsed time exceeds max_time, the methods exits the mcmc loop.
        """
        # timer_counter
        timer_mcmc_total = 0.0
        timer_mcmc_update_init = 0.0
        timer_mcmc_update = 0.0
        timer_e_L = 0.0
        timer_de_L_dR_dr = 0.0
        timer_dln_Psi_dR_dr = 0.0
        timer_dln_Psi_dc = 0.0
        timer_de_L_dc = 0.0
        mcmc_total_start = time.perf_counter()

        # MCMC electron position update function
        mcmc_update_init_start = time.perf_counter()
        logger.info("Start compilation of the MCMC_update funciton.")

        # Note: This jit drastically accelarates the computation!!
        @partial(jit, static_argnums=3)
        def _update_electron_positions(
            init_r_up_carts, init_r_dn_carts, jax_PRNG_key, num_mcmc_per_measurement, hamiltonian_data, Dt, epsilon_AS
        ):
            """Update electron positions based on the MH method.

            Args:
                init_r_up_carts (jnpt.ArrayLike): up electron position. dim: (N_e^up, 3)
                init_r_dn_carts (jnpt.ArrayLike): down electron position. dim: (N_e^dn, 3)
                jax_PRNG_key (jnpt.ArrayLike): jax PRIN key.
                num_mcmc_per_measurement (int): the number of iterarations (i.e. the number of proposal in updating electron positions.)
                hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data.
                Dt (float): the step size in the MH method.
                epsilon_AS (float): the exponent of the AS regularization.

            Returns:
                jax_PRNG_key (jnpt.ArrayLike): updated jax_PRNG_key.
                accepted_moves (int): the number of accepted moves
                rejected_moves (int): the number of rejected moves
                updated_r_up_cart (jnpt.ArrayLike): up electron position. dim: (N_e^up, 3)
                updated_r_dn_cart (jnpt.ArrayLike): down electron position. dim: (N_e^down, 3)
            """
            accepted_moves = 0
            rejected_moves = 0
            r_up_carts = init_r_up_carts
            r_dn_carts = init_r_dn_carts

            def body_fun(_, carry):
                accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key = carry
                total_electrons = len(r_up_carts) + len(r_dn_carts)

                # Choose randomly if the electron comes from up or dn
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                rand_num = jax.random.randint(subkey, shape=(), minval=0, maxval=total_electrons)

                # boolen: "up" or "dn"
                # is_up == True -> up、False -> dn
                is_up = rand_num < len(r_up_carts)

                # an index chosen from up electons
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                up_index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(r_up_carts))

                # an index chosen from dn electrons
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                dn_index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(r_dn_carts))

                selected_electron_index = jnp.where(is_up, up_index, dn_index)

                # choose an up or dn electron from old_r_cart
                old_r_cart = jnp.where(is_up, r_up_carts[selected_electron_index], r_dn_carts[selected_electron_index])

                # choose the nearest atom index
                nearest_atom_index = find_nearest_index_jax(hamiltonian_data.structure_data, old_r_cart)

                # charges
                if hamiltonian_data.coulomb_potential_data.ecp_flag:
                    charges = jnp.array(hamiltonian_data.structure_data.atomic_numbers) - jnp.array(
                        hamiltonian_data.coulomb_potential_data.z_cores
                    )
                else:
                    charges = jnp.array(hamiltonian_data.structure_data.atomic_numbers)

                # coords
                coords = hamiltonian_data.structure_data.positions_cart_jnp

                R_cart = coords[nearest_atom_index]
                Z = charges[nearest_atom_index]
                norm_r_R = jnp.linalg.norm(old_r_cart - R_cart)
                f_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

                logger.devel(f"nearest_atom_index = {nearest_atom_index}")
                logger.devel(f"norm_r_R = {norm_r_R}")
                logger.devel(f"f_l  = {f_l}")

                sigma = f_l * Dt
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                g = jax.random.normal(subkey, shape=()) * sigma

                # choose x,y,or,z
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                random_index = jax.random.randint(subkey, shape=(), minval=0, maxval=3)

                # plug g into g_vector
                g_vector = jnp.zeros(3)
                g_vector = g_vector.at[random_index].set(g)

                logger.devel(f"jn = {random_index}, g \\equiv dstep  = {g_vector}")
                new_r_cart = old_r_cart + g_vector

                # set proposed r_up_carts and r_dn_carts.
                proposed_r_up_carts = lax.cond(
                    is_up,
                    lambda _: r_up_carts.at[selected_electron_index].set(new_r_cart),
                    lambda _: r_up_carts,
                    operand=None,
                )

                proposed_r_dn_carts = lax.cond(
                    is_up,
                    lambda _: r_dn_carts,
                    lambda _: r_dn_carts.at[selected_electron_index].set(new_r_cart),
                    operand=None,
                )

                # choose the nearest atom index
                nearest_atom_index = find_nearest_index_jax(hamiltonian_data.structure_data, new_r_cart)

                R_cart = coords[nearest_atom_index]
                Z = charges[nearest_atom_index]
                norm_r_R = jnp.linalg.norm(new_r_cart - R_cart)
                f_prime_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

                logger.devel(f"nearest_atom_index = {nearest_atom_index}")
                logger.devel(f"norm_r_R = {norm_r_R}")
                logger.devel(f"f_prime_l  = {f_prime_l}")

                logger.devel(f"The selected electron is {selected_electron_index + 1}-th {is_up} electron.")
                logger.devel(f"The selected electron position is {old_r_cart}.")
                logger.devel(f"The proposed electron position is {new_r_cart}.")

                T_ratio = (f_l / f_prime_l) * jnp.exp(
                    -(jnp.linalg.norm(new_r_cart - old_r_cart) ** 2)
                    * (1.0 / (2.0 * f_prime_l**2 * Dt**2) - 1.0 / (2.0 * f_l**2 * Dt**2))
                )

                # original trial WFs
                Jastrow_T_p = compute_Jastrow_part_jax(
                    jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                    r_up_carts=proposed_r_up_carts,
                    r_dn_carts=proposed_r_dn_carts,
                )

                Jastrow_T_o = compute_Jastrow_part_jax(
                    jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

                Det_T_p = compute_det_geminal_all_elements_jax(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    r_up_carts=proposed_r_up_carts,
                    r_dn_carts=proposed_r_dn_carts,
                )

                Det_T_o = compute_det_geminal_all_elements_jax(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

                # compute AS regularization factors, R_AS and R_AS_eps
                R_AS_p = compute_AS_regularization_factor_jax(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    r_up_carts=proposed_r_up_carts,
                    r_dn_carts=proposed_r_dn_carts,
                )
                R_AS_p_eps = jnp.maximum(R_AS_p, epsilon_AS)

                R_AS_o = compute_AS_regularization_factor_jax(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
                R_AS_o_eps = jnp.maximum(R_AS_o, epsilon_AS)

                # modified trial WFs
                R_AS_ratio = (R_AS_p_eps / R_AS_p) / (R_AS_o_eps / R_AS_o)
                WF_ratio = jnp.exp(Jastrow_T_p - Jastrow_T_o) * (Det_T_p / Det_T_o)

                # compute R_ratio
                R_ratio = (R_AS_ratio * WF_ratio) ** 2.0

                logger.devel(f"R_ratio, T_ratio = {R_ratio}, {T_ratio}")
                acceptance_ratio = jnp.min(jnp.array([1.0, R_ratio * T_ratio]))
                logger.devel(f"acceptance_ratio = {acceptance_ratio}")

                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                b = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)
                logger.devel(f"b = {b}.")

                def _accepted_fun(_):
                    # Move accepted
                    return (accepted_moves + 1, rejected_moves, proposed_r_up_carts, proposed_r_dn_carts)

                def _rejected_fun(_):
                    # Move rejected
                    return (accepted_moves, rejected_moves + 1, r_up_carts, r_dn_carts)

                # judge accept or reject the propsed move using jax.lax.cond
                accepted_moves, rejected_moves, r_up_carts, r_dn_carts = lax.cond(
                    b < acceptance_ratio, _accepted_fun, _rejected_fun, operand=None
                )

                carry = (accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key)
                return carry

            accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key = jax.lax.fori_loop(
                0, num_mcmc_per_measurement, body_fun, (accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key)
            )

            return (accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key)

        # MCMC update compilation.
        logger.info("  Compilation is in progress...")
        (
            _,
            _,
            _,
            _,
            _,
        ) = vmap(_update_electron_positions, in_axes=(0, 0, 0, None, None, None, None))(
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
            self.__jax_PRNG_key_list,
            self.__num_mcmc_per_measurement,
            self.__hamiltonian_data,
            self.__Dt,
            self.__epsilon_AS,
        )
        _ = vmap(compute_local_energy_jax, in_axes=(None, 0, 0))(
            self.__hamiltonian_data,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
        )
        _ = vmap(compute_AS_regularization_factor_jax, in_axes=(None, 0, 0))(
            self.__hamiltonian_data.wavefunction_data.geminal_data,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
        )
        _ = vmap(evaluate_ln_wavefunction_jax, in_axes=(None, 0, 0))(
            self.__hamiltonian_data.wavefunction_data,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
        )
        if self.__comput_position_deriv:
            _, _, _ = vmap(grad(compute_local_energy_jax, argnums=(0, 1, 2)), in_axes=(None, 0, 0))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )

            _ = vmap(evaluate_ln_wavefunction_jax, in_axes=(None, 0, 0))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )

            _, _, _ = vmap(grad(evaluate_ln_wavefunction_jax, argnums=(0, 1, 2)), in_axes=(None, 0, 0))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )

            _ = vmap(evaluate_swct_omega_jax, in_axes=(None, 0))(
                self.__swct_data,
                self.__latest_r_up_carts,
            )

            _ = vmap(evaluate_swct_omega_jax, in_axes=(None, 0))(
                self.__swct_data,
                self.__latest_r_dn_carts,
            )

            _ = vmap(evaluate_swct_domega_jax, in_axes=(None, 0))(
                self.__swct_data,
                self.__latest_r_up_carts,
            )

            _ = vmap(evaluate_swct_domega_jax, in_axes=(None, 0))(
                self.__swct_data,
                self.__latest_r_dn_carts,
            )
            _ = vmap(grad(evaluate_ln_wavefunction_jax, argnums=0), in_axes=(None, 0, 0))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )

        if self.__comput_param_deriv:
            _ = vmap(grad(evaluate_ln_wavefunction_jax, argnums=0), in_axes=(None, 0, 0))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )

            """ for Linear method
            _ = vmap(grad(compute_local_energy_api, argnums=0), in_axes=(None, 0, 0))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            """

        mcmc_update_init_end = time.perf_counter()
        timer_mcmc_update_init += mcmc_update_init_end - mcmc_update_init_start
        logger.info("End compilation of the MCMC_update funciton.")
        logger.info(f"Elapsed Time = {mcmc_update_init_end - mcmc_update_init_start:.2f} sec.")
        logger.info("")

        # MAIN MCMC loop from here !!!
        logger.info("Start MCMC")
        num_mcmc_done = 0
        progress = (self.__mcmc_counter) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
        mcmc_total_current = time.perf_counter()
        logger.info(
            f"  Progress: MCMC step= {self.__mcmc_counter}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.0f} %. Elapsed time = {(mcmc_total_current - mcmc_total_start):.1f} sec."
        )
        mcmc_interval = max(1, int(num_mcmc_steps / 10))  # %

        # adjust_epsilon_AS = self.__adjust_epsilon_AS

        for i_mcmc_step in range(num_mcmc_steps):
            if (i_mcmc_step + 1) % mcmc_interval == 0:
                progress = (i_mcmc_step + self.__mcmc_counter + 1) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
                mcmc_total_current = time.perf_counter()
                logger.info(
                    f"  Progress: MCMC step = {i_mcmc_step + self.__mcmc_counter + 1}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %. Elapsed time = {(mcmc_total_current - mcmc_total_start):.1f} sec."
                )

            # electron positions are goint to be updated!
            start = time.perf_counter()
            (
                accepted_moves_nw,
                rejected_moves_nw,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                self.__jax_PRNG_key_list,
            ) = vmap(_update_electron_positions, in_axes=(0, 0, 0, None, None, None, None))(
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                self.__jax_PRNG_key_list,
                self.__num_mcmc_per_measurement,
                self.__hamiltonian_data,
                self.__Dt,
                self.__epsilon_AS,
            )
            end = time.perf_counter()
            timer_mcmc_update += end - start

            # store vmapped outcomes
            self.__accepted_moves += jnp.sum(accepted_moves_nw)
            self.__rejected_moves += jnp.sum(rejected_moves_nw)

            # evaluate observables
            start = time.perf_counter()
            e_L = vmap(compute_local_energy_jax, in_axes=(None, 0, 0))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            logger.devel(f"e_L = {e_L}")
            end = time.perf_counter()
            timer_e_L += end - start

            self.__stored_e_L.append(e_L)
            self.__stored_e_L2.append(e_L**2)

            # compute AS regularization factors, R_AS and R_AS_eps
            R_AS = vmap(compute_AS_regularization_factor_jax, in_axes=(None, 0, 0))(
                self.__hamiltonian_data.wavefunction_data.geminal_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            R_AS_eps = jnp.maximum(R_AS, self.__epsilon_AS)

            # logger.info(f"R_AS = {R_AS}.")
            # logger.info(f"R_AS_eps = {R_AS_eps}.")

            w_L = (R_AS / R_AS_eps) ** 2
            if (i_mcmc_step + 1) % mcmc_interval == 0:
                logger.debug(f"      min, mean, max of weights are {np.min(w_L):.2f}, {np.mean(w_L):.2f}, {np.max(w_L):.2f}.")
            self.__stored_w_L.append(w_L)

            # logger.info(f"      The current epsilon_AS = {self.__epsilon_AS:.5f}")
            """ deactivated for the time being
            if adjust_epsilon_AS:
                # Update adjust_epsilon_AS so that the average of weights approaches target_weight. Proportional control.
                epsilon_AS_max = 1.0e-0
                epsilon_AS_min = 0.0
                gain_weight = 5.0e-3
                target_weight = 0.8
                torrelance_of_weight = 0.05

                ## Calculate the average of weights
                average_weight = np.mean(w_L)
                average_weight = mpi_comm.allreduce(average_weight, op=MPI.SUM)
                average_weight = average_weight / mpi_size
                logger.debug(f"      The current epsilon_AS = {self.__epsilon_AS:.5f}")
                logger.debug(f"      The current averaged weights = {average_weight:.2f}")

                ## Calculate the error as the difference between the current average and the target
                diff_weight = average_weight - target_weight

                ## switch off self.__adjust_epsilon_AS:
                if np.abs(diff_weight) < torrelance_of_weight:
                    # logger.info(f"      The averaged weights is converged within the torrelance of {torrelance_of_weight:.5f}.")
                    adjust_epsilon_AS = False
                else:
                    ## Update epsilon proportionally to the error
                    self.__epsilon_AS = self.__epsilon_AS + gain_weight * diff_weight

                    ## Clip new_epsilon to ensure it remains within defined bounds for stability
                    self.__epsilon_AS = max(min(self.__epsilon_AS, epsilon_AS_max), epsilon_AS_min)

                    logger.info(f"      epsilon_AS is updated to {self.__epsilon_AS:.5f}")
            """

            if self.__comput_position_deriv:
                # """
                start = time.perf_counter()
                grad_e_L_h, grad_e_L_r_up, grad_e_L_r_dn = vmap(
                    grad(compute_local_energy_jax, argnums=(0, 1, 2)), in_axes=(None, 0, 0)
                )(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                end = time.perf_counter()
                timer_de_L_dR_dr += end - start

                self.__stored_grad_e_L_r_up.append(grad_e_L_r_up)
                self.__stored_grad_e_L_r_dn.append(grad_e_L_r_dn)

                """ it works only for MOs_data
                grad_e_L_R = (
                    grad_e_L_h.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.structure_data.positions
                    + grad_e_L_h.wavefunction_data.geminal_data.orb_data_dn_spin.aos_data.structure_data.positions
                    + grad_e_L_h.coulomb_potential_data.structure_data.positions
                )
                """

                grad_e_L_R = (
                    grad_e_L_h.wavefunction_data.geminal_data.orb_data_up_spin.structure_data.positions
                    + grad_e_L_h.wavefunction_data.geminal_data.orb_data_dn_spin.structure_data.positions
                    + grad_e_L_h.coulomb_potential_data.structure_data.positions
                )

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_one_body_data is not None:
                    grad_e_L_R += grad_e_L_h.wavefunction_data.jastrow_data.jastrow_one_body_data.structure_data.positions

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data is not None:
                    grad_e_L_R += (
                        grad_e_L_h.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data.structure_data.positions
                    )

                self.__stored_grad_e_L_dR.append(grad_e_L_R)
                # """

                # """
                logger.devel(f"de_L_dR(coulomb_potential_data) = {grad_e_L_h.coulomb_potential_data.structure_data.positions}")
                logger.devel(f"de_L_dR = {grad_e_L_R}")
                logger.devel(f"de_L_dr_up = {grad_e_L_r_up}")
                logger.devel(f"de_L_dr_dn= {grad_e_L_r_dn}")
                # """

                # """
                start = time.perf_counter()
                grad_ln_Psi_h, grad_ln_Psi_r_up, grad_ln_Psi_r_dn = vmap(
                    grad(evaluate_ln_wavefunction_jax, argnums=(0, 1, 2)), in_axes=(None, 0, 0)
                )(
                    self.__hamiltonian_data.wavefunction_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                end = time.perf_counter()
                timer_dln_Psi_dR_dr += end - start

                logger.devel(f"dln_Psi_dr_up = {grad_ln_Psi_r_up}")
                logger.devel(f"dln_Psi_dr_dn = {grad_ln_Psi_r_dn}")
                self.__stored_grad_ln_Psi_r_up.append(grad_ln_Psi_r_up)
                self.__stored_grad_ln_Psi_r_dn.append(grad_ln_Psi_r_dn)

                grad_ln_Psi_dR = (
                    grad_ln_Psi_h.geminal_data.orb_data_up_spin.structure_data.positions
                    + grad_ln_Psi_h.geminal_data.orb_data_dn_spin.structure_data.positions
                )

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_one_body_data is not None:
                    grad_ln_Psi_dR += grad_ln_Psi_h.jastrow_data.jastrow_one_body_data.structure_data.positions

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data is not None:
                    grad_ln_Psi_dR += grad_ln_Psi_h.jastrow_data.jastrow_three_body_data.orb_data.structure_data.positions

                # stored dln_Psi / dR
                logger.devel(f"dln_Psi_dR = {grad_ln_Psi_dR}")
                self.__stored_grad_ln_Psi_dR.append(grad_ln_Psi_dR)
                # """

                omega_up = vmap(evaluate_swct_omega_jax, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_up_carts,
                )

                omega_dn = vmap(evaluate_swct_omega_jax, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_dn_carts,
                )

                logger.devel(f"omega_up = {omega_up}")
                logger.devel(f"omega_dn = {omega_dn}")

                self.__stored_omega_up.append(omega_up)
                self.__stored_omega_dn.append(omega_dn)

                grad_omega_dr_up = vmap(evaluate_swct_domega_jax, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_up_carts,
                )

                grad_omega_dr_dn = vmap(evaluate_swct_domega_jax, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_dn_carts,
                )

                logger.devel(f"grad_omega_dr_up = {grad_omega_dr_up}")
                logger.devel(f"grad_omega_dr_dn = {grad_omega_dr_dn}")

                self.__stored_grad_omega_r_up.append(grad_omega_dr_up)
                self.__stored_grad_omega_r_dn.append(grad_omega_dr_dn)

            if self.__comput_param_deriv:
                start = time.perf_counter()
                grad_ln_Psi_h = vmap(grad(evaluate_ln_wavefunction_jax, argnums=0), in_axes=(None, 0, 0))(
                    self.__hamiltonian_data.wavefunction_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                end = time.perf_counter()
                timer_dln_Psi_dc += end - start

                start = time.perf_counter()
                """ for Linear method
                grad_e_L_h = vmap(grad(compute_local_energy_api, argnums=0), in_axes=(None, 0, 0))(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                """
                end = time.perf_counter()
                timer_de_L_dc += end - start

                # 1b Jastrow
                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_one_body_data is not None:
                    grad_ln_Psi_jas1b = grad_ln_Psi_h.jastrow_data.jastrow_one_body_data.jastrow_1b_param
                    logger.devel(f"grad_ln_Psi_jas1b.shape = {grad_ln_Psi_jas1b.shape}")
                    logger.devel(f"  grad_ln_Psi_jas1b = {grad_ln_Psi_jas1b}")
                    self.__stored_grad_ln_Psi_jas1b.append(grad_ln_Psi_jas1b)

                    """ for Linear method
                    grad_e_L_jas2b = grad_e_L_h.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
                    logger.devel(f"grad_e_L_jas2b.shape = {grad_e_L_jas2b.shape}")
                    logger.devel(f"  grad_e_L_jas2b = {grad_e_L_jas2b}")
                    self.__stored_grad_e_L_jas2b.append(grad_e_L_jas2b)
                    """

                # 2b Jastrow
                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data is not None:
                    grad_ln_Psi_jas2b = grad_ln_Psi_h.jastrow_data.jastrow_two_body_data.jastrow_2b_param
                    logger.devel(f"grad_ln_Psi_jas2b.shape = {grad_ln_Psi_jas2b.shape}")
                    logger.devel(f"  grad_ln_Psi_jas2b = {grad_ln_Psi_jas2b}")
                    self.__stored_grad_ln_Psi_jas2b.append(grad_ln_Psi_jas2b)

                    """ for Linear method
                    grad_e_L_jas2b = grad_e_L_h.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
                    logger.devel(f"grad_e_L_jas2b.shape = {grad_e_L_jas2b.shape}")
                    logger.devel(f"  grad_e_L_jas2b = {grad_e_L_jas2b}")
                    self.__stored_grad_e_L_jas2b.append(grad_e_L_jas2b)
                    """

                # 3b Jastrow
                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data is not None:
                    grad_ln_Psi_jas1b3b_j_matrix = grad_ln_Psi_h.jastrow_data.jastrow_three_body_data.j_matrix
                    logger.devel(f"grad_ln_Psi_jas1b3b_j_matrix.shape={grad_ln_Psi_jas1b3b_j_matrix.shape}")
                    logger.devel(f"  grad_ln_Psi_jas1b3b_j_matrix = {grad_ln_Psi_jas1b3b_j_matrix}")
                    self.__stored_grad_ln_Psi_jas1b3b_j_matrix.append(grad_ln_Psi_jas1b3b_j_matrix)

                    """ for Linear method
                    grad_e_L_jas1b3b_j_matrix = grad_e_L_h.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix
                    logger.devel(f"grad_e_L_jas1b3b_j_matrix.shape = {grad_e_L_jas1b3b_j_matrix.shape}")
                    logger.devel(f"  grad_e_L_jas1b3b_j_matrix = {grad_e_L_jas1b3b_j_matrix}")
                    self.__stored_grad_e_L_jas1b3b_j_matrix.append(grad_e_L_jas1b3b_j_matrix)
                    """

                # lambda_matrix
                grad_ln_Psi_lambda_matrix = grad_ln_Psi_h.geminal_data.lambda_matrix
                logger.devel(f"grad_ln_Psi_lambda_matrix.shape={grad_ln_Psi_lambda_matrix.shape}")
                logger.devel(f"  grad_ln_Psi_lambda_matrix = {grad_ln_Psi_lambda_matrix}")
                self.__stored_grad_ln_Psi_lambda_matrix.append(grad_ln_Psi_lambda_matrix)

                """ for Linear method
                grad_e_L_lambda_matrix = grad_e_L_h.wavefunction_data.geminal_data.lambda_matrix
                logger.devel(f"grad_e_L_lambda_matrix.shape = {grad_e_L_lambda_matrix.shape}")
                logger.devel(f"  grad_e_L_lambda_matrix = {grad_e_L_lambda_matrix}")
                self.__stored_grad_e_L_lambda_matrix.append(grad_e_L_lambda_matrix)
                """

            num_mcmc_done += 1

            # check max time
            mcmc_current = time.perf_counter()
            if max_time < mcmc_current - mcmc_total_start:
                logger.info(f"max_time = {max_time} sec. exceeds.")
                logger.info("break the mcmc loop.")
                break

        logger.info("End MCMC")
        logger.info("")

        # count up the mcmc counter
        # count up mcmc_counter
        self.__mcmc_counter += num_mcmc_done

        mcmc_total_end = time.perf_counter()
        timer_mcmc_total += mcmc_total_end - mcmc_total_start
        timer_misc = timer_mcmc_total - (
            timer_mcmc_update_init
            + timer_mcmc_update
            + timer_e_L
            + timer_de_L_dR_dr
            + timer_dln_Psi_dR_dr
            + timer_dln_Psi_dc
            + timer_de_L_dc
        )

        self.__timer_mcmc_total += timer_mcmc_total
        self.__timer_mcmc_update_init += timer_mcmc_update_init
        self.__timer_mcmc_update += timer_mcmc_update
        self.__timer_e_L += timer_e_L
        self.__timer_de_L_dR_dr += timer_de_L_dR_dr
        self.__timer_dln_Psi_dR_dr += timer_dln_Psi_dR_dr
        self.__timer_dln_Psi_dc += timer_dln_Psi_dc
        self.__timer_de_L_dc += timer_de_L_dc
        self.__timer_misc += timer_misc

        logger.info(f"Total elapsed time for MCMC {num_mcmc_done} steps. = {timer_mcmc_total:.2f} sec.")
        logger.info(f"Pre-compilation time for MCMC = {timer_mcmc_update_init:.2f} sec.")
        logger.info(f"Net total time for MCMC = {timer_mcmc_total - timer_mcmc_update_init:.2f} sec.")
        logger.info(f"Elapsed times per MCMC step, averaged over {num_mcmc_done} steps.")
        logger.info(f"  Time for MCMC update = {timer_mcmc_update / num_mcmc_done * 10**3:.2f} msec.")
        logger.info(f"  Time for computing e_L = {timer_e_L / num_mcmc_done * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/dR and de_L/dr = {timer_de_L_dR_dr / num_mcmc_done * 10**3:.2f} msec.")
        logger.info(f"  Time for computing dln_Psi/dR and dln_Psi/dr = {timer_dln_Psi_dR_dr / num_mcmc_done * 10**3:.2f} msec.")
        logger.info(f"  Time for computing dln_Psi/dc = {timer_dln_Psi_dc / num_mcmc_done * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/dc = {timer_de_L_dc / num_mcmc_done * 10**3:.2f} msec.")
        logger.info(f"  Time for misc. (others) = {timer_misc / num_mcmc_done * 10**3:.2f} msec.")
        logger.info(f"Average of walker weights is {np.mean(self.__stored_w_L):.3f}. Ideal is ~ 0.800. Adjust epsilon_AS.")
        logger.info(
            f"Acceptance ratio is {self.__accepted_moves / (self.__accepted_moves + self.__rejected_moves) * 100:.2f} %.  Ideal is ~ 50.00%. Adjust Dt."
        )
        logger.info("")

    # hamiltonian
    @property
    def hamiltonian_data(self):
        """Return hamiltonian_data."""
        return self.__hamiltonian_data

    @hamiltonian_data.setter
    def hamiltonian_data(self, hamiltonian_data):
        """Set hamiltonian_data."""
        if self.__comput_param_deriv and not self.__comput_position_deriv:
            self.__hamiltonian_data = Hamiltonian_data_deriv_params.from_base(hamiltonian_data)
        # bug?
        # elif not self.__comput_param_deriv and self.__comput_position_deriv:
        #    self.__hamiltonian_data = Hamiltonian_data_deriv_R.from_base(hamiltonian_data)
        else:
            self.__hamiltonian_data = hamiltonian_data
        self.__init_attributes()

    # dimensions of observables
    @property
    def mcmc_counter(self) -> int:
        """Return current MCMC counter."""
        return self.__mcmc_counter

    @property
    def num_walkers(self):
        """The number of walkers."""
        return self.__num_walkers

    # weights
    @property
    def w_L(self) -> npt.NDArray:
        """Return the stored weight array. dim: (mcmc_counter, num_walkers)."""
        # self.__stored_w_L = np.ones((self.mcmc_counter, self.num_walkers))  # tentative
        return np.array(self.__stored_w_L)

    # observables
    @property
    def e_L(self) -> npt.NDArray:
        """Return the stored e_L array. dim: (mcmc_counter, num_walkers)."""
        return np.array(self.__stored_e_L)

    # observables
    @property
    def e_L2(self) -> npt.NDArray:
        """Return the stored e_L^2 array. dim: (mcmc_counter, num_walkers)."""
        return np.array(self.__stored_e_L2)

    @property
    def de_L_dR(self) -> npt.NDArray:
        """Return the stored de_L/dR array. dim: (mcmc_counter, num_walkers)."""
        return np.array(self.__stored_grad_e_L_dR)

    @property
    def de_L_dr_up(self) -> npt.NDArray:
        """Return the stored de_L/dr_up array. dim: (mcmc_counter, num_walkers, num_electrons_up, 3)."""
        return np.array(self.__stored_grad_e_L_r_up)

    @property
    def de_L_dr_dn(self) -> npt.NDArray:
        """Return the stored de_L/dr_dn array. dim: (mcmc_counter, num_walkers, num_electrons_dn, 3)."""
        return np.array(self.__stored_grad_e_L_r_dn)

    @property
    def dln_Psi_dr_up(self) -> npt.NDArray:
        """Return the stored dln_Psi/dr_up array. dim: (mcmc_counter, num_walkers, num_electrons_up, 3)."""
        return np.array(self.__stored_grad_ln_Psi_r_up)

    @property
    def dln_Psi_dr_dn(self) -> npt.NDArray:
        """Return the stored dln_Psi/dr_down array. dim: (mcmc_counter, num_walkers, num_electrons_dn, 3)."""
        return np.array(self.__stored_grad_ln_Psi_r_dn)

    @property
    def dln_Psi_dR(self) -> npt.NDArray:
        """Return the stored dln_Psi/dR array. dim: (mcmc_counter, num_walkers, num_atoms, 3)."""
        return np.array(self.__stored_grad_ln_Psi_dR)

    @property
    def omega_up(self) -> npt.NDArray:
        """Return the stored Omega (for up electrons) array. dim: (mcmc_counter, num_walkers, num_atoms, num_electrons_up)."""
        return np.array(self.__stored_omega_up)

    @property
    def omega_dn(self) -> npt.NDArray:
        """Return the stored Omega (for down electrons) array. dim: (mcmc_counter, num_walkers, num_atoms, num_electons_dn)."""
        return np.array(self.__stored_omega_dn)

    @property
    def domega_dr_up(self) -> npt.NDArray:
        """Return the stored dOmega/dr_up array. dim: (mcmc_counter, num_walkers, num_electons_dn, 3)."""
        return np.array(self.__stored_grad_omega_r_up)

    @property
    def domega_dr_dn(self) -> npt.NDArray:
        """Return the stored dOmega/dr_dn array. dim: (mcmc_counter, num_walkers, num_electons_dn, 3)."""
        return np.array(self.__stored_grad_omega_r_dn)

    @property
    def dln_Psi_dc_jas_1b(self) -> npt.NDArray:
        """Return the stored dln_Psi/dc_J1 array. dim: (mcmc_counter, num_walkers, num_J1_param)."""
        return np.array(self.__stored_grad_ln_Psi_jas1b)

    @property
    def dln_Psi_dc_jas_2b(self) -> npt.NDArray:
        """Return the stored dln_Psi/dc_J2 array. dim: (mcmc_counter, num_walkers, num_J2_param)."""
        return np.array(self.__stored_grad_ln_Psi_jas2b)

    @property
    def dln_Psi_dc_jas_1b3b(self) -> npt.NDArray:
        """Return the stored dln_Psi/dc_J1_3 array. dim: (mcmc_counter, num_walkers, num_J1_J3_param)."""
        return np.array(self.__stored_grad_ln_Psi_jas1b3b_j_matrix)

    '''
    @property
    def de_L_dc_jas_2b(self) -> npt.NDArray:
        """Return the stored de_L/dc_J2 array. dim: (mcmc_counter, num_walkers, num_J2_param)."""
        return np.array(self.__stored_grad_e_L_jas2b)

    @property
    def de_L_dc_jas_1b3b(self) -> npt.NDArray:
        """Return the stored de_L/dc_J1_3 array. dim: (mcmc_counter, num_walkers, num_J1_J3_param)."""
        return np.array(self.__stored_grad_e_L_jas1b3b_j_matrix)
    '''

    @property
    def dln_Psi_dc_lambda_matrix(self) -> npt.NDArray:
        """Return the stored dln_Psi/dc_lambda_matrix array. dim: (mcmc_counter, num_walkers, num_lambda_matrix_param)."""
        return np.array(self.__stored_grad_ln_Psi_lambda_matrix)

    '''
    @property
    def de_L_dc_lambda_matrix(self) -> npt.NDArray:
        """Return the stored de_L/dc_lambda_matrix array. dim: (mcmc_counter, num_walkers, num_lambda_matrix_param)."""
        return np.array(self.__stored_grad_e_L_lambda_matrix)
    '''

    @property
    def comput_position_deriv(self) -> bool:
        """Return the flag for computing the derivatives of E wrt. atomic positions."""
        return self.__comput_position_deriv

    # dict for WF optimization
    @property
    def opt_param_dict(self):
        """Return a dictionary containing information about variational parameters to be optimized.

        Refactoring in progress.

        Return:
            dc_param_list (list): labels of the parameters with derivatives computed.
            dln_Psi_dc_list (list): dln_Psi_dc instances computed by JAX-grad.
            dc_size_list (list): sizes of dln_Psi_dc instances
            dc_shape_list (list): shapes of dln_Psi_dc instances
            dc_flattened_index_list (list): indices of dln_Psi_dc instances for the flattened parameter
        #
        """
        dc_param_list = []
        dln_Psi_dc_list = []
        # de_L_dc_list = [] # for linear method
        dc_size_list = []
        dc_shape_list = []
        dc_flattened_index_list = []

        if self.__comput_param_deriv:
            # jastrow 1-body
            if self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_one_body_data is not None:
                dc_param = "j1_param"
                dln_Psi_dc = self.dln_Psi_dc_jas_1b
                # de_L_dc = self.de_L_dc_jas_1b # for linear method
                dc_size = 1
                dc_shape = (1,)
                dc_flattened_index = [len(dc_param_list)] * dc_size

                dc_param_list.append(dc_param)
                dln_Psi_dc_list.append(dln_Psi_dc)
                # de_L_dc_list.append(de_L_dc) # for linear method
                dc_size_list.append(dc_size)
                dc_shape_list.append(dc_shape)
                dc_flattened_index_list += dc_flattened_index
            # jastrow 2-body
            if self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data is not None:
                dc_param = "j2_param"
                dln_Psi_dc = self.dln_Psi_dc_jas_2b
                # de_L_dc = self.de_L_dc_jas_2b # for linear method
                dc_size = 1
                dc_shape = (1,)
                dc_flattened_index = [len(dc_param_list)] * dc_size

                dc_param_list.append(dc_param)
                dln_Psi_dc_list.append(dln_Psi_dc)
                # de_L_dc_list.append(de_L_dc) # for linear method
                dc_size_list.append(dc_size)
                dc_shape_list.append(dc_shape)
                dc_flattened_index_list += dc_flattened_index

            # jastrow 3-body
            if self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data is not None:
                dc_param = "j3_matrix"
                dln_Psi_dc = self.dln_Psi_dc_jas_1b3b
                # de_L_dc = self.de_L_dc_jas_1b3b # for linear method
                dc_size = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix.size
                dc_shape = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix.shape
                dc_flattened_index = [len(dc_param_list)] * dc_size

                dc_param_list.append(dc_param)
                dln_Psi_dc_list.append(dln_Psi_dc)
                # de_L_dc_list.append(de_L_dc) # for linear method
                dc_size_list.append(dc_size)
                dc_shape_list.append(dc_shape)
                dc_flattened_index_list += dc_flattened_index

            # lambda_matrix
            dc_param = "lambda_matrix"
            dln_Psi_dc = self.dln_Psi_dc_lambda_matrix
            # de_L_dc = self.de_L_dc_lambda # for linear method
            dc_size = self.hamiltonian_data.wavefunction_data.geminal_data.lambda_matrix.size
            dc_shape = self.hamiltonian_data.wavefunction_data.geminal_data.lambda_matrix.shape
            dc_flattened_index = [len(dc_param_list)] * dc_size

            dc_param_list.append(dc_param)
            dln_Psi_dc_list.append(dln_Psi_dc)
            # de_L_dc_list.append(de_L_dc) # for linear method
            dc_size_list.append(dc_size)
            dc_shape_list.append(dc_shape)
            dc_flattened_index_list += dc_flattened_index

        return {
            "dc_param_list": dc_param_list,
            "dln_Psi_dc_list": dln_Psi_dc_list,
            # "de_L_dc_list": de_L_dc_list, # for linear method
            "dc_size_list": dc_size_list,
            "dc_shape_list": dc_shape_list,
            "dc_flattened_index_list": dc_flattened_index_list,
        }


# accumurate weights
@partial(jit, static_argnums=1)
def compute_G_L(w_L, num_gfmc_collect_steps):
    """Return accumulate weights for multi-dimensional w_L.

    Note: The dimension of w_L is (num_mcmc, num_walkers)

    """
    A, x = w_L.shape

    def get_slice(n):
        return jax.lax.dynamic_slice(w_L, (n - num_gfmc_collect_steps, 0), (num_gfmc_collect_steps, x))

    indices = jnp.arange(num_gfmc_collect_steps, A)
    G_L_matrix = vmap(get_slice)(indices)  # (A - num_gfmc_collect_steps, num_gfmc_collect_steps, x)
    G_L = jnp.prod(G_L_matrix, axis=1)  # (A - num_gfmc_collect_steps, x)

    return G_L


class GFMC_fixed_projection_time:
    """GFMC class.

    GFMC class. Runing GFMC.

    Args:
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
        num_walkers (int): the number of walkers
        num_gfmc_collect_steps(int): the number of steps to collect the GFMC data
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
        num_gfmc_collect_steps: int = 5,
        mcmc_seed: int = 34467,
        tau: float = 0.1,
        alat: float = 0.1,
        non_local_move: str = "tmove",
    ) -> None:
        """Init.

        Initialize a MCMC class, creating list holding results, etc...

        """
        # check sanity of hamiltonian_data
        hamiltonian_data.sanity_check()

        # attributes
        self.__hamiltonian_data = hamiltonian_data
        self.__num_walkers = num_walkers
        self.__num_gfmc_collect_steps = num_gfmc_collect_steps
        self.__mcmc_seed = mcmc_seed
        self.__tau = tau
        self.__alat = alat
        self.__non_local_move = non_local_move

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
        self.__mpi_seed = self.__mcmc_seed * (mpi_rank + 1)
        self.__jax_PRNG_key = jax.random.PRNGKey(self.__mpi_seed)
        self.__jax_PRNG_key_list_init = jnp.array(
            [jax.random.fold_in(self.__jax_PRNG_key, nw) for nw in range(self.__num_walkers)]
        )
        self.__jax_PRNG_key_list = self.__jax_PRNG_key_list_init

        # Place electrons around each nucleus with improved spin assignment

        tot_num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        tot_num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn

        r_carts_up_list = []
        r_carts_dn_list = []

        np.random.seed(self.__mpi_seed)

        for _ in range(self.__num_walkers):
            # Initialization
            r_carts_up = []
            r_carts_dn = []
            total_assigned_up = 0
            total_assigned_dn = 0

            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
                    hamiltonian_data.coulomb_potential_data.z_cores
                )
            else:
                charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

            logger.devel(f"charges = {charges}.")
            coords = hamiltonian_data.structure_data.positions_cart_jnp

            # Place electrons for each atom
            for i in range(len(coords)):
                charge = charges[i]
                n_elec = int(np.round(charge))  # Number of electrons to place for this atom
                logger.devel(f"charge = {charge}.")
                logger.devel(f"num_electrons = {n_elec}.")

                x, y, z = coords[i]

                # Determine the number of up and dn electrons for this atom
                if n_elec % 2 == 0:
                    num_up = n_elec // 2
                    num_dn = n_elec // 2
                else:
                    # For odd numbers, decide which spin gets the extra electron based on the overall shortage
                    remaining_up = tot_num_electron_up - total_assigned_up
                    remaining_dn = tot_num_electron_dn - total_assigned_dn
                    if remaining_up >= remaining_dn:
                        num_up = n_elec // 2 + 1
                        num_dn = n_elec // 2
                    else:
                        num_up = n_elec // 2
                        num_dn = n_elec // 2 + 1

                # Place up electrons
                for _ in range(num_up):
                    distance = np.random.uniform(0.1, 1.0)
                    theta = np.random.uniform(0, np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    dx = distance * np.sin(theta) * np.cos(phi)
                    dy = distance * np.sin(theta) * np.sin(phi)
                    dz = distance * np.cos(theta)
                    electron_position = np.array([x + dx, y + dy, z + dz])
                    r_carts_up.append(electron_position)
                total_assigned_up += num_up

                # Place dn electrons
                for _ in range(num_dn):
                    distance = np.random.uniform(0.1, 1.0)
                    theta = np.random.uniform(0, np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    dx = distance * np.sin(theta) * np.cos(phi)
                    dy = distance * np.sin(theta) * np.sin(phi)
                    dz = distance * np.cos(theta)
                    electron_position = np.array([x + dx, y + dy, z + dz])
                    r_carts_dn.append(electron_position)
                total_assigned_dn += num_dn

            # Electron assignment for all atoms is complete
            logger.devel(f"Total assigned up electrons: {total_assigned_up} (target {tot_num_electron_up}).")
            logger.devel(f"Total assigned dn electrons: {total_assigned_dn} (target {tot_num_electron_dn}).")

            # If necessary, include a check/adjustment step to ensure the overall assignment matches the targets
            # (Here it is assumed that sum(round(charge)) equals tot_num_electron_up + tot_num_electron_dn)

            r_carts_up_list.append(r_carts_up)
            r_carts_dn_list.append(r_carts_dn)

        self.__latest_r_up_carts = jnp.array(r_carts_up_list)
        self.__latest_r_dn_carts = jnp.array(r_carts_dn_list)

        logger.info(f"The number of MPI process = {mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self.__num_walkers}.")

        logger.devel(f"initial r_up_carts= {self.__latest_r_up_carts}")
        logger.devel(f"initial r_dn_carts = {self.__latest_r_dn_carts}")
        logger.devel(f"initial r_up_carts.shape = {self.__latest_r_up_carts.shape}")
        logger.devel(f"initial r_dn_carts.shape = {self.__latest_r_dn_carts.shape}")
        logger.info("")

        # print out hamiltonian info
        logger.info("Printing out information in hamitonian_data instance.")
        self.__hamiltonian_data.logger_info()
        logger.info("")

        logger.info("Compilation of fundamental functions starts.")

        logger.info("  Compilation e_L starts.")
        _ = compute_kinetic_energy_jax(
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts[0],
            r_dn_carts=self.__latest_r_dn_carts[0],
        )
        _, _, _ = compute_discretized_kinetic_energy_jax(
            alat=self.__alat,
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts[0],
            r_dn_carts=self.__latest_r_dn_carts[0],
            RT=jnp.eye(3, 3),
        )
        _ = compute_bare_coulomb_potential_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            r_up_carts=self.__latest_r_up_carts[0],
            r_dn_carts=self.__latest_r_dn_carts[0],
        )
        if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
            _ = compute_ecp_local_parts_all_pairs_jax(
                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                r_up_carts=self.__latest_r_up_carts[0],
                r_dn_carts=self.__latest_r_dn_carts[0],
            )
            if self.__non_local_move == "tmove":
                _, _, _, _ = compute_ecp_non_local_parts_nearest_neighbors_jax(
                    coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=self.__latest_r_up_carts[0],
                    r_dn_carts=self.__latest_r_dn_carts[0],
                    flag_determinant_only=False,
                )
            elif self.__non_local_move == "dltmove":
                _, _, _, _ = compute_ecp_non_local_parts_nearest_neighbors_jax(
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

        # init attributes
        self.__init_attributes()

    def __init_attributes(self):
        # mcmc counter
        self.__mcmc_counter = 0

        # gfmc accepted/rejected moves
        self.__num_survived_walkers = 0
        self.__num_killed_walkers = 0

        # stored local energy (w_L)
        self.__stored_w_L = []

        # stored local energy (e_L)
        self.__stored_e_L = []

        # stored local energy (e_L)
        self.__stored_e_L2 = []

        # average projection counter
        self.__stored_average_projection_counter = []

    # hamiltonian
    @property
    def hamiltonian_data(self):
        """Return hamiltonian_data."""
        return self.__hamiltonian_data

    @hamiltonian_data.setter
    def hamiltonian_data(self, hamiltonian_data):
        """Set hamiltonian_data."""
        self.__hamiltonian_data = hamiltonian_data
        self.__init_attributes()

    # collecting factor
    @property
    def num_gfmc_collect_steps(self):
        """Return num_gfmc_collect_steps."""
        return self.__num_gfmc_collect_steps

    @num_gfmc_collect_steps.setter
    def num_gfmc_collect_steps(self, num_gfmc_collect_steps):
        """Set num_gfmc_collect_steps."""
        self.__num_gfmc_collect_steps = num_gfmc_collect_steps

    # dimensions of observables
    @property
    def mcmc_counter(self) -> int:
        """Return current MCMC counter."""
        return self.__mcmc_counter - self.__num_gfmc_collect_steps

    @property
    def num_walkers(self):
        """The number of walkers."""
        return self.__num_walkers

    # weights
    @property
    def w_L(self) -> npt.NDArray:
        """Return the stored weight array. dim: (mcmc_counter, 1)."""
        # logger.info(f"np.array(self.__stored_w_L).shape = {np.array(self.__stored_w_L).shape}.")
        return compute_G_L(np.array(self.__stored_w_L), self.__num_gfmc_collect_steps)

    # observables
    @property
    def e_L(self) -> npt.NDArray:
        """Return the stored e_L array. dim: (mcmc_counter, 1)."""
        # logger.info(f"np.array(self.__stored_e_L).shape = {np.array(self.__stored_e_L).shape}.")
        return np.array(self.__stored_e_L)[self.__num_gfmc_collect_steps :]

    # observables
    @property
    def e_L2(self) -> npt.NDArray:
        """Return the stored e_L2 array. dim: (mcmc_counter, 1)."""
        # logger.info(f"np.array(self.__stored_e_L2).shape = {np.array(self.__stored_e_L).shape}.")
        return np.array(self.__stored_e_L2)[self.__num_gfmc_collect_steps :]

    def run(self, num_mcmc_steps: int = 50, max_time: int = 86400) -> None:
        """Run LRDMC with multiple walkers.

        Args:
            num_branching (int): number of branching (reconfiguration of walkers).
            max_time (int): maximum time in sec.
        """
        # set timer
        timer_projection_init = 0.0
        timer_projection_total = 0.0
        timer_observable = 0.0
        timer_mpi_barrier = 0.0
        timer_reconfiguration = 0.0
        gmfc_total_start = time.perf_counter()

        # initialize numpy random seed
        np.random.seed(self.__mpi_seed)

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

        # Note: This jit drastically accelarates the computation!!
        @partial(jit, static_argnums=6)
        def _projection(
            projection_counter: int,
            tau_left: float,
            w_L: float,
            r_up_carts: jax.Array,
            r_dn_carts: jax.Array,
            jax_PRNG_key: jax.Array,
            non_local_move: bool,
            alat: float,
            hamiltonian_data: Hamiltonian_data,
        ):
            """Do projection, compatible with vmap.

            Do projection for a set of (r_up_cart, r_dn_cart).

            Args:
                projection_counter(int): the counter of projection steps
                tau_left (float): left projection time
                w_L (float): weight before projection
                r_up_carts (N_e^up, 3) before projection
                r_dn_carts (N_e^dn, 3) after projection
                jax_PRNG_key (jax.Array): jax PRNG key
                non_local_move (bool): treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
                alat (float): discretized grid length (bohr)
                hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data

            Returns:
                e_L (float): e_L after the final projection.
                projection_counter(int): the counter of projection steps
                tau_left (float): left projection time
                w_L (float): weight after the final projection
                r_up_carts (N_e^up, 3) after the final projection
                r_dn_carts (N_e^dn, 3) after the final projection
                jax_PRNG_key (jax.Array): jax PRNG key
            """
            logger.devel(f"jax_PRNG_key={jax_PRNG_key}")

            # projection counter
            projection_counter = lax.cond(
                tau_left > 0.0,
                lambda pc: pc + 1,
                lambda pc: pc,
                projection_counter,
            )

            #''' coulomb regularization
            # compute diagonal elements, kinetic part
            diagonal_kinetic_part = 3.0 / (2.0 * alat**2) * (len(r_up_carts) + len(r_dn_carts))

            # compute continuum kinetic energy
            diagonal_kinetic_continuum_elements_up, diagonal_kinetic_continuum_elements_dn = (
                compute_kinetic_energy_all_elements_jax(
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # generate a random rotation matrix
            jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
            alpha, beta, gamma = jax.random.uniform(
                subkey, shape=(3,), minval=-2 * jnp.pi, maxval=2 * jnp.pi
            )  # Rotation angle around the x,y,z-axis (in radians)

            R = generate_rotation_matrix(alpha, beta, gamma)  # Rotate in the order x -> y -> z

            # compute discretized kinetic energy and mesh (with a random rotation)
            mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                compute_discretized_kinetic_energy_jax(
                    alat=alat,
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                    RT=R.T,
                )
            )
            # spin-filp
            elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
            non_diagonal_sum_hamiltonian_kinetic = jnp.sum(elements_non_diagonal_kinetic_part_FN)
            diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
            # regularizations
            elements_non_diagonal_kinetic_part_all = elements_non_diagonal_kinetic_part.reshape(-1, 6)
            sign_flip_flags_elements = jnp.any(elements_non_diagonal_kinetic_part_all >= 0, axis=1)
            non_diagonal_kinetic_part_elements = jnp.sum(elements_non_diagonal_kinetic_part_all + 1.0 / (4.0 * alat**2), axis=1)
            sign_flip_flags_elements_up, sign_flip_flags_elements_dn = jnp.split(sign_flip_flags_elements, [len(r_up_carts)])
            non_diagonal_kinetic_part_elements_up, non_diagonal_kinetic_part_elements_dn = jnp.split(
                non_diagonal_kinetic_part_elements, [len(r_up_carts)]
            )

            # compute diagonal elements, el-el
            diagonal_bare_coulomb_part_el_el = compute_bare_coulomb_potential_el_el_jax(
                r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
            )

            # compute diagonal elements, ion-ion
            diagonal_bare_coulomb_part_ion_ion = compute_bare_coulomb_potential_ion_ion_jax(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data
            )

            # compute diagonal elements, el-ion
            diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                compute_bare_coulomb_potential_el_ion_element_wise_jax(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # compute diagonal elements, el-ion, discretized
            (
                diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
            ) = compute_discretized_bare_coulomb_potential_el_ion_element_wise_jax(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                alat=alat,
            )

            # compose discretized el-ion potentials
            diagonal_bare_coulomb_part_el_ion_zv_up = (
                diagonal_bare_coulomb_part_el_ion_elements_up
                + diagonal_kinetic_continuum_elements_up
                - non_diagonal_kinetic_part_elements_up
            )
            # """
            # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
            # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effectice core potential calculations. (c.f., parcutg=1 in TurboRVB).
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_elements_up
            else:
                diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_discretized_elements_up
            diagonal_bare_coulomb_part_el_ion_max_up = jnp.maximum(
                diagonal_bare_coulomb_part_el_ion_zv_up, diagonal_bare_coulomb_part_el_ion_ei_up
            )
            diagonal_bare_coulomb_part_el_ion_opt_up = jnp.where(
                sign_flip_flags_elements_up, diagonal_bare_coulomb_part_el_ion_max_up, diagonal_bare_coulomb_part_el_ion_zv_up
            )
            # diagonal_bare_coulomb_part_el_ion_opt_up = diagonal_bare_coulomb_part_el_ion_max_up
            # diagonal_bare_coulomb_part_el_ion_opt_up = diagonal_bare_coulomb_part_el_ion_zv_up
            # """

            # compose discretized el-ion potentials
            diagonal_bare_coulomb_part_el_ion_zv_dn = (
                diagonal_bare_coulomb_part_el_ion_elements_dn
                + diagonal_kinetic_continuum_elements_dn
                - non_diagonal_kinetic_part_elements_dn
            )
            # """
            # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
            # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effectice core potential calculations. (c.f., parcutg=1 in TurboRVB).
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_elements_dn
            else:
                diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_discretized_elements_dn
            diagonal_bare_coulomb_part_el_ion_max_dn = jnp.maximum(
                diagonal_bare_coulomb_part_el_ion_zv_dn, diagonal_bare_coulomb_part_el_ion_ei_dn
            )
            diagonal_bare_coulomb_part_el_ion_opt_dn = jnp.where(
                sign_flip_flags_elements_dn, diagonal_bare_coulomb_part_el_ion_max_dn, diagonal_bare_coulomb_part_el_ion_zv_dn
            )
            # diagonal_bare_coulomb_part_el_ion_opt_dn = diagonal_bare_coulomb_part_el_ion_max_dn
            # diagonal_bare_coulomb_part_el_ion_opt_dn = diagonal_bare_coulomb_part_el_ion_zv_dn
            # """

            # final bare coulomb part
            discretized_diagonal_bare_coulomb_part = (
                diagonal_bare_coulomb_part_el_el
                + diagonal_bare_coulomb_part_ion_ion
                + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_up)
                + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_dn)
            )
            #'''

            # """ if-else for all-ele, ecp with tmove, and ecp with dltmove
            # with ECP
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                # ecp local
                diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs_jax(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

                if non_local_move == "tmove":
                    # ecp non-local (t-move)
                    mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                        compute_ecp_non_local_parts_nearest_neighbors_jax(
                            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                            wavefunction_data=hamiltonian_data.wavefunction_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                            flag_determinant_only=False,
                        )
                    )

                    V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                    diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                    non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                elif non_local_move == "dltmove":
                    mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                        compute_ecp_non_local_parts_nearest_neighbors_jax(
                            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                            wavefunction_data=hamiltonian_data.wavefunction_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                            flag_determinant_only=True,
                        )
                    )

                    V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                    diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                    Jastrow_ref = compute_Jastrow_part_jax(
                        jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    Jastrow_on_mesh = vmap(compute_Jastrow_part_jax, in_axes=(None, 0, 0))(
                        hamiltonian_data.wavefunction_data.jastrow_data,
                        mesh_non_local_ecp_part_r_up_carts,
                        mesh_non_local_ecp_part_r_dn_carts,
                    )
                    Jastrow_ratio = jnp.exp(Jastrow_on_mesh - Jastrow_ref)
                    V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

                    non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                else:
                    logger.error(f"non_local_move = {non_local_move} is not yet implemented.")
                    raise NotImplementedError

                # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                e_L = (
                    diagonal_kinetic_part
                    + discretized_diagonal_bare_coulomb_part
                    + diagonal_ecp_local_part
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

            # with all electrons
            else:
                non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic
                # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                e_L = (
                    diagonal_kinetic_part
                    + discretized_diagonal_bare_coulomb_part
                    + diagonal_kinetic_part_SP
                    + non_diagonal_sum_hamiltonian
                )

                p_list = jnp.ravel(elements_non_diagonal_kinetic_part_FN)
                non_diagonal_move_probabilities = p_list / p_list.sum()
                non_diagonal_move_mesh_r_up_carts = mesh_kinetic_part_r_up_carts
                non_diagonal_move_mesh_r_dn_carts = mesh_kinetic_part_r_dn_carts

            logger.devel(f"  e_L={e_L}")
            # """

            # compute the time the walker remaining in the same configuration
            jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
            xi = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
            tau_update = jnp.minimum(tau_left, jnp.log(1 - xi) / non_diagonal_sum_hamiltonian)
            logger.devel(f"  tau_update={tau_update}")

            # update weight
            logger.devel(f"  old: w_L={w_L}")
            w_L = w_L * jnp.exp(-tau_update * e_L)
            logger.devel(f"  new: w_L={w_L}")

            # update tau_left
            tau_left = tau_left - tau_update
            logger.devel(f"tau_left = {tau_left}.")

            # electron position update
            # random choice
            # k = np.random.choice(len(non_diagonal_move_probabilities), p=non_diagonal_move_probabilities)
            jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
            cdf = jnp.cumsum(non_diagonal_move_probabilities)
            random_value = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
            k = jnp.searchsorted(cdf, random_value)
            logger.devel(f"len(non_diagonal_move_probabilities) = {len(non_diagonal_move_probabilities)}.")
            logger.devel(f"chosen update electron index, k = {k}.")
            proposed_r_up_carts = non_diagonal_move_mesh_r_up_carts[k]
            proposed_r_dn_carts = non_diagonal_move_mesh_r_dn_carts[k]

            logger.devel(f"old: r_up_carts = {r_up_carts}")
            logger.devel(f"old: r_dn_carts = {r_dn_carts}")
            new_r_up_carts = jnp.where(tau_left <= 0.0, r_up_carts, proposed_r_up_carts)
            new_r_dn_carts = jnp.where(tau_left <= 0.0, r_dn_carts, proposed_r_dn_carts)
            logger.devel(f"new: r_up_carts={new_r_up_carts}.")
            logger.devel(f"new: r_dn_carts={new_r_dn_carts}.")

            return (e_L, projection_counter, tau_left, w_L, new_r_up_carts, new_r_dn_carts, jax_PRNG_key)

        # projection compilation.
        logger.info("  Compilation is in progress...")
        projection_counter_list = jnp.array([0 for _ in range(self.__num_walkers)])
        tau_left_list = jnp.array([self.__tau for _ in range(self.__num_walkers)])
        w_L_list = jnp.array([1.0 for _ in range(self.__num_walkers)])
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = vmap(_projection, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))(
            projection_counter_list,
            tau_left_list,
            w_L_list,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
            self.__jax_PRNG_key_list,
            self.__non_local_move,
            self.__alat,
            self.__hamiltonian_data,
        )

        end_init = time.perf_counter()
        timer_projection_init += end_init - start_init
        logger.info("End compilation of the GMFC projection funciton.")
        logger.info(f"Elapsed Time = {timer_projection_init:.2f} sec.")
        logger.info("")

        # Main branching loop.
        gfmc_interval = int(np.maximum(num_mcmc_steps / 100, 1))  # gfmc_projection set print-interval

        logger.info("-Start branching-")
        progress = (self.__gfmc_branching_counter) / (num_mcmc_steps + self.__gfmc_branching_counter) * 100.0
        gmfc_total_current = time.perf_counter()
        logger.info(
            f"  branching step = {self.__gfmc_branching_counter}/{num_mcmc_steps + self.__gfmc_branching_counter}: {progress:.1f} %. Elapsed time = {(gmfc_total_current - gmfc_total_start):.1f} sec."
        )

        num_branching_done = 0
        for i_branching in range(num_mcmc_steps):
            if (i_branching + 1) % gfmc_interval == 0:
                progress = (
                    (i_branching + self.__gfmc_branching_counter + 1) / (num_mcmc_steps + self.__gfmc_branching_counter) * 100.0
                )
                gmfc_total_current = time.perf_counter()
                logger.info(
                    f"  branching step = {i_branching + self.__gfmc_branching_counter + 1}/{num_mcmc_steps + self.__gfmc_branching_counter}: {progress:.1f} %. Elapsed time = {(gmfc_total_current - gmfc_total_start):.1f} sec."
                )

            # Always set the initial weight list to 1.0
            projection_counter_list = jnp.array([0 for _ in range(self.__num_walkers)])
            tau_left_list = jnp.array([self.__tau for _ in range(self.__num_walkers)])
            w_L_list = jnp.array([1.0 for _ in range(self.__num_walkers)])

            logger.devel("  Projection is on going....")

            start_projection = time.perf_counter()
            # projection loop
            while True:
                max_progress = (np.max(tau_left_list) / (self.__tau)) * 100.0
                min_progress = (np.min(tau_left_list) / (self.__tau)) * 100.0
                logger.devel(
                    f"  max. Left projection time = {np.max(tau_left_list):.2f}/{self.__tau:.2f}: {max_progress:.1f} %."
                )
                logger.devel(
                    f"  min. Left projection time = {np.min(tau_left_list):.2f}/{self.__tau:.2f}: {min_progress:.1f} %."
                )
                logger.devel(f"  in: w_L_list = {w_L_list}.")
                (
                    e_L_list,
                    projection_counter_list,
                    tau_left_list,
                    w_L_list,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__jax_PRNG_key_list,
                ) = vmap(_projection, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))(
                    projection_counter_list,
                    tau_left_list,
                    w_L_list,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__jax_PRNG_key_list,
                    self.__non_local_move,
                    self.__alat,
                    self.__hamiltonian_data,
                )
                logger.devel(f"  out: w_L_list = {w_L_list}.")
                logger.devel(f"max(tau_left_list) = {np.max(tau_left_list)}.")
                logger.devel(f"min(tau_left_list) = {np.min(tau_left_list)}.")
                if np.max(tau_left_list) <= 0.0:
                    logger.devel(f"max(tau_left_list) = {np.max(tau_left_list)} <= 0.0. Exit the projection loop.")
                    break

            # sync. jax arrays computations.
            e_L_list.block_until_ready()
            projection_counter_list.block_until_ready()
            tau_left_list.block_until_ready()
            w_L_list.block_until_ready()
            self.__latest_r_up_carts.block_until_ready()
            self.__latest_r_dn_carts.block_until_ready()
            self.__jax_PRNG_key_list.block_until_ready()

            end_projection = time.perf_counter()
            timer_projection_total += end_projection - start_projection

            # projection ends
            logger.devel("  Projection ends.")

            # evaluate observables
            start_observable = time.perf_counter()
            # e_L evaluation is not necesarily repeated here.
            """
            if self.__non_local_move == "tmove":
                e_list_debug = vmap(compute_local_energy_api, in_axes=(None, 0, 0))(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                logger.info(f"max(e_list - e_list_debug) = {np.max(np.abs(e_L_list - e_list_debug))}.")
                argmax_i = np.argmax(np.abs(e_L_list - e_list_debug))
                logger.info(f"e_L_list[argmax_i] = {e_L_list[argmax_i]}.")
                logger.info(f"e_list_debug[argmax_i] = {e_list_debug[argmax_i]}.")
                # np.testing.assert_almost_equal(np.array(e_L_list), np.array(e_list_debug), decimal=6)
            """
            # to be implemented other observables, such as derivatives.
            end_observable = time.perf_counter()
            timer_observable += end_observable - start_observable

            # Barrier before MPI operation
            start_mpi_barrier = time.perf_counter()
            mpi_comm.Barrier()
            end_mpi_barrier = time.perf_counter()
            timer_mpi_barrier += end_mpi_barrier - start_mpi_barrier

            # Branching starts
            start_reconfiguration = time.perf_counter()

            # jnp.array -> np.array
            w_L_latest = np.array(w_L_list)
            e_L_latest = np.array(e_L_list)

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

            # num projection counter
            ave_projection_counter = np.mean(projection_counter_list)
            ave_projection_counter_gathered = mpi_comm.gather(ave_projection_counter, root=0)

            if mpi_rank == 0:
                logger.devel(f"e_L_gathered_dyad={e_L_gathered_dyad}")
                logger.devel(f"w_L_gathered_dyad={w_L_gathered_dyad}")
                r_up_carts_gathered_dict = dict(r_up_carts_gathered_dyad)
                r_dn_carts_gathered_dict = dict(r_dn_carts_gathered_dyad)
                e_L_gathered_dict = dict(e_L_gathered_dyad)
                w_L_gathered_dict = dict(w_L_gathered_dyad)
                logger.devel(f"  e_L_gathered_dict = {e_L_gathered_dict} Ha")
                logger.devel(f"  w_L_gathered_dict = {w_L_gathered_dict}")
                r_up_carts_gathered = np.concatenate([r_up_carts_gathered_dict[i] for i in range(mpi_size)])
                r_dn_carts_gathered = np.concatenate([r_dn_carts_gathered_dict[i] for i in range(mpi_size)])
                e_L_gathered = np.concatenate([e_L_gathered_dict[i] for i in range(mpi_size)])
                w_L_gathered = np.concatenate([w_L_gathered_dict[i] for i in range(mpi_size)])
                logger.devel(f"  e_L_gathered = {e_L_gathered} Ha")
                logger.devel(f"  w_L_gathered = {w_L_gathered}")
                e_L2_averaged = np.sum(w_L_gathered * e_L_gathered**2) / np.sum(w_L_gathered)
                e_L_averaged = np.sum(w_L_gathered * e_L_gathered) / np.sum(w_L_gathered)
                w_L_averaged = np.average(w_L_gathered)
                logger.devel(f"  e_L_averaged = {e_L_averaged} Ha")
                logger.devel(f"  w_L_averaged = {w_L_averaged}")
                # add a dummy dim
                e_L2_averaged = np.expand_dims(e_L2_averaged, axis=0)
                e_L_averaged = np.expand_dims(e_L_averaged, axis=0)
                w_L_averaged = np.expand_dims(w_L_averaged, axis=0)
                # store  # This should stored only for MPI-rank = 0 !!!
                self.__stored_e_L2.append(e_L2_averaged)
                self.__stored_e_L.append(e_L_averaged)
                self.__stored_w_L.append(w_L_averaged)
                w_L_list = w_L_gathered
                logger.devel(f"w_L_list = {w_L_list}")
                probabilities = w_L_list / w_L_list.sum()
                logger.devel(f"probabilities = {probabilities}")

                # correlated choice (see Sandro's textbook, page 182)
                zeta = float(np.random.random())
                # self.__jax_PRNG_key, subkey = jax.random.split(self.__jax_PRNG_key) # slow w/o jit!!
                # zeta = float(jax.random.uniform(subkey, minval=0.0, maxval=1.0)) # slow w/o jit!!
                z_list = [(alpha + zeta) / len(probabilities) for alpha in range(len(probabilities))]
                logger.devel(f"z_list = {z_list}")
                cumulative_prob = np.cumsum(probabilities)
                chosen_walker_indices = np.array(
                    [next(idx for idx, prob in enumerate(cumulative_prob) if z <= prob) for z in z_list]
                )
                logger.devel(f"The chosen walker indices = {chosen_walker_indices}")
                logger.devel(f"The chosen walker indices.shape = {chosen_walker_indices.shape}")
                logger.devel(f"r_up_carts_gathered.shape = {r_up_carts_gathered.shape}")
                logger.devel(f"r_dn_carts_gathered.shape = {r_dn_carts_gathered.shape}")

                proposed_r_up_carts = r_up_carts_gathered[chosen_walker_indices]
                proposed_r_dn_carts = r_dn_carts_gathered[chosen_walker_indices]

                self.__num_survived_walkers += len(set(chosen_walker_indices))
                self.__num_killed_walkers += len(w_L_list) - len(set(chosen_walker_indices))
                self.__stored_average_projection_counter.append(np.mean(ave_projection_counter_gathered))
                logger.devel(f"num_survived_walkers={self.__num_survived_walkers}")
                logger.devel(f"num_killed_walkers={self.__num_killed_walkers}")
            else:
                self.__num_survived_walkers = None
                self.__num_killed_walkers = None
                self.__stored_average_projection_counter = None
                proposed_r_up_carts = None
                proposed_r_dn_carts = None

            logger.devel(f"Before branching: rank={mpi_rank}:gfmc.r_up_carts = {self.__latest_r_up_carts}")
            logger.devel(f"Before branching: rank={mpi_rank}:gfmc.r_dn_carts = {self.__latest_r_dn_carts}")

            self.__num_survived_walkers = mpi_comm.bcast(self.__num_survived_walkers, root=0)
            self.__num_killed_walkers = mpi_comm.bcast(self.__num_killed_walkers, root=0)
            self.__stored_average_projection_counter = mpi_comm.bcast(self.__stored_average_projection_counter, root=0)

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

            logger.devel(f"*After branching: rank={mpi_rank}:gfmc.r_up_carts = {self.__latest_r_up_carts}")
            logger.devel(f"*After branching: rank={mpi_rank}:gfmc.r_dn_carts = {self.__latest_r_dn_carts}")

            end_reconfiguration = time.perf_counter()
            timer_reconfiguration += end_reconfiguration - start_reconfiguration

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
        logger.info(f"  MPI barrier time per branching = {timer_mpi_barrier / num_branching_done * 10**3: .3f} msec.")
        logger.info(
            f"  Walker reconfiguration time per branching = {timer_reconfiguration / num_branching_done * 10**3: .3f} msec."
        )
        logger.devel(f"Survived walkers = {self.__num_survived_walkers}")
        logger.devel(f"killed walkers = {self.__num_killed_walkers}")
        logger.info(
            f"Survived walkers ratio = {self.__num_survived_walkers / (self.__num_survived_walkers + self.__num_killed_walkers) * 100:.2f} %"
        )
        logger.info(f"Ave. projection num. = {np.mean(self.__stored_average_projection_counter)}")
        logger.info("")

        self.__timer_gmfc_total += timer_gmfc_total
        self.__timer_projection_init += timer_projection_init
        self.__timer_projection_total += timer_projection_total
        self.__timer_branching += timer_reconfiguration
        self.__timer_observable += timer_observable


class GFMC_fixed_num_projection:
    """GFMC class. Runing GFMC with multiple walkers.

    Args:
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
        num_walkers (int): the number of walkers
        mcmc_seed (int): seed for the MCMC chain.
        E_scf (float): Self-consistent E (Hartree)
        alat (float): discretized grid length (bohr)
        non_local_move (str):
            treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
            Valid only for ECP calculations. Do not specify this value for all-electron calculations.
        comput_position_deriv (bool): if True, compute the derivatives of E wrt. atomic positions.
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data = None,
        num_walkers: int = 40,
        num_mcmc_per_measurement: int = 16,
        num_gfmc_collect_steps: int = 5,
        mcmc_seed: int = 34467,
        E_scf: float = 0.0,
        alat: float = 0.1,
        non_local_move: str = "tmove",
        comput_position_deriv: bool = False,
    ) -> None:
        """Init.

        Initialize a GFMC class, creating list holding results, etc...

        """
        # check sanity of hamiltonian_data
        hamiltonian_data.sanity_check()

        # attributes
        self.__hamiltonian_data = hamiltonian_data
        self.__num_walkers = num_walkers
        self.__num_mcmc_per_measurement = num_mcmc_per_measurement
        self.__num_gfmc_collect_steps = num_gfmc_collect_steps
        self.__mcmc_seed = mcmc_seed
        self.__E_scf = E_scf
        self.__alat = alat
        self.__non_local_move = non_local_move

        # timer for GFMC
        self.__timer_gfmc_init = 0.0
        self.__timer_gfmc_total = 0.0
        self.__timer_projection_init = 0.0
        self.__timer_projection_total = 0.0
        self.__timer_branching = 0.0
        # time for observables
        self.__timer_e_L = 0.0
        self.__timer_de_L_dR_dr = 0.0
        self.__timer_dln_Psi_dR_dr = 0.0
        self.__timer_dln_Psi_dc = 0.0
        self.__timer_de_L_dc = 0.0
        self.__timer_misc = 0.0

        # derivative flags
        self.__comput_position_deriv = comput_position_deriv

        start = time.perf_counter()
        # Initialization
        self.__mpi_seed = self.__mcmc_seed * (mpi_rank + 1)
        self.__jax_PRNG_key = jax.random.PRNGKey(self.__mpi_seed)
        self.__jax_PRNG_key_list_init = jnp.array(
            [jax.random.fold_in(self.__jax_PRNG_key, nw) for nw in range(self.__num_walkers)]
        )
        self.__jax_PRNG_key_list = self.__jax_PRNG_key_list_init

        # Place electrons around each nucleus with improved spin assignment

        tot_num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        tot_num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn

        r_carts_up_list = []
        r_carts_dn_list = []

        np.random.seed(self.__mpi_seed)

        for _ in range(self.__num_walkers):
            # Initialization
            r_carts_up = []
            r_carts_dn = []
            total_assigned_up = 0
            total_assigned_dn = 0

            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
                    hamiltonian_data.coulomb_potential_data.z_cores
                )
            else:
                charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

            logger.devel(f"charges = {charges}.")
            coords = hamiltonian_data.structure_data.positions_cart_jnp

            # Place electrons for each atom
            for i in range(len(coords)):
                charge = charges[i]
                n_elec = int(np.round(charge))  # Number of electrons to place for this atom
                logger.devel(f"charge = {charge}.")
                logger.devel(f"num_electrons = {n_elec}.")

                x, y, z = coords[i]

                # Determine the number of up and dn electrons for this atom
                if n_elec % 2 == 0:
                    num_up = n_elec // 2
                    num_dn = n_elec // 2
                else:
                    # For odd numbers, decide which spin gets the extra electron based on the overall shortage
                    remaining_up = tot_num_electron_up - total_assigned_up
                    remaining_dn = tot_num_electron_dn - total_assigned_dn
                    if remaining_up >= remaining_dn:
                        num_up = n_elec // 2 + 1
                        num_dn = n_elec // 2
                    else:
                        num_up = n_elec // 2
                        num_dn = n_elec // 2 + 1

                # Place up electrons
                for _ in range(num_up):
                    distance = np.random.uniform(0.1, 1.0)
                    theta = np.random.uniform(0, np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    dx = distance * np.sin(theta) * np.cos(phi)
                    dy = distance * np.sin(theta) * np.sin(phi)
                    dz = distance * np.cos(theta)
                    electron_position = np.array([x + dx, y + dy, z + dz])
                    r_carts_up.append(electron_position)
                total_assigned_up += num_up

                # Place dn electrons
                for _ in range(num_dn):
                    distance = np.random.uniform(0.1, 1.0)
                    theta = np.random.uniform(0, np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    dx = distance * np.sin(theta) * np.cos(phi)
                    dy = distance * np.sin(theta) * np.sin(phi)
                    dz = distance * np.cos(theta)
                    electron_position = np.array([x + dx, y + dy, z + dz])
                    r_carts_dn.append(electron_position)
                total_assigned_dn += num_dn

            # Electron assignment for all atoms is complete
            logger.devel(f"Total assigned up electrons: {total_assigned_up} (target {tot_num_electron_up}).")
            logger.devel(f"Total assigned dn electrons: {total_assigned_dn} (target {tot_num_electron_dn}).")

            # If necessary, include a check/adjustment step to ensure the overall assignment matches the targets
            # (Here it is assumed that sum(round(charge)) equals tot_num_electron_up + tot_num_electron_dn)

            r_carts_up_list.append(r_carts_up)
            r_carts_dn_list.append(r_carts_dn)

        self.__latest_r_up_carts = jnp.array(r_carts_up_list)
        self.__latest_r_dn_carts = jnp.array(r_carts_dn_list)

        logger.info(f"The number of MPI process = {mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self.__num_walkers}.")

        logger.devel(f"initial r_up_carts= {self.__latest_r_up_carts}")
        logger.devel(f"initial r_dn_carts = {self.__latest_r_dn_carts}")
        logger.devel(f"initial r_up_carts.shape = {self.__latest_r_up_carts.shape}")
        logger.devel(f"initial r_dn_carts.shape = {self.__latest_r_dn_carts.shape}")
        logger.info("")

        # SWCT data
        self.__swct_data = SWCT_data(structure=self.__hamiltonian_data.structure_data)

        # print out hamiltonian info
        logger.info("Printing out information in hamitonian_data instance.")
        self.__hamiltonian_data.logger_info()
        logger.info("")

        logger.info("Compilation of fundamental functions starts.")

        logger.info("  Compilation e_L starts.")
        _ = compute_kinetic_energy_jax(
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts[0],
            r_dn_carts=self.__latest_r_dn_carts[0],
        )
        _, _, _ = compute_discretized_kinetic_energy_jax(
            alat=self.__alat,
            wavefunction_data=self.__hamiltonian_data.wavefunction_data,
            r_up_carts=self.__latest_r_up_carts[0],
            r_dn_carts=self.__latest_r_dn_carts[0],
            RT=jnp.eye(3, 3),
        )
        _ = compute_bare_coulomb_potential_jax(
            coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
            r_up_carts=self.__latest_r_up_carts[0],
            r_dn_carts=self.__latest_r_dn_carts[0],
        )
        if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
            _ = compute_ecp_local_parts_all_pairs_jax(
                coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                r_up_carts=self.__latest_r_up_carts[0],
                r_dn_carts=self.__latest_r_dn_carts[0],
            )
            if self.__non_local_move == "tmove":
                _, _, _, _ = compute_ecp_non_local_parts_nearest_neighbors_jax(
                    coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=self.__latest_r_up_carts[0],
                    r_dn_carts=self.__latest_r_dn_carts[0],
                    flag_determinant_only=False,
                )
            elif self.__non_local_move == "dltmove":
                _, _, _, _ = compute_ecp_non_local_parts_nearest_neighbors_jax(
                    coulomb_potential_data=self.__hamiltonian_data.coulomb_potential_data,
                    wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                    r_up_carts=self.__latest_r_up_carts[0],
                    r_dn_carts=self.__latest_r_dn_carts[0],
                    flag_determinant_only=True,
                )
            else:
                logger.error(f"non_local_move = {self.__non_local_move} is not yet implemented.")
                raise NotImplementedError

        _ = compute_G_L(np.zeros((self.__num_gfmc_collect_steps * 2, 1)), self.__num_gfmc_collect_steps)

        end = time.perf_counter()
        self.__timer_gfmc_init += end - start
        logger.info("  Compilation e_L is done.")

        if self.__comput_position_deriv:
            logger.info("  Compilation dln_Psi/dR starts.")
            start = time.perf_counter()
            _, _, _ = grad(evaluate_ln_wavefunction_jax, argnums=(0, 1, 2))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts[0],
                self.__latest_r_dn_carts[0],
            )
            end = time.perf_counter()
            logger.info("  Compilation dln_Psi/dR is done.")
            logger.info(f"  Elapsed Time = {end - start:.2f} sec.")
            self.__timer_gfmc_init += end - start

            logger.info("  Compilation domega/dR starts.")
            start = time.perf_counter()
            _ = evaluate_swct_domega_jax(
                self.__swct_data,
                self.__latest_r_up_carts[0],
            )
            end = time.perf_counter()
            logger.info("  Compilation domega/dR is done.")
            logger.info(f"  Elapsed Time = {end - start:.2f} sec.")
            self.__timer_gfmc_init += end - start

        logger.info("Compilation of fundamental functions is done.")
        logger.info(f"Elapsed Time = {self.__timer_gfmc_init:.2f} sec.")
        logger.info("")

        # init attributes
        self.__init_attributes()

    def __init_attributes(self):
        # mcmc counter
        self.__mcmc_counter = 0

        # gfmc accepted/rejected moves
        self.__num_survived_walkers = 0
        self.__num_killed_walkers = 0

        # stored local energy (w_L)
        self.__stored_w_L = []

        # stored local energy (e_L)
        self.__stored_e_L = []

        # stored local energy (e_L2)
        self.__stored_e_L2 = []

        # stored de_L / dR
        self.__stored_grad_e_L_dR = []

        # stored de_L / dr_up
        self.__stored_grad_e_L_r_up = []

        # stored de_L / dr_dn
        self.__stored_grad_e_L_r_dn = []

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

    # hamiltonian
    @property
    def hamiltonian_data(self):
        """Return hamiltonian_data."""
        return self.__hamiltonian_data

    @hamiltonian_data.setter
    def hamiltonian_data(self, hamiltonian_data):
        """Set hamiltonian_data."""
        self.__hamiltonian_data = hamiltonian_data
        self.__init_attributes()

    # collecting factor
    @property
    def num_gfmc_collect_steps(self):
        """Return num_gfmc_collect_steps."""
        return self.__num_gfmc_collect_steps

    @num_gfmc_collect_steps.setter
    def num_gfmc_collect_steps(self, num_gfmc_collect_steps):
        """Set num_gfmc_collect_steps."""
        self.__num_gfmc_collect_steps = num_gfmc_collect_steps

    # dimensions of observables
    @property
    def mcmc_counter(self) -> int:
        """Return current MCMC counter."""
        return self.__mcmc_counter - self.__num_gfmc_collect_steps

    @property
    def num_walkers(self):
        """The number of walkers."""
        return self.__num_walkers

    # weights
    @property
    def w_L(self) -> npt.NDArray:
        """Return the stored weight array. dim: (mcmc_counter, 1)."""
        # logger.info(f"np.array(self.__stored_w_L).shape = {np.array(self.__stored_w_L).shape}.")
        return compute_G_L(np.array(self.__stored_w_L), self.__num_gfmc_collect_steps)

    # observables
    @property
    def e_L(self) -> npt.NDArray:
        """Return the stored e_L array. dim: (mcmc_counter, 1)."""
        # logger.info(f"np.array(self.__stored_e_L).shape = {np.array(self.__stored_e_L).shape}.")
        return np.array(self.__stored_e_L)[self.__num_gfmc_collect_steps :]

    # observables
    @property
    def e_L2(self) -> npt.NDArray:
        """Return the stored e_L^2 array. dim: (mcmc_counter, 1)."""
        # logger.info(f"np.array(self.__stored_e_L2).shape = {np.array(self.__stored_e_L).shape}.")
        return np.array(self.__stored_e_L2)[self.__num_gfmc_collect_steps :]

    @property
    def de_L_dR(self) -> npt.NDArray:
        """Return the stored de_L/dR array. dim: (mcmc_counter, 1)."""
        return np.array(self.__stored_grad_e_L_dR)[self.__num_gfmc_collect_steps :]

    @property
    def de_L_dr_up(self) -> npt.NDArray:
        """Return the stored de_L/dr_up array. dim: (mcmc_counter, 1, num_electrons_up, 3)."""
        return np.array(self.__stored_grad_e_L_r_up)[self.__num_gfmc_collect_steps :]

    @property
    def de_L_dr_dn(self) -> npt.NDArray:
        """Return the stored de_L/dr_dn array. dim: (mcmc_counter, 1, num_electrons_dn, 3)."""
        return np.array(self.__stored_grad_e_L_r_dn)[self.__num_gfmc_collect_steps :]

    @property
    def dln_Psi_dr_up(self) -> npt.NDArray:
        """Return the stored dln_Psi/dr_up array. dim: (mcmc_counter, 1, num_electrons_up, 3)."""
        return np.array(self.__stored_grad_ln_Psi_r_up)[self.__num_gfmc_collect_steps :]

    @property
    def dln_Psi_dr_dn(self) -> npt.NDArray:
        """Return the stored dln_Psi/dr_down array. dim: (mcmc_counter, 1, num_electrons_dn, 3)."""
        return np.array(self.__stored_grad_ln_Psi_r_dn)[self.__num_gfmc_collect_steps :]

    @property
    def dln_Psi_dR(self) -> npt.NDArray:
        """Return the stored dln_Psi/dR array. dim: (mcmc_counter, 1, num_atoms, 3)."""
        return np.array(self.__stored_grad_ln_Psi_dR)[self.__num_gfmc_collect_steps :]

    @property
    def omega_up(self) -> npt.NDArray:
        """Return the stored Omega (for up electrons) array. dim: (mcmc_counter, 1, num_atoms, num_electrons_up)."""
        return np.array(self.__stored_omega_up)[self.__num_gfmc_collect_steps :]

    @property
    def omega_dn(self) -> npt.NDArray:
        """Return the stored Omega (for down electrons) array. dim: (mcmc_counter,1, num_atoms, num_electons_dn)."""
        return np.array(self.__stored_omega_dn)[self.__num_gfmc_collect_steps :]

    @property
    def domega_dr_up(self) -> npt.NDArray:
        """Return the stored dOmega/dr_up array. dim: (mcmc_counter, 1, num_electons_dn, 3)."""
        return np.array(self.__stored_grad_omega_r_up)[self.__num_gfmc_collect_steps :]

    @property
    def domega_dr_dn(self) -> npt.NDArray:
        """Return the stored dOmega/dr_dn array. dim: (mcmc_counter, 1, num_electons_dn, 3)."""
        return np.array(self.__stored_grad_omega_r_dn)[self.__num_gfmc_collect_steps :]

    @property
    def comput_position_deriv(self) -> bool:
        """Return the flag for computing the derivatives of E wrt. atomic positions."""
        return self.__comput_position_deriv

    def run(self, num_mcmc_steps: int = 50, max_time: int = 86400) -> None:
        """Run LRDMC with multiple walkers.

        Args:
            num_branching (int): number of branching (reconfiguration of walkers).
            max_time (int): maximum time in sec.
        """
        # initialize numpy random seed
        np.random.seed(self.__mpi_seed)

        # set timer
        timer_projection_init = 0.0
        timer_projection_total = 0.0
        timer_e_L = 0.0
        timer_de_L_dR_dr = 0.0
        timer_dln_Psi_dR_dr = 0.0
        timer_dln_Psi_dc = 0.0
        timer_de_L_dc = 0.0
        timer_reconfiguration = 0.0

        gfmc_total_start = time.perf_counter()

        # projection function.
        start_init = time.perf_counter()
        logger.info("Start compilation of the GFMC projection funciton.")

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
            num_mcmc_per_measurement: int,
            non_local_move: bool,
            alat: float,
            hamiltonian_data: Hamiltonian_data,
        ):
            """Do projection, compatible with vmap.

            Do projection for a set of (r_up_cart, r_dn_cart).

            Args:
                E(float): trial total energy
                init_w_L (float): weight before projection
                init_r_up_carts (N_e^up, 3) before projection
                init_r_dn_carts (N_e^dn, 3) before projection
                E_scf (float): Self-consistent E (Hartree)
                num_mcmc_per_measurement (int): the number of MCMC steps per measurement
                non_local_move (bool): treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
                alat (float): discretized grid length (bohr)
                hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data

            Returns:
                latest_w_L (float): weight after the final projection
                latest_r_up_carts (N_e^up, 3) after the final projection
                latest_r_dn_carts (N_e^dn, 3) after the final projection
            """
            logger.devel(f"init_jax_PRNG_key={init_jax_PRNG_key}")

            @jit
            def body_fun(_, carry):
                w_L, r_up_carts, r_dn_carts, jax_PRNG_key = carry

                # compute diagonal elements, kinetic part
                diagonal_kinetic_part = 3.0 / (2.0 * alat**2) * (len(r_up_carts) + len(r_dn_carts))

                # compute continuum kinetic energy
                diagonal_kinetic_continuum_elements_up, diagonal_kinetic_continuum_elements_dn = (
                    compute_kinetic_energy_all_elements_jax(
                        wavefunction_data=hamiltonian_data.wavefunction_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )
                )

                # generate a random rotation matrix
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                alpha, beta, gamma = jax.random.uniform(
                    subkey, shape=(3,), minval=-2 * jnp.pi, maxval=2 * jnp.pi
                )  # Rotation angle around the x,y,z-axis (in radians)

                R = generate_rotation_matrix(alpha, beta, gamma)  # Rotate in the order x -> y -> z

                # compute discretized kinetic energy and mesh (with a random rotation)
                mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                    compute_discretized_kinetic_energy_jax(
                        alat=alat,
                        wavefunction_data=hamiltonian_data.wavefunction_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                        RT=R.T,
                    )
                )
                # spin-filp
                elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
                non_diagonal_sum_hamiltonian_kinetic = jnp.sum(elements_non_diagonal_kinetic_part_FN)
                diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
                # regularizations
                elements_non_diagonal_kinetic_part_all = elements_non_diagonal_kinetic_part.reshape(-1, 6)
                sign_flip_flags_elements = jnp.any(elements_non_diagonal_kinetic_part_all >= 0, axis=1)
                non_diagonal_kinetic_part_elements = jnp.sum(
                    elements_non_diagonal_kinetic_part_all + 1.0 / (4.0 * alat**2), axis=1
                )
                sign_flip_flags_elements_up, sign_flip_flags_elements_dn = jnp.split(
                    sign_flip_flags_elements, [len(r_up_carts)]
                )
                non_diagonal_kinetic_part_elements_up, non_diagonal_kinetic_part_elements_dn = jnp.split(
                    non_diagonal_kinetic_part_elements, [len(r_up_carts)]
                )

                # compute diagonal elements, el-el
                diagonal_bare_coulomb_part_el_el = compute_bare_coulomb_potential_el_el_jax(
                    r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
                )

                # compute diagonal elements, ion-ion
                diagonal_bare_coulomb_part_ion_ion = compute_bare_coulomb_potential_ion_ion_jax(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data
                )

                # compute diagonal elements, el-ion
                diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                    compute_bare_coulomb_potential_el_ion_element_wise_jax(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )
                )

                # compute diagonal elements, el-ion, discretized
                (
                    diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                    diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
                ) = compute_discretized_bare_coulomb_potential_el_ion_element_wise_jax(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                    alat=alat,
                )

                # compose discretized el-ion potentials
                diagonal_bare_coulomb_part_el_ion_zv_up = (
                    diagonal_bare_coulomb_part_el_ion_elements_up
                    + diagonal_kinetic_continuum_elements_up
                    - non_diagonal_kinetic_part_elements_up
                )
                # """
                # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
                # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effectice core potential calculations. (c.f., parcutg=1 in TurboRVB).
                if hamiltonian_data.coulomb_potential_data.ecp_flag:
                    diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_elements_up
                else:
                    diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_discretized_elements_up
                diagonal_bare_coulomb_part_el_ion_max_up = jnp.maximum(
                    diagonal_bare_coulomb_part_el_ion_zv_up, diagonal_bare_coulomb_part_el_ion_ei_up
                )
                diagonal_bare_coulomb_part_el_ion_opt_up = jnp.where(
                    sign_flip_flags_elements_up,
                    diagonal_bare_coulomb_part_el_ion_max_up,
                    diagonal_bare_coulomb_part_el_ion_zv_up,
                )
                # diagonal_bare_coulomb_part_el_ion_opt_up = (
                #    diagonal_bare_coulomb_part_el_ion_max_up  # more strict regularization
                # )
                # diagonal_bare_coulomb_part_el_ion_opt_up = diagonal_bare_coulomb_part_el_ion_zv_up # for debug
                # """

                # compose discretized el-ion potentials
                diagonal_bare_coulomb_part_el_ion_zv_dn = (
                    diagonal_bare_coulomb_part_el_ion_elements_dn
                    + diagonal_kinetic_continuum_elements_dn
                    - non_diagonal_kinetic_part_elements_dn
                )
                # """
                # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
                # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effectice core potential calculations. (c.f., parcutg=1 in TurboRVB).
                if hamiltonian_data.coulomb_potential_data.ecp_flag:
                    diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_elements_dn
                else:
                    diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_discretized_elements_dn
                diagonal_bare_coulomb_part_el_ion_max_dn = jnp.maximum(
                    diagonal_bare_coulomb_part_el_ion_zv_dn, diagonal_bare_coulomb_part_el_ion_ei_dn
                )
                diagonal_bare_coulomb_part_el_ion_opt_dn = jnp.where(
                    sign_flip_flags_elements_dn,
                    diagonal_bare_coulomb_part_el_ion_max_dn,
                    diagonal_bare_coulomb_part_el_ion_zv_dn,
                )
                # diagonal_bare_coulomb_part_el_ion_opt_dn = (
                #    diagonal_bare_coulomb_part_el_ion_max_dn  # more strict regularization
                # )
                # diagonal_bare_coulomb_part_el_ion_opt_dn = diagonal_bare_coulomb_part_el_ion_zv_dn # for debug
                # """

                # final bare coulomb part
                discretized_diagonal_bare_coulomb_part = (
                    diagonal_bare_coulomb_part_el_el
                    + diagonal_bare_coulomb_part_ion_ion
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_up)
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_dn)
                )

                # """ if-else for all-ele, ecp with tmove, and ecp with dltmove
                # with ECP
                if hamiltonian_data.coulomb_potential_data.ecp_flag:
                    # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                    # ecp local
                    diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs_jax(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    if non_local_move == "tmove":
                        # ecp non-local (t-move)
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors_jax(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
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
                            diagonal_kinetic_part
                            + discretized_diagonal_bare_coulomb_part
                            + diagonal_ecp_local_part
                            + diagonal_kinetic_part_SP
                            + diagonal_ecp_part_SP
                        )

                    elif non_local_move == "dltmove":
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors_jax(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=True,
                            )
                        )

                        V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                        diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))

                        Jastrow_ref = compute_Jastrow_part_jax(
                            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                        )
                        Jastrow_on_mesh = vmap(compute_Jastrow_part_jax, in_axes=(None, 0, 0))(
                            hamiltonian_data.wavefunction_data.jastrow_data,
                            mesh_non_local_ecp_part_r_up_carts,
                            mesh_non_local_ecp_part_r_dn_carts,
                        )
                        Jastrow_ratio = jnp.exp(Jastrow_on_mesh - Jastrow_ref)

                        V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

                        non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                        non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                        diagonal_sum_hamiltonian = (
                            diagonal_kinetic_part
                            + discretized_diagonal_bare_coulomb_part
                            + diagonal_ecp_local_part
                            + diagonal_kinetic_part_SP
                            + diagonal_ecp_part_SP
                        )

                    else:
                        logger.error(f"non_local_move = {non_local_move} is not yet implemented.")
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
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic
                    # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                    p_list = jnp.ravel(elements_non_diagonal_kinetic_part_FN)
                    non_diagonal_move_probabilities = p_list / p_list.sum()
                    non_diagonal_move_mesh_r_up_carts = mesh_kinetic_part_r_up_carts
                    non_diagonal_move_mesh_r_dn_carts = mesh_kinetic_part_r_dn_carts

                    diagonal_sum_hamiltonian = (
                        diagonal_kinetic_part + discretized_diagonal_bare_coulomb_part + diagonal_kinetic_part_SP
                    )

                # compute b_L_bar
                b_x_bar = -1.0 * non_diagonal_sum_hamiltonian
                logger.devel(f"  b_x_bar={b_x_bar}")

                # compute bar_b_L
                logger.devel(f"  diagonal_sum_hamiltonian={diagonal_sum_hamiltonian}")
                logger.devel(f"  E_scf={E_scf}")
                b_x = 1.0 / (diagonal_sum_hamiltonian - E_scf) * b_x_bar
                logger.devel(f"  b_x={b_x}")

                # update weight
                logger.devel(f"  old: w_L={w_L}")
                w_L = w_L * b_x
                logger.devel(f"  new: w_L={w_L}")

                # electron position update
                # random choice
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                cdf = jnp.cumsum(non_diagonal_move_probabilities)
                random_value = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
                k = jnp.searchsorted(cdf, random_value)
                logger.devel(f"len(non_diagonal_move_probabilities) = {len(non_diagonal_move_probabilities)}.")
                logger.devel(f"chosen update electron index, k = {k}.")
                logger.devel(f"old: r_up_carts = {r_up_carts}")
                logger.devel(f"old: r_dn_carts = {r_dn_carts}")
                r_up_carts = non_diagonal_move_mesh_r_up_carts[k]
                r_dn_carts = non_diagonal_move_mesh_r_dn_carts[k]
                logger.devel(f"new: r_up_carts={r_up_carts}.")
                logger.devel(f"new: r_dn_carts={r_dn_carts}.")

                carry = (w_L, r_up_carts, r_dn_carts, jax_PRNG_key)
                return carry

            latest_w_L, latest_r_up_carts, latest_r_dn_carts, latest_jax_PRNG_key = jax.lax.fori_loop(
                0, num_mcmc_per_measurement, body_fun, (init_w_L, init_r_up_carts, init_r_dn_carts, init_jax_PRNG_key)
            )

            return (latest_w_L, latest_r_up_carts, latest_r_dn_carts, latest_jax_PRNG_key)

        @partial(jit, static_argnums=4)
        def _compute_V_elements(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jax.Array,
            r_dn_carts: jax.Array,
            jax_PRNG_key: jax.Array,
            non_local_move: bool,
            alat: float,
        ):
            """Compute V elements."""
            #''' coulomb reguralization
            # compute diagonal elements, kinetic part
            diagonal_kinetic_part = 3.0 / (2.0 * alat**2) * (len(r_up_carts) + len(r_dn_carts))

            # compute continuum kinetic energy
            diagonal_kinetic_continuum_elements_up, diagonal_kinetic_continuum_elements_dn = (
                compute_kinetic_energy_all_elements_jax(
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # generate a random rotation matrix
            jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
            alpha, beta, gamma = jax.random.uniform(
                subkey, shape=(3,), minval=-2 * jnp.pi, maxval=2 * jnp.pi
            )  # Rotation angle around the x,y,z-axis (in radians)

            R = generate_rotation_matrix(alpha, beta, gamma)  # Rotate in the order x -> y -> z

            # compute discretized kinetic energy and mesh (with a random rotation)
            _, _, elements_non_diagonal_kinetic_part = compute_discretized_kinetic_energy_jax(
                alat=alat,
                wavefunction_data=hamiltonian_data.wavefunction_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                RT=R.T,
            )
            # spin-filp
            elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
            non_diagonal_sum_hamiltonian_kinetic = jnp.sum(elements_non_diagonal_kinetic_part_FN)
            diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
            # regularizations
            elements_non_diagonal_kinetic_part_all = elements_non_diagonal_kinetic_part.reshape(-1, 6)
            sign_flip_flags_elements = jnp.any(elements_non_diagonal_kinetic_part_all >= 0, axis=1)
            non_diagonal_kinetic_part_elements = jnp.sum(elements_non_diagonal_kinetic_part_all + 1.0 / (4.0 * alat**2), axis=1)
            sign_flip_flags_elements_up, sign_flip_flags_elements_dn = jnp.split(sign_flip_flags_elements, [len(r_up_carts)])
            non_diagonal_kinetic_part_elements_up, non_diagonal_kinetic_part_elements_dn = jnp.split(
                non_diagonal_kinetic_part_elements, [len(r_up_carts)]
            )

            # compute diagonal elements, el-el
            diagonal_bare_coulomb_part_el_el = compute_bare_coulomb_potential_el_el_jax(
                r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
            )

            # compute diagonal elements, ion-ion
            diagonal_bare_coulomb_part_ion_ion = compute_bare_coulomb_potential_ion_ion_jax(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data
            )

            # compute diagonal elements, el-ion
            diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                compute_bare_coulomb_potential_el_ion_element_wise_jax(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # compute diagonal elements, el-ion, discretized
            (
                diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
            ) = compute_discretized_bare_coulomb_potential_el_ion_element_wise_jax(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                alat=alat,
            )

            # compose discretized el-ion potentials
            diagonal_bare_coulomb_part_el_ion_zv_up = (
                diagonal_bare_coulomb_part_el_ion_elements_up
                + diagonal_kinetic_continuum_elements_up
                - non_diagonal_kinetic_part_elements_up
            )
            # """
            # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
            # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effectice core potential calculations. (c.f., parcutg=1 in TurboRVB).
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_elements_up
            else:
                diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_discretized_elements_up
            diagonal_bare_coulomb_part_el_ion_max_up = jnp.maximum(
                diagonal_bare_coulomb_part_el_ion_zv_up, diagonal_bare_coulomb_part_el_ion_ei_up
            )
            diagonal_bare_coulomb_part_el_ion_opt_up = jnp.where(
                sign_flip_flags_elements_up, diagonal_bare_coulomb_part_el_ion_max_up, diagonal_bare_coulomb_part_el_ion_zv_up
            )
            # diagonal_bare_coulomb_part_el_ion_opt_up = diagonal_bare_coulomb_part_el_ion_max_up  # more strict regularization
            # diagonal_bare_coulomb_part_el_ion_opt_up = diagonal_bare_coulomb_part_el_ion_zv_up # for debug
            # """

            # compose discretized el-ion potentials
            diagonal_bare_coulomb_part_el_ion_zv_dn = (
                diagonal_bare_coulomb_part_el_ion_elements_dn
                + diagonal_kinetic_continuum_elements_dn
                - non_diagonal_kinetic_part_elements_dn
            )
            # """
            # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
            # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effectice core potential calculations. (c.f., parcutg=1 in TurboRVB).
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_elements_dn
            else:
                diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_discretized_elements_dn
            diagonal_bare_coulomb_part_el_ion_max_dn = jnp.maximum(
                diagonal_bare_coulomb_part_el_ion_zv_dn, diagonal_bare_coulomb_part_el_ion_ei_dn
            )
            diagonal_bare_coulomb_part_el_ion_opt_dn = jnp.where(
                sign_flip_flags_elements_dn, diagonal_bare_coulomb_part_el_ion_max_dn, diagonal_bare_coulomb_part_el_ion_zv_dn
            )
            # diagonal_bare_coulomb_part_el_ion_opt_dn = diagonal_bare_coulomb_part_el_ion_max_dn  # more strict regularization
            # diagonal_bare_coulomb_part_el_ion_opt_dn = diagonal_bare_coulomb_part_el_ion_zv_dn # for debug
            # """

            # final bare coulomb part
            discretized_diagonal_bare_coulomb_part = (
                diagonal_bare_coulomb_part_el_el
                + diagonal_bare_coulomb_part_ion_ion
                + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_up)
                + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_dn)
            )
            #'''

            # with ECP
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                # ecp local
                diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs_jax(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

                if non_local_move == "tmove":
                    # ecp non-local (t-move)
                    mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                        compute_ecp_non_local_parts_nearest_neighbors_jax(
                            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                            wavefunction_data=hamiltonian_data.wavefunction_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                            flag_determinant_only=False,
                        )
                    )

                    V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                    diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                    non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                elif non_local_move == "dltmove":
                    mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                        compute_ecp_non_local_parts_nearest_neighbors_jax(
                            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                            wavefunction_data=hamiltonian_data.wavefunction_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                            flag_determinant_only=True,
                        )
                    )

                    V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                    diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))

                    Jastrow_ref = compute_Jastrow_part_jax(
                        jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    Jastrow_on_mesh = vmap(compute_Jastrow_part_jax, in_axes=(None, 0, 0))(
                        hamiltonian_data.wavefunction_data.jastrow_data,
                        mesh_non_local_ecp_part_r_up_carts,
                        mesh_non_local_ecp_part_r_dn_carts,
                    )
                    Jastrow_ratio = jnp.exp(Jastrow_on_mesh - Jastrow_ref)
                    V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

                    non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                else:
                    raise NotImplementedError

                V_diag = (
                    diagonal_kinetic_part
                    + discretized_diagonal_bare_coulomb_part
                    + diagonal_ecp_local_part
                    + diagonal_kinetic_part_SP
                    + diagonal_ecp_part_SP
                )

                V_nondiag = non_diagonal_sum_hamiltonian

            # with all electrons
            else:
                non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic

                V_diag = diagonal_kinetic_part + discretized_diagonal_bare_coulomb_part + diagonal_kinetic_part_SP

                V_nondiag = non_diagonal_sum_hamiltonian

            return (V_diag, V_nondiag)

        @partial(jit, static_argnums=4)
        def _compute_local_energy(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jax.Array,
            r_dn_carts: jax.Array,
            jax_PRNG_key: jax.Array,
            non_local_move: bool,
            alat: float,
        ):
            V_diag, V_nondiag = _compute_V_elements(
                hamiltonian_data, r_up_carts, r_dn_carts, jax_PRNG_key, non_local_move, alat
            )
            return V_diag + V_nondiag

        # projection compilation.
        logger.info("  Compilation is in progress...")
        w_L_list = jnp.array([1.0 for _ in range(self.__num_walkers)])
        (
            _,
            _,
            _,
            _,
        ) = vmap(_projection, in_axes=(0, 0, 0, 0, None, None, None, None, None))(
            w_L_list,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
            self.__jax_PRNG_key_list,
            self.__E_scf,
            self.__num_mcmc_per_measurement,
            self.__non_local_move,
            self.__alat,
            self.__hamiltonian_data,
        )

        _, _ = vmap(_compute_V_elements, in_axes=(None, 0, 0, 0, None, None))(
            self.__hamiltonian_data,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
            self.__jax_PRNG_key_list,
            self.__non_local_move,
            self.__alat,
        )

        if self.__comput_position_deriv:
            _, _, _ = vmap(grad(_compute_local_energy, argnums=(0, 1, 2)), in_axes=(None, 0, 0, 0, None, None))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                self.__jax_PRNG_key_list,
                self.__non_local_move,
                self.__alat,
            )
        end_init = time.perf_counter()
        timer_projection_init += end_init - start_init
        logger.info("End compilation of the GFMC projection funciton.")
        logger.info(f"Elapsed Time = {timer_projection_init:.2f} sec.")
        logger.info("")

        # MAIN MCMC loop from here !!!
        logger.info("Start GFMC")
        num_mcmc_done = 0
        progress = (self.__mcmc_counter) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
        gfmc_total_current = time.perf_counter()
        logger.info(
            f"  Progress: GFMC step = {self.__mcmc_counter}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.0f} %. Elapsed time = {(gfmc_total_current - gfmc_total_start):.1f} sec."
        )
        mcmc_interval = int(np.maximum(num_mcmc_steps / 100, 1))

        for i_mcmc_step in range(num_mcmc_steps):
            if (i_mcmc_step + 1) % mcmc_interval == 0:
                progress = (i_mcmc_step + self.__mcmc_counter + 1) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
                gfmc_total_current = time.perf_counter()
                logger.info(
                    f"  Progress: GFMC step = {i_mcmc_step + self.__mcmc_counter + 1}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %. Elapsed time = {(gfmc_total_current - gfmc_total_start):.1f} sec."
                )

            # Always set the initial weight list to 1.0
            w_L_list = jnp.array([1.0 for _ in range(self.__num_walkers)])

            logger.devel("  Projection is on going....")

            start_projection = time.perf_counter()

            # projection loop
            (
                w_L_list,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                self.__jax_PRNG_key_list,
            ) = vmap(_projection, in_axes=(0, 0, 0, 0, None, None, None, None, None))(
                w_L_list,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                self.__jax_PRNG_key_list,
                self.__E_scf,
                self.__num_mcmc_per_measurement,
                self.__non_local_move,
                self.__alat,
                self.__hamiltonian_data,
            )

            # sync. jax arrays computations.
            w_L_list.block_until_ready()
            self.__latest_r_up_carts.block_until_ready()
            self.__latest_r_dn_carts.block_until_ready()
            self.__jax_PRNG_key_list.block_until_ready()

            end_projection = time.perf_counter()
            timer_projection_total += end_projection - start_projection

            # projection ends
            logger.devel("  Projection ends.")

            # evaluate observables
            start = time.perf_counter()
            # V_diag and e_L
            V_diag_list, V_nondiag_list = vmap(_compute_V_elements, in_axes=(None, 0, 0, 0, None, None))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                self.__jax_PRNG_key_list,
                self.__non_local_move,
                self.__alat,
            )
            e_L_list = V_diag_list + V_nondiag_list
            # logger.info(f"  e_L_list = {e_L_list}")
            # logger.info(f"  V_diag_list = {V_diag_list}")
            # logger.info(f"  V_nondiag_list = {V_nondiag_list}")
            e_L_list.block_until_ready()

            """
            if self.__non_local_move == "tmove":
                e_list_debug = vmap(compute_local_energy_api, in_axes=(None, 0, 0))(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                if np.max(np.abs(e_L_list - e_list_debug)) > 1.0e-6:
                    logger.info(f"max(e_list - e_list_debug) = {np.max(np.abs(e_L_list - e_list_debug))}.")
                    logger.info(f"w_L_list = {w_L_list}.")
                    logger.info(f"e_L_list = {e_L_list}.")
                    logger.info(f"V_diag_list - E_scf = {V_diag_list - E_scf}.")
                    logger.info(f"e_list_debug = {e_list_debug}.")
                # np.testing.assert_almost_equal(np.array(e_L_list), np.array(e_list_debug), decimal=6)
            """

            end = time.perf_counter()
            timer_e_L += end - start

            # atomic force related
            if self.__comput_position_deriv:
                start = time.perf_counter()
                grad_e_L_h, grad_e_L_r_up, grad_e_L_r_dn = vmap(
                    grad(_compute_local_energy, argnums=(0, 1, 2)), in_axes=(None, 0, 0, 0, None, None)
                )(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__jax_PRNG_key_list,
                    self.__non_local_move,
                    self.__alat,
                )
                grad_e_L_r_up.block_until_ready()
                grad_e_L_r_dn.block_until_ready()
                end = time.perf_counter()
                timer_de_L_dR_dr += end - start

                grad_e_L_R = (
                    grad_e_L_h.wavefunction_data.geminal_data.orb_data_up_spin.structure_data.positions
                    + grad_e_L_h.wavefunction_data.geminal_data.orb_data_dn_spin.structure_data.positions
                    + grad_e_L_h.coulomb_potential_data.structure_data.positions
                )

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_one_body_data is not None:
                    grad_e_L_R += grad_e_L_h.wavefunction_data.jastrow_data.jastrow_one_body_data.structure_data.positions

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data is not None:
                    grad_e_L_R += (
                        grad_e_L_h.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data.structure_data.positions
                    )

                logger.devel(f"de_L_dR = {grad_e_L_R}")
                logger.devel(f"de_L_dr_up = {grad_e_L_r_up}")
                logger.devel(f"de_L_dr_dn= {grad_e_L_r_dn}")

                start = time.perf_counter()
                grad_ln_Psi_h, grad_ln_Psi_r_up, grad_ln_Psi_r_dn = vmap(
                    grad(evaluate_ln_wavefunction_jax, argnums=(0, 1, 2)), in_axes=(None, 0, 0)
                )(
                    self.__hamiltonian_data.wavefunction_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                grad_ln_Psi_r_up.block_until_ready()
                grad_ln_Psi_r_dn.block_until_ready()
                end = time.perf_counter()
                timer_dln_Psi_dR_dr += end - start

                grad_ln_Psi_dR = (
                    grad_ln_Psi_h.geminal_data.orb_data_up_spin.structure_data.positions
                    + grad_ln_Psi_h.geminal_data.orb_data_dn_spin.structure_data.positions
                )

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_one_body_data is not None:
                    grad_ln_Psi_dR += grad_ln_Psi_h.jastrow_data.jastrow_one_body_data.structure_data.positions

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data is not None:
                    grad_ln_Psi_dR += grad_ln_Psi_h.jastrow_data.jastrow_three_body_data.orb_data.structure_data.positions

                omega_up = vmap(evaluate_swct_omega_jax, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_up_carts,
                )

                omega_dn = vmap(evaluate_swct_omega_jax, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_dn_carts,
                )

                grad_omega_dr_up = vmap(evaluate_swct_domega_jax, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_up_carts,
                )

                grad_omega_dr_dn = vmap(evaluate_swct_domega_jax, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_dn_carts,
                )

                omega_up.block_until_ready()
                omega_dn.block_until_ready()
                grad_omega_dr_up.block_until_ready()
                grad_omega_dr_dn.block_until_ready()

            # Barrier before MPI operation
            # mpi_comm.Barrier()

            # Branching starts
            start_reconfiguration = time.perf_counter()

            # jnp.array -> np.array
            w_L_latest = np.array(w_L_list)
            e_L_latest = np.array(e_L_list)
            V_diag_E_latest = np.array(V_diag_list) - self.__E_scf

            if self.__comput_position_deriv:
                grad_e_L_r_up_latest = np.array(grad_e_L_r_up)
                grad_e_L_r_dn_latest = np.array(grad_e_L_r_dn)
                grad_e_L_R_latest = np.array(grad_e_L_R)
                grad_ln_Psi_r_up_latest = np.array(grad_ln_Psi_r_up)
                grad_ln_Psi_r_dn_latest = np.array(grad_ln_Psi_r_dn)
                grad_ln_Psi_dR_latest = np.array(grad_ln_Psi_dR)
                omega_up_latest = np.array(omega_up)
                omega_dn_latest = np.array(omega_dn)
                grad_omega_dr_up_latest = np.array(grad_omega_dr_up)
                grad_omega_dr_dn_latest = np.array(grad_omega_dr_dn)

            # jnp.array -> np.array
            self.__latest_r_up_carts = np.array(self.__latest_r_up_carts)
            self.__latest_r_dn_carts = np.array(self.__latest_r_dn_carts)

            # MPI reduce
            r_up_carts_shape = self.__latest_r_up_carts.shape
            r_up_carts_gathered_dyad = (mpi_rank, self.__latest_r_up_carts)
            r_dn_carts_shape = self.__latest_r_dn_carts.shape
            r_dn_carts_gathered_dyad = (mpi_rank, self.__latest_r_dn_carts)

            r_up_carts_gathered_dyad = mpi_comm.gather(r_up_carts_gathered_dyad, root=0)
            r_dn_carts_gathered_dyad = mpi_comm.gather(r_dn_carts_gathered_dyad, root=0)

            # MPI reduce
            e_L_gathered_dyad = (mpi_rank, e_L_latest)
            w_L_gathered_dyad = (mpi_rank, w_L_latest)
            V_diag_E_gathered_dyad = (mpi_rank, V_diag_E_latest)
            if self.__comput_position_deriv:
                grad_e_L_r_up_dyad = (mpi_rank, grad_e_L_r_up_latest)
                grad_e_L_r_dn_dyad = (mpi_rank, grad_e_L_r_dn_latest)
                grad_e_L_R_dyad = (mpi_rank, grad_e_L_R_latest)
                grad_ln_Psi_r_up_dyad = (mpi_rank, grad_ln_Psi_r_up_latest)
                grad_ln_Psi_r_dn_dyad = (mpi_rank, grad_ln_Psi_r_dn_latest)
                grad_ln_Psi_dR_dyad = (mpi_rank, grad_ln_Psi_dR_latest)
                omega_up_dyad = (mpi_rank, omega_up_latest)
                omega_dn_dyad = (mpi_rank, omega_dn_latest)
                grad_omega_dr_up_dyad = (mpi_rank, grad_omega_dr_up_latest)
                grad_omega_dr_dn_dyad = (mpi_rank, grad_omega_dr_dn_latest)

            e_L_gathered_dyad = mpi_comm.gather(e_L_gathered_dyad, root=0)
            w_L_gathered_dyad = mpi_comm.gather(w_L_gathered_dyad, root=0)
            V_diag_E_gathered_dyad = mpi_comm.gather(V_diag_E_gathered_dyad, root=0)
            if self.__comput_position_deriv:
                grad_e_L_r_up_dyad = mpi_comm.gather(grad_e_L_r_up_dyad, root=0)
                grad_e_L_r_dn_dyad = mpi_comm.gather(grad_e_L_r_dn_dyad, root=0)
                grad_e_L_R_dyad = mpi_comm.gather(grad_e_L_R_dyad, root=0)
                grad_ln_Psi_r_up_dyad = mpi_comm.gather(grad_ln_Psi_r_up_dyad, root=0)
                grad_ln_Psi_r_dn_dyad = mpi_comm.gather(grad_ln_Psi_r_dn_dyad, root=0)
                grad_ln_Psi_dR_dyad = mpi_comm.gather(grad_ln_Psi_dR_dyad, root=0)
                omega_up_dyad = mpi_comm.gather(omega_up_dyad, root=0)
                omega_dn_dyad = mpi_comm.gather(omega_dn_dyad, root=0)
                grad_omega_dr_up_dyad = mpi_comm.gather(grad_omega_dr_up_dyad, root=0)
                grad_omega_dr_dn_dyad = mpi_comm.gather(grad_omega_dr_dn_dyad, root=0)

            if mpi_rank == 0:
                # dict
                r_up_carts_gathered_dict = dict(r_up_carts_gathered_dyad)
                r_dn_carts_gathered_dict = dict(r_dn_carts_gathered_dyad)
                e_L_gathered_dict = dict(e_L_gathered_dyad)
                w_L_gathered_dict = dict(w_L_gathered_dyad)
                V_diag_E_gathered_dict = dict(V_diag_E_gathered_dyad)
                if self.__comput_position_deriv:
                    grad_e_L_r_up_gathered_dict = dict(grad_e_L_r_up_dyad)
                    grad_e_L_r_dn_gathered_dict = dict(grad_e_L_r_dn_dyad)
                    grad_e_L_R_gathered_dict = dict(grad_e_L_R_dyad)
                    grad_ln_Psi_r_up_gathered_dict = dict(grad_ln_Psi_r_up_dyad)
                    grad_ln_Psi_r_dn_gathered_dict = dict(grad_ln_Psi_r_dn_dyad)
                    grad_ln_Psi_dR_gathered_dict = dict(grad_ln_Psi_dR_dyad)
                    omega_up_gathered_dict = dict(omega_up_dyad)
                    omega_dn_gathered_dict = dict(omega_dn_dyad)
                    grad_omega_dr_up_gathered_dict = dict(grad_omega_dr_up_dyad)
                    grad_omega_dr_dn_gathered_dict = dict(grad_omega_dr_dn_dyad)
                # gathered
                r_up_carts_gathered = np.concatenate([r_up_carts_gathered_dict[i] for i in range(mpi_size)])
                r_dn_carts_gathered = np.concatenate([r_dn_carts_gathered_dict[i] for i in range(mpi_size)])
                e_L_gathered = np.concatenate([e_L_gathered_dict[i] for i in range(mpi_size)])
                w_L_gathered = np.concatenate([w_L_gathered_dict[i] for i in range(mpi_size)])
                V_diag_E_gathered = np.concatenate([V_diag_E_gathered_dict[i] for i in range(mpi_size)])
                if self.__comput_position_deriv:
                    grad_e_L_r_up_gathered = np.concatenate([grad_e_L_r_up_gathered_dict[i] for i in range(mpi_size)])
                    grad_e_L_r_dn_gathered = np.concatenate([grad_e_L_r_dn_gathered_dict[i] for i in range(mpi_size)])
                    grad_e_L_R_gathered = np.concatenate([grad_e_L_R_gathered_dict[i] for i in range(mpi_size)])
                    grad_ln_Psi_r_up_gathered = np.concatenate([grad_ln_Psi_r_up_gathered_dict[i] for i in range(mpi_size)])
                    grad_ln_Psi_r_dn_gathered = np.concatenate([grad_ln_Psi_r_dn_gathered_dict[i] for i in range(mpi_size)])
                    grad_ln_Psi_dR_gathered = np.concatenate([grad_ln_Psi_dR_gathered_dict[i] for i in range(mpi_size)])
                    omega_up_gathered = np.concatenate([omega_up_gathered_dict[i] for i in range(mpi_size)])
                    omega_dn_gathered = np.concatenate([omega_dn_gathered_dict[i] for i in range(mpi_size)])
                    grad_omega_dr_up_gathered = np.concatenate([grad_omega_dr_up_gathered_dict[i] for i in range(mpi_size)])
                    grad_omega_dr_dn_gathered = np.concatenate([grad_omega_dr_dn_gathered_dict[i] for i in range(mpi_size)])
                # sum
                w_L_sum = np.sum(w_L_gathered / V_diag_E_gathered)
                e_L_sum = np.sum(w_L_gathered / V_diag_E_gathered * e_L_gathered)
                e_L2_sum = np.sum(w_L_gathered / V_diag_E_gathered * e_L_gathered**2)
                if self.__comput_position_deriv:
                    grad_e_L_r_up_sum = np.einsum("i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_e_L_r_up_gathered)
                    grad_e_L_r_dn_sum = np.einsum("i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_e_L_r_dn_gathered)
                    grad_e_L_R_sum = np.einsum("i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_e_L_R_gathered)
                    grad_ln_Psi_r_up_sum = np.einsum("i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_ln_Psi_r_up_gathered)
                    grad_ln_Psi_r_dn_sum = np.einsum("i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_ln_Psi_r_dn_gathered)
                    grad_ln_Psi_dR_sum = np.einsum("i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_ln_Psi_dR_gathered)
                    omega_up_sum = np.einsum("i,ijk->jk", w_L_gathered / V_diag_E_gathered, omega_up_gathered)
                    omega_dn_sum = np.einsum("i,ijk->jk", w_L_gathered / V_diag_E_gathered, omega_dn_gathered)
                    grad_omega_dr_up_sum = np.einsum("i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_omega_dr_up_gathered)
                    grad_omega_dr_dn_sum = np.einsum("i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_omega_dr_dn_gathered)
                # averaged
                w_L_averaged = np.average(w_L_gathered / V_diag_E_gathered)
                e_L_averaged = e_L_sum / w_L_sum
                e_L2_averaged = e_L2_sum / w_L_sum

                """ wrong maybe
                w_L_sum = np.sum(w_L_gathered)
                e_L_sum = np.sum(w_L_gathered * e_L_gathered)
                if self.__comput_position_deriv:
                    grad_e_L_r_up_sum = np.einsum("i,ijk->jk", w_L_gathered, grad_e_L_r_up_gathered)
                    grad_e_L_r_dn_sum = np.einsum("i,ijk->jk", w_L_gathered, grad_e_L_r_dn_gathered)
                    grad_e_L_R_sum = np.einsum("i,ijk->jk", w_L_gathered, grad_e_L_R_gathered)
                    grad_ln_Psi_r_up_sum = np.einsum("i,ijk->jk", w_L_gathered, grad_ln_Psi_r_up_gathered)
                    grad_ln_Psi_r_dn_sum = np.einsum("i,ijk->jk", w_L_gathered, grad_ln_Psi_r_dn_gathered)
                    grad_ln_Psi_dR_sum = np.einsum("i,ijk->jk", w_L_gathered, grad_ln_Psi_dR_gathered)
                    omega_up_sum = np.einsum("i,ijk->jk", w_L_gathered, omega_up_gathered)
                    omega_dn_sum = np.einsum("i,ijk->jk", w_L_gathered, omega_dn_gathered)
                    grad_omega_dr_up_sum = np.einsum("i,ijk->jk", w_L_gathered, grad_omega_dr_up_gathered)
                    grad_omega_dr_dn_sum = np.einsum("i,ijk->jk", w_L_gathered, grad_omega_dr_dn_gathered)
                # averaged
                w_L_averaged = np.average(w_L_gathered) * np.average(1.0 / V_diag_E_gathered)
                e_L_averaged = e_L_sum / w_L_sum
                """
                if self.__comput_position_deriv:
                    grad_e_L_r_up_averaged = grad_e_L_r_up_sum / w_L_sum
                    grad_e_L_r_dn_averaged = grad_e_L_r_dn_sum / w_L_sum
                    grad_e_L_R_averaged = grad_e_L_R_sum / w_L_sum
                    grad_ln_Psi_r_up_averaged = grad_ln_Psi_r_up_sum / w_L_sum
                    grad_ln_Psi_r_dn_averaged = grad_ln_Psi_r_dn_sum / w_L_sum
                    grad_ln_Psi_dR_averaged = grad_ln_Psi_dR_sum / w_L_sum
                    omega_up_averaged = omega_up_sum / w_L_sum
                    omega_dn_averaged = omega_dn_sum / w_L_sum
                    grad_omega_dr_up_averaged = grad_omega_dr_up_sum / w_L_sum
                    grad_omega_dr_dn_averaged = grad_omega_dr_dn_sum / w_L_sum
                # add a dummy dim
                e_L2_averaged = np.expand_dims(e_L2_averaged, axis=0)
                e_L_averaged = np.expand_dims(e_L_averaged, axis=0)
                w_L_averaged = np.expand_dims(w_L_averaged, axis=0)
                if self.__comput_position_deriv:
                    grad_e_L_r_up_averaged = np.expand_dims(grad_e_L_r_up_averaged, axis=0)
                    grad_e_L_r_dn_averaged = np.expand_dims(grad_e_L_r_dn_averaged, axis=0)
                    grad_e_L_R_averaged = np.expand_dims(grad_e_L_R_averaged, axis=0)
                    grad_ln_Psi_r_up_averaged = np.expand_dims(grad_ln_Psi_r_up_averaged, axis=0)
                    grad_ln_Psi_r_dn_averaged = np.expand_dims(grad_ln_Psi_r_dn_averaged, axis=0)
                    grad_ln_Psi_dR_averaged = np.expand_dims(grad_ln_Psi_dR_averaged, axis=0)
                    omega_up_averaged = np.expand_dims(omega_up_averaged, axis=0)
                    omega_dn_averaged = np.expand_dims(omega_dn_averaged, axis=0)
                    grad_omega_dr_up_averaged = np.expand_dims(grad_omega_dr_up_averaged, axis=0)
                    grad_omega_dr_dn_averaged = np.expand_dims(grad_omega_dr_dn_averaged, axis=0)
                # store  # This should stored only for MPI-rank = 0 !!!
                self.__stored_e_L2.append(e_L2_averaged)
                self.__stored_e_L.append(e_L_averaged)
                self.__stored_w_L.append(w_L_averaged)
                if self.__comput_position_deriv:
                    self.__stored_grad_e_L_r_up.append(grad_e_L_r_up_averaged)
                    self.__stored_grad_e_L_r_dn.append(grad_e_L_r_dn_averaged)
                    self.__stored_grad_e_L_dR.append(grad_e_L_R_averaged)
                    self.__stored_grad_ln_Psi_r_up.append(grad_ln_Psi_r_up_averaged)
                    self.__stored_grad_ln_Psi_r_dn.append(grad_ln_Psi_r_dn_averaged)
                    self.__stored_grad_ln_Psi_dR.append(grad_ln_Psi_dR_averaged)
                    self.__stored_omega_up.append(omega_up_averaged)
                    self.__stored_omega_dn.append(omega_dn_averaged)
                    self.__stored_grad_omega_r_up.append(grad_omega_dr_up_averaged)
                    self.__stored_grad_omega_r_dn.append(grad_omega_dr_dn_averaged)
                # for branching
                w_L_list = w_L_gathered
                logger.devel(f"w_L_list = {w_L_list}")
                probabilities = w_L_list / w_L_list.sum()
                logger.devel(f"probabilities = {probabilities}")

                # correlated choice (see Sandro's textbook, page 182)
                """ very slow w/o jax-jit!!
                self.__jax_PRNG_key, subkey = jax.random.split(self.__jax_PRNG_key)
                zeta = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
                """
                zeta = float(np.random.random())
                z_list = [(alpha + zeta) / len(probabilities) for alpha in range(len(probabilities))]
                logger.devel(f"z_list = {z_list}")
                cumulative_prob = np.cumsum(probabilities)
                chosen_walker_indices = np.array(
                    [next(idx for idx, prob in enumerate(cumulative_prob) if z <= prob) for z in z_list]
                )
                logger.devel(f"The chosen walker indices = {chosen_walker_indices}")
                logger.devel(f"The chosen walker indices.shape = {chosen_walker_indices.shape}")
                logger.devel(f"r_up_carts_gathered.shape = {r_up_carts_gathered.shape}")
                logger.devel(f"r_dn_carts_gathered.shape = {r_dn_carts_gathered.shape}")

                proposed_r_up_carts = r_up_carts_gathered[chosen_walker_indices]
                proposed_r_dn_carts = r_dn_carts_gathered[chosen_walker_indices]

                self.__num_survived_walkers += len(set(chosen_walker_indices))
                self.__num_killed_walkers += len(w_L_list) - len(set(chosen_walker_indices))
                logger.devel(f"num_survived_walkers={self.__num_survived_walkers}")
                logger.devel(f"num_killed_walkers={self.__num_killed_walkers}")
            else:
                self.__num_survived_walkers = None
                self.__num_killed_walkers = None
                proposed_r_up_carts = None
                proposed_r_dn_carts = None

            logger.devel(f"Before branching: rank={mpi_rank}:gfmc.r_up_carts = {self.__latest_r_up_carts}")
            logger.devel(f"Before branching: rank={mpi_rank}:gfmc.r_dn_carts = {self.__latest_r_dn_carts}")

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

            logger.devel(f"*After branching: rank={mpi_rank}:gfmc.r_up_carts = {self.__latest_r_up_carts}")
            logger.devel(f"*After branching: rank={mpi_rank}:gfmc.r_dn_carts = {self.__latest_r_dn_carts}")

            end_reconfiguration = time.perf_counter()
            timer_reconfiguration += end_reconfiguration - start_reconfiguration

            # update E_scf
            eq_steps = 20
            if (i_mcmc_step + 1) % mcmc_interval == 0:
                if i_mcmc_step >= eq_steps:
                    self.__E_scf, E_scf_std = self.get_E_on_the_fly(
                        num_gfmc_warmup_steps=np.minimum(eq_steps, i_mcmc_step - eq_steps),
                        num_gfmc_bin_blocks=10,
                        num_gfmc_collect_steps=10,
                    )
                    logger.debug(f"    Updated E_scf = {self.__E_scf:.5f} +- {E_scf_std:.5f} Ha.")
                else:
                    logger.debug(f"    Init E_scf = {self.__E_scf:.5f} Ha. Being equilibrated.")

            num_mcmc_done += 1
            gfmc_current = time.perf_counter()
            if max_time < gfmc_current - gfmc_total_start:
                logger.info(f"  Max_time = {max_time} sec. exceeds.")
                logger.info("  Break the branching loop.")
                break

        logger.info("-End branching-")
        logger.info("")

        # count up mcmc_counter
        self.__mcmc_counter += num_mcmc_done

        gfmc_total_end = time.perf_counter()
        timer_gfmc_total = gfmc_total_end - gfmc_total_start
        timer_misc = timer_gfmc_total - (
            timer_projection_init
            + timer_projection_total
            + timer_e_L
            + timer_de_L_dR_dr
            + timer_dln_Psi_dR_dr
            + timer_dln_Psi_dc
            + timer_de_L_dc
            + timer_reconfiguration
        )

        logger.info(f"Total GFMC time for {num_mcmc_done} branching steps = {timer_gfmc_total: .3f} sec.")
        logger.info(f"Pre-compilation time for GFMC = {timer_projection_init: .3f} sec.")
        logger.info(f"Net GFMC time without pre-compilations = {timer_gfmc_total - timer_projection_init: .3f} sec.")
        logger.info(f"Elapsed times per branching, averaged over {num_mcmc_done} branching steps.")
        logger.info(f"  Projection between branching = {timer_projection_total / num_mcmc_done * 10**3: .3f} msec.")
        logger.info(f"  Time for computing e_L = {timer_e_L / num_mcmc_done * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/dR and de_L/dr = {timer_de_L_dR_dr / num_mcmc_done * 10**3:.2f} msec.")
        logger.info(f"  Time for computing dln_Psi/dR and dln_Psi/dr = {timer_dln_Psi_dR_dr / num_mcmc_done * 10**3:.2f} msec.")
        logger.info(f"  Time for computing dln_Psi/dc = {timer_dln_Psi_dc / num_mcmc_done * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/dc = {timer_de_L_dc / num_mcmc_done * 10**3:.2f} msec.")
        logger.info(f"  Time for misc. (others) = {timer_misc / num_mcmc_done * 10**3:.2f} msec.")
        logger.info(f"  Walker reconfiguration time per branching = {timer_reconfiguration / num_mcmc_done * 10**3: .3f} msec.")
        logger.devel(f"Survived walkers = {self.__num_survived_walkers}")
        logger.devel(f"killed walkers = {self.__num_killed_walkers}")
        logger.info(
            f"Survived walkers ratio = {self.__num_survived_walkers / (self.__num_survived_walkers + self.__num_killed_walkers) * 100:.2f} %. Ideal is ~ 98 %. Adjust num_mcmc_per_measurement."
        )
        logger.info("")

        self.__timer_gfmc_total += timer_gfmc_total
        self.__timer_projection_init += timer_projection_init
        self.__timer_projection_total += timer_projection_total
        self.__timer_branching += timer_reconfiguration
        self.__timer_e_L += timer_e_L
        self.__timer_de_L_dR_dr += timer_de_L_dR_dr
        self.__timer_dln_Psi_dR_dr += timer_dln_Psi_dR_dr
        self.__timer_dln_Psi_dc += timer_dln_Psi_dc
        self.__timer_de_L_dc += timer_de_L_dc

    def get_E_on_the_fly(
        self, num_gfmc_warmup_steps: int = 3, num_gfmc_bin_blocks: int = 10, num_gfmc_collect_steps: int = 2
    ) -> float:
        """Get e_L."""
        logger.devel("- Comput. e_L -")
        if mpi_rank == 0:
            e_L_eq = self.__stored_e_L[num_gfmc_warmup_steps + num_gfmc_collect_steps :]
            w_L_eq = self.__stored_w_L[num_gfmc_warmup_steps:]
            # logger.info(f" AS (e_L_eq) = {(e_L_eq)}")
            # logger.info(f"  (w_L_eq) = {(w_L_eq)}")
            logger.devel("  Progress: Computing G_eq and G_e_L_eq.")

            w_L_eq = jnp.array(w_L_eq)
            e_L_eq = jnp.array(e_L_eq)
            G_eq = compute_G_L(w_L_eq, num_gfmc_collect_steps)
            G_e_L_eq = e_L_eq * G_eq
            G_eq = np.array(G_eq)
            G_e_L_eq = np.array(G_e_L_eq)

            logger.devel(f"  Progress: Computing binned G_e_L_eq and G_eq with # binned blocks = {num_gfmc_bin_blocks}.")
            G_e_L_split = np.array_split(G_e_L_eq, num_gfmc_bin_blocks)
            G_e_L_binned = np.array([np.sum(G_e_L_list) for G_e_L_list in G_e_L_split])
            G_split = np.array_split(G_eq, num_gfmc_bin_blocks)
            G_binned = np.array([np.sum(G_list) for G_list in G_split])

            logger.devel(f"  Progress: Computing jackknife samples with # binned blocks = {num_gfmc_bin_blocks}.")

            G_e_L_binned_sum = np.sum(G_e_L_binned)
            G_binned_sum = np.sum(G_binned)

            E_jackknife = [
                (G_e_L_binned_sum - G_e_L_binned[m]) / (G_binned_sum - G_binned[m]) for m in range(num_gfmc_bin_blocks)
            ]

            logger.devel("  Progress: Computing jackknife mean and std.")
            E_mean = np.average(E_jackknife)
            E_std = np.sqrt(num_gfmc_bin_blocks - 1) * np.std(E_jackknife)
            E_mean = float(E_mean)
            E_std = float(E_std)
        else:
            E_mean = None
            E_std = None

        E_mean = mpi_comm.bcast(E_mean, root=0)
        E_std = mpi_comm.bcast(E_std, root=0)

        return E_mean, E_std


class QMC:
    """QMC class. QMC using MCMC or GFMC.

    Args:
        mcmc (MCMC | GFMC): an instance of MCMC or GFMC.
    """

    def __init__(self, mcmc: MCMC = None) -> None:
        """Initialization."""
        self.mcmc = mcmc
        self.__i_opt = 0

    def run(self, num_mcmc_steps: int = 0, max_time: int = 86400) -> None:
        """Launch single-shot VMC.

        Args:
            num_mcmc_steps(int):
                The number of MCMC samples per walker.
            max_time(int):
                The maximum time (sec.) If maximum time exceeds,
                the method exits the MCMC loop.
        """
        self.mcmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)

    def run_optimize(
        self,
        num_mcmc_steps: int = 100,
        num_opt_steps: int = 1,
        delta: float = 0.001,
        epsilon: float = 1.0e-3,
        wf_dump_freq: int = 10,
        max_time: int = 86400,
        num_mcmc_warmup_steps: int = 0,
        num_mcmc_bin_blocks: int = 100,
        opt_J1_param: bool = True,
        opt_J2_param: bool = True,
        opt_J3_param: bool = True,
        opt_lambda_param: bool = False,
        num_param_opt: int = 0,
    ):
        """Optimizing wavefunction.

        Optimizing Wavefunction using the Stochastic Reconfiguration Method.

        Args:
            num_mcmc_steps(int): The number of MCMC samples per walker.
            num_opt_steps(int): The number of WF optimization step.
            delta(float):
                The prefactor of the SR matrix for adjusting the optimization step.
                i.e., c_i <- c_i + delta * S^{-1} f
            epsilon(float):
                The regralization factor of the SR matrix
                i.e., S <- S + I * delta
            wf_dump_freq(int):
                The frequency of WF data (i.e., hamiltonian_data.chk)
            max_time(int):
                The maximum time (sec.) If maximum time exceeds,
                the method exits the MCMC loop.
            num_mcmc_warmup_steps (int): number of equilibration steps.
            num_mcmc_bin_blocks (int): number of blocks for reblocking.
            opt_J1_param (bool): optimize one-body Jastrow # to be implemented.
            opt_J2_param (bool): optimize two-body Jastrow
            opt_J3_param (bool): optimize three-body Jastrow
            opt_lambda_param (bool): optimize lambda_matrix in the determinant part.
            num_param_opt (int): the number of parameters to optimize in the descending order of |f|/|std f|. If zero, all parameters are optimized.
        """
        vmcopt_total_start = time.perf_counter()

        # main vmcopt loop
        for i_opt in range(num_opt_steps):
            logger.info("=" * num_sep_line)
            logger.info(f"Optimization step = {i_opt + 1 + self.__i_opt}/{num_opt_steps + self.__i_opt}.")
            logger.info("=" * num_sep_line)

            logger.info(f"MCMC steps this iteration = {num_mcmc_steps}.")
            logger.info(f"Warmup steps = {num_mcmc_warmup_steps}.")
            logger.info(f"Bin blocks = {num_mcmc_bin_blocks}.")
            logger.info("")

            # run MCMC
            self.mcmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)

            # get E
            E, E_std, _, _ = self.get_E(num_mcmc_warmup_steps=num_mcmc_warmup_steps, num_mcmc_bin_blocks=num_mcmc_bin_blocks)
            logger.info("Total Energy before update of wavefunction.")
            logger.info("-" * num_sep_line)
            logger.info(f"E = {E:.5f} +- {E_std:.5f} Ha")
            logger.info("-" * num_sep_line)
            logger.info("")

            # get opt param
            dc_param_list = self.mcmc.opt_param_dict["dc_param_list"]
            dc_flattened_index_list = self.mcmc.opt_param_dict["dc_flattened_index_list"]
            # Indices of variational parameters
            ## chosen_param_index
            ## index of optimized parameters in the dln_wf_dc.
            chosen_param_index = []
            ## opt_param_index_dict
            ## index in the vector theta (i.e., natural gradient) for the chosen opt parameters.
            ## This is used when updating the parameters.
            opt_param_index_dict = {}

            for ii, dc_param in enumerate(dc_param_list):
                if opt_J1_param and dc_param == "j1_param":
                    new_param_index = [i for i, v in enumerate(dc_flattened_index_list) if v == ii]
                    opt_param_index_dict[dc_param] = np.array(range(len(new_param_index)), dtype=np.int32) + len(
                        chosen_param_index
                    )
                    chosen_param_index += new_param_index
                if opt_J2_param and dc_param == "j2_param":
                    logger.devel(
                        f"  twobody param before opt. = {self.mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param}"
                    )
                    new_param_index = [i for i, v in enumerate(dc_flattened_index_list) if v == ii]
                    opt_param_index_dict[dc_param] = np.array(range(len(new_param_index)), dtype=np.int32) + len(
                        chosen_param_index
                    )
                    chosen_param_index += new_param_index
                if opt_J3_param and dc_param == "j3_matrix":
                    new_param_index = [i for i, v in enumerate(dc_flattened_index_list) if v == ii]
                    opt_param_index_dict[dc_param] = np.array(range(len(new_param_index)), dtype=np.int32) + len(
                        chosen_param_index
                    )
                    chosen_param_index += new_param_index
                if opt_lambda_param and dc_param == "lambda_matrix":
                    new_param_index = [i for i, v in enumerate(dc_flattened_index_list) if v == ii]
                    opt_param_index_dict[dc_param] = np.array(range(len(new_param_index)), dtype=np.int32) + len(
                        chosen_param_index
                    )
                    chosen_param_index += new_param_index
            chosen_param_index = np.array(chosen_param_index)

            logger.info(f"Number of variational parameters = {len(chosen_param_index)}.")

            # get f and f_std (generalized forces)
            f, f_std = self.get_gF(
                mpi_broadcast=False,
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
                chosen_param_index=chosen_param_index,
            )

            if mpi_rank == 0:
                logger.debug(f"shape of f = {f.shape}.")
                logger.devel(f"f_std.shape = {f_std.shape}.")
                signal_to_noise_f = np.abs(f) / f_std
                f_argmax = np.argmax(np.abs(f))
                logger.info("-" * num_sep_line)
                logger.info(f"Max f = {f[f_argmax]:.3f} +- {f_std[f_argmax]:.3f} Ha/a.u.")
                logger.info(f"Max of signal-to-noise of f = max(|f|/|std f|) = {np.max(signal_to_noise_f):.3f}.")
                logger.info("-" * num_sep_line)
                if num_param_opt != 0:
                    if num_param_opt > len(signal_to_noise_f):
                        num_param_opt = len(signal_to_noise_f)
                    logger.info(
                        f"Optimizing only {num_param_opt} variational parameters with the largest signal to noise ratios of f."
                    )
                    signal_to_noise_f_max_indices = np.argsort(signal_to_noise_f)[::-1][:num_param_opt]
                else:
                    logger.info("Optimizing all variational parameters.")
                    signal_to_noise_f_max_indices = np.arange(signal_to_noise_f.size)
            else:
                signal_to_noise_f = None
                signal_to_noise_f_max_indices = None

            mpi_comm.bcast(signal_to_noise_f, root=0)
            mpi_comm.bcast(signal_to_noise_f_max_indices, root=0)

            # """
            logger.info("Computing the natural gradient, i.e., {S+epsilon*I}^{-1}*f")

            if self.mcmc.e_L.size != 0:
                w_L = self.mcmc.w_L[num_mcmc_warmup_steps:]
                # logger.info(f"w_L  = {w_L}.")
                # logger.info(f"w_L.shape = {w_L.shape}.")
                w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                w_L_binned = list(np.ravel([np.mean(arr, axis=0) for arr in w_L_split]))

                e_L = self.mcmc.e_L[num_mcmc_warmup_steps:]
                # logger.info(f"e_L.shape = {e_L.shape}.")
                e_L_split = np.array_split(e_L, num_mcmc_bin_blocks, axis=0)
                e_L_binned = list(np.ravel([np.mean(arr, axis=0) for arr in e_L_split]))

                w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
                w_L_e_L_binned = list(np.ravel([np.mean(w_L_e_L, axis=0) for w_L_e_L in zip(w_L_e_L_split)]))
                # logger.info(f"w_L_e_L_binned.shape = {np.array(w_L_e_L_binned).shape}.")

                O_matrix = self.get_dln_WF(num_mcmc_warmup_steps=num_mcmc_warmup_steps, chosen_param_index=chosen_param_index)
                # logger.info(f"O_matrix.shape = {O_matrix.shape}.")
                O_matrix_split = np.array_split(O_matrix, num_mcmc_bin_blocks, axis=0)
                O_matrix_ave = np.array([np.mean(O_matrix, axis=0) for O_matrix in O_matrix_split])
                O_matrix_binned_shape = (
                    O_matrix_ave.shape[0] * O_matrix_ave.shape[1],
                    O_matrix_ave.shape[2],
                )
                O_matrix_binned = list(O_matrix_ave.reshape(O_matrix_binned_shape))

                w_L_O_matrix = np.einsum("iw,iwj->iwj", w_L, O_matrix)
                # logger.info(f"w_L_O_matrix.shape = {w_L_O_matrix.shape}.")
                w_L_O_matrix_split = np.array_split(w_L_O_matrix, num_mcmc_bin_blocks, axis=0)
                w_L_O_matrix_ave = np.array(
                    [
                        np.sum(w_L_O_matrix, axis=0) / np.sum(w_L, axis=0)[:, np.newaxis]
                        for w_L_O_matrix, w_L in zip(w_L_O_matrix_split, w_L_split)
                    ]
                )
                w_L_O_matrix_binned_shape = (
                    w_L_O_matrix_ave.shape[0] * w_L_O_matrix_ave.shape[1],
                    w_L_O_matrix_ave.shape[2],
                )
                w_L_O_matrix_binned = list(w_L_O_matrix_ave.reshape(w_L_O_matrix_binned_shape))

                # logger.info(f"O_matrix.shape = {O_matrix.shape}")
                # logger.info(f"w_L_O_matrix_ave.shape = {w_L_O_matrix_ave.shape}")
                # logger.info(f"w_L_O_matrix_binned.shape = {np.array(w_L_O_matrix_binned).shape}")

            else:
                w_L_binned = []
                e_L_binned = []
                O_matrix_binned = []
                w_L_e_L_binned = []
                w_L_O_matrix_binned = []

            w_L_binned = mpi_comm.reduce(w_L_binned, op=MPI.SUM, root=0)
            e_L_binned = mpi_comm.reduce(e_L_binned, op=MPI.SUM, root=0)
            O_matrix_binned = mpi_comm.reduce(O_matrix_binned, op=MPI.SUM, root=0)
            w_L_e_L_binned = mpi_comm.reduce(w_L_e_L_binned, op=MPI.SUM, root=0)
            w_L_O_matrix_binned = mpi_comm.reduce(w_L_O_matrix_binned, op=MPI.SUM, root=0)

            if mpi_rank == 0:
                w_L_binned = np.array(w_L_binned)
                e_L_binned = np.array(e_L_binned)
                O_matrix_binned = np.array(O_matrix_binned)
                w_L_e_L_binned = np.array(w_L_e_L_binned)
                w_L_O_matrix_binned = np.array(w_L_O_matrix_binned)

                # compute weighted averages
                O_bar = np.sum(w_L_O_matrix_binned, axis=0) / np.sum(w_L_binned)
                e_L_bar = np.sum(w_L_e_L_binned) / np.sum(w_L_binned)

                # compute the following variables
                #     X_{i,k}: \equiv (O_{i, k} - \bar{O}_{k}),
                #     X_w_{i,k} \equiv w_i O_{i, k} / {\sum_{i} w_i}
                #     F_i \equiv -2.0 * (e_L_{i} - E)
                # logger.info(f"w_L_binned={w_L_binned}")
                X = (O_matrix_binned - O_bar).T
                X_w = w_L_binned * ((w_L_O_matrix_binned - O_bar) / np.sum(w_L_binned)).T
                F = -2.0 * (e_L_binned - e_L_bar).T

                X_w_F = X_w @ F
                f_argmax = np.argmax(np.abs(X_w_F))
                logger.debug(f"Max dot(X_w, F) \equiv f = {X_w_F[f_argmax]:.3f} Ha/a.u.")

                """
                logger.info(f"w_L_binned.shape = {w_L_binned.shape}.")
                logger.info(f"w_L_binned.T.shape = {w_L_binned.T.shape}.")
                logger.info(f"w_L_O_matrix_binned.shape = {w_L_O_matrix_binned.shape}.")
                logger.info(f"O_bar.shape = {O_bar.shape}")
                logger.info(f"X_w.shape = {X_w.shape}.")
                logger.info(f"X.shape = {X.shape}.")
                logger.info(f"F.shape = {F.shape}.")
                logger.info(f"w_L_binned = {w_L_binned}.")
                logger.info(f"X_w = {X_w}.")
                logger.info(f"X = {X}.")
                logger.info(f"F = {F}.")
                logger.info(f"X_w/X = {X_w / X}.")
                """

                # make the SR matrix scale-invariant (i.e., normalize)
                S = X_w @ X.T
                diag_S = np.diag(S)
                logger.debug(f"max. and min. diag_S = {np.max(diag_S)}, {np.min(diag_S)}.")
                X = X / np.sqrt(diag_S)[:, np.newaxis]
                X_w = X_w / np.sqrt(diag_S)[:, np.newaxis]

                if X.shape[0] < X.shape[1]:
                    logger.debug("X is a wide matrix. Proceed w/o the push-through identity.")
                    logger.debug("theta = (S+epsilon*I)^{-1}*f = (X * X^T + epsilon*I)^{-1} * X F...")
                    X_w_X_T = X_w @ X.T
                    logger.debug(f"X_w @ X.T.shape = {X_w_X_T.shape}.")
                    X_w_X_T[np.diag_indices_from(X_w_X_T)] += epsilon
                    # (X_w X^T + eps*I) x = X_w F ->solve-> x = (X_w  X^T + eps*I)^{-1} X_w F
                    X_w_X_T_inv_X_w_F = scipy.linalg.solve(X_w_X_T, X_w @ F, assume_a="sym")
                    # theta = (X_w X^T + eps*I)^{-1} X_w F
                    theta_all = X_w_X_T_inv_X_w_F
                    logger.devel(f"theta_all = {theta_all}.")
                else:
                    logger.debug("X is a tall matrix. Proceed w/ the push-through identity.")
                    logger.debug("theta = (S+epsilon*I)^{-1}*f = X(X^T * X + epsilon*I)^{-1} * F...")
                    X_T_X_w = X.T @ X_w
                    logger.debug(f"X_T_X_w.shape = {X_T_X_w.shape}.")
                    X_T_X_w[np.diag_indices_from(X_T_X_w)] += epsilon
                    # (X^T X_w + eps*I) x = F ->solve-> x = (X^T X_w + eps*I)^{-1} F
                    X_T_X_w_inv_F = scipy.linalg.solve(X_T_X_w, F, assume_a="sym")
                    # theta = X_w (X^T X_w + eps*I)^{-1} F
                    theta_all = X_w @ X_T_X_w_inv_F
                    logger.devel(f"theta_all = {theta_all}.")

                # theta, back to the original scale
                theta_all = theta_all / np.sqrt(diag_S)

                # Extract only the signal-to-noise ratio maximized parameters
                theta = np.zeros_like(theta_all)
                theta[signal_to_noise_f_max_indices] = theta_all[signal_to_noise_f_max_indices]

            else:
                theta = None

            # broadcast theta
            theta = mpi_comm.bcast(theta, root=0)

            # logger.devel(f"XX for MPI-rank={mpi_rank} is {theta}")
            # logger.devel(f"XX.shape for MPI-rank={mpi_rank} is {theta.shape}")
            logger.debug(f"theta.size = {theta.size}.")
            logger.debug(f"np.count_nonzero(theta) = {np.count_nonzero(theta)}.")
            logger.debug(f"max. and min. of theta are {np.max(theta)} and {np.min(theta)}.")

            dc_param_list = self.mcmc.opt_param_dict["dc_param_list"]
            dc_shape_list = self.mcmc.opt_param_dict["dc_shape_list"]
            dc_flattened_index_list = self.mcmc.opt_param_dict["dc_flattened_index_list"]

            # optimized parameters
            if self.mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_one_body_data is not None:
                j1_param = self.mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_one_body_data.jastrow_1b_param
            if self.mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data is not None:
                j2_param = self.mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
            if self.mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data is not None:
                j3_matrix = self.mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix
            if self.mcmc.hamiltonian_data.wavefunction_data.geminal_data is not None:
                lambda_matrix = self.mcmc.hamiltonian_data.wavefunction_data.geminal_data.lambda_matrix

            logger.devel(f"dX.shape for MPI-rank={mpi_rank} is {theta.shape}")

            for ii, dc_param in enumerate(dc_param_list):
                dc_shape = dc_shape_list[ii]
                if theta.shape == (1,):
                    dX = theta[0]
                if opt_J1_param and dc_param == "j1_param":
                    logger.info("Update J1 parameters.")
                    dX = theta[opt_param_index_dict[dc_param]].reshape(dc_shape)
                    j1_param += delta * dX
                if opt_J2_param and dc_param == "j2_param":
                    logger.info("Update J2 parameters.")
                    dX = theta[opt_param_index_dict[dc_param]].reshape(dc_shape)
                    j2_param += delta * dX
                if opt_J3_param and dc_param == "j3_matrix":
                    logger.info("Update J3 parameters.")
                    dX = theta[opt_param_index_dict[dc_param]].reshape(dc_shape)
                    # j1 part (rectanglar)
                    j3_matrix[:, -1] += delta * dX[:, -1]
                    # j3 part (square)
                    if np.allclose(j3_matrix[:, :-1], j3_matrix[:, :-1].T, atol=1e-8):
                        logger.info("The j3 matrix is symmetric. Keep it while updating.")
                        dX = 1.0 / 2.0 * (dX[:, :-1] + dX[:, :-1].T)
                    else:
                        dX = dX[:, :-1]
                    j3_matrix[:, :-1] += delta * dX
                    """To be implemented. Opt only the block diagonal parts, i.e. only the J3 part."""
                if opt_lambda_param and dc_param == "lambda_matrix":
                    logger.info("Updadate lambda matrix.")
                    dX = theta[opt_param_index_dict[dc_param]].reshape(dc_shape)
                    if np.allclose(lambda_matrix, lambda_matrix.T, atol=1e-8):
                        logger.info("The lambda matrix is symmetric. Keep it while updating.")
                        dX = 1.0 / 2.0 * (dX + dX.T)
                    lambda_matrix += delta * dX
                    """To be implemented. Symmetrize or Anti-symmetrize the updated matrices!!!"""
                    """To be implemented. Considering symmetries of the AGP lambda matrix."""

            structure_data = self.mcmc.hamiltonian_data.structure_data
            coulomb_potential_data = self.mcmc.hamiltonian_data.coulomb_potential_data
            geminal_data = Geminal_data(
                num_electron_up=self.mcmc.hamiltonian_data.wavefunction_data.geminal_data.num_electron_up,
                num_electron_dn=self.mcmc.hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn,
                orb_data_up_spin=self.mcmc.hamiltonian_data.wavefunction_data.geminal_data.orb_data_up_spin,
                orb_data_dn_spin=self.mcmc.hamiltonian_data.wavefunction_data.geminal_data.orb_data_dn_spin,
                lambda_matrix=lambda_matrix,
            )
            if self.mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_one_body_data is not None:
                jastrow_one_body_data = Jastrow_one_body_data(
                    jastrow_1b_param=j1_param,
                    structure_data=self.mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_one_body_data.structure_data,
                    core_electrons=self.mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_one_body_data.core_electrons,
                )
            else:
                jastrow_one_body_data = None
            if self.mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data is not None:
                jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=j2_param)
            else:
                jastrow_two_body_data = None
            if self.mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data is not None:
                jastrow_three_body_data = Jastrow_three_body_data(
                    orb_data=self.mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data,
                    j_matrix=j3_matrix,
                )
            else:
                jastrow_three_body_data = None
            jastrow_data = Jastrow_data(
                jastrow_one_body_data=jastrow_one_body_data,
                jastrow_two_body_data=jastrow_two_body_data,
                jastrow_three_body_data=jastrow_three_body_data,
            )
            wavefunction_data = Wavefunction_data(geminal_data=geminal_data, jastrow_data=jastrow_data)
            hamiltonian_data = Hamiltonian_data(
                structure_data=structure_data,
                wavefunction_data=wavefunction_data,
                coulomb_potential_data=coulomb_potential_data,
            )
            logger.info("Wavefunction has been updated. Optimization loop is done.")
            logger.info("")
            self.mcmc.hamiltonian_data = hamiltonian_data

            # dump WF
            if mpi_rank == 0:
                if (i_opt + 1) % wf_dump_freq == 0 or (i_opt + 1) == num_opt_steps:
                    hamiltonian_data_filename = f"hamiltonian_data_opt_step_{i_opt + 1 + self.__i_opt}.chk"
                    logger.info(f"Hamiltonian data is dumped as a checkpoint file: {hamiltonian_data_filename}.")
                    self.mcmc.hamiltonian_data.dump(hamiltonian_data_filename)

            # check max time
            vmcopt_current = time.perf_counter()
            if max_time < vmcopt_current - vmcopt_total_start:
                logger.info(f"max_time = {max_time} sec. exceeds.")
                logger.info("break the vmcopt loop.")
                break

        # update WF opt counter
        self.__i_opt += i_opt + 1

    def get_E(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> tuple[float, float]:
        """Return the mean and std of the computed local energy.

        Args:
            num_mcmc_warmup_steps (int): the number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks

        Return:
            tuple[float, float, float, float]:
                The mean and std values of the totat energy and those of the variance
                estimated by the Jackknife method with the Args. (E_mean, E_std, Var_mean, Var_std).
        """
        if self.mcmc.e_L.size != 0:
            e_L = self.mcmc.e_L[num_mcmc_warmup_steps:]
            e_L2 = self.mcmc.e_L2[num_mcmc_warmup_steps:]
            w_L = self.mcmc.w_L[num_mcmc_warmup_steps:]
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
            w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
            w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
            w_L_e_L2_split = np.array_split(w_L * e_L2, num_mcmc_bin_blocks, axis=0)
            w_L_e_L2_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L2_split]))
        else:
            w_L_binned = []
            w_L_e_L_binned = []
            w_L_e_L2_binned = []

        w_L_binned = mpi_comm.reduce(w_L_binned, op=MPI.SUM, root=0)
        w_L_e_L_binned = mpi_comm.reduce(w_L_e_L_binned, op=MPI.SUM, root=0)
        w_L_e_L2_binned = mpi_comm.reduce(w_L_e_L2_binned, op=MPI.SUM, root=0)

        if mpi_rank == 0:
            w_L_binned = np.array(w_L_binned)
            w_L_e_L_binned = np.array(w_L_e_L_binned)
            w_L_e_L2_binned = np.array(w_L_e_L2_binned)

            # jackknife implementation
            w_L_binned_sum = np.sum(w_L_binned)
            w_L_e_L_binned_sum = np.sum(w_L_e_L_binned)
            w_L_e_L2_binned_sum = np.sum(w_L_e_L2_binned)

            M = w_L_binned.size
            logger.debug(f"Total number of binned samples = {M}")

            E_jackknife_binned = np.array(
                [(w_L_e_L_binned_sum - w_L_e_L_binned[m]) / (w_L_binned_sum - w_L_binned[m]) for m in range(M)]
            )

            E2_jackknife_binned = np.array(
                [(w_L_e_L2_binned_sum - w_L_e_L2_binned[m]) / (w_L_binned_sum - w_L_binned[m]) for m in range(M)]
            )

            Var_jackknife_binned = E2_jackknife_binned - E_jackknife_binned**2  # E^2 = <E^2> - <E>^2

            E_mean = np.average(E_jackknife_binned)
            E_std = np.sqrt(M - 1) * np.std(E_jackknife_binned)
            Var_mean = np.average(Var_jackknife_binned)
            Var_std = np.sqrt(M - 1) * np.std(Var_jackknife_binned)

            logger.devel(f"E = {E_mean} +- {E_std} Ha.")
            logger.devel(f"Var(E) = {Var_mean} +- {Var_std} Ha^2.")
        else:
            E_mean = 0.0
            E_std = 0.0
            Var_mean = 0.0
            Var_std = 0.0

        E_mean = mpi_comm.bcast(E_mean, root=0)
        E_std = mpi_comm.bcast(E_std, root=0)
        Var_mean = mpi_comm.bcast(Var_mean, root=0)
        Var_std = mpi_comm.bcast(Var_std, root=0)

        return (E_mean, E_std, Var_mean, Var_std)

    def get_aF(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ):
        """Return the mean and std of the computed atomic forces.

        Args:
            num_mcmc_warmup_steps (int): the number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks

        Return:
            tuple[npt.NDArray, npt.NDArray]:
                The mean and std values of the computed atomic forces
                estimated by the Jackknife method with the Args.
                The dimention of the arrays is (N, 3).
        """
        if self.mcmc.e_L.size != 0:
            w_L = self.mcmc.w_L[num_mcmc_warmup_steps:]
            e_L = self.mcmc.e_L[num_mcmc_warmup_steps:]
            de_L_dR = self.mcmc.de_L_dR[num_mcmc_warmup_steps:]
            de_L_dr_up = self.mcmc.de_L_dr_up[num_mcmc_warmup_steps:]
            de_L_dr_dn = self.mcmc.de_L_dr_dn[num_mcmc_warmup_steps:]
            dln_Psi_dr_up = self.mcmc.dln_Psi_dr_up[num_mcmc_warmup_steps:]
            dln_Psi_dr_dn = self.mcmc.dln_Psi_dr_dn[num_mcmc_warmup_steps:]
            dln_Psi_dR = self.mcmc.dln_Psi_dR[num_mcmc_warmup_steps:]
            omega_up = self.mcmc.omega_up[num_mcmc_warmup_steps:]
            omega_dn = self.mcmc.omega_dn[num_mcmc_warmup_steps:]
            domega_dr_up = self.mcmc.domega_dr_up[num_mcmc_warmup_steps:]
            domega_dr_dn = self.mcmc.domega_dr_dn[num_mcmc_warmup_steps:]

            logger.devel(f"w_L.shape for MPI-rank={mpi_rank} is {w_L.shape}")

            logger.devel(f"e_L.shape for MPI-rank={mpi_rank} is {e_L.shape}")

            logger.devel(f"de_L_dR.shape for MPI-rank={mpi_rank} is {de_L_dR.shape}")
            logger.devel(f"de_L_dr_up.shape for MPI-rank={mpi_rank} is {de_L_dr_up.shape}")
            logger.devel(f"de_L_dr_dn.shape for MPI-rank={mpi_rank} is {de_L_dr_dn.shape}")

            logger.devel(f"dln_Psi_dr_up.shape for MPI-rank={mpi_rank} is {dln_Psi_dr_up.shape}")
            logger.devel(f"dln_Psi_dr_dn.shape for MPI-rank={mpi_rank} is {dln_Psi_dr_dn.shape}")
            logger.devel(f"dln_Psi_dR.shape for MPI-rank={mpi_rank} is {dln_Psi_dR.shape}")

            logger.devel(f"omega_up.shape for MPI-rank={mpi_rank} is {omega_up.shape}")
            logger.devel(f"omega_dn.shape for MPI-rank={mpi_rank} is {omega_dn.shape}")
            logger.devel(f"domega_dr_up.shape for MPI-rank={mpi_rank} is {domega_dr_up.shape}")
            logger.devel(f"domega_dr_dn.shape for MPI-rank={mpi_rank} is {domega_dr_dn.shape}")

            force_HF = (
                de_L_dR
                + np.einsum("iwjk,iwkl->iwjl", omega_up, de_L_dr_up)
                + np.einsum("iwjk,iwkl->iwjl", omega_dn, de_L_dr_dn)
            )

            force_PP = (
                dln_Psi_dR
                + np.einsum("iwjk,iwkl->iwjl", omega_up, dln_Psi_dr_up)
                + np.einsum("iwjk,iwkl->iwjl", omega_dn, dln_Psi_dr_dn)
                + 1.0 / 2.0 * (domega_dr_up + domega_dr_dn)
            )

            E_L_force_PP = np.einsum("iw,iwjk->iwjk", e_L, force_PP)

            logger.devel(f"w_L.shape for MPI-rank={mpi_rank} is {w_L.shape}")
            logger.devel(f"e_L.shape for MPI-rank={mpi_rank} is {e_L.shape}")
            logger.devel(f"force_HF.shape for MPI-rank={mpi_rank} is {force_HF.shape}")
            logger.devel(f"force_PP.shape for MPI-rank={mpi_rank} is {force_PP.shape}")
            logger.devel(f"E_L_force_PP.shape for MPI-rank={mpi_rank} is {E_L_force_PP.shape}")

            # split and binning with multiple walkers
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
            w_L_force_HF_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_HF), num_mcmc_bin_blocks, axis=0)
            w_L_force_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_PP), num_mcmc_bin_blocks, axis=0)
            w_L_E_L_force_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, E_L_force_PP), num_mcmc_bin_blocks, axis=0)

            w_L_binned = list(np.ravel(np.sum(np.stack(w_L_split), axis=1)))
            w_L_e_L_binned = list(np.ravel(np.sum(np.stack(w_L_e_L_split), axis=1)))

            w_L_force_HF_sum = np.array([np.sum(arr, axis=0) for arr in w_L_force_HF_split])
            w_L_force_HF_binned_shape = (
                w_L_force_HF_sum.shape[0] * w_L_force_HF_sum.shape[1],
                w_L_force_HF_sum.shape[2],
                w_L_force_HF_sum.shape[3],
            )
            w_L_force_HF_binned = list(w_L_force_HF_sum.reshape(w_L_force_HF_binned_shape))

            w_L_force_PP_sum = np.array([np.sum(arr, axis=0) for arr in w_L_force_PP_split])
            w_L_force_PP_binned_shape = (
                w_L_force_PP_sum.shape[0] * w_L_force_PP_sum.shape[1],
                w_L_force_PP_sum.shape[2],
                w_L_force_PP_sum.shape[3],
            )
            w_L_force_PP_binned = list(w_L_force_PP_sum.reshape(w_L_force_PP_binned_shape))

            w_L_E_L_force_PP_sum = np.array([np.sum(arr, axis=0) for arr in w_L_E_L_force_PP_split])
            w_L_E_L_force_PP_binned_shape = (
                w_L_E_L_force_PP_sum.shape[0] * w_L_E_L_force_PP_sum.shape[1],
                w_L_E_L_force_PP_sum.shape[2],
                w_L_E_L_force_PP_sum.shape[3],
            )
            w_L_E_L_force_PP_binned = list(w_L_E_L_force_PP_sum.reshape(w_L_E_L_force_PP_binned_shape))

        else:
            w_L_binned = []
            w_L_e_L_binned = []
            w_L_force_HF_binned = []
            w_L_force_PP_binned = []
            w_L_E_L_force_PP_binned = []

        # MPI reduce
        w_L_binned = mpi_comm.reduce(w_L_binned, op=MPI.SUM, root=0)
        w_L_e_L_binned = mpi_comm.reduce(w_L_e_L_binned, op=MPI.SUM, root=0)
        w_L_force_HF_binned = mpi_comm.reduce(w_L_force_HF_binned, op=MPI.SUM, root=0)
        w_L_force_PP_binned = mpi_comm.reduce(w_L_force_PP_binned, op=MPI.SUM, root=0)
        w_L_E_L_force_PP_binned = mpi_comm.reduce(w_L_E_L_force_PP_binned, op=MPI.SUM, root=0)

        if mpi_rank == 0:
            w_L_binned = np.array(w_L_binned)
            w_L_e_L_binned = np.array(w_L_e_L_binned)
            w_L_force_HF_binned = np.array(w_L_force_HF_binned)
            w_L_force_PP_binned = np.array(w_L_force_PP_binned)
            w_L_E_L_force_PP_binned = np.array(w_L_E_L_force_PP_binned)

            logger.devel(f"w_L_binned.shape for MPI-rank={mpi_rank} is {w_L_binned.shape}")
            logger.devel(f"w_L_e_L_binned.shape for MPI-rank={mpi_rank} is {w_L_e_L_binned.shape}")
            logger.devel(f"w_L_force_HF_binned.shape for MPI-rank={mpi_rank} is {w_L_force_HF_binned.shape}")
            logger.devel(f"w_L_force_PP_binned.shape for MPI-rank={mpi_rank} is {w_L_force_PP_binned.shape}")
            logger.devel(f"w_L_E_L_force_PP_binned.shape for MPI-rank={mpi_rank} is {w_L_E_L_force_PP_binned.shape}")

            M = w_L_binned.size
            logger.devel(f"Total number of binned samples = {M}")

            force_HF_jn = np.array(
                [
                    (np.sum(w_L_force_HF_binned, axis=0) - w_L_force_HF_binned[j])
                    / (np.sum(w_L_binned, axis=0) - w_L_binned[j])
                    for j in range(M)
                ]
            )

            force_Pulay_jn = np.array(
                [
                    -2.0
                    * (
                        (np.sum(w_L_E_L_force_PP_binned, axis=0) - w_L_E_L_force_PP_binned[j])
                        / (np.sum(w_L_binned) - w_L_binned[j])
                        - (
                            (np.sum(w_L_e_L_binned) - w_L_e_L_binned[j])
                            / (np.sum(w_L_binned) - w_L_binned[j])
                            * (np.sum(w_L_force_PP_binned, axis=0) - w_L_force_PP_binned[j])
                            / (np.sum(w_L_binned) - w_L_binned[j])
                        )
                    )
                    for j in range(M)
                ]
            )

            logger.devel(f"force_HF_jn.shape for MPI-rank={mpi_rank} is {force_HF_jn.shape}")
            logger.devel(f"force_Pulay_jn.shape for MPI-rank={mpi_rank} is {force_Pulay_jn.shape}")

            force_jn = force_HF_jn + force_Pulay_jn

            force_mean = np.average(force_jn, axis=0)
            force_std = np.sqrt(M - 1) * np.std(force_jn, axis=0)

            logger.devel(f"force_mean.shape  = {force_mean.shape}.")
            logger.devel(f"force_std.shape  = {force_std.shape}.")

            logger.devel(f"force = {force_mean} +- {force_std} Ha.")

        else:
            force_mean = np.array([])
            force_std = np.array([])

        force_mean = mpi_comm.bcast(force_mean, root=0)
        force_std = mpi_comm.bcast(force_std, root=0)

        return (force_mean, force_std)

    def get_dln_WF(self, num_mcmc_warmup_steps: int = 50, chosen_param_index: list = None):
        """Return the derivativs of ln_WF wrt variational parameters.

        Args:
            num_mcmc_warmup_steps (int): The number of warmup steps.
            chosen_param_index (list):
                The chosen parameter index to compute the generalized forces.
                if None, all parameters are used.

        Return:
            O_matrix(npt.NDArray): The matrix containing O_k = d ln Psi / dc_k,
            where k is the flattened variational parameter index. The dimenstion
            of O_matrix is (M, nw, k), where M is the MCMC step and nw is the walker index.
        """
        dln_Psi_dc_list = self.mcmc.opt_param_dict["dln_Psi_dc_list"]

        # here, the thrid index indicates the flattened variational parameter index.
        O_matrix = np.empty((self.mcmc.mcmc_counter, self.mcmc.num_walkers, 0))

        for dln_Psi_dc in dln_Psi_dc_list:
            logger.devel(f"dln_Psi_dc.shape={dln_Psi_dc.shape}.")
            if dln_Psi_dc.ndim == 2:  # i.e., sclar variational param.
                dln_Psi_dc_reshaped = dln_Psi_dc.reshape(dln_Psi_dc.shape[0], dln_Psi_dc.shape[1], 1)
            else:
                dln_Psi_dc_reshaped = dln_Psi_dc.reshape(
                    dln_Psi_dc.shape[0], dln_Psi_dc.shape[1], int(np.prod(dln_Psi_dc.shape[2:]))
                )
            O_matrix = np.concatenate((O_matrix, dln_Psi_dc_reshaped), axis=2)

        logger.devel(f"O_matrix.shape = {O_matrix.shape}")
        if chosen_param_index is None:
            O_matrix_chosen = O_matrix[num_mcmc_warmup_steps:]
        else:
            O_matrix_chosen = O_matrix[num_mcmc_warmup_steps:, :, chosen_param_index]  # O.... (x....) (M, nw, L) matrix
        logger.devel(f"O_matrix_chosen.shape = {O_matrix_chosen.shape}")
        return O_matrix_chosen

    def get_gF(
        self,
        mpi_broadcast: bool = True,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
        chosen_param_index: list = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Compute the derivatives of E wrt variational parameters, a.k.a. generalized forces.

        Args:
            mpi_broadcast (bool):
                If true, the computed S is shared among all MPI processes.
                If false, only the root node has it.
            num_mcmc_warmup_steps (int): The number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks
            chosen_param_index (npt.NDArray):
                The chosen parameter index to compute the generalized forces.
                If None, all parameters are used.

        Return:
            tuple[npt.NDArray, npt.NDArray]: mean and std of generalized forces.
            Dim. is 1D vector with L elements, where L is the number of flattened
            variational parameters.
        """
        logger.info("Computing the generalized force vector f...")
        if self.mcmc.e_L.size != 0:
            w_L = self.mcmc.w_L[num_mcmc_warmup_steps:]
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))

            e_L = self.mcmc.e_L[num_mcmc_warmup_steps:]
            w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
            w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))

            O_matrix = self.get_dln_WF(num_mcmc_warmup_steps=num_mcmc_warmup_steps, chosen_param_index=chosen_param_index)
            w_L_O_matrix_split = np.array_split(np.einsum("iw,iwj->iwj", w_L, O_matrix), num_mcmc_bin_blocks, axis=0)
            w_L_O_matrix_sum = np.array([np.sum(arr, axis=0) for arr in w_L_O_matrix_split])
            w_L_O_matrix_binned_shape = (
                w_L_O_matrix_sum.shape[0] * w_L_O_matrix_sum.shape[1],
                w_L_O_matrix_sum.shape[2],
            )
            w_L_O_matrix_binned = list(w_L_O_matrix_sum.reshape(w_L_O_matrix_binned_shape))

            e_L_O_matrix = np.einsum("iw,iwj->iwj", e_L, O_matrix)
            w_L_e_L_O_matrix_split = np.array_split(np.einsum("iw,iwj->iwj", w_L, e_L_O_matrix), num_mcmc_bin_blocks, axis=0)
            w_L_e_L_O_matrix_sum = np.array([np.sum(arr, axis=0) for arr in w_L_e_L_O_matrix_split])
            w_L_e_L_O_matrix_binned_shape = (
                w_L_e_L_O_matrix_sum.shape[0] * w_L_e_L_O_matrix_sum.shape[1],
                w_L_e_L_O_matrix_sum.shape[2],
            )
            w_L_e_L_O_matrix_binned = list(w_L_e_L_O_matrix_sum.reshape(w_L_e_L_O_matrix_binned_shape))
        else:
            w_L_binned = []
            w_L_e_L_binned = []
            w_L_O_matrix_binned = []
            w_L_e_L_O_matrix_binned = []

        w_L_binned = mpi_comm.reduce(w_L_binned, op=MPI.SUM, root=0)
        w_L_e_L_binned = mpi_comm.reduce(w_L_e_L_binned, op=MPI.SUM, root=0)
        w_L_O_matrix_binned = mpi_comm.reduce(w_L_O_matrix_binned, op=MPI.SUM, root=0)
        w_L_e_L_O_matrix_binned = mpi_comm.reduce(w_L_e_L_O_matrix_binned, op=MPI.SUM, root=0)

        if mpi_rank == 0:
            w_L_binned = np.array(w_L_binned)
            w_L_e_L_binned = np.array(w_L_e_L_binned)
            w_L_O_matrix_binned = np.array(w_L_O_matrix_binned)
            w_L_e_L_O_matrix_binned = np.array(w_L_e_L_O_matrix_binned)

            M = w_L_binned.size
            logger.debug(f"Total number of binned samples = {M}")

            eL_O_jn = np.array(
                [
                    (np.sum(w_L_e_L_O_matrix_binned, axis=0) - w_L_e_L_O_matrix_binned[j])
                    / (np.sum(w_L_binned) - w_L_binned[j])
                    for j in range(M)
                ]
            )
            logger.devel(f"eL_O_jn = {eL_O_jn}")
            logger.devel(f"eL_O_jn.shape = {eL_O_jn.shape}")

            eL_jn = np.array(
                [(np.sum(w_L_e_L_binned, axis=0) - w_L_e_L_binned[j]) / (np.sum(w_L_binned) - w_L_binned[j]) for j in range(M)]
            )
            logger.devel(f"eL_jn = {eL_jn}")
            logger.devel(f"eL_jn.shape = {eL_jn.shape}")

            O_jn = np.array(
                [
                    (np.sum(w_L_O_matrix_binned, axis=0) - w_L_O_matrix_binned[j]) / (np.sum(w_L_binned) - w_L_binned[j])
                    for j in range(M)
                ]
            )

            logger.devel(f"O_jn = {O_jn}")
            logger.devel(f"O_jn.shape = {O_jn.shape}")

            eL_barO_jn = np.einsum("i,ij->ij", eL_jn, O_jn)

            logger.devel(f"eL_barO_jn = {eL_barO_jn}")
            logger.devel(f"eL_barO_jn.shape = {eL_barO_jn.shape}")

            generalized_force_mean = np.average(-2.0 * (eL_O_jn - eL_barO_jn), axis=0)
            generalized_force_std = np.sqrt(M - 1) * np.std(-2.0 * (eL_O_jn - eL_barO_jn), axis=0)

            logger.devel(f"generalized_force_mean = {generalized_force_mean}")
            logger.devel(f"generalized_force_std = {generalized_force_std}")

            logger.devel(f"generalized_force_mean.shape = {generalized_force_mean.shape}")
            logger.devel(f"generalized_force_std.shape = {generalized_force_std.shape}")

        else:
            generalized_force_mean = None
            generalized_force_std = None

        if mpi_broadcast:
            # comm.Bcast(generalized_force_mean, root=0)
            # comm.Bcast(generalized_force_std, root=0)
            generalized_force_mean = mpi_comm.bcast(generalized_force_mean, root=0)
            generalized_force_std = mpi_comm.bcast(generalized_force_std, root=0)

        return (
            generalized_force_mean,
            generalized_force_std,
        )  # (L vector, L vector)

    ''' linear method (it works, but very slow.)
    def get_de_L(self, num_mcmc_warmup_steps: int = 50):
        """Return the derivativs of e_L wrt variational parameters.

        Args:
            num_mcmc_warmup_steps (int): The number of warmup steps.

        Return:
            de_L_matrix(npt.NDArray): The matrix containing de_L_k = d e_L / dc_k,
            where k is the flattened variational parameter index. The dimenstion
            of de_L_matrix is (M, nw, k), where M is the MCMC step and nw is the walker index.
        """
        opt_param_dict = self.__mcmc.opt_param_dict

        # dc_param_list = opt_param_dict["dc_param_list"]
        de_L_dc_list = opt_param_dict["de_L_dc_list"]
        # dc_size_list = opt_param_dict["dc_size_list"]
        # dc_shape_list = opt_param_dict["dc_shape_list"]
        # dc_flattened_index_list = opt_param_dict["dc_flattened_index_list"]

        # here, the thrid index indicates the flattened variational parameter index.
        de_L_matrix = np.empty((self.__mcmc.mcmc_counter, self.__mcmc.num_walkers, 0))

        for de_L_dc in de_L_dc_list:
            logger.devel(f"de_L_dc.shape={de_L_dc.shape}.")
            if de_L_dc.ndim == 2:  # i.e., sclar variational param.
                de_L_dc_reshaped = de_L_dc.reshape(de_L_dc.shape[0], de_L_dc.shape[1], 1)
            else:
                de_L_dc_reshaped = de_L_dc.reshape(de_L_dc.shape[0], de_L_dc.shape[1], int(np.prod(de_L_dc.shape[2:])))
            de_L_matrix = np.concatenate((de_L_matrix, de_L_dc_reshaped), axis=2)

        logger.devel(f"de_L_matrix.shape = {de_L_matrix.shape}")
        return de_L_matrix[num_mcmc_warmup_steps:]  # O.... (x....) (M, nw, L) matrix

    def get_H(
        self,
        mpi_broadcast: int = False,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute the surrogate Hessian matrix H.

        Args:
            mpi_broadcast (bool):
                If true, the computed H is shared among all MPI processes.
                If false, only the root node has it.
            num_mcmc_warmup_steps (int): The number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks

        Return:
            H_matrix (npt.NDArray):
                The mean and std of the surrogate matrix S.
                dim is (L, L) for both, where L is the number of variational parameter.
                L indicates the flattened variational parameter index.
        """
        logger.info("Computing the stochastic matrix S...")

        if self.__mcmc.e_L.size != 0:
            w_L = self.__mcmc.w_L[num_mcmc_warmup_steps:]
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_binned = list(np.ravel([np.mean(arr, axis=0) for arr in w_L_split]))

            O_matrix = self.get_dln_WF(num_mcmc_warmup_steps=num_mcmc_warmup_steps)
            O_matrix_split = np.array_split(O_matrix, num_mcmc_bin_blocks, axis=0)
            O_matrix_ave = np.array([np.mean(arr, axis=0) for arr in O_matrix_split])
            O_matrix_binned_shape = (
                O_matrix_ave.shape[0] * O_matrix_ave.shape[1],
                O_matrix_ave.shape[2],
            )
            O_matrix_binned = list(O_matrix_ave.reshape(O_matrix_binned_shape))

            w_L_O_matrix_split = np.array_split(np.einsum("iw,iwj->iwj", w_L, O_matrix), num_mcmc_bin_blocks, axis=0)
            w_L_O_matrix_ave = np.array([np.mean(arr, axis=0) for arr in w_L_O_matrix_split])
            w_L_O_matrix_binned_shape = (
                w_L_O_matrix_ave.shape[0] * w_L_O_matrix_ave.shape[1],
                w_L_O_matrix_ave.shape[2],
            )
            w_L_O_matrix_binned = list(w_L_O_matrix_ave.reshape(w_L_O_matrix_binned_shape))

            e_L = self.__mcmc.e_L[num_mcmc_warmup_steps:]
            e_L_split = np.array_split(e_L, num_mcmc_bin_blocks, axis=0)
            e_L_binned = list(np.ravel([np.mean(arr, axis=0) for arr in e_L_split]))

            e_L_O_matrix_split = np.array_split(np.einsum("iw, iwj -> iwj", e_L, O_matrix), num_mcmc_bin_blocks, axis=0)
            e_L_O_matrix_ave = np.array([np.mean(arr, axis=0) for arr in e_L_O_matrix_split])
            e_L_O_matrix_binned_shape = (
                e_L_O_matrix_ave.shape[0] * e_L_O_matrix_ave.shape[1],
                e_L_O_matrix_ave.shape[2],
            )
            e_L_O_matrix_binned = list(e_L_O_matrix_ave.reshape(e_L_O_matrix_binned_shape))

            de_L_matrix = self.get_de_L(num_mcmc_warmup_steps=num_mcmc_warmup_steps)
            de_L_matrix_split = np.array_split(de_L_matrix, num_mcmc_bin_blocks, axis=0)
            de_L_matrix_ave = np.array([np.mean(arr, axis=0) for arr in de_L_matrix_split])
            de_L_matrix_binned_shape = (
                de_L_matrix_ave.shape[0] * de_L_matrix_ave.shape[1],
                de_L_matrix_ave.shape[2],
            )
            de_L_matrix_binned = list(de_L_matrix_ave.reshape(de_L_matrix_binned_shape))

        else:
            w_L_binned = []
            e_L_binned = []
            O_matrix_binned = []
            e_L_O_matrix_binned = []
            w_L_O_matrix_binned = []
            de_L_matrix_binned = []

        w_L_binned = mpi_comm.reduce(w_L_binned, op=MPI.SUM, root=0)
        e_L_binned = mpi_comm.reduce(e_L_binned, op=MPI.SUM, root=0)
        O_matrix_binned = mpi_comm.reduce(O_matrix_binned, op=MPI.SUM, root=0)
        e_L_O_matrix_binned = mpi_comm.reduce(e_L_O_matrix_binned, op=MPI.SUM, root=0)
        w_L_O_matrix_binned = mpi_comm.reduce(w_L_O_matrix_binned, op=MPI.SUM, root=0)
        de_L_matrix_binned = mpi_comm.reduce(de_L_matrix_binned, op=MPI.SUM, root=0)

        if mpi_rank == 0:
            w_L_binned = np.array(w_L_binned)
            e_L_binned = np.array(e_L_binned)
            O_matrix_binned = np.array(O_matrix_binned)
            e_L_O_matrix_binned = np.array(e_L_O_matrix_binned)
            w_L_O_matrix_binned = np.array(w_L_O_matrix_binned)
            de_L_matrix_binned = np.array(de_L_matrix_binned)
            logger.info(f"w_L_binned.shape = {w_L_binned.shape}")
            logger.info(f"e_L_binned.shape = {e_L_binned.shape}")
            logger.info(f"O_matrix_binned.shape = {O_matrix_binned.shape}")
            logger.info(f"e_L_O_matrix_binned.shape = {e_L_O_matrix_binned.shape}")
            logger.info(f"w_L_O_matrix_binned.shape = {w_L_O_matrix_binned.shape}")
            logger.info(f"de_L_matrix_binned.shape = {de_L_matrix_binned.shape}")
            # S_mean = np.array(np.cov(O_matrix_binned, bias=True, rowvar=False)) # old
            O_bar = np.sum(w_L_O_matrix_binned, axis=0) / np.sum(w_L_binned, axis=0)
            de_L_bar = np.sum(de_L_matrix_binned, axis=0) / np.sum(w_L_binned, axis=0)
            e_L_O_bar = np.einsum("i,k->ik", e_L_binned, O_bar)
            w_O_bar = np.einsum("i,k->ik", w_L_binned, O_bar)
            logger.info(f"O_bar.shape = {O_bar.shape}")
            logger.info(f"e_L_O_bar.shape = {e_L_O_bar.shape}")
            logger.info(f"w_O_bar.shape = {w_O_bar.shape}")
            B_mean = (
                (w_L_O_matrix_binned - w_O_bar).T @ (de_L_matrix_binned - de_L_bar) / np.sum(w_L_binned)
            )  # weighted variance-covariance matrix
            K_mean = (
                (w_L_O_matrix_binned - w_O_bar).T @ (e_L_O_matrix_binned - e_L_O_bar) / np.sum(w_L_binned)
            )  # weighted variance-covariance matrix
            H_mean = B_mean + K_mean
            H_std = np.zeros(H_mean.size)
            logger.info(f"H_mean.shape = {H_mean.shape}")
            logger.devel(f"H_mean.is_nan for MPI-rank={mpi_rank} is {np.isnan(H_mean).any()}")
            logger.devel(f"H_mean.shape for MPI-rank={mpi_rank} is {H_mean.shape}")
        else:
            H_mean = None
            H_std = None

        if mpi_broadcast:
            # comm.Bcast(S_mean, root=0)
            # comm.Bcast(S_std, root=0)
            H_mean = mpi_comm.bcast(H_mean, root=0)
            H_std = mpi_comm.bcast(H_std, root=0)

        return (H_mean, H_std)  # (H_mu,nu ...., var(H)_mu,nu....) (L*L matrix, L*L matrix)

    '''

    ''' SR method (old)
    def get_S(
        self,
        mpi_broadcast: int = False,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute the preconditioning matrix S.

        Args:
            mpi_broadcast (bool):
                If true, the computed S is shared among all MPI processes.
                If false, only the root node has it.
            num_mcmc_warmup_steps (int): The number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks

        Return:
            S_matrix (npt.NDArray):
                The mean and std of the preconditioning matrix S.
                dim is (L, L) for both, where L is the number of variational parameter.
                L indicates the flattened variational parameter index.
        """
        logger.info("Computing the stochastic matrix S...")

        if self.__mcmc.e_L.size != 0:
            w_L = self.__mcmc.w_L[num_mcmc_warmup_steps:]
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_binned = list(np.ravel([np.mean(arr, axis=0) for arr in w_L_split]))

            O_matrix = self.get_dln_WF(num_mcmc_warmup_steps=num_mcmc_warmup_steps)
            O_matrix_split = np.array_split(O_matrix, num_mcmc_bin_blocks, axis=0)
            O_matrix_ave = np.array([np.mean(arr, axis=0) for arr in O_matrix_split])
            O_matrix_binned_shape = (
                O_matrix_ave.shape[0] * O_matrix_ave.shape[1],
                O_matrix_ave.shape[2],
            )
            O_matrix_binned = list(O_matrix_ave.reshape(O_matrix_binned_shape))

            w_L_O_matrix_split = np.array_split(np.einsum("iw,iwj->iwj", w_L, O_matrix), num_mcmc_bin_blocks, axis=0)
            w_L_O_matrix_ave = np.array([np.mean(arr, axis=0) for arr in w_L_O_matrix_split])
            w_L_O_matrix_binned_shape = (
                w_L_O_matrix_ave.shape[0] * w_L_O_matrix_ave.shape[1],
                w_L_O_matrix_ave.shape[2],
            )
            w_L_O_matrix_binned = list(w_L_O_matrix_ave.reshape(w_L_O_matrix_binned_shape))

        else:
            w_L_binned = []
            O_matrix_binned = []
            w_L_O_matrix_binned = []

        w_L_binned = mpi_comm.reduce(w_L_binned, op=MPI.SUM, root=0)
        O_matrix_binned = mpi_comm.reduce(O_matrix_binned, op=MPI.SUM, root=0)
        w_L_O_matrix_binned = mpi_comm.reduce(w_L_O_matrix_binned, op=MPI.SUM, root=0)

        if mpi_rank == 0:
            w_L_binned = np.array(w_L_binned)
            O_matrix_binned = np.array(O_matrix_binned)
            w_L_O_matrix_binned = np.array(w_L_O_matrix_binned)
            logger.info(f"w_L_binned.shape = {w_L_binned.shape}")
            logger.info(f"O_matrix_binned.shape = {O_matrix_binned.shape}")
            logger.info(f"w_L_O_matrix_binned.shape = {w_L_O_matrix_binned.shape}")
            # S_mean_old = np.array(np.cov(O_matrix_binned, bias=True, rowvar=False))  # old
            O_bar = np.sum(w_L_O_matrix_binned, axis=0) / np.sum(w_L_binned)
            w_O_bar = np.einsum("i,k->ik", w_L_binned, O_bar)
            logger.info(f"O_bar.shape = {O_bar.shape}")
            logger.info(f"w_O_bar.shape = {w_O_bar.shape}")
            S_mean = (
                (w_L_O_matrix_binned - w_O_bar).T @ (O_matrix_binned - O_bar) / np.sum(w_L_binned)
            )  # weighted variance-covariance matrix
            S_std = np.zeros(S_mean.size)
            # logger.info(f"np.max(np.abs(S_mean - S_mean_old)) = {np.max(np.abs(S_mean - S_mean_old))}.")
            logger.info(f"S_mean.shape = {S_mean.shape}")
            logger.devel(f"S_mean.is_nan for MPI-rank={mpi_rank} is {np.isnan(S_mean).any()}")
            logger.devel(f"S_mean.shape for MPI-rank={mpi_rank} is {S_mean.shape}")
        else:
            S_mean = None
            S_std = None

        if mpi_broadcast:
            # comm.Bcast(S_mean, root=0)
            # comm.Bcast(S_std, root=0)
            S_mean = mpi_comm.bcast(S_mean, root=0)
            S_std = mpi_comm.bcast(S_std, root=0)

        return (S_mean, S_std)  # (S_mu,nu ...., var(S)_mu,nu....) (L*L matrix, L*L matrix)

    def run_optimize_old(
        self,
        num_mcmc_steps: int = 100,
        num_opt_steps: int = 1,
        delta: float = 0.001,
        epsilon: float = 1.0e-3,
        wf_dump_freq: int = 10,
        max_time: int = 86400,
        num_mcmc_warmup_steps: int = 0,
        num_mcmc_bin_blocks: int = 100,
        # opt_J1_param: bool = True, # to be implemented.
        opt_J2_param: bool = True,
        opt_J3_param: bool = True,
        opt_J4_param: bool = False,
        opt_lambda_param: bool = False,
    ):
        """Optimizing wavefunction.

        Optimizing Wavefunction using the Stochastic Reconfiguration Method.

        Args:
            num_mcmc_steps(int): The number of MCMC samples per walker.
            num_opt_steps(int): The number of WF optimization step.
            delta(float):
                The prefactor of the SR matrix for adjusting the optimization step.
                i.e., c_i <- c_i + delta * S^{-1} f
            epsilon(float):
                The regralization factor of the SR matrix
                i.e., S <- S + I * delta
            wf_dump_freq(int):
                The frequency of WF data (i.e., hamiltonian_data.chk)
            max_time(int):
                The maximum time (sec.) If maximum time exceeds,
                the method exits the MCMC loop.
            num_mcmc_warmup_steps (int): number of equilibration steps.
            num_mcmc_bin_blocks (int): number of blocks for reblocking.
            opt_J1_param (bool): optimize one-body Jastrow # to be implemented.
            opt_J2_param (bool): optimize two-body Jastrow
            opt_J3_param (bool): optimize three-body Jastrow
            opt_J4_param (bool): optimize four-body Jastrow # to be implemented.
            opt_lambda_param (bool): optimize lambda_matrix in the determinant part.

        """
        vmcopt_total_start = time.perf_counter()

        dc_size_list = self.__mcmc.opt_param_dict["dc_size_list"]
        logger.info(f"The number of variational paramers = {np.sum(dc_size_list)}.")

        # main vmcopt loop
        for i_opt in range(num_opt_steps):
            logger.info(f"i_opt={i_opt + 1 + self.__i_opt}/{num_opt_steps + self.__i_opt}.")

            if mpi_rank == 0:
                logger.info(f"num_mcmc_warmup_steps={num_mcmc_warmup_steps}.")
                logger.info(f"num_mcmc_bin_blocks={num_mcmc_bin_blocks}.")
                logger.info(f"num_mcmc_steps={num_mcmc_steps}.")

            logger.info(
                f"twobody param before opt. = {self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param}"
            )

            # run MCMC
            self.__mcmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)

            # get e_L
            E, E_std = self.get_E(num_mcmc_warmup_steps=num_mcmc_warmup_steps, num_mcmc_bin_blocks=num_mcmc_bin_blocks)
            logger.info(f"E = {E} +- {E_std} Ha")

            # get f and f_std (generalized forces)
            f, f_std = self.get_gF(
                mpi_broadcast=False, num_mcmc_warmup_steps=num_mcmc_warmup_steps, num_mcmc_bin_blocks=num_mcmc_bin_blocks
            )
            # get S (preconditioning matrix)
            S, _ = self.get_S(
                mpi_broadcast=False, num_mcmc_warmup_steps=num_mcmc_warmup_steps, num_mcmc_bin_blocks=num_mcmc_bin_blocks
            )

            """ linear method
            # get H (surrogate Hessian matrix)
            H, _ = self.get_H(
                mpi_broadcast=False, num_mcmc_warmup_steps=num_mcmc_warmup_steps, num_mcmc_bin_blocks=num_mcmc_bin_blocks
            )
            """

            if mpi_rank == 0:
                signal_to_noise_f = np.abs(f) / f_std
                logger.info(f"Max |f| = {np.max(np.abs(f)):.3f} Ha/a.u.")
                logger.devel(f"f_std of Max |f| = {f_std[np.argmax(np.abs(f))]:.3f} Ha/a.u.")
                logger.info(f"Max of signal-to-noise of f = max(|f|/|std f|) = {np.max(signal_to_noise_f):.3f}.")

            logger.info("Computing the inverse of the stochastic matrix S^{-1}f...")

            if mpi_rank == 0:
                """ LR method, to be removed
                # SR with linear method
                if S.ndim != 0:
                    I = np.eye(S.shape[0])
                    S_prime = S + epsilon * I
                    # solve Sx=f
                    S_inv_f = scipy.linalg.solve(S_prime, f, assume_a="sym")

                    H_0 = E
                    H_1 = -1.0 / 2.0 * (S_inv_f.T @ f)
                    H_2 = S_inv_f.T @ H @ S_inv_f
                    S_2 = S_inv_f.T @ S_prime @ S_inv_f

                    logger.info(f"H_0 = {H_0}.")
                    logger.info(f"H_1 = {H_1}.")
                    logger.info(f"S_2 = {S_2}.")
                    logger.info(f"H_2 = {H_2}.")
                    logger.info(f"(H_2 + 2 * H_0 * H_1) ** 2 - 8 * H_1**3) = {(H_2 + 2 * H_0 * H_1) ** 2 - 8 * H_1**3}.")

                    gamma_plus = (H_2 + 2 * H_0 * H_1 + np.sqrt((H_2 + 2 * H_0 * H_1) ** 2 - 8 * H_1**3)) / (-4.0 * H_1**2)
                    gamma_minus = (H_2 + 2 * H_0 * H_1 - np.sqrt((H_2 + 2 * H_0 * H_1) ** 2 - 8 * H_1**3)) / (-4.0 * H_1**2)
                    logger.info(f"gamma_plus = {gamma_plus}")
                    logger.info(f"gamma_minus = {gamma_minus}")
                    gamma_chosen = np.maximum(gamma_plus, gamma_minus)
                    logger.info(f"gamma_chosen = {gamma_chosen}")
                    if gamma_chosen < 0:
                        logger.warning(f"gamma_chosen = {gamma_chosen} is negative!!")
                    X = gamma_chosen * S_inv_f

                else:
                    raise NotImplementedError
                    I = 1.0
                    S_prime = S + epsilon * I
                    # solve Sx=f
                    X = 1.0 / S_prime * f
                """

                # """ # SR
                if S.ndim != 0:
                    # I = np.eye(S.shape[0])
                    # S_prime = S + epsilon * I
                    S_prime = S.copy()
                    S_prime[np.diag_indices_from(S_prime)] += epsilon
                    # solve Sx=f
                    X = scipy.linalg.solve(S_prime, f, assume_a="sym")
                else:
                    # I = 1.0
                    # S_prime = S + epsilon * I
                    S_prime = S + epsilon
                    # solve Sx=f
                    X = 1.0 / S_prime * f

                # logger.info(f"The condition number of the matrix S is {np.linalg.cond(S)}.")
                # logger.info(f"The diagonal elements of S_prime = {np.diag(S_prime)}.")
                # logger.info(f"The S_prime is symmetric? = {np.allclose(S_prime, S_prime.T, atol=1.0e-10)}.")
                # logger.info(f"The condition number of the matrix S_prime is {np.linalg.cond(S_prime)}.")
                # """

                # steepest decent (SD)
                # X = f

            else:
                X = None

            X = mpi_comm.bcast(X, root=0)
            logger.devel(f"X for MPI-rank={mpi_rank} is {X}")
            logger.devel(f"X.shape for MPI-rank={mpi_rank} is {X.shape}")
            logger.info(f"max(dX) for MPI-rank={mpi_rank} is {np.max(X)}")

            dc_param_list = self.__mcmc.opt_param_dict["dc_param_list"]
            dc_shape_list = self.__mcmc.opt_param_dict["dc_shape_list"]
            dc_flattened_index_list = self.__mcmc.opt_param_dict["dc_flattened_index_list"]

            j2_param = self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
            jastrow_two_body_pade_flag = self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_flag
            jastrow_three_body_flag = self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_flag
            j3_orb_data = self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
            j3_matrix = self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix
            lambda_matrix = self.__mcmc.hamiltonian_data.wavefunction_data.geminal_data.lambda_matrix

            for ii, opt_param in enumerate(dc_param_list):
                param_shape = dc_shape_list[ii]
                param_index = [i for i, v in enumerate(dc_flattened_index_list) if v == ii]
                dX = X[param_index].reshape(param_shape)
                logger.info(f"dX.shape for MPI-rank={mpi_rank} is {dX.shape}")
                if dX.shape == (1,):
                    dX = dX[0]
                if opt_J2_param and opt_param == "j2_param":
                    j2_param += delta * dX
                if opt_J3_param and opt_param == "j3_matrix":
                    # j1 part (rectanglar)
                    j3_matrix[:, -1] += delta * dX[:, -1]
                    # j3 part (square)
                    if np.allclose(j3_matrix[:, :-1], j3_matrix[:, :-1].T, atol=1e-8):
                        logger.info("The j3 matrix is symmetric. Keep it while updating.")
                        dX = 1.0 / 2.0 * (dX[:, :-1] + dX[:, :-1].T)
                    else:
                        dX = dX[:, :-1]
                    j3_matrix[:, :-1] += delta * dX
                    """To be implemented. Opt only the block diagonal parts, i.e. only the J3 part."""
                if opt_lambda_param and opt_param == "lambda_matrix":
                    if np.allclose(lambda_matrix, lambda_matrix.T, atol=1e-8):
                        logger.info("The lambda matrix is symmetric. Keep it while updating.")
                        dX = 1.0 / 2.0 * (dX + dX.T)
                    lambda_matrix += delta * dX
                    """To be implemented. Symmetrize or Anti-symmetrize the updated matrices!!!"""
                    """To be implemented. Considering symmetries of the AGP lambda matrix."""

            structure_data = self.__mcmc.hamiltonian_data.structure_data
            coulomb_potential_data = self.__mcmc.hamiltonian_data.coulomb_potential_data
            geminal_data = Geminal_data(
                num_electron_up=self.__mcmc.hamiltonian_data.wavefunction_data.geminal_data.num_electron_up,
                num_electron_dn=self.__mcmc.hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn,
                orb_data_up_spin=self.__mcmc.hamiltonian_data.wavefunction_data.geminal_data.orb_data_up_spin,
                orb_data_dn_spin=self.__mcmc.hamiltonian_data.wavefunction_data.geminal_data.orb_data_dn_spin,
                lambda_matrix=lambda_matrix,
            )
            jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=j2_param)
            jastrow_three_body_data = Jastrow_three_body_data(
                orb_data=j3_orb_data,
                j_matrix=j3_matrix,
            )
            jastrow_data = Jastrow_data(
                jastrow_two_body_data=jastrow_two_body_data,
                jastrow_three_body_data=jastrow_three_body_data,
                jastrow_two_body_flag=jastrow_two_body_pade_flag,
                jastrow_three_body_flag=jastrow_three_body_flag,
            )
            wavefunction_data = Wavefunction_data(geminal_data=geminal_data, jastrow_data=jastrow_data)
            hamiltonian_data = Hamiltonian_data(
                structure_data=structure_data,
                wavefunction_data=wavefunction_data,
                coulomb_potential_data=coulomb_potential_data,
            )

            logger.info("WF updated")
            self.__mcmc.hamiltonian_data = hamiltonian_data

            logger.info(
                f"twobody param after opt. = {self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param}"
            )

            # dump WF
            if mpi_rank == 0:
                if (i_opt + 1) % wf_dump_freq == 0 or (i_opt + 1) == num_opt_steps:
                    logger.info("Hamiltonian data is dumped as a checkpoint file.")
                    self.__mcmc.hamiltonian_data.dump(f"hamiltonian_data_opt_step_{i_opt + 1}.chk")

            # check max time
            vmcopt_current = time.perf_counter()
            if max_time < vmcopt_current - vmcopt_total_start:
                logger.info(f"max_time = {max_time} sec. exceeds.")
                logger.info("break the vmcopt loop.")
                break

        # update WF opt counter
        self.__i_opt += i_opt + 1
    '''


if __name__ == "__main__":
    import os
    import pickle
    from logging import Formatter, StreamHandler, getLogger

    from .trexio_wrapper import read_trexio_file

    logger_level = "MPI-DEBUG"

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
            log.setLevel("ERROR")
            stream_handler = StreamHandler()
            stream_handler.setLevel("ERROR")
            handler_format = Formatter(f"MPI-rank={mpi_rank}: %(name)s - %(levelname)s - %(lineno)d - %(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
    elif logger_level == "MPI-DEBUG":
        if mpi_rank == 0:
            log.setLevel("DEBUG")
            stream_handler = StreamHandler()
            stream_handler.setLevel("DEBUG")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
        else:
            log.setLevel("ERROR")
            stream_handler = StreamHandler()
            stream_handler.setLevel("ERROR")
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
        mos_data,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "water_ccpvtz_trexio.hdf5"))
    """

    """
    # water cc-pVTZ with Mitas ccECP (8 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "water_ccecp_ccpvtz_cart.hdf5"))
    """

    # """
    # H2 dimer cc-pV5Z with Mitas ccECP (2 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "H2_dimer_ccpv5z_trexio.hdf5"))
    # """

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

    # """
    jastrow_one_body_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.75)
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=mos_data)

    # define data
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_one_body_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
    )

    # conversion of SD to AGP
    geminal_data = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)

    # geminal_data = geminal_mo_data
    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    # """

    #'''
    # """
    hamiltonian_chk = "hamiltonian_data_water.chk"
    # hamiltonian_chk = "hamiltonian_data_water_methane.chk"
    # hamiltonian_chk = "hamiltonian_data_benzene.chk"
    hamiltonian_chk = "hamiltonian_data_C60.chk"

    with open(hamiltonian_chk, "rb") as f:
        hamiltonian_data = pickle.load(f)
    # """

    num_walkers = 1
    num_mcmc_warmup_steps = 0
    num_mcmc_bin_blocks = 50
    mcmc_seed = 34356

    # run VMC single-shot
    mcmc = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=2.0,
        mcmc_seed=mcmc_seed,
        epsilon_AS=1.0e-6,
        # adjust_epsilon_AS=False,
        num_walkers=num_walkers,
        comput_position_deriv=False,
        comput_param_deriv=False,
    )
    vmc = QMC(mcmc)
    vmc.run(num_mcmc_steps=100, max_time=3600)
    E_mean, E_std, Var_mean, Var_std = vmc.get_E(
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )
    logger.info(f"E = {E_mean} +- {E_std} Ha.")
    logger.info(f"Var = {Var_mean} +- {Var_std} Ha^2.")
    # """

    """
    f_mean, f_std = vmc.get_aF(
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )

    logger.info(f"f_mean = {f_mean} Ha/bohr.")
    logger.info(f"f_std = {f_std} Ha/bohr.")
    """
    #'''

    """
    hamiltonian_chk = "hamiltonian_data_water.chk"
    # hamiltonian_chk = "hamiltonian_data_water_methane.chk"
    # hamiltonian_chk = "hamiltonian_data_benzene.chk"
    # hamiltonian_chk = "hamiltonian_data_C60.chk"

    with open(hamiltonian_chk, "rb") as f:
        hamiltonian_data = pickle.load(f)

    num_walkers = 4
    mcmc_seed = 34356

    # run VMCopt
    mcmc = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=1.8,
        mcmc_seed=mcmc_seed,
        epsilon_AS=1.0e-3,
        # adjust_epsilon_AS=True,
        num_walkers=num_walkers,
        comput_position_deriv=False,
        comput_param_deriv=True,
    )
    vmc = QMC(mcmc)
    vmc.run_optimize(
        num_mcmc_steps=50,
        num_opt_steps=50,
        delta=1e-4,
        epsilon=1e-3,
        wf_dump_freq=10,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=50,
        opt_J1_param=False,
        opt_J2_param=True,
        opt_J3_param=True,
        opt_lambda_param=False,
    )
    """

    """
    # hamiltonian
    hamiltonian_chk = "hamiltonian_data_water.chk"
    # hamiltonian_chk = "hamiltonian_data_water_methane.chk"
    # hamiltonian_chk = "hamiltonian_data_benzene.chk"
    # hamiltonian_chk = "hamiltonian_data_C60.chk"

    with open(hamiltonian_chk, "rb") as f:
        hamiltonian_data = pickle.load(f)

    # GFMC param
    num_walkers = 4
    mcmc_seed = 3446
    E_scf = -17.00
    alat = 0.30
    num_mcmc_per_measurement = 50
    num_mcmc_bin_blocks = 20
    num_mcmc_warmup_steps = 10
    num_gfmc_collect_steps = 10
    non_local_move = "tmove"

    # run GFMC single-shot
    gfmc = GFMC_fixed_num_projection(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        num_mcmc_per_measurement=num_mcmc_per_measurement,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
        mcmc_seed=mcmc_seed,
        E_scf=E_scf,
        alat=alat,
        non_local_move=non_local_move,
    )
    gfmc = QMC(gfmc)
    gfmc.run(num_mcmc_steps=500, max_time=3600)
    E_mean, E_std, Var_mean, Var_std = gfmc.get_E(
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )
    logger.info(f"E = {E_mean} +- {E_std} Ha.")
    logger.info(f"Var E = {Var_mean} +- {Var_std} Ha.")
    """

    """
    f_mean, f_std = gfmc.get_aF(
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )

    logger.info(f"f_mean = {f_mean} Ha/bohr.")
    logger.info(f"f_std = {f_std} Ha/bohr.")
    """

    """
    # hamiltonian
    # hamiltonian_chk = "hamiltonian_data_water.chk"
    # hamiltonian_chk = "hamiltonian_data_water_methane.chk"
    # hamiltonian_chk = "hamiltonian_data_benzene.chk"
    hamiltonian_chk = "hamiltonian_data_C60.chk"

    with open(hamiltonian_chk, "rb") as f:
        hamiltonian_data = pickle.load(f)

    # GFMC param
    num_walkers = 4
    mcmc_seed = 3446
    tau = 0.05
    alat = 0.30
    num_mcmc_warmup_steps = 5
    num_mcmc_bin_blocks = 5
    num_gfmc_collect_steps = 2
    non_local_move = "tmove"

    # run GFMC single-shot
    gfmc = GFMC_fixed_projection_time(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
        tau=tau,
        alat=alat,
        non_local_move=non_local_move,
    )
    gfmc = QMC(gfmc)
    gfmc.run(num_mcmc_steps=50, max_time=3600)
    E_mean, E_std, Var_mean, Var_std = gfmc.get_E(
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )
    logger.info(f"E = {E_mean} +- {E_std} Ha.")
    logger.info(f"Var E = {Var_mean} +- {Var_std} Ha.")
    """
