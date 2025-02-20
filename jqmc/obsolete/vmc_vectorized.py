"""VMC module."""

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
import time
from functools import partial
from logging import getLogger

# JAX
import jax
import numpy as np
import numpy.typing as npt
import scipy
from jax import grad, jit, lax
from jax import grad, jit, lax, vmap
from jax import numpy as jnp
from mpi4py import MPI

from .hamiltonians import Hamiltonian_data, compute_local_energy_api

# jQMC module
from .jastrow_factor import Jastrow_data, Jastrow_three_body_data, Jastrow_two_body_data
from .structure import find_nearest_index_jax
from .swct import SWCT_data, evaluate_swct_domega_api, evaluate_swct_omega_api
from .wavefunction import Wavefunction_data, evaluate_ln_wavefunction_api, evaluate_wavefunction_api

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


class MCMC_multiple_walkers:
    """MCMC with multiple walker class.

    MCMC class. Runing MCMC with multiple walkers. The independent 'num_walkers' MCMCs are
    vectrized via the jax-vmap function.

    Args:
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
        num_walkers (int): the number of walkers.
        num_mcmc_per_measurement (int): the number of MCMC steps between a value (e.g., local energy) measurement.
        init_r_up_carts (npt.NDArray): starting electron positions for up electrons. dim. (num_walkers, N_e^up ,3)
        init_r_dn_carts (npt.NDArray): starting electron positions for dn electrons. dim. (num_walkers, N_e^dn ,3)
        Dt (float): electron move step (bohr)
        comput_jas_param_deriv (bool): if True, compute the derivatives of E wrt. Jastrow parameters.
        comput_position_deriv (bool): if True, compute the derivatives of E wrt. atomic positions.
        jax_PRNG_key (jax.Array): Jax PRIN key to handle random numbers.
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data = None,
        num_walkers: int = 40,
        num_mcmc_per_measurement: int = 16,
        init_r_up_carts: npt.NDArray[np.float64] = None,
        init_r_dn_carts: npt.NDArray[np.float64] = None,
        Dt: float = 2.0,
        comput_jas_param_deriv: bool = False,
        comput_position_deriv: bool = False,
        jax_PRNG_key_list: jax.Array = None,
    ) -> None:
        """Initialize a MCMC class, creating list holding results."""
        self.__hamiltonian_data = hamiltonian_data
        self.__num_walkers = num_walkers
        self.__num_mcmc_per_measurement = num_mcmc_per_measurement
        self.__Dt = Dt
        self.__jax_PRNG_key_list = jax_PRNG_key_list
        self.__comput_jas_param_deriv = comput_jas_param_deriv
        self.__comput_position_deriv = comput_position_deriv

        # timer
        self.__timer_mcmc_total = 0.0
        self.__timer_mcmc_init = 0.0
        self.__timer_mcmc_update_init = 0.0
        self.__timer_mcmc_update = 0.0
        self.__timer_e_L = 0.0
        self.__timer_de_L_dR_dr = 0.0
        self.__timer_dln_Psi_dR_dr = 0.0
        self.__timer_dln_Psi_dc_jas1b2b3b = 0.0
        self.__timer_misc = 0.0

        # latest electron positions
        self.__latest_r_up_carts = init_r_up_carts
        self.__latest_r_dn_carts = init_r_dn_carts

        # SWCT data
        self.__swct_data = SWCT_data(structure=self.__hamiltonian_data.structure_data)

        # compiling methods
        logger.info("Compilation of fundamental functions starts.")

        logger.info("  Compilation e_L starts.")
        start = time.perf_counter()
        _ = compute_local_energy_api(
            hamiltonian_data=self.__hamiltonian_data,
            r_up_carts=self.__latest_r_up_carts[0],
            r_dn_carts=self.__latest_r_dn_carts[0],
        )
        end = time.perf_counter()
        logger.info("  Compilation e_L is done.")
        logger.info(f"  Elapsed Time = {end - start:.2f} sec.")
        self.__timer_mcmc_init += end - start

        if self.__comput_position_deriv:
            logger.info("  Compilation de_L starts.")
            start = time.perf_counter()
            _, _, _ = grad(compute_local_energy_api, argnums=(0, 1, 2))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts[0],
                self.__latest_r_dn_carts[0],
            )
            end = time.perf_counter()
            logger.info("  Compilation de_L is done.")
            logger.info(f"  Elapsed Time = {end - start:.2f} sec.")
            self.__timer_mcmc_init += end - start

            logger.info("  Compilation dln_Psi starts.")
            start = time.perf_counter()
            _, _, _ = grad(evaluate_ln_wavefunction_api, argnums=(0, 1, 2))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts[0],
                self.__latest_r_dn_carts[0],
            )
            end = time.perf_counter()
            logger.info("  Compilation dln_Psi is done.")
            logger.info(f"  Elapsed Time = {end - start:.2f} sec.")
            self.__timer_mcmc_init += end - start

            logger.info("  Compilation domega starts.")
            start = time.perf_counter()
            _ = evaluate_swct_domega_api(
                self.__swct_data,
                self.__latest_r_up_carts[0],
            )
            end = time.perf_counter()
            logger.info("  Compilation domega is done.")
            logger.info(f"  Elapsed Time = {end - start:.2f} sec.")
            self.__timer_mcmc_init += end - start

        if self.__comput_jas_param_deriv:
            logger.info("  Compilation dln_Psi starts.")
            start = time.perf_counter()
            _ = grad(evaluate_ln_wavefunction_api, argnums=(0))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts[0],
                self.__latest_r_dn_carts[0],
            )
            end = time.perf_counter()
            logger.info("  Compilation dln_Psi is done.")
            logger.info(f"  Elapsed Time = {end - start:.2f} sec.")
            self.__timer_mcmc_init += end - start

        logger.info("Compilation of fundamental functions is done.")
        logger.info(f"Elapsed Time = {self.__timer_mcmc_init:.2f} sec.")
        logger.info("")

        # init attributes
        self.__init_attributes()

    def __init_attributes(self):
        # mcmc counter
        self.__mcmc_counter = 0

        # mcmc accepted/rejected moves
        self.__accepted_moves = 0
        self.__rejected_moves = 0

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

        # stored dln_Psi / dr_up
        self.__stored_grad_ln_Psi_r_up = []

        # stored dln_Psi / dc_jas2b
        self.__stored_grad_ln_Psi_jas2b = []

        # stored dln_Psi / dc_jas1b3b
        self.__stored_grad_ln_Psi_jas1b3b_j_matrix = []

        # total number of electrons
        self.__total_electrons = len(self.__latest_r_up_carts[0]) + len(self.__latest_r_dn_carts[0])

        # charges
        if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
            self.__charges = jnp.array(self.__hamiltonian_data.structure_data.atomic_numbers) - jnp.array(
                self.__hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            self.__charges = jnp.array(self.__hamiltonian_data.structure_data.atomic_numbers)

        # coords
        self.__coords = jnp.array(self.__hamiltonian_data.structure_data.positions_cart)

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
        timer_dln_Psi_dc_jas1b2b3b = 0.0
        mcmc_total_start = time.perf_counter()

        # MCMC electron position update function
        mcmc_update_init_start = time.perf_counter()
        logger.info("Start compilation of the MCMC_update funciton.")

        # Note: This jit drastically accelarates the computation!!
        @partial(jit, static_argnums=3)
        def _update_electron_positions(init_r_up_carts, init_r_dn_carts, jax_PRNG_key, num_mcmc_per_measurement):
            """Update electron positions based on the MH method.

            Args:
                init_r_up_carts (jnpt.ArrayLike): up electron position. dim: (N_e^up, 3)
                init_r_dn_carts (jnpt.ArrayLike): down electron position. dim: (N_e^dn, 3)
                jax_PRNG_key (jnpt.ArrayLike): jax PRIN key.
                self.__num_mcmc_per_measurement (int): the number of iterarations (i.e. the number of proposal in updating electron positions.)

            Returns:
                jax_PRNG_key (jnpt.ArrayLike): updated jax_PRNG_key.
                accepted_moves (int): the number of accepted moves
                rejected_moves (int): the number of rejected moves
                updated_r_up_cart (jnpt.ArrayLike): up electron position. dim: (N_e^up, 3)
                updated_r_dn_cart (jnpt.ArrayLike): down electron position. dim: (N_e^down, 3)
            """
            accepted_moves = 0
            rejected_moves = 0

            latest_r_up_carts = init_r_up_carts
            latest_r_dn_carts = init_r_dn_carts

            for _ in range(num_mcmc_per_measurement):
                # Choose randomly if the electron comes from up or dn
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                rand_num = jax.random.randint(subkey, shape=(), minval=0, maxval=self.__total_electrons)

                # boolen: "up" or "dn"
                # is_up == True -> up、False -> dn
                is_up = rand_num < len(latest_r_up_carts)

                # an index chosen from up electons
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                up_index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(latest_r_up_carts))

                # an index chosen from dn electrons
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                dn_index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(latest_r_dn_carts))

                selected_electron_index = jnp.where(is_up, up_index, dn_index)

                # choose an up or dn electron from old_r_cart
                old_r_cart = jnp.where(
                    is_up, latest_r_up_carts[selected_electron_index], latest_r_dn_carts[selected_electron_index]
                )

                # choose the nearest atom index
                nearest_atom_index = find_nearest_index_jax(self.__hamiltonian_data.structure_data, old_r_cart)

                R_cart = self.__coords[nearest_atom_index]
                Z = self.__charges[nearest_atom_index]
                norm_r_R = jnp.linalg.norm(old_r_cart - R_cart)
                f_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

                logger.debug(f"nearest_atom_index = {nearest_atom_index}")
                logger.debug(f"norm_r_R = {norm_r_R}")
                logger.debug(f"f_l  = {f_l}")

                sigma = f_l * self.__Dt
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                g = jax.random.normal(subkey, shape=()) * sigma

                # choose x,y,or,z
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                random_index = jax.random.randint(subkey, shape=(), minval=0, maxval=3)

                # plug g into g_vector
                g_vector = jnp.zeros(3)
                g_vector = g_vector.at[random_index].set(g)

                logger.debug(f"jn = {random_index}, g \\equiv dstep  = {g_vector}")
                new_r_cart = old_r_cart + g_vector

                # set proposed r_up_carts and r_dn_carts.
                proposed_r_up_carts = lax.cond(
                    is_up,
                    lambda _: latest_r_up_carts.at[selected_electron_index].set(new_r_cart),
                    lambda _: latest_r_up_carts,
                    operand=None,
                )

                proposed_r_dn_carts = lax.cond(
                    is_up,
                    lambda _: latest_r_dn_carts,
                    lambda _: latest_r_dn_carts.at[selected_electron_index].set(new_r_cart),
                    operand=None,
                )

                # choose the nearest atom index
                nearest_atom_index = find_nearest_index_jax(self.__hamiltonian_data.structure_data, new_r_cart)

                R_cart = self.__coords[nearest_atom_index]
                Z = self.__charges[nearest_atom_index]
                norm_r_R = jnp.linalg.norm(new_r_cart - R_cart)
                f_prime_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

                logger.debug(f"nearest_atom_index = {nearest_atom_index}")
                logger.debug(f"norm_r_R = {norm_r_R}")
                logger.debug(f"f_prime_l  = {f_prime_l}")

                logger.debug(f"The selected electron is {selected_electron_index + 1}-th {is_up} electron.")
                logger.debug(f"The selected electron position is {old_r_cart}.")
                logger.debug(f"The proposed electron position is {new_r_cart}.")

                T_ratio = (f_l / f_prime_l) * jnp.exp(
                    -(jnp.linalg.norm(new_r_cart - old_r_cart) ** 2)
                    * (1.0 / (2.0 * f_prime_l**2 * self.__Dt**2) - 1.0 / (2.0 * f_l**2 * self.__Dt**2))
                )

                R_ratio = (
                    evaluate_wavefunction_api(
                        wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                        r_up_carts=proposed_r_up_carts,
                        r_dn_carts=proposed_r_dn_carts,
                    )
                    / evaluate_wavefunction_api(
                        wavefunction_data=self.__hamiltonian_data.wavefunction_data,
                        r_up_carts=latest_r_up_carts,
                        r_dn_carts=latest_r_dn_carts,
                    )
                ) ** 2.0

                logger.debug(f"R_ratio, T_ratio = {R_ratio}, {T_ratio}")
                acceptance_ratio = jnp.min(jnp.array([1.0, R_ratio * T_ratio]))
                logger.debug(f"acceptance_ratio = {acceptance_ratio}")

                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                b = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)
                logger.debug(f"b = {b}.")

                def _accepted_fun(_):
                    # Move accepted
                    return (accepted_moves + 1, rejected_moves, proposed_r_up_carts, proposed_r_dn_carts)

                def _rejected_fun(_):
                    # Move rejected
                    return (accepted_moves, rejected_moves + 1, latest_r_up_carts, latest_r_dn_carts)

                # judge accept or reject the propsed move using jax.lax.cond
                accepted_moves, rejected_moves, latest_r_up_carts, latest_r_dn_carts = lax.cond(
                    b < acceptance_ratio, _accepted_fun, _rejected_fun, operand=None
                )

            return accepted_moves, rejected_moves, latest_r_up_carts, latest_r_dn_carts, jax_PRNG_key

        # MCMC update compilation.
        logger.info("  Compilation is in progress...")
        (
            _,
            _,
            _,
            _,
            _,
        ) = vmap(_update_electron_positions, in_axes=(0, 0, 0, None))(
            self.__latest_r_up_carts, self.__latest_r_dn_carts, self.__jax_PRNG_key_list, self.__num_mcmc_per_measurement
        )
        _ = vmap(compute_local_energy_api, in_axes=(None, 0, 0))(
            self.__hamiltonian_data,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
        )
        if self.__comput_position_deriv:
            _, _, _ = vmap(grad(compute_local_energy_api, argnums=(0, 1, 2)), in_axes=(None, 0, 0))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )

            _ = vmap(evaluate_ln_wavefunction_api, in_axes=(None, 0, 0))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )

            _, _, _ = vmap(grad(evaluate_ln_wavefunction_api, argnums=(0, 1, 2)), in_axes=(None, 0, 0))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )

            _ = vmap(evaluate_swct_omega_api, in_axes=(None, 0))(
                self.__swct_data,
                self.__latest_r_up_carts,
            )

            _ = vmap(evaluate_swct_omega_api, in_axes=(None, 0))(
                self.__swct_data,
                self.__latest_r_dn_carts,
            )

            _ = vmap(evaluate_swct_domega_api, in_axes=(None, 0))(
                self.__swct_data,
                self.__latest_r_up_carts,
            )

            _ = vmap(evaluate_swct_domega_api, in_axes=(None, 0))(
                self.__swct_data,
                self.__latest_r_dn_carts,
            )
            _ = vmap(grad(evaluate_ln_wavefunction_api, argnums=0), in_axes=(None, 0, 0))(
                self.__hamiltonian_data.wavefunction_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
        mcmc_update_init_end = time.perf_counter()
        timer_mcmc_update_init += mcmc_update_init_end - mcmc_update_init_start
        logger.info("End compilation of the MCMC_update funciton.")
        logger.info("")

        # MAIN MCMC loop from here !!!
        logger.info("Start MCMC")
        progress = (self.__mcmc_counter) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
        logger.info(f"  Progress: MCMC step= {self.__mcmc_counter}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.0f} %.")
        mcmc_interval = max(1, int(num_mcmc_steps / 10))  # %

        for i_mcmc_step in range(num_mcmc_steps):
            if (i_mcmc_step + 1) % mcmc_interval == 0:
                progress = (i_mcmc_step + self.__mcmc_counter + 1) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
                logger.info(
                    f"  Progress: MCMC step = {i_mcmc_step + self.__mcmc_counter + 1}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %."
                )

            # electron positions are goint to be updated!
            start = time.perf_counter()
            (
                accepted_moves_nw,
                rejected_moves_nw,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                self.__jax_PRNG_key_list,
            ) = vmap(_update_electron_positions, in_axes=(0, 0, 0, None))(
                self.__latest_r_up_carts, self.__latest_r_dn_carts, self.__jax_PRNG_key_list, self.__num_mcmc_per_measurement
            )
            end = time.perf_counter()
            timer_mcmc_update += end - start

            # store vmapped outcomes
            self.__accepted_moves += jnp.sum(accepted_moves_nw)
            self.__rejected_moves += jnp.sum(rejected_moves_nw)

            # evaluate observables
            start = time.perf_counter()
            e_L = vmap(compute_local_energy_api, in_axes=(None, 0, 0))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            logger.debug(f"e_L = {e_L}.")
            end = time.perf_counter()
            timer_e_L += end - start

            self.__stored_e_L.append(e_L)

            if self.__comput_position_deriv:
                # """
                start = time.perf_counter()
                grad_e_L_h, grad_e_L_r_up, grad_e_L_r_dn = vmap(
                    grad(compute_local_energy_api, argnums=(0, 1, 2)), in_axes=(None, 0, 0)
                )(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                end = time.perf_counter()
                timer_de_L_dR_dr += end - start

                self.__stored_grad_e_L_r_up.append(grad_e_L_r_up)
                self.__stored_grad_e_L_r_dn.append(grad_e_L_r_dn)

                grad_e_L_R = (
                    grad_e_L_h.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.structure_data.positions
                    + grad_e_L_h.wavefunction_data.geminal_data.orb_data_dn_spin.aos_data.structure_data.positions
                    + grad_e_L_h.coulomb_potential_data.structure_data.positions
                )

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_flag:
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

                start = time.perf_counter()
                ln_Psi = vmap(evaluate_ln_wavefunction_api, in_axes=(None, 0, 0))(
                    self.__hamiltonian_data.wavefunction_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                end = time.perf_counter()
                logger.devel(f"ln Psi evaluation: Time = {(end - start) * 1000:.3f} msec.")

                logger.devel(f"ln_Psi = {ln_Psi}")
                self.__stored_ln_Psi.append(ln_Psi)

                # """
                start = time.perf_counter()
                grad_ln_Psi_h, grad_ln_Psi_r_up, grad_ln_Psi_r_dn = vmap(
                    grad(evaluate_ln_wavefunction_api, argnums=(0, 1, 2)), in_axes=(None, 0, 0)
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
                    grad_ln_Psi_h.geminal_data.orb_data_up_spin.aos_data.structure_data.positions
                    + grad_ln_Psi_h.geminal_data.orb_data_dn_spin.aos_data.structure_data.positions
                )

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_flag:
                    grad_ln_Psi_dR += grad_ln_Psi_h.jastrow_data.jastrow_three_body_data.orb_data.structure_data.positions

                # stored dln_Psi / dR
                logger.devel(f"dln_Psi_dR = {grad_ln_Psi_dR}")
                self.__stored_grad_ln_Psi_dR.append(grad_ln_Psi_dR)
                # """

                omega_up = vmap(evaluate_swct_omega_api, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_up_carts,
                )

                omega_dn = vmap(evaluate_swct_omega_api, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_dn_carts,
                )

                logger.devel(f"omega_up = {omega_up}")
                logger.devel(f"omega_dn = {omega_dn}")

                self.__stored_omega_up.append(omega_up)
                self.__stored_omega_dn.append(omega_dn)

                grad_omega_dr_up = vmap(evaluate_swct_domega_api, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_up_carts,
                )

                grad_omega_dr_dn = vmap(evaluate_swct_domega_api, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_dn_carts,
                )

                logger.devel(f"grad_omega_dr_up = {grad_omega_dr_up}")
                logger.devel(f"grad_omega_dr_dn = {grad_omega_dr_dn}")

                self.__stored_grad_omega_r_up.append(grad_omega_dr_up)
                self.__stored_grad_omega_r_dn.append(grad_omega_dr_dn)

            if self.__comput_jas_param_deriv:
                start = time.perf_counter()
                grad_ln_Psi_h = vmap(grad(evaluate_ln_wavefunction_api, argnums=0), in_axes=(None, 0, 0))(
                    self.__hamiltonian_data.wavefunction_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                end = time.perf_counter()
                timer_dln_Psi_dc_jas1b2b3b += end - start

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_flag:
                    grad_ln_Psi_jas2b = grad_ln_Psi_h.jastrow_data.jastrow_two_body_data.jastrow_2b_param
                    logger.devel(f"grad_ln_Psi_jas2b.shape = {grad_ln_Psi_jas2b.shape}")
                    logger.devel(f"  grad_ln_Psi_jas2b = {grad_ln_Psi_jas2b}")
                    self.__stored_grad_ln_Psi_jas2b.append(grad_ln_Psi_jas2b)

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_flag:
                    grad_ln_Psi_jas1b3b_j_matrix = grad_ln_Psi_h.jastrow_data.jastrow_three_body_data.j_matrix
                    logger.devel(f"grad_ln_Psi_jas1b3b_j_matrix.shape={grad_ln_Psi_jas1b3b_j_matrix.shape}")
                    logger.devel(f"  grad_ln_Psi_jas1b3b_j_matrix = {grad_ln_Psi_jas1b3b_j_matrix}")
                    self.__stored_grad_ln_Psi_jas1b3b_j_matrix.append(grad_ln_Psi_jas1b3b_j_matrix)

            # check max time
            mcmc_current = time.perf_counter()
            if max_time < mcmc_current - mcmc_total_start:
                logger.info(f"max_time = {max_time} sec. exceeds.")
                logger.info("break the mcmc loop.")
                break

        logger.info("End MCMC")
        logger.info("")

        # count up the mcmc counter
        self.__mcmc_counter += i_mcmc_step + 1

        mcmc_total_end = time.perf_counter()
        timer_mcmc_total += mcmc_total_end - mcmc_total_start
        timer_misc = timer_mcmc_total - (
            timer_mcmc_update + timer_e_L + timer_de_L_dR_dr + timer_dln_Psi_dR_dr + timer_dln_Psi_dc_jas1b2b3b
        )

        self.__timer_mcmc_total += timer_mcmc_total
        self.__timer_mcmc_update_init += timer_mcmc_update_init
        self.__timer_mcmc_update += timer_mcmc_update
        self.__timer_e_L += timer_e_L
        self.__timer_de_L_dR_dr += timer_de_L_dR_dr
        self.__timer_dln_Psi_dR_dr += timer_dln_Psi_dR_dr
        self.__timer_dln_Psi_dc_jas1b2b3b += timer_dln_Psi_dc_jas1b2b3b
        self.__timer_misc += timer_misc

        logger.info(f"Total elapsed time for MCMC {num_mcmc_steps} steps. = {timer_mcmc_total:.2f} sec.")
        logger.info(f"Pre-compilation time for MCMC = {timer_mcmc_update_init:.2f} sec.")
        logger.info(f"Net total time for MCMC = {timer_mcmc_total - timer_mcmc_update_init:.2f} sec.")
        logger.info(f"Elapsed times per MCMC step, averaged over {num_mcmc_steps} steps.")
        logger.info(f"  Time for MCMC updated = {timer_mcmc_update / num_mcmc_steps * 10**3:.2f} msec.")
        logger.info(f"  Time for computing e_L = {timer_e_L / num_mcmc_steps * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/dR and de_L/dr = {timer_de_L_dR_dr / num_mcmc_steps * 10**3:.2f} msec.")
        logger.info(
            f"  Time for computing dln_Psi/dR and dln_Psi/dr = {timer_dln_Psi_dR_dr / num_mcmc_steps * 10**3:.2f} msec."
        )
        logger.info(
            f"  Time for computing dln_Psi/dc (jastrow 1b2b3b) = {timer_dln_Psi_dc_jas1b2b3b / num_mcmc_steps * 10**3:.2f} msec."
        )
        logger.info(f"  Time for misc. (others) = {timer_misc / num_mcmc_steps * 10**3:.2f} msec.")
        logger.info(
            f"Acceptance ratio is {self.__accepted_moves / (self.__accepted_moves + self.__rejected_moves) * 100:.3f} %"
        )
        logger.info("")

    @property
    def hamiltonian_data(self):
        """Return hamiltonian_data."""
        return self.__hamiltonian_data

    @hamiltonian_data.setter
    def hamiltonian_data(self, hamiltonian_data):
        """Set hamiltonian_data."""
        self.__hamiltonian_data = hamiltonian_data
        self.__init_attributes()

    @property
    def e_L(self) -> npt.NDArray:
        """Return the stored e_L array."""
        return np.array(self.__stored_e_L)

    @property
    def de_L_dR(self) -> npt.NDArray:
        """Return the stored de_L/dR array."""
        return np.array(self.__stored_grad_e_L_dR)

    @property
    def de_L_dr_up(self) -> npt.NDArray:
        """Return the stored de_L/dr_up array."""
        return np.array(self.__stored_grad_e_L_r_up)

    @property
    def de_L_dr_dn(self) -> npt.NDArray:
        """Return the stored de_L/dr_dn array."""
        return np.array(self.__stored_grad_e_L_r_dn)

    @property
    def dln_Psi_dr_up(self) -> npt.NDArray:
        """Return the stored dln_Psi/dr_up array."""
        return np.array(self.__stored_grad_ln_Psi_r_up)

    @property
    def dln_Psi_dr_dn(self) -> npt.NDArray:
        """Return the stored dln_Psi/dr_down array."""
        return np.array(self.__stored_grad_ln_Psi_r_dn)

    @property
    def dln_Psi_dR(self) -> npt.NDArray:
        """Return the stored dln_Psi/dR array."""
        return np.array(self.__stored_grad_ln_Psi_dR)

    @property
    def omega_up(self) -> npt.NDArray:
        """Return the stored Omega (for up electrons) array."""
        return np.array(self.__stored_omega_up)

    @property
    def omega_dn(self) -> npt.NDArray:
        """Return the stored Omega (for down electrons) array."""
        return np.array(self.__stored_omega_dn)

    @property
    def domega_dr_up(self) -> npt.NDArray:
        """Return the stored dOmega/dr_up array."""
        return np.array(self.__stored_grad_omega_r_up)

    @property
    def domega_dr_dn(self) -> npt.NDArray:
        """Return the stored dOmega/dr_dn array."""
        return np.array(self.__stored_grad_omega_r_dn)

    @property
    def dln_Psi_dc_jas_2b(self) -> npt.NDArray:
        """Return the stored dln_Psi/dc_J2 array."""
        return np.array(self.__stored_grad_ln_Psi_jas2b)

    @property
    def dln_Psi_dc_jas_1b3b(self) -> npt.NDArray:
        """Return the stored dln_Psi/dc_J1_3 array."""
        return np.array(self.__stored_grad_ln_Psi_jas1b3b_j_matrix)

    @property
    def latest_r_up_carts(self) -> npt.NDArray:
        """Latest updated electron position for up-spin."""
        return self.__latest_r_up_carts

    @property
    def latest_r_dn_carts(self) -> npt.NDArray:
        """Latest updated electron position for dn-spin."""
        return self.__latest_r_dn_carts

    @property
    def Dt(self) -> float:
        """Return Dt."""
        return self.__Dt

    @property
    def mcmc_counter(self) -> int:
        """Return current MCMC counter."""
        return self.__mcmc_counter

    @property
    def num_walkers(self):
        """The number of walkers."""
        return self.__num_walkers

    @property
    def timer_mcmc_init(self):
        """Return the measured elapsed time for MCMC initialization."""
        return self.__timer_mcmc_init

    @property
    def timer_mcmc_updated(self):
        """Return the measured elapsed time for MCMC update."""
        return self.__timer_mcmc_update

    @property
    def timer_e_L(self):
        """Return the measured elapsed time for computing e_L."""
        return self.__timer_e_L

    @property
    def timer_de_L_dR_dr(self):
        """Return the measured elapsed time for computing de_L/dR and de_L/dr."""
        return self.__timer_de_L_dR_dr

    @property
    def timer_dln_Psi_dR_dr(self):
        """Return the measured elapsed time for computing dln_Psi/dR and dln_Psi/dr."""
        return self.__timer_dln_Psi_dR_dr

    @property
    def timer_dln_Psi_dc_jas1b2b3b(self):
        """Return the measured elapsed time for computing dln_Psi/dc(jas1b,2b,3b)."""
        return self.__timer_dln_Psi_dc_jas1b2b3b

    @property
    def timer_misc(self):
        """Return the measured elapsed time for computing miscs."""
        return self.__timer_misc

    @property
    def opt_param_dict(self):
        """Return a dictionary containing information about variational parameters to be optimized.

        Refactoring in progress.

        Return:
            opt_param_list (list): labels of the parameters to be optimized.
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
            if self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_flag:
                opt_param = "jastrow_2b_param"
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
                opt_param = "j_matrix"
                dln_Psi_dc = self.dln_Psi_dc_jas_1b3b
                dln_Psi_dc_size = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix.size
                dln_Psi_dc_shape = self.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix.shape
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


class VMC_multiple_walkers:
    """VMC class.

    Runing VMC using MCMC.

    Args:
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data.
        mcmc_seed (int): random seed for generating initial electron positions.
        num_walkers (int): the number of walkers.
        Dt (float): electron move step (bohr)
        comput_jas_param_deriv (bool): if True, compute the derivatives of E wrt. Jastrow parameters.
        comput_position_deriv (bool): if True, compute the derivatives of E wrt. atomic positions.
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data = None,
        mcmc_seed: int = 34467,
        num_walkers: int = 40,
        Dt: float = 2.0,
        comput_jas_param_deriv=False,
        comput_position_deriv=False,
    ) -> None:
        """Initialization."""
        self.__mcmc_seed = mcmc_seed
        self.__mpi_seed = mcmc_seed * (mpi_rank + 1)
        self.__num_walkers = num_walkers
        self.__comput_jas_param_deriv = comput_jas_param_deriv
        self.__comput_position_deriv = comput_position_deriv

        logger.debug(f"mcmc_seed for MPI-rank={mpi_rank} is {self.__mpi_seed}.")

        # set JAX random seed
        self.__jax_PRNG_key = jax.random.PRNGKey(self.__mpi_seed)
        self.__jax_PRNG_key_list_init = jnp.array(
            [jax.random.fold_in(self.__jax_PRNG_key, nw) for nw in range(self.__num_walkers)]
        )

        # set the initial electron configurations
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

            r_carts_up = np.array(r_carts_up)
            r_carts_dn = np.array(r_carts_dn)

            r_carts_up_list.append(r_carts_up)
            r_carts_dn_list.append(r_carts_dn)

        init_r_up_carts = jnp.array(r_carts_up_list)
        init_r_dn_carts = jnp.array(r_carts_dn_list)

        logger.info(f"The number of MPI process = {mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self.__num_walkers}.")
        logger.debug(f"initial r_up_carts= {init_r_up_carts}")
        logger.debug(f"initial r_dn_carts = {init_r_dn_carts}")
        logger.debug(f"initial r_up_carts.shape = {init_r_up_carts.shape}")
        logger.debug(f"initial r_dn_carts.shape = {init_r_dn_carts.shape}")

        self.__mcmc = MCMC_multiple_walkers(
            hamiltonian_data=hamiltonian_data,
            init_r_up_carts=init_r_up_carts,
            init_r_dn_carts=init_r_dn_carts,
            num_walkers=self.__num_walkers,
            num_mcmc_per_measurement=16,
            Dt=Dt,
            comput_jas_param_deriv=self.__comput_jas_param_deriv,
            comput_position_deriv=self.__comput_position_deriv,
            jax_PRNG_key_list=self.__jax_PRNG_key_list_init,
        )

        # WF optimization counter
        self.__i_opt = 0

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
    def timer_mcmc_init(self):
        """Return the measured elapsed time for MCMC initialization."""
        return self.__mcmc.timer_mcmc_init

    @property
    def timer_mcmc_updated(self):
        """Return the measured elapsed time for MCMC update."""
        return self.__mcmc.timer_mcmc_updated

    @property
    def timer_e_L(self):
        """Return the measured elapsed time for computing e_L."""
        return self.__mcmc.timer_e_L

    @property
    def timer_de_L_dR_dr(self):
        """Return the measured elapsed time for computing de_L/dR and de_L/dr."""
        return self.__mcmc.timer_de_L_dR_dr

    @property
    def timer_dln_Psi_dR_dr(self):
        """Return the measured elapsed time for computing dln_Psi/dR and dln_Psi/dr."""
        return self.__mcmc.timer_dln_Psi_dR_dr

    @property
    def timer_dln_Psi_dc_jas1b2b3b(self):
        """Return the measured elapsed time for computing dln_Psi/dc(jas1b,2b,3b)."""
        return self.__mcmc.timer_dln_Psi_dc_jas1b2b3b

    @property
    def timer_misc(self):
        """Return the measured elapsed time for computing miscs."""
        return self.__mcmc.timer_misc

    def run_single_shot(self, num_mcmc_steps: int = 0, max_time: int = 86400) -> None:
        """Launch single-shot VMC.

        Args:
            num_mcmc_steps(int):
                The number of MCMC samples per walker.
            max_time(int):
                The maximum time (sec.) If maximum time exceeds,
                the method exits the MCMC loop.
        """
        self.__mcmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)

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

        """
        vmcopt_total_start = time.perf_counter()

        dln_Psi_dc_size_list = self.__mcmc.opt_param_dict["dln_Psi_dc_size_list"]
        logger.info(f"The number of variational paramers = {np.sum(dln_Psi_dc_size_list)}.")

        # main vmcopt loop
        for i_opt in range(num_opt_steps):
            logger.info(f"i_opt={i_opt + 1 + self.__i_opt}/{num_opt_steps + self.__i_opt}.")

            if mpi_rank == 0:
                logger.info(f"num_mcmc_warmup_steps={num_mcmc_warmup_steps}.")
                logger.info(f"num_mcmc_bin_blocks={num_mcmc_bin_blocks}.")
                logger.info(f"num_mcmc_steps={num_mcmc_steps}.")
                logger.info(f"Optimize Jastrow 1b2b3b={self.__comput_jas_param_deriv}")

            logger.info(
                f"twobody param before opt. = {self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param}"
            )

            # run MCMC
            self.__mcmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)

            # get e_L
            e_L, e_L_std = self.get_e_L(num_mcmc_warmup_steps=num_mcmc_warmup_steps, num_mcmc_bin_blocks=num_mcmc_bin_blocks)
            logger.info(f"e_L = {e_L} +- {e_L_std} Ha")

            # get f and f_std (generalized forces)
            f, f_std = self.get_generalized_forces(
                mpi_broadcast=False, num_mcmc_warmup_steps=num_mcmc_warmup_steps, num_mcmc_bin_blocks=num_mcmc_bin_blocks
            )
            # get S (preconditioning matrix)
            S, _ = self.get_stochastic_matrix(
                mpi_broadcast=False, num_mcmc_warmup_steps=num_mcmc_warmup_steps, num_mcmc_bin_blocks=num_mcmc_bin_blocks
            )

            if mpi_rank == 0:
                signal_to_noise_f = np.abs(f) / f_std
                logger.info(f"Max |f| = {np.max(np.abs(f)):.3f} Ha/a.u.")
                logger.debug(f"f_std of Max |f| = {f_std[np.argmax(np.abs(f))]:.3f} Ha/a.u.")
                logger.info(f"Max of signal-to-noise of f = max(|f|/|std f|) = {np.max(signal_to_noise_f):.3f}.")

            if mpi_rank == 0:
                logger.info("Computing the inverse of the stochastic matrix S^{-1}...")
                # logger.info(f"The matrix S_prime is symmetric? = {np.allclose(S_prime, S_prime.T, atol=1.0e-10)}")
                # logger.info(f"The condition number of the matrix S is {np.linalg.cond(S)}")
                # logger.info(f"The condition number of the matrix S_prime is {np.linalg.cond(S_prime)}")
                # SR
                if S.ndim != 0:
                    I = np.eye(S.shape[0])
                    S_prime = S + epsilon * I
                    # solve Sx=f
                    X = scipy.linalg.solve(S_prime, f, assume_a="sym")
                else:
                    I = 1.0
                    S_prime = S + epsilon * I
                    # solve Sx=f
                    X = 1.0 / S_prime * f
                # steepest decent (SD)
                # X = f

            else:
                X = None

            X = mpi_comm.bcast(X, root=0)
            logger.debug(f"X for MPI-rank={mpi_rank} is {X}")
            logger.debug(f"X.shape for MPI-rank={mpi_rank} is {X.shape}")
            logger.info(f"max(dX) for MPI-rank={mpi_rank} is {np.max(X)}")

            opt_param_list = self.__mcmc.opt_param_dict["opt_param_list"]
            dln_Psi_dc_shape_list = self.__mcmc.opt_param_dict["dln_Psi_dc_shape_list"]
            dln_Psi_dc_flattened_index_list = self.__mcmc.opt_param_dict["dln_Psi_dc_flattened_index_list"]

            jastrow_2b_param = (
                self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
            )
            jastrow_two_body_pade_flag = self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_flag
            jastrow_three_body_flag = self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_flag
            aos_data = self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
            j_matrix = self.__mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix

            for ii, opt_param in enumerate(opt_param_list):
                param_shape = dln_Psi_dc_shape_list[ii]
                param_index = [i for i, v in enumerate(dln_Psi_dc_flattened_index_list) if v == ii]
                dX = X[param_index].reshape(param_shape)
                logger.info(f"dX.shape for MPI-rank={mpi_rank} is {dX.shape}")
                if dX.shape == (1,):
                    dX = dX[0]
                if opt_param == "jastrow_2b_param":
                    jastrow_2b_param += delta * dX
                if opt_param == "j_matrix":
                    j_matrix += delta * dX

            structure_data = self.__mcmc.hamiltonian_data.structure_data
            coulomb_potential_data = self.__mcmc.hamiltonian_data.coulomb_potential_data
            geminal_data = self.__mcmc.hamiltonian_data.wavefunction_data.geminal_data
            jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=jastrow_2b_param)
            jastrow_three_body_data = Jastrow_three_body_data(
                orb_data=aos_data,
                j_matrix=j_matrix,
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

    def get_deriv_ln_WF(self, num_mcmc_warmup_steps: int = 50):
        """Return the derivativs of ln_WF wrt variational parameters.

        Args:
            num_mcmc_warmup_steps (int): The number of warmup steps.

        Return:
            O_matrix(npt.NDArray): The matrix containing O_k = d ln Psi / dc_k,
            where k is the flattened variational parameter index. The dimenstion
            of O_matrix is (M, nw, k), where M is the MCMC step and nw is the walker index.
        """
        opt_param_dict = self.__mcmc.opt_param_dict

        # opt_param_list = opt_param_dict["opt_param_list"]
        dln_Psi_dc_list = opt_param_dict["dln_Psi_dc_list"]
        # dln_Psi_dc_size_list = opt_param_dict["dln_Psi_dc_size_list"]
        # dln_Psi_dc_shape_list = opt_param_dict["dln_Psi_dc_shape_list"]
        # dln_Psi_dc_flattened_index_list = opt_param_dict["dln_Psi_dc_flattened_index_list"]

        # here, the thrid index indicates the flattened variational parameter index.
        O_matrix = np.empty((self.__mcmc.mcmc_counter, self.__num_walkers, 0))

        for dln_Psi_dc in dln_Psi_dc_list:
            logger.devel(f"dln_Psi_dc.shape={dln_Psi_dc.shape}.")
            if dln_Psi_dc.ndim == 2:  # i.e., sclar variational param.
                dln_Psi_dc_reshaped = dln_Psi_dc.reshape(dln_Psi_dc.shape[0], dln_Psi_dc.shape[1], 1)
            else:
                dln_Psi_dc_reshaped = dln_Psi_dc.reshape(
                    dln_Psi_dc.shape[0], dln_Psi_dc.shape[1], int(np.prod(dln_Psi_dc.shape[2:]))
                )
            O_matrix = np.concatenate((O_matrix, dln_Psi_dc_reshaped), axis=2)

        logger.debug(f"O_matrix.shape = {O_matrix.shape}")
        return O_matrix[num_mcmc_warmup_steps:]  # O.... (x....) (M, nw, L) matrix

    def get_generalized_forces(
        self, mpi_broadcast: bool = True, num_mcmc_warmup_steps: int = 50, num_mcmc_bin_blocks: int = 10
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Compute the derivatives of E wrt variational parameters, a.k.a. generalized forces.

        Args:
            mpi_broadcast (bool):
                If true, the computed S is shared among all MPI processes.
                If false, only the root node has it.
            num_mcmc_warmup_steps (int): The number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks

        Return:
            tuple[npt.NDArray, npt.NDArray]: mean and std of generalized forces.
            Dim. is 1D vector with L elements, where L is the number of flattened
            variational parameters.
        """
        logger.info("Computing the generalized force vector f...")
        e_L = self.__mcmc.e_L[num_mcmc_warmup_steps:]
        e_L_split = np.array_split(e_L, num_mcmc_bin_blocks, axis=0)
        e_L_binned = list(np.ravel([np.mean(arr, axis=0) for arr in e_L_split]))

        logger.debug(f"[before reduce] len(e_L_binned) for MPI-rank={mpi_rank} is {len(e_L_binned)}")

        e_L_binned = mpi_comm.reduce(e_L_binned, op=MPI.SUM, root=0)

        if mpi_rank == 0:
            logger.debug(f"[before reduce] len(e_L_binned) for MPI-rank={mpi_rank} is {len(e_L_binned)}")

        O_matrix = self.get_deriv_ln_WF(num_mcmc_warmup_steps=num_mcmc_warmup_steps)
        O_matrix_split = np.array_split(O_matrix, num_mcmc_bin_blocks, axis=0)
        O_matrix_ave = np.array([np.mean(arr, axis=0) for arr in O_matrix_split])
        O_matrix_binned_shape = (
            O_matrix_ave.shape[0] * O_matrix_ave.shape[1],
            O_matrix_ave.shape[2],
        )
        O_matrix_binned = list(O_matrix_ave.reshape(O_matrix_binned_shape))

        logger.debug(f"O_matrix.shape = {O_matrix.shape}")
        # logger.info(f"O_matrix_split.shape = {np.array(O_matrix_split).shape}")
        logger.debug(f"O_matrix_ave.shape = {O_matrix_ave.shape}")
        logger.debug(f"O_matrix_binned.shape [before reduce] = {np.array(O_matrix_binned).shape}")

        O_matrix_binned = mpi_comm.reduce(O_matrix_binned, op=MPI.SUM, root=0)

        eL_O_matrix = np.einsum("iw,iwj->iwj", e_L, O_matrix)
        eL_O_matrix_split = np.array_split(eL_O_matrix, num_mcmc_bin_blocks, axis=0)
        eL_O_matrix_ave = np.array([np.mean(arr, axis=0) for arr in eL_O_matrix_split])
        eL_O_matrix_binned_shape = (
            eL_O_matrix_ave.shape[0] * eL_O_matrix_ave.shape[1],
            eL_O_matrix_ave.shape[2],
        )
        eL_O_matrix_binned = list(eL_O_matrix_ave.reshape(eL_O_matrix_binned_shape))

        logger.debug(f"eL_O_matrix.shape = {eL_O_matrix.shape}")
        # logger.info(f"eL_O_matrix_split.shape = {np.array(eL_O_matrix_split).shape}")
        logger.debug(f"eL_O_matrix_ave.shape = {eL_O_matrix_ave.shape}")
        logger.debug(f"eL_O_matrix_binned.shape [before reduce] = {np.array(eL_O_matrix_binned).shape}")

        eL_O_matrix_binned = mpi_comm.reduce(eL_O_matrix_binned, op=MPI.SUM, root=0)

        if mpi_rank == 0:
            logger.debug(f"[after reduce] O_matrix_binned.shape = {np.array(O_matrix_binned).shape}")

            e_L_binned = np.array(e_L_binned)
            O_matrix_binned = np.array(O_matrix_binned)
            eL_O_matrix_binned = np.array(eL_O_matrix_binned)

            M = num_mcmc_bin_blocks * self.__mcmc.num_walkers * mpi_comm.size
            logger.info(f"Total number of binned samples = {M}")

            eL_O_jn = 1.0 / (M - 1) * np.array([np.sum(eL_O_matrix_binned, axis=0) - eL_O_matrix_binned[j] for j in range(M)])
            logger.debug(f"eL_O_jn = {eL_O_jn}")
            logger.debug(f"eL_O_jn.shape = {eL_O_jn.shape}")

            eL_jn = 1.0 / (M - 1) * np.array([np.sum(e_L_binned, axis=0) - e_L_binned[j] for j in range(M)])
            logger.debug(f"eL_jn = {eL_jn}")
            logger.debug(f"eL_jn.shape = {eL_jn.shape}")

            O_jn = 1.0 / (M - 1) * np.array([np.sum(O_matrix_binned, axis=0) - O_matrix_binned[j] for j in range(M)])

            logger.debug(f"O_jn = {O_jn}")
            logger.debug(f"O_jn.shape = {O_jn.shape}")

            eL_barO_jn = np.einsum("i,ij->ij", eL_jn, O_jn)

            logger.debug(f"eL_barO_jn = {eL_barO_jn}")
            logger.debug(f"eL_barO_jn.shape = {eL_barO_jn.shape}")

            generalized_force_mean = np.average(-2.0 * (eL_O_jn - eL_barO_jn), axis=0)
            generalized_force_std = np.sqrt(M - 1) * np.std(-2.0 * (eL_O_jn - eL_barO_jn), axis=0)

            logger.devel(f"generalized_force_mean = {generalized_force_mean}")
            logger.devel(f"generalized_force_std = {generalized_force_std}")

            logger.info(f"generalized_force_mean.shape = {generalized_force_mean.shape}")
            logger.info(f"generalized_force_std.shape = {generalized_force_std.shape}")

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

    def get_stochastic_matrix(
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
        O_matrix = self.get_deriv_ln_WF(num_mcmc_warmup_steps=num_mcmc_warmup_steps)
        O_matrix_split = np.array_split(O_matrix, num_mcmc_bin_blocks, axis=0)

        O_matrix_ave = np.array([np.mean(arr, axis=0) for arr in O_matrix_split])
        O_matrix_binned_shape = (
            O_matrix_ave.shape[0] * O_matrix_ave.shape[1],
            O_matrix_ave.shape[2],
        )
        O_matrix_binned = list(O_matrix_ave.reshape(O_matrix_binned_shape))

        O_matrix_binned = mpi_comm.reduce(O_matrix_binned, op=MPI.SUM, root=0)

        if mpi_rank == 0:
            O_matrix_binned = np.array(O_matrix_binned)
            logger.devel(f"O_matrix_binned = {O_matrix_binned}")
            logger.info(f"O_matrix_binned.shape = {O_matrix_binned.shape}")
            S_mean = np.array(np.cov(O_matrix_binned, bias=True, rowvar=False))
            S_std = np.zeros(S_mean.size)
            logger.devel(f"S_mean = {S_mean}")
            logger.info(f"S_mean.shape = {S_mean.shape}")
            logger.debug(f"S_mean.is_nan for MPI-rank={mpi_rank} is {np.isnan(S_mean).any()}")
            logger.debug(f"S_mean.shape for MPI-rank={mpi_rank} is {S_mean.shape}")
            logger.devel(f"S_mean.type for MPI-rank={mpi_rank} is {type(S_mean)}")
        else:
            S_mean = None
            S_std = None

        if mpi_broadcast:
            # comm.Bcast(S_mean, root=0)
            # comm.Bcast(S_std, root=0)
            S_mean = mpi_comm.bcast(S_mean, root=0)
            S_std = mpi_comm.bcast(S_std, root=0)

        return (S_mean, S_std)  # (S_mu,nu ...., var(S)_mu,nu....) (L*L matrix, L*L matrix)

    def get_e_L(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> tuple[float, float]:
        """Return the mean and std of the computed local energy.

        Args:
            num_mcmc_warmup_steps (int): the number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks

        Return:
            tuple[float, float]:
                The mean and std values of the computed local energy
                estimated by the Jackknife method with the Args.
        """
        # analysis VMC
        e_L = self.__mcmc.e_L[num_mcmc_warmup_steps:]
        e_L_split = np.array_split(e_L, num_mcmc_bin_blocks, axis=0)
        e_L_binned = list(np.ravel([np.mean(arr, axis=0) for arr in e_L_split]))
        # e_L_binned_check = list(np.ravel([np.average(e_arr, axis=0) for e_arr in e_L_split]))
        # logger.info(e_L_binned)
        # logger.info(e_L_binned_check)

        logger.debug(f"[before reduce] len(e_L_binned) for MPI-rank={mpi_rank} is {len(e_L_binned)}.")

        e_L_binned = mpi_comm.reduce(e_L_binned, op=MPI.SUM, root=0)

        if mpi_rank == 0:
            logger.debug(f"[after reduce] len(e_L_binned) for MPI-rank={mpi_rank} is {len(e_L_binned)}.")
            logger.devel(f"e_L_binned = {e_L_binned}.")
            # jackknife implementation
            # https://www2.yukawa.kyoto-u.ac.jp/~etsuko.itou/old-HP/Notes/Jackknife-method.pdf
            e_L_jackknife_binned = [np.average(np.delete(e_L_binned, i)) for i in range(len(e_L_binned))]

            logger.debug(f"len(e_L_jackknife_binned)  = {len(e_L_jackknife_binned)}.")

            e_L_mean = np.average(e_L_jackknife_binned)
            e_L_std = np.sqrt(len(e_L_binned) - 1) * np.std(e_L_jackknife_binned)

            logger.debug(f"e_L = {e_L_mean} +- {e_L_std} Ha.")
        else:
            e_L_mean = 0.0
            e_L_std = 0.0

        e_L_mean = mpi_comm.bcast(e_L_mean, root=0)
        e_L_std = mpi_comm.bcast(e_L_std, root=0)

        return (e_L_mean, e_L_std)

    def get_atomic_forces(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ):
        """Return the mean and std of the computed local energy.

        Args:
            num_mcmc_warmup_steps (int): the number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks

        Return:
            tuple[npt.NDArray, npt.NDArray]:
                The mean and std values of the computed atomic forces
                estimated by the Jackknife method with the Args.
                The dimention of the arrays is (N, 3).
        """
        if not self.__comput_position_deriv:
            force_mean = np.array([])
            force_std = np.array([])
            return (force_mean, force_std)

        else:
            e_L = self.__mcmc.e_L[num_mcmc_warmup_steps:]
            de_L_dR = self.__mcmc.de_L_dR[num_mcmc_warmup_steps:]
            de_L_dr_up = self.__mcmc.de_L_dr_up[num_mcmc_warmup_steps:]
            de_L_dr_dn = self.__mcmc.de_L_dr_dn[num_mcmc_warmup_steps:]
            dln_Psi_dr_up = self.__mcmc.dln_Psi_dr_up[num_mcmc_warmup_steps:]
            dln_Psi_dr_dn = self.__mcmc.dln_Psi_dr_dn[num_mcmc_warmup_steps:]
            dln_Psi_dR = self.__mcmc.dln_Psi_dR[num_mcmc_warmup_steps:]
            omega_up = self.__mcmc.omega_up[num_mcmc_warmup_steps:]
            omega_dn = self.__mcmc.omega_dn[num_mcmc_warmup_steps:]
            domega_dr_up = self.__mcmc.domega_dr_up[num_mcmc_warmup_steps:]
            domega_dr_dn = self.__mcmc.domega_dr_dn[num_mcmc_warmup_steps:]

            logger.info(f"e_L.shape for MPI-rank={mpi_rank} is {e_L.shape}")

            logger.info(f"de_L_dR.shape for MPI-rank={mpi_rank} is {de_L_dR.shape}")
            logger.info(f"de_L_dr_up.shape for MPI-rank={mpi_rank} is {de_L_dr_up.shape}")
            logger.info(f"de_L_dr_dn.shape for MPI-rank={mpi_rank} is {de_L_dr_dn.shape}")

            logger.info(f"dln_Psi_dr_up.shape for MPI-rank={mpi_rank} is {dln_Psi_dr_up.shape}")
            logger.info(f"dln_Psi_dr_dn.shape for MPI-rank={mpi_rank} is {dln_Psi_dr_dn.shape}")
            logger.info(f"dln_Psi_dR.shape for MPI-rank={mpi_rank} is {dln_Psi_dR.shape}")

            logger.info(f"omega_up.shape for MPI-rank={mpi_rank} is {omega_up.shape}")
            logger.info(f"omega_dn.shape for MPI-rank={mpi_rank} is {omega_dn.shape}")
            logger.info(f"domega_dr_up.shape for MPI-rank={mpi_rank} is {domega_dr_up.shape}")
            logger.info(f"domega_dr_dn.shape for MPI-rank={mpi_rank} is {domega_dr_dn.shape}")

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

            logger.info(f"e_L.shape for MPI-rank={mpi_rank} is {e_L.shape}")
            logger.info(f"force_HF.shape for MPI-rank={mpi_rank} is {force_HF.shape}")
            logger.info(f"force_PP.shape for MPI-rank={mpi_rank} is {force_PP.shape}")
            logger.info(f"E_L_force_PP.shape for MPI-rank={mpi_rank} is {E_L_force_PP.shape}")

            # split and binning with multiple walkers
            e_L_split = np.array_split(e_L, num_mcmc_bin_blocks, axis=0)
            force_HF_split = np.array_split(force_HF, num_mcmc_bin_blocks, axis=0)
            force_PP_split = np.array_split(force_PP, num_mcmc_bin_blocks, axis=0)
            E_L_force_PP_split = np.array_split(E_L_force_PP, num_mcmc_bin_blocks, axis=0)

            e_L_binned = list(np.ravel(np.average(np.stack(e_L_split), axis=1)))

            force_HF_ave = np.array([np.mean(arr, axis=0) for arr in force_HF_split])
            force_HF_binned_shape = (
                force_HF_ave.shape[0] * force_HF_ave.shape[1],
                force_HF_ave.shape[2],
                force_HF_ave.shape[3],
            )
            force_HF_binned = list(force_HF_ave.reshape(force_HF_binned_shape))

            force_PP_ave = np.array([np.mean(arr, axis=0) for arr in force_PP_split])
            force_PP_binned_shape = (
                force_PP_ave.shape[0] * force_PP_ave.shape[1],
                force_PP_ave.shape[2],
                force_PP_ave.shape[3],
            )
            force_PP_binned = list(force_PP_ave.reshape(force_PP_binned_shape))

            E_L_force_PP_ave = np.array([np.mean(arr, axis=0) for arr in E_L_force_PP_split])
            E_L_force_PP_binned_shape = (
                E_L_force_PP_ave.shape[0] * E_L_force_PP_ave.shape[1],
                E_L_force_PP_ave.shape[2],
                E_L_force_PP_ave.shape[3],
            )
            E_L_force_PP_binned = list(E_L_force_PP_ave.reshape(E_L_force_PP_binned_shape))

            # MPI reduce
            e_L_binned = mpi_comm.reduce(e_L_binned, op=MPI.SUM, root=0)
            force_HF_binned = mpi_comm.reduce(force_HF_binned, op=MPI.SUM, root=0)
            force_PP_binned = mpi_comm.reduce(force_PP_binned, op=MPI.SUM, root=0)
            E_L_force_PP_binned = mpi_comm.reduce(E_L_force_PP_binned, op=MPI.SUM, root=0)

            if mpi_rank == 0:
                e_L_binned = np.array(e_L_binned)
                force_HF_binned = np.array(force_HF_binned)
                force_PP_binned = np.array(force_PP_binned)
                E_L_force_PP_binned = np.array(E_L_force_PP_binned)

                logger.info(f"e_L_binned.shape for MPI-rank={mpi_rank} is {e_L_binned.shape}")
                logger.info(f"force_HF_binned.shape for MPI-rank={mpi_rank} is {force_HF_binned.shape}")
                logger.info(f"force_PP_binned.shape for MPI-rank={mpi_rank} is {force_PP_binned.shape}")
                logger.info(f"E_L_force_PP_binned.shape for MPI-rank={mpi_rank} is {E_L_force_PP_binned.shape}")

                M = num_mcmc_bin_blocks * self.__mcmc.num_walkers * mpi_comm.size
                logger.info(f"Total number of binned samples = {M}")

                force_HF_jn = np.array(
                    [-1.0 / (M - 1) * (np.sum(force_HF_binned, axis=0) - force_HF_binned[j]) for j in range(M)]
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

                logger.info(f"force_HF_jn.shape for MPI-rank={mpi_rank} is {force_HF_jn.shape}")
                logger.info(f"force_Pulay_jn.shape for MPI-rank={mpi_rank} is {force_Pulay_jn.shape}")

                force_jn = force_HF_jn + force_Pulay_jn

                force_mean = np.average(force_jn, axis=0)
                force_std = np.sqrt(M - 1) * np.std(force_jn, axis=0)

                logger.info(f"force_mean.shape  = {force_mean.shape}.")
                logger.info(f"force_std.shape  = {force_std.shape}.")

                logger.devel(f"force = {force_mean} +- {force_std} Ha.")

            else:
                force_mean = np.array([])
                force_std = np.array([])

            force_mean = mpi_comm.bcast(force_mean, root=0)
            force_std = mpi_comm.bcast(force_std, root=0)

            return (force_mean, force_std)


if __name__ == "__main__":
    import pickle
    from logging import Formatter, StreamHandler, getLogger

    # import os
    # from .trexio_wrapper import read_trexio_file

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
    """

    """
    # H2 dimer cc-pV5Z with Mitas ccECP (2 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "H2_dimer_ccpv5z_trexio.hdf5"))
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

    """
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.75)
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

    # VMC parameters
    num_walkers = 1
    num_mcmc_warmup_steps = 5
    num_mcmc_bin_blocks = 5
    mcmc_seed = 34356

    # """
    # run VMC single-shot
    vmc = VMC_multiple_walkers(
        hamiltonian_data=hamiltonian_data,
        Dt=2.0,
        mcmc_seed=mcmc_seed,
        num_walkers=num_walkers,
        comput_position_deriv=True,
        comput_jas_param_deriv=False,
    )
    vmc.run_single_shot(num_mcmc_steps=100)
    e_L_mean, e_L_std = vmc.get_e_L(
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )
    logger.info(f"e_L = {e_L_mean} +- {e_L_std} Ha.")
    # """

    # """
    f_mean, f_std = vmc.get_atomic_forces(
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )

    logger.info(f"f_mean = {f_mean} Ha/bohr.")
    logger.info(f"f_std = {f_std} Ha/bohr.")
    # """

    """
    # run VMCopt
    vmc = VMC_multiple_walkers(
        hamiltonian_data=hamiltonian_data,
        Dt=2.0,
        mcmc_seed=mcmc_seed,
        num_walkers=num_walkers,
        comput_position_deriv=False,
        comput_jas_param_deriv=True,
    )
    vmc.run_optimize(
        num_mcmc_steps=50,
        num_opt_steps=20,
        delta=0.002,
        epsilon=0.001,
        wf_dump_freq=1,
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )
    """
