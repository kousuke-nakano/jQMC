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
# * Neither the name of the jqmc project nor the names of its
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

from logging import getLogger

import jax
import numpy as np
import numpy.typing as npt
from jax import grad, vmap
from jax import numpy as jnp
from mpi4py import MPI

from .determinant import compute_AS_regularization_factor_jax, compute_det_geminal_all_elements_jax
from .hamiltonians import (
    Hamiltonian_data,
    compute_local_energy_jax,
)
from .jastrow_factor import (
    compute_Jastrow_part_jax,
)
from .structure import find_nearest_index_jax
from .swct import SWCT_data, evaluate_swct_domega_jax, evaluate_swct_omega_jax
from .wavefunction import (
    evaluate_ln_wavefunction_jax,
)

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# MPI related
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


class MCMC_debug:
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
        comput_param_deriv: bool = False,
        comput_position_deriv: bool = False,
        random_discretized_mesh: bool = True,
    ) -> None:
        """Initialize a MCMC class, creating list holding results."""
        self.__mcmc_seed = mcmc_seed
        self.__num_walkers = num_walkers
        self.__num_mcmc_per_measurement = num_mcmc_per_measurement
        self.__Dt = Dt
        self.__epsilon_AS = epsilon_AS
        self.__comput_param_deriv = comput_param_deriv
        self.__comput_position_deriv = comput_position_deriv
        self.__random_discretized_mesh = random_discretized_mesh

        # set hamiltonian_data
        self.__hamiltonian_data = hamiltonian_data

        # seeds
        self.__mpi_seed = self.__mcmc_seed * (mpi_rank + 1)
        self.__jax_PRNG_key = jax.random.PRNGKey(self.__mpi_seed)
        self.__jax_PRNG_key_list = jnp.array([jax.random.fold_in(self.__jax_PRNG_key, nw) for nw in range(self.__num_walkers)])

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

            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
                    hamiltonian_data.coulomb_potential_data.z_cores
                )
            else:
                charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

            logger.devel(f"charges = {charges}.")
            coords = hamiltonian_data.structure_data.positions_cart_jnp

            # Place electrons for each atom
            # 1) Convert each atomic charge to an integer electron count
            n_i_list = [int(round(charge)) for charge in charges]

            # 2) Determine the base number of paired electrons for each atom: floor(n_i/2)
            base_up_list = [n_i // 2 for n_i in n_i_list]
            base_dn_list = base_up_list.copy()

            # 3) If an atom has an odd number of electrons, assign the leftover one to up-spin
            leftover_list = [n_i - 2 * base for n_i, base in zip(n_i_list, base_up_list)]
            # leftover_i is either 0 or 1
            base_up_list = [u + o for u, o in zip(base_up_list, leftover_list)]

            # 4) Compute the current totals of up and down electrons
            base_up_sum = sum(base_up_list)
            # base_dn_sum = sum(base_dn_list)

            # 5) Compute how many extra up/down electrons are needed to reach the target totals
            extra_up = tot_num_electron_up - base_up_sum  # positive → need more up; negative → need more down

            # 6) Initialize final per-atom assignment lists
            assign_up = base_up_list.copy()
            assign_dn = base_dn_list.copy()

            # 7) Distribute extra up-spin electrons in a round-robin fashion if extra_up > 0
            if extra_up > 0:
                # Prefer atoms that currently have at least one down-spin electron; fall back to all atoms
                eligible = [i for i, dn in enumerate(assign_dn) if dn > 0]
                if not eligible:
                    eligible = list(range(len(coords)))
                for k in range(extra_up):
                    atom = eligible[k % len(eligible)]
                    assign_up[atom] += 1
                    assign_dn[atom] -= 1

            # 8) Distribute extra down-spin electrons in a round-robin fashion if extra_up < 0
            elif extra_up < 0:
                # Now extra_dn = -extra_up > 0
                eligible = [i for i, up in enumerate(assign_up) if up > 0]
                if not eligible:
                    eligible = list(range(len(coords)))
                for k in range(-extra_up):
                    atom = eligible[k % len(eligible)]
                    assign_up[atom] -= 1
                    assign_dn[atom] += 1

            # 9) Recompute totals and log them
            total_assigned_up = sum(assign_up)
            total_assigned_dn = sum(assign_dn)

            # 10) Random placement of electrons using assign_up and assign_dn
            for i, (x, y, z) in enumerate(coords):
                # Place up-spin electrons for atom i
                for _ in range(assign_up[i]):
                    distance = np.random.uniform(0.1, 1.0)
                    theta = np.random.uniform(0, np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    dx = distance * np.sin(theta) * np.cos(phi)
                    dy = distance * np.sin(theta) * np.sin(phi)
                    dz = distance * np.cos(theta)
                    r_carts_up.append([x + dx, y + dy, z + dz])

                # Place down-spin electrons for atom i
                for _ in range(assign_dn[i]):
                    distance = np.random.uniform(0.1, 1.0)
                    theta = np.random.uniform(0, np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    dx = distance * np.sin(theta) * np.cos(phi)
                    dy = distance * np.sin(theta) * np.sin(phi)
                    dz = distance * np.cos(theta)
                    r_carts_dn.append([x + dx, y + dy, z + dz])

            # Electron assignment for all atoms is complete
            logger.info(f"Total assigned up electrons: {total_assigned_up} (target {tot_num_electron_up}).")
            logger.info(f"Total assigned dn electrons: {total_assigned_dn} (target {tot_num_electron_dn}).")
            logger.info("")

            # If necessary, include a check/adjustment step to ensure the overall assignment matches the targets
            # (Here it is assumed that sum(round(charge)) equals tot_num_electron_up + tot_num_electron_dn)

            r_carts_up_list.append(r_carts_up)
            r_carts_dn_list.append(r_carts_dn)

        self.__latest_r_up_carts = jnp.array(r_carts_up_list)
        self.__latest_r_dn_carts = jnp.array(r_carts_dn_list)

        logger.info(f"The number of MPI process = {mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self.__num_walkers}.")
        logger.info("")

        # SWCT data
        self.__swct_data = SWCT_data(structure=self.__hamiltonian_data.structure_data)

        # init_attributes
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

        # stored dln_Psi / dc_lambda_matrix
        self.__stored_grad_ln_Psi_lambda_matrix = []

    def run(self, num_mcmc_steps: int = 0) -> None:
        """Launch MCMCs with the set multiple walkers.

        Args:
            num_mcmc_steps (int): the number of total mcmc steps per walker.
            max_time(int):
                Max elapsed time (sec.). If the elapsed time exceeds max_time, the methods exits the mcmc loop.
        """
        # MAIN MCMC loop from here !!!
        logger.info("Start MCMC")
        num_mcmc_done = 0
        progress = (self.__mcmc_counter) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
        logger.info(f"  Progress: MCMC step= {self.__mcmc_counter}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.0f} %.")
        mcmc_interval = max(1, int(num_mcmc_steps / 10))  # %

        for i_mcmc_step in range(num_mcmc_steps):
            if (i_mcmc_step + 1) % mcmc_interval == 0:
                progress = (i_mcmc_step + self.__mcmc_counter + 1) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
                logger.info(
                    f"  Progress: MCMC step = {i_mcmc_step + self.__mcmc_counter + 1}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %"
                )

            accepted_moves_nw = np.zeros(self.__num_walkers, dtype=np.int32)
            rejected_moves_nw = np.zeros(self.__num_walkers, dtype=np.int32)
            latest_r_up_carts = np.array(self.__latest_r_up_carts)
            latest_r_dn_carts = np.array(self.__latest_r_dn_carts)
            jax_PRNG_key_list = np.array(self.__jax_PRNG_key_list)

            for i_walker in range(self.__num_walkers):
                accepted_moves = 0
                rejected_moves = 0
                r_up_carts = latest_r_up_carts[i_walker]
                r_dn_carts = latest_r_dn_carts[i_walker]
                jax_PRNG_key = jax_PRNG_key_list[i_walker]

                num_mcmc_per_measurement = self.__num_mcmc_per_measurement
                hamiltonian_data = self.__hamiltonian_data
                Dt = self.__Dt
                epsilon_AS = self.__epsilon_AS

                for _ in range(num_mcmc_per_measurement):
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

                    if is_up:
                        selected_electron_index = up_index
                        old_r_cart = r_up_carts[selected_electron_index]
                    else:
                        selected_electron_index = dn_index
                        old_r_cart = r_dn_carts[selected_electron_index]

                    # choose the nearest atom index
                    nearest_atom_index = find_nearest_index_jax(hamiltonian_data.structure_data, old_r_cart)

                    # charges
                    if hamiltonian_data.coulomb_potential_data.ecp_flag:
                        charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - jnp.array(
                            hamiltonian_data.coulomb_potential_data.z_cores
                        )
                    else:
                        charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

                    # coords
                    coords = hamiltonian_data.structure_data.positions_cart_np

                    R_cart = coords[nearest_atom_index]
                    Z = charges[nearest_atom_index]
                    norm_r_R = np.linalg.norm(old_r_cart - R_cart)
                    f_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

                    sigma = f_l * Dt
                    jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                    g = jax.random.normal(subkey, shape=()) * sigma

                    # choose x,y,or,z
                    jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                    random_index = jax.random.randint(subkey, shape=(), minval=0, maxval=3)

                    # plug g into g_vector
                    g_vector = np.zeros(3)
                    g_vector[random_index] = g

                    new_r_cart = old_r_cart + g_vector

                    # set proposed r_up_carts and r_dn_carts.
                    if is_up:
                        proposed_r_up_carts = r_up_carts.copy()
                        proposed_r_up_carts[selected_electron_index] = new_r_cart
                        proposed_r_dn_carts = r_dn_carts
                    else:
                        proposed_r_up_carts = r_up_carts
                        proposed_r_dn_carts = r_dn_carts.copy()
                        proposed_r_dn_carts[selected_electron_index] = new_r_cart

                    # choose the nearest atom index
                    nearest_atom_index = find_nearest_index_jax(hamiltonian_data.structure_data, new_r_cart)

                    R_cart = coords[nearest_atom_index]
                    Z = charges[nearest_atom_index]
                    norm_r_R = np.linalg.norm(new_r_cart - R_cart)
                    f_prime_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

                    T_ratio = (f_l / f_prime_l) * jnp.exp(
                        -(np.linalg.norm(new_r_cart - old_r_cart) ** 2)
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
                    WF_ratio = np.exp(Jastrow_T_p - Jastrow_T_o) * (Det_T_p / Det_T_o)

                    # compute R_ratio
                    R_ratio = (R_AS_ratio * WF_ratio) ** 2.0

                    logger.devel(f"R_ratio, T_ratio = {R_ratio}, {T_ratio}")
                    acceptance_ratio = np.min(jnp.array([1.0, R_ratio * T_ratio]))
                    logger.devel(f"acceptance_ratio = {acceptance_ratio}")

                    jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                    b = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)

                    if b < acceptance_ratio:
                        accepted_moves += 1
                        r_up_carts = proposed_r_up_carts
                        r_dn_carts = proposed_r_dn_carts
                    else:
                        rejected_moves += 1

                    accepted_moves_nw[i_walker] = accepted_moves
                    rejected_moves_nw[i_walker] = rejected_moves
                    latest_r_up_carts[i_walker] = r_up_carts
                    latest_r_dn_carts[i_walker] = r_dn_carts
                    jax_PRNG_key_list[i_walker] = jax_PRNG_key

            # store vmapped outcomes
            self.__accepted_moves += jnp.sum(accepted_moves_nw)
            self.__rejected_moves += jnp.sum(rejected_moves_nw)
            self.__latest_r_up_carts = jnp.array(latest_r_up_carts)
            self.__latest_r_dn_carts = jnp.array(latest_r_dn_carts)
            self.__jax_PRNG_key_list = jnp.array(jax_PRNG_key_list)

            # generate rotation matrices (for non-local ECPs)
            RTs = []
            for jax_PRNG_key in self.__jax_PRNG_key_list:
                if self.__random_discretized_mesh:
                    # key -> (new_key, subkey)
                    _, subkey = jax.random.split(jax_PRNG_key)
                    # sampling angles
                    alpha, beta, gamma = jax.random.uniform(subkey, shape=(3,), minval=-2 * jnp.pi, maxval=2 * jnp.pi)
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
                    RTs.append(R.T)
                else:
                    RTs.append(jnp.eye(3))
            RTs = jnp.array(RTs)

            # evaluate observables
            e_L = vmap(compute_local_energy_jax, in_axes=(None, 0, 0, 0))(
                self.__hamiltonian_data, self.__latest_r_up_carts, self.__latest_r_dn_carts, RTs
            )
            self.__stored_e_L.append(e_L)
            self.__stored_e_L2.append(e_L**2)

            # compute AS regularization factors, R_AS and R_AS_eps
            R_AS = vmap(compute_AS_regularization_factor_jax, in_axes=(None, 0, 0))(
                self.__hamiltonian_data.wavefunction_data.geminal_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            R_AS_eps = jnp.maximum(R_AS, self.__epsilon_AS)

            w_L = (R_AS / R_AS_eps) ** 2
            self.__stored_w_L.append(w_L)

            if self.__comput_position_deriv:
                grad_e_L_h, grad_e_L_r_up, grad_e_L_r_dn = vmap(
                    grad(compute_local_energy_jax, argnums=(0, 1, 2)), in_axes=(None, 0, 0, 0)
                )(self.__hamiltonian_data, self.__latest_r_up_carts, self.__latest_r_dn_carts, RTs)

                self.__stored_grad_e_L_r_up.append(grad_e_L_r_up)
                self.__stored_grad_e_L_r_dn.append(grad_e_L_r_dn)

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

                grad_ln_Psi_h, grad_ln_Psi_r_up, grad_ln_Psi_r_dn = vmap(
                    grad(evaluate_ln_wavefunction_jax, argnums=(0, 1, 2)), in_axes=(None, 0, 0)
                )(
                    self.__hamiltonian_data.wavefunction_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )

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

                self.__stored_grad_ln_Psi_dR.append(grad_ln_Psi_dR)

                omega_up = vmap(evaluate_swct_omega_jax, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_up_carts,
                )

                omega_dn = vmap(evaluate_swct_omega_jax, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_dn_carts,
                )

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

                self.__stored_grad_omega_r_up.append(grad_omega_dr_up)
                self.__stored_grad_omega_r_dn.append(grad_omega_dr_dn)

            if self.__comput_param_deriv:
                grad_ln_Psi_h = vmap(grad(evaluate_ln_wavefunction_jax, argnums=0), in_axes=(None, 0, 0))(
                    self.__hamiltonian_data.wavefunction_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )

                # 1b Jastrow
                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_one_body_data is not None:
                    grad_ln_Psi_jas1b = grad_ln_Psi_h.jastrow_data.jastrow_one_body_data.jastrow_1b_param
                    self.__stored_grad_ln_Psi_jas1b.append(grad_ln_Psi_jas1b)

                # 2b Jastrow
                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data is not None:
                    grad_ln_Psi_jas2b = grad_ln_Psi_h.jastrow_data.jastrow_two_body_data.jastrow_2b_param
                    self.__stored_grad_ln_Psi_jas2b.append(grad_ln_Psi_jas2b)

                # 3b Jastrow
                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data is not None:
                    grad_ln_Psi_jas1b3b_j_matrix = grad_ln_Psi_h.jastrow_data.jastrow_three_body_data.j_matrix
                    self.__stored_grad_ln_Psi_jas1b3b_j_matrix.append(grad_ln_Psi_jas1b3b_j_matrix)

                # lambda_matrix
                grad_ln_Psi_lambda_matrix = grad_ln_Psi_h.geminal_data.lambda_matrix
                self.__stored_grad_ln_Psi_lambda_matrix.append(grad_ln_Psi_lambda_matrix)

            num_mcmc_done += 1

        logger.info("End MCMC")
        logger.info("")

        self.__mcmc_counter += num_mcmc_done

    # hamiltonian
    @property
    def hamiltonian_data(self):
        """Return hamiltonian_data."""
        return self.__hamiltonian_data

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

    @property
    def dln_Psi_dc_lambda_matrix(self) -> npt.NDArray:
        """Return the stored dln_Psi/dc_lambda_matrix array. dim: (mcmc_counter, num_walkers, num_lambda_matrix_param)."""
        return np.array(self.__stored_grad_ln_Psi_lambda_matrix)


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

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
        jax.distributed.initialize(cluster_detection_method="mpi4py")
        logger.info("JAX distributed initialization is successful.")
        logger.info(f"JAX backend = {jax.default_backend()}.")
        logger.info("")
    except Exception as e:
        logger.info("Running on CPUs or single GPU. JAX distributed initialization is skipped.")
        logger.debug(f"Distributed initialization Exception: {e}")
        logger.info("")

    if jax.distributed.is_initialized():
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
