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
from jax import typing as jnpt
from mpi4py import MPI

from .coulomb_potential import (
    compute_bare_coulomb_potential_el_el_jax,
    compute_bare_coulomb_potential_el_ion_element_wise_jax,
    compute_bare_coulomb_potential_ion_ion_jax,
    compute_discretized_bare_coulomb_potential_el_ion_element_wise_jax,
    compute_ecp_local_parts_all_pairs_jax,
    compute_ecp_non_local_parts_nearest_neighbors_jax,
)
from .hamiltonians import Hamiltonian_data, compute_local_energy_jax
from .jastrow_factor import (
    compute_Jastrow_part_jax,
)
from .swct import SWCT_data, evaluate_swct_domega_jax, evaluate_swct_omega_jax
from .wavefunction import (
    compute_discretized_kinetic_energy_jax,
    compute_kinetic_energy_all_elements_jax,
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


# accumurate weights
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


class GFMC_fixed_projection_time_debug:
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

        # gfmc branching counter
        self.__gfmc_branching_counter = 0

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

            r_carts_up = jnp.array(r_carts_up, dtype=jnp.float64)
            r_carts_dn = jnp.array(r_carts_dn, dtype=jnp.float64)

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

        self.__init_attributes()

    def __init_attributes(self):
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

    # collecting factor
    @property
    def num_gfmc_collect_steps(self):
        """Return num_gfmc_collect_steps."""
        return self.__num_gfmc_collect_steps

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

    def run(self, num_mcmc_steps: int = 50) -> None:
        """Run LRDMC with multiple walkers.

        Args:
            num_branching (int): number of branching (reconfiguration of walkers).
            max_time (int): maximum time in sec.
        """
        # initialize numpy random seed
        np.random.seed(self.__mpi_seed)

        # Main branching loop.
        gfmc_interval = int(np.maximum(num_mcmc_steps / 100, 1))  # gfmc_projection set print-interval

        logger.info("-Start branching-")
        progress = (self.__gfmc_branching_counter) / (num_mcmc_steps + self.__gfmc_branching_counter) * 100.0
        logger.info(
            f"  branching step = {self.__gfmc_branching_counter}/{num_mcmc_steps + self.__gfmc_branching_counter}: {progress:.1f} %."
        )

        num_mcmc_done = 0
        for i_branching in range(num_mcmc_steps):
            if (i_branching + 1) % gfmc_interval == 0:
                progress = (
                    (i_branching + self.__gfmc_branching_counter + 1) / (num_mcmc_steps + self.__gfmc_branching_counter) * 100.0
                )
                logger.info(
                    f"  branching step = {i_branching + self.__gfmc_branching_counter + 1}/{num_mcmc_steps + self.__gfmc_branching_counter}: {progress:.1f} %."
                )

            # Always set the initial weight list to 1.0
            projection_counter_list = jnp.array([0 for _ in range(self.__num_walkers)])
            e_L_list = jnp.array([1.0 for _ in range(self.__num_walkers)])
            w_L_list = jnp.array([1.0 for _ in range(self.__num_walkers)])

            logger.devel("  Projection is on going....")

            # projection loop
            projection_counter_list = np.array(projection_counter_list)
            e_L_list = np.array(e_L_list)
            w_L_list = np.array(w_L_list)
            latest_r_up_carts = np.array(self.__latest_r_up_carts)
            latest_r_dn_carts = np.array(self.__latest_r_dn_carts)
            jax_PRNG_key_list = np.array(self.__jax_PRNG_key_list)

            non_local_move = self.__non_local_move
            alat = self.__alat
            hamiltonian_data = self.__hamiltonian_data

            for i_walker in range(self.__num_walkers):
                projection_counter = projection_counter_list[i_walker]
                tau_left = self.__tau
                w_L = w_L_list[i_walker]

                r_up_carts = latest_r_up_carts[i_walker]
                r_dn_carts = latest_r_dn_carts[i_walker]
                jax_PRNG_key = jax_PRNG_key_list[i_walker]

                while tau_left > 0.0:
                    projection_counter += 1

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
                    R = jnp.eye(3)  # Rotate in the order x -> y -> z

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

                    # compose discretized el-ion potentials
                    diagonal_bare_coulomb_part_el_ion_zv_dn = (
                        diagonal_bare_coulomb_part_el_ion_elements_dn
                        + diagonal_kinetic_continuum_elements_dn
                        - non_diagonal_kinetic_part_elements_dn
                    )
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
                                    RT=R.T,
                                )
                            )

                            V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                            diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                            non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                            non_diagonal_sum_hamiltonian = (
                                non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp
                            )

                        elif non_local_move == "dltmove":
                            mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                                compute_ecp_non_local_parts_nearest_neighbors_jax(
                                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                    wavefunction_data=hamiltonian_data.wavefunction_data,
                                    r_up_carts=r_up_carts,
                                    r_dn_carts=r_dn_carts,
                                    flag_determinant_only=True,
                                    RT=R.T,
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
                            non_diagonal_sum_hamiltonian = (
                                non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp
                            )

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
                    logger.debug(f"  tau_update={tau_update}")

                    # update weight
                    w_L = w_L * jnp.exp(-tau_update * e_L)

                    # update tau_left
                    tau_left = tau_left - tau_update
                    logger.debug(f"tau_left = {tau_left}.")

                    if tau_left <= 0.0:  # '= is very important!!'
                        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                        break
                    else:
                        # electron position update
                        # random choice
                        # k = np.random.choice(len(non_diagonal_move_probabilities), p=non_diagonal_move_probabilities)
                        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                        cdf = jnp.cumsum(non_diagonal_move_probabilities)
                        random_value = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
                        k = jnp.searchsorted(cdf, random_value)
                        r_up_carts = non_diagonal_move_mesh_r_up_carts[k]
                        r_dn_carts = non_diagonal_move_mesh_r_dn_carts[k]

                projection_counter_list[i_walker] = projection_counter
                e_L_list[i_walker] = e_L
                w_L_list[i_walker] = w_L
                latest_r_up_carts[i_walker] = r_up_carts
                latest_r_dn_carts[i_walker] = r_dn_carts
                jax_PRNG_key_list[i_walker] = jax_PRNG_key

            # """
            # adjust jax_PRNG_key (consistent with the production code)
            num_max_projection = np.max(projection_counter_list)
            for i_walker in range(self.__num_walkers):
                jax_PRNG_key = jax_PRNG_key_list[i_walker]
                for _ in range(num_max_projection - projection_counter_list[i_walker]):
                    jax_PRNG_key, _ = jax.random.split(jax_PRNG_key)
                    jax_PRNG_key, _ = jax.random.split(jax_PRNG_key)
                    jax_PRNG_key, _ = jax.random.split(jax_PRNG_key)
                jax_PRNG_key_list[i_walker] = jax_PRNG_key
            # """

            # projection ends
            projection_counter_list = jnp.array(projection_counter_list)
            e_L_list = jnp.array(e_L_list)
            w_L_list = jnp.array(w_L_list)
            self.__latest_r_up_carts = jnp.array(latest_r_up_carts)
            self.__latest_r_dn_carts = jnp.array(latest_r_dn_carts)
            self.__jax_PRNG_key_list = jnp.array(jax_PRNG_key_list)

            logger.debug("  Projection ends.")

            # jnp.array -> np.array
            w_L_latest = np.array(w_L_list)
            e_L_latest = np.array(e_L_list)

            # jnp.array -> np.array
            latest_r_up_carts_before_branching = np.array(self.__latest_r_up_carts)
            latest_r_dn_carts_before_branching = np.array(self.__latest_r_dn_carts)

            # MPI reduce
            r_up_carts_shape = latest_r_up_carts_before_branching.shape
            r_up_carts_gathered_dyad = (mpi_rank, latest_r_up_carts_before_branching)
            r_up_carts_gathered_dyad = mpi_comm.gather(r_up_carts_gathered_dyad, root=0)

            r_dn_carts_shape = latest_r_dn_carts_before_branching.shape
            r_dn_carts_gathered_dyad = (mpi_rank, latest_r_dn_carts_before_branching)
            r_dn_carts_gathered_dyad = mpi_comm.gather(r_dn_carts_gathered_dyad, root=0)

            e_L_gathered_dyad = (mpi_rank, e_L_latest)
            e_L_gathered_dyad = mpi_comm.gather(e_L_gathered_dyad, root=0)
            w_L_gathered_dyad = (mpi_rank, w_L_latest)
            w_L_gathered_dyad = mpi_comm.gather(w_L_gathered_dyad, root=0)

            # num projection counter
            ave_projection_counter = np.mean(projection_counter_list)
            ave_projection_counter_gathered = mpi_comm.gather(ave_projection_counter, root=0)

            if mpi_rank == 0:
                zeta = float(np.random.random())
                r_up_carts_gathered_dict = dict(r_up_carts_gathered_dyad)
                r_dn_carts_gathered_dict = dict(r_dn_carts_gathered_dyad)
                e_L_gathered_dict = dict(e_L_gathered_dyad)
                w_L_gathered_dict = dict(w_L_gathered_dyad)
                r_up_carts_gathered = np.concatenate([r_up_carts_gathered_dict[i] for i in range(mpi_size)])
                r_dn_carts_gathered = np.concatenate([r_dn_carts_gathered_dict[i] for i in range(mpi_size)])
                e_L_gathered = np.concatenate([e_L_gathered_dict[i] for i in range(mpi_size)])
                w_L_gathered = np.concatenate([w_L_gathered_dict[i] for i in range(mpi_size)])
                e_L2_averaged = np.sum(w_L_gathered * e_L_gathered**2) / np.sum(w_L_gathered)
                e_L_averaged = np.sum(w_L_gathered * e_L_gathered) / np.sum(w_L_gathered)
                w_L_averaged = np.average(w_L_gathered)
                # add a dummy dim
                e_L2_averaged = np.expand_dims(e_L2_averaged, axis=0)
                e_L_averaged = np.expand_dims(e_L_averaged, axis=0)
                w_L_averaged = np.expand_dims(w_L_averaged, axis=0)
                # store  # This should stored only for MPI-rank = 0 !!!
                self.__stored_e_L2.append(e_L2_averaged)
                self.__stored_e_L.append(e_L_averaged)
                self.__stored_w_L.append(w_L_averaged)

                # branching
                probabilities = w_L_gathered / w_L_gathered.sum()
                # correlated choice (see Sandro's textbook, page 182)
                z_list = [(alpha + zeta) / len(probabilities) for alpha in range(len(probabilities))]
                logger.devel(f"z_list = {z_list}")
                cumulative_prob = np.cumsum(probabilities)
                chosen_walker_indices_old = np.array(
                    [next(idx for idx, prob in enumerate(cumulative_prob) if z <= prob) for z in z_list]
                )
                proposed_r_up_carts = r_up_carts_gathered[chosen_walker_indices_old]
                proposed_r_dn_carts = r_dn_carts_gathered[chosen_walker_indices_old]

                num_survived_walkers = len(set(chosen_walker_indices_old))
                num_killed_walkers = len(w_L_gathered) - len(set(chosen_walker_indices_old))
                stored_average_projection_counter = np.mean(ave_projection_counter_gathered)
            else:
                num_survived_walkers = None
                num_killed_walkers = None
                stored_average_projection_counter = None
                proposed_r_up_carts = None
                proposed_r_dn_carts = None

            num_survived_walkers = mpi_comm.bcast(num_survived_walkers, root=0)
            num_killed_walkers = mpi_comm.bcast(num_killed_walkers, root=0)
            stored_average_projection_counter = mpi_comm.bcast(stored_average_projection_counter, root=0)

            proposed_r_up_carts = mpi_comm.bcast(proposed_r_up_carts, root=0)
            proposed_r_dn_carts = mpi_comm.bcast(proposed_r_dn_carts, root=0)

            proposed_r_up_carts = proposed_r_up_carts.reshape(
                mpi_size, r_up_carts_shape[0], r_up_carts_shape[1], r_up_carts_shape[2]
            )
            proposed_r_dn_carts = proposed_r_dn_carts.reshape(
                mpi_size, r_dn_carts_shape[0], r_dn_carts_shape[1], r_dn_carts_shape[2]
            )

            # set new r_up_carts and r_dn_carts, and, np.array -> jnp.array
            latest_r_up_carts_after_branching = proposed_r_up_carts[mpi_rank, :, :, :]
            latest_r_dn_carts_after_branching = proposed_r_dn_carts[mpi_rank, :, :, :]

            # np.array -> jnp.array
            self.__num_survived_walkers += num_survived_walkers
            self.__num_killed_walkers += num_killed_walkers
            self.__stored_average_projection_counter.append(stored_average_projection_counter)
            self.__latest_r_up_carts = jnp.array(latest_r_up_carts_after_branching)
            self.__latest_r_dn_carts = jnp.array(latest_r_dn_carts_after_branching)

            num_mcmc_done += 1

        logger.info("-End branching-")
        logger.info("")

        # count up
        self.__gfmc_branching_counter += i_branching + 1


class GFMC_fixed_num_projection_debug:
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
        random_discretized_mesh=False,
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
        self.__random_discretized_mesh = random_discretized_mesh
        self.__non_local_move = non_local_move

        # derivative flags
        self.__comput_position_deriv = comput_position_deriv

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

            r_carts_up = jnp.array(r_carts_up, dtype=jnp.float64)
            r_carts_dn = jnp.array(r_carts_dn, dtype=jnp.float64)

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

    # collecting factor
    @property
    def num_gfmc_collect_steps(self):
        """Return num_gfmc_collect_steps."""
        return self.__num_gfmc_collect_steps

    @num_gfmc_collect_steps.setter
    def num_gfmc_collect_steps(self, num_gfmc_collect_steps):
        """Set num_gfmc_collect_steps."""
        self.__num_gfmc_collect_steps = num_gfmc_collect_steps

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

    def run(self, num_mcmc_steps: int = 50) -> None:
        """Run LRDMC with multiple walkers.

        Args:
            num_branching (int): number of branching (reconfiguration of walkers).
            max_time (int): maximum time in sec.
        """
        # initialize numpy random seed
        np.random.seed(self.__mpi_seed)

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

        def _projection(
            init_w_L: float,
            init_r_up_carts: jnpt.ArrayLike,
            init_r_dn_carts: jnpt.ArrayLike,
            init_jax_PRNG_key: jnpt.ArrayLike,
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
            w_L, r_up_carts, r_dn_carts, jax_PRNG_key = init_w_L, init_r_up_carts, init_r_dn_carts, init_jax_PRNG_key

            for _ in range(num_mcmc_per_measurement):
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
                if self.__random_discretized_mesh:
                    alpha, beta, gamma = jax.random.uniform(
                        subkey, shape=(3,), minval=-2 * jnp.pi, maxval=2 * jnp.pi
                    )  # Rotation angle around the x,y,z-axis (in radians)
                else:
                    alpha, beta, gamma = (0.0, 0.0, 0.0)
                R = generate_rotation_matrix(alpha, beta, gamma)

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
                                RT=R.T,
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
                                RT=R.T,
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

                # compute bar_b_L
                b_x = 1.0 / (diagonal_sum_hamiltonian - E_scf) * b_x_bar

                # update weight
                w_L = w_L * b_x

                # electron position update
                # random choice
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                cdf = jnp.cumsum(non_diagonal_move_probabilities)
                random_value = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
                k = jnp.searchsorted(cdf, random_value)
                r_up_carts = non_diagonal_move_mesh_r_up_carts[k]
                r_dn_carts = non_diagonal_move_mesh_r_dn_carts[k]

            latest_w_L, latest_r_up_carts, latest_r_dn_carts, latest_jax_PRNG_key, latest_RT = (
                w_L,
                r_up_carts,
                r_dn_carts,
                jax_PRNG_key,
                R.T,
            )

            return (latest_w_L, latest_r_up_carts, latest_r_dn_carts, latest_jax_PRNG_key, latest_RT)

        def _compute_V_elements(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            RT: jnpt.ArrayLike,
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

            # compute discretized kinetic energy and mesh (with a random rotation)
            _, _, elements_non_diagonal_kinetic_part = compute_discretized_kinetic_energy_jax(
                alat=alat,
                wavefunction_data=hamiltonian_data.wavefunction_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                RT=RT,
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
                            RT=RT,
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
                            RT=RT,
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

        def _compute_local_energy(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            RT: jnpt.ArrayLike,
            non_local_move: bool,
            alat: float,
        ):
            V_diag, V_nondiag = _compute_V_elements(hamiltonian_data, r_up_carts, r_dn_carts, RT, non_local_move, alat)
            return V_diag + V_nondiag

        # MAIN MCMC loop from here !!!
        logger.info("Start GFMC")
        num_mcmc_done = 0
        progress = (self.__mcmc_counter) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
        logger.info(f"  Progress: GFMC step = {self.__mcmc_counter}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.0f} %.")
        mcmc_interval = int(np.maximum(num_mcmc_steps / 100, 1))

        for i_mcmc_step in range(num_mcmc_steps):
            if (i_mcmc_step + 1) % mcmc_interval == 0:
                progress = (i_mcmc_step + self.__mcmc_counter + 1) / (num_mcmc_steps + self.__mcmc_counter) * 100.0

                logger.info(
                    f"  Progress: GFMC step = {i_mcmc_step + self.__mcmc_counter + 1}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %."
                )

            # Always set the initial weight list to 1.0
            w_L_list = jnp.array([1.0 for _ in range(self.__num_walkers)])

            logger.devel("  Projection is on going....")

            # projection loop
            (w_L_list, self.__latest_r_up_carts, self.__latest_r_dn_carts, self.__jax_PRNG_key_list, latest_RT) = vmap(
                _projection, in_axes=(0, 0, 0, 0, None, None, None, None, None)
            )(
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

            # projection ends
            logger.devel("  Projection ends.")

            # evaluate observables
            # V_diag and e_L
            V_diag_list, V_nondiag_list = vmap(_compute_V_elements, in_axes=(None, 0, 0, 0, None, None))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                latest_RT,
                self.__non_local_move,
                self.__alat,
            )
            e_L_list = V_diag_list + V_nondiag_list

            if self.__non_local_move == "tmove":
                e_list_debug = vmap(compute_local_energy_jax, in_axes=(None, 0, 0, 0))(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    latest_RT,
                )
                if np.max(np.abs(e_L_list - e_list_debug)) > 1.0e-6:
                    logger.info(f"max(e_list - e_list_debug) = {np.max(np.abs(e_L_list - e_list_debug))}.")
                    logger.info(f"w_L_list = {w_L_list}.")
                    logger.info(f"e_L_list = {e_L_list}.")
                    logger.info(f"e_list_debug = {e_list_debug}.")
                np.testing.assert_almost_equal(np.array(e_L_list), np.array(e_list_debug), decimal=6)

            # atomic force related
            if self.__comput_position_deriv:
                grad_e_L_h, grad_e_L_r_up, grad_e_L_r_dn = vmap(
                    grad(_compute_local_energy, argnums=(0, 1, 2)), in_axes=(None, 0, 0, 0, None, None)
                )(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    latest_RT,
                    self.__non_local_move,
                    self.__alat,
                )

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

                grad_ln_Psi_h, grad_ln_Psi_r_up, grad_ln_Psi_r_dn = vmap(
                    grad(evaluate_ln_wavefunction_jax, argnums=(0, 1, 2)), in_axes=(None, 0, 0)
                )(
                    self.__hamiltonian_data.wavefunction_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )

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
            latest_r_up_carts_before_branching_old = np.array(self.__latest_r_up_carts)
            latest_r_dn_carts_before_branching_old = np.array(self.__latest_r_dn_carts)

            # MPI reduce
            r_up_carts_shape = latest_r_up_carts_before_branching_old.shape
            r_up_carts_gathered_dyad = (mpi_rank, latest_r_up_carts_before_branching_old)
            r_dn_carts_shape = latest_r_dn_carts_before_branching_old.shape
            r_dn_carts_gathered_dyad = (mpi_rank, latest_r_dn_carts_before_branching_old)

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
                # Start branching
                logger.devel(f"w_L_gathered = {w_L_gathered}")
                probabilities = w_L_gathered / w_L_gathered.sum()
                logger.devel(f"probabilities = {probabilities}")

                # correlated choice (see Sandro's textbook, page 182)
                zeta = float(np.random.random())
                z_list = [(alpha + zeta) / len(probabilities) for alpha in range(len(probabilities))]
                logger.devel(f"z_list = {z_list}")
                cumulative_prob = np.cumsum(probabilities)
                chosen_walker_indices_old = np.array(
                    [next(idx for idx, prob in enumerate(cumulative_prob) if z <= prob) for z in z_list]
                )
                proposed_r_up_carts = r_up_carts_gathered[chosen_walker_indices_old]
                proposed_r_dn_carts = r_dn_carts_gathered[chosen_walker_indices_old]

                num_survived_walkers = len(set(chosen_walker_indices_old))
                num_killed_walkers = len(w_L_gathered) - len(set(chosen_walker_indices_old))
            else:
                num_survived_walkers = None
                num_killed_walkers = None
                proposed_r_up_carts = None
                proposed_r_dn_carts = None

            num_survived_walkers = mpi_comm.bcast(num_survived_walkers, root=0)
            num_killed_walkers = mpi_comm.bcast(num_killed_walkers, root=0)

            proposed_r_up_carts = mpi_comm.bcast(proposed_r_up_carts, root=0)
            proposed_r_dn_carts = mpi_comm.bcast(proposed_r_dn_carts, root=0)

            proposed_r_up_carts = proposed_r_up_carts.reshape(
                mpi_size, r_up_carts_shape[0], r_up_carts_shape[1], r_up_carts_shape[2]
            )
            proposed_r_dn_carts = proposed_r_dn_carts.reshape(
                mpi_size, r_dn_carts_shape[0], r_dn_carts_shape[1], r_dn_carts_shape[2]
            )

            # set new r_up_carts and r_dn_carts, and, np.array -> jnp.array
            latest_r_up_carts_after_branching = proposed_r_up_carts[mpi_rank, :, :, :]
            latest_r_dn_carts_after_branching = proposed_r_dn_carts[mpi_rank, :, :, :]

            # here update the walker positions!!
            self.__num_survived_walkers += num_survived_walkers
            self.__num_killed_walkers += num_killed_walkers
            self.__latest_r_up_carts = jnp.array(latest_r_up_carts_after_branching)
            self.__latest_r_dn_carts = jnp.array(latest_r_dn_carts_after_branching)

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

        logger.info("-End branching-")
        logger.info("")

        # count up mcmc_counter
        self.__mcmc_counter += num_mcmc_done

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
