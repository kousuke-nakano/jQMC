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

import logging
import os
import time
from functools import partial
from itertools import groupby
from logging import getLogger

import jax
import numpy as np
import numpy.typing as npt
import toml
from jax import grad, jit, lax, vmap
from jax import numpy as jnp
from jax import typing as jnpt
from jax.scipy import linalg as jsp_linalg
from mpi4py import MPI

from .coulomb_potential import (
    compute_bare_coulomb_potential_el_el,
    compute_bare_coulomb_potential_el_ion_element_wise,
    compute_bare_coulomb_potential_ion_ion,
    # compute_bare_coulomb_potential_jax,
    compute_discretized_bare_coulomb_potential_el_ion_element_wise,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors,
    compute_ecp_non_local_parts_nearest_neighbors_fast_update,
)
from .determinant import (
    compute_geminal_all_elements,
    compute_geminal_dn_one_column_elements,
    compute_geminal_up_one_row_elements,
)
from .diff_mask import DiffMask, apply_diff_mask
from .hamiltonians import (
    Hamiltonian_data,
    compute_local_energy,
)
from .jastrow_factor import (
    compute_Jastrow_part,
    compute_ratio_Jastrow_part,
)
from .jqmc_utility import _generate_init_electron_configurations
from .setting import (
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_WARMUP_STEPS,
    GFMC_ON_THE_FLY_BIN_BLOCKS,
    GFMC_ON_THE_FLY_COLLECT_STEPS,
    GFMC_ON_THE_FLY_WARMUP_STEPS,
)
from .swct import SWCT_data, evaluate_swct_domega, evaluate_swct_omega
from .wavefunction import (
    compute_discretized_kinetic_energy,
    compute_discretized_kinetic_energy_fast_update,
    compute_kinetic_energy_all_elements,
    compute_kinetic_energy_all_elements_fast_update,
    evaluate_ln_wavefunction,
)

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
jax.config.update("jax_traceback_filtering", "off")

# separator
num_sep_line = 66

# MPI related
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


# accumurate weights
@partial(jit, static_argnums=1)
def _compute_G_L(w_L, num_gfmc_collect_steps):
    """Return accumulate weights for multi-dimensional w_L.

    Note: The dimension of w_L is (num_mcmc, 1)

    """
    A, x = w_L.shape

    def get_slice(n):
        return jax.lax.dynamic_slice(w_L, (n - num_gfmc_collect_steps, 0), (num_gfmc_collect_steps, x))

    indices = jnp.arange(num_gfmc_collect_steps, A)
    G_L_matrix = vmap(get_slice)(indices)  # (A - num_gfmc_collect_steps, num_gfmc_collect_steps, x)
    G_L = jnp.prod(G_L_matrix, axis=1)  # (A - num_gfmc_collect_steps, x)

    return G_L


def _compute_G_L_debug(w_L, num_gfmc_collect_steps):
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


class GFMC_t:
    """GFMC class.

    GFMC class. Runing GFMC.

    Args:
        hamiltonian_data (Hamiltonian_data):
            an instance of Hamiltonian_data
        num_walkers (int):
            the number of walkers
        num_gfmc_collect_steps(int):
            the number of steps to collect the GFMC data
        mcmc_seed (int):
            seed for the MCMC chain.
        tau (float):
            projection time (bohr^-1)
        alat (float):
            discretized grid length (bohr)
        random_discretized_mesh (bool)
            Flag for the random discretization mesh in the kinetic part and in the non-local part of ECPs.
            Valid both for all-electron and ECP calculations.
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
        random_discretized_mesh: bool = True,
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
        self.__random_discretized_mesh = random_discretized_mesh
        self.__non_local_move = non_local_move

        # gfmc branching counter
        self.__mcmc_counter = 0

        # Initialization
        self.__mpi_seed = self.__mcmc_seed * (mpi_rank + 1)
        self.__jax_PRNG_key = jax.random.PRNGKey(self.__mpi_seed)
        self.__jax_PRNG_key_list_init = jnp.array(
            [jax.random.fold_in(self.__jax_PRNG_key, nw) for nw in range(self.__num_walkers)]
        )
        self.__jax_PRNG_key_list = self.__jax_PRNG_key_list_init

        # initialize random seed
        np.random.seed(self.__mpi_seed)

        # Place electrons around each nucleus with improved spin assignment
        ## check the number of electrons
        tot_num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        tot_num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        if hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
                hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

        coords = hamiltonian_data.structure_data._positions_cart_jnp

        ## generate initial electron configurations
        r_carts_up, r_carts_dn, up_owner, dn_owner = _generate_init_electron_configurations(
            tot_num_electron_up, tot_num_electron_dn, self.__num_walkers, charges, coords
        )

        ## Electron assignment for all atoms is complete. Check the assignment.
        for i_walker in range(self.__num_walkers):
            logger.debug(f"--Walker No.{i_walker + 1}: electrons assignment--")
            nion = coords.shape[0]
            up_counts = np.bincount(up_owner[i_walker], minlength=nion)
            dn_counts = np.bincount(dn_owner[i_walker], minlength=nion)
            logger.debug(f"  Charges: {charges}")
            logger.debug(f"  up counts: {up_counts}")
            logger.debug(f"  dn counts: {dn_counts}")
            logger.debug(f"  Total counts: {up_counts + dn_counts}")

        self.__latest_r_up_carts = jnp.array(r_carts_up)
        self.__latest_r_dn_carts = jnp.array(r_carts_dn)

        logger.debug(f"  initial r_up_carts= {self.__latest_r_up_carts}")
        logger.debug(f"  initial r_dn_carts = {self.__latest_r_dn_carts}")
        logger.debug(f"  initial r_up_carts.shape = {self.__latest_r_up_carts.shape}")
        logger.debug(f"  initial r_dn_carts.shape = {self.__latest_r_dn_carts.shape}")
        logger.debug("")

        # print out the number of walkers/MPI processes
        logger.info(f"The number of MPI process = {mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self.__num_walkers}.")
        logger.info("")

        # print out hamiltonian info
        logger.info("Printing out information in hamitonian_data instance.")
        self.__hamiltonian_data._logger_info()
        logger.info("")

        # init attributes
        self.hamiltonian_data = self.__hamiltonian_data
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
        self.__hamiltonian_data = apply_diff_mask(hamiltonian_data, DiffMask(params=False, coords=False))
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

    @property
    def alat(self):
        """Return alat."""
        return self.__alat

    # weights
    @property
    def w_L(self) -> npt.NDArray:
        """Return the stored weight array. dim: (mcmc_counter, 1)."""
        # logger.info(f"np.array(self.__stored_w_L).shape = {np.array(self.__stored_w_L).shape}.")
        return _compute_G_L(np.array(self.__stored_w_L), self.__num_gfmc_collect_steps)

    # weights
    @property
    def bare_w_L(self) -> npt.NDArray:
        """Return the stored weight array. dim: (mcmc_counter, 1)."""
        # logger.info(f"np.array(self.__stored_w_L).shape = {np.array(self.__stored_w_L).shape}.")
        return np.array(self.__stored_w_L)

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
        timer_collection = 0.0
        timer_reconfiguration = 0.0
        gfmc_total_start = time.perf_counter()

        # toml(control) filename
        toml_filename = "external_control_gfmc.toml"

        # create a toml file to control the run
        if mpi_rank == 0:
            data = {"external_control": {"stop": False}}
            # Check if file exists
            if os.path.exists(toml_filename):
                logger.info(f"{toml_filename} exists, overwriting it.")
            # Write (or overwrite) the TOML file
            with open(toml_filename, "w") as f:
                logger.info(f"{toml_filename} is generated. ")
                toml.dump(data, f)
            logger.info("")
        mpi_comm.Barrier()

        # initialize numpy random seed
        np.random.seed(self.__mpi_seed)

        # precompute geminal inverses per walker for fast kinetic updates
        def _compute_initial_A_inv_t(r_up_carts, r_dn_carts):
            geminal = compute_geminal_all_elements(
                geminal_data=self.__hamiltonian_data.wavefunction_data.geminal_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )
            lu, piv = jsp_linalg.lu_factor(geminal)
            return jsp_linalg.lu_solve((lu, piv), jnp.eye(geminal.shape[0], dtype=geminal.dtype))

        self.__latest_A_old_inv = vmap(_compute_initial_A_inv_t, in_axes=(0, 0))(
            self.__latest_r_up_carts, self.__latest_r_dn_carts
        )

        # projection function.
        @jit
        def _generate_rotation_matrix_t(alpha, beta, gamma):
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
        @partial(jit, static_argnums=(7, 8, 9))
        def _projection_t(
            projection_counter: int,
            tau_left: float,
            w_L: float,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            A_old_inv: jnpt.ArrayLike,
            jax_PRNG_key: jnpt.ArrayLike,
            random_discretized_mesh: bool,
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
                jax_PRNG_key (jnpt.ArrayLike): jax PRNG key
                random_discretized_mesh (bool): Flag for the random discretization mesh in the kinetic part and the non-local part of ECPs.
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
                A_old_inv: cached inverse geminal matrix after the final projection
                jax_PRNG_key (jnpt.ArrayLike): jax PRNG key
                R.T: rotation matrix used for the discretized mesh
            """
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
                compute_kinetic_energy_all_elements(
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # generate a random rotation matrix
            jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
            if random_discretized_mesh:
                alpha, beta, gamma = jax.random.uniform(
                    subkey, shape=(3,), minval=-2 * jnp.pi, maxval=2 * jnp.pi
                )  # Rotation angle around the x,y,z-axis (in radians)
            else:
                alpha, beta, gamma = 0.0, 0.0, 0.0
            R = _generate_rotation_matrix_t(alpha, beta, gamma)  # Rotate in the order x -> y -> z

            # compute discretized kinetic energy and mesh (with a random rotation)
            mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                compute_discretized_kinetic_energy(
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
            diagonal_bare_coulomb_part_el_el = compute_bare_coulomb_potential_el_el(
                r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
            )

            # compute diagonal elements, ion-ion
            diagonal_bare_coulomb_part_ion_ion = compute_bare_coulomb_potential_ion_ion(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data
            )

            # compute diagonal elements, el-ion
            diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                compute_bare_coulomb_potential_el_ion_element_wise(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # compute diagonal elements, el-ion, discretized
            (
                diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
            ) = compute_discretized_bare_coulomb_potential_el_ion_element_wise(
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
                diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

                if non_local_move == "tmove":
                    # ecp non-local (t-move)
                    mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                        compute_ecp_non_local_parts_nearest_neighbors(
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

                elif non_local_move == "dltmove":
                    mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                        compute_ecp_non_local_parts_nearest_neighbors(
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
                    Jastrow_ref = compute_Jastrow_part(
                        jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    Jastrow_on_mesh = vmap(compute_Jastrow_part, in_axes=(None, 0, 0))(
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
            # """

            # compute the time the walker remaining in the same configuration
            jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
            xi = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
            tau_update = jnp.minimum(tau_left, jnp.log(1 - xi) / non_diagonal_sum_hamiltonian)

            # update weight
            w_L = w_L * jnp.exp(-tau_update * e_L)

            # update tau_left
            tau_left = tau_left - tau_update

            # electron position update
            # random choice
            # k = np.random.choice(len(non_diagonal_move_probabilities), p=non_diagonal_move_probabilities)
            jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
            cdf = jnp.cumsum(non_diagonal_move_probabilities)
            random_value = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
            k = jnp.searchsorted(cdf, random_value)
            proposed_r_up_carts = non_diagonal_move_mesh_r_up_carts[k]
            proposed_r_dn_carts = non_diagonal_move_mesh_r_dn_carts[k]

            new_r_up_carts = jnp.where(tau_left <= 0.0, r_up_carts, proposed_r_up_carts)  # '=' is very important!!!
            new_r_dn_carts = jnp.where(tau_left <= 0.0, r_dn_carts, proposed_r_dn_carts)  # '=' is very important!!!

            # recompute inverse for the updated configuration
            G_new = compute_geminal_all_elements(
                geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                r_up_carts=new_r_up_carts,
                r_dn_carts=new_r_dn_carts,
            )
            lu, piv = jsp_linalg.lu_factor(G_new)
            A_new_inv = jsp_linalg.lu_solve((lu, piv), jnp.eye(G_new.shape[0], dtype=G_new.dtype))

            return (
                e_L,
                projection_counter,
                tau_left,
                w_L,
                new_r_up_carts,
                new_r_dn_carts,
                A_new_inv,
                jax_PRNG_key,
                R.T,
            )

        # projection compilation.
        start_init = time.perf_counter()
        logger.info("Start compilation of the GFMC projection funciton.")
        logger.info("  Compilation is in progress...")
        projection_counter_list = jnp.array([0 for _ in range(self.__num_walkers)])
        tau_left_list = jnp.array([self.__tau for _ in range(self.__num_walkers)])
        w_L_list = jnp.array([1.0 for _ in range(self.__num_walkers)])
        (_, _, _, _, _, _, _, _, _) = vmap(_projection_t, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None))(
            projection_counter_list,
            tau_left_list,
            w_L_list,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
            self.__latest_A_old_inv,
            self.__jax_PRNG_key_list,
            self.__random_discretized_mesh,
            self.__non_local_move,
            self.__alat,
            self.__hamiltonian_data,
        )
        end_init = time.perf_counter()
        timer_projection_init += end_init - start_init
        logger.info("End compilation of the GFMC projection funciton.")
        logger.info(f"Elapsed Time = {timer_projection_init:.2f} sec.")
        logger.info("")

        # Main branching loop.
        gfmc_interval = int(np.maximum(num_mcmc_steps / 100, 1))  # gfmc_projection set print-interval

        logger.info("-Start branching-")
        progress = (self.__mcmc_counter) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
        gmfc_total_current = time.perf_counter()
        logger.info(
            f"  branching step = {self.__mcmc_counter}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %. Elapsed time = {(gmfc_total_current - gfmc_total_start):.1f} sec."
        )

        num_mcmc_done = 0
        for i_branching in range(num_mcmc_steps):
            if (i_branching + 1) % gfmc_interval == 0:
                progress = (i_branching + self.__mcmc_counter + 1) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
                gmfc_total_current = time.perf_counter()
                logger.info(
                    f"  branching step = {i_branching + self.__mcmc_counter + 1}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %. Elapsed time = {(gmfc_total_current - gfmc_total_start):.1f} sec."
                )

            # Always set the initial weight list to 1.0
            projection_counter_list = jnp.array([0 for _ in range(self.__num_walkers)])
            tau_left_list = jnp.array([self.__tau for _ in range(self.__num_walkers)])
            w_L_list = jnp.array([1.0 for _ in range(self.__num_walkers)])

            start_projection = time.perf_counter()
            # projection loop
            while True:
                (
                    e_L_list,
                    projection_counter_list,
                    tau_left_list,
                    w_L_list,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__latest_A_old_inv,
                    self.__jax_PRNG_key_list,
                    _,
                ) = vmap(_projection_t, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None))(
                    projection_counter_list,
                    tau_left_list,
                    w_L_list,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__latest_A_old_inv,
                    self.__jax_PRNG_key_list,
                    self.__random_discretized_mesh,
                    self.__non_local_move,
                    self.__alat,
                    self.__hamiltonian_data,
                )
                if np.max(tau_left_list) <= 0.0:
                    break

            # sync. jax arrays computations.
            e_L_list.block_until_ready()
            projection_counter_list.block_until_ready()
            tau_left_list.block_until_ready()
            w_L_list.block_until_ready()
            self.__latest_r_up_carts.block_until_ready()
            self.__latest_r_dn_carts.block_until_ready()
            self.__latest_A_old_inv.block_until_ready()
            self.__jax_PRNG_key_list.block_until_ready()

            end_projection = time.perf_counter()
            timer_projection_total += end_projection - start_projection

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
            start_collection = time.perf_counter()

            # random number for the later use
            """ very slow w/o jax-jit!!
            self.__jax_PRNG_key, subkey = jax.random.split(self.__jax_PRNG_key)
            zeta = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
            """
            if mpi_rank == 0:
                zeta = float(np.random.random())
            else:
                zeta = None
            zeta = mpi_comm.bcast(zeta, root=0)

            # jnp.array -> np.array
            w_L_latest = np.array(w_L_list)
            e_L_latest = np.array(e_L_list)

            # sum
            nw_sum = len(w_L_latest)
            w_L_sum = np.sum(w_L_latest)
            e_L_sum = np.sum(w_L_latest * e_L_latest)
            e_L2_sum = np.sum(w_L_latest * e_L_latest**2)

            # reduce
            nw_sum = mpi_comm.reduce(nw_sum, op=MPI.SUM, root=0)
            w_L_sum = mpi_comm.reduce(w_L_sum, op=MPI.SUM, root=0)
            e_L_sum = mpi_comm.reduce(e_L_sum, op=MPI.SUM, root=0)
            e_L2_sum = mpi_comm.reduce(e_L2_sum, op=MPI.SUM, root=0)

            if mpi_rank == 0:
                # averaged
                w_L_averaged = w_L_sum / nw_sum
                e_L_averaged = e_L_sum / w_L_sum
                e_L2_averaged = e_L2_sum / w_L_sum

                # add a dummy dim
                e_L2_averaged = np.expand_dims(e_L2_averaged, axis=0)
                e_L_averaged = np.expand_dims(e_L_averaged, axis=0)
                w_L_averaged = np.expand_dims(w_L_averaged, axis=0)

                # store  # This should stored only for MPI-rank = 0 !!!
                self.__stored_e_L2.append(e_L2_averaged)
                self.__stored_e_L.append(e_L_averaged)
                self.__stored_w_L.append(w_L_averaged)

            mpi_comm.Barrier()

            end_collection = time.perf_counter()
            timer_collection += end_collection - start_collection

            start_reconfiguration = time.perf_counter()

            # branching
            latest_r_up_carts_before_branching = np.array(self.__latest_r_up_carts)
            latest_r_dn_carts_before_branching = np.array(self.__latest_r_dn_carts)

            #########################################
            # 1. Gather only the weights to MPI_rank=0 and perform branching calculation
            #########################################
            start_ = time.perf_counter()

            # Each process computes the sum of its local walker weights.
            local_weight_sum = np.sum(w_L_latest)

            # Use pickle‐based allreduce here (allowed for this part)
            global_weight_sum = mpi_comm.allreduce(local_weight_sum, op=MPI.SUM)

            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 1.1 = {(end_ - start_) * 1e3:.3f} msec.")

            start_ = time.perf_counter()

            # Compute the local probabilities for each walker.
            local_probabilities = w_L_latest / global_weight_sum

            # Compute the local cumulative probabilities.
            local_cumprob = np.cumsum(local_probabilities)
            local_sum_arr = np.array(np.sum(local_probabilities), dtype=np.float64)
            offset_arr = np.zeros(1, dtype=np.float64)
            mpi_comm.Exscan([local_sum_arr, MPI.DOUBLE], [offset_arr, MPI.DOUBLE], op=MPI.SUM)
            if mpi_rank == 0:
                offset = 0.0
            else:
                offset = float(offset_arr[0])
            local_cumprob += offset

            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 1.2 = {(end_ - start_) * 1e3:.3f} msec.")

            start_ = time.perf_counter()

            # Gather the local cumulative probability arrays from all processes.
            total_walkers = self.num_walkers * mpi_size
            global_cumprob = np.empty(total_walkers, dtype=np.float64)
            mpi_comm.Allgather([local_cumprob, MPI.DOUBLE], [global_cumprob, MPI.DOUBLE])
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 1.3 = {(end_ - start_) * 1e3:.3f} msec.")

            # Total number of walkers across all processes.
            # Compute index range for this rank
            start_ = time.perf_counter()
            start_idx = mpi_rank * self.num_walkers
            end_idx = start_idx + self.num_walkers

            # Build only local z-array (length = self.num_walkers)
            z_local = (np.arange(start_idx, end_idx) + zeta) / total_walkers

            # Perform searchsorted and cast the result to int32
            local_chosen_indices = np.searchsorted(global_cumprob, z_local).astype(np.int32)
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 1.4 = {(end_ - start_) * 1e3:.3f} msec.")

            # Gather all local_chosen_indices across ranks using MPI.INT
            start_ = time.perf_counter()

            # Allocate buffer to receive all ranks' indices (Number of indices per rank is identical on every rank)
            all_chosen_buf = np.empty(self.num_walkers * mpi_size, dtype=np.int32)

            # Perform the all-gather operation with 32-bit integers
            mpi_comm.Allgather([local_chosen_indices, MPI.INT], [all_chosen_buf, MPI.INT])

            # Use the gathered indices for global statistics
            chosen_walker_indices = all_chosen_buf
            num_survived_walkers = len(np.unique(chosen_walker_indices))
            num_killed_walkers = total_walkers - num_survived_walkers

            # Build the local assignment list of (source_rank, source_local_index)
            local_assignment = [
                (src_global_idx // self.num_walkers, src_global_idx % self.num_walkers)
                for src_global_idx in local_chosen_indices
            ]

            # num projection counter
            ## Compute the local average of the projection counter list.
            ave_projection_counter = np.mean(projection_counter_list)

            ## Use MPI allgather to collect the local averages from all processes.
            ave_projection_counter_gathered = mpi_comm.allgather(ave_projection_counter)

            ## Each process computes the overall (global) average projection counter.
            stored_average_projection_counter = np.mean(ave_projection_counter_gathered)

            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 1.5 = {(end_ - start_) * 1e3:.3f} msec.")

            #########################################
            # 2. In each process, prepare for data exchange based on the new walker selection
            #########################################
            start_ = time.perf_counter()
            latest_r_up_carts_after_branching = np.empty_like(latest_r_up_carts_before_branching)
            latest_r_dn_carts_after_branching = np.empty_like(latest_r_dn_carts_before_branching)

            reqs = {}
            for dest_idx, (src_rank, src_local_idx) in enumerate(local_assignment):
                if src_rank == mpi_rank:
                    # Local copy: no communication needed
                    latest_r_up_carts_after_branching[dest_idx] = latest_r_up_carts_before_branching[src_local_idx]
                    latest_r_dn_carts_after_branching[dest_idx] = latest_r_dn_carts_before_branching[src_local_idx]
                else:
                    reqs.setdefault(src_rank, []).append((dest_idx, src_local_idx))
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 2 = {(end_ - start_) * 1e3:.3f} msec.")

            #########################################
            # 3. Exchange only the necessary walker data between processes using asynchronous communication
            #########################################

            # 3.1.1: Flatten `reqs` into an (N_req × 3) int32 array of triplets
            start_ = time.perf_counter()
            flat_list = [
                (src_rank, dest_idx, src_local_idx) for src_rank, pairs in reqs.items() for dest_idx, src_local_idx in pairs
            ]
            triplets = np.array(flat_list, dtype=np.int32) if flat_list else np.empty((0, 3), dtype=np.int32)
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.1.1 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.2: Compute how many ints to send to each rank (3 ints per request)
            start_ = time.perf_counter()
            counts_per_rank = np.bincount(triplets[:, 0], minlength=mpi_size)  # # reqs per src_rank
            send_counts = (counts_per_rank * 3).astype(np.int32)  # # ints per src_rank
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.1.2 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.3: Post nonblocking Alltoall to exchange counts
            start_ = time.perf_counter()
            recv_counts = np.empty_like(send_counts)
            req_counts = mpi_comm.Ialltoall([send_counts, MPI.INT], [recv_counts, MPI.INT])
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.1.3 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.4: Build send_buf while counts exchange is in flight
            start_ = time.perf_counter()
            #   sort by src_rank so that each destination's data is contiguous
            order = np.argsort(triplets[:, 0], kind="mergesort") if triplets.size else np.empty(0, dtype=np.int32)
            sorted_tr = triplets[order]  # shape = (N_req, 3)
            send_buf = sorted_tr.ravel()  # shape = (N_req*3,)
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.1.4 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.5: Wait for counts exchange to complete
            start_ = time.perf_counter()
            req_counts.Wait()
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.1.5 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.6: Build displacements for send/recv from counts
            start_ = time.perf_counter()
            send_displs = np.zeros_like(send_counts)
            send_displs[1:] = np.cumsum(send_counts)[:-1]
            recv_displs = np.zeros_like(recv_counts)
            recv_displs[1:] = np.cumsum(recv_counts)[:-1]
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.1.6 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.7: Allocate recv buffer of the exact size
            start_ = time.perf_counter()
            total_recv = int(np.sum(recv_counts))
            recv_buf = np.empty(total_recv, dtype=np.int32)
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.1.7 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.8: Post blocking Alltoallv to exchange the triplets
            start_ = time.perf_counter()
            mpi_comm.Alltoallv([send_buf, send_counts, send_displs, MPI.INT], [recv_buf, recv_counts, recv_displs, MPI.INT])
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.1.8 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.9: Wait for data to arrive and reconstruct per‐process request dicts
            start_ = time.perf_counter()
            all_reqs = []
            for p in range(mpi_size):
                off = recv_displs[p]
                cnt = recv_counts[p]
                block = recv_buf[off : off + cnt]
                if block.size == 0:
                    all_reqs.append({})
                    continue
                rec_tr = block.reshape(-1, 3)
                proc_dict = {}
                for sr, dest_idx, src_local_idx in rec_tr:
                    proc_dict.setdefault(int(sr), []).append((int(dest_idx), int(src_local_idx)))
                all_reqs.append(proc_dict)
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.1.9 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.10: Filter out empty request dicts
            start_ = time.perf_counter()
            non_empty_all_reqs = [(p, rd) for p, rd in enumerate(all_reqs) if rd]
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.1.10 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-2. Build incoming_reqs: who needs data from me? ---
            start_ = time.perf_counter()
            incoming_reqs = [
                (p, src_local_idx, dest_idx)
                for p, proc_req in non_empty_all_reqs
                if p != mpi_rank
                for dest_idx, src_local_idx in proc_req.get(mpi_rank, [])
            ]
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.2 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-3. Post nonblocking receives using Irecv for both up and dn buffers. ---
            start_ = time.perf_counter()
            recv_buffers = {}
            recv_reqs_up = {}
            recv_reqs_dn = {}
            for src_rank, req_list in reqs.items():
                if not req_list:
                    continue
                count = len(req_list)
                shape = latest_r_up_carts_before_branching.shape[1:]
                buf_up = np.empty((count, *shape), dtype=latest_r_up_carts_before_branching.dtype)
                buf_dn = np.empty((count, *shape), dtype=latest_r_dn_carts_before_branching.dtype)
                recv_buffers[src_rank] = (buf_up, buf_dn)
                recv_reqs_up[src_rank] = mpi_comm.Irecv([buf_up, MPI.DOUBLE], source=src_rank, tag=200)
                recv_reqs_dn[src_rank] = mpi_comm.Irecv([buf_dn, MPI.DOUBLE], source=src_rank, tag=201)
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.3 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-4. Prepare and post nonblocking sends using Isend. ---
            start_ = time.perf_counter()
            send_requests = []
            for dest_rank, group in groupby(sorted(incoming_reqs, key=lambda x: x[0]), key=lambda x: x[0]):
                idxs = [src_local for (_, src_local, _) in group]
                buf_up = latest_r_up_carts_before_branching[idxs]
                buf_dn = latest_r_dn_carts_before_branching[idxs]
                send_requests.append(mpi_comm.Isend([buf_up, MPI.DOUBLE], dest=dest_rank, tag=200))
                send_requests.append(mpi_comm.Isend([buf_dn, MPI.DOUBLE], dest=dest_rank, tag=201))
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.4 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-5. Wait for all nonblocking sends to complete. ---
            start_ = time.perf_counter()
            MPI.Request.Waitall(send_requests)
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.5 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-6. Process the received walker data. ---
            start_ = time.perf_counter()
            for src_rank, req_list in reqs.items():
                if not req_list:
                    continue
                recv_reqs_up[src_rank].Wait()
                recv_reqs_dn[src_rank].Wait()
                buf_up, buf_dn = recv_buffers[src_rank]
                dest_idxs = [dest for (dest, _) in req_list]
                latest_r_up_carts_after_branching[dest_idxs] = buf_up
                latest_r_dn_carts_after_branching[dest_idxs] = buf_dn
            end_ = time.perf_counter()
            logger.debug(f"    timer_reconfigration step 3.6 = {(end_ - start_) * 1e3:.3f} msec.")

            # np.array -> jnp.array
            self.__num_survived_walkers += num_survived_walkers
            self.__num_killed_walkers += num_killed_walkers
            self.__stored_average_projection_counter.append(stored_average_projection_counter)
            self.__latest_r_up_carts = jnp.array(latest_r_up_carts_after_branching)
            self.__latest_r_dn_carts = jnp.array(latest_r_dn_carts_after_branching)
            self.__latest_A_old_inv = vmap(_compute_initial_A_inv_t, in_axes=(0, 0))(
                self.__latest_r_up_carts, self.__latest_r_dn_carts
            )

            # Barrier after MPI operation
            mpi_comm.Barrier()

            # timer end
            end_reconfiguration = time.perf_counter()
            timer_reconfiguration += end_reconfiguration - start_reconfiguration

            # check current time
            gmfc_current = time.perf_counter()
            if max_time < gmfc_current - gfmc_total_start:
                logger.info(f"  Stopping... Max_time = {max_time} sec. exceeds.")
                logger.info("  Break the branching loop.")
                break

            # check toml file (stop flag)
            if os.path.isfile(toml_filename):
                dict_toml = toml.load(open(toml_filename))
                try:
                    stop_flag = dict_toml["external_control"]["stop"]
                except KeyError:
                    stop_flag = False
                if stop_flag:
                    logger.info(f"  Stopping... stop_flag in {toml_filename} is true.")
                    logger.info("  Break the mcmc loop.")
                    break

            # count up, here is the end of the branching step.
            num_mcmc_done += 1

        logger.info("")

        # count up
        self.__mcmc_counter += num_mcmc_done

        gfmc_total_end = time.perf_counter()
        timer_gfmc_total = gfmc_total_end - gfmc_total_start
        timer_misc = timer_gfmc_total - (
            timer_projection_init
            + timer_projection_total
            + timer_observable
            + timer_mpi_barrier
            + timer_reconfiguration
            + timer_collection
        )

        # remove the toml file
        mpi_comm.Barrier()
        if mpi_rank == 0:
            if os.path.isfile(toml_filename):
                logger.info(f"Delete {toml_filename}")
                os.remove(toml_filename)

        # net GFMC time
        timer_net_gfmc_total = timer_gfmc_total - timer_projection_init

        # average among MPI processes
        ave_timer_gfmc_total = mpi_comm.allreduce(timer_gfmc_total, op=MPI.SUM) / mpi_size
        ave_timer_projection_init = mpi_comm.allreduce(timer_projection_init, op=MPI.SUM) / mpi_size
        ave_timer_net_gfmc_total = mpi_comm.allreduce(timer_net_gfmc_total, op=MPI.SUM) / mpi_size
        ave_timer_projection_total = mpi_comm.allreduce(timer_projection_total, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_observable = mpi_comm.allreduce(timer_observable, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_mpi_barrier = mpi_comm.allreduce(timer_mpi_barrier, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_collection = mpi_comm.allreduce(timer_collection, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_reconfiguration = mpi_comm.allreduce(timer_reconfiguration, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_misc = mpi_comm.allreduce(timer_misc, op=MPI.SUM) / mpi_size / num_mcmc_done

        sum_killed_walkers = mpi_comm.allreduce(self.__num_killed_walkers, op=MPI.SUM)
        sum_survived_walkers = mpi_comm.allreduce(self.__num_survived_walkers, op=MPI.SUM)
        ave_stored_average_projection_counter = (
            mpi_comm.allreduce(np.mean(self.__stored_average_projection_counter), op=MPI.SUM) / mpi_size
        )

        logger.info(f"Total GFMC time for {num_mcmc_done} branching steps = {ave_timer_gfmc_total: .3f} sec.")
        logger.info(f"Pre-compilation time for GFMC = {ave_timer_projection_init: .3f} sec.")
        logger.info(f"Net GFMC time without pre-compilations = {ave_timer_net_gfmc_total: .3f} sec.")
        logger.info(f"Elapsed times per branching, averaged over {num_mcmc_done} branching steps.")
        logger.info(f"  Projection time per branching = {ave_timer_projection_total * 10**3: .3f} msec.")
        logger.info(f"  Time for Observable measurement time per branching = {ave_timer_observable * 10**3: .3f} msec.")
        logger.info(f"  Time for MPI barrier before branching = {ave_timer_mpi_barrier * 10**3:.2f} msec.")
        logger.info(f"  Time for walker observable collections time per branching = {ave_timer_collection * 10**3: .3f} msec.")
        logger.info(f"  Time for walker reconfiguration time per branching = {ave_timer_reconfiguration * 10**3: .3f} msec.")
        logger.info(f"  Time for misc. (others) = {ave_timer_misc * 10**3:.2f} msec.")
        logger.info(
            f"Survived walkers ratio = {sum_survived_walkers / (sum_survived_walkers + sum_killed_walkers) * 100:.2f} %"
        )
        logger.info(f"Average of the number of projections  = {ave_stored_average_projection_counter:.0f}")
        logger.info("")

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
        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if num_mcmc_warmup_steps < GFMC_MIN_WARMUP_STEPS:
            logger.warning(f"num_mcmc_warmup_steps should be larger than {GFMC_MIN_WARMUP_STEPS}")
        if num_mcmc_bin_blocks < GFMC_MIN_BIN_BLOCKS:
            logger.warning(f"num_mcmc_bin_blocks should be larger than {GFMC_MIN_BIN_BLOCKS}")

        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if self.mcmc_counter < num_mcmc_warmup_steps:
            logger.error("mcmc_counter should be larger than num_mcmc_warmup_steps")
            raise ValueError
        if self.mcmc_counter - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
            logger.error("(mcmc_counter - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")
            raise ValueError

        if num_mcmc_bin_blocks < mpi_size or mpi_size == 1:
            if mpi_rank == 0:
                e_L = self.e_L[num_mcmc_warmup_steps:]
                e_L2 = self.e_L2[num_mcmc_warmup_steps:]
                w_L = self.w_L[num_mcmc_warmup_steps:]
                w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
                w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
                w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
                w_L_e_L2_split = np.array_split(w_L * e_L2, num_mcmc_bin_blocks, axis=0)
                w_L_e_L2_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L2_split]))

                w_L_binned_local = np.array(w_L_binned)
                w_L_e_L_binned_local = np.array(w_L_e_L_binned)
                w_L_e_L2_binned_local = np.array(w_L_e_L2_binned)

                ## local sum
                w_L_binned_local_sum = np.sum(w_L_binned_local, axis=0)
                w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local, axis=0)
                w_L_e_L2_binned_local_sum = np.sum(w_L_e_L2_binned_local, axis=0)

                ## jackknie binned samples
                M_local = w_L_binned_local.size
                M_total = M_local

                E_jackknife_binned_local = np.array(
                    [
                        (w_L_e_L_binned_local_sum - w_L_e_L_binned_local[m]) / (w_L_binned_local_sum - w_L_binned_local[m])
                        for m in range(M_local)
                    ]
                )

                E2_jackknife_binned_local = np.array(
                    [
                        (w_L_e_L2_binned_local_sum - w_L_e_L2_binned_local[m]) / (w_L_binned_local_sum - w_L_binned_local[m])
                        for m in range(M_local)
                    ]
                )

                Var_jackknife_binned_local = E2_jackknife_binned_local - E_jackknife_binned_local**2

                # E: jackknife mean and std
                sum_E_local = np.sum(E_jackknife_binned_local)
                sumsq_E_local = np.sum(E_jackknife_binned_local**2)

                E_mean = sum_E_local / M_local
                E_var = (sumsq_E_local / M_local) - (sum_E_local / M_local) ** 2
                E_std = np.sqrt((M_local - 1) * E_var)

                # Var: jackknife mean and std
                sum_Var_local = np.sum(Var_jackknife_binned_local)
                sumsq_Var_local = np.sum(Var_jackknife_binned_local**2)

                Var_mean = sum_Var_local / M_total
                Var_var = (sumsq_Var_local / M_total) - (sum_Var_local / M_local) ** 2
                Var_std = np.sqrt((M_total - 1) * Var_var)

            else:
                E_mean = None
                E_std = None
                Var_mean = None
                Var_std = None

            # MPI broadcast
            E_mean = mpi_comm.bcast(E_mean, root=0)
            E_std = mpi_comm.bcast(E_std, root=0)
            Var_mean = mpi_comm.bcast(Var_mean, root=0)
            Var_std = mpi_comm.bcast(Var_std, root=0)

        else:
            if mpi_rank == 0:
                e_L = self.e_L[num_mcmc_warmup_steps:]
                e_L2 = self.e_L2[num_mcmc_warmup_steps:]
                w_L = self.w_L[num_mcmc_warmup_steps:]
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

            if mpi_rank == 0:
                w_L_binned_split = np.array_split(w_L_binned, mpi_size)
                w_L_e_L_binned_split = np.array_split(w_L_e_L_binned, mpi_size)
                w_L_e_L2_binned_split = np.array_split(w_L_e_L2_binned, mpi_size)
            else:
                w_L_binned_split = None
                w_L_e_L_binned_split = None
                w_L_e_L2_binned_split = None

            # scatter
            w_L_binned_local = mpi_comm.scatter(w_L_binned_split, root=0)
            w_L_e_L_binned_local = mpi_comm.scatter(w_L_e_L_binned_split, root=0)
            w_L_e_L2_binned_local = mpi_comm.scatter(w_L_e_L2_binned_split, root=0)

            w_L_binned_local = np.array(w_L_binned_local)
            w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
            w_L_e_L2_binned_local = np.array(w_L_e_L2_binned_local)

            ## local sum
            w_L_binned_local_sum = np.sum(w_L_binned_local, axis=0)
            w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local, axis=0)
            w_L_e_L2_binned_local_sum = np.sum(w_L_e_L2_binned_local, axis=0)

            ## glolbal sum
            w_L_binned_global_sum = np.empty_like(w_L_binned_local_sum)
            w_L_e_L_binned_global_sum = np.empty_like(w_L_e_L_binned_local_sum)
            w_L_e_L2_binned_global_sum = np.empty_like(w_L_e_L2_binned_local_sum)

            ## mpi Allreduce
            mpi_comm.Allreduce([w_L_binned_local_sum, MPI.DOUBLE], [w_L_binned_global_sum, MPI.DOUBLE], op=MPI.SUM)
            mpi_comm.Allreduce([w_L_e_L_binned_local_sum, MPI.DOUBLE], [w_L_e_L_binned_global_sum, MPI.DOUBLE], op=MPI.SUM)
            mpi_comm.Allreduce([w_L_e_L2_binned_local_sum, MPI.DOUBLE], [w_L_e_L2_binned_global_sum, MPI.DOUBLE], op=MPI.SUM)

            ## jackknie binned samples
            M_local = w_L_binned_local.size
            M_total = mpi_comm.allreduce(M_local, op=MPI.SUM)

            E_jackknife_binned_local = np.array(
                [
                    (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                    for m in range(M_local)
                ]
            )

            E2_jackknife_binned_local = np.array(
                [
                    (w_L_e_L2_binned_global_sum - w_L_e_L2_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                    for m in range(M_local)
                ]
            )

            Var_jackknife_binned_local = E2_jackknife_binned_local - E_jackknife_binned_local**2

            # E: jackknife mean and std
            sum_E_local = np.sum(E_jackknife_binned_local)
            sumsq_E_local = np.sum(E_jackknife_binned_local**2)

            sum_E_global = np.empty_like(sum_E_local)
            sumsq_E_global = np.empty_like(sumsq_E_local)

            mpi_comm.Allreduce([sum_E_local, MPI.DOUBLE], [sum_E_global, MPI.DOUBLE], op=MPI.SUM)
            mpi_comm.Allreduce([sumsq_E_local, MPI.DOUBLE], [sumsq_E_global, MPI.DOUBLE], op=MPI.SUM)

            E_mean = sum_E_global / M_total
            E_var = (sumsq_E_global / M_total) - (sum_E_global / M_total) ** 2
            E_std = np.sqrt((M_total - 1) * E_var)

            # Var: jackknife mean and std
            sum_Var_local = np.sum(Var_jackknife_binned_local)
            sumsq_Var_local = np.sum(Var_jackknife_binned_local**2)

            sum_Var_global = np.empty_like(sum_Var_local)
            sumsq_Var_global = np.empty_like(sumsq_Var_local)

            mpi_comm.Allreduce([sum_Var_local, MPI.DOUBLE], [sum_Var_global, MPI.DOUBLE], op=MPI.SUM)
            mpi_comm.Allreduce([sumsq_Var_local, MPI.DOUBLE], [sumsq_Var_global, MPI.DOUBLE], op=MPI.SUM)

            Var_mean = sum_Var_global / M_total
            Var_var = (sumsq_Var_global / M_total) - (sum_Var_global / M_total) ** 2
            Var_std = np.sqrt((M_total - 1) * Var_var)

        # return
        return (E_mean, E_std, Var_mean, Var_std)


class _GFMC_t_debug:
    """GFMC class.

    GFMC class. Runing GFMC.

    Args:
        hamiltonian_data (Hamiltonian_data):
            an instance of Hamiltonian_data
        num_walkers (int):
            the number of walkers
        num_gfmc_collect_steps(int):
            the number of steps to collect the GFMC data
        mcmc_seed (int):
            seed for the MCMC chain.
        tau (float):
            projection time (bohr^-1)
        alat (float):
            discretized grid length (bohr)
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

        # Initialization
        self.__mpi_seed = self.__mcmc_seed * (mpi_rank + 1)
        self.__jax_PRNG_key = jax.random.PRNGKey(self.__mpi_seed)
        self.__jax_PRNG_key_list_init = jnp.array(
            [jax.random.fold_in(self.__jax_PRNG_key, nw) for nw in range(self.__num_walkers)]
        )
        self.__jax_PRNG_key_list = self.__jax_PRNG_key_list_init

        # initialize random seed
        np.random.seed(self.__mpi_seed)

        # Place electrons around each nucleus with improved spin assignment
        ## check the number of electrons
        tot_num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        tot_num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        if hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
                hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

        coords = hamiltonian_data.structure_data._positions_cart_jnp

        ## generate initial electron configurations
        r_carts_up, r_carts_dn, up_owner, dn_owner = _generate_init_electron_configurations(
            tot_num_electron_up, tot_num_electron_dn, self.__num_walkers, charges, coords
        )

        ## Electron assignment for all atoms is complete. Check the assignment.
        for i_walker in range(self.__num_walkers):
            logger.debug(f"--Walker No.{i_walker + 1}: electrons assignment--")
            nion = coords.shape[0]
            up_counts = np.bincount(up_owner[i_walker], minlength=nion)
            dn_counts = np.bincount(dn_owner[i_walker], minlength=nion)
            logger.debug(f"  Charges: {charges}")
            logger.debug(f"  up counts: {up_counts}")
            logger.debug(f"  dn counts: {dn_counts}")
            logger.debug(f"  Total counts: {up_counts + dn_counts}")

        self.__latest_r_up_carts = jnp.array(r_carts_up)
        self.__latest_r_dn_carts = jnp.array(r_carts_dn)

        logger.debug(f"  initial r_up_carts= {self.__latest_r_up_carts}")
        logger.debug(f"  initial r_dn_carts = {self.__latest_r_dn_carts}")
        logger.debug(f"  initial r_up_carts.shape = {self.__latest_r_up_carts.shape}")
        logger.debug(f"  initial r_dn_carts.shape = {self.__latest_r_dn_carts.shape}")
        logger.debug("")

        # print out the number of walkers/MPI processes
        logger.info(f"The number of MPI process = {mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self.__num_walkers}.")
        logger.info("")

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
        return _compute_G_L_debug(np.array(self.__stored_w_L), self.__num_gfmc_collect_steps)

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
        progress = (self.__mcmc_counter) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
        logger.info(f"  branching step = {self.__mcmc_counter}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %.")

        num_mcmc_done = 0
        for i_branching in range(num_mcmc_steps):
            if (i_branching + 1) % gfmc_interval == 0:
                progress = (i_branching + self.__mcmc_counter + 1) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
                logger.info(
                    f"  branching step = {i_branching + self.__mcmc_counter + 1}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %."
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
                        compute_kinetic_energy_all_elements(
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
                        compute_discretized_kinetic_energy(
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
                    diagonal_bare_coulomb_part_el_el = compute_bare_coulomb_potential_el_el(
                        r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
                    )

                    # compute diagonal elements, ion-ion
                    diagonal_bare_coulomb_part_ion_ion = compute_bare_coulomb_potential_ion_ion(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data
                    )

                    # compute diagonal elements, el-ion
                    diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                        compute_bare_coulomb_potential_el_ion_element_wise(
                            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                        )
                    )

                    # compute diagonal elements, el-ion, discretized
                    (
                        diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                        diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
                    ) = compute_discretized_bare_coulomb_potential_el_ion_element_wise(
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
                        diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
                            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                        )

                        if non_local_move == "tmove":
                            # ecp non-local (t-move)
                            mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                                compute_ecp_non_local_parts_nearest_neighbors(
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
                                compute_ecp_non_local_parts_nearest_neighbors(
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
                            Jastrow_ref = compute_Jastrow_part(
                                jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                            )

                            Jastrow_on_mesh = vmap(compute_Jastrow_part, in_axes=(None, 0, 0))(
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

            # count up, here is the end of the branching step.
            num_mcmc_done += 1

        logger.info("")

        # count up mcmc_counter
        self.__mcmc_counter += num_mcmc_done

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
        if mpi_rank == 0:
            e_L = self.e_L[num_mcmc_warmup_steps:]
            e_L2 = self.e_L2[num_mcmc_warmup_steps:]
            w_L = self.w_L[num_mcmc_warmup_steps:]
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
            w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
            w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
            w_L_e_L2_split = np.array_split(w_L * e_L2, num_mcmc_bin_blocks, axis=0)
            w_L_e_L2_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L2_split]))

            w_L_binned_local = w_L_binned
            w_L_e_L_binned_local = w_L_e_L_binned
            w_L_e_L2_binned_local = w_L_e_L2_binned

            w_L_binned_local = np.array(w_L_binned_local)
            w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
            w_L_e_L2_binned_local = np.array(w_L_e_L2_binned_local)

            # old implementation (keep this just for debug, for the time being. To be deleted.)
            w_L_binned_global_sum = np.sum(w_L_binned_local, axis=0)
            w_L_e_L_binned_global_sum = np.sum(w_L_e_L_binned_local, axis=0)
            w_L_e_L2_binned_global_sum = np.sum(w_L_e_L2_binned_local, axis=0)

            M_local = w_L_binned_local.size
            logger.debug(f"The number of local binned samples = {M_local}")

            E_jackknife_binned_local = [
                (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                for m in range(M_local)
            ]

            E2_jackknife_binned_local = [
                (w_L_e_L2_binned_global_sum - w_L_e_L2_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                for m in range(M_local)
            ]

            Var_jackknife_binned_local = list(np.array(E2_jackknife_binned_local) - np.array(E_jackknife_binned_local) ** 2)

            # convert to numpy array
            E_jackknife_binned = np.array(E_jackknife_binned_local)
            Var_jackknife_binned = np.array(Var_jackknife_binned_local)
            M_total = len(E_jackknife_binned)
            logger.debug(f"The number of total binned samples = {M_total}")

            # jackknife mean and std
            E_mean = np.average(E_jackknife_binned)
            E_std = np.sqrt(M_total - 1) * np.std(E_jackknife_binned)
            Var_mean = np.average(Var_jackknife_binned)
            Var_std = np.sqrt(M_total - 1) * np.std(Var_jackknife_binned)

            logger.info(f"E = {E_mean} +- {E_std} Ha.")
            logger.info(f"Var(E) = {Var_mean} +- {Var_std} Ha^2.")

        else:
            E_mean = None
            E_std = None
            Var_mean = None
            Var_std = None

        # MPI broadcast
        E_mean = mpi_comm.bcast(E_mean, root=0)
        E_std = mpi_comm.bcast(E_std, root=0)
        Var_mean = mpi_comm.bcast(Var_mean, root=0)
        Var_std = mpi_comm.bcast(Var_std, root=0)

        # return
        return (E_mean, E_std, Var_mean, Var_std)


class GFMC_n:
    """GFMC class. Runing GFMC with multiple walkers.

    Args:
        hamiltonian_data (Hamiltonian_data):
            an instance of Hamiltonian_data
        num_walkers (int):
            the number of walkers
        mcmc_seed (int):
            seed for the MCMC chain.
        E_scf (float):
            Self-consistent E (Hartree)
        alat (float):
            discretized grid length (bohr)
        random_discretized_mesh (bool)
            Flag for the random discretization mesh in the kinetic part and the non-local part of ECPs.
            Valid both for all-electron and ECP calculations.
        non_local_move (str):
            treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
            Valid only for ECP calculations. Do not specify this value for all-electron calculations.
        comput_position_deriv (bool):
            if True, compute the derivatives of E wrt. atomic positions.
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
        random_discretized_mesh: bool = True,
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

        # initialize random seed
        np.random.seed(self.__mpi_seed)

        # Place electrons around each nucleus with improved spin assignment
        ## check the number of electrons
        tot_num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        tot_num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        if hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
                hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

        coords = hamiltonian_data.structure_data._positions_cart_jnp

        ## generate initial electron configurations
        r_carts_up, r_carts_dn, up_owner, dn_owner = _generate_init_electron_configurations(
            tot_num_electron_up, tot_num_electron_dn, self.__num_walkers, charges, coords
        )

        ## Electron assignment for all atoms is complete. Check the assignment.
        for i_walker in range(self.__num_walkers):
            logger.debug(f"--Walker No.{i_walker + 1}: electrons assignment--")
            nion = coords.shape[0]
            up_counts = np.bincount(up_owner[i_walker], minlength=nion)
            dn_counts = np.bincount(dn_owner[i_walker], minlength=nion)
            logger.debug(f"  Charges: {charges}")
            logger.debug(f"  up counts: {up_counts}")
            logger.debug(f"  dn counts: {dn_counts}")
            logger.debug(f"  Total counts: {up_counts + dn_counts}")

        self.__latest_r_up_carts = jnp.array(r_carts_up)
        self.__latest_r_dn_carts = jnp.array(r_carts_dn)

        logger.debug(f"  initial r_up_carts= {self.__latest_r_up_carts}")
        logger.debug(f"  initial r_dn_carts = {self.__latest_r_dn_carts}")
        logger.debug(f"  initial r_up_carts.shape = {self.__latest_r_up_carts.shape}")
        logger.debug(f"  initial r_dn_carts.shape = {self.__latest_r_dn_carts.shape}")
        logger.debug("")

        # print out the number of walkers/MPI processes
        logger.info(f"The number of MPI process = {mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self.__num_walkers}.")
        logger.info("")

        # SWCT data
        self.__swct_data = SWCT_data(structure=self.__hamiltonian_data.structure_data)

        # print out hamiltonian info
        logger.info("Printing out information in hamitonian_data instance.")
        self.__hamiltonian_data._logger_info()
        logger.info("")

        # init attributes
        self.hamiltonian_data = self.__hamiltonian_data
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

        # stored G_L and G_e_L for updating the E_scf
        self.__G_L = []
        self.__G_e_L = []

    # hamiltonian
    @property
    def hamiltonian_data(self):
        """Return hamiltonian_data."""
        return self.__hamiltonian_data

    @hamiltonian_data.setter
    def hamiltonian_data(self, hamiltonian_data):
        """Set hamiltonian_data."""
        if self.__comput_position_deriv:
            # Keep gradients for both params and coords when position derivatives are needed.
            self.__hamiltonian_data = apply_diff_mask(hamiltonian_data, DiffMask(params=True, coords=True))
        else:
            self.__hamiltonian_data = apply_diff_mask(hamiltonian_data, DiffMask(params=False, coords=False))
        self.__init_attributes()

    # collecting factor
    @property
    def num_gfmc_collect_steps(self):
        """Return num_gfmc_collect_steps."""
        return self.__num_gfmc_collect_steps

    @num_gfmc_collect_steps.setter
    def num_gfmc_collect_steps(self, num_gfmc_collect_steps):
        """Set num_gfmc_collect_steps."""
        if num_gfmc_collect_steps < GFMC_MIN_COLLECT_STEPS:
            logger.warning(f"num_gfmc_collect_steps should be larger than {GFMC_MIN_COLLECT_STEPS}")
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

    @property
    def alat(self):
        """Return alat."""
        return self.__alat

    # weights
    @property
    def w_L(self) -> npt.NDArray:
        """Return the stored weight array. dim: (mcmc_counter, 1)."""
        # logger.info(f"np.array(self.__stored_w_L).shape = {np.array(self.__stored_w_L).shape}.")
        return _compute_G_L(np.array(self.__stored_w_L), self.__num_gfmc_collect_steps)

    # weights
    @property
    def bare_w_L(self) -> npt.NDArray:
        """Return the stored weight array. dim: (mcmc_counter, 1)."""
        # logger.info(f"np.array(self.__stored_w_L).shape = {np.array(self.__stored_w_L).shape}.")
        return np.array(self.__stored_w_L)

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
        timer_mpi_barrier = 0.0
        timer_reconfiguration = 0.0
        timer_collection = 0.0
        timer_update_E_scf = 0.0

        # toml(control) filename
        toml_filename = "external_control_gfmc.toml"

        # create a toml file to control the run
        if mpi_rank == 0:
            data = {"external_control": {"stop": False}}
            # Check if file exists
            if os.path.exists(toml_filename):
                logger.info(f"{toml_filename} exists, overwriting it.")
            # Write (or overwrite) the TOML file
            with open(toml_filename, "w") as f:
                logger.info(f"{toml_filename} is generated. ")
                toml.dump(data, f)
            logger.info("")
        mpi_comm.Barrier()

        gfmc_total_start = time.perf_counter()

        # precompute geminal inverses per walker for fast updates across projections
        def _compute_initial_A_inv_n(r_up_carts, r_dn_carts):
            geminal = compute_geminal_all_elements(
                geminal_data=self.__hamiltonian_data.wavefunction_data.geminal_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )
            lu, piv = jsp_linalg.lu_factor(geminal)
            return jsp_linalg.lu_solve((lu, piv), jnp.eye(geminal.shape[0], dtype=geminal.dtype))

        self.__latest_A_old_inv = vmap(_compute_initial_A_inv_n, in_axes=(0, 0))(
            self.__latest_r_up_carts, self.__latest_r_dn_carts
        )

        # projection function.
        @jit
        def _generate_rotation_matrix_n(alpha, beta, gamma):
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

        @partial(jit, static_argnums=(6, 7, 8))
        def _projection_n(
            init_w_L: float,
            init_r_up_carts: jnpt.ArrayLike,
            init_r_dn_carts: jnpt.ArrayLike,
            init_A_old_inv: jnpt.ArrayLike,
            init_jax_PRNG_key: jnpt.ArrayLike,
            E_scf: float,
            num_mcmc_per_measurement: int,
            random_discretized_mesh: bool,
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
                random_discretized_mesh (bool): Flag for the random discretization mesh in the kinetic part.
                non_local_move (bool): treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
                alat (float): discretized grid length (bohr)
                hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data

            Returns:
                latest_w_L (float): weight after the final projection
                latest_r_up_carts (N_e^up, 3) after the final projection
                latest_r_dn_carts (N_e^dn, 3) after the final projection
                latest_A_old_inv: cached inverse geminal matrix after the final projection
                latest_RT (3, 3) rotation matrix used in the last projection
                latest_V_diag (float): diagonal part of H (importance sampled) at the last projection
                latest_V_nondiag (float): non-diagonal part of H (importance sampled) at the last projection
            """

            @jit
            def _body_fun_n(i, carry):
                (
                    w_L,
                    r_up_carts,
                    r_dn_carts,
                    RT,
                    A_old_inv,
                    _,
                    _,
                ) = carry

                # compute diagonal elements, kinetic part
                diagonal_kinetic_part = 3.0 / (2.0 * alat**2) * (len(r_up_carts) + len(r_dn_carts))

                # compute continuum kinetic energy
                diagonal_kinetic_continuum_elements_up, diagonal_kinetic_continuum_elements_dn = (
                    compute_kinetic_energy_all_elements_fast_update(
                        wavefunction_data=hamiltonian_data.wavefunction_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                        geminal_inverse=A_old_inv,
                    )
                )

                # generate a random rotation matrix
                rot_key = rotation_keys[i]
                if random_discretized_mesh:
                    alpha, beta, gamma = jax.random.uniform(
                        rot_key, shape=(3,), minval=-2 * jnp.pi, maxval=2 * jnp.pi
                    )  # Rotation angle around the x,y,z-axis (in radians)
                else:
                    alpha, beta, gamma = (0.0, 0.0, 0.0)
                R = _generate_rotation_matrix_n(alpha, beta, gamma)  # Rotate in the order x -> y -> z

                # compute discretized kinetic energy and mesh (with a random rotation)
                mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                    compute_discretized_kinetic_energy_fast_update(
                        alat=alat,
                        wavefunction_data=hamiltonian_data.wavefunction_data,
                        A_old_inv=A_old_inv,
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
                diagonal_bare_coulomb_part_el_el = compute_bare_coulomb_potential_el_el(
                    r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
                )

                # compute diagonal elements, ion-ion
                diagonal_bare_coulomb_part_ion_ion = compute_bare_coulomb_potential_ion_ion(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data
                )

                # compute diagonal elements, el-ion
                diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                    compute_bare_coulomb_potential_el_ion_element_wise(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )
                )

                # compute diagonal elements, el-ion, discretized
                (
                    diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                    diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
                ) = compute_discretized_bare_coulomb_potential_el_ion_element_wise(
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
                    diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    if non_local_move == "tmove":
                        # ecp non-local (t-move)
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=False,
                                A_old_inv=A_old_inv,
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
                            compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=True,
                                A_old_inv=A_old_inv,
                                RT=R.T,
                            )
                        )

                        V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                        diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))

                        Jastrow_ratio = compute_ratio_Jastrow_part(
                            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                            old_r_up_carts=r_up_carts,
                            old_r_dn_carts=r_dn_carts,
                            new_r_up_carts_arr=mesh_non_local_ecp_part_r_up_carts,
                            new_r_dn_carts_arr=mesh_non_local_ecp_part_r_dn_carts,
                        )

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
                move_key = move_keys[i]
                cdf = jnp.cumsum(non_diagonal_move_probabilities)
                random_value = jax.random.uniform(move_key, minval=0.0, maxval=1.0)
                k = jnp.searchsorted(cdf, random_value)
                proposed_r_up_carts = non_diagonal_move_mesh_r_up_carts[k]
                proposed_r_dn_carts = non_diagonal_move_mesh_r_dn_carts[k]

                num_up_electrons = r_up_carts.shape[0]
                num_dn_electrons = r_dn_carts.shape[0]

                if num_up_electrons == 0:
                    has_up_move = False
                    up_index = 0
                else:
                    up_diff = jnp.any(r_up_carts != proposed_r_up_carts, axis=1)
                    has_up_move = jnp.any(up_diff)
                    up_index = jnp.argmax(up_diff)

                if num_dn_electrons == 0:
                    has_dn_move = False
                    dn_index = 0
                else:
                    dn_diff = jnp.any(r_dn_carts != proposed_r_dn_carts, axis=1)
                    has_dn_move = jnp.any(dn_diff)
                    dn_index = jnp.argmax(dn_diff)

                def _update_inv_up_n(_):
                    v = (
                        compute_geminal_up_one_row_elements(
                            geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                            r_up_cart=jnp.reshape(proposed_r_up_carts[up_index], (1, 3)),
                            r_dn_carts=r_dn_carts,
                        )
                        - compute_geminal_up_one_row_elements(
                            geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                            r_up_cart=jnp.reshape(r_up_carts[up_index], (1, 3)),
                            r_dn_carts=r_dn_carts,
                        )
                    )[:, None]
                    u = jax.nn.one_hot(up_index, num_up_electrons)[:, None]
                    Ainv_u = A_old_inv @ u
                    vT_Ainv = v.T @ A_old_inv
                    det_ratio = 1.0 + (v.T @ Ainv_u)[0, 0]
                    return A_old_inv - (Ainv_u @ vT_Ainv) / det_ratio

                def _no_update_n(_):
                    return A_old_inv

                if num_dn_electrons == 0:
                    _update_inv_dn_n = _no_update_n
                else:

                    def _update_inv_dn_n(_):
                        u = (
                            compute_geminal_dn_one_column_elements(
                                geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                                r_up_carts=r_up_carts,
                                r_dn_cart=jnp.reshape(proposed_r_dn_carts[dn_index], (1, 3)),
                            )
                            - compute_geminal_dn_one_column_elements(
                                geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                                r_up_carts=r_up_carts,
                                r_dn_cart=jnp.reshape(r_dn_carts[dn_index], (1, 3)),
                            )
                        )[:, None]
                        v = jax.nn.one_hot(dn_index, num_up_electrons)[:, None]
                        Ainv_u = A_old_inv @ u
                        vT_Ainv = v.T @ A_old_inv
                        det_ratio = 1.0 + (v.T @ Ainv_u)[0, 0]
                        return A_old_inv - (Ainv_u @ vT_Ainv) / det_ratio

                if num_up_electrons == 0:
                    A_new_inv = A_old_inv
                else:
                    A_new_inv = lax.cond(
                        has_up_move,
                        _update_inv_up_n,
                        lambda __: lax.cond(has_dn_move, _update_inv_dn_n, _no_update_n, operand=None),
                        operand=None,
                    )

                r_up_carts = proposed_r_up_carts
                r_dn_carts = proposed_r_dn_carts

                carry = (
                    w_L,
                    r_up_carts,
                    r_dn_carts,
                    R.T,
                    A_new_inv,
                    diagonal_sum_hamiltonian,
                    non_diagonal_sum_hamiltonian,
                )
                return carry

            def _split_step_keys(key, num_steps):
                def _split_body(current_key, _):
                    current_key, rot_key = jax.random.split(current_key)
                    current_key, move_key = jax.random.split(current_key)
                    return current_key, (rot_key, move_key)

                return lax.scan(_split_body, key, xs=None, length=num_steps)

            latest_jax_PRNG_key, (rotation_keys, move_keys) = _split_step_keys(init_jax_PRNG_key, num_mcmc_per_measurement)

            (
                latest_w_L,
                latest_r_up_carts,
                latest_r_dn_carts,
                latest_RT,
                latest_A_old_inv,
                latest_diagonal_sum_hamiltonian,
                latest_non_diagonal_sum_hamiltonian,
            ) = jax.lax.fori_loop(
                0,
                num_mcmc_per_measurement,
                _body_fun_n,
                (
                    init_w_L,
                    init_r_up_carts,
                    init_r_dn_carts,
                    jnp.eye(3),
                    init_A_old_inv,
                    jnp.asarray(0.0),
                    jnp.asarray(0.0),
                ),
            )

            return (
                latest_w_L,
                latest_r_up_carts,
                latest_r_dn_carts,
                latest_A_old_inv,
                latest_jax_PRNG_key,
                latest_RT,
                latest_diagonal_sum_hamiltonian,
                latest_non_diagonal_sum_hamiltonian,
            )

        @partial(jit, static_argnums=(4, 6))
        def _compute_V_elements_n(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            RT: jnpt.ArrayLike,
            non_local_move: bool,
            alat: float,
            use_fast_update: bool = True,
        ):
            """Compute V elements."""
            #''' coulomb reguralization
            # compute diagonal elements, kinetic part
            diagonal_kinetic_part = 3.0 / (2.0 * alat**2) * (len(r_up_carts) + len(r_dn_carts))

            # compute continuum kinetic energy
            diagonal_kinetic_continuum_elements_up, diagonal_kinetic_continuum_elements_dn = (
                compute_kinetic_energy_all_elements(
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            if use_fast_update:
                # precompute geminal inverse for fast updates (single-electron moves)
                geminal = compute_geminal_all_elements(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
                lu, piv = jsp_linalg.lu_factor(geminal)
                A_old_inv = jsp_linalg.lu_solve((lu, piv), jnp.eye(geminal.shape[0], dtype=geminal.dtype))

                # compute discretized kinetic energy and mesh (with a random rotation)
                _, _, elements_non_diagonal_kinetic_part = compute_discretized_kinetic_energy_fast_update(
                    alat=alat,
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    A_old_inv=A_old_inv,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                    RT=RT,
                )
            else:
                A_old_inv = None
                # compute discretized kinetic energy and mesh (with a random rotation)
                _, _, elements_non_diagonal_kinetic_part = compute_discretized_kinetic_energy(
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
            diagonal_bare_coulomb_part_el_el = compute_bare_coulomb_potential_el_el(
                r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
            )

            # compute diagonal elements, ion-ion
            diagonal_bare_coulomb_part_ion_ion = compute_bare_coulomb_potential_ion_ion(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data
            )

            # compute diagonal elements, el-ion
            diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                compute_bare_coulomb_potential_el_ion_element_wise(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # compute diagonal elements, el-ion, discretized
            (
                diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
            ) = compute_discretized_bare_coulomb_potential_el_ion_element_wise(
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
                diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

                if non_local_move == "tmove":
                    # ecp non-local (t-move)
                    if use_fast_update:
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=False,
                                A_old_inv=A_old_inv,
                                RT=RT,
                            )
                        )
                    else:
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors(
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
                    if use_fast_update:
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=True,
                                A_old_inv=A_old_inv,
                                RT=RT,
                            )
                        )
                    else:
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors(
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

                    if use_fast_update:
                        Jastrow_ratio = compute_ratio_Jastrow_part(
                            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                            old_r_up_carts=r_up_carts,
                            old_r_dn_carts=r_dn_carts,
                            new_r_up_carts_arr=mesh_non_local_ecp_part_r_up_carts,
                            new_r_dn_carts_arr=mesh_non_local_ecp_part_r_dn_carts,
                        )
                    else:
                        Jastrow_ref = compute_Jastrow_part(
                            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                        )
                        Jastrow_on_mesh = vmap(compute_Jastrow_part, in_axes=(None, 0, 0))(
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

        @partial(jit, static_argnums=(4, 6))
        def _compute_local_energy_n(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            RT: jnpt.ArrayLike,
            non_local_move: bool,
            alat: float,
            use_fast_update: bool = True,
        ):
            V_diag, V_nondiag = _compute_V_elements_n(
                hamiltonian_data=hamiltonian_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                RT=RT,
                non_local_move=non_local_move,
                alat=alat,
                use_fast_update=use_fast_update,
            )
            return V_diag + V_nondiag

        # projection compilation.
        start_init = time.perf_counter()
        logger.info("Start compilation of the GFMC projection funciton.")
        logger.info("  Compilation is in progress...")
        w_L_list = jnp.array([1.0 for _ in range(self.__num_walkers)])
        (
            _,
            _,
            _,
            _,
            _,
            RTs,
            _,
            _,
        ) = vmap(_projection_n, in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None))(
            w_L_list,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
            self.__latest_A_old_inv,
            self.__jax_PRNG_key_list,
            self.__E_scf,
            self.__num_mcmc_per_measurement,
            self.__random_discretized_mesh,
            self.__non_local_move,
            self.__alat,
            self.__hamiltonian_data,
        )

        # compile the e_L recomputation path for debug parity
        _, _ = vmap(_compute_V_elements_n, in_axes=(None, 0, 0, 0, None, None, None))(
            self.__hamiltonian_data,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
            RTs,
            self.__non_local_move,
            self.__alat,
            True,
        )

        if self.__comput_position_deriv:
            _, _, _ = vmap(grad(_compute_local_energy_n, argnums=(0, 1, 2)), in_axes=(None, 0, 0, 0, None, None, None))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                RTs,
                self.__non_local_move,
                self.__alat,
                False,
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

            start_projection = time.perf_counter()

            # projection loop
            (
                w_L_list,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                self.__latest_A_old_inv,
                self.__jax_PRNG_key_list,
                latest_RTs,
                V_diag_list,
                V_nondiag_list,
            ) = vmap(_projection_n, in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None))(
                w_L_list,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                self.__latest_A_old_inv,
                self.__jax_PRNG_key_list,
                self.__E_scf,
                self.__num_mcmc_per_measurement,
                self.__random_discretized_mesh,
                self.__non_local_move,
                self.__alat,
                self.__hamiltonian_data,
            )

            # sync. jax arrays computations.
            w_L_list.block_until_ready()
            self.__latest_r_up_carts.block_until_ready()
            self.__latest_r_dn_carts.block_until_ready()
            self.__latest_A_old_inv.block_until_ready()
            self.__jax_PRNG_key_list.block_until_ready()

            end_projection = time.perf_counter()
            timer_projection_total += end_projection - start_projection
            logger.devel(f"    timer_projection_total = {(end_projection - start_projection) * 1e3:.2f} msec.")

            # evaluate observables
            start_e_L = time.perf_counter()
            V_diag_list, V_nondiag_list = vmap(_compute_V_elements_n, in_axes=(None, 0, 0, 0, None, None, None))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                latest_RTs,
                self.__non_local_move,
                self.__alat,
                True,
            )
            e_L_list = V_diag_list + V_nondiag_list
            e_L_list.block_until_ready()

            end_e_L = time.perf_counter()
            timer_e_L += end_e_L - start_e_L

            # atomic force related
            if self.__comput_position_deriv:
                start = time.perf_counter()
                grad_e_L_h, grad_e_L_r_up, grad_e_L_r_dn = vmap(
                    grad(_compute_local_energy_n, argnums=(0, 1, 2)), in_axes=(None, 0, 0, 0, None, None, None)
                )(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    latest_RTs,
                    self.__non_local_move,
                    self.__alat,
                    False,
                )
                grad_e_L_r_up.block_until_ready()
                grad_e_L_r_dn.block_until_ready()
                end = time.perf_counter()
                timer_de_L_dR_dr += end - start
                logger.devel(f"    timer_de_L_dR_dr = {(end - start) * 1e3:.2f} msec.")
                # grad_e_L_r_up and grad_e_L_r_dn are jax arrays, so we need to convert them to numpy arrays

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

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_nn_data is not None:
                    grad_e_L_R += grad_e_L_h.wavefunction_data.jastrow_data.jastrow_nn_data.structure_data.positions

                start = time.perf_counter()
                grad_ln_Psi_h, grad_ln_Psi_r_up, grad_ln_Psi_r_dn = vmap(
                    grad(evaluate_ln_wavefunction, argnums=(0, 1, 2)), in_axes=(None, 0, 0)
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

                if self.__hamiltonian_data.wavefunction_data.jastrow_data.jastrow_nn_data is not None:
                    grad_ln_Psi_dR += grad_ln_Psi_h.jastrow_data.jastrow_nn_data.structure_data.positions

                omega_up = vmap(evaluate_swct_omega, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_up_carts,
                )

                omega_dn = vmap(evaluate_swct_omega, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_dn_carts,
                )

                grad_omega_dr_up = vmap(evaluate_swct_domega, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_up_carts,
                )

                grad_omega_dr_dn = vmap(evaluate_swct_domega, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_dn_carts,
                )

                omega_up.block_until_ready()
                omega_dn.block_until_ready()
                grad_omega_dr_up.block_until_ready()
                grad_omega_dr_dn.block_until_ready()

            # Barrier before MPI operation
            start_mpi_barrier = time.perf_counter()
            mpi_comm.Barrier()
            end_mpi_barrier = time.perf_counter()
            timer_mpi_barrier += end_mpi_barrier - start_mpi_barrier

            # Branching starts
            start_collection = time.perf_counter()

            # random number for the later use
            """ very slow w/o jax-jit!!
            self.__jax_PRNG_key, subkey = jax.random.split(self.__jax_PRNG_key)
            zeta = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
            """
            if mpi_rank == 0:
                zeta = float(np.random.random())
            else:
                zeta = None
            zeta = mpi_comm.bcast(zeta, root=0)

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

            # sum
            nw_sum = len(w_L_latest)
            w_L_sum = np.sum(w_L_latest)
            w_L_weighted_sum = np.sum(w_L_latest / V_diag_E_latest)
            e_L_weighted_sum = np.sum(w_L_latest / V_diag_E_latest * e_L_latest)
            e_L2_weighted_sum = np.sum(w_L_latest / V_diag_E_latest * e_L_latest**2)
            if self.__comput_position_deriv:
                grad_e_L_r_up_weighted_sum = np.einsum("i,ijk->jk", w_L_latest / V_diag_E_latest, grad_e_L_r_up_latest)
                grad_e_L_r_dn_weighted_sum = np.einsum("i,ijk->jk", w_L_latest / V_diag_E_latest, grad_e_L_r_dn_latest)
                grad_e_L_R_weighted_sum = np.einsum("i,ijk->jk", w_L_latest / V_diag_E_latest, grad_e_L_R_latest)
                grad_ln_Psi_r_up_weighted_sum = np.einsum("i,ijk->jk", w_L_latest / V_diag_E_latest, grad_ln_Psi_r_up_latest)
                grad_ln_Psi_r_dn_weighted_sum = np.einsum("i,ijk->jk", w_L_latest / V_diag_E_latest, grad_ln_Psi_r_dn_latest)
                grad_ln_Psi_dR_weighted_sum = np.einsum("i,ijk->jk", w_L_latest / V_diag_E_latest, grad_ln_Psi_dR_latest)
                omega_up_weighted_sum = np.einsum("i,ijk->jk", w_L_latest / V_diag_E_latest, omega_up_latest)
                omega_dn_weighted_sum = np.einsum("i,ijk->jk", w_L_latest / V_diag_E_latest, omega_dn_latest)
                grad_omega_dr_up_weighted_sum = np.einsum("i,ijk->jk", w_L_latest / V_diag_E_latest, grad_omega_dr_up_latest)
                grad_omega_dr_dn_weighted_sum = np.einsum("i,ijk->jk", w_L_latest / V_diag_E_latest, grad_omega_dr_dn_latest)
            # reduce
            nw_sum = mpi_comm.reduce(nw_sum, op=MPI.SUM, root=0)
            w_L_sum = mpi_comm.reduce(w_L_sum, op=MPI.SUM, root=0)
            w_L_weighted_sum = mpi_comm.reduce(w_L_weighted_sum, op=MPI.SUM, root=0)
            e_L_weighted_sum = mpi_comm.reduce(e_L_weighted_sum, op=MPI.SUM, root=0)
            e_L2_weighted_sum = mpi_comm.reduce(e_L2_weighted_sum, op=MPI.SUM, root=0)
            if self.__comput_position_deriv:
                grad_e_L_r_up_weighted_sum = mpi_comm.reduce(grad_e_L_r_up_weighted_sum, op=MPI.SUM, root=0)
                grad_e_L_r_dn_weighted_sum = mpi_comm.reduce(grad_e_L_r_dn_weighted_sum, op=MPI.SUM, root=0)
                grad_e_L_R_weighted_sum = mpi_comm.reduce(grad_e_L_R_weighted_sum, op=MPI.SUM, root=0)
                grad_ln_Psi_r_up_weighted_sum = mpi_comm.reduce(grad_ln_Psi_r_up_weighted_sum, op=MPI.SUM, root=0)
                grad_ln_Psi_r_dn_weighted_sum = mpi_comm.reduce(grad_ln_Psi_r_dn_weighted_sum, op=MPI.SUM, root=0)
                grad_ln_Psi_dR_weighted_sum = mpi_comm.reduce(grad_ln_Psi_dR_weighted_sum, op=MPI.SUM, root=0)
                omega_up_weighted_sum = mpi_comm.reduce(omega_up_weighted_sum, op=MPI.SUM, root=0)
                omega_dn_weighted_sum = mpi_comm.reduce(omega_dn_weighted_sum, op=MPI.SUM, root=0)
                grad_omega_dr_up_weighted_sum = mpi_comm.reduce(grad_omega_dr_up_weighted_sum, op=MPI.SUM, root=0)
                grad_omega_dr_dn_weighted_sum = mpi_comm.reduce(grad_omega_dr_dn_weighted_sum, op=MPI.SUM, root=0)

            if mpi_rank == 0:
                # averaged
                w_L_averaged = w_L_sum / nw_sum
                e_L_averaged = e_L_weighted_sum / w_L_weighted_sum
                e_L2_averaged = e_L2_weighted_sum / w_L_weighted_sum
                if self.__comput_position_deriv:
                    grad_e_L_r_up_averaged = grad_e_L_r_up_weighted_sum / w_L_weighted_sum
                    grad_e_L_r_dn_averaged = grad_e_L_r_dn_weighted_sum / w_L_weighted_sum
                    grad_e_L_R_averaged = grad_e_L_R_weighted_sum / w_L_weighted_sum
                    grad_ln_Psi_r_up_averaged = grad_ln_Psi_r_up_weighted_sum / w_L_weighted_sum
                    grad_ln_Psi_r_dn_averaged = grad_ln_Psi_r_dn_weighted_sum / w_L_weighted_sum
                    grad_ln_Psi_dR_averaged = grad_ln_Psi_dR_weighted_sum / w_L_weighted_sum
                    omega_up_averaged = omega_up_weighted_sum / w_L_weighted_sum
                    omega_dn_averaged = omega_dn_weighted_sum / w_L_weighted_sum
                    grad_omega_dr_up_averaged = grad_omega_dr_up_weighted_sum / w_L_weighted_sum
                    grad_omega_dr_dn_averaged = grad_omega_dr_dn_weighted_sum / w_L_weighted_sum
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

            mpi_comm.Barrier()

            end_collection = time.perf_counter()
            timer_collection += end_collection - start_collection
            logger.devel(f"    timer_collection = {(end_collection - start_collection) * 1e3:.2f} msec.")

            # branching
            start_reconfiguration = time.perf_counter()
            latest_r_up_carts_before_branching = np.array(self.__latest_r_up_carts)
            latest_r_dn_carts_before_branching = np.array(self.__latest_r_dn_carts)

            #########################################
            # 1. Gather only the weights to MPI_rank=0 and perform branching calculation
            #########################################
            start_ = time.perf_counter()

            # Each process computes the sum of its local walker weights.
            local_weight_sum = np.sum(w_L_latest)

            # Use pickle‐based allreduce here (allowed for this part)
            global_weight_sum = mpi_comm.allreduce(local_weight_sum, op=MPI.SUM)

            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.1 = {(end_ - start_) * 1e3:.3f} msec.")

            start_ = time.perf_counter()

            # Compute the local probabilities for each walker.
            local_probabilities = w_L_latest / global_weight_sum

            # Compute the local cumulative probabilities.
            local_cumprob = np.cumsum(local_probabilities)
            local_sum_arr = np.array(np.sum(local_probabilities), dtype=np.float64)
            offset_arr = np.zeros(1, dtype=np.float64)
            mpi_comm.Exscan([local_sum_arr, MPI.DOUBLE], [offset_arr, MPI.DOUBLE], op=MPI.SUM)
            if mpi_rank == 0:
                offset = 0.0
            else:
                offset = float(offset_arr[0])
            local_cumprob += offset

            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.2 = {(end_ - start_) * 1e3:.3f} msec.")

            start_ = time.perf_counter()

            # Gather the local cumulative probability arrays from all processes.
            total_walkers = self.num_walkers * mpi_size
            global_cumprob = np.empty(total_walkers, dtype=np.float64)
            mpi_comm.Allgather([local_cumprob, MPI.DOUBLE], [global_cumprob, MPI.DOUBLE])
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.3 = {(end_ - start_) * 1e3:.3f} msec.")

            # Total number of walkers across all processes.
            # Compute index range for this rank
            start_ = time.perf_counter()
            start_idx = mpi_rank * self.num_walkers
            end_idx = start_idx + self.num_walkers

            # Build only local z-array (length = self.num_walkers)
            z_local = (np.arange(start_idx, end_idx) + zeta) / total_walkers

            # Perform searchsorted and cast the result to int32
            local_chosen_indices = np.searchsorted(global_cumprob, z_local).astype(np.int32)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.4 = {(end_ - start_) * 1e3:.3f} msec.")

            # Gather all local_chosen_indices across ranks using MPI.INT
            start_ = time.perf_counter()

            # Allocate buffer to receive all ranks' indices (Number of indices per rank is identical on every rank)
            all_chosen_buf = np.empty(self.num_walkers * mpi_size, dtype=np.int32)

            # Perform the all-gather operation with 32-bit integers
            mpi_comm.Allgather([local_chosen_indices, MPI.INT], [all_chosen_buf, MPI.INT])

            # Use the gathered indices for global statistics
            chosen_walker_indices = all_chosen_buf
            num_survived_walkers = len(np.unique(chosen_walker_indices))
            num_killed_walkers = total_walkers - num_survived_walkers

            # Build the local assignment list of (source_rank, source_local_index)
            local_assignment = [
                (src_global_idx // self.num_walkers, src_global_idx % self.num_walkers)
                for src_global_idx in local_chosen_indices
            ]

            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.5 = {(end_ - start_) * 1e3:.3f} msec.")

            #########################################
            # 2. In each process, prepare for data exchange based on the new walker selection
            #########################################
            start_ = time.perf_counter()
            latest_r_up_carts_after_branching = np.empty_like(latest_r_up_carts_before_branching)
            latest_r_dn_carts_after_branching = np.empty_like(latest_r_dn_carts_before_branching)

            reqs = {}
            for dest_idx, (src_rank, src_local_idx) in enumerate(local_assignment):
                if src_rank == mpi_rank:
                    # Local copy: no communication needed
                    latest_r_up_carts_after_branching[dest_idx] = latest_r_up_carts_before_branching[src_local_idx]
                    latest_r_dn_carts_after_branching[dest_idx] = latest_r_dn_carts_before_branching[src_local_idx]
                else:
                    reqs.setdefault(src_rank, []).append((dest_idx, src_local_idx))
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 2 = {(end_ - start_) * 1e3:.3f} msec.")

            #########################################
            # 3. Exchange only the necessary walker data between processes using asynchronous communication
            #########################################

            # 3.1.1: Flatten `reqs` into an (N_req × 3) int32 array of triplets
            start_ = time.perf_counter()
            flat_list = [
                (src_rank, dest_idx, src_local_idx) for src_rank, pairs in reqs.items() for dest_idx, src_local_idx in pairs
            ]
            triplets = np.array(flat_list, dtype=np.int32) if flat_list else np.empty((0, 3), dtype=np.int32)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.1 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.2: Compute how many ints to send to each rank (3 ints per request)
            start_ = time.perf_counter()
            counts_per_rank = np.bincount(triplets[:, 0], minlength=mpi_size)  # # reqs per src_rank
            send_counts = (counts_per_rank * 3).astype(np.int32)  # # ints per src_rank
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.2 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.3: Post nonblocking Alltoall to exchange counts
            start_ = time.perf_counter()
            recv_counts = np.empty_like(send_counts)
            req_counts = mpi_comm.Ialltoall([send_counts, MPI.INT], [recv_counts, MPI.INT])
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.3 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.4: Build send_buf while counts exchange is in flight
            start_ = time.perf_counter()
            #   sort by src_rank so that each destination's data is contiguous
            order = np.argsort(triplets[:, 0], kind="mergesort") if triplets.size else np.empty(0, dtype=np.int32)
            sorted_tr = triplets[order]  # shape = (N_req, 3)
            send_buf = sorted_tr.ravel()  # shape = (N_req*3,)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.4 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.5: Wait for counts exchange to complete
            start_ = time.perf_counter()
            req_counts.Wait()
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.5 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.6: Build displacements for send/recv from counts
            start_ = time.perf_counter()
            send_displs = np.zeros_like(send_counts)
            send_displs[1:] = np.cumsum(send_counts)[:-1]
            recv_displs = np.zeros_like(recv_counts)
            recv_displs[1:] = np.cumsum(recv_counts)[:-1]
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.6 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.7: Allocate recv buffer of the exact size
            start_ = time.perf_counter()
            total_recv = int(np.sum(recv_counts))
            recv_buf = np.empty(total_recv, dtype=np.int32)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.7 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.8: Post blocking Alltoallv to exchange the triplets
            start_ = time.perf_counter()
            mpi_comm.Alltoallv([send_buf, send_counts, send_displs, MPI.INT], [recv_buf, recv_counts, recv_displs, MPI.INT])
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.8 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.9: Wait for data to arrive and reconstruct per‐process request dicts
            start_ = time.perf_counter()
            all_reqs = []
            for p in range(mpi_size):
                off = recv_displs[p]
                cnt = recv_counts[p]
                block = recv_buf[off : off + cnt]
                if block.size == 0:
                    all_reqs.append({})
                    continue
                rec_tr = block.reshape(-1, 3)
                proc_dict = {}
                for sr, dest_idx, src_local_idx in rec_tr:
                    proc_dict.setdefault(int(sr), []).append((int(dest_idx), int(src_local_idx)))
                all_reqs.append(proc_dict)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.9 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.10: Filter out empty request dicts
            start_ = time.perf_counter()
            non_empty_all_reqs = [(p, rd) for p, rd in enumerate(all_reqs) if rd]
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.10 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-2. Build incoming_reqs: who needs data from me? ---
            start_ = time.perf_counter()
            incoming_reqs = [
                (p, src_local_idx, dest_idx)
                for p, proc_req in non_empty_all_reqs
                if p != mpi_rank
                for dest_idx, src_local_idx in proc_req.get(mpi_rank, [])
            ]
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.2 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-3. Post nonblocking receives using Irecv for both up and dn buffers. ---
            start_ = time.perf_counter()
            recv_buffers = {}
            recv_reqs_up = {}
            recv_reqs_dn = {}
            for src_rank, req_list in reqs.items():
                if not req_list:
                    continue
                count = len(req_list)
                shape = latest_r_up_carts_before_branching.shape[1:]
                buf_up = np.empty((count, *shape), dtype=latest_r_up_carts_before_branching.dtype)
                buf_dn = np.empty((count, *shape), dtype=latest_r_dn_carts_before_branching.dtype)
                recv_buffers[src_rank] = (buf_up, buf_dn)
                recv_reqs_up[src_rank] = mpi_comm.Irecv([buf_up, MPI.DOUBLE], source=src_rank, tag=200)
                recv_reqs_dn[src_rank] = mpi_comm.Irecv([buf_dn, MPI.DOUBLE], source=src_rank, tag=201)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.3 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-4. Prepare and post nonblocking sends using Isend. ---
            start_ = time.perf_counter()
            send_requests = []
            for dest_rank, group in groupby(sorted(incoming_reqs, key=lambda x: x[0]), key=lambda x: x[0]):
                idxs = [src_local for (_, src_local, _) in group]
                buf_up = latest_r_up_carts_before_branching[idxs]
                buf_dn = latest_r_dn_carts_before_branching[idxs]
                send_requests.append(mpi_comm.Isend([buf_up, MPI.DOUBLE], dest=dest_rank, tag=200))
                send_requests.append(mpi_comm.Isend([buf_dn, MPI.DOUBLE], dest=dest_rank, tag=201))
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.4 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-5. Wait for all nonblocking sends to complete. ---
            start_ = time.perf_counter()
            MPI.Request.Waitall(send_requests)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.5 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-6. Process the received walker data. ---
            start_ = time.perf_counter()
            for src_rank, req_list in reqs.items():
                if not req_list:
                    continue
                recv_reqs_up[src_rank].Wait()
                recv_reqs_dn[src_rank].Wait()
                buf_up, buf_dn = recv_buffers[src_rank]
                dest_idxs = [dest for (dest, _) in req_list]
                latest_r_up_carts_after_branching[dest_idxs] = buf_up
                latest_r_dn_carts_after_branching[dest_idxs] = buf_dn
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.6 = {(end_ - start_) * 1e3:.3f} msec.")

            # here update the walker positions!!
            self.__num_survived_walkers += num_survived_walkers
            self.__num_killed_walkers += num_killed_walkers
            self.__latest_r_up_carts = jnp.array(latest_r_up_carts_after_branching)
            self.__latest_r_dn_carts = jnp.array(latest_r_dn_carts_after_branching)
            self.__latest_A_old_inv = vmap(_compute_initial_A_inv_n, in_axes=(0, 0))(
                self.__latest_r_up_carts, self.__latest_r_dn_carts
            )

            mpi_comm.Barrier()

            end_reconfiguration = time.perf_counter()
            timer_reconfiguration += end_reconfiguration - start_reconfiguration
            logger.devel(f"    timer_reconfiguration total = {(end_reconfiguration - start_reconfiguration) * 1e3:.2f} msec.")

            # update E_scf
            start_update_E_scf = time.perf_counter()

            ## parameters for E_scf
            eq_steps = GFMC_ON_THE_FLY_WARMUP_STEPS
            num_gfmc_collect_steps = GFMC_ON_THE_FLY_COLLECT_STEPS
            num_gfmc_bin_blocks = GFMC_ON_THE_FLY_BIN_BLOCKS

            if mpi_rank == 0:
                if i_mcmc_step >= num_gfmc_collect_steps:
                    e_L = self.__stored_e_L[-1]
                    w_L = self.__stored_w_L[-num_gfmc_collect_steps - 1 : -1]
                    G_L = np.prod(w_L, axis=0)
                    self.__G_L.append(G_L)
                    self.__G_e_L.append(G_L * e_L)

            if (i_mcmc_step + 1) % mcmc_interval == 0:
                if i_mcmc_step > eq_steps:
                    if mpi_rank == 0:
                        num_gfmc_warmup_steps = np.minimum(eq_steps, i_mcmc_step - eq_steps)
                        logger.debug(f"    Computing E_scf at step {i_mcmc_step}.")
                        G_eq = np.array(self.__G_L[num_gfmc_warmup_steps:])
                        G_e_L_eq = np.array(self.__G_e_L[num_gfmc_warmup_steps:])
                        G_e_L_split = np.array_split(G_e_L_eq, num_gfmc_bin_blocks)
                        G_e_L_binned = np.array([np.sum(G_e_L_list) for G_e_L_list in G_e_L_split])
                        G_split = np.array_split(G_eq, num_gfmc_bin_blocks)
                        G_binned = np.array([np.sum(G_list) for G_list in G_split])
                        G_e_L_binned_sum = np.sum(G_e_L_binned)
                        G_binned_sum = np.sum(G_binned)
                        E_jackknife = [
                            (G_e_L_binned_sum - G_e_L_binned[m]) / (G_binned_sum - G_binned[m])
                            for m in range(num_gfmc_bin_blocks)
                        ]
                        E_mean = np.average(E_jackknife)
                        E_std = np.sqrt(num_gfmc_bin_blocks - 1) * np.std(E_jackknife)
                        E_mean = float(E_mean)
                        E_std = float(E_std)
                    else:
                        E_mean = None
                        E_std = None

                    E_mean = mpi_comm.bcast(E_mean, root=0)
                    E_std = mpi_comm.bcast(E_std, root=0)

                    self.__E_scf = E_mean
                    E_scf_std = E_std

                    logger.debug(f"    Updated E_scf = {self.__E_scf:.5f} +- {E_scf_std:.5f} Ha.")
                else:
                    logger.debug(f"    Init E_scf = {self.__E_scf:.5f} Ha. Being equilibrated.")

            mpi_comm.Barrier()
            end_update_E_scf = time.perf_counter()
            timer_update_E_scf += end_update_E_scf - start_update_E_scf
            logger.devel(f"    timer_update_E_scf = {(end_update_E_scf - start_update_E_scf) * 1e3:.2f} msec.")

            gfmc_current = time.perf_counter()

            if max_time < gfmc_current - gfmc_total_start:
                logger.info(f"  Stopping... Max_time = {max_time} sec. exceeds.")
                logger.info("  Break the branching loop.")
                break

            # check toml file (stop flag)
            if os.path.isfile(toml_filename):
                dict_toml = toml.load(open(toml_filename))
                try:
                    stop_flag = dict_toml["external_control"]["stop"]
                except KeyError:
                    stop_flag = False
                if stop_flag:
                    logger.info(f"  Stopping... stop_flag in {toml_filename} is true.")
                    logger.info("  Break the mcmc loop.")
                    break

            # count up, here is the end of the branching step.
            num_mcmc_done += 1

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
            + timer_mpi_barrier
            + timer_collection
            + timer_reconfiguration
            + timer_update_E_scf
        )

        # remove the toml file
        mpi_comm.Barrier()
        if mpi_rank == 0:
            if os.path.isfile(toml_filename):
                logger.info(f"Delete {toml_filename}")
                os.remove(toml_filename)

        # net GFMC time
        timer_net_gfmc_total = timer_gfmc_total - timer_projection_init

        # average among MPI processes
        ave_timer_gfmc_total = mpi_comm.allreduce(timer_gfmc_total, op=MPI.SUM) / mpi_size
        ave_timer_projection_init = mpi_comm.allreduce(timer_projection_init, op=MPI.SUM) / mpi_size
        ave_timer_net_gfmc_total = mpi_comm.allreduce(timer_net_gfmc_total, op=MPI.SUM) / mpi_size
        ave_timer_projection_total = mpi_comm.allreduce(timer_projection_total, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_e_L = mpi_comm.allreduce(timer_e_L, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_de_L_dR_dr = mpi_comm.allreduce(timer_de_L_dR_dr, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_dln_Psi_dR_dr = mpi_comm.allreduce(timer_dln_Psi_dR_dr, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_dln_Psi_dc = mpi_comm.allreduce(timer_dln_Psi_dc, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_de_L_dc = mpi_comm.allreduce(timer_de_L_dc, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_mpi_barrier = mpi_comm.allreduce(timer_mpi_barrier, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_collection = mpi_comm.allreduce(timer_collection, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_reconfiguration = mpi_comm.allreduce(timer_reconfiguration, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_update_E_scf = mpi_comm.allreduce(timer_update_E_scf, op=MPI.SUM) / mpi_size / num_mcmc_done
        ave_timer_misc = mpi_comm.allreduce(timer_misc, op=MPI.SUM) / mpi_size / num_mcmc_done
        sum_killed_walkers = mpi_comm.allreduce(self.__num_killed_walkers, op=MPI.SUM)
        sum_survived_walkers = mpi_comm.allreduce(self.__num_survived_walkers, op=MPI.SUM)

        logger.info(f"Total GFMC time for {num_mcmc_done} branching steps = {ave_timer_gfmc_total: .3f} sec.")
        logger.info(f"Pre-compilation time for GFMC = {ave_timer_projection_init: .3f} sec.")
        logger.info(f"Net GFMC time without pre-compilations = {ave_timer_net_gfmc_total: .3f} sec.")
        logger.info(f"Elapsed times per branching, averaged over {num_mcmc_done} branching steps.")
        logger.info(f"  Projection between branching = {ave_timer_projection_total * 10**3: .3f} msec.")
        logger.info(f"  Time for computing e_L = {ave_timer_e_L * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/dR and de_L/dr = {ave_timer_de_L_dR_dr * 10**3:.2f} msec.")
        logger.info(f"  Time for computing dln_Psi/dR and dln_Psi/dr = {ave_timer_dln_Psi_dR_dr * 10**3:.2f} msec.")
        logger.info(f"  Time for computing dln_Psi/dc = {ave_timer_dln_Psi_dc * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/dc = {ave_timer_de_L_dc * 10**3:.2f} msec.")
        logger.info(f"  Time for MPI barrier before branching = {ave_timer_mpi_barrier * 10**3:.2f} msec.")
        logger.info(f"  Time for walker observable collections time per branching = {ave_timer_collection * 10**3: .3f} msec.")
        logger.info(f"  Time for walker reconfiguration time per branching = {ave_timer_reconfiguration * 10**3: .3f} msec.")
        logger.info(f"  Time for updating E_scf = {ave_timer_update_E_scf * 10**3:.2f} msec.")
        logger.info(f"  Time for misc. (others) = {ave_timer_misc * 10**3:.2f} msec.")
        logger.info(
            f"Survived walkers ratio = {sum_survived_walkers / (sum_survived_walkers + sum_killed_walkers) * 100:.2f} %. Ideal is ~ 98 %. Adjust num_mcmc_per_measurement."
        )
        logger.info("")

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
        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if num_mcmc_warmup_steps < GFMC_MIN_WARMUP_STEPS:
            logger.warning(f"num_mcmc_warmup_steps should be larger than {GFMC_MIN_WARMUP_STEPS}")
        if num_mcmc_bin_blocks < GFMC_MIN_BIN_BLOCKS:
            logger.warning(f"num_mcmc_bin_blocks should be larger than {GFMC_MIN_BIN_BLOCKS}")

        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if self.mcmc_counter < num_mcmc_warmup_steps:
            logger.error("mcmc_counter should be larger than num_mcmc_warmup_steps")
            raise ValueError
        if self.mcmc_counter - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
            logger.error("(mcmc_counter - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")
            raise ValueError

        if num_mcmc_bin_blocks < mpi_size or mpi_size == 1:
            if mpi_rank == 0:
                e_L = self.e_L[num_mcmc_warmup_steps:]
                e_L2 = self.e_L2[num_mcmc_warmup_steps:]
                w_L = self.w_L[num_mcmc_warmup_steps:]
                w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
                w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
                w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
                w_L_e_L2_split = np.array_split(w_L * e_L2, num_mcmc_bin_blocks, axis=0)
                w_L_e_L2_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L2_split]))

                w_L_binned_local = np.array(w_L_binned)
                w_L_e_L_binned_local = np.array(w_L_e_L_binned)
                w_L_e_L2_binned_local = np.array(w_L_e_L2_binned)

                ## local sum
                w_L_binned_local_sum = np.sum(w_L_binned_local, axis=0)
                w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local, axis=0)
                w_L_e_L2_binned_local_sum = np.sum(w_L_e_L2_binned_local, axis=0)

                ## jackknie binned samples
                M_local = w_L_binned_local.size
                M_total = M_local

                E_jackknife_binned_local = np.array(
                    [
                        (w_L_e_L_binned_local_sum - w_L_e_L_binned_local[m]) / (w_L_binned_local_sum - w_L_binned_local[m])
                        for m in range(M_local)
                    ]
                )

                E2_jackknife_binned_local = np.array(
                    [
                        (w_L_e_L2_binned_local_sum - w_L_e_L2_binned_local[m]) / (w_L_binned_local_sum - w_L_binned_local[m])
                        for m in range(M_local)
                    ]
                )

                Var_jackknife_binned_local = E2_jackknife_binned_local - E_jackknife_binned_local**2

                # E: jackknife mean and std
                sum_E_local = np.sum(E_jackknife_binned_local)
                sumsq_E_local = np.sum(E_jackknife_binned_local**2)

                E_mean = sum_E_local / M_local
                E_var = (sumsq_E_local / M_local) - (sum_E_local / M_local) ** 2
                E_std = np.sqrt((M_local - 1) * E_var)

                # Var: jackknife mean and std
                sum_Var_local = np.sum(Var_jackknife_binned_local)
                sumsq_Var_local = np.sum(Var_jackknife_binned_local**2)

                Var_mean = sum_Var_local / M_total
                Var_var = (sumsq_Var_local / M_total) - (sum_Var_local / M_local) ** 2
                Var_std = np.sqrt((M_total - 1) * Var_var)

            else:
                E_mean = None
                E_std = None
                Var_mean = None
                Var_std = None

            # MPI broadcast
            E_mean = mpi_comm.bcast(E_mean, root=0)
            E_std = mpi_comm.bcast(E_std, root=0)
            Var_mean = mpi_comm.bcast(Var_mean, root=0)
            Var_std = mpi_comm.bcast(Var_std, root=0)

        else:
            if mpi_rank == 0:
                e_L = self.e_L[num_mcmc_warmup_steps:]
                e_L2 = self.e_L2[num_mcmc_warmup_steps:]
                w_L = self.w_L[num_mcmc_warmup_steps:]
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

            if mpi_rank == 0:
                w_L_binned_split = np.array_split(w_L_binned, mpi_size)
                w_L_e_L_binned_split = np.array_split(w_L_e_L_binned, mpi_size)
                w_L_e_L2_binned_split = np.array_split(w_L_e_L2_binned, mpi_size)
            else:
                w_L_binned_split = None
                w_L_e_L_binned_split = None
                w_L_e_L2_binned_split = None

            # scatter
            w_L_binned_local = mpi_comm.scatter(w_L_binned_split, root=0)
            w_L_e_L_binned_local = mpi_comm.scatter(w_L_e_L_binned_split, root=0)
            w_L_e_L2_binned_local = mpi_comm.scatter(w_L_e_L2_binned_split, root=0)

            w_L_binned_local = np.array(w_L_binned_local)
            w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
            w_L_e_L2_binned_local = np.array(w_L_e_L2_binned_local)

            ## local sum
            w_L_binned_local_sum = np.sum(w_L_binned_local, axis=0)
            w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local, axis=0)
            w_L_e_L2_binned_local_sum = np.sum(w_L_e_L2_binned_local, axis=0)

            ## glolbal sum
            w_L_binned_global_sum = np.empty_like(w_L_binned_local_sum)
            w_L_e_L_binned_global_sum = np.empty_like(w_L_e_L_binned_local_sum)
            w_L_e_L2_binned_global_sum = np.empty_like(w_L_e_L2_binned_local_sum)

            ## mpi Allreduce
            mpi_comm.Allreduce([w_L_binned_local_sum, MPI.DOUBLE], [w_L_binned_global_sum, MPI.DOUBLE], op=MPI.SUM)
            mpi_comm.Allreduce([w_L_e_L_binned_local_sum, MPI.DOUBLE], [w_L_e_L_binned_global_sum, MPI.DOUBLE], op=MPI.SUM)
            mpi_comm.Allreduce([w_L_e_L2_binned_local_sum, MPI.DOUBLE], [w_L_e_L2_binned_global_sum, MPI.DOUBLE], op=MPI.SUM)

            ## jackknie binned samples
            M_local = w_L_binned_local.size
            M_total = mpi_comm.allreduce(M_local, op=MPI.SUM)

            E_jackknife_binned_local = np.array(
                [
                    (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                    for m in range(M_local)
                ]
            )

            E2_jackknife_binned_local = np.array(
                [
                    (w_L_e_L2_binned_global_sum - w_L_e_L2_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                    for m in range(M_local)
                ]
            )

            Var_jackknife_binned_local = E2_jackknife_binned_local - E_jackknife_binned_local**2

            # E: jackknife mean and std
            sum_E_local = np.sum(E_jackknife_binned_local)
            sumsq_E_local = np.sum(E_jackknife_binned_local**2)

            sum_E_global = np.empty_like(sum_E_local)
            sumsq_E_global = np.empty_like(sumsq_E_local)

            mpi_comm.Allreduce([sum_E_local, MPI.DOUBLE], [sum_E_global, MPI.DOUBLE], op=MPI.SUM)
            mpi_comm.Allreduce([sumsq_E_local, MPI.DOUBLE], [sumsq_E_global, MPI.DOUBLE], op=MPI.SUM)

            E_mean = sum_E_global / M_total
            E_var = (sumsq_E_global / M_total) - (sum_E_global / M_total) ** 2
            E_std = np.sqrt((M_total - 1) * E_var)

            # Var: jackknife mean and std
            sum_Var_local = np.sum(Var_jackknife_binned_local)
            sumsq_Var_local = np.sum(Var_jackknife_binned_local**2)

            sum_Var_global = np.empty_like(sum_Var_local)
            sumsq_Var_global = np.empty_like(sumsq_Var_local)

            mpi_comm.Allreduce([sum_Var_local, MPI.DOUBLE], [sum_Var_global, MPI.DOUBLE], op=MPI.SUM)
            mpi_comm.Allreduce([sumsq_Var_local, MPI.DOUBLE], [sumsq_Var_global, MPI.DOUBLE], op=MPI.SUM)

            Var_mean = sum_Var_global / M_total
            Var_var = (sumsq_Var_global / M_total) - (sum_Var_global / M_total) ** 2
            Var_std = np.sqrt((M_total - 1) * Var_var)

        # return
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
        if num_mcmc_bin_blocks < mpi_size or mpi_size == 1:
            if mpi_rank == 0:
                w_L = self.w_L[num_mcmc_warmup_steps:]
                e_L = self.e_L[num_mcmc_warmup_steps:]
                de_L_dR = self.de_L_dR[num_mcmc_warmup_steps:]
                de_L_dr_up = self.de_L_dr_up[num_mcmc_warmup_steps:]
                de_L_dr_dn = self.de_L_dr_dn[num_mcmc_warmup_steps:]
                dln_Psi_dr_up = self.dln_Psi_dr_up[num_mcmc_warmup_steps:]
                dln_Psi_dr_dn = self.dln_Psi_dr_dn[num_mcmc_warmup_steps:]
                dln_Psi_dR = self.dln_Psi_dR[num_mcmc_warmup_steps:]
                omega_up = self.omega_up[num_mcmc_warmup_steps:]
                omega_dn = self.omega_dn[num_mcmc_warmup_steps:]
                domega_dr_up = self.domega_dr_up[num_mcmc_warmup_steps:]
                domega_dr_dn = self.domega_dr_dn[num_mcmc_warmup_steps:]

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

                # split and binning with multiple walkers
                w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
                w_L_force_HF_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_HF), num_mcmc_bin_blocks, axis=0)
                w_L_force_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_PP), num_mcmc_bin_blocks, axis=0)
                w_L_E_L_force_PP_split = np.array_split(
                    np.einsum("iw,iwjk->iwjk", w_L, E_L_force_PP), num_mcmc_bin_blocks, axis=0
                )

                # binned sum
                w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
                w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))

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

                w_L_binned_local = np.array(w_L_binned)
                w_L_e_L_binned_local = np.array(w_L_e_L_binned)
                w_L_force_HF_binned_local = np.array(w_L_force_HF_binned)
                w_L_force_PP_binned_local = np.array(w_L_force_PP_binned)
                w_L_E_L_force_PP_binned_local = np.array(w_L_E_L_force_PP_binned)

                ## local sum
                w_L_binned_local_sum = np.sum(w_L_binned_local, axis=0)
                w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local, axis=0)
                w_L_force_HF_binned_local_sum = np.sum(w_L_force_HF_binned_local, axis=0)
                w_L_force_PP_binned_local_sum = np.sum(w_L_force_PP_binned_local, axis=0)
                w_L_E_L_force_PP_binned_local_sum = np.sum(w_L_E_L_force_PP_binned_local, axis=0)

                ## jackknie binned samples
                M_local = w_L_binned_local.size

                force_HF_jn_local = -1.0 * np.array(
                    [
                        (w_L_force_HF_binned_local_sum - w_L_force_HF_binned_local[j])
                        / (w_L_binned_local_sum - w_L_binned_local[j])
                        for j in range(M_local)
                    ]
                )

                force_Pulay_jn_local = -2.0 * np.array(
                    [
                        (
                            (w_L_E_L_force_PP_binned_local_sum - w_L_E_L_force_PP_binned_local[j])
                            / (w_L_binned_local_sum - w_L_binned_local[j])
                            - (
                                (w_L_e_L_binned_local_sum - w_L_e_L_binned_local[j])
                                / (w_L_binned_local_sum - w_L_binned_local[j])
                                * (w_L_force_PP_binned_local_sum - w_L_force_PP_binned_local[j])
                                / (w_L_binned_local_sum - w_L_binned_local[j])
                            )
                        )
                        for j in range(M_local)
                    ]
                )

                force_jn_local = force_HF_jn_local + force_Pulay_jn_local

                sum_force_local = np.sum(force_jn_local, axis=0)
                sumsq_force_local = np.sum(force_jn_local**2, axis=0)

                ## mean and var = E[x^2] - (E[x])^2
                mean_force_global = sum_force_local / M_local
                var_force_global = (sumsq_force_local / M_local) - (sum_force_local / M_local) ** 2

                ## mean and std
                force_mean = mean_force_global
                force_std = np.sqrt((M_local - 1) * var_force_global)

            else:
                force_mean = None
                force_std = None

            # broadcast the results
            force_mean = mpi_comm.bcast(force_mean, root=0)
            force_std = mpi_comm.bcast(force_std, root=0)

        else:
            if mpi_rank == 0:
                w_L = self.w_L[num_mcmc_warmup_steps:]
                e_L = self.e_L[num_mcmc_warmup_steps:]
                de_L_dR = self.de_L_dR[num_mcmc_warmup_steps:]
                de_L_dr_up = self.de_L_dr_up[num_mcmc_warmup_steps:]
                de_L_dr_dn = self.de_L_dr_dn[num_mcmc_warmup_steps:]
                dln_Psi_dr_up = self.dln_Psi_dr_up[num_mcmc_warmup_steps:]
                dln_Psi_dr_dn = self.dln_Psi_dr_dn[num_mcmc_warmup_steps:]
                dln_Psi_dR = self.dln_Psi_dR[num_mcmc_warmup_steps:]
                omega_up = self.omega_up[num_mcmc_warmup_steps:]
                omega_dn = self.omega_dn[num_mcmc_warmup_steps:]
                domega_dr_up = self.domega_dr_up[num_mcmc_warmup_steps:]
                domega_dr_dn = self.domega_dr_dn[num_mcmc_warmup_steps:]

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

                # split and binning with multiple walkers
                w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
                w_L_force_HF_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_HF), num_mcmc_bin_blocks, axis=0)
                w_L_force_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_PP), num_mcmc_bin_blocks, axis=0)
                w_L_E_L_force_PP_split = np.array_split(
                    np.einsum("iw,iwjk->iwjk", w_L, E_L_force_PP), num_mcmc_bin_blocks, axis=0
                )

                # binned sum
                w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
                w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))

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

            if mpi_rank == 0:
                w_L_binned_split = np.array_split(w_L_binned, mpi_size)
                w_L_e_L_binned_split = np.array_split(w_L_e_L_binned, mpi_size)
                w_L_force_HF_binned_split = np.array_split(w_L_force_HF_binned, mpi_size)
                w_L_force_PP_binned_split = np.array_split(w_L_force_PP_binned, mpi_size)
                w_L_E_L_force_PP_binned_split = np.array_split(w_L_E_L_force_PP_binned, mpi_size)
            else:
                w_L_binned_split = None
                w_L_e_L_binned_split = None
                w_L_force_HF_binned_split = None
                w_L_force_PP_binned_split = None
                w_L_E_L_force_PP_binned_split = None

            # scatter
            w_L_binned_local = mpi_comm.scatter(w_L_binned_split, root=0)
            w_L_e_L_binned_local = mpi_comm.scatter(w_L_e_L_binned_split, root=0)
            w_L_force_HF_binned_local = mpi_comm.scatter(w_L_force_HF_binned_split, root=0)
            w_L_force_PP_binned_local = mpi_comm.scatter(w_L_force_PP_binned_split, root=0)
            w_L_E_L_force_PP_binned_local = mpi_comm.scatter(w_L_E_L_force_PP_binned_split, root=0)

            # convert to numpy arrays
            w_L_binned_local = np.array(w_L_binned_local)
            w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
            w_L_force_HF_binned_local = np.array(w_L_force_HF_binned_local)
            w_L_force_PP_binned_local = np.array(w_L_force_PP_binned_local)
            w_L_E_L_force_PP_binned_local = np.array(w_L_E_L_force_PP_binned_local)

            ## local sum
            w_L_binned_local_sum = np.sum(w_L_binned_local, axis=0)
            w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local, axis=0)
            w_L_force_HF_binned_local_sum = np.sum(w_L_force_HF_binned_local, axis=0)
            w_L_force_PP_binned_local_sum = np.sum(w_L_force_PP_binned_local, axis=0)
            w_L_E_L_force_PP_binned_local_sum = np.sum(w_L_E_L_force_PP_binned_local, axis=0)

            ## glolbal sum
            w_L_binned_global_sum = np.empty_like(w_L_binned_local_sum)
            w_L_e_L_binned_global_sum = np.empty_like(w_L_e_L_binned_local_sum)
            w_L_force_HF_binned_global_sum = np.empty_like(w_L_force_HF_binned_local_sum)
            w_L_force_PP_binned_global_sum = np.empty_like(w_L_force_PP_binned_local_sum)
            w_L_E_L_force_PP_binned_global_sum = np.empty_like(w_L_E_L_force_PP_binned_local_sum)

            ## mpi Allreduce
            mpi_comm.Allreduce([w_L_binned_local_sum, MPI.DOUBLE], [w_L_binned_global_sum, MPI.DOUBLE], op=MPI.SUM)
            mpi_comm.Allreduce([w_L_e_L_binned_local_sum, MPI.DOUBLE], [w_L_e_L_binned_global_sum, MPI.DOUBLE], op=MPI.SUM)
            mpi_comm.Allreduce(
                [w_L_force_HF_binned_local_sum, MPI.DOUBLE], [w_L_force_HF_binned_global_sum, MPI.DOUBLE], op=MPI.SUM
            )
            mpi_comm.Allreduce(
                [w_L_force_PP_binned_local_sum, MPI.DOUBLE], [w_L_force_PP_binned_global_sum, MPI.DOUBLE], op=MPI.SUM
            )
            mpi_comm.Allreduce(
                [w_L_E_L_force_PP_binned_local_sum, MPI.DOUBLE], [w_L_E_L_force_PP_binned_global_sum, MPI.DOUBLE], op=MPI.SUM
            )

            ## jackknie binned samples
            M_local = w_L_binned_local.size
            M_total = mpi_comm.allreduce(M_local, op=MPI.SUM)

            force_HF_jn_local = -1.0 * np.array(
                [
                    (w_L_force_HF_binned_global_sum - w_L_force_HF_binned_local[j])
                    / (w_L_binned_global_sum - w_L_binned_local[j])
                    for j in range(M_local)
                ]
            )

            force_Pulay_jn_local = -2.0 * np.array(
                [
                    (
                        (w_L_E_L_force_PP_binned_global_sum - w_L_E_L_force_PP_binned_local[j])
                        / (w_L_binned_global_sum - w_L_binned_local[j])
                        - (
                            (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[j])
                            / (w_L_binned_global_sum - w_L_binned_local[j])
                            * (w_L_force_PP_binned_global_sum - w_L_force_PP_binned_local[j])
                            / (w_L_binned_global_sum - w_L_binned_local[j])
                        )
                    )
                    for j in range(M_local)
                ]
            )

            force_jn_local = force_HF_jn_local + force_Pulay_jn_local

            sum_force_local = np.sum(force_jn_local, axis=0)
            sumsq_force_local = np.sum(force_jn_local**2, axis=0)

            sum_force_global = np.empty_like(sum_force_local)
            sumsq_force_global = np.empty_like(sumsq_force_local)

            mpi_comm.Allreduce([sum_force_local, MPI.DOUBLE], [sum_force_global, MPI.DOUBLE], op=MPI.SUM)
            mpi_comm.Allreduce([sumsq_force_local, MPI.DOUBLE], [sumsq_force_global, MPI.DOUBLE], op=MPI.SUM)

            ## mean and var = E[x^2] - (E[x])^2
            mean_force_global = sum_force_global / M_total
            var_force_global = (sumsq_force_global / M_total) - (sum_force_global / M_total) ** 2

            ## mean and std
            force_mean = mean_force_global
            force_std = np.sqrt((M_total - 1) * var_force_global)

        return (force_mean, force_std)


class _GFMC_n_debug:
    """GFMC class. Runing GFMC with multiple walkers.

    Args:
        hamiltonian_data (Hamiltonian_data):
            an instance of Hamiltonian_data
        num_walkers (int):
            the number of walkers
        mcmc_seed (int):
            seed for the MCMC chain.
        E_scf (float):
            Self-consistent E (Hartree)
        alat (float):
            discretized grid length (bohr)
        non_local_move (str):
            treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
            Valid only for ECP calculations. Do not specify this value for all-electron calculations.
        comput_position_deriv (bool):
            if True, compute the derivatives of E wrt. atomic positions.
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

        # initialize random seed
        np.random.seed(self.__mpi_seed)

        # Place electrons around each nucleus with improved spin assignment
        ## check the number of electrons
        tot_num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        tot_num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        if hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
                hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

        coords = hamiltonian_data.structure_data._positions_cart_jnp

        ## generate initial electron configurations
        r_carts_up, r_carts_dn, up_owner, dn_owner = _generate_init_electron_configurations(
            tot_num_electron_up, tot_num_electron_dn, self.__num_walkers, charges, coords
        )

        ## Electron assignment for all atoms is complete. Check the assignment.
        for i_walker in range(self.__num_walkers):
            logger.debug(f"--Walker No.{i_walker + 1}: electrons assignment--")
            nion = coords.shape[0]
            up_counts = np.bincount(up_owner[i_walker], minlength=nion)
            dn_counts = np.bincount(dn_owner[i_walker], minlength=nion)
            logger.debug(f"  Charges: {charges}")
            logger.debug(f"  up counts: {up_counts}")
            logger.debug(f"  dn counts: {dn_counts}")
            logger.debug(f"  Total counts: {up_counts + dn_counts}")

        self.__latest_r_up_carts = jnp.array(r_carts_up)
        self.__latest_r_dn_carts = jnp.array(r_carts_dn)

        logger.debug(f"  initial r_up_carts= {self.__latest_r_up_carts}")
        logger.debug(f"  initial r_dn_carts = {self.__latest_r_dn_carts}")
        logger.debug(f"  initial r_up_carts.shape = {self.__latest_r_up_carts.shape}")
        logger.debug(f"  initial r_dn_carts.shape = {self.__latest_r_dn_carts.shape}")
        logger.debug("")

        # print out the number of walkers/MPI processes
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
        if num_gfmc_collect_steps < GFMC_MIN_COLLECT_STEPS:
            logger.warning(f"num_gfmc_collect_steps should be larger than {GFMC_MIN_COLLECT_STEPS}.")
        self.__num_gfmc_collect_steps = num_gfmc_collect_steps

    # weights
    @property
    def w_L(self) -> npt.NDArray:
        """Return the stored weight array. dim: (mcmc_counter, 1)."""
        # logger.info(f"np.array(self.__stored_w_L).shape = {np.array(self.__stored_w_L).shape}.")
        return _compute_G_L_debug(np.array(self.__stored_w_L), self.__num_gfmc_collect_steps)

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

        def _generate_rotation_matrix_n_debug(alpha, beta, gamma):
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

        def _projection_n_debug(
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
                    compute_kinetic_energy_all_elements(
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
                R = _generate_rotation_matrix_n_debug(alpha, beta, gamma)

                # compute discretized kinetic energy and mesh (with a random rotation)
                mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                    compute_discretized_kinetic_energy(
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
                diagonal_bare_coulomb_part_el_el = compute_bare_coulomb_potential_el_el(
                    r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
                )

                # compute diagonal elements, ion-ion
                diagonal_bare_coulomb_part_ion_ion = compute_bare_coulomb_potential_ion_ion(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data
                )

                # compute diagonal elements, el-ion
                diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                    compute_bare_coulomb_potential_el_ion_element_wise(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )
                )

                # compute diagonal elements, el-ion, discretized
                (
                    diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                    diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
                ) = compute_discretized_bare_coulomb_potential_el_ion_element_wise(
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
                    diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    if non_local_move == "tmove":
                        # ecp non-local (t-move)
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors(
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
                            compute_ecp_non_local_parts_nearest_neighbors(
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

                        Jastrow_ref = compute_Jastrow_part(
                            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                        )
                        Jastrow_on_mesh = vmap(compute_Jastrow_part, in_axes=(None, 0, 0))(
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

        def _compute_V_elements_n_debug(
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
                compute_kinetic_energy_all_elements(
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # compute discretized kinetic energy and mesh (with a random rotation)
            _, _, elements_non_diagonal_kinetic_part = compute_discretized_kinetic_energy(
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
            diagonal_bare_coulomb_part_el_el = compute_bare_coulomb_potential_el_el(
                r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
            )

            # compute diagonal elements, ion-ion
            diagonal_bare_coulomb_part_ion_ion = compute_bare_coulomb_potential_ion_ion(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data
            )

            # compute diagonal elements, el-ion
            diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                compute_bare_coulomb_potential_el_ion_element_wise(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # compute diagonal elements, el-ion, discretized
            (
                diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
            ) = compute_discretized_bare_coulomb_potential_el_ion_element_wise(
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
                diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

                if non_local_move == "tmove":
                    # ecp non-local (t-move)
                    mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                        compute_ecp_non_local_parts_nearest_neighbors(
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
                        compute_ecp_non_local_parts_nearest_neighbors(
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

                    Jastrow_ref = compute_Jastrow_part(
                        jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    Jastrow_on_mesh = vmap(compute_Jastrow_part, in_axes=(None, 0, 0))(
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

        def _compute_local_energy_n_debug(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            RT: jnpt.ArrayLike,
            non_local_move: bool,
            alat: float,
        ):
            V_diag, V_nondiag = _compute_V_elements_n_debug(hamiltonian_data, r_up_carts, r_dn_carts, RT, non_local_move, alat)
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
                _projection_n_debug, in_axes=(0, 0, 0, 0, None, None, None, None, None)
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
            V_diag_list, V_nondiag_list = vmap(_compute_V_elements_n_debug, in_axes=(None, 0, 0, 0, None, None))(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                latest_RT,
                self.__non_local_move,
                self.__alat,
            )
            e_L_list = V_diag_list + V_nondiag_list

            if self.__non_local_move == "tmove":
                e_list_debug = vmap(compute_local_energy, in_axes=(None, 0, 0, 0))(
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
                    grad(_compute_local_energy_n_debug, argnums=(0, 1, 2)), in_axes=(None, 0, 0, 0, None, None)
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
                    grad(evaluate_ln_wavefunction, argnums=(0, 1, 2)), in_axes=(None, 0, 0)
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

                omega_up = vmap(evaluate_swct_omega, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_up_carts,
                )

                omega_dn = vmap(evaluate_swct_omega, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_dn_carts,
                )

                grad_omega_dr_up = vmap(evaluate_swct_domega, in_axes=(None, 0))(
                    self.__swct_data,
                    self.__latest_r_up_carts,
                )

                grad_omega_dr_dn = vmap(evaluate_swct_domega, in_axes=(None, 0))(
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
                w_L_weighted_sum = np.sum(w_L_gathered / V_diag_E_gathered)
                e_L_weighted_sum = np.sum(w_L_gathered / V_diag_E_gathered * e_L_gathered)
                e_L2_weighted_sum = np.sum(w_L_gathered / V_diag_E_gathered * e_L_gathered**2)
                if self.__comput_position_deriv:
                    grad_e_L_r_up_weighted_sum = np.einsum(
                        "i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_e_L_r_up_gathered
                    )
                    grad_e_L_r_dn_weighted_sum = np.einsum(
                        "i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_e_L_r_dn_gathered
                    )
                    grad_e_L_R_weighted_sum = np.einsum("i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_e_L_R_gathered)
                    grad_ln_Psi_r_up_weighted_sum = np.einsum(
                        "i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_ln_Psi_r_up_gathered
                    )
                    grad_ln_Psi_r_dn_weighted_sum = np.einsum(
                        "i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_ln_Psi_r_dn_gathered
                    )
                    grad_ln_Psi_dR_weighted_sum = np.einsum(
                        "i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_ln_Psi_dR_gathered
                    )
                    omega_up_weighted_sum = np.einsum("i,ijk->jk", w_L_gathered / V_diag_E_gathered, omega_up_gathered)
                    omega_dn_weighted_sum = np.einsum("i,ijk->jk", w_L_gathered / V_diag_E_gathered, omega_dn_gathered)
                    grad_omega_dr_up_weighted_sum = np.einsum(
                        "i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_omega_dr_up_gathered
                    )
                    grad_omega_dr_dn_weighted_sum = np.einsum(
                        "i,ijk->jk", w_L_gathered / V_diag_E_gathered, grad_omega_dr_dn_gathered
                    )
                # averaged
                w_L_averaged = np.average(w_L_gathered)
                e_L_averaged = e_L_weighted_sum / w_L_weighted_sum
                e_L2_averaged = e_L2_weighted_sum / w_L_weighted_sum
                if self.__comput_position_deriv:
                    grad_e_L_r_up_averaged = grad_e_L_r_up_weighted_sum / w_L_weighted_sum
                    grad_e_L_r_dn_averaged = grad_e_L_r_dn_weighted_sum / w_L_weighted_sum
                    grad_e_L_R_averaged = grad_e_L_R_weighted_sum / w_L_weighted_sum
                    grad_ln_Psi_r_up_averaged = grad_ln_Psi_r_up_weighted_sum / w_L_weighted_sum
                    grad_ln_Psi_r_dn_averaged = grad_ln_Psi_r_dn_weighted_sum / w_L_weighted_sum
                    grad_ln_Psi_dR_averaged = grad_ln_Psi_dR_weighted_sum / w_L_weighted_sum
                    omega_up_averaged = omega_up_weighted_sum / w_L_weighted_sum
                    omega_dn_averaged = omega_dn_weighted_sum / w_L_weighted_sum
                    grad_omega_dr_up_averaged = grad_omega_dr_up_weighted_sum / w_L_weighted_sum
                    grad_omega_dr_dn_averaged = grad_omega_dr_dn_weighted_sum / w_L_weighted_sum
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
            eq_steps = GFMC_ON_THE_FLY_WARMUP_STEPS
            num_gfmc_collect_steps = GFMC_ON_THE_FLY_COLLECT_STEPS
            num_gfmc_bin_blocks = GFMC_ON_THE_FLY_BIN_BLOCKS
            if (i_mcmc_step + 1) % mcmc_interval == 0:
                if i_mcmc_step > eq_steps:
                    self.__E_scf, E_scf_std = self.get_E_on_the_fly(
                        num_gfmc_warmup_steps=np.minimum(eq_steps, i_mcmc_step - eq_steps),
                        num_gfmc_bin_blocks=num_gfmc_bin_blocks,
                        num_gfmc_collect_steps=num_gfmc_collect_steps,
                    )
                    logger.debug(f"    Updated E_scf = {self.__E_scf:.5f} +- {E_scf_std:.5f} Ha.")
                else:
                    logger.debug(f"    Init E_scf = {self.__E_scf:.5f} Ha. Being equilibrated.")

            # count up, here is the end of the branching step.
            num_mcmc_done += 1

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
            G_eq = _compute_G_L_debug(w_L_eq, num_gfmc_collect_steps)
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
        if mpi_rank == 0:
            e_L = self.e_L[num_mcmc_warmup_steps:]
            e_L2 = self.e_L2[num_mcmc_warmup_steps:]
            w_L = self.w_L[num_mcmc_warmup_steps:]
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
            w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
            w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
            w_L_e_L2_split = np.array_split(w_L * e_L2, num_mcmc_bin_blocks, axis=0)
            w_L_e_L2_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L2_split]))

            w_L_binned_local = w_L_binned
            w_L_e_L_binned_local = w_L_e_L_binned
            w_L_e_L2_binned_local = w_L_e_L2_binned

            w_L_binned_local = np.array(w_L_binned_local)
            w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
            w_L_e_L2_binned_local = np.array(w_L_e_L2_binned_local)

            # old implementation (keep this just for debug, for the time being. To be deleted.)
            w_L_binned_global_sum = np.sum(w_L_binned_local, axis=0)
            w_L_e_L_binned_global_sum = np.sum(w_L_e_L_binned_local, axis=0)
            w_L_e_L2_binned_global_sum = np.sum(w_L_e_L2_binned_local, axis=0)

            M_local = w_L_binned_local.size
            logger.debug(f"The number of local binned samples = {M_local}")

            E_jackknife_binned_local = [
                (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                for m in range(M_local)
            ]

            E2_jackknife_binned_local = [
                (w_L_e_L2_binned_global_sum - w_L_e_L2_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                for m in range(M_local)
            ]

            Var_jackknife_binned_local = list(np.array(E2_jackknife_binned_local) - np.array(E_jackknife_binned_local) ** 2)

            # MPI allreduce
            E_jackknife_binned = E_jackknife_binned_local
            Var_jackknife_binned = Var_jackknife_binned_local
            M_total = len(E_jackknife_binned)
            logger.debug(f"The number of total binned samples = {M_total}")

            # jackknife mean and std
            E_mean = np.average(E_jackknife_binned)
            E_std = np.sqrt(M_total - 1) * np.std(E_jackknife_binned)
            Var_mean = np.average(Var_jackknife_binned)
            Var_std = np.sqrt(M_total - 1) * np.std(Var_jackknife_binned)

            logger.info(f"E = {E_mean} +- {E_std} Ha.")
            logger.info(f"Var(E) = {Var_mean} +- {Var_std} Ha^2.")

        else:
            E_mean = None
            E_std = None
            Var_mean = None
            Var_std = None

        # broadcast the results
        E_mean = mpi_comm.bcast(E_mean, root=0)
        E_std = mpi_comm.bcast(E_std, root=0)
        Var_mean = mpi_comm.bcast(Var_mean, root=0)
        Var_std = mpi_comm.bcast(Var_std, root=0)

        # return the results
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
        if mpi_rank == 0:
            w_L = self.w_L[num_mcmc_warmup_steps:]
            e_L = self.e_L[num_mcmc_warmup_steps:]
            de_L_dR = self.de_L_dR[num_mcmc_warmup_steps:]
            de_L_dr_up = self.de_L_dr_up[num_mcmc_warmup_steps:]
            de_L_dr_dn = self.de_L_dr_dn[num_mcmc_warmup_steps:]
            dln_Psi_dr_up = self.dln_Psi_dr_up[num_mcmc_warmup_steps:]
            dln_Psi_dr_dn = self.dln_Psi_dr_dn[num_mcmc_warmup_steps:]
            dln_Psi_dR = self.dln_Psi_dR[num_mcmc_warmup_steps:]
            omega_up = self.omega_up[num_mcmc_warmup_steps:]
            omega_dn = self.omega_dn[num_mcmc_warmup_steps:]
            domega_dr_up = self.domega_dr_up[num_mcmc_warmup_steps:]
            domega_dr_dn = self.domega_dr_dn[num_mcmc_warmup_steps:]

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

            # split and binning with multiple walkers
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
            w_L_force_HF_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_HF), num_mcmc_bin_blocks, axis=0)
            w_L_force_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_PP), num_mcmc_bin_blocks, axis=0)
            w_L_E_L_force_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, E_L_force_PP), num_mcmc_bin_blocks, axis=0)

            # binned sum
            w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
            w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))

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

            w_L_binned_local = np.array(w_L_binned)
            w_L_e_L_binned_local = np.array(w_L_e_L_binned)
            w_L_force_HF_binned_local = np.array(w_L_force_HF_binned)
            w_L_force_PP_binned_local = np.array(w_L_force_PP_binned)
            w_L_E_L_force_PP_binned_local = np.array(w_L_E_L_force_PP_binned)

            # old implementation (keep this just for debug, for the time being. To be deleted.)
            w_L_binned_global_sum = np.sum(w_L_binned_local, axis=0)
            w_L_e_L_binned_global_sum = np.sum(w_L_e_L_binned_local, axis=0)
            w_L_force_HF_binned_global_sum = np.sum(w_L_force_HF_binned_local, axis=0)
            w_L_force_PP_binned_global_sum = np.sum(w_L_force_PP_binned_local, axis=0)
            w_L_E_L_force_PP_binned_global_sum = np.sum(w_L_E_L_force_PP_binned_local, axis=0)

            M_local = w_L_binned_local.size
            logger.debug(f"The number of local binned samples = {M_local}")

            force_HF_jn_local = -1.0 * np.array(
                [
                    (w_L_force_HF_binned_global_sum - w_L_force_HF_binned_local[j])
                    / (w_L_binned_global_sum - w_L_binned_local[j])
                    for j in range(M_local)
                ]
            )

            force_Pulay_jn_local = -2.0 * np.array(
                [
                    (
                        (w_L_E_L_force_PP_binned_global_sum - w_L_E_L_force_PP_binned_local[j])
                        / (w_L_binned_global_sum - w_L_binned_local[j])
                        - (
                            (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[j])
                            / (w_L_binned_global_sum - w_L_binned_local[j])
                            * (w_L_force_PP_binned_global_sum - w_L_force_PP_binned_local[j])
                            / (w_L_binned_global_sum - w_L_binned_local[j])
                        )
                    )
                    for j in range(M_local)
                ]
            )

            force_jn_local = list(force_HF_jn_local + force_Pulay_jn_local)

            force_jn = force_jn_local
            M_total = len(force_jn)
            logger.debug(f"The number of total binned samples = {M_total}")

            force_mean = np.average(force_jn, axis=0)
            force_std = np.sqrt(M_total - 1) * np.std(force_jn, axis=0)

            logger.devel(f"force_mean.shape  = {force_mean.shape}.")
            logger.devel(f"force_std.shape  = {force_std.shape}.")
            logger.info(f"force = {force_mean} +- {force_std} Ha.")

        else:
            force_mean = None
            force_std = None

        # broadcast the results
        force_mean = mpi_comm.bcast(force_mean, root=0)
        force_std = mpi_comm.bcast(force_std, root=0)

        return (force_mean, force_std)


"""
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
"""
