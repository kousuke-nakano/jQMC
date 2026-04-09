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
from collections import defaultdict
from functools import partial
from logging import getLogger
from typing import Any

import jax
import numpy as np
import numpy.typing as npt
import optax
import scipy
import toml
from jax import grad, jit, lax, vmap
from jax import numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve  # noqa: F401  (kept for external callers / _MCMC_debug)
from mpi4py import MPI

from ._diff_mask import DiffMask, apply_diff_mask
from ._jqmc_utility import _generate_init_electron_configurations
from ._setting import (
    MCMC_MIN_BIN_BLOCKS,
    MCMC_MIN_WARMUP_STEPS,
    EPS_rcond_SVD,
    EPS_zero_division,
    atol_consistency,
    min_S_diag_abs,
)
from .atomic_orbital import compute_overlap_matrix
from .determinant import (
    Geminal_data,
    compute_AS_regularization_factor,
    compute_AS_regularization_factor_fast_update,
    compute_det_geminal_all_elements,
    compute_geminal_all_elements,
    compute_geminal_dn_one_column_elements,
    compute_geminal_up_one_row_elements,
)
from .hamiltonians import (
    Hamiltonian_data,
    compute_local_energy,
    compute_local_energy_fast,
)
from .jastrow_factor import _compute_ratio_Jastrow_part_rank1_update, compute_Jastrow_part
from .structure import _find_nearest_index_jnp
from .swct import evaluate_swct_domega, evaluate_swct_omega
from .wavefunction import evaluate_ln_wavefunction, evaluate_ln_wavefunction_fast

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


class MCMC:
    """Production VMC/MCMC driver with multiple walkers.

    This class drives Metropolis–Hastings sampling for many independent walkers in parallel
    (vectorized with ``jax.vmap``) and stores all observables needed by downstream analysis
    and optimization. All public methods are part of the supported API; private helpers are
    internal and subject to change.
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
        comput_log_WF_param_deriv: bool = False,
        comput_e_L_param_deriv: bool = False,
        comput_position_deriv: bool = False,
        random_discretized_mesh: bool = True,
    ) -> None:
        """Build an MCMC driver and initialize walker state.

        Args:
            hamiltonian_data (Hamiltonian_data): Problem definition; ``sanity_check`` is called before sampling.
            mcmc_seed (int, optional): Base RNG seed; folded with MPI rank/walker. Defaults to 34467.
            num_walkers (int, optional): Number of walkers on this rank. Defaults to 40.
            num_mcmc_per_measurement (int, optional): Proposals between observable evaluations. Defaults to 16.
            Dt (float, optional): Electron displacement scale (bohr). Defaults to 2.0.
            epsilon_AS (float, optional): Regularization exponent for antisymmetric stabilization. Defaults to 1e-1.
            comput_log_WF_param_deriv (bool, optional): Keep variational parameter derivatives (d ln Psi / dc). Defaults to False.
            comput_e_L_param_deriv (bool, optional): Keep local energy variational parameter derivatives (de_L / dc). Defaults to False.
            comput_position_deriv (bool, optional): Keep nuclear position derivatives. Defaults to False.
            random_discretized_mesh (bool, optional): Randomize quadrature mesh for non-local ECP terms. Defaults to True.

        Notes:
            - Seeds are folded with MPI rank and walker index to avoid correlation.
            - Initial electron configurations are placed near nuclei with balanced spins.
            - Logger prints walker and Hamiltonian diagnostics during initialization.
        """
        self.__mcmc_seed = mcmc_seed
        self.__num_walkers = num_walkers
        self.__num_mcmc_per_measurement = num_mcmc_per_measurement
        self.__Dt = Dt
        self.__epsilon_AS = epsilon_AS
        # self.__adjust_epsilon_AS = adjust_epsilon_AS
        self.__comput_log_WF_param_deriv = comput_log_WF_param_deriv
        self.__comput_e_L_param_deriv = comput_e_L_param_deriv
        self.__comput_position_deriv = comput_position_deriv
        self.__random_discretized_mesh = random_discretized_mesh

        # check sanity of hamiltonian_data
        hamiltonian_data.sanity_check()

        # set hamiltonian_data
        self.__hamiltonian_data = hamiltonian_data
        self.__param_grad_flags = self.__default_param_grad_flags()

        # optimizer runtime container (used for optax restarts)
        self.__optimizer_runtime = None
        self.__ensure_optimizer_runtime()

        # optimization counter
        self.__i_opt = 0

        # seeds
        self.__mpi_seed = self.__mcmc_seed * (mpi_rank + 1)
        self.__jax_PRNG_key = jax.random.PRNGKey(self.__mpi_seed)
        self.__jax_PRNG_key_list = jnp.array([jax.random.fold_in(self.__jax_PRNG_key, nw) for nw in range(self.__num_walkers)])

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

        # check if only up electrons are updated
        if tot_num_electron_dn == 0:
            logger.info("  Only up electrons are updated in the MCMC.")
            self.__only_up_electron = True
        else:
            self.__only_up_electron = False

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

        # init_attributes
        self.hamiltonian_data = self.__hamiltonian_data
        self.__init_attributes()

    def __init_attributes(self):
        # mcmc counter
        self.__mcmc_counter = 0

        # mcmc accepted/rejected moves
        self.__accepted_moves = 0
        self.__rejected_moves = 0

        # Compute dimensions for pre-shaped empty arrays
        nw = self.__num_walkers
        n_atoms = self.__hamiltonian_data.structure_data.natom

        # stored weight (w_L)
        self.__stored_w_L = np.zeros((0, nw))

        # stored local energy (e_L)
        self.__stored_e_L = np.zeros((0, nw))

        # stored local energy (e_L2)
        self.__stored_e_L2 = np.zeros((0, nw))

        # stored force_HF per walker (HF force = de_L/dR + Omega . de_L/dr)
        self.__stored_force_HF = np.zeros((0, nw, n_atoms, 3))

        # stored force_PP per walker (Pulay force = dln_Psi/dR + Omega . dln_Psi/dr + 1/2 * d_omega/dr)
        self.__stored_force_PP = np.zeros((0, nw, n_atoms, 3))

        # stored E_L * force_PP per walker (for covariance in Pulay force)
        self.__stored_E_L_force_PP = np.zeros((0, nw, n_atoms, 3))

        # stored parameter gradients keyed by block name
        self.__stored_log_WF_param_grads: dict[str, list] = defaultdict(list)

        # stored local energy parameter gradients keyed by block name (de_L / dc)
        self.__stored_e_L_param_grads: dict[str, list] = defaultdict(list)

    def __validate_stored_shapes(self) -> None:
        """Assert that all stored observable arrays have consistent shapes."""
        ns = self.__mcmc_counter  # expected number of stored steps
        nw = self.__num_walkers
        n_atoms = self.__hamiltonian_data.structure_data.natom

        expected = {
            "e_L": ((ns, nw), self.__stored_e_L),
            "e_L2": ((ns, nw), self.__stored_e_L2),
            "w_L": ((ns, nw), self.__stored_w_L),
        }
        if self.__comput_position_deriv:
            expected.update(
                {
                    "force_HF": ((ns, nw, n_atoms, 3), self.__stored_force_HF),
                    "force_PP": ((ns, nw, n_atoms, 3), self.__stored_force_PP),
                    "E_L_force_PP": ((ns, nw, n_atoms, 3), self.__stored_E_L_force_PP),
                }
            )

        for name, (shape, arr) in expected.items():
            assert arr.shape == shape, f"stored shape mismatch: {name}.shape={arr.shape}, expected={shape}"

    @staticmethod
    def __default_param_grad_flags() -> dict[str, bool]:
        return {
            "j1_param": True,
            "j2_param": True,
            "j3_matrix": True,
            "jastrow_nn_params": True,
            "lambda_matrix": True,
            "j3_basis_exp": False,
            "j3_basis_coeff": False,
            "lambda_basis_exp": False,
            "lambda_basis_coeff": False,
        }

    def __param_gradient_flags(self) -> dict[str, bool]:
        """Return a copy of the current per-block gradient flags."""
        return dict(self.__param_grad_flags)

    def __set_param_gradient_flags(self, **flags: bool | None) -> None:
        """Update per-block gradient flags (True enables, False disables)."""
        allowed = set(self.__param_grad_flags)
        for name, value in flags.items():
            if name not in allowed:
                raise ValueError(f"Unknown variational block '{name}'.")
            if value is None:
                continue
            self.__param_grad_flags[name] = bool(value)

    def __needs_param_grad_mask(self) -> bool:
        return any(not enabled for enabled in self.__param_grad_flags.values())

    def __wavefunction_data_for_param_grads(self):
        flags = self.__param_grad_flags
        return self.__hamiltonian_data.wavefunction_data.with_param_grad_mask(
            opt_J1_param=flags.get("j1_param", True),
            opt_J2_param=flags.get("j2_param", True),
            opt_J3_param=flags.get("j3_matrix", True),
            opt_JNN_param=flags.get("jastrow_nn_params", True),
            opt_lambda_param=flags.get("lambda_matrix", True),
            opt_J3_basis_exp=flags.get("j3_basis_exp", False),
            opt_J3_basis_coeff=flags.get("j3_basis_coeff", False),
            opt_lambda_basis_exp=flags.get("lambda_basis_exp", False),
            opt_lambda_basis_coeff=flags.get("lambda_basis_coeff", False),
        )

    def __prepare_param_grad_objects(self):
        """Return (Wavefunction_data, Hamiltonian_data) for variational parameter gradient computations.

        wavefunction_data: per-block masking applied; used for dln_Psi/dc via
            grad(evaluate_ln_wavefunction, argnums=0).
        hamiltonian_data: R stop-gradiented with active variational parameters injected;
            used for de_L/dc via grad(compute_local_energy, argnums=0).

        [Approx, faster] alternative (not implemented):
            Zeroes out d(V_ECP_NL)/dc; T dominates anyway.
            Build a local _e_L_approx(wf_data, r_up, r_dn, RT) that calls
            compute_kinetic_energy (with wf_data) and compute_coulomb_potential
            (with stop_gradient on wf_data), pre-compile it as
            jit(vmap(grad(_e_L_approx, argnums=0), in_axes=(None, 0, 0, 0))),
            and return it as a third element of the tuple alongside
            (masked_wavefunction, self.__hamiltonian_data).
            In run(), unpack as a 3-tuple and call the compiled approx grad
            instead of _jit_vmap_grad_e_L_h.
        """
        wavefunction_data = self.__hamiltonian_data.wavefunction_data
        if not (self.__comput_log_WF_param_deriv or self.__comput_e_L_param_deriv):
            return wavefunction_data, self.__hamiltonian_data

        masked_wavefunction = (
            self.__wavefunction_data_for_param_grads() if self.__needs_param_grad_mask() else wavefunction_data
        )
        # stop_gradient on R so that grad(compute_local_energy, argnums=0)
        # yields de_L/dc only, not de_L/dR.  ECP wf-ratio Psi(r')/Psi(r) contributes
        # to de_L/dc because wavefunction_data still carries gradients.
        h_sg_R = jax.lax.stop_gradient(self.__hamiltonian_data).replace(wavefunction_data=masked_wavefunction)
        return masked_wavefunction, h_sg_R

    def __prepare_position_grad_objects(self):
        """Return (Wavefunction_data, Hamiltonian_data) suitable for de_L/dR and dln_Psi/dR computations.

        All variational parameters are masked with stop_gradient so that JAX
        does not backpropagate through them (e.g. ECP wf-ratio tables).
        AO position leaves (orb_data.structure_data.positions) still carry
        gradients (DiffMask coords=True), so dT/dR and dV_ECP/dR are correct.
        """
        wavefunction_data = self.__hamiltonian_data.wavefunction_data
        if wavefunction_data is None or not self.__comput_position_deriv:
            return wavefunction_data, self.__hamiltonian_data

        # Always mask ALL variational parameters for position gradient computation.
        masked_wavefunction = wavefunction_data.with_param_grad_mask(
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=False,
            opt_JNN_param=False,
            opt_lambda_param=False,
        )
        masked_hamiltonian = self.__hamiltonian_data.replace(wavefunction_data=masked_wavefunction)
        return masked_wavefunction, masked_hamiltonian

    def __ensure_optimizer_runtime(self) -> None:
        if not hasattr(self, "_MCMC__optimizer_runtime") or self.__optimizer_runtime is None:
            self.__optimizer_runtime = {
                "method": None,
                "hyperparameters": None,
                "optax_state": None,
                "optax_param_size": None,
            }

    def __set_optimizer_runtime(
        self,
        *,
        method: str | None,
        hyperparameters: dict[str, Any] | None,
        optax_state: Any,
        optax_param_size: int | None,
    ) -> None:
        self.__ensure_optimizer_runtime()
        hyper_copy = dict(hyperparameters) if hyperparameters is not None else None
        self.__optimizer_runtime = {
            "method": method,
            "hyperparameters": hyper_copy,
            "optax_state": optax_state,
            "optax_param_size": optax_param_size,
        }

    def run(self, num_mcmc_steps: int = 0, max_time=86400) -> None:
        """Execute Metropolis–Hastings sampling for all walkers.

        Args:
            num_mcmc_steps (int, optional): Metropolis updates per walker; values <= 0 are no-ops. Defaults to 0.
            max_time (int, optional): Wall-clock budget in seconds. Defaults to 86400.

        Notes:
            - Creates ``external_control_mcmc.toml`` to allow external stop requests.
            - Accumulates energies, weights, forces, and wavefunction gradients into public buffers (``w_L``, ``e_L``, ``dln_Psi_*`` etc.).
            - Logs timing statistics and acceptance ratios at the end of the run.
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
        timer_MPI_barrier = 0.0

        # mcmc timer starts
        mcmc_total_start = time.perf_counter()

        # toml(control) filename
        toml_filename = "external_control_mcmc.toml"

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

        # MCMC electron position update function
        mcmc_update_init_start = time.perf_counter()

        wavefunction_for_param_grads, hamiltonian_for_param_grads = self.__prepare_param_grad_objects()
        wavefunction_for_position_grads, hamiltonian_for_position_grads = self.__prepare_position_grad_objects()

        # All JIT kernels and vmap wrappers are now defined at module level so
        # that repeated calls to run() (e.g. inside run_optimize) reuse the
        # same Python function objects and hit JAX's compilation cache instead
        # of triggering a full re-compilation each time.

        geminal, geminal_inv, _, _ = _geminal_inv_batched(
            self.__hamiltonian_data.wavefunction_data.geminal_data,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
        )

        RTs = jnp.broadcast_to(jnp.eye(3), (len(self.__jax_PRNG_key_list), 3, 3))

        # Warm-up compilation: trigger JIT tracing on the first run() call
        # so that the MCMC loop does not stall on the first step.
        # On subsequent calls the JAX cache is already warm and these calls
        # return almost instantly, but we still skip them to avoid the overhead.
        if not getattr(self, "_MCMC__mcmc_kernels_warmed_up", False):
            logger.info("Start compilation of the MCMC_update funciton.")
            if self.__only_up_electron:
                _ = _jit_vmap_update_up(
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__jax_PRNG_key_list,
                    self.__num_mcmc_per_measurement,
                    self.__hamiltonian_data,
                    self.__Dt,
                    self.__epsilon_AS,
                    geminal_inv,
                    geminal,
                )
            else:
                _ = _jit_vmap_update(
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__jax_PRNG_key_list,
                    self.__num_mcmc_per_measurement,
                    self.__hamiltonian_data,
                    self.__Dt,
                    self.__epsilon_AS,
                    geminal_inv,
                    geminal,
                )
            _ = _jit_vmap_e_L_fast(
                self.__hamiltonian_data, self.__latest_r_up_carts, self.__latest_r_dn_carts, RTs, geminal_inv
            )
            _ = _jit_vmap_as_reg(
                self.__hamiltonian_data.wavefunction_data.geminal_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            _ = _jit_vmap_as_reg_fast(geminal, geminal_inv)
            if self.__comput_position_deriv:
                _, _ = _jit_vmap_grad_e_L_r(
                    hamiltonian_for_position_grads,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    RTs,
                )
                _ = _jit_vmap_grad_e_L_h(
                    hamiltonian_for_position_grads,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    RTs,
                )

                _, _, _ = _jit_vmap_grad_ln_psi(
                    wavefunction_for_position_grads,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )

                _ = _jit_vmap_swct_omega(
                    self.__hamiltonian_data.structure_data,
                    self.__latest_r_up_carts,
                )

                _ = _jit_vmap_swct_omega(
                    self.__hamiltonian_data.structure_data,
                    self.__latest_r_dn_carts,
                )

                _ = _jit_vmap_swct_domega(
                    self.__hamiltonian_data.structure_data,
                    self.__latest_r_up_carts,
                )

                _ = _jit_vmap_swct_domega(
                    self.__hamiltonian_data.structure_data,
                    self.__latest_r_dn_carts,
                )
                _ = _jit_vmap_grad_ln_psi_params(
                    wavefunction_for_param_grads,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )

            if self.__comput_log_WF_param_deriv:
                _ = _jit_vmap_grad_ln_psi_params(
                    wavefunction_for_param_grads,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                _ = _jit_vmap_grad_ln_psi_params_fast(
                    wavefunction_for_param_grads,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    geminal_inv,
                )

            if self.__comput_e_L_param_deriv:
                _ = _jit_vmap_grad_e_L_h(
                    hamiltonian_for_param_grads,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    RTs,
                )

            self.__mcmc_kernels_warmed_up = True
            mcmc_update_init_end = time.perf_counter()
            timer_mcmc_update_init += mcmc_update_init_end - mcmc_update_init_start
            logger.info("End compilation of the MCMC_update funciton.")
            logger.info(f"Elapsed Time = {mcmc_update_init_end - mcmc_update_init_start:.2f} sec.")
            logger.info("")
        else:
            logger.info("Skipping compilation (JAX cache is warm from previous run).")
            logger.info("")

        # MAIN MCMC loop from here !!!
        logger.info("Start MCMC")
        num_mcmc_done = 0
        progress = (self.__mcmc_counter) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
        mcmc_total_current = time.perf_counter()
        logger.info(
            f"  Progress: MCMC step= {self.__mcmc_counter}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.0f} %. Elapsed time = {(mcmc_total_current - mcmc_total_start):.1f} sec."
        )
        mcmc_interval = max(1, int(num_mcmc_steps / 100))  # %

        # adjust_epsilon_AS = self.__adjust_epsilon_AS

        # -- Extend stored arrays with zero-padding for new steps --
        nw = self.__num_walkers
        n_atoms = self.__hamiltonian_data.structure_data.natom

        self.__stored_e_L = np.concatenate([self.__stored_e_L, np.zeros((num_mcmc_steps, nw))])
        self.__stored_e_L2 = np.concatenate([self.__stored_e_L2, np.zeros((num_mcmc_steps, nw))])
        self.__stored_w_L = np.concatenate([self.__stored_w_L, np.zeros((num_mcmc_steps, nw))])
        if self.__comput_position_deriv:
            self.__stored_force_HF = np.concatenate([self.__stored_force_HF, np.zeros((num_mcmc_steps, nw, n_atoms, 3))])
            self.__stored_force_PP = np.concatenate([self.__stored_force_PP, np.zeros((num_mcmc_steps, nw, n_atoms, 3))])
            self.__stored_E_L_force_PP = np.concatenate(
                [self.__stored_E_L_force_PP, np.zeros((num_mcmc_steps, nw, n_atoms, 3))]
            )

        geminal, geminal_inv, _, _ = _geminal_inv_batched(
            self.__hamiltonian_data.wavefunction_data.geminal_data,
            self.__latest_r_up_carts,
            self.__latest_r_dn_carts,
        )

        for i_mcmc_step in range(num_mcmc_steps):
            if (i_mcmc_step + 1) % mcmc_interval == 0:
                progress = (i_mcmc_step + self.__mcmc_counter + 1) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
                mcmc_total_current = time.perf_counter()
                logger.info(
                    f"  Progress: MCMC step = {i_mcmc_step + self.__mcmc_counter + 1}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %. Elapsed time = {(mcmc_total_current - mcmc_total_start):.1f} sec."
                )

            # electron positions are goint to be updated!
            start = time.perf_counter()
            if self.__only_up_electron:
                (
                    accepted_moves_nw,
                    rejected_moves_nw,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__jax_PRNG_key_list,
                    geminal_inv,
                    geminal,
                ) = _jit_vmap_update_up(
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__jax_PRNG_key_list,
                    self.__num_mcmc_per_measurement,
                    self.__hamiltonian_data,
                    self.__Dt,
                    self.__epsilon_AS,
                    geminal_inv,
                    geminal,
                )
            else:
                (
                    accepted_moves_nw,
                    rejected_moves_nw,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__jax_PRNG_key_list,
                    geminal_inv,
                    geminal,
                ) = _jit_vmap_update(
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__jax_PRNG_key_list,
                    self.__num_mcmc_per_measurement,
                    self.__hamiltonian_data,
                    self.__Dt,
                    self.__epsilon_AS,
                    geminal_inv,
                    geminal,
                )
            self.__latest_r_up_carts.block_until_ready()
            self.__latest_r_dn_carts.block_until_ready()
            end = time.perf_counter()
            timer_mcmc_update += end - start

            # store vmapped outcomes
            self.__accepted_moves = self.__accepted_moves + int(np.sum(np.asarray(accepted_moves_nw)))
            self.__rejected_moves = self.__rejected_moves + int(np.sum(np.asarray(rejected_moves_nw)))

            # generate rotation matrices (for non-local ECPs)
            if self.__random_discretized_mesh:
                RTs = _jit_vmap_generate_RTs(self.__jax_PRNG_key_list)
            else:
                RTs = jnp.broadcast_to(jnp.eye(3), (len(self.__jax_PRNG_key_list), 3, 3))

            # Evaluate observables each MCMC cycle
            start = time.perf_counter()
            # logger.debug("    Evaluating e_L ...")
            e_L_step = _jit_vmap_e_L_fast(
                self.__hamiltonian_data, self.__latest_r_up_carts, self.__latest_r_dn_carts, RTs, geminal_inv
            )
            e_L_step.block_until_ready()
            end = time.perf_counter()
            timer_e_L += end - start

            self.__stored_e_L[self.__mcmc_counter + num_mcmc_done] = np.array(e_L_step)
            self.__stored_e_L2[self.__mcmc_counter + num_mcmc_done] = np.array(e_L_step**2)

            # AS weights
            R_AS_step = _jit_vmap_as_reg_fast(geminal, geminal_inv)
            R_AS_eps_step = jnp.maximum(R_AS_step, self.__epsilon_AS)
            w_L_step = (R_AS_step / R_AS_eps_step) ** 2

            self.__stored_w_L[self.__mcmc_counter + num_mcmc_done] = np.array(w_L_step)

            if self.__comput_position_deriv:
                start = time.perf_counter()
                # logger.debug("    Evaluating de_L/dR and de_L/dr ...")

                # de_L/dr_up, de_L/dr_dn: differentiate w.r.t. r_up and r_dn only.
                # Using argnums=(1,2) avoids backpropagating through wavefunction
                # parameters inside the ECP non-local wf-ratio, which is the dominant
                # cost when hamiltonian (argnums=0) is included.
                grad_e_L_r_up_step, grad_e_L_r_dn_step = _jit_vmap_grad_e_L_r(
                    hamiltonian_for_position_grads,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    RTs,
                )
                grad_e_L_r_up_step.block_until_ready()
                grad_e_L_r_dn_step.block_until_ready()

                # de_L/dR: differentiate w.r.t. hamiltonian (argnums=0) only.
                # hamiltonian_for_position_grads already has all variational params masked
                # (stop_gradient), while AO position leaves still carry gradients so
                # dT/dR and dV_ECP/dR are computed correctly.
                grad_e_L_h_step = _jit_vmap_grad_e_L_h(
                    hamiltonian_for_position_grads,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    RTs,
                )
                end = time.perf_counter()
                timer_de_L_dR_dr += end - start

                grad_e_L_R = self.__hamiltonian_data.accumulate_position_grad(grad_e_L_h_step)

                start = time.perf_counter()
                logger.devel("    Evaluating dln_Psi/dR and dln_Psi/dr ...")
                grad_ln_Psi_h_step, grad_ln_Psi_r_up_step, grad_ln_Psi_r_dn_step = _jit_vmap_grad_ln_psi(
                    wavefunction_for_position_grads,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                )
                grad_ln_Psi_r_up_step.block_until_ready()
                grad_ln_Psi_r_dn_step.block_until_ready()
                end = time.perf_counter()
                timer_dln_Psi_dR_dr += end - start

                grad_ln_Psi_dR = self.__hamiltonian_data.wavefunction_data.accumulate_position_grad(grad_ln_Psi_h_step)

                omega_up_step = _jit_vmap_swct_omega(
                    self.__hamiltonian_data.structure_data,
                    self.__latest_r_up_carts,
                )
                omega_dn_step = _jit_vmap_swct_omega(
                    self.__hamiltonian_data.structure_data,
                    self.__latest_r_dn_carts,
                )
                grad_omega_dr_up_step = _jit_vmap_swct_domega(
                    self.__hamiltonian_data.structure_data,
                    self.__latest_r_up_carts,
                )
                grad_omega_dr_dn_step = _jit_vmap_swct_domega(
                    self.__hamiltonian_data.structure_data,
                    self.__latest_r_dn_carts,
                )

                # Compute per-walker force products preserving cross-correlations
                _grad_e_L_r_up_np = np.array(grad_e_L_r_up_step)  # (nw, n_up, 3)
                _grad_e_L_r_dn_np = np.array(grad_e_L_r_dn_step)  # (nw, n_dn, 3)
                _grad_e_L_R_np = np.array(grad_e_L_R)  # (nw, n_atoms, 3)
                _omega_up_np = np.array(omega_up_step)  # (nw, n_atoms, n_up)
                _omega_dn_np = np.array(omega_dn_step)  # (nw, n_atoms, n_dn)
                _grad_omega_dr_up_np = np.array(grad_omega_dr_up_step)  # (nw, n_atoms, 3)
                _grad_omega_dr_dn_np = np.array(grad_omega_dr_dn_step)  # (nw, n_atoms, 3)
                _grad_ln_Psi_r_up_np = np.array(grad_ln_Psi_r_up_step)  # (nw, n_up, 3)
                _grad_ln_Psi_r_dn_np = np.array(grad_ln_Psi_r_dn_step)  # (nw, n_dn, 3)
                _grad_ln_Psi_dR_np = np.array(grad_ln_Psi_dR)  # (nw, n_atoms, 3)

                # force_HF = de_L/dR + Omega_up . de_L/dr_up + Omega_dn . de_L/dr_dn
                _force_HF = (
                    _grad_e_L_R_np
                    + np.einsum("wjk,wkl->wjl", _omega_up_np, _grad_e_L_r_up_np)
                    + np.einsum("wjk,wkl->wjl", _omega_dn_np, _grad_e_L_r_dn_np)
                )  # (nw, n_atoms, 3)

                # force_PP = dln_Psi/dR + Omega . dln_Psi/dr + 1/2 * sum_i d_omega/d_r_i
                _force_PP = (
                    _grad_ln_Psi_dR_np
                    + np.einsum("wjk,wkl->wjl", _omega_up_np, _grad_ln_Psi_r_up_np)
                    + np.einsum("wjk,wkl->wjl", _omega_dn_np, _grad_ln_Psi_r_dn_np)
                    + 0.5 * (_grad_omega_dr_up_np + _grad_omega_dr_dn_np)
                )  # (nw, n_atoms, 3)

                _e_L_np = np.array(e_L_step)  # (nw,)
                _E_L_force_PP = np.einsum("w,wjk->wjk", _e_L_np, _force_PP)  # (nw, n_atoms, 3)

                self.__stored_force_HF[self.__mcmc_counter + num_mcmc_done] = _force_HF
                self.__stored_force_PP[self.__mcmc_counter + num_mcmc_done] = _force_PP
                self.__stored_E_L_force_PP[self.__mcmc_counter + num_mcmc_done] = _E_L_force_PP

            if self.__comput_log_WF_param_deriv:
                start = time.perf_counter()
                # logger.debug("    Evaluating dln_Psi/dc ...")
                grad_ln_Psi_h = _jit_vmap_grad_ln_psi_params_fast(
                    wavefunction_for_param_grads,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    geminal_inv,
                )
                param_grads = self.__hamiltonian_data.wavefunction_data.collect_param_grads(grad_ln_Psi_h)
                flat_param_grads = self.__hamiltonian_data.wavefunction_data.flatten_param_grads(
                    param_grads, self.__num_walkers
                )

                for name, grad_val in flat_param_grads.items():
                    if not self.__param_grad_flags.get(name, True):
                        continue
                    if hasattr(grad_val, "block_until_ready"):
                        grad_val.block_until_ready()
                    self.__stored_log_WF_param_grads[name].append(np.array(grad_val))

                end = time.perf_counter()
                timer_dln_Psi_dc += end - start

            if self.__comput_e_L_param_deriv:
                start = time.perf_counter()
                # logger.debug("    Evaluating de_L/dc ...")
                # Gradient flows through full compute_local_energy including
                # ECP wf-ratio Psi(r')/Psi(r), so d(V_ECP_NL)/dc is included.
                # hamiltonian_for_param_grads has R stop-gradiented; wf params are
                # active (set up in __prepare_param_grad_objects).

                # Diagnostics: check walker positions and RTs before grad call
                _r_up = np.asarray(self.__latest_r_up_carts)
                _r_dn = np.asarray(self.__latest_r_dn_carts)
                _RTs = np.asarray(RTs)
                _nan_r_up = int(np.sum(~np.isfinite(_r_up)))
                _nan_r_dn = int(np.sum(~np.isfinite(_r_dn)))
                _nan_RTs = int(np.sum(~np.isfinite(_RTs)))
                logger.devel(
                    f"    [de_L/dc] r_up_carts: shape={_r_up.shape} non-finite={_nan_r_up}/{_r_up.size} "
                    f"    min={np.nanmin(_r_up):.3e} max={np.nanmax(_r_up):.3e}"
                )
                if _r_dn.size > 0:
                    logger.devel(
                        f"    [de_L/dc] r_dn_carts: shape={_r_dn.shape} non-finite={_nan_r_dn}/{_r_dn.size} "
                        f"    min={np.nanmin(_r_dn):.3e} max={np.nanmax(_r_dn):.3e}"
                    )
                else:
                    logger.devel(f"    [de_L/dc] r_dn_carts: shape={_r_dn.shape} (empty — no down-spin electrons)")
                logger.devel(f"    [de_L/dc] RTs: shape={_RTs.shape} non-finite={_nan_RTs}/{_RTs.size}")

                grad_e_L_h_step = _jit_vmap_grad_e_L_h(
                    hamiltonian_for_param_grads,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    RTs,
                )
                param2_grads = self.__hamiltonian_data.wavefunction_data.collect_param_grads(grad_e_L_h_step.wavefunction_data)
                flat_param2_grads = self.__hamiltonian_data.wavefunction_data.flatten_param_grads(
                    param2_grads, self.__num_walkers
                )

                # Diagnostics: check flat_param2_grads after grad call
                for _pname, _pval in flat_param2_grads.items():
                    _parr = np.asarray(_pval)
                    _pnan = int(np.sum(~np.isfinite(_parr)))
                    logger.devel(
                        f"    [de_L/dc] flat_param2_grads['{_pname}']: shape={_parr.shape} "
                        f"    non-finite={_pnan}/{_parr.size} ({_pnan / _parr.size:.1%})"
                    )

                for name, grad_val in flat_param2_grads.items():
                    if not self.__param_grad_flags.get(name, True):
                        continue
                    if hasattr(grad_val, "block_until_ready"):
                        grad_val.block_until_ready()
                    self.__stored_e_L_param_grads[name].append(np.array(grad_val))

                end = time.perf_counter()
                timer_de_L_dc += end - start

            num_mcmc_done += 1

            # check max time
            mcmc_current = time.perf_counter()
            if max_time < mcmc_current - mcmc_total_start:
                logger.info(f"  Stopping... max_time = {max_time} sec. exceeds.")
                logger.info("  Break the mcmc loop.")
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

        # Barrier after MCMC operation
        start = time.perf_counter()
        mpi_comm.Barrier()
        end = time.perf_counter()
        timer_MPI_barrier += end - start

        logger.info("End MCMC")
        logger.info("")

        # count up the mcmc counter
        # count up mcmc_counter
        self.__mcmc_counter += num_mcmc_done

        # -- Truncate stored arrays to actual number of steps completed --
        self.__stored_e_L = self.__stored_e_L[: self.__mcmc_counter]
        self.__stored_e_L2 = self.__stored_e_L2[: self.__mcmc_counter]
        self.__stored_w_L = self.__stored_w_L[: self.__mcmc_counter]
        if self.__comput_position_deriv:
            self.__stored_force_HF = self.__stored_force_HF[: self.__mcmc_counter]
            self.__stored_force_PP = self.__stored_force_PP[: self.__mcmc_counter]
            self.__stored_E_L_force_PP = self.__stored_E_L_force_PP[: self.__mcmc_counter]

        # test the shapes of stored arrays are consistent with the number of MCMC steps done and the number of walkers
        self.__validate_stored_shapes()

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
            + timer_MPI_barrier
        )

        # remove the toml file
        mpi_comm.Barrier()
        if mpi_rank == 0:
            if os.path.isfile(toml_filename):
                logger.info(f"Delete {toml_filename}")
                os.remove(toml_filename)

        # net MCMC time
        timer_net_mcmc_total = timer_mcmc_total - timer_mcmc_update_init

        # average among MPI processes
        ave_timer_mcmc_total = mpi_comm.allreduce(timer_mcmc_total, op=MPI.SUM) / mpi_size
        ave_timer_mcmc_update_init = mpi_comm.allreduce(timer_mcmc_update_init, op=MPI.SUM) / mpi_size
        ave_timer_net_mcmc_total = mpi_comm.allreduce(timer_net_mcmc_total, op=MPI.SUM) / mpi_size
        if num_mcmc_done > 0:
            ave_timer_mcmc_update = mpi_comm.allreduce(timer_mcmc_update, op=MPI.SUM) / mpi_size / num_mcmc_done
            ave_timer_e_L = mpi_comm.allreduce(timer_e_L, op=MPI.SUM) / mpi_size / num_mcmc_done
            ave_timer_de_L_dR_dr = mpi_comm.allreduce(timer_de_L_dR_dr, op=MPI.SUM) / mpi_size / num_mcmc_done
            ave_timer_dln_Psi_dR_dr = mpi_comm.allreduce(timer_dln_Psi_dR_dr, op=MPI.SUM) / mpi_size / num_mcmc_done
            ave_timer_dln_Psi_dc = mpi_comm.allreduce(timer_dln_Psi_dc, op=MPI.SUM) / mpi_size / num_mcmc_done
            ave_timer_de_L_dc = mpi_comm.allreduce(timer_de_L_dc, op=MPI.SUM) / mpi_size / num_mcmc_done
            ave_timer_MPI_barrier = mpi_comm.allreduce(timer_MPI_barrier, op=MPI.SUM) / mpi_size / num_mcmc_done
            ave_timer_misc = mpi_comm.allreduce(timer_misc, op=MPI.SUM) / mpi_size / num_mcmc_done
        else:
            ave_timer_mcmc_update = 0.0
            ave_timer_e_L = 0.0
            ave_timer_de_L_dR_dr = 0.0
            ave_timer_dln_Psi_dR_dr = 0.0
            ave_timer_dln_Psi_dc = 0.0
            ave_timer_de_L_dc = 0.0
            ave_timer_MPI_barrier = 0.0
            ave_timer_misc = 0.0
        ave_stored_w_L = mpi_comm.allreduce(np.mean(self.__stored_w_L), op=MPI.SUM) / mpi_size
        sum_accepted_moves = mpi_comm.allreduce(int(self.__accepted_moves), op=MPI.SUM)
        sum_rejected_moves = mpi_comm.allreduce(int(self.__rejected_moves), op=MPI.SUM)

        logger.info(f"Total elapsed time for MCMC {num_mcmc_done} steps. = {ave_timer_mcmc_total:.2f} sec.")
        logger.info(f"Pre-compilation time for MCMC = {ave_timer_mcmc_update_init:.2f} sec.")
        logger.info(f"Net total time for MCMC = {ave_timer_net_mcmc_total:.2f} sec.")
        logger.info(f"Elapsed times per MCMC step, averaged over {num_mcmc_done} steps.")
        logger.info(f"  Time for MCMC update = {ave_timer_mcmc_update * 10**3:.2f} msec.")
        logger.info(f"  Time for computing e_L = {ave_timer_e_L * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/dR and de_L/dr = {ave_timer_de_L_dR_dr * 10**3:.2f} msec.")
        logger.info(f"  Time for computing dln_Psi/dR and dln_Psi/dr = {ave_timer_dln_Psi_dR_dr * 10**3:.2f} msec.")
        logger.info(f"  Time for computing dln_Psi/dc = {ave_timer_dln_Psi_dc * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/dc = {ave_timer_de_L_dc * 10**3:.2f} msec.")
        logger.info(f"  Time for MPI barrier after MCMC update = {ave_timer_MPI_barrier * 10**3:.2f} msec.")
        logger.info(f"  Time for misc. (others) = {ave_timer_misc * 10**3:.2f} msec.")
        logger.info(f"Average of walker weights is {ave_stored_w_L:.3f}. Ideal is ~ 0.800. Adjust epsilon_AS.")
        if sum_accepted_moves + sum_rejected_moves > 0:
            logger.info(
                f"Acceptance ratio is {sum_accepted_moves / (sum_accepted_moves + sum_rejected_moves) * 100:.2f} %.  Ideal is ~ 50.00%. Adjust Dt."
            )
        else:
            logger.info("Acceptance ratio is N/A (no moves performed).")
        logger.info("")

    def get_E(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> tuple[float, float]:
        """Estimate total energy and variance with jackknife error bars.

        Args:
            num_mcmc_warmup_steps (int, optional): Samples to discard as warmup. Defaults to 50.
            num_mcmc_bin_blocks (int, optional): Number of jackknife blocks. Defaults to 10.

        Returns:
            tuple[float, float, float, float]: ``(E_mean, E_std, Var_mean, Var_std)`` aggregated across MPI ranks.

        Raises:
            ValueError: If there are insufficient post-warmup samples to form the requested blocks.

        Notes:
            Warns when warmup or block counts fall below ``MCMC_MIN_WARMUP_STEPS`` / ``MCMC_MIN_BIN_BLOCKS``. All reductions are MPI-aware.
        """
        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if num_mcmc_warmup_steps < MCMC_MIN_WARMUP_STEPS:
            logger.warning(f"num_mcmc_warmup_steps should be larger than {MCMC_MIN_WARMUP_STEPS}")
        if num_mcmc_bin_blocks < MCMC_MIN_BIN_BLOCKS:
            logger.warning(f"num_mcmc_bin_blocks should be larger than {MCMC_MIN_BIN_BLOCKS}")

        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if self.mcmc_counter < num_mcmc_warmup_steps:
            logger.error("mcmc_counter should be larger than num_mcmc_warmup_steps")
            raise ValueError
        if self.mcmc_counter - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
            logger.error("(mcmc_counter - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")
            raise ValueError

        e_L = self.e_L[num_mcmc_warmup_steps:]
        e_L2 = self.e_L2[num_mcmc_warmup_steps:]
        w_L = self.w_L[num_mcmc_warmup_steps:]
        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
        w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
        w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
        w_L_e_L2_split = np.array_split(w_L * e_L2, num_mcmc_bin_blocks, axis=0)
        w_L_e_L2_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L2_split]))

        # MCMC case
        w_L_binned_local = w_L_binned
        w_L_e_L_binned_local = w_L_e_L_binned
        w_L_e_L2_binned_local = w_L_e_L2_binned

        w_L_binned_local = np.array(w_L_binned_local)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
        w_L_e_L2_binned_local = np.array(w_L_e_L2_binned_local)

        ## local sum
        w_L_binned_local_sum = np.sum(w_L_binned_local)
        w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local)
        w_L_e_L2_binned_local_sum = np.sum(w_L_e_L2_binned_local)

        ## glolbal sum
        w_L_binned_global_sum = mpi_comm.allreduce(np.sum(w_L_binned_local), op=MPI.SUM)
        w_L_e_L_binned_global_sum = mpi_comm.allreduce(np.sum(w_L_e_L_binned_local), op=MPI.SUM)
        w_L_e_L2_binned_global_sum = mpi_comm.allreduce(np.sum(w_L_e_L2_binned_local), op=MPI.SUM)

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

        # E: global sums
        sum_E_global = mpi_comm.allreduce(sum_E_local, op=MPI.SUM)
        sumsq_E_global = mpi_comm.allreduce(sumsq_E_local, op=MPI.SUM)

        E_mean = sum_E_global / M_total
        E_var = (sumsq_E_global / M_total) - (sum_E_global / M_total) ** 2
        E_std = np.sqrt((M_total - 1) * E_var)

        # Var: jackknife mean and std
        sum_Var_local = np.sum(Var_jackknife_binned_local)
        sumsq_Var_local = np.sum(Var_jackknife_binned_local**2)

        # Var: global sums
        sum_Var_global = mpi_comm.allreduce(sum_Var_local, op=MPI.SUM)
        sumsq_Var_global = mpi_comm.allreduce(sumsq_Var_local, op=MPI.SUM)

        Var_mean = sum_Var_global / M_total
        Var_var = (sumsq_Var_global / M_total) - (sum_Var_global / M_total) ** 2
        Var_std = np.sqrt((M_total - 1) * Var_var)

        logger.devel(f"E = {E_mean} +- {E_std} Ha.")
        logger.devel(f"Var(E) = {Var_mean} +- {Var_std} Ha^2.")

        return (E_mean, E_std, Var_mean, Var_std)

    def get_aF(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ):
        """Compute Hellmann–Feynman + Pulay forces with jackknife statistics.

        Args:
            num_mcmc_warmup_steps (int, optional): Samples to drop for warmup. Defaults to 50.
            num_mcmc_bin_blocks (int, optional): Number of jackknife blocks. Defaults to 10.

        Returns:
            tuple[npt.NDArray, npt.NDArray]: ``(force_mean, force_std)`` shaped ``(num_atoms, 3)`` in Hartree/bohr.

        Notes:
            Uses stored per-walker weights, energies, and pre-computed force products accumulated during :meth:`run`; reductions are MPI-aware.
        """
        w_L = self.w_L[num_mcmc_warmup_steps:]
        e_L = self.e_L[num_mcmc_warmup_steps:]
        force_HF = self.__stored_force_HF[num_mcmc_warmup_steps:]
        force_PP = self.__stored_force_PP[num_mcmc_warmup_steps:]
        E_L_force_PP = self.__stored_E_L_force_PP[num_mcmc_warmup_steps:]

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

        w_L_binned_local = w_L_binned
        w_L_e_L_binned_local = w_L_e_L_binned
        w_L_force_HF_binned_local = w_L_force_HF_binned
        w_L_force_PP_binned_local = w_L_force_PP_binned
        w_L_E_L_force_PP_binned_local = w_L_E_L_force_PP_binned

        w_L_binned_local = np.array(w_L_binned_local)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
        w_L_force_HF_binned_local = np.array(w_L_force_HF_binned_local)
        w_L_force_PP_binned_local = np.array(w_L_force_PP_binned_local)
        w_L_E_L_force_PP_binned_local = np.array(w_L_E_L_force_PP_binned_local)

        ## local sum
        w_L_binned_local_sum = np.sum(w_L_binned_local)
        w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local)
        w_L_force_HF_binned_local_sum = np.sum(w_L_force_HF_binned_local, axis=0)
        w_L_force_PP_binned_local_sum = np.sum(w_L_force_PP_binned_local, axis=0)
        w_L_E_L_force_PP_binned_local_sum = np.sum(w_L_E_L_force_PP_binned_local, axis=0)

        ## glolbal sum
        ### mpi allreduce (scalar)
        w_L_binned_global_sum = mpi_comm.allreduce(w_L_binned_local_sum, op=MPI.SUM)
        w_L_e_L_binned_global_sum = mpi_comm.allreduce(w_L_e_L_binned_local_sum, op=MPI.SUM)

        ### mpi Allreduce (array)
        w_L_force_HF_binned_global_sum = np.empty_like(w_L_force_HF_binned_local_sum)
        w_L_force_PP_binned_global_sum = np.empty_like(w_L_force_PP_binned_local_sum)
        w_L_E_L_force_PP_binned_global_sum = np.empty_like(w_L_E_L_force_PP_binned_local_sum)
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
                (w_L_force_HF_binned_global_sum - w_L_force_HF_binned_local[j]) / (w_L_binned_global_sum - w_L_binned_local[j])
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

        logger.devel(f"force_mean.shape  = {force_mean.shape}.")
        logger.devel(f"force_std.shape  = {force_std.shape}.")
        logger.devel(f"force = {force_mean} +- {force_std} Ha.")

        return (force_mean, force_std)

    def get_dln_WF(
        self,
        blocks: list,
        num_mcmc_warmup_steps: int = 50,
        chosen_param_index: list | None = None,
        lambda_projectors: tuple | None = None,
        num_orb_projection: int | None = None,
    ):
        """Assemble per-sample derivatives of ln Psi w.r.t. variational parameters.

        Args:
            blocks (list[VariationalParameterBlock]): Ordered variational blocks used for concatenation.
            num_mcmc_warmup_steps (int, optional): Samples to discard as warmup. Defaults to 50.
            chosen_param_index (list | None, optional): Optional subset of flattened indices; ``None`` keeps all. Defaults to None.
            lambda_projectors (tuple[npt.NDArray, npt.NDArray] | None, optional): ``(left_projector, right_projector)`` for lambda-subspace projection.
            num_orb_projection (int | None, optional): Number of paired/down-spin orbitals used to split the lambda block.

        Returns:
            npt.NDArray: ``O_matrix`` with shape ``(M, num_walkers, K)`` after warmup, where ``K`` follows the provided blocks (or subset).

        Notes:
            Validates the concatenated gradient size against block metadata and uses gradients stored during :meth:`run`.
        """
        grad_map = self.dln_Psi_dc

        # Collect gradients in the order of the provided variational blocks.
        dln_Psi_dc_list = []
        matched_blocks = []
        for block in blocks:
            if block.name in grad_map:
                dln_Psi_dc_list.append(grad_map[block.name])
                matched_blocks.append(block)

        # here, the third index indicates the flattened variational parameter index.
        O_matrix = np.empty((self.mcmc_counter, self.num_walkers, 0))

        for dln_Psi_dc, block in zip(dln_Psi_dc_list, matched_blocks):
            logger.devel(f"dln_Psi_dc.shape={dln_Psi_dc.shape}.")
            if dln_Psi_dc.ndim == 2:  # scalar variational param.
                dln_Psi_dc_reshaped = dln_Psi_dc.reshape(dln_Psi_dc.shape[0], dln_Psi_dc.shape[1], 1)
            else:
                dln_Psi_dc_reshaped = dln_Psi_dc.reshape(
                    dln_Psi_dc.shape[0], dln_Psi_dc.shape[1], int(np.prod(dln_Psi_dc.shape[2:]))
                )
            O_matrix = np.concatenate((O_matrix, dln_Psi_dc_reshaped), axis=2)

        # basic sanity check for consistency with blocks
        total_size_from_blocks = sum(block.size for block in blocks)
        assert total_size_from_blocks == O_matrix.shape[2], (
            "Mismatch between total block size and O_matrix parameter dimension."
        )

        # Step 2 projection (Nakano et al., JCP 152, 204121 (2020), Sec. D;
        # Becca and Sorella, 2017, Eq. 12.39):
        # Remove vir-vir component from derivative observables.
        #
        # Uses orthogonal-basis projectors to avoid oblique-projection amplification:
        #   1. Transform O_k to S^{-1/2}-orthogonalized basis
        #   2. Apply orthogonal projection:  Õ' = O' - (I-L') O' (I-R')
        #      where L' = S^{1/2} C_up C_up^T S^{1/2},  R' = S^{1/2} C_dn C_dn^T S^{1/2}
        #   3. Keep O' in orthogonal basis (theta will be back-transformed later)
        if lambda_projectors is not None and num_orb_projection is not None:
            left_projector, right_projector, inv_sqrt_overlap_up, inv_sqrt_overlap_dn = lambda_projectors
            identity = np.eye(left_projector.shape[0], dtype=np.float64)

            start = 0
            for block in blocks:
                end = start + block.size
                if block.name == "lambda_matrix":
                    block_matrix = O_matrix[:, :, start:end].reshape(O_matrix.shape[0], O_matrix.shape[1], *block.shape)
                    n_paired_cols = right_projector.shape[0]  # n_AO (projector dimension)
                    paired_block = block_matrix[:, :, :, :n_paired_cols]
                    unpaired_block = block_matrix[:, :, :, n_paired_cols:]

                    # Transform paired O_k to orthogonal basis: O' = S^{-1/2}_up @ O @ S^{-1/2}_dn
                    # Use @ with broadcasting over (m, w) batch dims instead of einsum for BLAS speed.
                    paired_orth = inv_sqrt_overlap_up @ paired_block @ inv_sqrt_overlap_dn
                    # Apply orthogonal projection: Õ' = O' - (I-L') O' (I-R')
                    comp_L = identity - left_projector
                    comp_R = identity - right_projector
                    correction = comp_L @ paired_orth @ comp_R
                    projected_paired = paired_orth - correction

                    # Transform unpaired to orthogonal basis: O'_unpaired = S^{-1/2}_up @ O
                    unpaired_orth = inv_sqrt_overlap_up @ unpaired_block

                    corrected_block = np.concatenate((projected_paired, unpaired_orth), axis=3)
                    O_matrix[:, :, start:end] = corrected_block.reshape(O_matrix.shape[0], O_matrix.shape[1], -1)
                    break
                start = end

        # ------------------------------------------------------------------
        # Symmetrize O_matrix for blocks with internal symmetry constraints.
        # This ensures that f, S, theta all respect the symmetry automatically,
        # removing the need for downstream symmetrization of S/N ratios,
        # parameter updates, etc.
        # All symmetrize_metric functions are batch-aware: they accept
        # (batch, block_size) input and symmetrize each row.
        # ------------------------------------------------------------------
        _sym_start = 0
        for block in blocks:
            _sym_end = _sym_start + block.size
            if block.symmetrize_metric is not None:
                _blk_slice = O_matrix[:, :, _sym_start:_sym_end]
                _orig_shape = _blk_slice.shape  # (M, nw, block_size)
                _flat_2d = _blk_slice.reshape(-1, block.size)  # (M*nw, block_size)
                _flat_2d[:] = block.symmetrize_metric(_flat_2d)
                O_matrix[:, :, _sym_start:_sym_end] = _flat_2d.reshape(_orig_shape)
            _sym_start = _sym_end

        logger.devel(f"O_matrix.shape = {O_matrix.shape}")
        if chosen_param_index is None:
            O_matrix_chosen = O_matrix[num_mcmc_warmup_steps:]
        else:
            O_matrix_chosen = O_matrix[num_mcmc_warmup_steps:, :, chosen_param_index]
        logger.devel(f"O_matrix_chosen.shape = {O_matrix_chosen.shape}")

        # ------------------------------------------------------------------
        # DEBUG: per-sample derivative statistics
        # Useful for diagnosing NaN propagation when epsilon_AS > 0.
        # ------------------------------------------------------------------
        _om = np.asarray(O_matrix_chosen)
        _om_nan = int(np.sum(~np.isfinite(_om)))
        _om_fin = _om[np.isfinite(_om)]
        logger.devel(
            "[dln_Psi/dc] shape=%s  NaN or Inf=%d/%d  min=%.3e  max=%.3e  mean=%.3e  std=%.3e",
            _om.shape,
            _om_nan,
            _om.size,
            float(np.min(_om_fin)) if _om_fin.size else float("nan"),
            float(np.max(_om_fin)) if _om_fin.size else float("nan"),
            float(np.mean(_om_fin)) if _om_fin.size else float("nan"),
            float(np.std(_om_fin)) if _om_fin.size else float("nan"),
        )
        if _om_nan > 0:
            nan_param_cols = np.where(np.any(~np.isfinite(_om.reshape(-1, _om.shape[-1])), axis=0))[0]
            logger.devel(
                "[dln_Psi/dc] param indices with NaN or Inf (first 30): %s",
                nan_param_cols[:30].tolist(),
            )

        return O_matrix_chosen

    def get_gF(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
        chosen_param_index: list | None = None,
        blocks: list | None = None,
        lambda_projectors: tuple | None = None,
        num_orb_projection: int | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Evaluate generalized forces (dE/dc_k) with jackknife error bars.

        Args:
            num_mcmc_warmup_steps (int, optional): Samples to discard as warmup. Defaults to 50.
            num_mcmc_bin_blocks (int, optional): Number of jackknife blocks. Defaults to 10.
            chosen_param_index (list | None, optional): Optional subset of flattened indices. Defaults to None.
            blocks (list | None, optional): Variational blocks for parameter ordering; defaults to current wavefunction blocks.
            lambda_projectors (tuple[npt.NDArray, npt.NDArray] | None, optional): Optional projectors forwarded to :meth:`get_dln_WF`.
            num_orb_projection (int | None, optional): Number of paired/down-spin orbitals for lambda projection.

        Returns:
            tuple[npt.NDArray, npt.NDArray]:
                ``(generalized_force_mean, generalized_force_std)`` as arrays over
                the filtered parameter space.

        Notes:
            Reuses :meth:`get_dln_WF` after warmup and applies jackknife statistics across MPI ranks.
        """
        w_L = self.w_L[num_mcmc_warmup_steps:]
        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))

        e_L = self.e_L[num_mcmc_warmup_steps:]
        w_L_e_L_split = np.array_split(np.einsum("iw,iw->iw", w_L, e_L), num_mcmc_bin_blocks, axis=0)
        w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))

        O_matrix = self.get_dln_WF(
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            chosen_param_index=chosen_param_index,
            blocks=blocks,
            lambda_projectors=lambda_projectors,
            num_orb_projection=num_orb_projection,
        )
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

        # MCMC case
        w_L_binned_local = w_L_binned
        w_L_e_L_binned_local = w_L_e_L_binned
        w_L_O_matrix_binned_local = w_L_O_matrix_binned
        w_L_e_L_O_matrix_binned_local = w_L_e_L_O_matrix_binned

        w_L_binned_local = np.array(w_L_binned_local)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
        w_L_O_matrix_binned_local = np.array(w_L_O_matrix_binned_local)
        w_L_e_L_O_matrix_binned_local = np.array(w_L_e_L_O_matrix_binned_local)

        ## local sum
        w_L_binned_local_sum = np.sum(w_L_binned_local)
        w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local)
        w_L_O_matrix_binned_local_sum = np.sum(w_L_O_matrix_binned_local, axis=0)
        w_L_e_L_O_matrix_binned_local_sum = np.sum(w_L_e_L_O_matrix_binned_local, axis=0)

        ## glolbal sum
        ### allreduce (scalar)
        w_L_binned_global_sum = mpi_comm.allreduce(w_L_binned_local_sum, op=MPI.SUM)
        w_L_e_L_binned_global_sum = mpi_comm.allreduce(w_L_e_L_binned_local_sum, op=MPI.SUM)

        ### Allreduce (array)
        w_L_O_matrix_binned_global_sum = np.empty_like(w_L_O_matrix_binned_local_sum)
        w_L_e_L_O_matrix_binned_global_sum = np.empty_like(w_L_e_L_O_matrix_binned_local_sum)
        mpi_comm.Allreduce(
            [w_L_O_matrix_binned_local_sum, MPI.DOUBLE], [w_L_O_matrix_binned_global_sum, MPI.DOUBLE], op=MPI.SUM
        )
        mpi_comm.Allreduce(
            [w_L_e_L_O_matrix_binned_local_sum, MPI.DOUBLE], [w_L_e_L_O_matrix_binned_global_sum, MPI.DOUBLE], op=MPI.SUM
        )

        ## jackknie binned samples
        M_local = w_L_binned_local.size
        M_total = mpi_comm.allreduce(M_local, op=MPI.SUM)

        eL_O_jn_local = np.array(
            [
                (w_L_e_L_O_matrix_binned_global_sum - w_L_e_L_O_matrix_binned_local[j])
                / (w_L_binned_global_sum - w_L_binned_local[j])
                for j in range(M_local)
            ]
        )

        eL_jn_local = np.array(
            [
                (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[j]) / (w_L_binned_global_sum - w_L_binned_local[j])
                for j in range(M_local)
            ]
        )

        O_jn_local = np.array(
            [
                (w_L_O_matrix_binned_global_sum - w_L_O_matrix_binned_local[j]) / (w_L_binned_global_sum - w_L_binned_local[j])
                for j in range(M_local)
            ]
        )

        bar_eL_bar_O_jn_local = np.einsum("i,ij->ij", eL_jn_local, O_jn_local)

        force_local = -2.0 * (eL_O_jn_local - bar_eL_bar_O_jn_local)  # (M_local, D)
        sum_local = np.sum(force_local, axis=0)  # shape (D,)
        sumsq_local = np.sum(force_local**2, axis=0)  # shape (D,)

        sum_global = np.empty_like(sum_local)
        sumsq_global = np.empty_like(sumsq_local)

        mpi_comm.Allreduce([sum_local, MPI.DOUBLE], [sum_global, MPI.DOUBLE], op=MPI.SUM)
        mpi_comm.Allreduce([sumsq_local, MPI.DOUBLE], [sumsq_global, MPI.DOUBLE], op=MPI.SUM)

        ## mean and var = E[x^2] - (E[x])^2
        mean_global = sum_global / M_total
        var_global = (sumsq_global / M_total) - (sum_global / M_total) ** 2

        ## mean and std
        generalized_force_mean = mean_global
        generalized_force_std = np.sqrt((M_total - 1) * var_global)

        logger.devel(f"generalized_force_mean = {generalized_force_mean}")
        logger.devel(f"generalized_force_std = {generalized_force_std}")
        logger.devel(f"generalized_force_mean.shape = {generalized_force_mean.shape}")
        logger.devel(f"generalized_force_std.shape = {generalized_force_std.shape}")

        return (
            generalized_force_mean,
            generalized_force_std,
        )  # (L vector, L vector)

    @staticmethod
    def _safe_stats(arr: npt.NDArray, name: str = "array") -> dict:
        """Calculate statistics safely, handling all-NaN arrays without warnings.

        Args:
            arr: Input array
            name: Name for logging

        Returns:
            dict with keys: shape, nan_count, nan_frac, min, max, mean, std
        """
        stats = {
            "shape": arr.shape,
            "nan_count": int(np.sum(np.isnan(arr))),
            "nan_frac": float(np.sum(np.isnan(arr)) / arr.size) if arr.size > 0 else 1.0,
        }

        # Only compute statistics if there are non-NaN values
        if stats["nan_frac"] < 1.0:
            with np.errstate(all="ignore"):  # Suppress warnings
                stats["min"] = float(np.nanmin(arr))
                stats["max"] = float(np.nanmax(arr))
                stats["mean"] = float(np.nanmean(arr))
                stats["std"] = float(np.nanstd(arr))
        else:
            stats["min"] = float("nan")
            stats["max"] = float("nan")
            stats["mean"] = float("nan")
            stats["std"] = float("nan")

        return stats

    def get_aH(
        self,
        blocks: list,
        g: npt.NDArray | None = None,
        num_mcmc_warmup_steps: int = 50,
        chosen_param_index: list | None = None,
        lambda_projectors: tuple | None = None,
        num_orb_projection: int | None = None,
        return_matrices: bool = False,
        collective_obs: npt.NDArray | None = None,
    ) -> tuple:
        r"""Compute aSR scalars or full LM matrices.

        **aSR mode** (``return_matrices=False``, default):

        Using the decomposition

        .. math::

            E_{\\alpha+\\delta\\alpha}
            = \\frac{H_0 + 2\\gamma H_1 + \\gamma^2 H_2}{1 + \\gamma^2 S_2},

        where :math:`\\delta\\alpha = \\gamma g` and :math:`g = S^{-1}f`.

        Returns ``(H_0, H_1, H_2, S_2)`` — scalars for aSR gamma optimization.
        Requires ``g`` (natural gradient direction).

        **LM mode** (``return_matrices=True``):

        Builds the full S, K, B matrices and force vector f for
        the Linear Method generalized eigenvalue problem.
        ``g`` is ignored.

        Returns ``(H_0, f_vec, S_matrix, K_matrix, B_matrix)``.

        Args:
            g (npt.NDArray | None): Natural gradient vector :math:`g = S^{-1}f`.
                Required when ``return_matrices=False``. Ignored when ``return_matrices=True``.
            blocks (list): Ordered variational blocks (same ordering as used for ``g``).
            num_mcmc_warmup_steps (int, optional): Samples to discard as warmup. Defaults to 50.
            chosen_param_index (list | None, optional): Optional subset of flattened parameter indices. Defaults to None.
            lambda_projectors (tuple | None, optional): Passed to :meth:`get_dln_WF`.
            num_orb_projection (int | None, optional): Passed to :meth:`get_dln_WF`.
            return_matrices (bool, optional): If True, return full matrices for LM. Defaults to False.

        Returns:
            tuple: ``(H_0, H_1, H_2, S_2)`` if ``return_matrices=False``, or
            ``(H_0, f_vec, S_matrix, K_matrix, B_matrix)`` if ``return_matrices=True``.

        Raises:
            RuntimeError: If ``compute_log_WF_param_deriv`` or ``comput_e_L_param_deriv`` is False.
        """
        if not self.__comput_log_WF_param_deriv:
            raise RuntimeError("get_aH requires compute_log_WF_param_deriv=True.")
        if not self.__comput_e_L_param_deriv:
            raise RuntimeError("get_aH requires comput_e_L_param_deriv=True (for B matrix / de_L/dc).")

        # Diagnostic output at entry
        logger.devel("[get_aH] ENTRY (return_matrices=%s)", return_matrices)
        if g is not None:
            g_stats = self._safe_stats(g, "g")
            logger.devel(
                f"[get_aH] INPUT g: shape={g_stats['shape']} "
                f"NaN={g_stats['nan_count']}/{g.size} min={g_stats['min']:.3e} "
                f"max={g_stats['max']:.3e} mean={g_stats['mean']:.3e}"
            )
        else:
            logger.devel("[get_aH] INPUT g: None (LM mode)")

        # ---- raw samples after warmup ----
        w_L = self.w_L[num_mcmc_warmup_steps:]  # (M, nw)
        e_L = self.e_L[num_mcmc_warmup_steps:]  # (M, nw)

        # ---- O_matrix: d ln Psi / dc  shape (M, nw, K) ----
        # When building LM matrices with a collective variable:
        #   - If collective_obs is provided (pre-computed O_SR), only fetch subspace params.
        #   - If g is provided but no collective_obs, fetch full params (to compute O_SR = dO @ g).
        _need_full_O = return_matrices and g is not None and collective_obs is None
        _cpi_for_dln = None if _need_full_O else chosen_param_index
        O_matrix = self.get_dln_WF(
            blocks=blocks,
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            chosen_param_index=_cpi_for_dln,
            lambda_projectors=lambda_projectors,
            num_orb_projection=num_orb_projection,
        )  # (M, nw, K_full) or (M, nw, K_sub)

        # Diagnostics: block.values (to detect parameter divergence)
        for block in blocks:
            vals = np.asarray(block.values)
            val_stats = self._safe_stats(vals, f"{block.name}.values")
            logger.devel(
                f"[get_aH] block.values['{block.name}']: shape={val_stats['shape']} "
                f"NaN={val_stats['nan_count']}/{vals.size} ({val_stats['nan_frac']:.2%}) "
                f"min={val_stats['min']:.3e} max={val_stats['max']:.3e} "
                f"mean={val_stats['mean']:.3e} std={val_stats['std']:.3e}"
            )

        # ---- dE_matrix: de_L / dc  shape (M, nw, K) ----
        de_L_dc_map = self.de_L_dc

        # Diagnostics: check de_L_dc_map
        logger.devel(f"[get_aH] de_L_dc_map keys: {list(de_L_dc_map.keys())}")
        for block_name, arr in de_L_dc_map.items():
            arr_stats = self._safe_stats(arr, f"de_L_dc[{block_name}]")
            logger.devel(
                f"[get_aH] de_L_dc['{block_name}']: shape={arr_stats['shape']} "
                f"NaN={arr_stats['nan_count']}/{arr.size} ({arr_stats['nan_frac']:.2%}) "
                f"min={arr_stats['min']:.3e} max={arr_stats['max']:.3e} "
                f"mean={arr_stats['mean']:.3e}"
            )

        de_L_dc_list = []
        for block in blocks:
            if block.name in de_L_dc_map:
                arr = de_L_dc_map[block.name]
                if arr.ndim == 2:
                    arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
                else:
                    arr = arr.reshape(arr.shape[0], arr.shape[1], int(np.prod(arr.shape[2:])))
                de_L_dc_list.append(arr)
        dE_matrix = np.concatenate(de_L_dc_list, axis=2)[num_mcmc_warmup_steps:]  # (M, nw, K)

        # ------------------------------------------------------------------
        # Transform dE_matrix to orthogonal basis when lambda_projectors is
        # active.  O_matrix was already transformed in get_dln_WF; dE_matrix
        # must be in the same basis for the B-matrix contribution
        #   g^T B g = <w * (g^T dO) * (g^T ddE)>_w
        # to be consistent (g is in orthogonal basis).
        # Same transform as get_dln_WF:
        #   paired:   dE' = S^{-1/2}_up @ dE @ S^{-1/2}_dn
        #   unpaired: dE' = S^{-1/2}_up @ dE
        # Projection (vir-vir removal) is NOT applied to dE because
        # g already has zero vir-vir components.
        # ------------------------------------------------------------------
        if lambda_projectors is not None and num_orb_projection is not None:
            _, _, inv_sqrt_overlap_up, inv_sqrt_overlap_dn = lambda_projectors
            _start = 0
            for block in blocks:
                _end = _start + block.size
                if block.name == "lambda_matrix":
                    dE_block = dE_matrix[:, :, _start:_end].reshape(dE_matrix.shape[0], dE_matrix.shape[1], *block.shape)
                    n_paired_cols = inv_sqrt_overlap_dn.shape[0]
                    dE_paired = dE_block[:, :, :, :n_paired_cols]
                    dE_unpaired = dE_block[:, :, :, n_paired_cols:]
                    # Two-sided transform for paired
                    dE_paired_orth = inv_sqrt_overlap_up @ dE_paired @ inv_sqrt_overlap_dn
                    # Left-only transform for unpaired
                    dE_unpaired_orth = inv_sqrt_overlap_up @ dE_unpaired
                    dE_corrected = np.concatenate((dE_paired_orth, dE_unpaired_orth), axis=3)
                    dE_matrix[:, :, _start:_end] = dE_corrected.reshape(dE_matrix.shape[0], dE_matrix.shape[1], -1)
                    logger.devel("[get_aH] Transformed dE_matrix lambda block to orthogonal basis.")
                    break
                _start = _end

        # Slice dE_matrix to match O_matrix, UNLESS we need full dE for
        # collective variable dE_SR computation (collective_obs path).
        _need_full_dE = return_matrices and g is not None and collective_obs is not None
        if _need_full_dE:
            pass  # keep full dE_matrix; will slice after computing dE_SR
        elif _cpi_for_dln is not None:
            dE_matrix = dE_matrix[:, :, chosen_param_index]
        elif chosen_param_index is not None and not (return_matrices and g is not None):
            dE_matrix = dE_matrix[:, :, chosen_param_index]
        # else: LM fallback (no collective_obs) — keep full dE_matrix, slice later

        # Diagnostics: dE_matrix
        dE_matrix_stats = self._safe_stats(dE_matrix, "dE_matrix")
        logger.devel(
            f"[get_aH] dE_matrix: shape={dE_matrix_stats['shape']} "
            f"NaN={dE_matrix_stats['nan_count']}/{dE_matrix.size} ({dE_matrix_stats['nan_frac']:.2%}) "
            f"min={dE_matrix_stats['min']:.3e} max={dE_matrix_stats['max']:.3e} "
            f"mean={dE_matrix_stats['mean']:.3e}"
        )

        # ---- flatten (M, nw) -> (N,) ----
        w_flat = np.ravel(w_L)  # (N,)
        e_flat = np.ravel(e_L)  # (N,)
        N = w_flat.shape[0]
        K = O_matrix.shape[2]
        K_dE = dE_matrix.shape[2]  # may differ from K when dE is kept full for collective_obs
        O_flat = O_matrix.reshape(N, K)  # (N, K)
        dE_flat = dE_matrix.reshape(N, K_dE)  # (N, K_dE)

        # Shape assertions
        assert w_flat.shape == (N,), f"w_flat shape {w_flat.shape} != ({N},)"
        assert e_flat.shape == (N,), f"e_flat shape {e_flat.shape} != ({N},)"
        assert O_flat.shape == (N, K), f"O_flat shape {O_flat.shape} != ({N}, {K})"
        assert dE_flat.shape == (N, K_dE), f"dE_flat shape {dE_flat.shape} != ({N}, {K_dE})"
        if collective_obs is not None:
            assert collective_obs.shape == (N,), f"collective_obs shape {collective_obs.shape} != ({N},)"
            assert K_dE >= K, f"dE must be full when collective_obs is used: K_dE={K_dE} < K={K}"
        else:
            assert K == K_dE, f"O and dE dimension mismatch: K={K} != K_dE={K_dE}"
        if g is not None:
            assert g.shape[0] == K_dE, f"g dimension {g.shape[0]} != K_dE={K_dE}"

        # Diagnostics: dE_flat
        dE_flat_stats = self._safe_stats(dE_flat, "dE_flat")
        logger.devel(
            f"[get_aH] dE_flat: shape={dE_flat_stats['shape']} "
            f"NaN={dE_flat_stats['nan_count']}/{dE_flat.size} ({dE_flat_stats['nan_frac']:.2%}) "
            f"min={dE_flat_stats['min']:.3e} max={dE_flat_stats['max']:.3e} "
            f"mean={dE_flat_stats['mean']:.3e} std={dE_flat_stats['std']:.3e}"
        )

        # ---- MPI-aware weighted averages ----
        total_w_local = np.sum(w_flat)
        we_local = np.dot(w_flat, e_flat)
        wO_local = np.einsum("i,ik->k", w_flat, O_flat)  # (K,)
        wdE_local = np.einsum("i,ik->k", w_flat, dE_flat)  # (K_dE,)

        # Diagnostics: wdE_local (BEFORE MPI Allreduce)
        wdE_local_stats = self._safe_stats(wdE_local, "wdE_local")
        logger.devel(
            f"[get_aH] wdE_local (BEFORE Allreduce): shape={wdE_local_stats['shape']} "
            f"NaN={wdE_local_stats['nan_count']}/{wdE_local.size} ({wdE_local_stats['nan_frac']:.2%}) "
            f"min={wdE_local_stats['min']:.3e} max={wdE_local_stats['max']:.3e} "
            f"mean={wdE_local_stats['mean']:.3e} std={wdE_local_stats['std']:.3e}"
        )

        total_w = mpi_comm.allreduce(total_w_local, op=MPI.SUM)

        we_global = np.empty(1)
        wO_global = np.empty(K)
        wdE_global = np.empty(K_dE)
        mpi_comm.Allreduce([np.array([we_local]), MPI.DOUBLE], [we_global, MPI.DOUBLE], op=MPI.SUM)
        mpi_comm.Allreduce([wO_local, MPI.DOUBLE], [wO_global, MPI.DOUBLE], op=MPI.SUM)
        mpi_comm.Allreduce([wdE_local, MPI.DOUBLE], [wdE_global, MPI.DOUBLE], op=MPI.SUM)

        # Diagnostics: wdE_global (AFTER Allreduce)
        wdE_global_stats = self._safe_stats(wdE_global, "wdE_global")
        logger.devel(
            f"[get_aH] wdE_global (AFTER Allreduce): shape={wdE_global_stats['shape']} "
            f"NaN={wdE_global_stats['nan_count']}/{wdE_global.size} ({wdE_global_stats['nan_frac']:.2%}) "
            f"min={wdE_global_stats['min']:.3e} max={wdE_global_stats['max']:.3e} "
            f"mean={wdE_global_stats['mean']:.3e} std={wdE_global_stats['std']:.3e}"
        )

        e_bar = float(we_global[0]) / total_w  # scalar
        O_bar = wO_global / total_w  # (K,)
        dE_bar = wdE_global / total_w  # (K,)

        # Diagnostics: dE_bar
        dE_bar_stats = self._safe_stats(dE_bar, "dE_bar")
        logger.devel(
            f"[get_aH] dE_bar: shape={dE_bar_stats['shape']} "
            f"NaN={dE_bar_stats['nan_count']}/{dE_bar.size} ({dE_bar_stats['nan_frac']:.2%}) "
            f"min={dE_bar_stats['min']:.3e} max={dE_bar_stats['max']:.3e} "
            f"mean={dE_bar_stats['mean']:.3e} std={dE_bar_stats['std']:.3e}"
        )

        # ---- H_0 ----
        H_0 = e_bar

        # ---- f_k = -2 <w (e_L - E)(O_k - O_bar_k)>  (generalized force) ----
        dO_flat = O_flat - O_bar[np.newaxis, :]  # (N, K)
        f_local = -2.0 * np.einsum("i,i,ik->k", w_flat, (e_flat - e_bar), dO_flat)
        f_global = np.empty(K)
        mpi_comm.Allreduce([f_local, MPI.DOUBLE], [f_global, MPI.DOUBLE], op=MPI.SUM)
        f_vec = f_global / total_w  # (K,)

        if return_matrices:
            # ---- LM mode: build full S, K, B matrices ----
            ddE_flat = dE_flat - dE_bar[np.newaxis, :]  # (N, K)

            # If g (SR direction) is provided, add collective variable to subspace.
            if g is not None:
                if collective_obs is not None:
                    # Memory-efficient path: O_SR pre-computed during SR solve.
                    # dO_flat is already subspace-only (fetched with chosen_param_index).
                    O_SR = collective_obs  # (N_local,)
                    # dE_SR needs full ddE — ddE_flat is full K_full here because
                    # dE_matrix was NOT sliced by _cpi_for_dln (see else branch below).
                    dE_SR = ddE_flat @ g  # (N,)
                    # Slice ddE to subspace (dO_flat is already subspace)
                    if chosen_param_index is not None:
                        ddE_flat = ddE_flat[:, chosen_param_index]
                else:
                    # Fallback: compute O_SR from full dO (needs full O_matrix).
                    O_SR = dO_flat @ g  # (N,)
                    dE_SR = ddE_flat @ g  # (N,)
                    if chosen_param_index is not None:
                        dO_flat = dO_flat[:, chosen_param_index]
                        ddE_flat = ddE_flat[:, chosen_param_index]

                # Prepend collective variable column
                dO_flat = np.column_stack([O_SR, dO_flat])  # (N, p+1)
                ddE_flat = np.column_stack([dE_SR, ddE_flat])  # (N, p+1)
                K = dO_flat.shape[1]

                # Recompute f_vec for the extended space
                f_local_ext = -2.0 * np.einsum("i,i,ik->k", w_flat, (e_flat - e_bar), dO_flat)
                f_global_ext = np.empty(K)
                mpi_comm.Allreduce([f_local_ext, MPI.DOUBLE], [f_global_ext, MPI.DOUBLE], op=MPI.SUM)
                f_vec = f_global_ext / total_w

            # Shape assertions before matrix construction
            assert dO_flat.shape == ddE_flat.shape, (
                f"dO/ddE shape mismatch after collective variable: dO={dO_flat.shape} ddE={ddE_flat.shape}"
            )
            assert dO_flat.shape[0] == N, f"dO_flat samples {dO_flat.shape[0]} != N={N}"
            assert f_vec.shape == (K,), f"f_vec shape {f_vec.shape} != ({K},)"

            # S_matrix = <w * dO_k * dO_k'>_w
            w_dO = w_flat[:, np.newaxis] * dO_flat  # (N, K)
            S_local = dO_flat.T @ w_dO  # (K, K)
            S_matrix = np.empty_like(S_local)
            mpi_comm.Allreduce(S_local, S_matrix, op=MPI.SUM)
            S_matrix /= total_w

            # K_matrix = <w * e_L * dO_k * dO_k'>_w
            we_dO = (w_flat * e_flat)[:, np.newaxis] * dO_flat  # (N, K)
            K_local = dO_flat.T @ we_dO  # (K, K)
            K_matrix = np.empty_like(K_local)
            mpi_comm.Allreduce(K_local, K_matrix, op=MPI.SUM)
            K_matrix /= total_w

            # B_matrix = <w * ddE_k * dO_k'>_w
            w_ddE = w_flat[:, np.newaxis] * ddE_flat  # (N, K)
            B_local = w_ddE.T @ dO_flat  # (K, K)
            B_matrix = np.empty_like(B_local)
            mpi_comm.Allreduce(B_local, B_matrix, op=MPI.SUM)
            B_matrix /= total_w

            # Final shape assertions
            assert S_matrix.shape == (K, K), f"S_matrix shape {S_matrix.shape} != ({K}, {K})"
            assert K_matrix.shape == (K, K), f"K_matrix shape {K_matrix.shape} != ({K}, {K})"
            assert B_matrix.shape == (K, K), f"B_matrix shape {B_matrix.shape} != ({K}, {K})"
            assert f_vec.shape == (K,), f"f_vec shape {f_vec.shape} != ({K},)"

            _p_label = f"p={K}" + (" (incl. SR collective)" if g is not None else "")
            logger.info(
                f"  LM matrices: {_p_label}, H_0={H_0:.6f}, "
                f"||f||={np.linalg.norm(f_vec):.3e}, "
                f"diag(S): min={np.min(np.diag(S_matrix)):.3e} max={np.max(np.diag(S_matrix)):.3e}"
            )
            return H_0, f_vec, S_matrix, K_matrix, B_matrix

        # ---- aSR mode: scalar projections ----
        assert g is not None, "g is required for aSR mode (return_matrices=False)"

        # ---- H_1 = -1/2 * g^T f ----
        H_1 = -0.5 * float(np.dot(g, f_vec))

        # ---- S_2 = g^T S g = <w * (g^T dO)^2>_w  (exact, computed from samples) ----
        # Do NOT use S_2 = g^T f = -2*H_1.  The SR solved (S_scaled + sr_epsilon*I) g_scaled = b,
        # so  g^T f = g^T S g + sr_epsilon * ||g_scaled||^2.
        # The sr_epsilon correction term overestimates S_2, making the denominator of
        # E(gamma) = (H0 + 2*gamma*H1 + gamma^2*H2) / (1 + gamma^2*S2) grow too fast,
        # which in turn makes the optimal gamma unrealistically small.
        gdO_flat = dO_flat @ g  # (N,)
        gSg_local = np.dot(w_flat, gdO_flat**2)
        gSg_arr = np.empty(1)
        mpi_comm.Allreduce([np.array([gSg_local]), MPI.DOUBLE], [gSg_arr, MPI.DOUBLE], op=MPI.SUM)
        S_2 = float(gSg_arr[0]) / total_w

        # ---- K matrix contribution: g^T K g = <w * e_L * (g^T dO)^2>_w ----
        gKg_local = np.einsum("i,i,i->", w_flat, e_flat, gdO_flat**2)
        gKg_arr = np.empty(1)
        mpi_comm.Allreduce([np.array([gKg_local]), MPI.DOUBLE], [gKg_arr, MPI.DOUBLE], op=MPI.SUM)
        gKg = float(gKg_arr[0]) / total_w

        # ---- B matrix contribution: g^T B g = <w * (g^T dO) * (g^T (dE - dE_bar))>_w ----
        ddE_flat = dE_flat - dE_bar[np.newaxis, :]  # (N, K)
        gdE_flat = ddE_flat @ g  # (N,)
        gBg_local = np.einsum("i,i,i->", w_flat, gdO_flat, gdE_flat)
        gBg_arr = np.empty(1)
        mpi_comm.Allreduce([np.array([gBg_local]), MPI.DOUBLE], [gBg_arr, MPI.DOUBLE], op=MPI.SUM)
        gBg = float(gBg_arr[0]) / total_w

        # ---- H_2 = g^T (B + K) g ----
        H_2 = gBg + gKg

        logger.info(f"aSR: H_0 = {H_0:.6f}, H_1 = {H_1:.6f}, H_2 = {H_2:.6f}, S_2 = {S_2:.6f}")
        return H_0, H_1, H_2, S_2

    @staticmethod
    def compute_asr_gamma(H_0: float, H_1: float, H_2: float, S_2: float) -> float:
        r"""Solve for the optimal gamma in the accelerated SR energy minimization.

        Finds :math:`\\gamma` satisfying

        .. math::

            \\frac{\\partial E_{\\alpha+\\gamma g}}{\\partial \\gamma} = 0,

        where :math:`E(\\gamma) = (H_0 + 2\\gamma H_1 + \\gamma^2 H_2) / (1 + \\gamma^2 S_2)`.

        Setting the derivative to zero yields the quadratic

        .. math::

            -H_1 S_2 \\gamma^2 + (H_2 - H_0 S_2)\\gamma + H_1 = 0,

        whose solution is

        .. math::

            \\gamma = \\frac{H_0 S_2 - H_2 \\pm
                     \\sqrt{(H_2 - H_0 S_2)^2 + 4 H_1^2 S_2}}
                    {-2 H_1 S_2}.

        The root with the smaller absolute value is returned.  This avoids
        spuriously large steps when the quadratic coefficient
        :math:`a = -H_1 S_2` is small,
        which causes one root to diverge while the other stays near zero.

        Args:
            H_0 (float): Current energy :math:`E_{\\alpha}`.
            H_1 (float): :math:`H_1 = -\\tfrac{1}{2} g^T f`.
            H_2 (float): :math:`H_2 = g^T (B+K) g`.
            S_2 (float): :math:`S_2 = g^T S g = \\langle w (g^T \\delta O)^2 \\rangle_w`.

        Returns:
            float: Optimal :math:`\\gamma`.
        """
        B = H_2 - H_0 * S_2  # coefficient of gamma in the quadratic
        discriminant = B**2 + 4.0 * H_1**2 * S_2
        if discriminant < 0.0:
            logger.warning(f"aSR: discriminant is negative ({discriminant:.3e}); setting to 0.")
            discriminant = 0.0
        denom = -2.0 * H_1 * S_2
        if abs(denom) < EPS_zero_division:
            logger.warning(
                "aSR: denom = -2*H_1*S_2 ~ 0 (H_1=%.3e, S_2=%.3e); no meaningful optimization direction. Returning gamma = 0.",
                H_1,
                S_2,
            )
            return 0.0
        sqrt_d = np.sqrt(discriminant)
        gamma_plus = (-B + sqrt_d) / denom
        gamma_minus = (-B - sqrt_d) / denom

        logger.info(f"aSR: gamma+ = {gamma_plus:.6f}, gamma- = {gamma_minus:.6f}")

        # Choose the root with the smaller absolute value (conservative step).
        if abs(gamma_plus) <= abs(gamma_minus):
            gamma = gamma_plus
        else:
            gamma = gamma_minus

        if gamma > 0.0:
            logger.debug("aSR: steepest-descent direction (gamma > 0).")
        else:
            logger.debug("aSR: anti-steepest-descent direction (gamma < 0).")

        logger.info(f"aSR: selected gamma = {gamma:.6f}")
        return gamma

    @staticmethod
    def solve_linear_method(
        H_0: float,
        f_vec: npt.NDArray,
        S_matrix: npt.NDArray,
        K_matrix: npt.NDArray,
        B_matrix: npt.NDArray,
        epsilon: float,
    ) -> tuple[npt.NDArray, float]:
        r"""Solve the Linear Method generalized eigenvalue problem.

        Constructs extended matrices :math:`\bar H` and :math:`\bar S` of
        dimension :math:`(p'+1) \times (p'+1)` (after S eigenvalue cutoff)
        and solves :math:`\bar H v = E \bar S v`.

        The eigenvector with the largest :math:`|v_0|^2` is selected, and the
        parameter update is :math:`c_k = v_k / v_0`.

        Args:
            H_0: Current energy :math:`E_\alpha`.
            f_vec: Generalized force vector, shape ``(p,)``.
            S_matrix: SR matrix, shape ``(p, p)``.
            K_matrix: K matrix, shape ``(p, p)``.
            B_matrix: B matrix, shape ``(p, p)``.
            epsilon: Eigenvalue cutoff for S matrix.

        Returns:
            tuple: ``(c_vec, E_lm)`` where ``c_vec`` has shape ``(p,)``
            (in the original parameter space) and ``E_lm`` is the selected eigenvalue.
        """
        p = len(f_vec)

        # ==================================================================
        # Preconditioning (dgelscut algorithm, following TurboRVB)
        #
        # 1. Remove parameters with negligible diag(S)
        # 2. Normalize S to a correlation matrix (diag = 1)
        # 3. Iteratively remove parameters until the minimum eigenvalue
        #    of the correlation matrix >= epsilon (= epsdgel)
        # This guarantees condition number <= 1/epsilon.
        # ==================================================================

        # ---- Step 1: Remove parameters with near-zero diag(S) ----
        diag_S = np.diag(S_matrix)
        max_diag_S = np.max(np.abs(diag_S))
        # parcut2 ~ machine_precision^2, effectively only removes exact zeros
        parcut2 = np.finfo(np.float64).eps ** 2
        alive = np.abs(diag_S) > parcut2 * max_diag_S
        n_removed_step1 = p - int(np.count_nonzero(alive))
        if n_removed_step1 > 0:
            logger.info("  LM dgelscut: Step 1 removed %d/%d parameters (tiny diag(S)).", n_removed_step1, p)

        if not np.any(alive):
            logger.warning("  LM dgelscut: all parameters removed in Step 1; returning zero update.")
            return np.zeros(p), H_0

        # ---- Step 2: Build correlation matrix for alive parameters ----
        alive_idx = np.where(alive)[0]
        D_inv_sqrt = np.zeros(p)
        D_inv_sqrt[alive_idx] = 1.0 / np.sqrt(np.abs(diag_S[alive_idx]))

        # ---- Step 3: Iteratively remove parameters until well-conditioned ----
        while True:
            idx = np.where(alive)[0]
            n_alive = len(idx)
            if n_alive == 0:
                logger.warning("  LM dgelscut: all parameters removed; returning zero update.")
                return np.zeros(p), H_0

            # Build correlation matrix for current alive set
            D_sub = D_inv_sqrt[idx]  # (n_alive,)
            S_sub = S_matrix[np.ix_(idx, idx)]  # (n_alive, n_alive)
            C = D_sub[:, np.newaxis] * S_sub * D_sub[np.newaxis, :]  # correlation matrix
            np.fill_diagonal(C, 1.0)  # enforce exact 1 on diagonal

            # Eigenvalue decomposition of correlation matrix
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            min_eigval = eigvals_C[0]  # smallest eigenvalue

            if min_eigval >= epsilon:
                break  # well-conditioned

            # Find the parameter contributing most to the smallest eigenvector
            # (the one with largest |component| in the problematic eigenvector)
            worst_local = int(np.argmax(np.abs(eigvecs_C[:, 0])))
            worst_global = idx[worst_local]
            alive[worst_global] = False
            logger.debug(
                "  LM dgelscut: removing param %d (min eigval=%.3e < eps=%.3e), %d remaining.",
                worst_global,
                min_eigval,
                epsilon,
                int(np.count_nonzero(alive)),
            )

        n_final = int(np.count_nonzero(alive))
        logger.info(
            "  LM dgelscut: %d/%d parameters kept (correlation matrix cond <= %.0f).",
            n_final,
            p,
            1.0 / epsilon,
        )

        # ==================================================================
        # Build LM matrices for the surviving parameters
        # ==================================================================
        idx = np.where(alive)[0]

        # Symmetrize H = K + B (B is not symmetric due to finite-sample noise)
        H_matrix = K_matrix + B_matrix
        H_matrix = 0.5 * (H_matrix + H_matrix.T)

        # Extract sub-matrices for alive parameters
        S_alive = S_matrix[np.ix_(idx, idx)]
        H_alive = H_matrix[np.ix_(idx, idx)]
        f_alive = f_vec[idx]

        # ---- S-orthonormalization: S = U Λ U^T, P = U Λ^{-1/2} ----
        eigvals_S, eigvecs_S = np.linalg.eigh(S_alive)
        # After dgelscut, all eigenvalues should be positive, but clip for safety
        pos_mask = eigvals_S > 0
        U = eigvecs_S[:, pos_mask]
        Lambda = eigvals_S[pos_mask]
        p_prime = len(Lambda)

        if p_prime == 0:
            logger.warning("  LM: no positive S eigenvalues after dgelscut; returning zero update.")
            return np.zeros(p), H_0

        # P = U Λ^{-1/2} (S-orthonormal basis)
        inv_sqrt_Lambda = 1.0 / np.sqrt(Lambda)
        P = U * inv_sqrt_Lambda[np.newaxis, :]  # (n_alive, p')

        # Transform H and f to S-orthonormal basis
        H_new = P.T @ H_alive @ P  # (p', p') — should be near-identity S
        f_new = P.T @ f_alive  # (p',)

        # ---- Build extended matrices (p'+1) x (p'+1) ----
        dim = p_prime + 1
        H_bar = np.zeros((dim, dim))
        S_bar = np.eye(dim)  # identity (S-orthonormal basis)

        H_bar[0, 0] = H_0
        H_bar[0, 1:] = -0.5 * f_new
        H_bar[1:, 0] = -0.5 * f_new
        H_bar[1:, 1:] = H_new

        # ---- Standard eigenvalue problem (S_bar = I) ----
        eigvals_lm, eigvecs_lm = np.linalg.eigh(H_bar)

        # ---- Select eigenvector with max |v_0|^2 ----
        v0_sq = eigvecs_lm[0, :] ** 2
        best_idx = int(np.argmax(v0_sq))
        E_lm = float(eigvals_lm[best_idx])

        # Diagnostic
        lowest_idx = 0
        if lowest_idx != best_idx:
            logger.debug(
                "  LM: lowest eigenvalue E_LM = %.6f (|v0|^2 = %.4f), selected (max |v0|^2) E_LM = %.6f (|v0|^2 = %.4f)",
                eigvals_lm[lowest_idx],
                v0_sq[lowest_idx],
                E_lm,
                v0_sq[best_idx],
            )
        else:
            logger.debug("  LM: selected eigenvalue E_LM = %.6f (|v0|^2 = %.4f)", E_lm, v0_sq[best_idx])

        if v0_sq[best_idx] < 0.01:
            logger.warning("  LM: max |v0|^2 = %.4f is small; update may be unreliable.", v0_sq[best_idx])

        w = eigvecs_lm[:, best_idx]
        w0 = w[0]
        c_new = w[1:] / w0  # (p',) in S-orthonormal basis

        # ---- Back-transform: P @ c_new → alive parameter space → full space ----
        c_alive = P @ c_new  # (n_alive,)
        c_vec = np.zeros(p)
        c_vec[idx] = c_alive

        logger.info(
            "  LM: E_LM = %.6f (|v0|^2 = %.4f), ||c|| = %.3e, max|c| = %.3e",
            E_lm,
            v0_sq[best_idx],
            np.linalg.norm(c_vec),
            np.max(np.abs(c_vec)),
        )

        return c_vec, E_lm

    def run_optimize(
        self,
        num_mcmc_steps: int = 100,
        num_opt_steps: int = 1,
        wf_dump_freq: int = 10,
        max_time: int = 86400,
        num_mcmc_warmup_steps: int = 0,
        num_mcmc_bin_blocks: int = 100,
        opt_J1_param: bool = True,
        opt_J2_param: bool = True,
        opt_J3_param: bool = True,
        opt_JNN_param: bool = True,
        opt_lambda_param: bool = False,
        opt_with_projected_MOs: bool = False,
        opt_J3_basis_exp: bool = False,
        opt_J3_basis_coeff: bool = False,
        opt_lambda_basis_exp: bool = False,
        opt_lambda_basis_coeff: bool = False,
        optimizer_kwargs: dict | None = None,
    ):
        """Optimize wavefunction parameters using SR or optax.

        Args:
            num_mcmc_steps (int, optional): MCMC samples per walker per iteration. Defaults to 100.
            num_opt_steps (int, optional): Number of optimization iterations. Defaults to 1.
            wf_dump_freq (int, optional): Dump frequency for ``hamiltonian_data.chk``. Defaults to 10.
            max_time (int, optional): Per-iteration MCMC wall-clock budget (sec). Defaults to 86400.
            num_mcmc_warmup_steps (int, optional): Warmup samples discarded each iteration. Defaults to 0.
            num_mcmc_bin_blocks (int, optional): Jackknife bins for statistics. Defaults to 100.
            opt_J1_param (bool, optional): Optimize one-body Jastrow. Defaults to True.
            opt_J2_param (bool, optional): Optimize two-body Jastrow. Defaults to True.
            opt_J3_param (bool, optional): Optimize three-body Jastrow. Defaults to True.
            opt_JNN_param (bool, optional): Optimize NN Jastrow. Defaults to True.
            opt_lambda_param (bool, optional): Optimize determinant lambda matrix. Defaults to False.
            opt_with_projected_MOs (bool, optional): If True, enable AO-space lambda optimization flow.
                At every optimization step, build projection operators from the current MO state,
                convert MO->AO for MCMC/gradient evaluation, update AO parameters, then project AO->MO
                with fixed ``num_eigenvectors=num_electron_dn`` to finish the step.
            opt_J3_basis_exp (bool, optional): Optimize J3 AO Gaussian exponents. Defaults to False.
                Cannot be combined with ``opt_with_projected_MOs``.
            opt_J3_basis_coeff (bool, optional): Optimize J3 AO contraction coefficients. Defaults to False.
                Cannot be combined with ``opt_with_projected_MOs``.
            opt_lambda_basis_exp (bool, optional): Optimize Geminal AO Gaussian exponents (up and down). Defaults to False.
                Cannot be combined with ``opt_with_projected_MOs``.
            opt_lambda_basis_coeff (bool, optional): Optimize Geminal AO contraction coefficients (up and down). Defaults to False.
                Cannot be combined with ``opt_with_projected_MOs``.
            optimizer_kwargs (dict | None, optional): Optimizer configuration.
                ``method='sr'`` uses SR keys (``sr_delta``, ``sr_epsilon``, ``cg_flag``,
                ``cg_max_iter``, ``cg_tol``, ``adaptive_learning_rate``);
                ``use_lm=True`` enables LM with keys (``lm_subspace_dim``, ``lm_cond``,
                ``lm_subspace_dim``, ``lm_cond``); ``adaptive_learning_rate=True`` (SR only) enables
                accelerated SR (aSR) gamma scaling and requires
                ``compute_log_WF_param_deriv=True`` and ``comput_e_L_param_deriv=True``;
                other ``method`` names are optax constructors (e.g., ``"adam"``) and
                receive remaining keys.

        Notes:
            - Persists optax optimizer state across calls when method and hyperparameters match.
            - Writes ``external_control_opt.toml`` to allow external stop requests.
            - Updates :class:`Hamiltonian_data` in-place and increments the optimization counter.
        """
        optax_supported = {
            "sgd": optax.sgd,
            "adam": optax.adam,
            "adamw": optax.adamw,
            "rmsprop": optax.rmsprop,
            "adagrad": optax.adagrad,
            "yogi": optax.yogi,
        }

        optimizer_kwargs = dict(optimizer_kwargs or {})
        optimizer_mode = optimizer_kwargs.pop("method", "sr")
        if not isinstance(optimizer_mode, str):
            raise TypeError("optimizer_kwargs['method'] must be a string if provided.")
        optimizer_mode = optimizer_mode.lower()
        use_sr = optimizer_mode == "sr"
        optax_name = optimizer_mode if not use_sr else None

        # SR parameters (including LM options)
        delta = epsilon = sr_cg_flag = sr_cg_max_iter = sr_cg_tol = None
        use_lm = False
        lm_subspace_dim = 0
        if use_sr:
            delta = float(optimizer_kwargs.get("delta", 1.0e-3))
            epsilon = float(optimizer_kwargs.get("epsilon", 1.0e-3))
            sr_cg_flag = bool(optimizer_kwargs.get("cg_flag", True))
            sr_cg_max_iter = int(optimizer_kwargs.get("cg_max_iter", int(1e6)))
            sr_cg_tol = float(optimizer_kwargs.get("cg_tol", 1.0e-8))
            use_lm = bool(optimizer_kwargs.get("use_lm", False))
            lm_subspace_dim = int(optimizer_kwargs.get("lm_subspace_dim", 0))
            lm_cond = float(optimizer_kwargs.get("lm_cond", 1.0e-3))

        # use_lm requires derivative computations
        if use_lm:
            if not self.__comput_log_WF_param_deriv:
                raise RuntimeError("use_lm requires compute_log_WF_param_deriv=True.")
            if not self.__comput_e_L_param_deriv:
                raise RuntimeError("use_lm requires comput_e_L_param_deriv=True.")

        optax_kwargs = {
            k: v
            for k, v in optimizer_kwargs.items()
            if k
            not in {
                "delta",
                "epsilon",
                "cg_flag",
                "cg_max_iter",
                "cg_tol",
                "cg_x0_strategy",
                "use_lm",
                "lm_subspace_dim",
                "lm_cond",
            }
        }

        optax_tx = None
        optax_state = None
        optax_param_size = None

        self.__ensure_optimizer_runtime()
        optimizer_runtime = self.__optimizer_runtime
        stored_method = optimizer_runtime.get("method")
        stored_hparams = optimizer_runtime.get("hyperparameters")
        stored_optax_state = optimizer_runtime.get("optax_state")
        stored_param_size = optimizer_runtime.get("optax_param_size")

        optimizer_hparams: dict[str, float | int | str] = {"method": optimizer_mode}

        if not use_sr:
            if optax_name not in optax_supported:
                raise ValueError(
                    f"Unsupported optimizer '{optimizer_mode}'. Supported optax options: {sorted(optax_supported)}."
                )
            optax_config = dict(optax_kwargs)
            optax_config.setdefault("learning_rate", 1.0e-3)
            optax_tx = optax_supported[optax_name](**optax_config)
            optimizer_hparams = {"method": optimizer_mode, **optax_config}
        else:
            optimizer_hparams = {
                "method": optimizer_mode,
                "delta": delta,
                "epsilon": epsilon,
                "cg_flag": sr_cg_flag,
                "cg_max_iter": sr_cg_max_iter,
                "cg_tol": sr_cg_tol,
                "use_lm": use_lm,
                "lm_subspace_dim": lm_subspace_dim,
                "lm_cond": lm_cond,
            }

        if use_sr:
            self.__set_optimizer_runtime(
                method=optimizer_mode,
                hyperparameters=optimizer_hparams,
                optax_state=None,
                optax_param_size=None,
            )
        else:
            if stored_method == optimizer_mode and stored_hparams == optimizer_hparams:
                optax_state = stored_optax_state
                optax_param_size = stored_param_size
                if optax_state is not None:
                    logger.info("Resuming optax '%s' optimizer state from checkpoint.", optax_name)
            else:
                if stored_optax_state is not None:
                    if stored_method is not None and stored_method != optimizer_mode:
                        logger.info(
                            "Stored optimizer state for method '%s' ignored because '%s' was requested.",
                            stored_method,
                            optimizer_mode,
                        )
                    else:
                        logger.info("Stored optimizer hyperparameters differ from requested values; resetting optimizer state.")
                self.__set_optimizer_runtime(
                    method=optimizer_mode,
                    hyperparameters=optimizer_hparams,
                    optax_state=None,
                    optax_param_size=None,
                )

        # optimizer info
        logger.info(f"The chosen optimizer is '{optimizer_mode}'.")
        if use_sr:
            logger.info("  The homemade 'SR (aka natural gradient)' optimizer is used for wavefunction optimization.")
            if use_lm and lm_subspace_dim != 0:
                logger.info(
                    "  use_lm=True, lm_subspace_dim=%d: Linear Method with SR collective variable.",
                    lm_subspace_dim,
                )
            elif use_lm:
                logger.info("  use_lm=True, lm_subspace_dim=0: accelerated SR (aSR) gamma scaling.")
            logger.info("  Hyperparameters: %s", ", ".join(f"{k}={v}" for k, v in sorted(optimizer_hparams.items())))
            logger.info("")
        else:
            logger.info(f"  The optax '{optax_name}' optimizer is used for wavefunction optimization.")
            logger.info("  Hyperparameters: %s", ", ".join(f"{k}={v}" for k, v in sorted(optimizer_hparams.items())))
            logger.info("")

        def _conjugate_gradient_numpy(
            b: npt.NDArray[np.float64],
            apply_A,
            x0: npt.NDArray[np.float64],
            max_iter: int,
            tol: float,
        ) -> tuple[npt.NDArray[np.float64], float, int]:
            x = np.array(x0, dtype=np.float64, copy=True)
            r = np.array(b, dtype=np.float64, copy=False) - apply_A(x)
            p = np.array(r, dtype=np.float64, copy=True)
            rs_old = float(np.dot(r, r))

            if not np.isfinite(rs_old):
                raise FloatingPointError("Non-finite initial residual encountered in SR-CG.")

            if np.sqrt(rs_old) <= tol:
                return x, np.sqrt(rs_old), 0

            tiny = np.finfo(np.float64).tiny
            num_iter = 0
            for i in range(int(max_iter)):
                Ap = apply_A(p)
                denom = float(np.dot(p, Ap))
                if not np.isfinite(denom) or abs(denom) <= tiny:
                    logger.warning(
                        "[CG] Breakdown detected (non-finite/near-zero denominator) at iteration %d; terminating early.",
                        i,
                    )
                    break

                alpha = rs_old / denom
                x = x + alpha * p
                r = r - alpha * Ap
                rs_new = float(np.dot(r, r))
                num_iter = i + 1

                if not np.isfinite(rs_new):
                    raise FloatingPointError("Non-finite residual encountered in SR-CG.")

                if np.sqrt(rs_new) <= tol:
                    rs_old = rs_new
                    break

                beta = rs_new / rs_old
                p = r + beta * p
                rs_old = rs_new

            return x, np.sqrt(rs_old), num_iter

        self.__set_param_gradient_flags(
            j1_param=opt_J1_param,
            j2_param=opt_J2_param,
            j3_matrix=opt_J3_param,
            jastrow_nn_params=opt_JNN_param,
            lambda_matrix=opt_lambda_param,
            j3_basis_exp=opt_J3_basis_exp,
            j3_basis_coeff=opt_J3_basis_coeff,
            lambda_basis_exp=opt_lambda_basis_exp,
            lambda_basis_coeff=opt_lambda_basis_coeff,
        )

        if opt_with_projected_MOs:
            if not opt_lambda_param:
                raise ValueError("opt_with_projected_MOs=True requires opt_lambda_param=True.")

            if any([opt_lambda_basis_exp, opt_lambda_basis_coeff]):
                raise ValueError(
                    "Geminal AO basis optimization (opt_lambda_basis_exp/coeff) "
                    "cannot be combined with opt_with_projected_MOs. "
                    "Changing Geminal AO exponents/coefficients invalidates the overlap matrix "
                    "used by the MO projection operators."
                )

            geminal_init = self.hamiltonian_data.wavefunction_data.geminal_data
            if not (geminal_init.is_mo_representation or geminal_init.is_ao_representation):
                raise ValueError(
                    "opt_with_projected_MOs=True requires geminal orbital representation to be either MO/MO or AO/AO."
                )

            if geminal_init.is_mo_representation:
                initial_geminal_is_ao_representation = False
                logger.info("opt_with_projected_MOs=True: initial geminal is MO representation.")
            else:
                initial_geminal_is_ao_representation = True
                logger.info("opt_with_projected_MOs=True: initial geminal is AO representation.")

                geminal_init_mo = Geminal_data.convert_from_AOs_to_MOs(
                    geminal_init,
                    num_eigenvectors="all",
                )
                wavefunction_data_mo = type(self.hamiltonian_data.wavefunction_data)(
                    jastrow_data=self.hamiltonian_data.wavefunction_data.jastrow_data,
                    geminal_data=geminal_init_mo,
                )
                self.hamiltonian_data = Hamiltonian_data(
                    structure_data=self.hamiltonian_data.structure_data,
                    wavefunction_data=wavefunction_data_mo,
                    coulomb_potential_data=self.hamiltonian_data.coulomb_potential_data,
                )
                logger.info("opt_with_projected_MOs=True: converted initial AO geminal to MO before optimization loop.")

        # toml(control) filename
        toml_filename = "external_control_opt.toml"

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

        # timer
        vmcopt_total_start = time.perf_counter()

        # timer_counter
        timer_opt_total = 0.0
        timer_mcmc_run = 0.0
        timer_get_E = 0.0
        timer_get_gF = 0.0
        timer_optimizer = 0.0
        timer_param_update = 0.0
        timer_MPI_barrier = 0.0
        num_opt_done = 0

        sr_cg_warm_start_primal = None
        sr_cg_warm_start_dual = None

        # main vmcopt loop
        for i_opt in range(num_opt_steps):
            logger.info("=" * num_sep_line)
            logger.info(f"Optimization step = {i_opt + 1 + self.__i_opt}/{num_opt_steps + self.__i_opt}.")
            logger.info("=" * num_sep_line)

            logger.info(f"MCMC steps this iteration = {num_mcmc_steps}.")
            logger.info(f"Warmup steps = {num_mcmc_warmup_steps}.")
            logger.info(f"Bin blocks = {num_mcmc_bin_blocks}.")
            logger.info("")

            lambda_projectors = None
            num_orb_projection = None
            if opt_with_projected_MOs:
                wavefunction_data_step = self.hamiltonian_data.wavefunction_data
                geminal_mo_current = wavefunction_data_step.geminal_data
                num_orb_projection = int(geminal_mo_current.num_electron_dn)

                mo_coefficients_up = np.asarray(geminal_mo_current.orb_data_up_spin.mo_coefficients, dtype=np.float64)
                mo_coefficients_dn = np.asarray(geminal_mo_current.orb_data_dn_spin.mo_coefficients, dtype=np.float64)
                overlap_up = np.asarray(compute_overlap_matrix(geminal_mo_current.orb_data_up_spin.aos_data), dtype=np.float64)
                overlap_dn = np.asarray(compute_overlap_matrix(geminal_mo_current.orb_data_dn_spin.aos_data), dtype=np.float64)
                overlap_up = 0.5 * (overlap_up + overlap_up.T)
                overlap_dn = 0.5 * (overlap_dn + overlap_dn.T)

                p_matrix_cols_up = mo_coefficients_up[:num_orb_projection, :].T  # (n_AO, n_dn)
                p_matrix_cols_dn = mo_coefficients_dn[:num_orb_projection, :].T  # (n_AO, n_dn)

                # Build S^{1/2} and S^{-1/2} via eigendecomposition
                eigvals_up, eigvecs_up = np.linalg.eigh(overlap_up)
                eigvals_dn, eigvecs_dn = np.linalg.eigh(overlap_dn)
                sqrt_overlap_up = eigvecs_up @ np.diag(np.sqrt(eigvals_up)) @ eigvecs_up.T
                sqrt_overlap_dn = eigvecs_dn @ np.diag(np.sqrt(eigvals_dn)) @ eigvecs_dn.T
                inv_sqrt_overlap_up = eigvecs_up @ np.diag(1.0 / np.sqrt(eigvals_up)) @ eigvecs_up.T
                inv_sqrt_overlap_dn = eigvecs_dn @ np.diag(1.0 / np.sqrt(eigvals_dn)) @ eigvecs_dn.T

                # Orthogonal-basis MO coefficients: C' = S^{1/2} C
                orth_mo_up = sqrt_overlap_up @ p_matrix_cols_up  # (n_AO, n_dn)
                orth_mo_dn = sqrt_overlap_dn @ p_matrix_cols_dn  # (n_AO, n_dn)

                # Orthogonal projectors (symmetric, idempotent in Euclidean metric)
                # L' = S^{1/2} C_up C_up^T S^{1/2},  R' = S^{1/2} C_dn C_dn^T S^{1/2}
                left_projector = orth_mo_up @ orth_mo_up.T  # (n_AO, n_AO)
                right_projector = orth_mo_dn @ orth_mo_dn.T  # (n_AO, n_AO)

                lambda_projectors = (left_projector, right_projector, inv_sqrt_overlap_up, inv_sqrt_overlap_dn)
                logger.devel(
                    "opt_with_projected_MOs: P_up.shape=%s, P_dn.shape=%s, S_up.shape=%s, S_dn.shape=%s",
                    p_matrix_cols_up.shape,
                    p_matrix_cols_dn.shape,
                    overlap_up.shape,
                    overlap_dn.shape,
                )

                # ------------------------------------------------------------------
                # DEVEL: orthogonal complement-projector diagnostics  (I - L') and (I - R')
                # ------------------------------------------------------------------
                _I = np.eye(left_projector.shape[0], dtype=np.float64)
                _comp_L = _I - left_projector  # (I - L')  — symmetric
                _comp_R = _I - right_projector  # (I - R')  — symmetric

                # basic statistics
                logger.devel(
                    "[projector] (I - L'): shape=%s  min=%.6e  max=%.6e  Frobenius=%.6e",
                    _comp_L.shape,
                    float(np.min(_comp_L)),
                    float(np.max(_comp_L)),
                    float(np.linalg.norm(_comp_L, "fro")),
                )
                logger.devel(
                    "[projector] (I - R'): shape=%s  min=%.6e  max=%.6e  Frobenius=%.6e",
                    _comp_R.shape,
                    float(np.min(_comp_R)),
                    float(np.max(_comp_R)),
                    float(np.linalg.norm(_comp_R, "fro")),
                )
                # symmetry check
                _sym_err_L = float(np.linalg.norm(left_projector - left_projector.T, "fro"))
                _sym_err_R = float(np.linalg.norm(right_projector - right_projector.T, "fro"))
                logger.devel(
                    "[projector] symmetry ||L' - L'^T||_F = %.6e,  ||R' - R'^T||_F = %.6e",
                    _sym_err_L,
                    _sym_err_R,
                )

                # idempotency check: (I - L')^2 == (I - L')  and  (I - R')^2 == (I - R')
                _comp_L_sq = _comp_L @ _comp_L
                _comp_R_sq = _comp_R @ _comp_R
                _idem_err_L = float(np.linalg.norm(_comp_L_sq - _comp_L, "fro"))
                _idem_err_R = float(np.linalg.norm(_comp_R_sq - _comp_R, "fro"))
                _idem_ok_L = "OK" if _idem_err_L < atol_consistency else "FAIL"
                _idem_ok_R = "OK" if _idem_err_R < atol_consistency else "FAIL"
                logger.devel(
                    "[projector] idempotency ||(I-L')^2 - (I-L')||_F = %.6e  [%s]",
                    _idem_err_L,
                    _idem_ok_L,
                )
                logger.devel(
                    "[projector] idempotency ||(I-R')^2 - (I-R')||_F = %.6e  [%s]",
                    _idem_err_R,
                    _idem_ok_R,
                )
                # spectral norm of complement projector (should be exactly 1 for orthogonal projector)
                _specnorm_L = float(np.linalg.norm(_comp_L, 2))
                _specnorm_R = float(np.linalg.norm(_comp_R, 2))
                logger.devel(
                    "[projector] spectral norm ||(I-L')||_2 = %.6e,  ||(I-R')||_2 = %.6e  (should be 1.0)",
                    _specnorm_L,
                    _specnorm_R,
                )
                # clean up temporaries
                del _I, _comp_L, _comp_R, _comp_L_sq, _comp_R_sq

                geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_current)
                wavefunction_data_ao = type(wavefunction_data_step)(
                    jastrow_data=wavefunction_data_step.jastrow_data,
                    geminal_data=geminal_ao,
                )

                self.hamiltonian_data = Hamiltonian_data(
                    structure_data=self.hamiltonian_data.structure_data,
                    wavefunction_data=wavefunction_data_ao,
                    coulomb_potential_data=self.hamiltonian_data.coulomb_potential_data,
                )
                logger.info("opt_with_projected_MOs=True: converted current MO geminal to AO for this optimization step.")

            # run MCMC
            start = time.perf_counter()
            self.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)
            end = time.perf_counter()
            timer_mcmc_run += end - start

            # get E
            start = time.perf_counter()
            E, E_std, _, _ = self.get_E(num_mcmc_warmup_steps=num_mcmc_warmup_steps, num_mcmc_bin_blocks=num_mcmc_bin_blocks)
            end = time.perf_counter()
            timer_get_E += end - start
            logger.info("Total Energy before update of wavefunction.")
            logger.info("-" * num_sep_line)
            logger.info(f"E = {E:.5f} +- {E_std:.5f} Ha")
            logger.info("-" * num_sep_line)
            logger.info("")

            # Abort optimization if energy is NaN/Inf — the wavefunction is
            # corrupted and further updates would be meaningless.
            if not np.isfinite(E):
                logger.error(
                    "Non-finite energy detected (E=%s). Aborting optimization loop.",
                    E,
                )
                break

            # collect variational parameter blocks from the wavefunction data
            blocks = self.hamiltonian_data.wavefunction_data.get_variational_blocks(
                opt_J1_param=opt_J1_param,
                opt_J2_param=opt_J2_param,
                opt_J3_param=opt_J3_param,
                opt_JNN_param=opt_JNN_param,
                opt_lambda_param=opt_lambda_param,
                opt_J3_basis_exp=opt_J3_basis_exp,
                opt_J3_basis_coeff=opt_J3_basis_coeff,
                opt_lambda_basis_exp=opt_lambda_basis_exp,
                opt_lambda_basis_coeff=opt_lambda_basis_coeff,
            )

            # flatten index mapping for the blocks
            offsets = []
            start = 0
            for block in blocks:
                offsets.append((block, start, start + block.size))
                start += block.size
            total_num_params = start

            logger.info(f"Number of variational parameters = {total_num_params}.")

            if not (use_sr or use_lm):
                if blocks:
                    flat_param_vector = np.concatenate([np.ravel(np.array(block.values, dtype=np.float64)) for block in blocks])
                else:
                    flat_param_vector = np.array([], dtype=np.float64)

                if optax_state is None:
                    optax_param_size = flat_param_vector.size
                    optax_state = optax_tx.init(jnp.array(flat_param_vector))
                elif flat_param_vector.size != optax_param_size:
                    raise ValueError("The number of variational parameters changed after initializing the optax optimizer.")

            # get f and f_std (generalized forces)
            start = time.perf_counter()
            f, f_std = self.get_gF(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
                blocks=blocks,
                lambda_projectors=lambda_projectors,
                num_orb_projection=num_orb_projection,
            )
            end = time.perf_counter()
            timer_get_gF += end - start

            if mpi_rank == 0:
                logger.devel(f"shape of f = {f.shape}.")
                logger.devel(f"f_std.shape = {f_std.shape}.")
                tol = np.finfo(f_std.dtype).eps if np.issubdtype(f_std.dtype, np.floating) else 0.0
                finite_mask = np.isfinite(f_std) & (np.abs(f_std) > tol)
                signal_to_noise_f = np.zeros_like(f)
                signal_to_noise_f[finite_mask] = np.abs(f[finite_mask]) / f_std[finite_mask]
                f_argmax = np.argmax(np.abs(f))
                logger.info("-" * num_sep_line)
                logger.info(f"Max f = {f[f_argmax]:.3f} +- {f_std[f_argmax]:.3f} Ha/a.u.")

                # S/N symmetrization is no longer needed — O_k is symmetrized
                # at source in get_dln_WF, so f and f_std are already symmetric.

                logger.info(f"Max of signal-to-noise of f = max(|f|/|std f|) = {np.max(signal_to_noise_f):.3f}.")
                logger.info("-" * num_sep_line)

                # Show top parameters by SN ratio (diagnostic)
                _sn_sorted_indices = np.argsort(signal_to_noise_f)[::-1]
                _show_k = min(30, len(_sn_sorted_indices))
                _show_indices = _sn_sorted_indices[:_show_k]
                _sel_info = []
                for _idx in _show_indices:
                    _block_name = "unknown"
                    _local_idx = int(_idx)
                    for _blk, _s, _e in offsets:
                        if _s <= _idx < _e:
                            _block_name = _blk.name
                            _local_idx = int(_idx - _s)
                            break
                    _sel_info.append(
                        f"  flat={_idx} block={_block_name} local={_local_idx} "
                        f"|f|/std={signal_to_noise_f[_idx]:.3f} f={f[_idx]:.6f}"
                    )
                logger.debug("Top %d parameters by |f|/|std f|:", _show_k)
                for _line in _sel_info:
                    logger.debug(_line)
            else:
                signal_to_noise_f = None

            signal_to_noise_f = mpi_comm.bcast(signal_to_noise_f, root=0)

            #############################
            # optimization step
            #############################
            start = time.perf_counter()
            if use_sr:
                logger.info("Computing the natural gradient, i.e., {S+epsilon*I}^{-1}*f")

                # Retrieve local data (samples assigned to this rank)
                w_L_local = self.w_L[num_mcmc_warmup_steps:]  # shape: (num_mcmc, num_walker)
                e_L_local = self.e_L[num_mcmc_warmup_steps:]  # shape: (num_mcmc, num_walker)
                w_L_local = np.ravel(w_L_local)  # shape: (num_mcmc * num_walker, )
                e_L_local = np.ravel(e_L_local)  # shape: (num_mcmc * num_walker, )
                O_matrix_local = self.get_dln_WF(
                    num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                    blocks=blocks,
                    lambda_projectors=lambda_projectors,
                    num_orb_projection=num_orb_projection,
                )  # shape: (num_mcmc, num_walker, num_param)
                O_matrix_local_shape = (
                    O_matrix_local.shape[0] * O_matrix_local.shape[1],
                    O_matrix_local.shape[2],
                )
                O_matrix_local = O_matrix_local.reshape(O_matrix_local_shape)  # shape: (num_mcmc * num_walker, num_param)

                # Compute local partial sums
                local_Ow = np.einsum("i,ij->j", w_L_local, O_matrix_local)  # weighted sum for observables, shape: (num_param,)
                local_Ew = np.dot(w_L_local, e_L_local)  # weighted sum of energies, shape: scalar
                local_weight_sum = np.sum(w_L_local)  # scalar: sum of weights, shape: scalar

                # Aggregate across all ranks
                total_weight = mpi_comm.allreduce(local_weight_sum, op=MPI.SUM)  # total sum of weights, shape: scalar
                total_Ow = mpi_comm.allreduce(local_Ow, op=MPI.SUM)  # aggregated observable sums, shape: (num_param,)
                total_Ew = mpi_comm.allreduce(local_Ew, op=MPI.SUM)  # aggregated energy sum, shape: scalar

                # Compute global averages
                O_bar = total_Ow / total_weight  # average observables, shape: (num_param,)
                e_L_bar = total_Ew / total_weight  # average energy, shape: scalar

                # ------------------------------------------------------------------
                # DEBUG: global averages sanity
                # ------------------------------------------------------------------
                _ob_nan = int(np.sum(~np.isfinite(O_bar)))
                _ob_fin = O_bar[np.isfinite(O_bar)]
                logger.devel(
                    "[SR averages] total_weight=%.6e  e_L_bar=%.6f  O_bar: NaN or Inf=%d/%d  min=%.3e  max=%.3e",
                    float(total_weight),
                    float(e_L_bar),
                    _ob_nan,
                    O_bar.size,
                    float(np.min(_ob_fin)) if _ob_fin.size else float("nan"),
                    float(np.max(_ob_fin)) if _ob_fin.size else float("nan"),
                )

                # compute the following variables
                #     X_{i,k} \equiv np.sqrt(w_i) O_{i, k} / np.sqrt({\sum_{i} w_i})
                #     F_i \equiv -2.0 * np.sqrt(w_i) (e_L_{i} - E) / np.sqrt({\sum_{i} w_i})

                X_local = (
                    (O_matrix_local - O_bar) * np.sqrt(w_L_local)[:, np.newaxis] / np.sqrt(total_weight)
                ).T  # shape (num_param, num_mcmc * num_walker) because it's transposed.
                F_local = (
                    -2.0 * np.sqrt(w_L_local) * (e_L_local - e_L_bar) / np.sqrt(total_weight)
                )  # shape (num_mcmc * num_walker, )

                # ------------------------------------------------------------------
                # DEVEL: X_local and F_local health check
                # ------------------------------------------------------------------
                _x_nan = int(np.sum(~np.isfinite(X_local)))
                _f_nan = int(np.sum(~np.isfinite(F_local)))
                _x_fin = X_local[np.isfinite(X_local)]
                _f_fin = F_local[np.isfinite(F_local)]
                logger.devel(
                    "[SR X/F | rank=%d] X_local: shape=%s  NaN or Inf=%d/%d  min=%.3e  max=%.3e  std=%.3e",
                    mpi_rank,
                    X_local.shape,
                    _x_nan,
                    X_local.size,
                    float(np.min(_x_fin)) if _x_fin.size else float("nan"),
                    float(np.max(_x_fin)) if _x_fin.size else float("nan"),
                    float(np.std(_x_fin)) if _x_fin.size else float("nan"),
                )
                logger.devel(
                    "[SR X/F | rank=%d] F_local: shape=%s  NaN or Inf=%d/%d  min=%.3e  max=%.3e  std=%.3e",
                    mpi_rank,
                    F_local.shape,
                    _f_nan,
                    F_local.size,
                    float(np.min(_f_fin)) if _f_fin.size else float("nan"),
                    float(np.max(_f_fin)) if _f_fin.size else float("nan"),
                    float(np.std(_f_fin)) if _f_fin.size else float("nan"),
                )

                logger.devel(f"X_local.shape = {X_local.shape}.")
                logger.devel(f"F_local.shape = {F_local.shape}.")

                # compute X_w@F
                X_F_local = X_local @ F_local  # shape (num_param, )
                X_F = np.empty(X_F_local.shape, dtype=np.float64)
                mpi_comm.Allreduce(X_F_local, X_F, op=MPI.SUM)

                # compute f_argmax (index in reduced space)
                f_argmax = np.argmax(np.abs(X_F))
                logger.devel(
                    f"Max dot(X, F) = {X_F[f_argmax]:.3f} Ha/a.u. should be equal to Max f = {f[f_argmax]:.3f} Ha/a.u."
                )

                # matrix shape info
                num_params = X_local.shape[0]
                num_samples_local = X_local.shape[1]
                num_samples_total = mpi_comm.allreduce(num_samples_local, op=MPI.SUM)

                logger.info("The binning technique is not used to compute the natural gradient.")
                logger.info(f"The number of local samples is {num_samples_local}.")
                logger.info(f"The number of total samples is {num_samples_total}.")
                logger.info(f"SR matrix dimension: {num_params} x {num_params}.")

                # make the SR matrix scale-invariant (i.e., normalize)
                ## compute X_w@X.T
                diag_S_local = np.einsum("jk,kj->j", X_local, X_local.T)
                diag_S = np.empty(diag_S_local.shape, dtype=np.float64)
                mpi_comm.Allreduce(diag_S_local, diag_S, op=MPI.SUM)
                logger.info(f"max. and min. diag_S = {np.max(diag_S)}, {np.min(diag_S)}.")
                # ------------------------------------------------------------------
                # DEVEL: diag_S detail before clamping
                # ------------------------------------------------------------------
                _ds_nan = int(np.sum(~np.isfinite(diag_S)))
                _ds_neg = int(np.sum(diag_S <= 0))
                _ds_tiny = int(np.sum((diag_S > 0) & (diag_S < 1e-30)))
                logger.devel(
                    "[SR diag_S] NaN or Inf=%d  non-positive=%d  tiny(<1e-30)=%d  min=%.3e  max=%.3e  median=%.3e",
                    _ds_nan,
                    _ds_neg,
                    _ds_tiny,
                    float(np.min(diag_S)),
                    float(np.max(diag_S)),
                    float(np.median(diag_S)),
                )
                # Absolute floor for diag_S to prevent 1/sqrt(diag_S)
                # from amplifying near-zero components to catastrophic levels.
                diag_S_floor = min_S_diag_abs
                logger.devel("[SR diag_S] floor = %.3e", diag_S_floor)
                # Freeze parameters whose diag_S is below the absolute threshold.
                # These parameters have near-zero derivative variance, so SR
                # normalization would amplify noise into catastrophically large updates.
                _sr_frozen_mask = ~(np.isfinite(diag_S) & (diag_S > min_S_diag_abs))
                _n_frozen = int(np.count_nonzero(_sr_frozen_mask))
                if _n_frozen > 0:
                    logger.info(
                        "Freezing %d/%d parameters with diag_S < %.1e (converged).",
                        _n_frozen,
                        diag_S.size,
                        min_S_diag_abs,
                    )
                    # Per-block breakdown of frozen parameters
                    for _blk, _s, _e in offsets:
                        _blk_frozen = int(np.count_nonzero(_sr_frozen_mask[_s:_e]))
                        if _blk_frozen > 0:
                            logger.info(
                                "  Frozen in block=%s: %d/%d",
                                _blk.name,
                                _blk_frozen,
                                _blk.size,
                            )
                diag_S = np.where(np.isfinite(diag_S) & (diag_S > diag_S_floor), diag_S, diag_S_floor)
                X_local = X_local / np.sqrt(diag_S)[:, np.newaxis]  # shape (num_param, num_mcmc * num_walker)

                if num_params < num_samples_total:
                    # if True:
                    logger.debug("X is a wide matrix. Proceed w/o the push-through identity.")
                    logger.debug("theta = (S+epsilon*I)^{-1}*f = (X * X^T + epsilon*I)^{-1} * X F...")
                    if not sr_cg_flag:
                        logger.info("Using the direct solver for the inverse of S.")
                        logger.debug(
                            f"Estimated X_local @ X_local.T.bytes per MPI = {X_local.shape[0] ** 2 * X_local.dtype.itemsize / (2**30)} gib."
                        )
                        # compute local sum of X * X^T
                        X_X_T_local = X_local @ X_local.T
                        logger.devel(f"X_X_T_local.shape = {X_X_T_local.shape}.")
                        # compute global sum of X * X^T
                        if mpi_rank == 0:
                            X_X_T = np.empty(X_X_T_local.shape, dtype=np.float64)
                        else:
                            X_X_T = None
                        mpi_comm.Reduce(X_X_T_local, X_X_T, op=MPI.SUM, root=0)
                        # compute local sum of X @ F
                        X_F_local = X_local @ F_local  # shape (num_param, )
                        logger.devel(f"X_F_local.shape = {X_F_local.shape}.")
                        # compute global sum of X @ F
                        if mpi_rank == 0:
                            X_F = np.empty(X_F_local.shape, dtype=np.float64)
                        else:
                            X_F = None
                        mpi_comm.Reduce(X_F_local, X_F, op=MPI.SUM, root=0)
                        # compute theta
                        if mpi_rank == 0:
                            logger.devel(f"X @ X.T.shape = {X_X_T.shape}.")
                            logger.devel(f"X @ F.shape = {X_F.shape}.")
                            # (X X^T + eps*I) x = X F ->solve-> x = (X  X^T + eps*I)^{-1} X F
                            X_X_T[np.diag_indices_from(X_X_T)] += epsilon

                            X_X_T_inv_X_F = scipy.linalg.solve(X_X_T, X_F, assume_a="sym")
                            # theta = (X_w X^T + eps*I)^{-1} X_w F
                            theta_all = X_X_T_inv_X_F
                        else:
                            theta_all = None
                        # Broadcast theta_all to all ranks
                        theta_all = mpi_comm.bcast(theta_all, root=0)
                        logger.devel(f"[new] theta_all (w/o the push through identity) = {theta_all}.")
                        logger.devel(
                            f"[new] theta_all (w/o the push through identity): min, max = {np.min(theta_all)}, {np.max(theta_all)}."
                        )
                    else:
                        logger.info("Using conjugate gradient for the inverse of S.")
                        logger.info(f"  [CG] threshold {sr_cg_tol}.")
                        logger.info(f"  [CG] max iteration: {sr_cg_max_iter}.")
                        # conjugate gradient solver
                        # Compute b = X @ F (distributed)
                        X_F_local = X_local @ F_local  # shape (num_param, )
                        X_F = np.zeros_like(X_F_local)
                        mpi_comm.Allreduce(X_F_local, X_F, op=MPI.SUM)

                        def apply_S_primal_numpy(v):
                            XTv_local = X_local.T @ v
                            XXTv_local = X_local @ XTv_local
                            XXTv_global = np.empty_like(XXTv_local)
                            mpi_comm.Allreduce(XXTv_local, XXTv_global, op=MPI.SUM)
                            return XXTv_global + epsilon * v

                        if sr_cg_warm_start_primal is not None and sr_cg_warm_start_primal.shape == X_F.shape:
                            x0 = sr_cg_warm_start_primal
                        else:
                            x0 = np.zeros_like(X_F)

                        theta_all, final_residual, num_steps = _conjugate_gradient_numpy(
                            np.asarray(X_F, dtype=np.float64),
                            apply_S_primal_numpy,
                            np.asarray(x0, dtype=np.float64),
                            sr_cg_max_iter,
                            sr_cg_tol,
                        )
                        sr_cg_warm_start_primal = np.array(theta_all, copy=True)
                        logger.devel(f"  [CG] Final residual: {final_residual:.3e}")
                        logger.info(f"  [CG] Converged in {num_steps} steps")
                        if num_steps == sr_cg_max_iter:
                            logger.info("  [CG] Conjugate gradient did not converge!!")
                        logger.devel(f"[new/cg] theta_all (w/o the push through identity) = {theta_all}.")
                        logger.devel(
                            f"[new/cg] theta_all (w/o the push through identity): min, max = {np.min(theta_all)}, {np.max(theta_all)}."
                        )

                else:  # num_params >= num_samples:
                    # if True:
                    logger.debug("X is a tall matrix. Proceed w/ the push-through identity.")
                    logger.debug("theta = (S+epsilon*I)^{-1}*f = X(X^T * X + epsilon*I)^{-1} * F...")

                    # Get local shapes
                    N, M = X_local.shape
                    P = mpi_size  # number of ranks

                    # Compute how many rows each rank should own (distribute the remainder)
                    counts = [N // P + (1 if i < (N % P) else 0) for i in range(P)]

                    # Compute starting row index for each rank in the original array
                    displs = [sum(counts[:i]) for i in range(P)]
                    N_local = counts[mpi_rank]  # number of rows this rank will receive

                    # Build send buffers by slicing X and Xw into P row‑chunks
                    # Each chunk is flattened so we can send in one go.
                    sendbuf_X = np.concatenate([X_local[displs[i] : displs[i] + counts[i], :].ravel() for i in range(P)])

                    # Prepare sendcounts and displacements in units of elements
                    sendcounts = [counts[i] * M for i in range(P)]
                    sdispls = [sum(sendcounts[:i]) for i in range(P)]

                    # Prepare recvcounts and displacements:
                    # each rank will receive 'counts[mpi_rank]*M' elements from each of the P ranks
                    recvcounts = [counts[mpi_rank] * M] * P
                    rdispls = [i * counts[mpi_rank] * M for i in range(P)]

                    # Allocate receive buffers
                    recvbuf_X = np.empty(sum(recvcounts), dtype=X_local.dtype)

                    # Perform the all‑to‑all variable‑sized exchange
                    mpi_comm.Alltoallv(
                        [sendbuf_X, sendcounts, sdispls, MPI.DOUBLE], [recvbuf_X, recvcounts, rdispls, MPI.DOUBLE]
                    )

                    # Reshape the flat receive buffer into a 3D array
                    #    shape = (P sources, N_local rows, M cols)
                    buf_X = recvbuf_X.reshape(P, N_local, M)

                    # Rearrange into final 2D arrays of shape (N_local, M * P)
                    #    by stacking each source’s M columns side by side
                    X_re_local = np.hstack([buf_X[i] for i in range(P)])  # shape (num_param/P, num_mcmc * num_walker * P)
                    logger.devel(f"X_re_local.shape = {X_re_local.shape}.")

                    if not sr_cg_flag:
                        logger.info("Using the direct solver for the inverse of S.")
                        logger.devel(
                            f"Estimated X_local.T @ X_local.bytes per MPI = {X_re_local.shape[1] ** 2 * X_re_local.dtype.itemsize / (2**30)} gib."
                        )
                        # compute local sum of X^T * X
                        X_T_X_local = X_re_local.T @ X_re_local
                        logger.devel(f"X_T_X_local.shape = {X_T_X_local.shape}.")
                        # compute global sum of X^T * X
                        if mpi_rank == 0:
                            X_T_X = np.empty(X_T_X_local.shape, dtype=np.float64)
                        else:
                            X_T_X = None
                        mpi_comm.Reduce(X_T_X_local, X_T_X, op=MPI.SUM, root=0)
                        # gather F_local from all ranks (concatenation, not element-wise sum)
                        F_local_count = F_local.shape[0]
                        F_recvcounts = mpi_comm.gather(F_local_count, root=0)
                        if mpi_rank == 0:
                            F_displs = [sum(F_recvcounts[:i]) for i in range(len(F_recvcounts))]
                            F = np.empty(sum(F_recvcounts), dtype=np.float64)
                        else:
                            F_displs = None
                            F = None
                        mpi_comm.Gatherv(
                            [F_local, MPI.DOUBLE],
                            [F, (F_recvcounts, F_displs), MPI.DOUBLE] if mpi_rank == 0 else [F, None],
                            root=0,
                        )
                        if mpi_rank == 0:
                            logger.devel(f"X_T_X.shape = {X_T_X.shape}.")
                            logger.devel(f"F.shape = {F.shape}.")
                            X_T_X[np.diag_indices_from(X_T_X)] += epsilon
                            # (X^T X_w + eps*I) x = F ->solve-> x = (X^T X_w + eps*I)^{-1} F
                            X_T_X_inv_F = scipy.linalg.solve(X_T_X, F, assume_a="sym")
                            K = X_T_X_inv_F.shape[0] // mpi_size
                        else:
                            X_T_X_inv_F = None
                            K = None
                        # Broadcast K to all ranks so they know how big each chunk is
                        K = mpi_comm.bcast(K, root=0)

                        X_T_X_inv_F_local = np.empty(K, dtype=np.float64)

                        mpi_comm.Scatter(
                            [X_T_X_inv_F, MPI.DOUBLE],  # send buffer (only significant on root)
                            X_T_X_inv_F_local,  # receive buffer (on each rank)
                            root=0,
                        )
                        # theta = X_w (X^T X_w + eps*I)^{-1} F
                        theta_all_local = X_local @ X_T_X_inv_F_local
                        theta_all = np.empty(theta_all_local.shape, dtype=np.float64)
                        mpi_comm.Allreduce(theta_all_local, theta_all, op=MPI.SUM)
                        logger.devel(f"[new] theta_all (w/ the push through identity) = {theta_all}.")
                        logger.devel(
                            f"[new] theta_all (w/ the push through identity): min, max = {np.min(theta_all)}, {np.max(theta_all)}."
                        )
                    else:
                        logger.info("Using conjugate gradient for the inverse of S.")
                        logger.info(f"  [CG] threshold {sr_cg_tol}.")
                        logger.info(f"  [CG] max iteration: {sr_cg_max_iter}.")

                        def apply_dual_S_numpy(v):
                            Xv_local = X_re_local @ v
                            XTXv_local = X_re_local.T @ Xv_local
                            XTXv_global = np.empty_like(XTXv_local)
                            mpi_comm.Allreduce(XTXv_local, XTXv_global, op=MPI.SUM)
                            return XTXv_global + epsilon * v

                        # Gather F_local from all ranks (concatenation) to form F_total of length M*P
                        F_local_count = F_local.shape[0]
                        F_recvcounts = mpi_comm.allgather(F_local_count)
                        F_displs = [sum(F_recvcounts[:i]) for i in range(len(F_recvcounts))]
                        F_total = np.empty(sum(F_recvcounts), dtype=np.float64)
                        mpi_comm.Allgatherv(
                            [F_local, MPI.DOUBLE],
                            [F_total, (F_recvcounts, F_displs), MPI.DOUBLE],
                        )
                        if sr_cg_warm_start_dual is not None and sr_cg_warm_start_dual.shape == F_total.shape:
                            x0 = sr_cg_warm_start_dual
                        else:
                            x0 = np.zeros_like(F_total)
                        x_sol, final_residual, num_steps = _conjugate_gradient_numpy(
                            F_total,
                            apply_dual_S_numpy,
                            np.asarray(x0, dtype=np.float64),
                            sr_cg_max_iter,
                            sr_cg_tol,
                        )
                        sr_cg_warm_start_dual = np.array(x_sol, copy=True)

                        # theta = X @ x_sol, evaluated locally over X_re_local (N_local rows)
                        theta_local = X_re_local @ x_sol  # shape (N_local,)
                        theta_local = np.asarray(theta_local)
                        N_local = theta_local.shape[0]

                        recvcounts = mpi_comm.allgather(N_local)
                        displs = [sum(recvcounts[:i]) for i in range(mpi_comm.Get_size())]

                        theta_all = np.empty(sum(recvcounts), dtype=theta_local.dtype)
                        mpi_comm.Allgatherv([theta_local, MPI.DOUBLE], [theta_all, (recvcounts, displs), MPI.DOUBLE])

                        logger.devel(f"  [CG] Final residual: {final_residual:.3e}")
                        logger.info(f"  [CG] Converged in {num_steps} steps")
                        if num_steps == sr_cg_max_iter:
                            logger.logger("  [CG] Conjugate gradient did not converge!")
                        logger.devel(f"[new/cg] theta_all (w/o the push through identity) = {theta_all}.")
                        logger.devel(
                            f"[new/cg] theta_all (w/ the push through identity): min, max = {np.min(theta_all)}, {np.max(theta_all)}."
                        )

                # theta, back to the original scale
                theta_all = theta_all / np.sqrt(diag_S)

                # Zero out theta for parameters frozen due to near-zero diag_S.
                # These parameters have near-zero derivative variance; the
                # scale-invariant de-normalization amplifies noise into
                # catastrophically large updates.
                if _n_frozen > 0:
                    theta_all[_sr_frozen_mask] = 0.0

                # ------------------------------------------------------------------
                # DEVEL: theta_all after scale-back — key NaN diagnosis point
                # ------------------------------------------------------------------
                _t_nan = int(np.sum(~np.isfinite(theta_all)))
                _t_fin = theta_all[np.isfinite(theta_all)]
                _sqrt_ds = np.sqrt(diag_S)
                _sds_nan = int(np.sum(~np.isfinite(_sqrt_ds)))
                _sds_zero = int(np.sum(_sqrt_ds == 0.0))
                logger.devel(
                    "[SR theta_all] After /sqrt(diag_S): NaN or Inf=%d/%d  min=%.3e  max=%.3e  norm=%.3e",
                    _t_nan,
                    theta_all.size,
                    float(np.min(_t_fin)) if _t_fin.size else float("nan"),
                    float(np.max(_t_fin)) if _t_fin.size else float("nan"),
                    float(np.linalg.norm(_t_fin)) if _t_fin.size else float("nan"),
                )
                logger.devel(
                    "[SR theta_all] sqrt(diag_S): NaN or Inf=%d  exact-zero=%d  min=%.3e  max=%.3e",
                    _sds_nan,
                    _sds_zero,
                    float(np.min(_sqrt_ds)),
                    float(np.max(_sqrt_ds)),
                )

                # ------------------------------------------------------------------
                # DEBUG: per-parameter trace for top SN parameters
                # ------------------------------------------------------------------
                _trace_n = 5
                _trace_idx = np.argsort(signal_to_noise_f)[::-1][:_trace_n]
                for _ti in _trace_idx:
                    logger.debug(
                        "  [SR trace] flat=%d  f=%.6e  diag_S=%.6e  theta_SR=%.6e  SN=%.3f",
                        int(_ti),
                        float(f[_ti]),
                        float(diag_S[_ti]),
                        float(theta_all[_ti]),
                        float(signal_to_noise_f[_ti]),
                    )

                # Pre-compute collective variable observables for LM while
                # O_matrix_local is still in memory (avoids reloading full
                # O_matrix in get_aH).
                _lm_collective_obs = None
                if use_lm and lm_subspace_dim != 0:
                    dO_local = O_matrix_local - O_bar[np.newaxis, :]  # (N_local, K_full)
                    _lm_collective_obs = dO_local @ theta_all  # (N_local,) = Ō_SR per sample
                    del dO_local  # free immediately

                # Free SR-solve temporaries that are no longer needed
                del O_matrix_local

            #############################
            # optax optimizer
            #############################
            else:
                params = jnp.array(flat_param_vector)
                grads = -jnp.array(f)
                updates, optax_state = optax_tx.update(grads, optax_state, params)
                theta_all = np.array(updates, dtype=np.float64)
                if optax_param_size is None:
                    optax_param_size = flat_param_vector.size
                self.__set_optimizer_runtime(
                    method=optimizer_mode,
                    hyperparameters=optimizer_hparams,
                    optax_state=optax_state,
                    optax_param_size=optax_param_size,
                )

            end = time.perf_counter()
            timer_optimizer += end - start

            # ------------------------------------------------------------------
            # 1) Expand theta_all to full parameter space.
            # ------------------------------------------------------------------
            if use_sr:
                theta = np.zeros(total_num_params, dtype=np.float64)
                theta[:] = theta_all
            else:
                # optax
                theta = np.zeros(total_num_params, dtype=np.float64)
                theta[:] = theta_all

            # ------------------------------------------------------------------
            # DEVEL: per-block f and theta comparison (in full space, after expand)
            #   This enables side-by-side comparison of projected vs unprojected runs.
            # ------------------------------------------------------------------
            if use_sr:
                _opt_label = "LM" if (use_lm and lm_subspace_dim != 0) else "aSR" if use_lm else "SR"
                for _blk, _s, _e in offsets:
                    _f_blk = f[_s:_e]
                    _t_blk = theta[_s:_e]
                    _f_fin = _f_blk[np.isfinite(_f_blk)]
                    _t_fin = _t_blk[np.isfinite(_t_blk)]
                    logger.devel(
                        "[%s per-block] block=%-16s size=%5d  "
                        "f: min=%+.3e max=%+.3e norm=%.3e  "
                        "theta: min=%+.3e max=%+.3e norm=%.3e",
                        _opt_label,
                        _blk.name,
                        _blk.size,
                        float(np.min(_f_fin)) if _f_fin.size else float("nan"),
                        float(np.max(_f_fin)) if _f_fin.size else float("nan"),
                        float(np.linalg.norm(_f_fin)) if _f_fin.size else float("nan"),
                        float(np.min(_t_fin)) if _t_fin.size else float("nan"),
                        float(np.max(_t_fin)) if _t_fin.size else float("nan"),
                        float(np.linalg.norm(_t_fin)) if _t_fin.size else float("nan"),
                    )

            # ------------------------------------------------------------------
            # LM / aSR step.  Must happen BEFORE vir-vir re-projection so that
            # the re-projection is applied to the final theta (including LM/aSR
            # modifications).
            # ------------------------------------------------------------------
            if use_lm and lm_subspace_dim != 0:
                # ---- LM with SR collective variable ----
                # g = theta (SR natural gradient) is the collective variable
                g_sr = theta.copy()

                # Subspace selection for individual parameters
                if lm_subspace_dim == -1 or lm_subspace_dim >= total_num_params:
                    subspace_indices = np.arange(total_num_params, dtype=np.intp)
                    logger.info(f"  LM: subspace = SR collective + all {total_num_params} parameters")
                else:
                    _sn_sorted = np.argsort(signal_to_noise_f)[::-1]
                    # Include all parameters tied at the boundary S/N value
                    _sn_cutoff = signal_to_noise_f[_sn_sorted[lm_subspace_dim - 1]]
                    _n_selected = lm_subspace_dim
                    while _n_selected < len(_sn_sorted) and signal_to_noise_f[_sn_sorted[_n_selected]] >= _sn_cutoff:
                        _n_selected += 1
                    subspace_indices = _sn_sorted[:_n_selected]
                    if _n_selected > lm_subspace_dim:
                        logger.info(
                            f"  LM: subspace = SR collective + {_n_selected} parameters "
                            f"(requested {lm_subspace_dim}, expanded to include tied SN={_sn_cutoff:.3f})"
                        )
                    else:
                        logger.info(f"  LM: subspace = SR collective + top {lm_subspace_dim} parameters (by SN ratio)")

                # Build LM matrices with collective variable
                # _lm_collective_obs was pre-computed during SR solve (memory-efficient)
                H_0_lm, f_vec_lm, S_mat, K_mat, B_mat = self.get_aH(
                    blocks=blocks,
                    g=g_sr,
                    num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                    chosen_param_index=subspace_indices,
                    lambda_projectors=lambda_projectors,
                    num_orb_projection=num_orb_projection,
                    return_matrices=True,
                    collective_obs=_lm_collective_obs,
                )

                # Solve LM eigenvalue problem
                c_vec, E_lm = self.solve_linear_method(H_0_lm, f_vec_lm, S_mat, K_mat, B_mat, epsilon=lm_cond)

                # Back-transform: c_vec[0] = c₀ (SR direction), c_vec[1:] = c_k (individual params)
                theta = np.zeros(total_num_params, dtype=np.float64)
                theta[:] += c_vec[0] * g_sr  # SR collective variable (affects all params)
                if lm_subspace_dim == -1 or lm_subspace_dim >= total_num_params:
                    theta[:] += c_vec[1:]
                else:
                    theta[subspace_indices] += c_vec[1:]

                # DEBUG: trace theta after LM replacement
                for _ti in _trace_idx:
                    _c0_g = c_vec[0] * g_sr[_ti]
                    _c_ind = 0.0
                    if _ti in subspace_indices:
                        _sub_pos = int(np.searchsorted(subspace_indices, _ti))
                        if _sub_pos < len(subspace_indices) and subspace_indices[_sub_pos] == _ti:
                            _c_ind = c_vec[1 + _sub_pos]
                    logger.debug(
                        "  [LM trace] flat=%d  theta_LM=%.6e  (c0*g=%.6e  c_ind=%.6e)  c0=%.6e",
                        int(_ti),
                        float(theta[_ti]),
                        float(_c0_g),
                        float(_c_ind),
                        float(c_vec[0]),
                    )

            elif use_lm:
                # ---- aSR (lm_subspace_dim = 0) ----
                if not np.any(theta):
                    logger.info("aSR: theta is all zeros (all parameters frozen); skipping gamma scaling.")
                else:
                    logger.info("aSR: computing optimal gamma via accelerated SR.")
                    H_0, H_1, H_2, S_2 = self.get_aH(
                        blocks=blocks,
                        g=theta,
                        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                        lambda_projectors=lambda_projectors,
                        num_orb_projection=num_orb_projection,
                    )
                    gamma = self.compute_asr_gamma(H_0, H_1, H_2, S_2)
                    logger.info(f"aSR: scaling theta by gamma = {gamma:.6f}.")
                    theta = theta * gamma

            # ------------------------------------------------------------------
            # Re-project theta in orthogonal basis to remove vir-vir noise.
            # Applied after LM/aSR so that the final theta is cleaned.
            # ------------------------------------------------------------------
            if use_sr and lambda_projectors is not None and len(lambda_projectors) == 4:
                _left_proj, _right_proj, _, _ = lambda_projectors
                _identity_proj = np.eye(_left_proj.shape[0], dtype=np.float64)
                _comp_L = _identity_proj - _left_proj
                _comp_R = _identity_proj - _right_proj
                for _blk, _s, _e in offsets:
                    if _blk.name == "lambda_matrix":
                        _theta_mat = theta[_s:_e].reshape(_blk.shape)
                        _n_paired = _right_proj.shape[0]
                        _theta_paired = _theta_mat[:, :_n_paired]
                        _vv_correction = _comp_L @ _theta_paired @ _comp_R
                        _vv_norm_before = float(np.linalg.norm(_theta_paired))
                        _theta_mat[:, :_n_paired] = _theta_paired - _vv_correction
                        _vv_norm_removed = float(np.linalg.norm(_vv_correction))
                        theta[_s:_e] = _theta_mat.ravel()
                        logger.devel(
                            "[SR theta] Re-projected theta in orth basis: "
                            "||theta_paired||=%.3e  ||vv_removed||=%.3e  ratio=%.3e",
                            _vv_norm_before,
                            _vv_norm_removed,
                            _vv_norm_removed / max(_vv_norm_before, 1e-300),
                        )
                        break

            # DEBUG: trace theta after vir-vir re-projection and before back-transform
            if use_sr and use_lm and lm_subspace_dim != 0:
                _theta_before_bt = theta.copy()
                for _ti in _trace_idx:
                    logger.debug(
                        "  [post-virvir trace] flat=%d  theta=%.6e",
                        int(_ti),
                        float(theta[_ti]),
                    )

            # ------------------------------------------------------------------
            # Compute ||Delta ln|Psi||| = delta * sqrt(theta^T S theta)
            # ------------------------------------------------------------------
            if use_sr:
                # Using X_local (normalized): theta^T S theta = ||X_local^T (sqrt(diag_S) * theta)||^2
                _v = np.sqrt(diag_S) * theta
                _Xt_v_local = X_local.T @ _v
                _norm_sq_local = np.dot(_Xt_v_local, _Xt_v_local)
                _norm_sq = mpi_comm.allreduce(_norm_sq_local, op=MPI.SUM)
                _delta_ln_psi_norm = delta * np.sqrt(_norm_sq)
                logger.debug(f"  Change in ln|Psi| norm = {_delta_ln_psi_norm}")

            # ------------------------------------------------------------------
            # 2) Back-transform theta from orthogonal basis to AO basis
            #    for the lambda_matrix block.
            #      paired:   θ_AO = S^{-1/2}_up @ θ'_orth @ S^{-1/2}_dn
            #      unpaired: θ_AO = S^{-1/2}_up @ θ'_orth
            # ------------------------------------------------------------------
            if lambda_projectors is not None and len(lambda_projectors) == 4:
                _, _, _inv_sqrt_up, _inv_sqrt_dn = lambda_projectors
                for _blk, _s, _e in offsets:
                    if _blk.name == "lambda_matrix":
                        _theta_mat = theta[_s:_e].reshape(_blk.shape)
                        _n_paired = _inv_sqrt_dn.shape[0]
                        _theta_paired = _theta_mat[:, :_n_paired]
                        _theta_unpaired = _theta_mat[:, _n_paired:]
                        # paired: two-sided back-transform
                        _theta_paired_ao = _inv_sqrt_up @ _theta_paired @ _inv_sqrt_dn
                        # unpaired: left-only back-transform
                        _theta_unpaired_ao = _inv_sqrt_up @ _theta_unpaired
                        _theta_ao = np.hstack((_theta_paired_ao, _theta_unpaired_ao))
                        theta[_s:_e] = _theta_ao.ravel()
                        logger.devel(
                            "Back-transformed lambda block from orthogonal to AO basis. "
                            "paired: min=%.3e max=%.3e  unpaired: min=%.3e max=%.3e",
                            float(np.min(_theta_paired_ao)),
                            float(np.max(_theta_paired_ao)),
                            float(np.min(_theta_unpaired_ao)) if _theta_unpaired_ao.size else 0.0,
                            float(np.max(_theta_unpaired_ao)) if _theta_unpaired_ao.size else 0.0,
                        )
                        break

            # DEBUG: trace theta after back-transform (final update direction)
            if use_sr and use_lm and lm_subspace_dim != 0:
                for _ti in _trace_idx:
                    _bt_change = float(theta[_ti] - _theta_before_bt[_ti]) if "_theta_before_bt" in dir() else 0.0
                    logger.debug(
                        "  [final trace] flat=%d  theta_final=%.6e  delta*theta=%.6e  bt_change=%.6e",
                        int(_ti),
                        float(theta[_ti]),
                        float(delta * theta[_ti]),
                        float(_bt_change),
                    )

            logger.info(f"theta.size = {theta.size}.")
            logger.info(f"np.count_nonzero(theta) = {np.count_nonzero(theta)}.")
            logger.info(f"max. and min. of theta are {np.max(theta)} and {np.min(theta)}.")

            # Common log helpers
            _log_delta = delta if use_sr else 1.0
            _log_label = (
                ("LM" if (use_lm and lm_subspace_dim != 0) else "aSR" if use_lm else "SR") if use_sr else optimizer_mode
            )

            # Guard against NaN/Inf components before applying the update and
            # report which blocks contain problematic entries.
            non_finite_mask = ~np.isfinite(theta)
            if np.any(non_finite_mask):
                bad_blocks: list[str] = []
                for block, start, end in offsets:
                    block_mask = non_finite_mask[start:end]
                    if np.any(block_mask):
                        count_bad = int(np.count_nonzero(block_mask))
                        bad_blocks.append(f"{block.name}({count_bad}/{block.size})")
                logger.error(
                    "Detected non-finite entries in %s update vector; zeroing them. Blocks=%s",
                    _log_label,
                    ", ".join(bad_blocks) if bad_blocks else "unknown",
                )
                theta = np.where(np.isfinite(theta), theta, 0.0)

            # Emit per-block statistics so we can confirm that parameters are
            # actually updated (or detect suspiciously large steps).
            for block, start, end in offsets:
                block_theta = theta[start:end]
                if not np.any(block_theta):
                    logger.info(
                        "  [%s update] – block=%s size=%d  theta=ALL ZERO (no update)",
                        _log_label,
                        block.name,
                        block.size,
                    )
                    continue
                block_norm = float(np.linalg.norm(block_theta))
                block_max = float(np.max(np.abs(block_theta)))
                block_delta_max = float(_log_delta * block_max)
                logger.info(
                    "  [%s update] – block=%s size=%d  ||theta||=%.3e  max|theta|=%.3e  max|delta*theta|=%.3e",
                    _log_label,
                    block.name,
                    block.size,
                    block_norm,
                    block_max,
                    block_delta_max,
                )

            start = time.perf_counter()
            logger.info(f"Updating parameters with optimizer '{optimizer_mode}'.")
            logger.devel(f"dX.shape for MPI-rank={mpi_rank} is {theta.shape}")

            structure_data = self.hamiltonian_data.structure_data
            coulomb_potential_data = self.hamiltonian_data.coulomb_potential_data

            wavefunction_data_old = self.hamiltonian_data.wavefunction_data
            block_learning_rate = delta if use_sr else 1.0
            wavefunction_data = wavefunction_data_old.apply_block_updates(
                blocks=blocks,
                thetas=theta,
                learning_rate=block_learning_rate,
            )

            if opt_with_projected_MOs:
                geminal_ao_current = wavefunction_data.geminal_data

                geminal_mo_rescaled = Geminal_data.convert_from_AOs_to_MOs(
                    geminal_data=geminal_ao_current,
                    num_eigenvectors=geminal_ao_current.num_electron_up,
                )
                wavefunction_data = type(wavefunction_data)(
                    jastrow_data=wavefunction_data.jastrow_data,
                    geminal_data=geminal_mo_rescaled,
                )
                logger.info("opt_with_projected_MOs=True: projected updated AO geminal to MO (truncated, scaled).")

            hamiltonian_data = Hamiltonian_data(
                structure_data=structure_data,
                wavefunction_data=wavefunction_data,
                coulomb_potential_data=coulomb_potential_data,
            )
            logger.info("Wavefunction has been updated. Optimization loop is done.")
            logger.info("")
            self.hamiltonian_data = hamiltonian_data
            end = time.perf_counter()
            timer_param_update += end - start

            # MPI barrier after all optimization operation (timed)
            start = time.perf_counter()
            mpi_comm.Barrier()
            end = time.perf_counter()
            timer_MPI_barrier += end - start

            num_opt_done += 1

            # check max time
            vmcopt_current = time.perf_counter()

            # if max_time exceeded, break the loop
            if max_time < vmcopt_current - vmcopt_total_start:
                logger.info(f"Stopping... max_time = {max_time} sec. exceeds.")
                # dump WF
                if mpi_rank == 0:
                    hamiltonian_data_filename = f"hamiltonian_data_opt_step_{i_opt + 1 + self.__i_opt}.h5"
                    logger.info(f"Hamiltonian data is dumped as an HDF5 file: {hamiltonian_data_filename}.")
                    hamiltonian_data_to_dump = self.hamiltonian_data
                    if opt_with_projected_MOs and initial_geminal_is_ao_representation:
                        wf_data = hamiltonian_data_to_dump.wavefunction_data
                        geminal = wf_data.geminal_data
                        geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal)
                        wf_data_ao = type(wf_data)(
                            jastrow_data=wf_data.jastrow_data,
                            geminal_data=geminal_ao,
                        )
                        hamiltonian_data_to_dump = Hamiltonian_data(
                            structure_data=hamiltonian_data_to_dump.structure_data,
                            wavefunction_data=wf_data_ao,
                            coulomb_potential_data=hamiltonian_data_to_dump.coulomb_potential_data,
                        )
                    hamiltonian_data_to_dump.save_to_hdf5(hamiltonian_data_filename)
                logger.info("Break the vmcopt loop.")
                break

            # MPI barrier before checking toml file
            mpi_comm.Barrier()

            # check toml file (stop flag)
            if os.path.isfile(toml_filename):
                dict_toml = toml.load(open(toml_filename))
                try:
                    stop_flag = dict_toml["external_control"]["stop"]
                except KeyError:
                    stop_flag = False
                if stop_flag:
                    logger.info(f"Stopping... stop_flag in {toml_filename} is true.")
                    # dump WF
                    if mpi_rank == 0:
                        hamiltonian_data_filename = f"hamiltonian_data_opt_step_{i_opt + 1 + self.__i_opt}.h5"
                        logger.info(f"Hamiltonian data is dumped as an HDF5 file: {hamiltonian_data_filename}.")
                        hamiltonian_data_to_dump = self.hamiltonian_data
                        if opt_with_projected_MOs and initial_geminal_is_ao_representation:
                            wf_data = hamiltonian_data_to_dump.wavefunction_data
                            geminal = wf_data.geminal_data
                            geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal)
                            wf_data_ao = type(wf_data)(
                                jastrow_data=wf_data.jastrow_data,
                                geminal_data=geminal_ao,
                            )
                            hamiltonian_data_to_dump = Hamiltonian_data(
                                structure_data=hamiltonian_data_to_dump.structure_data,
                                wavefunction_data=wf_data_ao,
                                coulomb_potential_data=hamiltonian_data_to_dump.coulomb_potential_data,
                            )
                        hamiltonian_data_to_dump.save_to_hdf5(hamiltonian_data_filename)
                    logger.info("Break the optimization loop.")
                    logger.info("")
                    break

            # dump WF
            if mpi_rank == 0:
                if (i_opt + 1) % wf_dump_freq == 0 or (i_opt + 1) == num_opt_steps:
                    hamiltonian_data_filename = f"hamiltonian_data_opt_step_{i_opt + 1 + self.__i_opt}.h5"
                    logger.info(f"Hamiltonian data is dumped as an HDF5 file: {hamiltonian_data_filename}.")
                    logger.info("")
                    hamiltonian_data_to_dump = self.hamiltonian_data
                    if opt_with_projected_MOs and initial_geminal_is_ao_representation:
                        wf_data = hamiltonian_data_to_dump.wavefunction_data
                        geminal = wf_data.geminal_data
                        geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal)
                        wf_data_ao = type(wf_data)(
                            jastrow_data=wf_data.jastrow_data,
                            geminal_data=geminal_ao,
                        )
                        hamiltonian_data_to_dump = Hamiltonian_data(
                            structure_data=hamiltonian_data_to_dump.structure_data,
                            wavefunction_data=wf_data_ao,
                            coulomb_potential_data=hamiltonian_data_to_dump.coulomb_potential_data,
                        )
                    hamiltonian_data_to_dump.save_to_hdf5(hamiltonian_data_filename)

        if opt_with_projected_MOs and self.hamiltonian_data.wavefunction_data.geminal_data.is_ao_representation:
            wf_data = self.hamiltonian_data.wavefunction_data
            geminal = wf_data.geminal_data
            geminal_mo = Geminal_data.convert_from_AOs_to_MOs(
                geminal_data=geminal,
                num_eigenvectors=geminal.num_electron_up,
            )
            wf_data_mo = type(wf_data)(
                jastrow_data=wf_data.jastrow_data,
                geminal_data=geminal_mo,
            )
            self.hamiltonian_data = Hamiltonian_data(
                structure_data=self.hamiltonian_data.structure_data,
                wavefunction_data=wf_data_mo,
                coulomb_potential_data=self.hamiltonian_data.coulomb_potential_data,
            )
            logger.info("opt_with_projected_MOs=True: projected final AO geminal back to MO representation.")
            logger.info("")

        # update WF opt counter
        self.__i_opt += i_opt + 1

        vmcopt_total_end = time.perf_counter()
        timer_opt_total += vmcopt_total_end - vmcopt_total_start
        timer_misc = timer_opt_total - (
            timer_mcmc_run + timer_get_E + timer_get_gF + timer_optimizer + timer_param_update + timer_MPI_barrier
        )

        # average among MPI processes
        ave_timer_opt_total = mpi_comm.allreduce(timer_opt_total, op=MPI.SUM) / mpi_size
        if num_opt_done > 0:
            ave_timer_mcmc_run = mpi_comm.allreduce(timer_mcmc_run, op=MPI.SUM) / mpi_size / num_opt_done
            ave_timer_get_E = mpi_comm.allreduce(timer_get_E, op=MPI.SUM) / mpi_size / num_opt_done
            ave_timer_get_gF = mpi_comm.allreduce(timer_get_gF, op=MPI.SUM) / mpi_size / num_opt_done
            ave_timer_optimizer = mpi_comm.allreduce(timer_optimizer, op=MPI.SUM) / mpi_size / num_opt_done
            ave_timer_param_update = mpi_comm.allreduce(timer_param_update, op=MPI.SUM) / mpi_size / num_opt_done
            ave_timer_MPI_barrier = mpi_comm.allreduce(timer_MPI_barrier, op=MPI.SUM) / mpi_size / num_opt_done
            ave_timer_misc = mpi_comm.allreduce(timer_misc, op=MPI.SUM) / mpi_size / num_opt_done
        else:
            ave_timer_mcmc_run = 0.0
            ave_timer_get_E = 0.0
            ave_timer_get_gF = 0.0
            ave_timer_optimizer = 0.0
            ave_timer_param_update = 0.0
            ave_timer_MPI_barrier = 0.0
            ave_timer_misc = 0.0

        logger.info(f"Total elapsed time for optimization {num_opt_done} steps. = {ave_timer_opt_total:.2f} sec.")
        logger.info(f"Elapsed times per optimization step, averaged over {num_opt_done} steps.")
        logger.info(f"  Time for MCMC run = {ave_timer_mcmc_run:.2f} sec.")
        logger.info(f"  Time for computing E = {ave_timer_get_E:.2f} sec.")
        logger.info(f"  Time for computing generalized forces (gF) = {ave_timer_get_gF:.2f} sec.")
        logger.info(f"  Time for optimizer (SR/optax) = {ave_timer_optimizer:.2f} sec.")
        logger.info(f"  Time for parameter update = {ave_timer_param_update:.2f} sec.")
        logger.info(f"  Time for MPI barrier = {ave_timer_MPI_barrier:.2f} sec.")
        logger.info(f"  Time for misc. (others) = {ave_timer_misc:.2f} sec.")
        logger.info("")

        # remove the toml file
        mpi_comm.Barrier()
        if mpi_rank == 0:
            if os.path.isfile(toml_filename):
                logger.info(f"Delete {toml_filename}")
                os.remove(toml_filename)

    # ------------------------------------------------------------------ #
    #  HDF5 checkpoint save / load
    # ------------------------------------------------------------------ #

    def save_to_hdf5(self, filepath: str) -> None:
        """Write this rank's state to a temporary per-rank HDF5 file.

        The file is later merged by :func:`jqmc.checkpoint.merge_rank_checkpoints`.

        Args:
            filepath: Output path (e.g. ``._restart_rank0.h5``).
        """
        from ._checkpoint import save_rank_checkpoint

        # -- driver_config (scalar execution parameters) --
        driver_config: dict[str, Any] = {
            "mcmc_seed": self.__mcmc_seed,
            "num_walkers": self.__num_walkers,
            "num_mcmc_per_measurement": self.__num_mcmc_per_measurement,
            "Dt": self.__Dt,
            "epsilon_AS": self.__epsilon_AS,
            "comput_log_WF_param_deriv": self.__comput_log_WF_param_deriv,
            "comput_e_L_param_deriv": self.__comput_e_L_param_deriv,
            "comput_position_deriv": self.__comput_position_deriv,
            "random_discretized_mesh": self.__random_discretized_mesh,
            "mcmc_counter": self.__mcmc_counter,
            "accepted_moves": self.__accepted_moves,
            "rejected_moves": self.__rejected_moves,
            "i_opt": self.__i_opt,
        }

        # -- param_grad_flags (per-block gradient on/off) --
        for k, v in self.__param_grad_flags.items():
            driver_config[f"param_grad_flag_{k}"] = v

        # -- rng_state --
        rng_state: dict[str, Any] = {
            "jax_PRNG_key_list": np.asarray(self.__jax_PRNG_key_list),
            "mpi_seed": self.__mpi_seed,
        }

        # -- walker_state --
        walker_state: dict[str, Any] = {
            "latest_r_up_carts": np.asarray(self.__latest_r_up_carts),
            "latest_r_dn_carts": np.asarray(self.__latest_r_dn_carts),
        }

        # -- observables --
        observables: dict[str, Any] = {}

        _obs_map = {
            "e_L": self.__stored_e_L,
            "e_L2": self.__stored_e_L2,
            "w_L": self.__stored_w_L,
            "force_HF": self.__stored_force_HF,
            "force_PP": self.__stored_force_PP,
            "E_L_force_PP": self.__stored_E_L_force_PP,
        }
        for key, val in _obs_map.items():
            arr = np.asarray(val) if val.size > 0 else np.empty(0)
            observables[key] = arr

        # dict-keyed parameter gradients
        param_grads: dict[str, np.ndarray] = {}
        for name, val_list in self.__stored_log_WF_param_grads.items():
            if len(val_list) > 0:
                param_grads[name] = np.asarray(val_list)
        observables["param_grads"] = param_grads

        e_L_param_grads: dict[str, np.ndarray] = {}
        for name, val_list in self.__stored_e_L_param_grads.items():
            if len(val_list) > 0:
                e_L_param_grads[name] = np.asarray(val_list)
        observables["e_L_param_grads"] = e_L_param_grads

        # -- optimizer_state --
        self.__ensure_optimizer_runtime()
        optimizer_state = self.__optimizer_runtime

        save_rank_checkpoint(
            filepath,
            driver_type="MCMC",
            driver_config=driver_config,
            rng_state=rng_state,
            walker_state=walker_state,
            observables=observables,
            optimizer_state=optimizer_state,
        )

    @classmethod
    def load_from_hdf5(cls, filepath: str, rank: int | None = None) -> "MCMC":
        """Restore an MCMC instance from a merged HDF5 checkpoint.

        This bypasses ``__init__`` entirely and reconstructs all internal
        attributes from the stored data and the embedded ``hamiltonian_data``.

        Args:
            filepath: Path to the merged checkpoint (e.g. ``restart.h5``).
            rank: MPI rank to load.  Defaults to the current MPI rank.

        Returns:
            A fully restored :class:`MCMC` instance.
        """
        from ._checkpoint import load_hamiltonian_from_checkpoint, load_rank_checkpoint

        if rank is None:
            rank = mpi_rank

        data = load_rank_checkpoint(filepath, rank=rank)
        cfg = data["driver_config"]
        rng = data["rng_state"]
        ws = data["walker_state"]
        obs = data["observables"]
        opt = data["optimizer_state"]

        # Load Hamiltonian_data from the checkpoint root
        hamiltonian_data = load_hamiltonian_from_checkpoint(filepath)

        # Create an empty instance without calling __init__
        obj = cls.__new__(cls)

        # -- Scalar config --
        obj._MCMC__mcmc_seed = cfg["mcmc_seed"]
        obj._MCMC__num_walkers = cfg["num_walkers"]
        obj._MCMC__num_mcmc_per_measurement = cfg["num_mcmc_per_measurement"]
        obj._MCMC__Dt = cfg["Dt"]
        obj._MCMC__epsilon_AS = cfg["epsilon_AS"]
        obj._MCMC__comput_log_WF_param_deriv = bool(cfg.get("comput_log_WF_param_deriv", False))
        obj._MCMC__comput_e_L_param_deriv = bool(cfg.get("comput_e_L_param_deriv", False))
        obj._MCMC__comput_position_deriv = bool(cfg.get("comput_position_deriv", False))
        obj._MCMC__random_discretized_mesh = bool(cfg.get("random_discretized_mesh", True))

        # Counters
        obj._MCMC__mcmc_counter = cfg.get("mcmc_counter", 0)
        obj._MCMC__accepted_moves = cfg.get("accepted_moves", 0)
        obj._MCMC__rejected_moves = cfg.get("rejected_moves", 0)
        obj._MCMC__i_opt = cfg.get("i_opt", 0)

        # -- param_grad_flags --
        flags = cls._MCMC__default_param_grad_flags()
        for k in flags:
            stored_key = f"param_grad_flag_{k}"
            if stored_key in cfg:
                flags[k] = bool(cfg[stored_key])
        obj._MCMC__param_grad_flags = flags

        # -- Hamiltonian data (apply DiffMask as the normal setter does) --
        obj._MCMC__hamiltonian_data = hamiltonian_data
        obj.hamiltonian_data = hamiltonian_data  # triggers setter → DiffMask + __init_attributes

        # -- Overwrite __init_attributes results with loaded state --
        obj._MCMC__mcmc_counter = cfg.get("mcmc_counter", 0)
        obj._MCMC__accepted_moves = cfg.get("accepted_moves", 0)
        obj._MCMC__rejected_moves = cfg.get("rejected_moves", 0)

        # -- RNG state --
        obj._MCMC__mpi_seed = rng["mpi_seed"]
        obj._MCMC__jax_PRNG_key = jax.random.PRNGKey(rng["mpi_seed"])
        obj._MCMC__jax_PRNG_key_list = jnp.array(rng["jax_PRNG_key_list"])

        # -- Walker state --
        obj._MCMC__latest_r_up_carts = jnp.array(ws["latest_r_up_carts"])
        obj._MCMC__latest_r_dn_carts = jnp.array(ws["latest_r_dn_carts"])

        # -- only_up_electron (derived from hamiltonian) --
        tot_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        obj._MCMC__only_up_electron = tot_dn == 0

        # -- Observables (keep __init_attributes defaults for missing/empty data) --
        def _load_obs(obs_arr, default):
            """Return loaded ndarray if non-empty, otherwise keep the shaped default."""
            if obs_arr is None or (isinstance(obs_arr, np.ndarray) and obs_arr.size == 0):
                return default
            return np.asarray(obs_arr)

        def _to_list(arr):
            """Convert ndarray back to list-of-arrays (for dict-keyed param grads)."""
            if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
                return []
            return [arr[i] for i in range(arr.shape[0])]

        obj._MCMC__stored_e_L = _load_obs(obs.get("e_L"), obj._MCMC__stored_e_L)
        obj._MCMC__stored_e_L2 = _load_obs(obs.get("e_L2"), obj._MCMC__stored_e_L2)
        obj._MCMC__stored_w_L = _load_obs(obs.get("w_L"), obj._MCMC__stored_w_L)
        obj._MCMC__stored_force_HF = _load_obs(obs.get("force_HF"), obj._MCMC__stored_force_HF)
        obj._MCMC__stored_force_PP = _load_obs(obs.get("force_PP"), obj._MCMC__stored_force_PP)
        obj._MCMC__stored_E_L_force_PP = _load_obs(obs.get("E_L_force_PP"), obj._MCMC__stored_E_L_force_PP)

        # dict-keyed parameter gradients
        pg = obs.get("param_grads", {})
        stored_pg: dict[str, list] = defaultdict(list)
        if isinstance(pg, dict):
            for name, arr in pg.items():
                stored_pg[name] = _to_list(arr)
        obj._MCMC__stored_log_WF_param_grads = stored_pg

        epg = obs.get("e_L_param_grads", {})
        stored_epg: dict[str, list] = defaultdict(list)
        if isinstance(epg, dict):
            for name, arr in epg.items():
                stored_epg[name] = _to_list(arr)
        obj._MCMC__stored_e_L_param_grads = stored_epg

        # -- Optimizer runtime --
        if opt is not None:
            obj._MCMC__optimizer_runtime = opt
        else:
            obj._MCMC__optimizer_runtime = None
            obj._MCMC__ensure_optimizer_runtime()

        return obj

    # hamiltonian
    @property
    def hamiltonian_data(self):
        """Access the mutable :class:`Hamiltonian_data` backing this sampler."""
        return self.__hamiltonian_data

    @hamiltonian_data.setter
    def hamiltonian_data(self, hamiltonian_data):
        """Set :class:`Hamiltonian_data`, applying masks for gradient tracking.

        When only parameter or position derivatives are requested, the incoming Hamiltonian
        is wrapped with ``DiffMask`` to avoid tracing disabled parts. Setting this property
        reinitializes internal storage buffers.
        """
        if self.__comput_log_WF_param_deriv and not self.__comput_position_deriv:
            self.__hamiltonian_data = apply_diff_mask(hamiltonian_data, DiffMask(params=True, coords=False))
        elif not self.__comput_log_WF_param_deriv and self.__comput_position_deriv:
            # self.__hamiltonian_data = Hamiltonian_data.from_base(hamiltonian_data)
            self.__hamiltonian_data = apply_diff_mask(hamiltonian_data, DiffMask(params=False, coords=True))
        elif not self.__comput_log_WF_param_deriv and not self.__comput_position_deriv:
            self.__hamiltonian_data = apply_diff_mask(hamiltonian_data, DiffMask(params=False, coords=False))
        else:
            self.__hamiltonian_data = hamiltonian_data
        self.__init_attributes()

    # dimensions of observables
    @property
    def mcmc_counter(self) -> int:
        """Number of Metropolis steps accumulated (rows in stored observables)."""
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
        return np.asarray(self.__stored_w_L)

    # observables
    @property
    def e_L(self) -> npt.NDArray:
        """Return the stored e_L array. dim: (mcmc_counter, num_walkers)."""
        return np.asarray(self.__stored_e_L)

    # observables
    @property
    def e_L2(self) -> npt.NDArray:
        """Return the stored e_L^2 array. dim: (mcmc_counter, num_walkers)."""
        return np.asarray(self.__stored_e_L2)

    @property
    def force_HF_stored(self) -> npt.NDArray:
        """Return the stored force_HF array. dim: (mcmc_counter, num_walkers, num_atoms, 3)."""
        return np.asarray(self.__stored_force_HF)

    @property
    def force_PP_stored(self) -> npt.NDArray:
        """Return the stored force_PP array. dim: (mcmc_counter, num_walkers, num_atoms, 3)."""
        return np.asarray(self.__stored_force_PP)

    @property
    def E_L_force_PP_stored(self) -> npt.NDArray:
        """Return the stored E_L * force_PP array. dim: (mcmc_counter, num_walkers, num_atoms, 3)."""
        return np.asarray(self.__stored_E_L_force_PP)

    @property
    def dln_Psi_dc(self) -> dict[str, npt.NDArray]:
        """Return stored parameter gradients (d ln Psi / dc) keyed by block name."""
        return {name: np.asarray(values) for name, values in self.__stored_log_WF_param_grads.items()}

    @property
    def de_L_dc(self) -> dict[str, npt.NDArray]:
        """Return stored local energy parameter gradients (de_L / dc) keyed by block name."""
        return {name: np.asarray(values) for name, values in self.__stored_e_L_param_grads.items()}

    @property
    def comput_position_deriv(self) -> bool:
        """Return the flag for computing the derivatives of E wrt. atomic positions."""
        return self.__comput_position_deriv

    @property
    def compute_log_WF_param_deriv(self) -> bool:
        """Return the flag for computing the derivatives of E wrt. variational parameters."""
        return self.__comput_log_WF_param_deriv

    @property
    def comput_e_L_param_deriv(self) -> bool:
        """Return the flag for computing the local energy derivatives (de_L/dc) wrt. variational parameters."""
        return self.__comput_e_L_param_deriv


# ---------------------------------------------------------------------------
# Module-level JIT kernels for MCMC sampling
#
# These functions are defined at module scope so that they are compiled only
# once by JAX regardless of how many times ``MCMC.run()`` is called.
# Previously, they were local closures re-created on every ``run()``
# invocation, which forced a full re-compilation each time.
# ---------------------------------------------------------------------------


@jit
def _generate_rotation_matrix(jax_PRNG_key):
    """Sample a random 3×3 rotation matrix (Euler angles)."""
    _, subkey = jax.random.split(jax_PRNG_key)
    alpha, beta, gamma = jax.random.uniform(subkey, shape=(3,), minval=-2 * jnp.pi, maxval=2 * jnp.pi)
    cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
    cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
    cos_g, sin_g = jnp.cos(gamma), jnp.sin(gamma)
    R = jnp.array(
        [
            [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
            [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
            [-sin_b, cos_b * sin_a, cos_a * cos_b],
        ]
    )
    return R.T


@jit
def _geminal_inv_single(geminal_data, I, r_up_carts, r_dn_carts):
    """Build G and invert via SVD-based pseudoinverse (single sample)."""
    G = compute_geminal_all_elements(
        geminal_data=geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    U, s, Vt = jnp.linalg.svd(G, full_matrices=False)
    s_inv = jnp.where(s > EPS_rcond_SVD * s[0], 1.0 / s, 0.0)
    Ginv = (Vt.T * s_inv[jnp.newaxis, :]) @ U.T
    return G, Ginv, jnp.zeros_like(G), jnp.zeros(G.shape[0], dtype=jnp.int32)


@jit
def _geminal_inv_batched(geminal_data, r_up_batch, r_dn_batch):
    """Batched geminal inverse over walkers."""
    N_up = r_up_batch.shape[-2]
    I = jnp.eye(N_up)
    G_b, Ginv_b, lu_b, piv_b = vmap(
        _geminal_inv_single,
        in_axes=(None, None, 0, 0),
        out_axes=(0, 0, 0, 0),
    )(geminal_data, I, r_up_batch, r_dn_batch)
    return G_b, Ginv_b, lu_b, piv_b


@partial(jit, static_argnums=3)
def _update_electron_positions(
    init_r_up_carts,
    init_r_dn_carts,
    jax_PRNG_key,
    num_mcmc_per_measurement,
    hamiltonian_data,
    Dt,
    epsilon_AS,
    geminal_inv_init,
    geminal_init,
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
    geminal = geminal_init
    geminal_inv = geminal_inv_init

    def body_fun(_, carry):
        accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key, geminal_inv, geminal = carry
        total_electrons = len(r_up_carts) + len(r_dn_carts)
        num_up_electrons = len(r_up_carts)

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
        nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, old_r_cart)

        # charges
        if hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = jnp.array(hamiltonian_data.structure_data.atomic_numbers) - jnp.array(
                hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = jnp.array(hamiltonian_data.structure_data.atomic_numbers)

        # coords
        coords = hamiltonian_data.structure_data._positions_cart_jnp

        R_cart = coords[nearest_atom_index]
        Z = charges[nearest_atom_index]
        norm_r_R = jnp.linalg.norm(old_r_cart - R_cart)
        f_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

        sigma = f_l * Dt
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        g = jax.random.normal(subkey, shape=()) * sigma

        # choose x,y,or,z
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        random_index = jax.random.randint(subkey, shape=(), minval=0, maxval=3)

        # plug g into g_vector
        g_vector = jnp.zeros(3)
        g_vector = g_vector.at[random_index].set(g)

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
        nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, new_r_cart)

        R_cart = coords[nearest_atom_index]
        Z = charges[nearest_atom_index]
        norm_r_R = jnp.linalg.norm(new_r_cart - R_cart)
        f_prime_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

        T_ratio = (f_l / f_prime_l) * jnp.exp(
            -(jnp.linalg.norm(new_r_cart - old_r_cart) ** 2)
            * (1.0 / (2.0 * f_prime_l**2 * Dt**2) - 1.0 / (2.0 * f_l**2 * Dt**2))
        )

        # Jastrow ratio via dedicated fast-update API (includes exp)
        J_ratio = _compute_ratio_Jastrow_part_rank1_update(
            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
            old_r_up_carts=r_up_carts,
            old_r_dn_carts=r_dn_carts,
            new_r_up_carts_arr=jnp.expand_dims(proposed_r_up_carts, axis=0),
            new_r_dn_carts_arr=jnp.expand_dims(proposed_r_dn_carts, axis=0),
        )[0]

        # Determinant part, fast update using the matrix determinant lemma
        v = lax.cond(
            is_up,
            lambda _: (
                compute_geminal_up_one_row_elements(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    # inline "as_row3": force (1,3) even if source is (3,)
                    r_up_cart=jnp.reshape(proposed_r_up_carts[selected_electron_index], (1, 3)),
                    r_dn_carts=r_dn_carts,
                )
                - compute_geminal_up_one_row_elements(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    r_up_cart=jnp.reshape(r_up_carts[selected_electron_index], (1, 3)),
                    r_dn_carts=r_dn_carts,
                )
            )[:, None],
            lambda _: jax.nn.one_hot(selected_electron_index, num_up_electrons)[:, None],
            operand=None,
        )

        u = lax.cond(
            is_up,
            lambda _: jax.nn.one_hot(selected_electron_index, num_up_electrons)[:, None],  # (N_up, 1)
            lambda _: (
                compute_geminal_dn_one_column_elements(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    r_up_carts=r_up_carts,
                    r_dn_cart=jnp.reshape(proposed_r_dn_carts[selected_electron_index], (1, 3)),  # inline "as_row3"
                )
                - compute_geminal_dn_one_column_elements(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    r_up_carts=r_up_carts,
                    r_dn_cart=jnp.reshape(r_dn_carts[selected_electron_index], (1, 3)),
                )
            )[:, None],  # -> (N_up, 1)
            operand=None,
        )

        # Determinant ratio and rank-1 inverse update:
        # det(A+uv^T)/det(A) = 1 + v^T A^{-1} u
        Ainv_u = geminal_inv @ u  # (N_up,1)
        vT_Ainv = v.T @ geminal_inv  # (1,N_up)
        Det_T_ratio = 1.0 + (v.T @ Ainv_u)[0, 0]  # scalar

        # (A+uv^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)
        geminal_inv_new = geminal_inv - (Ainv_u @ vT_Ainv) / Det_T_ratio

        geminal_new = lax.cond(
            is_up,
            # Row update: row[i, :] += Δrow_i  (v is (N_cols,1) -> squeeze last dim)
            lambda _: geminal.at[selected_electron_index, :].add(v.squeeze(-1)),
            # Column update: col[:, j] += Δcol_j (u is (N_up,1) -> squeeze last dim)
            lambda _: geminal.at[:, selected_electron_index].add(u.squeeze(-1)),
            operand=None,
        )
        # compute AS regularization factors, R_AS and R_AS_eps
        R_AS_p = compute_AS_regularization_factor_fast_update(geminal_new, geminal_inv_new)
        R_AS_p_eps = jnp.maximum(R_AS_p, epsilon_AS)

        R_AS_o = compute_AS_regularization_factor_fast_update(geminal, geminal_inv)
        R_AS_o_eps = jnp.maximum(R_AS_o, epsilon_AS)

        # modified trial WFs
        R_AS_ratio = (R_AS_p_eps / R_AS_p) / (R_AS_o_eps / R_AS_o)
        WF_ratio = J_ratio * Det_T_ratio

        # compute R_ratio
        R_ratio = (R_AS_ratio * WF_ratio) ** 2.0

        acceptance_ratio = jnp.min(jnp.array([1.0, R_ratio * T_ratio]))

        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        b = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)

        def _accepted_fun(_):
            # Move accepted
            return (
                accepted_moves + 1,
                rejected_moves,
                proposed_r_up_carts,
                proposed_r_dn_carts,
                geminal_inv_new,
                geminal_new,
            )

        def _rejected_fun(_):
            # Move rejected
            return (accepted_moves, rejected_moves + 1, r_up_carts, r_dn_carts, geminal_inv, geminal)

        # judge accept or reject the propsed move using jax.lax.cond
        accepted_moves, rejected_moves, r_up_carts, r_dn_carts, geminal_inv, geminal = lax.cond(
            b < acceptance_ratio, _accepted_fun, _rejected_fun, operand=None
        )

        carry = (accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key, geminal_inv, geminal)
        return carry

    # main MCMC loop
    accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key, geminal_inv, geminal = jax.lax.fori_loop(
        0,
        num_mcmc_per_measurement,
        body_fun,
        (accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key, geminal_inv, geminal),
    )

    return (accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key, geminal_inv, geminal)


@partial(jit, static_argnums=3)
def _update_electron_positions_only_up_electron(
    init_r_up_carts,
    init_r_dn_carts,
    jax_PRNG_key,
    num_mcmc_per_measurement,
    hamiltonian_data,
    Dt,
    epsilon_AS,
    geminal_inv_init,
    geminal_init,
):
    """Update electron positions based on the MH method (up-spin electrons only)."""
    accepted_moves = 0
    rejected_moves = 0
    r_up_carts = init_r_up_carts
    r_dn_carts = init_r_dn_carts
    geminal_inv = geminal_inv_init
    geminal = geminal_init

    def body_fun(_, carry):
        accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key, geminal_inv, geminal = carry
        num_up_electrons = len(r_up_carts)

        # dummy jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)

        # Choose randomly if the electron comes from up or dn
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        up_index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(r_up_carts))
        selected_electron_index = up_index

        # dummy jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)

        # choose an up or dn electron from old_r_cart
        old_r_cart = r_up_carts[selected_electron_index]

        # choose the nearest atom index
        nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, old_r_cart)

        # charges
        if hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = jnp.array(hamiltonian_data.structure_data.atomic_numbers) - jnp.array(
                hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = jnp.array(hamiltonian_data.structure_data.atomic_numbers)

        # coords
        coords = hamiltonian_data.structure_data._positions_cart_jnp

        R_cart = coords[nearest_atom_index]
        Z = charges[nearest_atom_index]
        norm_r_R = jnp.linalg.norm(old_r_cart - R_cart)
        f_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

        sigma = f_l * Dt
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        g = jax.random.normal(subkey, shape=()) * sigma

        # choose x,y,or,z
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        random_index = jax.random.randint(subkey, shape=(), minval=0, maxval=3)

        # plug g into g_vector
        g_vector = jnp.zeros(3)
        g_vector = g_vector.at[random_index].set(g)

        new_r_cart = old_r_cart + g_vector

        # set proposed r_up_carts and r_dn_carts.
        proposed_r_up_carts = r_up_carts.at[selected_electron_index].set(new_r_cart)
        proposed_r_dn_carts = r_dn_carts

        # choose the nearest atom index
        nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, new_r_cart)

        R_cart = coords[nearest_atom_index]
        Z = charges[nearest_atom_index]
        norm_r_R = jnp.linalg.norm(new_r_cart - R_cart)
        f_prime_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

        T_ratio = (f_l / f_prime_l) * jnp.exp(
            -(jnp.linalg.norm(new_r_cart - old_r_cart) ** 2)
            * (1.0 / (2.0 * f_prime_l**2 * Dt**2) - 1.0 / (2.0 * f_l**2 * Dt**2))
        )

        # original trial WFs
        Jastrow_T_p = compute_Jastrow_part(
            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
            r_up_carts=proposed_r_up_carts,
            r_dn_carts=proposed_r_dn_carts,
        )

        Jastrow_T_o = compute_Jastrow_part(
            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        # Determinant part, fast update using the matrix determinant lemma
        v = (
            compute_geminal_up_one_row_elements(
                geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                # inline "as_row3": force (1,3) even if source is (3,)
                r_up_cart=jnp.reshape(proposed_r_up_carts[selected_electron_index], (1, 3)),
                r_dn_carts=r_dn_carts,
            )
            - compute_geminal_up_one_row_elements(
                geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                r_up_cart=jnp.reshape(r_up_carts[selected_electron_index], (1, 3)),
                r_dn_carts=r_dn_carts,
            )
        )[:, None]

        u = jax.nn.one_hot(selected_electron_index, num_up_electrons)[:, None]

        # Determinant ratio and rank-1 inverse update:
        # det(A+uv^T)/det(A) = 1 + v^T A^{-1} u
        Ainv_u = geminal_inv @ u
        vT_Ainv = v.T @ geminal_inv
        Det_T_ratio = 1.0 + (v.T @ Ainv_u)[0, 0]  # scalar

        # (A+uv^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)
        geminal_inv_new = geminal_inv - (Ainv_u @ vT_Ainv) / Det_T_ratio

        geminal_new = geminal.at[selected_electron_index, :].add(v.squeeze(-1))

        # compute AS regularization factors, R_AS and R_AS_eps
        R_AS_p = compute_AS_regularization_factor_fast_update(geminal_new, geminal_inv_new)
        R_AS_p_eps = jnp.maximum(R_AS_p, epsilon_AS)

        R_AS_o = compute_AS_regularization_factor_fast_update(geminal, geminal_inv)
        R_AS_o_eps = jnp.maximum(R_AS_o, epsilon_AS)

        # modified trial WFs
        R_AS_ratio = (R_AS_p_eps / R_AS_p) / (R_AS_o_eps / R_AS_o)
        WF_ratio = jnp.exp(Jastrow_T_p - Jastrow_T_o) * (Det_T_ratio)

        # compute R_ratio
        R_ratio = (R_AS_ratio * WF_ratio) ** 2.0

        acceptance_ratio = jnp.min(jnp.array([1.0, R_ratio * T_ratio]))

        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        b = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)

        def _accepted_fun(_):
            # Move accepted
            return (
                accepted_moves + 1,
                rejected_moves,
                proposed_r_up_carts,
                proposed_r_dn_carts,
                geminal_inv_new,
                geminal_new,
            )

        def _rejected_fun(_):
            # Move rejected
            return (accepted_moves, rejected_moves + 1, r_up_carts, r_dn_carts, geminal_inv, geminal)

        # judge accept or reject the propsed move using jax.lax.cond
        accepted_moves, rejected_moves, r_up_carts, r_dn_carts, geminal_inv, geminal = lax.cond(
            b < acceptance_ratio, _accepted_fun, _rejected_fun, operand=None
        )

        carry = (accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key, geminal_inv, geminal)
        return carry

    # main MCMC loop
    accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key, geminal_inv, geminal = jax.lax.fori_loop(
        0,
        num_mcmc_per_measurement,
        body_fun,
        (accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key, geminal_inv, geminal),
    )

    return (accepted_moves, rejected_moves, r_up_carts, r_dn_carts, jax_PRNG_key, geminal_inv, geminal)


# Module-level vmap/jit wrappers for MCMC kernels.
# Created once at import time so subsequent MCMC.run() calls reuse
# the same Python function objects and hit JAX's compilation cache.
_jit_vmap_update = jit(
    vmap(_update_electron_positions, in_axes=(0, 0, 0, None, None, None, None, 0, 0)),
    static_argnums=3,
)
_jit_vmap_update_up = jit(
    vmap(_update_electron_positions_only_up_electron, in_axes=(0, 0, 0, None, None, None, None, 0, 0)),
    static_argnums=3,
)
_jit_vmap_e_L_fast = jit(vmap(compute_local_energy_fast, in_axes=(None, 0, 0, 0, 0)))
_jit_vmap_as_reg = jit(vmap(compute_AS_regularization_factor, in_axes=(None, 0, 0)))
_jit_vmap_generate_RTs = jit(vmap(_generate_rotation_matrix, in_axes=0))
_jit_vmap_as_reg_fast = jit(vmap(compute_AS_regularization_factor_fast_update, in_axes=(0, 0)))
_jit_vmap_swct_omega = jit(vmap(evaluate_swct_omega, in_axes=(None, 0)))
_jit_vmap_swct_domega = jit(vmap(evaluate_swct_domega, in_axes=(None, 0)))

# grad-based wrappers
_jit_vmap_grad_e_L_r = jit(vmap(grad(compute_local_energy, argnums=(1, 2)), in_axes=(None, 0, 0, 0)))
_jit_vmap_grad_e_L_h = jit(vmap(grad(compute_local_energy, argnums=0), in_axes=(None, 0, 0, 0)))
_jit_vmap_grad_ln_psi = jit(vmap(grad(evaluate_ln_wavefunction, argnums=(0, 1, 2)), in_axes=(None, 0, 0)))
_jit_vmap_grad_ln_psi_params = jit(vmap(grad(evaluate_ln_wavefunction, argnums=0), in_axes=(None, 0, 0)))
_jit_vmap_grad_ln_psi_params_fast = jit(vmap(grad(evaluate_ln_wavefunction_fast, argnums=0), in_axes=(None, 0, 0, 0)))


class _MCMC_debug:
    """MCMC with multiple walker class.

    This is for debugging purpose, to check the correctness of the MCMC implementation and the derivatives.

    MCMC class. Runing MCMC with multiple walkers. The independent 'num_walkers' MCMCs are
    vectrized via the jax-vmap function.

    Args:
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data.
        mcmc_seed (int): seed for the MCMC chain.
        num_walkers (int): the number of walkers.
        num_mcmc_per_measurement (int): the number of MCMC steps between a value (e.g., local energy) measurement.
        Dt (float): electron move step (bohr)
        epsilon_AS (float): the exponent of the AS regularization
        comput_log_WF_param_deriv (bool): if True, compute the derivatives of E wrt. variational parameters.
        comput_e_L_param_deriv (bool, optional): Keep local energy variational parameter derivatives (de_L / dc).
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
        comput_log_WF_param_deriv: bool = False,
        comput_e_L_param_deriv: bool = False,
        comput_position_deriv: bool = False,
        random_discretized_mesh: bool = True,
    ) -> None:
        """Initialize a MCMC class, creating list holding results."""
        self.__mcmc_seed = mcmc_seed
        self.__num_walkers = num_walkers
        self.__num_mcmc_per_measurement = num_mcmc_per_measurement
        self.__Dt = Dt
        self.__epsilon_AS = epsilon_AS
        self.__comput_log_WF_param_deriv = comput_log_WF_param_deriv
        self.__comput_e_L_param_deriv = comput_e_L_param_deriv
        self.__comput_position_deriv = comput_position_deriv
        self.__random_discretized_mesh = random_discretized_mesh

        # set hamiltonian_data
        self.__hamiltonian_data = hamiltonian_data

        # seeds
        self.__mpi_seed = self.__mcmc_seed * (mpi_rank + 1)
        self.__jax_PRNG_key = jax.random.PRNGKey(self.__mpi_seed)
        self.__jax_PRNG_key_list = jnp.array([jax.random.fold_in(self.__jax_PRNG_key, nw) for nw in range(self.__num_walkers)])

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

        # check if only up electrons are updated
        if tot_num_electron_dn == 0:
            self.__only_up_electron = True
        else:
            self.__only_up_electron = False

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
        # (SWCT functions now take structure_data directly; no wrapper needed.)

        # init_attributes
        self.__init_attributes()

    def __init_attributes(self):
        # mcmc counter
        self.__mcmc_counter = 0

        # mcmc accepted/rejected moves (kept on device; convert to host only when logging)
        self.__accepted_moves = jnp.array(0, dtype=jnp.int64)
        self.__rejected_moves = jnp.array(0, dtype=jnp.int64)

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

        # stored parameter gradients keyed by block name
        self.__stored_log_WF_param_grads: dict[str, list] = defaultdict(list)

        # stored de_L/dc parameter gradients keyed by block name
        self.__stored_e_L_param_grads: dict[str, list] = defaultdict(list)

    def run(self, num_mcmc_steps: int = 0) -> None:
        """Launch MCMCs with the set multiple walkers.

        Args:
            num_mcmc_steps (int):
                the number of total mcmc steps per walker.
            max_time(int):
                Max elapsed time (sec.). If the elapsed time exceeds max_time, the methods exits the mcmc loop.
        """
        logger.info("")
        logger.info("This is a debugging class! It supposed to be very slow.")
        logger.info("")

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
                    nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, old_r_cart)

                    # charges
                    if hamiltonian_data.coulomb_potential_data.ecp_flag:
                        charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - jnp.array(
                            hamiltonian_data.coulomb_potential_data.z_cores
                        )
                    else:
                        charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

                    # coords
                    coords = hamiltonian_data.structure_data._positions_cart_np

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
                    nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, new_r_cart)

                    R_cart = coords[nearest_atom_index]
                    Z = charges[nearest_atom_index]
                    norm_r_R = np.linalg.norm(new_r_cart - R_cart)
                    f_prime_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

                    T_ratio = (f_l / f_prime_l) * jnp.exp(
                        -(np.linalg.norm(new_r_cart - old_r_cart) ** 2)
                        * (1.0 / (2.0 * f_prime_l**2 * Dt**2) - 1.0 / (2.0 * f_l**2 * Dt**2))
                    )

                    # original trial WFs
                    Jastrow_T_p = compute_Jastrow_part(
                        jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                        r_up_carts=proposed_r_up_carts,
                        r_dn_carts=proposed_r_dn_carts,
                    )

                    Jastrow_T_o = compute_Jastrow_part(
                        jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    Det_T_p = compute_det_geminal_all_elements(
                        geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                        r_up_carts=proposed_r_up_carts,
                        r_dn_carts=proposed_r_dn_carts,
                    )

                    Det_T_o = compute_det_geminal_all_elements(
                        geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    # compute AS regularization factors, R_AS and R_AS_eps
                    R_AS_p = compute_AS_regularization_factor(
                        geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                        r_up_carts=proposed_r_up_carts,
                        r_dn_carts=proposed_r_dn_carts,
                    )
                    R_AS_p_eps = jnp.maximum(R_AS_p, epsilon_AS)

                    R_AS_o = compute_AS_regularization_factor(
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
            self.__accepted_moves = self.__accepted_moves + np.sum(accepted_moves_nw)
            self.__rejected_moves = self.__rejected_moves + np.sum(rejected_moves_nw)
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
            e_L = jnp.stack(
                [
                    compute_local_energy(
                        self.__hamiltonian_data, self.__latest_r_up_carts[i], self.__latest_r_dn_carts[i], RTs[i]
                    )
                    for i in range(self.__num_walkers)
                ]
            )
            self.__stored_e_L.append(e_L)
            self.__stored_e_L2.append(e_L**2)

            # compute AS regularization factors, R_AS and R_AS_eps
            R_AS = jnp.stack(
                [
                    compute_AS_regularization_factor(
                        self.__hamiltonian_data.wavefunction_data.geminal_data,
                        self.__latest_r_up_carts[i],
                        self.__latest_r_dn_carts[i],
                    )
                    for i in range(self.__num_walkers)
                ]
            )
            R_AS_eps = jnp.maximum(R_AS, self.__epsilon_AS)

            w_L = (R_AS / R_AS_eps) ** 2
            self.__stored_w_L.append(w_L)

            if self.__comput_position_deriv:
                _grad_e_L_fn = grad(compute_local_energy, argnums=(0, 1, 2))
                _grad_e_L_results = [
                    _grad_e_L_fn(self.__hamiltonian_data, self.__latest_r_up_carts[i], self.__latest_r_dn_carts[i], RTs[i])
                    for i in range(self.__num_walkers)
                ]
                grad_e_L_h = jax.tree_util.tree_map(lambda *arrs: jnp.stack(arrs), *[r[0] for r in _grad_e_L_results])
                grad_e_L_r_up = jnp.stack([r[1] for r in _grad_e_L_results])
                grad_e_L_r_dn = jnp.stack([r[2] for r in _grad_e_L_results])

                self.__stored_grad_e_L_r_up.append(grad_e_L_r_up)
                self.__stored_grad_e_L_r_dn.append(grad_e_L_r_dn)

                grad_e_L_R = self.__hamiltonian_data.accumulate_position_grad(grad_e_L_h)
                self.__stored_grad_e_L_dR.append(grad_e_L_R)

                _grad_ln_psi_fn = grad(evaluate_ln_wavefunction, argnums=(0, 1, 2))
                _grad_ln_psi_results = [
                    _grad_ln_psi_fn(
                        self.__hamiltonian_data.wavefunction_data,
                        self.__latest_r_up_carts[i],
                        self.__latest_r_dn_carts[i],
                    )
                    for i in range(self.__num_walkers)
                ]
                grad_ln_Psi_h = jax.tree_util.tree_map(lambda *arrs: jnp.stack(arrs), *[r[0] for r in _grad_ln_psi_results])
                grad_ln_Psi_r_up = jnp.stack([r[1] for r in _grad_ln_psi_results])
                grad_ln_Psi_r_dn = jnp.stack([r[2] for r in _grad_ln_psi_results])

                self.__stored_grad_ln_Psi_r_up.append(grad_ln_Psi_r_up)
                self.__stored_grad_ln_Psi_r_dn.append(grad_ln_Psi_r_dn)

                grad_ln_Psi_dR = self.__hamiltonian_data.wavefunction_data.accumulate_position_grad(grad_ln_Psi_h)
                self.__stored_grad_ln_Psi_dR.append(grad_ln_Psi_dR)

                omega_up = jnp.stack(
                    [
                        evaluate_swct_omega(self.__hamiltonian_data.structure_data, self.__latest_r_up_carts[i])
                        for i in range(self.__num_walkers)
                    ]
                )

                omega_dn = jnp.stack(
                    [
                        evaluate_swct_omega(self.__hamiltonian_data.structure_data, self.__latest_r_dn_carts[i])
                        for i in range(self.__num_walkers)
                    ]
                )

                self.__stored_omega_up.append(omega_up)
                self.__stored_omega_dn.append(omega_dn)

                grad_omega_dr_up = jnp.stack(
                    [
                        evaluate_swct_domega(self.__hamiltonian_data.structure_data, self.__latest_r_up_carts[i])
                        for i in range(self.__num_walkers)
                    ]
                )

                grad_omega_dr_dn = jnp.stack(
                    [
                        evaluate_swct_domega(self.__hamiltonian_data.structure_data, self.__latest_r_dn_carts[i])
                        for i in range(self.__num_walkers)
                    ]
                )

                self.__stored_grad_omega_r_up.append(grad_omega_dr_up)
                self.__stored_grad_omega_r_dn.append(grad_omega_dr_dn)

            if self.__comput_log_WF_param_deriv:
                _grad_ln_psi_p_fn = grad(evaluate_ln_wavefunction, argnums=0)
                _grad_ln_psi_p_results = [
                    _grad_ln_psi_p_fn(
                        self.__hamiltonian_data.wavefunction_data,
                        self.__latest_r_up_carts[i],
                        self.__latest_r_dn_carts[i],
                    )
                    for i in range(self.__num_walkers)
                ]
                grad_ln_Psi_h = jax.tree_util.tree_map(lambda *arrs: jnp.stack(arrs), *_grad_ln_psi_p_results)

                param_grads = self.__hamiltonian_data.wavefunction_data.collect_param_grads(grad_ln_Psi_h)
                flat_param_grads = self.__hamiltonian_data.wavefunction_data.flatten_param_grads(
                    param_grads, self.__num_walkers
                )

                for name, grad_val in flat_param_grads.items():
                    self.__stored_log_WF_param_grads[name].append(grad_val)
                    if hasattr(grad_val, "block_until_ready"):
                        grad_val.block_until_ready()

            if self.__comput_e_L_param_deriv:
                _e_L_wf_grad_fn = grad(
                    lambda wf, r_up, r_dn, RT: compute_local_energy(
                        self.__hamiltonian_data.replace(wavefunction_data=wf), r_up, r_dn, RT
                    )
                )
                _e_L_wf_results = [
                    _e_L_wf_grad_fn(
                        self.__hamiltonian_data.wavefunction_data,
                        self.__latest_r_up_carts[i],
                        self.__latest_r_dn_carts[i],
                        RTs[i],
                    )
                    for i in range(self.__num_walkers)
                ]
                grad_e_L_wf = jax.tree_util.tree_map(lambda *arrs: jnp.stack(arrs), *_e_L_wf_results)
                flat_grads = self.__hamiltonian_data.wavefunction_data.flatten_param_grads(
                    self.__hamiltonian_data.wavefunction_data.collect_param_grads(grad_e_L_wf),
                    self.__num_walkers,
                )
                for name, grad_val in flat_grads.items():
                    self.__stored_e_L_param_grads[name].append(grad_val)

            num_mcmc_done += 1

        logger.info("End MCMC")
        logger.info("")

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
        e_L = self.e_L[num_mcmc_warmup_steps:]
        e_L2 = self.e_L2[num_mcmc_warmup_steps:]
        w_L = self.w_L[num_mcmc_warmup_steps:]
        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
        w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
        w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
        w_L_e_L2_split = np.array_split(w_L * e_L2, num_mcmc_bin_blocks, axis=0)
        w_L_e_L2_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L2_split]))

        # MCMC case
        w_L_binned_local = w_L_binned
        w_L_e_L_binned_local = w_L_e_L_binned
        w_L_e_L2_binned_local = w_L_e_L2_binned

        w_L_binned_local = np.array(w_L_binned_local)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
        w_L_e_L2_binned_local = np.array(w_L_e_L2_binned_local)

        w_L_binned_global_sum = mpi_comm.allreduce(np.sum(w_L_binned_local, axis=0), op=MPI.SUM)
        w_L_e_L_binned_global_sum = mpi_comm.allreduce(np.sum(w_L_e_L_binned_local, axis=0), op=MPI.SUM)
        w_L_e_L2_binned_global_sum = mpi_comm.allreduce(np.sum(w_L_e_L2_binned_local, axis=0), op=MPI.SUM)

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
        E_jackknife_binned = mpi_comm.allreduce(E_jackknife_binned_local, op=MPI.SUM)
        Var_jackknife_binned = mpi_comm.allreduce(Var_jackknife_binned_local, op=MPI.SUM)
        E_jackknife_binned = np.array(E_jackknife_binned)
        Var_jackknife_binned = np.array(Var_jackknife_binned)
        M_total = len(E_jackknife_binned)
        logger.debug(f"The number of total binned samples = {M_total}")

        # jackknife mean and std
        E_mean = np.average(E_jackknife_binned)
        E_std = np.sqrt(M_total - 1) * np.std(E_jackknife_binned)
        Var_mean = np.average(Var_jackknife_binned)
        Var_std = np.sqrt(M_total - 1) * np.std(Var_jackknife_binned)

        logger.info(f"E = {E_mean} +- {E_std} Ha.")
        logger.info(f"Var(E) = {Var_mean} +- {Var_std} Ha^2.")

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
            de_L_dR + np.einsum("iwjk,iwkl->iwjl", omega_up, de_L_dr_up) + np.einsum("iwjk,iwkl->iwjl", omega_dn, de_L_dr_dn)
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

        # MCMC case
        w_L_binned_local = w_L_binned
        w_L_e_L_binned_local = w_L_e_L_binned
        w_L_force_HF_binned_local = w_L_force_HF_binned
        w_L_force_PP_binned_local = w_L_force_PP_binned
        w_L_E_L_force_PP_binned_local = w_L_E_L_force_PP_binned

        w_L_binned_local = np.array(w_L_binned_local)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
        w_L_force_HF_binned_local = np.array(w_L_force_HF_binned_local)
        w_L_force_PP_binned_local = np.array(w_L_force_PP_binned_local)
        w_L_E_L_force_PP_binned_local = np.array(w_L_E_L_force_PP_binned_local)

        # old implementation (keep this just for debug, for the time being. To be deleted.)
        w_L_binned_global_sum = mpi_comm.allreduce(np.sum(w_L_binned_local, axis=0), op=MPI.SUM)
        w_L_e_L_binned_global_sum = mpi_comm.allreduce(np.sum(w_L_e_L_binned_local, axis=0), op=MPI.SUM)
        w_L_force_HF_binned_global_sum = mpi_comm.allreduce(np.sum(w_L_force_HF_binned_local, axis=0), op=MPI.SUM)
        w_L_force_PP_binned_global_sum = mpi_comm.allreduce(np.sum(w_L_force_PP_binned_local, axis=0), op=MPI.SUM)
        w_L_E_L_force_PP_binned_global_sum = mpi_comm.allreduce(np.sum(w_L_E_L_force_PP_binned_local, axis=0), op=MPI.SUM)

        M_local = w_L_binned_local.size
        logger.debug(f"The number of local binned samples = {M_local}")

        force_HF_jn_local = -1.0 * np.array(
            [
                (w_L_force_HF_binned_global_sum - w_L_force_HF_binned_local[j]) / (w_L_binned_global_sum - w_L_binned_local[j])
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

        # MPI allreduce
        force_jn = mpi_comm.allreduce(force_jn_local, op=MPI.SUM)
        force_jn = np.array(force_jn)
        M_total = len(force_jn)
        logger.debug(f"The number of total binned samples = {M_total}")

        force_mean = np.average(force_jn, axis=0)
        force_std = np.sqrt(M_total - 1) * np.std(force_jn, axis=0)

        logger.devel(f"force_mean.shape  = {force_mean.shape}.")
        logger.devel(f"force_std.shape  = {force_std.shape}.")
        logger.info(f"force = {force_mean} +- {force_std} Ha.")

        w_L_binned_local = np.array(w_L_binned_local)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
        w_L_force_HF_binned_local = np.array(w_L_force_HF_binned_local)
        w_L_force_PP_binned_local = np.array(w_L_force_PP_binned_local)
        w_L_E_L_force_PP_binned_local = np.array(w_L_E_L_force_PP_binned_local)

        logger.devel(f"force_mean.shape  = {force_mean.shape}.")
        logger.devel(f"force_std.shape  = {force_std.shape}.")
        logger.devel(f"force = {force_mean} +- {force_std} Ha.")

        return (force_mean, force_std)

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
    def dln_Psi_dc(self) -> dict[str, npt.NDArray]:
        """Return stored d ln Psi / dc parameter gradients keyed by block name."""
        return {name: np.array(values) for name, values in self.__stored_log_WF_param_grads.items()}

    @property
    def de_L_dc(self) -> dict[str, npt.NDArray]:
        """Return stored de_L / dc parameter gradients keyed by block name."""
        return {name: np.array(values) for name, values in self.__stored_e_L_param_grads.items()}

    @property
    def comput_e_L_param_deriv(self) -> bool:
        """Return whether de_L/dc computation is enabled."""
        return self.__comput_e_L_param_deriv

    def get_aH(
        self,
        blocks: list,
        g: npt.NDArray | None = None,
        num_mcmc_warmup_steps: int = 50,
        chosen_param_index: list | None = None,
        lambda_projectors=None,
        num_orb_projection=None,
        return_matrices: bool = False,
    ) -> tuple:
        r"""Compute H_0, H_1, H_2, S_2 for accelerated SR (aSR) gamma optimization.

        This is the debug (single-rank, no vmap, explicit-step) counterpart of
        ``MCMC.get_aH``.  Every intermediate quantity is computed in a named
        numpy step that maps directly to the mathematical formula so that the
        logic can be followed line by line.

        .. note::

            ``return_matrices=True`` (LM mode) is **not implemented** in this
            debug class.  It will raise ``NotImplementedError``.

        Mathematical background
        -----------------------
        The energy along the aSR step  alpha -> alpha + gamma * g  is
        approximated as

            E(alpha + gamma*g)  =    (H0 + 2*gamma*H1 + gamma^2*H2)
                                    --------------------------------
                                          1 + gamma^2 * S2

        where g = S^{-1} f is the natural gradient and:

            H0  = E_alpha                              (current energy)
            H1  = -1/2 * g^T f                         (f = generalized force)
            H2  = g^T (B + K) g                        (curvature along g)
            S2  = g^T S g = g^T f = -2*H1             (overlap along g)

        Intermediate quantities (N = total samples, K = # params):

            w(i)        sample weight                   shape (N,)
            e_L(i)      local energy                    shape (N,)
            W           sum_i w(i)                      scalar
            E_bar       sum_i w(i) e_L(i) / W           scalar
            O_{i,k}     d ln Psi(i) / d c_k             shape (N, K)
            O_bar_k     sum_i w(i) O_{i,k} / W          shape (K,)
            dO_{i,k}    O_{i,k} - O_bar_k               shape (N, K)  centered
            dE_{i,k}    d e_L(i) / d c_k                shape (N, K)
            dE_bar_k    sum_i w(i) dE_{i,k} / W         shape (K,)
            ddE_{i,k}   dE_{i,k} - dE_bar_k             shape (N, K)  centered
            f_k         -2/W * sum_i w(i)(e_L-E_bar) dO_{i,k}   shape (K,)

        Args:
            blocks: Ordered variational blocks (same ordering as g).
            g: Natural gradient vector S^{-1}f. Required for aSR mode.
            num_mcmc_warmup_steps: Samples to discard as warmup.
            chosen_param_index: Optional subset of flattened indices.
            lambda_projectors: Not used in this debug implementation (accepted
                for API compatibility with MCMC.get_aH).
            num_orb_projection: Not used in this debug implementation.
            return_matrices: If True, raise NotImplementedError (LM not supported in debug).

        Returns:
            (H_0, H_1, H_2, S_2) — aSR mode only.
        """
        if not self.__comput_log_WF_param_deriv:
            raise RuntimeError("get_aH requires compute_log_WF_param_deriv=True.")
        if not self.__comput_e_L_param_deriv:
            raise RuntimeError("get_aH requires comput_e_L_param_deriv=True.")

        # ── Step 1: Raw samples after warmup ──────────────────────────────────
        # e_L_2d, w_L_2d have shape (M_steps, num_walkers).
        e_L_2d = self.e_L[num_mcmc_warmup_steps:]  # (M, nw)
        w_L_2d = self.w_L[num_mcmc_warmup_steps:]  # (M, nw)

        # Flatten (M, nw) -> (N,) so each sample is one entry.
        M, nw = e_L_2d.shape
        N = M * nw
        e_L = e_L_2d.ravel()  # (N,)
        w = w_L_2d.ravel()  # (N,)

        # ── Step 2: Build O_matrix  (d ln Psi / dc)  shape (N, K) ────────────
        # dln_Psi_dc is a dict  block_name -> array (M, nw, ...).
        # We gather blocks in the same order as `blocks` and flatten the
        # parameter dimensions to get a single (N, K) matrix.
        dln_Psi_dc_map = self.dln_Psi_dc
        O_cols = []
        for block in blocks:
            if block.name not in dln_Psi_dc_map:
                # grad flag was off for this block; treat as zero
                O_cols.append(np.zeros((N, block.size)))
                continue
            arr = dln_Psi_dc_map[block.name][num_mcmc_warmup_steps:]  # (M, nw, ...)
            arr = arr.reshape(N, -1)  # (N, block.size)
            O_cols.append(arr)
        O_matrix = np.concatenate(O_cols, axis=1)  # (N, K)

        # ── Step 3: Build dE_matrix  (de_L / dc)  shape (N, K) ───────────────
        de_L_dc_map = self.de_L_dc
        dE_cols = []
        for block in blocks:
            if block.name not in de_L_dc_map:
                dE_cols.append(np.zeros((N, block.size)))
                continue
            arr = de_L_dc_map[block.name][num_mcmc_warmup_steps:]  # (M, nw, ...)
            arr = arr.reshape(N, -1)  # (N, block.size)
            dE_cols.append(arr)
        dE_matrix = np.concatenate(dE_cols, axis=1)  # (N, K)

        # Apply optional parameter-index subset.
        if chosen_param_index is not None:
            O_matrix = O_matrix[:, chosen_param_index]
            dE_matrix = dE_matrix[:, chosen_param_index]
            g = g[chosen_param_index]

        # ── Step 4: Weighted averages ─────────────────────────────────────────
        W = float(np.sum(w))  # total weight
        E_bar = float(np.dot(w, e_L) / W)  # <e_L>_w
        O_bar = (w @ O_matrix) / W  # (K,)   <O_k>_w
        dE_bar = (w @ dE_matrix) / W  # (K,)   <dE_k>_w

        # ── Step 5: H_0  (current energy estimate) ──────────────────────────
        H_0 = E_bar

        # ── Step 6: Centered observables ──────────────────────────────────────
        dO = O_matrix - O_bar[np.newaxis, :]  # (N, K)  O_k(i) - <O_k>
        ddE = dE_matrix - dE_bar[np.newaxis, :]  # (N, K)  dE_k(i) - <dE_k>

        # ── Step 7: Generalized force  f_k = -2/W sum_i w_i (e_L_i - E_bar) dO_{i,k}
        de = e_L - E_bar  # (N,)  local energy fluctuation
        f_vec = -2.0 * (w * de) @ dO / W  # (K,)

        # ── LM mode: build full matrices ─────────────────────────────────────
        if return_matrices:
            # If g (SR direction) is provided, prepend collective variable
            if g is not None:
                O_SR = dO @ g  # (N,)
                dE_SR = ddE @ g  # (N,)
                dO = np.column_stack([O_SR, dO])  # (N, K+1)
                ddE = np.column_stack([dE_SR, ddE])  # (N, K+1)
                # Recompute f_vec for extended space
                f_vec = -2.0 * (w * de) @ dO / W

            # S_{k,k'} = (dO^T @ diag(w) @ dO) / W
            w_dO = w[:, np.newaxis] * dO
            S_matrix = (dO.T @ w_dO) / W

            # K_{k,k'} = (dO^T @ diag(w * e_L) @ dO) / W
            we_dO = (w * e_L)[:, np.newaxis] * dO
            K_matrix = (dO.T @ we_dO) / W

            # B_{k,k'} = (ddE^T @ diag(w) @ dO) / W
            w_ddE = w[:, np.newaxis] * ddE
            B_matrix = (w_ddE.T @ dO) / W

            return H_0, f_vec, S_matrix, K_matrix, B_matrix

        # ── aSR mode: scalar projections along g ─────────────────────────────
        assert g is not None, "g is required for aSR mode (return_matrices=False)"

        # ── Step 8: H_1 = -1/2 * g^T f ──────────────────────────────────────
        H_1 = -0.5 * float(np.dot(g, f_vec))

        # ── Step 9: S_2 = g^T S g = <w (g^T dO)^2>_w  (exact, computed from samples) ──
        # Do NOT use S_2 = g^T f (= -2*H_1).  The SR solved
        # (S_scaled + sr_epsilon*I) g_scaled = b, so
        #   g^T f = g^T S g + sr_epsilon * ||g_scaled||^2.
        # Using g^T f overestimates S_2, makes the denominator of
        # E(gamma) grow too fast, and drives the optimal gamma to be unrealistically small.
        gdO = dO @ g  # (N,)  g-projected centered observable
        S_2 = float(np.dot(w, gdO**2) / W)

        # ── Step 10: K matrix contribution  g^T K g ─────────────────────────
        #
        #   K_{k,k'} = 1/W sum_i  w_i * e_L_i * dO_{i,k} * dO_{i,k'}
        #
        #   g^T K g  = 1/W sum_i  w_i * e_L_i * (sum_k g_k dO_{i,k})^2
        gKg = float(np.dot(w * e_L * gdO, gdO) / W)

        # ── Step 11: B matrix contribution  g^T B g ─────────────────────────
        #
        #   B_{k,k'} = 1/W sum_i  w_i * dO_{i,k} * ddE_{i,k'}
        #   (B is generally not symmetric)
        #
        #   g^T B g  = 1/W sum_i  w_i * (sum_k g_k dO_{i,k}) * (sum_k' g_k' ddE_{i,k'})
        gdE = ddE @ g  # (N,)  g-projected centered de_L
        gBg = float(np.dot(w * gdO, gdE) / W)

        # ── Step 12: H_2 = g^T (B + K) g ────────────────────────────────────
        H_2 = gBg + gKg

        return H_0, H_1, H_2, S_2

    @staticmethod
    def solve_linear_method(
        H_0: float,
        f_vec: npt.NDArray,
        S_matrix: npt.NDArray,
        K_matrix: npt.NDArray,
        B_matrix: npt.NDArray,
        epsilon: float,
    ) -> tuple[npt.NDArray, float]:
        r"""Debug implementation of the Linear Method with dgelscut preconditioning.

        This mirrors ``MCMC.solve_linear_method`` using the same dgelscut
        algorithm (correlation-matrix-based parameter removal) followed by
        S-orthonormalization and standard eigenvalue problem.

        Args:
            H_0: Current energy.
            f_vec: Generalized force vector, shape (p,).
            S_matrix: SR (overlap) matrix, shape (p, p).
            K_matrix: K matrix, shape (p, p).
            B_matrix: B matrix, shape (p, p).
            epsilon: dgelscut threshold (correlation matrix min eigenvalue).

        Returns:
            (c_vec, E_lm): parameter update in original space and selected eigenvalue.
        """
        # Delegate to MCMC.solve_linear_method — the production version uses
        # the same dgelscut + S-orthonormalization + standard eigenvalue problem.
        # Duplicating the dgelscut loop in explicit form adds no clarity;
        # the debug value comes from get_aH (matrix construction), not the solver.
        return MCMC.solve_linear_method(H_0, f_vec, S_matrix, K_matrix, B_matrix, epsilon)

    @staticmethod
    def compute_asr_gamma(H_0: float, H_1: float, H_2: float, S_2: float) -> float:
        r"""Solve for the optimal gamma in the accelerated SR energy minimisation.

        Finds gamma that minimizes  E(alpha + gamma*g)  by solving

            d/d(gamma)  (H0 + 2*gamma*H1 + gamma^2*H2) / (1 + gamma^2*S2)  =  0

        Setting the derivative to zero gives the quadratic:

            -H1*S2 * gamma^2  +  (H2 - H0*S2) * gamma  +  H1  =  0

        whose roots are:

            gamma = ( (H0*S2 - H2) +/- sqrt((H2 - H0*S2)^2 + 4*H1^2*S2) )
                    / (-2 * H1 * S2)

        The positive root is returned; if neither is positive a warning is
        logged and the root with the larger absolute value is returned.

        Args:
            H_0: Current energy E_alpha.
            H_1: H1 = -1/2 g^T f.
            H_2: H2 = g^T (B + K) g.
            S_2: S2 = g^T S g = <w (g^T dO)^2>_w.

        Returns:
            Optimal gamma (float).
        """
        B = H_2 - H_0 * S_2
        discriminant = B**2 + 4.0 * H_1**2 * S_2

        if discriminant < 0.0:
            logger.warning(f"aSR: discriminant is negative ({discriminant:.3e}); setting to 0.")
            discriminant = 0.0

        denom = -2.0 * H_1 * S_2
        sqrt_d = np.sqrt(discriminant)
        gamma_plus = (-B + sqrt_d) / denom
        gamma_minus = (-B - sqrt_d) / denom

        logger.info(f"aSR: gamma+ = {gamma_plus:.6f}, gamma- = {gamma_minus:.6f}")

        # Return the positive root; prefer the smaller one when both are positive.
        if gamma_plus > 0.0 and gamma_minus <= 0.0:
            gamma = gamma_plus
        elif gamma_minus > 0.0 and gamma_plus <= 0.0:
            gamma = gamma_minus
        elif gamma_plus > 0.0 and gamma_minus > 0.0:
            gamma = min(gamma_plus, gamma_minus)
            logger.warning(
                f"aSR: both roots positive (gamma+={gamma_plus:.6f}, gamma-={gamma_minus:.6f}); "
                f"using smaller root gamma={gamma:.6f}."
            )
        else:
            gamma = gamma_plus if abs(gamma_plus) >= abs(gamma_minus) else gamma_minus
            logger.warning(
                f"aSR: neither root is positive (gamma+={gamma_plus:.6f}, gamma-={gamma_minus:.6f}); "
                f"using root with larger absolute value gamma={gamma:.6f}."
            )

        return float(gamma)


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
