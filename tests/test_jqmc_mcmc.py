"""collections of unit tests."""

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

import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

# Add the project root directory to sys.path to allow executing this script directly
# This is necessary because relative imports (e.g. 'from ..jqmc') are not allowed
# when running a script directly (as __main__).
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance_min
from jqmc._setting import atol_consistency, rtol_consistency
from jqmc.determinant import Geminal_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_NN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.jqmc_mcmc import MCMC, _MCMC_debug
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import VariationalParameterBlock, Wavefunction_data, evaluate_ln_wavefunction

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# (trexio_file, with_1b_jastrow, with_2b_jastrow, with_3b_jastrow, with_nn_jastrow)
param_grid = [
    ("H2_ae_ccpvdz_cart.h5", True, True, True, False),
    ("H2_ae_ccpvdz_cart.h5", True, True, True, True),
    ("H_ae_ccpvdz_cart.h5", True, False, False, False),
    ("Li_ae_ccpvdz_cart.h5", False, False, False, False),
    # Open-shell (n_up=2, n_dn=1): exercises the J2 num_up>1 dense pair path
    # under force evaluation (de_L/dr second-order AD). NN-off variant added
    # alongside the NN-on variant to isolate Jastrow-2b regressions.
    ("Li_ae_ccpvdz_cart.h5", True, True, True, False),
    ("Li_ae_ccpvdz_cart.h5", True, True, True, True),
    ("H2_ecp_ccpvtz.h5", True, True, True, True),
    ("N_ae_ccpvdz_cart.h5", False, False, False, False),
    # n_up=4, n_dn=3 with J2 only: covers J2 dense pair path on a larger
    # open-shell system (no J3/NN) to keep regression detection narrow.
    ("N_ae_ccpvdz_cart.h5", True, True, False, False),
]


@pytest.mark.parametrize("trexio_file,with_1b_jastrow,with_2b_jastrow,with_3b_jastrow,with_nn_jastrow", param_grid)
def test_jqmc_mcmc(trexio_file, with_1b_jastrow, with_2b_jastrow, with_3b_jastrow, with_nn_jastrow):
    """Test comparison with MCMC debug and MCMC production implementations."""
    # e_L / w_L cross ao_eval/jastrow_eval/det_eval/coulomb/wf_kinetic zones; the
    # achievable debug-vs-jax agreement is bounded by the weakest (fp32 in mixed).
    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
        "strict",
    )
    (
        structure_data,
        _,
        mos_data,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file), store_tuple=True
    )

    jastrow_onebody_data = None
    if with_1b_jastrow:
        jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=tuple([0] * len(structure_data.atomic_numbers)),
            jastrow_1b_type="exp",
        )

    jastrow_twobody_data = None
    if with_2b_jastrow:
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp")

    jastrow_threebody_data = None
    if with_3b_jastrow:
        jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=mos_data, random_init=True, random_scale=1.0e-3, seed=123
        )

    jastrow_nn_data = None
    if with_nn_jastrow:
        jastrow_nn_data = Jastrow_NN_data.init_from_structure(
            structure_data=structure_data, hidden_dim=2, num_layers=1, num_rbf=2, cutoff=5.0
        )

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
        jastrow_nn_data=jastrow_nn_data,
    )

    jastrow_data.sanity_check()

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    wavefunction_data.sanity_check()

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    hamiltonian_data.sanity_check()

    num_walkers = 2
    num_mcmc_steps = 50
    mcmc_seed = 34356
    Dt = 2.0
    epsilon_AS = 1.0e-6

    # run VMC single-shot
    mcmc_debug = _MCMC_debug(
        hamiltonian_data=hamiltonian_data,
        Dt=Dt,
        mcmc_seed=mcmc_seed,
        epsilon_AS=epsilon_AS,
        num_walkers=num_walkers,
        comput_position_deriv=True,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=True,
    )
    mcmc_debug.run(num_mcmc_steps=num_mcmc_steps)

    mcmc_jax = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=Dt,
        mcmc_seed=mcmc_seed,
        epsilon_AS=epsilon_AS,
        num_walkers=num_walkers,
        comput_position_deriv=True,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=True,
    )
    mcmc_jax.run(num_mcmc_steps=num_mcmc_steps)

    # w_L
    w_L_debug = mcmc_debug.w_L
    w_L_jax = mcmc_jax.w_L
    assert not np.any(np.isnan(np.asarray(w_L_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(w_L_jax))), "NaN detected in second argument"
    np.testing.assert_allclose(w_L_debug, w_L_jax, atol=atol, rtol=rtol)

    # e_L
    e_L_debug = mcmc_debug.e_L
    e_L_jax = mcmc_jax.e_L
    assert not np.any(np.isnan(np.asarray(e_L_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(e_L_jax))), "NaN detected in second argument"
    np.testing.assert_allclose(e_L_debug, e_L_jax, atol=atol, rtol=rtol)

    # e_L2
    e_L2_debug = mcmc_debug.e_L2
    e_L2_jax = mcmc_jax.e_L2
    assert not np.any(np.isnan(np.asarray(e_L2_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(e_L2_jax))), "NaN detected in second argument"
    np.testing.assert_allclose(e_L2_debug, e_L2_jax, atol=atol, rtol=rtol)

    # E
    E_debug, E_err_debug, Var_debug, Var_err_debug = mcmc_debug.get_E(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    E_jax, E_err_jax, Var_jax, Var_err_jax = mcmc_jax.get_E(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    assert not np.any(np.isnan(E_debug)), f"E_debug contains NaN: {E_debug}"
    assert not np.any(np.isnan(E_jax)), f"E_jax contains NaN: {E_jax}"
    assert not np.any(np.isnan(E_err_debug)), f"E_err_debug contains NaN: {E_err_debug}"
    assert not np.any(np.isnan(E_err_jax)), f"E_err_jax contains NaN: {E_err_jax}"
    assert not np.any(np.isnan(Var_debug)), f"Var_debug contains NaN: {Var_debug}"
    assert not np.any(np.isnan(Var_jax)), f"Var_jax contains NaN: {Var_jax}"
    assert not np.any(np.isnan(Var_err_debug)), f"Var_err_debug contains NaN: {Var_err_debug}"
    assert not np.any(np.isnan(Var_err_jax)), f"Var_err_jax contains NaN: {Var_err_jax}"
    assert not np.any(np.isnan(np.asarray(E_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(E_jax))), "NaN detected in second argument"
    np.testing.assert_allclose(E_debug, E_jax, atol=atol, rtol=rtol)
    assert not np.any(np.isnan(np.asarray(E_err_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(E_err_jax))), "NaN detected in second argument"
    np.testing.assert_allclose(E_err_debug, E_err_jax, atol=atol, rtol=rtol)
    assert not np.any(np.isnan(np.asarray(Var_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(Var_jax))), "NaN detected in second argument"
    np.testing.assert_allclose(Var_debug, Var_jax, atol=atol, rtol=rtol)
    assert not np.any(np.isnan(np.asarray(Var_err_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(Var_err_jax))), "NaN detected in second argument"
    np.testing.assert_allclose(Var_err_debug, Var_err_jax, atol=atol, rtol=rtol)

    # aF
    force_mean_debug, force_std_debug = mcmc_debug.get_aF(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    force_mean_jax, force_std_jax = mcmc_jax.get_aF(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    assert not np.any(np.isnan(force_mean_debug)), f"force_mean_debug contains NaN: {force_mean_debug}"
    assert not np.any(np.isnan(force_mean_jax)), f"force_mean_jax contains NaN: {force_mean_jax}"
    assert not np.any(np.isnan(force_std_debug)), f"force_std_debug contains NaN: {force_std_debug}"
    assert not np.any(np.isnan(force_std_jax)), f"force_std_jax contains NaN: {force_std_jax}"
    assert not np.any(np.isnan(np.asarray(force_mean_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(force_mean_jax))), "NaN detected in second argument"
    np.testing.assert_allclose(force_mean_debug, force_mean_jax, atol=atol, rtol=rtol)
    assert not np.any(np.isnan(np.asarray(force_std_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(force_std_jax))), "NaN detected in second argument"
    np.testing.assert_allclose(force_std_debug, force_std_jax, atol=atol, rtol=rtol)

    jax.clear_caches()


@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
def test_jqmc_vmc(trexio_file, monkeypatch):
    """Test if parameters are correctly updated/hold."""
    (
        structure_data,
        aos_data,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file), store_tuple=True
    )

    jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=tuple([0] * len(structure_data.atomic_numbers)),
        jastrow_1b_type="pade",
    )
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade")
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)
    jastrow_nn_data = Jastrow_NN_data.init_from_structure(
        structure_data=structure_data, hidden_dim=2, num_layers=1, num_rbf=2, cutoff=5.0
    )

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
        jastrow_nn_data=jastrow_nn_data,
    )

    jastrow_data.sanity_check()

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    wavefunction_data.sanity_check()

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    hamiltonian_data.sanity_check()

    num_walkers = 2
    num_opt_steps = 1
    num_mcmc_steps = 50
    mcmc_seed = 34356
    Dt = 2.0
    epsilon_AS = 1.0e-6

    # Prepare deterministic fake parameters that respect the shapes of the real wavefunction components.
    wf_data = hamiltonian_data.wavefunction_data
    base_params = {}
    if wf_data.jastrow_data.jastrow_one_body_data is not None:
        base_params["j1_param"] = np.ones_like(np.array(wf_data.jastrow_data.jastrow_one_body_data.jastrow_1b_param))
    if wf_data.jastrow_data.jastrow_two_body_data is not None:
        base_params["j2_param"] = np.ones_like(np.array(wf_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param))
    if wf_data.jastrow_data.jastrow_three_body_data is not None:
        base_params["j3_matrix"] = np.ones_like(np.array(wf_data.jastrow_data.jastrow_three_body_data.j_matrix))
    if wf_data.jastrow_data.jastrow_nn_data is not None and wf_data.jastrow_data.jastrow_nn_data.params is not None:
        flat_nn = np.array(wf_data.jastrow_data.jastrow_nn_data.flatten_fn(wf_data.jastrow_data.jastrow_nn_data.params))
        base_params["jastrow_nn_params"] = np.ones_like(flat_nn)
    # Provide a lambda block even if the geminal lacks it, so we can still exercise the flag logic.
    if wf_data.geminal_data.lambda_matrix is not None:
        base_params["lambda_matrix"] = np.ones_like(np.array(wf_data.geminal_data.lambda_matrix))
    else:
        base_params["lambda_matrix"] = np.array([[2.0, -2.0], [3.0, -3.0]], dtype=float)
    # AO basis blocks for J3 and Geminal.
    if wf_data.jastrow_data.jastrow_three_body_data is not None:
        base_params["j3_basis_exp"] = np.ones_like(np.array(wf_data.jastrow_data.jastrow_three_body_data.ao_exponents))
        base_params["j3_basis_coeff"] = np.ones_like(np.array(wf_data.jastrow_data.jastrow_three_body_data.ao_coefficients))
    if wf_data.geminal_data is not None:
        base_params["lambda_basis_exp"] = np.concatenate(
            [
                np.ones_like(np.array(wf_data.geminal_data.ao_exponents_up)),
                np.ones_like(np.array(wf_data.geminal_data.ao_exponents_dn)),
            ]
        )
        base_params["lambda_basis_coeff"] = np.concatenate(
            [
                np.ones_like(np.array(wf_data.geminal_data.ao_coefficients_up)),
                np.ones_like(np.array(wf_data.geminal_data.ao_coefficients_dn)),
            ]
        )

    # Registry keyed by wavefunction id to hold mutable parameter snapshots.
    params_registry: dict[int, dict[str, np.ndarray]] = {}

    def register_params(wf, params):
        """Store a mutable parameter snapshot keyed by a wavefunction object's id."""
        params_registry[id(wf)] = params

    def lookup_params(wf):
        """Retrieve the mutable parameter snapshot for a given wavefunction object."""
        return params_registry[id(wf)]

    def fake_get_variational_blocks(
        self,
        opt_J1_param=True,
        opt_J2_param=True,
        opt_J3_param=True,
        opt_JNN_param=True,
        opt_lambda_param=False,
        opt_J3_basis_exp=False,
        opt_J3_basis_coeff=False,
        opt_lambda_basis_exp=False,
        opt_lambda_basis_coeff=False,
    ):
        """Return deterministic VariationalParameterBlock list honoring the optimization flags.

        Uses the per-wavefunction registry to pull the current parameter arrays; avoids touching
        real TREXIO-driven parameters so the test stays fast and deterministic.
        """
        blocks = []
        pos = lookup_params(self)
        if opt_J1_param and "j1_param" in pos:
            arr = pos["j1_param"]
            blocks.append(VariationalParameterBlock(name="j1_param", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_J2_param and "j2_param" in pos:
            arr = pos["j2_param"]
            blocks.append(VariationalParameterBlock(name="j2_param", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_J3_param and "j3_matrix" in pos:
            arr = pos["j3_matrix"]
            blocks.append(VariationalParameterBlock(name="j3_matrix", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_J3_basis_exp and "j3_basis_exp" in pos:
            arr = pos["j3_basis_exp"]
            blocks.append(VariationalParameterBlock(name="j3_basis_exp", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_J3_basis_coeff and "j3_basis_coeff" in pos:
            arr = pos["j3_basis_coeff"]
            blocks.append(VariationalParameterBlock(name="j3_basis_coeff", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_JNN_param and "jastrow_nn_params" in pos:
            arr = pos["jastrow_nn_params"]
            blocks.append(VariationalParameterBlock(name="jastrow_nn_params", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_lambda_param and "lambda_matrix" in pos:
            arr = pos["lambda_matrix"]
            blocks.append(VariationalParameterBlock(name="lambda_matrix", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_lambda_basis_exp and "lambda_basis_exp" in pos:
            arr = pos["lambda_basis_exp"]
            blocks.append(VariationalParameterBlock(name="lambda_basis_exp", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_lambda_basis_coeff and "lambda_basis_coeff" in pos:
            arr = pos["lambda_basis_coeff"]
            blocks.append(VariationalParameterBlock(name="lambda_basis_coeff", values=arr, shape=arr.shape, size=int(arr.size)))
        return blocks

    def fake_apply_block_updates(self, blocks, thetas, learning_rate):
        """Apply additive updates to the registry-stored parameters, mirroring Wavefunction_data.apply_block_updates."""
        params = lookup_params(self)
        idx = 0
        for block in blocks:
            blk_slice = thetas[idx : idx + block.size]
            idx += block.size
            if blk_slice.size == 0:
                continue
            delta = blk_slice.reshape(block.shape)
            params[block.name] = params[block.name] + learning_rate * delta
        return self

    def fake_run(self, num_mcmc_steps: int = 0, max_time=None):
        """No-op MCMC run to skip sampling in the unit test."""
        return

    def fake_get_dln_WF(
        self,
        blocks,
        num_mcmc_warmup_steps=0,
        chosen_param_index=None,
        lambda_projectors=None,
        num_orb_projection=None,
    ):
        """Return a dummy zero O_matrix of shape (1, num_walkers, total_param_size)."""
        total = sum(block.size for block in blocks)
        return np.zeros((1, self.num_walkers, total), dtype=float)

    def fake_get_E(self, num_mcmc_warmup_steps: int = 0, num_mcmc_bin_blocks: int = 1):
        """Return dummy energy tuple so optimization can proceed without real computation."""
        return (0.0, 0.0, 0.0, 0.0)

    def fake_get_gF(
        self,
        num_mcmc_warmup_steps,
        num_mcmc_bin_blocks,
        blocks,
        lambda_projectors=None,
        num_orb_projection=None,
        chosen_param_index=None,
    ):
        """Return unit generalized forces with std sized to flattened blocks for deterministic updates."""
        total = sum(block.size for block in blocks)
        f = np.ones(total, dtype=float)
        f_std = np.ones(total, dtype=float)
        return f, f_std

    # Monkeypatch class methods (restored after test) to avoid assigning to frozen instances.
    monkeypatch.setattr(Wavefunction_data, "get_variational_blocks", fake_get_variational_blocks, raising=False)
    monkeypatch.setattr(Wavefunction_data, "apply_block_updates", fake_apply_block_updates, raising=False)
    monkeypatch.setattr(MCMC, "run", fake_run, raising=False)
    monkeypatch.setattr(MCMC, "get_E", fake_get_E, raising=False)
    monkeypatch.setattr(MCMC, "get_gF", fake_get_gF, raising=False)
    monkeypatch.setattr(MCMC, "get_dln_WF", fake_get_dln_WF, raising=False)
    # Provide dummy w_L / e_L so the SR path has consistent sample data when fake_run records nothing.
    monkeypatch.setattr(MCMC, "w_L", property(lambda self: np.ones((1, self.num_walkers))), raising=False)
    monkeypatch.setattr(MCMC, "e_L", property(lambda self: np.zeros((1, self.num_walkers))), raising=False)

    def make_mcmc_with_patches(mcmc_instance: MCMC):
        """Clone base_params for a given MCMC instance and register them for the monkeypatched helpers."""
        current_params = {k: v.copy() for k, v in base_params.items()}
        register_params(mcmc_instance.hamiltonian_data.wavefunction_data, current_params)
        return mcmc_instance, current_params

    cases = [
        {
            "name": "j1_only",
            "flags": dict(
                opt_J1_param=True, opt_J2_param=False, opt_J3_param=False, opt_JNN_param=False, opt_lambda_param=False
            ),
            "expect_change": {
                "j1_param": True,
                "j2_param": False,
                "j3_matrix": False,
                "jastrow_nn_params": False,
                "lambda_matrix": False,
            },
        },
        {
            "name": "nn_and_lambda",
            "flags": dict(
                opt_J1_param=False, opt_J2_param=False, opt_J3_param=False, opt_JNN_param=True, opt_lambda_param=True
            ),
            "expect_change": {
                "j1_param": False,
                "j2_param": False,
                "j3_matrix": False,
                "jastrow_nn_params": True,
                "lambda_matrix": True,
            },
        },
        {
            "name": "all_on",
            "flags": dict(opt_J1_param=True, opt_J2_param=True, opt_J3_param=True, opt_JNN_param=True, opt_lambda_param=True),
            "expect_change": {
                "j1_param": True,
                "j2_param": True,
                "j3_matrix": True,
                "jastrow_nn_params": True,
                "lambda_matrix": True,
            },
        },
        # -- AO basis optimization cases --
        {
            "name": "j3_basis_exp_only",
            "flags": dict(
                opt_J1_param=False,
                opt_J2_param=False,
                opt_J3_param=False,
                opt_JNN_param=False,
                opt_lambda_param=False,
                opt_J3_basis_exp=True,
                opt_J3_basis_coeff=False,
            ),
            "expect_change": {
                "j1_param": False,
                "j2_param": False,
                "j3_matrix": False,
                "jastrow_nn_params": False,
                "lambda_matrix": False,
                "j3_basis_exp": True,
                "j3_basis_coeff": False,
                "lambda_basis_exp": False,
                "lambda_basis_coeff": False,
            },
        },
        {
            "name": "j3_basis_both",
            "flags": dict(
                opt_J1_param=False,
                opt_J2_param=False,
                opt_J3_param=True,
                opt_JNN_param=False,
                opt_lambda_param=False,
                opt_J3_basis_exp=True,
                opt_J3_basis_coeff=True,
            ),
            "expect_change": {
                "j1_param": False,
                "j2_param": False,
                "j3_matrix": True,
                "jastrow_nn_params": False,
                "lambda_matrix": False,
                "j3_basis_exp": True,
                "j3_basis_coeff": True,
                "lambda_basis_exp": False,
                "lambda_basis_coeff": False,
            },
        },
        {
            "name": "lambda_basis_exp_only",
            "flags": dict(
                opt_J1_param=False,
                opt_J2_param=False,
                opt_J3_param=False,
                opt_JNN_param=False,
                opt_lambda_param=False,
                opt_lambda_basis_exp=True,
                opt_lambda_basis_coeff=False,
            ),
            "expect_change": {
                "j1_param": False,
                "j2_param": False,
                "j3_matrix": False,
                "jastrow_nn_params": False,
                "lambda_matrix": False,
                "j3_basis_exp": False,
                "j3_basis_coeff": False,
                "lambda_basis_exp": True,
                "lambda_basis_coeff": False,
            },
        },
        {
            "name": "all_basis_on",
            "flags": dict(
                opt_J1_param=False,
                opt_J2_param=False,
                opt_J3_param=False,
                opt_JNN_param=False,
                opt_lambda_param=False,
                opt_J3_basis_exp=True,
                opt_J3_basis_coeff=True,
                opt_lambda_basis_exp=True,
                opt_lambda_basis_coeff=True,
            ),
            "expect_change": {
                "j1_param": False,
                "j2_param": False,
                "j3_matrix": False,
                "jastrow_nn_params": False,
                "lambda_matrix": False,
                "j3_basis_exp": True,
                "j3_basis_coeff": True,
                "lambda_basis_exp": True,
                "lambda_basis_coeff": True,
            },
        },
    ]

    for case in cases:
        mcmc_case = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=Dt,
            mcmc_seed=mcmc_seed,
            epsilon_AS=epsilon_AS,
            num_walkers=num_walkers,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )

        mcmc_patched, current_params = make_mcmc_with_patches(mcmc_case)

        before = {k: v.copy() for k, v in current_params.items()}

        mcmc_patched.run_optimize(
            num_mcmc_steps=num_mcmc_steps,
            num_opt_steps=num_opt_steps,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=1,
            optimizer_kwargs={"method": "sgd", "learning_rate": 1.0},
            **case["flags"],
        )

        for name, should_change in case["expect_change"].items():
            if should_change:
                assert not np.array_equal(before[name], current_params[name]), f"{case['name']}: expected {name} to change"
            else:
                np.testing.assert_array_equal(
                    before[name], current_params[name], err_msg=f"{case['name']}: expected {name} unchanged"
                )

        jax.clear_caches()

    # -- use_lm (LM/aSR) smoke test ------------------------------------------
    # A separate MCMC instance with comput_e_L_param_deriv=True.
    # get_aH is patched at instance level so that no real sampled data are
    # needed; the dummy return values have H_1 < 0 so compute_asr_gamma
    # produces a positive learning-rate scalar.
    mcmc_asr = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=Dt,
        mcmc_seed=mcmc_seed,
        num_walkers=num_walkers,
        epsilon_AS=epsilon_AS,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=True,
    )

    import types as _types

    def _fake_get_aH(
        self_inner,
        g=None,
        blocks=None,
        num_mcmc_warmup_steps=0,
        chosen_param_index=None,
        lambda_projectors=None,
        num_orb_projection=None,
        return_matrices=False,
    ):
        H_0 = -1.0
        H_1 = -0.1
        H_2 = 0.5
        S_2 = -2.0 * H_1
        return H_0, H_1, H_2, S_2

    mcmc_asr.get_aH = _types.MethodType(_fake_get_aH, mcmc_asr)

    make_mcmc_with_patches(mcmc_asr)

    mcmc_asr.run_optimize(
        num_mcmc_steps=num_mcmc_steps,
        num_opt_steps=1,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=1,
        opt_J1_param=False,
        opt_J2_param=False,
        opt_J3_param=False,
        opt_JNN_param=False,
        opt_lambda_param=True,
        optimizer_kwargs={
            "method": "sr",
            "delta": 1e-3,
            "epsilon": 1e-3,
            "use_lm": True,
        },
    )

    jax.clear_caches()


@pytest.mark.parametrize(
    "regime,cg_flag",
    [
        ("wide", False),  # num_params < num_samples, direct solver
        ("wide", True),  # num_params < num_samples, CG solver
        ("tall", False),  # num_params >= num_samples, direct solver
        ("tall", True),  # num_params >= num_samples, CG solver
    ],
)
@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
def test_sr_wide_and_tall_matrix(trexio_file, regime, cg_flag, monkeypatch):
    """SR optimization must run without error for both primal (wide) and dual (tall)
    matrix branches, with both the direct solver and CG solver.

    Wide matrix:  num_params < num_samples_total  (primal formulation)
    Tall matrix:  num_params >= num_samples_total  (dual / push-through identity)

    The test is MPI-aware: num_samples_total = num_mcmc * num_walkers * mpi_size,
    so the regime thresholds are computed accordingly.
    """
    from mpi4py import MPI as _MPI

    mpi_size = _MPI.COMM_WORLD.Get_size()

    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file), store_tuple=True
    )

    # Minimal Jastrow (j1 + j2 only) to keep param count small and controllable.
    jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=tuple([0] * len(structure_data.atomic_numbers)),
        jastrow_1b_type="pade",
    )
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade")

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=None,
        jastrow_nn_data=None,
    )
    jastrow_data.sanity_check()

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    wavefunction_data.sanity_check()

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    num_walkers = 2
    Dt = 2.0
    mcmc_seed = 12345
    epsilon_AS = 1.0e-6

    # Build base parameter registry -- j1, j2, lambda.
    # For the "tall" regime, pad lambda_matrix so that total_params stays
    # larger than num_samples_total even with multiple MPI ranks.
    base_params = {}
    if wavefunction_data.jastrow_data.jastrow_one_body_data is not None:
        base_params["j1_param"] = np.ones_like(np.array(wavefunction_data.jastrow_data.jastrow_one_body_data.jastrow_1b_param))
    if wavefunction_data.jastrow_data.jastrow_two_body_data is not None:
        base_params["j2_param"] = np.ones_like(np.array(wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param))

    num_mcmc = 1  # default for tall
    fixed_param_size = sum(v.size for v in base_params.values())

    if regime == "tall":
        # num_samples_total = num_mcmc * num_walkers * mpi_size
        # We need total_params >= num_samples_total, so pad lambda_matrix.
        min_samples_total = num_mcmc * num_walkers * mpi_size
        lambda_size_needed = max(1, min_samples_total - fixed_param_size + 1)
        base_params["lambda_matrix"] = np.ones(lambda_size_needed, dtype=float)
    else:
        if wavefunction_data.geminal_data.lambda_matrix is not None:
            base_params["lambda_matrix"] = np.ones_like(np.array(wavefunction_data.geminal_data.lambda_matrix))
        else:
            base_params["lambda_matrix"] = np.array([[2.0, -2.0], [3.0, -3.0]], dtype=float)

    total_params = sum(v.size for v in base_params.values())

    if regime == "wide":
        # num_samples_total = num_mcmc * num_walkers * mpi_size > total_params
        num_mcmc = total_params // (num_walkers * mpi_size) + 2

    num_samples_total = num_mcmc * num_walkers * mpi_size

    # Sanity-check the regime we configured (accounting for all MPI ranks).
    if regime == "wide":
        assert total_params < num_samples_total, f"Expected wide (params < samples_total): {total_params} < {num_samples_total}"
    else:
        assert total_params >= num_samples_total, (
            f"Expected tall (params >= samples_total): {total_params} >= {num_samples_total}"
        )

    # Deterministic fake data with non-trivial variance so X and F are non-zero.
    rng = np.random.default_rng(42)
    fake_w_L_data = np.ones((num_mcmc, num_walkers))
    fake_e_L_data = rng.standard_normal((num_mcmc, num_walkers)) * 0.1

    # -- monkeypatch helpers --------------------------------------------------
    params_registry: dict[int, dict[str, np.ndarray]] = {}

    def register_params(wf, params):
        params_registry[id(wf)] = params

    def lookup_params(wf):
        return params_registry[id(wf)]

    def fake_get_variational_blocks(
        self,
        opt_J1_param=True,
        opt_J2_param=True,
        opt_J3_param=True,
        opt_JNN_param=True,
        opt_lambda_param=False,
        opt_J3_basis_exp=False,
        opt_J3_basis_coeff=False,
        opt_lambda_basis_exp=False,
        opt_lambda_basis_coeff=False,
    ):
        blocks = []
        pos = lookup_params(self)
        if opt_J1_param and "j1_param" in pos:
            arr = pos["j1_param"]
            blocks.append(VariationalParameterBlock(name="j1_param", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_J2_param and "j2_param" in pos:
            arr = pos["j2_param"]
            blocks.append(VariationalParameterBlock(name="j2_param", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_lambda_param and "lambda_matrix" in pos:
            arr = pos["lambda_matrix"]
            blocks.append(VariationalParameterBlock(name="lambda_matrix", values=arr, shape=arr.shape, size=int(arr.size)))
        return blocks

    def fake_apply_block_updates(self, blocks, thetas, learning_rate):
        params = lookup_params(self)
        idx = 0
        for block in blocks:
            blk_slice = thetas[idx : idx + block.size]
            idx += block.size
            if blk_slice.size == 0:
                continue
            delta = blk_slice.reshape(block.shape)
            params[block.name] = params[block.name] + learning_rate * delta
        return self

    def fake_run(self, num_mcmc_steps: int = 0, max_time=None):
        return None

    def fake_get_dln_WF(
        self,
        blocks,
        num_mcmc_warmup_steps=0,
        chosen_param_index=None,
        lambda_projectors=None,
        num_orb_projection=None,
    ):
        total = sum(block.size for block in blocks)
        rng_local = np.random.default_rng(123)
        return rng_local.standard_normal((num_mcmc, self.num_walkers, total)) * 0.01

    def fake_get_E(self, num_mcmc_warmup_steps: int = 0, num_mcmc_bin_blocks: int = 1):
        return (0.0, 0.0, 0.0, 0.0)

    def fake_get_gF(
        self,
        num_mcmc_warmup_steps,
        num_mcmc_bin_blocks,
        blocks,
        lambda_projectors=None,
        num_orb_projection=None,
        chosen_param_index=None,
    ):
        total = sum(block.size for block in blocks)
        return np.ones(total, dtype=float), np.ones(total, dtype=float)

    monkeypatch.setattr(Wavefunction_data, "get_variational_blocks", fake_get_variational_blocks, raising=False)
    monkeypatch.setattr(Wavefunction_data, "apply_block_updates", fake_apply_block_updates, raising=False)
    monkeypatch.setattr(MCMC, "run", fake_run, raising=False)
    monkeypatch.setattr(MCMC, "get_E", fake_get_E, raising=False)
    monkeypatch.setattr(MCMC, "get_gF", fake_get_gF, raising=False)
    monkeypatch.setattr(MCMC, "get_dln_WF", fake_get_dln_WF, raising=False)
    monkeypatch.setattr(MCMC, "w_L", property(lambda self: fake_w_L_data), raising=False)
    monkeypatch.setattr(MCMC, "e_L", property(lambda self: fake_e_L_data), raising=False)

    # -- run the test ---------------------------------------------------------
    mcmc = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=Dt,
        mcmc_seed=mcmc_seed,
        epsilon_AS=epsilon_AS,
        num_walkers=num_walkers,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=True,
    )

    current_params = {k: v.copy() for k, v in base_params.items()}
    register_params(mcmc.hamiltonian_data.wavefunction_data, current_params)

    before = {k: v.copy() for k, v in current_params.items()}

    mcmc.run_optimize(
        num_mcmc_steps=num_mcmc,
        num_opt_steps=1,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=1,
        opt_J1_param=True,
        opt_J2_param=True,
        opt_J3_param=False,
        opt_JNN_param=False,
        opt_lambda_param=True,
        optimizer_kwargs={
            "method": "sr",
            "delta": 1.0e-3,
            "epsilon": 1.0e-3,
            "cg_flag": cg_flag,
        },
    )

    # At least one param should have been updated (non-trivial SR solve).
    any_changed = any(not np.array_equal(before[k], current_params[k]) for k in before)
    assert any_changed, f"Expected at least one parameter to change (regime={regime}, cg_flag={cg_flag})"

    jax.clear_caches()


@pytest.mark.parametrize(
    "regime,cg_flag",
    [
        ("wide", False),
        ("wide", True),
        ("tall", False),
        ("tall", True),
    ],
)
@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
def test_sr_device_matches_cpu(trexio_file, regime, cg_flag, monkeypatch):
    """Each of the four SR solve paths (wide/tall x direct/CG) must produce
    the same parameter update on the JAX-native device branch as on the
    legacy NumPy/SciPy/mpi4py CPU branch, given identical inputs.

    Single-process: ``psum`` is trivial and the device path reduces to its
    local computation; this test verifies numerical agreement only.
    Multi-rank validation requires actually running ``mpirun`` and is out of
    scope for the unit suite.
    """
    from mpi4py import MPI as _MPI

    if _MPI.COMM_WORLD.Get_size() != 1:
        pytest.skip("Numerical-agreement test runs single-process only.")

    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file), store_tuple=True
    )

    jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=tuple([0] * len(structure_data.atomic_numbers)),
        jastrow_1b_type="pade",
    )
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade")
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=None,
        jastrow_nn_data=None,
    )
    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    num_walkers = 2
    Dt = 2.0
    mcmc_seed = 12345
    epsilon_AS = 1.0e-6

    # Build a parameter set sized for the requested regime. Single-process,
    # so num_samples_total = num_mcmc * num_walkers.
    base_params: dict[str, np.ndarray] = {
        "j1_param": np.ones_like(np.array(jastrow_onebody_data.jastrow_1b_param)),
        "j2_param": np.ones_like(np.array(jastrow_twobody_data.jastrow_2b_param)),
    }
    fixed_param_size = sum(v.size for v in base_params.values())

    if regime == "tall":
        num_mcmc = 1
        min_samples_total = num_mcmc * num_walkers
        # Pad lambda_matrix so num_params >= num_samples_total.
        lambda_size_needed = max(1, min_samples_total - fixed_param_size + 1)
        base_params["lambda_matrix"] = np.ones(lambda_size_needed, dtype=float)
    else:
        base_params["lambda_matrix"] = np.array([[2.0, -2.0], [3.0, -3.0]], dtype=float)
        total_params_tmp = sum(v.size for v in base_params.values())
        num_mcmc = total_params_tmp // num_walkers + 2

    total_params = sum(v.size for v in base_params.values())
    num_samples_total = num_mcmc * num_walkers
    if regime == "wide":
        assert total_params < num_samples_total, f"wide setup invalid: {total_params} >= {num_samples_total}"
    else:
        assert total_params >= num_samples_total, f"tall setup invalid: {total_params} < {num_samples_total}"

    rng = np.random.default_rng(42)
    fake_w_L_data = np.ones((num_mcmc, num_walkers))
    fake_e_L_data = rng.standard_normal((num_mcmc, num_walkers)) * 0.1

    params_registry: dict[int, dict[str, np.ndarray]] = {}

    def register_params(wf, params):
        params_registry[id(wf)] = params

    def lookup_params(wf):
        return params_registry[id(wf)]

    def fake_get_variational_blocks(
        self,
        opt_J1_param=True,
        opt_J2_param=True,
        opt_J3_param=True,
        opt_JNN_param=True,
        opt_lambda_param=False,
        opt_J3_basis_exp=False,
        opt_J3_basis_coeff=False,
        opt_lambda_basis_exp=False,
        opt_lambda_basis_coeff=False,
    ):
        blocks = []
        pos = lookup_params(self)
        if opt_J1_param and "j1_param" in pos:
            arr = pos["j1_param"]
            blocks.append(VariationalParameterBlock(name="j1_param", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_J2_param and "j2_param" in pos:
            arr = pos["j2_param"]
            blocks.append(VariationalParameterBlock(name="j2_param", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_lambda_param and "lambda_matrix" in pos:
            arr = pos["lambda_matrix"]
            blocks.append(VariationalParameterBlock(name="lambda_matrix", values=arr, shape=arr.shape, size=int(arr.size)))
        return blocks

    def fake_apply_block_updates(self, blocks, thetas, learning_rate):
        params = lookup_params(self)
        idx = 0
        for block in blocks:
            blk_slice = thetas[idx : idx + block.size]
            idx += block.size
            if blk_slice.size == 0:
                continue
            delta = blk_slice.reshape(block.shape)
            params[block.name] = params[block.name] + learning_rate * delta
        return self

    def fake_run(self, num_mcmc_steps: int = 0, max_time=None):
        return None

    # Deterministic O matrix so both runs see identical inputs.
    def fake_get_dln_WF(
        self,
        blocks,
        num_mcmc_warmup_steps=0,
        chosen_param_index=None,
        lambda_projectors=None,
        num_orb_projection=None,
    ):
        total = sum(block.size for block in blocks)
        rng_local = np.random.default_rng(123)
        return rng_local.standard_normal((num_mcmc, self.num_walkers, total)) * 0.01

    def fake_get_E(self, num_mcmc_warmup_steps: int = 0, num_mcmc_bin_blocks: int = 1):
        return (0.0, 0.0, 0.0, 0.0)

    def fake_get_gF(
        self,
        num_mcmc_warmup_steps,
        num_mcmc_bin_blocks,
        blocks,
        lambda_projectors=None,
        num_orb_projection=None,
        chosen_param_index=None,
    ):
        total = sum(block.size for block in blocks)
        return np.ones(total, dtype=float), np.ones(total, dtype=float)

    monkeypatch.setattr(Wavefunction_data, "get_variational_blocks", fake_get_variational_blocks, raising=False)
    monkeypatch.setattr(Wavefunction_data, "apply_block_updates", fake_apply_block_updates, raising=False)
    monkeypatch.setattr(MCMC, "run", fake_run, raising=False)
    monkeypatch.setattr(MCMC, "get_E", fake_get_E, raising=False)
    monkeypatch.setattr(MCMC, "get_gF", fake_get_gF, raising=False)
    monkeypatch.setattr(MCMC, "get_dln_WF", fake_get_dln_WF, raising=False)
    monkeypatch.setattr(MCMC, "w_L", property(lambda self: fake_w_L_data), raising=False)
    monkeypatch.setattr(MCMC, "e_L", property(lambda self: fake_e_L_data), raising=False)

    def run_once(use_device_collectives: bool):
        mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=Dt,
            mcmc_seed=mcmc_seed,
            epsilon_AS=epsilon_AS,
            num_walkers=num_walkers,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        params = {k: v.copy() for k, v in base_params.items()}
        register_params(mcmc.hamiltonian_data.wavefunction_data, params)
        mcmc.run_optimize(
            num_mcmc_steps=num_mcmc,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=1,
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=False,
            opt_JNN_param=False,
            opt_lambda_param=True,
            optimizer_kwargs={
                "method": "sr",
                "delta": 1.0e-3,
                "epsilon": 1.0e-3,
                "cg_flag": cg_flag,
                "cg_max_iter": 200,
                # CG iteration tolerance well below the project consistency
                # tolerance, so the device-vs-CPU difference is dominated by
                # float64 round-off, not CG residual.
                "cg_tol": 1.0e-12,
            },
            use_device_collectives=use_device_collectives,
        )
        return params

    cpu_params = run_once(use_device_collectives=False)
    dev_params = run_once(use_device_collectives=True)

    # Both branches do exactly the same float64 SR arithmetic, just with
    # NumPy/SciPy/mpi4py vs JAX/shard_map; difference is round-off only.
    # Use the project's strict-float64 consistency tolerance.
    for key in cpu_params:
        cpu_v = cpu_params[key]
        dev_v = dev_params[key]
        # Sanity: CPU branch produced a non-trivial update.
        assert not np.array_equal(cpu_v, base_params[key]), f"baseline CPU update is trivial for {key}"
        np.testing.assert_allclose(
            dev_v,
            cpu_v,
            atol=atol_consistency,
            rtol=rtol_consistency,
            err_msg=f"device vs CPU mismatch for {key} (regime={regime}, cg={cg_flag})",
        )

    jax.clear_caches()


@pytest.mark.parametrize(
    "regime,cg_flag",
    [
        ("wide", False),
        ("wide", True),
        ("tall", False),
        ("tall", True),
    ],
)
@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
def test_sr_device_matches_cpu_multirank(trexio_file, regime, cg_flag, monkeypatch):
    """Multi-rank counterpart to ``test_sr_device_matches_cpu``.

    Verifies that under ``mpirun -n N>=2``:

    - The legacy CPU branch (``mpi_comm.Reduce`` / ``Allreduce`` / ``Alltoallv``
      via mpi4py) and
    - The device branch (``jax.lax.psum`` / ``all_gather`` via NCCL on GPU
      or Gloo on CPU, dispatched through ``shard_map``)

    produce numerically equivalent ``theta`` updates for all four SR
    paths (wide/tall x direct/CG).

    Each MPI rank is given *different* fake samples (via a rank-dependent
    seed) so that the cross-rank reduction has actual work to do; if the
    fixture-installed ``jax.distributed.initialize`` were missing, the
    device branch would silently produce per-rank-local results that
    wouldn't agree with the CPU branch's globally-aggregated result.

    Skipped on single-process runs (the single-process variant lives in
    ``test_sr_device_matches_cpu``).
    """
    from mpi4py import MPI as _MPI

    comm = _MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    if mpi_size < 2:
        pytest.skip("Multi-rank agreement test requires at least 2 MPI ranks.")
    if jax.process_count() < 2:
        pytest.skip(
            "Multi-rank agreement test requires jax.distributed to be initialized "
            "(JAX sees only 1 process despite multiple MPI ranks). The conftest "
            "fixture should auto-init under ``mpirun -n N pytest``; check that the "
            "init didn't silently fail (proxy env vars, network sandboxing)."
        )

    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file), store_tuple=True
    )

    jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=tuple([0] * len(structure_data.atomic_numbers)),
        jastrow_1b_type="pade",
    )
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade")
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=None,
        jastrow_nn_data=None,
    )
    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    num_walkers = 2
    Dt = 2.0
    mcmc_seed = 12345
    epsilon_AS = 1.0e-6

    # Build a parameter set sized for the requested regime; use mpi_size in
    # the sample-count budget since the SR system sees all ranks' samples.
    base_params: dict[str, np.ndarray] = {
        "j1_param": np.ones_like(np.array(jastrow_onebody_data.jastrow_1b_param)),
        "j2_param": np.ones_like(np.array(jastrow_twobody_data.jastrow_2b_param)),
    }
    fixed_param_size = sum(v.size for v in base_params.values())

    if regime == "tall":
        num_mcmc = 1
        min_samples_total = num_mcmc * num_walkers * mpi_size
        lambda_size_needed = max(1, min_samples_total - fixed_param_size + 1)
        base_params["lambda_matrix"] = np.ones(lambda_size_needed, dtype=float)
    else:
        base_params["lambda_matrix"] = np.array([[2.0, -2.0], [3.0, -3.0]], dtype=float)
        total_params_tmp = sum(v.size for v in base_params.values())
        # Ensure num_mcmc * num_walkers * mpi_size > total_params_tmp.
        num_mcmc = max(1, total_params_tmp // (num_walkers * mpi_size) + 2)

    total_params = sum(v.size for v in base_params.values())
    num_samples_total = num_mcmc * num_walkers * mpi_size
    if regime == "wide":
        assert total_params < num_samples_total, f"wide setup invalid: {total_params} >= {num_samples_total}"
    else:
        assert total_params >= num_samples_total, f"tall setup invalid: {total_params} < {num_samples_total}"

    # Rank-dependent fake data: each rank sees different samples so the
    # cross-rank reduction is meaningful. Same seeds in both run_once calls
    # so CPU and device branches see identical inputs.
    fake_w_L_data = np.ones((num_mcmc, num_walkers))
    rng = np.random.default_rng(42 + mpi_rank)
    fake_e_L_data = rng.standard_normal((num_mcmc, num_walkers)) * 0.1

    params_holder: dict[str, dict[str, np.ndarray] | None] = {"params": None}

    def register_params(_wf, params):
        params_holder["params"] = params

    def lookup_params(_wf):
        return params_holder["params"]

    def fake_get_variational_blocks(
        self,
        opt_J1_param=True,
        opt_J2_param=True,
        opt_J3_param=True,
        opt_JNN_param=True,
        opt_lambda_param=False,
        opt_J3_basis_exp=False,
        opt_J3_basis_coeff=False,
        opt_lambda_basis_exp=False,
        opt_lambda_basis_coeff=False,
    ):
        blocks = []
        pos = lookup_params(self)
        if opt_J1_param and "j1_param" in pos:
            arr = pos["j1_param"]
            blocks.append(VariationalParameterBlock(name="j1_param", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_J2_param and "j2_param" in pos:
            arr = pos["j2_param"]
            blocks.append(VariationalParameterBlock(name="j2_param", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_lambda_param and "lambda_matrix" in pos:
            arr = pos["lambda_matrix"]
            blocks.append(VariationalParameterBlock(name="lambda_matrix", values=arr, shape=arr.shape, size=int(arr.size)))
        return blocks

    def fake_apply_block_updates(self, blocks, thetas, learning_rate):
        params = lookup_params(self)
        idx = 0
        for block in blocks:
            blk_slice = thetas[idx : idx + block.size]
            idx += block.size
            if blk_slice.size == 0:
                continue
            delta = blk_slice.reshape(block.shape)
            params[block.name] = params[block.name] + learning_rate * delta
        return self

    def fake_run(self, num_mcmc_steps: int = 0, max_time=None):
        return None

    def fake_get_dln_WF(
        self,
        blocks,
        num_mcmc_warmup_steps=0,
        chosen_param_index=None,
        lambda_projectors=None,
        num_orb_projection=None,
    ):
        total = sum(block.size for block in blocks)
        rng_local = np.random.default_rng(123 + mpi_rank)
        return rng_local.standard_normal((num_mcmc, self.num_walkers, total)) * 0.01

    def fake_get_E(self, num_mcmc_warmup_steps: int = 0, num_mcmc_bin_blocks: int = 1):
        return (0.0, 0.0, 0.0, 0.0)

    def fake_get_gF(
        self,
        num_mcmc_warmup_steps,
        num_mcmc_bin_blocks,
        blocks,
        lambda_projectors=None,
        num_orb_projection=None,
        chosen_param_index=None,
    ):
        total = sum(block.size for block in blocks)
        return np.ones(total, dtype=float), np.ones(total, dtype=float)

    monkeypatch.setattr(Wavefunction_data, "get_variational_blocks", fake_get_variational_blocks, raising=False)
    monkeypatch.setattr(Wavefunction_data, "apply_block_updates", fake_apply_block_updates, raising=False)
    monkeypatch.setattr(MCMC, "run", fake_run, raising=False)
    monkeypatch.setattr(MCMC, "get_E", fake_get_E, raising=False)
    monkeypatch.setattr(MCMC, "get_gF", fake_get_gF, raising=False)
    monkeypatch.setattr(MCMC, "get_dln_WF", fake_get_dln_WF, raising=False)
    monkeypatch.setattr(MCMC, "w_L", property(lambda self: fake_w_L_data), raising=False)
    monkeypatch.setattr(MCMC, "e_L", property(lambda self: fake_e_L_data), raising=False)

    def run_once(use_device_collectives: bool):
        mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=Dt,
            mcmc_seed=mcmc_seed,
            epsilon_AS=epsilon_AS,
            num_walkers=num_walkers,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        params = {k: v.copy() for k, v in base_params.items()}
        register_params(mcmc.hamiltonian_data.wavefunction_data, params)
        mcmc.run_optimize(
            num_mcmc_steps=num_mcmc,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=1,
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=False,
            opt_JNN_param=False,
            opt_lambda_param=True,
            optimizer_kwargs={
                "method": "sr",
                "delta": 1.0e-3,
                "epsilon": 1.0e-3,
                "cg_flag": cg_flag,
                "cg_max_iter": 200,
                "cg_tol": 1.0e-14,
            },
            use_device_collectives=use_device_collectives,
        )
        return params

    cpu_params = run_once(use_device_collectives=False)
    dev_params = run_once(use_device_collectives=True)

    # Both branches must agree to consistency tolerance on every rank.
    for key in cpu_params:
        cpu_v = cpu_params[key]
        dev_v = dev_params[key]
        assert not np.array_equal(cpu_v, base_params[key]), f"baseline CPU update is trivial for {key} (rank={mpi_rank})"
        np.testing.assert_allclose(
            dev_v,
            cpu_v,
            atol=atol_consistency,
            rtol=rtol_consistency,
            err_msg=(f"device vs CPU multirank mismatch for {key} (regime={regime}, cg={cg_flag}, rank={mpi_rank}/{mpi_size})"),
        )

    # Sanity: CPU branch's bcast / device branch's psum both replicate theta
    # across ranks, so the wf updates should agree across ranks too.
    rank0_cpu = comm.bcast({k: v.copy() for k, v in cpu_params.items()}, root=0)
    for key in cpu_params:
        np.testing.assert_allclose(
            cpu_params[key],
            rank0_cpu[key],
            atol=atol_consistency,
            rtol=rtol_consistency,
            err_msg=f"CPU branch theta differs across ranks for {key} (rank={mpi_rank})",
        )
    rank0_dev = comm.bcast({k: v.copy() for k, v in dev_params.items()}, root=0)
    for key in dev_params:
        np.testing.assert_allclose(
            dev_params[key],
            rank0_dev[key],
            atol=atol_consistency,
            rtol=rtol_consistency,
            err_msg=f"device branch theta differs across ranks for {key} (rank={mpi_rank})",
        )

    jax.clear_caches()


@pytest.mark.parametrize(
    "lm_subspace_dim,cg_flag,num_mcmc,num_walkers",
    [
        # aSR (gamma scaling): smooth function, strict at any size.
        (0, False, 10, 2),
        (0, True, 10, 2),
        # Subspace LM (size 2 + SR collective = 3 dims): well-conditioned
        # once samples >> 3, so strict at 200 mcmc * 4 walkers = 800 samples.
        (2, False, 200, 4),
        (2, True, 200, 4),
    ],
)
def test_sr_lm_device_matches_cpu(lm_subspace_dim, cg_flag, num_mcmc, num_walkers, monkeypatch):
    """LM / aSR end-to-end optimization with ``use_device_collectives``
    toggled.

    The device branch only replaces the SR direct/CG solve; everything
    downstream (``get_aH``, ``solve_linear_method``, aSR gamma) still runs
    on the CPU/mpi4py path.

    Tested LM modes (cf. ``run_optimize`` ``optimizer_kwargs``):
        - ``lm_subspace_dim = 0``: aSR (gamma from H_0/H_1/H_2/S_2)
        - ``lm_subspace_dim = N`` (positive small): subspace LM (top-N + SR collective)

    What is compared, and why:
        - aSR (``lm_subspace_dim = 0``): gamma scaling is a smooth function
          of the SR direction, so the final wf parameters depend
          continuously on ``theta_SR`` and can be compared at strict
          tolerance.
        - Subspace LM (``lm_subspace_dim != 0``): ``solve_linear_method``
          contains two argmax operations (dgelscut parameter elimination
          and ``argmax(|v_0|^2)`` eigenvector selection) that are
          discontinuous in their inputs -- a round-off-level perturbation
          can flip the selected mode and produce O(1e-3) jumps in the
          downstream output (final wf parameters, ``E_lm``, etc.) even
          though both branches run deterministically. Note in particular
          that ``E_lm = eigvals_lm[argmax(|v_0|^2)]`` is *not* a
          continuous function of ``H_bar``: the ranking by ``|v_0|^2``
          is unrelated to the eigenvalue ordering, so an argmax flip can
          jump ``E_lm`` by the gap between two arbitrary eigenvalues.
          Instead, compare the inputs to ``solve_linear_method``
          (``H_0, f_vec, S, K, B``) -- these depend continuously on
          ``theta_SR``, so they are the right boundary at which to
          verify that the device-branch SR solve agrees with the CPU
          branch. Whatever ``solve_linear_method`` does downstream
          (including any argmax flips) is shared CPU code and not part
          of what this test is meant to cover.

    Single optimization step only: ``num_opt_steps > 1`` would diverge the
    MCMC trajectories once round-off-level wf differences accumulate.
    """
    from mpi4py import MPI as _MPI

    if _MPI.COMM_WORLD.Get_size() != 1:
        pytest.skip("Numerical-agreement test runs single-process only.")

    trexio_file_path = os.path.join(os.path.dirname(__file__), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")

    def build_mcmc():
        (
            structure_data,
            aos_data,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=trexio_file_path, store_tuple=True)

        jastrow_data = Jastrow_data(
            jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
                jastrow_1b_param=1.0,
                structure_data=structure_data,
                core_electrons=tuple([0] * len(structure_data.atomic_numbers)),
                jastrow_1b_type="pade",
            ),
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
                jastrow_2b_param=0.5, jastrow_2b_type="pade"
            ),
            jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data),
        )
        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )
        return MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=2.0,
            mcmc_seed=12345,
            num_walkers=num_walkers,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,  # required by use_lm=True
        )

    # Pristine reference, captured before any monkeypatching so chained
    # spies (one per ``run_once`` call) all delegate to the real solver.
    orig_solve_linear_method = MCMC.solve_linear_method

    def run_once(use_device_collectives: bool):
        lm_inputs: list[dict] = []

        def spy(H_0, f_vec, S_matrix, K_matrix, B_matrix, epsilon):
            lm_inputs.append(
                {
                    "H_0": float(H_0),
                    "f_vec": np.asarray(f_vec).copy(),
                    "S": np.asarray(S_matrix).copy(),
                    "K": np.asarray(K_matrix).copy(),
                    "B": np.asarray(B_matrix).copy(),
                }
            )
            return orig_solve_linear_method(H_0, f_vec, S_matrix, K_matrix, B_matrix, epsilon)

        monkeypatch.setattr(MCMC, "solve_linear_method", staticmethod(spy))

        mcmc = build_mcmc()
        mcmc.run_optimize(
            num_mcmc_steps=num_mcmc,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=1,
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=True,
            opt_lambda_param=True,
            optimizer_kwargs={
                "method": "sr",
                "use_lm": True,
                "lm_subspace_dim": lm_subspace_dim,
                "lm_cond": 1.0e-3,
                "delta": 0.1,
                "epsilon": 1.0e-6,
                "cg_flag": cg_flag,
                # NB: cg_tol=1e-14 (near machine eps) is needed for the
                # LM step to receive bit-comparable theta_SR from both
                # branches. With cg_tol=1e-12, CG can early-terminate at
                # mutually different points along the iteration trajectory,
                # producing O(1e-3) differences that the LM step preserves.
                "cg_max_iter": 2000,
                "cg_tol": 1.0e-14,
            },
            use_device_collectives=use_device_collectives,
        )
        wf = mcmc.hamiltonian_data.wavefunction_data
        wf_params: dict[str, np.ndarray] = {}
        if wf.jastrow_data.jastrow_one_body_data is not None:
            wf_params["j1_param"] = np.asarray(wf.jastrow_data.jastrow_one_body_data.jastrow_1b_param)
        if wf.jastrow_data.jastrow_two_body_data is not None:
            wf_params["j2_param"] = np.asarray(wf.jastrow_data.jastrow_two_body_data.jastrow_2b_param)
        if wf.jastrow_data.jastrow_three_body_data is not None:
            wf_params["j3_matrix"] = np.asarray(wf.jastrow_data.jastrow_three_body_data.j_matrix)
        if wf.geminal_data is not None:
            wf_params["lambda_matrix"] = np.asarray(wf.geminal_data.lambda_matrix)
        return wf_params, lm_inputs

    cpu_params, cpu_lm_inputs = run_once(use_device_collectives=False)
    dev_params, dev_lm_inputs = run_once(use_device_collectives=True)

    if lm_subspace_dim == 0:
        # aSR path: solve_linear_method is not invoked; final wf params are
        # Lipschitz in theta_SR via gamma scaling.
        assert cpu_lm_inputs == [] and dev_lm_inputs == []
        for key in cpu_params:
            np.testing.assert_allclose(
                dev_params[key],
                cpu_params[key],
                atol=atol_consistency,
                rtol=rtol_consistency,
                err_msg=(f"device vs CPU aSR mismatch for {key} (lm_subspace_dim={lm_subspace_dim}, cg_flag={cg_flag})"),
            )
    else:
        # Subspace LM: compare only the inputs to solve_linear_method.
        # These depend continuously on theta_SR, so they are the natural
        # boundary at which the device-branch SR solve can be verified
        # against the CPU branch. Anything past this point (E_lm, c_vec,
        # final wf params) goes through argmax operations inside
        # solve_linear_method and is not safe to compare strictly.
        assert len(cpu_lm_inputs) == len(dev_lm_inputs) > 0, (
            f"solve_linear_method was not invoked (cpu={len(cpu_lm_inputs)}, dev={len(dev_lm_inputs)})"
        )
        for step, (c_in, d_in) in enumerate(zip(cpu_lm_inputs, dev_lm_inputs)):
            for key in ("H_0", "f_vec", "S", "K", "B"):
                np.testing.assert_allclose(
                    d_in[key],
                    c_in[key],
                    atol=atol_consistency,
                    rtol=rtol_consistency,
                    err_msg=(
                        f"device vs CPU LM-input mismatch for {key} at step {step} "
                        f"(lm_subspace_dim={lm_subspace_dim}, cg_flag={cg_flag})"
                    ),
                )

    jax.clear_caches()


@pytest.mark.parametrize("regime", ["wide", "tall"])
@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
def test_sr_cg_warm_start_device_matches_cpu(trexio_file, regime, monkeypatch):
    """Multi-step CG with warm-start: device branch must mirror CPU branch
    after multiple optimization iterations.

    Each iteration the CG solver carries the previous step's solution as the
    initial guess (``sr_cg_warm_start_primal`` for wide, ``sr_cg_warm_start_dual``
    for tall). Both CPU and device branches must persist this state correctly
    so the final wf parameters agree to consistency tolerance.

    To make the warm-start path actually exercise the iteration-to-iteration
    carry, the fake ``O`` matrix is varied per call (using a counter that is
    reset between the two ``run_once`` invocations so both branches see the
    same input sequence).
    """
    from mpi4py import MPI as _MPI

    if _MPI.COMM_WORLD.Get_size() != 1:
        pytest.skip("Numerical-agreement test runs single-process only.")

    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file), store_tuple=True
    )

    jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=tuple([0] * len(structure_data.atomic_numbers)),
        jastrow_1b_type="pade",
    )
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade")
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=None,
        jastrow_nn_data=None,
    )
    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    num_walkers = 2
    Dt = 2.0
    mcmc_seed = 12345
    epsilon_AS = 1.0e-6
    num_opt_steps = 3

    base_params: dict[str, np.ndarray] = {
        "j1_param": np.ones_like(np.array(jastrow_onebody_data.jastrow_1b_param)),
        "j2_param": np.ones_like(np.array(jastrow_twobody_data.jastrow_2b_param)),
    }
    fixed_param_size = sum(v.size for v in base_params.values())

    if regime == "tall":
        num_mcmc = 1
        min_samples_total = num_mcmc * num_walkers
        lambda_size_needed = max(1, min_samples_total - fixed_param_size + 1)
        base_params["lambda_matrix"] = np.ones(lambda_size_needed, dtype=float)
    else:
        base_params["lambda_matrix"] = np.array([[2.0, -2.0], [3.0, -3.0]], dtype=float)
        total_params_tmp = sum(v.size for v in base_params.values())
        num_mcmc = total_params_tmp // num_walkers + 2

    fake_w_L_data = np.ones((num_mcmc, num_walkers))
    rng = np.random.default_rng(42)
    fake_e_L_data = rng.standard_normal((num_mcmc, num_walkers)) * 0.1

    # Single-slot holder for the live params dict. We can't key by ``id(wf)``
    # because ``MCMC.hamiltonian_data`` setter calls ``apply_diff_mask`` which
    # rewraps the wavefunction with a fresh instance every time it's reassigned
    # (i.e. at the end of every optimization iteration). The single-slot
    # approach assumes one MCMC instance is alive at a time inside this test.
    params_holder: dict[str, dict[str, np.ndarray] | None] = {"params": None}

    def register_params(_wf, params):
        params_holder["params"] = params

    def lookup_params(_wf):
        return params_holder["params"]

    # Counter that varies the fake O matrix per get_dln_WF call, so successive
    # SR systems differ and CG warm-start has actual work to do. Reset between
    # the two run_once invocations so both branches see identical input streams.
    call_idx = {"count": 0}

    def fake_get_variational_blocks(
        self,
        opt_J1_param=True,
        opt_J2_param=True,
        opt_J3_param=True,
        opt_JNN_param=True,
        opt_lambda_param=False,
        opt_J3_basis_exp=False,
        opt_J3_basis_coeff=False,
        opt_lambda_basis_exp=False,
        opt_lambda_basis_coeff=False,
    ):
        blocks = []
        pos = lookup_params(self)
        if opt_J1_param and "j1_param" in pos:
            arr = pos["j1_param"]
            blocks.append(VariationalParameterBlock(name="j1_param", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_J2_param and "j2_param" in pos:
            arr = pos["j2_param"]
            blocks.append(VariationalParameterBlock(name="j2_param", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_lambda_param and "lambda_matrix" in pos:
            arr = pos["lambda_matrix"]
            blocks.append(VariationalParameterBlock(name="lambda_matrix", values=arr, shape=arr.shape, size=int(arr.size)))
        return blocks

    def fake_apply_block_updates(self, blocks, thetas, learning_rate):
        params = lookup_params(self)
        idx = 0
        for block in blocks:
            blk_slice = thetas[idx : idx + block.size]
            idx += block.size
            if blk_slice.size == 0:
                continue
            delta = blk_slice.reshape(block.shape)
            params[block.name] = params[block.name] + learning_rate * delta
        return self

    def fake_run(self, num_mcmc_steps: int = 0, max_time=None):
        return None

    def fake_get_dln_WF(
        self,
        blocks,
        num_mcmc_warmup_steps=0,
        chosen_param_index=None,
        lambda_projectors=None,
        num_orb_projection=None,
    ):
        call_idx["count"] += 1
        total = sum(block.size for block in blocks)
        rng_local = np.random.default_rng(123 + call_idx["count"])
        return rng_local.standard_normal((num_mcmc, self.num_walkers, total)) * 0.01

    def fake_get_E(self, num_mcmc_warmup_steps: int = 0, num_mcmc_bin_blocks: int = 1):
        return (0.0, 0.0, 0.0, 0.0)

    def fake_get_gF(
        self,
        num_mcmc_warmup_steps,
        num_mcmc_bin_blocks,
        blocks,
        lambda_projectors=None,
        num_orb_projection=None,
        chosen_param_index=None,
    ):
        total = sum(block.size for block in blocks)
        return np.ones(total, dtype=float), np.ones(total, dtype=float)

    monkeypatch.setattr(Wavefunction_data, "get_variational_blocks", fake_get_variational_blocks, raising=False)
    monkeypatch.setattr(Wavefunction_data, "apply_block_updates", fake_apply_block_updates, raising=False)
    monkeypatch.setattr(MCMC, "run", fake_run, raising=False)
    monkeypatch.setattr(MCMC, "get_E", fake_get_E, raising=False)
    monkeypatch.setattr(MCMC, "get_gF", fake_get_gF, raising=False)
    monkeypatch.setattr(MCMC, "get_dln_WF", fake_get_dln_WF, raising=False)
    monkeypatch.setattr(MCMC, "w_L", property(lambda self: fake_w_L_data), raising=False)
    monkeypatch.setattr(MCMC, "e_L", property(lambda self: fake_e_L_data), raising=False)

    def run_once(use_device_collectives: bool):
        call_idx["count"] = 0  # reset so both branches see the same per-iter inputs
        mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=Dt,
            mcmc_seed=mcmc_seed,
            epsilon_AS=epsilon_AS,
            num_walkers=num_walkers,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        params = {k: v.copy() for k, v in base_params.items()}
        register_params(mcmc.hamiltonian_data.wavefunction_data, params)
        mcmc.run_optimize(
            num_mcmc_steps=num_mcmc,
            num_opt_steps=num_opt_steps,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=1,
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=False,
            opt_JNN_param=False,
            opt_lambda_param=True,
            optimizer_kwargs={
                "method": "sr",
                "delta": 1.0e-3,
                "epsilon": 1.0e-3,
                "cg_flag": True,
                "cg_max_iter": 200,
                "cg_tol": 1.0e-12,
            },
            use_device_collectives=use_device_collectives,
        )
        return params

    cpu_params = run_once(use_device_collectives=False)
    dev_params = run_once(use_device_collectives=True)

    for key in cpu_params:
        cpu_v = cpu_params[key]
        dev_v = dev_params[key]
        # Sanity: 3 iters of warm-started CG produced a non-trivial param trail.
        assert not np.array_equal(cpu_v, base_params[key]), f"baseline CPU update is trivial for {key}"
        np.testing.assert_allclose(
            dev_v,
            cpu_v,
            atol=atol_consistency,
            rtol=rtol_consistency,
            err_msg=f"device vs CPU CG warm-start mismatch for {key} (regime={regime})",
        )

    jax.clear_caches()


@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
def test_opt_with_projected_MOs(trexio_file, monkeypatch):
    """After run_optimize with opt_with_projected_MOs=True the final wavefunction
    (in MO representation) must give the same ln|Psi| as its AO-converted counterpart.

    This validates that the MO->AO->MO round-trip inside each optimisation step
    preserves the wavefunction exactly, even after the lambda matrix has been
    updated in AO space.
    """
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file),
        store_tuple=True,
    )

    # Minimal 2-body Jastrow -- no 3-body/NN to keep the test fast.
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp"),
        jastrow_three_body_data=None,
        jastrow_nn_data=None,
    )

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    mcmc = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=2.0,
        mcmc_seed=42,
        num_walkers=2,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=False,
    )

    # --- monkeypatches: skip sampling, return dummy energy/forces ---------
    def fake_run(self, num_mcmc_steps=0, max_time=None):
        return None

    def fake_get_E(self, num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=1):
        return (0.0, 0.0, 0.0, 0.0)

    def fake_get_gF(
        self,
        num_mcmc_warmup_steps,
        num_mcmc_bin_blocks,
        blocks,
        lambda_projectors=None,
        num_orb_projection=None,
        chosen_param_index=None,
    ):
        # Return non-zero unit forces so the lambda actually changes.
        total = sum(block.size for block in blocks)
        f = np.ones(total, dtype=float)
        f_std = np.ones(total, dtype=float)
        return f, f_std

    monkeypatch.setattr(MCMC, "run", fake_run, raising=False)
    monkeypatch.setattr(MCMC, "get_E", fake_get_E, raising=False)
    monkeypatch.setattr(MCMC, "get_gF", fake_get_gF, raising=False)
    # get_variational_blocks and apply_block_updates are NOT patched so that
    # the real AO lambda update and MO/AO conversion logic is exercised.
    # ----------------------------------------------------------------------

    mcmc.run_optimize(
        num_mcmc_steps=1,
        num_opt_steps=2,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=1,
        opt_J1_param=False,
        opt_J2_param=False,
        opt_J3_param=False,
        opt_JNN_param=False,
        opt_lambda_param=True,
        opt_with_projected_MOs=True,
        optimizer_kwargs={"method": "sgd", "learning_rate": 0.01},
    )

    final_wf = mcmc.hamiltonian_data.wavefunction_data
    final_geminal = final_wf.geminal_data

    # opt_with_projected_MOs must return the geminal in MO representation.
    assert final_geminal.is_mo_representation, (
        "opt_with_projected_MOs=True should leave the geminal in MO representation after run_optimize"
    )

    # The MO representation must be consistent with its AO counterpart.
    geminal_ao = Geminal_data.convert_from_MOs_to_AOs(final_geminal)
    wf_ao = type(final_wf)(geminal_data=geminal_ao, jastrow_data=final_wf.jastrow_data)

    rng = np.random.default_rng(123)
    r_up = rng.uniform(-3.0, 3.0, (final_geminal.num_electron_up, 3))
    r_dn = rng.uniform(-3.0, 3.0, (final_geminal.num_electron_dn, 3))

    ln_psi_mo = float(evaluate_ln_wavefunction(final_wf, r_up, r_dn))
    ln_psi_ao = float(evaluate_ln_wavefunction(wf_ao, r_up, r_dn))

    # ln|Psi| crosses ao_eval/jastrow_eval/det_eval; bound by weakest zone.
    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "det_eval"),
        "strict",
    )
    assert not np.any(np.isnan(np.asarray(ln_psi_mo))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(ln_psi_ao))), "NaN detected in second argument"
    np.testing.assert_allclose(ln_psi_mo, ln_psi_ao, atol=atol, rtol=rtol)

    jax.clear_caches()


# ---------------------------------------------------------------------------
# L3: VMC optimization loop -- symmetry preservation tests
# ---------------------------------------------------------------------------

# Test parameters: (j3_type, lambda_type)
_SYMMETRY_TEST_CASES = [
    # L3-1: baseline, both symmetric, all params
    ("sym", "square_sym"),
    # L3-5: j3 symmetric, lambda non-symmetric -> only j3 preserved
    ("sym", "square_nonsym"),
    # L3-6: j3 non-symmetric, lambda symmetric -> only lambda preserved
    ("nonsym", "square_sym"),
    # L3-7: both non-symmetric -> no symmetrization (no-op)
    ("nonsym", "square_nonsym"),
    # L3-8: j3 symmetric, rectangular lambda with paired-symmetric sub-block
    ("sym", "rect_paired_sym"),
]


@pytest.mark.parametrize("j3_type,lambda_type", _SYMMETRY_TEST_CASES)
def test_vmc_symmetry_preservation(j3_type, lambda_type, monkeypatch):
    """Verify that j3 and lambda symmetry is preserved through the full VMC optimization loop.

    The test uses **real** ``get_variational_blocks`` and ``apply_block_updates`` (not
    monkeypatched), so the ``symmetrize_metric`` wrapper and ``symmetrize_j3`` /
    ``symmetrize_lambda`` are fully exercised through Steps 0, 1, 2, 3 and the final
    parameter apply.
    """
    trexio_file = os.path.join(os.path.dirname(__file__), "trexio_example_files", "H2_ae_ccpvtz_cart.h5")
    (
        structure_data,
        aos_data,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=trexio_file, store_tuple=True)

    # -- Build j3 matrix ------------------------------------------------------
    rng = np.random.RandomState(42)
    n_orb = aos_data._num_orb
    if j3_type == "sym":
        sq = rng.randn(n_orb, n_orb)
        sq = 0.5 * (sq + sq.T)
        last_col = rng.randn(n_orb)
        j3_matrix = np.column_stack([sq, last_col])
    else:
        j3_matrix = rng.randn(n_orb, n_orb + 1)
        j3_matrix[0, 1] = 100.0  # force asymmetry in [:, :-1]

    jastrow_threebody_data = Jastrow_three_body_data(orb_data=aos_data, j_matrix=j3_matrix)
    jastrow_data = Jastrow_data(jastrow_three_body_data=jastrow_threebody_data)

    # -- Build lambda matrix --------------------------------------------------
    n_up_elec = geminal_mo_data.num_electron_up
    n_dn_elec = geminal_mo_data.num_electron_dn
    orb_num = geminal_mo_data.orb_num_up  # = orb_num_dn for MO geminals

    if lambda_type == "square_sym":
        lam = rng.randn(orb_num, orb_num)
        lam = 0.5 * (lam + lam.T)
    elif lambda_type == "square_nonsym":
        if orb_num < 2:
            pytest.skip("Cannot create nonsymmetric square lambda with orb_num < 2")
        lam = rng.randn(orb_num, orb_num)
        lam[0, 1] = 100.0  # force asymmetry
    elif lambda_type == "rect_paired_sym":
        # For rectangular: shape (n_up_orbs, n_dn_orbs + extra_up)
        # We need num_electron_up > num_electron_dn for open-shell.
        # H2 is closed-shell (1up, 1dn), so we simulate rectangular
        # by constructing lambda shape (orb_num, orb_num + 1).
        n_extra = 1
        paired = rng.randn(orb_num, orb_num)
        paired = 0.5 * (paired + paired.T)
        unpaired = rng.randn(orb_num, n_extra)
        lam = np.column_stack([paired, unpaired])
        # Override electron counts for open-shell simulation
        n_up_elec = n_dn_elec + n_extra
    else:
        raise ValueError(f"Unknown lambda_type: {lambda_type}")

    geminal_data = Geminal_data(
        num_electron_up=n_up_elec,
        num_electron_dn=n_dn_elec,
        orb_data_up_spin=geminal_mo_data.orb_data_up_spin,
        orb_data_dn_spin=geminal_mo_data.orb_data_dn_spin,
        lambda_matrix=lam,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    num_walkers = 2

    # -- Mock only the sampling/energy; leave get_variational_blocks and
    #    apply_block_updates real so symmetrize_metric is exercised. --------

    def fake_run(self, num_mcmc_steps=0, max_time=None):
        return None

    def fake_get_E(self, num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=1):
        return (0.0, 0.0, 0.0, 0.0)

    def fake_get_gF(
        self,
        num_mcmc_warmup_steps,
        num_mcmc_bin_blocks,
        blocks,
        lambda_projectors=None,
        num_orb_projection=None,
        chosen_param_index=None,
    ):
        """Return symmetric forces (mirrors real get_gF where O_k is symmetrized at source)."""
        total = sum(block.size for block in blocks)
        rng_gf = np.random.RandomState(99)
        f = rng_gf.randn(total)
        f_std = np.abs(rng_gf.randn(total)) + 0.1  # positive std
        # Apply per-block symmetrization (same as O_matrix symmetrization in get_dln_WF)
        offset = 0
        for block in blocks:
            if block.symmetrize_metric is not None:
                f[offset : offset + block.size] = block.symmetrize_metric(
                    f[offset : offset + block.size].reshape(1, -1)
                ).ravel()
                f_std[offset : offset + block.size] = np.abs(
                    block.symmetrize_metric(f_std[offset : offset + block.size].reshape(1, -1))
                ).ravel()
            offset += block.size
        return f, f_std

    monkeypatch.setattr(MCMC, "run", fake_run, raising=False)
    monkeypatch.setattr(MCMC, "get_E", fake_get_E, raising=False)
    monkeypatch.setattr(MCMC, "get_gF", fake_get_gF, raising=False)
    monkeypatch.setattr(MCMC, "w_L", property(lambda self: np.ones((1, self.num_walkers))), raising=False)
    monkeypatch.setattr(MCMC, "e_L", property(lambda self: np.zeros((1, self.num_walkers))), raising=False)
    # NOTE: get_variational_blocks and apply_block_updates are NOT patched.

    mcmc = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=2.0,
        mcmc_seed=123,
        num_walkers=num_walkers,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=True,
    )

    # Record before
    j3_before = np.asarray(mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix).copy()
    lam_before = np.asarray(mcmc.hamiltonian_data.wavefunction_data.geminal_data.lambda_matrix).copy()

    mcmc.run_optimize(
        num_mcmc_steps=1,
        num_opt_steps=1,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=1,
        opt_J1_param=False,
        opt_J2_param=False,
        opt_J3_param=True,
        opt_JNN_param=False,
        opt_lambda_param=True,
        optimizer_kwargs={"method": "sgd", "learning_rate": 0.01},
    )

    # Extract updated matrices
    j3_after = np.asarray(mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix)
    lam_after = np.asarray(mcmc.hamiltonian_data.wavefunction_data.geminal_data.lambda_matrix)

    # -- Assertions -----------------------------------------------------------
    # j3 / lambda_matrix live in jastrow_eval / det_eval zones; symmetry is a structural
    # property of the matrix itself, so use those zones' tolerances.
    atol, rtol = get_tolerance_min(("jastrow_eval", "det_eval"), "strict")
    if j3_type == "sym":
        np.testing.assert_allclose(
            j3_after[:, :-1],
            j3_after[:, :-1].T,
            atol=atol,
            rtol=rtol,
            err_msg="j3 sub-block symmetry broken after VMC update",
        )
    else:
        # j3 non-symmetric: just verify no crash, no NaN
        assert np.all(np.isfinite(j3_after)), "NaN or Inf in j3 after update"

    if lambda_type == "square_sym":
        np.testing.assert_allclose(
            lam_after,
            lam_after.T,
            atol=atol,
            rtol=rtol,
            err_msg="square lambda symmetry broken after VMC update",
        )
    elif lambda_type == "rect_paired_sym":
        n_paired = orb_num
        np.testing.assert_allclose(
            lam_after[:, :n_paired],
            lam_after[:, :n_paired].T,
            atol=atol,
            rtol=rtol,
            err_msg="rectangular lambda paired sub-block symmetry broken after VMC update",
        )
    else:
        assert np.all(np.isfinite(lam_after)), "NaN or Inf in lambda after update"

    # Verify something actually changed
    j3_changed = not np.array_equal(j3_before, j3_after)
    lam_changed = not np.array_equal(lam_before, lam_after)
    assert j3_changed or lam_changed, "Expected at least one parameter to change"

    jax.clear_caches()


# ---------------------------------------------------------------------------
# End-to-end optimization smoke tests (no monkeypatching)
# ---------------------------------------------------------------------------

# Each case is a dict of opt_* flags passed to run_optimize.
_E2E_OPT_CASES = [
    pytest.param(
        {"opt_J1_param": True, "opt_J2_param": False, "opt_J3_param": False, "opt_lambda_param": False},
        id="j1_only",
    ),
    pytest.param(
        {"opt_J1_param": True, "opt_J2_param": True, "opt_J3_param": True, "opt_lambda_param": True},
        id="j123_lambda",
    ),
    pytest.param(
        {
            "opt_J1_param": False,
            "opt_J2_param": False,
            "opt_J3_param": False,
            "opt_lambda_param": False,
            "opt_J3_basis_exp": True,
            "opt_J3_basis_coeff": True,
        },
        id="j3_basis_only",
    ),
    pytest.param(
        {
            "opt_J1_param": False,
            "opt_J2_param": False,
            "opt_J3_param": False,
            "opt_lambda_param": False,
            "opt_lambda_basis_exp": True,
            "opt_lambda_basis_coeff": True,
        },
        id="lambda_basis_only",
    ),
    pytest.param(
        {
            "opt_J1_param": True,
            "opt_J2_param": True,
            "opt_J3_param": True,
            "opt_lambda_param": True,
            "opt_J3_basis_exp": True,
            "opt_J3_basis_coeff": True,
            "opt_lambda_basis_exp": True,
            "opt_lambda_basis_coeff": True,
        },
        id="all_on",
    ),
]


@pytest.mark.parametrize("opt_flags", _E2E_OPT_CASES)
def test_optimize_e2e_smoke(opt_flags):
    """End-to-end 1-step optimisation without monkeypatching.

    Verifies that the full pipeline (MCMC sampling -> gradient computation ->
    SR solve -> parameter update) runs without NaN/Inf for various flag combos.
    """
    trexio_file = os.path.join(os.path.dirname(__file__), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")
    (
        structure_data,
        aos_data,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=trexio_file, store_tuple=True)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=tuple([0] * len(structure_data.atomic_numbers)),
            jastrow_1b_type="pade",
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade"),
        jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data),
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    num_walkers = 2
    mcmc = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=2.0,
        mcmc_seed=12345,
        num_walkers=num_walkers,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=False,
    )

    mcmc.run_optimize(
        num_mcmc_steps=10,
        num_opt_steps=1,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=1,
        optimizer_kwargs={"method": "sr", "delta": 1e-3, "epsilon": 1e-3},
        **opt_flags,
    )

    # Verify no NaN/Inf in any parameter after optimisation
    wf = mcmc.hamiltonian_data.wavefunction_data
    if wf.jastrow_data.jastrow_one_body_data is not None:
        assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_one_body_data.jastrow_1b_param)))
    if wf.jastrow_data.jastrow_two_body_data is not None:
        assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_two_body_data.jastrow_2b_param)))
    if wf.jastrow_data.jastrow_three_body_data is not None:
        assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_three_body_data.j_matrix)))
        assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_three_body_data.ao_exponents)))
        assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_three_body_data.ao_coefficients)))
    if wf.geminal_data is not None:
        assert np.all(np.isfinite(np.asarray(wf.geminal_data.lambda_matrix)))
        assert np.all(np.isfinite(np.asarray(wf.geminal_data.ao_exponents_up)))
        assert np.all(np.isfinite(np.asarray(wf.geminal_data.ao_exponents_dn)))
        assert np.all(np.isfinite(np.asarray(wf.geminal_data.ao_coefficients_up)))
        assert np.all(np.isfinite(np.asarray(wf.geminal_data.ao_coefficients_dn)))

    jax.clear_caches()


# ---------------------------------------------------------------------------
# solve_linear_method unit tests
# ---------------------------------------------------------------------------


class TestSolveLinearMethod:
    """Unit tests for MCMC.solve_linear_method (no MCMC instance needed)."""

    def test_trivial_2x2(self):
        """p=1: 2x2 eigenvalue problem. E_lm <= H_0."""
        H_0 = -1.0
        f_vec = np.array([0.2])
        S_matrix = np.array([[1.0]])
        K_matrix = np.array([[-0.5]])
        B_matrix = np.array([[-0.1]])
        c_vec, E_lm, v0_sq = MCMC.solve_linear_method(H_0, f_vec, S_matrix, K_matrix, B_matrix, epsilon=1e-10)
        assert c_vec.shape == (1,)
        assert E_lm <= H_0 + 1e-10, f"E_lm={E_lm} should be <= H_0={H_0}"
        assert 0.0 <= v0_sq <= 1.0 + 1e-10

    def test_diagonal_known_solution(self):
        """Diagonal H, S: verify c_vec has correct shape and E_lm is valid."""
        p = 5
        H_0 = -2.0
        f_vec = np.random.default_rng(42).standard_normal(p)
        S_matrix = np.diag(np.linspace(0.1, 1.0, p))
        K_matrix = np.diag(np.linspace(-1.0, -0.1, p))
        B_matrix = np.diag(np.linspace(-0.5, -0.05, p))
        c_vec, E_lm, v0_sq = MCMC.solve_linear_method(H_0, f_vec, S_matrix, K_matrix, B_matrix, epsilon=1e-10)
        assert c_vec.shape == (p,)
        assert np.all(np.isfinite(c_vec))
        assert np.isfinite(E_lm)
        assert 0.0 <= v0_sq <= 1.0 + 1e-10

    def test_epsilon_cutoff(self):
        """S eigenvalues below epsilon are cut; p' < p."""
        p = 4
        H_0 = -1.0
        f_vec = np.ones(p) * 0.1
        S_matrix = np.diag([1.0, 0.5, 1e-8, 1e-10])
        K_matrix = np.eye(p) * (-0.5)
        B_matrix = np.eye(p) * (-0.1)
        c_vec, E_lm, v0_sq = MCMC.solve_linear_method(H_0, f_vec, S_matrix, K_matrix, B_matrix, epsilon=1e-6)
        assert c_vec.shape == (p,)
        assert np.isfinite(E_lm)
        assert 0.0 <= v0_sq <= 1.0 + 1e-10

    def test_all_diag_S_zero(self):
        """All diag(S) = 0 -> dgelscut removes all parameters -> zero update, E_lm == H_0."""
        p = 3
        H_0 = -1.5
        f_vec = np.ones(p) * 0.1
        S_matrix = np.zeros((p, p))
        K_matrix = np.eye(p) * (-0.5)
        B_matrix = np.eye(p) * (-0.1)
        c_vec, E_lm, v0_sq = MCMC.solve_linear_method(H_0, f_vec, S_matrix, K_matrix, B_matrix, epsilon=1e-6)
        np.testing.assert_array_equal(c_vec, np.zeros(p))
        assert E_lm == H_0
        assert v0_sq == 0.0

    def test_v0_max_selection(self):
        """The eigenvector with max |v_0|^2 is selected."""
        # Construct a case where the lowest eigenvector is not the one with max |v_0|^2
        p = 2
        H_0 = 0.0
        f_vec = np.array([0.01, 0.01])
        S_matrix = np.eye(p)
        K_matrix = np.diag([-10.0, -0.1])
        B_matrix = np.zeros((p, p))
        c_vec, E_lm, v0_sq = MCMC.solve_linear_method(H_0, f_vec, S_matrix, K_matrix, B_matrix, epsilon=1e-10)
        assert c_vec.shape == (p,)
        assert np.isfinite(E_lm)
        assert 0.0 <= v0_sq <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# LM end-to-end smoke test
# ---------------------------------------------------------------------------


def test_optimize_lm_e2e_smoke():
    """End-to-end LM optimisation -- verifies no NaN/Inf after 1 step."""
    trexio_file = os.path.join(os.path.dirname(__file__), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")
    (
        structure_data,
        aos_data,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=trexio_file, store_tuple=True)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=tuple([0] * len(structure_data.atomic_numbers)),
            jastrow_1b_type="pade",
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade"),
        jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data),
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    mcmc = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=2.0,
        mcmc_seed=12345,
        num_walkers=2,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=True,
    )

    mcmc.run_optimize(
        num_mcmc_steps=10,
        num_opt_steps=1,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=1,
        opt_J1_param=True,
        opt_J2_param=True,
        opt_J3_param=True,
        opt_lambda_param=True,
        optimizer_kwargs={
            "method": "sr",
            "use_lm": True,
            "delta": 0.1,
            "epsilon": 1e-6,
            "lm_subspace_dim": 0,
        },
    )

    wf = mcmc.hamiltonian_data.wavefunction_data
    if wf.jastrow_data.jastrow_one_body_data is not None:
        assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_one_body_data.jastrow_1b_param)))
    if wf.jastrow_data.jastrow_two_body_data is not None:
        assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_two_body_data.jastrow_2b_param)))
    if wf.jastrow_data.jastrow_three_body_data is not None:
        assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_three_body_data.j_matrix)))
    if wf.geminal_data is not None:
        assert np.all(np.isfinite(np.asarray(wf.geminal_data.lambda_matrix)))

    jax.clear_caches()


# ---------------------------------------------------------------------------
# Debug vs Production: get_aH and solve_linear_method consistency
# ---------------------------------------------------------------------------


def test_get_aH_and_solve_lm_debug_vs_production():
    """Verify that _MCMC_debug.get_aH and solve_linear_method agree with MCMC versions.

    Both aSR (scalar) and LM (matrix) modes are tested.
    """
    trexio_file = os.path.join(os.path.dirname(__file__), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")
    (
        structure_data,
        aos_data,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=trexio_file, store_tuple=True)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=tuple([0] * len(structure_data.atomic_numbers)),
            jastrow_1b_type="pade",
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade"),
        jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data),
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    num_walkers = 2
    num_mcmc_steps = 30
    mcmc_seed = 12345
    Dt = 2.0
    epsilon_AS = 1.0e-6
    warmup = 5

    # Create debug and production MCMC instances with derivative computation enabled
    mcmc_debug = _MCMC_debug(
        hamiltonian_data=hamiltonian_data,
        Dt=Dt,
        mcmc_seed=mcmc_seed,
        epsilon_AS=epsilon_AS,
        num_walkers=num_walkers,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=True,
    )
    mcmc_debug.run(num_mcmc_steps=num_mcmc_steps)

    mcmc_prod = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=Dt,
        mcmc_seed=mcmc_seed,
        epsilon_AS=epsilon_AS,
        num_walkers=num_walkers,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=True,
    )
    mcmc_prod.run(num_mcmc_steps=num_mcmc_steps)

    # Get variational blocks
    blocks = hamiltonian_data.wavefunction_data.get_variational_blocks()

    # H_0/f/S/K/B cross the full e_L path + optimization assembly; bound by weakest zone.
    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "local_energy"),
        "strict",
    )

    # --- Test 1: get_aH in LM mode (return_matrices=True) ---
    H_0_d, f_d, S_d, K_d, B_d = mcmc_debug.get_aH(
        blocks=blocks,
        num_mcmc_warmup_steps=warmup,
        return_matrices=True,
    )
    H_0_p, f_p, S_p, K_p, B_p = mcmc_prod.get_aH(
        blocks=blocks,
        num_mcmc_warmup_steps=warmup,
        return_matrices=True,
    )

    np.testing.assert_allclose(H_0_d, H_0_p, atol=atol, rtol=rtol)
    np.testing.assert_allclose(f_d, f_p, atol=atol, rtol=rtol)
    np.testing.assert_allclose(S_d, S_p, atol=atol, rtol=rtol)
    np.testing.assert_allclose(K_d, K_p, atol=atol, rtol=rtol)
    np.testing.assert_allclose(B_d, B_p, atol=atol, rtol=rtol)

    # --- Test 2: get_aH in aSR mode (return_matrices=False) ---
    # Use a simple direction vector g for the aSR scalar projection test
    K_params = len(f_d)
    g = np.random.default_rng(42).standard_normal(K_params)

    H_0_d2, H_1_d, H_2_d, S_2_d = mcmc_debug.get_aH(
        blocks=blocks,
        g=g,
        num_mcmc_warmup_steps=warmup,
        return_matrices=False,
    )
    H_0_p2, H_1_p, H_2_p, S_2_p = mcmc_prod.get_aH(
        blocks=blocks,
        g=g,
        num_mcmc_warmup_steps=warmup,
        return_matrices=False,
    )

    np.testing.assert_allclose(H_0_d2, H_0_p2, atol=atol, rtol=rtol)
    np.testing.assert_allclose(H_1_d, H_1_p, atol=atol, rtol=rtol)
    np.testing.assert_allclose(H_2_d, H_2_p, atol=atol, rtol=rtol)
    np.testing.assert_allclose(S_2_d, S_2_p, atol=atol, rtol=rtol)

    # --- Test 3: aSR scalars should be consistent with LM matrices ---
    # H_1 = -1/2 g^T f,  S_2 = g^T S g,  H_2 = g^T (K+B) g
    H_1_from_mat = -0.5 * np.dot(g, f_d)
    S_2_from_mat = g @ S_d @ g
    H_2_from_mat = g @ (K_d + B_d) @ g
    np.testing.assert_allclose(H_1_d, H_1_from_mat, atol=atol, rtol=rtol)
    np.testing.assert_allclose(S_2_d, S_2_from_mat, atol=atol, rtol=rtol)
    np.testing.assert_allclose(H_2_d, H_2_from_mat, atol=atol, rtol=rtol)

    # --- Test 4: solve_linear_method with identical inputs ---
    # Use the production matrices for both to verify the two implementations
    # produce identical results when given the exact same input.
    epsilon_lm = 1e-6
    c_debug, E_debug, v0_debug = _MCMC_debug.solve_linear_method(H_0_p, f_p, S_p, K_p, B_p, epsilon_lm)
    c_prod, E_prod, v0_prod = MCMC.solve_linear_method(H_0_p, f_p, S_p, K_p, B_p, epsilon_lm)
    np.testing.assert_allclose(c_debug, c_prod, atol=atol, rtol=rtol)
    np.testing.assert_allclose(E_debug, E_prod, atol=atol, rtol=rtol)
    np.testing.assert_allclose(v0_debug, v0_prod, atol=atol, rtol=rtol)

    jax.clear_caches()


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    for trexio_file, w1b, w2b, w3b, wnn in param_grid:
        test_jqmc_mcmc(
            trexio_file=trexio_file, with_1b_jastrow=w1b, with_2b_jastrow=w2b, with_3b_jastrow=w3b, with_nn_jastrow=wnn
        )
