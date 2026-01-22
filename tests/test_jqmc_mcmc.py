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

from jqmc.hamiltonians import Hamiltonian_data  # noqa: E402
from jqmc.jastrow_factor import (  # noqa: E402
    Jastrow_data,
    Jastrow_NN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.jqmc_mcmc import MCMC, _MCMC_debug  # noqa: E402
from jqmc.setting import decimal_debug_vs_production  # noqa: E402
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402
from jqmc.wavefunction import VariationalParameterBlock, Wavefunction_data  # noqa: E402

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

test_trexio_files = ["H2_ecp_ccpvdz_cart.h5", "H2_ae_ccpvdz_cart.h5", "H_ecp_ccpvdz_cart.h5"]

nn_param_grid = [False, True]
jastrow_3b_param_grid = [True]


@pytest.mark.parametrize("trexio_file", test_trexio_files)
@pytest.mark.parametrize("with_3b_jastrow", jastrow_3b_param_grid)
@pytest.mark.parametrize("with_nn_jastrow", nn_param_grid)
def test_jqmc_mcmc(trexio_file, with_nn_jastrow, with_3b_jastrow):
    """Test comparison with MCMC debug and MCMC production implementations."""
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

    jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=tuple([0] * len(structure_data.atomic_numbers)),
    )
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)
    jastrow_threebody_data = None
    if with_3b_jastrow:
        jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=mos_data, random_init=True, random_scale=1.0e-3, seed=123
        )

    jastrow_nn_data = None
    if with_nn_jastrow:
        jastrow_nn_data = Jastrow_NN_data.init_from_structure(
            structure_data=structure_data, hidden_dim=2, num_layers=1, cutoff=5.0
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
        comput_param_deriv=False,
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
        comput_param_deriv=False,
        random_discretized_mesh=True,
    )
    mcmc_jax.run(num_mcmc_steps=num_mcmc_steps)

    # w_L
    w_L_debug = mcmc_debug.w_L
    w_L_jax = mcmc_jax.w_L
    np.testing.assert_array_almost_equal(w_L_debug, w_L_jax, decimal=decimal_debug_vs_production)

    # e_L
    e_L_debug = mcmc_debug.e_L
    e_L_jax = mcmc_jax.e_L
    np.testing.assert_array_almost_equal(e_L_debug, e_L_jax, decimal=decimal_debug_vs_production)

    # e_L2
    e_L2_debug = mcmc_debug.e_L2
    e_L2_jax = mcmc_jax.e_L2
    np.testing.assert_array_almost_equal(e_L2_debug, e_L2_jax, decimal=decimal_debug_vs_production)

    # E
    E_debug, E_err_debug, Var_debug, Var_err_debug = mcmc_debug.get_E(
        num_mcmc_warmup_steps=25,
        num_mcmc_bin_blocks=5,
    )
    E_jax, E_err_jax, Var_jax, Var_err_jax = mcmc_jax.get_E(
        num_mcmc_warmup_steps=25,
        num_mcmc_bin_blocks=5,
    )
    np.testing.assert_array_almost_equal(E_debug, E_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(E_err_debug, E_err_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(Var_debug, Var_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(Var_err_debug, Var_err_jax, decimal=decimal_debug_vs_production)

    # aF
    force_mean_debug, force_std_debug = mcmc_debug.get_aF(
        num_mcmc_warmup_steps=25,
        num_mcmc_bin_blocks=5,
    )
    force_mean_jax, force_std_jax = mcmc_jax.get_aF(
        num_mcmc_warmup_steps=25,
        num_mcmc_bin_blocks=5,
    )
    np.testing.assert_array_almost_equal(force_mean_debug, force_mean_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(force_std_debug, force_std_jax, decimal=decimal_debug_vs_production)

    jax.clear_caches()


def test_jqmc_vmc(monkeypatch):
    """Test if parameters are correctly updated/hold."""
    trexio_file = "H2_ae_ccpvtz_cart.h5"
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
        jastrow_1b_param=1.0, structure_data=structure_data, core_electrons=tuple([0, 0])
    )
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5)
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)
    jastrow_nn_data = Jastrow_NN_data.init_from_structure(structure_data=structure_data, hidden_dim=5, num_layers=2, cutoff=5.0)

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

    # Registry keyed by wavefunction id to hold mutable parameter snapshots.
    params_registry: dict[int, dict[str, np.ndarray]] = {}

    def register_params(wf, params):
        """Store a mutable parameter snapshot keyed by a wavefunction object's id."""
        params_registry[id(wf)] = params

    def lookup_params(wf):
        """Retrieve the mutable parameter snapshot for a given wavefunction object."""
        return params_registry[id(wf)]

    def fake_get_variational_blocks(
        self, opt_J1_param=True, opt_J2_param=True, opt_J3_param=True, opt_JNN_param=True, opt_lambda_param=False
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
        if opt_JNN_param and "jastrow_nn_params" in pos:
            arr = pos["jastrow_nn_params"]
            blocks.append(VariationalParameterBlock(name="jastrow_nn_params", values=arr, shape=arr.shape, size=int(arr.size)))
        if opt_lambda_param and "lambda_matrix" in pos:
            arr = pos["lambda_matrix"]
            blocks.append(VariationalParameterBlock(name="lambda_matrix", values=arr, shape=arr.shape, size=int(arr.size)))
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
        return None

    def fake_get_E(self, num_mcmc_warmup_steps: int = 0, num_mcmc_bin_blocks: int = 1):
        """Return dummy energy tuple so optimization can proceed without real computation."""
        return (0.0, 0.0, 0.0, 0.0)

    def fake_get_gF(self, num_mcmc_warmup_steps, num_mcmc_bin_blocks, chosen_param_index, blocks):
        """Return unit generalized forces (and std) sized to the flattened blocks to drive a deterministic update."""
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
    ]

    for case in cases:
        mcmc_case = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=Dt,
            mcmc_seed=mcmc_seed,
            epsilon_AS=epsilon_AS,
            num_walkers=num_walkers,
            comput_position_deriv=False,
            comput_param_deriv=True,
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


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    for trexio_file in test_trexio_files:
        test_jqmc_mcmc(trexio_file=trexio_file)
