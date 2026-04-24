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
from mpi4py import MPI

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance, get_tolerance_min  # noqa: E402
from jqmc.hamiltonians import Hamiltonian_data  # noqa: E402
from jqmc.jastrow_factor import (  # noqa: E402
    Jastrow_data,
    Jastrow_NN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.jqmc_gfmc import GFMC_t, _GFMC_t_debug  # noqa: E402
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402
from jqmc.wavefunction import Wavefunction_data  # noqa: E402

# MPI related
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# (trexio_file, with_1b_jastrow, with_2b_jastrow, with_3b_jastrow, with_nn_jastrow, non_local_move)
param_grid = [
    # ECP cases (non_local_move required)
    ("H_ecp_ccpvdz_cart.h5", False, False, False, False, "tmove"),
    # ("H_ecp_ccpvdz_cart.h5", False, False, False, False, "dltmove"),
    ("H_ecp_ccpvdz_cart.h5", True, True, True, False, "tmove"),
    # ("H_ecp_ccpvdz_cart.h5", True, True, True, False, "dltmove"),
    # ("H_ecp_ccpvdz_cart.h5", True, True, True, True, "tmove"),
    ("H_ecp_ccpvdz_cart.h5", True, True, True, True, "dltmove"),
    # AE cases (no non_local_move)
    ("H2_ae_ccpvdz_cart.h5", True, True, True, True, None),
    # AE open-shell (n_up != n_dn): Li atom (2 up, 1 dn)
    ("Li_ae_ccpvdz_cart.h5", True, True, True, False, None),
]


@pytest.mark.parametrize(
    "trexio_file,with_1b_jastrow,with_2b_jastrow,with_3b_jastrow,with_nn_jastrow,non_local_move", param_grid
)
def test_jqmc_gfmc_t(trexio_file, with_1b_jastrow, with_2b_jastrow, with_3b_jastrow, with_nn_jastrow, non_local_move):
    """LRDMC debug vs production comparison."""
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
            jastrow_1b_type="pade",
        )

    jastrow_twobody_data = None
    if with_2b_jastrow:
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="pade")

    jastrow_threebody_data = None
    if with_3b_jastrow:
        jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=mos_data, random_init=True, random_scale=1.0e-3, seed=123
        )

    jastrow_nn_data = None
    if with_nn_jastrow:
        jastrow_nn_data = Jastrow_NN_data.init_from_structure(
            structure_data=structure_data, hidden_dim=2, num_layers=1, cutoff=5.0, key=jax.random.PRNGKey(0)
        )

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
        jastrow_nn_data=jastrow_nn_data,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    wavefunction_data.sanity_check()

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    hamiltonian_data.sanity_check()

    # GFMC param
    num_mcmc_steps = 60
    num_walkers = 2
    mcmc_seed = 3446
    alat = 0.30
    tau = 0.20
    num_gfmc_collect_steps = 10

    nlm_kwargs = {"non_local_move": non_local_move} if non_local_move is not None else {}

    # run LRDMC single-shots
    gfmc_debug = _GFMC_t_debug(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
        mcmc_seed=mcmc_seed,
        tau=tau,
        alat=alat,
        comput_position_deriv=True,
        **nlm_kwargs,
    )
    gfmc_debug.run(num_mcmc_steps=num_mcmc_steps)

    # run LRDMC single-shots
    gfmc_jax = GFMC_t(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
        mcmc_seed=mcmc_seed,
        tau=tau,
        alat=alat,
        random_discretized_mesh=False,
        comput_position_deriv=True,
        **nlm_kwargs,
    )
    gfmc_jax.run(num_mcmc_steps=num_mcmc_steps)

    # e_L / e_L2 / w_L cross orb_eval/jastrow/geminal/coulomb/kinetic/gfmc zones;
    # the achievable debug-vs-jax agreement is bounded by the weakest (fp32 in mixed).
    atol, rtol = get_tolerance_min(
        ("orb_eval", "jastrow", "geminal", "determinant", "coulomb", "kinetic", "gfmc"),
        "strict",
    )

    if mpi_rank == 0:
        # w_L
        w_L_debug = gfmc_debug.w_L
        w_L_jax = gfmc_jax.w_L
        assert not np.any(np.isnan(np.asarray(w_L_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(w_L_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(w_L_debug, w_L_jax, atol=atol, rtol=rtol)

        # e_L
        e_L_debug = gfmc_debug.e_L
        e_L_jax = gfmc_jax.e_L
        assert not np.any(np.isnan(np.asarray(e_L_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(e_L_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(e_L_debug, e_L_jax, atol=atol, rtol=rtol)

        # e_L2
        e_L2_debug = gfmc_debug.e_L2
        e_L2_jax = gfmc_jax.e_L2
        assert not np.any(np.isnan(np.asarray(e_L2_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(e_L2_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(e_L2_debug, e_L2_jax, atol=atol, rtol=rtol)

    # average_projection_counter
    # Both GFMC_t and _GFMC_t_debug now store local averages per rank.
    apc_debug = gfmc_debug.average_projection_counter
    apc_jax = gfmc_jax.average_projection_counter
    assert not np.any(np.isnan(np.asarray(apc_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(apc_jax))), "NaN detected in second argument"
    np.testing.assert_allclose(apc_debug, apc_jax, atol=atol, rtol=rtol)

    # E
    E_debug, E_err_debug, Var_debug, Var_err_debug = gfmc_debug.get_E(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    E_jax, E_err_jax, Var_jax, Var_err_jax = gfmc_jax.get_E(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
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
    force_mean_debug, force_std_debug = gfmc_debug.get_aF(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    force_mean_jax, force_std_jax = gfmc_jax.get_aF(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    assert not np.any(np.isnan(np.asarray(force_mean_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(force_mean_jax))), "NaN detected in second argument"
    np.testing.assert_allclose(force_mean_debug, force_mean_jax, atol=atol, rtol=rtol)
    assert not np.any(np.isnan(np.asarray(force_std_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(force_std_jax))), "NaN detected in second argument"
    np.testing.assert_allclose(force_std_debug, force_std_jax, atol=atol, rtol=rtol)

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
