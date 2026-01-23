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

from jqmc.hamiltonians import Hamiltonian_data  # noqa: E402
from jqmc.jastrow_factor import (  # noqa: E402
    Jastrow_data,
    Jastrow_NN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.jqmc_gfmc import GFMC_n, _GFMC_n_debug  # noqa: E402
from jqmc.setting import decimal_debug_vs_production  # noqa: E402
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402
from jqmc.wavefunction import Wavefunction_data  # noqa: E402

# MPI related
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

test_trexio_files_ecp = ["H2_ecp_ccpvdz_cart.h5", "H_ecp_ccpvdz_cart.h5"]
test_trexio_files_ae = ["H2_ae_ccpvdz_cart.h5"]
nn_param_grid = [False]
jastrow_3b_param_grid = [False]
non_local_move_grid = ["tmove", "dltmove"]


@pytest.mark.parametrize("trexio_file", test_trexio_files_ecp)
@pytest.mark.parametrize("with_3b_jastrow", jastrow_3b_param_grid)
@pytest.mark.parametrize("with_nn_jastrow", nn_param_grid)
@pytest.mark.parametrize("non_local_move", non_local_move_grid)
def test_jqmc_gfmc_n_with_ecp(trexio_file, with_nn_jastrow, with_3b_jastrow, non_local_move):
    """LRDMC with tmove non-local move."""
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
    E_scf = -1.00
    alat = 0.30
    num_mcmc_per_measurement = 10
    num_gfmc_collect_steps = 10

    # run LRDMC single-shots
    gfmc_debug = _GFMC_n_debug(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        num_mcmc_per_measurement=num_mcmc_per_measurement,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
        mcmc_seed=mcmc_seed,
        E_scf=E_scf,
        alat=alat,
        random_discretized_mesh=True,
        comput_position_deriv=True,
        non_local_move=non_local_move,
    )
    gfmc_debug.run(num_mcmc_steps=num_mcmc_steps)

    # run LRDMC single-shots
    gfmc_jax = GFMC_n(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        num_mcmc_per_measurement=num_mcmc_per_measurement,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
        mcmc_seed=mcmc_seed,
        E_scf=E_scf,
        alat=alat,
        random_discretized_mesh=True,
        comput_position_deriv=True,
        non_local_move=non_local_move,
    )
    gfmc_jax.run(num_mcmc_steps=num_mcmc_steps)

    if mpi_rank == 0:
        # w_L
        w_L_debug = gfmc_debug.w_L
        w_L_jax = gfmc_jax.w_L
        np.testing.assert_array_almost_equal(w_L_debug, w_L_jax, decimal=decimal_debug_vs_production)

        # e_L
        e_L_debug = gfmc_debug.e_L
        e_L_jax = gfmc_jax.e_L
        np.testing.assert_array_almost_equal(e_L_debug, e_L_jax, decimal=decimal_debug_vs_production)

        # e_L2
        e_L2_debug = gfmc_debug.e_L2
        e_L2_jax = gfmc_jax.e_L2
        np.testing.assert_array_almost_equal(e_L2_debug, e_L2_jax, decimal=decimal_debug_vs_production)

    # E
    E_debug, E_err_debug, Var_debug, Var_err_debug = gfmc_debug.get_E(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    E_jax, E_err_jax, Var_jax, Var_err_jax = gfmc_jax.get_E(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    np.testing.assert_array_almost_equal(E_debug, E_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(E_err_debug, E_err_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(Var_debug, Var_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(Var_err_debug, Var_err_jax, decimal=decimal_debug_vs_production)

    # aF
    force_mean_debug, force_std_debug = gfmc_debug.get_aF(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    force_mean_jax, force_std_jax = gfmc_jax.get_aF(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    np.testing.assert_array_almost_equal(force_mean_debug, force_mean_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(force_std_debug, force_std_jax, decimal=decimal_debug_vs_production)

    jax.clear_caches()


@pytest.mark.parametrize("trexio_file", test_trexio_files_ae)
@pytest.mark.parametrize("with_3b_jastrow", jastrow_3b_param_grid)
@pytest.mark.parametrize("with_nn_jastrow", nn_param_grid)
def test_jqmc_gfmc_n_with_ae(trexio_file, with_nn_jastrow, with_3b_jastrow):
    """LRDMC all-electron case (no ECP)."""
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
    E_scf = -1.00
    alat = 0.30
    num_mcmc_per_measurement = 10
    num_gfmc_collect_steps = 10

    gfmc_debug = _GFMC_n_debug(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        num_mcmc_per_measurement=num_mcmc_per_measurement,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
        mcmc_seed=mcmc_seed,
        E_scf=E_scf,
        alat=alat,
        random_discretized_mesh=True,
        comput_position_deriv=True,
    )
    gfmc_debug.run(num_mcmc_steps=num_mcmc_steps)

    gfmc_jax = GFMC_n(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        num_mcmc_per_measurement=num_mcmc_per_measurement,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
        mcmc_seed=mcmc_seed,
        E_scf=E_scf,
        alat=alat,
        random_discretized_mesh=True,
        comput_position_deriv=True,
    )
    gfmc_jax.run(num_mcmc_steps=num_mcmc_steps)

    if mpi_rank == 0:
        w_L_debug = gfmc_debug.w_L
        w_L_jax = gfmc_jax.w_L
        np.testing.assert_array_almost_equal(w_L_debug, w_L_jax, decimal=decimal_debug_vs_production)

        e_L_debug = gfmc_debug.e_L
        e_L_jax = gfmc_jax.e_L
        np.testing.assert_array_almost_equal(e_L_debug, e_L_jax, decimal=decimal_debug_vs_production)

        e_L2_debug = gfmc_debug.e_L2
        e_L2_jax = gfmc_jax.e_L2
        np.testing.assert_array_almost_equal(e_L2_debug, e_L2_jax, decimal=decimal_debug_vs_production)

    E_debug, E_err_debug, Var_debug, Var_err_debug = gfmc_debug.get_E(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    E_jax, E_err_jax, Var_jax, Var_err_jax = gfmc_jax.get_E(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    np.testing.assert_array_almost_equal(E_debug, E_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(E_err_debug, E_err_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(Var_debug, Var_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(Var_err_debug, Var_err_jax, decimal=decimal_debug_vs_production)

    force_mean_debug, force_std_debug = gfmc_debug.get_aF(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    force_mean_jax, force_std_jax = gfmc_jax.get_aF(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    np.testing.assert_array_almost_equal(force_mean_debug, force_mean_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(force_std_debug, force_std_jax, decimal=decimal_debug_vs_production)

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
