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

import jax
import numpy as np
import pytest
from mpi4py import MPI

from ..jqmc.hamiltonians import Hamiltonian_data
from ..jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from ..jqmc.jqmc_gfmc import GFMC_fixed_projection_time, GFMC_fixed_projection_time_debug
from ..jqmc.trexio_wrapper import read_trexio_file
from ..jqmc.wavefunction import Wavefunction_data

# MPI related
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

test_trexio_files = ["H2_ecp_ccpvtz_cart.h5"]


@pytest.mark.parametrize("trexio_file", test_trexio_files)
def test_jqmc_gfmc_fixed_projection_time_tmove(trexio_file):
    """LRDMC with tmove non-local move."""
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

    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=None,
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
    non_local_move = "tmove"

    # run LRDMC single-shots
    gfmc_debug = GFMC_fixed_projection_time_debug(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
        mcmc_seed=mcmc_seed,
        tau=tau,
        alat=alat,
        non_local_move=non_local_move,
    )
    gfmc_debug.run(num_mcmc_steps=num_mcmc_steps)

    # run LRDMC single-shots
    gfmc_jax = GFMC_fixed_projection_time(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
        mcmc_seed=mcmc_seed,
        tau=tau,
        alat=alat,
        random_discretized_mesh=False,
        non_local_move=non_local_move,
    )
    gfmc_jax.run(num_mcmc_steps=num_mcmc_steps)

    if mpi_rank == 0:
        # w_L
        w_L_debug = gfmc_debug.w_L
        w_L_jax = gfmc_jax.w_L
        np.testing.assert_array_almost_equal(w_L_debug, w_L_jax, decimal=6)

        # e_L
        e_L_debug = gfmc_debug.e_L
        e_L_jax = gfmc_jax.e_L
        np.testing.assert_array_almost_equal(e_L_debug, e_L_jax, decimal=6)

        # e_L2
        e_L2_debug = gfmc_debug.e_L2
        e_L2_jax = gfmc_jax.e_L2
        np.testing.assert_array_almost_equal(e_L2_debug, e_L2_jax, decimal=6)

    # E
    E_debug, E_err_debug, Var_debug, Var_err_debug = gfmc_debug.get_E(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    E_jax, E_err_jax, Var_jax, Var_err_jax = gfmc_jax.get_E(
        num_mcmc_warmup_steps=30,
        num_mcmc_bin_blocks=10,
    )
    np.testing.assert_array_almost_equal(E_debug, E_jax, decimal=6)
    np.testing.assert_array_almost_equal(E_err_debug, E_err_jax, decimal=6)
    np.testing.assert_array_almost_equal(Var_debug, Var_jax, decimal=6)
    np.testing.assert_array_almost_equal(Var_err_debug, Var_err_jax, decimal=6)

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
