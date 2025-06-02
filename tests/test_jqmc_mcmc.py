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

from ..jqmc.hamiltonians import Hamiltonian_data
from ..jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from ..jqmc.jqmc_mcmc import MCMC, MCMC_debug
from ..jqmc.trexio_wrapper import read_trexio_file
from ..jqmc.wavefunction import Wavefunction_data

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def test_jqmc_mcmc():
    """Test comparison with MCMC debug and MCMC production implementations."""
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "H2_ecp_ccpvtz_cart.h5"), store_tuple=True
    )

    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=None,
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

    num_walkers = 4
    num_mcmc_steps = 20
    mcmc_seed = 34356
    Dt = 2.0
    epsilon_AS = 1.0e-6

    # run VMC single-shot
    mcmc_debug = MCMC_debug(
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
    np.testing.assert_array_almost_equal(w_L_debug, w_L_jax, decimal=6)

    # e_L
    e_L_debug = mcmc_debug.e_L
    e_L_jax = mcmc_jax.e_L
    np.testing.assert_array_almost_equal(e_L_debug, e_L_jax, decimal=6)

    # e_L2
    e_L2_debug = mcmc_debug.e_L2
    e_L2_jax = mcmc_jax.e_L2
    np.testing.assert_array_almost_equal(e_L2_debug, e_L2_jax, decimal=6)

    # E
    E_debug, E_err_debug, Var_debug, Var_err_debug = mcmc_debug.get_E(
        num_mcmc_warmup_steps=5,
        num_mcmc_bin_blocks=5,
    )
    E_jax, E_err_jax, Var_jax, Var_err_jax = mcmc_jax.get_E(
        num_mcmc_warmup_steps=5,
        num_mcmc_bin_blocks=5,
    )
    np.testing.assert_array_almost_equal(E_debug, E_jax, decimal=6)
    np.testing.assert_array_almost_equal(E_err_debug, E_err_jax, decimal=6)
    np.testing.assert_array_almost_equal(Var_debug, Var_jax, decimal=6)
    np.testing.assert_array_almost_equal(Var_err_debug, Var_err_jax, decimal=6)

    # aF
    force_mean_debug, force_std_debug = mcmc_debug.get_aF(
        num_mcmc_warmup_steps=5,
        num_mcmc_bin_blocks=5,
    )
    force_mean_jax, force_std_jax = mcmc_jax.get_aF(
        num_mcmc_warmup_steps=5,
        num_mcmc_bin_blocks=5,
    )
    np.testing.assert_array_almost_equal(force_mean_debug, force_mean_jax, decimal=6)
    np.testing.assert_array_almost_equal(force_std_debug, force_std_jax, decimal=6)

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
