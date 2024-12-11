"""collections of unit tests."""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# This file is part of phonopy.
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
# * Neither the name of the phonopy project nor the names of its
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
from logging import Formatter, StreamHandler, getLogger

import jax
import numpy as np

from ..jqmc.hamiltonians import Hamiltonian_data
from ..jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data, Jastrow_two_body_data
from ..jqmc.swct import SWCT_data, evaluate_swct_domega_api, evaluate_swct_omega_api
from ..jqmc.trexio_wrapper import read_trexio_file
from ..jqmc.vmc import VMC
from ..jqmc.wavefunction import Wavefunction_data

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

log = getLogger("myqmc")
log.setLevel("DEBUG")
stream_handler = StreamHandler()
stream_handler.setLevel("DEBUG")
handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
stream_handler.setFormatter(handler_format)
log.addHandler(stream_handler)


# @pytest.mark.skip
def test_debug_and_jax_SWCT_omega():
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"))

    swct_data = SWCT_data(structure=structure_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -5.0, +5.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    omega_up_debug = evaluate_swct_omega_api(swct_data=swct_data, r_carts=r_up_carts, debug=True)
    omega_dn_debug = evaluate_swct_omega_api(swct_data=swct_data, r_carts=r_dn_carts, debug=True)
    omega_up_jax = evaluate_swct_omega_api(swct_data=swct_data, r_carts=r_up_carts, debug=False)
    omega_dn_jax = evaluate_swct_omega_api(swct_data=swct_data, r_carts=r_dn_carts, debug=False)

    np.testing.assert_almost_equal(omega_up_debug, omega_up_jax, decimal=6)
    np.testing.assert_almost_equal(omega_dn_debug, omega_dn_jax, decimal=6)

    domega_up_debug = evaluate_swct_domega_api(swct_data=swct_data, r_carts=r_up_carts, debug=True)
    domega_dn_debug = evaluate_swct_domega_api(swct_data=swct_data, r_carts=r_dn_carts, debug=True)
    domega_up_jax = evaluate_swct_domega_api(swct_data=swct_data, r_carts=r_up_carts, debug=False)
    domega_dn_jax = evaluate_swct_domega_api(swct_data=swct_data, r_carts=r_dn_carts, debug=False)

    np.testing.assert_almost_equal(domega_up_debug, domega_up_jax, decimal=6)
    np.testing.assert_almost_equal(domega_dn_debug, domega_dn_jax, decimal=6)

    jax.clear_caches()


def test_vmc_force_with_SWCT():
    # """
    # H2 dimer cc-pV5Z with Mitas ccECP (2 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "H2_dimer_trexio.hdf5"))
    # """

    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5)
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_two_body_pade_flag=True,
        jastrow_three_body_data=jastrow_threebody_data,
        jastrow_three_body_flag=False,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    # VMC parameters
    num_mcmc_warmup_steps = 10
    num_mcmc_bin_blocks = 5
    mcmc_seed = 34356

    # run VMC
    vmc = VMC(
        hamiltonian_data=hamiltonian_data,
        Dt_init=2.0,
        mcmc_seed=mcmc_seed,
        comput_position_deriv=True,
        comput_jas_param_deriv=False,
    )
    vmc.run_single_shot(num_mcmc_steps=50)
    vmc.get_e_L(
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )
    force_mean, force_std = vmc.get_atomic_forces(
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )
    # print(force_mean, force_std)

    # See [J. Chem. Phys. 156, 034101 (2022)]
    np.testing.assert_almost_equal(np.array(force_mean[0]), -1.0 * np.array(force_mean[1]), decimal=6)
    np.testing.assert_almost_equal(np.array(force_std[0]), np.array(force_std[1]), decimal=6)
