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

from ..jqmc.swct import SWCT_data, evaluate_swct_domega_api, evaluate_swct_omega_api
from ..jqmc.trexio_wrapper import read_trexio_file

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

    omega_up_debug = evaluate_swct_omega_api(swct_data=swct_data, r_carts=r_up_carts, debug_flag=True)
    omega_dn_debug = evaluate_swct_omega_api(swct_data=swct_data, r_carts=r_dn_carts, debug_flag=True)
    omega_up_jax = evaluate_swct_omega_api(swct_data=swct_data, r_carts=r_up_carts, debug_flag=False)
    omega_dn_jax = evaluate_swct_omega_api(swct_data=swct_data, r_carts=r_dn_carts, debug_flag=False)

    np.testing.assert_almost_equal(omega_up_debug, omega_up_jax, decimal=6)
    np.testing.assert_almost_equal(omega_dn_debug, omega_dn_jax, decimal=6)

    domega_up_debug = evaluate_swct_domega_api(swct_data=swct_data, r_carts=r_up_carts, debug_flag=True)
    domega_dn_debug = evaluate_swct_domega_api(swct_data=swct_data, r_carts=r_dn_carts, debug_flag=True)
    domega_up_jax = evaluate_swct_domega_api(swct_data=swct_data, r_carts=r_up_carts, debug_flag=False)
    domega_dn_jax = evaluate_swct_domega_api(swct_data=swct_data, r_carts=r_dn_carts, debug_flag=False)

    np.testing.assert_almost_equal(domega_up_debug, domega_up_jax, decimal=6)
    np.testing.assert_almost_equal(domega_dn_debug, domega_dn_jax, decimal=6)

    jax.clear_caches()
