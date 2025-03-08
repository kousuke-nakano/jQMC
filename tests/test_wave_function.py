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

from ..jqmc.determinant import compute_geminal_all_elements_api
from ..jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from ..jqmc.trexio_wrapper import read_trexio_file
from ..jqmc.wavefunction import (
    Wavefunction_data,
    _compute_kinetic_energy_debug,
    _compute_kinetic_energy_jax,
    _compute_kinetic_energy_all_elements_debug,
    _compute_kinetic_energy_all_elements_jax,
    compute_discretized_kinetic_energy_api,
    compute_discretized_kinetic_energy_api_fast_update,
    compute_discretized_kinetic_energy_debug,
)

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


def test_debug_and_jax_kinetic_energy():
    """Test the kinetic energy computation."""
    (
        _,
        _,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"))

    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=None,
    )

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -5.0, +5.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    K_debug = _compute_kinetic_energy_debug(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
    K_jax = _compute_kinetic_energy_jax(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

    np.testing.assert_almost_equal(K_debug, K_jax, decimal=5)


def test_debug_and_jax_kinetic_energy_all_elements():
    """Test the kinetic energy computation."""
    (
        _,
        _,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"))

    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=None,
    )

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -5.0, +5.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    K_elements_up_debug, K_elements_dn_debug = _compute_kinetic_energy_all_elements_debug(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
    K_elements_up_jax, K_elements_dn_jax = _compute_kinetic_energy_all_elements_jax(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

    np.testing.assert_almost_equal(K_elements_up_debug, K_elements_up_jax, decimal=5)
    np.testing.assert_almost_equal(K_elements_dn_debug, K_elements_dn_jax, decimal=5)

def test_debug_and_jax_discretized_kinetic_energy():
    """Test the discretized kinetic energy computation."""
    (
        _,
        _,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"))

    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=None,
    )

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -5.0, +5.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    alat = 0.05
    RT = np.eye(3)
    mesh_kinetic_part_r_up_carts_debug, mesh_kinetic_part_r_dn_carts_debug, elements_kinetic_part_debug = (
        compute_discretized_kinetic_energy_debug(
            alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )
    )

    mesh_kinetic_part_r_up_carts_jax, mesh_kinetic_part_r_dn_carts_jax, elements_kinetic_part_jax = (
        compute_discretized_kinetic_energy_api(
            alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts, RT=RT
        )
    )

    A = compute_geminal_all_elements_api(geminal_data=geminal_mo_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
    A_old_inv = np.linalg.inv(A)
    (
        mesh_kinetic_part_r_up_carts_jax_fast_update,
        mesh_kinetic_part_r_dn_carts_jax_fast_update,
        elements_kinetic_part_jax_fast_update,
    ) = compute_discretized_kinetic_energy_api_fast_update(
        alat=alat, A_old_inv=A_old_inv, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts, RT=RT
    )

    np.testing.assert_array_almost_equal(mesh_kinetic_part_r_up_carts_jax, mesh_kinetic_part_r_up_carts_debug, decimal=8)
    np.testing.assert_array_almost_equal(mesh_kinetic_part_r_dn_carts_jax, mesh_kinetic_part_r_dn_carts_debug, decimal=8)
    np.testing.assert_array_almost_equal(
        mesh_kinetic_part_r_up_carts_jax_fast_update, mesh_kinetic_part_r_up_carts_debug, decimal=8
    )
    np.testing.assert_array_almost_equal(
        mesh_kinetic_part_r_dn_carts_jax_fast_update, mesh_kinetic_part_r_dn_carts_debug, decimal=8
    )
    np.testing.assert_array_almost_equal(elements_kinetic_part_jax, elements_kinetic_part_debug, decimal=8)
    np.testing.assert_array_almost_equal(elements_kinetic_part_jax_fast_update, elements_kinetic_part_debug, decimal=8)
