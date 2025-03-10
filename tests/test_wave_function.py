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

import jax
import numpy as np
from jax import numpy as jnp

from ..jqmc.determinant import compute_geminal_all_elements_api
from ..jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from ..jqmc.trexio_wrapper import read_trexio_file
from ..jqmc.wavefunction import (
    Wavefunction_data,
    _compute_kinetic_energy_all_elements_debug,
    _compute_kinetic_energy_all_elements_jax,
    _compute_kinetic_energy_debug,
    _compute_kinetic_energy_jax,
    compute_discretized_kinetic_energy_api,
    compute_discretized_kinetic_energy_api_fast_update,
    compute_discretized_kinetic_energy_debug,
)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


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
    jastrow_data.sanity_check()

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)
    wavefunction_data.sanity_check()

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

    r_up_carts_np = np.array(
        [
            [0.64878536, -0.83275288, 0.33532629],
            [0.55271273, 0.72310605, 0.93443775],
            [0.66767275, 0.1206456, -0.36521208],
            [-0.93165236, -0.0120386, 0.33003036],
        ]
    )
    r_dn_carts_np = np.array(
        [
            [1.0347816, 1.26162081, 0.42301735],
            [-0.57843435, 1.03651987, -0.55091542],
            [-1.56091964, -0.58952149, -0.99268141],
            [0.61863233, -0.14903326, 0.51962683],
        ]
    )

    r_up_carts_jnp = jnp.array(r_up_carts_np)
    r_dn_carts_jnp = jnp.array(r_dn_carts_np)

    K_elements_up_debug, K_elements_dn_debug = _compute_kinetic_energy_all_elements_debug(
        wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_np, r_dn_carts=r_dn_carts_np
    )
    K_elements_up_jax, K_elements_dn_jax = _compute_kinetic_energy_all_elements_jax(
        wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp
    )

    np.testing.assert_array_almost_equal(K_elements_up_debug, K_elements_up_jax, decimal=5)
    np.testing.assert_array_almost_equal(K_elements_dn_debug, K_elements_dn_jax, decimal=5)


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
    jastrow_data.sanity_check()

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)
    wavefunction_data.sanity_check()

    r_up_carts_np = np.array(
        [
            [0.64878536, -0.83275288, 0.33532629],
            [0.55271273, 0.72310605, 0.93443775],
            [0.66767275, 0.1206456, -0.36521208],
            [-0.93165236, -0.0120386, 0.33003036],
        ]
    )
    r_dn_carts_np = np.array(
        [
            [1.0347816, 1.26162081, 0.42301735],
            [-0.57843435, 1.03651987, -0.55091542],
            [-1.56091964, -0.58952149, -0.99268141],
            [0.61863233, -0.14903326, 0.51962683],
        ]
    )

    r_up_carts_jnp = jnp.array(r_up_carts_np)
    r_dn_carts_jnp = jnp.array(r_dn_carts_np)

    alat = 0.05
    RT = np.eye(3)
    mesh_kinetic_part_r_up_carts_debug, mesh_kinetic_part_r_dn_carts_debug, elements_kinetic_part_debug = (
        compute_discretized_kinetic_energy_debug(
            alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_np, r_dn_carts=r_dn_carts_np
        )
    )

    # elements_kinetic_part_debug_all = np.array(elements_kinetic_part_debug).reshape(-1, 6)
    # print(np.array(elements_kinetic_part_debug))
    # print(elements_kinetic_part_debug_all.shape)
    # print(elements_kinetic_part_debug_all)

    mesh_kinetic_part_r_up_carts_jax, mesh_kinetic_part_r_dn_carts_jax, elements_kinetic_part_jax = (
        compute_discretized_kinetic_energy_api(
            alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp, RT=RT
        )
    )

    A = compute_geminal_all_elements_api(geminal_data=geminal_mo_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp)
    A_old_inv = np.linalg.inv(A)
    (
        mesh_kinetic_part_r_up_carts_jax_fast_update,
        mesh_kinetic_part_r_dn_carts_jax_fast_update,
        elements_kinetic_part_jax_fast_update,
    ) = compute_discretized_kinetic_energy_api_fast_update(
        alat=alat,
        A_old_inv=A_old_inv,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
        RT=RT,
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


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
