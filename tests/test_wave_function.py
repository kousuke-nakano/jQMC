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
from jax import numpy as jnp

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.determinant import compute_geminal_all_elements  # noqa: E402
from jqmc.jastrow_factor import (  # noqa: E402
    Jastrow_data,
    Jastrow_NN_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.setting import (  # noqa: E402
    atol_auto_vs_numerical_deriv,
    decimal_auto_vs_analytic_deriv,
    decimal_auto_vs_numerical_deriv,
    decimal_debug_vs_production,
    rtol_auto_vs_numerical_deriv,
)
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402
from jqmc.wavefunction import (  # noqa: E402
    Wavefunction_data,
    _compute_discretized_kinetic_energy_debug,
    _compute_kinetic_energy_all_elements_auto,
    _compute_kinetic_energy_all_elements_debug,
    _compute_kinetic_energy_all_elements_fast_update_debug,
    _compute_kinetic_energy_auto,
    _compute_kinetic_energy_debug,
    compute_discretized_kinetic_energy,
    compute_discretized_kinetic_energy_fast_update,
    compute_kinetic_energy,
    compute_kinetic_energy_all_elements,
    compute_kinetic_energy_all_elements_fast_update,
)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def test_kinetic_energy_analytic_and_numerical():
    """Test the kinetic energy computation."""
    (
        structure_data,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5)
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )
    jastrow_nn_data = Jastrow_NN_data.init_from_structure(structure_data=structure_data, hidden_dim=5, num_layers=2, cutoff=5.0)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
        jastrow_nn_data=jastrow_nn_data,
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
    K_jax = compute_kinetic_energy(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
    np.testing.assert_allclose(
        np.asarray(K_debug),
        np.asarray(K_jax),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )


def test_kinetic_energy_analytic_and_auto():
    """Compare analytic and autodiff kinetic energy implementations."""
    (
        _,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
    )

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -5.0, +5.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    K_analytic = compute_kinetic_energy(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
    K_auto = _compute_kinetic_energy_auto(
        wavefunction_data=wavefunction_data,
        r_up_carts=jnp.asarray(r_up_carts),
        r_dn_carts=jnp.asarray(r_dn_carts),
    )

    np.testing.assert_almost_equal(K_analytic, K_auto, decimal=decimal_auto_vs_analytic_deriv)


def test_debug_and_auto_kinetic_energy_all_elements():
    """Debug vs autodiff kinetic energy per-electron arrays."""
    (
        _,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )
    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
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
    K_elements_up_auto, K_elements_dn_auto = _compute_kinetic_energy_all_elements_auto(
        wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp
    )

    np.testing.assert_array_almost_equal(K_elements_up_debug, K_elements_up_auto, decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_array_almost_equal(K_elements_dn_debug, K_elements_dn_auto, decimal=decimal_auto_vs_numerical_deriv)

    np.testing.assert_allclose(
        np.asarray(K_elements_up_debug),
        np.asarray(K_elements_up_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )

    np.testing.assert_allclose(
        np.asarray(K_elements_dn_debug),
        np.asarray(K_elements_dn_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )


def test_auto_and_analytic_kinetic_energy_all_elements():
    """Autodiff vs analytic kinetic energy per-electron arrays."""
    (
        _,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
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

    K_elements_up_auto, K_elements_dn_auto = _compute_kinetic_energy_all_elements_auto(
        wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp
    )
    K_elements_up_analytic, K_elements_dn_analytic = compute_kinetic_energy_all_elements(
        wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp
    )

    np.testing.assert_array_almost_equal(K_elements_up_auto, K_elements_up_analytic, decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_array_almost_equal(K_elements_dn_auto, K_elements_dn_analytic, decimal=decimal_auto_vs_analytic_deriv)


def test_fast_update_kinetic_energy_all_elements():
    """Fast-update per-electron kinetic energy should match the standard analytic path."""
    (
        _,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"),
        store_tuple=True,
    )

    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
    )

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -3.0, +3.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    r_up_carts_jnp = jnp.asarray(r_up_carts)
    r_dn_carts_jnp = jnp.asarray(r_dn_carts)

    # Standard analytic per-electron kinetic energy
    ke_up_debug, ke_dn_debug = _compute_kinetic_energy_all_elements_fast_update_debug(
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
    )

    # Build geminal inverse explicitly for the fast path
    A = compute_geminal_all_elements(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
    )
    A_inv = jnp.asarray(np.linalg.inv(np.array(A)))

    ke_up_fast, ke_dn_fast = compute_kinetic_energy_all_elements_fast_update(
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
        geminal_inverse=A_inv,
    )

    np.testing.assert_array_almost_equal(ke_up_fast, ke_up_debug, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(ke_dn_fast, ke_dn_debug, decimal=decimal_debug_vs_production)


def test_debug_and_jax_discretized_kinetic_energy():
    """Test the discretized kinetic energy computation."""
    (
        _,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
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
        _compute_discretized_kinetic_energy_debug(
            alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_np, r_dn_carts=r_dn_carts_np
        )
    )

    # elements_kinetic_part_debug_all = np.array(elements_kinetic_part_debug).reshape(-1, 6)
    # print(np.array(elements_kinetic_part_debug))
    # print(elements_kinetic_part_debug_all.shape)
    # print(elements_kinetic_part_debug_all)

    mesh_kinetic_part_r_up_carts_jax, mesh_kinetic_part_r_dn_carts_jax, elements_kinetic_part_jax = (
        compute_discretized_kinetic_energy(
            alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp, RT=RT
        )
    )

    A = compute_geminal_all_elements(geminal_data=geminal_mo_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp)
    A_old_inv = np.linalg.inv(A)
    (
        mesh_kinetic_part_r_up_carts_jax_fast_update,
        mesh_kinetic_part_r_dn_carts_jax_fast_update,
        elements_kinetic_part_jax_fast_update,
    ) = compute_discretized_kinetic_energy_fast_update(
        alat=alat,
        A_old_inv=A_old_inv,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
        RT=RT,
    )

    np.testing.assert_array_almost_equal(
        mesh_kinetic_part_r_up_carts_jax,
        mesh_kinetic_part_r_up_carts_debug,
        decimal=decimal_debug_vs_production,
    )
    np.testing.assert_array_almost_equal(
        mesh_kinetic_part_r_dn_carts_jax,
        mesh_kinetic_part_r_dn_carts_debug,
        decimal=decimal_debug_vs_production,
    )
    np.testing.assert_array_almost_equal(
        mesh_kinetic_part_r_up_carts_jax_fast_update,
        mesh_kinetic_part_r_up_carts_debug,
        decimal=decimal_debug_vs_production,
    )
    np.testing.assert_array_almost_equal(
        mesh_kinetic_part_r_dn_carts_jax_fast_update,
        mesh_kinetic_part_r_dn_carts_debug,
        decimal=decimal_debug_vs_production,
    )
    np.testing.assert_array_almost_equal(
        elements_kinetic_part_jax, elements_kinetic_part_debug, decimal=decimal_debug_vs_production
    )
    np.testing.assert_array_almost_equal(
        elements_kinetic_part_jax_fast_update,
        elements_kinetic_part_debug,
        decimal=decimal_debug_vs_production,
    )


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
