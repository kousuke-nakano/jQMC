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
from jax import numpy as jnp

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.coulomb_potential import (  # noqa: E402
    _compute_bare_coulomb_potential_debug,
    _compute_bare_coulomb_potential_el_ion_element_wise_debug,
    _compute_discretized_bare_coulomb_potential_el_ion_element_wise_debug,
    _compute_ecp_local_parts_all_pairs_debug,
    _compute_ecp_non_local_parts_all_pairs_debug,
    _compute_ecp_non_local_parts_nearest_neighbors_debug,
    compute_bare_coulomb_potential,
    compute_bare_coulomb_potential_el_ion_element_wise,
    compute_discretized_bare_coulomb_potential_el_ion_element_wise,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors,
)
from jqmc.jastrow_factor import Jastrow_data  # noqa: E402
from jqmc.setting import (  # noqa: E402
    decimal_debug_vs_production,
)
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402
from jqmc.wavefunction import Wavefunction_data  # noqa: E402

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# params
angle_values = [
    (0.0, 0.0, 0.0),
    (+1.0 / 18.0 * np.pi, -1.0 / 12.0 * np.pi, +1.0 / 95.0 * np.pi),
    (+1.0 / 12.0 * np.pi, -1.0 / 17.0 * np.pi, +1.0 / 15.0 * np.pi),
    (+1.0 / 22.0 * np.pi, +1.0 / 54.0 * np.pi, +1.0 / 75.0 * np.pi),
    (-1.0 / 56.0 * np.pi, -1.0 / 13.0 * np.pi, +1.0 / 25.0 * np.pi),
    (+np.pi, +np.pi, +np.pi),
    (-np.pi, -np.pi, -np.pi),
    (+2.0 * np.pi, +2.0 * np.pi, +2.0 * np.pi),
    (-2.0 * np.pi, -2.0 * np.pi, -2.0 * np.pi),
]

# angle parameters
angle_params = [
    pytest.param(alpha, beta, gamma, id=f"(alpha,beta,gamma)=({alpha:.2f},{beta:.2f},{gamma:.2f})")
    for alpha, beta, gamma in angle_values
]

# Nv parameters
Nv_params = [pytest.param(Nv, id=f"Nv={Nv}") for Nv in (4, 6, 12, 18)]


def test_debug_and_jax_bare_coulomb():
    """Test the bare coulomb potential computation."""
    (
        _,
        _,
        _,
        _,
        _,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    # check its input
    coulomb_potential_data.sanity_check()

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

    # bare coulomb
    vpot_bare_jax = compute_bare_coulomb_potential(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
    )

    vpot_bare_debug = _compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts_np,
        r_dn_carts=r_dn_carts_np,
    )

    # print(f"vpot_bare_jax = {vpot_bare_jax}")
    # print(f"vpot_bare_debug = {vpot_bare_debug}")
    np.testing.assert_almost_equal(vpot_bare_jax, vpot_bare_debug, decimal=decimal_debug_vs_production)


def test_debug_and_jax_ecp_local():
    """Test the local ECP potential computation."""
    (
        _,
        _,
        _,
        _,
        _,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    # check its input
    coulomb_potential_data.sanity_check()

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

    # ecp local
    vpot_ecp_local_full_NN_jax = compute_ecp_local_parts_all_pairs(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
    )

    vpot_ecp_local_full_NN_debug = _compute_ecp_local_parts_all_pairs_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts_np,
        r_dn_carts=r_dn_carts_np,
    )

    np.testing.assert_almost_equal(
        vpot_ecp_local_full_NN_jax, vpot_ecp_local_full_NN_debug, decimal=decimal_debug_vs_production
    )


@pytest.mark.parametrize("Nv", Nv_params)
@pytest.mark.parametrize("alpha, beta, gamma", angle_params)
def test_debug_and_jax_ecp_non_local_full_NN(Nv, alpha, beta, gamma):
    """Test the non-local ECP potential computation with the full neibohrs."""
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    # check its input
    coulomb_potential_data.sanity_check()

    # n_atom
    n_atom = structure_data.natom

    # define data
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=None,
    )  # no jastrow for the time-being.

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

    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    cos_g, sin_g = np.cos(gamma), np.sin(gamma)

    R_np = np.array(
        [
            [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
            [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
            [-sin_b, cos_b * sin_a, cos_a * cos_b],
        ]
    )
    R_jnp = jnp.array(R_np)

    # ecp non-local (full_NN)
    (
        mesh_non_local_ecp_part_r_up_carts_full_NN_jax,
        mesh_non_local_ecp_part_r_dn_carts_full_NN_jax,
        V_nonlocal_full_NN_jax,
        sum_V_nonlocal_full_NN_jax,
    ) = compute_ecp_non_local_parts_all_pairs(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
        Nv=Nv,
        RT=R_jnp.T,
    )

    (
        mesh_non_local_ecp_part_r_up_carts_full_NN_debug,
        mesh_non_local_ecp_part_r_dn_carts_full_NN_debug,
        V_nonlocal_full_NN_debug,
        sum_V_nonlocal_full_NN_debug,
    ) = _compute_ecp_non_local_parts_all_pairs_debug(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts_np,
        r_dn_carts=r_dn_carts_np,
        Nv=Nv,
        RT=R_np.T,
    )

    np.testing.assert_almost_equal(
        sum_V_nonlocal_full_NN_debug, sum_V_nonlocal_full_NN_jax, decimal=decimal_debug_vs_production
    )

    mesh_non_local_r_up_carts_max_full_NN_jax = mesh_non_local_ecp_part_r_up_carts_full_NN_jax[
        np.argmax(V_nonlocal_full_NN_jax)
    ]
    mesh_non_local_r_up_carts_max_full_NN_debug = mesh_non_local_ecp_part_r_up_carts_full_NN_debug[
        np.argmax(V_nonlocal_full_NN_debug)
    ]
    mesh_non_local_r_dn_carts_max_full_NN_jax = mesh_non_local_ecp_part_r_dn_carts_full_NN_jax[
        np.argmax(V_nonlocal_full_NN_jax)
    ]
    mesh_non_local_r_dn_carts_max_full_NN_debug = mesh_non_local_ecp_part_r_dn_carts_full_NN_debug[
        np.argmax(V_nonlocal_full_NN_debug)
    ]
    V_ecp_non_local_max_full_NN_jax = V_nonlocal_full_NN_jax[np.argmax(V_nonlocal_full_NN_jax)]
    V_ecp_non_local_max_full_NN_debug = V_nonlocal_full_NN_debug[np.argmax(V_nonlocal_full_NN_debug)]

    np.testing.assert_almost_equal(
        V_ecp_non_local_max_full_NN_jax, V_ecp_non_local_max_full_NN_debug, decimal=decimal_debug_vs_production
    )
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_up_carts_max_full_NN_jax,
        mesh_non_local_r_up_carts_max_full_NN_debug,
        decimal=decimal_debug_vs_production,
    )
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_dn_carts_max_full_NN_jax,
        mesh_non_local_r_dn_carts_max_full_NN_debug,
        decimal=decimal_debug_vs_production,
    )

    # ecp non-local (NN, N=max)
    (
        mesh_non_local_ecp_part_r_up_carts_NN_check_jax,
        mesh_non_local_ecp_part_r_dn_carts_NN_check_jax,
        V_nonlocal_NN_check_jax,
        sum_V_nonlocal_NN_check_jax,
    ) = compute_ecp_non_local_parts_nearest_neighbors(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        NN=n_atom,
        Nv=Nv,
        RT=R_jnp.T,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
    )

    (
        mesh_non_local_ecp_part_r_up_carts_NN_check_debug,
        mesh_non_local_ecp_part_r_dn_carts_NN_check_debug,
        V_nonlocal_NN_check_debug,
        sum_V_nonlocal_NN_check_debug,
    ) = _compute_ecp_non_local_parts_nearest_neighbors_debug(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        NN=n_atom,
        Nv=Nv,
        RT=R_np.T,
        r_up_carts=r_up_carts_np,
        r_dn_carts=r_dn_carts_np,
    )

    # debug, full-NN vs check-NN
    np.testing.assert_almost_equal(
        sum_V_nonlocal_full_NN_debug, sum_V_nonlocal_NN_check_debug, decimal=decimal_debug_vs_production
    )

    # jax, full-NN vs check-NN
    np.testing.assert_almost_equal(sum_V_nonlocal_full_NN_jax, sum_V_nonlocal_NN_check_jax, decimal=decimal_debug_vs_production)

    mesh_non_local_r_up_carts_max_NN_check_jax = mesh_non_local_ecp_part_r_up_carts_NN_check_jax[
        np.argmax(V_nonlocal_NN_check_jax)
    ]
    mesh_non_local_r_up_carts_max_NN_check_debug = mesh_non_local_ecp_part_r_up_carts_NN_check_debug[
        np.argmax(V_nonlocal_NN_check_debug)
    ]
    mesh_non_local_r_dn_carts_max_NN_check_jax = mesh_non_local_ecp_part_r_dn_carts_NN_check_jax[
        np.argmax(V_nonlocal_NN_check_jax)
    ]
    mesh_non_local_r_dn_carts_max_NN_check_debug = mesh_non_local_ecp_part_r_dn_carts_NN_check_debug[
        np.argmax(V_nonlocal_NN_check_debug)
    ]
    V_ecp_non_local_max_NN_check_jax = V_nonlocal_NN_check_jax[np.argmax(V_nonlocal_NN_check_jax)]
    V_ecp_non_local_max_NN_check_debug = V_nonlocal_NN_check_debug[np.argmax(V_nonlocal_NN_check_debug)]

    # debug, full-NN vs check-NN
    np.testing.assert_almost_equal(
        V_ecp_non_local_max_full_NN_debug,
        V_ecp_non_local_max_NN_check_debug,
        decimal=decimal_debug_vs_production,
    )
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_up_carts_max_full_NN_debug,
        mesh_non_local_r_up_carts_max_NN_check_debug,
        decimal=decimal_debug_vs_production,
    )
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_dn_carts_max_full_NN_debug,
        mesh_non_local_r_dn_carts_max_NN_check_debug,
        decimal=decimal_debug_vs_production,
    )

    # jax, full-NN vs check-NN
    np.testing.assert_almost_equal(
        V_ecp_non_local_max_full_NN_jax,
        V_ecp_non_local_max_NN_check_jax,
        decimal=decimal_debug_vs_production,
    )
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_up_carts_max_full_NN_jax,
        mesh_non_local_r_up_carts_max_NN_check_jax,
        decimal=decimal_debug_vs_production,
    )
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_dn_carts_max_full_NN_jax,
        mesh_non_local_r_dn_carts_max_NN_check_jax,
        decimal=decimal_debug_vs_production,
    )


@pytest.mark.parametrize("Nv", Nv_params)
@pytest.mark.parametrize("alpha, beta, gamma", angle_params)
def test_debug_and_jax_ecp_non_local_partial_NN(Nv, alpha, beta, gamma):
    """Test the non-local ECP potential computation with partial neibohrs."""
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    # check its input
    coulomb_potential_data.sanity_check()

    # n_atom
    n_atom = structure_data.natom

    # define data
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=None,
    )  # no jastrow for the time-being.

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

    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    cos_g, sin_g = np.cos(gamma), np.sin(gamma)

    R_np = np.array(
        [
            [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
            [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
            [-sin_b, cos_b * sin_a, cos_a * cos_b],
        ]
    )
    R_jnp = jnp.array(R_np)

    for NN in range(1, n_atom):
        # ecp non-local (NN, NN=NN)
        (
            mesh_non_local_ecp_part_r_up_carts_NN_jax,
            mesh_non_local_ecp_part_r_dn_carts_NN_jax,
            V_nonlocal_NN_jax,
            sum_V_nonlocal_NN_jax,
        ) = compute_ecp_non_local_parts_nearest_neighbors(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts_jnp,
            r_dn_carts=r_dn_carts_jnp,
            Nv=Nv,
            NN=NN,
            RT=R_jnp.T,
        )

        (
            mesh_non_local_ecp_part_r_up_carts_NN_debug,
            mesh_non_local_ecp_part_r_dn_carts_NN_debug,
            V_nonlocal_NN_debug,
            sum_V_nonlocal_NN_debug,
        ) = _compute_ecp_non_local_parts_nearest_neighbors_debug(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts_np,
            r_dn_carts=r_dn_carts_np,
            Nv=Nv,
            NN=NN,
            RT=R_np.T,
        )

        np.testing.assert_almost_equal(sum_V_nonlocal_NN_debug, sum_V_nonlocal_NN_jax, decimal=decimal_debug_vs_production)

        mesh_non_local_r_up_carts_max_NN_jax = mesh_non_local_ecp_part_r_up_carts_NN_jax[np.argmax(V_nonlocal_NN_jax)]
        mesh_non_local_r_up_carts_max_NN_debug = mesh_non_local_ecp_part_r_up_carts_NN_debug[np.argmax(V_nonlocal_NN_debug)]
        mesh_non_local_r_dn_carts_max_NN_jax = mesh_non_local_ecp_part_r_dn_carts_NN_jax[np.argmax(V_nonlocal_NN_jax)]
        mesh_non_local_r_dn_carts_max_NN_debug = mesh_non_local_ecp_part_r_dn_carts_NN_debug[np.argmax(V_nonlocal_NN_debug)]
        V_ecp_non_local_max_NN_jax = V_nonlocal_NN_jax[np.argmax(V_nonlocal_NN_jax)]
        V_ecp_non_local_max_NN_debug = V_nonlocal_NN_debug[np.argmax(V_nonlocal_NN_debug)]

        np.testing.assert_almost_equal(
            V_ecp_non_local_max_NN_jax, V_ecp_non_local_max_NN_debug, decimal=decimal_debug_vs_production
        )
        np.testing.assert_array_almost_equal(
            mesh_non_local_r_up_carts_max_NN_jax,
            mesh_non_local_r_up_carts_max_NN_debug,
            decimal=decimal_debug_vs_production,
        )
        np.testing.assert_array_almost_equal(
            mesh_non_local_r_dn_carts_max_NN_jax,
            mesh_non_local_r_dn_carts_max_NN_debug,
            decimal=decimal_debug_vs_production,
        )


def test_debug_and_jax_bare_el_ion_elements():
    """Test the bare couloumb potential computation."""
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    # check its input
    coulomb_potential_data.sanity_check()

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

    interactions_R_r_up_debug, interactions_R_r_dn_debug = _compute_bare_coulomb_potential_el_ion_element_wise_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts_np,
        r_dn_carts=r_dn_carts_np,
    )

    interactions_R_r_up_jax, interactions_R_r_dn_jax = compute_bare_coulomb_potential_el_ion_element_wise(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
    )

    np.testing.assert_almost_equal(interactions_R_r_up_debug, interactions_R_r_up_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_almost_equal(interactions_R_r_dn_debug, interactions_R_r_dn_jax, decimal=decimal_debug_vs_production)


def test_debug_and_jax_discretized_bare_el_ion_elements():
    """Test the bare couloumb potential computation."""
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    # check its input
    coulomb_potential_data.sanity_check()

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

    alat = 0.5

    interactions_R_r_up_debug, interactions_R_r_dn_debug = (
        _compute_discretized_bare_coulomb_potential_el_ion_element_wise_debug(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts_np,
            r_dn_carts=r_dn_carts_np,
            alat=alat,
        )
    )

    interactions_R_r_up_jax, interactions_R_r_dn_jax = compute_discretized_bare_coulomb_potential_el_ion_element_wise(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
        alat=alat,
    )

    np.testing.assert_almost_equal(interactions_R_r_up_debug, interactions_R_r_up_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_almost_equal(interactions_R_r_dn_debug, interactions_R_r_dn_jax, decimal=decimal_debug_vs_production)


"""
def test_debug_and_jax_ecp_total():
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True)

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=None,
        jastrow_two_body_pade_flag=False,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )  # no jastrow for the time-being.

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    old_r_up_carts = np.array(
        [
            [0.64878536, -0.83275288, 0.33532629],
            [0.55271273, 0.72310605, 0.93443775],
            [0.66767275, 0.1206456, -0.36521208],
            [-0.93165236, -0.0120386, 0.33003036],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [1.0347816, 1.26162081, 0.42301735],
            [-0.57843435, 1.03651987, -0.55091542],
            [-1.56091964, -0.58952149, -0.99268141],
            [0.61863233, -0.14903326, 0.51962683],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_dn_carts[3] = [0.618632327645002, -0.149033260668010, 0.131889254514777]

    # ecp total
    vpot_ecp_jax = _compute_ecp_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    vpot_ecp_debug = _compute_ecp_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    # print(f"vpot_ecp_jax = {vpot_ecp_jax}")
    # print(f"vpot_ecp_debug = {vpot_ecp_debug}")
    np.testing.assert_almost_equal(vpot_ecp_jax, vpot_ecp_debug, decimal=10)
"""

if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
