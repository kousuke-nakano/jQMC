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
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.atomic_orbital import AOs_sphe_data, compute_overlap_matrix  # noqa: E402
from jqmc.determinant import (  # noqa: E402
    Geminal_data,
    _compute_AS_regularization_factor_debug,
    _compute_det_geminal_all_elements_debug,
    _compute_geminal_all_elements,
    _compute_geminal_all_elements_debug,
    _compute_grads_and_laplacian_ln_Det_auto,
    _compute_grads_and_laplacian_ln_Det_debug,
    _compute_grads_and_laplacian_ln_Det_fast_debug,
    _compute_ratio_determinant_part_debug,
    _compute_ratio_determinant_part_rank1_update,
    compute_AS_regularization_factor,
    compute_det_geminal_all_elements,
    compute_geminal_all_elements,
    compute_geminal_dn_one_column_elements,
    compute_geminal_up_one_row_elements,
    compute_grads_and_laplacian_ln_Det,
    compute_grads_and_laplacian_ln_Det_fast,
    compute_ln_det_geminal_all_elements,
    compute_ln_det_geminal_all_elements_fast,
)
from jqmc.molecular_orbital import MOs_data  # noqa: E402
from jqmc.setting import (  # noqa: E402
    atol_auto_vs_numerical_deriv,
    decimal_auto_vs_analytic_deriv,
    decimal_auto_vs_numerical_deriv,
    decimal_consistency,
    decimal_debug_vs_production,
    rtol_auto_vs_numerical_deriv,
)
from jqmc.structure import Structure_data  # noqa: E402
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def test_convert_from_MOs_to_AOs_closed_shell():
    """Test the consistency between AO and MO geminals."""
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

    # check geminal_data
    geminal_mo_data.sanity_check()

    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    # Initialization
    r_up_carts = []
    r_dn_carts = []

    total_electrons = 0

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data._positions_cart_np

    # Place electrons around each nucleus
    for i in range(len(coords)):
        charge = charges[i]
        num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

        # Retrieve the position coordinates
        x, y, z = coords[i]

        # Place electrons
        for _ in range(num_electrons):
            # Calculate distance range
            distance = np.random.uniform(1.0 / charge, 2.0 / charge)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            # Convert spherical to Cartesian coordinates
            dx = distance * np.sin(theta) * np.cos(phi)
            dy = distance * np.sin(theta) * np.sin(phi)
            dz = distance * np.cos(theta)

            # Position of the electron
            electron_position = np.array([x + dx, y + dy, z + dz])

            # Assign spin
            if len(r_up_carts) < num_electron_up:
                r_up_carts.append(electron_position)
            else:
                r_dn_carts.append(electron_position)

        total_electrons += num_electrons

    # Handle surplus electrons
    remaining_up = num_electron_up - len(r_up_carts)
    remaining_dn = num_electron_dn - len(r_dn_carts)

    # Randomly place any remaining electrons
    for _ in range(remaining_up):
        r_up_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))
    for _ in range(remaining_dn):
        r_dn_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))

    r_up_carts = np.array(r_up_carts)
    r_dn_carts = np.array(r_dn_carts)

    geminal_mo_debug = _compute_geminal_all_elements_debug(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    geminal_mo_jax = _compute_geminal_all_elements(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    assert not np.any(np.isnan(np.asarray(geminal_mo_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(geminal_mo_jax))), "NaN detected in second argument"
    np.testing.assert_almost_equal(geminal_mo_debug, geminal_mo_jax, decimal=decimal_debug_vs_production)

    geminal_mo = geminal_mo_jax

    """
    mo_lambda_matrix_paired, mo_lambda_matrix_unpaired = np.hsplit(geminal_mo_data.lambda_matrix, [geminal_mo_data.orb_num_dn])

    # generate matrices for the test
    ao_lambda_matrix_paired = np.dot(
        mos_data_up.mo_coefficients.T,
        np.dot(mo_lambda_matrix_paired, mos_data_dn.mo_coefficients),
    )
    ao_lambda_matrix_unpaired = np.dot(mos_data_up.mo_coefficients.T, mo_lambda_matrix_unpaired)
    ao_lambda_matrix = np.hstack([ao_lambda_matrix_paired, ao_lambda_matrix_unpaired])

    geminal_ao_data = Geminal_data(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        orb_data_up_spin=aos_data,
        orb_data_dn_spin=aos_data,
        lambda_matrix=ao_lambda_matrix,
    )
    """

    geminal_ao_data = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
    geminal_ao_data.sanity_check()

    geminal_ao_debug = _compute_geminal_all_elements_debug(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    geminal_ao_jax = _compute_geminal_all_elements_debug(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    assert not np.any(np.isnan(np.asarray(geminal_ao_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(geminal_ao_jax))), "NaN detected in second argument"
    np.testing.assert_almost_equal(geminal_ao_debug, geminal_ao_jax, decimal=decimal_debug_vs_production)

    geminal_ao = geminal_ao_jax

    # check if geminals with AO and MO representations are consistent
    assert not np.any(np.isnan(np.asarray(geminal_ao))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(geminal_mo))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(geminal_ao, geminal_mo, decimal=decimal_debug_vs_production)

    det_geminal_mo_debug = _compute_det_geminal_all_elements_debug(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    det_geminal_mo_jax = compute_det_geminal_all_elements(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    assert not np.any(np.isnan(np.asarray(det_geminal_mo_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(det_geminal_mo_jax))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(det_geminal_mo_debug, det_geminal_mo_jax, decimal=decimal_debug_vs_production)

    det_geminal_mo = det_geminal_mo_jax

    det_geminal_ao_debug = _compute_det_geminal_all_elements_debug(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    det_geminal_ao_jax = compute_det_geminal_all_elements(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    assert not np.any(np.isnan(np.asarray(det_geminal_ao_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(det_geminal_ao_jax))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(det_geminal_ao_debug, det_geminal_ao_jax, decimal=decimal_debug_vs_production)
    det_geminal_ao = det_geminal_ao_jax

    assert not np.any(np.isnan(np.asarray(det_geminal_ao))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(det_geminal_mo))), "NaN detected in second argument"
    np.testing.assert_almost_equal(det_geminal_ao, det_geminal_mo, decimal=decimal_debug_vs_production)

    jax.clear_caches()


def _build_sphe_aos_l_le6(rng: np.random.Generator) -> AOs_sphe_data:
    nucleus_index: list[int] = []
    orbital_indices: list[int] = []
    exponents: list[float] = []
    coefficients: list[float] = []
    angular_momentums: list[int] = []
    magnetic_quantum_numbers: list[int] = []

    ao_idx = 0
    for l in range(7):
        exp_l = rng.uniform(0.5, 2.0)
        coef_l = rng.uniform(0.7, 1.3)
        for m in range(-l, l + 1):
            nucleus_index.append(0)
            orbital_indices.append(ao_idx)
            exponents.append(exp_l)
            coefficients.append(coef_l)
            angular_momentums.append(l)
            magnetic_quantum_numbers.append(m)
            ao_idx += 1

    structure_data = Structure_data(
        pbc_flag=False,
        positions=np.zeros((1, 3), dtype=np.float64),
        atomic_numbers=(1,),
        element_symbols=("X",),
        atomic_labels=("X",),
    )

    return AOs_sphe_data(
        structure_data=structure_data,
        nucleus_index=tuple(nucleus_index),
        num_ao=len(angular_momentums),
        num_ao_prim=len(exponents),
        orbital_indices=tuple(orbital_indices),
        exponents=tuple(exponents),
        coefficients=tuple(coefficients),
        angular_momentums=tuple(angular_momentums),
        magnetic_quantum_numbers=tuple(magnetic_quantum_numbers),
    )


def test_geminal_sphe_to_cart_AOs_data():
    """Round-trip AOs l<=6: spherical→Cartesian keeps geminal values/grads."""
    rng = np.random.default_rng(321)

    aos_sphe = _build_sphe_aos_l_le6(rng)
    num_electron_up = 4
    num_electron_dn = 4

    lambda_matrix = rng.uniform(-0.2, 0.2, size=(aos_sphe.num_ao, aos_sphe.num_ao))
    geminal_sph = Geminal_data(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        orb_data_up_spin=aos_sphe,
        orb_data_dn_spin=aos_sphe,
        lambda_matrix=lambda_matrix,
    )
    geminal_cart = geminal_sph.to_cartesian()

    r_up_carts = rng.uniform(-1.0, 1.0, size=(num_electron_up, 3))
    r_dn_carts = rng.uniform(-1.0, 1.0, size=(num_electron_dn, 3))

    G_sph = compute_geminal_all_elements(geminal_sph, r_up_carts, r_dn_carts)
    G_cart = compute_geminal_all_elements(geminal_cart, r_up_carts, r_dn_carts)
    assert not np.any(np.isnan(np.asarray(np.asarray(G_sph)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(G_cart)))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(np.asarray(G_sph), np.asarray(G_cart), decimal=decimal_consistency)

    grads_sph = compute_grads_and_laplacian_ln_Det(geminal_sph, r_up_carts, r_dn_carts)
    grads_cart = compute_grads_and_laplacian_ln_Det(geminal_cart, r_up_carts, r_dn_carts)
    for sph, cart in zip(grads_sph, grads_cart, strict=True):
        assert not np.any(np.isnan(np.asarray(np.asarray(sph)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(cart)))), "NaN detected in second argument"
        np.testing.assert_array_almost_equal(np.asarray(sph), np.asarray(cart), decimal=decimal_consistency)

    jax.clear_caches()


def test_geminal_cart_to_sphe_AOs_data():
    """Round-trip AOs l<=6: Cartesian→spherical keeps geminal values/grads."""
    rng = np.random.default_rng(654)

    aos_sphe = _build_sphe_aos_l_le6(rng)
    num_electron_up = 5
    num_electron_dn = 5

    lambda_matrix = rng.uniform(-0.2, 0.2, size=(aos_sphe.num_ao, aos_sphe.num_ao))
    geminal_sph = Geminal_data(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        orb_data_up_spin=aos_sphe,
        orb_data_dn_spin=aos_sphe,
        lambda_matrix=lambda_matrix,
    )

    geminal_cart = geminal_sph.to_cartesian()
    geminal_cart_to_sph = geminal_cart.to_spherical()

    r_up_carts = rng.uniform(-1.0, 1.0, size=(num_electron_up, 3))
    r_dn_carts = rng.uniform(-1.0, 1.0, size=(num_electron_dn, 3))

    G_cart = compute_geminal_all_elements(geminal_cart, r_up_carts, r_dn_carts)
    G_sph = compute_geminal_all_elements(geminal_cart_to_sph, r_up_carts, r_dn_carts)
    assert not np.any(np.isnan(np.asarray(np.asarray(G_cart)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(G_sph)))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(np.asarray(G_cart), np.asarray(G_sph), decimal=decimal_consistency)

    grads_cart = compute_grads_and_laplacian_ln_Det(geminal_cart, r_up_carts, r_dn_carts)
    grads_sph = compute_grads_and_laplacian_ln_Det(geminal_cart_to_sph, r_up_carts, r_dn_carts)
    for cart, sph in zip(grads_cart, grads_sph, strict=True):
        assert not np.any(np.isnan(np.asarray(np.asarray(cart)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(sph)))), "NaN detected in second argument"
        np.testing.assert_array_almost_equal(np.asarray(cart), np.asarray(sph), decimal=decimal_consistency)

    jax.clear_caches()


def test_geminal_sphe_to_cart_MOs_data():
    """Round-trip MOs built on l<=6 AOs: spherical→Cartesian keeps geminal values/grads."""
    rng = np.random.default_rng(777)

    aos_sphe = _build_sphe_aos_l_le6(rng)
    num_mo = aos_sphe.num_ao
    mo_coefficients = rng.uniform(-0.5, 0.5, size=(num_mo, aos_sphe.num_ao))
    mos_sphe = MOs_data(num_mo=num_mo, aos_data=aos_sphe, mo_coefficients=mo_coefficients)
    mos_sphe.sanity_check()

    num_electron_up = 5
    num_electron_dn = 5
    lambda_matrix = rng.uniform(-0.2, 0.2, size=(mos_sphe.num_mo, mos_sphe.num_mo))
    geminal_sph = Geminal_data(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        orb_data_up_spin=mos_sphe,
        orb_data_dn_spin=mos_sphe,
        lambda_matrix=lambda_matrix,
    )
    geminal_cart = geminal_sph.to_cartesian()

    r_up_carts = rng.uniform(-1.0, 1.0, size=(num_electron_up, 3))
    r_dn_carts = rng.uniform(-1.0, 1.0, size=(num_electron_dn, 3))

    G_sph = compute_geminal_all_elements(geminal_sph, r_up_carts, r_dn_carts)
    G_cart = compute_geminal_all_elements(geminal_cart, r_up_carts, r_dn_carts)
    assert not np.any(np.isnan(np.asarray(np.asarray(G_sph)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(G_cart)))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(np.asarray(G_sph), np.asarray(G_cart), decimal=decimal_consistency)

    grads_sph = compute_grads_and_laplacian_ln_Det(geminal_sph, r_up_carts, r_dn_carts)
    grads_cart = compute_grads_and_laplacian_ln_Det(geminal_cart, r_up_carts, r_dn_carts)
    for sph, cart in zip(grads_sph, grads_cart, strict=True):
        assert not np.any(np.isnan(np.asarray(np.asarray(sph)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(cart)))), "NaN detected in second argument"
        np.testing.assert_array_almost_equal(np.asarray(sph), np.asarray(cart), decimal=decimal_consistency)

    jax.clear_caches()


def test_geminal_cart_to_sphe_MOs_data():
    """Round-trip MOs l<=6: Cartesian→spherical keeps geminal values/grads."""
    rng = np.random.default_rng(888)

    aos_sphe = _build_sphe_aos_l_le6(rng)
    num_mo = aos_sphe.num_ao
    mo_coefficients = rng.uniform(-0.5, 0.5, size=(num_mo, aos_sphe.num_ao))
    mos_sphe = MOs_data(num_mo=num_mo, aos_data=aos_sphe, mo_coefficients=mo_coefficients)
    mos_sphe.sanity_check()

    num_electron_up = 6
    num_electron_dn = 6
    lambda_matrix = rng.uniform(-0.2, 0.2, size=(mos_sphe.num_mo, mos_sphe.num_mo))
    geminal_sph = Geminal_data(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        orb_data_up_spin=mos_sphe,
        orb_data_dn_spin=mos_sphe,
        lambda_matrix=lambda_matrix,
    )

    geminal_cart = geminal_sph.to_cartesian()
    geminal_cart_to_sph = geminal_cart.to_spherical()

    r_up_carts = rng.uniform(-1.0, 1.0, size=(num_electron_up, 3))
    r_dn_carts = rng.uniform(-1.0, 1.0, size=(num_electron_dn, 3))

    G_cart = compute_geminal_all_elements(geminal_cart, r_up_carts, r_dn_carts)
    G_sph = compute_geminal_all_elements(geminal_cart_to_sph, r_up_carts, r_dn_carts)
    assert not np.any(np.isnan(np.asarray(np.asarray(G_cart)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(G_sph)))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(np.asarray(G_cart), np.asarray(G_sph), decimal=decimal_consistency)

    grads_cart = compute_grads_and_laplacian_ln_Det(geminal_cart, r_up_carts, r_dn_carts)
    grads_sph = compute_grads_and_laplacian_ln_Det(geminal_cart_to_sph, r_up_carts, r_dn_carts)
    for cart, sph in zip(grads_cart, grads_sph, strict=True):
        assert not np.any(np.isnan(np.asarray(np.asarray(cart)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(sph)))), "NaN detected in second argument"
        np.testing.assert_array_almost_equal(np.asarray(cart), np.asarray(sph), decimal=decimal_consistency)

    jax.clear_caches()


def _build_small_sphe_aos_for_conversion() -> AOs_sphe_data:
    structure_data = Structure_data(
        pbc_flag=False,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        atomic_numbers=(1,),
        element_symbols=("X",),
        atomic_labels=("X",),
    )
    return AOs_sphe_data(
        structure_data=structure_data,
        nucleus_index=(0, 0, 0, 0),
        num_ao=4,
        num_ao_prim=4,
        orbital_indices=(0, 1, 2, 3),
        exponents=(1.20, 1.00, 1.00, 1.00),
        coefficients=(1.0, 1.0, 1.0, 1.0),
        angular_momentums=(0, 1, 1, 1),
        magnetic_quantum_numbers=(0, -1, 0, 1),
    )


def test_convert_from_AOs_to_MOs_full_projection_closed_shell():
    """AO->MO (all eigenvectors) followed by MO->AO recovers the AO lambda matrix."""
    rng = np.random.default_rng(1234)
    aos_data = _build_small_sphe_aos_for_conversion()
    aos_data.sanity_check()

    num_electron_up = 2
    num_electron_dn = 2

    lambda_matrix_paired = rng.uniform(-0.25, 0.25, size=(aos_data.num_ao, aos_data.num_ao))
    lambda_matrix_paired = 0.5 * (lambda_matrix_paired + lambda_matrix_paired.T)
    lambda_matrix = np.hstack([lambda_matrix_paired, np.zeros((aos_data.num_ao, 0), dtype=np.float64)])

    geminal_ao = Geminal_data(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        orb_data_up_spin=aos_data,
        orb_data_dn_spin=aos_data,
        lambda_matrix=lambda_matrix,
    )
    geminal_mo = Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors="all")
    geminal_ao_back = Geminal_data.convert_from_MOs_to_AOs(geminal_mo)

    assert not np.any(np.isnan(np.asarray(np.asarray(geminal_ao_back.lambda_matrix)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(geminal_ao.lambda_matrix)))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        np.asarray(geminal_ao_back.lambda_matrix),
        np.asarray(geminal_ao.lambda_matrix),
        decimal=decimal_consistency,
    )


def test_convert_from_AOs_to_MOs_full_projection_open_shell():
    """AO->MO (all eigenvectors) round-trip recovers AO lambda matrix for open-shell case."""
    rng = np.random.default_rng(1334)
    aos_data = _build_small_sphe_aos_for_conversion()
    aos_data.sanity_check()

    num_electron_up = 3
    num_electron_dn = 2
    num_unpaired = num_electron_up - num_electron_dn

    lambda_matrix_paired = rng.uniform(-0.25, 0.25, size=(aos_data.num_ao, aos_data.num_ao))
    lambda_matrix_paired = 0.5 * (lambda_matrix_paired + lambda_matrix_paired.T)
    lambda_matrix_unpaired = rng.uniform(-0.25, 0.25, size=(aos_data.num_ao, num_unpaired))
    lambda_matrix = np.hstack([lambda_matrix_paired, lambda_matrix_unpaired])

    geminal_ao = Geminal_data(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        orb_data_up_spin=aos_data,
        orb_data_dn_spin=aos_data,
        lambda_matrix=lambda_matrix,
    )
    geminal_mo = Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors="all")
    geminal_ao_back = Geminal_data.convert_from_MOs_to_AOs(geminal_mo)

    assert not np.any(np.isnan(np.asarray(np.asarray(geminal_ao_back.lambda_matrix)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(geminal_ao.lambda_matrix)))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        np.asarray(geminal_ao_back.lambda_matrix),
        np.asarray(geminal_ao.lambda_matrix),
        decimal=decimal_consistency,
    )


def test_convert_from_AOs_to_MOs_truncated_mode_closed_shell():
    """AO->MO integer mode accepts boundary values without enforcing eigenvalue scaling."""
    rng = np.random.default_rng(4321)
    aos_data = _build_small_sphe_aos_for_conversion()
    aos_data.sanity_check()

    num_electron_up = 2
    num_electron_dn = 2

    lambda_matrix_paired = rng.uniform(-0.20, 0.20, size=(aos_data.num_ao, aos_data.num_ao))
    lambda_matrix_paired = 0.5 * (lambda_matrix_paired + lambda_matrix_paired.T)
    lambda_matrix = np.hstack([lambda_matrix_paired, np.zeros((aos_data.num_ao, 0), dtype=np.float64)])

    geminal_ao = Geminal_data(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        orb_data_up_spin=aos_data,
        orb_data_dn_spin=aos_data,
        lambda_matrix=lambda_matrix,
    )

    geminal_mo = Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors=3)
    paired_block, _ = np.hsplit(np.asarray(geminal_mo.lambda_matrix), [geminal_mo.orb_num_dn])

    geminal_mo_small = Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors=2)
    assert geminal_mo_small.orb_num_up == 2
    assert geminal_mo_small.orb_num_dn == 2

    geminal_mo_full = Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors=4)
    assert geminal_mo_full.orb_num_up == aos_data.num_ao
    assert geminal_mo_full.orb_num_dn == aos_data.num_ao

    geminal_mo_clipped = Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors=5)
    assert geminal_mo_clipped.orb_num_up == aos_data.num_ao
    assert geminal_mo_clipped.orb_num_dn == aos_data.num_ao


def test_convert_from_AOs_to_MOs_truncated_mode_open_shell():
    """Open-shell truncation keeps projected unpaired block without paired-eigenvalue scaling."""
    rng = np.random.default_rng(2468)
    aos_data = _build_sphe_aos_l_le6(rng)
    aos_data.sanity_check()

    num_electron_up = 3
    num_electron_dn = 2
    num_unpaired = num_electron_up - num_electron_dn

    lambda_matrix_paired = rng.uniform(-0.2, 0.2, size=(aos_data.num_ao, aos_data.num_ao))
    lambda_matrix_unpaired = rng.uniform(-0.2, 0.2, size=(aos_data.num_ao, num_unpaired))
    lambda_matrix = np.hstack([lambda_matrix_paired, lambda_matrix_unpaired])

    geminal_ao = Geminal_data(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        orb_data_up_spin=aos_data,
        orb_data_dn_spin=aos_data,
        lambda_matrix=lambda_matrix,
    )

    geminal_mo = Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors=5)
    paired_block, unpaired_block = np.hsplit(np.asarray(geminal_mo.lambda_matrix), [geminal_mo.orb_num_dn])

    assert unpaired_block.shape == (5, 1)


def test_apply_ao_projected_paired_update_and_reproject_fixed_num_dn():
    """AO-corrected paired update is applied then reprojected with fixed N=num_electron_dn."""
    rng = np.random.default_rng(97531)
    aos_data = _build_small_sphe_aos_for_conversion()
    aos_data.sanity_check()

    num_electron_up = 2
    num_electron_dn = 2
    num_mo = aos_data.num_ao

    mo_coefficients = rng.uniform(-0.4, 0.4, size=(num_mo, aos_data.num_ao))
    mo_coefficients += np.eye(num_mo)
    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
    mos_data.sanity_check()

    lambda_mo_paired = rng.uniform(-0.2, 0.2, size=(num_mo, num_mo))
    lambda_mo_paired = 0.5 * (lambda_mo_paired + lambda_mo_paired.T)
    lambda_mo = np.hstack([lambda_mo_paired, np.zeros((num_mo, 0), dtype=np.float64)])

    geminal_mo = Geminal_data(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        orb_data_up_spin=mos_data,
        orb_data_dn_spin=mos_data,
        lambda_matrix=lambda_mo,
    )

    ao_paired_direction = rng.uniform(-0.1, 0.1, size=(aos_data.num_ao, aos_data.num_ao))
    step_size = 0.3

    overlap = np.asarray(compute_overlap_matrix(aos_data), dtype=np.float64)
    overlap = 0.5 * (overlap + overlap.T)

    p_matrix_cols = np.asarray(mos_data.mo_coefficients, dtype=np.float64).T
    right_projector = (overlap @ p_matrix_cols) @ p_matrix_cols.T
    left_projector = right_projector.T
    identity = np.eye(aos_data.num_ao, dtype=np.float64)

    corrected_direction = ao_paired_direction - (
        (identity - left_projector.T) @ ao_paired_direction @ (identity - right_projector.T)
    )

    geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo)
    ao_lambda_paired, ao_lambda_unpaired = np.hsplit(np.asarray(geminal_ao.lambda_matrix), [geminal_ao.orb_num_dn])
    ao_lambda_updated = np.hstack([ao_lambda_paired + step_size * corrected_direction, ao_lambda_unpaired])

    expected = Geminal_data.convert_from_AOs_to_MOs(
        Geminal_data(
            num_electron_up=geminal_ao.num_electron_up,
            num_electron_dn=geminal_ao.num_electron_dn,
            orb_data_up_spin=geminal_ao.orb_data_up_spin,
            orb_data_dn_spin=geminal_ao.orb_data_dn_spin,
            lambda_matrix=ao_lambda_updated,
        ),
        num_eigenvectors=num_electron_dn,
    )

    actual = geminal_mo.apply_ao_projected_paired_update_and_reproject(
        ao_paired_direction=ao_paired_direction,
        step_size=step_size,
    )

    assert actual.orb_num_up == num_electron_dn
    assert actual.orb_num_dn == num_electron_dn
    assert not np.any(np.isnan(np.asarray(np.asarray(actual.lambda_matrix)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(expected.lambda_matrix)))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        np.asarray(actual.lambda_matrix),
        np.asarray(expected.lambda_matrix),
        decimal=decimal_consistency,
    )


def test_apply_ao_projected_paired_update_and_reproject_input_validation():
    """AO-projected paired update rejects AO representation and wrong direction shape."""
    rng = np.random.default_rng(24680)
    aos_data = _build_small_sphe_aos_for_conversion()

    num_electron_up = 2
    num_electron_dn = 2

    lambda_matrix_paired = rng.uniform(-0.2, 0.2, size=(aos_data.num_ao, aos_data.num_ao))
    lambda_matrix_paired = 0.5 * (lambda_matrix_paired + lambda_matrix_paired.T)
    lambda_matrix = np.hstack([lambda_matrix_paired, np.zeros((aos_data.num_ao, 0), dtype=np.float64)])

    geminal_ao = Geminal_data(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        orb_data_up_spin=aos_data,
        orb_data_dn_spin=aos_data,
        lambda_matrix=lambda_matrix,
    )

    with pytest.raises(ValueError):
        geminal_ao.apply_ao_projected_paired_update_and_reproject(
            ao_paired_direction=np.zeros((aos_data.num_ao, aos_data.num_ao), dtype=np.float64),
            step_size=1.0,
        )

    num_mo = aos_data.num_ao
    mo_coefficients = rng.uniform(-0.4, 0.4, size=(num_mo, aos_data.num_ao)) + np.eye(num_mo)
    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    geminal_mo = Geminal_data(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        orb_data_up_spin=mos_data,
        orb_data_dn_spin=mos_data,
        lambda_matrix=lambda_matrix,
    )

    with pytest.raises(ValueError):
        geminal_mo.apply_ao_projected_paired_update_and_reproject(
            ao_paired_direction=np.zeros((aos_data.num_ao - 1, aos_data.num_ao), dtype=np.float64),
            step_size=1.0,
        )


def test_grads_and_laplacian_fast_update():
    """compute_grads_and_laplacian_ln_Det_fast matches _fast_debug output."""
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

    geminal_mo_data.sanity_check()

    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    r_cart_min, r_cart_max = -2.0, 2.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_electron_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_electron_dn, 3) + r_cart_min

    # Build geminal and its inverse (mirrors determinant.py logic)
    lambda_matrix_paired, lambda_matrix_unpaired = jnp.hsplit(geminal_mo_data.lambda_matrix, [geminal_mo_data.orb_num_dn])

    ao_matrix_up = geminal_mo_data.compute_orb_api(geminal_mo_data.orb_data_up_spin, r_up_carts)
    ao_matrix_dn = geminal_mo_data.compute_orb_api(geminal_mo_data.orb_data_dn_spin, r_dn_carts)

    geminal_paired = jnp.dot(ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn))
    geminal_unpaired = jnp.dot(ao_matrix_up.T, lambda_matrix_unpaired)
    geminal = jnp.hstack([geminal_paired, geminal_unpaired])

    P, L, U = jsp_linalg.lu(geminal)
    n = geminal.shape[0]
    I = jnp.eye(n, dtype=geminal.dtype)
    Y = jsp_linalg.solve_triangular(L, jnp.dot(P.T, I), lower=True)
    geminal_inverse = jsp_linalg.solve_triangular(U, Y, lower=False)

    # Fast path (requires inverse)
    grad_up_fast, grad_dn_fast, lap_up_fast, lap_dn_fast = compute_grads_and_laplacian_ln_Det_fast(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        geminal_inverse=geminal_inverse,
    )

    # Debug helper
    grad_up_debug, grad_dn_debug, lap_up_debug, lap_dn_debug = _compute_grads_and_laplacian_ln_Det_fast_debug(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    assert not np.any(np.isnan(np.asarray(grad_up_fast))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(grad_up_debug))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(grad_up_fast, grad_up_debug, decimal=decimal_debug_vs_production)
    assert not np.any(np.isnan(np.asarray(grad_dn_fast))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(grad_dn_debug))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(grad_dn_fast, grad_dn_debug, decimal=decimal_debug_vs_production)
    assert not np.any(np.isnan(np.asarray(lap_up_fast))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(lap_up_debug))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(lap_up_fast, lap_up_debug, decimal=decimal_debug_vs_production)
    assert not np.any(np.isnan(np.asarray(lap_dn_fast))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(lap_dn_debug))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(lap_dn_fast, lap_dn_debug, decimal=decimal_debug_vs_production)


def test_comparing_AS_regularization():
    """Test the consistency between AS_regularization_debug and AS_regularization_jax."""
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

    # check geminal_data
    geminal_mo_data.sanity_check()

    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    # Initialization
    r_up_carts = []
    r_dn_carts = []

    total_electrons = 0

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data._positions_cart_np

    # Place electrons around each nucleus
    for i in range(len(coords)):
        charge = charges[i]
        num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

        # Retrieve the position coordinates
        x, y, z = coords[i]

        # Place electrons
        for _ in range(num_electrons):
            # Calculate distance range
            distance = np.random.uniform(1.0 / charge, 2.0 / charge)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            # Convert spherical to Cartesian coordinates
            dx = distance * np.sin(theta) * np.cos(phi)
            dy = distance * np.sin(theta) * np.sin(phi)
            dz = distance * np.cos(theta)

            # Position of the electron
            electron_position = np.array([x + dx, y + dy, z + dz])

            # Assign spin
            if len(r_up_carts) < num_electron_up:
                r_up_carts.append(electron_position)
            else:
                r_dn_carts.append(electron_position)

        total_electrons += num_electrons

    # Handle surplus electrons
    remaining_up = num_electron_up - len(r_up_carts)
    remaining_dn = num_electron_dn - len(r_dn_carts)

    # Randomly place any remaining electrons
    for _ in range(remaining_up):
        r_up_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))
    for _ in range(remaining_dn):
        r_dn_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))

    r_up_carts = np.array(r_up_carts)
    r_dn_carts = np.array(r_dn_carts)

    R_AS_debug = _compute_AS_regularization_factor_debug(
        geminal_data=geminal_mo_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    R_AS_jax = compute_AS_regularization_factor(geminal_data=geminal_mo_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

    assert not np.any(np.isnan(np.asarray(R_AS_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(R_AS_jax))), "NaN detected in second argument"
    np.testing.assert_almost_equal(R_AS_debug, R_AS_jax, decimal=decimal_debug_vs_production)

    jax.clear_caches()


def test_one_row_or_one_column_update():
    """Test the update of one row or one column in the geminal wave function."""
    """Test the consistency between AS_regularization_debug and AS_regularization_jax."""
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

    # check geminal_data
    geminal_mo_data.sanity_check()

    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    # Initialization
    r_up_carts = []
    r_dn_carts = []

    total_electrons = 0

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data._positions_cart_np

    # Place electrons around each nucleus
    for i in range(len(coords)):
        charge = charges[i]
        num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

        # Retrieve the position coordinates
        x, y, z = coords[i]

        # Place electrons
        for _ in range(num_electrons):
            # Calculate distance range
            distance = np.random.uniform(1.0 / charge, 2.0 / charge)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            # Convert spherical to Cartesian coordinates
            dx = distance * np.sin(theta) * np.cos(phi)
            dy = distance * np.sin(theta) * np.sin(phi)
            dz = distance * np.cos(theta)

            # Position of the electron
            electron_position = np.array([x + dx, y + dy, z + dz])

            # Assign spin
            if len(r_up_carts) < num_electron_up:
                r_up_carts.append(electron_position)
            else:
                r_dn_carts.append(electron_position)

        total_electrons += num_electrons

    # Handle surplus electrons
    remaining_up = num_electron_up - len(r_up_carts)
    remaining_dn = num_electron_dn - len(r_dn_carts)

    # Randomly place any remaining electrons
    for _ in range(remaining_up):
        r_up_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))
    for _ in range(remaining_dn):
        r_dn_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))

    r_up_carts = np.array(r_up_carts)
    r_dn_carts = np.array(r_dn_carts)

    geminal_mo = compute_geminal_all_elements(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # Pick test indices (any valid indices are fine)
    i_up = 0
    j_dn = 0

    # Compute the single "up row" against all dn electrons
    geminal_mo_up_one_row = compute_geminal_up_one_row_elements(
        geminal_data=geminal_mo_data,
        r_up_cart=np.reshape(r_up_carts[i_up], (1, 3)),  # enforce (1,3) for single point
        r_dn_carts=r_dn_carts,
    )

    # Compute the single "dn column" against all up electrons
    geminal_mo_dn_one_column = compute_geminal_dn_one_column_elements(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_cart=np.reshape(r_dn_carts[j_dn], (1, 3)),  # enforce (1,3) for single point
    )

    # --- Numerical consistency asserts (no shape checks) ---
    # up-one-row must equal the i-th row of the full geminal
    assert not np.any(np.isnan(np.asarray(np.asarray(geminal_mo_up_one_row).ravel()))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(geminal_mo[i_up, :])))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        np.asarray(geminal_mo_up_one_row).ravel(),
        np.asarray(geminal_mo[i_up, :]),
        decimal=decimal_debug_vs_production,
    )

    # dn-one-column must equal the j-th *paired* column of the full geminal
    assert not np.any(np.isnan(np.asarray(np.asarray(geminal_mo_dn_one_column).ravel()))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(geminal_mo[:, j_dn])))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        np.asarray(geminal_mo_dn_one_column).ravel(),
        np.asarray(geminal_mo[:, j_dn]),
        decimal=decimal_debug_vs_production,
    )


def test_numerial_and_auto_grads_and_laplacians_ln_Det():
    """Test the numerical and automatic gradients of the logarithm of the determinant of the geminal wave function."""
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"),
        store_tuple=True,
    )

    geminal_mo_data.sanity_check()

    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    # Initialization
    r_up_carts = []
    r_dn_carts = []

    total_electrons = 0

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data._positions_cart_np

    # Place electrons around each nucleus
    for i in range(len(coords)):
        charge = charges[i]
        num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

        # Retrieve the position coordinates
        x, y, z = coords[i]

        # Place electrons
        for _ in range(num_electrons):
            # Calculate distance range
            distance = np.random.uniform(0.5 / charge, 1.5 / charge)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            # Convert spherical to Cartesian coordinates
            dx = distance * np.sin(theta) * np.cos(phi)
            dy = distance * np.sin(theta) * np.sin(phi)
            dz = distance * np.cos(theta)

            # Position of the electron
            electron_position = np.array([x + dx, y + dy, z + dz])

            # Assign spin
            if len(r_up_carts) < num_electron_up:
                r_up_carts.append(electron_position)
            else:
                r_dn_carts.append(electron_position)

        total_electrons += num_electrons

    # Handle surplus electrons
    remaining_up = num_electron_up - len(r_up_carts)
    remaining_dn = num_electron_dn - len(r_dn_carts)

    # Randomly place any remaining electrons
    for _ in range(remaining_up):
        r_up_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))
    for _ in range(remaining_dn):
        r_dn_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))

    r_up_carts = np.array(r_up_carts)
    r_dn_carts = np.array(r_dn_carts)

    """
    mo_lambda_matrix_paired, mo_lambda_matrix_unpaired = np.hsplit(geminal_mo_data.lambda_matrix, [geminal_mo_data.orb_num_dn])

    # generate matrices for the test
    ao_lambda_matrix_paired = np.dot(
        mos_data_up.mo_coefficients.T,
        np.dot(mo_lambda_matrix_paired, mos_data_dn.mo_coefficients),
    )
    ao_lambda_matrix_unpaired = np.dot(mos_data_up.mo_coefficients.T, mo_lambda_matrix_unpaired)
    ao_lambda_matrix = np.hstack([ao_lambda_matrix_paired, ao_lambda_matrix_unpaired])

    geminal_ao_data = Geminal_data(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        orb_data_up_spin=aos_data,
        orb_data_dn_spin=aos_data,
        lambda_matrix=ao_lambda_matrix,
    )
    """

    geminal_ao_data = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
    geminal_ao_data.sanity_check()

    grad_ln_D_up_numerical, grad_ln_D_dn_numerical, lap_ln_D_up_numerical, lap_ln_D_dn_numerical = (
        _compute_grads_and_laplacian_ln_Det_debug(
            geminal_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
    )

    grad_ln_D_up_auto, grad_ln_D_dn_auto, lap_ln_D_up_auto, lap_ln_D_dn_auto = _compute_grads_and_laplacian_ln_Det_auto(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    assert not np.any(np.isnan(np.asarray(np.asarray(grad_ln_D_up_numerical)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(grad_ln_D_up_auto)))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        np.asarray(grad_ln_D_up_numerical),
        np.asarray(grad_ln_D_up_auto),
        decimal=decimal_auto_vs_numerical_deriv,
    )
    assert not np.any(np.isnan(np.asarray(np.asarray(grad_ln_D_dn_numerical)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(grad_ln_D_dn_auto)))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        np.asarray(grad_ln_D_dn_numerical),
        np.asarray(grad_ln_D_dn_auto),
        decimal=decimal_auto_vs_numerical_deriv,
    )
    assert not np.any(np.isnan(np.asarray(np.asarray(lap_ln_D_up_numerical)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(lap_ln_D_up_auto)))), "NaN detected in second argument"
    np.testing.assert_allclose(
        np.asarray(lap_ln_D_up_numerical),
        np.asarray(lap_ln_D_up_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )
    assert not np.any(np.isnan(np.asarray(np.asarray(lap_ln_D_dn_numerical)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(lap_ln_D_dn_auto)))), "NaN detected in second argument"
    np.testing.assert_allclose(
        np.asarray(lap_ln_D_dn_numerical),
        np.asarray(lap_ln_D_dn_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )

    jax.clear_caches()


@pytest.mark.parametrize(
    "trexio_file",
    ["H2_ae_ccpvqz.h5", "H2_ae_ccpvtz_cart.h5", "H2_ecp_ccpvtz.h5", "H2_ecp_ccpvtz_cart.h5", "water_ccecp_ccpvqz.h5"],
)
def test_analytic_and_auto_grads_and_laplacians_ln_Det(trexio_file: str):
    """Test the analytic and automatic gradients of the logarithm of the determinant of the geminal wave function."""
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file),
        store_tuple=True,
    )

    geminal_mo_data.sanity_check()

    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    # Initialization
    r_up_carts = []
    r_dn_carts = []

    total_electrons = 0

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data._positions_cart_np

    # Place electrons around each nucleus
    for i in range(len(coords)):
        charge = charges[i]
        num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

        # Retrieve the position coordinates
        x, y, z = coords[i]

        # Place electrons
        for _ in range(num_electrons):
            # Calculate distance range
            distance = np.random.uniform(0.5 / charge, 1.5 / charge)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            # Convert spherical to Cartesian coordinates
            dx = distance * np.sin(theta) * np.cos(phi)
            dy = distance * np.sin(theta) * np.sin(phi)
            dz = distance * np.cos(theta)

            # Position of the electron
            electron_position = np.array([x + dx, y + dy, z + dz])

            # Assign spin
            if len(r_up_carts) < num_electron_up:
                r_up_carts.append(electron_position)
            else:
                r_dn_carts.append(electron_position)

        total_electrons += num_electrons

    # Handle surplus electrons
    remaining_up = num_electron_up - len(r_up_carts)
    remaining_dn = num_electron_dn - len(r_dn_carts)

    # Randomly place any remaining electrons
    for _ in range(remaining_up):
        r_up_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))
    for _ in range(remaining_dn):
        r_dn_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))

    r_up_carts = np.array(r_up_carts)
    r_dn_carts = np.array(r_dn_carts)

    geminal_ao_data = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
    geminal_ao_data.sanity_check()

    grad_ln_D_up_analytic, grad_ln_D_dn_analytic, lap_ln_D_up_analytic, lap_ln_D_dn_analytic = (
        compute_grads_and_laplacian_ln_Det(
            geminal_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
    )

    grad_ln_D_up_auto, grad_ln_D_dn_auto, lap_ln_D_up_auto, lap_ln_D_dn_auto = _compute_grads_and_laplacian_ln_Det_auto(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    assert not np.any(np.isnan(np.asarray(np.asarray(grad_ln_D_up_analytic)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(grad_ln_D_up_auto)))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        np.asarray(grad_ln_D_up_analytic),
        np.asarray(grad_ln_D_up_auto),
        decimal=decimal_auto_vs_analytic_deriv,
    )
    assert not np.any(np.isnan(np.asarray(np.asarray(grad_ln_D_dn_analytic)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(grad_ln_D_dn_auto)))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        np.asarray(grad_ln_D_dn_analytic),
        np.asarray(grad_ln_D_dn_auto),
        decimal=decimal_auto_vs_analytic_deriv,
    )
    assert not np.any(np.isnan(np.asarray(np.asarray(lap_ln_D_up_analytic)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(lap_ln_D_up_auto)))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        np.asarray(lap_ln_D_up_analytic),
        np.asarray(lap_ln_D_up_auto),
        decimal=decimal_auto_vs_analytic_deriv,
    )
    assert not np.any(np.isnan(np.asarray(np.asarray(lap_ln_D_dn_analytic)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(lap_ln_D_dn_auto)))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        np.asarray(lap_ln_D_dn_analytic),
        np.asarray(lap_ln_D_dn_auto),
        decimal=decimal_auto_vs_analytic_deriv,
    )

    jax.clear_caches()


def _prepare_ratio_determinant_inputs():
    (
        _structure_data,
        _aos_data,
        _mos_data_up,
        _mos_data_dn,
        geminal_mo_data,
        _coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"),
        store_tuple=True,
    )

    geminal_mo_data.sanity_check()

    rng = np.random.default_rng(0)
    r_up_carts = rng.uniform(-1.0, 1.0, size=(geminal_mo_data.num_electron_up, 3))
    r_dn_carts = rng.uniform(-1.0, 1.0, size=(geminal_mo_data.num_electron_dn, 3))

    geminal_old = compute_geminal_all_elements(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    P, L, U = jsp_linalg.lu(geminal_old)
    n = geminal_old.shape[0]
    I = jnp.eye(n, dtype=geminal_old.dtype)
    Y = jsp_linalg.solve_triangular(L, jnp.dot(P.T, I), lower=True)
    A_old_inv = jsp_linalg.solve_triangular(U, Y, lower=False)

    return geminal_mo_data, r_up_carts, r_dn_carts, A_old_inv


def _build_three_grid_moves_for_determinant(old_r_up_carts: np.ndarray, old_r_dn_carts: np.ndarray, pattern: str):
    new_r_up_carts_arr = np.repeat(old_r_up_carts[None, ...], 3, axis=0)
    new_r_dn_carts_arr = np.repeat(old_r_dn_carts[None, ...], 3, axis=0)

    up_alt_idx = min(1, old_r_up_carts.shape[0] - 1)
    dn_alt_idx = min(1, old_r_dn_carts.shape[0] - 1)

    if pattern == "all_moved":
        new_r_up_carts_arr[0, 0, 0] += 0.05
        new_r_dn_carts_arr[1, 0, 1] -= 0.07
        new_r_up_carts_arr[2, up_alt_idx, 2] += 0.09
    elif pattern == "none_moved":
        pass
    elif pattern == "mixed":
        new_r_up_carts_arr[0, 0, 0] += 0.05
        new_r_dn_carts_arr[2, dn_alt_idx, 1] += 0.04
    else:
        raise ValueError(f"Unknown pattern for ratio grid construction: {pattern}")

    return new_r_up_carts_arr, new_r_dn_carts_arr


@pytest.mark.parametrize("pattern", ["all_moved", "none_moved", "mixed"])
def test_ratio_determinant_rank1_update(pattern: str):
    """Rank-1 determinant ratio matches debug across three-grid scenarios."""
    geminal_data, old_r_up_carts, old_r_dn_carts, A_old_inv = _prepare_ratio_determinant_inputs()

    new_r_up_carts_arr, new_r_dn_carts_arr = _build_three_grid_moves_for_determinant(
        old_r_up_carts,
        old_r_dn_carts,
        pattern,
    )

    ratio_debug = _compute_ratio_determinant_part_debug(
        geminal_data=geminal_data,
        old_r_up_carts=old_r_up_carts,
        old_r_dn_carts=old_r_dn_carts,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )

    ratio_rank1 = _compute_ratio_determinant_part_rank1_update(
        geminal_data=geminal_data,
        A_old_inv=A_old_inv,
        old_r_up_carts=old_r_up_carts,
        old_r_dn_carts=old_r_dn_carts,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )

    assert not np.any(np.isnan(np.asarray(np.asarray(ratio_debug)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(ratio_rank1)))), "NaN detected in second argument"
    np.testing.assert_almost_equal(np.asarray(ratio_debug), np.asarray(ratio_rank1), decimal=decimal_debug_vs_production)

    if pattern == "none_moved":
        assert not np.any(np.isnan(np.asarray(np.asarray(ratio_debug)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.ones_like(np.asarray(ratio_debug))))), "NaN detected in second argument"
        np.testing.assert_array_almost_equal(
            np.asarray(ratio_debug), np.ones_like(np.asarray(ratio_debug)), decimal=decimal_consistency
        )

    jax.clear_caches()


@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H2_ecp_ccpvtz_cart.h5"])
def test_compute_ln_det_geminal_all_elements_fast_forward(trexio_file):
    """Forward value of compute_ln_det_geminal_all_elements_fast must match the standard variant."""
    _, _, _, _, geminal_data, _ = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file),
        store_tuple=True,
    )
    rng = np.random.default_rng(0)
    n_up = geminal_data.num_electron_up
    n_dn = geminal_data.num_electron_dn

    for _ in range(10):
        r_up = jnp.array(rng.standard_normal((n_up, 3)) * 1.2, dtype=jnp.float64)
        r_dn = jnp.array(rng.standard_normal((n_dn, 3)) * 1.2, dtype=jnp.float64)
        G = compute_geminal_all_elements(geminal_data, r_up, r_dn)
        G_inv = jnp.linalg.inv(G)

        val_ref = float(compute_ln_det_geminal_all_elements(geminal_data, r_up, r_dn))
        val_fast = float(compute_ln_det_geminal_all_elements_fast(geminal_data, r_up, r_dn, G_inv))

        assert np.isfinite(val_ref), f"Reference value is not finite: {val_ref}"
        assert np.isfinite(val_fast), f"Fast value is not finite: {val_fast}"
        np.testing.assert_almost_equal(
            val_fast,
            val_ref,
            decimal=decimal_debug_vs_production,
            err_msg=f"Forward mismatch: fast={val_fast:.15f}, ref={val_ref:.15f}",
        )

    jax.clear_caches()


@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H2_ecp_ccpvtz_cart.h5"])
def test_compute_ln_det_geminal_all_elements_fast_backward(trexio_file):
    """Gradient of compute_ln_det_geminal_all_elements_fast w.r.t. geminal_data must match the standard variant."""
    _, _, _, _, geminal_data, _ = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file),
        store_tuple=True,
    )
    rng = np.random.default_rng(1)
    n_up = geminal_data.num_electron_up
    n_dn = geminal_data.num_electron_dn

    grad_ref_fn = jax.grad(compute_ln_det_geminal_all_elements, argnums=0)
    grad_fast_fn = jax.grad(compute_ln_det_geminal_all_elements_fast, argnums=0)

    for _ in range(10):
        r_up = jnp.array(rng.standard_normal((n_up, 3)) * 1.2, dtype=jnp.float64)
        r_dn = jnp.array(rng.standard_normal((n_dn, 3)) * 1.2, dtype=jnp.float64)
        G = compute_geminal_all_elements(geminal_data, r_up, r_dn)
        G_inv = jnp.linalg.inv(G)

        grad_ref = grad_ref_fn(geminal_data, r_up, r_dn)
        grad_fast = grad_fast_fn(geminal_data, r_up, r_dn, G_inv)

        jax.tree_util.tree_map(
            lambda a, b: np.testing.assert_array_almost_equal(
                np.asarray(a),
                np.asarray(b),
                decimal=decimal_debug_vs_production,
                err_msg="Backward mismatch in compute_ln_det_geminal_all_elements_fast",
            ),
            grad_ref,
            grad_fast,
        )

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
