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

from jqmc.determinant import (  # noqa: E402
    Geminal_data,
    _compute_AS_regularization_factor_debug,
    _compute_det_geminal_all_elements_debug,
    _compute_geminal_all_elements,
    _compute_geminal_all_elements_debug,
    _compute_grads_and_laplacian_ln_Det_auto,
    _compute_grads_and_laplacian_ln_Det_debug,
    _compute_grads_and_laplacian_ln_Det_fast_debug,
    compute_AS_regularization_factor,
    compute_det_geminal_all_elements,
    compute_geminal_all_elements,
    compute_geminal_dn_one_column_elements,
    compute_geminal_up_one_row_elements,
    compute_grads_and_laplacian_ln_Det,
    compute_grads_and_laplacian_ln_Det_fast,
)
from jqmc.setting import (  # noqa: E402
    atol_auto_vs_numerical_deriv,
    decimal_auto_vs_analytic_deriv,
    decimal_auto_vs_numerical_deriv,
    decimal_debug_vs_production,
    rtol_auto_vs_numerical_deriv,
)
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def test_comparing_AO_and_MO_geminals():
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

    np.testing.assert_almost_equal(geminal_ao_debug, geminal_ao_jax, decimal=decimal_debug_vs_production)

    geminal_ao = geminal_ao_jax

    # check if geminals with AO and MO representations are consistent
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

    np.testing.assert_array_almost_equal(det_geminal_ao_debug, det_geminal_ao_jax, decimal=decimal_debug_vs_production)
    det_geminal_ao = det_geminal_ao_jax

    np.testing.assert_almost_equal(det_geminal_ao, det_geminal_mo, decimal=decimal_debug_vs_production)

    jax.clear_caches()


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

    np.testing.assert_array_almost_equal(grad_up_fast, grad_up_debug, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(grad_dn_fast, grad_dn_debug, decimal=decimal_debug_vs_production)
    np.testing.assert_array_almost_equal(lap_up_fast, lap_up_debug, decimal=decimal_debug_vs_production)
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
    np.testing.assert_array_almost_equal(
        np.asarray(geminal_mo_up_one_row).ravel(),
        np.asarray(geminal_mo[i_up, :]),
        decimal=decimal_debug_vs_production,
    )

    # dn-one-column must equal the j-th *paired* column of the full geminal
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

    np.testing.assert_array_almost_equal(
        np.asarray(grad_ln_D_up_numerical),
        np.asarray(grad_ln_D_up_auto),
        decimal=decimal_auto_vs_numerical_deriv,
    )
    np.testing.assert_array_almost_equal(
        np.asarray(grad_ln_D_dn_numerical),
        np.asarray(grad_ln_D_dn_auto),
        decimal=decimal_auto_vs_numerical_deriv,
    )
    np.testing.assert_allclose(
        np.asarray(lap_ln_D_up_numerical),
        np.asarray(lap_ln_D_up_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )
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

    np.testing.assert_array_almost_equal(
        np.asarray(grad_ln_D_up_analytic),
        np.asarray(grad_ln_D_up_auto),
        decimal=decimal_auto_vs_analytic_deriv,
    )
    np.testing.assert_array_almost_equal(
        np.asarray(grad_ln_D_dn_analytic),
        np.asarray(grad_ln_D_dn_auto),
        decimal=decimal_auto_vs_analytic_deriv,
    )
    np.testing.assert_array_almost_equal(
        np.asarray(lap_ln_D_up_analytic),
        np.asarray(lap_ln_D_up_auto),
        decimal=decimal_auto_vs_analytic_deriv,
    )
    np.testing.assert_array_almost_equal(
        np.asarray(lap_ln_D_dn_analytic),
        np.asarray(lap_ln_D_dn_auto),
        decimal=decimal_auto_vs_analytic_deriv,
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
