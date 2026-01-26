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

import sys
from pathlib import Path

import jax
import numpy as np

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.atomic_orbital import AOs_sphe_data  # noqa: E402
from jqmc.jastrow_factor import (  # noqa: E402
    Jastrow_data,
    Jastrow_NN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _compute_grads_and_laplacian_Jastrow_one_body_auto,
    _compute_grads_and_laplacian_Jastrow_one_body_debug,
    _compute_grads_and_laplacian_Jastrow_part_auto,
    _compute_grads_and_laplacian_Jastrow_part_debug,
    _compute_grads_and_laplacian_Jastrow_three_body_auto,
    _compute_grads_and_laplacian_Jastrow_three_body_debug,
    _compute_grads_and_laplacian_Jastrow_two_body_auto,
    _compute_grads_and_laplacian_Jastrow_two_body_debug,
    _compute_Jastrow_one_body_debug,
    _compute_Jastrow_three_body_debug,
    _compute_Jastrow_two_body_debug,
    _compute_ratio_Jastrow_part_debug,
    compute_grads_and_laplacian_Jastrow_one_body,
    compute_grads_and_laplacian_Jastrow_part,
    compute_grads_and_laplacian_Jastrow_three_body,
    compute_grads_and_laplacian_Jastrow_two_body,
    compute_Jastrow_one_body,
    compute_Jastrow_three_body,
    compute_Jastrow_two_body,
    compute_ratio_Jastrow_part,
)
from jqmc.molecular_orbital import MOs_data  # noqa: E402
from jqmc.setting import (  # noqa: E402
    atol_auto_vs_numerical_deriv,
    decimal_auto_vs_analytic_deriv,
    decimal_auto_vs_numerical_deriv,
    decimal_debug_vs_production,
    rtol_auto_vs_numerical_deriv,
)
from jqmc.structure import Structure_data  # noqa: E402


def test_Jastrow_onebody_part():
    """Test the three-body Jastrow factor, comparing the debug and JAX implementations, using AOs data."""
    num_r_up_cart_samples = 8
    num_r_dn_cart_samples = 4
    num_R_cart_samples = 6

    # generate matrices for the test
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([6] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )

    core_electrons = tuple([3] * num_R_cart_samples)

    jastrow_one_body_data = Jastrow_one_body_data(
        jastrow_1b_param=1.0, structure_data=structure_data, core_electrons=core_electrons
    )

    J1_debug = _compute_Jastrow_one_body_debug(
        jastrow_one_body_data=jastrow_one_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    J1_jax = compute_Jastrow_one_body(
        jastrow_one_body_data=jastrow_one_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    np.testing.assert_almost_equal(J1_debug, J1_jax, decimal=decimal_debug_vs_production)

    jax.clear_caches()


def test_numerical_and_auto_grads_Jastrow_onebody_part():
    """Test numerical and JAX grads of the one-body Jastrow factor."""
    num_r_up_cart_samples = 6
    num_r_dn_cart_samples = 3
    num_R_cart_samples = 5

    r_cart_min, r_cart_max = -1.5, 1.5
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([6] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )

    core_electrons = tuple([3] * num_R_cart_samples)
    jastrow_one_body_data = Jastrow_one_body_data(
        jastrow_1b_param=1.0, structure_data=structure_data, core_electrons=core_electrons
    )
    grad_up_num, grad_dn_num, lap_up_num, lap_dn_num = _compute_grads_and_laplacian_Jastrow_one_body_debug(
        jastrow_one_body_data,
        r_up_carts,
        r_dn_carts,
    )

    grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_one_body_auto(
        jastrow_one_body_data,
        r_up_carts,
        r_dn_carts,
    )

    np.testing.assert_almost_equal(np.asarray(grad_up_num), np.asarray(grad_up_auto), decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_almost_equal(np.asarray(grad_dn_num), np.asarray(grad_dn_auto), decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_allclose(
        np.asarray(lap_up_num),
        np.asarray(lap_up_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )
    np.testing.assert_allclose(
        np.asarray(lap_dn_num),
        np.asarray(lap_dn_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )

    jax.clear_caches()


def test_analytical_and_auto_grads_Jastrow_onebody_part():
    """Analytic vs auto-diff gradients/laplacian for one-body Jastrow."""
    num_r_up_cart_samples = 5
    num_r_dn_cart_samples = 4
    num_R_cart_samples = 4

    r_cart_min, r_cart_max = -2.0, 2.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([6] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )

    core_electrons = tuple([3] * num_R_cart_samples)
    jastrow_one_body_data = Jastrow_one_body_data(
        jastrow_1b_param=1.0, structure_data=structure_data, core_electrons=core_electrons
    )
    grad_up_an, grad_dn_an, lap_up_an, lap_dn_an = compute_grads_and_laplacian_Jastrow_one_body(
        jastrow_one_body_data,
        r_up_carts,
        r_dn_carts,
    )

    grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_one_body_auto(
        jastrow_one_body_data,
        r_up_carts,
        r_dn_carts,
    )

    np.testing.assert_almost_equal(np.asarray(grad_up_an), np.asarray(grad_up_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(grad_dn_an), np.asarray(grad_dn_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(lap_up_an), np.asarray(lap_up_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(lap_dn_an), np.asarray(lap_dn_auto), decimal=decimal_auto_vs_analytic_deriv)

    jax.clear_caches()


def test_Jastrow_twobody_part():
    """Test the two-body Jastrow factor, comparing the debug and JAX implementations."""
    num_r_up_cart_samples = 5
    num_r_dn_cart_samples = 2

    r_cart_min, r_cart_max = -3.0, 3.0

    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min

    jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=1.0)
    J2_debug = _compute_Jastrow_two_body_debug(
        jastrow_two_body_data=jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    # print(f"jastrow_two_body_debug = {jastrow_two_body_debug}")

    J2_jax = compute_Jastrow_two_body(jastrow_two_body_data=jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

    # print(f"jastrow_two_body_jax = {jastrow_two_body_jax}")

    np.testing.assert_almost_equal(J2_debug, J2_jax, decimal=decimal_debug_vs_production)

    jax.clear_caches()


def test_Jastrow_threebody_part_with_AOs_data():
    """Test the three-body Jastrow factor, comparing the debug and JAX implementations, using AOs data."""
    num_r_up_cart_samples = 4
    num_r_dn_cart_samples = 2
    num_R_cart_samples = 6
    num_ao = 6
    num_ao_prim = 6
    orbital_indices = [0, 1, 2, 3, 4, 5]
    exponents = [1.2, 0.5, 0.1, 0.05, 0.05, 0.05]
    coefficients = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    angular_momentums = [0, 0, 0, 1, 1, 1]
    magnetic_quantum_numbers = [0, 0, 0, 0, +1, -1]

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    # generate matrices for the test
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )

    aos_data = AOs_sphe_data(
        structure_data=structure_data,
        nucleus_index=tuple(list(range(num_R_cart_samples))),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    j_matrix = np.random.rand(aos_data.num_ao, aos_data.num_ao + 1)

    jastrow_three_body_data = Jastrow_three_body_data(orb_data=aos_data, j_matrix=j_matrix)

    J3_debug = _compute_Jastrow_three_body_debug(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # print(f"J3_debug = {J3_debug}")

    J3_jax = compute_Jastrow_three_body(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # print(f"J3_jax = {J3_jax}")

    np.testing.assert_almost_equal(J3_debug, J3_jax, decimal=decimal_debug_vs_production)

    jax.clear_caches()


def test_Jastrow_threebody_part_with_MOs_data():
    """Test the three-body Jastrow factor, comparing the debug and JAX implementations, using MOs data."""
    num_el = 10
    num_mo = 5
    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 0, 1, 2]
    exponents = [50.0, 20.0, 10.0, 5.0]
    coefficients = [1.0, 1.0, 1.0, 0.5]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 0, -1]

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    num_r_up_cart_samples = num_r_dn_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -5.0, 5.0
    R_cart_min, R_cart_max = 10.0, 10.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    mo_coefficients = np.random.rand(num_mo, num_ao)

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )

    aos_data = AOs_sphe_data(
        structure_data=structure_data,
        nucleus_index=tuple(list(range(num_R_cart_samples))),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    j_matrix = np.random.rand(mos_data.num_mo, mos_data.num_mo + 1)

    jastrow_three_body_data = Jastrow_three_body_data(orb_data=mos_data, j_matrix=j_matrix)

    J3_debug = _compute_Jastrow_three_body_debug(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # print(f"J3_debug = {J3_debug}")

    J3_jax = compute_Jastrow_three_body(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # print(f"J3_jax = {J3_jax}")

    np.testing.assert_almost_equal(J3_debug, J3_jax, decimal=decimal_debug_vs_production)

    jax.clear_caches()


def test_numerical_and_auto_grads_Jastrow_threebody_part_with_AOs_data():
    """Test numerical and JAX grads of the three-body Jastrow factor, comparing the debug and JAX implementations, using AOs data."""
    num_r_up_cart_samples = 4
    num_r_dn_cart_samples = 2
    num_R_cart_samples = 6
    num_ao = 6
    num_ao_prim = 6
    orbital_indices = [0, 1, 2, 3, 4, 5]
    exponents = [1.2, 0.5, 0.1, 0.05, 0.05, 0.05]
    coefficients = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    angular_momentums = [0, 0, 0, 1, 1, 1]
    magnetic_quantum_numbers = [0, 0, 0, 0, +1, -1]

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    # generate matrices for the test
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )

    aos_data = AOs_sphe_data(
        structure_data=structure_data,
        nucleus_index=tuple(list(range(num_R_cart_samples))),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    j_matrix = np.random.rand(aos_data.num_ao, aos_data.num_ao + 1)

    jastrow_three_body_data = Jastrow_three_body_data(orb_data=aos_data, j_matrix=j_matrix)

    J3_debug = _compute_Jastrow_three_body_debug(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    J3_jax = compute_Jastrow_three_body(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    np.testing.assert_almost_equal(J3_debug, J3_jax, decimal=decimal_debug_vs_production)

    (
        grad_jastrow_J3_up_debug,
        grad_jastrow_J3_dn_debug,
        lap_J3_up_debug,
        lap_J3_dn_debug,
    ) = _compute_grads_and_laplacian_Jastrow_three_body_debug(
        jastrow_three_body_data,
        r_up_carts,
        r_dn_carts,
    )

    grad_jastrow_J3_up_auto, grad_jastrow_J3_dn_auto, lap_J3_up_auto, lap_J3_dn_auto = (
        _compute_grads_and_laplacian_Jastrow_three_body_auto(
            jastrow_three_body_data,
            r_up_carts,
            r_dn_carts,
        )
    )

    np.testing.assert_almost_equal(grad_jastrow_J3_up_debug, grad_jastrow_J3_up_auto, decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_almost_equal(grad_jastrow_J3_dn_debug, grad_jastrow_J3_dn_auto, decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_allclose(
        np.asarray(lap_J3_up_debug),
        np.asarray(lap_J3_up_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )
    np.testing.assert_allclose(
        np.asarray(lap_J3_dn_debug),
        np.asarray(lap_J3_dn_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )

    jax.clear_caches()


def test_numerical_and_auto_grads_Jastrow_threebody_part_with_MOs_data():
    """Test numerical and JAX grads of the three-body Jastrow factor, comparing the debug and JAX implementations, using MOs data."""
    num_el = 10
    num_mo = 5
    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 0, 1, 2]
    exponents = [50.0, 20.0, 10.0, 5.0]
    coefficients = [1.0, 1.0, 1.0, 0.5]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 0, -1]

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    num_r_up_cart_samples = num_r_dn_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -5.0, 5.0
    R_cart_min, R_cart_max = 10.0, 10.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    mo_coefficients = np.random.rand(num_mo, num_ao)

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )

    aos_data = AOs_sphe_data(
        structure_data=structure_data,
        nucleus_index=tuple(list(range(num_R_cart_samples))),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    j_matrix = np.random.rand(mos_data.num_mo, mos_data.num_mo + 1)

    jastrow_three_body_data = Jastrow_three_body_data(orb_data=mos_data, j_matrix=j_matrix)

    J3_debug = _compute_Jastrow_three_body_debug(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    J3_jax = compute_Jastrow_three_body(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    np.testing.assert_almost_equal(J3_debug, J3_jax, decimal=decimal_debug_vs_production)

    (
        grad_jastrow_J3_up_debug,
        grad_jastrow_J3_dn_debug,
        lap_J3_up_debug,
        lap_J3_dn_debug,
    ) = _compute_grads_and_laplacian_Jastrow_three_body_debug(
        jastrow_three_body_data,
        r_up_carts,
        r_dn_carts,
    )

    grad_jastrow_J3_up_jax, grad_jastrow_J3_dn_jax, lap_J3_up_jax, lap_J3_dn_jax = (
        _compute_grads_and_laplacian_Jastrow_three_body_auto(
            jastrow_three_body_data,
            r_up_carts,
            r_dn_carts,
        )
    )

    np.testing.assert_almost_equal(grad_jastrow_J3_up_debug, grad_jastrow_J3_up_jax, decimal=decimal_debug_vs_production)
    np.testing.assert_almost_equal(grad_jastrow_J3_dn_debug, grad_jastrow_J3_dn_jax, decimal=decimal_debug_vs_production)

    np.testing.assert_allclose(
        np.asarray(lap_J3_up_debug),
        np.asarray(lap_J3_up_jax),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )
    np.testing.assert_allclose(
        np.asarray(lap_J3_dn_debug),
        np.asarray(lap_J3_dn_jax),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )

    jax.clear_caches()


def test_numerical_and_auto_grads_Jastrow_twobody_part():
    """Test numerical and JAX grads of the two-body Jastrow factor, comparing the debug and JAX implementations."""
    num_r_up_cart_samples = 5
    num_r_dn_cart_samples = 2

    r_cart_min, r_cart_max = -3.0, 3.0

    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min

    jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=1.0)
    J2_debug = _compute_Jastrow_two_body_debug(
        jastrow_two_body_data=jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    # print(f"jastrow_two_body_debug = {jastrow_two_body_debug}")

    J2_jax = compute_Jastrow_two_body(jastrow_two_body_data=jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

    # print(f"jastrow_two_body_jax = {jastrow_two_body_jax}")

    np.testing.assert_almost_equal(J2_debug, J2_jax, decimal=decimal_debug_vs_production)

    (
        grad_J2_up_debug,
        grad_J2_dn_debug,
        lap_J2_up_debug,
        lap_J2_dn_debug,
    ) = _compute_grads_and_laplacian_Jastrow_two_body_debug(
        jastrow_two_body_data,
        r_up_carts,
        r_dn_carts,
    )

    # print(f"grad_J2_up_debug = {grad_J2_up_debug}")
    # print(f"grad_J2_dn_debug = {grad_J2_dn_debug}")
    # print(f"sum_laplacian_J2_debug = {sum_laplacian_J2_debug}")

    grad_J2_up_auto, grad_J2_dn_auto, lap_J2_up_auto, lap_J2_dn_auto = _compute_grads_and_laplacian_Jastrow_two_body_auto(
        jastrow_two_body_data,
        r_up_carts,
        r_dn_carts,
    )

    # print(f"grad_J2_up_jax = {grad_J2_up_jax}")
    # print(f"grad_J2_dn_jax = {grad_J2_dn_jax}")
    # print(f"sum_laplacian_J2_jax = {sum_laplacian_J2_jax}")

    np.testing.assert_almost_equal(grad_J2_up_debug, grad_J2_up_auto, decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_almost_equal(grad_J2_dn_debug, grad_J2_dn_auto, decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_allclose(
        np.asarray(lap_J2_up_debug),
        np.asarray(lap_J2_up_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )
    np.testing.assert_allclose(
        np.asarray(lap_J2_dn_debug),
        np.asarray(lap_J2_dn_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )

    jax.clear_caches()


def test_analytic_and_auto_grads_Jastrow_threebody_part_with_AOs_data():
    """Analytic vs auto-diff gradients/laplacian for three-body Jastrow (AOs)."""
    num_r_up_cart_samples = 4
    num_r_dn_cart_samples = 2
    num_R_cart_samples = 5
    num_ao = 5
    num_ao_prim = 5
    orbital_indices = [0, 1, 2, 3, 4]
    exponents = [1.2, 0.6, 0.2, 0.1, 0.05]
    coefficients = [1.0, 0.9, 1.0, 1.0, 1.0]
    angular_momentums = [0, 0, 0, 1, 1]
    magnetic_quantum_numbers = [0, 0, 0, 0, 1]

    r_cart_min, r_cart_max = -1.5, 1.5
    R_cart_min, R_cart_max = -0.5, 0.5
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )

    aos_data = AOs_sphe_data(
        structure_data=structure_data,
        nucleus_index=tuple(list(range(num_R_cart_samples))),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=tuple(orbital_indices),
        exponents=tuple(exponents),
        coefficients=tuple(coefficients),
        angular_momentums=tuple(angular_momentums),
        magnetic_quantum_numbers=tuple(magnetic_quantum_numbers),
    )

    j_matrix = np.random.rand(aos_data.num_ao, aos_data.num_ao + 1)
    jastrow_three_body_data = Jastrow_three_body_data(orb_data=aos_data, j_matrix=j_matrix)

    grad_up_an, grad_dn_an, lap_up_an, lap_dn_an = compute_grads_and_laplacian_Jastrow_three_body(
        jastrow_three_body_data,
        r_up_carts,
        r_dn_carts,
    )

    grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_three_body_auto(
        jastrow_three_body_data,
        r_up_carts,
        r_dn_carts,
    )

    np.testing.assert_almost_equal(np.asarray(grad_up_an), np.asarray(grad_up_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(grad_dn_an), np.asarray(grad_dn_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(lap_up_an), np.asarray(lap_up_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(lap_dn_an), np.asarray(lap_dn_auto), decimal=decimal_auto_vs_analytic_deriv)

    jax.clear_caches()


def test_analytic_and_auto_grads_Jastrow_threebody_part_with_MOs_data():
    """Analytic vs auto-diff gradients/laplacian for three-body Jastrow (MOs)."""
    num_el = 8
    num_mo = 4
    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 0, 1, 2]
    exponents = [10.0, 6.0, 3.0, 1.5]
    coefficients = [1.0, 1.0, 0.8, 0.7]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 1, -1]

    r_cart_min, r_cart_max = -2.0, 2.0
    R_cart_min, R_cart_max = -1.0, 1.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_el, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_el, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_ao, 3) + R_cart_min

    mo_coefficients = np.random.rand(num_mo, num_ao)

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_ao),
        element_symbols=tuple(["X"] * num_ao),
        atomic_labels=tuple(["X"] * num_ao),
    )

    aos_data = AOs_sphe_data(
        structure_data=structure_data,
        nucleus_index=tuple(list(range(num_ao))),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=tuple(orbital_indices),
        exponents=tuple(exponents),
        coefficients=tuple(coefficients),
        angular_momentums=tuple(angular_momentums),
        magnetic_quantum_numbers=tuple(magnetic_quantum_numbers),
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    j_matrix = np.random.rand(mos_data.num_mo, mos_data.num_mo + 1)
    jastrow_three_body_data = Jastrow_three_body_data(orb_data=mos_data, j_matrix=j_matrix)

    grad_up_an, grad_dn_an, lap_up_an, lap_dn_an = compute_grads_and_laplacian_Jastrow_three_body(
        jastrow_three_body_data,
        r_up_carts,
        r_dn_carts,
    )

    grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_three_body_auto(
        jastrow_three_body_data,
        r_up_carts,
        r_dn_carts,
    )

    np.testing.assert_almost_equal(np.asarray(grad_up_an), np.asarray(grad_up_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(grad_dn_an), np.asarray(grad_dn_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(lap_up_an), np.asarray(lap_up_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(lap_dn_an), np.asarray(lap_dn_auto), decimal=decimal_auto_vs_analytic_deriv)

    jax.clear_caches()


def test_analytic_and_auto_grads_Jastrow_twobody_part():
    """Analytic vs auto-diff gradients/laplacian for two-body Jastrow (Pade)."""
    num_r_up_cart_samples = 5
    num_r_dn_cart_samples = 2

    r_cart_min, r_cart_max = -3.0, 3.0

    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min

    jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=1.0)

    grad_J2_up_analytic, grad_J2_dn_analytic, lap_J2_up_analytic, lap_J2_dn_analytic = (
        compute_grads_and_laplacian_Jastrow_two_body(
            jastrow_two_body_data,
            r_up_carts,
            r_dn_carts,
        )
    )

    grad_J2_up_auto, grad_J2_dn_auto, lap_J2_up_auto, lap_J2_dn_auto = _compute_grads_and_laplacian_Jastrow_two_body_auto(
        jastrow_two_body_data,
        r_up_carts,
        r_dn_carts,
    )

    np.testing.assert_almost_equal(
        np.asarray(grad_J2_up_analytic), np.asarray(grad_J2_up_auto), decimal=decimal_auto_vs_analytic_deriv
    )
    np.testing.assert_almost_equal(
        np.asarray(grad_J2_dn_analytic), np.asarray(grad_J2_dn_auto), decimal=decimal_auto_vs_analytic_deriv
    )
    np.testing.assert_almost_equal(
        np.asarray(lap_J2_up_analytic), np.asarray(lap_J2_up_auto), decimal=decimal_auto_vs_analytic_deriv
    )
    np.testing.assert_almost_equal(
        np.asarray(lap_J2_dn_analytic), np.asarray(lap_J2_dn_auto), decimal=decimal_auto_vs_analytic_deriv
    )

    jax.clear_caches()


def _build_jastrow_data_for_part_tests(include_nn: bool):
    num_r_up_cart_samples = 4
    num_r_dn_cart_samples = 3
    num_R_cart_samples = 4
    num_ao = 4
    num_ao_prim = 4

    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([6] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )

    core_electrons = tuple([3] * num_R_cart_samples)
    jastrow_one_body_data = Jastrow_one_body_data(
        jastrow_1b_param=1.0, structure_data=structure_data, core_electrons=core_electrons
    )

    jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=1.0)

    orbital_indices = (0, 1, 2, 3)
    exponents = (1.2, 0.5, 0.1, 0.05)
    coefficients = (1.0, 1.0, 1.0, 1.0)
    angular_momentums = (0, 0, 1, 1)
    magnetic_quantum_numbers = (0, 0, 1, -1)

    aos_data = AOs_sphe_data(
        structure_data=structure_data,
        nucleus_index=tuple(list(range(num_R_cart_samples))),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    j_matrix = np.random.rand(aos_data.num_ao, aos_data.num_ao + 1)
    jastrow_three_body_data = Jastrow_three_body_data(orb_data=aos_data, j_matrix=j_matrix)

    jastrow_nn_data = None
    if include_nn:
        jastrow_nn_data = Jastrow_NN_data.init_from_structure(
            structure_data=structure_data,
            hidden_dim=16,
            num_layers=2,
            num_rbf=8,
            cutoff=5.0,
            key=jax.random.PRNGKey(0),
        )

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_one_body_data,
        jastrow_two_body_data=jastrow_two_body_data,
        jastrow_three_body_data=jastrow_three_body_data,
        jastrow_nn_data=jastrow_nn_data,
    )

    return jastrow_data, r_up_carts, r_dn_carts


def test_numerical_and_auto_grads_Jastrow_part():
    """Numerical vs auto-diff gradients/laplacian for J1+J2+J3."""
    jastrow_data, r_up_carts, r_dn_carts = _build_jastrow_data_for_part_tests(include_nn=False)

    grad_up_num, grad_dn_num, lap_up_num, lap_dn_num = _compute_grads_and_laplacian_Jastrow_part_debug(
        jastrow_data,
        r_up_carts,
        r_dn_carts,
    )

    grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_part_auto(
        jastrow_data,
        r_up_carts,
        r_dn_carts,
    )

    np.testing.assert_almost_equal(np.asarray(grad_up_num), np.asarray(grad_up_auto), decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_almost_equal(np.asarray(grad_dn_num), np.asarray(grad_dn_auto), decimal=decimal_auto_vs_numerical_deriv)

    np.testing.assert_allclose(
        np.asarray(lap_up_num),
        np.asarray(lap_up_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )
    np.testing.assert_allclose(
        np.asarray(lap_dn_num),
        np.asarray(lap_dn_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )

    jax.clear_caches()


def test_numerical_and_auto_grads_Jastrow_part_with_NN():
    """Numerical vs auto-diff gradients/laplacian for J1+J2+J3+JNN."""
    jastrow_data, r_up_carts, r_dn_carts = _build_jastrow_data_for_part_tests(include_nn=True)

    grad_up_num, grad_dn_num, lap_up_num, lap_dn_num = _compute_grads_and_laplacian_Jastrow_part_debug(
        jastrow_data,
        r_up_carts,
        r_dn_carts,
    )

    grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_part_auto(
        jastrow_data,
        r_up_carts,
        r_dn_carts,
    )

    np.testing.assert_almost_equal(np.asarray(grad_up_num), np.asarray(grad_up_auto), decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_almost_equal(np.asarray(grad_dn_num), np.asarray(grad_dn_auto), decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_allclose(
        np.asarray(lap_up_num),
        np.asarray(lap_up_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )
    np.testing.assert_allclose(
        np.asarray(lap_dn_num),
        np.asarray(lap_dn_auto),
        rtol=rtol_auto_vs_numerical_deriv,
        atol=atol_auto_vs_numerical_deriv,
    )

    jax.clear_caches()


def test_analytical_and_auto_grads_Jastrow_part():
    """Analytic vs auto-diff gradients/laplacian for J1+J2+J3."""
    jastrow_data, r_up_carts, r_dn_carts = _build_jastrow_data_for_part_tests(include_nn=False)

    grad_up_an, grad_dn_an, lap_up_an, lap_dn_an = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data,
        r_up_carts,
        r_dn_carts,
    )

    grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_part_auto(
        jastrow_data,
        r_up_carts,
        r_dn_carts,
    )

    np.testing.assert_almost_equal(np.asarray(grad_up_an), np.asarray(grad_up_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(grad_dn_an), np.asarray(grad_dn_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(lap_up_an), np.asarray(lap_up_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(lap_dn_an), np.asarray(lap_dn_auto), decimal=decimal_auto_vs_analytic_deriv)

    jax.clear_caches()


def test_analytical_and_auto_grads_Jastrow_part_with_NN():
    """Analytic (J1+J2+J3) + auto (JNN) vs auto (full)."""
    jastrow_data, r_up_carts, r_dn_carts = _build_jastrow_data_for_part_tests(include_nn=True)

    grad_up_an, grad_dn_an, lap_up_an, lap_dn_an = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data,
        r_up_carts,
        r_dn_carts,
    )

    grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_part_auto(
        jastrow_data,
        r_up_carts,
        r_dn_carts,
    )

    np.testing.assert_almost_equal(np.asarray(grad_up_an), np.asarray(grad_up_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(grad_dn_an), np.asarray(grad_dn_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(lap_up_an), np.asarray(lap_up_auto), decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_almost_equal(np.asarray(lap_dn_an), np.asarray(lap_dn_auto), decimal=decimal_auto_vs_analytic_deriv)

    jax.clear_caches()


def _build_ratio_grids(r_up_carts: np.ndarray, r_dn_carts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    new_r_up_carts_arr = []
    new_r_dn_carts_arr = []
    for i in range(len(r_up_carts)):
        for axis in range(3):
            new_r_up = r_up_carts.copy()
            new_r_dn = r_dn_carts.copy()
            new_r_up[i, axis] += 0.05 * new_r_up[i, axis]
            new_r_up_carts_arr.append(new_r_up)
            new_r_dn_carts_arr.append(new_r_dn)
            new_r_up = r_up_carts.copy()
            new_r_dn = r_dn_carts.copy()
            new_r_up[i, axis] -= 0.05 * new_r_up[i, axis]
            new_r_up_carts_arr.append(new_r_up)
            new_r_dn_carts_arr.append(new_r_dn)

    for i in range(len(r_dn_carts)):
        for axis in range(3):
            new_r_up = r_up_carts.copy()
            new_r_dn = r_dn_carts.copy()
            new_r_dn[i, axis] += 0.05 * new_r_dn[i, axis]
            new_r_up_carts_arr.append(new_r_up)
            new_r_dn_carts_arr.append(new_r_dn)
            new_r_up = r_up_carts.copy()
            new_r_dn = r_dn_carts.copy()
            new_r_dn[i, axis] -= 0.05 * new_r_dn[i, axis]
            new_r_up_carts_arr.append(new_r_up)
            new_r_dn_carts_arr.append(new_r_dn)

    return np.array(new_r_up_carts_arr), np.array(new_r_dn_carts_arr)


def test_ratio_Jastrow_part_debug():
    """Compare ratio Jastrow part: debug vs auto implementation (no NN)."""
    jastrow_data, r_up_carts, r_dn_carts = _build_jastrow_data_for_part_tests(include_nn=False)

    old_r_up_carts = r_up_carts
    old_r_dn_carts = r_dn_carts

    new_r_up_carts_arr, new_r_dn_carts_arr = _build_ratio_grids(old_r_up_carts, old_r_dn_carts)

    ratio_debug = _compute_ratio_Jastrow_part_debug(
        jastrow_data,
        old_r_up_carts=old_r_up_carts,
        old_r_dn_carts=old_r_dn_carts,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )

    ratio_auto = compute_ratio_Jastrow_part(
        jastrow_data,
        old_r_up_carts=old_r_up_carts,
        old_r_dn_carts=old_r_dn_carts,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )

    np.testing.assert_almost_equal(np.asarray(ratio_debug), np.asarray(ratio_auto), decimal=decimal_debug_vs_production)

    jax.clear_caches()


def test_ratio_Jastrow_part_debug_with_NN():
    """Compare ratio Jastrow part: debug vs auto implementation (with NN)."""
    jastrow_data, r_up_carts, r_dn_carts = _build_jastrow_data_for_part_tests(include_nn=True)

    old_r_up_carts = r_up_carts
    old_r_dn_carts = r_dn_carts

    new_r_up_carts_arr, new_r_dn_carts_arr = _build_ratio_grids(old_r_up_carts, old_r_dn_carts)

    ratio_debug = _compute_ratio_Jastrow_part_debug(
        jastrow_data,
        old_r_up_carts=old_r_up_carts,
        old_r_dn_carts=old_r_dn_carts,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )

    ratio_auto = compute_ratio_Jastrow_part(
        jastrow_data,
        old_r_up_carts=old_r_up_carts,
        old_r_dn_carts=old_r_dn_carts,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )

    np.testing.assert_almost_equal(np.asarray(ratio_debug), np.asarray(ratio_auto), decimal=decimal_debug_vs_production)

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
