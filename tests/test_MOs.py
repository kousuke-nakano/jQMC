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

# Add the project root directory to sys.path to allow executing this script directly
# This is necessary because relative imports (e.g. 'from ..jqmc') are not allowed
# when running a script directly (as __main__).
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.atomic_orbital import (  # noqa: E402
    AOs_cart_data,
    AOs_sphe_data,
)
from jqmc.molecular_orbital import (  # noqa: E402
    MOs_data,
    _cart_to_spherical_matrix,
    _compute_MOs_debug,
    _compute_MOs_grad_autodiff,
    _compute_MOs_grad_debug,
    _compute_MOs_laplacian_autodiff,
    _compute_MOs_laplacian_debug,
    compute_MOs,
    compute_MOs_grad,
    compute_MOs_laplacian,
)
from jqmc.setting import (  # noqa: E402
    decimal_auto_vs_analytic_deriv,
    decimal_auto_vs_numerical_deriv,
    decimal_debug_vs_production,
)
from jqmc.structure import Structure_data  # noqa: E402

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def test_MOs_comparing_jax_and_debug_implemenetations():
    """Test the MO computation, comparing JAX and debug implementations."""
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

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -5.0, 5.0
    R_cart_min, R_cart_max = -6.0, 6.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
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

    mos_data.sanity_check()

    mo_ans_all_jax = compute_MOs(mos_data=mos_data, r_carts=r_carts)

    mo_ans_all_debug = _compute_MOs_debug(mos_data=mos_data, r_carts=r_carts)

    assert not np.any(np.isnan(np.asarray(mo_ans_all_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mo_ans_all_jax))), "NaN detected in second argument"
    np.testing.assert_almost_equal(mo_ans_all_debug, mo_ans_all_jax, decimal=decimal_debug_vs_production)

    num_el = 10
    num_mo = 5
    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 0, 1, 2]
    exponents = [10.0, 5.0, 1.0, 1.0]
    coefficients = [1.0, 1.0, 1.0, 0.5]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 1, -1]

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -5.0, 5.0
    R_cart_min, R_cart_max = -6.0, 6.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
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
    mos_data.sanity_check()

    mo_ans_all_jax = compute_MOs(mos_data=mos_data, r_carts=r_carts)

    mo_ans_all_debug = _compute_MOs_debug(mos_data=mos_data, r_carts=r_carts)

    assert not np.any(np.isnan(np.asarray(mo_ans_all_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mo_ans_all_jax))), "NaN detected in second argument"
    np.testing.assert_almost_equal(mo_ans_all_debug, mo_ans_all_jax, decimal=decimal_debug_vs_production)

    jax.clear_caches()


def test_MOs_comparing_auto_and_numerical_grads():
    """Test the MO gradient computation, comparing JAX and debug implementations."""
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

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -5.0, 5.0
    R_cart_min, R_cart_max = 10.0, 10.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
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
    mos_data.sanity_check()

    mo_matrix_grad_x_auto, mo_matrix_grad_y_auto, mo_matrix_grad_z_auto = compute_MOs_grad(mos_data=mos_data, r_carts=r_carts)

    (
        mo_matrix_grad_x_numerical,
        mo_matrix_grad_y_numerical,
        mo_matrix_grad_z_numerical,
    ) = _compute_MOs_grad_autodiff(mos_data=mos_data, r_carts=r_carts)

    assert not np.any(np.isnan(np.asarray(mo_matrix_grad_x_auto))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mo_matrix_grad_x_numerical))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        mo_matrix_grad_x_auto, mo_matrix_grad_x_numerical, decimal=decimal_auto_vs_numerical_deriv
    )
    assert not np.any(np.isnan(np.asarray(mo_matrix_grad_y_auto))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mo_matrix_grad_y_numerical))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        mo_matrix_grad_y_auto, mo_matrix_grad_y_numerical, decimal=decimal_auto_vs_numerical_deriv
    )

    assert not np.any(np.isnan(np.asarray(mo_matrix_grad_z_auto))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mo_matrix_grad_z_numerical))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        mo_matrix_grad_z_auto, mo_matrix_grad_z_numerical, decimal=decimal_auto_vs_numerical_deriv
    )

    num_el = 10
    num_mo = 5
    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 0, 1, 2]
    exponents = [10.0, 5.0, 1.0, 1.0]
    coefficients = [1.0, 1.0, 1.0, 0.5]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 1, -1]

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 3.0, 3.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
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
    mos_data.sanity_check()

    mo_matrix_grad_x_auto, mo_matrix_grad_y_auto, mo_matrix_grad_z_auto = compute_MOs_grad(mos_data=mos_data, r_carts=r_carts)

    (
        mo_matrix_grad_x_numerical,
        mo_matrix_grad_y_numerical,
        mo_matrix_grad_z_numerical,
    ) = _compute_MOs_grad_debug(mos_data=mos_data, r_carts=r_carts)

    assert not np.any(np.isnan(np.asarray(mo_matrix_grad_x_auto))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mo_matrix_grad_x_numerical))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        mo_matrix_grad_x_auto, mo_matrix_grad_x_numerical, decimal=decimal_auto_vs_numerical_deriv
    )
    assert not np.any(np.isnan(np.asarray(mo_matrix_grad_y_auto))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mo_matrix_grad_y_numerical))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        mo_matrix_grad_y_auto, mo_matrix_grad_y_numerical, decimal=decimal_auto_vs_numerical_deriv
    )
    assert not np.any(np.isnan(np.asarray(mo_matrix_grad_z_auto))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mo_matrix_grad_z_numerical))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        mo_matrix_grad_z_auto, mo_matrix_grad_z_numerical, decimal=decimal_auto_vs_numerical_deriv
    )

    jax.clear_caches()


def test_MOs_comparing_auto_and_numerical_laplacians():
    """Test the MO Laplacian computation, comparing JAX and debug implementations."""
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

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -5.0, 5.0
    R_cart_min, R_cart_max = 10.0, 10.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
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
    mos_data.sanity_check()

    mo_matrix_laplacian_numerical = _compute_MOs_laplacian_debug(mos_data=mos_data, r_carts=r_carts)

    mo_matrix_laplacian_auto = _compute_MOs_laplacian_autodiff(mos_data=mos_data, r_carts=r_carts)

    assert not np.any(np.isnan(np.asarray(mo_matrix_laplacian_auto))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mo_matrix_laplacian_numerical))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(
        mo_matrix_laplacian_auto, mo_matrix_laplacian_numerical, decimal=decimal_auto_vs_numerical_deriv
    )

    jax.clear_caches()


def test_MOs_comparing_analytic_and_auto_grads():
    """Test analytic MO gradients vs autodiff grads."""
    num_el = 8
    num_mo = 4
    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 0, 1, 2]
    exponents = [20.0, 10.0, 5.0, 2.0]
    coefficients = [1.0, 0.8, 1.1, 0.7]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 1, -1]

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    r_cart_min, r_cart_max = -2.0, 2.0
    R_cart_min, R_cart_max = -1.0, 1.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_el, 3) + r_cart_min
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
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
    mos_data.sanity_check()

    grad_x_an, grad_y_an, grad_z_an = compute_MOs_grad(mos_data=mos_data, r_carts=r_carts)

    grad_x_auto, grad_y_auto, grad_z_auto = _compute_MOs_grad_autodiff(mos_data=mos_data, r_carts=r_carts)

    assert not np.any(np.isnan(np.asarray(grad_x_an))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(grad_x_auto))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(grad_x_an, grad_x_auto, decimal=decimal_auto_vs_analytic_deriv)
    assert not np.any(np.isnan(np.asarray(grad_y_an))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(grad_y_auto))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(grad_y_an, grad_y_auto, decimal=decimal_auto_vs_analytic_deriv)
    assert not np.any(np.isnan(np.asarray(grad_z_an))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(grad_z_auto))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(grad_z_an, grad_z_auto, decimal=decimal_auto_vs_analytic_deriv)

    jax.clear_caches()


def test_MOs_comparing_analytic_and_auto_laplacians():
    """Test analytic MO Laplacian vs autodiff Laplacian."""
    num_el = 8
    num_mo = 4
    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 0, 1, 2]
    exponents = [15.0, 8.0, 4.0, 2.5]
    coefficients = [1.0, 0.9, 0.6, 0.7]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 1, -1]

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    r_cart_min, r_cart_max = -2.0, 2.0
    R_cart_min, R_cart_max = -1.0, 1.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_el, 3) + r_cart_min
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
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
    mos_data.sanity_check()

    mo_lap_an = compute_MOs_laplacian(mos_data=mos_data, r_carts=r_carts)

    mo_lap_auto = _compute_MOs_laplacian_autodiff(mos_data=mos_data, r_carts=r_carts)

    assert not np.any(np.isnan(np.asarray(mo_lap_an))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mo_lap_auto))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(mo_lap_an, mo_lap_auto, decimal=decimal_auto_vs_analytic_deriv)

    jax.clear_caches()


def test_MOs_sphe_to_cart():
    """Ensure spherical -> Cartesian conversion preserves MO values up to l=6."""

    rng = np.random.default_rng(0)

    nucleus_index = []
    orbital_indices = []
    exponents = []
    coefficients = []
    angular_momentums = []
    magnetic_quantum_numbers = []

    ao_idx = 0
    for l in range(7):  # l = 0..6
        for m in range(-l, l + 1):
            nucleus_index.append(0)
            orbital_indices.append(ao_idx)
            exponents.append(float(l + 1))
            coefficients.append(1.0)
            angular_momentums.append(l)
            magnetic_quantum_numbers.append(m)
            ao_idx += 1

    num_ao = ao_idx
    num_ao_prim = len(orbital_indices)
    num_mo = 6
    num_el = 12

    r_carts = rng.standard_normal((num_el, 3))
    R_carts = np.array([[0.1, -0.2, 0.3]])

    mo_coefficients = rng.standard_normal((num_mo, num_ao))

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=(1,),
        element_symbols=("X",),
        atomic_labels=("X",),
    )

    aos_data = AOs_sphe_data(
        structure_data=structure_data,
        nucleus_index=tuple(nucleus_index),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=tuple(orbital_indices),
        exponents=tuple(exponents),
        coefficients=tuple(coefficients),
        angular_momentums=tuple(angular_momentums),
        magnetic_quantum_numbers=tuple(magnetic_quantum_numbers),
    )

    mos_sphe = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
    mos_sphe.sanity_check()

    mos_cart = mos_sphe.to_cartesian()

    assert isinstance(mos_cart.aos_data, AOs_cart_data)

    mo_sphe = compute_MOs(mos_data=mos_sphe, r_carts=r_carts)
    mo_cart = compute_MOs(mos_data=mos_cart, r_carts=r_carts)

    assert not np.any(np.isnan(np.asarray(mo_cart))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mo_sphe))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(mo_cart, mo_sphe, decimal=decimal_debug_vs_production)

    grad_sphe = compute_MOs_grad(mos_data=mos_sphe, r_carts=r_carts)
    grad_cart = compute_MOs_grad(mos_data=mos_cart, r_carts=r_carts)

    for g_cart, g_sphe in zip(grad_cart, grad_sphe, strict=True):
        assert not np.any(np.isnan(np.asarray(g_cart))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(g_sphe))), "NaN detected in second argument"
        np.testing.assert_array_almost_equal(g_cart, g_sphe, decimal=decimal_debug_vs_production)

    lap_sphe = compute_MOs_laplacian(mos_data=mos_sphe, r_carts=r_carts)
    lap_cart = compute_MOs_laplacian(mos_data=mos_cart, r_carts=r_carts)

    assert not np.any(np.isnan(np.asarray(lap_cart))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(lap_sphe))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(lap_cart, lap_sphe, decimal=decimal_debug_vs_production)

    jax.clear_caches()


def test_MOs_cart_to_sphe():
    """Ensure Cartesian -> spherical conversion preserves MO values up to l=6."""

    rng = np.random.default_rng(1)

    nucleus_index = []
    orbital_indices = []
    exponents = []
    coefficients = []
    angular_momentums = []
    polynominal_order_x = []
    polynominal_order_y = []
    polynominal_order_z = []

    shell_cart_indices: list[list[int]] = []
    ao_idx = 0
    for l in range(7):  # l = 0..6
        shell_indices: list[int] = []
        for nx, ny, nz in [(nx, ny, l - nx - ny) for nx in range(l, -1, -1) for ny in range(l - nx, -1, -1)]:
            nucleus_index.append(0)
            orbital_indices.append(ao_idx)
            exponents.append(float(l + 1))
            coefficients.append(1.0)
            angular_momentums.append(l)
            polynominal_order_x.append(nx)
            polynominal_order_y.append(ny)
            polynominal_order_z.append(nz)
            shell_indices.append(ao_idx)
            ao_idx += 1
        shell_cart_indices.append(shell_indices)

    num_ao = ao_idx
    num_ao_prim = len(orbital_indices)
    num_mo = 6
    num_el = 12

    r_carts = rng.standard_normal((num_el, 3))
    R_carts = np.array([[0.1, -0.2, 0.3]])

    # Seed Cartesian coefficients from random spherical blocks to ensure pure-l content
    mo_coefficients = np.zeros((num_mo, num_ao))
    for l, cart_indices in enumerate(shell_cart_indices):
        transform = _cart_to_spherical_matrix(l)
        sph_block = rng.standard_normal((num_mo, 2 * l + 1))
        cart_block = sph_block @ transform.T
        mo_coefficients[:, cart_indices] = cart_block

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=(1,),
        element_symbols=("X",),
        atomic_labels=("X",),
    )

    aos_data = AOs_cart_data(
        structure_data=structure_data,
        nucleus_index=tuple(nucleus_index),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=tuple(orbital_indices),
        exponents=tuple(exponents),
        coefficients=tuple(coefficients),
        angular_momentums=tuple(angular_momentums),
        polynominal_order_x=tuple(polynominal_order_x),
        polynominal_order_y=tuple(polynominal_order_y),
        polynominal_order_z=tuple(polynominal_order_z),
    )

    mos_cart = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
    mos_cart.sanity_check()

    mos_sphe = mos_cart.to_spherical()

    assert isinstance(mos_sphe.aos_data, AOs_sphe_data)

    mo_cart = compute_MOs(mos_data=mos_cart, r_carts=r_carts)
    mo_sphe = compute_MOs(mos_data=mos_sphe, r_carts=r_carts)

    assert not np.any(np.isnan(np.asarray(mo_sphe))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mo_cart))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(mo_sphe, mo_cart, decimal=decimal_debug_vs_production)

    grad_cart = compute_MOs_grad(mos_data=mos_cart, r_carts=r_carts)
    grad_sphe = compute_MOs_grad(mos_data=mos_sphe, r_carts=r_carts)

    for g_cart, g_sphe in zip(grad_cart, grad_sphe, strict=True):
        assert not np.any(np.isnan(np.asarray(g_sphe))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(g_cart))), "NaN detected in second argument"
        np.testing.assert_array_almost_equal(g_sphe, g_cart, decimal=decimal_debug_vs_production)

    lap_cart = compute_MOs_laplacian(mos_data=mos_cart, r_carts=r_carts)
    lap_sphe = compute_MOs_laplacian(mos_data=mos_sphe, r_carts=r_carts)

    assert not np.any(np.isnan(np.asarray(lap_sphe))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(lap_cart))), "NaN detected in second argument"
    np.testing.assert_array_almost_equal(lap_sphe, lap_cart, decimal=decimal_debug_vs_production)

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
