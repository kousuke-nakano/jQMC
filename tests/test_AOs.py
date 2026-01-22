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

import itertools
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from numpy import linalg as LA
from numpy.testing import assert_almost_equal

# Add the project root directory to sys.path to allow executing this script directly
# This is necessary because relative imports (e.g. 'from ..jqmc') are not allowed
# when running a script directly (as __main__).
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.atomic_orbital import (  # noqa: E402
    AOs_cart_data,
    AOs_sphe_data,
    _compute_AOs_cart,
    _compute_AOs_cart_debug,
    _compute_AOs_grad_autodiff,
    _compute_AOs_grad_debug,
    _compute_AOs_laplacian_autodiff,
    _compute_AOs_laplacian_debug,
    _compute_AOs_sphe,
    _compute_AOs_sphe_debug,
    _compute_S_l_m,
    _compute_S_l_m_debug,
    # compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
)
from jqmc.setting import (  # noqa: E402
    decimal_auto_vs_analytic_deriv,
    decimal_auto_vs_numerical_deriv,
    decimal_debug_vs_production,
    decimal_numerical_vs_analytic_deriv,
)
from jqmc.structure import Structure_data  # noqa: E402

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


@pytest.mark.parametrize(
    ["l", "m"],
    list(itertools.chain.from_iterable([[pytest.param(l, m, id=f"l={l}, m={m}") for m in range(-l, l + 1)] for l in range(7)])),
)
def test_spherical_harmonics_debug_vs_production(l, m):
    """Test the spherical harmonics."""

    def Y_l_m_ref(l=0, m=0, r_cart_rel=None):
        if r_cart_rel is None:
            r_cart_rel = [0.0, 0.0, 0.0]
        """See https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics"""
        x, y, z = r_cart_rel[..., 0], r_cart_rel[..., 1], r_cart_rel[..., 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        # s orbital
        if (l, m) == (0, 0):
            return 1.0 / 2.0 * np.sqrt(1.0 / np.pi) * r**0.0
        # p orbitals
        elif (l, m) == (1, -1):
            return np.sqrt(3.0 / (4 * np.pi)) * y / r
        elif (l, m) == (1, 0):
            return np.sqrt(3.0 / (4 * np.pi)) * z / r
        elif (l, m) == (1, 1):
            return np.sqrt(3.0 / (4 * np.pi)) * x / r
        # d orbitals
        elif (l, m) == (2, -2):
            return 1.0 / 2.0 * np.sqrt(15.0 / (np.pi)) * x * y / r**2
        elif (l, m) == (2, -1):
            return 1.0 / 2.0 * np.sqrt(15.0 / (np.pi)) * y * z / r**2
        elif (l, m) == (2, 0):
            return 1.0 / 4.0 * np.sqrt(5.0 / (np.pi)) * (3 * z**2 - r**2) / r**2
        elif (l, m) == (2, 1):
            return 1.0 / 2.0 * np.sqrt(15.0 / (np.pi)) * x * z / r**2
        elif (l, m) == (2, 2):
            return 1.0 / 4.0 * np.sqrt(15.0 / (np.pi)) * (x**2 - y**2) / r**2
        # f orbitals
        elif (l, m) == (3, -3):
            return 1.0 / 4.0 * np.sqrt(35.0 / (2 * np.pi)) * y * (3 * x**2 - y**2) / r**3
        elif (l, m) == (3, -2):
            return 1.0 / 2.0 * np.sqrt(105.0 / (np.pi)) * x * y * z / r**3
        elif (l, m) == (3, -1):
            return 1.0 / 4.0 * np.sqrt(21.0 / (2 * np.pi)) * y * (5 * z**2 - r**2) / r**3
        elif (l, m) == (3, 0):
            return 1.0 / 4.0 * np.sqrt(7.0 / (np.pi)) * (5 * z**3 - 3 * z * r**2) / r**3
        elif (l, m) == (3, 1):
            return 1.0 / 4.0 * np.sqrt(21.0 / (2 * np.pi)) * x * (5 * z**2 - r**2) / r**3
        elif (l, m) == (3, 2):
            return 1.0 / 4.0 * np.sqrt(105.0 / (np.pi)) * (x**2 - y**2) * z / r**3
        elif (l, m) == (3, 3):
            return 1.0 / 4.0 * np.sqrt(35.0 / (2 * np.pi)) * x * (x**2 - 3 * y**2) / r**3
        # g orbitals
        elif (l, m) == (4, -4):
            return 3.0 / 4.0 * np.sqrt(35.0 / (np.pi)) * x * y * (x**2 - y**2) / r**4
        elif (l, m) == (4, -3):
            return 3.0 / 4.0 * np.sqrt(35.0 / (2 * np.pi)) * y * z * (3 * x**2 - y**2) / r**4
        elif (l, m) == (4, -2):
            return 3.0 / 4.0 * np.sqrt(5.0 / (np.pi)) * x * y * (7 * z**2 - r**2) / r**4
        elif (l, m) == (4, -1):
            return 3.0 / 4.0 * np.sqrt(5.0 / (2 * np.pi)) * y * (7 * z**3 - 3 * z * r**2) / r**4
        elif (l, m) == (4, 0):
            return 3.0 / 16.0 * np.sqrt(1.0 / (np.pi)) * (35 * z**4 - 30 * z**2 * r**2 + 3 * r**4) / r**4
        elif (l, m) == (4, 1):
            return 3.0 / 4.0 * np.sqrt(5.0 / (2 * np.pi)) * x * (7 * z**3 - 3 * z * r**2) / r**4
        elif (l, m) == (4, 2):
            return 3.0 / 8.0 * np.sqrt(5.0 / (np.pi)) * (x**2 - y**2) * (7 * z**2 - r**2) / r**4
        elif (l, m) == (4, 3):
            return 3.0 / 4.0 * np.sqrt(35.0 / (2 * np.pi)) * x * z * (x**2 - 3 * y**2) / r**4
        elif (l, m) == (4, 4):
            return 3.0 / 16.0 * np.sqrt(35.0 / (np.pi)) * (x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2)) / r**4
        elif (l, m) == (5, -5):
            return 3.0 / 16.0 * np.sqrt(77.0 / (2 * np.pi)) * (5 * x**4 * y - 10 * x**2 * y**3 + y**5) / r**5
        elif (l, m) == (5, -4):
            return 3.0 / 16.0 * np.sqrt(385.0 / np.pi) * 4 * x * y * z * (x**2 - y**2) / r**5
        elif (l, m) == (5, -3):
            return 1.0 / 16.0 * np.sqrt(385.0 / (2 * np.pi)) * -1 * (y**3 - 3 * x**2 * y) * (9 * z**2 - r**2) / r**5
        elif (l, m) == (5, -2):
            return 1.0 / 8.0 * np.sqrt(1155 / np.pi) * 2 * x * y * (3 * z**3 - z * r**2) / r**5
        elif (l, m) == (5, -1):
            return 1.0 / 16.0 * np.sqrt(165 / np.pi) * y * (21 * z**4 - 14 * z**2 * r**2 + r**4) / r**5
        elif (l, m) == (5, 0):
            return 1.0 / 16.0 * np.sqrt(11 / np.pi) * (63 * z**5 - 70 * z**3 * r**2 + 15 * z * r**4) / r**5
        elif (l, m) == (5, 1):
            return 1.0 / 16.0 * np.sqrt(165 / np.pi) * x * (21 * z**4 - 14 * z**2 * r**2 + r**4) / r**5
        elif (l, m) == (5, 2):
            return 1.0 / 8.0 * np.sqrt(1155 / np.pi) * (x**2 - y**2) * (3 * z**3 - z * r**2) / r**5
        elif (l, m) == (5, 3):
            return 1.0 / 16.0 * np.sqrt(385.0 / (2 * np.pi)) * (x**3 - 3 * x * y**2) * (9 * z**2 - r**2) / r**5
        elif (l, m) == (5, 4):
            return 3.0 / 16.0 * np.sqrt(385.0 / np.pi) * (x**2 * z * (x**2 - 3 * y**2) - y**2 * z * (3 * x**2 - y**2)) / r**5
        elif (l, m) == (5, 5):
            return 3.0 / 16.0 * np.sqrt(77.0 / (2 * np.pi)) * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4) / r**5
        elif (l, m) == (6, -6):
            return 1.0 / 64.0 * np.sqrt(6006.0 / np.pi) * (6 * x**5 * y - 20 * x**3 * y**3 + 6 * x * y**5) / r**6
        elif (l, m) == (6, -5):
            return 3.0 / 32.0 * np.sqrt(2002.0 / np.pi) * z * (5 * x**4 * y - 10 * x**2 * y**3 + y**5) / r**6
        elif (l, m) == (6, -4):
            return 3.0 / 32.0 * np.sqrt(91.0 / np.pi) * 4 * x * y * (11 * z**2 - r**2) * (x**2 - y**2) / r**6
        elif (l, m) == (6, -3):
            return 1.0 / 32.0 * np.sqrt(2730.0 / np.pi) * -1 * (11 * z**3 - 3 * z * r**2) * (y**3 - 3 * x**2 * y) / r**6
        elif (l, m) == (6, -2):
            return 1.0 / 64.0 * np.sqrt(2730.0 / np.pi) * 2 * x * y * (33 * z**4 - 18 * z**2 * r**2 + r**4) / r**6
        elif (l, m) == (6, -1):
            return 1.0 / 16.0 * np.sqrt(273.0 / np.pi) * y * (33 * z**5 - 30 * z**3 * r**2 + 5 * z * r**4) / r**6
        elif (l, m) == (6, 0):
            return 1.0 / 32.0 * np.sqrt(13.0 / np.pi) * (231 * z**6 - 315 * z**4 * r**2 + 105 * z**2 * r**4 - 5 * r**6) / r**6
        elif (l, m) == (6, 1):
            return 1.0 / 16.0 * np.sqrt(273.0 / np.pi) * x * (33 * z**5 - 30 * z**3 * r**2 + 5 * z * r**4) / r**6
        elif (l, m) == (6, 2):
            return 1.0 / 64.0 * np.sqrt(2730.0 / np.pi) * (x**2 - y**2) * (33 * z**4 - 18 * z**2 * r**2 + r**4) / r**6
        elif (l, m) == (6, 3):
            return 1.0 / 32.0 * np.sqrt(2730.0 / np.pi) * (11 * z**3 - 3 * z * r**2) * (x**3 - 3 * x * y**2) / r**6
        elif (l, m) == (6, 4):
            return (
                3.0
                / 32.0
                * np.sqrt(91.0 / np.pi)
                * (11 * z**2 - r**2)
                * (x**2 * (x**2 - 3 * y**2) + y**2 * (y**2 - 3 * x**2))
                / r**6
            )
        elif (l, m) == (6, 5):
            return 3.0 / 32.0 * np.sqrt(2002.0 / np.pi) * z * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4) / r**6
        elif (l, m) == (6, 6):
            return 1.0 / 64.0 * np.sqrt(6006.0 / np.pi) * (x**6 - 15 * x**4 * y**2 + 15 * x**2 * y**4 - y**6) / r**6
        else:
            raise NotImplementedError

    num_samples = 1
    R_cart = [0.0, 0.0, 1.0]
    r_cart_min, r_cart_max = -10.0, 10.0
    r_x_rand = (r_cart_max - r_cart_min) * np.random.rand(num_samples) + r_cart_min
    r_y_rand = (r_cart_max - r_cart_min) * np.random.rand(num_samples) + r_cart_min
    r_z_rand = (r_cart_max - r_cart_min) * np.random.rand(num_samples) + r_cart_min

    for r_cart in zip(r_x_rand, r_y_rand, r_z_rand, strict=True):
        r_norm = LA.norm(np.array(R_cart) - np.array(r_cart))
        r_cart_rel = np.array(r_cart) - np.array(R_cart)
        test_S_lm = _compute_S_l_m_debug(
            atomic_center_cart=R_cart,
            angular_momentum=l,
            magnetic_quantum_number=m,
            r_cart=r_cart,
        )
        ref_S_lm = np.sqrt((4 * np.pi) / (2 * l + 1)) * r_norm**l * Y_l_m_ref(l=l, m=m, r_cart_rel=r_cart_rel)
        assert_almost_equal(test_S_lm, ref_S_lm, decimal=decimal_debug_vs_production)

    jax.clear_caches()


def test_solid_harmonics_debug_vs_production():
    """Test the solid harmonics with a batch."""
    seed = 34487
    np.random.seed(seed)

    num_R_cart_samples = 49  # fixed
    num_r_cart_samples = 10
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min
    r_R_diffs_uq = r_carts[None, :, :] - R_carts[:, None, :]

    ml_list = list(itertools.chain.from_iterable([[(l, m) for m in range(-l, l + 1)] for l in range(7)]))

    # S_l_m debug
    S_l_m_debug = np.array(
        [
            [
                [
                    _compute_S_l_m_debug(
                        angular_momentum=l, magnetic_quantum_number=m, atomic_center_cart=R_cart, r_cart=r_cart
                    )
                    for r_cart in r_carts
                ]
                for R_cart in R_carts
            ]
            for l, m in ml_list
        ]
    )

    # S_l_m jax
    _, S_l_m_jax = _compute_S_l_m(r_R_diffs_uq)

    # print(f"batch_S_l_m.shape = {batch_S_l_m.shape}.")

    np.testing.assert_array_almost_equal(S_l_m_debug, S_l_m_jax, decimal=decimal_debug_vs_production)
    jax.clear_caches()


def test_AOs_sphe_debug_vs_production():
    """Test the AOs computation, comparing the JAX and debug implementations."""
    ml_list = list(itertools.chain.from_iterable([[(l, m) for m in range(-l, l + 1)] for l in range(7)]))
    num_el = 100
    num_ao = len(ml_list)
    num_ao_prim = len(ml_list)
    orbital_indices = list(range(len(ml_list)))
    exponents = [5.0] * len(ml_list)
    coefficients = [1.0] * len(ml_list)
    angular_momentums = [l for l, _ in ml_list]
    magnetic_quantum_numbers = [m for _, m in ml_list]

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

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
    aos_data.sanity_check()

    aos_jax = _compute_AOs_sphe(aos_data=aos_data, r_carts=r_carts)
    aos_debug = _compute_AOs_sphe_debug(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(aos_jax, aos_debug, decimal=decimal_debug_vs_production)

    num_el = 150
    num_ao = len(ml_list)
    num_ao_prim = len(ml_list)
    orbital_indices = list(range(len(ml_list)))
    exponents = [3.4] * len(ml_list)
    coefficients = [1.0] * len(ml_list)
    angular_momentums = angular_momentums
    magnetic_quantum_numbers = magnetic_quantum_numbers

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = -1.0, 1.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

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
    aos_data.sanity_check()

    aos_jax = _compute_AOs_sphe(aos_data=aos_data, r_carts=r_carts)
    aos_debug = _compute_AOs_sphe_debug(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(aos_jax, aos_debug, decimal=decimal_debug_vs_production)

    jax.clear_caches()


def test_AOs_cart_debug_vs_production():
    """Test the AOs computation, comparing the JAX and debug implementations."""
    l_max = 7
    angular_momentums = []
    polynominal_order_x = []
    polynominal_order_y = []
    polynominal_order_z = []
    for l in range(l_max):
        poly_orders = ["".join(p) for p in itertools.combinations_with_replacement("xyz", l)]
        poly_x = [poly_order.count("x") for poly_order in poly_orders]
        poly_y = [poly_order.count("y") for poly_order in poly_orders]
        poly_z = [poly_order.count("z") for poly_order in poly_orders]
        num_ao_mag_moms = len(poly_orders)
        angular_momentums += [l] * num_ao_mag_moms
        polynominal_order_x += poly_x
        polynominal_order_y += poly_y
        polynominal_order_z += poly_z

    num_el = 100
    num_ao = len(angular_momentums)
    num_ao_prim = num_ao
    orbital_indices = list(range(num_ao))
    exponents = [5.0] * num_ao
    coefficients = [1.0] * num_ao

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    polynominal_order_x = tuple(polynominal_order_x)
    polynominal_order_y = tuple(polynominal_order_y)
    polynominal_order_z = tuple(polynominal_order_z)

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

    aos_data = AOs_cart_data(
        structure_data=structure_data,
        nucleus_index=tuple(list(range(num_R_cart_samples))),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        polynominal_order_x=polynominal_order_x,
        polynominal_order_y=polynominal_order_y,
        polynominal_order_z=polynominal_order_z,
    )
    aos_data.sanity_check()

    aos_jax = _compute_AOs_cart(aos_data=aos_data, r_carts=r_carts)
    aos_debug = _compute_AOs_cart_debug(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(aos_jax, aos_debug, decimal=decimal_debug_vs_production)

    jax.clear_caches()


def test_AOs_sphe_and_cart_grads_analytic_vs_auto():
    """Analytic AOs gradients match JAX autodiff implementation."""
    seed = 2025
    np.random.seed(seed)

    num_r_cart_samples = 6
    num_R_cart_samples = 3
    r_carts = np.random.uniform(-1.5, 1.5, size=(num_r_cart_samples, 3))
    R_carts = np.random.uniform(-0.5, 0.5, size=(num_R_cart_samples, 3))

    num_ao = 3
    num_ao_prim = 3
    orbital_indices = tuple(range(num_ao))
    exponents = tuple([0.8, 1.1, 0.6])
    coefficients = tuple([1.0, 0.7, 1.3])
    angular_momentums = tuple([0, 1, 2])
    polynominal_order_x = tuple([0, 1, 2])
    polynominal_order_y = tuple([0, 0, 0])
    polynominal_order_z = tuple([0, 0, 0])

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

    aos_data = AOs_cart_data(
        structure_data=structure_data,
        nucleus_index=tuple(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        polynominal_order_x=polynominal_order_x,
        polynominal_order_y=polynominal_order_y,
        polynominal_order_z=polynominal_order_z,
    )
    aos_data.sanity_check()

    gx_auto, gy_auto, gz_auto = _compute_AOs_grad_autodiff(aos_data=aos_data, r_carts=r_carts)
    gx_an, gy_an, gz_an = compute_AOs_grad(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(gx_an, gx_auto, decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_array_almost_equal(gy_an, gy_auto, decimal=decimal_auto_vs_analytic_deriv)
    np.testing.assert_array_almost_equal(gz_an, gz_auto, decimal=decimal_auto_vs_analytic_deriv)

    jax.clear_caches()


def test_AOs_sphe_and_cart_grads_auto_vs_numerical():
    """Test the grad AOs computation, comparing the JAX and debug implementations."""
    # Cartesian case
    num_r_cart_samples = 8
    num_R_cart_samples = 3
    r_cart_min, r_cart_max = -2.0, +2.0
    R_cart_min, R_cart_max = -1.0, +1.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    num_ao = 3
    num_ao_prim = 3
    orbital_indices = tuple(range(num_ao))
    exponents = tuple([1.2, 0.9, 0.7])
    coefficients = tuple([1.0, 0.8, 0.6])
    angular_momentums = tuple([0, 1, 2])
    polynominal_order_x = tuple([0, 1, 2])
    polynominal_order_y = tuple([0, 0, 0])
    polynominal_order_z = tuple([0, 0, 0])

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

    aos_data_cart = AOs_cart_data(
        structure_data=structure_data,
        nucleus_index=tuple(list(range(num_R_cart_samples))),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        polynominal_order_x=polynominal_order_x,
        polynominal_order_y=polynominal_order_y,
        polynominal_order_z=polynominal_order_z,
    )
    aos_data_cart.sanity_check()

    gx_auto_cart, gy_auto_cart, gz_auto_cart = _compute_AOs_grad_autodiff(aos_data=aos_data_cart, r_carts=r_carts)
    gx_num_cart, gy_num_cart, gz_num_cart = _compute_AOs_grad_debug(aos_data=aos_data_cart, r_carts=r_carts)

    np.testing.assert_array_almost_equal(gx_auto_cart, gx_num_cart, decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_array_almost_equal(gy_auto_cart, gy_num_cart, decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_array_almost_equal(gz_auto_cart, gz_num_cart, decimal=decimal_auto_vs_numerical_deriv)

    # Spherical case
    num_r_cart_samples = 10
    num_R_cart_samples = 4
    r_cart_min, r_cart_max = -5.0, +5.0
    R_cart_min, R_cart_max = -3.0, +3.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    num_ao = 4
    num_ao_prim = 5
    orbital_indices = [0, 1, 2, 2, 3]
    exponents = [3.0, 1.0, 0.5, 0.5, 0.5]
    coefficients = [1.0, 1.0, 0.5, 0.5, 0.5]
    angular_momentums = [0, 0, 0, 0]
    magnetic_quantum_numbers = [0, 0, 0, 0]

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

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
    aos_data.sanity_check()

    gx_auto_sphe, gy_auto_sphe, gz_auto_sphe = _compute_AOs_grad_autodiff(aos_data=aos_data, r_carts=r_carts)

    (
        gx_num_sphe,
        gy_num_sphe,
        gz_num_sphe,
    ) = _compute_AOs_grad_debug(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(gx_auto_sphe, gx_num_sphe, decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_array_almost_equal(gy_auto_sphe, gy_num_sphe, decimal=decimal_auto_vs_numerical_deriv)

    np.testing.assert_array_almost_equal(gz_auto_sphe, gz_num_sphe, decimal=decimal_auto_vs_numerical_deriv)

    # Spherical case (additional coverage)
    num_r_cart_samples = 2
    num_R_cart_samples = 4
    r_cart_min, r_cart_max = -3.0, +3.0
    R_cart_min, R_cart_max = -3.0, +3.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    num_ao = 4
    num_ao_prim = 5
    orbital_indices = [0, 1, 2, 2, 3]
    exponents = [3.0, 1.0, 0.5, 0.5, 0.5]
    coefficients = [1.0, 1.0, 0.5, 0.5, 0.5]
    angular_momentums = [0, 0, 0, 0]
    magnetic_quantum_numbers = [0, 0, 0, 0]

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

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
    aos_data.sanity_check()

    gx_auto_sphe, gy_auto_sphe, gz_auto_sphe = _compute_AOs_grad_autodiff(aos_data=aos_data, r_carts=r_carts)

    (
        gx_num_sphe,
        gy_num_sphe,
        gz_num_sphe,
    ) = _compute_AOs_grad_debug(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(gx_auto_sphe, gx_num_sphe, decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_array_almost_equal(gy_auto_sphe, gy_num_sphe, decimal=decimal_auto_vs_numerical_deriv)
    np.testing.assert_array_almost_equal(gz_auto_sphe, gz_num_sphe, decimal=decimal_auto_vs_numerical_deriv)

    jax.clear_caches()


def test_AOs_sphe_and_cart_grads_analytic_vs_numerical():
    """Analytic AO gradients match numerical finite-difference implementation."""
    seed = 2028
    np.random.seed(seed)

    # Cartesian case
    num_r_cart_samples = 5
    num_R_cart_samples = 3
    r_carts = np.random.uniform(-1.2, 1.2, size=(num_r_cart_samples, 3))
    R_carts = np.random.uniform(-0.6, 0.6, size=(num_R_cart_samples, 3))

    num_ao = 3
    num_ao_prim = 3
    orbital_indices = tuple(range(num_ao))
    exponents = tuple([0.9, 1.3, 0.7])
    coefficients = tuple([1.0, 0.8, 1.2])
    angular_momentums = tuple([0, 1, 2])
    polynominal_order_x = tuple([0, 1, 2])
    polynominal_order_y = tuple([0, 0, 0])
    polynominal_order_z = tuple([0, 0, 0])

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

    aos_data = AOs_cart_data(
        structure_data=structure_data,
        nucleus_index=tuple(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        polynominal_order_x=polynominal_order_x,
        polynominal_order_y=polynominal_order_y,
        polynominal_order_z=polynominal_order_z,
    )
    aos_data.sanity_check()

    gx_num_cart, gy_num_cart, gz_num_cart = _compute_AOs_grad_debug(aos_data=aos_data, r_carts=r_carts)
    gx_an_cart, gy_an_cart, gz_an_cart = compute_AOs_grad(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(gx_an_cart, gx_num_cart, decimal=decimal_numerical_vs_analytic_deriv)
    np.testing.assert_array_almost_equal(gy_an_cart, gy_num_cart, decimal=decimal_numerical_vs_analytic_deriv)
    np.testing.assert_array_almost_equal(gz_an_cart, gz_num_cart, decimal=decimal_numerical_vs_analytic_deriv)

    # Spherical case
    num_r_cart_samples = 3
    num_R_cart_samples = 4
    r_carts = np.random.uniform(-2.5, 2.5, size=(num_r_cart_samples, 3))
    R_carts = np.random.uniform(-1.0, 1.0, size=(num_R_cart_samples, 3))

    num_ao = 4
    num_ao_prim = 5
    orbital_indices = tuple([0, 1, 2, 2, 3])
    exponents = tuple([3.0, 1.6, 0.9, 0.9, 2.2])
    coefficients = tuple([1.0, 0.9, 1.1, 0.7, 1.0])
    angular_momentums = tuple([0, 1, 1, 2])
    magnetic_quantum_numbers = tuple([0, -1, 1, 0])

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

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
    aos_data.sanity_check()

    gx_num_sphe, gy_num_sphe, gz_num_sphe = _compute_AOs_grad_debug(aos_data=aos_data, r_carts=r_carts)
    gx_an_sphe, gy_an_sphe, gz_an_sphe = compute_AOs_grad(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(gx_an_sphe, gx_num_sphe, decimal=decimal_numerical_vs_analytic_deriv)
    np.testing.assert_array_almost_equal(gy_an_sphe, gy_num_sphe, decimal=decimal_numerical_vs_analytic_deriv)
    np.testing.assert_array_almost_equal(gz_an_sphe, gz_num_sphe, decimal=decimal_numerical_vs_analytic_deriv)

    jax.clear_caches()


def test_AOs_shpe_and_cart_laplacians_analytic_vs_auto():
    """Analytic AO Laplacians match JAX autodiff implementation."""
    seed = 2026
    np.random.seed(seed)

    # Cartesian case
    num_r_cart_samples = 5
    num_R_cart_samples = 3
    r_carts = np.random.uniform(-1.2, 1.2, size=(num_r_cart_samples, 3))
    R_carts = np.random.uniform(-0.4, 0.4, size=(num_R_cart_samples, 3))

    num_ao = 3
    num_ao_prim = 3
    orbital_indices = tuple(range(num_ao))
    exponents = tuple([0.9, 1.2, 0.7])
    coefficients = tuple([1.0, 0.8, 1.1])
    angular_momentums = tuple([0, 1, 2])
    polynominal_order_x = tuple([0, 1, 2])
    polynominal_order_y = tuple([0, 0, 0])
    polynominal_order_z = tuple([0, 0, 0])

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

    aos_data = AOs_cart_data(
        structure_data=structure_data,
        nucleus_index=tuple(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        polynominal_order_x=polynominal_order_x,
        polynominal_order_y=polynominal_order_y,
        polynominal_order_z=polynominal_order_z,
    )
    aos_data.sanity_check()

    lap_auto_cart = _compute_AOs_laplacian_autodiff(aos_data=aos_data, r_carts=r_carts)
    lap_an_cart = compute_AOs_laplacian(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(lap_an_cart, lap_auto_cart, decimal=decimal_auto_vs_analytic_deriv)

    # Spherical case
    num_r_cart_samples = 3
    num_R_cart_samples = 4
    r_carts = np.random.uniform(-2.0, 2.0, size=(num_r_cart_samples, 3))
    R_carts = np.random.uniform(-0.7, 0.7, size=(num_R_cart_samples, 3))

    num_ao = 4
    num_ao_prim = 5
    orbital_indices = tuple([0, 1, 2, 2, 3])
    exponents = tuple([3.0, 1.5, 0.8, 0.8, 2.2])
    coefficients = tuple([1.0, 0.9, 1.1, 0.7, 1.0])
    angular_momentums = tuple([0, 1, 1, 2])
    magnetic_quantum_numbers = tuple([0, -1, 1, 0])

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

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
    aos_data.sanity_check()

    lap_auto_sphe = _compute_AOs_laplacian_autodiff(aos_data=aos_data, r_carts=r_carts)
    lap_an_sphe = compute_AOs_laplacian(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(lap_an_sphe, lap_auto_sphe, decimal=decimal_auto_vs_analytic_deriv)


def test_AOs_shpe_and_cart_laplacians_analytic_vs_numerical():
    """Analytic Laplacians match numerical finite-difference implementation."""
    seed = 2027
    np.random.seed(seed)

    # Cartesian case
    num_r_cart_samples = 4
    num_R_cart_samples = 2
    r_carts = np.random.uniform(-1.0, 1.0, size=(num_r_cart_samples, 3))
    R_carts = np.random.uniform(-0.6, 0.6, size=(num_R_cart_samples, 3))

    num_ao = 2
    num_ao_prim = 3
    orbital_indices = tuple([0, 0, 1])
    exponents = tuple([1.4, 0.9, 1.1])
    coefficients = tuple([1.0, 0.7, 0.9])
    angular_momentums = tuple([0, 1])
    polynominal_order_x = tuple([0, 1])
    polynominal_order_y = tuple([0, 0])
    polynominal_order_z = tuple([0, 0])

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

    aos_data = AOs_cart_data(
        structure_data=structure_data,
        nucleus_index=tuple(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        polynominal_order_x=polynominal_order_x,
        polynominal_order_y=polynominal_order_y,
        polynominal_order_z=polynominal_order_z,
    )
    aos_data.sanity_check()

    lap_num_cart = _compute_AOs_laplacian_debug(aos_data=aos_data, r_carts=r_carts)
    lap_an_cart = compute_AOs_laplacian(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(lap_an_cart, lap_num_cart, decimal=decimal_numerical_vs_analytic_deriv)

    # Spherical case
    num_r_cart_samples = 3
    num_R_cart_samples = 3
    r_carts = np.random.uniform(-1.5, 1.5, size=(num_r_cart_samples, 3))
    R_carts = np.random.uniform(-0.8, 0.8, size=(num_R_cart_samples, 3))

    num_ao = 3
    num_ao_prim = 4
    orbital_indices = tuple([0, 1, 1, 2])
    exponents = tuple([2.0, 1.6, 1.1, 0.9])
    coefficients = tuple([1.0, 0.8, 1.2, 0.7])
    angular_momentums = tuple([0, 1, 1])
    magnetic_quantum_numbers = tuple([0, 0, 1])

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

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
    aos_data.sanity_check()

    lap_num_sphe = _compute_AOs_laplacian_debug(aos_data=aos_data, r_carts=r_carts)
    lap_an_sphe = compute_AOs_laplacian(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(lap_an_sphe, lap_num_sphe, decimal=decimal_numerical_vs_analytic_deriv)

    jax.clear_caches()


def test_AOs_shpe_and_cart_laplacians_auto_vs_numerical():
    """Test the laplacian AOs computation, comparing the JAX and debug implementations."""
    # Cartesian case
    num_r_cart_samples = 5
    num_R_cart_samples = 3
    r_cart_min, r_cart_max = -2.0, +2.0
    R_cart_min, R_cart_max = -1.0, +1.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 1, 2, 2]
    exponents = [1.4, 0.9, 0.7, 0.7]
    coefficients = [1.0, 0.8, 0.6, 0.5]
    angular_momentums = [0, 1, 1]
    polynominal_order_x = [0, 1, 1]
    polynominal_order_y = [0, 0, 0]
    polynominal_order_z = [0, 0, 0]

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

    aos_data = AOs_cart_data(
        structure_data=structure_data,
        nucleus_index=tuple(list(range(num_R_cart_samples))),
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
    aos_data.sanity_check()

    lap_num_cart = _compute_AOs_laplacian_autodiff(aos_data=aos_data, r_carts=r_carts)
    lap_auto_cart = _compute_AOs_laplacian_debug(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(lap_auto_cart, lap_num_cart, decimal=decimal_auto_vs_numerical_deriv)

    # Spherical cases
    num_r_cart_samples = 10
    num_R_cart_samples = 3
    r_cart_min, r_cart_max = -5.0, +5.0
    R_cart_min, R_cart_max = -3.0, +3.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 1, 2, 2]
    exponents = [3.0, 1.0, 0.5, 0.5]
    coefficients = [1.0, 1.0, 0.5, 0.5]
    angular_momentums = [0, 0, 0]
    magnetic_quantum_numbers = [0, 0, 0]

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

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
    aos_data.sanity_check()

    lap_auto_sphe = _compute_AOs_laplacian_autodiff(aos_data=aos_data, r_carts=r_carts)

    lap_num_sphe = _compute_AOs_laplacian_debug(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(lap_num_sphe, lap_auto_sphe, decimal=decimal_auto_vs_numerical_deriv)

    num_r_cart_samples = 2
    num_R_cart_samples = 3
    r_cart_min, r_cart_max = -3.0, +3.0
    R_cart_min, R_cart_max = -3.0, +3.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    num_ao = 3
    num_ao_prim = 3
    orbital_indices = [0, 1, 2]
    exponents = [30.0, 10.0, 8.5]
    coefficients = [1.0, 1.0, 1.0]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 1, -1]

    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_R_cart_samples),
        element_symbols=tuple(["X"] * num_R_cart_samples),
        atomic_labels=tuple(["X"] * num_R_cart_samples),
    )
    structure_data.sanity_check()

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
    aos_data.sanity_check()

    lap_auto_sphe = _compute_AOs_laplacian_autodiff(aos_data=aos_data, r_carts=r_carts)

    lap_num_sphe = _compute_AOs_laplacian_debug(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(lap_num_sphe, lap_auto_sphe, decimal=decimal_auto_vs_numerical_deriv)

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
