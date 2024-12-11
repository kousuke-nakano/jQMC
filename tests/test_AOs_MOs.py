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

import itertools
import os
from logging import Formatter, StreamHandler, getLogger

import jax
import numpy as np
import pytest
from numpy import linalg as LA
from numpy.testing import assert_almost_equal

from ..jqmc.atomic_orbital import (
    AO_data,
    AOs_data,
    _compute_AOs_debug,
    _compute_AOs_grad_debug,
    _compute_AOs_grad_jax,
    _compute_AOs_jax,
    _compute_AOs_laplacian_debug,
    _compute_AOs_laplacian_jax,
    _compute_S_l_m_debug,
    _compute_S_l_m_jax,
)
from ..jqmc.determinant import (
    Geminal_data,
    _compute_det_geminal_all_elements_debug,
    _compute_det_geminal_all_elements_jax,
    _compute_geminal_all_elements_debug,
    _compute_geminal_all_elements_jax,
    _compute_grads_and_laplacian_ln_Det_debug,
    _compute_grads_and_laplacian_ln_Det_jax,
)
from ..jqmc.molecular_orbital import (
    MO_data,
    MOs_data,
    _compute_MOs_debug,
    _compute_MOs_grad_debug,
    _compute_MOs_grad_jax,
    _compute_MOs_jax,
    _compute_MOs_laplacian_debug,
    _compute_MOs_laplacian_jax,
    compute_MO,
)
from ..jqmc.structure import Structure_data
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
@pytest.mark.parametrize(
    ["l", "m"],
    list(itertools.chain.from_iterable([[pytest.param(l, m, id=f"l={l}, m={m}") for m in range(-l, l + 1)] for l in range(7)])),
)
def test_spherical_harmonics(l, m):
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

    for r_cart in zip(r_x_rand, r_y_rand, r_z_rand):
        r_norm = LA.norm(np.array(R_cart) - np.array(r_cart))
        r_cart_rel = np.array(r_cart) - np.array(R_cart)
        test_S_lm = _compute_S_l_m_debug(
            atomic_center_cart=R_cart,
            angular_momentum=l,
            magnetic_quantum_number=m,
            r_cart=r_cart,
        )
        ref_S_lm = np.sqrt((4 * np.pi) / (2 * l + 1)) * r_norm**l * Y_l_m_ref(l=l, m=m, r_cart_rel=r_cart_rel)
        assert_almost_equal(test_S_lm, ref_S_lm, decimal=8)

    jax.clear_caches()


@pytest.mark.parametrize(
    ["l", "m"],
    list(itertools.chain.from_iterable([[pytest.param(l, m, id=f"l={l}, m={m}") for m in range(-l, l + 1)] for l in range(7)])),
)
def test_solid_harmonics(l, m):
    num_samples = 1
    R_cart = [0.0, 0.0, 1.0]
    r_cart_min, r_cart_max = -10.0, 10.0
    r_x_rand = (r_cart_max - r_cart_min) * np.random.rand(num_samples) + r_cart_min
    r_y_rand = (r_cart_max - r_cart_min) * np.random.rand(num_samples) + r_cart_min
    r_z_rand = (r_cart_max - r_cart_min) * np.random.rand(num_samples) + r_cart_min

    for r_cart in zip(r_x_rand, r_y_rand, r_z_rand):
        test_S_lm = _compute_S_l_m_jax(
            R_cart=R_cart,
            l=l,
            m=m,
            r_cart=r_cart,
        )
        ref_S_lm = _compute_S_l_m_debug(
            atomic_center_cart=R_cart,
            angular_momentum=l,
            magnetic_quantum_number=m,
            r_cart=r_cart,
        )
        assert_almost_equal(test_S_lm, ref_S_lm, decimal=8)

    jax.clear_caches()


# @pytest.mark.skip
def test_AOs_comparing_jax_and_debug_implemenetations():
    num_el = 100
    num_ao = 25
    num_ao_prim = 25
    orbital_indices = list(range(25))
    exponents = [5.0] * 25
    coefficients = [1.0] * 25
    angular_momentums = [
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
    ]
    magnetic_quantum_numbers = [
        0,
        -1,
        0,
        1,
        -2,
        -1,
        0,
        1,
        2,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
    ]

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * num_R_cart_samples,
        element_symbols=["X"] * num_R_cart_samples,
        atomic_labels=["X"] * num_R_cart_samples,
    )

    aos_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_jax = _compute_AOs_jax(aos_data=aos_data, r_carts=r_carts)
    aos_debug = _compute_AOs_debug(aos_data=aos_data, r_carts=r_carts)

    assert np.allclose(aos_jax, aos_debug, rtol=1e-12, atol=1e-05)

    num_el = 150
    num_ao = 25
    num_ao_prim = 25
    orbital_indices = list(range(25))
    exponents = [3.4] * 25
    coefficients = [1.0] * 25
    angular_momentums = angular_momentums
    magnetic_quantum_numbers = magnetic_quantum_numbers

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = -1.0, 1.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * num_R_cart_samples,
        element_symbols=["X"] * num_R_cart_samples,
        atomic_labels=["X"] * num_R_cart_samples,
    )

    aos_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_jax = _compute_AOs_jax(aos_data=aos_data, r_carts=r_carts)
    aos_debug = _compute_AOs_debug(aos_data=aos_data, r_carts=r_carts)

    assert np.allclose(aos_jax, aos_debug, rtol=1e-12, atol=1e-05)

    jax.clear_caches()


def test_AOs_comparing_auto_and_numerical_grads():
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

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * num_R_cart_samples,
        element_symbols=["X"] * num_R_cart_samples,
        atomic_labels=["X"] * num_R_cart_samples,
    )

    aos_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    ao_matrix_grad_x_auto, ao_matrix_grad_y_auto, ao_matrix_grad_z_auto = _compute_AOs_grad_jax(
        aos_data=aos_data, r_carts=r_carts
    )

    (
        ao_matrix_grad_x_numerical,
        ao_matrix_grad_y_numerical,
        ao_matrix_grad_z_numerical,
    ) = _compute_AOs_grad_debug(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(ao_matrix_grad_x_auto, ao_matrix_grad_x_numerical, decimal=7)
    np.testing.assert_array_almost_equal(ao_matrix_grad_y_auto, ao_matrix_grad_y_numerical, decimal=7)

    np.testing.assert_array_almost_equal(ao_matrix_grad_z_auto, ao_matrix_grad_z_numerical, decimal=7)

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

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * num_R_cart_samples,
        element_symbols=["X"] * num_R_cart_samples,
        atomic_labels=["X"] * num_R_cart_samples,
    )

    aos_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    ao_matrix_grad_x_auto, ao_matrix_grad_y_auto, ao_matrix_grad_z_auto = _compute_AOs_grad_jax(
        aos_data=aos_data, r_carts=r_carts
    )

    (
        ao_matrix_grad_x_numerical,
        ao_matrix_grad_y_numerical,
        ao_matrix_grad_z_numerical,
    ) = _compute_AOs_grad_debug(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(ao_matrix_grad_x_auto, ao_matrix_grad_x_numerical, decimal=7)
    np.testing.assert_array_almost_equal(ao_matrix_grad_y_auto, ao_matrix_grad_y_numerical, decimal=7)

    np.testing.assert_array_almost_equal(ao_matrix_grad_z_auto, ao_matrix_grad_z_numerical, decimal=7)

    jax.clear_caches()


def test_AOs_comparing_auto_and_numerical_laplacians():
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

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * num_R_cart_samples,
        element_symbols=["X"] * num_R_cart_samples,
        atomic_labels=["X"] * num_R_cart_samples,
    )

    aos_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    ao_matrix_laplacian_numerical = _compute_AOs_laplacian_jax(aos_data=aos_data, r_carts=r_carts)

    ao_matrix_laplacian_auto = _compute_AOs_laplacian_debug(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(ao_matrix_laplacian_auto, ao_matrix_laplacian_numerical, decimal=5)

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

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * num_R_cart_samples,
        element_symbols=["X"] * num_R_cart_samples,
        atomic_labels=["X"] * num_R_cart_samples,
    )

    aos_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    ao_matrix_laplacian_numerical = _compute_AOs_laplacian_jax(aos_data=aos_data, r_carts=r_carts)

    ao_matrix_laplacian_auto = _compute_AOs_laplacian_debug(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(ao_matrix_laplacian_auto, ao_matrix_laplacian_numerical, decimal=5)

    jax.clear_caches()


def test_MOs_comparing_jax_and_debug_implemenetations():
    num_el = 10
    num_mo = 5
    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 0, 1, 2]
    exponents = [50.0, 20.0, 10.0, 5.0]
    coefficients = [1.0, 1.0, 1.0, 0.5]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 0, -1]

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -5.0, 5.0
    R_cart_min, R_cart_max = -6.0, 6.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    mo_coefficients = np.random.rand(num_mo, num_ao)

    # compute each MO step by step
    mo_ans_step_by_step = []

    ao_data_l = [
        AO_data(
            num_ao_prim=orbital_indices.count(i),
            atomic_center_cart=R_carts[i],
            exponents=[exponents[k] for (k, v) in enumerate(orbital_indices) if v == i],
            coefficients=[coefficients[k] for (k, v) in enumerate(orbital_indices) if v == i],
            angular_momentum=angular_momentums[i],
            magnetic_quantum_number=magnetic_quantum_numbers[i],
        )
        for i in range(num_ao)
    ]

    for mo_coeff in mo_coefficients:
        mo_data = MO_data(ao_data_l=ao_data_l, mo_coefficients=mo_coeff)
        mo_ans_step_by_step.append([compute_MO(mo_data=mo_data, r_cart=r_cart) for r_cart in r_carts])
    mo_ans_step_by_step = np.array(mo_ans_step_by_step)

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * num_R_cart_samples,
        element_symbols=["X"] * num_R_cart_samples,
        atomic_labels=["X"] * num_R_cart_samples,
    )

    aos_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    mo_ans_all_jax = _compute_MOs_jax(mos_data=mos_data, r_carts=r_carts)

    mo_ans_all_debug = _compute_MOs_debug(mos_data=mos_data, r_carts=r_carts)

    assert np.allclose(mo_ans_step_by_step, mo_ans_all_jax)
    assert np.allclose(mo_ans_step_by_step, mo_ans_all_debug)

    num_el = 10
    num_mo = 5
    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 0, 1, 2]
    exponents = [10.0, 5.0, 1.0, 1.0]
    coefficients = [1.0, 1.0, 1.0, 0.5]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 1, -1]

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -5.0, 5.0
    R_cart_min, R_cart_max = -6.0, 6.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    mo_coefficients = np.random.rand(num_mo, num_ao)

    # compute each MO step by step
    mo_ans_step_by_step = []

    ao_data_l = [
        AO_data(
            num_ao_prim=orbital_indices.count(i),
            atomic_center_cart=R_carts[i],
            exponents=[exponents[k] for (k, v) in enumerate(orbital_indices) if v == i],
            coefficients=[coefficients[k] for (k, v) in enumerate(orbital_indices) if v == i],
            angular_momentum=angular_momentums[i],
            magnetic_quantum_number=magnetic_quantum_numbers[i],
        )
        for i in range(num_ao)
    ]

    for mo_coeff in mo_coefficients:
        mo_data = MO_data(ao_data_l=ao_data_l, mo_coefficients=mo_coeff)
        mo_ans_step_by_step.append([compute_MO(mo_data=mo_data, r_cart=r_cart) for r_cart in r_carts])
    mo_ans_step_by_step = np.array(mo_ans_step_by_step)

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * num_R_cart_samples,
        element_symbols=["X"] * num_R_cart_samples,
        atomic_labels=["X"] * num_R_cart_samples,
    )

    aos_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    mo_ans_all_jax = _compute_MOs_jax(mos_data=mos_data, r_carts=r_carts)

    mo_ans_all_debug = _compute_MOs_debug(mos_data=mos_data, r_carts=r_carts)

    assert np.allclose(mo_ans_step_by_step, mo_ans_all_jax)
    assert np.allclose(mo_ans_step_by_step, mo_ans_all_debug)

    jax.clear_caches()


# @pytest.mark.skip_if_enable_jit
def test_MOs_comparing_auto_and_numerical_grads(request):
    # if request.config.getoption("--enable-jit"):
    #    pytest.skip(reason="Bug of flux.struct with @jit.")
    num_el = 10
    num_mo = 5
    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 0, 1, 2]
    exponents = [50.0, 20.0, 10.0, 5.0]
    coefficients = [1.0, 1.0, 1.0, 0.5]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 0, -1]

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -5.0, 5.0
    R_cart_min, R_cart_max = 10.0, 10.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    mo_coefficients = np.random.rand(num_mo, num_ao)

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * num_R_cart_samples,
        element_symbols=["X"] * num_R_cart_samples,
        atomic_labels=["X"] * num_R_cart_samples,
    )

    aos_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    mo_matrix_grad_x_auto, mo_matrix_grad_y_auto, mo_matrix_grad_z_auto = _compute_MOs_grad_jax(
        mos_data=mos_data, r_carts=r_carts
    )

    (
        mo_matrix_grad_x_numerical,
        mo_matrix_grad_y_numerical,
        mo_matrix_grad_z_numerical,
    ) = _compute_MOs_grad_debug(mos_data=mos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(mo_matrix_grad_x_auto, mo_matrix_grad_x_numerical, decimal=6)
    np.testing.assert_array_almost_equal(mo_matrix_grad_y_auto, mo_matrix_grad_y_numerical, decimal=6)

    np.testing.assert_array_almost_equal(mo_matrix_grad_z_auto, mo_matrix_grad_z_numerical, decimal=6)

    num_el = 10
    num_mo = 5
    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 0, 1, 2]
    exponents = [10.0, 5.0, 1.0, 1.0]
    coefficients = [1.0, 1.0, 1.0, 0.5]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 1, -1]

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 3.0, 3.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    mo_coefficients = np.random.rand(num_mo, num_ao)

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * num_R_cart_samples,
        element_symbols=["X"] * num_R_cart_samples,
        atomic_labels=["X"] * num_R_cart_samples,
    )

    aos_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    mo_matrix_grad_x_auto, mo_matrix_grad_y_auto, mo_matrix_grad_z_auto = _compute_MOs_grad_jax(
        mos_data=mos_data, r_carts=r_carts
    )

    (
        mo_matrix_grad_x_numerical,
        mo_matrix_grad_y_numerical,
        mo_matrix_grad_z_numerical,
    ) = _compute_MOs_grad_debug(mos_data=mos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(mo_matrix_grad_x_auto, mo_matrix_grad_x_numerical, decimal=6)
    np.testing.assert_array_almost_equal(mo_matrix_grad_y_auto, mo_matrix_grad_y_numerical, decimal=6)

    np.testing.assert_array_almost_equal(mo_matrix_grad_z_auto, mo_matrix_grad_z_numerical, decimal=6)

    jax.clear_caches()


def test_MOs_comparing_auto_and_numerical_laplacians():
    num_el = 10
    num_mo = 5
    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 0, 1, 2]
    exponents = [50.0, 20.0, 10.0, 5.0]
    coefficients = [1.0, 1.0, 1.0, 0.5]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 0, -1]

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -5.0, 5.0
    R_cart_min, R_cart_max = 10.0, 10.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    mo_coefficients = np.random.rand(num_mo, num_ao)

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * num_R_cart_samples,
        element_symbols=["X"] * num_R_cart_samples,
        atomic_labels=["X"] * num_R_cart_samples,
    )

    aos_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    mo_matrix_laplacian_numerical = _compute_MOs_laplacian_jax(mos_data=mos_data, r_carts=r_carts)

    mo_matrix_laplacian_auto = _compute_MOs_laplacian_debug(mos_data=mos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(mo_matrix_laplacian_auto, mo_matrix_laplacian_numerical, decimal=6)

    jax.clear_caches()


@pytest.mark.parametrize(
    "filename",
    ["water_trexio.hdf5"],
    ids=["water_trexio.hdf5"],
)
def test_read_trexio_files(filename: str):
    read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", filename))
    jax.clear_caches()


def test_comparing_AO_and_MO_geminals():
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"))
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

    coords = structure_data.positions_cart

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

    geminal_mo_jax = _compute_geminal_all_elements_jax(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    np.testing.assert_almost_equal(geminal_mo_debug, geminal_mo_jax, decimal=15)

    geminal_mo = geminal_mo_jax

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

    np.testing.assert_almost_equal(geminal_ao_debug, geminal_ao_jax, decimal=15)

    geminal_ao = geminal_ao_jax

    # check if geminals with AO and MO representations are consistent
    np.testing.assert_array_almost_equal(geminal_ao, geminal_mo, decimal=15)

    det_geminal_mo_debug = _compute_det_geminal_all_elements_debug(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    det_geminal_mo_jax = _compute_det_geminal_all_elements_jax(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    np.testing.assert_array_almost_equal(det_geminal_mo_debug, det_geminal_mo_jax, decimal=15)
    det_geminal_mo = det_geminal_mo_jax

    det_geminal_ao_debug = _compute_det_geminal_all_elements_debug(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    det_geminal_ao_jax = _compute_det_geminal_all_elements_jax(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    np.testing.assert_array_almost_equal(det_geminal_ao_debug, det_geminal_ao_jax, decimal=15)
    det_geminal_ao = det_geminal_ao_jax

    np.testing.assert_almost_equal(det_geminal_ao, det_geminal_mo, decimal=15)

    jax.clear_caches()


if __name__ == "__main__":
    logger = getLogger("myqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    np.set_printoptions(threshold=1.0e8)

    pass


def test_numerial_and_auto_grads_ln_Det():
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"))

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

    coords = structure_data.positions_cart

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

    grad_ln_D_up_numerical, grad_ln_D_dn_numerical, sum_laplacian_ln_D_numerical = _compute_grads_and_laplacian_ln_Det_debug(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    grad_ln_D_up_auto, grad_ln_D_dn_auto, sum_laplacian_ln_D_auto = _compute_grads_and_laplacian_ln_Det_jax(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    np.testing.assert_almost_equal(np.array(grad_ln_D_up_numerical), np.array(grad_ln_D_up_auto), decimal=5)
    np.testing.assert_almost_equal(np.array(grad_ln_D_dn_numerical), np.array(grad_ln_D_dn_auto), decimal=5)
    np.testing.assert_almost_equal(sum_laplacian_ln_D_numerical, sum_laplacian_ln_D_auto, decimal=1)

    jax.clear_caches()
