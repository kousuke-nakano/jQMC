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

# MPI
from mpi4py import MPI

# Add the project root directory to sys.path to allow executing this script directly
# This is necessary because relative imports (e.g. 'from ..jqmc') are not allowed
# when running a script directly (as __main__).
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.atomic_orbital import (  # noqa: E402
    AO_sphe_data,
    AOs_sphe_data,
)
from jqmc.molecular_orbital import (  # noqa: E402
    MO_data,
    MOs_data,
    compute_MO,
    compute_MOs_debug,
    compute_MOs_grad_debug,
    compute_MOs_grad_jax,
    compute_MOs_jax,
    compute_MOs_laplacian_debug,
    compute_MOs_laplacian_jax,
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

    # compute each MO step by step
    mo_ans_step_by_step = []

    ao_data_l = [
        AO_sphe_data(
            num_ao_prim=orbital_indices.count(i),
            atomic_center_cart=R_carts[i],
            exponents=tuple([exponents[k] for (k, v) in enumerate(orbital_indices) if v == i]),
            coefficients=tuple([coefficients[k] for (k, v) in enumerate(orbital_indices) if v == i]),
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

    mo_ans_all_jax = compute_MOs_jax(mos_data=mos_data, r_carts=r_carts)

    mo_ans_all_debug = compute_MOs_debug(mos_data=mos_data, r_carts=r_carts)

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

    # compute each MO step by step
    mo_ans_step_by_step = []

    ao_data_l = [
        AO_sphe_data(
            num_ao_prim=orbital_indices.count(i),
            atomic_center_cart=R_carts[i],
            exponents=tuple([exponents[k] for (k, v) in enumerate(orbital_indices) if v == i]),
            coefficients=tuple([coefficients[k] for (k, v) in enumerate(orbital_indices) if v == i]),
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

    mo_ans_all_jax = compute_MOs_jax(mos_data=mos_data, r_carts=r_carts)

    mo_ans_all_debug = compute_MOs_debug(mos_data=mos_data, r_carts=r_carts)

    assert np.allclose(mo_ans_step_by_step, mo_ans_all_jax)
    assert np.allclose(mo_ans_step_by_step, mo_ans_all_debug)

    jax.clear_caches()


@pytest.mark.obsolete(reasons="Gradients are now implemented by fully exploiting JAX modules.")
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

    mo_matrix_grad_x_auto, mo_matrix_grad_y_auto, mo_matrix_grad_z_auto = compute_MOs_grad_jax(
        mos_data=mos_data, r_carts=r_carts
    )

    (
        mo_matrix_grad_x_numerical,
        mo_matrix_grad_y_numerical,
        mo_matrix_grad_z_numerical,
    ) = compute_MOs_grad_debug(mos_data=mos_data, r_carts=r_carts)

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

    mo_matrix_grad_x_auto, mo_matrix_grad_y_auto, mo_matrix_grad_z_auto = compute_MOs_grad_jax(
        mos_data=mos_data, r_carts=r_carts
    )

    (
        mo_matrix_grad_x_numerical,
        mo_matrix_grad_y_numerical,
        mo_matrix_grad_z_numerical,
    ) = compute_MOs_grad_debug(mos_data=mos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(mo_matrix_grad_x_auto, mo_matrix_grad_x_numerical, decimal=6)
    np.testing.assert_array_almost_equal(mo_matrix_grad_y_auto, mo_matrix_grad_y_numerical, decimal=6)

    np.testing.assert_array_almost_equal(mo_matrix_grad_z_auto, mo_matrix_grad_z_numerical, decimal=6)

    jax.clear_caches()


@pytest.mark.obsolete(reasons="Laplacians are now implemented by fully exploiting JAX modules.")
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

    mo_matrix_laplacian_numerical = compute_MOs_laplacian_jax(mos_data=mos_data, r_carts=r_carts)

    mo_matrix_laplacian_auto = compute_MOs_laplacian_debug(mos_data=mos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(mo_matrix_laplacian_auto, mo_matrix_laplacian_numerical, decimal=6)

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
