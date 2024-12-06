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

import os
import pickle
from logging import Formatter, StreamHandler, getLogger

import jax
import numpy as np

from ..jqmc.atomic_orbital import (
    AOs_data,
)
from ..jqmc.jastrow_factor import (
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    compute_grads_and_laplacian_Jastrow_three_body_debug,
    compute_grads_and_laplacian_Jastrow_three_body_jax,
    compute_grads_and_laplacian_Jastrow_two_body_debug,
    compute_grads_and_laplacian_Jastrow_two_body_jax,
    compute_Jastrow_three_body_debug,
    compute_Jastrow_three_body_jax,
    compute_Jastrow_two_body_debug,
    compute_Jastrow_two_body_jax,
)
from ..jqmc.structure import Structure_data

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


def test_numerical_and_auto_grads_Jastrow_threebody_part():
    # test MOs
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

    # generate matrices for the test
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min
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

    j_matrix = np.random.rand(aos_data.num_ao, aos_data.num_ao + 1)

    jastrow_three_body_data = Jastrow_three_body_data(orb_data=aos_data, j_matrix=j_matrix)

    J3_debug = compute_Jastrow_three_body_debug(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # print(f"J3_debug = {J3_debug}")

    J3_jax = compute_Jastrow_three_body_jax(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # print(f"J3_jax = {J3_jax}")

    np.testing.assert_almost_equal(J3_debug, J3_jax, decimal=8)

    (
        grad_jastrow_J3_up_debug,
        grad_jastrow_J3_dn_debug,
        sum_laplacian_J3_debug,
    ) = compute_grads_and_laplacian_Jastrow_three_body_debug(
        jastrow_three_body_data,
        r_up_carts,
        r_dn_carts,
    )

    # print(f"grad_jastrow_J3_up_debug = {grad_jastrow_J3_up_debug}")
    # print(f"grad_jastrow_J3_dn_debug = {grad_jastrow_J3_dn_debug}")
    # print(f"sum_laplacian_J3_debug = {sum_laplacian_J3_debug}")

    grad_jastrow_J3_up_jax, grad_jastrow_J3_dn_jax, sum_laplacian_J3_jax = compute_grads_and_laplacian_Jastrow_three_body_jax(
        jastrow_three_body_data,
        r_up_carts,
        r_dn_carts,
    )

    # print(f"grad_jastrow_J3_up_jax = {grad_jastrow_J3_up_jax}")
    # print(f"grad_jastrow_J3_dn_jax = {grad_jastrow_J3_dn_jax}")
    # print(f"sum_laplacian_J3_jax = {sum_laplacian_J3_jax}")

    np.testing.assert_almost_equal(grad_jastrow_J3_up_debug, grad_jastrow_J3_up_jax, decimal=4)
    np.testing.assert_almost_equal(grad_jastrow_J3_dn_debug, grad_jastrow_J3_dn_jax, decimal=4)
    np.testing.assert_almost_equal(sum_laplacian_J3_debug, sum_laplacian_J3_jax, decimal=4)

    jax.clear_caches()


def test_numerical_and_auto_grads_Jastrow_twobody_part():
    # test MOs
    num_r_up_cart_samples = 5
    num_r_dn_cart_samples = 2

    r_cart_min, r_cart_max = -3.0, 3.0

    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min

    jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=1.0)
    J2_debug = compute_Jastrow_two_body_debug(
        jastrow_two_body_data=jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    # print(f"jastrow_two_body_debug = {jastrow_two_body_debug}")

    J2_jax = compute_Jastrow_two_body_jax(
        jastrow_two_body_data=jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    # print(f"jastrow_two_body_jax = {jastrow_two_body_jax}")

    np.testing.assert_almost_equal(J2_debug, J2_jax, decimal=10)

    (
        grad_J2_up_debug,
        grad_J2_dn_debug,
        sum_laplacian_J2_debug,
    ) = compute_grads_and_laplacian_Jastrow_two_body_debug(
        jastrow_two_body_data,
        r_up_carts,
        r_dn_carts,
    )

    # print(f"grad_J2_up_debug = {grad_J2_up_debug}")
    # print(f"grad_J2_dn_debug = {grad_J2_dn_debug}")
    # print(f"sum_laplacian_J2_debug = {sum_laplacian_J2_debug}")

    grad_J2_up_jax, grad_J2_dn_jax, sum_laplacian_J2_jax = compute_grads_and_laplacian_Jastrow_two_body_jax(
        jastrow_two_body_data,
        r_up_carts,
        r_dn_carts,
    )

    # print(f"grad_J2_up_jax = {grad_J2_up_jax}")
    # print(f"grad_J2_dn_jax = {grad_J2_dn_jax}")
    # print(f"sum_laplacian_J2_jax = {sum_laplacian_J2_jax}")

    np.testing.assert_almost_equal(grad_J2_up_debug, grad_J2_up_jax, decimal=8)
    np.testing.assert_almost_equal(grad_J2_dn_debug, grad_J2_dn_jax, decimal=8)
    np.testing.assert_almost_equal(sum_laplacian_J2_debug, sum_laplacian_J2_jax, decimal=4)

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
