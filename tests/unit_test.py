import itertools
import os
from logging import Formatter, StreamHandler, getLogger

import jax
import numpy as np
import pytest
from numpy import linalg as LA
from numpy.testing import assert_almost_equal

from ..jqmc.atomic_orbital import (
    AO_data_debug,
    AOs_data_debug,
    compute_AOs_api,
    compute_AOs_grad_api,
    compute_AOs_laplacian_api,
    compute_S_l_m_debug,
    compute_S_l_m_jax,
)
from ..jqmc.coulomb_potential import (
    compute_bare_coulomb_potential_api,
    compute_ecp_coulomb_potential_api,
)
from ..jqmc.determinant import (
    Geminal_data,
    compute_det_geminal_all_elements_api,
    compute_geminal_all_elements_api,
    compute_grads_and_laplacian_ln_Det_api,
)
from ..jqmc.hamiltonians import Hamiltonian_data
from ..jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    compute_grads_and_laplacian_Jastrow_three_body_api,
    compute_grads_and_laplacian_Jastrow_two_body_api,
    compute_Jastrow_three_body_api,
    compute_Jastrow_two_body_api,
)
from ..jqmc.molecular_orbital import (
    MO_data,
    MOs_data,
    compute_MO,
    compute_MOs_api,
    compute_MOs_grad_api,
    compute_MOs_laplacian_api,
)
from ..jqmc.swct import SWCT_data, evaluate_swct_domega_api, evaluate_swct_omega_api
from ..jqmc.trexio_wrapper import read_trexio_file
from ..jqmc.wavefunction import (
    Wavefunction_data,
    compute_kinetic_energy_api,
    evaluate_wavefunction_api,
)

# JAX float64
jax.config.update("jax_enable_x64", True)

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
    list(
        itertools.chain.from_iterable(
            [[pytest.param(l, m, id=f"l={l}, m={m}") for m in range(-l, l + 1)] for l in range(7)]
        )
    ),
)
def test_spherical_harmonics(l, m):
    def Y_l_m_ref(l=0, m=0, r_cart_rel=[0.0, 0.0, 0.0]):
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
            return (
                3.0
                / 16.0
                * np.sqrt(1.0 / (np.pi))
                * (35 * z**4 - 30 * z**2 * r**2 + 3 * r**4)
                / r**4
            )
        elif (l, m) == (4, 1):
            return 3.0 / 4.0 * np.sqrt(5.0 / (2 * np.pi)) * x * (7 * z**3 - 3 * z * r**2) / r**4
        elif (l, m) == (4, 2):
            return 3.0 / 8.0 * np.sqrt(5.0 / (np.pi)) * (x**2 - y**2) * (7 * z**2 - r**2) / r**4
        elif (l, m) == (4, 3):
            return 3.0 / 4.0 * np.sqrt(35.0 / (2 * np.pi)) * x * z * (x**2 - 3 * y**2) / r**4
        elif (l, m) == (4, 4):
            return (
                3.0
                / 16.0
                * np.sqrt(35.0 / (np.pi))
                * (x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2))
                / r**4
            )
        elif (l, m) == (5, -5):
            return (
                3.0
                / 16.0
                * np.sqrt(77.0 / (2 * np.pi))
                * (5 * x**4 * y - 10 * x**2 * y**3 + y**5)
                / r**5
            )
        elif (l, m) == (5, -4):
            return 3.0 / 16.0 * np.sqrt(385.0 / np.pi) * 4 * x * y * z * (x**2 - y**2) / r**5
        elif (l, m) == (5, -3):
            return (
                1.0
                / 16.0
                * np.sqrt(385.0 / (2 * np.pi))
                * -1
                * (y**3 - 3 * x**2 * y)
                * (9 * z**2 - r**2)
                / r**5
            )
        elif (l, m) == (5, -2):
            return 1.0 / 8.0 * np.sqrt(1155 / np.pi) * 2 * x * y * (3 * z**3 - z * r**2) / r**5
        elif (l, m) == (5, -1):
            return (
                1.0 / 16.0 * np.sqrt(165 / np.pi) * y * (21 * z**4 - 14 * z**2 * r**2 + r**4) / r**5
            )
        elif (l, m) == (5, 0):
            return (
                1.0
                / 16.0
                * np.sqrt(11 / np.pi)
                * (63 * z**5 - 70 * z**3 * r**2 + 15 * z * r**4)
                / r**5
            )
        elif (l, m) == (5, 1):
            return (
                1.0 / 16.0 * np.sqrt(165 / np.pi) * x * (21 * z**4 - 14 * z**2 * r**2 + r**4) / r**5
            )
        elif (l, m) == (5, 2):
            return 1.0 / 8.0 * np.sqrt(1155 / np.pi) * (x**2 - y**2) * (3 * z**3 - z * r**2) / r**5
        elif (l, m) == (5, 3):
            return (
                1.0
                / 16.0
                * np.sqrt(385.0 / (2 * np.pi))
                * (x**3 - 3 * x * y**2)
                * (9 * z**2 - r**2)
                / r**5
            )
        elif (l, m) == (5, 4):
            return (
                3.0
                / 16.0
                * np.sqrt(385.0 / np.pi)
                * (x**2 * z * (x**2 - 3 * y**2) - y**2 * z * (3 * x**2 - y**2))
                / r**5
            )
        elif (l, m) == (5, 5):
            return (
                3.0
                / 16.0
                * np.sqrt(77.0 / (2 * np.pi))
                * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4)
                / r**5
            )
        elif (l, m) == (6, -6):
            return (
                1.0
                / 64.0
                * np.sqrt(6006.0 / np.pi)
                * (6 * x**5 * y - 20 * x**3 * y**3 + 6 * x * y**5)
                / r**6
            )
        elif (l, m) == (6, -5):
            return (
                3.0
                / 32.0
                * np.sqrt(2002.0 / np.pi)
                * z
                * (5 * x**4 * y - 10 * x**2 * y**3 + y**5)
                / r**6
            )
        elif (l, m) == (6, -4):
            return (
                3.0
                / 32.0
                * np.sqrt(91.0 / np.pi)
                * 4
                * x
                * y
                * (11 * z**2 - r**2)
                * (x**2 - y**2)
                / r**6
            )
        elif (l, m) == (6, -3):
            return (
                1.0
                / 32.0
                * np.sqrt(2730.0 / np.pi)
                * -1
                * (11 * z**3 - 3 * z * r**2)
                * (y**3 - 3 * x**2 * y)
                / r**6
            )
        elif (l, m) == (6, -2):
            return (
                1.0
                / 64.0
                * np.sqrt(2730.0 / np.pi)
                * 2
                * x
                * y
                * (33 * z**4 - 18 * z**2 * r**2 + r**4)
                / r**6
            )
        elif (l, m) == (6, -1):
            return (
                1.0
                / 16.0
                * np.sqrt(273.0 / np.pi)
                * y
                * (33 * z**5 - 30 * z**3 * r**2 + 5 * z * r**4)
                / r**6
            )
        elif (l, m) == (6, 0):
            return (
                1.0
                / 32.0
                * np.sqrt(13.0 / np.pi)
                * (231 * z**6 - 315 * z**4 * r**2 + 105 * z**2 * r**4 - 5 * r**6)
                / r**6
            )
        elif (l, m) == (6, 1):
            return (
                1.0
                / 16.0
                * np.sqrt(273.0 / np.pi)
                * x
                * (33 * z**5 - 30 * z**3 * r**2 + 5 * z * r**4)
                / r**6
            )
        elif (l, m) == (6, 2):
            return (
                1.0
                / 64.0
                * np.sqrt(2730.0 / np.pi)
                * (x**2 - y**2)
                * (33 * z**4 - 18 * z**2 * r**2 + r**4)
                / r**6
            )
        elif (l, m) == (6, 3):
            return (
                1.0
                / 32.0
                * np.sqrt(2730.0 / np.pi)
                * (11 * z**3 - 3 * z * r**2)
                * (x**3 - 3 * x * y**2)
                / r**6
            )
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
            return (
                3.0
                / 32.0
                * np.sqrt(2002.0 / np.pi)
                * z
                * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4)
                / r**6
            )
        elif (l, m) == (6, 6):
            return (
                1.0
                / 64.0
                * np.sqrt(6006.0 / np.pi)
                * (x**6 - 15 * x**4 * y**2 + 15 * x**2 * y**4 - y**6)
                / r**6
            )
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
        test_S_lm = compute_S_l_m_debug(
            atomic_center_cart=R_cart,
            angular_momentum=l,
            magnetic_quantum_number=m,
            r_cart=r_cart,
        )
        ref_S_lm = (
            np.sqrt((4 * np.pi) / (2 * l + 1))
            * r_norm**l
            * Y_l_m_ref(l=l, m=m, r_cart_rel=r_cart_rel)
        )
        assert_almost_equal(test_S_lm, ref_S_lm, decimal=8)


# @pytest.mark.skip
@pytest.mark.parametrize(
    ["l", "m"],
    list(
        itertools.chain.from_iterable(
            [[pytest.param(l, m, id=f"l={l}, m={m}") for m in range(-l, l + 1)] for l in range(5)]
        )
    ),
)
def test_solid_harmonics(l, m):
    num_samples = 1
    R_cart = [0.0, 0.0, 1.0]
    r_cart_min, r_cart_max = -10.0, 10.0
    r_x_rand = (r_cart_max - r_cart_min) * np.random.rand(num_samples) + r_cart_min
    r_y_rand = (r_cart_max - r_cart_min) * np.random.rand(num_samples) + r_cart_min
    r_z_rand = (r_cart_max - r_cart_min) * np.random.rand(num_samples) + r_cart_min

    for r_cart in zip(r_x_rand, r_y_rand, r_z_rand):
        test_S_lm = compute_S_l_m_jax(
            R_cart=R_cart,
            l=l,
            m=m,
            r_cart=r_cart,
        )
        ref_S_lm = compute_S_l_m_debug(
            atomic_center_cart=R_cart,
            angular_momentum=l,
            magnetic_quantum_number=m,
            r_cart=r_cart,
        )
        assert_almost_equal(test_S_lm, ref_S_lm, decimal=8)


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

    aos_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_jax = compute_AOs_api(aos_data=aos_data, r_carts=r_carts, debug_flag=False)
    aos_debug = compute_AOs_api(aos_data=aos_data, r_carts=r_carts, debug_flag=True)

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

    aos_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_jax = compute_AOs_api(aos_data=aos_data, r_carts=r_carts, debug_flag=False)
    aos_debug = compute_AOs_api(aos_data=aos_data, r_carts=r_carts, debug_flag=True)

    assert np.allclose(aos_jax, aos_debug, rtol=1e-12, atol=1e-05)


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

    aos_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    ao_matrix_grad_x_auto, ao_matrix_grad_y_auto, ao_matrix_grad_z_auto = compute_AOs_grad_api(
        aos_data=aos_data, r_carts=r_carts, debug_flag=False
    )

    (
        ao_matrix_grad_x_numerical,
        ao_matrix_grad_y_numerical,
        ao_matrix_grad_z_numerical,
    ) = compute_AOs_grad_api(aos_data=aos_data, r_carts=r_carts, debug_flag=True)

    np.testing.assert_array_almost_equal(
        ao_matrix_grad_x_auto, ao_matrix_grad_x_numerical, decimal=7
    )
    np.testing.assert_array_almost_equal(
        ao_matrix_grad_y_auto, ao_matrix_grad_y_numerical, decimal=7
    )

    np.testing.assert_array_almost_equal(
        ao_matrix_grad_z_auto, ao_matrix_grad_z_numerical, decimal=7
    )

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

    aos_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    ao_matrix_grad_x_auto, ao_matrix_grad_y_auto, ao_matrix_grad_z_auto = compute_AOs_grad_api(
        aos_data=aos_data, r_carts=r_carts, debug_flag=False
    )

    (
        ao_matrix_grad_x_numerical,
        ao_matrix_grad_y_numerical,
        ao_matrix_grad_z_numerical,
    ) = compute_AOs_grad_api(aos_data=aos_data, r_carts=r_carts, debug_flag=True)

    np.testing.assert_array_almost_equal(
        ao_matrix_grad_x_auto, ao_matrix_grad_x_numerical, decimal=7
    )
    np.testing.assert_array_almost_equal(
        ao_matrix_grad_y_auto, ao_matrix_grad_y_numerical, decimal=7
    )

    np.testing.assert_array_almost_equal(
        ao_matrix_grad_z_auto, ao_matrix_grad_z_numerical, decimal=7
    )


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

    aos_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    ao_matrix_laplacian_numerical = compute_AOs_laplacian_api(
        aos_data=aos_data, r_carts=r_carts, debug_flag=True
    )

    ao_matrix_laplacian_auto = compute_AOs_laplacian_api(
        aos_data=aos_data, r_carts=r_carts, debug_flag=False
    )

    np.testing.assert_array_almost_equal(
        ao_matrix_laplacian_auto, ao_matrix_laplacian_numerical, decimal=5
    )

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

    aos_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    ao_matrix_laplacian_numerical = compute_AOs_laplacian_api(
        aos_data=aos_data, r_carts=r_carts, debug_flag=True
    )

    ao_matrix_laplacian_auto = compute_AOs_laplacian_api(
        aos_data=aos_data, r_carts=r_carts, debug_flag=False
    )

    np.testing.assert_array_almost_equal(
        ao_matrix_laplacian_auto, ao_matrix_laplacian_numerical, decimal=5
    )


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
        AO_data_debug(
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
        mo_ans_step_by_step.append(
            [compute_MO(mo_data=mo_data, r_cart=r_cart) for r_cart in r_carts]
        )
    mo_ans_step_by_step = np.array(mo_ans_step_by_step)

    aos_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    mo_ans_all_jax = compute_MOs_api(mos_data=mos_data, r_carts=r_carts, debug_flag=False)

    mo_ans_all_debug = compute_MOs_api(mos_data=mos_data, r_carts=r_carts, debug_flag=True)

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
        AO_data_debug(
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
        mo_ans_step_by_step.append(
            [compute_MO(mo_data=mo_data, r_cart=r_cart) for r_cart in r_carts]
        )
    mo_ans_step_by_step = np.array(mo_ans_step_by_step)

    aos_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    mo_ans_all_jax = compute_MOs_api(mos_data=mos_data, r_carts=r_carts, debug_flag=False)

    mo_ans_all_debug = compute_MOs_api(mos_data=mos_data, r_carts=r_carts, debug_flag=True)

    assert np.allclose(mo_ans_step_by_step, mo_ans_all_jax)
    assert np.allclose(mo_ans_step_by_step, mo_ans_all_debug)


def test_MOs_comparing_auto_and_numerical_grads():
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

    aos_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    mo_matrix_grad_x_auto, mo_matrix_grad_y_auto, mo_matrix_grad_z_auto = compute_MOs_grad_api(
        mos_data=mos_data, r_carts=r_carts, debug_flag=False
    )

    (
        mo_matrix_grad_x_numerical,
        mo_matrix_grad_y_numerical,
        mo_matrix_grad_z_numerical,
    ) = compute_MOs_grad_api(mos_data=mos_data, r_carts=r_carts, debug_flag=True)

    np.testing.assert_array_almost_equal(
        mo_matrix_grad_x_auto, mo_matrix_grad_x_numerical, decimal=6
    )
    np.testing.assert_array_almost_equal(
        mo_matrix_grad_y_auto, mo_matrix_grad_y_numerical, decimal=6
    )

    np.testing.assert_array_almost_equal(
        mo_matrix_grad_z_auto, mo_matrix_grad_z_numerical, decimal=6
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

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 3.0, 3.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    mo_coefficients = np.random.rand(num_mo, num_ao)

    aos_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    mo_matrix_grad_x_auto, mo_matrix_grad_y_auto, mo_matrix_grad_z_auto = compute_MOs_grad_api(
        mos_data=mos_data, r_carts=r_carts, debug_flag=False
    )

    (
        mo_matrix_grad_x_numerical,
        mo_matrix_grad_y_numerical,
        mo_matrix_grad_z_numerical,
    ) = compute_MOs_grad_api(mos_data=mos_data, r_carts=r_carts, debug_flag=True)

    np.testing.assert_array_almost_equal(
        mo_matrix_grad_x_auto, mo_matrix_grad_x_numerical, decimal=6
    )
    np.testing.assert_array_almost_equal(
        mo_matrix_grad_y_auto, mo_matrix_grad_y_numerical, decimal=6
    )

    np.testing.assert_array_almost_equal(
        mo_matrix_grad_z_auto, mo_matrix_grad_z_numerical, decimal=6
    )


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

    aos_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

    mo_matrix_laplacian_numerical = compute_MOs_laplacian_api(
        mos_data=mos_data, r_carts=r_carts, debug_flag=True
    )

    mo_matrix_laplacian_auto = compute_MOs_laplacian_api(
        mos_data=mos_data, r_carts=r_carts, debug_flag=False
    )

    np.testing.assert_array_almost_equal(
        mo_matrix_laplacian_auto, mo_matrix_laplacian_numerical, decimal=6
    )


@pytest.mark.parametrize(
    "filename",
    ["water_trexio.hdf5"],
    ids=["water_trexio.hdf5"],
)
def test_read_trexio_files(filename: str):
    read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", filename)
    )


def test_comparing_AO_and_MO_geminals():
    # test MOs
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"
        )
    )
    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    trial = 3
    for _ in range(trial):
        # Initialization
        r_up_carts = []
        r_dn_carts = []

        total_electrons = 0

        if coulomb_potential_data.ecp_flag:
            charges = np.array(structure_data.atomic_numbers) - np.array(
                coulomb_potential_data.z_cores
            )
        else:
            charges = np.array(structure_data.atomic_numbers)

        coords = structure_data.positions_cart

        # Place electrons around each nucleus
        for i in range(len(coords)):
            charge = charges[i]
            num_electrons = int(
                np.round(charge)
            )  # Number of electrons to place based on the charge

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

        geminal_mo = compute_geminal_all_elements_api(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        mo_lambda_matrix_paired, mo_lambda_matrix_unpaired = np.hsplit(
            geminal_mo_data.lambda_matrix, [geminal_mo_data.orb_num_dn]
        )

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
            compute_orb_api=compute_AOs_api,
            lambda_matrix=ao_lambda_matrix,
        )

        geminal_ao = compute_geminal_all_elements_api(
            geminal_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        # check if geminals with AO and MO representations are consistent
        np.testing.assert_array_almost_equal(geminal_ao, geminal_mo, decimal=15)

        det_geminal_mo = compute_det_geminal_all_elements_api(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        det_geminal_ao = compute_det_geminal_all_elements_api(
            geminal_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        np.testing.assert_almost_equal(det_geminal_ao, det_geminal_mo, decimal=15)


# @pytest.mark.skip
def test_debug_and_jax_SWCT_omega():
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"
        )
    )

    swct_data = SWCT_data(structure=structure_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -5.0, +5.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    omega_up_debug = evaluate_swct_omega_api(
        swct_data=swct_data, r_carts=r_up_carts, debug_flag=True
    )
    omega_dn_debug = evaluate_swct_omega_api(
        swct_data=swct_data, r_carts=r_dn_carts, debug_flag=True
    )
    omega_up_jax = evaluate_swct_omega_api(
        swct_data=swct_data, r_carts=r_up_carts, debug_flag=False
    )
    omega_dn_jax = evaluate_swct_omega_api(
        swct_data=swct_data, r_carts=r_dn_carts, debug_flag=False
    )

    np.testing.assert_almost_equal(omega_up_debug, omega_up_jax, decimal=6)
    np.testing.assert_almost_equal(omega_dn_debug, omega_dn_jax, decimal=6)

    domega_up_debug = evaluate_swct_domega_api(
        swct_data=swct_data, r_carts=r_up_carts, debug_flag=True
    )
    domega_dn_debug = evaluate_swct_domega_api(
        swct_data=swct_data, r_carts=r_dn_carts, debug_flag=True
    )
    domega_up_jax = evaluate_swct_domega_api(
        swct_data=swct_data, r_carts=r_up_carts, debug_flag=False
    )
    domega_dn_jax = evaluate_swct_domega_api(
        swct_data=swct_data, r_carts=r_dn_carts, debug_flag=False
    )

    np.testing.assert_almost_equal(domega_up_debug, domega_up_jax, decimal=6)
    np.testing.assert_almost_equal(domega_dn_debug, domega_dn_jax, decimal=6)


# @pytest.mark.skip
def test_numerial_and_auto_grads_ln_Det():
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"
        )
    )

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

    mo_lambda_matrix_paired, mo_lambda_matrix_unpaired = np.hsplit(
        geminal_mo_data.lambda_matrix, [geminal_mo_data.orb_num_dn]
    )

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
        compute_orb_api=compute_AOs_api,
        lambda_matrix=ao_lambda_matrix,
    )

    grad_ln_D_up_numerical, grad_ln_D_dn_numerical, sum_laplacian_ln_D_numerical = (
        compute_grads_and_laplacian_ln_Det_api(
            geminal_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )
    )

    grad_ln_D_up_auto, grad_ln_D_dn_auto, sum_laplacian_ln_D_auto = (
        compute_grads_and_laplacian_ln_Det_api(
            geminal_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=False,
        )
    )

    np.testing.assert_almost_equal(
        np.array(grad_ln_D_up_numerical), np.array(grad_ln_D_up_auto), decimal=6
    )
    np.testing.assert_almost_equal(
        np.array(grad_ln_D_dn_numerical), np.array(grad_ln_D_dn_auto), decimal=6
    )
    np.testing.assert_almost_equal(sum_laplacian_ln_D_numerical, sum_laplacian_ln_D_auto, decimal=1)


def test_comparing_values_with_TurboRVB_code():
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"
        )
    )

    jastrow_data = Jastrow_data(
        jastrow_two_body_data=None,
        jastrow_two_body_type="off",
        jastrow_three_body_data=None,
        jastrow_three_body_type="off",
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    old_r_up_carts = np.array(
        [
            [-0.64878536, -0.83275288, 0.33532629],
            [0.55271273, 0.72310605, 0.93443775],
            [0.66767275, 0.1206456, -0.36521208],
            [-0.93165236, -0.0120386, 0.33003036],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [-1.0347816, 1.26162081, 0.42301735],
            [-0.57843435, 1.03651987, -0.55091542],
            [-1.56091964, -0.58952149, -0.99268141],
            [0.61863233, -0.14903326, 0.51962683],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_dn_carts[3] = [0.618632327645002, -0.149033260668010, 0.131889254514777]

    WF_ratio = (
        evaluate_wavefunction_api(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction_api(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy_api(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = compute_bare_coulomb_potential_api(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        debug_flag=True,
    )

    vpot_bare_jax = compute_bare_coulomb_potential_api(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        debug_flag=False,
    )

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=6)

    vpot_ecp_debug = compute_ecp_coulomb_potential_api(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
        debug_flag=True,
    )

    vpot_ecp_jax = compute_ecp_coulomb_potential_api(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
        debug_flag=False,
    )

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax+vpot_ecp_debug} Ha")

    WF_ratio_ref_turborvb = 1.04447207308308
    kinc_ref_turborvb = 9.77796571601343
    vpot_ref_turborvb = -27.9099792589717
    vpotoff_ref_turborvb = 0.159136845080957

    # print(f"wf_ratio_ref={WF_ratio_ref_turborvb} Ha")
    # print(f"kinc_ref={kinc_ref_turborvb} Ha")
    # print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=8)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(
        vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3
    )
    np.testing.assert_almost_equal(
        vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3
    )


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

    aos_up_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_dn_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    j_matrix_up_up = np.random.rand(aos_up_data.num_ao, aos_up_data.num_ao + 1)
    j_matrix_dn_dn = np.random.rand(aos_dn_data.num_ao, aos_dn_data.num_ao + 1)
    j_matrix_up_dn = np.random.rand(aos_up_data.num_ao, aos_dn_data.num_ao)

    jastrow_three_body_data = Jastrow_three_body_data(
        orb_data_up_spin=aos_up_data,
        orb_data_dn_spin=aos_dn_data,
        j_matrix_up_up=j_matrix_up_up,
        j_matrix_dn_dn=j_matrix_dn_dn,
        j_matrix_up_dn=j_matrix_up_dn,
    )

    J3_debug = compute_Jastrow_three_body_api(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        debug_flag=True,
    )

    # print(f"J3_debug = {J3_debug}")

    J3_jax = compute_Jastrow_three_body_api(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        debug_flag=False,
    )

    # print(f"J3_jax = {J3_jax}")

    np.testing.assert_almost_equal(J3_debug, J3_jax, decimal=8)

    (
        grad_jastrow_J3_up_debug,
        grad_jastrow_J3_dn_debug,
        sum_laplacian_J3_debug,
    ) = compute_grads_and_laplacian_Jastrow_three_body_api(
        jastrow_three_body_data,
        r_up_carts,
        r_dn_carts,
        True,
    )

    # print(f"grad_jastrow_J3_up_debug = {grad_jastrow_J3_up_debug}")
    # print(f"grad_jastrow_J3_dn_debug = {grad_jastrow_J3_dn_debug}")
    # print(f"sum_laplacian_J3_debug = {sum_laplacian_J3_debug}")

    grad_jastrow_J3_up_jax, grad_jastrow_J3_dn_jax, sum_laplacian_J3_jax = (
        compute_grads_and_laplacian_Jastrow_three_body_api(
            jastrow_three_body_data,
            r_up_carts,
            r_dn_carts,
            False,
        )
    )

    # print(f"grad_jastrow_J3_up_jax = {grad_jastrow_J3_up_jax}")
    # print(f"grad_jastrow_J3_dn_jax = {grad_jastrow_J3_dn_jax}")
    # print(f"sum_laplacian_J3_jax = {sum_laplacian_J3_jax}")

    np.testing.assert_almost_equal(grad_jastrow_J3_up_debug, grad_jastrow_J3_up_jax, decimal=4)
    np.testing.assert_almost_equal(grad_jastrow_J3_dn_debug, grad_jastrow_J3_dn_jax, decimal=4)
    np.testing.assert_almost_equal(sum_laplacian_J3_debug, sum_laplacian_J3_jax, decimal=4)


def test_numerical_and_auto_grads_Jastrow_twobody_part():
    # test MOs
    num_r_up_cart_samples = 5
    num_r_dn_cart_samples = 2

    r_cart_min, r_cart_max = -3.0, 3.0

    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min

    jastrow_two_body_data = Jastrow_two_body_data(
        param_anti_parallel_spin=1.0, param_parallel_spin=1.0
    )
    J2_debug = compute_Jastrow_two_body_api(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        debug_flag=True,
    )

    # print(f"jastrow_two_body_debug = {jastrow_two_body_debug}")

    J2_jax = compute_Jastrow_two_body_api(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        debug_flag=False,
    )

    # print(f"jastrow_two_body_jax = {jastrow_two_body_jax}")

    np.testing.assert_almost_equal(J2_debug, J2_jax, decimal=10)

    (
        grad_J2_up_debug,
        grad_J2_dn_debug,
        sum_laplacian_J2_debug,
    ) = compute_grads_and_laplacian_Jastrow_two_body_api(
        jastrow_two_body_data,
        r_up_carts,
        r_dn_carts,
        True,
    )

    # print(f"grad_J2_up_debug = {grad_J2_up_debug}")
    # print(f"grad_J2_dn_debug = {grad_J2_dn_debug}")
    # print(f"sum_laplacian_J2_debug = {sum_laplacian_J2_debug}")

    grad_J2_up_jax, grad_J2_dn_jax, sum_laplacian_J2_jax = (
        compute_grads_and_laplacian_Jastrow_two_body_api(
            jastrow_two_body_data,
            r_up_carts,
            r_dn_carts,
            False,
        )
    )

    # print(f"grad_J2_up_jax = {grad_J2_up_jax}")
    # print(f"grad_J2_dn_jax = {grad_J2_dn_jax}")
    # print(f"sum_laplacian_J2_jax = {sum_laplacian_J2_jax}")

    np.testing.assert_almost_equal(grad_J2_up_debug, grad_J2_up_jax, decimal=8)
    np.testing.assert_almost_equal(grad_J2_dn_debug, grad_J2_dn_jax, decimal=8)
    np.testing.assert_almost_equal(sum_laplacian_J2_debug, sum_laplacian_J2_jax, decimal=4)


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
