import itertools
import pytest

import numpy as np
from numpy import linalg as LA
from numpy.testing import assert_almost_equal

from logging import getLogger, StreamHandler, Formatter

from ..myqmc.atomic_orbital import (
    AO_data,
    compute_S_l_m,
    AOs_data,
    compute_AOs_api,
)

from ..myqmc.molecular_orbital import MO_data, MOs_data, compute_MO, compute_MOs
from ..myqmc.determinant import Geminal_data, compute_geminal_all_elements

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
            [
                [pytest.param(l, m, id=f"l={l}, m={m}") for m in range(-l, l + 1)]
                for l in range(5)
            ]
        )
    ),
)
def test_spherical_part_of_AO(l, m):
    def Y_l_m_ref(l=0, m=0, r_cart_rel=[0.0, 0.0, 0.0]):
        """see https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics"""
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
            return (
                1.0 / 4.0 * np.sqrt(35.0 / (2 * np.pi)) * y * (3 * x**2 - y**2) / r**3
            )
        elif (l, m) == (3, -2):
            return 1.0 / 2.0 * np.sqrt(105.0 / (np.pi)) * x * y * z / r**3
        elif (l, m) == (3, -1):
            return (
                1.0 / 4.0 * np.sqrt(21.0 / (2 * np.pi)) * y * (5 * z**2 - r**2) / r**3
            )
        elif (l, m) == (3, 0):
            return 1.0 / 4.0 * np.sqrt(7.0 / (np.pi)) * (5 * z**3 - 3 * z * r**2) / r**3
        elif (l, m) == (3, 1):
            return (
                1.0 / 4.0 * np.sqrt(21.0 / (2 * np.pi)) * x * (5 * z**2 - r**2) / r**3
            )
        elif (l, m) == (3, 2):
            return 1.0 / 4.0 * np.sqrt(105.0 / (np.pi)) * (x**2 - y**2) * z / r**3
        elif (l, m) == (3, 3):
            return (
                1.0 / 4.0 * np.sqrt(35.0 / (2 * np.pi)) * x * (x**2 - 3 * y**2) / r**3
            )
        # g orbitals
        elif (l, m) == (4, -4):
            return 3.0 / 4.0 * np.sqrt(35.0 / (np.pi)) * x * y * (x**2 - y**2) / r**4
        elif (l, m) == (4, -3):
            return (
                3.0
                / 4.0
                * np.sqrt(35.0 / (2 * np.pi))
                * y
                * z
                * (3 * x**2 - y**2)
                / r**4
            )
        elif (l, m) == (4, -2):
            return 3.0 / 4.0 * np.sqrt(5.0 / (np.pi)) * x * y * (7 * z**2 - r**2) / r**4
        elif (l, m) == (4, -1):
            return (
                3.0
                / 4.0
                * np.sqrt(5.0 / (2 * np.pi))
                * y
                * (7 * z**3 - 3 * z * r**2)
                / r**4
            )
        elif (l, m) == (4, 0):
            return (
                3.0
                / 16.0
                * np.sqrt(1.0 / (np.pi))
                * (35 * z**4 - 30 * z**2 * r**2 + 3 * r**4)
                / r**4
            )
        elif (l, m) == (4, 1):
            return (
                3.0
                / 4.0
                * np.sqrt(5.0 / (2 * np.pi))
                * x
                * (7 * z**3 - 3 * z * r**2)
                / r**4
            )
        elif (l, m) == (4, 2):
            return (
                3.0
                / 8.0
                * np.sqrt(5.0 / (np.pi))
                * (x**2 - y**2)
                * (7 * z**2 - r**2)
                / r**4
            )
        elif (l, m) == (4, 3):
            return (
                3.0
                / 4.0
                * np.sqrt(35.0 / (2 * np.pi))
                * x
                * z
                * (x**2 - 3 * y**2)
                / r**4
            )
        elif (l, m) == (4, 4):
            return (
                3.0
                / 16.0
                * np.sqrt(35.0 / (np.pi))
                * (x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2))
                / r**4
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
        test_S_lm = compute_S_l_m(
            atomic_center_cart=R_cart,
            angular_momentum=l,
            magnetic_quantum_number=m,
            r_cart=r_cart,
        )
        ref_S_lm = r_norm**l * Y_l_m_ref(l=l, m=m, r_cart_rel=r_cart_rel)
        assert_almost_equal(test_S_lm, ref_S_lm, decimal=10)


# @pytest.mark.skip
def test_AOs():
    num_el = 1000
    num_ao = 4
    num_ao_prim = 5
    orbital_indices = [0, 1, 1, 2, 3]
    exponents = [50.0, 50.0, 20.0, 10.0, 5.0]
    coefficients = [1.0, 1.0, 1.0, 1.0, 0.5]
    angular_momentums = [0, 1, 1, 1]
    magnetic_quantum_numbers = [0, 0, 0, -1]

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_cart_samples, 3
    ) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(
        num_R_cart_samples, 3
    ) + R_cart_min

    aos_data = AOs_data(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_jax = compute_AOs_api(aos_data=aos_data, r_carts=r_carts, jax_flag=True)
    aos_debug = compute_AOs_api(aos_data=aos_data, r_carts=r_carts, jax_flag=False)
    assert np.allclose(aos_jax, aos_debug, rtol=1e-05, atol=1e-08)


def test_MOs():
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
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_cart_samples, 3
    ) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(
        num_R_cart_samples, 3
    ) + R_cart_min

    mo_coefficients = np.random.rand(num_mo, num_ao)

    # compute each MO step by step
    mo_ans_step_by_step = []

    ao_data_l = [
        AO_data(
            num_ao_prim=orbital_indices.count(i),
            atomic_center_cart=R_carts[i],
            exponents=[exponents[k] for (k, v) in enumerate(orbital_indices) if v == i],
            coefficients=[
                coefficients[k] for (k, v) in enumerate(orbital_indices) if v == i
            ],
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

    aos_data = AOs_data(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_data = MOs_data(
        num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients
    )

    mo_ans_all_jax = compute_MOs(mos_data=mos_data, r_carts=r_carts, jax_flag=True)

    mo_ans_all_debug = compute_MOs(mos_data=mos_data, r_carts=r_carts, jax_flag=False)

    assert np.allclose(mo_ans_step_by_step, mo_ans_all_jax)
    assert np.allclose(mo_ans_step_by_step, mo_ans_all_debug)


def test_geminals():
    # test MOs
    num_r_up_cart_samples = 9
    num_r_dn_cart_samples = 5
    num_R_cart_samples = 2
    num_ao = 2
    num_mo_up = num_r_up_cart_samples  # Slater Determinant
    num_mo_dn = num_r_dn_cart_samples  # Slater Determinant
    num_ao_prim = 3
    orbital_indices = [0, 1, 1]
    exponents = [50.0, 20.0, 10.0]
    coefficients = [1.0, 1.0, 1.0]
    angular_momentums = [0, 1]
    magnetic_quantum_numbers = [0, 0]

    # generate matrices for the test
    mo_coefficients_up = np.random.rand(num_mo_up, num_ao)
    mo_coefficients_dn = np.random.rand(num_mo_dn, num_ao)
    mo_lambda_matrix_paired = np.eye(num_mo_up, num_mo_dn, k=0)
    mo_lambda_matrix_unpaired = np.eye(num_mo_up, num_mo_up - num_mo_dn, k=-num_mo_dn)
    mo_lambda_matrix = np.hstack([mo_lambda_matrix_paired, mo_lambda_matrix_unpaired])

    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_up_cart_samples, 3
    ) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_dn_cart_samples, 3
    ) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(
        num_R_cart_samples, 3
    ) + R_cart_min

    aos_up_data = AOs_data(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_dn_data = AOs_data(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_up_data = MOs_data(
        num_mo=num_mo_up, mo_coefficients=mo_coefficients_up, aos_data=aos_up_data
    )

    mos_dn_data = MOs_data(
        num_mo=num_mo_dn, mo_coefficients=mo_coefficients_dn, aos_data=aos_dn_data
    )

    geminal_mo_data = Geminal_data(
        num_electron_up=num_r_up_cart_samples,
        num_electron_dn=num_r_dn_cart_samples,
        orb_data_up_spin=mos_up_data,
        orb_data_dn_spin=mos_dn_data,
        compute_orb=compute_MOs,
        lambda_matrix=mo_lambda_matrix,
    )

    geminal_mo = compute_geminal_all_elements(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # generate matrices for the test
    ao_lambda_matrix_paired = np.dot(
        mo_coefficients_up.T, np.dot(mo_lambda_matrix_paired, mo_coefficients_dn)
    )
    ao_lambda_matrix_unpaired = np.dot(mo_coefficients_up.T, mo_lambda_matrix_unpaired)
    ao_lambda_matrix = np.hstack([ao_lambda_matrix_paired, ao_lambda_matrix_unpaired])

    geminal_ao_data = Geminal_data(
        num_electron_up=num_r_up_cart_samples,
        num_electron_dn=num_r_dn_cart_samples,
        orb_data_up_spin=aos_up_data,
        orb_data_dn_spin=aos_dn_data,
        compute_orb=compute_AOs_api,
        lambda_matrix=ao_lambda_matrix,
    )

    geminal_ao = compute_geminal_all_elements(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # check if geminals with AO and MO representations are consistent
    np.testing.assert_array_almost_equal(geminal_ao, geminal_mo, decimal=15)
