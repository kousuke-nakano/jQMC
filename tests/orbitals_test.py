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
    compute_AOs,
    compute_AO,
)

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
                for l in range(4)
            ]
        )
    ),
)
def test_spherical_part_of_AO(l, m):
    def S_l_m_ref(l=0, m=0, r_cart_rel=[0.0, 0.0, 0.0]):
        """see https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics"""
        x, y, z = r_cart_rel
        r = LA.norm(r_cart_rel)
        # s orbital
        if (l, m) == (0, 0):
            return 1.0 / 2.0 * np.sqrt(1.0 / np.pi)
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
        test_Y_lm = (
            compute_S_l_m(
                atomic_center_cart=R_cart,
                angular_momentum=l,
                magnetic_quantum_number=m,
                r_cart=r_cart,
            )
            / r_norm**l
        )
        ref_Y_lm = S_l_m_ref(l=l, m=m, r_cart_rel=r_cart_rel)
        assert_almost_equal(test_Y_lm, ref_Y_lm, decimal=12)

        ao_data = AO_data(
            num_ao_prim=1,
            atomic_center_cart=R_cart,
            exponents=[0.0],
            coefficients=[1.0],
            angular_momentum=l,
            magnetic_quantum_number=m,
        )

        test_Y_lm = compute_AO(ao_data=ao_data, r_cart=r_cart) / r_norm**l
        assert_almost_equal(test_Y_lm, ref_Y_lm, decimal=12)


# @pytest.mark.skip
def test_AOs():
    factor = 1000
    num_el = 1000
    num_ao = 3 * factor
    num_ao_prim = 4 * factor
    orbital_indices = [0, 0, 1, 2] * factor
    exponents = [50.0, 20.0, 10.0, 5.0] * factor
    coefficients = [1.0, 1.0, 1.0, 0.5] * factor
    angular_momentums = [1, 1, 1] * factor
    magnetic_quantum_numbers = [0, 0, -1] * factor

    num_r_cart_samples = num_el
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_cart_samples, 3
    ) + r_cart_min
    R_cart = (R_cart_max - R_cart_min) * np.random.rand(
        num_R_cart_samples, 3
    ) + R_cart_min

    aos_data = AOs_data(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_cart,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_debug = compute_AOs(aos_data=aos_data, r_carts=r_carts, debug_flag=True)
    aos_fast = compute_AOs(aos_data=aos_data, r_carts=r_carts, debug_flag=False)
    assert np.allclose(aos_fast, aos_debug, rtol=1e-05, atol=1e-08)
