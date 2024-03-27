import itertools
import pytest

import numpy as np
from numpy import linalg as LA
from numpy.testing import assert_almost_equal

from logging import getLogger, StreamHandler, Formatter

from ..myqmc.atomic_orbital import AO_sphe

log = getLogger("myqmc")
log.setLevel("DEBUG")
stream_handler = StreamHandler()
stream_handler.setLevel("DEBUG")
handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
stream_handler.setFormatter(handler_format)
log.addHandler(stream_handler)


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

    def S_l_m_ref(l=0, m=0, r_cart=[0.0, 0.0, 0.0]):
        """see https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics"""
        x, y, z = r_cart
        r = LA.norm(r_cart)
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

    num_samples = 40
    r_cart_min, r_cart_max = -10.0, 10.0
    r_x_rand = (r_cart_max - r_cart_min) * np.random.rand(num_samples) + r_cart_min
    r_y_rand = (r_cart_max - r_cart_min) * np.random.rand(num_samples) + r_cart_min
    r_z_rand = (r_cart_max - r_cart_min) * np.random.rand(num_samples) + r_cart_min

    ao_sphe = AO_sphe(angular_momentum=l, magnetic_quantum_number=m)

    for r_cart in zip(r_x_rand, r_y_rand, r_z_rand):
        test_Y_lm = ao_sphe.S_l_m(r_cart=r_cart) / LA.norm(r_cart) ** l
        ref_Y_lm = S_l_m_ref(l=l, m=m, r_cart=r_cart)
        assert_almost_equal(test_Y_lm, ref_Y_lm, decimal=12)
