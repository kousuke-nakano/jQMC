"""Atomic Orbital module"""

# python modules
from dataclasses import dataclass, field
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

# set logger
from logging import getLogger, StreamHandler, Formatter

# myqmc module
from atomic_orbital import AO_data, compute_AO

logger = getLogger("myqmc").getChild(__name__)


@dataclass
class MO_data:
    """
    The class contains data for computing a molecular orbital using AO instances. Just for testing purpose.
    For fast computations, use MOs_data and MOs.

    Args:
        coefficients (list[float | complex]): List of coefficients of the AO.
        atomic_orbitals (list[AO]): List of AO instances.
    """

    ao_coefficients: list[float | complex] = field(default_factory=list)
    ao_data_l: list[AO_data] = field(default_factory=list)


def compute_MO(mo_data: MO_data, r_cart: list[float]) -> float:
    """
    The class contains information for computing a molecular orbital. Just for testing purpose.
    For fast computations, use MOs_data and compute_MOs.

    Args:
        mo_data (MO_data): an instance of MO_data
        r_cart: Cartesian coordinate of an electron

    Returns:
        Value of the MO value at r_cart.
    """

    return np.inner(
        np.array(mo_data.ao_coefficients),
        np.array(
            [
                compute_AO(ao_data=ao_data, r_cart=r_cart)
                for ao_data in mo_data.ao_data_l
            ]
        ),
    )


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    num_el = 10
    num_ao = 3
    num_ao_prim = 4
    orbital_indices = [0, 0, 1, 2]
    exponents = [50.0, 20.0, 10.0, 5.0]
    coefficients = [1.0, 1.0, 1.0, 0.5]
    angular_momentums = [1, 1, 1]
    magnetic_quantum_numbers = [0, 0, -1]

    num_r_cart_samples = 1
    num_R_cart_samples = num_ao
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_cart_samples, 3
    ) + r_cart_min
    R_cart = (R_cart_max - R_cart_min) * np.random.rand(
        num_R_cart_samples, 3
    ) + R_cart_min

    ao_coefficients = [1.0, 1.0, 1.0]

    ao_data_l = [
        AO_data(
            num_ao_prim=orbital_indices.count(i),
            atomic_center_cart=R_cart[i],
            exponents=[exponents[k] for (k, v) in enumerate(orbital_indices) if v == i],
            coefficients=[
                coefficients[k] for (k, v) in enumerate(orbital_indices) if v == i
            ],
            angular_momentum=angular_momentums[i],
            magnetic_quantum_number=magnetic_quantum_numbers[i],
        )
        for i in range(num_ao)
    ]

    mo_data = MO_data(ao_data_l=ao_data_l, ao_coefficients=ao_coefficients)
    print(compute_MO(mo_data=mo_data, r_cart=r_carts[0]))
