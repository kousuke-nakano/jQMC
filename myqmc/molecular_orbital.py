"""Atomic Orbital module"""

# python modules
from dataclasses import dataclass, field
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

# set logger
from logging import getLogger, StreamHandler, Formatter

# myqmc module
from atomic_orbital import AO

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

    coefficients: list[float | complex] = field(default_factory=list)
    atomic_orbitals: list[AO] = field(default_factory=list)


class MO:
    """
    The class contains information for computing a molecular orbital. Just for testing purpose.
    For fast computations, use MOs_data and MOs.

    Args:
        mo_data (MO_data): an instance of MO_data
    """

    def __init__(self, mo_data: MO_data):
        self.mo_data = mo_data

    def compute(self, r_cart: list[float]) -> float:
        """
        Compute the value of the MO at r_cart

        Args:
            r_cart: Cartesian coordinate of an electron

        Returns:
            Value of the MO value at r_cart.
        """
        return np.inner(
            np.array(self.mo_data.coefficients),
            np.array(
                [
                    ao_instance.compute(r_cart)
                    for ao_instance in self.mo_data.atomic_orbitals
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

    exponents = [13.0, 5.0, 1.0]
    coefficients = [0.001, 0.01, -1.0]

    ao_data = AO_data(exponents=exponents, coefficients=coefficients)
    ao_sphe = AO(ao_data=ao_data)
    r_cart = [0.0, 0.0, 1.0]
    print(ao_sphe.compute(r_cart=r_cart))
