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

    coefficients: list[float | complex] = field(default_factory=list)
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
        np.array(mo_data.coefficients),
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
