"""Atomic Orbital module"""

# python modules
from dataclasses import dataclass, field
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

# set logger
from logging import getLogger, StreamHandler, Formatter

# myqmc module
from .atomic_orbital import AO_data, compute_AO, AOs_data, compute_AOs

logger = getLogger("myqmc").getChild(__name__)


@dataclass
class MOs_data:
    """
    The class contains data for computing a molecular orbitals.

    Args:
        num_mo: The number of MOs.
        ao_coefficients (npt.NDArray[np.float64|np.complex128]): array of AO coefficients. dim. num_mo, num_ao
        aos_data (AOs_data): aos_data instances
    """

    num_mo: int = 0
    ao_coefficients: npt.NDArray[np.float64 | np.complex128] = None
    aos_data: AOs_data = None

    def __post_init__(self) -> None:
        if self.ao_coefficients.shape != (self.num_mo, self.aos_data.num_ao):
            logger.error(
                f"dim. of ao_coefficients = {self.ao_coefficients.shape} is wrong. Inconsistent with the expected value = {(self.num_mo, self.aos_data.num_ao)}"
            )
            raise ValueError


def compute_MOs(
    mos_data: MOs_data, r_carts: npt.NDArray[np.float64], debug_flag: bool = True
) -> npt.NDArray[np.float64]:
    """
    The class contains information for computing molecular orbitals at r_carts simlunateously.

    Args:
        mos_data (MOs_data): an instance of MOs_data
        r_carts: Cartesian coordinates of electrons (dim: N_e, 3)
        debug_flag: if True, AOs are computed one by one using compute_AO

    Returns:
        Arrays containing values of the MOs at r_carts. (dim: num_mo, N_e)
    """

    answer = np.dot(
        mos_data.ao_coefficients,
        compute_AOs(aos_data=mos_data.aos_data, r_carts=r_carts, debug_flag=debug_flag),
    )

    if answer.shape != (mos_data.num_mo, len(r_carts)):
        logger.error(
            f"answer.shape = {answer.shape} is inconsistent with the expected one = {(mos_data.num_mo, len(r_carts))}"
        )
        raise ValueError

    return answer


@dataclass
class MO_data:
    """
    The class contains data for computing a molecular orbital. Just for testing purpose.
    For fast computations, use MOs_data and MOs.

    Args:
        ao_coefficients (list[float | complex]): List of coefficients of the AO.
        ao_data_l (list[AO_Data]): List of ao_data instances
    """

    ao_coefficients: list[float | complex] = field(default_factory=list)
    ao_data_l: list[AO_data] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.ao_data_l) != len(self.ao_coefficients):
            logger.error("dim. of self.ao_data_l or len(self.coefficients is wrong")
            raise ValueError


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
