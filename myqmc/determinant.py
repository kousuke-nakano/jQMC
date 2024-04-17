"""Geminal module"""

# python modules
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
import numpy.typing as npt

# set logger
from logging import getLogger, StreamHandler, Formatter

# myqmc module
from .atomic_orbital import AOs_data, compute_AOs
from .molecular_orbital import MOs_data, compute_MOs

logger = getLogger("myqmc").getChild(__name__)


@dataclass
class Geminal_data:
    """
    The class contains data for computing geminal function.

    Args:
        num_electron_up (int): number of up electrons.
        num_electron_dn (int): number of dn electrons.
        orb_data_up_spin (AOs_data | MOs_data): AOs data or MOs data for up-spin.
        orb_data_dn_spin (AOs_data | MOs_data): AOs data or MOs data for dn-spin.
        compute_orb Callable[..., npt.NDArray[np.float64]]: Method to compute AOs or MOs values at an electronic configuration.
        lambda_matrix (npt.NDArray[np.float64]): geminal matrix. dim. (orb_data_up_spin.num_ao/mo, orb_data_dn_spin.num_ao/mo + num_electron_up - num_electron_dn)).
    """

    num_electron_up: int = 0
    num_electron_dn: int = 0
    orb_data_up_spin: AOs_data | MOs_data = None
    orb_data_dn_spin: AOs_data | MOs_data = None
    compute_orb: Callable[..., npt.NDArray[np.float64]] = None
    lambda_matrix: npt.NDArray[np.float64] = None

    def __post_init__(self) -> None:
        if self.lambda_matrix.shape != (
            self.orb_num_up,
            self.orb_num_dn + (self.num_electron_up - self.num_electron_dn),
        ):
            logger.error(
                f"dim. of lambda_matrix = {self.lambda_matrix.shape} is imcompatible with the expected one = ({self.orb_num_up}, {self.orb_num_dn + (self.num_electron_up - self.num_electron_dn)}).",
            )
            raise ValueError

    @property
    def orb_num_up(self) -> int:
        if self.compute_orb == compute_AOs:
            return self.orb_data_up_spin.num_ao
        elif self.compute_orb == compute_MOs:
            return self.orb_data_up_spin.num_mo
        else:
            raise NotImplementedError

    @property
    def orb_num_dn(self) -> int:
        if self.compute_orb == compute_AOs:
            return self.orb_data_dn_spin.num_ao
        elif self.compute_orb == compute_MOs:
            return self.orb_data_dn_spin.num_mo
        else:
            raise NotImplementedError


def compute_det_ao_geminal_all_elements(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> np.float64 | np.complex128:

    return np.linalg.det(
        compute_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
    )


def compute_geminal_all_elements(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64 | np.complex128]:
    """
    The method is for computing geminal matrix elements with the given atomic orbitals at (r_up_carts, r_dn_carts).

    Args:
        geminal_data (Geminal_data): an instance of Geminal_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        Arrays containing values of the geminal function with r_up_carts and r_dn_carts. (dim: N_e^{up}, N_e^{up})
    """

    if (
        len(r_up_carts) != geminal_data.num_electron_up
        or len(r_dn_carts) != geminal_data.num_electron_dn
    ):
        logger.info(
            f"Number of up and dn electrons (N_up, N_dn) = ({len(r_up_carts)}, {len(r_dn_carts)}) are not consistent with the expected values. (N_up, N_dn) = {geminal_data.num_electron_up}, {geminal_data.num_electron_dn})"
        )
        raise ValueError

    if len(r_up_carts) != len(r_dn_carts):
        if len(r_up_carts) - len(r_dn_carts) > 0:
            logger.info(
                f"Number of up and dn electrons are different. (N_el - N_dn = {len(r_up_carts) - len(r_dn_carts)})"
            )
        else:
            logger.error(
                f"Number of up electron is smaller than dn electrons. (N_el - N_dn = {len(r_up_carts) - len(r_dn_carts)})"
            )
            raise ValueError
    else:
        logger.info("There is no unpaired electrons.")

    lambda_matrix_paired, lambda_matrix_unpaired = np.hsplit(
        geminal_data.lambda_matrix, [geminal_data.orb_num_dn]
    )

    ao_matrix_up = geminal_data.compute_orb(geminal_data.orb_data_up_spin, r_up_carts)
    ao_matrix_dn = geminal_data.compute_orb(geminal_data.orb_data_dn_spin, r_dn_carts)

    if ao_matrix_up.shape != (geminal_data.orb_num_up, len(r_up_carts)):
        logger.error(
            f"answer.shape = {ao_matrix_up.shape} is inconsistent with the expected one = {(len(geminal_data.orb_num_up), len(r_up_carts))}"
        )
        raise ValueError

    if ao_matrix_dn.shape != (geminal_data.orb_num_dn, len(r_dn_carts)):
        logger.error(
            f"answer.shape = {ao_matrix_dn.shape} is inconsistent with the expected one = {(len(geminal_data.orb_num_dn), len(r_dn_carts))}"
        )
        raise ValueError

    # compute geminal values
    geminal_paired = np.dot(ao_matrix_up.T, np.dot(lambda_matrix_paired, ao_matrix_dn))
    geminal_unpaired = np.dot(ao_matrix_up.T, lambda_matrix_unpaired)
    geminal = np.hstack([geminal_paired, geminal_unpaired])

    if geminal.shape != (len(r_up_carts), len(r_up_carts)):
        logger.error(
            f"geminal.shape = {geminal.shape} is inconsistent with the expected one = {(len(r_up_carts), len(r_up_carts))}"
        )
        raise ValueError

    return geminal


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
