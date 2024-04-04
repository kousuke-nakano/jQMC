"""Geminal module"""

# python modules
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt

# set logger
from logging import getLogger, StreamHandler, Formatter

# myqmc module
from atomic_orbital import AOs_data, compute_AOs

logger = getLogger("myqmc").getChild(__name__)


@dataclass
class Geminal_ao_data:
    """
    The class contains data for computing geminal function.

    Args:
        aos_data_up_spin (AOs_data): AOs data for up-spin.
        aos_data_dn_spin (AOs_data): AOs data for dn-spin.
        lambda_matrix (npt.NDArray[np.float64]): geminal matrix. dim. (aos_data_up_spin.num_ao, aos_data_up_spin.num_ao).
    """

    aos_data_up_spin: AOs_data = None
    aos_data_dn_spin: AOs_data = None
    lambda_matrix: npt.NDArray[np.float64] = None

    def __post_init__(self) -> None:
        if self.lambda_matrix.shape != (
            self.aos_data_up_spin.num_ao,
            self.aos_data_up_spin.num_ao,
        ):
            logger.error(
                f"dim. of lambda_matrix = {self.lambda_matrix.shape} is imcompatible with dim. of aos_data.num_ao = ({self.aos_data_up_spin.num_ao}, {self.aos_data_up_spin.num_ao}).",
            )
            raise ValueError


def compute_det_geminal_all_elements(
    geminal_ao_data: Geminal_ao_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> np.float64 | np.complex128:

    return np.linalg.det(
        compute_geminal_all_elements(
            geminal_ao_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
    )


def compute_geminal_all_elements(
    geminal_ao_data: Geminal_ao_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64 | np.complex128]:
    """
    The method is for computing geminal matrix elements with the given atomic orbitals at (r_up_carts, r_dn_carts).

    Args:
        geminal_ao_data (Geminal_ao_data): an instance of Geminal_ao_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        Arrays containing values of the geminal function with r_up_carts and r_dn_carts. (dim: num_ao, num_ao)
    """

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

    lambda_matrix = geminal_ao_data.lambda_matrix
    ao_matrix_up = compute_AOs(
        aos_data=geminal_ao_data.aos_data_up_spin, r_carts=r_up_carts
    )
    ao_matrix_dn = compute_AOs(
        aos_data=geminal_ao_data.aos_data_dn_spin, r_carts=r_dn_carts
    )

    if ao_matrix_up.shape != (geminal_ao_data.aos_data_up_spin.num_ao, len(r_up_carts)):
        logger.error(
            f"answer.shape = {ao_matrix_up.shape} is inconsistent with the expected one = {(len(geminal_ao_data.aos_data_up_spin.num_ao), len(r_up_carts))}"
        )
        raise ValueError

    if ao_matrix_dn.shape != (geminal_ao_data.aos_data_dn_spin.num_ao, len(r_dn_carts)):
        logger.error(
            f"answer.shape = {ao_matrix_dn.shape} is inconsistent with the expected one = {(len(geminal_ao_data.aos_data_up_spin.num_ao), len(r_dn_carts))}"
        )
        raise ValueError

    ao_matrix_dn_with_const_orbital = np.hstack(
        [
            ao_matrix_dn,
            np.ones((ao_matrix_dn.shape[0], len(r_up_carts) - len(r_dn_carts))),
        ]
    )

    # compute geminal
    answer = np.dot(
        ao_matrix_up.T, np.dot(lambda_matrix, ao_matrix_dn_with_const_orbital)
    )

    if answer.shape != (len(r_up_carts), len(r_up_carts)):
        logger.error(
            f"answer.shape = {answer.shape} is inconsistent with the expected one = {(len(r_up_carts), len(r_up_carts))}"
        )
        raise ValueError

    return answer


def compute_geminal_up_element(
    geminal_ao_data: Geminal_ao_data,
    r_up_cart: list[float],
    r_dn_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64 | np.complex128]:
    """
    The method is for computing one row of geminal matrix elements with the given atomic orbitals at (r_up_cart, r_dn_carts).

    Args:
        geminal_ao_data (Geminal_ao_data): an instance of Geminal_ao_data.
        r_up_cart (list[float]): Cartesian coordinate of up-spin electron (dim: 1, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        Arrays containing values of the geminal function with r_up_cart and r_dn_carts. (dim: num_ao, 1)
    """

    pass


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    num_r_up_cart_samples = 7
    num_r_dn_cart_samples = 5
    num_R_cart_samples = 2
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

    num_ao = 2
    num_ao_prim = 3
    orbital_indices = [0, 1, 1]
    exponents = [50.0, 20.0, 10.0]
    coefficients = [1.0, 1.0, 1.0]
    angular_momentums = [0, 1]
    magnetic_quantum_numbers = [0, 0]

    lambda_matrix = np.random.rand(num_ao, num_ao)

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

    geminal_ao_data = Geminal_ao_data(
        aos_data_up_spin=aos_data,
        aos_data_dn_spin=aos_data,
        lambda_matrix=lambda_matrix,
    )
    print(
        compute_geminal_all_elements(
            geminal_ao_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
    )

    print(
        compute_det_geminal_all_elements(
            geminal_ao_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
    )
