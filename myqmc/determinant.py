"""Geminal module"""

# python modules
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt

# set logger
from logging import getLogger, StreamHandler, Formatter

# myqmc module
from .atomic_orbital import AO_data, compute_AO, AOs_data, compute_AOs

logger = getLogger("myqmc").getChild(__name__)


@dataclass
class Geminal_ao_data:
    """
    The class contains data for computing geminal function.

    Args:
        aos_data (AOs_data): the number of MOs or AOs data.
        lambda_matrix (npt.NDArray[np.float64]): geminal matrix. (ao.num, ao.num) or (mo.num, mo.num)
    """

    aos_data: AOs_data = None
    lambda_matrix: npt.NDArray[np.float64] = None

    def __post_init__(self) -> None:
        if self.lambda_matrix.shape != (self.aos_data.num_ao, self.aos_data.num_ao):
            logger.error(
                f"dim. of lambda_matrix = {self.lambda_matrix.shape} is imcompatible with",
                f"dim. of aos_data.num_ao = ({aos_data.num_ao}, {aos_data.num_ao}).",
            )
            raise ValueError


def compute_geminal_all_elements(
    geminal_ao_data: Geminal_ao_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64 | np.complex128]:
    """
    The method is for computing the value of the given atomic orbital at r_carts

    Args:
        geminal_ao_data (Geminal_ao_data): an instance of Geminal_ao_data
        r_up_carts: Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts: Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        Arrays containing values of the geminal function with r_up_carts and r_dn_carts. (dim: num_ao, num_ao)
    """
    lambda_matrix = geminal_ao_data.lambda_matrix
    ao_matrix_dn = XXX
    ao_matrix_up = XXX

    answer = np.dot(ao_matrix_dn.T, np.dot(lambda_matrix, ao_matrix_up))

    if answer.shape != (len(r_up_carts), len(r_dn_carts)):
        logger.error(
            f"answer.shape = {answer.shape} is inconsistent with the expected one = {(len(r_up_carts), len(r_dn_carts))}"
        )
        raise ValueError

    return answer


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    num_r_cart_samples = 10
    num_R_cart_samples = 2
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_cart_samples, 3
    ) + r_cart_min
    R_cart = (R_cart_max - R_cart_min) * np.random.rand(
        num_R_cart_samples, 3
    ) + R_cart_min

    num_ao = 2
    num_ao_prim = 3
    orbital_indices = [0, 1, 1]
    exponents = [50.0, 20.0, 10.0]
    coefficients = [1.0, 1.0, 1.0]
    angular_momentums = [0, 1]
    magnetic_quantum_numbers = [0, 0]

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

    aos_compute_fast = compute_AOs(aos_data=aos_data, r_carts=r_carts, debug_flag=False)
    print(aos_compute_fast)
