"""Geminal module"""

# python modules
import random
from dataclasses import dataclass
import itertools
import numpy as np
import numpy.typing as npt

# set logger
from logging import getLogger, StreamHandler, Formatter

logger = getLogger("myqmc").getChild(__name__)


@dataclass
class Bare_coulomb_potential_data:
    """
    The class contains data for computing geminal function.

    Args:
        num_atoms (int): the number of nuclei in the system
        atomic_center_carts (npt.NDArray[np.float64]): Centers of the nuclei (dim: num_atoms, 3).
        effective_nuclei_charges (list[float]]): the nuclei effective charges (dim: num_atoms).
    """

    num_atoms: int = 0
    atomic_center_carts: npt.NDArray[np.float64] = None
    effective_nuclei_charges: list[float] = None

    def __post_init__(self) -> None:
        pass


def compute_bare_coulomb_potential(
    bare_coulomb_potential_data: Bare_coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64 | np.complex128]:
    """
    The method is for computing the bare coulomb potentials including all electron-electron,
    electron-ion, and ion-ion interactions at (r_up_carts, r_dn_carts).

    Args:
        bare_coulomb_potential_data (Bare_coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        Arrays containing values of the geminal function with r_up_carts and r_dn_carts. (dim: N_e^{up}, N_e^{up})
    """

    R_carts = bare_coulomb_potential_data.atomic_center_carts
    R_charges = bare_coulomb_potential_data.effective_nuclei_charges
    r_up_charges = [-1 for _ in range(len(r_up_carts))]
    r_dn_charges = [-1 for _ in range(len(r_dn_carts))]

    all_carts = np.vstack([R_carts, r_up_carts, r_dn_carts])
    all_charges = R_charges + r_up_charges + r_dn_charges

    logger.debug(R_carts)
    logger.debug(r_up_carts)
    logger.debug(r_dn_carts)

    return np.sum(
        [
            (Z_a * Z_b) / np.linalg.norm(r_a - r_b)
            for (Z_a, r_a), (Z_b, r_b) in itertools.combinations(
                zip(all_charges, all_carts), 2
            )
        ]
    )


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    num_r_up_cart_samples = 5
    num_r_dn_cart_samples = 5
    num_R_cart_samples = 2
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = -1.0, 1.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_up_cart_samples, 3
    ) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_dn_cart_samples, 3
    ) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(
        num_R_cart_samples, 3
    ) + R_cart_min

    # Ensure that each element can be at least 1
    if num_r_up_cart_samples + num_r_dn_cart_samples < num_R_cart_samples:
        raise ValueError(
            "The total sum is less than the number of elements, no valid list can be generated."
        )

    # Subtract one from total_sum for each element to ensure they are at least 1
    remaining_sum = num_r_up_cart_samples + num_r_dn_cart_samples - num_R_cart_samples

    # Generate random split points within the range [0, remaining_sum]
    points = [0, remaining_sum] + [
        random.randint(0, remaining_sum) for _ in range(num_R_cart_samples - 1)
    ]
    points.sort()

    # Calculate the differences between consecutive points and add 1 to each to ensure each element is at least 1
    R_charges = [points[i + 1] - points[i] + 1 for i in range(num_R_cart_samples)]

    bare_coulomb_potential_data = Bare_coulomb_potential_data(
        num_atoms=num_R_cart_samples,
        atomic_center_carts=R_carts,
        effective_nuclei_charges=R_charges,
    )

    bare_coulomb = compute_bare_coulomb_potential(
        bare_coulomb_potential_data=bare_coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    logger.debug(bare_coulomb)
