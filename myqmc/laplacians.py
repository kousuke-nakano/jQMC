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
class Laplacian_data:
    """
    The class contains data for computing laplacians

    Args:
        jastrow_data (Jastrow_data)
        geminal_data (Geminal_data)
    """

    jastrow_data: Jastrow_data = None
    geminal_data: Geminal_data = None

    def __post_init__(self) -> None:
        pass


def compute_lapcalian(
    laplacian_data: Laplacian_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """
    The method is for computing laplacians of electrons at (r_up_carts, r_dn_carts).

    Args:
        laplacian_data (Laplacian_data): an instance of Laplacian_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        Arrays containing values of the geminal function with r_up_carts and r_dn_carts. (dim: N_e^{up}, N_e^{up})
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
