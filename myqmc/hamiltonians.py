"""Geminal module"""

# python modules
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

# set logger
from logging import getLogger, StreamHandler, Formatter

from wavefunction import Wavefunction_data, compute_laplacian
from coulomb_potential import Coulomb_potential_data, compute_coulomb_potential

logger = getLogger("myqmc").getChild(__name__)


@dataclass
class Hamiltonian_data:
    """
    The class contains data for computing laplacians

    Args:
        coulomb_data (Coulomb_data)
        wavefunction_data (Wavefunction_data)
    """

    coulomb_potential_data: Coulomb_potential_data = None
    wavefunction_data: Wavefunction_data = None

    def __post_init__(self) -> None:
        pass


def compute_local_energy(
    hamiltonian_data: Hamiltonian_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """
    The method is for computing the local energy at (r_up_carts, r_dn_carts).

    Args:
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The value of local energy with the given wavefunction (float | complex)
    """

    T = compute_laplacian(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    V = compute_coulomb_potential(
        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return T + V


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
