"""Molecular Orbital module"""

# python modules
from dataclasses import dataclass, field
import scipy
import numpy as np
import numpy.typing as npt
import itertools

# jax modules
import jax
from jax import vmap, jit
import jax.numpy as jnp

# JAX float64
jax.config.update("jax_enable_x64", True)

# set logger
from logging import getLogger, StreamHandler, Formatter

# myqmc module
from .atomic_orbital import AO_data, compute_AO, AOs_data, compute_AOs_api

logger = getLogger("myqmc").getChild(__name__)


@dataclass
class MOs_data:
    """
    The class contains data for computing a molecular orbitals.

    Args:
        num_mo: The number of MOs.
        mo_coefficients (npt.NDArray[np.float64|np.complex128]): array of MO coefficients. dim. num_mo, num_ao
        aos_data (AOs_data): aos_data instances
    """

    num_mo: int = 0
    mo_coefficients: npt.NDArray[np.float64 | np.complex128] = None
    aos_data: AOs_data = None

    def __post_init__(self) -> None:
        if self.mo_coefficients.shape != (self.num_mo, self.aos_data.num_ao):
            logger.error(
                f"dim. of ao_coefficients = {self.mo_coefficients.shape} is wrong. Inconsistent with the expected value = {(self.num_mo, self.aos_data.num_ao)}"
            )
            raise ValueError


def compute_MOs_api(
    mos_data: MOs_data, r_carts: npt.NDArray[np.float64], jax_flag: bool = True
) -> npt.NDArray[np.float64]:
    """
    The class contains information for computing molecular orbitals at r_carts simlunateously.

    Args:
        mos_data (MOs_data): an instance of MOs_data
        r_carts: Cartesian coordinates of electrons (dim: N_e, 3)
        jax_flag: if False, AOs are computed one by one using compute_AO_debug

    Returns:
        Arrays containing values of the MOs at r_carts. (dim: num_mo, N_e)
    """

    if jax_flag:
        answer = jnp.dot(
            mos_data.mo_coefficients,
            compute_AOs_api(aos_data=mos_data.aos_data, r_carts=r_carts, jax_flag=True),
        )
    else:
        answer = np.dot(
            mos_data.mo_coefficients,
            compute_AOs_api(
                aos_data=mos_data.aos_data, r_carts=r_carts, jax_flag=False
            ),
        )

    if answer.shape != (mos_data.num_mo, len(r_carts)):
        logger.error(
            f"answer.shape = {answer.shape} is inconsistent with the expected one = {(mos_data.num_mo, len(r_carts))}"
        )
        raise ValueError

    return answer


def compute_MOs_overlap_matrix(mos_data: MOs_data, method: str = "numerical"):
    """
    The method is for computing overlap matrix (S) of the given molecular orbitals

    Args:
        mo_datas (MOs_data): an instance of MOs_data
        method: method to compute S, numerical or analytical. numerical is just for the debugging purpose.

    Returns:
        Arrays containing the overlap matrix (S) of the given MOs (dim: num_mo, num_mo)
    """

    if method == "numerical":
        nx = 300
        x_min = -5.0
        x_max = 5.0

        ny = 300
        y_min = -5.0
        y_max = 5.0

        nz = 300
        z_min = -5.0
        z_max = 5.0

        x, w_x = scipy.special.roots_legendre(n=nx)
        y, w_y = scipy.special.roots_legendre(n=ny)
        z, w_z = scipy.special.roots_legendre(n=nz)

        # Use itertools.product to generate all combinations of points across dimensions
        points = list(itertools.product(x, y, z))
        weights = list(itertools.product(w_x, w_y, w_z))

        # Create the matrix of coordinates r from combinations
        r_prime = np.array(points)  # Shape: (n^3, 3)

        A = 1.0 / 2.0 * np.array([[x_max + x_min], [y_max + y_min], [z_max + z_min]])
        B = 1.0 / 2.0 * np.diag([x_max - x_min, y_max - y_min, z_max - z_min])

        # Create the weight vector W (calculate the product of each set of weights)
        W = np.array([w[0] * w[1] * w[2] for w in weights])  # Length: n^3
        Jacob = 1.0 / 8.0 * (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        W_prime = Jacob * np.tile(W, (mos_data.num_mo, 1))

        Psi = compute_MOs_api(
            mos_data=mos_data, r_carts=(A + np.dot(B, r_prime.T)).T, jax_flag=True
        )

        S = np.dot(Psi, (W_prime * Psi).T)

        return S

    else:
        raise NotImplementedError


@dataclass
class MO_data:
    """
    The class contains data for computing a molecular orbital. Just for testing purpose.
    For fast computations, use MOs_data and MOs.

    Args:
        mo_coefficients (list[float | complex]): List of coefficients of the AO.
        ao_data_l (list[AO_Data]): List of ao_data instances
    """

    mo_coefficients: list[float | complex] = field(default_factory=list)
    ao_data_l: list[AO_data] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.ao_data_l) != len(self.mo_coefficients):
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
        np.array(mo_data.mo_coefficients),
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
