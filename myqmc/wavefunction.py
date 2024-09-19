"""Geminal module"""

# python modules
# from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from logging import getLogger, StreamHandler, Formatter

# import jax
import jax
from jax import jit
import jax.numpy as jnp
from flax import struct

from .determinant import (
    Geminal_data,
    compute_det_geminal_all_elements_api,
    compute_grads_and_laplacian_ln_Det_api,
)

from .jastrow_factor import (
    Jastrow_data,
    compute_Jastrow_part_api,
    compute_grads_and_laplacian_Jastrow_part_api,
)

# set logger
logger = getLogger("myqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


@struct.dataclass
class Wavefunction_data:
    """
    The class contains data for wavefunction

    Args:
        jastrow_data (Jastrow_data)
        geminal_data (Geminal_data)
    """

    jastrow_data: Jastrow_data = struct.field(pytree_node=True)
    geminal_data: Geminal_data = struct.field(pytree_node=True)

    def __post_init__(self) -> None:
        pass


@jit
def evaluate_ln_wavefunction_api(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """
    The method is for evaluate the logarithm of |wavefunction| (ln |Psi|) at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The value of the given wavefunction (float | complex)
    """

    return jnp.log(
        jnp.abs(
            evaluate_wavefunction_api(
                wavefunction_data=wavefunction_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )
        )
    )


def evaluate_wavefunction_api(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """
    The method is for evaluate wavefunction (Psi) at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The value of the given wavefunction (float | complex)
    """

    Jastrow_part = compute_Jastrow_part_api(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        debug_flag=False,
    )

    Determinant_part = compute_det_geminal_all_elements_api(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        debug_flag=False,
    )

    return jnp.exp(Jastrow_part) * Determinant_part


@jit
def compute_kinetic_energy_api(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """
    The method is for computing kinetic energy of the given WF at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The value of laplacian the given wavefunction (float | complex)
    """
    grad_J_up, grad_J_dn, sum_laplacian_J = (
        compute_grads_and_laplacian_Jastrow_part_api(
            jastrow_data=wavefunction_data.jastrow_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=False,
        )
    )

    grad_ln_D_up, grad_ln_D_dn, sum_laplacian_ln_D = (
        compute_grads_and_laplacian_ln_Det_api(
            geminal_data=wavefunction_data.geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=False,
        )
    )

    # compute kinetic energy
    L = (
        1.0
        / 2.0
        * (
            -(sum_laplacian_J + sum_laplacian_ln_D)
            - (
                jnp.sum((grad_J_up + grad_ln_D_up) * (grad_J_up + grad_ln_D_up))
                + jnp.sum((grad_J_dn + grad_ln_D_dn) * (grad_J_dn + grad_ln_D_dn))
            )
        )
    )

    return L


def compute_quantum_force(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    The method is for computing quantum forces at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The value of quantum forces of the given wavefunction -> return tuple[(N_e^{up}, 3), (N_e^{dn}, 3)]
    """

    grad_J_up, grad_J_dn, _ = 0, 0, 0  # tentative

    grad_ln_D_up, grad_ln_D_dn, _ = compute_grads_and_laplacian_ln_Det_api(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    grad_ln_WF_up = grad_J_up + grad_ln_D_up
    grad_ln_WF_dn = grad_J_dn + grad_ln_D_dn

    return 2.0 * grad_ln_WF_up, 2.0 * grad_ln_WF_dn


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
