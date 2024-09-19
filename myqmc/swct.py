"""SWCT module"""

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

# jaxQMC module
from .structure import Structure_data

# set logger
logger = getLogger("myqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


@struct.dataclass
class SWCT_data:
    """
    The class contains data for SWCT

    Args:
        structure_data (Structure_data)
    """

    structure: Structure_data = struct.field(pytree_node=True)

    def __post_init__(self) -> None:
        pass


def evaluate_swct_omega_api(
    swct_data: SWCT_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    debug_flag: bool = False,
) -> npt.NDArray[np.float64]:
    """
    The method is for evaluate the omega(R_alpha, r_up_carts, r_dn_carts) for SWCT.

    Args:
        swct_data (SWCT_data): an instance of SWCT_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        debug_flag: if True, numerical derivatives are computed for debuging purpose

    Returns:
        The omega_up (dim: N_a, N_e_up) and omega_dn (dim: N_a, N_e_dn)
        with the given structure (npt.NDArray[np.float64])
    """

    if debug_flag:
        evaluate_swct_omega_debug(
            swct_data=swct_data, r_up_carts=r_dn_carts, r_dn_carts=r_dn_carts
        )

    return np.nan


def evaluate_swct_omega_debug(
    swct_data: SWCT_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:

    swct_data.structure.positions_cart

    pass


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
