"""SWCT module"""

# python modules
import os

# from dataclasses import dataclass
from logging import Formatter, StreamHandler, getLogger

# import jax
import jax
import numpy as np
import numpy.typing as npt
from flax import struct

# jaxQMC module
from .structure import Structure_data

# set logger
logger = getLogger("jqmc").getChild(__name__)

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

    Returns
    -------
        The omega_up (dim: N_a, N_e_up) and omega_dn (dim: N_a, N_e_dn)
        with the given structure (npt.NDArray[np.float64], npt.NDArray[np.float64])
    """
    if debug_flag:
        omega_up, omega_dn = evaluate_swct_omega_debug(
            swct_data=swct_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )
    else:
        raise NotImplementedError

    return (omega_up, omega_dn)


def evaluate_swct_omega_debug(
    swct_data: SWCT_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    R_carts = swct_data.structure.positions_cart
    omega_up = np.zeros((len(R_carts), len(r_up_carts)))
    omega_dn = np.zeros((len(R_carts), len(r_dn_carts)))

    for alpha, i in zip(range(len(R_carts)), range(len(r_up_carts))):
        kappa = 1.0 / np.linalg.norm(r_up_carts[i] - R_carts[alpha]) ** 4
        kappa_sum = np.sum(
            [1.0 / np.abs(r_up_carts[i] - R_carts[beta]) ** 4 for beta in range(len(R_carts))]
        )
        omega_up[alpha, i] = kappa / kappa_sum

    for alpha, i in zip(range(len(R_carts)), range(len(r_dn_carts))):
        kappa = 1.0 / np.linalg.norm(r_dn_carts[i] - R_carts[alpha]) ** 4
        kappa_sum = np.sum(
            [1.0 / np.abs(r_dn_carts[i] - R_carts[beta]) ** 4 for beta in range(len(R_carts))]
        )
        omega_dn[alpha, i] = kappa / kappa_sum

    return (omega_up, omega_dn)


if __name__ == "__main__":
    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    from .trexio_wrapper import read_trexio_file

    # water  cc-pVTZ with Mitas ccECP.
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "water_trexio.hdf5"))

    swct_data = SWCT_data(structure=structure_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -5.0, +5.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    print(
        evaluate_swct_omega_api(
            swct_data=swct_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts, debug_flag=True
        )
    )
