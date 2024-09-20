"""SWCT module"""

# python modules
import os

# from dataclasses import dataclass
from logging import Formatter, StreamHandler, getLogger

# import jax
import jax
from jax import jacrev
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import numpy as jnp

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
    r_carts: npt.NDArray[np.float64],
    debug_flag: bool = False,
) -> npt.NDArray[np.float64]:
    """
    The method is for evaluate the omega(R_alpha, r_up_carts or r_dn_carts) for SWCT.

    Args:
        swct_data (SWCT_data): an instance of SWCT_data
        r_carts (npt.NDArray[np.float64]): Cartesian coordinates of up- or dn-spin electrons (dim: N_e, 3)
        debug_flag: if True, numerical derivatives are computed for debuging purpose

    Returns
    -------
        The omega_up (dim: N_a, N_e_up) and omega_dn (dim: N_a, N_e_dn)
        with the given structure (npt.NDArray[np.float64], npt.NDArray[np.float64])
    """
    if debug_flag:
        omega = evaluate_swct_omega_debug(swct_data=swct_data, r_carts=r_carts)
    else:
        omega = evaluate_swct_omega_jax(swct_data=swct_data, r_carts=r_carts)

    return omega


def evaluate_swct_omega_debug(
    swct_data: SWCT_data,
    r_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    R_carts = swct_data.structure.positions_cart
    omega = np.zeros((len(R_carts), len(r_carts)))

    for alpha, i in zip(range(len(R_carts)), range(len(r_up_carts))):
        kappa = 1.0 / np.linalg.norm(r_carts[i] - R_carts[alpha]) ** 4
        kappa_sum = np.sum(
            [1.0 / np.abs(r_carts[i] - R_carts[beta]) ** 4 for beta in range(len(R_carts))]
        )
        omega[alpha, i] = kappa / kappa_sum

    return omega


# WIP, to replace the for loops with vmap
def evaluate_swct_omega_jax(
    swct_data: SWCT_data,
    r_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    R_carts = swct_data.structure.positions_cart
    omega = jnp.zeros((len(R_carts), len(r_carts)))

    for alpha, i in zip(range(len(R_carts)), range(len(r_carts))):
        kappa = 1.0 / jnp.linalg.norm(r_carts[i] - R_carts[alpha]) ** 4
        kappa_sum = jnp.sum(
            jnp.array(
                [1.0 / jnp.abs(r_carts[i] - R_carts[beta]) ** 4 for beta in range(len(R_carts))]
            )
        )
        omega = omega.at[alpha, i].set(kappa / kappa_sum)

    return omega


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

    omega_up = evaluate_swct_omega_api(swct_data=swct_data, r_carts=r_up_carts, debug_flag=True)
    omega_dn = evaluate_swct_omega_api(swct_data=swct_data, r_carts=r_dn_carts, debug_flag=True)
    print(f'shape(omega_up) = {omega_up.shape}')
    print(f'shape(omega_dn) = {omega_dn.shape}')

    jacob_omega_up = jacrev(evaluate_swct_omega_api, argnums=1)(swct_data, r_up_carts)
    print(f'shape(jacob_omega_up) = {jacob_omega_up.shape}')
    jacob_omega_dn = jacrev(evaluate_swct_omega_api, argnums=1)(swct_data, r_dn_carts)
    print(f'shape(jacob_omega_dn) = {jacob_omega_dn.shape}')
