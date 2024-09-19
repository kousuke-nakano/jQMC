"""Geminal module"""

# python modules
import itertools
import numpy as np
import numpy.typing as npt

# jax modules
import jax
import jax.numpy as jnp
from jax import grad, jacrev, jit, vmap
from flax import struct

# set logger
from logging import getLogger, StreamHandler, Formatter

# set logger
logger = getLogger("myqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


# @dataclass
@struct.dataclass
class Jastrow_two_body_data:
    """
    The class contains data for evaluating a Jastrow function.

    Args:
        param_parallel_spin (float): parameter for parallel spins
        param_antiparallel_spin (float): parameter for anti-parallel spins
    """

    param_parallel_spin: float = struct.field(pytree_node=True)
    param_anti_parallel_spin: float = struct.field(pytree_node=True)

    def __post_init__(self) -> None:
        pass


# @dataclass
@struct.dataclass
class Jastrow_data:
    """
    The class contains data for evaluating a Jastrow function.

    Args:
        jastrow_two_body_data (Jastrow_two_body_data): parameter for parallel spins
        param_antiparallel_spin (float): parameter for anti-parallel spins
    """

    jastrow_two_body_data: Jastrow_two_body_data = struct.field(pytree_node=True)
    jastrow_two_body_type: str = struct.field(pytree_node=False)  # off, pade

    def __post_init__(self) -> None:
        pass


def compute_Jastrow_part_api(
    jastrow_data: Jastrow_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    debug_flag: bool = False,
) -> float:

    # two-body
    if jastrow_data.jastrow_two_body_type == "off":
        J2 = 0.0
    elif jastrow_data.jastrow_two_body_type == "pade":
        J2 = compute_Jastrow_two_body_api(
            jastrow_data.jastrow_two_body_data,
            r_up_carts,
            r_dn_carts,
            debug_flag=debug_flag,
        )
    else:
        raise NotImplementedError

    J = J2

    return J


def compute_grads_and_laplacian_Jastrow_part_api(
    jastrow_data: Jastrow_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    debug_flag: bool = False,
) -> tuple[
    npt.NDArray[np.float64 | np.complex128],
    npt.NDArray[np.float64 | np.complex128],
    float | complex,
]:
    # two-body
    if jastrow_data.jastrow_two_body_type == "off":
        grad_J2_up, grad_J2_dn, sum_laplacian_J2 = 0.0, 0.0, 0.0
    elif jastrow_data.jastrow_two_body_type == "pade":
        grad_J2_up, grad_J2_dn, sum_laplacian_J2 = (
            compute_grads_and_laplacian_Jastrow_two_body_api(
                jastrow_data.jastrow_two_body_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                debug_flag=debug_flag,
            )
        )
    else:
        raise NotImplementedError

    grad_J_up = grad_J2_up
    grad_J_dn = grad_J2_dn
    sum_laplacian_J = sum_laplacian_J2

    return grad_J_up, grad_J_dn, sum_laplacian_J


def compute_Jastrow_two_body_api(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    debug_flag: bool = False,
) -> float:
    if debug_flag:
        return compute_Jastrow_two_body_debug(
            jastrow_two_body_data, r_up_carts, r_dn_carts
        )
    else:
        return compute_Jastrow_two_body_jax(
            jastrow_two_body_data, r_up_carts, r_dn_carts
        )


def compute_Jastrow_two_body_debug(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:

    def two_body_jastrow_anti_parallel_spins(
        param: float, rel_r_cart: npt.NDArray[np.float64]
    ) -> float:
        two_body_jastrow = (
            jnp.linalg.norm(rel_r_cart)
            / 2.0
            * (1.0 + param * jnp.linalg.norm(rel_r_cart)) ** (-1.0)
        )
        return two_body_jastrow

    def two_body_jastrow_parallel_spins(
        param: float, rel_r_cart: npt.NDArray[np.float64]
    ) -> float:
        two_body_jastrow = (
            jnp.linalg.norm(rel_r_cart)
            / 4.0
            * (1.0 + param * jnp.linalg.norm(rel_r_cart)) ** (-1.0)
        )
        return two_body_jastrow

    two_body_jastrow = (
        np.sum(
            [
                two_body_jastrow_anti_parallel_spins(
                    param=jastrow_two_body_data.param_anti_parallel_spin,
                    rel_r_cart=r_up_cart - r_dn_cart,
                )
                for (r_up_cart, r_dn_cart) in itertools.product(r_up_carts, r_dn_carts)
            ]
        )
        + np.sum(
            [
                two_body_jastrow_parallel_spins(
                    param=jastrow_two_body_data.param_parallel_spin,
                    rel_r_cart=r_up_cart_i - r_up_cart_j,
                )
                for (r_up_cart_i, r_up_cart_j) in itertools.combinations(r_up_carts, 2)
            ]
        )
        + np.sum(
            [
                two_body_jastrow_parallel_spins(
                    param=jastrow_two_body_data.param_parallel_spin,
                    rel_r_cart=r_dn_cart_i - r_dn_cart_j,
                )
                for (r_dn_cart_i, r_dn_cart_j) in itertools.combinations(r_dn_carts, 2)
            ]
        )
    )

    return two_body_jastrow


@jit
def compute_Jastrow_two_body_jax(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:

    def J2_anti_parallel_spins(r_cart_i, r_cart_j):
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j)
            / 2.0
            * (
                1.0
                + jastrow_two_body_data.param_anti_parallel_spin
                * jnp.linalg.norm(r_cart_i - r_cart_j)
            )
            ** (-1.0)
        )
        return two_body_jastrow

    def J2_parallel_spins(r_cart_i, r_cart_j):
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j)
            / 4.0
            * (
                1.0
                + jastrow_two_body_data.param_parallel_spin
                * jnp.linalg.norm(r_cart_i - r_cart_j)
            )
            ** (-1.0)
        )
        return two_body_jastrow

    vmap_two_body_jastrow_anti_parallel_spins = vmap(
        vmap(J2_anti_parallel_spins, in_axes=(None, 0)), in_axes=(0, None)
    )

    vmap_two_body_jastrow_parallel_spins = vmap(
        vmap(J2_parallel_spins, in_axes=(None, 0)), in_axes=(0, None)
    )

    two_body_jastrow = (
        jnp.sum(vmap_two_body_jastrow_anti_parallel_spins(r_up_carts, r_dn_carts))
        + 1.0
        / 2.0
        * jnp.sum(vmap_two_body_jastrow_parallel_spins(r_up_carts, r_up_carts))
        + 1.0
        / 2.0
        * jnp.sum(vmap_two_body_jastrow_parallel_spins(r_dn_carts, r_dn_carts))
    )

    return two_body_jastrow


def compute_grads_and_laplacian_Jastrow_two_body_api(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    debug_flag: bool = False,
) -> tuple[
    npt.NDArray[np.float64 | np.complex128],
    npt.NDArray[np.float64 | np.complex128],
    float | complex,
]:
    """
    The method is for computing the gradients and the sum of laplacians of J at (r_up_carts, r_dn_carts).

    Args:
        jastrow_two_body_data (Jastrow_two_body_data): an instance of Jastrow_two_body_data class
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        debug_flag: if True, numerical derivatives are computed for debuging purpose
    Returns:
        the gradients(x,y,z) of J(twobody) and the sum of laplacians of J(twobody) at (r_up_carts, r_dn_carts).
    """

    if debug_flag:
        grad_J2_up, grad_J2_dn, sum_laplacian_J2 = (
            compute_grads_and_laplacian_Jastrow_two_body_debug(
                jastrow_two_body_data, r_up_carts, r_dn_carts
            )
        )
    else:
        grad_J2_up, grad_J2_dn, sum_laplacian_J2 = (
            compute_grads_and_laplacian_Jastrow_two_body_jax(
                jastrow_two_body_data, r_up_carts, r_dn_carts
            )
        )

    if grad_J2_up.shape != r_up_carts.shape:
        logger.error(
            f"grad_J2_up.shape = {grad_J2_up.shape} is inconsistent with the expected one = {r_up_carts.shape}"
        )
        raise ValueError

    if grad_J2_dn.shape != r_dn_carts.shape:
        logger.error(
            f"grad_J2_dn.shape = {grad_J2_dn.shape} is inconsistent with the expected one = {r_dn_carts.shape}"
        )
        raise ValueError

    return grad_J2_up, grad_J2_dn, sum_laplacian_J2


def compute_grads_and_laplacian_Jastrow_two_body_debug(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64 | np.complex128],
    npt.NDArray[np.float64 | np.complex128],
    float | complex,
]:
    diff_h = 1.0e-5

    # grad up
    grad_x_up = []
    grad_y_up = []
    grad_z_up = []
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up_carts = r_up_carts.copy()
        diff_p_y_r_up_carts = r_up_carts.copy()
        diff_p_z_r_up_carts = r_up_carts.copy()
        diff_p_x_r_up_carts[r_i][0] += diff_h
        diff_p_y_r_up_carts[r_i][1] += diff_h
        diff_p_z_r_up_carts[r_i][2] += diff_h

        J2_p_x_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_x_r_up_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )
        J2_p_y_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_y_r_up_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )
        J2_p_z_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_z_r_up_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )

        diff_m_x_r_up_carts = r_up_carts.copy()
        diff_m_y_r_up_carts = r_up_carts.copy()
        diff_m_z_r_up_carts = r_up_carts.copy()
        diff_m_x_r_up_carts[r_i][0] -= diff_h
        diff_m_y_r_up_carts[r_i][1] -= diff_h
        diff_m_z_r_up_carts[r_i][2] -= diff_h

        J2_m_x_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_x_r_up_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )
        J2_m_y_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_y_r_up_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )
        J2_m_z_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_z_r_up_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )

        grad_x_up.append((J2_p_x_up - J2_m_x_up) / (2.0 * diff_h))
        grad_y_up.append((J2_p_y_up - J2_m_y_up) / (2.0 * diff_h))
        grad_z_up.append((J2_p_z_up - J2_m_z_up) / (2.0 * diff_h))

    # grad dn
    grad_x_dn = []
    grad_y_dn = []
    grad_z_dn = []
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn_carts = r_dn_carts.copy()
        diff_p_y_r_dn_carts = r_dn_carts.copy()
        diff_p_z_r_dn_carts = r_dn_carts.copy()
        diff_p_x_r_dn_carts[r_i][0] += diff_h
        diff_p_y_r_dn_carts[r_i][1] += diff_h
        diff_p_z_r_dn_carts[r_i][2] += diff_h

        J2_p_x_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn_carts,
            debug_flag=True,
        )
        J2_p_y_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn_carts,
            debug_flag=True,
        )
        J2_p_z_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn_carts,
            debug_flag=True,
        )

        diff_m_x_r_dn_carts = r_dn_carts.copy()
        diff_m_y_r_dn_carts = r_dn_carts.copy()
        diff_m_z_r_dn_carts = r_dn_carts.copy()
        diff_m_x_r_dn_carts[r_i][0] -= diff_h
        diff_m_y_r_dn_carts[r_i][1] -= diff_h
        diff_m_z_r_dn_carts[r_i][2] -= diff_h

        J2_m_x_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn_carts,
            debug_flag=True,
        )
        J2_m_y_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn_carts,
            debug_flag=True,
        )
        J2_m_z_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn_carts,
            debug_flag=True,
        )

        grad_x_dn.append((J2_p_x_dn - J2_m_x_dn) / (2.0 * diff_h))
        grad_y_dn.append((J2_p_y_dn - J2_m_y_dn) / (2.0 * diff_h))
        grad_z_dn.append((J2_p_z_dn - J2_m_z_dn) / (2.0 * diff_h))

    grad_J2_up = np.array([grad_x_up, grad_y_up, grad_z_up]).T
    grad_J2_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn]).T

    # laplacian
    diff_h2 = 1.0e-3  # for laplacian

    J2_ref = compute_Jastrow_two_body_api(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        debug_flag=True,
    )

    sum_laplacian_J2 = 0.0

    # laplacians up
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up2_carts = r_up_carts.copy()
        diff_p_y_r_up2_carts = r_up_carts.copy()
        diff_p_z_r_up2_carts = r_up_carts.copy()
        diff_p_x_r_up2_carts[r_i][0] += diff_h2
        diff_p_y_r_up2_carts[r_i][1] += diff_h2
        diff_p_z_r_up2_carts[r_i][2] += diff_h2

        J2_p_x_up2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )
        J2_p_y_up2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )

        J2_p_z_up2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )

        diff_m_x_r_up2_carts = r_up_carts.copy()
        diff_m_y_r_up2_carts = r_up_carts.copy()
        diff_m_z_r_up2_carts = r_up_carts.copy()
        diff_m_x_r_up2_carts[r_i][0] -= diff_h2
        diff_m_y_r_up2_carts[r_i][1] -= diff_h2
        diff_m_z_r_up2_carts[r_i][2] -= diff_h2

        J2_m_x_up2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )
        J2_m_y_up2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )
        J2_m_z_up2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )

        gradgrad_x_up = (J2_p_x_up2 + J2_m_x_up2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_y_up = (J2_p_y_up2 + J2_m_y_up2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_z_up = (J2_p_z_up2 + J2_m_z_up2 - 2 * J2_ref) / (diff_h2**2)

        sum_laplacian_J2 += gradgrad_x_up + gradgrad_y_up + gradgrad_z_up

    # laplacians dn
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn2_carts = r_dn_carts.copy()
        diff_p_y_r_dn2_carts = r_dn_carts.copy()
        diff_p_z_r_dn2_carts = r_dn_carts.copy()
        diff_p_x_r_dn2_carts[r_i][0] += diff_h2
        diff_p_y_r_dn2_carts[r_i][1] += diff_h2
        diff_p_z_r_dn2_carts[r_i][2] += diff_h2

        J2_p_x_dn2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn2_carts,
            debug_flag=True,
        )
        J2_p_y_dn2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn2_carts,
            debug_flag=True,
        )

        J2_p_z_dn2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn2_carts,
            debug_flag=True,
        )

        diff_m_x_r_dn2_carts = r_dn_carts.copy()
        diff_m_y_r_dn2_carts = r_dn_carts.copy()
        diff_m_z_r_dn2_carts = r_dn_carts.copy()
        diff_m_x_r_dn2_carts[r_i][0] -= diff_h2
        diff_m_y_r_dn2_carts[r_i][1] -= diff_h2
        diff_m_z_r_dn2_carts[r_i][2] -= diff_h2

        J2_m_x_dn2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn2_carts,
            debug_flag=True,
        )
        J2_m_y_dn2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn2_carts,
            debug_flag=True,
        )
        J2_m_z_dn2 = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn2_carts,
            debug_flag=True,
        )

        gradgrad_x_dn = (J2_p_x_dn2 + J2_m_x_dn2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_y_dn = (J2_p_y_dn2 + J2_m_y_dn2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_z_dn = (J2_p_z_dn2 + J2_m_z_dn2 - 2 * J2_ref) / (diff_h2**2)

        sum_laplacian_J2 += gradgrad_x_dn + gradgrad_y_dn + gradgrad_z_dn

    return grad_J2_up, grad_J2_dn, sum_laplacian_J2


@jit
def compute_grads_and_laplacian_Jastrow_two_body_jax(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[jnp.float64],
    r_dn_carts: npt.NDArray[jnp.float64],
) -> tuple[
    npt.NDArray[jnp.float64 | jnp.complex128],
    npt.NDArray[jnp.float64 | jnp.complex128],
    float | complex,
]:

    def J2_anti_parallel_spins(r_cart_i, r_cart_j):
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j)
            / 2.0
            * (
                1.0
                + jastrow_two_body_data.param_anti_parallel_spin
                * jnp.linalg.norm(r_cart_i - r_cart_j)
            )
            ** (-1.0)
        )
        return two_body_jastrow

    def J2_parallel_spins(r_cart_i, r_cart_j):
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j)
            / 4.0
            * (
                1.0
                + jastrow_two_body_data.param_parallel_spin
                * jnp.linalg.norm(r_cart_i - r_cart_j)
            )
            ** (-1.0)
        )
        return two_body_jastrow

    # compute grad

    # Function to compute the sum of gradients for each r_j_cart
    def compute_grad_sum_anti_parallel(r_i_cart, r_j_carts):
        grads = vmap(
            lambda r_j_cart: grad(J2_anti_parallel_spins, argnums=0)(r_i_cart, r_j_cart)
        )(r_j_carts)
        # Calculate norms and reshape for broadcasting
        norms = jnp.linalg.norm(r_i_cart - r_j_carts, axis=1)
        norms = norms[:, None]  # Reshape to (n, 1) for broadcasting with grads
        non_zero_grads = jnp.where(norms != 0, grads, 0)
        return jnp.sum(non_zero_grads, axis=0)

    # Apply the above function to all r_i_cart
    def grad_vmap_two_body_jastrow_anti_parallel_spins(r_i_carts, r_j_carts):
        return vmap(
            lambda r_i_cart: compute_grad_sum_anti_parallel(r_i_cart, r_j_carts)
        )(r_i_carts)

    # Function to compute the sum of gradients for each r_j_cart
    def compute_grad_sum_parallel(r_i_cart, r_j_carts):
        grads = vmap(
            lambda r_j_cart: grad(J2_parallel_spins, argnums=0)(r_i_cart, r_j_cart)
        )(r_j_carts)
        # Calculate norms and reshape for broadcasting
        norms = jnp.linalg.norm(r_i_cart - r_j_carts, axis=1)
        norms = norms[:, None]  # Reshape to (n, 1) for broadcasting with grads
        non_zero_grads = jnp.where(norms != 0, grads, 0)
        return jnp.sum(non_zero_grads, axis=0)

    # Apply the above function to all r_i_cart
    def grad_vmap_two_body_jastrow_parallel_spins(r_i_carts, r_j_carts):
        return vmap(lambda r_i_cart: compute_grad_sum_parallel(r_i_cart, r_j_carts))(
            r_i_carts
        )

    grad_J2_up = grad_vmap_two_body_jastrow_anti_parallel_spins(
        r_up_carts, r_dn_carts
    ) + grad_vmap_two_body_jastrow_parallel_spins(r_up_carts, r_up_carts)

    grad_J2_dn = grad_vmap_two_body_jastrow_anti_parallel_spins(
        r_dn_carts, r_up_carts
    ) + grad_vmap_two_body_jastrow_parallel_spins(r_dn_carts, r_dn_carts)

    # compute laplacian

    # Function to compute the sum of gradients for each r_j_cart
    def compute_laplacian_anti_parallel(r_i_cart, r_j_carts):
        laplacian = vmap(
            lambda r_j_cart: jnp.diag(
                jacrev(grad(J2_anti_parallel_spins, argnums=0), argnums=0)(
                    r_i_cart, r_j_cart
                )
            )
        )(r_j_carts)
        # Calculate norms and reshape for broadcasting
        norms = jnp.linalg.norm(r_i_cart - r_j_carts, axis=1)
        norms = norms[:, None]  # Reshape to (n, 1) for broadcasting with grads
        non_zero_laplacian = jnp.where(norms != 0, laplacian, 0)
        return jnp.sum(non_zero_laplacian, axis=0)

    # Apply the above function to all r_i_cart
    def laplacian_vmap_two_body_jastrow_anti_parallel_spins(r_i_carts, r_j_carts):
        return vmap(
            lambda r_i_cart: compute_laplacian_anti_parallel(r_i_cart, r_j_carts)
        )(r_i_carts)

    # Function to compute the sum of gradients for each r_j_cart
    def compute_laplacian_parallel(r_i_cart, r_j_carts):
        laplacian = vmap(
            lambda r_j_cart: jnp.diag(
                jacrev(grad(J2_parallel_spins, argnums=0), argnums=0)(
                    r_i_cart, r_j_cart
                )
            )
        )(r_j_carts)
        # Calculate norms and reshape for broadcasting
        norms = jnp.linalg.norm(r_i_cart - r_j_carts, axis=1)
        norms = norms[:, None]  # Reshape to (n, 1) for broadcasting with grads
        non_zero_laplacian = jnp.where(norms != 0, laplacian, 0)
        return jnp.sum(non_zero_laplacian, axis=0)

    # Apply the above function to all r_i_cart
    def laplacian_vmap_two_body_jastrow_parallel_spins(r_i_carts, r_j_carts):
        return vmap(lambda r_i_cart: compute_laplacian_parallel(r_i_cart, r_j_carts))(
            r_i_carts
        )

    laplacian_J2_up = laplacian_vmap_two_body_jastrow_anti_parallel_spins(
        r_up_carts, r_dn_carts
    ) + laplacian_vmap_two_body_jastrow_parallel_spins(r_up_carts, r_up_carts)

    laplacian_J2_dn = laplacian_vmap_two_body_jastrow_anti_parallel_spins(
        r_dn_carts, r_up_carts
    ) + laplacian_vmap_two_body_jastrow_parallel_spins(r_dn_carts, r_dn_carts)

    sum_laplacian_J2 = jnp.sum(laplacian_J2_up) + jnp.sum(laplacian_J2_dn)

    return grad_J2_up, grad_J2_dn, sum_laplacian_J2


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    # test MOs
    num_r_up_cart_samples = 5
    num_r_dn_cart_samples = 2

    r_cart_min, r_cart_max = -3.0, 3.0

    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_up_cart_samples, 3
    ) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_dn_cart_samples, 3
    ) + r_cart_min

    jastrow_two_body_data = Jastrow_two_body_data(
        param_anti_parallel_spin=1.0, param_parallel_spin=1.0
    )
    jastrow_two_body_debug = compute_Jastrow_two_body_api(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        debug_flag=True,
    )

    logger.debug(f"jastrow_two_body_debug = {jastrow_two_body_debug}")

    jastrow_two_body_jax = compute_Jastrow_two_body_api(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        debug_flag=False,
    )

    logger.debug(f"jastrow_two_body_jax = {jastrow_two_body_jax}")

    np.testing.assert_almost_equal(
        jastrow_two_body_debug, jastrow_two_body_jax, decimal=10
    )

    (
        grad_jastrow_two_body_up_debug,
        grad_jastrow_two_body_dn_debug,
        sum_laplacian_J2_debug,
    ) = compute_grads_and_laplacian_Jastrow_two_body_api(
        jastrow_two_body_data,
        r_up_carts,
        r_dn_carts,
        True,
    )

    logger.debug(f"grad_jastrow_two_body_up_debug = {grad_jastrow_two_body_up_debug}")
    logger.debug(f"grad_jastrow_two_body_dn_debug = {grad_jastrow_two_body_dn_debug}")
    logger.debug(f"sum_laplacian_J2_debug = {sum_laplacian_J2_debug}")

    grad_jastrow_two_body_up_jax, grad_jastrow_two_body_dn_jax, sum_laplacian_J2_jax = (
        compute_grads_and_laplacian_Jastrow_two_body_api(
            jastrow_two_body_data,
            r_up_carts,
            r_dn_carts,
            False,
        )
    )

    logger.debug(f"grad_jastrow_two_body_up_jax = {grad_jastrow_two_body_up_jax}")
    logger.debug(f"grad_jastrow_two_body_dn_jax = {grad_jastrow_two_body_dn_jax}")
    logger.debug(f"sum_laplacian_J2_jax = {sum_laplacian_J2_jax}")
