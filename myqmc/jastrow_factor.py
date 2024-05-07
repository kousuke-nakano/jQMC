"""Geminal module"""

# python modules
import itertools
import numpy as np
import numpy.typing as npt

# jax modules
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
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
    The class contains data for evaluating a geminal function.

    Args:
        param_parallel_spin (float): parameter for parallel spins
        param_antiparallel_spin (float): parameter for anti-parallel spins
    """

    param_parallel_spin: float = struct.field(pytree_node=False)
    param_anti_parallel_spin: float = struct.field(pytree_node=False)

    def __post_init__(self) -> None:
        pass


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

    r_up_carts = jnp.array(r_up_carts)
    r_dn_carts = jnp.array(r_dn_carts)

    rel_r_carts_up_dn = jnp.array(
        [
            r_up_cart - r_dn_cart
            for (r_up_cart, r_dn_cart) in itertools.product(r_up_carts, r_dn_carts)
        ]
    )

    rel_r_carts_up_up = jnp.array(
        [
            r_up_cart_i - r_up_cart_j
            for (r_up_cart_i, r_up_cart_j) in itertools.combinations(r_up_carts, 2)
        ]
    )

    rel_r_carts_dn_dn = jnp.array(
        [
            r_dn_cart_i - r_dn_cart_j
            for (r_dn_cart_i, r_dn_cart_j) in itertools.combinations(r_dn_carts, 2)
        ]
    )

    vmap_two_body_jastrow_anti_parallel_spins = vmap(
        two_body_jastrow_anti_parallel_spins, in_axes=(None, 0)
    )

    vmap_two_body_jastrow_parallel_spins = vmap(
        two_body_jastrow_parallel_spins, in_axes=(None, 0)
    )

    two_body_jastrow = jnp.sum(
        vmap_two_body_jastrow_anti_parallel_spins(
            jastrow_two_body_data.param_anti_parallel_spin, rel_r_carts_up_dn
        )
    )

    two_body_jastrow += jnp.sum(
        vmap_two_body_jastrow_parallel_spins(
            jastrow_two_body_data.param_parallel_spin, rel_r_carts_up_up
        )
    )

    two_body_jastrow += jnp.sum(
        vmap_two_body_jastrow_parallel_spins(
            jastrow_two_body_data.param_parallel_spin, rel_r_carts_dn_dn
        )
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
            compute_grads_and_laplacian_Jastrow_two_body_debug(
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
        )
        J2_p_y_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_p_z_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_z_r_up_carts,
            r_dn_carts=r_dn_carts,
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
        )
        J2_m_y_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_z_up = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        grad_x_up.append(
            (np.log(np.abs(J2_p_x_up)) - np.log(np.abs(J2_m_x_up))) / (2.0 * diff_h)
        )
        grad_y_up.append(
            (np.log(np.abs(J2_p_y_up)) - np.log(np.abs(J2_m_y_up))) / (2.0 * diff_h)
        )
        grad_z_up.append(
            (np.log(np.abs(J2_p_z_up)) - np.log(np.abs(J2_m_z_up))) / (2.0 * diff_h)
        )

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
        )
        J2_p_y_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn_carts,
        )
        J2_p_z_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn_carts,
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
        )
        J2_m_y_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn_carts,
        )
        J2_m_z_dn = compute_Jastrow_two_body_api(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn_carts,
        )

        grad_x_dn.append(
            (np.log(np.abs(J2_p_x_dn)) - np.log(np.abs(J2_m_x_dn))) / (2.0 * diff_h)
        )
        grad_y_dn.append(
            (np.log(np.abs(J2_p_y_dn)) - np.log(np.abs(J2_m_y_dn))) / (2.0 * diff_h)
        )
        grad_z_dn.append(
            (np.log(np.abs(J2_p_z_dn)) - np.log(np.abs(J2_m_z_dn))) / (2.0 * diff_h)
        )

    grad_ln_D_up = np.array([grad_x_up, grad_y_up, grad_z_up]).T
    grad_ln_D_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn]).T

    sum_laplacian_J2 = 0.0

    return grad_ln_D_up, grad_ln_D_dn, sum_laplacian_J2


@WIP
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
    lambda_matrix_paired, lambda_matrix_unpaired = jnp.hsplit(
        geminal_data.lambda_matrix, [geminal_data.orb_num_dn]
    )

    # AOs/MOs
    ao_matrix_up = geminal_data.compute_orb(
        geminal_data.orb_data_up_spin, r_up_carts, debug_flag=False
    )
    ao_matrix_dn = geminal_data.compute_orb(
        geminal_data.orb_data_dn_spin, r_dn_carts, debug_flag=False
    )

    ao_matrix_up_grad_x, ao_matrix_up_grad_y, ao_matrix_up_grad_z = (
        geminal_data.compute_orb_grad_api(
            geminal_data.orb_data_up_spin, r_up_carts, debug_flag=False
        )
    )
    ao_matrix_dn_grad_x, ao_matrix_dn_grad_y, ao_matrix_dn_grad_z = (
        geminal_data.compute_orb_grad_api(
            geminal_data.orb_data_dn_spin, r_dn_carts, debug_flag=False
        )
    )
    ao_matrix_laplacian_up = geminal_data.compute_orb_laplacian_api(
        geminal_data.orb_data_up_spin, r_up_carts, debug_flag=False
    )
    ao_matrix_laplacian_dn = geminal_data.compute_orb_laplacian_api(
        geminal_data.orb_data_dn_spin, r_dn_carts, debug_flag=False
    )

    # compute Laplacians of Geminal
    geminal_paired = jnp.dot(
        ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn)
    )
    geminal_unpaired = jnp.dot(ao_matrix_up.T, lambda_matrix_unpaired)
    geminal = jnp.hstack([geminal_paired, geminal_unpaired])

    # up electron
    geminal_grad_up_x_paired = jnp.dot(
        ao_matrix_up_grad_x.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn)
    )
    geminal_grad_up_x_unpaired = jnp.dot(ao_matrix_up_grad_x.T, lambda_matrix_unpaired)
    geminal_grad_up_x = jnp.hstack(
        [geminal_grad_up_x_paired, geminal_grad_up_x_unpaired]
    )

    geminal_grad_up_y_paired = jnp.dot(
        ao_matrix_up_grad_y.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn)
    )
    geminal_grad_up_y_unpaired = jnp.dot(ao_matrix_up_grad_y.T, lambda_matrix_unpaired)
    geminal_grad_up_y = jnp.hstack(
        [geminal_grad_up_y_paired, geminal_grad_up_y_unpaired]
    )

    geminal_grad_up_z_paired = jnp.dot(
        ao_matrix_up_grad_z.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn)
    )
    geminal_grad_up_z_unpaired = jnp.dot(ao_matrix_up_grad_z.T, lambda_matrix_unpaired)
    geminal_grad_up_z = jnp.hstack(
        [geminal_grad_up_z_paired, geminal_grad_up_z_unpaired]
    )

    geminal_laplacian_up_paired = jnp.dot(
        ao_matrix_laplacian_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn)
    )
    geminal_laplacian_up_unpaired = jnp.dot(
        ao_matrix_laplacian_up.T, lambda_matrix_unpaired
    )
    geminal_laplacian_up = jnp.hstack(
        [geminal_laplacian_up_paired, geminal_laplacian_up_unpaired]
    )

    # dn electron
    geminal_grad_dn_x_paired = jnp.dot(
        ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn_grad_x)
    )
    geminal_grad_dn_x_unpaired = jnp.zeros(
        [
            geminal_data.num_electron_up,
            geminal_data.num_electron_up - geminal_data.num_electron_dn,
        ]
    )
    geminal_grad_dn_x = jnp.hstack(
        [geminal_grad_dn_x_paired, geminal_grad_dn_x_unpaired]
    )

    geminal_grad_dn_y_paired = jnp.dot(
        ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn_grad_y)
    )
    geminal_grad_dn_y_unpaired = jnp.zeros(
        [
            geminal_data.num_electron_up,
            geminal_data.num_electron_up - geminal_data.num_electron_dn,
        ]
    )
    geminal_grad_dn_y = jnp.hstack(
        [geminal_grad_dn_y_paired, geminal_grad_dn_y_unpaired]
    )

    geminal_grad_dn_z_paired = jnp.dot(
        ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn_grad_z)
    )
    geminal_grad_dn_z_unpaired = jnp.zeros(
        [
            geminal_data.num_electron_up,
            geminal_data.num_electron_up - geminal_data.num_electron_dn,
        ]
    )
    geminal_grad_dn_z = jnp.hstack(
        [geminal_grad_dn_z_paired, geminal_grad_dn_z_unpaired]
    )

    geminal_laplacian_dn_paired = jnp.dot(
        ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_laplacian_dn)
    )
    geminal_laplacian_dn_unpaired = jnp.zeros(
        [
            geminal_data.num_electron_up,
            geminal_data.num_electron_up - geminal_data.num_electron_dn,
        ]
    )
    geminal_laplacian_dn = jnp.hstack(
        [geminal_laplacian_dn_paired, geminal_laplacian_dn_unpaired]
    )

    geminal_inverse = jnp.linalg.inv(geminal)

    grad_ln_D_up_x = jnp.diag(jnp.dot(geminal_grad_up_x, geminal_inverse))
    grad_ln_D_up_y = jnp.diag(jnp.dot(geminal_grad_up_y, geminal_inverse))
    grad_ln_D_up_z = jnp.diag(jnp.dot(geminal_grad_up_z, geminal_inverse))
    grad_ln_D_dn_x = jnp.diag(jnp.dot(geminal_inverse, geminal_grad_dn_x))
    grad_ln_D_dn_y = jnp.diag(jnp.dot(geminal_inverse, geminal_grad_dn_y))
    grad_ln_D_dn_z = jnp.diag(jnp.dot(geminal_inverse, geminal_grad_dn_z))

    grad_ln_D_up = jnp.array([grad_ln_D_up_x, grad_ln_D_up_y, grad_ln_D_up_z]).T
    grad_ln_D_dn = jnp.array([grad_ln_D_dn_x, grad_ln_D_dn_y, grad_ln_D_dn_z]).T

    sum_laplacian_ln_D = (
        -1
        * (
            (jnp.trace(jnp.dot(geminal_grad_up_x, geminal_inverse) ** 2.0))
            + (jnp.trace(jnp.dot(geminal_grad_up_y, geminal_inverse) ** 2.0))
            + (jnp.trace(jnp.dot(geminal_grad_up_z, geminal_inverse) ** 2.0))
            + (jnp.trace(jnp.dot(geminal_inverse, geminal_grad_dn_x) ** 2.0))
            + (jnp.trace(jnp.dot(geminal_inverse, geminal_grad_dn_y) ** 2.0))
            + (jnp.trace(jnp.dot(geminal_inverse, geminal_grad_dn_z) ** 2.0))
        )
        + (jnp.trace(jnp.dot(geminal_laplacian_up, geminal_inverse)))
        + (jnp.trace(jnp.dot(geminal_inverse, geminal_laplacian_dn)))
    )

    return grad_ln_D_up, grad_ln_D_dn, sum_laplacian_ln_D


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

    r_cart_min, r_cart_max = -1.0, 1.0

    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_up_cart_samples, 3
    ) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_dn_cart_samples, 3
    ) + r_cart_min

    jastrow_two_body_data = Jastrow_two_body_data(
        param_anti_parallel_spin=1.5, param_parallel_spin=1.0
    )
    jastrow_two_body_debug = compute_Jastrow_two_body_api(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        debug_flag=True,
    )

    jastrow_two_body_jax = compute_Jastrow_two_body_api(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        debug_flag=False,
    )

    # """
    grad_jastrow_two_body_up_debug, grad_jastrow_two_body_dn_debug, _ = (
        compute_grads_and_laplacian_Jastrow_two_body_api(
            jastrow_two_body_data,
            r_up_carts,
            r_dn_carts,
            False,
        )
    )
    # """

    logger.debug(f"jastrow_two_body_debug = {jastrow_two_body_debug}")
    logger.debug(f"jastrow_two_body_jax = {jastrow_two_body_jax}")
    logger.debug(f"grad_jastrow_two_body_up_debug = {grad_jastrow_two_body_up_debug}")
    logger.debug(f"grad_jastrow_two_body_dn_debug = {grad_jastrow_two_body_dn_debug}")
