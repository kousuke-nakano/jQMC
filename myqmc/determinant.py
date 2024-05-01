"""Geminal module"""

# python modules
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from logging import getLogger, StreamHandler, Formatter

# jax modules
# from jax.debug import print as jprint
import jax
import jax.numpy as jnp
from jax import jit
from flax import struct

# myqmc module
from .atomic_orbital import (
    AOs_data_debug,
    compute_AOs_api,
    compute_AOs_grad_api,
    compute_AOs_laplacian_api,
)
from .molecular_orbital import (
    MOs_data,
    compute_MOs_api,
    compute_MOs_grad_api,
    compute_MOs_laplacian_api,
)

# set logger
logger = getLogger("myqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


# @dataclass
@struct.dataclass
class Geminal_data:
    """
    The class contains data for evaluating a geminal function.

    Args:
        num_electron_up (int): number of up electrons.
        num_electron_dn (int): number of dn electrons.
        orb_data_up_spin (AOs_data | MOs_data): AOs data or MOs data for up-spin.
        orb_data_dn_spin (AOs_data | MOs_data): AOs data or MOs data for dn-spin.
        compute_orb Callable[..., npt.NDArray[np.float64]]: Method to compute AOs or MOs values at an electronic configuration.
        lambda_matrix (npt.NDArray[np.float64]): geminal matrix. dim. (orb_data_up_spin.num_ao/mo, orb_data_dn_spin.num_ao/mo + num_electron_up - num_electron_dn)).
    """

    num_electron_up: int = struct.field(pytree_node=False)
    num_electron_dn: int = struct.field(pytree_node=False)
    orb_data_up_spin: AOs_data_debug | MOs_data = struct.field(pytree_node=True)
    orb_data_dn_spin: AOs_data_debug | MOs_data = struct.field(pytree_node=True)
    compute_orb_api: Callable[..., npt.NDArray[np.float64]] = struct.field(
        pytree_node=False
    )
    lambda_matrix: npt.NDArray[np.float64] = struct.field(pytree_node=True)

    def __post_init__(self) -> None:
        if self.lambda_matrix.shape != (
            self.orb_num_up,
            self.orb_num_dn + (self.num_electron_up - self.num_electron_dn),
        ):
            logger.error(
                f"dim. of lambda_matrix = {self.lambda_matrix.shape} is imcompatible with the expected one "
                + f"= ({self.orb_num_up}, {self.orb_num_dn + (self.num_electron_up - self.num_electron_dn)}).",
            )
            raise ValueError

        logger.debug(f"compute_orb={self.compute_orb_api}")

    @property
    def orb_num_up(self) -> int:
        if self.compute_orb_api == compute_AOs_api:
            return self.orb_data_up_spin.num_ao
        elif self.compute_orb_api == compute_MOs_api:
            return self.orb_data_up_spin.num_mo
        else:
            raise NotImplementedError

    @property
    def orb_num_dn(self) -> int:
        if self.compute_orb_api == compute_AOs_api:
            return self.orb_data_dn_spin.num_ao
        elif self.compute_orb_api == compute_MOs_api:
            return self.orb_data_dn_spin.num_mo
        else:
            raise NotImplementedError

    @property
    def compute_orb_grad_api(self) -> Callable[..., npt.NDArray[np.float64]]:
        if self.compute_orb_api == compute_AOs_api:
            return compute_AOs_grad_api
        elif self.compute_orb_api == compute_MOs_api:
            return compute_MOs_grad_api
        else:
            raise NotImplementedError

    @property
    def compute_orb_laplacian_api(self) -> Callable[..., npt.NDArray[np.float64]]:
        if self.compute_orb_api == compute_AOs_api:
            return compute_AOs_laplacian_api
        elif self.compute_orb_api == compute_MOs_api:
            return compute_MOs_laplacian_api
        else:
            raise NotImplementedError


def compute_det_geminal_all_elements_api(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    debug_flag: bool = False,
) -> np.float64 | np.complex128:
    if debug_flag:
        return np.linalg.det(
            compute_geminal_all_elements_api(
                geminal_data=geminal_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                debug_flag=True,
            )
        )
    else:
        return jnp.linalg.det(
            compute_geminal_all_elements_api(
                geminal_data=geminal_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                debug_flag=False,
            )
        )


def compute_geminal_all_elements_api(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    debug_flag: bool = False,
) -> npt.NDArray[np.float64 | np.complex128]:
    """
    The method is for computing geminal matrix elements with the given atomic/molecular orbitals at (r_up_carts, r_dn_carts).

    Args:
        geminal_data (Geminal_data): an instance of Geminal_data class
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        Arrays containing values of the given geminal functions f(i,j) where r_up_carts[i] and r_dn_carts[j]. (dim: N_e^{up}, N_e^{up})
    """

    if (
        len(r_up_carts) != geminal_data.num_electron_up
        or len(r_dn_carts) != geminal_data.num_electron_dn
    ):
        logger.info(
            f"Number of up and dn electrons (N_up, N_dn) = ({len(r_up_carts)}, {len(r_dn_carts)}) are not consistent "
            + f"with the expected values. (N_up, N_dn) = {geminal_data.num_electron_up}, {geminal_data.num_electron_dn})"
        )
        raise ValueError

    if len(r_up_carts) != len(r_dn_carts):
        if len(r_up_carts) - len(r_dn_carts) > 0:
            logger.info(
                f"Number of up and dn electrons are different. (N_el - N_dn = {len(r_up_carts) - len(r_dn_carts)})"
            )
        else:
            logger.error(
                f"Number of up electron is smaller than dn electrons. (N_el - N_dn = {len(r_up_carts) - len(r_dn_carts)})"
            )
            raise ValueError
    else:
        pass
        # logger.debug("There is no unpaired electrons.")

    # jprint(f"geminal:debug_flag={debug_flag}, type={type(debug_flag)}")

    if debug_flag:
        geminal = compute_geminal_all_elements_debug(
            geminal_data, r_up_carts, r_dn_carts
        )
    else:
        geminal = compute_geminal_all_elements_jax(geminal_data, r_up_carts, r_dn_carts)

    if geminal.shape != (len(r_up_carts), len(r_up_carts)):
        logger.error(
            f"geminal.shape = {geminal.shape} is inconsistent with the expected one = {(len(r_up_carts), len(r_up_carts))}"
        )
        raise ValueError

    return geminal


def compute_geminal_all_elements_debug(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64 | np.complex128]:
    lambda_matrix_paired, lambda_matrix_unpaired = np.hsplit(
        geminal_data.lambda_matrix, [geminal_data.orb_num_dn]
    )

    orb_matrix_up = geminal_data.compute_orb_api(
        geminal_data.orb_data_up_spin, r_up_carts, debug_flag=False
    )
    orb_matrix_dn = geminal_data.compute_orb_api(
        geminal_data.orb_data_dn_spin, r_dn_carts, debug_flag=False
    )

    # compute geminal values
    geminal_paired = np.dot(
        orb_matrix_up.T, np.dot(lambda_matrix_paired, orb_matrix_dn)
    )
    geminal_unpaired = np.dot(orb_matrix_up.T, lambda_matrix_unpaired)
    geminal = np.hstack([geminal_paired, geminal_unpaired])

    return geminal


# it cannot be jitted!? because _api methods
# in which crude if statements are included.
# but why? other _api can be jitted...
# -> probably it's related to pytree... now it works
@jit
def compute_geminal_all_elements_jax(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64 | np.complex128]:
    lambda_matrix_paired, lambda_matrix_unpaired = jnp.hsplit(
        geminal_data.lambda_matrix, [geminal_data.orb_num_dn]
    )

    orb_matrix_up = geminal_data.compute_orb_api(
        geminal_data.orb_data_up_spin, r_up_carts, debug_flag=False
    )
    orb_matrix_dn = geminal_data.compute_orb_api(
        geminal_data.orb_data_dn_spin, r_dn_carts, debug_flag=False
    )

    # compute geminal values
    geminal_paired = jnp.dot(
        orb_matrix_up.T, jnp.dot(lambda_matrix_paired, orb_matrix_dn)
    )
    geminal_unpaired = jnp.dot(orb_matrix_up.T, lambda_matrix_unpaired)
    geminal = jnp.hstack([geminal_paired, geminal_unpaired])

    return geminal


def compute_grads_and_laplacian_ln_Det_api(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    debug_flag: bool = False,
) -> tuple[
    npt.NDArray[np.float64 | np.complex128],
    npt.NDArray[np.float64 | np.complex128],
    float | complex,
]:
    """
    The method is for computing the sum of laplacians of ln WF at (r_up_carts, r_dn_carts).

    Args:
        geminal_data (Geminal_data): an instance of Geminal_data class
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        debug_flag: if True, numerical derivatives are computed for debuging purpose
    Returns:
        the gradients(x,y,z) of ln Det and the sum of laplacians of ln Det at (r_up_carts, r_dn_carts).
    """

    if (
        len(r_up_carts) != geminal_data.num_electron_up
        or len(r_dn_carts) != geminal_data.num_electron_dn
    ):
        logger.info(
            f"Number of up and dn electrons (N_up, N_dn) = ({len(r_up_carts)}, {len(r_dn_carts)}) are not consistent "
            + f"with the expected values. (N_up, N_dn) = {geminal_data.num_electron_up}, {geminal_data.num_electron_dn})"
        )
        raise ValueError

    if len(r_up_carts) != len(r_dn_carts):
        if len(r_up_carts) - len(r_dn_carts) > 0:
            logger.info(
                f"Number of up and dn electrons are different. (N_el - N_dn = {len(r_up_carts) - len(r_dn_carts)})"
            )
        else:
            logger.error(
                f"Number of up electron is smaller than dn electrons. (N_el - N_dn = {len(r_up_carts) - len(r_dn_carts)})"
            )
            raise ValueError
    else:
        logger.debug("There is no unpaired electrons.")

    if debug_flag:
        grad_ln_D_up, grad_ln_D_dn, sum_laplacian_ln_D = (
            compute_grads_and_laplacian_ln_Det_debug(
                geminal_data, r_up_carts, r_dn_carts
            )
        )
    else:
        grad_ln_D_up, grad_ln_D_dn, sum_laplacian_ln_D = (
            compute_grads_and_laplacian_ln_Det_jax(geminal_data, r_up_carts, r_dn_carts)
        )

    if grad_ln_D_up.shape != (geminal_data.num_electron_up, 3):
        logger.error(
            f"grad_ln_D_up.shape = {grad_ln_D_up.shape} is inconsistent with the expected one = {(geminal_data.num_electron_up, 3)}"
        )
        raise ValueError

    if grad_ln_D_dn.shape != (geminal_data.num_electron_dn, 3):
        logger.error(
            f"grad_ln_D_up.shape = {grad_ln_D_up.shape} is inconsistent with the expected one = {(geminal_data.num_electron_dn, 3)}"
        )
        raise ValueError

    return grad_ln_D_up, grad_ln_D_dn, sum_laplacian_ln_D


def compute_grads_and_laplacian_ln_Det_debug(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64 | np.complex128],
    npt.NDArray[np.float64 | np.complex128],
    float | complex,
]:
    diff_h = 1.0e-5

    det_geminal = compute_det_geminal_all_elements_api(
        geminal_data=geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    sum_laplacian_ln_D = 0.0

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

        det_geminal_p_x_up = compute_det_geminal_all_elements_api(
            geminal_data=geminal_data,
            r_up_carts=diff_p_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        det_geminal_p_y_up = compute_det_geminal_all_elements_api(
            geminal_data=geminal_data,
            r_up_carts=diff_p_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        det_geminal_p_z_up = compute_det_geminal_all_elements_api(
            geminal_data=geminal_data,
            r_up_carts=diff_p_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        diff_m_x_r_up_carts = r_up_carts.copy()
        diff_m_y_r_up_carts = r_up_carts.copy()
        diff_m_z_r_up_carts = r_up_carts.copy()
        diff_m_x_r_up_carts[r_i][0] -= diff_h
        diff_m_y_r_up_carts[r_i][1] -= diff_h
        diff_m_z_r_up_carts[r_i][2] -= diff_h

        det_geminal_m_x_up = compute_det_geminal_all_elements_api(
            geminal_data=geminal_data,
            r_up_carts=diff_m_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        det_geminal_m_y_up = compute_det_geminal_all_elements_api(
            geminal_data=geminal_data,
            r_up_carts=diff_m_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        det_geminal_m_z_up = compute_det_geminal_all_elements_api(
            geminal_data=geminal_data,
            r_up_carts=diff_m_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        grad_x_up.append(
            (np.log(np.abs(det_geminal_p_x_up)) - np.log(np.abs(det_geminal_m_x_up)))
            / (2.0 * diff_h)
        )
        grad_y_up.append(
            (np.log(np.abs(det_geminal_p_y_up)) - np.log(np.abs(det_geminal_m_y_up)))
            / (2.0 * diff_h)
        )
        grad_z_up.append(
            (np.log(np.abs(det_geminal_p_z_up)) - np.log(np.abs(det_geminal_m_z_up)))
            / (2.0 * diff_h)
        )

        gradgrad_x_up = (
            np.log(np.abs(det_geminal_p_x_up))
            + np.log(np.abs(det_geminal_m_x_up))
            - 2.0 * np.log(np.abs(det_geminal))
        ) / (diff_h**2)

        gradgrad_y_up = (
            np.log(np.abs(det_geminal_p_y_up))
            + np.log(np.abs(det_geminal_m_y_up))
            - 2.0 * np.log(np.abs(det_geminal))
        ) / (diff_h**2)

        gradgrad_z_up = (
            np.log(np.abs(det_geminal_p_z_up))
            + np.log(np.abs(det_geminal_m_z_up))
            - 2.0 * np.log(np.abs(det_geminal))
        ) / (diff_h**2)

        sum_laplacian_ln_D += gradgrad_x_up + gradgrad_y_up + gradgrad_z_up

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

        det_geminal_p_x_dn = compute_det_geminal_all_elements_api(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn_carts,
        )
        det_geminal_p_y_dn = compute_det_geminal_all_elements_api(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn_carts,
        )
        det_geminal_p_z_dn = compute_det_geminal_all_elements_api(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn_carts,
        )

        diff_m_x_r_dn_carts = r_dn_carts.copy()
        diff_m_y_r_dn_carts = r_dn_carts.copy()
        diff_m_z_r_dn_carts = r_dn_carts.copy()
        diff_m_x_r_dn_carts[r_i][0] -= diff_h
        diff_m_y_r_dn_carts[r_i][1] -= diff_h
        diff_m_z_r_dn_carts[r_i][2] -= diff_h

        det_geminal_m_x_dn = compute_det_geminal_all_elements_api(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn_carts,
        )
        det_geminal_m_y_dn = compute_det_geminal_all_elements_api(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn_carts,
        )
        det_geminal_m_z_dn = compute_det_geminal_all_elements_api(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn_carts,
        )

        grad_x_dn.append(
            (np.log(np.abs(det_geminal_p_x_dn)) - np.log(np.abs(det_geminal_m_x_dn)))
            / (2.0 * diff_h)
        )
        grad_y_dn.append(
            (np.log(np.abs(det_geminal_p_y_dn)) - np.log(np.abs(det_geminal_m_y_dn)))
            / (2.0 * diff_h)
        )
        grad_z_dn.append(
            (np.log(np.abs(det_geminal_p_z_dn)) - np.log(np.abs(det_geminal_m_z_dn)))
            / (2.0 * diff_h)
        )

        gradgrad_x_dn = (
            np.log(np.abs(det_geminal_p_x_dn))
            + np.log(np.abs(det_geminal_m_x_dn))
            - 2.0 * np.log(np.abs(det_geminal))
        ) / (diff_h**2)

        gradgrad_y_dn = (
            np.log(np.abs(det_geminal_p_y_dn))
            + np.log(np.abs(det_geminal_m_y_dn))
            - 2.0 * np.log(np.abs(det_geminal))
        ) / (diff_h**2)

        gradgrad_z_dn = (
            np.log(np.abs(det_geminal_p_z_dn))
            + np.log(np.abs(det_geminal_m_z_dn))
            - 2.0 * np.log(np.abs(det_geminal))
        ) / (diff_h**2)

        sum_laplacian_ln_D += gradgrad_x_dn + gradgrad_y_dn + gradgrad_z_dn

    grad_ln_D_up = np.array([grad_x_up, grad_y_up, grad_z_up]).T
    grad_ln_D_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn]).T

    return grad_ln_D_up, grad_ln_D_dn, sum_laplacian_ln_D


@jit
def compute_grads_and_laplacian_ln_Det_jax(
    geminal_data: Geminal_data,
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
    ao_matrix_up = geminal_data.compute_orb_api(
        geminal_data.orb_data_up_spin, r_up_carts, debug_flag=False
    )
    ao_matrix_dn = geminal_data.compute_orb_api(
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
    num_r_up_cart_samples = 2
    num_r_dn_cart_samples = 2
    num_R_cart_samples = 6
    num_ao = 6
    num_mo_up = num_mo_dn = num_r_up_cart_samples  # Slater Determinant
    num_ao_prim = 6
    orbital_indices = [0, 1, 2, 3, 4, 5]
    exponents = [1.2, 0.5, 0.1, 0.05, 0.05, 0.05]
    coefficients = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    angular_momentums = [0, 0, 0, 1, 1, 1]
    magnetic_quantum_numbers = [0, 0, 0, 0, +1, -1]

    # generate matrices for the test
    mo_coefficients_up = mo_coefficients_dn = np.random.rand(num_mo_up, num_ao)
    mo_lambda_matrix_paired = np.eye(num_mo_up, num_mo_dn, k=0)
    mo_lambda_matrix_unpaired = np.eye(num_mo_up, num_mo_up - num_mo_dn, k=-num_mo_dn)
    mo_lambda_matrix = np.hstack([mo_lambda_matrix_paired, mo_lambda_matrix_unpaired])

    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_up_cart_samples, 3
    ) + r_cart_min
    """
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_dn_cart_samples, 3
    ) + r_cart_min
    """
    r_dn_carts = r_up_carts
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(
        num_R_cart_samples, 3
    ) + R_cart_min

    aos_up_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_dn_data = AOs_data_debug(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_up_data = MOs_data(
        num_mo=num_mo_up, mo_coefficients=mo_coefficients_up, aos_data=aos_up_data
    )

    mos_dn_data = MOs_data(
        num_mo=num_mo_dn, mo_coefficients=mo_coefficients_dn, aos_data=aos_dn_data
    )

    geminal_mo_data = Geminal_data(
        num_electron_up=num_r_up_cart_samples,
        num_electron_dn=num_r_dn_cart_samples,
        orb_data_up_spin=mos_up_data,
        orb_data_dn_spin=mos_dn_data,
        compute_orb_api=compute_MOs_api,
        lambda_matrix=mo_lambda_matrix,
    )

    geminal_mo_matrix = compute_geminal_all_elements_api(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # generate matrices for the test
    ao_lambda_matrix_paired = np.dot(
        mo_coefficients_up.T, np.dot(mo_lambda_matrix_paired, mo_coefficients_dn)
    )
    ao_lambda_matrix_unpaired = np.dot(mo_coefficients_up.T, mo_lambda_matrix_unpaired)
    ao_lambda_matrix = np.hstack([ao_lambda_matrix_paired, ao_lambda_matrix_unpaired])

    # check if generated ao_lambda_matrix is symmetric:
    assert np.allclose(ao_lambda_matrix, ao_lambda_matrix.T)

    geminal_ao_data = Geminal_data(
        num_electron_up=num_r_up_cart_samples,
        num_electron_dn=num_r_dn_cart_samples,
        orb_data_up_spin=aos_up_data,
        orb_data_dn_spin=aos_dn_data,
        compute_orb_api=compute_MOs_api,
        lambda_matrix=ao_lambda_matrix,
    )

    geminal_ao_matrix = compute_geminal_all_elements_api(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # check if geminals with AO and MO representations are consistent
    np.testing.assert_array_almost_equal(
        geminal_ao_matrix, geminal_mo_matrix, decimal=15
    )

    grad_ln_D_up, grad_ln_D_dn, sum_laplacian_ln_D = (
        compute_grads_and_laplacian_ln_Det_api(
            geminal_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
    )

    print(grad_ln_D_up)
    print(grad_ln_D_dn)
    print(sum_laplacian_ln_D)
