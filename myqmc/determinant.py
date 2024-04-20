"""Geminal module"""

# python modules
# from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
import numpy.typing as npt

# jax modules
import jax.numpy as jnp
from flax import struct
from jax import jacrev

# set logger
from logging import getLogger, StreamHandler, Formatter

# myqmc module
from atomic_orbital import AOs_data, compute_AOs_api
from molecular_orbital import MOs_data, compute_MOs

logger = getLogger("myqmc").getChild(__name__)


# @dataclass
@struct.dataclass
class Geminal_data:
    """
    The class contains data for computing geminal function.

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
    orb_data_up_spin: AOs_data | MOs_data = struct.field(pytree_node=True)
    orb_data_dn_spin: AOs_data | MOs_data = struct.field(pytree_node=True)
    compute_orb: Callable[..., npt.NDArray[np.float64]] = struct.field(
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

    @property
    def orb_num_up(self) -> int:
        if self.compute_orb == compute_AOs_api:
            return self.orb_data_up_spin.num_ao
        elif self.compute_orb == compute_MOs:
            return self.orb_data_up_spin.num_mo
        else:
            raise NotImplementedError

    @property
    def orb_num_dn(self) -> int:
        if self.compute_orb == compute_AOs_api:
            return self.orb_data_dn_spin.num_ao
        elif self.compute_orb == compute_MOs:
            return self.orb_data_dn_spin.num_mo
        else:
            raise NotImplementedError


def compute_det_ao_geminal_all_elements(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> np.float64 | np.complex128:

    return np.linalg.det(
        compute_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
    )


def compute_geminal_all_elements(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64 | np.complex128]:
    """
    The method is for computing geminal matrix elements with the given atomic orbitals at (r_up_carts, r_dn_carts).

    Args:
        geminal_data (Geminal_data): an instance of Geminal_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        Arrays containing values of the geminal function with r_up_carts and r_dn_carts. (dim: N_e^{up}, N_e^{up})
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
        logger.info("There is no unpaired electrons.")

    lambda_matrix_paired, lambda_matrix_unpaired = np.hsplit(
        geminal_data.lambda_matrix, [geminal_data.orb_num_dn]
    )

    ao_matrix_up = geminal_data.compute_orb(geminal_data.orb_data_up_spin, r_up_carts)
    ao_matrix_dn = geminal_data.compute_orb(geminal_data.orb_data_dn_spin, r_dn_carts)

    if ao_matrix_up.shape != (geminal_data.orb_num_up, len(r_up_carts)):
        logger.error(
            f"answer.shape = {ao_matrix_up.shape} is inconsistent with the expected one = {(len(geminal_data.orb_num_up), len(r_up_carts))}"
        )
        raise ValueError

    if ao_matrix_dn.shape != (geminal_data.orb_num_dn, len(r_dn_carts)):
        logger.error(
            f"answer.shape = {ao_matrix_dn.shape} is inconsistent with the expected one = {(len(geminal_data.orb_num_dn), len(r_dn_carts))}"
        )
        raise ValueError

    # compute geminal values
    geminal_paired = np.dot(ao_matrix_up.T, np.dot(lambda_matrix_paired, ao_matrix_dn))
    assert np.allclose(geminal_paired, geminal_paired.T)
    geminal_unpaired = np.dot(ao_matrix_up.T, lambda_matrix_unpaired)
    geminal = np.hstack([geminal_paired, geminal_unpaired])

    if geminal.shape != (len(r_up_carts), len(r_up_carts)):
        logger.error(
            f"geminal.shape = {geminal.shape} is inconsistent with the expected one = {(len(r_up_carts), len(r_up_carts))}"
        )
        raise ValueError

    return geminal


def compute_gradients_and_laplacians_geminal(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64 | np.complex128]:
    """
    The method is for computing geminal matrix elements with the given atomic orbitals at (r_up_carts, r_dn_carts).

    Args:
        geminal_data (Geminal_data): an instance of Geminal_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        Arrays containing values of the geminal function with r_up_carts and r_dn_carts. (dim: N_e^{up}, N_e^{up})
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
        logger.info("There is no unpaired electrons.")

    lambda_matrix_paired, lambda_matrix_unpaired = np.hsplit(
        geminal_data.lambda_matrix, [geminal_data.orb_num_dn]
    )

    # AOs
    ao_matrix_up = geminal_data.compute_orb(geminal_data.orb_data_up_spin, r_up_carts)
    ao_matrix_dn = geminal_data.compute_orb(geminal_data.orb_data_dn_spin, r_dn_carts)

    if ao_matrix_up.shape != (geminal_data.orb_num_up, len(r_up_carts)):
        logger.error(
            f"ao_matrix_up.shape = {ao_matrix_up.shape} is inconsistent with the expected one = {(geminal_data.orb_num_up, len(r_up_carts))}"
        )
        raise ValueError

    if ao_matrix_dn.shape != (geminal_data.orb_num_dn, len(r_dn_carts)):
        logger.error(
            f"ao_matrix_dn.shape = {ao_matrix_dn.shape} is inconsistent with the expected one = {(geminal_data.orb_num_dn, len(r_dn_carts))}"
        )
        raise ValueError

    # Gradientss of AOs (up-spin)
    ao_matrix_up_jacrev = jacrev(geminal_data.compute_orb, argnums=1)(
        geminal_data.orb_data_up_spin, r_up_carts, jax_flag=True
    )

    ao_matrix_grad_up_x_ = ao_matrix_up_jacrev[:, :, :, 0]
    ao_matrix_grad_up_y_ = ao_matrix_up_jacrev[:, :, :, 1]
    ao_matrix_grad_up_z_ = ao_matrix_up_jacrev[:, :, :, 2]
    ao_matrix_up_grad_x = jnp.sum(ao_matrix_grad_up_x_, axis=2)
    ao_matrix_up_grad_y = jnp.sum(ao_matrix_grad_up_y_, axis=2)
    ao_matrix_up_grad_z = jnp.sum(ao_matrix_grad_up_z_, axis=2)

    logger.debug(ao_matrix_up_grad_x)
    logger.debug(ao_matrix_up_grad_y)
    logger.debug(ao_matrix_up_grad_z)

    if ao_matrix_up_grad_x.shape != (geminal_data.orb_num_up, len(r_up_carts)):
        logger.error(
            f"aao_matrix_up_grad_x.shape = {ao_matrix_up_grad_x.shape} is inconsistent with the expected one = {(geminal_data.orb_num_up, len(r_up_carts))}"
        )
        raise ValueError

    if ao_matrix_up_grad_y.shape != (geminal_data.orb_num_up, len(r_up_carts)):
        logger.error(
            f"ao_matrix_up_grad_y.shape = {ao_matrix_up_grad_y.shape} is inconsistent with the expected one = {(geminal_data.orb_num_up, len(r_up_carts))}"
        )
        raise ValueError

    if ao_matrix_up_grad_z.shape != (geminal_data.orb_num_up, len(r_up_carts)):
        logger.error(
            f"ao_matrix_up_grad_z.shape = {ao_matrix_up_grad_y.shape} is inconsistent with the expected one = {(geminal_data.orb_num_up, len(r_up_carts))}"
        )
        raise ValueError

    # Gradientss of AOs (dn-spin)
    ao_matrix_dn_jacrev = jacrev(geminal_data.compute_orb, argnums=1)(
        geminal_data.orb_data_dn_spin, r_dn_carts, jax_flag=True
    )

    ao_matrix_grad_dn_x_ = ao_matrix_dn_jacrev[:, :, :, 0]
    ao_matrix_grad_dn_y_ = ao_matrix_dn_jacrev[:, :, :, 1]
    ao_matrix_grad_dn_z_ = ao_matrix_dn_jacrev[:, :, :, 2]
    ao_matrix_dn_grad_x = jnp.sum(ao_matrix_grad_dn_x_, axis=2)
    ao_matrix_dn_grad_y = jnp.sum(ao_matrix_grad_dn_y_, axis=2)
    ao_matrix_dn_grad_z = jnp.sum(ao_matrix_grad_dn_z_, axis=2)

    logger.debug(ao_matrix_dn_grad_x)
    logger.debug(ao_matrix_dn_grad_y)
    logger.debug(ao_matrix_dn_grad_z)

    if ao_matrix_dn_grad_x.shape != (geminal_data.orb_num_dn, len(r_dn_carts)):
        logger.error(
            f"ao_matrix_dn_grad_x.shape = {ao_matrix_dn_grad_x.shape} is inconsistent with the expected one = {(geminal_data.orb_num_dn, len(r_dn_carts))}"
        )
        raise ValueError

    if ao_matrix_dn_grad_y.shape != (geminal_data.orb_num_dn, len(r_dn_carts)):
        logger.error(
            f"ao_matrix_dn_grad_y.shape = {ao_matrix_dn_grad_y.shape} is inconsistent with the expected one = {(geminal_data.orb_num_dn, len(r_dn_carts))}"
        )
        raise ValueError

    if ao_matrix_dn_grad_z.shape != (geminal_data.orb_num_dn, len(r_dn_carts)):
        logger.error(
            f"ao_matrix_dn_grad_z.shape = {ao_matrix_dn_grad_y.shape} is inconsistent with the expected one = {(geminal_data.orb_num_dn, len(r_dn_carts))}"
        )
        raise ValueError

    # Laplacians of AOs (up-spin)
    ao_matrix_up_hessian = jacrev(geminal_data.compute_orb, argnums=1)(
        geminal_data.orb_data_up_spin, r_up_carts, jax_flag=True
    )

    ao_matrix_hessian_up_x2_ = ao_matrix_up_hessian[:, :, :, 0]
    ao_matrix_hessian_up_y2_ = ao_matrix_up_hessian[:, :, :, 1]
    ao_matrix_hessian_up_z2_ = ao_matrix_up_hessian[:, :, :, 2]
    ao_matrix_hessian_up_x2 = jnp.sum(ao_matrix_hessian_up_x2_, axis=2)
    ao_matrix_hessian_up_y2 = jnp.sum(ao_matrix_hessian_up_y2_, axis=2)
    ao_matrix_hessian_up_z2 = jnp.sum(ao_matrix_hessian_up_z2_, axis=2)

    ao_matrix_laplacian_up = (
        ao_matrix_hessian_up_x2 + ao_matrix_hessian_up_y2 + ao_matrix_hessian_up_z2
    )

    if ao_matrix_laplacian_up.shape != (geminal_data.orb_num_up, len(r_up_carts)):
        logger.error(
            f"ao_matrix_hessian_up.shape = {ao_matrix_laplacian_up.shape} is inconsistent with the expected one = {(geminal_data.orb_num_up, len(r_up_carts))}"
        )
        raise ValueError

    # Laplacians of AOs (dn-spin)
    ao_matrix_dn_hessian = jacrev(geminal_data.compute_orb, argnums=1)(
        geminal_data.orb_data_dn_spin, r_dn_carts, jax_flag=True
    )

    ao_matrix_hessian_dn_x2_ = ao_matrix_dn_hessian[:, :, :, 0]
    ao_matrix_hessian_dn_y2_ = ao_matrix_dn_hessian[:, :, :, 1]
    ao_matrix_hessian_dn_z2_ = ao_matrix_dn_hessian[:, :, :, 2]
    ao_matrix_hessian_dn_x2 = jnp.sum(ao_matrix_hessian_dn_x2_, axis=2)
    ao_matrix_hessian_dn_y2 = jnp.sum(ao_matrix_hessian_dn_y2_, axis=2)
    ao_matrix_hessian_dn_z2 = jnp.sum(ao_matrix_hessian_dn_z2_, axis=2)

    ao_matrix_laplacian_dn = (
        ao_matrix_hessian_dn_x2 + ao_matrix_hessian_dn_y2 + ao_matrix_hessian_dn_z2
    )

    logger.debug(ao_matrix_laplacian_up)
    logger.debug(ao_matrix_laplacian_dn)

    if ao_matrix_laplacian_dn.shape != (geminal_data.orb_num_dn, len(r_dn_carts)):
        logger.error(
            f"ao_matrix_hessian_dn.shape = {ao_matrix_laplacian_dn.shape} is inconsistent with the expected one = {(geminal_data.orb_num_dn, len(r_dn_carts))}"
        )
        raise ValueError

    # compute Laplacians of Geminal
    geminal_paired = np.dot(ao_matrix_up.T, np.dot(lambda_matrix_paired, ao_matrix_dn))
    geminal_unpaired = np.dot(ao_matrix_up.T, lambda_matrix_unpaired)
    geminal = np.hstack([geminal_paired, geminal_unpaired])

    logger.debug(geminal)

    # up electron
    geminal_grad_up_x_paired = np.dot(
        ao_matrix_up_grad_x.T, np.dot(lambda_matrix_paired, ao_matrix_dn)
    )
    geminal_grad_up_x_unpaired = np.dot(ao_matrix_up_grad_x.T, lambda_matrix_unpaired)
    geminal_grad_up_x = np.hstack(
        [geminal_grad_up_x_paired, geminal_grad_up_x_unpaired]
    )

    geminal_grad_up_y_paired = np.dot(
        ao_matrix_up_grad_y.T, np.dot(lambda_matrix_paired, ao_matrix_dn)
    )
    geminal_grad_up_y_unpaired = np.dot(ao_matrix_up_grad_y.T, lambda_matrix_unpaired)
    geminal_grad_up_y = np.hstack(
        [geminal_grad_up_y_paired, geminal_grad_up_y_unpaired]
    )

    geminal_grad_up_z_paired = np.dot(
        ao_matrix_up_grad_y.T, np.dot(lambda_matrix_paired, ao_matrix_dn)
    )
    geminal_grad_up_z_unpaired = np.dot(ao_matrix_up_grad_z.T, lambda_matrix_unpaired)
    geminal_grad_up_z = np.hstack(
        [geminal_grad_up_z_paired, geminal_grad_up_z_unpaired]
    )

    geminal_laplacian_up_paired = np.dot(
        ao_matrix_laplacian_up.T, np.dot(lambda_matrix_paired, ao_matrix_dn)
    )
    geminal_laplacian_up_unpaired = np.dot(
        ao_matrix_laplacian_up.T, lambda_matrix_unpaired
    )
    geminal_laplacian_up = np.hstack(
        [geminal_laplacian_up_paired, geminal_laplacian_up_unpaired]
    )

    # dn electron
    geminal_grad_dn_x_paired = np.dot(
        ao_matrix_up.T, np.dot(lambda_matrix_paired, ao_matrix_dn_grad_x)
    )
    geminal_grad_dn_x_unpaired = np.zeros(
        [
            geminal_data.num_electron_up,
            geminal_data.num_electron_up - geminal_data.num_electron_dn,
        ]
    )
    geminal_grad_dn_x = np.hstack(
        [geminal_grad_dn_x_paired, geminal_grad_dn_x_unpaired]
    )

    geminal_grad_dn_y_paired = np.dot(
        ao_matrix_up.T, np.dot(lambda_matrix_paired, ao_matrix_dn_grad_y)
    )
    geminal_grad_dn_y_unpaired = np.zeros(
        [
            geminal_data.num_electron_up,
            geminal_data.num_electron_up - geminal_data.num_electron_dn,
        ]
    )
    geminal_grad_dn_y = np.hstack(
        [geminal_grad_dn_y_paired, geminal_grad_dn_y_unpaired]
    )

    geminal_grad_dn_z_paired = np.dot(
        ao_matrix_up.T, np.dot(lambda_matrix_paired, ao_matrix_dn_grad_z)
    )
    geminal_grad_dn_z_unpaired = np.zeros(
        [
            geminal_data.num_electron_up,
            geminal_data.num_electron_up - geminal_data.num_electron_dn,
        ]
    )
    geminal_grad_dn_z = np.hstack(
        [geminal_grad_dn_z_paired, geminal_grad_dn_z_unpaired]
    )

    geminal_laplacian_dn_paired = np.dot(
        ao_matrix_up.T, np.dot(lambda_matrix_paired, ao_matrix_laplacian_dn)
    )
    geminal_laplacian_dn_unpaired = np.zeros(
        [
            geminal_data.num_electron_up,
            geminal_data.num_electron_up - geminal_data.num_electron_dn,
        ]
    )
    geminal_laplacian_dn = np.hstack(
        [geminal_laplacian_dn_paired, geminal_laplacian_dn_unpaired]
    )

    logger.info(f"The condition number of geminal matrix is {jnp.linalg.cond(geminal)}")
    geminal_inverse = jnp.linalg.inv(geminal)

    logger.debug(geminal_inverse)

    vec_F_D_up_x = jnp.diag(np.dot(geminal_grad_up_x, geminal_inverse))
    vec_F_D_up_y = jnp.diag(np.dot(geminal_grad_up_y, geminal_inverse))
    vec_F_D_up_z = jnp.diag(np.dot(geminal_grad_up_z, geminal_inverse))
    vec_F_D_dn_x = jnp.diag(np.dot(geminal_inverse, geminal_grad_dn_x))
    vec_F_D_dn_y = jnp.diag(np.dot(geminal_inverse, geminal_grad_dn_y))
    vec_F_D_dn_z = jnp.diag(np.dot(geminal_inverse, geminal_grad_dn_z))

    logger.info(vec_F_D_up_x)
    logger.info(vec_F_D_up_y)
    logger.info(vec_F_D_up_z)
    logger.info(vec_F_D_dn_x)
    logger.info(vec_F_D_dn_y)
    logger.info(vec_F_D_dn_z)

    T_D = (
        -((jnp.trace(np.dot(geminal_grad_up_x, geminal_inverse))) ** 2)
        - (jnp.trace(np.dot(geminal_grad_up_y, geminal_inverse))) ** 2
        - (jnp.trace(np.dot(geminal_grad_up_z, geminal_inverse))) ** 2
        - (jnp.trace(np.dot(geminal_inverse, geminal_grad_dn_x))) ** 2
        - (jnp.trace(np.dot(geminal_inverse, geminal_grad_dn_y))) ** 2
        - (jnp.trace(np.dot(geminal_inverse, geminal_grad_dn_z))) ** 2
        + (jnp.trace(np.dot(geminal_laplacian_up, geminal_inverse))) ** 2
        + (jnp.trace(np.dot(geminal_inverse, geminal_laplacian_dn))) ** 2
    )

    logger.info(T_D)

    return (
        (vec_F_D_up_x, vec_F_D_up_y, vec_F_D_up_z),
        (vec_F_D_dn_x, vec_F_D_dn_y, vec_F_D_dn_z),
        T_D,
    )


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

    aos_up_data = AOs_data(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_dn_data = AOs_data(
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
        compute_orb=compute_MOs,
        lambda_matrix=mo_lambda_matrix,
    )

    geminal_mo_matrix = compute_geminal_all_elements(
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
        compute_orb=compute_AOs_api,
        lambda_matrix=ao_lambda_matrix,
    )

    geminal_ao_matrix = compute_geminal_all_elements(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # check if geminals with AO and MO representations are consistent
    np.testing.assert_array_almost_equal(
        geminal_ao_matrix, geminal_mo_matrix, decimal=15
    )
