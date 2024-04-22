"""Atomic Orbital module"""

# import sys

# python modules
from dataclasses import dataclass, field
from functools import partial
import itertools
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

# scipy
import scipy  # type: ignore

# jax modules
from jax import vmap, jit
import jax.scipy as jscipy
import jax.numpy as jnp
from flax import struct
from jax.lax import cond

# set logger
from logging import getLogger, StreamHandler, Formatter

logger = getLogger("myqmc").getChild(__name__)


# @dataclass
@struct.dataclass
class AOs_data:
    """
    The class contains data for computing atomic orbitals simltaneously

    Args:
        num_ao : the number of atomic orbitals.
        num_ao_prim : the number of primitive atomic orbitals.
        atomic_center_carts (npt.NDArray[np.float64]): Centers of the nuclei associated to the AOs (dim: num_AOs, 3).
        orbital_indices (list[int]): index for what exponents and coefficients are associated to each atomic orbital. dim: num_ao_prim
        exponents (list[float]): List of exponents of the AOs. dim: num_ao_prim.
        coefficients (list[float | complex]): List of coefficients of the AOs. dim: num_ao_prim
        angular_momentums (list[int]): Angular momentum of the AOs, i.e., l. dim: num_ao
        magnetic_quantum_numbers (list[int]): Magnetic quantum number of the AOs, i.e m = -l .... +l. dim: num_ao
    """

    num_ao: int = struct.field(pytree_node=False)
    num_ao_prim: int = struct.field(pytree_node=False)
    atomic_center_carts: npt.NDArray[np.float64] = struct.field(pytree_node=True)
    orbital_indices: list[int] = struct.field(pytree_node=False)
    exponents: list[float] = struct.field(pytree_node=True)
    coefficients: list[float | complex] = struct.field(pytree_node=True)
    angular_momentums: list[int] = struct.field(pytree_node=False)
    magnetic_quantum_numbers: list[int] = struct.field(pytree_node=False)

    def __post_init__(self) -> None:
        if self.atomic_center_carts.shape != (self.num_ao, 3):
            logger.error("dim. of atomic_center_cart is wrong")
            raise ValueError
        if len(np.unique(self.orbital_indices)) != self.num_ao:
            logger.error(
                f"num_ao={self.num_ao} and/or num_ao_prim={self.num_ao_prim} is wrong"
            )
        if len(self.exponents) != self.num_ao_prim:
            logger.error("dim. of self.exponents is wrong")
            raise ValueError
        if len(self.coefficients) != self.num_ao_prim:
            logger.error("dim. of self.coefficients is wrong")
            raise ValueError
        if len(self.angular_momentums) != self.num_ao:
            logger.error("dim. of self.angular_momentums is wrong")
            raise ValueError
        if len(self.magnetic_quantum_numbers) != self.num_ao:
            logger.error("dim. of self.magnetic_quantum_numbers is wrong")
            raise ValueError


def compute_AOs_overlap_matrix(aos_data: AOs_data, method: str = "numerical"):
    """
    The method is for computing overlap matrix (S) of the given atomic orbitals

    Args:
        ao_datas (AOs_data): an instance of AOs_data
        method: method to compute S, numerical or analytical. numerical is just for the debugging purpose.

    Returns:
        Arrays containing the overlap matrix (S) of the given AOs (dim: num_ao, num_ao)
    """

    if method == "numerical":
        nx = 300
        x_min = -6.0
        x_max = 6.0

        ny = 300
        y_min = -5.0
        y_max = 7.0

        nz = 300
        z_min = -9.0
        z_max = 3.0

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
        W_prime = Jacob * np.tile(W, (aos_data.num_ao, 1))

        Psi = compute_AOs_api(
            aos_data=aos_data, r_carts=(A + np.dot(B, r_prime.T)).T, jax_flag=True
        )

        S = np.dot(Psi, (W_prime * Psi).T)

        return S

    else:
        raise NotImplementedError


def compute_AOs_api(
    aos_data: AOs_data,
    r_carts: npt.NDArray[np.float64],
    jax_flag: bool = True,
) -> npt.NDArray[np.float64 | np.complex128]:
    """
    The method is for computing the value of the given atomic orbital at r_carts

    Args:
        ao_datas (AOs_data): an instance of AOs_data
        r_carts: Cartesian coordinates of electrons (dim: N_e, 3)
        jax_flag: if False, AOs are computed one by one using compute_AO for debuging purpose

    Returns:
    Arrays containing values of the AOs at r_carts. (dim: num_ao, N_e)
    """
    if jax_flag:
        return compute_AOs_jax(aos_data, r_carts)
    else:
        return compute_AOs_debug(aos_data, r_carts)


def compute_AOs_debug(
    aos_data: AOs_data, r_carts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64 | np.complex128]:
    """
    This is for debuging compute_AOs method. It is written in a very straightforward way.

    Args:
        ao_datas (AOs_data): an instance of AOs_data
        r_carts: Cartesian coordinates of electrons (dim: N_e, 3)

    Returns:
    Arrays containing values of the AOs at r_carts. (dim: num_ao, N_e)
    """

    def compute_each_AO(ao_index):
        atomic_center_cart = aos_data.atomic_center_carts[ao_index]
        shell_indices = [
            i for i, v in enumerate(aos_data.orbital_indices) if v == ao_index
        ]
        exponents = [aos_data.exponents[i] for i in shell_indices]
        coefficients = [aos_data.coefficients[i] for i in shell_indices]
        angular_momentum = aos_data.angular_momentums[ao_index]
        magnetic_quantum_number = aos_data.magnetic_quantum_numbers[ao_index]
        num_ao_prim = len(exponents)

        ao_data = AO_data(
            num_ao_prim=num_ao_prim,
            atomic_center_cart=atomic_center_cart,
            exponents=exponents,
            coefficients=coefficients,
            angular_momentum=angular_momentum,
            magnetic_quantum_number=magnetic_quantum_number,
        )

        ao_values = np.array(
            [compute_AO(ao_data=ao_data, r_cart=r_cart) for r_cart in r_carts]
        )

        return ao_values

    aos_values = np.array(
        [compute_each_AO(ao_index) for ao_index in range(aos_data.num_ao)]
    )

    return aos_values


@jit
def compute_AOs_jax(
    aos_data: AOs_data, r_carts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64 | np.complex128]:
    """
    This is jit-jax version compute_AOs method.

    Args:
        ao_datas (AOs_data): an instance of AOs_data
        r_carts: Cartesian coordinates of electrons (dim: N_e, 3)

    Returns:
    Arrays containing values of the AOs at r_carts. (dim: num_ao, N_e)
    """

    atomic_center_carts = aos_data.atomic_center_carts
    atomic_center_carts_dup = jnp.array(
        [atomic_center_carts[i] for i in aos_data.orbital_indices]
    )
    exponents = aos_data.exponents
    coefficients = aos_data.coefficients
    angular_momentums = aos_data.angular_momentums
    magnetic_quantum_numbers = aos_data.magnetic_quantum_numbers
    angular_momentums_dup = [angular_momentums[i] for i in aos_data.orbital_indices]

    # compute R_n inc. the whole normalization factor
    R_carts_dup = atomic_center_carts_dup
    c_jnp = jnp.array(coefficients)
    Z_jnp = jnp.array(exponents)
    l_jnp = jnp.array(angular_momentums_dup)
    N_n_dup = jnp.sqrt(
        (
            2.0 ** (2 * l_jnp + 3)
            * jscipy.special.factorial(l_jnp + 1)
            * (2 * Z_jnp) ** (l_jnp + 1.5)
        )
        / (jscipy.special.factorial(2 * l_jnp + 2) * jnp.sqrt(jnp.pi))
    )  # normalization factors

    def _compute_R_n_dup(N_n, coefficient, exponent, R_cart, r_carts):
        return (
            N_n
            * coefficient
            * jnp.exp(-1.0 * exponent * jnp.linalg.norm(r_carts - R_cart) ** 2)
        )

    vmap_compute_R_n_dup = vmap(
        vmap(_compute_R_n_dup, in_axes=(None, None, None, None, 0)),
        in_axes=(0, 0, 0, 0, None),
    )
    R_n_dup = vmap_compute_R_n_dup(N_n_dup, c_jnp, Z_jnp, R_carts_dup, r_carts)

    n_el = r_carts.shape[0]
    R_n = jnp.zeros([aos_data.num_ao, n_el])
    unique_indices = np.unique(aos_data.orbital_indices)
    for ui in unique_indices:
        mask = aos_data.orbital_indices == ui
        R_n = R_n.at[ui].set(R_n_dup[mask].sum(axis=0))

    # compute S_n
    def _compute_S_l_m(
        l: int,
        m: int,
        R_cart: npt.NDArray[np.float64],
        r_carts: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r_carts_rel = r_carts - R_cart
        x, y, z = r_carts_rel[..., 0], r_carts_rel[..., 1], r_carts_rel[..., 2]
        r_norm = jnp.sqrt(x**2 + y**2 + z**2)

        conditions = [
            (l == 0) & (m == 0),
            (l == 1) & (m == -1),
            (l == 1) & (m == 0),
            (l == 1) & (m == 1),
            (l == 2) & (m == -2),
            (l == 2) & (m == -1),
            (l == 2) & (m == 0),
            (l == 2) & (m == 1),
            (l == 2) & (m == 2),
            (l == 3) & (m == -3),
            (l == 3) & (m == -2),
            (l == 3) & (m == -1),
            (l == 3) & (m == 0),
            (l == 3) & (m == 1),
            (l == 3) & (m == 2),
            (l == 3) & (m == 3),
            (l == 4) & (m == -4),
            (l == 4) & (m == -3),
            (l == 4) & (m == -2),
            (l == 4) & (m == -1),
            (l == 4) & (m == 0),
            (l == 4) & (m == 1),
            (l == 4) & (m == 2),
            (l == 4) & (m == 3),
            (l == 4) & (m == 4),
        ]

        """see https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics"""
        S_l_m_values = [
            # s orbital
            1.0 / 2.0 * jnp.sqrt(1.0 / jnp.pi) * r_norm**0.0,  # (l, m) == (0, 0)
            # p orbitals
            jnp.sqrt(3.0 / (4 * jnp.pi)) * y,  # (l, m) == (1, -1)
            jnp.sqrt(3.0 / (4 * jnp.pi)) * z,  # (l, m) == (1, 0)
            jnp.sqrt(3.0 / (4 * jnp.pi)) * x,  # (l, m) == (1, 1)
            # d orbitals
            1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * x * y,  # (l, m) == (2, -2)
            1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * y * z,  # (l, m) == (2, -1)
            1.0
            / 4.0
            * jnp.sqrt(5.0 / (jnp.pi))
            * (3 * z**2 - r_norm**2),  # (l, m) == (2, 0):
            1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * x * z,  # (l, m) == (2, 1)
            1.0 / 4.0 * jnp.sqrt(15.0 / (jnp.pi)) * (x**2 - y**2),  # (l, m) == (2, 2)
            # f orbitals
            1.0
            / 4.0
            * jnp.sqrt(35.0 / (2 * jnp.pi))
            * y
            * (3 * x**2 - y**2),  # (l, m) == (3, -3)
            1.0 / 2.0 * jnp.sqrt(105.0 / (jnp.pi)) * x * y * z,  # (l, m) == (3, -2)
            1.0
            / 4.0
            * jnp.sqrt(21.0 / (2 * jnp.pi))
            * y
            * (5 * z**2 - r_norm**2),  # (l, m) == (3, -1)
            1.0
            / 4.0
            * jnp.sqrt(7.0 / (jnp.pi))
            * (5 * z**3 - 3 * z * r_norm**2),  # (l, m) == (3, 0)
            1.0
            / 4.0
            * jnp.sqrt(21.0 / (2 * jnp.pi))
            * x
            * (5 * z**2 - r_norm**2),  # (l, m) == (3, 1)
            1.0
            / 4.0
            * jnp.sqrt(105.0 / (jnp.pi))
            * (x**2 - y**2)
            * z,  # (l, m) == (3, 2)
            1.0
            / 4.0
            * jnp.sqrt(35.0 / (2 * jnp.pi))
            * x
            * (x**2 - 3 * y**2),  # (l, m) == (3, 3)
            # g orbitals
            3.0
            / 4.0
            * jnp.sqrt(35.0 / (jnp.pi))
            * x
            * y
            * (x**2 - y**2),  # (l, m) == (4, -4)
            3.0
            / 4.0
            * jnp.sqrt(35.0 / (2 * jnp.pi))
            * y
            * z
            * (3 * x**2 - y**2),  # (l, m) == (4, -3)
            3.0
            / 4.0
            * jnp.sqrt(5.0 / (jnp.pi))
            * x
            * y
            * (7 * z**2 - r_norm**2),  # (l, m) == (4, -2)
            (
                3.0
                / 4.0
                * jnp.sqrt(5.0 / (2 * jnp.pi))
                * y
                * (7 * z**3 - 3 * z * r_norm**2)
            ),  # (l, m) == (4, -1)
            (
                3.0
                / 16.0
                * jnp.sqrt(1.0 / (jnp.pi))
                * (35 * z**4 - 30 * z**2 * r_norm**2 + 3 * r_norm**4)
            ),  # (l, m) == (4, 0)
            (
                3.0
                / 4.0
                * jnp.sqrt(5.0 / (2 * jnp.pi))
                * x
                * (7 * z**3 - 3 * z * r_norm**2)
            ),  # (l, m) == (4, 1)
            (
                3.0
                / 8.0
                * jnp.sqrt(5.0 / (jnp.pi))
                * (x**2 - y**2)
                * (7 * z**2 - r_norm**2)
            ),  # (l, m) == (4, 2)
            (
                3.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * x * z * (x**2 - 3 * y**2)
            ),  # (l, m) == (4, 3)
            (
                3.0
                / 16.0
                * jnp.sqrt(35.0 / (jnp.pi))
                * (x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2))
            ),  # (l, m) == (4, 4)
        ]

        return jnp.select(conditions, S_l_m_values, default=jnp.nan)

    vmap_compute_S_l_m = vmap(
        vmap(_compute_S_l_m, in_axes=(None, None, None, 0)), in_axes=(0, 0, 0, None)
    )
    R_carts = atomic_center_carts
    l_jnp = jnp.array(angular_momentums)
    m_jnp = jnp.array(magnetic_quantum_numbers)
    S_l_m = vmap_compute_S_l_m(l_jnp, m_jnp, R_carts, r_carts)

    # final answer
    answer = R_n * S_l_m

    logger.debug(f"shape R_n is {R_n.shape}")
    logger.debug(f"shape S_l_m is {S_l_m.shape}")

    if answer.shape != (aos_data.num_ao, len(r_carts)):
        logger.error(
            f"answer.shape = {answer.shape} is inconsistent with the expected one = {(aos_data.num_ao, len(r_carts))}"
        )
        logger.error(f"R_n.shape = {R_n.shape}")
        logger.error(f"S_l_m.shape = {S_l_m.shape}")
        raise ValueError

    return answer


@dataclass
class AO_data:
    """
    The class contains data for computing an atomic orbital. Just for testing purpose.
    For fast computations, use AOs_data and AOs.

    Args:
        num_ao : the number of atomic orbitals.
        num_ao_prim : the number of primitive atomic orbitals.
        atomic_center_cart (list[float]): Center of the nucleus associated to the AO. dim: 3
        exponents (list[float]): List of exponents of the AO. dim: num_ao_prim
        coefficients (list[float | complex]): List of coefficients of the AO. dim: num_ao_prim
        angular_momentum (int): Angular momentum of the AO, i.e., l. dim: 1
        magnetic_quantum_number (int): Magnetic quantum number of the AO, i.e m = -l .... +l. dim: 1
    """

    num_ao_prim: int = 0
    atomic_center_cart: list[float] = field(default_factory=list)
    exponents: list[float] = field(default_factory=list)
    coefficients: list[float | complex] = field(default_factory=list)
    angular_momentum: int = 0
    magnetic_quantum_number: int = 0

    def __post_init__(self) -> None:
        if len(self.atomic_center_cart) != 3:
            logger.error("dim. of atomic_center_cart is wrong")
            raise ValueError
        if len(self.exponents) != self.num_ao_prim:
            logger.error("dim. of self.exponents is wrong")
            raise ValueError
        if len(self.coefficients) != self.num_ao_prim:
            logger.error("dim. of self.coefficients is wrong")
            raise ValueError
        if self.angular_momentum < np.abs(self.magnetic_quantum_number):
            logger.error(
                "angular_momentum(l) is smaller than magnetic_quantum_number(|m|)."
            )
            raise ValueError


def compute_AO(ao_data: AO_data, r_cart: list[float]) -> float | complex:
    """
    The method is for computing the value of the given atomic orbital at r_cart
    Just for testing purpose. For fast computations, use AOs_data and AOs.

    Args:
        ao_data (AO_data): an instance of AO_data
        r_cart: Cartesian coordinate of an electron

    Returns:
    Value of the AO value at r_cart.

    Note:
        The faster way to compute all AOs at the same time because one can avoid X-times calling np.exp and np.sphe calls.

        Atomic orbitals are given in the followng Gaussian form:
        \phi_{l+\pm |m|, \alpha}(\vec{r}) =
            e^{-Z_\alpha * |\vec{R_\alpha} - \vec{r}|^2} * |\vec{R_\alpha} - \vec{r}|^l [Y_{l,m,\alpha}(\phi, \theta) +- Y_{l,-m,\alpha}(\phi, \theta)]
        where [Y_{l,m,\alpha}(\phi, \theta) +- Y_{l,-m,\alpha}(\phi, \theta)] are real spherical harmonics function.

        As written in the following, the spherical harmonics function is not used in practice because it has singular points.
        Instead, the so-called solid harmonics function is computed, which is defined as
        Sha_{l,\pm|m|,\alpha} = |\vec{R_{\alpha} - \vec{r}|^l [Y_{l,m,\alpha}(\phi, \theta) +- Y_{l,-m,\alpha}(\phi, \theta)].

        Rad{\alpha}(r_cart) = e^{-Z_\alpha * |\vec{R_\alpha} - \vec{r}|^2}

        Finally, an AO, \phi_{l+\pm |m|, \alpha}(\vec{r}), is computed as:
            \phi_{l+\pm |m|, \alpha}(\vec{r})  = \mathcal{N}_{l,\alpha} * Rad{\alpha}(r_cart) * Sha_{l,\pm|m|,\alpha}(r_cart)
        where N is the normalization factor. N is computed as:
            \mathcal{N}_{l,\alpha} = \sqrt{\frac{2^{2l+3}(l+1)!(2Z_\alpha)^{l+\frac{3}{2}}}{(2l+2)!\sqrt{\pi}}}.
        Notice that this normalization factor is just for the primitive GTO. The contracted GTO is not explicitly normalized.
    """

    R_n = np.array(
        [
            compute_R_n(
                atomic_center_cart=ao_data.atomic_center_cart,
                exponent=Z,
                r_cart=r_cart,
            )
            for Z in ao_data.exponents
        ]
    )
    N_n_l = np.array(
        [
            compute_normalization_fator(ao_data.angular_momentum, Z)
            for Z in ao_data.exponents
        ]
    )
    C_n = np.array(ao_data.coefficients)
    S_l_m = compute_S_l_m(
        atomic_center_cart=ao_data.atomic_center_cart,
        angular_momentum=ao_data.angular_momentum,
        magnetic_quantum_number=ao_data.magnetic_quantum_number,
        r_cart=r_cart,
    )

    return np.sum(N_n_l * C_n * R_n) * S_l_m


def compute_R_n(
    atomic_center_cart: list[float],
    exponent: float,
    r_cart: list[float],
) -> float | complex:
    """
    Radial part of the AO

    Args:
        atomic_center_cart (list[float]): Center of the nucleus associated to the AO.
        exponent (float): the exponent of the target AO.
        r_cart: Cartesian coordinate of an electron

    Returns:
        Value of the pure radial part.
    """

    return np.exp(
        -1 * exponent * LA.norm(np.array(r_cart) - np.array(atomic_center_cart)) ** 2
    )


def compute_S_l_m(
    atomic_center_cart: list[float],
    angular_momentum: int,
    magnetic_quantum_number: int,
    r_cart: list[float],
) -> float:
    """
    r^l * spherical hamonics part (i.e., regular solid harmonics) of the AO

    Args:
        atomic_center_cart (list[float]): Center of the nucleus associated to the AO.
        angular_momentum (int): Angular momentum of the AO, i.e., l
        magnetic_quantum_number (int): Magnetic quantum number of the AO, i.e m = -l .... +l
        r_cart: Cartesian coordinate of an electron

    Returns:
        Value of the spherical harmonics part * r^l (i.e., regular solid harmonics).

    Note:
        A real basis of spherical harmonics Y_{l,m} : S^2 -> R can be defined in terms of
        their complex analogues  Y_{l}^{m} : S^2 -> C by setting:
        Y_{l,m}(theta, phi) =
                sqrt(2) * (-1)^m * \Im[Y_l^{|m|}] (if m < 0)
                Y_l^{0} (if m = 0)
                sqrt(2) * (-1)^m * \Re[Y_l^{|m|}] (if m > 0)

        A conversion from cartesian to spherical coordinate is:
                r = sqrt(x**2 + y**2 + z**2)
                theta = arccos(z/r)
                phi = sgn(y)arccos(x/sqrt(x**2+y**2))

        It indicates that there are two singular points
                1) the origin (x,y,z) = (0,0,0)
                2) points on the z axis (0,0,z)

        Therefore, instead, the so-called solid harmonics function is computed, which is defined as
        S_{l,\pm|m|} = \sqrt(\cfrac{2 * l + 1}{4 * np.pi}) * |\vec{R} - \vec{r}|^l [Y_{l,m,\alpha}(\phi, \theta) +- Y_{l,-m,\alpha}(\phi, \theta)].

        The real solid harmonics function are tabulated in many textbooks and websites such as Wikipedia.
        They can be hardcoded into a code, or they can be computed analytically (e.g., https://en.wikipedia.org/wiki/Solid_harmonics).
        The latter one is the strategy employed in this code,
    """

    R_cart = atomic_center_cart
    x, y, z = np.array(r_cart) - np.array(R_cart)
    r_norm = LA.norm(np.array(r_cart) - np.array(R_cart))
    l, m = angular_momentum, magnetic_quantum_number
    m_abs = np.abs(m)

    # solid harmonics for (x,y) dependent part:
    def A_m(x: float, y: float) -> float:
        return np.sum(
            [
                scipy.special.binom(m_abs, p)
                * x ** (p)
                * y ** (m_abs - p)
                * np.cos((m_abs - p) * (np.pi / 2.0))
                for p in range(0, m_abs + 1)
            ]
        )

    def B_m(x: float, y: float) -> float:
        return np.sum(
            [
                scipy.special.binom(m_abs, p)
                * x ** (p)
                * y ** (m_abs - p)
                * np.sin((m_abs - p) * (np.pi / 2.0))
                for p in range(0, m_abs + 1)
            ]
        )

    # solid harmonics for (z) dependent part:
    def lambda_lm(k: int) -> float:
        # logger.debug(f"l={l}, type ={type(l)}")
        return (
            (-1.0) ** (k)
            * 2.0 ** (-l)
            * scipy.special.binom(l, k)
            * scipy.special.binom(2 * l - 2 * k, l)
            * scipy.special.factorial(l - 2 * k)
            / scipy.special.factorial(l - 2 * k - m_abs)
        )

    # solid harmonics for (z) dependent part:
    def Lambda_lm(r_norm: float, z: float) -> float:
        return np.sqrt(
            (2 - int(m_abs == 0))
            * scipy.special.factorial(l - m_abs)
            / scipy.special.factorial(l + m_abs)
        ) * np.sum(
            [
                lambda_lm(k) * r_norm ** (2 * k) * z ** (l - 2 * k - m_abs)
                for k in range(0, int((l - m_abs) / 2) + 1)
            ]
        )

    # logger.debug(f"z = {z}")
    # logger.debug(f"l,m = {(l,m)}")
    # logger.debug(f"r_cart = {r_cart}")
    # logger.debug(f"R_cart = {R_cart}")
    # logger.debug(f"r_norm = {r_norm}")
    # logger.debug(f"Lambda_lm(r_norm, z)={Lambda_lm(r_norm, z)}")
    # logger.debug(f"A_m(x, y)={A_m(x, y)}")
    # logger.debug(f"B_m(x, y)={B_m(x, y)}")

    # solid harmonics eveluated in Cartesian coord. (x,y,z):
    if m >= 0:
        gamma = np.sqrt((2 * l + 1) / (4 * np.pi)) * Lambda_lm(r_norm, z) * A_m(x, y)
    if m < 0:
        gamma = np.sqrt((2 * l + 1) / (4 * np.pi)) * Lambda_lm(r_norm, z) * B_m(x, y)
    return gamma


def compute_normalization_fator(l: int, Z: float):
    # return 1.0
    return np.sqrt(
        (2.0 ** (2 * l + 3) * scipy.special.factorial(l + 1) * (2 * Z) ** (l + 1.5))
        / (scipy.special.factorial(2 * l + 2) * np.sqrt(np.pi))
    )


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    """
    num_r_cart_samples = 10
    num_R_cart_samples = 2
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_cart_samples, 3
    ) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(
        num_R_cart_samples, 3
    ) + R_cart_min

    num_ao = 2
    num_ao_prim = 3
    orbital_indices = [0, 1, 1]
    exponents = [50.0, 20.0, 10.0]
    coefficients = [1.0, 1.0, 1.0]
    angular_momentums = [0, 1]
    magnetic_quantum_numbers = [0, 0]

    aos_data = AOs_data(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_compute_fast = compute_AOs_api(
        aos_data=aos_data, r_carts=r_carts, jax_flag=False
    )

    # overlap matrix

    num_ao = 3
    num_ao_prim = 3
    orbital_indices = [0, 1, 2]
    R_carts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    exponents = [3.0, 1.0, 0.5]
    coefficients = [1.0, 1.0, 1.0]
    angular_momentums = [0, 0, 0]
    magnetic_quantum_numbers = [0, 0, 0]

    aos_data = AOs_data(
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        atomic_center_carts=R_carts,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    print(1 / (4 * np.pi) * np.sqrt(np.pi / (2 * 10.0)) ** 3)
    print(1 / (4 * np.pi) * np.sqrt(np.pi / (2 * 3.0)) ** 3)
    print(1 / (4 * np.pi) * np.sqrt(np.pi / (2 * 1.0)) ** 3)
    compute_AOs_overlap_matrix(aos_data=aos_data)
    """

    @jit
    def comp_S_l_m(l, m, r_cart):
        def A(r_cart):
            return jnp.linalg.norm(r_cart) * m * l

        def B(r_cart):
            return jnp.linalg.norm(r_cart) * m * l

        # 要素ごとの論理AND演算を行うには '&' を使用
        return jnp.where((m == 0) & (l == 0), A(r_cart), B(r_cart))

    # データの定義
    r_R = jnp.array([[0, 0, 0], [2, 2, 2], [2, 2, 2], [3, 3, 3]])
    l_list = [0, 1, 1]
    m_list = [0, -2, 2]

    # l, mの各ペアに対してr_Rの全行に関数を適用するためのvmap
    vmap_compute_S_l_m = vmap(
        vmap(comp_S_l_m, in_axes=(None, None, 0)), in_axes=(0, 0, None)
    )
    l_jnp = jnp.array(l_list)
    m_jnp = jnp.array(m_list)
    S_l_m = vmap_compute_S_l_m(l_jnp, m_jnp, r_R)

    print(S_l_m)
