"""Atomic Orbital module"""

# import sys

# python modules
from dataclasses import dataclass, field
import itertools
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

# scipy
import scipy  # type: ignore

# jax modules
import jax.scipy as jscipy
import jax.numpy as jnp
from flax import struct
from jax import grad

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


"""
def compute_AOs_fast(
    aos_data: AOs_data, r_carts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64 | np.complex128]:

    atomic_center_carts = aos_data.atomic_center_carts
    atomic_center_carts_dup = np.array(
        [atomic_center_carts[i] for i in aos_data.orbital_indices]
    )
    exponents = aos_data.exponents
    coefficients = aos_data.coefficients
    angular_momentums = aos_data.angular_momentums
    magnetic_quantum_numbers = aos_data.magnetic_quantum_numbers

    # compute R_n
    n_el = r_carts.shape[0]
    sq_r_R = np.array(
        [
            LA.norm(v[0] - v[1]) ** 2
            for v in itertools.product(atomic_center_carts_dup, r_carts)
        ]
    ).reshape(aos_data.num_ao_prim, n_el)

    logger.debug(sq_r_R)
    logger.debug(np.array([exponents]).T)

    R_n_dup = np.array([coefficients]).T * np.exp(-1 * np.array([exponents]).T * sq_r_R)

    R_n = np.zeros([aos_data.num_ao, n_el])
    unique_indices = np.unique(aos_data.orbital_indices)
    for ui in unique_indices:
        mask = aos_data.orbital_indices == ui
        R_n[ui] = R_n_dup[mask].sum(axis=0)

    # compute S_n

    # direct product of r_cart * R
    r_R = np.array(
        [v[1] - v[0] for v in itertools.product(atomic_center_carts, r_carts)]
    ).reshape(aos_data.num_ao, n_el, 3)

    def __compute_S_l_m(
        r_cart_rel: npt.NDArray[np.float64], l: int, m: int
    ) -> npt.NDArray[np.float64]:
        x, y, z = r_cart_rel[..., 0], r_cart_rel[..., 1], r_cart_rel[..., 2]
        r_norm = np.sqrt(x**2 + y**2 + z**2)
        m_abs = np.abs(m)

        # solid harmonics for (x,y) dependent part:
        def A_m(
            x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            return np.sum(
                [
                    scipy.special.binom(m_abs, p)
                    * x ** (p)
                    * y ** (m_abs - p)
                    * np.cos((m_abs - p) * (np.pi / 2.0))
                    for p in range(0, m_abs + 1)
                ],
                axis=0,
            )

        def B_m(
            x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            return np.sum(
                [
                    scipy.special.binom(m_abs, p)
                    * x ** (p)
                    * y ** (m_abs - p)
                    * np.sin((m_abs - p) * (np.pi / 2.0))
                    for p in range(0, m_abs + 1)
                ],
                axis=0,
            )

        # solid harmonics for (z) dependent part:
        def lambda_lm(k: int) -> float:
            return (
                (-1) ** (k)
                * 2 ** (-l)
                * scipy.special.binom(l, k)
                * scipy.special.binom(2 * l - 2 * k, l)
                * scipy.special.factorial(l - 2 * k)
                / scipy.special.factorial(l - 2 * k - m_abs)
            )

        # solid harmonics for (z) dependent part:
        def Lambda_lm(
            r_norm: npt.NDArray[np.float64], z: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            return np.sqrt(
                (2 - int(m_abs == 0))
                * scipy.special.factorial(l - m_abs)
                / scipy.special.factorial(l + m_abs)
            ) * np.sum(
                [
                    lambda_lm(k) * r_norm ** (2 * k) * z ** (l - 2 * k - m_abs)
                    for k in range(0, int((l - m_abs) / 2) + 1)
                ],
                axis=0,
            )

        # solid harmonics in Cartesian (x,y,z):
        if m >= 0:
            gamma = (
                np.sqrt((2 * l + 1) / (4 * np.pi)) * Lambda_lm(r_norm, z) * A_m(x, y)
            )
        if m < 0:
            gamma = (
                np.sqrt((2 * l + 1) / (4 * np.pi)) * Lambda_lm(r_norm, z) * B_m(x, y)
            )
        return gamma

    S_l_m = np.array(
        [
            __compute_S_l_m(
                r_cart_rel=r_cart_rel,
                l=angular_momentums[i],
                m=magnetic_quantum_numbers[i],
            )
            for i, r_cart_rel in enumerate(r_R)
        ]
    )

    # final answer
    answer = R_n * S_l_m

    if answer.shape != (aos_data.num_ao, len(r_carts)):
        logger.error(
            f"answer.shape = {answer.shape} is inconsistent with the expected one = {(aos_data.num_ao, len(r_carts))}"
        )
        logger.error(f"R_n.shape = {R_n.shape}")
        logger.error(f"S_l_m.shape = {S_l_m.shape}")
        raise ValueError

    return answer
"""


def compute_AOs_jax(
    aos_data: AOs_data, r_carts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64 | np.complex128]:
    """
    This is a straightforward jax code for compute_AOs method.

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

    # compute R_n
    n_el = r_carts.shape[0]
    sq_r_R = jnp.array(
        [
            jnp.linalg.norm(v[0] - v[1]) ** 2
            for v in itertools.product(atomic_center_carts_dup, r_carts)
        ]
    ).reshape(aos_data.num_ao_prim, n_el)

    R_n_dup = jnp.array([coefficients]).T * jnp.exp(
        -1 * jnp.array([exponents]).T * sq_r_R
    )

    R_n = jnp.zeros([aos_data.num_ao, n_el])
    unique_indices = np.unique(aos_data.orbital_indices)
    for ui in unique_indices:
        mask = aos_data.orbital_indices == ui
        R_n = R_n.at[ui].set(R_n_dup[mask].sum(axis=0))

    # compute S_n

    # direct product of r_cart * R
    r_R = jnp.array(
        [v[1] - v[0] for v in itertools.product(atomic_center_carts, r_carts)]
    ).reshape(aos_data.num_ao, n_el, 3)

    def __compute_S_l_m(
        r_cart_rel: npt.NDArray[np.float64], l: int, m: int
    ) -> npt.NDArray[np.float64]:
        x, y, z = r_cart_rel[..., 0], r_cart_rel[..., 1], r_cart_rel[..., 2]
        r_norm = jnp.sqrt(x**2 + y**2 + z**2)
        m_abs = jnp.abs(m)

        # solid harmonics for (x,y) dependent part:
        def A_m(
            x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            return jnp.sum(
                jnp.array(
                    [
                        scipy.special.binom(m_abs, p)
                        * x ** (p)
                        * y ** (m_abs - p)
                        * jnp.cos((m_abs - p) * (jnp.pi / 2.0))
                        for p in range(0, m_abs + 1)
                    ]
                ),
                axis=0,
            )

        def B_m(
            x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            return jnp.sum(
                jnp.array(
                    [
                        scipy.special.binom(m_abs, p)
                        * x ** (p)
                        * y ** (m_abs - p)
                        * jnp.sin((m_abs - p) * (jnp.pi / 2.0))
                        for p in range(0, m_abs + 1)
                    ]
                ),
                axis=0,
            )

        # solid harmonics for (z) dependent part:
        def lambda_lm(k: int) -> float:
            return (
                (-1) ** (k)
                * 2 ** (-l)
                * scipy.special.binom(l, k)
                * scipy.special.binom(2 * l - 2 * k, l)
                * jscipy.special.factorial(l - 2 * k)
                / jscipy.special.factorial(l - 2 * k - m_abs)
            )

        # solid harmonics for (z) dependent part:
        def Lambda_lm(
            r_norm: npt.NDArray[np.float64], z: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            return jnp.sqrt(
                (2 - int(m_abs == 0))
                * jscipy.special.factorial(l - m_abs)
                / jscipy.special.factorial(l + m_abs)
            ) * jnp.sum(
                jnp.array(
                    [
                        lambda_lm(k) * r_norm ** (2 * k) * z ** (l - 2 * k - m_abs)
                        for k in range(0, int((l - m_abs) / 2) + 1)
                    ]
                ),
                axis=0,
            )

        # solid harmonics in Cartesian (x,y,z):
        if m >= 0:
            gamma = (
                jnp.sqrt((2 * l + 1) / (4 * np.pi)) * Lambda_lm(r_norm, z) * A_m(x, y)
            )
        if m < 0:
            gamma = (
                jnp.sqrt((2 * l + 1) / (4 * np.pi)) * Lambda_lm(r_norm, z) * B_m(x, y)
            )
        return gamma

    S_l_m = jnp.array(
        [
            __compute_S_l_m(
                r_cart_rel=r_cart_rel,
                l=angular_momentums[i],
                m=magnetic_quantum_numbers[i],
            )
            for i, r_cart_rel in enumerate(r_R)
        ]
    )

    # final answer
    answer = R_n * S_l_m

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
            \phi_{l+\pm |m|, \alpha}(\vec{r})  = Rad{\alpha}(r_cart) * Sha_{l,\pm|m|,\alpha}(r_cart)
    """

    R_n = compute_R_n(
        atomic_center_cart=ao_data.atomic_center_cart,
        exponents=ao_data.exponents,
        coefficients=ao_data.coefficients,
        r_cart=r_cart,
    )
    S_l_m = compute_S_l_m(
        atomic_center_cart=ao_data.atomic_center_cart,
        angular_momentum=ao_data.angular_momentum,
        magnetic_quantum_number=ao_data.magnetic_quantum_number,
        r_cart=r_cart,
    )

    # logger.debug(f"R_n = {R_n}")
    # logger.debug(f"S_l_m = {S_l_m}")

    return R_n * S_l_m


def compute_R_n(
    atomic_center_cart: list[float],
    exponents: list[float],
    coefficients: list[float | complex],
    r_cart: list[float],
) -> float | complex:
    """
    Radial part of the AO

    Args:
        atomic_center_cart (list[float]): Center of the nucleus associated to the AO.
        exponents (list[float]): List of exponents of the AO.
        coefficients (list[float | complex]): List of coefficients of the AO.
        r_cart: Cartesian coordinate of an electron

    Returns:
        Value of the radial part.
    """
    R_cart = atomic_center_cart
    return np.inner(
        np.array(coefficients),
        np.exp(
            -1 * np.array(exponents) * LA.norm(np.array(r_cart) - np.array(R_cart)) ** 2
        ),
    )


def compute_S_l_m(
    atomic_center_cart: list[float],
    angular_momentum: int,
    magnetic_quantum_number: int,
    r_cart: list[float],
) -> float:
    """
    Spherical part of the AO

    Args:
        atomic_center_cart (list[float]): Center of the nucleus associated to the AO.
        angular_momentum (int): Angular momentum of the AO, i.e., l
        magnetic_quantum_number (int): Magnetic quantum number of the AO, i.e m = -l .... +l
        r_cart: Cartesian coordinate of an electron

    Returns:
        Value of the spherical part.

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
        return (
            (-1) ** (k)
            * 2 ** (-l)
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

    # solid harmonics in Cartesian (x,y,z):
    if m >= 0:
        gamma = np.sqrt((2 * l + 1) / (4 * np.pi)) * Lambda_lm(r_norm, z) * A_m(x, y)
    if m < 0:
        gamma = np.sqrt((2 * l + 1) / (4 * np.pi)) * Lambda_lm(r_norm, z) * B_m(x, y)
    return gamma


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

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
        aos_data=aos_data, r_carts=r_carts, debug_flag=False
    )
    print(aos_compute_fast)
