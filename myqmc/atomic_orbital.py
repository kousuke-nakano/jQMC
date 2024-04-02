"""Atomic Orbital module"""

# python modules
from dataclasses import dataclass, field
import itertools
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

from scipy.special import binom, factorial  # type: ignore

# set logger
from logging import getLogger, StreamHandler, Formatter

logger = getLogger("myqmc").getChild(__name__)


@dataclass
class AO_data:
    """
    The class contains data for computing an atomic orbital. Just for testing purpose.
    For fast computations, use AOs_data and AOs.

    Args:
        vec_a (list[float]): lattice vector a. The unit is Bohr
        vec_b (list[float]): lattice vector b. The unit is Bohr
        vec_c (list[float]): lattice vector c. The unit is Bohr
        atomic_center_cart (list[float]): Center of the nucleus associated to the AO.
        exponents (list[float]): List of exponents of the AO.
        coefficients (list[float | complex]): List of coefficients of the AO.
        angular_momentum (int): Angular momentum of the AO, i.e., l
        magnetic_quantum_number (int): Magnetic quantum number of the AO, i.e m = -l .... +l
    """

    vec_a: list[float] = field(default_factory=list)
    vec_b: list[float] = field(default_factory=list)
    vec_c: list[float] = field(default_factory=list)
    atomic_center_cart: list[float] = field(default_factory=list)
    exponents: list[float] = field(default_factory=list)
    coefficients: list[float | complex] = field(default_factory=list)
    angular_momentum: int = 0
    magnetic_quantum_number: int = 0

    def __post__init__(self) -> None:
        if self.angular_momentum < np.abs(self.magnetic_quantum_number):
            logger.error(
                "angular_momentum(l) is smaller than magnetic_quantum_number(|m|)."
            )
            raise ValueError

    @property
    def norm_vec_a(self) -> float:
        return LA.norm(self.vec_a)

    @property
    def norm_vec_b(self) -> float:
        return LA.norm(self.vec_b)

    @property
    def norm_vec_c(self) -> float:
        return LA.norm(self.vec_c)

    @property
    def cell(self) -> npt.NDArray[np.float64]:
        """
        Returns:
            3x3 cell matrix containing `vec_a`, `vec_b`, and `vec_c`
        """
        cell = np.array([self.vec_a, self.vec_b, self.vec_c])
        return cell


@dataclass
class AOs_data:
    """
    The class contains data for computing atomic orbitals simltaneously

    Args:
        vec_a (list[float]): lattice vector a. The unit is Bohr
        vec_b (list[float]): lattice vector b. The unit is Bohr
        vec_c (list[float]): lattice vector c. The unit is Bohr
        atomic_center_carts (npt.NDArray[np.float64]): Centers of the nuclei associated to the AOs (dim: num_AOs, 3).
        orbital_indices (list[int]): index for what exponents and coefficients are associated to each atomic orbital
        exponents (list[float]): List of exponents of the AOs.
        coefficients (list[float | complex]): List of coefficients of the AOs.
        angular_momentums (list[int]): Angular momentum of the AOs, i.e., l
        magnetic_quantum_numbers (list[int]): Magnetic quantum number of the AOs, i.e m = -l .... +l
    """

    vec_a: list[float] = field(default_factory=list)
    vec_b: list[float] = field(default_factory=list)
    vec_c: list[float] = field(default_factory=list)
    atomic_center_carts: npt.NDArray[np.float64] = np.array([[]])
    orbital_indices: list[int] = field(default_factory=list)
    exponents: list[float] = field(default_factory=list)
    coefficients: list[float | complex] = field(default_factory=list)
    angular_momentums: list[int] = field(default_factory=list)
    magnetic_quantum_numbers: list[int] = field(default_factory=list)

    """ to be refactored
    def __post__init__(self) -> None:
        if self.angular_momentum < np.abs(self.magnetic_quantum_number):
            logger.error(
                "angular_momentum(l) is smaller than magnetic_quantum_number(|m|)."
            )
            raise ValueError
    """

    @property
    def num_AOs(self) -> int:
        return len(self.angular_momentums)

    @property
    def norm_vec_a(self) -> float:
        return LA.norm(self.vec_a)

    @property
    def norm_vec_b(self) -> float:
        return LA.norm(self.vec_b)

    @property
    def norm_vec_c(self) -> float:
        return LA.norm(self.vec_c)

    @property
    def cell(self) -> npt.NDArray[np.float64]:
        """
        Returns:
            3x3 cell matrix containing `vec_a`, `vec_b`, and `vec_c`
        """
        cell = np.array([self.vec_a, self.vec_b, self.vec_c])
        return cell


def compute_AOs(
    aos_data: AOs_data, r_carts: npt.NDArray[np.float64], debug_flag=True
) -> npt.NDArray[np.float64 | np.complex128]:
    """
    The method is for computing the value of the given atomic orbital at r_carts

    Args:
        ao_datas (AOs_data): an instance of AOs_data
        r_carts: Cartesian coordinates of electrons (dim: N_e, 3)
        debug_flag: if True, AOs are computed one by one using compute_AO

    Returns:
    Arrays containing values of the AOs at r_carts. (dim: orbital_index, N_e)
    """

    if debug_flag:
        vec_a = aos_data.vec_a
        vec_b = aos_data.vec_b
        vec_c = aos_data.vec_c

        def compute_each_AO(ao_index):
            atomic_center_cart = aos_data.atomic_center_carts[ao_index]
            orbital_indices = [
                i for i, v in enumerate(aos_data.orbital_indices) if v == ao_index
            ]
            exponents = [aos_data.exponents[i] for i in orbital_indices]
            coefficients = [aos_data.coefficients[i] for i in orbital_indices]
            angular_momentum = aos_data.angular_momentums[ao_index]
            magnetic_quantum_number = aos_data.magnetic_quantum_numbers[ao_index]

            ao_data = AO_data(
                vec_a=vec_a,
                vec_b=vec_b,
                vec_c=vec_c,
                atomic_center_cart=atomic_center_cart,
                exponents=exponents,
                coefficients=coefficients,
                angular_momentum=angular_momentum,
                magnetic_quantum_number=magnetic_quantum_number,
            )

            ao_values = [
                compute_AO(ao_data=ao_data, r_cart=r_cart) for r_cart in r_carts
            ]

            return ao_values

        aos_values = np.array(
            [compute_each_AO(ao_index) for ao_index in range(aos_data.num_AOs)]
        )

        return aos_values

    else:
        atomic_center_carts = aos_data.atomic_center_carts
        atomic_center_carts_dup = np.array(
            [atomic_center_carts[i] for i in aos_data.orbital_indices]
        )
        exponents = aos_data.exponents
        coefficients = aos_data.coefficients
        angular_momentums = aos_data.angular_momentums
        magnetic_quantum_numbers = aos_data.magnetic_quantum_numbers

        n_orb = atomic_center_carts_dup.shape[0]
        n_el = r_carts.shape[0]
        logger.debug(f"n_orb={n_orb}")
        logger.debug(f"n_el={n_el}")
        sq_r_R = np.array(
            [
                LA.norm(v[0] - v[1]) ** 2
                for v in itertools.product(atomic_center_carts_dup, r_carts)
            ]
        ).reshape(n_orb, n_el)

        logger.debug(sq_r_R)
        logger.debug(np.array([exponents]).T)

        R_n_dup = np.array([coefficients]).T * np.exp(
            -1 * np.array([exponents]).T * sq_r_R
        )
        R_n = R_n_dup  # sum over the same orbital indices.
        S_l_m = 1  # to be implemented.
        return R_n * S_l_m


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

    atomic_center_cart = ao_data.atomic_center_cart
    exponents = ao_data.exponents
    coefficients = ao_data.coefficients
    angular_momentum = ao_data.angular_momentum
    magnetic_quantum_number = ao_data.magnetic_quantum_number
    R_n = compute_R_n(
        atomic_center_cart=atomic_center_cart,
        exponents=exponents,
        coefficients=coefficients,
        r_cart=r_cart,
    )
    S_l_m = compute_S_l_m(
        atomic_center_cart=angular_momentum,
        angular_momentum=angular_momentum,
        magnetic_quantum_number=magnetic_quantum_number,
        r_cart=r_cart,
    )

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
                binom(m_abs, p)
                * x ** (p)
                * y ** (m_abs - p)
                * np.cos((m_abs - p) * (np.pi / 2.0))
                for p in range(0, m_abs + 1)
            ]
        )

    def B_m(x: float, y: float) -> float:
        return np.sum(
            [
                binom(m_abs, p)
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
            * binom(l, k)
            * binom(2 * l - 2 * k, l)
            * factorial(l - 2 * k)
            / factorial(l - 2 * k - m_abs)
        )

    # solid harmonics for (z) dependent part:
    def Lambda_lm(r_norm: float, z: float) -> float:
        return np.sqrt(
            (2 - int(m_abs == 0)) * factorial(l - m_abs) / factorial(l + m_abs)
        ) * np.sum(
            [
                lambda_lm(k) * r_norm ** (2 * k) * z ** (l - 2 * k - m_abs)
                for k in range(0, int((l - m_abs) / 2) + 1)
            ]
        )

    logger.debug(f"l,m = {(l,m)}")
    logger.debug(f"r_cart = {r_cart}")
    logger.debug(f"r_norm = {r_norm}")
    logger.debug(f"Lambda_lm(r_norm, z)={Lambda_lm(r_norm, z)}")
    logger.debug(f"A_m(x, y)={A_m(x, y)}")
    logger.debug(f"B_m(x, y)={B_m(x, y)}")

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
    R_cart = (R_cart_max - R_cart_min) * np.random.rand(
        num_R_cart_samples, 3
    ) + R_cart_min

    orbital_indices = [0, 1, 1]
    exponents = [50.0, 20.0, 10.0]
    coefficients = [1.0, 1.0, 1.0]
    angular_momentums = [0, 1]
    magnetic_quantum_numbers = [0, 0]

    aos_data = AOs_data(
        atomic_center_carts=R_cart,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_compute_fast = compute_AOs(aos_data=aos_data, r_carts=r_carts, debug_flag=False)
    print(aos_compute_fast)
