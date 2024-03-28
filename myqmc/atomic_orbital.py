"""Atomic Orbital module"""

# python modules
from dataclasses import dataclass, field
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


class AO:
    """
    The class contains info. for computing an atomic orbital. Just for testing purpose.
    For fast computations, use AOs_data and AOs.

    Args:
        ao_data (AO_data): an instance of AO_data

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

    def __init__(self, ao_data: AO_data):
        self.ao_data = ao_data

    def R_n(self, r_cart: list[float]) -> float | complex:
        """
        Radial part of the AO

        Args:
            r_cart: Cartesian coordinate of an electron

        Returns:
            Value of the radial part.
        """
        R_cart = self.ao_data.atomic_center_cart
        return np.inner(
            np.array(self.ao_data.coefficients),
            np.exp(
                np.array(self.ao_data.exponents)
                * LA.norm(np.array(r_cart) - np.array(R_cart)) ** 2
            ),
        )

    def S_l_m(self, r_cart: list[float]) -> float:
        """
        Spherical part of the AO

        Args:
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

        R_cart = self.ao_data.atomic_center_cart
        x, y, z = np.array(r_cart) - np.array(R_cart)
        l, m = self.ao_data.angular_momentum, self.ao_data.magnetic_quantum_number
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
        def Lambda_lm(r_cart: list[float], z: float) -> float:
            return np.sqrt(
                (2 - int(m_abs == 0)) * factorial(l - m_abs) / factorial(l + m_abs)
            ) * np.sum(
                [
                    lambda_lm(k) * LA.norm(r_cart) ** (2 * k) * z ** (l - 2 * k - m_abs)
                    for k in range(0, int((l - m_abs) / 2) + 1)
                ]
            )

        logger.debug(f"l,m = {(l,m)}")
        logger.debug(f"r_cart = {r_cart}")
        logger.debug(f"Lambda_lm(r_cart, z)={Lambda_lm(r_cart, z)}")
        logger.debug(f"A_m(x, y)={A_m(x, y)}")
        logger.debug(f"B_m(x, y)={B_m(x, y)}")

        # solid harmonics in Cartesian (x,y,z):
        if m >= 0:
            gamma = (
                np.sqrt((2 * l + 1) / (4 * np.pi)) * Lambda_lm(r_cart, z) * A_m(x, y)
            )
        if m < 0:
            gamma = (
                np.sqrt((2 * l + 1) / (4 * np.pi)) * Lambda_lm(r_cart, z) * B_m(x, y)
            )
        return gamma

    def compute(self, r_cart: list[float]) -> float | complex:
        """
        Compute the value of the AO at r_cart

        Args:
            r_cart: Cartesian coordinate of an electron

        Returns:
            Value of the AO value at r_cart.
        """
        return self.R_n(r_cart=r_cart) * self.S_l_m(r_cart=r_cart)


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    exponents = [13.0, 5.0, 1.0]
    coefficients = [0.001, 0.01, -1.0]

    ao_data = AO_data(exponents=exponents, coefficients=coefficients)
    ao_sphe = AO(ao_data=ao_data)
    r_cart = [0.0, 0.0, 1.0]
    print(ao_sphe.compute(r_cart=r_cart))


@dataclass
class AOs_data:
    """
    The class contains data for computing atomic orbitals

    Args:
        vec_a (list[float]): lattice vector a. The unit is Bohr
        vec_b (list[float]): lattice vector b. The unit is Bohr
        vec_c (list[float]): lattice vector c. The unit is Bohr
        atomic_center_cart (npt.NDArray[np.float64]): Centers of the nuclei associated to the AOs. len(orbital_index) * 3 dimension
        orbital_index (list[int]): index for what exponents and coefficients are associated to each atomic orbital
        exponents (list[float]): List of exponents of the AOs.
        coefficients (list[float | complex]): List of coefficients of the AOs.
        angular_momentum (list[int]): Angular momentum of the AOs, i.e., l
        magnetic_quantum_number (list[int]): Magnetic quantum number of the AOs, i.e m = -l .... +l
    """

    vec_a: list[float] = field(default_factory=list)
    vec_b: list[float] = field(default_factory=list)
    vec_c: list[float] = field(default_factory=list)
    atomic_center_cart: npt.NDArray[np.float64] = np.array([[]])
    orbita_index: list[int] = field(default_factory=list)
    exponents: list[float] = field(default_factory=list)
    coefficients: list[float | complex] = field(default_factory=list)
    angular_momentum: list[int] = field(default_factory=list)
    magnetic_quantum_number: list[int] = field(default_factory=list)

    """ to be refactored
    def __post__init__(self) -> None:
        if self.angular_momentum < np.abs(self.magnetic_quantum_number):
            logger.error(
                "angular_momentum(l) is smaller than magnetic_quantum_number(|m|)."
            )
            raise ValueError
    """

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
