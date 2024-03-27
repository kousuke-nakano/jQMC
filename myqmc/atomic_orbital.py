"""Atomic Orbital module"""

# python modules
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

from scipy.special import binom, factorial

# set logger
from logging import getLogger, StreamHandler, Formatter

logger = getLogger("myqmc").getChild(__name__)


class Atomic_orbital_sphe:
    """Atomic_orbital_sphe class

    The class contains info. for computing an atomic orbital.

    Args:
        atomic_center_cart (npt.NDArray[np.float]): Center of the nucleus associated to the AO.
        exponents (npt.NDArray[np.float]): List of exponents of the AO.
        coefficients (npt.NDArray[np.float]): List of coefficients of the AO.
        angular_momentum (int): Angular momentum of the AO, i.e., l
        magnetic_quantum_number (int): Magnetic quantum number of the AO, i.e m = -l .... +l

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

    def __init__(
        self,
        atomic_center_cart: npt.NDArray[np.float] = np.array([0.0, 0.0, 0.0]),
        exponents: npt.NDArray[np.float] = np.array([0.0]),
        coefficients: npt.NDArray[np.float] = np.array([1.0]),
        angular_momentum: int = 0,
        magnetic_quantum_number: int = 0,
    ):
        self.__atomic_center_cart = atomic_center_cart
        self.__exponents = exponents
        self.__coefficients = coefficients
        self.__angular_momentum = angular_momentum
        self.__magnetic_quantum_number = magnetic_quantum_number

    def R_n(self, r_cart: npt.NDArray[np.float]) -> np.float:
        """
        Radial part of the AO

        Args:
            r_cart: Cartesian coordinate of an electron

        Returns:
            Value of the radial part.
        """
        R_cart = self.__atomic_center_cart
        return np.dot(
            self.__coefficients,
            np.exp(self.__exponents * LA.norm(r_cart - R_cart) ** 2),
        )

    def S_l_m(self, r_cart: npt.NDArray[np.float]) -> np.float:
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
            S_{l,\pm|m|} = |\vec{R} - \vec{r}|^l [Y_{l,m,\alpha}(\phi, \theta) +- Y_{l,-m,\alpha}(\phi, \theta)].

            The real solid harmonics function are tabulated in many textbooks and websites such as Wikipedia.
            They can be hardcoded into a code, or they can be computed analytically (e.g., https://en.wikipedia.org/wiki/Solid_harmonics).
            The latter one is the strategy employed in this code,

        """
        R_cart = self.__atomic_center_cart
        x, y, z = r_cart - R_cart
        l, m = self.__angular_momentum, self.__magnetic_quantum_number

        # solid harmonics for (x,y) dependent part:
        def A_m(x: float, y: float) -> np.float:
            return np.sum(
                [
                    binom(m, p)
                    * x ** (p)
                    * y ** (m - p)
                    * np.cos(m - p)
                    * (np.pi / 2.0)
                    for p in range(0, m + 1)
                ]
            )

        def B_m(x: float, y: float) -> np.float:
            return np.sum(
                [
                    binom(m, p)
                    * x ** (p)
                    * y ** (m - p)
                    * np.sin(m - p)
                    * (np.pi / 2.0)
                    for p in range(0, m + 1)
                ]
            )

        # solid harmonics for (z) dependent part:
        def lambda_lm(k: int) -> np.float:
            return (
                (-1) ** (-k)
                * 2 ** (-l)
                * binom(l, k)
                * binom(2 * l - 2 * k, l)
                * factorial(l - 2 * k)
                / factorial(l - 2 * k - 2 * m)
            )

        # solid harmonics for (z) dependent part:
        def Lambda_lm(r_cart: npt.NDArray[np.float], z: float) -> np.float:
            return np.sum(
                [
                    lambda_lm(k) * LA.norm(r_cart) ** (2 * k) * z ** (l - 2 * k - m)
                    for k in range(0, int((l - m) / 2) + 1)
                ]
            )

        # solid harmonics in Cartesian (x,y,z):
        if m >= 0:
            return (
                np.sqrt((2 - int(m == 0)) * factorial(l - m) / factorial(l + m))
                * Lambda_lm(r_cart, z)
                * A_m(x, y)
            )
        if m < 0:
            m = np.abs(m)
            return (
                np.sqrt(2 * factorial(l - m) / factorial(l + m))
                * Lambda_lm(r_cart, z)
                * B_m(x, y)
            )

    def compute(self, r_cart: npt.NDArray[np.float]) -> np.float:
        return self.R_n(r_cart=r_cart) * self.S_l_m(r_cart=r_cart)


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    exponents = np.array([13.0, 5.0, 1.0])
    coefficients = np.array([0.001, 0.01, -1.0])

    ao_sphe = Atomic_orbital_sphe(exponents=exponents, coefficients=coefficients)
    r_cart = np.array([0.0, 0.0, 1.0])
    print(ao_sphe.calc(r_cart=r_cart))
