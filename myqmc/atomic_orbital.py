"""Atomic Orbital module"""

# python modules
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

# set logger
from logging import getLogger, StreamHandler, Formatter

logger = getLogger("myqmc").getChild(__name__)

class Atomic_orbital_sphe:
    def __init__(
        self,
        atomic_center_cart:npt.NDArray[float] = np.array([0.0,0.0,0.0]),
        exponents:npt.NDArray[float]=np.array([0.0]),
        coefficients:npt.NDArray[float]=np.array([1.0]),
        angular_momentum:int=0,
        magnetic_quantum_number:int=0,
    ):
        self.__atomic_center_cart = atomic_center_cart
        self.__exponents = exponents
        self.__coefficients = coefficients
        self.__angular_momentum = angular_momentum
        self.__magnetic_quantum_number = magnetic_quantum_number
        
    def R_n(self, r_cart: npt.NDArray[float]) -> float:
        R_cart = self.__atomic_center_cart
        return np.dot(self.__coefficients, np.exp(self.__exponents*LA.norm(r_cart-R_cart)**2))

    def Y_l_m(self, r_cart: npt.NDArray[float]) -> float:
        R_cart = self.__atomic_center_cart
        theta, phi = r_cart, R_cart
        return 1.0
 
    def calc(self, r_cart: npt.NDArray[float]) -> float:
        return self.R_n(r_cart=r_cart) * self.Y_l_m(r_cart=r_cart)

if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
    
    exponents=np.array([13.0, 5.0, 1.0])
    coefficients=np.array([0.001, 0.01, -1.0])
    
    ao_sphe=Atomic_orbital_sphe(exponents=exponents, coefficients=coefficients)
    r_cart=np.array([0.0, 0.0, 1.0])
    print(ao_sphe.calc(r_cart=r_cart))