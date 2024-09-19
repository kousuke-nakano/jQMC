"""Hamiltonian module"""

# python modules
import numpy as np
import numpy.typing as npt
from logging import getLogger, StreamHandler, Formatter

# JAX
import jax
from flax import struct

from .wavefunction import Wavefunction_data, compute_kinetic_energy_api
from .coulomb_potential import Coulomb_potential_data, compute_coulomb_potential_api
from .structure import Structure_data

# set logger
logger = getLogger("myqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


@struct.dataclass
class Hamiltonian_data:
    """
    The class contains data for computing laplacians

    Args:
        structure_data (Structure_data)
        coulomb_data (Coulomb_data)
        wavefunction_data (Wavefunction_data)

    Comments:
        Heres are the differentiable arguments, i.e., pytree_node = True

        WF parameters related:
            - mo_coefficients in MOs_data (molecular_orbital.py) (switched off, for the time being)
            - (ao_)coefficients in AOs_data (atomic_orbital.py) (switched off, for the time being)
            - (ao_)exponents in AOs_data (atomic_orbital.py) (switched off, for the time being)
            - lambda_matrix in Geminal_data (determinant.py) (switched off, for the time being)
            - param_parallel_spin in Jastrow_two_body_data (jastrow_factor.py)
            - param_anti_parallel_spin in Jastrow_two_body_data (jastrow_factor.py)

        Atomic positions related:
            - structure_data.positions in AOs_data (atomic_orbital.py)
            - structure_data.positions in Coulomb_potential_data (coulomb_potential.py)

    """

    structure_data: Structure_data = None
    coulomb_potential_data: Coulomb_potential_data = None
    wavefunction_data: Wavefunction_data = None

    def __post_init__(self) -> None:
        pass


def compute_local_energy(
    hamiltonian_data: Hamiltonian_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """
    The method is for computing the local energy at (r_up_carts, r_dn_carts).

    Args:
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The value of local energy with the given wavefunction (float | complex)
    """

    T = compute_kinetic_energy_api(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    V = compute_coulomb_potential_api(
        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        wavefunction_data=hamiltonian_data.wavefunction_data,
    )

    logger.debug(f"e_L = {T+V} Ha")

    return T + V


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
