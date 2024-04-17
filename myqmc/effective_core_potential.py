"""Effective core potential module"""

# python modules
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from scipy.special import eval_legendre

# set logger
from logging import getLogger, StreamHandler, Formatter

logger = getLogger("myqmc").getChild(__name__)


@dataclass
class Effective_core_potential_data:
    """
    The class contains data for computing effective core potentials (ECPs).

    Args:
        num_atoms (int): the number of nuclei in the system
        atomic_center_carts (npt.NDArray[np.float64]): Centers of the nuclei (dim: num_atoms, 3).
        z_cores (list[float]]): Number of core electrons to remove per atom (dim: num_atoms).
        max_ang_mom_plus_1 (list[int]): l_{max}+1, one higher than the max angular momentum in the removed core orbitals (dim: num_atoms)
        num_ecps (list[int]): Total number of ECP functions for all atoms and all values of l
        ang_moms (list[int]): One-to-one correspondence between ECP items and the angular momentum l (dim:num_ecps)
        nucleus_index (list[int]): One-to-one correspondence between ECP items and the atom index (dim:num_ecps)
        exponents (list[float]): all ECP exponents (dim:num_ecps)
        coefficients (list[float]): all ECP coefficients (dim:num_ecps)
        powers (list[int]): all ECP powers (dim:num_ecps)

    """

    num_atoms: int = None
    atomic_center_carts: npt.NDArray[np.float64] = None
    z_cores: list[float] = None
    max_ang_mom_plus_1: list[int] = None
    num_ecps: list[int] = None
    ang_moms: list[int] = None
    nucleus_index: list[int] = None
    exponents: list[float] = None
    coefficients: list[float] = None
    powers: list[int] = None

    def __post_init__(self) -> None:
        pass


def compute_nearest_neighbors_nuclei_indices(
    num_nearest_neighbors_nuclei: int,
    atomic_center_carts: npt.NDArray[np.float64],
    r_carts: npt.NDArray[np.float64],
) -> list[tuple[int]]:
    """
    The method returning num_nearest_neighbors_nuclei indices.

    Args:
        num_nearest_neighbors_nuclei (int): number of searched nearest-neighbor nuclei
        atomic_center_carts (npt.NDArray[np.float64]): Centers of the nuclei (dim: num_atoms, 3).
        r_carts (npt.NDArray[np.float64]): Cartesian coordinates of electrons (dim: N_e, 3)

    Returns:
        list containing tuples including indices for nearest neighbors nuclei for each electron. (dim: N_e)
    """


def compute_local_parts(
    effective_core_potential_data: Effective_core_potential_data,
    r_carts: npt.NDArray[np.float64],
    debug_flag: bool = True,
) -> float:

    # very slow but straightforward implementation, just for debugging purpose
    if debug_flag:
        V_local = 0.0
        for i_atom in range(effective_core_potential_data.num_atoms):
            R_cart = effective_core_potential_data.atomic_center_carts[i_atom]
            max_ang_mom_plus_1 = effective_core_potential_data.max_ang_mom_plus_1[
                i_atom
            ]
            nucleus_indices = [
                i
                for i, v in enumerate(effective_core_potential_data.nucleus_index)
                if v == i_atom
            ]
            ang_moms = [
                effective_core_potential_data.ang_moms[i] for i in nucleus_indices
            ]
            exponents = [
                effective_core_potential_data.exponents[i] for i in nucleus_indices
            ]
            coefficients = [
                effective_core_potential_data.coefficients[i] for i in nucleus_indices
            ]
            powers = [effective_core_potential_data.powers[i] for i in nucleus_indices]

            ang_mom_indices = [
                i for i, v in enumerate(ang_moms) if v == max_ang_mom_plus_1
            ]
            exponents = [exponents[i] for i in ang_mom_indices]
            coefficients = [coefficients[i] for i in ang_mom_indices]
            powers = [exponents[i] for i in ang_mom_indices]

            for r_cart in r_carts:
                V_local += np.linalg.norm(R_cart - r_cart) ** -2 * np.sum(
                    [
                        a
                        * np.linalg.norm(R_cart - r_cart) ** n
                        * np.exp(-b * np.linalg.norm(R_cart - r_cart) ** 2)
                        for a, n, b in zip(coefficients, powers, exponents)
                    ]
                )
        return V_local
    else:
        raise NotImplementedError


def compute_nonlocal_parts(
    effective_core_potential_data: Effective_core_potential_data,
    # wave_function_data: Wave_function_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    debug_flag: bool = True,
) -> float:

    # super slow but straightforward implementation, just for debugging purpose
    if debug_flag:
        V_nonlocal = 0.0

        for i_atom in range(effective_core_potential_data.num_atoms):
            R_cart = effective_core_potential_data.atomic_center_carts[i_atom]
            max_ang_mom_plus_1 = effective_core_potential_data.max_ang_mom_plus_1[
                i_atom
            ]
            nucleus_indices = [
                i
                for i, v in enumerate(effective_core_potential_data.nucleus_index)
                if v == i_atom
            ]

            ang_moms_all = [
                effective_core_potential_data.ang_moms[i] for i in nucleus_indices
            ]
            exponents_all = [
                effective_core_potential_data.exponents[i] for i in nucleus_indices
            ]
            coefficients_all = [
                effective_core_potential_data.coefficients[i] for i in nucleus_indices
            ]
            powers_all = [
                effective_core_potential_data.powers[i] for i in nucleus_indices
            ]

            for ang_mom in range(max_ang_mom_plus_1):
                ang_mom_indices = [
                    i for i, v in enumerate(ang_moms_all) if v == ang_mom
                ]
                exponents = [exponents_all[i] for i in ang_mom_indices]
                coefficients = [coefficients_all[i] for i in ang_mom_indices]
                powers = [powers_all[i] for i in ang_mom_indices]

                for r_up_i, r_up_cart in enumerate(r_up_carts):
                    V_l = np.linalg.norm(R_cart - r_up_cart) ** -2 * np.sum(
                        [
                            a
                            * np.linalg.norm(R_cart - r_up_cart) ** n
                            * np.exp(-b * np.linalg.norm(R_cart - r_up_cart) ** 2)
                            for a, n, b in zip(coefficients, powers, exponents)
                        ]
                    )

                    P_l = 0

                    vec_delta_list = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])

                    for vec_delta in vec_delta_list:
                        r_up_carts_on_mesh = r_up_carts.copy()
                        r_up_carts_on_mesh[r_up_i] += (
                            np.linalg.norm(R_cart - r_up_cart) * vec_delta
                        )

                        P_l += (2 * ang_mom + 1) * eval_legendre(
                            ang_mom,
                            np.dot(
                                (R_cart - r_up_cart)
                                / np.linalg.norm(R_cart - r_up_cart),
                                (
                                    (R_cart - r_up_cart)
                                    / np.linalg.norm(R_cart - r_up_cart)
                                    + vec_delta
                                ),
                            ),
                        )
                        # * eval_wavefunction(
                        #    r_up_carts_on_mesh, r_dn_carts
                        # ) / eval_wavefunction(r_up_carts, r_dn_carts)

                    V_nonlocal += V_l * P_l

        return V_nonlocal
    else:
        raise NotImplementedError


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    # Hyrdogen dimer
    num_atoms = 2
    atomic_center_carts = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, +1.0]])
    z_cores = [0, 0]
    max_ang_mom_plus_1 = [1, 1]
    num_ecps = 8
    ang_moms = [1, 1, 1, 0, 1, 1, 1, 0]
    nucleus_index = [0, 0, 0, 0, 1, 1, 1, 1]
    coefficients = [
        1.00000000000000,
        21.24359508259891,
        -10.85192405303825,
        0.00000000000000,
        1.00000000000000,
        21.24359508259891,
        -10.85192405303825,
        0.00000000000000,
    ]
    exponents = [
        21.24359508259891,
        21.24359508259891,
        21.77696655044365,
        1.000000000000000,
        21.24359508259891,
        21.24359508259891,
        21.77696655044365,
        1.000000000000000,
    ]
    powers = [1, 3, 2, 2, 1, 3, 2, 2]

    effective_core_potential_data = Effective_core_potential_data(
        num_atoms=num_atoms,
        atomic_center_carts=atomic_center_carts,
        z_cores=z_cores,
        max_ang_mom_plus_1=max_ang_mom_plus_1,
        num_ecps=num_ecps,
        ang_moms=ang_moms,
        nucleus_index=nucleus_index,
        coefficients=coefficients,
        exponents=exponents,
        powers=powers,
    )

    num_r_cart_samples = 2
    r_cart_min, r_cart_max = -1.0, 1.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(
        num_r_cart_samples, 3
    ) + r_cart_min

    V_local = compute_local_parts(
        effective_core_potential_data=effective_core_potential_data, r_carts=r_carts
    )

    print(V_local)

    V_nonlocal = compute_nonlocal_parts(
        effective_core_potential_data=effective_core_potential_data,
        r_up_carts=r_carts,
        r_dn_carts=r_carts,
    )

    print(V_nonlocal)
