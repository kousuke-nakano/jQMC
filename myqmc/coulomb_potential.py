"""Effective core potential module"""

# python modules
from typing import NamedTuple
import itertools
from logging import getLogger, StreamHandler, Formatter

import numpy as np
import numpy.typing as npt
from scipy.special import eval_legendre

# JAX
from jax import lax
from jax.debug import print as jprint
from jax import vmap, jit
import jax.numpy as jnp
from flax import struct

from .structure import Structure_data
from .wavefunction import Wavefunction_data, evaluate_wavefunction


logger = getLogger("myqmc").getChild(__name__)


# non local PPs, Mesh Info. taken from Mitas's paper [J. Chem. Phys., 95, 5, (1991)]
class Mesh(NamedTuple):
    Nv: int
    weights: list[float]
    grid_points: npt.NDArray[np.float64]


# Tetrahedron symmetry quadrature (Nv=4)
q = 1 / np.sqrt(3)
A = 1.0 / 4.0
tetrahedron_sym_mesh_Nv4 = Mesh(
    Nv=4,
    weights=[A, A, A, A],
    grid_points=np.array([[q, q, q], [q, -q, -q], [-q, q, -q], [-q, -q, q]]),
)

# Octahedron symmetry quadrature (Nv=6)
A = 1.0 / 6.0
octahedron_sym_mesh_Nv6 = Mesh(
    Nv=6,
    weights=[A, A, A, A, A, A],
    grid_points=np.array(
        [
            [+1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, +1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, +1.0],
            [0.0, 0.0, -1.0],
        ]
    ),
)

# Octahedron symmetry quadrature (Nv=18)
A = 1.0 / 6.0
B = 1.0 / 15.0
p = 1.0 / np.sqrt(2)
octahedron_sym_mesh_Nv18 = Mesh(
    Nv=18,
    weights=[A, A, A, A, A, A, B, B, B, B, B, B, B, B, B, B, B, B],
    grid_points=np.array(
        [
            [+1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, +1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, +1.0],
            [0.0, 0.0, -1.0],
            [+p, +p, 0.0],
            [+p, -p, 0.0],
            [-p, +p, 0.0],
            [-p, -p, 0.0],
            [+p, 0.0, +p],
            [+p, 0.0, -p],
            [-p, 0.0, +p],
            [-p, 0.0, -p],
            [0.0, +p, +p],
            [0.0, -p, +p],
            [0.0, +p, -p],
            [0.0, -p, -p],
        ]
    ),
)


@struct.dataclass
class Coulomb_potential_data:
    """
    The class contains data for computing effective core potentials (ECPs).

    Args:
        # Structure part
        structure_data (Structure_data): Instance of a structure_data
        # Effective core potential part
        ecp_flag (bool) : If True, ECPs are used. The following values should be defined.
        z_cores (list[float]]): Number of core electrons to remove per atom (dim: num_atoms).
        max_ang_mom_plus_1 (list[int]): l_{max}+1, one higher than the max angular momentum in the removed core orbitals (dim: num_atoms)
        num_ecps (list[int]): Total number of ECP functions for all atoms and all values of l
        ang_moms (list[int]): One-to-one correspondence between ECP items and the angular momentum l (dim:num_ecps)
        nucleus_index (list[int]): One-to-one correspondence between ECP items and the atom index (dim:num_ecps)
        exponents (list[float]): all ECP exponents (dim:num_ecps)
        coefficients (list[float]): all ECP coefficients (dim:num_ecps)
        powers (list[int]): all ECP powers (dim:num_ecps)
    """

    structure_data: Structure_data = struct.field(pytree_node=True)
    ecp_flag: bool = struct.field(pytree_node=False)
    z_cores: list[float] = struct.field(pytree_node=False)
    max_ang_mom_plus_1: list[int] = struct.field(pytree_node=False)
    num_ecps: list[int] = struct.field(pytree_node=False)
    ang_moms: list[int] = struct.field(pytree_node=False)
    nucleus_index: list[int] = struct.field(pytree_node=False)
    exponents: list[float] = struct.field(pytree_node=False)
    coefficients: list[float] = struct.field(pytree_node=False)
    powers: list[int] = struct.field(pytree_node=False)

    def __post_init__(self) -> None:
        pass


def compute_ecp_local_parts_api(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    debug_flag: bool = True,
) -> float:

    # very slow but straightforward implementation, just for debugging purpose
    if debug_flag:
        V_local = compute_ecp_local_parts_debug(
            coulomb_potential_data, r_up_carts, r_dn_carts
        )
    else:
        V_local = compute_ecp_local_parts_jax(
            coulomb_potential_data, r_up_carts, r_dn_carts
        )

    return V_local


def compute_ecp_local_parts_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    V_local = 0.0
    for i_atom in range(coulomb_potential_data.structure_data.natom):
        R_cart = coulomb_potential_data.structure_data.positions_cart[i_atom]
        max_ang_mom_plus_1 = coulomb_potential_data.max_ang_mom_plus_1[i_atom]
        nucleus_indices = [
            i for i, v in enumerate(coulomb_potential_data.nucleus_index) if v == i_atom
        ]
        ang_moms = [coulomb_potential_data.ang_moms[i] for i in nucleus_indices]
        exponents = [coulomb_potential_data.exponents[i] for i in nucleus_indices]
        coefficients = [coulomb_potential_data.coefficients[i] for i in nucleus_indices]
        powers = [coulomb_potential_data.powers[i] for i in nucleus_indices]

        ang_mom_indices = [i for i, v in enumerate(ang_moms) if v == max_ang_mom_plus_1]
        exponents = [exponents[i] for i in ang_mom_indices]
        coefficients = [coefficients[i] for i in ang_mom_indices]
        powers = [powers[i] for i in ang_mom_indices]

        for r_cart in r_up_carts:
            V_local += np.linalg.norm(R_cart - r_cart) ** -2.0 * np.sum(
                [
                    a
                    * np.linalg.norm(R_cart - r_cart) ** n
                    * np.exp(-b * np.linalg.norm(R_cart - r_cart) ** 2)
                    for a, n, b in zip(coefficients, powers, exponents)
                ]
            )
        for r_cart in r_dn_carts:
            V_local += np.linalg.norm(R_cart - r_cart) ** -2.0 * np.sum(
                [
                    a
                    * np.linalg.norm(R_cart - r_cart) ** n
                    * np.exp(-b * np.linalg.norm(R_cart - r_cart) ** 2)
                    for a, n, b in zip(coefficients, powers, exponents)
                ]
            )
    return V_local


# WIP
def compute_ecp_local_parts_jax(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    V_local = 0.0
    for i_atom in range(coulomb_potential_data.structure_data.natom):
        R_cart = coulomb_potential_data.structure_data.positions_cart[i_atom]
        max_ang_mom_plus_1 = coulomb_potential_data.max_ang_mom_plus_1[i_atom]
        nucleus_indices = [
            i for i, v in enumerate(coulomb_potential_data.nucleus_index) if v == i_atom
        ]
        ang_moms = [coulomb_potential_data.ang_moms[i] for i in nucleus_indices]
        exponents = [coulomb_potential_data.exponents[i] for i in nucleus_indices]
        coefficients = [coulomb_potential_data.coefficients[i] for i in nucleus_indices]
        powers = [coulomb_potential_data.powers[i] for i in nucleus_indices]

        ang_mom_indices = [i for i, v in enumerate(ang_moms) if v == max_ang_mom_plus_1]
        exponents = [exponents[i] for i in ang_mom_indices]
        coefficients = [coefficients[i] for i in ang_mom_indices]
        powers = [powers[i] for i in ang_mom_indices]

        for r_cart in r_up_carts:
            V_local += np.linalg.norm(R_cart - r_cart) ** -2.0 * np.sum(
                [
                    a
                    * np.linalg.norm(R_cart - r_cart) ** n
                    * np.exp(-b * np.linalg.norm(R_cart - r_cart) ** 2)
                    for a, n, b in zip(coefficients, powers, exponents)
                ]
            )
        for r_cart in r_dn_carts:
            V_local += np.linalg.norm(R_cart - r_cart) ** -2.0 * np.sum(
                [
                    a
                    * np.linalg.norm(R_cart - r_cart) ** n
                    * np.exp(-b * np.linalg.norm(R_cart - r_cart) ** 2)
                    for a, n, b in zip(coefficients, powers, exponents)
                ]
            )
    return V_local


def compute_ecp_nonlocal_parts(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    Nv: int = 6,
    debug_flag: bool = True,
) -> float:

    logger.debug(f"Nv={Nv} for non-local PP evaluations.")
    if Nv == 4:
        weights = tetrahedron_sym_mesh_Nv4.weights
        grid_points = tetrahedron_sym_mesh_Nv4.grid_points
    elif Nv == 6:
        weights = octahedron_sym_mesh_Nv6.weights
        grid_points = octahedron_sym_mesh_Nv6.grid_points
    elif Nv == 18:
        weights = octahedron_sym_mesh_Nv18.weights
        grid_points = octahedron_sym_mesh_Nv18.grid_points
    else:
        raise NotImplementedError

    # super slow but straightforward implementation, just for debugging purpose
    if debug_flag:
        V_nonlocal = 0.0

        wavefunction_denominator = evaluate_wavefunction(
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        logger.debug(f"wavefunction_denominator = {wavefunction_denominator}")

        for i_atom in range(coulomb_potential_data.structure_data.natom):
            logger.debug(f"^^^^^non-local part calc. for i_atom={i_atom}^^^^^")
            R_cart = coulomb_potential_data.structure_data.positions_cart[i_atom]
            logger.debug(f"  R_cart={R_cart}")
            max_ang_mom_plus_1 = coulomb_potential_data.max_ang_mom_plus_1[i_atom]
            nucleus_indices = [
                i
                for i, v in enumerate(coulomb_potential_data.nucleus_index)
                if v == i_atom
            ]

            ang_moms_all = [coulomb_potential_data.ang_moms[i] for i in nucleus_indices]
            exponents_all = [
                coulomb_potential_data.exponents[i] for i in nucleus_indices
            ]
            coefficients_all = [
                coulomb_potential_data.coefficients[i] for i in nucleus_indices
            ]
            powers_all = [coulomb_potential_data.powers[i] for i in nucleus_indices]

            for ang_mom in range(max_ang_mom_plus_1):
                logger.debug(f"=====non-local part calc. for ang_mom={ang_mom}======")
                ang_mom_indices = [
                    i for i, v in enumerate(ang_moms_all) if v == ang_mom
                ]
                exponents = [exponents_all[i] for i in ang_mom_indices]
                coefficients = [coefficients_all[i] for i in ang_mom_indices]
                powers = [powers_all[i] for i in ang_mom_indices]

                # up electrons
                for r_up_i, r_up_cart in enumerate(r_up_carts):
                    logger.debug(f"  r_up_i={r_up_i}, r_up_cart={r_up_cart}")
                    logger.debug(f"  coefficients={coefficients}")
                    logger.debug(f"  exponents={exponents}")
                    logger.debug(f"  powers={powers}")

                    V_l = np.linalg.norm(R_cart - r_up_cart) ** -2.0 * np.sum(
                        [
                            a
                            * np.linalg.norm(R_cart - r_up_cart) ** n
                            * np.exp(-b * np.linalg.norm(R_cart - r_up_cart) ** 2.0)
                            for a, n, b in zip(coefficients, powers, exponents)
                        ]
                    )

                    logger.debug(
                        [
                            a * np.exp(-b * np.linalg.norm(R_cart - r_up_cart) ** 2.0)
                            for a, n, b in zip(coefficients, powers, exponents)
                        ]
                    )
                    # logger.debug(f"  Local part:V_l = {V_l}")

                    for weight, vec_delta in zip(weights, grid_points):
                        logger.debug(f"  vec_delta={vec_delta}")
                        r_up_carts_on_mesh = r_up_carts.copy()
                        r_up_carts_on_mesh[r_up_i] = (
                            R_cart + np.linalg.norm(R_cart - r_up_cart) * vec_delta
                        )
                        logger.debug(
                            f"  r_up_carts_center[r_up_i]={r_up_carts[r_up_i]}"
                        )
                        logger.debug(
                            f"  r_up_carts_on_mesh[r_up_i]={r_up_carts_on_mesh[r_up_i]}"
                        )

                        cos_theta = np.dot(
                            (R_cart - r_up_cart) / np.linalg.norm(R_cart - r_up_cart),
                            ((vec_delta) / np.linalg.norm(vec_delta)),
                        )

                        logger.debug(f"  cos_theta={cos_theta}")
                        logger.debug(
                            f"  Legendre(cos_theta)={eval_legendre(ang_mom, cos_theta)}"
                        )

                        wf_ratio = (
                            evaluate_wavefunction(
                                wavefunction_data=wavefunction_data,
                                r_up_carts=r_up_carts_on_mesh,
                                r_dn_carts=r_dn_carts,
                            )
                            / wavefunction_denominator
                        )

                        logger.debug(f"  wf_ratio={wf_ratio}")

                        P_l = (
                            (2 * ang_mom + 1)
                            * eval_legendre(
                                ang_mom,
                                np.dot(
                                    (R_cart - r_up_cart)
                                    / np.linalg.norm(R_cart - r_up_cart),
                                    (vec_delta / np.linalg.norm(vec_delta)),
                                ),
                            )
                            * weight
                            * evaluate_wavefunction(
                                wavefunction_data=wavefunction_data,
                                r_up_carts=r_up_carts_on_mesh,
                                r_dn_carts=r_dn_carts,
                            )
                            / wavefunction_denominator
                        )
                        logger.debug(f"V_l * P_l={V_l * P_l}")
                        V_nonlocal += V_l * P_l

                # dn electrons
                for r_dn_i, r_dn_cart in enumerate(r_dn_carts):
                    V_l = np.linalg.norm(R_cart - r_dn_cart) ** -2 * np.sum(
                        [
                            a
                            * np.linalg.norm(R_cart - r_dn_cart) ** n
                            * np.exp(-b * np.linalg.norm(R_cart - r_dn_cart) ** 2)
                            for a, n, b in zip(coefficients, powers, exponents)
                        ]
                    )

                    for weight, vec_delta in zip(weights, grid_points):
                        r_dn_carts_on_mesh = r_dn_carts.copy()
                        r_dn_carts_on_mesh[r_dn_i] = (
                            R_cart + np.linalg.norm(R_cart - r_dn_cart) * vec_delta
                        )

                        P_l = (
                            (2 * ang_mom + 1)
                            * eval_legendre(
                                ang_mom,
                                np.dot(
                                    (R_cart - r_dn_cart)
                                    / np.linalg.norm(R_cart - r_dn_cart),
                                    vec_delta / np.linalg.norm(vec_delta),
                                ),
                            )
                            * weight
                            * evaluate_wavefunction(
                                wavefunction_data=wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts_on_mesh,
                            )
                            / wavefunction_denominator
                        )
                        logger.debug(f"V_l * P_l = {V_l * P_l}")
                        V_nonlocal += V_l * P_l

        return V_nonlocal
    else:
        raise NotImplementedError


def compute_bare_coulomb_potential_api(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    debug_flag: bool = False,
) -> float:
    """
    The method is for computing the bare coulomb potentials including all electron-electron,
    electron-ion (except. ECPs), and ion-ion interactions at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The bare Coulomb potential with r_up_carts and r_dn_carts. (float)
    """

    if debug_flag:
        bare_coulomb_potential = compute_bare_coulomb_potential_debug(
            coulomb_potential_data, r_up_carts, r_dn_carts
        )
    else:
        logger.debug("called compute_bare_coulomb_potential_jax")
        bare_coulomb_potential = compute_bare_coulomb_potential_jax(
            coulomb_potential_data, r_up_carts, r_dn_carts
        )

    return bare_coulomb_potential


def compute_bare_coulomb_potential_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:

    R_carts = coulomb_potential_data.structure_data.positions_cart
    if coulomb_potential_data.ecp_flag:
        R_charges = list(
            np.array(coulomb_potential_data.structure_data.atomic_numbers)
            - np.array(coulomb_potential_data.z_cores)
        )
    else:
        R_charges = coulomb_potential_data.structure_data.atomic_numbers
    r_up_charges = [-1 for _ in range(len(r_up_carts))]
    r_dn_charges = [-1 for _ in range(len(r_dn_carts))]

    all_carts = np.vstack([R_carts, r_up_carts, r_dn_carts])
    all_charges = R_charges + r_up_charges + r_dn_charges

    bare_coulomb_potential = np.sum(
        [
            (Z_a * Z_b) / np.linalg.norm(r_a - r_b)
            for (Z_a, r_a), (Z_b, r_b) in itertools.combinations(
                zip(all_charges, all_carts), 2
            )
        ]
    )

    return bare_coulomb_potential


@jit
def compute_bare_coulomb_potential_jax(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:

    R_carts = coulomb_potential_data.structure_data.positions_cart

    """
    def return_R_charges_ecp(atomic_numbers, z_cores):
        return jnp.array(atomic_numbers) - jnp.array(z_cores)

    def return_R_charges_all_electrons(atomic_numbers, z_cores):
        return jnp.array(atomic_numbers)

    R_charges = lax.cond(
        coulomb_potential_data.ecp_flag,
        return_R_charges_ecp,
        return_R_charges_all_electrons,
        coulomb_potential_data.structure_data.atomic_numbers,
        coulomb_potential_data.z_cores,
    )

    """

    # jprint(
    #    f"type(coulomb_potential_data.ecp_flag) = {type(coulomb_potential_data.ecp_flag)}"
    # )
    # jprint(f"coulomb_potential_data.ecp_flag={coulomb_potential_data.ecp_flag}")

    if coulomb_potential_data.ecp_flag:
        R_charges = jnp.array(
            coulomb_potential_data.structure_data.atomic_numbers
        ) - jnp.array(coulomb_potential_data.z_cores)
    else:
        R_charges = jnp.array(coulomb_potential_data.structure_data.atomic_numbers)

    r_up_charges = jnp.array([-1 for _ in range(len(r_up_carts))])
    r_dn_charges = jnp.array([-1 for _ in range(len(r_dn_carts))])

    all_carts = jnp.vstack([R_carts, r_up_carts, r_dn_carts])
    all_charges = jnp.hstack([R_charges, r_up_charges, r_dn_charges])

    bare_coulomb_potential = jnp.sum(
        jnp.array(
            [
                (Z_a * Z_b) / jnp.linalg.norm(r_a - r_b)
                for (Z_a, r_a), (Z_b, r_b) in itertools.combinations(
                    zip(all_charges, all_carts), 2
                )
            ]
        )
    )

    return bare_coulomb_potential


def compute_coulomb_potential_api(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    wavefunction_data: Wavefunction_data = None,
    debug_flag: bool = False,
) -> float:
    """
    The method is for computing the bare coulomb potentials including all electron-electron,
    electron-ion (inc. ECPs), and ion-ion interactions at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        wavefunction_data (Wavefunction_data): Wavefunction information needed to compute the non-local part

    Returns:
        Potential Energy at r_up_carts and r_dn_carts. (float)
    """

    return 0.0

    # all-electron
    if not coulomb_potential_data.ecp_flag:
        bare_coulomb_potential = compute_bare_coulomb_potential_api(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        ecp_local_coulomb_potential = 0
        ecp_nonlocal_coulomb_potential = 0

    # pseudo-potential
    else:
        bare_coulomb_potential = compute_bare_coulomb_potential_api(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=debug_flag,
        )
        ecp_local_coulomb_potential = compute_ecp_local_parts_api(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )
        ecp_nonlocal_coulomb_potential = compute_ecp_nonlocal_parts(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            debug_flag=True,
        )

        logger.info(f"bare_coulomb_potential = {bare_coulomb_potential}.")
        logger.info(f"ecp_local_coulomb_potential  = {ecp_local_coulomb_potential}.")
        logger.info(
            f"ecp_nonlocal_coulomb_potential  = {ecp_nonlocal_coulomb_potential}."
        )

    return (
        bare_coulomb_potential
        + ecp_local_coulomb_potential
        + ecp_nonlocal_coulomb_potential
    )


if __name__ == "__main__":
    import os
    from .trexio_wrapper import read_trexio_file
    from .hamiltonians import Hamiltonian_data
    from .wavefunction import evaluate_wavefunction, compute_kinetic_energy

    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(__file__),
            "../",
            "tests",
            "trexio_example_files",
            "water_trexio.hdf5",
        )
    )

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    old_r_up_carts = np.array(
        [
            [-0.64878536, -0.83275288, 0.33532629],
            [0.55271273, 0.72310605, 0.93443775],
            [0.66767275, 0.1206456, -0.36521208],
            [-0.93165236, -0.0120386, 0.33003036],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [-1.0347816, 1.26162081, 0.42301735],
            [-0.57843435, 1.03651987, -0.55091542],
            [-1.56091964, -0.58952149, -0.99268141],
            [0.61863233, -0.14903326, 0.51962683],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_dn_carts[3] = [0.618632327645002, -0.149033260668010, 0.131889254514777]

    WF_ratio = (
        evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = compute_bare_coulomb_potential_api(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        debug_flag=False,
    )

    vpot_bare_debug = compute_bare_coulomb_potential_api(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        debug_flag=True,
    )

    logger.debug(f"vpot_bare_jax = {vpot_bare_jax}")
    logger.debug(f"vpot_bare_debug = {vpot_bare_debug}")
    np.testing.assert_almost_equal(vpot_bare_jax, vpot_bare_debug, decimal=10)

    vpot_bare_jax = compute_bare_coulomb_potential_api(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        debug_flag=False,
    )

    vpot_bare_jax = compute_bare_coulomb_potential_api(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        debug_flag=False,
    )

    """
    vpot_ecp_local = compute_ecp_local_parts(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_ecp_nonlocal = compute_ecp_nonlocal_parts(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )
    """
