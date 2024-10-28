"""Effective core potential module.

Module containing classes and methods related to Effective core potential
and bare Coulomb potentials

Todo:
    * Replace numpy and jax.numpy typings with jaxtyping
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# python modules
import itertools
from logging import Formatter, StreamHandler, getLogger
from typing import NamedTuple

# JAX
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import jit, lax, vmap
from jax import typing as jnpt
from scipy.special import eval_legendre

from .jastrow_factor import Jastrow_data
from .miscs.function_collections import legendre_tablated as jnp_legendre_tablated
from .structure import Structure_data, get_min_dist_rel_R_cart_jnp, get_min_dist_rel_R_cart_np
from .wavefunction import Wavefunction_data, evaluate_wavefunction_api

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


# non local PPs, Mesh Info. taken from Mitas's paper [J. Chem. Phys., 95, 5, (1991)]
class _Mesh(NamedTuple):
    Nv: int
    weights: list[float]
    grid_points: npt.NDArray[np.float64]


# Tetrahedron symmetry quadrature (Nv=4)
q = 1 / np.sqrt(3)
A = 1.0 / 4.0
tetrahedron_sym_mesh_Nv4 = _Mesh(
    Nv=4,
    weights=[A, A, A, A],
    grid_points=np.array([[q, q, q], [q, -q, -q], [-q, q, -q], [-q, -q, q]]),
)

# Octahedron symmetry quadrature (Nv=6)
A = 1.0 / 6.0
octahedron_sym_mesh_Nv6 = _Mesh(
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
octahedron_sym_mesh_Nv18 = _Mesh(
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
    """Coulomb_potential dataclass.

    The class contains data for computing effective core potentials (ECPs).

    Args:
        structure_data (Structure_data):
            Instance of a structure_data
        ecp_flag (bool) :
            If True, ECPs are used. The following values should be defined.
        z_cores (list[float]]):
            Number of core electrons to remove per atom (dim: num_atoms).
        max_ang_mom_plus_1 (list[int]):
            l_{max}+1, one higher than the max angular momentum in the
            removed core orbitals (dim: num_atoms)
        num_ecps (list[int]):
            Total number of ECP functions for all atoms and all values of l
        ang_moms (list[int]):
            One-to-one correspondence between ECP items and the angular momentum l (dim:num_ecps)
        nucleus_index (list[int]):
            One-to-one correspondence between ECP items and the atom index (dim:num_ecps)
        exponents (list[float]):
            all ECP exponents (dim:num_ecps)
        coefficients (list[float]):
            all ECP coefficients (dim:num_ecps)
        powers (list[int]):
            all ECP powers (dim:num_ecps)

    Examples:
        NA

    Note:
        NA

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
        """Initialization of the class.

        This magic function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.

        Examples:
            NA

        Note:
            NA

        Todo:
            To be implemented.

        """
        pass

    @property
    def effective_charges(self) -> npt.NDArray:
        """effective_charges.

        Return nucleus charge (all-electron) or effective charge (with ECP)

        Return:
            npt.NDAarray: nucleus charge (effective charge)
        """
        if self.ecp_flag:
            return np.array(self.structure_data.atomic_numbers) - np.array(self.z_cores)
        else:
            return np.array(self.structure_data.atomic_numbers)

    @property
    def effective_charges_jnp(self) -> jax.Array:
        """effective_charges.

        Return nucleus charge (all-electron) or effective charge (with ECP)

        Return:
            jax.Array: nucleus charge (effective charge)
        """
        if self.ecp_flag:
            return jnp.array(self.structure_data.atomic_numbers) - jnp.array(self.z_cores)
        else:
            return jnp.array(self.structure_data.atomic_numbers)


def compute_ecp_coulomb_potential_api(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    Nv: int = 6,
) -> float:
    """
    The method is for computing the local and non-local parts of the given ECPs at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        Nv (int): The number of quadrature points for the spherical part.

    Returns
    -------
        The sum of local and non-local parts of the given ECPs with r_up_carts and r_dn_carts. (float)
    """

    V_ecp = compute_ecp_coulomb_potential_jax(
        coulomb_potential_data, wavefunction_data, r_up_carts, r_dn_carts, Nv
    )

    return V_ecp


def compute_ecp_local_parts_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """
    The method is for computing the local part of the given ECPs at (r_up_carts, r_dn_carts).
    A very straightforward (so very slow) implementation. Just for debudding purpose.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        Nv (int): The number of quadrature points for the spherical part.
        debug_flag: if True, the non-local part is computed in a very straightforward way for debuging purpose

    Returns
    -------
        The sum of local part of the given ECPs with r_up_carts and r_dn_carts. (float)
    """
    V_local = 0.0
    for i_atom in range(coulomb_potential_data.structure_data.natom):
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

        for r_up_cart in r_up_carts:
            rel_R_cart_min_dist = get_min_dist_rel_R_cart_np(
                structure_data=coulomb_potential_data.structure_data,
                r_cart=r_up_cart,
                i_atom=i_atom,
            )
            V_local += np.linalg.norm(rel_R_cart_min_dist) ** -2.0 * np.sum(
                [
                    a
                    * np.linalg.norm(rel_R_cart_min_dist) ** n
                    * np.exp(-b * np.linalg.norm(rel_R_cart_min_dist) ** 2)
                    for a, n, b in zip(coefficients, powers, exponents)
                ]
            )
        for r_dn_cart in r_dn_carts:
            rel_R_cart_min_dist = get_min_dist_rel_R_cart_np(
                structure_data=coulomb_potential_data.structure_data,
                r_cart=r_dn_cart,
                i_atom=i_atom,
            )
            V_local += np.linalg.norm(rel_R_cart_min_dist) ** -2.0 * np.sum(
                [
                    a
                    * np.linalg.norm(rel_R_cart_min_dist) ** n
                    * np.exp(-b * np.linalg.norm(rel_R_cart_min_dist) ** 2)
                    for a, n, b in zip(coefficients, powers, exponents)
                ]
            )
    return V_local


def compute_ecp_nonlocal_parts_debug(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    Nv: int = 6,
) -> float:
    """
    The method is for computing the non-local part of the given ECPs at (r_up_carts, r_dn_carts).
    A very straightforward (so super slow) implementation. Just for debudding purpose.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        Nv (int): The number of quadrature points for the spherical part.

    Returns:
        The sum of the non-local part of the given ECPs with r_up_carts and r_dn_carts. (float)
    """

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

    V_nonlocal = 0.0

    wf_denominator = evaluate_wavefunction_api(
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    for i_atom in range(coulomb_potential_data.structure_data.natom):
        max_ang_mom_plus_1 = coulomb_potential_data.max_ang_mom_plus_1[i_atom]
        nucleus_indices = [
            i for i, v in enumerate(coulomb_potential_data.nucleus_index) if v == i_atom
        ]

        ang_moms_all = [coulomb_potential_data.ang_moms[i] for i in nucleus_indices]
        exponents_all = [coulomb_potential_data.exponents[i] for i in nucleus_indices]
        coefficients_all = [coulomb_potential_data.coefficients[i] for i in nucleus_indices]
        powers_all = [coulomb_potential_data.powers[i] for i in nucleus_indices]

        for ang_mom in range(max_ang_mom_plus_1):
            ang_mom_indices = [i for i, v in enumerate(ang_moms_all) if v == ang_mom]
            exponents = [exponents_all[i] for i in ang_mom_indices]
            coefficients = [coefficients_all[i] for i in ang_mom_indices]
            powers = [powers_all[i] for i in ang_mom_indices]

            # up electrons
            for r_up_i, r_up_cart in enumerate(r_up_carts):
                rel_R_cart_min_dist = get_min_dist_rel_R_cart_np(
                    structure_data=coulomb_potential_data.structure_data,
                    r_cart=r_up_cart,
                    i_atom=i_atom,
                )
                V_l = np.linalg.norm(rel_R_cart_min_dist) ** -2.0 * np.sum(
                    [
                        a
                        * np.linalg.norm(rel_R_cart_min_dist) ** n
                        * np.exp(-b * np.linalg.norm(rel_R_cart_min_dist) ** 2.0)
                        for a, n, b in zip(coefficients, powers, exponents)
                    ]
                )

                for weight, vec_delta in zip(weights, grid_points):
                    r_up_carts_on_mesh = r_up_carts.copy()
                    r_up_carts_on_mesh[r_up_i] = (
                        r_up_cart
                        + rel_R_cart_min_dist
                        + np.linalg.norm(rel_R_cart_min_dist) * vec_delta
                    )

                    cos_theta = np.dot(
                        -1.0 * (rel_R_cart_min_dist) / np.linalg.norm(rel_R_cart_min_dist),
                        ((vec_delta) / np.linalg.norm(vec_delta)),
                    )

                    wf_numerator = evaluate_wavefunction_api(
                        wavefunction_data=wavefunction_data,
                        r_up_carts=r_up_carts_on_mesh,
                        r_dn_carts=r_dn_carts,
                    )

                    wf_ratio = wf_numerator / wf_denominator

                    P_l = (2 * ang_mom + 1) * eval_legendre(ang_mom, cos_theta) * weight * wf_ratio
                    V_nonlocal += V_l * P_l

            # dn electrons
            for r_dn_i, r_dn_cart in enumerate(r_dn_carts):
                rel_R_cart_min_dist = get_min_dist_rel_R_cart_np(
                    structure_data=coulomb_potential_data.structure_data,
                    r_cart=r_dn_cart,
                    i_atom=i_atom,
                )
                V_l = np.linalg.norm(rel_R_cart_min_dist) ** -2 * np.sum(
                    [
                        a
                        * np.linalg.norm(rel_R_cart_min_dist) ** n
                        * np.exp(-b * np.linalg.norm(rel_R_cart_min_dist) ** 2)
                        for a, n, b in zip(coefficients, powers, exponents)
                    ]
                )

                for weight, vec_delta in zip(weights, grid_points):
                    r_dn_carts_on_mesh = r_dn_carts.copy()
                    r_dn_carts_on_mesh[r_dn_i] = (
                        r_dn_cart
                        + rel_R_cart_min_dist
                        + np.linalg.norm(rel_R_cart_min_dist) * vec_delta
                    )

                    cos_theta = np.dot(
                        -1.0 * (rel_R_cart_min_dist) / np.linalg.norm(rel_R_cart_min_dist),
                        vec_delta / np.linalg.norm(vec_delta),
                    )

                    wf_numerator = evaluate_wavefunction_api(
                        wavefunction_data=wavefunction_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts_on_mesh,
                    )

                    wf_ratio = wf_numerator / wf_denominator

                    P_l = (2 * ang_mom + 1) * eval_legendre(ang_mom, cos_theta) * weight * wf_ratio
                    V_nonlocal += V_l * P_l

    return V_nonlocal


def compute_ecp_coulomb_potential_debug(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    Nv: int = 6,
) -> float:
    """
    The method is for computing the local and non-local parts of the given ECPs at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        Nv (int): The number of quadrature points for the spherical part.

    Returns:
        The sum of local and non-local parts of the given ECPs with r_up_carts and r_dn_carts. (float)
    """

    ecp_local_parts = compute_ecp_local_parts_debug(
        coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    ecp_nonlocal_parts = compute_ecp_nonlocal_parts_debug(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        Nv=Nv,
    )

    V_ecp = ecp_local_parts + ecp_nonlocal_parts

    return V_ecp


def compute_ecp_coulomb_potential_jax(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    Nv: int = 6,
) -> float:
    """
    The method is for computing the local and non-local parts of the given ECPs at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        Nv (int): The number of quadrature points for the spherical part.

    Returns:
        The sum of local and non-local parts of the given ECPs with r_up_carts and r_dn_carts. (float)
    """
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

    V_ecp = compute_ecp_coulomb_potential_jax_weights_grid_points(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        weights=weights,
        grid_points=grid_points,
    )

    return V_ecp


@jit
def compute_ecp_coulomb_potential_jax_weights_grid_points(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    weights: list,
    grid_points: npt.NDArray[np.float64],
) -> float:
    """
    The method is for computing the local and non-local parts of the given ECPs at (r_up_carts, r_dn_carts).
    To avoid for the nested loops, jax-vmap function (i.e. efficient vectrization for compilation) is fully
    exploitted in the method.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        weights: weights for numerical integration
        grid_points: grid_points for numerical integration

    Returns:
        The sum of local and non-local parts of the given ECPs with r_up_carts and r_dn_carts. (float)
    """

    weights = jnp.array(weights)
    grid_points = jnp.array(grid_points)

    wf_denominator = evaluate_wavefunction_api(
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # Compute the local part. To understand the flow, please refer to the debug version.
    # @jit
    def compute_V_l(r_cart, i_atom, exponent, coefficient, power):
        rel_R_cart_min_dist = get_min_dist_rel_R_cart_jnp(
            structure_data=coulomb_potential_data.structure_data,
            r_cart=r_cart,
            i_atom=i_atom,
        )
        V_l = (
            jnp.linalg.norm(rel_R_cart_min_dist) ** -2.0
            * coefficient
            * jnp.linalg.norm(rel_R_cart_min_dist) ** power
            * jnp.exp(-exponent * (jnp.linalg.norm(rel_R_cart_min_dist) ** 2))
        )

        return V_l

    # Compute the Projection of WF. for a up electron
    # To understand the flow, please refer to the debug version.
    # @jit
    def compute_P_l_up(ang_mom, r_up_i, r_up_cart, i_atom, weight, vec_delta):
        rel_R_cart_min_dist = get_min_dist_rel_R_cart_jnp(
            structure_data=coulomb_potential_data.structure_data,
            r_cart=r_up_cart,
            i_atom=i_atom,
        )
        r_up_carts_on_mesh = jnp.array(r_up_carts)
        r_up_carts_on_mesh = r_up_carts_on_mesh.at[r_up_i].set(
            r_up_cart + rel_R_cart_min_dist + jnp.linalg.norm(rel_R_cart_min_dist) * vec_delta
        )

        cos_theta_up = jnp.dot(
            -1.0 * (rel_R_cart_min_dist) / jnp.linalg.norm(rel_R_cart_min_dist),
            ((vec_delta) / jnp.linalg.norm(vec_delta)),
        )
        wf_numerator_up = evaluate_wavefunction_api(
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts_on_mesh,
            r_dn_carts=r_dn_carts,
        )

        wf_ratio_up = wf_numerator_up / wf_denominator

        P_l_up = (
            (2 * ang_mom + 1) * jnp_legendre_tablated(ang_mom, cos_theta_up) * weight * wf_ratio_up
        )

        return P_l_up

    # Compute the Projection of WF. for a down electron
    # To understand the flow, please refer to the debug version.
    # @jit
    def compute_P_l_dn(ang_mom, r_dn_i, r_dn_cart, i_atom, weight, vec_delta):
        rel_R_cart_min_dist = get_min_dist_rel_R_cart_jnp(
            structure_data=coulomb_potential_data.structure_data,
            r_cart=r_dn_cart,
            i_atom=i_atom,
        )
        r_dn_carts_on_mesh = jnp.array(r_dn_carts)
        r_dn_carts_on_mesh = r_dn_carts_on_mesh.at[r_dn_i].set(
            r_dn_cart + rel_R_cart_min_dist + jnp.linalg.norm(rel_R_cart_min_dist) * vec_delta
        )

        cos_theta_dn = jnp.dot(
            -1.0 * (rel_R_cart_min_dist) / jnp.linalg.norm(rel_R_cart_min_dist),
            ((vec_delta) / jnp.linalg.norm(vec_delta)),
        )
        wf_numerator_dn = evaluate_wavefunction_api(
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts_on_mesh,
        )

        wf_ratio_dn = wf_numerator_dn / wf_denominator

        P_l_dn = (
            (2 * ang_mom + 1) * jnp_legendre_tablated(ang_mom, cos_theta_dn) * weight * wf_ratio_dn
        )
        return P_l_dn

    # Vectrize the functions
    vmap_compute_P_l_up = vmap(compute_P_l_up, in_axes=(None, None, None, None, 0, 0))
    vmap_compute_P_l_dn = vmap(compute_P_l_dn, in_axes=(None, None, None, None, 0, 0))

    # Compute the local part V_l * Projection of WF. for a up electron
    # To understand the flow, please refer to the debug version.
    # @jit
    def compute_V_nonlocal_up(
        r_up_i,
        r_up_cart,
        ang_mom,
        i_atom,
        exponent,
        coefficient,
        power,
    ):
        V_l_up = compute_V_l(r_up_cart, i_atom, exponent, coefficient, power)
        P_l_up = jnp.sum(
            vmap_compute_P_l_up(ang_mom, r_up_i, r_up_cart, i_atom, weights, grid_points)
        )
        return V_l_up * P_l_up

    # Compute the local part V_l * Projection of WF. for a down electron
    # To understand the flow, please refer to the debug version.
    # @jit
    def compute_V_nonlocal_dn(
        r_dn_i,
        r_dn_cart,
        ang_mom,
        i_atom,
        exponent,
        coefficient,
        power,
    ):
        V_l_dn = compute_V_l(r_dn_cart, i_atom, exponent, coefficient, power)
        P_l_dn = jnp.sum(
            vmap_compute_P_l_dn(ang_mom, r_dn_i, r_dn_cart, i_atom, weights, grid_points)
        )
        return V_l_dn * P_l_dn

    # Compute the local part V_l for a up electron.
    # This is activate when the given ang_mom == max_ang_mom_plus_1
    # i.e. the projection is not needed for the highest angular momentum
    # To understand the flow, please refer to the debug version.
    # @jit
    def compute_V_local(
        r_i,
        r_cart,
        ang_mom,
        i_atom,
        exponent,
        coefficient,
        power,
    ):
        V_l = compute_V_l(r_cart, i_atom, exponent, coefficient, power)
        return V_l

    # Choose the local part V_l * Projection of WF. or only the local part
    # for a up electron, depending on max_ang_mom_plus_1 (by using jax.lax)
    # @jit
    def compute_ecp_up(
        r_up_i,
        r_up_cart,
        max_ang_mom_plus_1,
        ang_mom,
        i_atom,
        exponent,
        coefficient,
        power,
    ):
        return lax.cond(
            max_ang_mom_plus_1 != ang_mom,
            compute_V_nonlocal_up,
            compute_V_local,
            r_up_i,
            r_up_cart,
            ang_mom,
            i_atom,
            exponent,
            coefficient,
            power,
        )

    # Choose the local part V_l * Projection of WF. or only the local part
    # for a down electron, depending on max_ang_mom_plus_1 (by using jax.lax)
    # @jit
    def compute_ecp_dn(
        r_dn_i,
        r_dn_cart,
        max_ang_mom_plus_1,
        ang_mom,
        i_atom,
        exponent,
        coefficient,
        power,
    ):
        return lax.cond(
            max_ang_mom_plus_1 != ang_mom,
            compute_V_nonlocal_dn,
            compute_V_local,
            r_dn_i,
            r_dn_cart,
            ang_mom,
            i_atom,
            exponent,
            coefficient,
            power,
        )

    # vectrize compute_ecp_up and compute_ecp_dn
    vmap_vmap_compute_ecp_up = vmap(
        vmap(
            compute_ecp_up,
            in_axes=(0, 0, None, None, None, None, None, None),
        ),
        in_axes=(None, None, 0, 0, 0, 0, 0, 0),
    )
    vmap_vmap_compute_ecp_dn = vmap(
        vmap(
            compute_ecp_dn,
            in_axes=(0, 0, None, None, None, None, None, None),
        ),
        in_axes=(None, None, 0, 0, 0, 0, 0, 0),
    )

    # Vectrized (flatten) arguments are prepared here.
    r_up_i_jnp = jnp.arange(len(r_up_carts))
    r_up_carts_jnp = jnp.array(r_up_carts)
    r_dn_i_jnp = jnp.arange(len(r_dn_carts))
    r_dn_carts_jnp = jnp.array(r_dn_carts)
    max_ang_mom_plus_1 = jnp.array(coulomb_potential_data.max_ang_mom_plus_1)
    nucleus_index = jnp.array(coulomb_potential_data.nucleus_index, dtype=jnp.int32)
    max_ang_mom_plus_1_jnp = max_ang_mom_plus_1[nucleus_index]
    ang_mom_jnp = jnp.array(coulomb_potential_data.ang_moms)
    i_atom_jnp = jnp.array(coulomb_potential_data.nucleus_index)
    exponent_jnp = jnp.array(coulomb_potential_data.exponents)
    coefficient_jnp = jnp.array(coulomb_potential_data.coefficients)
    power_jnp = jnp.array(coulomb_potential_data.powers)

    V_ecp_up = jnp.sum(
        vmap_vmap_compute_ecp_up(
            r_up_i_jnp,
            r_up_carts_jnp,
            max_ang_mom_plus_1_jnp,
            ang_mom_jnp,
            i_atom_jnp,
            exponent_jnp,
            coefficient_jnp,
            power_jnp,
        )
    )

    V_ecp_dn = jnp.sum(
        vmap_vmap_compute_ecp_dn(
            r_dn_i_jnp,
            r_dn_carts_jnp,
            max_ang_mom_plus_1_jnp,
            ang_mom_jnp,
            i_atom_jnp,
            exponent_jnp,
            coefficient_jnp,
            power_jnp,
        )
    )

    V_ecp = V_ecp_up + V_ecp_dn

    return V_ecp


def compute_bare_coulomb_potential_api(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """
    The method is for computing the bare coulomb potentials including all electron-electron,
    electron-ion (except. ECPs), and ion-ion interactions at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        debug_flag: if True, the non-local part is computed in a very straightforward way for debuging purpose

    Returns
    -------
        The bare Coulomb potential with r_up_carts and r_dn_carts. (float)
    """

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
    R_charges = coulomb_potential_data.effective_charges
    r_up_charges = [-1 for _ in range(len(r_up_carts))]
    r_dn_charges = [-1 for _ in range(len(r_dn_carts))]

    all_carts = np.vstack([R_carts, r_up_carts, r_dn_carts])
    all_charges = np.hstack([R_charges, r_up_charges, r_dn_charges])

    bare_coulomb_potential = np.sum(
        [
            (Z_a * Z_b) / np.linalg.norm(r_a - r_b)
            for (Z_a, r_a), (Z_b, r_b) in itertools.combinations(zip(all_charges, all_carts), 2)
        ]
    )

    return bare_coulomb_potential


@jit
def compute_bare_coulomb_potential_jax(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """bare for loop, old
    R_carts = coulomb_potential_data.structure_data.positions_cart
    R_charges = coulomb_potential_data.effective_charges_jnp
    r_up_charges = jnp.array([-1 for _ in range(len(r_up_carts))])
    r_dn_charges = jnp.array([-1 for _ in range(len(r_dn_carts))])

    all_carts = jnp.vstack([R_carts, r_up_carts, r_dn_carts])
    all_charges = jnp.hstack([R_charges, r_up_charges, r_dn_charges])

    bare_coulomb_potential = jnp.sum(
        jnp.array(
            [
                (Z_a * Z_b) / jnp.linalg.norm(r_a - r_b)
                for (Z_a, r_a), (Z_b, r_b) in itertools.combinations(zip(all_charges, all_carts), 2)
            ]
        )
    )

    return bare_coulomb_potential
    """

    # """
    R_carts = coulomb_potential_data.structure_data.positions_cart
    R_charges = coulomb_potential_data.effective_charges_jnp
    r_up_charges = jnp.full(len(r_up_carts), -1.0, dtype=jnp.float64)
    r_dn_charges = jnp.full(len(r_dn_carts), -1.0, dtype=jnp.float64)

    all_carts = jnp.vstack([R_carts, r_up_carts, r_dn_carts])
    all_charges = jnp.hstack([R_charges, r_up_charges, r_dn_charges])

    # Number of particles
    N = all_charges.shape[0]

    # Generate all unique pairs indices (i < j)
    idx_i, idx_j = jnp.triu_indices(N, k=1)

    # Extract charges and positions for each pair
    Z_i = all_charges[idx_i]  # Shape: (M,)
    Z_j = all_charges[idx_j]  # Shape: (M,)
    r_i = all_carts[idx_i]  # Shape: (M, D)
    r_j = all_carts[idx_j]  # Shape: (M, D)

    # Define a function to compute interaction for a pair
    def pair_interaction(Z_i, Z_j, r_i, r_j):
        distance = jnp.linalg.norm(r_i - r_j) + 1e-12  # Add epsilon to avoid division by zero
        interaction = (Z_i * Z_j) / distance
        return interaction

    # Vectorize the function over all pairs
    interactions = jax.vmap(pair_interaction)(Z_i, Z_j, r_i, r_j)  # Shape: (M,)

    # Sum all interactions
    bare_coulomb_potential = jnp.sum(interactions)

    return bare_coulomb_potential

    # """


def compute_coulomb_potential_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    wavefunction_data: Wavefunction_data = None,
) -> float:
    """
    The method is for computing the bare coulomb potentials including all electron-electron,
    electron-ion (inc. ECPs), and ion-ion interactions at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        wavefunction_data (Wavefunction_data): Wavefunction information needed to compute the non-local part

    Returns
    -------
        Potential Energy at r_up_carts and r_dn_carts. (float)
    """
    # all-electron
    if not coulomb_potential_data.ecp_flag:
        bare_coulomb_potential = compute_bare_coulomb_potential_debug(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        ecp_coulomb_potential = 0

    # pseudo-potential
    else:
        bare_coulomb_potential = compute_bare_coulomb_potential_debug(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        ecp_coulomb_potential = compute_ecp_coulomb_potential_debug(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            wavefunction_data=wavefunction_data,
            Nv=6,
        )

    return bare_coulomb_potential + ecp_coulomb_potential


def compute_coulomb_potential_api(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    wavefunction_data: Wavefunction_data = None,
) -> float:
    """
    The method is for computing the bare coulomb potentials including all electron-electron,
    electron-ion (inc. ECPs), and ion-ion interactions at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        wavefunction_data (Wavefunction_data): Wavefunction information needed to compute the non-local part

    Returns
    -------
        Potential Energy at r_up_carts and r_dn_carts. (float)
    """
    # all-electron
    if not coulomb_potential_data.ecp_flag:
        bare_coulomb_potential = compute_bare_coulomb_potential_api(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        ecp_coulomb_potential = 0

    # pseudo-potential
    else:
        bare_coulomb_potential = compute_bare_coulomb_potential_api(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        ecp_coulomb_potential = compute_ecp_coulomb_potential_api(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            wavefunction_data=wavefunction_data,
            Nv=6,
        )

    return bare_coulomb_potential + ecp_coulomb_potential


if __name__ == "__main__":
    import os

    from .hamiltonians import Hamiltonian_data
    from .trexio_wrapper import read_trexio_file

    log = getLogger("jqmc")
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

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=None,
        jastrow_two_body_pade_flag="off",
        jastrow_three_body_data=None,
        jastrow_three_body_flag="off",
    )  # no jastrow for the time-being.

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    """
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
    """

    old_r_up_carts = np.array(
        [
            [0.64878536, -0.83275288, 0.33532629],
            [0.55271273, 0.72310605, 0.93443775],
            [0.66767275, 0.1206456, -0.36521208],
            [-0.93165236, -0.0120386, 0.33003036],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [1.0347816, 1.26162081, 0.42301735],
            [-0.57843435, 1.03651987, -0.55091542],
            [-1.56091964, -0.58952149, -0.99268141],
            [0.61863233, -0.14903326, 0.51962683],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_dn_carts[3] = [0.618632327645002, -0.149033260668010, 0.131889254514777]

    weights = octahedron_sym_mesh_Nv6.weights
    grid_points = octahedron_sym_mesh_Nv6.grid_points

    vpot_bare_jax = compute_bare_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    logger.debug(f"vpot_bare_jax = {vpot_bare_jax}")
    logger.debug(f"vpot_bare_debug = {vpot_bare_debug}")
    np.testing.assert_almost_equal(vpot_bare_jax, vpot_bare_debug, decimal=10)

    vpot_ecp_jax = compute_ecp_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    vpot_ecp_debug = compute_ecp_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    logger.debug(f"vpot_ecp_jax = {vpot_ecp_jax}")
    logger.debug(f"vpot_ecp_debug = {vpot_ecp_debug}")
    np.testing.assert_almost_equal(vpot_ecp_jax, vpot_ecp_debug, decimal=10)
