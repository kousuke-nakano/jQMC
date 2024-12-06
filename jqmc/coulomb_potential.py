"""Effective core potential module.

Module containing classes and methods related to Effective core potential
and bare Coulomb potentials

Todo:
    * Replace numpy and jax.numpy typings with jaxtyping
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
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
from scipy.special import eval_legendre

from .miscs.function_collections import legendre_tablated as jnp_legendre_tablated
from .structure import Structure_data, get_min_dist_rel_R_cart_jnp, get_min_dist_rel_R_cart_np
from .wavefunction import Wavefunction_data, evaluate_determinant_api, evaluate_wavefunction_api

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


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
        structure_data (Structure_data):
            Instance of a structure_data

    """

    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    ecp_flag: bool = struct.field(pytree_node=False, default=False)
    z_cores: list[float] = struct.field(pytree_node=False, default_factory=list)
    max_ang_mom_plus_1: list[int] = struct.field(pytree_node=False, default_factory=list)
    num_ecps: list[int] = struct.field(pytree_node=False, default_factory=list)
    ang_moms: list[int] = struct.field(pytree_node=False, default_factory=list)
    nucleus_index: list[int] = struct.field(pytree_node=False, default_factory=list)
    exponents: list[float] = struct.field(pytree_node=False, default_factory=list)
    coefficients: list[float] = struct.field(pytree_node=False, default_factory=list)
    powers: list[int] = struct.field(pytree_node=False, default_factory=list)

    def __post_init__(self) -> None:
        """Initialization of the class.

        This magic function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.

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
    def ang_mom_local_part(self) -> npt.NDArray:
        """ang_mom_local_part.

        Return angular momentum of the local part (i.e., = max_ang_mom_plus1)

        Return:
            npt.NDAarray: momentum of the local part (effective charge)
        """
        return np.array(self.max_ang_mom_plus_1)

    @property
    def ang_mom_non_local_part(self) -> npt.NDArray:
        """ang_mom_non_local_part.

        Return angular momentum of the non_local part (i.e., = max_ang_mom_plus1)

        Return:
            npt.NDAarray: momentum of the non_local part (effective charge)
        """
        return np.array(self.ang_moms[self.non_local_part_index])

    @property
    def local_part_index(self) -> npt.NDArray:
        """local_part_index.

        Return a list containing index of the local part

        Return:
            npt.NDAarray: a list containing index of the local part
        """
        local_part_index = np.array(
            [
                i
                for i, v in enumerate(self.nucleus_index)
                if v in range(self.structure_data.natom) and self.ang_moms[i] == self.max_ang_mom_plus_1[v]
            ]
        )
        return local_part_index

    @property
    def non_local_part_index(self) -> npt.NDArray:
        """non_local_part_index.

        Return a list containing index of the non-local part

        Return:
            npt.NDAarray: a list containing index of the non-local part
        """
        non_local_part_index = np.array(
            [
                i
                for i, v in enumerate(self.nucleus_index)
                if v in range(self.structure_data.natom) and self.ang_moms[i] != self.max_ang_mom_plus_1[v]
            ]
        )
        return non_local_part_index

    @property
    def nucleus_index_local_part(self) -> npt.NDArray:
        """nucleus_index local_part.

        Return a list containing nucleus_index of the local part

        Return:
            npt.NDAarray: a list containing nucleus_index of the local part
        """
        return np.array(self.nucleus_index)[self.local_part_index]

    @property
    def nucleus_index_non_local_part(self) -> npt.NDArray:
        """nucleus_index non_local_part.

        Return a list containing nucleus_index of the non-local part

        Return:
            npt.NDAarray: a list containing nucleus_index of the non-local part
        """
        return np.array(self.nucleus_index)[self.non_local_part_index]

    @property
    def exponents_local_part(self) -> npt.NDArray:
        """Exponents local_part.

        Return a list containing exponents of the local part

        Return:
            npt.NDAarray: a list containing exponents of the local part
        """
        return np.array(self.exponents)[self.local_part_index]

    @property
    def exponents_non_local_part(self) -> npt.NDArray:
        """Exponents non_local_part.

        Return a list containing exponents of the non-local part

        Return:
            npt.NDAarray: a list containing exponents of the non-local part
        """
        return np.array(self.exponents)[self.non_local_part_index]

    @property
    def coefficients_local_part(self) -> npt.NDArray:
        """Coefficients local_part.

        Return a list containing coefficients of the local part

        Return:
            npt.NDAarray: a list containing coefficients of the local part
        """
        return np.array(self.coefficients)[self.local_part_index]

    @property
    def coefficients_non_local_part(self) -> npt.NDArray:
        """Coefficients non_local_part.

        Return a list containing coefficients of the non-local part

        Return:
            npt.NDAarray: a list containing coefficients of the non-local part
        """
        return np.array(self.coefficients)[self.non_local_part_index]

    @property
    def powers_local_part(self) -> npt.NDArray:
        """Powers local_part.

        Return a list containing powers of the local part

        Return:
            npt.NDAarray: a list containing powers of the local part
        """
        return np.array(self.powers)[self.local_part_index]

    @property
    def powers_non_local_part(self) -> npt.NDArray:
        """Powers non_local_part.

        Return a list containing powers of the non-local part

        Return:
            npt.NDAarray: a list containing powers of the non-local part
        """
        return np.array(self.powers)[self.non_local_part_index]


def compute_ecp_coulomb_potential_api(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    Nv: int = 6,
    debug: bool = False,
) -> float:
    """Compute effective core potential term.

    The method is for computing the local and non-local parts of the given ECPs at
    a given electronic configuration (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        Nv (int): The number of quadrature points for the spherical part.
        debug (bool): if True, the value is computed via _debug function for debuging purpose

    Returns:
        float: The sum of local and non-local parts of the given ECPs with r_up_carts and r_dn_carts. (float)
    """
    if debug:
        V_ecp = _compute_ecp_coulomb_potential_debug(coulomb_potential_data, wavefunction_data, r_up_carts, r_dn_carts, Nv)
    else:
        V_ecp = _compute_ecp_coulomb_potential_jax(coulomb_potential_data, wavefunction_data, r_up_carts, r_dn_carts, Nv)

    return V_ecp


def _compute_ecp_local_parts_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """Compute ecp local parts.

    The method is for computing the local part of the given ECPs at (r_up_carts, r_dn_carts).
    A very straightforward (so very slow) implementation. Just for debudding purpose.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        Nv (int): The number of quadrature points for the spherical part.

    Returns:
        float: The sum of local part of the given ECPs with r_up_carts and r_dn_carts.
    """
    V_local = 0.0
    for i_atom in range(coulomb_potential_data.structure_data.natom):
        max_ang_mom_plus_1 = coulomb_potential_data.max_ang_mom_plus_1[i_atom]
        nucleus_indices = [i for i, v in enumerate(coulomb_potential_data.nucleus_index) if v == i_atom]
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
                    a * np.linalg.norm(rel_R_cart_min_dist) ** n * np.exp(-b * np.linalg.norm(rel_R_cart_min_dist) ** 2)
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
                    a * np.linalg.norm(rel_R_cart_min_dist) ** n * np.exp(-b * np.linalg.norm(rel_R_cart_min_dist) ** 2)
                    for a, n, b in zip(coefficients, powers, exponents)
                ]
            )
    return V_local


def _compute_ecp_non_local_parts_debug(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    Nv: int = 6,
    flag_determinant_only: bool = False,
) -> float:
    """Compute ecp non-local parts.

    The method is for computing the non-local part of the given ECPs at (r_up_carts, r_dn_carts).
    A very straightforward (so very slow) implementation. Just for debudding purpose.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        Nv (int): The number of quadrature points for the spherical part.
        flag_determinant_only (bool): If True, only the determinant part is considered for the non-local ECP part.

    Returns:
        list[(np.NDArray, np.NDArray)]: The list of grids on which the non-local part is computed.
        list[float]: The list of non-local part of the given ECPs with r_up_carts and r_dn_carts.
        float: sum of the V_nonlocal
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

    mesh_non_local_ecp_part = []
    V_nonlocal = []
    sum_V_nonlocal = 0.0

    if flag_determinant_only:
        wf_denominator = evaluate_determinant_api(
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
    else:
        wf_denominator = evaluate_wavefunction_api(
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

    for i_atom in range(coulomb_potential_data.structure_data.natom):
        max_ang_mom_plus_1 = coulomb_potential_data.max_ang_mom_plus_1[i_atom]
        nucleus_indices = [i for i, v in enumerate(coulomb_potential_data.nucleus_index) if v == i_atom]

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
                        a * np.linalg.norm(rel_R_cart_min_dist) ** n * np.exp(-b * np.linalg.norm(rel_R_cart_min_dist) ** 2.0)
                        for a, n, b in zip(coefficients, powers, exponents)
                    ]
                )

                for weight, vec_delta in zip(weights, grid_points):
                    r_up_carts_on_mesh = r_up_carts.copy()
                    r_up_carts_on_mesh[r_up_i] = (
                        r_up_cart + rel_R_cart_min_dist + np.linalg.norm(rel_R_cart_min_dist) * vec_delta
                    )

                    cos_theta = np.dot(
                        -1.0 * (rel_R_cart_min_dist) / np.linalg.norm(rel_R_cart_min_dist),
                        ((vec_delta) / np.linalg.norm(vec_delta)),
                    )

                    if flag_determinant_only:
                        wf_numerator = evaluate_determinant_api(
                            wavefunction_data=wavefunction_data,
                            r_up_carts=r_up_carts_on_mesh,
                            r_dn_carts=r_dn_carts,
                        )
                    else:
                        wf_numerator = evaluate_wavefunction_api(
                            wavefunction_data=wavefunction_data,
                            r_up_carts=r_up_carts_on_mesh,
                            r_dn_carts=r_dn_carts,
                        )
                    wf_ratio = wf_numerator / wf_denominator

                    P_l = (2 * ang_mom + 1) * eval_legendre(ang_mom, cos_theta) * weight * wf_ratio

                    mesh_non_local_ecp_part.append((r_up_carts_on_mesh, r_dn_carts))
                    V_nonlocal.append(V_l * P_l)
                    sum_V_nonlocal += V_l * P_l

            # dn electrons
            for r_dn_i, r_dn_cart in enumerate(r_dn_carts):
                rel_R_cart_min_dist = get_min_dist_rel_R_cart_np(
                    structure_data=coulomb_potential_data.structure_data,
                    r_cart=r_dn_cart,
                    i_atom=i_atom,
                )
                V_l = np.linalg.norm(rel_R_cart_min_dist) ** -2 * np.sum(
                    [
                        a * np.linalg.norm(rel_R_cart_min_dist) ** n * np.exp(-b * np.linalg.norm(rel_R_cart_min_dist) ** 2)
                        for a, n, b in zip(coefficients, powers, exponents)
                    ]
                )

                for weight, vec_delta in zip(weights, grid_points):
                    r_dn_carts_on_mesh = r_dn_carts.copy()
                    r_dn_carts_on_mesh[r_dn_i] = (
                        r_dn_cart + rel_R_cart_min_dist + np.linalg.norm(rel_R_cart_min_dist) * vec_delta
                    )

                    cos_theta = np.dot(
                        -1.0 * (rel_R_cart_min_dist) / np.linalg.norm(rel_R_cart_min_dist),
                        vec_delta / np.linalg.norm(vec_delta),
                    )

                    if flag_determinant_only:
                        wf_numerator = evaluate_determinant_api(
                            wavefunction_data=wavefunction_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts_on_mesh,
                        )
                    else:
                        wf_numerator = evaluate_wavefunction_api(
                            wavefunction_data=wavefunction_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts_on_mesh,
                        )

                    wf_ratio = wf_numerator / wf_denominator

                    P_l = (2 * ang_mom + 1) * eval_legendre(ang_mom, cos_theta) * weight * wf_ratio
                    mesh_non_local_ecp_part.append((r_up_carts, r_dn_carts_on_mesh))
                    V_nonlocal.append(V_l * P_l)
                    sum_V_nonlocal += V_l * P_l

    mesh_non_local_ecp_part = list(mesh_non_local_ecp_part)
    V_nonlocal = list(V_nonlocal)

    return mesh_non_local_ecp_part, V_nonlocal, sum_V_nonlocal


def _compute_ecp_coulomb_potential_debug(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    Nv: int = 6,
) -> float:
    """Compute ecp local and non-local parts.

    The method is for computing the local and non-local part of the given ECPs at (r_up_carts, r_dn_carts).
    A very straightforward (so very slow) implementation. Just for debudding purpose.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        Nv (int): The number of quadrature points for the spherical part.

    Returns:
        float: The sum of non-local part of the given ECPs with r_up_carts and r_dn_carts.
    """
    ecp_local_parts = _compute_ecp_local_parts_debug(
        coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    _, _, ecp_nonlocal_parts = _compute_ecp_non_local_parts_debug(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        Nv=Nv,
    )

    V_ecp = ecp_local_parts + ecp_nonlocal_parts

    return V_ecp


@jit
def _compute_ecp_local_parts_jax(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """Compute ecp local parts.

    The method is for computing the local part of the given ECPs at (r_up_carts, r_dn_carts).
    A much faster implementation using JAX.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        float: The sum of local part of the given ECPs with r_up_carts and r_dn_carts.
    """

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

    # Compute the local part V_l for a up electron.
    # This is activate when the given ang_mom == max_ang_mom_plus_1
    # i.e. the projection is not needed for the highest angular momentum
    # To understand the flow, please refer to the debug version.
    # @jit
    def compute_V_local(
        r_cart,
        i_atom,
        exponent,
        coefficient,
        power,
    ):
        V_l = compute_V_l(r_cart, i_atom, exponent, coefficient, power)
        return V_l

    # vectrize compute_ecp_up and compute_ecp_dn
    vmap_vmap_compute_ecp_up = vmap(
        vmap(
            compute_V_local,
            in_axes=(0, None, None, None, None),
        ),
        in_axes=(None, 0, 0, 0, 0),
    )
    vmap_vmap_compute_ecp_dn = vmap(
        vmap(
            compute_V_local,
            in_axes=(0, None, None, None, None),
        ),
        in_axes=(None, 0, 0, 0, 0),
    )

    # Vectrized (flatten) arguments are prepared here.
    r_up_carts_jnp = jnp.array(r_up_carts)
    r_dn_carts_jnp = jnp.array(r_dn_carts)

    i_atom_np = np.array(coulomb_potential_data.nucleus_index_local_part)
    exponent_np = np.array(coulomb_potential_data.exponents_local_part)
    coefficient_np = np.array(coulomb_potential_data.coefficients_local_part)
    power_np = np.array(coulomb_potential_data.powers_local_part)

    V_ecp_up = jnp.sum(
        vmap_vmap_compute_ecp_up(
            r_up_carts_jnp,
            i_atom_np,
            exponent_np,
            coefficient_np,
            power_np,
        )
    )

    V_ecp_dn = jnp.sum(
        vmap_vmap_compute_ecp_dn(
            r_dn_carts_jnp,
            i_atom_np,
            exponent_np,
            coefficient_np,
            power_np,
        )
    )

    V_ecp = V_ecp_up + V_ecp_dn

    return V_ecp


def _compute_ecp_non_local_parts_jax(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    Nv: int = 6,
    flag_determinant_only: bool = False,
) -> float:
    """Compute ecp non-local parts using JAX.

    The method is for computing the non-local part of the given ECPs at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        Nv (int): The number of quadrature points for the spherical part.
        flag_determinant_only (bool): If True, only the determinant part is considered for the non-local ECP part.

    Returns:
        list[(np.NDArray, np.NDArray)]: The list of grids on which the non-local part is computed.
        list[float]: The list of non-local part of the given ECPs with r_up_carts and r_dn_carts.
        float: The sum of non-local part of the given ECPs with r_up_carts and r_dn_carts.
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

    r_up_carts_on_mesh, r_dn_carts_on_mesh, V_ecp_up, V_ecp_dn, sum_V_nonlocal = (
        _compute_ecp_non_local_part_jax_weights_grid_points(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            weights=weights,
            grid_points=grid_points,
            flag_determinant_only=int(flag_determinant_only),
        )
    )

    # print(f"r_up_carts_on_mesh.shape={r_up_carts_on_mesh.shape}")
    # print(f"r_dn_carts_on_mesh.shape={r_dn_carts_on_mesh.shape}")
    # print(f"V_ecp_up.shape={V_ecp_up.shape}")
    # print(f"V_ecp_dn.shape={V_ecp_dn.shape}")

    _, uq_indices = np.unique(coulomb_potential_data.nucleus_index_non_local_part, return_index=True)
    r_up_carts_on_mesh = jnp.array([r_up_carts_on_mesh[i] for i in uq_indices])
    r_dn_carts_on_mesh = jnp.array([r_dn_carts_on_mesh[i] for i in uq_indices])

    nucleus_index_non_local_part = np.array(coulomb_potential_data.nucleus_index_non_local_part, dtype=np.int32)
    num_segments = len(set(coulomb_potential_data.nucleus_index_non_local_part))
    V_ecp_up = jax.ops.segment_sum(V_ecp_up, nucleus_index_non_local_part, num_segments=num_segments)
    V_ecp_dn = jax.ops.segment_sum(V_ecp_dn, nucleus_index_non_local_part, num_segments=num_segments)

    # print(f"r_up_carts_on_mesh.shape={r_up_carts_on_mesh.shape}")
    # print(f"r_dn_carts_on_mesh.shape={r_dn_carts_on_mesh.shape}")
    # print(f"V_ecp_up.shape={V_ecp_up.shape}")
    # print(f"V_ecp_dn.shape={V_ecp_dn.shape}")

    r_up_new_shape = (np.prod(r_up_carts_on_mesh.shape[:3]),) + r_up_carts_on_mesh.shape[3:]
    r_up_carts_on_mesh = r_up_carts_on_mesh.reshape(r_up_new_shape)
    r_dn_new_shape = (np.prod(r_dn_carts_on_mesh.shape[:3]),) + r_dn_carts_on_mesh.shape[3:]
    r_dn_carts_on_mesh = r_dn_carts_on_mesh.reshape(r_dn_new_shape)

    V_ecp_up_new_shape = (np.prod(V_ecp_up.shape[:3]),)
    V_ecp_up = V_ecp_up.reshape(V_ecp_up_new_shape)
    V_ecp_dn_new_shape = (np.prod(V_ecp_dn.shape[:3]),)
    V_ecp_dn = V_ecp_dn.reshape(V_ecp_dn_new_shape)

    # print(f"r_up_carts_on_mesh.shape={r_up_carts_on_mesh.shape}")
    # print(f"r_dn_carts_on_mesh.shape={r_dn_carts_on_mesh.shape}")
    # print(f"V_ecp_up.shape={V_ecp_up.shape}")
    # print(f"V_ecp_dn.shape={V_ecp_dn.shape}")

    mesh_non_local_ecp_part = [(r_up_carts_m, r_dn_carts) for r_up_carts_m in r_up_carts_on_mesh] + [
        (r_up_carts, r_dn_carts_m) for r_dn_carts_m in r_dn_carts_on_mesh
    ]

    V_nonlocal = list(V_ecp_up) + list(V_ecp_dn)

    return mesh_non_local_ecp_part, V_nonlocal, sum_V_nonlocal


@jit  # this jit drastically accelarates the computation!
def _compute_ecp_non_local_part_jax_weights_grid_points(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    weights: list,
    grid_points: npt.NDArray[np.float64],
    flag_determinant_only: int = 0,
) -> float:
    """Compute ecp non-local parts using JAX.

    The method is for computing the non-local parts of the given ECPs at (r_up_carts, r_dn_carts).
    To avoid for the nested loops, jax-vmap function (i.e. efficient vectrization for compilation) is fully
    exploitted in the method.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        weights (list[np.float]): weights for numerical integration
        grid_points (npt.NDArray[np.float64]): grid_points for numerical integration
        flag_determinant_only (int):
            If True (i.e., 1), only the determinant part is considered for the non-local ECP part.
            If False (i.e., 0), both the Jastrow and determinant part is considered for the non-local ECP part.

    Returns:
        npt.NDArray: grid points used for the up electron
        npt.NDArray: grid points used for the dn electron
        npt.NDArray: V_ecp_up for the grid points for up electron
        npt.NDArray: V_ecp_dn for the grid points for up electron
        float: The sum of non-local part of the given ECPs with r_up_carts and r_dn_carts.

    Notes:
        This part of @jit drastically accelarates the computation!
        The procesure is so complicated that one should refer to the debug version.
        to understand the flow
    """
    weights = jnp.array(weights)
    grid_points = jnp.array(grid_points)

    """
    wf_denominator = evaluate_wavefunction_api(
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    """
    wf_denominator = lax.switch(
        flag_determinant_only,
        (evaluate_wavefunction_api, evaluate_determinant_api),
        *(wavefunction_data, r_up_carts, r_dn_carts),
    )

    # Compute the local part. To understand the flow, please refer to the debug version.
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

        """
        wf_numerator_up = evaluate_wavefunction_api(
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts_on_mesh,
            r_dn_carts=r_dn_carts,
        )
        """

        wf_numerator_up = lax.switch(
            flag_determinant_only,
            (evaluate_wavefunction_api, evaluate_determinant_api),
            *(wavefunction_data, r_up_carts_on_mesh, r_dn_carts),
        )

        wf_ratio_up = wf_numerator_up / wf_denominator

        P_l_up = (2 * ang_mom + 1) * jnp_legendre_tablated(ang_mom, cos_theta_up) * weight * wf_ratio_up

        return r_up_carts_on_mesh, P_l_up

    # Compute the Projection of WF. for a down electron
    # To understand the flow, please refer to the debug version.
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
        """
        wf_numerator_dn = evaluate_wavefunction_api(
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts_on_mesh,
        )
        """
        wf_numerator_dn = lax.switch(
            flag_determinant_only,
            (evaluate_wavefunction_api, evaluate_determinant_api),
            *(wavefunction_data, r_up_carts, r_dn_carts_on_mesh),
        )

        wf_ratio_dn = wf_numerator_dn / wf_denominator

        P_l_dn = (2 * ang_mom + 1) * jnp_legendre_tablated(ang_mom, cos_theta_dn) * weight * wf_ratio_dn
        return r_dn_carts_on_mesh, P_l_dn

    # Vectrize the functions
    vmap_compute_P_l_up = vmap(compute_P_l_up, in_axes=(None, None, None, None, 0, 0))
    vmap_compute_P_l_dn = vmap(compute_P_l_dn, in_axes=(None, None, None, None, 0, 0))

    # Compute the local part V_l * Projection of WF. for a up electron
    # To understand the flow, please refer to the debug version.
    # @jit
    # vmap in_axes=(0, 0, None, None, None, None, None) and in_axes=(None, None, 0, 0, 0, 0, 0)
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
        r_up_carts_on_mesh, P_l_up = vmap_compute_P_l_up(ang_mom, r_up_i, r_up_cart, i_atom, weights, grid_points)
        return r_up_carts_on_mesh, (V_l_up * P_l_up)

    # Compute the local part V_l * Projection of WF. for a down electron
    # To understand the flow, please refer to the debug version.
    # @jit
    # vmap in_axes=(0, 0, None, None, None, None, None) and in_axes=(None, None, 0, 0, 0, 0, 0)
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
        r_dn_carts_on_mesh, P_l_dn = vmap_compute_P_l_dn(ang_mom, r_dn_i, r_dn_cart, i_atom, weights, grid_points)
        return r_dn_carts_on_mesh, (V_l_dn * P_l_dn)

    # vectrize compute_ecp_up and compute_ecp_dn
    vmap_vmap_compute_ecp_up = vmap(
        vmap(compute_V_nonlocal_up, in_axes=(0, 0, None, None, None, None, None)), in_axes=(None, None, 0, 0, 0, 0, 0)
    )
    vmap_vmap_compute_ecp_dn = vmap(
        vmap(compute_V_nonlocal_dn, in_axes=(0, 0, None, None, None, None, None)), in_axes=(None, None, 0, 0, 0, 0, 0)
    )

    # Vectrized (flatten) arguments are prepared here.
    r_up_i_jnp = jnp.arange(len(r_up_carts))
    r_up_carts_jnp = jnp.array(r_up_carts)
    r_dn_i_jnp = jnp.arange(len(r_dn_carts))
    r_dn_carts_jnp = jnp.array(r_dn_carts)

    i_atom_np = np.array(coulomb_potential_data.nucleus_index_non_local_part)
    ang_mom_np = np.array(coulomb_potential_data.ang_mom_non_local_part)
    exponent_np = np.array(coulomb_potential_data.exponents_non_local_part)
    coefficient_np = np.array(coulomb_potential_data.coefficients_non_local_part)
    power_np = np.array(coulomb_potential_data.powers_non_local_part)

    r_up_carts_on_mesh, V_ecp_up = vmap_vmap_compute_ecp_up(
        r_up_i_jnp,
        r_up_carts_jnp,
        ang_mom_np,
        i_atom_np,
        exponent_np,
        coefficient_np,
        power_np,
    )

    r_dn_carts_on_mesh, V_ecp_dn = vmap_vmap_compute_ecp_dn(
        r_dn_i_jnp,
        r_dn_carts_jnp,
        ang_mom_np,
        i_atom_np,
        exponent_np,
        coefficient_np,
        power_np,
    )

    sum_V_nonlocal = jnp.sum(V_ecp_up) + jnp.sum(V_ecp_dn)

    return r_up_carts_on_mesh, r_dn_carts_on_mesh, V_ecp_up, V_ecp_dn, sum_V_nonlocal


def _compute_ecp_coulomb_potential_jax(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    Nv: int = 6,
) -> float:
    """Compute ecp local and non-local parts.

    The method is for computing the local and non-local part of the given ECPs at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        Nv (int): The number of quadrature points for the spherical part.

    Returns:
        float: The sum of local and non-local part of the given ECPs with r_up_carts and r_dn_carts.
    """
    ecp_local_parts = _compute_ecp_local_parts_jax(
        coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    _, _, ecp_nonlocal_parts = _compute_ecp_non_local_parts_jax(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        Nv=Nv,
    )

    V_ecp = ecp_local_parts + ecp_nonlocal_parts

    return V_ecp


def compute_bare_coulomb_potential_api(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    debug: bool = False,
) -> float:
    """Compute bare Coulomb potential.

    The method is for computing the bare coulomb potentials including both electron-electron,
    electron-ion (except. ECPs), and ion-ion interactions at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        debug (bool): if True, the value is computed via _debug function for debuging purpose

    Returns:
        The bare Coulomb potential with r_up_carts and r_dn_carts. (float)
    """
    if debug:
        bare_coulomb_potential = _compute_bare_coulomb_potential_debug(coulomb_potential_data, r_up_carts, r_dn_carts)
    else:
        bare_coulomb_potential = _compute_bare_coulomb_potential_jax(coulomb_potential_data, r_up_carts, r_dn_carts)

    return bare_coulomb_potential


def _compute_bare_coulomb_potential_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """See compute_bare_coulomb_potential_api."""
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


# it cannot be jitted?
# There is a related issue on github.
# ValueError when re-compiling function with a multi-dimensional array as a static field #24204
# For the time being, we can unjit it to avoid errors in unit_test.py
# This error is tied with the choice of pytree=True/False flag
@jit
def _compute_bare_coulomb_potential_jax(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """See compute_bare_coulomb_potential_api."""
    R_carts = jnp.array(coulomb_potential_data.structure_data.positions_cart)
    R_charges = np.array(coulomb_potential_data.effective_charges)
    r_up_charges = np.full(len(r_up_carts), -1.0, dtype=np.float64)
    r_dn_charges = np.full(len(r_dn_carts), -1.0, dtype=np.float64)

    r_up_carts = jnp.array(r_up_carts)
    r_dn_carts = jnp.array(r_dn_carts)

    all_charges = np.hstack([R_charges, r_up_charges, r_dn_charges])
    all_carts = jnp.vstack([R_carts, r_up_carts, r_dn_carts])

    # Number of particles
    N_np = all_charges.shape[0]
    N_jnp = all_carts.shape[0]

    # Generate all unique pairs indices (i < j)
    idx_i_np, idx_j_np = np.triu_indices(N_np, k=1)
    idx_i_jnp, idx_j_jnp = jnp.triu_indices(N_jnp, k=1)

    # Extract charges and positions for each pair
    Z_i = all_charges[idx_i_np]  # Shape: (M,)
    Z_j = all_charges[idx_j_np]  # Shape: (M,)
    r_i = all_carts[idx_i_jnp]  # Shape: (M, D)
    r_j = all_carts[idx_j_jnp]  # Shape: (M, D)

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


def _compute_coulomb_potential_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    wavefunction_data: Wavefunction_data = None,
) -> float:
    """See compute_coulomb_potential_api."""
    # all-electron
    if not coulomb_potential_data.ecp_flag:
        bare_coulomb_potential = _compute_bare_coulomb_potential_debug(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        ecp_coulomb_potential = 0

    # pseudo-potential
    else:
        bare_coulomb_potential = _compute_bare_coulomb_potential_debug(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        ecp_coulomb_potential = _compute_ecp_coulomb_potential_debug(
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
    debug: bool = False,
) -> float:
    """Compute coulomb potential including bare electron-ion, electron-electron, ecp local and non-local parts.

    The method is for computing coulomb potential including bare electron-ion, electron-electron,
    ecp local and non-local parts, of the given ECPs at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data):
            an instance of Coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]):
            Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]):
            Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        debug (bool): if True, the value is computed via _debug function for debuging purpose

    Returns:
        float:  The sum of bare electron-ion, electron-electron, local and non-local parts of the given
                ECPs with r_up_carts and r_dn_carts. (float)
    """
    # all-electron
    if not coulomb_potential_data.ecp_flag:
        bare_coulomb_potential = compute_bare_coulomb_potential_api(
            coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts, debug=debug
        )
        ecp_coulomb_potential = 0

    # pseudo-potential
    else:
        bare_coulomb_potential = compute_bare_coulomb_potential_api(
            coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts, debug=debug
        )

        # ecp_coulomb_potential = 0.0

        # """
        ecp_coulomb_potential = compute_ecp_coulomb_potential_api(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            wavefunction_data=wavefunction_data,
            Nv=6,
            debug=debug,
        )
        # """

    return bare_coulomb_potential + ecp_coulomb_potential


if __name__ == "__main__":
    import os

    from .jastrow_factor import Jastrow_data, Jastrow_two_body_data

    # from .hamiltonians import Hamiltonian_data
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
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "water_ccpvtz_trexio.hdf5"))

    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    # WF data
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_two_body_pade_flag=True,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    # Initialization
    r_up_carts = []
    r_dn_carts = []

    total_electrons = 0

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data.positions_cart

    # Place electrons around each nucleus
    for i in range(len(coords)):
        charge = charges[i]
        num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

        # Retrieve the position coordinates
        x, y, z = coords[i]

        # Place electrons
        for _ in range(num_electrons):
            # Calculate distance range
            distance = np.random.uniform(1.0 / charge, 2.0 / charge)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            # Convert spherical to Cartesian coordinates
            dx = distance * np.sin(theta) * np.cos(phi)
            dy = distance * np.sin(theta) * np.sin(phi)
            dz = distance * np.cos(theta)

            # Position of the electron
            electron_position = np.array([x + dx, y + dy, z + dz])

            # Assign spin
            if len(r_up_carts) < num_electron_up:
                r_up_carts.append(electron_position)
            else:
                r_dn_carts.append(electron_position)

        total_electrons += num_electrons

    # Handle surplus electrons
    remaining_up = num_electron_up - len(r_up_carts)
    remaining_dn = num_electron_dn - len(r_dn_carts)

    # Randomly place any remaining electrons
    for _ in range(remaining_up):
        r_up_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))
    for _ in range(remaining_dn):
        r_dn_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))

    r_up_carts = np.array(r_up_carts)
    r_dn_carts = np.array(r_dn_carts)

    mesh_non_local_ecp_part_jax, V_nonlocal_jax, sum_V_nonlocal_jax = _compute_ecp_non_local_parts_jax(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        Nv=6,
        flag_determinant_only=False,
    )

    mesh_non_local_ecp_part_debug, V_nonlocal_debug, sum_V_nonlocal_debug = _compute_ecp_non_local_parts_debug(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        Nv=6,
        flag_determinant_only=False,
    )

    np.testing.assert_almost_equal(sum_V_nonlocal_jax, sum_V_nonlocal_debug, decimal=5)

    V_ecp_non_local_max_debug = V_nonlocal_debug[np.argmax(V_nonlocal_debug)]
    V_ecp_non_local_max_jax = V_nonlocal_jax[np.argmax(V_nonlocal_jax)]

    np.testing.assert_almost_equal(V_ecp_non_local_max_debug, V_ecp_non_local_max_jax, decimal=5)

    mesh_non_local_max_debug = mesh_non_local_ecp_part_debug[np.argmax(V_nonlocal_debug)]
    mesh_non_local_max_jax = mesh_non_local_ecp_part_jax[np.argmax(V_nonlocal_jax)]

    np.testing.assert_array_almost_equal(mesh_non_local_max_debug, mesh_non_local_max_jax, decimal=5)

    mesh_non_local_ecp_part_only_det_jax, V_nonlocal_only_det_jax, sum_V_nonlocal_only_det_jax = (
        _compute_ecp_non_local_parts_jax(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            Nv=6,
            flag_determinant_only=True,
        )
    )

    mesh_non_local_ecp_part_only_det_debug, V_nonlocal_only_det_debug, sum_V_nonlocal_only_det_debug = (
        _compute_ecp_non_local_parts_debug(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            Nv=6,
            flag_determinant_only=True,
        )
    )

    np.testing.assert_almost_equal(sum_V_nonlocal_only_det_jax, sum_V_nonlocal_only_det_debug, decimal=5)
