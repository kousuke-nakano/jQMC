"""Effective core potential module.

Module containing classes and methods related to Effective core potential
and bare Coulomb potentials

Todo:
    Remove the native 'for' loops for up and down electron positions in the function
    '_compute_ecp_non_local_parts_NN_jax' and replace them with e.g., jax.lax.scan.

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
# * Neither the name of the jqmc project nor the names of its
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

import itertools
import time
from functools import partial
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

from .determinant import compute_det_geminal_all_elements_jax
from .function_collections import legendre_tablated as jnp_legendre_tablated
from .jastrow_factor import compute_Jastrow_part_jax
from .setting import NN_default, Nv_default
from .structure import (
    Structure_data,
    find_nearest_nucleus_indices_jnp,
    find_nearest_nucleus_indices_np,
    get_min_dist_rel_R_cart_jnp,
    get_min_dist_rel_R_cart_np,
)
from .wavefunction import Wavefunction_data

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


# non local PPs, Mesh Info. taken from Mitas's paper [J. Chem. Phys., 95, 5, (1991)]
# why namedtuple? because it should be immutable
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

# Icosahedron symmetry quadrature (Nv=12)
A = 1.0 / 12.0
B = 1.0 / 12.0
grid_points_sphe = np.array(
    [
        [0.0, 0.0],
        [np.pi, 0.0],
        [np.arctan(2), (2.0 * 0 * np.pi) / 5.0],
        [np.arctan(2), (2.0 * 1 * np.pi) / 5.0],
        [np.arctan(2), (2.0 * 2 * np.pi) / 5.0],
        [np.arctan(2), (2.0 * 3 * np.pi) / 5.0],
        [np.arctan(2), (2.0 * 4 * np.pi) / 5.0],
        [np.pi - np.arctan(2), (((2.0 * 0) + 1.0) * np.pi) / 5.0],
        [np.pi - np.arctan(2), (((2.0 * 1) + 1.0) * np.pi) / 5.0],
        [np.pi - np.arctan(2), (((2.0 * 2) + 1.0) * np.pi) / 5.0],
        [np.pi - np.arctan(2), (((2.0 * 3) + 1.0) * np.pi) / 5.0],
        [np.pi - np.arctan(2), (((2.0 * 4) + 1.0) * np.pi) / 5.0],
    ]
)
theta = grid_points_sphe[:, 0]
phi = grid_points_sphe[:, 1]
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
grid_points_cart = np.vstack((x, y, z)).T
icosahedron_sym_mesh_Nv12 = _Mesh(Nv=12, weights=[A, A, B, B, B, B, B, B, B, B, B, B], grid_points=grid_points_cart)

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
        z_cores (list[float] | tuple[float]]):
            Number of core electrons to remove per atom (dim: num_atoms).
        max_ang_mom_plus_1 (list[int] | tuple[int]):
            l_{max}+1, one higher than the max angular momentum in the
            removed core orbitals (dim: num_atoms)
        num_ecps (int):
            Total number of ECP functions for all atoms and all values of l
        ang_moms (list[int] | tuple[int]):
            One-to-one correspondence between ECP items and the angular momentum l (dim:num_ecps)
        nucleus_index (list[int] | tuple[int]):
            One-to-one correspondence between ECP items and the atom index (dim:num_ecps)
        exponents (list[float] | tuple[float]):
            all ECP exponents (dim:num_ecps)
        coefficients (list[float] | tuple[float]):
            all ECP coefficients (dim:num_ecps)
        powers (list[int] | tuple[int]):
            all ECP powers (dim:num_ecps)
        structure_data (Structure_data):
            Instance of a structure_data

    """

    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    ecp_flag: bool = struct.field(pytree_node=False, default=False)
    z_cores: list[float] | tuple[float] = struct.field(pytree_node=False, default_factory=tuple)
    max_ang_mom_plus_1: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    num_ecps: int = struct.field(pytree_node=False, default=0)
    ang_moms: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    nucleus_index: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    exponents: list[float] | tuple[float] = struct.field(pytree_node=False, default_factory=tuple)
    coefficients: list[float] | tuple[float] = struct.field(pytree_node=False, default_factory=tuple)
    powers: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.ecp_flag:
            if len(self.z_cores) != self.structure_data.natom:
                raise ValueError("dim. of self.z_cores is wrong")
            if len(self.max_ang_mom_plus_1) != self.structure_data.natom:
                raise ValueError("dim. of self.max_ang_mom_plus_1 is wrong")
            if len(self.ang_moms) != self.num_ecps:
                raise ValueError("dim. of self.num_ecps is wrong")
            if len(self.nucleus_index) != self.num_ecps:
                raise ValueError("dim. of self.nucleus_index is wrong")
            if len(self.exponents) != self.num_ecps:
                raise ValueError("dim. of self.ang_moms is wrong")
            if len(self.coefficients) != self.num_ecps:
                raise ValueError("dim. of self.coefficients is wrong")
            if len(self.powers) != self.num_ecps:
                raise ValueError("dim. of self.powers is wrong")
        else:
            if len(self.z_cores) != 0:
                raise ValueError("dim. of self.z_cores is wrong")
            if len(self.max_ang_mom_plus_1) != 0:
                raise ValueError("dim. of self.max_ang_mom_plus_1 is wrong")
            if len(self.ang_moms) != 0:
                raise ValueError("dim. of self.ang_moms is wrong")
            if len(self.nucleus_index) != 0:
                raise ValueError("dim. of self.nucleus_index is wrong")
            if len(self.exponents) != 0:
                raise ValueError("dim. of self.exponents is wrong")
            if len(self.coefficients) != 0:
                raise ValueError("dim. of self.coefficients is wrong")
            if len(self.powers) != 0:
                raise ValueError("dim. of self.powers is wrong")

        if not isinstance(self.ecp_flag, bool):
            raise ValueError(f"ecp_flag = {type(self.ecp_flag)} must be a bool.")
        if not isinstance(self.z_cores, (list, tuple)):
            logger.warning(f"z_cores = {type(self.z_cores)} must be a list or tuple. ValueError in a future release.")
            # raise ValueError(f"z_cores = {type(self.z_cores)} must be a list or tuple.")
        if not isinstance(self.max_ang_mom_plus_1, (list, tuple)):
            logger.warning(
                f"max_ang_mom_plus_1 = {type(self.max_ang_mom_plus_1)} must be a list or tuple. ValueError in a future release."
            )
            # raise ValueError(f"max_ang_mom_plus_1 = {type(self.max_ang_mom_plus_1)} must be a list or tuple.")
        if not isinstance(self.num_ecps, int):
            raise ValueError(f"num_ecps = {type(self.num_ecps)} must be an int.")
        if not isinstance(self.ang_moms, (list, tuple)):
            logger.warning(f"ang_moms = {type(self.ang_moms)} must be a list or tuple. ValueError in a future release.")
            # raise ValueError(f"ang_moms = {type(self.ang_moms)} must be a list or tuple.")
        if not isinstance(self.exponents, (list, tuple)):
            logger.warning(f"exponents = {type(self.exponents)} must be a list or tuple. ValueError in a future release.")
            # raise ValueError(f"exponents = {type(self.exponents)} must be a list or tuple.")
        if not isinstance(self.coefficients, (list, tuple)):
            logger.warning(f"coefficients = {type(self.coefficients)} must be a list or tuple. ValueError in a future release.")
            # raise ValueError(f"coefficients = {type(self.coefficients)} must be a list or tuple.")
        if not isinstance(self.powers, (list, tuple)):
            logger.warning(f"powers = {type(self.powers)} must be a list or tuple. ValueError in a future release.")
            # raise ValueError(f"powers = {type(self.powers)} must be a list or tuple.")

        self.structure_data.sanity_check()

    def get_info(self) -> list[str]:
        """Return a list of strings containing the attribute information."""
        info_lines = ["**" + self.__class__.__name__]
        info_lines.append(f"  ecp_flag = {self.ecp_flag}")
        return info_lines

    def logger_info(self) -> None:
        """Log the information from get_info() using logger.info."""
        for line in self.get_info():
            logger.info(line)

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
    def global_max_ang_mom_plus_1(self) -> int:
        """The maximum number of ang_mom_plus_1 among all atoms."""
        return np.max(self.max_ang_mom_plus_1)

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
        return np.array(self.ang_moms)[self.non_local_part_index]

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

    # padding
    @property
    def n_atom(self) -> int:
        """Number of atoms inculded in the system."""
        return np.max(self.nucleus_index_local_part) + 1

    @property
    def padded_parameters_tuple(self):
        """Padding parameters for jit(vmap).

        Ensure that each atom has the global max ang_mom and the same number of
        params for jit vmap. Padded jnp.arrays are returned.
        If an atom's local max ang_mom is less than the global max,
        append a dummy element with ang_mom = global_ang_mom_non_local_part.

        Returns updated lists of:
            -ang_mom_non_local_part_padded_jnp,
            -exponents_non_local_part_padded_jnp,
            -coefficients_non_local_part_padded_jnp,
            -powers_non_local_part_padded_jnp,
        so that each atom's maximum ang_mom matches the global max,
        and the same dim. for i_atom (nucleus index).
        """
        # 1) Infer the number of atoms (n_atom) from the maximum index
        n_atom = max(self.nucleus_index_non_local_part) + 1

        # 2) Prepare a temporary data structure to group items by atom
        #    grouped[i_atom] = [(ang_mom, exponent), (ang_mom, exponent), ...]
        grouped = [[] for _ in range(n_atom)]
        for i in range(len(self.nucleus_index_non_local_part)):
            i_atom = self.nucleus_index_non_local_part[i]
            l_val = self.ang_mom_non_local_part[i]
            expo = self.exponents_non_local_part[i]
            coeff = self.coefficients_non_local_part[i]
            power = self.powers_non_local_part[i]
            grouped[i_atom].append((l_val, expo, coeff, power))

        # 3) For each atom, if the local max ang_mom is less than the global, append one dummy element
        for i_atom in range(n_atom):
            if len(grouped[i_atom]) == 0:
                # If the atom has no elements at all, skip or add one by choice
                continue

            local_max = max(pair[0] for pair in grouped[i_atom])
            if local_max < self.global_max_ang_mom_plus_1:
                # Append one element with ang_mom = global and expo, coeff, power = 0.0
                grouped[i_atom].append((self.global_max_ang_mom_plus_1, 0.0, 0.0, 0.0))

        # 4) Flatten the data structure back into lists
        nucleus_index_with_global_lmax_plus1 = []
        new_ang_mom = []
        new_exponents = []
        new_coefficients = []
        new_powers = []
        for i_atom in range(n_atom):
            for l_val, expo, coeff, power in grouped[i_atom]:
                nucleus_index_with_global_lmax_plus1.append(i_atom)
                new_ang_mom.append(l_val)
                new_exponents.append(expo)
                new_coefficients.append(coeff)
                new_powers.append(power)

        nucleus_index_with_global_lmax_plus1 = np.array(nucleus_index_with_global_lmax_plus1)
        ang_mom_with_global_lmax_plus1 = np.array(new_ang_mom)
        exponents_global_lmax_plus1 = np.array(new_exponents)
        coefficients_global_lmax_plus1 = np.array(new_coefficients)
        powers_global_lmax_plus1 = np.array(new_powers)

        # 5) count the max of i_atom appearance (i.e., max num of params.)
        counts = np.bincount(nucleus_index_with_global_lmax_plus1)
        max_param_num = counts.max()

        # 6) padding ang_mom_non_local_part_padded_np
        ang_mom_non_local_part_padded_np = np.zeros((self.n_atom, max_param_num), dtype=ang_mom_with_global_lmax_plus1.dtype)
        row_counts = np.zeros((self.n_atom,), dtype=np.int32)
        for i in range(ang_mom_with_global_lmax_plus1.shape[0]):
            i_atom = nucleus_index_with_global_lmax_plus1[i]
            j = row_counts[i_atom]
            ang_mom_non_local_part_padded_np[i_atom, j] = ang_mom_with_global_lmax_plus1[i]
            row_counts[i_atom] += 1

        # 7) padding exponents_non_local_part_padded_np
        exponents_non_local_part_padded_np = np.zeros((self.n_atom, max_param_num), dtype=exponents_global_lmax_plus1.dtype)
        row_counts = np.zeros((self.n_atom,), dtype=np.int32)
        for i in range(exponents_global_lmax_plus1.shape[0]):
            i_atom = nucleus_index_with_global_lmax_plus1[i]
            j = row_counts[i_atom]
            exponents_non_local_part_padded_np[i_atom, j] = exponents_global_lmax_plus1[i]
            row_counts[i_atom] += 1

        # 8) padding coefficients_non_local_part_padded_np
        coefficients_non_local_part_padded_np = np.zeros(
            (self.n_atom, max_param_num), dtype=coefficients_global_lmax_plus1.dtype
        )
        row_counts = np.zeros((self.n_atom,), dtype=np.int32)
        for i in range(coefficients_global_lmax_plus1.shape[0]):
            i_atom = nucleus_index_with_global_lmax_plus1[i]
            j = row_counts[i_atom]
            coefficients_non_local_part_padded_np[i_atom, j] = coefficients_global_lmax_plus1[i]
            row_counts[i_atom] += 1

        # 9) padding coefficients_non_local_part_padded_np
        powers_non_local_part_padded_np = np.zeros((self.n_atom, max_param_num), dtype=powers_global_lmax_plus1.dtype)
        row_counts = np.zeros((self.n_atom,), dtype=np.int32)
        for i in range(powers_global_lmax_plus1.shape[0]):
            i_atom = nucleus_index_with_global_lmax_plus1[i]
            j = row_counts[i_atom]
            powers_non_local_part_padded_np[i_atom, j] = powers_global_lmax_plus1[i]
            row_counts[i_atom] += 1

        ang_mom_non_local_part_padded_jnp = jnp.array(ang_mom_non_local_part_padded_np)
        exponents_non_local_part_padded_jnp = jnp.array(exponents_non_local_part_padded_np)
        coefficients_non_local_part_padded_jnp = jnp.array(coefficients_non_local_part_padded_np)
        powers_non_local_part_padded_jnp = jnp.array(powers_non_local_part_padded_np)

        return (
            ang_mom_non_local_part_padded_jnp,
            exponents_non_local_part_padded_jnp,
            coefficients_non_local_part_padded_jnp,
            powers_non_local_part_padded_jnp,
        )


@struct.dataclass
class Coulomb_potential_data_no_deriv:
    """See Coulomb_potential_data."""

    structure_data: Structure_data = struct.field(pytree_node=False, default_factory=lambda: Structure_data())
    ecp_flag: bool = struct.field(pytree_node=False, default=False)
    z_cores: list[float] | tuple[float] = struct.field(pytree_node=False, default_factory=tuple)
    max_ang_mom_plus_1: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    num_ecps: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    ang_moms: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    nucleus_index: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    exponents: list[float] | tuple[float] = struct.field(pytree_node=False, default_factory=tuple)
    coefficients: list[float] | tuple[float] = struct.field(pytree_node=False, default_factory=tuple)
    powers: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)

    @classmethod
    def from_base(cls, coulomb_potential_data: Coulomb_potential_data):
        """Switch pytree_node."""
        structure_data = coulomb_potential_data.structure_data
        ecp_flag = coulomb_potential_data.ecp_flag
        z_cores = coulomb_potential_data.z_cores
        max_ang_mom_plus_1 = coulomb_potential_data.max_ang_mom_plus_1
        num_ecps = coulomb_potential_data.num_ecps
        ang_moms = coulomb_potential_data.ang_moms
        nucleus_index = coulomb_potential_data.nucleus_index
        exponents = coulomb_potential_data.exponents
        coefficients = coulomb_potential_data.coefficients
        powers = coulomb_potential_data.powers
        return cls(
            structure_data,
            ecp_flag,
            z_cores,
            max_ang_mom_plus_1,
            num_ecps,
            ang_moms,
            nucleus_index,
            exponents,
            coefficients,
            powers,
        )


def compute_ecp_local_parts_all_pairs_debug(
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


def compute_ecp_non_local_parts_all_pairs_debug(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    RT: npt.NDArray = np.eye(3),
    Nv: int = Nv_default,
    flag_determinant_only: bool = False,
) -> tuple[list, list, list, float]:
    """Compute ecp non-local parts, considering all nucleus-electron pairs.

    The method is for computing the non-local part of the given ECPs at (r_up_carts, r_dn_carts).
    A very straightforward (so very slow) implementation. Just for debudding purpose.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (npt.NDArray): Rotation matrix. equiv R.T
        Nv (int): The number of quadrature points for the spherical part.
        flag_determinant_only (bool): If True, only the determinant part is considered for the non-local ECP part.

    Returns:
        list[npt.NDArray]: The list of grids for up electrons on which the non-local part is computed.
        list[npt.NDArray]: The list of grids for dn electrons on which the non-local part is computed.
        list[float]: The list of non-local part of the given ECPs with r_up_carts and r_dn_carts.
        float: sum of the V_nonlocal
    """
    if Nv == 4:
        weights = tetrahedron_sym_mesh_Nv4.weights
        grid_points = tetrahedron_sym_mesh_Nv4.grid_points
    elif Nv == 6:
        weights = octahedron_sym_mesh_Nv6.weights
        grid_points = octahedron_sym_mesh_Nv6.grid_points
    elif Nv == 12:
        weights = icosahedron_sym_mesh_Nv12.weights
        grid_points = icosahedron_sym_mesh_Nv12.grid_points
    elif Nv == 18:
        weights = octahedron_sym_mesh_Nv18.weights
        grid_points = octahedron_sym_mesh_Nv18.grid_points
    else:
        raise NotImplementedError

    grid_points = grid_points @ RT  # rotate the grid points. dim. (N,3) @ (3,3) = (N,3)

    mesh_non_local_ecp_part = []
    V_nonlocal = []
    sum_V_nonlocal = 0.0

    if flag_determinant_only:
        jastrow_denominator = 1.0
    else:
        jastrow_denominator = compute_Jastrow_part_jax(
            jastrow_data=wavefunction_data.jastrow_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

    det_denominator = compute_det_geminal_all_elements_jax(
        geminal_data=wavefunction_data.geminal_data,
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
                        jastrow_numerator = 1
                    else:
                        jastrow_numerator = compute_Jastrow_part_jax(
                            jastrow_data=wavefunction_data.jastrow_data,
                            r_up_carts=r_up_carts_on_mesh,
                            r_dn_carts=r_dn_carts,
                        )

                    det_numerator = compute_det_geminal_all_elements_jax(
                        geminal_data=wavefunction_data.geminal_data,
                        r_up_carts=r_up_carts_on_mesh,
                        r_dn_carts=r_dn_carts,
                    )

                    wf_ratio = np.exp(jastrow_numerator - jastrow_denominator) * det_numerator / det_denominator

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
                        det_numerator = compute_det_geminal_all_elements_jax(
                            geminal_data=wavefunction_data.geminal_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts_on_mesh,
                        )
                    else:
                        det_numerator = compute_det_geminal_all_elements_jax(
                            geminal_data=wavefunction_data.geminal_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts_on_mesh,
                        )

                    wf_ratio = np.exp(jastrow_numerator - jastrow_denominator) * det_numerator / det_denominator

                    P_l = (2 * ang_mom + 1) * eval_legendre(ang_mom, cos_theta) * weight * wf_ratio
                    mesh_non_local_ecp_part.append((r_up_carts, r_dn_carts_on_mesh))
                    V_nonlocal.append(V_l * P_l)
                    sum_V_nonlocal += V_l * P_l

    mesh_non_local_ecp_part_r_up_carts = np.array([up for up, _ in mesh_non_local_ecp_part])
    mesh_non_local_ecp_part_r_dn_carts = np.array([dn for _, dn in mesh_non_local_ecp_part])
    V_nonlocal = np.array(V_nonlocal)

    return mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, sum_V_nonlocal


def compute_ecp_non_local_parts_nearest_neighbors_debug(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    RT: npt.NDArray = np.eye(3),
    NN: int = NN_default,
    Nv: int = Nv_default,
    flag_determinant_only: bool = False,
) -> tuple[list, list, list, float]:
    """Compute ecp non-local parts.

    The method is for computing the non-local part of the given ECPs at (r_up_carts, r_dn_carts)
    with a cutoff considering only up to NN-th nearest neighbors. A very straightforward (so very slow) implementation.
    Just for debudding purpose.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (npt.NDArray): Rotation matrix. equiv R.T
        NN (int): Consider only up to NN-th nearest neighbors.
        Nv (int): The number of quadrature points for the spherical part.
        flag_determinant_only (bool): If True, only the determinant part is considered for the non-local ECP part.

    Returns:
        list[npt.NDArray]: The list of grids for up electrons on which the non-local part is computed.
        list[npt.NDArray]: The list of grids for dn electrons on which the non-local part is computed.
        list[float]: The list of non-local part of the given ECPs with r_up_carts and r_dn_carts.
        float: sum of the V_nonlocal
    """
    if Nv == 4:
        weights = tetrahedron_sym_mesh_Nv4.weights
        grid_points = tetrahedron_sym_mesh_Nv4.grid_points
    elif Nv == 6:
        weights = octahedron_sym_mesh_Nv6.weights
        grid_points = octahedron_sym_mesh_Nv6.grid_points
    elif Nv == 12:
        weights = icosahedron_sym_mesh_Nv12.weights
        grid_points = icosahedron_sym_mesh_Nv12.grid_points
    elif Nv == 18:
        weights = octahedron_sym_mesh_Nv18.weights
        grid_points = octahedron_sym_mesh_Nv18.grid_points
    else:
        raise NotImplementedError

    grid_points = grid_points @ RT  # rotate the grid points. dim. (N,3) @ (3,3) = (N,3)

    V_nonlocal = []
    sum_V_nonlocal = 0.0

    if flag_determinant_only:
        jastrow_denominator = 1.0
    else:
        jastrow_denominator = compute_Jastrow_part_jax(
            jastrow_data=wavefunction_data.jastrow_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

    det_denominator = compute_det_geminal_all_elements_jax(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    i_atom_np = np.array(coulomb_potential_data.nucleus_index_non_local_part)
    ang_mom_np = np.array(coulomb_potential_data.ang_mom_non_local_part)
    exponent_np = np.array(coulomb_potential_data.exponents_non_local_part)
    coefficient_np = np.array(coulomb_potential_data.coefficients_non_local_part)
    power_np = np.array(coulomb_potential_data.powers_non_local_part)

    # up electrons
    up_mesh_non_local_ecp_part_up = []
    up_mesh_non_local_ecp_part_dn = []

    for r_up_i, r_up_cart in enumerate(r_up_carts):
        i_atom_list = find_nearest_nucleus_indices_np(
            structure_data=coulomb_potential_data.structure_data,
            r_cart=r_up_cart,
            N=NN,
        )

        for i_atom in i_atom_list:
            rel_R_cart_min_dist = get_min_dist_rel_R_cart_np(
                structure_data=coulomb_potential_data.structure_data,
                r_cart=r_up_cart,
                i_atom=i_atom,
            )

            max_ang_mom_plus_1 = coulomb_potential_data.max_ang_mom_plus_1[i_atom]
            i_target = [i for i, v in enumerate(i_atom_np) if v == i_atom]
            ang_moms = ang_mom_np[i_target]
            exponents = exponent_np[i_target]
            coefficients = coefficient_np[i_target]
            powers = power_np[i_target]

            # dim (#ang_moms, 1) # note: ang_moms include the same l because each l has seveal (exp, coeff, and powers).
            # jax.ops.segsum cares about it later.
            V_l_list = (
                np.linalg.norm(rel_R_cart_min_dist) ** -2.0
                * coefficients
                * np.linalg.norm(rel_R_cart_min_dist) ** powers
                * np.exp(-exponents * np.linalg.norm(rel_R_cart_min_dist) ** 2.0)
            )

            weight_list = []
            cos_theta_list = []
            wf_ratio_list = []
            for weight, vec_delta in zip(weights, grid_points):
                weight_list.append(weight)
                r_up_carts_on_mesh = r_up_carts.copy()
                r_up_carts_on_mesh[r_up_i] = r_up_cart + rel_R_cart_min_dist + np.linalg.norm(rel_R_cart_min_dist) * vec_delta
                up_mesh_non_local_ecp_part_up.append(r_up_carts_on_mesh)
                up_mesh_non_local_ecp_part_dn.append(r_dn_carts)

                cos_theta = np.dot(
                    -1.0 * (rel_R_cart_min_dist) / np.linalg.norm(rel_R_cart_min_dist),
                    ((vec_delta) / np.linalg.norm(vec_delta)),
                )
                cos_theta_list.append(cos_theta)

                if flag_determinant_only:
                    jastrow_numerator = 1.0
                else:
                    jastrow_numerator = compute_Jastrow_part_jax(
                        jastrow_data=wavefunction_data.jastrow_data,
                        r_up_carts=r_up_carts_on_mesh,
                        r_dn_carts=r_dn_carts,
                    )

                det_numerator = compute_det_geminal_all_elements_jax(
                    geminal_data=wavefunction_data.geminal_data,
                    r_up_carts=r_up_carts_on_mesh,
                    r_dn_carts=r_dn_carts,
                )

                wf_ratio = np.exp(jastrow_numerator - jastrow_denominator) * det_numerator / det_denominator
                wf_ratio_list.append(wf_ratio)

            for ang_mom in range(max_ang_mom_plus_1):
                ll_target = [i for i, v in enumerate(ang_moms) if v == ang_mom]
                V_l = np.sum([V_l_list[ll] for ll in ll_target])
                P_l = np.array(
                    [
                        (2 * ang_mom + 1) * eval_legendre(ang_mom, cos_theta) * weight * wf_ratio
                        for cos_theta, weight, wf_ratio in zip(cos_theta_list, weight_list, wf_ratio_list)
                    ]
                )
                ans = list(V_l * P_l)
                V_nonlocal += ans
                sum_V_nonlocal += np.sum(ans)

    # dn electrons
    dn_mesh_non_local_ecp_part_up = []
    dn_mesh_non_local_ecp_part_dn = []

    for r_dn_i, r_dn_cart in enumerate(r_dn_carts):
        i_atom_list = find_nearest_nucleus_indices_np(
            structure_data=coulomb_potential_data.structure_data,
            r_cart=r_dn_cart,
            N=NN,
        )

        for i_atom in i_atom_list:
            rel_R_cart_min_dist = get_min_dist_rel_R_cart_np(
                structure_data=coulomb_potential_data.structure_data,
                r_cart=r_dn_cart,
                i_atom=i_atom,
            )

            max_ang_mom_plus_1 = coulomb_potential_data.max_ang_mom_plus_1[i_atom]
            i_target = [i for i, v in enumerate(i_atom_np) if v == i_atom]
            ang_moms = ang_mom_np[i_target]
            exponents = exponent_np[i_target]
            coefficients = coefficient_np[i_target]
            powers = power_np[i_target]

            # dim (#ang_moms, 1) # note: ang_moms include the same l because each l has seveal (exp, coeff, and powers).
            # jax.ops.segsum cares about it later.
            V_l_list = (
                np.linalg.norm(rel_R_cart_min_dist) ** -2.0
                * coefficients
                * np.linalg.norm(rel_R_cart_min_dist) ** powers
                * np.exp(-exponents * np.linalg.norm(rel_R_cart_min_dist) ** 2.0)
            )

            weight_list = []
            cos_theta_list = []
            wf_ratio_list = []
            for weight, vec_delta in zip(weights, grid_points):
                weight_list.append(weight)
                r_dn_carts_on_mesh = r_dn_carts.copy()
                r_dn_carts_on_mesh[r_dn_i] = r_dn_cart + rel_R_cart_min_dist + np.linalg.norm(rel_R_cart_min_dist) * vec_delta
                dn_mesh_non_local_ecp_part_up.append(r_up_carts)
                dn_mesh_non_local_ecp_part_dn.append(r_dn_carts_on_mesh)

                cos_theta = np.dot(
                    -1.0 * (rel_R_cart_min_dist) / np.linalg.norm(rel_R_cart_min_dist),
                    ((vec_delta) / np.linalg.norm(vec_delta)),
                )
                cos_theta_list.append(cos_theta)

                if flag_determinant_only:
                    jastrow_numerator = 1.0
                else:
                    jastrow_numerator = compute_Jastrow_part_jax(
                        jastrow_data=wavefunction_data.jastrow_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts_on_mesh,
                    )

                det_numerator = compute_det_geminal_all_elements_jax(
                    geminal_data=wavefunction_data.geminal_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts_on_mesh,
                )

                wf_ratio = np.exp(jastrow_numerator - jastrow_denominator) * det_numerator / det_denominator
                wf_ratio_list.append(wf_ratio)

            for ang_mom in range(max_ang_mom_plus_1):
                ll_target = [i for i, v in enumerate(ang_moms) if v == ang_mom]
                V_l = np.sum([V_l_list[ll] for ll in ll_target])
                P_l = np.array(
                    [
                        (2 * ang_mom + 1) * eval_legendre(ang_mom, cos_theta) * weight * wf_ratio
                        for cos_theta, weight, wf_ratio in zip(cos_theta_list, weight_list, wf_ratio_list)
                    ]
                )
                ans = list(V_l * P_l)
                V_nonlocal += ans
                sum_V_nonlocal += np.sum(ans)

    mesh_non_local_ecp_part_r_up_carts = np.array(up_mesh_non_local_ecp_part_up + dn_mesh_non_local_ecp_part_up)
    mesh_non_local_ecp_part_r_dn_carts = np.array(up_mesh_non_local_ecp_part_dn + dn_mesh_non_local_ecp_part_dn)
    V_nonlocal = np.array(V_nonlocal)

    return mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, sum_V_nonlocal


def compute_ecp_coulomb_potential_debug(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    RT: npt.NDArray = np.eye(3),
    NN: int = NN_default,
    Nv: int = Nv_default,
) -> float:
    """Compute ecp local and non-local parts.

    The method is for computing the local and non-local part of the given ECPs at (r_up_carts, r_dn_carts).
    A very straightforward (so very slow) implementation. Just for debudding purpose.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (npt.NDArray): Rotation matrix. equiv R.T used for non-local part
        NN (int): Consider only up to NN-th nearest neighbors.
        Nv (int): The number of quadrature points for the spherical part.
    Returns:
        float: The sum of non-local part of the given ECPs with r_up_carts and r_dn_carts.
    """
    ecp_local_parts = compute_ecp_local_parts_all_pairs_debug(
        coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    _, _, _, ecp_nonlocal_parts = compute_ecp_non_local_parts_nearest_neighbors_debug(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
        Nv=Nv,
        NN=NN,
        flag_determinant_only=False,
    )

    V_ecp = ecp_local_parts + ecp_nonlocal_parts

    return V_ecp


@jit
def compute_ecp_local_parts_all_pairs_jax(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> float:
    """Compute ecp local parts, considering all nucleus-electron pairs.

    The method is for computing the local part of the given ECPs at (r_up_carts, r_dn_carts).
    A much faster implementation using JAX.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

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


@partial(jit, static_argnums=(5, 6, 7))
def compute_ecp_non_local_parts_nearest_neighbors_jax(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    NN: int = NN_default,
    Nv: int = Nv_default,
    flag_determinant_only: bool = False,
) -> tuple[list, list, list, float]:
    """Compute ecp non-local parts.

    The method is for computing the non-local part of the given ECPs at (r_up_carts, r_dn_carts)
    with a cutoff considering only up to NN-th nearest neighbors.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (jnpt.ArrayLike): Rotation matrix. equiv R.T
        NN (int): Consider only up to N-th nearest neighbors.
        Nv (int): The number of quadrature points for the spherical part.
        flag_determinant_only (bool): If True, only the determinant part is considered for the non-local ECP part.

    Returns:
        list[jax.Array]: The list of grids for up electrons on which the non-local part is computed.
        list[jax.Array]: The list of grids for dn electrons on which the non-local part is computed.
        list[float]: The list of non-local part of the given ECPs with r_up_carts and r_dn_carts.
        float: sum of the V_nonlocal
    """
    if Nv == 4:
        weights = jnp.array(tetrahedron_sym_mesh_Nv4.weights)
        grid_points = jnp.array(tetrahedron_sym_mesh_Nv4.grid_points)
    elif Nv == 6:
        weights = jnp.array(octahedron_sym_mesh_Nv6.weights)
        grid_points = jnp.array(octahedron_sym_mesh_Nv6.grid_points)
    elif Nv == 12:
        weights = jnp.array(icosahedron_sym_mesh_Nv12.weights)
        grid_points = jnp.array(icosahedron_sym_mesh_Nv12.grid_points)
    elif Nv == 18:
        weights = jnp.array(octahedron_sym_mesh_Nv18.weights)
        grid_points = jnp.array(octahedron_sym_mesh_Nv18.grid_points)
    else:
        raise NotImplementedError

    grid_points = grid_points @ RT  # rotate the grid points. dim. (N,3) @ (3,3) = (N,3)

    # jnp variables
    ang_mom_all, exponent_all, coefficient_all, power_all = coulomb_potential_data.padded_parameters_tuple
    global_max_ang_mom_plus_1 = coulomb_potential_data.global_max_ang_mom_plus_1

    # stored
    non_local_ecp_part_r_carts_up = jnp.zeros((0, len(r_up_carts), 3))
    non_local_ecp_part_r_carts_dn = jnp.zeros((0, len(r_dn_carts), 3))
    cos_theta_all = jnp.zeros((0,))
    weight_all = jnp.zeros((0,))
    V_l_mapped_all = jnp.zeros((global_max_ang_mom_plus_1, 0))

    @jit
    def compute_V_l(rel_R_cart_min_dist, exponents, coefficients, powers):
        V_l = (
            jnp.linalg.norm(rel_R_cart_min_dist) ** -2.0
            * coefficients
            * jnp.linalg.norm(rel_R_cart_min_dist) ** powers
            * jnp.exp(-exponents * (jnp.linalg.norm(rel_R_cart_min_dist) ** 2))
        )

        return V_l

    @jit
    def compute_P_l(ang_mom, cos_theta, weight, wf_ratio):
        P_l = (2 * ang_mom + 1) * jnp_legendre_tablated(ang_mom, cos_theta) * weight * wf_ratio
        return P_l

    # up electrons
    for r_up_i, r_up_cart in enumerate(r_up_carts):
        i_atom_list = find_nearest_nucleus_indices_jnp(
            structure_data=coulomb_potential_data.structure_data,
            r_cart=r_up_cart,
            N=NN,
        )

        for i_atom in i_atom_list:
            rel_R_cart_min_dist = get_min_dist_rel_R_cart_jnp(
                structure_data=coulomb_potential_data.structure_data,
                r_cart=r_up_cart,
                i_atom=i_atom,
            )
            rel_R_cart_norm = jnp.linalg.norm(rel_R_cart_min_dist)

            max_ang_mom_plus_1 = coulomb_potential_data.global_max_ang_mom_plus_1
            ang_moms = ang_mom_all[i_atom]
            exponents = exponent_all[i_atom]
            coefficients = coefficient_all[i_atom]
            powers = power_all[i_atom]

            V_l_vmapped = compute_V_l(rel_R_cart_min_dist, exponents, coefficients, powers)
            V_l_mapped = jax.ops.segment_sum(V_l_vmapped, ang_moms, num_segments=max_ang_mom_plus_1)
            pad_width = (0, global_max_ang_mom_plus_1 - V_l_mapped.shape[0])
            V_l_padded = jnp.pad(
                V_l_mapped,
                pad_width=pad_width,
                mode="constant",
            )
            V_l_dup = V_l_padded[:, None].repeat(len(weights), axis=1)
            V_l_mapped_all = jnp.concatenate([V_l_mapped_all, V_l_dup], axis=1)

            offsets = rel_R_cart_min_dist + rel_R_cart_norm * grid_points
            updated_carts = r_up_cart + offsets
            r_up_carts_on_mesh = jnp.repeat(r_up_carts[jnp.newaxis, ...], grid_points.shape[0], axis=0)
            up_non_local_ecp_part_r_carts_up = r_up_carts_on_mesh.at[:, r_up_i, :].set(updated_carts)
            up_non_local_ecp_part_r_carts_dn = jnp.repeat(r_dn_carts[jnp.newaxis, ...], len(weights), axis=0)
            non_local_ecp_part_r_carts_up = jnp.concatenate(
                [non_local_ecp_part_r_carts_up, up_non_local_ecp_part_r_carts_up], axis=0
            )
            non_local_ecp_part_r_carts_dn = jnp.concatenate(
                [non_local_ecp_part_r_carts_dn, up_non_local_ecp_part_r_carts_dn], axis=0
            )

            cos_thetas = jnp.einsum(
                "ij,ij->i",
                -rel_R_cart_min_dist[None, :] / rel_R_cart_norm,
                grid_points / jnp.linalg.norm(grid_points, axis=1, keepdims=True),
            )
            cos_theta_all = jnp.concatenate([cos_theta_all, cos_thetas], axis=0)
            weight_all = jnp.concatenate([weight_all, weights], axis=0)

    # dn electrons
    for r_dn_i, r_dn_cart in enumerate(r_dn_carts):
        i_atom_list = find_nearest_nucleus_indices_jnp(
            structure_data=coulomb_potential_data.structure_data,
            r_cart=r_dn_cart,
            N=NN,
        )
        for i_atom in i_atom_list:
            rel_R_cart_min_dist = get_min_dist_rel_R_cart_jnp(
                structure_data=coulomb_potential_data.structure_data,
                r_cart=r_dn_cart,
                i_atom=i_atom,
            )
            rel_R_cart_norm = jnp.linalg.norm(rel_R_cart_min_dist)

            max_ang_mom_plus_1 = coulomb_potential_data.global_max_ang_mom_plus_1
            ang_moms = ang_mom_all[i_atom]
            exponents = exponent_all[i_atom]
            coefficients = coefficient_all[i_atom]
            powers = power_all[i_atom]

            V_l_vmapped = compute_V_l(rel_R_cart_min_dist, exponents, coefficients, powers)
            V_l_mapped = jax.ops.segment_sum(V_l_vmapped, ang_moms, num_segments=max_ang_mom_plus_1)
            pad_width = (0, global_max_ang_mom_plus_1 - V_l_mapped.shape[0])
            V_l_padded = jnp.pad(
                V_l_mapped,
                pad_width=pad_width,
                mode="constant",
            )
            V_l_dup = V_l_padded[:, None].repeat(len(weights), axis=1)
            V_l_mapped_all = jnp.concatenate([V_l_mapped_all, V_l_dup], axis=1)

            offsets = rel_R_cart_min_dist + rel_R_cart_norm * grid_points
            updated_carts = r_dn_cart + offsets
            r_dn_carts_on_mesh = jnp.repeat(r_dn_carts[jnp.newaxis, ...], grid_points.shape[0], axis=0)
            dn_non_local_ecp_part_r_carts_dn = r_dn_carts_on_mesh.at[:, r_dn_i, :].set(updated_carts)
            dn_non_local_ecp_part_r_carts_up = jnp.repeat(r_up_carts[jnp.newaxis, ...], len(weights), axis=0)
            non_local_ecp_part_r_carts_up = jnp.concatenate(
                [non_local_ecp_part_r_carts_up, dn_non_local_ecp_part_r_carts_up], axis=0
            )
            non_local_ecp_part_r_carts_dn = jnp.concatenate(
                [non_local_ecp_part_r_carts_dn, dn_non_local_ecp_part_r_carts_dn], axis=0
            )
            # Compute cos_theta for all grid points
            cos_thetas = jnp.einsum(
                "ij,ij->i",
                -rel_R_cart_min_dist[None, :] / rel_R_cart_norm,
                grid_points / jnp.linalg.norm(grid_points, axis=1, keepdims=True),
            )
            cos_theta_all = jnp.concatenate([cos_theta_all, cos_thetas], axis=0)
            weight_all = jnp.concatenate([weight_all, weights], axis=0)

    non_local_ecp_part_r_carts_up = jnp.array(non_local_ecp_part_r_carts_up)
    non_local_ecp_part_r_carts_dn = jnp.array(non_local_ecp_part_r_carts_dn)

    # jastrow_ratio
    if flag_determinant_only:
        jastrow_x = 1.0
        jastrow_xp = 1.0
    else:
        jastrow_x = compute_Jastrow_part_jax(wavefunction_data.jastrow_data, r_up_carts, r_dn_carts)
        jastrow_xp = vmap(compute_Jastrow_part_jax, in_axes=(None, 0, 0))(
            wavefunction_data.jastrow_data, non_local_ecp_part_r_carts_up, non_local_ecp_part_r_carts_dn
        )

    # det_ratio
    det_x = compute_det_geminal_all_elements_jax(wavefunction_data.geminal_data, r_up_carts, r_dn_carts)
    det_xp = vmap(compute_det_geminal_all_elements_jax, in_axes=(None, 0, 0))(
        wavefunction_data.geminal_data, non_local_ecp_part_r_carts_up, non_local_ecp_part_r_carts_dn
    )

    wf_ratio_all = jnp.exp(jastrow_xp - jastrow_x) * det_xp / det_x

    cos_theta_all = jnp.array(cos_theta_all)
    weight_all = jnp.array(weight_all)
    wf_ratio_all = jnp.array(wf_ratio_all)

    P_l_mapped_all = vmap(vmap(compute_P_l, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, None))(
        jnp.arange(global_max_ang_mom_plus_1), cos_theta_all, weight_all, wf_ratio_all
    )

    V_nl = V_l_mapped_all * P_l_mapped_all
    V_nonlocal = jnp.sum(V_nl, axis=0)
    sum_V_nonlocal = jnp.sum(V_nl)

    return non_local_ecp_part_r_carts_up, non_local_ecp_part_r_carts_dn, V_nonlocal, sum_V_nonlocal


@partial(jit, static_argnums=(5, 6))
def compute_ecp_non_local_parts_all_pairs_jax(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    Nv: int = Nv_default,
    flag_determinant_only: bool = False,
) -> tuple[list, list, list, float]:
    """Compute ecp non-local parts using JAX, considering all nucleus-electron pairs.

    The method is for computing the non-local part of the given ECPs at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (jnpt.ArrayLike): Rotation matrix. equiv R.T
        Nv (int): The number of quadrature points for the spherical part.
        flag_determinant_only (bool): If True, only the determinant part is considered for the non-local ECP part.

    Returns:
        list[jax.Array]: The list of grids for up electrons on which the non-local part is computed.
        list[jax.Array]: The list of grids for dn electrons on which the non-local part is computed.
        list[float]: The list of non-local part of the given ECPs with r_up_carts and r_dn_carts.
        float: The sum of non-local part of the given ECPs with r_up_carts and r_dn_carts.
    """
    if Nv == 4:
        weights = tetrahedron_sym_mesh_Nv4.weights
        grid_points = tetrahedron_sym_mesh_Nv4.grid_points
    elif Nv == 6:
        weights = octahedron_sym_mesh_Nv6.weights
        grid_points = octahedron_sym_mesh_Nv6.grid_points
    elif Nv == 12:
        weights = icosahedron_sym_mesh_Nv12.weights
        grid_points = icosahedron_sym_mesh_Nv12.grid_points
    elif Nv == 18:
        weights = octahedron_sym_mesh_Nv18.weights
        grid_points = octahedron_sym_mesh_Nv18.grid_points
    else:
        raise NotImplementedError

    grid_points = grid_points @ RT  # rotate the grid points. dim. (N,3) @ (3,3) = (N,3)

    # start = time.perf_counter()
    r_up_carts_on_mesh, r_dn_carts_on_mesh, V_ecp_up, V_ecp_dn, sum_V_nonlocal = (
        compute_ecp_non_local_part_all_pairs_jax_weights_grid_points(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            weights=weights,
            grid_points=grid_points,
            flag_determinant_only=int(flag_determinant_only),
        )
    )
    # end = time.perf_counter()
    # logger.info(f"Comput. elapsed Time = {(end-start)*1e3:.3f} msec.")

    # print(f"r_up_carts_on_mesh.shape={r_up_carts_on_mesh.shape}")
    # print(f"r_dn_carts_on_mesh.shape={r_dn_carts_on_mesh.shape}")
    # print(f"V_ecp_up.shape={V_ecp_up.shape}")
    # print(f"V_ecp_dn.shape={V_ecp_dn.shape}")

    # start = time.perf_counter()
    _, uq_indices = np.unique(coulomb_potential_data.nucleus_index_non_local_part, return_index=True)
    r_up_carts_on_mesh = r_up_carts_on_mesh[uq_indices]
    r_dn_carts_on_mesh = r_dn_carts_on_mesh[uq_indices]
    # end = time.perf_counter()
    # logger.info(f"Extract unique indices elapsed Time = {(end-start)*1e3:.3f} msec.")

    # start = time.perf_counter()
    nucleus_index_non_local_part = np.array(coulomb_potential_data.nucleus_index_non_local_part, dtype=np.int32)
    num_segments = len(set(coulomb_potential_data.nucleus_index_non_local_part))
    V_ecp_up = jax.ops.segment_sum(V_ecp_up, nucleus_index_non_local_part, num_segments=num_segments)
    V_ecp_dn = jax.ops.segment_sum(V_ecp_dn, nucleus_index_non_local_part, num_segments=num_segments)
    # end = time.perf_counter()
    # logger.info(f"Segment sum elapsed Time = {(end-start)*1e3:.3f} msec.")

    # print(f"r_up_carts_on_mesh.shape={r_up_carts_on_mesh.shape}")
    # print(f"r_dn_carts_on_mesh.shape={r_dn_carts_on_mesh.shape}")
    # print(f"V_ecp_up.shape={V_ecp_up.shape}")
    # print(f"V_ecp_dn.shape={V_ecp_dn.shape}")

    # start = time.perf_counter()
    r_up_new_shape = (np.prod(r_up_carts_on_mesh.shape[:3]),) + r_up_carts_on_mesh.shape[3:]
    r_up_carts_on_mesh = r_up_carts_on_mesh.reshape(r_up_new_shape)
    r_dn_new_shape = (np.prod(r_dn_carts_on_mesh.shape[:3]),) + r_dn_carts_on_mesh.shape[3:]
    r_dn_carts_on_mesh = r_dn_carts_on_mesh.reshape(r_dn_new_shape)

    V_ecp_up_new_shape = (np.prod(V_ecp_up.shape[:3]),)
    V_ecp_up = V_ecp_up.reshape(V_ecp_up_new_shape)
    V_ecp_dn_new_shape = (np.prod(V_ecp_dn.shape[:3]),)
    V_ecp_dn = V_ecp_dn.reshape(V_ecp_dn_new_shape)
    # end = time.perf_counter()
    # logger.info(f"Reshape elapsed Time = {(end-start)*1e3:.3f} msec.")

    # print(f"r_up_carts_on_mesh.shape={r_up_carts_on_mesh.shape}")
    # print(f"r_dn_carts_on_mesh.shape={r_dn_carts_on_mesh.shape}")
    # print(f"V_ecp_up.shape={V_ecp_up.shape}")
    # print(f"V_ecp_dn.shape={V_ecp_dn.shape}")

    # start = time.perf_counter()
    # Repeat up-spin electrons for down-spin configurations
    r_up_carts_repeated_dn = jnp.repeat(
        r_up_carts[None, :, :], r_dn_carts_on_mesh.shape[0], axis=0
    )  # Shape: (num_dn_configs, N_up, 3)
    # Repeat down-spin electrons for up-spin configurations
    r_dn_carts_repeated_up = jnp.repeat(
        r_dn_carts[None, :, :], r_up_carts_on_mesh.shape[0], axis=0
    )  # Shape: (num_up_configs, N_dn, 3)
    # Combine configurations
    mesh_non_local_ecp_part_r_up_carts = jnp.concatenate(
        [r_up_carts_on_mesh, r_up_carts_repeated_dn], axis=0
    )  # Shape: (N_configs, N_up, 3)
    mesh_non_local_ecp_part_r_dn_carts = jnp.concatenate(
        [r_dn_carts_repeated_up, r_dn_carts_on_mesh], axis=0
    )  # Shape: (N_configs, N_dn, 3)

    V_nonlocal = jnp.concatenate([V_ecp_up, V_ecp_dn], axis=0)

    # end = time.perf_counter()
    # logger.info(f"Post elapsed Time = {(end-start)*1e3:.3f} msec.")

    return mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, sum_V_nonlocal


@jit  # this jit drastically accelarates the computation!
def compute_ecp_non_local_part_all_pairs_jax_weights_grid_points(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    weights: list,
    grid_points: npt.NDArray[np.float64],
    flag_determinant_only: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, float]:
    """Compute ecp non-local parts using JAX.

    The method is for computing the non-local parts of the given ECPs at (r_up_carts, r_dn_carts).
    To avoid for the nested loops, jax-vmap function (i.e. efficient vectrization for compilation) is fully
    exploitted in the method.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        weights (list[np.float]): weights for numerical integration
        grid_points (npt.NDArray[np.float64]): grid_points for numerical integration
        flag_determinant_only (int):
            If True (i.e., 1), only the determinant part is considered for the non-local ECP part.
            If False (i.e., 0), both the Jastrow and determinant part is considered for the non-local ECP part.

    Returns:
        jnpt.ArrayLike: grid points used for the up electron
        jnpt.ArrayLike: grid points used for the dn electron
        jnpt.ArrayLike: V_ecp_up for the grid points for up electron
        jnpt.ArrayLike: V_ecp_dn for the grid points for up electron
        float: The sum of non-local part of the given ECPs with r_up_carts and r_dn_carts.

    Notes:
        This part of @jit drastically accelarates the computation!
        The procesure is so complicated that one should refer to the debug version.
        to understand the flow
    """
    # V_l_cutoff = 1e-5

    weights = jnp.array(weights)
    grid_points = jnp.array(grid_points)

    jastrow_denominator = lax.switch(
        flag_determinant_only,
        (compute_Jastrow_part_jax, lambda *args, **kwargs: 1.0),
        *(wavefunction_data.jastrow_data, r_up_carts, r_dn_carts),
    )

    det_denominator = compute_det_geminal_all_elements_jax(wavefunction_data.geminal_data, r_up_carts, r_dn_carts)

    # Compute the local part. To understand the flow, please refer to the debug version.
    @jit
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
    @jit
    def compute_P_l_up(ang_mom, r_up_i, r_up_cart, i_atom, weight, vec_delta):
        rel_R_cart_min_dist = get_min_dist_rel_R_cart_jnp(
            structure_data=coulomb_potential_data.structure_data,
            r_cart=r_up_cart,
            i_atom=i_atom,
        )
        r_up_carts_on_mesh = r_up_carts
        r_up_carts_on_mesh = r_up_carts_on_mesh.at[r_up_i].set(
            r_up_cart + rel_R_cart_min_dist + jnp.linalg.norm(rel_R_cart_min_dist) * vec_delta
        )

        cos_theta_up = jnp.dot(
            -1.0 * (rel_R_cart_min_dist) / jnp.linalg.norm(rel_R_cart_min_dist),
            ((vec_delta) / jnp.linalg.norm(vec_delta)),
        )

        jastrow_numerator_up = lax.switch(
            flag_determinant_only,
            (compute_Jastrow_part_jax, lambda *args, **kwargs: 1.0),
            *(wavefunction_data.jastrow_data, r_up_carts_on_mesh, r_dn_carts),
        )

        det_numerator_up = compute_det_geminal_all_elements_jax(wavefunction_data.geminal_data, r_up_carts_on_mesh, r_dn_carts)

        wf_ratio_up = jnp.exp(jastrow_numerator_up - jastrow_denominator) * det_numerator_up / det_denominator

        P_l_up = (2 * ang_mom + 1) * jnp_legendre_tablated(ang_mom, cos_theta_up) * weight * wf_ratio_up

        return r_up_carts_on_mesh, P_l_up

    # Compute the Projection of WF. for a down electron
    # To understand the flow, please refer to the debug version.
    @jit
    def compute_P_l_dn(ang_mom, r_dn_i, r_dn_cart, i_atom, weight, vec_delta):
        rel_R_cart_min_dist = get_min_dist_rel_R_cart_jnp(
            structure_data=coulomb_potential_data.structure_data,
            r_cart=r_dn_cart,
            i_atom=i_atom,
        )
        r_dn_carts_on_mesh = r_dn_carts
        r_dn_carts_on_mesh = r_dn_carts_on_mesh.at[r_dn_i].set(
            r_dn_cart + rel_R_cart_min_dist + jnp.linalg.norm(rel_R_cart_min_dist) * vec_delta
        )

        cos_theta_dn = jnp.dot(
            -1.0 * (rel_R_cart_min_dist) / jnp.linalg.norm(rel_R_cart_min_dist),
            ((vec_delta) / jnp.linalg.norm(vec_delta)),
        )

        jastrow_numerator_dn = lax.switch(
            flag_determinant_only,
            (compute_Jastrow_part_jax, lambda *args, **kwargs: 1.0),
            *(wavefunction_data.jastrow_data, r_up_carts, r_dn_carts_on_mesh),
        )

        det_numerator_dn = compute_det_geminal_all_elements_jax(wavefunction_data.geminal_data, r_up_carts, r_dn_carts_on_mesh)

        wf_ratio_dn = jnp.exp(jastrow_numerator_dn - jastrow_denominator) * det_numerator_dn / det_denominator

        P_l_dn = (2 * ang_mom + 1) * jnp_legendre_tablated(ang_mom, cos_theta_dn) * weight * wf_ratio_dn
        return r_dn_carts_on_mesh, P_l_dn

    # Vectrize the functions
    vmap_compute_P_l_up = vmap(compute_P_l_up, in_axes=(None, None, None, None, 0, 0))
    vmap_compute_P_l_dn = vmap(compute_P_l_dn, in_axes=(None, None, None, None, 0, 0))

    # Compute the local part V_l * Projection of WF. for a up electron
    # To understand the flow, please refer to the debug version.
    # vmap in_axes=(0, 0, None, None, None, None, None) and in_axes=(None, None, 0, 0, 0, 0, 0)
    @jit
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
        """ slower...
        r_up_carts_on_mesh, P_l_up = lax.cond(
            V_l_up > V_l_cutoff,
            vmap_compute_P_l_up,
            lambda *args: (jnp.zeros((len(weights), len(r_up_carts), 3)), jnp.zeros((len(weights)))),
            *(ang_mom, r_up_i, r_up_cart, i_atom, weights, grid_points),
        )
        """
        r_up_carts_on_mesh, P_l_up = vmap_compute_P_l_up(ang_mom, r_up_i, r_up_cart, i_atom, weights, grid_points)
        return r_up_carts_on_mesh, (V_l_up * P_l_up)

    # Compute the local part V_l * Projection of WF. for a down electron
    # To understand the flow, please refer to the debug version.
    # @jit
    # vmap in_axes=(0, 0, None, None, None, None, None) and in_axes=(None, None, 0, 0, 0, 0, 0)
    @jit
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
        """ slower...
        r_dn_carts_on_mesh, P_l_dn = lax.cond(
            V_l_dn > V_l_cutoff,
            vmap_compute_P_l_dn,
            lambda *args: (jnp.zeros((len(weights), len(r_dn_carts), 3)), jnp.zeros((len(weights)))),
            *(ang_mom, r_dn_i, r_dn_cart, i_atom, weights, grid_points),
        )
        """
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

    i_atom_np = jnp.array(coulomb_potential_data.nucleus_index_non_local_part)
    ang_mom_np = jnp.array(coulomb_potential_data.ang_mom_non_local_part)
    exponent_np = jnp.array(coulomb_potential_data.exponents_non_local_part)
    coefficient_np = jnp.array(coulomb_potential_data.coefficients_non_local_part)
    power_np = jnp.array(coulomb_potential_data.powers_non_local_part)

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


def compute_ecp_coulomb_potential_jax(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    RT: jnpt.ArrayLike = jnp.eye(3),
    NN: int = NN_default,
    Nv: int = Nv_default,
) -> float:
    """Compute effective core potential term.

    The method is for computing the local and non-local parts of the given ECPs at
    a given electronic configuration (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (jnpt.ArrayLike): Rotation matrix. equiv R.T used for the non-local part.
        NN (int): Consider only up to NN-th nearest neighbors.
        Nv (int): The number of quadrature points for the spherical part.


    Returns:
        float: The sum of local and non-local parts of the given ECPs with r_up_carts and r_dn_carts. (float)
    """
    ecp_local_parts = compute_ecp_local_parts_all_pairs_jax(
        coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    """ full NNs
    _, _, _, ecp_nonlocal_parts = _compute_ecp_non_local_parts_full_NN_jax(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        Nv=Nv,
        RT=RT,
        flag_determinant_only=False,
    )
    """

    #''' NNs
    _, _, _, ecp_nonlocal_parts = compute_ecp_non_local_parts_nearest_neighbors_jax(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
        Nv=Nv,
        NN=NN,
        flag_determinant_only=False,
    )
    #'''

    V_ecp = ecp_local_parts + ecp_nonlocal_parts

    return V_ecp


def compute_bare_coulomb_potential_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """See compute_bare_coulomb_potential_api."""
    R_carts = coulomb_potential_data.structure_data.positions_cart_np
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
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> float:
    """Compute bare Coulomb potential.

    The method is for computing the bare coulomb potentials including both electron-electron,
    electron-ion (except. ECPs), and ion-ion interactions at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Bare_coulomb_potential_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        debug (bool): if True, the value is computed via _debug function for debuging purpose

    Returns:
        The bare Coulomb potential with r_up_carts and r_dn_carts. (float)
    """
    interactions_ion_ion = compute_bare_coulomb_potential_ion_ion_jax(coulomb_potential_data)
    interactions_el_ion_elements_up, interactions_el_ion_elements_dn = compute_bare_coulomb_potential_el_ion_element_wise_jax(
        coulomb_potential_data, r_up_carts, r_dn_carts
    )
    interactions_el_el = compute_bare_coulomb_potential_el_el_jax(r_up_carts, r_dn_carts)

    return (
        interactions_ion_ion
        + jnp.sum(interactions_el_ion_elements_up)
        + jnp.sum(interactions_el_ion_elements_dn)
        + interactions_el_el
    )


@jit
def compute_bare_coulomb_potential_el_ion_element_wise_jax(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """See compute_bare_coulomb_potential_api."""
    R_carts = jnp.array(coulomb_potential_data.structure_data.positions_cart_jnp)
    R_charges = np.array(coulomb_potential_data.effective_charges)
    r_up_charges = np.full(len(r_up_carts), -1.0, dtype=np.float64)
    r_dn_charges = np.full(len(r_dn_carts), -1.0, dtype=np.float64)

    r_up_carts = jnp.array(r_up_carts)
    r_dn_carts = jnp.array(r_dn_carts)

    # Define a function to compute interaction for a pair
    def el_ion_interaction(Z_i, Z_j, r_i, r_j):
        distance = jnp.linalg.norm(r_i - r_j, axis=1)
        interaction = (Z_i * Z_j) / distance
        return interaction

    # Vectorize the function over all pairs
    interactions_R_r_up = jnp.sum(
        jax.vmap(el_ion_interaction, in_axes=(None, 0, None, 0))(R_charges, r_up_charges, R_carts, r_up_carts), axis=1
    )
    interactions_R_r_dn = jnp.sum(
        jax.vmap(el_ion_interaction, in_axes=(None, 0, None, 0))(R_charges, r_dn_charges, R_carts, r_dn_carts), axis=1
    )

    return interactions_R_r_up, interactions_R_r_dn


@jit
def compute_discretized_bare_coulomb_potential_el_ion_element_wise_jax(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    alat: float,
) -> tuple[jax.Array, jax.Array]:
    """See compute_bare_coulomb_potential_api."""
    R_carts = jnp.array(coulomb_potential_data.structure_data.positions_cart_jnp)
    R_charges = np.array(coulomb_potential_data.effective_charges)
    r_up_charges = np.full(len(r_up_carts), -1.0, dtype=np.float64)
    r_dn_charges = np.full(len(r_dn_carts), -1.0, dtype=np.float64)

    r_up_carts = jnp.array(r_up_carts)
    r_dn_carts = jnp.array(r_dn_carts)

    # Define a function to compute interaction for a pair
    def el_ion_interaction(Z_i, Z_j, r_i, r_j, alat):
        distance = jnp.maximum(jnp.linalg.norm(r_i - r_j, axis=1), alat)
        interaction = (Z_i * Z_j) / distance
        return interaction

    # Vectorize the function over all pairs
    interactions_R_r_up = jnp.sum(
        jax.vmap(el_ion_interaction, in_axes=(None, 0, None, 0, None))(R_charges, r_up_charges, R_carts, r_up_carts, alat),
        axis=1,
    )
    interactions_R_r_dn = jnp.sum(
        jax.vmap(el_ion_interaction, in_axes=(None, 0, None, 0, None))(R_charges, r_dn_charges, R_carts, r_dn_carts, alat),
        axis=1,
    )

    return interactions_R_r_up, interactions_R_r_dn


def compute_bare_coulomb_potential_el_ion_element_wise_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """See compute_bare_coulomb_potential_api."""
    R_carts = coulomb_potential_data.structure_data.positions_cart_np
    R_charges = coulomb_potential_data.effective_charges
    r_up_charges = [-1 for _ in range(len(r_up_carts))]
    r_dn_charges = [-1 for _ in range(len(r_dn_carts))]

    interactions_R_r_up = np.zeros(len(r_up_carts))
    interactions_R_r_dn = np.zeros(len(r_dn_carts))

    for i, (r_up_charge, r_up_cart) in enumerate(zip(r_up_charges, r_up_carts)):
        interactions_R_r_up[i] = np.sum(
            [(R_charge * r_up_charge) / np.linalg.norm(R_cart - r_up_cart) for R_charge, R_cart in zip(R_charges, R_carts)]
        )

    for i, (r_dn_charge, r_dn_cart) in enumerate(zip(r_dn_charges, r_dn_carts)):
        interactions_R_r_dn[i] = np.sum(
            [(R_charge * r_dn_charge) / np.linalg.norm(R_cart - r_dn_cart) for R_charge, R_cart in zip(R_charges, R_carts)]
        )

    return interactions_R_r_up, interactions_R_r_dn


def compute_discretized_bare_coulomb_potential_el_ion_element_wise_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    alat: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """See compute_bare_coulomb_potential_api."""
    R_carts = coulomb_potential_data.structure_data.positions_cart_np
    R_charges = coulomb_potential_data.effective_charges
    r_up_charges = [-1 for _ in range(len(r_up_carts))]
    r_dn_charges = [-1 for _ in range(len(r_dn_carts))]

    interactions_R_r_up = np.zeros(len(r_up_carts))
    interactions_R_r_dn = np.zeros(len(r_dn_carts))

    for i, (r_up_charge, r_up_cart) in enumerate(zip(r_up_charges, r_up_carts)):
        interactions_R_r_up[i] = np.sum(
            [
                (R_charge * r_up_charge) / np.maximum(np.linalg.norm(R_cart - r_up_cart), alat)
                for R_charge, R_cart in zip(R_charges, R_carts)
            ]
        )

    for i, (r_dn_charge, r_dn_cart) in enumerate(zip(r_dn_charges, r_dn_carts)):
        interactions_R_r_dn[i] = np.sum(
            [
                (R_charge * r_dn_charge) / np.maximum(np.linalg.norm(R_cart - r_dn_cart), alat)
                for R_charge, R_cart in zip(R_charges, R_carts)
            ]
        )

    return interactions_R_r_up, interactions_R_r_dn


@jit
def compute_bare_coulomb_potential_el_el_jax(
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> float:
    """See compute_bare_coulomb_potential_api."""
    r_up_charges = np.full(len(r_up_carts), -1.0, dtype=np.float64)
    r_dn_charges = np.full(len(r_dn_carts), -1.0, dtype=np.float64)

    r_up_carts = jnp.array(r_up_carts)
    r_dn_carts = jnp.array(r_dn_carts)

    all_charges = np.hstack([r_up_charges, r_dn_charges])
    all_carts = jnp.vstack([r_up_carts, r_dn_carts])

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
    def el_el_interaction(Z_i, Z_j, r_i, r_j):
        distance = jnp.linalg.norm(r_i - r_j)
        interaction = (Z_i * Z_j) / distance
        return interaction

    # Vectorize the function over all pairs
    interactions = jax.vmap(el_el_interaction)(Z_i, Z_j, r_i, r_j)  # Shape: (M,)

    # Sum all interactions
    bare_coulomb_potential_el_el = jnp.sum(interactions)

    return bare_coulomb_potential_el_el


@jit
def compute_bare_coulomb_potential_ion_ion_jax(
    coulomb_potential_data: Coulomb_potential_data,
) -> float:
    """See compute_bare_coulomb_potential_api."""
    R_carts = jnp.array(coulomb_potential_data.structure_data.positions_cart_jnp)
    R_charges = np.array(coulomb_potential_data.effective_charges)

    all_charges = R_charges
    all_carts = R_carts

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
    def ion_ion_interaction(Z_i, Z_j, r_i, r_j):
        distance = jnp.linalg.norm(r_i - r_j)
        interaction = (Z_i * Z_j) / distance
        return interaction

    # Vectorize the function over all pairs
    interactions = jax.vmap(ion_ion_interaction)(Z_i, Z_j, r_i, r_j)  # Shape: (M,)

    # Sum all interactions
    bare_coulomb_potential_ion_ion = jnp.sum(interactions)

    return bare_coulomb_potential_ion_ion


def compute_coulomb_potential_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    RT: npt.NDArray = np.eye(3),
    NN: int = NN_default,
    Nv: int = Nv_default,
    wavefunction_data: Wavefunction_data = None,
) -> float:
    """See compute_coulomb_potential_api."""
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
            RT=RT,
            Nv=Nv,
            NN=NN,
        )

    return bare_coulomb_potential + ecp_coulomb_potential


def compute_coulomb_potential_jax(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    RT: jnpt.ArrayLike = jnp.eye(3),
    NN: int = NN_default,
    Nv: int = Nv_default,
    wavefunction_data: Wavefunction_data = None,
) -> float:
    """Compute coulomb potential including bare electron-ion, electron-electron, ecp local and non-local parts.

    The method is for computing coulomb potential including bare electron-ion, electron-electron,
    ecp local and non-local parts, of the given ECPs at (r_up_carts, r_dn_carts).

    Args:
        coulomb_potential_data (Coulomb_potential_data):
            an instance of Coulomb_potential_data
        r_up_carts (jnpt.ArrayLike):
            Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike):
            Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (jnpt.ArrayLike): Rotation matrix. equiv R.T used for non-local part
        NN (int): Consider only up to NN-th nearest neighbors.
        Nv (int): The number of quadrature points for the spherical part.
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data

    Returns:
        float:  The sum of bare electron-ion, electron-electron, local and non-local parts of the given
                ECPs with r_up_carts and r_dn_carts. (float)
    """
    # all-electron
    if not coulomb_potential_data.ecp_flag:
        bare_coulomb_potential = compute_bare_coulomb_potential_jax(
            coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )
        ecp_coulomb_potential = 0

    # pseudo-potential
    else:
        bare_coulomb_potential = compute_bare_coulomb_potential_jax(
            coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )

        ecp_coulomb_potential = compute_ecp_coulomb_potential_jax(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            wavefunction_data=wavefunction_data,
            RT=RT,
            NN=NN,
            Nv=Nv,
        )

    return bare_coulomb_potential + ecp_coulomb_potential


if __name__ == "__main__":
    import pickle

    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    """
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
    """

    hamiltonian_chk = "hamiltonian_data_water.chk"
    # hamiltonian_chk = "hamiltonian_data_AcOH.chk"
    # hamiltonian_chk = "hamiltonian_data_benzene.chk"
    # hamiltonian_chk = "hamiltonian_data_C60.chk"
    with open(hamiltonian_chk, "rb") as f:
        hamiltonian_data = pickle.load(f)

    structure_data = hamiltonian_data.structure_data
    coulomb_potential_data = hamiltonian_data.coulomb_potential_data
    wavefunction_data = hamiltonian_data.wavefunction_data
    num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
    num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn

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

    r_up_carts_jnp = jnp.array(r_up_carts)
    r_dn_carts_jnp = jnp.array(r_dn_carts)

    """
    # Full WF, Full-neighbours
    mesh_non_local_ecp_part_r_up_carts_jax, mesh_non_local_ecp_part_r_dn_carts_jax, V_nonlocal_jax, sum_V_nonlocal_jax = (
        _compute_ecp_non_local_parts_jax(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            Nv=6,
            flag_determinant_only=False,
        )
    )

    (
        mesh_non_local_ecp_part_r_up_carts_debug,
        mesh_non_local_ecp_part_r_dn_carts_debug,
        V_nonlocal_debug,
        sum_V_nonlocal_debug,
    ) = _compute_ecp_non_local_parts_debug(
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

    mesh_non_local_r_up_carts_max_debug = mesh_non_local_ecp_part_r_up_carts_debug[np.argmax(V_nonlocal_debug)]
    mesh_non_local_r_up_carts_max_jax = mesh_non_local_ecp_part_r_up_carts_jax[np.argmax(V_nonlocal_jax)]

    np.testing.assert_array_almost_equal(mesh_non_local_r_up_carts_max_debug, mesh_non_local_r_up_carts_max_jax, decimal=5)

    # Only determinant, Full-neighbours
    (
        mesh_non_local_ecp_part_r_up_carts_only_det_jax,
        mesh_non_local_ecp_part_r_dn_carts_only_det_jax,
        V_nonlocal_only_det_jax,
        sum_V_nonlocal_only_det_jax,
    ) = _compute_ecp_non_local_parts_jax(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        Nv=6,
        flag_determinant_only=True,
    )

    (
        mesh_non_local_ecp_part_r_up_carts_only_det_debug,
        mesh_non_local_ecp_part_r_dn_carts_only_det_debug,
        V_nonlocal_only_det_debug,
        sum_V_nonlocal_only_det_debug,
    ) = _compute_ecp_non_local_parts_debug(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        Nv=6,
        flag_determinant_only=True,
    )

    np.testing.assert_almost_equal(sum_V_nonlocal_only_det_jax, sum_V_nonlocal_only_det_debug, decimal=5)
    """

    # Full WF, NN Nearest-neighbours
    (
        mesh_non_local_ecp_part_r_up_carts_NN_debug,
        mesh_non_local_ecp_part_r_dn_carts_NN_debug,
        V_nonlocal_NN_debug,
        sum_V_nonlocal_NN_debug,
    ) = compute_ecp_non_local_parts_nearest_neighbors_debug(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        Nv=6,
        flag_determinant_only=False,
    )

    (
        mesh_non_local_ecp_part_r_up_carts_NN_jax,
        mesh_non_local_ecp_part_r_dn_carts_NN_jax,
        V_nonlocal_NN_jax,
        sum_V_nonlocal_NN_jax,
    ) = compute_ecp_non_local_parts_nearest_neighbors_jax(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
        Nv=6,
        flag_determinant_only=False,
    )
    mesh_non_local_ecp_part_r_up_carts_NN_jax.block_until_ready()
    mesh_non_local_ecp_part_r_dn_carts_NN_jax.block_until_ready()
    V_nonlocal_NN_jax.block_until_ready()
    sum_V_nonlocal_NN_jax.block_until_ready()

    start = time.perf_counter()
    (
        mesh_non_local_ecp_part_r_up_carts_NN_jax,
        mesh_non_local_ecp_part_r_dn_carts_NN_jax,
        V_nonlocal_NN_jax,
        sum_V_nonlocal_NN_jax,
    ) = compute_ecp_non_local_parts_nearest_neighbors_jax(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
        Nv=6,
        flag_determinant_only=False,
    )
    mesh_non_local_ecp_part_r_up_carts_NN_jax.block_until_ready()
    mesh_non_local_ecp_part_r_dn_carts_NN_jax.block_until_ready()
    V_nonlocal_NN_jax.block_until_ready()
    sum_V_nonlocal_NN_jax.block_until_ready()
    end = time.perf_counter()
    print(f"Total elapsed Time = {(end - start) * 1e3:.3f} msec.")

    np.testing.assert_almost_equal(sum_V_nonlocal_NN_debug, sum_V_nonlocal_NN_jax, decimal=6)

    mesh_non_local_r_up_carts_max_NN_jax = mesh_non_local_ecp_part_r_up_carts_NN_jax[np.argmax(V_nonlocal_NN_jax)]
    mesh_non_local_r_up_carts_max_NN_debug = mesh_non_local_ecp_part_r_up_carts_NN_debug[np.argmax(V_nonlocal_NN_debug)]
    V_ecp_non_local_max_NN_jax = V_nonlocal_NN_jax[np.argmax(V_nonlocal_NN_jax)]
    V_ecp_non_local_max_NN_debug = V_nonlocal_NN_debug[np.argmax(V_nonlocal_NN_debug)]

    np.testing.assert_almost_equal(V_ecp_non_local_max_NN_jax, V_ecp_non_local_max_NN_debug, decimal=6)
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_up_carts_max_NN_jax, mesh_non_local_r_up_carts_max_NN_debug, decimal=6
    )

    mesh_non_local_ecp_part_r_up_carts_jax, mesh_non_local_ecp_part_r_dn_carts_jax, V_nonlocal_jax, sum_V_nonlocal_jax = (
        compute_ecp_non_local_parts_all_pairs_jax(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts_jnp,
            r_dn_carts=r_dn_carts_jnp,
            Nv=6,
            flag_determinant_only=False,
        )
    )
    mesh_non_local_ecp_part_r_up_carts_jax.block_until_ready()
    mesh_non_local_ecp_part_r_dn_carts_jax.block_until_ready()
    V_nonlocal_jax.block_until_ready()
    sum_V_nonlocal_jax.block_until_ready()

    start = time.perf_counter()
    mesh_non_local_ecp_part_r_up_carts_jax, mesh_non_local_ecp_part_r_dn_carts_jax, V_nonlocal_jax, sum_V_nonlocal_jax = (
        compute_ecp_non_local_parts_all_pairs_jax(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts_jnp,
            r_dn_carts=r_dn_carts_jnp,
            Nv=6,
            flag_determinant_only=False,
        )
    )
    mesh_non_local_ecp_part_r_up_carts_jax.block_until_ready()
    mesh_non_local_ecp_part_r_dn_carts_jax.block_until_ready()
    V_nonlocal_jax.block_until_ready()
    sum_V_nonlocal_jax.block_until_ready()
    end = time.perf_counter()
    print(f"Total elapsed Time = {(end - start) * 1e3:.3f} msec.")
