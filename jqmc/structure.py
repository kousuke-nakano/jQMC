"""Structure module."""

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

# python modules
import itertools
from functools import partial
from logging import getLogger

# JAX
import jax
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import jit
from jax import numpy as jnp
from jax import typing as jnpt
from numpy import linalg as LA

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)

# separator
num_sep_line = 50


@struct.dataclass
class Structure_data:
    """Atomic structure and cell metadata.

    Stores Cartesian coordinates (Bohr), optional lattice vectors for PBC, and basic
    element metadata used by AOs/MOs, Coulomb, and Hamiltonian builders.

    Attributes:
        positions (npt.NDArray | jax.Array): Atomic Cartesian coordinates with shape ``(N, 3)`` in Bohr.
        pbc_flag (bool): Whether periodic boundary conditions are active. If ``True``, lattice
            vectors ``vec_a|b|c`` must be provided; otherwise they must be empty.
        vec_a (list[float] | tuple[float]): Lattice vector **a** (Bohr) when ``pbc_flag=True``.
        vec_b (list[float] | tuple[float]): Lattice vector **b** (Bohr) when ``pbc_flag=True``.
        vec_c (list[float] | tuple[float]): Lattice vector **c** (Bohr) when ``pbc_flag=True``.
        atomic_numbers (list[int] | tuple[int]): Atomic numbers ``Z`` for each site (len ``N``).
        element_symbols (list[str] | tuple[str]): Element symbols for each site (len ``N``).
        atomic_labels (list[str] | tuple[str]): Human-readable labels for each site (len ``N``).

    Examples:
        Minimal H2 setup (Bohr)::

            import numpy as np
            from jqmc.structure import Structure_data

            structure = Structure_data(
                positions=np.array([[0.0, 0.0, -0.70], [0.0, 0.0, 0.70]]),
                pbc_flag=False,
                atomic_numbers=[1, 1],
                element_symbols=["H", "H"],
                atomic_labels=["H1", "H2"],
            )
            structure.sanity_check()
    """

    #: Atomic Cartesian coordinates with shape ``(N, 3)`` in Bohr.
    positions: npt.NDArray | jnpt.ArrayLike = struct.field(pytree_node=True, default_factory=lambda: np.array([]))
    #: Whether periodic boundary conditions are active.
    pbc_flag: bool = struct.field(pytree_node=False, default=False)
    #: Lattice vector **a** in Bohr (requires ``pbc_flag=True``).
    vec_a: list[float] | tuple[float] = struct.field(pytree_node=False, default_factory=tuple)
    #: Lattice vector **b** in Bohr (requires ``pbc_flag=True``).
    vec_b: list[float] | tuple[float] = struct.field(pytree_node=False, default_factory=tuple)
    #: Lattice vector **c** in Bohr (requires ``pbc_flag=True``).
    vec_c: list[float] | tuple[float] = struct.field(pytree_node=False, default_factory=tuple)
    #: Atomic numbers ``Z`` per site (len ``N``).
    atomic_numbers: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    #: Element symbols per site (len ``N``).
    element_symbols: list[str] | tuple[str] = struct.field(pytree_node=False, default_factory=tuple)
    #: Human-readable labels per site (len ``N``).
    atomic_labels: list[str] | tuple[str] = struct.field(pytree_node=False, default_factory=tuple)

    def sanity_check(self) -> None:
        """Validate consistency of positions, labels, and lattice metadata.

        Ensures all per-atom arrays share length ``N``, lattice vectors are either all
        present (length 3 each) when ``pbc_flag=True`` or all empty when ``pbc_flag=False``,
        and basic types are as expected.

        Raises:
            ValueError: If list lengths mismatch, lattice vectors are inconsistent with
                ``pbc_flag``, or field types are incorrect.
        """
        if len(self.element_symbols) != len(self.atomic_numbers):
            raise ValueError("The length of element_symbols and atomic_numbers must be the same.")
        if len(self.atomic_labels) != len(self.atomic_numbers):
            raise ValueError("The length of atomic_labels and atomic_numbers must be the same.")
        if len(self.positions) != len(self.atomic_numbers):
            raise ValueError("The length of positions and atomic_numbers must be the same.")
        if not isinstance(self.pbc_flag, bool):
            raise ValueError("The pbc_flag must be a boolen.")
        if self.pbc_flag:
            if len(self.vec_a) != 3 or len(self.vec_b) != 3 or len(self.vec_c) != 3:
                raise ValueError("The length of lattice vectors must be 3.")
        else:
            if len(self.vec_a) != 0 or len(self.vec_b) != 0 or len(self.vec_c) != 0:
                raise ValueError("The lattice vectors must be empty.")

        if not isinstance(self.pbc_flag, bool):
            raise ValueError(f"pbc_flag = {type(self.pbc_flag)} must be a boolen.")
        if not isinstance(self.vec_a, (list, tuple)):
            raise ValueError(f"vec_a = {type(self.vec_a)} must be a list or tuple.")
        if not isinstance(self.vec_b, (list, tuple)):
            raise ValueError(f"vec_b = {type(self.vec_b)} must be a list or tuple.")
        if not isinstance(self.vec_c, (list, tuple)):
            raise ValueError(f"vec_c = {type(self.vec_c)} must be a list or tuple.")
        if not isinstance(self.atomic_numbers, (list, tuple)):
            raise ValueError(f"atomic_numbers = {type(self.atomic_numbers)} must be a list or tuple.")
        if not isinstance(self.element_symbols, (list, tuple)):
            raise ValueError(f"element_symbols = {type(self.element_symbols)} must be a list or tuple.")
        if not isinstance(self.atomic_labels, (list, tuple)):
            raise ValueError(f"atomic_labels = {type(self.atomic_labels)} must be a list or tuple.")

    def _get_info(self) -> list[str]:
        """Return human-readable summary lines for logging."""
        info_lines = []
        info_lines.extend(["**" + self.__class__.__name__])
        info_lines.extend([f"  PBC flag = {self.pbc_flag}"])
        if self.pbc_flag:
            info_lines.extend([f"  vec A = {self.vec_a} Bohr"])
            info_lines.extend([f"  vec B = {self.vec_b} Bohr"])
            info_lines.extend([f"  vec C = {self.vec_c} Bohr"])
        info_lines.extend(["  " + "-" * num_sep_line])
        info_lines.extend(["  element, label, Z, x, y, z in cartesian (Bohr)"])
        info_lines.extend(["  " + "-" * num_sep_line])
        for atomic_number, element_symbol, atomic_label, position in zip(
            self.atomic_numbers, self.element_symbols, self.atomic_labels, self._positions_cart_np, strict=True
        ):
            info_lines.extend(
                [
                    f"  {element_symbol:s}, {atomic_label:s}, {atomic_number:.1f}, "
                    f"{position[0]:.8f}, {position[1]:.8f}, {position[2]:.8f}"
                ]
            )
        info_lines.extend(["  " + "-" * num_sep_line])
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @property
    def cell(self) -> npt.NDArray[np.float64]:
        """Lattice vectors as a ``(3, 3)`` matrix in Bohr (``[a, b, c]``)."""
        cell = np.array([self.vec_a, self.vec_b, self.vec_c])
        return cell

    @property
    def recip_cell(self) -> npt.NDArray[np.float64]:
        r"""Reciprocal lattice vectors ``(3, 3)`` in Bohr^{-1}.

        Uses the standard definition

        .. math::
           G_a = 2\pi \frac{T_b \times T_c}{T_a \cdot (T_b \times T_c)},
           \quad G_b = 2\pi \frac{T_c \times T_a}{T_b \cdot (T_c \times T_a)},
           \quad G_c = 2\pi \frac{T_a \times T_b}{T_c \cdot (T_a \times T_b)},

        and asserts the orthonormality condition :math:`T_i \cdot G_j = 2\pi\,\delta_{ij}`.
        """
        recip_a = 2 * np.pi * (np.cross(self.vec_b, self.vec_c)) / (np.dot(self.vec_a, np.cross(self.vec_b, self.vec_c)))
        recip_b = 2 * np.pi * (np.cross(self.vec_c, self.vec_a)) / (np.dot(self.vec_b, np.cross(self.vec_c, self.vec_a)))
        recip_c = 2 * np.pi * (np.cross(self.vec_a, self.vec_b)) / (np.dot(self.vec_c, np.cross(self.vec_a, self.vec_b)))

        # check if the implementations are correct
        lattice_vec_list = [self.vec_a, self.vec_b, self.vec_c]
        recip_vec_list = [recip_a, recip_b, recip_c]
        for (lattice_vec_i, lattice_vec), (recip_vec_j, recip_vec) in itertools.product(
            enumerate(lattice_vec_list), enumerate(recip_vec_list)
        ):
            if lattice_vec_i == recip_vec_j:
                np.testing.assert_almost_equal(np.dot(lattice_vec, recip_vec), 2 * np.pi, decimal=15)
            else:
                np.testing.assert_almost_equal(np.dot(lattice_vec, recip_vec), 0.0, decimal=15)

        recip_cell = np.array([recip_a, recip_b, recip_c])
        return recip_cell

    @property
    def lattice_vec_a(self) -> tuple:
        """Return lattice vector A (in Bohr).

        Returns:
            tuple[np.float64]: the lattice vector A (in Bohr).

        """
        return tuple(self.cell[0])

    @property
    def lattice_vec_b(self) -> tuple:
        """Return lattice vector B (in Bohr).

        Returns:
            tuple[np.float64]: the lattice vector B (in Bohr).

        """
        return tuple(self.cell[1])

    @property
    def lattice_vec_c(self) -> tuple:
        """Return lattice vector C (in Bohr).

        Returns:
            tuple[np.float64]: the lattice vector C (in Bohr).

        """
        return tuple(self.cell[2])

    @property
    def recip_vec_a(self) -> tuple:
        """Return reciprocal lattice vector A (in Bohr).

        Returns:
            tuple[np.float64]: the reciprocal lattice vector A (in Bohr).

        """
        return tuple(self.recip_cell[0])

    @property
    def recip_vec_b(self) -> tuple:
        """Return reciprocal lattice vector B (in Bohr).

        Returns:
            tuple[np.float64]: the reciprocal lattice vector B (in Bohr).

        """
        return tuple(self.recip_cell[1])

    @property
    def recip_vec_c(self) -> tuple:
        """Return reciprocal lattice vector C (in Bohr).

        Returns:
            tuple[np.float64]: the reciprocal lattice vector C (in Bohr).

        """
        return tuple(self.recip_cell[2])

    @property
    def norm_vec_a(self) -> float:
        """Return the norm of the lattice vector A (in Bohr).

        Returns:
            np.float64: the norm of the lattice vector A (in Bohr).

        """
        return LA.norm(self.vec_a)

    @property
    def norm_vec_b(self) -> float:
        """Return the norm of the lattice vector B (in Bohr).

        Returns:
            np.float64: the norm of the lattice vector C (in Bohr).

        """
        return LA.norm(self.vec_b)

    @property
    def norm_vec_c(self) -> float:
        """Return the norm of the lattice vector C (in Bohr).

        Returns:
            np.float64: the norm of the lattice vector C (in Bohr).

        """
        return LA.norm(self.vec_c)

    @property
    def _positions_cart_np(self) -> npt.NDArray[np.float64]:
        """Atomic positions as ``numpy.ndarray`` with shape ``(N, 3)`` in Bohr."""
        return np.array(self.positions)

    @property
    def _positions_cart_jnp(self) -> jax.Array:
        """Atomic positions as ``jax.Array`` with shape ``(N, 3)`` in Bohr."""
        return jnp.array(self.positions)

    @property
    def _positions_frac_np(self) -> npt.NDArray[np.float64]:
        """Fractional (crystal) coordinates as ``numpy.ndarray`` with shape ``(N, 3)``."""
        h = np.array([self.vec_a, self.vec_b, self.vec_c])
        positions_frac = np.array([np.dot(np.array(pos), np.linalg.inv(h)) for pos in self._positions_cart_np])
        return positions_frac

    @property
    def natom(self) -> int:
        """The number of atoms in the system.

        Returns:
            int:The number of atoms in the system.
        """
        return len(self.atomic_numbers)

    @property
    def ntyp(self) -> int:
        """The number of element types in the system.

        Returns:
            int: The number of element types in the system.
        """
        return len(list(set(self.atomic_numbers)))


def _find_nearest_index_np(structure: Structure_data, r_cart: list[float]) -> int:
    """Find the nearest atom index for the give position.

    Args:
        structure (Structure_data): an instance of Structure_data
        r_cart (list[float, float, float]): reference position (in Bohr)

    Return:
        int: The index of the nearest neigbhor nucleus

    Todo:
        Implementing PBC (i.e., considering mirror images).
    """
    return _find_nearest_nucleus_indices_np(structure, r_cart, 1)[0]


def _find_nearest_index_jnp(structure: Structure_data, r_cart: list[float]) -> int:
    """Find the nearest atom index for the give position.

    Args:
        structure (Structure_data): an instance of Structure_data
        r_cart (list[float, float, float]): reference position (in Bohr)

    Return:
        int: The index of the nearest neigbhor nucleus

    Todo:
        Implementing PBC (i.e., considering mirror images).
    """
    return _find_nearest_nucleus_indices_jnp(structure, r_cart, 1)[0]


def _find_nearest_nucleus_indices_np(structure_data: Structure_data, r_cart, N):
    """See find_nearest_index."""
    positions = structure_data._positions_cart_np
    r_cart = np.array(r_cart)
    diffs = positions - r_cart
    if structure_data.pbc_flag:
        cell = structure_data.cell
        inv_cell = np.linalg.inv(cell)
        diffs_frac = diffs @ inv_cell
        diffs_frac = diffs_frac - np.round(diffs_frac)
        diffs = diffs_frac @ cell

    distances = np.sqrt(np.sum(diffs**2, axis=1))
    nearest_indices = np.argsort(distances)
    return nearest_indices[:N]


@partial(jit, static_argnums=2)
def _find_nearest_nucleus_indices_jnp(structure_data: Structure_data, r_cart, N):
    """See find_nearest_index."""
    positions = structure_data._positions_cart_jnp
    r_cart = jnp.array(r_cart)
    diffs = positions - r_cart
    if structure_data.pbc_flag:
        cell = jnp.array(structure_data.cell)
        inv_cell = jnp.linalg.inv(cell)
        diffs_frac = diffs @ inv_cell
        diffs_frac = diffs_frac - jnp.round(diffs_frac)
        diffs = diffs_frac @ cell

    distances = jnp.sqrt(jnp.sum(diffs**2, axis=1))
    nearest_indices = jnp.argsort(distances)
    return nearest_indices[:N]


def _get_min_dist_rel_R_cart_np(structure_data: Structure_data, r_cart: list[float, float, float], i_atom: int) -> float:
    """Minimum-distance atomic position with respect to the given r_cart.

    Args:
        structure (Structure_data): an instance of Structure_data
        r_cart (list[float, float, float]): reference position (in Bohr)
        int: the index of the target atom

    Returns:
        npt.NDAarray: rel_R_cart_min_dist containing minimum-distance atomic positions
        with respect to the given r_cart in cartesian. The unit is Bohr

    """
    r_cart = np.array(r_cart)
    R_cart = structure_data._positions_cart_np[i_atom]
    diff = R_cart - r_cart
    if structure_data.pbc_flag:
        cell = structure_data.cell
        inv_cell = np.linalg.inv(cell)
        diff_frac = diff @ inv_cell
        diff_frac = diff_frac - np.round(diff_frac)
        diff = diff_frac @ cell
    return diff


@jit
def _get_min_dist_rel_R_cart_jnp(structure_data: Structure_data, r_cart: list[float, float, float], i_atom: int) -> float:
    """See get_min_dist_rel_R_cart_np."""
    r_cart = jnp.array(r_cart)
    R_cart = jnp.array(structure_data._positions_cart_jnp[i_atom])
    diff = R_cart - r_cart
    if structure_data.pbc_flag:
        cell = jnp.array(structure_data.cell)
        inv_cell = jnp.linalg.inv(cell)
        diff_frac = diff @ inv_cell
        diff_frac = diff_frac - jnp.round(diff_frac)
        diff = diff_frac @ cell
    return diff


"""
if __name__ == "__main__":
    import os

    from .trexio_wrapper import read_trexio_file

    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
"""
