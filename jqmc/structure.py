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
from logging import Formatter, StreamHandler, getLogger

# JAX
import jax
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import jit, lax
from jax import numpy as jnp
from numpy import linalg as LA

# modules
from .units import Bohr_to_Angstrom

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)

# separator
num_sep_line = 50


@struct.dataclass
class Structure_data:
    """Structure class.

    This class contains information about the given structure.

    Args:
        positions (npt.NDArray[np.float64]):
            (N x 3) np.array containing atomic positions in cartesian. The unit is Bohr
        pbc_flag (list[bool]): pbc_flags in the a, b, and c directions.
        vec_a (list[float]): lattice vector a. The unit is Bohr
        vec_b (list[float]): lattice vector b. The unit is Bohr
        vec_c (list[float]): lattice vector c. The unit is Bohr
        atomic_numbers (list[int]): list of atomic numbers in the system.
        element_symbols (list[str]): list of element symbols in the system.
        atomic_labels (list[str]): list of labels for the atoms in the system.
    """

    positions: npt.NDArray[np.float64] = struct.field(pytree_node=True, default_factory=lambda: np.array([], dtype=np.float64))
    pbc_flag: bool = struct.field(pytree_node=False, default=False)
    vec_a: list[float] = struct.field(pytree_node=False, default_factory=list)
    vec_b: list[float] = struct.field(pytree_node=False, default_factory=list)
    vec_c: list[float] = struct.field(pytree_node=False, default_factory=list)
    atomic_numbers: list[int] = struct.field(pytree_node=False, default_factory=list)
    element_symbols: list[str] = struct.field(pytree_node=False, default_factory=list)
    atomic_labels: list[str] = struct.field(pytree_node=False, default_factory=list)

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if len(self.element_symbols) != len(self.atomic_numbers):
            logger.error("The length of element_symbols and atomic_numbers must be the same.")
            raise ValueError("The length of element_symbols and atomic_numbers must be the same.")
        if len(self.element_symbols) != len(self.atomic_numbers):
            logger.error("The length of element_symbols and atomic_numbers must be the same.")
            raise ValueError("The length of element_symbols and atomic_numbers must be the same.")
        if len(self.atomic_labels) != len(self.atomic_numbers):
            logger.error("The length of atomic_labels and atomic_numbers must be the same.")
            raise ValueError("The length of atomic_labels and atomic_numbers must be the same.")
        if len(self.positions) != len(self.atomic_numbers):
            logger.error("The length of positions and atomic_numbers must be the same.")
            raise ValueError("The length of positions and atomic_numbers must be the same.")
        if not isinstance(self.pbc_flag, bool):
            logger.error("The pbc_flag must be a boolen.")
            raise ValueError("The pbc_flag must be a boolen.")
        if self.pbc_flag:
            if len(self.vec_a) != 3 or len(self.vec_b) != 3 or len(self.vec_c) != 3:
                logger.error("The length of lattice vectors must be 3.")
                raise ValueError("The length of lattice vectors must be 3.")
        else:
            if len(self.vec_a) != 0 or len(self.vec_b) != 0 or len(self.vec_c) != 0:
                logger.error("The lattice vectors must be empty.")
                raise ValueError("The lattice vectors must be empty.")

    def get_info(self) -> list[str]:
        """Return a list of strings containing information about the Structure attribute."""
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
            self.atomic_numbers, self.element_symbols, self.atomic_labels, self.positions_cart_np
        ):
            info_lines.extend(
                [
                    f"  {element_symbol:s}, {atomic_label:s}, {atomic_number:.1f}, "
                    f"{position[0]:.8f}, {position[1]:.8f}, {position[2]:.8f}"
                ]
            )
        info_lines.extend(["  " + "-" * num_sep_line])
        return info_lines

    def logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self.get_info():
            logger.info(line)

    @property
    def cell(self) -> npt.NDArray[np.float64]:
        """Cell vectors.

        Returns:
            npt.NDAarray[np.float64]:
                3x3 cell matrix containing the cell vectors,
                `vec_a`, `vec_b`, and `vec_c`. The unit is Bohr.
        """
        cell = np.array([self.vec_a, self.vec_b, self.vec_c])
        return cell

    @property
    def recip_cell(self) -> npt.NDArray[np.float64]:
        r"""Reciprocal Lattice vectors.

        Returns:
            npt.NDAarray[np.float64]:
                3x3 cell matrix containing the reciprocal cell vectors,
                `recip_vec_a`, `recip_vec_b`, and `recip_vec_c`
                The unit is Bohr^{-1}

        Notes:
            Definitions of reciprocal lattice vectors are;
            T_a, T_b, T_c are given lattice vectors

            G_a = 2 \\pi * { T_b \\times T_c } / {T_a \\cdot ( T_b \\times T_c )}
            G_b = 2 \\pi * { T_c \\times T_a } / {T_b \\cdot ( T_c \\times T_a )}
            G_c = 2 \\pi * { T_a \\times T_b } / {T_c \\cdot ( T_a \\times T_b )}

            one can easily check if the implementations are correct by using the
            following orthonormality condition, T_i \cdot G_j = 2 \pi * \delta_{i,j}
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
    def lattice_vec_a(self) -> list:
        """Return lattice vector A (in Bohr).

        Returns:
            list[np.float64]: the lattice vector A (in Bohr).

        """
        return list(self.cell[0])

    @property
    def lattice_vec_b(self) -> list:
        """Return lattice vector B (in Bohr).

        Returns:
            list[np.float64]: the lattice vector B (in Bohr).

        """
        return list(self.cell[1])

    @property
    def lattice_vec_c(self) -> list:
        """Return lattice vector C (in Bohr).

        Returns:
            list[np.float64]: the lattice vector C (in Bohr).

        """
        return list(self.cell[2])

    @property
    def recip_vec_a(self) -> list:
        """Return reciprocal lattice vector A (in Bohr).

        Returns:
            list[np.float64]: the reciprocal lattice vector A (in Bohr).

        """
        return list(self.recip_cell[0])

    @property
    def recip_vec_b(self) -> list:
        """Return reciprocal lattice vector B (in Bohr).

        Returns:
            list[np.float64]: the reciprocal lattice vector B (in Bohr).

        """
        return list(self.recip_cell[1])

    @property
    def recip_vec_c(self) -> list:
        """Return reciprocal lattice vector C (in Bohr).

        Returns:
            list[np.float64]: the reciprocal lattice vector C (in Bohr).

        """
        return list(self.recip_cell[2])

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
    def positions_cart_np(self) -> npt.NDArray[np.float64]:
        """Return atomic positions in cartesian (Bohr).

        Returns:
            npt.NDAarray[np.float64]: (N x 3) np.array containing atomic positions in cartesian.
            The unit is Bohr
        """
        return np.array(self.positions)

    @property
    def positions_cart_jnp(self) -> jax.Array:
        """Return atomic positions in cartesian (Bohr).

        Returns:
            npt.NDAarray[np.float64]: (N x 3) np.array containing atomic positions in cartesian.
            The unit is Bohr
        """
        return jnp.array(self.positions)

    @property
    def positions_frac(self) -> npt.NDArray[np.float64]:
        """Return atomic positions in cartesian (Bohr).

        Returns:
            npt.NDAarray[np.float64]:
                (N x 3) np.array containing atomic positions in crystal (fractional) coordinate.
            The unit is Bohr
        """
        h = np.array([self.vec_a, self.vec_b, self.vec_c])
        positions_frac = np.array([np.dot(np.array(pos), np.linalg.inv(h)) for pos in self.positions_cart_np])
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

    ''' unsupported
    @classmethod
    def parse_structure_from_ase_atom(cls, ase_atom: Atoms) -> "Structure_data":
        """
        Returns:
            Struture class, by parsing an ASE Atoms instance.

        Args:
            Atoms: ASE Atoms instance
        """
        pbc_flag = ase_atom.get_pbc()
        if any(pbc_flag):
            vec_a = list(ase_atom.get_cell()[0] * Angstrom_to_Bohr)
            vec_b = list(ase_atom.get_cell()[1] * Angstrom_to_Bohr)
            vec_c = list(ase_atom.get_cell()[2] * Angstrom_to_Bohr)
        else:
            vec_a = None
            vec_b = None
            vec_c = None

        atomic_numbers = ase_atom.get_atomic_numbers()
        element_symbols = ase_atom.get_chemical_symbols()
        positions = ase_atom.get_positions() * Angstrom_to_Bohr

        return cls(
            pbc_flag=pbc_flag,
            vec_a=vec_a,
            vec_b=vec_b,
            vec_c=vec_c,
            atomic_numbers=atomic_numbers,
            element_symbols=element_symbols,
            atomic_labels=element_symbols,
            positions=positions,
        )
    '''

    @classmethod
    def parse_structure_from_file(cls, filename: str) -> "Structure_data":
        """Parse structure file.

        Args:
            filename (str): Filename of the input structure file.
                            See the ASE manual for the supported formats

        Returns:
            Structure: Struture class read from the input file

        Notes:
            The ASE module should be installed to use its read function.

        """
        # python material modules
        from ase.io import read  # type: ignore

        logger.info(f"Structure is read from {filename} using the ASE read function.")
        atoms = read(filename)
        return cls.parse_structure_from_ase_atom(atoms)

    def write_to_file(self, filename: str) -> None:
        """Write the stored sturcute information to a file.

        Args:
            filename (str): Filename of the output structure file.
                            See the ASE manual for the supported formats

        Notes:
            The ASE module should be installed to use its write function.

        """
        # python material modules
        from ase import Atoms
        from ase.io import write  # type: ignore

        if any(self.pbc_flag):
            ase_atom = Atoms(self.element_symbols, positions=self.positions_cart_np * Bohr_to_Angstrom)
            ase_atom.set_cell(
                np.array(
                    [
                        self.cell[0] * Bohr_to_Angstrom,
                        self.cell[1] * Bohr_to_Angstrom,
                        self.cell[2] * Bohr_to_Angstrom,
                    ]
                )
            )
            ase_atom.set_pbc(self.pbc_flag)
        else:
            ase_atom = Atoms(self.element_symbols, positions=self.positions_cart_np * Bohr_to_Angstrom)
            ase_atom.set_pbc(self.pbc_flag)

        write(filename, ase_atom)


def find_nearest_index(structure: Structure_data, r_cart: list[float]) -> int:
    """Find the nearest atom index for the give position.

    Args:
        structure (Structure_data): an instance of Structure_data
        r_cart (list[float, float, float]): reference position (in Bohr)

    Return:
        int: The index of the nearest neigbhor nucleus

    Todo:
        Implementing PBC (i.e., considering mirror images).
    """
    if structure.pbc_flag:
        raise NotImplementedError
    else:
        return find_nearest_nucleus_indices_np(structure, r_cart, 1)[0]


def find_nearest_index_jax(structure: Structure_data, r_cart: list[float]) -> int:
    """Find the nearest atom index for the give position.

    Args:
        structure (Structure_data): an instance of Structure_data
        r_cart (list[float, float, float]): reference position (in Bohr)

    Return:
        int: The index of the nearest neigbhor nucleus

    Todo:
        Implementing PBC (i.e., considering mirror images).
    """
    if structure.pbc_flag:
        raise NotImplementedError
    else:
        return find_nearest_nucleus_indices_jnp(structure, r_cart, 1)[0]


def find_nearest_nucleus_indices_np(structure_data: Structure_data, r_cart, N):
    """See find_nearest_index."""
    if structure_data.pbc_flag:
        raise NotImplementedError
    else:
        # Calculate the distance between each row of R_carts and r_cart
        distances = np.sqrt(np.sum((structure_data.positions_cart_np - np.array(r_cart)) ** 2, axis=1))
        # Sort indices based on the calculated distances
        nearest_indices = np.argsort(distances)
        # Select the indices of the nearest N rows
        return nearest_indices[:N]


@partial(jit, static_argnums=2)
def find_nearest_nucleus_indices_jnp(structure_data: Structure_data, r_cart, N):
    """See find_nearest_index."""
    # Calculate the distance between each row of R_carts and r_cart
    distances = jnp.sqrt(jnp.sum((structure_data.positions_cart_jnp - jnp.array(r_cart)) ** 2, axis=1))
    # Sort indices based on the calculated distances
    nearest_indices = jnp.argsort(distances)
    # Select the indices of the nearest N rows
    return nearest_indices[:N]


def get_min_dist_rel_R_cart_np(structure_data: Structure_data, r_cart: list[float, float, float], i_atom: int) -> float:
    """Minimum-distance atomic position with respect to the given r_cart.

    Args:
        structure (Structure_data): an instance of Structure_data
        r_cart (list[float, float, float]): reference position (in Bohr)
        int: the index of the target atom

    Returns:
        npt.NDAarray: rel_R_cart_min_dist containing minimum-distance atomic positions
        with respect to the given r_cart in cartesian. The unit is Bohr

    """

    def mapping(r_cart, R_cart):
        # dummy, which will be replaced in PBC cases
        return np.array(R_cart) - np.array(r_cart)

    def non_mapping(r_cart, R_cart):
        return np.array(R_cart) - np.array(r_cart)

    if np.linalg.norm(r_cart - structure_data.positions_cart_np[i_atom]) > 0.0:  # dummy, which will be replaced in PBC cases
        rel_R_cart_min_dist = mapping(r_cart, structure_data.positions_cart_np[i_atom])
    else:
        rel_R_cart_min_dist = non_mapping(r_cart, structure_data.positions_cart_np[i_atom])

    return rel_R_cart_min_dist


@jit
def get_min_dist_rel_R_cart_jnp(structure_data: Structure_data, r_cart: list[float, float, float], i_atom: int) -> float:
    """See get_min_dist_rel_R_cart_np."""
    r_cart = jnp.array(r_cart)
    R_carts = jnp.array(structure_data.positions_cart_jnp)

    def mapping(r, R):
        # dummy, which will be replaced in PBC cases
        return jnp.array(R) - jnp.array(r)

    def non_mapping(r, R):
        return jnp.array(R) - jnp.array(r)

    rel_R_cart_min_dist = lax.cond(
        jnp.linalg.norm(r_cart - R_carts[i_atom]) < 0.0,  # dummy, which will be replaced in PBC cases
        mapping,
        non_mapping,
        r_cart,
        R_carts[i_atom],
    )

    return rel_R_cart_min_dist


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

    # struct = Structure_data().parse_structure_from_file(filename="benzene.xyz")
    # struct = Structure_data().parse_structure_from_file(filename="benzene.xyz")
    # struct = Structure_data().parse_structure_from_file(filename="silicon_oxide.cif")

    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "water_ccpvtz_trexio.hdf5"))

    structure_data.logger_info()
