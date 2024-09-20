"""Structure module"""

# python modules
import itertools
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


@struct.dataclass
class Structure_data:
    """Structure class

    The class contains all information about the given structure.

    Args:
        pbc_flag (list[bool]): pbc_flags in the a, b, and c directions.
        vec_a (list[float]): lattice vector a. The unit is Bohr
        vec_b (list[float]): lattice vector b. The unit is Bohr
        vec_c (list[float]): lattice vector c. The unit is Bohr
        atomic_numbers (list[int]): list of atomic numbers in the system.
        element_symbols (list[str]): list of element symbols in the system.
        atomic_labels (list[str]): list of labels for the atoms in the system.
        positions (npt.NDArray[np.float64]): (N x 3) np.array containing atomic positions in cartesian. The unit is Bohr
    """

    pbc_flag: list[bool] = struct.field(pytree_node=False)
    vec_a: list[float] = struct.field(pytree_node=False)
    vec_b: list[float] = struct.field(pytree_node=False)
    vec_c: list[float] = struct.field(pytree_node=False)
    atomic_numbers: list[int] = struct.field(pytree_node=False)
    element_symbols: list[str] = struct.field(pytree_node=False)
    atomic_labels: list[str] = struct.field(pytree_node=False)
    positions: npt.NDArray[np.float64] = struct.field(pytree_node=True)

    @property
    def cell(self) -> npt.NDArray[np.float64]:
        """
        Returns
        -------
            3x3 cell matrix containing the cell vectors, `vec_a`, `vec_b`, and `vec_c`
            The unit is Bohr.
        """
        cell = np.array([self.vec_a, self.vec_b, self.vec_c])
        return cell

    @property
    def recip_cell(self) -> npt.NDArray[np.float64]:
        """
        Returns
        -------
            3x3 cell matrix containing the reciprocal cell vectors,
            `recip_vec_a`, `recip_vec_b`, and `recip_vec_c`
            The unit is Bohr^{-1}
        """
        # definitions of reciprocal lattice vectors are;
        # T_a, T_b, T_c are given lattice vectors
        #
        # G_a = 2 \pi * { T_b \times T_c } / {T_a \cdot ( T_b \times T_c )}
        # G_b = 2 \pi * { T_c \times T_a } / {T_b \cdot ( T_c \times T_a )}
        # G_c = 2 \pi * { T_a \times T_b } / {T_c \cdot ( T_a \times T_b )}
        #
        # one can easily check if the implementations are correct by using the
        # following orthonormality condition, T_i \cdot G_j = 2 \pi * \delta_{i,j}
        #

        recip_a = (
            2
            * np.pi
            * (np.cross(self.vec_b, self.vec_c))
            / (np.dot(self.vec_a, np.cross(self.vec_b, self.vec_c)))
        )
        recip_b = (
            2
            * np.pi
            * (np.cross(self.vec_c, self.vec_a))
            / (np.dot(self.vec_b, np.cross(self.vec_c, self.vec_a)))
        )
        recip_c = (
            2
            * np.pi
            * (np.cross(self.vec_a, self.vec_b))
            / (np.dot(self.vec_c, np.cross(self.vec_a, self.vec_b)))
        )

        # check if the implementations are correct
        lattice_vec_list = [self.vec_a, self.vec_b, self.vec_c]
        recip_vec_list = [recip_a, recip_b, recip_c]
        for (lattice_vec_i, lattice_vec), (recip_vec_j, recip_vec) in itertools.product(
            enumerate(lattice_vec_list), enumerate(recip_vec_list)
        ):
            if lattice_vec_i == recip_vec_j:
                np.testing.assert_almost_equal(
                    np.dot(lattice_vec, recip_vec), 2 * np.pi, decimal=15
                )
            else:
                np.testing.assert_almost_equal(np.dot(lattice_vec, recip_vec), 0.0, decimal=15)

        recip_cell = np.array([recip_a, recip_b, recip_c])
        return recip_cell

    @property
    def lattice_vec_a(self) -> list:
        return list(self.cell[0])

    @property
    def lattice_vec_b(self) -> list:
        return list(self.cell[1])

    @property
    def lattice_vec_c(self) -> list:
        return list(self.cell[2])

    @property
    def recip_vec_a(self) -> list:
        return list(self.recip_cell[0])

    @property
    def recip_vec_b(self) -> list:
        return list(self.recip_cell[1])

    @property
    def recip_vec_c(self) -> list:
        return list(self.recip_cell[2])

    @property
    def norm_vec_a(self) -> float:
        return LA.norm(self.vec_a)

    @property
    def norm_vec_b(self) -> float:
        return LA.norm(self.vec_b)

    @property
    def norm_vec_c(self) -> float:
        return LA.norm(self.vec_c)

    @property
    def positions_cart(self) -> npt.NDArray[np.float64]:
        """
        Returns
        -------
            (N x 3) np.array containing atomic positions in cartesian. The unit is Bohr
        """
        return self.positions

    @property
    def positions_frac(self) -> npt.NDArray[np.float64]:
        """
        Returns
        -------
            (N x 3) np.array containing atomic positions in crystal (fractional) coordinate.
        """
        h = np.array([self.vec_a, self.vec_b, self.vec_c])
        positions_frac = np.array(
            [np.dot(np.array(pos), np.linalg.inv(h)) for pos in self.positions_cart]
        )
        return positions_frac

    @property
    def natom(self) -> int:
        """
        Returns
        -------
            The number of atoms in the system.
        """
        return len(self.atomic_numbers)

    @property
    def ntyp(self) -> int:
        """
        Returns
        -------
            The number of element types in the system.
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
        """
        Returns
        -------
            Struture class from a file using the ASE read function.

        Args:
            file, See the ASE manual for the supported formats
        """
        # python material modules
        from ase.io import read  # type: ignore

        logger.info(f"Structure is read from {filename} using the ASE read function.")
        atoms = read(filename)
        return cls.parse_structure_from_ase_atom(atoms)

    def write_to_file(self, filename: str) -> None:
        """
        Write the stored sturcute information to a file

        Args:
            filename, See the ASE manual for the supported formats
        """
        # python material modules
        from ase import Atoms
        from ase.io import write  # type: ignore

        if any(self.pbc_flag):
            ase_atom = Atoms(self.element_symbols, positions=self.positions_cart * Bohr_to_Angstrom)
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
            ase_atom = Atoms(self.element_symbols, positions=self.positions_cart * Bohr_to_Angstrom)
            ase_atom.set_pbc(self.pbc_flag)

        write(filename, ase_atom)


def find_nearest_index(structure: Structure_data, r_cart: list[float, float, float]) -> int:
    """
    Args:
        structure (Structure_data): an instance of Structure_data
        r_cart (list[float, float, float]): reference position
    Return:
        The index of the nearest neigbhor nucleus (int)
    """
    if any(structure.pbc_flag):
        raise NotImplementedError
    else:
        return find_nearest_nucleus_indices_np(structure, r_cart, 1)[0]


def find_nearest_nucleus_indices_np(structure_data: Structure_data, r_cart, N):
    if any(structure_data.pbc_flag):
        raise NotImplementedError
    else:
        # Calculate the distance between each row of R_carts and r_cart
        distances = np.sqrt(np.sum((structure_data.positions_cart - np.array(r_cart)) ** 2, axis=1))
        # Sort indices based on the calculated distances
        nearest_indices = np.argsort(distances)
        # Select the indices of the nearest N rows
        return nearest_indices[:N]


def find_nearest_nucleus_indices_jnp(structure_data: Structure_data, r_cart, N):
    # Calculate the distance between each row of R_carts and r_cart
    distances = jnp.sqrt(jnp.sum((structure_data.positions_cart - jnp.array(r_cart)) ** 2, axis=1))
    # Sort indices based on the calculated distances
    nearest_indices = jnp.argsort(distances)
    # Select the indices of the nearest N rows
    return nearest_indices[:N]


def get_min_dist_rel_R_cart_np(
    structure_data: Structure_data, r_cart: list[float, float, float], i_atom: int
) -> float:
    """
    Returns
    -------
        rel_R_cart_min_dist containing minimum-distance atomic positions with respect to the given r_cart in cartesian. The unit is Bohr
    """

    def mapping(r_cart, R_cart):
        return np.array(R_cart) - np.array(r_cart)

    def non_mapping(r_cart, R_cart):
        return np.array(R_cart) - np.array(r_cart)

    if (
        np.linalg.norm(r_cart - structure_data.positions_cart[i_atom]) > 0.0
    ):  # dummy, which will be replaced in PBC cases
        rel_R_cart_min_dist = mapping(r_cart, structure_data.positions_cart[i_atom])
    else:
        rel_R_cart_min_dist = non_mapping(r_cart, structure_data.positions_cart[i_atom])

    return rel_R_cart_min_dist


@jit
def get_min_dist_rel_R_cart_jnp(
    structure_data: Structure_data, r_cart: list[float, float, float], i_atom: int
) -> float:
    """
    Returns
    -------
        rel_R_cart_min_dist minimum-distance atomic positions with respect to the given r_cart in cartesian. The unit is Bohr
    """

    def mapping(r_cart, R_cart):
        return jnp.array(R_cart) - jnp.array(r_cart)

    def non_mapping(r_cart, R_cart):
        return jnp.array(R_cart) - jnp.array(r_cart)

    rel_R_cart_min_dist = lax.cond(
        jnp.linalg.norm(r_cart - structure_data.positions_cart[i_atom])
        < 0.0,  # dummy, which will be replaced in PBC cases
        mapping,
        non_mapping,
        r_cart,
        structure_data.positions_cart[i_atom],
    )

    return rel_R_cart_min_dist


if __name__ == "__main__":
    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    struct = Structure_data().parse_structure_from_file(filename="benzene.xyz")

    struct = Structure_data().parse_structure_from_file(filename="benzene.xyz")

    struct = Structure_data().parse_structure_from_file(filename="silicon_oxide.cif")
    print(struct.recip_cell)
