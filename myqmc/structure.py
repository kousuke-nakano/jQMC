"""Structure module"""

# python modules
from dataclasses import dataclass, field
import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

# python material modules
from ase import Atoms  # type: ignore
from ase.io import write, read  # type: ignore

# set logger
from logging import getLogger, StreamHandler, Formatter

# modules
from utilities.units import Angstrom_to_Bohr, Bohr_to_Angstrom

logger = getLogger("myqmc").getChild(__name__)


@dataclass
class Structure_data:
    """Structure class

    The class contains all information about a structure.

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

    pbc_flag: list[bool] = field(default_factory=lambda: [False, False, False])
    vec_a: list[float] = field(default_factory=list)
    vec_b: list[float] = field(default_factory=list)
    vec_c: list[float] = field(default_factory=list)
    atomic_numbers: list[int] = field(default_factory=list)
    element_symbols: list[str] = field(default_factory=list)
    atomic_labels: list[str] = field(default_factory=list)
    positions: npt.NDArray[np.float64] = np.array([])

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
    def cell(self) -> npt.NDArray[np.float64]:
        """
        Returns:
            3x3 cell matrix containing `vec_a`, `vec_b`, and `vec_c`
        """
        cell = np.array([self.vec_a, self.vec_b, self.vec_c])
        return cell

    @property
    def positions_cart(self) -> npt.NDArray[np.float64]:
        """
        Returns:
            (N x 3) np.array containing atomic positions in cartesian. The unit is Bohr
        """
        return self.positions

    @property
    def positions_frac(self) -> npt.NDArray[np.float64]:
        """
        Returns:
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
        Returns:
            The number of atoms in the system.
        """
        return len(self.atomic_numbers)

    @property
    def ntyp(self) -> int:
        """
        Returns:
            The number of element types in the system.
        """
        return len(list(set(self.atomic_numbers)))

    def get_ase_atom(self) -> Atoms:
        """
        Returns:
            ASE Atoms instance.

        Notes:
            # define ASE-type structure (used inside this class)
            # Note! unit in ASE is angstrom, so one should convert bohr -> ang
        """
        if self.pbc_flag:
            ase_atom = Atoms(
                self.element_symbols, positions=self.positions_cart * Bohr_to_Angstrom
            )
            ase_atom.set_cell(
                np.array(
                    [
                        self.cell[0] * Bohr_to_Angstrom,
                        self.cell[1] * Bohr_to_Angstrom,
                        self.cell[2] * Bohr_to_Angstrom,
                    ]
                )
            )
            ase_atom.set_pbc(True)
        else:
            ase_atom = Atoms(
                self.element_symbols, positions=self.positions_cart * Bohr_to_Angstrom
            )
            ase_atom.set_pbc(False)

        return ase_atom

    @classmethod
    def parse_structure_from_ase_atom(cls, ase_atom: Atoms) -> "Structure_data":
        """
        Returns:
            Struture class, by parsing an ASE Atoms instance.

        Args:
            Atoms: ASE Atoms instance
        """
        pbc_flag = ase_atom.get_pbc()
        if all(pbc_flag):
            vec_a = list(ase_atom.get_cell()[0] * Angstrom_to_Bohr)
            vec_b = list(ase_atom.get_cell()[1] * Angstrom_to_Bohr)
            vec_c = list(ase_atom.get_cell()[2] * Angstrom_to_Bohr)
        else:
            vec_a = [0.0, 0.0, 0.0]
            vec_b = [0.0, 0.0, 0.0]
            vec_c = [0.0, 0.0, 0.0]

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

    @classmethod
    def parse_structure_from_file(cls, filename: str) -> "Structure_data":
        """
        Returns:
            Struture class from a file using the ASE read function.

        Args:
            file, See the ASE manual for the supported formats
        """
        logger.info(f"Structure is read from {filename} using the ASE read function.")
        atoms = read(filename)
        return cls.parse_structure_from_ase_atom(atoms)

    def write_to_file(self, filename: str) -> None:
        """
        Write the stored sturcute information to a file

        Args:
            filename, See the ASE manual for the supported formats
        """
        atoms = self.get_ase_atom()
        write(filename, atoms)


if __name__ == "__main__":
    log = getLogger("myqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    struct = Structure_data().parse_structure_from_file(filename="benzene.xyz")
