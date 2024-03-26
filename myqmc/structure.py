"""Structure module"""

from __future__ import annotations

# python modules
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


class Structure:
    """Structure class

    The class contains all information about a structure.

    Args:
        vec_a (npt.NDArray[np.float]): lattice vector a. The unit is Bohr
        vec_b (npt.NDArray[np.float]): lattice vector b. The unit is Bohr
        vec_c (npt.NDArray[np.float]): lattice vector c. The unit is Bohr
        atomic_numbers (list[int]): list of atomic numbers in the system.
        element_symbols (list[str]): list of element symbols in the system.
        atomic_labels (list[str]): list of labels for the atoms in the system.
        positions (npt.NDArray[np.float]): (N x 3) np.array containing atomic positions in cartesian. The unit is Bohr
    """

    def __init__(
        self,
        vec_a: None | npt.NDArray[np.float] = None,
        vec_b: None | npt.NDArray[np.float] = None,
        vec_c: None | npt.NDArray[np.float] = None,
        atomic_numbers: None | list[int] = None,
        element_symbols: None | list[str] = None,
        atomic_labels: None | list[str] = None,
        positions: None | npt.NDArray[np.float] = None,
    ):

        # initialization
        if vec_a is None:
            vec_a = np.array([0.0, 0.0, 0.0], dtype=float)
        if vec_b is None:
            vec_b = np.array([0.0, 0.0, 0.0], dtype=float)
        if vec_c is None:
            vec_c = np.array([0.0, 0.0, 0.0], dtype=float)

        if atomic_numbers is None:
            atomic_numbers = []
        if element_symbols is None:
            element_symbols = []
        if atomic_labels is None:
            atomic_labels = []
        if positions is None:
            positions = np.array([[]])

        self.__vec_a = vec_a
        self.__vec_b = vec_b
        self.__vec_c = vec_c
        self.__norm_vec_a = LA.norm(self.__vec_a)
        self.__norm_vec_b = LA.norm(self.__vec_b)
        self.__norm_vec_c = LA.norm(self.__vec_c)
        self.__atomic_numbers = atomic_numbers
        self.__element_symbols = element_symbols
        self.__atomic_labels = atomic_labels
        self.__positions = positions

    @property
    def cell(self) -> npt.NDArray[np.float]:
        """
        Returns:
            3x3 cell matrix containing `vec_a`, `vec_b`, and `vec_c`
        """
        _cell = np.array([self.__vec_a, self.__vec_b, self.__vec_c])
        return _cell

    @property
    def atomic_numbers(self) -> list[int]:
        """
        Returns:
            list of atomic numbers
        """
        return self.__atomic_numbers

    @property
    def element_symbols(self) -> list[str]:
        """
        Returns:
            list of element symbols
        """
        return self.__element_symbols

    @property
    def positions_cart(self) -> npt.NDArray[np.float]:
        """
        Returns:
            (N x 3) np.array containing atomic positions in cartesian. The unit is Bohr
        """
        return self.__positions

    @property
    def positions_frac(self) -> npt.NDArray[np.float]:
        """
        Returns:
            (N x 3) np.array containing atomic positions in crystal (fractional) coordinate.
        """
        h = np.array([self.__vec_a, self.__vec_b, self.__vec_c])
        self.__positions_frac = np.array(
            [np.dot(np.array(pos), np.linalg.inv(h)) for pos in self.positions_cart]
        )
        return self.__positions_frac

    @property
    def natom(self) -> int:
        """
        Returns:
            The number of atoms in the system.
        """
        return len(self.__atomic_numbers)

    @property
    def ntyp(self) -> int:
        """
        Returns:
            The number of element types in the system.
        """
        return len(list(set(self.__atomic_numbers)))

    @property
    def pbc_flag(self) -> bool:
        """
        Returns:
            Flag if the system is under PBC (i.e. a crystal) or not (i.e. a molecule)
        """
        if (
            self.__norm_vec_a == 0.0
            and self.__norm_vec_b == 0.0
            and self.__norm_vec_c == 0.0
        ):
            return False
        else:
            return True

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
                self.__element_symbols, positions=self.positions_cart * Bohr_to_Angstrom
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
                self.__element_symbols, positions=self.positions_cart * Bohr_to_Angstrom
            )
            ase_atom.set_pbc(False)

        return ase_atom

    @classmethod
    def parse_structure_from_ase_atom(cls, ase_atom: Atoms) -> Structure:
        """
        Returns:
            Struture class, by parsing an ASE Atoms instance.

        Args:
            Atoms: ASE Atoms instance
        """
        if all(ase_atom.get_pbc()):
            vec_a = ase_atom.get_cell()[0] * Angstrom_to_Bohr
            vec_b = ase_atom.get_cell()[1] * Angstrom_to_Bohr
            vec_c = ase_atom.get_cell()[2] * Angstrom_to_Bohr
        else:
            vec_a = np.array([0.0, 0.0, 0.0])
            vec_b = np.array([0.0, 0.0, 0.0])
            vec_c = np.array([0.0, 0.0, 0.0])

        atomic_numbers = ase_atom.get_atomic_numbers()
        element_symbols = ase_atom.get_chemical_symbols()
        positions = ase_atom.get_positions() * Angstrom_to_Bohr

        return cls(
            vec_a=vec_a,
            vec_b=vec_b,
            vec_c=vec_c,
            atomic_numbers=atomic_numbers,
            element_symbols=element_symbols,
            atomic_labels=element_symbols,
            positions=positions,
        )

    @classmethod
    def parse_structure_from_file(cls, filename: str) -> Structure:
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
    
    struct=Structure().parse_structure_from_file(file='benzene.xyz')
