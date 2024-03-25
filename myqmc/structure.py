#!python -u
# -*- coding: utf-8 -*-

"""
Structure related classes and methods

Todo:
    * docstrings are not completed.
    * refactoring assert sentences. The assert should not be used for any on-the-fly check.
    * implementing __str__ method.
    * implementing sanity_check method.

"""

# python modules
import math
import numpy as np
from numpy import linalg as LA
from typing import Optional

# python material modules
from ase import Atoms
from ase.visualize import view
from ase.io import write, read

# set logger
from logging import getLogger, StreamHandler, Formatter

# turbogenius module
from utils.units import Angstrom

logger = getLogger("myqmc").getChild(__name__)

class Structure:
    def __init__(
        self,
        vec_a: Optional[np.array] = None,
        vec_b: Optional[np.array] = None,
        vec_c: Optional[np.array] = None,
        atomic_numbers: Optional[list] = None,
        element_symbols: Optional[list] = None,
        positions: Optional[np.ndarray] = None
    ):

        if vec_a is None:
            np.array([0.0, 0.0, 0.0], dtype=float)
        if vec_b is None:
            np.array([0.0, 0.0, 0.0], dtype=float)
        if vec_c is None:
            np.array([0.0, 0.0, 0.0], dtype=float)
        
        if atomic_numbers is None:
            atomic_numbers = []
        if element_symbols is None:
            element_symbols = []
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
        self.__positions = positions

    def __str__(self):
        pass

    @property
    def cell(self):
        return self.__cell

    @property
    def atomic_numbers(self):
        return self.__atomic_numbers

    @property
    def element_symbols(self):
        return self.__element_symbols

    @property
    def positions_cart(self):
        return self.__positions

    @property
    def positions_frac(self):
        h = np.array([self.__vec_a, self.__vec_b, self.__vec_c])
        self.__positions_frac = np.array([np.dot(np.array(pos), np.linalg.inv(h)) for pos in self.positions])
        return self.__positions_frac

    @property
    def natom(self):
        return len(self.__atomic_numbers)

    @property
    def ntyp(self):
        return len(list(set(self.__atomic_numbers)))

    @property
    def pbc_flag(self):
        return self.__cell.pbc_flag

    def get_ase_atom(self):
        # define ASE-type structure (used inside this class)
        # Note! unit in ASE is angstrom, so one should convert bohr -> ang
        if self.__cell.pbc_flag:
            ase_atom = Atoms(
                self.__element_symbols, positions=self.__positions / Angstrom
            )
            ase_atom.set_cell(
                np.array(
                    [
                        self.__cell.vec_a / Angstrom,
                        self.__cell.vec_b / Angstrom,
                        self.__cell.vec_c / Angstrom,
                    ]
                )
            )
            ase_atom.set_pbc(True)
        else:
            ase_atom = Atoms(
                self.__element_symbols, positions=self.__positions / Angstrom
            )
            ase_atom.set_pbc(False)

        return ase_atom

    @classmethod
    def parse_structure_from_ase_atom(cls, ase_atom):
        if all(ase_atom.get_pbc()):
            vec_a = ase_atom.get_cell()[0] * Angstrom
            vec_b = ase_atom.get_cell()[1] * Angstrom
            vec_c = ase_atom.get_cell()[2] * Angstrom

            cell = Cell(vec_a=vec_a, vec_b=vec_b, vec_c=vec_c)
        else:
            cell = Cell()

        atomic_numbers = ase_atom.get_atomic_numbers()
        element_symbols = ase_atom.get_chemical_symbols()
        positions = ase_atom.get_positions() * Angstrom

        return cls(
            cell=cell,
            atomic_numbers=atomic_numbers,
            element_symbols=element_symbols,
            positions=positions,
        )

    @classmethod
    def parse_structure_from_file(cls, file):
        logger.info(f"Structure is read from {file} using the ASE read function.")
        atoms = read(file)
        return cls.parse_structure_from_ase_atom(atoms)

    def write_to_file(self, file):
        atoms = self.get_ase_atom()
        write(file, atoms)

if __name__ == "__main__":
    log = getLogger("pyturbo")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    # moved to examples
