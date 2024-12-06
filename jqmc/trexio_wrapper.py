"""TREXIO wrapper modules."""

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

# import python modules
# logger
from logging import Formatter, StreamHandler, getLogger

import numpy as np
import scipy

# import trexio
import trexio

from .atomic_orbital import AOs_data
from .coulomb_potential import Coulomb_potential_data
from .determinant import Geminal_data
from .molecular_orbital import MOs_data

# import myQMC
from .structure import Structure_data

logger = getLogger("jqmc").getChild(__name__)


def read_trexio_file(trexio_file: str):
    """
    The method reads a TREXIO file and return AOs_data, MOs_data,
    Structure_data, and Coulomb_potential_data instances.

    Args:
        trexio_file (str): the path to a TREXIO file

    Returns
    -------
        instances of AOs_data, MOs_data, Structure_data, and
        Coulomb_potential_data.
    """
    # prefix and file names
    logger.info(f"TREXIO file = {trexio_file}")

    # read a trexio file
    file_r = trexio.File(
        trexio_file,
        mode="r",
        back_end=trexio.TREXIO_HDF5,
    )

    # check if the system is PBC or not.
    periodic = trexio.read_pbc_periodic(file_r)

    if periodic:
        logger.info("Crystal (Periodic boundary condition)")
        pbc_flag = [True, True, True]
        # cell_a = trexio.read_cell_a(file_r)
        # cell_b = trexio.read_cell_b(file_r)
        # cell_c = trexio.read_cell_c(file_r)
        # k_point = trexio.read_pbc_k_point(file_r)
        raise NotImplementedError
    else:
        pbc_flag = [False, False, False]
        logger.info("Molecule (Open boundary condition)")

    # read electron num
    num_ele_up = trexio.read_electron_up_num(file_r)
    num_ele_dn = trexio.read_electron_dn_num(file_r)

    # read structure info.
    # nucleus_num_r = trexio.read_nucleus_num(file_r)
    labels_r = trexio.read_nucleus_label(file_r)
    # charges_r = trexio.read_nucleus_charge(file_r)
    coords_r = trexio.read_nucleus_coord(file_r)

    # Reading basis sets info
    # basis_type = trexio.read_basis_type(file_r)
    basis_shell_num = trexio.read_basis_shell_num(file_r)
    basis_shell_index = trexio.read_basis_shell_index(file_r)
    # basis_prim_num = trexio.read_basis_prim_num(file_r)
    basis_nucleus_index = trexio.read_basis_nucleus_index(file_r)
    basis_shell_ang_mom = trexio.read_basis_shell_ang_mom(file_r)
    basis_shell_factor = trexio.read_basis_shell_factor(file_r)
    basis_shell_index = trexio.read_basis_shell_index(file_r)
    basis_exponent = trexio.read_basis_exponent(file_r)
    basis_coefficient = trexio.read_basis_coefficient(file_r)
    basis_prim_factor = trexio.read_basis_prim_factor(file_r)
    logger.info(f"max angular momentum l = {np.max(basis_shell_ang_mom)}.")

    # ao info
    ao_cartesian = trexio.read_ao_cartesian(file_r)
    ao_num = trexio.read_ao_num(file_r)
    # ao_shell = trexio.read_ao_shell(file_r)
    ao_normalization = trexio.read_ao_normalization(file_r)

    # ao spherical part check
    if ao_cartesian:
        raise NotImplementedError

    # mo info
    # mo_type = trexio.read_mo_type(file_r)
    mo_num = trexio.read_mo_num(file_r)
    # mo_occupation = trexio.read_mo_occupation(file_r)
    mo_coefficient_real = trexio.read_mo_coefficient(file_r)

    # mo spin check
    try:
        mo_spin = trexio.read_mo_spin(file_r)
        if all(x == 0 for x in mo_spin):
            spin_restricted = True
        else:
            spin_restricted = False
    except trexio.Error:  # backward compatibility
        mo_spin = [0 for _ in range(mo_num)]
        spin_restricted = True

    # MO complex check
    if trexio.has_mo_coefficient_im(file_r):
        logger.info("The WF is complex")
        # mo_coefficient_imag = trexio.read_mo_coefficient_im(file_r)
        # complex_flag = True
        raise NotImplementedError
    else:
        logger.info("The WF is real")
        # complex_flag = False

    # Pseudo potentials info
    if trexio.has_ecp_num(file_r):
        ecp_flag = True
        ecp_max_ang_mom_plus_1 = trexio.read_ecp_max_ang_mom_plus_1(file_r)
        ecp_z_core = trexio.read_ecp_z_core(file_r)
        ecp_num = trexio.read_ecp_num(file_r)
        ecp_ang_mom = trexio.read_ecp_ang_mom(file_r)
        ecp_nucleus_index = trexio.read_ecp_nucleus_index(file_r)
        ecp_exponent = trexio.read_ecp_exponent(file_r)
        ecp_coefficient = trexio.read_ecp_coefficient(file_r)
        ecp_power = trexio.read_ecp_power(file_r)
    else:
        ecp_flag = False
    file_r.close()

    # Structure_data instance
    structure_data = Structure_data(
        pbc_flag=pbc_flag,
        vec_a=[],
        vec_b=[],
        vec_c=[],
        atomic_numbers=convert_from_atomic_labels_to_atomic_numbers(labels_r),
        element_symbols=labels_r,
        atomic_labels=labels_r,
        positions=coords_r,
    )

    # AOs_data instance
    ao_num_count = 0
    ao_prim_num_count = 0

    # values to be stored
    nucleus_index = []
    atomic_center_carts = []
    angular_momentums = []
    magnetic_quantum_numbers = []
    orbital_indices = []
    exponents = []
    coefficients = []

    for i_shell in range(basis_shell_num):
        b_nucleus_index = basis_nucleus_index[i_shell]
        b_coord = list(coords_r[b_nucleus_index])
        b_ang_mom = basis_shell_ang_mom[i_shell]
        ao_mag_mom_list = [0] + [i * (-1) ** j for i in range(1, b_ang_mom + 1) for j in range(2)]
        num_ao_mag_moms = len(ao_mag_mom_list)

        ao_nucleus_index = [b_nucleus_index for _ in range(num_ao_mag_moms)]
        ao_coords = [b_coord for _ in range(num_ao_mag_moms)]
        ao_ang_moms = [b_ang_mom for _ in range(num_ao_mag_moms)]

        b_prim_indices = [i for i, v in enumerate(basis_shell_index) if v == i_shell]
        b_prim_num = len(b_prim_indices)
        b_normalizations = [
            np.sqrt(
                (
                    2.0 ** (2 * b_ang_mom + 3)
                    * scipy.special.factorial(b_ang_mom + 1)
                    * (2 * basis_exponent[k]) ** (b_ang_mom + 1.5)
                )
                / (scipy.special.factorial(2 * b_ang_mom + 2) * np.sqrt(np.pi))
            )
            for k in b_prim_indices
        ]
        b_prim_exponents = [basis_exponent[k] for k in b_prim_indices]
        b_prim_coefficients = [
            basis_shell_factor[i_shell]
            * basis_prim_factor[k]
            * np.sqrt(4 * np.pi)
            / np.sqrt(2 * b_ang_mom + 1)
            / b_normalizations[i]
            * basis_coefficient[k]
            for i, k in enumerate(b_prim_indices)
        ]
        orbital_indices_all = [ao_num_count + j for j in range(num_ao_mag_moms) for _ in range(b_prim_num)]
        ao_exponents = b_prim_exponents * num_ao_mag_moms
        ao_coefficients_list = b_prim_coefficients * num_ao_mag_moms
        ao_coefficients = [
            ao_coefficients_list[k] * ao_normalization[orbital_indices_all[k]] for k in range(len(ao_coefficients_list))
        ]
        ao_num_count += num_ao_mag_moms
        ao_prim_num_count += num_ao_mag_moms * b_prim_num

        nucleus_index += ao_nucleus_index
        atomic_center_carts += ao_coords
        angular_momentums += ao_ang_moms
        magnetic_quantum_numbers += ao_mag_mom_list
        orbital_indices += orbital_indices_all
        exponents += ao_exponents
        coefficients += ao_coefficients

    if ao_num_count != ao_num:
        logger.error(f"ao_num_count = {ao_num_count} is inconsistent with the read ao_num = {ao_num}")
        raise ValueError

    """ old!!
    aos_data = AOs_data_debug(
        num_ao=ao_num_count,
        num_ao_prim=ao_prim_num_count,
        atomic_center_carts=np.array(atomic_center_carts),
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
    )
    """

    aos_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=nucleus_index,
        num_ao=ao_num_count,
        num_ao_prim=ao_prim_num_count,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
    )

    # MOs_data instance
    if spin_restricted:
        mo_indices = [i for (i, v) in enumerate(mo_spin) if v == 0]
        mo_coefficient_real_up = mo_coefficient_real_dn = mo_coefficient_real[mo_indices]
        mo_num_up = mo_num_dn = mo_num
        mos_data_up = MOs_data(num_mo=mo_num_up, mo_coefficients=mo_coefficient_real_up, aos_data=aos_data)
        mos_data_dn = MOs_data(num_mo=mo_num_dn, mo_coefficients=mo_coefficient_real_dn, aos_data=aos_data)

        mo_lambda_paired_occ = np.eye(num_ele_up, num_ele_dn, k=0)

        mo_lambda_matrix_unpaired = np.eye(num_ele_up, num_ele_up - num_ele_dn, k=-num_ele_dn)
        mo_lambda_matrix = np.block(
            [
                [
                    mo_lambda_paired_occ,
                    np.zeros((num_ele_up, mo_num_dn - num_ele_dn)),
                    mo_lambda_matrix_unpaired,
                ],
                [
                    np.zeros((mo_num_up - num_ele_up, num_ele_dn)),
                    np.zeros((mo_num_up - num_ele_up, mo_num_dn - num_ele_dn)),
                    np.zeros((mo_num_up - num_ele_up, num_ele_up - num_ele_dn)),
                ],
            ]
        )
    else:
        raise NotImplementedError

    geminal_data = Geminal_data(
        num_electron_up=num_ele_up,
        num_electron_dn=num_ele_dn,
        orb_data_up_spin=mos_data_up,
        orb_data_dn_spin=mos_data_dn,
        lambda_matrix=mo_lambda_matrix,
    )

    # Coulomb_potential_data instance
    if ecp_flag:
        coulomb_potential_data = Coulomb_potential_data(
            structure_data=structure_data,
            ecp_flag=True,
            z_cores=ecp_z_core,
            max_ang_mom_plus_1=ecp_max_ang_mom_plus_1,
            num_ecps=ecp_num,
            ang_moms=ecp_ang_mom,
            nucleus_index=ecp_nucleus_index,
            exponents=ecp_exponent,
            coefficients=ecp_coefficient,
            powers=ecp_power + 2,
        )
    else:
        coulomb_potential_data = Coulomb_potential_data(structure_data=structure_data, ecp_flag=False)

    return (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_data,
        coulomb_potential_data,
    )


def convert_from_atomic_numbers_to_atomic_labels(charges_r: list[int]) -> list[str]:
    # Dictionary mapping atomic numbers to symbols, up to atomic number 86
    atomic_number_to_element = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
        21: "Sc",
        22: "Ti",
        23: "V",
        24: "Cr",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        31: "Ga",
        32: "Ge",
        33: "As",
        34: "Se",
        35: "Br",
        36: "Kr",
        37: "Rb",
        38: "Sr",
        39: "Y",
        40: "Zr",
        41: "Nb",
        42: "Mo",
        43: "Tc",
        44: "Ru",
        45: "Rh",
        46: "Pd",
        47: "Ag",
        48: "Cd",
        49: "In",
        50: "Sn",
        51: "Sb",
        52: "Te",
        53: "I",
        54: "Xe",
        55: "Cs",
        56: "Ba",
        57: "La",
        58: "Ce",
        59: "Pr",
        60: "Nd",
        61: "Pm",
        62: "Sm",
        63: "Eu",
        64: "Gd",
        65: "Tb",
        66: "Dy",
        67: "Ho",
        68: "Er",
        69: "Tm",
        70: "Yb",
        71: "Lu",
        72: "Hf",
        73: "Ta",
        74: "W",
        75: "Re",
        76: "Os",
        77: "Ir",
        78: "Pt",
        79: "Au",
        80: "Hg",
        81: "Tl",
        82: "Pb",
        83: "Bi",
        84: "Po",
        85: "At",
        86: "Rn",
    }

    labels_r = []

    for charge in charges_r:
        if charge <= 0:
            raise ValueError("Atomic number must be greater than 0.")
        elif charge > 86:
            raise NotImplementedError("Atomic numbers above 86 are not implemented.")

        if charge in atomic_number_to_element:
            labels_r.append(atomic_number_to_element[charge])
        else:
            raise ValueError(f"No element for atomic number: {charge}")

    return labels_r


def convert_from_atomic_labels_to_atomic_numbers(labels_r: list[str]) -> list[int]:
    # Mapping of element symbols to their atomic numbers up to 86
    element_to_number = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V": 23,
        "Cr": 24,
        "Mn": 25,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Ga": 31,
        "Ge": 32,
        "As": 33,
        "Se": 34,
        "Br": 35,
        "Kr": 36,
        "Rb": 37,
        "Sr": 38,
        "Y": 39,
        "Zr": 40,
        "Nb": 41,
        "Mo": 42,
        "Tc": 43,
        "Ru": 44,
        "Rh": 45,
        "Pd": 46,
        "Ag": 47,
        "Cd": 48,
        "In": 49,
        "Sn": 50,
        "Sb": 51,
        "Te": 52,
        "I": 53,
        "Xe": 54,
        "Cs": 55,
        "Ba": 56,
        "La": 57,
        "Ce": 58,
        "Pr": 59,
        "Nd": 60,
        "Pm": 61,
        "Sm": 62,
        "Eu": 63,
        "Gd": 64,
        "Tb": 65,
        "Dy": 66,
        "Ho": 67,
        "Er": 68,
        "Tm": 69,
        "Yb": 70,
        "Lu": 71,
        "Hf": 72,
        "Ta": 73,
        "W": 74,
        "Re": 75,
        "Os": 76,
        "Ir": 77,
        "Pt": 78,
        "Au": 79,
        "Hg": 80,
        "Tl": 81,
        "Pb": 82,
        "Bi": 83,
        "Po": 84,
        "At": 85,
        "Rn": 86,
    }

    # Convert labels to atomic numbers, checking for validity
    atomic_numbers = []
    for label in labels_r:
        if label in element_to_number:
            atomic_number = element_to_number[label]
            if atomic_number > 86:
                raise NotImplementedError("Atomic numbers above 86 are not implemented.")
            atomic_numbers.append(atomic_number)
        else:
            raise ValueError(f"No atomic number found for the label '{label}'")
    return atomic_numbers


if __name__ == "__main__":
    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
    log.addHandler(stream_handler)
