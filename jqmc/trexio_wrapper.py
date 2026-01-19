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

# import python modules
# logger
import itertools
from logging import getLogger

import numpy as np
import scipy

# import trexio
import trexio

from .atomic_orbital import AOs_cart_data, AOs_sphe_data
from .coulomb_potential import Coulomb_potential_data
from .determinant import Geminal_data
from .molecular_orbital import MOs_data

# import myQMC
from .structure import Structure_data

logger = getLogger("jqmc").getChild(__name__)


def read_trexio_file(
    trexio_file: str, store_tuple: bool = False
) -> tuple[Structure_data, AOs_sphe_data, MOs_data, MOs_data, Geminal_data, Coulomb_potential_data]:
    """Load a TREXIO HDF5 file into jqmc data containers.

    Args:
        trexio_file (str): Path to the TREXIO file to read (HDF5 backend expected).
        store_tuple (bool, optional): Store list-like fields as tuples for immutability
            (useful in tests with JAX/Flax), at the cost of slower production runs.
            Defaults to ``False``.

    Returns:
        tuple: ``(structure_data, aos_data, mos_data_up, mos_data_dn, geminal_data, coulomb_potential_data)``
        where:

        - ``structure_data`` is a `Structure_data` describing atoms and geometry.
        - ``aos_data`` is either `AOs_cart_data` or `AOs_sphe_data` depending on the basis.
        - ``mos_data_up`` and ``mos_data_dn`` are `MOs_data` for spin-up/down orbitals.
        - ``geminal_data`` is a `Geminal_data` assembled from the MO block.
        - ``coulomb_potential_data`` is a `Coulomb_potential_data` describing (E)CPs.

    Raises:
        NotImplementedError: If periodic cells (PBC) or complex molecular orbitals are encountered.
        ValueError: If atomic labels are unsupported or AO counts are inconsistent.

    Notes:
        - Periodic boundary conditions are parsed but not supported yet.
        - Molecular orbitals are assumed real-valued; complex coefficients are rejected.

    Examples:
        >>> from jqmc.trexio_wrapper import read_trexio_file
        >>> structure_data, aos_data, mos_up, mos_dn, geminal_data, coulomb_data = read_trexio_file("molecule.h5")
        >>> structure_data.atomic_labels[:3]
        ['O', 'H', 'H']
    """
    # prefix and file names
    # logger.info(f"TREXIO file = {trexio_file}")

    # read a trexio file
    file_r = trexio.File(
        trexio_file,
        mode="r",
        back_end=trexio.TREXIO_HDF5,
    )

    # check if the system is PBC or not.
    try:
        periodic = trexio.read_pbc_periodic(file_r)
    except trexio.Error:
        periodic = False
    if periodic:
        # logger.info("Crystal (Periodic boundary condition)")
        pbc_flag = True
        # cell_a = trexio.read_cell_a(file_r)
        # cell_b = trexio.read_cell_b(file_r)
        # cell_c = trexio.read_cell_c(file_r)
        # k_point = trexio.read_pbc_k_point(file_r)
        raise NotImplementedError
    else:
        pbc_flag = False
        # logger.info("Molecule (Open boundary condition)")
    logger.info(f"pbc_flag = {pbc_flag}")

    # read electron num
    num_ele_up = trexio.read_electron_up_num(file_r)
    num_ele_dn = trexio.read_electron_dn_num(file_r)

    if num_ele_up - num_ele_dn != 0:
        spin_polarized = True
    else:
        spin_polarized = False
    logger.info(f"num_ele_up = {num_ele_up}, num_ele_dn = {num_ele_dn}, spin_polarized = {spin_polarized}")

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
    # logger.info(f"max angular momentum l = {np.max(basis_shell_ang_mom)}.")

    # ao info
    ao_cartesian = trexio.read_ao_cartesian(file_r)
    ao_num = trexio.read_ao_num(file_r)
    # ao_shell = trexio.read_ao_shell(file_r)
    ao_normalization = trexio.read_ao_normalization(file_r)

    # print(f"len(ao_normalization) = {len(ao_normalization)}")

    # mo info
    # mo_type = trexio.read_mo_type(file_r)
    # mo_num = trexio.read_mo_num(file_r)
    mo_coefficient_real = trexio.read_mo_coefficient(file_r)
    mo_occupation = trexio.read_mo_occupation(file_r)

    # mo spin check
    mo_spin = trexio.read_mo_spin(file_r)

    if all(x == 0 for x in mo_spin):
        spin_dependent = False
    else:
        spin_dependent = True

    logger.info(f"spin_dependent MOs = {spin_dependent}")

    # MO complex check
    if trexio.has_mo_coefficient_im(file_r):
        # logger.info("The WF is complex")
        # mo_coefficient_imag = trexio.read_mo_coefficient_im(file_r)
        # complex_flag = True
        logger.error("Complex WFs are not supported.")
        raise NotImplementedError
    else:
        pass
        # logger.info("The WF is real")
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
    if store_tuple:
        structure_data = Structure_data(
            pbc_flag=pbc_flag,
            vec_a=(),
            vec_b=(),
            vec_c=(),
            atomic_numbers=tuple(_convert_from_atomic_labels_to_atomic_numbers(labels_r)),
            element_symbols=tuple(labels_r),
            atomic_labels=tuple(labels_r),
            positions=np.array(coords_r),
        )
    else:
        structure_data = Structure_data(
            pbc_flag=pbc_flag,
            vec_a=[],
            vec_b=[],
            vec_c=[],
            atomic_numbers=list(_convert_from_atomic_labels_to_atomic_numbers(labels_r)),
            element_symbols=list(labels_r),
            atomic_labels=list(labels_r),
            positions=np.array(coords_r),
        )

    # ao spherical part check
    if ao_cartesian:
        logger.info("Cartesian basis functions.")
        # AOs_data instance
        ao_num_count = 0
        ao_prim_num_count = 0

        # values to be stored
        nucleus_index = []
        atomic_center_carts = []
        angular_momentums = []
        orbital_indices = []
        exponents = []
        coefficients = []
        polynominal_order_x = []
        polynominal_order_y = []
        polynominal_order_z = []

        for i_shell in range(basis_shell_num):
            # print(f"i_shell={i_shell}")
            b_nucleus_index = basis_nucleus_index[i_shell]
            b_coord = list(coords_r[b_nucleus_index])
            b_ang_mom = basis_shell_ang_mom[i_shell]
            # print(f"b_ang_mom={b_ang_mom}")
            poly_orders = ["".join(p) for p in itertools.combinations_with_replacement("xyz", b_ang_mom)]
            poly_x = [poly_order.count("x") for poly_order in poly_orders]
            poly_y = [poly_order.count("y") for poly_order in poly_orders]
            poly_z = [poly_order.count("z") for poly_order in poly_orders]
            num_ao_mag_moms = len(poly_orders)
            # print(f"num_ao_mag_moms={num_ao_mag_moms}")

            ao_nucleus_index = [b_nucleus_index for _ in range(num_ao_mag_moms)]
            ao_coords = [b_coord for _ in range(num_ao_mag_moms)]
            ao_ang_moms = [b_ang_mom for _ in range(num_ao_mag_moms)]

            # print(f"ao_ang_moms={ao_ang_moms}")

            b_prim_indices = [i for i, v in enumerate(basis_shell_index) if v == i_shell]
            b_prim_num = len(b_prim_indices)

            # print(f"b_prim_indices={b_prim_indices}")
            # print(f"b_prim_num={b_prim_num}")
            # print(f"poly_x={poly_x}")
            # print(f"poly_y={poly_y}")
            # print(f"poly_z={poly_z}")

            N_n_dup_fuctorial_part = [
                (scipy.special.factorial(nx) * scipy.special.factorial(ny) * scipy.special.factorial(nz))
                / (scipy.special.factorial(2 * nx) * scipy.special.factorial(2 * ny) * scipy.special.factorial(2 * nz))
                for nx, ny, nz in zip(poly_x, poly_y, poly_z, strict=True)
            ]

            # print(f"len(N_n_dup_fuctorial_part) = {len(N_n_dup_fuctorial_part)}.")
            # print(f"N_n_dup_fuctorial_part={N_n_dup_fuctorial_part}")

            N_n_dup_Z_part = [
                (2.0 * basis_exponent[k] / np.pi) ** (3.0 / 2.0) * (8.0 * basis_exponent[k]) ** b_ang_mom
                for k in b_prim_indices
            ]
            # print(f"len(N_n_dup_Z_part) = {len(N_n_dup_Z_part)}.")
            b_prim_exponents = [basis_exponent[k] for k in b_prim_indices]

            ao_exponents = b_prim_exponents * num_ao_mag_moms
            ao_coefficients_list = []
            for p in range(num_ao_mag_moms):
                ao_coefficients_list += [
                    basis_shell_factor[i_shell]
                    * basis_prim_factor[k]
                    / np.sqrt(N_n_dup_Z_part[i] * N_n_dup_fuctorial_part[p])
                    * basis_coefficient[k]
                    for i, k in enumerate(b_prim_indices)
                ]

            # print(f"len(ao_coefficients_list) = {len(ao_coefficients_list)}.")

            orbital_indices_all = [ao_num_count + j for j in range(num_ao_mag_moms) for _ in range(b_prim_num)]
            # print(f"orbital_indices_all={orbital_indices_all}")

            ao_coefficients = [
                ao_coefficients_list[k] * ao_normalization[orbital_indices_all[k]] for k in range(len(ao_coefficients_list))
            ]
            ao_num_count += num_ao_mag_moms
            ao_prim_num_count += num_ao_mag_moms * b_prim_num

            nucleus_index += ao_nucleus_index
            atomic_center_carts += ao_coords
            angular_momentums += ao_ang_moms
            polynominal_order_x += poly_x
            polynominal_order_y += poly_y
            polynominal_order_z += poly_z
            orbital_indices += orbital_indices_all
            exponents += ao_exponents
            coefficients += ao_coefficients

        if ao_num_count != ao_num:
            logger.error(f"ao_num_count = {ao_num_count} is inconsistent with the read ao_num = {ao_num}")
            raise ValueError

        if store_tuple:
            aos_data = AOs_cart_data(
                structure_data=structure_data,
                nucleus_index=tuple(nucleus_index),
                num_ao=ao_num_count,
                num_ao_prim=ao_prim_num_count,
                angular_momentums=tuple(angular_momentums),
                polynominal_order_x=tuple(polynominal_order_x),
                polynominal_order_y=tuple(polynominal_order_y),
                polynominal_order_z=tuple(polynominal_order_z),
                orbital_indices=tuple(orbital_indices),
                exponents=tuple(exponents),
                coefficients=tuple(coefficients),
            )
        else:
            aos_data = AOs_cart_data(
                structure_data=structure_data,
                nucleus_index=list(nucleus_index),
                num_ao=ao_num_count,
                num_ao_prim=ao_prim_num_count,
                angular_momentums=list(angular_momentums),
                polynominal_order_x=list(polynominal_order_x),
                polynominal_order_y=list(polynominal_order_y),
                polynominal_order_z=list(polynominal_order_z),
                orbital_indices=list(orbital_indices),
                exponents=list(exponents),
                coefficients=list(coefficients),
            )
    else:
        logger.debug("Spherical basis functions.")
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

        if store_tuple:
            aos_data = AOs_sphe_data(
                structure_data=structure_data,
                nucleus_index=tuple(nucleus_index),
                num_ao=ao_num_count,
                num_ao_prim=ao_prim_num_count,
                angular_momentums=tuple(angular_momentums),
                magnetic_quantum_numbers=tuple(magnetic_quantum_numbers),
                orbital_indices=tuple(orbital_indices),
                exponents=tuple(exponents),
                coefficients=tuple(coefficients),
            )
        else:
            aos_data = AOs_sphe_data(
                structure_data=structure_data,
                nucleus_index=list(nucleus_index),
                num_ao=ao_num_count,
                num_ao_prim=ao_prim_num_count,
                angular_momentums=list(angular_momentums),
                magnetic_quantum_numbers=list(magnetic_quantum_numbers),
                orbital_indices=list(orbital_indices),
                exponents=list(exponents),
                coefficients=list(coefficients),
            )

    # MOs_data instance
    threshold_mo_occ = 1.0e-6

    if (not spin_dependent and not spin_polarized) or (
        not spin_dependent and spin_polarized
    ):  # RHF for closed shell or ORHF for open shell
        mo_indices = [i for (i, v) in enumerate(mo_spin) if v == 0]
        mo_coefficient_real_up = mo_coefficient_real_dn = mo_coefficient_real[mo_indices]
        mo_occ = mo_occupation[mo_indices]
        mo_considered_indices = [i for (i, v) in enumerate(mo_occ) if v >= threshold_mo_occ]
        # mo_considered_occ = mo_occ[mo_considered_indices]
        mo_considered_num = len(mo_considered_indices)
        mo_considered_coefficient_real_up = mo_coefficient_real_up[mo_considered_indices]
        mo_considered_coefficient_real_dn = mo_coefficient_real_dn[mo_considered_indices]

        mos_data_up = MOs_data(num_mo=mo_considered_num, mo_coefficients=mo_considered_coefficient_real_up, aos_data=aos_data)
        mos_data_dn = MOs_data(num_mo=mo_considered_num, mo_coefficients=mo_considered_coefficient_real_dn, aos_data=aos_data)

        num_ele_diff = num_ele_up - num_ele_dn

        mo_lambda_paired = np.pad(
            np.eye(num_ele_dn, dtype=np.float64), ((0, mo_considered_num - num_ele_dn), (0, mo_considered_num - num_ele_dn))
        )

        mo_lambda_unpaired = np.pad(
            np.eye(num_ele_diff, dtype=np.float64), ((num_ele_dn, mo_considered_num - num_ele_dn - num_ele_diff), (0, 0))
        )
        mo_lambda_matrix = np.hstack([mo_lambda_paired, mo_lambda_unpaired])

    elif spin_dependent and spin_polarized:  # UHF for open shell
        mo_indices_up = [i for (i, v) in enumerate(mo_spin) if v == 0]
        mo_indices_dn = [i for (i, v) in enumerate(mo_spin) if v == 1]
        mo_coefficient_real_up = mo_coefficient_real[mo_indices_up]
        mo_coefficient_real_dn = mo_coefficient_real[mo_indices_dn]
        mo_occ_up = mo_occupation[mo_indices_up]
        mo_occ_dn = mo_occupation[mo_indices_dn]

        mo_considered_indices_up = [i for (i, v) in enumerate(mo_occ_up) if v >= threshold_mo_occ]
        mo_considered_indices_dn = [i for (i, v) in enumerate(mo_occ_dn) if v >= threshold_mo_occ]

        if len(mo_considered_indices_up) < len(mo_considered_indices_dn):
            raise ValueError(
                f"The number of occ. orbitals for up spins = {len(mo_considered_indices_up)} should be larger than those of down spins = {len(mo_considered_indices_dn)}."
            )

        mo_considered_num = len(mo_considered_indices_up)
        mo_considered_indices = mo_considered_indices_up[:mo_considered_num]

        mo_considered_coefficient_real_up = mo_coefficient_real_up[mo_considered_indices]
        mo_considered_coefficient_real_dn = mo_coefficient_real_dn[mo_considered_indices]

        mos_data_up = MOs_data(num_mo=mo_considered_num, mo_coefficients=mo_considered_coefficient_real_up, aos_data=aos_data)
        mos_data_dn = MOs_data(num_mo=mo_considered_num, mo_coefficients=mo_considered_coefficient_real_dn, aos_data=aos_data)

        num_ele_diff = num_ele_up - num_ele_dn

        mo_lambda_paired = np.pad(
            np.eye(num_ele_dn, dtype=np.float64), ((0, mo_considered_num - num_ele_dn), (0, mo_considered_num - num_ele_dn))
        )

        mo_lambda_unpaired = np.pad(
            np.eye(num_ele_diff, dtype=np.float64), ((num_ele_dn, mo_considered_num - num_ele_dn - num_ele_diff), (0, 0))
        )
        mo_lambda_matrix = np.hstack([mo_lambda_paired, mo_lambda_unpaired])
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
        if store_tuple:
            coulomb_potential_data = Coulomb_potential_data(
                structure_data=structure_data,
                ecp_flag=True,
                z_cores=tuple(ecp_z_core),
                max_ang_mom_plus_1=tuple(ecp_max_ang_mom_plus_1),
                num_ecps=ecp_num,
                ang_moms=tuple(ecp_ang_mom),
                nucleus_index=tuple(ecp_nucleus_index),
                exponents=tuple(ecp_exponent),
                coefficients=tuple(ecp_coefficient),
                powers=tuple(ecp_power + 2),
            )
        else:
            coulomb_potential_data = Coulomb_potential_data(
                structure_data=structure_data,
                ecp_flag=True,
                z_cores=list(ecp_z_core),
                max_ang_mom_plus_1=list(ecp_max_ang_mom_plus_1),
                num_ecps=ecp_num,
                ang_moms=list(ecp_ang_mom),
                nucleus_index=list(ecp_nucleus_index),
                exponents=list(ecp_exponent),
                coefficients=list(ecp_coefficient),
                powers=list(ecp_power + 2),
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


"""
def _convert_from_atomic_numbers_to_atomic_labels(charges_r: list[int]) -> list[str]:
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
"""


def _convert_from_atomic_labels_to_atomic_numbers(labels_r: list[str]) -> list[int]:
    """Mapping of element symbols to their atomic numbers up to 86."""
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


"""
if __name__ == "__main__":
    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
    log.addHandler(stream_handler)
"""
