# read turborvb WF and PPs and generate jQMC WF and PP instances.
import pickle

import numpy as np
from turbogenius.pyturbo.io_fort10 import IO_fort10

from jqmc.atomic_orbital import AOs_sphe_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_three_body_data, Jastrow_two_body_data
from jqmc.structure import Structure_data

suffix_list = ["w_2b_1b3b_w_ecp", "w_2b_3b_w_ecp", "w_1b_2b_1b3b_ae"]

for suffix in suffix_list:
    turborvb_WF_file = f"turborvb_WF_{suffix}.txt"

    # fort10
    io_fort10 = IO_fort10(turborvb_WF_file)
    num_ele = io_fort10.f10header.nel

    # jastrow twobody
    f10jastwobody = io_fort10.f10jastwobody
    if f10jastwobody.jastrow_type not in (-5, -15):
        raise NotImplementedError

    if f10jastwobody.jastrow_type == -5:
        assert len(f10jastwobody.twobody_list) == 1
        twobody_parameter = f10jastwobody.twobody_list[0]
    elif f10jastwobody.jastrow_type == -15:
        assert len(f10jastwobody.twobody_list) == 1
        assert len(f10jastwobody.onebody_list) == 1
        twobody_parameter = f10jastwobody.twobody_list[0]
        onebody_parameter = f10jastwobody.onebody_list[0]
    else:
        raise NotImplementedError
    # print(twobody_parameter)

    # structure
    f10structure = io_fort10.f10structure
    pbc_flag = f10structure.pbc_flag
    vec_a = f10structure.vec_a
    vec_b = f10structure.vec_b
    vec_c = f10structure.vec_c
    atomic_numbers = [int(N) for N in f10structure.atomic_numbers]
    valence_electrons = [int(N) for N in f10structure.valence_electrons]
    element_symbols = f10structure.structure.element_symbols
    atomic_labels = f10structure.structure.element_symbols
    positions = f10structure.positions
    core_electrons = [Z - Z_val for Z, Z_val in zip(atomic_numbers, valence_electrons)]

    pbc_flag = bool(pbc_flag)
    vec_a = tuple(vec_a)
    vec_b = tuple(vec_b)
    vec_c = tuple(vec_c)
    atomic_numbers = tuple(atomic_numbers)
    valence_electrons = tuple(valence_electrons)
    element_symbols = tuple(element_symbols)
    atomic_labels = tuple(atomic_labels)
    core_electrons = tuple(core_electrons)
    positions = np.array(positions)

    structure_data = Structure_data(
        pbc_flag=pbc_flag,
        vec_a=(),
        vec_b=(),
        vec_c=(),
        atomic_numbers=atomic_numbers,
        element_symbols=element_symbols,
        atomic_labels=atomic_labels,
        positions=positions,
    )

    # Jastrow AO_datas
    f10jasbasissets = io_fort10.f10jasbasissets
    f10jasocc = io_fort10.f10jasocc
    jas_basis_sets = f10jasbasissets.jas_basis_sets

    structure_data = structure_data
    basis_nucleus_index = jas_basis_sets.nucleus_index
    basis_shell_num = jas_basis_sets.shell_num
    basis_shell_ang_mom = jas_basis_sets.shell_ang_mom
    basis_shell_index = jas_basis_sets.shell_index
    basis_exponent = jas_basis_sets.exponent
    basis_coefficient = jas_basis_sets.coefficient

    ao_num_count = 0
    ao_prim_num_count = 0

    nucleus_index = []
    angular_momentums = []
    magnetic_quantum_numbers = []
    orbital_indices = []
    exponents = []
    coefficients = []

    for i_shell in range(basis_shell_num):
        ao_nucleus_index = basis_nucleus_index[i_shell]
        ao_ang_mom = basis_shell_ang_mom[i_shell]
        if ao_ang_mom == 0:
            ao_mag_moms = [0]  # s orbital
        elif ao_ang_mom == 1:
            # "(makefun notation): px(m=+1), py(m=-1), pz(m=0)")
            ao_mag_moms = [+1, -1, 0]  # p orbital
        elif ao_ang_mom == 2:
            # "(makefun notation): dz2(m=0), dx2-y2(m=+2), dxy(m=-2), dyz(m=-1), dzx=(m=+1)"
            ao_mag_moms = [0, +2, -2, -1, +1]  # d orbital
        else:
            # "(makefun notation): l=0, l=+-1, l=+2, ...
            ao_mag_moms = [0] + [i * (-1) ** j for i in range(1, ao_ang_mom + 1) for j in range(2)]
        num_mag_moms = len(ao_mag_moms)

        ao_nucleus_index_dup = [ao_nucleus_index for _ in range(num_mag_moms)]
        ao_ang_moms = [ao_ang_mom for _ in range(num_mag_moms)]

        ao_prim_indices = [i for i, v in enumerate(basis_shell_index) if v == i_shell]
        ao_prim_num = len(ao_prim_indices)
        ao_exponents = [basis_exponent[k] for k in ao_prim_indices]
        ao_coefficients = [basis_coefficient[k] for k in ao_prim_indices]

        orbital_indices_all = [ao_num_count + j for j in range(num_mag_moms) for _ in range(ao_prim_num)]
        ao_exponents_all = ao_exponents * num_mag_moms
        ao_coefficients_all = ao_coefficients * num_mag_moms

        ao_num_count += num_mag_moms
        ao_prim_num_count += num_mag_moms * ao_prim_num

        nucleus_index += ao_nucleus_index_dup
        angular_momentums += ao_ang_moms
        magnetic_quantum_numbers += ao_mag_moms
        orbital_indices += orbital_indices_all
        exponents += ao_exponents_all
        coefficients += ao_coefficients_all

    nucleus_index = tuple(nucleus_index)
    angular_momentums = tuple(angular_momentums)
    magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)
    orbital_indices = tuple(orbital_indices)
    exponents = tuple(exponents)
    coefficients = tuple(coefficients)

    jas_aos_data = AOs_sphe_data(
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

    # Jastrow matrix
    f10jasmatrix = io_fort10.f10jasmatrix

    max_row = max(f10jasmatrix.row)
    max_col = max(f10jasmatrix.col)

    assert max_row == max_col
    const_jas_orb_index = max_row

    j1_matrix = np.zeros((jas_aos_data.num_ao))

    for i, (row, col) in enumerate(zip(f10jasmatrix.row, f10jasmatrix.col)):
        if col == const_jas_orb_index and row != const_jas_orb_index:
            j1_matrix[row - 1] = f10jasmatrix.coeff[i] * (num_ele - 1)

    # print(j1_matrix_up.T)
    # print(j1_matrix_dn.T)

    j3_matrix = np.zeros((jas_aos_data.num_ao, jas_aos_data.num_ao))

    for i, (row, col) in enumerate(zip(f10jasmatrix.row, f10jasmatrix.col)):
        if col != const_jas_orb_index and row != const_jas_orb_index:
            # upper diagonal
            j3_matrix[row - 1, col - 1] = f10jasmatrix.coeff[i]

            # lower diagonal
            j3_matrix[col - 1, row - 1] = f10jasmatrix.coeff[i]

    # print(j3_matrix_up_up)
    # print(j3_matrix_dn_dn)
    # print(j3_matrix_up_dn)

    j_matrix = np.column_stack((j3_matrix, j1_matrix))

    jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=twobody_parameter)

    jastrow_three_body_data = Jastrow_three_body_data(
        orb_data=jas_aos_data,
        j_matrix=j_matrix,
    )

    if f10jastwobody.jastrow_type == -15:
        jastrow_one_body_data = Jastrow_one_body_data(
            jastrow_1b_param=onebody_parameter, structure_data=structure_data, core_electrons=core_electrons
        )

    if f10jastwobody.jastrow_type == -5:
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=jastrow_two_body_data,
            jastrow_three_body_data=jastrow_three_body_data,
        )
    elif f10jastwobody.jastrow_type == -15:
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=jastrow_one_body_data,
            jastrow_two_body_data=jastrow_two_body_data,
            jastrow_three_body_data=jastrow_three_body_data,
        )
    else:
        raise NotImplementedError

    with open(f"jastrow_data_{suffix}.pkl", "wb") as f:
        pickle.dump(jastrow_data, f)
