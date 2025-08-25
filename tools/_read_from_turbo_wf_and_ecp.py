# read turborvb WF and PPs and generate jQMC WF and PP instances.
import numpy as np
from turbogenius.pyturbo.io_fort10 import IO_fort10
from turbogenius.pyturbo.pseudopotentials import Pseudopotentials

from jqmc.atomic_orbital import AOs_data
from jqmc.coulomb_potential import Coulomb_potential_data
from jqmc.determinant import Geminal_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data, Jastrow_two_body_data
from jqmc.molecular_orbital import MOs_data, compute_MOs_api
from jqmc.structure import Structure_data
from jqmc.vmc import VMC

# structure
io_fort10 = IO_fort10()
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

structure_data = Structure_data(
    pbc_flag=pbc_flag,
    vec_a=vec_a,
    vec_b=vec_b,
    vec_c=vec_c,
    atomic_numbers=atomic_numbers,
    element_symbols=element_symbols,
    atomic_labels=atomic_labels,
    positions=positions,
)

# coulomb
if io_fort10.pp_flag:
    pseudopotentials = Pseudopotentials.parse_pseudopotential_from_turborvb_pseudo_dat(file="pseudo.dat")

    z_cores = [num_atomic - num_valence for num_atomic, num_valence in zip(atomic_numbers, valence_electrons)]
    max_ang_mom_plus_1 = pseudopotentials.max_ang_mom_plus_1
    ang_moms = pseudopotentials.ang_mom
    nucleus_index = pseudopotentials.nucleus_index
    exponents = pseudopotentials.exponent
    coefficients = pseudopotentials.coefficient
    powers = pseudopotentials.power

    num_ecps = [nucleus_index.count(i) for i in set(nucleus_index)]

    coulomb_potential_data = Coulomb_potential_data(
        structure_data=structure_data,
        ecp_flag=True,
        z_cores=z_cores,
        max_ang_mom_plus_1=max_ang_mom_plus_1,
        num_ecps=num_ecps,
        ang_moms=ang_moms,
        nucleus_index=nucleus_index,
        exponents=exponents,
        coefficients=coefficients,
        powers=powers,
    )
else:
    coulomb_potential_data = Coulomb_potential_data(structure_data=structure_data, ecp_flag=False)

# print(coulomb_potential_data)

# atomic basis in determinant
f10detbasissets = io_fort10.f10detbasissets
f10detocc = io_fort10.f10detocc
det_basis_sets = f10detbasissets.det_basis_sets

structure_data = structure_data
basis_nucleus_index = det_basis_sets.nucleus_index
basis_shell_num = det_basis_sets.shell_num
basis_shell_ang_mom = det_basis_sets.shell_ang_mom
basis_shell_index = det_basis_sets.shell_index
basis_exponent = det_basis_sets.exponent
basis_coefficient = det_basis_sets.coefficient

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

det_aos_data_dn = det_aos_data_up = AOs_data(
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

# mo_coefficients / MO_data
num_mo_up = 0
mo_coefficients_up = 0
det_mos_data_up = MOs_data(num_mo=num_mo_up, mo_coefficients=mo_coefficients_up, aos_data=det_aos_data_up)
num_mo_dn = 0
mo_coefficients_dn = 0
det_mos_data_dn = MOs_data(num_mo=num_mo_dn, mo_coefficients=mo_coefficients_dn, aos_data=det_aos_data_dn)

# Geminal
f10detmatrix = io_fort10.f10detmatrix
# print(f10detmatrix.row)
# print(f10detmatrix.col)
# print(f10detmatrix.coeff_real)

num_electron_up = io_fort10.header.nelup
num_electron_dn = io_fort10.header.neldn
lambda_matrix = np.diag((num_electron_up, num_electron_dn))

geminal_data = Geminal_data(
    num_electron_up=num_electron_up,
    num_electron_dn=num_electron_dn,
    orb_data_up_spin=det_mos_data_up,
    orb_data_dn_spin=det_mos_data_dn,
    compute_orb_api=compute_MOs_api,
    lambda_matrix=lambda_matrix,
)

"""
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

    orbital_indices_all = [
        ao_num_count + j for j in range(num_mag_moms) for _ in range(ao_prim_num)
    ]
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

jas_aos_data_dn = jas_aos_data_up = AOs_data(
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
# print(f10jasmatrix.jasmat)
# print(f10jasmatrix.row)
# print(f10jasmatrix.col)
# print(f10jasmatrix.coeff)

j3_matrix_up_up = np.zeros((jas_aos_data_up.num_ao, jas_aos_data_up.num_ao))
j3_matrix_dn_dn = np.zeros((jas_aos_data_dn.num_ao, jas_aos_data_dn.num_ao))
j3_matrix_up_dn = np.zeros((jas_aos_data_up.num_ao, jas_aos_data_dn.num_ao))
"""
