# read turborvb WF and PPs and generate jQMC WF and PP instances.
import pickle
import json

import numpy as np

from jqmc.atomic_orbital import AOs_sphe_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data, Jastrow_two_body_data
from jqmc.structure import Structure_data

# read json data
with open("jastrow_turbo_wf.json") as f:
    d = json.load(f)

## structure data
pbc_flag = d["pbc_flag"]  # bool or [bool, bool, bool]
vec_a = np.array(d["vec_a"], dtype=float)
vec_b = np.array(d["vec_b"], dtype=float)
vec_c = np.array(d["vec_c"], dtype=float)
atomic_numbers = np.array(d["atomic_numbers"], dtype=int)
element_symbols = list(d["element_symbols"])
atomic_labels = list(d["atomic_labels"])
positions = np.array(d["positions"], dtype=float)
## jastrow AOs
nucleus_index = list(d["nucleus_index"])
num_ao = int(d["num_ao"])
num_ao_prim = int(d["num_ao_prim"])
angular_momentums = list(d["angular_momentums"])
magnetic_quantum_numbers = list(d["magnetic_quantum_numbers"])
orbital_indices = list(d["orbital_indices"])
exponents = list(d["exponents"])
coefficients = list(d["coefficients"])
# jastrow other parameters
j2_parameter = int(d["j2_parameter"])
j1_matrix = np.array(d["j1_matrix"])
j3_matrix = np.array(d["j3_matrix"])

# structure data
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

# Jastrow AO_datas
jas_aos_data = AOs_sphe_data(
    structure_data=structure_data,
    nucleus_index=nucleus_index,
    num_ao=num_ao,
    num_ao_prim=num_ao_prim,
    angular_momentums=angular_momentums,
    magnetic_quantum_numbers=magnetic_quantum_numbers,
    orbital_indices=orbital_indices,
    exponents=exponents,
    coefficients=coefficients,
)

# Jastrow matrix
j_matrix = np.column_stack((j3_matrix, j1_matrix))

# Jastrow dataclasses
jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=j2_parameter)

jastrow_three_body_data = Jastrow_three_body_data(
    orb_data=jas_aos_data,
    j_matrix=j_matrix,
)

jastrow_data = Jastrow_data(
    jastrow_two_body_data=jastrow_two_body_data,
    jastrow_three_body_data=jastrow_three_body_data,
)

with open("jastrow_data.pkl", "wb") as f:
    pickle.dump(jastrow_data, f)
