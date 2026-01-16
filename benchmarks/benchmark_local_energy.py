"""Benchmark compute_local_energy vs compute_local_energy_auto.

Uses trexio file benchmarks/trexio_example_files/water_ccecp_ccpvqz.h5.
All timings include `.block_until_ready()` so async JAX execution is fully measured.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from jqmc.hamiltonians import Hamiltonian_data, _compute_local_energy_auto, compute_local_energy
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

REPEATS = 5
SEED = 0

TREXIO_FILE = os.path.join(
    os.path.dirname(__file__),
    "water_ccecp_ccpvqz.h5",
)

(
    structure_data,
    _,
    _,
    _,
    geminal_mo_data,
    coulomb_potential_data,
) = read_trexio_file(trexio_file=TREXIO_FILE, store_tuple=True)

jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

jastrow_data = Jastrow_data(
    jastrow_one_body_data=None,
    jastrow_two_body_data=jastrow_twobody_data,
    jastrow_three_body_data=None,
)

wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

hamiltonian_data = Hamiltonian_data(
    structure_data=structure_data,
    coulomb_potential_data=coulomb_potential_data,
    wavefunction_data=wavefunction_data,
)

num_ele_up = geminal_mo_data.num_electron_up
num_ele_dn = geminal_mo_data.num_electron_dn

rng = np.random.default_rng(SEED)
r_cart_min, r_cart_max = -5.0, +5.0
r_up_carts_np = (r_cart_max - r_cart_min) * rng.random((num_ele_up, 3)) + r_cart_min
r_dn_carts_np = (r_cart_max - r_cart_min) * rng.random((num_ele_dn, 3)) + r_cart_min

r_up_carts = jnp.asarray(r_up_carts_np)
r_dn_carts = jnp.asarray(r_dn_carts_np)

RT = jnp.eye(3)

# Warmup
out_analytic = compute_local_energy(hamiltonian_data, r_up_carts, r_dn_carts, RT)
for leaf in jax.tree_util.tree_leaves(out_analytic):
    leaf.block_until_ready()

out_auto = _compute_local_energy_auto(hamiltonian_data, r_up_carts, r_dn_carts, RT)
for leaf in jax.tree_util.tree_leaves(out_auto):
    leaf.block_until_ready()

analytic_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = compute_local_energy(hamiltonian_data, r_up_carts, r_dn_carts, RT)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    analytic_times.append(time.perf_counter() - start)

auto_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = _compute_local_energy_auto(hamiltonian_data, r_up_carts, r_dn_carts, RT)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    auto_times.append(time.perf_counter() - start)

print("Local energy benchmark (seconds, mean over repeats):")
print(f"  analytic : {np.mean(analytic_times):.6f}")
print(f"  autodiff : {np.mean(auto_times):.6f}")
