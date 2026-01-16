"""Benchmark compute_kinetic_energy vs compute_kinetic_energy_auto.

Uses trexio file benchmarks/trexio_example_files/water_ccecp_ccpvqz.h5.
All timings include `.block_until_ready()` so async JAX execution is fully measured.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from jqmc.jastrow_factor import Jastrow_data, Jastrow_NN_data, Jastrow_two_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import (
    Wavefunction_data,
    _compute_kinetic_energy_all_elements_auto,
    _compute_kinetic_energy_auto,
    compute_kinetic_energy,
    compute_kinetic_energy_all_elements,
)

REPEATS = 3
SEED = 0
BENCH_NN = False

NN_HIDDEN_DIM = 32
NN_NUM_LAYERS = 3
NN_NUM_RBF = 16
NN_CUTOFF = 5.0

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
    _,
) = read_trexio_file(trexio_file=TREXIO_FILE, store_tuple=True)

jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

jastrow_data = Jastrow_data(
    jastrow_one_body_data=None,
    jastrow_two_body_data=jastrow_twobody_data,
    jastrow_three_body_data=None,
)

wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

num_ele_up = geminal_mo_data.num_electron_up
num_ele_dn = geminal_mo_data.num_electron_dn

rng = np.random.default_rng(SEED)
r_cart_min, r_cart_max = -5.0, +5.0
r_up_carts_np = (r_cart_max - r_cart_min) * rng.random((num_ele_up, 3)) + r_cart_min
r_dn_carts_np = (r_cart_max - r_cart_min) * rng.random((num_ele_dn, 3)) + r_cart_min

r_up_carts = jnp.asarray(r_up_carts_np)
r_dn_carts = jnp.asarray(r_dn_carts_np)

# Warmup
out_analytic = compute_kinetic_energy(wavefunction_data, r_up_carts, r_dn_carts)
for leaf in jax.tree_util.tree_leaves(out_analytic):
    leaf.block_until_ready()

out_auto = _compute_kinetic_energy_auto(wavefunction_data, r_up_carts, r_dn_carts)
for leaf in jax.tree_util.tree_leaves(out_auto):
    leaf.block_until_ready()

analytic_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = compute_kinetic_energy(wavefunction_data, r_up_carts, r_dn_carts)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    analytic_times.append(time.perf_counter() - start)

auto_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = _compute_kinetic_energy_auto(wavefunction_data, r_up_carts, r_dn_carts)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    auto_times.append(time.perf_counter() - start)

print("Kinetic energy benchmark (seconds, mean over repeats):")
print("  (no NN)")
print(f"    analytic : {np.mean(analytic_times):.6f}")
print(f"    autodiff : {np.mean(auto_times):.6f}")

out_all_elements_analytic = compute_kinetic_energy_all_elements(wavefunction_data, r_up_carts, r_dn_carts)
for leaf in jax.tree_util.tree_leaves(out_all_elements_analytic):
    leaf.block_until_ready()

out_all_elements_auto = _compute_kinetic_energy_all_elements_auto(wavefunction_data, r_up_carts, r_dn_carts)
for leaf in jax.tree_util.tree_leaves(out_all_elements_auto):
    leaf.block_until_ready()

all_elements_analytic_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = compute_kinetic_energy_all_elements(wavefunction_data, r_up_carts, r_dn_carts)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    all_elements_analytic_times.append(time.perf_counter() - start)

all_elements_auto_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = _compute_kinetic_energy_all_elements_auto(wavefunction_data, r_up_carts, r_dn_carts)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    all_elements_auto_times.append(time.perf_counter() - start)

print("    all_elements analytic : {:.6f}".format(np.mean(all_elements_analytic_times)))
print("    all_elements autodiff : {:.6f}".format(np.mean(all_elements_auto_times)))

if BENCH_NN:
    jastrow_nn_data = Jastrow_NN_data.init_from_structure(
        structure_data=structure_data,
        hidden_dim=NN_HIDDEN_DIM,
        num_layers=NN_NUM_LAYERS,
        num_rbf=NN_NUM_RBF,
        cutoff=NN_CUTOFF,
        key=jax.random.PRNGKey(SEED + 2),
    )
    jastrow_data_nn = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=None,
        jastrow_nn_data=jastrow_nn_data,
    )
    wavefunction_data_nn = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data_nn)

    out_analytic = compute_kinetic_energy(wavefunction_data_nn, r_up_carts, r_dn_carts)
    for leaf in jax.tree_util.tree_leaves(out_analytic):
        leaf.block_until_ready()

    out_auto = _compute_kinetic_energy_auto(wavefunction_data_nn, r_up_carts, r_dn_carts)
    for leaf in jax.tree_util.tree_leaves(out_auto):
        leaf.block_until_ready()

    analytic_times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = compute_kinetic_energy(wavefunction_data_nn, r_up_carts, r_dn_carts)
        for leaf in jax.tree_util.tree_leaves(out):
            leaf.block_until_ready()
        analytic_times.append(time.perf_counter() - start)

    auto_times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = _compute_kinetic_energy_auto(wavefunction_data_nn, r_up_carts, r_dn_carts)
        for leaf in jax.tree_util.tree_leaves(out):
            leaf.block_until_ready()
        auto_times.append(time.perf_counter() - start)

    print("  (with NN)")
    print(f"    analytic : {np.mean(analytic_times):.6f}")
    print(f"    autodiff : {np.mean(auto_times):.6f}")

    out_all_elements_analytic = compute_kinetic_energy_all_elements(wavefunction_data_nn, r_up_carts, r_dn_carts)
    for leaf in jax.tree_util.tree_leaves(out_all_elements_analytic):
        leaf.block_until_ready()

    out_all_elements_auto = _compute_kinetic_energy_all_elements_auto(wavefunction_data_nn, r_up_carts, r_dn_carts)
    for leaf in jax.tree_util.tree_leaves(out_all_elements_auto):
        leaf.block_until_ready()

    all_elements_analytic_times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = compute_kinetic_energy_all_elements(wavefunction_data_nn, r_up_carts, r_dn_carts)
        for leaf in jax.tree_util.tree_leaves(out):
            leaf.block_until_ready()
        all_elements_analytic_times.append(time.perf_counter() - start)

    all_elements_auto_times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = _compute_kinetic_energy_all_elements_auto(wavefunction_data_nn, r_up_carts, r_dn_carts)
        for leaf in jax.tree_util.tree_leaves(out):
            leaf.block_until_ready()
        all_elements_auto_times.append(time.perf_counter() - start)

    print("    all_elements analytic : {:.6f}".format(np.mean(all_elements_analytic_times)))
    print("    all_elements autodiff : {:.6f}".format(np.mean(all_elements_auto_times)))
