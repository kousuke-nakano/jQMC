"""Benchmark analytic vs autodiff Jastrow gradients and Laplacians.

Configure parameters below, run the script, and timings will be printed.
All timings include `.block_until_ready()` so async JAX execution is fully measured.

Two patterns are benchmarked:
- J1 + J2 + J3
- J1 + J2 + J3 + JNN (analytic baseline uses J1/J2/J3 analytic + JNN autodiff)
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from jqmc.atomic_orbital import AOs_sphe_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_NN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _compute_grads_and_laplacian_Jastrow_part_auto,
    compute_grads_and_laplacian_Jastrow_one_body,
    compute_grads_and_laplacian_Jastrow_three_body,
    compute_grads_and_laplacian_Jastrow_two_body,
)
from jqmc.structure import Structure_data

NUM_UP = 16
NUM_DN = 16
NUM_NUC = 4
NUM_AO = 24
NUM_AO_PRIM = 24
L_MAX = 3
REPEATS = 5
SEED = 0

J1_PARAM = 1.0
J2_PARAM = 1.0

NN_HIDDEN_DIM = 32
NN_NUM_LAYERS = 3
NN_NUM_RBF = 16
NN_CUTOFF = 5.0
NN_PARAMS_SCALE = 1.0e-10

rng = np.random.default_rng(SEED)
positions = rng.uniform(-1.0, 1.0, size=(NUM_NUC, 3))
structure_data = Structure_data(
    pbc_flag=False,
    positions=positions,
    atomic_numbers=tuple([6] * NUM_NUC),
    element_symbols=tuple(["X"] * NUM_NUC),
    atomic_labels=tuple(["X"] * NUM_NUC),
)
structure_data.sanity_check()

core_electrons = tuple([3] * NUM_NUC)

rng_e = np.random.default_rng(SEED + 1)
r_up_carts = rng_e.uniform(-2.0, 2.0, size=(NUM_UP, 3))
r_dn_carts = rng_e.uniform(-2.0, 2.0, size=(NUM_DN, 3))

jastrow_one_body_data = Jastrow_one_body_data(
    jastrow_1b_param=J1_PARAM,
    structure_data=structure_data,
    core_electrons=core_electrons,
)

jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=J2_PARAM)

orbital_indices = tuple(range(NUM_AO))
nucleus_index = tuple(i % NUM_NUC for i in range(NUM_AO))
exponents = tuple(rng.uniform(0.5, 2.0) for _ in range(NUM_AO))
coefficients = tuple(rng.uniform(0.5, 1.5) for _ in range(NUM_AO))
angular_momentums = []
magnetic_quantum_numbers = []
for _ in range(NUM_AO):
    l_val = int(rng.integers(0, L_MAX + 1))
    m_val = int(rng.integers(-l_val, l_val + 1)) if l_val > 0 else 0
    angular_momentums.append(l_val)
    magnetic_quantum_numbers.append(m_val)

orb_data = AOs_sphe_data(
    structure_data=structure_data,
    nucleus_index=nucleus_index,
    num_ao=NUM_AO,
    num_ao_prim=NUM_AO_PRIM,
    orbital_indices=orbital_indices,
    exponents=exponents,
    coefficients=coefficients,
    angular_momentums=tuple(angular_momentums),
    magnetic_quantum_numbers=tuple(magnetic_quantum_numbers),
)
orb_data.sanity_check()

j_matrix = rng.uniform(0.0, 1.0, size=(NUM_AO, NUM_AO + 1))
jastrow_three_body_data = Jastrow_three_body_data(orb_data=orb_data, j_matrix=j_matrix)

print(
    "Configured Jastrow:",
    f"num_up={NUM_UP}",
    f"num_dn={NUM_DN}",
    f"num_nuc={NUM_NUC}",
    f"num_ao={NUM_AO}",
    f"l_max={L_MAX}",
)


def _analytic_j123(r_up, r_dn):
    grad_J1_up, grad_J1_dn, lap_J1_up, lap_J1_dn = compute_grads_and_laplacian_Jastrow_one_body(
        jastrow_one_body_data,
        r_up,
        r_dn,
    )
    grad_J2_up, grad_J2_dn, lap_J2_up, lap_J2_dn = compute_grads_and_laplacian_Jastrow_two_body(
        jastrow_two_body_data,
        r_up,
        r_dn,
    )
    grad_J3_up, grad_J3_dn, lap_J3_up, lap_J3_dn = compute_grads_and_laplacian_Jastrow_three_body(
        jastrow_three_body_data,
        r_up,
        r_dn,
    )

    grad_up = jnp.asarray(grad_J1_up) + jnp.asarray(grad_J2_up) + jnp.asarray(grad_J3_up)
    grad_dn = jnp.asarray(grad_J1_dn) + jnp.asarray(grad_J2_dn) + jnp.asarray(grad_J3_dn)
    lap_up = jnp.asarray(lap_J1_up) + jnp.asarray(lap_J2_up) + jnp.asarray(lap_J3_up)
    lap_dn = jnp.asarray(lap_J1_dn) + jnp.asarray(lap_J2_dn) + jnp.asarray(lap_J3_dn)
    return grad_up, grad_dn, lap_up, lap_dn


# --- J1 + J2 + J3 benchmark ---
jastrow_data = Jastrow_data(
    jastrow_one_body_data=jastrow_one_body_data,
    jastrow_two_body_data=jastrow_two_body_data,
    jastrow_three_body_data=jastrow_three_body_data,
    jastrow_nn_data=None,
)

analytic_out = _analytic_j123(r_up_carts, r_dn_carts)
for leaf in jax.tree_util.tree_leaves(analytic_out):
    leaf.block_until_ready()

auto_out = _compute_grads_and_laplacian_Jastrow_part_auto(jastrow_data, r_up_carts, r_dn_carts)
for leaf in jax.tree_util.tree_leaves(auto_out):
    leaf.block_until_ready()

analytic_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = _analytic_j123(r_up_carts, r_dn_carts)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    analytic_times.append(time.perf_counter() - start)

auto_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = _compute_grads_and_laplacian_Jastrow_part_auto(jastrow_data, r_up_carts, r_dn_carts)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    auto_times.append(time.perf_counter() - start)

print("\nJ1+J2+J3 (seconds, mean over repeats):")
print(f"  analytic : {np.mean(analytic_times):.6f}")
print(f"  autodiff : {np.mean(auto_times):.6f}")

# --- J1 + J2 + J3 + JNN benchmark ---
jastrow_nn_data = Jastrow_NN_data.init_from_structure(
    structure_data=structure_data,
    hidden_dim=NN_HIDDEN_DIM,
    num_layers=NN_NUM_LAYERS,
    num_rbf=NN_NUM_RBF,
    cutoff=NN_CUTOFF,
    key=jax.random.PRNGKey(SEED + 2),
)
if NN_PARAMS_SCALE != 1.0e-10:
    flat = jastrow_nn_data.flatten_fn(jastrow_nn_data.params)
    scale = NN_PARAMS_SCALE / 1.0e-10
    params_rescaled = jastrow_nn_data.unflatten_fn(flat * scale)
    jastrow_nn_data = jastrow_nn_data.replace(params=params_rescaled)

jastrow_data_nn = Jastrow_data(
    jastrow_one_body_data=jastrow_one_body_data,
    jastrow_two_body_data=jastrow_two_body_data,
    jastrow_three_body_data=jastrow_three_body_data,
    jastrow_nn_data=jastrow_nn_data,
)

jastrow_nn_only = Jastrow_data(
    jastrow_one_body_data=None,
    jastrow_two_body_data=None,
    jastrow_three_body_data=None,
    jastrow_nn_data=jastrow_nn_data,
)

analytic_nn_out = _analytic_j123(r_up_carts, r_dn_carts)
for leaf in jax.tree_util.tree_leaves(analytic_nn_out):
    leaf.block_until_ready()

nn_out = _compute_grads_and_laplacian_Jastrow_part_auto(jastrow_nn_only, r_up_carts, r_dn_carts)
for leaf in jax.tree_util.tree_leaves(nn_out):
    leaf.block_until_ready()

auto_nn_out = _compute_grads_and_laplacian_Jastrow_part_auto(jastrow_data_nn, r_up_carts, r_dn_carts)
for leaf in jax.tree_util.tree_leaves(auto_nn_out):
    leaf.block_until_ready()

analytic_nn_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out_j123 = _analytic_j123(r_up_carts, r_dn_carts)
    out_nn = _compute_grads_and_laplacian_Jastrow_part_auto(jastrow_nn_only, r_up_carts, r_dn_carts)
    out = (
        jnp.asarray(out_j123[0]) + jnp.asarray(out_nn[0]),
        jnp.asarray(out_j123[1]) + jnp.asarray(out_nn[1]),
        jnp.asarray(out_j123[2]) + jnp.asarray(out_nn[2]),
    )
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    analytic_nn_times.append(time.perf_counter() - start)

auto_nn_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = _compute_grads_and_laplacian_Jastrow_part_auto(jastrow_data_nn, r_up_carts, r_dn_carts)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    auto_nn_times.append(time.perf_counter() - start)

print("\nJ1+J2+J3+JNN (seconds, mean over repeats):")
print("  analytic+autodiff : (J1/J2/J3 analytic + JNN autodiff)")
print(f"                   : {np.mean(analytic_nn_times):.6f}")
print(f"  autodiff          : {np.mean(auto_nn_times):.6f}")

jax.tree_util.tree_leaves((analytic_out, auto_out, analytic_nn_out, auto_nn_out))
