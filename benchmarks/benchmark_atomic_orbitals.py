"""Benchmark analytic vs autodiff AO gradients and Laplacians (no helper functions).

Configure parameters below, run the script, and timings will be printed.
All timings include `.block_until_ready()` so async JAX execution is fully measured.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from jqmc.atomic_orbital import (
    AOs_cart_data,
    AOs_sphe_data,
    _compute_AOs_grad_autodiff,
    _compute_AOs_laplacian_autodiff,
    compute_AOs_grad,
    compute_AOs_laplacian,
)
from jqmc.structure import Structure_data

NUM_AO = 32
NUM_ELECTRONS = 32
NUM_CENTERS = 4
L_MAX = 3
REPEATS = 5
SEED = 0

rng = np.random.default_rng(SEED)
positions = rng.uniform(-1.0, 1.0, size=(NUM_CENTERS, 3))
structure_data = Structure_data(
    pbc_flag=False,
    positions=positions,
    atomic_numbers=tuple([0] * NUM_CENTERS),
    element_symbols=tuple(["X"] * NUM_CENTERS),
    atomic_labels=tuple(["X"] * NUM_CENTERS),
)
structure_data.sanity_check()

orbital_indices = tuple(range(NUM_AO))
nucleus_index = tuple(i % NUM_CENTERS for i in range(NUM_AO))
exponents = tuple(rng.uniform(0.5, 2.0) for _ in range(NUM_AO))
coefficients = tuple(rng.uniform(0.5, 1.5) for _ in range(NUM_AO))
angular_momentums = []
magnetic_quantum_numbers = []
for _ in range(NUM_AO):
    l_val = int(rng.integers(0, L_MAX + 1))
    m_val = int(rng.integers(-l_val, l_val + 1)) if l_val > 0 else 0
    angular_momentums.append(l_val)
    magnetic_quantum_numbers.append(m_val)

aos_data = AOs_sphe_data(
    structure_data=structure_data,
    nucleus_index=nucleus_index,
    num_ao=NUM_AO,
    num_ao_prim=NUM_AO,
    orbital_indices=orbital_indices,
    exponents=exponents,
    coefficients=coefficients,
    angular_momentums=tuple(angular_momentums),
    magnetic_quantum_numbers=tuple(magnetic_quantum_numbers),
)
aos_data.sanity_check()

rng_e = np.random.default_rng(SEED + 1)
r_carts = rng_e.uniform(-2.0, 2.0, size=(NUM_ELECTRONS, 3))
r_carts_jnp = jnp.asarray(r_carts)

print(
    "Configured spherical AOs:",
    f"num_ao={NUM_AO}",
    f"num_centers={NUM_CENTERS}",
    f"l_max={L_MAX}",
    f"num_electrons={NUM_ELECTRONS}",
)

grad_analytic_out = compute_AOs_grad(aos_data, r_carts_jnp)
for leaf in jax.tree_util.tree_leaves(grad_analytic_out):
    leaf.block_until_ready()

grad_autodiff_out = _compute_AOs_grad_autodiff(aos_data, r_carts_jnp)
for leaf in jax.tree_util.tree_leaves(grad_autodiff_out):
    leaf.block_until_ready()

lap_analytic_out = compute_AOs_laplacian(aos_data, r_carts_jnp)
for leaf in jax.tree_util.tree_leaves(lap_analytic_out):
    leaf.block_until_ready()

lap_autodiff_out = _compute_AOs_laplacian_autodiff(aos_data, r_carts_jnp)
for leaf in jax.tree_util.tree_leaves(lap_autodiff_out):
    leaf.block_until_ready()

grad_analytic_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = compute_AOs_grad(aos_data, r_carts_jnp)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    grad_analytic_times.append(time.perf_counter() - start)

grad_autodiff_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = _compute_AOs_grad_autodiff(aos_data, r_carts_jnp)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    grad_autodiff_times.append(time.perf_counter() - start)

lap_analytic_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = compute_AOs_laplacian(aos_data, r_carts_jnp)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    lap_analytic_times.append(time.perf_counter() - start)

lap_autodiff_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = _compute_AOs_laplacian_autodiff(aos_data, r_carts_jnp)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    lap_autodiff_times.append(time.perf_counter() - start)

print("\nGradients (seconds, mean over repeats):")
print(f"  analytic : {np.mean(grad_analytic_times):.6f}")
print(f"  autodiff : {np.mean(grad_autodiff_times):.6f}")

print("\nLaplacians (seconds, mean over repeats):")
print(f"  analytic : {np.mean(lap_analytic_times):.6f}")
print(f"  autodiff : {np.mean(lap_autodiff_times):.6f}")

jax.tree_util.tree_leaves((grad_analytic_out, grad_autodiff_out, lap_analytic_out, lap_autodiff_out))


# Cartesian AO benchmark
cart_orbital_indices = tuple(range(NUM_AO))
cart_nucleus_index = tuple(i % NUM_CENTERS for i in range(NUM_AO))
cart_exponents = tuple(rng.uniform(0.5, 2.0) for _ in range(NUM_AO))
cart_coefficients = tuple(rng.uniform(0.5, 1.5) for _ in range(NUM_AO))
cart_angular_momentums = []
cart_nx = []
cart_ny = []
cart_nz = []
for _ in range(NUM_AO):
    l_val = int(rng.integers(0, L_MAX + 1))
    if l_val == 0:
        nx_val, ny_val, nz_val = 0, 0, 0
    else:
        nx_val = int(rng.integers(0, l_val + 1))
        ny_val = int(rng.integers(0, l_val - nx_val + 1))
        nz_val = l_val - nx_val - ny_val
    cart_angular_momentums.append(l_val)
    cart_nx.append(nx_val)
    cart_ny.append(ny_val)
    cart_nz.append(nz_val)

aos_cart = AOs_cart_data(
    structure_data=structure_data,
    nucleus_index=cart_nucleus_index,
    num_ao=NUM_AO,
    num_ao_prim=NUM_AO,
    orbital_indices=cart_orbital_indices,
    exponents=cart_exponents,
    coefficients=cart_coefficients,
    angular_momentums=tuple(cart_angular_momentums),
    polynominal_order_x=tuple(cart_nx),
    polynominal_order_y=tuple(cart_ny),
    polynominal_order_z=tuple(cart_nz),
)
aos_cart.sanity_check()

print(
    "\nConfigured cartesian AOs:",
    f"num_ao={NUM_AO}",
    f"num_centers={NUM_CENTERS}",
    f"l_max={L_MAX}",
    f"num_electrons={NUM_ELECTRONS}",
)

cart_grad_analytic_out = compute_AOs_grad(aos_cart, r_carts_jnp)
for leaf in jax.tree_util.tree_leaves(cart_grad_analytic_out):
    leaf.block_until_ready()

cart_grad_autodiff_out = _compute_AOs_grad_autodiff(aos_cart, r_carts_jnp)
for leaf in jax.tree_util.tree_leaves(cart_grad_autodiff_out):
    leaf.block_until_ready()

cart_lap_analytic_out = compute_AOs_laplacian(aos_cart, r_carts_jnp)
for leaf in jax.tree_util.tree_leaves(cart_lap_analytic_out):
    leaf.block_until_ready()

cart_lap_autodiff_out = _compute_AOs_laplacian_autodiff(aos_cart, r_carts_jnp)
for leaf in jax.tree_util.tree_leaves(cart_lap_autodiff_out):
    leaf.block_until_ready()

cart_grad_analytic_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = compute_AOs_grad(aos_cart, r_carts_jnp)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    cart_grad_analytic_times.append(time.perf_counter() - start)

cart_grad_autodiff_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = _compute_AOs_grad_autodiff(aos_cart, r_carts_jnp)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    cart_grad_autodiff_times.append(time.perf_counter() - start)

cart_lap_analytic_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = compute_AOs_laplacian(aos_cart, r_carts_jnp)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    cart_lap_analytic_times.append(time.perf_counter() - start)

cart_lap_autodiff_times = []
for _ in range(REPEATS):
    start = time.perf_counter()
    out = _compute_AOs_laplacian_autodiff(aos_cart, r_carts_jnp)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()
    cart_lap_autodiff_times.append(time.perf_counter() - start)

print("\nGradients (cartesian, seconds, mean over repeats):")
print(f"  analytic : {np.mean(cart_grad_analytic_times):.6f}")
print(f"  autodiff : {np.mean(cart_grad_autodiff_times):.6f}")

print("\nLaplacians (cartesian, seconds, mean over repeats):")
print(f"  analytic : {np.mean(cart_lap_analytic_times):.6f}")
print(f"  autodiff : {np.mean(cart_lap_autodiff_times):.6f}")

jax.tree_util.tree_leaves(
    (
        cart_grad_analytic_out,
        cart_grad_autodiff_out,
        cart_lap_analytic_out,
        cart_lap_autodiff_out,
    )
)
