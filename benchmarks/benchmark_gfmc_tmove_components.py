"""Benchmark GFMC_n t-move components (ECP systems).

Uses benchmarks/C6H6_ccecp_augccpvtz.h5 and reports per-call timings (ms).
All timings include ``.block_until_ready()`` so async JAX execution is fully measured.

Measured operations mirror those called inside the GFMC_n projection hot loop
(_body_fun_n) in jqmc_gfmc.py:
  - compute_geminal_all_elements + inverse   (precomputed once per projection step)
  - compute_kinetic_energy_all_elements_fast_update   (continuum, reuses A_old_inv)
  - compute_discretized_kinetic_energy_fast_update    (non-diagonal mesh, fast_update)
  - compute_bare_coulomb_potential_el_el
  - compute_bare_coulomb_potential_ion_ion
  - compute_bare_coulomb_potential_el_ion_element_wise
  - compute_discretized_bare_coulomb_potential_el_ion_element_wise
  - compute_ecp_local_parts_all_pairs                 (ECP only)
  - compute_ecp_non_local_parts_nearest_neighbors_fast_update  (ECP t-move, fast_update)
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy import linalg as jsp_linalg

from jqmc.coulomb_potential import (
    compute_bare_coulomb_potential_el_el,
    compute_bare_coulomb_potential_el_ion_element_wise,
    compute_bare_coulomb_potential_ion_ion,
    compute_discretized_bare_coulomb_potential_el_ion_element_wise,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors_fast_update,
)
from jqmc.determinant import compute_geminal_all_elements
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import (
    Wavefunction_data,
    compute_discretized_kinetic_energy_fast_update,
    compute_kinetic_energy_all_elements_fast_update,
)

# ── configuration ─────────────────────────────────────────────────────────────
REPEATS = 5
SEED = 0
ALAT = 0.30
R_CART_MIN, R_CART_MAX = -5.0, +5.0

TREXIO_FILE = os.path.join(
    os.path.dirname(__file__),
    "C6H6_ccecp_augccpvtz.h5",
)

# ── load system ───────────────────────────────────────────────────────────────
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

num_ele_up = geminal_mo_data.num_electron_up
num_ele_dn = geminal_mo_data.num_electron_dn

# ── random electron positions (single walker) ─────────────────────────────────
rng = np.random.default_rng(SEED)
r_up_carts = jnp.asarray(rng.uniform(R_CART_MIN, R_CART_MAX, size=(num_ele_up, 3)))
r_dn_carts = jnp.asarray(rng.uniform(R_CART_MIN, R_CART_MAX, size=(num_ele_dn, 3)))

RT = jnp.eye(3)

# ── pre-compute geminal & inverse (mirrors start of each projection step) ─────
G = compute_geminal_all_elements(
    geminal_data=geminal_mo_data,
    r_up_carts=r_up_carts,
    r_dn_carts=r_dn_carts,
)
lu, piv = jsp_linalg.lu_factor(G)
A_old_inv = jsp_linalg.lu_solve((lu, piv), jnp.eye(G.shape[0], dtype=G.dtype))


# ── compiled callable for geminal + inverse (timed together as a unit) ────────
@jit
def _compute_geminal_and_inv(r_up_carts, r_dn_carts):
    G = compute_geminal_all_elements(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    lu, piv = jsp_linalg.lu_factor(G)
    return jsp_linalg.lu_solve((lu, piv), jnp.eye(G.shape[0], dtype=G.dtype))


# ── helpers ───────────────────────────────────────────────────────────────────
def block_until_ready(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def time_fn(label, fn):
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = fn()
        block_until_ready(out)
        times.append(time.perf_counter() - start)
    print(f"  {label:<58s}  {np.mean(times) * 1e3:8.2f} msec")


# ── warmup (trigger JIT compilation before timing) ────────────────────────────
print("Warming up (JIT compilation)...")

block_until_ready(_compute_geminal_and_inv(r_up_carts, r_dn_carts))
block_until_ready(
    compute_kinetic_energy_all_elements_fast_update(
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        geminal_inverse=A_old_inv,
    )
)
block_until_ready(
    compute_discretized_kinetic_energy_fast_update(
        alat=ALAT,
        wavefunction_data=wavefunction_data,
        A_old_inv=A_old_inv,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
    )
)
block_until_ready(compute_bare_coulomb_potential_el_el(r_up_carts=r_up_carts, r_dn_carts=r_dn_carts))
block_until_ready(compute_bare_coulomb_potential_ion_ion(coulomb_potential_data=coulomb_potential_data))
block_until_ready(
    compute_bare_coulomb_potential_el_ion_element_wise(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
)
block_until_ready(
    compute_discretized_bare_coulomb_potential_el_ion_element_wise(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        alat=ALAT,
    )
)
block_until_ready(
    compute_ecp_local_parts_all_pairs(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
)
block_until_ready(
    compute_ecp_non_local_parts_nearest_neighbors_fast_update(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        flag_determinant_only=False,
        A_old_inv=A_old_inv,
        RT=RT,
    )
)

print(f"\nGFMC_n component benchmarks — C6H6 ECP  (single walker, mean over {REPEATS} repeats)\n")
print(f"  {'operation':<58s}  {'time':>8s}")
print(f"  {'-' * 58}  {'-' * 8}")

time_fn(
    "geminal + inverse  [A_old_inv setup per step]",
    lambda: _compute_geminal_and_inv(r_up_carts, r_dn_carts),
)
time_fn(
    "kinetic_continuum_fast_update",
    lambda: compute_kinetic_energy_all_elements_fast_update(
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        geminal_inverse=A_old_inv,
    ),
)
time_fn(
    "kinetic_discretized_fast_update",
    lambda: compute_discretized_kinetic_energy_fast_update(
        alat=ALAT,
        wavefunction_data=wavefunction_data,
        A_old_inv=A_old_inv,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
    ),
)
time_fn(
    "bare_coulomb_el_el",
    lambda: compute_bare_coulomb_potential_el_el(r_up_carts=r_up_carts, r_dn_carts=r_dn_carts),
)
time_fn(
    "bare_coulomb_ion_ion",
    lambda: compute_bare_coulomb_potential_ion_ion(coulomb_potential_data=coulomb_potential_data),
)
time_fn(
    "bare_coulomb_el_ion",
    lambda: compute_bare_coulomb_potential_el_ion_element_wise(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    ),
)
time_fn(
    "bare_coulomb_el_ion_discretized",
    lambda: compute_discretized_bare_coulomb_potential_el_ion_element_wise(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        alat=ALAT,
    ),
)
time_fn(
    "ecp_local",
    lambda: compute_ecp_local_parts_all_pairs(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    ),
)
time_fn(
    "ecp_non_local_tmove_fast_update",
    lambda: compute_ecp_non_local_parts_nearest_neighbors_fast_update(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        flag_determinant_only=False,
        A_old_inv=A_old_inv,
        RT=RT,
    ),
)

print()
