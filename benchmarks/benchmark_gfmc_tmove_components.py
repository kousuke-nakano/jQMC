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
from jax import jit, vmap
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
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
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
N_WALKERS = 4  # number of walkers for multi-walker benchmark (in addition to single-walker)

TREXIO_FILE = os.path.join(
    os.path.dirname(__file__),
    "C6H6_ccecp_augccpvtz.h5",
)

# ── load system ───────────────────────────────────────────────────────────────
(
    structure_data,
    aos_data,
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

# ── Jastrow 1b+2b+3b variant ──────────────────────────────────────────────────
jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
    jastrow_1b_param=1.0,
    structure_data=structure_data,
    core_electrons=coulomb_potential_data.z_cores,
)
jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data, random_init=True, seed=SEED)
jastrow_data_full = Jastrow_data(
    jastrow_one_body_data=jastrow_onebody_data,
    jastrow_two_body_data=jastrow_twobody_data,
    jastrow_three_body_data=jastrow_threebody_data,
)
wavefunction_data_full = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data_full)

# ── Jastrow 1b+2b variant ──────────────────────────────────────────────────────
jastrow_data_1b2b = Jastrow_data(
    jastrow_one_body_data=jastrow_onebody_data,
    jastrow_two_body_data=jastrow_twobody_data,
    jastrow_three_body_data=None,
)
wavefunction_data_1b2b = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data_1b2b)

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
# warmup for 1b+2b+3b variant (Jastrow-affected ops)
block_until_ready(
    compute_kinetic_energy_all_elements_fast_update(
        wavefunction_data=wavefunction_data_full,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        geminal_inverse=A_old_inv,
    )
)
block_until_ready(
    compute_discretized_kinetic_energy_fast_update(
        alat=ALAT,
        wavefunction_data=wavefunction_data_full,
        A_old_inv=A_old_inv,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
    )
)
block_until_ready(
    compute_ecp_non_local_parts_nearest_neighbors_fast_update(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data_full,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        flag_determinant_only=False,
        A_old_inv=A_old_inv,
        RT=RT,
    )
)
# warmup for 1b+2b variant
block_until_ready(
    compute_kinetic_energy_all_elements_fast_update(
        wavefunction_data=wavefunction_data_1b2b,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        geminal_inverse=A_old_inv,
    )
)
block_until_ready(
    compute_discretized_kinetic_energy_fast_update(
        alat=ALAT,
        wavefunction_data=wavefunction_data_1b2b,
        A_old_inv=A_old_inv,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
    )
)
block_until_ready(
    compute_ecp_non_local_parts_nearest_neighbors_fast_update(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data_1b2b,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        flag_determinant_only=False,
        A_old_inv=A_old_inv,
        RT=RT,
    )
)

print(f"\nGFMC_n component benchmarks — C6H6 ECP  (single walker, mean over {REPEATS} repeats)")
print(f"\n  Jastrow: 2b only\n")
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

# ── Jastrow 1b+2b comparison (only Jastrow-dependent ops) ───────────────────────
print(f"\n  Jastrow: 1b+2b  (Jastrow-affected ops)\n")
print(f"  {'operation':<58s}  {'time':>8s}")
print(f"  {'-' * 58}  {'-' * 8}")
time_fn(
    "kinetic_continuum_fast_update  [+1b]",
    lambda: compute_kinetic_energy_all_elements_fast_update(
        wavefunction_data=wavefunction_data_1b2b,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        geminal_inverse=A_old_inv,
    ),
)
time_fn(
    "kinetic_discretized_fast_update  [+1b]",
    lambda: compute_discretized_kinetic_energy_fast_update(
        alat=ALAT,
        wavefunction_data=wavefunction_data_1b2b,
        A_old_inv=A_old_inv,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
    ),
)
time_fn(
    "ecp_non_local_tmove_fast_update  [+1b]",
    lambda: compute_ecp_non_local_parts_nearest_neighbors_fast_update(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data_1b2b,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        flag_determinant_only=False,
        A_old_inv=A_old_inv,
        RT=RT,
    ),
)

# ── Jastrow 1b+2b+3b comparison (only Jastrow-dependent ops) ─────────────────
print(f"\n  Jastrow: 1b+2b+3b  (Jastrow-affected ops; n_AO={aos_data._num_orb})\n")
print(f"  {'operation':<58s}  {'time':>8s}")
print(f"  {'-' * 58}  {'-' * 8}")
time_fn(
    "kinetic_continuum_fast_update  [+1b +3b]",
    lambda: compute_kinetic_energy_all_elements_fast_update(
        wavefunction_data=wavefunction_data_full,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        geminal_inverse=A_old_inv,
    ),
)
time_fn(
    "kinetic_discretized_fast_update  [+1b +3b]",
    lambda: compute_discretized_kinetic_energy_fast_update(
        alat=ALAT,
        wavefunction_data=wavefunction_data_full,
        A_old_inv=A_old_inv,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
    ),
)
time_fn(
    "ecp_non_local_tmove_fast_update  [+1b +3b]",
    lambda: compute_ecp_non_local_parts_nearest_neighbors_fast_update(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data_full,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        flag_determinant_only=False,
        A_old_inv=A_old_inv,
        RT=RT,
    ),
)

print()

# ════════════════════════════════════════════════════════════════════════════
# Multi-walker benchmark  (N_WALKERS walkers, vmapped)
# ════════════════════════════════════════════════════════════════════════════
rng_multi = np.random.default_rng(SEED + 99)
r_up_batch = jnp.asarray(rng_multi.uniform(R_CART_MIN, R_CART_MAX, size=(N_WALKERS, num_ele_up, 3)))
r_dn_batch = jnp.asarray(rng_multi.uniform(R_CART_MIN, R_CART_MAX, size=(N_WALKERS, num_ele_dn, 3)))

# A_old_inv per walker via vmap
_compute_geminal_and_inv_vmap = jit(vmap(_compute_geminal_and_inv))
A_old_inv_batch = _compute_geminal_and_inv_vmap(r_up_batch, r_dn_batch)

# vmapped callables
_kinetic_cont_vmap = jit(
    vmap(
        lambda wf, r_up, r_dn, inv: compute_kinetic_energy_all_elements_fast_update(
            wavefunction_data=wf, r_up_carts=r_up, r_dn_carts=r_dn, geminal_inverse=inv
        ),
        in_axes=(None, 0, 0, 0),
    )
)
_kinetic_disc_vmap = jit(
    vmap(
        lambda wf, inv, r_up, r_dn: compute_discretized_kinetic_energy_fast_update(
            alat=ALAT, wavefunction_data=wf, A_old_inv=inv, r_up_carts=r_up, r_dn_carts=r_dn, RT=RT
        ),
        in_axes=(None, 0, 0, 0),
    )
)
_el_el_vmap = jit(
    vmap(
        lambda r_up, r_dn: compute_bare_coulomb_potential_el_el(r_up_carts=r_up, r_dn_carts=r_dn),
        in_axes=(0, 0),
    )
)
_el_ion_vmap = jit(
    vmap(
        lambda r_up, r_dn: compute_bare_coulomb_potential_el_ion_element_wise(
            coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up, r_dn_carts=r_dn
        ),
        in_axes=(0, 0),
    )
)
_el_ion_disc_vmap = jit(
    vmap(
        lambda r_up, r_dn: compute_discretized_bare_coulomb_potential_el_ion_element_wise(
            coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up, r_dn_carts=r_dn, alat=ALAT
        ),
        in_axes=(0, 0),
    )
)
_ecp_local_vmap = jit(
    vmap(
        lambda r_up, r_dn: compute_ecp_local_parts_all_pairs(
            coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up, r_dn_carts=r_dn
        ),
        in_axes=(0, 0),
    )
)
_ecp_nonlocal_vmap = jit(
    vmap(
        lambda wf, r_up, r_dn, inv: compute_ecp_non_local_parts_nearest_neighbors_fast_update(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wf,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            flag_determinant_only=False,
            A_old_inv=inv,
            RT=RT,
        ),
        in_axes=(None, 0, 0, 0),
    )
)

# warmup for multi-walker
print(f"Warming up multi-walker (JIT compilation, W={N_WALKERS})...")
block_until_ready(_compute_geminal_and_inv_vmap(r_up_batch, r_dn_batch))
block_until_ready(_kinetic_cont_vmap(wavefunction_data, r_up_batch, r_dn_batch, A_old_inv_batch))
block_until_ready(_kinetic_disc_vmap(wavefunction_data, A_old_inv_batch, r_up_batch, r_dn_batch))
block_until_ready(_el_el_vmap(r_up_batch, r_dn_batch))
block_until_ready(_el_ion_vmap(r_up_batch, r_dn_batch))
block_until_ready(_el_ion_disc_vmap(r_up_batch, r_dn_batch))
block_until_ready(_ecp_local_vmap(r_up_batch, r_dn_batch))
block_until_ready(_ecp_nonlocal_vmap(wavefunction_data, r_up_batch, r_dn_batch, A_old_inv_batch))
block_until_ready(_kinetic_cont_vmap(wavefunction_data_full, r_up_batch, r_dn_batch, A_old_inv_batch))
block_until_ready(_kinetic_disc_vmap(wavefunction_data_full, A_old_inv_batch, r_up_batch, r_dn_batch))
block_until_ready(_ecp_nonlocal_vmap(wavefunction_data_full, r_up_batch, r_dn_batch, A_old_inv_batch))
block_until_ready(_kinetic_cont_vmap(wavefunction_data_1b2b, r_up_batch, r_dn_batch, A_old_inv_batch))
block_until_ready(_kinetic_disc_vmap(wavefunction_data_1b2b, A_old_inv_batch, r_up_batch, r_dn_batch))
block_until_ready(_ecp_nonlocal_vmap(wavefunction_data_1b2b, r_up_batch, r_dn_batch, A_old_inv_batch))

print(f"\nGFMC_n component benchmarks — C6H6 ECP  ({N_WALKERS} walkers vmapped, mean over {REPEATS} repeats)")
print(f"\n  Jastrow: 2b only\n")
print(f"  {'operation':<58s}  {'time':>8s}")
print(f"  {'-' * 58}  {'-' * 8}")

time_fn(
    "geminal + inverse  [A_old_inv setup per step]",
    lambda: _compute_geminal_and_inv_vmap(r_up_batch, r_dn_batch),
)
time_fn(
    "kinetic_continuum_fast_update",
    lambda: _kinetic_cont_vmap(wavefunction_data, r_up_batch, r_dn_batch, A_old_inv_batch),
)
time_fn(
    "kinetic_discretized_fast_update",
    lambda: _kinetic_disc_vmap(wavefunction_data, A_old_inv_batch, r_up_batch, r_dn_batch),
)
time_fn(
    "bare_coulomb_el_el",
    lambda: _el_el_vmap(r_up_batch, r_dn_batch),
)
time_fn(
    "bare_coulomb_ion_ion",
    lambda: compute_bare_coulomb_potential_ion_ion(coulomb_potential_data=coulomb_potential_data),
)
time_fn(
    "bare_coulomb_el_ion",
    lambda: _el_ion_vmap(r_up_batch, r_dn_batch),
)
time_fn(
    "bare_coulomb_el_ion_discretized",
    lambda: _el_ion_disc_vmap(r_up_batch, r_dn_batch),
)
time_fn(
    "ecp_local",
    lambda: _ecp_local_vmap(r_up_batch, r_dn_batch),
)
time_fn(
    "ecp_non_local_tmove_fast_update",
    lambda: _ecp_nonlocal_vmap(wavefunction_data, r_up_batch, r_dn_batch, A_old_inv_batch),
)

print(f"\n  Jastrow: 1b+2b  (Jastrow-affected ops)\n")
print(f"  {'operation':<58s}  {'time':>8s}")
print(f"  {'-' * 58}  {'-' * 8}")
time_fn(
    "kinetic_continuum_fast_update  [+1b]",
    lambda: _kinetic_cont_vmap(wavefunction_data_1b2b, r_up_batch, r_dn_batch, A_old_inv_batch),
)
time_fn(
    "kinetic_discretized_fast_update  [+1b]",
    lambda: _kinetic_disc_vmap(wavefunction_data_1b2b, A_old_inv_batch, r_up_batch, r_dn_batch),
)
time_fn(
    "ecp_non_local_tmove_fast_update  [+1b]",
    lambda: _ecp_nonlocal_vmap(wavefunction_data_1b2b, r_up_batch, r_dn_batch, A_old_inv_batch),
)

print(f"\n  Jastrow: 1b+2b+3b  (Jastrow-affected ops; n_AO={aos_data._num_orb})\n")
print(f"  {'operation':<58s}  {'time':>8s}")
print(f"  {'-' * 58}  {'-' * 8}")
time_fn(
    "kinetic_continuum_fast_update  [+1b +3b]",
    lambda: _kinetic_cont_vmap(wavefunction_data_full, r_up_batch, r_dn_batch, A_old_inv_batch),
)
time_fn(
    "kinetic_discretized_fast_update  [+1b +3b]",
    lambda: _kinetic_disc_vmap(wavefunction_data_full, A_old_inv_batch, r_up_batch, r_dn_batch),
)
time_fn(
    "ecp_non_local_tmove_fast_update  [+1b +3b]",
    lambda: _ecp_nonlocal_vmap(wavefunction_data_full, r_up_batch, r_dn_batch, A_old_inv_batch),
)

print()
