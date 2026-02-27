"""Benchmark MCMC components (ECP system) — single-walker timings.

Uses benchmarks/C6H6_ccecp_augccpvtz.h5 and reports per-call timings (ms).
All timings include ``.block_until_ready()`` so async JAX execution is fully measured.

Measured operations mirror those called inside the MCMC hot loop in jqmc_mcmc.py:
  - evaluate_ln_wavefunction
  - compute_local_energy
  - compute_AS_regularization_factor
  - grad(ln|Psi|) w.r.t. positions               [quantum force / drift]
  - grad(ln|Psi|) w.r.t. params                  [param grad]
  - grad(ln|Psi|) w.r.t. (r_up, r_dn)           [pos grad]
  - grad(E_L)     w.r.t. positions               [E_L pos grad]
  - grad(E_L)     w.r.t. (hamiltonian, r_up, r_dn) [E_L full grad]
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax.scipy import linalg as jsp_linalg

from jqmc.determinant import (
    compute_AS_regularization_factor,
    compute_geminal_all_elements,
)
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction

# ── configuration ─────────────────────────────────────────────────────────────
REPEATS = 5
SEED = 0
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
hamiltonian_data = Hamiltonian_data(
    structure_data=structure_data,
    coulomb_potential_data=coulomb_potential_data,
    wavefunction_data=wavefunction_data,
)

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
hamiltonian_data_full = Hamiltonian_data(
    structure_data=structure_data,
    coulomb_potential_data=coulomb_potential_data,
    wavefunction_data=wavefunction_data_full,
)

# ── Jastrow 1b+2b variant ──────────────────────────────────────────────────────
jastrow_data_1b2b = Jastrow_data(
    jastrow_one_body_data=jastrow_onebody_data,
    jastrow_two_body_data=jastrow_twobody_data,
    jastrow_three_body_data=None,
)
wavefunction_data_1b2b = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data_1b2b)
hamiltonian_data_1b2b = Hamiltonian_data(
    structure_data=structure_data,
    coulomb_potential_data=coulomb_potential_data,
    wavefunction_data=wavefunction_data_1b2b,
)

num_ele_up = geminal_mo_data.num_electron_up
num_ele_dn = geminal_mo_data.num_electron_dn

# ── masked hamiltonian variants for split param / position grad benchmarks ────
_wf_mask_kw = dict(opt_J1_param=False, opt_J2_param=False, opt_J3_param=False, opt_JNN_param=False, opt_lambda_param=False)
# param grads only: stop_gradient on R (atomic positions), wf params remain active
hamiltonian_for_param_grads = jax.lax.stop_gradient(hamiltonian_data).replace(wavefunction_data=wavefunction_data)
hamiltonian_for_param_grads_1b2b = jax.lax.stop_gradient(hamiltonian_data_1b2b).replace(
    wavefunction_data=wavefunction_data_1b2b
)
hamiltonian_for_param_grads_full = jax.lax.stop_gradient(hamiltonian_data_full).replace(
    wavefunction_data=wavefunction_data_full
)
# position grads only: wf params masked with stop_gradient, R remains active
hamiltonian_for_position_grads = hamiltonian_data.replace(
    wavefunction_data=wavefunction_data.with_param_grad_mask(**_wf_mask_kw)
)
hamiltonian_for_position_grads_1b2b = hamiltonian_data_1b2b.replace(
    wavefunction_data=wavefunction_data_1b2b.with_param_grad_mask(**_wf_mask_kw)
)
hamiltonian_for_position_grads_full = hamiltonian_data_full.replace(
    wavefunction_data=wavefunction_data_full.with_param_grad_mask(**_wf_mask_kw)
)

# ── random electron positions (single walker) ─────────────────────────────────
rng = np.random.default_rng(SEED)
r_up_carts = jnp.asarray(rng.uniform(R_CART_MIN, R_CART_MAX, size=(num_ele_up, 3)))
r_dn_carts = jnp.asarray(rng.uniform(R_CART_MIN, R_CART_MAX, size=(num_ele_dn, 3)))

RT = jnp.eye(3)

# ── compiled single-walker callables ──────────────────────────────────────────
# grad(ln|Psi|) w.r.t. electron positions — quantum force / drift vector
_grad_ln_psi_pos = jit(grad(evaluate_ln_wavefunction, argnums=(1, 2)))
# grad(ln|Psi|) w.r.t. wavefunction parameters
_grad_ln_psi_params = jit(grad(evaluate_ln_wavefunction, argnums=0))
# grad(E_L) w.r.t. electron positions r (argnums=(1,2))
_grad_e_L_r = jit(grad(compute_local_energy, argnums=(1, 2)))
# grad(E_L) w.r.t. hamiltonian_data (argnums=0) — includes nuclear positions R and wf params
_grad_e_L_R = jit(grad(compute_local_energy, argnums=0))


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

block_until_ready(evaluate_ln_wavefunction(wavefunction_data, r_up_carts, r_dn_carts))
block_until_ready(compute_local_energy(hamiltonian_data, r_up_carts, r_dn_carts, RT))
block_until_ready(compute_AS_regularization_factor(geminal_mo_data, r_up_carts, r_dn_carts))
block_until_ready(_grad_ln_psi_pos(wavefunction_data, r_up_carts, r_dn_carts))
block_until_ready(_grad_ln_psi_params(wavefunction_data, r_up_carts, r_dn_carts))
block_until_ready(_grad_e_L_r(hamiltonian_data, r_up_carts, r_dn_carts, RT))
block_until_ready(_grad_e_L_R(hamiltonian_for_param_grads, r_up_carts, r_dn_carts, RT))
block_until_ready(_grad_e_L_R(hamiltonian_for_position_grads, r_up_carts, r_dn_carts, RT))
# warmup for 1b+2b+3b variant
block_until_ready(evaluate_ln_wavefunction(wavefunction_data_full, r_up_carts, r_dn_carts))
block_until_ready(compute_local_energy(hamiltonian_data_full, r_up_carts, r_dn_carts, RT))
block_until_ready(_grad_ln_psi_pos(wavefunction_data_full, r_up_carts, r_dn_carts))
block_until_ready(_grad_ln_psi_params(wavefunction_data_full, r_up_carts, r_dn_carts))
block_until_ready(_grad_e_L_r(hamiltonian_data_full, r_up_carts, r_dn_carts, RT))
block_until_ready(_grad_e_L_R(hamiltonian_for_param_grads_full, r_up_carts, r_dn_carts, RT))
block_until_ready(_grad_e_L_R(hamiltonian_for_position_grads_full, r_up_carts, r_dn_carts, RT))
# warmup for 1b+2b variant
block_until_ready(evaluate_ln_wavefunction(wavefunction_data_1b2b, r_up_carts, r_dn_carts))
block_until_ready(compute_local_energy(hamiltonian_data_1b2b, r_up_carts, r_dn_carts, RT))
block_until_ready(_grad_ln_psi_pos(wavefunction_data_1b2b, r_up_carts, r_dn_carts))
block_until_ready(_grad_ln_psi_params(wavefunction_data_1b2b, r_up_carts, r_dn_carts))
block_until_ready(_grad_e_L_r(hamiltonian_data_1b2b, r_up_carts, r_dn_carts, RT))
block_until_ready(_grad_e_L_R(hamiltonian_for_param_grads_1b2b, r_up_carts, r_dn_carts, RT))
block_until_ready(_grad_e_L_R(hamiltonian_for_position_grads_1b2b, r_up_carts, r_dn_carts, RT))

print(f"\nMCMC component benchmarks — C6H6 ECP  (single walker, mean over {REPEATS} repeats)")
print(f"\n  Jastrow: 2b only\n")
print(f"  {'operation':<58s}  {'time':>8s}")
print(f"  {'-' * 58}  {'-' * 8}")

time_fn(
    "evaluate_ln_wavefunction",
    lambda: evaluate_ln_wavefunction(wavefunction_data, r_up_carts, r_dn_carts),
)
time_fn(
    "compute_local_energy",
    lambda: compute_local_energy(hamiltonian_data, r_up_carts, r_dn_carts, RT),
)
time_fn(
    "compute_AS_regularization_factor",
    lambda: compute_AS_regularization_factor(geminal_mo_data, r_up_carts, r_dn_carts),
)
time_fn(
    "grad(ln|Psi|) w.r.t. positions        [quantum force]",
    lambda: _grad_ln_psi_pos(wavefunction_data, r_up_carts, r_dn_carts),
)
time_fn(
    "grad(ln|Psi|) w.r.t. params           [param grad]",
    lambda: _grad_ln_psi_params(wavefunction_data, r_up_carts, r_dn_carts),
)
time_fn(
    "grad(ln|Psi|) w.r.t. (r_up,r_dn)        [pos grad]",
    lambda: _grad_ln_psi_pos(wavefunction_data, r_up_carts, r_dn_carts),
)
time_fn(
    "grad(E_L)     w.r.t. r             [de_L/dr]",
    lambda: _grad_e_L_r(hamiltonian_data, r_up_carts, r_dn_carts, RT),
)
time_fn(
    "grad(E_L)     w.r.t. hamiltonian   [de_L/dc, param grads]",
    lambda: _grad_e_L_R(hamiltonian_for_param_grads, r_up_carts, r_dn_carts, RT),
)
time_fn(
    "grad(E_L)     w.r.t. hamiltonian   [de_L/dR, position grads]",
    lambda: _grad_e_L_R(hamiltonian_for_position_grads, r_up_carts, r_dn_carts, RT),
)

# ── Jastrow 1b+2b comparison ──────────────────────────────────────────────────
print(f"\n  Jastrow: 1b+2b\n")
print(f"  {'operation':<58s}  {'time':>8s}")
print(f"  {'-' * 58}  {'-' * 8}")
time_fn(
    "evaluate_ln_wavefunction  [+1b]",
    lambda: evaluate_ln_wavefunction(wavefunction_data_1b2b, r_up_carts, r_dn_carts),
)
time_fn(
    "compute_local_energy  [+1b]",
    lambda: compute_local_energy(hamiltonian_data_1b2b, r_up_carts, r_dn_carts, RT),
)
time_fn(
    "grad(ln|Psi|) w.r.t. positions  [+1b]",
    lambda: _grad_ln_psi_pos(wavefunction_data_1b2b, r_up_carts, r_dn_carts),
)
time_fn(
    "grad(ln|Psi|) w.r.t. params  [+1b]",
    lambda: _grad_ln_psi_params(wavefunction_data_1b2b, r_up_carts, r_dn_carts),
)
time_fn(
    "grad(ln|Psi|) w.r.t. (r_up,r_dn)        [pos grad, +1b]",
    lambda: _grad_ln_psi_pos(wavefunction_data_1b2b, r_up_carts, r_dn_carts),
)
time_fn(
    "grad(E_L)     w.r.t. r             [de_L/dr, +1b]",
    lambda: _grad_e_L_r(hamiltonian_data_1b2b, r_up_carts, r_dn_carts, RT),
)
time_fn(
    "grad(E_L)     w.r.t. hamiltonian   [de_L/dc, param grads, +1b]",
    lambda: _grad_e_L_R(hamiltonian_for_param_grads_1b2b, r_up_carts, r_dn_carts, RT),
)
time_fn(
    "grad(E_L)     w.r.t. hamiltonian   [de_L/dR, position grads, +1b]",
    lambda: _grad_e_L_R(hamiltonian_for_position_grads_1b2b, r_up_carts, r_dn_carts, RT),
)

# ── Jastrow 1b+2b+3b comparison ────────────────────────────────────────────────
print(f"\n  Jastrow: 1b+2b+3b  (n_AO={aos_data._num_orb})\n")
print(f"  {'operation':<58s}  {'time':>8s}")
print(f"  {'-' * 58}  {'-' * 8}")
time_fn(
    "evaluate_ln_wavefunction  [+1b +3b]",
    lambda: evaluate_ln_wavefunction(wavefunction_data_full, r_up_carts, r_dn_carts),
)
time_fn(
    "compute_local_energy  [+1b +3b]",
    lambda: compute_local_energy(hamiltonian_data_full, r_up_carts, r_dn_carts, RT),
)
time_fn(
    "grad(ln|Psi|) w.r.t. positions  [+1b +3b]",
    lambda: _grad_ln_psi_pos(wavefunction_data_full, r_up_carts, r_dn_carts),
)
time_fn(
    "grad(ln|Psi|) w.r.t. params  [+1b +3b]",
    lambda: _grad_ln_psi_params(wavefunction_data_full, r_up_carts, r_dn_carts),
)
time_fn(
    "grad(ln|Psi|) w.r.t. (r_up,r_dn)        [pos grad, +1b +3b]",
    lambda: _grad_ln_psi_pos(wavefunction_data_full, r_up_carts, r_dn_carts),
)
time_fn(
    "grad(E_L)     w.r.t. r             [de_L/dr, +1b +3b]",
    lambda: _grad_e_L_r(hamiltonian_data_full, r_up_carts, r_dn_carts, RT),
)
time_fn(
    "grad(E_L)     w.r.t. hamiltonian   [de_L/dc, param grads, +1b +3b]",
    lambda: _grad_e_L_R(hamiltonian_for_param_grads_full, r_up_carts, r_dn_carts, RT),
)
time_fn(
    "grad(E_L)     w.r.t. hamiltonian   [de_L/dR, position grads, +1b +3b]",
    lambda: _grad_e_L_R(hamiltonian_for_position_grads_full, r_up_carts, r_dn_carts, RT),
)

print()

# ════════════════════════════════════════════════════════════════════════════
# Multi-walker benchmark  (N_WALKERS walkers, vmapped)
# ════════════════════════════════════════════════════════════════════════════
rng_multi = np.random.default_rng(SEED + 99)
r_up_batch = jnp.asarray(rng_multi.uniform(R_CART_MIN, R_CART_MAX, size=(N_WALKERS, num_ele_up, 3)))
r_dn_batch = jnp.asarray(rng_multi.uniform(R_CART_MIN, R_CART_MAX, size=(N_WALKERS, num_ele_dn, 3)))

# vmapped single-walker callables (wavefunction_data is static -> in_axes=None)
_eval_ln_psi_vmap = jit(vmap(evaluate_ln_wavefunction, in_axes=(None, 0, 0)))
_local_energy_vmap = jit(vmap(compute_local_energy, in_axes=(None, 0, 0, None)))
_as_reg_vmap = jit(vmap(compute_AS_regularization_factor, in_axes=(None, 0, 0)))
_grad_ln_psi_pos_vmap = jit(vmap(_grad_ln_psi_pos, in_axes=(None, 0, 0)))
_grad_ln_psi_params_vmap = jit(vmap(_grad_ln_psi_params, in_axes=(None, 0, 0)))
_grad_e_L_r_vmap = jit(vmap(_grad_e_L_r, in_axes=(None, 0, 0, None)))
_grad_e_L_R_vmap = jit(vmap(_grad_e_L_R, in_axes=(None, 0, 0, None)))

# warmup for multi-walker
print(f"Warming up multi-walker (JIT compilation, W={N_WALKERS})...")
block_until_ready(_eval_ln_psi_vmap(wavefunction_data, r_up_batch, r_dn_batch))
block_until_ready(_local_energy_vmap(hamiltonian_data, r_up_batch, r_dn_batch, RT))
block_until_ready(_as_reg_vmap(geminal_mo_data, r_up_batch, r_dn_batch))
block_until_ready(_grad_ln_psi_pos_vmap(wavefunction_data, r_up_batch, r_dn_batch))
block_until_ready(_grad_ln_psi_params_vmap(wavefunction_data, r_up_batch, r_dn_batch))
block_until_ready(_grad_e_L_r_vmap(hamiltonian_data, r_up_batch, r_dn_batch, RT))
block_until_ready(_grad_e_L_R_vmap(hamiltonian_for_param_grads, r_up_batch, r_dn_batch, RT))
block_until_ready(_grad_e_L_R_vmap(hamiltonian_for_position_grads, r_up_batch, r_dn_batch, RT))
block_until_ready(_eval_ln_psi_vmap(wavefunction_data_full, r_up_batch, r_dn_batch))
block_until_ready(_local_energy_vmap(hamiltonian_data_full, r_up_batch, r_dn_batch, RT))
block_until_ready(_grad_ln_psi_pos_vmap(wavefunction_data_full, r_up_batch, r_dn_batch))
block_until_ready(_grad_ln_psi_params_vmap(wavefunction_data_full, r_up_batch, r_dn_batch))
block_until_ready(_grad_e_L_r_vmap(hamiltonian_data_full, r_up_batch, r_dn_batch, RT))
block_until_ready(_grad_e_L_R_vmap(hamiltonian_for_param_grads_full, r_up_batch, r_dn_batch, RT))
block_until_ready(_grad_e_L_R_vmap(hamiltonian_for_position_grads_full, r_up_batch, r_dn_batch, RT))
block_until_ready(_eval_ln_psi_vmap(wavefunction_data_1b2b, r_up_batch, r_dn_batch))
block_until_ready(_local_energy_vmap(hamiltonian_data_1b2b, r_up_batch, r_dn_batch, RT))
block_until_ready(_grad_ln_psi_pos_vmap(wavefunction_data_1b2b, r_up_batch, r_dn_batch))
block_until_ready(_grad_ln_psi_params_vmap(wavefunction_data_1b2b, r_up_batch, r_dn_batch))
block_until_ready(_grad_e_L_r_vmap(hamiltonian_data_1b2b, r_up_batch, r_dn_batch, RT))
block_until_ready(_grad_e_L_R_vmap(hamiltonian_for_param_grads_1b2b, r_up_batch, r_dn_batch, RT))
block_until_ready(_grad_e_L_R_vmap(hamiltonian_for_position_grads_1b2b, r_up_batch, r_dn_batch, RT))

print(f"\nMCMC component benchmarks — C6H6 ECP  ({N_WALKERS} walkers vmapped, mean over {REPEATS} repeats)")
print(f"\n  Jastrow: 2b only\n")
print(f"  {'operation':<58s}  {'time':>8s}")
print(f"  {'-' * 58}  {'-' * 8}")

time_fn(
    "evaluate_ln_wavefunction",
    lambda: _eval_ln_psi_vmap(wavefunction_data, r_up_batch, r_dn_batch),
)
time_fn(
    "compute_local_energy",
    lambda: _local_energy_vmap(hamiltonian_data, r_up_batch, r_dn_batch, RT),
)
time_fn(
    "compute_AS_regularization_factor",
    lambda: _as_reg_vmap(geminal_mo_data, r_up_batch, r_dn_batch),
)
time_fn(
    "grad(ln|Psi|) w.r.t. positions        [quantum force]",
    lambda: _grad_ln_psi_pos_vmap(wavefunction_data, r_up_batch, r_dn_batch),
)
time_fn(
    "grad(ln|Psi|) w.r.t. params           [param grad]",
    lambda: _grad_ln_psi_params_vmap(wavefunction_data, r_up_batch, r_dn_batch),
)
time_fn(
    "grad(ln|Psi|) w.r.t. (r_up,r_dn)        [pos grad]",
    lambda: _grad_ln_psi_pos_vmap(wavefunction_data, r_up_batch, r_dn_batch),
)
time_fn(
    "grad(E_L)     w.r.t. r             [de_L/dr]",
    lambda: _grad_e_L_r_vmap(hamiltonian_data, r_up_batch, r_dn_batch, RT),
)
time_fn(
    "grad(E_L)     w.r.t. hamiltonian   [de_L/dc, param grads]",
    lambda: _grad_e_L_R_vmap(hamiltonian_for_param_grads, r_up_batch, r_dn_batch, RT),
)
time_fn(
    "grad(E_L)     w.r.t. hamiltonian   [de_L/dR, position grads]",
    lambda: _grad_e_L_R_vmap(hamiltonian_for_position_grads, r_up_batch, r_dn_batch, RT),
)

print(f"\n  Jastrow: 1b+2b\n")
print(f"  {'operation':<58s}  {'time':>8s}")
print(f"  {'-' * 58}  {'-' * 8}")
time_fn(
    "evaluate_ln_wavefunction  [+1b]",
    lambda: _eval_ln_psi_vmap(wavefunction_data_1b2b, r_up_batch, r_dn_batch),
)
time_fn(
    "compute_local_energy  [+1b]",
    lambda: _local_energy_vmap(hamiltonian_data_1b2b, r_up_batch, r_dn_batch, RT),
)
time_fn(
    "grad(ln|Psi|) w.r.t. positions  [+1b]",
    lambda: _grad_ln_psi_pos_vmap(wavefunction_data_1b2b, r_up_batch, r_dn_batch),
)
time_fn(
    "grad(ln|Psi|) w.r.t. params  [+1b]",
    lambda: _grad_ln_psi_params_vmap(wavefunction_data_1b2b, r_up_batch, r_dn_batch),
)
time_fn(
    "grad(ln|Psi|) w.r.t. (r_up,r_dn)        [pos grad, +1b]",
    lambda: _grad_ln_psi_pos_vmap(wavefunction_data_1b2b, r_up_batch, r_dn_batch),
)
time_fn(
    "grad(E_L)     w.r.t. r             [de_L/dr, +1b]",
    lambda: _grad_e_L_r_vmap(hamiltonian_data_1b2b, r_up_batch, r_dn_batch, RT),
)
time_fn(
    "grad(E_L)     w.r.t. hamiltonian   [de_L/dc, param grads, +1b]",
    lambda: _grad_e_L_R_vmap(hamiltonian_for_param_grads_1b2b, r_up_batch, r_dn_batch, RT),
)
time_fn(
    "grad(E_L)     w.r.t. hamiltonian   [de_L/dR, position grads, +1b]",
    lambda: _grad_e_L_R_vmap(hamiltonian_for_position_grads_1b2b, r_up_batch, r_dn_batch, RT),
)

print(f"\n  Jastrow: 1b+2b+3b  (n_AO={aos_data._num_orb})\n")
print(f"  {'operation':<58s}  {'time':>8s}")
print(f"  {'-' * 58}  {'-' * 8}")
time_fn(
    "evaluate_ln_wavefunction  [+1b +3b]",
    lambda: _eval_ln_psi_vmap(wavefunction_data_full, r_up_batch, r_dn_batch),
)
time_fn(
    "compute_local_energy  [+1b +3b]",
    lambda: _local_energy_vmap(hamiltonian_data_full, r_up_batch, r_dn_batch, RT),
)
time_fn(
    "grad(ln|Psi|) w.r.t. positions  [+1b +3b]",
    lambda: _grad_ln_psi_pos_vmap(wavefunction_data_full, r_up_batch, r_dn_batch),
)
time_fn(
    "grad(ln|Psi|) w.r.t. params  [+1b +3b]",
    lambda: _grad_ln_psi_params_vmap(wavefunction_data_full, r_up_batch, r_dn_batch),
)
time_fn(
    "grad(ln|Psi|) w.r.t. (r_up,r_dn)        [pos grad, +1b +3b]",
    lambda: _grad_ln_psi_pos_vmap(wavefunction_data_full, r_up_batch, r_dn_batch),
)
time_fn(
    "grad(E_L)     w.r.t. r             [de_L/dr, +1b +3b]",
    lambda: _grad_e_L_r_vmap(hamiltonian_data_full, r_up_batch, r_dn_batch, RT),
)
time_fn(
    "grad(E_L)     w.r.t. hamiltonian   [de_L/dc, param grads, +1b +3b]",
    lambda: _grad_e_L_R_vmap(hamiltonian_for_param_grads_full, r_up_batch, r_dn_batch, RT),
)
time_fn(
    "grad(E_L)     w.r.t. hamiltonian   [de_L/dR, position grads, +1b +3b]",
    lambda: _grad_e_L_R_vmap(hamiltonian_data_full, r_up_batch, r_dn_batch, RT),
)

print()
