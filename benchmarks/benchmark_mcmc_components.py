"""Benchmark MCMC components (ECP system) — single-walker timings.

Uses benchmarks/C6H6_ccecp_augccpvtz.h5 and reports per-call timings (ms).
All timings include ``.block_until_ready()`` so async JAX execution is fully measured.

Measured operations mirror those called inside the MCMC hot loop in jqmc_mcmc.py:
  - evaluate_ln_wavefunction
  - compute_local_energy
  - compute_AS_regularization_factor
  - grad(ln|Psi|) w.r.t. positions               [quantum force / drift]
  - grad(ln|Psi|) w.r.t. params                  [param grad]
  - grad(ln|Psi|) w.r.t. (params, r_up, r_dn)   [full grad]
  - grad(E_L)     w.r.t. positions               [E_L pos grad]
  - grad(E_L)     w.r.t. (hamiltonian, r_up, r_dn) [E_L full grad]
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
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
# grad(ln|Psi|) w.r.t. (params, r_up, r_dn)
_grad_ln_psi_full = jit(grad(evaluate_ln_wavefunction, argnums=(0, 1, 2)))
# grad(E_L) w.r.t. electron positions
_grad_e_L_pos = jit(grad(compute_local_energy, argnums=(1, 2)))
# grad(E_L) w.r.t. (hamiltonian_data, r_up, r_dn)
_grad_e_L_full = jit(grad(compute_local_energy, argnums=(0, 1, 2)))


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
block_until_ready(_grad_ln_psi_full(wavefunction_data, r_up_carts, r_dn_carts))
block_until_ready(_grad_e_L_pos(hamiltonian_data, r_up_carts, r_dn_carts, RT))
block_until_ready(_grad_e_L_full(hamiltonian_data, r_up_carts, r_dn_carts, RT))
# warmup for 1b+2b+3b variant
block_until_ready(evaluate_ln_wavefunction(wavefunction_data_full, r_up_carts, r_dn_carts))
block_until_ready(compute_local_energy(hamiltonian_data_full, r_up_carts, r_dn_carts, RT))
block_until_ready(_grad_ln_psi_pos(wavefunction_data_full, r_up_carts, r_dn_carts))
block_until_ready(_grad_ln_psi_params(wavefunction_data_full, r_up_carts, r_dn_carts))
block_until_ready(_grad_ln_psi_full(wavefunction_data_full, r_up_carts, r_dn_carts))
block_until_ready(_grad_e_L_pos(hamiltonian_data_full, r_up_carts, r_dn_carts, RT))
block_until_ready(_grad_e_L_full(hamiltonian_data_full, r_up_carts, r_dn_carts, RT))
# warmup for 1b+2b variant
block_until_ready(evaluate_ln_wavefunction(wavefunction_data_1b2b, r_up_carts, r_dn_carts))
block_until_ready(compute_local_energy(hamiltonian_data_1b2b, r_up_carts, r_dn_carts, RT))
block_until_ready(_grad_ln_psi_pos(wavefunction_data_1b2b, r_up_carts, r_dn_carts))
block_until_ready(_grad_ln_psi_params(wavefunction_data_1b2b, r_up_carts, r_dn_carts))
block_until_ready(_grad_ln_psi_full(wavefunction_data_1b2b, r_up_carts, r_dn_carts))
block_until_ready(_grad_e_L_pos(hamiltonian_data_1b2b, r_up_carts, r_dn_carts, RT))
block_until_ready(_grad_e_L_full(hamiltonian_data_1b2b, r_up_carts, r_dn_carts, RT))

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
    "grad(ln|Psi|) w.r.t. (params,r_up,r_dn) [full grad]",
    lambda: _grad_ln_psi_full(wavefunction_data, r_up_carts, r_dn_carts),
)
time_fn(
    "grad(E_L)     w.r.t. positions        [E_L pos grad]",
    lambda: _grad_e_L_pos(hamiltonian_data, r_up_carts, r_dn_carts, RT),
)
time_fn(
    "grad(E_L)     w.r.t. (h,r_up,r_dn)   [E_L full grad]",
    lambda: _grad_e_L_full(hamiltonian_data, r_up_carts, r_dn_carts, RT),
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
    "grad(ln|Psi|) w.r.t. (params,r_up,r_dn)  [+1b]",
    lambda: _grad_ln_psi_full(wavefunction_data_1b2b, r_up_carts, r_dn_carts),
)
time_fn(
    "grad(E_L)     w.r.t. positions  [+1b]",
    lambda: _grad_e_L_pos(hamiltonian_data_1b2b, r_up_carts, r_dn_carts, RT),
)
time_fn(
    "grad(E_L)     w.r.t. (h,r_up,r_dn)  [+1b]",
    lambda: _grad_e_L_full(hamiltonian_data_1b2b, r_up_carts, r_dn_carts, RT),
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
    "grad(ln|Psi|) w.r.t. (params,r_up,r_dn)  [+1b +3b]",
    lambda: _grad_ln_psi_full(wavefunction_data_full, r_up_carts, r_dn_carts),
)
time_fn(
    "grad(E_L)     w.r.t. positions  [+1b +3b]",
    lambda: _grad_e_L_pos(hamiltonian_data_full, r_up_carts, r_dn_carts, RT),
)
time_fn(
    "grad(E_L)     w.r.t. (h,r_up,r_dn)  [+1b +3b]",
    lambda: _grad_e_L_full(hamiltonian_data_full, r_up_carts, r_dn_carts, RT),
)

print()
