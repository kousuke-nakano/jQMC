"""MCMC component benchmarks for Nsight Systems / Nsight Compute profiling.

Profiles every computational kernel called within the MCMC hot loop
of ``jqmc_mcmc.py``.

Designed to be run under NVIDIA profiling tools:

    # Nsight Systems -- captures only the execution phase (after JIT compilation)
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
        --trace cuda,nvtx -o prof python benchmark_mcmc_components.py

    # Nsight Compute -- same capture-range filtering
    ncu --replay-mode application --nvtx --nvtx-include "exec/" \\
        -o mcmc_kernels python benchmark_mcmc_components.py

Profiling strategy:
  - cudaProfilerStart/Stop brackets the execution phase so that JIT
    compilation is excluded from the nsys/ncu capture.
  - NVTX ranges (via ctypes + libnvToolsExt.so, no pip install needed)
    label each kernel on the Nsight Systems timeline.
  - jax.named_scope embeds names into XLA HLO so Nsight Compute can
    map individual CUDA kernels back to source operations.

Kernels benchmarked (per Jastrow config: 2b, 1b+2b, 1b+2b+3b):
  1. evaluate_ln_wavefunction                      (ln|Psi|)
  2. compute_local_energy                           (E_L)
  3. compute_AS_regularization_factor               (AS reg.)
  4. grad(ln|Psi|) w.r.t. positions                 (quantum force / drift)
  5. grad(ln|Psi|) w.r.t. params                    (param grad)
  6. grad(E_L)     w.r.t. r                         (E_L position grad)
  7. grad(E_L)     w.r.t. hamiltonian  [param]      (dE_L/dc)
  8. grad(E_L)     w.r.t. hamiltonian  [position]   (dE_L/dR)
"""

import ctypes
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from jqmc.determinant import compute_AS_regularization_factor
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction

# -- CUDA profiler & NVTX support (via ctypes -- no pip install needed) --------
#
# 1. cudaProfilerStart / cudaProfilerStop
#    Bracket the execution phase so nsys/ncu capture only post-compilation work.
#    Usage:  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...
#
# 2. NVTX ranges (libnvToolsExt.so)
#    Label each kernel call on the Nsight Systems timeline.
#    Each bench() call appears as a named range whose duration includes
#    GPU execution (because we block_until_ready inside the range).
#
# 3. jax.named_scope (inside JIT)
#    Embeds names into XLA HLO so Nsight Compute can map CUDA kernels
#    back to source operations.


def _load_cudart():
    """Try to load libcudart via ctypes."""
    for name in ("libcudart.so", "libcudart.dylib"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


_libcudart = _load_cudart()
_nvtx_mod = None  # will be set below

if _libcudart is None:
    print("[INFO] libcudart not found -- cudaProfilerStart/Stop disabled (CPU-only run).")

# Try ``pip install nvtx`` package first (bundles its own .so),
# then fall back to ctypes libnvToolsExt.so from CUDA toolkit.
try:
    import nvtx as _nvtx_mod  # noqa: F811

    print("[INFO] NVTX ranges enabled via ``nvtx`` pip package.")
except ImportError:
    _nvtx_mod = None
    _libnvtx_ctypes = None
    for name in ("libnvToolsExt.so", "libnvToolsExt.dylib"):
        try:
            _libnvtx_ctypes = ctypes.CDLL(name)
            print("[INFO] NVTX ranges enabled via libnvToolsExt (ctypes).")
            break
        except OSError:
            continue
    if _libnvtx_ctypes is None:
        print("[INFO] NVTX not available -- pip install nvtx to enable host-side NVTX ranges.")


def cuda_profiler_start():
    """Signal nsys/ncu to begin capturing."""
    if _libcudart is not None:
        _libcudart.cudaProfilerStart()


def cuda_profiler_stop():
    """Signal nsys/ncu to stop capturing."""
    if _libcudart is not None:
        _libcudart.cudaProfilerStop()


def nvtx_push(name: str):
    """Push an NVTX range (appears on nsys timeline)."""
    if _nvtx_mod is not None:
        _nvtx_mod.push_range(name)
    elif "_libnvtx_ctypes" in globals() and _libnvtx_ctypes is not None:
        _libnvtx_ctypes.nvtxRangePushA(name.encode("utf-8"))


def nvtx_pop():
    """Pop the current NVTX range."""
    if _nvtx_mod is not None:
        _nvtx_mod.pop_range()
    elif "_libnvtx_ctypes" in globals() and _libnvtx_ctypes is not None:
        _libnvtx_ctypes.nvtxRangePop()


# ==============================================================================
# Configuration  (edit these directly)
# ==============================================================================
TREXIO_FILE = os.path.join(os.path.dirname(__file__), "C6H6_ccecp_augccpvtz.h5")
N_WALKERS = 4096  # number of walkers (vmapped)
REPEATS = 1  # repeats per kernel (after warmup)
SEED = 0
R_CART_MIN, R_CART_MAX = -5.0, +5.0

# ==============================================================================
# Helpers
# ==============================================================================


def block_until_ready(tree):
    """Synchronise all JAX arrays in a pytree."""
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def bench(label, fn, repeats=REPEATS):
    """Warmup once, then time *repeats* calls.  Reports mean +/- std (ms).

    Each call is wrapped with an NVTX range so nsys shows named regions.
    """
    # warmup (already JIT-compiled, but prime the cache)
    out = fn()
    block_until_ready(out)

    times = []
    for _ in range(repeats):
        nvtx_push(label)
        start = time.perf_counter()
        out = fn()
        block_until_ready(out)
        elapsed = time.perf_counter() - start
        nvtx_pop()
        times.append(elapsed)
    mean_ms = np.mean(times) * 1e3
    std_ms = np.std(times) * 1e3
    print(f"  {label:<62s}  {mean_ms:9.3f} +/- {std_ms:7.3f} ms")
    return times


# ==============================================================================
# Load system
# ==============================================================================
print(f"Loading system from {TREXIO_FILE} ...")
(
    structure_data,
    aos_data,
    _,
    _,
    geminal_mo_data,
    coulomb_potential_data,
) = read_trexio_file(trexio_file=TREXIO_FILE, store_tuple=True)

num_ele_up = geminal_mo_data.num_electron_up
num_ele_dn = geminal_mo_data.num_electron_dn

# -- Jastrow: 2b only ---------------------------------------------------------
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

# -- Jastrow: 1b+2b -----------------------------------------------------------
jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
    jastrow_1b_param=1.0,
    structure_data=structure_data,
    core_electrons=coulomb_potential_data.z_cores,
)
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

# -- Jastrow: 1b+2b+3b --------------------------------------------------------
jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
    orb_data=aos_data,
    random_init=True,
    seed=SEED,
)
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

# -- Masked hamiltonian variants for split param / position grad benchmarks ---
_wf_mask_kw = dict(
    opt_J1_param=False,
    opt_J2_param=False,
    opt_J3_param=False,
    opt_JNN_param=False,
    opt_lambda_param=False,
)
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

print(f"  electrons: {num_ele_up} up, {num_ele_dn} dn")
print(f"  walkers: {N_WALKERS} | repeats per kernel: {REPEATS}\n")

# ==============================================================================
# Initialize walkers  (batched arrays of shape (N_WALKERS, ...))
# ==============================================================================
rng = np.random.default_rng(SEED)
r_up_batch = jnp.asarray(rng.uniform(R_CART_MIN, R_CART_MAX, size=(N_WALKERS, num_ele_up, 3)))
r_dn_batch = jnp.asarray(rng.uniform(R_CART_MIN, R_CART_MAX, size=(N_WALKERS, num_ele_dn, 3)))

RT = jnp.eye(3)

# ==============================================================================
# Build vmapped + jitted kernels
#
# Each kernel is wrapped with jax.named_scope so that Nsight Compute can
# identify which CUDA kernels belong to which MCMC sub-operation.
# ==============================================================================

# 1. evaluate_ln_wavefunction


def _eval_ln_psi(wf, r_up, r_dn):
    with jax.named_scope("evaluate_ln_wavefunction"):
        return evaluate_ln_wavefunction(wf, r_up, r_dn)


kernel_eval_ln_psi = jit(vmap(_eval_ln_psi, in_axes=(None, 0, 0)))

# 2. compute_local_energy


def _local_energy(h, r_up, r_dn, RT):
    with jax.named_scope("compute_local_energy"):
        return compute_local_energy(h, r_up, r_dn, RT)


kernel_local_energy = jit(vmap(_local_energy, in_axes=(None, 0, 0, None)))

# 3. compute_AS_regularization_factor


def _as_reg(g, r_up, r_dn):
    with jax.named_scope("AS_regularization"):
        return compute_AS_regularization_factor(g, r_up, r_dn)


kernel_as_reg = jit(vmap(_as_reg, in_axes=(None, 0, 0)))

# 4. grad(ln|Psi|) w.r.t. positions  [quantum force / drift]
_grad_ln_psi_pos_raw = grad(evaluate_ln_wavefunction, argnums=(1, 2))


def _grad_ln_psi_pos(wf, r_up, r_dn):
    with jax.named_scope("grad_ln_psi_positions"):
        return _grad_ln_psi_pos_raw(wf, r_up, r_dn)


kernel_grad_ln_psi_pos = jit(vmap(_grad_ln_psi_pos, in_axes=(None, 0, 0)))

# 5. grad(ln|Psi|) w.r.t. params  [param grad]
_grad_ln_psi_params_raw = grad(evaluate_ln_wavefunction, argnums=0)


def _grad_ln_psi_params(wf, r_up, r_dn):
    with jax.named_scope("grad_ln_psi_params"):
        return _grad_ln_psi_params_raw(wf, r_up, r_dn)


kernel_grad_ln_psi_params = jit(vmap(_grad_ln_psi_params, in_axes=(None, 0, 0)))

# 6. grad(E_L) w.r.t. r  [E_L position grad]
_grad_e_L_r_raw = grad(compute_local_energy, argnums=(1, 2))


def _grad_e_L_r(h, r_up, r_dn, RT):
    with jax.named_scope("grad_EL_positions"):
        return _grad_e_L_r_raw(h, r_up, r_dn, RT)


kernel_grad_e_L_r = jit(vmap(_grad_e_L_r, in_axes=(None, 0, 0, None)))

# 7 & 8. grad(E_L) w.r.t. hamiltonian  [param grads / position grads]
_grad_e_L_H_raw = grad(compute_local_energy, argnums=0)


def _grad_e_L_H(h, r_up, r_dn, RT):
    with jax.named_scope("grad_EL_hamiltonian"):
        return _grad_e_L_H_raw(h, r_up, r_dn, RT)


kernel_grad_e_L_H = jit(vmap(_grad_e_L_H, in_axes=(None, 0, 0, None)))

# ==============================================================================
# Warmup  (JIT compile all kernels)
# ==============================================================================
print("=" * 80)
print("Warming up (JIT compilation) ...")
t0 = time.perf_counter()

# -- 2b ------------------------------------------------------------------------
block_until_ready(kernel_eval_ln_psi(wavefunction_data, r_up_batch, r_dn_batch))
block_until_ready(kernel_local_energy(hamiltonian_data, r_up_batch, r_dn_batch, RT))
block_until_ready(kernel_as_reg(geminal_mo_data, r_up_batch, r_dn_batch))
block_until_ready(kernel_grad_ln_psi_pos(wavefunction_data, r_up_batch, r_dn_batch))
block_until_ready(kernel_grad_ln_psi_params(wavefunction_data, r_up_batch, r_dn_batch))
block_until_ready(kernel_grad_e_L_r(hamiltonian_data, r_up_batch, r_dn_batch, RT))
block_until_ready(kernel_grad_e_L_H(hamiltonian_for_param_grads, r_up_batch, r_dn_batch, RT))
block_until_ready(kernel_grad_e_L_H(hamiltonian_for_position_grads, r_up_batch, r_dn_batch, RT))

# -- 1b+2b ---------------------------------------------------------------------
block_until_ready(kernel_eval_ln_psi(wavefunction_data_1b2b, r_up_batch, r_dn_batch))
block_until_ready(kernel_local_energy(hamiltonian_data_1b2b, r_up_batch, r_dn_batch, RT))
block_until_ready(kernel_grad_ln_psi_pos(wavefunction_data_1b2b, r_up_batch, r_dn_batch))
block_until_ready(kernel_grad_ln_psi_params(wavefunction_data_1b2b, r_up_batch, r_dn_batch))
block_until_ready(kernel_grad_e_L_r(hamiltonian_data_1b2b, r_up_batch, r_dn_batch, RT))
block_until_ready(kernel_grad_e_L_H(hamiltonian_for_param_grads_1b2b, r_up_batch, r_dn_batch, RT))
block_until_ready(kernel_grad_e_L_H(hamiltonian_for_position_grads_1b2b, r_up_batch, r_dn_batch, RT))

# -- 1b+2b+3b ------------------------------------------------------------------
block_until_ready(kernel_eval_ln_psi(wavefunction_data_full, r_up_batch, r_dn_batch))
block_until_ready(kernel_local_energy(hamiltonian_data_full, r_up_batch, r_dn_batch, RT))
block_until_ready(kernel_grad_ln_psi_pos(wavefunction_data_full, r_up_batch, r_dn_batch))
block_until_ready(kernel_grad_ln_psi_params(wavefunction_data_full, r_up_batch, r_dn_batch))
block_until_ready(kernel_grad_e_L_r(hamiltonian_data_full, r_up_batch, r_dn_batch, RT))
block_until_ready(kernel_grad_e_L_H(hamiltonian_for_param_grads_full, r_up_batch, r_dn_batch, RT))
block_until_ready(kernel_grad_e_L_H(hamiltonian_for_position_grads_full, r_up_batch, r_dn_batch, RT))

print(f"  compilation done in {time.perf_counter() - t0:.1f} s\n")

# ==============================================================================
# Start CUDA profiler -- everything below this line is captured by nsys/ncu
# when using --capture-range=cudaProfilerApi
# ==============================================================================
cuda_profiler_start()
nvtx_push("exec/all_benchmarks")

# ==============================================================================
# Jastrow: 2b only
# ==============================================================================
print("=" * 80)
print(f"MCMC component benchmarks  ({N_WALKERS} walkers vmapped, mean +/- std over {REPEATS} repeats)")
print(f"\n  Jastrow: 2b only\n")
print(f"  {'kernel':<62s}  {'time':>18s}")
print(f"  {'-' * 62}  {'-' * 18}")

nvtx_push("exec/jastrow_2b")

bench(
    "1. evaluate_ln_wavefunction",
    lambda: kernel_eval_ln_psi(wavefunction_data, r_up_batch, r_dn_batch),
)
bench(
    "2. compute_local_energy",
    lambda: kernel_local_energy(hamiltonian_data, r_up_batch, r_dn_batch, RT),
)
bench(
    "3. compute_AS_regularization_factor",
    lambda: kernel_as_reg(geminal_mo_data, r_up_batch, r_dn_batch),
)
bench(
    "4. grad(ln|Psi|) w.r.t. positions  [quantum force]",
    lambda: kernel_grad_ln_psi_pos(wavefunction_data, r_up_batch, r_dn_batch),
)
bench(
    "5. grad(ln|Psi|) w.r.t. params     [param grad]",
    lambda: kernel_grad_ln_psi_params(wavefunction_data, r_up_batch, r_dn_batch),
)
bench(
    "6. grad(E_L) w.r.t. r              [dE_L/dr]",
    lambda: kernel_grad_e_L_r(hamiltonian_data, r_up_batch, r_dn_batch, RT),
)
bench(
    "7. grad(E_L) w.r.t. hamiltonian    [dE_L/dc, param]",
    lambda: kernel_grad_e_L_H(hamiltonian_for_param_grads, r_up_batch, r_dn_batch, RT),
)
bench(
    "8. grad(E_L) w.r.t. hamiltonian    [dE_L/dR, position]",
    lambda: kernel_grad_e_L_H(hamiltonian_for_position_grads, r_up_batch, r_dn_batch, RT),
)

nvtx_pop()  # close "exec/jastrow_2b"

# ==============================================================================
# Jastrow: 1b+2b
# ==============================================================================
print(f"\n  Jastrow: 1b+2b\n")
print(f"  {'kernel':<62s}  {'time':>18s}")
print(f"  {'-' * 62}  {'-' * 18}")

nvtx_push("exec/jastrow_1b2b")

bench(
    "1. evaluate_ln_wavefunction  [+1b]",
    lambda: kernel_eval_ln_psi(wavefunction_data_1b2b, r_up_batch, r_dn_batch),
)
bench(
    "2. compute_local_energy  [+1b]",
    lambda: kernel_local_energy(hamiltonian_data_1b2b, r_up_batch, r_dn_batch, RT),
)
bench(
    "4. grad(ln|Psi|) w.r.t. positions  [+1b]",
    lambda: kernel_grad_ln_psi_pos(wavefunction_data_1b2b, r_up_batch, r_dn_batch),
)
bench(
    "5. grad(ln|Psi|) w.r.t. params     [+1b]",
    lambda: kernel_grad_ln_psi_params(wavefunction_data_1b2b, r_up_batch, r_dn_batch),
)
bench(
    "6. grad(E_L) w.r.t. r              [dE_L/dr, +1b]",
    lambda: kernel_grad_e_L_r(hamiltonian_data_1b2b, r_up_batch, r_dn_batch, RT),
)
bench(
    "7. grad(E_L) w.r.t. hamiltonian    [dE_L/dc, param, +1b]",
    lambda: kernel_grad_e_L_H(hamiltonian_for_param_grads_1b2b, r_up_batch, r_dn_batch, RT),
)
bench(
    "8. grad(E_L) w.r.t. hamiltonian    [dE_L/dR, position, +1b]",
    lambda: kernel_grad_e_L_H(hamiltonian_for_position_grads_1b2b, r_up_batch, r_dn_batch, RT),
)

nvtx_pop()  # close "exec/jastrow_1b2b"

# ==============================================================================
# Jastrow: 1b+2b+3b
# ==============================================================================
print(f"\n  Jastrow: 1b+2b+3b  (n_AO={aos_data._num_orb})\n")
print(f"  {'kernel':<62s}  {'time':>18s}")
print(f"  {'-' * 62}  {'-' * 18}")

nvtx_push("exec/jastrow_1b2b3b")

bench(
    "1. evaluate_ln_wavefunction  [+1b +3b]",
    lambda: kernel_eval_ln_psi(wavefunction_data_full, r_up_batch, r_dn_batch),
)
bench(
    "2. compute_local_energy  [+1b +3b]",
    lambda: kernel_local_energy(hamiltonian_data_full, r_up_batch, r_dn_batch, RT),
)
bench(
    "4. grad(ln|Psi|) w.r.t. positions  [+1b +3b]",
    lambda: kernel_grad_ln_psi_pos(wavefunction_data_full, r_up_batch, r_dn_batch),
)
bench(
    "5. grad(ln|Psi|) w.r.t. params     [+1b +3b]",
    lambda: kernel_grad_ln_psi_params(wavefunction_data_full, r_up_batch, r_dn_batch),
)
bench(
    "6. grad(E_L) w.r.t. r              [dE_L/dr, +1b +3b]",
    lambda: kernel_grad_e_L_r(hamiltonian_data_full, r_up_batch, r_dn_batch, RT),
)
bench(
    "7. grad(E_L) w.r.t. hamiltonian    [dE_L/dc, param, +1b +3b]",
    lambda: kernel_grad_e_L_H(hamiltonian_for_param_grads_full, r_up_batch, r_dn_batch, RT),
)
bench(
    "8. grad(E_L) w.r.t. hamiltonian    [dE_L/dR, position, +1b +3b]",
    lambda: kernel_grad_e_L_H(hamiltonian_for_position_grads_full, r_up_batch, r_dn_batch, RT),
)

nvtx_pop()  # close "exec/jastrow_1b2b3b"

# ==============================================================================
# Stop CUDA profiler
# ==============================================================================
nvtx_pop()  # close "exec/all_benchmarks"
cuda_profiler_stop()

# ==============================================================================
# Nsight usage hints
# ==============================================================================
print()
print("=" * 80)
print("Nsight profiling commands (compilation excluded via cudaProfilerApi):")
print()
print("  # Nsight Systems -- timeline with NVTX labels per kernel")
print("  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\")
print("      --trace cuda,nvtx -o prof python benchmark_mcmc_components.py")
print()
print("  # Post-process -> CSV for LLM analysis")
print("  nsys stats prof.nsys-rep --report cuda_gpu_kern_sum --format csv > summary.csv")
print("  nsys stats prof.nsys-rep --report nvtx_gpu_proj_trace --format csv > nvtx_trace.csv")
print()
print("  # Nsight Compute")
print("  ncu --replay-mode application -o mcmc_kernels python benchmark_mcmc_components.py")
print("  ncu -i mcmc_kernels.ncu-rep --csv > ncu_results.csv")
print()
