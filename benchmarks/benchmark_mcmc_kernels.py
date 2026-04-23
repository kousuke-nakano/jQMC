"""MCMC component benchmarks for Nsight Systems / Nsight Compute profiling.

Profiles forward-pass computational kernels of the MCMC hot loop
(``jqmc_mcmc.py``) using a **synthetic** system -- no TREXIO file needed.

System size and wavefunction ansatz are controlled by the configuration
block below.

Designed to be run under NVIDIA profiling tools:

    # Nsight Systems -- captures only the execution phase (after JIT compilation)
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
        --trace cuda,nvtx -o prof python benchmark_mcmc_kernels.py

    # Nsight Compute -- same capture-range filtering
    ncu --replay-mode application --nvtx --nvtx-include "exec/" \\
        -o mcmc_kernels python benchmark_mcmc_kernels.py

Profiling strategy:
  - cudaProfilerStart/Stop brackets the execution phase so that JIT
    compilation is excluded from the nsys/ncu capture.
  - NVTX ranges (via ctypes + libnvToolsExt.so, no pip install needed)
    label each kernel on the Nsight Systems timeline.
  - jax.named_scope embeds names into XLA HLO so Nsight Compute can
    map individual CUDA kernels back to source operations.

All kernels use the **fast_update** variants (pre-computed geminal inverse),
matching the actual MCMC hot loop in ``jqmc_mcmc.py``.

Kernels benchmarked:
  1. evaluate_ln_wavefunction_fast     (ln|Psi|, pre-computed geminal_inv)
  2. compute_local_energy_fast         (E_L, pre-computed geminal_inverse)
  3. compute_AS_regularization_factor_fast_update  (AS reg., no SVD)
"""

import ctypes
import time

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from jax import jit, vmap

from jqmc.atomic_orbital import AOs_cart_data
from jqmc.coulomb_potential import Coulomb_potential_data
from jqmc.determinant import (
    Geminal_data,
    compute_AS_regularization_factor_fast_update,
    compute_geminal_all_elements,
)
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy_fast
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.structure import Structure_data
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction_fast

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
# Configuration  (edit these to control the synthetic benchmark system)
# ==============================================================================
# --- System size ---
N_ATOMS = 12  # number of atoms
VALENCE_ELECTRONS_PER_ATOM = 4  # valence electrons per atom (neutral charge)
N_AO_PER_ATOM_DET = 10  # AOs per atom for determinant
N_AO_PER_ATOM_J3 = 5  # AOs per atom for J3 (only if J3_FLAG=True)
ATOM_SPACING = 3.0  # inter-atom distance (Bohr)

# --- Jastrow ---
J1_FLAG = True  # one-body Jastrow
J2_FLAG = True  # two-body Jastrow
J3_FLAG = True  # three-body Jastrow (analytic)
JNN_FLAG = False  # neural-network Jastrow (not yet supported)

# --- ECP ---
ECP_FLAG = True  # effective core potential
ECP_CORE_ELECTRONS = 2  # core electrons per atom (only if ECP_FLAG)

# --- Benchmark ---
N_WALKERS = 4096  # number of walkers (vmapped)
REPEATS = 1  # repeats per kernel (after warmup)
SEED = 42

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
# Build synthetic system
# ==============================================================================


def _cartesian_ao_specs(n_ao_per_atom):
    """Return a list of ``(l, nx, ny, nz, exponent)`` for Cartesian AOs.

    Fills complete angular-momentum shells (s=1, p=3, d=6, f=10, ...)
    until *n_ao_per_atom* AOs are reached; the last shell may be partial.
    """
    specs = []
    l = 0
    # Moderate exponents per shell -- broad enough to overlap
    _exp_by_l = [5.0, 3.0, 1.5, 0.8, 0.4, 0.2]
    while len(specs) < n_ao_per_atom:
        exp = _exp_by_l[min(l, len(_exp_by_l) - 1)]
        for nx in range(l, -1, -1):
            for ny in range(l - nx, -1, -1):
                nz = l - nx - ny
                if len(specs) >= n_ao_per_atom:
                    return specs
                specs.append((l, nx, ny, nz, exp))
        l += 1
    return specs


def _build_aos_cart(structure_data, n_ao_per_atom, n_atoms):
    """Build an ``AOs_cart_data`` with *n_ao_per_atom* primitives per atom."""
    specs = _cartesian_ao_specs(n_ao_per_atom)
    num_ao = n_atoms * len(specs)
    num_ao_prim = num_ao  # one primitive per contracted AO

    nucleus_index = []
    orbital_indices = []
    exponents_list = []
    coefficients_list = []
    angular_momentums = []
    poly_x, poly_y, poly_z = [], [], []

    ao_idx = 0
    for atom_idx in range(n_atoms):
        for l, nx, ny, nz, exp in specs:
            nucleus_index.append(atom_idx)
            orbital_indices.append(ao_idx)
            exponents_list.append(exp)
            coefficients_list.append(1.0)
            angular_momentums.append(l)
            poly_x.append(nx)
            poly_y.append(ny)
            poly_z.append(nz)
            ao_idx += 1

    return AOs_cart_data(
        structure_data=structure_data,
        nucleus_index=tuple(nucleus_index),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=tuple(orbital_indices),
        exponents=jnp.array(exponents_list, dtype=jnp.float64),
        coefficients=jnp.array(coefficients_list, dtype=jnp.float64),
        angular_momentums=tuple(angular_momentums),
        polynominal_order_x=tuple(poly_x),
        polynominal_order_y=tuple(poly_y),
        polynominal_order_z=tuple(poly_z),
    )


print("Building synthetic system ...")

# --- Atom positions (cubic grid) ----------------------------------------
_side = int(np.ceil(N_ATOMS ** (1.0 / 3.0)))
_positions_list = []
for _ix in range(_side):
    for _iy in range(_side):
        for _iz in range(_side):
            if len(_positions_list) >= N_ATOMS:
                break
            _positions_list.append([_ix * ATOM_SPACING, _iy * ATOM_SPACING, _iz * ATOM_SPACING])
        if len(_positions_list) >= N_ATOMS:
            break
    if len(_positions_list) >= N_ATOMS:
        break
positions = np.array(_positions_list[:N_ATOMS])

# --- Atomic numbers / core electrons ------------------------------------
if ECP_FLAG:
    atomic_numbers = tuple([VALENCE_ELECTRONS_PER_ATOM + ECP_CORE_ELECTRONS] * N_ATOMS)
    z_cores = tuple([float(ECP_CORE_ELECTRONS)] * N_ATOMS)
else:
    atomic_numbers = tuple([VALENCE_ELECTRONS_PER_ATOM] * N_ATOMS)
    z_cores = tuple([0.0] * N_ATOMS)

structure_data = Structure_data(
    positions=positions,
    pbc_flag=False,
    atomic_numbers=atomic_numbers,
    element_symbols=tuple(["X"] * N_ATOMS),
    atomic_labels=tuple([f"X{i}" for i in range(N_ATOMS)]),
)

# --- Determinant basis --------------------------------------------------
aos_data_det = _build_aos_cart(structure_data, N_AO_PER_ATOM_DET, N_ATOMS)
num_ao_det = aos_data_det.num_ao

# --- Electron counts ----------------------------------------------------
total_val_el = N_ATOMS * VALENCE_ELECTRONS_PER_ATOM
num_ele_up = (total_val_el + 1) // 2
num_ele_dn = total_val_el // 2
delta_spin = num_ele_up - num_ele_dn

# --- Lambda matrix (identity + small perturbation for numerical safety) --
rng = np.random.default_rng(SEED)
_lambda_np = np.eye(num_ao_det, num_ao_det + delta_spin)
_lambda_np += rng.normal(0, 0.01, size=_lambda_np.shape)
lambda_matrix = jnp.array(_lambda_np, dtype=jnp.float64)

geminal_data = Geminal_data(
    num_electron_up=num_ele_up,
    num_electron_dn=num_ele_dn,
    orb_data_up_spin=aos_data_det,
    orb_data_dn_spin=aos_data_det,
    lambda_matrix=lambda_matrix,
)

# --- Coulomb potential --------------------------------------------------
if ECP_FLAG:
    # 3 terms per atom: non-local l=0, non-local l=1, local l=2
    # Convention: local channel has ang_mom == max_ang_mom_plus_1[atom]
    _ecp_ang, _ecp_nuc, _ecp_exp, _ecp_coef, _ecp_pow = [], [], [], [], []
    for _a in range(N_ATOMS):
        # non-local s-channel (l = 0)
        _ecp_ang.append(0)
        _ecp_nuc.append(_a)
        _ecp_exp.append(3.0)
        _ecp_coef.append(-1.0)
        _ecp_pow.append(2)
        # non-local p-channel (l = 1)
        _ecp_ang.append(1)
        _ecp_nuc.append(_a)
        _ecp_exp.append(2.0)
        _ecp_coef.append(-0.5)
        _ecp_pow.append(2)
        # local channel (l = 2 = max_ang_mom_plus_1)
        _ecp_ang.append(2)
        _ecp_nuc.append(_a)
        _ecp_exp.append(5.0)
        _ecp_coef.append(float(VALENCE_ELECTRONS_PER_ATOM))
        _ecp_pow.append(2)

    coulomb_potential_data = Coulomb_potential_data(
        structure_data=structure_data,
        ecp_flag=True,
        z_cores=z_cores,
        max_ang_mom_plus_1=tuple([2] * N_ATOMS),
        num_ecps=N_ATOMS * 3,
        ang_moms=tuple(_ecp_ang),
        nucleus_index=tuple(_ecp_nuc),
        exponents=tuple(_ecp_exp),
        coefficients=tuple(_ecp_coef),
        powers=tuple(_ecp_pow),
    )
else:
    coulomb_potential_data = Coulomb_potential_data(
        structure_data=structure_data,
        ecp_flag=False,
    )

# --- Jastrow components -------------------------------------------------
jastrow_twobody_data = None
jastrow_onebody_data = None
jastrow_threebody_data = None
aos_data_j3 = None

if J2_FLAG:
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0,
    )

if J1_FLAG:
    jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=z_cores,
    )

if J3_FLAG:
    aos_data_j3 = _build_aos_cart(structure_data, N_AO_PER_ATOM_J3, N_ATOMS)
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data_j3,
        random_init=True,
        seed=SEED,
    )

if JNN_FLAG:
    raise NotImplementedError("JNN_FLAG=True is not yet supported for synthetic data.")

jastrow_data = Jastrow_data(
    jastrow_one_body_data=jastrow_onebody_data,
    jastrow_two_body_data=jastrow_twobody_data,
    jastrow_three_body_data=jastrow_threebody_data,
)

# --- Wavefunction and Hamiltonian ---------------------------------------
wavefunction_data = Wavefunction_data(
    geminal_data=geminal_data,
    jastrow_data=jastrow_data,
)
hamiltonian_data = Hamiltonian_data(
    structure_data=structure_data,
    coulomb_potential_data=coulomb_potential_data,
    wavefunction_data=wavefunction_data,
)

# --- Print summary ------------------------------------------------------
_jastrow_parts = []
if J1_FLAG:
    _jastrow_parts.append("J1")
if J2_FLAG:
    _jastrow_parts.append("J2")
if J3_FLAG:
    _jastrow_parts.append(f"J3(n_AO={aos_data_j3._num_orb})")
if JNN_FLAG:
    _jastrow_parts.append("JNN")
jastrow_str = "+".join(_jastrow_parts) if _jastrow_parts else "none"

print(f"  atoms: {N_ATOMS} | val.el./atom: {VALENCE_ELECTRONS_PER_ATOM}")
print(f"  electrons: {num_ele_up} up, {num_ele_dn} dn")
print(f"  AOs (det): {num_ao_det} ({N_AO_PER_ATOM_DET}/atom)")
if J3_FLAG:
    print(f"  AOs (J3):  {aos_data_j3.num_ao} ({N_AO_PER_ATOM_J3}/atom)")
print(f"  Jastrow: {jastrow_str} | ECP: {ECP_FLAG}")
print(f"  walkers: {N_WALKERS} | repeats per kernel: {REPEATS}\n")

# ==============================================================================
# Initialize walkers  (electrons placed near atoms with Gaussian scatter)
# ==============================================================================
_atom_idx_up = np.arange(num_ele_up) % N_ATOMS
_atom_idx_dn = np.arange(num_ele_dn) % N_ATOMS
_centers_up = positions[_atom_idx_up]  # (num_ele_up, 3)
_centers_dn = positions[_atom_idx_dn]  # (num_ele_dn, 3)

r_up_batch = jnp.asarray(np.tile(_centers_up, (N_WALKERS, 1, 1)) + rng.normal(0, 1.0, size=(N_WALKERS, num_ele_up, 3)))
r_dn_batch = jnp.asarray(np.tile(_centers_dn, (N_WALKERS, 1, 1)) + rng.normal(0, 1.0, size=(N_WALKERS, num_ele_dn, 3)))

RT = jnp.eye(3)
RT_batch = jnp.broadcast_to(RT, (N_WALKERS, 3, 3))

# Pre-compute geminal matrix and its inverse per walker
# (in the actual MCMC loop these are maintained via Sherman-Morrison rank-1 updates)


@jit
def _compute_geminal_and_inv(r_up, r_dn):
    G = compute_geminal_all_elements(
        geminal_data=geminal_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
    )
    lu, piv = jsp_linalg.lu_factor(G)
    G_inv = jsp_linalg.lu_solve((lu, piv), jnp.eye(G.shape[0], dtype=G.dtype))
    return G, G_inv


_compute_geminal_and_inv_vmap = jit(vmap(_compute_geminal_and_inv))
geminal_batch, geminal_inv_batch = _compute_geminal_and_inv_vmap(r_up_batch, r_dn_batch)
block_until_ready(geminal_batch)
block_until_ready(geminal_inv_batch)

# ==============================================================================
# Build vmapped + jitted kernels
#
# Each kernel is wrapped with jax.named_scope so that Nsight Compute can
# identify which CUDA kernels belong to which MCMC sub-operation.
# ==============================================================================

# 1. evaluate_ln_wavefunction_fast  (uses pre-computed geminal_inv)


def _eval_ln_psi_fast(wf, r_up, r_dn, geminal_inv):
    with jax.named_scope("evaluate_ln_wavefunction_fast"):
        return evaluate_ln_wavefunction_fast(wf, r_up, r_dn, geminal_inv)


kernel_eval_ln_psi = jit(vmap(_eval_ln_psi_fast, in_axes=(None, 0, 0, 0)))

# 2. compute_local_energy_fast  (uses pre-computed geminal_inverse)


def _local_energy_fast(h, r_up, r_dn, RT, geminal_inverse):
    with jax.named_scope("compute_local_energy_fast"):
        return compute_local_energy_fast(h, r_up, r_dn, RT, geminal_inverse)


kernel_local_energy = jit(vmap(_local_energy_fast, in_axes=(None, 0, 0, 0, 0)))

# 3. compute_AS_regularization_factor (fast_update version -- no SVD)


def _as_reg(geminal, geminal_inv):
    with jax.named_scope("AS_regularization"):
        return compute_AS_regularization_factor_fast_update(geminal, geminal_inv)


kernel_as_reg = jit(vmap(_as_reg, in_axes=(0, 0)))

# ==============================================================================
# Warmup  (JIT compile all kernels)
# ==============================================================================
print("=" * 80)
print("Warming up (JIT compilation) ...")
t0 = time.perf_counter()

block_until_ready(kernel_eval_ln_psi(wavefunction_data, r_up_batch, r_dn_batch, geminal_inv_batch))
block_until_ready(kernel_local_energy(hamiltonian_data, r_up_batch, r_dn_batch, RT_batch, geminal_inv_batch))
block_until_ready(kernel_as_reg(geminal_batch, geminal_inv_batch))

print(f"  compilation done in {time.perf_counter() - t0:.1f} s\n")

# ==============================================================================
# Start CUDA profiler -- everything below this line is captured by nsys/ncu
# when using --capture-range=cudaProfilerApi
# ==============================================================================
cuda_profiler_start()
nvtx_push("exec/all_benchmarks")

# ==============================================================================
# Benchmark
# ==============================================================================
print("=" * 80)
print(f"MCMC component benchmarks  ({N_WALKERS} walkers vmapped, mean +/- std over {REPEATS} repeats)")
print(f"\n  Jastrow: {jastrow_str} | ECP: {ECP_FLAG}\n")
print(f"  {'kernel':<62s}  {'time':>18s}")
print(f"  {'-' * 62}  {'-' * 18}")

nvtx_push("exec/mcmc_kernels")

bench(
    "1. evaluate_ln_wavefunction_fast",
    lambda: kernel_eval_ln_psi(wavefunction_data, r_up_batch, r_dn_batch, geminal_inv_batch),
)
bench(
    "2. compute_local_energy_fast",
    lambda: kernel_local_energy(hamiltonian_data, r_up_batch, r_dn_batch, RT_batch, geminal_inv_batch),
)
bench(
    "3. compute_AS_regularization_factor_fast_update",
    lambda: kernel_as_reg(geminal_batch, geminal_inv_batch),
)

nvtx_pop()  # close "exec/mcmc_kernels"

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
print("      --trace cuda,nvtx -o prof python benchmark_mcmc_kernels.py")
print()
print("  # Post-process -> CSV for LLM analysis")
print("  nsys stats prof.nsys-rep --report cuda_gpu_kern_sum --format csv > summary.csv")
print("  nsys stats prof.nsys-rep --report nvtx_gpu_proj_trace --format csv > nvtx_trace.csv")
print()
print("  # Nsight Compute")
print("  ncu --replay-mode application -o mcmc_kernels python benchmark_mcmc_kernels.py")
print("  ncu -i mcmc_kernels.ncu-rep --csv > ncu_results.csv")
print()
