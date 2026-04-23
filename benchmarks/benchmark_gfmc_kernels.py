"""GFMC projection kernel benchmarks for Nsight Systems / Nsight Compute profiling.

Profiles every computational kernel called within the "Projection time per
branching" hot loop of GFMC_t and GFMC_n (``jqmc_gfmc.py``).

Designed to be run under NVIDIA profiling tools:

    # Nsight Systems -- captures only the execution phase (after JIT compilation)
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
        --trace cuda,nvtx -o prof python benchmark_gfmc_kernels.py

    # Nsight Compute -- same capture-range filtering
    ncu --replay-mode application --nvtx --nvtx-include "exec/" \\
        -o gfmc_kernels python benchmark_gfmc_kernels.py

Profiling strategy:
  - cudaProfilerStart/Stop brackets the execution phase so that JIT
    compilation is excluded from the nsys/ncu capture.
  - NVTX ranges (via ctypes + libnvToolsExt.so, no pip install needed)
    label each kernel on the Nsight Systems timeline.
  - jax.named_scope embeds names into XLA HLO so Nsight Compute can
    map individual CUDA kernels back to source operations.

Kernels benchmarked (in projection-step order):
  1. compute_kinetic_energy_all_elements_fast_update   (continuum kinetic)
  2. compute_discretized_kinetic_energy_fast_update     (discretized kinetic + mesh)
  3. compute_bare_coulomb_potential_el_el               (e-e Coulomb)
  4. compute_bare_coulomb_potential_ion_ion              (ion-ion Coulomb)
  5. compute_bare_coulomb_potential_el_ion_element_wise  (e-ion Coulomb)
  6. compute_discretized_bare_coulomb_potential_el_ion_element_wise  (discretized e-ion)
  7. compute_ecp_local_parts_all_pairs                  (ECP local, if ECP system)
  8. compute_ecp_non_local_parts_nearest_neighbors_fast_update  (ECP non-local)
  9. _compute_ratio_Jastrow_part_rank1_update           (Jastrow ratio, dltmove)
 10. compute_geminal_up_one_row_elements                (Sherman-Morrison up)
 11. compute_geminal_dn_one_column_elements             (Sherman-Morrison dn)
 12. full_projection_step                               (all of the above combined)
"""

import ctypes
import os
import sys
import time

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from functools import partial
from jax import jit, lax, vmap

from jqmc.coulomb_potential import (
    compute_bare_coulomb_potential_el_el,
    compute_bare_coulomb_potential_el_ion_element_wise,
    compute_bare_coulomb_potential_ion_ion,
    compute_discretized_bare_coulomb_potential_el_ion_element_wise,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors_fast_update,
)
from jqmc.determinant import (
    compute_geminal_all_elements,
    compute_geminal_dn_one_column_elements,
    compute_geminal_up_one_row_elements,
)
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _compute_ratio_Jastrow_part_rank1_update,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import (
    Wavefunction_data,
    compute_discretized_kinetic_energy_fast_update,
    compute_kinetic_energy_all_elements_fast_update,
)

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
SEED = 42
ALAT = 0.30
R_CART_MIN, R_CART_MAX = -5.0, +5.0
NON_LOCAL_MOVE = "tmove"  # "tmove" or "dltmove"

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

jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)
jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
    jastrow_1b_param=1.0,
    structure_data=structure_data,
    core_electrons=coulomb_potential_data.z_cores,
)
jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
    orb_data=aos_data,
    random_init=True,
    seed=SEED,
)
jastrow_data = Jastrow_data(
    jastrow_one_body_data=jastrow_onebody_data,
    jastrow_two_body_data=jastrow_twobody_data,
    jastrow_three_body_data=jastrow_threebody_data,
)
wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)
hamiltonian_data = Hamiltonian_data(
    structure_data=structure_data,
    coulomb_potential_data=coulomb_potential_data,
    wavefunction_data=wavefunction_data,
)

num_ele_up = geminal_mo_data.num_electron_up
num_ele_dn = geminal_mo_data.num_electron_dn
ecp_flag = coulomb_potential_data.ecp_flag

print(f"  electrons: {num_ele_up} up, {num_ele_dn} dn | ECP: {ecp_flag}")
print(f"  walkers: {N_WALKERS} | alat: {ALAT} | non_local_move: {NON_LOCAL_MOVE}")
print(f"  repeats per kernel: {REPEATS}\n")

# ==============================================================================
# Initialize walkers  (batched arrays of shape (N_WALKERS, ...))
# ==============================================================================
rng = np.random.default_rng(SEED)
r_up_batch = jnp.asarray(rng.uniform(R_CART_MIN, R_CART_MAX, size=(N_WALKERS, num_ele_up, 3)))
r_dn_batch = jnp.asarray(rng.uniform(R_CART_MIN, R_CART_MAX, size=(N_WALKERS, num_ele_dn, 3)))

RT = jnp.eye(3)


# Pre-compute geminal inverse per walker
@jit
def _compute_geminal_and_inv(r_up, r_dn):
    G = compute_geminal_all_elements(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
    )
    lu, piv = jsp_linalg.lu_factor(G)
    return jsp_linalg.lu_solve((lu, piv), jnp.eye(G.shape[0], dtype=G.dtype))


_compute_geminal_and_inv_vmap = jit(vmap(_compute_geminal_and_inv))
A_old_inv_batch = _compute_geminal_and_inv_vmap(r_up_batch, r_dn_batch)
block_until_ready(A_old_inv_batch)

# Pre-compute mesh of proposed moves (needed for Jastrow ratio & rank-1 benchmarks)
_disc_kin_single = compute_discretized_kinetic_energy_fast_update(
    alat=ALAT,
    wavefunction_data=wavefunction_data,
    A_old_inv=A_old_inv_batch[0],
    r_up_carts=r_up_batch[0],
    r_dn_carts=r_dn_batch[0],
    RT=RT,
)
mesh_r_up_carts = _disc_kin_single[0]  # (num_mesh, num_ele_up, 3)
mesh_r_dn_carts = _disc_kin_single[1]  # (num_mesh, num_ele_dn, 3)

# ==============================================================================
# Build vmapped + jitted kernels
#
# Each kernel is wrapped with jax.named_scope so that Nsight Compute can
# identify which CUDA kernels belong to which projection sub-operation.
# ==============================================================================


# 1. Continuum kinetic energy
@jit
@vmap
def kernel_kinetic_continuum(r_up, r_dn, inv):
    with jax.named_scope("kinetic_continuum"):
        return compute_kinetic_energy_all_elements_fast_update(
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            geminal_inverse=inv,
        )


# 2. Discretized kinetic energy
@jit
@vmap
def kernel_kinetic_discretized(r_up, r_dn, inv):
    with jax.named_scope("kinetic_discretized"):
        return compute_discretized_kinetic_energy_fast_update(
            alat=ALAT,
            wavefunction_data=wavefunction_data,
            A_old_inv=inv,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            RT=RT,
        )


# 3. Electron-electron Coulomb
@jit
@vmap
def kernel_coulomb_el_el(r_up, r_dn):
    with jax.named_scope("coulomb_el_el"):
        return compute_bare_coulomb_potential_el_el(r_up_carts=r_up, r_dn_carts=r_dn)


# 4. Ion-ion Coulomb  (not vmapped -- result is identical for all walkers)
@jit
def kernel_coulomb_ion_ion():
    with jax.named_scope("coulomb_ion_ion"):
        return compute_bare_coulomb_potential_ion_ion(
            coulomb_potential_data=coulomb_potential_data,
        )


# 5. Electron-ion Coulomb
@jit
@vmap
def kernel_coulomb_el_ion(r_up, r_dn):
    with jax.named_scope("coulomb_el_ion"):
        return compute_bare_coulomb_potential_el_ion_element_wise(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
        )


# 6. Discretized electron-ion Coulomb
@jit
@vmap
def kernel_coulomb_el_ion_disc(r_up, r_dn):
    with jax.named_scope("coulomb_el_ion_disc"):
        return compute_discretized_bare_coulomb_potential_el_ion_element_wise(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            alat=ALAT,
        )


# 7. ECP local  (only meaningful for ECP systems)
@jit
@vmap
def kernel_ecp_local(r_up, r_dn):
    with jax.named_scope("ecp_local"):
        return compute_ecp_local_parts_all_pairs(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
        )


# 8. ECP non-local (tmove, flag_determinant_only=False)
@jit
@vmap
def kernel_ecp_nonlocal_tmove(r_up, r_dn, inv):
    with jax.named_scope("ecp_nonlocal_tmove"):
        return compute_ecp_non_local_parts_nearest_neighbors_fast_update(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            RT=RT,
            A_old_inv=inv,
            flag_determinant_only=False,
        )


# 8b. ECP non-local (dltmove, flag_determinant_only=True)
@jit
@vmap
def kernel_ecp_nonlocal_dltmove(r_up, r_dn, inv):
    with jax.named_scope("ecp_nonlocal_dltmove"):
        return compute_ecp_non_local_parts_nearest_neighbors_fast_update(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            RT=RT,
            A_old_inv=inv,
            flag_determinant_only=True,
        )


# 9. Jastrow ratio  (single-walker; vmap over mesh is already internal)
@jit
def kernel_jastrow_ratio():
    with jax.named_scope("jastrow_ratio"):
        return _compute_ratio_Jastrow_part_rank1_update(
            jastrow_data=wavefunction_data.jastrow_data,
            old_r_up_carts=r_up_batch[0],
            old_r_dn_carts=r_dn_batch[0],
            new_r_up_carts_arr=mesh_r_up_carts,
            new_r_dn_carts_arr=mesh_r_dn_carts,
        )


# 10. Geminal up one-row (Sherman-Morrison)
@jit
def kernel_geminal_up_one_row():
    with jax.named_scope("geminal_up_one_row"):
        return compute_geminal_up_one_row_elements(
            geminal_data=wavefunction_data.geminal_data,
            r_up_cart=jnp.reshape(mesh_r_up_carts[0, 0], (1, 3)),
            r_dn_carts=r_dn_batch[0],
        )


# 11. Geminal dn one-column (Sherman-Morrison)
@jit
def kernel_geminal_dn_one_col():
    with jax.named_scope("geminal_dn_one_col"):
        return compute_geminal_dn_one_column_elements(
            geminal_data=wavefunction_data.geminal_data,
            r_up_carts=r_up_batch[0],
            r_dn_cart=jnp.reshape(mesh_r_dn_carts[0, 0], (1, 3)),
        )


# 12. Full single projection step  (mirrors _projection_t / _body_fun_n)
@partial(jit, static_argnums=(3,))
def _single_projection_step(r_up_carts, r_dn_carts, A_old_inv, non_local_move):
    """One projection step matching the body of _projection_t / _body_fun_n."""
    with jax.named_scope("full_projection_step"):
        alat = ALAT

        # Kinetic energy (continuum)
        with jax.named_scope("kinetic_continuum"):
            diagonal_kinetic_continuum_elements_up, diagonal_kinetic_continuum_elements_dn = (
                compute_kinetic_energy_all_elements_fast_update(
                    wavefunction_data=wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                    geminal_inverse=A_old_inv,
                )
            )

        # Discretized kinetic energy
        with jax.named_scope("kinetic_discretized"):
            mesh_kinetic_part_r_up, mesh_kinetic_part_r_dn, elements_non_diagonal_kinetic = (
                compute_discretized_kinetic_energy_fast_update(
                    alat=alat,
                    wavefunction_data=wavefunction_data,
                    A_old_inv=A_old_inv,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                    RT=RT,
                )
            )

        # Coulomb potentials
        with jax.named_scope("coulomb_el_el"):
            diagonal_bare_coulomb_el_el = compute_bare_coulomb_potential_el_el(
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )
        with jax.named_scope("coulomb_ion_ion"):
            diagonal_bare_coulomb_ion_ion = compute_bare_coulomb_potential_ion_ion(
                coulomb_potential_data=coulomb_potential_data,
            )
        with jax.named_scope("coulomb_el_ion"):
            el_ion_up, el_ion_dn = compute_bare_coulomb_potential_el_ion_element_wise(
                coulomb_potential_data=coulomb_potential_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )
        with jax.named_scope("coulomb_el_ion_disc"):
            el_ion_disc_up, el_ion_disc_dn = compute_discretized_bare_coulomb_potential_el_ion_element_wise(
                coulomb_potential_data=coulomb_potential_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                alat=alat,
            )

        # ECP (if applicable)
        if hamiltonian_data.coulomb_potential_data.ecp_flag:
            with jax.named_scope("ecp_local"):
                diagonal_ecp_local = compute_ecp_local_parts_all_pairs(
                    coulomb_potential_data=coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

            if non_local_move == "tmove":
                with jax.named_scope("ecp_nonlocal_tmove"):
                    mesh_ecp_r_up, mesh_ecp_r_dn, V_nonlocal, _ = compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                        coulomb_potential_data=coulomb_potential_data,
                        wavefunction_data=wavefunction_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                        RT=RT,
                        A_old_inv=A_old_inv,
                        flag_determinant_only=False,
                    )
            else:  # dltmove
                with jax.named_scope("ecp_nonlocal_dltmove"):
                    mesh_ecp_r_up, mesh_ecp_r_dn, V_nonlocal, _ = compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                        coulomb_potential_data=coulomb_potential_data,
                        wavefunction_data=wavefunction_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                        RT=RT,
                        A_old_inv=A_old_inv,
                        flag_determinant_only=True,
                    )
                with jax.named_scope("jastrow_ratio"):
                    Jastrow_ratio = _compute_ratio_Jastrow_part_rank1_update(
                        jastrow_data=wavefunction_data.jastrow_data,
                        old_r_up_carts=r_up_carts,
                        old_r_dn_carts=r_dn_carts,
                        new_r_up_carts_arr=mesh_ecp_r_up,
                        new_r_dn_carts_arr=mesh_ecp_r_dn,
                    )
                    V_nonlocal = jnp.minimum(V_nonlocal, 0.0) * Jastrow_ratio

            # Merge kinetic + ECP non-local meshes
            non_diag_FN = jnp.minimum(elements_non_diagonal_kinetic, 0.0)
            V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
            p_list = jnp.concatenate([jnp.ravel(non_diag_FN), jnp.ravel(V_nonlocal_FN)])
            mesh_r_up_all = jnp.concatenate([mesh_kinetic_part_r_up, mesh_ecp_r_up], axis=0)
            mesh_r_dn_all = jnp.concatenate([mesh_kinetic_part_r_dn, mesh_ecp_r_dn], axis=0)
        else:
            p_list = jnp.ravel(jnp.minimum(elements_non_diagonal_kinetic, 0.0))
            mesh_r_up_all = mesh_kinetic_part_r_up
            mesh_r_dn_all = mesh_kinetic_part_r_dn

        # Move selection
        cdf = jnp.cumsum(p_list / p_list.sum())
        k = jnp.searchsorted(cdf, 0.5)  # deterministic for benchmarking
        proposed_r_up = mesh_r_up_all[k]
        proposed_r_dn = mesh_r_dn_all[k]

        # Sherman-Morrison rank-1 update
        num_up = r_up_carts.shape[0]
        num_dn = r_dn_carts.shape[0]

        if num_up > 0:
            up_diff = jnp.any(r_up_carts != proposed_r_up, axis=1)
            has_up_move = jnp.any(up_diff)
            up_index = jnp.argmax(up_diff)

            with jax.named_scope("geminal_up_one_row"):
                v_new = compute_geminal_up_one_row_elements(
                    geminal_data=wavefunction_data.geminal_data,
                    r_up_cart=jnp.reshape(proposed_r_up[up_index], (1, 3)),
                    r_dn_carts=r_dn_carts,
                )
                v_old = compute_geminal_up_one_row_elements(
                    geminal_data=wavefunction_data.geminal_data,
                    r_up_cart=jnp.reshape(r_up_carts[up_index], (1, 3)),
                    r_dn_carts=r_dn_carts,
                )

        if num_dn > 0:
            dn_diff = jnp.any(r_dn_carts != proposed_r_dn, axis=1)
            has_dn_move = jnp.any(dn_diff)
            dn_index = jnp.argmax(dn_diff)

            with jax.named_scope("geminal_dn_one_col"):
                u_new = compute_geminal_dn_one_column_elements(
                    geminal_data=wavefunction_data.geminal_data,
                    r_up_carts=r_up_carts,
                    r_dn_cart=jnp.reshape(proposed_r_dn[dn_index], (1, 3)),
                )
                u_old = compute_geminal_dn_one_column_elements(
                    geminal_data=wavefunction_data.geminal_data,
                    r_up_carts=r_up_carts,
                    r_dn_cart=jnp.reshape(r_dn_carts[dn_index], (1, 3)),
                )

        return proposed_r_up, proposed_r_dn


# vmapped full projection step
_full_step_vmap = jit(
    vmap(
        lambda r_up, r_dn, inv: _single_projection_step(r_up, r_dn, inv, NON_LOCAL_MOVE),
    ),
)

# ==============================================================================
# Warmup  (JIT compile all kernels)
# ==============================================================================
print("=" * 80)
print("Warming up (JIT compilation) ...")
t0 = time.perf_counter()

block_until_ready(kernel_kinetic_continuum(r_up_batch, r_dn_batch, A_old_inv_batch))
block_until_ready(kernel_kinetic_discretized(r_up_batch, r_dn_batch, A_old_inv_batch))
block_until_ready(kernel_coulomb_el_el(r_up_batch, r_dn_batch))
block_until_ready(kernel_coulomb_ion_ion())
block_until_ready(kernel_coulomb_el_ion(r_up_batch, r_dn_batch))
block_until_ready(kernel_coulomb_el_ion_disc(r_up_batch, r_dn_batch))
if ecp_flag:
    block_until_ready(kernel_ecp_local(r_up_batch, r_dn_batch))
    block_until_ready(kernel_ecp_nonlocal_tmove(r_up_batch, r_dn_batch, A_old_inv_batch))
    block_until_ready(kernel_ecp_nonlocal_dltmove(r_up_batch, r_dn_batch, A_old_inv_batch))
    block_until_ready(kernel_jastrow_ratio())
if num_ele_up > 0:
    block_until_ready(kernel_geminal_up_one_row())
if num_ele_dn > 0:
    block_until_ready(kernel_geminal_dn_one_col())
block_until_ready(_full_step_vmap(r_up_batch, r_dn_batch, A_old_inv_batch))

print(f"  compilation done in {time.perf_counter() - t0:.1f} s\n")

# ==============================================================================
# Start CUDA profiler -- everything below this line is captured by nsys/ncu
# when using --capture-range=cudaProfilerApi
# ==============================================================================
cuda_profiler_start()
nvtx_push("exec/all_benchmarks")

# ==============================================================================
# Individual kernel benchmarks
# ==============================================================================
print("=" * 80)
print(f"Individual kernel benchmarks  ({N_WALKERS} walkers vmapped, mean +/- std over {REPEATS} repeats)")
print(f"  {'kernel':<62s}  {'time':>18s}")
print(f"  {'-' * 62}  {'-' * 18}")

bench(
    "1. kinetic_continuum_fast_update",
    lambda: kernel_kinetic_continuum(r_up_batch, r_dn_batch, A_old_inv_batch),
)
bench(
    "2. kinetic_discretized_fast_update",
    lambda: kernel_kinetic_discretized(r_up_batch, r_dn_batch, A_old_inv_batch),
)
bench(
    "3. bare_coulomb_el_el",
    lambda: kernel_coulomb_el_el(r_up_batch, r_dn_batch),
)
bench(
    "4. bare_coulomb_ion_ion",
    lambda: kernel_coulomb_ion_ion(),
)
bench(
    "5. bare_coulomb_el_ion",
    lambda: kernel_coulomb_el_ion(r_up_batch, r_dn_batch),
)
bench(
    "6. bare_coulomb_el_ion_discretized",
    lambda: kernel_coulomb_el_ion_disc(r_up_batch, r_dn_batch),
)

if ecp_flag:
    bench(
        "7. ecp_local_all_pairs",
        lambda: kernel_ecp_local(r_up_batch, r_dn_batch),
    )
    bench(
        "8a. ecp_nonlocal_tmove_fast_update",
        lambda: kernel_ecp_nonlocal_tmove(r_up_batch, r_dn_batch, A_old_inv_batch),
    )
    bench(
        "8b. ecp_nonlocal_dltmove_fast_update",
        lambda: kernel_ecp_nonlocal_dltmove(r_up_batch, r_dn_batch, A_old_inv_batch),
    )
    bench(
        "9. jastrow_ratio_rank1_update",
        lambda: kernel_jastrow_ratio(),
    )
else:
    print(f"  {'7-9. (skipped -- not an ECP system)':<62s}")

if num_ele_up > 0:
    bench(
        "10. geminal_up_one_row  (Sherman-Morrison)",
        lambda: kernel_geminal_up_one_row(),
    )
else:
    print(f"  {'10. (skipped -- no up electrons)':<62s}")

if num_ele_dn > 0:
    bench(
        "11. geminal_dn_one_col  (Sherman-Morrison)",
        lambda: kernel_geminal_dn_one_col(),
    )
else:
    print(f"  {'11. (skipped -- no dn electrons)':<62s}")

# ==============================================================================
# Full projection step benchmark
# ==============================================================================
print()
print(f"  {'-' * 62}  {'-' * 18}")
bench(
    "12. FULL projection step  (all kernels combined)",
    lambda: _full_step_vmap(r_up_batch, r_dn_batch, A_old_inv_batch),
)

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
print("      --trace cuda,nvtx -o prof python benchmark_gfmc_kernels.py")
print()
print("  # Post-process -> CSV for LLM analysis")
print("  nsys stats prof.nsys-rep --report cuda_gpu_kern_sum --format csv > summary.csv")
print("  nsys stats prof.nsys-rep --report nvtx_gpu_proj_trace --format csv > nvtx_trace.csv")
print()
print("  # Nsight Compute")
print("  ncu --replay-mode application -o gfmc_kernels python benchmark_gfmc_kernels.py")
print("  ncu -i gfmc_kernels.ncu-rep --csv > ncu_results.csv")
print()
