"""Micro-benchmarks for `_projection_n` subroutines.

Times include `.block_until_ready()` to account for async execution. The first call
is used as warmup to exclude compilation overhead from the reported timings.
"""

import os
import time

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np

from jqmc.coulomb_potential import (
    compute_bare_coulomb_potential_el_el,
    compute_bare_coulomb_potential_el_ion_element_wise,
    compute_discretized_bare_coulomb_potential_el_ion_element_wise,
    compute_ecp_non_local_parts_nearest_neighbors_fast_update,
)
from jqmc.determinant import (
    compute_geminal_all_elements,
    compute_geminal_dn_one_column_elements,
    compute_geminal_up_one_row_elements,
)
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data, compute_ratio_Jastrow_part
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, compute_discretized_kinetic_energy_fast_update

REPEATS = 5
SEED = 0
ALAT = 0.5

TREXIO_FILE = os.path.join(os.path.dirname(__file__), "water_ccecp_ccpvqz.h5")

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
hamiltonian_data = Hamiltonian_data(
    structure_data=structure_data,
    coulomb_potential_data=coulomb_potential_data,
    wavefunction_data=wavefunction_data,
)

num_ele_up = geminal_mo_data.num_electron_up
num_ele_dn = geminal_mo_data.num_electron_dn

rng = np.random.default_rng(SEED)
r_cart_min, r_cart_max = -5.0, +5.0
r_up_carts_np = (r_cart_max - r_cart_min) * rng.random((num_ele_up, 3)) + r_cart_min
r_dn_carts_np = (r_cart_max - r_cart_min) * rng.random((num_ele_dn, 3)) + r_cart_min

r_up_carts = jnp.asarray(r_up_carts_np)
r_dn_carts = jnp.asarray(r_dn_carts_np)
RT = jnp.eye(3)

# Precompute inverse geminal for fast-update paths
_geminal = compute_geminal_all_elements(
    geminal_data=wavefunction_data.geminal_data,
    r_up_carts=r_up_carts,
    r_dn_carts=r_dn_carts,
)
_lu, _piv = jsp_linalg.lu_factor(_geminal)
A_old_inv = jsp_linalg.lu_solve((_lu, _piv), jnp.eye(_geminal.shape[0], dtype=_geminal.dtype))

# A small mesh of proposed moves for Jastrow ratio and determinant rank-1 updates
mesh_r_up_carts, mesh_r_dn_carts, _ = compute_discretized_kinetic_energy_fast_update(
    alat=ALAT,
    wavefunction_data=wavefunction_data,
    A_old_inv=A_old_inv,
    r_up_carts=r_up_carts,
    r_dn_carts=r_dn_carts,
    RT=RT,
)

jastrow_ratio_args = dict(
    jastrow_data=wavefunction_data.jastrow_data,
    old_r_up_carts=r_up_carts,
    old_r_dn_carts=r_dn_carts,
    new_r_up_carts_arr=mesh_r_up_carts,
    new_r_dn_carts_arr=mesh_r_dn_carts,
)

# Pick one up/down index to mimic Sherman–Morrison updates
up_index = 0 if num_ele_up > 0 else None
dn_index = 0 if num_ele_dn > 0 else None


def _time_call(fn, *args, **kwargs):
    # Warmup
    warm = fn(*args, **kwargs)
    for leaf in jax.tree_util.tree_leaves(warm):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()

    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        for leaf in jax.tree_util.tree_leaves(out):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        times.append(time.perf_counter() - start)
    return times


def bench_discretized_kinetic_fast_update():
    return _time_call(
        compute_discretized_kinetic_energy_fast_update,
        ALAT,
        wavefunction_data,
        A_old_inv,
        r_up_carts,
        r_dn_carts,
        RT,
    )


def bench_coulomb_el_el():
    return _time_call(compute_bare_coulomb_potential_el_el, r_up_carts, r_dn_carts)


def bench_coulomb_el_ion():
    return _time_call(
        compute_bare_coulomb_potential_el_ion_element_wise,
        coulomb_potential_data,
        r_up_carts,
        r_dn_carts,
    )


def bench_coulomb_el_ion_discretized():
    return _time_call(
        compute_discretized_bare_coulomb_potential_el_ion_element_wise,
        coulomb_potential_data,
        r_up_carts,
        r_dn_carts,
        ALAT,
    )


def bench_ecp_non_local_fast_update():
    return _time_call(
        compute_ecp_non_local_parts_nearest_neighbors_fast_update,
        coulomb_potential_data,
        wavefunction_data,
        r_up_carts,
        r_dn_carts,
        RT,
        A_old_inv,
    )


def bench_jastrow_ratio():
    return _time_call(compute_ratio_Jastrow_part, **jastrow_ratio_args)


def bench_geminal_rank1_updates():
    if up_index is None and dn_index is None:
        return []

    def _up():
        return compute_geminal_up_one_row_elements(
            geminal_data=wavefunction_data.geminal_data,
            r_up_cart=jnp.reshape(mesh_r_up_carts[0, up_index], (1, 3)),
            r_dn_carts=r_dn_carts,
        )

    def _dn():
        return compute_geminal_dn_one_column_elements(
            geminal_data=wavefunction_data.geminal_data,
            r_up_carts=r_up_carts,
            r_dn_cart=jnp.reshape(mesh_r_dn_carts[0, dn_index], (1, 3)),
        )

    times = []
    if up_index is not None:
        times.append(("geminal_up_one_row_elements", _time_call(lambda: _up())))
    if dn_index is not None:
        times.append(("geminal_dn_one_column_elements", _time_call(lambda: _dn())))
    return times


if __name__ == "__main__":
    sections = [
        ("discretized_kinetic_fast_update", bench_discretized_kinetic_fast_update),
        ("bare_coulomb_el_el", bench_coulomb_el_el),
        ("bare_coulomb_el_ion", bench_coulomb_el_ion),
        ("bare_coulomb_el_ion_discretized", bench_coulomb_el_ion_discretized),
        ("ecp_non_local_fast_update", bench_ecp_non_local_fast_update),
        ("jastrow_ratio", bench_jastrow_ratio),
    ]

    print("Projection_n hotspot micro-benchmarks (seconds per call, mean over repeats):")
    for name, fn in sections:
        times = fn()
        print(f"  {name:32s}: {np.mean(times):.6f} (std {np.std(times):.6f})")

    rank1_times = bench_geminal_rank1_updates()
    for name, times in rank1_times:
        print(f"  {name:32s}: {np.mean(times):.6f} (std {np.std(times):.6f})")
