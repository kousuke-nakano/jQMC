"""Profile internal cost breakdown: AO evaluation vs GEMM vs Jastrow."""

import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy import linalg as jsp_linalg

jax.config.update("jax_enable_x64", True)

from jqmc.atomic_orbital import compute_AOs
from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors_fast_update
from jqmc.determinant import compute_geminal_all_elements
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _compute_ratio_Jastrow_part_rank1_update,
    compute_grads_and_laplacian_Jastrow_three_body,
)
from jqmc.molecular_orbital import compute_MOs
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import (
    Wavefunction_data,
    compute_discretized_kinetic_energy_fast_update,
    _compute_ratio_determinant_part_rank1_update,
)

TREXIO_FILE = os.path.join(os.path.dirname(__file__), "C6H6_ccecp_augccpvtz.h5")
N = 50
ALAT = 0.30

_, _, _, _, gd, coulomb_data = read_trexio_file(TREXIO_FILE, store_tuple=True)
jd = Jastrow_data(None, Jastrow_two_body_data.init_jastrow_two_body_data(1.0), None)
wf_data = Wavefunction_data(geminal_data=gd, jastrow_data=jd)

n_up, n_dn = gd.num_electron_up, gd.num_electron_dn
rng = np.random.default_rng(0)
r_up = jnp.asarray(rng.uniform(-5.0, 5.0, (n_up, 3)))
r_dn = jnp.asarray(rng.uniform(-5.0, 5.0, (n_dn, 3)))
RT = jnp.eye(3)

G = compute_geminal_all_elements(gd, r_up, r_dn)
lu, piv = jsp_linalg.lu_factor(G)
A_inv = jsp_linalg.lu_solve((lu, piv), jnp.eye(G.shape[0], dtype=G.dtype))

# Full functions (warm-up)
compute_discretized_kinetic_energy_fast_update(ALAT, wf_data, A_inv, r_up, r_dn, RT)[0].block_until_ready()
compute_ecp_non_local_parts_nearest_neighbors_fast_update(coulomb_data, wf_data, r_up, r_dn, RT, A_inv)[0].block_until_ready()


def timefull(fn, *a):
    fn(*a)[0].block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N):
        fn(*a)[0].block_until_ready()
    return (time.perf_counter() - t0) / N * 1000


t_kin = timefull(compute_discretized_kinetic_energy_fast_update, ALAT, wf_data, A_inv, r_up, r_dn, RT)
t_ecp = timefull(compute_ecp_non_local_parts_nearest_neighbors_fast_update, coulomb_data, wf_data, r_up, r_dn, RT, A_inv)

print(f"n_up={n_up}, n_dn={n_dn}")
print(f"n_AO={gd.orb_data_up_spin.aos_data.num_ao}, n_prim={gd.orb_data_up_spin.aos_data.num_ao_prim}")
print(f"\nFull functions:")
print(f"  kinetic_discretized:              {t_kin:.3f} ms")
print(f"  ecp_non_local_tmove:              {t_ecp:.3f} ms")

# AO/MO sub-cost for representative batch sizes
aos_data_up = gd.orb_data_up_spin.aos_data
mos_data_up = gd.orb_data_up_spin
aos_fn = jit(compute_AOs)
mos_fn = jit(compute_MOs)

for N_grid in [1, 15, 90, 180]:
    r_batch = jnp.asarray(rng.uniform(-5.0, 5.0, (N_grid, 3)))
    aos_fn(aos_data_up, r_batch).block_until_ready()
    mos_fn(mos_data_up, r_batch).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N):
        aos_fn(aos_data_up, r_batch).block_until_ready()
    t_aos = (time.perf_counter() - t0) / N * 1000
    t0 = time.perf_counter()
    for _ in range(N):
        mos_fn(mos_data_up, r_batch).block_until_ready()
    t_mos = (time.perf_counter() - t0) / N * 1000
    n_prim = gd.orb_data_up_spin.aos_data.num_ao_prim
    print(
        f"\n  batch={N_grid:4d}: compute_AOs={t_aos:.3f} ms  compute_MOs={t_mos:.3f} ms  "
        f"(exp theory min={n_prim * N_grid * 5e-6:.3f} ms)"
    )

# Ratio sub-cost for kinetic batch
n_kin_batch = (n_up + n_dn) * 6
r_up_b = jnp.broadcast_to(r_up[None], (n_kin_batch, n_up, 3))
r_dn_b = jnp.broadcast_to(r_dn[None], (n_kin_batch, n_dn, 3))
det_fn = jit(_compute_ratio_determinant_part_rank1_update)
jas_fn = jit(_compute_ratio_Jastrow_part_rank1_update)
det_fn(gd, A_inv, r_up, r_dn, r_up_b, r_dn_b).block_until_ready()
jas_fn(jd, r_up, r_dn, r_up_b, r_dn_b).block_until_ready()

t0 = time.perf_counter()
for _ in range(N):
    det_fn(gd, A_inv, r_up, r_dn, r_up_b, r_dn_b).block_until_ready()
t_det = (time.perf_counter() - t0) / N * 1000
t0 = time.perf_counter()
for _ in range(N):
    jas_fn(jd, r_up, r_dn, r_up_b, r_dn_b).block_until_ready()
t_jas = (time.perf_counter() - t0) / N * 1000

print(f"\nRatio sub-cost ({n_kin_batch} configs = kinetic batch):")
print(f"  det ratio (rank-1):  {t_det:.3f} ms")
print(f"  jastrow ratio:       {t_jas:.3f} ms")
print(f"  combined:            {t_det + t_jas:.3f} ms  (kinetic full = {t_kin:.3f} ms)")

# ── 3b Jastrow breakdown ──────────────────────────────────────────────────────
trexio_file2 = TREXIO_FILE
_, aos_data_for_j3, _, _, _, coulomb_data2 = read_trexio_file(trexio_file2, store_tuple=True)

jastrow_1b = Jastrow_one_body_data.init_jastrow_one_body_data(
    jastrow_1b_param=1.0, structure_data=coulomb_data2.structure_data, core_electrons=coulomb_data2.z_cores
)
jastrow_3b = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data_for_j3, random_init=True, seed=0)
jd_full = Jastrow_data(
    jastrow_one_body_data=jastrow_1b,
    jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(1.0),
    jastrow_three_body_data=jastrow_3b,
)
wf_full = Wavefunction_data(geminal_data=gd, jastrow_data=jd_full)

# -- warm up --
from jqmc.atomic_orbital import compute_AOs_grad, compute_AOs_laplacian

compute_grads_and_laplacian_Jastrow_three_body_jit = jit(compute_grads_and_laplacian_Jastrow_three_body)
jas_fn_3b = jit(_compute_ratio_Jastrow_part_rank1_update)
compute_grads_and_laplacian_Jastrow_three_body_jit(jastrow_3b, r_up, r_dn)
jas_fn_3b(jd_full, r_up, r_dn, r_up_b, r_dn_b).block_until_ready()
compute_discretized_kinetic_energy_fast_update(ALAT, wf_full, A_inv, r_up, r_dn, RT)[0].block_until_ready()
compute_ecp_non_local_parts_nearest_neighbors_fast_update(coulomb_data, wf_full, r_up, r_dn, RT, A_inv)[0].block_until_ready()

aos_grad_fn = jit(compute_AOs_grad)
aos_lap_fn = jit(compute_AOs_laplacian)
aos_grad_fn(aos_data_for_j3, r_up)[0].block_until_ready()
aos_lap_fn(aos_data_for_j3, r_up).block_until_ready()

# -- measure AO grad/lap per electron batch --
for N_grid in [15, 30]:
    r_batch = jnp.asarray(rng.uniform(-5, 5, (N_grid, 3)))
    aos_grad_fn(aos_data_for_j3, r_batch)[0].block_until_ready()
    aos_lap_fn_b = jit(compute_AOs_laplacian)
    aos_lap_fn_b(aos_data_for_j3, r_batch).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N):
        aos_grad_fn(aos_data_for_j3, r_batch)[0].block_until_ready()
    t_ag = (time.perf_counter() - t0) / N * 1000
    t0 = time.perf_counter()
    for _ in range(N):
        aos_lap_fn_b(aos_data_for_j3, r_batch).block_until_ready()
    t_al = (time.perf_counter() - t0) / N * 1000
    print(f"\n  AO_grad(batch={N_grid}): {t_ag:.3f} ms   AO_laplacian(batch={N_grid}): {t_al:.3f} ms")

# -- grad/lap of J3 for single point (used in kinetic_continuum) --
t0 = time.perf_counter()
for _ in range(N):
    compute_grads_and_laplacian_Jastrow_three_body_jit(jastrow_3b, r_up, r_dn)
t_j3gl = (time.perf_counter() - t0) / N * 1000
print(f"\n  grads+lap J3 (single point, up+dn): {t_j3gl:.3f} ms")

# -- Jastrow ratio with 3b for 180 configs --
t0 = time.perf_counter()
for _ in range(N):
    jas_fn_3b(jd_full, r_up, r_dn, r_up_b, r_dn_b).block_until_ready()
t_j3r = (time.perf_counter() - t0) / N * 1000
print(f"  jastrow ratio (1b+2b+3b, batch={n_kin_batch}): {t_j3r:.3f} ms")

# -- just J3 ratio --
jd_3b_only = Jastrow_data(None, None, jastrow_3b)
jas_fn_3b_only = jit(_compute_ratio_Jastrow_part_rank1_update)
jas_fn_3b_only(jd_3b_only, r_up, r_dn, r_up_b, r_dn_b).block_until_ready()
t0 = time.perf_counter()
for _ in range(N):
    jas_fn_3b_only(jd_3b_only, r_up, r_dn, r_up_b, r_dn_b).block_until_ready()
t_j3r_only = (time.perf_counter() - t0) / N * 1000
print(f"  jastrow ratio (3b only,   batch={n_kin_batch}): {t_j3r_only:.3f} ms")

# -- full functions with 3b --
t_kin_3b = timefull(compute_discretized_kinetic_energy_fast_update, ALAT, wf_full, A_inv, r_up, r_dn, RT)
t_ecp_3b = timefull(compute_ecp_non_local_parts_nearest_neighbors_fast_update, coulomb_data, wf_full, r_up, r_dn, RT, A_inv)
print(f"\n  kinetic_discretized  [+3b]: {t_kin_3b:.3f} ms")
print(f"  ecp_non_local_tmove  [+3b]: {t_ecp_3b:.3f} ms")
