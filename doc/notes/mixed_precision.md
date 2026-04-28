# Mixed Precision

jQMC supports mixed precision computation, allowing selected parts of the
calculation to run in float32 while keeping numerically sensitive operations
in float64.  This can reduce memory usage by ~30-40% and improve GPU
throughput by ~1.5-2x for large molecules, with negligible impact on the
final energy.

## Quick start

Add a `[precision]` section to your TOML input file:

```toml
[precision]
mode = "mixed"
```

Or keep the default (all float64, backward compatible):

```toml
# [precision] section omitted → mode="full"
```

## Precision modes

| Mode    | Description |
|---------|-------------|
| `full`  | All zones float64 (default, backward compatible) |
| `mixed` | Recommended mixed precision (see zone table below) |

## Precision zones

jQMC divides the computation into 18 **Precision Zones**.  Each zone is
owned by exactly one module and is named for its *purpose* (not its
dtype).  The mapping from zone to dtype is determined entirely by the
chosen mode.

| Zone               | Owning module          | `full` | `mixed`  | risk     | E_L path   |
|--------------------|------------------------|--------|----------|----------|------------|
| `ao_eval`          | `atomic_orbital`       | f64    | **f32**  | low      | core       |
| `ao_grad`          | `atomic_orbital`       | f64    | **f32**  | low      | core       |
| `ao_lap`           | `atomic_orbital`       | f64    | f64      | high§    | core       |
| `mo_eval`          | `molecular_orbital`    | f64    | f64      | high\*   | core       |
| `mo_grad`          | `molecular_orbital`    | f64    | f64      | high     | core       |
| `mo_lap`           | `molecular_orbital`    | f64    | f64      | high     | core       |
| `jastrow_eval`     | `jastrow_factor`       | f64    | **f32**  | low      | core†      |
| `jastrow_grad_lap` | `jastrow_factor`       | f64    | **f32**  | low      | core       |
| `jastrow_ratio`    | `jastrow_factor`       | f64    | **f32**  | low      | indirect‡  |
| `det_eval`         | `determinant`          | f64    | f64      | high     | core       |
| `det_grad_lap`     | `determinant`          | f64    | f64      | high     | core       |
| `det_ratio`        | `determinant`          | f64    | f64      | high     | indirect‡  |
| `coulomb`          | `coulomb_potential`    | f64    | **f32**  | low-med  | core       |
| `wf_eval`          | `wavefunction`         | f64    | f64      | high     | core†      |
| `wf_kinetic`       | `wavefunction`         | f64    | f64      | high     | core       |
| `wf_ratio`         | `wavefunction`         | f64    | f64      | high     | no         |
| `local_energy`     | `hamiltonians`         | f64    | f64      | high     | core       |
| `swct`             | `swct`                 | f64    | f64      | high     | no         |

\* `mo_eval` is high-risk even though the consumed AO values are fp32:
the small `mo_coefficients @ aos` matmul runs in this zone, and its
output feeds the determinant matrix where fp32 round-off is amplified
by `log|det|`.

† `jastrow_eval` and `wf_eval` are on the E_L core path but their
forward values (J and ln|Psi|) do not enter the E_L formula directly
(E_L depends on *derivatives* of ln|Psi|).  Diagnostics show zero E_L
bias when these zones alone are fp32.

‡ `det_ratio` and `jastrow_ratio` affect E_L **indirectly** through the
ECP non-local potential, which evaluates `Psi(R')/Psi(R)` on a
quadrature grid via rank-1 ratio updates.  In non-ECP systems these
zones have no E_L impact.

§ `ao_lap` is kept fp64 in `mixed` mode because the analytic Laplacian
formula contains catastrophic-cancellation terms of the form
`4 Z² r² − 6 Z` and `(safe_div − 2 Z·base)² − safe_div² − 2 Z` that
amplify fp32 round-off into a force bias of order ~1 Ha/bohr in N₂
(diagnostic `bug/fp32/diag_07_ao_grad_vs_lap_split.py`).  The grad
counterpart `ao_grad` has no such cancellation and is safe at fp32
(max|dF| ≈ 5e-6 Ha/bohr).  This is the only zone pair in jQMC where the
grad and Laplacian halves take different dtypes, motivating the split
of the original `ao_grad_lap` zone into separate `ao_grad` / `ao_lap`
zones.

## Workflow integration

When using `jqmc_workflow`, pass the precision mode to any workflow class:

```python
from jqmc_workflow import VMC_Workflow

wf = VMC_Workflow(
    server_machine_name="cluster",
    num_opt_steps=20,
    precision_mode="mixed",
)
```

Per-zone assignments are defined in `_FULL_PRECISION` / `_MIXED_PRECISION`
inside `jqmc/_precision.py` and are not configurable from TOML or workflow
parameters.  Developers who need per-zone control for diagnostics can edit
those dicts directly or use `_set_zone()` after calling `configure()`.

## Design principles

The implementation rests on **three** principles documented at the top of
`jqmc/_precision.py`.  Principle 3 is the most important in practice; almost
every precision bug we have seen is a violation of 3a or 3b.

**Principle 1 — One Precision Zone is owned by exactly one module.**
A zone (e.g. `ao_eval`, `coulomb`) is *defined and consumed* in a single
module.  The mapping zone ↔ owning module is one-to-one.

**Principle 2 — A module may own multiple Precision Zones.**
Different code paths in the same module legitimately need different
precisions (e.g. `ao_eval` vs `ao_grad` vs `ao_lap`, or `det_eval` vs
`det_ratio`).  Each zone is named for its *purpose*, not for its dtype.

**Principle 3 — Cast responsibility lies with the function that does
arithmetic on the value, never with passthrough wrappers.**

* **3a (frozen args).** Function arguments are *frozen*: the parameter name
  must not be rebound for the entire body of the function.  Writing
  `arg = jnp.asarray(arg, dtype=...)` at the top of a function is forbidden
  — it silently coerces the argument for every later use, including
  forwarding to other functions.  When the function consumes `arg` as an
  arithmetic operand, the cast appears **inside the expression**
  (`arg.astype(dtype)`), or — if the cast result is reused — through a
  *new* local variable (e.g. `arg_local = arg.astype(dtype)`).  The
  original `arg` always remains frozen.

* **3b (local cast at the point of arithmetic).** A function casts a value
  to its own zone's dtype **immediately before** consuming it as an
  operand.  Inputs and outputs of the function's arithmetic both live in
  its zone.  For catastrophic cancellation (`r - R`): reconstruct the
  difference in the dtype the values were received in (the
  caller-supplied precision — fp64 in jQMC because the upstream MCMC
  walker state is fp64), then down-cast the result to the function's own
  zone.  The principle is "use the caller-supplied precision," **not**
  "hardcode fp64."

```python
# WRONG (3a violation): rebinding `r_carts` silently forwards a
# fp32-truncated array to compute_AOs even though `ao_eval` is fp64.
def compute_coulomb(r_carts, R_carts):
    dtype_jnp = get_dtype_jnp("coulomb")
    r_carts = jnp.asarray(r_carts, dtype=dtype_jnp)  # <-- forbidden
    ao = compute_AOs(..., r_carts, R_carts)          # downstream sees fp32
    diff = r_carts - R_carts
    ...

# RIGHT: forwarding stays in caller's dtype; reconstruction is in
# caller-supplied precision; downcast happens at the use site.
def compute_coulomb(r_carts, R_carts):
    ao = compute_AOs(..., r_carts, R_carts)          # forward as-is
    dtype_jnp = get_dtype_jnp("coulomb")
    diff = (r_carts - R_carts).astype(dtype_jnp)     # 3b
    ...
```

### No hardcoded dtype literals

Inside any module that owns a selectable-precision zone, **never hardcode**
`jnp.float64` / `np.float64` / `jnp.float32` / `np.float32` for arrays the
module produces or consumes.  Always go through `get_dtype_jnp("<zone>")`
/ `get_dtype_np("<zone>")` so the dtype follows the active mode
automatically.

The exemptions (modules whose data is *always fp64 by construction*,
independent of mode) are:

* `mcmc` / `gfmc` — MCMC and GFMC walker state.
* I/O modules — `structure`, `trexio_wrapper`, `_jqmc_utility`,
  `jqmc_tool`, and the `_load_dataclass_from_hdf5` /
  `_save_dataclass_to_hdf5` helpers in `hamiltonians`.  On-disk numerical
  data (AO exponents/coefficients, nuclear coordinates, geminal
  coefficients, etc.) is always fp64 because fp32 storage would silently
  lose precision that no downstream upcast can recover.
* **Basis-data storage accessors.** `_*_jnp` properties on
  selectable-precision dataclasses whose underlying storage field is
  typed `npt.NDArray[np.float64]` are *lift-only* adapters
  (numpy → `jax.Array`), not arithmetic.  The dtype is fp64 by
  construction (storage is loaded from HDF5/TREXIO/optimizer output);
  the consumer is responsible for casting the lifted array to its own
  zone at the use site (Principle 3b).  Concretely this covers
  `_exponents_jnp` / `_coefficients_jnp` /
  `_normalization_factorial_ratio_prim_jnp` in `atomic_orbital`,
  `_mo_coefficients_jnp` in `molecular_orbital`,
  `_lambda_matrix_jnp` in `determinant`, `_j_matrix_jnp` in
  `jastrow_factor`, and the `ShellPrimMap.from_aos_data` constructor in
  `atomic_orbital`.

## API reference

See {py:mod}`jqmc._precision` for the programmatic API:

* `get_dtype_jnp(zone)` / `get_dtype_np(zone)` — return the JAX / NumPy
  dtype currently assigned to *zone*.
* `get_eps(name, dtype)` — return a dtype-aware numerical-stability
  constant (e.g. `"rcond_svd"`, `"stabilizing_ao"`).
* `configure(mode)` — programmatically switch the active precision mode.
* `get_tolerance(zone, level)` — return `(atol, rtol)` for tests, scaled
  by the zone's current dtype (`level` = `"strict"` or `"loose"`).
* `get_tolerance_min(zones, level)` — return the loosest `(atol, rtol)`
  across the given zones.  Use this when a test compares two paths whose
  combined dtype span crosses multiple zones; the achievable agreement
  is bounded by the weakest zone on the path.
