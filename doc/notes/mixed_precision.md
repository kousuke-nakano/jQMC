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

jQMC divides the computation into 10 **Precision Zones**.  The mapping
from zone to dtype is determined entirely by the chosen mode:

| Zone           | Components                        | `full` | `mixed` | float32 risk |
|----------------|-----------------------------------|--------|---------|--------------|
| `orb_eval`     | AO/MO forward evaluation          | f64    | **f32** | low          |
| `jastrow`      | Jastrow factor (J1/J2/J3)         | f64    | **f32** | low          |
| `geminal`      | Geminal matrix elements            | f64    | f64     | high         |
| `determinant`  | log-det, SVD, AS regularization    | f64    | f64     | high         |
| `coulomb`      | Coulomb + ECP potential            | f64    | **f32** | low-medium   |
| `kinetic`      | Kinetic energy + AO/MO derivatives | f64    | f64     | high         |
| `mcmc`         | MCMC sampling                      | f64    | f64     | high         |
| `gfmc`         | GFMC propagation                   | f64    | f64     | high         |
| `optimization` | SR matrix, parameter updates       | f64    | f64     | high         |
| `io`           | I/O, structure data                | f64    | f64     | low-medium   |

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

1. **Explicit dtype declaration** — Every function declares its Precision Zone
   and specifies dtype for all arrays.  No reliance on JAX implicit promotion.

2. **Zone boundaries** — When results cross zone boundaries (e.g. Jastrow
   float32 → determinant float64), explicit casts ensure the higher-precision
   zone receives correctly typed inputs.

3. **Backward compatibility** — Default mode is `"full"` (all float64).
   Existing input files work without modification.

## API reference

See {py:mod}`jqmc._precision` for the programmatic API (`get_dtype`,
`configure`, `get_tolerance`, etc.).
