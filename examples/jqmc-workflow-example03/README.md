# jqmc-workflow-example03: Water Energy & Force with J3

This example computes **energies and atomic forces** for a water molecule
(ccECP / cc-pVTZ, Cartesian) using two method combinations:

| Pattern | Jastrow                        | Method | Optimizer (VMC)           |
|---------|--------------------------------|--------|---------------------------|
| 1       | J2=exp, J3=ao-small            | MCMC   | SR (adaptive LR=0.35)    |
| 2       | J2=exp, J3=ao-small            | LRDMC  | SR (adaptive LR=0.35)    |

All patterns share:

- **J1 = None** (no one-body Jastrow)
- **VMC opt steps = 100**, target error = 3.0e-3
- **MCMC / LRDMC target error = 5.0e-4**
- **LRDMC lattice spacing a = 0.30**
- **max_time = 76 000 s** (≈ 21 h)
- **max_continuation = 2** for all stages
- **atomic_force = True** for MCMC and LRDMC

## Pipeline

```
pySCF (local) → WF_Workflow → VMC_Workflow → MCMC_Workflow (energy + force)
                                           → LRDMC_Workflow (energy + force)
```

Both MCMC and LRDMC stages are submitted to a single `Launcher` for
maximum parallelism.

## Reproducing

```bash
cd examples/jqmc-workflow-example03
python run_pipelines.py      # Step 0–2: pySCF → WF → VMC → {MCMC, LRDMC}
```

Requires `jqmc_setting_local/` with machine & queue configuration.
