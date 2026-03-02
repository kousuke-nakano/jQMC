# jqmc-workflow-example01

Potential energy surface (PES) of the hydrogen molecule (H₂) computed using **jqmc-workflows**. All-electron calculations with Cartesian GTOs (cc-pVTZ basis). Atomic forces at each bond length are obtained via algorithmic differentiation (AD) in **JAX**.

## Overview

This example automates the full QMC pipeline for 20 bond lengths using `jqmc_workflow`:

```
pySCF (DFT) → WF conversion (JSD) → VMC optimization → MCMC + LRDMC (a = 0.2)
```

The workflow DAG is constructed programmatically in `run_pes_pipeline.py` and executed via `Launcher`, which resolves dependencies and runs independent R values in parallel.

### Ansatz and optimization

- **JSD** (Jastrow–Slater Determinant): J1 (one-body) + J2 (two-body) + J3 (three-body, MO basis)
- **VMC optimization**: 20 steps with `opt_with_projected_MOs = true` (MOs are optimized alongside Jastrow parameters)
- **MCMC**: production sampling with atomic forces
- **LRDMC**: lattice-regularized diffusion Monte Carlo at `a = 0.2` with atomic forces

### Bond lengths

```
R (Å) = 0.35  0.40  0.45  0.50  0.55  0.60  0.65  0.70
         0.74  0.80  0.85  0.90  0.95  1.00  1.05  1.10
         1.15  1.20  1.30  1.40
```

## Prerequisites

- `jqmc`, `jqmc-tool`, and `jqmc_workflow` installed
- `pyscf` with TREXIO support (`pyscf.tools.trexio`)
- Machine configuration in `jqmc_setting_local/`

## Quick start

```bash
cd examples/jqmc-workflow-example01
python run_pes_pipeline.py
```

The script performs the following steps automatically:

### Step 1 — DFT trial wavefunctions (pySCF)

For each bond length R, a DFT (LDA) calculation is run locally with `pySCF` to produce a TREXIO file. The pySCF script is generated and executed in `R_{R}/00_pyscf/`.

```python
# Generated automatically for each R
from pyscf import gto, scf
from pyscf.tools import trexio

R = 0.74  # angstrom
filename = f"H2_R_{R:.2f}.h5"

mol = gto.Mole()
mol.atom = f"""
H    0.000000000   0.000000000  {-R / 2}
H    0.000000000   0.000000000  {+R / 2}
"""
mol.basis = "ccpvtz"
mol.unit = "A"
mol.cart = True
mol.build()

mf = scf.KS(mol).density_fit()
mf.xc = "LDA_X,LDA_C_PZ"
mf.kernel()

trexio.to_trexio(mf, filename)
```

### Step 2 — WF conversion (JSD)

The TREXIO file is converted to `hamiltonian_data.h5` with a JSD Jastrow factor:

```python
WF_Workflow(
    trexio_file="H2_R_0.74.h5",
    j1_parameter=1.0,     # one-body Jastrow
    j2_parameter=1.0,     # two-body Jastrow
    j3_basis_type="mo",   # three-body Jastrow in MO basis
)
```

### Step 3 — VMC optimization

Jastrow parameters (J1, J2, J3) and molecular orbitals are optimized simultaneously:

```python
VMC_Workflow(
    server_machine_name="localhost",
    queue_label="local",
    num_opt_steps=20,
    opt_J1_param=True,
    opt_J2_param=True,
    opt_J3_param=True,
    opt_with_projected_MOs=True,
    target_error=0.001,
)
```

### Step 4 — MCMC and LRDMC production

MCMC and LRDMC production runs are launched **in parallel** (they both depend only on the VMC-optimized wavefunction):

```python
MCMC_Workflow(
    server_machine_name="localhost",
    queue_label="local",
    target_error=0.001,
    atomic_force=True,
)

LRDMC_Workflow(
    server_machine_name="localhost",
    queue_label="local",
    alat=0.2,
    target_error=0.001,
    atomic_force=True,
)
```

## Directory structure

After running the pipeline, each bond length has the following structure:

```
R_0.74/
├── 00_pyscf/          # pySCF DFT calculation
│   ├── run_pyscf.py
│   ├── H2_R_0.74.h5   # TREXIO file
│   └── H2_R_0.74.out  # pySCF output
├── 01_wf/             # WF_Workflow: TREXIO → hamiltonian_data.h5
├── 02_vmc/            # VMC_Workflow: Jastrow + MO optimization
│   ├── hamiltonian_data_opt_step_1.h5
│   ├── ...
│   └── hamiltonian_data_opt_step_20.h5
├── 03_mcmc/           # MCMC_Workflow: production sampling + forces
└── 04_lrdmc/          # LRDMC_Workflow: LRDMC (a=0.2) + forces
```

## Workflow DAG

For each R, the dependency graph is:

```
pySCF → WF → VMC ─┬─→ MCMC
                   └─→ LRDMC (a=0.2)
```

All 20 R values are independent and execute in parallel via `Launcher`.

## Machine configuration

This example uses local execution (`jqmc_setting_local/`):

```yaml
# machine_data.yaml
localhost:
  machine_type: local
  queuing: false
```

```toml
# localhost/queue_data.toml
[local]
    submit_template = 'submit_mpi.sh'
    num_cores = 4
    omp_num_threads = 1
```

To run on a remote cluster, change `SERVER` and `QUEUE_LABEL` in `run_pes_pipeline.py` and provide the appropriate machine configuration.

## Output

The script prints a summary table after all calculations complete:

```
| R (Å)  |  E_HF (Ha)    |  E_MCMC (Ha)    |  F_MCMC (Ha/Å) |  E_LRDMC (Ha)   | F_LRDMC (Ha/Å)  |
|--------|---------------|-----------------|-----------------|-----------------|------------------|
|   0.35 |     -0.XXXXXX |    -X.XXXXX(XX) |    -X.XXXXX(XX) |    -X.XXXXX(XX) |     -X.XXXXX(XX) |
|   ...  |      ...      |       ...       |       ...       |       ...       |        ...       |
|   1.40 |     -0.XXXXXX |    -X.XXXXX(XX) |    -X.XXXXX(XX) |    -X.XXXXX(XX) |     -X.XXXXX(XX) |
```

## References

[^2010SORjcp]: S. Sorella and L. Capriotti, *J. Chem. Phys.* **133**, 234111 (2010). Algorithmic differentiation in ab initio QMC.
