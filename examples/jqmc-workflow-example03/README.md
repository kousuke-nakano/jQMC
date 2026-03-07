# jqmc-workflow-example03: Water Energy & Force with J3

Energies and atomic forces for a water molecule (ccECP / cc-pVTZ, Cartesian) computed via **MCMC** and **LRDMC** using a J3 Jastrow factor. Forces are obtained via algorithmic differentiation (AD) in **JAX**.

## Overview

This example automates the full QMC pipeline for a water molecule using `jqmc_workflow`:

```
pySCF (HF) --> WF conversion (JSD) --> VMC optimization --> MCMC (energy + force)
                                                     --> LRDMC (energy + force)
```

The workflow DAG is constructed programmatically in `run_pipelines.py` and executed via `Launcher`, which resolves dependencies and runs MCMC and LRDMC in parallel.

### Target system

| Molecule | Electrons | Basis set       | ECP   |
|----------|-----------|-----------------|-------|
| Water    | 8         | cc-pVTZ (ccECP) | ccECP |

### Ansatz and optimization

- **JSD** (Jastrow-Slater Determinant): J2 (two-body, exponential) + J3 (three-body, AO-small basis). No J1.
- **VMC optimization**: 100 steps with SR optimizer (`adaptive_learning_rate = True`, `delta = 0.35`)
- Determinant part is **not** optimized (`opt_with_projected_MOs = False`)
- 1024 walkers per MPI process

### Production parameters

| Method | Target error | alat | atomic_force | max_time   | max_continuation |
|--------|-------------|------|--------------|------------|------------------|
| MCMC   | 5.0e-4 Ha   | --    | `true`       | 76 000 s   | 2                |
| LRDMC  | 5.0e-4 Ha   | 0.30 | `true`       | 76 000 s   | 2                |

## Prerequisites

- `jqmc`, `jqmc-tool`, and `jqmc_workflow` installed
- `pyscf` with TREXIO support (`pyscf.tools.trexio`)
- Machine configuration in `jqmc_setting_local/` (not included; copy from another workflow example and adjust)

## Quick start

```bash
cd examples/jqmc-workflow-example03
python run_pipelines.py
```

The script performs the following steps automatically:

### Step 0 -- DFT trial wavefunction (pySCF)

A Hartree-Fock calculation is run locally with `pySCF` to produce a TREXIO file. If the file already exists, this step is skipped.

```python
# Generated automatically (embedded in run_pipelines.py)
from pyscf import gto, scf
from pyscf.tools import trexio

mol = gto.Mole()
mol.atom = """
    O    5.00000000   7.14707700   7.65097100
    H    4.06806600   6.94297500   7.56376100
    H    5.38023700   6.89696300   6.80798400
"""
mol.basis = "ccecp-ccpvtz"
mol.unit = "A"
mol.ecp = "ccecp"
mol.cart = True
mol.build()

mf = scf.HF(mol)
mf.max_cycle = 200
mf.kernel()

trexio.to_trexio(mf, "water_ccecp_ccpvtz.h5")
```

### Step 1 -- WF conversion (JSD)

The TREXIO file is converted to `hamiltonian_data.h5` with a JSD Jastrow factor:

```python
WF_Workflow(
    trexio_file="water_ccecp_ccpvtz.h5",
    j1_parameter=None,        # no one-body Jastrow
    j2_parameter=1.0,         # two-body Jastrow (exponential)
    j3_basis_type="ao-small", # three-body Jastrow in AO-small basis
)
```

### Step 2 -- VMC optimization

J2 and J3 parameters are optimized (J1 = None, MOs not optimized):

```python
VMC_Workflow(
    server_machine_name="cluster",
    queue_label="cores-4-mpi-4-gpu-4-omp-1-3h",
    pilot_queue_label="cores-4-mpi-4-gpu-4-omp-1-30m",
    number_of_walkers=1024,
    num_opt_steps=100,
    opt_J1_param=False,
    opt_J2_param=True,
    opt_J3_param=True,
    opt_lambda_param=False,
    opt_with_projected_MOs=False,
    target_error=3.0e-3,
    optimizer_kwargs={
        "method": "sr",
        "delta": 0.350,
        "epsilon": 0.001,
        "adaptive_learning_rate": True,
    },
    max_time=76000,
    max_continuation=2,
)
```

### Step 3 -- MCMC and LRDMC production

MCMC and LRDMC production runs are launched **in parallel** (they both depend only on the VMC-optimized wavefunction). Both compute energies and atomic forces:

```python
MCMC_Workflow(
    server_machine_name="cluster",
    queue_label="cores-4-mpi-4-gpu-4-omp-1-3h",
    number_of_walkers=1024,
    Dt=2.0,
    target_error=5.0e-4,
    atomic_force=True,
    max_time=76000,
    max_continuation=2,
)

LRDMC_Workflow(
    server_machine_name="cluster",
    queue_label="cores-4-mpi-4-gpu-4-omp-1-3h",
    alat=0.30,
    number_of_walkers=1024,
    target_error=5.0e-4,
    atomic_force=True,
    num_gfmc_collect_steps=20,
    max_time=76000,
    max_continuation=2,
)
```

## Directory structure

After running the pipeline:

```
jqmc-workflow-example03/
├── run_pipelines.py          # Main script
├── 01_wf/                    # WF_Workflow: TREXIO --> hamiltonian_data.h5
├── 02_vmc/                   # VMC_Workflow: Jastrow optimization (100 steps)
│   ├── hamiltonian_data_opt_step_1.h5
│   ├── ...
│   └── hamiltonian_data_opt_step_100.h5
├── 03_mcmc/                  # MCMC_Workflow: production sampling + forces
└── 04_lrdmc/                 # LRDMC_Workflow: LRDMC (a=0.30) + forces
```

## Workflow DAG

```
pySCF --> WF --> VMC ─┬─--> MCMC  (energy + force)
                   └─--> LRDMC (energy + force)
```

## Machine configuration

To run on a different cluster, change `SERVER`, `QUEUE_LABEL`, and `PILOT_QUEUE_LABEL` in `run_pipelines.py` and provide the appropriate machine configuration in `jqmc_setting_local/`.

## Output

The script prints a summary table after all calculations complete:

```
  Water Energy & Force Summary  (ccECP / cc-pVTZ, Cartesian)
  LRDMC a = 0.3,  VMC opt steps = 100

|      Pattern |     Energy (Ha) |   Force (Ha/bohr) |
|--------------|-----------------|-------------------|
|        MCMC  |  -X.XXXXX(XX)  |    -X.XXXXX(XX)   |
|       LRDMC  |  -X.XXXXX(XX)  |    -X.XXXXX(XX)   |
```
