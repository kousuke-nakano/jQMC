# Example 10: Using jQMC Python API

This example demonstrates how to use jQMC as a Python library, bypassing the command-line interface (CLI). This allows for greater flexibility in constructing workflows, such as integrating with other tools or performing custom analysis loops.

## Overview

The workflow consists of two main scripts:

1.  `run_pyscf.py`: Runs a PySCF calculation (HF/DFT) and exports the results to a TREXIO file (`water_ccecp_ccpvtz.h5`).
2.  `run_jqmc.py`: Loads the TREXIO file, constructs the Hamiltonian, performs VMC optimization, runs VMC sampling, and executes LRDMC calculations with multiple lattice constants to perform energy extrapolation.

## Prerequisites

*   `pyscf`
*   `trexio`
*   `jqmc` (installed in your environment)

## How to Run

### 1. Generate Wavefunction (PySCF -> TREXIO)

First, run the PySCF script to generate the initial wavefunction and save it in TREXIO format.

```bash
python run_pyscf.py
```

This will create `water_ccecp_ccpvtz.h5`.

### 2. Run QMC Workflow (jQMC API)

Next, run the jQMC script.

```bash
python run_jqmc.py
```

This script performs the following steps:

1.  **Initialization**: Reads the TREXIO file and constructs the `Hamiltonian_data` object, initializing Jastrow factors (1-body and 2-body).
2.  **VMC Optimization**: Optimizes the Jastrow parameters using the `MCMC.run_optimize` method.
3.  **VMC Sampling**: Performs a VMC production run with the optimized parameters to get the VMC energy.
4.  **LRDMC Scan**: Runs Lattice Regularized Diffusion Monte Carlo (LRDMC) for a series of lattice constants (`alat = [0.5, 0.4, 0.3]`).
5.  **Extrapolation**: Performs a simple weighted linear regression on the LRDMC energies vs. $a^2$ to extrapolate the energy to $a \to 0$.

## Key Concepts

### Constructing Hamiltonian Data

Instead of using `jqmc_tool.py` from the command line, we use `read_trexio_file` and the `Hamiltonian_data` class directly.

```python
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.hamiltonians import Hamiltonian_data
# ... (Jastrow imports)

(structure_data, aos_data, mos_data, _, geminal_data, coulomb_potential_data) = read_trexio_file(trexio_file, store_tuple=True)
# ... (Initialize Jastrow objects)
hamiltonian_data = Hamiltonian_data(...)
```

### Running MCMC/VMC

The `MCMC` class handles Variational Monte Carlo.

```python
from jqmc.jqmc_mcmc import MCMC

mcmc = MCMC(hamiltonian_data=hamiltonian_data, ...)
mcmc.run_optimize(...) # Optimization
mcmc.run(...)          # Sampling
```

### Running LRDMC

The `GFMC_fixed_num_projection` class handles LRDMC (Green's Function Monte Carlo with fixed population).

```python
from jqmc.jqmc_gfmc import GFMC_fixed_num_projection

lrdmc = GFMC_fixed_num_projection(hamiltonian_data=hamiltonian_data, alat=0.5, ...)
lrdmc.run(...)
```
