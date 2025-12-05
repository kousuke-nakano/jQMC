# Example 11: Using jQMC Python API with MPI

This example demonstrates how to use jQMC as a Python library with MPI parallelization. It is similar to Example 10 but designed to run on multiple cores/nodes.

## Overview

The workflow consists of two main scripts:

1.  `run_pyscf.py`: Runs a PySCF calculation (HF/DFT) and exports the results to a TREXIO file (`water_ccecp_ccpvtz.h5`). This is a serial script.
2.  `run_jqmc_mpi.py`: The MPI-enabled jQMC script. It loads the TREXIO file, constructs the Hamiltonian, performs VMC optimization, runs VMC sampling, and executes LRDMC calculations with multiple lattice constants.

## Prerequisites

*   `pyscf`
*   `trexio`
*   `jqmc` (installed in your environment)
*   `mpi4py`
*   `mpi4jax`
*   MPI implementation (e.g., OpenMPI, MPICH)

## How to Run

### 1. Generate Wavefunction (PySCF -> TREXIO)

First, run the PySCF script to generate the initial wavefunction. This step is serial.

```bash
python run_pyscf.py
```

This will create `water_ccecp_ccpvtz.h5`.

### 2. Run QMC Workflow (jQMC API with MPI)

Next, run the jQMC script using `mpirun` (or `mpiexec`).

```bash
mpirun -np 4 python run_jqmc_mpi.py
```

Replace `4` with the number of MPI processes you wish to use.

## Key Concepts

### MPI Initialization

The script initializes MPI using `mpi4py`.

```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
```

### Logging and Output

Only the root process (rank 0) should print logs and save files to avoid race conditions and cluttered output.

```python
if rank == 0:
    logger.info(...)
    hamiltonian_data.save_to_hdf5(...)
```

### Parallel Execution

*   **Hamiltonian Construction**: All ranks read the TREXIO file and construct the `Hamiltonian_data` object independently. This ensures every rank has the necessary data structures.
*   **MCMC/GFMC**: The `MCMC` and `GFMC` classes in jQMC are designed to handle MPI automatically. When you instantiate them and call `run` or `run_optimize`, they coordinate across ranks.
    *   Walkers are distributed across MPI processes.
    *   `run_optimize` synchronizes parameters (gradients are summed across ranks).
    *   `get_E` performs an MPI reduction to return the global energy mean and standard deviation to all ranks.

```python
# All ranks execute this
mcmc.run_optimize(...)
mcmc.run(...)
E_mean, E_std, _, _ = mcmc.get_E(...) # Returns global stats
```

### Extrapolation

The final extrapolation step is performed only by rank 0, using the gathered results.
