# MCMC validation (workflow)

This example validates the `jQMC` MCMC implementation by comparing
Hartree–Fock (HF) and MCMC energies (without Jastrow factors) for
several atomic and molecular systems across different HF types
(RHF / ROHF / UHF) and GTO conventions (Cartesian / Spherical).

The entire pipeline — pySCF → TREXIO → `hamiltonian_data.h5` → MCMC —
is automated via `jqmc-workflow`.

## Prerequisites

- `pySCF` with TREXIO support
- `jqmc` and `jqmc-tool` installed
- `jqmc-workflow` installed
- Remote server configured in `~/.jqmc_setting/` (this example uses `genkai`)

## How to run

```bash
python run_validation_pipline.py
```

The script performs the following steps automatically:

1. **pySCF** (local) — Runs `run_pyscf.py` in each subdirectory to
   produce a TREXIO HDF5 file. Existing files are skipped.
2. **WF_Workflow** (local) — Converts each TREXIO file to
   `hamiltonian_data.h5` using `jqmc-tool trexio convert-to`.
   No Jastrow factors are added.
3. **MCMC_Workflow** (remote) — Submits MCMC production runs on the
   configured server. Includes automatic pilot-run estimation and
   continuation until the target error is achieved.
4. **Summary table** — Prints a Markdown table comparing HF and MCMC
   energies for all systems.

## Pipeline script

```{literalinclude} run_validation_pipline.py
:language: python
```

## System directories

Each subdirectory contains a `run_pyscf.py` script.
For example, `water_pGTOs_RHF/run_pyscf.py`:

```{literalinclude} water_pGTOs_RHF/run_pyscf.py
:language: python
```

## Server configuration

| Parameter           | Value                              |
|---------------------|------------------------------------|
| `server_machine_name` | `genkai`                         |
| `queue_label`         | `cores-480-mpi-120-omp-1-24h`   |
| `pilot_queue_label`   | `cores-120-mpi-120-omp-1-15m`   |
| `target_error`        | 0.001 Ha                         |

## Validation results

The HF and MCMC energies should be consistent within the MCMC error
bar, as no Jastrow factors are applied.

| System  | Spin     |  Type    |   basis        |  ECP    |GTOs           |  HF (Ha)      | MCMC (Ha)     |
|---------|----------|----------|----------------|---------|---------------|---------------|---------------|
| H2O     | 0        | RHF      | ccecp-ccpvqz   |  ccECP  | Cartesian     | -16.94503     | -16.94487(28) |
| H2O     | 0        | RHF      | ccecp-ccpvqz   |  ccECP  | Spherical     | -16.94490     | -16.94482(28) |
| Ar      | 0        | RHF      | ccecp-ccpv5z   |  ccECP  | Cartesian     | -20.77966     | -20.77960(22) |
| Ar      | 0        | RHF      | ccecp-ccpv5z   |  ccECP  | Spherical     | -20.77966     | -20.77960(22) |
| N       | 3        | ROHF     | ccecp-ccpvqz   |  ccECP  | Cartesian     |  -9.63387     |  -9.63371(28) |
| N       | 3        | ROHF     | ccecp-ccpvqz   |  ccECP  | Spherical     |  -9.63387     |  -9.63350(28) |
| N       | 3        | UHF      | ccecp-ccpvqz   |  ccECP  | Cartesian     |  -9.63859     |  -9.63815(27) |
| N       | 3        | UHF      | ccecp-ccpvqz   |  ccECP  | Spherical     |  -9.63856     |  -9.63835(28) |
| O2      | 2        | ROHF     | ccecp-ccpvqz   |  ccECP  | Cartesian     | -31.42286     | -31.42254(19) |
| O2      | 2        | ROHF     | ccecp-ccpvqz   |  ccECP  | Spherical     | -31.42194     | -31.42177(18) |
| O2      | 2        | UHF      | ccecp-ccpvqz   |  ccECP  | Cartesian     | -31.44677     | -31.44668(18) |
| O2      | 2        | UHF      | ccecp-ccpvqz   |  ccECP  | Spherical     | -31.44579     | -31.44589(18) |
