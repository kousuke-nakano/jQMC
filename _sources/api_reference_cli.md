(api_reference_cli_link)=

# API reference for the command-line interface (jqmc)

## Command-line `jqmc` usage

You can run `jqmc` tasks (`mcmc`, `vmc`, `lrdmc`, and `lrdmc-tau`) from the command line:

```bash
# Serial run
jqmc <input-file> > <output.log>

# MPI parallel run
mpirun -np <N> jqmc <input-file> > <output.log>
```

The input file is a JSON/YAML document whose keys match the parameters listed below.

> **Note**
> Throughout this document, “per MPI process and per walker” means the quantity is counted for each MPI rank and for each walker on that rank. When relevant, the total across all ranks and walkers is indicated explicitly.

---

## Input parameters for the command-line `jqmc`

### `control`

| Key                 |                  Default | Description                                                                                                                |
| ------------------- | -----------------------: | -------------------------------------------------------------------------------------------------------------------------- |
| `job_type`          |             **required** | Select the job: `"mcmc"`, `"vmc"`, `"lrdmc"`, or `"lrdmc-tau"`.                                                            |
| `mcmc_seed`         |                  `34456` | Random seed for MCMC/GFMC chain.                                                                                           |
| `number_of_walkers` |                      `4` | Number of walkers **per MPI process**.                                                                                     |
| `max_time`          |                  `86400` | Wall time limit in seconds.                                                                                                |
| `restart`           |                  `false` | If `true`, restart from a checkpoint.                                                                                      |
| `restart_chk`       |          `"restart.chk"` | Path to the restart checkpoint file (used when `restart=true`).                                                            |
| `hamiltonian_chk`   | `"hamiltonian_data.chk"` | Hamiltonian checkpoint file. When `restart=false`, this file is used to initialize/record Hamiltonian data as appropriate. |
| `verbosity`         |                  `"low"` | Verbosity level: `"low"`, `"high"`, `"devel"`, `"mpi-low"`, `"mpi-high"`, `"mpi-devel"`.                                   |

---

### `mcmc` (i.e., a single-shot MCMC without WF optimization)

| Key                        |      Default | Description                                                                                                                                                                                                                                   |
| -------------------------- | -----------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_mcmc_steps`           | **required** | Number of **measurement** steps per MPI process and per walker. Local energy and other observables are measured `num_mcmc_steps` times in total per (rank × walker). The global total equals `num_mcmc_steps × mpi_size × number_of_walkers`. |
| `num_mcmc_per_measurement` |         `40` | MCMC updates between successive measurements. Observables are recorded every `num_mcmc_per_measurement` steps.                                                                                                                                |
| `num_mcmc_warmup_steps`    |          `0` | Number of **warm-up** measurement steps to be discarded.                                                                                                                                                                                      |
| `num_mcmc_bin_blocks`      |          `1` | Number of binning blocks per MPI process and per walker (total binned blocks = `num_mcmc_bin_blocks × mpi_size × number_of_walkers`).                                                                                                         |
| `Dt`                       |        `2.0` | MCMC step size (Bohr).                                                                                                                                                                                                                        |
| `epsilon_AS`               |        `0.0` | ε parameter for the Attaccalite–Sorella regularization.                                                                                                                                                                                       |
| `atomic_force`             |      `false` | If `true`, compute atomic forces.                                                                                                                                                                                                             |

---

### `vmc` (stochastic reconfiguration / natural-gradient optimization)

| Key                        |      Default | Description                                                                                                          |
| -------------------------- | -----------: | -------------------------------------------------------------------------------------------------------------------- |
| `num_mcmc_steps`           | **required** | Same definition as in `vmc`.                                                                                         |
| `num_mcmc_per_measurement` |         `40` | Same as `vmc`.                                                                                                       |
| `num_mcmc_warmup_steps`    |          `0` | Warm-up steps to discard.                                                                                            |
| `num_mcmc_bin_blocks`      |          `1` | Binning blocks per MPI process and per walker.                                                                       |
| `Dt`                       |        `2.0` | MCMC step size (Bohr).                                                                                               |
| `epsilon_AS`               |        `0.0` | ε for Attaccalite–Sorella regularization.                                                                            |
| `num_opt_steps`            | **required** | Number of optimization iterations.                                                                                   |
| `wf_dump_freq`             |          `1` | Write wavefunction/Hamiltonian checkpoint every this many optimization steps.                                        |
| `optimizer_kwargs`         | `{ "method": "sr", "delta": 0.01, "epsilon": 0.001, "cg_flag": true, "cg_max_iter": 10000, "cg_tol": 1e-4 }` | Optimizer configuration. Set `method` to `"sr"` for stochastic reconfiguration or to an optax optimizer name (e.g., `"adam"`). `delta`/`epsilon` control SR step size and regularization; `cg_*` entries tune the SR conjugate-gradient solver. Any additional keys are forwarded to optax when `method` ≠ `"sr"`. |
| `opt_J1_param`             |      `false` | Optimize J1 parameters.                                                                                              |
| `opt_J2_param`             |       `true` | Optimize J2 parameters.                                                                                              |
| `opt_J3_param`             |       `true` | Optimize J3 parameters.                                                                                              |
| `opt_lambda_param`         |      `false` | Optimize geminal (λ) parameters.                                                                                     |
| `num_param_opt`            |          `0` | Number of parameters to optimize, chosen in descending order of \|f\| / std(f). If `0`, optimize **all** parameters. |

---

### `lrdmc`

| Key                        |      Default | Description                                                                                                                            |
| -------------------------- | -----------: | -------------------------------------------------------------------------------------------------------------------------------------- |
| `num_mcmc_steps`           | **required** | Number of **measurement** steps per MPI process and per walker during LRDMC.                                                           |
| `num_mcmc_per_measurement` |         `40` | Number of GFMC projections between measurements (observables recorded after each block of projections).                                |
| `alat`                     |       `0.30` | Lattice discretization parameter (grid spacing). The lattice spacing is `alat × a₀`, where `a₀` is the Bohr radius.                    |
| `non_local_move`           |    `"tmove"` | Treatment of non-local ECP terms: `"tmove"` (T-move) or `"dltmove"` (determinant-locality + T-move).                                   |
| `num_gfmc_warmup_steps`    |          `0` | Number of warm-up measurement steps to discard.                                                                                        |
| `num_gfmc_bin_blocks`      |          `1` | Number of binning blocks for GFMC. **Total binned blocks = `num_gfmc_bin_blocks`** (not multiplied by `mpi_size × number_of_walkers`). |
| `num_gfmc_collect_steps`   |          `0` | Number of pre-binning measurements used to collect/accumulate weights.                                                                 |
| `E_scf`                    |        `0.0` | Initial total-energy guess used to set the initial GFMC energy shift.                                                                  |
| `atomic_force`             |      `false` | If `true`, compute atomic forces.                                                                                                      |

---

### `lrdmc-tau`

| Key                      |      Default | Description                                                            |
| ------------------------ | -----------: | ---------------------------------------------------------------------- |
| `num_mcmc_steps`         | **required** | Number of **measurement** steps per MPI process and per walker.        |
| `tau`                    |       `0.10` | Imaginary-time step size between projections.                          |
| `alat`                   |       `0.30` | Lattice discretization parameter; lattice spacing `alat × a₀`.         |
| `non_local_move`         |    `"tmove"` | Non-local ECP treatment: `"tmove"` or `"dltmove"`.                     |
| `num_gfmc_warmup_steps`  |          `0` | Warm-up steps to discard.                                              |
| `num_gfmc_bin_blocks`    |          `1` | Binning blocks for GFMC (total binned blocks = `num_gfmc_bin_blocks`). |
| `num_gfmc_collect_steps` |          `0` | Pre-binning measurement count for weight collection.                   |

---

## Minimal schema example

```json
{
  "control": {
    "job_type": "mcmc",
    "number_of_walkers": 4,
    "verbosity": "low"
  },
  "mcmc": {
    "num_mcmc_steps": 1000,
    "num_mcmc_per_measurement": 40,
    "Dt": 2.0
  }
}
```

> **Tips**
>
> * Set `num_mcmc_warmup_steps` to a nonzero value to ensure equilibrated sampling before measurements.
> * For reproducibility across MPI runs, keep `mcmc_seed` fixed and the MPI topology unchanged.
> * Start LRDMC with a reasonable `E_scf` to reduce initial transients in the population control.
