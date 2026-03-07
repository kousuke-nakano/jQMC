# jqmc-workflow-example99: Local Integration Test

**Temporary local test** for `jqmc_workflow`.
Assumes `~/.jqmc_setting/` (= `jqmc_setting_local`) is already configured with a working `localhost` entry.

## What this tests

1. **Full pipeline**: pySCF --> WF --> VMC (5 opt steps) --> MCMC for H$_2$ at 2 bond lengths (R = 0.74, 1.00 $\text{\AA}$)
2. **Phase-1 diagnostic APIs** (newly implemented):
   - `get_all_workflow_statuses()` -- recursive `workflow_state.toml` collection
   - `get_workflow_summary()` -- per-directory status + result + jobs summary
   - `parse_vmc_output()` -- per-step VMC data (energy, S/N, walker weight, acceptance ratio, net time)
   - `parse_mcmc_output()` -- MCMC diagnostic data
   - `parse_input_params()` -- TOML parameter extraction

## Prerequisites

- `jqmc`, `jqmc-tool`, `jqmc_workflow` installed
- `pyscf` with TREXIO support
- `~/.jqmc_setting/` configured with `localhost` (`queuing: false`)

## Usage

```bash
cd examples/jqmc-workflow-example99
python run_test.py
```

## Configuration

Everything is intentionally minimal for fast local execution:

| Parameter         | Value  | Notes                     |
|-------------------|--------|---------------------------|
| `num_opt_steps`   | 5      | VMC optimization steps    |
| `num_walkers`     | 16     | Minimal walker count      |
| `target_error`    | 0.1 Ha | Very loose -- just finish  |
| `R_VALUES`        | 2      | 0.74, 1.00 $\text{\AA}$              |
| `server`          | localhost | No remote needed       |

## Output

After the pipeline, the script prints diagnostic output from the Phase-1 APIs:

```
--- get_all_workflow_statuses ---
  R_0.74/01_wf            label=wf-0.74         type=wf        status=completed
  R_0.74/02_vmc            label=vmc-0.74        type=vmc       status=completed
  R_0.74/03_mcmc           label=mcmc-0.74       type=mcmc      status=completed
  ...

--- parse_vmc_output ---
  R=0.74:  5 optimization steps parsed
    last step 5: E=-1.1709, snr=..., walker_wt=..., accept=..., time=...s
  ...

--- parse_mcmc_output ---
  R=0.74:  energy=-1.1745, error=0.0012, accept=0.51, ...
  ...
```
