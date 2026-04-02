(workflows_tags)=
# Autonomous Pipelines (jqmc_workflow)

## Overview

The **jqmc_workflow** package provides an autonomous pipeline engine for
**jQMC** calculations.
Users define a *pipeline* — a directed acyclic graph (DAG) of workflow steps —
and the engine takes care of:

- **Input generation** — TOML input files are created automatically from
  explicit parameter values (with sensible defaults).
- **Data transfer** — Files are uploaded to (and downloaded from) remote
  supercomputers via SSH/SFTP using [Paramiko](https://www.paramiko.org/).
- **Job submission, monitoring, and collection** — Jobs are submitted through
  the site's scheduler (PBS / Slurm / local `bash`), polled until completion,
  and output files are fetched back.
- **Dependency resolution** — The DAG-based `Launcher` identifies which
  workflow steps are *ready* (all dependencies satisfied) and executes them
  **in parallel** using Python `asyncio` tasks.
- **Target error-bar estimation** — When a `target_error` (Ha) is specified,
  a small *pilot* run is executed first; the statistical error is used to
  estimate the number of production steps required, together with the
  estimated wall time.

A single Python script can therefore express a full QMC pipeline — from
wavefunction preparation (TREXIO → `hamiltonian_data.h5`) through VMC
optimization, MCMC production sampling, and LRDMC extrapolation — and run it
end-to-end with automatic restarts across interruptions.


## Architecture

```text
run_pipeline.py
      │
      ▼
  ┌────────┐
  │Launcher│   DAG executor (asyncio)
  └──┬─────┘
     │  creates asyncio.Task per ready node
     │
     ├──► Container("vmc")
     │         └─► VMC_Workflow.configure() → .run()
     │
     ├──► Container("mcmc-prod")   ← runs in parallel
     │         └─► MCMC_Workflow.configure() → .run()
     │
     └──► Container("lrdmc-ext")   ← runs in parallel
               └─► LRDMC_Ext_Workflow.configure() → .run()
                       ├─► LRDMC_Workflow (alat=0.50)  ┐
                       ├─► LRDMC_Workflow (alat=0.40)  ├ parallel
                       └─► LRDMC_Workflow (alat=0.25)  ┘
```

### Key components

| Class / Module | Role |
|---|---|
| `Launcher` | Executes a DAG of `Container` nodes in topological order, launching independent nodes in parallel. Provides `get_session_state()`, `get_current_job()`, `get_job_history()` for session introspection. |
| `Container` | Wraps any `Workflow` subclass in a dedicated project directory with state tracking (`workflow_state.toml`). |
| `FileFrom` / `ValueFrom` | Declare inter-workflow dependencies (files or computed values). |
| `WF_Workflow` | Converts a TREXIO file to `hamiltonian_data.h5`. |
| `VMC_Workflow` | Jastrow / orbital optimization (`job_type=vmc`). |
| `MCMC_Workflow` | Production energy sampling (`job_type=mcmc`). |
| `LRDMC_Workflow` | Lattice-regularized diffusion Monte Carlo for a single $a$ value. |
| `LRDMC_Ext_Workflow` | Runs multiple `LRDMC_Workflow` instances at different lattice spacings and performs $a^2 \to 0$ extrapolation. |
| `ScientificPhase` | Enum defining the scientific phases of a workflow session (INIT → SCF → WF_BUILD → VMC → MCMC → LRDMC → COMPLETED). See [Phase management](#phase-management). |
| `WorkflowStatus` / `JobStatus` | Enums for workflow-level and per-job status values. See [Status enums](#status-enums). |


### Target error-bar estimation

When `target_error` is set, each workflow (MCMC and LRDMC) operates in
a pilot + production cycle.  A separate queue may be used for the pilot
via `pilot_queue_label` (defaults to `queue_label`).

#### MCMC / VMC

1. **Pilot** — A short run of `pilot_steps` steps.
2. **Production** — Step count estimated from the pilot error bar.

#### LRDMC

LRDMC has an additional calibration stage to automatically determine
`num_projection_per_measurement` (GFMC projections per measurement) from
a `target_survived_walkers_ratio` (default 0.97):

1. **Calibration** (`_pilot_a/_pilot1` – `_pilot_a/_pilot3`, parallel) —
   Three short LRDMC runs with
   `num_projection_per_measurement = Ne × k × (0.3/alat)²` (k=2,4,6;
   $N_e$ is the total electron count).  A quadratic is fit to the
   observed survived-walkers ratio and the optimal
   `num_projection_per_measurement` is determined.
2. **Error-bar pilot** (`_pilot_b`) — A run with the calibrated
   `num_projection_per_measurement`; its error bar estimates the production
   step count.
3. **Production** (`_1`, `_2`, …) — Start from scratch, accumulate
   statistics until `target_error` is achieved.

If `num_projection_per_measurement` is given explicitly, the calibration
stage is skipped and only the error-bar pilot is executed.

For `LRDMC_Ext_Workflow` (multi-alat extrapolation), every alat value
independently runs its own calibration, error-bar pilot, and production
in parallel.  There is no inter-alat interaction until the final
extrapolation step.

#### VMC convergence checks

After all production runs complete, the VMC workflow checks whether
the optimization has converged.  Two independent criteria are
available; when both are active, both must pass for convergence.

##### Signal-to-noise (S/N) check

Enabled when `target_snr` is set (not `None`).  The workflow
averages the signal-to-noise ratio (S/N = max(|f| / |std f|)) over
the last `snr_avg_window` optimization steps (default 5).  If there
are fewer steps than the window size, all available values are used.

The convergence criterion is:

$$
\overline{\text{S/N}}_{\text{last } W} \le \text{target\_snr}
$$

where $W$ is `snr_avg_window`.

##### Energy-slope check

Enabled when `energy_slope_sigma_threshold` is set (not `None`).
A weighted linear regression is fitted to the last
`energy_slope_window_size` optimisation steps (default 5):

$$
E_k = a + b \cdot k + \varepsilon_k, \quad w_k = 1/\sigma_k^2
$$

The optimisation is considered converged (plateau) when the slope
$b$ is not significantly negative:

$$
b \ge -\sigma_b \times \text{energy\_slope\_sigma\_threshold}
$$

If instead $b < -\sigma_b \times \text{threshold}$, the energy is
still decreasing and optimisation has not yet plateaued.

##### Combined verdict

| `target_snr` | `energy_slope_sigma_threshold` | Behaviour |
|:---:|:---:|:---|
| `None` | `None` | No convergence check; always succeeds |
| set | `None` | S/N check only |
| `None` | set | Energy-slope check only |
| set | set | Both must pass |

All numerical values (averaged S/N, slope, slope std) are recorded
in `output_values` for downstream inspection.

In fixed-step mode (`num_mcmc_steps` is set), the convergence checks
are not performed.

#### Step estimation formula

The required number of production steps is estimated via

$$
N_{\text{eff,prod}} = N_{\text{eff,pilot}}
  \times \left(\frac{\sigma_{\text{pilot}}}{\sigma_{\text{target}}}\right)^2
  \times \frac{W_{\text{pilot}}}{W_{\text{prod}}}
$$

where $N_{\text{eff}} = N_{\text{total}} - N_{\text{warmup}}$ is the
number of steps that actually contribute to statistics (warmup steps
are discarded during post-processing via `-w`), and
$W = \text{walkers\_per\_MPI} \times \text{num\_MPI}$ is the
total number of walkers.  The ratio
$W_{\text{pilot}} / W_{\text{prod}}$ (the *walker ratio*) accounts
for the pilot queue using fewer (or more) MPI processes than the
production queue.  The number of MPI processes is read from
`queue_data.toml` (`num_cores`, or `mpi_per_node × nodes` as
fallback).

The total production steps are then
$N_{\text{prod}} = N_{\text{eff,prod}} + N_{\text{warmup}}$.

The estimated production wall time is based on the **net** computation
time parsed from the pilot output (``Net GFMC time`` for LRDMC,
``Net total time for MCMC`` for MCMC/VMC).  Only this net portion
scales with the step count; overhead (JIT compilation, file I/O,
queue wait) is treated as a constant:

$$
T_{\text{prod}} \approx (T_{\text{wall,pilot}} - T_{\text{net,pilot}})
  + T_{\text{net,pilot}} \times \frac{N_{\text{prod}}}{N_{\text{pilot}}}
$$

```text
-- LRDMC Step Estimation Summary (a=0.25) --------
  pilot steps       = 100
  warmup steps      = 10
  pilot error       = 0.00393172 Ha
  target error      = 0.001 Ha
  nmpm              = 32
  pilot MPI procs   = 48
  prod. MPI procs   = 480
  walker ratio      = 0.1
  estimated steps   = 155
  pilot wall time   = 5m 12s
  pilot net time    = 2m 42s
  est. prod. time   = 6m 43s
--------------------------------------------------
```

The first production run starts from scratch (no restart from the
pilot checkpoint, which lives in the ``_pilot_b/`` subdirectory).
After each production run, the error bar is re-evaluated.
If it exceeds the target, additional steps are estimated and a
continuation run is launched automatically, up to `max_continuation`
times.


### Restart behavior

Every job is recorded in `workflow_state.toml` with a lifecycle:

```text
submitted  →  completed  →  fetched
```

On restart, the engine checks each job's status:

- **`fetched`** — Input generation *and* submission are both skipped.
- **`submitted`** / **`completed`** — Input is not regenerated; the job
  is resumed (polled or fetched).
- **No record** — A fresh input file is generated and the job is submitted.

This means a pipeline can be interrupted at any point (Ctrl-C, node
failure, wall-time limit) and simply re-run; it will pick up exactly
where it left off.

When a job leaves the scheduler queue, the engine automatically collects
**job accounting** data (if `jobacct` is configured in `machine_data.yaml`)
before fetching output files.  The accounting command and output file
path are stored in the corresponding `[[jobs]]` record.  See
[Job accounting](#job-accounting).


### Phase management

A workflow session progresses through a sequence of **scientific phases**
defined by the `ScientificPhase` enum (module `_phase`):

```text
INIT → SCF → WF_BUILD → VMC_PILOT → VMC → MCMC_PILOT → MCMC
                                                         ↓
                        COMPLETED ← LRDMC_FIT ← LRDMC ← LRDMC_PILOT
```

Not every pipeline uses every phase (e.g. a VMC-only pipeline skips
LRDMC phases).  The allowed transitions are defined in
`PHASE_TRANSITIONS` — for example, from `VMC` you may advance to
`MCMC_PILOT`, `MCMC`, `LRDMC_PILOT`, `LRDMC`, or `COMPLETED`.

Each phase has a list of **allowed actions** (`PHASE_ALLOWED_ACTIONS`)
that further depends on the current `WorkflowStatus`:

- When `status == RUNNING`, configuration actions (`configure_*`) are
  filtered out.
- When `status == FAILED`, only recovery actions (`recover_*`) and
  `rollback_phase` are available.

A set of **always-allowed actions** (`advance_phase`, `rollback_phase`,
`close_session`, `register_artifact`, `mark_unhealthy`) is appended
regardless of phase/status.

The `require_action()` function enforces these rules at the boundary
between an MCP tool call and a workflow method — if the action is not
permitted, a `ValueError` is raised immediately.


### Status enums

Workflow status and job status are represented by `str`-based enums
(`WorkflowStatus`, `JobStatus`) so they remain human-readable in
`workflow_state.toml` and can be compared directly with strings.

**WorkflowStatus** values:

| Value | Meaning |
|---|---|
| `pending` | Not yet started |
| `copying` | Input files being transferred |
| `submitted` | Job submitted to scheduler |
| `running` | Execution in progress |
| `completed` | Finished successfully |
| `failed` | Terminated with an error |
| `cancelled` | Manually cancelled |

**JobStatus** values (per-job, stored in `[[jobs]]` of
`workflow_state.toml`):

| Value | Meaning |
|---|---|
| `submitted` | Job submitted |
| `completed` | Job finished (output not yet fetched) |
| `fetched` | Output files retrieved |
| `failed` | Job failed |

Each `[[jobs]]` record contains:

| Field | Description |
|---|---|
| `input_file` | Basename of the generated TOML input file |
| `output_file` | Basename of the stdout capture file |
| `job_id` | Scheduler job ID (or `"local"` for local runs) |
| `server_machine` | Machine name |
| `status` | One of the `JobStatus` values above |
| `submitted_at` | ISO 8601 timestamp |
| `step` | Step index (0 = pilot, 1, 2, … = production) |
| `run_id` | Short hex identifier for the job |
| `completed_at` | ISO 8601 timestamp (set on completion) |
| `fetched_at` | ISO 8601 timestamp (set on fetch) |
| `job_stdout` | Scheduler stdout path (queuing systems only) |
| `job_stderr` | Scheduler stderr path (queuing systems only) |
| `job_acct_command` | Accounting command executed (queuing systems only) |
| `job_acct_file` | Path to raw accounting output file (queuing systems only) |


### Artifact registry

The engine records file lineage in the `[[artifacts]]` array of
`workflow_state.toml`.  Each entry tracks:

| Field | Description |
|---|---|
| `filename` | Basename of the artifact file |
| `produced_by_job` | Input file that produced this artifact |
| `produced_at` | ISO 8601 timestamp |
| `artifact_type` | `"file"` (default) |
| `upstream` | Optional list of `{label, file}` dicts tracing the dependency chain |

Use `register_artifact()` to add an entry, `get_artifact_lineage()` to
look up a single file, and `get_artifact_registry()` to list all
artifacts.


### Error recording

When a workflow fails, the `[error]` section of `workflow_state.toml`
is populated via `set_error()`:

```toml
[error]
message = "Job killed after 86400s"
exception_type = "RuntimeError"
traceback = "..."
```

The engine records raw data only — failure classification and recovery
strategy are responsibilities of external tooling (e.g. an MCP agent).


### Job accounting

For queuing systems (PBS, Slurm, Fujitsu TCS, etc.), the engine can
collect scheduler accounting data after a job leaves the queue.  This
is configured via the optional `jobacct` field in `machine_data.yaml`
(see [machine_data.yaml parameters](#parameters)).

The engine executes `{jobacct} {job_id}` and writes the raw output to
a separate file `job_accounting_{job_id}.txt`.  The accounting command
and file path are stored per-job in the `[[jobs]]` record:

```toml
[[jobs]]
input_file = "vmc-H2-0.74_1_aebf13bd.toml"
output_file = "vmc-H2-0.74_1_aebf13bd.out"
job_id = "12345"
server_machine = "my-cluster"
status = "fetched"
step = 1
run_id = "aebf13bd"
job_stdout = "job_vmc-H2-0.74.o"
job_stderr = "job_vmc-H2-0.74.e"
job_acct_command = "sacct -j 12345 --format=State,ExitCode,MaxRSS,Elapsed -P"
job_acct_file = "job_accounting_12345.txt"
```

No parsing or interpretation is performed — that responsibility belongs
to external tooling.  If `jobacct` is not configured, the
`job_acct_command` and `job_acct_file` fields are simply absent.


### Session state queries

The `Launcher` provides three methods for programmatic introspection
(useful for MCP adapters and monitoring tools):

| Method | Returns |
|---|---|
| `get_session_state()` | Dict with per-workflow summaries, dependency graph, and progress counters (completed / failed / running / pending / total). |
| `get_current_job()` | The first workflow with status `running` or `submitted`, or `None`. |
| `get_job_history()` | Flat, chronologically-sorted list of all jobs across all workflows. |


### Machine catalog

Two helper functions are available for machine discovery:

| Function | Description |
|---|---|
| `list_machines()` | Returns a list of dicts summarising all machines defined in `machine_data.yaml` (name, machine_type, queuing, ssh_host, workspace_root). |
| `probe_environment(machine_name)` | Tests connectivity to the named machine (SSH for remote, always reachable for local). Returns `{"reachable": True/False, ...}`. |


## Configuration

Configuration files are managed in the following directory hierarchy.
The engine looks for a project-local override first, then falls back to
the user-global directory:

1. `./jqmc_setting_local/` — project-local override (if it exists in CWD)
2. `~/.jqmc_setting/` — user-global default

On the very first run, if neither directory exists, the template shipped
with the package is copied to `~/.jqmc_setting/` and the user is asked
to edit it.

### Directory structure

```text
~/.jqmc_setting/
├── machine_data.yaml           # Server machine definitions
├── localhost/                   # Settings for localhost
│   ├── queue_data.toml
│   └── submit_mpi.sh           # (name is user-defined)
├── my-cluster/                 # Settings for a remote cluster (nickname)
│   ├── queue_data.toml
│   └── submit_mpi.sh
└── ...
```

### `machine_data.yaml`

A YAML file that defines each compute machine. The top-level keys are
nicknames (arbitrary labels); they do **not** need to match the SSH host.
Remote machines use the `ssh_host` field to specify the SSH connection
target.

#### Example

```yaml
localhost:
  machine_type: local
  queuing: false
  workspace_root: /home/user/jqmc_work

my-cluster:                          # nickname (freely chosen)
  ssh_host: pbs-cluster              # Host alias in ~/.ssh/config
  machine_type: remote
  queuing: true
  workspace_root: /home/user/jqmc_work
  jobsubmit: /opt/pbs/bin/qsub
  jobcheck: /opt/pbs/bin/qstat
  jobdel: /opt/pbs/bin/qdel
  jobnum_index: 0
  jobacct: /opt/pbs/bin/qstat -xf

my-slurm:                            # Slurm example
  ssh_host: slurm-cluster
  machine_type: remote
  queuing: true
  workspace_root: /home/user/jqmc_work
  jobsubmit: sbatch
  jobcheck: squeue --noheader
  jobdel: scancel
  jobnum_index: 3
  jobacct: sacct -j --format=State,ExitCode,MaxRSS,Elapsed,Timelimit -P --noheader

my-fujitsu:                          # Fujitsu TCS example
  ssh_host: fujitsu
  machine_type: remote
  queuing: true
  workspace_root: /home/user/jqmc_work
  jobsubmit: /usr/local/bin/pjsub
  jobcheck: /usr/local/bin/pjstat
  jobdel: /usr/local/bin/pjdel
  jobnum_index: 5
  jobacct: pjstat -H -s --choose jid,st,ec,elp,pc --data
```

#### Parameters

| Key | Type | Required | Description |
|---|---|---|---|
| `ssh_host` | String | When `machine_type: remote` | SSH connection target. Typically a `Host` alias defined in `~/.ssh/config`. |
| `machine_type` | `"local"` or `"remote"` | Yes | `local`: execute on the same host. `remote`: connect via SSH (Paramiko). |
| `queuing` | Boolean | Yes | `true`: use a batch scheduler. `false`: direct execution. |
| `workspace_root` | String (path) | Yes | Root directory for file management. Upload/download paths are resolved relative to this directory. Must be set on **both** localhost and the remote server. |
| `jobsubmit` | String (command) | When `queuing: true` | Command to submit a job (e.g. `qsub`, `sbatch`, `pjsub`). Not needed for `queuing: false`. |
| `jobcheck` | String (command) | When `queuing: true` | Command to check job status (e.g. `qstat`, `squeue --noheader`, `pjstat`). |
| `jobdel` | String (command) | When `queuing: true` | Command to cancel a job (e.g. `qdel`, `scancel`, `pjdel`). |
| `jobnum_index` | Integer | When `queuing: true` | 0-based column index of the job ID in the output of `jobsubmit`. For PBS (`42.server`), use `0`. For Slurm (`Submitted batch job 42`), use `3`. |
| `jobacct` | String (command) | No | Scheduler accounting command **with flags**. The engine executes `{jobacct} {job_id}` after a job leaves the queue and saves the raw output to `job_accounting_{job_id}.txt`. No parsing is performed. Examples: `sacct -j --format=State,ExitCode,MaxRSS,Elapsed,Timelimit -P --noheader` (Slurm), `qstat -xf` (PBS), `pjstat -H -s --choose jid,st,ec,elp,pc --data` (Fujitsu TCS). If omitted, no accounting data is collected. |
| `ip` | String | No | IP address or hostname of the remote machine. Usually the SSH alias in `~/.ssh/config` is used instead. |

#### SSH connection

For `machine_type: remote`, the `ssh_host` field specifies the SSH
connection target.  This is typically a `Host` alias defined in
`~/.ssh/config`.  The engine reads `~/.ssh/config` via Paramiko to
resolve `HostName`, `User`, `IdentityFile`, `Port`, and `ProxyJump`
(multi-hop) settings.
Your SSH key must be accessible without a passphrase prompt (use
`ssh-agent` or a passphrase-less key).

> **Note:** `CanonicalizeHostname` in `~/.ssh/config` may trigger a
> Paramiko bug.
> The engine automatically works around this by removing the directive
> in-memory before connecting.


### `queue_data.toml`

Defines batch queue presets for each machine. Written in TOML format with
queue labels as table keys.

#### Example

```toml
[default]
    submit_template = 'submit_mpi.sh'
    max_job_submit = 10
    queue = 'batch'
    num_cores = 120
    omp_num_threads = 1
    nodes = 1
    mpi_per_node = 120
    max_time = '24:00:00'

[large]
    submit_template = 'submit_mpi.sh'
    max_job_submit = 10
    queue = 'batch'
    num_cores = 480
    omp_num_threads = 1
    nodes = 4
    mpi_per_node = 120
    max_time = '24:00:00'
```

#### Parameters

| Key | Type | Required | Description |
|---|---|---|---|
| `submit_template` | String | Yes | Name of the job submission script template file placed in the same directory (e.g. `"submit_mpi.sh"`, `"submit_gpu.sh"`). |
| `max_job_submit` | Integer | Yes | Maximum number of concurrently submitted jobs for this queue. |
| *custom keys* | Any | No | Any additional key-value pairs. These are substituted into job script templates as `_KEY_` (upper-case). Common keys: `num_cores`, `omp_num_threads`, `nodes`, `mpi_per_node`, `max_time`, `queue`, `account`, `partition`. |

#### TOML format notes

- Strings must be quoted: `queue = "small"`.
- Booleans: lowercase `true` / `false` only.
- Time values (e.g. `max_time`) **must** be quoted to avoid TOML
  local-time parsing: `max_time = "24:00:00"`.


### Job script templates

Shell scripts placed in the per-machine directory, referenced by the
`submit_template` key in `queue_data.toml`.  Template variables
(written as `_KEY_` with underscores) are replaced at submission time.
You can name the file anything you like (e.g. `submit_gpu.sh`).

#### Predefined variables

| Variable | Description | Default |
|---|---|---|
| `_INPUT_` | Path to the jqmc input TOML file | (set by workflow) |
| `_OUTPUT_` | Path to the jqmc stdout+stderr capture file | `"out.o"` |
| `_JOBNAME_` | Job name | `"jqmc-wf"` |
| `_JOB_STDOUT_` | Path for the **scheduler** stdout file (e.g. PBS `-o`, Slurm `--output`) | `"job_{jobname}.o"` |
| `_JOB_STDERR_` | Path for the **scheduler** stderr file (e.g. PBS `-e`, Slurm `--error`) | `"job_{jobname}.e"` |

`_JOB_STDOUT_` and `_JOB_STDERR_` allow the engine to track where the
scheduler writes its output, which is useful for failure diagnosis.
If these placeholders are not present in the template the scheduler's
default naming convention is used (backward-compatible).
The file paths are recorded per-job in the `[[jobs]]` records of
`workflow_state.toml` (as `job_stdout` and `job_stderr`).

#### Custom variables

All keys from `queue_data.toml` are available in upper-case with
surrounding underscores. For example, `num_cores = 48` becomes
`_NUM_CORES_` in the template.

#### Example (`submit_mpi.sh` for PBS)

```bash
#!/bin/sh
#PBS -N _JOBNAME_
#PBS -q _QUEUE_
#PBS -l nodes=_NODES_:ppn=_MPI_PER_NODE_
#PBS -l walltime=_MAX_TIME_
#PBS -o _JOB_STDOUT_
#PBS -e _JOB_STDERR_

export OMP_NUM_THREADS=_OMP_NUM_THREADS_
INPUT=_INPUT_
OUTPUT=_OUTPUT_

cd ${PBS_O_WORKDIR}
mpirun -np _NUM_CORES_ jqmc ${INPUT} > ${OUTPUT} 2>&1
```

#### Example (`submit_mpi.sh` for Slurm)

```bash
#!/bin/bash
#SBATCH --job-name=_JOBNAME_
#SBATCH --partition=_QUEUE_
#SBATCH --nodes=_NODES_
#SBATCH --ntasks=_NUM_CORES_
#SBATCH --time=_MAX_TIME_
#SBATCH --output=_JOB_STDOUT_
#SBATCH --error=_JOB_STDERR_

export OMP_NUM_THREADS=_OMP_NUM_THREADS_
INPUT=_INPUT_
OUTPUT=_OUTPUT_

srun jqmc ${INPUT} > ${OUTPUT} 2>&1
```

#### Example (`submit_serial.sh`)

```bash
#!/bin/sh
#PBS -N _JOBNAME_
#PBS -q _QUEUE_
#PBS -l nodes=_NODES_
#PBS -l walltime=_MAX_TIME_
#PBS -o _JOB_STDOUT_
#PBS -e _JOB_STDERR_

export OMP_NUM_THREADS=_OMP_NUM_THREADS_
INPUT=_INPUT_
OUTPUT=_OUTPUT_

cd ${PBS_O_WORKDIR}
jqmc ${INPUT} > ${OUTPUT} 2>&1
```


## Pipeline example

A minimal pipeline script that runs VMC → MCMC + LRDMC extrapolation:

```python
from jqmc_workflow import (
    Container,
    FileFrom,
    Launcher,
    LRDMC_Ext_Workflow,
    MCMC_Workflow,
    ValueFrom,
    VMC_Workflow,
)

server = "pbs-cluster"
h5 = "hamiltonian_data.h5"

vmc = Container(
    label="vmc",
    dirname="vmc",
    input_files=[h5],
    workflow=VMC_Workflow(
        server_machine_name=server,
        hamiltonian_file=h5,
        queue_label="default",
        pilot_queue_label="small",
        jobname="vmc",
        target_error=0.001,
    ),
)

mcmc = Container(
    label="mcmc-prod",
    dirname="mcmc_prod",
    input_files=[
        h5,
        FileFrom("vmc", "hamiltonian_data_opt_step_*.h5"),
    ],
    workflow=MCMC_Workflow(
        server_machine_name=server,
        hamiltonian_file=h5,
        queue_label="default",
        pilot_queue_label="small",
        jobname="mcmc",
        target_error=0.001,
    ),
)

lrdmc = Container(
    label="lrdmc-ext",
    dirname="lrdmc_ext",
    input_files=[
        h5,
        FileFrom("vmc", "hamiltonian_data_opt_step_*.h5"),
    ],
    workflow=LRDMC_Ext_Workflow(
        server_machine_name=server,
        alat_list=[0.10, 0.20, 0.25, 0.30],
        hamiltonian_file=h5,
        queue_label="default",
        pilot_queue_label="small",
        jobname_prefix="lrdmc",
        target_survived_walkers_ratio=0.97,
        E_scf=ValueFrom("mcmc-prod", "energy"),
        target_error=0.001,
    ),
)

pipeline = Launcher(workflows=[vmc, mcmc, lrdmc])
pipeline.launch()
```

In this example, `mcmc-prod` and `lrdmc-ext` depend on `vmc` (via
`FileFrom`).  Additionally, `lrdmc-ext` depends on `mcmc-prod` (via
`ValueFrom` for `E_scf`), so the DAG becomes
VMC → MCMC → LRDMC-ext.  The `target_survived_walkers_ratio`
triggers automatic calibration of `num_projection_per_measurement`
independently at each lattice spacing.  All alat values run their
calibration, error-bar pilot, and production phases in parallel.

## Job Manager CLI (`jqmc-jobmanager`)

The `jqmc-jobmanager` command-line tool monitors and manages running
pipelines.  It recursively discovers `workflow_state.toml` files under
the current directory and displays a summary tree.

### Commands

| Command | Description |
|---|---|
| `show`  | Print the workflow tree.  Add `--id N` to display full detail for one job. |
| `check` | Print the tree **and** query the scheduler on the remote machine for live job status.  Use `--id N` to target a specific job. |
| `del`   | Cancel a queued/running job and mark the workflow as `cancelled`.  Requires `--id N`. |

### Common options

| Option | Description |
|---|---|
| `--id N` | Numeric job ID shown in the tree (0-based). |
| `-s` / `--server` | Server machine name (defaults to `localhost`). Used by `check` and `del` to connect to the remote scheduler. |
| `--log-level` | `INFO` (default) or `DEBUG`. |

### Usage examples

Show the full workflow tree:

```bash
jqmc-jobmanager show
```

Show tree and detailed info for job 2:

```bash
jqmc-jobmanager show --id 2
```

Check live queue status for job 4 on `pbs-cluster`:

```bash
jqmc-jobmanager check --id 4 -s pbs-cluster
```

Cancel job 4 on `pbs-cluster`:

```bash
jqmc-jobmanager del --id 4 -s pbs-cluster
```

### Example output

```text
==============================================================================
  Workflow Job Tree
  Root: /home/user/project
==============================================================================
   ID  Status       Label                Type                 Server       Job#
  --------------------------------------------------------------------------
    0  completed    wf                   WF_Workflow          ?            -
       dir: 00_wf
    1  completed    vmc                  VMC_Workflow         pbs-cluster  12345
       dir: 01_vmc
       energy: -17.17 +- 0.005 Ha
    2  running      mcmc-prod            MCMC_Workflow        pbs-cluster  12367
       dir: 02_mcmc
    3  running      lrdmc-ext            LRDMC_Ext_Workflow   ?            -
       dir: 03_lrdmc_ext
```

For the full API reference, see
{ref}`API reference for the pipeline (jqmc_workflow) <api_ref_workflow>`.
