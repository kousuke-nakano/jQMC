"""Workflow state persistence via workflow_state.toml.

Each workflow directory gets a human-and-machine-readable TOML file that
tracks the current status, full job history, and results.

Workflow-level statuses:
    pending     -- waiting for dependencies
    copying     -- input files being transferred
    submitted   -- job submitted to queue
    running     -- job confirmed running
    completed   -- finished successfully
    failed      -- finished with error
    cancelled   -- manually cancelled

Per-job statuses (stored in ``[[jobs]]``):
    submitted   -- job submitted to scheduler / started locally
    completed   -- scheduler reports job finished
    fetched     -- results transferred back to local machine
    failed      -- job failed
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the jqmc project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
from datetime import datetime
from enum import Enum
from logging import getLogger

import toml

logger = getLogger("jqmc-workflow").getChild(__name__)


class WorkflowStatus(str, Enum):
    """Workflow-level status values."""

    PENDING = "pending"
    COPYING = "copying"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(str, Enum):
    """Per-job status values stored in ``[[jobs]]``."""

    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FETCHED = "fetched"
    FAILED = "failed"


# Legacy sets — kept for backward compatibility during transition.
# New code should use WorkflowStatus / JobStatus enums.
VALID_STATUSES = {s.value for s in WorkflowStatus}
VALID_JOB_STATUSES = {s.value for s in JobStatus}

STATE_FILENAME = "workflow_state.toml"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def create_state(
    directory: str,
    label: str,
    workflow_type: str,
    status: str = "pending",
) -> dict:
    """Create (or reset) workflow_state.toml in *directory*.

    If the file already exists, the ``[estimation]`` and ``[[jobs]]``
    sections are preserved so that pilot-run results and job history
    survive a restart.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")

    # Preserve data from a previous run
    existing = read_state(directory)
    preserved_estimation = existing.get("estimation", {})
    preserved_jobs = existing.get("jobs", [])

    state = {
        "workflow": {
            "label": label,
            "type": workflow_type,
            "status": status,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        },
        "jobs": preserved_jobs,
        "result": {},
    }

    if preserved_estimation:
        state["estimation"] = preserved_estimation

    _write(directory, state)
    return state


def read_state(directory: str) -> dict:
    """Read workflow_state.toml from *directory*. Returns empty dict if absent."""
    path = os.path.join(directory, STATE_FILENAME)
    if not os.path.isfile(path):
        return {}
    state = toml.load(path)
    # Migrate legacy single [job] -> [[jobs]] list
    if "job" in state and "jobs" not in state:
        old_job = state.pop("job")
        if old_job:
            state["jobs"] = [old_job]
        else:
            state["jobs"] = []
    return state


def _check_normal_termination(directory: str, jobs: list) -> list[str]:
    """Check fetched output files for the ``Program ends`` marker.

    Returns a list of output-file names that exist on disk but do **not**
    contain the ``Program ends`` line — a strong signal that the
    computation was killed (e.g. wall-time expiration) before normal
    termination.

    Files that are absent, unreadable, or binary are silently skipped.
    """
    abnormal: list[str] = []
    for job in jobs:
        output_file = job.get("output_file", "")
        if not output_file:
            continue
        filepath = os.path.join(directory, output_file)
        if not os.path.isfile(filepath):
            continue  # not fetched yet — nothing to check
        try:
            with open(filepath, "r", errors="replace") as f:
                # Read only the tail (last 8 KiB) for efficiency;
                # "Program ends ..." is always the last log line.
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 8192))
                tail = f.read()
            if "Program ends" not in tail:
                abnormal.append(output_file)
        except OSError:
            continue  # unreadable — skip
    return abnormal


def update_status(
    directory: str,
    status: str | WorkflowStatus,
    phase: str | None = None,
    **extra_fields,
):
    """Update the status field (and optional extra fields) in workflow_state.toml.

    Parameters
    ----------
    directory : str
        Working directory containing workflow_state.toml.
    status : str or WorkflowStatus
        New workflow status.
    phase : str or None
        Scientific phase to record.  If given, written to
        ``[workflow] phase``.
    **extra_fields
        Additional fields.  Keys starting with ``result_`` go into
        ``[result]``; everything else goes into ``[workflow]``.

    When *status* is ``"completed"``, every fetched output file listed in
    the ``[[jobs]]`` table is checked for the ``Program ends`` footer.  If
    any output file exists on disk but lacks this marker the status is
    automatically downgraded to ``"failed"`` and an ``error`` field is
    recorded, because the absence almost certainly indicates abnormal
    termination (e.g. wall-time expiration on a supercomputer).
    """
    # Ensure we work with raw string for VALID_STATUSES check
    status_str = status.value if isinstance(status, WorkflowStatus) else status
    if status_str not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status_str}'. Must be one of {VALID_STATUSES}")
    state = read_state(directory)
    if not state:
        logger.warning(f"No workflow_state.toml in {directory}; creating minimal one.")
        state = {"workflow": {}, "jobs": [], "result": {}}

    # ── Abnormal-termination guard ────────────────────────────────
    if status_str == "completed":
        abnormal = _check_normal_termination(directory, state.get("jobs", []))
        if abnormal:
            files_str = ", ".join(abnormal)
            error_msg = f"Abnormal termination detected: 'Program ends' marker missing in output file(s): {files_str}"
            logger.error(error_msg)
            status_str = "failed"
            extra_fields.setdefault("error", error_msg)

    state.setdefault("workflow", {})
    state["workflow"]["status"] = status_str
    state["workflow"]["updated_at"] = _now_iso()

    if phase is not None:
        state["workflow"]["phase"] = phase.value if hasattr(phase, "value") else phase

    # Merge extra fields into appropriate sections
    for key, value in extra_fields.items():
        if key.startswith("result_"):
            state.setdefault("result", {})[key[7:]] = value
        else:
            state["workflow"][key] = value

    _write(directory, state)
    return state


# ── Job history (replaces old set_job_info / get_job_info) ────────


def add_job(
    directory: str,
    input_file: str,
    output_file: str,
    job_id: str,
    server_machine: str,
    status: str = "submitted",
) -> dict:
    """Append a new job record to the ``[[jobs]]`` list.

    Returns the newly created job dict.
    """
    if status not in VALID_JOB_STATUSES:
        raise ValueError(f"Invalid job status '{status}'. Must be one of {VALID_JOB_STATUSES}")

    state = read_state(directory)
    state.setdefault("jobs", [])

    job = {
        "input_file": input_file,
        "output_file": output_file,
        "job_id": str(job_id),
        "server_machine": server_machine,
        "status": status,
        "submitted_at": _now_iso(),
    }
    state["jobs"].append(job)
    state.setdefault("workflow", {})["updated_at"] = _now_iso()
    _write(directory, state)
    return job


def update_job(directory: str, input_file: str, **fields):
    """Update the latest job record matching *input_file*.

    Common fields: ``status``, ``completed_at``, ``fetched_at``.
    """
    state = read_state(directory)
    jobs = state.get("jobs", [])

    # Find the last matching job (in case of retries)
    target = None
    for j in reversed(jobs):
        if j.get("input_file") == input_file:
            target = j
            break

    if target is None:
        logger.warning(f"No job record for {input_file} in {directory}")
        return

    target.update(fields)
    state.setdefault("workflow", {})["updated_at"] = _now_iso()
    _write(directory, state)


def get_job(directory: str, input_file: str) -> dict:
    """Get the latest job record for *input_file*.

    Returns an empty dict if no matching record is found.
    """
    state = read_state(directory)
    for j in reversed(state.get("jobs", [])):
        if j.get("input_file") == input_file:
            return j
    return {}


def get_jobs(directory: str) -> list:
    """Get all job records from ``[[jobs]]``.

    Returns an empty list if no records or no state file.
    """
    state = read_state(directory)
    return state.get("jobs", [])


# ── Result / estimation helpers (unchanged) ───────────────────────


def set_result(directory: str, **result_fields):
    """Write result data (energy, error, etc.) to workflow_state.toml."""
    state = read_state(directory)
    if not state:
        return
    state.setdefault("result", {}).update(result_fields)
    state["workflow"]["updated_at"] = _now_iso()
    _write(directory, state)


def set_estimation(directory: str, **kwargs):
    """Store step-estimation results in workflow_state.toml.

    Example::

        set_estimation(dir, pilot_steps=100, pilot_error=0.005,
                       target_error=0.001, estimated_steps=3000)
    """
    state = read_state(directory)
    state.setdefault("estimation", {}).update(kwargs)
    state.setdefault("workflow", {})["updated_at"] = _now_iso()
    _write(directory, state)


def get_estimation(directory: str) -> dict:
    """Read the ``[estimation]`` section from workflow_state.toml.

    Returns an empty dict if the section or the file is absent.
    """
    state = read_state(directory)
    return state.get("estimation", {})


def get_all_workflow_statuses(base_dir: str) -> list:
    """Recursively find all ``workflow_state.toml`` files under *base_dir*.

    Returns a list of dicts, each containing:

    - ``directory`` – absolute path to the workflow directory
    - ``label``     – workflow label (from ``[workflow]``)
    - ``type``      – workflow type (e.g. ``"vmc"``)
    - ``status``    – current workflow status

    Directories without a ``workflow_state.toml`` are silently skipped.
    """
    results = []
    base_dir = os.path.abspath(base_dir)
    for dirpath, dirnames, filenames in os.walk(base_dir):
        # Skip pilot-run subdirectories — they have workflow_state.toml
        # but are internal bookkeeping, not user-facing workflows.
        dirnames[:] = [d for d in dirnames if not d.startswith("_pilot")]
        if STATE_FILENAME in filenames:
            state = read_state(dirpath)
            wf = state.get("workflow", {})
            results.append(
                {
                    "directory": dirpath,
                    "label": wf.get("label", ""),
                    "type": wf.get("type", ""),
                    "status": wf.get("status", "unknown"),
                }
            )
    # Sort by directory for deterministic output
    results.sort(key=lambda r: r["directory"])
    return results


def get_workflow_summary(directory: str) -> dict:
    """Return a comprehensive summary of the workflow in *directory*.

    The returned dict contains:

    - ``workflow`` – label, type, status, timestamps
    - ``phase``    – current scientific phase (str or ``"init"``)
    - ``allowed_actions`` – list of permitted MCP actions
    - ``result``   – any stored results (energy, etc.)
    - ``estimation`` – step-estimation data (if present)
    - ``jobs``     – list of job records with their statuses
    - ``num_jobs`` – total number of job records
    - ``error``    – ``[error]`` section or ``None``
    - ``job_accounting`` – ``[job_accounting]`` section or ``None``
    - ``job_files`` – ``[job_files]`` section or ``None``
    - ``artifacts`` – ``[[artifacts]]`` list

    Returns an empty dict if no ``workflow_state.toml`` is found.
    """
    state = read_state(directory)
    if not state:
        return {}

    jobs = state.get("jobs", [])
    phase_str = state.get("workflow", {}).get("phase", "init")
    status_str = state.get("workflow", {}).get("status", "pending")

    # Compute allowed_actions (lazy import to avoid circular dependency)
    try:
        from ._phase import ScientificPhase, allowed_actions

        aa = allowed_actions(ScientificPhase(phase_str), WorkflowStatus(status_str))
    except (ValueError, ImportError):
        aa = []

    return {
        "workflow": state.get("workflow", {}),
        "phase": phase_str,
        "allowed_actions": aa,
        "result": state.get("result", {}),
        "estimation": state.get("estimation", {}),
        "jobs": jobs,
        "num_jobs": len(jobs),
        "error": state.get("error", None),
        "job_accounting": state.get("job_accounting", None),
        "job_files": state.get("job_files", None),
        "artifacts": state.get("artifacts", []),
    }


# ── Error / accounting / artifact helpers ─────────────────────────


def set_error(directory: str, message: str, **context) -> None:
    """Write error information to the ``[error]`` section.

    Parameters
    ----------
    message : str
        Human-readable error description (exception message, etc.).
    **context
        Arbitrary extra fields (``traceback``, ``exception_type``, …).
    """
    state = read_state(directory)
    state["error"] = {"message": message, **context}
    state.setdefault("workflow", {})["updated_at"] = _now_iso()
    _write(directory, state)


def set_job_accounting(
    directory: str,
    command: str,
    stdout: str,
    stderr: str = "",
    job_id: str = "",
) -> None:
    """Record scheduler accounting raw output.

    The raw stdout is written to a separate file
    ``job_accounting_{job_id}.txt`` next to ``workflow_state.toml``.
    The ``[job_accounting]`` section stores only the command and a
    reference to that file.  No parsing or interpretation is performed.
    """
    # Write raw output to a separate file
    if job_id:
        acct_filename = f"job_accounting_{job_id}.txt"
    else:
        acct_filename = "job_accounting.txt"
    acct_path = os.path.join(directory, acct_filename)
    os.makedirs(directory, exist_ok=True)
    with open(acct_path, "w") as f:
        f.write(f"# command: {command}\n")
        f.write(f"# job_id: {job_id}\n\n")
        f.write("--- stdout ---\n")
        f.write(stdout)
        if stderr:
            f.write("\n--- stderr ---\n")
            f.write(stderr)

    # Record reference in toml
    state = read_state(directory)
    state["job_accounting"] = {
        "command": command,
        "file": acct_filename,
    }
    state.setdefault("workflow", {})["updated_at"] = _now_iso()
    _write(directory, state)


def register_artifact(
    directory: str,
    filename: str,
    produced_by_job: str = "",
    artifact_type: str = "file",
    upstream: list[dict] | None = None,
) -> None:
    """Register (or update) a file artifact in ``[[artifacts]]``."""
    state = read_state(directory)
    artifacts = state.setdefault("artifacts", [])
    # Replace existing entry for the same filename
    artifacts[:] = [a for a in artifacts if a.get("filename") != filename]
    entry: dict = {
        "filename": filename,
        "produced_by_job": produced_by_job,
        "produced_at": _now_iso(),
        "artifact_type": artifact_type,
    }
    if upstream:
        entry["upstream"] = upstream
    artifacts.append(entry)
    state["artifacts"] = artifacts
    state.setdefault("workflow", {})["updated_at"] = _now_iso()
    _write(directory, state)


def get_artifact_lineage(directory: str, filename: str) -> dict | None:
    """Return the artifact record for *filename*, or ``None``."""
    for a in read_state(directory).get("artifacts", []):
        if a.get("filename") == filename:
            return a
    return None


def get_artifact_registry(directory: str) -> list[dict]:
    """Return all artifact records from ``[[artifacts]]``."""
    return read_state(directory).get("artifacts", [])


def _write(directory: str, state: dict):
    """Write state dict to workflow_state.toml."""
    path = os.path.join(directory, STATE_FILENAME)
    os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        toml.dump(state, f)
