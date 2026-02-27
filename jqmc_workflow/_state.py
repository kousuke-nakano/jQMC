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
from logging import getLogger

import toml

logger = getLogger("jqmc-workflow").getChild(__name__)

VALID_STATUSES = {
    "pending",
    "copying",
    "submitted",
    "running",
    "completed",
    "failed",
    "cancelled",
}

VALID_JOB_STATUSES = {"submitted", "completed", "fetched", "failed"}

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


def update_status(directory: str, status: str, **extra_fields):
    """Update the status field (and optional extra fields) in workflow_state.toml."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")

    state = read_state(directory)
    if not state:
        logger.warning(f"No workflow_state.toml in {directory}; creating minimal one.")
        state = {"workflow": {}, "jobs": [], "result": {}}

    state.setdefault("workflow", {})
    state["workflow"]["status"] = status
    state["workflow"]["updated_at"] = _now_iso()

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


def _write(directory: str, state: dict):
    """Write state dict to workflow_state.toml."""
    path = os.path.join(directory, STATE_FILENAME)
    os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        toml.dump(state, f)
