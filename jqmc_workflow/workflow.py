"""Base Workflow and Encapsulated Workflow for jqmc-workflow.

Workflow state is tracked via workflow_state.toml (human+machine readable).
Dependencies between workflows are declared with FileFrom / ValueFrom.
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

import asyncio
import os
import shutil
from logging import getLogger
from typing import Any, List, Optional

from ._job import JobSubmission
from ._state import (
    _now_iso,
    add_job,
    create_state,
    get_job,
    read_state,
    update_job,
    update_status,
)

logger = getLogger("jqmc-workflow").getChild(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Dependency specification helpers
# ═══════════════════════════════════════════════════════════════════


class FileFrom:
    """Declare that an input file should come from another workflow's output.

    Used inside :class:`Container` definitions to express
    inter-workflow file dependencies.  The :class:`Launcher` resolves
    these placeholders before a workflow is launched.

    Parameters
    ----------
    label : str
        Label of the upstream workflow that produces the file.
    filename : str
        Filename (basename) to pull from the upstream workflow's
        output directory.

    Examples
    --------
    Pass an optimised Hamiltonian from a VMC step to an MCMC step::

        Container(
            label="mcmc-run",
            dirname="mcmc",
            input_files=[FileFrom("vmc-opt", "hamiltonian_data_opt_step_9.h5")],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=MCMC_Workflow(...),
        )

    See Also
    --------
    ValueFrom : Declare a scalar-value dependency.
    Launcher  : Resolves ``FileFrom`` / ``ValueFrom`` at launch time.
    """

    def __init__(self, label: str, filename: str):
        self.label = label
        self.filename = filename

    def __repr__(self):
        return f"FileFrom({self.label!r}, {self.filename!r})"


class ValueFrom:
    """Declare that a parameter value should come from another workflow's output.

    Used when a downstream workflow needs a *scalar* result (energy,
    error, filename string, etc.) produced by an upstream workflow.
    The :class:`Launcher` resolves these placeholders before launch.

    Parameters
    ----------
    label : str
        Label of the upstream workflow that produces the value.
    key : str
        Key name in the upstream workflow's ``output_values`` dict.

    Examples
    --------
    Feed the MCMC energy into an LRDMC workflow as ``trial_energy``::

        LRDMC_Workflow(
            trial_energy=ValueFrom("mcmc-run", "energy"),
            ...
        )

    See Also
    --------
    FileFrom : Declare a file dependency.
    Launcher : Resolves ``FileFrom`` / ``ValueFrom`` at launch time.
    """

    def __init__(self, label: str, key: str):
        self.label = label
        self.key = key

    def __repr__(self):
        return f"ValueFrom({self.label!r}, {self.key!r})"


def _is_dependency(obj) -> bool:
    """Check if an object is a dependency placeholder."""
    return isinstance(obj, (FileFrom, ValueFrom))


# ═══════════════════════════════════════════════════════════════════
#  Base Workflow
# ═══════════════════════════════════════════════════════════════════


class Workflow:
    """Abstract base class for all jQMC computation workflows.

    Every concrete workflow (VMC, MCMC, LRDMC, WF, …) inherits from
    this class and overrides :meth:`async_launch`.

    Parameters
    ----------
    project_dir : str, optional
        Absolute path to the working directory for this workflow.
        When *None* (the default), ``project_dir`` is set to the
        process CWD at the time :meth:`async_launch` is first called.
        :class:`Container` sets this explicitly before launching the
        inner workflow.

    Attributes
    ----------
    status : str
        Current lifecycle status (``"init"``, ``"success"``, ``"failed"``).
    output_files : list[str]
        Filenames produced by the workflow (populated after launch).
    output_values : dict
        Scalar results (energy, error, …) produced by the workflow.
    project_dir : str or None
        Working directory for file I/O.  Resolved to an absolute path.

    Notes
    -----
    **Subclass contract:**

    * Override :meth:`async_launch` and return
      ``(status, output_files, output_values)``.
    * Call ``super().__init__()`` in your constructor.

    Examples
    --------
    Minimal custom workflow::

        class MyWorkflow(Workflow):
            async def async_launch(self):
                # ... do work ...
                self.status = "success"
                return self.status, ["result.h5"], {"energy": -1.23}
    """

    def __init__(self, project_dir: Optional[str] = None):
        self.status = "init"
        self.output_files: List[str] = []
        self.output_values: dict = {}
        self.project_dir: Optional[str] = os.path.abspath(project_dir) if project_dir else None

    def _ensure_project_dir(self):
        """Lazily resolve *project_dir* to CWD when not set explicitly."""
        if self.project_dir is None:
            self.project_dir = os.path.abspath(os.getcwd())

    # ── Full lifecycle (backward-compatible) ──────────────────────

    async def async_launch(self):
        """Override in subclass. Must return (status, output_files, output_values)."""
        self._ensure_project_dir()
        return self.status, self.output_files, self.output_values

    def launch(self):
        return asyncio.run(self.async_launch())

    # ── Phased execution (new API for MCP / external orchestration) ──

    async def async_submit(self) -> dict:
        """Submit initial job(s) and return tracking information.

        Override in subclass.  The default raises ``NotImplementedError``.

        Returns
        -------
        dict
            At minimum ``{"status": "submitted"}``.
        """
        raise NotImplementedError

    async def async_poll(self) -> str:
        """Check whether submitted jobs have completed.

        Override in subclass.  The default raises ``NotImplementedError``.

        Returns
        -------
        str
            One of ``"running"``, ``"completed"``, ``"failed"``.
        """
        raise NotImplementedError

    async def async_collect(self) -> dict:
        """Collect and return results from completed jobs.

        Override in subclass.  The default raises ``NotImplementedError``.

        Returns
        -------
        dict
            Workflow-specific result mapping.
        """
        raise NotImplementedError

    # ── Common job helpers (used by VMC / MCMC / LRDMC) ───────────
    #
    # These methods require the following attributes on *self*:
    #   server_machine_name, hamiltonian_file, queue_label,
    #   jobname, poll_interval
    # They are therefore only callable from concrete subclasses that
    # set those attributes.

    async def _submit_and_wait(
        self,
        input_file,
        output_file,
        work_dir,
        extra_from_objects=None,
        fetch_from_objects=None,
        queue_label=None,
    ):
        """Submit a job, poll until done, fetch results.

        This method is CWD-safe — it never calls ``os.chdir()``.
        All path context is passed explicitly via *work_dir*.

        Restart behaviour is driven by ``workflow_state.toml``:

        * ``fetched``   — skip entirely (already done)
        * ``completed`` — fetch results only
        * ``submitted`` — resume waiting, then fetch
        * no record     — submit a new job

        Parameters
        ----------
        input_file : str
            Basename of the input TOML file (relative to *work_dir*).
        output_file : str
            Basename of the stdout capture file.
        work_dir : str
            Absolute path to the directory where the job runs.
        extra_from_objects : list, optional
            Additional files to upload with the job (e.g. checkpoint).
        fetch_from_objects : list, optional
            Glob patterns for files to fetch.  Defaults to
            ``["*.h5", output_file]``.
        queue_label : str, optional
            Override for the queue label.
        """
        if fetch_from_objects is None:
            fetch_from_objects = ["*.h5", output_file]

        # ── Restart detection via job history ─────────────────────
        recorded = get_job(work_dir, input_file)

        if recorded.get("status") == "fetched":
            logger.info(f"  {input_file}: already fetched. Skipping.")
            return

        if recorded.get("status") == "completed":
            logger.info(f"  {input_file}: completed but not fetched. Fetching...")
            job = self._make_job(input_file, output_file, queue_label=queue_label)
            job.fetch_job(from_objects=fetch_from_objects, exclude_patterns=[], work_dir=work_dir)
            update_job(work_dir, input_file, status="fetched", fetched_at=_now_iso())
            return

        if recorded.get("status") == "submitted":
            stored_job_id = recorded.get("job_id")
            logger.info(f"  Resuming previously submitted job {stored_job_id}")
            job = self._make_job(input_file, output_file, queue_label=queue_label)
            job.job_number = stored_job_id
            while job.jobcheck():
                logger.info(f"  Job {stored_job_id} still running, waiting {self.poll_interval}s...")
                await asyncio.sleep(self.poll_interval)
            logger.info("  Job completed.")
            update_job(work_dir, input_file, status="completed", completed_at=_now_iso())
            job.fetch_job(from_objects=fetch_from_objects, exclude_patterns=[], work_dir=work_dir)
            update_job(work_dir, input_file, status="fetched", fetched_at=_now_iso())
            return

        # ── New submission ────────────────────────────────────────
        job = self._make_job(input_file, output_file, queue_label=queue_label)
        job.generate_script(work_dir=work_dir)

        from_objects = [input_file, self.hamiltonian_file, "submit.sh"]
        if extra_from_objects:
            from_objects.extend(extra_from_objects)
        submitted, job_number = job.job_submit(from_objects=from_objects, work_dir=work_dir)
        if not submitted:
            raise RuntimeError("Job submission failed (queue limit or error).")

        logger.info(f"  Job submitted: {job_number}")
        add_job(
            work_dir,
            input_file=input_file,
            output_file=output_file,
            job_id=str(job_number) if job_number else "local",
            server_machine=self.server_machine_name,
        )

        while job.jobcheck():
            logger.info(f"  Job {job_number} still running, waiting {self.poll_interval}s...")
            await asyncio.sleep(self.poll_interval)

        logger.info("  Job completed.")
        update_job(work_dir, input_file, status="completed", completed_at=_now_iso())
        job.fetch_job(from_objects=fetch_from_objects, exclude_patterns=[], work_dir=work_dir)
        update_job(work_dir, input_file, status="fetched", fetched_at=_now_iso())

    def _make_job(self, input_file, output_file, queue_label=None):
        """Create a :class:`JobSubmission` with current workflow settings.

        Requires *self.server_machine_name*, *self.queue_label*, and
        *self.jobname* to be set by the concrete subclass.
        """
        return JobSubmission(
            server_machine_name=self.server_machine_name,
            input_file=input_file,
            output_file=output_file,
            queue_label=queue_label or self.queue_label,
            jobname=self.jobname,
        )


# ═══════════════════════════════════════════════════════════════════
#  Container
# ═══════════════════════════════════════════════════════════════════


class Container:
    """Run a :class:`Workflow` inside a dedicated project directory.

    ``Container`` is the standard wrapper used with the
    :class:`Launcher`.  It manages:

    * **Directory creation** — a self-contained project directory is
      created under the current working directory.
    * **Input file copying** — source files (or resolved
      :class:`FileFrom` references) are copied into the project dir.
    * **State tracking** — a ``workflow_state.toml`` file records
      lifecycle status (``pending`` → ``running`` → ``completed``).
    * **Re-entrance** — if the directory already exists with status
      ``completed``, the workflow is *not* re-run; outputs are read
      from the state file instead.

    Parameters
    ----------
    label : str
        Human-readable label; also used as the key for dependency
        resolution in the :class:`Launcher`.
    dirname : str
        Directory name to create (relative to CWD).
    input_files : list[str | FileFrom]
        Files to copy into the project directory before launch.
        Items may be plain paths or :class:`FileFrom` objects.
    rename_input_files : list[str], optional
        If provided (same length as *input_files*), each copied file
        is renamed to the corresponding entry.
    workflow : Workflow
        The inner :class:`Workflow` instance to execute.

    Attributes
    ----------
    output_files : list[str]
        Output filenames (populated after launch).
    output_values : dict
        Scalar results from the inner workflow.
    status : str
        Current status.
    project_dir : str
        Absolute path to the project directory.

    Examples
    --------
    Wrap a VMC optimization in its own directory::

        enc = Container(
            label="vmc-opt",
            dirname="01_vmc",
            input_files=["hamiltonian_data.h5"],
            workflow=VMC_Workflow(
                server_machine_name="cluster",
                num_opt_steps=10,
                target_error=0.001,
            ),
        )
        status, files, values = enc.launch()

    See Also
    --------
    Launcher : Execute multiple ``Container`` objects as a DAG.
    FileFrom : Reference an output file from another workflow.
    """

    def __init__(
        self,
        label: Optional[str] = "workflow",
        dirname: Optional[str] = "workflow",
        input_files: Optional[list] = None,
        rename_input_files: Optional[list] = None,
        workflow: Optional[Workflow] = None,
    ):
        self.label = label
        self.dirname = dirname
        self.input_files = input_files or []
        self.rename_input_files = rename_input_files or []
        self.workflow = workflow or Workflow()

        # Output placeholders (populated after launch)
        self.output_files: List[str] = []
        self.output_values: dict = {}
        self.status = "init"

        # Directories
        self.root_dir = os.getcwd()
        self.project_dir = os.path.join(self.root_dir, self.dirname)

    # ── Preparation ───────────────────────────────────────────────

    def _prepare(self):
        """Create project dir, copy input files, write initial state."""

        state = read_state(self.project_dir)
        existing_status = state.get("workflow", {}).get("status", "")

        if existing_status in ("completed", "running"):
            logger.info(f"[{self.label}] Already {existing_status}. Delete project dir to restart from scratch.")
            return

        if os.path.isdir(self.project_dir):
            logger.info(f"[{self.label}] Project dir exists, skipping file copy.")
        else:
            logger.info(f"[{self.label}] Creating project dir: {self.project_dir}")
            os.makedirs(self.project_dir, exist_ok=False)
            self._copy_input_files()

        # Write state
        create_state(
            directory=self.project_dir,
            label=self.label,
            workflow_type=type(self.workflow).__name__,
            status="pending",
        )

    def _copy_input_files(self):
        """Copy input files into the project directory."""
        rename = len(self.rename_input_files) > 0 and len(self.rename_input_files) == len(self.input_files)

        for i, src in enumerate(self.input_files):
            src = str(src)  # resolve pathlib objects
            # Resolve relative paths against root_dir
            if not os.path.isabs(src):
                src = os.path.join(self.root_dir, src)
            if rename:
                dst_name = os.path.basename(str(self.rename_input_files[i]))
            else:
                dst_name = os.path.basename(src)

            dst = os.path.join(self.project_dir, dst_name)

            if os.path.isfile(src):
                shutil.copy(src, dst)
            elif os.path.isdir(src):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                logger.warning(f"[{self.label}] Input not found: {src}")

    # ── Launch ────────────────────────────────────────────────────

    async def async_launch(self):
        proj = os.path.abspath(self.project_dir)

        self._prepare()

        # Check if already completed
        state = read_state(proj)
        if state.get("workflow", {}).get("status") == "completed":
            logger.info(f"[{self.label}] Already completed, no re-run.")
            self.status = "success"
            self._collect_outputs()
            return self.status, self.output_files, self.output_values

        # Run the workflow — pass project_dir explicitly instead of
        # relying on os.chdir().
        update_status(proj, "running")
        self.workflow.project_dir = proj

        try:
            self.status, self.output_files, self.output_values = await self.workflow.async_launch()
        except Exception as e:
            update_status(proj, "failed", error=str(e))
            raise

        # Write completion
        result_fields = {}
        for k, v in self.output_values.items():
            result_fields[f"result_{k}"] = v
        update_status(proj, "completed", **result_fields)

        return self.status, self.output_files, self.output_values

    def _collect_outputs(self):
        """Re-collect output info from state file (for already-completed runs)."""
        state = read_state(self.project_dir)
        self.output_values = state.get("result", {})
        # Gather all files in project dir as potential outputs
        if os.path.isdir(self.project_dir):
            self.output_files = [
                f
                for f in os.listdir(self.project_dir)
                if os.path.isfile(os.path.join(self.project_dir, f)) and f != "workflow_state.toml"
            ]

    def launch(self):
        return asyncio.run(self.async_launch())
