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
import hashlib
import os
import shutil
import uuid
from logging import getLogger
from typing import List, Optional

from ._job import JobSubmission
from ._phase import ScientificPhase, require_action
from ._state import (
    WorkflowStatus,
    _now_iso,
    add_job,
    create_state,
    get_input_fingerprints,
    get_job,
    get_job_by_step,
    get_workflow_summary,
    read_state,
    save_job_accounting,
    set_error,
    set_input_fingerprints,
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
    filename : str or ValueFrom
        Filename (basename) to pull from the upstream workflow's
        output directory.  Can be a plain string when the filename is
        known at definition time, or a :class:`ValueFrom` for names
        that are only determined at runtime (e.g. the optimised
        Hamiltonian whose step number depends on convergence).

    Examples
    --------
    Static filename (step number known in advance)::

        Container(
            label="mcmc-run",
            dirname="mcmc",
            input_files=[FileFrom("vmc-opt", "hamiltonian_data_opt_step_9.h5")],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=MCMC_Workflow(...),
        )

    Dynamic filename (resolved from upstream ``output_values``)::

        Container(
            label="mcmc-run",
            dirname="mcmc",
            input_files=[
                FileFrom("vmc-opt",
                         ValueFrom("vmc-opt", "optimized_hamiltonian")),
            ],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=MCMC_Workflow(...),
        )

    See Also
    --------
    ValueFrom : Declare a scalar-value dependency.
    Launcher  : Resolves ``FileFrom`` / ``ValueFrom`` at launch time.
    """

    def __init__(self, label: str, filename):
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
        See the *Output Values* section of each workflow class for
        available keys:

        * :class:`VMC_Workflow` — ``optimized_hamiltonian``,
          ``energy``, ``energy_error``, ``checkpoint``, …
        * :class:`MCMC_Workflow` — ``energy``, ``energy_error``,
          ``restart_chk``, ``forces``, …
        * :class:`LRDMC_Workflow` — ``energy``, ``energy_error``,
          ``alat``, ``restart_chk``, ``forces``, …
        * :class:`LRDMC_Ext_Workflow` — ``extrapolated_energy``,
          ``extrapolated_energy_error``, ``per_alat_results``, …

    Examples
    --------
    Feed the MCMC energy into an LRDMC workflow as ``trial_energy``::

        LRDMC_Workflow(
            trial_energy=ValueFrom("mcmc-run", "energy"),
            ...
        )

    Pass the VMC-optimised Hamiltonian dynamically via ``FileFrom``::

        FileFrom("vmc-opt", ValueFrom("vmc-opt", "optimized_hamiltonian"))

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
    this class and overrides :meth:`configure` and :meth:`run`.

    Parameters
    ----------
    project_dir : str, optional
        Absolute path to the working directory for this workflow.
        When *None* (the default), ``project_dir`` is set to the
        process CWD at the time :meth:`run` is first called.
        :class:`Container` sets this explicitly before launching the
        inner workflow.

    Attributes
    ----------
    status : WorkflowStatus
        Current lifecycle status.
    phase : ScientificPhase
        Current scientific phase.
    output_files : list[str]
        Filenames produced by the workflow (populated after run).
    output_values : dict
        Scalar results (energy, error, …) produced by the workflow.
    project_dir : str or None
        Working directory for file I/O.  Resolved to an absolute path.

    Notes
    -----
    **Subclass contract:**

    * Override :meth:`configure` and :meth:`run`, return
      ``(status, output_files, output_values)`` from ``run()``.
    * Call ``super().__init__()`` in your constructor.

    Examples
    --------
    Minimal custom workflow::

        class MyWorkflow(Workflow):
            def configure(self):
                return {"param": 42}

            async def run(self):
                # ... do work ...
                self.status = WorkflowStatus.COMPLETED
                return self.status, ["result.h5"], {"energy": -1.23}
    """

    def __init__(self, project_dir: Optional[str] = None):
        self.status: WorkflowStatus = WorkflowStatus.PENDING
        self.phase: ScientificPhase = ScientificPhase.INIT
        self.output_files: List[str] = []
        self.output_values: dict = {}
        self.project_dir: Optional[str] = os.path.abspath(project_dir) if project_dir else None
        self._bg_task: Optional[asyncio.Task] = None

    # ── Filename generation (per-job run_id) ──────────────────────

    @staticmethod
    def _new_run_id() -> str:
        """Generate a short random identifier for a single job."""
        return uuid.uuid4().hex[:8]

    def _input_filename(self, index: int, run_id: str) -> str:
        """Input TOML filename: ``input_{jobname}_{index}_{run_id}.toml``."""
        return f"input_{self.jobname}_{index}_{run_id}.toml"

    def _output_filename(self, index: int, run_id: str) -> str:
        """Stdout-capture filename: ``output_{jobname}_{index}_{run_id}.out``."""
        return f"output_{self.jobname}_{index}_{run_id}.out"

    def _submit_script_name(self, run_id: str) -> str:
        return f"submit_{run_id}.sh"

    def _ensure_project_dir(self):
        """Lazily resolve *project_dir* to CWD when not set explicitly."""
        if self.project_dir is None:
            self.project_dir = os.path.abspath(os.getcwd())

    # ── configure / run (new primary interface) ─────────────────────

    def configure(self) -> dict:
        """Validate parameters and generate inputs (no execution).

        Override in subclass.  Returns a summary dict.
        """
        return {}

    async def run(self) -> tuple:
        """Execute the workflow (submit → poll → fetch → convergence loop).

        Override in subclass.  Must return
        ``(status, output_files, output_values)``.

        Intermediate progress is written to ``workflow_state.toml``
        via :func:`update_status`.
        """
        self._ensure_project_dir()
        return self.status, self.output_files, self.output_values

    # ── Full lifecycle (backward-compatible) ──────────────────────

    async def async_launch(self):
        """Run configure() + run().  Backward-compatible entry point."""
        self._ensure_project_dir()
        self.configure()
        return await self.run()

    def launch(self):
        return asyncio.run(self.async_launch())

    # ── Phased execution (MCP interactive mode) ───────────────────
    #
    # Used by MCP tools: submit(action) → poll() → collect().
    # submit() starts run() as a background asyncio.Task.
    #
    # Usage pattern::
    #
    #     info = await wf.async_submit(action="run_vmc")
    #     while (status := await wf.async_poll()) == "running":
    #         await asyncio.sleep(60)
    #     result = await wf.async_collect()

    async def async_submit(self, action: str = "run") -> dict:
        """Start the workflow in the background and return tracking info.

        Parameters
        ----------
        action : str
            MCP tool name (e.g. ``"run_vmc"``).  Checked against
            :func:`allowed_actions` for the current phase and status.

        Returns
        -------
        dict
            ``{"status": "submitted", "project_dir": ...}``.

        Raises
        ------
        ValueError
            If *action* is not allowed in the current phase/status.
        RuntimeError
            If the workflow has already been submitted.
        """
        if self._bg_task is not None and not self._bg_task.done():
            raise RuntimeError("Workflow already submitted and still running.")
        require_action(action, self.phase, self.status)
        self._ensure_project_dir()
        self.configure()
        self._bg_task = asyncio.create_task(self.run())
        return {"status": "submitted", "project_dir": self.project_dir}

    async def async_poll(self) -> dict:
        """Check whether the submitted workflow has completed.

        Returns
        -------
        dict
            Status dict.  Includes ``get_workflow_summary()`` when
            the task is still running.
        """
        if self._bg_task is None:
            return {"status": "not_submitted"}
        if not self._bg_task.done():
            summary = get_workflow_summary(self.project_dir) if self.project_dir else {}
            return {"status": "running", **summary}
        if self._bg_task.exception() is not None:
            return {"status": "failed", "error": str(self._bg_task.exception())}
        return {"status": "completed"}

    async def async_collect(self) -> dict:
        """Collect results from the completed workflow.

        Returns
        -------
        dict
            ``{"status": ..., "output_files": [...], **output_values}``.

        Raises
        ------
        RuntimeError
            If the workflow was not submitted or is still running.
        Exception
            Re-raises the original exception if the workflow failed.
        """
        if self._bg_task is None:
            raise RuntimeError("No workflow has been submitted. Call async_submit() first.")
        if not self._bg_task.done():
            raise RuntimeError("Workflow is still running. Call async_poll() to check status.")
        exc = self._bg_task.exception()
        if exc is not None:
            raise exc
        status, output_files, output_values = self._bg_task.result()
        return {
            "status": status,
            "output_files": output_files,
            **output_values,
        }

    # ── Common job helpers (used by VMC / MCMC / LRDMC) ───────────
    #
    # These methods require the following attributes on *self*:
    #   server_machine_name, hamiltonian_file, queue_label,
    #   jobname, poll_interval
    # They are therefore only callable from concrete subclasses that
    # set those attributes.

    @staticmethod
    def _collect_job_acct(job: JobSubmission, work_dir: str, input_file: str) -> None:
        """Run scheduler accounting, save to file, and attach to [[jobs]] record."""
        result = job.job_acct()
        if result is None:
            return
        command, stdout, stderr = result
        acct_cmd, acct_file = save_job_accounting(
            work_dir,
            command=command,
            stdout=stdout,
            stderr=stderr,
            job_id=job.job_number or "",
        )
        update_job(work_dir, input_file, job_acct_command=acct_cmd, job_acct_file=acct_file)

    async def _submit_and_wait(
        self,
        input_file,
        output_file,
        work_dir,
        extra_from_objects=None,
        fetch_from_objects=None,
        queue_label=None,
        step=None,
        run_id="",
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
        step : int, optional
            Step index (0 for pilot, 1+ for production).  Used for
            cross-run continuation detection.
        run_id : str
            Per-job identifier used for scheduler script and stdout/stderr
            naming.
        """
        if fetch_from_objects is None:
            fetch_from_objects = ["*.h5", output_file]

        # Include scheduler stdout/stderr in fetch list when queuing.
        # These are non-essential: warn (not error) if missing on server.
        optional_fetch = []
        job_tmp = self._make_job(input_file, output_file, queue_label=queue_label, run_id=run_id)
        if job_tmp.server_machine.queuing:
            for jf in (job_tmp.job_stdout, job_tmp.job_stderr):
                if jf and jf not in fetch_from_objects:
                    fetch_from_objects.append(jf)
                    optional_fetch.append(jf)

        # ── Restart detection via job history ─────────────────────
        if step is not None:
            recorded = get_job_by_step(work_dir, step)
        else:
            recorded = get_job(work_dir, input_file)

        if recorded.get("status") == "fetched":
            logger.info(f"  {input_file}: already fetched (step {step}). Skipping.")
            return

        if recorded.get("status") == "completed":
            logger.info(f"  {input_file}: completed but not fetched. Fetching...")
            job = self._make_job(input_file, output_file, queue_label=queue_label, run_id=run_id)
            job.fetch_job(
                from_objects=fetch_from_objects, exclude_patterns=[], work_dir=work_dir, optional_patterns=optional_fetch
            )
            update_job(work_dir, input_file, status="fetched", fetched_at=_now_iso())
            return

        if recorded.get("status") == "submitted":
            stored_job_id = recorded.get("job_id")
            logger.info(f"  Resuming previously submitted job {stored_job_id}")
            job = self._make_job(input_file, output_file, queue_label=queue_label, run_id=run_id)
            job.job_number = stored_job_id
            while job.jobcheck():
                logger.info(f"  Job {stored_job_id} still running, waiting {self.poll_interval}s...")
                await asyncio.sleep(self.poll_interval)
            logger.info("  Job completed.")
            update_job(work_dir, input_file, status="completed", completed_at=_now_iso())
            self._collect_job_acct(job, work_dir, input_file)
            job.fetch_job(
                from_objects=fetch_from_objects, exclude_patterns=[], work_dir=work_dir, optional_patterns=optional_fetch
            )
            update_job(work_dir, input_file, status="fetched", fetched_at=_now_iso())
            return

        # ── New submission ────────────────────────────────────────
        job = self._make_job(input_file, output_file, queue_label=queue_label, run_id=run_id)
        submit_sh = self._submit_script_name(run_id)
        job.generate_script(submission_script=submit_sh, work_dir=work_dir)

        from_objects = [input_file, self.hamiltonian_file, submit_sh]
        if extra_from_objects:
            from_objects.extend(extra_from_objects)
        submitted, job_number = job.job_submit(submission_script=submit_sh, from_objects=from_objects, work_dir=work_dir)
        if not submitted:
            raise RuntimeError("Job submission failed (queue limit or error).")

        logger.info(f"  Job submitted: {job_number}")
        add_job(
            work_dir,
            input_file=input_file,
            output_file=output_file,
            job_id=str(job_number) if job_number else "local",
            server_machine=self.server_machine_name,
            step=step,
            run_id=run_id,
            job_stdout=job.job_stdout if job.server_machine.queuing else "",
            job_stderr=job.job_stderr if job.server_machine.queuing else "",
        )

        while job.jobcheck():
            logger.info(f"  Job {job_number} still running, waiting {self.poll_interval}s...")
            await asyncio.sleep(self.poll_interval)

        logger.info("  Job completed.")
        update_job(work_dir, input_file, status="completed", completed_at=_now_iso())

        # Collect scheduler accounting before fetch
        self._collect_job_acct(job, work_dir, input_file)

        job.fetch_job(from_objects=fetch_from_objects, exclude_patterns=[], work_dir=work_dir, optional_patterns=optional_fetch)
        update_job(work_dir, input_file, status="fetched", fetched_at=_now_iso())

    def _make_job(self, input_file, output_file, queue_label=None, run_id=""):
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
            run_id=run_id,
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
        self._bg_task: Optional[asyncio.Task] = None

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
            logger.info(f"[{self.label}] Project dir exists, ensuring input files are present.")
            self._ensure_input_files()
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

    @staticmethod
    def _dst_basename(src: str, rename_list: list, index: int) -> str:
        """Compute the destination basename for the *index*-th input file.

        If *rename_list* has a non-``None`` entry at *index*, that value
        is used as the destination name; otherwise the source basename
        is preserved.
        """
        if index < len(rename_list) and rename_list[index] is not None:
            return os.path.basename(str(rename_list[index]))
        return os.path.basename(src)

    def _copy_input_files(self):
        """Copy input files into the project directory.

        Raises
        ------
        FileNotFoundError
            If a required input file or directory does not exist.
        """
        for i, src in enumerate(self.input_files):
            src = str(src)  # resolve pathlib objects
            # Resolve relative paths against root_dir
            if not os.path.isabs(src):
                src = os.path.join(self.root_dir, src)
            dst_name = self._dst_basename(src, self.rename_input_files, i)

            dst = os.path.join(self.project_dir, dst_name)

            if os.path.isfile(src):
                shutil.copy(src, dst)
            elif os.path.isdir(src):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                raise FileNotFoundError(f"[{self.label}] Required input not found: {src}")

    def _ensure_input_files(self):
        """Copy any missing input files into an existing project directory.

        Unlike :meth:`_copy_input_files`, this does *not* overwrite files
        that already exist in the project directory — it only fills in
        the gaps (e.g. after a failed first run that created the
        directory but never completed the copy).
        """
        for i, src in enumerate(self.input_files):
            src = str(src)
            if not os.path.isabs(src):
                src = os.path.join(self.root_dir, src)
            dst_name = self._dst_basename(src, self.rename_input_files, i)
            dst = os.path.join(self.project_dir, dst_name)

            if os.path.exists(dst):
                continue  # already present

            if os.path.isfile(src):
                shutil.copy(src, dst)
                logger.info(f"[{self.label}] Copied missing input: {dst_name}")
            elif os.path.isdir(src):
                shutil.copytree(src, dst)
                logger.info(f"[{self.label}] Copied missing input dir: {dst_name}")
            else:
                raise FileNotFoundError(f"[{self.label}] Required input not found: {src}")

    def _validate_input_files(self, proj: str):
        """Verify that all files listed in ``input_files`` exist in *proj*.

        Only the entries in ``self.input_files`` (after rename) are
        checked.  Workflow-internal files (e.g. ``hamiltonian_file``)
        are **not** validated here because some workflows (e.g.
        ``WF_Workflow``) *produce* them rather than consume them.

        Raises
        ------
        FileNotFoundError
            With a clear message listing all missing files, raised
            *before* any job is submitted.
        """
        missing = []

        for i, src in enumerate(self.input_files):
            dst_name = self._dst_basename(str(src), self.rename_input_files, i)
            if not os.path.exists(os.path.join(proj, dst_name)):
                missing.append(dst_name)

        if missing:
            raise FileNotFoundError(
                f"[{self.label}] Required file(s) missing in '{self.dirname}/' "
                f"before workflow launch: {missing}. "
                f"Check that input_files and rename_input_files are "
                f"configured correctly."
            )

    def _compute_input_fingerprints(self) -> dict[str, dict]:
        """Return ``{basename: {sha256: hex_digest}}`` for each resolved input file."""
        fingerprints: dict[str, dict] = {}
        for i, src in enumerate(self.input_files):
            src = str(src)
            if not os.path.isabs(src):
                src = os.path.join(self.root_dir, src)
            key = self._dst_basename(src, self.rename_input_files, i)
            if os.path.exists(src):
                h = hashlib.sha256()
                with open(src, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                fingerprints[key] = {"sha256": h.hexdigest()}
        return fingerprints

    def _check_input_staleness(self, proj: str) -> bool:
        """Compare current input fingerprints against the recorded ones.

        Returns ``True`` if any input has changed (stale), ``False``
        if everything matches or no fingerprints were recorded.
        """
        recorded = get_input_fingerprints(proj)
        if not recorded:
            return False  # no fingerprints recorded — cannot check
        current = self._compute_input_fingerprints()
        for name, cur_fp in current.items():
            rec_fp = recorded.get(name)
            if rec_fp is None:
                # New input file not in original — treat as stale
                return True
            if cur_fp.get("sha256") != rec_fp.get("sha256"):
                logger.warning(
                    f"[{self.label}] Input '{name}' has changed since last run "
                    f"(sha256: {rec_fp.get('sha256', '?')[:12]}… → {cur_fp.get('sha256', '?')[:12]}…)."
                )
                return True
        return False

    # ── Launch ────────────────────────────────────────────────────

    async def async_launch(self):
        proj = os.path.abspath(self.project_dir)

        self._prepare()

        # Check if already completed or running (resume)
        state = read_state(proj)
        prev_status = state.get("workflow", {}).get("status")

        if prev_status in ("completed", "running"):
            if self._check_input_staleness(proj):
                logger.warning(
                    f"[{self.label}] Inputs have changed since the previous run. "
                    f"Delete '{self.dirname}/' to re-run with the updated inputs."
                )

        # Record input-file fingerprints after staleness check but
        # before any execution, so that even interrupted runs have a
        # baseline for the next invocation.
        set_input_fingerprints(proj, self._compute_input_fingerprints())

        if prev_status == "completed":
            logger.info(f"[{self.label}] Already completed, no re-run.")
            self.status = WorkflowStatus.COMPLETED
            self._collect_outputs()
            return self.status, self.output_files, self.output_values

        # Validate required files before running.
        self._validate_input_files(proj)

        # Run the workflow — pass project_dir explicitly instead of
        # relying on os.chdir().
        update_status(proj, WorkflowStatus.RUNNING)
        self.workflow.project_dir = proj

        try:
            self.status, self.output_files, self.output_values = await self.workflow.async_launch()
        except Exception as e:
            set_error(proj, str(e))
            update_status(proj, WorkflowStatus.FAILED)
            raise

        # Write completion — but only if the workflow did not fail.
        if self.status != WorkflowStatus.FAILED:
            result_fields = {}
            for k, v in self.output_values.items():
                result_fields[f"result_{k}"] = v
            update_status(proj, WorkflowStatus.COMPLETED, **result_fields)
        else:
            error_msg = self.output_values.get("error", f"workflow returned status={self.status}")
            set_error(proj, error_msg)
            update_status(proj, WorkflowStatus.FAILED)
            logger.warning(f"[{self.label}] {error_msg}")

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

    # ── Phased execution (delegates to inner Workflow) ────────────

    async def async_submit(self, action: str = "run") -> dict:
        """Start the container's workflow in the background.

        Prepares the project directory, copies input files, then
        starts the inner workflow via ``asyncio.Task``.  Use
        :meth:`async_poll` and :meth:`async_collect` to monitor
        and retrieve results.

        Parameters
        ----------
        action : str
            MCP tool name for action validation.

        Returns
        -------
        dict
            ``{"status": "submitted", "label": ..., "project_dir": ...}``.
        """
        if self._bg_task is not None and not self._bg_task.done():
            raise RuntimeError(f"[{self.label}] Already submitted and still running.")
        self._bg_task = asyncio.create_task(self.async_launch())
        return {
            "status": "submitted",
            "label": self.label,
            "project_dir": self.project_dir,
        }

    async def async_poll(self) -> dict:
        """Check whether the container's workflow has completed.

        Returns
        -------
        dict
            Status dict with ``get_workflow_summary()`` when running.
        """
        if self._bg_task is None:
            return {"status": "not_submitted"}
        if not self._bg_task.done():
            summary = get_workflow_summary(self.project_dir) if self.project_dir else {}
            return {"status": "running", **summary}
        if self._bg_task.exception() is not None:
            return {"status": "failed", "error": str(self._bg_task.exception())}
        return {"status": "completed"}

    async def async_collect(self) -> dict:
        """Collect results from the completed container workflow.

        Returns
        -------
        dict
            ``{"status": ..., "label": ..., "output_files": [...],
            **output_values}``.

        Raises
        ------
        RuntimeError
            If not submitted or still running.
        Exception
            Re-raises the original exception if the workflow failed.
        """
        if self._bg_task is None:
            raise RuntimeError(f"[{self.label}] Not submitted. Call async_submit() first.")
        if not self._bg_task.done():
            raise RuntimeError(f"[{self.label}] Still running. Call async_poll() to check.")
        exc = self._bg_task.exception()
        if exc is not None:
            raise exc
        status, output_files, output_values = self._bg_task.result()
        return {
            "status": status,
            "label": self.label,
            "output_files": output_files,
            **output_values,
        }
