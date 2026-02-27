"""VMC_Workflow — Jastrow / orbital optimization via ``jqmc`` (job_type=vmc).

Generates a VMC input TOML, submits the ``jqmc`` binary on a remote (or local)
machine, monitors until completion, and fetches the results.  The optimized
``hamiltonian_data_opt_step_N.h5`` files and checkpoint are collected as outputs.
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
import glob
import os
import re
import time
from logging import getLogger
from typing import Optional

from ._error_estimator import (
    _format_duration,
    estimate_required_steps,
    parse_net_time,
    suffixed_name,
)
from ._input_generator import generate_input_toml, resolve_with_defaults
from ._job import JobSubmission, get_num_mpi, load_queue_data
from ._state import _now_iso, add_job, get_estimation, get_job, set_estimation, update_job
from .workflow import Workflow

logger = getLogger("jqmc-workflow").getChild(__name__)


class VMC_Workflow(Workflow):
    """VMC (Variational Monte Carlo) Jastrow / orbital optimisation workflow.

    Generates a ``job_type=vmc`` input TOML, submits ``jqmc``, monitors
    until completion, and collects the optimised
    ``hamiltonian_data_opt_step_N.h5`` files and checkpoint.

    The workflow operates in two phases:

    1. **Pilot VMC run** (``_0``) — Runs a short optimisation with
       ``pilot_vmc_steps`` optimisation steps and ``pilot_mcmc_steps``
       MCMC steps per step.  The statistical error of the *last*
       optimisation step is used to estimate the MCMC steps per
       opt-step required to achieve ``target_error`` via
       $\\sigma \\propto 1/\\sqrt{N}$.
    2. **Production VMC runs** (``_1``, ``_2``, …) — Full optimisation
       with ``num_opt_steps`` and the estimated MCMC steps per step.
       If a run is interrupted by the wall-time limit, the next
       continuation restarts from the checkpoint.  At most
       ``max_continuation`` runs are attempted.

    Parameters
    ----------
    server_machine_name : str
        Name of the target machine (must be configured in ``~/.jqmc_setting/``).
    num_opt_steps : int
        Number of optimization iterations for production runs.
    hamiltonian_file : str
        Input ``hamiltonian_data.h5``.
    input_file : str
        Name of the generated TOML input file.
    output_file : str
        Name of the stdout capture file.
    queue_label : str
        Queue/partition label from ``queue_data.toml``.
    jobname : str
        Job name for the scheduler.
    number_of_walkers : int
        Walkers per MPI process.
    max_time : int
        Wall-time limit in seconds.
    Dt : float, optional
        MCMC step size (bohr).  Default from ``jqmc_miscs``.
    epsilon_AS : float, optional
        Attacalite-Sorella regularization parameter.  Default from ``jqmc_miscs``.
    num_mcmc_per_measurement : int, optional
        MCMC updates per measurement.  Default from ``jqmc_miscs``.
    num_mcmc_warmup_steps : int, optional
        Warmup measurement steps to discard.  Default from ``jqmc_miscs``.
    num_mcmc_bin_blocks : int, optional
        Binning blocks.  Default from ``jqmc_miscs``.
    wf_dump_freq : int, optional
        Wavefunction dump frequency.  Default from ``jqmc_miscs``.
    opt_J1_param : bool, optional
        Optimize J1 Jastrow parameters.  Default from ``jqmc_miscs``.
    opt_J2_param : bool, optional
        Optimize J2 Jastrow parameters.  Default from ``jqmc_miscs``.
    opt_J3_param : bool, optional
        Optimize J3 Jastrow parameters.  Default from ``jqmc_miscs``.
    opt_JNN_param : bool, optional
        Optimize neural-network Jastrow parameters.  Default from ``jqmc_miscs``.
    opt_lambda_param : bool, optional
        Optimize lambda (geminal) parameters.  Default from ``jqmc_miscs``.
    opt_with_projected_MOs : bool, optional
        Optimize in a restricted MO space.  Default from ``jqmc_miscs``.
    num_param_opt : int, optional
        Number of parameters to optimize (0 = all).  Default from ``jqmc_miscs``.
    optimizer_kwargs : dict, optional
        Optimizer configuration dict.  Default from ``jqmc_miscs``.
    mcmc_seed : int, optional
        Random seed for MCMC.  Default from ``jqmc_miscs``.
    verbosity : str, optional
        Verbosity level.  Default from ``jqmc_miscs``.
    poll_interval : int
        Seconds between job-status polls.
    target_error : float
        Target statistical error (Ha) per optimization step.
    pilot_mcmc_steps : int
        MCMC steps per opt-step for the pilot run.
    pilot_vmc_steps : int
        Number of optimization steps in the pilot run (small; just
        enough to estimate the error bar).
    pilot_queue_label : str, optional
        Queue label for the pilot run.  Defaults to *queue_label*.
        Use a shorter/smaller queue for the pilot to save resources.
    max_continuation : int
        Maximum number of production runs after the pilot.

    Examples
    --------
    Standalone launch::

        wf = VMC_Workflow(
            server_machine_name="cluster",
            num_opt_steps=20,
            target_error=0.001,
            pilot_mcmc_steps=50,
            pilot_vmc_steps=5,
            number_of_walkers=8,
        )
        status, files, values = wf.launch()
        print(values["optimized_hamiltonian"])

    As part of a :class:`Launcher` pipeline::

        enc = Container(
            label="vmc",
            dirname="01_vmc",
            input_files=[FileFrom("wf", "hamiltonian_data.h5")],
            workflow=VMC_Workflow(
                server_machine_name="cluster",
                num_opt_steps=20,
                target_error=0.001,
            ),
        )

    Notes
    -----
    * The pilot uses a small number of opt steps (``pilot_vmc_steps``)
      just to estimate the error.  The real optimisation happens in
      production runs with the full ``num_opt_steps``.
    * The estimation is stored in ``workflow_state.toml`` under
      ``[estimation]``; on re-entrance the pilot is skipped.

    See Also
    --------
    MCMC_Workflow : VMC production sampling (job_type=mcmc).
    LRDMC_Workflow : Diffusion Monte Carlo (job_type=lrdmc).
    WF_Workflow : TREXIO → hamiltonian_data conversion.
    """

    def __init__(
        self,
        server_machine_name: str = "localhost",
        num_opt_steps: int = 20,
        hamiltonian_file: str = "hamiltonian_data.h5",
        input_file: str = "input.toml",
        output_file: str = "out.o",
        queue_label: str = "default",
        jobname: str = "jqmc-vmc",
        number_of_walkers: int = 4,
        max_time: int = 86400,
        # -- [vmc] section parameters --
        Dt: Optional[float] = None,
        epsilon_AS: Optional[float] = None,
        num_mcmc_per_measurement: Optional[int] = None,
        num_mcmc_warmup_steps: Optional[int] = None,
        num_mcmc_bin_blocks: Optional[int] = None,
        wf_dump_freq: Optional[int] = None,
        opt_J1_param: Optional[bool] = None,
        opt_J2_param: Optional[bool] = None,
        opt_J3_param: Optional[bool] = None,
        opt_JNN_param: Optional[bool] = None,
        opt_lambda_param: Optional[bool] = None,
        opt_with_projected_MOs: Optional[bool] = None,
        num_param_opt: Optional[int] = None,
        optimizer_kwargs: Optional[dict] = None,
        # -- [control] section parameters --
        mcmc_seed: Optional[int] = None,
        verbosity: Optional[str] = None,
        # -- workflow parameters --
        poll_interval: int = 60,
        target_error: float = 0.001,
        pilot_mcmc_steps: int = 50,
        pilot_vmc_steps: int = 5,
        pilot_queue_label: Optional[str] = None,
        max_continuation: int = 5,
    ):
        super().__init__()
        self.server_machine_name = server_machine_name
        self.num_opt_steps = num_opt_steps
        self.hamiltonian_file = hamiltonian_file
        self.input_file = input_file
        self.output_file = output_file
        self.queue_label = queue_label
        self.jobname = jobname
        self.number_of_walkers = number_of_walkers
        self.max_time = max_time
        # [vmc] section
        self.Dt = Dt
        self.epsilon_AS = epsilon_AS
        self.num_mcmc_per_measurement = num_mcmc_per_measurement
        self.num_mcmc_warmup_steps = num_mcmc_warmup_steps
        self.num_mcmc_bin_blocks = num_mcmc_bin_blocks
        self.wf_dump_freq = wf_dump_freq
        self.opt_J1_param = opt_J1_param
        self.opt_J2_param = opt_J2_param
        self.opt_J3_param = opt_J3_param
        self.opt_JNN_param = opt_JNN_param
        self.opt_lambda_param = opt_lambda_param
        self.opt_with_projected_MOs = opt_with_projected_MOs
        self.num_param_opt = num_param_opt
        self.optimizer_kwargs = optimizer_kwargs
        # [control] section
        self.mcmc_seed = mcmc_seed
        self.verbosity = verbosity
        # workflow
        self.poll_interval = poll_interval
        self.target_error = target_error
        self.pilot_mcmc_steps = pilot_mcmc_steps
        self.pilot_vmc_steps = pilot_vmc_steps
        self.pilot_queue_label = pilot_queue_label or queue_label
        self.max_continuation = max_continuation

    # ── Input generation ──────────────────────────────────────────

    def _generate_input(
        self,
        num_mcmc_steps,
        num_opt_steps,
        input_file,
        restart=False,
        restart_chk=None,
    ):
        """Generate a VMC TOML input file.

        Parameters
        ----------
        num_mcmc_steps : int
            MCMC measurement steps per optimization step.
        num_opt_steps : int
            Number of optimization iterations.
        input_file : str
            Output filename for the TOML.
        restart : bool
            Whether to restart from a checkpoint.
        restart_chk : str, optional
            Checkpoint file to restart from.
        """
        control_ov = resolve_with_defaults(
            "control",
            {
                "number_of_walkers": self.number_of_walkers,
                "max_time": self.max_time,
                "hamiltonian_h5": self.hamiltonian_file,
                "restart": restart,
                "mcmc_seed": self.mcmc_seed,
                "verbosity": self.verbosity,
            },
        )
        if restart and restart_chk:
            control_ov["restart_chk"] = restart_chk
        vmc_ov = resolve_with_defaults(
            "vmc",
            {
                "num_opt_steps": num_opt_steps,
                "num_mcmc_steps": num_mcmc_steps,
                "Dt": self.Dt,
                "epsilon_AS": self.epsilon_AS,
                "num_mcmc_per_measurement": self.num_mcmc_per_measurement,
                "num_mcmc_warmup_steps": self.num_mcmc_warmup_steps,
                "num_mcmc_bin_blocks": self.num_mcmc_bin_blocks,
                "wf_dump_freq": self.wf_dump_freq,
                "opt_J1_param": self.opt_J1_param,
                "opt_J2_param": self.opt_J2_param,
                "opt_J3_param": self.opt_J3_param,
                "opt_JNN_param": self.opt_JNN_param,
                "opt_lambda_param": self.opt_lambda_param,
                "opt_with_projected_MOs": self.opt_with_projected_MOs,
                "num_param_opt": self.num_param_opt,
                "optimizer_kwargs": self.optimizer_kwargs,
            },
        )
        overrides = {
            "control": control_ov,
            "vmc": vmc_ov,
        }
        generate_input_toml(
            job_type="vmc",
            overrides=overrides,
            filename=input_file,
        )

    # ── Submit / poll / fetch ─────────────────────────────────────

    async def _submit_and_wait(
        self,
        input_file,
        output_file,
        fetch_from_objects=None,
        queue_label=None,
    ):
        """Create job, submit, poll until done, fetch results.

        All decisions are driven by ``workflow_state.toml``:

        * ``fetched``   -- skip entirely (already done)
        * ``completed`` -- fetch results only
        * ``submitted`` -- resume waiting, then fetch
        * no record     -- submit a new job

        Parameters
        ----------
        input_file : str
            Path to the input TOML file.
        output_file : str
            Name of stdout capture file.
        fetch_from_objects : list, optional
            Glob patterns for files to fetch.  Defaults to
            ``["*.h5", "*.chk", output_file]``.
        """
        if fetch_from_objects is None:
            fetch_from_objects = ["*.h5", "*.chk", output_file]
        cwd = os.path.abspath(os.getcwd())

        # ── Restart detection via job history ─────────────────────
        recorded = get_job(cwd, input_file)

        if recorded.get("status") == "fetched":
            logger.info(f"  {input_file}: already fetched. Skipping.")
            return

        if recorded.get("status") == "completed":
            logger.info(f"  {input_file}: completed but not fetched. Fetching...")
            os.chdir(cwd)
            job = self._make_job(input_file, output_file, queue_label=queue_label)
            job.fetch_job(from_objects=fetch_from_objects, exclude_patterns=[])
            update_job(cwd, input_file, status="fetched", fetched_at=_now_iso())
            return

        if recorded.get("status") == "submitted":
            stored_job_id = recorded.get("job_id")
            logger.info(f"  Resuming previously submitted job {stored_job_id}")
            job = self._make_job(input_file, output_file, queue_label=queue_label)
            job.job_number = stored_job_id
            while job.jobcheck():
                logger.info(f"  Job {stored_job_id} still running, waiting {self.poll_interval}s...")
                await asyncio.sleep(self.poll_interval)
                os.chdir(cwd)
            logger.info("  Job completed.")
            update_job(cwd, input_file, status="completed", completed_at=_now_iso())
            os.chdir(cwd)
            job.fetch_job(from_objects=fetch_from_objects, exclude_patterns=[])
            update_job(cwd, input_file, status="fetched", fetched_at=_now_iso())
            return

        # ── New submission ────────────────────────────────────────
        job = self._make_job(input_file, output_file, queue_label=queue_label)
        job.generate_script()

        from_objects = [input_file, self.hamiltonian_file, "submit.sh"]
        submitted, job_number = job.job_submit(from_objects=from_objects)
        if not submitted:
            raise RuntimeError("Job submission failed (queue limit or error).")

        logger.info(f"  Job submitted: {job_number}")
        add_job(
            cwd,
            input_file=input_file,
            output_file=output_file,
            job_id=str(job_number) if job_number else "local",
            server_machine=self.server_machine_name,
        )

        while job.jobcheck():
            logger.info(f"  Job {job_number} still running, waiting {self.poll_interval}s...")
            await asyncio.sleep(self.poll_interval)
            os.chdir(cwd)

        logger.info("  Job completed.")
        update_job(cwd, input_file, status="completed", completed_at=_now_iso())
        os.chdir(cwd)
        job.fetch_job(from_objects=fetch_from_objects, exclude_patterns=[])
        update_job(cwd, input_file, status="fetched", fetched_at=_now_iso())

    def _make_job(self, input_file, output_file, queue_label=None):
        """Create a JobSubmission with current workflow settings."""
        return JobSubmission(
            server_machine_name=self.server_machine_name,
            input_file=input_file,
            output_file=output_file,
            queue_label=queue_label or self.queue_label,
            jobname=self.jobname,
        )

    # ── Launch ────────────────────────────────────────────────────

    async def async_launch(self):
        """Run the VMC optimization workflow with automatic step estimation.

        1. Pilot VMC run in ``_pilot/`` with ``pilot_vmc_steps`` opt steps
           and ``pilot_mcmc_steps`` MCMC steps to estimate the required
           MCMC steps per opt step (skipped on continuation).
           May use a different queue (``pilot_queue_label``).
        2. Production VMC runs (``_1``, ``_2``, ...) start from scratch
           with the full ``num_opt_steps`` and estimated MCMC steps until
           all optimization steps complete or ``max_continuation`` is
           reached.
        """
        # Save absolute CWD — other async tasks may chdir while we await.
        _wd = os.path.abspath(os.getcwd())

        # ── Phase 0: pilot estimation (skip on continuation) ──────
        estimation = get_estimation(_wd)

        if estimation.get("estimated_mcmc_steps") is not None:
            estimated_mcmc_steps = int(estimation["estimated_mcmc_steps"])
            logger.info(
                "Estimation already done (continuation): "
                f"estimated_mcmc_steps={estimated_mcmc_steps} per opt step. "
                "Skipping pilot."
            )
        else:
            # ── Run pilot in a subdirectory ───────────────────────
            pilot_dir = os.path.join(_wd, "_pilot")
            os.makedirs(pilot_dir, exist_ok=True)

            h5_src = os.path.join(_wd, self.hamiltonian_file)
            h5_dst = os.path.join(pilot_dir, self.hamiltonian_file)
            if os.path.isfile(h5_src) and not os.path.exists(h5_dst):
                os.symlink(h5_src, h5_dst)

            os.chdir(pilot_dir)

            input_0 = suffixed_name(self.input_file, 0)
            output_0 = suffixed_name(self.output_file, 0)

            recorded_0 = get_job(pilot_dir, input_0)
            if recorded_0.get("status") not in ("submitted", "completed", "fetched"):
                self._generate_input(
                    self.pilot_mcmc_steps,
                    self.pilot_vmc_steps,
                    input_0,
                )
            else:
                logger.info(f"  {input_0}: already {recorded_0['status']}. Resuming...")
            logger.info("")
            logger.info("-- VMC Phase 0: Pilot " + "-" * 28)
            logger.info(
                f"  {input_0}: {self.pilot_vmc_steps} opt steps x {self.pilot_mcmc_steps} MCMC steps/step (queue: {self.pilot_queue_label})"
            )

            pilot_t0 = time.monotonic()
            await self._submit_and_wait(input_0, output_0, queue_label=self.pilot_queue_label)
            os.chdir(pilot_dir)
            pilot_wall_sec = time.monotonic() - pilot_t0

            # Parse the last optimization step's error from pilot output
            pilot_energy, pilot_error = self._parse_last_opt_energy(output_0)
            if pilot_error is None:
                raise RuntimeError(
                    f"Could not parse energy error from the last optimization step of the pilot VMC output ({output_0})."
                )

            logger.info(
                f"Pilot VMC last-step energy: {pilot_energy} +- {pilot_error} Ha (from {self.pilot_mcmc_steps} MCMC steps/step)"
            )

            # Walker ratio: pilot queue may have different MPI count
            pilot_qd = load_queue_data(self.server_machine_name, self.pilot_queue_label)
            prod_qd = load_queue_data(self.server_machine_name, self.queue_label)
            pilot_mpi = get_num_mpi(pilot_qd)
            prod_mpi = get_num_mpi(prod_qd)
            walker_ratio = pilot_mpi / prod_mpi

            warmup = self.num_mcmc_warmup_steps or 0
            estimated_mcmc_steps = estimate_required_steps(
                self.pilot_mcmc_steps - warmup,
                pilot_error,
                self.target_error,
                walker_ratio=walker_ratio,
            )
            # Add warmup back: production also discards warmup steps
            estimated_mcmc_steps += warmup

            total_pilot_cost = self.pilot_vmc_steps * self.pilot_mcmc_steps
            total_prod_cost = self.num_opt_steps * estimated_mcmc_steps
            cost_ratio = total_prod_cost / total_pilot_cost if total_pilot_cost > 0 else 0

            # Time estimate: only Net time scales with step count
            net_pilot_sec = parse_net_time(output_0)
            if net_pilot_sec is not None and net_pilot_sec > 0:
                overhead_sec = pilot_wall_sec - net_pilot_sec
                est_prod_sec = overhead_sec + net_pilot_sec * cost_ratio
            else:
                est_prod_sec = pilot_wall_sec * cost_ratio
            logger.info("")
            logger.info("-- VMC Step Estimation Summary " + "-" * 19)
            logger.info(
                f"  pilot MCMC/step     = {self.pilot_mcmc_steps}\n"
                f"  warmup steps        = {warmup}\n"
                f"  pilot VMC steps     = {self.pilot_vmc_steps}\n"
                f"  pilot error (last)  = {pilot_error:.6g} Ha\n"
                f"  target error        = {self.target_error:.6g} Ha\n"
                f"  pilot MPI procs     = {pilot_mpi}\n"
                f"  prod. MPI procs     = {prod_mpi}\n"
                f"  walker ratio        = {walker_ratio:.4g}\n"
                f"  estimated MCMC/step = {estimated_mcmc_steps}\n"
                f"  production VMC steps= {self.num_opt_steps}\n"
                f"  total pilot cost    = {total_pilot_cost} steps\n"
                f"  total prod. cost    = {total_prod_cost} steps\n"
                f"  pilot wall time     = {_format_duration(pilot_wall_sec)}\n"
                f"  pilot net time      = "
                f"{_format_duration(net_pilot_sec) if net_pilot_sec else 'N/A'}\n"
                f"  est. production time= {_format_duration(est_prod_sec)}"
            )
            logger.info("-" * 50)

            # Save estimation to main working directory
            os.chdir(_wd)
            set_estimation(
                _wd,
                pilot_mcmc_steps=self.pilot_mcmc_steps,
                pilot_vmc_steps=self.pilot_vmc_steps,
                pilot_error=pilot_error,
                target_error=self.target_error,
                estimated_mcmc_steps=estimated_mcmc_steps,
                pilot_queue_label=self.pilot_queue_label,
                walker_ratio=walker_ratio,
            )

        os.chdir(_wd)

        # ── Production runs (phase 1..N) ──────────────────────────
        last_run = 0
        for i in range(1, self.max_continuation + 1):
            input_i = suffixed_name(self.input_file, i)
            output_i = suffixed_name(self.output_file, i)

            # Check if optimization already completed before running
            opt_files = sorted(glob.glob("hamiltonian_data_opt_step_*.h5"))
            if len(opt_files) >= self.num_opt_steps:
                logger.info(
                    f"Optimization already completed ({len(opt_files)} opt step files found). Skipping production run {i}."
                )
                break

            # Skip input generation if this step already has a job record
            recorded = get_job(_wd, input_i)
            if recorded.get("status") in ("submitted", "completed", "fetched"):
                if recorded["status"] == "fetched":
                    logger.info(f"  {input_i}: already fetched. Skipping.")
                    last_run = i
                    continue
                # submitted or completed -- let _submit_and_wait handle resume
                logger.info(f"  {input_i}: already {recorded['status']}. Resuming...")
            else:
                if i == 1:
                    self._generate_input(
                        estimated_mcmc_steps,
                        self.num_opt_steps,
                        input_i,
                    )
                else:
                    restart_chk = self._find_restart_chk()
                    self._generate_input(
                        estimated_mcmc_steps,
                        self.num_opt_steps,
                        input_i,
                        restart=bool(restart_chk),
                        restart_chk=restart_chk,
                    )
                logger.info("")
                logger.info(f"-- VMC Phase 1: Production run {i}/{self.max_continuation} " + "-" * 10)
                logger.info(f"  {input_i}: {self.num_opt_steps} opt steps x {estimated_mcmc_steps} MCMC steps/step")

            await self._submit_and_wait(input_i, output_i)
            os.chdir(_wd)
            last_run = i

            # Check if optimization completed
            opt_files = sorted(glob.glob("hamiltonian_data_opt_step_*.h5"))
            if len(opt_files) >= self.num_opt_steps:
                logger.info(f"  VMC optimization completed (run {i}/{self.max_continuation})")
                break
            elif i < self.max_continuation:
                logger.info(
                    f"  VMC optimization incomplete "
                    f"({len(opt_files)}/{self.num_opt_steps} steps) -- "
                    f"continuing ({i}/{self.max_continuation})"
                )
            else:
                logger.warning(
                    f"  VMC optimization incomplete "
                    f"({len(opt_files)}/{self.num_opt_steps} steps) -- "
                    f"max_continuation ({self.max_continuation}) reached"
                )

        # ── Collect outputs ───────────────────────────────────────
        opt_files = sorted(glob.glob("hamiltonian_data_opt_step_*.h5"))
        chk_files = sorted(glob.glob("*.chk"))
        output_logs = [
            suffixed_name(self.output_file, j)
            for j in range(last_run + 1)
            if os.path.isfile(suffixed_name(self.output_file, j))
        ]
        self.output_files = opt_files + chk_files + output_logs

        if opt_files:
            self.output_values["optimized_hamiltonian"] = opt_files[-1]
        if chk_files:
            self.output_values["checkpoint"] = chk_files[-1]
        self.output_values["estimated_mcmc_steps"] = estimated_mcmc_steps

        # Parse last production output for energy
        last_output = suffixed_name(self.output_file, last_run)
        self._parse_output(last_output)

        self.status = "success"
        return self.status, self.output_files, self.output_values

    # ── Utility methods ───────────────────────────────────────────

    def _find_restart_chk(self) -> Optional[str]:
        """Locate a VMC restart checkpoint file."""
        for pattern in ["vmc.chk", "*.chk"]:
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[-1]
        return None

    # ── Output parsing ────────────────────────────────────────────

    def _parse_output(self, output_file=None):
        """Extract the last optimization energy from *output_file*."""
        if output_file is None:
            output_file = self.output_file
        if not os.path.isfile(output_file):
            return

        energy_pattern = re.compile(r"E\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s*\+\-\s*(\d+\.\d+(?:[eE][+-]?\d+)?)")
        last_match = None
        try:
            with open(output_file, "r") as f:
                for line in f:
                    m = energy_pattern.search(line)
                    if m:
                        last_match = m
        except Exception:
            return

        if last_match:
            self.output_values["energy"] = float(last_match.group(1))
            self.output_values["energy_error"] = float(last_match.group(2))
            logger.info(f"  VMC energy: {self.output_values['energy']} +- {self.output_values['energy_error']} Ha")

    @staticmethod
    def _parse_last_opt_energy(output_file):
        """Parse the last ``E = <val> +- <err>`` from a VMC output file.

        Extracts the energy from the *last* optimization step, which
        reflects the optimized wavefunction quality.

        Returns
        -------
        tuple
            ``(energy, error)`` or ``(None, None)``.
        """
        if not os.path.isfile(output_file):
            return None, None

        energy_pattern = re.compile(r"E\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\+\-\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)")
        last_match = None
        try:
            with open(output_file, "r") as f:
                for line in f:
                    m = energy_pattern.search(line)
                    if m:
                        last_match = m
        except Exception:
            return None, None

        if last_match:
            return float(last_match.group(1)), float(last_match.group(2))
        return None, None
