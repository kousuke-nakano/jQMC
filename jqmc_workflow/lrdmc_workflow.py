"""LRDMC_Workflow — Lattice-Regularized Diffusion Monte Carlo run.

Generates an LRDMC input TOML, submits ``jqmc`` (job_type=lrdmc-bra or
job_type=lrdmc-tau) on a remote/local machine, monitors until completion,
fetches the checkpoint, and post-processes with
``jqmc-tool lrdmc compute-energy``.

Two operating modes are available:

* **GFMC_n mode** (``job_type=lrdmc-bra``) — activated when
  *target_survived_walkers_ratio* or *num_mcmc_per_measurement* is set.
  Uses discrete projections per measurement.
* **GFMC_t mode** (``job_type=lrdmc-tau``) — activated when
  *time_projection_tau* is used (default).  Uses a continuous imaginary
  time step between projections.  No calibration pilot is needed.
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
import subprocess
import time
from logging import getLogger
from typing import Optional

from ._error_estimator import (
    _format_duration,
    estimate_additional_steps,
    estimate_required_steps,
    parse_net_time,
    suffixed_name,
)
from ._input_generator import generate_input_toml, resolve_with_defaults
from ._job import JobSubmission, get_num_mpi, load_queue_data
from ._lrdmc_calibration import (
    fit_num_mcmc_per_measurement,
    get_num_electrons,
    parse_survived_walkers_ratio,
)
from ._setting import (
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_WARMUP_STEPS,
)
from ._state import _now_iso, add_job, get_estimation, get_job, set_estimation, update_job
from .workflow import Workflow

logger = getLogger("jqmc-workflow").getChild(__name__)


class LRDMC_Workflow(Workflow):
    """Single LRDMC (Lattice-Regularized Diffusion Monte Carlo) run.

    Generates a ``job_type=lrdmc-bra`` (GFMC_n) or ``job_type=lrdmc-tau``
    (GFMC_t) input TOML at a fixed lattice spacing ``alat``, submits
    ``jqmc``, monitors until completion, fetches the checkpoint, and
    post-processes with ``jqmc-tool lrdmc compute-energy`` to extract
    the DMC energy ± error.

    **Mode selection** (mutually exclusive):

    * **GFMC_t** (default) — set *time_projection_tau* (default 0.10).
      Uses continuous imaginary-time projection.  Only the error-bar
      pilot is run (no calibration phase).
    * **GFMC_n** — set *target_survived_walkers_ratio* or
      *num_mcmc_per_measurement*.  Uses discrete GFMC projections.
      When *target_survived_walkers_ratio* is set (and
      *num_mcmc_per_measurement* is *None*), an automatic calibration
      pilot determines the optimal *num_mcmc_per_measurement*.

    The workflow operates in two phases:

    1. **Pilot run** (``_0``) — A short run with ``pilot_steps``
       measurement steps.  The resulting error estimates the steps
       required for ``target_error`` via $\\sigma \\propto 1/\\sqrt{N}$.
       In GFMC_n mode with calibration, three additional short runs
       precede this to determine *num_mcmc_per_measurement*.
    2. **Production runs** (``_1``, ``_2``, …) — Continuation runs
       with the estimated step count.  The loop terminates when the
       error is ≤ ``target_error`` or ``max_continuation`` is reached.

    Parameters
    ----------
    server_machine_name : str
        Target machine name.
    alat : float
        Lattice discretization parameter (bohr).
    hamiltonian_file : str
        Input ``hamiltonian_data.h5``.
    input_file : str
        Generated TOML input filename.
    output_file : str
        Stdout capture filename.
    queue_label : str
        Queue/partition label.
    jobname : str
        Scheduler job name.
    number_of_walkers : int
        Walkers per MPI process.
    max_time : int
        Wall-time limit (seconds).
    num_gfmc_bin_blocks : int
        Binning blocks for post-processing.
    num_gfmc_warmup_steps : int
        Warmup steps to discard in post-processing.
    num_gfmc_collect_steps : int
        Weight-collection steps for energy post-processing.
    time_projection_tau : float, optional
        Imaginary time step between projections (bohr) for GFMC_t
        mode.  Default ``0.10``.  Ignored when
        *target_survived_walkers_ratio* or *num_mcmc_per_measurement*
        is set.
    target_survived_walkers_ratio : float, optional
        Target survived-walkers ratio for automatic
        ``num_mcmc_per_measurement`` calibration.  Setting this
        activates GFMC_n mode.  The pilot phase runs three short
        calculations at ``Ne*k*(0.3/alat)²`` projections (k=2,4,6),
        fits a linear model to the observed survived-walkers ratio,
        and picks the value that achieves this target.
    num_mcmc_per_measurement : int, optional
        GFMC projections per measurement (GFMC_n mode).  When given
        explicitly, the automatic calibration is skipped.
    non_local_move : str, optional
        Non-local move treatment (``"tmove"`` or ``"dltmove"``).  Default from ``jqmc_miscs``.
    E_scf : float, optional
        Initial energy guess for the GFMC shift (GFMC_n only).
        Default from ``jqmc_miscs``.
    atomic_force : bool, optional
        Compute atomic forces.  Default from ``jqmc_miscs``.
    epsilon_PW : float, optional
        Pathak–Wagner regularization parameter (Bohr). When > 0,
        the force estimator is regularized near the nodal surface.
        Default from ``jqmc_miscs``.
    mcmc_seed : int, optional
        Random seed for MCMC.  Default from ``jqmc_miscs``.
    verbosity : str, optional
        Verbosity level.  Default from ``jqmc_miscs``.
    poll_interval : int
        Seconds between job-status polls.
    target_error : float
        Target statistical error (Ha).
    pilot_steps : int
        Measurement steps for the pilot estimation run.
    pilot_queue_label : str, optional
        Queue label for the pilot run.  Defaults to *queue_label*.
        Use a shorter/smaller queue for the pilot to save resources.
    max_continuation : int
        Maximum number of production runs after the pilot.

    Examples
    --------
    GFMC_t mode (default)::

        wf = LRDMC_Workflow(
            server_machine_name="cluster",
            alat=0.3,
            target_error=0.0005,
            number_of_walkers=8,
        )
        status, files, values = wf.launch()
        print(values["energy"], values["energy_error"])

    GFMC_n mode with calibration::

        wf = LRDMC_Workflow(
            server_machine_name="cluster",
            alat=0.3,
            target_error=0.0005,
            target_survived_walkers_ratio=0.97,
            number_of_walkers=8,
        )

    As part of a :class:`Launcher` pipeline::

        enc = Container(
            label="lrdmc-a0.30",
            dirname="03_lrdmc",
            input_files=[FileFrom("mcmc-run", "hamiltonian_data.h5")],
            workflow=LRDMC_Workflow(
                server_machine_name="cluster",
                alat=0.3,
                target_error=0.001,
            ),
        )

    Notes
    -----
    * For a²→0 continuum-limit extrapolation, use
      :class:`LRDMC_Ext_Workflow` instead.
    * The pilot is skipped on re-entrance if an estimation already
      exists in ``workflow_state.toml``.

    See Also
    --------
    LRDMC_Ext_Workflow : Multi-alat extrapolation wrapper.
    MCMC_Workflow : VMC production sampling (job_type=mcmc).
    VMC_Workflow : Wavefunction optimisation (job_type=vmc).
    """

    def __init__(
        self,
        server_machine_name: str = "localhost",
        alat: float = 0.30,
        hamiltonian_file: str = "hamiltonian_data.h5",
        input_file: str = "input.toml",
        output_file: str = "out.o",
        queue_label: str = "default",
        jobname: str = "jqmc-lrdmc",
        number_of_walkers: int = 4,
        max_time: int = 86400,
        num_gfmc_bin_blocks: int = 5,
        num_gfmc_warmup_steps: int = 0,
        num_gfmc_collect_steps: int = 5,
        # -- [lrdmc-bra / lrdmc-tau] section parameters --
        time_projection_tau: Optional[float] = 0.10,
        target_survived_walkers_ratio: Optional[float] = None,
        num_mcmc_per_measurement: Optional[int] = None,
        non_local_move: Optional[str] = None,
        E_scf: Optional[float] = None,
        atomic_force: Optional[bool] = None,
        epsilon_PW: Optional[float] = None,
        # -- [control] section parameters --
        mcmc_seed: Optional[int] = None,
        verbosity: Optional[str] = None,
        # -- workflow parameters --
        poll_interval: int = 60,
        target_error: float = 0.001,
        pilot_steps: int = 100,
        pilot_queue_label: Optional[str] = None,
        max_continuation: int = 1,
    ):
        super().__init__()
        self.server_machine_name = server_machine_name
        self.alat = alat
        self.hamiltonian_file = hamiltonian_file
        self.input_file = input_file
        self.output_file = output_file
        self.queue_label = queue_label
        self.jobname = jobname
        self.number_of_walkers = number_of_walkers
        self.max_time = max_time
        # Validate GFMC post-processing parameters
        if num_gfmc_warmup_steps < GFMC_MIN_WARMUP_STEPS:
            raise ValueError(
                f"num_gfmc_warmup_steps ({num_gfmc_warmup_steps}) must be "
                f">= {GFMC_MIN_WARMUP_STEPS}. E_scf is unreliable before "
                f"step {GFMC_MIN_WARMUP_STEPS}."
            )
        if num_gfmc_bin_blocks < GFMC_MIN_BIN_BLOCKS:
            raise ValueError(f"num_gfmc_bin_blocks ({num_gfmc_bin_blocks}) must be >= {GFMC_MIN_BIN_BLOCKS}.")
        if num_gfmc_collect_steps < GFMC_MIN_COLLECT_STEPS:
            raise ValueError(f"num_gfmc_collect_steps ({num_gfmc_collect_steps}) must be >= {GFMC_MIN_COLLECT_STEPS}.")
        self.num_gfmc_bin_blocks = num_gfmc_bin_blocks
        self.num_gfmc_warmup_steps = num_gfmc_warmup_steps
        self.num_gfmc_collect_steps = num_gfmc_collect_steps
        # Mode selection: GFMC_n vs GFMC_t
        #   GFMC_n: target_survived_walkers_ratio or num_mcmc_per_measurement is set
        #   GFMC_t: otherwise (uses time_projection_tau)
        self._use_gfmc_n = target_survived_walkers_ratio is not None or num_mcmc_per_measurement is not None
        # [lrdmc-bra / lrdmc-tau] section
        self.time_projection_tau = time_projection_tau
        self.target_survived_walkers_ratio = target_survived_walkers_ratio
        self.num_mcmc_per_measurement = num_mcmc_per_measurement
        self.non_local_move = non_local_move
        self.E_scf = E_scf
        self.atomic_force = atomic_force
        self.epsilon_PW = epsilon_PW
        # [control] section
        self.mcmc_seed = mcmc_seed
        self.verbosity = verbosity
        # workflow
        self.poll_interval = poll_interval
        self.target_error = target_error
        self.pilot_steps = pilot_steps
        self.pilot_queue_label = pilot_queue_label or queue_label
        self.max_continuation = max_continuation

    @property
    def job_type(self) -> str:
        """Return the jqmc job type string for TOML generation."""
        return "lrdmc-bra" if self._use_gfmc_n else "lrdmc-tau"

    # ── Input generation ──────────────────────────────────────────

    def _generate_input(
        self,
        num_steps,
        input_file,
        restart=False,
        restart_chk=None,
        num_mcmc_per_measurement=None,
    ):
        """Generate LRDMC TOML input file.

        Parameters
        ----------
        num_steps : int
            Number of measurement steps.
        input_file : str
            Output filename.
        restart : bool
            Whether this is a restart run.
        restart_chk : str or None
            Restart checkpoint filename.
        num_mcmc_per_measurement : int or None
            Override for GFMC projections per measurement (GFMC_n only).
            Falls back to ``self.num_mcmc_per_measurement``.
        """
        jt = self.job_type
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

        if self._use_gfmc_n:
            # GFMC_n mode: job_type="lrdmc-bra"
            nmpm = num_mcmc_per_measurement or self.num_mcmc_per_measurement
            section_ov = resolve_with_defaults(
                "lrdmc-bra",
                {
                    "num_mcmc_steps": num_steps,
                    "alat": self.alat,
                    "num_mcmc_per_measurement": nmpm,
                    "non_local_move": self.non_local_move,
                    "num_gfmc_warmup_steps": self.num_gfmc_warmup_steps,
                    "num_gfmc_bin_blocks": self.num_gfmc_bin_blocks,
                    "num_gfmc_collect_steps": self.num_gfmc_collect_steps,
                    "E_scf": self.E_scf,
                    "atomic_force": self.atomic_force,
                    "epsilon_PW": self.epsilon_PW,
                },
            )
        else:
            # GFMC_t mode: job_type="lrdmc-tau"
            section_ov = resolve_with_defaults(
                "lrdmc-tau",
                {
                    "num_mcmc_steps": num_steps,
                    "tau": self.time_projection_tau,
                    "alat": self.alat,
                    "non_local_move": self.non_local_move,
                    "num_gfmc_warmup_steps": self.num_gfmc_warmup_steps,
                    "num_gfmc_bin_blocks": self.num_gfmc_bin_blocks,
                    "num_gfmc_collect_steps": self.num_gfmc_collect_steps,
                    "atomic_force": self.atomic_force,
                    "epsilon_PW": self.epsilon_PW,
                },
            )

        overrides = {
            "control": control_ov,
            jt: section_ov,
        }
        generate_input_toml(
            job_type=jt,
            overrides=overrides,
            filename=input_file,
        )

    # ── Submit / poll / fetch ─────────────────────────────────────

    async def _submit_and_wait(
        self,
        input_file,
        output_file,
        extra_from_objects=None,
        fetch_from_objects=None,
        queue_label=None,
    ):
        """Create job, submit, poll until done, fetch results.

        All decisions are driven by ``workflow_state.toml``:

        * ``fetched``   -- skip entirely (already done)
        * ``completed`` -- fetch results only
        * ``submitted`` -- resume waiting, then fetch
        * no record     -- submit a new job
        """
        if fetch_from_objects is None:
            fetch_from_objects = ["*.h5", output_file]
        cwd = os.path.abspath(os.getcwd())

        # -- Restart detection via job history ---------------------
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

        # -- New submission ----------------------------------------
        job = self._make_job(input_file, output_file, queue_label=queue_label)
        job.generate_script()

        from_objects = [input_file, self.hamiltonian_file, "submit.sh"]
        if extra_from_objects:
            from_objects.extend(extra_from_objects)

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
        """Run the LRDMC workflow with automatic step estimation.

        **GFMC_n mode** (``target_survived_walkers_ratio`` or
        ``num_mcmc_per_measurement`` is set):

        When ``target_survived_walkers_ratio`` is set and
        ``num_mcmc_per_measurement`` is *None*, the pilot phase has
        two stages:

        1. **Calibration** (``_pilot_a/_pilot1`` – ``_pilot_a/_pilot3``,
           parallel) — Three short LRDMC runs at
           ``Ne*k*(0.3/alat)²`` projections (k=2,4,6).  A linear fit
           to the survived-walkers ratio determines the optimal
           ``num_mcmc_per_measurement``.
        2. **Error-bar pilot** (``_pilot_b``) — A run with the
           calibrated ``num_mcmc_per_measurement``; its error bar
           estimates the production step count.

        When ``num_mcmc_per_measurement`` is given explicitly, only
        ``_pilot_b`` (error-bar estimation) is executed.

        **GFMC_t mode** (default, ``time_projection_tau``):

        Only the error-bar pilot (``_pilot_b``) is executed — no
        calibration is needed since ``tau`` directly controls the
        imaginary time step.

        Production runs (``_1``, ``_2``, …) start from scratch and
        accumulate statistics until ``target_error`` is achieved or
        ``max_continuation`` is reached.
        """
        # Save absolute CWD -- other async tasks may chdir while we await.
        _wd = os.path.abspath(os.getcwd())

        # -- Phase 0: pilot estimation (skip on continuation) ------
        estimation = get_estimation(_wd)

        if estimation.get("estimated_steps") is not None:
            estimated_steps = int(estimation["estimated_steps"])
            # Restore calibrated nmpm from saved state (GFMC_n only)
            if self._use_gfmc_n and estimation.get("num_mcmc_per_measurement") is not None:
                self.num_mcmc_per_measurement = int(estimation["num_mcmc_per_measurement"])
            mode_str = f"nmpm={self.num_mcmc_per_measurement}" if self._use_gfmc_n else f"tau={self.time_projection_tau}"
            logger.info(
                f"Estimation already done (continuation): estimated_steps={estimated_steps}, {mode_str}. Skipping pilot."
            )
        else:
            # ── Phase A: calibrate num_mcmc_per_measurement (GFMC_n only) ──
            h5_src = os.path.join(_wd, self.hamiltonian_file)

            need_calibration = (
                self._use_gfmc_n and self.num_mcmc_per_measurement is None and self.target_survived_walkers_ratio is not None
            )

            if need_calibration:
                pilot_a_dir = os.path.join(_wd, "_pilot_a")
                os.makedirs(pilot_a_dir, exist_ok=True)

                n_electrons = get_num_electrons(os.path.join(_wd, self.hamiltonian_file))
                alat_scale = (0.3 / self.alat) ** 2
                trial_nmpm = [int(n_electrons * k * alat_scale) for k in (2, 4, 6)]

                logger.info("")
                logger.info(f"-- LRDMC Phase A: Calibrate num_mcmc_per_measurement (a={self.alat}) " + "-" * 8)
                logger.info(
                    f"  Ne={n_electrons}, "
                    f"trial nmpm = {trial_nmpm}, "
                    f"target survived ratio = "
                    f"{self.target_survived_walkers_ratio:.2%}"
                )

                # Run 3 calibration jobs in parallel sub-directories
                calib_tasks = []
                for idx, nmpm_val in enumerate(trial_nmpm, start=1):
                    calib_sub = os.path.join(pilot_a_dir, f"_pilot{idx}")
                    os.makedirs(calib_sub, exist_ok=True)
                    h5_link = os.path.join(calib_sub, self.hamiltonian_file)
                    if os.path.isfile(h5_src) and not os.path.exists(h5_link):
                        os.symlink(h5_src, h5_link)

                    inp = suffixed_name(self.input_file, 0)
                    out = suffixed_name(self.output_file, 0)

                    async def _run_calib(sub_dir, inp_f, out_f, nmpm_v, _idx=idx):
                        os.chdir(sub_dir)
                        rec = get_job(sub_dir, inp_f)
                        if rec.get("status") not in (
                            "submitted",
                            "completed",
                            "fetched",
                        ):
                            self._generate_input(
                                self.pilot_steps,
                                inp_f,
                                num_mcmc_per_measurement=nmpm_v,
                            )
                        else:
                            logger.info(f"  {inp_f} (nmpm={nmpm_v}): already {rec['status']}. Resuming...")
                        logger.info(f"  _pilot{_idx}: nmpm={nmpm_v}, {self.pilot_steps} steps")
                        await self._submit_and_wait(
                            inp_f,
                            out_f,
                            queue_label=self.pilot_queue_label,
                        )
                        os.chdir(sub_dir)
                        # Parse survived walkers ratio from output
                        ratio = parse_survived_walkers_ratio(out_f)
                        logger.info(f"  _pilot{_idx} (nmpm={nmpm_v}): survived ratio = {ratio:.4f}" if ratio else "N/A")
                        return nmpm_v, ratio

                    calib_tasks.append(asyncio.create_task(_run_calib(calib_sub, inp, out, nmpm_val)))

                calib_results = await asyncio.gather(*calib_tasks)
                os.chdir(_wd)

                # Collect results and fit
                x_vals = []
                y_vals = []
                for nmpm_v, ratio in calib_results:
                    if ratio is not None:
                        x_vals.append(nmpm_v)
                        y_vals.append(ratio)

                if len(x_vals) < 2:
                    raise RuntimeError(
                        f"Only {len(x_vals)} calibration runs returned a survived walkers ratio. Need at least 2 for linear fit."
                    )

                optimal_nmpm = fit_num_mcmc_per_measurement(
                    x_vals,
                    y_vals,
                    self.target_survived_walkers_ratio,
                )
                self.num_mcmc_per_measurement = optimal_nmpm

                logger.info("")
                logger.info(f"-- LRDMC Calibration Summary (a={self.alat}) " + "-" * 18)
                for xv, yv in zip(x_vals, y_vals):
                    logger.info(f"  nmpm={xv:>6d}: survived ratio = {yv:.4f}")
                logger.info(f"  target ratio  = {self.target_survived_walkers_ratio:.4f}")
                logger.info(f"  optimal nmpm  = {optimal_nmpm}")
                logger.info("-" * 50)

            # ── Phase B: error-bar pilot (_pilot_b) ───────────────
            pilot_b_dir = os.path.join(_wd, "_pilot_b")
            os.makedirs(pilot_b_dir, exist_ok=True)
            h5_link_b = os.path.join(pilot_b_dir, self.hamiltonian_file)
            if os.path.isfile(h5_src) and not os.path.exists(h5_link_b):
                os.symlink(h5_src, h5_link_b)

            os.chdir(pilot_b_dir)

            input_pb = suffixed_name(self.input_file, 0)
            output_pb = suffixed_name(self.output_file, 0)

            recorded_pb = get_job(pilot_b_dir, input_pb)
            if recorded_pb.get("status") not in (
                "submitted",
                "completed",
                "fetched",
            ):
                self._generate_input(self.pilot_steps, input_pb)
            else:
                logger.info(f"  {input_pb}: already {recorded_pb['status']}. Resuming...")
            mode_info = f"nmpm={self.num_mcmc_per_measurement}" if self._use_gfmc_n else f"tau={self.time_projection_tau}"
            logger.info("")
            logger.info(f"-- LRDMC Phase B: Error-bar Pilot (a={self.alat}) " + "-" * 12)
            logger.info(f"  {input_pb}: {self.pilot_steps} steps, {mode_info} (queue: {self.pilot_queue_label})")

            pilot_t0 = time.monotonic()
            await self._submit_and_wait(input_pb, output_pb, queue_label=self.pilot_queue_label)
            os.chdir(pilot_b_dir)
            pilot_wall_sec = time.monotonic() - pilot_t0

            restart_chk = self._find_restart_chk()
            if not restart_chk:
                raise RuntimeError("No checkpoint found after pilot run. Cannot estimate required steps.")

            _, pilot_error = self._compute_energy(restart_chk)
            if pilot_error is None:
                raise RuntimeError("Could not parse energy error from pilot run.")

            # Walker ratio: pilot queue may have different MPI count
            pilot_qd = load_queue_data(self.server_machine_name, self.pilot_queue_label)
            prod_qd = load_queue_data(self.server_machine_name, self.queue_label)
            pilot_mpi = get_num_mpi(pilot_qd)
            prod_mpi = get_num_mpi(prod_qd)
            walker_ratio = pilot_mpi / prod_mpi

            estimated_steps = estimate_required_steps(
                self.pilot_steps - self.num_gfmc_warmup_steps,
                pilot_error,
                self.target_error,
                walker_ratio=walker_ratio,
            )
            # Add warmup back: production also discards warmup steps
            estimated_steps += self.num_gfmc_warmup_steps

            # Time estimate: only Net time scales with step count
            step_ratio = estimated_steps / self.pilot_steps if self.pilot_steps > 0 else 0
            net_pilot_sec = parse_net_time(output_pb)
            if net_pilot_sec is not None and net_pilot_sec > 0:
                overhead_sec = pilot_wall_sec - net_pilot_sec
                est_prod_sec = overhead_sec + net_pilot_sec * step_ratio
            else:
                est_prod_sec = pilot_wall_sec * step_ratio
            logger.info("")
            logger.info(f"-- LRDMC Step Estimation Summary (a={self.alat}) " + "-" * 8)
            logger.info(
                f"  pilot steps       = {self.pilot_steps}\n"
                f"  warmup steps      = {self.num_gfmc_warmup_steps}\n"
                f"  pilot error       = {pilot_error:.6g} Ha\n"
                f"  target error      = {self.target_error:.6g} Ha\n"
                f"  mode              = {'GFMC_n' if self._use_gfmc_n else 'GFMC_t'}\n"
                f"  {'nmpm' if self._use_gfmc_n else 'tau':17s} = {self.num_mcmc_per_measurement if self._use_gfmc_n else self.time_projection_tau}\n"
                f"  pilot MPI procs   = {pilot_mpi}\n"
                f"  prod. MPI procs   = {prod_mpi}\n"
                f"  walker ratio      = {walker_ratio:.4g}\n"
                f"  estimated steps   = {estimated_steps}\n"
                f"  pilot wall time   = "
                f"{_format_duration(pilot_wall_sec)}\n"
                f"  pilot net time    = "
                f"{_format_duration(net_pilot_sec) if net_pilot_sec else 'N/A'}\n"
                f"  est. prod. time   = "
                f"{_format_duration(est_prod_sec)}"
            )
            logger.info("-" * 50)

            # Save estimation to main working directory
            os.chdir(_wd)
            est_kwargs = dict(
                pilot_steps=self.pilot_steps,
                pilot_error=pilot_error,
                target_error=self.target_error,
                estimated_steps=estimated_steps,
                pilot_queue_label=self.pilot_queue_label,
                walker_ratio=walker_ratio,
            )
            if self._use_gfmc_n:
                est_kwargs["num_mcmc_per_measurement"] = self.num_mcmc_per_measurement
            else:
                est_kwargs["time_projection_tau"] = self.time_projection_tau
            set_estimation(_wd, **est_kwargs)

        os.chdir(_wd)

        # ── Re-compute energy if post-processing parameters changed ──
        _postproc_changed = (
            estimation.get("last_num_gfmc_bin_blocks") != self.num_gfmc_bin_blocks
            or estimation.get("last_num_gfmc_warmup_steps") != self.num_gfmc_warmup_steps
            or estimation.get("last_num_gfmc_collect_steps") != self.num_gfmc_collect_steps
        )
        if _postproc_changed and estimation.get("last_energy") is not None:
            logger.info(
                "  Post-processing parameters changed "
                f"(bin_blocks: {estimation.get('last_num_gfmc_bin_blocks')}"
                f" -> {self.num_gfmc_bin_blocks}, "
                f"warmup: {estimation.get('last_num_gfmc_warmup_steps')}"
                f" -> {self.num_gfmc_warmup_steps}, "
                f"collect: {estimation.get('last_num_gfmc_collect_steps')}"
                f" -> {self.num_gfmc_collect_steps}); "
                "re-computing energy."
            )
            restart_chk = self._find_restart_chk()
            if restart_chk:
                energy, error = self._compute_energy(restart_chk)
                if energy is not None:
                    set_estimation(
                        _wd,
                        last_energy=energy,
                        last_energy_error=error,
                        last_num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
                        last_num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
                        last_num_gfmc_collect_steps=self.num_gfmc_collect_steps,
                    )
                    estimation = get_estimation(_wd)

        # ── Early exit if target already met ──────────────────────
        # Use cached energy/error from previous runs (saved in
        # workflow_state.toml) to avoid re-running compute-energy.
        cached_energy = estimation.get("last_energy")
        cached_error = estimation.get("last_energy_error")
        if cached_energy is not None and cached_error is not None:
            if cached_error <= self.target_error * 1.20:
                restart_chk = self._find_restart_chk()
                logger.info(
                    f"  Target already achieved (cached): {cached_error:.6g} <= {self.target_error * 1.20:.6g} Ha (target*1.20)"
                )
                self.output_values.update(
                    energy=cached_energy,
                    energy_error=cached_error,
                    alat=self.alat,
                    restart_chk=restart_chk or "",
                    estimated_steps=estimated_steps,
                    num_mcmc_per_measurement=self.num_mcmc_per_measurement,
                )
                self.output_files = sorted(glob.glob("*.h5"))
                self.status = "success"
                return self.status, self.output_files, self.output_values

        # ── Production runs (phase 1..N) ──────────────────────────
        # Production starts from scratch (no restart from pilot).
        # Checkpoint preserves all accumulated statistics across
        # production restarts.
        accumulated_steps = 0
        last_run = 0
        for i in range(1, self.max_continuation + 1):
            input_i = suffixed_name(self.input_file, i)
            output_i = suffixed_name(self.output_file, i)

            # Skip input generation if this step already has a job record
            recorded = get_job(_wd, input_i)
            if recorded.get("status") in ("submitted", "completed", "fetched"):
                if recorded["status"] == "fetched":
                    logger.info(f"  {input_i}: already fetched. Skipping.")
                    accumulated_steps += estimated_steps
                    last_run = i
                    continue
                # submitted or completed -- let _submit_and_wait handle resume
                logger.info(f"  {input_i}: already {recorded['status']}. Resuming...")
            else:
                if i == 1:
                    # First production run: start from scratch
                    self._generate_input(estimated_steps, input_i)
                else:
                    restart_chk = self._find_restart_chk()
                    if restart_chk is None:
                        raise RuntimeError(
                            f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {os.getcwd()}"
                        )
                    self._generate_input(
                        estimated_steps,
                        input_i,
                        restart=bool(restart_chk),
                        restart_chk=restart_chk,
                    )
                logger.info("")
                logger.info(f"-- LRDMC Phase 1: Production run {i}/{self.max_continuation} (a={self.alat}) " + "-" * 10)
                logger.info(f"  {input_i}: {estimated_steps} steps")

            restart_chk = self._find_restart_chk() if i > 1 else None
            if i > 1 and restart_chk is None:
                raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {os.getcwd()}")
            extra_from = [restart_chk] if restart_chk else []

            await self._submit_and_wait(
                input_i,
                output_i,
                extra_from_objects=extra_from,
            )
            os.chdir(_wd)
            accumulated_steps += estimated_steps
            last_run = i

            # Check convergence
            restart_chk = self._find_restart_chk()
            if restart_chk:
                energy, error = self._compute_energy(restart_chk)
                if energy is not None:
                    self.output_values["energy"] = energy
                    self.output_values["energy_error"] = error
                    self.output_values["alat"] = self.alat
                    self.output_values["restart_chk"] = restart_chk
                    logger.info(f"  LRDMC energy (a={self.alat}): {energy} +- {error} Ha")

                    # Cache for restart
                    set_estimation(
                        _wd,
                        last_energy=energy,
                        last_energy_error=error,
                        last_num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
                        last_num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
                        last_num_gfmc_collect_steps=self.num_gfmc_collect_steps,
                    )

                    if error <= self.target_error * 1.20:
                        logger.info(
                            f"  Target error achieved: {error:.6g} <= "
                            f"{self.target_error * 1.20:.6g} Ha (target*1.20) "
                            f"(run {i}/{self.max_continuation})"
                        )
                        break
                    elif i < self.max_continuation:
                        # Statistics accumulate across restarts (pickle
                        # preserves the full GFMC state), so we estimate
                        # the *additional* steps needed.
                        old_steps = estimated_steps
                        estimated_steps = estimate_additional_steps(
                            accumulated_steps,
                            error,
                            self.target_error,
                        )
                        logger.info(
                            f"  Re-estimated: {old_steps} -> {estimated_steps} "
                            f"additional steps (accumulated so far: {accumulated_steps})"
                        )
                    else:
                        logger.warning(
                            f"Error {error:.6g} > target "
                            f"{self.target_error:.6g} Ha -- "
                            f"max_continuation ({self.max_continuation}) reached"
                        )

        # ── Final energy computation if skipped ───────────────────
        # When all production runs are already fetched, the loop body
        # never calls _compute_energy.  Compute it now so the caller
        # always receives an up-to-date energy with the current
        # post-processing parameters.
        if "energy" not in self.output_values:
            restart_chk = self._find_restart_chk()
            if restart_chk:
                energy, error = self._compute_energy(restart_chk)
                if energy is not None:
                    self.output_values["energy"] = energy
                    self.output_values["energy_error"] = error
                    self.output_values["alat"] = self.alat
                    self.output_values["restart_chk"] = restart_chk
                    set_estimation(
                        _wd,
                        last_energy=energy,
                        last_energy_error=error,
                        last_num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
                        last_num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
                        last_num_gfmc_collect_steps=self.num_gfmc_collect_steps,
                    )

        # ── Collect outputs ───────────────────────────────────────
        chk_files = sorted(glob.glob("*.h5"))
        output_logs = [
            suffixed_name(self.output_file, j)
            for j in range(last_run + 1)
            if os.path.isfile(suffixed_name(self.output_file, j))
        ]
        self.output_files = chk_files + output_logs
        self.output_values["estimated_steps"] = estimated_steps
        if self._use_gfmc_n:
            self.output_values["num_mcmc_per_measurement"] = self.num_mcmc_per_measurement
        else:
            self.output_values["time_projection_tau"] = self.time_projection_tau

        self.status = "success"
        return self.status, self.output_files, self.output_values

    # ── Utility methods ───────────────────────────────────────────

    def _find_restart_chk(self) -> Optional[str]:
        """Locate the LRDMC restart checkpoint file."""
        for pattern in ["restart.h5", "lrdmc.h5", "*.h5"]:
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[-1]
        return None

    def _compute_energy(self, restart_chk: str):
        """Run ``jqmc-tool lrdmc compute-energy`` and parse output.

        Returns
        -------
        tuple
            ``(energy, error)`` or ``(None, None)``.
        """
        cmd = (
            f"jqmc-tool lrdmc compute-energy {restart_chk} "
            f"-b {self.num_gfmc_bin_blocks} "
            f"-w {self.num_gfmc_warmup_steps} "
            f"-c {self.num_gfmc_collect_steps}"
        )
        logger.info(f"  Running: {cmd}")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            return self._parse_energy_output(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"compute-energy failed: {e.stderr}")
            return None, None

    @staticmethod
    def _parse_energy_output(text: str):
        """Parse ``E = <value> +- <error> Ha.`` from jqmc-tool output."""
        pattern = re.compile(r"E\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\+\-\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)")
        match = pattern.search(text)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None, None
