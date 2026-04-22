"""LRDMC_Workflow — Lattice-Regularized Diffusion Monte Carlo run.

Generates an LRDMC input TOML, submits ``jqmc`` (job_type=lrdmc-bra or
job_type=lrdmc-tau) on a remote/local machine, monitors until completion,
fetches the checkpoint, and post-processes with
``jqmc-tool lrdmc compute-energy``.

Two operating modes are available:

* **GFMC_n mode** (``job_type=lrdmc-bra``) — activated when
  *target_survived_walkers_ratio* or *num_projection_per_measurement* is set.
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
)
from ._input_generator import generate_input_toml, resolve_with_defaults
from ._job import get_num_mpi, load_queue_data
from ._lrdmc_calibration import (
    fit_num_projection_per_measurement,
    get_num_electrons,
    parse_survived_walkers_ratio,
)
from ._output_parser import parse_force_table
from ._setting import (
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_WARMUP_STEPS,
)
from ._state import (
    CompletionStatus,
    WorkflowStatus,
    get_estimation,
    get_job_by_step,
    set_estimation,
    validate_completion,
)
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
      *num_projection_per_measurement*.  Uses discrete GFMC projections.
      When *target_survived_walkers_ratio* is set (and
      *num_projection_per_measurement* is *None*), an automatic calibration
      pilot determines the optimal *num_projection_per_measurement*.

    The workflow supports two operating modes:

    **Automatic mode** (default, ``num_gfmc_projections=None``):

    1. **Pilot run** (``_0``) — A short run with ``pilot_steps``
       measurement steps.  The resulting error estimates the steps
       required for ``target_error`` via $\sigma \propto 1/\sqrt{N}$.
       In GFMC_n mode with calibration, three additional short runs
       precede this to determine *num_projection_per_measurement*.
    2. **Production runs** (``_1``, ``_2``, …) — Continuation runs
       with the estimated step count.  The loop terminates when the
       error is ≤ ``target_error`` or ``max_continuation`` is reached.

    **Fixed-step mode** (``num_gfmc_projections`` is set):

    The error-bar pilot (``_pilot_b``) is skipped and ``target_error``
    is ignored.  If calibration is needed (GFMC_n mode with
    ``target_survived_walkers_ratio``), ``_pilot_a`` still runs.
    Each production run uses exactly ``num_gfmc_projections``
    measurement steps, and ``max_continuation`` runs are executed
    unconditionally.

    Parameters
    ----------
    server_machine_name : str
        Target machine name.
    alat : float
        Lattice discretization parameter (bohr).
    hamiltonian_file : str
        Input ``hamiltonian_data.h5``.
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
        *target_survived_walkers_ratio* or *num_projection_per_measurement*
        is set.
    target_survived_walkers_ratio : float, optional
        Target survived-walkers ratio for automatic
        ``num_projection_per_measurement`` calibration.  Setting this
        activates GFMC_n mode.  The pilot phase runs three short
        calculations at ``Ne*k*(0.3/alat)²`` projections (k=2,4,6),
        fits a linear model to the observed survived-walkers ratio,
        and picks the value that achieves this target.
    num_projection_per_measurement : int, optional
        GFMC projections per measurement (GFMC_n mode).  When given
        explicitly, the automatic calibration is skipped.
    non_local_move : str, optional
        Non-local move treatment (``"tmove"`` or ``"dltmove"``).  Default from ``jqmc_miscs``.
    E_scf : float, optional
        Initial energy guess for the GFMC shift (GFMC_n only).
        Default from ``jqmc_miscs``.
    atomic_force : bool, optional
        Compute atomic forces.  Default from ``jqmc_miscs``.
    use_swct : bool, optional
        Apply Space Warp Coordinate Transformation (SWCT) to atomic forces.
        Default is False for LRDMC.
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
    num_gfmc_projections : int, optional
        Fixed number of measurement steps per production run.
        When set, the error-bar pilot (``_pilot_b``) is skipped,
        ``target_error`` is ignored, and all ``max_continuation``
        production runs are executed unconditionally.  Calibration
        (``_pilot_a``) still runs when needed (GFMC_n mode with
        ``target_survived_walkers_ratio``).  Default *None*
        (automatic mode).
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

    Fixed-step mode (skip error-bar pilot)::

        wf = LRDMC_Workflow(
            server_machine_name="cluster",
            alat=0.3,
            num_gfmc_projections=500,
            max_continuation=3,
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

    Output Values
    -------------
    After ``launch()`` completes, ``output_values`` may contain:

    energy : float
        DMC energy (Ha).
    energy_error : float
        Statistical error on ``energy`` (Ha).
    alat : float
        Lattice spacing used for this run.
    restart_chk : str
        Basename of the restart checkpoint file.
    forces : object
        Atomic forces (only when ``atomic_force=True``).
    estimated_steps : int
        Estimated total measurement steps.
    num_projection_per_measurement : int
        Number of GFMC projections per measurement
        (GFMC_n mode only).
    time_projection_tau : float
        Imaginary-time projection step (GFMC_t mode only).

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
        num_projection_per_measurement: Optional[int] = None,
        non_local_move: Optional[str] = None,
        E_scf: Optional[float] = None,
        atomic_force: Optional[bool] = None,
        use_swct: Optional[bool] = None,
        epsilon_PW: Optional[float] = None,
        # -- [control] section parameters --
        mcmc_seed: Optional[int] = None,
        verbosity: Optional[str] = None,
        # -- workflow parameters --
        poll_interval: int = 60,
        target_error: float = 0.001,
        pilot_steps: int = 100,
        num_gfmc_projections: Optional[int] = None,
        pilot_queue_label: Optional[str] = None,
        max_continuation: int = 1,
    ):
        super().__init__()
        self.server_machine_name = server_machine_name
        self.alat = alat
        self.hamiltonian_file = hamiltonian_file
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
        #   GFMC_n: target_survived_walkers_ratio or num_projection_per_measurement is set
        #   GFMC_t: otherwise (uses time_projection_tau)
        self._use_gfmc_n = target_survived_walkers_ratio is not None or num_projection_per_measurement is not None
        # [lrdmc-bra / lrdmc-tau] section
        self.time_projection_tau = time_projection_tau
        self.target_survived_walkers_ratio = target_survived_walkers_ratio
        self.num_projection_per_measurement = num_projection_per_measurement
        self.non_local_move = non_local_move
        self.E_scf = E_scf
        self.atomic_force = atomic_force
        self.use_swct = use_swct
        self.epsilon_PW = epsilon_PW
        # [control] section
        self.mcmc_seed = mcmc_seed
        self.verbosity = verbosity
        # workflow
        self.poll_interval = poll_interval
        self.target_error = target_error
        self.pilot_steps = pilot_steps
        self.num_gfmc_projections = num_gfmc_projections
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
        num_projection_per_measurement=None,
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
        num_projection_per_measurement : int or None
            Override for GFMC projections per measurement (GFMC_n only).
            Falls back to ``self.num_projection_per_measurement``.
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
            nmpm = num_projection_per_measurement or self.num_projection_per_measurement
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
                    "use_swct": self.use_swct,
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
                    "use_swct": self.use_swct,
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
    # _submit_and_wait() and _make_job() are inherited from Workflow.

    # ── configure / run ──────────────────────────────────────────

    def configure(self) -> dict:
        """Validate parameters and return configuration summary."""
        return {
            "jobname": self.jobname,
            "alat": self.alat,
            "target_error": self.target_error,
            "hamiltonian_file": self.hamiltonian_file,
            "server_machine": self.server_machine_name,
            "number_of_walkers": self.number_of_walkers,
            "max_time": self.max_time,
            "pilot_steps": self.pilot_steps,
            "max_continuation": self.max_continuation,
        }

    async def run(self) -> tuple:
        """Run the LRDMC workflow.

        **Fixed-step mode** (``num_gfmc_projections`` is set):
        The error-bar pilot (``_pilot_b``) is skipped.  Calibration
        (``_pilot_a``) still runs if needed.  Each production run
        uses exactly ``num_gfmc_projections`` steps and all
        ``max_continuation`` runs are executed unconditionally.

        **Automatic mode** (``num_gfmc_projections`` is *None*, default):

        1. Calibration pilot (``_pilot_a``, GFMC_n only) — Three short
           LRDMC runs to determine ``num_projection_per_measurement``.
        2. Error-bar pilot (``_pilot_b``) — estimates production steps.
        3. Production runs (``_1``, ``_2``, …) — accumulate statistics
           until ``target_error`` is achieved or ``max_continuation``
           is reached.
        """
        self._ensure_project_dir()
        _wd = self.project_dir

        # ── Fixed-step mode ───────────────────────────────────────
        if self.num_gfmc_projections is not None:
            return await self._launch_fixed_steps(_wd)

        # ── Automatic mode (pilot + target_error) ─────────────────
        return await self._launch_auto(_wd)

    async def _launch_fixed_steps(self, _wd):
        """Fixed-step production: skip error-bar pilot, ignore target_error.

        Calibration (``_pilot_a``) still runs when needed (GFMC_n mode
        with ``target_survived_walkers_ratio``).
        """
        estimated_steps = self.num_gfmc_projections

        # ── Phase A: calibrate num_projection_per_measurement (GFMC_n only) ──
        need_calibration = (
            self._use_gfmc_n and self.num_projection_per_measurement is None and self.target_survived_walkers_ratio is not None
        )
        if need_calibration:
            await self._run_calibration(_wd)

        mode_info = f"nmpm={self.num_projection_per_measurement}" if self._use_gfmc_n else f"tau={self.time_projection_tau}"
        logger.info("")
        logger.info("-- LRDMC Fixed-step mode " + "-" * 26)
        logger.info(
            f"  num_gfmc_projections = {estimated_steps}\n"
            f"  max_continuation     = {self.max_continuation}\n"
            f"  mode                 = {'GFMC_n' if self._use_gfmc_n else 'GFMC_t'}\n"
            f"  {mode_info}"
        )

        step_files = {}  # {step: (input, output, run_id)}
        last_run = 0
        for i in range(1, self.max_continuation + 1):
            recorded = get_job_by_step(_wd, i)
            status_i = recorded.get("status")

            if status_i == "fetched":
                step_files[i] = (recorded["input_file"], recorded["output_file"], recorded.get("run_id", ""))
                logger.info(f"  step {i}: already fetched. Skipping.")
                last_run = i
                continue
            elif status_i in ("submitted", "completed"):
                input_i = recorded["input_file"]
                output_i = recorded["output_file"]
                run_id_i = recorded.get("run_id", "")
                logger.info(f"  step {i}: already {status_i}. Resuming...")
            else:
                run_id_i = self._new_run_id()
                input_i = self._input_filename(i, run_id_i)
                output_i = self._output_filename(i, run_id_i)
                if i == 1:
                    self._generate_input(estimated_steps, os.path.join(_wd, input_i))
                else:
                    restart_chk = self._find_restart_chk(_wd)
                    if restart_chk is None:
                        raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {_wd}")
                    self._generate_input(
                        estimated_steps,
                        os.path.join(_wd, input_i),
                        restart=True,
                        restart_chk=restart_chk,
                    )
                logger.info("")
                logger.info(f"-- LRDMC Production run {i}/{self.max_continuation} (a={self.alat}) " + "-" * 10)
                logger.info(f"  {input_i}: {estimated_steps} steps")

            step_files[i] = (input_i, output_i, run_id_i)
            restart_chk = self._find_restart_chk(_wd) if i > 1 else None
            if i > 1 and restart_chk is None:
                raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {_wd}")
            extra_from = [restart_chk] if restart_chk else []

            await self._submit_and_wait(
                input_i,
                output_i,
                work_dir=_wd,
                extra_from_objects=extra_from,
                step=i,
                run_id=run_id_i,
            )
            last_run = i

            # Post-process energy (informational only, no convergence check)
            restart_chk = self._find_restart_chk(_wd)
            if restart_chk:
                energy, error = self._compute_energy(restart_chk, work_dir=_wd, output_file=output_i)
                if energy is not None:
                    self.output_values["energy"] = energy
                    self.output_values["energy_error"] = error
                    self.output_values["alat"] = self.alat
                    self.output_values["restart_chk"] = restart_chk
                    logger.info(f"  LRDMC energy (a={self.alat}): {energy} +- {error} Ha")
                    if self.atomic_force:
                        forces = self._compute_force(restart_chk, work_dir=_wd, output_file=output_i)
                        if forces is not None:
                            self.output_values["forces"] = forces
                    set_estimation(
                        _wd,
                        last_energy=energy,
                        last_energy_error=error,
                        last_num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
                        last_num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
                        last_num_gfmc_collect_steps=self.num_gfmc_collect_steps,
                    )

            # ── Abnormal-termination guard (single source of truth) ──
            # Fixed-step mode has no convergence criterion, so only the
            # Program-ends / non-finite-energy checks are active here.
            vstatus, vmsg = validate_completion(_wd, self.output_values)
            if vstatus == CompletionStatus.FAILED:
                logger.error(vmsg)
                self.output_values["error"] = vmsg
                self.status = WorkflowStatus.FAILED
                break

        # ── Final energy computation ─────────────────────────────
        last_output = step_files[last_run][1] if last_run in step_files else None
        restart_chk = self._find_restart_chk(_wd)
        if restart_chk:
            energy, error = self._compute_energy(restart_chk, work_dir=_wd, output_file=last_output)
            if energy is not None:
                self.output_values["energy"] = energy
                self.output_values["energy_error"] = error
                self.output_values["alat"] = self.alat
                self.output_values["restart_chk"] = restart_chk
                if self.atomic_force:
                    forces = self._compute_force(restart_chk, work_dir=_wd, output_file=last_output)
                    if forces is not None:
                        self.output_values["forces"] = forces
                set_estimation(
                    _wd,
                    last_energy=energy,
                    last_energy_error=error,
                    last_num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
                    last_num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
                    last_num_gfmc_collect_steps=self.num_gfmc_collect_steps,
                )

        # ── Collect outputs ───────────────────────────────────────
        chk_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.h5")))
        output_logs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.out")))
        self.output_files = chk_files + output_logs
        self.output_values["estimated_steps"] = estimated_steps
        if self._use_gfmc_n:
            self.output_values["num_projection_per_measurement"] = self.num_projection_per_measurement
        else:
            self.output_values["time_projection_tau"] = self.time_projection_tau

        if self.status != WorkflowStatus.FAILED:
            self.status = WorkflowStatus.COMPLETED
        return self.status, self.output_files, self.output_values

    async def _run_calibration(self, _wd):
        """Phase A: calibrate num_projection_per_measurement (GFMC_n only)."""
        h5_src = os.path.join(_wd, self.hamiltonian_file)
        pilot_a_dir = os.path.join(_wd, "_pilot_a")
        os.makedirs(pilot_a_dir, exist_ok=True)

        n_electrons = get_num_electrons(os.path.join(_wd, self.hamiltonian_file))
        alat_scale = (0.3 / self.alat) ** 2
        trial_nmpm = [int(n_electrons * k * alat_scale) for k in (2, 4, 6)]

        logger.info("")
        logger.info(f"-- LRDMC Phase A: Calibrate num_projection_per_measurement (a={self.alat}) " + "-" * 8)
        logger.info(
            f"  Ne={n_electrons}, trial nmpm = {trial_nmpm}, target survived ratio = {self.target_survived_walkers_ratio:.2%}"
        )

        # Run 3 calibration jobs in parallel sub-directories
        calib_tasks = []
        for idx, nmpm_val in enumerate(trial_nmpm, start=1):
            calib_sub = os.path.join(pilot_a_dir, f"_pilot{idx}")
            os.makedirs(calib_sub, exist_ok=True)
            h5_link = os.path.join(calib_sub, self.hamiltonian_file)
            if os.path.isfile(h5_src) and not os.path.exists(h5_link):
                os.symlink(h5_src, h5_link)

            async def _run_calib(sub_dir, nmpm_v, _idx=idx):
                rec = get_job_by_step(sub_dir, 0)
                status_c = rec.get("status")
                if status_c in ("fetched", "submitted", "completed"):
                    inp_f = rec["input_file"]
                    out_f = rec["output_file"]
                    run_id_c = rec.get("run_id", "")
                    if status_c == "fetched":
                        logger.info(f"  {inp_f} (nmpm={nmpm_v}): already fetched. Skipping.")
                    else:
                        logger.info(f"  {inp_f} (nmpm={nmpm_v}): already {status_c}. Resuming...")
                else:
                    run_id_c = self._new_run_id()
                    inp_f = self._input_filename(0, run_id_c)
                    out_f = self._output_filename(0, run_id_c)
                    self._generate_input(
                        self.pilot_steps,
                        os.path.join(sub_dir, inp_f),
                        num_projection_per_measurement=nmpm_v,
                    )
                logger.info(f"  _pilot{_idx}: nmpm={nmpm_v}, {self.pilot_steps} steps")
                await self._submit_and_wait(
                    inp_f,
                    out_f,
                    work_dir=sub_dir,
                    queue_label=self.pilot_queue_label,
                    step=0,
                    run_id=run_id_c,
                )
                # Parse survived walkers ratio from output
                out_path = os.path.join(sub_dir, out_f)
                ratio = parse_survived_walkers_ratio(out_path)
                logger.info(f"  _pilot{_idx} (nmpm={nmpm_v}): survived ratio = {ratio:.4f}" if ratio else "N/A")
                return nmpm_v, ratio

            calib_tasks.append(asyncio.create_task(_run_calib(calib_sub, nmpm_val)))

        calib_results = await asyncio.gather(*calib_tasks)

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

        optimal_nmpm = fit_num_projection_per_measurement(
            x_vals,
            y_vals,
            self.target_survived_walkers_ratio,
        )
        self.num_projection_per_measurement = optimal_nmpm

        logger.info("")
        logger.info(f"-- LRDMC Calibration Summary (a={self.alat}) " + "-" * 18)
        for xv, yv in zip(x_vals, y_vals):
            logger.info(f"  nmpm={xv:>6d}: survived ratio = {yv:.4f}")
        logger.info(f"  target ratio  = {self.target_survived_walkers_ratio:.4f}")
        logger.info(f"  optimal nmpm  = {optimal_nmpm}")
        logger.info("-" * 50)

    async def _launch_auto(self, _wd):
        """Automatic mode: pilot + target_error convergence."""
        estimation = get_estimation(_wd)

        if estimation.get("estimated_steps") is not None:
            estimated_steps = int(estimation["estimated_steps"])
            # Restore calibrated nmpm from saved state (GFMC_n only)
            if self._use_gfmc_n and estimation.get("num_projection_per_measurement") is not None:
                self.num_projection_per_measurement = int(estimation["num_projection_per_measurement"])
            mode_str = f"nmpm={self.num_projection_per_measurement}" if self._use_gfmc_n else f"tau={self.time_projection_tau}"
            logger.info(
                f"Estimation already done (continuation): estimated_steps={estimated_steps}, {mode_str}. Skipping pilot."
            )
        else:
            # ── Phase A: calibrate num_projection_per_measurement (GFMC_n only) ──
            h5_src = os.path.join(_wd, self.hamiltonian_file)

            need_calibration = (
                self._use_gfmc_n
                and self.num_projection_per_measurement is None
                and self.target_survived_walkers_ratio is not None
            )

            if need_calibration:
                await self._run_calibration(_wd)

            # ── Phase B: error-bar pilot (_pilot_b) ───────────────
            pilot_b_dir = os.path.join(_wd, "_pilot_b")
            os.makedirs(pilot_b_dir, exist_ok=True)
            h5_link_b = os.path.join(pilot_b_dir, self.hamiltonian_file)
            if os.path.isfile(h5_src) and not os.path.exists(h5_link_b):
                os.symlink(h5_src, h5_link_b)

            input_pb = None
            output_pb = None
            run_id_pb = ""

            recorded_pb = get_job_by_step(pilot_b_dir, 0)
            status_pb = recorded_pb.get("status")
            if status_pb in ("fetched", "submitted", "completed"):
                input_pb = recorded_pb["input_file"]
                output_pb = recorded_pb["output_file"]
                run_id_pb = recorded_pb.get("run_id", "")
                if status_pb == "fetched":
                    logger.info(f"  {input_pb}: already fetched. Skipping pilot_b.")
                else:
                    logger.info(f"  {input_pb}: already {status_pb}. Resuming...")
            else:
                run_id_pb = self._new_run_id()
                input_pb = self._input_filename(0, run_id_pb)
                output_pb = self._output_filename(0, run_id_pb)
                self._generate_input(self.pilot_steps, os.path.join(pilot_b_dir, input_pb))
            mode_info = f"nmpm={self.num_projection_per_measurement}" if self._use_gfmc_n else f"tau={self.time_projection_tau}"
            logger.info("")
            logger.info(f"-- LRDMC Phase B: Error-bar Pilot (a={self.alat}) " + "-" * 12)
            logger.info(f"  {input_pb}: {self.pilot_steps} steps, {mode_info} (queue: {self.pilot_queue_label})")

            pilot_t0 = time.monotonic()
            await self._submit_and_wait(
                input_pb, output_pb, work_dir=pilot_b_dir, queue_label=self.pilot_queue_label, step=0, run_id=run_id_pb
            )
            pilot_wall_sec = time.monotonic() - pilot_t0

            restart_chk = self._find_restart_chk(pilot_b_dir)
            if not restart_chk:
                raise RuntimeError("No checkpoint found after pilot run. Cannot estimate required steps.")

            _, pilot_error = self._compute_energy(restart_chk, work_dir=pilot_b_dir, output_file=output_pb)
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
                min_steps=self.num_gfmc_bin_blocks,
            )
            # Add warmup back: production also discards warmup steps
            estimated_steps += self.num_gfmc_warmup_steps

            # Time estimate: only Net time scales with step count
            step_ratio = estimated_steps / self.pilot_steps if self.pilot_steps > 0 else 0
            pilot_output_path = os.path.join(pilot_b_dir, output_pb)
            net_pilot_sec = parse_net_time(pilot_output_path)
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
                f"  {'nmpm' if self._use_gfmc_n else 'tau':17s} = {self.num_projection_per_measurement if self._use_gfmc_n else self.time_projection_tau}\n"
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
            est_kwargs = dict(
                pilot_steps=self.pilot_steps,
                pilot_error=pilot_error,
                target_error=self.target_error,
                estimated_steps=estimated_steps,
                pilot_queue_label=self.pilot_queue_label,
                walker_ratio=walker_ratio,
                pilot_wall_sec=pilot_wall_sec,
                net_pilot_sec=net_pilot_sec or 0,
            )
            if self._use_gfmc_n:
                est_kwargs["num_projection_per_measurement"] = self.num_projection_per_measurement
            else:
                est_kwargs["time_projection_tau"] = self.time_projection_tau
            set_estimation(_wd, **est_kwargs)

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
            restart_chk = self._find_restart_chk(_wd)
            if restart_chk:
                energy, error = self._compute_energy(restart_chk, work_dir=_wd)
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
        cached_energy = estimation.get("last_energy")
        cached_error = estimation.get("last_energy_error")
        if cached_energy is not None and cached_error is not None:
            if cached_error <= self.target_error * 1.20:
                restart_chk = self._find_restart_chk(_wd)
                logger.info(
                    f"  Target already achieved (cached): {cached_error:.6g} <= {self.target_error * 1.20:.6g} Ha (target*1.20)"
                )
                self.output_values.update(
                    energy=cached_energy,
                    energy_error=cached_error,
                    alat=self.alat,
                    restart_chk=restart_chk or "",
                    estimated_steps=estimated_steps,
                    num_projection_per_measurement=self.num_projection_per_measurement,
                )
                if self.atomic_force and restart_chk:
                    forces = self._compute_force(restart_chk, work_dir=_wd)
                    if forces is not None:
                        self.output_values["forces"] = forces
                self.output_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.h5")))
                self.status = WorkflowStatus.COMPLETED
                return self.status, self.output_files, self.output_values

        # ── Production runs (phase 1..N) ──────────────────────────
        # Three phases:
        #   A. Scan existing runs → find resume point
        #   B. Re-estimate from accumulated data (if resuming)
        #   C. Production loop for remaining runs
        #
        # Checkpoint preserves all accumulated statistics across
        # production restarts.  accumulated_measurement tracks
        # total measurement steps (excluding warmup) and is
        # persisted in estimation state for correct resume.

        warmup = self.num_gfmc_warmup_steps
        step_files = {}  # {step: (input, output, run_id)}
        last_run = 0
        first_new_run = self.max_continuation + 1  # assume all done

        # ── Phase A: scan existing runs ──
        for i in range(1, self.max_continuation + 1):
            recorded = get_job_by_step(_wd, i)
            status_i = recorded.get("status")
            if status_i == "fetched":
                step_files[i] = (recorded["input_file"], recorded["output_file"], recorded.get("run_id", ""))
                logger.info(f"  step {i}: already fetched. Skipping.")
                last_run = i
            else:
                first_new_run = i
                break

        # ── Phase B: re-estimate from accumulated data ──
        accumulated_measurement = 0  # measurement steps only (excl. warmup)
        if first_new_run > 1:
            cached_accum = estimation.get("accumulated_measurement_steps")
            if cached_accum is not None:
                accumulated_measurement = int(cached_accum)
            else:
                accumulated_measurement = (first_new_run - 1) * max(estimated_steps - warmup, 0)

            _re_chk = self._find_restart_chk(_wd)
            if _re_chk:
                _re_energy, _re_error = self._compute_energy(_re_chk, work_dir=_wd)
                if _re_energy is not None and _re_error is not None:
                    if _re_error <= self.target_error * 1.20:
                        logger.info(
                            f"  Target already met after prior runs: {_re_error:.6g} <= {self.target_error * 1.20:.6g} Ha"
                        )
                        self.output_values.update(
                            energy=_re_energy,
                            energy_error=_re_error,
                            alat=self.alat,
                            restart_chk=_re_chk,
                        )
                        if self.atomic_force:
                            forces = self._compute_force(_re_chk, work_dir=_wd)
                            if forces is not None:
                                self.output_values["forces"] = forces
                        first_new_run = self.max_continuation + 1  # skip loop
                    else:
                        _additional = estimate_additional_steps(
                            accumulated_measurement,
                            _re_error,
                            self.target_error,
                        )
                        estimated_steps = _additional + warmup
                        logger.info(
                            f"  Resuming after {first_new_run - 1} prior run(s): "
                            f"error={_re_error:.6g} Ha > target "
                            f"{self.target_error:.6g} Ha -> "
                            f"{estimated_steps} steps "
                            f"(measurement: {_additional}, warmup: {warmup}, "
                            f"accumulated measurement: {accumulated_measurement})"
                        )

        # ── Phase C: production loop ──
        _prev_run_steps = None
        for i in range(first_new_run, self.max_continuation + 1):
            recorded = get_job_by_step(_wd, i)
            status_i = recorded.get("status")

            if status_i in ("submitted", "completed"):
                input_i = recorded["input_file"]
                output_i = recorded["output_file"]
                run_id_i = recorded.get("run_id", "")
                logger.info(f"  step {i}: already {status_i}. Resuming...")
            else:
                run_id_i = self._new_run_id()
                input_i = self._input_filename(i, run_id_i)
                output_i = self._output_filename(i, run_id_i)
                # First-ever production run starts fresh; all others restart
                if i == 1 and first_new_run == 1:
                    self._generate_input(estimated_steps, os.path.join(_wd, input_i))
                else:
                    restart_chk = self._find_restart_chk(_wd)
                    if restart_chk is None:
                        raise RuntimeError(f"No restart checkpoint found for run {i}. Expected .h5 file in {_wd}")
                    self._generate_input(
                        estimated_steps,
                        os.path.join(_wd, input_i),
                        restart=True,
                        restart_chk=restart_chk,
                    )
                logger.info("")
                logger.info(f"-- LRDMC Phase 1: Production run {i}/{self.max_continuation} (a={self.alat}) " + "-" * 10)
                logger.info(f"  {input_i}: {estimated_steps} steps")

            step_files[i] = (input_i, output_i, run_id_i)
            need_restart = i > 1 or first_new_run > 1
            restart_chk = self._find_restart_chk(_wd) if need_restart else None
            if need_restart and restart_chk is None:
                raise RuntimeError(f"No restart checkpoint found for run {i}. Expected .h5 file in {_wd}")
            extra_from = [restart_chk] if restart_chk else []

            # Estimate run time from the latest available output
            _ref_net = None
            for _j in range(i - 1, 0, -1):
                if _j not in step_files:
                    continue
                _ref_net = parse_net_time(os.path.join(_wd, step_files[_j][1]))
                if _ref_net and _ref_net > 0:
                    _ref_steps = _prev_run_steps if _prev_run_steps else estimated_steps
                    _est_sec = _ref_net * (estimated_steps / _ref_steps) if _ref_steps > 0 else _ref_net
                    logger.info(f"  est. Net run time (w/o JAX compilation) = {_format_duration(_est_sec)}")
                    break
            else:
                _pilot_outs = sorted(glob.glob(os.path.join(_wd, "_pilot_b", "*.out")))
                _ref_net = parse_net_time(_pilot_outs[-1]) if _pilot_outs else None
                if _ref_net and _ref_net > 0:
                    _p_steps = estimation.get("pilot_steps") or self.pilot_steps
                    if _p_steps > 0:
                        logger.info(
                            f"  est. Net run time (w/o JAX compilation) = {_format_duration(_ref_net * estimated_steps / _p_steps)}"
                        )

            await self._submit_and_wait(
                input_i,
                output_i,
                work_dir=_wd,
                extra_from_objects=extra_from,
                step=i,
                run_id=run_id_i,
            )
            accumulated_measurement += estimated_steps - warmup
            _prev_run_steps = estimated_steps
            last_run = i

            # ── Side-effects: compute energy from checkpoint (if any) ──
            restart_chk = self._find_restart_chk(_wd)
            energy = error = None
            if restart_chk:
                energy, error = self._compute_energy(restart_chk, work_dir=_wd, output_file=output_i)
                if energy is not None:
                    self.output_values["energy"] = energy
                    self.output_values["energy_error"] = error
                    self.output_values["alat"] = self.alat
                    self.output_values["restart_chk"] = restart_chk
                    logger.info(f"  LRDMC energy (a={self.alat}): {energy} +- {error} Ha")
                    if self.atomic_force:
                        forces = self._compute_force(restart_chk, work_dir=_wd, output_file=output_i)
                        if forces is not None:
                            self.output_values["forces"] = forces

                    set_estimation(
                        _wd,
                        last_energy=energy,
                        last_energy_error=error,
                        accumulated_measurement_steps=accumulated_measurement,
                        last_num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
                        last_num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
                        last_num_gfmc_collect_steps=self.num_gfmc_collect_steps,
                    )

            # ── Termination decision — single source of truth ──
            vstatus, vmsg = validate_completion(
                _wd,
                self.output_values,
                target_error=self.target_error,
                target_tol=1.20,
            )
            if vstatus == CompletionStatus.FAILED:
                logger.error(vmsg)
                self.output_values["error"] = vmsg
                self.status = WorkflowStatus.FAILED
                break
            if vstatus == CompletionStatus.OK:
                logger.info(f"  Target error achieved: {vmsg} (run {i}/{self.max_continuation})")
                break
            # INCOMPLETE — prepare next iteration if we have an error estimate
            if energy is not None:
                if i < self.max_continuation:
                    old_steps = estimated_steps
                    _additional = estimate_additional_steps(
                        accumulated_measurement,
                        error,
                        self.target_error,
                    )
                    estimated_steps = _additional + warmup
                    logger.info(
                        f"  Re-estimated: {old_steps} -> {estimated_steps} steps "
                        f"(measurement: {_additional}, warmup: {warmup}, "
                        f"accumulated measurement: {accumulated_measurement})"
                    )
                else:
                    logger.warning(
                        f"Error {error:.6g} > target "
                        f"{self.target_error:.6g} Ha -- "
                        f"max_continuation ({self.max_continuation}) reached"
                    )

        # ── Final energy computation (safety net) ─────────────────
        last_output = step_files[last_run][1] if last_run in step_files else None
        restart_chk = self._find_restart_chk(_wd)
        if restart_chk:
            energy, error = self._compute_energy(restart_chk, work_dir=_wd, output_file=last_output)
            if energy is not None:
                self.output_values["energy"] = energy
                self.output_values["energy_error"] = error
                self.output_values["alat"] = self.alat
                self.output_values["restart_chk"] = restart_chk
                if self.atomic_force:
                    forces = self._compute_force(restart_chk, work_dir=_wd, output_file=last_output)
                    if forces is not None:
                        self.output_values["forces"] = forces
                set_estimation(
                    _wd,
                    last_energy=energy,
                    last_energy_error=error,
                    accumulated_measurement_steps=accumulated_measurement,
                    last_num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
                    last_num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
                    last_num_gfmc_collect_steps=self.num_gfmc_collect_steps,
                )

        # ── Collect outputs ───────────────────────────────────────
        chk_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.h5")))
        output_logs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.out")))
        self.output_files = chk_files + output_logs
        self.output_values["estimated_steps"] = estimated_steps
        if self._use_gfmc_n:
            self.output_values["num_projection_per_measurement"] = self.num_projection_per_measurement
        else:
            self.output_values["time_projection_tau"] = self.time_projection_tau

        if self.status != WorkflowStatus.FAILED:
            self.status = WorkflowStatus.COMPLETED
        return self.status, self.output_files, self.output_values

    # ── Utility methods ───────────────────────────────────────────

    def _find_restart_chk(self, work_dir: str) -> Optional[str]:
        """Locate the LRDMC restart checkpoint file in *work_dir*."""
        for pattern in ["restart.h5", "lrdmc.h5", "*.h5"]:
            matches = sorted(glob.glob(os.path.join(work_dir, pattern)))
            if matches:
                return os.path.basename(matches[-1])
        return None

    def _compute_energy(self, restart_chk: str, work_dir: str, output_file: Optional[str] = None):
        """Parse energy from *output_file* or run ``jqmc-tool lrdmc compute-energy``.

        When *output_file* is given the energy is read directly from
        the ``jqmc`` stdout (``Total Energy: E = … +- … Ha.``).
        This avoids the overhead of re-running ``jqmc-tool`` when
        the post-processing parameters (-b, -w, -c) are the same as
        in the input TOML — which is always the case for a fresh run.

        Falls back to ``jqmc-tool`` when *output_file* is *None* or
        when stdout parsing fails.

        Parameters
        ----------
        restart_chk : str
            Checkpoint filename (basename).
        work_dir : str
            Directory in which to run the command.
        output_file : str, optional
            Stdout filename (basename) of the ``jqmc`` run.

        Returns
        -------
        tuple
            ``(energy, error)`` or ``(None, None)``.
        """
        # Fast path: parse from jqmc stdout
        if output_file is not None:
            out_path = os.path.join(work_dir, output_file)
            if os.path.isfile(out_path):
                try:
                    with open(out_path) as fh:
                        text = fh.read()
                    energy, error = self._parse_energy_output(text)
                    if energy is not None:
                        logger.info(f"  Energy from {output_file} (jqmc-tool skipped): E = {energy} +- {error} Ha")
                        return energy, error
                except OSError:
                    pass

        # Fallback: jqmc-tool
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
                cwd=work_dir,
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

    def _compute_force(self, restart_chk: str, work_dir: str, output_file: Optional[str] = None):
        """Parse forces from *output_file* or run ``jqmc-tool lrdmc compute-force``.

        When *output_file* is given, forces are read directly from
        the ``jqmc`` stdout (``Atomic Forces:`` table).  Falls back
        to ``jqmc-tool`` when *output_file* is *None* or parsing fails.

        Parameters
        ----------
        restart_chk : str
            Checkpoint filename (basename).
        work_dir : str
            Directory in which to run the command.
        output_file : str, optional
            Stdout filename (basename) of the ``jqmc`` run.

        Returns
        -------
        list of dict or None
            Each dict has keys ``label``, ``Fx``, ``Fx_err``,
            ``Fy``, ``Fy_err``, ``Fz``, ``Fz_err``.
            Returns *None* on failure.
        """
        # Fast path: parse from jqmc stdout
        if output_file is not None:
            out_path = os.path.join(work_dir, output_file)
            if os.path.isfile(out_path):
                try:
                    with open(out_path) as fh:
                        text = fh.read()
                    forces = parse_force_table(text)
                    if forces:
                        logger.info(f"  Forces from {output_file} (jqmc-tool skipped):")
                        for f in forces:
                            logger.info(
                                f"  {f['label']:8s}"
                                f" Fx={f['Fx']:+.6f}+-{f['Fx_err']:.6f}"
                                f" Fy={f['Fy']:+.6f}+-{f['Fy_err']:.6f}"
                                f" Fz={f['Fz']:+.6f}+-{f['Fz_err']:.6f}"
                                f" Ha/bohr"
                            )
                        return forces
                except OSError:
                    pass

        # Fallback: jqmc-tool
        cmd = (
            f"jqmc-tool lrdmc compute-force {restart_chk} "
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
                cwd=work_dir,
            )
            forces = parse_force_table(result.stdout)
            if forces:
                for f in forces:
                    logger.info(
                        f"  {f['label']:8s}"
                        f" Fx={f['Fx']:+.6f}+-{f['Fx_err']:.6f}"
                        f" Fy={f['Fy']:+.6f}+-{f['Fy_err']:.6f}"
                        f" Fz={f['Fz']:+.6f}+-{f['Fz_err']:.6f}"
                        f" Ha/bohr"
                    )
            return forces
        except subprocess.CalledProcessError as e:
            logger.error(f"compute-force failed: {e.stderr}")
            return None
