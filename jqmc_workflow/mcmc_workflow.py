"""MCMC_Workflow — MCMC production run (sampling) via ``jqmc`` (job_type=mcmc).

Generates an MCMC input TOML, submits ``jqmc`` on a remote/local machine,
monitors until completion, fetches results, and post-processes the checkpoint
with ``jqmc-tool mcmc compute-energy`` to extract the VMC energy ± error.
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
from ._job import get_num_mpi, load_queue_data
from ._output_parser import parse_force_table
from ._state import get_estimation, get_job, set_estimation
from .workflow import Workflow

logger = getLogger("jqmc-workflow").getChild(__name__)


class MCMC_Workflow(Workflow):
    """MCMC (VMC production-run / sampling) workflow.

    Generates a ``job_type=mcmc`` input TOML, submits ``jqmc`` on a
    remote or local machine, monitors until completion, fetches results,
    and post-processes the checkpoint with
    ``jqmc-tool mcmc compute-energy`` to obtain the VMC energy ± error.

    The workflow supports two modes:

    **Automatic mode** (default, ``num_mcmc_steps=None``):

    1. **Pilot run** (``_0``) — A short MCMC run with ``pilot_steps``
       measurement steps.  The resulting statistical error is used to
       estimate the total steps required for ``target_error`` via the
       CLT scaling $\\sigma \\propto 1/\\sqrt{N}$.
    2. **Production runs** (``_1``, ``_2``, …) — Continuation runs
       with the estimated step count.  After each run, the checkpoint
       is post-processed; if the error is at or below ``target_error``
       the loop terminates.  At most ``max_continuation`` production
       runs are attempted.

    **Fixed-step mode** (``num_mcmc_steps`` is set):

    The pilot run is skipped entirely and ``target_error`` is ignored.
    Each production run uses exactly ``num_mcmc_steps`` measurement
    steps, and ``max_continuation`` runs are executed unconditionally.

    Parameters
    ----------
    server_machine_name : str
        Name of the target machine (configured in ``~/.jqmc_setting/``).
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
    num_mcmc_bin_blocks : int
        Binning blocks for post-processing.
    num_mcmc_warmup_steps : int
        Warmup steps to discard in post-processing.
    Dt : float, optional
        MCMC step size (bohr).  Default from ``jqmc_miscs``.
    epsilon_AS : float, optional
        Attacalite-Sorella regularization parameter.  Default from ``jqmc_miscs``.
    num_mcmc_per_measurement : int, optional
        MCMC updates per measurement.  Default from ``jqmc_miscs``.
    atomic_force : bool, optional
        Compute atomic forces.  Default from ``jqmc_miscs``.
    parameter_derivatives : bool, optional
        Compute parameter derivatives.  Default from ``jqmc_miscs``.
    mcmc_seed : int, optional
        Random seed for MCMC.  Default from ``jqmc_miscs``.
    verbosity : str, optional
        Verbosity level.  Default from ``jqmc_miscs``.
    poll_interval : int
        Seconds between job-status polls.
    target_error : float
        Target statistical error (Ha).  Ignored when
        *num_mcmc_steps* is set.
    num_mcmc_steps : int, optional
        Fixed number of measurement steps per production run.  When
        set, the pilot run is skipped and ``target_error`` is ignored;
        each of the ``max_continuation`` production runs uses exactly
        this many steps.
    pilot_steps : int
        Measurement steps for the pilot estimation run.  Ignored when
        *num_mcmc_steps* is set.
    pilot_queue_label : str, optional
        Queue label for the pilot run.  Defaults to *queue_label*.
        Use a shorter/smaller queue for the pilot to save resources.
    max_continuation : int
        Maximum number of production runs after the pilot.

    Examples
    --------
    Standalone launch (automatic mode)::

        wf = MCMC_Workflow(
            server_machine_name="cluster",
            target_error=0.0005,
            pilot_steps=200,
            number_of_walkers=8,
        )
        status, files, values = wf.launch()
        print(values["energy"], values["energy_error"])

    Fixed-step mode (no pilot, no target_error check)::

        wf = MCMC_Workflow(
            server_machine_name="cluster",
            num_mcmc_steps=5000,
            number_of_walkers=8,
            max_continuation=3,
        )
        status, files, values = wf.launch()

    As part of a :class:`Launcher` pipeline::

        enc = Container(
            label="mcmc",
            dirname="02_mcmc",
            input_files=[FileFrom("vmc-opt", "hamiltonian_data_opt_step_9.h5")],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=MCMC_Workflow(
                server_machine_name="cluster",
                target_error=0.001,
            ),
        )

    Notes
    -----
    * The pilot run is skipped on re-entrance if an estimation already
      exists in ``workflow_state.toml``.
    * Continuation runs restart from the most recent ``.h5``
      checkpoint file.

    See Also
    --------
    VMC_Workflow : Wavefunction optimisation (job_type=vmc).
    LRDMC_Workflow : Diffusion Monte Carlo (job_type=lrdmc-bra / lrdmc-tau).
    """

    def __init__(
        self,
        server_machine_name: str = "localhost",
        hamiltonian_file: str = "hamiltonian_data.h5",
        input_file: str = "input.toml",
        output_file: str = "out.o",
        queue_label: str = "default",
        jobname: str = "jqmc-mcmc",
        number_of_walkers: int = 4,
        max_time: int = 86400,
        num_mcmc_bin_blocks: int = 1,
        num_mcmc_warmup_steps: int = 0,
        # -- [mcmc] section parameters --
        Dt: Optional[float] = None,
        epsilon_AS: Optional[float] = None,
        num_mcmc_per_measurement: Optional[int] = None,
        atomic_force: Optional[bool] = None,
        parameter_derivatives: Optional[bool] = None,
        # -- [control] section parameters --
        mcmc_seed: Optional[int] = None,
        verbosity: Optional[str] = None,
        # -- workflow parameters --
        poll_interval: int = 60,
        target_error: float = 0.001,
        num_mcmc_steps: Optional[int] = None,
        pilot_steps: int = 100,
        pilot_queue_label: Optional[str] = None,
        max_continuation: int = 1,
    ):
        super().__init__()
        self.server_machine_name = server_machine_name
        self.hamiltonian_file = hamiltonian_file
        self.input_file = input_file
        self.output_file = output_file
        self.queue_label = queue_label
        self.jobname = jobname
        self.number_of_walkers = number_of_walkers
        self.max_time = max_time
        self.num_mcmc_bin_blocks = num_mcmc_bin_blocks
        self.num_mcmc_warmup_steps = num_mcmc_warmup_steps
        # [mcmc] section
        self.Dt = Dt
        self.epsilon_AS = epsilon_AS
        self.num_mcmc_per_measurement = num_mcmc_per_measurement
        self.atomic_force = atomic_force
        self.parameter_derivatives = parameter_derivatives
        # [control] section
        self.mcmc_seed = mcmc_seed
        self.verbosity = verbosity
        # workflow
        self.poll_interval = poll_interval
        self.target_error = target_error
        self.num_mcmc_steps = num_mcmc_steps
        self.pilot_steps = pilot_steps
        self.pilot_queue_label = pilot_queue_label or queue_label
        self.max_continuation = max_continuation

    # ── Input generation ──────────────────────────────────────────

    def _generate_input(
        self,
        num_steps,
        input_file,
        restart=False,
        restart_chk=None,
    ):
        """Generate MCMC TOML input file."""
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
        mcmc_ov = resolve_with_defaults(
            "mcmc",
            {
                "num_mcmc_steps": num_steps,
                "Dt": self.Dt,
                "epsilon_AS": self.epsilon_AS,
                "num_mcmc_per_measurement": self.num_mcmc_per_measurement,
                "num_mcmc_warmup_steps": self.num_mcmc_warmup_steps,
                "num_mcmc_bin_blocks": self.num_mcmc_bin_blocks,
                "atomic_force": self.atomic_force,
                "parameter_derivatives": self.parameter_derivatives,
            },
        )
        overrides = {
            "control": control_ov,
            "mcmc": mcmc_ov,
        }
        generate_input_toml(
            job_type="mcmc",
            overrides=overrides,
            filename=input_file,
        )

    # ── Submit / poll / fetch ─────────────────────────────────────
    # _submit_and_wait() and _make_job() are inherited from Workflow.

    # ── Launch ────────────────────────────────────────────────────

    async def async_launch(self):
        """Run the MCMC workflow.

        **Fixed-step mode** (``num_mcmc_steps`` is set):
        The pilot run is skipped.  Each production run uses exactly
        ``num_mcmc_steps`` steps and all ``max_continuation`` runs
        are executed unconditionally.

        **Automatic mode** (``num_mcmc_steps`` is *None*, default):

        1. Pilot run in ``_pilot/`` subdirectory estimates required steps
           (skipped on continuation).  May use a different queue from
           production (``pilot_queue_label``).
        2. Production runs (``_1``, ``_2``, ...) start from scratch and
           accumulate statistics until ``target_error`` is achieved or
           ``max_continuation`` is reached.
        """
        self._ensure_project_dir()
        _wd = self.project_dir

        # ── Fixed-step mode ───────────────────────────────────────
        if self.num_mcmc_steps is not None:
            return await self._launch_fixed_steps(_wd)

        # ── Automatic mode (pilot + target_error) ─────────────────
        return await self._launch_auto(_wd)

    async def _launch_fixed_steps(self, _wd):
        """Fixed-step production: skip pilot, ignore target_error."""
        estimated_steps = self.num_mcmc_steps
        logger.info("")
        logger.info("-- MCMC Fixed-step mode " + "-" * 26)
        logger.info(f"  num_mcmc_steps    = {estimated_steps}\n  max_continuation  = {self.max_continuation}")

        last_run = 0
        for i in range(1, self.max_continuation + 1):
            input_i = suffixed_name(self.input_file, i)
            output_i = suffixed_name(self.output_file, i)

            recorded = get_job(_wd, input_i)
            if recorded.get("status") in ("submitted", "completed", "fetched"):
                if recorded["status"] == "fetched":
                    logger.info(f"  {input_i}: already fetched. Skipping.")
                    last_run = i
                    continue
                logger.info(f"  {input_i}: already {recorded['status']}. Resuming...")
            else:
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
                logger.info(f"-- MCMC Production run {i}/{self.max_continuation} " + "-" * 10)
                logger.info(f"  {input_i}: {estimated_steps} steps")

            restart_chk = self._find_restart_chk(_wd) if i > 1 else None
            if i > 1 and restart_chk is None:
                raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {_wd}")
            extra_from = [restart_chk] if restart_chk else []

            await self._submit_and_wait(
                input_i,
                output_i,
                work_dir=_wd,
                extra_from_objects=extra_from,
            )
            last_run = i

            # Post-process energy (informational only, no convergence check)
            restart_chk = self._find_restart_chk(_wd)
            if restart_chk:
                energy, error = self._compute_energy(restart_chk, work_dir=_wd)
                if energy is not None:
                    self.output_values["energy"] = energy
                    self.output_values["energy_error"] = error
                    self.output_values["restart_chk"] = restart_chk
                    logger.info(f"  MCMC energy: {energy} +- {error} Ha")
                    if self.atomic_force:
                        forces = self._compute_force(restart_chk, work_dir=_wd)
                        if forces is not None:
                            self.output_values["forces"] = forces
                    set_estimation(
                        _wd,
                        last_energy=energy,
                        last_energy_error=error,
                        last_num_mcmc_bin_blocks=self.num_mcmc_bin_blocks,
                        last_num_mcmc_warmup_steps=self.num_mcmc_warmup_steps,
                    )

        # ── Final energy computation ─────────────────────────────
        restart_chk = self._find_restart_chk(_wd)
        if restart_chk:
            energy, error = self._compute_energy(restart_chk, work_dir=_wd)
            if energy is not None:
                self.output_values["energy"] = energy
                self.output_values["energy_error"] = error
                self.output_values["restart_chk"] = restart_chk
                if self.atomic_force:
                    forces = self._compute_force(restart_chk, work_dir=_wd)
                    if forces is not None:
                        self.output_values["forces"] = forces
                set_estimation(
                    _wd,
                    last_energy=energy,
                    last_energy_error=error,
                    last_num_mcmc_bin_blocks=self.num_mcmc_bin_blocks,
                    last_num_mcmc_warmup_steps=self.num_mcmc_warmup_steps,
                )

        # ── Collect outputs ───────────────────────────────────────
        chk_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.h5")))
        output_logs = [
            suffixed_name(self.output_file, j)
            for j in range(last_run + 1)
            if os.path.isfile(os.path.join(_wd, suffixed_name(self.output_file, j)))
        ]
        self.output_files = chk_files + output_logs
        self.output_values["num_mcmc_steps"] = estimated_steps

        self.status = "success"
        return self.status, self.output_files, self.output_values

    async def _launch_auto(self, _wd):
        """Automatic mode: pilot + target_error convergence."""
        # -- Phase 0: pilot estimation (skip on continuation) ------
        estimation = get_estimation(_wd)

        if estimation.get("estimated_steps") is not None:
            estimated_steps = int(estimation["estimated_steps"])
            logger.info(f"Estimation already done (continuation): estimated_steps={estimated_steps}. Skipping pilot.")
        else:
            # ── Run pilot in a subdirectory ───────────────────────
            pilot_dir = os.path.join(_wd, "_pilot")
            os.makedirs(pilot_dir, exist_ok=True)

            # Symlink hamiltonian file into pilot dir
            h5_src = os.path.join(_wd, self.hamiltonian_file)
            h5_dst = os.path.join(pilot_dir, self.hamiltonian_file)
            if os.path.isfile(h5_src) and not os.path.exists(h5_dst):
                os.symlink(h5_src, h5_dst)

            input_0 = suffixed_name(self.input_file, 0)
            output_0 = suffixed_name(self.output_file, 0)

            recorded_0 = get_job(pilot_dir, input_0)
            if recorded_0.get("status") not in ("submitted", "completed", "fetched"):
                self._generate_input(self.pilot_steps, os.path.join(pilot_dir, input_0))
            else:
                logger.info(f"  {input_0}: already {recorded_0['status']}. Resuming...")
            logger.info("")
            logger.info("-- MCMC Phase 0: Pilot " + "-" * 27)
            logger.info(f"  {input_0}: {self.pilot_steps} steps (queue: {self.pilot_queue_label})")

            pilot_t0 = time.monotonic()
            await self._submit_and_wait(input_0, output_0, work_dir=pilot_dir, queue_label=self.pilot_queue_label)
            pilot_wall_sec = time.monotonic() - pilot_t0

            restart_chk = self._find_restart_chk(pilot_dir)
            if not restart_chk:
                raise RuntimeError("No checkpoint found after pilot run. Cannot estimate required steps.")

            _, pilot_error = self._compute_energy(restart_chk, work_dir=pilot_dir)
            if pilot_error is None:
                raise RuntimeError("Could not parse energy error from pilot run.")

            # Walker ratio: pilot queue may have different MPI count
            pilot_qd = load_queue_data(self.server_machine_name, self.pilot_queue_label)
            prod_qd = load_queue_data(self.server_machine_name, self.queue_label)
            pilot_mpi = get_num_mpi(pilot_qd)
            prod_mpi = get_num_mpi(prod_qd)
            walker_ratio = pilot_mpi / prod_mpi

            estimated_steps = estimate_required_steps(
                self.pilot_steps - self.num_mcmc_warmup_steps,
                pilot_error,
                self.target_error,
                walker_ratio=walker_ratio,
            )
            # Add warmup back: production also discards warmup steps
            estimated_steps += self.num_mcmc_warmup_steps

            # Time estimate: only Net time scales with step count
            step_ratio = estimated_steps / self.pilot_steps if self.pilot_steps > 0 else 0
            pilot_output_path = os.path.join(pilot_dir, output_0)
            net_pilot_sec = parse_net_time(pilot_output_path)
            if net_pilot_sec is not None and net_pilot_sec > 0:
                overhead_sec = pilot_wall_sec - net_pilot_sec
                est_prod_sec = overhead_sec + net_pilot_sec * step_ratio
            else:
                est_prod_sec = pilot_wall_sec * step_ratio
            logger.info("")
            logger.info("-- MCMC Step Estimation Summary " + "-" * 18)
            logger.info(
                f"  pilot steps       = {self.pilot_steps}\n"
                f"  warmup steps      = {self.num_mcmc_warmup_steps}\n"
                f"  pilot error       = {pilot_error:.6g} Ha\n"
                f"  target error      = {self.target_error:.6g} Ha\n"
                f"  pilot MPI procs   = {pilot_mpi}\n"
                f"  prod. MPI procs   = {prod_mpi}\n"
                f"  walker ratio      = {walker_ratio:.4g}\n"
                f"  estimated steps   = {estimated_steps}\n"
                f"  pilot wall time   = {_format_duration(pilot_wall_sec)}\n"
                f"  pilot net time    = "
                f"{_format_duration(net_pilot_sec) if net_pilot_sec else 'N/A'}\n"
                f"  est. prod. time   = {_format_duration(est_prod_sec)}"
            )
            logger.info("-" * 50)

            # Save estimation to main working directory
            set_estimation(
                _wd,
                pilot_steps=self.pilot_steps,
                pilot_error=pilot_error,
                target_error=self.target_error,
                estimated_steps=estimated_steps,
                pilot_queue_label=self.pilot_queue_label,
                walker_ratio=walker_ratio,
            )

        # ── Re-compute energy if post-processing parameters changed ──
        _postproc_changed = (
            estimation.get("last_num_mcmc_bin_blocks") != self.num_mcmc_bin_blocks
            or estimation.get("last_num_mcmc_warmup_steps") != self.num_mcmc_warmup_steps
        )
        if _postproc_changed and estimation.get("last_energy") is not None:
            logger.info(
                "  Post-processing parameters changed "
                f"(bin_blocks: {estimation.get('last_num_mcmc_bin_blocks')}"
                f" -> {self.num_mcmc_bin_blocks}, "
                f"warmup: {estimation.get('last_num_mcmc_warmup_steps')}"
                f" -> {self.num_mcmc_warmup_steps}); "
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
                        last_num_mcmc_bin_blocks=self.num_mcmc_bin_blocks,
                        last_num_mcmc_warmup_steps=self.num_mcmc_warmup_steps,
                    )
                    estimation = get_estimation(_wd)

        # ── Early exit if target already met ──────────────────────
        cached_energy = estimation.get("last_energy")
        cached_error = estimation.get("last_energy_error")
        if cached_energy is not None and cached_error is not None:
            if cached_error <= self.target_error * 1.05:
                restart_chk = self._find_restart_chk(_wd)
                logger.info(
                    f"  Target already achieved (cached): {cached_error:.6g} <= {self.target_error * 1.05:.6g} Ha (target*1.05)"
                )
                self.output_values.update(
                    energy=cached_energy,
                    energy_error=cached_error,
                    restart_chk=restart_chk or "",
                    estimated_steps=estimated_steps,
                )
                if self.atomic_force and restart_chk:
                    forces = self._compute_force(restart_chk, work_dir=_wd)
                    if forces is not None:
                        self.output_values["forces"] = forces
                self.output_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.h5")))
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
                logger.info(f"  {input_i}: already {recorded['status']}. Resuming...")
            else:
                if i == 1:
                    # First production run: start from scratch
                    self._generate_input(estimated_steps, os.path.join(_wd, input_i))
                else:
                    restart_chk = self._find_restart_chk(_wd)
                    if restart_chk is None:
                        raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {_wd}")
                    self._generate_input(
                        estimated_steps,
                        os.path.join(_wd, input_i),
                        restart=bool(restart_chk),
                        restart_chk=restart_chk,
                    )
                logger.info("")
                logger.info(f"-- MCMC Phase 1: Production run {i}/{self.max_continuation} " + "-" * 10)
                logger.info(f"  {input_i}: {estimated_steps} steps")

            restart_chk = self._find_restart_chk(_wd) if i > 1 else None
            if i > 1 and restart_chk is None:
                raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {_wd}")
            extra_from = [restart_chk] if restart_chk else []

            await self._submit_and_wait(
                input_i,
                output_i,
                work_dir=_wd,
                extra_from_objects=extra_from,
            )
            accumulated_steps += estimated_steps
            last_run = i

            # Check convergence
            restart_chk = self._find_restart_chk(_wd)
            if restart_chk:
                energy, error = self._compute_energy(restart_chk, work_dir=_wd)
                if energy is not None:
                    self.output_values["energy"] = energy
                    self.output_values["energy_error"] = error
                    self.output_values["restart_chk"] = restart_chk
                    logger.info(f"  MCMC energy: {energy} +- {error} Ha")
                    if self.atomic_force:
                        forces = self._compute_force(restart_chk, work_dir=_wd)
                        if forces is not None:
                            self.output_values["forces"] = forces

                    # Cache for restart
                    set_estimation(
                        _wd,
                        last_energy=energy,
                        last_energy_error=error,
                        last_num_mcmc_bin_blocks=self.num_mcmc_bin_blocks,
                        last_num_mcmc_warmup_steps=self.num_mcmc_warmup_steps,
                    )

                    if error <= self.target_error * 1.05:
                        logger.info(
                            f"  Target error achieved: {error:.6g} <= "
                            f"{self.target_error * 1.05:.6g} Ha (target*1.05) "
                            f"(run {i}/{self.max_continuation})"
                        )
                        break
                    elif i < self.max_continuation:
                        old_steps = estimated_steps
                        estimated_steps = estimate_additional_steps(
                            accumulated_steps,
                            error,
                            self.target_error,
                        )
                        logger.info(
                            f"  Re-estimated: {old_steps} -> {estimated_steps} additional steps (accumulated so far: {accumulated_steps})"
                        )
                    else:
                        msg = (
                            f"Error {error:.6g} > target "
                            f"{self.target_error:.6g} Ha -- "
                            f"max_continuation ({self.max_continuation}) reached"
                        )
                        logger.error(msg)
                        self.status = "failed"
                        raise RuntimeError(msg)

        # ── Final energy computation ─────────────────────────────
        restart_chk = self._find_restart_chk(_wd)
        if restart_chk:
            energy, error = self._compute_energy(restart_chk, work_dir=_wd)
            if energy is not None:
                self.output_values["energy"] = energy
                self.output_values["energy_error"] = error
                self.output_values["restart_chk"] = restart_chk
                if self.atomic_force:
                    forces = self._compute_force(restart_chk, work_dir=_wd)
                    if forces is not None:
                        self.output_values["forces"] = forces
                set_estimation(
                    _wd,
                    last_energy=energy,
                    last_energy_error=error,
                    last_num_mcmc_bin_blocks=self.num_mcmc_bin_blocks,
                    last_num_mcmc_warmup_steps=self.num_mcmc_warmup_steps,
                )

        # ── Collect outputs ───────────────────────────────────────
        chk_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.h5")))
        output_logs = [
            suffixed_name(self.output_file, j)
            for j in range(last_run + 1)
            if os.path.isfile(os.path.join(_wd, suffixed_name(self.output_file, j)))
        ]
        self.output_files = chk_files + output_logs
        self.output_values["estimated_steps"] = estimated_steps

        self.status = "success"
        return self.status, self.output_files, self.output_values

    # ── Utility methods ───────────────────────────────────────────

    def _find_restart_chk(self, work_dir: str) -> Optional[str]:
        """Locate the MCMC restart checkpoint file in *work_dir*."""
        for pattern in ["restart.h5", "mcmc.h5", "*.h5"]:
            matches = sorted(glob.glob(os.path.join(work_dir, pattern)))
            if matches:
                return os.path.basename(matches[-1])
        return None

    def _compute_energy(self, restart_chk: str, work_dir: str):
        """Run ``jqmc-tool mcmc compute-energy`` and parse output.

        Parameters
        ----------
        restart_chk : str
            Checkpoint filename (basename).
        work_dir : str
            Directory in which to run the command.

        Returns
        -------
        tuple
            ``(energy, error)`` or ``(None, None)``.
        """
        cmd = f"jqmc-tool mcmc compute-energy {restart_chk} -b {self.num_mcmc_bin_blocks} -w {self.num_mcmc_warmup_steps}"
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

    def _compute_force(self, restart_chk: str, work_dir: str):
        """Run ``jqmc-tool mcmc compute-force`` and parse output.

        Parameters
        ----------
        restart_chk : str
            Checkpoint filename (basename).
        work_dir : str
            Directory in which to run the command.

        Returns
        -------
        list of dict or None
            Each dict has keys ``label``, ``Fx``, ``Fx_err``,
            ``Fy``, ``Fy_err``, ``Fz``, ``Fz_err``.
            Returns *None* on failure.
        """
        cmd = f"jqmc-tool mcmc compute-force {restart_chk} -b {self.num_mcmc_bin_blocks} -w {self.num_mcmc_warmup_steps}"
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
