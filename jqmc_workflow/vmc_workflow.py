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
)
from ._input_generator import generate_input_toml, resolve_with_defaults
from ._job import get_num_mpi, load_queue_data
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


class VMC_Workflow(Workflow):
    """VMC (Variational Monte Carlo) Jastrow / orbital optimisation workflow.

    Generates a ``job_type=vmc`` input TOML, submits ``jqmc``, monitors
    until completion, and collects the optimised
    ``hamiltonian_data_opt_step_N.h5`` files and checkpoint.

    The workflow supports two modes:

    **Automatic mode** (default, ``num_mcmc_steps=None``):

    1. **Pilot VMC run** (``_0``) — Runs a short optimisation with
       ``pilot_vmc_steps`` optimisation steps and ``pilot_mcmc_steps``
       MCMC steps per step.  The statistical error of the *last*
       optimisation step is used to estimate the MCMC steps per
       opt-step required to achieve ``target_error`` via
       $\sigma \propto 1/\sqrt{N}$.
    2. **Production VMC runs** (``_1``, ``_2``, …) — Full optimisation
       with ``num_opt_steps`` and the estimated MCMC steps per step.
       If a run is interrupted by the wall-time limit, the next
       continuation restarts from the checkpoint.  At most
       ``max_continuation`` runs are attempted.

    **Fixed-step mode** (``num_mcmc_steps`` is set):

    The pilot run is skipped entirely and ``target_error`` is ignored.
    Each production run uses exactly ``num_mcmc_steps`` MCMC steps per
    optimisation step, and ``max_continuation`` runs are executed
    unconditionally.

    Parameters
    ----------
    server_machine_name : str
        Name of the target machine (must be configured in ``~/.jqmc_setting/``).
    num_opt_steps : int
        Number of optimization iterations for production runs.
    hamiltonian_file : str
        Input ``hamiltonian_data.h5``.
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
    opt_J3_basis_exp : bool, optional
        Optimize J3 AO Gaussian exponents.  Default from ``jqmc_miscs``.
    opt_J3_basis_coeff : bool, optional
        Optimize J3 AO contraction coefficients.  Default from ``jqmc_miscs``.
    opt_lambda_basis_exp : bool, optional
        Optimize Geminal AO Gaussian exponents.  Default from ``jqmc_miscs``.
    opt_lambda_basis_coeff : bool, optional
        Optimize Geminal AO contraction coefficients.  Default from ``jqmc_miscs``.
    optimizer_kwargs : dict, optional
        Optimizer configuration dict.  Default from ``jqmc_miscs``.
    mcmc_seed : int, optional
        Random seed for MCMC.  Default from ``jqmc_miscs``.
    verbosity : str, optional
        Verbosity level.  Default from ``jqmc_miscs``.
    poll_interval : int
        Seconds between job-status polls.
    target_error : float
        Target statistical error (Ha) per optimization step.  Ignored
        when *num_mcmc_steps* is set.
    num_mcmc_steps : int, optional
        Fixed number of MCMC measurement steps per optimisation step.
        When set, the pilot run is skipped and ``target_error`` is
        ignored; each of the ``max_continuation`` production runs uses
        exactly this many MCMC steps per opt step.
    pilot_mcmc_steps : int
        MCMC steps per opt-step for the pilot run.  Ignored when
        *num_mcmc_steps* is set.
    pilot_vmc_steps : int
        Number of optimization steps in the pilot run (small; just
        enough to estimate the error bar).
    pilot_queue_label : str, optional
        Queue label for the pilot run.  Defaults to *queue_label*.
        Use a shorter/smaller queue for the pilot to save resources.
    max_continuation : int
        Maximum number of production runs after the pilot.
    target_snr : float
        Target signal-to-noise ratio ``max(|f|/|std f|)`` for force
        convergence.  The workflow continues until the averaged
        S/N drops to or below this threshold.
    snr_avg_window : int
        Number of trailing optimization steps over which to average
        the signal-to-noise ratio for the convergence check.
        When there are fewer S/N values than this window, all
        available values are averaged.
    cleanup_patterns : list[str], optional
        Glob patterns for files to delete after successful completion
        (e.g. ``["restart.h5", "hamiltonian_opt*.h5"]``).  Local files
        are always removed; remote files are removed only when the
        workflow targets a remote machine.  Default *None* (no cleanup).

    Examples
    --------
    Standalone launch (automatic mode)::

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

    Fixed-step mode (no pilot, no target_error check)::

        wf = VMC_Workflow(
            server_machine_name="cluster",
            num_opt_steps=20,
            num_mcmc_steps=500,
            number_of_walkers=8,
            max_continuation=3,
        )
        status, files, values = wf.launch()

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

    Output Values
    -------------
    After ``launch()`` completes, ``output_values`` may contain:

    optimized_hamiltonian : str
        Basename of the last optimised Hamiltonian file
        (e.g. ``"hamiltonian_data_opt_step_91.h5"``).
        Use with ``ValueFrom("vmc", "optimized_hamiltonian")``
        to pass the filename dynamically to downstream workflows.
    checkpoint : str
        Basename of the restart checkpoint file.
    num_mcmc_steps : int
        Estimated MCMC steps per optimisation step
        (automatic mode).  In fixed-step mode this key is
        ``estimated_mcmc_steps`` instead.
    energy : float
        Energy from the last optimisation step (Ha).
    energy_error : float
        Statistical error on ``energy`` (Ha).
    signal_to_noise : float
        Average signal-to-noise ratio over the trailing window
        (only when force-convergence is enabled).
    signal_to_noise_last : float
        Signal-to-noise ratio of the last optimisation step.
    energy_slope : float
        Slope of energy vs. step from the trailing window
        (only when ``energy_slope_sigma_threshold`` is set).
    energy_slope_std : float
        Standard deviation of the energy slope.

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
    LRDMC_Workflow : Diffusion Monte Carlo (job_type=lrdmc-bra / lrdmc-tau).
    WF_Workflow : TREXIO → hamiltonian_data conversion.
    """

    def __init__(
        self,
        server_machine_name: str = "localhost",
        num_opt_steps: int = 20,
        hamiltonian_file: str = "hamiltonian_data.h5",
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
        opt_J3_basis_exp: Optional[bool] = None,
        opt_J3_basis_coeff: Optional[bool] = None,
        opt_lambda_basis_exp: Optional[bool] = None,
        opt_lambda_basis_coeff: Optional[bool] = None,
        optimizer_kwargs: Optional[dict] = None,
        # -- [control] section parameters --
        mcmc_seed: Optional[int] = None,
        verbosity: Optional[str] = None,
        # -- workflow parameters --
        poll_interval: int = 60,
        target_error: float = 0.001,
        num_mcmc_steps: Optional[int] = None,
        pilot_mcmc_steps: int = 50,
        pilot_vmc_steps: int = 5,
        pilot_queue_label: Optional[str] = None,
        max_continuation: int = 1,
        target_snr: Optional[float] = None,
        snr_avg_window: int = 5,
        energy_slope_sigma_threshold: Optional[float] = None,
        energy_slope_window_size: int = 5,
        cleanup_patterns: Optional[list] = None,
    ):
        super().__init__(cleanup_patterns=cleanup_patterns)
        self.server_machine_name = server_machine_name
        self.num_opt_steps = num_opt_steps
        self.hamiltonian_file = hamiltonian_file
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
        self.opt_J3_basis_exp = opt_J3_basis_exp
        self.opt_J3_basis_coeff = opt_J3_basis_coeff
        self.opt_lambda_basis_exp = opt_lambda_basis_exp
        self.opt_lambda_basis_coeff = opt_lambda_basis_coeff
        self.optimizer_kwargs = optimizer_kwargs
        # [control] section
        self.mcmc_seed = mcmc_seed
        self.verbosity = verbosity
        # workflow
        self.poll_interval = poll_interval
        self.target_error = target_error
        self.num_mcmc_steps = num_mcmc_steps
        self.pilot_mcmc_steps = pilot_mcmc_steps
        self.pilot_vmc_steps = pilot_vmc_steps
        self.pilot_queue_label = pilot_queue_label or queue_label
        self.max_continuation = max_continuation
        self.target_snr = target_snr
        self.snr_avg_window = snr_avg_window
        self.energy_slope_sigma_threshold = energy_slope_sigma_threshold
        self.energy_slope_window_size = energy_slope_window_size

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
                "opt_J3_basis_exp": self.opt_J3_basis_exp,
                "opt_J3_basis_coeff": self.opt_J3_basis_coeff,
                "opt_lambda_basis_exp": self.opt_lambda_basis_exp,
                "opt_lambda_basis_coeff": self.opt_lambda_basis_coeff,
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
    # _submit_and_wait() and _make_job() are inherited from Workflow.

    # ── configure / run ──────────────────────────────────────────

    def configure(self) -> dict:
        """Validate parameters and return configuration summary."""
        mode = "fixed" if self.num_mcmc_steps is not None else "auto"
        return {
            "jobname": self.jobname,
            "num_opt_steps": self.num_opt_steps,
            "num_mcmc_steps": self.num_mcmc_steps,
            "target_error": self.target_error,
            "mode": mode,
            "hamiltonian_file": self.hamiltonian_file,
            "server_machine": self.server_machine_name,
            "number_of_walkers": self.number_of_walkers,
            "max_time": self.max_time,
            "target_snr": self.target_snr,
            "pilot_mcmc_steps": self.pilot_mcmc_steps,
            "pilot_vmc_steps": self.pilot_vmc_steps,
            "max_continuation": self.max_continuation,
        }

    async def run(self) -> tuple:
        """Run the VMC optimization workflow.

        **Fixed-step mode** (``num_mcmc_steps`` is set):
        The pilot run is skipped.  Each production run uses exactly
        ``num_mcmc_steps`` MCMC steps per opt step and all
        ``max_continuation`` runs are executed unconditionally.

        **Automatic mode** (``num_mcmc_steps`` is *None*, default):

        1. Pilot VMC run in ``_pilot/`` with ``pilot_vmc_steps`` opt steps
           and ``pilot_mcmc_steps`` MCMC steps to estimate the required
           MCMC steps per opt step (skipped on continuation).
           May use a different queue (``pilot_queue_label``).
        2. Production VMC runs (``_1``, ``_2``, ...) start from scratch
           with the full ``num_opt_steps`` and estimated MCMC steps until
           all optimization steps complete or ``max_continuation`` is
           reached.
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
        estimated_mcmc_steps = self.num_mcmc_steps
        logger.info("")
        logger.info("-- VMC Fixed-step mode " + "-" * 27)
        logger.info(
            f"  num_mcmc_steps    = {estimated_mcmc_steps}\n"
            f"  num_opt_steps     = {self.num_opt_steps}\n"
            f"  max_continuation  = {self.max_continuation}"
        )

        last_run = 0
        step_files = {}  # {step: (input, output, run_id)}
        for i in range(1, self.max_continuation + 1):
            recorded = get_job_by_step(_wd, i)
            status = recorded.get("status")
            if status == "fetched":
                logger.info(f"  step {i}: already fetched. Skipping.")
                step_files[i] = (recorded["input_file"], recorded["output_file"], recorded.get("run_id", ""))
                last_run = i
                continue
            elif status in ("submitted", "completed"):
                input_i = recorded["input_file"]
                output_i = recorded["output_file"]
                run_id_i = recorded.get("run_id", "")
                logger.info(f"  step {i}: already {status}. Resuming...")
            else:
                run_id_i = self._new_run_id()
                input_i = self._input_filename(i, run_id_i)
                output_i = self._output_filename(i, run_id_i)
                if i == 1:
                    self._generate_input(
                        estimated_mcmc_steps,
                        self.num_opt_steps,
                        os.path.join(_wd, input_i),
                    )
                else:
                    restart_chk = self._find_restart_chk(_wd)
                    if restart_chk is None:
                        raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {_wd}")
                    self._generate_input(
                        estimated_mcmc_steps,
                        self.num_opt_steps,
                        os.path.join(_wd, input_i),
                        restart=True,
                        restart_chk=restart_chk,
                    )
                logger.info("")
                logger.info(f"-- VMC Production run {i}/{self.max_continuation} " + "-" * 10)
                logger.info(f"  {input_i}: {self.num_opt_steps} opt steps x {estimated_mcmc_steps} MCMC steps/step")

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
            step_files[i] = (input_i, output_i, run_id_i)
            last_run = i

            logger.info(f"  VMC production run {i}/{self.max_continuation} completed.")

            # ── Abnormal-termination guard (single source of truth) ──
            # target_error=None → only Program-ends / non-finite-energy
            # checks are active. VMC's SNR/slope convergence is decided
            # separately at end-of-workflow.
            vstatus, vmsg = validate_completion(_wd, self.output_values)
            if vstatus == CompletionStatus.FAILED:
                logger.error(vmsg)
                self.output_values["error"] = vmsg
                self.status = WorkflowStatus.FAILED
                break

        # ── Collect outputs ───────────────────────────────────────
        h5_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.h5")))
        output_logs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.out")))
        self.output_files = h5_files + output_logs

        def _h5_step_num(path: str) -> int:
            m = re.search(r"hamiltonian_data_opt_step_(\d+)\.h5$", path)
            return int(m.group(1)) if m else -1

        opt_files = sorted(glob.glob(os.path.join(_wd, "hamiltonian_data_opt_step_*.h5")), key=_h5_step_num)
        if opt_files:
            self.output_values["optimized_hamiltonian"] = os.path.basename(opt_files[-1])
        restart_chk = self._find_restart_chk(_wd)
        if restart_chk:
            self.output_values["checkpoint"] = restart_chk
        self.output_values["num_mcmc_steps"] = estimated_mcmc_steps

        # Parse last production output for energy
        if last_run in step_files:
            last_output = os.path.join(_wd, step_files[last_run][1])
            self._parse_output(last_output)

        if self.status != WorkflowStatus.FAILED:
            self.status = WorkflowStatus.COMPLETED
        return self.status, self.output_files, self.output_values

    async def _launch_auto(self, _wd):
        """Automatic mode: pilot + target_error convergence."""
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

            recorded_0 = get_job_by_step(pilot_dir, 0)
            status_0 = recorded_0.get("status")
            if status_0 in ("fetched", "submitted", "completed"):
                input_0 = recorded_0["input_file"]
                output_0 = recorded_0["output_file"]
                run_id_0 = recorded_0.get("run_id", "")
                if status_0 == "fetched":
                    logger.info(f"  {input_0}: already fetched. Skipping pilot.")
                else:
                    logger.info(f"  {input_0}: already {status_0}. Resuming...")
            else:
                run_id_0 = self._new_run_id()
                input_0 = self._input_filename(0, run_id_0)
                output_0 = self._output_filename(0, run_id_0)
                self._generate_input(
                    self.pilot_mcmc_steps,
                    self.pilot_vmc_steps,
                    os.path.join(pilot_dir, input_0),
                )
            logger.info("")
            logger.info("-- VMC Phase 0: Pilot " + "-" * 28)
            logger.info(
                f"  {input_0}: {self.pilot_vmc_steps} opt steps x {self.pilot_mcmc_steps} MCMC steps/step (queue: {self.pilot_queue_label})"
            )

            pilot_t0 = time.monotonic()
            await self._submit_and_wait(
                input_0, output_0, work_dir=pilot_dir, queue_label=self.pilot_queue_label, step=0, run_id=run_id_0
            )
            pilot_wall_sec = time.monotonic() - pilot_t0

            # Parse the last optimization step's error from pilot output
            pilot_output_path = os.path.join(pilot_dir, output_0)
            pilot_energy, pilot_error = self._parse_last_opt_energy(pilot_output_path)
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
                min_steps=self.num_mcmc_bin_blocks or 0,
            )
            # Add warmup back: production also discards warmup steps
            estimated_mcmc_steps += warmup

            total_pilot_cost = self.pilot_vmc_steps * self.pilot_mcmc_steps
            total_prod_cost = self.num_opt_steps * estimated_mcmc_steps
            cost_ratio = total_prod_cost / total_pilot_cost if total_pilot_cost > 0 else 0

            # Time estimate: only Net time scales with step count
            net_pilot_sec = parse_net_time(pilot_output_path)
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
            set_estimation(
                _wd,
                pilot_mcmc_steps=self.pilot_mcmc_steps,
                pilot_vmc_steps=self.pilot_vmc_steps,
                pilot_error=pilot_error,
                target_error=self.target_error,
                estimated_mcmc_steps=estimated_mcmc_steps,
                pilot_queue_label=self.pilot_queue_label,
                walker_ratio=walker_ratio,
                pilot_wall_sec=pilot_wall_sec,
                net_pilot_sec=net_pilot_sec or 0,
            )

        # ── Production runs (phase 1..N) ──────────────────────────
        _has_convergence_criteria = self.target_snr is not None or self.energy_slope_sigma_threshold is not None
        last_run = 0
        step_files = {}  # {step: (input, output, run_id)}
        _checked_fetched_convergence = False
        converged = converged_snr = converged_slope = None
        for i in range(1, self.max_continuation + 1):
            # Skip input generation if this step already has a job record
            recorded = get_job_by_step(_wd, i)
            status = recorded.get("status")
            if status == "fetched":
                logger.info(f"  step {i}: already fetched. Skipping.")
                step_files[i] = (recorded["input_file"], recorded["output_file"], recorded.get("run_id", ""))
                last_run = i
                continue
            elif status in ("submitted", "completed"):
                input_i = recorded["input_file"]
                output_i = recorded["output_file"]
                run_id_i = recorded.get("run_id", "")
                logger.info(f"  step {i}: already {status}. Resuming...")
            else:
                # ── Re-evaluate convergence from fetched runs ─────
                if _has_convergence_criteria and last_run > 0 and not _checked_fetched_convergence:
                    _checked_fetched_convergence = True
                    converged, converged_snr, converged_slope = self._check_convergence(
                        _wd,
                        step_files,
                        last_run,
                    )
                    if converged:
                        logger.info(f"  Convergence already met after fetched runs (step {last_run}). No further runs needed.")
                        break

                run_id_i = self._new_run_id()
                input_i = self._input_filename(i, run_id_i)
                output_i = self._output_filename(i, run_id_i)
                if i == 1:
                    self._generate_input(
                        estimated_mcmc_steps,
                        self.num_opt_steps,
                        os.path.join(_wd, input_i),
                    )
                else:
                    restart_chk = self._find_restart_chk(_wd)
                    if restart_chk is None:
                        raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {_wd}")
                    self._generate_input(
                        estimated_mcmc_steps,
                        self.num_opt_steps,
                        os.path.join(_wd, input_i),
                        restart=bool(restart_chk),
                        restart_chk=restart_chk,
                    )
                logger.info("")
                logger.info(f"-- VMC Phase 1: Production run {i}/{self.max_continuation} " + "-" * 10)
                logger.info(f"  {input_i}: {self.num_opt_steps} opt steps x {estimated_mcmc_steps} MCMC steps/step")

            restart_chk = self._find_restart_chk(_wd) if i > 1 else None
            if i > 1 and restart_chk is None:
                raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {_wd}")
            extra_from = [restart_chk] if restart_chk else []

            # Estimate run time from the latest available output
            _prod_cost = self.num_opt_steps * estimated_mcmc_steps
            _ref_net = None
            for _j in range(i - 1, 0, -1):
                if _j not in step_files:
                    continue
                _ref_net = parse_net_time(os.path.join(_wd, step_files[_j][1]))
                if _ref_net and _ref_net > 0:
                    # All production runs have the same step count
                    logger.info(f"  est. Net run time (w/o JAX compilation) = {_format_duration(_ref_net)}")
                    break
            else:
                # First production run: use pilot output
                _pilot_outs = sorted(glob.glob(os.path.join(_wd, "_pilot", "*.out")))
                _ref_net = parse_net_time(_pilot_outs[-1]) if _pilot_outs else None
                if _ref_net and _ref_net > 0:
                    _p_vmc = estimation.get("pilot_vmc_steps") or self.pilot_vmc_steps
                    _p_mcmc = estimation.get("pilot_mcmc_steps") or self.pilot_mcmc_steps
                    _pilot_cost = _p_vmc * _p_mcmc
                    if _pilot_cost > 0:
                        logger.info(
                            f"  est. Net run time (w/o JAX compilation) = {_format_duration(_ref_net * _prod_cost / _pilot_cost)}"
                        )

            await self._submit_and_wait(input_i, output_i, work_dir=_wd, extra_from_objects=extra_from, step=i, run_id=run_id_i)
            step_files[i] = (input_i, output_i, run_id_i)
            last_run = i
            converged = converged_snr = converged_slope = None

            logger.info(f"  VMC production run {i}/{self.max_continuation} completed.")

            # ── Abnormal-termination guard (single source of truth) ──
            # target_error=None → only Program-ends / non-finite-energy
            # checks; SNR/slope convergence is evaluated separately below.
            vstatus, vmsg = validate_completion(_wd, self.output_values)
            if vstatus == CompletionStatus.FAILED:
                logger.error(vmsg)
                self.output_values["error"] = vmsg
                self.status = WorkflowStatus.FAILED
                break

            # ── Early exit if convergence criteria met ────────────
            if _has_convergence_criteria and i < self.max_continuation:
                converged, converged_snr, converged_slope = self._check_convergence(
                    _wd,
                    step_files,
                    last_run,
                )
                if converged:
                    logger.info(f"  Convergence achieved at run {i}/{self.max_continuation}. Stopping early.")
                    break

        # ── Collect outputs ───────────────────────────────────────
        h5_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.h5")))
        output_logs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.out")))
        self.output_files = h5_files + output_logs

        def _h5_step_num(path: str) -> int:
            m = re.search(r"hamiltonian_data_opt_step_(\d+)\.h5$", path)
            return int(m.group(1)) if m else -1

        opt_files = sorted(glob.glob(os.path.join(_wd, "hamiltonian_data_opt_step_*.h5")), key=_h5_step_num)
        if opt_files:
            self.output_values["optimized_hamiltonian"] = os.path.basename(opt_files[-1])
        restart_chk = self._find_restart_chk(_wd)
        if restart_chk:
            self.output_values["checkpoint"] = restart_chk
        self.output_values["estimated_mcmc_steps"] = estimated_mcmc_steps

        # Parse last production output for energy
        if last_run in step_files:
            last_output = os.path.join(_wd, step_files[last_run][1])
            self._parse_output(last_output)

        # ── Final convergence check ───────────────────────────────
        if converged is None:
            converged, converged_snr, converged_slope = self._check_convergence(
                _wd,
                step_files,
                last_run,
            )
        if not _has_convergence_criteria:
            logger.info("  No convergence criteria set; treating as converged.")
        elif converged:
            logger.info("  VMC converged (all active checks passed).")
        else:
            reasons = []
            if not converged_snr:
                reasons.append(f"S/N ({self.output_values.get('signal_to_noise', '?'):.3f}) > target ({self.target_snr})")
            if not converged_slope:
                reasons.append(
                    f"energy slope ({self.output_values.get('energy_slope', '?'):.6f} Ha/step) still significantly negative"
                )
            msg = f"VMC NOT converged after {self.max_continuation} continuation run(s): {'; '.join(reasons)}"
            logger.error(f"  {msg}")
            self.status = WorkflowStatus.FAILED

        if self.status != WorkflowStatus.FAILED:
            self.status = WorkflowStatus.COMPLETED
        return self.status, self.output_files, self.output_values

    # ── Utility methods ───────────────────────────────────────────

    def _check_convergence(
        self,
        work_dir: str,
        step_files: dict,
        last_run: int,
    ) -> tuple:
        """Evaluate SNR and energy-slope convergence criteria.

        Returns ``(converged, converged_snr, converged_slope)`` where
        each flag is *True* when the criterion is met or not active.
        """
        converged_snr = True
        converged_slope = True

        # ── (A) SNR check ──
        if self.target_snr is not None:
            all_snr = []
            for j in range(last_run, 0, -1):
                if j not in step_files:
                    continue
                out_j = os.path.join(work_dir, step_files[j][1])
                all_snr = self._parse_all_snr(out_j) + all_snr
                if len(all_snr) >= self.snr_avg_window:
                    break
            if all_snr:
                window = all_snr[-self.snr_avg_window :]
                avg_snr = sum(window) / len(window)
                self.output_values["signal_to_noise"] = avg_snr
                self.output_values["signal_to_noise_last"] = all_snr[-1]
                logger.info(f"  S/N avg over last {len(window)} step(s): {avg_snr:.3f}  (last step: {all_snr[-1]:.3f})")
                converged_snr = avg_snr <= self.target_snr
            else:
                converged_snr = False
                logger.warning("  Could not parse S/N from production output.")

        # ── (B) Energy slope check ──
        if self.energy_slope_sigma_threshold is not None:
            all_energies: list[tuple[float, float]] = []
            for j in range(last_run, 0, -1):
                if j not in step_files:
                    continue
                out_j = os.path.join(work_dir, step_files[j][1])
                all_energies = self._parse_all_energies(out_j) + all_energies
                if len(all_energies) >= self.energy_slope_window_size:
                    break

            window_e = all_energies[-self.energy_slope_window_size :]
            if len(window_e) >= 2:
                energies = [e for e, _ in window_e]
                errors = [s for _, s in window_e]
                slope, slope_std = self._fit_energy_slope(energies, errors)
                self.output_values["energy_slope"] = slope
                self.output_values["energy_slope_std"] = slope_std
                logger.info(f"  Energy slope over last {len(window_e)} step(s): {slope:.6f} \u00b1 {slope_std:.6f} Ha/step")
                converged_slope = slope >= -slope_std * self.energy_slope_sigma_threshold
                if converged_slope:
                    logger.info(
                        f"  Energy plateau: slope ({slope:.6f} Ha/step) >= "
                        f"-{slope_std:.6f} * {self.energy_slope_sigma_threshold}"
                    )
                else:
                    logger.info(
                        f"  Energy still decreasing: slope ({slope:.6f} Ha/step) < "
                        f"-{slope_std:.6f} * {self.energy_slope_sigma_threshold}"
                    )
            else:
                converged_slope = False
                logger.warning(f"  Not enough energy data for slope check ({len(window_e)} < 2 steps).")

        converged = converged_snr and converged_slope
        return converged, converged_snr, converged_slope

    def _find_restart_chk(self, work_dir: str) -> Optional[str]:
        """Locate a VMC restart checkpoint file in *work_dir*."""
        for pattern in ["restart.h5", "vmc.h5", "*.h5"]:
            matches = sorted(glob.glob(os.path.join(work_dir, pattern)))
            if matches:
                return os.path.basename(matches[-1])
        return None

    # ── Output parsing ────────────────────────────────────────────

    def _parse_output(self, output_file=None):
        """Extract the last optimization energy from *output_file*."""
        if output_file is None:
            return
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
    def _parse_all_snr(output_file):
        """Parse all signal-to-noise ratios from a VMC output file.

        Returns
        -------
        list[float]
            All ``max(|f|/|std f|)`` values in order, one per
            optimization step.  Empty list if the file is missing
            or contains no S/N lines.
        """
        if not os.path.isfile(output_file):
            return []

        snr_pattern = re.compile(r"Max of signal-to-noise of f = max\(\|f\|/\|std f\|\) = ([-+]?\d+(?:\.\d+)?)")
        values = []
        try:
            with open(output_file, "r") as f:
                for line in f:
                    m = snr_pattern.search(line)
                    if m:
                        values.append(float(m.group(1)))
        except Exception:
            return []
        return values

    @staticmethod
    def _parse_all_energies(output_file: str) -> list[tuple[float, float]]:
        """Extract per-step ``(energy, energy_error)`` from a VMC output file.

        Uses the existing ``_parse_vmc_log_text()`` parser to obtain
        :class:`VMC_Step_Data` and returns the energy/error pairs.

        Returns
        -------
        list[tuple[float, float]]
            ``[(E_1, σ_1), (E_2, σ_2), ...]`` in file order.
            Empty list if the file is missing or unparseable.
        """
        if not os.path.isfile(output_file):
            return []
        try:
            with open(output_file, "r") as f:
                text = f.read()
            from ._output_parser import _parse_vmc_log_text

            steps = _parse_vmc_log_text(text)
            return [(s.energy, s.energy_error) for s in steps if s.energy is not None and s.energy_error is not None]
        except Exception:
            return []

    @staticmethod
    def _fit_energy_slope(
        energies: list[float],
        energy_errors: list[float],
    ) -> tuple[float, float]:
        """Weighted linear regression of energy vs optimisation step.

        Model: ``E_k = a + b * k + ε_k``, weight ``w_k = 1 / σ_k²``.

        Parameters
        ----------
        energies : list[float]
            Energy value per optimisation step (length *N* >= 2).
        energy_errors : list[float]
            Statistical error per step (length *N*, positive).

        Returns
        -------
        slope : float
            Weighted least-squares slope *b*.
        slope_std : float
            Standard error of *b*.
        """
        import numpy as np

        E = np.asarray(energies, dtype=float)
        sigma = np.asarray(energy_errors, dtype=float)
        w = 1.0 / sigma**2
        k = np.arange(len(E), dtype=float)

        S = np.sum(w)
        Sk = np.sum(w * k)
        Skk = np.sum(w * k**2)
        SE = np.sum(w * E)
        SkE = np.sum(w * k * E)

        delta = S * Skk - Sk**2
        b = (S * SkE - Sk * SE) / delta
        sigma_b = np.sqrt(S / delta)
        return float(b), float(sigma_b)

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
