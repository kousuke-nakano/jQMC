# -*- coding: utf-8 -*-
"""LRDMC_Ext_Workflow — LRDMC extrapolation to the a²→0 limit.

Orchestrates multiple :class:`LRDMC_Workflow` runs at different lattice
spacings (``alat`` values), then post-processes with
``jqmc-tool lrdmc extrapolate-energy`` to obtain the continuum-limit energy.

This is a *composite* workflow: it spawns one :class:`Container`
per ``alat`` value, runs them (potentially in parallel via the Launcher),
and finally performs the extrapolation.
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
import re
import subprocess
from logging import getLogger
from typing import List, Optional

from ._setting import (
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_WARMUP_STEPS,
)
from .lrdmc_workflow import LRDMC_Workflow
from ._state import WorkflowStatus
from .workflow import Container, Workflow

logger = getLogger("jqmc-workflow").getChild(__name__)


class LRDMC_Ext_Workflow(Workflow):
    """LRDMC a²→0 continuum-limit extrapolation workflow.

    Orchestrates multiple :class:`LRDMC_Workflow` runs at different
    lattice spacings (``alat`` values), then post-processes with
    ``jqmc-tool lrdmc extrapolate-energy`` to obtain the
    continuum-limit energy.

    Each ``alat`` run is wrapped in its own :class:`Container`
    and all alat values are executed in parallel.  Every ``alat``
    independently calibrates its own ``num_projection_per_measurement``
    (when ``target_survived_walkers_ratio`` is set in GFMC_n mode),
    runs an error-bar pilot, and then runs production.

    **Mode selection** follows the same rules as
    :class:`LRDMC_Workflow`:

    * **GFMC_t** (default) — set *time_projection_tau* (default 0.10).
    * **GFMC_n** — set *target_survived_walkers_ratio* or
      *num_projection_per_measurement*.

    Parameters
    ----------
    server_machine_name : str
        Target machine name (shared by all sub-runs).
    alat_list : list[float]
        List of lattice discretization values, e.g. ``[0.5, 0.4, 0.3]``.
    hamiltonian_file : str
        Input ``hamiltonian_data.h5`` (must exist in the parent directory
        or be resolved by ``FileFrom``).
    queue_label : str
        Queue/partition label for production runs.
    pilot_queue_label : str, optional
        Queue/partition label for pilot runs.  Defaults to
        ``queue_label`` when *None*.  A shorter queue is often
        sufficient for the pilot.
    jobname_prefix : str
        Prefix for each sub-run job name.
    number_of_walkers : int
        Walkers per MPI process.
    max_time : int
        Wall-time limit per sub-run (seconds).
    polynomial_order : int
        Polynomial order for the a²→0 extrapolation (default: 2).
    num_gfmc_bin_blocks : int
        Binning blocks for post-processing.
    num_gfmc_warmup_steps : int
        Warmup steps to discard.
    num_gfmc_collect_steps : int
        Weight-collection steps.
    time_projection_tau : float, optional
        Imaginary time step for GFMC_t mode (default 0.10).  Ignored
        when *target_survived_walkers_ratio* or
        *num_projection_per_measurement* is set.
    target_survived_walkers_ratio : float, optional
        Target survived-walkers ratio (default *None*).  Each ``alat``
        independently runs a calibration pilot (``_pilot_a``) to
        find its own optimal ``num_projection_per_measurement``.
        Set to *None* to disable auto-calibration (requires explicit
        *num_projection_per_measurement*).  Activates GFMC_n mode.
    num_projection_per_measurement : int, optional
        GFMC projections per measurement.  When given explicitly,
        automatic calibration is disabled and this value is used
        for every ``alat``.  Activates GFMC_n mode.
    non_local_move : str, optional
        Non-local move treatment.  Default from ``jqmc_miscs``.
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
        Target statistical error (Ha) for each sub-LRDMC run.
        Passed through to each :class:`LRDMC_Workflow`.
    pilot_steps : int
        Pilot measurement steps for target-error estimation.
    num_gfmc_projections : int, optional
        Fixed number of measurement steps per production run.
        When set, the error-bar pilot is skipped for each sub-LRDMC
        and all ``max_continuation`` runs are executed unconditionally.
        Passed through to each :class:`LRDMC_Workflow`.
        Default *None* (automatic mode).
    max_continuation : int
        Maximum number of production runs per sub-LRDMC.

    Examples
    --------
    GFMC_t mode (default)::

        wf = LRDMC_Ext_Workflow(
            server_machine_name="cluster",
            alat_list=[0.5, 0.4, 0.3],
            target_error=0.001,
            number_of_walkers=8,
        )
        status, files, values = wf.launch()
        print(values["extrapolated_energy"],
              values["extrapolated_energy_error"])

    GFMC_n mode with calibration::

        wf = LRDMC_Ext_Workflow(
            server_machine_name="cluster",
            alat_list=[0.5, 0.4, 0.3],
            target_survived_walkers_ratio=0.97,
            target_error=0.001,
            number_of_walkers=8,
        )

    As part of a :class:`Launcher` pipeline::

        enc = Container(
            label="lrdmc-ext",
            dirname="03_lrdmc",
            input_files=[FileFrom("mcmc-run", "hamiltonian_data.h5")],
            workflow=LRDMC_Ext_Workflow(
                server_machine_name="cluster",
                alat_list=[0.5, 0.4, 0.3],
                target_error=0.001,
            ),
        )

    Notes
    -----
    * At least two ``alat`` values are required for extrapolation.
      With a single value, per-alat results are returned but no
      extrapolation is performed.
    * Each sub-run directory is named ``lrdmc_alat_<value>/``.

    See Also
    --------
    LRDMC_Workflow : Single-alat LRDMC run.
    """

    def __init__(
        self,
        server_machine_name: str = "localhost",
        alat_list: Optional[List[float]] = None,
        hamiltonian_file: str = "hamiltonian_data.h5",
        queue_label: str = "default",
        pilot_queue_label: Optional[str] = None,
        jobname_prefix: str = "jqmc-lrdmc",
        number_of_walkers: int = 4,
        max_time: int = 86400,
        polynomial_order: int = 2,
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
        epsilon_PW: Optional[float] = None,
        # -- [control] section parameters --
        mcmc_seed: Optional[int] = None,
        verbosity: Optional[str] = None,
        # -- workflow parameters --
        poll_interval: int = 60,
        target_error: float = 0.001,
        pilot_steps: int = 100,
        num_gfmc_projections: Optional[int] = None,
        max_continuation: int = 5,
    ):
        super().__init__()
        self.server_machine_name = server_machine_name
        self.alat_list = alat_list or [0.5, 0.4, 0.3]
        self.hamiltonian_file = hamiltonian_file
        self.queue_label = queue_label
        self.pilot_queue_label = pilot_queue_label or queue_label
        self.jobname_prefix = jobname_prefix
        self.number_of_walkers = number_of_walkers
        self.max_time = max_time
        self.polynomial_order = polynomial_order
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
        # [lrdmc-bra / lrdmc-tau] section
        self.time_projection_tau = time_projection_tau
        self.target_survived_walkers_ratio = target_survived_walkers_ratio
        self.num_projection_per_measurement = num_projection_per_measurement
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
        self.num_gfmc_projections = num_gfmc_projections
        self.max_continuation = max_continuation

    def _make_lrdmc_workflow(self, alat):
        """Create one :class:`Container` for a given *alat* value.

        Parameters
        ----------
        alat : float
            Lattice spacing.

        Returns
        -------
        Container
        """
        label = f"lrdmc-a{alat:.3f}"
        dirname = f"lrdmc_alat_{alat:.3f}"

        wf = LRDMC_Workflow(
            server_machine_name=self.server_machine_name,
            alat=alat,
            hamiltonian_file=self.hamiltonian_file,
            queue_label=self.queue_label,
            pilot_queue_label=self.pilot_queue_label,
            jobname=f"{self.jobname_prefix}-a{alat:.2f}",
            number_of_walkers=self.number_of_walkers,
            max_time=self.max_time,
            num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
            num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
            num_gfmc_collect_steps=self.num_gfmc_collect_steps,
            time_projection_tau=self.time_projection_tau,
            target_survived_walkers_ratio=self.target_survived_walkers_ratio,
            num_projection_per_measurement=self.num_projection_per_measurement,
            non_local_move=self.non_local_move,
            E_scf=self.E_scf,
            atomic_force=self.atomic_force,
            epsilon_PW=self.epsilon_PW,
            mcmc_seed=self.mcmc_seed,
            verbosity=self.verbosity,
            poll_interval=self.poll_interval,
            target_error=self.target_error,
            pilot_steps=self.pilot_steps,
            num_gfmc_projections=self.num_gfmc_projections,
            max_continuation=self.max_continuation,
        )
        enc = Container(
            label=label,
            dirname=dirname,
            input_files=[self.hamiltonian_file],
            workflow=wf,
        )
        return enc

    def configure(self) -> dict:
        """Validate parameters and return configuration summary."""
        return {
            "alat_list": self.alat_list,
            "polynomial_order": self.polynomial_order,
            "hamiltonian_file": self.hamiltonian_file,
            "server_machine": self.server_machine_name,
            "number_of_walkers": self.number_of_walkers,
            "max_time": self.max_time,
            "target_error": self.target_error,
            "max_continuation": self.max_continuation,
        }

    async def run(self) -> tuple:
        """Run LRDMC at each alat, then extrapolate to a²→0.

        Every ``alat`` value is launched in parallel.  Each child
        :class:`LRDMC_Workflow` independently handles its own
        calibration (``_pilot_a``), error-bar pilot (``_pilot_b``),
        and production phase.

        Returns
        -------
        tuple
            ``(status, output_files, output_values)``
        """
        self._ensure_project_dir()
        _wd = self.project_dir
        sorted_alats = sorted(self.alat_list, reverse=True)

        # -- helper: run a single alat, return a uniform result tuple ------
        async def _run_one(enc):
            try:
                status, out_files, out_values = await enc.async_launch()
                return enc, status, out_files, out_values, None
            except Exception as exc:
                return enc, "failed", [], {}, exc

        # Create and launch all alat runs in parallel
        # Set each child Container's root_dir to this workflow's project_dir
        # so that lrdmc_alat_XXX directories are created inside it, not in CWD.
        enc_workflows = [self._make_lrdmc_workflow(alat) for alat in sorted_alats]
        for enc in enc_workflows:
            enc.root_dir = _wd
            enc.project_dir = os.path.join(_wd, enc.dirname)
        logger.info(f"Launching {len(enc_workflows)} LRDMC runs in parallel...")
        for enc in enc_workflows:
            logger.info(f"  {enc.label}")

        tasks = [asyncio.create_task(_run_one(enc)) for enc in enc_workflows]
        all_results = list(await asyncio.gather(*tasks))

        # -- collect results -----------------------------------------------
        restart_chks = []
        per_alat_results = []
        errors = []

        for enc, status, out_files, out_values, error in all_results:
            if error is not None:
                logger.error(f"[{enc.label}] failed: {error}")
                errors.append(str(error))
                continue
            if status not in ("success", "completed", WorkflowStatus.COMPLETED):
                logger.error(f"[{enc.label}] returned status={status}")
                errors.append(f"{enc.label}: status={status}")
                continue

            # Collect the restart checkpoint from the sub-run directory
            chk = out_values.get("restart_chk")
            if chk:
                full_path = os.path.join(enc.project_dir, chk)
                if os.path.isfile(full_path):
                    restart_chks.append(full_path)

            per_alat_results.append(
                {
                    "alat": out_values.get("alat", None),
                    "energy": out_values.get("energy", None),
                    "energy_error": out_values.get("energy_error", None),
                }
            )

        if errors:
            self.status = WorkflowStatus.FAILED
            self.output_values["errors"] = errors
            return self.status, [], {"error": "; ".join(errors)}

        logger.info(f"All {len(self.alat_list)} LRDMC runs completed.")

        # ── Extrapolation ─────────────────────────────────────────
        if len(restart_chks) >= 2:
            ext_energy, ext_error = self._extrapolate_energy(restart_chks)
            if ext_energy is not None:
                self.output_values["extrapolated_energy"] = ext_energy
                self.output_values["extrapolated_energy_error"] = ext_error
                logger.info(f"Extrapolated energy (a->0): {ext_energy} +- {ext_error} Ha")
        else:
            msg = f"Only {len(restart_chks)} checkpoint(s) found; cannot extrapolate."
            logger.error(msg)
            self.status = WorkflowStatus.FAILED
            self.output_values["error"] = msg
            return self.status, [], {"error": msg}

        self.output_values["per_alat_results"] = per_alat_results
        self.output_files = restart_chks
        self.status = WorkflowStatus.COMPLETED
        return self.status, self.output_files, self.output_values

    def _extrapolate_energy(self, restart_chks: List[str]):
        """Run ``jqmc-tool lrdmc extrapolate-energy``.

        Returns
        -------
        tuple
            ``(energy, error)`` or ``(None, None)``.
        """
        chk_args = " ".join(restart_chks)
        cmd = (
            f"jqmc-tool lrdmc extrapolate-energy {chk_args} "
            f"-p {self.polynomial_order} "
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
                cwd=self.project_dir,
            )
            return self._parse_extrapolation_output(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"extrapolate-energy failed: {e.stderr}")
            return None, None

    @staticmethod
    def _parse_extrapolation_output(text: str):
        """Parse ``For a -> 0 bohr: E = <val> +- <err> Ha.`` from output."""
        pattern = re.compile(
            r"For a -> 0 bohr:\s*E\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"
            r"\s*\+\-\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)"
        )
        match = pattern.search(text)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None, None
