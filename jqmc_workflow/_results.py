"""Structured result types for jQMC output parsing.

These dataclasses represent **facts only** — deterministic data extracted
from jQMC stdout/stderr and associated files.  They contain no heuristic
judgments; diagnostic analysis (failure categorization, convergence
assessment, etc.) belongs to a higher-level layer (e.g. jqmc-mcp).

Classes
-------
VMC_Step_Data
    One optimization step of a VMC run.
VMC_Diagnostic_Data
    Aggregated parse result for an entire VMC optimization.
MCMC_Diagnostic_Data
    Parse result for an MCMC sampling run.
LRDMC_Diagnostic_Data
    Parse result for an LRDMC calculation.
LRDMC_Ext_Diagnostic_Data
    Parse result for an LRDMC extrapolation (multi-alat).
Input_Parameters
    Key parameters extracted from a TOML input file.
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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── VMC ───────────────────────────────────────────────────────────


@dataclass
class VMC_Step_Data:
    """Data for one VMC optimization step.

    Attributes
    ----------
    step : int
        Optimization step number (``Optimization step = N/M`` → N).
    energy : float or None
        Total energy ``E = X +- Y`` → X  (Ha).
    energy_error : float or None
        Energy statistical error → Y  (Ha).
    max_force : float or None
        Maximum force ``Max f = X +- Y`` → X  (Ha/a.u.).
    max_force_error : float or None
        Force error → Y  (Ha/a.u.).
    signal_to_noise_ratio : float or None
        ``Max of signal-to-noise of f = max(|f|/|std f|) = X``.
    avg_walker_weight : float or None
        ``Average of walker weights is X``.
    acceptance_ratio : float or None
        ``Acceptance ratio is X %`` → X / 100.
    total_time_sec : float or None
        ``Total elapsed time for MCMC N steps. = X sec.``
    precompilation_time_sec : float or None
        ``Pre-compilation time for MCMC = X sec.``
    net_time_sec : float or None
        ``Net total time for MCMC = X sec.``
    timing_breakdown : dict
        Per-MCMC-step timing breakdown (msec).  Keys match the
        jQMC log lines, e.g. ``"mcmc_update"``, ``"e_L"``, etc.
    """

    step: int
    energy: Optional[float] = None
    energy_error: Optional[float] = None
    max_force: Optional[float] = None
    max_force_error: Optional[float] = None
    signal_to_noise_ratio: Optional[float] = None
    avg_walker_weight: Optional[float] = None
    acceptance_ratio: Optional[float] = None
    total_time_sec: Optional[float] = None
    precompilation_time_sec: Optional[float] = None
    net_time_sec: Optional[float] = None
    timing_breakdown: dict = field(default_factory=dict)


@dataclass
class VMC_Diagnostic_Data:
    """Aggregated parse result for an entire VMC optimization.

    Attributes
    ----------
    steps : list of VMC_Step_Data
        Per-step data in chronological order.
    total_opt_steps : int or None
        Total optimization steps (``Optimization step = N/M`` → M).
    total_opt_time_sec : float or None
        ``Total elapsed time for optimization N steps. = X sec.``
    opt_timing_breakdown : dict
        Per-optimization-step timing breakdown (sec).  Keys:
        ``"mcmc_run"``, ``"get_E"``, ``"get_gF"``, ``"optimizer"``,
        ``"param_update"``, ``"mpi_barrier"``, ``"misc"``.
    optimized_hamiltonian : str or None
        Path to the last ``hamiltonian_data_opt_step_*.h5`` file found.
    restart_checkpoint : str or None
        Restart file name from ``Dump restart checkpoint file(s) to X.``.
        ``None`` if the line was not found (indicates abnormal termination).
    stderr_tail : str
        Last portion of stderr (up to 200 lines).
    """

    steps: list = field(default_factory=list)  # list[VMC_Step_Data]
    total_opt_steps: Optional[int] = None
    total_opt_time_sec: Optional[float] = None
    opt_timing_breakdown: dict = field(default_factory=dict)
    optimized_hamiltonian: Optional[str] = None
    restart_checkpoint: Optional[str] = None
    stderr_tail: str = ""


# ── MCMC ──────────────────────────────────────────────────────────


@dataclass
class MCMC_Diagnostic_Data:
    """Parse result for an MCMC sampling run.

    Attributes
    ----------
    acceptance_ratio : float or None
        ``Acceptance ratio is X %`` → X / 100.
    avg_walker_weight : float or None
        ``Average of walker weights is X``.
    total_time_sec : float or None
        ``Total elapsed time for MCMC N steps. = X sec.``
    precompilation_time_sec : float or None
        ``Pre-compilation time for MCMC = X sec.``
    net_time_sec : float or None
        ``Net total time for MCMC = X sec.``
    timing_breakdown : dict
        Per-MCMC-step timing breakdown (msec).  Keys: ``"mcmc_update"``,
        ``"e_L"``, ``"de_L_dR_dr"``, ``"dln_Psi_dR_dr"``,
        ``"dln_Psi_dc"``, ``"de_L_dc"``, ``"mpi_barrier"``, ``"misc"``.
    energy : float or None
        Energy from jqmc-tool post-processing.
    energy_error : float or None
        Energy error from jqmc-tool post-processing.
    hamiltonian_data_file : str or None
        ``[control] hamiltonian_h5`` value from the input TOML.
    restart_checkpoint : str or None
        Restart file name from ``Dump restart checkpoint file(s) to X.``.
        ``None`` if the line was not found.
    stderr_tail : str
        Last portion of stderr (up to 200 lines).
    """

    acceptance_ratio: Optional[float] = None
    avg_walker_weight: Optional[float] = None
    total_time_sec: Optional[float] = None
    precompilation_time_sec: Optional[float] = None
    net_time_sec: Optional[float] = None
    timing_breakdown: dict = field(default_factory=dict)
    energy: Optional[float] = None
    energy_error: Optional[float] = None
    hamiltonian_data_file: Optional[str] = None
    restart_checkpoint: Optional[str] = None
    stderr_tail: str = ""


# ── LRDMC ─────────────────────────────────────────────────────────


@dataclass
class LRDMC_Diagnostic_Data:
    """Parse result for an LRDMC calculation.

    Attributes
    ----------
    survived_walkers_ratio : float or None
        ``Survived walkers ratio = X %`` → X / 100.
    avg_num_projections : float or None
        ``Average of the number of projections = X``.
    total_time_sec : float or None
        ``Total GFMC time for N branching steps = X sec.``
    precompilation_time_sec : float or None
        ``Pre-compilation time for GFMC = X sec.``
    net_time_sec : float or None
        ``Net GFMC time without pre-compilations = X sec.``
    timing_breakdown : dict
        Per-branching timing breakdown (msec).  Keys vary by LRDMC
        variant, e.g. ``"projection"``, ``"observable"``,
        ``"mpi_barrier"``, ``"collection"``, ``"reconfiguration"``,
        ``"e_L"``, ``"de_L_dR_dr"``, ``"update_E_scf"``, ``"misc"``.
    energy : float or None
        Energy from jqmc-tool post-processing.
    energy_error : float or None
        Energy error from jqmc-tool post-processing.
    hamiltonian_data_file : str or None
        ``[control] hamiltonian_h5`` value from the input TOML.
    restart_checkpoint : str or None
        Restart file name from ``Dump restart checkpoint file(s) to X.``.
        ``None`` if the line was not found.
    stderr_tail : str
        Last portion of stderr (up to 200 lines).
    """

    survived_walkers_ratio: Optional[float] = None
    avg_num_projections: Optional[float] = None
    total_time_sec: Optional[float] = None
    precompilation_time_sec: Optional[float] = None
    net_time_sec: Optional[float] = None
    timing_breakdown: dict = field(default_factory=dict)
    energy: Optional[float] = None
    energy_error: Optional[float] = None
    hamiltonian_data_file: Optional[str] = None
    restart_checkpoint: Optional[str] = None
    stderr_tail: str = ""


# ── LRDMC extrapolation ──────────────────────────────────────────


@dataclass
class LRDMC_Ext_Diagnostic_Data:
    """Parse result for an LRDMC a²→0 extrapolation.

    Attributes
    ----------
    extrapolated_energy : float or None
        ``For a -> 0 bohr: E = X +- Y Ha.`` → X.
    extrapolated_energy_error : float or None
        Y from the above.
    per_alat_results : list of dict
        Each dict has ``{"alat": float, "energy": float, "energy_error": float}``.
    stderr_tail : str
        Last portion of stderr (up to 200 lines).
    """

    extrapolated_energy: Optional[float] = None
    extrapolated_energy_error: Optional[float] = None
    per_alat_results: list = field(default_factory=list)
    stderr_tail: str = ""


# ── Input parameters ──────────────────────────────────────────────


@dataclass
class Input_Parameters:
    """Key parameters extracted from a workflow directory.

    Parameters are sourced from the TOML input files recorded in
    ``workflow_state.toml``, with defaults filled from
    ``jqmc.jqmc_miscs.cli_parameters``.

    Each entry in ``per_input`` is a dict::

        {
            "input_file": "input_1.toml",
            "output_file": "out_1.o",
            "job_type": "vmc",
            "control": { ... all [control] params with defaults ... },
            "<job_type>": { ... all job-type params with defaults ... },
        }

    Attributes
    ----------
    actual_opt_steps : int or None
        For VMC: last completed optimization step stored in
        ``restart.h5`` (``rank_0/driver_config`` attrs ``i_opt``).
        ``None`` for non-VMC workflows.
    per_input : list of dict
        Per-input-file parameters.  One dict per ``[[jobs]]`` entry
        in ``workflow_state.toml``.
    """

    actual_opt_steps: Optional[int] = None
    per_input: list = field(default_factory=list)  # list[dict]
