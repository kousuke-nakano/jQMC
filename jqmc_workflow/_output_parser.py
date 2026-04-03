"""Deterministic output parser for jQMC stdout/stderr.

This module extracts **facts only** from jQMC log files using regular
expressions.  It contains no heuristic judgments or convergence assessments
— those belong to a higher-level diagnostics layer (e.g. jqmc-mcp).

The parser unifies and extends the scattered parse logic previously found
in ``vmc_workflow._parse_output``, ``vmc_workflow._parse_all_snr``,
``_error_estimator.parse_net_time``, and ``_lrdmc_calibration.parse_survived_walkers_ratio``.

Key improvement over the old parsers: per-optimization-step data extraction
for VMC (the old parsers only kept the last value).

Public API
----------
parse_vmc_output(work_dir)
    Parse VMC optimization stdout/stderr → VMC_Diagnostic_Data.
parse_mcmc_output(work_dir)
    Parse MCMC sampling stdout/stderr → MCMC_Diagnostic_Data.
parse_lrdmc_output(work_dir)
    Parse LRDMC stdout/stderr → LRDMC_Diagnostic_Data.
parse_input_params(toml_path)
    Extract key parameters from a TOML input file → Input_Parameters.
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

import glob
import os
import re
from logging import getLogger
from typing import Optional

import toml

from ._results import (
    Input_Parameters,
    LRDMC_Diagnostic_Data,
    LRDMC_Ext_Diagnostic_Data,
    MCMC_Diagnostic_Data,
    VMC_Diagnostic_Data,
    VMC_Step_Data,
)

logger = getLogger("jqmc-workflow").getChild(__name__)


# ── Atomic-force table parser ─────────────────────────────────────


def parse_ufloat_short(text: str):
    """Parse uncertainties short format, e.g. ``+0.123(45)`` → (0.123, 0.045).

    Handles several notations produced by jqmc:

    * ``+0.0114(14)``   — integer uncertainty digits in last decimal place
    * ``+3(8)e-05``     — scientific notation with integer uncertainty
    * ``+3.9(3.5)e-05`` — scientific notation with decimal uncertainty

    Parameters
    ----------
    text : str
        A single token like ``"+0.0114(14)"``, ``"-1.23(4)"``,
        ``"+3(8)e-05"``, or ``"+3.9(3.5)e-05"``.

    Returns
    -------
    tuple
        ``(value, uncertainty)`` or ``(None, None)`` on failure.
    """
    m = re.match(r"([+-]?\d+\.?\d*)\((\d+\.?\d*)\)([eE][+-]?\d+)?", text.strip())
    if not m:
        return None, None
    val_str = m.group(1)
    unc_str = m.group(2)
    exp_str = m.group(3)  # e.g. "e-05" or None
    val = float(val_str)
    if "." in unc_str:
        # Decimal uncertainty: use the value directly at the mantissa scale
        unc = float(unc_str)
    else:
        # Integer uncertainty: digits in the last decimal place of the value
        if "." in val_str:
            n_decimals = len(val_str.split(".")[1])
        else:
            n_decimals = 0
        unc = int(unc_str) * 10 ** (-n_decimals)
    if exp_str is not None:
        exp_factor = 10 ** int(exp_str[1:])  # skip 'e'/'E'
        val *= exp_factor
        unc *= exp_factor
    return val, unc


def parse_force_table(text: str):
    """Parse an ``Atomic Forces:`` table from jqmc-tool output.

    Expected format::

        Atomic Forces:
        ------------------------------------------------
        Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
        ------------------------------------------------
        O       -0.0114(14)  -0.0082(11)  -0.0173(13)
        H       +0.0112(11)  +0.0050(7)   +0.0054(9)
        ------------------------------------------------

    Parameters
    ----------
    text : str
        Full stdout from ``jqmc-tool {mcmc,lrdmc} compute-force``.

    Returns
    -------
    list of dict or None
        Each dict: ``{label, Fx, Fx_err, Fy, Fy_err, Fz, Fz_err}``.
        Returns *None* when no force data is found.
    """
    lines = text.splitlines()
    forces = []
    in_data = False
    for line in lines:
        if "Atomic Forces:" in line:
            in_data = False
            continue
        stripped = line.strip()
        if "Fx" in stripped:
            in_data = True
            continue
        if stripped.startswith("---"):
            if in_data and forces:
                break
            continue
        if not in_data:
            continue
        tokens = stripped.split()
        if len(tokens) < 4:
            continue
        label = tokens[0]
        fx, fx_err = parse_ufloat_short(tokens[1])
        fy, fy_err = parse_ufloat_short(tokens[2])
        fz, fz_err = parse_ufloat_short(tokens[3])
        if fx is not None:
            forces.append(
                {
                    "label": label,
                    "Fx": fx,
                    "Fx_err": fx_err,
                    "Fy": fy,
                    "Fy_err": fy_err,
                    "Fz": fz,
                    "Fz_err": fz_err,
                }
            )
    return forces or None


def repair_forces_from_output(work_dir: str) -> bool:
    """Re-parse forces from output files and update ``workflow_state.toml``.

    This repairs corrupted force data caused by the pre-fix
    ``parse_ufloat_short`` that ignored scientific notation (e.g.
    ``+3(8)e-05`` was parsed as ``3.0`` instead of ``3e-05``).

    Returns *True* if the TOML was updated, *False* otherwise.
    """
    state_path = os.path.join(work_dir, "workflow_state.toml")
    if not os.path.isfile(state_path):
        return False

    out_files = _find_output_files(work_dir)
    if not out_files:
        return False
    last_out = out_files[-1]

    # Read and parse the force table from the output file
    try:
        with open(last_out, "r") as f:
            text = f.read()
    except Exception:
        return False

    forces = parse_force_table(text)
    if forces is None:
        return False

    # Update the TOML
    state = toml.load(state_path)
    state.setdefault("result", {})["forces"] = forces
    with open(state_path, "w") as f:
        toml.dump(state, f)

    logger.info(f"  Repaired forces in {work_dir} from {os.path.basename(last_out)}")
    return True


# ── Compiled regex patterns ───────────────────────────────────────
#
# All patterns are compiled once at module level for efficiency.

# "Optimization step =   1/10" or "Optimization step = 1/10."
_RE_OPT_STEP = re.compile(r"Optimization\s+step\s*=\s*(\d+)\s*/\s*(\d+)")

# "E = -76.438901 +- 0.000123 Ha" (energy line)
_RE_ENERGY = re.compile(
    r"E\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"
    r"\s*\+\-\s*"
    r"(\d+\.?\d*(?:[eE][+-]?\d+)?)"
)

# "Max f = 17.984 +- 0.330 Ha/a.u." or "Max f = 17.984 +- 0.330"
_RE_MAX_FORCE = re.compile(
    r"Max\s+f\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"
    r"\s*\+\-\s*"
    r"(\d+\.?\d*(?:[eE][+-]?\d+)?)"
)

# "Max of signal-to-noise of f = max(|f|/|std f|) = 126.871."
_RE_SNR = re.compile(
    r"Max of signal-to-noise of f\s*=\s*max\(\|f\|/\|std f\|\)\s*=\s*"
    r"([-+]?\d+(?:\.\d+)?)"
)

# "Average of walker weights is 0.799. Ideal is ~ 0.800. Adjust epsilon_AS."
_RE_WALKER_WEIGHT = re.compile(r"Average of walker weights is\s+(\d+(?:\.\d+)?)")

# "Acceptance ratio is 51.46 %.  Ideal is ~ 50.00%. Adjust Dt."
_RE_ACCEPTANCE = re.compile(r"Acceptance ratio is\s+(\d+(?:\.\d+)?)\s*%")

# "Net total time for MCMC = 9.72 sec."
_RE_NET_MCMC = re.compile(r"Net total time for MCMC\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")

# "Total elapsed time for MCMC 1000 steps. = 21.07 sec."
_RE_TOTAL_MCMC = re.compile(
    r"Total elapsed time for MCMC\s+\d+\s+steps\.?\s*=\s*"
    r"([\d.]+(?:[eE][+-]?\d+)?)\s*sec"
)

# "Pre-compilation time for MCMC = 11.35 sec."
_RE_PRECOMP_MCMC = re.compile(r"Pre-compilation time for MCMC\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")

# Per-MCMC-step timing breakdown (msec)
_RE_TIME_MCMC_UPDATE = re.compile(r"Time for MCMC update\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_E_L = re.compile(r"Time for computing e_L\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_DE_L_DR = re.compile(r"Time for computing de_L/dR and de_L/dr\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_DLN_PSI_DR = re.compile(r"Time for computing dln_Psi/dR and dln_Psi/dr\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_DLN_PSI_DC = re.compile(r"Time for computing dln_Psi/dc\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_DE_L_DC = re.compile(r"Time for computing de_L/dc\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_MPI_MCMC = re.compile(r"Time for MPI barrier after MCMC update\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_MISC_MCMC = re.compile(r"Time for misc\.\s*\(others\)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")

# VMC optimization-level timing (sec, appears once after all opt steps)
# "Total elapsed time for optimization 10 steps. = 123.45 sec."
_RE_TOTAL_OPT = re.compile(
    r"Total elapsed time for optimization\s+\d+\s+steps\.?\s*=\s*"
    r"([\d.]+(?:[eE][+-]?\d+)?)\s*sec"
)
# Per-opt-step breakdown (sec)
_RE_OPT_MCMC_RUN = re.compile(r"Time for MCMC run\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")
_RE_OPT_GET_E = re.compile(r"Time for computing E\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")
_RE_OPT_GET_GF = re.compile(r"Time for computing generalized forces \(gF\)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")
_RE_OPT_OPTIMIZER = re.compile(r"Time for optimizer \(SR/optax\)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")
_RE_OPT_PARAM_UPDATE = re.compile(r"Time for parameter update\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")
_RE_OPT_MPI_BARRIER = re.compile(r"Time for MPI barrier\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")
_RE_OPT_MISC = re.compile(r"Time for misc\.\s*\(others\)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")

# "Net GFMC time without pre-compilations = 2832.326 sec."
_RE_NET_GFMC = re.compile(
    r"Net GFMC time without pre-compilations\s*=\s*"
    r"([\d.]+(?:[eE][+-]?\d+)?)\s*sec"
)

# "Total GFMC time for 1000 branching steps = 3000.0 sec."
_RE_TOTAL_GFMC = re.compile(
    r"Total GFMC time for\s+\d+\s+branching steps\s*=\s*"
    r"([\d.]+(?:[eE][+-]?\d+)?)\s*sec"
)

# "Pre-compilation time for GFMC = 167.674 sec."
_RE_PRECOMP_GFMC = re.compile(r"Pre-compilation time for GFMC\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")

# Per-branching timing breakdown (msec) — GFMC
_RE_TIME_PROJECTION = re.compile(
    r"Projection(?:\s+time per branching|\s+between branching)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec"
)
_RE_TIME_OBSERVABLE = re.compile(r"Time for Observable measurement.*?=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_MPI_GFMC = re.compile(r"Time for MPI barrier before branching\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_COLLECTION = re.compile(r"Time for walker observable collections.*?=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_RECONFIGURATION = re.compile(r"Time for walker reconfiguration.*?=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_UPDATE_ESCF = re.compile(r"Time for updating E_scf\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_MISC_GFMC = re.compile(r"Time for misc\.\s*\(others\)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")

# "Survived walkers ratio = 95.2 %"
_RE_SURVIVED = re.compile(r"Survived walkers ratio\s*=\s*(\d+\.?\d*)\s*%")

# "Average of the number of projections  = 123"
_RE_AVG_PROJECTIONS = re.compile(r"Average of the number of projections\s*=\s*(\d+(?:\.\d+)?)")

# "For a -> 0 bohr: E = -76.43 +- 0.01 Ha."  (LRDMC extrapolation)
_RE_EXTRAP_ENERGY = re.compile(
    r"For a -> 0 bohr:\s*E\s*=\s*"
    r"([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"
    r"\s*\+\-\s*"
    r"(\d+\.?\d*(?:[eE][+-]?\d+)?)"
)

_STDERR_TAIL_LINES = 200

# "Dump restart checkpoint file(s) to restart.h5."
_RE_RESTART_CHECKPOINT = re.compile(r"Dump restart checkpoint file\(s\) to\s+(\S+)\.\s*$", re.MULTILINE)

# ── Run-level metadata (header section, appears once per output file) ──

# "The number of MPI process = 4."
_RE_MPI_PROCESSES = re.compile(r"The number of MPI process\s*=\s*(\d+)")

# "The number of walkers assigned for each MPI process = 4."
_RE_WALKERS_PER_PROCESS = re.compile(r"The number of walkers assigned for each MPI process\s*=\s*(\d+)")

# "JAX backend = gpu."
_RE_JAX_BACKEND = re.compile(r"JAX backend\s*=\s*(\w+)")

# Fallback: "Running on CPUs or single GPU. JAX distributed initialization is skipped."
_RE_JAX_CPU_FALLBACK = re.compile(r"Running on CPUs or single GPU")

# "*** XLA Global devices recognized by JAX***"
# followed by e.g. "[CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]"
_RE_XLA_GLOBAL_HEADER = re.compile(r"\*{3}\s*XLA Global devices recognized by JAX\s*\*{3}")
_RE_XLA_DEVICE_LIST = re.compile(r"\[([^\]]+)\]")


# ── Internal helpers ──────────────────────────────────────────────


def _read_text(path: str) -> Optional[str]:
    """Read a text file, returning *None* if it doesn't exist or can't be read."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read()
    except OSError as exc:
        logger.debug("Cannot read %s: %s", path, exc)
        return None


def _tail(text: str, max_lines: int = _STDERR_TAIL_LINES) -> str:
    """Return the last *max_lines* lines of *text*."""
    lines = text.splitlines(keepends=True)
    return "".join(lines[-max_lines:])


def _find_input_files(work_dir: str) -> list:
    """Return jQMC input TOML files in *work_dir*, sorted by step index.

    Reads ``workflow_state.toml`` ``[[jobs]]`` records and returns the
    ``input_file`` paths that exist on disk, ordered by ``step``.
    """
    state_path = os.path.join(work_dir, "workflow_state.toml")
    if not os.path.isfile(state_path):
        return []
    try:
        state = toml.load(state_path)
    except Exception:
        return []
    files = []
    for job in state.get("jobs", []):
        name = job.get("input_file", "")
        if name:
            path = os.path.join(work_dir, name)
            if os.path.isfile(path):
                files.append((job.get("step", 0), path))
    files.sort()
    return [path for _, path in files]


def _find_hamiltonian_h5(work_dir: str) -> Optional[str]:
    """Extract ``[control] hamiltonian_h5`` from input TOML files in *work_dir*.

    Uses ``workflow_state.toml`` ``[[jobs]]`` to locate input files.
    Returns the value from the last input that contains
    ``[control] hamiltonian_h5``, or *None* if not found.
    """
    for fpath in reversed(_find_input_files(work_dir)):
        try:
            data = toml.load(fpath)
            h5 = data.get("control", {}).get("hamiltonian_h5")
            if h5:
                return h5
        except Exception:
            continue
    return None


def _parse_run_metadata(text: str) -> dict:
    """Extract run-level metadata from a jQMC output file header.

    Parses the header section that appears once per output file before
    any computation output.  Returns a dict with keys::

        {
            "num_mpi_processes": int | None,
            "num_walkers_per_process": int | None,
            "jax_backend": str | None,      # "gpu", "cpu", etc.
            "jax_devices": list | None,      # ["CudaDevice(id=0)", ...]
        }
    """
    meta: dict = {
        "num_mpi_processes": None,
        "num_walkers_per_process": None,
        "jax_backend": None,
        "jax_devices": None,
    }

    for line in text.splitlines():
        # MPI processes
        m = _RE_MPI_PROCESSES.search(line)
        if m:
            meta["num_mpi_processes"] = int(m.group(1))
            continue

        # Walkers per process
        m = _RE_WALKERS_PER_PROCESS.search(line)
        if m:
            meta["num_walkers_per_process"] = int(m.group(1))
            continue

        # JAX backend (explicit)
        m = _RE_JAX_BACKEND.search(line)
        if m:
            meta["jax_backend"] = m.group(1).lower()
            continue

        # JAX backend (fallback: CPU/single GPU)
        if _RE_JAX_CPU_FALLBACK.search(line):
            if meta["jax_backend"] is None:
                meta["jax_backend"] = "cpu"
            continue

    # XLA Global devices — parse from the full text (may span lines)
    m_header = _RE_XLA_GLOBAL_HEADER.search(text)
    if m_header:
        # The device list is on the next non-empty line after the header
        rest = text[m_header.end() :]
        for next_line in rest.splitlines():
            stripped = next_line.strip()
            if not stripped:
                continue
            m_devs = _RE_XLA_DEVICE_LIST.search(stripped)
            if m_devs:
                # Split "CudaDevice(id=0), CudaDevice(id=1), ..."
                raw_devices = m_devs.group(1)
                meta["jax_devices"] = [d.strip() for d in raw_devices.split(",") if d.strip()]
            break

    return meta


def _apply_run_metadata(result, meta: dict) -> None:
    """Apply run metadata dict to a diagnostic data object."""
    result.num_mpi_processes = meta["num_mpi_processes"]
    result.num_walkers_per_process = meta["num_walkers_per_process"]
    result.jax_backend = meta["jax_backend"]
    result.jax_devices = meta["jax_devices"]


def _find_output_files(work_dir: str) -> list:
    """Return jQMC stdout files in *work_dir*, sorted by step index.

    Reads ``workflow_state.toml`` ``[[jobs]]`` records and returns the
    ``output_file`` paths that exist on disk, ordered by ``step``.
    """
    state_path = os.path.join(work_dir, "workflow_state.toml")
    if not os.path.isfile(state_path):
        return []
    try:
        state = toml.load(state_path)
    except Exception:
        return []
    files = []
    for job in state.get("jobs", []):
        name = job.get("output_file", "")
        if name:
            path = os.path.join(work_dir, name)
            if os.path.isfile(path):
                files.append((job.get("step", 0), path))
    files.sort()
    return [path for _, path in files]


# ── VMC parser ────────────────────────────────────────────────────


def _parse_vmc_log_text(text: str) -> list:
    """Parse VMC log text into a list of :class:`VMC_Step_Data`.

    The VMC output repeats blocks like::

        ==================================================================
        Optimization step = 1/10.
        ==================================================================
        ...
        Net total time for MCMC = 9.72 sec.
        Average of walker weights is 0.799. ...
        Acceptance ratio is 51.46 %. ...
        E = -16.61072 +- 0.00704 Ha
        Max f = 17.984 +- 0.330 Ha/a.u.
        Max of signal-to-noise of f = max(|f|/|std f|) = 126.871.
        ...

    This function groups the data by optimization step.  Lines before
    the first ``Optimization step`` header are ignored.

    Returns
    -------
    list of VMC_Step_Data
        One entry per optimization step found.
    """
    steps: list[VMC_Step_Data] = []
    current: Optional[VMC_Step_Data] = None
    total_opt_steps: Optional[int] = None

    for line in text.splitlines():
        # ── Optimization step header ──
        m = _RE_OPT_STEP.search(line)
        if m:
            step_num = int(m.group(1))
            total_opt_steps = int(m.group(2))
            current = VMC_Step_Data(step=step_num)
            steps.append(current)
            continue

        if current is None:
            continue

        # ── Net MCMC time ──
        m = _RE_NET_MCMC.search(line)
        if m:
            current.net_time_sec = float(m.group(1))
            continue

        # ── Total MCMC time ──
        m = _RE_TOTAL_MCMC.search(line)
        if m:
            current.total_time_sec = float(m.group(1))
            continue

        # ── Pre-compilation MCMC time ──
        m = _RE_PRECOMP_MCMC.search(line)
        if m:
            current.precompilation_time_sec = float(m.group(1))
            continue

        # ── Per-step MCMC timing breakdown (msec) ──
        m = _RE_TIME_MCMC_UPDATE.search(line)
        if m:
            current.timing_breakdown["mcmc_update"] = float(m.group(1))
            continue
        m = _RE_TIME_E_L.search(line)
        if m:
            current.timing_breakdown["e_L"] = float(m.group(1))
            continue
        m = _RE_TIME_DE_L_DR.search(line)
        if m:
            current.timing_breakdown["de_L_dR_dr"] = float(m.group(1))
            continue
        m = _RE_TIME_DLN_PSI_DR.search(line)
        if m:
            current.timing_breakdown["dln_Psi_dR_dr"] = float(m.group(1))
            continue
        m = _RE_TIME_DLN_PSI_DC.search(line)
        if m:
            current.timing_breakdown["dln_Psi_dc"] = float(m.group(1))
            continue
        m = _RE_TIME_DE_L_DC.search(line)
        if m:
            current.timing_breakdown["de_L_dc"] = float(m.group(1))
            continue
        m = _RE_TIME_MPI_MCMC.search(line)
        if m:
            current.timing_breakdown["mpi_barrier"] = float(m.group(1))
            continue
        m = _RE_TIME_MISC_MCMC.search(line)
        if m:
            current.timing_breakdown["misc"] = float(m.group(1))
            continue

        # ── Walker weight ──
        m = _RE_WALKER_WEIGHT.search(line)
        if m:
            current.avg_walker_weight = float(m.group(1))
            continue

        # ── Acceptance ratio ──
        m = _RE_ACCEPTANCE.search(line)
        if m:
            current.acceptance_ratio = float(m.group(1)) / 100.0
            continue

        # ── Signal-to-noise (must be checked BEFORE energy) ──
        m = _RE_SNR.search(line)
        if m:
            current.signal_to_noise_ratio = float(m.group(1))
            continue

        # ── Max force (must be checked BEFORE energy) ──
        m = _RE_MAX_FORCE.search(line)
        if m:
            current.max_force = float(m.group(1))
            current.max_force_error = float(m.group(2))
            continue

        # ── Energy ──
        m = _RE_ENERGY.search(line)
        if m:
            current.energy = float(m.group(1))
            current.energy_error = float(m.group(2))
            continue

    return steps


def parse_vmc_output(work_dir: str) -> VMC_Diagnostic_Data:
    """Parse VMC optimization output from *work_dir*.

    Discovers output files from ``workflow_state.toml`` ``[[jobs]]``
    records, parses per-step data, and looks for
    ``hamiltonian_data_opt_step_*.h5``.

    Parameters
    ----------
    work_dir : str
        Path to the VMC working directory.

    Returns
    -------
    VMC_Diagnostic_Data
        Structured parse result containing per-step data and metadata.
    """
    result = VMC_Diagnostic_Data()

    if not os.path.isdir(work_dir):
        logger.warning("parse_vmc_output: directory not found: %s", work_dir)
        return result

    # ── Discover and parse stdout files ──
    output_files = _find_output_files(work_dir)
    all_steps: list[VMC_Step_Data] = []

    for fpath in output_files:
        text = _read_text(fpath)
        if text is None:
            continue
        file_steps = _parse_vmc_log_text(text)
        all_steps.extend(file_steps)

    result.steps = all_steps

    # ── Run-level metadata (MPI, walkers, JAX) from first output file ──
    if output_files:
        first_text = _read_text(output_files[0])
        if first_text:
            _apply_run_metadata(result, _parse_run_metadata(first_text))

    # Extract total_opt_steps from the last output file (the M in N/M)
    if output_files:
        last_text = _read_text(output_files[-1])
        if last_text:
            for line in last_text.splitlines():
                m = _RE_OPT_STEP.search(line)
                if m:
                    result.total_opt_steps = int(m.group(2))

            # ── Optimization-level timing (appears once after all steps) ──
            m = _RE_TOTAL_OPT.search(last_text)
            if m:
                result.total_opt_time_sec = float(m.group(1))

            _opt_breakdown_patterns = [
                (_RE_OPT_MCMC_RUN, "mcmc_run"),
                (_RE_OPT_GET_E, "get_E"),
                (_RE_OPT_GET_GF, "get_gF"),
                (_RE_OPT_OPTIMIZER, "optimizer"),
                (_RE_OPT_PARAM_UPDATE, "param_update"),
                (_RE_OPT_MPI_BARRIER, "mpi_barrier"),
                (_RE_OPT_MISC, "misc"),
            ]
            for pattern, key in _opt_breakdown_patterns:
                m_bd = pattern.search(last_text)
                if m_bd:
                    result.opt_timing_breakdown[key] = float(m_bd.group(1))

    # ── optimized hamiltonian ──
    # Find hamiltonian_data_opt_step_*.h5 and sort numerically by step.
    h5_pattern = os.path.join(work_dir, "hamiltonian_data_opt_step_*.h5")
    h5_files = glob.glob(h5_pattern)
    if h5_files:

        def _h5_step_num(path: str) -> int:
            m = re.search(r"hamiltonian_data_opt_step_(\d+)\.h5$", path)
            return int(m.group(1)) if m else -1

        h5_files.sort(key=_h5_step_num)
        result.optimized_hamiltonian = h5_files[-1]
    else:
        logger.warning(
            "parse_vmc_output: no hamiltonian_data_opt_step_*.h5 found in %s; optimized_hamiltonian will be None.",
            work_dir,
        )

    # ── restart checkpoint ──
    # Search all output files for the last "Dump restart checkpoint" line.
    for fpath in reversed(output_files):
        text = _read_text(fpath)
        if text:
            for m_rc in _RE_RESTART_CHECKPOINT.finditer(text):
                result.restart_checkpoint = m_rc.group(1)
            if result.restart_checkpoint is not None:
                break

    # ── stderr tail ──
    stderr_candidates = [
        os.path.join(work_dir, "stderr"),
        os.path.join(work_dir, "err"),
    ]
    # Also check output files for stderr content
    for fpath in stderr_candidates:
        text = _read_text(fpath)
        if text:
            result.stderr_tail = _tail(text)
            break

    return result


# ── MCMC parser ───────────────────────────────────────────────────


def parse_mcmc_output(work_dir: str) -> MCMC_Diagnostic_Data:
    """Parse MCMC sampling output from *work_dir*.

    Extracts acceptance ratio, walker weights, and net time from stdout.
    Energy/error are extracted from the ``workflow_state.toml`` result
    section (populated by jqmc-tool post-processing) or from stdout
    if ``jqmc-tool mcmc compute-energy`` output is present.

    Parameters
    ----------
    work_dir : str
        Path to the MCMC working directory.

    Returns
    -------
    MCMC_Diagnostic_Data
        Structured parse result.
    """
    result = MCMC_Diagnostic_Data()

    if not os.path.isdir(work_dir):
        logger.warning("parse_mcmc_output: directory not found: %s", work_dir)
        return result

    output_files = _find_output_files(work_dir)

    # ── Run-level metadata (MPI, walkers, JAX) from first output file ──
    if output_files:
        first_text = _read_text(output_files[0])
        if first_text:
            _apply_run_metadata(result, _parse_run_metadata(first_text))

    # Parse the last output file (most recent run)
    last_text = None
    for fpath in reversed(output_files):
        text = _read_text(fpath)
        if text:
            last_text = text
            break

    if last_text:
        # Walker weight — take the last occurrence
        for m in _RE_WALKER_WEIGHT.finditer(last_text):
            result.avg_walker_weight = float(m.group(1))

        # Acceptance ratio — take the last occurrence
        for m in _RE_ACCEPTANCE.finditer(last_text):
            result.acceptance_ratio = float(m.group(1)) / 100.0

        # Total time — take the last occurrence
        for m in _RE_TOTAL_MCMC.finditer(last_text):
            result.total_time_sec = float(m.group(1))

        # Pre-compilation time — take the last occurrence
        for m in _RE_PRECOMP_MCMC.finditer(last_text):
            result.precompilation_time_sec = float(m.group(1))

        # Net time — take the last occurrence
        for m in _RE_NET_MCMC.finditer(last_text):
            result.net_time_sec = float(m.group(1))

        # Per-step timing breakdown (msec) — take last occurrence of each
        _mcmc_breakdown_patterns = [
            (_RE_TIME_MCMC_UPDATE, "mcmc_update"),
            (_RE_TIME_E_L, "e_L"),
            (_RE_TIME_DE_L_DR, "de_L_dR_dr"),
            (_RE_TIME_DLN_PSI_DR, "dln_Psi_dR_dr"),
            (_RE_TIME_DLN_PSI_DC, "dln_Psi_dc"),
            (_RE_TIME_DE_L_DC, "de_L_dc"),
            (_RE_TIME_MPI_MCMC, "mpi_barrier"),
            (_RE_TIME_MISC_MCMC, "misc"),
        ]
        for pattern, key in _mcmc_breakdown_patterns:
            for m in pattern.finditer(last_text):
                result.timing_breakdown[key] = float(m.group(1))

        # Restart checkpoint — take the last occurrence
        for m in _RE_RESTART_CHECKPOINT.finditer(last_text):
            result.restart_checkpoint = m.group(1)

    # ── hamiltonian_data_file from input TOML ──
    result.hamiltonian_data_file = _find_hamiltonian_h5(work_dir)

    # ── Energy & forces from workflow_state.toml result section ──
    state_path = os.path.join(work_dir, "workflow_state.toml")
    if os.path.isfile(state_path):
        try:
            state = toml.load(state_path)
            res = state.get("result", {})
            if "energy" in res:
                result.energy = float(res["energy"])
            if "energy_error" in res:
                result.energy_error = float(res["energy_error"])
            if "forces" in res and isinstance(res["forces"], list):
                result.atomic_forces = res["forces"]
        except Exception:
            pass

    # ── stderr tail ──
    for name in ("stderr", "err"):
        text = _read_text(os.path.join(work_dir, name))
        if text:
            result.stderr_tail = _tail(text)
            break

    return result


# ── LRDMC parser ──────────────────────────────────────────────────


def parse_lrdmc_output(work_dir: str) -> LRDMC_Diagnostic_Data:
    """Parse LRDMC calculation output from *work_dir*.

    Extracts survived walkers ratio, average number of projections,
    and net GFMC time from stdout.  Energy/error come from
    ``workflow_state.toml`` result section.

    Parameters
    ----------
    work_dir : str
        Path to the LRDMC working directory.

    Returns
    -------
    LRDMC_Diagnostic_Data
        Structured parse result.
    """
    result = LRDMC_Diagnostic_Data()

    if not os.path.isdir(work_dir):
        logger.warning("parse_lrdmc_output: directory not found: %s", work_dir)
        return result

    output_files = _find_output_files(work_dir)

    # ── Run-level metadata (MPI, walkers, JAX) from first output file ──
    if output_files:
        first_text = _read_text(output_files[0])
        if first_text:
            _apply_run_metadata(result, _parse_run_metadata(first_text))

    # Parse the last output file
    last_text = None
    for fpath in reversed(output_files):
        text = _read_text(fpath)
        if text:
            last_text = text
            break

    if last_text:
        # Survived walkers ratio — take the last occurrence
        for m in _RE_SURVIVED.finditer(last_text):
            result.survived_walkers_ratio = float(m.group(1)) / 100.0

        # Average number of projections — take the last occurrence
        for m in _RE_AVG_PROJECTIONS.finditer(last_text):
            result.avg_num_projections = float(m.group(1))

        # Total GFMC time — take the last occurrence
        for m in _RE_TOTAL_GFMC.finditer(last_text):
            result.total_time_sec = float(m.group(1))

        # Pre-compilation GFMC time — take the last occurrence
        for m in _RE_PRECOMP_GFMC.finditer(last_text):
            result.precompilation_time_sec = float(m.group(1))

        # Net GFMC time — take the last occurrence
        for m in _RE_NET_GFMC.finditer(last_text):
            result.net_time_sec = float(m.group(1))

        # Per-branching timing breakdown (msec) — take last occurrence of each
        _gfmc_breakdown_patterns = [
            (_RE_TIME_PROJECTION, "projection"),
            (_RE_TIME_OBSERVABLE, "observable"),
            (_RE_TIME_MPI_GFMC, "mpi_barrier"),
            (_RE_TIME_COLLECTION, "collection"),
            (_RE_TIME_RECONFIGURATION, "reconfiguration"),
            (_RE_TIME_E_L, "e_L"),
            (_RE_TIME_DE_L_DR, "de_L_dR_dr"),
            (_RE_TIME_DLN_PSI_DR, "dln_Psi_dR_dr"),
            (_RE_TIME_DLN_PSI_DC, "dln_Psi_dc"),
            (_RE_TIME_DE_L_DC, "de_L_dc"),
            (_RE_TIME_UPDATE_ESCF, "update_E_scf"),
            (_RE_TIME_MISC_GFMC, "misc"),
        ]
        for pattern, key in _gfmc_breakdown_patterns:
            for m in pattern.finditer(last_text):
                result.timing_breakdown[key] = float(m.group(1))

        # Restart checkpoint — take the last occurrence
        for m in _RE_RESTART_CHECKPOINT.finditer(last_text):
            result.restart_checkpoint = m.group(1)

    # ── hamiltonian_data_file from input TOML ──
    result.hamiltonian_data_file = _find_hamiltonian_h5(work_dir)

    # ── Energy & forces from workflow_state.toml result section ──
    state_path = os.path.join(work_dir, "workflow_state.toml")
    if os.path.isfile(state_path):
        try:
            state = toml.load(state_path)
            res = state.get("result", {})
            if "energy" in res:
                result.energy = float(res["energy"])
            if "energy_error" in res:
                result.energy_error = float(res["energy_error"])
            if "forces" in res and isinstance(res["forces"], list):
                result.atomic_forces = res["forces"]
        except Exception:
            pass

    # ── stderr tail ──
    for name in ("stderr", "err"):
        text = _read_text(os.path.join(work_dir, name))
        if text:
            result.stderr_tail = _tail(text)
            break

    return result


# ── LRDMC extrapolation parser ────────────────────────────────────


def parse_lrdmc_ext_output(work_dir: str) -> LRDMC_Ext_Diagnostic_Data:
    """Parse LRDMC extrapolation output from *work_dir*.

    Looks for ``For a -> 0 bohr: E = ...`` in the stdout of the
    extrapolation step.

    Parameters
    ----------
    work_dir : str
        Path to the LRDMC extrapolation working directory.

    Returns
    -------
    LRDMC_Ext_Diagnostic_Data
        Structured parse result.
    """
    result = LRDMC_Ext_Diagnostic_Data()

    if not os.path.isdir(work_dir):
        logger.warning("parse_lrdmc_ext_output: directory not found: %s", work_dir)
        return result

    output_files = _find_output_files(work_dir)

    for fpath in reversed(output_files):
        text = _read_text(fpath)
        if text is None:
            continue
        m = _RE_EXTRAP_ENERGY.search(text)
        if m:
            result.extrapolated_energy = float(m.group(1))
            result.extrapolated_energy_error = float(m.group(2))
            break

    # ── per-alat results from workflow_state.toml ──
    state_path = os.path.join(work_dir, "workflow_state.toml")
    if os.path.isfile(state_path):
        try:
            state = toml.load(state_path)
            res = state.get("result", {})
            if "per_alat_results" in res:
                result.per_alat_results = res["per_alat_results"]
        except Exception:
            pass

    # ── stderr tail ──
    for name in ("stderr", "err"):
        text = _read_text(os.path.join(work_dir, name))
        if text:
            result.stderr_tail = _tail(text)
            break

    return result


# ── Input parameters parser ──────────────────────────────────────


def _get_cli_defaults() -> dict:
    """Return a **deep copy** of ``cli_parameters`` from ``jqmc.jqmc_miscs``.

    Falls back to an empty dict if jqmc is not installed.
    """
    try:
        import copy

        from jqmc.jqmc_miscs import cli_parameters

        return copy.deepcopy(cli_parameters)
    except Exception:
        return {}


def parse_input_params(work_dir: str) -> Input_Parameters:
    """Extract key parameters from a workflow directory, **per input file**.

    For each ``[[jobs]]`` entry in ``workflow_state.toml``, the
    corresponding TOML input file is loaded and merged with the
    default values defined in ``jqmc.jqmc_miscs.cli_parameters``.
    The result is a list of per-input dicts stored in
    ``Input_Parameters.per_input``.

    ``actual_opt_steps`` is read from ``restart.h5`` when available
    (VMC only).

    Parameters
    ----------
    work_dir : str
        Path to the workflow working directory.

    Returns
    -------
    Input_Parameters
        Structured parameter data with per-input detail.
    """
    result = Input_Parameters()

    if not os.path.isdir(work_dir):
        logger.warning("parse_input_params: directory not found: %s", work_dir)
        return result

    # ── 1) restart.h5 → actual_opt_steps (VMC only) ──
    restart_path = os.path.join(work_dir, "restart.h5")
    if os.path.isfile(restart_path):
        try:
            import h5py

            with h5py.File(restart_path, "r") as f:
                if "rank_0/driver_config" in f:
                    dc_attrs = dict(f["rank_0/driver_config"].attrs)
                    i_opt = dc_attrs.get("i_opt")
                    if i_opt is not None:
                        result.actual_opt_steps = int(i_opt)
        except Exception as exc:
            logger.warning(
                "parse_input_params: cannot read restart.h5 in %s: %s",
                work_dir,
                exc,
            )

    # ── 2) workflow_state.toml → [[jobs]] ──
    state_path = os.path.join(work_dir, "workflow_state.toml")
    jobs: list = []
    if os.path.isfile(state_path):
        try:
            state = toml.load(state_path)
            jobs = state.get("jobs", [])
        except Exception:
            pass

    # ── 3) Per-input parameter extraction ──
    defaults = _get_cli_defaults()

    for job_rec in jobs:
        input_name = job_rec.get("input_file", "")
        output_name = job_rec.get("output_file", "")
        entry: dict = {
            "input_file": input_name,
            "output_file": output_name,
        }

        toml_path = os.path.join(work_dir, input_name) if input_name else ""
        if toml_path and os.path.isfile(toml_path):
            try:
                raw = toml.load(toml_path)
            except Exception:
                raw = {}
        else:
            raw = {}

        # -- [control] section (merge defaults) --
        ctrl_defaults = {k: v for k, v in defaults.get("control", {}).items()}
        ctrl_user = raw.get("control", {})
        ctrl_merged = {**ctrl_defaults, **ctrl_user}
        entry["control"] = ctrl_merged

        job_type = ctrl_merged.get("job_type", "")
        entry["job_type"] = job_type if job_type else ""

        # -- job-type section (merge defaults) --
        if job_type and job_type in defaults:
            jt_defaults = {k: v for k, v in defaults[job_type].items()}
            jt_user = raw.get(job_type, {})
            jt_merged = {**jt_defaults, **jt_user}
            entry[job_type] = jt_merged
        elif job_type:
            # No defaults known — just use raw values
            entry[job_type] = raw.get(job_type, {})

        result.per_input.append(entry)

    return result
