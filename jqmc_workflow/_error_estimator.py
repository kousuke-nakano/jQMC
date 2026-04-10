"""Step estimation utilities for the target error bar feature.

Given a short pilot run and a desired target statistical error, estimates
the number of measurement steps required for a production run.

The central-limit-theorem scaling σ ∝ 1/√N is used:

    N_required = ⌈ N_pilot × (σ_pilot / σ_target)² ⌉
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

import math
import os
import re
from logging import getLogger

logger = getLogger("jqmc-workflow").getChild(__name__)


def estimate_required_steps(
    pilot_steps: int,
    pilot_error: float,
    target_error: float,
    walker_ratio: float = 1.0,
    min_steps: int = 0,
) -> int:
    """Estimate total measurement steps to achieve *target_error*.

    Uses the CLT scaling
    $\\sigma \\propto 1/\\sqrt{N \\times W}$
    where *N* is the number of steps and *W* is the total number of
    walkers.  When the pilot and production runs use different queue
    configurations (different MPI process counts), *walker_ratio*
    accounts for the difference in effective sample rate:

    .. math::

        N_{\\text{prod}} = N_{\\text{pilot}}
            \\times (\\sigma_{\\text{pilot}} / \\sigma_{\\text{target}})^2
            \\times W_{\\text{pilot}} / W_{\\text{prod}}

    Parameters
    ----------
    pilot_steps : int
        Number of measurement steps used in the pilot run.
    pilot_error : float
        Statistical error (standard error) obtained from the pilot run.
    target_error : float
        Desired statistical error for the production run.
    walker_ratio : float
        Ratio of effective walker counts: ``pilot_walkers / prod_walkers``.
        When walkers-per-MPI is constant this equals
        ``pilot_num_mpi / prod_num_mpi``.
        Default 1.0 (same queue for pilot and production).
    min_steps : int
        Minimum number of steps to return (e.g. warmup + bin_blocks).
        Default 0 (no minimum).

    Returns
    -------
    int
        Estimated number of steps for the production run.
    """
    if target_error <= 0:
        raise ValueError(f"target_error must be positive, got {target_error}")
    if pilot_error <= 0:
        logger.warning(f"pilot_error={pilot_error} is non-positive; returning pilot_steps={pilot_steps} as fallback.")
        return max(pilot_steps, min_steps)

    ratio_sq = (pilot_error / target_error) ** 2
    n_required = max(int(pilot_steps * ratio_sq * walker_ratio), min_steps)

    logger.info(
        f"Step estimation (sigma ~ 1/sqrt(N*W)):\n"
        f"  pilot_steps    = {pilot_steps}\n"
        f"  pilot_error    = {pilot_error:.6g} Ha\n"
        f"  target_error   = {target_error:.6g} Ha\n"
        f"  (sig_p/sig_t)^2 = ({pilot_error:.6g}/{target_error:.6g})^2 = {ratio_sq:.2f}\n"
        f"  walker_ratio   = {walker_ratio:.4g}\n"
        f"  min_steps      = {min_steps}\n"
        f"  N_required     = max(int({pilot_steps} * {ratio_sq:.2f} * {walker_ratio:.4g}), {min_steps}) = {n_required}"
    )
    return n_required


def estimate_additional_steps(
    accumulated_steps: int,
    current_error: float,
    target_error: float,
    min_additional: int = 100,
) -> int:
    """Estimate **additional** measurement steps when data is accumulated.

    After *accumulated_steps* have already been collected (with the
    checkpoint holding all statistics), compute how many more steps
    are needed to bring the error from *current_error* down to
    *target_error*.

    Uses σ ∝ 1/√N:

        N_total = ⌈ accumulated_steps × (current_error / target_error)² ⌉
        additional = N_total − accumulated_steps

    Parameters
    ----------
    accumulated_steps : int
        Total measurement steps already in the checkpoint.
    current_error : float
        Statistical error after *accumulated_steps*.
    target_error : float
        Desired statistical error.
    min_additional : int
        Floor on additional steps to avoid trivially short runs.

    Returns
    -------
    int
        Number of *additional* steps to run.
    """
    if target_error <= 0:
        raise ValueError(f"target_error must be positive, got {target_error}")
    if current_error <= target_error:
        return 0

    ratio_sq = (current_error / target_error) ** 2
    n_total = math.ceil(accumulated_steps * ratio_sq)
    additional = max(n_total - accumulated_steps, min_additional)

    logger.info(
        f"Additional-step estimation (accumulated, sigma ~ 1/sqrt(N)):\n"
        f"  accumulated    = {accumulated_steps}\n"
        f"  current error  = {current_error:.6g} Ha\n"
        f"  target error   = {target_error:.6g} Ha\n"
        f"  N_total needed = {n_total}\n"
        f"  additional     = {additional}"
    )
    return additional


def suffixed_name(filename: str, index: int) -> str:
    """Insert an integer suffix before the file extension.

    Examples
    --------
    >>> suffixed_name("input.toml", 0)
    'input_0.toml'
    >>> suffixed_name("out.o", 2)
    'out_2.o'
    """
    base, ext = os.path.splitext(filename)
    return f"{base}_{index}{ext}"


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Examples
    --------
    >>> _format_duration(90)
    '1m 30s'
    >>> _format_duration(3661)
    '1h 1m 1s'
    >>> _format_duration(86400)
    '1d 0h 0m'
    """
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        m = int(minutes)
        s = int(seconds - m * 60)
        return f"{m}m {s}s"
    hours = minutes / 60
    if hours < 24:
        h = int(hours)
        m = int(minutes - h * 60)
        s = int(seconds - h * 3600 - m * 60)
        return f"{h}h {m}m {s}s"
    days = int(hours / 24)
    h = int(hours - days * 24)
    m = int(minutes - days * 1440 - h * 60)
    return f"{days}d {h}h {m}m"


# ── Patterns for "Net" times in jQMC output ──────────────────

# LRDMC: "Net GFMC time without pre-compilations =  2832.326 sec."
_RE_NET_GFMC = re.compile(
    r"Net GFMC time without pre-compilations\s*=\s*"
    r"([\d.]+(?:[eE][+-]?\d+)?)\s*sec",
)
# MCMC / VMC: "Net total time for MCMC = 6.22 sec."
_RE_NET_MCMC = re.compile(
    r"Net total time for MCMC\s*=\s*"
    r"([\d.]+(?:[eE][+-]?\d+)?)\s*sec",
)


def parse_net_time(output_file: str) -> float | None:
    """Parse the net computation time from a jQMC output file.

    Searches for::

        Net GFMC time without pre-compilations = <value> sec.   (LRDMC)
        Net total time for MCMC = <value> sec.                  (MCMC/VMC)

    Parameters
    ----------
    output_file : str
        Path to the jQMC stdout/stderr output file.

    Returns
    -------
    float or None
        Net time in seconds, or *None* if the pattern is not found.
    """
    if not os.path.isfile(output_file):
        logger.debug(f"parse_net_time: file not found: {output_file}")
        return None

    try:
        with open(output_file, "r", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        logger.debug(f"parse_net_time: cannot read {output_file}: {exc}")
        return None

    # Try LRDMC pattern first, then MCMC/VMC.
    # Use findall + sum so that VMC outputs (which print the line
    # once per optimization step) return the *total* net time.
    for pat in (_RE_NET_GFMC, _RE_NET_MCMC):
        matches = pat.findall(text)
        if matches:
            return sum(float(v) for v in matches)
    return None
