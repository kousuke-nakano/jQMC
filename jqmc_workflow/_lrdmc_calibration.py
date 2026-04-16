"""LRDMC calibration utilities — survived walkers ratio.

Provides helper functions for determining the optimal
``num_projection_per_measurement`` based on a target survived-walkers ratio.

The calibration procedure is:

1. Run short LRDMC calculations with varying
   ``num_projection_per_measurement`` values (e.g. ``Ne*2, Ne*4, Ne*6``
   where *Ne* is the total number of electrons).
2. Parse the ``Survived walkers ratio`` from each output file.
3. Fit a linear ``f(x) = a*x + b`` via least squares and solve for
   the ``num_projection_per_measurement`` that gives the target
   survived-walkers ratio (default 97 %).
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
import re
from logging import getLogger
from typing import List, Optional

import h5py

logger = getLogger("jqmc-workflow").getChild(__name__)


# ── HDF5 electron count ──────────────────────────────────────────


def get_num_electrons(hamiltonian_file: str) -> int:
    """Read the total number of electrons from a hamiltonian HDF5 file.

    Parameters
    ----------
    hamiltonian_file : str
        Path to ``hamiltonian_data.h5``.

    Returns
    -------
    int
        Total electron count ``num_electron_up + num_electron_dn``.

    Raises
    ------
    RuntimeError
        If the electron counts cannot be found in the file.
    """
    try:
        with h5py.File(hamiltonian_file, "r") as f:
            geminal = f["wavefunction_data/geminal_data"]
            n_up = int(geminal.attrs["num_electron_up"])
            n_dn = int(geminal.attrs["num_electron_dn"])
        return n_up + n_dn
    except Exception as e:
        raise RuntimeError(f"Cannot read electron counts from {hamiltonian_file}: {e}") from e


# ── Survived walkers ratio parsing ───────────────────────────────

_SURVIVED_PATTERN = re.compile(r"Survived walkers ratio\s*=\s*(\d+\.?\d*)\s*%")


def parse_survived_walkers_ratio(output_file: str) -> Optional[float]:
    """Parse the survived walkers ratio from an LRDMC output file.

    Searches for the line
    ``Survived walkers ratio = <value> %``
    and returns the **last** occurrence as a fraction (0.0–1.0).

    Parameters
    ----------
    output_file : str
        Path to the jqmc stdout file.

    Returns
    -------
    float or None
        Survived walkers ratio as a fraction, or *None* if not found.
    """
    last_value = None
    try:
        with open(output_file, "r") as f:
            for line in f:
                m = _SURVIVED_PATTERN.search(line)
                if m:
                    last_value = float(m.group(1)) / 100.0
    except Exception:
        return None
    return last_value


# ── Linear fitting ───────────────────────────────────────────────


def fit_num_projection_per_measurement(
    x_values: List[int],
    y_values: List[float],
    target_ratio: float,
) -> int:
    r"""Determine the optimal ``num_projection_per_measurement`` by linear fit.

    Given two or more data points
    ``(num_projection_per_measurement, survived_walkers_ratio)`` the function
    fits a linear model :math:`f(x) = a x + b` via least squares and
    solves for the *x* at which :math:`f(x) = \text{target\_ratio}`.

    Parameters
    ----------
    x_values : list[int]
        ``num_projection_per_measurement`` values used in calibration runs.
    y_values : list[float]
        Corresponding survived-walkers ratios (fractions, 0.0–1.0).
    target_ratio : float
        Target survived-walkers ratio (e.g. 0.97).

    Returns
    -------
    int
        Optimal ``num_projection_per_measurement`` (rounded up to the nearest
        even integer, minimum 2).

    Raises
    ------
    RuntimeError
        If the linear fit cannot determine a positive root.
    """
    if len(x_values) < 2 or len(y_values) < 2:
        raise ValueError(f"Need at least 2 data points, got {len(x_values)}")

    # -- Fit linear y = a*x + b via least-squares (no numpy) -----
    n = len(x_values)
    sx = sum(x_values)
    sx2 = sum(xi**2 for xi in x_values)
    sy = sum(y_values)
    sxy = sum(xi * yi for xi, yi in zip(x_values, y_values))

    denom = n * sx2 - sx * sx
    if abs(denom) < 1e-30:
        raise RuntimeError("Degenerate fit: all x values are identical.")

    a = (n * sxy - sx * sy) / denom
    b = (sy * sx2 - sx * sxy) / denom

    logger.info(f"Linear fit: f(x) = {a:.6g}*x + {b:.6g}")
    for xi, yi in zip(x_values, y_values):
        fitted = a * xi + b
        logger.info(f"  nmpm={xi:>6d}: measured={yi:.4f}, fitted={fitted:.4f}")

    # Solve a*x + b = target_ratio  =>  x = (target_ratio - b) / a
    if abs(a) < 1e-15:
        raise RuntimeError(f"Linear fit slope is ~0 (a={a:.6g}). Cannot solve for target_ratio={target_ratio:.4f}.")

    x_opt = (target_ratio - b) / a
    if x_opt <= 0:
        raise RuntimeError(
            f"Linear fit gives non-positive root x={x_opt:.2f} "
            f"for target_ratio={target_ratio:.4f}. "
            f"Coefficients: a={a:.6g}, b={b:.6g}"
        )

    # Round up to nearest even integer, minimum 2
    result = max(2, int(math.ceil(x_opt)))
    if result % 2 != 0:
        result += 1

    logger.info(f"Optimal num_projection_per_measurement for target ratio {target_ratio:.4f}: raw={x_opt:.2f} -> {result}")
    return result


def scale_num_projection_per_measurement(
    nmpm_ref: int,
    alat_ref: float,
    alat: float,
) -> int:
    r"""Scale ``num_projection_per_measurement`` to a different lattice spacing.

    The optimal ``num_projection_per_measurement`` is approximately proportional
    to :math:`1/a^2`.  Given a reference value calibrated at ``alat_ref``,
    the value at a different ``alat`` is:

    .. math::

        \text{nmpm}(\text{alat}) = \text{nmpm\_ref}
            \times \left(\frac{\text{alat\_ref}}{\text{alat}}\right)^{2}

    Parameters
    ----------
    nmpm_ref : int
        Calibrated ``num_projection_per_measurement`` at ``alat_ref``.
    alat_ref : float
        Reference lattice spacing (bohr).
    alat : float
        Target lattice spacing (bohr).

    Returns
    -------
    int
        Scaled ``num_projection_per_measurement`` (rounded up to nearest even
        integer, minimum 2).
    """
    raw = nmpm_ref * (alat_ref / alat) ** 2
    result = max(2, int(math.ceil(raw)))
    if result % 2 != 0:
        result += 1
    return result
