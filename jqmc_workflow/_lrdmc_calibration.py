"""LRDMC calibration utilities — survived walkers ratio.

Provides helper functions for determining the optimal
``num_mcmc_per_measurement`` based on a target survived-walkers ratio.

The calibration procedure is:

1. Run three short LRDMC calculations with
   ``num_mcmc_per_measurement = Ne*2, Ne*4, Ne*6``
   (where *Ne* is the total number of electrons).
2. Parse the ``Survived walkers ratio`` from each output file.
3. Fit a quadratic ``f(x) = a*x² + b*x + c`` to the three points
   and solve for the ``num_mcmc_per_measurement`` that gives the
   target survived-walkers ratio (default 97 %).
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
from typing import List, Optional, Tuple

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


# ── Quadratic fitting ────────────────────────────────────────────


def fit_num_mcmc_per_measurement(
    x_values: List[int],
    y_values: List[float],
    target_ratio: float,
) -> int:
    r"""Determine the optimal ``num_mcmc_per_measurement`` by quadratic fit.

    Given three (or more) data points
    ``(num_mcmc_per_measurement, survived_walkers_ratio)`` the function
    fits a quadratic :math:`f(x) = a x^2 + b x + c` and solves for the
    *x* at which :math:`f(x) = \text{target\_ratio}`.

    Parameters
    ----------
    x_values : list[int]
        ``num_mcmc_per_measurement`` values used in calibration runs.
    y_values : list[float]
        Corresponding survived-walkers ratios (fractions, 0.0–1.0).
    target_ratio : float
        Target survived-walkers ratio (e.g. 0.97).

    Returns
    -------
    int
        Optimal ``num_mcmc_per_measurement`` (rounded up to the nearest
        even integer, minimum 2).

    Raises
    ------
    RuntimeError
        If the quadratic has no real root above 0 for the target.
    """
    if len(x_values) < 3 or len(y_values) < 3:
        raise ValueError(f"Need at least 3 data points, got {len(x_values)}")

    # -- Fit quadratic via least-squares (no numpy dependency) ----
    # Using the normal equations for y = a*x^2 + b*x + c
    n = len(x_values)
    sx = sum(x_values)
    sx2 = sum(xi**2 for xi in x_values)
    sx3 = sum(xi**3 for xi in x_values)
    sx4 = sum(xi**4 for xi in x_values)
    sy = sum(y_values)
    sxy = sum(xi * yi for xi, yi in zip(x_values, y_values))
    sx2y = sum(xi**2 * yi for xi, yi in zip(x_values, y_values))

    # Solve 3×3 system:
    #   [ sx4  sx3  sx2 ] [a]   [sx2y]
    #   [ sx3  sx2  sx  ] [b] = [sxy ]
    #   [ sx2  sx   n   ] [c]   [sy  ]
    a, b, c = _solve_3x3(
        [[sx4, sx3, sx2], [sx3, sx2, sx], [sx2, sx, n]],
        [sx2y, sxy, sy],
    )

    logger.info(f"Quadratic fit: f(x) = {a:.6g}*x^2 + {b:.6g}*x + {c:.6g}")
    for xi, yi in zip(x_values, y_values):
        fitted = a * xi**2 + b * xi + c
        logger.info(f"  nmpm={xi:>6d}: measured={yi:.4f}, fitted={fitted:.4f}")

    # Solve a*x^2 + b*x + (c - target_ratio) = 0
    c_shifted = c - target_ratio
    discriminant = b**2 - 4.0 * a * c_shifted

    if discriminant < 0:
        raise RuntimeError(
            f"Quadratic fit has no real root for target_ratio={target_ratio:.4f}. "
            f"Coefficients: a={a:.6g}, b={b:.6g}, c={c:.6g}, "
            f"discriminant={discriminant:.6g}"
        )

    sqrt_d = math.sqrt(discriminant)
    if abs(a) < 1e-15:
        # Degenerate to linear: b*x + c_shifted = 0
        if abs(b) < 1e-15:
            raise RuntimeError("Degenerate fit: both a and b are ~0.")
        x_opt = -c_shifted / b
    else:
        # Two roots — pick the positive one (or larger positive one)
        x1 = (-b + sqrt_d) / (2.0 * a)
        x2 = (-b - sqrt_d) / (2.0 * a)
        candidates = [x for x in (x1, x2) if x > 0]
        if not candidates:
            raise RuntimeError(f"No positive root found. Roots: {x1:.2f}, {x2:.2f}")
        # Pick the smaller positive root (physically meaningful)
        x_opt = min(candidates)

    # Round up to nearest even integer, minimum 2
    result = max(2, int(math.ceil(x_opt)))
    if result % 2 != 0:
        result += 1

    logger.info(f"Optimal num_mcmc_per_measurement for target ratio {target_ratio:.4f}: raw={x_opt:.2f} -> {result}")
    return result


def scale_num_mcmc_per_measurement(
    nmpm_ref: int,
    alat_ref: float,
    alat: float,
) -> int:
    r"""Scale ``num_mcmc_per_measurement`` to a different lattice spacing.

    The optimal ``num_mcmc_per_measurement`` is approximately proportional
    to :math:`1/a^2`.  Given a reference value calibrated at ``alat_ref``,
    the value at a different ``alat`` is:

    .. math::

        \text{nmpm}(\text{alat}) = \text{nmpm\_ref}
            \times \left(\frac{\text{alat\_ref}}{\text{alat}}\right)^{2}

    Parameters
    ----------
    nmpm_ref : int
        Calibrated ``num_mcmc_per_measurement`` at ``alat_ref``.
    alat_ref : float
        Reference lattice spacing (bohr).
    alat : float
        Target lattice spacing (bohr).

    Returns
    -------
    int
        Scaled ``num_mcmc_per_measurement`` (rounded up to nearest even
        integer, minimum 2).
    """
    raw = nmpm_ref * (alat_ref / alat) ** 2
    result = max(2, int(math.ceil(raw)))
    if result % 2 != 0:
        result += 1
    return result


# ── Internal: simple 3×3 linear solver (no numpy required) ──────


def _solve_3x3(
    A: List[List[float]],
    b: List[float],
) -> Tuple[float, float, float]:
    """Solve a 3×3 linear system Ax = b via Cramer's rule.

    Parameters
    ----------
    A : list[list[float]]
        3×3 coefficient matrix.
    b : list[float]
        3-element right-hand side.

    Returns
    -------
    tuple[float, float, float]
        Solution vector.
    """

    def det3(m):
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    det_A = det3(A)
    if abs(det_A) < 1e-30:
        raise RuntimeError("Singular matrix in 3×3 solver.")

    def replace_col(col_idx):
        M = [row[:] for row in A]
        for i in range(3):
            M[i][col_idx] = b[i]
        return det3(M)

    return (
        replace_col(0) / det_A,
        replace_col(1) / det_A,
        replace_col(2) / det_A,
    )
