"""Mixed precision configuration for jQMC.

Every computational function declares its Precision Zone and explicitly
specifies dtype for all variables it creates or consumes.  This design
does NOT rely on JAX's implicit dtype propagation, ensuring robustness
against future changes in JAX's type promotion semantics.

Users choose one of two modes:

- ``"full"``  (default) — all zones float64 (backward compatible).
- ``"mixed"`` — recommended mixed precision; low-risk zones become
  float32 while numerically sensitive zones stay float64.

Individual zone assignments are **not** user-configurable; they are
defined in ``_FULL_PRECISION`` and ``_MIXED_PRECISION`` below.
Developers who need to tweak per-zone dtypes should edit those dicts
directly in this file.

Precision Zones
---------------

==============  ============================  =========  ========  ============
Zone            Components                    Default    Mixed     float32 risk
==============  ============================  =========  ========  ============
``orb_eval``    AO/MO forward evaluation      float64    float32   low
``jastrow``     Jastrow factor (J1/J2/J3)     float64    float32   low
``geminal``     Geminal matrix elements        float64    float64   high
``determinant`` log-det, SVD, AS reg.          float64    float64   high
``coulomb``     Coulomb + ECP potential        float64    float32   low-medium
``kinetic``     Kinetic energy + AO/MO derivs  float64    float64   high
``mcmc``        MCMC sampling                  float64    float64   high
``gfmc``        GFMC propagation               float64    float64   high
``optimization``SR matrix, parameter updates   float64    float64   high
``io``          I/O, structure data            float64    float64   low-medium
==============  ============================  =========  ========  ============

File-to-zone mapping
--------------------

- ``atomic_orbital.py``: ``orb_eval`` (forward), ``kinetic`` (grad/laplacian)
- ``molecular_orbital.py``: ``orb_eval`` (forward), ``kinetic`` (grad/laplacian)
- ``jastrow_factor.py``: ``jastrow`` (forward), ``kinetic`` (grad/laplacian),
  ``mcmc`` (ratio/update)
- ``determinant.py``: ``geminal`` (matrix elements), ``determinant`` (log-det/SVD),
  ``kinetic`` (grad/laplacian)
- ``coulomb_potential.py``: ``coulomb``
- ``wavefunction.py``: ``kinetic`` + zone-boundary casts
- ``hamiltonians.py``: ``kinetic`` (zone-boundary aggregation of T + V)
- ``jqmc_mcmc.py``: ``mcmc`` (sampling), ``optimization`` (SR/LM)
- ``jqmc_gfmc.py``: ``gfmc``
- ``structure.py``, ``trexio_wrapper.py``, ``jqmc_tool.py``: ``io``
- ``_jqmc_utility.py``: ``io``
- ``swct.py``: ``kinetic``

Usage::

    from jqmc._precision import get_dtype

    def compute_AOs(aos_data, r_carts):
        dtype = get_dtype("orb_eval")
        r_carts = jnp.asarray(r_carts, dtype=dtype)
        ...
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

import logging

import jax.numpy as jnp

logger = logging.getLogger(__name__)

# --- mode="full" (all float64, backward compatible) ---
_FULL_PRECISION: dict[str, str] = {
    "orb_eval": "float64",  # AO/MO forward evaluation
    "jastrow": "float64",  # Jastrow factor
    "geminal": "float64",  # Geminal matrix elements
    "determinant": "float64",  # log-det, SVD, AS regularization
    "coulomb": "float64",  # Coulomb + ECP potential
    "kinetic": "float64",  # Kinetic energy + AO/MO derivatives
    "mcmc": "float64",  # MCMC sampling (proposal, SM update, accept/reject, accumulation)
    "gfmc": "float64",  # GFMC propagation
    "optimization": "float64",  # SR matrix, parameter updates
    "io": "float64",  # I/O, structure data
}

# --- mode="mixed" (recommended mixed precision) ---
# float32 risk:
#   orb_eval    - low: smooth Gaussian basis + linear combination.
#                 (Heavy AO eval stays in fp32; compute_MOs upcasts the small
#                  matmul to the determinant zone, see molecular_orbital.py.)
#   jastrow     - low: smooth correlation function, pre-exp value
#   geminal     - HIGH: this matrix is the input to LU/det; even ε≈1e-7 on
#                 entries amplifies into log|det| errors of O(1) for ~32x32
#                 systems with non-trivial condition numbers
#                 (see bug/fp32 diagnostics).  Kept in fp64.
#   coulomb     - low-medium: sum of 1/r + ECP spherical quadrature
#   determinant - high: log(det) cancellation, SVD 1/s, eigenvalue ops
#   kinetic     - high: second derivative of ln|Psi|, cancellation-sensitive
#   mcmc        - high: SM inverse error accumulation, acceptance ratio, statistics
#   gfmc        - high: weighted branching/pruning, population collapse in float32
#   optimization- high: S^{-1}F linear system, ill-conditioned matrix
#   io          - low-medium: file I/O + nuclear coordinates
_MIXED_PRECISION: dict[str, str] = {
    "orb_eval": "float32",  # low risk (heavy kernel only)
    "jastrow": "float32",  # low risk
    "geminal": "float64",  # high risk: feeds LU/det
    "determinant": "float64",  # high risk
    "coulomb": "float32",  # low-medium risk
    "kinetic": "float64",  # high risk
    "mcmc": "float64",  # high risk
    "gfmc": "float64",  # high risk
    "optimization": "float64",  # high risk
    "io": "float64",  # low-medium risk
}

ALL_ZONES = frozenset(_FULL_PRECISION.keys())

# Runtime zone -> dtype mapping
_zone_dtypes: dict[str, type] = {}


def _str_to_dtype(s: str) -> type:
    """Convert a string dtype name to the corresponding JAX/NumPy dtype type.

    Args:
        s: Either ``"float32"`` or ``"float64"``.

    Returns:
        ``jnp.float32`` or ``jnp.float64``.

    Raises:
        ValueError: If *s* is not one of the two accepted strings.
    """
    if s == "float32":
        return jnp.float32
    elif s == "float64":
        return jnp.float64
    else:
        raise ValueError(f"Invalid dtype '{s}'. Must be 'float32' or 'float64'.")


def configure(mode: str = "full") -> None:
    """Activate a precision mode.

    Args:
        mode: ``"full"`` (default, all float64) or ``"mixed"``
            (recommended mixed precision).

    Raises:
        ValueError: If *mode* is not ``"full"`` or ``"mixed"``.
    """
    _zone_dtypes.clear()

    if mode == "full":
        base = _FULL_PRECISION
    elif mode == "mixed":
        base = _MIXED_PRECISION
    else:
        raise ValueError(f"Unknown precision mode '{mode}'. Must be 'full' or 'mixed'.")

    for zone, dtype_str in base.items():
        _zone_dtypes[zone] = _str_to_dtype(dtype_str)

    logger.info(summary())


def _set_zone(zone: str, dtype_str: str) -> None:
    """Override a single zone's dtype at runtime (developer use only).

    Must be called **after** :func:`configure`.  This is intentionally
    private — normal users select ``"full"`` or ``"mixed"`` mode and the
    per-zone mapping is determined by ``_FULL_PRECISION`` /
    ``_MIXED_PRECISION``.

    Args:
        zone: Precision Zone name.
        dtype_str: ``"float32"`` or ``"float64"``.
    """
    if zone not in ALL_ZONES:
        raise ValueError(f"Unknown precision zone '{zone}'. Available zones: {sorted(ALL_ZONES)}")
    _zone_dtypes[zone] = _str_to_dtype(dtype_str)


def get_dtype(zone: str) -> type:
    """Return the dtype for a Precision Zone.

    When :func:`configure` has not been called, ``jnp.float64`` is returned
    for any zone name (backward compatible).  After :func:`configure` has
    been called, an unknown *zone* raises :class:`ValueError`.

    Args:
        zone: Precision Zone name (e.g. ``"orb_eval"``, ``"kinetic"``).

    Returns:
        ``jnp.float32`` or ``jnp.float64``.
    """
    if _zone_dtypes:
        # configure() has been called: validate zone name
        if zone not in ALL_ZONES:
            raise ValueError(f"Unknown precision zone '{zone}'. Available zones: {sorted(ALL_ZONES)}")
    return _zone_dtypes.get(zone, jnp.float64)


def is_mixed_precision_enabled() -> bool:
    """Return ``True`` if at least one zone is set to float32."""
    return any(d == jnp.float32 for d in _zone_dtypes.values())


def get_tolerance(zone: str, level: str = "strict") -> tuple[float, float]:
    """Return test tolerances ``(atol, rtol)`` scaled by the zone's dtype.

    Args:
        zone: Precision Zone name.
        level: ``"strict"`` or ``"loose"``.

    Returns:
        ``(atol, rtol)`` tuple appropriate for the zone's current dtype.
    """
    from jqmc._setting import _TOLERANCE

    dtype_key = "float32" if get_dtype(zone) == jnp.float32 else "float64"
    return _TOLERANCE[level][dtype_key]


def get_tolerance_min(zones, level: str = "strict") -> tuple[float, float]:
    """Return ``(atol, rtol)`` loose enough for the lowest-precision zone.

    Use for comparing two exact computations whose path crosses
    multiple zones; the achievable agreement is bounded by the
    weakest (largest tolerance) zone on the path.

    Args:
        zones: Iterable of Precision Zone names.
        level: ``"strict"`` or ``"loose"``.

    Returns:
        ``(atol, rtol)`` tuple using the maximum of each component.
    """
    atols, rtols = zip(*(get_tolerance(z, level) for z in zones))
    return max(atols), max(rtols)


def summary() -> str:
    """Return a human-readable summary of the current precision configuration."""
    lines = ["Precision configuration:"]
    for zone in sorted(ALL_ZONES):
        dtype = get_dtype(zone)
        tag = "float32" if dtype == jnp.float32 else "float64"
        lines.append(f"  {zone}: {tag}")
    return "\n".join(lines)
