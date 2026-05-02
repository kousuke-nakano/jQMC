r"""Mixed precision configuration for jQMC.

================================================================================
DESIGN PRINCIPLES (READ THIS FIRST)
================================================================================

The mixed-precision implementation rests on **three** principles. Principle 3
is the most important in practice; almost every precision bug we have seen is
a violation of 3a or 3b.

------------------------------------------------------------
Principle 1 — One Precision Zone is owned by exactly one module.
------------------------------------------------------------
A zone (e.g. ``ao_eval``, ``coulomb``) is *defined and consumed* in a single
module. The mapping zone ↔ owning module is one-to-one and is documented in
the table below (and enforced by convention in ``_FULL_PRECISION`` /
``_MIXED_PRECISION``).

------------------------------------------------------------
Principle 2 — A module may own multiple Precision Zones.
------------------------------------------------------------
Different code paths in the same module legitimately need different precisions
(e.g. ``ao_eval`` vs ``ao_grad_lap``, or ``det_eval`` vs ``det_ratio``). Each
zone is named for its *purpose*, not for its dtype.

------------------------------------------------------------
Principle 3 — Cast responsibility lies with the function that does
              arithmetic on the value, never with passthrough wrappers.
------------------------------------------------------------

Definition. *arithmetic* means consuming a value as an operand of a numerical
operation (``+ - * /``, ``jnp.linalg.norm``, ``jnp.dot``, ``@``, ``jnp.exp``,
…) **or** as an input to ``jax.grad`` / ``jax.jacrev`` / ``jax.hessian``.
Operations that do *not* count as arithmetic and therefore do *not* trigger a
cast: ``len(x)``, ``x.shape``, ``x[i]`` (index lookup), the *target* of
``.at[i].set(y)``, and forwarding ``x`` as an argument to another function.

Principle 3a — Arguments are frozen.
    Function arguments are treated as **frozen**, in the same sense as the
    attributes of a ``@dataclass(frozen=True)``: the name introduced by the
    parameter list **must not be rebound** for the entire body of the
    function. In particular, ``arg = jnp.asarray(arg, dtype=...)`` at the
    top of a function is forbidden — it silently coerces the argument for
    every later use, including forwarding.

    Consequences (not extra rules — direct corollaries of "frozen"):
      * Forwarding neutrality. A value forwarded to a callee transits in
        the dtype it was received in; the callee is responsible for casting
        it to *its* own zone (Principle 3b).
      * Cast at the use site. When the function consumes ``arg`` as an
        operand of its own arithmetic, the cast appears **inside the
        expression** (``arg.astype(dtype)``). Do *not* preemptively
        introduce a local alias just to hold the cast — only do so when
        the cast result is reused multiple times, in which case introduce
        a *new* local variable with a different name (e.g.
        ``arg_local = arg.astype(dtype)``). The original ``arg`` always
        remains frozen.

Principle 3b — Local cast at the point of arithmetic.
    A function casts a value to its own zone's dtype **immediately before**
    consuming it as an operand. Inputs and outputs of the function's
    arithmetic both live in its zone. Intermediate computations may use a
    higher precision when needed for numerical reasons (the canonical case
    being ``r - R``: reconstruct the difference in the **dtype the value
    was received in** — i.e. the precision chosen by the upper layer —
    to avoid catastrophic cancellation, then down-cast the result back to
    the function's own zone). In jQMC the upstream (mcmc walker state) is
    always fp64, so in practice the reconstruction happens in fp64; the
    *principle*, however, is "use the caller-supplied precision," not
    "hardcode fp64." Concretely: do not write ``jnp.float64`` as a
    literal, and avoid pinning the reconstruction to a specific zone
    name when the incoming value's own dtype already carries the right
    precision.

Worked example (the ECP → AO bug this design prevents)::

    # WRONG: rebinding `r_carts` at the top of compute_coulomb forwards a
    # fp32-truncated array to compute_AOs, even though `ao_eval` is fp64.
    def compute_coulomb(r_carts, R_carts):
        dtype_jnp = get_dtype_jnp("coulomb")
        r_carts = jnp.asarray(r_carts, dtype=dtype_jnp)  # 3a violation
        R_carts = jnp.asarray(R_carts, dtype=dtype_jnp)
        ao = compute_AOs(..., r_carts, R_carts)          # downstream sees fp32
        diff = r_carts - R_carts
        ...

    # RIGHT: forwarding stays in the caller's dtype; the local arithmetic
    # reconstructs the difference in the dtype the values were received in
    # (the upper-layer precision — fp64 in jQMC because mcmc walker state
    # is fp64) and casts the result back to the function's own zone.
    def compute_coulomb(r_carts, R_carts):
        ao = compute_AOs(..., r_carts, R_carts)          # 3a: forward as-is
        dtype_jnp = get_dtype_jnp("coulomb")
        # reconstruct in the caller-supplied precision, then down-cast
        diff = (r_carts - R_carts).astype(dtype_jnp)     # 3b
        ...

Auditing recipe.
    To verify a module:
      * (3a) Search for ``arg = jnp.asarray(arg, dtype=...)`` at the top of
        any public function. Each occurrence is a 3a candidate violation —
        the rebind silently coerces the argument for any subsequent
        forwarding too.
      * (3b) For each arithmetic expression, check that all operands have
        been cast to the function's zone in the immediately preceding lines.
      * (catastrophic cancellation) For each ``r - R`` style difference of
        coordinates, check the reconstruct-in-caller-precision-then-downcast
        pattern (in jQMC this is fp64 in practice because the upstream is
        always fp64, but the rule is "use the dtype the value was received
        in," not "hardcode fp64").

No hardcoded dtypes inside selectable-precision modules.
    Inside any module that owns one or more selectable-precision zones
    (i.e. any zone whose dtype can differ between ``"full"`` and
    ``"mixed"`` mode), **never hardcode** ``jnp.float64`` / ``np.float64``
    / ``jnp.float32`` / ``np.float32`` as a literal dtype for arrays the
    module produces or consumes. Always go through the accessors
    ``get_dtype_jnp("<zone>")`` / ``get_dtype_np("<zone>")`` so the dtype
    follows the active mode automatically. The only legitimate exception
    is a module whose owned data is **always fp64 by construction**,
    independent of any selectable zone:

      * ``mcmc`` / ``gfmc`` (MCMC and GFMC walker state, always fp64).
      * I/O modules that load and store external numerical data
        (``structure``, ``trexio_wrapper``, ``_jqmc_utility``,
        ``jqmc_tool``, and the ``_load_dataclass_from_hdf5`` /
        ``_save_dataclass_to_hdf5`` helpers in ``hamiltonians``):
        on-disk numerical data (AO exponents/coefficients, nuclear
        coordinates, geminal coefficients, etc.) is always fp64
        because fp32 storage would silently lose precision that no
        downstream fp64 upcast can recover.
      * Basis-data storage accessors. ``_*_jnp`` properties on
        selectable-precision dataclasses whose underlying storage
        field is typed ``npt.NDArray[np.float64]`` are *lift-only*
        adapters (numpy → ``jax.Array``), not arithmetic. The dtype
        is fp64 by construction: storage is loaded from
        HDF5/TREXIO/optimizer output (see Phase A1 numpy-storage
        migration), and downcasting at the accessor would silently
        lose precision that no downstream upcast can recover. The
        consumer is responsible for casting the lifted array to its
        own zone at the use site (Principle 3b). Concretely this
        covers the ``_*_jnp`` accessors for AO exponents/coefficients
        and normalization-factorial-ratio caches in
        ``atomic_orbital``, ``_mo_coefficients_jnp`` in
        ``molecular_orbital``, ``_lambda_matrix_jnp`` in
        ``determinant``, ``_j_matrix_jnp`` in ``jastrow_factor``,
        and the ``ShellPrimMap.from_aos_data`` constructor in
        ``atomic_orbital``.

    These modules may use ``jnp.float64`` / ``np.float64`` directly
    because the dtype is not a function of ``mode``. Audit with::

        grep -nE 'jnp\.float(32|64)|np\.float(32|64)' jqmc/<module>.py

    Each hit inside a selectable-precision module is a candidate violation.

================================================================================

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

==================  =================================  =========  ========  =====  =========
Zone                Owning module                      Default    Mixed     risk   E_L path
==================  =================================  =========  ========  =====  =========
``ao_eval``         atomic_orbital.py (forward)        float64    float32   low    core
``ao_grad_lap``     atomic_orbital.py (grad/Lap)       float64    float64   high§  core
``mo_eval``         molecular_orbital.py (forward)     float64    float64   high*  core
``mo_grad``         molecular_orbital.py (gradient)    float64    float64   high   core
``mo_lap``          molecular_orbital.py (Laplacian)   float64    float64   high   core
``jastrow_eval``    jastrow_factor.py (forward)        float64    float32   low    core†
``jastrow_grad_lap`` jastrow_factor.py (grad/lap)      float64    float32   low    core
``jastrow_ratio``   jastrow_factor.py (ratio update)   float64    float32   low    indirect‡
``det_eval``        determinant.py (geminal + log-det) float64    float64   high   core
``det_grad_lap``    determinant.py (grad/lap of lnDet) float64    float64   high   core
``det_ratio``       determinant.py (SM ratio update)   float64    float64   high   indirect‡
``coulomb``         coulomb_potential.py               float64    float32   low-med core
``wf_eval``         wavefunction.py (Psi, ln Psi)      float64    float64   high   core†
``wf_kinetic``      wavefunction.py (T_L assembly)     float64    float64   high   core
``wf_ratio``        wavefunction.py (Psi(R')/Psi(R))   float64    float64   high   no
``local_energy``    hamiltonians.py (T + V assembly)   float64    float64   high   core
``swct``            swct.py                            float64    float64   high   no
==================  =================================  =========  ========  =====  =========

\\* ``mo_eval`` is a high-risk zone even though the consumed AO values are
fp32: the small ``mo_coefficients @ aos`` matmul is run in this zone, and
its output feeds the determinant matrix, where fp32 round-off is
amplified by log|det|.  See ``bug/fp32`` diagnostics.

† ``jastrow_eval`` and ``wf_eval`` are on the E_L core path but their
forward values (J and ln|Psi|) do not enter the E_L formula directly
(E_L depends on *derivatives* of ln|Psi|).  Diagnostics show zero E_L
bias when these zones alone are fp32.

§ ``ao_grad_lap`` is fp64 even in mixed mode because the analytic
Laplacian kernel for spherical AOs contains catastrophic cancellation
(``4 Z² r² − 6 Z`` and ``(safe_div − 2 Z·base)² − safe_div² − 2 Z``
terms) that fp32 cannot resolve for tight Gaussians. Diagnostic
``bug/fp32/diag_07_ao_grad_vs_lap_split.py`` showed that
``ao_lap=fp32`` alone reproduces the full atomic-force bias
(``max|dF| ≈ 1.9 Ha/bohr`` on N₂ at scale=0.3, ``≈ 2e−2 Ha/bohr`` on
the water-cluster-8 system); the historical ``ao_grad=fp32`` zone was
safe in isolation (``max|dF| < 8e−3 Ha/bohr``) but is merged here with
``ao_lap`` because the fused ``compute_AOs_value_grad_lap`` kernel
shares one heavy expression (``exp / pow / phi / S_l_m``) across grad
and lap. Running that shared kernel at fp32 would break the lap path,
so the unified zone is fp64 always — a small extra cost on the
standalone ``compute_AOs_grad`` (which is not on the per-step hot
path) in exchange for a single source of truth for the shared kernel
dtype.

‡ ``det_ratio`` and ``jastrow_ratio`` affect E_L **indirectly** through
the ECP non-local potential, which evaluates Psi(R')/Psi(R) on a
quadrature grid via rank-1 ratio updates (see
``coulomb_potential.compute_ecp_non_local_parts_nearest_neighbors_fast_update``).
In non-ECP systems these zones have no E_L impact.

Usage::

    from jqmc._precision import get_dtype_jnp

    def compute_AOs(aos_data, r_carts):
        # Forwarding-only path: do NOT rebind r_carts here (Principle 3a).
        return _compute_AOs_kernel(aos_data, r_carts)

    def _compute_AOs_kernel(aos_data, r_carts):
        # This function performs arithmetic on r_carts, so it casts at the
        # use site (Principle 3b). The (r - R) reconstruction is done in
        # the dtype the values were received in (caller-supplied
        # precision: fp64 in jQMC because mcmc walker state is fp64 and
        # the atomic centers are loaded from disk as fp64). The result
        # is then down-cast to this function's own zone (``ao_eval``).
        # NOTE: never reach for another module's zone (e.g.
        # ``get_dtype_jnp("local_energy")``) here — that violates
        # Principle 1 (zone ↔ owning module is 1:1). atomic_orbital.py
        # may only consult ao_eval / ao_grad_lap.
        dtype_jnp = get_dtype_jnp("ao_eval")
        R_carts = aos_data._atomic_center_carts_jnp
        diff = (r_carts - R_carts).astype(dtype_jnp)
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
import numpy as np

logger = logging.getLogger(__name__)

# --- mode="full" (all float64, backward compatible) ---
# Zones are listed grouped by owning module for readability.
_FULL_PRECISION: dict[str, str] = {
    # atomic_orbital.py
    "ao_eval": "float64",  # AO forward evaluation
    "ao_grad_lap": "float64",  # AO gradient + Laplacian (unified for fused kernel)
    # molecular_orbital.py
    "mo_eval": "float64",  # MO forward evaluation (mo_coef @ AO)
    "mo_grad": "float64",  # MO gradient
    "mo_lap": "float64",  # MO Laplacian
    # jastrow_factor.py
    "jastrow_eval": "float64",  # Jastrow factor (J1/J2/J3)
    "jastrow_grad_lap": "float64",  # Jastrow gradient / Laplacian
    "jastrow_ratio": "float64",  # Jastrow ratio (rank-1 update)
    # determinant.py
    "det_eval": "float64",  # geminal matrix elements + log-det / SVD / AS reg
    "det_grad_lap": "float64",  # gradient / Laplacian of ln|Det|
    "det_ratio": "float64",  # |Det(R')|/|Det(R)| Sherman-Morrison rank-1 update
    # coulomb_potential.py
    "coulomb": "float64",  # Coulomb + ECP potential
    # wavefunction.py
    "wf_eval": "float64",  # Psi, ln Psi evaluators
    "wf_kinetic": "float64",  # T_L = -1/2 (lap_lnPsi + |grad_lnPsi|^2) assembly
    "wf_ratio": "float64",  # Psi(R')/Psi(R) = exp(J' - J) * det'/det (LRDMC discretized)
    # hamiltonians.py
    "local_energy": "float64",  # E_L = T + V assembly
    # swct.py
    "swct": "float64",  # SWCT omega / domega
}

# --- mode="mixed" (recommended mixed precision) ---
# Four "low risk" zones drop to float32:
#
#   ao_eval          - smooth Gaussian basis kernel; the dominant cost.
#                      The downstream consumer (mo_eval / det_eval /
#                      jastrow_eval) is fp64 and explicitly casts the AO
#                      result up before any sensitive arithmetic.
#   jastrow_eval     - smooth correlation function value (pre-exp).
#   jastrow_grad_lap - nabla J, nabla^2 J; smooth Jastrow factor, low
#                      cancellation. Diagnostics show bias < 8e-06 Ha
#                      at 32 electrons (0.05 kcal/mol margin ×11).
#                      Kept as a single zone (no grad/lap split) because
#                      both halves share the same fp32 risk profile and
#                      Jastrow grad/lap functions compute the two
#                      together (``compute_grads_and_laplacian_*``).
#   jastrow_ratio    - J(R')-J(R) log-ratio; smooth and well-behaved.
#                      Diagnostics show bias < 2e-06 Ha (margin ×44).
#
# All other zones stay fp64 because numerical experiments (see
# bug/fp32 diagnostics) show fp32 in those zones produces
# unacceptable bias on E_L for ~32-electron systems, OR the
# kernel is cheap enough that fp32 is not worth the bias:
#
#   ao_grad_lap   - analytic gradient + Laplacian kernel for spherical/
#                   Cartesian AOs.  Lap arithmetic contains catastrophic
#                   cancellation (``4 Z² r² − 6 Z`` and
#                   ``(safe_div − 2 Z·base)² − safe_div² − 2 Z``).
#                   diag_07 showed lap=fp32 alone yields max|dF| ≈ 1.9
#                   Ha/bohr on N₂ (scale=0.3), reproducing the entire
#                   bias of grad+lap=fp32. fp64 mandatory.  This zone
#                   merges the historical ``ao_grad`` (which was safe at
#                   fp32 in isolation) with ``ao_lap`` because the fused
#                   ``compute_AOs_value_grad_lap`` kernel evaluates
#                   ``exp / pow / phi / S_l_m`` once and reuses it across
#                   grad and lap; running the shared path at fp32 would
#                   break the lap output.
#   coulomb       - sum of 1/r + ECP spherical quadrature.  Cheap
#                   (O(N_e^2) el-el + O(N_e * N_nuc) el-ion, vs
#                   O(N_e * N_ao) AO eval) but contributes the
#                   largest individual bias among fp32 candidates
#                   (~6e-5 Ha at 64e/512 AO).  Cost/benefit favors fp64.
#
#   mo_eval       - mo_coef @ AO matmul feeds the determinant matrix;
#                   fp32 here amplifies into log|det| errors of O(1).
#   det_eval      - geminal matrix + log(det) + SVD; cancellation in
#                   log(det), SVD 1/s near-singular, ε≈1e-7 entries
#                   produce O(1) log|det| error.
#   mo_grad / mo_lap / det_grad_lap
#                 - second derivatives of ln|Psi|; cancellation-sensitive
#                   on the determinant side (the AO-side fp32 is absorbed
#                   by the fp64 mo_coef matmul). jastrow_grad_lap is the
#                   exception (smooth Jastrow, no severe cancellation).
#                   det_grad_lap is kept as a single zone (no grad/lap
#                   split) for symmetry with jastrow_grad_lap and because
#                   the determinant grad/lap functions naturally compute
#                   both quantities together (``compute_grads_and_laplacian_*``).
#   wf_kinetic    - sum (lap_J + lap_lnD) + |grad_J + grad_lnD|^2; cancellation.
#   local_energy  - T + V assembly; small differences between large terms.
#   det_ratio     - SM rank-1 ratio used by MCMC accept/reject AND
#                   ECP non-local Psi(R')/Psi(R) quadrature.
#   swct          - geometric SWCT correction, derivative-sensitive.
_MIXED_PRECISION: dict[str, str] = {
    # atomic_orbital.py
    "ao_eval": "float32",  # low risk (heavy kernel)
    "ao_grad_lap": "float64",  # high risk (catastrophic cancellation in 4Z²r²-6Z terms;
    # unified zone — historical ao_grad was safe at fp32 but is merged with ao_lap so
    # the fused compute_AOs_value_grad_lap kernel can share one heavy kernel at fp64)
    # molecular_orbital.py
    "mo_eval": "float64",  # high risk (feeds det_eval)
    "mo_grad": "float64",  # high risk
    "mo_lap": "float64",  # high risk
    # jastrow_factor.py
    "jastrow_eval": "float32",  # low risk
    "jastrow_grad_lap": "float32",  # low risk (smooth J; bias < 8e-06 Ha at 32e)
    "jastrow_ratio": "float32",  # low risk (smooth J ratio; bias < 2e-06 Ha at 32e)
    # determinant.py
    "det_eval": "float64",  # high risk (LU/det / SVD)
    "det_grad_lap": "float64",  # high risk (kept unsplit for symmetry with jastrow)
    "det_ratio": "float64",  # high risk (SM update error + ECP non-local ratio)
    # coulomb_potential.py
    "coulomb": "float64",  # cheap kernel + largest single fp32 bias (~6e-5 Ha)
    # wavefunction.py
    "wf_eval": "float64",  # high risk
    "wf_kinetic": "float64",  # high risk
    "wf_ratio": "float64",  # high risk (exp(J'-J)*det'/det in LRDMC)
    # hamiltonians.py
    "local_energy": "float64",  # high risk
    # swct.py
    "swct": "float64",  # high risk
}

ALL_ZONES = frozenset(_FULL_PRECISION.keys())

# Runtime zone -> dtype string mapping ("float32" or "float64").
# Strings are stored (not numpy/jax dtype types) so the str -> jnp.* / np.*
# conversion lives inside the per-flavor accessors below.  This keeps the
# concrete dtype flavor (jnp vs np) cleanly separated at the API boundary.
#
# This module-level dict is the **single source of truth** for the precision
# state within a Python process: ``configure(mode)`` clears and refills it,
# and ``get_dtype_jnp`` / ``get_dtype_np`` read from it.  No other variable
# (class attribute, environment variable, etc.) holds the active dtype
# mapping — this dict is the only place to consult or mutate.
_zone_dtypes: dict[str, str] = {}


def _validate_dtype_str(s: str) -> str:
    """Validate that *s* is one of the accepted dtype strings.

    Args:
        s: Either ``"float32"`` or ``"float64"``.

    Returns:
        The same string, unchanged.

    Raises:
        ValueError: If *s* is not one of the two accepted strings.
    """
    if s not in ("float32", "float64"):
        raise ValueError(f"Invalid dtype '{s}'. Must be 'float32' or 'float64'.")
    return s


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
        _zone_dtypes[zone] = _validate_dtype_str(dtype_str)


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
    _zone_dtypes[zone] = _validate_dtype_str(dtype_str)


def _get_zone_str(zone: str) -> str:
    """Return the stored dtype string for *zone* (``"float32"`` or ``"float64"``).

    When :func:`configure` has not been called, ``"float64"`` is returned
    for any zone name (backward compatible).  After :func:`configure` has
    been called, an unknown *zone* raises :class:`ValueError`.
    """
    if _zone_dtypes:
        if zone not in ALL_ZONES:
            raise ValueError(f"Unknown precision zone '{zone}'. Available zones: {sorted(ALL_ZONES)}")
    return _zone_dtypes.get(zone, "float64")


def get_dtype_jnp(zone: str) -> type:
    """Return the JAX dtype for a Precision Zone.

    Args:
        zone: Precision Zone name (e.g. ``"ao_eval"``, ``"wf_kinetic"``).

    Returns:
        ``jnp.float32`` or ``jnp.float64``.
    """
    return jnp.float32 if _get_zone_str(zone) == "float32" else jnp.float64


def get_dtype_np(zone: str) -> type:
    """Return the numpy dtype for a Precision Zone.

    Convenience helper for numpy-only code paths (e.g. ``_debug`` reference
    implementations) where importing or branching on ``jnp`` is awkward.

    Args:
        zone: Precision Zone name (e.g. ``"ao_eval"``, ``"wf_kinetic"``).

    Returns:
        ``np.float32`` or ``np.float64``.
    """
    return np.float32 if _get_zone_str(zone) == "float32" else np.float64


def is_mixed_precision_enabled() -> bool:
    """Return ``True`` if at least one zone is set to float32."""
    return any(s == "float32" for s in _zone_dtypes.values())


def get_tolerance(zone: str, level: str = "strict") -> tuple[float, float]:
    """Return test tolerances ``(atol, rtol)`` scaled by the zone's dtype.

    Args:
        zone: Precision Zone name.
        level: ``"strict"`` or ``"loose"``.

    Returns:
        ``(atol, rtol)`` tuple appropriate for the zone's current dtype.
    """
    from jqmc._setting import _TOLERANCE

    return _TOLERANCE[level][_get_zone_str(zone)]


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


def mode_label() -> str:
    """Return a short label for the active precision mode.

    Returns:
        ``"Full Precision (FP64)"`` or ``"Mixed Precision (FP32 + FP64)"``.
    """
    if is_mixed_precision_enabled():
        return "Mixed Precision (FP32 + FP64)"
    return "Full Precision (FP64)"


def zone_detail() -> str:
    """Return a per-zone detail string of the current precision configuration."""
    lines = []
    for zone in sorted(ALL_ZONES):
        lines.append(f"  {zone}: {_get_zone_str(zone)}")
    return "\n".join(lines)
