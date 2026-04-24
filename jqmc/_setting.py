"""Setting module.

Contains physical constants, numerical stability parameters, and
test tolerance definitions used across jQMC.
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

import numpy as np

# Unit conversion
Bohr_to_Angstrom = 0.529177210903
Angstrom_to_Bohr = 1.0 / Bohr_to_Angstrom

# default ECP integral parameters
Nv_default = 6
NN_default = 1

# Minimal warmup steps (VMC)
MCMC_MIN_WARMUP_STEPS = 30
MCMC_MIN_BIN_BLOCKS = 10

# Minimal warmup steps (GFMC)
GFMC_MIN_WARMUP_STEPS = 25
GFMC_MIN_BIN_BLOCKS = 10
GFMC_MIN_COLLECT_STEPS = 5

# on the fly statistics param
GFMC_ON_THE_FLY_WARMUP_STEPS = 20
GFMC_ON_THE_FLY_COLLECT_STEPS = 10
GFMC_ON_THE_FLY_BIN_BLOCKS = 10

# Test tolerance settings (gradients, laplacians)
atol_auto_vs_numerical_deriv = 1.0e-1
rtol_auto_vs_numerical_deriv = 1.0e-4
atol_numerical_vs_analytic_deriv = 1.0e-1
rtol_numerical_vs_analytic_deriv = 1.0e-4
atol_auto_vs_analytic_deriv = 1.0e-8
rtol_auto_vs_analytic_deriv = 1.0e-6

# Test tolerance settings (others)
atol_debug_vs_production = 1.0e-8
rtol_debug_vs_production = 1.0e-6
atol_consistency = 1.0e-8
rtol_consistency = 1.0e-6

# --- Test tolerance dict (dtype-aware) ---
#
# Accessed via ``_precision.get_tolerance(zone, level)`` which resolves the
# zone's current dtype and returns ``(atol, rtol)``.
#
# Levels:
#   strict — two exact implementations of the same quantity (debug vs
#            production, analytic vs autodiff).  Difference is pure
#            floating-point round-off.
#   loose  — comparison involving numerical differentiation or quadrature.
#            Finite-difference truncation error dominates, so tolerances
#            are much wider.
_TOLERANCE: dict[str, dict[str, tuple[float, float]]] = {
    "strict": {"float64": (1e-8, 1e-6), "float32": (1e-5, 1e-3)},
    "loose": {"float64": (1e-1, 1e-4), "float32": (1e-1, 1e-3)},
}

# --- Dtype-aware EPS constants ---
#
# Some EPS values are tuned for float64 and break under float32 (underflow,
# loss of stabilization).  Use ``get_eps(name, dtype)`` to obtain the
# appropriate value for the current precision zone.
#
# Constants:
#   machine_precision — floor for safe ratio in diagnostics.
#   stabilizing_ao    — small epsilon for AO Cartesian derivative stabilization.
#   rcond_svd         — threshold for SVD pseudoinverse of the geminal matrix.
_EPS_DTYPE_AWARE: dict[str, dict[str, float]] = {
    "machine_precision": {"float64": 1e-300, "float32": 1e-38},
    "stabilizing_ao": {"float64": 1e-16, "float32": 1e-7},
    "rcond_svd": {"float64": 1e-20, "float32": 1e-6},
}


def get_eps(name: str, dtype) -> float:
    """Return a dtype-aware numerical stability constant.

    Args:
        name: One of ``"machine_precision"``, ``"stabilizing_ao"``,
            ``"rcond_svd"``.
        dtype: A NumPy/JAX dtype (e.g. ``jnp.float32``, ``np.float64``).

    Returns:
        The appropriate epsilon value for the given dtype.

    Raises:
        KeyError: If *name* is not a known EPS constant.
    """
    dtype_key = "float32" if np.dtype(dtype) == np.float32 else "float64"
    return _EPS_DTYPE_AWARE[name][dtype_key]


# Numerical stability settings for AO
EPS_stabilizing_jax_AO_cart_deriv = 1.0e-16

# Threshold for SVD pseudoinverse of the geminal matrix G.
# Singular values below EPS_rcond_SVD * s_max are zeroed to avoid 1/~0 NaN.
# Must be very small (e.g. 1e-20); see compute_grads_and_laplacian_ln_Det
# docstring in determinant.py for why a larger value breaks the analytic path.
EPS_rcond_SVD = 1.0e-20

# Threshold for zero-division guards.
# Denominators with absolute value below this are treated as zero.
EPS_zero_division = 1.0e-30

# Small epsilon for safe distance / sqrt stabilization.
# Added to squared distances before sqrt to keep gradients finite
# when particles coincide (e.g. r_eN -> 0).
EPS_safe_distance = 1.0e-12

# Near-machine-precision floor for safe ratio in diagnostics.
EPS_machine_precision = 1.0e-300

# Absolute floor for diag(S) in scale-invariant SR.
# Parameters with diag_S below this threshold are considered converged
# (the wavefunction derivative has near-zero variance) and are frozen
# to prevent the scale-invariant normalization from amplifying noise
# into catastrophically large parameter updates.
min_S_diag_abs = 1.0e-10
