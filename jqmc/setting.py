"""setting."""

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
decimal_auto_vs_numerical_deriv = 3
decimal_numerical_vs_analytic_deriv = 3
decimal_auto_vs_analytic_deriv = 8

# Test tolerance settings (numerical laplacian)
atol_auto_vs_numerical_deriv = 1.0e-1
rtol_auto_vs_numerical_deriv = 1.0e-4
atol_numerical_vs_analytic_deriv = 1.0e-1
rtol_numerical_vs_analytic_deriv = 1.0e-4

# Test tolerance settings (others)
decimal_debug_vs_production = 8

# Numerical stability settings
EPS_stabilizing_jax_AO_cart_deriv = 1.0e-16
