"""Collections of useful functions."""

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

# from typing import Any, Callable, Mapping, Optional
# JAX
import jax
from jax import jit
from jax import numpy as jnp

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


@jit
def _legendre_tablated(n: int, x: float) -> float:
    """Tabulated Legendre polynomials.

    # see https://en.wikipedia.org/wiki/Legendre_polynomials

    """
    conditions = [n == 0, n == 1, n == 2, n == 3, n == 4, n == 5, n == 6]
    P_n = [
        1,
        x,
        1.0 / 2.0 * (3.0 * x**2 - 1),
        1.0 / 2.0 * (5.0 * x**3 - 3.0 * x),
        1.0 / 8.0 * (35.0 * x**4 - 30.0 * x**2 + 3.0),
        1.0 / 8.0 * (63.0 * x**5 - 70 * x**3 + 15 * x),
        1.0 / 16.0 * (231.0 * x**6 - 315.0 * x**4 + 105.0 * x**2 - 5),
    ]
    return jnp.select(conditions, P_n, jnp.nan)


'''
@jit
def eval_legendre(n: jnpt.ArrayLike, x: jnpt.ArrayLike) -> jax.Array:
    """Evaluate Legendre polynomials of specified degrees at provided point(s).

    This function makes use of a vectorized version of the Legendre polynomial recurrence
    relation to compute the necessary polynomials up to the maximum degree found in 'n'.
    It then selects and returns the values of the polynomials at the degrees specified in 'n'
    and evaluated at the points in 'x'.

    Args:
        n (jnp.ndarray):
            An array of integer degrees for which the Legendre polynomials are to be evaluated.
            Each element must be a non-negative integer and the array can be of any shape.
        x (jnp.ndarray):
            The point(s) at which the Legendre polynomials are to be evaluated. Can be a single
            point (float) or an array of points. The shape must be broadcastable to the shape
            of 'n'.

    Returns:
        jnp.ndarray:
            An array of Legendre polynomial values. The output has the same shape as 'n' and 'x'
            after broadcasting. The i-th entry corresponds to the Legendre polynomial of degree
            'n[i]' evaluated at point 'x[i]'.

    Notes:
        This function makes use of the vectorized map (vmap) functionality in JAX to efficiently
        compute and select the necessary Legendre polynomial values.
    """
    n = jnp.asarray(n)
    x = jnp.asarray(x)

    p = jnp.where(
        n.ndim == 1 and x.ndim == 1,
        jnp.diagonal(jax.vmap(lambda ni: jax.vmap(lambda xi: legendre_tablated(ni, xi))(x))(n)),
        jax.vmap(lambda ni: jax.vmap(lambda xi: legendre_tablated(ni, xi))(x))(n),
    )

    return jnp.squeeze(p)
'''

"""
if __name__ == "__main__":
    n = np.arange(7)
    print(jnp.max(n))

    print(f"n = {n}")
    print(f"n shape = {n.shape}")
    print(f"n ndim = {n.ndim}")

    x = np.linspace(-1, 1, n.shape[0])

    print(f"x = {x}")
    print(f"x shape = {x.shape}")
    print(f"x ndim = {x.ndim}")

    y_pred = eval_legendre(n, x)
    y = scipy.special.eval_legendre(n, x)

    print(f"y_pred = {y_pred}")
    print(f"y_pred shape = {y_pred.shape}")
    print(f"y = {y}")
    print(f"y shape = {y.shape}")

    assert np.allclose(y_pred, y, rtol=1e-5, atol=1e-8), "Results do not match"
    print("Results match")
"""
