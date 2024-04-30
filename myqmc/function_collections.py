import numpy as np
import scipy
import jax
from jax import numpy as jnp
from jax import jit
from jaxtyping import Array, Float, Int
from typing import Any, Callable, Mapping, Optional

import scipy.special


@jit
def legendre_tablated(n, x):
    # see https://en.wikipedia.org/wiki/Legendre_polynomials
    conditions = [n == 0, n == 1, n == 2, n == 3]
    P_n = [1, x, 1.0 / 2.0 * (3.0 * x**2 - 1), 1.0 / 2.0 * (5.0 * x**3 - 3.0 * x)]
    return jnp.select(conditions, P_n, jnp.nan)


@jit
def eval_legendre(n: Int[Array, "n"], x: Float[Array, "m"]) -> Float[Array, "n m"]:
    """
    Evaluate Legendre polynomials of specified degrees at provided point(s).

    This function makes use of a vectorized version of the Legendre polynomial recurrence relation to
    compute the necessary polynomials up to the maximum degree found in 'n'. It then selects and returns
    the values of the polynomials at the degrees specified in 'n' and evaluated at the points in 'x'.

    Parameters:
        n (jnp.ndarray): An array of integer degrees for which the Legendre polynomials are to be evaluated.
                        Each element must be a non-negative integer and the array can be of any shape.
        x (jnp.ndarray): The point(s) at which the Legendre polynomials are to be evaluated. Can be a single
                        point (float) or an array of points. The shape must be broadcastable to the shape of 'n'.

    Returns:
        jnp.ndarray: An array of Legendre polynomial values. The output has the same shape as 'n' and 'x' after broadcasting.
                    The i-th entry corresponds to the Legendre polynomial of degree 'n[i]' evaluated at point 'x[i]'.

    Notes:
        This function makes use of the vectorized map (vmap) functionality in JAX to efficiently compute and select
        the necessary Legendre polynomial values.
    """

    n = jnp.asarray(n)
    x = jnp.asarray(x)

    p = jnp.where(
        n.ndim == 1 and x.ndim == 1,
        jnp.diagonal(
            jax.vmap(lambda ni: jax.vmap(lambda xi: legendre_tablated(ni, xi))(x))(n)
        ),
        jax.vmap(lambda ni: jax.vmap(lambda xi: legendre_tablated(ni, xi))(x))(n),
    )

    return jnp.squeeze(p)


def test_eval_legendre():
    n = np.array([0, 1])
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


if __name__ == "__main__":
    test_eval_legendre()
