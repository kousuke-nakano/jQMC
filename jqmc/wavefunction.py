"""Wavefunction module."""

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

# python modules
# from dataclasses import dataclass
from logging import Formatter, StreamHandler, getLogger

# import jax
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import grad, hessian, jit, jvp, vmap
from jax import typing as jnpt

from .determinant import (
    Geminal_data,
    Geminal_data_deriv_params,
    Geminal_data_deriv_R,
    compute_det_geminal_all_elements_jax,
    compute_grads_and_laplacian_ln_Det_jax,
    compute_ln_det_geminal_all_elements_jax,
    compute_ratio_determinant_part_jax,
)
from .jastrow_factor import (
    Jastrow_data,
    Jastrow_data_deriv_params,
    Jastrow_data_deriv_R,
    compute_grads_and_laplacian_Jastrow_part_jax,
    compute_Jastrow_part_jax,
    compute_ratio_Jastrow_part_jax,
)

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


@struct.dataclass
class Wavefunction_data:
    """The class contains data for computing wavefunction.

    Args:
        jastrow_data (Jastrow_data)
        geminal_data (Geminal_data)
    """

    jastrow_data: Jastrow_data = struct.field(pytree_node=True, default_factory=lambda: Jastrow_data())
    geminal_data: Geminal_data = struct.field(pytree_node=True, default_factory=lambda: Wavefunction_data())

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        self.jastrow_data.sanity_check()
        self.geminal_data.sanity_check()

    def get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        # Replace geminal_data.logger_info() with geminal_data.get_info() output.
        info_lines.extend(self.geminal_data.get_info())
        # Replace jastrow_data.logger_info() with jastrow_data.get_info() output.
        info_lines.extend(self.jastrow_data.get_info())
        return info_lines

    def logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self.get_info():
            logger.info(line)


@struct.dataclass
class Wavefunction_data_deriv_params(Wavefunction_data):
    """See Wavefunction_data."""

    jastrow_data: Jastrow_data = struct.field(pytree_node=True)
    geminal_data: Geminal_data = struct.field(pytree_node=True)

    @classmethod
    def from_base(cls, wavefunction_data: Wavefunction_data):
        """Switch pytree_node."""
        jastrow_data = Jastrow_data_deriv_params.from_base(wavefunction_data.jastrow_data)
        geminal_data = Geminal_data_deriv_params.from_base(wavefunction_data.geminal_data)
        return cls(jastrow_data=jastrow_data, geminal_data=geminal_data)


@struct.dataclass
class Wavefunction_data_deriv_R(Wavefunction_data):
    """See Wavefunction_data."""

    jastrow_data: Jastrow_data = struct.field(pytree_node=True)
    geminal_data: Geminal_data = struct.field(pytree_node=True)

    @classmethod
    def from_base(cls, wavefunction_data: Wavefunction_data):
        """Switch pytree_node."""
        jastrow_data = Jastrow_data_deriv_R.from_base(wavefunction_data.jastrow_data)
        geminal_data = Geminal_data_deriv_R.from_base(wavefunction_data.geminal_data)
        return cls(jastrow_data=jastrow_data, geminal_data=geminal_data)


@struct.dataclass
class Wavefunction_data_no_deriv(Wavefunction_data):
    """See Wavefunction_data."""

    jastrow_data: Jastrow_data = struct.field(pytree_node=False)
    geminal_data: Geminal_data = struct.field(pytree_node=False)

    @classmethod
    def from_base(cls, wavefunction_data: Wavefunction_data):
        """Switch pytree_node."""
        jastrow_data = wavefunction_data.jastrow_data
        geminal_data = wavefunction_data.geminal_data
        return cls(jastrow_data=jastrow_data, geminal_data=geminal_data)


@jit
def evaluate_ln_wavefunction_jax(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> float:
    """Evaluate the value of Wavefunction.

    The method is for evaluate the logarithm of |wavefunction| (ln |Psi|) at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The log value of the given wavefunction (float)
    """
    Jastrow_part = compute_Jastrow_part_jax(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    Determinant_part = compute_det_geminal_all_elements_jax(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return Jastrow_part + jnp.log(jnp.abs(Determinant_part))


@jit
def evaluate_wavefunction_jax(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> float | complex:
    """The method is for evaluate wavefunction (Psi) at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The value of the given wavefunction (float).
    """
    Jastrow_part = compute_Jastrow_part_jax(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    Determinant_part = compute_det_geminal_all_elements_jax(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return jnp.exp(Jastrow_part) * Determinant_part


def evaluate_jastrow_jax(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> float:
    """The method is for evaluate the Jastrow part of the wavefunction (Psi) at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The value of the given exp(Jastrow (float))  Notice that the Jastrow factor here includes the exp factor, i.e., exp(J).
    """
    Jastrow_part = compute_Jastrow_part_jax(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return jnp.exp(Jastrow_part)


def evaluate_determinant_jax(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> float:
    """The method is for evaluate the determinant part of the wavefunction (Psi) at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The value of the given determinant (float)
    """
    Determinant_part = compute_det_geminal_all_elements_jax(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return Determinant_part


@jit
def compute_kinetic_energy_jax(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> float | complex:
    """The method is for computing kinetic energy of the given WF at (r_up_carts, r_dn_carts).

    Fully exploit the JAX library for the kinetic energy calculation.

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The kinetic energy with the given wavefunction (float | complex)
    """
    kinetic_energy_all_elements_up, kinetic_energy_all_elements_dn = compute_kinetic_energy_all_elements_jax(
        wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    K = jnp.sum(kinetic_energy_all_elements_up) + jnp.sum(kinetic_energy_all_elements_dn)

    return K


def compute_kinetic_energy_debug(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """See compute_kinetic_energy_api."""
    kinetic_energy_all_elements_up, kinetic_energy_all_elements_dn = compute_kinetic_energy_all_elements_debug(
        wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    return np.sum(kinetic_energy_all_elements_up) + np.sum(kinetic_energy_all_elements_dn)


def compute_kinetic_energy_all_elements_debug(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """See compute_kinetic_energy_api."""
    """
    # compute grad
    diff_h = 1.0e-5

    n_up, d_up = r_up_carts.shape
    grad_ln_Psi_up = np.zeros((n_up, d_up))
    for i in range(n_up):
        for d in range(d_up):
            r_up_plus = r_up_carts.copy()
            r_up_minus = r_up_carts.copy()
            r_up_plus[i, d] += diff_h
            r_up_minus[i, d] -= diff_h

            ln_Psi_plus = evaluate_ln_wavefunction_api(wavefunction_data, r_up_plus, r_dn_carts)
            ln_Psi_minus = evaluate_ln_wavefunction_api(wavefunction_data, r_up_minus, r_dn_carts)

            grad_ln_Psi_up[i][d] = (ln_Psi_plus - ln_Psi_minus) / (2 * diff_h)

    n_dn, d_dn = r_dn_carts.shape
    grad_ln_Psi_dn = np.zeros((n_dn, d_dn))
    for i in range(n_dn):
        for d in range(d_dn):
            r_dn_plus = r_dn_carts.copy()
            r_dn_minus = r_dn_carts.copy()
            r_dn_plus[i, d] += diff_h
            r_dn_minus[i, d] -= diff_h

            ln_Psi_plus = evaluate_ln_wavefunction_api(wavefunction_data, r_up_carts, r_dn_plus)
            ln_Psi_minus = evaluate_ln_wavefunction_api(wavefunction_data, r_up_carts, r_dn_minus)

            grad_ln_Psi_dn[i][d] = (ln_Psi_plus - ln_Psi_minus) / (2 * diff_h)

    # compute laplacians
    diff_h = 1.0e-3

    ln_Psi = evaluate_ln_wavefunction_api(wavefunction_data, r_up_carts, r_dn_carts)

    n_up, d_up = r_up_carts.shape
    laplacian_ln_Psi_up = np.zeros(n_up)
    for i in range(n_up):
        for d in range(d_up):
            r_up_plus = r_up_carts.copy()
            r_up_minus = r_up_carts.copy()
            r_up_plus[i, d] += diff_h
            r_up_minus[i, d] -= diff_h

            ln_Psi_plus = evaluate_ln_wavefunction_api(wavefunction_data, r_up_plus, r_dn_carts)
            ln_Psi_minus = evaluate_ln_wavefunction_api(wavefunction_data, r_up_minus, r_dn_carts)

            laplacian_ln_Psi_up[i] += (ln_Psi_plus + ln_Psi_minus - 2 * ln_Psi) / (diff_h**2)

    n_dn, d_dn = r_dn_carts.shape
    laplacian_ln_Psi_dn = np.zeros(n_dn)
    for i in range(n_dn):
        for d in range(d_dn):
            r_dn_plus = r_dn_carts.copy()
            r_dn_minus = r_dn_carts.copy()
            r_dn_plus[i, d] += diff_h
            r_dn_minus[i, d] -= diff_h

            ln_Psi_plus = evaluate_ln_wavefunction_api(wavefunction_data, r_up_carts, r_dn_plus)
            ln_Psi_minus = evaluate_ln_wavefunction_api(wavefunction_data, r_up_carts, r_dn_minus)

            laplacian_ln_Psi_dn[i] += (ln_Psi_plus + ln_Psi_minus - 2 * ln_Psi) / (diff_h**2)

    kinetic_energy_all_elements_up = -1.0 / 2.0 * (laplacian_ln_Psi_up + np.sum(grad_ln_Psi_up**2, axis=1))
    kinetic_energy_all_elements_dn = -1.0 / 2.0 * (laplacian_ln_Psi_dn + np.sum(grad_ln_Psi_dn**2, axis=1))
    """

    # compute laplacians
    diff_h = 2.0e-4

    Psi = evaluate_wavefunction_jax(wavefunction_data, r_up_carts, r_dn_carts)

    n_up, d_up = r_up_carts.shape
    laplacian_Psi_up = np.zeros(n_up)
    for i in range(n_up):
        for d in range(d_up):
            r_up_plus = r_up_carts.copy()
            r_up_minus = r_up_carts.copy()
            r_up_plus[i, d] += diff_h
            r_up_minus[i, d] -= diff_h

            Psi_plus = evaluate_wavefunction_jax(wavefunction_data, r_up_plus, r_dn_carts)
            Psi_minus = evaluate_wavefunction_jax(wavefunction_data, r_up_minus, r_dn_carts)

            laplacian_Psi_up[i] += (Psi_plus + Psi_minus - 2 * Psi) / (diff_h**2)

    n_dn, d_dn = r_dn_carts.shape
    laplacian_Psi_dn = np.zeros(n_dn)
    for i in range(n_dn):
        for d in range(d_dn):
            r_dn_plus = r_dn_carts.copy()
            r_dn_minus = r_dn_carts.copy()
            r_dn_plus[i, d] += diff_h
            r_dn_minus[i, d] -= diff_h

            Psi_plus = evaluate_wavefunction_jax(wavefunction_data, r_up_carts, r_dn_plus)
            Psi_minus = evaluate_wavefunction_jax(wavefunction_data, r_up_carts, r_dn_minus)

            laplacian_Psi_dn[i] += (Psi_plus + Psi_minus - 2 * Psi) / (diff_h**2)

    kinetic_energy_all_elements_up = -1.0 / 2.0 * laplacian_Psi_up / Psi
    kinetic_energy_all_elements_dn = -1.0 / 2.0 * laplacian_Psi_dn / Psi

    return (kinetic_energy_all_elements_up, kinetic_energy_all_elements_dn)


@jit
def compute_kinetic_energy_all_elements_jax(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> jax.Array:
    """See compute_kinetic_energy_api."""
    # compute gradients
    grad_J_up = grad(compute_Jastrow_part_jax, argnums=1)(wavefunction_data.jastrow_data, r_up_carts, r_dn_carts)
    grad_J_dn = grad(compute_Jastrow_part_jax, argnums=2)(wavefunction_data.jastrow_data, r_up_carts, r_dn_carts)
    grad_ln_Det_up = grad(compute_ln_det_geminal_all_elements_jax, argnums=1)(
        wavefunction_data.geminal_data, r_up_carts, r_dn_carts
    )
    grad_ln_Det_dn = grad(compute_ln_det_geminal_all_elements_jax, argnums=2)(
        wavefunction_data.geminal_data, r_up_carts, r_dn_carts
    )

    grad_ln_Psi_up = grad_J_up + grad_ln_Det_up
    grad_ln_Psi_dn = grad_J_dn + grad_ln_Det_dn

    # compute laplacians
    hessian_J_up = hessian(compute_Jastrow_part_jax, argnums=1)(wavefunction_data.jastrow_data, r_up_carts, r_dn_carts)
    laplacian_J_up = jnp.einsum("ijij->i", hessian_J_up)
    hessian_J_dn = hessian(compute_Jastrow_part_jax, argnums=2)(wavefunction_data.jastrow_data, r_up_carts, r_dn_carts)
    laplacian_J_dn = jnp.einsum("ijij->i", hessian_J_dn)

    hessian_ln_Det_up = hessian(compute_ln_det_geminal_all_elements_jax, argnums=1)(
        wavefunction_data.geminal_data, r_up_carts, r_dn_carts
    )
    laplacian_ln_Det_up = jnp.einsum("ijij->i", hessian_ln_Det_up)
    hessian_ln_Det_dn = hessian(compute_ln_det_geminal_all_elements_jax, argnums=2)(
        wavefunction_data.geminal_data, r_up_carts, r_dn_carts
    )
    laplacian_ln_Det_dn = jnp.einsum("ijij->i", hessian_ln_Det_dn)

    laplacian_Psi_up = laplacian_J_up + laplacian_ln_Det_up
    laplacian_Psi_dn = laplacian_J_dn + laplacian_ln_Det_dn

    kinetic_energy_all_elements_up = -1.0 / 2.0 * (laplacian_Psi_up + jnp.sum(grad_ln_Psi_up**2, axis=1))
    kinetic_energy_all_elements_dn = -1.0 / 2.0 * (laplacian_Psi_dn + jnp.sum(grad_ln_Psi_dn**2, axis=1))

    return (kinetic_energy_all_elements_up, kinetic_energy_all_elements_dn)


@jit
def compute_kinetic_energy_all_elements_jax_tricky(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> jax.Array:
    """See compute_kinetic_energy_api."""
    # compute gradients
    grad_ln_Psi_up = grad(evaluate_ln_wavefunction_jax, argnums=1)(wavefunction_data, r_up_carts, r_dn_carts)
    grad_ln_Psi_dn = grad(evaluate_ln_wavefunction_jax, argnums=2)(wavefunction_data, r_up_carts, r_dn_carts)

    # compute laplacians. The above Hessian implemenation is redundant since the nondiagonal parts are not needed for Laplacian calculations.
    # The following implementation is more efficient, while it's a little bit tricky.
    def laplacian_wrt_arg(func, arg):
        ## Flatten the argument to a 1D array for computation
        arg_flat = arg.reshape(-1)

        ## Helper function that reshapes the flattened argument back to its original shape
        def func_flat(x):
            return func(x.reshape(arg.shape))

        ## Obtain the gradient function with respect to the flattened argument
        grad_func = grad(func_flat)

        ## Define a function that computes the directional derivative (i.e., the Hessian-vector product) using jax.jvp
        def hvp(e):
            # jax.jvp returns a tuple (function value, tangent value); here we use the tangent value
            _, hvp_val = jvp(grad_func, (arg_flat,), (e,))
            return hvp_val

        ## Create the standard basis (unit vectors for each dimension)
        n = arg_flat.shape[0]
        basis = jnp.eye(n)

        ## Compute the Hessian-vector product for each standard basis vector.
        ## The dot product of the basis vector with its Hessian-vector product gives the corresponding diagonal element.
        diag = vmap(lambda e: jnp.dot(e, hvp(e)))(basis)
        # The Laplacian is the sum of the diagonal elements
        return diag

    ## For r_up, compute the Laplacian while keeping r_dn fixed
    def f_r_up(r_up):
        return evaluate_ln_wavefunction_jax(wavefunction_data, r_up, r_dn_carts)

    laplacian_r_up = laplacian_wrt_arg(f_r_up, r_up_carts).reshape(r_up_carts.shape[0], -1).sum(axis=1)

    ## Similarly, for r_dn, compute the Laplacian while keeping r_up fixed
    def f_r_dn(r_dn):
        return evaluate_ln_wavefunction_jax(wavefunction_data, r_up_carts, r_dn)

    laplacian_r_dn = laplacian_wrt_arg(f_r_dn, r_dn_carts).reshape(r_dn_carts.shape[0], -1).sum(axis=1)

    kinetic_energy_all_elements_up = -1.0 / 2.0 * (laplacian_r_up + jnp.sum(grad_ln_Psi_up**2, axis=1))
    kinetic_energy_all_elements_dn = -1.0 / 2.0 * (laplacian_r_dn + jnp.sum(grad_ln_Psi_dn**2, axis=1))

    return (kinetic_energy_all_elements_up, kinetic_energy_all_elements_dn)


def compute_discretized_kinetic_energy_debug(
    alat: float, wavefunction_data: Wavefunction_data, r_up_carts: npt.NDArray, r_dn_carts: npt.NDArray
) -> list[tuple[npt.NDArray, npt.NDArray]]:
    r"""_summary.

    Args:
        alat (float): Hamiltonian discretization (bohr), which will be replaced with LRDMC_data.
        wavefunction_data (Wavefunction_data): an instance of Qavefunction_data, which will be replaced with LRDMC_data.
        r_carts_up (npt.NDArray): up electron position (N_e,3).
        r_carts_dn (npt.NDArray): down electron position (N_e,3).

    Returns:
        list[tuple[npt.NDArray, npt.NDArray]], list[npt.NDArray]:
            return mesh for the LRDMC kinetic part, a list containing tuples containing (r_carts_up, r_carts_dn),
            and a list containing values of the \Psi(x')/\Psi(x) corresponding to the grid.
    """
    mesh_kinetic_part = []

    # up electron
    for r_up_i in range(len(r_up_carts)):
        # x, plus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 0] += alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # x, minus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 0] -= alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # y, plus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 1] += alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # y, minus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 1] -= alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # z, plus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 2] += alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))
        # z, minus
        r_up_carts_p = r_up_carts.copy()
        r_up_carts_p[r_up_i, 2] -= alat
        mesh_kinetic_part.append((r_up_carts_p, r_dn_carts))

    # dn electron
    for r_dn_i in range(len(r_dn_carts)):
        # x, plus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 0] += alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # x, minus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 0] -= alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # y, plus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 1] += alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # y, minus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 1] -= alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # z, plus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 2] += alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))
        # z, minus
        r_dn_carts_p = r_dn_carts.copy()
        r_dn_carts_p[r_dn_i, 2] -= alat
        mesh_kinetic_part.append((r_up_carts, r_dn_carts_p))

    elements_kinetic_part = [
        float(
            -1.0
            / (2.0 * alat**2)
            * evaluate_wavefunction_jax(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_, r_dn_carts=r_dn_carts_)
            / evaluate_wavefunction_jax(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
        )
        for r_up_carts_, r_dn_carts_ in mesh_kinetic_part
    ]

    r_up_carts_combined = np.array([up for up, _ in mesh_kinetic_part])
    r_dn_carts_combined = np.array([dn for _, dn in mesh_kinetic_part])

    return r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part


@jit
def compute_discretized_kinetic_energy_jax(
    alat: float, wavefunction_data, r_up_carts: jnp.ndarray, r_dn_carts: jnp.ndarray, RT: jnp.ndarray
) -> tuple[list[tuple[npt.NDArray, npt.NDArray]], list[npt.NDArray], jax.Array]:
    r"""Function for computing discretized kinetic grid points and thier energies with a given lattice space (alat).

    Args:
        alat (float): Hamiltonian discretization (bohr), which will be replaced with LRDMC_data.
        wavefunction_data (Wavefunction_data): an instance of Qavefunction_data, which will be replaced with LRDMC_data.
        r_carts_up (npt.NDArray): up electron position (N_e,3).
        r_carts_dn (npt.NDArray): down electron position (N_e,3).
        RT (npt.NDArray): Rotation matrix. \equiv R.T

    Returns:
        list[tuple[npt.NDArray, npt.NDArray]], list[npt.NDArray], jax.Array:
            return mesh for the LRDMC kinetic part, a list containing tuples containing (r_carts_up, r_carts_dn),
            a list containing values of the \Psi(x')/\Psi(x) corresponding to the grid, and the new jax_PRNG_key
            that should be used in the next call of this @jitted function.
    """
    # Define the shifts to apply (+/- alat in each coordinate direction)
    shifts = alat * jnp.array(
        [
            [1, 0, 0],  # x+
            [-1, 0, 0],  # x-
            [0, 1, 0],  # y+
            [0, -1, 0],  # y-
            [0, 0, 1],  # z+
            [0, 0, -1],  # z-
        ]
    )  # Shape: (6, 3)

    shifts = shifts @ RT  # Shape: (6, 3)

    # num shift
    num_shifts = shifts.shape[0]

    # Process up-spin electrons
    num_up_electrons = r_up_carts.shape[0]
    num_up_configs = num_up_electrons * num_shifts

    # Create base positions repeated for each configuration
    base_positions_up = jnp.repeat(r_up_carts[None, :, :], num_up_configs, axis=0)  # Shape: (num_up_configs, N_up, 3)

    # Initialize shifts_to_apply_up
    shifts_to_apply_up = jnp.zeros_like(base_positions_up)

    # Create indices for configurations
    config_indices_up = jnp.arange(num_up_configs)
    electron_indices_up = jnp.repeat(jnp.arange(num_up_electrons), num_shifts)
    shift_indices_up = jnp.tile(jnp.arange(num_shifts), num_up_electrons)

    # Apply shifts to the appropriate electron in each configuration
    shifts_to_apply_up = shifts_to_apply_up.at[config_indices_up, electron_indices_up, :].set(shifts[shift_indices_up])

    # Apply shifts to base positions
    r_up_carts_shifted = base_positions_up + shifts_to_apply_up  # Shape: (num_up_configs, N_up, 3)

    # Repeat down-spin electrons for up-spin configurations
    r_dn_carts_repeated_up = jnp.repeat(r_dn_carts[None, :, :], num_up_configs, axis=0)  # Shape: (num_up_configs, N_dn, 3)

    # Process down-spin electrons
    num_dn_electrons = r_dn_carts.shape[0]
    num_dn_configs = num_dn_electrons * num_shifts

    base_positions_dn = jnp.repeat(r_dn_carts[None, :, :], num_dn_configs, axis=0)  # Shape: (num_dn_configs, N_dn, 3)
    shifts_to_apply_dn = jnp.zeros_like(base_positions_dn)

    config_indices_dn = jnp.arange(num_dn_configs)
    electron_indices_dn = jnp.repeat(jnp.arange(num_dn_electrons), num_shifts)
    shift_indices_dn = jnp.tile(jnp.arange(num_shifts), num_dn_electrons)

    # Apply shifts to the appropriate electron in each configuration
    shifts_to_apply_dn = shifts_to_apply_dn.at[config_indices_dn, electron_indices_dn, :].set(shifts[shift_indices_dn])

    r_dn_carts_shifted = base_positions_dn + shifts_to_apply_dn  # Shape: (num_dn_configs, N_dn, 3)

    # Repeat up-spin electrons for down-spin configurations
    r_up_carts_repeated_dn = jnp.repeat(r_up_carts[None, :, :], num_dn_configs, axis=0)  # Shape: (num_dn_configs, N_up, 3)

    # Combine configurations
    r_up_carts_combined = jnp.concatenate([r_up_carts_shifted, r_up_carts_repeated_dn], axis=0)  # Shape: (N_configs, N_up, 3)
    r_dn_carts_combined = jnp.concatenate([r_dn_carts_repeated_up, r_dn_carts_shifted], axis=0)  # Shape: (N_configs, N_dn, 3)

    # Evaluate the wavefunction at the original positions
    jastrow_x = compute_Jastrow_part_jax(wavefunction_data.jastrow_data, r_up_carts, r_dn_carts)
    # Evaluate the wavefunction at the shifted positions using vectorization
    jastrow_xp = vmap(compute_Jastrow_part_jax, in_axes=(None, 0, 0))(
        wavefunction_data.jastrow_data, r_up_carts_combined, r_dn_carts_combined
    )
    # Evaluate the wavefunction at the original positions
    det_x = compute_det_geminal_all_elements_jax(wavefunction_data.geminal_data, r_up_carts, r_dn_carts)
    # Evaluate the wavefunction at the shifted positions using vectorization
    det_xp = vmap(compute_det_geminal_all_elements_jax, in_axes=(None, 0, 0))(
        wavefunction_data.geminal_data, r_up_carts_combined, r_dn_carts_combined
    )
    wf_ratio = jnp.exp(jastrow_xp - jastrow_x) * det_xp / det_x

    # Compute the kinetic part elements
    elements_kinetic_part = -1.0 / (2.0 * alat**2) * wf_ratio

    # Return the combined configurations and the kinetic elements
    return r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part


# no longer used in the main code
@jit
def compute_discretized_kinetic_energy_jax_fast_update(
    alat: float,
    wavefunction_data: Wavefunction_data,
    A_old_inv: jnp.ndarray,
    r_up_carts: jnp.ndarray,
    r_dn_carts: jnp.ndarray,
    RT: jnp.ndarray,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Function for computing discretized kinetic grid points and thier energies with a given lattice space (alat).

    Args:
        alat (float): Hamiltonian discretization (bohr), which will be replaced with LRDMC_data.
        wavefunction_data (Wavefunction_data): an instance of Qavefunction_data, which will be replaced with LRDMC_data.
        A_old_inv (npt.NDArray): the inverse of geminal matrix with (r_up_carts, r_dn_carts)
        r_up_carts (npt.NDArray): up electron position (N_e,3).
        r_dn_carts (npt.NDArray): down electron position (N_e,3).
        RT (npt.NDArray): Rotation matrix. \equiv R.T

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]:
            return mesh for the LRDMC kinetic part, npt.NDArrays, r_carts_up_arr and r_carts_dn_arr, whose dimensions
            are (N_grid, N_up, 3) and (N_grid, N_dn, 3), respectively. A (N_grid, 1) npt.NDArray \Psi(x')/\Psi(x)
            corresponding to the grid.
    """
    # Define the shifts to apply (+/- alat in each coordinate direction)
    shifts = alat * jnp.array(
        [
            [1, 0, 0],  # x+
            [-1, 0, 0],  # x-
            [0, 1, 0],  # y+
            [0, -1, 0],  # y-
            [0, 0, 1],  # z+
            [0, 0, -1],  # z-
        ]
    )  # Shape: (6, 3)

    shifts = shifts @ RT  # Shape: (6, 3)

    # num shift
    num_shifts = shifts.shape[0]

    # Process up-spin electrons
    num_up_electrons = r_up_carts.shape[0]
    num_up_configs = num_up_electrons * num_shifts

    # Create base positions repeated for each configuration
    base_positions_up = jnp.repeat(r_up_carts[None, :, :], num_up_configs, axis=0)  # Shape: (num_up_configs, N_up, 3)

    # Initialize shifts_to_apply_up
    shifts_to_apply_up = jnp.zeros_like(base_positions_up)

    # Create indices for configurations
    config_indices_up = jnp.arange(num_up_configs)
    electron_indices_up = jnp.repeat(jnp.arange(num_up_electrons), num_shifts)
    shift_indices_up = jnp.tile(jnp.arange(num_shifts), num_up_electrons)

    # Apply shifts to the appropriate electron in each configuration
    shifts_to_apply_up = shifts_to_apply_up.at[config_indices_up, electron_indices_up, :].set(shifts[shift_indices_up])

    # Apply shifts to base positions
    r_up_carts_shifted = base_positions_up + shifts_to_apply_up  # Shape: (num_up_configs, N_up, 3)

    # Repeat down-spin electrons for up-spin configurations
    r_dn_carts_repeated_up = jnp.repeat(r_dn_carts[None, :, :], num_up_configs, axis=0)  # Shape: (num_up_configs, N_dn, 3)

    # Process down-spin electrons
    num_dn_electrons = r_dn_carts.shape[0]
    num_dn_configs = num_dn_electrons * num_shifts

    base_positions_dn = jnp.repeat(r_dn_carts[None, :, :], num_dn_configs, axis=0)  # Shape: (num_dn_configs, N_dn, 3)
    shifts_to_apply_dn = jnp.zeros_like(base_positions_dn)

    config_indices_dn = jnp.arange(num_dn_configs)
    electron_indices_dn = jnp.repeat(jnp.arange(num_dn_electrons), num_shifts)
    shift_indices_dn = jnp.tile(jnp.arange(num_shifts), num_dn_electrons)

    # Apply shifts to the appropriate electron in each configuration
    shifts_to_apply_dn = shifts_to_apply_dn.at[config_indices_dn, electron_indices_dn, :].set(shifts[shift_indices_dn])

    r_dn_carts_shifted = base_positions_dn + shifts_to_apply_dn  # Shape: (num_dn_configs, N_dn, 3)

    # Repeat up-spin electrons for down-spin configurations
    r_up_carts_repeated_dn = jnp.repeat(r_up_carts[None, :, :], num_dn_configs, axis=0)  # Shape: (num_dn_configs, N_up, 3)

    # Combine configurations
    r_up_carts_combined = jnp.concatenate([r_up_carts_shifted, r_up_carts_repeated_dn], axis=0)  # Shape: (N_configs, N_up, 3)
    r_dn_carts_combined = jnp.concatenate([r_dn_carts_repeated_up, r_dn_carts_shifted], axis=0)  # Shape: (N_configs, N_dn, 3)

    # Evaluate the ratios of wavefunctions between the shifted positions and the original position
    wf_ratio = compute_ratio_determinant_part_jax(
        geminal_data=wavefunction_data.geminal_data,
        A_old_inv=A_old_inv,
        old_r_up_carts=r_up_carts,
        old_r_dn_carts=r_dn_carts,
        new_r_up_carts_arr=r_up_carts_combined,
        new_r_dn_carts_arr=r_dn_carts_combined,
    ) * compute_ratio_Jastrow_part_jax(
        jastrow_data=wavefunction_data.jastrow_data,
        old_r_up_carts=r_up_carts,
        old_r_dn_carts=r_dn_carts,
        new_r_up_carts_arr=r_up_carts_combined,
        new_r_dn_carts_arr=r_dn_carts_combined,
    )

    # Compute the kinetic part elements
    elements_kinetic_part = -1.0 / (2.0 * alat**2) * wf_ratio

    # Return the combined configurations and the kinetic elements
    return r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part


# no longer used in the main code
@jit
def compute_kinetic_energy_api_element_wise(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """The method is for computing kinetic energy of the given WF at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The kinetic energy with the given wavefunction (float | complex)
    """
    # grad_J_up, grad_J_dn, sum_laplacian_J = 0.0, 0.0, 0.0
    # """
    grad_J_up, grad_J_dn, sum_laplacian_J = compute_grads_and_laplacian_Jastrow_part_jax(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    # """

    # grad_ln_D_up, grad_ln_D_dn, sum_laplacian_ln_D = 0.0, 0.0, 0.0
    # """
    grad_ln_D_up, grad_ln_D_dn, sum_laplacian_ln_D = compute_grads_and_laplacian_ln_Det_jax(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    # """

    # compute kinetic energy
    L = (
        1.0
        / 2.0
        * (
            -(sum_laplacian_J + sum_laplacian_ln_D)
            - (
                jnp.sum((grad_J_up + grad_ln_D_up) * (grad_J_up + grad_ln_D_up))
                + jnp.sum((grad_J_dn + grad_ln_D_dn) * (grad_J_dn + grad_ln_D_dn))
            )
        )
    )

    return L


# no longer used in the main code
def compute_quantum_force_api(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The method is for computing quantum forces at (r_up_carts, r_dn_carts).

    Args:
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        The value of quantum forces of the given wavefunction -> return tuple[(N_e^{up}, 3), (N_e^{dn}, 3)]
    """
    grad_J_up, grad_J_dn, _ = 0, 0, 0  # tentative

    grad_ln_D_up, grad_ln_D_dn, _ = compute_grads_and_laplacian_ln_Det_jax(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    grad_ln_WF_up = grad_J_up + grad_ln_D_up
    grad_ln_WF_dn = grad_J_dn + grad_ln_D_dn

    return 2.0 * grad_ln_WF_up, 2.0 * grad_ln_WF_dn


if __name__ == "__main__":
    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    """
    # test jax grad
    grad_ln_Psi_h = grad(evaluate_ln_wavefunction_api, argnums=(0))(
        hamiltonian_data.wavefunction_data,
        r_up_carts,
        r_dn_carts,
    )

    grad_ln_Psi_jastrow2b_param_jax = grad_ln_Psi_h.jastrow_data.jastrow_two_body_data.jastrow_2b_param

    d_jastrow2b_param = 1.0e-5

    # WF data
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0 + d_jastrow2b_param)

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_two_body_pade_flag=True,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wavefunction_data
    )

    ln_Psi_h_p = evaluate_ln_wavefunction_api(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

    # WF data
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0 - d_jastrow2b_param)

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_two_body_pade_flag=True,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wavefunction_data
    )

    ln_Psi_h_m = evaluate_ln_wavefunction_api(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

    grad_ln_Psi_jastrow2b_param_fdm = (ln_Psi_h_p - ln_Psi_h_m) / (2.0 * d_jastrow2b_param)

    np.testing.assert_almost_equal(grad_ln_Psi_jastrow2b_param_fdm, grad_ln_Psi_jastrow2b_param_jax, decimal=6)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wavefunction_data
    )

    ln_Psi_h_m = evaluate_ln_wavefunction_api(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

    grad_ln_Psi_jastrow2b_param_fdm = (ln_Psi_h_p - ln_Psi_h_m) / (2.0 * d_jastrow2b_param)

    np.testing.assert_almost_equal(grad_ln_Psi_jastrow2b_param_fdm, grad_ln_Psi_jastrow2b_param_jax, decimal=6)
    """
