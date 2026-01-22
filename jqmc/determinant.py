"""Determinant module."""

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
from collections.abc import Callable
from logging import getLogger

# jqmc module
from typing import TYPE_CHECKING

# jax modules
# from jax.debug import print as jprint
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import jit, vmap

from .atomic_orbital import (
    AOs_cart_data,
    AOs_sphe_data,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
)
from .molecular_orbital import MOs_data, compute_MOs, compute_MOs_grad, compute_MOs_laplacian

if TYPE_CHECKING:  # pragma: no cover - typing-only import to avoid circular dependency
    from .wavefunction import VariationalParameterBlock

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


# @dataclass
@struct.dataclass
class Geminal_data:
    """Geminal (AGP) parameters and orbital references.

    Args:
        num_electron_up (int): Number of spin-up electrons.
        num_electron_dn (int): Number of spin-down electrons.
        orb_data_up_spin (AOs_data | MOs_data): Basis/orbitals for spin-up electrons.
        orb_data_dn_spin (AOs_data | MOs_data): Basis/orbitals for spin-down electrons.
        lambda_matrix (npt.NDArray | jax.Array): Geminal pairing matrix with shape
            ``(orb_num_up, orb_num_dn + num_electron_up - num_electron_dn)``.

    Notes:
        - For closed shells, ``orb_num_up == orb_num_dn`` and ``lambda_matrix`` is square.
        - For open shells, the right block encodes unpaired spin-up orbitals.
    """

    num_electron_up: int = struct.field(pytree_node=False, default=0)  #: Number of spin-up electrons.
    num_electron_dn: int = struct.field(pytree_node=False, default=0)  #: Number of spin-down electrons.
    orb_data_up_spin: AOs_sphe_data | AOs_cart_data | MOs_data = struct.field(
        pytree_node=True, default_factory=lambda: AOs_sphe_data()
    )  #: Orbital data (AOs or MOs) for spin-up electrons.
    orb_data_dn_spin: AOs_sphe_data | AOs_cart_data | MOs_data = struct.field(
        pytree_node=True, default_factory=lambda: AOs_sphe_data()
    )  #: Orbital data (AOs or MOs) for spin-down electrons.
    lambda_matrix: npt.NDArray | jax.Array = struct.field(
        pytree_node=True, default_factory=lambda: np.array([])
    )  #: Geminal pairing matrix; see class notes for expected shape.

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.orb_num_up != self.orb_num_dn:
            raise ValueError(
                f"The number of up and down orbitals ({self.orb_num_up}, {self.orb_num_dn}) should be the same such that the lambda_matrix is square."
            )
        if self.lambda_matrix.shape != (
            self.orb_num_up,
            self.orb_num_dn + (self.num_electron_up - self.num_electron_dn),
        ):
            raise ValueError(
                f"dim. of lambda_matrix = {self.lambda_matrix.shape} is imcompatible with the expected one "
                + f"= ({self.orb_num_up}, {self.orb_num_dn + (self.num_electron_up - self.num_electron_dn)}).",
            )
        if not isinstance(self.num_electron_up, (int, np.integer)):
            raise ValueError(f"num_electron_up = {type(self.num_electron_up)} must be an int.")
        if not isinstance(self.num_electron_dn, (int, np.integer)):
            raise ValueError(f"num_electron_dn = {type(self.num_electron_dn)} must be an int.")

        self.orb_data_up_spin.sanity_check()
        self.orb_data_dn_spin.sanity_check()

    def _get_info(self) -> list[str]:
        """Return a list of strings containing the information stored in the attributes."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  dim. of lambda_matrix = {self.lambda_matrix.shape}")
        info_lines.append(
            f"  lambda_matrix is symmetric? = {np.allclose(self.lambda_matrix[: self.orb_num_up, : self.orb_num_up], self.lambda_matrix[: self.orb_num_up, : self.orb_num_up].T)}"
        )
        lambda_matrix_paired, lambda_matrix_unpaired = np.hsplit(self.lambda_matrix, [self.orb_num_dn])
        info_lines.append(f"  lambda_matrix_paired.shape = {lambda_matrix_paired.shape}")
        info_lines.append(f"  lambda_matrix_unpaired.shape = {lambda_matrix_unpaired.shape}")
        info_lines.append(f"  num_electron_up = {self.num_electron_up}")
        info_lines.append(f"  num_electron_dn = {self.num_electron_dn}")
        info_lines.extend(self.orb_data_up_spin._get_info())
        info_lines.extend(self.orb_data_dn_spin._get_info())

        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    def apply_block_update(self, block: "VariationalParameterBlock") -> "Geminal_data":
        """Apply a single variational-parameter block update to this Geminal object.

        This method is the Geminal-specific counterpart of
        :meth:`Wavefunction_data.apply_block_updates`.  It receives a generic
        :class:`VariationalParameterBlock` whose ``values`` have already been
        updated (typically by ``block.apply_update`` inside the SR/MCMC driver),
        and interprets that block according to the structure of the geminal
        (lambda) matrix.

        Responsibilities of this method are:

        * Map the block name (currently ``"lambda_matrix"``) to the internal
          geminal parameters.
        * Handle the splitting of a rectangular lambda matrix into paired and
          unpaired parts when needed.
        * Enforce Geminal-specific structural constraints, especially the
          symmetry conditions on the paired block of the lambda matrix.

        All details about how the lambda parameters are stored and constrained
        live here (and in the surrounding ``Geminal_data`` class), not in
        :class:`VariationalParameterBlock` or in the optimizer.  This keeps the
        SR/MCMC machinery and the block abstraction structure-agnostic: adding
        new Geminal parameters should only require updating the block
        construction in ``Wavefunction_data.get_variational_blocks`` and adding
        the corresponding handling in this method.
        """
        if block.name != "lambda_matrix":
            return self

        lambda_old = np.array(self.lambda_matrix)
        lambda_new = np.array(block.values)

        # If the paired part of lambda_matrix is symmetric, keep it symmetric
        # after the update. The unpaired block (if any) is left as-is.
        if self.orb_num_up == self.orb_num_dn:
            # Full square matrix: check and enforce symmetry on the whole block.
            if np.allclose(lambda_old, lambda_old.T, atol=1e-8):
                lambda_new = 0.5 * (lambda_new + lambda_new.T)
        else:
            # Rectangular: split into paired (square) and unpaired parts.
            paired_old, unpaired_old = np.hsplit(lambda_old, [self.orb_num_dn])
            paired_new, unpaired_new = np.hsplit(lambda_new, [self.orb_num_dn])

            if np.allclose(paired_old, paired_old.T, atol=1e-8):
                paired_new = 0.5 * (paired_new + paired_new.T)

            lambda_new = np.hstack([paired_new, unpaired_new])

        return Geminal_data(
            num_electron_up=self.num_electron_up,
            num_electron_dn=self.num_electron_dn,
            orb_data_up_spin=self.orb_data_up_spin,
            orb_data_dn_spin=self.orb_data_dn_spin,
            lambda_matrix=lambda_new,
        )

    def accumulate_position_grad(self, grad_geminal: "Geminal_data"):
        """Aggregate position gradients from geminal-related structures."""
        grad = 0.0
        if hasattr(grad_geminal, "orb_data_up_spin"):
            grad += grad_geminal.orb_data_up_spin.structure_data.positions
        if hasattr(grad_geminal, "orb_data_dn_spin"):
            grad += grad_geminal.orb_data_dn_spin.structure_data.positions
        return grad

    def collect_param_grads(self, grad_geminal: "Geminal_data") -> dict[str, object]:
        """Collect parameter gradients into a flat dict keyed by block name."""
        grads: dict[str, any] = {}
        if hasattr(grad_geminal, "lambda_matrix"):
            grads["lambda_matrix"] = grad_geminal.lambda_matrix
        return grads

    @property
    def orb_num_up(self) -> int:
        """orb_num_up.

        The number of atomic orbitals or molecular orbitals for up electrons,
        depending on the instance stored in the attribute orb_data_up.

        Return:
            int: The number of atomic orbitals or molecular orbitals for up electrons.

        Raises:
            NotImplementedError:
                If the instance of orb_data_up_spin is neither AOs_data nor MOs_data.

        """
        if isinstance(self.orb_data_up_spin, AOs_sphe_data) or isinstance(self.orb_data_up_spin, AOs_cart_data):
            return self.orb_data_up_spin.num_ao
        elif isinstance(self.orb_data_up_spin, MOs_data):
            return self.orb_data_up_spin.num_mo
        else:
            raise NotImplementedError

    @property
    def orb_num_dn(self) -> int:
        """orb_num_dn.

        The number of atomic orbitals or molecular orbitals for down electrons,
        depending on the instance stored in the attribute orb_data_up.

        Return:
            int: The number of atomic orbitals or molecular orbitals for down electrons.

        Raises:
            NotImplementedError:
                If the instance of orb_data_dn_spin is neither AOs_data nor MOs_data.
        """
        if isinstance(self.orb_data_dn_spin, AOs_sphe_data) or isinstance(self.orb_data_dn_spin, AOs_cart_data):
            return self.orb_data_dn_spin.num_ao
        elif isinstance(self.orb_data_dn_spin, MOs_data):
            return self.orb_data_dn_spin.num_mo
        else:
            raise NotImplementedError

    @property
    def compute_orb_api(self) -> Callable[..., npt.NDArray[np.float64]]:
        """Function for computing AOs or MOs.

        The api method to compute AOs or MOs corresponding to instances
        stored in self.orb_data_up_spin and self.orb_data_dn_spin

        Return:
            Callable: The api method to compute AOs or MOs.

        Raises:
            NotImplementedError:
                If the instances of orb_data_up_spin/orb_data_dn_spin are
                neither AOs_data/AOs_data nor MOs_data/MOs_data.
        """
        if isinstance(self.orb_data_up_spin, AOs_sphe_data) and isinstance(self.orb_data_dn_spin, AOs_sphe_data):
            return compute_AOs
        elif isinstance(self.orb_data_up_spin, AOs_cart_data) and isinstance(self.orb_data_dn_spin, AOs_cart_data):
            return compute_AOs
        elif isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data):
            return compute_MOs
        else:
            raise NotImplementedError

    @property
    def compute_orb_grad_api(self) -> Callable[..., npt.NDArray[np.float64]]:
        """Function for computing AOs or MOs grads.

        The api method to compute AOs or MOs grads corresponding to instances
        stored in self.orb_data_up_spin and self.orb_data_dn_spin.

        Return:
            Callable: The api method to compute AOs or MOs grads.

        Raises:
            NotImplementedError:
                If the instances of orb_data_up_spin/orb_data_dn_spin are
                neither AOs_data/AOs_data nor MOs_data/MOs_data.
        """
        if isinstance(self.orb_data_up_spin, AOs_sphe_data) and isinstance(self.orb_data_dn_spin, AOs_sphe_data):
            return compute_AOs_grad
        elif isinstance(self.orb_data_up_spin, AOs_cart_data) and isinstance(self.orb_data_dn_spin, AOs_cart_data):
            return compute_AOs_grad
        elif isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data):
            return compute_MOs_grad
        else:
            raise NotImplementedError

    @property
    def compute_orb_laplacian_api(self) -> Callable[..., npt.NDArray[np.float64]]:
        """Function for computing AOs or MOs laplacians.

        The api method to compute AOs or MOs laplacians corresponding to instances
        stored in self.orb_data_up_spin and self.orb_data_dn_spin.

        Return:
            Callable: The api method to compute AOs or MOs laplacians.

        Raises:
            NotImplementedError:
                If the instances of orb_data_up_spin/orb_data_dn_spin are
                neither AOs_data/AOs_data nor MOs_data/MOs_data.
        """
        if isinstance(self.orb_data_up_spin, AOs_sphe_data) and isinstance(self.orb_data_dn_spin, AOs_sphe_data):
            return compute_AOs_laplacian
        elif isinstance(self.orb_data_up_spin, AOs_cart_data) and isinstance(self.orb_data_dn_spin, AOs_cart_data):
            return compute_AOs_laplacian
        elif isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data):
            return compute_MOs_laplacian
        else:
            raise NotImplementedError

    @classmethod
    def convert_from_MOs_to_AOs(cls, geminal_data: "Geminal_data") -> "Geminal_data":
        """Convert MOs to AOs."""
        if isinstance(geminal_data.orb_data_up_spin, AOs_sphe_data) and isinstance(
            geminal_data.orb_data_dn_spin, AOs_sphe_data
        ):
            return geminal_data
        elif isinstance(geminal_data.orb_data_up_spin, AOs_cart_data) and isinstance(
            geminal_data.orb_data_dn_spin, AOs_cart_data
        ):
            return geminal_data
        elif isinstance(geminal_data.orb_data_up_spin, MOs_data) and isinstance(geminal_data.orb_data_dn_spin, MOs_data):
            # split mo_lambda_matrix
            mo_lambda_matrix_paired, mo_lambda_matrix_unpaired = np.hsplit(
                geminal_data.lambda_matrix, [geminal_data.orb_num_dn]
            )

            # extract AOs data
            aos_data_up_spin = geminal_data.orb_data_up_spin.aos_data
            aos_data_dn_spin = geminal_data.orb_data_dn_spin.aos_data

            # convert MOs lambda to AO lambda
            aos_lambda_matrix_paired = np.dot(
                geminal_data.orb_data_up_spin.mo_coefficients.T,
                np.dot(mo_lambda_matrix_paired, geminal_data.orb_data_dn_spin.mo_coefficients),
            )
            aos_lambda_matrix_unpaired = np.dot(geminal_data.orb_data_up_spin.mo_coefficients.T, mo_lambda_matrix_unpaired)
            aos_lambda_matrix = np.hstack([aos_lambda_matrix_paired, aos_lambda_matrix_unpaired])
            return cls(
                geminal_data.num_electron_up,
                geminal_data.num_electron_dn,
                aos_data_up_spin,
                aos_data_dn_spin,
                aos_lambda_matrix,
            )
        else:
            raise NotImplementedError


@jax.custom_vjp
@jit
def compute_ln_det_geminal_all_elements(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    r"""Compute $\ln|\det G|$ for the geminal matrix.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of spin-down electrons with shape ``(N_dn, 3)``.

    Returns:
        float: Scalar log-determinant of the geminal matrix.
    """
    return jnp.log(
        jnp.abs(
            jnp.linalg.det(
                compute_geminal_all_elements(geminal_data=geminal_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
            )
        )
    )


# Forward pass for custom VJP.
def _ln_det_fwd(geminal_data, r_up_carts, r_dn_carts):
    """Forward pass for custom VJP.

    The custom derivative is needed for ln |Det(G)| because the jax native grad
    and hessian introduce numerical instability. The custom derivative exploits
    the LU decomposition of G instead of the direct inverse of G, achieving
    numerically stable calculations.

    Returns:
        - primal output: ln|det(G)|
        - residuals: (inputs and LU factors) for use in backward pass
    """
    G = compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)
    ln_det = jnp.log(jnp.abs(jnp.linalg.det(G)))
    # Compute LU decomposition: G = P @ L @ U
    P, L, U = jsp_linalg.lu(G)
    # We stash the original inputs plus the LU factors for the backward pass
    return ln_det, (geminal_data, r_up_carts, r_dn_carts, P, L, U)


# Backward pass for custom VJP.
def _ln_det_bwd(res, g):
    """Backward pass for custom VJP.

    The custom derivative is needed for ln |Det(G)| because the jax native grad
    and hessian introduce numerical instability. The custom derivative exploits
    the LU decomposition of G instead of the direct inverse of G, achieving
    numerically stable calculations.

    Args:
        res: residuals from forward pass
        g: cotangent of the primal output

    Returns:
        Gradients with respect to (geminal_data, r_up_carts, r_dn_carts)
    """
    geminal_data, r_up_carts, r_dn_carts, P, L, U = res

    # Build identity matrix of appropriate shape
    n = U.shape[0]
    I = jnp.eye(n, dtype=U.dtype)

    # Solve L @ Y = P^T @ I  for Y
    Y = jsp_linalg.solve_triangular(L, jnp.dot(P.T, I), lower=True)
    # Solve U @ X = Y for X, so that X = G^{-1}
    X = jsp_linalg.solve_triangular(U, Y, lower=False)

    # d ln|det G| / dG = (G^{-1})^T, scaled by incoming cotangent g
    grad_G = g * X.T

    # Now backpropagate through compute_geminal_all_elements_jax
    _, vjp_fun = jax.vjp(compute_geminal_all_elements, geminal_data, r_up_carts, r_dn_carts)
    # Apply VJP to produce gradients for each input
    return vjp_fun(grad_G)


# Register the custom VJP rule !!
compute_ln_det_geminal_all_elements.defvjp(_ln_det_fwd, _ln_det_bwd)


@jit
def compute_det_geminal_all_elements(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    r"""Compute $\det G$ for the geminal matrix.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of spin-down electrons with shape ``(N_dn, 3)``.

    Returns:
        float: Scalar determinant of the geminal matrix.
    """
    return jnp.linalg.det(compute_geminal_all_elements(geminal_data=geminal_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts))


def _compute_det_geminal_all_elements_debug(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> np.float64:
    """See compute_det_geminal_all_elements_api."""
    return np.linalg.det(
        _compute_geminal_all_elements_debug(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
    )


def compute_AS_regularization_factor_fast_update(
    geminal: npt.NDArray[np.float64], geminal_inv: npt.NDArray[np.float64]
) -> jax.Array:
    """Compute Attaccalite–Sorella regularization via fast update.

    Args:
        geminal: Geminal matrix with shape ``(N_up, N_up)``.
        geminal_inv: Inverse geminal matrix with shape ``(N_up, N_up)``.

    Returns:
        jax.Array: Scalar AS regularization factor.
    """
    # compute the AS factor
    theta = 3.0 / 8.0

    # compute F \equiv the square of Frobenius norm of geminal_inv
    F = jnp.sum(geminal_inv**2)

    # compute the scaling factor
    S = jnp.min(jnp.sum(geminal**2, axis=0))

    # compute R_AS
    R_AS = (S * F) ** (-theta)

    return R_AS


def _compute_AS_regularization_factor_debug(
    geminal_data: Geminal_data, r_up_carts: npt.NDArray[np.float64], r_dn_carts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """See compute_AS_regularization_factor_jax."""
    geminal = compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)

    # compute the AS factor
    theta = 3.0 / 8.0

    # compute F \equiv the square of Frobenius norm of geminal_inv
    geminal_inv = np.linalg.inv(geminal)
    F = np.sum(geminal_inv**2)

    # compute the scaling factor
    S = np.min(np.sum(geminal**2, axis=0))

    # compute R_AS
    R_AS = (S * F) ** (-theta)

    return R_AS


@jit
def compute_AS_regularization_factor(geminal_data: Geminal_data, r_up_carts: jax.Array, r_dn_carts: jax.Array) -> jax.Array:
    """Compute Attaccalite–Sorella regularization from electron coordinates.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of spin-down electrons with shape ``(N_dn, 3)``.

    Returns:
        jax.Array: Scalar AS regularization factor.
    """
    geminal = compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)

    # compute the AS factor
    theta = 3.0 / 8.0

    # compute F \equiv the square of Frobenius norm of geminal_inv
    sigma = jnp.linalg.svd(geminal, compute_uv=False)
    F = jnp.sum(1.0 / (sigma**2))

    # compute the scaling factor
    S = jnp.min(jnp.sum(geminal**2, axis=0))

    # compute R_AS
    R_AS = (S * F) ** (-theta)

    return R_AS


def compute_geminal_all_elements(geminal_data: Geminal_data, r_up_carts: jax.Array, r_dn_carts: jax.Array) -> jax.Array:
    """Compute geminal matrix $G$ for all electron pairs.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of spin-down electrons with shape ``(N_dn, 3)``.

    Returns:
        jax.Array: Geminal matrix with shape ``(N_up, N_up)`` combining paired and unpaired blocks.
    """
    if len(r_up_carts) != geminal_data.num_electron_up or len(r_dn_carts) != geminal_data.num_electron_dn:
        logger.info(
            f"Number of up and dn electrons (N_up, N_dn) = ({len(r_up_carts)}, {len(r_dn_carts)}) are not consistent "
            + f"with the expected values. (N_up, N_dn) = {geminal_data.num_electron_up}, {geminal_data.num_electron_dn})"
        )
        raise ValueError

    if len(r_up_carts) != len(r_dn_carts):
        if len(r_up_carts) - len(r_dn_carts) < 0:
            logger.error(
                f"Number of up electron is smaller than dn electrons. (N_up - N_dn = {len(r_up_carts) - len(r_dn_carts)})"
            )
            raise ValueError

    geminal = _compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)

    if geminal.shape != (len(r_up_carts), len(r_up_carts)):
        logger.error(
            f"geminal.shape = {geminal.shape} is inconsistent with the expected one = {(len(r_up_carts), len(r_up_carts))}"
        )
        raise ValueError

    return geminal


@jit
def _compute_geminal_all_elements(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    """See compute_geminal_all_elements_api."""
    lambda_matrix_paired, lambda_matrix_unpaired = jnp.hsplit(geminal_data.lambda_matrix, [geminal_data.orb_num_dn])

    orb_matrix_up = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_carts)
    orb_matrix_dn = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_carts)

    # compute geminal values
    geminal_paired = jnp.dot(orb_matrix_up.T, jnp.dot(lambda_matrix_paired, orb_matrix_dn))
    geminal_unpaired = jnp.dot(orb_matrix_up.T, lambda_matrix_unpaired)
    geminal = jnp.hstack([geminal_paired, geminal_unpaired])

    return geminal


def _compute_geminal_all_elements_debug(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """See compute_geminal_all_elements_api."""
    lambda_matrix_paired, lambda_matrix_unpaired = np.hsplit(geminal_data.lambda_matrix, [geminal_data.orb_num_dn])

    orb_matrix_up = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_carts)
    orb_matrix_dn = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_carts)

    # compute geminal values
    geminal_paired = np.dot(orb_matrix_up.T, np.dot(lambda_matrix_paired, orb_matrix_dn))
    geminal_unpaired = np.dot(orb_matrix_up.T, lambda_matrix_unpaired)
    geminal = np.hstack([geminal_paired, geminal_unpaired])

    return geminal


@jax.jit
def compute_geminal_up_one_row_elements(
    geminal_data,
    r_up_cart: jax.Array,  # shape: (3,) or (1,3)
    r_dn_carts: jax.Array,  # shape: (N_dn, 3)
) -> jax.Array:
    """Single row of the geminal matrix for one spin-up electron.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_cart: Cartesian coordinate for one spin-up electron with shape ``(3,)`` or ``(1, 3)``.
        r_dn_carts: Cartesian coordinates for all spin-down electrons with shape ``(N_dn, 3)``.

    Returns:
        jax.Array: Row vector with shape ``(N_dn + N_unpaired,)``.
    """
    # Split lambda into paired/unpaired blocks along columns
    lambda_matrix_paired, lambda_matrix_unpaired = jnp.hsplit(
        geminal_data.lambda_matrix, [geminal_data.orb_num_dn]
    )  # shapes: (n_orb_up, n_orb_dn), (n_orb_up, num_unpaired)

    # Orbital values:
    # - up: single position -> 1D vector (n_orb_up,)
    # - dn: batched positions -> (n_orb_dn, N_dn)
    orb_up_vec = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_cart)
    orb_up_vec = jnp.reshape(orb_up_vec, (-1,))  # ensure (n_orb_up,)
    orb_matrix_dn = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_carts)
    # ensure (n_orb_dn, N_dn)
    orb_matrix_dn = jnp.asarray(orb_matrix_dn)

    # Paired block row:  (n_orb_up,) @ (n_orb_up, N_dn) -> (N_dn,)
    paired_right = lambda_matrix_paired @ orb_matrix_dn  # (n_orb_up, N_dn)
    row_paired = orb_up_vec @ paired_right  # (N_dn,)

    # Unpaired block row: (n_orb_up,) @ (n_orb_up, num_unpaired) -> (num_unpaired,)
    row_unpaired = orb_up_vec @ lambda_matrix_unpaired  # (num_unpaired,)

    # Concatenate horizontally to match the full geminal row
    row = jnp.hstack([row_paired, row_unpaired])  # (N_dn + num_unpaired,)
    return row


@jax.jit
def compute_geminal_dn_one_column_elements(
    geminal_data,
    r_up_carts: jax.Array,  # shape: (N_up, 3)
    r_dn_cart: jax.Array,  # shape: (3,) or (1,3)
) -> jax.Array:
    """Single column of the geminal matrix for one spin-down electron.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_cart: Cartesian coordinate for one spin-down electron with shape ``(3,)`` or ``(1, 3)``.

    Returns:
        jax.Array: Column vector for the paired block with shape ``(N_up,)``.
    """
    # Split lambda into paired/unpaired blocks along columns
    lambda_matrix_paired, _lambda_matrix_unpaired = jnp.hsplit(
        geminal_data.lambda_matrix, [geminal_data.orb_num_dn]
    )  # lambda_matrix_paired: (n_orb_up, n_orb_dn)

    # Orbital values:
    # - up: batched positions -> (n_orb_up, N_up)
    # - dn: single position -> 1D vector (n_orb_dn,)
    orb_matrix_up = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_carts)
    orb_matrix_up = jnp.asarray(orb_matrix_up)  # (n_orb_up, N_up)

    orb_dn_vec = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_cart)
    orb_dn_vec = jnp.reshape(orb_dn_vec, (-1,))  # (n_orb_dn,)

    # Column of paired block:
    # w = (n_orb_up, n_orb_dn) @ (n_orb_dn,) -> (n_orb_up,)
    w = lambda_matrix_paired @ orb_dn_vec  # (n_orb_up,)
    # col = (N_up, n_orb_up) @ (n_orb_up,) -> (N_up,)
    col = orb_matrix_up.T @ w  # (N_up,)

    return col


@jit
def compute_ratio_determinant_part(
    geminal_data: Geminal_data,
    A_old_inv: jax.Array,
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_up_carts_arr: jax.Array,
    new_r_dn_carts_arr: jax.Array,
) -> jax.Array:
    r"""Determinant ratio $\det G(\mathbf r')/\det G(\mathbf r)$ for batched moves.

    Args:
        geminal_data: Geminal parameters and orbital references.
        A_old_inv: Inverse geminal matrix for the reference configuration with shape ``(N_up, N_up)``.
        old_r_up_carts: Original spin-up electron coordinates with shape ``(N_up, 3)``.
        old_r_dn_carts: Original spin-down electron coordinates with shape ``(N_dn, 3)``.
        new_r_up_carts_arr: Proposed spin-up coordinates per grid with shape ``(N_grid, N_up, 3)``.
        new_r_dn_carts_arr: Proposed spin-down coordinates per grid with shape ``(N_grid, N_dn, 3)``.

    Returns:
        jax.Array: Determinant ratios per grid with shape ``(N_grid,)``.
    """
    # split, geminal_data.lambda_matrix
    lambda_matrix_paired, lambda_matrix_unpaired = jnp.split(
        geminal_data.lambda_matrix, indices_or_sections=[geminal_data.orb_num_dn], axis=1
    )

    num_up = old_r_up_carts.shape[0]
    num_dn = old_r_dn_carts.shape[0]
    if num_up == 0 or num_dn == 0:
        det_x = compute_det_geminal_all_elements(geminal_data, old_r_up_carts, old_r_dn_carts)
        det_xp = vmap(compute_det_geminal_all_elements, in_axes=(None, 0, 0))(
            geminal_data, new_r_up_carts_arr, new_r_dn_carts_arr
        )
        return det_xp / det_x

    orb_matrix_up_old = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, old_r_up_carts)
    orb_matrix_dn_old = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, old_r_dn_carts)

    def compute_one_grid(
        orb_matrix_up_old,
        orb_matrix_dn_old,
        A_old_inv,
        lambda_matrix_paired,
        lambda_matrix_unpaired,
        new_r_up_carts,
        new_r_dn_carts,
        old_r_up_carts,
        old_r_dn_carts,
    ):
        delta_up = new_r_up_carts - old_r_up_carts
        delta_dn = new_r_dn_carts - old_r_dn_carts
        num_up = old_r_up_carts.shape[0]
        num_dn = old_r_dn_carts.shape[0]
        if num_up == 0:
            up_all_zero = True
            diff = delta_dn
        elif num_dn == 0:
            up_all_zero = False
            diff = delta_up
        else:
            up_all_zero = jnp.all(delta_up == 0.0)
            diff = jax.lax.cond(up_all_zero, lambda _: delta_dn, lambda _: delta_up, operand=None)
        nonzero_in_rows = jnp.any(diff != 0.0, axis=1)
        idx = jnp.argmax(nonzero_in_rows)

        def up_case(A_old_inv, idx, lambda_matrix_paired, lambda_matrix_unpaired, new_r_up_carts, new_r_dn_carts):
            new_r_up_carts_extracted = jnp.expand_dims(new_r_up_carts[idx], axis=0)  # shape=(1,3)
            A_old_inv_vec = jnp.expand_dims(A_old_inv[:, idx], axis=1)

            # orb
            orb_matrix_up = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, new_r_up_carts_extracted)
            orb_matrix_dn = orb_matrix_dn_old
            # geminal
            geminal_paired = jnp.dot(orb_matrix_up.T, jnp.dot(lambda_matrix_paired, orb_matrix_dn))
            geminal_unpaired = jnp.dot(orb_matrix_up.T, lambda_matrix_unpaired)
            geminal = jnp.hstack([geminal_paired, geminal_unpaired])

            return jnp.dot(geminal, A_old_inv_vec)[0][0]

        def dn_case(A_old_inv, idx, lambda_matrix_paired, lambda_matrix_unpaired, new_r_up_carts, new_r_dn_carts):
            new_r_dn_carts_extracted = jnp.expand_dims(new_r_dn_carts[idx], axis=0)  # shape=(1,3)
            A_old_inv_vec = jnp.expand_dims(A_old_inv[idx, :], axis=0)

            # orb
            orb_matrix_up = orb_matrix_up_old
            orb_matrix_dn = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, new_r_dn_carts_extracted)
            # geminal
            geminal_paired = jnp.dot(orb_matrix_up.T, jnp.dot(lambda_matrix_paired, orb_matrix_dn))
            geminal_unpaired = jnp.dot(orb_matrix_up.T, lambda_matrix_unpaired)
            geminal = jnp.hstack([geminal_paired, geminal_unpaired])

            return jnp.dot(A_old_inv_vec, geminal)[0][0]

        return jax.lax.cond(
            up_all_zero,
            dn_case,
            up_case,
            *(A_old_inv, idx, lambda_matrix_paired, lambda_matrix_unpaired, new_r_up_carts, new_r_dn_carts),
        )

    # vectorization along grid
    determinant_ratios = vmap(compute_one_grid, in_axes=(None, None, None, None, None, 0, 0, None, None))(
        orb_matrix_up_old,
        orb_matrix_dn_old,
        A_old_inv,
        lambda_matrix_paired,
        lambda_matrix_unpaired,
        new_r_up_carts_arr,
        new_r_dn_carts_arr,
        old_r_up_carts,
        old_r_dn_carts,
    )
    return determinant_ratios


def _compute_ratio_determinant_part_debug(
    geminal_data: Geminal_data,
    old_r_up_carts: npt.NDArray[np.float64],
    old_r_dn_carts: npt.NDArray[np.float64],
    new_r_up_carts_arr: npt.NDArray[np.float64],
    new_r_dn_carts_arr: npt.NDArray[np.float64],
) -> npt.NDArray:
    """See _api method."""
    return np.array(
        [
            compute_det_geminal_all_elements(geminal_data, new_r_up_carts, new_r_dn_carts)
            / compute_det_geminal_all_elements(geminal_data, old_r_up_carts, old_r_dn_carts)
            for new_r_up_carts, new_r_dn_carts in zip(new_r_up_carts_arr, new_r_dn_carts_arr, strict=True)
        ]
    )


@jit
def compute_grads_and_laplacian_ln_Det(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    r"""Gradients and Laplacians of $\ln\det G$ for each electron.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of spin-down electrons with shape ``(N_dn, 3)``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            - Gradients for spin-up electrons with shape ``(N_up, 3)``.
            - Gradients for spin-down electrons with shape ``(N_dn, 3)``.
            - Laplacians for spin-up electrons with shape ``(N_up,)``.
            - Laplacians for spin-down electrons with shape ``(N_dn,)``.
    """
    lambda_matrix_paired, lambda_matrix_unpaired = jnp.hsplit(geminal_data.lambda_matrix, [geminal_data.orb_num_dn])

    ao_matrix_up = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_carts)
    ao_matrix_dn = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_carts)

    ao_matrix_up_grad_x, ao_matrix_up_grad_y, ao_matrix_up_grad_z = geminal_data.compute_orb_grad_api(
        geminal_data.orb_data_up_spin, r_up_carts
    )
    ao_matrix_dn_grad_x, ao_matrix_dn_grad_y, ao_matrix_dn_grad_z = geminal_data.compute_orb_grad_api(
        geminal_data.orb_data_dn_spin, r_dn_carts
    )
    ao_matrix_laplacian_up = geminal_data.compute_orb_laplacian_api(geminal_data.orb_data_up_spin, r_up_carts)
    ao_matrix_laplacian_dn = geminal_data.compute_orb_laplacian_api(geminal_data.orb_data_dn_spin, r_dn_carts)

    geminal_paired = jnp.dot(ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn))
    geminal_unpaired = jnp.dot(ao_matrix_up.T, lambda_matrix_unpaired)
    geminal = jnp.hstack([geminal_paired, geminal_unpaired])

    geminal_grad_up_x_paired = jnp.dot(ao_matrix_up_grad_x.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn))
    geminal_grad_up_x_unpaired = jnp.dot(ao_matrix_up_grad_x.T, lambda_matrix_unpaired)
    geminal_grad_up_x = jnp.hstack([geminal_grad_up_x_paired, geminal_grad_up_x_unpaired])

    geminal_grad_up_y_paired = jnp.dot(ao_matrix_up_grad_y.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn))
    geminal_grad_up_y_unpaired = jnp.dot(ao_matrix_up_grad_y.T, lambda_matrix_unpaired)
    geminal_grad_up_y = jnp.hstack([geminal_grad_up_y_paired, geminal_grad_up_y_unpaired])

    geminal_grad_up_z_paired = jnp.dot(ao_matrix_up_grad_z.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn))
    geminal_grad_up_z_unpaired = jnp.dot(ao_matrix_up_grad_z.T, lambda_matrix_unpaired)
    geminal_grad_up_z = jnp.hstack([geminal_grad_up_z_paired, geminal_grad_up_z_unpaired])

    geminal_laplacian_up_paired = jnp.dot(ao_matrix_laplacian_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn))
    geminal_laplacian_up_unpaired = jnp.dot(ao_matrix_laplacian_up.T, lambda_matrix_unpaired)
    geminal_laplacian_up = jnp.hstack([geminal_laplacian_up_paired, geminal_laplacian_up_unpaired])

    geminal_grad_dn_x_paired = jnp.dot(ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn_grad_x))
    geminal_grad_dn_x_unpaired = jnp.zeros(
        [
            geminal_data.num_electron_up,
            geminal_data.num_electron_up - geminal_data.num_electron_dn,
        ]
    )
    geminal_grad_dn_x = jnp.hstack([geminal_grad_dn_x_paired, geminal_grad_dn_x_unpaired])

    geminal_grad_dn_y_paired = jnp.dot(ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn_grad_y))
    geminal_grad_dn_y_unpaired = jnp.zeros(
        [
            geminal_data.num_electron_up,
            geminal_data.num_electron_up - geminal_data.num_electron_dn,
        ]
    )
    geminal_grad_dn_y = jnp.hstack([geminal_grad_dn_y_paired, geminal_grad_dn_y_unpaired])

    geminal_grad_dn_z_paired = jnp.dot(ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn_grad_z))
    geminal_grad_dn_z_unpaired = jnp.zeros(
        [
            geminal_data.num_electron_up,
            geminal_data.num_electron_up - geminal_data.num_electron_dn,
        ]
    )
    geminal_grad_dn_z = jnp.hstack([geminal_grad_dn_z_paired, geminal_grad_dn_z_unpaired])

    geminal_laplacian_dn_paired = jnp.dot(ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_laplacian_dn))
    geminal_laplacian_dn_unpaired = jnp.zeros(
        [
            geminal_data.num_electron_up,
            geminal_data.num_electron_up - geminal_data.num_electron_dn,
        ]
    )
    geminal_laplacian_dn = jnp.hstack([geminal_laplacian_dn_paired, geminal_laplacian_dn_unpaired])

    P, L, U = jsp_linalg.lu(geminal)
    n = geminal.shape[0]
    I = jnp.eye(n, dtype=geminal.dtype)
    Y = jsp_linalg.solve_triangular(L, jnp.dot(P.T, I), lower=True)
    geminal_inverse = jsp_linalg.solve_triangular(U, Y, lower=False)

    grad_ln_D_up_x = jnp.einsum("ij,ji->i", geminal_grad_up_x, geminal_inverse)
    grad_ln_D_up_y = jnp.einsum("ij,ji->i", geminal_grad_up_y, geminal_inverse)
    grad_ln_D_up_z = jnp.einsum("ij,ji->i", geminal_grad_up_z, geminal_inverse)
    grad_ln_D_dn_x = jnp.einsum("ij,ji->i", geminal_inverse, geminal_grad_dn_x)
    grad_ln_D_dn_y = jnp.einsum("ij,ji->i", geminal_inverse, geminal_grad_dn_y)
    grad_ln_D_dn_z = jnp.einsum("ij,ji->i", geminal_inverse, geminal_grad_dn_z)

    grad_ln_D_up = jnp.array([grad_ln_D_up_x, grad_ln_D_up_y, grad_ln_D_up_z]).T
    grad_ln_D_dn = jnp.array([grad_ln_D_dn_x, grad_ln_D_dn_y, grad_ln_D_dn_z]).T

    lap_ln_D_up = -(
        grad_ln_D_up_x * grad_ln_D_up_x + grad_ln_D_up_y * grad_ln_D_up_y + grad_ln_D_up_z * grad_ln_D_up_z
    ) + jnp.einsum("ij,ji->i", geminal_laplacian_up, geminal_inverse)

    lap_ln_D_dn = -(
        grad_ln_D_dn_x * grad_ln_D_dn_x + grad_ln_D_dn_y * grad_ln_D_dn_y + grad_ln_D_dn_z * grad_ln_D_dn_z
    ) + jnp.einsum("ij,ji->i", geminal_inverse, geminal_laplacian_dn)

    return grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn


@jit
def compute_grads_and_laplacian_ln_Det_fast(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inverse: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Gradients and Laplacians of ln det G using a precomputed inverse.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of spin-down electrons with shape ``(N_dn, 3)``.
        geminal_inverse: Precomputed inverse of the geminal matrix ``G``.

    Returns:
        Gradients (up/down) and Laplacians (up/down) of ln det G per electron.
    """
    if geminal_inverse is None:
        raise ValueError("geminal_inverse must be provided for fast evaluation")

    lambda_matrix_paired, lambda_matrix_unpaired = jnp.hsplit(geminal_data.lambda_matrix, [geminal_data.orb_num_dn])

    ao_matrix_up = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_carts)
    ao_matrix_dn = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_carts)

    ao_matrix_up_grad_x, ao_matrix_up_grad_y, ao_matrix_up_grad_z = geminal_data.compute_orb_grad_api(
        geminal_data.orb_data_up_spin, r_up_carts
    )
    ao_matrix_dn_grad_x, ao_matrix_dn_grad_y, ao_matrix_dn_grad_z = geminal_data.compute_orb_grad_api(
        geminal_data.orb_data_dn_spin, r_dn_carts
    )
    ao_matrix_laplacian_up = geminal_data.compute_orb_laplacian_api(geminal_data.orb_data_up_spin, r_up_carts)
    ao_matrix_laplacian_dn = geminal_data.compute_orb_laplacian_api(geminal_data.orb_data_dn_spin, r_dn_carts)

    geminal_grad_up_x_paired = jnp.dot(ao_matrix_up_grad_x.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn))
    geminal_grad_up_x_unpaired = jnp.dot(ao_matrix_up_grad_x.T, lambda_matrix_unpaired)
    geminal_grad_up_x = jnp.hstack([geminal_grad_up_x_paired, geminal_grad_up_x_unpaired])

    geminal_grad_up_y_paired = jnp.dot(ao_matrix_up_grad_y.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn))
    geminal_grad_up_y_unpaired = jnp.dot(ao_matrix_up_grad_y.T, lambda_matrix_unpaired)
    geminal_grad_up_y = jnp.hstack([geminal_grad_up_y_paired, geminal_grad_up_y_unpaired])

    geminal_grad_up_z_paired = jnp.dot(ao_matrix_up_grad_z.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn))
    geminal_grad_up_z_unpaired = jnp.dot(ao_matrix_up_grad_z.T, lambda_matrix_unpaired)
    geminal_grad_up_z = jnp.hstack([geminal_grad_up_z_paired, geminal_grad_up_z_unpaired])

    geminal_grad_dn_x_paired = jnp.dot(ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn_grad_x))
    geminal_grad_dn_x_unpaired = jnp.zeros(
        [geminal_data.num_electron_up, geminal_data.num_electron_up - geminal_data.num_electron_dn]
    )
    geminal_grad_dn_x = jnp.hstack([geminal_grad_dn_x_paired, geminal_grad_dn_x_unpaired])

    geminal_grad_dn_y_paired = jnp.dot(ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn_grad_y))
    geminal_grad_dn_y_unpaired = jnp.zeros(
        [geminal_data.num_electron_up, geminal_data.num_electron_up - geminal_data.num_electron_dn]
    )
    geminal_grad_dn_y = jnp.hstack([geminal_grad_dn_y_paired, geminal_grad_dn_y_unpaired])

    geminal_grad_dn_z_paired = jnp.dot(ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn_grad_z))
    geminal_grad_dn_z_unpaired = jnp.zeros(
        [geminal_data.num_electron_up, geminal_data.num_electron_up - geminal_data.num_electron_dn]
    )
    geminal_grad_dn_z = jnp.hstack([geminal_grad_dn_z_paired, geminal_grad_dn_z_unpaired])

    geminal_laplacian_up_paired = jnp.dot(ao_matrix_laplacian_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_dn))
    geminal_laplacian_up_unpaired = jnp.dot(ao_matrix_laplacian_up.T, lambda_matrix_unpaired)
    geminal_laplacian_up = jnp.hstack([geminal_laplacian_up_paired, geminal_laplacian_up_unpaired])

    geminal_laplacian_dn_paired = jnp.dot(ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_laplacian_dn))
    geminal_laplacian_dn_unpaired = jnp.zeros(
        [geminal_data.num_electron_up, geminal_data.num_electron_up - geminal_data.num_electron_dn]
    )
    geminal_laplacian_dn = jnp.hstack([geminal_laplacian_dn_paired, geminal_laplacian_dn_unpaired])

    grad_ln_D_up_x = jnp.einsum("ij,ji->i", geminal_grad_up_x, geminal_inverse)
    grad_ln_D_up_y = jnp.einsum("ij,ji->i", geminal_grad_up_y, geminal_inverse)
    grad_ln_D_up_z = jnp.einsum("ij,ji->i", geminal_grad_up_z, geminal_inverse)
    grad_ln_D_dn_x = jnp.einsum("ij,ji->i", geminal_inverse, geminal_grad_dn_x)
    grad_ln_D_dn_y = jnp.einsum("ij,ji->i", geminal_inverse, geminal_grad_dn_y)
    grad_ln_D_dn_z = jnp.einsum("ij,ji->i", geminal_inverse, geminal_grad_dn_z)

    grad_ln_D_up = jnp.array([grad_ln_D_up_x, grad_ln_D_up_y, grad_ln_D_up_z]).T
    grad_ln_D_dn = jnp.array([grad_ln_D_dn_x, grad_ln_D_dn_y, grad_ln_D_dn_z]).T

    lap_ln_D_up = -(
        grad_ln_D_up_x * grad_ln_D_up_x + grad_ln_D_up_y * grad_ln_D_up_y + grad_ln_D_up_z * grad_ln_D_up_z
    ) + jnp.einsum("ij,ji->i", geminal_laplacian_up, geminal_inverse)

    lap_ln_D_dn = -(
        grad_ln_D_dn_x * grad_ln_D_dn_x + grad_ln_D_dn_y * grad_ln_D_dn_y + grad_ln_D_dn_z * grad_ln_D_dn_z
    ) + jnp.einsum("ij,ji->i", geminal_inverse, geminal_laplacian_dn)

    return grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn


def _compute_grads_and_laplacian_ln_Det_fast_debug(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Debug helper that builds geminal inverse then calls the fast path."""
    # Reuse the fast path for gradients/Laplacians
    grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = compute_grads_and_laplacian_ln_Det(
        geminal_data=geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn


@jit
def _compute_grads_and_laplacian_ln_Det_auto(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Auto-diff version of grads and laplacian of ln Det.

    Uses autodiff on ln|det(G)| to compute gradients w.r.t. electron positions
    and per-electron Laplacians.
    """

    def ln_det_fn(r_up, r_dn):
        return compute_ln_det_geminal_all_elements(geminal_data, r_up, r_dn)

    grad_ln_D_up = jax.grad(ln_det_fn, argnums=0)(r_up_carts, r_dn_carts)
    grad_ln_D_dn = jax.grad(ln_det_fn, argnums=1)(r_up_carts, r_dn_carts)

    def grad_up_fn(r_up):
        return jax.grad(ln_det_fn, argnums=0)(r_up, r_dn_carts)

    def grad_dn_fn(r_dn):
        return jax.grad(ln_det_fn, argnums=1)(r_up_carts, r_dn)

    jac_up = jax.jacfwd(grad_up_fn)(r_up_carts)
    jac_dn = jax.jacfwd(grad_dn_fn)(r_dn_carts)

    laplacian_ln_D_up = jnp.einsum("ijij->i", jac_up)
    laplacian_ln_D_dn = jnp.einsum("ijij->i", jac_dn)

    return grad_ln_D_up, grad_ln_D_dn, laplacian_ln_D_up, laplacian_ln_D_dn


def _compute_grads_and_laplacian_ln_Det_debug(
    geminal_data: Geminal_data,
    r_up_carts: np.ndarray,
    r_dn_carts: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """See compute_grads_and_laplacian_ln_Det_api."""
    det_geminal = compute_det_geminal_all_elements(
        geminal_data=geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    #############################################################
    # Gradients part
    #############################################################

    diff_h = 1.0e-5  # for grad

    # grad up
    grad_x_up = []
    grad_y_up = []
    grad_z_up = []

    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up2_carts = r_up_carts.copy()
        diff_p_y_r_up2_carts = r_up_carts.copy()
        diff_p_z_r_up2_carts = r_up_carts.copy()
        diff_p_x_r_up2_carts[r_i][0] += diff_h
        diff_p_y_r_up2_carts[r_i][1] += diff_h
        diff_p_z_r_up2_carts[r_i][2] += diff_h

        det_geminal_p_x_up2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=diff_p_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        det_geminal_p_y_up2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=diff_p_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        det_geminal_p_z_up2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=diff_p_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        diff_m_x_r_up2_carts = r_up_carts.copy()
        diff_m_y_r_up2_carts = r_up_carts.copy()
        diff_m_z_r_up2_carts = r_up_carts.copy()
        diff_m_x_r_up2_carts[r_i][0] -= diff_h
        diff_m_y_r_up2_carts[r_i][1] -= diff_h
        diff_m_z_r_up2_carts[r_i][2] -= diff_h

        det_geminal_m_x_up2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=diff_m_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        det_geminal_m_y_up2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=diff_m_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        det_geminal_m_z_up2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=diff_m_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        """ mathematically correct, but numerically unstable.
        grad_x_up.append(
            (np.log(np.abs(det_geminal_p_x_up2)) - np.log(np.abs(det_geminal_m_x_up2)))
            / (2.0 * diff_h)
        )
        grad_y_up.append(
            (np.log(np.abs(det_geminal_p_y_up2)) - np.log(np.abs(det_geminal_m_y_up2)))
            / (2.0 * diff_h)
        )
        grad_z_up.append(
            (np.log(np.abs(det_geminal_p_z_up2)) - np.log(np.abs(det_geminal_m_z_up2)))
            / (2.0 * diff_h)
        )
        """

        # compute f'(x)
        grad_x_up.append((det_geminal_p_x_up2 - det_geminal_m_x_up2) / (2.0 * diff_h))
        grad_y_up.append((det_geminal_p_y_up2 - det_geminal_m_y_up2) / (2.0 * diff_h))
        grad_z_up.append((det_geminal_p_z_up2 - det_geminal_m_z_up2) / (2.0 * diff_h))

    # grad dn
    grad_x_dn = []
    grad_y_dn = []
    grad_z_dn = []

    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn2_carts = r_dn_carts.copy()
        diff_p_y_r_dn2_carts = r_dn_carts.copy()
        diff_p_z_r_dn2_carts = r_dn_carts.copy()
        diff_p_x_r_dn2_carts[r_i][0] += diff_h
        diff_p_y_r_dn2_carts[r_i][1] += diff_h
        diff_p_z_r_dn2_carts[r_i][2] += diff_h

        det_geminal_p_x_dn2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn2_carts,
        )
        det_geminal_p_y_dn2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn2_carts,
        )
        det_geminal_p_z_dn2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn2_carts,
        )

        diff_m_x_r_dn2_carts = r_dn_carts.copy()
        diff_m_y_r_dn2_carts = r_dn_carts.copy()
        diff_m_z_r_dn2_carts = r_dn_carts.copy()
        diff_m_x_r_dn2_carts[r_i][0] -= diff_h
        diff_m_y_r_dn2_carts[r_i][1] -= diff_h
        diff_m_z_r_dn2_carts[r_i][2] -= diff_h

        det_geminal_m_x_dn2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn2_carts,
        )
        det_geminal_m_y_dn2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn2_carts,
        )
        det_geminal_m_z_dn2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn2_carts,
        )

        """ mathematically correct, but numerically unstable.
        grad_x_dn.append(
            (np.log(np.abs(det_geminal_p_x_dn2)) - np.log(np.abs(det_geminal_m_x_dn2)))
            / (2.0 * diff_h)
        )
        grad_y_dn.append(
            (np.log(np.abs(det_geminal_p_y_dn2)) - np.log(np.abs(det_geminal_m_y_dn2)))
            / (2.0 * diff_h)
        )
        grad_z_dn.append(
            (np.log(np.abs(det_geminal_p_z_dn2)) - np.log(np.abs(det_geminal_m_z_dn2)))
            / (2.0 * diff_h)
        )
        """

        # compute f'(x)
        grad_x_dn.append((det_geminal_p_x_dn2 - det_geminal_m_x_dn2) / (2.0 * diff_h))
        grad_y_dn.append((det_geminal_p_y_dn2 - det_geminal_m_y_dn2) / (2.0 * diff_h))
        grad_z_dn.append((det_geminal_p_z_dn2 - det_geminal_m_z_dn2) / (2.0 * diff_h))

    # since d/dx ln |f(x)| = f'(x) / f(x)
    grad_ln_D_up = np.array([grad_x_up, grad_y_up, grad_z_up]).T / det_geminal
    grad_ln_D_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn]).T / det_geminal

    #############################################################
    # Laplacian part
    #############################################################

    diff_h2 = 1.0e-4  # for laplacian

    laplacian_ln_D_up = np.zeros(len(r_up_carts))
    laplacian_ln_D_dn = np.zeros(len(r_dn_carts))

    # laplacians up
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up2_carts = r_up_carts.copy()
        diff_p_y_r_up2_carts = r_up_carts.copy()
        diff_p_z_r_up2_carts = r_up_carts.copy()
        diff_p_x_r_up2_carts[r_i][0] += diff_h2
        diff_p_y_r_up2_carts[r_i][1] += diff_h2
        diff_p_z_r_up2_carts[r_i][2] += diff_h2

        det_geminal_p_x_up2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=diff_p_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        det_geminal_p_y_up2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=diff_p_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        det_geminal_p_z_up2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=diff_p_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        diff_m_x_r_up2_carts = r_up_carts.copy()
        diff_m_y_r_up2_carts = r_up_carts.copy()
        diff_m_z_r_up2_carts = r_up_carts.copy()
        diff_m_x_r_up2_carts[r_i][0] -= diff_h2
        diff_m_y_r_up2_carts[r_i][1] -= diff_h2
        diff_m_z_r_up2_carts[r_i][2] -= diff_h2

        det_geminal_m_x_up2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=diff_m_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        det_geminal_m_y_up2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=diff_m_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        det_geminal_m_z_up2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=diff_m_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        """ mathematically correct, but numerically unstablle
        gradgrad_x_up = (
            np.log(np.abs(det_geminal_p_x_up2))
            + np.log(np.abs(det_geminal_m_x_up2))
            - 2.0 * np.log(np.abs(det_geminal))
        ) / (diff_h2**2)

        gradgrad_y_up = (
            np.log(np.abs(det_geminal_p_y_up2))
            + np.log(np.abs(det_geminal_m_y_up2))
            - 2.0 * np.log(np.abs(det_geminal))
        ) / (diff_h2**2)

        gradgrad_z_up = (
            np.log(np.abs(det_geminal_p_z_up2))
            + np.log(np.abs(det_geminal_m_z_up2))
            - 2.0 * np.log(np.abs(det_geminal))
        ) / (diff_h2**2)
        """

        # compute f''(x)
        gradgrad_x_up = (det_geminal_p_x_up2 + det_geminal_m_x_up2 - 2.0 * det_geminal) / (diff_h2**2)

        gradgrad_y_up = (det_geminal_p_y_up2 + det_geminal_m_y_up2 - 2.0 * det_geminal) / (diff_h2**2)

        gradgrad_z_up = (det_geminal_p_z_up2 + det_geminal_m_z_up2 - 2.0 * det_geminal) / (diff_h2**2)

        _grad_x_up = grad_x_up[r_i]
        _grad_y_up = grad_y_up[r_i]
        _grad_z_up = grad_z_up[r_i]

        # since d^2/dx^2 ln(|f(x)|) = (f''(x)*f(x) - f'(x)^2) / f(x)^2
        laplacian_ln_D_up[r_i] = (
            (gradgrad_x_up * det_geminal - _grad_x_up**2) / det_geminal**2
            + (gradgrad_y_up * det_geminal - _grad_y_up**2) / det_geminal**2
            + (gradgrad_z_up * det_geminal - _grad_z_up**2) / det_geminal**2
        )

    # laplacians dn
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn2_carts = r_dn_carts.copy()
        diff_p_y_r_dn2_carts = r_dn_carts.copy()
        diff_p_z_r_dn2_carts = r_dn_carts.copy()
        diff_p_x_r_dn2_carts[r_i][0] += diff_h2
        diff_p_y_r_dn2_carts[r_i][1] += diff_h2
        diff_p_z_r_dn2_carts[r_i][2] += diff_h2

        det_geminal_p_x_dn2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn2_carts,
        )
        det_geminal_p_y_dn2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn2_carts,
        )
        det_geminal_p_z_dn2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn2_carts,
        )

        diff_m_x_r_dn2_carts = r_dn_carts.copy()
        diff_m_y_r_dn2_carts = r_dn_carts.copy()
        diff_m_z_r_dn2_carts = r_dn_carts.copy()
        diff_m_x_r_dn2_carts[r_i][0] -= diff_h2
        diff_m_y_r_dn2_carts[r_i][1] -= diff_h2
        diff_m_z_r_dn2_carts[r_i][2] -= diff_h2

        det_geminal_m_x_dn2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn2_carts,
        )
        det_geminal_m_y_dn2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn2_carts,
        )
        det_geminal_m_z_dn2 = compute_det_geminal_all_elements(
            geminal_data=geminal_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn2_carts,
        )

        """ mathematically correct, but numerically unstable
        gradgrad_x_dn = (
            np.log(np.abs(det_geminal_p_x_dn2))
            + np.log(np.abs(det_geminal_m_x_dn2))
            - 2.0 * np.log(np.abs(det_geminal))
        ) / (diff_h2**2)

        gradgrad_y_dn = (
            np.log(np.abs(det_geminal_p_y_dn2))
            + np.log(np.abs(det_geminal_m_y_dn2))
            - 2.0 * np.log(np.abs(det_geminal))
        ) / (diff_h2**2)

        gradgrad_z_dn = (
            np.log(np.abs(det_geminal_p_z_dn2))
            + np.log(np.abs(det_geminal_m_z_dn2))
            - 2.0 * np.log(np.abs(det_geminal))
        ) / (diff_h2**2)
        """

        # compute f''(x)
        gradgrad_x_dn = (det_geminal_p_x_dn2 + det_geminal_m_x_dn2 - 2.0 * det_geminal) / (diff_h2**2)

        gradgrad_y_dn = (det_geminal_p_y_dn2 + det_geminal_m_y_dn2 - 2.0 * det_geminal) / (diff_h2**2)

        gradgrad_z_dn = (det_geminal_p_z_dn2 + det_geminal_m_z_dn2 - 2.0 * det_geminal) / (diff_h2**2)

        _grad_x_dn = grad_x_dn[r_i]
        _grad_y_dn = grad_y_dn[r_i]
        _grad_z_dn = grad_z_dn[r_i]

        # since d^2/dx^2 ln(|f(x)|) = (f''(x)*f(x) - f'(x)^2) / f(x)^2
        laplacian_ln_D_dn[r_i] = (
            (gradgrad_x_dn * det_geminal - _grad_x_dn**2) / det_geminal**2
            + (gradgrad_y_dn * det_geminal - _grad_y_dn**2) / det_geminal**2
            + (gradgrad_z_dn * det_geminal - _grad_z_dn**2) / det_geminal**2
        )

    # Returning answers
    return grad_ln_D_up, grad_ln_D_dn, laplacian_ln_D_up, laplacian_ln_D_dn


'''
if __name__ == "__main__":
    import pickle
    import time

    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    """
    # test MOs
    num_r_up_cart_samples = 2
    num_r_dn_cart_samples = 2
    num_R_cart_samples = 6
    num_ao = 6
    num_mo_up = num_mo_dn = num_r_up_cart_samples  # Slater Determinant
    num_ao_prim = 6
    orbital_indices = [0, 1, 2, 3, 4, 5]
    exponents = [1.2, 0.5, 0.1, 0.05, 0.05, 0.05]
    coefficients = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    angular_momentums = [0, 0, 0, 1, 1, 1]
    magnetic_quantum_numbers = [0, 0, 0, 0, +1, -1]

    # generate matrices for the test
    mo_coefficients_up = mo_coefficients_dn = np.random.rand(num_mo_up, num_ao)
    mo_lambda_matrix_paired = np.eye(num_mo_up, num_mo_dn, k=0)
    mo_lambda_matrix_unpaired = np.eye(num_mo_up, num_mo_up - num_mo_dn, k=-num_mo_dn)
    mo_lambda_matrix = np.hstack([mo_lambda_matrix_paired, mo_lambda_matrix_unpaired])

    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = r_up_carts
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * num_R_cart_samples,
        element_symbols=["X"] * num_R_cart_samples,
        atomic_labels=["X"] * num_R_cart_samples,
    )

    aos_up_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    aos_dn_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(num_R_cart_samples)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )

    mos_up_data = MOs_data(num_mo=num_mo_up, mo_coefficients=mo_coefficients_up, aos_data=aos_up_data)

    mos_dn_data = MOs_data(num_mo=num_mo_dn, mo_coefficients=mo_coefficients_dn, aos_data=aos_dn_data)

    geminal_mo_data = Geminal_data(
        num_electron_up=num_r_up_cart_samples,
        num_electron_dn=num_r_dn_cart_samples,
        orb_data_up_spin=mos_up_data,
        orb_data_dn_spin=mos_dn_data,
        lambda_matrix=mo_lambda_matrix,
    )

    geminal_mo_matrix = compute_geminal_all_elements_api(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # generate matrices for the test
    ao_lambda_matrix_paired = np.dot(mo_coefficients_up.T, np.dot(mo_lambda_matrix_paired, mo_coefficients_dn))
    ao_lambda_matrix_unpaired = np.dot(mo_coefficients_up.T, mo_lambda_matrix_unpaired)
    ao_lambda_matrix = np.hstack([ao_lambda_matrix_paired, ao_lambda_matrix_unpaired])

    # check if generated ao_lambda_matrix is symmetric:
    assert np.allclose(ao_lambda_matrix, ao_lambda_matrix.T)

    geminal_ao_data = Geminal_data(
        num_electron_up=num_r_up_cart_samples,
        num_electron_dn=num_r_dn_cart_samples,
        orb_data_up_spin=aos_up_data,
        orb_data_dn_spin=aos_dn_data,
        lambda_matrix=ao_lambda_matrix,
    )

    geminal_ao_matrix = compute_geminal_all_elements_api(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # check if geminals with AO and MO representations are consistent
    np.testing.assert_array_almost_equal(geminal_ao_matrix, geminal_mo_matrix, decimal=15)

    grad_ln_D_up, grad_ln_D_dn, sum_laplacian_ln_D = compute_grads_and_laplacian_ln_Det_api(
        geminal_data=geminal_ao_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    print(grad_ln_D_up)
    print(grad_ln_D_dn)
    print(sum_laplacian_ln_D)
    """

    # ratio
    hamiltonian_chk = "hamiltonian_data.chk"
    with open(hamiltonian_chk, "rb") as f:
        hamiltonian_data = pickle.load(f)
    geminal_data = hamiltonian_data.wavefunction_data.geminal_data

    # test MOs
    num_electron_up = 4
    num_electron_dn = 4

    # Initialization
    r_carts_up = []
    r_carts_dn = []

    total_electrons = 0

    if hamiltonian_data.coulomb_potential_data.ecp_flag:
        charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
            hamiltonian_data.coulomb_potential_data.z_cores
        )
    else:
        charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

    coords = hamiltonian_data.structure_data.positions_cart

    # Place electrons around each nucleus
    for i in range(len(coords)):
        charge = charges[i]
        num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

        # Retrieve the position coordinates
        x, y, z = coords[i]

        # Place electrons
        for _ in range(num_electrons):
            # Calculate distance range
            distance = np.random.uniform(0.1, 2.0)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            # Convert spherical to Cartesian coordinates
            dx = distance * np.sin(theta) * np.cos(phi)
            dy = distance * np.sin(theta) * np.sin(phi)
            dz = distance * np.cos(theta)

            # Position of the electron
            electron_position = np.array([x + dx, y + dy, z + dz])

            # Assign spin
            if len(r_carts_up) < num_electron_up:
                r_carts_up.append(electron_position)
            else:
                r_carts_dn.append(electron_position)

        total_electrons += num_electrons

    # Handle surplus electrons
    remaining_up = num_electron_up - len(r_carts_up)
    remaining_dn = num_electron_dn - len(r_carts_dn)

    # Randomly place any remaining electrons
    for _ in range(remaining_up):
        r_carts_up.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))
    for _ in range(remaining_dn):
        r_carts_dn.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))

    r_up_carts = np.array(r_carts_up)
    r_dn_carts = np.array(r_carts_dn)

    N_grid_up = len(r_up_carts)
    N_grid_dn = len(r_dn_carts)
    old_r_up_carts = r_up_carts
    old_r_dn_carts = r_dn_carts
    new_r_up_carts_arr = []
    new_r_dn_carts_arr = []
    for i in range(N_grid_up):
        new_r_up_carts = old_r_up_carts.copy()
        new_r_dn_carts = old_r_dn_carts.copy()
        new_r_up_carts[i][0] += 0.05 * new_r_up_carts[i][0]
        new_r_up_carts_arr.append(new_r_up_carts)
        new_r_dn_carts_arr.append(new_r_dn_carts)
        new_r_up_carts = old_r_up_carts.copy()
        new_r_dn_carts = old_r_dn_carts.copy()
        new_r_up_carts[i][1] += 0.05 * new_r_up_carts[i][1]
        new_r_up_carts_arr.append(new_r_up_carts)
        new_r_dn_carts_arr.append(new_r_dn_carts)
        new_r_up_carts = old_r_up_carts.copy()
        new_r_dn_carts = old_r_dn_carts.copy()
        new_r_up_carts[i][2] += 0.05 * new_r_up_carts[i][2]
        new_r_up_carts_arr.append(new_r_up_carts)
        new_r_dn_carts_arr.append(new_r_dn_carts)
        new_r_up_carts = old_r_up_carts.copy()
        new_r_dn_carts = old_r_dn_carts.copy()
        new_r_up_carts[i][0] -= 0.05 * new_r_up_carts[i][0]
        new_r_up_carts_arr.append(new_r_up_carts)
        new_r_dn_carts_arr.append(new_r_dn_carts)
        new_r_up_carts = old_r_up_carts.copy()
        new_r_dn_carts = old_r_dn_carts.copy()
        new_r_up_carts[i][1] -= 0.05 * new_r_up_carts[i][1]
        new_r_up_carts_arr.append(new_r_up_carts)
        new_r_dn_carts_arr.append(new_r_dn_carts)
        new_r_up_carts = old_r_up_carts.copy()
        new_r_dn_carts = old_r_dn_carts.copy()
        new_r_up_carts[i][2] -= 0.05 * new_r_up_carts[i][2]
        new_r_up_carts_arr.append(new_r_up_carts)
        new_r_dn_carts_arr.append(new_r_dn_carts)
    for i in range(N_grid_dn):
        new_r_up_carts = old_r_up_carts.copy()
        new_r_dn_carts = old_r_dn_carts.copy()
        new_r_dn_carts[i][0] += 0.05 * new_r_dn_carts[i][0]
        new_r_up_carts_arr.append(new_r_up_carts)
        new_r_dn_carts_arr.append(new_r_dn_carts)
        new_r_up_carts = old_r_up_carts.copy()
        new_r_dn_carts = old_r_dn_carts.copy()
        new_r_dn_carts[i][1] += 0.05 * new_r_dn_carts[i][1]
        new_r_up_carts_arr.append(new_r_up_carts)
        new_r_dn_carts_arr.append(new_r_dn_carts)
        new_r_up_carts = old_r_up_carts.copy()
        new_r_dn_carts = old_r_dn_carts.copy()
        new_r_dn_carts[i][2] += 0.05 * new_r_dn_carts[i][2]
        new_r_up_carts_arr.append(new_r_up_carts)
        new_r_dn_carts_arr.append(new_r_dn_carts)
        new_r_up_carts = old_r_up_carts.copy()
        new_r_dn_carts = old_r_dn_carts.copy()
        new_r_dn_carts[i][0] -= 0.05 * new_r_dn_carts[i][0]
        new_r_up_carts_arr.append(new_r_up_carts)
        new_r_dn_carts_arr.append(new_r_dn_carts)
        new_r_up_carts = old_r_up_carts.copy()
        new_r_dn_carts = old_r_dn_carts.copy()
        new_r_dn_carts[i][1] -= 0.05 * new_r_dn_carts[i][1]
        new_r_up_carts_arr.append(new_r_up_carts)
        new_r_dn_carts_arr.append(new_r_dn_carts)
        new_r_up_carts = old_r_up_carts.copy()
        new_r_dn_carts = old_r_dn_carts.copy()
        new_r_dn_carts[i][2] -= 0.05 * new_r_dn_carts[i][2]
        new_r_up_carts_arr.append(new_r_up_carts)
        new_r_dn_carts_arr.append(new_r_dn_carts)

    new_r_up_carts_arr = np.array(new_r_up_carts_arr)
    new_r_dn_carts_arr = np.array(new_r_dn_carts_arr)

    determinant_ratios_debug = compute_ratio_determinant_part_debug(
        geminal_data=geminal_data,
        old_r_up_carts=old_r_up_carts,
        old_r_dn_carts=old_r_dn_carts,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )

    start = time.perf_counter()
    determinant_ratios_debug = compute_ratio_determinant_part_debug(
        geminal_data=geminal_data,
        old_r_up_carts=old_r_up_carts,
        old_r_dn_carts=old_r_dn_carts,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )

    end = time.perf_counter()
    print(f"Elapsed Time = {(end - start) * 1e3:.3f} msec.")
    # print(determinant_ratios_debug)

    determinant_ratios_jax = compute_ratio_determinant_part_jax(
        geminal_data=geminal_data,
        old_r_up_carts=jnp.array(old_r_up_carts),
        old_r_dn_carts=jnp.array(old_r_dn_carts),
        new_r_up_carts_arr=jnp.array(new_r_up_carts_arr),
        new_r_dn_carts_arr=jnp.array(new_r_dn_carts_arr),
    )
    determinant_ratios_jax.block_until_ready()

    start = time.perf_counter()
    determinant_ratios_jax = compute_ratio_determinant_part_jax(
        geminal_data=geminal_data,
        old_r_up_carts=jnp.array(old_r_up_carts),
        old_r_dn_carts=jnp.array(old_r_dn_carts),
        new_r_up_carts_arr=jnp.array(new_r_up_carts_arr),
        new_r_dn_carts_arr=jnp.array(new_r_dn_carts_arr),
    )
    determinant_ratios_jax.block_until_ready()
    end = time.perf_counter()
    print(f"Elapsed Time = {(end - start) * 1e3:.3f} msec.")
    # print(determinant_ratios_jax)

    np.testing.assert_array_almost_equal(determinant_ratios_debug, determinant_ratios_jax, decimal=12)
'''
