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
    _aos_cart_to_sphe,
    _aos_sphe_to_cart,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
    compute_overlap_matrix,
)
from .molecular_orbital import MOs_data, compute_MOs, compute_MOs_grad, compute_MOs_laplacian
from .setting import EPS_rcond_SVD

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
    def is_mo_representation(self) -> bool:
        """Whether both spin channels are represented by molecular orbitals."""
        return isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data)

    @property
    def is_ao_representation(self) -> bool:
        """Whether both spin channels are represented by atomic orbitals."""
        return isinstance(self.orb_data_up_spin, (AOs_sphe_data, AOs_cart_data)) and isinstance(
            self.orb_data_dn_spin, (AOs_sphe_data, AOs_cart_data)
        )

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

    def to_cartesian(self) -> "Geminal_data":
        """Convert spherical orbitals to Cartesian and transform the lambda matrix.

        If the underlying orbitals are MOs, defer to ``MOs_data.to_cartesian``
        (the lambda matrix is unchanged). For Cartesian inputs, return self.
        """
        if isinstance(self.orb_data_up_spin, MOs_data) or isinstance(self.orb_data_dn_spin, MOs_data):
            if not (isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data)):
                raise ValueError("Cartesian conversion requires both spin channels to use MOs or both to use AOs.")
            return Geminal_data(
                num_electron_up=self.num_electron_up,
                num_electron_dn=self.num_electron_dn,
                orb_data_up_spin=self.orb_data_up_spin.to_cartesian(),
                orb_data_dn_spin=self.orb_data_dn_spin.to_cartesian(),
                lambda_matrix=self.lambda_matrix,
            )

        if isinstance(self.orb_data_up_spin, AOs_cart_data) and isinstance(self.orb_data_dn_spin, AOs_cart_data):
            return self
        if not isinstance(self.orb_data_up_spin, (AOs_sphe_data, AOs_cart_data)) or not isinstance(
            self.orb_data_dn_spin, (AOs_sphe_data, AOs_cart_data)
        ):
            raise ValueError("Cartesian conversion is only available from spherical/cartesian AOs or MOs.")
        aos_up_cart, transform_up = _aos_sphe_to_cart(self.orb_data_up_spin)
        aos_dn_cart, transform_dn = _aos_sphe_to_cart(self.orb_data_dn_spin)

        lambda_matrix = np.asarray(self.lambda_matrix, dtype=np.float64)
        lambda_matrix_paired, lambda_matrix_unpaired = np.hsplit(lambda_matrix, [self.orb_num_dn])
        lambda_paired_cart = transform_up.T @ lambda_matrix_paired @ transform_dn
        lambda_unpaired_cart = transform_up.T @ lambda_matrix_unpaired
        lambda_cart = np.hstack([lambda_paired_cart, lambda_unpaired_cart])
        lambda_cart = lambda_cart.astype(np.asarray(self.lambda_matrix).dtype, copy=False)

        return Geminal_data(
            num_electron_up=self.num_electron_up,
            num_electron_dn=self.num_electron_dn,
            orb_data_up_spin=aos_up_cart,
            orb_data_dn_spin=aos_dn_cart,
            lambda_matrix=lambda_cart,
        )

    def to_spherical(self) -> "Geminal_data":
        """Convert Cartesian orbitals to spherical and transform the lambda matrix.

        If the underlying orbitals are MOs, defer to ``MOs_data.to_spherical``
        (the lambda matrix is unchanged). For spherical inputs, return self.
        """
        if isinstance(self.orb_data_up_spin, MOs_data) or isinstance(self.orb_data_dn_spin, MOs_data):
            if not (isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data)):
                raise ValueError("Spherical conversion requires both spin channels to use MOs or both to use AOs.")
            return Geminal_data(
                num_electron_up=self.num_electron_up,
                num_electron_dn=self.num_electron_dn,
                orb_data_up_spin=self.orb_data_up_spin.to_spherical(),
                orb_data_dn_spin=self.orb_data_dn_spin.to_spherical(),
                lambda_matrix=self.lambda_matrix,
            )

        if isinstance(self.orb_data_up_spin, AOs_sphe_data) and isinstance(self.orb_data_dn_spin, AOs_sphe_data):
            return self
        if not isinstance(self.orb_data_up_spin, (AOs_sphe_data, AOs_cart_data)) or not isinstance(
            self.orb_data_dn_spin, (AOs_sphe_data, AOs_cart_data)
        ):
            raise ValueError("Spherical conversion is only available from cartesian/spherical AOs or MOs.")
        aos_up_sphe, transform_up = _aos_cart_to_sphe(self.orb_data_up_spin)
        aos_dn_sphe, transform_dn = _aos_cart_to_sphe(self.orb_data_dn_spin)

        lambda_matrix = np.asarray(self.lambda_matrix, dtype=np.float64)
        lambda_matrix_paired, lambda_matrix_unpaired = np.hsplit(lambda_matrix, [self.orb_num_dn])
        lambda_paired_sph = transform_up.T @ lambda_matrix_paired @ transform_dn
        lambda_unpaired_sph = transform_up.T @ lambda_matrix_unpaired
        lambda_sph = np.hstack([lambda_paired_sph, lambda_unpaired_sph])
        lambda_sph = lambda_sph.astype(np.asarray(self.lambda_matrix).dtype, copy=False)

        return Geminal_data(
            num_electron_up=self.num_electron_up,
            num_electron_dn=self.num_electron_dn,
            orb_data_up_spin=aos_up_sphe,
            orb_data_dn_spin=aos_dn_sphe,
            lambda_matrix=lambda_sph,
        )

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

    @classmethod
    def convert_from_AOs_to_MOs(
        cls,
        geminal_data: "Geminal_data",
        num_eigenvectors: int | str = "all",
    ) -> "Geminal_data":
        """Convert AOs to MOs via generalized eigenvectors of :math:`fSP=PL`.

        Args:
            geminal_data: Input geminal representation.
            num_eigenvectors: ``"all"`` for full AO-space projection. If an integer ``N`` is
                supplied, keep only the ``N`` largest-magnitude eigenvalue vectors.

        Returns:
            Geminal_data in MO representation.
        """
        if isinstance(geminal_data.orb_data_up_spin, MOs_data) and isinstance(geminal_data.orb_data_dn_spin, MOs_data):
            return geminal_data

        if not (
            isinstance(geminal_data.orb_data_up_spin, (AOs_sphe_data, AOs_cart_data))
            and isinstance(geminal_data.orb_data_dn_spin, (AOs_sphe_data, AOs_cart_data))
        ):
            raise NotImplementedError

        aos_data_up_spin = geminal_data.orb_data_up_spin
        aos_data_dn_spin = geminal_data.orb_data_dn_spin

        if aos_data_up_spin.num_ao != aos_data_dn_spin.num_ao:
            raise ValueError("AO->MO conversion currently requires identical up/down AO dimensions.")

        if type(aos_data_up_spin) is not type(aos_data_dn_spin):
            raise ValueError("AO->MO conversion requires matching AO basis types for up/down channels.")

        ao_dim = aos_data_up_spin.num_ao
        if isinstance(num_eigenvectors, str):
            if num_eigenvectors != "all":
                raise ValueError(f"num_eigenvectors must be 'all' or int, got {num_eigenvectors}.")
            num_mo = ao_dim
            use_truncated_mode = False
        elif isinstance(num_eigenvectors, (int, np.integer)):
            requested_num_mo = int(num_eigenvectors)
            if requested_num_mo <= 0:
                raise ValueError(f"num_eigenvectors={requested_num_mo} must be positive.")
            num_mo = min(requested_num_mo, ao_dim)
            use_truncated_mode = True
            num_truncated = ao_dim - num_mo
            logger.info(
                "AO->MO truncated mode: kept %d eigenvalues out of %d (truncated %d).",
                num_mo,
                ao_dim,
                num_truncated,
            )
        else:
            raise ValueError(f"num_eigenvectors must be 'all' or int, got {type(num_eigenvectors)}.")

        ao_lambda_matrix_paired, ao_lambda_matrix_unpaired = np.hsplit(
            np.asarray(geminal_data.lambda_matrix), [geminal_data.orb_num_dn]
        )
        ao_lambda_matrix_paired = np.asarray(ao_lambda_matrix_paired, dtype=np.float64)
        ao_lambda_matrix_unpaired = np.asarray(ao_lambda_matrix_unpaired, dtype=np.float64)

        overlap_up = np.asarray(compute_overlap_matrix(aos_data_up_spin), dtype=np.float64)
        overlap_dn = np.asarray(compute_overlap_matrix(aos_data_dn_spin), dtype=np.float64)

        overlap_up = 0.5 * (overlap_up + overlap_up.T)
        overlap_dn = 0.5 * (overlap_dn + overlap_dn.T)

        eigvals_up, eigvecs_up = np.linalg.eigh(overlap_up)
        eigvals_dn, eigvecs_dn = np.linalg.eigh(overlap_dn)

        min_eig_up = float(np.min(eigvals_up))
        min_eig_dn = float(np.min(eigvals_dn))
        if min_eig_up <= 0.0:
            raise ValueError(f"AO overlap matrix (up) is not positive definite (min eigenvalue={min_eig_up}).")
        if min_eig_dn <= 0.0:
            raise ValueError(f"AO overlap matrix (dn) is not positive definite (min eigenvalue={min_eig_dn}).")

        sqrt_overlap_up = eigvecs_up @ np.diag(np.sqrt(eigvals_up)) @ eigvecs_up.T
        inv_sqrt_overlap_up = eigvecs_up @ np.diag(1.0 / np.sqrt(eigvals_up)) @ eigvecs_up.T
        sqrt_overlap_dn = eigvecs_dn @ np.diag(np.sqrt(eigvals_dn)) @ eigvecs_dn.T
        inv_sqrt_overlap_dn = eigvecs_dn @ np.diag(1.0 / np.sqrt(eigvals_dn)) @ eigvecs_dn.T

        f_biorth_repr = sqrt_overlap_up @ ao_lambda_matrix_paired @ sqrt_overlap_dn
        u_mat, singular_vals, v_h = np.linalg.svd(f_biorth_repr, full_matrices=False)

        selected_evals = np.real_if_close(singular_vals[:num_mo], tol=1000).astype(np.float64)
        selected_vectors_up = np.real_if_close(u_mat[:, :num_mo], tol=1000).astype(np.float64)
        selected_vectors_dn = np.real_if_close(v_h.T[:, :num_mo], tol=1000).astype(np.float64)

        if use_truncated_mode:
            logger.debug(
                "[MOOPT-TRACE] AO->MO selected eigenvalues before scaling: %s",
                np.array2string(selected_evals, precision=10, separator=", "),
            )

        if use_truncated_mode:
            num_scale = min(int(geminal_data.num_electron_dn), selected_evals.size)
            if num_scale > 0:
                selected_evals[:num_scale] = 1.0

            logger.debug(
                "[MOOPT-TRACE] AO->MO selected eigenvalues after scaling: %s",
                np.array2string(selected_evals, precision=10, separator=", "),
            )

        p_matrix_cols_up = inv_sqrt_overlap_up @ selected_vectors_up
        p_matrix_cols_dn = inv_sqrt_overlap_dn @ selected_vectors_dn
        mo_coefficients_up = p_matrix_cols_up.T
        mo_coefficients_dn = p_matrix_cols_dn.T

        mo_lambda_matrix_paired = np.diag(selected_evals)
        mo_lambda_matrix_unpaired = mo_coefficients_up @ overlap_up @ ao_lambda_matrix_unpaired

        mo_lambda_matrix = np.hstack([mo_lambda_matrix_paired, mo_lambda_matrix_unpaired])

        mo_dtype = np.asarray(geminal_data.lambda_matrix).dtype
        mo_lambda_matrix = mo_lambda_matrix.astype(mo_dtype, copy=False)

        mos_data_up_spin = MOs_data(num_mo=num_mo, aos_data=aos_data_up_spin, mo_coefficients=mo_coefficients_up)
        mos_data_dn_spin = MOs_data(num_mo=num_mo, aos_data=aos_data_dn_spin, mo_coefficients=mo_coefficients_dn)

        psp_up = mo_coefficients_up @ overlap_up @ mo_coefficients_up.T
        psp_dn = mo_coefficients_dn @ overlap_dn @ mo_coefficients_dn.T
        identity = np.eye(num_mo, dtype=np.float64)
        psp_up_diff = psp_up - identity
        psp_dn_diff = psp_dn - identity

        up_frob_norm = float(np.linalg.norm(psp_up_diff))
        up_max_abs_diff = float(np.max(np.abs(psp_up_diff))) if psp_up_diff.size else 0.0
        up_is_orthonormal = bool(np.allclose(psp_up, identity, atol=1.0e-8, rtol=1.0e-6))

        dn_frob_norm = float(np.linalg.norm(psp_dn_diff))
        dn_max_abs_diff = float(np.max(np.abs(psp_dn_diff))) if psp_dn_diff.size else 0.0
        dn_is_orthonormal = bool(np.allclose(psp_dn, identity, atol=1.0e-8, rtol=1.0e-6))

        logger.debug(
            "[MOOPT-TRACE] AO->MO orthogonality check (up): allclose(P^TSP, I)=%s, ||P^TSP-I||_F=%.3e, max|P^TSP-I|=%.3e",
            up_is_orthonormal,
            up_frob_norm,
            up_max_abs_diff,
        )
        logger.debug(
            "[MOOPT-TRACE] AO->MO orthogonality check (dn): allclose(P^TSP, I)=%s, ||P^TSP-I||_F=%.3e, max|P^TSP-I|=%.3e",
            dn_is_orthonormal,
            dn_frob_norm,
            dn_max_abs_diff,
        )

        return cls(
            num_electron_up=geminal_data.num_electron_up,
            num_electron_dn=geminal_data.num_electron_dn,
            orb_data_up_spin=mos_data_up_spin,
            orb_data_dn_spin=mos_data_dn_spin,
            lambda_matrix=mo_lambda_matrix,
        )

    def apply_ao_projected_paired_update_and_reproject(
        self,
        ao_paired_direction: npt.NDArray | jax.Array,
        step_size: float = 1.0,
    ) -> "Geminal_data":
        r"""Apply AO-space corrected paired update and reproject back to MOs.

        This method is intended for MO-coefficient optimization workflows where
        the paired AO-space direction :math:`D` is corrected by the projector
        formula

        .. math::
            D^c = D - (I-L^T)D(I-R^T),

        with

        .. math::
            R=(SP)P^T, \quad L=R^T,

        where ``S`` is the AO overlap matrix and ``P`` contains the current MO
        coefficients as AO-column vectors.

        After applying the AO paired update, the geminal is projected back to an
        MO representation using a fixed rank
        ``num_eigenvectors=self.num_electron_dn``.

        Args:
            ao_paired_direction: AO-space direction matrix :math:`D` for the paired
                block, shape ``(num_ao, num_ao)``.
            step_size: Scalar factor multiplied to :math:`D^c` before applying the
                update.

        Returns:
            Updated geminal in MO representation with fixed
            ``num_mo=self.num_electron_dn`` (subject to AO-dimension clipping in
            ``convert_from_AOs_to_MOs``).

        Raises:
            ValueError: If this geminal is not in MO representation for both spins,
                or if shapes/types are inconsistent.
        """
        if not (isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data)):
            raise ValueError("AO-projected paired update requires MO representation on both spin channels.")

        if not isinstance(step_size, (int, float, np.integer, np.floating)):
            raise ValueError(f"step_size must be numeric, got {type(step_size)}.")

        aos_data_up_spin = self.orb_data_up_spin.aos_data
        aos_data_dn_spin = self.orb_data_dn_spin.aos_data
        if aos_data_up_spin.num_ao != aos_data_dn_spin.num_ao:
            raise ValueError("AO-projected paired update requires identical up/down AO dimensions.")
        if type(aos_data_up_spin) is not type(aos_data_dn_spin):
            raise ValueError("AO-projected paired update requires matching AO basis types for up/down channels.")

        ao_dim = aos_data_up_spin.num_ao
        direction = np.asarray(ao_paired_direction, dtype=np.float64)
        if direction.shape != (ao_dim, ao_dim):
            raise ValueError(f"ao_paired_direction shape {direction.shape} is incompatible with expected ({ao_dim}, {ao_dim}).")

        overlap_up = np.asarray(compute_overlap_matrix(aos_data_up_spin), dtype=np.float64)
        overlap_up = 0.5 * (overlap_up + overlap_up.T)

        p_matrix_cols = np.asarray(self.orb_data_up_spin.mo_coefficients, dtype=np.float64).T
        if p_matrix_cols.shape[0] != ao_dim:
            raise ValueError("MO coefficients are inconsistent with AO dimension " + f"({p_matrix_cols.shape[0]} != {ao_dim}).")

        right_projector = (overlap_up @ p_matrix_cols) @ p_matrix_cols.T
        left_projector = right_projector.T

        identity = np.eye(ao_dim, dtype=np.float64)
        corrected_direction = direction - ((identity - left_projector.T) @ direction @ (identity - right_projector.T))

        geminal_ao = Geminal_data.convert_from_MOs_to_AOs(self)
        ao_lambda_paired, ao_lambda_unpaired = np.hsplit(np.asarray(geminal_ao.lambda_matrix), [geminal_ao.orb_num_dn])

        ao_lambda_paired_updated = ao_lambda_paired + float(step_size) * corrected_direction
        ao_lambda_updated = np.hstack([ao_lambda_paired_updated, ao_lambda_unpaired])
        ao_lambda_updated = ao_lambda_updated.astype(np.asarray(geminal_ao.lambda_matrix).dtype, copy=False)

        geminal_ao_updated = Geminal_data(
            num_electron_up=geminal_ao.num_electron_up,
            num_electron_dn=geminal_ao.num_electron_dn,
            orb_data_up_spin=geminal_ao.orb_data_up_spin,
            orb_data_dn_spin=geminal_ao.orb_data_dn_spin,
            lambda_matrix=ao_lambda_updated,
        )

        return Geminal_data.convert_from_AOs_to_MOs(
            geminal_data=geminal_ao_updated,
            num_eigenvectors=self.num_electron_dn,
        )


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
    and hessian introduce numerical instability. The custom derivative uses SVD
    to compute G^{-1} in the backward pass, avoiding NaN when G is near-singular
    (small singular values are zeroed rather than producing 1/~0 from LU).

    Returns:
        - primal output: ln|det(G)|
        - residuals: (inputs and SVD factors) for use in backward pass
    """
    G = compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)
    ln_det = jnp.log(jnp.abs(jnp.linalg.det(G)))
    # Compute SVD: G = U_svd @ diag(s) @ Vt
    U_svd, s, Vt = jnp.linalg.svd(G, full_matrices=False)
    return ln_det, (geminal_data, r_up_carts, r_dn_carts, U_svd, s, Vt)


# Backward pass for custom VJP.
def _ln_det_bwd(res, g):
    """Backward pass for custom VJP.

    The custom derivative is needed for ln |Det(G)| because the jax native grad
    and hessian introduce numerical instability. The custom derivative uses SVD
    to compute G^{-1} in the backward pass, avoiding NaN when G is near-singular
    (small singular values are zeroed rather than producing 1/~0 from LU).

    Args:
        res: residuals from forward pass
        g: cotangent of the primal output

    Returns:
        Gradients with respect to (geminal_data, r_up_carts, r_dn_carts)
    """
    geminal_data, r_up_carts, r_dn_carts, U_svd, s, Vt = res

    # Compute G^{-1} via SVD pseudoinverse with thresholding.
    # Singular values below rcond * s_max are zeroed to avoid NaN from 1/~0.
    rcond = jnp.finfo(jnp.float64).eps * float(s.shape[0])
    s_inv = jnp.where(s > rcond * s[0], 1.0 / s, 0.0)
    X = (Vt.T * s_inv[jnp.newaxis, :]) @ U_svd.T  # G^{-1}, shape (n, n)

    # d ln|det G| / dG = (G^{-1})^T, scaled by incoming cotangent g
    grad_G = g * X.T

    # Now backpropagate through compute_geminal_all_elements_jax
    _, vjp_fun = jax.vjp(compute_geminal_all_elements, geminal_data, r_up_carts, r_dn_carts)
    # Apply VJP to produce gradients for each input
    return vjp_fun(grad_G)


# Register the custom VJP rule !!
compute_ln_det_geminal_all_elements.defvjp(_ln_det_fwd, _ln_det_bwd)


@jax.custom_vjp
def compute_ln_det_geminal_all_elements_fast(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inv: jax.Array,
) -> jax.Array:
    r"""Compute :math:`\ln|\det G|` using pre-computed ``geminal_inv`` in gradients.

    Mirrors :func:`compute_ln_det_geminal_all_elements` in the forward direction.
    The **backward pass** replaces the implicit ``G^{-1}`` computation that JAX
    would normally perform (via a fresh LU decomposition) with the pre-computed
    ``geminal_inv`` — the Sherman-Morrison running inverse.  This avoids
    catastrophic NaN for near-singular geminal matrices sampled when
    ``epsilon_AS > 0``.

    Args:
        geminal_data: Geminal parameters (lambda matrix etc.).
        r_up_carts: Cartesian coordinates of up-spin electrons ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons ``(N_dn, 3)``.
        geminal_inv: Pre-computed inverse geminal matrix ``(N_up, N_up)``.

    Returns:
        Scalar :math:`\ln|\det G|`.

    Warning:
        ``geminal_inv`` **must** equal ``G(r_up_carts, r_dn_carts)^{-1}`` exactly
        at the supplied electron positions.  This is only guaranteed when the
        inverse is maintained via **single-electron (rank-1) Sherman-Morrison
        updates** starting from a freshly initialised LU inverse — the pattern
        used in the MCMC loop.  Passing an inverse that corresponds to different
        electron positions silently produces incorrect gradients.
    """
    G = compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)
    return jnp.log(jnp.abs(jnp.linalg.det(G)))


def _ln_det_fast_fwd(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inv: jax.Array,
):
    G = compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)
    val = jnp.log(jnp.abs(jnp.linalg.det(G)))
    # Save inputs for backward (geminal_inv replaces G^{-1} in bwd)
    return val, (geminal_data, r_up_carts, r_dn_carts, geminal_inv)


def _ln_det_fast_bwd(res, g):
    geminal_data, r_up_carts, r_dn_carts, geminal_inv = res
    # d(ln|det G|)/d(G_{ij}) = (G^{-T})_{ij}
    # Use the pre-computed inverse instead of re-solving.
    G_bar = g * geminal_inv.T  # cotangent w.r.t. G, shape (N_up, N_up)
    # Propagate cotangent back through G = compute_geminal_all_elements(...)
    _, vjp_fn = jax.vjp(compute_geminal_all_elements, geminal_data, r_up_carts, r_dn_carts)
    grad_geminal_data, grad_r_up, grad_r_dn = vjp_fn(G_bar)
    # geminal_inv is a non-differentiable constant; return zeros.
    return grad_geminal_data, grad_r_up, grad_r_dn, jnp.zeros_like(geminal_inv)


# Register the custom VJP rule !!
compute_ln_det_geminal_all_elements_fast.defvjp(_ln_det_fast_fwd, _ln_det_fast_bwd)


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
    # Guard: when S*F == 0 (e.g. SVD-truncated geminal_inv with a near-zero
    # column norm), return 0 so the walker is fully down-weighted rather than
    # producing +inf -> w_L = (inf/inf)^2 = NaN.
    SF = S * F
    R_AS = jnp.where(SF > 0.0, SF ** (-theta), 0.0)

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
    # Use SVD with conservative threshold to avoid Inf from 1/sigma^2 for tiny sigma
    sigma = jnp.linalg.svd(geminal, compute_uv=False)
    sigma_sq_inv = jnp.where(sigma > EPS_rcond_SVD * sigma[0], 1.0 / (sigma**2), 0.0)
    F = jnp.sum(sigma_sq_inv)

    # compute the scaling factor
    S = jnp.min(jnp.sum(geminal**2, axis=0))

    # compute R_AS
    # Guard: S*F can be 0*∞ = NaN when G is near-singular (S→0, F→∞).
    # Return 0 in that case to fully down-weight the walker instead of NaN.
    SF = S * F
    R_AS = jnp.where(jnp.isfinite(SF) & (SF > 0.0), SF ** (-theta), 0.0)

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
def _compute_ratio_determinant_part_rank1_update(
    geminal_data: Geminal_data,
    A_old_inv: jax.Array,
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_up_carts_arr: jax.Array,
    new_r_dn_carts_arr: jax.Array,
) -> jax.Array:
    r"""Determinant ratio $\det G(\mathbf r')/\det G(\mathbf r)$ for batched moves.

    Optimized for the non-local ECP mesh where *exactly one* electron moves per grid.
    We identify the moved electron once, batch-evaluate the new AO/MO row or column,
    and apply a rank-1 update formula ``row @ A_old_inv[:, idx]`` or
    ``A_old_inv[idx, :] @ col`` with ``vmap``.

    Args:
        geminal_data: Geminal parameters and orbital references.
        A_old_inv: Inverse geminal matrix for the reference configuration with shape ``(N_up, N_up)``.
        old_r_up_carts: Original spin-up electron coordinates with shape ``(N_up, 3)``.
        old_r_dn_carts: Original spin-down electron coordinates with shape ``(N_dn, 3)``.
        new_r_up_carts_arr: Proposed spin-up coordinates per grid with shape ``(N_grid, N_up, 3)``.
        new_r_dn_carts_arr: Proposed spin-down coordinates per grid with shape ``(N_grid, N_dn, 3)``.

    Returns:
        jax.Array: Determinant ratios per grid with shape ``(N_grid,)``.

    Warning:
        Each proposed configuration in ``new_r_up_carts_arr`` / ``new_r_dn_carts_arr``
        must differ from ``old_r_up_carts`` / ``old_r_dn_carts`` in **exactly one
        electron**.  The moved electron is identified via ``jnp.argmax`` on the
        change mask; if two or more electrons differ in the same config, only the
        first changed electron is detected and the ratio is silently incorrect.
        This function is intended exclusively for the non-local ECP integration
        grid generated by the MCMC loop, where exactly one electron is displaced
        per grid point by construction.
    """
    num_up = old_r_up_carts.shape[0]
    num_dn = old_r_dn_carts.shape[0]

    # Degenerate cases (no up or no down) fall back to full determinant evaluation.
    if num_up == 0 or num_dn == 0:
        det_x = compute_det_geminal_all_elements(geminal_data, old_r_up_carts, old_r_dn_carts)
        det_xp = vmap(compute_det_geminal_all_elements, in_axes=(None, 0, 0))(
            geminal_data, new_r_up_carts_arr, new_r_dn_carts_arr
        )
        return det_xp / det_x

    # Which electron moved on each grid? Only one electron moves per grid by construction.
    delta_up = new_r_up_carts_arr - old_r_up_carts
    delta_dn = new_r_dn_carts_arr - old_r_dn_carts
    moved_up_mask = jnp.any(delta_up != 0.0, axis=2)  # (N_grid, N_up)
    moved_dn_mask = jnp.any(delta_dn != 0.0, axis=2)  # (N_grid, N_dn)
    moved_up_exists = jnp.any(moved_up_mask, axis=1)

    # Indices of the moved electron per grid (argmax is fine because only one is non-zero).
    idx_up = jnp.argmax(moved_up_mask.astype(jnp.int32), axis=1)  # (N_grid,)
    idx_dn = jnp.argmax(moved_dn_mask.astype(jnp.int32), axis=1)  # (N_grid,)

    # Gather the moved coordinates for batch AO/MO evaluation.
    r_up_new = jnp.take_along_axis(new_r_up_carts_arr, idx_up[:, None, None], axis=1)  # (N_grid, 1, 3)
    r_dn_new = jnp.take_along_axis(new_r_dn_carts_arr, idx_dn[:, None, None], axis=1)  # (N_grid, 1, 3)

    # Flatten grid axis for a single batched AO/MO evaluation (reduces JAX HLO count).
    r_up_new_flat = r_up_new.reshape(-1, 3)  # (N_grid, 3)
    r_dn_new_flat = r_dn_new.reshape(-1, 3)  # (N_grid, 3)

    # lambda split
    lambda_matrix_paired, lambda_matrix_unpaired = jnp.split(
        geminal_data.lambda_matrix, indices_or_sections=[geminal_data.orb_num_dn], axis=1
    )

    # Precompute old AO matrices once.
    orb_matrix_up_old = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, old_r_up_carts)
    orb_matrix_dn_old = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, old_r_dn_carts)

    # Batched AO for moved electrons (up) -> rows
    orb_up_new_batch = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_new_flat)  # (n_orb_up, G)
    tmp_up = jnp.dot(orb_up_new_batch.T, lambda_matrix_paired)  # (G, n_orb_dn)
    row_paired = jnp.dot(tmp_up, orb_matrix_dn_old)  # (G, N_dn)
    row_unpaired = jnp.dot(orb_up_new_batch.T, lambda_matrix_unpaired)  # (G, num_unpaired)
    new_rows_up = jnp.hstack([row_paired, row_unpaired])  # (G, N_up)

    # Batched AO for moved electrons (dn) -> columns
    orb_dn_new_batch = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_new_flat)  # (n_orb_dn, G)
    w_batch = jnp.dot(lambda_matrix_paired, orb_dn_new_batch)  # (n_orb_up, G)
    cols = jnp.dot(orb_matrix_up_old.T, w_batch)  # (N_up, G)
    new_cols_dn = cols.T  # (G, N_up)

    # rank-1 determinant ratios for up-move grids and dn-move grids.
    # Use matrix-matrix contractions first to maximize BLAS/TensorCore utilization,
    # then extract the moved-electron component per grid.
    up_all_cols = jnp.dot(new_rows_up, A_old_inv)  # (N_grid, N_up)
    det_ratio_up = jnp.take_along_axis(up_all_cols, idx_up[:, None], axis=1).reshape(-1)

    dn_all_rows = jnp.dot(A_old_inv, new_cols_dn.T).T  # (N_grid, N_up)
    det_ratio_dn = jnp.take_along_axis(dn_all_rows, idx_dn[:, None], axis=1).reshape(-1)

    # Select per grid based on which spin moved.
    determinant_ratios = jnp.where(moved_up_exists, det_ratio_up, det_ratio_dn)
    return determinant_ratios


def _compute_ratio_determinant_part_split_spin(
    geminal_data: Geminal_data,
    A_old_inv: jax.Array,
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_up_shifted: jax.Array,
    new_r_dn_shifted: jax.Array,
) -> jax.Array:
    r"""Determinant ratio for a block-structured mesh where up and dn electrons move separately.

    Avoids computing MOs for the unchanged spin block.  Compared with
    ``_compute_ratio_determinant_part_rank1_update`` called on the concatenated
    (G_up + G_dn) array, this evaluates only ``G_up`` up-spin AOs and ``G_dn``
    dn-spin AOs instead of evaluating both spins for all ``G_up + G_dn`` configs.

    Args:
        geminal_data: Geminal parameters and orbital references.
        A_old_inv: Inverse geminal matrix with shape ``(N_up, N_up)``.
        old_r_up_carts: Reference up-spin coordinates ``(N_up, 3)``.
        old_r_dn_carts: Reference dn-spin coordinates ``(N_dn, 3)``.
        new_r_up_shifted: Up-block proposed coords ``(G_up, N_up, 3)``.  Exactly one
            up electron differs from ``old_r_up_carts`` per config.
        new_r_dn_shifted: Dn-block proposed coords ``(G_dn, N_dn, 3)``.  Exactly one
            dn electron differs from ``old_r_dn_carts`` per config.

    Returns:
        jax.Array: Concatenated determinant ratios ``(G_up + G_dn,)``.

    Warning:
        Each config in ``new_r_up_shifted`` must differ from ``old_r_up_carts``
        in **exactly one up electron**, and each config in ``new_r_dn_shifted``
        must differ from ``old_r_dn_carts`` in **exactly one dn electron**.
        The moved electron is located via ``jnp.argmax`` on the change mask;
        if two or more electrons differ in the same config, only the first is
        detected and the ratio is silently incorrect.  This function is intended
        exclusively for the block-structured non-local ECP grids produced by
        the MCMC loop.
    """
    num_up = old_r_up_carts.shape[0]
    num_dn = old_r_dn_carts.shape[0]

    # Degenerate cases fall back to the general function on empty slices.
    if num_up == 0 or num_dn == 0:
        combined_up = jnp.concatenate(
            [new_r_up_shifted, jnp.broadcast_to(old_r_up_carts[None], (new_r_dn_shifted.shape[0], num_up, 3))],
            axis=0,
        )
        combined_dn = jnp.concatenate(
            [jnp.broadcast_to(old_r_dn_carts[None], (new_r_up_shifted.shape[0], num_dn, 3)), new_r_dn_shifted],
            axis=0,
        )
        return _compute_ratio_determinant_part_rank1_update(
            geminal_data, A_old_inv, old_r_up_carts, old_r_dn_carts, combined_up, combined_dn
        )

    lambda_matrix_paired, lambda_matrix_unpaired = jnp.split(
        geminal_data.lambda_matrix, indices_or_sections=[geminal_data.orb_num_dn], axis=1
    )

    # Precompute old AO matrices once.
    orb_matrix_up_old = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, old_r_up_carts)
    orb_matrix_dn_old = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, old_r_dn_carts)

    # ── UP BLOCK: up electron moved, dn unchanged ──────────────────────────────
    g_up = new_r_up_shifted.shape[0]
    delta_up = new_r_up_shifted - old_r_up_carts  # (G_up, N_up, 3)
    moved_up_mask = jnp.any(delta_up != 0.0, axis=2)  # (G_up, N_up)
    idx_up = jnp.argmax(moved_up_mask.astype(jnp.int32), axis=1)  # (G_up,)
    r_up_new_flat = jnp.take_along_axis(new_r_up_shifted, idx_up[:, None, None], axis=1).reshape(-1, 3)

    # Only evaluate up-spin MOs for the moved electron positions.
    orb_up_new_batch = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_new_flat)  # (n_orb_up, G_up)
    tmp_up = jnp.dot(orb_up_new_batch.T, lambda_matrix_paired)  # (G_up, n_orb_dn)
    row_paired = jnp.dot(tmp_up, orb_matrix_dn_old)  # (G_up, N_dn)
    row_unpaired = jnp.dot(orb_up_new_batch.T, lambda_matrix_unpaired)  # (G_up, num_unpaired)
    new_rows_up = jnp.hstack([row_paired, row_unpaired])  # (G_up, N_up)

    A_col_for_up = jnp.take(A_old_inv, idx_up, axis=1).T  # (G_up, N_up)
    det_ratio_up_block = jnp.sum(new_rows_up * A_col_for_up, axis=1)  # (G_up,)

    # ── DN BLOCK: dn electron moved, up unchanged ──────────────────────────────
    delta_dn = new_r_dn_shifted - old_r_dn_carts  # (G_dn, N_dn, 3)
    moved_dn_mask = jnp.any(delta_dn != 0.0, axis=2)  # (G_dn, N_dn)
    idx_dn = jnp.argmax(moved_dn_mask.astype(jnp.int32), axis=1)  # (G_dn,)
    r_dn_new_flat = jnp.take_along_axis(new_r_dn_shifted, idx_dn[:, None, None], axis=1).reshape(-1, 3)

    # Only evaluate dn-spin MOs for the moved electron positions.
    orb_dn_new_batch = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_new_flat)  # (n_orb_dn, G_dn)
    w_batch = jnp.dot(lambda_matrix_paired, orb_dn_new_batch)  # (n_orb_up, G_dn)
    new_cols_dn = jnp.dot(orb_matrix_up_old.T, w_batch).T  # (G_dn, N_up)

    A_row_for_dn = jnp.take(A_old_inv, idx_dn, axis=0)  # (G_dn, N_up)
    det_ratio_dn_block = jnp.sum(A_row_for_dn * new_cols_dn, axis=1)  # (G_dn,)

    return jnp.concatenate([det_ratio_up_block, det_ratio_dn_block])


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


@jax.custom_vjp
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

    The function uses a custom VJP to avoid NaN in the backward pass when G is
    near-singular. The standard JAX SVD backward computes ``1/(s_i^2 - s_j^2)``
    terms that blow up for degenerate singular values. The custom VJP instead uses
    the matrix identity ``d(G^{-1}) = -G^{-1} dG G^{-1}`` with a thresholded SVD
    pseudoinverse, which is both exact and numerically stable.

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
    # Compute G_inv via SVD pseudoinverse (numerically stable, avoids LU NaN).
    G = compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)
    _U, _s, _Vt = jnp.linalg.svd(G, full_matrices=False)
    # Use conservative threshold to prevent G^{-2} and G^{-3} terms in the
    # backward pass from diverging. Standard numpy.linalg.pinv uses max(M,N)*eps,
    # but for de_L/dc (which involves G_inv^2 in the chain rule) we need a larger
    # safety margin to avoid Inf/NaN in the gradient. EPS_rcond_SVD is set in setting.py
    # to handle near-singular G while preserving well-conditioned singular values.
    _s_inv = jnp.where(_s > EPS_rcond_SVD * _s[0], 1.0 / _s, 0.0)
    geminal_inverse = (_Vt.T * _s_inv[jnp.newaxis, :]) @ _U.T

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

    ao_up_grads = jnp.stack([ao_matrix_up_grad_x, ao_matrix_up_grad_y, ao_matrix_up_grad_z], axis=0)
    ao_dn_grads = jnp.stack([ao_matrix_dn_grad_x, ao_matrix_dn_grad_y, ao_matrix_dn_grad_z], axis=0)

    paired_dn = jnp.dot(lambda_matrix_paired, ao_matrix_dn)
    paired_dn_grads = jnp.einsum("ab,gbn->gan", lambda_matrix_paired, ao_dn_grads)

    geminal_grad_up_paired = jnp.einsum("gia,aj->gij", jnp.swapaxes(ao_up_grads, 1, 2), paired_dn)
    geminal_grad_up_unpaired = jnp.einsum("gia,ak->gik", jnp.swapaxes(ao_up_grads, 1, 2), lambda_matrix_unpaired)
    geminal_grad_up = jnp.concatenate([geminal_grad_up_paired, geminal_grad_up_unpaired], axis=2)

    geminal_grad_dn_paired = jnp.einsum("ia,gaj->gij", ao_matrix_up.T, paired_dn_grads)
    geminal_grad_dn_unpaired = jnp.zeros(
        (3, geminal_data.num_electron_up, geminal_data.num_electron_up - geminal_data.num_electron_dn),
        dtype=geminal_grad_dn_paired.dtype,
    )
    geminal_grad_dn = jnp.concatenate([geminal_grad_dn_paired, geminal_grad_dn_unpaired], axis=2)

    geminal_laplacian_up_paired = jnp.dot(ao_matrix_laplacian_up.T, paired_dn)
    geminal_laplacian_up_unpaired = jnp.dot(ao_matrix_laplacian_up.T, lambda_matrix_unpaired)
    geminal_laplacian_up = jnp.hstack([geminal_laplacian_up_paired, geminal_laplacian_up_unpaired])

    geminal_laplacian_dn_paired = jnp.dot(ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_laplacian_dn))
    geminal_laplacian_dn_unpaired = jnp.zeros(
        [geminal_data.num_electron_up, geminal_data.num_electron_up - geminal_data.num_electron_dn],
        dtype=geminal_laplacian_dn_paired.dtype,
    )
    geminal_laplacian_dn = jnp.hstack([geminal_laplacian_dn_paired, geminal_laplacian_dn_unpaired])

    grad_ln_D_up_stack = jnp.einsum("gij,ji->gi", geminal_grad_up, geminal_inverse)
    grad_ln_D_dn_stack = jnp.einsum("ij,gji->gi", geminal_inverse, geminal_grad_dn)

    grad_ln_D_up = grad_ln_D_up_stack.T
    grad_ln_D_dn = grad_ln_D_dn_stack.T

    grad_ln_D_up_x, grad_ln_D_up_y, grad_ln_D_up_z = grad_ln_D_up_stack
    grad_ln_D_dn_x, grad_ln_D_dn_y, grad_ln_D_dn_z = grad_ln_D_dn_stack

    lap_ln_D_up = -(
        grad_ln_D_up_x * grad_ln_D_up_x + grad_ln_D_up_y * grad_ln_D_up_y + grad_ln_D_up_z * grad_ln_D_up_z
    ) + jnp.einsum("ij,ji->i", geminal_laplacian_up, geminal_inverse)

    lap_ln_D_dn = -(
        grad_ln_D_dn_x * grad_ln_D_dn_x + grad_ln_D_dn_y * grad_ln_D_dn_y + grad_ln_D_dn_z * grad_ln_D_dn_z
    ) + jnp.einsum("ij,ji->i", geminal_inverse, geminal_laplacian_dn)

    return grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn


def _grads_lap_body(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inverse: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pure functional core of grad/laplacian computation.

    Contains the same computation as ``compute_grads_and_laplacian_ln_Det_fast``
    but without the ``@jit`` decorator or ``None``-check, so it can be safely
    passed to ``jax.vjp`` inside the custom VJP backward pass without creating
    a dependency on the public fast function.
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

    ao_up_grads = jnp.stack([ao_matrix_up_grad_x, ao_matrix_up_grad_y, ao_matrix_up_grad_z], axis=0)
    ao_dn_grads = jnp.stack([ao_matrix_dn_grad_x, ao_matrix_dn_grad_y, ao_matrix_dn_grad_z], axis=0)

    paired_dn = jnp.dot(lambda_matrix_paired, ao_matrix_dn)
    paired_dn_grads = jnp.einsum("ab,gbn->gan", lambda_matrix_paired, ao_dn_grads)

    geminal_grad_up_paired = jnp.einsum("gia,aj->gij", jnp.swapaxes(ao_up_grads, 1, 2), paired_dn)
    geminal_grad_up_unpaired = jnp.einsum("gia,ak->gik", jnp.swapaxes(ao_up_grads, 1, 2), lambda_matrix_unpaired)
    geminal_grad_up = jnp.concatenate([geminal_grad_up_paired, geminal_grad_up_unpaired], axis=2)

    geminal_grad_dn_paired = jnp.einsum("ia,gaj->gij", ao_matrix_up.T, paired_dn_grads)
    geminal_grad_dn_unpaired = jnp.zeros(
        (3, geminal_data.num_electron_up, geminal_data.num_electron_up - geminal_data.num_electron_dn),
        dtype=geminal_grad_dn_paired.dtype,
    )
    geminal_grad_dn = jnp.concatenate([geminal_grad_dn_paired, geminal_grad_dn_unpaired], axis=2)

    geminal_laplacian_up_paired = jnp.dot(ao_matrix_laplacian_up.T, paired_dn)
    geminal_laplacian_up_unpaired = jnp.dot(ao_matrix_laplacian_up.T, lambda_matrix_unpaired)
    geminal_laplacian_up = jnp.hstack([geminal_laplacian_up_paired, geminal_laplacian_up_unpaired])

    geminal_laplacian_dn_paired = jnp.dot(ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_laplacian_dn))
    geminal_laplacian_dn_unpaired = jnp.zeros(
        [geminal_data.num_electron_up, geminal_data.num_electron_up - geminal_data.num_electron_dn],
        dtype=geminal_laplacian_dn_paired.dtype,
    )
    geminal_laplacian_dn = jnp.hstack([geminal_laplacian_dn_paired, geminal_laplacian_dn_unpaired])

    grad_ln_D_up_stack = jnp.einsum("gij,ji->gi", geminal_grad_up, geminal_inverse)
    grad_ln_D_dn_stack = jnp.einsum("ij,gji->gi", geminal_inverse, geminal_grad_dn)

    grad_ln_D_up = grad_ln_D_up_stack.T
    grad_ln_D_dn = grad_ln_D_dn_stack.T

    grad_ln_D_up_x, grad_ln_D_up_y, grad_ln_D_up_z = grad_ln_D_up_stack
    grad_ln_D_dn_x, grad_ln_D_dn_y, grad_ln_D_dn_z = grad_ln_D_dn_stack

    lap_ln_D_up = -(
        grad_ln_D_up_x * grad_ln_D_up_x + grad_ln_D_up_y * grad_ln_D_up_y + grad_ln_D_up_z * grad_ln_D_up_z
    ) + jnp.einsum("ij,ji->i", geminal_laplacian_up, geminal_inverse)

    lap_ln_D_dn = -(
        grad_ln_D_dn_x * grad_ln_D_dn_x + grad_ln_D_dn_y * grad_ln_D_dn_y + grad_ln_D_dn_z * grad_ln_D_dn_z
    ) + jnp.einsum("ij,ji->i", geminal_inverse, geminal_laplacian_dn)

    return grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn


def _grads_lap_fwd(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
):
    """Forward pass: compute stable G_inv and primal outputs."""
    G = compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)
    _U, _s, _Vt = jnp.linalg.svd(G, full_matrices=False)
    # Use same conservative threshold as in compute_grads_and_laplacian_ln_Det
    _s_inv = jnp.where(_s > EPS_rcond_SVD * _s[0], 1.0 / _s, 0.0)
    G_inv_stable = (_Vt.T * _s_inv[jnp.newaxis, :]) @ _U.T
    primals = _grads_lap_body(geminal_data, r_up_carts, r_dn_carts, G_inv_stable)
    return primals, (geminal_data, r_up_carts, r_dn_carts, G_inv_stable)


def _grads_lap_bwd(res, g):
    """Backward pass using the matrix identity d(G^{-1}) = -G^{-1} dG G^{-1}.

    The gradient has two contributions:
    1. Direct: d/d(lambda) via lambda -> AO matrices -> G_grad/G_lap -> output.
       Obtained by differentiating _grads_lap_body w.r.t.
       all inputs (including G_inv_stable as an explicit argument).
    2. Inverse: d/d(lambda) via lambda -> G -> G^{-1} -> output.
       Obtained by converting G_inv_bar -> G_bar via -G_inv.T @ G_inv_bar @ G_inv.T,
       then propagating G_bar back through compute_geminal_all_elements.

    Neither path requires differentiating through jnp.linalg.svd, so degenerate
    singular values cannot produce NaN.
    """
    geminal_data, r_up_carts, r_dn_carts, G_inv_stable = res

    # Step 1: differentiate _grads_lap_body w.r.t. all args.
    # This gives direct gradients (AO path) and G_inv_bar (cotangent for G_inv).
    _, vjp_fn = jax.vjp(
        _grads_lap_body,
        geminal_data,
        r_up_carts,
        r_dn_carts,
        G_inv_stable,
    )
    d_geminal_direct, d_r_up_direct, d_r_dn_direct, G_inv_bar = vjp_fn(g)

    # Step 2: convert G_inv_bar -> G_bar using d(G^{-1}) = -G^{-1} dG G^{-1}:
    #   <G_inv_bar, dG^{-1}> = -<G_inv.T @ G_inv_bar @ G_inv.T, dG>
    G_bar_from_inv = -(G_inv_stable.T @ G_inv_bar @ G_inv_stable.T)

    # Step 3: propagate G_bar back through G = compute_geminal_all_elements(...).
    _, vjp_fn2 = jax.vjp(compute_geminal_all_elements, geminal_data, r_up_carts, r_dn_carts)
    d_geminal_inv, d_r_up_inv, d_r_dn_inv = vjp_fn2(G_bar_from_inv)

    # Total: sum both contributions.
    d_geminal = jax.tree_util.tree_map(lambda a, b: a + b, d_geminal_direct, d_geminal_inv)
    return d_geminal, d_r_up_direct + d_r_up_inv, d_r_dn_direct + d_r_dn_inv


compute_grads_and_laplacian_ln_Det.defvjp(_grads_lap_fwd, _grads_lap_bwd)


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
        geminal_inverse: Precomputed inverse of the geminal matrix ``G`` at the
            supplied ``(r_up_carts, r_dn_carts)``.

    Returns:
        Gradients (up/down) and Laplacians (up/down) of ln det G per electron.

    Warning:
        ``geminal_inverse`` **must** equal ``G(r_up_carts, r_dn_carts)^{-1}``
        exactly at the supplied electron positions.  This is only guaranteed when
        the inverse is maintained via **single-electron (rank-1) Sherman-Morrison
        updates** starting from a freshly initialised LU inverse — the pattern
        used in the MCMC loop.  Passing an inverse that corresponds to different
        electron positions silently produces incorrect kinetic energy.
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

    ao_up_grads = jnp.stack([ao_matrix_up_grad_x, ao_matrix_up_grad_y, ao_matrix_up_grad_z], axis=0)
    ao_dn_grads = jnp.stack([ao_matrix_dn_grad_x, ao_matrix_dn_grad_y, ao_matrix_dn_grad_z], axis=0)

    paired_dn = jnp.dot(lambda_matrix_paired, ao_matrix_dn)
    paired_dn_grads = jnp.einsum("ab,gbn->gan", lambda_matrix_paired, ao_dn_grads)

    geminal_grad_up_paired = jnp.einsum("gia,aj->gij", jnp.swapaxes(ao_up_grads, 1, 2), paired_dn)
    geminal_grad_up_unpaired = jnp.einsum("gia,ak->gik", jnp.swapaxes(ao_up_grads, 1, 2), lambda_matrix_unpaired)
    geminal_grad_up = jnp.concatenate([geminal_grad_up_paired, geminal_grad_up_unpaired], axis=2)

    geminal_grad_dn_paired = jnp.einsum("ia,gaj->gij", ao_matrix_up.T, paired_dn_grads)
    geminal_grad_dn_unpaired = jnp.zeros(
        (3, geminal_data.num_electron_up, geminal_data.num_electron_up - geminal_data.num_electron_dn),
        dtype=geminal_grad_dn_paired.dtype,
    )
    geminal_grad_dn = jnp.concatenate([geminal_grad_dn_paired, geminal_grad_dn_unpaired], axis=2)

    geminal_laplacian_up_paired = jnp.dot(ao_matrix_laplacian_up.T, paired_dn)
    geminal_laplacian_up_unpaired = jnp.dot(ao_matrix_laplacian_up.T, lambda_matrix_unpaired)
    geminal_laplacian_up = jnp.hstack([geminal_laplacian_up_paired, geminal_laplacian_up_unpaired])

    geminal_laplacian_dn_paired = jnp.dot(ao_matrix_up.T, jnp.dot(lambda_matrix_paired, ao_matrix_laplacian_dn))
    geminal_laplacian_dn_unpaired = jnp.zeros(
        [geminal_data.num_electron_up, geminal_data.num_electron_up - geminal_data.num_electron_dn],
        dtype=geminal_laplacian_dn_paired.dtype,
    )
    geminal_laplacian_dn = jnp.hstack([geminal_laplacian_dn_paired, geminal_laplacian_dn_unpaired])

    grad_ln_D_up_stack = jnp.einsum("gij,ji->gi", geminal_grad_up, geminal_inverse)
    grad_ln_D_dn_stack = jnp.einsum("ij,gji->gi", geminal_inverse, geminal_grad_dn)

    grad_ln_D_up = grad_ln_D_up_stack.T
    grad_ln_D_dn = grad_ln_D_dn_stack.T

    grad_ln_D_up_x, grad_ln_D_up_y, grad_ln_D_up_z = grad_ln_D_up_stack
    grad_ln_D_dn_x, grad_ln_D_dn_y, grad_ln_D_dn_z = grad_ln_D_dn_stack

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
