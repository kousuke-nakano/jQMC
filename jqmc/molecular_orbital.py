"""Molecular Orbital module."""

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
from logging import getLogger

# jax modules
# from jax.debug import print as jprint
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import jit
from jax import typing as jnpt

# myqmc module
from .atomic_orbital import (
    AOs_cart_data,
    AOs_sphe_data,
    _compute_AOs_debug,
    _compute_AOs_grad_autodiff,
    _compute_AOs_laplacian_autodiff,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
)

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


@struct.dataclass
class MOs_data:
    """Molecular orbital (MO) coefficients and metadata.

    Holds the contraction matrix that maps atomic orbitals (AOs) to molecular orbitals (MOs).
    MO values are obtained as ``mo_coefficients @ AO_values`` in float64 (``jax_enable_x64=True``).

    Attributes:
        num_mo (int): Number of molecular orbitals.
        aos_data (AOs_sphe_data | AOs_cart_data): AO definition supplying centers, exponents/coefficients,
            angular data, and contraction mapping.
        mo_coefficients (npt.NDArray | jax.Array): Coefficient matrix of shape ``(num_mo, num_ao)``. Rows
            correspond to MOs; columns correspond to contracted AOs.

    Examples:
        Minimal runnable setup (2 AOs -> 1 MO)::

            import numpy as np
            from jqmc.structure import Structure_data
            from jqmc.atomic_orbital import AOs_sphe_data
            from jqmc.molecular_orbital import MOs_data

            structure = Structure_data(
                positions=[[0.0, 0.0, -0.70], [0.0, 0.0, 0.70]],
                pbc_flag=False,
                atomic_numbers=[1, 1],
                element_symbols=["H", "H"],
                atomic_labels=["H1", "H2"],
            )

            aos = AOs_sphe_data(
                structure_data=structure,
                nucleus_index=[0, 1],
                num_ao=2,
                num_ao_prim=2,
                orbital_indices=[0, 1],
                exponents=[1.0, 1.2],
                coefficients=[1.0, 0.8],
                angular_momentums=[0, 0],
                magnetic_quantum_numbers=[0, 0],
            )
            aos.sanity_check()

            mo_coeffs = np.array([[0.7, 0.7]], dtype=float)  # shape (1, 2)
            mos = MOs_data(num_mo=1, aos_data=aos, mo_coefficients=mo_coeffs)
            mos.sanity_check()
    """

    #: Number of molecular orbitals.
    num_mo: int = struct.field(pytree_node=False, default=0)
    #: AO definition supplying centers, exponents/coefficients, angular data, and contraction mapping.
    aos_data: AOs_sphe_data | AOs_cart_data = struct.field(pytree_node=True, default_factory=lambda: AOs_sphe_data())
    #: MO coefficient matrix, shape ``(num_mo, num_ao)``.
    mo_coefficients: npt.NDArray | jnpt.ArrayLike = struct.field(pytree_node=True, default_factory=lambda: np.array([]))

    def sanity_check(self) -> None:
        """Validate internal consistency.

        Ensures ``mo_coefficients`` matches ``(num_mo, aos_data.num_ao)``, verifies ``num_mo`` is an int,
        and delegates AO validation to ``aos_data.sanity_check()``.

        Raises:
            ValueError: If coefficient shape or ``num_mo`` type is invalid, or if ``aos_data`` fails its check.
        """
        if self.mo_coefficients.shape != (self.num_mo, self.aos_data.num_ao):
            raise ValueError(
                f"dim. of ao_coefficients = {self.mo_coefficients.shape} is wrong. Inconsistent with the expected value = {(self.num_mo, self.aos_data.num_ao)}"
            )
        if not isinstance(self.num_mo, (int, np.integer)):
            raise ValueError(f"num_mo = {type(self.num_mo)} must be an int.")
        self.aos_data.sanity_check()

    def _get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  Number of MOs = {self.num_mo}")
        info_lines.append(f"  dim. of MOs coeff = {self.mo_coefficients.shape}")
        # Replace aos_data.logger_info() with aos_data.get_info() output.
        info_lines.extend(self.aos_data._get_info())
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @property
    def structure_data(self):
        """Return structure_data of the aos_data instance."""
        return self.aos_data.structure_data

    @property
    def _num_orb(self) -> int:
        """Return the number of orbitals."""
        return self.num_mo


@jit
def compute_MOs(mos_data: MOs_data, r_carts: jax.Array) -> jax.Array:
    """Evaluate molecular orbitals at electron coordinates.

    Args:
        mos_data (MOs_data): MO/AO definition and coefficient matrix.
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)`` in float64 (same convention as
            AO evaluators).

    Returns:
        jax.Array: MO values with shape ``(num_mo, N_e)``.
    """
    answer = jnp.dot(
        mos_data.mo_coefficients,
        compute_AOs(aos_data=mos_data.aos_data, r_carts=r_carts),
    )

    return answer


def _compute_MOs_debug(mos_data: MOs_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """See _api method."""
    answer = np.dot(
        mos_data.mo_coefficients,
        _compute_AOs_debug(aos_data=mos_data.aos_data, r_carts=r_carts),
    )
    return answer


@jit
def _compute_MOs_laplacian_autodiff(mos_data: MOs_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """See _api method."""
    mo_matrix_laplacian = jnp.dot(
        mos_data.mo_coefficients,
        _compute_AOs_laplacian_autodiff(mos_data.aos_data, r_carts),
    )

    return mo_matrix_laplacian


@jit
def compute_MOs_laplacian(mos_data: MOs_data, r_carts: jax.Array) -> jax.Array:
    """Compute MO laplacians at electron coordinates.

    Args:
        mos_data (MOs_data): MO/AO definition and coefficient matrix.
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)`` in float64.

    Returns:
        jax.Array: Laplacians of each MO, shape ``(num_mo, N_e)``.
    """
    ao_lap = compute_AOs_laplacian(mos_data.aos_data, r_carts)
    return jnp.dot(mos_data.mo_coefficients, ao_lap)


def _compute_MOs_laplacian_debug(mos_data: MOs_data, r_carts: npt.NDArray[np.float64]):
    """See _api method."""
    # Laplacians of AOs (numerical)
    diff_h = 1.0e-5

    mo_matrix = compute_MOs(mos_data, r_carts)

    # laplacians x^2
    diff_p_x_r_carts = r_carts.copy()
    diff_p_x_r_carts[:, 0] += diff_h
    mo_matrix_diff_p_x = compute_MOs(mos_data, diff_p_x_r_carts)
    diff_m_x_r_carts = r_carts.copy()
    diff_m_x_r_carts[:, 0] -= diff_h
    mo_matrix_diff_m_x = compute_MOs(mos_data, diff_m_x_r_carts)

    # laplacians y^2
    diff_p_y_r_carts = r_carts.copy()
    diff_p_y_r_carts[:, 1] += diff_h
    mo_matrix_diff_p_y = compute_MOs(mos_data, diff_p_y_r_carts)
    diff_m_y_r_carts = r_carts.copy()
    diff_m_y_r_carts[:, 1] -= diff_h
    mo_matrix_diff_m_y = compute_MOs(mos_data, diff_m_y_r_carts)

    # laplacians z^2
    diff_p_z_r_carts = r_carts.copy()
    diff_p_z_r_carts[:, 2] += diff_h
    mo_matrix_diff_p_z = compute_MOs(mos_data, diff_p_z_r_carts)
    diff_m_z_r_carts = r_carts.copy()
    diff_m_z_r_carts[:, 2] -= diff_h
    mo_matrix_diff_m_z = compute_MOs(mos_data, diff_m_z_r_carts)

    mo_matrix_grad2_x = (mo_matrix_diff_p_x + mo_matrix_diff_m_x - 2 * mo_matrix) / (diff_h) ** 2
    mo_matrix_grad2_y = (mo_matrix_diff_p_y + mo_matrix_diff_m_y - 2 * mo_matrix) / (diff_h) ** 2
    mo_matrix_grad2_z = (mo_matrix_diff_p_z + mo_matrix_diff_m_z - 2 * mo_matrix) / (diff_h) ** 2

    mo_matrix_laplacian = mo_matrix_grad2_x + mo_matrix_grad2_y + mo_matrix_grad2_z

    return mo_matrix_laplacian


@jit
def compute_MOs_grad(
    mos_data: MOs_data, r_carts: jax.Array
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Compute MO gradients (x, y, z components) at electron coordinates.

    Args:
        mos_data (MOs_data): MO/AO definition and coefficient matrix.
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)`` in float64.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: Gradients per component
        ``(grad_x, grad_y, grad_z)``, each of shape ``(num_mo, N_e)``.
    """
    mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z = compute_AOs_grad(mos_data.aos_data, r_carts)
    mo_matrix_grad_x = jnp.dot(mos_data.mo_coefficients, mo_matrix_grad_x)
    mo_matrix_grad_y = jnp.dot(mos_data.mo_coefficients, mo_matrix_grad_y)
    mo_matrix_grad_z = jnp.dot(mos_data.mo_coefficients, mo_matrix_grad_z)

    return mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z


@jit
def _compute_MOs_grad_autodiff(
    mos_data: MOs_data, r_carts: npt.NDArray[np.float64]
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """This method is for computing the gradients (x,y,z) of the given molecular orbital at r_carts."""
    mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z = _compute_AOs_grad_autodiff(mos_data.aos_data, r_carts)
    mo_matrix_grad_x = jnp.dot(mos_data.mo_coefficients, mo_matrix_grad_x)
    mo_matrix_grad_y = jnp.dot(mos_data.mo_coefficients, mo_matrix_grad_y)
    mo_matrix_grad_z = jnp.dot(mos_data.mo_coefficients, mo_matrix_grad_z)

    return mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z


def _compute_MOs_grad_debug(
    mos_data: MOs_data,
    r_carts: npt.NDArray[np.float64],
):
    """See _api method."""
    # Gradients of AOs (numerical)
    diff_h = 1.0e-5

    # grad x
    diff_p_x_r_carts = r_carts.copy()
    diff_p_x_r_carts[:, 0] += diff_h
    mo_matrix_diff_p_x = compute_MOs(mos_data, diff_p_x_r_carts)
    diff_m_x_r_carts = r_carts.copy()
    diff_m_x_r_carts[:, 0] -= diff_h
    mo_matrix_diff_m_x = compute_MOs(mos_data, diff_m_x_r_carts)

    # grad y
    diff_p_y_r_carts = r_carts.copy()
    diff_p_y_r_carts[:, 1] += diff_h
    mo_matrix_diff_p_y = compute_MOs(mos_data, diff_p_y_r_carts)
    diff_m_y_r_carts = r_carts.copy()
    diff_m_y_r_carts[:, 1] -= diff_h
    mo_matrix_diff_m_y = compute_MOs(mos_data, diff_m_y_r_carts)

    # grad z
    diff_p_z_r_carts = r_carts.copy()
    diff_p_z_r_carts[:, 2] += diff_h
    mo_matrix_diff_p_z = compute_MOs(mos_data, diff_p_z_r_carts)
    diff_m_z_r_carts = r_carts.copy()
    diff_m_z_r_carts[:, 2] -= diff_h
    mo_matrix_diff_m_z = compute_MOs(mos_data, diff_m_z_r_carts)

    mo_matrix_grad_x = (mo_matrix_diff_p_x - mo_matrix_diff_m_x) / (2.0 * diff_h)
    mo_matrix_grad_y = (mo_matrix_diff_p_y - mo_matrix_diff_m_y) / (2.0 * diff_h)
    mo_matrix_grad_z = (mo_matrix_diff_p_z - mo_matrix_diff_m_z) / (2.0 * diff_h)

    return mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z


"""
if __name__ == "__main__":
    import os

    from .trexio_wrapper import read_trexio_file

    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
"""
