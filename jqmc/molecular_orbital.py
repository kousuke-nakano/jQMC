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
# * Neither the name of the phonopy project nor the names of its
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
from dataclasses import dataclass, field

# set logger
from logging import Formatter, StreamHandler, getLogger

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
    AO_sphe_data,
    AOs_cart_data,
    AOs_sphe_data,
    AOs_sphe_data_deriv_R,
    _compute_AO_sphe,
    compute_AOs_grad_jax,
    compute_AOs_jax,
    compute_AOs_laplacian_jax,
)

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


@struct.dataclass
class MOs_data:
    """The class contains data for computing a molecular orbitals.

    Args:
        num_mo: The number of MOs.
        aos_data (AOs_data): aos_data instances
        mo_coefficients (npt.NDArray[np.float64|np.complex128]): array of MO coefficients. dim. num_mo, num_ao
    """

    num_mo: int = struct.field(pytree_node=False, default=0)
    aos_data: AOs_sphe_data | AOs_cart_data = struct.field(pytree_node=True, default_factory=lambda: AOs_sphe_data())
    mo_coefficients: npt.NDArray[np.float64] = struct.field(pytree_node=True, default_factory=lambda: np.array([]))

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.mo_coefficients.shape != (self.num_mo, self.aos_data.num_ao):
            logger.error(
                f"dim. of ao_coefficients = {self.mo_coefficients.shape} is wrong. Inconsistent with the expected value = {(self.num_mo, self.aos_data.num_ao)}"
            )
            raise ValueError("The dimension of a given mo_coefficients is wrong.")

        self.aos_data.sanity_check()

    def get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  Number of MOs = {self.num_mo}")
        # Replace aos_data.logger_info() with aos_data.get_info() output.
        info_lines.extend(self.aos_data.get_info())
        return info_lines

    def logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self.get_info():
            logger.info(line)

    @property
    def structure_data(self):
        """Return structure_data of the aos_data instance."""
        return self.aos_data.structure_data

    @property
    def num_orb(self) -> int:
        """Return the number of orbitals."""
        return self.num_mo


@struct.dataclass
class MOs_data_deriv_R(MOs_data):
    """See MOs_data class."""

    num_mo: int = struct.field(pytree_node=False, default=0)
    aos_data: AOs_sphe_data | AOs_cart_data = struct.field(pytree_node=True, default_factory=lambda: AOs_sphe_data())
    mo_coefficients: npt.NDArray[np.float64] = struct.field(pytree_node=False, default_factory=lambda: np.array([]))

    @classmethod
    def from_base(cls, mos_data: MOs_data):
        """Switch pytree_node."""
        num_mo = mos_data.num_mo
        if isinstance(mos_data.aos_data, AOs_sphe_data):
            aos_data = AOs_sphe_data_deriv_R.from_base(aos_data=mos_data.aos_data)
        elif isinstance(mos_data.aos_data, AOs_cart_data):
            aos_data = AOs_cart_data.from_base(aos_data=mos_data.aos_data)  # to be implemented AOs_cart_data_deriv_R
        mo_coefficients = mos_data.mo_coefficients
        return cls(num_mo, aos_data, mo_coefficients)


@struct.dataclass
class MOs_data_no_deriv(MOs_data):
    """See MOs_data class."""

    num_mo: int = struct.field(pytree_node=False, default=0)
    aos_data: AOs_sphe_data | AOs_cart_data = struct.field(pytree_node=False, default_factory=lambda: AOs_sphe_data())
    mo_coefficients: npt.NDArray[np.float64] = struct.field(pytree_node=False, default_factory=lambda: np.array([]))

    @classmethod
    def from_base(cls, mos_data: MOs_data):
        """Switch pytree_node."""
        num_mo = mos_data.num_mo
        aos_data = mos_data.aos_data
        mo_coefficients = mos_data.mo_coefficients
        return cls(num_mo, aos_data, mo_coefficients)


def compute_MOs_jax(mos_data: MOs_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """The class contains information for computing molecular orbitals at r_carts simlunateously.

    Args:
        mos_data (MOs_data): an instance of MOs_data
        r_carts (jnpt.ArrayLike): Cartesian coordinates of electrons (dim: N_e, 3)

    Returns:
        Arrays containing values of the MOs at r_carts. (dim: num_mo, N_e)
    """
    answer = _compute_MOs_jax(mos_data, r_carts)

    if answer.shape != (mos_data.num_mo, len(r_carts)):
        logger.error(f"answer.shape = {answer.shape} is inconsistent with the expected one = {(mos_data.num_mo, len(r_carts))}")
        raise ValueError

    return answer


@jit
def _compute_MOs_jax(mos_data: MOs_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """See _api method."""
    answer = jnp.dot(
        mos_data.mo_coefficients,
        compute_AOs_jax(aos_data=mos_data.aos_data, r_carts=r_carts),
    )
    return answer


def compute_MOs_debug(mos_data: MOs_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """See _api method."""
    answer = np.dot(
        mos_data.mo_coefficients,
        compute_AOs_jax(aos_data=mos_data.aos_data, r_carts=r_carts),
    )
    return answer


#############################################################################################################
#
# The following functions are no longer used in the main code. They are kept for future reference.
#
#############################################################################################################


# no longer used in the main code
def compute_MOs_laplacian_jax(mos_data: MOs_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Function for computing laplacians with a given MOs_data.

    The method is for computing the laplacians of the given molecular orbital (MOs_data) at r_carts.

    Args:
        mo_datas (MOs_data): an instance of MOs_data
        r_carts: Cartesian coordinates of electrons (dim: N_e, 3)

    Returns:
        An array containing laplacians of the MOs at r_carts. The dim. is (num_mo, N_e)
    """
    mo_matrix_laplacian = _compute_MOs_laplacian_jax(mos_data, r_carts)

    if mo_matrix_laplacian.shape != (mos_data.num_mo, len(r_carts)):
        logger.error(
            f"mo_matrix_laplacian.shape = {mo_matrix_laplacian.shape} is inconsistent with the expected one = {mos_data.num_mo, len(r_carts)}"
        )
        raise ValueError

    return mo_matrix_laplacian


# no longer used in the main code
@jit
def _compute_MOs_laplacian_jax(mos_data: MOs_data, r_carts: npt.NDArray[np.float64]):
    """See _api method."""
    mo_matrix_laplacian = jnp.dot(
        mos_data.mo_coefficients,
        compute_AOs_laplacian_jax(mos_data.aos_data, r_carts),
    )

    return mo_matrix_laplacian


# no longer used in the main code
def compute_MOs_laplacian_debug(mos_data: MOs_data, r_carts: npt.NDArray[np.float64]):
    """See _api method."""
    # Laplacians of AOs (numerical)
    diff_h = 1.0e-5

    mo_matrix = compute_MOs_jax(mos_data, r_carts)

    # laplacians x^2
    diff_p_x_r_carts = r_carts.copy()
    diff_p_x_r_carts[:, 0] += diff_h
    mo_matrix_diff_p_x = compute_MOs_jax(mos_data, diff_p_x_r_carts)
    diff_m_x_r_carts = r_carts.copy()
    diff_m_x_r_carts[:, 0] -= diff_h
    mo_matrix_diff_m_x = compute_MOs_jax(mos_data, diff_m_x_r_carts)

    # laplacians y^2
    diff_p_y_r_carts = r_carts.copy()
    diff_p_y_r_carts[:, 1] += diff_h
    mo_matrix_diff_p_y = compute_MOs_jax(mos_data, diff_p_y_r_carts)
    diff_m_y_r_carts = r_carts.copy()
    diff_m_y_r_carts[:, 1] -= diff_h
    mo_matrix_diff_m_y = compute_MOs_jax(mos_data, diff_m_y_r_carts)

    # laplacians z^2
    diff_p_z_r_carts = r_carts.copy()
    diff_p_z_r_carts[:, 2] += diff_h
    mo_matrix_diff_p_z = compute_MOs_jax(mos_data, diff_p_z_r_carts)
    diff_m_z_r_carts = r_carts.copy()
    diff_m_z_r_carts[:, 2] -= diff_h
    mo_matrix_diff_m_z = compute_MOs_jax(mos_data, diff_m_z_r_carts)

    mo_matrix_grad2_x = (mo_matrix_diff_p_x + mo_matrix_diff_m_x - 2 * mo_matrix) / (diff_h) ** 2
    mo_matrix_grad2_y = (mo_matrix_diff_p_y + mo_matrix_diff_m_y - 2 * mo_matrix) / (diff_h) ** 2
    mo_matrix_grad2_z = (mo_matrix_diff_p_z + mo_matrix_diff_m_z - 2 * mo_matrix) / (diff_h) ** 2

    mo_matrix_laplacian = mo_matrix_grad2_x + mo_matrix_grad2_y + mo_matrix_grad2_z

    return mo_matrix_laplacian


# no longer used in the main code
def compute_MOs_grad_jax(
    mos_data: MOs_data, r_carts: npt.NDArray[np.float64]
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """This method is for computing the gradients (x,y,z) of the given molecular orbital at r_carts.

    Args:
        mo_datas (MOs_data): an instance of MOs_data
        r_carts: Cartesian coordinates of electrons (dim: N_e, 3)
        debug (bool): if True, this is computed via _debug function for debuging purpose

    Returns:
        tuple containing gradients of the MOs at r_carts. (grad_x, grad_y, grad_z). The dim. of each matrix is (num_mo, N_e)
    """
    mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z = _compute_MOs_grad_jax(mos_data, r_carts)

    if mo_matrix_grad_x.shape != (mos_data.num_mo, len(r_carts)):
        logger.error(
            f"mo_matrix_grad_x.shape = {mo_matrix_grad_x.shape} is inconsistent with the expected one = {mos_data.num_mo, len(r_carts)}"
        )
        raise ValueError

    if mo_matrix_grad_y.shape != (mos_data.num_mo, len(r_carts)):
        logger.error(
            f"mo_matrix_grad_y.shape = {mo_matrix_grad_y.shape} is inconsistent with the expected one = {mos_data.num_mo, len(r_carts)}"
        )
        raise ValueError

    if mo_matrix_grad_z.shape != (mos_data.num_mo, len(r_carts)):
        logger.error(
            f"mo_matrix_grad_z.shape = {mo_matrix_grad_z.shape} is inconsistent with the expected one = {mos_data.num_ao, len(r_carts)}"
        )
        raise ValueError

    return mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z


# no longer used in the main code
@jit
def _compute_MOs_grad_jax(
    mos_data: MOs_data,
    r_carts: npt.NDArray[np.float64],
):
    """See _api method."""
    mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z = compute_AOs_grad_jax(mos_data.aos_data, r_carts)
    mo_matrix_grad_x = jnp.dot(mos_data.mo_coefficients, mo_matrix_grad_x)
    mo_matrix_grad_y = jnp.dot(mos_data.mo_coefficients, mo_matrix_grad_y)
    mo_matrix_grad_z = jnp.dot(mos_data.mo_coefficients, mo_matrix_grad_z)

    return mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z


# no longer used in the main code
def compute_MOs_grad_debug(
    mos_data: MOs_data,
    r_carts: npt.NDArray[np.float64],
):
    """See _api method."""
    # Gradients of AOs (numerical)
    diff_h = 1.0e-5

    # grad x
    diff_p_x_r_carts = r_carts.copy()
    diff_p_x_r_carts[:, 0] += diff_h
    mo_matrix_diff_p_x = compute_MOs_jax(mos_data, diff_p_x_r_carts)
    diff_m_x_r_carts = r_carts.copy()
    diff_m_x_r_carts[:, 0] -= diff_h
    mo_matrix_diff_m_x = compute_MOs_jax(mos_data, diff_m_x_r_carts)

    # grad y
    diff_p_y_r_carts = r_carts.copy()
    diff_p_y_r_carts[:, 1] += diff_h
    mo_matrix_diff_p_y = compute_MOs_jax(mos_data, diff_p_y_r_carts)
    diff_m_y_r_carts = r_carts.copy()
    diff_m_y_r_carts[:, 1] -= diff_h
    mo_matrix_diff_m_y = compute_MOs_jax(mos_data, diff_m_y_r_carts)

    # grad z
    diff_p_z_r_carts = r_carts.copy()
    diff_p_z_r_carts[:, 2] += diff_h
    mo_matrix_diff_p_z = compute_MOs_jax(mos_data, diff_p_z_r_carts)
    diff_m_z_r_carts = r_carts.copy()
    diff_m_z_r_carts[:, 2] -= diff_h
    mo_matrix_diff_m_z = compute_MOs_jax(mos_data, diff_m_z_r_carts)

    mo_matrix_grad_x = (mo_matrix_diff_p_x - mo_matrix_diff_m_x) / (2.0 * diff_h)
    mo_matrix_grad_y = (mo_matrix_diff_p_y - mo_matrix_diff_m_y) / (2.0 * diff_h)
    mo_matrix_grad_z = (mo_matrix_diff_p_z - mo_matrix_diff_m_z) / (2.0 * diff_h)

    return mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z


# no longer used in the main code
@dataclass
class MO_data:
    """MO_data for debuggin purpose.

    The class contains data for computing a molecular orbital. Just for testing purpose.
    For fast computations, use MOs_data and MOs.

    Args:
        mo_coefficients (list[float | complex]): List of coefficients of the AO.
        ao_data_l (list[AO_Data]): List of ao_data instances
    """

    mo_coefficients: list[float | complex] = field(default_factory=list)
    ao_data_l: list[AO_sphe_data] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialization of the class.

        This magic function checks the consistencies among the arguments.
        To be implemented.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if len(self.ao_data_l) != len(self.mo_coefficients):
            logger.error("dim. of self.ao_data_l or len(self.coefficients is wrong")
            raise ValueError


# no longer used in the main code
def compute_MO(mo_data: MO_data, r_cart: list[float]) -> float:
    """Compute a MO for debugging purpose.

    The method is for computing a molecular orbital. Just for testing purpose.
    For fast computations, use MOs_data and compute_MOs.

    Args:
        mo_data (MO_data): an instance of MO_data
        r_cart: Cartesian coordinate of an electron

    Returns:
        Value of the MO value at r_cart.
    """
    return np.inner(
        np.array(mo_data.mo_coefficients),
        np.array([_compute_AO_sphe(ao_data=ao_data, r_cart=r_cart) for ao_data in mo_data.ao_data_l]),
    )


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

    logger.devel("test")

    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "water_trexio.hdf5"))

    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    # Initialization
    r_up_carts = []
    r_dn_carts = []

    total_electrons = 0

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data.positions_cart_np

    # Place electrons around each nucleus
    for i in range(len(coords)):
        charge = charges[i]
        num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

        # Retrieve the position coordinates
        x, y, z = coords[i]

        # Place electrons
        for _ in range(num_electrons):
            # Calculate distance range
            distance = np.random.uniform(1.0 / charge, 2.0 / charge)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            # Convert spherical to Cartesian coordinates
            dx = distance * np.sin(theta) * np.cos(phi)
            dy = distance * np.sin(theta) * np.sin(phi)
            dz = distance * np.cos(theta)

            # Position of the electron
            electron_position = np.array([x + dx, y + dy, z + dz])

            # Assign spin
            if len(r_up_carts) < num_electron_up:
                r_up_carts.append(electron_position)
            else:
                r_dn_carts.append(electron_position)

        total_electrons += num_electrons

    # Handle surplus electrons
    remaining_up = num_electron_up - len(r_up_carts)
    remaining_dn = num_electron_dn - len(r_dn_carts)

    # Randomly place any remaining electrons
    for _ in range(remaining_up):
        r_up_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))
    for _ in range(remaining_dn):
        r_dn_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))

    r_up_carts = np.array(r_up_carts)
    r_dn_carts = np.array(r_dn_carts)

    mos_up_debug = compute_MOs_debug(mos_data=mos_data_up, r_carts=r_up_carts)
    mos_up_jax = _compute_MOs_jax(mos_data=mos_data_up, r_carts=r_up_carts)

    print(mos_up_debug - mos_up_jax)
