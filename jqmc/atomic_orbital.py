"""Atomic Orbitals module.

Module containing classes and methods related to Atomic Orbitals

Todo:
    * Laplacian computation without JAX
    * Replace numpy and jax.numpy typings with jaxtyping
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# This file is part of phonopy.
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

from dataclasses import dataclass, field
from logging import Formatter, StreamHandler, getLogger

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
import numpy.typing as npt
import scipy
from flax import struct
from jax import grad, jacrev, jit
from jax import typing as jnpt
from jax import vmap
from numpy import linalg as LA

from .structure import Structure_data

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


@struct.dataclass
class AOs_data:
    """Atomic Orbitals dataclass.

    The class contains data for computing atomic orbitals simltaneously

    Args:
        structure_data(Structure_data):
            an instance of Structure_data
        nucleus_index (list[int]):
            One-to-one correspondence between AO items and the atom index (dim:num_ao)
        num_ao (int):
            the number of atomic orbitals.
        num_ao_prim (int):
            the number of primitive atomic orbitals.
        orbital_indices (list[int]):
            index for what exponents and coefficients are associated to each atomic orbital.
            dim: num_ao_prim
        exponents (list[float]):
            List of exponents of the AOs. dim: num_ao_prim.
        coefficients (list[float]):
            List of coefficients of the AOs. dim: num_ao_prim
        angular_momentums (list[int]):
            Angular momentum of the AOs, i.e., l. dim: num_ao
        magnetic_quantum_numbers (list[int]):
            Magnetic quantum number of the AOs, i.e m = -l .... +l. dim: num_ao
    """

    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    nucleus_index: list[int] = struct.field(pytree_node=False, default_factory=list)
    num_ao: int = struct.field(pytree_node=False, default=0)
    num_ao_prim: int = struct.field(pytree_node=False, default=0)
    orbital_indices: list[int] = struct.field(pytree_node=False, default_factory=list)
    exponents: list[float] = struct.field(pytree_node=False, default_factory=list)
    coefficients: list[float] = struct.field(pytree_node=False, default_factory=list)
    angular_momentums: list[int] = struct.field(pytree_node=False, default_factory=list)
    magnetic_quantum_numbers: list[int] = struct.field(pytree_node=False, default_factory=list)

    def __post_init__(self) -> None:
        """Initialization of the class.

        This magic function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if len(self.nucleus_index) != self.num_ao:
            logger.error("dim. of self.nucleus_index is wrong")
            raise ValueError
        if len(np.unique(self.orbital_indices)) != self.num_ao:
            logger.error(f"num_ao={self.num_ao} and/or num_ao_prim={self.num_ao_prim} is wrong")
        if len(self.exponents) != self.num_ao_prim:
            logger.error("dim. of self.exponents is wrong")
            raise ValueError
        if len(self.coefficients) != self.num_ao_prim:
            logger.error("dim. of self.coefficients is wrong")
            raise ValueError
        if len(self.angular_momentums) != self.num_ao:
            logger.error("dim. of self.angular_momentums is wrong")
            raise ValueError
        if len(self.magnetic_quantum_numbers) != self.num_ao:
            logger.error("dim. of self.magnetic_quantum_numbers is wrong")
            raise ValueError

    @property
    def atomic_center_carts(self) -> npt.NDArray[np.float64]:
        """Atomic positions in cartesian.

        Returns atomic positions in cartesian

        Returns:
            npt.NDArray[np.float64]: atomic positions in cartesian
        """
        return np.array([self.structure_data.positions_cart[i] for i in self.nucleus_index])

    @property
    def atomic_center_carts_jnp(self) -> jax.Array:
        """Atomic positions in cartesian.

        Returns atomic positions in cartesian

        Returns:
            jax.Array: atomic positions in cartesian
        """
        # this is super slow!!! Do not use list comprehension.
        # return jnp.array([self.structure_data.positions_cart[i] for i in self.nucleus_index])
        return jnp.array(self.structure_data.positions_cart)[np.array(self.nucleus_index)]

    @property
    def atomic_center_carts_prim(self) -> npt.NDArray[np.float64]:
        """Atomic positions in cartesian for primitve orbitals.

        Returns atomic positions in cartesian for primitive orbitals

        Returns:
            npt.NDArray[np.float]: atomic positions in cartesian for primitive orbitals
        """
        return np.array([self.atomic_center_carts[i] for i in self.orbital_indices])

    @property
    def atomic_center_carts_prim_jnp(self) -> jax.Array:
        """Atomic positions in cartesian for primitve orbitals.

        Returns atomic positions in cartesian for primitive orbitals

        Returns:
            jax.Array: atomic positions in cartesian for primitive orbitals
        """
        # this is super slow!!! Do not use list comprehension.
        # return jnp.array([self.atomic_center_carts_jnp[i] for i in self.orbital_indices])
        return self.atomic_center_carts_jnp[np.array(self.orbital_indices)]

    @property
    def angular_momentums_prim(self) -> npt.NDArray[np.float64]:
        """Angular momentums for primitive orbitals.

        Returns angular momentums for primitive orbitals

        Returns:
            npt.NDArray[np.float64]: angular momentums for primitive orbitals
        """
        return np.array([self.angular_momentums[i] for i in self.orbital_indices])

    @property
    def magnetic_quantum_numbers_prim(self) -> npt.NDArray[np.int64]:
        """Magnetic quantum numbers for primitive orbitals.

        Returns magnetic quantum numbers for primitive orbitals

        Returns:
            npt.NDArray[np.int64]: magnetic quantum numbers for primitive orbitals
        """
        return np.array([self.magnetic_quantum_numbers[i] for i in self.orbital_indices])


def compute_AOs_laplacian_api(aos_data: AOs_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Compute laplacians of the give AOs at r_carts.

    The method is for computing the laplacians of the given atomic orbital at r_carts

    Args:
        ao_datas (AOs_data): an instance of AOs_data
        r_carts (jnpt.ArrayLike): Cartesian coordinates of electrons (dim: N_e, 3)
        debug_flag (bool): if True, numerical derivatives are computed for debuging purpose

    Returns:
        jax.Array:
            Array containing laplacians of the AOs at r_carts. The dim. is (num_ao, N_e)

    """
    return compute_AOs_laplacian_jax(aos_data, r_carts)


@jit
def compute_AOs_laplacian_jax(aos_data: AOs_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Compute laplacians of the give AOs at r_carts.

    See compute_AOs_laplacian_api

    """
    # expansion with respect to the primitive AOs
    atomic_center_carts_dup = aos_data.atomic_center_carts_prim_jnp
    angular_momentums_dup = aos_data.angular_momentums_prim
    magnetic_quantum_numbers_dup = aos_data.magnetic_quantum_numbers_prim
    exponents = aos_data.exponents
    coefficients = aos_data.coefficients

    # compute R_n inc. the whole normalization factor
    R_carts_jnp = jnp.array(atomic_center_carts_dup)
    c_jnp = np.array(coefficients)
    Z_jnp = np.array(exponents)
    l_jnp = np.array(angular_momentums_dup)
    m_jnp = np.array(magnetic_quantum_numbers_dup)

    vmap_compute_AOs_laplacian_dup = vmap(
        vmap(
            compute_primitive_AOs_laplacians_jax,
            in_axes=(None, None, None, None, None, 0),
        ),
        in_axes=(0, 0, 0, 0, 0, None),
    )

    AOs_laplacian_dup = vmap_compute_AOs_laplacian_dup(c_jnp, Z_jnp, l_jnp, m_jnp, R_carts_jnp, r_carts)

    orbital_indices = jnp.array(aos_data.orbital_indices, dtype=jnp.int32)
    num_segments = aos_data.num_ao
    ao_matrix_laplacian = jax.ops.segment_sum(AOs_laplacian_dup, orbital_indices, num_segments=num_segments)

    return ao_matrix_laplacian


def compute_AOs_laplacian_debug(aos_data: AOs_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute laplacians of the give AOs at r_carts.

    The method is for computing the laplacians of the given atomic orbital at r_carts
    using the FDM method for debuging purpose.

    Args:
        ao_datas (AOs_data): an instance of AOs_data
        r_carts (npt.NDArray[np.float64]): Cartesian coordinates of electrons (dim: N_e, 3)
        debug_flag (bool): if True, numerical derivatives are computed for debuging purpose

    Returns:
        npt.NDArray[np.float64]:
            Array containing laplacians of the AOs at r_carts. The dim. is (num_ao, N_e)

    """
    # Laplacians of AOs (numerical)
    diff_h = 1.0e-5

    ao_matrix = compute_AOs_api(aos_data, r_carts)

    # laplacians x^2
    diff_p_x_r_carts = r_carts.copy()
    diff_p_x_r_carts[:, 0] += diff_h
    ao_matrix_diff_p_x = compute_AOs_api(aos_data, diff_p_x_r_carts)
    diff_m_x_r_carts = r_carts.copy()
    diff_m_x_r_carts[:, 0] -= diff_h
    ao_matrix_diff_m_x = compute_AOs_api(aos_data, diff_m_x_r_carts)

    # laplacians y^2
    diff_p_y_r_carts = r_carts.copy()
    diff_p_y_r_carts[:, 1] += diff_h
    ao_matrix_diff_p_y = compute_AOs_api(aos_data, diff_p_y_r_carts)
    diff_m_y_r_carts = r_carts.copy()
    diff_m_y_r_carts[:, 1] -= diff_h
    ao_matrix_diff_m_y = compute_AOs_api(aos_data, diff_m_y_r_carts)

    # laplacians z^2
    diff_p_z_r_carts = r_carts.copy()
    diff_p_z_r_carts[:, 2] += diff_h
    ao_matrix_diff_p_z = compute_AOs_api(aos_data, diff_p_z_r_carts)
    diff_m_z_r_carts = r_carts.copy()
    diff_m_z_r_carts[:, 2] -= diff_h
    ao_matrix_diff_m_z = compute_AOs_api(aos_data, diff_m_z_r_carts)

    ao_matrix_grad2_x = (ao_matrix_diff_p_x + ao_matrix_diff_m_x - 2 * ao_matrix) / (diff_h) ** 2
    ao_matrix_grad2_y = (ao_matrix_diff_p_y + ao_matrix_diff_m_y - 2 * ao_matrix) / (diff_h) ** 2
    ao_matrix_grad2_z = (ao_matrix_diff_p_z + ao_matrix_diff_m_z - 2 * ao_matrix) / (diff_h) ** 2

    ao_matrix_laplacian = ao_matrix_grad2_x + ao_matrix_grad2_y + ao_matrix_grad2_z

    if ao_matrix_laplacian.shape != (aos_data.num_ao, len(r_carts)):
        logger.error(
            f"ao_matrix_laplacian.shape = {ao_matrix_laplacian.shape} is \
                inconsistent with the expected one = {aos_data.num_ao, len(r_carts)}"
        )
        raise ValueError

    return ao_matrix_laplacian


def compute_AOs_grad_api(aos_data: AOs_data, r_carts: jnpt.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute Cartesian Gradients of AOs.

    The method is for computing the Carteisan gradients (x,y,z) of
    the given atomic orbital at r_carts

    Args:
        ao_datas(AOs_data): an instance of AOs_data
        r_carts(jnpt.ArrayLike): Cartesian coordinates of electrons (dim: N_e, 3)

    Returns:
        tuple: tuple containing gradients of the AOs at r_carts. (grad_x, grad_y, grad_z).
        The dim. of each matrix is (num_ao, N_e)

    """
    ao_matrix_grad_x, ao_matrix_grad_y, ao_matrix_grad_z = compute_AOs_grad_jax(aos_data, r_carts)

    if ao_matrix_grad_x.shape != (aos_data.num_ao, len(r_carts)):
        logger.error(
            f"aao_matrix_grad_x.shape = {ao_matrix_grad_x.shape} is \
                inconsistent with the expected one = {aos_data.num_ao, len(r_carts)}"
        )
        raise ValueError

    if ao_matrix_grad_y.shape != (aos_data.num_ao, len(r_carts)):
        logger.error(
            f"ao_matrix_grad_y.shape = {ao_matrix_grad_y.shape} is \
                inconsistent with the expected one = {aos_data.num_ao, len(r_carts)}"
        )
        raise ValueError

    if ao_matrix_grad_z.shape != (aos_data.num_ao, len(r_carts)):
        logger.error(
            f"ao_matrix_grad_z.shape = {ao_matrix_grad_y.shape} is \
                inconsistent with the expected one = {aos_data.num_ao, len(r_carts)}"
        )
        raise ValueError

    return ao_matrix_grad_x, ao_matrix_grad_y, ao_matrix_grad_z


@jit
def compute_AOs_grad_jax(aos_data: AOs_data, r_carts: jnpt.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute Cartesian Gradients of AOs.

    See compute_AOs_grad_api

    """
    # expansion with respect to the primitive AOs
    atomic_center_carts_dup = aos_data.atomic_center_carts_prim_jnp
    angular_momentums_dup = aos_data.angular_momentums_prim
    magnetic_quantum_numbers_dup = aos_data.magnetic_quantum_numbers_prim
    exponents = aos_data.exponents
    coefficients = aos_data.coefficients

    # compute R_n inc. the whole normalization factor
    R_carts_jnp = jnp.array(atomic_center_carts_dup)
    c_jnp = np.array(coefficients)
    Z_jnp = np.array(exponents)
    l_jnp = np.array(angular_momentums_dup)
    m_jnp = np.array(magnetic_quantum_numbers_dup)

    # grad in compute_primitive_AOs_grad_jax
    vmap_compute_AOs_grad_dup = vmap(
        vmap(
            compute_primitive_AOs_grad_jax,
            in_axes=(None, None, None, None, None, 0),
        ),
        in_axes=(0, 0, 0, 0, 0, None),
    )

    AOs_grad_x_dup, AOs_grad_y_dup, AOs_grad_z_dup = vmap_compute_AOs_grad_dup(c_jnp, Z_jnp, l_jnp, m_jnp, R_carts_jnp, r_carts)

    orbital_indices = jnp.array(aos_data.orbital_indices, dtype=jnp.int32)
    num_segments = aos_data.num_ao
    ao_matrix_grad_x = jax.ops.segment_sum(AOs_grad_x_dup, orbital_indices, num_segments=num_segments)
    ao_matrix_grad_y = jax.ops.segment_sum(AOs_grad_y_dup, orbital_indices, num_segments=num_segments)
    ao_matrix_grad_z = jax.ops.segment_sum(AOs_grad_z_dup, orbital_indices, num_segments=num_segments)
    return ao_matrix_grad_x, ao_matrix_grad_y, ao_matrix_grad_z


@jit
def compute_AOs_grad_jax_old(
    aos_data: AOs_data,
    r_carts: jnpt.ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute Cartesian Gradients of AOs (old).

    The method is for computing the Carteisan gradients (x,y,z) of
    the given atomic orbital at r_carts

    Args:
        ao_datas(AOs_data): an instance of AOs_data
        r_carts(jnpt.ArrayLike): Cartesian coordinates of electrons (dim: N_e, 3)

    Returns:
        tuple: tuple containing gradients of the AOs at r_carts. (grad_x, grad_y, grad_z).
        The dim. of each matrix is (num_ao, N_e)

    Note:
        Gradients of AOs (autograd via google-JAX)
        This method gives correct answers, but slow because the full Jacobian calculation is
        not needed for computing gradients.
        grad should be pluged into compute_AOs_jax() in the future for accelaration.
    """
    ao_matrix_jacrev = jacrev(compute_AOs_api, argnums=1)(aos_data, r_carts)

    ao_matrix_grad_x_ = ao_matrix_jacrev[:, :, :, 0]
    ao_matrix_grad_y_ = ao_matrix_jacrev[:, :, :, 1]
    ao_matrix_grad_z_ = ao_matrix_jacrev[:, :, :, 2]
    ao_matrix_grad_x = jnp.sum(ao_matrix_grad_x_, axis=2)
    ao_matrix_grad_y = jnp.sum(ao_matrix_grad_y_, axis=2)
    ao_matrix_grad_z = jnp.sum(ao_matrix_grad_z_, axis=2)

    return ao_matrix_grad_x, ao_matrix_grad_y, ao_matrix_grad_z


def compute_AOs_numerical_grad(
    aos_data: AOs_data,
    r_carts: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Cartesian Gradients of AOs.

    The method is for computing the Carteisan gradients (x,y,z) of
    the given atomic orbital at r_carts using FDM for debugging JAX
    implementations

    Args:
        ao_datas(AOs_data): an instance of AOs_data
        r_carts(npt.NDArray[np.float64]): Cartesian coordinates of electrons (dim: N_e, 3)

    Returns:
        tuple: tuple containing gradients of the AOs at r_carts. (grad_x, grad_y, grad_z).
        The dim. of each matrix is (num_ao, N_e)

    """
    # Gradients of AOs (numerical)
    diff_h = 1.0e-5

    # grad x
    diff_p_x_r_carts = r_carts.copy()
    diff_p_x_r_carts[:, 0] += diff_h
    ao_matrix_diff_p_x = compute_AOs_api(aos_data, diff_p_x_r_carts)
    diff_m_x_r_carts = r_carts.copy()
    diff_m_x_r_carts[:, 0] -= diff_h
    ao_matrix_diff_m_x = compute_AOs_api(aos_data, diff_m_x_r_carts)

    # grad y
    diff_p_y_r_carts = r_carts.copy()
    diff_p_y_r_carts[:, 1] += diff_h
    ao_matrix_diff_p_y = compute_AOs_api(aos_data, diff_p_y_r_carts)
    diff_m_y_r_carts = r_carts.copy()
    diff_m_y_r_carts[:, 1] -= diff_h
    ao_matrix_diff_m_y = compute_AOs_api(aos_data, diff_m_y_r_carts)

    # grad z
    diff_p_z_r_carts = r_carts.copy()
    diff_p_z_r_carts[:, 2] += diff_h
    ao_matrix_diff_p_z = compute_AOs_api(aos_data, diff_p_z_r_carts)
    diff_m_z_r_carts = r_carts.copy()
    diff_m_z_r_carts[:, 2] -= diff_h
    ao_matrix_diff_m_z = compute_AOs_api(aos_data, diff_m_z_r_carts)

    ao_matrix_grad_x = (ao_matrix_diff_p_x - ao_matrix_diff_m_x) / (2.0 * diff_h)
    ao_matrix_grad_y = (ao_matrix_diff_p_y - ao_matrix_diff_m_y) / (2.0 * diff_h)
    ao_matrix_grad_z = (ao_matrix_diff_p_z - ao_matrix_diff_m_z) / (2.0 * diff_h)

    return ao_matrix_grad_x, ao_matrix_grad_y, ao_matrix_grad_z


def compute_AOs_api(aos_data: AOs_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Compute AO values at the given r_carts.

    The method is for computing the value of the given atomic orbital at r_carts

    Args:
        ao_datas (AOs_data): an instance of AOs_data
        r_carts (jnpt.ArrayLike): Cartesian coordinates of electrons (dim: N_e, 3)

    Returns:
        jax.Array: Arrays containing values of the AOs at r_carts. (dim: num_ao, N_e)
    """
    AOs = compute_AOs_jax(aos_data, r_carts)

    if AOs.shape != (aos_data.num_ao, len(r_carts)):
        logger.error(
            f"AOs.shape = {AOs.shape} is inconsistent with the expected one \
                = {(aos_data.num_ao, len(r_carts))}"
        )
        raise ValueError

    return AOs


def compute_AOs_debug(aos_data: AOs_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute AO values at the given r_carts.

    The method is for computing the value of the given atomic orbital at r_carts
    for debugging purpose.

    Args:
        ao_datas (AOs_data): an instance of AOs_data
        r_carts (npt.NDArray[np.float64]): Cartesian coordinates of electrons (dim: N_e, 3)

    Returns:
        Arrays containing values of the AOs at r_carts. (dim: num_ao, N_e)
    """

    def compute_each_AO(ao_index):
        atomic_center_cart = aos_data.atomic_center_carts[ao_index]
        shell_indices = [i for i, v in enumerate(aos_data.orbital_indices) if v == ao_index]
        exponents = [aos_data.exponents[i] for i in shell_indices]
        coefficients = [aos_data.coefficients[i] for i in shell_indices]
        angular_momentum = aos_data.angular_momentums[ao_index]
        magnetic_quantum_number = aos_data.magnetic_quantum_numbers[ao_index]
        num_ao_prim = len(exponents)

        ao_data = AO_data(
            num_ao_prim=num_ao_prim,
            atomic_center_cart=atomic_center_cart,
            exponents=exponents,
            coefficients=coefficients,
            angular_momentum=angular_momentum,
            magnetic_quantum_number=magnetic_quantum_number,
        )

        ao_values = np.array([compute_AO(ao_data=ao_data, r_cart=r_cart) for r_cart in r_carts])

        return ao_values

    aos_values = np.array([compute_each_AO(ao_index) for ao_index in range(aos_data.num_ao)])

    return aos_values


@jit
def compute_AOs_jax(aos_data: AOs_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Compute AO values at the given r_carts.

    See compute_AOs_api

    """
    # Indices with respect to the contracted AOs
    atomic_center_carts_dup = aos_data.atomic_center_carts_prim_jnp
    angular_momentums_dup = aos_data.angular_momentums_prim
    magnetic_quantum_numbers_dup = aos_data.magnetic_quantum_numbers_prim
    exponents = aos_data.exponents
    coefficients = aos_data.coefficients

    # compute R_n inc. the whole normalization factor
    R_carts_jnp = jnp.array(atomic_center_carts_dup)
    c_jnp = np.array(coefficients)
    Z_jnp = np.array(exponents)
    l_jnp = np.array(angular_momentums_dup)
    m_jnp = np.array(magnetic_quantum_numbers_dup)

    vmap_compute_AOs_dup = vmap(
        vmap(compute_primitive_AOs_jax, in_axes=(None, None, None, None, None, 0)),
        in_axes=(0, 0, 0, 0, 0, None),
    )

    AOs_dup = vmap_compute_AOs_dup(c_jnp, Z_jnp, l_jnp, m_jnp, R_carts_jnp, r_carts)

    orbital_indices = jnp.array(aos_data.orbital_indices, dtype=jnp.int32)
    num_segments = aos_data.num_ao
    AOs = jax.ops.segment_sum(AOs_dup, orbital_indices, num_segments=num_segments)
    return AOs


@dataclass
class AO_data:
    """AO data class for debugging.

    The class contains data for computing an atomic orbital. Just for testing purpose.
    For fast computations, use AOs_data and AOs.

    Args:
        num_ao : the number of atomic orbitals.
        num_ao_prim : the number of primitive atomic orbitals.
        atomic_center_cart (list[float]): Center of the nucleus associated to the AO. dim: 3
        exponents (list[float]): List of exponents of the AO. dim: num_ao_prim
        coefficients (list[float | complex]): List of coefficients of the AO. dim: num_ao_prim
        angular_momentum (int): Angular momentum of the AO, i.e., l. dim: 1
        magnetic_quantum_number (int): Magnetic quantum number of the AO, i.e m = -l .... +l. dim: 1
    """

    num_ao_prim: int = 0
    atomic_center_cart: list[float] = field(default_factory=list)
    exponents: list[float] = field(default_factory=list)
    coefficients: list[float | complex] = field(default_factory=list)
    angular_momentum: int = 0
    magnetic_quantum_number: int = 0

    def __post_init__(self) -> None:
        """Initialization of the class.

        This magic function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if len(self.atomic_center_cart) != 3:
            logger.error("dim. of atomic_center_cart is wrong")
            raise ValueError
        if len(self.exponents) != self.num_ao_prim:
            logger.error("dim. of self.exponents is wrong")
            raise ValueError
        if len(self.coefficients) != self.num_ao_prim:
            logger.error("dim. of self.coefficients is wrong")
            raise ValueError
        if self.angular_momentum < np.abs(self.magnetic_quantum_number):
            logger.error("angular_momentum(l) is smaller than magnetic_quantum_number(|m|).")
            raise ValueError


def compute_AO(ao_data: AO_data, r_cart: list[float]) -> float:
    r"""Compute single AO for debugging.

    The method is for computing the value of the given atomic orbital at r_cart
    Just for testing purpose. For fast computations, use AOs_data and AOs.

    Args:
        ao_data (AO_data): an instance of AO_data
        r_cart (list[float]): Cartesian coordinate of an electron

    Returns:
        Value of the AO value at r_cart.

    Note:
        The faster way to compute all AOs at the same time because one can avoid X-times calling np.exp and np.sphe calls.

        Atomic orbitals are given in the followng Gaussian form:
        \phi_{l+\pm |m|, \alpha}(\vec{r}) =
            e^{-Z_\alpha * |\vec{R_\alpha} - \vec{r}|^2} * |\vec{R_\alpha} - \vec{r}|^l [Y_{l,m,\alpha}(\phi, \theta) +- Y_{l,-m,\alpha}(\phi, \theta)]
        where [Y_{l,m,\alpha}(\phi, \theta) +- Y_{l,-m,\alpha}(\phi, \theta)] are real spherical harmonics function.

        As written in the following, the spherical harmonics function is not used in practice because it has singular points.
        Instead, the so-called solid harmonics function is computed, which is defined as
        Sha_{l,\pm|m|,\alpha} = |\vec{R_{\alpha} - \vec{r}|^l [Y_{l,m,\alpha}(\phi, \theta) +- Y_{l,-m,\alpha}(\phi, \theta)].

        Rad{\alpha}(r_cart) = e^{-Z_\alpha * |\vec{R_\alpha} - \vec{r}|^2}

        Finally, an AO, \phi_{l+\pm |m|, \alpha}(\vec{r}), is computed as:
            \phi_{l+\pm |m|, \alpha}(\vec{r})  = \mathcal{N}_{l,\alpha} * Rad{\alpha}(r_cart) * Sha_{l,\pm|m|,\alpha}(r_cart)
        where N is the normalization factor. N is computed as:
            \mathcal{N}_{l,\alpha} = \sqrt{\frac{2^{2l+3}(l+1)!(2Z_\alpha)^{l+\frac{3}{2}}}{(2l+2)!\sqrt{\pi}}}.
        Notice that this normalization factor is just for the primitive GTO. The contracted GTO is not explicitly normalized.
    """
    R_n = np.array(
        [
            compute_R_n_debug(
                coefficient=c,
                exponent=Z,
                R_cart=ao_data.atomic_center_cart,
                r_cart=r_cart,
            )
            for c, Z in zip(ao_data.coefficients, ao_data.exponents)
        ]
    )
    N_n_l = np.array([compute_normalization_fator_debug(ao_data.angular_momentum, Z) for Z in ao_data.exponents])
    S_l_m = compute_S_l_m_debug(
        atomic_center_cart=ao_data.atomic_center_cart,
        angular_momentum=ao_data.angular_momentum,
        magnetic_quantum_number=ao_data.magnetic_quantum_number,
        r_cart=r_cart,
    )

    return np.sum(N_n_l * R_n) * np.sqrt((2 * ao_data.angular_momentum + 1) / (4 * np.pi)) * S_l_m


def compute_R_n_debug(
    coefficient: float,
    exponent: float,
    R_cart: list[float],
    r_cart: list[float],
) -> float:
    """Radial part of the primitive AO.

    Compute Radial part of a primitive AO, for debugging

    Args:
        coefficient (float): the coefficient of the target AO.
        exponent (float): the exponent of the target AO.
        R_cart (list[float]): Center of the nucleus associated to the AO.
        r_cart (list[float]): Cartesian coordinate of an electron

    Returns:
        float: Value of the pure radial part.
    """
    return coefficient * np.exp(-1.0 * exponent * LA.norm(np.array(r_cart) - np.array(R_cart)) ** 2)


@jit
def compute_R_n_jax(
    coefficient: float,
    exponent: float,
    R_cart: npt.NDArray[np.float64],
    r_cart: npt.NDArray[np.float64],
) -> float:
    """Radial part of a primitive AO.

    Compute Radial part of a primitive AO, for debugging

    Args:
        coefficient (float): the coefficient of the target AO.
        exponent (float): the exponent of the target AO.
        R_cart (npt.NDArray[np.float64]): Center of the nucleus associated to the AO.
        r_cart (npt.NDArray[np.float64]): Cartesian coordinate of an electron

    Returns:
        float: Value of the pure radial part.
    """
    return coefficient * jnp.exp(-1.0 * exponent * jnp.linalg.norm(r_cart - R_cart) ** 2)


def compute_S_l_m_debug(
    angular_momentum: int,
    magnetic_quantum_number: int,
    atomic_center_cart: list[float],
    r_cart: list[float],
) -> float:
    r"""Solid harmonics part of a primitve AO.

    Compute the solid harmonics, i.e., r^l * spherical hamonics part (c.f., regular solid harmonics) of a given AO

    Args:
        angular_momentum (int): Angular momentum of the AO, i.e., l
        magnetic_quantum_number (int): Magnetic quantum number of the AO, i.e m = -l .... +l
        atomic_center_cart (list[float]): Center of the nucleus associated to the AO.
        r_cart (list[float]): Cartesian coordinate of an electron

    Returns:
        float: Value of the spherical harmonics part * r^l (i.e., regular solid harmonics).

    Note:
        A real basis of spherical harmonics Y_{l,m} : S^2 -> R can be defined in terms of
        their complex analogues  Y_{l}^{m} : S^2 -> C by setting:
        Y_{l,m}(theta, phi) =
                sqrt(2) * (-1)^m * \Im[Y_l^{|m|}] (if m < 0)
                Y_l^{0} (if m = 0)
                sqrt(2) * (-1)^m * \Re[Y_l^{|m|}] (if m > 0)

        A conversion from cartesian to spherical coordinate is:
                r = sqrt(x**2 + y**2 + z**2)
                theta = arccos(z/r)
                phi = sgn(y)arccos(x/sqrt(x**2+y**2))

        It indicates that there are two singular points
                1) the origin (x,y,z) = (0,0,0)
                2) points on the z axis (0,0,z)

        Therefore, instead, the so-called solid harmonics function is computed, which is defined as
        S_{l,\pm|m|} = \sqrt(\cfrac{4 * np.pi}{2 * l + 1}) * |\vec{R} - \vec{r}|^l [Y_{l,m,\alpha}(\phi, \theta) +- Y_{l,-m,\alpha}(\phi, \theta)].

        The real solid harmonics function are tabulated in many textbooks and websites such as Wikipedia.
        They can be hardcoded into a code, or they can be computed analytically (e.g., https://en.wikipedia.org/wiki/Solid_harmonics).
        The latter one is the strategy employed in this code,
    """
    R_cart = atomic_center_cart
    x, y, z = np.array(r_cart) - np.array(R_cart)
    r_norm = LA.norm(np.array(r_cart) - np.array(R_cart))
    l, m = angular_momentum, magnetic_quantum_number
    m_abs = np.abs(m)

    # solid harmonics for (x,y) dependent part:
    def A_m(x: float, y: float) -> float:
        return np.sum(
            [
                scipy.special.binom(m_abs, p) * x ** (p) * y ** (m_abs - p) * np.cos((m_abs - p) * (np.pi / 2.0))
                for p in range(0, m_abs + 1)
            ]
        )

    def B_m(x: float, y: float) -> float:
        return np.sum(
            [
                scipy.special.binom(m_abs, p) * x ** (p) * y ** (m_abs - p) * np.sin((m_abs - p) * (np.pi / 2.0))
                for p in range(0, m_abs + 1)
            ]
        )

    # solid harmonics for (z) dependent part:
    def lambda_lm(k: int) -> float:
        # logger.debug(f"l={l}, type ={type(l)}")
        return (
            (-1.0) ** (k)
            * 2.0 ** (-l)
            * scipy.special.binom(l, k)
            * scipy.special.binom(2 * l - 2 * k, l)
            * scipy.special.factorial(l - 2 * k)
            / scipy.special.factorial(l - 2 * k - m_abs)
        )

    # solid harmonics for (z) dependent part:
    def Lambda_lm(r_norm: float, z: float) -> float:
        return np.sqrt(
            (2 - int(m_abs == 0)) * scipy.special.factorial(l - m_abs) / scipy.special.factorial(l + m_abs)
        ) * np.sum([lambda_lm(k) * r_norm ** (2 * k) * z ** (l - 2 * k - m_abs) for k in range(0, int((l - m_abs) / 2) + 1)])

    # solid harmonics eveluated in Cartesian coord. (x,y,z):
    if m >= 0:
        gamma = Lambda_lm(r_norm, z) * A_m(x, y)
    if m < 0:
        gamma = Lambda_lm(r_norm, z) * B_m(x, y)
    return gamma


@jit
def compute_S_l_m_jax(
    l: int,
    m: int,
    R_cart: npt.NDArray[np.float64],
    r_cart: npt.NDArray[np.float64],
) -> float:
    r"""Solid harmonics part of a primitve AO.

    Compute the solid harmonics, i.e., r^l * spherical hamonics part (c.f., regular solid harmonics) of a given AO

    Args:
        angular_momentum (int): Angular momentum of the AO, i.e., l
        magnetic_quantum_number (int): Magnetic quantum number of the AO, i.e m = -l .... +l
        atomic_center_cart (npt.NDArray[np.float64]): Center of the nucleus associated to the AO.
        r_cart (npt.NDArray[np.float64]): Cartesian coordinate of an electron

    Returns:
        float: Value of the spherical harmonics part * r^l (i.e., regular solid harmonics).

    Note:
        See compute_S_l_m_debug for the details.
    """
    r_cart_rel = jnp.array(r_cart) - jnp.array(R_cart)
    x, y, z = r_cart_rel[..., 0], r_cart_rel[..., 1], r_cart_rel[..., 2]
    r_norm = jnp.sqrt(x**2 + y**2 + z**2)

    conditions = [
        (l == 0) & (m == 0),
        (l == 1) & (m == -1),
        (l == 1) & (m == 0),
        (l == 1) & (m == 1),
        (l == 2) & (m == -2),
        (l == 2) & (m == -1),
        (l == 2) & (m == 0),
        (l == 2) & (m == 1),
        (l == 2) & (m == 2),
        (l == 3) & (m == -3),
        (l == 3) & (m == -2),
        (l == 3) & (m == -1),
        (l == 3) & (m == 0),
        (l == 3) & (m == 1),
        (l == 3) & (m == 2),
        (l == 3) & (m == 3),
        (l == 4) & (m == -4),
        (l == 4) & (m == -3),
        (l == 4) & (m == -2),
        (l == 4) & (m == -1),
        (l == 4) & (m == 0),
        (l == 4) & (m == 1),
        (l == 4) & (m == 2),
        (l == 4) & (m == 3),
        (l == 4) & (m == 4),
        (l == 5) & (m == -5),
        (l == 5) & (m == -4),
        (l == 5) & (m == -3),
        (l == 5) & (m == -2),
        (l == 5) & (m == -1),
        (l == 5) & (m == 0),
        (l == 5) & (m == 1),
        (l == 5) & (m == 2),
        (l == 5) & (m == 3),
        (l == 5) & (m == 4),
        (l == 5) & (m == 5),
        (l == 6) & (m == -6),
        (l == 6) & (m == -5),
        (l == 6) & (m == -4),
        (l == 6) & (m == -3),
        (l == 6) & (m == -2),
        (l == 6) & (m == -1),
        (l == 6) & (m == 0),
        (l == 6) & (m == 1),
        (l == 6) & (m == 2),
        (l == 6) & (m == 3),
        (l == 6) & (m == 4),
        (l == 6) & (m == 5),
        (l == 6) & (m == 6),
    ]

    def lnorm(l):
        return jnp.sqrt((4 * np.pi) / (2 * l + 1))

    """see https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics (l=0-4)"""
    """Useful tool to generate spherical harmonics generator [https://github.com/elerac/sh_table]"""
    S_l_m_values = [
        # s orbital
        lnorm(l=0) * 1.0 / 2.0 * jnp.sqrt(1.0 / jnp.pi) * r_norm**0.0,  # (l, m) == (0, 0)
        # p orbitals
        lnorm(l=1) * jnp.sqrt(3.0 / (4 * jnp.pi)) * y,  # (l, m) == (1, -1)
        lnorm(l=1) * jnp.sqrt(3.0 / (4 * jnp.pi)) * z,  # (l, m) == (1, 0)
        lnorm(l=1) * jnp.sqrt(3.0 / (4 * jnp.pi)) * x,  # (l, m) == (1, 1)
        # d orbitals
        lnorm(l=2) * 1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * x * y,  # (l, m) == (2, -2)
        lnorm(l=2) * 1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * y * z,  # (l, m) == (2, -1)
        lnorm(l=2) * 1.0 / 4.0 * jnp.sqrt(5.0 / (jnp.pi)) * (3 * z**2 - r_norm**2),  # (l, m) == (2, 0):
        lnorm(l=2) * 1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * x * z,  # (l, m) == (2, 1)
        lnorm(l=2) * 1.0 / 4.0 * jnp.sqrt(15.0 / (jnp.pi)) * (x**2 - y**2),  # (l, m) == (2, 2)
        # f orbitals
        lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * y * (3 * x**2 - y**2),  # (l, m) == (3, -3)
        lnorm(l=3) * 1.0 / 2.0 * jnp.sqrt(105.0 / (jnp.pi)) * x * y * z,  # (l, m) == (3, -2)
        lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(21.0 / (2 * jnp.pi)) * y * (5 * z**2 - r_norm**2),  # (l, m) == (3, -1)
        lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(7.0 / (jnp.pi)) * (5 * z**3 - 3 * z * r_norm**2),  # (l, m) == (3, 0)
        lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(21.0 / (2 * jnp.pi)) * x * (5 * z**2 - r_norm**2),  # (l, m) == (3, 1)
        lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(105.0 / (jnp.pi)) * (x**2 - y**2) * z,  # (l, m) == (3, 2)
        lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * x * (x**2 - 3 * y**2),  # (l, m) == (3, 3)
        # g orbitals
        lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(35.0 / (jnp.pi)) * x * y * (x**2 - y**2),  # (l, m) == (4, -4)
        lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * y * z * (3 * x**2 - y**2),  # (l, m) == (4, -3)
        lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(5.0 / (jnp.pi)) * x * y * (7 * z**2 - r_norm**2),  # (l, m) == (4, -2)
        (lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(5.0 / (2 * jnp.pi)) * y * (7 * z**3 - 3 * z * r_norm**2)),  # (l, m) == (4, -1)
        (
            lnorm(l=4) * 3.0 / 16.0 * jnp.sqrt(1.0 / (jnp.pi)) * (35 * z**4 - 30 * z**2 * r_norm**2 + 3 * r_norm**4)
        ),  # (l, m) == (4, 0)
        (lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(5.0 / (2 * jnp.pi)) * x * (7 * z**3 - 3 * z * r_norm**2)),  # (l, m) == (4, 1)
        (lnorm(l=4) * 3.0 / 8.0 * jnp.sqrt(5.0 / (jnp.pi)) * (x**2 - y**2) * (7 * z**2 - r_norm**2)),  # (l, m) == (4, 2)
        lnorm(l=4) * (3.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * x * z * (x**2 - 3 * y**2)),  # (l, m) == (4, 3)
        (
            lnorm(l=4) * 3.0 / 16.0 * jnp.sqrt(35.0 / (jnp.pi)) * (x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2))
        ),  # (l, m) == (4, 4)
        lnorm(5) * 3.0 / 16.0 * jnp.sqrt(77.0 / (2 * jnp.pi)) * (5 * x**4 * y - 10 * x**2 * y**3 + y**5),  # (l, m) == (5, -5)
        lnorm(5) * 3.0 / 16.0 * jnp.sqrt(385.0 / jnp.pi) * 4 * x * y * z * (x**2 - y**2),  # (l, m) == (5, -4)
        lnorm(5)
        * 1.0
        / 16.0
        * np.sqrt(385.0 / (2 * np.pi))
        * -1
        * (y**3 - 3 * x**2 * y)
        * (9 * z**2 - (x**2 + y**2 + z**2)),  # (l, m) == (5, -3)
        lnorm(5) * 1.0 / 8.0 * jnp.sqrt(1155 / jnp.pi) * 2 * x * y * (3 * z**3 - z * (x**2 + y**2 + z**2)),  # (l, m) == (5, -2)
        lnorm(5)
        * 1.0
        / 16.0
        * jnp.sqrt(165 / jnp.pi)
        * y
        * (21 * z**4 - 14 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2),  # (l, m) == (5, -1)
        lnorm(5)
        * 1.0
        / 16.0
        * jnp.sqrt(11 / jnp.pi)
        * (63 * z**5 - 70 * z**3 * (x**2 + y**2 + z**2) + 15 * z * (x**2 + y**2 + z**2) ** 2),  # (l, m) == (5, 0)
        lnorm(5)
        * 1.0
        / 16.0
        * jnp.sqrt(165 / jnp.pi)
        * x
        * (21 * z**4 - 14 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2),  # (l, m) == (5, 1)
        lnorm(5)
        * 1.0
        / 8.0
        * jnp.sqrt(1155 / jnp.pi)
        * (x**2 - y**2)
        * (3 * z**3 - z * (x**2 + y**2 + z**2)),  # (l, m) == (5, 2)
        lnorm(5)
        * 1.0
        / 16.0
        * jnp.sqrt(385.0 / (2 * jnp.pi))
        * (x**3 - 3 * x * y**2)
        * (9 * z**2 - (x**2 + y**2 + z**2)),  # (l, m) == (5, 3)
        lnorm(5)
        * 3.0
        / 16.0
        * jnp.sqrt(385.0 / jnp.pi)
        * (x**2 * z * (x**2 - 3 * y**2) - y**2 * z * (3 * x**2 - y**2)),  # (l, m) == (5, 4)
        lnorm(5) * 3.0 / 16.0 * jnp.sqrt(77.0 / (2 * jnp.pi)) * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4),  # (l, m) == (5, 5)
        lnorm(6)
        * 1.0
        / 64.0
        * jnp.sqrt(6006.0 / jnp.pi)
        * (6 * x**5 * y - 20 * x**3 * y**3 + 6 * x * y**5),  # (l, m) == (6, -6)
        lnorm(6) * 3.0 / 32.0 * jnp.sqrt(2002.0 / jnp.pi) * z * (5 * x**4 * y - 10 * x**2 * y**3 + y**5),  # (l, m) == (6, -5)
        lnorm(6)
        * 3.0
        / 32.0
        * jnp.sqrt(91.0 / jnp.pi)
        * 4
        * x
        * y
        * (11 * z**2 - (x**2 + y**2 + z**2))
        * (x**2 - y**2),  # (l, m) == (6, -4)
        lnorm(6)
        * 1.0
        / 32.0
        * jnp.sqrt(2730.0 / jnp.pi)
        * -1
        * (11 * z**3 - 3 * z * (x**2 + y**2 + z**2))
        * (y**3 - 3 * x**2 * y),  # (l, m) == (6, -3)
        lnorm(6)
        * 1.0
        / 64.0
        * jnp.sqrt(2730.0 / jnp.pi)
        * 2
        * x
        * y
        * (33 * z**4 - 18 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2),  # (l, m) == (6, -2)
        lnorm(6)
        * 1.0
        / 16.0
        * jnp.sqrt(273.0 / jnp.pi)
        * y
        * (33 * z**5 - 30 * z**3 * (x**2 + y**2 + z**2) + 5 * z * (x**2 + y**2 + z**2) ** 2),  # (l, m) == (6, -1)
        lnorm(6)
        * 1.0
        / 32.0
        * jnp.sqrt(13.0 / jnp.pi)
        * (
            231 * z**6
            - 315 * z**4 * (x**2 + y**2 + z**2)
            + 105 * z**2 * (x**2 + y**2 + z**2) ** 2
            - 5 * (x**2 + y**2 + z**2) ** 3
        ),  # (l, m) == (6, 0)
        lnorm(6)
        * 1.0
        / 16.0
        * jnp.sqrt(273.0 / jnp.pi)
        * x
        * (33 * z**5 - 30 * z**3 * (x**2 + y**2 + z**2) + 5 * z * (x**2 + y**2 + z**2) ** 2),  # (l, m) == (6, 1)
        lnorm(6)
        * 1.0
        / 64.0
        * jnp.sqrt(2730.0 / jnp.pi)
        * (x**2 - y**2)
        * (33 * z**4 - 18 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2),  # (l, m) == (6, 2)
        lnorm(6)
        * 1.0
        / 32.0
        * jnp.sqrt(2730.0 / jnp.pi)
        * (11 * z**3 - 3 * z * (x**2 + y**2 + z**2))
        * (x**3 - 3 * x * y**2),  # (l, m) == (6, 3)
        lnorm(6)
        * 3.0
        / 32.0
        * jnp.sqrt(91.0 / jnp.pi)
        * (11 * z**2 - (x**2 + y**2 + z**2))
        * (x**2 * (x**2 - 3 * y**2) + y**2 * (y**2 - 3 * x**2)),  # (l, m) == (6, 4)
        lnorm(6) * 3.0 / 32.0 * jnp.sqrt(2002.0 / jnp.pi) * z * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4),  # (l, m) == (6, 5)
        lnorm(6)
        * 1.0
        / 64.0
        * jnp.sqrt(6006.0 / jnp.pi)
        * (x**6 - 15 * x**4 * y**2 + 15 * x**2 * y**4 - y**6),  # (l, m) == (6, 6)
    ]

    return jnp.select(conditions, S_l_m_values, default=jnp.nan)


def compute_normalization_fator_debug(l: int, Z: float) -> float:
    """Compute the normalization factor of a primitve AO.

    Compute the normalization factor of a primitve AO.

    Args:
        l (int): Angular momentum of the primitve AO
        Z (float): The exponent of the radial part of the primitive AO

    Returns:
        float: the normalization factor fo the primitive AO

    Note:
        This normalization factor is for the (real) 'spherical' harmonics! There is another
        normalization convention with the (regular) 'solid' harmonics!!
    """
    N_n = np.sqrt(
        (2.0 ** (2 * l + 3) * scipy.special.factorial(l + 1) * (2 * Z) ** (l + 1.5))
        / (scipy.special.factorial(2 * l + 2) * np.sqrt(np.pi))
    )
    return N_n


@jit
def compute_normalization_fator_jax(l: int, Z: float) -> float:
    """Compute the normalization factor of a primitve AO.

    Compute the normalization factor of a primitve AO.

    Args:
        l (int): Angular momentum of the primitve AO
        Z (float): The exponent of the radial part of the primitive AO

    Returns:
        float: the normalization factor fo the primitive AO

    Note:
        This normalization factor is for the (real) 'spherical' harmonics! There is another
        normalization convention with the (regular) 'solid' harmonics!!
    """
    N_n_jnp = jnp.sqrt(
        (2.0 ** (2 * l + 3) * jscipy.special.factorial(l + 1) * (2 * Z) ** (l + 1.5))
        / (jscipy.special.factorial(2 * l + 2) * jnp.sqrt(jnp.pi))
    )
    return N_n_jnp


@jit
def compute_primitive_AOs_jax(
    coefficient: float,
    exponent: float,
    l: int,
    m: int,
    R_cart: npt.NDArray[np.float64],
    r_cart: npt.NDArray[np.float64],
) -> float:
    """Compute the value of a primitve AO at the given r_cart.

    Compute the value of a primitve AO at the given r_cart.

    Args:
        coefficient (float) the coefficient of the given AO
        exponent (float): the exponent of the given AO
        l (int): Angular momentum of the AO, i.e., l. dim: 1
        m (int): Magnetic quantum number of the given AO, i.e m = -l .... +l. dim: 1
        R_cart (npt.NDArray[np.float64]): Center of the nucleus associated to the AO. dim: 3
        r_cart (npt.NDArray[np.float64]): electron position. dim: 3

    Return:
        float: the value of a primitve AO at the given r_cart.

    """
    N_n_dup = compute_normalization_fator_jax(l, exponent)
    R_n_dup = compute_R_n_jax(coefficient, exponent, R_cart, r_cart)
    S_l_m_dup = compute_S_l_m_jax(l, m, R_cart, r_cart)

    return N_n_dup * R_n_dup * jnp.sqrt((2 * l + 1) / (4 * np.pi)) * S_l_m_dup


@jit
def compute_primitive_AOs_grad_jax(
    coefficient: float,
    exponent: float,
    l: int,
    m: int,
    R_cart: npt.NDArray[np.float64],
    r_cart: npt.NDArray[np.float64],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the gradients of a primitve AO at the given r_cart.

    Compute the gradients of a primitve AO at the given r_cart.

    Args:
        coefficient (float) the coefficient of the given AO
        exponent (float): the exponent of the given AO
        l (int): Angular momentum of the AO, i.e., l. dim: 1
        m (int): Magnetic quantum number of the given AO, i.e m = -l .... +l. dim: 1
        R_cart (npt.NDArray[np.float64]): Center of the nucleus associated to the AO. dim: 3
        r_cart (npt.NDArray[np.float64]): electron position. dim: 3

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: the gradients of a primitve AO at the given r_cart.
    """
    # """grad. Correct but not the fastest...
    grad_x, grad_y, grad_z = grad(compute_primitive_AOs_jax, argnums=5)(coefficient, exponent, l, m, R_cart, r_cart)
    # """

    """
    # What if grad is replaced with the analytical one? (test using FDM) / To be refactored
    diff_h = 1.0e-5
    diff_p_x = compute_primitive_AOs_jax(
        coefficient, exponent, l, m, R_cart, r_cart + jnp.array([+diff_h, 0.0, 0.0])
    )
    diff_m_x = compute_primitive_AOs_jax(
        coefficient, exponent, l, m, R_cart, r_cart + jnp.array([-diff_h, 0.0, 0.0])
    )
    diff_p_y = compute_primitive_AOs_jax(
        coefficient, exponent, l, m, R_cart, r_cart + jnp.array([0.0, +diff_h, 0.0])
    )
    diff_m_y = compute_primitive_AOs_jax(
        coefficient, exponent, l, m, R_cart, r_cart + jnp.array([0.0, -diff_h, 0.0])
    )
    diff_p_z = compute_primitive_AOs_jax(
        coefficient, exponent, l, m, R_cart, r_cart + jnp.array([0.0, 0.0, +diff_h])
    )
    diff_m_z = compute_primitive_AOs_jax(
        coefficient, exponent, l, m, R_cart, r_cart + jnp.array([0.0, 0.0, -diff_h])
    )
    grad_x = (diff_p_x - diff_m_x) / (2 * diff_h)
    grad_y = (diff_p_y - diff_m_y) / (2 * diff_h)
    grad_z = (diff_p_z - diff_m_z) / (2 * diff_h)
    """

    return grad_x, grad_y, grad_z


@jit
def compute_primitive_AOs_laplacians_jax(
    coefficient: float,
    exponent: float,
    l: int,
    m: int,
    R_cart: npt.NDArray[np.float64],
    r_cart: npt.NDArray[np.float64],
) -> float:
    """Compute the laplacian of a primitve AO at the given r_cart.

    Compute the laplacian of a primitve AO at the given r_cart.

    Args:
        coefficient (float) the coefficient of the given AO
        exponent (float): the exponent of the given AO
        l (int): Angular momentum of the AO, i.e., l. dim: 1
        m (int): Magnetic quantum number of the given AO, i.e m = -l .... +l. dim: 1
        R_cart (npt.NDArray[np.float64]): Center of the nucleus associated to the AO. dim: 3
        r_cart (npt.NDArray[np.float64]): electron position. dim: 3

    Returns:
        float: the laplacian of the given primitve AO at the given r_cart.
    """
    # """jacrev(grad). Correct but not the fastest...
    laplacians = jnp.sum(
        jnp.diag(jacrev(grad(compute_primitive_AOs_jax, argnums=5), argnums=5)(coefficient, exponent, l, m, R_cart, r_cart))
    )
    # """

    """
    # What if jacrev(grad) is replaced with the analytical one? (test using FDM) / To be refactored
    diff_h = 1.0e-5
    p = compute_primitive_AOs_jax(coefficient, exponent, l, m, R_cart, r_cart)
    diff_p_x = compute_primitive_AOs_jax(
        coefficient, exponent, l, m, R_cart, r_cart + jnp.array([+diff_h, 0.0, 0.0])
    )
    diff_m_x = compute_primitive_AOs_jax(
        coefficient, exponent, l, m, R_cart, r_cart + jnp.array([-diff_h, 0.0, 0.0])
    )
    diff_p_y = compute_primitive_AOs_jax(
        coefficient, exponent, l, m, R_cart, r_cart + jnp.array([0.0, +diff_h, 0.0])
    )
    diff_m_y = compute_primitive_AOs_jax(
        coefficient, exponent, l, m, R_cart, r_cart + jnp.array([0.0, -diff_h, 0.0])
    )
    diff_p_z = compute_primitive_AOs_jax(
        coefficient, exponent, l, m, R_cart, r_cart + jnp.array([0.0, 0.0, +diff_h])
    )
    diff_m_z = compute_primitive_AOs_jax(
        coefficient, exponent, l, m, R_cart, r_cart + jnp.array([0.0, 0.0, -diff_h])
    )
    grad2_x = (diff_p_x + diff_m_x - 2 * p) / (diff_h) ** 2
    grad2_y = (diff_p_y + diff_m_y - 2 * p) / (diff_h) ** 2
    grad2_z = (diff_p_z + diff_m_z - 2 * p) / (diff_h) ** 2

    laplacians = grad2_x + grad2_y + grad2_z
    """

    return laplacians


if __name__ == "__main__":
    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    num_r_cart_samples = 10
    num_R_cart_samples = 3
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

    num_ao = 3
    num_ao_prim = 3
    orbital_indices = [0, 1, 2]
    exponents = [3.0, 1.0, 0.5]
    coefficients = [1.0, 1.0, 1.0]
    angular_momentums = [0, 0, 0]
    magnetic_quantum_numbers = [0, 0, 0]

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * num_R_cart_samples,
        element_symbols=["X"] * num_R_cart_samples,
        atomic_labels=["X"] * num_R_cart_samples,
    )

    aos_data = AOs_data(
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

    ao_matrix_grad_x_auto, ao_matrix_grad_y_auto, ao_matrix_grad_z_auto = compute_AOs_grad_api(
        aos_data=aos_data, r_carts=r_carts
    )

    (
        ao_matrix_grad_x_numerical,
        ao_matrix_grad_y_numerical,
        ao_matrix_grad_z_numerical,
    ) = compute_AOs_grad_api(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(ao_matrix_grad_x_auto, ao_matrix_grad_x_numerical, decimal=7)
    np.testing.assert_array_almost_equal(ao_matrix_grad_y_auto, ao_matrix_grad_y_numerical, decimal=7)

    np.testing.assert_array_almost_equal(ao_matrix_grad_z_auto, ao_matrix_grad_z_numerical, decimal=7)

    ao_matrix_laplacian_numerical = compute_AOs_laplacian_api(aos_data=aos_data, r_carts=r_carts)

    print(ao_matrix_laplacian_numerical)

    ao_matrix_laplacian_auto = compute_AOs_laplacian_api(aos_data=aos_data, r_carts=r_carts)

    np.testing.assert_array_almost_equal(ao_matrix_laplacian_auto, ao_matrix_laplacian_numerical, decimal=5)
    np.testing.assert_array_almost_equal(ao_matrix_laplacian_auto, ao_matrix_laplacian_numerical, decimal=5)
