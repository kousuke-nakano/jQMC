"""Atomic Orbitals module.

Module containing classes and methods related to Atomic Orbitals

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
from jax import grad, hessian, jacrev, jit, vmap
from jax import typing as jnpt
from numpy import linalg as LA

from .structure import Structure_data

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


@struct.dataclass
class AOs_cart_data:
    """Atomic Orbitals dataclass.

    The class contains data for computing atomic orbitals simltaneously. The radial part is the polynominal.

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
        polynominal_order_x (list[int]):
            polynominal order x of the angular part. dim: num_ao
        polynominal_order_y (list[int]):
            polynominal order y of the angular part. dim: num_ao
        polynominal_order_z (list[int]):
            polynominal order z of the angular part. dim: num_ao
    """

    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    nucleus_index: list[int] = struct.field(pytree_node=False, default_factory=list)
    num_ao: int = struct.field(pytree_node=False, default=0)
    num_ao_prim: int = struct.field(pytree_node=False, default=0)
    orbital_indices: list[int] = struct.field(pytree_node=False, default_factory=list)
    exponents: list[float] = struct.field(pytree_node=False, default_factory=list)
    coefficients: list[float] = struct.field(pytree_node=False, default_factory=list)
    angular_momentums: list[int] = struct.field(pytree_node=False, default_factory=list)
    polynominal_order_x: list[int] = struct.field(pytree_node=False, default_factory=list)
    polynominal_order_y: list[int] = struct.field(pytree_node=False, default_factory=list)
    polynominal_order_z: list[int] = struct.field(pytree_node=False, default_factory=list)

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

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
        if len(self.polynominal_order_x) != self.num_ao:
            logger.error("dim. of self.polynominal_order_x is wrong")
            raise ValueError
        if len(self.polynominal_order_y) != self.num_ao:
            logger.error("dim. of self.polynominal_order_y is wrong")
            raise ValueError
        if len(self.polynominal_order_z) != self.num_ao:
            logger.error("dim. of self.polynominal_order_z is wrong")
            raise ValueError
        self.structure_data.sanity_check()

    def get_info(self) -> list[str]:
        """Return a list of strings containing information about the class attributes."""
        info_lines = []
        info_lines.extend(["**" + self.__class__.__name__])
        info_lines.extend([f"  Number of AOs = {self.num_ao}"])
        info_lines.extend([f"  Number of primitive AOs = {self.num_ao_prim}"])
        info_lines.extend(["  Radial part is the polynominal (cartesian) function."])
        return info_lines

    def logger_info(self) -> None:
        """Output the information from get_info using logger.info."""
        for line in self.get_info():
            logger.info(line)

    @property
    def nucleus_index_np(self) -> npt.NDArray[np.int32]:
        """nucleus_index."""
        return np.array(self.nucleus_index, dtype=np.int32)

    @property
    def nucleus_index_jnp(self) -> jax.Array:
        """nucleus_index."""
        return jnp.array(self.nucleus_index, dtype=jnp.int32)

    @property
    def nucleus_index_prim_np(self) -> npt.NDArray[np.int32]:
        """nucleus_index."""
        return np.array(self.nucleus_index)[self.orbital_indices_np]

    @property
    def nucleus_index_prim_jnp(self) -> jax.Array:
        """nucleus_index."""
        return jnp.array(self.nucleus_index_prim_np, dtype=jnp.int32)

    @property
    def orbital_indices_np(self) -> npt.NDArray[np.int32]:
        """orbital_index."""
        return np.array(self.orbital_indices, dtype=np.int32)

    @property
    def orbital_indices_jnp(self) -> jax.Array:
        """orbital_index."""
        return jnp.array(self.orbital_indices, dtype=jnp.int32)

    @property
    def atomic_center_carts_np(self) -> npt.NDArray[np.float64]:
        """Atomic positions in cartesian.

        Returns atomic positions in cartesian

        Returns:
            npt.NDArray[np.float64]: atomic positions in cartesian
        """
        return self.structure_data.positions_cart_np[self.nucleus_index_np]

    @property
    def atomic_center_carts_jnp(self) -> jax.Array:
        """Atomic positions in cartesian.

        Returns atomic positions in cartesian

        Returns:
            jax.Array: atomic positions in cartesian
        """
        # this is super slow!!! Do not use list comprehension.
        # return jnp.array([self.structure_data.positions_cart[i] for i in self.nucleus_index])
        return self.structure_data.positions_cart_jnp[self.nucleus_index_jnp]

    @property
    def atomic_center_carts_unique_jnp(self) -> jax.Array:
        """Unique atomic positions in cartesian.

        Returns unique atomic positions in cartesian

        Returns:
            jax.Array: atomic positions in cartesian
        """
        return self.structure_data.positions_cart_jnp
        """ the same as above.
        _, first_indices = np.unique(self.nucleus_index_np, return_index=True)
        sorted_order = jnp.argsort(first_indices)
        return self.structure_data.positions_cart_jnp[sorted_order]
        """

    @property
    def atomic_center_carts_prim_np(self) -> npt.NDArray[np.float64]:
        """Atomic positions in cartesian for primitve orbitals.

        Returns atomic positions in cartesian for primitive orbitals

        Returns:
            npt.NDArray[np.float]: atomic positions in cartesian for primitive orbitals
        """
        return self.atomic_center_carts_np[self.orbital_indices]

    @property
    def atomic_center_carts_prim_jnp(self) -> jax.Array:
        """Atomic positions in cartesian for primitve orbitals.

        Returns atomic positions in cartesian for primitive orbitals

        Returns:
            jax.Array: atomic positions in cartesian for primitive orbitals
        """
        # this is super slow!!! Do not use list comprehension.
        # return jnp.array([self.atomic_center_carts_jnp[i] for i in self.orbital_indices])
        return self.atomic_center_carts_jnp[self.orbital_indices_jnp]

    @property
    def angular_momentums_prim_np(self) -> npt.NDArray[np.int32]:
        """Angular momentums for primitive orbitals.

        Returns angular momentums for primitive orbitals

        Returns:
            npt.NDArray[np.float64]: angular momentums for primitive orbitals
        """
        return np.array(self.angular_momentums, dtype=np.int32)[self.orbital_indices_np]

    @property
    def angular_momentums_prim_jnp(self) -> jax.Array:
        """Angular momentums for primitive orbitals.

        Returns angular momentums for primitive orbitals

        Returns:
            jax.Array: angular momentums for primitive orbitals
        """
        return jnp.array(self.angular_momentums_prim_np, dtype=jnp.int32)

    @property
    def polynominal_order_x_prim_np(self) -> npt.NDArray[np.int32]:
        """Polynominal order of x for primitive orbitals.

        Returns Polynominal order of x for primitive orbitals

        Returns:
            jax.Array: Polynominal order of x for primitive orbitals
        """
        return np.array(self.polynominal_order_x, dtype=np.int32)[self.orbital_indices_np]

    @property
    def polynominal_order_x_prim_jnp(self) -> jax.Array:
        """Polynominal order of x for primitive orbitals.

        Returns Polynominal order of x for primitive orbitals

        Returns:
            jax.Array: Polynominal order of x for primitive orbitals
        """
        return jnp.array(self.polynominal_order_x_prim_np, dtype=np.int32)

    @property
    def polynominal_order_y_prim_np(self) -> npt.NDArray[np.int32]:
        """Polynominal order of y for primitive orbitals.

        Returns Polynominal order of y for primitive orbitals

        Returns:
            jax.Array: Polynominal order of y for primitive orbitals
        """
        return np.array(self.polynominal_order_y, dtype=np.int32)[self.orbital_indices_np]

    @property
    def polynominal_order_y_prim_jnp(self) -> jax.Array:
        """Polynominal order of y for primitive orbitals.

        Returns Polynominal order of y for primitive orbitals

        Returns:
            jax.Array: Polynominal order of y for primitive orbitals
        """
        return jnp.array(self.polynominal_order_y_prim_np, dtype=np.int32)

    @property
    def polynominal_order_z_prim_np(self) -> npt.NDArray[np.int32]:
        """Polynominal order of z for primitive orbitals.

        Returns Polynominal order of z for primitive orbitals

        Returns:
            jax.Array: Polynominal order of z for primitive orbitals
        """
        return np.array(self.polynominal_order_z, dtype=np.int32)[self.orbital_indices_np]

    @property
    def polynominal_order_z_prim_jnp(self) -> jax.Array:
        """Polynominal order of z for primitive orbitals.

        Returns Polynominal order of z for primitive orbitals

        Returns:
            jax.Array: Polynominal order of z for primitive orbitals
        """
        return jnp.array(self.polynominal_order_z_prim_np, dtype=np.int32)

    @property
    def exponents_jnp(self) -> jax.Array:
        """Return exponents."""
        return jnp.array(self.exponents, dtype=jnp.float64)

    @property
    def coefficients_jnp(self) -> jax.Array:
        """Return coefficients."""
        return jnp.array(self.coefficients, dtype=jnp.float64)

    @property
    def num_orb(self) -> int:
        """Return the number of orbitals."""
        return self.num_ao


@struct.dataclass
class AOs_sphe_data:
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

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

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
        self.structure_data.sanity_check()

    def get_info(self) -> list[str]:
        """Return a list of strings containing information about the class attributes."""
        info_lines = []
        info_lines.extend(["**" + self.__class__.__name__])
        info_lines.extend([f"  Number of AOs = {self.num_ao}"])
        info_lines.extend([f"  Number of primitive AOs = {self.num_ao_prim}"])
        info_lines.extend(["  Radial part is the real spherical (solid) Harmonics."])
        return info_lines

    def logger_info(self) -> None:
        """Output the information from get_info using logger.info."""
        for line in self.get_info():
            logger.info(line)

    @property
    def nucleus_index_np(self) -> npt.NDArray[np.int32]:
        """nucleus_index."""
        return np.array(self.nucleus_index, dtype=np.int32)

    @property
    def nucleus_index_jnp(self) -> jax.Array:
        """nucleus_index."""
        return jnp.array(self.nucleus_index, dtype=jnp.int32)

    @property
    def nucleus_index_prim_np(self) -> npt.NDArray[np.int32]:
        """nucleus_index."""
        return np.array(self.nucleus_index)[self.orbital_indices_np]

    @property
    def nucleus_index_prim_jnp(self) -> jax.Array:
        """nucleus_index."""
        return jnp.array(self.nucleus_index_prim_np, dtype=jnp.int32)

    @property
    def orbital_indices_np(self) -> npt.NDArray[np.int32]:
        """orbital_index."""
        return np.array(self.orbital_indices, dtype=np.int32)

    @property
    def orbital_indices_jnp(self) -> jax.Array:
        """orbital_index."""
        return jnp.array(self.orbital_indices, dtype=jnp.int32)

    @property
    def atomic_center_carts_np(self) -> npt.NDArray[np.float64]:
        """Atomic positions in cartesian.

        Returns atomic positions in cartesian

        Returns:
            npt.NDArray[np.float64]: atomic positions in cartesian
        """
        return self.structure_data.positions_cart_np[self.nucleus_index_np]

    @property
    def atomic_center_carts_jnp(self) -> jax.Array:
        """Atomic positions in cartesian.

        Returns atomic positions in cartesian

        Returns:
            jax.Array: atomic positions in cartesian
        """
        # this is super slow!!! Do not use list comprehension.
        # return jnp.array([self.structure_data.positions_cart[i] for i in self.nucleus_index])
        return self.structure_data.positions_cart_jnp[self.nucleus_index_jnp]

    @property
    def atomic_center_carts_unique_jnp(self) -> jax.Array:
        """Unique atomic positions in cartesian.

        Returns unique atomic positions in cartesian

        Returns:
            jax.Array: atomic positions in cartesian
        """
        return self.structure_data.positions_cart_jnp
        """ the same as above.
        _, first_indices = np.unique(self.nucleus_index_np, return_index=True)
        sorted_order = jnp.argsort(first_indices)
        return self.structure_data.positions_cart_jnp[sorted_order]
        """

    @property
    def atomic_center_carts_prim_np(self) -> npt.NDArray[np.float64]:
        """Atomic positions in cartesian for primitve orbitals.

        Returns atomic positions in cartesian for primitive orbitals

        Returns:
            npt.NDArray[np.float]: atomic positions in cartesian for primitive orbitals
        """
        return self.atomic_center_carts_np[self.orbital_indices]

    @property
    def atomic_center_carts_prim_jnp(self) -> jax.Array:
        """Atomic positions in cartesian for primitve orbitals.

        Returns atomic positions in cartesian for primitive orbitals

        Returns:
            jax.Array: atomic positions in cartesian for primitive orbitals
        """
        # this is super slow!!! Do not use list comprehension.
        # return jnp.array([self.atomic_center_carts_jnp[i] for i in self.orbital_indices])
        return self.atomic_center_carts_jnp[self.orbital_indices_jnp]

    @property
    def angular_momentums_prim_np(self) -> npt.NDArray[np.int32]:
        """Angular momentums for primitive orbitals.

        Returns angular momentums for primitive orbitals

        Returns:
            npt.NDArray[np.float64]: angular momentums for primitive orbitals
        """
        return np.array(self.angular_momentums, dtype=np.int32)[self.orbital_indices_np]

    @property
    def angular_momentums_prim_jnp(self) -> jax.Array:
        """Angular momentums for primitive orbitals.

        Returns angular momentums for primitive orbitals

        Returns:
            jax.Array: angular momentums for primitive orbitals
        """
        return jnp.array(self.angular_momentums_prim_np, dtype=jnp.int32)

    @property
    def magnetic_quantum_numbers_prim_np(self) -> npt.NDArray[np.int32]:
        """Magnetic quantum numbers for primitive orbitals.

        Returns magnetic quantum numbers for primitive orbitals

        Returns:
            jax.Array: magnetic quantum numbers for primitive orbitals
        """
        return np.array(self.magnetic_quantum_numbers, dtype=np.int32)[self.orbital_indices_np]

    @property
    def magnetic_quantum_numbers_prim_jnp(self) -> jax.Array:
        """Magnetic quantum numbers for primitive orbitals.

        Returns magnetic quantum numbers for primitive orbitals

        Returns:
            npt.NDArray[np.int64]: magnetic quantum numbers for primitive orbitals
        """
        return jnp.array(self.magnetic_quantum_numbers_prim_np, dtype=jnp.int32)

    @property
    def exponents_jnp(self) -> jax.Array:
        """Return exponents."""
        return jnp.array(self.exponents, dtype=jnp.float64)

    @property
    def coefficients_jnp(self) -> jax.Array:
        """Return coefficients."""
        return jnp.array(self.coefficients, dtype=jnp.float64)

    @property
    def num_orb(self) -> int:
        """Return the number of orbitals."""
        return self.num_ao


@struct.dataclass
class AOs_sphe_data_deriv_R(AOs_sphe_data):
    """See AOs_data."""

    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    nucleus_index: list[int] = struct.field(pytree_node=False, default_factory=list)
    num_ao: int = struct.field(pytree_node=False, default=0)
    num_ao_prim: int = struct.field(pytree_node=False, default=0)
    orbital_indices: list[int] = struct.field(pytree_node=False, default_factory=list)
    exponents: list[float] = struct.field(pytree_node=False, default_factory=list)
    coefficients: list[float] = struct.field(pytree_node=False, default_factory=list)
    angular_momentums: list[int] = struct.field(pytree_node=False, default_factory=list)
    magnetic_quantum_numbers: list[int] = struct.field(pytree_node=False, default_factory=list)

    @classmethod
    def from_base(cls, aos_data: AOs_sphe_data):
        """Switch pytree_node."""
        structure_data = aos_data.structure_data
        nucleus_index = aos_data.nucleus_index
        num_ao = aos_data.num_ao
        num_ao_prim = aos_data.num_ao_prim
        orbital_indices = aos_data.orbital_indices
        exponents = aos_data.exponents
        coefficients = aos_data.coefficients
        angular_momentums = aos_data.angular_momentums
        magnetic_quantum_numbers = aos_data.magnetic_quantum_numbers

        return cls(
            structure_data,
            nucleus_index,
            num_ao,
            num_ao_prim,
            orbital_indices,
            exponents,
            coefficients,
            angular_momentums,
            magnetic_quantum_numbers,
        )


@struct.dataclass
class AOs_sphe_data_no_deriv(AOs_sphe_data):
    """See AOs_data."""

    structure_data: Structure_data = struct.field(pytree_node=False, default_factory=lambda: Structure_data())
    nucleus_index: list[int] = struct.field(pytree_node=False, default_factory=list)
    num_ao: int = struct.field(pytree_node=False, default=0)
    num_ao_prim: int = struct.field(pytree_node=False, default=0)
    orbital_indices: list[int] = struct.field(pytree_node=False, default_factory=list)
    exponents: list[float] = struct.field(pytree_node=False, default_factory=list)
    coefficients: list[float] = struct.field(pytree_node=False, default_factory=list)
    angular_momentums: list[int] = struct.field(pytree_node=False, default_factory=list)
    magnetic_quantum_numbers: list[int] = struct.field(pytree_node=False, default_factory=list)

    @classmethod
    def from_base(cls, aos_data: AOs_sphe_data):
        """Switch pytree_node."""
        structure_data = aos_data.structure_data
        nucleus_index = aos_data.nucleus_index
        num_ao = aos_data.num_ao
        num_ao_prim = aos_data.num_ao_prim
        orbital_indices = aos_data.orbital_indices
        exponents = aos_data.exponents
        coefficients = aos_data.coefficients
        angular_momentums = aos_data.angular_momentums
        magnetic_quantum_numbers = aos_data.magnetic_quantum_numbers

        return cls(
            structure_data,
            nucleus_index,
            num_ao,
            num_ao_prim,
            orbital_indices,
            exponents,
            coefficients,
            angular_momentums,
            magnetic_quantum_numbers,
        )


def compute_AOs_api(aos_data: AOs_sphe_data | AOs_cart_data, r_carts: jnpt.ArrayLike, debug=False) -> jax.Array:
    """Compute AO values at the given r_carts.

    The method is for computing the value of the given atomic orbital at r_carts

    Args:
        ao_datas (AOs_data): an instance of AOs_data
        r_carts (jnpt.ArrayLike): Cartesian coordinates of electrons (dim: N_e, 3)
        debug (bool): if True, AOs are computed via _debug function for debuging purpose

    Returns:
        jax.Array: Arrays containing values of the AOs at r_carts. (dim: num_ao, N_e)
    """
    if isinstance(aos_data, AOs_sphe_data):
        if debug:
            AOs = _compute_AOs_shpe_debug(aos_data, r_carts)
        else:
            AOs = _compute_AOs_sphe_jax(aos_data, r_carts)

    elif isinstance(aos_data, AOs_cart_data):
        if debug:
            AOs = _compute_AOs_cart_debug(aos_data, r_carts)
        else:
            AOs = _compute_AOs_cart_jax(aos_data, r_carts)
    else:
        raise NotImplementedError
    return AOs


def _compute_AOs_shpe_debug(aos_data: AOs_sphe_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute AO values at the given r_carts.

    The method is for computing the value of the given atomic orbital at r_carts
    for debugging purpose. See compute_AOs_api.
    """
    aos_values = []

    for ao_index in range(aos_data.num_ao):
        atomic_center_cart = aos_data.atomic_center_carts_np[ao_index]
        shell_indices = [i for i, v in enumerate(aos_data.orbital_indices) if v == ao_index]
        exponents = [aos_data.exponents[i] for i in shell_indices]
        coefficients = [aos_data.coefficients[i] for i in shell_indices]
        angular_momentum = aos_data.angular_momentums[ao_index]
        magnetic_quantum_number = aos_data.magnetic_quantum_numbers[ao_index]
        ao_value = []
        for r_cart in r_carts:
            # radial part
            R_n = np.array(
                [
                    coefficient * np.exp(-1.0 * exponent * LA.norm(np.array(r_cart) - np.array(atomic_center_cart)) ** 2)
                    for coefficient, exponent in zip(coefficients, exponents)
                ]
            )
            # normalization part
            N_n_l = np.array(
                [
                    np.sqrt(
                        (
                            2.0 ** (2 * angular_momentum + 3)
                            * scipy.special.factorial(angular_momentum + 1)
                            * (2 * Z) ** (angular_momentum + 1.5)
                        )
                        / (scipy.special.factorial(2 * angular_momentum + 2) * np.sqrt(np.pi))
                    )
                    for Z in exponents
                ]
            )
            # angular part
            S_l_m = _compute_S_l_m_debug(
                atomic_center_cart=atomic_center_cart,
                angular_momentum=angular_momentum,
                magnetic_quantum_number=magnetic_quantum_number,
                r_cart=r_cart,
            )

            ao_value.append(np.sum(N_n_l * R_n) * np.sqrt((2 * angular_momentum + 1) / (4 * np.pi)) * S_l_m)

        aos_values.append(ao_value)

    return aos_values


def _compute_AOs_cart_debug(aos_data: AOs_cart_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute AO values at the given r_carts.

    The method is for computing the value of the given atomic orbital at r_carts
    for debugging purpose. See compute_AOs_api.
    """
    aos_values = []

    for ao_index in range(aos_data.num_ao):
        R_cart = aos_data.atomic_center_carts_np[ao_index]
        l = aos_data.angular_momentums[ao_index]
        shell_indices = [i for i, v in enumerate(aos_data.orbital_indices) if v == ao_index]
        exponents = [aos_data.exponents[i] for i in shell_indices]
        coefficients = [aos_data.coefficients[i] for i in shell_indices]
        nx = aos_data.polynominal_order_x[ao_index]
        ny = aos_data.polynominal_order_y[ao_index]
        nz = aos_data.polynominal_order_z[ao_index]

        ao_value = []
        for r_cart in r_carts:
            # radial part
            R_n = np.array(
                [
                    coefficient * np.exp(-1.0 * exponent * LA.norm(np.array(r_cart) - np.array(R_cart)) ** 2)
                    for coefficient, exponent in zip(coefficients, exponents)
                ]
            )
            # normalization part
            N_n_l = np.array(
                [
                    np.sqrt(
                        (2.0 * Z / np.pi) ** (3.0 / 2.0)
                        * (8.0 * Z) ** l
                        * scipy.special.factorial(nx)
                        * scipy.special.factorial(ny)
                        * scipy.special.factorial(nz)
                        / (scipy.special.factorial(2 * nx) * scipy.special.factorial(2 * ny) * scipy.special.factorial(2 * nz))
                    )
                    for Z in exponents
                ]
            )
            # angular part
            x, y, z = np.array(r_cart) - np.array(R_cart)
            P_l_nx_ny_nz = x**nx * y**ny * z**nz

            ao_value.append(np.sum(N_n_l * R_n) * P_l_nx_ny_nz)

        aos_values.append(ao_value)

    return aos_values


@jit
def _compute_AOs_cart_jax(aos_data: AOs_cart_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Compute AO values at the given r_carts.

    See compute_AOs_api

    """
    # Indices with respect to the contracted AOs
    R_carts_jnp = aos_data.atomic_center_carts_prim_jnp
    c_jnp = aos_data.coefficients_jnp
    Z_jnp = aos_data.exponents_jnp
    l_jnp = aos_data.angular_momentums_prim_jnp
    nx_jnp = aos_data.polynominal_order_x_prim_jnp
    ny_jnp = aos_data.polynominal_order_y_prim_jnp
    nz_jnp = aos_data.polynominal_order_z_prim_jnp

    N_n_dup_fuctorial_part = (
        jscipy.special.factorial(nx_jnp) * jscipy.special.factorial(ny_jnp) * jscipy.special.factorial(nz_jnp)
    ) / (jscipy.special.factorial(2 * nx_jnp) * jscipy.special.factorial(2 * ny_jnp) * jscipy.special.factorial(2 * nz_jnp))
    N_n_dup_Z_part = (2.0 * Z_jnp / jnp.pi) ** (3.0 / 2.0) * (8.0 * Z_jnp) ** l_jnp
    N_n_dup = jnp.sqrt(N_n_dup_Z_part * N_n_dup_fuctorial_part)
    r_R_diffs = r_carts[None, :, :] - R_carts_jnp[:, None, :]
    r_squared = jnp.sum(r_R_diffs**2, axis=-1)
    R_n_dup = c_jnp[:, None] * jnp.exp(-Z_jnp[:, None] * r_squared)

    x, y, z = r_R_diffs[..., 0], r_R_diffs[..., 1], r_R_diffs[..., 2]
    eps = 1.0e-16  # This is quite important to avoid some numerical instability in JAX!!
    P_l_nx_ny_nz_dup = (x + eps) ** (nx_jnp[:, None]) * (y + eps) ** (ny_jnp[:, None]) * (z + eps) ** (nz_jnp[:, None])

    """
    logger.info(f"Z_jnp={Z_jnp}.")
    logger.info(f"l_jnp={l_jnp}.")
    logger.info(f"nx_jnp={nx_jnp}.")
    logger.info(f"ny_jnp={ny_jnp}.")
    logger.info(f"nz_jnp={nz_jnp}.")
    logger.info(f"N_n_dup={N_n_dup.shape}, R_n_dup={R_n_dup.shape}")
    logger.info(f"N_n_dup={N_n_dup.shape}, R_n_dup={R_n_dup.shape}")
    logger.info(f"l_jnp={l_jnp.shape}, Z_jnp={Z_jnp.shape}.")
    logger.info(f"nx_jnp={nx_jnp.shape}, ny_jnp={ny_jnp.shape}, nz_jnp={nz_jnp.shape}")
    """

    AOs_dup = N_n_dup[:, None] * R_n_dup * P_l_nx_ny_nz_dup

    orbital_indices = aos_data.orbital_indices_jnp
    num_segments = aos_data.num_ao
    AOs = jax.ops.segment_sum(AOs_dup, orbital_indices, num_segments=num_segments)
    return AOs


@jit
def _compute_AOs_sphe_jax(aos_data: AOs_sphe_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Compute AO values at the given r_carts.

    See compute_AOs_api

    """
    # Indices with respect to the contracted AOs
    # compute R_n inc. the whole normalization factor
    nucleus_index_prim_jnp = aos_data.nucleus_index_prim_jnp
    R_carts_jnp = aos_data.atomic_center_carts_prim_jnp
    R_carts_unique_jnp = aos_data.atomic_center_carts_unique_jnp
    c_jnp = aos_data.coefficients_jnp
    Z_jnp = aos_data.exponents_jnp
    l_jnp = aos_data.angular_momentums_prim_jnp
    m_jnp = aos_data.magnetic_quantum_numbers_prim_jnp

    N_n_dup = jnp.sqrt(
        (2.0 ** (2 * l_jnp + 3) * jscipy.special.factorial(l_jnp + 1) * (2 * Z_jnp) ** (l_jnp + 1.5))
        / (jscipy.special.factorial(2 * l_jnp + 2) * jnp.sqrt(jnp.pi))
    )
    N_l_m_dup = jnp.sqrt((2 * l_jnp + 1) / (4 * jnp.pi))
    r_R_diffs = r_carts[None, :, :] - R_carts_jnp[:, None, :]
    r_squared = jnp.sum(r_R_diffs**2, axis=-1)
    R_n_dup = c_jnp[:, None] * jnp.exp(-Z_jnp[:, None] * r_squared)
    r_R_diffs_uq = r_carts[None, :, :] - R_carts_unique_jnp[:, None, :]

    max_ml, S_l_m_dup_all_l_m = _compute_S_l_m_jax(r_R_diffs_uq)
    S_l_m_dup_all_l_m_reshaped = S_l_m_dup_all_l_m.reshape(
        (S_l_m_dup_all_l_m.shape[0] * S_l_m_dup_all_l_m.shape[1], S_l_m_dup_all_l_m.shape[2]), order="F"
    )
    global_l_m_index = l_jnp**2 + (m_jnp + l_jnp)
    global_R_l_m_index = nucleus_index_prim_jnp * max_ml + global_l_m_index
    S_l_m_dup = S_l_m_dup_all_l_m_reshaped[global_R_l_m_index]

    AOs_dup = N_n_dup[:, None] * R_n_dup * N_l_m_dup[:, None] * S_l_m_dup

    orbital_indices = aos_data.orbital_indices_jnp
    num_segments = aos_data.num_ao
    AOs = jax.ops.segment_sum(AOs_dup, orbital_indices, num_segments=num_segments)
    return AOs


@jit
def _compute_S_l_m_jax(
    r_R_diffs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Solid harmonics part of a primitve AO.

    Compute the solid harmonics, i.e., r^l * spherical hamonics part (c.f., regular solid harmonics) of a given AO

    Args:
        r_R_diffs (npt.NDArray[np.float64]): Cartesian coordinate of N electrons - Cartesian corrdinates of M nuclei. dim: (N,M,3)

    Returns:
        npt.NDArray[np.float64]: dim:(49,N,M) arrays of the spherical harmonics part * r^l (i.e., regular solid harmonics) for all (l,m) pairs.
    """
    x, y, z = r_R_diffs[..., 0], r_R_diffs[..., 1], r_R_diffs[..., 2]
    r_norm = jnp.sqrt(x**2 + y**2 + z**2)

    def lnorm(l):
        return jnp.sqrt((4 * jnp.pi) / (2 * l + 1))

    """see https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics (l=0-4)"""
    """Useful tool to generate spherical harmonics generator [https://github.com/elerac/sh_table]"""
    max_ml = 49
    # s orbital
    s_0 = lnorm(l=0) * 1.0 / 2.0 * jnp.sqrt(1.0 / jnp.pi) * r_norm**0.0  # (l, m) == (0, 0)
    # p orbitals
    p_m1 = lnorm(l=1) * jnp.sqrt(3.0 / (4 * jnp.pi)) * y  # (l, m) == (1, -1)
    p_0 = lnorm(l=1) * jnp.sqrt(3.0 / (4 * jnp.pi)) * z  # (l, m) == (1, 0)
    p_p1 = lnorm(l=1) * jnp.sqrt(3.0 / (4 * jnp.pi)) * x  # (l, m) == (1, 1)
    # d orbitals
    d_m2 = lnorm(l=2) * 1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * x * y  # (l, m) == (2, -2)
    d_m1 = lnorm(l=2) * 1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * y * z  # (l, m) == (2, -1)
    d_0 = lnorm(l=2) * 1.0 / 4.0 * jnp.sqrt(5.0 / (jnp.pi)) * (3 * z**2 - r_norm**2)  # (l, m) == (2, 0):
    d_p1 = lnorm(l=2) * 1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * x * z  # (l, m) == (2, 1)
    d_p2 = lnorm(l=2) * 1.0 / 4.0 * jnp.sqrt(15.0 / (jnp.pi)) * (x**2 - y**2)  # (l, m) == (2, 2)
    # f orbitals
    f_m3 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * y * (3 * x**2 - y**2)  # (l, m) == (3, -3)
    f_m2 = lnorm(l=3) * 1.0 / 2.0 * jnp.sqrt(105.0 / (jnp.pi)) * x * y * z  # (l, m) == (3, -2)
    f_m1 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(21.0 / (2 * jnp.pi)) * y * (5 * z**2 - r_norm**2)  # (l, m) == (3, -1)
    f_0 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(7.0 / (jnp.pi)) * (5 * z**3 - 3 * z * r_norm**2)  # (l, m) == (3, 0)
    f_p1 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(21.0 / (2 * jnp.pi)) * x * (5 * z**2 - r_norm**2)  # (l, m) == (3, 1)
    f_p2 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(105.0 / (jnp.pi)) * (x**2 - y**2) * z  # (l, m) == (3, 2)
    f_p3 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * x * (x**2 - 3 * y**2)  # (l, m) == (3, 3)
    # g orbitals
    g_m4 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(35.0 / (jnp.pi)) * x * y * (x**2 - y**2)  # (l, m) == (4, -4)
    g_m3 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * y * z * (3 * x**2 - y**2)  # (l, m) == (4, -3)
    g_m2 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(5.0 / (jnp.pi)) * x * y * (7 * z**2 - r_norm**2)  # (l, m) == (4, -2)
    g_m1 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(5.0 / (2 * jnp.pi)) * y * (7 * z**3 - 3 * z * r_norm**2)  # (l, m) == (4, -1)
    g_0 = (
        lnorm(l=4) * 3.0 / 16.0 * jnp.sqrt(1.0 / (jnp.pi)) * (35 * z**4 - 30 * z**2 * r_norm**2 + 3 * r_norm**4)
    )  # (l, m) == (4, 0)
    g_p1 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(5.0 / (2 * jnp.pi)) * x * (7 * z**3 - 3 * z * r_norm**2)  # (l, m) == (4, 1)
    g_p2 = lnorm(l=4) * 3.0 / 8.0 * jnp.sqrt(5.0 / (jnp.pi)) * (x**2 - y**2) * (7 * z**2 - r_norm**2)  # (l, m) == (4, 2)
    g_p3 = lnorm(l=4) * (3.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * x * z * (x**2 - 3 * y**2))  # (l, m) == (4, 3)
    g_p4 = (
        lnorm(l=4) * 3.0 / 16.0 * jnp.sqrt(35.0 / (jnp.pi)) * (x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2))
    )  # (l, m) == (4, 4)
    # h orbitals
    h_m5 = lnorm(5) * 3.0 / 16.0 * jnp.sqrt(77.0 / (2 * jnp.pi)) * (5 * x**4 * y - 10 * x**2 * y**3 + y**5)  # (l, m) == (5, -5)
    h_m4 = lnorm(5) * 3.0 / 16.0 * jnp.sqrt(385.0 / jnp.pi) * 4 * x * y * z * (x**2 - y**2)  # (l, m) == (5, -4)
    h_m3 = (
        lnorm(5) * 1.0 / 16.0 * jnp.sqrt(385.0 / (2 * jnp.pi)) * -1 * (y**3 - 3 * x**2 * y) * (9 * z**2 - (x**2 + y**2 + z**2))
    )  # (l, m) == (5, -3)
    h_m2 = (
        lnorm(5) * 1.0 / 8.0 * jnp.sqrt(1155 / jnp.pi) * 2 * x * y * (3 * z**3 - z * (x**2 + y**2 + z**2))
    )  # (l, m) == (5, -2)
    h_m1 = (
        lnorm(5)
        * 1.0
        / 16.0
        * jnp.sqrt(165 / jnp.pi)
        * y
        * (21 * z**4 - 14 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (5, -1)
    h_0 = (
        lnorm(5)
        * 1.0
        / 16.0
        * jnp.sqrt(11 / jnp.pi)
        * (63 * z**5 - 70 * z**3 * (x**2 + y**2 + z**2) + 15 * z * (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (5, 0)
    h_p1 = (
        lnorm(5)
        * 1.0
        / 16.0
        * jnp.sqrt(165 / jnp.pi)
        * x
        * (21 * z**4 - 14 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (5, 1)
    h_p2 = (
        lnorm(5) * 1.0 / 8.0 * jnp.sqrt(1155 / jnp.pi) * (x**2 - y**2) * (3 * z**3 - z * (x**2 + y**2 + z**2))
    )  # (l, m) == (5, 2)
    h_p3 = (
        lnorm(5) * 1.0 / 16.0 * jnp.sqrt(385.0 / (2 * jnp.pi)) * (x**3 - 3 * x * y**2) * (9 * z**2 - (x**2 + y**2 + z**2))
    )  # (l, m) == (5, 3)
    h_p4 = (
        lnorm(5) * 3.0 / 16.0 * jnp.sqrt(385.0 / jnp.pi) * (x**2 * z * (x**2 - 3 * y**2) - y**2 * z * (3 * x**2 - y**2))
    )  # (l, m) == (5, 4)
    h_p5 = lnorm(5) * 3.0 / 16.0 * jnp.sqrt(77.0 / (2 * jnp.pi)) * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4)  # (l, m) == (5, 5)
    # i orbitals
    i_m6 = (
        lnorm(6) * 1.0 / 64.0 * jnp.sqrt(6006.0 / jnp.pi) * (6 * x**5 * y - 20 * x**3 * y**3 + 6 * x * y**5)
    )  # (l, m) == (6, -6)
    i_m5 = lnorm(6) * 3.0 / 32.0 * jnp.sqrt(2002.0 / jnp.pi) * z * (5 * x**4 * y - 10 * x**2 * y**3 + y**5)  # (l, m) == (6, -5)
    i_m4 = (
        lnorm(6) * 3.0 / 32.0 * jnp.sqrt(91.0 / jnp.pi) * 4 * x * y * (11 * z**2 - (x**2 + y**2 + z**2)) * (x**2 - y**2)
    )  # (l, m) == (6, -4)
    i_m3 = (
        lnorm(6)
        * 1.0
        / 32.0
        * jnp.sqrt(2730.0 / jnp.pi)
        * -1
        * (11 * z**3 - 3 * z * (x**2 + y**2 + z**2))
        * (y**3 - 3 * x**2 * y)
    )  # (l, m) == (6, -3)
    i_m2 = (
        lnorm(6)
        * 1.0
        / 64.0
        * jnp.sqrt(2730.0 / jnp.pi)
        * 2
        * x
        * y
        * (33 * z**4 - 18 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (6, -2)
    i_m1 = (
        lnorm(6)
        * 1.0
        / 16.0
        * jnp.sqrt(273.0 / jnp.pi)
        * y
        * (33 * z**5 - 30 * z**3 * (x**2 + y**2 + z**2) + 5 * z * (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (6, -1)
    i_0 = (
        lnorm(6)
        * 1.0
        / 32.0
        * jnp.sqrt(13.0 / jnp.pi)
        * (
            231 * z**6
            - 315 * z**4 * (x**2 + y**2 + z**2)
            + 105 * z**2 * (x**2 + y**2 + z**2) ** 2
            - 5 * (x**2 + y**2 + z**2) ** 3
        )
    )  # (l, m) == (6, 0)
    i_p1 = (
        lnorm(6)
        * 1.0
        / 16.0
        * jnp.sqrt(273.0 / jnp.pi)
        * x
        * (33 * z**5 - 30 * z**3 * (x**2 + y**2 + z**2) + 5 * z * (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (6, 1)
    i_p2 = (
        lnorm(6)
        * 1.0
        / 64.0
        * jnp.sqrt(2730.0 / jnp.pi)
        * (x**2 - y**2)
        * (33 * z**4 - 18 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (6, 2)
    i_p3 = (
        lnorm(6) * 1.0 / 32.0 * jnp.sqrt(2730.0 / jnp.pi) * (11 * z**3 - 3 * z * (x**2 + y**2 + z**2)) * (x**3 - 3 * x * y**2)
    )  # (l, m) == (6, 3)
    i_p4 = (
        lnorm(6)
        * 3.0
        / 32.0
        * jnp.sqrt(91.0 / jnp.pi)
        * (11 * z**2 - (x**2 + y**2 + z**2))
        * (x**2 * (x**2 - 3 * y**2) + y**2 * (y**2 - 3 * x**2))
    )  # (l, m) == (6, 4)
    i_p5 = lnorm(6) * 3.0 / 32.0 * jnp.sqrt(2002.0 / jnp.pi) * z * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4)  # (l, m) == (6, 5)
    i_p6 = (
        lnorm(6) * 1.0 / 64.0 * jnp.sqrt(6006.0 / jnp.pi) * (x**6 - 15 * x**4 * y**2 + 15 * x**2 * y**4 - y**6)
    )  # (l, m) == (6, 6)

    S_l_m_values = jnp.stack(
        [
            s_0,
            p_m1,
            p_0,
            p_p1,
            d_m2,
            d_m1,
            d_0,
            d_p1,
            d_p2,
            f_m3,
            f_m2,
            f_m1,
            f_0,
            f_p1,
            f_p2,
            f_p3,
            g_m4,
            g_m3,
            g_m2,
            g_m1,
            g_0,
            g_p1,
            g_p2,
            g_p3,
            g_p4,
            h_m5,
            h_m4,
            h_m3,
            h_m2,
            h_m1,
            h_0,
            h_p1,
            h_p2,
            h_p3,
            h_p4,
            h_p5,
            i_m6,
            i_m5,
            i_m4,
            i_m3,
            i_m2,
            i_m1,
            i_0,
            i_p1,
            i_p2,
            i_p3,
            i_p4,
            i_p5,
            i_p6,
        ],
        axis=0,
    )
    return max_ml, S_l_m_values


def _compute_S_l_m_debug(
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
        # logger.devel(f"l={l}, type ={type(l)}")
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


#############################################################################################################
#
# The following functions are no longer used in the main code. They are kept for future reference.
#
#############################################################################################################


# no longer used in the main code
def compute_AOs_laplacian_api(
    aos_data: AOs_sphe_data,
    r_carts: jnpt.ArrayLike,
    debug=False,
) -> jax.Array:
    """Compute laplacians of the give AOs at r_carts.

    The method is for computing the laplacians of the given atomic orbital at r_carts

    Args:
        ao_datas (AOs_data): an instance of AOs_data
        r_carts (jnpt.ArrayLike): Cartesian coordinates of electrons (dim: N_e, 3)
        debug (bool): if True, numerical derivatives are computed via _debug function for debuging purpose

    Returns:
        jax.Array:
            Array containing laplacians of the AOs at r_carts. The dim. is (num_ao, N_e)

    """
    if debug:
        return _compute_AOs_laplacian_debug(aos_data, r_carts)
    else:
        return _compute_AOs_laplacian_jax(aos_data, r_carts)


# no longer used in the main code
@jit
def _compute_AOs_laplacian_jax(aos_data: AOs_sphe_data | AOs_cart_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Compute laplacians of the give AOs at r_carts.

    See compute_AOs_laplacian_api

    """
    # not very fast, but it works.
    ao_matrix_hessian = hessian(compute_AOs_api, argnums=1)(aos_data, r_carts)
    ao_matrix_laplacian = jnp.einsum("m i i u i u -> mi", ao_matrix_hessian)

    return ao_matrix_laplacian


# no longer used in the main code
def _compute_AOs_laplacian_debug(aos_data: AOs_sphe_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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


# no longer used in the main code
def compute_AOs_grad_api(
    aos_data: AOs_sphe_data, r_carts: jnpt.ArrayLike, debug=False
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute Cartesian Gradients of AOs.

    The method is for computing the Carteisan gradients (x,y,z) of
    the given atomic orbital at r_carts

    Args:
        ao_datas(AOs_data): an instance of AOs_data
        r_carts(jnpt.ArrayLike): Cartesian coordinates of electrons (dim: N_e, 3)
        debug (bool): if True, numerical derivatives are computed via _debug function for debuging purpose

    Returns:
        tuple: tuple containing gradients of the AOs at r_carts. (grad_x, grad_y, grad_z).
        The dim. of each matrix is (num_ao, N_e)

    """
    if debug:
        ao_matrix_grad_x, ao_matrix_grad_y, ao_matrix_grad_z = _compute_AOs_grad_debug(aos_data, r_carts)
    else:
        ao_matrix_grad_x, ao_matrix_grad_y, ao_matrix_grad_z = _compute_AOs_grad_jax(aos_data, r_carts)

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


# no longer used in the main code
@jit
def _compute_AOs_grad_jax(
    aos_data: AOs_sphe_data | AOs_cart_data, r_carts: jnpt.ArrayLike
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute Cartesian Gradients of AOs.

    See compute_AOs_grad_api

    """
    """
    # expansion with respect to the primitive AOs
    # compute R_n inc. the whole normalization factor
    R_carts_jnp = aos_data.atomic_center_carts_prim_jnp
    c_jnp = aos_data.coefficients_jnp
    Z_jnp = aos_data.exponents_jnp
    l_jnp = aos_data.angular_momentums_prim_jnp
    m_jnp = aos_data.magnetic_quantum_numbers_prim_jnp

    # grad in compute_primitive_AOs_grad_jax
    vmap_compute_AOs_grad_dup = vmap(
        vmap(
            _compute_primitive_AOs_grad_jax,
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

    """

    grad_full = jacrev(compute_AOs_api, argnums=1)(aos_data, r_carts)
    grad_diag = jnp.diagonal(grad_full, axis1=1, axis2=2)
    grad_diag = jnp.swapaxes(grad_diag, 1, 2)
    ao_matrix_grad_x = grad_diag[..., 0]  # (M, N)
    ao_matrix_grad_y = grad_diag[..., 1]  # (M, N)
    ao_matrix_grad_z = grad_diag[..., 2]  # (M, N)
    return ao_matrix_grad_x, ao_matrix_grad_y, ao_matrix_grad_z


# no longer used in the main code
@jit
def __compute_AOs_grad_jax_old(
    aos_data: AOs_sphe_data,
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


# no longer used in the main code
def _compute_AOs_grad_debug(
    aos_data: AOs_sphe_data,
    r_carts: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Cartesian Gradients of AOs.

    The method is for computing the Carteisan gradients (x,y,z) of
    the given atomic orbital at r_carts using FDM for debugging JAX
    implementations. See compute_AOs_grad_api
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


# no longer used in the main code
@jit
def _compute_AOs_jax_old(aos_data: AOs_sphe_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Compute AO values at the given r_carts.

    See compute_AOs_api

    """
    # Indices with respect to the contracted AOs
    # compute R_n inc. the whole normalization factor
    R_carts_jnp = aos_data.atomic_center_carts_prim_jnp
    c_jnp = aos_data.coefficients_jnp
    Z_jnp = aos_data.exponents_jnp
    l_jnp = aos_data.angular_momentums_prim_jnp
    m_jnp = aos_data.magnetic_quantum_numbers_prim_jnp

    vmap_compute_AOs_dup = vmap(
        vmap(_compute_primitive_AOs_jax, in_axes=(None, None, None, None, None, 0)),
        in_axes=(0, 0, 0, 0, 0, None),
    )

    AOs_dup = vmap_compute_AOs_dup(c_jnp, Z_jnp, l_jnp, m_jnp, R_carts_jnp, r_carts)

    orbital_indices = aos_data.orbital_indices_jnp
    num_segments = aos_data.num_ao
    AOs = jax.ops.segment_sum(AOs_dup, orbital_indices, num_segments=num_segments)
    return AOs


# no longer used in the main code
@dataclass
class AO_sphe_data:
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


# no longer used in the main code
def compute_AO_sphe(ao_data: AO_sphe_data, r_cart: list[float]) -> float:
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
            _compute_R_n_debug(
                coefficient=c,
                exponent=Z,
                R_cart=ao_data.atomic_center_cart,
                r_cart=r_cart,
            )
            for c, Z in zip(ao_data.coefficients, ao_data.exponents)
        ]
    )
    N_n_l = np.array([_compute_normalization_fator_debug(ao_data.angular_momentum, Z) for Z in ao_data.exponents])
    S_l_m = _compute_S_l_m_debug(
        atomic_center_cart=ao_data.atomic_center_cart,
        angular_momentum=ao_data.angular_momentum,
        magnetic_quantum_number=ao_data.magnetic_quantum_number,
        r_cart=r_cart,
    )

    return np.sum(N_n_l * R_n) * np.sqrt((2 * ao_data.angular_momentum + 1) / (4 * np.pi)) * S_l_m


# no longer used in the main code
def _compute_R_n_debug(
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


# no longer used in the main code
@jit
def _compute_R_n_jax(
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
    # return coefficient * jnp.exp(-1.0 * exponent * jnp.linalg.norm(r_cart - R_cart) ** 2)
    return coefficient * jnp.exp(-1.0 * exponent * jnp.sum(jnp.square(r_cart - R_cart)))


# no longer used in the main code
@jit
def _compute_S_l_m_jax_old(
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
        return jnp.sqrt((4 * jnp.pi) / (2 * l + 1))

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
        * jnp.sqrt(385.0 / (2 * jnp.pi))
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


# no longer used in the main code
def _compute_normalization_fator_debug(l: int, Z: float) -> float:
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


# no longer used in the main code
@jit
def _compute_normalization_fator_jax(l: int, Z: float) -> float:
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
    # N_n_jnp = jnp.sqrt(
    #    (2.0 ** (2 * l + 3) * jscipy.special.gamma(l + 2) * (2 * Z) ** (l + 1.5))
    #    / (jscipy.special.gamma(2 * l + 3) * jnp.sqrt(jnp.pi))
    # )
    return N_n_jnp


# no longer used in the main code
@jit
def _compute_primitive_AOs_jax(
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
    N_n_dup = _compute_normalization_fator_jax(l, exponent)
    R_n_dup = _compute_R_n_jax(coefficient, exponent, R_cart, r_cart)
    S_l_m_dup = _compute_S_l_m_jax_old(l, m, R_cart, r_cart)

    return N_n_dup * R_n_dup * jnp.sqrt((2 * l + 1) / (4 * jnp.pi)) * S_l_m_dup


# no longer used in the main code
@jit
def _compute_primitive_AOs_grad_jax(
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
    grad_x, grad_y, grad_z = grad(_compute_primitive_AOs_jax, argnums=5)(coefficient, exponent, l, m, R_cart, r_cart)
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


# no longer used in the main code
@jit
def _compute_primitive_AOs_laplacians_jax(
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
        jnp.diag(jacrev(grad(_compute_primitive_AOs_jax, argnums=5), argnums=5)(coefficient, exponent, l, m, R_cart, r_cart))
    )
    # """

    """
    # What if jacrev(grad) is replaced with the analytical one? (test using FDM) / To be refactored
    diff_h = 1.0e-5
    p = compute_primitive_AOs_jax(coefficient, exponent, l, m, R_cart, r_cart)
    diff_p_x = compute_primitive_AOs_jax(coefficient, exponent, l, m, R_cart, r_cart + jnp.array([+diff_h, 0.0, 0.0]))
    diff_m_x = compute_primitive_AOs_jax(coefficient, exponent, l, m, R_cart, r_cart + jnp.array([-diff_h, 0.0, 0.0]))
    diff_p_y = compute_primitive_AOs_jax(coefficient, exponent, l, m, R_cart, r_cart + jnp.array([0.0, +diff_h, 0.0]))
    diff_m_y = compute_primitive_AOs_jax(coefficient, exponent, l, m, R_cart, r_cart + jnp.array([0.0, -diff_h, 0.0]))
    diff_p_z = compute_primitive_AOs_jax(coefficient, exponent, l, m, R_cart, r_cart + jnp.array([0.0, 0.0, +diff_h]))
    diff_m_z = compute_primitive_AOs_jax(coefficient, exponent, l, m, R_cart, r_cart + jnp.array([0.0, 0.0, -diff_h]))
    grad2_x = (diff_p_x + diff_m_x - 2 * p) / (diff_h) ** 2
    grad2_y = (diff_p_y + diff_m_y - 2 * p) / (diff_h) ** 2
    grad2_z = (diff_p_z + diff_m_z - 2 * p) / (diff_h) ** 2

    laplacians = grad2_x + grad2_y + grad2_z
    """

    return laplacians


if __name__ == "__main__":
    import os

    # from functools import partial
    # from jax.experimental import sparse
    from .trexio_wrapper import read_trexio_file

    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)

    trial = 50

    """
    M = 160  # Number of GTO parameters (water, only prim.)
    N = 4  # Number of r vectors (water)
    M = 16000  # Number of GTO parameters (benzene, only prim.)
    N = 400  # Number of r vectors (benzene)
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(N, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(N, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(M, 3) + R_cart_min

    all_lm_pairs = [(l, m) for l in range(7) for m in range(-l, l + 1)]
    chosen_lm_pairs = random.choices(all_lm_pairs, k=M)
    l = [p[0] for p in chosen_lm_pairs]
    m = [p[1] for p in chosen_lm_pairs]
    c = np.linspace(1.0, 2.0, M)
    Z = np.linspace(1.0, 1.5, M)

    num_ao = M
    num_ao_prim = M
    orbital_indices = jnp.arange(M)
    exponents = jnp.array(Z)
    coefficients = jnp.array(c)
    angular_momentums = jnp.array(l)
    magnetic_quantum_numbers = jnp.array(m)

    r_up_carts = jnp.array(r_up_carts)
    r_dn_carts = jnp.array(r_dn_carts)

    structure_data = Structure_data(
        pbc_flag=[False, False, False],
        positions=R_carts,
        atomic_numbers=[0] * M,
        element_symbols=["X"] * M,
        atomic_labels=["X"] * M,
    )

    aos_data = AOs_data(
        structure_data=structure_data,
        nucleus_index=list(range(M)),
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
    )
    """

    # """
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "water_ccecp_ccpvtz_cart.hdf5"))
    # """

    """
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "benzene_ccpv6z_trexio.hdf5")
    )
    """

    """
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "AcOH_dimer_augccpv6z.hdf5"))
    """

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -3.0, +3.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    r_up_carts = jnp.array(r_up_carts)
    r_dn_carts = jnp.array(r_dn_carts)
    # """

    # print(aos_data)

    aos_jax_up = _compute_AOs_cart_jax(aos_data=aos_data, r_carts=r_up_carts)
    aos_jax_dn = _compute_AOs_cart_jax(aos_data=aos_data, r_carts=r_dn_carts)
    aos_jax_up.block_until_ready()
    aos_jax_dn.block_until_ready()
