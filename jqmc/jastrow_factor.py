"""Jastrow module."""

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
import itertools
from collections.abc import Callable

# set logger
from logging import Formatter, StreamHandler, getLogger

# jax modules
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import grad, hessian, jit, vmap
from jax import typing as jnpt

# jqmc module
from .atomic_orbital import AOs_cart_data, AOs_sphe_data, compute_AOs_jax
from .molecular_orbital import MOs_data, compute_MOs_jax
from .structure import Structure_data

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


# @dataclass
@struct.dataclass
class Jastrow_one_body_data:
    """Jastrow one-body dataclass.

    The class contains data for evaluating the one-body Jastrow function.

    Args:
        jastrow_1b_param (float): the parameter for 1b Jastrow part
        structure_data (Structure_data): an instance of Struructure_data
        core_electrons (list[float]): a list containing the number of removed core electrons (for ECPs)
    """

    jastrow_1b_param: float = struct.field(pytree_node=True, default=1.0)
    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    core_electrons: list[float] = struct.field(pytree_node=False, default_factory=list())

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.jastrow_1b_param < 0.0:
            logger.error(f"jastrow_1b_param = {self.jastrow_1b_param} must be non-negative.")
            raise ValueError("Invalid jastrow_1b_param.")
        if len(self.core_electrons) != len(self.structure_data.positions):
            logger.error(
                f"len(core_electrons) = {len(self.core_electrons)} must be the same as len(structure_data.positions) = {len(self.structure_data.positions)}."
            )
            raise ValueError("Inconsistent dimensions of core_electrons.")
        self.structure_data.sanity_check()

    def get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  Jastrow 1b param = {self.jastrow_1b_param}")
        info_lines.append("  1b Jastrow functional form is the exp type.")
        return info_lines

    def logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self.get_info():
            logger.info(line)

    @classmethod
    def init_jastrow_one_body_data(cls, jastrow_1b_param, structure_data, core_electrons):
        """Initialization."""
        jastrow_one_body_data = cls(
            jastrow_1b_param=jastrow_1b_param, structure_data=structure_data, core_electrons=core_electrons
        )
        return jastrow_one_body_data


@jit
def compute_Jastrow_one_body_jax(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """Function for computing Jastrow factor with the given jastrow_one_body_data.

    The api method to compute Jastrow factor with the given jastrow_one_body_data.
    Notice that the Jastrow factor does not contain exp factor. Attach this
    J to a WF with the modification, exp(J).

    Args:
        jastrow_one_body_data (Jastrow_one_body_data): an instance of Jastrow_one_body_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up electrons (dim: N_e^up, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of up electrons (dim: N_e^dn, 3)
        debug (bool): if True, this is computed via _debug function for debuging purpose

    Return:
        float: The value of Jastrow factor. Notice that the Jastrow factor does not
        contain exp factor. Attach this J to a WF with the modification, exp(J).
    """
    # Retrieve structure data and convert to JAX arrays
    R_carts = jnp.array(jastrow_one_body_data.structure_data.positions)
    atomic_numbers = jnp.array(jastrow_one_body_data.structure_data.atomic_numbers)
    core_electrons = jnp.array(jastrow_one_body_data.core_electrons)
    effective_charges = atomic_numbers - core_electrons

    def one_body_jastrow_exp(
        param: float,
        coeff: float,
        r_cart: jnpt.ArrayLike,
        R_cart: jnpt.ArrayLike,
    ) -> float:
        """Exponential form of J1."""
        one_body_jastrow = 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * coeff * jnp.linalg.norm(r_cart - R_cart)))
        return one_body_jastrow

    # Function to compute the contribution from one atom
    def atom_contrib(r_cart, R_cart, Z_eff):
        j1b = jastrow_one_body_data.jastrow_1b_param
        coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
        return -((2.0 * Z_eff) ** (3.0 / 4.0)) * one_body_jastrow_exp(j1b, coeff, r_cart, R_cart)

    # Sum the contributions from all atoms for a single electron
    def electron_contrib(r_cart, R_carts, effective_charges):
        # Apply vmap over positions and effective_charges
        return jnp.sum(jax.vmap(atom_contrib, in_axes=(None, 0, 0))(r_cart, R_carts, effective_charges))

    # Sum contributions for all spin-up electrons
    J1_up = jnp.sum(jax.vmap(electron_contrib, in_axes=(0, None, None))(r_up_carts, R_carts, effective_charges))
    # Sum contributions for all spin-down electrons
    J1_dn = jnp.sum(jax.vmap(electron_contrib, in_axes=(0, None, None))(r_dn_carts, R_carts, effective_charges))

    return J1_up + J1_dn


def compute_Jastrow_one_body_debug(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """See compute_Jastrow_one_body_api."""
    positions = jastrow_one_body_data.structure_data.positions
    atomic_numbers = jastrow_one_body_data.structure_data.atomic_numbers
    core_electrons = jastrow_one_body_data.core_electrons
    effective_charges = np.array(atomic_numbers) - np.array(core_electrons)

    def one_body_jastrow_exp(
        param: float, coeff: float, r_cart: npt.NDArray[np.float64], R_cart: npt.NDArray[np.float64]
    ) -> float:
        """Exponential form of J1."""
        one_body_jastrow = 1.0 / (2.0 * param) * (1.0 - np.exp(-param * coeff * np.linalg.norm(r_cart - R_cart)))
        return one_body_jastrow

    J1_up = 0.0
    for r_up in r_up_carts:
        for R_cart, Z_eff in zip(positions, effective_charges):
            coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
            J1_up += -((2.0 * Z_eff) ** (3.0 / 4.0)) * one_body_jastrow_exp(
                jastrow_one_body_data.jastrow_1b_param, coeff, r_up, R_cart
            )

    J1_dn = 0.0
    for r_up in r_dn_carts:
        for R_cart, Z_eff in zip(positions, effective_charges):
            coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
            J1_dn += -((2.0 * Z_eff) ** (3.0 / 4.0)) * one_body_jastrow_exp(
                jastrow_one_body_data.jastrow_1b_param, coeff, r_up, R_cart
            )

    J1 = J1_up + J1_dn

    return J1


# @dataclass
@struct.dataclass
class Jastrow_two_body_data:
    """Jastrow two-body dataclass.

    The class contains data for evaluating the two-body Jastrow function.

    Args:
        jastrow_2b_param (float): the parameter for 2b Jastrow part
    """

    jastrow_2b_param: float = struct.field(pytree_node=True, default=1.0)

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.jastrow_2b_param < 0.0:
            logger.error(f"jastrow_2b_param = {self.jastrow_2b_param} must be non-negative.")
            raise ValueError("Invalid jastrow_2b_param.")

    def get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  Jastrow 2b param = {self.jastrow_2b_param}")
        info_lines.append("  2b Jastrow functional form is the pade type.")
        return info_lines

    def logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self.get_info():
            logger.info(line)

    @classmethod
    def init_jastrow_two_body_data(cls, jastrow_2b_param=1.0):
        """Initialization."""
        jastrow_two_body_data = cls(jastrow_2b_param=jastrow_2b_param)
        return jastrow_two_body_data


@jit
def compute_Jastrow_two_body_jax(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> float:
    """Function for computing Jastrow factor with the given jastrow_two_body_data.

    The api method to compute Jastrow factor with the given jastrow_two_body_data.
    Notice that the Jastrow factor does not contain exp factor. Attach this
    J to a WF with the modification, exp(J).

    Args:
        jastrow_two_body_data (Jastrow_two_body_data): an instance of Jastrow_two_body_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up electrons (dim: N_e^up, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of up electrons (dim: N_e^dn, 3)
        debug (bool): if True, this is computed via _debug function for debuging purpose

    Return:
        float: The value of Jastrow factor. Notice that the Jastrow factor does not
        contain exp factor. Attach this J to a WF with the modification, exp(J).
    """

    def two_body_jastrow_anti_parallel_spins_exp(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Exponential form of J2 for anti-parallel spins."""
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * jnp.linalg.norm(r_cart_i - r_cart_j)))
        return two_body_jastrow

    def two_body_jastrow_parallel_spins_exp(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Exponential form of J2 for parallel spins."""
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * jnp.linalg.norm(r_cart_i - r_cart_j)))
        return two_body_jastrow

    def two_body_jastrow_anti_parallel_spins_pade(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Pade form of J2 for anti-parallel spins."""
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * jnp.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)
        )
        return two_body_jastrow

    def two_body_jastrow_parallel_spins_pade(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Pade form of J2 for parallel spins."""
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * jnp.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)
        )
        return two_body_jastrow

    vmap_two_body_jastrow_anti_parallel_spins = vmap(
        vmap(two_body_jastrow_anti_parallel_spins_pade, in_axes=(None, None, 0)), in_axes=(None, 0, None)
    )

    two_body_jastrow_anti_parallel = jnp.sum(
        vmap_two_body_jastrow_anti_parallel_spins(jastrow_two_body_data.jastrow_2b_param, r_up_carts, r_dn_carts)
    )

    def compute_parallel_sum(r_carts):
        num_particles = r_carts.shape[0]
        idx_i, idx_j = jnp.triu_indices(num_particles, k=1)
        r_i = r_carts[idx_i]
        r_j = r_carts[idx_j]
        vmap_two_body_jastrow_parallel_spins = vmap(two_body_jastrow_parallel_spins_pade, in_axes=(None, 0, 0))(
            jastrow_two_body_data.jastrow_2b_param, r_i, r_j
        )
        return jnp.sum(vmap_two_body_jastrow_parallel_spins)

    two_body_jastrow_parallel_up = compute_parallel_sum(r_up_carts)
    two_body_jastrow_parallel_dn = compute_parallel_sum(r_dn_carts)

    two_body_jastrow = two_body_jastrow_anti_parallel + two_body_jastrow_parallel_up + two_body_jastrow_parallel_dn

    return two_body_jastrow


def compute_Jastrow_two_body_debug(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """See _api method."""

    def two_body_jastrow_anti_parallel_spins_exp(
        param: float, r_cart_i: npt.NDArray[np.float64], r_cart_j: npt.NDArray[np.float64]
    ) -> float:
        """Exponential form of J2 for anti-parallel spins."""
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - np.exp(-param * np.linalg.norm(r_cart_i - r_cart_j)))
        return two_body_jastrow

    def two_body_jastrow_parallel_spins_exp(
        param: float, r_cart_i: npt.NDArray[np.float64], r_cart_j: npt.NDArray[np.float64]
    ) -> float:
        """Exponential form of J2 for parallel spins."""
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - np.exp(-param * np.linalg.norm(r_cart_i - r_cart_j)))
        return two_body_jastrow

    def two_body_jastrow_anti_parallel_spins_pade(
        param: float, r_cart_i: npt.NDArray[np.float64], r_cart_j: npt.NDArray[np.float64]
    ) -> float:
        """Pade form of J2 for anti-parallel spins."""
        two_body_jastrow = (
            np.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * np.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)
        )
        return two_body_jastrow

    def two_body_jastrow_parallel_spins_pade(
        param: float, r_cart_i: npt.NDArray[np.float64], r_cart_j: npt.NDArray[np.float64]
    ) -> float:
        """Pade form of J2 for parallel spins."""
        two_body_jastrow = (
            np.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * np.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)
        )
        return two_body_jastrow

    two_body_jastrow = (
        np.sum(
            [
                two_body_jastrow_anti_parallel_spins_pade(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    r_cart_i=r_up_cart,
                    r_cart_j=r_dn_cart,
                )
                for (r_up_cart, r_dn_cart) in itertools.product(r_up_carts, r_dn_carts)
            ]
        )
        + np.sum(
            [
                two_body_jastrow_parallel_spins_pade(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    r_cart_i=r_up_cart_i,
                    r_cart_j=r_up_cart_j,
                )
                for (r_up_cart_i, r_up_cart_j) in itertools.combinations(r_up_carts, 2)
            ]
        )
        + np.sum(
            [
                two_body_jastrow_parallel_spins_pade(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    r_cart_i=r_dn_cart_i,
                    r_cart_j=r_dn_cart_j,
                )
                for (r_dn_cart_i, r_dn_cart_j) in itertools.combinations(r_dn_carts, 2)
            ]
        )
    )

    return two_body_jastrow


# @dataclass
@struct.dataclass
class Jastrow_three_body_data:
    """Jastrow_three_body dataclass.

    The class contains data for evaluating the three-body Jastrow function.

    Args:
        orb_data (AOs_sphe_data | AOs_cart_data | MOs_data): AOs data for up-spin and dn-spin.
        j_matrix (npt.NDArray[np.float64]): J matrix dim. (orb_data.num_ao, orb_data.num_ao+1))
    """

    orb_data: AOs_sphe_data | AOs_cart_data | MOs_data = struct.field(pytree_node=True, default_factory=lambda: AOs_sphe_data())
    j_matrix: npt.NDArray[np.float64] = struct.field(pytree_node=True, default_factory=lambda: np.array([]))

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.j_matrix.shape != (
            self.orb_num,
            self.orb_num + 1,
        ):
            logger.error(
                f"dim. of j_matrix = {self.j_matrix.shape} is imcompatible with the expected one "
                + f"= ({self.orb_num}, {self.orb_num + 1}).",
            )
            raise ValueError("Inconsistent dimensions of j_matrix.")

    def get_info(self) -> list[str]:
        """Return a list of strings containing the information stored in the attributes."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  dim. of jastrow_3b_matrix = {self.j_matrix.shape}")
        info_lines.append(
            f"  j3 part of the jastrow_3b_matrix is symmetric? = {np.allclose(self.j_matrix[:, :-1], self.j_matrix[:, :-1].T)}"
        )
        # Replace orb_data.logger_info() with orb_data.get_info() output.
        info_lines.extend(self.orb_data.get_info())
        return info_lines

    def logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self.get_info():
            logger.info(line)

    @property
    def orb_num(self) -> int:
        """Get number of atomic orbitals.

        Returns:
            int: get number of atomic orbitals.
        """
        return self.orb_data.num_orb

    @property
    def compute_orb_api(self) -> Callable[..., npt.NDArray[np.float64]]:
        """Function for computing AOs or MOs.

        The api method to compute AOs or MOs corresponding to instances
        stored in self.orb_data

        Return:
            Callable: The api method to compute AOs or MOs.

        Raises:
            NotImplementedError:
                If the instances of orb_data is neither AOs_data nor MOs_data.
        """
        if isinstance(self.orb_data, AOs_sphe_data):
            return compute_AOs_jax
        elif isinstance(self.orb_data, AOs_cart_data):
            return compute_AOs_jax
        elif isinstance(self.orb_data, MOs_data):
            return compute_MOs_jax
        else:
            raise NotImplementedError

    @classmethod
    def init_jastrow_three_body_data(cls, orb_data: AOs_sphe_data | AOs_cart_data | MOs_data):
        """Initialization."""
        j_matrix = np.zeros((orb_data.num_orb, orb_data.num_orb + 1))
        # j_matrix = np.random.uniform(0.01, 0.10, size=(orb_data.num_orb, orb_data.num_orb + 1))
        jastrow_three_body_data = cls(
            orb_data=orb_data,
            j_matrix=j_matrix,
        )
        return jastrow_three_body_data


@struct.dataclass
class Jastrow_three_body_data_deriv_params(Jastrow_three_body_data):
    """See Jastrow_three_body_data."""

    orb_data: MOs_data | AOs_sphe_data | AOs_cart_data = struct.field(
        pytree_node=False, default_factory=lambda: AOs_sphe_data()
    )
    j_matrix: npt.NDArray[np.float64] = struct.field(pytree_node=True, default_factory=lambda: np.array([]))

    @classmethod
    def from_base(cls, jastrow_three_body_data: Jastrow_three_body_data):
        """Switch pytree_node."""
        return cls(orb_data=jastrow_three_body_data.orb_data, j_matrix=jastrow_three_body_data.j_matrix)


@struct.dataclass
class Jastrow_three_body_data_deriv_R(Jastrow_three_body_data):
    """See Jastrow_three_body_data."""

    orb_data: MOs_data | AOs_sphe_data | AOs_cart_data = struct.field(pytree_node=True, default_factory=lambda: AOs_sphe_data())
    j_matrix: npt.NDArray[np.float64] = struct.field(pytree_node=False, default_factory=lambda: np.array([]))

    @classmethod
    def from_base(cls, jastrow_three_body_data: Jastrow_three_body_data):
        """Switch pytree_node."""
        return cls(orb_data=jastrow_three_body_data.orb_data, j_matrix=jastrow_three_body_data.j_matrix)


def compute_Jastrow_three_body_jax(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> float:
    """Function for computing Jastrow factor with the given jastrow_three_body_data.

    The api method to compute Jastrow factor with the given jastrow_three_body_data.
    Notice that the Jastrow factor does not contain exp factor. Attach this
    J to a WF with the modification, exp(J).

    Args:
        jastrow_three_body_data (Jastrow_three_body_data): an instance of Jastrow_three_body_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up electrons (dim: N_e^up, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of up electrons (dim: N_e^dn, 3)

    Return:
        float: The value of Jastrow factor. Notice that the Jastrow factor does not
        contain exp factor. Attach this J to a WF with the modification, exp(J).
    """
    num_electron_up = len(r_up_carts)
    num_electron_dn = len(r_dn_carts)

    aos_up = jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, r_up_carts))
    aos_dn = jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, r_dn_carts))

    K_up = jnp.tril(jnp.ones((num_electron_up, num_electron_up)), k=-1)
    K_dn = jnp.tril(jnp.ones((num_electron_dn, num_electron_dn)), k=-1)

    j1_matrix_up = jastrow_three_body_data.j_matrix[:, -1]
    j1_matrix_dn = jastrow_three_body_data.j_matrix[:, -1]
    j3_matrix_up_up = jastrow_three_body_data.j_matrix[:, :-1]
    j3_matrix_dn_dn = jastrow_three_body_data.j_matrix[:, :-1]
    j3_matrix_up_dn = jastrow_three_body_data.j_matrix[:, :-1]

    e_up = jnp.ones(num_electron_up).T
    e_dn = jnp.ones(num_electron_dn).T

    # print(f"aos_up.shape={aos_up.shape}")
    # print(f"aos_dn.shape={aos_dn.shape}")
    # print(f"e_up.shape={e_up.shape}")
    # print(f"e_dn.shape={e_dn.shape}")
    # print(f"j3_matrix_up_up.shape={j3_matrix_up_up.shape}")
    # print(f"j3_matrix_dn_dn.shape={j3_matrix_dn_dn.shape}")
    # print(f"j3_matrix_up_dn.shape={j3_matrix_up_dn.shape}")

    J3 = (
        j1_matrix_up @ aos_up @ e_up
        + j1_matrix_dn @ aos_dn @ e_dn
        + jnp.trace(aos_up.T @ j3_matrix_up_up @ aos_up @ K_up)
        + jnp.trace(aos_dn.T @ j3_matrix_dn_dn @ aos_dn @ K_dn)
        + e_up.T @ aos_up.T @ j3_matrix_up_dn @ aos_dn @ e_dn
    )

    return J3


def compute_Jastrow_three_body_debug(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """See _api method."""
    aos_up = jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, r_up_carts)
    aos_dn = jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, r_dn_carts)

    # compute one body
    J_1_up = 0.0
    j1_vector_up = jastrow_three_body_data.j_matrix[:, -1]
    for i in range(len(r_up_carts)):
        ao_up = aos_up[:, i]
        for al in range(len(ao_up)):
            J_1_up += j1_vector_up[al] * ao_up[al]

    J_1_dn = 0.0
    j1_vector_dn = jastrow_three_body_data.j_matrix[:, -1]
    for i in range(len(r_dn_carts)):
        ao_dn = aos_dn[:, i]
        for al in range(len(ao_dn)):
            J_1_dn += j1_vector_dn[al] * ao_dn[al]

    # compute three-body
    J_3_up_up = 0.0
    j3_matrix_up_up = jastrow_three_body_data.j_matrix[:, :-1]
    for i in range(len(r_up_carts)):
        for j in range(i + 1, len(r_up_carts)):
            ao_up_i = aos_up[:, i]
            ao_up_j = aos_up[:, j]
            for al in range(len(ao_up_i)):
                for bm in range(len(ao_up_j)):
                    J_3_up_up += j3_matrix_up_up[al, bm] * ao_up_i[al] * ao_up_j[bm]

    J_3_dn_dn = 0.0
    j3_matrix_dn_dn = jastrow_three_body_data.j_matrix[:, :-1]
    for i in range(len(r_dn_carts)):
        for j in range(i + 1, len(r_dn_carts)):
            ao_dn_i = aos_dn[:, i]
            ao_dn_j = aos_dn[:, j]
            for al in range(len(ao_dn_i)):
                for bm in range(len(ao_dn_j)):
                    J_3_dn_dn += j3_matrix_dn_dn[al, bm] * ao_dn_i[al] * ao_dn_j[bm]

    J_3_up_dn = 0.0
    j3_matrix_up_dn = jastrow_three_body_data.j_matrix[:, :]
    for i in range(len(r_up_carts)):
        for j in range(len(r_dn_carts)):
            ao_up_i = aos_up[:, i]
            ao_dn_j = aos_dn[:, j]
            for al in range(len(ao_up_i)):
                for bm in range(len(ao_dn_j)):
                    J_3_up_dn += j3_matrix_up_dn[al, bm] * ao_up_i[al] * ao_dn_j[bm]

    J3 = J_1_up + J_1_dn + J_3_up_up + J_3_dn_dn + J_3_up_dn

    return J3


@struct.dataclass
class Jastrow_data:
    """Jastrow dataclass.

    The class contains data for evaluating a Jastrow function.

    Args:
        jastrow_one_body_data (Jastrow_one_body_data):
            An instance of Jastrow_one_body_data. If None, the one-body Jastrow is turned off.
        jastrow_two_body_data (Jastrow_two_body_data):
            An instance of Jastrow_two_body_data. If None, the two-body Jastrow is turned off.
        jastrow_three_body_data (Jastrow_three_body_data):
            An instance of Jastrow_three_body_data. if None, the three-body Jastrow is turned off.
    """

    jastrow_one_body_data: Jastrow_one_body_data = struct.field(pytree_node=True, default=None)
    jastrow_two_body_data: Jastrow_two_body_data = struct.field(pytree_node=True, default=None)
    jastrow_three_body_data: Jastrow_three_body_data = struct.field(pytree_node=True, default=None)

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.jastrow_one_body_data is not None:
            self.jastrow_one_body_data.sanity_check()
        if self.jastrow_two_body_data is not None:
            self.jastrow_two_body_data.sanity_check()
        if self.jastrow_three_body_data is not None:
            self.jastrow_three_body_data.sanity_check()

    def get_info(self) -> list[str]:
        """Return a list of strings representing the logged information from Jastrow data attributes."""
        info_lines = []
        # Replace jastrow_one_body_data.logger_info() with its get_info() output if available.
        if self.jastrow_one_body_data is not None:
            info_lines.extend(self.jastrow_one_body_data.get_info())
        # Replace jastrow_two_body_data.logger_info() with its get_info() output if available.
        if self.jastrow_two_body_data is not None:
            info_lines.extend(self.jastrow_two_body_data.get_info())
        # Replace jastrow_three_body_data.logger_info() with its get_info() output if available.
        if self.jastrow_three_body_data is not None:
            info_lines.extend(self.jastrow_three_body_data.get_info())
        return info_lines

    def logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self.get_info():
            logger.info(line)


@struct.dataclass
class Jastrow_data_deriv_params(Jastrow_data):
    """See Jastrow_data."""

    jastrow_one_body_data: Jastrow_one_body_data = struct.field(pytree_node=True, default=None)
    jastrow_two_body_data: Jastrow_two_body_data = struct.field(pytree_node=True, default=None)
    jastrow_three_body_data: Jastrow_three_body_data = struct.field(pytree_node=True, default=None)

    @classmethod
    def from_base(cls, jastrow_data: Jastrow_data):
        """Switch pytree_node."""
        jastrow_one_body_data = jastrow_data.jastrow_one_body_data
        jastrow_two_body_data = jastrow_data.jastrow_two_body_data
        jastrow_three_body_data = Jastrow_three_body_data_deriv_params.from_base(jastrow_data.jastrow_three_body_data)

        return cls(
            jastrow_one_body_data=jastrow_one_body_data,
            jastrow_two_body_data=jastrow_two_body_data,
            jastrow_three_body_data=jastrow_three_body_data,
        )


@struct.dataclass
class Jastrow_data_deriv_R(Jastrow_data):
    """See Jastrow_data."""

    jastrow_one_body_data: Jastrow_one_body_data = struct.field(pytree_node=True, default=None)
    jastrow_two_body_data: Jastrow_two_body_data = struct.field(pytree_node=False, default=None)
    jastrow_three_body_data: Jastrow_three_body_data = struct.field(pytree_node=True, default=None)

    @classmethod
    def from_base(cls, jastrow_data: Jastrow_data):
        """Switch pytree_node."""
        jastrow_one_body_data = jastrow_data.jastrow_one_body_data
        jastrow_two_body_data = jastrow_data.jastrow_two_body_data
        jastrow_three_body_data = Jastrow_three_body_data_deriv_R.from_base(jastrow_data.jastrow_three_body_data)

        return cls(
            jastrow_one_body_data=jastrow_one_body_data,
            jastrow_two_body_data=jastrow_two_body_data,
            jastrow_three_body_data=jastrow_three_body_data,
        )


def compute_Jastrow_part_jax(jastrow_data: Jastrow_data, r_up_carts: jnpt.ArrayLike, r_dn_carts: jnpt.ArrayLike) -> float:
    """Function for computing Jastrow factor with the given jastrow_data.

    The api method to compute Jastrow factor with the given jastrow_data.
    Notice that the Jastrow factor does not contain exp factor. Attach this
    J to a WF with the modification, exp(J).

    Args:
        jastrow_data (Jastrow_data): an instance of Jastrow_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up electrons (dim: N_e^up, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of up electrons (dim: N_e^dn, 3)

    Return:
        float: The value of Jastrow factor. Notice that the Jastrow factor does not
        contain exp factor. Attach this J to a WF with the modification, exp(J).
    """
    J1 = 0.0
    J2 = 0.0
    J3 = 0.0

    # one-body
    if jastrow_data.jastrow_one_body_data is not None:
        J1 += compute_Jastrow_one_body_jax(jastrow_data.jastrow_one_body_data, r_up_carts, r_dn_carts)

    # two-body
    if jastrow_data.jastrow_two_body_data is not None:
        J2 += compute_Jastrow_two_body_jax(jastrow_data.jastrow_two_body_data, r_up_carts, r_dn_carts)

    # three-body
    if jastrow_data.jastrow_three_body_data is not None:
        J3 += compute_Jastrow_three_body_jax(jastrow_data.jastrow_three_body_data, r_up_carts, r_dn_carts)

    J = J1 + J2 + J3

    return J


def compute_Jastrow_part_debug(
    jastrow_data: Jastrow_data, r_up_carts: npt.NDArray[np.float64], r_dn_carts: npt.NDArray[np.float64]
) -> float:
    """See compute_Jastrow_part_jax for more details."""
    J1 = 0.0
    J2 = 0.0
    J3 = 0.0

    # one-body
    if jastrow_data.jastrow_one_body_data is not None:
        J1 += compute_Jastrow_one_body_debug(jastrow_data.jastrow_one_body_data, r_up_carts, r_dn_carts)

    # two-body
    if jastrow_data.jastrow_two_body_data is not None:
        J2 += compute_Jastrow_two_body_debug(jastrow_data.jastrow_two_body_data, r_up_carts, r_dn_carts)

    # three-body
    if jastrow_data.jastrow_three_body_data is not None:
        J3 += compute_Jastrow_three_body_debug(jastrow_data.jastrow_three_body_data, r_up_carts, r_dn_carts)

    J = J1 + J2 + J3

    return J


#############################################################################################################
#
# The following functions are no longer used in the main code. They are kept for future reference.
#
#############################################################################################################


# no longer used in the main code
def compute_ratio_Jastrow_part_jax(
    jastrow_data: Jastrow_data,
    old_r_up_carts: npt.NDArray[np.float64],
    old_r_dn_carts: npt.NDArray[np.float64],
    new_r_up_carts_arr: npt.NDArray[np.float64],
    new_r_dn_carts_arr: npt.NDArray[np.float64],
) -> npt.NDArray:
    """Function for computing the ratio of the Jastrow factor with the given jastrow_data between new_r_up_carts and old_r_up_carts.

    The api method to compute the ratio of the Jastrow factor with the given jastrow_data between new_r_up_carts and old_r_up_carts.
    i.e., J(new_r_carts_arr) / J(old_r_carts)

    Notice that the Jastrow factor does contain exp factor!

    Args:
        jastrow_data (Jastrow_data): an instance of Jastrow_data
        old_r_up_carts (jnpt.ArrayLike): Old Cartesian coordinates of up electrons (dim: N_e^up, 3)
        old_r_dn_carts (jnpt.ArrayLike): Old Cartesian coordinates of down electrons (dim: N_e^dn, 3)
        new_r_up_carts_arr (jnpt.ArrayLike): New Cartesian coordinate grids of up electrons (dim: N_grid, N_e^up, 3)
        new_r_dn_carts_arr (jnpt.ArrayLike): New Cartesian coordinate grids of down electrons (dim: N_grid, N_e^dn, 3)
        debug (bool): if True, this is computed via _debug function for debuging purpose

    Return:
        npt.NDArray: The value of Jastrow factor ratios. Notice that the Jastrow factor does contain exp factor. (dim: N_grid)
    """
    J_ratio = 1.0

    def two_body_jastrow_anti_parallel_spins_exp(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Exponential form of J2 for anti-parallel spins."""
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * jnp.linalg.norm(r_cart_i - r_cart_j)))
        return two_body_jastrow

    def two_body_jastrow_parallel_spins_exp(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Exponential form of J2 for parallel spins."""
        two_body_jastrow = 1.0 / (2.0 * param) * (1.0 - jnp.exp(-param * jnp.linalg.norm(r_cart_i - r_cart_j)))
        return two_body_jastrow

    def two_body_jastrow_anti_parallel_spins_pade(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Pade form of J2 for anti-parallel spins."""
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * jnp.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)
        )
        return two_body_jastrow

    def two_body_jastrow_parallel_spins_pade(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Pade form of J2 for parallel spins."""
        two_body_jastrow = (
            jnp.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * jnp.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)
        )
        return two_body_jastrow

    def compute_one_grid_J2(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
        delta_up = new_r_up_carts - old_r_up_carts
        delta_dn = new_r_dn_carts - old_r_dn_carts
        up_all_zero = jnp.all(delta_up == 0)

        diff = jax.lax.cond(up_all_zero, lambda _: delta_dn, lambda _: delta_up, operand=None)
        nonzero_in_rows = jnp.any(diff != 0, axis=1)
        idx = jnp.argmax(nonzero_in_rows)

        def up_case(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
            new_r_up_carts_extracted = jnp.expand_dims(new_r_up_carts[idx], axis=0)  # shape=(1,3)
            old_r_up_carts_extracted = jnp.expand_dims(old_r_up_carts[idx], axis=0)  # shape=(1,3)
            J2_up_up_new = jnp.sum(
                vmap(two_body_jastrow_parallel_spins_pade, in_axes=(None, None, 0))(
                    jastrow_2b_param, new_r_up_carts_extracted, new_r_up_carts
                )
            )
            J2_up_up_old = jnp.sum(
                vmap(two_body_jastrow_parallel_spins_pade, in_axes=(None, None, 0))(
                    jastrow_2b_param, old_r_up_carts_extracted, old_r_up_carts
                )
            )
            J2_up_dn_new = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins_pade, in_axes=(None, None, 0))(
                    jastrow_2b_param, new_r_up_carts_extracted, old_r_dn_carts
                )
            )
            J2_up_dn_old = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins_pade, in_axes=(None, None, 0))(
                    jastrow_2b_param, old_r_up_carts_extracted, old_r_dn_carts
                )
            )
            return jnp.exp(J2_up_dn_new - J2_up_dn_old + J2_up_up_new - J2_up_up_old)

        def dn_case(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
            new_r_dn_carts_extracted = jnp.expand_dims(new_r_dn_carts[idx], axis=0)  # shape=(1,3)
            old_r_dn_carts_extracted = jnp.expand_dims(old_r_dn_carts[idx], axis=0)  # shape=(1,3)
            J2_dn_dn_new = jnp.sum(
                vmap(two_body_jastrow_parallel_spins_pade, in_axes=(None, None, 0))(
                    jastrow_2b_param, new_r_dn_carts_extracted, new_r_dn_carts
                )
            )
            J2_dn_dn_old = jnp.sum(
                vmap(two_body_jastrow_parallel_spins_pade, in_axes=(None, None, 0))(
                    jastrow_2b_param, old_r_dn_carts_extracted, old_r_dn_carts
                )
            )
            J2_up_dn_new = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins_pade, in_axes=(None, 0, None))(
                    jastrow_2b_param, old_r_up_carts, new_r_dn_carts_extracted
                )
            )
            J2_up_dn_old = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins_pade, in_axes=(None, 0, None))(
                    jastrow_2b_param, old_r_up_carts, old_r_dn_carts_extracted
                )
            )

            return jnp.exp(J2_up_dn_new - J2_up_dn_old + J2_dn_dn_new - J2_dn_dn_old)

        return jax.lax.cond(
            up_all_zero,
            dn_case,
            up_case,
            *(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts),
        )

    def compute_one_grid_J3(jastrow_three_body_data, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
        delta_up = new_r_up_carts - old_r_up_carts
        delta_dn = new_r_dn_carts - old_r_dn_carts
        up_all_zero = jnp.all(delta_up == 0)

        diff = jax.lax.cond(up_all_zero, lambda _: delta_dn, lambda _: delta_up, operand=None)
        nonzero_in_rows = jnp.any(diff != 0, axis=1)
        idx = jnp.argmax(nonzero_in_rows)

        num_electron_up = len(old_r_up_carts)
        num_electron_dn = len(old_r_dn_carts)
        aos_up = jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, old_r_up_carts))
        aos_dn = jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, old_r_dn_carts))
        j1_matrix_up = jastrow_three_body_data.j_matrix[:, -1]
        j1_matrix_dn = jastrow_three_body_data.j_matrix[:, -1]
        j3_matrix_up_up = jastrow_three_body_data.j_matrix[:, :-1]
        j3_matrix_dn_dn = jastrow_three_body_data.j_matrix[:, :-1]
        j3_matrix_up_dn = jastrow_three_body_data.j_matrix[:, :-1]
        e_up = jnp.ones(num_electron_up).T
        e_dn = jnp.ones(num_electron_dn).T

        def up_case(new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
            new_r_up_carts_extracted = jnp.expand_dims(new_r_up_carts[idx], axis=0)  # shape=(1,3)
            old_r_up_carts_extracted = jnp.expand_dims(old_r_up_carts[idx], axis=0)  # shape=(1,3)

            aos_up_p = jnp.array(
                jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, new_r_up_carts_extracted)
            ) - jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, old_r_up_carts_extracted))

            indices = jnp.arange(num_electron_up)
            Q_up_c = (idx < indices).astype(jnp.float64).reshape(-1, 1)
            Q_up_r = (idx > indices).astype(jnp.float64).reshape(1, -1)
            J3_ratio = jnp.exp(
                j1_matrix_up @ aos_up_p
                + jnp.trace(aos_up_p.T @ j3_matrix_up_up @ aos_up @ Q_up_c)
                + jnp.trace(aos_up.T @ j3_matrix_up_up @ aos_up_p @ Q_up_r)
                + aos_up_p.T @ j3_matrix_up_dn @ aos_dn @ e_dn
            )

            return J3_ratio

        def dn_case(new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts):
            new_r_dn_carts_extracted = jnp.expand_dims(new_r_dn_carts[idx], axis=0)  # shape=(1,3)
            old_r_dn_carts_extracted = jnp.expand_dims(old_r_dn_carts[idx], axis=0)  # shape=(1,3)

            aos_dn_p = jnp.array(
                jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, new_r_dn_carts_extracted)
            ) - jnp.array(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, old_r_dn_carts_extracted))

            indices = jnp.arange(num_electron_dn)
            Q_dn_c = (idx < indices).astype(jnp.float64).reshape(-1, 1)
            Q_dn_r = (idx > indices).astype(jnp.float64).reshape(1, -1)
            J3_ratio = jnp.exp(
                j1_matrix_dn @ aos_dn_p
                + jnp.trace(aos_dn_p.T @ j3_matrix_dn_dn @ aos_dn @ Q_dn_c)
                + jnp.trace(aos_dn.T @ j3_matrix_dn_dn @ aos_dn_p @ Q_dn_r)
                + e_up.T @ aos_up.T @ j3_matrix_up_dn @ aos_dn_p
            )

            return J3_ratio

        return jax.lax.cond(
            up_all_zero,
            dn_case,
            up_case,
            *(new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts),
        )

    # J2 part
    if jastrow_data.jastrow_two_body_data is not None:
        # vectorization along grid
        J2_ratio = vmap(compute_one_grid_J2, in_axes=(None, 0, 0, None, None))(
            jastrow_data.jastrow_two_body_data.jastrow_2b_param,
            new_r_up_carts_arr,
            new_r_dn_carts_arr,
            old_r_up_carts,
            old_r_dn_carts,
        )

        J_ratio *= jnp.ravel(J2_ratio)

    # """
    # J3 part
    if jastrow_data.jastrow_three_body_data is not None:
        # vectorization along grid
        J3_ratio = vmap(compute_one_grid_J3, in_axes=(None, 0, 0, None, None))(
            jastrow_data.jastrow_three_body_data,
            new_r_up_carts_arr,
            new_r_dn_carts_arr,
            old_r_up_carts,
            old_r_dn_carts,
        )

        J_ratio *= jnp.ravel(J3_ratio)
    # """
    return J_ratio


# no longer used in the main code
def compute_ratio_Jastrow_part_debug(
    jastrow_data: Jastrow_data,
    old_r_up_carts: npt.NDArray[np.float64],
    old_r_dn_carts: npt.NDArray[np.float64],
    new_r_up_carts_arr: npt.NDArray[np.float64],
    new_r_dn_carts_arr: npt.NDArray[np.float64],
) -> npt.NDArray:
    """See _api method."""
    return np.array(
        [
            np.exp(compute_Jastrow_part_jax(jastrow_data, new_r_up_carts, new_r_dn_carts))
            / np.exp(compute_Jastrow_part_jax(jastrow_data, old_r_up_carts, old_r_dn_carts))
            for new_r_up_carts, new_r_dn_carts in zip(new_r_up_carts_arr, new_r_dn_carts_arr)
        ]
    )


# no longer used in the main code
def compute_grads_and_laplacian_Jastrow_part_jax(
    jastrow_data: Jastrow_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float | complex,
]:
    """Function for computing grads and laplacians with a given Jastrow_data.

    The method is for computing the gradients and the sum of laplacians of J at (r_up_carts, r_dn_carts)
    with a given Jastrow_data.

    Args:
        jastrow_data (Jastrow_data): an instance of Jastrow_two_body_data class
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        the gradients(x,y,z) of J and the sum of laplacians of J at (r_up_carts, r_dn_carts).
    """
    grad_J2_up, grad_J2_dn, sum_laplacian_J2 = 0.0, 0.0, 0.0
    grad_J3_up, grad_J3_dn, sum_laplacian_J3 = 0.0, 0.0, 0.0

    # two-body
    if jastrow_data.jastrow_two_body_data is not None:
        grad_J2_up, grad_J2_dn, sum_laplacian_J2 = compute_grads_and_laplacian_Jastrow_two_body_jax(
            jastrow_data.jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )
        grad_J2_up += grad_J2_up
        grad_J2_dn += grad_J2_dn
        sum_laplacian_J2 += sum_laplacian_J2

    # three-body
    if jastrow_data.jastrow_three_body_data is not None:
        grad_J3_up_add, grad_J3_dn_add, sum_laplacian_J3_add = compute_grads_and_laplacian_Jastrow_three_body_jax(
            jastrow_data.jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        grad_J3_up += grad_J3_up_add
        grad_J3_dn += grad_J3_dn_add
        sum_laplacian_J3 += sum_laplacian_J3_add

    grad_J_up = grad_J2_up + grad_J3_up
    grad_J_dn = grad_J2_dn + grad_J3_dn
    sum_laplacian_J = sum_laplacian_J2 + sum_laplacian_J3

    return grad_J_up, grad_J_dn, sum_laplacian_J


# no longer used in the main code
def compute_grads_and_laplacian_Jastrow_two_body_jax(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
]:
    """Function for computing grads and laplacians with a given Jastrow_two_body_data.

    The method is for computing the gradients and the sum of laplacians of J at (r_up_carts, r_dn_carts)
    with a given Jastrow_two_body_data.

    Args:
        jastrow_two_body_data (Jastrow_two_body_data): an instance of Jastrow_two_body_data class
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        the gradients(x,y,z) of J(twobody) and the sum of laplacians of J(twobody) at (r_up_carts, r_dn_carts).
    """
    # grad_J2_up, grad_J2_dn, sum_laplacian_J2 = (
    #    compute_grads_and_laplacian_Jastrow_two_body_debug(
    #        jastrow_two_body_data, r_up_carts, r_dn_carts
    #    )
    # )
    grad_J2_up, grad_J2_dn, sum_laplacian_J2 = _compute_grads_and_laplacian_Jastrow_two_body_jax(
        jastrow_two_body_data, r_up_carts, r_dn_carts
    )

    if grad_J2_up.shape != r_up_carts.shape:
        logger.error(f"grad_J2_up.shape = {grad_J2_up.shape} is inconsistent with the expected one = {r_up_carts.shape}")
        raise ValueError

    if grad_J2_dn.shape != r_dn_carts.shape:
        logger.error(f"grad_J2_dn.shape = {grad_J2_dn.shape} is inconsistent with the expected one = {r_dn_carts.shape}")
        raise ValueError

    return grad_J2_up, grad_J2_dn, sum_laplacian_J2


# no longer used in the main code
@jit
def _compute_grads_and_laplacian_Jastrow_two_body_jax(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
]:
    """See _api method."""
    r_up_carts = jnp.array(r_up_carts)
    r_dn_carts = jnp.array(r_dn_carts)

    # compute grad
    grad_J2_up = grad(compute_Jastrow_two_body_jax, argnums=1)(jastrow_two_body_data, r_up_carts, r_dn_carts)

    grad_J2_dn = grad(compute_Jastrow_two_body_jax, argnums=2)(jastrow_two_body_data, r_up_carts, r_dn_carts)

    # compute laplacians
    hessian_J2_up = hessian(compute_Jastrow_two_body_jax, argnums=1)(jastrow_two_body_data, r_up_carts, r_dn_carts)
    sum_laplacian_J2_up = jnp.einsum("ijij->", hessian_J2_up)

    hessian_J2_dn = hessian(compute_Jastrow_two_body_jax, argnums=2)(jastrow_two_body_data, r_up_carts, r_dn_carts)
    sum_laplacian_J2_dn = jnp.einsum("ijij->", hessian_J2_dn)

    sum_laplacian_J2 = sum_laplacian_J2_up + sum_laplacian_J2_dn

    return grad_J2_up, grad_J2_dn, sum_laplacian_J2


# no longer used in the main code
def compute_grads_and_laplacian_Jastrow_two_body_debug(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64 | np.complex128],
    npt.NDArray[np.float64 | np.complex128],
    float,
]:
    """See _api method."""
    diff_h = 1.0e-5

    # grad up
    grad_x_up = []
    grad_y_up = []
    grad_z_up = []
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up_carts = r_up_carts.copy()
        diff_p_y_r_up_carts = r_up_carts.copy()
        diff_p_z_r_up_carts = r_up_carts.copy()
        diff_p_x_r_up_carts[r_i][0] += diff_h
        diff_p_y_r_up_carts[r_i][1] += diff_h
        diff_p_z_r_up_carts[r_i][2] += diff_h

        J2_p_x_up = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_p_y_up = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_p_z_up = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        diff_m_x_r_up_carts = r_up_carts.copy()
        diff_m_y_r_up_carts = r_up_carts.copy()
        diff_m_z_r_up_carts = r_up_carts.copy()
        diff_m_x_r_up_carts[r_i][0] -= diff_h
        diff_m_y_r_up_carts[r_i][1] -= diff_h
        diff_m_z_r_up_carts[r_i][2] -= diff_h

        J2_m_x_up = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_y_up = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_z_up = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        grad_x_up.append((J2_p_x_up - J2_m_x_up) / (2.0 * diff_h))
        grad_y_up.append((J2_p_y_up - J2_m_y_up) / (2.0 * diff_h))
        grad_z_up.append((J2_p_z_up - J2_m_z_up) / (2.0 * diff_h))

    # grad dn
    grad_x_dn = []
    grad_y_dn = []
    grad_z_dn = []
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn_carts = r_dn_carts.copy()
        diff_p_y_r_dn_carts = r_dn_carts.copy()
        diff_p_z_r_dn_carts = r_dn_carts.copy()
        diff_p_x_r_dn_carts[r_i][0] += diff_h
        diff_p_y_r_dn_carts[r_i][1] += diff_h
        diff_p_z_r_dn_carts[r_i][2] += diff_h

        J2_p_x_dn = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn_carts,
        )
        J2_p_y_dn = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn_carts,
        )
        J2_p_z_dn = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn_carts,
        )

        diff_m_x_r_dn_carts = r_dn_carts.copy()
        diff_m_y_r_dn_carts = r_dn_carts.copy()
        diff_m_z_r_dn_carts = r_dn_carts.copy()
        diff_m_x_r_dn_carts[r_i][0] -= diff_h
        diff_m_y_r_dn_carts[r_i][1] -= diff_h
        diff_m_z_r_dn_carts[r_i][2] -= diff_h

        J2_m_x_dn = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn_carts,
        )
        J2_m_y_dn = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn_carts,
        )
        J2_m_z_dn = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn_carts,
        )

        grad_x_dn.append((J2_p_x_dn - J2_m_x_dn) / (2.0 * diff_h))
        grad_y_dn.append((J2_p_y_dn - J2_m_y_dn) / (2.0 * diff_h))
        grad_z_dn.append((J2_p_z_dn - J2_m_z_dn) / (2.0 * diff_h))

    grad_J2_up = np.array([grad_x_up, grad_y_up, grad_z_up]).T
    grad_J2_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn]).T

    # laplacian
    diff_h2 = 1.0e-3  # for laplacian

    J2_ref = compute_Jastrow_two_body_jax(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    sum_laplacian_J2 = 0.0

    # laplacians up
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up2_carts = r_up_carts.copy()
        diff_p_y_r_up2_carts = r_up_carts.copy()
        diff_p_z_r_up2_carts = r_up_carts.copy()
        diff_p_x_r_up2_carts[r_i][0] += diff_h2
        diff_p_y_r_up2_carts[r_i][1] += diff_h2
        diff_p_z_r_up2_carts[r_i][2] += diff_h2

        J2_p_x_up2 = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_p_y_up2 = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        J2_p_z_up2 = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        diff_m_x_r_up2_carts = r_up_carts.copy()
        diff_m_y_r_up2_carts = r_up_carts.copy()
        diff_m_z_r_up2_carts = r_up_carts.copy()
        diff_m_x_r_up2_carts[r_i][0] -= diff_h2
        diff_m_y_r_up2_carts[r_i][1] -= diff_h2
        diff_m_z_r_up2_carts[r_i][2] -= diff_h2

        J2_m_x_up2 = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_y_up2 = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_z_up2 = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        gradgrad_x_up = (J2_p_x_up2 + J2_m_x_up2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_y_up = (J2_p_y_up2 + J2_m_y_up2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_z_up = (J2_p_z_up2 + J2_m_z_up2 - 2 * J2_ref) / (diff_h2**2)

        sum_laplacian_J2 += gradgrad_x_up + gradgrad_y_up + gradgrad_z_up

    # laplacians dn
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn2_carts = r_dn_carts.copy()
        diff_p_y_r_dn2_carts = r_dn_carts.copy()
        diff_p_z_r_dn2_carts = r_dn_carts.copy()
        diff_p_x_r_dn2_carts[r_i][0] += diff_h2
        diff_p_y_r_dn2_carts[r_i][1] += diff_h2
        diff_p_z_r_dn2_carts[r_i][2] += diff_h2

        J2_p_x_dn2 = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn2_carts,
        )
        J2_p_y_dn2 = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn2_carts,
        )

        J2_p_z_dn2 = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn2_carts,
        )

        diff_m_x_r_dn2_carts = r_dn_carts.copy()
        diff_m_y_r_dn2_carts = r_dn_carts.copy()
        diff_m_z_r_dn2_carts = r_dn_carts.copy()
        diff_m_x_r_dn2_carts[r_i][0] -= diff_h2
        diff_m_y_r_dn2_carts[r_i][1] -= diff_h2
        diff_m_z_r_dn2_carts[r_i][2] -= diff_h2

        J2_m_x_dn2 = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn2_carts,
        )
        J2_m_y_dn2 = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn2_carts,
        )
        J2_m_z_dn2 = compute_Jastrow_two_body_jax(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn2_carts,
        )

        gradgrad_x_dn = (J2_p_x_dn2 + J2_m_x_dn2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_y_dn = (J2_p_y_dn2 + J2_m_y_dn2 - 2 * J2_ref) / (diff_h2**2)
        gradgrad_z_dn = (J2_p_z_dn2 + J2_m_z_dn2 - 2 * J2_ref) / (diff_h2**2)

        sum_laplacian_J2 += gradgrad_x_dn + gradgrad_y_dn + gradgrad_z_dn

    return grad_J2_up, grad_J2_dn, sum_laplacian_J2


# no longer used in the main code
def compute_grads_and_laplacian_Jastrow_three_body_jax(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64 | np.complex128],
    npt.NDArray[np.float64 | np.complex128],
    float | complex,
]:
    """Function for computing grads and laplacians with a given Jastrow_three_body_data.

    The method is for computing the gradients and the sum of laplacians of J3 at (r_up_carts, r_dn_carts)
    with a given Jastrow_three_body_data.

    Args:
        jastrow_three_body_data (Jastrow_three_body_data): an instance of Jastrow_three_body_data class
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        the gradients(x,y,z) of J(threebody) and the sum of laplacians of J(threebody) at (r_up_carts, r_dn_carts).
    """
    grad_J3_up, grad_J3_dn, sum_laplacian_J3 = _compute_grads_and_laplacian_Jastrow_three_body_jax(
        jastrow_three_body_data, r_up_carts, r_dn_carts
    )

    if grad_J3_up.shape != r_up_carts.shape:
        logger.error(f"grad_J3_up.shape = {grad_J3_up.shape} is inconsistent with the expected one = {r_up_carts.shape}")
        raise ValueError

    if grad_J3_dn.shape != r_dn_carts.shape:
        logger.error(f"grad_J3_dn.shape = {grad_J3_dn.shape} is inconsistent with the expected one = {r_dn_carts.shape}")
        raise ValueError

    return grad_J3_up, grad_J3_dn, sum_laplacian_J3


# no longer used in the main code
@jit
def _compute_grads_and_laplacian_Jastrow_three_body_jax(
    jastrow_three_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
]:
    """See _api method."""
    # compute grad
    grad_J3_up = grad(compute_Jastrow_three_body_jax, argnums=1)(jastrow_three_body_data, r_up_carts, r_dn_carts)

    grad_J3_dn = grad(compute_Jastrow_three_body_jax, argnums=2)(jastrow_three_body_data, r_up_carts, r_dn_carts)

    # compute laplacians
    hessian_J3_up = hessian(compute_Jastrow_three_body_jax, argnums=1)(jastrow_three_body_data, r_up_carts, r_dn_carts)
    sum_laplacian_J3_up = jnp.einsum("ijij->", hessian_J3_up)

    hessian_J3_dn = hessian(compute_Jastrow_three_body_jax, argnums=2)(jastrow_three_body_data, r_up_carts, r_dn_carts)
    sum_laplacian_J3_dn = jnp.einsum("ijij->", hessian_J3_dn)

    sum_laplacian_J3 = sum_laplacian_J3_up + sum_laplacian_J3_dn

    return grad_J3_up, grad_J3_dn, sum_laplacian_J3


# no longer used in the main code
def compute_grads_and_laplacian_Jastrow_three_body_debug(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
]:
    """See _api method."""
    diff_h = 1.0e-5

    # grad up
    grad_x_up = []
    grad_y_up = []
    grad_z_up = []
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up_carts = r_up_carts.copy()
        diff_p_y_r_up_carts = r_up_carts.copy()
        diff_p_z_r_up_carts = r_up_carts.copy()
        diff_p_x_r_up_carts[r_i][0] += diff_h
        diff_p_y_r_up_carts[r_i][1] += diff_h
        diff_p_z_r_up_carts[r_i][2] += diff_h

        J3_p_x_up = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_p_y_up = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_p_z_up = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        diff_m_x_r_up_carts = r_up_carts.copy()
        diff_m_y_r_up_carts = r_up_carts.copy()
        diff_m_z_r_up_carts = r_up_carts.copy()
        diff_m_x_r_up_carts[r_i][0] -= diff_h
        diff_m_y_r_up_carts[r_i][1] -= diff_h
        diff_m_z_r_up_carts[r_i][2] -= diff_h

        J3_m_x_up = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_y_up = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_z_up = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        grad_x_up.append((J3_p_x_up - J3_m_x_up) / (2.0 * diff_h))
        grad_y_up.append((J3_p_y_up - J3_m_y_up) / (2.0 * diff_h))
        grad_z_up.append((J3_p_z_up - J3_m_z_up) / (2.0 * diff_h))

    # grad dn
    grad_x_dn = []
    grad_y_dn = []
    grad_z_dn = []
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn_carts = r_dn_carts.copy()
        diff_p_y_r_dn_carts = r_dn_carts.copy()
        diff_p_z_r_dn_carts = r_dn_carts.copy()
        diff_p_x_r_dn_carts[r_i][0] += diff_h
        diff_p_y_r_dn_carts[r_i][1] += diff_h
        diff_p_z_r_dn_carts[r_i][2] += diff_h

        J3_p_x_dn = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn_carts,
        )
        J3_p_y_dn = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn_carts,
        )
        J3_p_z_dn = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn_carts,
        )

        diff_m_x_r_dn_carts = r_dn_carts.copy()
        diff_m_y_r_dn_carts = r_dn_carts.copy()
        diff_m_z_r_dn_carts = r_dn_carts.copy()
        diff_m_x_r_dn_carts[r_i][0] -= diff_h
        diff_m_y_r_dn_carts[r_i][1] -= diff_h
        diff_m_z_r_dn_carts[r_i][2] -= diff_h

        J3_m_x_dn = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn_carts,
        )
        J3_m_y_dn = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn_carts,
        )
        J3_m_z_dn = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn_carts,
        )

        grad_x_dn.append((J3_p_x_dn - J3_m_x_dn) / (2.0 * diff_h))
        grad_y_dn.append((J3_p_y_dn - J3_m_y_dn) / (2.0 * diff_h))
        grad_z_dn.append((J3_p_z_dn - J3_m_z_dn) / (2.0 * diff_h))

    grad_J3_up = np.array([grad_x_up, grad_y_up, grad_z_up]).T
    grad_J3_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn]).T

    # laplacian
    diff_h2 = 1.0e-3  # for laplacian

    J3_ref = compute_Jastrow_three_body_jax(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    sum_laplacian_J3 = 0.0

    # laplacians up
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up2_carts = r_up_carts.copy()
        diff_p_y_r_up2_carts = r_up_carts.copy()
        diff_p_z_r_up2_carts = r_up_carts.copy()
        diff_p_x_r_up2_carts[r_i][0] += diff_h2
        diff_p_y_r_up2_carts[r_i][1] += diff_h2
        diff_p_z_r_up2_carts[r_i][2] += diff_h2

        J3_p_x_up2 = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_p_y_up2 = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        J3_p_z_up2 = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        diff_m_x_r_up2_carts = r_up_carts.copy()
        diff_m_y_r_up2_carts = r_up_carts.copy()
        diff_m_z_r_up2_carts = r_up_carts.copy()
        diff_m_x_r_up2_carts[r_i][0] -= diff_h2
        diff_m_y_r_up2_carts[r_i][1] -= diff_h2
        diff_m_z_r_up2_carts[r_i][2] -= diff_h2

        J3_m_x_up2 = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_x_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_y_up2 = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_y_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_z_up2 = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_z_r_up2_carts,
            r_dn_carts=r_dn_carts,
        )

        gradgrad_x_up = (J3_p_x_up2 + J3_m_x_up2 - 2 * J3_ref) / (diff_h2**2)
        gradgrad_y_up = (J3_p_y_up2 + J3_m_y_up2 - 2 * J3_ref) / (diff_h2**2)
        gradgrad_z_up = (J3_p_z_up2 + J3_m_z_up2 - 2 * J3_ref) / (diff_h2**2)

        sum_laplacian_J3 += gradgrad_x_up + gradgrad_y_up + gradgrad_z_up

    # laplacians dn
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn2_carts = r_dn_carts.copy()
        diff_p_y_r_dn2_carts = r_dn_carts.copy()
        diff_p_z_r_dn2_carts = r_dn_carts.copy()
        diff_p_x_r_dn2_carts[r_i][0] += diff_h2
        diff_p_y_r_dn2_carts[r_i][1] += diff_h2
        diff_p_z_r_dn2_carts[r_i][2] += diff_h2

        J3_p_x_dn2 = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn2_carts,
        )
        J3_p_y_dn2 = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn2_carts,
        )

        J3_p_z_dn2 = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn2_carts,
        )

        diff_m_x_r_dn2_carts = r_dn_carts.copy()
        diff_m_y_r_dn2_carts = r_dn_carts.copy()
        diff_m_z_r_dn2_carts = r_dn_carts.copy()
        diff_m_x_r_dn2_carts[r_i][0] -= diff_h2
        diff_m_y_r_dn2_carts[r_i][1] -= diff_h2
        diff_m_z_r_dn2_carts[r_i][2] -= diff_h2

        J3_m_x_dn2 = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn2_carts,
        )
        J3_m_y_dn2 = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn2_carts,
        )
        J3_m_z_dn2 = compute_Jastrow_three_body_jax(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn2_carts,
        )

        gradgrad_x_dn = (J3_p_x_dn2 + J3_m_x_dn2 - 2 * J3_ref) / (diff_h2**2)
        gradgrad_y_dn = (J3_p_y_dn2 + J3_m_y_dn2 - 2 * J3_ref) / (diff_h2**2)
        gradgrad_z_dn = (J3_p_z_dn2 + J3_m_z_dn2 - 2 * J3_ref) / (diff_h2**2)

        sum_laplacian_J3 += gradgrad_x_dn + gradgrad_y_dn + gradgrad_z_dn

    return grad_J3_up, grad_J3_dn, sum_laplacian_J3


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
    from .structure import Structure_data

    jastrow_two_body_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

    num_r_up_cart_samples = 5
    num_r_dn_cart_samples = 2

    r_cart_min, r_cart_max = -3.0, 3.0

    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min

    jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=1.0)
    jastrow_two_body_debug = _compute_Jastrow_two_body_debug(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # logger.devel(f"jastrow_two_body_debug = {jastrow_two_body_debug}")

    jastrow_two_body_jax = _compute_Jastrow_two_body_jax(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # logger.devel(f"jastrow_two_body_jax = {jastrow_two_body_jax}")

    np.testing.assert_almost_equal(jastrow_two_body_debug, jastrow_two_body_jax, decimal=10)

    (
        grad_jastrow_J2_up_debug,
        grad_jastrow_J2_dn_debug,
        sum_laplacian_J2_debug,
    ) = _compute_grads_and_laplacian_Jastrow_two_body_debug(jastrow_two_body_data, r_up_carts, r_dn_carts)

    # logger.devel(f"grad_jastrow_J2_up_debug = {grad_jastrow_J2_up_debug}")
    # logger.devel(f"grad_jastrow_J2_dn_debug = {grad_jastrow_J2_dn_debug}")
    # logger.devel(f"sum_laplacian_J2_debug = {sum_laplacian_J2_debug}")

    grad_jastrow_J2_up_jax, grad_jastrow_J2_dn_jax, sum_laplacian_J2_jax = _compute_grads_and_laplacian_Jastrow_two_body_jax(
        jastrow_two_body_data,
        r_up_carts,
        r_dn_carts,
    )

    # logger.devel(f"grad_jastrow_J2_up_jax = {grad_jastrow_J2_up_jax}")
    # logger.devel(f"grad_jastrow_J2_dn_jax = {grad_jastrow_J2_dn_jax}")
    # logger.devel(f"sum_laplacian_J2_jax = {sum_laplacian_J2_jax}")

    np.testing.assert_almost_equal(grad_jastrow_J2_up_debug, grad_jastrow_J2_up_jax, decimal=5)
    np.testing.assert_almost_equal(grad_jastrow_J2_dn_debug, grad_jastrow_J2_dn_jax, decimal=5)
    np.testing.assert_almost_equal(sum_laplacian_J2_debug, sum_laplacian_J2_jax, decimal=5)

    # test MOs
    num_r_up_cart_samples = 4
    num_r_dn_cart_samples = 2
    num_R_cart_samples = 6
    num_ao = 6
    num_ao_prim = 6
    orbital_indices = [0, 1, 2, 3, 4, 5]
    exponents = [1.2, 0.5, 0.1, 0.05, 0.05, 0.05]
    coefficients = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    angular_momentums = [0, 0, 0, 1, 1, 1]
    magnetic_quantum_numbers = [0, 0, 0, 0, +1, -1]

    # generate matrices for the test
    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min

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

    j_matrix = np.random.rand(aos_data.num_ao, aos_data.num_ao + 1)

    jastrow_three_body_data = Jastrow_three_body_data(orb_data=aos_data, j_matrix=j_matrix)

    J3_debug = _compute_Jastrow_three_body_debug(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # logger.devel(f"J3_debug = {J3_debug}")

    J3_jax = _compute_Jastrow_three_body_jax(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # logger.devel(f"J3_jax = {J3_jax}")

    np.testing.assert_almost_equal(J3_debug, J3_jax, decimal=8)

    (
        grad_jastrow_J3_up_debug,
        grad_jastrow_J3_dn_debug,
        sum_laplacian_J3_debug,
    ) = _compute_grads_and_laplacian_Jastrow_three_body_debug(
        jastrow_three_body_data,
        r_up_carts,
        r_dn_carts,
    )

    # logger.devel(f"grad_jastrow_J3_up_debug = {grad_jastrow_J3_up_debug}")
    # logger.devel(f"grad_jastrow_J3_dn_debug = {grad_jastrow_J3_dn_debug}")
    # logger.devel(f"sum_laplacian_J3_debug = {sum_laplacian_J3_debug}")

    grad_jastrow_J3_up_jax, grad_jastrow_J3_dn_jax, sum_laplacian_J3_jax = _compute_grads_and_laplacian_Jastrow_three_body_jax(
        jastrow_three_body_data,
        r_up_carts,
        r_dn_carts,
    )

    # logger.devel(f"grad_jastrow_J3_up_jax = {grad_jastrow_J3_up_jax}")
    # logger.devel(f"grad_jastrow_J3_dn_jax = {grad_jastrow_J3_dn_jax}")
    # logger.devel(f"sum_laplacian_J3_jax = {sum_laplacian_J3_jax}")

    np.testing.assert_almost_equal(grad_jastrow_J3_up_debug, grad_jastrow_J3_up_jax, decimal=5)
    np.testing.assert_almost_equal(grad_jastrow_J3_dn_debug, grad_jastrow_J3_dn_jax, decimal=5)
    np.testing.assert_almost_equal(sum_laplacian_J3_debug, sum_laplacian_J3_jax, decimal=5)
    """

    # ratio
    hamiltonian_chk = "hamiltonian_data.chk"
    with open(hamiltonian_chk, "rb") as f:
        hamiltonian_data = pickle.load(f)
    geminal_data = hamiltonian_data.wavefunction_data.geminal_data
    jastrow_data = hamiltonian_data.wavefunction_data.jastrow_data

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

    _ = compute_ratio_Jastrow_part_debug(
        jastrow_data=jastrow_data,
        old_r_up_carts=old_r_up_carts,
        old_r_dn_carts=old_r_dn_carts,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )

    start = time.perf_counter()
    jastrow_ratios_debug = compute_ratio_Jastrow_part_debug(
        jastrow_data=jastrow_data,
        old_r_up_carts=old_r_up_carts,
        old_r_dn_carts=old_r_dn_carts,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )
    end = time.perf_counter()
    print(f"Elapsed Time = {(end - start) * 1e3:.3f} msec.")

    # print(jastrow_ratios_debug)

    jastrow_ratios_jax = compute_ratio_Jastrow_part_jax(
        jastrow_data=jastrow_data,
        old_r_up_carts=old_r_up_carts,
        old_r_dn_carts=old_r_dn_carts,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )
    jastrow_ratios_jax.block_until_ready()

    start = time.perf_counter()
    jastrow_ratios_jax = compute_ratio_Jastrow_part_jax(
        jastrow_data=jastrow_data,
        old_r_up_carts=old_r_up_carts,
        old_r_dn_carts=old_r_dn_carts,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )
    jastrow_ratios_jax.block_until_ready()
    end = time.perf_counter()
    print(f"Elapsed Time = {(end - start) * 1e3:.3f} msec.")

    # print(jastrow_ratios_jax)

    np.testing.assert_almost_equal(jastrow_ratios_debug, jastrow_ratios_jax, decimal=12)
