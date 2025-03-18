"""Hamiltonian module."""

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

import pickle

# python modules
from logging import Formatter, StreamHandler, getLogger

# JAX
import jax
from flax import struct
from jax import jit
from jax import typing as jnpt

from .coulomb_potential import Coulomb_potential_data, compute_coulomb_potential_jax
from .structure import Structure_data
from .wavefunction import (
    Wavefunction_data,
    Wavefunction_data_deriv_params,
    Wavefunction_data_deriv_R,
    compute_kinetic_energy_jax,
)

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)

# separator
num_sep_line = 66


@struct.dataclass
class Hamiltonian_data:
    """Hamiltonian dataclass.

    The class contains data for computing Kinetic and Potential energy terms.

    Args:
        structure_data (Structure_data): an instance of Structure_data
        coulomb_data (Coulomb_data): an instance of Coulomb_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data

    Notes:
        Heres are the differentiable arguments, i.e., pytree_node = True
        This information is a little bit tricky in terms of a principle of the object-oriented programming,
        'Don't ask, but tell' (i.e., the Hamiltonian_data knows the details of the other classes
        too much), but there is no other choice to dynamically switch on and off pytree_nodes depending
        on optimized variational parameters chosen by a user because @dataclass is statistically generated.

        WF parameters related:
            - lambda in wavefunction_data.geminal_data (determinant.py)
            - jastrow_2b_param in wavefunction_data.jastrow_data.jastrow_two_body_data (jastrow_factor.py)
            - j_matrix in wavefunction_data.jastrow_data.jastrow_three_body_data (jastrow_factor.py)

        Atomic positions related:
            - positions in hamiltonian_data.structure_data (this file)
            - positions in wavefunction_data.geminal_data.mos_data/aos_data.structure_data (molecular_orbital.py/atomic_orbital.py)
            - positions in wavefunction_data.jastrow_data.jastrow_three_body_data.mos_data/aos_data.structure_data (jastrow_factor.py)
            - positions in Coulomb_potential_data.structure_data (coulomb_potential.py)
    """

    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    coulomb_potential_data: Coulomb_potential_data = struct.field(
        pytree_node=True, default_factory=lambda: Coulomb_potential_data()
    )
    wavefunction_data: Wavefunction_data = struct.field(pytree_node=True, default_factory=lambda: Wavefunction_data())

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        self.structure_data.sanity_check()
        self.coulomb_potential_data.sanity_check()
        self.wavefunction_data.sanity_check()

    def get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        # Add the top separator line.
        info_lines.append("=" * num_sep_line)
        # Replace attribute logger_info() calls with their get_info() outputs.
        info_lines.extend(self.structure_data.get_info())
        info_lines.extend(self.coulomb_potential_data.get_info())
        info_lines.extend(self.wavefunction_data.get_info())
        # Add the bottom separator line.
        info_lines.append("=" * num_sep_line)
        return info_lines

    def logger_info(self) -> None:
        """Log the information from get_info() using logger.info."""
        for line in self.get_info():
            logger.info(line)

    def dump(self, filepath="jqmc.chk") -> None:
        """_dump Hamiltonian data as a binary file.

        Args:
            filepath (str, optional): file path
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath="jqmc.chk") -> "Hamiltonian_data":
        """Read Hamiltonian data from a binary file.

        Args:
            filepath (str, optional): file path

        Returns:
            Hamiltonian_data: An instance of Hamiltonian_data.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)


@struct.dataclass
class Hamiltonian_data_deriv_params(Hamiltonian_data):
    """See Hamiltonian_data."""

    structure_data: Structure_data = struct.field(pytree_node=False, default_factory=lambda: Structure_data())
    coulomb_potential_data: Coulomb_potential_data = struct.field(
        pytree_node=False, default_factory=lambda: Coulomb_potential_data()
    )
    wavefunction_data: Wavefunction_data = struct.field(pytree_node=True, default_factory=lambda: Wavefunction_data())

    @classmethod
    def from_base(cls, hamiltonian_data: Hamiltonian_data):
        """Switch pytree_node."""
        structure_data = hamiltonian_data.structure_data
        coulomb_potential_data = hamiltonian_data.coulomb_potential_data
        wavefunction_data = Wavefunction_data_deriv_params.from_base(hamiltonian_data.wavefunction_data)

        return cls(
            structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wavefunction_data
        )


@struct.dataclass
class Hamiltonian_data_deriv_R(Hamiltonian_data):
    """See Hamiltonian_data."""

    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    coulomb_potential_data: Coulomb_potential_data = struct.field(
        pytree_node=True, default_factory=lambda: Coulomb_potential_data()
    )
    wavefunction_data: Wavefunction_data = struct.field(pytree_node=True, default_factory=lambda: Wavefunction_data())

    @classmethod
    def from_base(cls, hamiltonian_data: Hamiltonian_data):
        """Switch pytree_node."""
        structure_data = hamiltonian_data.structure_data
        coulomb_potential_data = hamiltonian_data.coulomb_potential_data
        wavefunction_data = Wavefunction_data_deriv_R.from_base(hamiltonian_data.wavefunction_data)

        return cls(
            structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wavefunction_data
        )


@jit
def compute_local_energy_jax(
    hamiltonian_data: Hamiltonian_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> float:
    """Compute Local Energy.

    The method is for computing the local energy at (r_up_carts, r_dn_carts).

    Args:
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
        r_up_carts (jnpt.ArrayLike): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        float: The value of local energy (e_L) with the given wavefunction (float)
    """
    T = compute_kinetic_energy_jax(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    V = compute_coulomb_potential_jax(
        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        wavefunction_data=hamiltonian_data.wavefunction_data,
    )

    return T + V


if __name__ == "__main__":
    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
