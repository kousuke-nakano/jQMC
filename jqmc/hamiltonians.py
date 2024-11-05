"""Hamiltonian module."""

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

# python modules
import time
from logging import Formatter, StreamHandler, getLogger

# JAX
import jax
import numpy as np
import numpy.typing as npt
from flax import struct

from .coulomb_potential import Coulomb_potential_data, compute_coulomb_potential_api
from .structure import Structure_data
from .wavefunction import Wavefunction_data, compute_kinetic_energy_api

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


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

        WF parameters related:
            - mo_coefficients in MOs_data (molecular_orbital.py) (switched off, for the time being)
            - (ao_)coefficients in AOs_data (atomic_orbital.py) (switched off, for the time being)
            - (ao_)exponents in AOs_data (atomic_orbital.py) (switched off, for the time being)
            - lambda_matrix in Geminal_data (determinant.py) (switched off, for the time being)
            - jastrow_2b_param in Jastrow_two_body_data (jastrow_factor.py)
            - j_matrix in Jastrow_three_body_data (jastrow_factor.py)

        Atomic positions related:
            - structure_data.positions in AOs_data (atomic_orbital.py)
            - structure_data.positions in Coulomb_potential_data (coulomb_potential.py)
    """

    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    coulomb_potential_data: Coulomb_potential_data = struct.field(
        pytree_node=True, default_factory=lambda: Coulomb_potential_data()
    )
    wavefunction_data: Wavefunction_data = struct.field(pytree_node=True, default_factory=lambda: Wavefunction_data())

    def __post_init__(self) -> None:
        """Initialization of the class.

        This magic function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.

        Todo:
            To be implemented.
        """
        pass


def compute_local_energy(
    hamiltonian_data: Hamiltonian_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float | complex:
    """Compute Local Energy.

    The method is for computing the local energy at (r_up_carts, r_dn_carts).

    Args:
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        float: The value of local energy (e_L) with the given wavefunction (float)
    """
    start = time.perf_counter()
    T = compute_kinetic_energy_api(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    end = time.perf_counter()
    logger.debug(f"Kinetic part in e_L: Time = {(end-start)*1000:.3f} msec.")

    start = time.perf_counter()
    V = compute_coulomb_potential_api(
        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        wavefunction_data=hamiltonian_data.wavefunction_data,
    )
    end = time.perf_counter()
    logger.debug(f"Coulomb Potential part in e_L: Time = {(end-start)*100:.3f} msec.")

    logger.debug(f"e_L = {T+V} Ha")

    return T + V


if __name__ == "__main__":
    log = getLogger("jqmc")
    log.setLevel("DEBUG")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    log.addHandler(stream_handler)
