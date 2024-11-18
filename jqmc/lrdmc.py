"""LRDMC module."""

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

import logging

# python modules
import os
from logging import Formatter, StreamHandler, getLogger

# import mpi4jax
# JAX
import jax
import numpy as np
import numpy.typing as npt

# MPI
from mpi4py import MPI

# jQMC module
from .hamiltonians import Hamiltonian_data, compute_local_energy
from .jastrow_factor import Jastrow_data, Jastrow_three_body_data, Jastrow_two_body_data
from .structure import Structure_data, find_nearest_index
from .swct import SWCT_data, evaluate_swct_domega_api, evaluate_swct_omega_api
from .trexio_wrapper import read_trexio_file
from .wavefunction import Wavefunction_data, evaluate_ln_wavefunction_api, evaluate_wavefunction_api

# MPI related
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# create new logger level for development
DEVEL_LEVEL = 5
logging.addLevelName(DEVEL_LEVEL, "DEVEL")


# a new method to create a new logger
def _loglevel_devel(self, message, *args, **kwargs):
    if self.isEnabledFor(DEVEL_LEVEL):
        self._log(DEVEL_LEVEL, message, args, **kwargs)


logging.Logger.devel = _loglevel_devel

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


def generate_kinetic_mesh_debug(
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
            * evaluate_wavefunction_api(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_, r_dn_carts=r_dn_carts_)
            / evaluate_wavefunction_api(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
        )
        for r_up_carts_, r_dn_carts_ in mesh_kinetic_part
    ]

    return mesh_kinetic_part, elements_kinetic_part


if __name__ == "__main__":
    # """
    # water cc-pVTZ with Mitas ccECP (8 electrons, feasible).
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_files", "water_ccpvtz_trexio.hdf5"))
    # """

    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    # WF data
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_two_body_pade_flag=True,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    # Initialization
    r_up_carts = []
    r_dn_carts = []

    total_electrons = 0

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data.positions_cart

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

    alat = 1.0
    mesh_kinetic_part, elements_kinetic_part = generate_kinetic_mesh_debug(
        alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    print(mesh_kinetic_part, elements_kinetic_part)
