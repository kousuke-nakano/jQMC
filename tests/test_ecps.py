"""collections of unit tests."""

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

import os
from logging import Formatter, StreamHandler, getLogger

import jax
import numpy as np

from ..jqmc.coulomb_potential import (
    _compute_bare_coulomb_potential_debug,
    _compute_bare_coulomb_potential_jax,
    _compute_ecp_coulomb_potential_debug,
    _compute_ecp_coulomb_potential_jax,
    _compute_ecp_local_parts_full_NN_debug,
    _compute_ecp_local_parts_full_NN_jax,
    _compute_ecp_non_local_parts_full_NN_debug,
    _compute_ecp_non_local_parts_full_NN_jax,
    _compute_ecp_non_local_parts_NN_debug,
    _compute_ecp_non_local_parts_NN_jax,
)
from ..jqmc.jastrow_factor import (
    Jastrow_data,
)
from ..jqmc.trexio_wrapper import read_trexio_file
from ..jqmc.wavefunction import Wavefunction_data

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

log = getLogger("myqmc")
log.setLevel("DEBUG")
stream_handler = StreamHandler()
stream_handler.setLevel("DEBUG")
handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
stream_handler.setFormatter(handler_format)
log.addHandler(stream_handler)


def test_debug_and_jax_bare_coulomb():
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"))

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=None,
        jastrow_two_body_pade_flag=False,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )  # no jastrow for the time-being.

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    old_r_up_carts = np.array(
        [
            [0.64878536, -0.83275288, 0.33532629],
            [0.55271273, 0.72310605, 0.93443775],
            [0.66767275, 0.1206456, -0.36521208],
            [-0.93165236, -0.0120386, 0.33003036],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [1.0347816, 1.26162081, 0.42301735],
            [-0.57843435, 1.03651987, -0.55091542],
            [-1.56091964, -0.58952149, -0.99268141],
            [0.61863233, -0.14903326, 0.51962683],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_dn_carts[3] = [0.618632327645002, -0.149033260668010, 0.131889254514777]

    # bare coulomb
    vpot_bare_jax = _compute_bare_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = _compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    # print(f"vpot_bare_jax = {vpot_bare_jax}")
    # print(f"vpot_bare_debug = {vpot_bare_debug}")
    np.testing.assert_almost_equal(vpot_bare_jax, vpot_bare_debug, decimal=10)


def test_debug_and_jax_ecp_local():
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"))

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=None,
        jastrow_two_body_pade_flag=False,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )  # no jastrow for the time-being.

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    old_r_up_carts = np.array(
        [
            [0.64878536, -0.83275288, 0.33532629],
            [0.55271273, 0.72310605, 0.93443775],
            [0.66767275, 0.1206456, -0.36521208],
            [-0.93165236, -0.0120386, 0.33003036],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [1.0347816, 1.26162081, 0.42301735],
            [-0.57843435, 1.03651987, -0.55091542],
            [-1.56091964, -0.58952149, -0.99268141],
            [0.61863233, -0.14903326, 0.51962683],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_dn_carts[3] = [0.618632327645002, -0.149033260668010, 0.131889254514777]

    # ecp local
    vpot_ecp_local_full_NN_jax = _compute_ecp_local_parts_full_NN_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_ecp_local_full_NN_debug = _compute_ecp_local_parts_full_NN_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    np.testing.assert_almost_equal(vpot_ecp_local_full_NN_jax, vpot_ecp_local_full_NN_debug, decimal=10)


def test_debug_and_jax_ecp_non_local():
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"))

    # n_atom
    n_atom = structure_data.natom

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=None,
        jastrow_two_body_pade_flag=False,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )  # no jastrow for the time-being.

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    old_r_up_carts = np.array(
        [
            [0.64878536, -0.83275288, 0.33532629],
            [0.55271273, 0.72310605, 0.93443775],
            [0.66767275, 0.1206456, -0.36521208],
            [-0.93165236, -0.0120386, 0.33003036],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [1.0347816, 1.26162081, 0.42301735],
            [-0.57843435, 1.03651987, -0.55091542],
            [-1.56091964, -0.58952149, -0.99268141],
            [0.61863233, -0.14903326, 0.51962683],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_dn_carts[3] = [0.618632327645002, -0.149033260668010, 0.131889254514777]

    # ecp non-local (full_NN)
    (
        mesh_non_local_ecp_part_r_up_carts_full_NN_jax,
        mesh_non_local_ecp_part_r_dn_carts_full_NN_jax,
        V_nonlocal_full_NN_jax,
        sum_V_nonlocal_full_NN_jax,
    ) = _compute_ecp_non_local_parts_full_NN_jax(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    (
        mesh_non_local_ecp_part_r_up_carts_full_NN_debug,
        mesh_non_local_ecp_part_r_dn_carts_full_NN_debug,
        V_nonlocal_full_NN_debug,
        sum_V_nonlocal_full_NN_debug,
    ) = _compute_ecp_non_local_parts_full_NN_debug(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    np.testing.assert_almost_equal(sum_V_nonlocal_full_NN_debug, sum_V_nonlocal_full_NN_jax, decimal=6)

    mesh_non_local_r_up_carts_max_full_NN_jax = mesh_non_local_ecp_part_r_up_carts_full_NN_jax[
        np.argmax(V_nonlocal_full_NN_jax)
    ]
    mesh_non_local_r_up_carts_max_full_NN_debug = mesh_non_local_ecp_part_r_up_carts_full_NN_debug[
        np.argmax(V_nonlocal_full_NN_debug)
    ]
    mesh_non_local_r_dn_carts_max_full_NN_jax = mesh_non_local_ecp_part_r_dn_carts_full_NN_jax[
        np.argmax(V_nonlocal_full_NN_jax)
    ]
    mesh_non_local_r_dn_carts_max_full_NN_debug = mesh_non_local_ecp_part_r_dn_carts_full_NN_debug[
        np.argmax(V_nonlocal_full_NN_debug)
    ]
    V_ecp_non_local_max_full_NN_jax = V_nonlocal_full_NN_jax[np.argmax(V_nonlocal_full_NN_jax)]
    V_ecp_non_local_max_full_NN_debug = V_nonlocal_full_NN_debug[np.argmax(V_nonlocal_full_NN_debug)]

    np.testing.assert_almost_equal(V_ecp_non_local_max_full_NN_jax, V_ecp_non_local_max_full_NN_debug, decimal=6)
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_up_carts_max_full_NN_jax, mesh_non_local_r_up_carts_max_full_NN_debug, decimal=6
    )
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_dn_carts_max_full_NN_jax, mesh_non_local_r_dn_carts_max_full_NN_debug, decimal=6
    )

    # ecp non-local (NN, N=max)
    (
        mesh_non_local_ecp_part_r_up_carts_NN_check_jax,
        mesh_non_local_ecp_part_r_dn_carts_NN_check_jax,
        V_nonlocal_NN_check_jax,
        sum_V_nonlocal_NN_check_jax,
    ) = _compute_ecp_non_local_parts_NN_jax(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        NN=n_atom,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    (
        mesh_non_local_ecp_part_r_up_carts_NN_check_debug,
        mesh_non_local_ecp_part_r_dn_carts_NN_check_debug,
        V_nonlocal_NN_check_debug,
        sum_V_nonlocal_NN_check_debug,
    ) = _compute_ecp_non_local_parts_NN_debug(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        NN=n_atom,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    # debug, full-NN vs check-NN
    np.testing.assert_almost_equal(sum_V_nonlocal_full_NN_debug, sum_V_nonlocal_NN_check_debug, decimal=6)

    # jax, full-NN vs check-NN
    np.testing.assert_almost_equal(sum_V_nonlocal_full_NN_jax, sum_V_nonlocal_NN_check_jax, decimal=6)

    mesh_non_local_r_up_carts_max_NN_check_jax = mesh_non_local_ecp_part_r_up_carts_NN_check_jax[
        np.argmax(V_nonlocal_NN_check_jax)
    ]
    mesh_non_local_r_up_carts_max_NN_check_debug = mesh_non_local_ecp_part_r_up_carts_NN_check_debug[
        np.argmax(V_nonlocal_NN_check_debug)
    ]
    mesh_non_local_r_dn_carts_max_NN_check_jax = mesh_non_local_ecp_part_r_dn_carts_NN_check_jax[
        np.argmax(V_nonlocal_NN_check_jax)
    ]
    mesh_non_local_r_dn_carts_max_NN_check_debug = mesh_non_local_ecp_part_r_dn_carts_NN_check_debug[
        np.argmax(V_nonlocal_NN_check_debug)
    ]
    V_ecp_non_local_max_NN_check_jax = V_nonlocal_NN_check_jax[np.argmax(V_nonlocal_NN_check_jax)]
    V_ecp_non_local_max_NN_check_debug = V_nonlocal_NN_check_debug[np.argmax(V_nonlocal_NN_check_debug)]

    # debug, full-NN vs check-NN
    np.testing.assert_almost_equal(V_ecp_non_local_max_full_NN_debug, V_ecp_non_local_max_NN_check_debug, decimal=6)
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_up_carts_max_full_NN_debug, mesh_non_local_r_up_carts_max_NN_check_debug, decimal=6
    )
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_dn_carts_max_full_NN_debug, mesh_non_local_r_dn_carts_max_NN_check_debug, decimal=6
    )

    # jax, full-NN vs check-NN
    np.testing.assert_almost_equal(V_ecp_non_local_max_full_NN_jax, V_ecp_non_local_max_NN_check_jax, decimal=6)
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_up_carts_max_full_NN_jax, mesh_non_local_r_up_carts_max_NN_check_jax, decimal=6
    )
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_dn_carts_max_full_NN_jax, mesh_non_local_r_dn_carts_max_NN_check_jax, decimal=6
    )

    # ecp non-local (NN, N=1[default])
    (
        mesh_non_local_ecp_part_r_up_carts_NN_jax,
        mesh_non_local_ecp_part_r_dn_carts_NN_jax,
        V_nonlocal_NN_jax,
        sum_V_nonlocal_NN_jax,
    ) = _compute_ecp_non_local_parts_NN_jax(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    (
        mesh_non_local_ecp_part_r_up_carts_NN_debug,
        mesh_non_local_ecp_part_r_dn_carts_NN_debug,
        V_nonlocal_NN_debug,
        sum_V_nonlocal_NN_debug,
    ) = _compute_ecp_non_local_parts_NN_debug(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    np.testing.assert_almost_equal(sum_V_nonlocal_NN_debug, sum_V_nonlocal_NN_jax, decimal=6)

    mesh_non_local_r_up_carts_max_NN_jax = mesh_non_local_ecp_part_r_up_carts_NN_jax[np.argmax(V_nonlocal_NN_jax)]
    mesh_non_local_r_up_carts_max_NN_debug = mesh_non_local_ecp_part_r_up_carts_NN_debug[np.argmax(V_nonlocal_NN_debug)]
    mesh_non_local_r_dn_carts_max_NN_jax = mesh_non_local_ecp_part_r_dn_carts_NN_jax[np.argmax(V_nonlocal_NN_jax)]
    mesh_non_local_r_dn_carts_max_NN_debug = mesh_non_local_ecp_part_r_dn_carts_NN_debug[np.argmax(V_nonlocal_NN_debug)]
    V_ecp_non_local_max_NN_jax = V_nonlocal_NN_jax[np.argmax(V_nonlocal_NN_jax)]
    V_ecp_non_local_max_NN_debug = V_nonlocal_NN_debug[np.argmax(V_nonlocal_NN_debug)]

    np.testing.assert_almost_equal(V_ecp_non_local_max_NN_jax, V_ecp_non_local_max_NN_debug, decimal=6)
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_up_carts_max_NN_jax, mesh_non_local_r_up_carts_max_NN_debug, decimal=6
    )
    np.testing.assert_array_almost_equal(
        mesh_non_local_r_dn_carts_max_NN_jax, mesh_non_local_r_dn_carts_max_NN_debug, decimal=6
    )


"""
def test_debug_and_jax_ecp_total():
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"))

    # define data
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=None,
        jastrow_two_body_pade_flag=False,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )  # no jastrow for the time-being.

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    old_r_up_carts = np.array(
        [
            [0.64878536, -0.83275288, 0.33532629],
            [0.55271273, 0.72310605, 0.93443775],
            [0.66767275, 0.1206456, -0.36521208],
            [-0.93165236, -0.0120386, 0.33003036],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [1.0347816, 1.26162081, 0.42301735],
            [-0.57843435, 1.03651987, -0.55091542],
            [-1.56091964, -0.58952149, -0.99268141],
            [0.61863233, -0.14903326, 0.51962683],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_dn_carts[3] = [0.618632327645002, -0.149033260668010, 0.131889254514777]

    # ecp total
    vpot_ecp_jax = _compute_ecp_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    vpot_ecp_debug = _compute_ecp_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    # print(f"vpot_ecp_jax = {vpot_ecp_jax}")
    # print(f"vpot_ecp_debug = {vpot_ecp_debug}")
    np.testing.assert_almost_equal(vpot_ecp_jax, vpot_ecp_debug, decimal=10)
"""
