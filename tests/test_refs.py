"""collections of unit tests."""

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

import os
import pickle
from logging import Formatter, StreamHandler, getLogger

import jax
import numpy as np
import pytest

from ..jqmc.coulomb_potential import (
    _compute_bare_coulomb_potential_debug,
    _compute_bare_coulomb_potential_jax,
    _compute_ecp_coulomb_potential_debug,
    _compute_ecp_coulomb_potential_jax,
)
from ..jqmc.hamiltonians import Hamiltonian_data
from ..jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from ..jqmc.trexio_wrapper import read_trexio_file
from ..jqmc.wavefunction import Wavefunction_data, compute_kinetic_energy_api, evaluate_wavefunction_api

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


@pytest.mark.activate_if_disable_jit
def test_comparison_with_TurboRVB_wo_Jastrow(request):
    if not request.config.getoption("--disable-jit"):
        pytest.skip(reason="Bug of flux.struct with @jit.")

    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"))

    jastrow_data = Jastrow_data(
        jastrow_two_body_data=None,
        jastrow_two_body_flag=False,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    old_r_up_carts = np.array(
        [
            [-1.13450385875760, -0.698914730480577, -6.290951981744008e-003],
            [-2.07761893946839, 1.30902541938751, -5.220902114745041e-002],
            [0.276215481293413, 0.422863618938476, 0.279866487253010],
            [-1.60902246286275, 0.499927465264998, 0.700105816369930],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [-1.48583455555933, -1.01189391902775, 1.83998639430367],
            [0.635659512640246, 0.398999201990364, -0.745191606127732],
            [-2.00590358216444, 1.90796788491204, -0.195294104680795],
            [-1.12726250654165, -0.739542218156325, -4.817447678670805e-002],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_up_carts[2] = [0.276215481293413, -0.270740090536313, 0.279866487253010]

    WF_ratio_ref_turborvb = 0.919592366177398
    kinc_ref_turborvb = 14.6961809427008
    vpot_ref_turborvb = -17.0152290468758
    vpotoff_ref_turborvb = 0.329197252921614

    # print(f"wf_ratio_ref={WF_ratio_ref_turborvb} Ha")
    # print(f"kinc_ref={kinc_ref_turborvb} Ha")
    # print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")

    WF_ratio = (
        evaluate_wavefunction_api(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction_api(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy_api(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = _compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = _compute_bare_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=6)

    vpot_ecp_debug = _compute_ecp_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    vpot_ecp_jax = _compute_ecp_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax+vpot_ecp_debug} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=8)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3)
    np.testing.assert_almost_equal(vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3)

    jax.clear_caches()


@pytest.mark.activate_if_disable_jit
def test_comparison_with_TurboRVB_w_2b_Jastrow(request):
    if not request.config.getoption("--disable-jit"):
        pytest.skip(reason="Bug of flux.struct with @jit.")

    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"))

    turborvb_2b_param = 0.896342988526927  # -6 !!
    jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=turborvb_2b_param)

    jastrow_data = Jastrow_data(
        jastrow_two_body_data=jastrow_two_body_data,
        jastrow_two_body_flag=True,
        jastrow_three_body_data=None,
        jastrow_three_body_flag=False,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    old_r_up_carts = np.array(
        [
            [-1.13450385875760, -0.698914730480577, -6.290951981744008e-003],
            [-2.30366220171161, 1.47326376760292, 0.126403765463162],
            [0.276215481293413, 0.422863618938476, 0.279866487253010],
            [-1.60902246286275, 0.499927465264998, 0.700105816369930],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [-1.42343008909407, -1.13669461924113, 0.525171318204107],
            [0.635659512640246, 0.398999201990364, -0.745191606127732],
            [-2.00590358216444, 1.90796788491204, -0.195294104680795],
            [-1.12726250654165, -0.678049640381367, -0.656537799033216],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_up_carts[2] = [0.276215481293413, -0.270740090536313, 0.279866487253010]

    WF_ratio_ref_turborvb = 0.872631278217550
    kinc_ref_turborvb = 13.5310405254930
    vpot_ref_turborvb = -30.1945862173100
    vpotoff_ref_turborvb = 0.250461990878211

    # print(f"wf_ratio_ref={WF_ratio_ref_turborvb} Ha")
    # print(f"kinc_ref={kinc_ref_turborvb} Ha")
    # print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")

    WF_ratio = (
        evaluate_wavefunction_api(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction_api(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy_api(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = _compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = _compute_bare_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=6)

    vpot_ecp_debug = _compute_ecp_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    vpot_ecp_jax = _compute_ecp_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax+vpot_ecp_debug} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=8)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3)
    np.testing.assert_almost_equal(vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3)

    jax.clear_caches()


jax.clear_caches()


@pytest.mark.activate_if_disable_jit
def test_comparison_with_TurboRVB_w_2b_3b_Jastrow(request):
    if not request.config.getoption("--disable-jit"):
        pytest.skip(reason="Bug of flux.struct with @jit.")
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"))

    with open(
        os.path.join(os.path.dirname(__file__), "trexio_example_files", "jastrow_data_w_2b_3b.pkl"),
        "rb",
    ) as f:
        jastrow_data = pickle.load(f)

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    old_r_up_carts = np.array(
        [
            [-1.13450385875760, -0.698914730480577, -6.290951981744008e-003],
            [-2.30366220171161, 2.32528986358581, -0.200085136796780],
            [0.390190526911041, 0.422863618938476, 1.09811717761730],
            [-2.40143573560450, 0.623761374394509, 0.700105816369930],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [-1.58454340030273, -1.01943210665261, 2.47097269788962],
            [1.90701925586575, 0.398999201990364, -0.745191606127732],
            [-2.00590358216444, 2.31787632191030, -0.195294104680795],
            [-0.103689059569662, -2.18500664943652, -0.318874284614467],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_up_carts[2] = [0.390190526911041, -0.270740090536313, 1.09811717761730]

    WF_ratio_ref_turborvb = 0.867706478518192
    kinc_ref_turborvb = 5.11234708991921
    vpot_ref_turborvb = -17.0140133127848
    vpotoff_ref_turborvb = 0.275054565511106

    print(f"wf_ratio_ref={WF_ratio_ref_turborvb} Ha")
    print(f"kinc_ref={kinc_ref_turborvb} Ha")
    print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")

    WF_ratio = (
        evaluate_wavefunction_api(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction_api(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy_api(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = _compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = _compute_bare_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=6)

    vpot_ecp_debug = _compute_ecp_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    vpot_ecp_jax = _compute_ecp_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    print(f"wf_ratio={WF_ratio} Ha")
    print(f"kinc={kinc} Ha")
    print(f"vpot={vpot_bare_jax + vpot_ecp_debug} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=8)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3)
    np.testing.assert_almost_equal(vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3)

    jax.clear_caches()


@pytest.mark.activate_if_disable_jit
def test_comparison_with_TurboRVB_w_2b_1b3b_Jastrow(request):
    if not request.config.getoption("--disable-jit"):
        pytest.skip(reason="Bug of flux.struct with @jit.")
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_trexio.hdf5"))

    with open(
        os.path.join(os.path.dirname(__file__), "trexio_example_files", "jastrow_data_w_2b_1b3b.pkl"),
        "rb",
    ) as f:
        jastrow_data = pickle.load(f)

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    old_r_up_carts = np.array(
        [
            [-1.45953855349650, -0.862585479538573, -6.290951981744008e-003],
            [-0.332901524462574, 0.626165379953289, -0.603559493748950],
            [-0.197062006804461, 0.371833444736005, 0.439075235222144],
            [-1.83684814645671, -8.976990228515924e-002, -2.462312627037627e-002],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [-5.347744437064250e-002, 0.623781376578920, 0.525171318204107],
            [-2.19220906931126, -0.310636827543933, 5.967026994100055e-002],
            [-1.81960258882794, 0.517427457629536, -0.195294104680795],
            [-1.12726250654165, -0.260900727811469, -1.45214401542009],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_up_carts[1] = [-0.150727555030462, 0.626165379953289, -0.603559493748950]

    WF_ratio_ref_turborvb = 0.745878160412662
    kinc_ref_turborvb = 12.2446576962106
    vpot_ref_turborvb = -29.6412525272157
    vpotoff_ref_turborvb = 0.995316391222278

    # print(f"wf_ratio_ref={WF_ratio_ref_turborvb} Ha")
    # print(f"kinc_ref={kinc_ref_turborvb} Ha")
    # print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")

    WF_ratio = (
        evaluate_wavefunction_api(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction_api(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy_api(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = _compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = _compute_bare_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=6)

    vpot_ecp_debug = _compute_ecp_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    vpot_ecp_jax = _compute_ecp_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax+vpot_ecp_debug} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=8)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=2)
    np.testing.assert_almost_equal(vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=2)

    jax.clear_caches()
