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

import os
import pickle

import jax
import numpy as np
import pytest

from ..jqmc.coulomb_potential import (
    compute_bare_coulomb_potential_debug,
    compute_bare_coulomb_potential_jax,
    compute_ecp_coulomb_potential_debug,
    compute_ecp_coulomb_potential_jax,
)
from ..jqmc.hamiltonians import Hamiltonian_data
from ..jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from ..jqmc.trexio_wrapper import read_trexio_file
from ..jqmc.wavefunction import Wavefunction_data, compute_kinetic_energy_jax, evaluate_wavefunction_jax

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def test_comparison_with_TurboRVB_wo_Jastrow_w_ecp():
    """Test comparison with the corresponding ECP TurboRVB calculation without Jastrow factor."""
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=None,
    )
    jastrow_data.sanity_check()

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    wavefunction_data.sanity_check()

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    hamiltonian_data.sanity_check()

    old_r_up_carts = np.array(
        [
            [-1.1345038587576, -0.698914730480577, -0.006290951981744008],
            [-2.07761893946839, 1.30902541938751, -0.05220902114745041],
            [0.276215481293413, 0.422863618938476, 0.27986648725301],
            [-1.60902246286275, 0.499927465264998, 0.70010581636993],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [-1.48583455555933, -1.01189391902775, 1.83998639430367],
            [0.635659512640246, 0.398999201990364, -0.745191606127732],
            [-2.00590358216444, 1.90796788491204, -0.195294104680795],
            [-1.12726250654165, -0.739542218156325, -0.04817447678670805],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_up_carts[2] = [0.276215481293413, -0.270740090536313, 0.27986648725301]

    WF_ratio_ref_turborvb = 0.919592366177397
    kinc_ref_turborvb = 14.6961809426982
    vpot_ref_turborvb = -17.0152290468758
    vpotoff_ref_turborvb = 0.329197252921634
    # print(f"wf_ratio_ref={WF_ratio_ref_turborvb} Ha")
    # print(f"kinc_ref={kinc_ref_turborvb} Ha")
    # print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")

    WF_ratio = (
        evaluate_wavefunction_jax(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction_jax(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy_jax(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = compute_bare_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=6)

    vpot_ecp_debug = compute_ecp_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    vpot_ecp_jax = compute_ecp_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax + vpot_ecp_jax} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3)
    np.testing.assert_almost_equal(vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3)

    jax.clear_caches()


def test_comparison_with_TurboRVB_w_2b_Jastrow_w_ecp():
    """Test comparison with the corresponding ECP TurboRVB calculation with 2b Jastrow factor."""
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    turborvb_2b_param = 0.676718854150191  # -5 !!
    jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=turborvb_2b_param)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=jastrow_two_body_data,
        jastrow_three_body_data=None,
    )
    jastrow_data.sanity_check()

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    wavefunction_data.sanity_check()

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    hamiltonian_data.sanity_check()

    old_r_up_carts = np.array(
        [
            [-1.1345038587576, -0.698914730480577, -0.006290951981744008],
            [-2.30366220171161, 1.47326376760292, 0.126403765463162],
            [0.276215481293413, 0.422863618938476, 0.27986648725301],
            [-2.54518559687882, 0.822753144911055, 0.70010581636993],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [-1.42343008909407, -1.13669461924113, 0.525171318204107],
            [1.90701925586575, 0.398999201990364, -0.745191606127732],
            [-2.00590358216444, 1.90796788491204, -0.195294104680795],
            [-1.12726250654165, -0.678049640381367, -0.656537799033216],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_up_carts[2] = [0.276215481293413, -0.270740090536313, 0.27986648725301]

    WF_ratio_ref_turborvb = 0.881124604511419
    kinc_ref_turborvb = 11.1237599317225
    vpot_ref_turborvb = -27.03387193107
    vpotoff_ref_turborvb = 0.244575316335042

    # print(f"wf_ratio_ref={WF_ratio_ref_turborvb} Ha")
    # print(f"kinc_ref={kinc_ref_turborvb} Ha")
    # print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")

    WF_ratio = (
        evaluate_wavefunction_jax(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction_jax(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy_jax(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = compute_bare_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=6)

    vpot_ecp_debug = compute_ecp_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    vpot_ecp_jax = compute_ecp_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax+vpot_ecp_debug} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3)
    np.testing.assert_almost_equal(vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3)

    jax.clear_caches()


def test_comparison_with_TurboRVB_w_2b_3b_Jastrow_w_ecp():
    """Test comparison with the corresponding ECP TurboRVB calculation with 2b,3b Jastrow factor."""
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    with open(
        os.path.join(os.path.dirname(__file__), "trexio_example_files", "jastrow_data_w_2b_3b_w_ecp.pkl"),
        "rb",
    ) as f:
        jastrow_data = pickle.load(f)
        jastrow_data.sanity_check()

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    wavefunction_data.sanity_check()

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    hamiltonian_data.sanity_check()

    old_r_up_carts = np.array(
        [
            [-1.1345038587576, -0.698914730480577, -0.006290951981744008],
            [-2.30366220171161, 2.32528986358581, -0.20008513679678],
            [0.390190526911041, 0.422863618938476, 1.0981171776173],
            [-2.4014357356045, 0.623761374394509, 0.70010581636993],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [-1.58454340030273, -1.01943210665261, 0.37014437052153],
            [1.90701925586575, 0.398999201990364, -0.745191606127732],
            [-2.00590358216444, 2.3178763219103, -0.195294104680795],
            [-0.103689059569662, -2.18500664943652, -1.56814885512335],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_up_carts[2] = [0.390190526911041, -0.270740090536313, 1.0981171776173]

    WF_ratio_ref_turborvb = 0.858468162763939
    kinc_ref_turborvb = 5.82890200054949
    vpot_ref_turborvb = -19.1676316230828
    vpotoff_ref_turborvb = 0.285186134621918

    # print(f"wf_ratio_ref={WF_ratio_ref_turborvb} Ha")
    # print(f"kinc_ref={kinc_ref_turborvb} Ha")
    # print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")

    WF_ratio = (
        evaluate_wavefunction_jax(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction_jax(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy_jax(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = compute_bare_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=6)

    vpot_ecp_debug = compute_ecp_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    vpot_ecp_jax = compute_ecp_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax + vpot_ecp_debug} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3)
    np.testing.assert_almost_equal(vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=3)

    jax.clear_caches()


def test_comparison_with_TurboRVB_w_2b_1b3b_Jastrow_w_ecp():
    """Test comparison with the corresponding ECP TurboRVB calculation with 2b,1b3b Jastrow factor."""
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"), store_tuple=True
    )

    with open(
        os.path.join(os.path.dirname(__file__), "trexio_example_files", "jastrow_data_w_2b_1b3b_w_ecp.pkl"),
        "rb",
    ) as f:
        jastrow_data = pickle.load(f)
        jastrow_data.sanity_check()

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    wavefunction_data.sanity_check()

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    hamiltonian_data.sanity_check()

    old_r_up_carts = np.array(
        [
            [-2.02906771233089, -0.726280132104733, -0.006290951981744008],
            [-0.332901524462574, 0.626165379953289, -0.60355949374895],
            [-0.197062006804461, -0.396462287261025, 0.207245244485559],
            [-2.13232697453793, 2.02938760506611, 0.626121128343523],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [-2.27723556201111, -0.226423326809174, 0.525171318204107],
            [0.635659512640246, -0.128318768826431, -0.479396452798511],
            [-2.00590358216444, 1.90796788491204, -0.195294104680795],
            [-1.12726250654165, -0.739542218156325, -0.25704043697001],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_dn_carts[0] = [-2.27723556201111, 0.7469747620327, 0.525171318204107]

    WF_ratio_ref_turborvb = 0.268078593287622
    kinc_ref_turborvb = 9.84051921791642
    vpot_ref_turborvb = -27.1676371839677
    vpotoff_ref_turborvb = 0.02700582402227284

    # print(f"wf_ratio_ref={WF_ratio_ref_turborvb} Ha")
    # print(f"kinc_ref={kinc_ref_turborvb} Ha")
    # print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")

    WF_ratio = (
        evaluate_wavefunction_jax(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction_jax(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy_jax(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = compute_bare_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=6)

    vpot_ecp_debug = compute_ecp_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    vpot_ecp_jax = compute_ecp_coulomb_potential_jax(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
    )

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax+vpot_ecp_debug} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=2)
    np.testing.assert_almost_equal(vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=2)

    jax.clear_caches()


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
