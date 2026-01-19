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
import sys
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp

# Add the project root directory to sys.path to allow executing this script directly
# This is necessary because relative imports (e.g. 'from ..jqmc') are not allowed
# when running a script directly (as __main__).
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.coulomb_potential import (  # noqa: E402
    _compute_bare_coulomb_potential_debug,
    _compute_ecp_coulomb_potential_debug,
    compute_bare_coulomb_potential,
    compute_ecp_coulomb_potential,
)
from jqmc.determinant import compute_geminal_all_elements  # noqa: E402
from jqmc.hamiltonians import Hamiltonian_data  # noqa: E402
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data  # noqa: E402
from jqmc.structure import _find_nearest_index_jnp  # noqa: E402
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402
from jqmc.wavefunction import Wavefunction_data, compute_kinetic_energy, evaluate_wavefunction  # noqa: E402

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

Nv = 6
NN = 1


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
    vpotoff_ref_turborvb = 0.329197252921634  # with rotation
    vpotoff_ref_turborvb = 0.328893830058865  # without rotation

    # print(f"wf_ratio_ref={WF_ratio_ref_turborvb} Ha")
    # print(f"kinc_ref={kinc_ref_turborvb} Ha")
    # print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")

    WF_ratio = (
        evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = _compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = compute_bare_coulomb_potential(
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
        Nv=Nv,
        NN=NN,
        RT=np.eye(3),
    )

    vpot_ecp_jax = compute_ecp_coulomb_potential(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
        Nv=Nv,
        NN=NN,
        RT=jnp.eye(3),
    )

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=10)
    np.testing.assert_almost_equal(vpot_ecp_debug, vpot_ecp_jax, decimal=10)

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax + vpot_ecp_jax} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=5)
    np.testing.assert_almost_equal(vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=5)

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
    vpotoff_ref_turborvb = 0.244575316335042  # with rotation
    vpotoff_ref_turborvb = 0.243517439611676  # without rotation

    # print(f"wf_ratio_ref={WF_ratio_ref_turborvb} Ha")
    # print(f"kinc_ref={kinc_ref_turborvb} Ha")
    # print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")

    WF_ratio = (
        evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = _compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = compute_bare_coulomb_potential(
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
        Nv=Nv,
        NN=NN,
        RT=np.eye(3),
    )

    vpot_ecp_jax = compute_ecp_coulomb_potential(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
        Nv=Nv,
        NN=NN,
        RT=jnp.eye(3),
    )

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=10)
    np.testing.assert_almost_equal(vpot_ecp_debug, vpot_ecp_jax, decimal=10)

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax + vpot_ecp_jax} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=5)
    np.testing.assert_almost_equal(vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=5)

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
    vpotoff_ref_turborvb = 0.285186134621918  # with rotation
    vpotoff_ref_turborvb = 0.284240877900265  # without rotation

    # print(f"wf_ratio_ref={WF_ratio_ref_turborvb} Ha")
    # print(f"kinc_ref={kinc_ref_turborvb} Ha")
    # print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")

    WF_ratio = (
        evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = _compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = compute_bare_coulomb_potential(
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
        Nv=Nv,
        NN=NN,
        RT=np.eye(3),
    )

    vpot_ecp_jax = compute_ecp_coulomb_potential(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
        Nv=Nv,
        NN=NN,
        RT=jnp.eye(3),
    )

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=10)
    np.testing.assert_almost_equal(vpot_ecp_debug, vpot_ecp_jax, decimal=10)

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax + vpot_ecp_jax} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=5)
    np.testing.assert_almost_equal(vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=5)

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
    vpotoff_ref_turborvb = 0.02700582402227284  # with rotation
    vpotoff_ref_turborvb = 0.02774284473669801  # without rotation

    # print(f"wf_ratio_ref={WF_ratio_ref_turborvb} Ha")
    # print(f"kinc_ref={kinc_ref_turborvb} Ha")
    # print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")

    WF_ratio = (
        evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = _compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = compute_bare_coulomb_potential(
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
        Nv=Nv,
        NN=NN,
        RT=np.eye(3),
    )

    vpot_ecp_jax = compute_ecp_coulomb_potential(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
        Nv=Nv,
        NN=NN,
        RT=jnp.eye(3),
    )

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=10)
    np.testing.assert_almost_equal(vpot_ecp_debug, vpot_ecp_jax, decimal=10)

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax + vpot_ecp_jax} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=5)
    np.testing.assert_almost_equal(vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=5)

    jax.clear_caches()


def test_full_comparison_with_TurboRVB_w_2b_1b3b_Jastrow_w_ecp():
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
            [-1.13450385875760, -0.698914730480577, -6.290951981744008e-003],
            [-2.25378719009775, 0.693895756460611, -4.612006323250584e-002],
            [-0.753191857352684, 0.314330338959413, 0.456739833308641],
            [-1.60902246286275, 0.499927465264998, 0.700105816369930],
        ]
    )
    old_r_dn_carts = np.array(
        [
            [-1.52590493546481, -1.13601932859996, 0.586518269898014],
            [0.635659512640246, 0.398999201990364, -0.745191606127732],
            [-2.00590358216444, -0.465069404417879, 0.360171216755478],
            [-0.302866379660751, -0.890252305196045, 0.345597836490454],
        ]
    )
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_up_carts[2] = [-0.753191857352684, -1.183650619518406e-002, 0.456739833308641]

    fa_turbo = 0.470249568592385
    fb_turbo = 0.437344721251066
    T_ratio_turbo = 1.06518922117014
    final_ratio_turbo = 0.901584512996174
    WF_ratio_ref_turborvb = 0.846407844801284
    kinc_ref_turborvb = 13.8637480286375
    vpot_ref_turborvb = -30.808883190726
    vpotoff_ref_turborvb = 0.168465630163985
    R_AS_turborvb_old = 0.124245223553222
    R_AS_turborvb_new = 0.116703654039403
    reweight_turborvb = 0.151330476290540  # (R_AS/R_AS_new)**2

    geminal_old_turborvb = np.array(
        [
            [0.186887184114679, 7.221020612173907e-003, 2.919097229181558e-002, 5.283664938570871e-002],
            [2.711743127945612e-002, -1.872067172512451e-002, 5.968147400894821e-002, -1.363982711796792e-002],
            [0.196787090743597, 9.308561211374290e-002, 5.042625023653007e-003, 0.135256480405370],
            [0.152575006966341, -3.569426461507245e-002, 9.441528784169728e-002, 1.156481954187638e-002],
        ]
    ).T
    geminal_old_inv_turborvb = np.array(
        [
            [28.3428553785392, 25.1090600332577, -6.46293061989623, -24.2896159779225],
            [60.8384654289715, 107.593342484689, -5.51752310057257, -86.5266697623310],
            [-12.6799632376411, 13.7073868055362, 5.86243683627820, 5.53407554454391],
            [-82.6337748432271, -111.090072627085, 20.3750774128231, 94.6820779038449],
        ]
    ).T

    F_old_turborvb = 53823.3438428566
    S_old_turborvb = 4.833741852724715e-003

    geminal_new_turborvb = np.array(
        [
            [0.186887184114679, 7.221020612173907e-03, 7.892154586169434e-02, 5.283664938570871e-02],
            [2.711743127945612e-02, -1.872067172512451e-02, 6.631227501216763e-02, -1.363982711796792e-02],
            [0.196787090743597, 9.308561211374290e-02, 4.145830624511430e-02, 0.135256480405370],
            [0.152575006966341, -3.569426461507245e-02, 0.140717058457383, 1.156481954187638e-02],
        ]
    ).T
    geminal_new_inv_turborvb = np.array(
        [
            [31.2877172833434, 21.9255838305659, -7.82445403955623, -25.5748790298734],
            [54.4536972959671, 114.495451333313, -2.56559814680722, -83.7400852494081],
            [-13.6339111349753, 14.7386305541198, 6.30348380056440, 5.95041900157655],
            [-78.8179295346585, -115.215105693377, 18.6108647192022, 93.0166806303557],
        ]
    ).T
    F_new_turborvb = 54231.8526090902
    S_new_turborvb = 5.669181330133306e-003

    # print(f"fa={fa_turbo}")
    # print(f"fb={fb_turbo}")
    # print(f"T_ratio={T_ratio_turbo}")
    # print(f"wf_ratio_ref={WF_ratio_ref_turborvb}")
    # print(f"kinc_ref={kinc_ref_turborvb} Ha")
    # print(f"vpot_ref={vpot_ref_turborvb + vpotoff_ref_turborvb} Ha")
    # print(f"R_AS_old={R_AS_turborvb_old}")
    # print(f"R_AS_new={R_AS_turborvb_new}")
    # print(f"reweight={reweight_turborvb}")
    # print(f"geminal_old_inv={geminal_old_inv_turborvb}")
    # print(f"geminal_new_inv={geminal_new_inv_turborvb}")
    # print(f"S_old_turborvb={S_old_turborvb}")
    # print(f"S_new_turborvb={S_new_turborvb}")
    # print(f"F_old_turborvb={F_old_turborvb}")
    # print(f"F_new_turborvb={F_new_turborvb}")

    # Dt parameter
    Dt = 2.0
    # charges
    charges = jnp.array(hamiltonian_data.structure_data.atomic_numbers) - jnp.array(
        hamiltonian_data.coulomb_potential_data.z_cores
    )
    # coords
    coords = hamiltonian_data.structure_data._positions_cart_jnp

    # comput f_a
    old_r_cart = old_r_up_carts[2]
    nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, old_r_cart)
    R_cart = coords[nearest_atom_index]
    Z = charges[nearest_atom_index]
    norm_r_R = jnp.linalg.norm(old_r_cart - R_cart)
    fa = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

    # comput f_b
    new_r_cart = new_r_up_carts[2]
    nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, new_r_cart)
    R_cart = coords[nearest_atom_index]
    Z = charges[nearest_atom_index]
    norm_r_R = jnp.linalg.norm(new_r_cart - R_cart)
    fb = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

    T_ratio = (fa / fb) * jnp.exp(
        -(jnp.linalg.norm(new_r_cart - old_r_cart) ** 2) * (1.0 / (2.0 * fb**2 * Dt**2) - 1.0 / (2.0 * fa**2 * Dt**2))
    )

    WF_ratio = (
        evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=new_r_up_carts,
            r_dn_carts=new_r_dn_carts,
        )
        / evaluate_wavefunction(
            wavefunction_data=hamiltonian_data.wavefunction_data,
            r_up_carts=old_r_up_carts,
            r_dn_carts=old_r_dn_carts,
        )
    ) ** 2.0

    kinc = compute_kinetic_energy(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_debug = _compute_bare_coulomb_potential_debug(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
    )

    vpot_bare_jax = compute_bare_coulomb_potential(
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
        Nv=Nv,
        NN=NN,
        RT=np.eye(3),
    )

    vpot_ecp_jax = compute_ecp_coulomb_potential(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=new_r_up_carts,
        r_dn_carts=new_r_dn_carts,
        wavefunction_data=wavefunction_data,
        Nv=Nv,
        NN=NN,
        RT=jnp.eye(3),
    )

    # compute the AS factor
    theta = 3.0 / 8.0
    epsilon = 0.30

    geminal_old = compute_geminal_all_elements(hamiltonian_data.wavefunction_data.geminal_data, old_r_up_carts, old_r_dn_carts)

    # compute F \equiv the square of Frobenius norm of geminal_inv
    geminal_old_inv = np.linalg.inv(geminal_old)
    F_old = np.sum(geminal_old_inv**2)

    # compute the scaling factor
    S_old = np.min(np.sum(geminal_old**2, axis=0))

    # compute R_AS
    R_AS_old = (S_old * F_old) ** (-theta)

    geminal_new = compute_geminal_all_elements(hamiltonian_data.wavefunction_data.geminal_data, new_r_up_carts, new_r_dn_carts)

    # compute F \equiv the square of Frobenius norm of geminal_inv
    geminal_new_inv = np.linalg.inv(geminal_new)
    F_new = np.sum(geminal_new_inv**2)

    # compute the scaling factor
    S_new = np.min(np.sum(geminal_new**2, axis=0))

    # compute R_AS
    R_AS_new = (S_new * F_new) ** (-theta)

    # eps
    R_AS_new_eps = jnp.maximum(R_AS_new, epsilon)
    R_AS_old_eps = jnp.maximum(R_AS_old, epsilon)
    R_AS_ratio = ((R_AS_new_eps / R_AS_new) / (R_AS_old_eps / R_AS_old)) ** 2
    WF_ratio = WF_ratio * R_AS_ratio
    reweight = (R_AS_new / R_AS_new_eps) ** 2

    final_ratio = WF_ratio * T_ratio

    np.testing.assert_almost_equal(vpot_bare_debug, vpot_bare_jax, decimal=10)
    np.testing.assert_almost_equal(vpot_ecp_debug, vpot_ecp_jax, decimal=10)

    # print(f"fa={fa}")
    # print(f"fb={fb}")
    # print(f"T_ratio={T_ratio}")
    # print(f"wf_ratio={WF_ratio}")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax + vpot_ecp_jax} Ha")
    # print(f"R_AS_old={R_AS_old}")
    # print(f"R_AS_new={R_AS_new}")
    # print(f"reweight={reweight}")
    # print(f"geminal_old_inv={geminal_old_inv}")
    # print(f"geminal_new_inv={geminal_new_inv}")
    # print(f"S_old={S_old}")
    # print(f"S_new={S_new}")
    # print(f"F_old={F_old}")
    # print(f"F_new={F_new}")

    np.testing.assert_almost_equal(fa_turbo, fa, decimal=6)
    np.testing.assert_almost_equal(fb_turbo, fb, decimal=6)
    np.testing.assert_almost_equal(T_ratio_turbo, T_ratio, decimal=6)
    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(final_ratio_turbo, final_ratio, decimal=6)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug + vpot_ecp_debug, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=5)
    np.testing.assert_almost_equal(vpot_bare_jax + vpot_ecp_jax, vpot_ref_turborvb + vpotoff_ref_turborvb, decimal=5)

    np.testing.assert_almost_equal(geminal_old_turborvb, geminal_old, decimal=6)
    np.testing.assert_almost_equal(geminal_new_turborvb, geminal_new, decimal=6)
    np.testing.assert_almost_equal(S_old_turborvb, S_old, decimal=6)
    np.testing.assert_almost_equal(S_new_turborvb, S_new, decimal=6)

    # np.testing.assert_almost_equal(geminal_old_inv_turborvb, geminal_old_inv, decimal=4)
    # np.testing.assert_almost_equal(geminal_new_inv_turborvb, geminal_new_inv, decimal=4)
    # np.testing.assert_almost_equal(F_old_turborvb, F_old, decimal=4)
    # np.testing.assert_almost_equal(F_new_turborvb, F_new, decimal=4)

    np.testing.assert_almost_equal(R_AS_turborvb_old, R_AS_old, decimal=6)
    np.testing.assert_almost_equal(R_AS_turborvb_new, R_AS_new, decimal=6)
    np.testing.assert_almost_equal(reweight_turborvb, reweight, decimal=6)

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
