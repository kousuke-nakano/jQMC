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

# Add the project root directory to sys.path to allow executing this script directly
# This is necessary because relative imports (e.g. 'from ..jqmc') are not allowed
# when running a script directly (as __main__).
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.coulomb_potential import _compute_bare_coulomb_potential_debug, compute_bare_coulomb_potential  # noqa: E402
from jqmc.hamiltonians import Hamiltonian_data  # noqa: E402
from jqmc.jastrow_factor import Jastrow_data  # noqa: E402
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402
from jqmc.wavefunction import Wavefunction_data, compute_kinetic_energy, evaluate_wavefunction  # noqa: E402

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def test_comparison_with_TurboRVB_wo_Jastrow_AE():
    """Test comparison with the corresponding all-electron TurboRVB calculation without Jastrow factor."""
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "H2_ae_ccpvqz.h5"), store_tuple=True
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

    old_r_up_carts = np.array([[1.24966813461677, 1.61428085239132, -0.164635126284132]])
    old_r_dn_carts = np.array([[0.429108885274501, -0.007401675646959482, -0.528818841020972]])
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_dn_carts[0] = [0.429108885274501, -0.007401675646959482, -1.65889610184907]

    WF_ratio_ref_turborvb = 1.27268619372079
    kinc_ref_turborvb = 1.00473369916001
    vpot_ref_turborvb = -2.32849688911382

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

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax + vpot_ecp_jax} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug, vpot_ref_turborvb, decimal=3)
    np.testing.assert_almost_equal(vpot_bare_jax, vpot_ref_turborvb, decimal=3)

    jax.clear_caches()


def test_comparison_with_TurboRVB_w_2b_1b3b_Jastrow_AE():
    """Test comparison with the corresponding all-electron TurboRVB calculation with the full Jastrow factor."""
    (
        structure_data,
        _,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "H2_ae_ccpvqz.h5"), store_tuple=True
    )

    with open(
        os.path.join(os.path.dirname(__file__), "trexio_example_files", "jastrow_data_w_1b_2b_1b3b_ae.pkl"),
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

    old_r_up_carts = np.array([[-0.140725692347622, 1.794610704318, 0.541399181483924]])
    old_r_dn_carts = np.array([[1.18814636744078, 0.02606967395580784, -1.62047650291381]])
    new_r_up_carts = old_r_up_carts.copy()
    new_r_dn_carts = old_r_dn_carts.copy()
    new_r_up_carts[0] = [0.985621336113153, 1.794610704318, 0.541399181483924]

    WF_ratio_ref_turborvb = 0.539734425254117
    kinc_ref_turborvb = 0.06762960720224656
    vpot_ref_turborvb = -1.22497631738529

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

    # print(f"wf_ratio={WF_ratio} Ha")
    # print(f"kinc={kinc} Ha")
    # print(f"vpot={vpot_bare_jax+vpot_ecp_debug} Ha")

    np.testing.assert_almost_equal(WF_ratio, WF_ratio_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(kinc, kinc_ref_turborvb, decimal=6)
    np.testing.assert_almost_equal(vpot_bare_debug, vpot_ref_turborvb, decimal=2)
    np.testing.assert_almost_equal(vpot_bare_jax, vpot_ref_turborvb, decimal=2)

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
