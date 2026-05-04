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
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from jax import numpy as jnp

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance, get_tolerance_min  # noqa: E402
from jqmc.determinant import compute_geminal_all_elements  # noqa: E402
from jqmc.jastrow_factor import (  # noqa: E402
    Jastrow_data,
    Jastrow_NN_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402
from jqmc.wavefunction import (  # noqa: E402
    Wavefunction_data,
    _advance_kinetic_energy_all_elements_streaming_state,
    _compute_discretized_kinetic_energy_debug,
    _compute_kinetic_energy_all_elements_auto,
    _compute_kinetic_energy_all_elements_debug,
    _compute_kinetic_energy_all_elements_fast_update_debug,
    _compute_kinetic_energy_auto,
    _compute_kinetic_energy_debug,
    _compute_nodal_distance_debug,
    _init_kinetic_energy_all_elements_streaming_state,
    _kinetic_energy_from_streaming_state,
    compute_discretized_kinetic_energy,
    compute_discretized_kinetic_energy_fast_update,
    compute_kinetic_energy,
    compute_kinetic_energy_all_elements,
    compute_kinetic_energy_all_elements_fast_update,
    compute_nodal_distance,
    evaluate_ln_wavefunction,
    evaluate_ln_wavefunction_fast,
    f_epsilon_PW,
)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


@pytest.mark.activate_if_skip_heavy
@pytest.mark.numerical_diff
@pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
def test_kinetic_energy_analytic_and_numerical(trexio_file: str):
    """Test the kinetic energy computation."""
    (
        structure_data,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file), store_tuple=True
    )

    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp")
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )
    jastrow_nn_data = Jastrow_NN_data.init_from_structure(structure_data=structure_data, hidden_dim=5, num_layers=2, cutoff=5.0)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
        jastrow_nn_data=jastrow_nn_data,
    )
    jastrow_data.sanity_check()

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)
    wavefunction_data.sanity_check()

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -2.0, +2.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    K_debug = _compute_kinetic_energy_debug(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
    K_jax = compute_kinetic_energy(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
    atol, rtol = get_tolerance("wf_kinetic", "loose")
    assert not np.any(np.isnan(np.asarray(np.asarray(K_debug)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(K_jax)))), "NaN detected in second argument"
    np.testing.assert_allclose(
        np.asarray(K_debug),
        np.asarray(K_jax),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
def test_kinetic_energy_analytic_and_auto(trexio_file: str):
    """Compare analytic and autodiff kinetic energy implementations."""
    (
        _,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file), store_tuple=True
    )

    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="pade")
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
    )

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -2.0, +2.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    K_analytic = compute_kinetic_energy(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
    K_auto = _compute_kinetic_energy_auto(
        wavefunction_data=wavefunction_data,
        r_up_carts=jnp.asarray(r_up_carts),
        r_dn_carts=jnp.asarray(r_dn_carts),
    )

    # T_L crosses ao_eval/jastrow_eval/jastrow_grad_lap/wf_kinetic zones; the
    # achievable analytic-vs-auto agreement is bounded by the weakest (fp32 in mixed).
    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"),
        "strict",
    )
    assert not np.any(np.isnan(np.asarray(K_analytic))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(K_auto))), "NaN detected in second argument"
    np.testing.assert_allclose(K_analytic, K_auto, atol=atol, rtol=rtol)


@pytest.mark.activate_if_skip_heavy
@pytest.mark.numerical_diff
@pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
def test_debug_and_auto_kinetic_energy_all_elements(trexio_file: str):
    """Debug vs autodiff kinetic energy per-electron arrays.

    The debug path computes ``-1/2 · ∇²Psi / Psi`` via central finite differences
    on Psi (h = 2e-4); under mixed precision the fp32 round-off in ao_eval /
    jastrow_eval propagates into Psi at ~1e-7 and is amplified by 1/h² = 2.5e7,
    giving an O(1) relative error in the FD Laplacian. Marked ``numerical_diff``
    so conftest skips it under ``--precision-mode=mixed``.
    """
    (
        _,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file), store_tuple=True
    )
    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp")
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
    )

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    rng = np.random.default_rng(42)
    r_up_carts_np = rng.uniform(-2.0, 2.0, size=(num_ele_up, 3))
    r_dn_carts_np = rng.uniform(-2.0, 2.0, size=(num_ele_dn, 3))

    r_up_carts_jnp = jnp.array(r_up_carts_np)
    r_dn_carts_jnp = jnp.array(r_dn_carts_np)

    K_elements_up_debug, K_elements_dn_debug = _compute_kinetic_energy_all_elements_debug(
        wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_np, r_dn_carts=r_dn_carts_np
    )
    K_elements_up_auto, K_elements_dn_auto = _compute_kinetic_energy_all_elements_auto(
        wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp
    )

    atol, rtol = get_tolerance("wf_kinetic", "loose")
    assert not np.any(np.isnan(np.asarray(K_elements_up_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(K_elements_up_auto))), "NaN detected in second argument"
    np.testing.assert_allclose(K_elements_up_debug, K_elements_up_auto, atol=atol, rtol=rtol)
    assert not np.any(np.isnan(np.asarray(K_elements_dn_debug))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(K_elements_dn_auto))), "NaN detected in second argument"
    np.testing.assert_allclose(K_elements_dn_debug, K_elements_dn_auto, atol=atol, rtol=rtol)

    assert not np.any(np.isnan(np.asarray(np.asarray(K_elements_up_debug)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(K_elements_up_auto)))), "NaN detected in second argument"
    np.testing.assert_allclose(
        np.asarray(K_elements_up_debug),
        np.asarray(K_elements_up_auto),
        rtol=rtol,
        atol=atol,
    )

    assert not np.any(np.isnan(np.asarray(np.asarray(K_elements_dn_debug)))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(np.asarray(K_elements_dn_auto)))), "NaN detected in second argument"
    np.testing.assert_allclose(
        np.asarray(K_elements_dn_debug),
        np.asarray(K_elements_dn_auto),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
def test_auto_and_analytic_kinetic_energy_all_elements(trexio_file: str):
    """Autodiff vs analytic kinetic energy per-electron arrays."""
    (
        _,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file), store_tuple=True
    )

    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="pade")
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
    )

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    rng = np.random.default_rng(43)
    r_up_carts_np = rng.uniform(-2.0, 2.0, size=(num_ele_up, 3))
    r_dn_carts_np = rng.uniform(-2.0, 2.0, size=(num_ele_dn, 3))

    r_up_carts_jnp = jnp.array(r_up_carts_np)
    r_dn_carts_jnp = jnp.array(r_dn_carts_np)

    K_elements_up_auto, K_elements_dn_auto = _compute_kinetic_energy_all_elements_auto(
        wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp
    )
    K_elements_up_analytic, K_elements_dn_analytic = compute_kinetic_energy_all_elements(
        wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp
    )

    # T_L crosses ao_eval/jastrow_eval/jastrow_grad_lap/wf_kinetic zones; the
    # achievable analytic-vs-auto agreement is bounded by the weakest (fp32 in mixed).
    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"),
        "strict",
    )
    assert not np.any(np.isnan(np.asarray(K_elements_up_auto))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(K_elements_up_analytic))), "NaN detected in second argument"
    np.testing.assert_allclose(K_elements_up_auto, K_elements_up_analytic, atol=atol, rtol=rtol)
    assert not np.any(np.isnan(np.asarray(K_elements_dn_auto))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(K_elements_dn_analytic))), "NaN detected in second argument"
    np.testing.assert_allclose(K_elements_dn_auto, K_elements_dn_analytic, atol=atol, rtol=rtol)


@pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
def test_fast_update_kinetic_energy_all_elements(trexio_file: str):
    """Fast-update per-electron kinetic energy should match the standard analytic path."""
    (
        _,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file),
        store_tuple=True,
    )

    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp")
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
    )

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -2.0, +2.0
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    r_up_carts_jnp = jnp.asarray(r_up_carts)
    r_dn_carts_jnp = jnp.asarray(r_dn_carts)

    # Standard analytic per-electron kinetic energy
    ke_up_debug, ke_dn_debug = _compute_kinetic_energy_all_elements_fast_update_debug(
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
    )

    # Build geminal inverse explicitly for the fast path
    A = compute_geminal_all_elements(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
    )
    A_inv = jnp.asarray(np.linalg.inv(np.array(A)))

    ke_up_fast, ke_dn_fast = compute_kinetic_energy_all_elements_fast_update(
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
        geminal_inverse=A_inv,
    )

    # Fast-update path crosses ao_eval/jastrow_eval/jastrow_grad_lap/det_ratio/
    # wf_kinetic zones; the achievable agreement is bounded by the weakest
    # (fp32 in mixed for ao_eval / jastrow_eval / jastrow_grad_lap).
    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_ratio", "wf_kinetic"),
        "strict",
    )
    assert not np.any(np.isnan(np.asarray(ke_up_fast))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(ke_up_debug))), "NaN detected in second argument"
    np.testing.assert_allclose(ke_up_fast, ke_up_debug, atol=atol, rtol=rtol)
    assert not np.any(np.isnan(np.asarray(ke_dn_fast))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(ke_dn_debug))), "NaN detected in second argument"
    np.testing.assert_allclose(ke_dn_fast, ke_dn_debug, atol=atol, rtol=rtol)


@pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
def test_debug_and_jax_discretized_kinetic_energy(trexio_file: str):
    """Test the discretized kinetic energy computation."""
    (
        _,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file), store_tuple=True
    )

    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="pade")
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
    )
    jastrow_data.sanity_check()

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)
    wavefunction_data.sanity_check()

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    rng = np.random.default_rng(44)
    r_up_carts_np = rng.uniform(-2.0, 2.0, size=(num_ele_up, 3))
    r_dn_carts_np = rng.uniform(-2.0, 2.0, size=(num_ele_dn, 3))

    r_up_carts_jnp = jnp.array(r_up_carts_np)
    r_dn_carts_jnp = jnp.array(r_dn_carts_np)

    alat = 0.05
    RT = np.eye(3)
    mesh_kinetic_part_r_up_carts_debug, mesh_kinetic_part_r_dn_carts_debug, elements_kinetic_part_debug = (
        _compute_discretized_kinetic_energy_debug(
            alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_np, r_dn_carts=r_dn_carts_np
        )
    )

    # elements_kinetic_part_debug_all = np.array(elements_kinetic_part_debug).reshape(-1, 6)
    # print(np.array(elements_kinetic_part_debug))
    # print(elements_kinetic_part_debug_all.shape)
    # print(elements_kinetic_part_debug_all)

    mesh_kinetic_part_r_up_carts_jax, mesh_kinetic_part_r_dn_carts_jax, elements_kinetic_part_jax = (
        compute_discretized_kinetic_energy(
            alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp, RT=RT
        )
    )

    A = compute_geminal_all_elements(geminal_data=geminal_mo_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp)
    A_old_inv = np.linalg.inv(A)
    (
        mesh_kinetic_part_r_up_carts_jax_fast_update,
        mesh_kinetic_part_r_dn_carts_jax_fast_update,
        elements_kinetic_part_jax_fast_update,
    ) = compute_discretized_kinetic_energy_fast_update(
        alat=alat,
        A_old_inv=A_old_inv,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
        RT=RT,
    )

    # Discretized kinetic energy (LRDMC) crosses ao_eval/jastrow_eval/
    # jastrow_grad_lap/jastrow_ratio/det_ratio/wf_ratio/wf_kinetic zones; the
    # fast_update path uses Sherman-Morrison ratios. Agreement is bounded by
    # the weakest (fp32 in mixed for ao_eval / jastrow_* zones).
    atol, rtol = get_tolerance_min(
        (
            "ao_eval",
            "jastrow_eval",
            "jastrow_grad_lap",
            "jastrow_ratio",
            "det_ratio",
            "wf_ratio",
            "wf_kinetic",
        ),
        "strict",
    )
    assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_up_carts_jax))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_up_carts_debug))), "NaN detected in second argument"
    np.testing.assert_allclose(
        mesh_kinetic_part_r_up_carts_jax,
        mesh_kinetic_part_r_up_carts_debug,
        atol=atol,
        rtol=rtol,
    )
    assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_dn_carts_jax))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_dn_carts_debug))), "NaN detected in second argument"
    np.testing.assert_allclose(
        mesh_kinetic_part_r_dn_carts_jax,
        mesh_kinetic_part_r_dn_carts_debug,
        atol=atol,
        rtol=rtol,
    )
    assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_up_carts_jax_fast_update))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_up_carts_debug))), "NaN detected in second argument"
    np.testing.assert_allclose(
        mesh_kinetic_part_r_up_carts_jax_fast_update,
        mesh_kinetic_part_r_up_carts_debug,
        atol=atol,
        rtol=rtol,
    )
    assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_dn_carts_jax_fast_update))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_dn_carts_debug))), "NaN detected in second argument"
    np.testing.assert_allclose(
        mesh_kinetic_part_r_dn_carts_jax_fast_update,
        mesh_kinetic_part_r_dn_carts_debug,
        atol=atol,
        rtol=rtol,
    )
    assert not np.any(np.isnan(np.asarray(elements_kinetic_part_jax))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(elements_kinetic_part_debug))), "NaN detected in second argument"
    np.testing.assert_allclose(elements_kinetic_part_jax, elements_kinetic_part_debug, atol=atol, rtol=rtol)
    assert not np.any(np.isnan(np.asarray(elements_kinetic_part_jax_fast_update))), "NaN detected in first argument"
    assert not np.any(np.isnan(np.asarray(elements_kinetic_part_debug))), "NaN detected in second argument"
    np.testing.assert_allclose(
        elements_kinetic_part_jax_fast_update,
        elements_kinetic_part_debug,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
def test_nodal_distance_analytic_vs_debug(trexio_file: str):
    """Analytic compute_nodal_distance should match _compute_nodal_distance_debug."""
    (
        structure_data,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file), store_tuple=True
    )

    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp")
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )
    jastrow_nn_data = Jastrow_NN_data.init_from_structure(structure_data=structure_data, hidden_dim=5, num_layers=2, cutoff=5.0)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
        jastrow_nn_data=jastrow_nn_data,
    )

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    num_ele_up = geminal_mo_data.num_electron_up
    num_ele_dn = geminal_mo_data.num_electron_dn
    r_cart_min, r_cart_max = -3.0, +3.0
    np.random.seed(42)
    r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

    r_up_carts_jnp = jnp.asarray(r_up_carts)
    r_dn_carts_jnp = jnp.asarray(r_dn_carts)

    # Analytic path
    nd_analytic = compute_nodal_distance(
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
    )

    # Debug path (paper formula)
    nd_debug = _compute_nodal_distance_debug(
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts_jnp,
        r_dn_carts=r_dn_carts_jnp,
    )

    # They should be identical up to numerical noise
    atol, rtol = get_tolerance("wf_kinetic", "loose")
    np.testing.assert_allclose(
        np.asarray(nd_analytic),
        np.asarray(nd_debug),
        rtol=rtol,
        atol=atol,
    )

    # Sanity: nodal distance should be positive
    assert float(nd_analytic) > 0.0
    assert float(nd_debug) > 0.0


def test_f_epsilon_PW_boundary_values():
    """Test f_epsilon_PW boundary conditions and properties."""
    epsilon_PW = 0.5

    # f(0) = 0
    np.testing.assert_allclose(float(f_epsilon_PW(jnp.array(0.0), epsilon_PW)), 0.0, atol=1e-14)

    # f(epsilon) = 1 (i.e., |t|=1)
    np.testing.assert_allclose(float(f_epsilon_PW(jnp.array(epsilon_PW), epsilon_PW)), 1.0, atol=1e-14)

    # f(x) = 1 for |x| >= epsilon
    np.testing.assert_allclose(float(f_epsilon_PW(jnp.array(2.0 * epsilon_PW), epsilon_PW)), 1.0, atol=1e-14)
    np.testing.assert_allclose(float(f_epsilon_PW(jnp.array(10.0 * epsilon_PW), epsilon_PW)), 1.0, atol=1e-14)

    # f'(0) = 0 and f'(1) = 0 (smooth at boundaries)
    # Check numerically: f(small) ~ 0 (derivative at origin is zero)
    delta = 1.0e-8
    f_at_delta = float(f_epsilon_PW(jnp.array(delta), epsilon_PW))
    # Should scale as t^2, so f(delta/eps) ~ 9*(delta/eps)^2
    assert f_at_delta < 1.0e-10  # very close to 0

    # f(x) >= 0 for interior
    t_vals = jnp.linspace(0.0, epsilon_PW, 100)
    f_vals = f_epsilon_PW(t_vals, epsilon_PW)
    assert jnp.all(f_vals >= -1e-14)


@pytest.mark.activate_if_skip_heavy
@pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
def test_evaluate_ln_wavefunction_fast_forward(trexio_file):
    """Forward value of evaluate_ln_wavefunction_fast must match evaluate_ln_wavefunction."""
    _, _, _, _, geminal_data, _ = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file),
        store_tuple=True,
    )
    wavefunction_data = Wavefunction_data(geminal_data=geminal_data)
    rng = np.random.default_rng(2)
    n_up = geminal_data.num_electron_up
    n_dn = geminal_data.num_electron_dn

    atol, rtol = get_tolerance("wf_kinetic", "strict")
    for _ in range(10):
        r_up = jnp.array(rng.standard_normal((n_up, 3)) * 1.2, dtype=jnp.float64)
        r_dn = jnp.array(rng.standard_normal((n_dn, 3)) * 1.2, dtype=jnp.float64)
        G = compute_geminal_all_elements(geminal_data, r_up, r_dn)
        G_inv = jnp.linalg.inv(G)

        val_ref = float(evaluate_ln_wavefunction(wavefunction_data, r_up, r_dn))
        val_fast = float(evaluate_ln_wavefunction_fast(wavefunction_data, r_up, r_dn, G_inv))

        assert np.isfinite(val_ref), f"Reference value is not finite: {val_ref}"
        assert np.isfinite(val_fast), f"Fast value is not finite: {val_fast}"
        np.testing.assert_allclose(
            val_fast,
            val_ref,
            atol=atol,
            rtol=rtol,
            err_msg=f"Forward mismatch: fast={val_fast:.15f}, ref={val_ref:.15f}",
        )


@pytest.mark.activate_if_skip_heavy
@pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
def test_evaluate_ln_wavefunction_fast_backward(trexio_file):
    """Gradient of evaluate_ln_wavefunction_fast w.r.t. wavefunction_data must match evaluate_ln_wavefunction."""
    _, _, _, _, geminal_data, _ = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file),
        store_tuple=True,
    )
    wavefunction_data = Wavefunction_data(geminal_data=geminal_data)
    rng = np.random.default_rng(3)
    n_up = geminal_data.num_electron_up
    n_dn = geminal_data.num_electron_dn

    grad_ref_fn = jax.grad(evaluate_ln_wavefunction, argnums=0)
    grad_fast_fn = jax.grad(evaluate_ln_wavefunction_fast, argnums=0)

    atol, rtol = get_tolerance("wf_kinetic", "strict")
    for _ in range(10):
        r_up = jnp.array(rng.standard_normal((n_up, 3)) * 1.2, dtype=jnp.float64)
        r_dn = jnp.array(rng.standard_normal((n_dn, 3)) * 1.2, dtype=jnp.float64)
        G = compute_geminal_all_elements(geminal_data, r_up, r_dn)
        G_inv = jnp.linalg.inv(G)

        grad_ref = grad_ref_fn(wavefunction_data, r_up, r_dn)
        grad_fast = grad_fast_fn(wavefunction_data, r_up, r_dn, G_inv)

        jax.tree_util.tree_map(
            lambda a, b: np.testing.assert_allclose(
                np.asarray(a),
                np.asarray(b),
                atol=atol,
                rtol=rtol,
                err_msg="Backward mismatch in evaluate_ln_wavefunction_fast",
            ),
            grad_ref,
            grad_fast,
        )


# ---------------------------------------------------------------------------
# Streaming kinetic-energy state tests (PR1: J3 streaming)
# ---------------------------------------------------------------------------


def _build_A_inv_from_carts(geminal_data, r_up_jnp, r_dn_jnp):
    """Compute A_inv = G(r_up, r_dn)^{-1} via SVD (matches the fast-update warning)."""
    A = compute_geminal_all_elements(
        geminal_data=geminal_data,
        r_up_carts=r_up_jnp,
        r_dn_carts=r_dn_jnp,
    )
    return jnp.linalg.inv(A)


def _streaming_step_consistency_one(wavefunction_data, r_up0, r_dn0, K, atol, rtol, seed=0):
    """Run K random single-electron moves through the streaming state and
    compare the resulting kinetic energies with a fresh fast-update call at
    the final configuration."""
    rng = np.random.RandomState(seed)
    r_up = np.asarray(r_up0, dtype=np.float64).copy()
    r_dn = np.asarray(r_dn0, dtype=np.float64).copy()
    n_up = r_up.shape[0]
    n_dn = r_dn.shape[0]

    A_inv = _build_A_inv_from_carts(wavefunction_data.geminal_data, jnp.asarray(r_up), jnp.asarray(r_dn))
    state = _init_kinetic_energy_all_elements_streaming_state(
        wavefunction_data=wavefunction_data,
        r_up_carts=jnp.asarray(r_up),
        r_dn_carts=jnp.asarray(r_dn),
        geminal_inverse=A_inv,
    )

    for _ in range(K):
        choices = []
        if n_up > 0:
            choices.append(0)
        if n_dn > 0:
            choices.append(1)
        spin = choices[rng.randint(0, len(choices))]
        if spin == 0:
            idx = rng.randint(0, n_up)
            r_up = r_up.copy()
            r_up[idx] = r_up[idx] + 0.05 * rng.randn(3)
            moved_spin_is_up = True
            moved_index = idx
        else:
            idx = rng.randint(0, n_dn)
            r_dn = r_dn.copy()
            r_dn[idx] = r_dn[idx] + 0.05 * rng.randn(3)
            moved_spin_is_up = False
            moved_index = idx

        # rebuild A_inv at the new configuration (mirrors what Sherman-Morrison
        # produces in the GFMC loop, modulo round-off — comparing at the same
        # numerical reference here).
        A_inv = _build_A_inv_from_carts(wavefunction_data.geminal_data, jnp.asarray(r_up), jnp.asarray(r_dn))
        state = _advance_kinetic_energy_all_elements_streaming_state(
            wavefunction_data=wavefunction_data,
            state=state,
            moved_spin_is_up=jnp.asarray(moved_spin_is_up),
            moved_index=jnp.asarray(moved_index, dtype=jnp.int32),
            r_up_carts_new=jnp.asarray(r_up),
            r_dn_carts_new=jnp.asarray(r_dn),
            A_new_inv=A_inv,
        )

    ke_up_stream, ke_dn_stream = _kinetic_energy_from_streaming_state(state)
    ke_up_fresh, ke_dn_fresh = compute_kinetic_energy_all_elements_fast_update(
        wavefunction_data=wavefunction_data,
        r_up_carts=jnp.asarray(r_up),
        r_dn_carts=jnp.asarray(r_dn),
        geminal_inverse=A_inv,
    )
    np.testing.assert_allclose(np.asarray(ke_up_stream), np.asarray(ke_up_fresh), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(ke_dn_stream), np.asarray(ke_dn_fresh), atol=atol, rtol=rtol)


def _build_wavefunction_J3(trexio_file, j2_type="exp", with_J1=False, with_J2=True):
    """Build a Wavefunction_data with J3 + optional J1/J2 from a trexio file.

    PR1 streaming requires J3 to be present (the dispatch demands it).
    """
    from jqmc.jastrow_factor import Jastrow_one_body_data

    (
        structure_data,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file),
        store_tuple=True,
    )

    if with_J1:
        jastrow_one_body_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=0.5,
            structure_data=structure_data,
            core_electrons=tuple([0] * len(structure_data.atomic_numbers)),
            jastrow_1b_type="pade",
        )
    else:
        jastrow_one_body_data = None

    if with_J2:
        jastrow_two_body_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type=j2_type)
    else:
        jastrow_two_body_data = None

    jastrow_three_body_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_one_body_data,
        jastrow_two_body_data=jastrow_two_body_data,
        jastrow_three_body_data=jastrow_three_body_data,
    )
    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)
    return wavefunction_data, geminal_mo_data


@pytest.mark.parametrize(
    "trexio_file",
    ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"],
)
def test_streaming_kinetic_energy_step_consistency(trexio_file):
    """K=32 random single-electron moves advanced via the streaming kinetic
    state must reproduce the fresh fast-update kinetic energy at the resulting
    configuration within strict tolerance."""
    wf, gem = _build_wavefunction_J3(trexio_file)
    n_up = gem.num_electron_up
    n_dn = gem.num_electron_dn
    rng = np.random.RandomState(0)
    r_up0 = 4.0 * rng.rand(n_up, 3) - 2.0
    r_dn0 = 4.0 * rng.rand(n_dn, 3) - 2.0
    atol, rtol = get_tolerance_min(["wf_kinetic", "jastrow_grad_lap"], "strict")
    _streaming_step_consistency_one(wf, r_up0, r_dn0, K=32, atol=atol, rtol=rtol)


@pytest.mark.parametrize("K", [32, 100, 1000])
def test_streaming_kinetic_drift_accumulation(K):
    """Drift accumulation: K-step advance vs fresh init at config_K must stay
    within ``loose`` tolerance even at K=1000, which sets the safety margin
    for ``num_mcmc_per_measurement``."""
    wf, gem = _build_wavefunction_J3("H2_ae_ccpvdz_cart.h5")
    rng = np.random.RandomState(1)
    r_up0 = 4.0 * rng.rand(gem.num_electron_up, 3) - 2.0
    r_dn0 = 4.0 * rng.rand(gem.num_electron_dn, 3) - 2.0
    atol, rtol = get_tolerance_min(["wf_kinetic", "jastrow_grad_lap"], "loose")
    _streaming_step_consistency_one(wf, r_up0, r_dn0, K=K, atol=atol, rtol=rtol, seed=2)


@pytest.mark.parametrize(
    "trexio_file",
    ["H2_ae_ccpvdz_cart.h5", "Li_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"],
)
def test_streaming_kinetic_edge_cases(trexio_file):
    """Edge cases: small electron counts and ``N_up != N_dn`` (Li, N) must
    still match the fresh fast-update result."""
    wf, gem = _build_wavefunction_J3(trexio_file)
    rng = np.random.RandomState(3)
    r_up0 = 4.0 * rng.rand(gem.num_electron_up, 3) - 2.0
    r_dn0 = 4.0 * rng.rand(gem.num_electron_dn, 3) - 2.0
    atol, rtol = get_tolerance_min(["wf_kinetic", "jastrow_grad_lap"], "strict")
    _streaming_step_consistency_one(wf, r_up0, r_dn0, K=24, atol=atol, rtol=rtol, seed=4)


@pytest.mark.parametrize("jastrow_combo", ["J3_only", "J1_J3", "J2_J3", "J1_J2_J3"])
def test_streaming_kinetic_jastrow_combinations(jastrow_combo):
    """Streaming path must work for every J3-containing Jastrow combination
    (PR1 dispatch requires J3 + ``jastrow_nn_data is None``)."""
    with_J1 = "J1" in jastrow_combo
    with_J2 = "J2" in jastrow_combo
    wf, gem = _build_wavefunction_J3("water_ccecp_ccpvqz.h5", with_J1=with_J1, with_J2=with_J2)
    rng = np.random.RandomState(5)
    r_up0 = 4.0 * rng.rand(gem.num_electron_up, 3) - 2.0
    r_dn0 = 4.0 * rng.rand(gem.num_electron_dn, 3) - 2.0
    atol, rtol = get_tolerance_min(["wf_kinetic", "jastrow_grad_lap"], "strict")
    _streaming_step_consistency_one(wf, r_up0, r_dn0, K=24, atol=atol, rtol=rtol, seed=6)


def test_streaming_kinetic_walker_axis_vmap():
    """``vmap`` over the walker axis must produce results equal to the
    independent per-walker streaming chains. Confirms the state pytree carries
    walkers correctly along the leading axis."""
    wf, gem = _build_wavefunction_J3("H2_ae_ccpvdz_cart.h5")
    n_walkers = 4
    rng = np.random.RandomState(7)
    r_up_w = jnp.asarray(4.0 * rng.rand(n_walkers, gem.num_electron_up, 3) - 2.0)
    r_dn_w = jnp.asarray(4.0 * rng.rand(n_walkers, gem.num_electron_dn, 3) - 2.0)

    # Per-walker A_inv and initial state, computed via vmap.
    def _make_init_state(r_up, r_dn):
        A_inv = _build_A_inv_from_carts(wf.geminal_data, r_up, r_dn)
        return _init_kinetic_energy_all_elements_streaming_state(
            wavefunction_data=wf,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            geminal_inverse=A_inv,
        ), A_inv

    states, A_invs = jax.vmap(_make_init_state, in_axes=(0, 0))(r_up_w, r_dn_w)

    # Single up-electron move on walker 0 only; other walkers see the same
    # advance call but with their own (unchanged) inputs.
    moved_spin_is_up = jnp.asarray([True] * n_walkers)
    moved_index = jnp.asarray([0] * n_walkers, dtype=jnp.int32)

    # Apply the same delta to electron 0 across walkers (just to exercise the
    # vmap; the per-walker chains remain independent because `state` and
    # `r_*_carts_new` are walker-batched).
    delta = 0.05 * rng.randn(3)
    r_up_w_new = r_up_w.at[:, 0, :].add(jnp.asarray(delta))
    A_invs_new = jax.vmap(lambda ru, rd: _build_A_inv_from_carts(wf.geminal_data, ru, rd), in_axes=(0, 0))(r_up_w_new, r_dn_w)

    advance_vmapped = jax.vmap(
        lambda st, msi, mi, ru, rd, ai: _advance_kinetic_energy_all_elements_streaming_state(
            wavefunction_data=wf,
            state=st,
            moved_spin_is_up=msi,
            moved_index=mi,
            r_up_carts_new=ru,
            r_dn_carts_new=rd,
            A_new_inv=ai,
        ),
        in_axes=(0, 0, 0, 0, 0, 0),
    )
    states_new = advance_vmapped(states, moved_spin_is_up, moved_index, r_up_w_new, r_dn_w, A_invs_new)

    # Reference: fresh evaluation per walker.
    ke_up_v, ke_dn_v = jax.vmap(_kinetic_energy_from_streaming_state)(states_new)
    atol, rtol = get_tolerance_min(["wf_kinetic", "jastrow_grad_lap"], "strict")
    for w in range(n_walkers):
        ke_up_ref, ke_dn_ref = compute_kinetic_energy_all_elements_fast_update(
            wavefunction_data=wf,
            r_up_carts=r_up_w_new[w],
            r_dn_carts=r_dn_w[w],
            geminal_inverse=A_invs_new[w],
        )
        np.testing.assert_allclose(np.asarray(ke_up_v[w]), np.asarray(ke_up_ref), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(ke_dn_v[w]), np.asarray(ke_dn_ref), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# j3_state-forwarding consistency tests (PR4/PR5: ECP non-local AO reuse +
# discretized kinetic AO reuse)
#
# These verify the Python-static dispatch in
# ``_compute_ratio_Jastrow_part_rank1_update`` and
# ``_compute_ratio_Jastrow_part_split_spin``: the with-state path must produce
# identical Jastrow ratios (and therefore identical kinetic / ECP elements) as
# the no-state path when the streaming state is consistent with
# ``(r_up_carts, r_dn_carts)``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trexio_file,jastrow_combo",
    [
        ("water_ccecp_ccpvqz.h5", "J3_only"),
        ("water_ccecp_ccpvqz.h5", "J1_J2_J3"),
        ("H2_ae_ccpvdz_cart.h5", "J2_J3"),
        ("N_ae_ccpvdz_cart.h5", "J1_J3"),
    ],
)
def test_streaming_discretized_kinetic_j3_state_consistency(trexio_file, jastrow_combo):
    """``compute_discretized_kinetic_energy_fast_update`` must return the same
    kinetic mesh elements whether the J3 streaming state is forwarded or not.

    Validates the Python-static dispatch in
    ``_compute_ratio_Jastrow_part_rank1_update`` (the ratio kernel called by
    the discretized kinetic for the LRDMC mesh).
    """
    with_J1 = "J1" in jastrow_combo
    with_J2 = "J2" in jastrow_combo
    wf, gem = _build_wavefunction_J3(trexio_file, with_J1=with_J1, with_J2=with_J2)
    rng = np.random.RandomState(11)
    r_up = jnp.asarray(4.0 * rng.rand(gem.num_electron_up, 3) - 2.0)
    r_dn = jnp.asarray(4.0 * rng.rand(gem.num_electron_dn, 3) - 2.0)
    A_inv = _build_A_inv_from_carts(wf.geminal_data, r_up, r_dn)
    state = _init_kinetic_energy_all_elements_streaming_state(
        wavefunction_data=wf, r_up_carts=r_up, r_dn_carts=r_dn, geminal_inverse=A_inv
    )

    alat = 0.40
    RT = jnp.eye(3, dtype=jnp.float64)

    rup_ref, rdn_ref, ke_ref = compute_discretized_kinetic_energy_fast_update(
        alat=alat,
        wavefunction_data=wf,
        A_old_inv=A_inv,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
        RT=RT,
        j3_state=None,
    )
    rup_st, rdn_st, ke_st = compute_discretized_kinetic_energy_fast_update(
        alat=alat,
        wavefunction_data=wf,
        A_old_inv=A_inv,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
        RT=RT,
        j3_state=state.j3_state,
    )

    atol, rtol = get_tolerance("wf_kinetic", "strict")
    np.testing.assert_allclose(np.asarray(rup_st), np.asarray(rup_ref), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(rdn_st), np.asarray(rdn_ref), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(ke_st), np.asarray(ke_ref), atol=atol, rtol=rtol)


@pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5"])
@pytest.mark.parametrize("jastrow_combo", ["J3_only", "J2_J3", "J1_J2_J3"])
def test_streaming_ecp_nonlocal_j3_state_consistency(trexio_file, jastrow_combo):
    """``compute_ecp_non_local_parts_nearest_neighbors_fast_update`` (tmove
    path, ``flag_determinant_only=False``) must return identical V_nonlocal
    whether the J3 streaming state is forwarded or not.

    Validates the Python-static dispatch in
    ``_compute_ratio_Jastrow_part_split_spin`` (the ratio kernel used for the
    block-structured non-local ECP grid).
    """
    from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors_fast_update

    with_J1 = "J1" in jastrow_combo
    with_J2 = "J2" in jastrow_combo
    (_, _, _, _, _, coulomb_potential_data) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_file),
        store_tuple=True,
    )
    wf, gem = _build_wavefunction_J3(trexio_file, with_J1=with_J1, with_J2=with_J2)
    rng = np.random.RandomState(13)
    r_up = jnp.asarray(4.0 * rng.rand(gem.num_electron_up, 3) - 2.0)
    r_dn = jnp.asarray(4.0 * rng.rand(gem.num_electron_dn, 3) - 2.0)
    A_inv = _build_A_inv_from_carts(wf.geminal_data, r_up, r_dn)
    state = _init_kinetic_energy_all_elements_streaming_state(
        wavefunction_data=wf, r_up_carts=r_up, r_dn_carts=r_dn, geminal_inverse=A_inv
    )

    RT = jnp.eye(3, dtype=jnp.float64)

    rup_ref, rdn_ref, V_ref, sV_ref = compute_ecp_non_local_parts_nearest_neighbors_fast_update(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wf,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
        RT=RT,
        A_old_inv=A_inv,
        flag_determinant_only=False,
        j3_state=None,
    )
    rup_st, rdn_st, V_st, sV_st = compute_ecp_non_local_parts_nearest_neighbors_fast_update(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wf,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
        RT=RT,
        A_old_inv=A_inv,
        flag_determinant_only=False,
        j3_state=state.j3_state,
    )

    atol, rtol = get_tolerance("wf_kinetic", "strict")
    np.testing.assert_allclose(np.asarray(rup_st), np.asarray(rup_ref), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(rdn_st), np.asarray(rdn_ref), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(V_st), np.asarray(V_ref), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(sV_st), np.asarray(sV_ref), atol=atol, rtol=rtol)


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
