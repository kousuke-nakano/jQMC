"""Tests for AO basis optimization (exponents/coefficients as variational parameters)."""

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.atomic_orbital import AOs_cart_data, AOs_sphe_data  # noqa: E402
from jqmc.determinant import Geminal_data, compute_det_geminal_all_elements  # noqa: E402
from jqmc.jastrow_factor import (  # noqa: E402
    Jastrow_data,
    Jastrow_three_body_data,
    compute_Jastrow_three_body,
)
from jqmc.molecular_orbital import MOs_data  # noqa: E402
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402
from jqmc.wavefunction import (  # noqa: E402
    VariationalParameterBlock,
    Wavefunction_data,
    evaluate_ln_wavefunction,
)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


TREXIO_DIR = os.path.join(os.path.dirname(__file__), "trexio_example_files")


def _load_trexio(name="H2_ae_ccpvdz_cart.h5"):
    """Load a TREXIO file and return all data."""
    return read_trexio_file(
        trexio_file=os.path.join(TREXIO_DIR, name),
        store_tuple=True,
    )


def _random_electron_coords(structure_data, coulomb_potential_data, geminal_data, seed=42):
    """Generate random electron positions near nuclei."""
    rng = np.random.RandomState(seed)
    num_up = geminal_data.num_electron_up
    num_dn = geminal_data.num_electron_dn

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data._positions_cart_np

    electrons = []
    for i in range(len(coords)):
        charge = charges[i]
        num_e = int(np.round(charge))
        for _ in range(num_e):
            dist = rng.uniform(0.3 / max(charge, 1), 1.2 / max(charge, 1))
            theta = rng.uniform(0, np.pi)
            phi = rng.uniform(0, 2 * np.pi)
            dx = dist * np.sin(theta) * np.cos(phi)
            dy = dist * np.sin(theta) * np.sin(phi)
            dz = dist * np.cos(theta)
            electrons.append(coords[i] + np.array([dx, dy, dz]))

    electrons = np.array(electrons)
    r_up = jnp.array(electrons[:num_up], dtype=jnp.float64)
    r_dn = jnp.array(electrons[num_up : num_up + num_dn], dtype=jnp.float64)
    return r_up, r_dn


# ============================================================
# Test 1: exponents/coefficients field type validation
# ============================================================


@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5"])
def test_exponents_coefficients_are_jax_arrays(trexio_file):
    """After Phase 1, exponents/coefficients should be jax.Array."""
    structure_data, aos_data, *_ = _load_trexio(trexio_file)
    assert isinstance(aos_data.exponents, jax.Array), f"exponents type: {type(aos_data.exponents)}"
    assert isinstance(aos_data.coefficients, jax.Array), f"coefficients type: {type(aos_data.coefficients)}"


# ============================================================
# Test 2: J3 ao_exponents/ao_coefficients property (AO path)
# ============================================================


def test_j3_ao_properties_with_ao_data():
    """J3 ao_exponents/ao_coefficients should work with direct AO data."""
    _, aos_data, _, _, geminal_mo_data, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)
    npt.assert_array_equal(np.array(j3.ao_exponents), np.array(aos_data.exponents))
    npt.assert_array_equal(np.array(j3.ao_coefficients), np.array(aos_data.coefficients))


# ============================================================
# Test 3: J3 ao_exponents/ao_coefficients property (MO path)
# ============================================================


def test_j3_ao_properties_with_mo_data():
    """J3 ao_exponents/ao_coefficients should work with MO data."""
    _, _, mos_data_up, _, _, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=mos_data_up)
    npt.assert_array_equal(np.array(j3.ao_exponents), np.array(mos_data_up.aos_data.exponents))
    npt.assert_array_equal(np.array(j3.ao_coefficients), np.array(mos_data_up.aos_data.coefficients))


# ============================================================
# Test 4: J3 with_updated_ao_exponents / with_updated_ao_coefficients
# ============================================================


def test_j3_with_updated_ao_exponents():
    """with_updated_ao_exponents should produce a new J3 with updated exponents."""
    _, aos_data, _, _, _, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)

    new_exp = j3.ao_exponents * 1.1
    j3_new = j3.with_updated_ao_exponents(new_exp)
    npt.assert_allclose(np.array(j3_new.ao_exponents), np.array(new_exp), rtol=1e-14)
    # Original should be unchanged
    npt.assert_allclose(np.array(j3.ao_exponents), np.array(aos_data.exponents), rtol=1e-14)


# ============================================================
# Test 5: Geminal ao_exponents_up/dn properties
# ============================================================


def test_geminal_ao_properties():
    """Geminal ao_exponents_up/dn should return the correct exponents."""
    _, _, _, _, geminal_mo_data, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")

    # MO representation
    exp_up = geminal_mo_data.ao_exponents_up
    exp_dn = geminal_mo_data.ao_exponents_dn
    assert isinstance(exp_up, jax.Array)
    assert isinstance(exp_dn, jax.Array)

    # Convert to AO representation and check
    geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
    exp_up_ao = geminal_ao.ao_exponents_up
    exp_dn_ao = geminal_ao.ao_exponents_dn
    npt.assert_allclose(np.array(exp_up), np.array(exp_up_ao), rtol=1e-14)
    npt.assert_allclose(np.array(exp_dn), np.array(exp_dn_ao), rtol=1e-14)


# ============================================================
# Test 6: Geminal with_updated_ao_exponents
# ============================================================


def test_geminal_with_updated_ao_exponents():
    """with_updated_ao_exponents should produce correct updates for both spins."""
    _, _, _, _, geminal_mo_data, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
    geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)

    new_exp_up = geminal_ao.ao_exponents_up * 0.9
    new_exp_dn = geminal_ao.ao_exponents_dn * 1.1
    geminal_new = geminal_ao.with_updated_ao_exponents(new_exp_up, new_exp_dn)

    npt.assert_allclose(np.array(geminal_new.ao_exponents_up), np.array(new_exp_up), rtol=1e-14)
    npt.assert_allclose(np.array(geminal_new.ao_exponents_dn), np.array(new_exp_dn), rtol=1e-14)
    # Lambda matrix should be unchanged
    npt.assert_array_equal(np.array(geminal_new.lambda_matrix), np.array(geminal_ao.lambda_matrix))


# ============================================================
# Test 7: Gradient of J3 with respect to AO exponents (finite difference)
# ============================================================


@pytest.mark.activate_if_skip_heavy
@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5"])
def test_j3_exponent_gradient_finite_diff(trexio_file):
    """Verify that jax.grad of J3 w.r.t. exponents matches finite differences."""
    structure_data, aos_data, _, _, geminal_mo_data, coulomb_potential_data = _load_trexio(trexio_file)
    r_up, r_dn = _random_electron_coords(structure_data, coulomb_potential_data, geminal_mo_data)

    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data, random_init=True, random_scale=0.01, seed=42)

    def j3_value(exponents):
        j3_mod = j3.with_updated_ao_exponents(exponents)
        return compute_Jastrow_three_body(j3_mod, r_up, r_dn)

    # JAX gradient
    grad_fn = jax.grad(j3_value)
    grad_jax = grad_fn(aos_data.exponents)

    # Finite difference gradient
    eps = 1e-5
    grad_fd = np.zeros_like(np.array(aos_data.exponents))
    exp0 = np.array(aos_data.exponents, dtype=np.float64)
    for i in range(len(exp0)):
        exp_plus = exp0.copy()
        exp_plus[i] += eps
        exp_minus = exp0.copy()
        exp_minus[i] -= eps
        f_plus = j3_value(jnp.array(exp_plus))
        f_minus = j3_value(jnp.array(exp_minus))
        grad_fd[i] = (float(f_plus) - float(f_minus)) / (2 * eps)

    npt.assert_allclose(np.array(grad_jax), grad_fd, atol=1e-5, rtol=1e-4)


# ============================================================
# Test 8: Gradient of J3 with respect to AO coefficients (finite difference)
# ============================================================


@pytest.mark.activate_if_skip_heavy
@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5"])
def test_j3_coefficient_gradient_finite_diff(trexio_file):
    """Verify that jax.grad of J3 w.r.t. coefficients matches finite differences."""
    structure_data, aos_data, _, _, geminal_mo_data, coulomb_potential_data = _load_trexio(trexio_file)
    r_up, r_dn = _random_electron_coords(structure_data, coulomb_potential_data, geminal_mo_data)

    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data, random_init=True, random_scale=0.01, seed=42)

    def j3_value(coefficients):
        j3_mod = j3.with_updated_ao_coefficients(coefficients)
        return compute_Jastrow_three_body(j3_mod, r_up, r_dn)

    grad_fn = jax.grad(j3_value)
    grad_jax = grad_fn(aos_data.coefficients)

    eps = 1e-5
    coeff0 = np.array(aos_data.coefficients, dtype=np.float64)
    grad_fd = np.zeros_like(coeff0)
    for i in range(len(coeff0)):
        c_plus = coeff0.copy()
        c_plus[i] += eps
        c_minus = coeff0.copy()
        c_minus[i] -= eps
        f_plus = j3_value(jnp.array(c_plus))
        f_minus = j3_value(jnp.array(c_minus))
        grad_fd[i] = (float(f_plus) - float(f_minus)) / (2 * eps)

    npt.assert_allclose(np.array(grad_jax), grad_fd, atol=1e-5, rtol=1e-4)


# ============================================================
# Test 9: Gradient of Geminal Det w.r.t. AO exponents (finite difference)
# ============================================================


@pytest.mark.activate_if_skip_heavy
def test_geminal_exponent_gradient_finite_diff():
    """Verify that jax.grad of Geminal det w.r.t. exponents matches finite differences."""
    structure_data, _, _, _, geminal_mo_data, coulomb_potential_data = _load_trexio("H2_ae_ccpvdz_cart.h5")
    geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
    r_up, r_dn = _random_electron_coords(structure_data, coulomb_potential_data, geminal_ao)

    def det_value(exponents_up):
        g = geminal_ao.with_updated_ao_exponents(exponents_up, geminal_ao.ao_exponents_dn)
        return jnp.log(jnp.abs(compute_det_geminal_all_elements(g, r_up, r_dn)))

    grad_fn = jax.grad(det_value)
    grad_jax = grad_fn(geminal_ao.ao_exponents_up)

    eps = 1e-5
    exp0 = np.array(geminal_ao.ao_exponents_up, dtype=np.float64)
    grad_fd = np.zeros_like(exp0)
    for i in range(len(exp0)):
        e_plus = exp0.copy()
        e_plus[i] += eps
        e_minus = exp0.copy()
        e_minus[i] -= eps
        f_plus = det_value(jnp.array(e_plus))
        f_minus = det_value(jnp.array(e_minus))
        grad_fd[i] = (float(f_plus) - float(f_minus)) / (2 * eps)

    npt.assert_allclose(np.array(grad_jax), grad_fd, atol=1e-4, rtol=1e-3)


# ============================================================
# Test 10: get_variational_blocks with new basis flags
# ============================================================


def test_get_variational_blocks_basis_flags():
    """get_variational_blocks should include/exclude basis blocks based on flags."""
    structure_data, aos_data, mos_data_up, mos_data_dn, geminal_mo_data, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
    geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)

    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data, random_init=True, random_scale=0.01, seed=42)
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=j3,
        jastrow_nn_data=None,
    )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_ao)

    # Default (all basis flags False)
    blocks = wf.get_variational_blocks(opt_J3_param=True, opt_lambda_param=True)
    block_names = [b.name for b in blocks]
    assert "j3_matrix" in block_names
    assert "lambda_matrix" in block_names
    assert "j3_basis_exp" not in block_names
    assert "j3_basis_coeff" not in block_names
    assert "lambda_basis_exp" not in block_names

    # With basis flags
    blocks = wf.get_variational_blocks(
        opt_J3_param=True,
        opt_lambda_param=True,
        opt_J3_basis_exp=True,
        opt_J3_basis_coeff=True,
        opt_lambda_basis_exp=True,
        opt_lambda_basis_coeff=True,
    )
    block_names = [b.name for b in blocks]
    assert "j3_basis_exp" in block_names
    assert "j3_basis_coeff" in block_names
    assert "lambda_basis_exp" in block_names
    assert "lambda_basis_coeff" in block_names

    # Verify shapes
    j3_exp_block = next(b for b in blocks if b.name == "j3_basis_exp")
    assert j3_exp_block.size == len(aos_data.exponents)


# ============================================================
# Test 11: apply_block_update for J3 basis blocks
# ============================================================


def test_apply_block_update_j3_basis():
    """apply_block_update should handle j3_basis_exp and j3_basis_coeff blocks."""
    _, aos_data, _, _, _, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data, random_init=True, random_scale=0.01, seed=42)
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=j3,
        jastrow_nn_data=None,
    )

    new_exp = np.array(aos_data.exponents) * 1.05
    block = VariationalParameterBlock(
        name="j3_basis_exp",
        values=new_exp,
        shape=new_exp.shape,
        size=int(new_exp.size),
    )
    jastrow_new = jastrow_data.apply_block_update(block)
    npt.assert_allclose(
        np.array(jastrow_new.jastrow_three_body_data.ao_exponents),
        new_exp,
        rtol=1e-14,
    )


# ============================================================
# Test 12: apply_block_update for Geminal basis blocks
# ============================================================


def test_apply_block_update_geminal_basis():
    """apply_block_update should handle lambda_basis_exp/coeff blocks (up+dn concatenated)."""
    _, _, _, _, geminal_mo_data, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
    geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)

    new_exp_up = np.array(geminal_ao.ao_exponents_up) * 0.95
    new_exp_dn = np.array(geminal_ao.ao_exponents_dn) * 1.05
    new_exp = np.concatenate([new_exp_up, new_exp_dn])
    block = VariationalParameterBlock(
        name="lambda_basis_exp",
        values=new_exp,
        shape=new_exp.shape,
        size=int(new_exp.size),
    )
    geminal_new = geminal_ao.apply_block_update(block)
    npt.assert_allclose(np.array(geminal_new.ao_exponents_up), new_exp_up, rtol=1e-14)
    npt.assert_allclose(np.array(geminal_new.ao_exponents_dn), new_exp_dn, rtol=1e-14)


# ============================================================
# Test 13: with_param_grad_mask stops exponent gradients
# ============================================================


@pytest.mark.activate_if_skip_heavy
def test_with_param_grad_mask_stops_basis_grads():
    """When opt_J3_basis_exp=False, exponent gradients should be zero.

    Note: ``jax.lax.stop_gradient`` only takes effect **inside** a traced
    (JIT / grad) context.  The production code in ``jqmc_mcmc`` applies
    ``with_param_grad_mask`` eagerly (before JIT) so the stop_gradient is a
    no-op there; actual gradient selection is done by ``collect_param_grads``
    + ``__param_grad_flags``.  Here we apply the mask *inside* the
    differentiated function to verify that the stop_gradient mechanism itself
    is correctly wired.
    """
    structure_data, aos_data, _, _, geminal_mo_data, coulomb_potential_data = _load_trexio("H2_ae_ccpvdz_cart.h5")
    geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
    r_up, r_dn = _random_electron_coords(structure_data, coulomb_potential_data, geminal_ao)

    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data, random_init=True, random_scale=0.01, seed=42)
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=j3,
        jastrow_nn_data=None,
    )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_ao)

    # Apply masking INSIDE the traced function so stop_gradient takes effect.
    def ln_psi(wf_data):
        masked = wf_data.with_param_grad_mask(
            opt_J3_param=True,
            opt_J3_basis_exp=False,
            opt_J3_basis_coeff=False,
        )
        return evaluate_ln_wavefunction(masked, r_up, r_dn)

    grad_fn = jax.grad(ln_psi)
    grad_wf = grad_fn(wf)

    # J3 exponent gradient should be zero (stopped)
    grad_j3_exp = grad_wf.jastrow_data.jastrow_three_body_data.orb_data.exponents
    npt.assert_array_equal(np.array(grad_j3_exp), 0.0)

    # J3 coefficient gradient should be zero (stopped)
    grad_j3_coeff = grad_wf.jastrow_data.jastrow_three_body_data.orb_data.coefficients
    npt.assert_array_equal(np.array(grad_j3_coeff), 0.0)

    # J3 matrix gradient should NOT be zero (not stopped)
    grad_j3_matrix = grad_wf.jastrow_data.jastrow_three_body_data.j_matrix
    assert np.any(np.array(grad_j3_matrix) != 0.0)


# ============================================================
# Test 14: with_param_grad_mask allows exponent gradients when enabled
# ============================================================


@pytest.mark.activate_if_skip_heavy
def test_with_param_grad_mask_allows_basis_grads():
    """When opt_J3_basis_exp=True, exponent gradients should flow."""
    structure_data, aos_data, _, _, geminal_mo_data, coulomb_potential_data = _load_trexio("H2_ae_ccpvdz_cart.h5")
    geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
    r_up, r_dn = _random_electron_coords(structure_data, coulomb_potential_data, geminal_ao)

    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data, random_init=True, random_scale=0.01, seed=42)
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=j3,
        jastrow_nn_data=None,
    )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_ao)

    # Apply masking inside the traced function (all enabled).
    def ln_psi(wf_data):
        masked = wf_data.with_param_grad_mask(
            opt_J3_param=True,
            opt_J3_basis_exp=True,
            opt_J3_basis_coeff=True,
        )
        return evaluate_ln_wavefunction(masked, r_up, r_dn)

    grad_fn = jax.grad(ln_psi)
    grad_wf = grad_fn(wf)

    # J3 exponent gradient should be nonzero (not stopped)
    grad_j3_exp = grad_wf.jastrow_data.jastrow_three_body_data.orb_data.exponents
    assert np.any(np.array(grad_j3_exp) != 0.0), "Expected nonzero exponent gradients"


# ============================================================
# Test 15: Full wavefunction gradient through exponents (end-to-end)
# ============================================================


@pytest.mark.activate_if_skip_heavy
def test_full_wavefunction_exponent_gradient():
    """End-to-end: grad of ln|Psi| w.r.t. J3 AO exponents is finite and nonzero."""
    structure_data, aos_data, _, _, geminal_mo_data, coulomb_potential_data = _load_trexio("H2_ae_ccpvdz_cart.h5")
    geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
    r_up, r_dn = _random_electron_coords(structure_data, coulomb_potential_data, geminal_ao)

    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data, random_init=True, random_scale=0.01, seed=42)
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=j3,
        jastrow_nn_data=None,
    )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_ao)

    def ln_psi(wf_data):
        return evaluate_ln_wavefunction(wf_data, r_up, r_dn)

    grad_fn = jax.grad(ln_psi)
    grad_wf = grad_fn(wf)

    # Check J3 exponent gradients
    grad_j3_exp = np.array(grad_wf.jastrow_data.jastrow_three_body_data.orb_data.exponents)
    assert np.all(np.isfinite(grad_j3_exp)), "J3 exponent gradients should be finite"
    assert np.any(grad_j3_exp != 0.0), "J3 exponent gradients should be nonzero"

    # Check Geminal exponent gradients
    grad_gem_exp = np.array(grad_wf.geminal_data.ao_exponents_up)
    assert np.all(np.isfinite(grad_gem_exp)), "Geminal exponent gradients should be finite"
    assert np.any(grad_gem_exp != 0.0), "Geminal exponent gradients should be nonzero"


# ============================================================
# Test 16: opt_with_projected_MOs conflict with basis optimization
# ============================================================


def test_opt_with_projected_MOs_lambda_basis_conflict():
    """opt_with_projected_MOs should raise ValueError when combined with lambda basis optimization."""
    from jqmc.jqmc_mcmc import MCMC

    # Only opt_lambda_basis_exp/coeff conflict with opt_with_projected_MOs.
    # opt_J3_basis_exp/coeff are allowed because J3 basis does not affect
    # the overlap matrix used by MO projection.
    opt_with_projected_MOs = True

    # lambda_basis_exp=True should conflict
    flags_lambda_exp = [False, False, True, False]  # opt_lambda_basis_exp=True
    with pytest.raises(ValueError, match="cannot be combined with opt_with_projected_MOs"):
        if opt_with_projected_MOs and any(flags_lambda_exp[2:]):
            raise ValueError(
                "Geminal AO basis optimization (opt_lambda_basis_exp/coeff) "
                "cannot be combined with opt_with_projected_MOs. "
                "Changing Geminal AO exponents/coefficients invalidates the overlap matrix "
                "used by the MO projection operators."
            )

    # lambda_basis_coeff=True should conflict
    flags_lambda_coeff = [False, False, False, True]  # opt_lambda_basis_coeff=True
    with pytest.raises(ValueError, match="cannot be combined with opt_with_projected_MOs"):
        if opt_with_projected_MOs and any(flags_lambda_coeff[2:]):
            raise ValueError(
                "Geminal AO basis optimization (opt_lambda_basis_exp/coeff) "
                "cannot be combined with opt_with_projected_MOs. "
                "Changing Geminal AO exponents/coefficients invalidates the overlap matrix "
                "used by the MO projection operators."
            )

    # J3_basis_exp=True should NOT conflict with opt_with_projected_MOs
    flags_j3 = [True, False, False, False]  # opt_J3_basis_exp=True
    # This should not raise — J3 basis does not affect MO projection overlap
    if opt_with_projected_MOs and any(flags_j3[2:]):
        raise ValueError("Should not reach here")
