"""Mixed precision dtype propagation tests.

These tests verify that when ``--precision-mode=mixed`` is active, each
Precision Zone produces outputs in the expected dtype. They catch JAX
dtype-promotion bugs where fp64 data (io/optimization zone) leaks into
fp32 compute kernels and silently promotes the entire computation to fp64.

Every test:
  1. Configures ``mode="mixed"`` explicitly (independent of the CLI flag).
  2. Calls the target function with realistic inputs.
  3. Asserts the output dtype matches the zone's configured dtype.

In ``mode="full"`` (the default), all zones are fp64 and these tests are
trivially satisfied, so they are skipped to save time.

Run with::

    pytest tests/test_mixed_precision.py -v --precision-mode=mixed
"""

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import configure, get_dtype_jnp  # noqa: E402
from jqmc.atomic_orbital import (  # noqa: E402
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
)
from jqmc.coulomb_potential import (  # noqa: E402
    compute_bare_coulomb_potential,
    compute_bare_coulomb_potential_el_el,
    compute_bare_coulomb_potential_el_ion_element_wise,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_part_all_pairs_jax_weights_grid_points,
)
from jqmc.determinant import (  # noqa: E402
    compute_geminal_all_elements,
    compute_geminal_dn_one_column_elements,
    compute_geminal_up_one_row_elements,
    compute_ln_det_geminal_all_elements,
)
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy  # noqa: E402
from jqmc.jastrow_factor import (  # noqa: E402
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    compute_Jastrow_one_body,
    compute_Jastrow_part,
    compute_Jastrow_three_body,
    compute_Jastrow_two_body,
)
from jqmc.molecular_orbital import (  # noqa: E402
    compute_MOs,
    compute_MOs_grad,
    compute_MOs_laplacian,
)
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402
from jqmc.wavefunction import (  # noqa: E402
    Wavefunction_data,
    compute_kinetic_energy,
    evaluate_ln_wavefunction,
)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TREXIO_DIR = os.path.join(os.path.dirname(__file__), "trexio_example_files")


@pytest.fixture(autouse=True)
def _skip_if_not_mixed(request):
    """Skip these tests unless --precision-mode=mixed is active."""
    if request.config.getoption("--precision-mode") != "mixed":
        pytest.skip("Only runs with --precision-mode=mixed")


@pytest.fixture(autouse=True)
def _configure_mixed():
    """Ensure mixed mode is active and JIT caches are cleared."""
    configure("mixed")
    jax.clear_caches()
    yield
    # Restore full mode after each test to avoid polluting other tests
    configure("full")
    jax.clear_caches()


def _load_trexio(filename: str) -> dict:
    """Helper that loads a TREXIO file and synthesizes random electron coordinates."""
    trexio_file = os.path.join(TREXIO_DIR, filename)
    structure_data, aos_data, mos_data_up, mos_data_dn, geminal_data, coulomb_data = read_trexio_file(
        trexio_file=trexio_file, store_tuple=True
    )
    rng = np.random.default_rng(42)
    n_up = geminal_data.num_electron_up
    n_dn = geminal_data.num_electron_dn
    r_up = jnp.array(rng.standard_normal((n_up, 3)) * 0.5, dtype=jnp.float64)
    r_dn = jnp.array(rng.standard_normal((n_dn, 3)) * 0.5, dtype=jnp.float64)
    return {
        "structure_data": structure_data,
        "aos_data": aos_data,
        "mos_data_up": mos_data_up,
        "mos_data_dn": mos_data_dn,
        "geminal_data": geminal_data,
        "coulomb_data": coulomb_data,
        "r_up": r_up,
        "r_dn": r_dn,
    }


@pytest.fixture
def h2_data():
    """Load H2 all-electron Cartesian basis test data."""
    return _load_trexio("H2_ae_ccpvdz_cart.h5")


@pytest.fixture
def h2_sphe_data():
    """Load H2 all-electron spherical basis test data (covers AO_sphe path)."""
    return _load_trexio("H2_ae_ccpvdz_sphe.h5")


@pytest.fixture
def h2_ecp_data():
    """Load H2 ECP test data (covers ECP local/non-local paths)."""
    return _load_trexio("H2_ecp_ccpvtz.h5")


def _assert_dtype(arr, expected, label):
    """Helper: assert array (or scalar) dtype matches expected."""
    actual = jnp.asarray(arr).dtype
    assert actual == expected, (
        f"{label} dtype is {actual}, expected {expected}. "
        "Check kernel-entry casts of (1) input r_carts, (2) all pytree float fields, "
        "(3) jnp.array/zeros/ones literals."
    )


def _assert_eval_shape_dtype(fn, expected, label, *args, **kwargs):
    """Helper: use jax.eval_shape (no actual execution) to assert output dtype.

    Useful for heavy kernels (ECP non-local) where executing in mixed mode
    is slow. Returns a ShapeDtypeStruct (or pytree thereof) and we walk the
    leaves to assert every float leaf has the expected dtype.
    """
    out = jax.eval_shape(fn, *args, **kwargs)
    leaves = jax.tree_util.tree_leaves(out)
    bad = [(i, leaf.dtype) for i, leaf in enumerate(leaves) if leaf.dtype.kind == "f" and leaf.dtype != expected]
    assert not bad, (
        f"{label} has float leaves with unexpected dtype: {bad}. Expected {expected}. "
        "Heavy kernels checked via jax.eval_shape (no execution)."
    )


# ---------------------------------------------------------------------------
# A. AO zone (orb_eval → float32 in mixed)
# ---------------------------------------------------------------------------


class TestAODtype:
    """Verify AO evaluation outputs are float32 in mixed mode."""

    def test_compute_AOs_output_dtype(self, h2_data):
        """compute_AOs must return float32 (ao_eval zone)."""
        AOs = compute_AOs(h2_data["aos_data"], h2_data["r_up"])
        expected = get_dtype_jnp("ao_eval")
        assert AOs.dtype == expected, (
            f"compute_AOs output dtype is {AOs.dtype}, expected {expected}. "
            "Likely cause: fp64 data (R_carts, exponents, coefficients) not cast to ao_eval dtype inside kernel."
        )

    def test_compute_AOs_grad_output_dtype(self, h2_data):
        """compute_AOs_grad must return ao_grad_lap zone dtype."""
        grad_x, grad_y, grad_z = compute_AOs_grad(h2_data["aos_data"], h2_data["r_up"])
        expected = get_dtype_jnp("ao_grad_lap")
        for name, arr in [("grad_x", grad_x), ("grad_y", grad_y), ("grad_z", grad_z)]:
            assert arr.dtype == expected, f"compute_AOs_grad {name} dtype is {arr.dtype}, expected {expected}."

    def test_compute_AOs_laplacian_output_dtype(self, h2_data):
        """compute_AOs_laplacian must return ao_grad_lap zone dtype."""
        lap = compute_AOs_laplacian(h2_data["aos_data"], h2_data["r_up"])
        expected = get_dtype_jnp("ao_grad_lap")
        assert lap.dtype == expected, f"compute_AOs_laplacian dtype is {lap.dtype}, expected {expected}."


# ---------------------------------------------------------------------------
# B. MO zone (orb_eval → float32 in mixed)
# ---------------------------------------------------------------------------


class TestMODtype:
    """Verify MO evaluation outputs use the determinant-zone dtype.

    Note: AO evaluation itself runs in ``orb_eval`` precision (e.g. fp32 in mixed
    mode), but the (small) MO matmul is upcast to the ``determinant`` zone dtype
    (fp64 by default).  This avoids amplifying fp32 round-off through the
    32x32 determinant / kinetic / energy paths while keeping the heavy AO
    kernels in fp32 (see bug/fp32 diagnostics).
    """

    def test_compute_MOs_output_dtype(self, h2_data):
        """compute_MOs must return mo_eval-zone dtype (fp64 in mixed)."""
        MOs = compute_MOs(h2_data["mos_data_up"], h2_data["r_up"])
        expected = get_dtype_jnp("mo_eval")
        assert MOs.dtype == expected, (
            f"compute_MOs output dtype is {MOs.dtype}, expected {expected}. "
            "compute_MOs should upcast its small matmul to the mo_eval zone "
            "to avoid fp32 amplification downstream."
        )


# ---------------------------------------------------------------------------
# C. Jastrow zone (jastrow → float32 in mixed)
# ---------------------------------------------------------------------------


class TestJastrowDtype:
    """Verify Jastrow outputs are float32 in mixed mode."""

    def test_jastrow_two_body_output_dtype(self, h2_data):
        """compute_Jastrow_two_body must return float32 (jastrow zone).

        Catches: jastrow_2b_param (fp64) not cast to jastrow dtype.
        """
        j2_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade")
        J2 = compute_Jastrow_two_body(j2_data, h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("jastrow_eval")
        assert jnp.asarray(J2).dtype == expected, (
            f"compute_Jastrow_two_body dtype is {jnp.asarray(J2).dtype}, expected {expected}. "
            "Likely cause: jastrow_2b_param not cast to jastrow dtype."
        )

    def test_jastrow_three_body_output_dtype(self, h2_data):
        """compute_Jastrow_three_body must return float32 (jastrow zone).

        Catches: j_matrix (fp64) not cast to jastrow dtype.
        """
        j3_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=h2_data["aos_data"], random_init=True, random_scale=1e-3
        )
        J3 = compute_Jastrow_three_body(j3_data, h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("jastrow_eval")
        assert jnp.asarray(J3).dtype == expected, (
            f"compute_Jastrow_three_body dtype is {jnp.asarray(J3).dtype}, expected {expected}. "
            "Likely cause: j_matrix not cast to jastrow dtype."
        )

    def test_jastrow_part_output_dtype(self, h2_data):
        """compute_Jastrow_part (J1+J2+J3) must return float32 (jastrow zone)."""
        j2_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp")
        j3_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=h2_data["aos_data"], random_init=True, random_scale=1e-3
        )
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=j2_data,
            jastrow_three_body_data=j3_data,
        )
        J = compute_Jastrow_part(jastrow_data, h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("jastrow_eval")
        assert jnp.asarray(J).dtype == expected, f"compute_Jastrow_part dtype is {jnp.asarray(J).dtype}, expected {expected}."


# ---------------------------------------------------------------------------
# D. Geminal zone (geminal → float32 in mixed)
# ---------------------------------------------------------------------------


class TestGeminalDtype:
    """Verify geminal matrix is float32 in mixed mode."""

    def test_geminal_matrix_output_dtype(self, h2_data):
        """compute_geminal_all_elements must return float32 (geminal zone).

        Catches: lambda_matrix or AO data not cast to geminal dtype.
        """
        G = compute_geminal_all_elements(h2_data["geminal_data"], h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("det_eval")
        assert G.dtype == expected, f"compute_geminal_all_elements dtype is {G.dtype}, expected {expected}."


# ---------------------------------------------------------------------------
# E. Determinant zone (determinant → float64 in mixed)
# ---------------------------------------------------------------------------


class TestDeterminantDtype:
    """Verify determinant outputs stay float64 in mixed mode."""

    def test_ln_det_output_dtype(self, h2_data):
        """compute_ln_det_geminal_all_elements must return float64 (determinant zone).

        Even though geminal matrix is float32, the log-det must be computed in float64.
        """
        ln_det = compute_ln_det_geminal_all_elements(h2_data["geminal_data"], h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("det_eval")
        assert jnp.asarray(ln_det).dtype == expected, (
            f"compute_ln_det dtype is {jnp.asarray(ln_det).dtype}, expected {expected}."
        )


# ---------------------------------------------------------------------------
# F. Coulomb zone (coulomb → float32 in mixed)
# ---------------------------------------------------------------------------


class TestCoulombDtype:
    """Verify Coulomb outputs are float32 in mixed mode."""

    def test_bare_coulomb_el_el_output_dtype(self, h2_data):
        """Electron-electron Coulomb must return float32 (coulomb zone)."""
        V = compute_bare_coulomb_potential_el_el(h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("coulomb")
        assert jnp.asarray(V).dtype == expected, (
            f"compute_bare_coulomb_potential_el_el dtype is {jnp.asarray(V).dtype}, expected {expected}."
        )

    def test_bare_coulomb_el_ion_output_dtype(self, h2_data):
        """Electron-ion Coulomb must return float32 (coulomb zone).

        Catches: R_charges (fp64) or structure positions (fp64) not cast.
        """
        V_up, V_dn = compute_bare_coulomb_potential_el_ion_element_wise(
            h2_data["coulomb_data"], h2_data["r_up"], h2_data["r_dn"]
        )
        expected = get_dtype_jnp("coulomb")
        assert V_up.dtype == expected, (
            f"el_ion V_up dtype is {V_up.dtype}, expected {expected}. "
            "Likely cause: R_charges or R_carts not cast to coulomb dtype."
        )

    def test_bare_coulomb_total_output_dtype(self, h2_data):
        """Total bare Coulomb must return float32 (coulomb zone)."""
        V = compute_bare_coulomb_potential(h2_data["coulomb_data"], h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("coulomb")
        assert jnp.asarray(V).dtype == expected, (
            f"compute_bare_coulomb_potential dtype is {jnp.asarray(V).dtype}, expected {expected}."
        )


# ---------------------------------------------------------------------------
# G. Kinetic zone (kinetic → float64 in mixed)
# ---------------------------------------------------------------------------


class TestKineticDtype:
    """Verify kinetic energy stays float64 in mixed mode."""

    def test_kinetic_energy_output_dtype(self, h2_data):
        """compute_kinetic_energy must return float64 (wf_kinetic zone)."""
        wf_data = Wavefunction_data(geminal_data=h2_data["geminal_data"])
        T = compute_kinetic_energy(wf_data, h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("wf_kinetic")
        assert jnp.asarray(T).dtype == expected, f"compute_kinetic_energy dtype is {jnp.asarray(T).dtype}, expected {expected}."


# ---------------------------------------------------------------------------
# H. Wavefunction zone boundary
# ---------------------------------------------------------------------------


class TestWavefunctionDtype:
    """Verify wavefunction evaluation zone boundaries."""

    def test_ln_wavefunction_output_dtype(self, h2_data):
        """evaluate_ln_wavefunction must return float64 (determinant zone).

        The output combines Jastrow (float32) and log-det (float64),
        cast to determinant zone dtype.
        """
        j2_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp")
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=j2_data,
            jastrow_three_body_data=None,
        )
        wf_data = Wavefunction_data(geminal_data=h2_data["geminal_data"], jastrow_data=jastrow_data)
        ln_psi = evaluate_ln_wavefunction(wf_data, h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("wf_eval")
        assert jnp.asarray(ln_psi).dtype == expected, (
            f"evaluate_ln_wavefunction dtype is {jnp.asarray(ln_psi).dtype}, expected {expected}."
        )


# ---------------------------------------------------------------------------
# Extended coverage (a): additional boundary kernels
# ---------------------------------------------------------------------------


class TestAOSpheDtype:
    """Verify AO spherical basis path also returns float32 (orb_eval)."""

    def test_compute_AOs_sphe_output_dtype(self, h2_sphe_data):
        AOs = compute_AOs(h2_sphe_data["aos_data"], h2_sphe_data["r_up"])
        _assert_dtype(AOs, get_dtype_jnp("ao_eval"), "compute_AOs (sphe)")

    def test_compute_AOs_sphe_grad_output_dtype(self, h2_sphe_data):
        gx, gy, gz = compute_AOs_grad(h2_sphe_data["aos_data"], h2_sphe_data["r_up"])
        for name, arr in [("grad_x", gx), ("grad_y", gy), ("grad_z", gz)]:
            _assert_dtype(arr, get_dtype_jnp("ao_grad_lap"), f"compute_AOs_grad sphe {name}")

    def test_compute_AOs_sphe_laplacian_output_dtype(self, h2_sphe_data):
        lap = compute_AOs_laplacian(h2_sphe_data["aos_data"], h2_sphe_data["r_up"])
        _assert_dtype(lap, get_dtype_jnp("ao_grad_lap"), "compute_AOs_laplacian (sphe)")


class TestMOExtendedDtype:
    """MO derivative kernels (kinetic zone)."""

    def test_compute_MOs_grad_output_dtype(self, h2_data):
        gx, gy, gz = compute_MOs_grad(h2_data["mos_data_up"], h2_data["r_up"])
        expected = get_dtype_jnp("mo_grad_lap")
        for name, arr in [("grad_x", gx), ("grad_y", gy), ("grad_z", gz)]:
            _assert_dtype(arr, expected, f"compute_MOs_grad {name}")

    def test_compute_MOs_laplacian_output_dtype(self, h2_data):
        lap = compute_MOs_laplacian(h2_data["mos_data_up"], h2_data["r_up"])
        _assert_dtype(lap, get_dtype_jnp("mo_grad_lap"), "compute_MOs_laplacian")


class TestJastrowOneBodyDtype:
    """Verify J1 forward standalone output (jastrow zone)."""

    def test_compute_Jastrow_one_body_output_dtype(self, h2_data):
        core_electrons = tuple([0.0] * len(h2_data["structure_data"].positions))
        j1_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=h2_data["structure_data"],
            core_electrons=core_electrons,
            jastrow_1b_type="pade",
        )
        J1 = compute_Jastrow_one_body(j1_data, h2_data["r_up"], h2_data["r_dn"])
        _assert_dtype(J1, get_dtype_jnp("jastrow_eval"), "compute_Jastrow_one_body")


class TestGeminalFastUpdateDtype:
    """Verify Geminal row/column kernels used in MCMC fast updates (geminal zone)."""

    @pytest.fixture
    def water_data(self):
        """Water ECP data with multiple electrons (needed for row/column tests)."""
        return _load_trexio("water_ccecp_ccpvqz.h5")

    def test_geminal_up_one_row_output_dtype(self, water_data):
        # Use [0:1] to get shape (1, 3) — compute_orb_api requires (N, 3), not (3,)
        row = compute_geminal_up_one_row_elements(water_data["geminal_data"], water_data["r_up"][0:1], water_data["r_dn"])
        _assert_dtype(row, get_dtype_jnp("det_ratio"), "compute_geminal_up_one_row_elements")

    def test_geminal_dn_one_column_output_dtype(self, water_data):
        # Use [0:1] to get shape (1, 3) — compute_orb_api requires (N, 3), not (3,)
        col = compute_geminal_dn_one_column_elements(water_data["geminal_data"], water_data["r_up"], water_data["r_dn"][0:1])
        _assert_dtype(col, get_dtype_jnp("det_ratio"), "compute_geminal_dn_one_column_elements")


class TestECPDtype:
    """Verify ECP local + non-local paths (coulomb zone).

    Local: executed directly. Non-local: heavy, checked via jax.eval_shape (no run).
    """

    def test_compute_ecp_local_parts_output_dtype(self, h2_ecp_data):
        V_loc = compute_ecp_local_parts_all_pairs(h2_ecp_data["coulomb_data"], h2_ecp_data["r_up"], h2_ecp_data["r_dn"])
        _assert_dtype(V_loc, get_dtype_jnp("coulomb"), "compute_ecp_local_parts_all_pairs")

    def test_compute_ecp_non_local_eval_shape_dtype(self, h2_ecp_data):
        """Heavy ECP non-local kernel: verify dtype via jax.eval_shape (no execution)."""
        wf_data = Wavefunction_data(geminal_data=h2_ecp_data["geminal_data"])
        # Minimal Lebedev-like quadrature placeholder (6-point). Values are not
        # used by eval_shape; only shapes/dtypes matter for static dtype tracing.
        weights = jnp.ones((6,), dtype=jnp.float64) / 6.0
        grid_points = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=jnp.float64,
        )
        _assert_eval_shape_dtype(
            compute_ecp_non_local_part_all_pairs_jax_weights_grid_points,
            get_dtype_jnp("coulomb"),
            "compute_ecp_non_local_part_all_pairs_jax_weights_grid_points",
            h2_ecp_data["coulomb_data"],
            wf_data,
            h2_ecp_data["r_up"],
            h2_ecp_data["r_dn"],
            weights,
            grid_points,
        )


class TestLocalEnergyDtype:
    """Final boundary: compute_local_energy aggregates kinetic + coulomb (kinetic zone)."""

    def test_compute_local_energy_output_dtype(self, h2_data):
        wf_data = Wavefunction_data(geminal_data=h2_data["geminal_data"])
        ham = Hamiltonian_data(
            structure_data=h2_data["structure_data"],
            wavefunction_data=wf_data,
            coulomb_potential_data=h2_data["coulomb_data"],
        )
        RT = jnp.eye(3, dtype=jnp.float64)
        e_L = compute_local_energy(ham, h2_data["r_up"], h2_data["r_dn"], RT)
        _assert_dtype(e_L, get_dtype_jnp("local_energy"), "compute_local_energy")


class TestKineticEvalShape:
    """jax.eval_shape coverage for kinetic energy fast/discretized variants.

    These exercise different code paths (Sherman-Morrison fast update,
    discretized DMC) without actually executing the heavy autodiff laplacian.
    """

    def test_kinetic_energy_eval_shape_dtype(self, h2_data):
        wf_data = Wavefunction_data(geminal_data=h2_data["geminal_data"])
        _assert_eval_shape_dtype(
            compute_kinetic_energy,
            get_dtype_jnp("wf_kinetic"),
            "compute_kinetic_energy (eval_shape)",
            wf_data,
            h2_data["r_up"],
            h2_data["r_dn"],
        )

    def test_evaluate_ln_wavefunction_eval_shape_dtype(self, h2_data):
        wf_data = Wavefunction_data(geminal_data=h2_data["geminal_data"])
        _assert_eval_shape_dtype(
            evaluate_ln_wavefunction,
            get_dtype_jnp("wf_eval"),
            "evaluate_ln_wavefunction (eval_shape)",
            wf_data,
            h2_data["r_up"],
            h2_data["r_dn"],
        )
