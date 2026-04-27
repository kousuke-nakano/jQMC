"""Tests for MCMC.save_to_hdf5 / MCMC.load_from_hdf5 — Phase 1."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax

jax.config.update("jax_enable_x64", True)

from jqmc._checkpoint import merge_rank_checkpoints
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_NN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

TREXIO_DIR = os.path.join(os.path.dirname(__file__), "trexio_example_files")

TREXIO_FILES = [
    "H2_ae_ccpvdz_cart.h5",  # all-electron
    "H2_ecp_ccpvdz_cart.h5",  # ECP
]

JASTROW_COMBOS = [
    "none",  # empty Jastrow
    "1b",  # one-body only
    "1b+2b",  # one-body + two-body
    "1b+2b+3b",  # one-body + two-body + three-body
    "1b+2b+3b+nn",  # one-body + two-body + three-body + NN
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_hamiltonian(trexio_file, jastrow_combo):
    """Build a Hamiltonian_data from a trexio file with a given Jastrow combination.

    Args:
        trexio_file: Basename of the trexio file (e.g. ``"H2_ae_ccpvdz_cart.h5"``).
        jastrow_combo: One of ``JASTROW_COMBOS``.

    Returns:
        Hamiltonian_data with the requested Jastrow configuration.
    """
    structure_data, aos_data, _, _, geminal_data, coulomb_potential_data = read_trexio_file(
        trexio_file=os.path.join(TREXIO_DIR, trexio_file),
        store_tuple=True,
    )

    jastrow_one_body_data = None
    jastrow_two_body_data = None
    jastrow_three_body_data = None
    nn_jastrow_data = None

    if jastrow_combo in ("1b", "1b+2b", "1b+2b+3b", "1b+2b+3b+nn"):
        core_electrons = tuple([0.0] * len(structure_data.positions))
        jastrow_one_body_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=core_electrons,
            jastrow_1b_type="pade",
        )

    if jastrow_combo in ("1b+2b", "1b+2b+3b", "1b+2b+3b+nn"):
        jastrow_two_body_data = Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0,
            jastrow_2b_type="exp",
        )

    if jastrow_combo in ("1b+2b+3b", "1b+2b+3b+nn"):
        jastrow_three_body_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=aos_data,
        )

    if jastrow_combo == "1b+2b+3b+nn":
        nn_jastrow_data = Jastrow_NN_data.init_from_structure(structure_data=structure_data)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_one_body_data,
        jastrow_two_body_data=jastrow_two_body_data,
        jastrow_three_body_data=jastrow_three_body_data,
        jastrow_nn_data=nn_jastrow_data,
    )
    jastrow_data.sanity_check()

    wfd = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)
    wfd.sanity_check()

    hd = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wfd,
    )
    hd.sanity_check()
    return hd


def _make_mcmc(hamiltonian_data, num_walkers=4, mcmc_seed=12345, **kwargs):
    """Create and return an MCMC instance."""
    from jqmc.jqmc_mcmc import MCMC

    return MCMC(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        mcmc_seed=mcmc_seed,
        **kwargs,
    )


def _save_merge(mcmc, hd, tmp_path, mpi_size=1, rank=0):
    """Save one rank and merge into a checkpoint HDF5 file."""
    rank_file = str(tmp_path / f"._restart_rank{rank}.h5")
    mcmc.save_to_hdf5(rank_file)

    merged = str(tmp_path / "restart.h5")
    tmp_pattern = str(tmp_path / "._restart_rank{rank}.h5")
    merge_rank_checkpoints(
        output_path=merged,
        mpi_size=mpi_size,
        driver_type="MCMC",
        hamiltonian_data=hd,
        tmp_pattern=tmp_pattern,
    )
    return merged


# ---------------------------------------------------------------------------
# Tests — basic round-trip (parameterized by trexio file × Jastrow combo)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trexio_file", TREXIO_FILES, ids=lambda f: f.replace(".h5", ""))
@pytest.mark.parametrize("jastrow_combo", JASTROW_COMBOS)
class TestMCMCSaveLoadRoundtrip:
    """Round-trip tests: MCMC → save → merge → load → compare."""

    @pytest.fixture(autouse=True)
    def _setup(self, trexio_file, jastrow_combo, tmp_path):
        """Build hamiltonian_data once per test method."""
        self.hd = _build_hamiltonian(trexio_file, jastrow_combo)
        self.tmp_path = tmp_path

    def test_basic_roundtrip(self):
        """Scalars, walker state, and observables survive round-trip."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd)
        mcmc.run(num_mcmc_steps=3)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        # Scalar configs
        assert mcmc2.mcmc_counter == mcmc.mcmc_counter
        assert mcmc2.num_walkers == mcmc.num_walkers

        # Observables
        np.testing.assert_array_equal(np.asarray(mcmc2.e_L), np.asarray(mcmc.e_L))
        np.testing.assert_array_equal(np.asarray(mcmc2.e_L2), np.asarray(mcmc.e_L2))
        np.testing.assert_array_equal(np.asarray(mcmc2.w_L), np.asarray(mcmc.w_L))

    def test_walker_state_roundtrip(self):
        """Walker coordinates survive round-trip."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd)
        mcmc.run(num_mcmc_steps=2)

        r_up_before = np.asarray(mcmc._MCMC__latest_r_up_carts)
        r_dn_before = np.asarray(mcmc._MCMC__latest_r_dn_carts)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        np.testing.assert_array_equal(
            np.asarray(mcmc2._MCMC__latest_r_up_carts),
            r_up_before,
        )
        np.testing.assert_array_equal(
            np.asarray(mcmc2._MCMC__latest_r_dn_carts),
            r_dn_before,
        )

    def test_rng_state_roundtrip(self):
        """RNG keys survive round-trip."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd)
        mcmc.run(num_mcmc_steps=1)

        keys_before = np.asarray(mcmc._MCMC__jax_PRNG_key_list)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        np.testing.assert_array_equal(
            np.asarray(mcmc2._MCMC__jax_PRNG_key_list),
            keys_before,
        )

    def test_counters_roundtrip(self):
        """Counters (mcmc_counter, accepted/rejected) are preserved."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd)
        mcmc.run(num_mcmc_steps=5)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        assert mcmc2._MCMC__mcmc_counter == mcmc._MCMC__mcmc_counter
        assert mcmc2._MCMC__accepted_moves == mcmc._MCMC__accepted_moves
        assert mcmc2._MCMC__rejected_moves == mcmc._MCMC__rejected_moves

    def test_continue_run_after_load(self):
        """A loaded MCMC can continue running without error."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd)
        mcmc.run(num_mcmc_steps=2)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        mcmc2.run(num_mcmc_steps=2)
        assert mcmc2.mcmc_counter == 4
        assert mcmc2.e_L.shape[0] == 4

    def test_hamiltonian_embedded(self):
        """The restored MCMC has a functional hamiltonian_data."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd)
        mcmc.run(num_mcmc_steps=1)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        hd = mcmc2.hamiltonian_data
        assert hd is not None
        n_up = self.hd.wavefunction_data.geminal_data.num_electron_up
        n_dn = self.hd.wavefunction_data.geminal_data.num_electron_dn
        assert hd.wavefunction_data.geminal_data.num_electron_up == n_up
        assert hd.wavefunction_data.geminal_data.num_electron_dn == n_dn

    def test_param_deriv_roundtrip(self):
        """Parameter gradient observables survive round-trip."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd, comput_log_WF_param_deriv=True)
        mcmc.run(num_mcmc_steps=2)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        dln_orig = mcmc.dln_Psi_dc
        dln_loaded = mcmc2.dln_Psi_dc
        assert set(dln_orig.keys()) == set(dln_loaded.keys())
        for k in dln_orig:
            np.testing.assert_array_equal(dln_loaded[k], dln_orig[k])

    def test_force_derivative_roundtrip(self):
        """Force observables survive round-trip when comput_position_deriv=True."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd, comput_position_deriv=True)
        mcmc.run(num_mcmc_steps=2)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        np.testing.assert_array_equal(np.asarray(mcmc2.force_HF_stored), np.asarray(mcmc.force_HF_stored))
        np.testing.assert_array_equal(np.asarray(mcmc2.force_PP_stored), np.asarray(mcmc.force_PP_stored))


# ---------------------------------------------------------------------------
# Tests — optax optimizer round-trip (only combos that have variational params)
# ---------------------------------------------------------------------------

# Jastrow combos that have variational parameters (needed for run_optimize)
_JASTROW_WITH_PARAMS = ["1b", "1b+2b", "1b+2b+3b", "1b+2b+3b+nn"]


@pytest.mark.parametrize("trexio_file", TREXIO_FILES, ids=lambda f: f.replace(".h5", ""))
@pytest.mark.parametrize("jastrow_combo", _JASTROW_WITH_PARAMS)
class TestMCMCOptaxRoundtrip:
    """Optax optimizer state round-trip through MCMC save/load."""

    @pytest.fixture(autouse=True)
    def _setup(self, trexio_file, jastrow_combo, tmp_path, monkeypatch):
        """Build hamiltonian_data once per test method.

        ``run_optimize`` writes ``hamiltonian_data_opt_step_*.h5`` checkpoint
        files relative to the current working directory.  Without isolating
        the CWD per test, parametrized runs (especially under pytest-xdist)
        race on the same filename and h5py raises
        ``BlockingIOError: Resource temporarily unavailable`` from the
        underlying file lock.  Switch CWD to the per-test ``tmp_path`` so
        each test owns its own checkpoint files.
        """
        self.hd = _build_hamiltonian(trexio_file, jastrow_combo)
        self.tmp_path = tmp_path
        monkeypatch.chdir(tmp_path)

    def test_optax_adam_state_roundtrip(self):
        """After 1 optax optimization step, optimizer_runtime survives save→load."""
        import jax.tree_util as tu

        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd, comput_log_WF_param_deriv=True)
        mcmc.run_optimize(
            num_mcmc_steps=5,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=5,
            optimizer_kwargs={"method": "adam", "learning_rate": 1e-3},
        )

        rt_before = mcmc._MCMC__optimizer_runtime
        assert rt_before is not None
        assert rt_before["method"] == "adam"
        assert rt_before["optax_state"] is not None

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        rt_after = mcmc2._MCMC__optimizer_runtime
        assert rt_after is not None
        assert rt_after["method"] == "adam"
        assert rt_after["hyperparameters"]["learning_rate"] == 1e-3
        assert rt_after["optax_state"] is not None

        orig_leaves = tu.tree_leaves(rt_before["optax_state"])
        rest_leaves = tu.tree_leaves(rt_after["optax_state"])
        assert len(orig_leaves) == len(rest_leaves)
        for a, b in zip(orig_leaves, rest_leaves, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_optax_continue_optimize_after_load(self):
        """A loaded MCMC can continue optax optimization without error."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd, comput_log_WF_param_deriv=True)
        mcmc.run_optimize(
            num_mcmc_steps=5,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=5,
            optimizer_kwargs={"method": "adam", "learning_rate": 1e-3},
        )

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        mcmc2.run_optimize(
            num_mcmc_steps=5,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=5,
            optimizer_kwargs={"method": "adam", "learning_rate": 1e-3},
        )
        assert mcmc2._MCMC__i_opt == 2

    def test_sr_optimizer_roundtrip(self):
        """SR optimizer state (no optax) survives round-trip."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd, comput_log_WF_param_deriv=True)
        mcmc.run_optimize(
            num_mcmc_steps=5,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=5,
            optimizer_kwargs={"method": "sr", "delta": 1e-3},
        )

        rt_before = mcmc._MCMC__optimizer_runtime
        assert rt_before["method"] == "sr"
        assert rt_before["optax_state"] is None

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        rt_after = mcmc2._MCMC__optimizer_runtime
        assert rt_after["method"] == "sr"
        assert rt_after["optax_state"] is None
        assert rt_after["hyperparameters"]["delta"] == 1e-3
