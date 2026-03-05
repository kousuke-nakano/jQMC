"""Tests for GFMC_t / GFMC_n save_to_hdf5 / load_from_hdf5 — Phase 2."""

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
    """Build a Hamiltonian_data from a trexio file with a given Jastrow combination."""
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


def _make_gfmc_t(hamiltonian_data, num_walkers=4, mcmc_seed=12345, **kwargs):
    """Create and return a GFMC_t instance."""
    from jqmc.jqmc_gfmc import GFMC_t

    return GFMC_t(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        mcmc_seed=mcmc_seed,
        **kwargs,
    )


def _make_gfmc_n(hamiltonian_data, num_walkers=4, mcmc_seed=12345, **kwargs):
    """Create and return a GFMC_n instance."""
    from jqmc.jqmc_gfmc import GFMC_n

    return GFMC_n(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        mcmc_seed=mcmc_seed,
        **kwargs,
    )


def _save_merge_gfmc_t(gfmc, hd, tmp_path, mpi_size=1, rank=0):
    """Save one rank and merge into a checkpoint HDF5 file (GFMC_t)."""
    rank_file = str(tmp_path / f"._restart_rank{rank}.h5")
    gfmc.save_to_hdf5(rank_file)

    merged = str(tmp_path / "restart.h5")
    tmp_pattern = str(tmp_path / "._restart_rank{rank}.h5")
    merge_rank_checkpoints(
        output_path=merged,
        mpi_size=mpi_size,
        driver_type="GFMC_t",
        hamiltonian_data=hd,
        tmp_pattern=tmp_pattern,
    )
    return merged


def _save_merge_gfmc_n(gfmc, hd, tmp_path, mpi_size=1, rank=0):
    """Save one rank and merge into a checkpoint HDF5 file (GFMC_n)."""
    rank_file = str(tmp_path / f"._restart_rank{rank}.h5")
    gfmc.save_to_hdf5(rank_file)

    merged = str(tmp_path / "restart.h5")
    tmp_pattern = str(tmp_path / "._restart_rank{rank}.h5")
    merge_rank_checkpoints(
        output_path=merged,
        mpi_size=mpi_size,
        driver_type="GFMC_n",
        hamiltonian_data=hd,
        tmp_pattern=tmp_pattern,
    )
    return merged


# ===========================================================================
# GFMC_t tests
# ===========================================================================


@pytest.mark.parametrize("trexio_file", TREXIO_FILES, ids=lambda f: f.replace(".h5", ""))
@pytest.mark.parametrize("jastrow_combo", JASTROW_COMBOS)
class TestGFMCtSaveLoadRoundtrip:
    """Round-trip tests: GFMC_t → save → merge → load → compare."""

    @pytest.fixture(autouse=True)
    def _setup(self, trexio_file, jastrow_combo, tmp_path):
        """Build hamiltonian_data once per test method."""
        self.hd = _build_hamiltonian(trexio_file, jastrow_combo)
        self.tmp_path = tmp_path

    def test_basic_roundtrip(self):
        """Scalars, walker state, and observables survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd)
        gfmc.run(num_mcmc_steps=3)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        # Scalar configs
        assert gfmc2.mcmc_counter == gfmc.mcmc_counter
        assert gfmc2.num_walkers == gfmc.num_walkers
        assert gfmc2.alat == gfmc.alat

        # Observables
        np.testing.assert_array_equal(np.asarray(gfmc2.bare_w_L), np.asarray(gfmc.bare_w_L))

    def test_walker_state_roundtrip(self):
        """Walker coordinates survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd)
        gfmc.run(num_mcmc_steps=2)

        r_up_before = np.asarray(gfmc._GFMC_t__latest_r_up_carts)
        r_dn_before = np.asarray(gfmc._GFMC_t__latest_r_dn_carts)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        np.testing.assert_array_equal(
            np.asarray(gfmc2._GFMC_t__latest_r_up_carts),
            r_up_before,
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._GFMC_t__latest_r_dn_carts),
            r_dn_before,
        )

    def test_rng_state_roundtrip(self):
        """RNG keys survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd)
        gfmc.run(num_mcmc_steps=1)

        keys_before = np.asarray(gfmc._GFMC_t__jax_PRNG_key_list)
        keys_init_before = np.asarray(gfmc._GFMC_t__jax_PRNG_key_list_init)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        np.testing.assert_array_equal(
            np.asarray(gfmc2._GFMC_t__jax_PRNG_key_list),
            keys_before,
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._GFMC_t__jax_PRNG_key_list_init),
            keys_init_before,
        )

    def test_counters_roundtrip(self):
        """Counters (mcmc_counter, survived/killed) are preserved."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd)
        gfmc.run(num_mcmc_steps=5)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        assert gfmc2._GFMC_t__mcmc_counter == gfmc._GFMC_t__mcmc_counter
        assert gfmc2._GFMC_t__num_survived_walkers == gfmc._GFMC_t__num_survived_walkers
        assert gfmc2._GFMC_t__num_killed_walkers == gfmc._GFMC_t__num_killed_walkers

    def test_continue_run_after_load(self):
        """A loaded GFMC_t can continue running without error."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd)
        gfmc.run(num_mcmc_steps=2)

        counter_before = gfmc._GFMC_t__mcmc_counter

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        gfmc2.run(num_mcmc_steps=2)
        assert gfmc2._GFMC_t__mcmc_counter == counter_before + 2

    def test_hamiltonian_embedded(self):
        """The restored GFMC_t has a functional hamiltonian_data."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd)
        gfmc.run(num_mcmc_steps=1)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        hd = gfmc2.hamiltonian_data
        assert hd is not None
        n_up = self.hd.wavefunction_data.geminal_data.num_electron_up
        n_dn = self.hd.wavefunction_data.geminal_data.num_electron_dn
        assert hd.wavefunction_data.geminal_data.num_electron_up == n_up
        assert hd.wavefunction_data.geminal_data.num_electron_dn == n_dn

    def test_gfmc_t_specific_config_roundtrip(self):
        """GFMC_t-specific configs (tau, non_local_move, etc.) survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd, tau=0.05, alat=0.2, non_local_move="tmove")
        gfmc.run(num_mcmc_steps=1)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        assert gfmc2._GFMC_t__tau == 0.05
        assert gfmc2._GFMC_t__alat == 0.2
        assert gfmc2._GFMC_t__non_local_move == "tmove"
        assert gfmc2._GFMC_t__num_gfmc_collect_steps == gfmc._GFMC_t__num_gfmc_collect_steps

    def test_force_derivative_roundtrip(self):
        """Force observables survive round-trip when comput_position_deriv=True."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd, comput_position_deriv=True)
        gfmc.run(num_mcmc_steps=2)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        assert gfmc2.comput_position_deriv is True
        np.testing.assert_array_equal(
            np.asarray(gfmc2._GFMC_t__stored_grad_e_L_dR),
            np.asarray(gfmc._GFMC_t__stored_grad_e_L_dR),
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._GFMC_t__stored_grad_ln_Psi_dR),
            np.asarray(gfmc._GFMC_t__stored_grad_ln_Psi_dR),
        )


# ===========================================================================
# GFMC_n tests
# ===========================================================================


@pytest.mark.parametrize("trexio_file", TREXIO_FILES, ids=lambda f: f.replace(".h5", ""))
@pytest.mark.parametrize("jastrow_combo", JASTROW_COMBOS)
class TestGFMCnSaveLoadRoundtrip:
    """Round-trip tests: GFMC_n → save → merge → load → compare."""

    @pytest.fixture(autouse=True)
    def _setup(self, trexio_file, jastrow_combo, tmp_path):
        """Build hamiltonian_data once per test method."""
        self.hd = _build_hamiltonian(trexio_file, jastrow_combo)
        self.tmp_path = tmp_path

    def test_basic_roundtrip(self):
        """Scalars, walker state, and observables survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd)
        gfmc.run(num_mcmc_steps=3)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        # Scalar configs
        assert gfmc2.mcmc_counter == gfmc.mcmc_counter
        assert gfmc2.num_walkers == gfmc.num_walkers
        assert gfmc2.alat == gfmc.alat

        # Observables
        np.testing.assert_array_equal(np.asarray(gfmc2.bare_w_L), np.asarray(gfmc.bare_w_L))

    def test_walker_state_roundtrip(self):
        """Walker coordinates survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd)
        gfmc.run(num_mcmc_steps=2)

        r_up_before = np.asarray(gfmc._GFMC_n__latest_r_up_carts)
        r_dn_before = np.asarray(gfmc._GFMC_n__latest_r_dn_carts)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        np.testing.assert_array_equal(
            np.asarray(gfmc2._GFMC_n__latest_r_up_carts),
            r_up_before,
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._GFMC_n__latest_r_dn_carts),
            r_dn_before,
        )

    def test_rng_state_roundtrip(self):
        """RNG keys survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd)
        gfmc.run(num_mcmc_steps=1)

        keys_before = np.asarray(gfmc._GFMC_n__jax_PRNG_key_list)
        keys_init_before = np.asarray(gfmc._GFMC_n__jax_PRNG_key_list_init)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        np.testing.assert_array_equal(
            np.asarray(gfmc2._GFMC_n__jax_PRNG_key_list),
            keys_before,
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._GFMC_n__jax_PRNG_key_list_init),
            keys_init_before,
        )

    def test_counters_roundtrip(self):
        """Counters (mcmc_counter, survived/killed) are preserved."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd)
        gfmc.run(num_mcmc_steps=5)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        assert gfmc2._GFMC_n__mcmc_counter == gfmc._GFMC_n__mcmc_counter
        assert gfmc2._GFMC_n__num_survived_walkers == gfmc._GFMC_n__num_survived_walkers
        assert gfmc2._GFMC_n__num_killed_walkers == gfmc._GFMC_n__num_killed_walkers

    def test_continue_run_after_load(self):
        """A loaded GFMC_n can continue running without error."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd)
        gfmc.run(num_mcmc_steps=2)

        counter_before = gfmc._GFMC_n__mcmc_counter

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        gfmc2.run(num_mcmc_steps=2)
        assert gfmc2._GFMC_n__mcmc_counter == counter_before + 2

    def test_hamiltonian_embedded(self):
        """The restored GFMC_n has a functional hamiltonian_data."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd)
        gfmc.run(num_mcmc_steps=1)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        hd = gfmc2.hamiltonian_data
        assert hd is not None
        n_up = self.hd.wavefunction_data.geminal_data.num_electron_up
        n_dn = self.hd.wavefunction_data.geminal_data.num_electron_dn
        assert hd.wavefunction_data.geminal_data.num_electron_up == n_up
        assert hd.wavefunction_data.geminal_data.num_electron_dn == n_dn

    def test_gfmc_n_specific_config_roundtrip(self):
        """GFMC_n-specific configs (E_scf, num_mcmc_per_measurement, etc.) survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd, E_scf=-1.5, alat=0.2, num_mcmc_per_measurement=8)
        gfmc.run(num_mcmc_steps=1)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        assert gfmc2._GFMC_n__E_scf == -1.5
        assert gfmc2._GFMC_n__alat == 0.2
        assert gfmc2._GFMC_n__num_mcmc_per_measurement == 8
        assert gfmc2._GFMC_n__non_local_move == gfmc._GFMC_n__non_local_move
        assert gfmc2._GFMC_n__num_gfmc_collect_steps == gfmc._GFMC_n__num_gfmc_collect_steps

    def test_force_derivative_roundtrip(self):
        """Force observables survive round-trip when comput_position_deriv=True."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd, comput_position_deriv=True)
        gfmc.run(num_mcmc_steps=2)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        assert gfmc2.comput_position_deriv is True
        np.testing.assert_array_equal(
            np.asarray(gfmc2._GFMC_n__stored_grad_e_L_dR),
            np.asarray(gfmc._GFMC_n__stored_grad_e_L_dR),
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._GFMC_n__stored_grad_ln_Psi_dR),
            np.asarray(gfmc._GFMC_n__stored_grad_ln_Psi_dR),
        )
