"""Tests for jqmc.checkpoint — Phase 0 infrastructure."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    _convert_attr,
    check_checkpoint_version,
    get_checkpoint_num_ranks,
    load_checkpoint_meta,
    load_observables_from_checkpoint,
    load_optax_state,
    load_rank_checkpoint,
    merge_rank_checkpoints,
    save_optax_state,
    save_rank_checkpoint,
)

# ---------------------------------------------------------------------------
# Optax state A+B hybrid tests
# ---------------------------------------------------------------------------


class TestOptaxStateSerialisation:
    """Tests for save_optax_state / load_optax_state."""

    def test_roundtrip_dict_state(self, tmp_path):
        """A simple dict (mimicking a trivial optax state) survives roundtrip."""
        import h5py

        fake_state = {
            "count": np.array(42),
            "mu": np.zeros(5),
            "nu": np.ones(5),
        }
        filepath = str(tmp_path / "opt.h5")
        with h5py.File(filepath, "w") as f:
            grp = f.create_group("optimizer_state")
            save_optax_state(grp, fake_state)

        with h5py.File(filepath, "r") as f:
            grp = f["optimizer_state"]
            restored = load_optax_state(grp)

        assert restored is not None
        assert restored["count"] == 42
        np.testing.assert_array_equal(restored["mu"], np.zeros(5))
        np.testing.assert_array_equal(restored["nu"], np.ones(5))

    def test_none_state_skipped(self, tmp_path):
        """save_optax_state(None) writes nothing; load returns None."""
        import h5py

        filepath = str(tmp_path / "opt.h5")
        with h5py.File(filepath, "w") as f:
            grp = f.create_group("optimizer_state")
            save_optax_state(grp, None)

        with h5py.File(filepath, "r") as f:
            grp = f["optimizer_state"]
            assert load_optax_state(grp) is None

    def test_fallback_on_corrupt_pickle(self, tmp_path):
        """If the pickle bytes are corrupt, load returns None (fallback A)."""
        import h5py

        filepath = str(tmp_path / "opt.h5")
        with h5py.File(filepath, "w") as f:
            grp = f.create_group("optimizer_state")
            grp.create_dataset("optax_state_pickle", data=np.void(b"not-valid-pickle"))

        with h5py.File(filepath, "r") as f:
            grp = f["optimizer_state"]
            assert load_optax_state(grp) is None

    def test_real_optax_state(self, tmp_path):
        """Round-trip a real optax Adam state."""
        try:
            import jax
            import jax.numpy as jnp
            import optax
        except ImportError:
            pytest.skip("optax/jax not available")

        import h5py

        jax.config.update("jax_enable_x64", True)

        tx = optax.adam(1e-3)
        params = jnp.ones(10)
        state = tx.init(params)

        filepath = str(tmp_path / "optax.h5")
        with h5py.File(filepath, "w") as f:
            grp = f.create_group("optimizer_state")
            save_optax_state(grp, state)

        with h5py.File(filepath, "r") as f:
            grp = f["optimizer_state"]
            restored = load_optax_state(grp)

        assert restored is not None
        # Compare leaves
        import jax.tree_util as tu

        orig_leaves = tu.tree_leaves(state)
        rest_leaves = tu.tree_leaves(restored)
        assert len(orig_leaves) == len(rest_leaves)
        for a, b in zip(orig_leaves, rest_leaves, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# Per-rank save / load tests
# ---------------------------------------------------------------------------


class TestRankCheckpoint:
    """Tests for save_rank_checkpoint / load_rank_checkpoint."""

    @staticmethod
    def _make_sample_data():
        """Create a minimal set of per-rank data dicts."""
        return dict(
            driver_type="MCMC",
            driver_config={
                "mcmc_seed": 12345,
                "num_walkers": 4,
                "Dt": 2.0,
                "epsilon_AS": 1e-1,
                "num_mcmc_per_measurement": 16,
            },
            rng_state={
                "jax_PRNG_key_list": np.arange(8, dtype=np.uint32).reshape(4, 2),
                "mpi_seed": 12345,
            },
            walker_state={
                "latest_r_up_carts": np.random.randn(4, 3, 3),
                "latest_r_dn_carts": np.random.randn(4, 2, 3),
            },
            observables={
                "e_L": np.random.randn(100, 4),
                "e_L2": np.random.randn(100, 4),
                "w_L": np.ones((100, 4)),
                "param_grads": {
                    "j1_param": np.random.randn(100, 4, 5),
                },
            },
        )

    def test_save_load_roundtrip(self, tmp_path):
        """Data survives save → merge → load."""
        data = self._make_sample_data()
        rank_file = str(tmp_path / "._restart_rank0.h5")
        save_rank_checkpoint(rank_file, **data)

        # Check file was created
        assert os.path.exists(rank_file)

        # Now merge into a single checkpoint (simulate rank 0)
        from jqmc._checkpoint import merge_rank_checkpoints

        # We need a Hamiltonian_data for merging — use a minimal mock
        merged = str(tmp_path / "restart.h5")
        _merge_single_rank(rank_file, merged, tmp_path)

        # Load back
        loaded = load_rank_checkpoint(merged, rank=0)

        # Verify driver_config
        assert loaded["driver_config"]["mcmc_seed"] == 12345
        assert loaded["driver_config"]["num_walkers"] == 4
        assert loaded["driver_config"]["Dt"] == 2.0

        # Verify rng_state
        np.testing.assert_array_equal(
            loaded["rng_state"]["jax_PRNG_key_list"],
            data["rng_state"]["jax_PRNG_key_list"],
        )
        assert loaded["rng_state"]["mpi_seed"] == 12345

        # Verify walker_state
        np.testing.assert_array_equal(
            loaded["walker_state"]["latest_r_up_carts"],
            data["walker_state"]["latest_r_up_carts"],
        )

        # Verify observables
        np.testing.assert_array_equal(loaded["observables"]["e_L"], data["observables"]["e_L"])
        np.testing.assert_array_equal(
            loaded["observables"]["param_grads"]["j1_param"],
            data["observables"]["param_grads"]["j1_param"],
        )

    def test_with_optimizer_state(self, tmp_path):
        """Optimizer state (method + hyperparams + optax pickle) roundtrips."""
        data = self._make_sample_data()
        data["optimizer_state"] = {
            "method": "adam",
            "hyperparameters": {"learning_rate": 1e-3, "b1": 0.9, "b2": 0.999},
            "optax_state": {"count": np.array(10), "mu": np.zeros(5)},
            "optax_param_size": 100,
        }
        rank_file = str(tmp_path / "._restart_rank0.h5")
        save_rank_checkpoint(rank_file, **data)

        merged = str(tmp_path / "restart.h5")
        _merge_single_rank(rank_file, merged, tmp_path)

        loaded = load_rank_checkpoint(merged, rank=0)
        opt = loaded["optimizer_state"]
        assert opt is not None
        assert opt["method"] == "adam"
        assert opt["hyperparameters"]["learning_rate"] == 1e-3
        assert opt["optax_param_size"] == 100
        assert opt["optax_state"]["count"] == 10

    def test_multi_rank_merge(self, tmp_path):
        """Two ranks can be merged and read independently."""
        for rank in range(2):
            data = self._make_sample_data()
            data["driver_config"]["mcmc_seed"] = 1000 + rank
            rank_file = str(tmp_path / f"._restart_rank{rank}.h5")
            save_rank_checkpoint(rank_file, **data)

        merged = str(tmp_path / "restart.h5")
        _merge_multi_rank(2, merged, tmp_path)

        assert get_checkpoint_num_ranks(merged) == 2

        for rank in range(2):
            loaded = load_rank_checkpoint(merged, rank=rank)
            assert loaded["driver_config"]["mcmc_seed"] == 1000 + rank


# ---------------------------------------------------------------------------
# Meta / version tests
# ---------------------------------------------------------------------------


class TestMetaAndVersion:
    """Tests for _meta group and version checking."""

    def test_meta_fields(self, tmp_path):
        """Merged checkpoint has correct _meta attributes."""
        data = TestRankCheckpoint._make_sample_data()
        rank_file = str(tmp_path / "._restart_rank0.h5")
        save_rank_checkpoint(rank_file, **data)

        merged = str(tmp_path / "restart.h5")
        _merge_single_rank(rank_file, merged, tmp_path)

        meta = load_checkpoint_meta(merged)
        assert meta["format_version"] == CHECKPOINT_FORMAT_VERSION
        assert meta["driver_type"] == "MCMC"
        assert meta["mpi_size"] == 1
        assert "timestamp" in meta
        assert "jqmc_version" in meta

    def test_version_check_passes(self, tmp_path):
        """check_checkpoint_version succeeds for current version."""
        data = TestRankCheckpoint._make_sample_data()
        rank_file = str(tmp_path / "._restart_rank0.h5")
        save_rank_checkpoint(rank_file, **data)

        merged = str(tmp_path / "restart.h5")
        _merge_single_rank(rank_file, merged, tmp_path)

        check_checkpoint_version(merged)  # should not raise

    def test_version_check_fails(self, tmp_path):
        """check_checkpoint_version raises on version mismatch."""
        import h5py

        filepath = str(tmp_path / "bad.h5")
        with h5py.File(filepath, "w") as f:
            meta = f.create_group("_meta")
            meta.attrs["format_version"] = "99.99"

        with pytest.raises(ValueError, match="format version mismatch"):
            check_checkpoint_version(filepath)


# ---------------------------------------------------------------------------
# Observable-only loading tests
# ---------------------------------------------------------------------------


class TestLoadObservables:
    """Tests for load_observables_from_checkpoint."""

    def test_load_selected_keys(self, tmp_path):
        """Only requested observable keys are loaded."""
        data = TestRankCheckpoint._make_sample_data()
        rank_file = str(tmp_path / "._restart_rank0.h5")
        save_rank_checkpoint(rank_file, **data)

        merged = str(tmp_path / "restart.h5")
        _merge_single_rank(rank_file, merged, tmp_path)

        obs_list = load_observables_from_checkpoint(merged, keys=["e_L", "w_L"])
        assert len(obs_list) == 1
        assert "e_L" in obs_list[0]
        assert "w_L" in obs_list[0]
        assert "e_L2" not in obs_list[0]

    def test_load_all_keys(self, tmp_path):
        """All observables loaded when keys=None."""
        data = TestRankCheckpoint._make_sample_data()
        rank_file = str(tmp_path / "._restart_rank0.h5")
        save_rank_checkpoint(rank_file, **data)

        merged = str(tmp_path / "restart.h5")
        _merge_single_rank(rank_file, merged, tmp_path)

        obs_list = load_observables_from_checkpoint(merged)
        assert len(obs_list) == 1
        assert "e_L" in obs_list[0]
        assert "e_L2" in obs_list[0]
        assert "w_L" in obs_list[0]


# ---------------------------------------------------------------------------
# Helpers for _convert_attr
# ---------------------------------------------------------------------------


class TestConvertAttr:
    def test_bytes_to_str(self):
        assert _convert_attr(b"hello") == "hello"

    def test_numpy_int(self):
        result = _convert_attr(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float(self):
        result = _convert_attr(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_numpy_bool(self):
        result = _convert_attr(np.bool_(True))
        assert result is True

    def test_none(self):
        assert _convert_attr(None) is None


# ---------------------------------------------------------------------------
# Test utilities — minimal merge helpers that don't require Hamiltonian_data
# ---------------------------------------------------------------------------


def _merge_single_rank(rank_file: str, merged_path: str, tmp_path) -> None:
    """Merge a single rank file with a dummy hamiltonian_data group."""
    import h5py

    if os.path.exists(merged_path):
        os.remove(merged_path)
    with h5py.File(merged_path, "w") as out:
        meta = out.create_group("_meta")
        meta.attrs["format_version"] = CHECKPOINT_FORMAT_VERSION
        meta.attrs["driver_type"] = "MCMC"
        meta.attrs["mpi_size"] = 1
        meta.attrs["jqmc_version"] = "test"
        meta.attrs["timestamp"] = "2026-03-05T00:00:00+00:00"

        # dummy hamiltonian_data group
        out.create_group("hamiltonian_data")

        with h5py.File(rank_file, "r") as tmp:
            out.copy(tmp, "rank_0")

    # cleanup tmp
    if os.path.exists(rank_file):
        os.remove(rank_file)


def _merge_multi_rank(num_ranks: int, merged_path: str, tmp_path) -> None:
    """Merge multiple rank files with a dummy hamiltonian_data group."""
    import h5py

    if os.path.exists(merged_path):
        os.remove(merged_path)
    with h5py.File(merged_path, "w") as out:
        meta = out.create_group("_meta")
        meta.attrs["format_version"] = CHECKPOINT_FORMAT_VERSION
        meta.attrs["driver_type"] = "MCMC"
        meta.attrs["mpi_size"] = num_ranks
        meta.attrs["jqmc_version"] = "test"
        meta.attrs["timestamp"] = "2026-03-05T00:00:00+00:00"

        out.create_group("hamiltonian_data")

        for rank in range(num_ranks):
            rank_file = str(tmp_path / f"._restart_rank{rank}.h5")
            with h5py.File(rank_file, "r") as tmp:
                out.copy(tmp, f"rank_{rank}")
            os.remove(rank_file)
