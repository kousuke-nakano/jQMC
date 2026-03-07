"""Checkpoint utilities for saving/loading QMC driver state to HDF5.

This module provides the infrastructure for the restart system.
All QMC driver state (MCMC, GFMC_t, GFMC_n) is saved to a single
HDF5 file with per-rank groups, and hamiltonian_data stored once
at the root level.

File layout::

    restart.h5
    ├── _meta/                     (format_version, driver_type, mpi_size, ...)
    ├── hamiltonian_data/          (shared, saved once)
    └── rank_{R}/                  (per MPI rank)
        ├── driver_config/
        ├── rng_state/
        ├── walker_state/
        ├── observables/
        └── optimizer_state/       (VMCopt only)

Write flow (mirrors the legacy zip approach)::

    1. Each rank writes a temporary HDF5: ``._restart_rank{R}.h5``
    2. MPI Barrier
    3. Rank 0 merges all temporaries + hamiltonian_data into one file
    4. MPI Barrier

Read flow::

    All ranks open the same file read-only and read their own group.
"""

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

from __future__ import annotations

import datetime
import os
import pickle
from logging import getLogger
from typing import Any

import h5py
import numpy as np

from .hamiltonians import Hamiltonian_data, _save_dataclass_to_hdf5

logger = getLogger("jqmc").getChild(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_FORMAT_VERSION = "1.0"


def _get_jqmc_version() -> str:
    """Return the jQMC version string, or 'unknown' if unavailable."""
    try:
        from ._version import __version__

        return __version__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Optax state serialization (A+B hybrid)
# ---------------------------------------------------------------------------


def save_optax_state(group: h5py.Group, optax_state: Any) -> None:
    """Save optax optimizer state as pickle bytes in an HDF5 dataset.

    The state is stored as a single opaque (``np.void``) dataset so that
    the file remains a valid HDF5.  If ``optax_state`` is ``None``, nothing
    is written.

    Args:
        group: HDF5 group to write into (e.g. ``f["rank_0/optimizer_state"]``).
        optax_state: The optax optimizer state (a pytree of NamedTuples).
    """
    if optax_state is None:
        return
    raw = pickle.dumps(optax_state, protocol=pickle.HIGHEST_PROTOCOL)
    group.create_dataset("optax_state_pickle", data=np.void(raw))


def load_optax_state(group: h5py.Group) -> Any | None:
    """Load optax optimizer state from an HDF5 group.

    Attempts ``pickle.loads`` first.  If that fails (e.g. after a JAX/optax
    version change), returns ``None`` so the caller can re-initialise the
    optimizer with ``optax_tx.init(params)``.

    Args:
        group: HDF5 group that may contain ``optax_state_pickle``.

    Returns:
        The restored optax state, or ``None`` on failure / absence.
    """
    if "optax_state_pickle" not in group:
        return None
    try:
        raw = group["optax_state_pickle"][()]
        return pickle.loads(raw.tobytes())
    except Exception:
        logger.warning("Could not restore optax state from checkpoint; it will be re-initialised.")
        return None


# ---------------------------------------------------------------------------
# Per-rank checkpoint I/O (temporary file written by each rank)
# ---------------------------------------------------------------------------


def save_rank_checkpoint(
    filepath: str,
    *,
    driver_type: str,
    driver_config: dict[str, Any],
    rng_state: dict[str, Any],
    walker_state: dict[str, Any],
    observables: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    """Write one MPI rank's data to a temporary HDF5 file.

    This file is later merged by :func:`merge_rank_checkpoints` on rank 0.

    Args:
        filepath: Path to write (e.g. ``._restart_rank0.h5``).
        driver_type: ``"MCMC"``, ``"GFMC_t"``, or ``"GFMC_n"``.
        driver_config: Scalar execution parameters (seed, Dt, etc.).
        rng_state: RNG arrays and seeds.
        walker_state: Latest walker positions / weights.
        observables: Accumulated measurements (e_L, w_L, ...).
        optimizer_state: VMCopt optimizer runtime (optional).
    """
    with h5py.File(filepath, "w") as f:
        # driver_config — scalars as attrs
        cfg_grp = f.create_group("driver_config")
        for k, v in driver_config.items():
            if v is None:
                continue
            if isinstance(v, (np.ndarray,)):
                cfg_grp.create_dataset(k, data=v)
            elif isinstance(v, (int, float, bool, str)):
                cfg_grp.attrs[k] = v
            else:
                # fallback: try as attr
                try:
                    cfg_grp.attrs[k] = v
                except Exception:
                    pass

        # rng_state
        rng_grp = f.create_group("rng_state")
        for k, v in rng_state.items():
            if v is None:
                continue
            if isinstance(v, (np.ndarray,)):
                rng_grp.create_dataset(k, data=v)
            elif isinstance(v, (int, float, bool, str)):
                rng_grp.attrs[k] = v

        # walker_state
        ws_grp = f.create_group("walker_state")
        for k, v in walker_state.items():
            if v is None:
                continue
            if isinstance(v, np.ndarray):
                ws_grp.create_dataset(k, data=v)
            elif isinstance(v, (int, float, bool, str)):
                ws_grp.attrs[k] = v

        # observables
        obs_grp = f.create_group("observables")
        for k, v in observables.items():
            if v is None:
                continue
            if isinstance(v, dict):
                # param_grads: dict[str, ndarray]
                sub = obs_grp.create_group(k)
                for sk, sv in v.items():
                    if sv is not None and isinstance(sv, np.ndarray) and sv.size > 0:
                        sub.create_dataset(sk, data=sv)
            elif isinstance(v, np.ndarray) and v.size > 0:
                obs_grp.create_dataset(k, data=v)

        # optimizer_state (optional, VMCopt only)
        if optimizer_state is not None:
            opt_grp = f.create_group("optimizer_state")
            method = optimizer_state.get("method")
            if method is not None:
                opt_grp.attrs["method"] = method
            hyperparams = optimizer_state.get("hyperparameters")
            if hyperparams is not None:
                hp_grp = opt_grp.create_group("hyperparameters")
                for k, v in hyperparams.items():
                    if v is not None:
                        try:
                            hp_grp.attrs[k] = v
                        except Exception:
                            pass
            optax_st = optimizer_state.get("optax_state")
            save_optax_state(opt_grp, optax_st)
            param_size = optimizer_state.get("optax_param_size")
            if param_size is not None:
                opt_grp.attrs["optax_param_size"] = param_size


def load_rank_checkpoint(filepath: str, rank: int) -> dict[str, Any]:
    """Read one rank's data from a merged checkpoint HDF5.

    Args:
        filepath: Path to the merged checkpoint (e.g. ``restart.h5``).
        rank: The MPI rank whose data to load.

    Returns:
        A dict with keys: ``driver_config``, ``rng_state``, ``walker_state``,
        ``observables``, ``optimizer_state`` (may be ``None``).
    """
    result: dict[str, Any] = {}
    with h5py.File(filepath, "r") as f:
        rank_grp = f[f"rank_{rank}"]

        # driver_config
        cfg_grp = rank_grp["driver_config"]
        driver_config: dict[str, Any] = {}
        for k in cfg_grp.attrs:
            driver_config[k] = _convert_attr(cfg_grp.attrs[k])
        for k in cfg_grp:
            driver_config[k] = cfg_grp[k][()]
        result["driver_config"] = driver_config

        # rng_state
        rng_grp = rank_grp["rng_state"]
        rng_state: dict[str, Any] = {}
        for k in rng_grp.attrs:
            rng_state[k] = _convert_attr(rng_grp.attrs[k])
        for k in rng_grp:
            rng_state[k] = rng_grp[k][()]
        result["rng_state"] = rng_state

        # walker_state
        ws_grp = rank_grp["walker_state"]
        walker_state: dict[str, Any] = {}
        for k in ws_grp.attrs:
            walker_state[k] = _convert_attr(ws_grp.attrs[k])
        for k in ws_grp:
            walker_state[k] = ws_grp[k][()]
        result["walker_state"] = walker_state

        # observables
        obs_grp = rank_grp["observables"]
        observables: dict[str, Any] = {}
        for k in obs_grp:
            item = obs_grp[k]
            if isinstance(item, h5py.Group):
                # sub-dict (e.g. param_grads)
                sub: dict[str, np.ndarray] = {}
                for sk in item:
                    sub[sk] = item[sk][()]
                observables[k] = sub
            else:
                observables[k] = item[()]
        result["observables"] = observables

        # optimizer_state
        if "optimizer_state" in rank_grp:
            opt_grp = rank_grp["optimizer_state"]
            optimizer_state: dict[str, Any] = {
                "method": opt_grp.attrs.get("method"),
                "hyperparameters": None,
                "optax_state": load_optax_state(opt_grp),
                "optax_param_size": _convert_attr(opt_grp.attrs.get("optax_param_size")),
            }
            if "hyperparameters" in opt_grp:
                hp_grp = opt_grp["hyperparameters"]
                hyperparams = {}
                for k in hp_grp.attrs:
                    hyperparams[k] = _convert_attr(hp_grp.attrs[k])
                optimizer_state["hyperparameters"] = hyperparams
            result["optimizer_state"] = optimizer_state
        else:
            result["optimizer_state"] = None

    return result


def _convert_attr(val: Any) -> Any:
    """Convert HDF5 attribute values to native Python types."""
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


# ---------------------------------------------------------------------------
# Merge / read top-level checkpoint
# ---------------------------------------------------------------------------


def merge_rank_checkpoints(
    output_path: str,
    *,
    mpi_size: int,
    driver_type: str,
    hamiltonian_data: Hamiltonian_data,
    tmp_pattern: str = "._restart_rank{rank}.h5",
    cleanup: bool = True,
) -> None:
    """Merge per-rank temporary HDF5 files into a single checkpoint.

    Called by rank 0 after the MPI barrier.

    Args:
        output_path: Destination path (e.g. ``restart.h5``).
        mpi_size: Total number of MPI ranks.
        driver_type: ``"MCMC"``, ``"GFMC_t"``, or ``"GFMC_n"``.
        hamiltonian_data: Shared Hamiltonian data (saved once at root).
        tmp_pattern: Filename pattern with ``{rank}`` placeholder.
        cleanup: Whether to remove temporary files after merging.
    """
    # Remove existing output
    if os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(output_path, "w") as out:
        # _meta
        meta = out.create_group("_meta")
        meta.attrs["format_version"] = CHECKPOINT_FORMAT_VERSION
        meta.attrs["driver_type"] = driver_type
        meta.attrs["mpi_size"] = mpi_size
        meta.attrs["jqmc_version"] = _get_jqmc_version()
        meta.attrs["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # hamiltonian_data (saved once at root)
        ham_grp = out.create_group("hamiltonian_data")
        _save_dataclass_to_hdf5(ham_grp, hamiltonian_data)

        # Copy each rank's data
        for rank in range(mpi_size):
            tmp_path = tmp_pattern.format(rank=rank)
            with h5py.File(tmp_path, "r") as tmp:
                out.copy(tmp, f"rank_{rank}")
            if cleanup:
                os.remove(tmp_path)


def load_hamiltonian_from_checkpoint(filepath: str) -> Hamiltonian_data:
    """Load only the hamiltonian_data from a checkpoint file.

    Useful for post-processing tools that need the wavefunction but not
    the per-rank QMC state.

    Args:
        filepath: Path to the merged checkpoint HDF5.

    Returns:
        Hamiltonian_data instance.
    """
    from .hamiltonians import _load_dataclass_from_hdf5

    with h5py.File(filepath, "r") as f:
        return _load_dataclass_from_hdf5(Hamiltonian_data, f["hamiltonian_data"])


def load_checkpoint_meta(filepath: str) -> dict[str, Any]:
    """Read the ``_meta`` group from a checkpoint.

    Args:
        filepath: Path to the merged checkpoint HDF5.

    Returns:
        Dict with keys: ``format_version``, ``driver_type``, ``mpi_size``,
        ``jqmc_version``, ``timestamp``.
    """
    meta: dict[str, Any] = {}
    with h5py.File(filepath, "r") as f:
        grp = f["_meta"]
        for k in grp.attrs:
            meta[k] = _convert_attr(grp.attrs[k])
    return meta


def get_checkpoint_num_ranks(filepath: str) -> int:
    """Return the number of MPI ranks stored in a checkpoint.

    Args:
        filepath: Path to the merged checkpoint HDF5.

    Returns:
        Number of ``rank_*`` groups.
    """
    with h5py.File(filepath, "r") as f:
        return sum(1 for k in f.keys() if k.startswith("rank_"))


# ---------------------------------------------------------------------------
# Observable-only reading helpers (for post-processing)
# ---------------------------------------------------------------------------


def load_driver_config_from_checkpoint(
    filepath: str,
    rank: int = 0,
) -> dict[str, Any]:
    """Read ``driver_config`` attributes for a single rank.

    This is a lightweight way to inspect execution parameters (e.g.
    ``comput_position_deriv``, ``alat``, ``num_gfmc_collect_steps``)
    without deserialising the full driver object.

    Args:
        filepath: Path to the merged checkpoint HDF5.
        rank: MPI rank whose config to read (default 0).

    Returns:
        Dict of driver-config key→value pairs.
    """
    config: dict[str, Any] = {}
    with h5py.File(filepath, "r") as f:
        grp = f[f"rank_{rank}/driver_config"]
        for k in grp.attrs:
            config[k] = _convert_attr(grp.attrs[k])
    return config


def compute_accumulated_weights(
    w_L: np.ndarray,
    num_gfmc_collect_steps: int,
) -> np.ndarray:
    """Compute accumulated GFMC weights from raw per-step weights.

    Pure-NumPy equivalent of :func:`jqmc.jqmc_gfmc._compute_G_L`.
    For each index *i* the accumulated weight is
    ``prod(w_L[i : i + k])`` where *k* = ``num_gfmc_collect_steps``.

    Args:
        w_L: Raw weight array, shape ``(A, x)``.
        num_gfmc_collect_steps: Window length *k*.

    Returns:
        Accumulated weights, shape ``(A - k, x)``.
    """
    A, x = w_L.shape
    k = num_gfmc_collect_steps
    n_out = A - k
    G_L = np.empty((n_out, x))
    for i in range(n_out):
        G_L[i] = np.prod(w_L[i : i + k], axis=0)
    return G_L


def load_observables_from_checkpoint(
    filepath: str,
    keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load observable datasets from all ranks without unpickling driver state.

    This is the fast-path for post-processing: only the requested HDF5
    datasets are read, avoiding full object deserialisation.

    Args:
        filepath: Path to the merged checkpoint HDF5.
        keys: Observable names to load (e.g. ``["e_L", "w_L"]``).
            If ``None``, all observables are loaded.

    Returns:
        A list of dicts, one per rank, each mapping observable name to ndarray.
    """
    result = []
    with h5py.File(filepath, "r") as f:
        num_ranks = sum(1 for k in f.keys() if k.startswith("rank_"))
        for rank in range(num_ranks):
            obs_grp = f[f"rank_{rank}/observables"]
            rank_obs: dict[str, Any] = {}
            iter_keys = keys if keys is not None else list(obs_grp.keys())
            for k in iter_keys:
                if k in obs_grp:
                    item = obs_grp[k]
                    if isinstance(item, h5py.Group):
                        sub: dict[str, np.ndarray] = {}
                        for sk in item:
                            sub[sk] = item[sk][()]
                        rank_obs[k] = sub
                    else:
                        rank_obs[k] = item[()]
            result.append(rank_obs)
    return result


# ---------------------------------------------------------------------------
# Version check helpers
# ---------------------------------------------------------------------------


def check_checkpoint_version(filepath: str) -> None:
    """Validate checkpoint format version, raising on incompatibility.

    Args:
        filepath: Path to a checkpoint HDF5.

    Raises:
        ValueError: If the format version is unsupported.
    """
    meta = load_checkpoint_meta(filepath)
    version = meta.get("format_version", "unknown")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format version mismatch: file has '{version}', "
            f"but this version of jQMC expects '{CHECKPOINT_FORMAT_VERSION}'."
        )
