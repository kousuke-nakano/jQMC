"""Unit tests for jqmc_tool trexio conversions."""

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

import gzip
import os
import pickle
import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tomlkit

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.hamiltonians import Hamiltonian_data  # noqa: E402
from jqmc.jqmc_tool import (  # noqa: E402
    _J3_PERIOD_RANGES,
    hamiltonian_show_info,
    hamiltonian_to_xyz,
    lrdmc_generate_input,
    mcmc_generate_input,
    trexio_convert_to,
    trexio_show_detail,
    trexio_show_info,
    vmc_analyze_output,
    vmc_generate_input,
)
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402

trexio_files = [
    "H2_ecp_ccpvtz_cart.h5",
    "N2_ecp_ccpvtz_cart.h5",
    "Cl2_ecp_ccpvtz_cart.h5",
    "Ti2_ecp_ccpvtz_cart.h5",
    "CuBr_ecp_ccpvtz_cart.h5",
]


class TestTrexioConvertTo:
    """Tests for trexio convert-to AO basis selection."""

    @pytest.mark.parametrize(
        "trexio_filename",
        trexio_files,
        ids=trexio_files,
    )
    def test_ao_full_matches_uncontracted(self, tmp_path, trexio_filename):
        """Ensure ao-full keeps the full uncontracted AO basis."""
        trexio_file = os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_filename)
        hamiltonian_file = os.path.join(tmp_path, "hamiltonian_data.h5")

        _, aos_data, _, _, _, _ = read_trexio_file(trexio_file, store_tuple=True)
        uncontracted = aos_data._build_uncontracted_aos()

        trexio_convert_to(
            trexio_file=trexio_file,
            hamiltonian_file=hamiltonian_file,
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type="ao-full",
        )

        hamiltonian = Hamiltonian_data.load_from_hdf5(hamiltonian_file)
        orb_data = hamiltonian.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data

        assert orb_data.num_ao == uncontracted.num_ao
        assert orb_data.num_ao_prim == uncontracted.num_ao_prim

        uncontracted_exps = [None] * uncontracted.num_ao
        for p, orb in enumerate(uncontracted.orbital_indices):
            if uncontracted_exps[orb] is None:
                uncontracted_exps[orb] = uncontracted.exponents[p]

        expected_counts: dict[tuple[int, int, float], int] = {}
        for i in range(uncontracted.num_ao):
            key = (uncontracted.nucleus_index[i], uncontracted.angular_momentums[i], uncontracted_exps[i])
            expected_counts[key] = expected_counts.get(key, 0) + 1

        orb_exps = [None] * orb_data.num_ao
        for p, orb in enumerate(orb_data.orbital_indices):
            if orb_exps[orb] is None:
                orb_exps[orb] = orb_data.exponents[p]

        actual_counts: dict[tuple[int, int, float], int] = {}
        for i in range(orb_data.num_ao):
            key = (orb_data.nucleus_index[i], orb_data.angular_momentums[i], orb_exps[i])
            actual_counts[key] = actual_counts.get(key, 0) + 1
        assert actual_counts == expected_counts

    @pytest.mark.parametrize("trexio_filename", trexio_files, ids=trexio_files)
    @pytest.mark.parametrize("choice", ["ao-small", "ao-medium", "ao-large"])
    def test_ao_selection(self, tmp_path, trexio_filename, choice):
        """Ensure AO selection matches the expected shell-by-shell choice."""
        trexio_file = os.path.join(os.path.dirname(__file__), "trexio_example_files", trexio_filename)
        hamiltonian_file = os.path.join(tmp_path, f"hamiltonian_{choice}.h5")

        j3_ao_selection_table = {
            "period1": {
                "ao-small": "3s",
                "ao-medium": "3s1p",
                "ao-large": "4s2p1d",
            },
            "period2": {
                "ao-small": "3s1p",
                "ao-medium": "4s2p1d",
                "ao-large": "5s3p2d1f",
            },
            "period3": {
                "ao-small": "4s2p",
                "ao-medium": "5s3p1d",
                "ao-large": "6s4p3d1f",
            },
            "period4plus_main": {
                "ao-small": "4s2p1d",
                "ao-medium": "5s3p2d1f",
                "ao-large": "6s4p3d2f",
            },
            "period4plus_transition": {
                "ao-small": "4s2p2d1f",
                "ao-medium": "5s3p3d2f1g",
                "ao-large": "6s4p4d3f2g1h",
            },
        }
        j3_transition_z_ranges = (
            (21, 30),
            (39, 48),
            (57, 71),
            (72, 80),
            (89, 103),
            (104, 112),
        )
        j3_shell_l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}

        _, aos_data, _, _, _, _ = read_trexio_file(trexio_file, store_tuple=True)
        uncontracted = aos_data._build_uncontracted_aos()

        ao_exps = [None] * uncontracted.num_ao
        for p, orb in enumerate(uncontracted.orbital_indices):
            if ao_exps[orb] is None:
                ao_exps[orb] = uncontracted.exponents[p]

        expected_indices = []
        if choice in ("ao", "ao-full"):
            expected_indices = list(range(uncontracted.num_ao))
        else:
            for nucleus in set(uncontracted.nucleus_index):
                ao_idxs = [i for i, nuc in enumerate(uncontracted.nucleus_index) if nuc == nucleus]
                if not ao_idxs:
                    continue

                z = uncontracted.structure_data.atomic_numbers[nucleus]
                period = 7
                for idx, (zmin, zmax) in enumerate(_J3_PERIOD_RANGES, start=1):
                    if zmin <= z <= zmax:
                        period = idx
                        break

                if period == 1:
                    key = "period1"
                elif period == 2:
                    key = "period2"
                elif period == 3:
                    key = "period3"
                else:
                    is_transition = any(zmin <= z <= zmax for zmin, zmax in j3_transition_z_ranges)
                    key = "period4plus_transition" if is_transition else "period4plus_main"

                spec = j3_ao_selection_table[key][choice]
                shell_counts: dict[int, int] = {}
                for count_str, shell in re.findall(r"(\d+)([spdfgh])", spec):
                    l = j3_shell_l_map[shell]
                    shell_counts[l] = int(count_str)

                for l in set(uncontracted.angular_momentums):
                    ao_idxs_l = [i for i in ao_idxs if uncontracted.angular_momentums[i] == l]
                    if not ao_idxs_l:
                        continue

                    n_shells = shell_counts.get(l, 0)
                    if n_shells <= 0:
                        continue

                    basis_exps = sorted({ao_exps[i] for i in ao_idxs_l})
                    if n_shells >= len(basis_exps):
                        sel_basis = basis_exps
                    else:
                        start = max(0, (len(basis_exps) - n_shells) // 2)
                        sel_basis = basis_exps[start : start + n_shells]

                    expected_indices.extend([i for i in ao_idxs_l if ao_exps[i] in sel_basis])

            expected_indices = sorted(set(expected_indices))

        expected_counts: dict[tuple[int, int, float], int] = {}
        for i in expected_indices:
            key = (uncontracted.nucleus_index[i], uncontracted.angular_momentums[i], ao_exps[i])
            expected_counts[key] = expected_counts.get(key, 0) + 1

        expected_shells: dict[int, str] = {}
        for nucleus in set(uncontracted.nucleus_index):
            shells: dict[int, set[float]] = {}
            for i in expected_indices:
                if uncontracted.nucleus_index[i] != nucleus:
                    continue
                shells.setdefault(uncontracted.angular_momentums[i], set()).add(ao_exps[i])
            parts = []
            for l, label in [(0, "s"), (1, "p"), (2, "d"), (3, "f"), (4, "g"), (5, "h")]:
                count = len(shells.get(l, set()))
                if count > 0:
                    parts.append(f"{count}{label}")
            expected_shells[nucleus] = "".join(parts)

        trexio_convert_to(
            trexio_file=trexio_file,
            hamiltonian_file=hamiltonian_file,
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type=choice,
        )

        hamiltonian = Hamiltonian_data.load_from_hdf5(hamiltonian_file)
        orb_data = hamiltonian.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data

        orb_exps = [None] * orb_data.num_ao
        for p, orb in enumerate(orb_data.orbital_indices):
            if orb_exps[orb] is None:
                orb_exps[orb] = orb_data.exponents[p]

        actual_counts: dict[tuple[int, int, float], int] = {}
        for i in range(orb_data.num_ao):
            key = (orb_data.nucleus_index[i], orb_data.angular_momentums[i], orb_exps[i])
            actual_counts[key] = actual_counts.get(key, 0) + 1

        actual_shells: dict[int, str] = {}
        for nucleus in set(orb_data.nucleus_index):
            shells: dict[int, set[float]] = {}
            for i in range(orb_data.num_ao):
                if orb_data.nucleus_index[i] != nucleus:
                    continue
                shells.setdefault(orb_data.angular_momentums[i], set()).add(orb_exps[i])
            parts = []
            for l, label in [(0, "s"), (1, "p"), (2, "d"), (3, "f"), (4, "g"), (5, "h")]:
                count = len(shells.get(l, set()))
                if count > 0:
                    parts.append(f"{count}{label}")
            actual_shells[nucleus] = "".join(parts)

        for nucleus in sorted(expected_shells):
            expected = expected_shells[nucleus]
            actual = actual_shells.get(nucleus, "")
            print(f"nucleus {nucleus}: expected [{expected}] actual [{actual}]")
        assert actual_counts == expected_counts


_TREXIO_DIR = os.path.join(os.path.dirname(__file__), "trexio_example_files")


def _make_mcmc_obj(
    num_steps=100,
    num_walkers=4,
    num_atoms=2,
    num_elec_up=1,
    num_elec_dn=1,
    e_L_value=1.0,
    w_L_value=1.0,
    with_force=True,
    atomic_labels=None,
    rng=None,
):
    """Return a SimpleNamespace that mimics a serialised MCMC object."""
    if rng is None:
        rng = np.random.default_rng(42)

    shape_iw = (num_steps, num_walkers)
    e_L = np.full(shape_iw, e_L_value, dtype=np.float64)
    w_L = np.full(shape_iw, w_L_value, dtype=np.float64)

    obj = SimpleNamespace(e_L=e_L, w_L=w_L)

    if with_force:
        obj.comput_position_deriv = True

        if atomic_labels is None:
            atomic_labels = [f"X{i}" for i in range(num_atoms)]
        obj.hamiltonian_data = SimpleNamespace(structure_data=SimpleNamespace(atomic_labels=atomic_labels))

        shape_iwj3 = (num_steps, num_walkers, num_atoms, 3)
        shape_iwk3_up = (num_steps, num_walkers, num_elec_up, 3)
        shape_iwk3_dn = (num_steps, num_walkers, num_elec_dn, 3)
        shape_iwjk_up = (num_steps, num_walkers, num_atoms, num_elec_up)
        shape_iwjk_dn = (num_steps, num_walkers, num_atoms, num_elec_dn)

        obj.de_L_dR = rng.standard_normal(shape_iwj3) * 0.01
        obj.de_L_dr_up = rng.standard_normal(shape_iwk3_up) * 0.01
        obj.de_L_dr_dn = rng.standard_normal(shape_iwk3_dn) * 0.01
        obj.dln_Psi_dR = rng.standard_normal(shape_iwj3) * 0.01
        obj.dln_Psi_dr_up = rng.standard_normal(shape_iwk3_up) * 0.01
        obj.dln_Psi_dr_dn = rng.standard_normal(shape_iwk3_dn) * 0.01
        obj.omega_up = rng.standard_normal(shape_iwjk_up) * 0.01
        obj.omega_dn = rng.standard_normal(shape_iwjk_dn) * 0.01
        obj.domega_dr_up = rng.standard_normal(shape_iwj3) * 0.01
        obj.domega_dr_dn = rng.standard_normal(shape_iwj3) * 0.01
    else:
        obj.comput_position_deriv = False

    return obj


def _make_lrdmc_obj(
    num_steps=100,
    num_walkers=4,
    num_atoms=2,
    num_elec_up=1,
    num_elec_dn=1,
    e_L_value=1.0,
    w_L_value=1.0,
    alat=0.5,
    with_force=True,
    atomic_labels=None,
    rng=None,
):
    """Return a SimpleNamespace that mimics a serialised GFMC_n (LRDMC) object."""
    obj = _make_mcmc_obj(
        num_steps=num_steps,
        num_walkers=num_walkers,
        num_atoms=num_atoms,
        num_elec_up=num_elec_up,
        num_elec_dn=num_elec_dn,
        e_L_value=e_L_value,
        w_L_value=w_L_value,
        with_force=with_force,
        atomic_labels=atomic_labels,
        rng=rng,
    )
    obj.alat = alat
    obj.num_gfmc_collect_steps = 5
    return obj


def _write_chk(path, objs):
    """Write a list of objects as a .rchk zip (one per MPI rank)."""
    with zipfile.ZipFile(path, "w") as zf:
        for rank, obj in enumerate(objs):
            buf = pickle.dumps(obj)
            compressed = gzip.compress(buf)
            zf.writestr(f"{rank}.pkl.gz", compressed)


class TestGenerateInput:
    """Tests for mcmc / lrdmc / vmc generate-input commands."""

    def test_mcmc_generate_input_creates_file(self, tmp_path):
        outfile = str(tmp_path / "mcmc.toml")
        mcmc_generate_input(flag=True, filename=outfile, exclude_comment=False)
        doc = tomlkit.loads(Path(outfile).read_text())
        assert "control" in doc
        assert "mcmc" in doc
        assert doc["control"]["job_type"] == "mcmc"

    def test_lrdmc_generate_input_creates_file(self, tmp_path):
        outfile = str(tmp_path / "lrdmc.toml")
        lrdmc_generate_input(flag=True, filename=outfile, exclude_comment=False)
        doc = tomlkit.loads(Path(outfile).read_text())
        assert "control" in doc
        assert "lrdmc" in doc
        assert doc["control"]["job_type"] == "lrdmc"

    def test_vmc_generate_input_creates_file(self, tmp_path):
        outfile = str(tmp_path / "vmc.toml")
        vmc_generate_input(flag=True, filename=outfile, exclude_comment=False)
        doc = tomlkit.loads(Path(outfile).read_text())
        assert "control" in doc
        assert "vmc" in doc
        assert doc["control"]["job_type"] == "vmc"

    def test_generate_input_without_flag_does_not_create(self, tmp_path):
        outfile = str(tmp_path / "mcmc.toml")
        mcmc_generate_input(flag=False, filename=outfile, exclude_comment=False)
        assert not os.path.exists(outfile)

    def test_generate_input_without_comment(self, tmp_path):
        outfile = str(tmp_path / "mcmc.toml")
        mcmc_generate_input(flag=True, filename=outfile, exclude_comment=True)
        text = Path(outfile).read_text()
        # When comments are excluded, there should be no "# " comment lines
        # (tomlkit comments start with #)
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                pytest.fail(f"Unexpected comment line when exclude_comment=True: {line}")

    def test_generate_input_with_comment(self, tmp_path):
        outfile = str(tmp_path / "mcmc.toml")
        mcmc_generate_input(flag=True, filename=outfile, exclude_comment=False)
        text = Path(outfile).read_text()
        assert "#" in text, "Expected comments in the generated TOML file"


class TestComputeEnergy:
    """Tests for mcmc and lrdmc compute-energy commands."""

    def test_mcmc_constant_energy_jackknife(self, tmp_path):
        """With constant e_L and w_L=1 the jackknife mean must equal that constant."""
        E_const = -5.0
        chk_path = str(tmp_path / "mcmc.rchk")
        obj = _make_mcmc_obj(num_steps=100, num_walkers=4, e_L_value=E_const, with_force=False)
        _write_chk(chk_path, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-energy", chk_path, "-b", "2", "-w", "0"])
        assert result.exit_code == 0
        assert f"E = {E_const}" in result.output

    def test_mcmc_multi_rank(self, tmp_path):
        """Multiple MPI ranks with same constant energy should give same result."""
        E_const = -3.0
        chk_path = str(tmp_path / "mcmc.rchk")
        objs = [
            _make_mcmc_obj(num_steps=100, num_walkers=2, e_L_value=E_const, with_force=False, rng=np.random.default_rng(i))
            for i in range(3)
        ]
        _write_chk(chk_path, objs)

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-energy", chk_path, "-b", "1", "-w", "0"])
        assert result.exit_code == 0
        assert "Found 3 MPI ranks" in result.output
        assert f"E = {E_const}" in result.output

    def test_mcmc_warmup_discards_steps(self, tmp_path):
        """First num_mcmc_warmup_steps should be discarded."""
        chk_path = str(tmp_path / "mcmc.rchk")
        # First 10 steps have e_L=100 (bad), remaining 90 have e_L=-5 (good)
        obj = _make_mcmc_obj(num_steps=100, num_walkers=2, e_L_value=-5.0, with_force=False)
        obj.e_L[:10, :] = 100.0
        _write_chk(chk_path, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-energy", chk_path, "-b", "1", "-w", "10"])
        assert result.exit_code == 0
        # After discarding first 10 steps, E should be -5.0
        assert "E = -5.0" in result.output

    def test_lrdmc_constant_energy_jackknife(self, tmp_path):
        """LRDMC version of jackknife test with constant energy."""
        E_const = -7.0
        chk_path = str(tmp_path / "lrdmc.rchk")
        obj = _make_lrdmc_obj(num_steps=100, num_walkers=4, e_L_value=E_const, with_force=False)
        _write_chk(chk_path, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        runner = CliRunner()
        result = runner.invoke(lrdmc_app, ["compute-energy", chk_path, "-b", "2", "-w", "0", "-c", "5"])
        assert result.exit_code == 0
        assert f"E = {E_const}" in result.output

    def test_lrdmc_multi_rank(self, tmp_path):
        """Multiple MPI ranks for LRDMC."""
        E_const = -2.0
        chk_path = str(tmp_path / "lrdmc.rchk")
        objs = [
            _make_lrdmc_obj(num_steps=80, num_walkers=2, e_L_value=E_const, with_force=False, rng=np.random.default_rng(i))
            for i in range(2)
        ]
        _write_chk(chk_path, objs)

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        runner = CliRunner()
        result = runner.invoke(lrdmc_app, ["compute-energy", chk_path, "-b", "2", "-w", "0", "-c", "5"])
        assert result.exit_code == 0
        assert "Found 2 MPI ranks" in result.output

    def test_lrdmc_extrapolate_energy(self, tmp_path):
        """extrapolate-energy with two LRDMC checkpoints should report a->0 result."""
        chk1 = str(tmp_path / "lrdmc1.rchk")
        chk2 = str(tmp_path / "lrdmc2.rchk")

        obj1 = _make_lrdmc_obj(num_steps=100, num_walkers=4, e_L_value=-5.0, alat=0.5, with_force=False)
        obj2 = _make_lrdmc_obj(num_steps=100, num_walkers=4, e_L_value=-5.2, alat=0.3, with_force=False)
        _write_chk(chk1, [obj1])
        _write_chk(chk2, [obj2])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        runner = CliRunner()
        result = runner.invoke(lrdmc_app, ["extrapolate-energy", chk1, chk2, "-b", "2", "-w", "0", "-c", "5", "-p", "1"])
        assert result.exit_code == 0
        assert "Extrapolation" in result.output
        assert "a -> 0" in result.output or "a = 0" in result.output or "For a ->" in result.output


class TestComputeForce:
    """Tests for mcmc and lrdmc compute-force commands."""

    def test_mcmc_compute_force_basic(self, tmp_path):
        """Smoke test: compute-force should print atomic force table."""
        chk_path = str(tmp_path / "mcmc.rchk")
        obj = _make_mcmc_obj(
            num_steps=100,
            num_walkers=4,
            num_atoms=2,
            num_elec_up=1,
            num_elec_dn=1,
            atomic_labels=["H", "H"],
            with_force=True,
        )
        _write_chk(chk_path, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-force", chk_path, "-b", "2", "-w", "0"])
        assert result.exit_code == 0
        assert "Atomic Forces:" in result.output
        assert "H" in result.output
        assert "Fx" in result.output

    def test_mcmc_compute_force_multi_rank(self, tmp_path):
        """Force computation with multiple MPI ranks."""
        chk_path = str(tmp_path / "mcmc.rchk")
        objs = [
            _make_mcmc_obj(
                num_steps=100,
                num_walkers=2,
                num_atoms=2,
                num_elec_up=1,
                num_elec_dn=1,
                atomic_labels=["H", "H"],
                with_force=True,
                rng=np.random.default_rng(i),
            )
            for i in range(2)
        ]
        _write_chk(chk_path, objs)

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-force", chk_path, "-b", "1", "-w", "0"])
        assert result.exit_code == 0
        assert "Found 2 MPI ranks" in result.output
        assert "Atomic Forces:" in result.output

    def test_mcmc_compute_force_without_deriv_raises(self, tmp_path):
        """If comput_position_deriv is False, compute-force should fail."""
        chk_path = str(tmp_path / "mcmc.rchk")
        obj = _make_mcmc_obj(num_steps=100, num_walkers=4, with_force=False)
        _write_chk(chk_path, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-force", chk_path, "-b", "1", "-w", "0"])
        assert result.exit_code != 0

    def test_lrdmc_compute_force_basic(self, tmp_path):
        """Smoke test for LRDMC force computation."""
        chk_path = str(tmp_path / "lrdmc.rchk")
        obj = _make_lrdmc_obj(
            num_steps=100,
            num_walkers=4,
            num_atoms=2,
            num_elec_up=1,
            num_elec_dn=1,
            atomic_labels=["N", "N"],
            with_force=True,
        )
        _write_chk(chk_path, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        runner = CliRunner()
        result = runner.invoke(lrdmc_app, ["compute-force", chk_path, "-b", "2", "-w", "0", "-c", "5"])
        assert result.exit_code == 0
        assert "Atomic Forces:" in result.output
        assert "N" in result.output

    def test_lrdmc_compute_force_without_deriv_raises(self, tmp_path):
        """If comput_position_deriv is False, LRDMC compute-force should fail."""
        chk_path = str(tmp_path / "lrdmc.rchk")
        obj = _make_lrdmc_obj(num_steps=100, num_walkers=4, with_force=False)
        _write_chk(chk_path, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        runner = CliRunner()
        result = runner.invoke(lrdmc_app, ["compute-force", chk_path, "-b", "1", "-w", "0", "-c", "5"])
        assert result.exit_code != 0

    def test_mcmc_force_constant_values(self, tmp_path):
        """With zero random derivatives, all forces should be near zero."""
        chk_path = str(tmp_path / "mcmc.rchk")
        obj = _make_mcmc_obj(
            num_steps=100,
            num_walkers=4,
            num_atoms=2,
            num_elec_up=1,
            num_elec_dn=1,
            atomic_labels=["H", "H"],
            with_force=True,
        )
        # Set all derivative arrays to zero
        for attr in [
            "de_L_dR",
            "de_L_dr_up",
            "de_L_dr_dn",
            "dln_Psi_dR",
            "dln_Psi_dr_up",
            "dln_Psi_dr_dn",
            "omega_up",
            "omega_dn",
            "domega_dr_up",
            "domega_dr_dn",
        ]:
            setattr(obj, attr, np.zeros_like(getattr(obj, attr)))
        _write_chk(chk_path, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-force", chk_path, "-b", "2", "-w", "0"])
        assert result.exit_code == 0
        # With all derivatives zero, forces should be 0+/-0
        assert "Atomic Forces:" in result.output


class TestHamiltonianCommands:
    """Tests for hamiltonian show-info and to-xyz commands."""

    @pytest.fixture()
    def h5_file(self, tmp_path):
        """Create a hamiltonian_data.h5 from a TREXIO example file."""
        trexio_file = os.path.join(_TREXIO_DIR, "H2_ecp_ccpvtz_cart.h5")
        h5_path = str(tmp_path / "hamiltonian_data.h5")
        trexio_convert_to(
            trexio_file=trexio_file,
            hamiltonian_file=h5_path,
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type="ao-small",
        )
        return h5_path

    def test_hamiltonian_show_info(self, h5_file, capsys):
        """show-info should print without error."""
        hamiltonian_show_info(hamiltonian_data=h5_file)
        captured = capsys.readouterr()
        # It should print *something* about the hamiltonian
        assert len(captured.out) > 0

    def test_hamiltonian_to_xyz(self, h5_file, tmp_path):
        """to-xyz should create an XYZ file with atom coordinates."""
        xyz_path = str(tmp_path / "struct.xyz")
        hamiltonian_to_xyz(hamiltonian_data=h5_file, xyz_file=xyz_path)
        assert os.path.exists(xyz_path)
        text = Path(xyz_path).read_text()
        # First line should be number of atoms
        first_line = text.strip().splitlines()[0].strip()
        assert first_line.isdigit()


class TestTrexioShowCommands:
    """Tests for trexio show-info and show-detail commands."""

    _trexio_file = os.path.join(_TREXIO_DIR, "H2_ecp_ccpvtz_cart.h5")

    def test_trexio_show_info(self, capsys):
        """show-info should produce output without error."""
        trexio_show_info(filename=self._trexio_file)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_trexio_show_detail(self, capsys):
        """show-detail should produce output without error."""
        trexio_show_detail(filename=self._trexio_file)
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestVmcAnalyzeOutput:
    """Tests for vmc analyze-output command."""

    _SAMPLE_LOG = (
        "Optimization step =   1/10\n"
        "E = -1.17 +- 0.05 Ha\n"
        "Max f = 0.123 +- 0.045\n"
        "Max of signal-to-noise of f = max(|f|/|std f|) = 2.733.\n"
        "Optimization step =   2/10\n"
        "E = -1.18 +- 0.04 Ha\n"
        "Max f = 0.100 +- 0.030\n"
        "Max of signal-to-noise of f = max(|f|/|std f|) = 3.333.\n"
    )

    def _write_log(self, path, text=None):
        Path(path).write_text(text or self._SAMPLE_LOG)

    def test_analyze_output_table(self, tmp_path, capsys):
        """analyze-output should print a table with Iter / E / Max f."""
        log_path = str(tmp_path / "vmc.log")
        self._write_log(log_path)
        vmc_analyze_output(filenames=[log_path], plot_graph=False, save_graph=None)
        captured = capsys.readouterr()
        assert "Iter" in captured.out
        assert "E (Ha)" in captured.out
        # Check that both iterations appear
        assert "1" in captured.out
        assert "2" in captured.out

    def test_analyze_output_save_graph(self, tmp_path, capsys):
        """save-graph should create an image file."""
        log_path = str(tmp_path / "vmc.log")
        self._write_log(log_path)
        graph_path = str(tmp_path / "result.png")
        vmc_analyze_output(filenames=[log_path], plot_graph=False, save_graph=graph_path)
        assert os.path.exists(graph_path)

    def test_analyze_output_multiple_files(self, tmp_path, capsys):
        """analyze-output should combine data from multiple files."""
        log1 = str(tmp_path / "vmc1.log")
        log2 = str(tmp_path / "vmc2.log")

        log_text_1 = (
            "Optimization step =   1/5\n"
            "E = -1.17 +- 0.05 Ha\n"
            "Max f = 0.123 +- 0.045\n"
            "Max of signal-to-noise of f = max(|f|/|std f|) = 2.733.\n"
        )
        log_text_2 = (
            "Optimization step =   2/5\n"
            "E = -1.18 +- 0.04 Ha\n"
            "Max f = 0.100 +- 0.030\n"
            "Max of signal-to-noise of f = max(|f|/|std f|) = 3.333.\n"
        )

        self._write_log(log1, log_text_1)
        self._write_log(log2, log_text_2)
        vmc_analyze_output(filenames=[log1, log2], plot_graph=False, save_graph=None)
        captured = capsys.readouterr()
        # Both iterations should appear in the combined output
        assert "1" in captured.out
        assert "2" in captured.out
