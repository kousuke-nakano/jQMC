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

import os
import re
import sys
from pathlib import Path

import pytest

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.hamiltonians import Hamiltonian_data  # noqa: E402
from jqmc.jqmc_tool import _J3_PERIOD_RANGES, trexio_convert_to  # noqa: E402
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402

trexio_files = [
    "H2_ecp_ccpvtz_cart.h5",
    "N2_ecp_ccpvtz_cart.h5",
    "Cl2_ecp_ccpvtz_cart.h5",
    "Ti2_ecp_ccpvtz_cart.h5",
    "CuBr_ecp_ccpvtz_cart.h5",
]


@pytest.mark.parametrize(
    "trexio_filename",
    trexio_files,
    ids=trexio_files,
)
def test_trexio_convert_to_ao_full_matches_uncontracted(tmp_path, trexio_filename):
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
def test_trexio_convert_to_ao_selection(tmp_path, trexio_filename, choice):
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
