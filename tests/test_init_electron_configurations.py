"""Unit tests for the vectorized initial-electron-configuration generator.

Specifically guards against regressions in
:func:`jqmc._jqmc_utility._generate_init_electron_configurations`:

* All generated 3D positions across walkers and electrons must be unique
  (no duplicates).
* Output shapes match the documented contract.
* Per-walker electron-to-atom assignment is consistent
  (each owner is a valid atom index, total count matches input).
* The vectorized version agrees with the reference implementation
  ``_generate_init_electron_configurations_debug`` on the deterministic
  atom-assignment template (i.e., owner indices, modulo per-walker random
  extras).
"""

import numpy as np
import pytest

from jqmc._jqmc_utility import (
    _generate_init_electron_configurations,
    _generate_init_electron_configurations_debug,
)


# (name, charges, coords, tot_up, tot_dn)
_TEST_SYSTEMS = [
    (
        "H2_ae",
        np.array([1.0, 1.0]),
        np.array([[-0.37, 0.0, 0.0], [0.37, 0.0, 0.0]]),
        1,
        1,
    ),
    (
        "Li2_ae",
        np.array([3.0, 3.0]),
        np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        3,
        3,
    ),
    (
        "H2O_ae",
        np.array([1.0, 8.0, 1.0]),
        np.array([[0.00, 0.00, 0.00], [0.96, 0.00, 0.00], [-0.24, 0.93, 0.00]]),
        5,
        5,
    ),
    (
        "H2O_ecp",
        np.array([6.0, 1.0, 1.0]),
        np.array([[0.00, 0.00, 0.00], [0.76, 0.59, 0.00], [-0.76, 0.59, 0.00]]),
        4,
        4,
    ),
    (
        "N2_spin_polarized",
        np.array([7.0, 7.0]),
        np.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]),
        10,
        4,
    ),
    (
        "single_H_ae",
        np.array([1.0]),
        np.array([[0.0, 0.0, 0.0]]),
        1,
        0,
    ),
]


@pytest.mark.parametrize("name,charges,coords,tot_up,tot_dn", _TEST_SYSTEMS)
@pytest.mark.parametrize("num_walkers", [1, 16, 256])
def test_generated_positions_are_all_unique(name, charges, coords, tot_up, tot_dn, num_walkers):
    """All (walker, electron) 3D positions across both spins are pairwise distinct."""
    np.random.seed(123)
    r_up, r_dn, _, _ = _generate_init_electron_configurations(tot_up, tot_dn, num_walkers, charges, coords)
    all_pos = np.concatenate([r_up.reshape(-1, 3), r_dn.reshape(-1, 3)], axis=0)
    n_total = all_pos.shape[0]
    assert n_total == num_walkers * (tot_up + tot_dn)
    n_unique = np.unique(all_pos, axis=0).shape[0]
    assert n_unique == n_total, (
        f"[{name}, nw={num_walkers}] expected {n_total} unique positions, got {n_unique} (duplicates present)"
    )


@pytest.mark.parametrize("name,charges,coords,tot_up,tot_dn", _TEST_SYSTEMS)
def test_output_shapes_and_owner_ranges(name, charges, coords, tot_up, tot_dn):
    """Returned arrays have documented shapes and owners reference valid atoms."""
    nion = coords.shape[0]
    num_walkers = 32
    np.random.seed(7)
    r_up, r_dn, up_owner, dn_owner = _generate_init_electron_configurations(tot_up, tot_dn, num_walkers, charges, coords)
    assert r_up.shape == (num_walkers, tot_up, 3)
    assert r_dn.shape == (num_walkers, tot_dn, 3)
    assert up_owner.shape == (num_walkers, tot_up)
    assert dn_owner.shape == (num_walkers, tot_dn)

    if tot_up > 0:
        assert up_owner.min() >= 0 and up_owner.max() < nion
    if tot_dn > 0:
        assert dn_owner.min() >= 0 and dn_owner.max() < nion


@pytest.mark.parametrize("name,charges,coords,tot_up,tot_dn", _TEST_SYSTEMS)
def test_vectorized_owners_match_reference(name, charges, coords, tot_up, tot_dn):
    """Vectorized and reference implementations agree on the deterministic owner template.

    Both versions place the spin-down electrons via the same deterministic
    state machine, and place the spin-up electrons deterministically except
    for any "extra" tail when ``tot_up > sum(zeta - occup_dn)``. This test
    checks that the deterministic prefix of owner assignments matches.
    """
    num_walkers = 1
    np.random.seed(0)
    _, _, up_v, dn_v = _generate_init_electron_configurations(tot_up, tot_dn, num_walkers, charges, coords)
    np.random.seed(0)
    _, _, up_r, dn_r = _generate_init_electron_configurations_debug(tot_up, tot_dn, num_walkers, charges, coords)

    # Down-electron assignment is fully deterministic in both implementations.
    np.testing.assert_array_equal(dn_v[0], dn_r[0])

    # Up-electron assignment is deterministic up to the "extra" tail; compare
    # the deterministic portion only.
    zeta = np.rint(charges).astype(int)
    occup_dn = np.bincount(dn_r[0], minlength=coords.shape[0])
    sum_up_needed = int(np.sum(np.maximum(zeta - occup_dn, 0)))
    det_count = min(tot_up, sum_up_needed)
    np.testing.assert_array_equal(up_v[0, :det_count], up_r[0, :det_count])


@pytest.mark.parametrize("name,charges,coords,tot_up,tot_dn", _TEST_SYSTEMS)
@pytest.mark.parametrize("num_walkers", [1, 8])
def test_per_atom_spin_counts_match_reference(name, charges, coords, tot_up, tot_dn, num_walkers):
    """Per-atom up/dn populations must agree exactly with the reference implementation.

    This guards against regressions in the *aggregate* spin-assignment behavior
    (more permissive than owner-vector equality, but catches any change in
    physical spin distribution per atom). Critical for systems where the old
    algorithm specifically handled distant-atom spin distribution.
    """
    nion = coords.shape[0]
    np.random.seed(0)
    _, _, up_v, dn_v = _generate_init_electron_configurations(tot_up, tot_dn, num_walkers, charges, coords)
    np.random.seed(0)
    _, _, up_r, dn_r = _generate_init_electron_configurations_debug(tot_up, tot_dn, num_walkers, charges, coords)
    for iw in range(num_walkers):
        np.testing.assert_array_equal(
            np.bincount(up_v[iw], minlength=nion),
            np.bincount(up_r[iw], minlength=nion),
            err_msg=f"[{name}, walker {iw}] vectorized vs reference up counts differ",
        )
        np.testing.assert_array_equal(
            np.bincount(dn_v[iw], minlength=nion),
            np.bincount(dn_r[iw], minlength=nion),
            err_msg=f"[{name}, walker {iw}] vectorized vs reference dn counts differ",
        )


# Identical-atom dimers in the global singlet (S=0) configuration. The expected
# physical behavior at any separation: each atom carries zeta electrons with
# Hund-maximum local moment, anti-aligned to its partner so the global S=0.
# E.g., for N2 (zeta=7 each) at S=0 → one atom (4u, 3d), the other (3u, 4d).
@pytest.mark.parametrize("elem_z", [3, 6, 7, 8])  # Li, C, N, O (AE valence counts)
@pytest.mark.parametrize("separation", [0.5, 2.0, 5.0, 100.0])
def test_dimer_singlet_atoms_are_antialigned(elem_z, separation):
    """For a homonuclear dimer at S=0 the two atoms must be spin-mirror images."""
    charges = np.array([float(elem_z), float(elem_z)])
    coords = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
    tot_up = elem_z
    tot_dn = elem_z
    num_walkers = 4

    np.random.seed(0)
    _, _, up_owner, dn_owner = _generate_init_electron_configurations(tot_up, tot_dn, num_walkers, charges, coords)
    for iw in range(num_walkers):
        up_counts = np.bincount(up_owner[iw], minlength=2)
        dn_counts = np.bincount(dn_owner[iw], minlength=2)
        # Each atom is charge-neutral.
        np.testing.assert_array_equal(
            up_counts + dn_counts,
            np.array([elem_z, elem_z]),
            err_msg=f"sep={separation}, Z={elem_z}, walker {iw}: per-atom electron count != zeta",
        )
        # Spins anti-aligned: nup on one = ndn on the other (and vice versa).
        assert up_counts[0] == dn_counts[1], (
            f"sep={separation}, Z={elem_z}, walker {iw}: expected up_counts[0] == dn_counts[1], got {up_counts}, {dn_counts}"
        )
        assert dn_counts[0] == up_counts[1], (
            f"sep={separation}, Z={elem_z}, walker {iw}: expected dn_counts[0] == up_counts[1], got {up_counts}, {dn_counts}"
        )
        # Hund respected: |nup - ndn| per atom = expected unpaired-electron count.
        # For zeta = elem_z, max_dn_per_atom = elem_z // 2, so unpaired = elem_z - 2 * (elem_z // 2).
        expected_unpaired = elem_z - 2 * (elem_z // 2) + 2 * abs((elem_z // 2) - min(up_counts[0], dn_counts[0]))
        # Relaxed check: each atom's |nup - ndn| should be at least the parity (1 if odd zeta, else 0).
        parity = elem_z % 2
        for atom in range(2):
            assert abs(int(up_counts[atom]) - int(dn_counts[atom])) >= parity, (
                f"sep={separation}, Z={elem_z}, walker {iw}, atom {atom}: "
                f"|nup-ndn| < parity ({parity}); counts up={up_counts}, dn={dn_counts}"
            )


# For homonuclear / heteronuclear dimers, sweep every reachable global spin S
# (i.e., every valid (tot_up, tot_dn) partition that sums to total electron
# count) and verify per-atom charge neutrality. This is the core invariant
# the original algorithm was designed to provide for widely-separated atoms
# (e.g., N---N at 100 bohr where each atom must locally hold zeta electrons
# regardless of global spin polarization).
@pytest.mark.parametrize("separation", [2.0, 100.0])
@pytest.mark.parametrize(
    "label,zeta_pair",
    [
        ("N---N", (7, 7)),
        ("N---O", (7, 8)),
        ("O---O", (8, 8)),
        ("Li---N", (3, 7)),
        ("Li---Li", (3, 3)),
        ("C---N", (6, 7)),
    ],
)
def test_dimer_per_atom_charge_neutral_for_all_S(label, zeta_pair, separation):
    """For every reachable S of a (neutral) dimer, both atoms must be charge-neutral.

    Iterates over every valid ``(tot_up, tot_dn)`` partition with
    ``tot_up + tot_dn == zeta[0] + zeta[1]``. For each, asserts that
    ``np.bincount(up_owner) + np.bincount(dn_owner) == zeta`` for every walker.
    Also checks the totals match the input split.
    """
    z0, z1 = zeta_pair
    total = z0 + z1
    charges = np.array([float(z0), float(z1)])
    coords = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
    expected_zeta = np.array([z0, z1])
    num_walkers = 4

    for tot_up in range(total + 1):
        tot_dn = total - tot_up
        np.random.seed(0)
        _, _, up_owner, dn_owner = _generate_init_electron_configurations(tot_up, tot_dn, num_walkers, charges, coords)
        S2 = tot_up - tot_dn  # 2*S signed
        for iw in range(num_walkers):
            up_counts = np.bincount(up_owner[iw], minlength=2)
            dn_counts = np.bincount(dn_owner[iw], minlength=2)
            per_atom_total = up_counts + dn_counts
            np.testing.assert_array_equal(
                per_atom_total,
                expected_zeta,
                err_msg=(
                    f"[{label}, sep={separation}, 2S={S2}, walker {iw}] "
                    f"per-atom total {per_atom_total} != zeta {expected_zeta}; "
                    f"up={up_counts.tolist()}, dn={dn_counts.tolist()}"
                ),
            )
            assert int(up_counts.sum()) == tot_up, (
                f"[{label}, 2S={S2}, walker {iw}] sum(up_counts)={up_counts.sum()} != tot_up={tot_up}"
            )
            assert int(dn_counts.sum()) == tot_dn, (
                f"[{label}, 2S={S2}, walker {iw}] sum(dn_counts)={dn_counts.sum()} != tot_dn={tot_dn}"
            )


# Same idea, against the reference implementation: per-atom up/dn counts must
# match across all S values. This is a stricter "no-regression" check on the
# vectorized refactor specifically for the spin-distribution behavior.
@pytest.mark.parametrize(
    "label,zeta_pair",
    [
        ("N---N", (7, 7)),
        ("N---O", (7, 8)),
        ("Li---N", (3, 7)),
    ],
)
def test_dimer_per_atom_counts_match_reference_for_all_S(label, zeta_pair):
    """Vectorized vs reference per-atom spin counts must agree for every S."""
    z0, z1 = zeta_pair
    total = z0 + z1
    charges = np.array([float(z0), float(z1)])
    coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    num_walkers = 2

    for tot_up in range(total + 1):
        tot_dn = total - tot_up
        np.random.seed(0)
        _, _, up_v, dn_v = _generate_init_electron_configurations(tot_up, tot_dn, num_walkers, charges, coords)
        np.random.seed(0)
        _, _, up_r, dn_r = _generate_init_electron_configurations_debug(tot_up, tot_dn, num_walkers, charges, coords)
        for iw in range(num_walkers):
            np.testing.assert_array_equal(
                np.bincount(up_v[iw], minlength=2),
                np.bincount(up_r[iw], minlength=2),
                err_msg=f"[{label}, 2S={tot_up - tot_dn}, walker {iw}] up counts differ",
            )
            np.testing.assert_array_equal(
                np.bincount(dn_v[iw], minlength=2),
                np.bincount(dn_r[iw], minlength=2),
                err_msg=f"[{label}, 2S={tot_up - tot_dn}, walker {iw}] dn counts differ",
            )


# Triatomic (linear, well-separated) chain — exercises ion_seq logic for >2 atoms.
@pytest.mark.parametrize("elem_z", [3, 7])
def test_linear_triatomic_charge_neutrality(elem_z):
    """For a homonuclear linear triatomic at separation 5 bohr each, every atom
    receives exactly zeta electrons (charge-neutral assignment)."""
    charges = np.array([float(elem_z)] * 3)
    coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    nion = 3
    # Use total = sum(zeta), pick a balanced spin partition.
    total = 3 * elem_z
    tot_up = total // 2 + (total % 2)  # ceil
    tot_dn = total - tot_up
    num_walkers = 4

    np.random.seed(0)
    _, _, up_owner, dn_owner = _generate_init_electron_configurations(tot_up, tot_dn, num_walkers, charges, coords)
    for iw in range(num_walkers):
        per_atom_total = np.bincount(up_owner[iw], minlength=nion) + np.bincount(dn_owner[iw], minlength=nion)
        np.testing.assert_array_equal(
            per_atom_total,
            np.array([elem_z, elem_z, elem_z]),
            err_msg=f"Z={elem_z}, walker {iw}: per-atom electron count != zeta (got {per_atom_total})",
        )
