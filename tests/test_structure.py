"""Unit tests for Structure_data and PBC helpers."""

import numpy as np
import pytest

from jqmc.setting import decimal_debug_vs_production
from jqmc.structure import (
    Structure_data,
    _find_nearest_index_jnp,
    _find_nearest_index_np,
    _find_nearest_nucleus_indices_jnp,
    _find_nearest_nucleus_indices_np,
    _get_min_dist_rel_R_cart_jnp,
    _get_min_dist_rel_R_cart_np,
)


def _make_pbc_structure():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )
    return Structure_data(
        positions=positions,
        pbc_flag=True,
        vec_a=(10.0, 0.0, 0.0),
        vec_b=(0.0, 12.0, 0.0),
        vec_c=(0.0, 0.0, 14.0),
        atomic_numbers=(1, 1, 1),
        element_symbols=("H", "H", "H"),
        atomic_labels=("H1", "H2", "H3"),
    )


def _make_non_pbc_structure():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    return Structure_data(
        positions=positions,
        pbc_flag=False,
        atomic_numbers=(1, 1, 1),
        element_symbols=("H", "H", "H"),
        atomic_labels=("H1", "H2", "H3"),
    )


def test_reciprocal_lattice_dot_2pi():
    """Test that the dot product of the cell and reciprocal cell gives 2pi delta_ij."""
    structure = _make_pbc_structure()
    recip = structure.recip_cell
    cell = structure.cell
    for i in range(3):
        for j in range(3):
            dot = np.dot(cell[i], recip[j])
            expected = 2.0 * np.pi if i == j else 0.0
            np.testing.assert_allclose(dot, expected, rtol=0.0, atol=10 ** (-decimal_debug_vs_production))


def test_np_jnp_consistency_non_pbc():
    """Test consistency between NumPy and JAX implementations for non-PBC structures."""
    structure = _make_non_pbc_structure()
    r_cart = np.array([0.2, 0.0, 0.0])

    np.testing.assert_allclose(
        structure._positions_cart_np,
        np.asarray(structure._positions_cart_jnp),
        rtol=0.0,
        atol=10 ** (-decimal_debug_vs_production),
    )

    idx_np = _find_nearest_index_np(structure, r_cart)
    idx_jnp = int(np.asarray(_find_nearest_index_jnp(structure, r_cart)))
    assert idx_np == idx_jnp

    nearest_np = _find_nearest_nucleus_indices_np(structure, r_cart, 2)
    nearest_jnp = np.asarray(_find_nearest_nucleus_indices_jnp(structure, r_cart, 2))
    np.testing.assert_array_equal(nearest_np, nearest_jnp)

    for i_atom in range(structure.natom):
        rel_np = _get_min_dist_rel_R_cart_np(structure, r_cart, i_atom)
        rel_jnp = np.asarray(_get_min_dist_rel_R_cart_jnp(structure, r_cart, i_atom))
        np.testing.assert_allclose(rel_np, rel_jnp, rtol=0.0, atol=10 ** (-decimal_debug_vs_production))


def test_pbc_minimum_image_and_nearest():
    """Test PBC minimum image convention and nearest nucleus finding."""
    structure = _make_pbc_structure()
    r_cart = np.array([9.1, 0.0, 0.0])

    idx_np = _find_nearest_index_np(structure, r_cart)
    idx_jnp = int(np.asarray(_find_nearest_index_jnp(structure, r_cart)))
    assert idx_np == idx_jnp == 1

    nearest_np = _find_nearest_nucleus_indices_np(structure, r_cart, 2)
    nearest_jnp = np.asarray(_find_nearest_nucleus_indices_jnp(structure, r_cart, 2))
    np.testing.assert_array_equal(nearest_np, nearest_jnp)

    rel_atom0 = _get_min_dist_rel_R_cart_np(structure, r_cart, 0)
    rel_atom1 = _get_min_dist_rel_R_cart_np(structure, r_cart, 1)

    np.testing.assert_allclose(
        rel_atom0,
        np.array([0.9, 0.0, 0.0]),
        rtol=0.0,
        atol=10 ** (-decimal_debug_vs_production),
    )
    np.testing.assert_allclose(
        rel_atom1,
        np.array([-0.1, 0.0, 0.0]),
        rtol=0.0,
        atol=10 ** (-decimal_debug_vs_production),
    )

    rel_atom0_jnp = np.asarray(_get_min_dist_rel_R_cart_jnp(structure, r_cart, 0))
    rel_atom1_jnp = np.asarray(_get_min_dist_rel_R_cart_jnp(structure, r_cart, 1))
    np.testing.assert_allclose(rel_atom0_jnp, rel_atom0, rtol=0.0, atol=10 ** (-decimal_debug_vs_production))
    np.testing.assert_allclose(rel_atom1_jnp, rel_atom1, rtol=0.0, atol=10 ** (-decimal_debug_vs_production))


@pytest.mark.parametrize("use_pbc", [False, True])
def test_find_nearest_index_matches_min_dist_jnp(use_pbc):
    """Test that the nearest index found matches the minimum distance calculation."""
    structure = _make_pbc_structure() if use_pbc else _make_non_pbc_structure()
    r_cart = np.array([9.1, 0.0, 0.0]) if use_pbc else np.array([1.8, 0.1, 0.0])

    idx = int(np.asarray(_find_nearest_index_jnp(structure, r_cart)))
    rel = np.asarray(_get_min_dist_rel_R_cart_jnp(structure, r_cart, idx))
    dist_idx = np.linalg.norm(rel)

    diffs = structure._positions_cart_np - r_cart
    if use_pbc:
        cell = structure.cell
        inv_cell = np.linalg.inv(cell)
        diffs_frac = diffs @ inv_cell
        diffs_frac = diffs_frac - np.round(diffs_frac)
        diffs = diffs_frac @ cell
    dist_all = np.linalg.norm(diffs, axis=1)

    np.testing.assert_allclose(dist_idx, np.min(dist_all), rtol=0.0, atol=10 ** (-decimal_debug_vs_production))
