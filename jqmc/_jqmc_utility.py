"""utility module.

Precision Zones:
    - ``io``: all functions in this module.

See :mod:`jqmc._precision` for details.
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

from functools import lru_cache
from logging import getLogger

import numpy as np
import numpy.typing as npt

# set logger
logger = getLogger("jqmc").getChild(__name__)

# separator
num_sep_line = 66


def _generate_init_electron_configurations(
    tot_num_electron_up: int,
    tot_num_electron_dn: int,
    num_walkers: int,
    charges: np.ndarray,
    coords: np.ndarray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Vectorized initial electron configuration generator for many walkers.

    Functionally equivalent to :func:`_generate_init_electron_configurations_debug`
    but avoids the per-walker Python loop. Designed for large walker counts
    (e.g. ``num_walkers = 16384``) where the reference version becomes the
    initialization bottleneck.

    Algorithm:
        1. Compute the deterministic atom-assignment templates for spin-up and
           spin-down electrons by replaying the original state machine **once**.
           These templates depend only on (charges, coords) — every walker
           shares them.
        2. Tile the templates to shape ``(num_walkers, ned)``.
        3. For the only branch in the original that uses per-walker randomness
           — Phase 1b "extra up" electrons when
           ``tot_num_electron_up > sum(zeta - occup_dn)`` — draw the random
           atom indices in one batched ``np.random.randint`` call.
        4. Draw all spherical random offsets in one batched call per spin and
           add them to the chosen atomic coordinates.

    The duplicate-avoidance retry loop in the reference implementation is
    omitted: spherical offsets are sampled from continuous uniform
    distributions, so the probability of two ``float64`` positions colliding
    within ``1e-6`` is effectively zero. The companion test
    ``test_init_electron_configurations`` verifies uniqueness across all
    generated positions.

    Parameters and returns are identical to
    :func:`_generate_init_electron_configurations_debug`.
    """
    min_dst = 0.1
    max_dst = 1.0
    dtype_np = np.float64

    nion = coords.shape[0]
    coords_np = np.asarray(coords, dtype=dtype_np)
    zeta = np.array([int(round(c)) for c in np.asarray(charges)], dtype=int)
    max_dn_per_atom = zeta // 2

    # 1) Build ion_seq (each next index is the atom farthest from the previous).
    ion_sel = np.ones(nion, dtype=bool)
    ion_seq = np.zeros(nion, dtype=int)
    ion_seq[0] = 0
    ion_sel[0] = False
    i_prev = 0
    for idx in range(1, nion):
        d2 = np.sum((coords_np[i_prev] - coords_np) ** 2, axis=1)
        d2_masked = np.where(ion_sel, d2, -1.0)
        best_i = int(np.argmax(d2_masked))
        ion_seq[idx] = best_i
        ion_sel[best_i] = False
        i_prev = best_i

    # 2) Replay the deterministic state machine ONCE to obtain owner templates.
    occup_total = np.zeros(nion, dtype=int)
    occup_dn = np.zeros(nion, dtype=int)
    occup_up = np.zeros(nion, dtype=int)

    # Phase 1a: place all spin-down electrons (deterministic).
    ned_dn = tot_num_electron_dn
    dn_owner_template = np.empty(ned_dn, dtype=int)
    j_counter = 0
    for idn in range(ned_dn):
        while True:
            atom = ion_seq[j_counter % nion]
            if np.any(occup_dn < max_dn_per_atom):
                cond = occup_dn[atom] < max_dn_per_atom[atom]
            else:
                mask_zero = (max_dn_per_atom == 0) & (occup_total < zeta)
                if np.any(mask_zero):
                    cond = (max_dn_per_atom[atom] == 0) and (occup_total[atom] < zeta[atom])
                else:
                    cond = occup_total[atom] < zeta[atom]
            if cond:
                dn_owner_template[idn] = atom
                occup_dn[atom] += 1
                occup_total[atom] += 1
                j_counter += 1
                break
            j_counter += 1

    # Phase 1b: place spin-up electrons; deterministic except for the
    # "extra" tail in Case 2 which is per-walker random.
    up_needed = zeta - occup_dn
    sum_up_needed = int(np.sum(up_needed))
    ned_up = tot_num_electron_up
    up_owner_template = np.empty(ned_up, dtype=int)
    n_random_extras = 0  # trailing electrons whose owner is random per walker

    if ned_up <= sum_up_needed:
        # Case 1: place exactly into the up_needed slots — fully deterministic.
        ptr = 0
        for iup in range(ned_up):
            while True:
                atom = ion_seq[ptr % nion]
                if occup_up[atom] < up_needed[atom]:
                    up_owner_template[iup] = atom
                    occup_up[atom] += 1
                    occup_total[atom] += 1
                    ptr += 1
                    break
                ptr += 1
    else:
        # Case 2: first satisfy every atom's up_needed (deterministic), then
        # the remainder is sampled per walker.
        cnt = 0
        for atom in ion_seq:
            to_give = int(up_needed[atom])
            for _ in range(to_give):
                up_owner_template[cnt] = atom
                occup_up[atom] += 1
                occup_total[atom] += 1
                cnt += 1
        n_random_extras = ned_up - sum_up_needed

    # 3) Build per-walker owner arrays.
    if ned_dn > 0:
        dn_owner = np.broadcast_to(dn_owner_template, (num_walkers, ned_dn)).copy()
    else:
        dn_owner = np.empty((num_walkers, 0), dtype=int)

    up_owner = np.empty((num_walkers, ned_up), dtype=int)
    if n_random_extras > 0:
        det_count = ned_up - n_random_extras
        if det_count > 0:
            up_owner[:, :det_count] = up_owner_template[:det_count][None, :]
        # Per-walker random pick from ion_seq, matching the original
        #   idx = int(floor(np.random.rand() * nion))
        rand_idx = np.floor(np.random.rand(num_walkers, n_random_extras) * nion).astype(int)
        # Clip to nion-1 just in case np.random.rand() returns 1.0 (it shouldn't).
        np.clip(rand_idx, 0, nion - 1, out=rand_idx)
        up_owner[:, det_count:] = ion_seq[rand_idx]
    else:
        up_owner[:] = up_owner_template[None, :]

    # 4) Draw all spherical random offsets in batched form.
    def _spherical_offsets(shape: tuple[int, int]) -> np.ndarray:
        distance = np.random.uniform(min_dst, max_dst, size=shape)
        theta = np.random.uniform(0.0, np.pi, size=shape)
        phi = np.random.uniform(0.0, 2.0 * np.pi, size=shape)
        sin_t = np.sin(theta)
        return np.stack(
            [
                distance * sin_t * np.cos(phi),
                distance * sin_t * np.sin(phi),
                distance * np.cos(theta),
            ],
            axis=-1,
        ).astype(dtype_np, copy=False)

    offset_dn = _spherical_offsets((num_walkers, ned_dn)) if ned_dn > 0 else np.zeros((num_walkers, 0, 3), dtype=dtype_np)
    offset_up = _spherical_offsets((num_walkers, ned_up)) if ned_up > 0 else np.zeros((num_walkers, 0, 3), dtype=dtype_np)

    # 5) Assemble final positions: r = coords[owner] + offset.
    r_carts_up = (coords_np[up_owner] + offset_up).astype(dtype_np, copy=False)
    r_carts_dn = (coords_np[dn_owner] + offset_dn).astype(dtype_np, copy=False)

    return r_carts_up, r_carts_dn, up_owner, dn_owner


def _generate_init_electron_configurations_debug(
    tot_num_electron_up: int,
    tot_num_electron_dn: int,
    num_walkers: int,
    charges: np.ndarray,
    coords: np.ndarray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Reference (per-walker Python loop) initial electron configuration generator.

    This is the original implementation kept for cross-checks against the
    vectorized :func:`_generate_init_electron_configurations`. It runs an
    ``O(num_walkers)`` Python loop and is too slow for large walker counts
    (e.g. ``num_walkers = 16384``); use only for tests / debugging.

    Generate initial electron configurations (up/down positions) for a set of walkers,
    using the same ion_seq idea as the Fortran initconf routine, but without
    any periodic boundary conditions or lattice parameters.

    Parameters:
        tot_num_electron_up (int):
            Total number of spin-up electrons in the system.

        tot_num_electron_dn (int):
            Total number of spin-down electrons in the system.

        num_walkers (int):
            Number of independent walkers (configurations) to generate.

        charges (np.ndarray of shape (nion,)):
            Atomic charges (should reflect valence electron count, e.g.
            atomic_number or atomic_number - z_core). They will be rounded to
            integers internally.

        coords (np.ndarray of shape (nion, 3)):
            Cartesian coordinates of each atom in the system.

    Returns:
        r_carts_up (np.ndarray of shape (num_walkers, tot_num_electron_up, 3)):
            Generated positions of all spin-up electrons for each walker.

        r_carts_dn (np.ndarray of shape (num_walkers, tot_num_electron_dn, 3)):
            Generated positions of all spin-down electrons for each walker.

        up_owner (np.ndarray of shape (num_walkers, tot_num_electron_up), dtype=int):
            For each walker `iw` and each up-electron `k`, the atom-index it was assigned to.

        dn_owner (np.ndarray of shape (num_walkers, tot_num_electron_dn), dtype=int):
            For each walker `iw` and each down-electron `k`, the atom-index it was assigned to.
    """
    # Fixed random displacement range (±dst/2 in each coordinate)
    min_dst = 0.1
    max_dst = 1.0

    dtype_np = np.float64

    # 1) zeta[i] = integer valence count per atom
    nion = coords.shape[0]
    zeta = np.array([int(round(c)) for c in charges], dtype=int)

    # 2) max_dn_per_atom = floor(zeta[i]/2) for Hund’s rule on down-electrons
    max_dn_per_atom = zeta // 2

    # 3) Build ion_seq so that each next index is the atom farthest from the previous
    ion_sel = np.ones(nion, dtype=bool)
    ion_seq = np.zeros(nion, dtype=int)
    ion_seq[0] = 0
    ion_sel[0] = False
    i_prev = 0
    for idx in range(1, nion):
        best_dist = -1.0
        best_i = -1
        for i in range(nion):
            if ion_sel[i]:
                d2 = np.sum((coords[i_prev] - coords[i]) ** 2)
                if d2 > best_dist:
                    best_dist = d2
                    best_i = i
        ion_seq[idx] = best_i
        ion_sel[best_i] = False
        i_prev = best_i

    # 4) Prepare storage for all walkers
    r_carts_up = np.zeros((num_walkers, tot_num_electron_up, 3), dtype=dtype_np)
    r_carts_dn = np.zeros((num_walkers, tot_num_electron_dn, 3), dtype=dtype_np)
    up_owner = np.zeros((num_walkers, tot_num_electron_up), dtype=int)
    dn_owner = np.zeros((num_walkers, tot_num_electron_dn), dtype=int)

    # 6) Loop over walkers
    for iw in range(num_walkers):
        # 6.1) Reset per-walker occupancy
        occup_total = np.zeros(nion, dtype=int)  # total electrons (↑+↓) on each atom
        occup_dn = np.zeros(nion, dtype=int)  # how many down-electrons on each atom
        occup_up = np.zeros(nion, dtype=int)  # how many up-electrons on each atom
        cdown = 0
        cup = 0

        # 6.2) Compute any “extra” beyond sum(zeta)
        nel = tot_num_electron_up + tot_num_electron_dn
        ztot = int(np.sum(zeta))
        nelupeff = nel - ztot if nel > ztot else 0

        # -----------------------------------------
        # Phase 1a: Place all down-electrons under Hund’s limit first
        # -----------------------------------------
        ned_dn = tot_num_electron_dn
        down_positions = np.zeros((ned_dn, 3), dtype=dtype_np)
        j_counter = 0

        for idn in range(ned_dn):
            placed = False
            while not placed:
                atom = ion_seq[j_counter % nion]

                # If any atom still has occup_dn < max_dn_per_atom, restrict to those atoms.
                if np.any(occup_dn < max_dn_per_atom):
                    cond = occup_dn[atom] < max_dn_per_atom[atom]
                else:
                    # All atoms have occup_dn == max_dn_per_atom.  Next fallback:
                    #   1) If any atom has max_dn_per_atom == 0 (e.g. H) AND occup_total < zeta,
                    #      restrict to those atoms first.
                    mask_zero = (max_dn_per_atom == 0) & (occup_total < zeta)
                    if np.any(mask_zero):
                        cond = (max_dn_per_atom[atom] == 0) and (occup_total[atom] < zeta[atom])
                    else:
                        # Otherwise, any atom with occup_total < zeta can accept a down
                        cond = occup_total[atom] < zeta[atom]

                if cond:
                    # Place one ↓-electron around coords[atom] + random_offset
                    x0, y0, z0 = coords[atom]
                    distance = np.random.uniform(min_dst, max_dst)
                    theta = np.random.uniform(0, np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    dx = distance * np.sin(theta) * np.cos(phi)
                    dy = distance * np.sin(theta) * np.sin(phi)
                    dz = distance * np.cos(theta)
                    new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])

                    # Avoid exact duplication among already-placed down-positions
                    ok = False
                    while not ok:
                        ok = True
                        for prev in down_positions[:cdown]:
                            if np.sum(np.abs(prev - new_pos)) < 1e-6:
                                distance = np.random.uniform(min_dst, max_dst)
                                theta = np.random.uniform(0, np.pi)
                                phi = np.random.uniform(0, 2 * np.pi)
                                dx = distance * np.sin(theta) * np.cos(phi)
                                dy = distance * np.sin(theta) * np.sin(phi)
                                dz = distance * np.cos(theta)
                                new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])
                                ok = False
                                break

                    down_positions[idn] = new_pos
                    dn_owner[iw, idn] = atom
                    occup_dn[atom] += 1
                    occup_total[atom] += 1
                    cdown += 1
                    placed = True

                j_counter += 1

        # -----------------------------------------
        # Phase 1b: Place up-electrons exactly to “fill to zeta” if possible
        # -----------------------------------------
        # Compute how many up each atom needs to reach zeta:
        up_needed = zeta - occup_dn  # array of length nion
        sum_up_needed = int(np.sum(up_needed))

        ned_up = tot_num_electron_up
        up_positions = np.zeros((ned_up, 3), dtype=dtype_np)

        # Case 1: ned_up <= sum_up_needed → place ned_up among those up_needed slots
        if ned_up <= sum_up_needed:
            ptr = 0
            for iup in range(ned_up):
                placed = False
                while not placed:
                    atom = ion_seq[ptr % nion]
                    if occup_up[atom] < up_needed[atom]:
                        # Place one ↑-electron here
                        x0, y0, z0 = coords[atom]
                        distance = np.random.uniform(min_dst, max_dst)
                        theta = np.random.uniform(0, np.pi)
                        phi = np.random.uniform(0, 2 * np.pi)
                        dx = distance * np.sin(theta) * np.cos(phi)
                        dy = distance * np.sin(theta) * np.sin(phi)
                        dz = distance * np.cos(theta)
                        new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])

                        # Avoid duplication among already-placed up-positions
                        ok = False
                        while not ok:
                            ok = True
                            for prev in up_positions[:cup]:
                                if np.sum(np.abs(prev - new_pos)) < 1e-6:
                                    distance = np.random.uniform(min_dst, max_dst)
                                    theta = np.random.uniform(0, np.pi)
                                    phi = np.random.uniform(0, 2 * np.pi)
                                    dx = distance * np.sin(theta) * np.cos(phi)
                                    dy = distance * np.sin(theta) * np.sin(phi)
                                    dz = distance * np.cos(theta)
                                    new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])
                                    ok = False
                                    break

                        up_positions[iup] = new_pos
                        up_owner[iw, iup] = atom
                        occup_up[atom] += 1
                        occup_total[atom] += 1
                        cup += 1
                        placed = True
                    ptr += 1

        # Case 2: ned_up > sum_up_needed → give each atom its up_needed, then place extras
        else:
            # (a) first satisfy every atom’s up_needed
            cnt = 0
            for atom in ion_seq:
                to_give = int(up_needed[atom])
                for _ in range(to_give):
                    x0, y0, z0 = coords[atom]
                    distance = np.random.uniform(min_dst, max_dst)
                    theta = np.random.uniform(0, np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    dx = distance * np.sin(theta) * np.cos(phi)
                    dy = distance * np.sin(theta) * np.sin(phi)
                    dz = distance * np.cos(theta)
                    new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])

                    ok = False
                    while not ok:
                        ok = True
                        for prev in up_positions[:cup]:
                            if np.sum(np.abs(prev - new_pos)) < 1e-6:
                                distance = np.random.uniform(min_dst, max_dst)
                                theta = np.random.uniform(0, np.pi)
                                phi = np.random.uniform(0, 2 * np.pi)
                                dx = distance * np.sin(theta) * np.cos(phi)
                                dy = distance * np.sin(theta) * np.sin(phi)
                                dz = distance * np.cos(theta)
                                new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])
                                ok = False
                                break

                    up_positions[cnt] = new_pos
                    up_owner[iw, cnt] = atom
                    occup_up[atom] += 1
                    occup_total[atom] += 1
                    cnt += 1
                    cup += 1

            # (b) now place the “extra” up = ned_up - sum_up_needed on any atom (fallback)
            extra_up = ned_up - sum_up_needed
            for _ in range(extra_up):
                idx = int(np.floor(np.random.rand() * nion))
                atom = ion_seq[idx]
                x0, y0, z0 = coords[atom]
                distance = np.random.uniform(min_dst, max_dst)
                theta = np.random.uniform(0, np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                dx = distance * np.sin(theta) * np.cos(phi)
                dy = distance * np.sin(theta) * np.sin(phi)
                dz = distance * np.cos(theta)
                new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])

                ok = False
                while not ok:
                    ok = True
                    for prev in up_positions[:cup]:
                        if np.sum(np.abs(prev - new_pos)) < 1e-6:
                            distance = np.random.uniform(min_dst, max_dst)
                            theta = np.random.uniform(0, np.pi)
                            phi = np.random.uniform(0, 2 * np.pi)
                            dx = distance * np.sin(theta) * np.cos(phi)
                            dy = distance * np.sin(theta) * np.sin(phi)
                            dz = distance * np.cos(theta)
                            new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])
                            ok = False
                            break

                up_positions[cnt] = new_pos
                up_owner[iw, cnt] = atom
                occup_up[atom] += 1
                occup_total[atom] += 1
                cnt += 1
                cup += 1

        # -----------------------------------------
        # Phase 2: If “extra” electrons remain (nelupeff > 0), place them now.
        #          (This is almost never needed if tot_up+tot_dn == sum(zeta), but we include it for completeness.)
        # -----------------------------------------
        if nelupeff > 1:
            # 2a) extra down beyond Hund’s limit
            sum_dn_assigned = int(np.sum(occup_dn))
            extra_dn = ned_dn - sum_dn_assigned
            for _ in range(extra_dn):
                idx = int(np.floor(np.random.rand() * nion))
                atom = ion_seq[idx]
                x0, y0, z0 = coords[atom]
                distance = np.random.uniform(min_dst, max_dst)
                theta = np.random.uniform(0, np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                dx = distance * np.sin(theta) * np.cos(phi)
                dy = distance * np.sin(theta) * np.sin(phi)
                dz = distance * np.cos(theta)
                new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])

                ok = False
                while not ok:
                    ok = True
                    for prev in down_positions[:cdown]:
                        if np.sum(np.abs(prev - new_pos)) < 1e-6:
                            distance = np.random.uniform(min_dst, max_dst)
                            theta = np.random.uniform(0, np.pi)
                            phi = np.random.uniform(0, 2 * np.pi)
                            dx = distance * np.sin(theta) * np.cos(phi)
                            dy = distance * np.sin(theta) * np.sin(phi)
                            dz = distance * np.cos(theta)
                            new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])
                            ok = False
                            break

                down_positions[cdown] = new_pos
                dn_owner[iw, cdown] = atom
                occup_dn[atom] += 1
                occup_total[atom] += 1
                cdown += 1

        # -----------------------------------------
        # 6.3) Final consistency check
        # -----------------------------------------
        if cup != tot_num_electron_up:
            raise RuntimeError(f"Walker {iw}: assigned up={cup}, expected {tot_num_electron_up}")
        if cdown != tot_num_electron_dn:
            raise RuntimeError(f"Walker {iw}: assigned dn={cdown}, expected {tot_num_electron_dn}")

        # -----------------------------------------
        # 6.4) Copy into outputs
        # -----------------------------------------
        r_carts_up[iw, :, :] = up_positions
        r_carts_dn[iw, :, :] = down_positions

    return r_carts_up, r_carts_dn, up_owner, dn_owner


@lru_cache(maxsize=None)
def _cart_to_spherical_matrix(l: int) -> np.ndarray:
    """Precomputed cart -> real-spherical transform for angular momentum ``l`` (0–6).

    The matrix has shape ``((l+1)(l+2)/2, 2l+1)`` and satisfies
    ``A_sph = A_cart @ T`` under the normalization used in the codebase. Values
    are deterministic and cached to avoid runtime fitting.
    """
    precomputed: dict[int, np.ndarray] = {
        0: np.array([[1.0]], dtype=np.float64),
        1: np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        2: np.array(
            [
                [0.0, 0.0, -0.5, 0.0, np.sqrt(3) / 2.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, -0.5, 0.0, -np.sqrt(3) / 2.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        3: np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -np.sqrt(6) / 4.0, 0.0, np.sqrt(10) / 4.0],
                [3.0 * np.sqrt(2) / 4.0, 0.0, -np.sqrt(30) / 20.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -3.0 * np.sqrt(5) / 10.0, 0.0, np.sqrt(3) / 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -np.sqrt(30) / 20.0, 0.0, -3.0 * np.sqrt(2) / 4.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, np.sqrt(30) / 5.0, 0.0, 0.0],
                [-np.sqrt(10) / 4.0, 0.0, -np.sqrt(6) / 4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -3.0 * np.sqrt(5) / 10.0, 0.0, -np.sqrt(3) / 2.0, 0.0],
                [0.0, 0.0, np.sqrt(30) / 5.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        4: np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 3.0 / 8.0, 0.0, -np.sqrt(5) / 4.0, 0.0, np.sqrt(35) / 8.0],
                [np.sqrt(5) / 2.0, 0.0, -np.sqrt(35) / 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(70) / 28.0, 0.0, np.sqrt(10) / 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 3.0 * np.sqrt(105) / 140.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(3) / 4.0],
                [0.0, 3.0 * np.sqrt(2) / 4.0, 0.0, -3.0 * np.sqrt(14) / 28.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(105) / 35.0, 0.0, 3.0 * np.sqrt(21) / 14.0, 0.0, 0.0],
                [-np.sqrt(5) / 2.0, 0.0, -np.sqrt(35) / 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(14) / 28.0, 0.0, -3.0 * np.sqrt(2) / 4.0, 0.0],
                [0.0, 0.0, 3.0 * np.sqrt(7) / 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(70) / 7.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 3.0 / 8.0, 0.0, np.sqrt(5) / 4.0, 0.0, np.sqrt(35) / 8.0],
                [0.0, -np.sqrt(10) / 4.0, 0.0, -3.0 * np.sqrt(70) / 28.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(105) / 35.0, 0.0, -3.0 * np.sqrt(21) / 14.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, np.sqrt(70) / 7.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        5: np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(15) / 8.0, 0.0, -np.sqrt(70) / 16.0, 0.0, 3.0 * np.sqrt(14) / 16.0],
                [5.0 * np.sqrt(14) / 16.0, 0.0, -np.sqrt(70) / 16.0, 0.0, np.sqrt(15) / 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 5.0 / 8.0, 0.0, -np.sqrt(105) / 12.0, 0.0, np.sqrt(35) / 8.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(35) / 28.0, 0.0, np.sqrt(30) / 24.0, 0.0, -5.0 * np.sqrt(6) / 8.0],
                [0.0, np.sqrt(5) / 2.0, 0.0, -np.sqrt(15) / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(35) / 14.0, 0.0, np.sqrt(30) / 6.0, 0.0, 0.0],
                [-5.0 * np.sqrt(6) / 8.0, 0.0, -np.sqrt(30) / 24.0, 0.0, np.sqrt(35) / 28.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(105) / 28.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(3) / 4.0, 0.0],
                [0.0, 0.0, np.sqrt(6) / 2.0, 0.0, -3.0 * np.sqrt(7) / 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -5.0 * np.sqrt(21) / 21.0, 0.0, np.sqrt(5) / 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(15) / 24.0, 0.0, np.sqrt(70) / 16.0, 0.0, 5.0 * np.sqrt(14) / 16.0],
                [0.0, -np.sqrt(5) / 2.0, 0.0, -np.sqrt(15) / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(7) / 14.0, 0.0, -np.sqrt(6) / 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, np.sqrt(15) / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(15) / 3.0, 0.0, 0.0, 0.0, 0.0],
                [3.0 * np.sqrt(14) / 16.0, 0.0, np.sqrt(70) / 16.0, 0.0, np.sqrt(15) / 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 5.0 / 8.0, 0.0, np.sqrt(105) / 12.0, 0.0, np.sqrt(35) / 8.0, 0.0],
                [0.0, 0.0, -np.sqrt(30) / 6.0, 0.0, -3.0 * np.sqrt(35) / 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -5.0 * np.sqrt(21) / 21.0, 0.0, -np.sqrt(5) / 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, np.sqrt(15) / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        6: np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -5.0 / 16.0,
                    0.0,
                    np.sqrt(210) / 32.0,
                    0.0,
                    -3.0 * np.sqrt(7) / 16.0,
                    0.0,
                    np.sqrt(462) / 32.0,
                ],
                [
                    3.0 * np.sqrt(42) / 16.0,
                    0.0,
                    -np.sqrt(693) / 44.0,
                    0.0,
                    np.sqrt(2310) / 176.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    5.0 * np.sqrt(231) / 88.0,
                    0.0,
                    -3.0 * np.sqrt(2310) / 176.0,
                    0.0,
                    3.0 * np.sqrt(14) / 16.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -5.0 * np.sqrt(33) / 176.0,
                    0.0,
                    np.sqrt(770) / 352.0,
                    0.0,
                    5.0 * np.sqrt(231) / 176.0,
                    0.0,
                    -15.0 * np.sqrt(14) / 32.0,
                ],
                [
                    0.0,
                    5.0 * np.sqrt(14) / 16.0,
                    0.0,
                    -3.0 * np.sqrt(2310) / 176.0,
                    0.0,
                    5.0 * np.sqrt(231) / 264.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    15.0 * np.sqrt(33) / 88.0,
                    0.0,
                    -np.sqrt(770) / 22.0,
                    0.0,
                    5.0 * np.sqrt(231) / 88.0,
                    0.0,
                    0.0,
                ],
                [-5.0 * np.sqrt(10) / 8.0, 0.0, 0.0, 0.0, 5.0 * np.sqrt(22) / 88.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    5.0 * np.sqrt(11) / 44.0,
                    0.0,
                    3.0 * np.sqrt(110) / 88.0,
                    0.0,
                    -5.0 * np.sqrt(6) / 8.0,
                    0.0,
                ],
                [0.0, 0.0, 5.0 * np.sqrt(33) / 22.0, 0.0, -np.sqrt(110) / 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0 * np.sqrt(55) / 22.0, 0.0, 5.0 * np.sqrt(22) / 22.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -5.0 * np.sqrt(33) / 176.0,
                    0.0,
                    -np.sqrt(770) / 352.0,
                    0.0,
                    5.0 * np.sqrt(231) / 176.0,
                    0.0,
                    15.0 * np.sqrt(14) / 32.0,
                ],
                [
                    0.0,
                    -5.0 * np.sqrt(6) / 8.0,
                    0.0,
                    -3.0 * np.sqrt(110) / 88.0,
                    0.0,
                    5.0 * np.sqrt(11) / 44.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0 * np.sqrt(385) / 308.0, 0.0, 0.0, 0.0, -9.0 * np.sqrt(55) / 44.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0 * np.sqrt(110) / 22.0, 0.0, -5.0 * np.sqrt(11) / 22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0 * np.sqrt(33) / 22.0, 0.0, np.sqrt(770) / 22.0, 0.0, 0.0, 0.0, 0.0],
                [
                    3.0 * np.sqrt(42) / 16.0,
                    0.0,
                    np.sqrt(693) / 44.0,
                    0.0,
                    np.sqrt(2310) / 176.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    5.0 * np.sqrt(231) / 264.0,
                    0.0,
                    3.0 * np.sqrt(2310) / 176.0,
                    0.0,
                    5.0 * np.sqrt(14) / 16.0,
                    0.0,
                ],
                [0.0, 0.0, -5.0 * np.sqrt(33) / 22.0, 0.0, -np.sqrt(110) / 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0 * np.sqrt(11) / 22.0, 0.0, -3.0 * np.sqrt(110) / 22.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, np.sqrt(2310) / 33.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(231) / 11.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -5.0 / 16.0,
                    0.0,
                    -np.sqrt(210) / 32.0,
                    0.0,
                    -3.0 * np.sqrt(7) / 16.0,
                    0.0,
                    -np.sqrt(462) / 32.0,
                ],
                [
                    0.0,
                    3.0 * np.sqrt(14) / 16.0,
                    0.0,
                    3.0 * np.sqrt(2310) / 176.0,
                    0.0,
                    5.0 * np.sqrt(231) / 88.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    15.0 * np.sqrt(33) / 88.0,
                    0.0,
                    np.sqrt(770) / 22.0,
                    0.0,
                    5.0 * np.sqrt(231) / 88.0,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.0, -5.0 * np.sqrt(22) / 22.0, 0.0, -5.0 * np.sqrt(55) / 22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0 * np.sqrt(33) / 22.0, 0.0, -np.sqrt(770) / 22.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(231) / 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    }

    if l not in precomputed:
        raise NotImplementedError("Cartesian conversion is implemented for l up to 6.")

    return precomputed[l]


def _spherical_to_cart_matrix(l: int) -> np.ndarray:
    """Spherical -> Cartesian transform as the transpose of the precomputed table.

    Only ``_cart_to_spherical_matrix`` stores the full analytic values; this helper
    exposes the inverse direction for readability and reuse.
    """
    return _cart_to_spherical_matrix(l).T


'''
if __name__ == "__main__":

    def assign_electrons_to_atoms(electrons, coords):
        """Assign electrons to atoms.

        Given electron positions and atom coordinates, assign each electron to its nearest atom index.
        Returns an integer array of length len(electrons) with values in [0..nion-1].

        """
        assignments = []
        for e in electrons:
            d = np.sqrt(np.sum((coords - e) ** 2, axis=1))
            assignments.append(np.argmin(d))
        return np.array(assignments, dtype=int)

    test_systems = {
        "H2": {
            "charges": np.array([1.0, 1.0]),
            "coords": np.array([[-0.37, 0.0, 0.0], [0.37, 0.0, 0.0]]),
            "tot_up": 1,
            "tot_dn": 1,
        },
        "Li2": {
            "charges": np.array([3.0, 3.0]),
            "coords": np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "tot_up": 3,
            "tot_dn": 3,
        },
        "H2O_ae_dimer": {
            "charges": np.array([1.0, 8.0, 1.0, 1.0, 8.0, 1.0]),
            "coords": np.array(
                [
                    [0.00, 0.00, 0.00],
                    [0.96, 0.00, 0.00],
                    [-0.24, 0.93, 0.00],
                    [5.00, 0.00, 0.00],
                    [5.96, 0.00, 0.00],
                    [4.76, 0.93, 0.00],
                ]
            ),
            "tot_up": 10,
            "tot_dn": 10,
        },
        "H2O_ecp_dimer": {
            "charges": np.array([6.0, 1.0, 1.0, 6.0, 1.0, 1.0]),
            "coords": np.array(
                [
                    [-1.32695823, -0.10593853, 0.01878815],
                    [-1.93166524, 1.60017432, -0.02171052],
                    [0.48664428, 0.07959809, 0.00986248],
                    [4.19683807, 0.05048742, 0.00117253],
                    [4.90854978, -0.77793084, 1.44893779],
                    [4.90031568, -0.84942468, -1.40743405],
                ]
            ),
            "tot_up": 8,
            "tot_dn": 8,
        },
        "N2": {
            "charges": np.array([7.0, 7.0]),
            "coords": np.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]),
            "tot_up": 10,
            "tot_dn": 4,
        },
        "O2": {
            "charges": np.array([8.0, 8.0]),
            "coords": np.array([[-0.58, 0.0, 0.0], [0.58, 0.0, 0.0]]),
            "tot_up": 9,
            "tot_dn": 7,
        },
    }

    np.random.seed(42)

    for name, sysinfo in test_systems.items():
        coords = sysinfo["coords"]
        charges = sysinfo["charges"]
        tot_up = sysinfo["tot_up"]
        tot_dn = sysinfo["tot_dn"]

        up_pos, dn_pos, up_owner, dn_owner = generate_init_electron_configurations(tot_up, tot_dn, 1, charges, coords)

        nion = coords.shape[0]
        up_counts = np.bincount(up_owner[0], minlength=nion)
        dn_counts = np.bincount(dn_owner[0], minlength=nion)

        print(f"System: {name}")
        print(" Atom indices:", np.arange(nion))
        print(" Charges:", sysinfo["charges"])
        print("  Up  counts:", up_counts)
        print("  Dn  counts:", dn_counts)
        print("  Total counts:", up_counts + dn_counts)
        print()
'''
