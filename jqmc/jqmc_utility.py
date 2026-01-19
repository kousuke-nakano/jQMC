"""utility module."""

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
    """Generate initial electron configurations for walkers.

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
    r_carts_up = np.zeros((num_walkers, tot_num_electron_up, 3), dtype=float)
    r_carts_dn = np.zeros((num_walkers, tot_num_electron_dn, 3), dtype=float)
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
        down_positions = np.zeros((ned_dn, 3), dtype=float)
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
        up_positions = np.zeros((ned_up, 3), dtype=float)

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
