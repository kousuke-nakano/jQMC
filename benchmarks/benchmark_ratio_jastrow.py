"""Benchmark Jastrow ratio: brute force vs fast update helpers.

Configure problem sizes and repeats below. Timings are wall-clock seconds
including JAX `.block_until_ready()` to measure execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from jqmc.atomic_orbital import AOs_sphe_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _compute_ratio_Jastrow_part_debug,
    compute_ratio_Jastrow_part,
)
from jqmc.structure import Structure_data

# --- Configurable parameters -------------------------------------------------
SEED = 1
REPEATS = 5

PROBLEM_SIZES: Sequence[dict[str, int]] = (
    {"num_up": 8, "num_dn": 8, "num_nuc": 4, "num_orb": 12, "num_moves": 32, "l_max": 3},
    {"num_up": 16, "num_dn": 16, "num_nuc": 6, "num_orb": 20, "num_moves": 48, "l_max": 4},
    {"num_up": 24, "num_dn": 24, "num_nuc": 8, "num_orb": 28, "num_moves": 64, "l_max": 4},
)

J1_PARAM = 1.0
J2_PARAM = 1.0


@dataclass
class BenchmarkCase:
    num_up: int
    num_dn: int
    num_nuc: int
    num_orb: int
    num_moves: int
    l_max: int
    seed: int


def make_structure(num_nuc: int, seed: int) -> Structure_data:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1.0, 1.0, size=(num_nuc, 3))
    structure = Structure_data(
        pbc_flag=False,
        positions=positions,
        atomic_numbers=tuple([6] * num_nuc),
        element_symbols=tuple(["X"] * num_nuc),
        atomic_labels=tuple([f"X{i}" for i in range(num_nuc)]),
    )
    structure.sanity_check()
    return structure


def make_ao_data(structure: Structure_data, num_orb: int, l_max: int, seed: int) -> AOs_sphe_data:
    rng = np.random.default_rng(seed)
    nucleus_index = tuple(i % len(structure.positions) for i in range(num_orb))
    exponents = tuple(rng.uniform(0.5, 2.0) for _ in range(num_orb))
    coefficients = tuple(rng.uniform(0.5, 1.5) for _ in range(num_orb))
    angular_momentums = []
    magnetic_quantum_numbers = []
    for _ in range(num_orb):
        l_val = int(rng.integers(0, l_max + 1))
        m_val = int(rng.integers(-l_val, l_val + 1)) if l_val > 0 else 0
        angular_momentums.append(l_val)
        magnetic_quantum_numbers.append(m_val)

    ao_data = AOs_sphe_data(
        structure_data=structure,
        nucleus_index=nucleus_index,
        num_ao=num_orb,
        num_ao_prim=num_orb,
        orbital_indices=tuple(range(num_orb)),
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=tuple(angular_momentums),
        magnetic_quantum_numbers=tuple(magnetic_quantum_numbers),
    )
    ao_data.sanity_check()
    return ao_data


def make_jastrow(case: BenchmarkCase) -> tuple[Jastrow_data, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    structure = make_structure(case.num_nuc, case.seed)
    ao_data = make_ao_data(structure, case.num_orb, case.l_max, case.seed + 1)

    rng = np.random.default_rng(case.seed + 2)
    r_up = rng.uniform(-2.0, 2.0, size=(case.num_up, 3))
    r_dn = rng.uniform(-2.0, 2.0, size=(case.num_dn, 3))

    jastrow_one_body_data = Jastrow_one_body_data(
        jastrow_1b_param=J1_PARAM, structure_data=structure, core_electrons=tuple([3] * case.num_nuc)
    )
    jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=J2_PARAM)
    j_matrix = rng.uniform(0.0, 1.0, size=(case.num_orb, case.num_orb + 1))
    jastrow_three_body_data = Jastrow_three_body_data(orb_data=ao_data, j_matrix=j_matrix)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_one_body_data,
        jastrow_two_body_data=jastrow_two_body_data,
        jastrow_three_body_data=jastrow_three_body_data,
        jastrow_nn_data=None,
    )

    # Proposed moves
    new_r_up = r_up + rng.normal(scale=0.05, size=(case.num_moves, case.num_up, 3))
    new_r_dn = r_dn + rng.normal(scale=0.05, size=(case.num_moves, case.num_dn, 3))

    return jastrow_data, r_up, r_dn, new_r_up, new_r_dn


def time_bruteforce(
    jastrow_data: Jastrow_data, r_up: np.ndarray, r_dn: np.ndarray, new_up: np.ndarray, new_dn: np.ndarray
) -> float:
    start = time.perf_counter()
    _compute_ratio_Jastrow_part_debug(
        jastrow_data=jastrow_data,
        old_r_up_carts=r_up,
        old_r_dn_carts=r_dn,
        new_r_up_carts_arr=new_up,
        new_r_dn_carts_arr=new_dn,
    )
    end = time.perf_counter()
    return end - start


def time_fast(jastrow_data: Jastrow_data, r_up: np.ndarray, r_dn: np.ndarray, new_up: np.ndarray, new_dn: np.ndarray) -> float:
    # Warm-up compile
    res = compute_ratio_Jastrow_part(
        jastrow_data=jastrow_data,
        old_r_up_carts=jnp.asarray(r_up),
        old_r_dn_carts=jnp.asarray(r_dn),
        new_r_up_carts_arr=jnp.asarray(new_up),
        new_r_dn_carts_arr=jnp.asarray(new_dn),
    )
    res.block_until_ready()

    start = time.perf_counter()
    res = compute_ratio_Jastrow_part(
        jastrow_data=jastrow_data,
        old_r_up_carts=jnp.asarray(r_up),
        old_r_dn_carts=jnp.asarray(r_dn),
        new_r_up_carts_arr=jnp.asarray(new_up),
        new_r_dn_carts_arr=jnp.asarray(new_dn),
    )
    res.block_until_ready()
    end = time.perf_counter()
    return end - start


def run_one(case: BenchmarkCase) -> None:
    jastrow_data, r_up, r_dn, new_up, new_dn = make_jastrow(case)

    brute_times = []
    fast_times = []
    for _ in range(REPEATS):
        brute_times.append(time_bruteforce(jastrow_data, r_up, r_dn, new_up, new_dn))
        fast_times.append(time_fast(jastrow_data, r_up, r_dn, new_up, new_dn))

    print(
        f"num_up={case.num_up:3d} num_dn={case.num_dn:3d} num_orb={case.num_orb:3d} moves={case.num_moves:3d} | "
        f"brute={np.mean(brute_times):.6f}s fast={np.mean(fast_times):.6f}s"
    )


def main() -> None:
    print("Benchmarking Jastrow ratio: brute force vs fast update")
    print(f"REPEATS={REPEATS}, SEED={SEED}\n")
    for idx, cfg in enumerate(PROBLEM_SIZES):
        case = BenchmarkCase(**cfg, seed=SEED + 10 * idx)
        run_one(case)


if __name__ == "__main__":
    main()
