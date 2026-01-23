"""Quick diagnostic script to evaluate the Jastrow factor from a Hamiltonian file.

Usage (from repo root):

    python tests/check_jastrow_from_hamiltonian.py \
        --ham-file tests/hamiltonian_data.chk \
        --num-samples 10 --seed 0 --attach-nn

It loads the pickled ``hamiltonian_data.chk``, optionally attaches an NN J3 term
(if the file was created before ``--j-nn-type`` existed), generates random
electron coordinates, and evaluates the total Jastrow value to check for NaNs.
"""

from __future__ import annotations

import argparse
import functools
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, NN_Jastrow_data, compute_Jastrow_part

jax.config.update("jax_enable_x64", True)


def _sample_electron_positions(structure, rng: np.random.Generator, num_electrons: int, sigma: float) -> np.ndarray:
    """Draw electron positions near random nuclei to mimic a molecular geometry."""
    if num_electrons == 0:
        return np.zeros((0, 3))

    positions = np.asarray(structure.positions)
    if positions.size == 0:
        centers = np.zeros((num_electrons, 3))
    else:
        indices = rng.integers(0, len(positions), size=num_electrons)
        centers = positions[indices]
    noise = rng.normal(scale=sigma, size=(num_electrons, 3))
    return centers + noise


def _maybe_attach_nn(jastrow_data: Jastrow_data, structure_data, args: argparse.Namespace) -> Jastrow_data:
    """Attach an NN J3 term if requested and missing from the Hamiltonian data."""
    if jastrow_data is None:
        raise ValueError("Hamiltonian wavefunction lacks Jastrow data.")

    if not args.attach_nn:
        return jastrow_data

    if jastrow_data.nn_jastrow_data is not None:
        print("Existing NN J3 detected; skipping --attach-nn.")
        return jastrow_data

    if structure_data is None:
        raise ValueError("Cannot attach NN J3 without structure information.")

    key = jax.random.PRNGKey(args.seed)
    nn_data = NN_Jastrow_data.init_from_structure(
        structure_data=structure_data,
        hidden_dim=args.nn_hidden_dim,
        num_layers=args.nn_num_layers,
        cutoff=args.nn_cutoff,
        key=key,
    )

    print(f"Attached NN J3 term (hidden_dim={args.nn_hidden_dim}, layers={args.nn_num_layers}, cutoff={args.nn_cutoff}).")

    return jastrow_data.replace(nn_jastrow_data=nn_data)


def _make_value_and_grad_fn(jastrow_data: Jastrow_data):
    """Return a callable that yields J value and gradients wrt electron coordinates."""
    value_fn = functools.partial(compute_Jastrow_part, jastrow_data)
    return jax.value_and_grad(value_fn, argnums=(0, 1))


def run_diagnostic(args: argparse.Namespace) -> int:
    """Evaluate the Jastrow factor for several random electron configurations."""
    ham_path = Path(args.ham_file)
    if not ham_path.exists():
        raise FileNotFoundError(f"Hamiltonian file not found: {ham_path}")

    with ham_path.open("rb") as f:
        hamiltonian: Hamiltonian_data = pickle.load(f)

    wf_data = hamiltonian.wavefunction_data
    jastrow_data = _maybe_attach_nn(wf_data.jastrow_data, hamiltonian.structure_data, args)
    print(jastrow_data)

    geminal = wf_data.geminal_data
    n_up = geminal.num_electron_up
    n_dn = geminal.num_electron_dn

    rng = np.random.default_rng(args.seed)
    value_and_grad_fn = _make_value_and_grad_fn(jastrow_data)

    finite_flags: list[bool] = []
    grad_finite_flags: list[bool] = []

    for idx in range(args.num_samples):
        r_up = _sample_electron_positions(hamiltonian.structure_data, rng, n_up, args.position_sigma)
        r_dn = _sample_electron_positions(hamiltonian.structure_data, rng, n_dn, args.position_sigma)

        try:
            J_value, (grad_up, grad_dn) = value_and_grad_fn(jnp.asarray(r_up), jnp.asarray(r_dn))
            J_value = float(J_value)
        except FloatingPointError as err:
            print(f"Sample {idx}: FloatingPointError -> {err}")
            finite_flags.append(False)
            grad_finite_flags.append(False)
            continue

        is_finite = bool(np.isfinite(J_value))
        finite_flags.append(is_finite)

        grad_up_np = np.asarray(grad_up)
        grad_dn_np = np.asarray(grad_dn)
        grads_are_finite = bool(np.all(np.isfinite(grad_up_np)) and np.all(np.isfinite(grad_dn_np)))
        grad_finite_flags.append(grads_are_finite)

        grad_components = []
        if grad_up_np.size:
            grad_components.append(np.abs(grad_up_np).ravel())
        if grad_dn_np.size:
            grad_components.append(np.abs(grad_dn_np).ravel())
        if grad_components:
            grad_inf_norm = float(np.max(np.concatenate(grad_components)))
        else:
            grad_inf_norm = 0.0

        print(
            f"Sample {idx}: J = {J_value:.6e} (finite={is_finite}); "
            f"grad finite={grads_are_finite}, ||grad||_inf={grad_inf_norm:.6e}"
        )

    num_finite = sum(finite_flags)
    num_grad_finite = sum(grad_finite_flags)
    print(f"\nSummary: {num_finite}/{args.num_samples} samples produced finite J values.")
    print(f"Gradients: {num_grad_finite}/{args.num_samples} samples produced finite gradients.")
    return 0 if (num_finite == args.num_samples and num_grad_finite == args.num_samples) else 1


def main() -> None:
    """Entry point when executing the script from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate Jastrow contributions in a Hamiltonian file.")
    parser.add_argument("--ham-file", default="hamiltonian_data.chk", help="Path to hamiltonian_data.chk")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of random electron configurations to test.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--position-sigma",
        type=float,
        default=0.5,
        help="Standard deviation (Bohr) of the Gaussian displacement around nuclei.",
    )
    parser.add_argument(
        "--attach-nn",
        action="store_true",
        help="If set and the Hamiltonian lacks an NN J3 term, build one on the fly from the stored structure.",
    )
    parser.add_argument("--nn-hidden-dim", type=int, default=64, help="Hidden dimension for the NN J3.")
    parser.add_argument("--nn-num-layers", type=int, default=3, help="Number of interaction blocks for NN J3.")
    parser.add_argument("--nn-cutoff", type=float, default=5.0, help="Cutoff radius (Bohr) for NN J3.")

    args = parser.parse_args()
    exit_code = run_diagnostic(args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
