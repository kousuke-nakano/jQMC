#!/usr/bin/env python
"""Validation pipeline: pySCF → TREXIO → hamiltonian_data.h5 → MCMC (no Jastrow).

For each system directory, this script:
  1. Runs `run_pyscf.py` locally to produce a TREXIO HDF5 file.
  2. Extracts the HF energy from the pySCF output log.
  3. Converts it to `hamiltonian_data.h5` via WF_Workflow (no Jastrow).
  4. Launches an MCMC production run via MCMC_Workflow on the remote server.
  5. Prints a summary table comparing HF and MCMC energies.

All 12 systems are independent so their WF→MCMC chains run in parallel
once the DAG is submitted to the Launcher.
"""

import os
import re
import subprocess
import sys

from jqmc_workflow import (
    Container,
    FileFrom,
    Launcher,
    MCMC_Workflow,
    WF_Workflow,
)

# ── Remote server configuration ──────────────────────────────────
SERVER = "genkai"
QUEUE_LABEL = "cores-480-mpi-120-omp-1-24h"
PILOT_QUEUE_LABEL = "cores-120-mpi-120-omp-1-15m"

# ── Validation systems ───────────────────────────────────────────
# (dirname, trexio_file, system_label, spin, hf_type, basis, ecp, gto_type)
SYSTEMS = [
    ("water_pGTOs_RHF", "water_ccecp_ccpvqz_cart.h5", "H2O", 0, "RHF", "ccecp-ccpvqz", "ccECP", "Cartesian"),
    # ("water_sGTOs_RHF", "water_ccecp_ccpvqz_sphe.h5", "H2O", 0, "RHF", "ccecp-ccpvqz", "ccECP", "Spherical"),
    # ("Ar_pGTOs_RHF", "Ar_ccecp_ccpvqz_cart.h5", "Ar", 0, "RHF", "ccecp-ccpv5z", "ccECP", "Cartesian"),
    # ("Ar_sGTOs_RHF", "Ar_ccecp_ccpvqz_sphe.h5", "Ar", 0, "RHF", "ccecp-ccpv5z", "ccECP", "Spherical"),
    # ("N_pGTOs_ROHF", "N_ccecp_ccpvqz_cart.h5", "N", 3, "ROHF", "ccecp-ccpvqz", "ccECP", "Cartesian"),
    # ("N_sGTOs_ROHF", "N_ccecp_ccpvqz_sphe.h5", "N", 3, "ROHF", "ccecp-ccpvqz", "ccECP", "Spherical"),
    # ("N_pGTOs_UHF", "N_ccecp_ccpvqz_cart.h5", "N", 3, "UHF", "ccecp-ccpvqz", "ccECP", "Cartesian"),
    # ("N_sGTOs_UHF", "N_ccecp_ccpvqz_sphe.h5", "N", 3, "UHF", "ccecp-ccpvqz", "ccECP", "Spherical"),
    ("O2_pGTOs_ROHF", "O2_ccecp_ccpvqz_cart.h5", "O2", 2, "ROHF", "ccecp-ccpvqz", "ccECP", "Cartesian"),
    # ("O2_sGTOs_ROHF", "O2_ccecp_ccpvqz_sphe.h5", "O2", 2, "ROHF", "ccecp-ccpvqz", "ccECP", "Spherical"),
    ("O2_pGTOs_UHF", "O2_ccecp_ccpvqz_cart.h5", "O2", 2, "UHF", "ccecp-ccpvqz", "ccECP", "Cartesian"),
    # ("O2_sGTOs_UHF", "O2_ccecp_ccpvqz_sphe.h5", "O2", 2, "UHF", "ccecp-ccpvqz", "ccECP", "Spherical"),
]

# pySCF output file names (defined in each run_pyscf.py)
PYSCF_OUTPUT_FILES = {
    "water_pGTOs_RHF": "water.out",
    "water_sGTOs_RHF": "water.out",
    "Ar_pGTOs_RHF": "Ar.out",
    "Ar_sGTOs_RHF": "Ar.out",
    "N_pGTOs_ROHF": "N.out",
    "N_sGTOs_ROHF": "N.out",
    "N_pGTOs_UHF": "N.out",
    "N_sGTOs_UHF": "N.out",
    "O2_pGTOs_ROHF": "O2.out",
    "O2_sGTOs_ROHF": "O2.out",
    "O2_pGTOs_UHF": "O2.out",
    "O2_sGTOs_UHF": "O2.out",
}


# ── HF energy extraction ─────────────────────────────────────────
def extract_hf_energy(pyscf_output_path: str) -> float | None:
    """Parse the converged HF energy from pySCF output.

    Looks for lines like:
      converged SCF energy = -16.9450309201805
    """
    if not os.path.isfile(pyscf_output_path):
        return None
    pattern = re.compile(r"converged SCF energy\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)")
    with open(pyscf_output_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    return None


# ── MCMC energy formatting ───────────────────────────────────────
def format_mcmc_energy(energy: float | None, error: float | None) -> str:
    """Format MCMC energy as  -X.XXXXX(YY)  (error in last digits)."""
    if energy is None or error is None:
        return "N/A"
    return _format_with_2usf(energy, error)


def _format_with_2usf(value: float, error: float, sign: bool = False) -> str:
    """Format value(error) with 2 uncertain significant figures.

    Examples:
        _format_with_2usf(-16.94494, 0.00066)  -> "-16.94494(66)"
        _format_with_2usf(0.09, 0.66, sign=True) -> "+0.09(66)"
    """
    if error == 0:
        fmt = f"{value:+.5f}" if sign else f"{value:.5f}"
        return f"{fmt}(0)"

    import math

    # Number of decimal places so that error has >= 2 significant digits
    n_dec = max(0, -math.floor(math.log10(abs(error))) + 1)
    err_in_last = round(error * 10**n_dec)
    fmt = f"{value:+.{n_dec}f}" if sign else f"{value:.{n_dec}f}"
    return f"{fmt}({err_in_last})"


# ── Step 1: Run pySCF locally ────────────────────────────────────
def run_pyscf_calculations(base_dir: str) -> dict[str, float | None]:
    """Run run_pyscf.py in each system directory and return HF energies."""
    print("=" * 60)
    print("  Step 1: Run pySCF calculations (local)")
    print("=" * 60)

    hf_energies: dict[str, float | None] = {}

    for dirname, trexio_file, *_ in SYSTEMS:
        sys_dir = os.path.join(base_dir, dirname)
        trexio_path = os.path.join(sys_dir, trexio_file)

        if os.path.isfile(trexio_path):
            print(f"  [skip] {dirname}/{trexio_file} already exists.")
        else:
            print(f"  [run]  pySCF in {dirname} ...")
            result = subprocess.run(
                [sys.executable, "run_pyscf.py"],
                cwd=sys_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"  [FAIL] {dirname}:\n{result.stderr}")
                hf_energies[dirname] = None
                continue
            print(f"  [done] {dirname}/{trexio_file}")

        # Extract HF energy from pySCF output
        pyscf_out = os.path.join(sys_dir, PYSCF_OUTPUT_FILES[dirname])
        hf_energies[dirname] = extract_hf_energy(pyscf_out)
        if hf_energies[dirname] is not None:
            print(f"         HF energy = {hf_energies[dirname]:.5f} Ha")
        else:
            print(f"         [warn] Could not extract HF energy from {pyscf_out}")

    return hf_energies


# ── Step 2: Build WF → MCMC pipeline ─────────────────────────────
def build_pipeline() -> tuple[list[Container], dict[str, Container]]:
    """Build Container list and return (all_workflows, mcmc_containers_by_dirname)."""
    workflows: list[Container] = []
    mcmc_containers: dict[str, Container] = {}

    for dirname, trexio_file, *_ in SYSTEMS:
        label_wf = f"wf-{dirname}"
        label_mcmc = f"mcmc-{dirname}"

        # WF: TREXIO → hamiltonian_data.h5 (local, no Jastrow)
        wf = Container(
            label=label_wf,
            dirname=os.path.join(dirname, "00_wf"),
            input_files=[os.path.join(dirname, trexio_file)],
            workflow=WF_Workflow(
                trexio_file=trexio_file,
            ),
        )

        # MCMC: production sampling (remote, no Jastrow)
        mcmc = Container(
            label=label_mcmc,
            dirname=os.path.join(dirname, "01_mcmc"),
            input_files=[FileFrom(label_wf, "hamiltonian_data.h5")],
            workflow=MCMC_Workflow(
                server_machine_name=SERVER,
                queue_label=QUEUE_LABEL,
                pilot_queue_label=PILOT_QUEUE_LABEL,
                jobname=f"mcmc-{dirname}",
                target_error=0.001,
                pilot_steps=200,
                num_mcmc_warmup_steps=50,
                num_mcmc_bin_blocks=100,
            ),
        )

        workflows.extend([wf, mcmc])
        mcmc_containers[dirname] = mcmc

    return workflows, mcmc_containers


# ── Step 3: Print summary table ──────────────────────────────────
def print_summary_table(
    hf_energies: dict[str, float | None],
    mcmc_containers: dict[str, Container],
) -> None:
    """Print a Markdown-formatted validation summary table."""
    print()
    print("=" * 100)
    print("  Validation Summary")
    print("=" * 100)
    print()

    # Header
    header = (
        f"| {'System':<7} | {'Spin':<8} | {'Type':<8} | {'basis':<14} "
        f"| {'ECP':<7} | {'GTOs':<13} | {'HF (Ha)':<13} | {'MCMC (Ha)':<13} | {'Diff (mHa)':<13} |"
    )
    separator = f"|{'-' * 9}|{'-' * 10}|{'-' * 10}|{'-' * 16}|{'-' * 9}|{'-' * 15}|{'-' * 15}|{'-' * 15}|{'-' * 15}|"
    print(header)
    print(separator)

    for dirname, _, system, spin, hf_type, basis, ecp, gto_type in SYSTEMS:
        # HF energy
        hf_e = hf_energies.get(dirname)
        hf_str = f"{hf_e:.5f}" if hf_e is not None else "N/A"

        # MCMC energy from workflow output_values
        mcmc_ctr = mcmc_containers.get(dirname)
        if mcmc_ctr is not None and mcmc_ctr.output_values:
            mcmc_e = mcmc_ctr.output_values.get("energy")
            mcmc_err = mcmc_ctr.output_values.get("energy_error")
        else:
            mcmc_e, mcmc_err = None, None
        mcmc_str = format_mcmc_energy(mcmc_e, mcmc_err)

        # Diff = MCMC - HF in mHa (2 uncertain significant figures)
        if hf_e is not None and mcmc_e is not None:
            diff_val = (mcmc_e - hf_e) * 1000.0
            if mcmc_err is not None:
                diff_err = mcmc_err * 1000.0
                diff_str = _format_with_2usf(diff_val, diff_err, sign=True)
            else:
                diff_str = f"{diff_val:+.2f}"
        else:
            diff_str = "N/A"

        row = (
            f"| {system:<7} | {spin:<8} | {hf_type:<8} | {basis:<14} "
            f"| {ecp:<7} | {gto_type:<13} | {hf_str:>13} | {mcmc_str:>13} | {diff_str:>13} |"
        )
        print(row)

    print()


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # 1) pySCF (local)
    hf_energies = run_pyscf_calculations(base_dir)

    # 2) WF conversion + MCMC (via jqmc-workflow)
    print()
    print("=" * 60)
    print("  Step 2: WF conversion + MCMC (via jqmc-workflow)")
    print("=" * 60)

    workflows, mcmc_containers = build_pipeline()
    launcher = Launcher(workflows=workflows, draw_graph=True)
    launcher.launch()

    # 3) Summary table
    print_summary_table(hf_energies, mcmc_containers)
