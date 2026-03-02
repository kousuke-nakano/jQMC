#!/usr/bin/env python
"""H2 PES pipeline: pySCF → WF → VMC (JSD, MO opt) → MCMC + LRDMC (a=0.2).

For each bond length R, this script:
  1. Runs pySCF locally to produce a TREXIO HDF5 file.
  2. Converts it to ``hamiltonian_data.h5`` with JSD Jastrow via WF_Workflow.
  3. Optimizes J1/J2/J3 parameters and projected MOs via VMC_Workflow.
  4. Launches MCMC and LRDMC (a=0.2) production runs (in parallel).
  5. Prints a summary table with energies and atomic forces.

All R values are independent; their WF → VMC → {MCMC, LRDMC} chains
run in parallel once the DAG is submitted to the Launcher.
"""

import math
import os
import re
import subprocess
import sys

from jqmc_workflow import (
    Container,
    FileFrom,
    Launcher,
    LRDMC_Workflow,
    MCMC_Workflow,
    VMC_Workflow,
    WF_Workflow,
)

# ── Configuration ─────────────────────────────────────────────────
SERVER = "genkai"
QUEUE_LABEL = "cores-120-mpi-120-omp-1-15m"

NUM_OPT_STEPS = 30  # VMC optimization steps
ALAT = 0.2  # LRDMC lattice spacing
TARGET_ERROR = 1e-5  # Target statistical error (Ha)

R_VALUES = [
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.74,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.30,
    1.40,
]


# ── pySCF script template ────────────────────────────────────────
PYSCF_TEMPLATE = '''\
from pyscf import gto, scf
from pyscf.tools import trexio

R = {R}  # angstrom
filename = f"H2_R_{{R:.2f}}.h5"

mol = gto.Mole()
mol.verbose = 5
mol.atom = f"""
H    0.000000000   0.000000000  {{-R / 2}}
H    0.000000000   0.000000000  {{+R / 2}}
"""
mol.basis = "ccpvtz"
mol.unit = "A"
mol.ecp = None
mol.charge = 0
mol.spin = 0
mol.symmetry = False
mol.cart = True
mol.output = f"H2_R_{{R:.2f}}.out"
mol.build()

mf = scf.KS(mol).density_fit()
mf.max_cycle = 200
mf.xc = "LDA_X,LDA_C_PZ"
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
'''


# ── Helpers ───────────────────────────────────────────────────────
def r_dir(R: float) -> str:
    """Return the directory name for a given bond length."""
    return f"R_{R:.2f}"


def trexio_filename(R: float) -> str:
    """Return the TREXIO filename for a given bond length."""
    return f"H2_R_{R:.2f}.h5"


def extract_hf_energy(pyscf_output: str) -> float | None:
    """Parse the converged SCF energy from pySCF output."""
    if not os.path.isfile(pyscf_output):
        return None
    pattern = re.compile(r"converged SCF energy\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)")
    with open(pyscf_output) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    return None


def format_energy(energy: float | None, error: float | None) -> str:
    """Format energy as ``-X.XXXXX(YY)`` (error in last digits)."""
    if energy is None or error is None:
        return "N/A"
    if error == 0:
        return f"{energy:.5f}(0)"
    n_dec = max(0, -math.floor(math.log10(error)) + 1)
    err_in_last = round(error * 10**n_dec)
    return f"{energy:.{n_dec}f}({err_in_last})"


def format_force(force: float | None, error: float | None) -> str:
    """Format atomic force similarly to energy."""
    if force is None or error is None:
        return "N/A"
    if error == 0:
        return f"{force:.5f}(0)"
    n_dec = max(0, -math.floor(math.log10(error)) + 1)
    err_in_last = round(error * 10**n_dec)
    return f"{force:.{n_dec}f}({err_in_last})"


# ── Step 1: Run pySCF locally ────────────────────────────────────
def run_pyscf_calculations(base_dir: str) -> dict[float, float | None]:
    """Run pySCF for each R value and return HF energies."""
    print("=" * 60)
    print("  Step 1: Run pySCF calculations (local)")
    print("=" * 60)

    hf_energies: dict[float, float | None] = {}

    for R in R_VALUES:
        pyscf_dir = os.path.join(base_dir, r_dir(R), "00_pyscf")
        os.makedirs(pyscf_dir, exist_ok=True)

        trexio_path = os.path.join(pyscf_dir, trexio_filename(R))
        script_path = os.path.join(pyscf_dir, "run_pyscf.py")

        # Write the pySCF script
        with open(script_path, "w") as f:
            f.write(PYSCF_TEMPLATE.format(R=R))

        if os.path.isfile(trexio_path):
            print(f"  [skip] {r_dir(R)}/00_pyscf/{trexio_filename(R)} already exists.")
        else:
            print(f"  [run]  pySCF for R={R:.2f} ...")
            result = subprocess.run(
                [sys.executable, "run_pyscf.py"],
                cwd=pyscf_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"  [FAIL] R={R:.2f}:\n{result.stderr}")
                hf_energies[R] = None
                continue
            print(f"  [done] {r_dir(R)}/00_pyscf/{trexio_filename(R)}")

        # Extract HF energy
        pyscf_out = os.path.join(pyscf_dir, f"H2_R_{R:.2f}.out")
        hf_energies[R] = extract_hf_energy(pyscf_out)
        if hf_energies[R] is not None:
            print(f"         HF energy = {hf_energies[R]:.6f} Ha")
        else:
            print("         [warn] Could not extract HF energy")

    return hf_energies


# ── Step 2: Build WF → VMC → {MCMC, LRDMC} pipeline ─────────────
def build_pipeline() -> tuple[list[Container], dict[float, Container], dict[float, Container]]:
    """Build Container list for all R values.

    Returns (all_workflows, mcmc_containers, lrdmc_containers).
    """
    workflows: list[Container] = []
    mcmc_containers: dict[float, Container] = {}
    lrdmc_containers: dict[float, Container] = {}

    for R in R_VALUES:
        label_wf = f"wf-{R:.2f}"
        label_vmc = f"vmc-{R:.2f}"
        label_mcmc = f"mcmc-{R:.2f}"
        label_lrdmc = f"lrdmc-{R:.2f}"
        trexio_file = trexio_filename(R)

        # WF: TREXIO → hamiltonian_data.h5 (JSD: J1 + J2 + J3-MO)
        wf = Container(
            label=label_wf,
            dirname=os.path.join(r_dir(R), "01_wf"),
            input_files=[os.path.join(r_dir(R), "00_pyscf", trexio_file)],
            workflow=WF_Workflow(
                trexio_file=trexio_file,
                j1_parameter=1.0,
                j2_parameter=1.0,
                j3_basis_type="ao-small",
            ),
        )

        # VMC: Jastrow + projected-MO optimization
        vmc = Container(
            label=label_vmc,
            dirname=os.path.join(r_dir(R), "02_vmc"),
            input_files=[FileFrom(label_wf, "hamiltonian_data.h5")],
            workflow=VMC_Workflow(
                server_machine_name=SERVER,
                queue_label=QUEUE_LABEL,
                jobname=f"vmc-H2-{R:.2f}",
                num_opt_steps=NUM_OPT_STEPS,
                opt_J1_param=True,
                opt_J2_param=True,
                opt_J3_param=True,
                opt_lambda_param=True,
                opt_with_projected_MOs=True,
                target_error=TARGET_ERROR,
            ),
        )

        # MCMC: production sampling with atomic forces
        mcmc = Container(
            label=label_mcmc,
            dirname=os.path.join(r_dir(R), "03_mcmc"),
            input_files=[
                FileFrom(label_vmc, f"hamiltonian_data_opt_step_{NUM_OPT_STEPS}.h5"),
            ],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=MCMC_Workflow(
                server_machine_name=SERVER,
                queue_label=QUEUE_LABEL,
                jobname=f"mcmc-H2-{R:.2f}",
                target_error=TARGET_ERROR,
                atomic_force=True,
                num_mcmc_warmup_steps=50,
                num_mcmc_bin_blocks=50,
                pilot_steps=200,
            ),
        )

        # LRDMC: diffusion Monte Carlo with atomic forces
        lrdmc = Container(
            label=label_lrdmc,
            dirname=os.path.join(r_dir(R), "04_lrdmc"),
            input_files=[
                FileFrom(label_vmc, f"hamiltonian_data_opt_step_{NUM_OPT_STEPS}.h5"),
            ],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=LRDMC_Workflow(
                server_machine_name=SERVER,
                queue_label=QUEUE_LABEL,
                jobname=f"lrdmc-H2-{R:.2f}",
                alat=ALAT,
                target_error=TARGET_ERROR,
                atomic_force=True,
                num_gfmc_warmup_steps=50,
                num_gfmc_bin_blocks=50,
                num_gfmc_collect_steps=20,
                pilot_steps=200,
            ),
        )

        workflows.extend([wf, vmc, mcmc, lrdmc])
        mcmc_containers[R] = mcmc
        lrdmc_containers[R] = lrdmc

    return workflows, mcmc_containers, lrdmc_containers


# ── Step 3: Print summary table ──────────────────────────────────
def print_summary_table(
    hf_energies: dict[float, float | None],
    mcmc_containers: dict[float, Container],
    lrdmc_containers: dict[float, Container],
) -> None:
    """Print a summary table of energies and forces for all R values."""
    print()
    print("=" * 110)
    print("  H2 PES Summary  (JSD, MO opt, all-electron, cc-pVTZ)")
    print("=" * 110)
    print()

    header = (
        f"| {'R (Å)':>6} "
        f"| {'E_HF (Ha)':>13} "
        f"| {'E_MCMC (Ha)':>15} "
        f"| {'F_MCMC (Ha/Å)':>15} "
        f"| {'E_LRDMC (Ha)':>15} "
        f"| {'F_LRDMC (Ha/Å)':>16} |"
    )
    separator = f"|{'-' * 8}|{'-' * 15}|{'-' * 17}|{'-' * 17}|{'-' * 17}|{'-' * 18}|"
    print(header)
    print(separator)

    for R in R_VALUES:
        # HF energy
        hf_e = hf_energies.get(R)
        hf_str = f"{hf_e:.6f}" if hf_e is not None else "N/A"

        # MCMC energy and force
        mcmc_ctr = mcmc_containers.get(R)
        if mcmc_ctr is not None and mcmc_ctr.output_values:
            mcmc_e = mcmc_ctr.output_values.get("energy")
            mcmc_err = mcmc_ctr.output_values.get("energy_error")
            mcmc_f = mcmc_ctr.output_values.get("force")
            mcmc_ferr = mcmc_ctr.output_values.get("force_error")
        else:
            mcmc_e, mcmc_err, mcmc_f, mcmc_ferr = None, None, None, None
        mcmc_e_str = format_energy(mcmc_e, mcmc_err)
        mcmc_f_str = format_force(mcmc_f, mcmc_ferr)

        # LRDMC energy and force
        lrdmc_ctr = lrdmc_containers.get(R)
        if lrdmc_ctr is not None and lrdmc_ctr.output_values:
            lrdmc_e = lrdmc_ctr.output_values.get("energy")
            lrdmc_err = lrdmc_ctr.output_values.get("energy_error")
            lrdmc_f = lrdmc_ctr.output_values.get("force")
            lrdmc_ferr = lrdmc_ctr.output_values.get("force_error")
        else:
            lrdmc_e, lrdmc_err, lrdmc_f, lrdmc_ferr = None, None, None, None
        lrdmc_e_str = format_energy(lrdmc_e, lrdmc_err)
        lrdmc_f_str = format_force(lrdmc_f, lrdmc_ferr)

        row = f"| {R:6.2f} | {hf_str:>13} | {mcmc_e_str:>15} | {mcmc_f_str:>15} | {lrdmc_e_str:>15} | {lrdmc_f_str:>16} |"
        print(row)

    print()


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # 1) pySCF (local)
    hf_energies = run_pyscf_calculations(base_dir)

    # 2) WF → VMC → {MCMC, LRDMC} (via jqmc-workflow)
    print()
    print("=" * 60)
    print("  Step 2: WF → VMC → MCMC + LRDMC (via jqmc-workflow)")
    print("=" * 60)

    workflows, mcmc_containers, lrdmc_containers = build_pipeline()
    launcher = Launcher(workflows=workflows, draw_graph=True)
    launcher.launch()

    # 3) Summary table
    print_summary_table(hf_energies, mcmc_containers, lrdmc_containers)
