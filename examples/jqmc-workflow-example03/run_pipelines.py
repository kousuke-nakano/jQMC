#!/usr/bin/env python
"""Water Energy & Force: J3 + MCMC / LRDMC (2 patterns).

For a water molecule (ccECP, cc-pVTZ, Cartesian), this script:
  1. Converts the TREXIO file to ``hamiltonian_data.h5``
     with J2=exp + J3=ao-small.
  2. Optimizes wavefunction parameters via VMC_Workflow (100 opt steps).
  3. Launches MCMC and LRDMC production runs with ``atomic_force=True``.
  4. Prints a summary table with energies and forces.

Patterns
--------
  A) J3 + MCMC   — SR adaptive_learning_rate=True, delta=0.35
  B) J3 + LRDMC  — same VMC; LRDMC a=0.30
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
QUEUE_LABEL = "cores-4-mpi-4-gpu-4-omp-1-3h"
PILOT_QUEUE_LABEL = "cores-4-mpi-4-gpu-4-omp-1-30m"

# VMC
NUM_OPT_STEPS = 100
TARGET_VMC_ERROR = 3.0e-3

# MCMC / LRDMC
Dt = 2.0
ALAT = 0.30
TARGET_PROD_ERROR = 5.0e-4

# Common
MAX_TIME = 76000
MAX_CONTINUATION = 2
NUMBER_OF_WALKERS = 1024
POLL_INTERVAL = 300

# pySCF output
TREXIO_FILE = "water_ccecp_ccpvtz.h5"
PYSCF_OUTPUT = "water.out"

# pySCF script (embedded from local_pyscf.py)
PYSCF_SCRIPT = '''\
from pyscf import gto, scf
from pyscf.tools import trexio

filename = "{trexio_file}"

mol = gto.Mole()
mol.verbose = 5
mol.atom = """
    O    5.00000000   7.14707700   7.65097100
    H    4.06806600   6.94297500   7.56376100
    H    5.38023700   6.89696300   6.80798400
"""
mol.basis = "ccecp-ccpvtz"
mol.unit = "A"
mol.ecp = "ccecp"
mol.charge = 0
mol.spin = 0
mol.symmetry = False
mol.cart = True
mol.output = "{pyscf_output}"
mol.build()

mf = scf.HF(mol)
mf.max_cycle = 200
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
'''


# ── Helpers ───────────────────────────────────────────────────────
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


def format_val(val: float | None, err: float | None) -> str:
    """Format a value with error in parentheses, e.g. ``-17.2345(12)``."""
    if val is None or err is None:
        return "N/A"
    if err == 0:
        return f"{val:.5f}(0)"
    n_dec = max(0, -math.floor(math.log10(err)) + 1)
    err_in_last = round(err * 10**n_dec)
    return f"{val:.{n_dec}f}({err_in_last})"


# ── Step 0: Run pySCF locally ────────────────────────────────────
def run_pyscf(base_dir: str) -> float | None:
    """Run pySCF for water and return HF energy."""
    print("=" * 60)
    print("  Step 0: Run pySCF calculation (local)")
    print("=" * 60)

    trexio_path = os.path.join(base_dir, TREXIO_FILE)

    if os.path.isfile(trexio_path):
        print(f"  [skip] {TREXIO_FILE} already exists.")
    else:
        print(f"  [run]  pySCF for water ...")
        script_path = os.path.join(base_dir, "_local_pyscf.py")
        with open(script_path, "w") as f:
            f.write(
                PYSCF_SCRIPT.format(
                    trexio_file=TREXIO_FILE,
                    pyscf_output=PYSCF_OUTPUT,
                )
            )
        result = subprocess.run(
            [sys.executable, "_local_pyscf.py"],
            cwd=base_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  [FAIL]:\n{result.stderr}")
            return None
        print(f"  [done] {TREXIO_FILE}")

    hf_energy = extract_hf_energy(os.path.join(base_dir, PYSCF_OUTPUT))
    if hf_energy is not None:
        print(f"         HF energy = {hf_energy:.6f} Ha")
    else:
        print("         [warn] Could not extract HF energy")
    return hf_energy


# ── Step 1: Build pipelines for J3 + MCMC / LRDMC ────────────────
def build_pipeline() -> tuple[
    list[Container],
    dict[str, Container],
]:
    """Build Container list for J3 + MCMC / LRDMC.

    Returns
    -------
    all_workflows : list[Container]
        Flat list of all containers (for Launcher).
    result_containers : dict[str, Container]
        Keyed by ``"mcmc"`` and ``"lrdmc"``.
    """
    workflows: list[Container] = []
    result_containers: dict[str, Container] = {}

    # ==================================================================
    # J2=exp + J3=ao-small  →  MCMC + LRDMC
    # ==================================================================

    # WF: TREXIO → hamiltonian_data.h5  (J1=None, J2=exp, J3=ao-small)
    wf = Container(
        label="wf",
        dirname="01_wf",
        input_files=[TREXIO_FILE],
        workflow=WF_Workflow(
            trexio_file=TREXIO_FILE,
            j1_parameter=None,
            j2_parameter=1.0,
            j3_basis_type="ao-small",
        ),
    )

    # VMC: SR with adaptive_learning_rate=True, delta=0.35
    vmc = Container(
        label="vmc",
        dirname="02_vmc",
        input_files=[FileFrom("wf", "hamiltonian_data.h5")],
        workflow=VMC_Workflow(
            server_machine_name=SERVER,
            queue_label=QUEUE_LABEL,
            pilot_queue_label=PILOT_QUEUE_LABEL,
            jobname="vmc-water",
            Dt=Dt,
            number_of_walkers=NUMBER_OF_WALKERS,
            num_opt_steps=NUM_OPT_STEPS,
            num_mcmc_bin_blocks=10,
            pilot_mcmc_steps=50,
            pilot_vmc_steps=20,
            opt_J1_param=False,
            opt_J2_param=True,
            opt_J3_param=True,
            opt_JNN_param=False,
            opt_lambda_param=False,
            opt_with_projected_MOs=False,
            target_error=TARGET_VMC_ERROR,
            optimizer_kwargs={
                "method": "sr",
                "delta": 0.350,
                "epsilon": 0.001,
                "adaptive_learning_rate": True,
            },
            max_time=MAX_TIME,
            poll_interval=POLL_INTERVAL,
            max_continuation=MAX_CONTINUATION,
        ),
    )

    # MCMC: production with force
    mcmc = Container(
        label="mcmc",
        dirname="03_mcmc",
        input_files=[
            FileFrom("vmc", f"hamiltonian_data_opt_step_{NUM_OPT_STEPS}.h5"),
        ],
        rename_input_files=["hamiltonian_data.h5"],
        workflow=MCMC_Workflow(
            server_machine_name=SERVER,
            queue_label=QUEUE_LABEL,
            jobname="mcmc-water",
            Dt=Dt,
            number_of_walkers=NUMBER_OF_WALKERS,
            target_error=TARGET_PROD_ERROR,
            atomic_force=True,
            num_mcmc_warmup_steps=30,
            num_mcmc_bin_blocks=10,
            pilot_steps=100,
            max_time=MAX_TIME,
            poll_interval=POLL_INTERVAL,
            max_continuation=MAX_CONTINUATION,
        ),
    )

    # LRDMC: production with force (a=0.30)
    lrdmc = Container(
        label="lrdmc",
        dirname="04_lrdmc",
        input_files=[
            FileFrom("vmc", f"hamiltonian_data_opt_step_{NUM_OPT_STEPS}.h5"),
        ],
        rename_input_files=["hamiltonian_data.h5"],
        workflow=LRDMC_Workflow(
            server_machine_name=SERVER,
            queue_label=QUEUE_LABEL,
            jobname="lrdmc-water",
            alat=ALAT,
            number_of_walkers=NUMBER_OF_WALKERS,
            target_error=TARGET_PROD_ERROR,
            atomic_force=True,
            num_gfmc_warmup_steps=30,
            num_gfmc_bin_blocks=10,
            num_gfmc_collect_steps=20,
            pilot_steps=100,
            max_time=MAX_TIME,
            poll_interval=POLL_INTERVAL,
            max_continuation=MAX_CONTINUATION,
        ),
    )

    workflows.extend([wf, vmc, mcmc, lrdmc])
    result_containers["mcmc"] = mcmc
    result_containers["lrdmc"] = lrdmc

    return workflows, result_containers


# ── Step 2: Print summary table ──────────────────────────────────
def print_summary_table(
    hf_energy: float | None,
    result_containers: dict[str, Container],
) -> None:
    """Print a summary table of energies and forces."""
    print()
    print("=" * 100)
    print("  Water Energy & Force Summary  (ccECP / cc-pVTZ, Cartesian)")
    print(f"  LRDMC a = {ALAT},  VMC opt steps = {NUM_OPT_STEPS}")
    if hf_energy is not None:
        print(f"  HF energy = {hf_energy:.6f} Ha")
    print("=" * 100)
    print()

    PATTERNS = [
        ("mcmc", "MCMC "),
        ("lrdmc", "LRDMC"),
    ]

    header = f"| {'Pattern':>12} | {'Energy (Ha)':>17} | {'Force (Ha/bohr)':>17} |"
    separator = f"|{'-' * 14}|{'-' * 19}|{'-' * 19}|"
    print(header)
    print(separator)

    for key, label in PATTERNS:
        ctr = result_containers.get(key)
        if ctr is not None and ctr.output_values:
            e = ctr.output_values.get("energy")
            e_err = ctr.output_values.get("energy_error")
            f_val = ctr.output_values.get("force")
            f_err = ctr.output_values.get("force_error")
        else:
            e, e_err, f_val, f_err = None, None, None, None

        e_str = format_val(e, e_err)
        f_str = format_val(f_val, f_err)

        row = f"| {label:>12} | {e_str:>17} | {f_str:>17} |"
        print(row)

    print()


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # 0) pySCF (local)
    hf_energy = run_pyscf(base_dir)

    # 1) WF → VMC → {MCMC, LRDMC}
    print()
    print("=" * 60)
    print("  Step 1: WF → VMC → MCMC + LRDMC  (J3)")
    print("=" * 60)

    workflows, result_containers = build_pipeline()
    launcher = Launcher(workflows=workflows, draw_graph=True)
    launcher.launch()

    # 2) Summary table
    print_summary_table(hf_energy, result_containers)
