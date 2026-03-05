#!/usr/bin/env python
"""Walker-scaling benchmark: pySCF → WF → VMC → MCMC + LRDMC.

For a water molecule (ccECP, cc-pVTZ), this script benchmarks
vectorization efficiency by sweeping the ``number_of_walkers``
parameter from 8 to 8192.

Pipeline
--------
  1. Run pySCF locally → TREXIO file.
  2. Convert to ``hamiltonian_data.h5`` with JSD Jastrow (WF_Workflow).
  3. Optimize J1/J2/J3 via VMC_Workflow.
  4. For **each walker count**, launch:
     - MCMC with explicit ``num_mcmc_steps``
     - LRDMC (GFMC_n, nmpm=40, a=0.30) with explicit ``num_mcmc_steps``
  5. Print a summary table with energies and wall times.

Step-count control
------------------
MCMC and LRDMC use an **explicit** number of measurement steps
rather than target-error–based automatic convergence.  Internally
we set ``pilot_steps = NUM_MCMC_STEPS`` and a large ``target_error``
(999 Ha) so that (a) the pilot phase runs with NUM_MCMC_STEPS and
(b) the production run uses the same step count (since
``estimate_required_steps`` clamps to ``max(computed, pilot_steps)``).
After one production run the error is trivially below 999 Ha and
the workflow exits.  The energy reported from ``output_values``
comes from the production run with exactly NUM_MCMC_STEPS steps.
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
QUEUE_LABEL_s = "cores-4-mpi-4-gpu-4-omp-1-30m"
QUEUE_LABEL_l = "cores-4-mpi-4-gpu-4-omp-1-3h"

NUM_OPT_STEPS = 50  # VMC optimization steps
Dt = 2.0  # MCMC hopping distance (bohr)
TARGET_VMC_ERROR = 1.0e-3  # Target VMC optimization error (Ha)
ALAT = 0.30  # LRDMC lattice spacing
NUM_MCMC_PER_MEASUREMENT = 40  # LRDMC GFMC projections per measurement

# Explicit measurement-step counts for the benchmark
# (no target-error automatic convergence)
NUM_MCMC_STEPS_MCMC = 1000  # MCMC measurement steps per production
NUM_MCMC_STEPS_LRDMC = 1000  # LRDMC measurement steps per production

# Walker counts for vectorization benchmark
WALKER_COUNTS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# pySCF file names
TREXIO_FILE = "water_trexio.hdf5"
PYSCF_OUTPUT = "water.out"


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


def format_energy(energy: float | None, error: float | None) -> str:
    """Format energy as ``-X.XXXXX(YY)`` (error in last digits)."""
    if energy is None or error is None:
        return "N/A"
    if error == 0:
        return f"{energy:.5f}(0)"
    n_dec = max(0, -math.floor(math.log10(error)) + 1)
    err_in_last = round(error * 10**n_dec)
    return f"{energy:.{n_dec}f}({err_in_last})"


def parse_net_time_from_file(filepath: str) -> float | None:
    """Parse net computation time (sec) from a jQMC output file."""
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    # LRDMC pattern
    m = re.search(r"Net GFMC time without pre-compilations\s*=\s*([0-9.]+)\s*sec", text)
    if m:
        return float(m.group(1))
    # MCMC/VMC pattern
    m = re.search(r"Net total time for MCMC\s*=\s*([0-9.]+)\s*sec", text)
    if m:
        return float(m.group(1))
    return None


# ── Step 1: Run pySCF locally ────────────────────────────────────
def run_pyscf(base_dir: str) -> float | None:
    """Run pySCF for water and return HF energy."""
    print("=" * 60)
    print("  Step 1: Run pySCF calculation (local)")
    print("=" * 60)

    trexio_path = os.path.join(base_dir, TREXIO_FILE)

    if os.path.isfile(trexio_path):
        print(f"  [skip] {TREXIO_FILE} already exists.")
    else:
        print(f"  [run]  pySCF for water ...")
        result = subprocess.run(
            [sys.executable, "run_pyscf.py"],
            cwd=base_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  [FAIL]:\n{result.stderr}")
            return None
        print(f"  [done] {TREXIO_FILE}")

    pyscf_out = os.path.join(base_dir, PYSCF_OUTPUT)
    hf_energy = extract_hf_energy(pyscf_out)
    if hf_energy is not None:
        print(f"         HF energy = {hf_energy:.6f} Ha")
    else:
        print("         [warn] Could not extract HF energy")
    return hf_energy


# ── Step 2: Build WF → VMC → {MCMC, LRDMC} × walkers pipeline ──
def build_pipeline() -> tuple[
    list[Container],
    dict[int, Container],
    dict[int, Container],
]:
    """Build Container list sweeping walker counts.

    Returns (all_workflows, mcmc_containers, lrdmc_containers).
    Keys are walker counts.
    """
    workflows: list[Container] = []
    mcmc_containers: dict[int, Container] = {}
    lrdmc_containers: dict[int, Container] = {}

    # ── Common stages (run once) ──────────────────────────────────
    # WF: TREXIO → hamiltonian_data.h5 (JSD: J1 + J2 + J3-ao-small)
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

    # VMC: Jastrow optimization
    vmc = Container(
        label="vmc",
        dirname="02_vmc",
        input_files=[FileFrom("wf", "hamiltonian_data.h5")],
        workflow=VMC_Workflow(
            server_machine_name=SERVER,
            queue_label=QUEUE_LABEL_l,
            pilot_queue_label=QUEUE_LABEL_s,
            jobname="vmc-water",
            Dt=Dt,
            number_of_walkers=256,
            num_opt_steps=NUM_OPT_STEPS,
            num_mcmc_bin_blocks=10,
            pilot_mcmc_steps=50,
            pilot_vmc_steps=20,
            opt_J1_param=False,
            opt_J2_param=True,
            opt_J3_param=True,
            opt_lambda_param=False,
            opt_with_projected_MOs=False,
            target_error=TARGET_VMC_ERROR,
            optimizer_kwargs={
                "method": "sr",
                "delta": 0.350,
                "epsilon": 0.001,
                "adaptive_learning_rate": True,
            },
            max_time=3000,
            poll_interval=300,
            max_continuation=1,
        ),
    )

    workflows.extend([wf, vmc])

    # ── Per-walker-count stages ───────────────────────────────────
    for nw in WALKER_COUNTS:
        label_mcmc = f"mcmc-w{nw}"
        label_lrdmc = f"lrdmc-w{nw}"

        # MCMC with explicit step count
        mcmc = Container(
            label=label_mcmc,
            dirname=f"03_mcmc/w{nw:05d}",
            input_files=[
                FileFrom("vmc", f"hamiltonian_data_opt_step_{NUM_OPT_STEPS}.h5"),
            ],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=MCMC_Workflow(
                server_machine_name=SERVER,
                queue_label=QUEUE_LABEL_s,
                jobname=f"mcmc-w{nw}",
                Dt=Dt,
                number_of_walkers=nw,
                num_mcmc_warmup_steps=25,
                num_mcmc_bin_blocks=10,
                # Explicit step count: pilot_steps = desired steps,
                # target_error = huge → production uses same count.
                pilot_steps=NUM_MCMC_STEPS_MCMC,
                max_time=3000,
                poll_interval=300,
                max_continuation=1,
            ),
        )

        # LRDMC (GFMC_n: nmpm=40, a=0.30) with explicit step count
        lrdmc = Container(
            label=label_lrdmc,
            dirname=f"04_lrdmc/w{nw:05d}",
            input_files=[
                FileFrom("vmc", f"hamiltonian_data_opt_step_{NUM_OPT_STEPS}.h5"),
            ],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=LRDMC_Workflow(
                server_machine_name=SERVER,
                queue_label=QUEUE_LABEL_s,
                jobname=f"lrdmc-w{nw}",
                alat=ALAT,
                number_of_walkers=nw,
                num_mcmc_per_measurement=NUM_MCMC_PER_MEASUREMENT,
                num_gfmc_warmup_steps=25,
                num_gfmc_bin_blocks=10,
                num_gfmc_collect_steps=5,
                pilot_steps=NUM_MCMC_STEPS_LRDMC,
                max_time=3000,
                poll_interval=300,
                max_continuation=1,
            ),
        )

        workflows.extend([mcmc, lrdmc])
        mcmc_containers[nw] = mcmc
        lrdmc_containers[nw] = lrdmc

    return workflows, mcmc_containers, lrdmc_containers


# ── Step 3: Print summary table ──────────────────────────────────
def print_summary_table(
    base_dir: str,
    hf_energy: float | None,
    mcmc_containers: dict[int, Container],
    lrdmc_containers: dict[int, Container],
) -> None:
    """Print a summary table of energies and wall times."""
    print()
    print("=" * 120)
    print(f"  Walker-scaling Benchmark Summary  (water, ccECP/cc-pVTZ, JSD)")
    print(
        f"  MCMC: {NUM_MCMC_STEPS_MCMC} steps,  LRDMC: {NUM_MCMC_STEPS_LRDMC} steps (nmpm={NUM_MCMC_PER_MEASUREMENT}, a={ALAT})"
    )
    if hf_energy is not None:
        print(f"  HF energy = {hf_energy:.6f} Ha")
    print("=" * 120)
    print()

    header = (
        f"| {'Walkers':>8} | {'E_MCMC (Ha)':>17} | {'MCMC t_net (s)':>14} | {'E_LRDMC (Ha)':>17} | {'LRDMC t_net (s)':>15} |"
    )
    separator = f"|{'-' * 10}|{'-' * 19}|{'-' * 16}|{'-' * 19}|{'-' * 17}|"
    print(header)
    print(separator)

    for nw in WALKER_COUNTS:
        # MCMC energy
        mcmc_ctr = mcmc_containers.get(nw)
        if mcmc_ctr is not None and mcmc_ctr.output_values:
            mcmc_e = mcmc_ctr.output_values.get("energy")
            mcmc_err = mcmc_ctr.output_values.get("energy_error")
        else:
            mcmc_e, mcmc_err = None, None
        mcmc_e_str = format_energy(mcmc_e, mcmc_err)

        # MCMC wall time (from production output)
        mcmc_dir = os.path.join(base_dir, f"03_mcmc/w{nw:05d}")
        mcmc_time = parse_net_time_from_file(os.path.join(mcmc_dir, "out.o_1"))
        mcmc_t_str = f"{mcmc_time:.1f}" if mcmc_time is not None else "N/A"

        # LRDMC energy
        lrdmc_ctr = lrdmc_containers.get(nw)
        if lrdmc_ctr is not None and lrdmc_ctr.output_values:
            lrdmc_e = lrdmc_ctr.output_values.get("energy")
            lrdmc_err = lrdmc_ctr.output_values.get("energy_error")
        else:
            lrdmc_e, lrdmc_err = None, None
        lrdmc_e_str = format_energy(lrdmc_e, lrdmc_err)

        # LRDMC wall time (from production output)
        lrdmc_dir = os.path.join(base_dir, f"04_lrdmc/w{nw:05d}")
        lrdmc_time = parse_net_time_from_file(os.path.join(lrdmc_dir, "out.o_1"))
        lrdmc_t_str = f"{lrdmc_time:.1f}" if lrdmc_time is not None else "N/A"

        row = f"| {nw:>8} | {mcmc_e_str:>17} | {mcmc_t_str:>14} | {lrdmc_e_str:>17} | {lrdmc_t_str:>15} |"
        print(row)

    print()


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # 1) pySCF (local)
    hf_energy = run_pyscf(base_dir)

    # 2) WF → VMC → {MCMC, LRDMC} × walkers (via jqmc-workflow)
    print()
    print("=" * 60)
    print("  Step 2: WF → VMC → MCMC + LRDMC (via jqmc-workflow)")
    print("=" * 60)

    workflows, mcmc_containers, lrdmc_containers = build_pipeline()
    launcher = Launcher(workflows=workflows, draw_graph=True)
    launcher.launch()

    # 3) Summary table
    print_summary_table(base_dir, hf_energy, mcmc_containers, lrdmc_containers)
