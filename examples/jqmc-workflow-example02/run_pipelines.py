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

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from jqmc_workflow import (
    Container,
    FileFrom,
    Launcher,
    LRDMC_Workflow,
    MCMC_Workflow,
    ValueFrom,
    VMC_Workflow,
    WF_Workflow,
    parse_lrdmc_output,
    parse_mcmc_output,
)

# ── Configuration ─────────────────────────────────────────────────
SERVER = "cluster"
QUEUE_LABEL_s = "cores-4-mpi-4-gpu-4-omp-1-30m"
QUEUE_LABEL_l = "cores-4-mpi-4-gpu-4-omp-1-3h"

NUM_OPT_STEPS = 50  # VMC optimization steps
WF_DUMP_FREQ = 10  # WF dumping freq
Dt = 2.0  # MCMC hopping distance (bohr)
TARGET_VMC_ERROR = 1.0e-3  # Target VMC optimization error (Ha)
ALAT = 0.30  # LRDMC lattice spacing
NUM_MCMC_PER_MEASUREMENT = 40  # LRDMC GFMC projections per measurement

# Explicit measurement-step counts for the benchmark
# (no target-error automatic convergence)
NUM_MCMC_STEPS_MCMC = 1000  # MCMC measurement steps per production
NUM_MCMC_STEPS_LRDMC = 1000  # LRDMC measurement steps per production

# Walker counts for vectorization benchmark
# WALKER_COUNTS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
WALKER_COUNTS = [8, 16, 32, 64]

# pySCF file names
TREXIO_FILE = "water_trexio.hdf5"
PYSCF_OUTPUT = "water.out"

# pySCF script (embedded from local_pyscf.py)
PYSCF_SCRIPT = '''\
from pyscf import gto, scf
from pyscf.tools import trexio

filename = "{trexio_file}"

mol = gto.Mole()
mol.verbose = 5
mol.atom = """
O 5.000000 7.147077 7.650971
H 4.068066 6.942975 7.563761
H 5.380237 6.896963 6.807984
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

mf = scf.KS(mol).density_fit()
mf.max_cycle = 200
mf.xc = "LDA_X,LDA_C_PZ"
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


def format_energy(energy: float | None, error: float | None) -> str:
    """Format energy as ``-X.XXXXX(YY)`` (error in last digits)."""
    if energy is None or error is None:
        return "N/A"
    if error == 0:
        return f"{energy:.5f}(0)"
    n_dec = max(0, -math.floor(math.log10(error)) + 1)
    err_in_last = round(error * 10**n_dec)
    return f"{energy:.{n_dec}f}({err_in_last})"


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
            wf_dump_freq=WF_DUMP_FREQ,
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
                num_mcmc_steps=NUM_MCMC_STEPS_MCMC,
                max_time=3000,
                poll_interval=300,
                max_continuation=1,
            ),
        )

        # LRDMC (GFMC_n: nmpm=40, a=0.30) with explicit step count
        # E_scf is taken from the MCMC energy to reduce warmup transients.
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
                E_scf=ValueFrom(label_mcmc, "energy"),
                num_gfmc_warmup_steps=25,
                num_gfmc_bin_blocks=10,
                num_gfmc_collect_steps=5,
                num_gfmc_projections=NUM_MCMC_STEPS_LRDMC,
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

        # MCMC net time (from output_parser)
        mcmc_dir = os.path.join(base_dir, f"03_mcmc/w{nw:05d}")
        mcmc_time = parse_mcmc_output(mcmc_dir).net_time_sec
        mcmc_t_str = f"{mcmc_time:.1f}" if mcmc_time is not None else "N/A"

        # LRDMC energy
        lrdmc_ctr = lrdmc_containers.get(nw)
        if lrdmc_ctr is not None and lrdmc_ctr.output_values:
            lrdmc_e = lrdmc_ctr.output_values.get("energy")
            lrdmc_err = lrdmc_ctr.output_values.get("energy_error")
        else:
            lrdmc_e, lrdmc_err = None, None
        lrdmc_e_str = format_energy(lrdmc_e, lrdmc_err)

        # LRDMC net time (from output_parser)
        lrdmc_dir = os.path.join(base_dir, f"04_lrdmc/w{nw:05d}")
        lrdmc_time = parse_lrdmc_output(lrdmc_dir).net_time_sec
        lrdmc_t_str = f"{lrdmc_time:.1f}" if lrdmc_time is not None else "N/A"

        row = f"| {nw:>8} | {mcmc_e_str:>17} | {mcmc_t_str:>14} | {lrdmc_e_str:>17} | {lrdmc_t_str:>15} |"
        print(row)

    print()


# ── Step 4: Plot throughput ───────────────────────────────────────
def plot_throughput(base_dir: str) -> None:
    """Plot normalized throughput vs number of walkers.

    (a) MCMC (VMC sampling): throughput = nw * NUM_MCMC_STEPS_MCMC / net_time
    (b) LRDMC: throughput = nw * NUM_MCMC_STEPS_LRDMC / net_time

    Both panels are normalized to the throughput at the smallest walker count.
    """
    # -- Collect net times --
    mcmc_nw, mcmc_tp = [], []
    lrdmc_nw, lrdmc_tp = [], []

    for nw in WALKER_COUNTS:
        mcmc_dir = os.path.join(base_dir, f"03_mcmc/w{nw:05d}")
        t_mcmc = parse_mcmc_output(mcmc_dir).net_time_sec
        if t_mcmc is not None and t_mcmc > 0:
            mcmc_nw.append(nw)
            mcmc_tp.append(nw * NUM_MCMC_STEPS_MCMC / t_mcmc)

        lrdmc_dir = os.path.join(base_dir, f"04_lrdmc/w{nw:05d}")
        t_lrdmc = parse_lrdmc_output(lrdmc_dir).net_time_sec
        if t_lrdmc is not None and t_lrdmc > 0:
            lrdmc_nw.append(nw)
            lrdmc_tp.append(nw * NUM_MCMC_STEPS_LRDMC / t_lrdmc)

    if not mcmc_nw and not lrdmc_nw:
        print("  [skip] No net-time data found; throughput plot not generated.")
        return

    mcmc_nw = np.array(mcmc_nw)
    mcmc_tp = np.array(mcmc_tp)
    lrdmc_nw = np.array(lrdmc_nw)
    lrdmc_tp = np.array(lrdmc_tp)

    # Normalize to first entry
    if len(mcmc_tp) > 0:
        mcmc_tp = mcmc_tp / mcmc_tp[0]
    if len(lrdmc_tp) > 0:
        lrdmc_tp = lrdmc_tp / lrdmc_tp[0]

    # -- Plot --
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.width"] = 1.0
    plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.linewidth"] = 1.5

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), facecolor="white", dpi=300)
    fig.subplots_adjust(wspace=0.15)

    panels = [
        (axes[0], "(a) MCMC", mcmc_nw, mcmc_tp),
        (axes[1], "(b) LRDMC", lrdmc_nw, lrdmc_tp),
    ]

    for idx, (ax, title, x, y) in enumerate(panels):
        ax.set_title(title)
        ax.set_xlabel("Number of walkers per GPU")
        if idx == 0:
            ax.set_ylabel("Normalized throughput")
        ax.set_xscale("log")
        ax.set_yscale("log")

        if len(x) > 0:
            ax.plot(x, y, color="r", marker="s", markersize=4, ls="-", alpha=0.9)

            ax.xaxis.set_major_locator(mticker.FixedLocator(x))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(int(v)) for v in x]))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.tick_params(axis="x", which="major", rotation=45, labelsize=10)
            ax.set_ylim([0.9, np.max(y) * 2.0])

    out_path = os.path.join(base_dir, "jqmc_vectorization_benchmark.jpg")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  Throughput plot saved to {out_path}")


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

    # 4) Throughput plot
    print()
    print("=" * 60)
    print("  Step 3: Plot vectorization throughput")
    print("=" * 60)
    plot_throughput(base_dir)
