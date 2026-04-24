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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import toml
from scipy.interpolate import CubicSpline

from jqmc_workflow import (
    Container,
    FileFrom,
    Launcher,
    LRDMC_Workflow,
    MCMC_Workflow,
    VMC_Workflow,
    WF_Workflow,
    parse_lrdmc_output,
    parse_mcmc_output,
)

# ── Configuration ─────────────────────────────────────────────────
SERVER = "cluster"
QUEUE_LABEL = "cores-120-mpi-120-omp-1-1h"

NUM_OPT_STEPS = 120  # VMC optimization steps
WF_DUMP_FREQ = 20  # WF dumping freq.
Dt = 1.2  # MCMC hopping distance
ALAT = 0.3  # LRDMC lattice spacing
TARGET_SNR = 6.0  # target signal_to_noise for convergence.
ENERGY_SLOPE_THRESHOLD = 2.0  # energy slope sigma threshold for convergence.
TARGET_VMC_ERROR = 5e-4  # Target statistical error (Ha)
TARGET_MCMC_ERROR = 5e-5  # Target statistical error (Ha)
TARGET_LRDMC_ERROR = 5e-5  # Target statistical error (Ha)

# Mixed precision: set to "mixed" to enable float32 for low-risk zones,
# or None (default) for all-float64. See doc/notes/mixed_precision.md.
PRECISION_MODE = None  # "mixed" or None

R_VALUES = [
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

mf = scf.RHF(mol)
mf.max_cycle = 200
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


def extract_max_force_norm(
    forces: list[dict] | None,
) -> tuple[float | None, float | None]:
    """Return (|F|_max, err) — max force norm over atoms.

    *forces* is ``[{"Fx": .., "Fx_err": .., "Fy": .., ...}, ...]``.
    """
    if not forces:
        return None, None
    best_norm, best_err = None, None
    for atom in forces:
        fx = atom.get("Fx", 0.0)
        fy = atom.get("Fy", 0.0)
        fz = atom.get("Fz", 0.0)
        fxe = atom.get("Fx_err", 0.0)
        fye = atom.get("Fy_err", 0.0)
        fze = atom.get("Fz_err", 0.0)
        norm = math.sqrt(fx**2 + fy**2 + fz**2)
        if norm > 0:
            err = math.sqrt((fx * fxe) ** 2 + (fy * fye) ** 2 + (fz * fze) ** 2) / norm
        else:
            err = math.sqrt(fxe**2 + fye**2 + fze**2)
        if best_norm is None or norm > best_norm:
            best_norm, best_err = norm, err
    return best_norm, best_err


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
        script_path = os.path.join(pyscf_dir, "local_pyscf.py")

        # Write the pySCF script
        with open(script_path, "w") as f:
            f.write(PYSCF_TEMPLATE.format(R=R))

        if os.path.isfile(trexio_path):
            print(f"  [skip] {r_dir(R)}/00_pyscf/{trexio_filename(R)} already exists.")
        else:
            print(f"  [run]  pySCF for R={R:.2f} ...")
            result = subprocess.run(
                [sys.executable, "local_pyscf.py"],
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
                j1_parameter=1.5,
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
                wf_dump_freq=WF_DUMP_FREQ,
                Dt=Dt,
                num_opt_steps=NUM_OPT_STEPS,
                pilot_mcmc_steps=50,
                pilot_vmc_steps=20,
                opt_J1_param=True,
                opt_J2_param=True,
                opt_J3_param=True,
                opt_lambda_param=True,
                opt_with_projected_MOs=True,
                target_error=TARGET_VMC_ERROR,
                target_snr=TARGET_SNR,
                energy_slope_sigma_threshold=ENERGY_SLOPE_THRESHOLD,
                optimizer_kwargs={"method": "sr", "delta": 0.300, "epsilon": 0.010, "use_lm": True},
                max_time=3000,
                poll_interval=120,
                max_continuation=1,
                precision_mode=PRECISION_MODE,
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
                Dt=Dt,
                target_error=TARGET_MCMC_ERROR,
                atomic_force=True,
                num_mcmc_warmup_steps=50,
                num_mcmc_bin_blocks=50,
                pilot_steps=200,
                max_time=3000,
                poll_interval=120,
                max_continuation=1,
                precision_mode=PRECISION_MODE,
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
                target_error=TARGET_LRDMC_ERROR,
                atomic_force=True,
                num_gfmc_warmup_steps=50,
                num_gfmc_bin_blocks=50,
                num_gfmc_collect_steps=20,
                pilot_steps=200,
                max_time=3000,
                poll_interval=120,
                max_continuation=1,
                precision_mode=PRECISION_MODE,
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 130)
    print("  H2 PES Summary  (JSD, MO opt, all-electron, cc-pVTZ)")
    print("=" * 130)
    print()

    header = (
        f"| {'R (Å)':>6} "
        f"| {'E_HF (Ha)':>13} "
        f"| {'E_MCMC (Ha)':>15} "
        f"| {'F_MCMC (Ha/Å)':>15} "
        f"| {'MCMC t_net':>10} "
        f"| {'E_LRDMC (Ha)':>15} "
        f"| {'F_LRDMC (Ha/Å)':>16} "
        f"| {'LRDMC t_net':>11} |"
    )
    separator = f"|{'-' * 8}|{'-' * 15}|{'-' * 17}|{'-' * 17}|{'-' * 12}|{'-' * 17}|{'-' * 18}|{'-' * 13}|"
    print(header)
    print(separator)

    for R in R_VALUES:
        # HF energy
        hf_e = hf_energies.get(R)
        hf_str = f"{hf_e:.6f}" if hf_e is not None else "N/A"

        # MCMC energy, force, and net time
        mcmc_ctr = mcmc_containers.get(R)
        if mcmc_ctr is not None and mcmc_ctr.output_values:
            mcmc_e = mcmc_ctr.output_values.get("energy")
            mcmc_err = mcmc_ctr.output_values.get("energy_error")
            mcmc_f, mcmc_ferr = extract_max_force_norm(mcmc_ctr.output_values.get("forces"))
        else:
            mcmc_e, mcmc_err, mcmc_f, mcmc_ferr = None, None, None, None
        mcmc_e_str = format_energy(mcmc_e, mcmc_err)
        mcmc_f_str = format_force(mcmc_f, mcmc_ferr)

        mcmc_dir = os.path.join(base_dir, r_dir(R), "03_mcmc")
        mcmc_t = parse_mcmc_output(mcmc_dir).net_time_sec
        mcmc_t_str = f"{mcmc_t:.1f}" if mcmc_t is not None else "N/A"

        # LRDMC energy, force, and net time
        lrdmc_ctr = lrdmc_containers.get(R)
        if lrdmc_ctr is not None and lrdmc_ctr.output_values:
            lrdmc_e = lrdmc_ctr.output_values.get("energy")
            lrdmc_err = lrdmc_ctr.output_values.get("energy_error")
            lrdmc_f, lrdmc_ferr = extract_max_force_norm(lrdmc_ctr.output_values.get("forces"))
        else:
            lrdmc_e, lrdmc_err, lrdmc_f, lrdmc_ferr = None, None, None, None
        lrdmc_e_str = format_energy(lrdmc_e, lrdmc_err)
        lrdmc_f_str = format_force(lrdmc_f, lrdmc_ferr)

        lrdmc_dir = os.path.join(base_dir, r_dir(R), "04_lrdmc")
        lrdmc_t = parse_lrdmc_output(lrdmc_dir).net_time_sec
        lrdmc_t_str = f"{lrdmc_t:.1f}" if lrdmc_t is not None else "N/A"

        row = (
            f"| {R:6.2f} | {hf_str:>13} "
            f"| {mcmc_e_str:>15} | {mcmc_f_str:>15} | {mcmc_t_str:>10} "
            f"| {lrdmc_e_str:>15} | {lrdmc_f_str:>16} | {lrdmc_t_str:>11} |"
        )
        print(row)

    print()


# ── Step 4: Plot PES ──────────────────────────────────────────────
BOHR_PER_ANG = 1.8897259886  # 1 Å = 1.8897 bohr

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.5


def _load_pes_data(base_dir: str, method_dir: str):
    """Load energy/error and Fz (atom 2) from workflow_state.toml."""
    energies, energy_errs = [], []
    forces_z, force_z_errs = [], []
    valid_R = []

    for R in R_VALUES:
        state_path = os.path.join(base_dir, f"R_{R:.2f}", method_dir, "workflow_state.toml")
        if not os.path.isfile(state_path):
            continue
        state = toml.load(state_path)
        res = state.get("result", {})
        e = res.get("energy")
        ee = res.get("energy_error")
        if e is None:
            continue

        valid_R.append(R)
        energies.append(e)
        energy_errs.append(ee if ee else 0.0)

        flist = res.get("forces", [])
        if len(flist) >= 2:
            fz = flist[1].get("Fz", None)
            fz_err = flist[1].get("Fz_err", None)
            if fz is not None:
                forces_z.append(fz * BOHR_PER_ANG)
                force_z_errs.append((fz_err if fz_err else 0.0) * BOHR_PER_ANG)
            else:
                forces_z.append(None)
                force_z_errs.append(None)
        else:
            forces_z.append(None)
            force_z_errs.append(None)

    return (
        np.array(valid_R),
        np.array(energies),
        np.array(energy_errs),
        forces_z,
        force_z_errs,
    )


def _plot_panel(ax, R, E, E_err, Fz, Fz_err, label):
    """Plot one panel (MCMC or LRDMC)."""
    E_min = np.min(E)
    E_rel = E - E_min

    cs = CubicSpline(R, E)
    R_fine = np.linspace(R.min(), R.max(), 300)
    E_fine_rel = cs(R_fine) - E_min
    dEdR_fine = cs(R_fine, 1)

    ax.plot(R_fine, E_fine_rel, "-", color="green", linewidth=1.5, label=label)
    ax.errorbar(R, E_rel, yerr=E_err, fmt="o", color="darkgreen", markersize=5, capsize=2, zorder=5)
    ax.set_xlabel("Bond Length (Angstrom)")
    ax.set_ylabel("Relative energy (Ha)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    R_eq_idx = np.argmin(cs(R_fine))
    R_eq = R_fine[R_eq_idx]
    ax.axvline(R_eq, color="green", linewidth=0.8, linestyle="--", alpha=0.7)

    ax2 = ax.twinx()
    ax2.plot(R_fine, dEdR_fine, "--", color="green", linewidth=1.2, alpha=0.7, label="dE/dR")

    R_f, neg_Fz, neg_Fz_err = [], [], []
    for i, fz in enumerate(Fz):
        if fz is not None:
            R_f.append(R[i])
            neg_Fz.append(-fz)
            neg_Fz_err.append(Fz_err[i] if Fz_err[i] is not None else 0.0)

    if R_f:
        ax2.errorbar(
            R_f,
            neg_Fz,
            yerr=neg_Fz_err,
            fmt="s",
            color="red",
            markersize=5,
            capsize=2,
            markerfacecolor="red",
            markeredgecolor="darkred",
            linewidth=0,
            elinewidth=1,
            label="$-F$",
            zorder=5,
        )

    ax2.set_ylabel("Force (Ha/Angstrom)")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    return ax2


def plot_pes(base_dir: str) -> str:
    """Generate H2 PES plot (MCMC + LRDMC) and return the PDF path."""
    R_mcmc, E_mcmc, Ee_mcmc, Fz_mcmc, Fze_mcmc = _load_pes_data(base_dir, "03_mcmc")
    R_lrdmc, E_lrdmc, Ee_lrdmc, Fz_lrdmc, Fze_lrdmc = _load_pes_data(base_dir, "04_lrdmc")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    _plot_panel(ax1, R_mcmc, E_mcmc, Ee_mcmc, Fz_mcmc, Fze_mcmc, "MCMC-JSD")
    _plot_panel(ax2, R_lrdmc, E_lrdmc, Ee_lrdmc, Fz_lrdmc, Fze_lrdmc, "LRDMC-JSD")

    fig.suptitle("H2 PES  (JSD, MO opt, all-electron, cc-pVTZ)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_pdf = os.path.join(base_dir, "H2_PES_mcmc_lrdmc.pdf")
    out_png = os.path.join(base_dir, "H2_PES_mcmc_lrdmc.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_pdf


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

    # 4) PES plot
    pdf_path = plot_pes(base_dir)
    print(f"PES plot saved: {pdf_path}")
