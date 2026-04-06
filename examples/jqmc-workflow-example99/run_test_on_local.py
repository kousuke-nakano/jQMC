#!/usr/bin/env python
"""Local integration test for jqmc_workflow.

H₂ at 2 bond lengths (R = 0.74, 1.00 Å) — all-electron, cc-pVTZ, JSD. It should run **locally**.

Pipeline per R:
    pySCF (DFT) → WF (JSD: J1-exp + J2 + J3-ao-small) → VMC (15 opt steps) → MCMC
                                                                            → LRDMC_t (GFMC_t, a=0.30)
                                                                            → LRDMC_n (GFMC_n, a=0.30, survival_ratio=0.95)

After the pipeline completes, the script exercises the new Phase-1 APIs:

* ``get_all_workflow_statuses()``  — list every workflow_state.toml
* ``get_workflow_summary()``       — detailed summary per directory
* ``parse_vmc_output()``           — per-step VMC diagnostic data
* ``parse_mcmc_output()``          — MCMC diagnostic data
* ``parse_lrdmc_output()``         — LRDMC diagnostic data
* ``parse_input_params()``         — TOML parameter extraction
"""

import dataclasses
import os
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
    # Phase 1 APIs
    get_all_workflow_statuses,
    get_workflow_summary,
    parse_input_params,
    parse_lrdmc_output,
    parse_mcmc_output,
    parse_vmc_output,
)

# ── Configuration ─────────────────────────────────────────────────
SERVER = "localhost"
QUEUE_LABEL = "qM"

# Tiny parameters for fast local testing
NUM_OPT_STEPS = 15
Dt = 1.5
TARGET_VMC_ERROR = 0.005  # very loose
TARGET_MCMC_ERROR = 0.005  # very loose
TARGET_LRDMC_ERROR = 0.005  # very loose
ALAT = 0.30  # LRDMC lattice spacing (bohr)

R_VALUES = [0.74, 1.00]
R_VALUES = [0.74]

# pySCF script template
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
    return f"R_{R:.2f}"


def trexio_filename(R: float) -> str:
    return f"H2_R_{R:.2f}.h5"


# ── Step 0: pySCF (local) ────────────────────────────────────────
def run_pyscf(base_dir: str) -> None:
    """Run pySCF for each R value."""
    print("=" * 60)
    print("  Step 0: Run pySCF calculations (local)")
    print("=" * 60)

    for R in R_VALUES:
        pyscf_dir = os.path.join(base_dir, r_dir(R), "00_pyscf")
        os.makedirs(pyscf_dir, exist_ok=True)

        trexio_path = os.path.join(pyscf_dir, trexio_filename(R))
        script_path = os.path.join(pyscf_dir, "local_pyscf.py")

        with open(script_path, "w") as f:
            f.write(PYSCF_TEMPLATE.format(R=R))

        if os.path.isfile(trexio_path):
            print(f"  [skip] {r_dir(R)}/00_pyscf/{trexio_filename(R)}")
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
                sys.exit(1)
            print(f"  [done] {r_dir(R)}/00_pyscf/{trexio_filename(R)}")


# ── Step 1: Build pipeline ───────────────────────────────────────
def build_pipeline() -> list[Container]:
    """Build WF → VMC → {MCMC, LRDMC} for each R value."""
    workflows: list[Container] = []

    for R in R_VALUES:
        prefix = r_dir(R)
        trexio = trexio_filename(R)

        wf = Container(
            label=f"wf-{R:.2f}",
            dirname=os.path.join(prefix, "01_wf"),
            input_files=[os.path.join(prefix, "00_pyscf", trexio)],
            workflow=WF_Workflow(
                trexio_file=trexio,
                j1_parameter=1.0,  # exp Jastrow, initial value 1.0
                j2_parameter=1.0,
                j3_basis_type="ao-small",
            ),
        )

        vmc = Container(
            label=f"vmc-{R:.2f}",
            dirname=os.path.join(prefix, "02_vmc"),
            input_files=[FileFrom(f"wf-{R:.2f}", "hamiltonian_data.h5")],
            workflow=VMC_Workflow(
                server_machine_name=SERVER,
                queue_label=QUEUE_LABEL,
                jobname=f"vmc-H2-{R:.2f}",
                Dt=Dt,
                number_of_walkers=4,
                num_opt_steps=NUM_OPT_STEPS,
                pilot_mcmc_steps=200,
                pilot_vmc_steps=20,
                opt_J1_param=True,
                opt_J2_param=True,
                opt_J3_param=True,
                opt_lambda_param=False,
                opt_with_projected_MOs=False,
                target_error=TARGET_VMC_ERROR,  # no convergence check
                optimizer_kwargs={
                    "method": "sr",
                    "delta": 0.100,
                    "epsilon": 0.100,
                    "adaptive_learning_rate": True,
                },
                max_time=600,
                poll_interval=120,
                max_continuation=2,
                target_snr=None,
                energy_slope_sigma_threshold=None,
            ),
        )

        mcmc = Container(
            label=f"mcmc-{R:.2f}",
            dirname=os.path.join(prefix, "03_mcmc"),
            input_files=[
                FileFrom(f"vmc-{R:.2f}", f"hamiltonian_data_opt_step_{NUM_OPT_STEPS}.h5"),
            ],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=MCMC_Workflow(
                server_machine_name=SERVER,
                queue_label=QUEUE_LABEL,
                jobname=f"mcmc-H2-{R:.2f}",
                Dt=Dt,
                number_of_walkers=4,
                target_error=TARGET_MCMC_ERROR,
                atomic_force=False,
                num_mcmc_warmup_steps=10,
                num_mcmc_bin_blocks=10,
                pilot_steps=50,
                max_time=600,
                poll_interval=120,
                max_continuation=2,
            ),
        )

        # LRDMC (GFMC_t mode): time_projection_tau=0.10 (default)
        lrdmc_t = Container(
            label=f"lrdmc-t-{R:.2f}",
            dirname=os.path.join(prefix, "04_lrdmc_t"),
            input_files=[
                FileFrom(f"vmc-{R:.2f}", f"hamiltonian_data_opt_step_{NUM_OPT_STEPS}.h5"),
            ],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=LRDMC_Workflow(
                server_machine_name=SERVER,
                queue_label=QUEUE_LABEL,
                jobname=f"lrdmc-t-H2-{R:.2f}",
                alat=ALAT,
                number_of_walkers=4,
                target_error=TARGET_LRDMC_ERROR,
                atomic_force=False,
                num_gfmc_warmup_steps=25,
                num_gfmc_bin_blocks=10,
                num_gfmc_collect_steps=5,
                pilot_steps=50,
                max_time=600,
                poll_interval=120,
                max_continuation=2,
            ),
        )

        # LRDMC (GFMC_n mode): target_survived_walkers_ratio calibration
        lrdmc_n = Container(
            label=f"lrdmc-n-{R:.2f}",
            dirname=os.path.join(prefix, "05_lrdmc_n"),
            input_files=[
                FileFrom(f"vmc-{R:.2f}", f"hamiltonian_data_opt_step_{NUM_OPT_STEPS}.h5"),
            ],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=LRDMC_Workflow(
                server_machine_name=SERVER,
                queue_label=QUEUE_LABEL,
                jobname=f"lrdmc-n-H2-{R:.2f}",
                alat=ALAT,
                number_of_walkers=4,
                target_error=TARGET_LRDMC_ERROR,
                atomic_force=False,
                target_survived_walkers_ratio=0.95,  # GFMC_n mode
                num_gfmc_warmup_steps=25,
                num_gfmc_bin_blocks=10,
                num_gfmc_collect_steps=5,
                pilot_steps=50,
                max_time=600,
                poll_interval=120,
                max_continuation=2,
            ),
        )

        workflows.extend([wf, vmc, mcmc, lrdmc_t, lrdmc_n])

    return workflows


# ── Step 2: Exercise Phase-1 diagnostic APIs ─────────────────────
def run_diagnostics(base_dir: str) -> bool:
    """Run the new diagnostic / query APIs and verify results.

    Returns True if all checks pass.
    """
    print()
    print("=" * 60)
    print("  Step 2: Phase-1 API diagnostics")
    print("=" * 60)

    ok = True

    # ── get_all_workflow_statuses ──
    print()
    print("--- get_all_workflow_statuses ---")
    statuses = get_all_workflow_statuses(base_dir)
    for s in statuses:
        rel = os.path.relpath(s["directory"], base_dir)
        print(f"  {rel:<30s}  label={s['label']:<15s}  type={s['type']:<8s}  status={s['status']}")

    expected = len(R_VALUES) * 5  # wf + vmc + mcmc + lrdmc_t + lrdmc_n per R
    if len(statuses) < expected:
        print(f"  [WARN] Expected at least {expected} workflows, found {len(statuses)}")
        # Not a hard failure — WF doesn't always create state
    else:
        print(f"  [OK] Found {len(statuses)} workflow states")

    # ── get_workflow_summary per vmc/mcmc ──
    print()
    print("--- get_workflow_summary ---")
    for R in R_VALUES:
        for step in ["02_vmc", "03_mcmc", "04_lrdmc_t", "05_lrdmc_n"]:
            d = os.path.join(base_dir, r_dir(R), step)
            sm = get_workflow_summary(d)
            if sm:
                wf_info = sm.get("workflow", {})
                res = sm.get("result", {})
                print(
                    f"  {r_dir(R)}/{step}:  status={wf_info.get('status', '?')}, "
                    f"energy={res.get('energy', 'N/A')}, "
                    f"num_jobs={sm.get('num_jobs', 0)}"
                )
            else:
                print(f"  {r_dir(R)}/{step}:  [no state file]")

    # ── helper: dump all dataclass fields ──
    def _dump_dataclass(label, obj):
        """Print all fields of a dataclass, truncating stderr_tail."""
        d = dataclasses.asdict(obj)
        # Truncate stderr_tail for readability
        if "stderr_tail" in d and d["stderr_tail"]:
            d["stderr_tail"] = f"<{len(d['stderr_tail'])} chars>"
        # For VMC steps, show compact per-step summaries
        if "steps" in d and isinstance(d["steps"], list):
            steps = d.pop("steps")
            print(f"  {label}:  {len(steps)} steps, metadata={d}")
            for s in steps:
                print(f"    step {s['step']}: {s}")
        else:
            print(f"  {label}: {d}")

    # ── parse_vmc_output ──
    print()
    print("--- parse_vmc_output ---")
    for R in R_VALUES:
        vmc_dir = os.path.join(base_dir, r_dir(R), "02_vmc")
        vmc_data = parse_vmc_output(vmc_dir)
        _dump_dataclass(f"R={R:.2f}", vmc_data)
        if vmc_data.optimized_hamiltonian:
            print(f"    optimized_hamiltonian = {os.path.basename(vmc_data.optimized_hamiltonian)}")
        else:
            print("    [WARN] No optimized hamiltonian found")
        if len(vmc_data.steps) == 0:
            print(f"  [FAIL] No VMC steps parsed for R={R:.2f}")
            ok = False

    # ── parse_mcmc_output ──
    print()
    print("--- parse_mcmc_output ---")
    for R in R_VALUES:
        mcmc_dir = os.path.join(base_dir, r_dir(R), "03_mcmc")
        mcmc_data = parse_mcmc_output(mcmc_dir)
        _dump_dataclass(f"R={R:.2f}", mcmc_data)

    # ── parse_lrdmc_output (GFMC_t) ──
    print()
    print("--- parse_lrdmc_output (GFMC_t) ---")
    for R in R_VALUES:
        lrdmc_dir = os.path.join(base_dir, r_dir(R), "04_lrdmc_t")
        lrdmc_data = parse_lrdmc_output(lrdmc_dir)
        _dump_dataclass(f"R={R:.2f}", lrdmc_data)

    # ── parse_lrdmc_output (GFMC_n) ──
    print()
    print("--- parse_lrdmc_output (GFMC_n) ---")
    for R in R_VALUES:
        lrdmc_dir = os.path.join(base_dir, r_dir(R), "05_lrdmc_n")
        lrdmc_data = parse_lrdmc_output(lrdmc_dir)
        _dump_dataclass(f"R={R:.2f}", lrdmc_data)

    # ── parse_input_params ──
    print()
    print("--- parse_input_params ---")
    for R in R_VALUES:
        for step_name, step_dir in [("vmc", "02_vmc"), ("mcmc", "03_mcmc"), ("lrdmc_t", "04_lrdmc_t")]:
            d = os.path.join(base_dir, r_dir(R), step_dir)
            params = parse_input_params(d)
            print(f"  R={R:.2f}/{step_name}:  actual_opt_steps={params.actual_opt_steps}")
            for entry in params.per_input:
                jt = entry.get("job_type", "?")
                print(f"    [{entry.get('input_file', '?')}] -> {entry.get('output_file', '?')}")
                print(f"      [control]: {entry.get('control', {})}")
                if jt and jt in entry:
                    print(f"      [{jt}]: {entry[jt]}")

    return ok


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # 0) pySCF
    run_pyscf(base_dir)

    # 1) Pipeline: WF → VMC → {MCMC, LRDMC}
    print()
    print("=" * 60)
    print("  Step 1: WF → VMC → {MCMC, LRDMC} (local, minimal)")
    print("=" * 60)

    workflows = build_pipeline()
    launcher = Launcher(workflows=workflows, draw_graph=False)
    launcher.launch()

    # 2) Diagnostics
    all_ok = run_diagnostics(base_dir)

    print()
    print("=" * 60)
    if all_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — see above for details")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)
