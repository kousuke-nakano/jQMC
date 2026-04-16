"""Scientific phase definitions and transition rules.

Each QMC workflow session progresses through a sequence of scientific phases
(SCF → wavefunction build → VMC → MCMC → LRDMC → fit).  This module defines
the allowed phase transitions and the actions permitted in each phase/status
combination.

The transition graph and action lists are consumed by:
  - *jqmc-workflow* itself (guard-rail enforcement via :func:`require_action`)
  - *jqmc-mcp* (to expose ``allowed_actions`` as an MCP resource)
"""
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

from __future__ import annotations

from enum import Enum

from ._state import WorkflowStatus

# ---------------------------------------------------------------------------
# Scientific phases
# ---------------------------------------------------------------------------


class ScientificPhase(str, Enum):
    """Scientific phases of a QMC workflow session."""

    INIT = "init"
    SCF = "scf"
    WF_BUILD = "wf_build"
    VMC_PILOT = "vmc_pilot"
    VMC = "vmc"
    MCMC_PILOT = "mcmc_pilot"
    MCMC = "mcmc"
    LRDMC_PILOT = "lrdmc_pilot"
    LRDMC = "lrdmc"
    LRDMC_FIT = "lrdmc_fit"
    COMPLETED = "completed"


# ---------------------------------------------------------------------------
# Allowed phase transitions
# ---------------------------------------------------------------------------

PHASE_TRANSITIONS: dict[ScientificPhase, set[ScientificPhase]] = {
    ScientificPhase.INIT: {ScientificPhase.SCF, ScientificPhase.WF_BUILD},
    ScientificPhase.SCF: {ScientificPhase.WF_BUILD},
    ScientificPhase.WF_BUILD: {
        ScientificPhase.VMC_PILOT,
        ScientificPhase.VMC,
        ScientificPhase.MCMC_PILOT,
        ScientificPhase.MCMC,
    },
    ScientificPhase.VMC_PILOT: {ScientificPhase.VMC},
    ScientificPhase.VMC: {
        ScientificPhase.VMC_PILOT,
        ScientificPhase.VMC,
        ScientificPhase.MCMC_PILOT,
        ScientificPhase.MCMC,
        ScientificPhase.LRDMC_PILOT,
        ScientificPhase.LRDMC,
        ScientificPhase.COMPLETED,
    },
    ScientificPhase.MCMC_PILOT: {ScientificPhase.MCMC},
    ScientificPhase.MCMC: {
        ScientificPhase.VMC_PILOT,
        ScientificPhase.VMC,
        ScientificPhase.MCMC_PILOT,
        ScientificPhase.MCMC,
        ScientificPhase.LRDMC_PILOT,
        ScientificPhase.LRDMC,
        ScientificPhase.COMPLETED,
    },
    ScientificPhase.LRDMC_PILOT: {ScientificPhase.LRDMC},
    ScientificPhase.LRDMC: {
        ScientificPhase.VMC_PILOT,
        ScientificPhase.VMC,
        ScientificPhase.MCMC_PILOT,
        ScientificPhase.MCMC,
        ScientificPhase.LRDMC_PILOT,
        ScientificPhase.LRDMC,
        ScientificPhase.LRDMC_FIT,
        ScientificPhase.COMPLETED,
    },
    ScientificPhase.LRDMC_FIT: {ScientificPhase.COMPLETED},
    ScientificPhase.COMPLETED: set(),
}


# ---------------------------------------------------------------------------
# Allowed actions per phase
# ---------------------------------------------------------------------------

PHASE_ALLOWED_ACTIONS: dict[ScientificPhase, list[str]] = {
    ScientificPhase.INIT: [
        "configure_scf",
        "configure_wavefunction",
        "select_machine",
    ],
    ScientificPhase.SCF: [
        "run_scf",
        "recover_scf",
        "poll_job",
        "fetch_job_outputs",
        "cancel_job",
    ],
    ScientificPhase.WF_BUILD: [
        "configure_wavefunction",
        "build_wavefunction",
        "recover_wf",
        "poll_job",
        "fetch_job_outputs",
    ],
    ScientificPhase.VMC_PILOT: [
        "configure_vmc_pilot",
        "run_vmc_pilot",
        "recover_vmc_pilot",
        "poll_job",
        "fetch_job_outputs",
        "cancel_job",
    ],
    ScientificPhase.VMC: [
        "configure_vmc",
        "run_vmc",
        "recover_vmc",
        "poll_job",
        "fetch_job_outputs",
        "cancel_job",
    ],
    ScientificPhase.MCMC_PILOT: [
        "configure_mcmc_pilot",
        "run_mcmc_pilot",
        "recover_mcmc_pilot",
        "poll_job",
        "fetch_job_outputs",
        "cancel_job",
    ],
    ScientificPhase.MCMC: [
        "configure_mcmc",
        "run_mcmc",
        "recover_mcmc",
        "poll_job",
        "fetch_job_outputs",
        "cancel_job",
    ],
    ScientificPhase.LRDMC_PILOT: [
        "configure_lrdmc_pilot",
        "run_lrdmc_pilot",
        "recover_lrdmc_pilot",
        "poll_job",
        "fetch_job_outputs",
        "cancel_job",
    ],
    ScientificPhase.LRDMC: [
        "configure_lrdmc",
        "run_lrdmc_point",
        "recover_lrdmc_point",
        "poll_job",
        "fetch_job_outputs",
        "cancel_job",
    ],
    ScientificPhase.LRDMC_FIT: [
        "run_lrdmc_fit",
        "recover_lrdmc_fit",
    ],
    ScientificPhase.COMPLETED: [],
}

ALWAYS_ALLOWED_ACTIONS: list[str] = [
    "advance_phase",
    "rollback_phase",
    "close_session",
    "register_artifact",
    "mark_unhealthy",
]


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def can_advance(current: ScientificPhase, target: ScientificPhase) -> bool:
    """Return ``True`` if *target* is a valid successor of *current*."""
    return target in PHASE_TRANSITIONS.get(current, set())


def allowed_actions(
    phase: ScientificPhase,
    status: WorkflowStatus,
) -> list[str]:
    """Return the list of actions allowed for the given *phase* / *status*.

    When *status* is ``FAILED`` only ``recover_*`` and ``rollback_phase``
    actions are kept.  When *status* is ``RUNNING`` configuration actions
    are excluded.
    """
    phase_actions = list(PHASE_ALLOWED_ACTIONS.get(phase, []))
    if status == WorkflowStatus.FAILED:
        phase_actions = [a for a in phase_actions if a.startswith("recover_")]
        phase_actions.append("rollback_phase")
    if status == WorkflowStatus.RUNNING:
        phase_actions = [a for a in phase_actions if not a.startswith("configure_")]
    return phase_actions + ALWAYS_ALLOWED_ACTIONS


def require_action(
    action: str,
    phase: ScientificPhase,
    status: WorkflowStatus,
) -> None:
    """Raise :class:`ValueError` if *action* is not allowed in *phase*/*status*.

    Call this at the entry of every MCP-tool → workflow-method boundary to
    enforce the guard-rail.
    """
    allowed = allowed_actions(phase, status)
    if action not in allowed:
        raise ValueError(
            f"Action {action!r} is not allowed in phase={phase.value!r}, status={status.value!r}. Allowed: {allowed}"
        )


# ---------------------------------------------------------------------------
# Phase mutation
# ---------------------------------------------------------------------------


def advance_phase(
    current: ScientificPhase,
    target: ScientificPhase,
) -> ScientificPhase:
    """Validate and return *target* as the new phase.

    Raises :class:`ValueError` if the transition is not allowed.
    """
    if not can_advance(current, target):
        raise ValueError(
            f"Cannot advance from {current.value!r} to {target.value!r}. "
            f"Allowed: {sorted(p.value for p in PHASE_TRANSITIONS.get(current, set()))}"
        )
    return target


def rollback_phase(current: ScientificPhase) -> ScientificPhase:
    """Return the most recent predecessor of *current*.

    Raises :class:`ValueError` if *current* has no predecessor (i.e. INIT).
    """
    predecessors = [p for p, targets in PHASE_TRANSITIONS.items() if current in targets]
    if not predecessors:
        raise ValueError(f"Cannot rollback from {current.value!r}: no predecessor.")
    return predecessors[-1]
