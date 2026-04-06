"""jqmc_workflow — Automated workflow manager for jQMC calculations.

Public API
----------
Workflow classes:
    :class:`WF_Workflow`        TREXIO → hamiltonian_data.h5 conversion.
    :class:`VMC_Workflow`       Jastrow / orbital optimisation (job_type=vmc).
    :class:`MCMC_Workflow`      VMC production sampling (job_type=mcmc).
    :class:`LRDMC_Workflow`     Lattice-Regularized DMC (job_type=lrdmc-bra / lrdmc-tau).
    :class:`LRDMC_Ext_Workflow` Multi-alat LRDMC a²→0 extrapolation.

Composition helpers:
    :class:`Workflow`              Abstract base for custom workflows.
    :class:`Container`  Wraps a workflow in a project directory.
    :class:`FileFrom`              Declare a file dependency on another workflow.
    :class:`ValueFrom`             Declare a value dependency on another workflow.
    :class:`Launcher`              DAG-based parallel workflow executor.
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

from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("jqmc")
except Exception:
    __version__ = "unknown"

from ._machine import list_machines, probe_environment
from ._output_parser import (
    parse_input_params,
    parse_lrdmc_ext_output,
    parse_lrdmc_output,
    parse_mcmc_output,
    parse_vmc_output,
    repair_forces_from_output,
)
from ._phase import (
    PHASE_ALLOWED_ACTIONS,
    PHASE_TRANSITIONS,
    ScientificPhase,
    advance_phase,
    allowed_actions,
    can_advance,
    rollback_phase,
)
from ._results import (
    Input_Parameters,
    LRDMC_Diagnostic_Data,
    LRDMC_Ext_Diagnostic_Data,
    MCMC_Diagnostic_Data,
    VMC_Diagnostic_Data,
    VMC_Step_Data,
)
from ._state import (
    JobStatus,
    WorkflowStatus,
    get_all_workflow_statuses,
    get_artifact_lineage,
    get_artifact_registry,
    get_workflow_summary,
    register_artifact,
    save_job_accounting,
    set_error,
)
from .launcher import Launcher
from .lrdmc_ext_workflow import LRDMC_Ext_Workflow
from .lrdmc_workflow import LRDMC_Workflow
from .mcmc_workflow import MCMC_Workflow
from .vmc_workflow import VMC_Workflow
from .wf_workflow import WF_Workflow
from .workflow import (
    Container,
    FileFrom,
    ValueFrom,
    Workflow,
)

__all__ = [
    "Workflow",
    "Container",
    "FileFrom",
    "ValueFrom",
    "Launcher",
    "WF_Workflow",
    "VMC_Workflow",
    "MCMC_Workflow",
    "LRDMC_Workflow",
    "LRDMC_Ext_Workflow",
    # Result types
    "VMC_Step_Data",
    "VMC_Diagnostic_Data",
    "MCMC_Diagnostic_Data",
    "LRDMC_Diagnostic_Data",
    "LRDMC_Ext_Diagnostic_Data",
    "Input_Parameters",
    # Parsers
    "parse_vmc_output",
    "parse_mcmc_output",
    "parse_lrdmc_output",
    "parse_lrdmc_ext_output",
    "parse_input_params",
    "repair_forces_from_output",
    # State queries
    "get_all_workflow_statuses",
    "get_workflow_summary",
    "set_error",
    "save_job_accounting",
    "register_artifact",
    "get_artifact_lineage",
    "get_artifact_registry",
    # Enums
    "WorkflowStatus",
    "JobStatus",
    # Phase management
    "ScientificPhase",
    "PHASE_TRANSITIONS",
    "PHASE_ALLOWED_ACTIONS",
    "can_advance",
    "allowed_actions",
    "advance_phase",
    "rollback_phase",
    # Machine catalog
    "list_machines",
    "probe_environment",
]
