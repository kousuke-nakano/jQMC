"""Launcher: DAG-based parallel workflow executor for jqmc-workflow.

True DAG execution: as soon as ALL predecessors of a node complete,
that node starts immediately — no waiting for the entire "layer".
Supports FileFrom / ValueFrom dependencies.
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

import asyncio
import os
import pathlib
from datetime import datetime
from logging import (
    FileHandler,
    Formatter,
    StreamHandler,
    getLogger,
)
from typing import List, Optional

from .workflow import (
    Container,
    FileFrom,
    ValueFrom,
    Workflow,
    _is_dependency,
)

logger = getLogger("jqmc-workflow").getChild(__name__)
_loggers_initialized = {}


class Launcher:
    """DAG-based parallel workflow executor.

    Accepts a list of :class:`Container` objects, automatically
    infers the dependency graph from :class:`FileFrom` / :class:`ValueFrom`
    references, and executes workflows with *true DAG parallelism*: as soon
    as **all** predecessors of a node complete, that node starts immediately
    — there is no layer-based grouping.

    Parameters
    ----------
    workflows : list[Container]
        Workflows to execute.  Labels must be unique.
    log_level : str
        Logging level (``"DEBUG"`` or ``"INFO"``).
    log_name : str
        Log file name (appended, not overwritten).
    draw_graph : bool
        If ``True``, render the dependency graph to ``dependency_graph.png``
        (requires the ``graphviz`` Python package).

    Raises
    ------
    ValueError
        If workflow labels are duplicated or a dependency references an
        undefined workflow label.

    Examples
    --------
    Typical three-stage QMC pipeline::

        from jqmc_workflow import (
            Launcher, Container, FileFrom,
            WF_Workflow, VMC_Workflow, MCMC_Workflow,
        )

        wf = Container(
            label="wf",
            dirname="00_wf",
            input_files=["trexio.h5"],
            workflow=WF_Workflow(trexio_file="trexio.h5"),
        )

        vmc = Container(
            label="vmc-opt",
            dirname="01_vmc",
            input_files=[FileFrom("wf", "hamiltonian_data.h5")],
            workflow=VMC_Workflow(
                server_machine_name="cluster",
                num_opt_steps=10,
                target_error=0.001,
            ),
        )

        mcmc = Container(
            label="mcmc-run",
            dirname="02_mcmc",
            input_files=[
                FileFrom("vmc-opt", "hamiltonian_data_opt_step_9.h5")
            ],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=MCMC_Workflow(
                server_machine_name="cluster",
                target_error=0.001,
            ),
        )

        launcher = Launcher(
            workflows=[wf, vmc, mcmc],
            draw_graph=True,
        )
        launcher.launch()

    Notes
    -----
    * The launcher changes the working directory during execution and
      restores it afterwards.
    * If a workflow fails, all downstream dependents are automatically
      skipped.

    See Also
    --------
    Container : Wraps a workflow in a project directory.
    FileFrom : File dependency placeholder.
    ValueFrom : Value dependency placeholder.
    """

    def __init__(
        self,
        workflows: Optional[List[Container]] = None,
        log_level: str = "INFO",
        log_name: str = "jqmc_workflow.log",
        draw_graph: bool = False,
    ):
        workflows = workflows or []

        # ── Logger setup ──────────────────────────────────────────
        self._setup_logger(log_level, log_name)

        from ._header_footer import _print_header

        _print_header()

        # ── Resolve config dir early (CWD is still the user dir) ──
        from ._config import get_config_dir

        _cfg = get_config_dir()
        logger.debug(f"Config dir resolved to: {_cfg}")

        # ── Attributes ────────────────────────────────────────────
        self.root_dir = os.getcwd()

        self.workflows = workflows
        self.workflows_by_label = {cw.label: cw for cw in workflows}

        # Validate unique labels
        if len(self.workflows_by_label) != len(workflows):
            labels = [cw.label for cw in workflows]
            dupes = [l for l in labels if labels.count(l) > 1]
            raise ValueError(f"Duplicate workflow labels: {set(dupes)}")

        # Build dependency graph
        self.dependency_dict = self._build_dependency_graph()

        if draw_graph:
            self._draw_graph()

        # Log summary
        logger.info("")
        logger.info("-" * 50)
        logger.info("  DAG Pipeline")
        logger.info("-" * 50)
        logger.info(f"  Root dir  : {self.root_dir}")
        logger.info(f"  Workflows : {len(workflows)}")
        for label, deps in self.dependency_dict.items():
            dep_str = ", ".join(deps) if deps else "(none)"
            logger.info(f"    {label} <- {dep_str}")
        logger.info("-" * 50)
        logger.info("")

    # ── Logger setup ──────────────────────────────────────────────

    def _setup_logger(self, log_level: str, log_name: str):
        global _loggers_initialized
        name = "jqmc-workflow"
        if _loggers_initialized.get(name):
            return

        log = getLogger(name)
        log.setLevel(log_level)

        if log_level == "DEBUG":
            fmt = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
        else:
            fmt = Formatter("%(message)s")

        sh = StreamHandler()
        sh.setLevel(log_level)
        sh.setFormatter(fmt)
        log.addHandler(sh)

        fh = FileHandler(log_name, "a")
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        log.addHandler(fh)

        _loggers_initialized[name] = True

    # ── Dependency graph ──────────────────────────────────────────

    def _build_dependency_graph(self) -> dict:
        """Walk all workflow attributes to find dependency placeholders."""
        dep_dict = {}

        for cw in self.workflows:
            dep_labels = set()
            self._collect_deps(cw, dep_labels)
            dep_dict[cw.label] = tuple(dep_labels)

        # Validate — all dependency labels must exist
        all_labels = set(self.workflows_by_label.keys())
        for label, deps in dep_dict.items():
            missing = set(deps) - all_labels
            if missing:
                raise ValueError(f"Workflow '{label}' depends on undefined workflows: {missing}")

        return dep_dict

    def _collect_deps(self, obj, dep_labels: set):
        """Recursively scan obj's attributes for dependency placeholders."""
        if _is_dependency(obj):
            dep_labels.add(obj.label)
            return

        if isinstance(obj, (str, int, float, bool, type(None))):
            return

        if isinstance(obj, list):
            for item in obj:
                self._collect_deps(item, dep_labels)
            return

        if isinstance(obj, dict):
            for v in obj.values():
                self._collect_deps(v, dep_labels)
            return

        # Check all attributes of the object (Workflow, Container, etc.)
        if hasattr(obj, "__dict__"):
            for attr_val in obj.__dict__.values():
                self._collect_deps(attr_val, dep_labels)

    def _draw_graph(self):
        """Render the dependency graph using graphviz."""
        try:
            from graphviz import Digraph
        except ImportError:
            logger.warning("graphviz not installed; skipping graph rendering.")
            return

        G = Digraph(format="png")
        G.attr("node", shape="box", style="rounded")

        for label in self.dependency_dict:
            G.node(label)
        for label, deps in self.dependency_dict.items():
            for dep in deps:
                G.edge(dep, label)

        G.render("dependency_graph", cleanup=True)
        logger.info("Dependency graph saved to dependency_graph.png")

    # ── Variable resolution ───────────────────────────────────────

    def _get_value(self, dep_obj):
        """Resolve a FileFrom / ValueFrom to its actual value."""
        label = dep_obj.label
        cw = self.workflows_by_label[label]

        if isinstance(dep_obj, FileFrom):
            filepath = os.path.join(cw.dirname, dep_obj.filename)
            p = pathlib.Path(filepath)
            return p.resolve().relative_to(pathlib.Path(self.root_dir).resolve())

        elif isinstance(dep_obj, ValueFrom):
            return cw.output_values.get(dep_obj.key)

        else:
            raise ValueError(f"Unknown dependency type: {dep_obj}")

    def _resolve_variables(self, cw: Container):
        """Replace all dependency placeholders in cw with resolved values."""
        self._resolve_obj(cw)

    def _resolve_obj(self, obj):
        """Recursively resolve dependency placeholders in obj's attributes."""
        if not hasattr(obj, "__dict__"):
            return

        for key, value in list(obj.__dict__.items()):
            if _is_dependency(value):
                resolved = self._get_value(value)
                setattr(obj, key, resolved)

            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if _is_dependency(item):
                        new_list.append(self._get_value(item))
                    else:
                        new_list.append(item)
                setattr(obj, key, new_list)

            elif isinstance(value, Workflow):
                self._resolve_obj(value)

    # ── Execution: true DAG parallelism ───────────────────────────

    def launch(self):
        asyncio.run(self.async_launch())

    async def async_launch(self):
        """Execute all workflows respecting DAG dependencies.

        As soon as ALL predecessors of a node complete, that node
        starts immediately — no layer-based grouping.
        """
        completed = set()
        failed = set()
        pending = set(self.workflows_by_label.keys())

        logger.info("")
        logger.info("=" * 50)
        logger.info("  Starting DAG execution...")
        logger.info("=" * 50)
        logger.info("")

        running = {}  # label -> asyncio.Task

        while pending or running:
            # Find workflows whose dependencies are ALL completed
            ready = []
            for label in list(pending):
                deps = self.dependency_dict[label]
                # If any dep failed, this workflow cannot run
                if any(d in failed for d in deps):
                    logger.error(f"[{label}] Skipping -- dependency failed: {[d for d in deps if d in failed]}")
                    pending.discard(label)
                    failed.add(label)
                    continue
                # All deps done?
                if all(d in completed for d in deps):
                    ready.append(label)

            # Launch ready workflows in parallel
            for label in ready:
                pending.discard(label)
                cw = self.workflows_by_label[label]
                self._resolve_variables(cw)
                logger.info("-" * 50)
                logger.info(f"  [{label}] Launching...")
                logger.info("-" * 50)
                task = asyncio.create_task(self._run_workflow(label, cw))
                running[label] = task

            if not running:
                if pending:
                    logger.error(f"Deadlock! Remaining: {pending}")
                    break
                else:
                    break

            # Wait for at least one task to complete
            done_tasks, _ = await asyncio.wait(running.values(), return_when=asyncio.FIRST_COMPLETED)

            for task in done_tasks:
                # Find which label this task corresponds to
                label = None
                for lbl, t in list(running.items()):
                    if t is task:
                        label = lbl
                        break
                if label is None:
                    continue

                del running[label]

                exc = task.exception()
                if exc:
                    logger.error(f"[{label}] FAILED: {exc}")
                    failed.add(label)
                else:
                    cw = self.workflows_by_label[label]
                    if getattr(cw, "status", None) == "failed":
                        logger.error(f"[{label}] FAILED (status=failed)")
                        failed.add(label)
                    else:
                        logger.info(f"[{label}] Completed.")
                        completed.add(label)

        # Summary
        logger.info("")
        logger.info("=" * 50)
        logger.info("  DAG execution summary")
        logger.info("-" * 50)
        logger.info(f"  Completed : {len(completed)}")
        logger.info(f"  Failed    : {len(failed)}")
        logger.info(f"  Skipped   : {len(pending)}")
        logger.info("=" * 50)

        from ._header_footer import _print_footer

        _print_footer()

    async def _run_workflow(self, label: str, cw: Container):
        """Run a single encapsulated workflow."""
        try:
            await cw.async_launch()
        except Exception:
            logger.exception(f"[{label}] Exception during execution")
            raise
