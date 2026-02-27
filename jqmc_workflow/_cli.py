"""CLI tool for monitoring / managing jqmc-workflow jobs.

Usage::

    jqmc-jobmanager show [--id N]
    jqmc-jobmanager check [--id N] [--server SERVER]
    jqmc-jobmanager del --id N [--server SERVER]

Discovers ``workflow_state.toml`` files in the directory tree and presents
a tree view with status information.
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

import os
import shutil
from datetime import datetime
from logging import Formatter, StreamHandler, getLogger

import toml
import typer
import yaml

from ._config import get_config_dir, template_dir

logger = getLogger("jqmc-workflow").getChild(__name__)

from importlib.metadata import version as _pkg_version

try:
    jqmc_workflow_version = _pkg_version("jqmc")
except Exception:
    jqmc_workflow_version = "unknown"


# ---------------------------------------------------------------------------
# Monitor class (unchanged logic)
# ---------------------------------------------------------------------------


class Monitor:
    """Walk directory tree and collect job information from workflow_state.toml."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.job_counter = 0
        self.entries = []  # list of dicts with path, state info

    def discover(self):
        """Walk tree and collect entries from workflow_state.toml."""
        self.entries = []
        self.job_counter = 0
        self._walk(self.root_dir)

    def _walk(self, path):
        state_file = os.path.join(path, "workflow_state.toml")
        if os.path.isfile(state_file):
            try:
                data = toml.load(state_file)
                # Get the latest job from [[jobs]] list (or legacy [job])
                jobs_list = data.get("jobs", [])
                if not jobs_list and "job" in data:
                    # Legacy single [job] migration
                    jobs_list = [data["job"]] if data["job"] else []
                latest_job = jobs_list[-1] if jobs_list else {}

                entry = {
                    "id": self.job_counter,
                    "dir": path,
                    "label": data.get("workflow", {}).get("label", "?"),
                    "type": data.get("workflow", {}).get("type", "?"),
                    "status": data.get("workflow", {}).get("status", "?"),
                    "job_id": latest_job.get("job_id", None),
                    "server": latest_job.get("server_machine", "?"),
                    "updated": data.get("workflow", {}).get("updated_at", "?"),
                    "error": data.get("workflow", {}).get("error", None),
                    "result": data.get("result", {}),
                    "estimation": data.get("estimation", {}),
                    "jobs": jobs_list,
                }
                self.entries.append(entry)
                self.job_counter += 1
            except Exception as e:
                logger.warning(f"Failed to read {state_file}: {e}")

        # Recurse into subdirs
        try:
            subdirs = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
        except PermissionError:
            return
        for d in subdirs:
            self._walk(os.path.join(path, d))

    def show_tree(self):
        """Print a summary table of all discovered workflows."""
        if not self.entries:
            self.discover()

        logger.info("")
        logger.info("=" * 78)
        logger.info("  Workflow Job Tree")
        logger.info(f"  Root: {self.root_dir}")
        logger.info("=" * 78)

        if not self.entries:
            logger.info("  (no workflows found)")
            return

        header = f"  {'ID':>3}  {'Status':<12} {'Label':<20} {'Type':<20} {'Server':<12} {'Job#':<8}"
        logger.info(header)
        logger.info("  " + "-" * 74)
        for e in self.entries:
            rel = os.path.relpath(e["dir"], self.root_dir)
            job_id_str = str(e.get("job_id") or "-")
            logger.info(f"  {e['id']:>3}  {e['status']:<12} {e['label']:<20} {e['type']:<20} {e['server']:<12} {job_id_str:<8}")
            logger.info(f"       dir: {rel}")

            # Show error if any
            if e.get("error"):
                logger.info(f"       error: {e['error']}")

            # Show key result values inline
            result = e.get("result", {})
            if result.get("energy") is not None:
                energy = result["energy"]
                err = result.get("energy_error")
                if err is not None:
                    logger.info(f"       energy: {energy} +- {err} Ha")
                else:
                    logger.info(f"       energy: {energy} Ha")

        logger.info("")

    def show_detail(self, job_id: int):
        """Print full details for a single workflow, including state and file listing."""
        if not self.entries:
            self.discover()

        matches = [e for e in self.entries if e["id"] == job_id]
        if not matches:
            logger.error(f"Job ID {job_id} not found.")
            return
        e = matches[0]

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Detail for JOB-ID {job_id}")
        logger.info("=" * 60)
        logger.info(f"  label   : {e['label']}")
        logger.info(f"  type    : {e['type']}")
        logger.info(f"  status  : {e['status']}")
        logger.info(f"  server  : {e['server']}")
        logger.info(f"  job_id  : {e.get('job_id') or '-'}")
        logger.info(f"  updated : {e.get('updated', '-')}")
        rel = os.path.relpath(e["dir"], self.root_dir)
        logger.info(f"  dir     : {rel}")

        if e.get("error"):
            logger.info(f"  error   : {e['error']}")

        # Show job history
        jobs_list = e.get("jobs", [])
        if jobs_list:
            logger.info("")
            logger.info("  [jobs]")
            logger.info(f"    {'#':<3} {'input_file':<20} {'job_id':<12} {'status':<10} {'server':<12}")
            logger.info(
                f"    {'---':<3} {'--------------------':<20} {'------------':<12} {'----------':<10} {'------------':<12}"
            )
            for idx, j in enumerate(jobs_list):
                logger.info(
                    f"    {idx:<3} {j.get('input_file', '?'):<20} "
                    f"{j.get('job_id', '-'):<12} {j.get('status', '?'):<10} "
                    f"{j.get('server_machine', '?'):<12}"
                )

        # Show result section
        result = e.get("result", {})
        if result:
            logger.info("")
            logger.info("  [result]")
            for k, v in result.items():
                logger.info(f"    {k} = {v}")

        # Show estimation section
        estimation = e.get("estimation", {})
        if estimation:
            logger.info("")
            logger.info("  [estimation]")
            for k, v in estimation.items():
                logger.info(f"    {k} = {v}")

        # Show files in the directory
        logger.info("")
        logger.info("  Files in directory:")
        work_dir = e["dir"]
        try:
            files = sorted(os.listdir(work_dir))
            for f in files:
                fpath = os.path.join(work_dir, f)
                if os.path.isdir(fpath):
                    logger.info(f"    {f}/")
                else:
                    size = os.path.getsize(fpath)
                    logger.info(f"    {f}  ({_human_size(size)})")
        except PermissionError:
            logger.info("    (permission denied)")

        # Show full toml content
        state_file = os.path.join(e["dir"], "workflow_state.toml")
        if os.path.isfile(state_file):
            logger.info("")
            logger.info("  workflow_state.toml:")
            logger.info("  " + "-" * 40)
            with open(state_file, "r") as f:
                for line in f:
                    logger.info(f"    {line.rstrip()}")
            logger.info("  " + "-" * 40)

        logger.info("")

    def check_job(self, job_id: int, server_machine_name: str):
        """Check live job status on the remote/local machine."""
        if not self.entries:
            self.discover()

        matches = [e for e in self.entries if e["id"] == job_id]
        if not matches:
            logger.error(f"Job ID {job_id} not found.")
            return

        e = matches[0]
        stored_job_id = e.get("job_id")
        server = e.get("server", server_machine_name)

        if not stored_job_id or stored_job_id in ("-", "local", "None"):
            logger.info(f"Job ID {job_id} ({e['label']}): no queued job_id (local direct execution).")
            logger.info(f"  Current status: {e['status']}")
            return

        logger.info(f"Checking job {stored_job_id} on {server} (workflow: {e['label']})...")

        try:
            from ._machine import Machine

            machine = Machine(server)
            job_list_text = machine.get_job_list_as_text()

            found = any(stored_job_id in line for line in job_list_text)
            if found:
                logger.info(f"  Job {stored_job_id} is RUNNING on {server}.")
            else:
                logger.info(f"  Job {stored_job_id} is NOT in the queue on {server}.")
                logger.info(f"  (it may have finished or been cancelled)")
            machine.ssh_close()
        except Exception as ex:
            logger.error(f"  Failed to check job on {server}: {ex}")

    def delete_job(self, job_id: int, server_machine_name: str):
        """Cancel/delete a queued job on the remote/local machine."""
        if not self.entries:
            self.discover()

        matches = [e for e in self.entries if e["id"] == job_id]
        if not matches:
            logger.error(f"Job ID {job_id} not found.")
            return

        e = matches[0]
        stored_job_id = e.get("job_id")
        server = e.get("server", server_machine_name)

        if not stored_job_id or stored_job_id in ("-", "local", "None"):
            logger.info(f"Job ID {job_id} ({e['label']}): no queued job_id -- marking as cancelled.")
        else:
            logger.info(f"Deleting job {stored_job_id} on {server} (workflow: {e['label']})...")
            try:
                from ._machine import Machine

                machine = Machine(server)
                result = machine.delete_job(stored_job_id)
                logger.info(f"  Queue response: {' '.join(result)}")
                machine.ssh_close()
            except Exception as ex:
                logger.error(f"  Failed to delete job on {server}: {ex}")

        # Update workflow_state.toml
        state_file = os.path.join(e["dir"], "workflow_state.toml")
        if os.path.isfile(state_file):
            data = toml.load(state_file)
            data.setdefault("workflow", {})["status"] = "cancelled"
            data["workflow"]["updated_at"] = datetime.now().isoformat()
            with open(state_file, "w") as f:
                toml.dump(data, f)

        logger.info(f"  Status set to 'cancelled' for JOB-ID {job_id}.")


def _human_size(nbytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


# ---------------------------------------------------------------------------
# Logging setup helper
# ---------------------------------------------------------------------------


def _setup_logger(log_level: str = "INFO") -> None:
    """Configure the jqmc-workflow root logger."""
    log = getLogger("jqmc-workflow")
    log.setLevel(log_level)
    sh = StreamHandler()
    sh.setLevel(log_level)
    if log_level == "DEBUG":
        fmt = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    else:
        fmt = Formatter("%(message)s")
    sh.setFormatter(fmt)
    log.addHandler(sh)


def _bootstrap_config() -> str:
    """Ensure config directory exists; return machine_data.yaml path."""
    cfg = get_config_dir()
    machine_info_yaml = os.path.join(cfg, "machine_data.yaml")
    if not os.path.isfile(machine_info_yaml):
        if not os.path.isdir(cfg):
            typer.echo(f"First run: creating {cfg} from template.")
            shutil.copytree(template_dir, cfg)
            typer.echo(f"Please edit {machine_info_yaml}")
            raise typer.Exit()
    return machine_info_yaml


# ---------------------------------------------------------------------------
# CLI: typer app (direct entry point, no click group wrapper)
# ---------------------------------------------------------------------------

job_app = typer.Typer(help="The jQMC workflow job manager.")


@job_app.command("show")
def show(
    job_id: int = typer.Option(-1, "--id", help="Job ID to show details for. Omit for tree only."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level (DEBUG or INFO)."),
):
    """Show workflow job tree (and optionally full detail for one job)."""
    _setup_logger(log_level)
    _bootstrap_config()

    from ._header_footer import _print_footer, _print_header

    _print_header()

    monitor = Monitor(root_dir=os.getcwd())
    monitor.discover()
    monitor.show_tree()

    if job_id >= 0:
        monitor.show_detail(job_id=job_id)

    _print_footer()


@job_app.command("check")
def check(
    job_id: int = typer.Option(-1, "--id", help="Job ID to check live status for."),
    server: str = typer.Option("localhost", "-s", "--server", help="Server machine name."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level (DEBUG or INFO)."),
):
    """Check live job status on the remote/local machine."""
    _setup_logger(log_level)
    _bootstrap_config()

    from ._header_footer import _print_footer, _print_header

    _print_header()

    monitor = Monitor(root_dir=os.getcwd())
    monitor.discover()
    monitor.show_tree()

    if job_id >= 0:
        monitor.check_job(job_id=job_id, server_machine_name=server)
    else:
        typer.echo("Tip: use --id N to check live status of a specific job.")

    _print_footer()


@job_app.command("del")
def delete(
    job_id: int = typer.Option(..., "--id", help="Job ID to cancel/delete (required)."),
    server: str = typer.Option("localhost", "-s", "--server", help="Server machine name."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level (DEBUG or INFO)."),
):
    """Cancel/delete a queued job on the remote/local machine."""
    _setup_logger(log_level)
    _bootstrap_config()

    from ._header_footer import _print_footer, _print_header

    _print_header()

    monitor = Monitor(root_dir=os.getcwd())
    monitor.discover()
    monitor.delete_job(job_id=job_id, server_machine_name=server)

    _print_footer()


def job_manager_cli():
    """Entry point for ``jqmc-jobmanager`` console script."""
    job_app()


if __name__ == "__main__":
    job_manager_cli()
