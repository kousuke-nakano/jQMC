"""Job submission, monitoring, and fetching for jqmc-workflow.

jqmc is the only target binary.  The binary name ("jqmc") is hard-coded;
there is no binary_path, preoption, postoption, or input_redirect.
Job state is managed exclusively via workflow_state.toml.
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
import re
import shutil
import time
from datetime import datetime
from logging import getLogger

import toml

from ._config import get_config_dir, template_dir
from ._machine import Machine
from ._transfer import Data_transfer

logger = getLogger("jqmc-workflow").getChild(__name__)


def load_queue_data(server_machine_name: str, queue_label: str) -> dict:
    """Load a single queue section from ``queue_data.toml``.

    Parameters
    ----------
    server_machine_name : str
        Machine name (directory under ``~/.jqmc_setting/``).
    queue_label : str
        Section key in ``queue_data.toml``.

    Returns
    -------
    dict
        The TOML table for *queue_label*.
    """
    machine = Machine(server_machine_name)
    cfg = get_config_dir()
    path = os.path.join(cfg, machine.name, "queue_data.toml")
    with open(path) as f:
        data = toml.load(f)
    if queue_label not in data:
        raise KeyError(f"queue_label='{queue_label}' not found in {path}.")
    return data[queue_label]


def get_num_mpi(queue_data: dict) -> int:
    """Extract the number of MPI processes from a queue configuration.

    Tries ``num_cores`` first, then ``mpi_per_node × nodes``.
    Defaults to 1 if neither key is present.
    """
    if "num_cores" in queue_data:
        return int(queue_data["num_cores"])
    mpi_per_node = queue_data.get("mpi_per_node", 1)
    nodes = queue_data.get("nodes", 1)
    return int(mpi_per_node) * int(nodes)


class JobSubmission:
    """Submit, monitor, and fetch a single jqmc job on a remote (or local) machine."""

    stat_time_sleep = 60  # sec between job-check retries

    def __init__(
        self,
        server_machine_name: str,
        input_file: str = "input.toml",
        output_file: str = "out.o",
        queue_label: str = "default",
        jobname: str = "jqmc-wf",
        safe_mode: bool = False,
    ):
        self.server_machine = Machine(server_machine_name)
        self.data_transfer = Data_transfer(
            server_machine_name=server_machine_name,
        )

        # Bootstrap config directory if needed
        cfg = get_config_dir()
        if not os.path.isdir(cfg):
            logger.info(f"Bootstrapping config at {cfg}")
            shutil.copytree(template_dir, cfg)
            raise ValueError(f"Please configure {cfg} first.")

        # ── Queue settings ────────────────────────────────────────
        self.queue_label = queue_label
        queue_data_path = os.path.join(
            cfg,
            self.server_machine.name,
            "queue_data.toml",
        )
        try:
            with open(queue_data_path) as f:
                dict_toml = toml.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"{queue_data_path} not found!")

        try:
            self.queue_data = dict_toml[queue_label]
        except KeyError:
            raise KeyError(f"queue_label='{queue_label}' not found in {queue_data_path}.")

        # ── Job template ──────────────────────────────────────────
        self.job_submission_template = self.queue_data["submit_template"]

        # ── Job parameters ────────────────────────────────────────
        self.jobname = jobname
        self.input_file = input_file
        self.output_file = output_file
        self.safe_mode = safe_mode

        # ── Job state ─────────────────────────────────────────────
        self.max_job_submit = self.queue_data.get("max_job_submit", 1000)
        self.job_number = None
        self.job_running = False
        self.job_dir = None
        self.job_submit_date = None
        self.job_check_last_time = None
        self.job_fetch_date = None

    # ── Script generation ─────────────────────────────────────────

    @property
    def is_local_direct(self) -> bool:
        """True when the job should be launched directly (no submit script)."""
        return self.server_machine.machine_type == "local" and not self.server_machine.queuing

    def generate_script(self, submission_script: str = "submit.sh", *, work_dir=None):
        """Generate job submission script from template + queue_data.toml vars.

        Parameters
        ----------
        submission_script : str
            Filename of the generated script (basename).
        work_dir : str, optional
            Directory where the script is written.  When *None*,
            falls back to the current working directory.
        """
        cfg = get_config_dir()
        template_path = os.path.join(
            cfg,
            self.server_machine.name,
            self.job_submission_template,
        )
        with open(template_path, "r") as f:
            lines = f.readlines()

        def replace_kw(lines, keyword, value):
            """Replace _KEYWORD_ placeholders in script lines."""
            buf = [l for l in lines if re.match(f".*{keyword}.*", l)]
            for b in buf:
                idx = lines.index(b)
                lines[idx] = lines[idx].replace(keyword.replace("\\", ""), str(value))
            return lines

        # Standard replacements (jqmc-specific)
        lines = replace_kw(lines, "_INPUT_", self.input_file)
        lines = replace_kw(lines, "_OUTPUT_", self.output_file)
        lines = replace_kw(lines, "_JOBNAME_", self.jobname)

        # Queue-specific variables from queue_data.toml
        _SKIP_KEYS = {"submit_template", "max_job_submit"}
        for key, value in self.queue_data.items():
            if key not in _SKIP_KEYS:
                lines = replace_kw(lines, f"_{key.upper()}_", value)

        script_path = os.path.join(work_dir, submission_script) if work_dir else submission_script
        with open(script_path, "w") as f:
            f.writelines(lines)

    # ── Job submission ────────────────────────────────────────────

    def job_submit(self, submission_script: str = "submit.sh", from_objects=None, *, work_dir=None):
        """Submit the job.

        Parameters
        ----------
        submission_script : str
            Basename of the submit script in *work_dir*.
        from_objects : list[str], optional
            Basenames of extra files to upload.
        work_dir : str, optional
            Absolute path to the local job directory.  When *None*,
            falls back to ``os.getcwd()`` for backward compatibility.
        """
        from_objects = from_objects or []

        if not self.jobnum_check():
            logger.info("Max job limit reached; not submitting.")
            self.job_submit_date = None
            self.job_number = None
            self.job_running = False
            return False, self.job_number

        try:
            local_cwd = os.path.abspath(work_dir) if work_dir else os.path.abspath(os.getcwd())

            # ── Local direct mode (run generated submit script) ──
            if self.is_local_direct:
                command = f"bash {submission_script}"
                logger.info(f"  Running directly: {command}")
                self._run_local_direct(command, local_cwd)
                self.job_number = None
                self.job_running = False
                self.job_dir = local_cwd
                self.job_submit_date = datetime.today()
                logger.info("  Local direct job finished.")
                return True, self.job_number

            # ── Submit via queuing system or remote submit script ──
            command = f"{self.server_machine.jobsubmit} {submission_script}"

            if self.server_machine.machine_type == "local":
                server_dir = local_cwd
            else:
                local_root = self.data_transfer._local_root
                server_root = self.server_machine.workspace_root
                if local_root and local_root not in local_cwd:
                    raise ValueError(f"work_dir ({local_cwd}) not under local workspace_root ({local_root}).")
                server_dir = local_cwd.replace(local_root, server_root)
                self.data_transfer.put_objects(from_objects=from_objects, work_dir=local_cwd)

            if self.server_machine.queuing:
                stdout, stderr = self.server_machine.run_command(command=command, execute_dir=server_dir)
                if not stdout:
                    logger.error(f"Empty stdout! stderr={stderr}")
                self.job_number = stdout.strip().split()[self.server_machine.jobnum_index]
                self.job_running = True
            else:
                # Remote non-queuing: still use submit script
                self.server_machine.run_command(command=command, execute_dir=server_dir)
                self.job_number = None
                self.job_running = False

            self.job_dir = server_dir
            self.job_submit_date = datetime.today()
            logger.info(f"  Job submitted (number={self.job_number}).")

            self._close_ssh()
            return True, self.job_number

        except Exception as e:
            self.job_number = None
            self.job_running = False
            logger.error(f"Job submission failed: {e}")
            raise

    def _run_local_direct(self, command: str, work_dir: str):
        """Execute the jqmc command directly as a subprocess.

        Raises
        ------
        RuntimeError
            If the command exits with a non-zero return code.
        """
        import subprocess

        logger.debug(f"  workdir: {work_dir}")
        proc = subprocess.run(
            command,
            shell=True,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            # Log full output for diagnostics
            if proc.stdout.strip():
                logger.error(f"stdout: {proc.stdout.strip()[:2000]}")
            if proc.stderr.strip():
                logger.error(f"stderr: {proc.stderr.strip()[:2000]}")
            raise RuntimeError(f"jqmc exited with rc={proc.returncode}. Check {self.output_file} for details.")

    # ── Job checking ──────────────────────────────────────────────

    def jobcheck(self) -> bool:
        """Return True if the job is still running."""
        self.job_check_last_time = datetime.today()

        if not self.server_machine.queuing:
            self._close_ssh()
            return False

        trial_num = 10
        for jjj in range(trial_num):
            job_list = self.server_machine.get_job_list_as_text()
            if job_list != "":
                break
            logger.warning(f"jobcheck attempt {jjj} returned empty. Retrying...")
            time.sleep(self.stat_time_sleep)
        else:
            raise RuntimeError("jobcheck failed after all retries.")

        # PBS may return "1428989.opbs" from qsub but show only
        # "1428989" in qstat (or vice-versa).  Compare both the full
        # job_number and its numeric prefix (part before the first dot).
        job_id_full = str(self.job_number)
        job_id_short = job_id_full.split(".")[0]
        if any(job_id_full in line or job_id_short in line for line in job_list):
            logger.info(f"  Job {self.job_number} is running.")
            self.job_running = True
            flag = True
        else:
            logger.info(f"  Job {self.job_number} is done.")
            self.job_running = False
            flag = False

        self._close_ssh()
        return flag

    def jobnum_check(self) -> bool:
        """Check if we are below the max_job_submit limit."""
        if not self.server_machine.queuing:
            return True

        job_list = self.server_machine.get_job_list_as_text()
        queue_name = self.queue_data.get("queue", "")
        username = self.server_machine.username

        count = sum(
            1
            for line in job_list
            if (
                re.match(f".*\\s{username}\\s.*\\s{queue_name}\\s.*", line)
                or re.match(f".*\\s{queue_name}\\s.*\\s{username}\\s.*", line)
            )
        )
        logger.info(f"  {count} jobs running on {self.server_machine.name}")

        flag = count < self.max_job_submit
        self._close_ssh()
        return flag

    # ── Fetch results ─────────────────────────────────────────────

    def fetch_job(self, from_objects=None, exclude_patterns=None, *, work_dir=None):
        """Fetch job results from the remote machine.

        Parameters
        ----------
        from_objects : list[str], optional
            Basenames or glob patterns of files to download.
        exclude_patterns : list[str], optional
            Glob patterns to exclude.
        work_dir : str, optional
            Absolute path to the local job directory.  When *None*,
            falls back to ``os.getcwd()`` for backward compatibility.
        """
        from_objects = from_objects or []
        exclude_patterns = exclude_patterns or []

        if self.server_machine.machine_type != "local":
            self.data_transfer.get_objects(
                from_objects=from_objects,
                exclude_patterns=exclude_patterns,
                work_dir=work_dir,
            )

        self.job_fetch_date = datetime.today()
        self._close_ssh()

    # ── Delete a running job ──────────────────────────────────────

    def delete_job(self):
        self.server_machine.delete_job(jobid=self.job_number)
        self.job_running = False
        self._close_ssh()

    # ── Helper ────────────────────────────────────────────────────

    def _close_ssh(self):
        self.server_machine.ssh_close()
        self.data_transfer.ssh_close()
