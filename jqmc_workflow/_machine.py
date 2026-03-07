"""Machine abstraction (local / remote via SSH+paramiko).

Provides the Machine class for command execution and filesystem queries,
and the Machines_handler class for SFTP-based file transfer.
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
import pathlib
import random
import re
import shutil
import stat
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from subprocess import PIPE

import paramiko
import yaml

from ._config import get_config_dir, template_dir

logger = getLogger("jqmc-workflow").getChild(__name__)


class Machine:
    """Represents a local or remote machine."""

    ssh_retry_time = 120
    ssh_retry_max_num = 10

    def __init__(self, machine: str):
        cfg = get_config_dir()
        self.machine_info_yaml = os.path.join(cfg, "machine_data.yaml")

        # Bootstrap config dir on first run
        if not os.path.isdir(cfg):
            logger.info(f"{cfg} not found. Bootstrapping from template (first run).")
            shutil.copytree(template_dir, cfg)
            logger.info(f"{cfg} has been generated.")
            logger.info(f"Please edit {self.machine_info_yaml}")
            raise FileNotFoundError(f"Please configure {self.machine_info_yaml} before running.")

        # Load machine data
        try:
            with open(self.machine_info_yaml, "r") as yf:
                self.data = yaml.safe_load(yf)[machine]
        except FileNotFoundError:
            logger.error(f"Config file {self.machine_info_yaml} not found!")
            raise
        except KeyError:
            logger.error(f"machine='{machine}' is not defined in {self.machine_info_yaml}. Please add it.")
            raise

        self.__name = machine
        self.ssh_status = False

    # ── SSH management ──────────────────────────────────────────────

    def ssh_open(self):
        if self.machine_type != "remote":
            return
        if self.ssh_status:
            logger.debug(f"SSH already open (id={id(self.ssh)})")
            return

        rw = random.randint(3, 6)
        logger.info(f"  Wait {rw}s before opening SSH to {self.name}")
        time.sleep(rw)

        ssh_config = paramiko.SSHConfig()
        config_file = os.path.join(os.getenv("HOME"), ".ssh/config")
        try:
            ssh_config.parse(open(config_file, "r"))
        except FileNotFoundError:
            logger.error(f"SSH config file ({config_file}) is required.")
            raise

        # Workaround: paramiko raises KeyError('canonicaldomains') when
        # CanonicalizeHostname is set but CanonicalDomains is missing.
        try:
            lkup = ssh_config.lookup(self.__name)
        except KeyError:
            from io import StringIO

            with open(config_file, "r") as fh:
                original = fh.read()
            # Disable CanonicalizeHostname to avoid the paramiko bug
            patched = re.sub(
                r"(?im)^(\s*CanonicalizeHostname\s+)yes",
                r"\1no",
                original,
            )
            ssh_config = paramiko.SSHConfig()
            ssh_config.parse(StringIO(patched))
            lkup = ssh_config.lookup(self.__name)

        hostname = lkup["hostname"]
        username = lkup["user"]
        key_filename = lkup["identityfile"]

        proxy_command = lkup.get("proxycommand")
        proxy_flag = proxy_command is not None

        self.username = username
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()

        for tt in range(self.ssh_retry_max_num):
            try:
                kwargs = dict(
                    hostname=hostname,
                    username=username,
                    key_filename=key_filename,
                )
                if proxy_flag:
                    kwargs["sock"] = paramiko.ProxyCommand(proxy_command)
                self.ssh.connect(**kwargs)
                logger.info(f"  SSH connected (attempt {tt + 1})")
                break
            except paramiko.ssh_exception.SSHException:
                logger.warning(f"SSH connect failed (attempt {tt + 1}). Retrying in {self.ssh_retry_time}s.")
                time.sleep(self.ssh_retry_time)
                if tt == self.ssh_retry_max_num - 1:
                    logger.error("SSH connect failed after all retries.")
                    raise

        self.sftp = self.ssh.open_sftp()
        self.ssh_status = True

    def ssh_close(self):
        if self.machine_type != "remote" or not self.ssh_status:
            return
        max_retries = 3
        timeout_sec = 5.0

        for obj_name, obj in [("ssh", self.ssh), ("sftp", self.sftp)]:
            for attempt in range(1, max_retries + 1):
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(obj.close)
                try:
                    future.result(timeout=timeout_sec)
                    logger.debug(f"{obj_name}.close() ok (attempt {attempt})")
                    break
                except Exception as e:
                    logger.warning(f"{obj_name}.close() attempt {attempt}: {e.__class__.__name__}: {e}")
                    if attempt == max_retries:
                        logger.error(f"{obj_name}.close() failed after {max_retries} attempts")
                        raise ValueError(f"Cannot close {obj_name}")
                    time.sleep(1)
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

        del self.ssh
        del self.sftp
        self.ssh_status = False

    # ── Properties (read from machine_data.yaml) ──────────────────

    def _get(self, key, default=None):
        try:
            return self.data[key]
        except KeyError:
            if default is not None:
                return default
            raise KeyError(f"'{key}' not found for machine '{self.name}'")

    @property
    def name(self):
        return self.__name

    @property
    def machine_type(self) -> str:
        val = self._get("machine_type")
        if val not in {"local", "remote"}:
            raise ValueError(f"machine_type must be 'local' or 'remote', got '{val}'")
        return val

    @property
    def username(self):
        try:
            return self._username
        except AttributeError:
            return os.getlogin()

    @username.setter
    def username(self, value):
        self._username = value

    @property
    def ip(self):
        return self._get("ip", default=None)

    @property
    def workspace_root(self):
        return self._get("workspace_root", default=None)

    @property
    def queuing(self) -> bool:
        return self._get("queuing")

    @property
    def jobsubmit(self) -> str:
        return self._get("jobsubmit")

    @property
    def jobcheck(self) -> str:
        return self._get("jobcheck")

    @property
    def jobdel(self) -> str:
        return self._get("jobdel")

    @property
    def jobnum_index(self) -> int:
        return self._get("jobnum_index")

    # ── Command execution ─────────────────────────────────────────

    def run_command(self, command: str, execute_dir: str = None):
        if execute_dir:
            if self.machine_type == "remote":
                self.ssh_open()
                fileattr = self.sftp.lstat(execute_dir)
                if not stat.S_ISDIR(fileattr.st_mode):
                    raise FileNotFoundError(f"{execute_dir} not found on remote machine.")
            else:
                if not os.path.isdir(execute_dir):
                    raise FileNotFoundError(f"{execute_dir} not found locally.")
            command_r = f"cd {execute_dir}; {command}"
        else:
            command_r = command

        if self.machine_type == "local":
            return self._run_local(command_r)
        else:
            return self._run_remote(command_r)

    def _run_local(self, command_r: str, max_retries: int = 10):
        for attempt in range(max_retries):
            for sub_attempt in range(3):
                try:
                    proc = subprocess.run(
                        command_r,
                        shell=True,
                        stdout=PIPE,
                        stderr=PIPE,
                        text=True,
                        timeout=1200,
                    )
                    if proc.returncode == 0:
                        return proc.stdout, proc.stderr
                    # Log the actual error before retrying
                    logger.warning(f"Local command returned rc={proc.returncode} (attempt {attempt}, sub {sub_attempt})")
                    if proc.stdout.strip():
                        logger.warning(f"  stdout: {proc.stdout.strip()[:500]}")
                    if proc.stderr.strip():
                        logger.warning(f"  stderr: {proc.stderr.strip()[:500]}")
                    break  # non-zero exit, go to outer retry
                except subprocess.TimeoutExpired:
                    logger.warning(f"Local command timeout (sub-attempt {sub_attempt})")
                    time.sleep(60)

            logger.warning(f"Local command failed (attempt {attempt}). Retrying in {self.ssh_retry_time}s.")
            time.sleep(self.ssh_retry_time)

        raise RuntimeError(f"Local command failed after {max_retries} retries: {command_r}")

    def _run_remote(self, command_r: str):
        self.ssh_open()
        _, pstdout, pstderr = self.ssh.exec_command(command=command_r)
        exit_status = pstdout.channel.recv_exit_status()
        stdout = pstdout.read().decode("utf-8").strip()
        stderr = pstderr.read().decode("utf-8").strip()
        if exit_status != 0:
            logger.error(f"Remote command failed: {command_r}")
            logger.error(f"stdout={stdout}")
            logger.error(f"stderr={stderr}")
            raise RuntimeError(f"Remote command failed (exit={exit_status}): {command_r}")
        return stdout, stderr

    # ── Filesystem queries ────────────────────────────────────────

    def _sftp_lstat_with_retry(self, path: str, max_retries=3, timeout_sec=5.0):
        self.ssh_open()
        for attempt in range(1, max_retries + 1):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self.sftp.lstat, path)
            try:
                return future.result(timeout=timeout_sec)
            except Exception as e:
                logger.warning(f"SFTP lstat attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"SFTP lstat failed for '{path}'")
                time.sleep(1)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

    def is_file(self, file_name: str) -> bool:
        if self.machine_type == "local":
            return os.path.isfile(file_name)
        fileattr = self._sftp_lstat_with_retry(file_name)
        return stat.S_ISREG(fileattr.st_mode)

    def is_dir(self, dir_name: str) -> bool:
        if self.machine_type == "local":
            return os.path.isdir(dir_name)
        fileattr = self._sftp_lstat_with_retry(dir_name)
        return stat.S_ISDIR(fileattr.st_mode)

    def exist(self, object_name: str) -> bool:
        if self.machine_type == "local":
            return os.path.exists(object_name)
        fileattr = self._sftp_lstat_with_retry(object_name)
        return stat.S_ISDIR(fileattr.st_mode) or stat.S_ISREG(fileattr.st_mode)

    # ── Job list queries ──────────────────────────────────────────

    def get_job_list(self):
        return self.run_command(self.jobcheck)

    def get_job_list_as_text(self):
        stdout, _ = self.get_job_list()
        return stdout.split("\n")

    def delete_job(self, jobid):
        stdout, _ = self.run_command(f"{self.jobdel} {jobid}")
        return stdout.split("\n")


class Machines_handler:
    """Handles data transfer between localhost and a server machine.

    The client is always localhost — only one Machine (server) is needed.
    """

    def __init__(self, server_machine_name: str):
        self.server_machine = Machine(server_machine_name)

    def ssh_close(self):
        self.server_machine.ssh_close()

    # ── put / get conveniences ────────────────────────────────────

    def put(self, from_file, to_file, exclude_patterns=None):
        self._transfer(from_file, to_file, exclude_patterns, dir_transfer=False, direction="put")

    def put_dir(self, from_dir, to_dir, exclude_patterns=None):
        self._transfer(from_dir, to_dir, exclude_patterns, dir_transfer=True, direction="put")

    def get(self, from_file, to_file, exclude_patterns=None):
        self._transfer(from_file, to_file, exclude_patterns, dir_transfer=False, direction="get")

    def get_dir(self, from_dir, to_dir, exclude_patterns=None):
        self._transfer(from_dir, to_dir, exclude_patterns, dir_transfer=True, direction="get")

    # ── SFTP primitives ───────────────────────────────────────────

    def _get_sftp_file(self, source, target, exclude_patterns):
        if exclude_patterns and any(re.match(p, os.path.basename(source)) for p in exclude_patterns):
            return
        os.makedirs(os.path.dirname(target), exist_ok=True)
        self.server_machine.ssh_open()
        self.server_machine.sftp.get(source, target)

    def _put_sftp_file(self, source, target, exclude_patterns):
        if exclude_patterns and any(re.match(p, os.path.basename(source)) for p in exclude_patterns):
            return
        self.server_machine.run_command(f"mkdir -p {os.path.dirname(target)}")
        self.server_machine.ssh_open()
        self.server_machine.sftp.put(source, target)

    def _get_sftp_dir(self, source, target, exclude_patterns):
        os.makedirs(target, exist_ok=True)
        self.server_machine.ssh_open()
        sftp = self.server_machine.sftp
        for item in sftp.listdir_attr(source):
            name = item.filename
            if exclude_patterns and any(re.match(p, name) for p in exclude_patterns):
                continue
            remote_path = os.path.join(source, name)
            local_path = os.path.join(target, name)
            if stat.S_ISREG(item.st_mode):
                sftp.get(remote_path, local_path)
            elif stat.S_ISDIR(item.st_mode):
                self._get_sftp_dir(remote_path, local_path, exclude_patterns)

    def _put_sftp_dir(self, source, target, exclude_patterns):
        self.server_machine.run_command(f"mkdir -p {target}")
        self.server_machine.ssh_open()
        sftp = self.server_machine.sftp
        for item in os.listdir(source):
            if exclude_patterns and any(re.match(p, item) for p in exclude_patterns):
                continue
            local_path = os.path.join(source, item)
            remote_path = os.path.join(target, item)
            if os.path.isfile(local_path):
                sftp.put(local_path, remote_path)
            elif os.path.isdir(local_path):
                self._put_sftp_dir(local_path, remote_path, exclude_patterns)

    # ── Core transfer logic ───────────────────────────────────────

    def _transfer(self, from_path, to_path, exclude_patterns, dir_transfer, direction):
        exclude_patterns = exclude_patterns or []

        if not pathlib.Path(from_path).is_absolute():
            raise ValueError(f"from_path must be absolute: {from_path}")
        if not pathlib.Path(to_path).is_absolute():
            raise ValueError(f"to_path must be absolute: {to_path}")

        # Ensure target directory exists
        to_dir = os.path.dirname(to_path) if not dir_transfer else to_path
        if direction == "put":
            self.server_machine.run_command(f"mkdir -p {to_dir}")
        else:
            os.makedirs(to_dir, exist_ok=True)

        if self.server_machine.machine_type == "local":
            logger.debug("Server is localhost; no network transfer needed.")
            return

        logger.info(f"  Transfer ({direction}): {from_path} -> {to_path}")

        if direction == "put":
            if dir_transfer:
                self._put_sftp_dir(from_path, to_path, exclude_patterns)
            else:
                self._put_sftp_file(from_path, to_path, exclude_patterns)
        else:
            if dir_transfer:
                self._get_sftp_dir(from_path, to_path, exclude_patterns)
            else:
                self._get_sftp_file(from_path, to_path, exclude_patterns)
