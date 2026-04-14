"""Data transfer between localhost and a remote server.

The client is always localhost, so only server_machine_name is required.
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

import fnmatch
import glob
import os
from logging import getLogger

from ._machine import Machine, Machines_handler

logger = getLogger("jqmc-workflow").getChild(__name__)


class Data_transfer:
    """Convenience layer over Machines_handler for local ↔ remote transfers.

    Parameters
    ----------
    server_machine_name : str
        Name of the server machine as defined in ~/.jqmc_setting/machine_data.yaml.
    safe_mode : bool
        If True, verify root directories exist before transfer.
    """

    def __init__(self, server_machine_name: str, safe_mode: bool = False):
        self.server_machine = Machine(server_machine_name)
        self.machine_handler = Machines_handler(self.server_machine)
        self.safe_mode = safe_mode

        # Local root from machine_data.yaml (for path mapping)
        self._local_root = self._resolve_local_root()

    def _resolve_local_root(self) -> str:
        """Determine the local workspace_root from localhost entry."""
        try:
            local_m = Machine("localhost")
            try:
                return local_m.workspace_root or os.path.expanduser("~")
            finally:
                local_m.ssh_close()
        except (KeyError, FileNotFoundError):
            return os.path.expanduser("~")

    def ssh_close(self):
        self.server_machine.ssh_close()
        self.machine_handler.ssh_close()

    # ── put (local → remote) ──────────────────────────────────────

    def put_objects(self, from_objects=None, exclude_patterns=None, *, work_dir=None):
        """Upload files from *work_dir* to the corresponding remote directory.

        Parameters
        ----------
        from_objects : list[str], optional
            Basenames or glob patterns of files to upload.  When empty,
            the entire *work_dir* is synced.
        exclude_patterns : list[str], optional
            Glob patterns to exclude from the transfer.
        work_dir : str, optional
            Local directory that maps to the remote workspace.  When
            *None*, falls back to ``os.getcwd()`` for backward
            compatibility, but callers should always pass this
            explicitly.
        """
        from_objects = from_objects or []
        exclude_patterns = exclude_patterns or []

        local_root = self._local_root
        server_root = self.server_machine.workspace_root

        if self.safe_mode:
            if server_root and not self.server_machine.is_dir(server_root):
                raise FileNotFoundError(f"Server root {server_root} not found.")

        local_cwd = os.path.abspath(work_dir) if work_dir else os.path.abspath(os.getcwd())

        if self.server_machine.machine_type == "local":
            logger.debug("Server is localhost; skipping put_objects.")
            return

        if server_root is None:
            raise ValueError("server workspace_root is not configured.")

        if local_root and local_root not in local_cwd:
            raise ValueError(f"work_dir ({local_cwd}) is not under local root ({local_root}). Cannot map paths to remote.")

        if not from_objects:
            # Sync entire work_dir
            client_dir = local_cwd
            server_dir = local_cwd.replace(local_root, server_root)
            self.machine_handler.put_dir(
                from_dir=client_dir,
                to_dir=server_dir,
                exclude_patterns=exclude_patterns,
            )
        else:
            # Expand local glob patterns (e.g. "*.h5")
            expanded = []
            for obj in from_objects:
                if any(c in obj for c in ("*", "?", "[")):
                    expanded.extend(glob.glob(os.path.join(local_cwd, obj)))
                else:
                    expanded.append(os.path.join(local_cwd, obj))

            for obj_abs in expanded:
                from_path = obj_abs
                to_path = obj_abs.replace(local_root, server_root)
                if os.path.isfile(from_path):
                    self.machine_handler.put(
                        from_file=from_path,
                        to_file=to_path,
                        exclude_patterns=exclude_patterns,
                    )
                elif os.path.isdir(from_path):
                    self.machine_handler.put_dir(
                        from_dir=from_path,
                        to_dir=to_path,
                        exclude_patterns=exclude_patterns,
                    )

    # ── get (remote → local) ──────────────────────────────────────

    def get_objects(self, from_objects=None, exclude_patterns=None, *, work_dir=None, optional_patterns=None):
        """Download files from the remote directory to *work_dir*.

        Parameters
        ----------
        from_objects : list[str], optional
            Basenames or glob patterns of files to download.  When
            empty, the entire remote directory is synced.
        exclude_patterns : list[str], optional
            Glob patterns to exclude from the transfer.
        work_dir : str, optional
            Local directory that maps to the remote workspace.  When
            *None*, falls back to ``os.getcwd()`` for backward
            compatibility, but callers should always pass this
            explicitly.
        optional_patterns : list[str], optional
            Basenames or glob patterns of files that are non-essential.
            When a file matching one of these patterns is missing on
            the server, a warning is logged instead of raising
            ``FileNotFoundError``.
        """
        from_objects = from_objects or []
        exclude_patterns = exclude_patterns or []
        optional_patterns = optional_patterns or []

        local_root = self._local_root
        server_root = self.server_machine.workspace_root

        if self.server_machine.machine_type == "local":
            logger.debug("Server is localhost; skipping get_objects.")
            return

        if server_root is None:
            raise ValueError("server workspace_root is not configured.")

        local_cwd = os.path.abspath(work_dir) if work_dir else os.path.abspath(os.getcwd())
        if local_root and local_root not in local_cwd:
            raise ValueError(f"work_dir ({local_cwd}) is not under local root ({local_root}).")

        client_dir = local_cwd
        server_dir = local_cwd.replace(local_root, server_root)

        if not from_objects:
            self.machine_handler.get_dir(
                from_dir=server_dir,
                to_dir=client_dir,
                exclude_patterns=exclude_patterns,
            )
        else:
            # Expand glob patterns (e.g. "*.h5") via SFTP listdir
            expanded = []
            for obj in from_objects:
                if any(c in obj for c in ("*", "?", "[")):
                    pattern = os.path.basename(obj)
                    self.server_machine.ssh_open()
                    try:
                        entries = self.server_machine.sftp.listdir(server_dir)
                    except IOError:
                        entries = []
                    matched = [e for e in entries if fnmatch.fnmatch(e, pattern)]
                    expanded.extend(matched)
                else:
                    expanded.append(obj)

            for obj in expanded:
                from_path = os.path.join(server_dir, obj)
                if not self.server_machine.exist(object_name=from_path):
                    if any(fnmatch.fnmatch(obj, pat) for pat in optional_patterns):
                        logger.warning(f"Optional file not found on server (skipped): {from_path}")
                        continue
                    raise FileNotFoundError(f"{from_path} not found on server.")
                to_path = from_path.replace(server_root, local_root)
                if self.server_machine.is_file(file_name=from_path):
                    self.machine_handler.get(
                        from_file=from_path,
                        to_file=to_path,
                        exclude_patterns=exclude_patterns,
                    )
                else:
                    self.machine_handler.get_dir(
                        from_dir=from_path,
                        to_dir=to_path,
                        exclude_patterns=exclude_patterns,
                    )
