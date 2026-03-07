"""Configuration paths for jqmc-workflow compute infrastructure.

Config directory resolution order:

1. ``./jqmc_setting_local/``  (project-local override, if it exists in CWD)
2. ``~/.jqmc_setting/``       (user-global default)

Contents of the config directory:
  - ``machine_data.yaml``    : machine definitions (localhost, remote clusters)
  - ``<machine_name>/``      : per-machine job templates and queue settings
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
from logging import getLogger

logger = getLogger("jqmc-workflow").getChild(__name__)

# Source paths
_source_dir = os.path.abspath(os.path.dirname(__file__))
_package_root = os.path.abspath(os.path.join(_source_dir, ".."))

# User-global config: ~/.jqmc_setting/
_global_config_dir = os.path.join(os.path.expanduser("~"), ".jqmc_setting")

# Project-local override directory name
_local_config_dirname = "jqmc_setting_local"

# Template shipped with the package (used to bootstrap on first run)
template_dir = os.path.join(_source_dir, "template")

# Test directory (if exists)
test_dir = os.path.join(_package_root, "tests")


# Resolved once and cached so that later os.chdir() calls
# (e.g. inside Launcher / workflow execution) do not break resolution.
_resolved_config_dir: str | None = None


def get_config_dir() -> str:
    """Return the active configuration directory.

    If ``jqmc_setting_local/`` exists in the current working directory
    it takes priority over the global ``~/.jqmc_setting/``.

    The result is cached after the first call so that subsequent
    ``os.chdir()`` inside the workflow engine does not alter resolution.
    Call :func:`reset_config_dir` to clear the cache (mainly for tests).
    """
    global _resolved_config_dir
    if _resolved_config_dir is not None:
        return _resolved_config_dir

    local = os.path.join(os.getcwd(), _local_config_dirname)
    if os.path.isdir(local):
        _resolved_config_dir = local
        logger.debug(f"Using local config: {_resolved_config_dir}")
    else:
        _resolved_config_dir = _global_config_dir
        logger.debug(f"Using global config: {_resolved_config_dir}")
    return _resolved_config_dir


def reset_config_dir() -> None:
    """Clear the cached config directory (useful in unit tests)."""
    global _resolved_config_dir
    _resolved_config_dir = None


# Backward-compatible alias (read-only usage only)
config_dir = _global_config_dir
