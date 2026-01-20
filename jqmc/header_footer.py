"""header and footer modules."""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# This file is part of jqmc.
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

from datetime import datetime
from logging import getLogger

# set logger
logger = getLogger("jqmc").getChild(__name__)

try:
    from jqmc._version import version as jqmc_version
except (ModuleNotFoundError, ImportError):
    jqmc_version = "unknown"


def _print_header() -> None:
    """Print Header."""
    logger.info(f"Program starts {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("-" * 48)
    logger.info("      d8b  .d88888b.  888b     d888  .d8888b.   ")
    logger.info('      Y8P d88P" "Y88b 8888b   d8888 d88P  Y88b  ')
    logger.info("          888     888 88888b.d88888 888    888  ")
    logger.info("     8888 888     888 888Y88888P888 888         ")
    logger.info("      888 888     888 888 Y888P 888 888         ")
    logger.info("      888 888 Y8b 888 888  Y8P  888 888    888  ")
    logger.info("      888 Y88b.Y8b88P 888       888 Y88b  d88P  ")
    logger.info('      888  "Y888888"  888       888   Y8888P    ')
    logger.info("      888        Y8b                            ")
    logger.info("     d88P                                       ")
    logger.info("   888P                           Made on earth ")
    logger.info("-" * 48)
    logger.info("")
    logger.info("jQMC: Python-based real-space ab-initio Quantum Monte Carlo package.")
    logger.info(f"version = {jqmc_version}.")
    logger.info("")
    logger.info("Authors: Kosuke Nakano [kousuke_1123@icloud.com]")
    logger.info("")
    logger.info("***The 3-Clause BSD License***")
    logger.info("")
    logger.info("Copyright (C) 2024- Kosuke Nakano")
    logger.info("All rights reserved.")
    logger.info("")


def _print_footer() -> None:
    """Print Footer."""
    logger.info(f"Program ends {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}")
