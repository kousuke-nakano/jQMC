"""WF_Workflow — TREXIO to hamiltonian_data.h5 conversion.

Wraps ``jqmc-tool trexio convert-to`` which converts a TREXIO file (.h5)
into the internal ``hamiltonian_data.h5`` format, optionally attaching
Jastrow one-body, two-body, three-body, and neural-network factors.

This workflow runs **locally** (no remote job submission).
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
import shlex
import subprocess
from logging import getLogger
from typing import List, Optional

from .workflow import Workflow

logger = getLogger("jqmc-workflow").getChild(__name__)


class WF_Workflow(Workflow):
    """Convert a TREXIO file to ``hamiltonian_data.h5``.

    Calls ``jqmc-tool trexio convert-to`` under the hood.

    Parameters
    ----------
    trexio_file : str
        Path to the input TREXIO ``.h5`` file.
    hamiltonian_file : str
        Output filename (default: ``"hamiltonian_data.h5"``).
    j1_parameter : float, optional
        Jastrow one-body parameter (``-j1``).
    j2_parameter : float, optional
        Jastrow two-body parameter (``-j2``).
    j3_basis_type : str, optional
        Jastrow three-body basis-set type (``-j3``).
        One of ``"ao"``, ``"ao-full"``, ``"ao-small"``, ``"ao-medium"``,
        ``"ao-large"``, ``"mo"``, ``"none"``, or ``None`` (disabled).
    j_nn_type : str, optional
        Neural-network Jastrow type (``-j-nn-type``), e.g. ``"schnet"``.
    j_nn_params : list[str], optional
        Extra NN Jastrow parameters (``-jp key=value``).
    ao_conv_to : str, optional
        Convert AOs after building the Hamiltonian (``--ao-conv-to``).
        ``"cart"``  → convert to Cartesian AOs,
        ``"sphe"`` → convert to spherical-harmonic AOs,
        ``None``    → keep the original representation.

    Example
    -------
    >>> wf = WF_Workflow(
    ...     trexio_file="molecular.h5",
    ...     j1_parameter=1.0,
    ...     j2_parameter=0.5,
    ...     j3_basis_type="ao-small",
    ... )
    >>> status, out_files, out_values = wf.launch()

    Notes
    -----
    This workflow runs **locally** — no remote job submission is
    involved.  It calls ``jqmc-tool trexio convert-to`` via
    :func:`subprocess.run`.

    See Also
    --------
    VMC_Workflow : Optimise the wavefunction produced by this step.
    """

    def __init__(
        self,
        trexio_file: str = "trexio.h5",
        hamiltonian_file: str = "hamiltonian_data.h5",
        j1_parameter: Optional[float] = None,
        j2_parameter: Optional[float] = None,
        j3_basis_type: Optional[str] = None,
        j_nn_type: Optional[str] = None,
        j_nn_params: Optional[List[str]] = None,
        ao_conv_to: Optional[str] = None,
    ):
        super().__init__()
        self.trexio_file = trexio_file
        self.hamiltonian_file = hamiltonian_file
        self.j1_parameter = j1_parameter
        self.j2_parameter = j2_parameter
        self.j3_basis_type = j3_basis_type
        self.j_nn_type = j_nn_type
        self.j_nn_params = j_nn_params or []

        if ao_conv_to is not None and ao_conv_to not in ("cart", "sphe"):
            raise ValueError(f"ao_conv_to must be None, 'cart', or 'sphe', got {ao_conv_to!r}")
        self.ao_conv_to = ao_conv_to

    def _build_command(self) -> str:
        """Build the ``jqmc-tool trexio convert-to`` CLI command."""
        cmd = ["jqmc-tool", "trexio", "convert-to", self.trexio_file]
        cmd += ["-o", self.hamiltonian_file]

        if self.j1_parameter is not None:
            cmd += ["-j1", str(self.j1_parameter)]
        if self.j2_parameter is not None:
            cmd += ["-j2", str(self.j2_parameter)]
        if self.j3_basis_type is not None:
            cmd += ["-j3", str(self.j3_basis_type)]
        if self.j_nn_type is not None:
            cmd += ["-j-nn-type", str(self.j_nn_type)]
        for param in self.j_nn_params:
            cmd += ["-jp", str(param)]
        if self.ao_conv_to is not None:
            cmd += ["--ao-conv-to", str(self.ao_conv_to)]

        return shlex.join(cmd)

    async def async_launch(self):
        """Run the TREXIO→hamiltonian conversion (locally).

        Returns
        -------
        tuple
            ``(status, output_files, output_values)``
        """
        command = self._build_command()
        logger.info(f"  Running: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(f"stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed (rc={e.returncode}): {e.stderr}")
            self.status = "failed"
            return self.status, [], {}

        if not os.path.isfile(self.hamiltonian_file):
            logger.error(f"Output file not found: {self.hamiltonian_file}")
            self.status = "failed"
            return self.status, [], {}

        self.status = "success"
        self.output_files = [self.hamiltonian_file]
        self.output_values = {"hamiltonian_file": self.hamiltonian_file}
        return self.status, self.output_files, self.output_values
