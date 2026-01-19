"""collections of unit tests."""

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
import sys
from pathlib import Path

import toml

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.jqmc_cli import _cli  # noqa: E402
from jqmc.jqmc_tool import (  # noqa: E402
    lrdmc_compute_energy,
    lrdmc_extrapolate_energy,
    lrdmc_generate_input,
    mcmc_compute_energy,
    mcmc_generate_input,
    trexio_convert_to,
    vmc_generate_input,
)


def test_jqmc_tool_trexio_conversion(tmp_path):
    """Test the conversion of TREXIO files to Hamiltonian data."""
    trexio_convert_to(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "water_ccecp_ccpvqz.h5"),
        hamiltonian_file=os.path.join(tmp_path, "hamiltonian_data.h5"),
        j1_parmeter=1.0,
        j2_parmeter=1.0,
        j3_basis_type="ao-medium",
    )


def test_jqmc_cli_run_mcmc(tmp_path, monkeypatch):
    """Test the MCMC run command."""
    root_dir = os.getcwd()
    # trexio conversion
    os.chdir(root_dir)
    trexio_convert_to(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "H2_ecp_ccpvtz_cart.h5"),
        hamiltonian_file=os.path.join(tmp_path, "hamiltonian_data.h5"),
        j1_parmeter=None,
        j2_parmeter=1.0,
        j3_basis_type="ao-small",
    )
    os.chdir(root_dir)

    # generate input
    os.chdir(root_dir)
    mcmc_generate_input(flag=True, filename=os.path.join(tmp_path, "mcmc_input.toml"), exclude_comment=True)
    with open(os.path.join(tmp_path, "mcmc_input.toml"), "r") as f:
        dict_toml = toml.load(f)
        dict_toml["control"]["restart"] = False
        dict_toml["control"]["hamiltonian_h5"] = "hamiltonian_data.h5"
        dict_toml["control"]["restart_chk"] = "restart.chk"
        dict_toml["mcmc"]["num_mcmc_steps"] = 50
        dict_toml["mcmc"]["num_mcmc_bin_blocks"] = 5
        dict_toml["mcmc"]["num_mcmc_warmup_steps"] = 30
    with open(os.path.join(tmp_path, "mcmc_input.toml"), "w") as f:
        toml.dump(dict_toml, f)
    os.chdir(root_dir)

    # run MCMC
    os.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["jqmc", "mcmc_input.toml"])
    _cli()
    os.chdir(root_dir)

    # post MCMC
    os.chdir(root_dir)
    mcmc_compute_energy(restart_chk=os.path.join(tmp_path, "restart.chk"), num_mcmc_bin_blocks=5, num_mcmc_warmup_steps=30)
    os.chdir(root_dir)

    os.chdir(tmp_path)
    mcmc_compute_energy(restart_chk="restart.chk", num_mcmc_bin_blocks=5, num_mcmc_warmup_steps=30)
    os.chdir(root_dir)

    # run MCMC(restart)
    os.chdir(root_dir)
    with open(os.path.join(tmp_path, "mcmc_input.toml"), "r") as f:
        dict_toml = toml.load(f)
        dict_toml["control"]["restart"] = True
        dict_toml["control"]["hamiltonian_h5"] = None
        dict_toml["control"]["restart_chk"] = "restart.chk"
        dict_toml["mcmc"]["num_mcmc_steps"] = 10
    with open(os.path.join(tmp_path, "mcmc_input.toml"), "w") as f:
        toml.dump(dict_toml, f)
    os.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["jqmc", "mcmc_input.toml"])
    _cli()
    os.chdir(root_dir)


def test_jqmc_cli_run_vmc(tmp_path, monkeypatch):
    """Test the VMC run command."""
    root_dir = os.getcwd()
    # trexio conversion
    os.chdir(root_dir)
    trexio_convert_to(
        trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "H2_ecp_ccpvtz_cart.h5"),
        hamiltonian_file=os.path.join(tmp_path, "hamiltonian_data.h5"),
        j1_parmeter=None,
        j2_parmeter=1.0,
        j3_basis_type="ao-small",
    )
    os.chdir(root_dir)

    # generate input
    os.chdir(root_dir)
    vmc_generate_input(flag=True, filename=os.path.join(tmp_path, "vmc_input.toml"), exclude_comment=True)
    with open(os.path.join(tmp_path, "vmc_input.toml"), "r") as f:
        dict_toml = toml.load(f)
        dict_toml["control"]["restart"] = False
        dict_toml["control"]["hamiltonian_h5"] = "hamiltonian_data.h5"
        dict_toml["control"]["restart_chk"] = "restart.chk"
        dict_toml["vmc"]["num_opt_steps"] = 2
        dict_toml["vmc"]["num_mcmc_steps"] = 50
        dict_toml["vmc"]["num_mcmc_bin_blocks"] = 10
        dict_toml["vmc"]["num_mcmc_warmup_steps"] = 0
    with open(os.path.join(tmp_path, "vmc_input.toml"), "w") as f:
        toml.dump(dict_toml, f)
    os.chdir(root_dir)

    # run VMC
    os.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["jqmc", "vmc_input.toml"])
    _cli()
    os.chdir(root_dir)

    # run VMCopt(restart)
    os.chdir(root_dir)
    with open(os.path.join(tmp_path, "vmc_input.toml"), "r") as f:
        dict_toml = toml.load(f)
        dict_toml["control"]["restart"] = True
        dict_toml["control"]["hamiltonian_chk"] = None
        dict_toml["control"]["restart_chk"] = "restart.chk"
    with open(os.path.join(tmp_path, "vmc_input.toml"), "w") as f:
        toml.dump(dict_toml, f)
    os.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["jqmc", "vmc_input.toml"])
    _cli()
    os.chdir(root_dir)


def test_jqmc_cli_run_lrdmc(tmp_path, monkeypatch):
    """Test the LRDMC run command."""
    root_dir = os.getcwd()

    # alat
    ## WIP: multiple alat values are not accepted due to a JAX compilation error
    alat_list = [0.2, 0.3, 0.4, 0.5]

    # generate input
    os.chdir(root_dir)
    for alat in alat_list:
        tmp_alat_path = os.path.join(tmp_path, str(alat))
        os.makedirs(tmp_alat_path, exist_ok=True)
        # trexio conversion
        os.chdir(root_dir)
        trexio_convert_to(
            trexio_file=os.path.join(os.path.dirname(__file__), "trexio_example_files", "H2_ecp_ccpvtz_cart.h5"),
            hamiltonian_file=os.path.join(tmp_alat_path, "hamiltonian_data.h5"),
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type="ao-small",
        )
        os.chdir(root_dir)
        lrdmc_generate_input(flag=True, filename=os.path.join(tmp_alat_path, "lrdmc_input.toml"), exclude_comment=True)
        with open(os.path.join(tmp_alat_path, "lrdmc_input.toml"), "r") as f:
            dict_toml = toml.load(f)
            dict_toml["control"]["restart"] = False
            dict_toml["control"]["hamiltonian_h5"] = "hamiltonian_data.h5"
            dict_toml["control"]["restart_chk"] = "restart.chk"
            dict_toml["lrdmc"]["num_mcmc_steps"] = 50
            dict_toml["lrdmc"]["alat"] = alat
            dict_toml["lrdmc"]["num_gfmc_bin_blocks"] = 5
            dict_toml["lrdmc"]["num_gfmc_warmup_steps"] = 30
            dict_toml["lrdmc"]["num_gfmc_collect_steps"] = 5
        with open(os.path.join(tmp_alat_path, "lrdmc_input.toml"), "w") as f:
            toml.dump(dict_toml, f)
        os.chdir(root_dir)

        # run LRDMC
        os.chdir(tmp_alat_path)
        monkeypatch.setattr(sys, "argv", ["jqmc", "lrdmc_input.toml"])
        _cli()
        os.chdir(root_dir)

    # post LRDMC (each alat)
    for alat in alat_list:
        tmp_alat_path = os.path.join(tmp_path, str(alat))
        os.chdir(root_dir)
        lrdmc_compute_energy(
            restart_chk=os.path.join(tmp_alat_path, "restart.chk"),
            num_gfmc_bin_block=5,
            num_gfmc_warmup_steps=30,
            num_gfmc_collect_steps=5,
        )
        os.chdir(tmp_alat_path)
        lrdmc_compute_energy(
            restart_chk="restart.chk", num_gfmc_bin_block=5, num_gfmc_warmup_steps=30, num_gfmc_collect_steps=5
        )
        os.chdir(root_dir)

    # post LRDMC (extrapolation)
    os.chdir(tmp_path)
    restart_chks = [os.path.join(tmp_path, str(alat), "restart.chk") for alat in alat_list]
    lrdmc_extrapolate_energy(
        restart_chks=restart_chks,
        polynomial_order=1,
        plot_graph=False,
        save_graph=None,
        num_gfmc_bin_block=5,
        num_gfmc_warmup_steps=30,
        num_gfmc_collect_steps=5,
    )
    os.chdir(root_dir)

    # run LRDMC(restart)
    os.chdir(root_dir)
    for alat in alat_list:
        tmp_alat_path = os.path.join(tmp_path, str(alat))
        with open(os.path.join(tmp_alat_path, "lrdmc_input.toml"), "r") as f:
            dict_toml = toml.load(f)
            dict_toml["control"]["restart"] = True
            dict_toml["control"]["hamiltonian_h5"] = None
            dict_toml["control"]["restart_chk"] = "restart.chk"
            dict_toml["lrdmc"]["num_mcmc_steps"] = 10
        with open(os.path.join(tmp_alat_path, "lrdmc_input.toml"), "w") as f:
            toml.dump(dict_toml, f)
        os.chdir(tmp_alat_path)
        monkeypatch.setattr(sys, "argv", ["jqmc", "lrdmc_input.toml"])
        _cli()
    os.chdir(root_dir)


if __name__ == "__main__":
    import tempfile
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    class _MonkeyPatch:
        def setattr(self, target, name, value):
            setattr(target, name, value)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Running tests in {tmp_dir}")
        monkeypatch = _MonkeyPatch()

        print("Running test_jqmc_tool_trexio_conversion...")
        test_jqmc_tool_trexio_conversion(tmp_dir)

        print("Running test_jqmc_cli_run_mcmc...")
        test_jqmc_cli_run_mcmc(tmp_dir, monkeypatch)

        print("Running test_jqmc_cli_run_vmc...")
        test_jqmc_cli_run_vmc(tmp_dir, monkeypatch)

        print("Running test_jqmc_cli_run_lrdmc...")
        test_jqmc_cli_run_lrdmc(tmp_dir, monkeypatch)

        print("All tests passed!")
