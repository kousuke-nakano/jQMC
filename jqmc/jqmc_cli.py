"""LRDMC module."""

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
# * Neither the name of the phonopy project nor the names of its
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

import logging

# python modules
import os
import pickle
import sys
import zipfile
from logging import Formatter, StreamHandler, getLogger

import jax
import toml

# MPI
from mpi4py import MPI

# jQMC
from .header_footer import print_footer, print_header
from .jqmc_kernel import GFMC, MCMC, QMC
from .jqmc_miscs import cli_parameters

# MPI related
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# create new logger level for development
DEVEL_LEVEL = 5
logging.addLevelName(DEVEL_LEVEL, "DEVEL")


# a new method to create a new logger
def _loglevel_devel(self, message, *args, **kwargs):
    if self.isEnabledFor(DEVEL_LEVEL):
        self._log(DEVEL_LEVEL, message, args, **kwargs)


logging.Logger.devel = _loglevel_devel

# set logger
logger = getLogger("jqmc").getChild(__name__)

# jax-MPI related
try:
    jax.distributed.initialize()
except ValueError:
    pass


def cli():
    """Main function for tests."""
    logger_level = "MPI-INFO"

    log = getLogger("jqmc")

    if logger_level == "MPI-INFO":
        if mpi_rank == 0:
            log.setLevel("INFO")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("INFO")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
        else:
            log.setLevel("ERROR")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("ERROR")
            handler_format = Formatter(f"MPI-rank={mpi_rank}: %(name)s - %(levelname)s - %(lineno)d - %(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
    else:
        log.setLevel(logger_level)
        stream_handler = StreamHandler(sys.stdout)
        stream_handler.setLevel(logger_level)
        handler_format = Formatter(f"MPI-rank={mpi_rank}: %(name)s - %(levelname)s - %(lineno)d - %(message)s")
        stream_handler.setFormatter(handler_format)
        log.addHandler(stream_handler)

    # print header
    print_header()

    # global JAX device
    global_device_info = jax.devices()
    # local JAX device
    num_devices = jax.local_devices()
    device_info_str = f"Rank {mpi_rank}: {num_devices}"
    local_device_info = mpi_comm.allgather(device_info_str)
    # print recognized XLA devices
    logger.info("*** XLA Global devices recognized by JAX***")
    logger.info(global_device_info)
    logger.info("*** XLA Local devices recognized by JAX***")
    logger.info(local_device_info)
    logger.info("")

    if len(sys.argv) == 1:
        raise ValueError("Please specify input toml file.")
    elif len(sys.argv) > 2:
        raise ValueError("More than one input toml files are not acceptable.")
    else:
        toml_file = sys.argv[1]
        if not os.path.isfile(toml_file):
            raise FileNotFoundError(f"toml_file = {toml_file} does not exist.")
        else:
            dict_toml = toml.load(open(toml_file))

    logger.info(f"Input file = {toml_file}")
    if not all([type(value) is dict for value in dict_toml.values()]):
        raise ValueError("The format of the toml file is wrong. See the tutorial.")

    logger.info("")
    logger.info("Input parameters are::")
    logger.info("")
    for section, dict_item in dict_toml.items():
        logger.info(f"**section:{section}**")
        for key, item in dict_item.items():
            logger.info(f"  {key}={item}")
        logger.info("")

    # default parameters
    parameters = cli_parameters.copy()

    # control section
    section = "control"
    for key in parameters[section].keys():
        try:
            parameters[section][key] = dict_toml[section][key]
        except KeyError:
            if parameters[section][key] is None:
                logger.error(f"{key} should be specified.")
                sys.exit(1)
            else:
                logger.warning(f"The default value of {key} = {parameters[section][key]}.")

    job_type = parameters["control"]["job_type"]
    mcmc_seed = parameters["control"]["mcmc_seed"]
    number_of_walkers = parameters["control"]["number_of_walkers"]
    max_time = parameters["control"]["max_time"]
    restart = parameters["control"]["restart"]
    restart_chk = parameters["control"]["restart_chk"]
    hamiltonian_chk = parameters["control"]["hamiltonian_chk"]

    # VMC
    if job_type == "vmc":
        logger.info("***Variational Monte Carlo***")

        # vmc section
        section = "vmc"
        for key in parameters[section].keys():
            try:
                parameters[section][key] = dict_toml[section][key]
            except KeyError:
                if parameters[section][key] is None:
                    logger.error(f"{key} should be specified.")
                    sys.exit(1)
                else:
                    logger.warning(f"The default value of {key} = {parameters[section][key]}.")

        # parameters
        num_mcmc_steps = parameters["vmc"]["num_mcmc_steps"]
        num_mcmc_per_measurement = parameters["vmc"]["num_mcmc_per_measurement"]
        num_mcmc_warmup_steps = parameters["vmc"]["num_mcmc_warmup_steps"]
        num_mcmc_bin_blocks = parameters["vmc"]["num_mcmc_bin_blocks"]
        epsilon_AS = parameters["vmc"]["epsilon_AS"]

        # check num_mcmc_steps, num_mcmc_warmup_steps, num_mcmc_bin_blocks
        if num_mcmc_steps < num_mcmc_warmup_steps:
            raise ValueError("num_mcmc_steps should be larger than num_mcmc_warmup_steps")
        if num_mcmc_steps - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
            raise ValueError("(num_mcmc_steps - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            """Unzip the checkpoint file for each process and load them."""
            filename = f"{mpi_rank}_{restart_chk}"
            with zipfile.ZipFile(restart_chk, "r") as zipf:
                data = zipf.read(filename)
                vmc = pickle.loads(data)

        else:
            with open(hamiltonian_chk, "rb") as f:
                hamiltonian_data = pickle.load(f)
                mcmc = MCMC(
                    hamiltonian_data=hamiltonian_data,
                    Dt=2.0,
                    mcmc_seed=mcmc_seed,
                    num_walkers=number_of_walkers,
                    num_mcmc_per_measurement=num_mcmc_per_measurement,
                    epsilon_AS=epsilon_AS,
                    comput_position_deriv=False,
                    comput_param_deriv=False,
                )
                vmc = QMC(mcmc)
        vmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)
        E_mean, E_std = vmc.get_E(
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            num_mcmc_bin_blocks=num_mcmc_bin_blocks,
        )

        logger.info("Final output(s):")
        logger.info(f"  Total Energy: E = {E_mean:.5f} +- {E_std:5f} Ha.")
        logger.info("")

        logger.info(f"Dump restart checkpoint file(s) to {restart_chk}.")
        logger.info("")

        # Save the checkpoint file for each process and zip them."""
        filename = f".{mpi_rank}_{restart_chk}"
        with open(filename, "wb") as f:
            pickle.dump(vmc, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Wait all MPI processes
        mpi_comm.Barrier()

        # Zip them.
        if mpi_rank == 0:
            filename_list = [f".{rank}_{restart_chk}" for rank in range(mpi_size)]
            with zipfile.ZipFile(restart_chk, "w", zipfile.ZIP_DEFLATED) as zipf:
                for filename in filename_list:
                    zipf.write(filename, arcname=filename.lstrip("."))
                    os.remove(filename)

    # VMCopt!
    if job_type == "vmcopt":
        logger.info("***WF optimization with Variational Monte Carlo***")

        # vmcopt section
        section = "vmcopt"
        for key in parameters[section].keys():
            try:
                parameters[section][key] = dict_toml[section][key]
            except KeyError:
                if parameters[section][key] is None:
                    logger.error(f"{key} should be specified.")
                    sys.exit(1)
                else:
                    logger.warning(f"The default value of {key} = {parameters[section][key]}.")

        logger.info("")

        # parameters
        num_mcmc_steps = parameters["vmcopt"]["num_mcmc_steps"]
        num_mcmc_per_measurement = parameters["vmcopt"]["num_mcmc_per_measurement"]
        num_mcmc_warmup_steps = parameters["vmcopt"]["num_mcmc_warmup_steps"]
        num_mcmc_bin_blocks = parameters["vmcopt"]["num_mcmc_bin_blocks"]
        epsilon_AS = parameters["vmcopt"]["epsilon_AS"]
        num_opt_steps = parameters["vmcopt"]["num_opt_steps"]
        wf_dump_freq = parameters["vmcopt"]["wf_dump_freq"]
        delta = parameters["vmcopt"]["delta"]
        epsilon = parameters["vmcopt"]["epsilon"]

        # check num_mcmc_steps, num_mcmc_warmup_steps, num_mcmc_bin_blocks
        if num_mcmc_steps < num_mcmc_warmup_steps:
            raise ValueError("num_mcmc_steps should be larger than num_mcmc_warmup_steps")
        if num_mcmc_steps - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
            raise ValueError("(num_mcmc_steps - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            """Unzip the checkpoint file for each process and load them."""
            filename = f"{mpi_rank}_{restart_chk}"
            with zipfile.ZipFile(restart_chk, "r") as zipf:
                data = zipf.read(filename)
                vmc = pickle.loads(data)
        else:
            with open(hamiltonian_chk, "rb") as f:
                hamiltonian_data = pickle.load(f)

                mcmc = MCMC(
                    hamiltonian_data=hamiltonian_data,
                    Dt=2.0,
                    mcmc_seed=mcmc_seed,
                    num_walkers=number_of_walkers,
                    num_mcmc_per_measurement=num_mcmc_per_measurement,
                    epsilon_AS=epsilon_AS,
                    comput_position_deriv=False,
                    comput_param_deriv=True,
                )
                vmc = QMC(mcmc)
        vmc.run_optimize(
            num_mcmc_steps=num_mcmc_steps,
            num_opt_steps=num_opt_steps,
            delta=delta,
            epsilon=epsilon,
            wf_dump_freq=wf_dump_freq,
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            max_time=max_time,
        )
        logger.info("")

        logger.info(f"Dump restart checkpoint file(s) to {restart_chk}.")

        logger.info("")

        # Save the checkpoint file for each process and zip them."""
        filename = f".{mpi_rank}_{restart_chk}"
        with open(filename, "wb") as f:
            pickle.dump(vmc, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Wait all MPI processes
        mpi_comm.Barrier()

        # Zip them.
        if mpi_rank == 0:
            filename_list = [f".{rank}_{restart_chk}" for rank in range(mpi_size)]
            with zipfile.ZipFile(restart_chk, "w", zipfile.ZIP_DEFLATED) as zipf:
                for filename in filename_list:
                    zipf.write(filename, arcname=filename.lstrip("."))
                    os.remove(filename)

        logger.info("")

    # LRDMC!
    if job_type == "lrdmc":
        logger.info("***Lattice Regularized diffusion Monte Carlo***")

        # vmcopt section
        section = "lrdmc"
        for key in parameters[section].keys():
            try:
                parameters[section][key] = dict_toml[section][key]
            except KeyError:
                if parameters[section][key] is None:
                    logger.error(f"{key} should be specified.")
                    sys.exit(1)
                else:
                    logger.warning(f"The default value of {key} = {parameters[section][key]}.")

        logger.info("")

        # parameters
        num_mcmc_steps = parameters["lrdmc"]["num_mcmc_steps"]
        num_mcmc_per_measurement = parameters["lrdmc"]["num_mcmc_per_measurement"]
        alat = parameters["lrdmc"]["alat"]
        non_local_move = parameters["lrdmc"]["non_local_move"]
        num_gfmc_warmup_steps = parameters["lrdmc"]["num_gfmc_warmup_steps"]
        num_gfmc_bin_blocks = parameters["lrdmc"]["num_gfmc_bin_blocks"]
        num_gfmc_collect_steps = parameters["lrdmc"]["num_gfmc_collect_steps"]
        E_scf = parameters["lrdmc"]["E_scf"]
        gamma = parameters["lrdmc"]["gamma"]

        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if num_mcmc_steps < num_gfmc_warmup_steps:
            raise ValueError("num_mcmc_steps should be larger than num_gfmc_warmup_steps")
        if num_mcmc_steps - num_gfmc_warmup_steps < num_gfmc_bin_blocks:
            raise ValueError("(num_mcmc_steps - num_gfmc_warmup_steps) should be larger than num_gfmc_bin_blocks.")
        if num_gfmc_bin_blocks < num_gfmc_collect_steps:
            raise ValueError("num_gfmc_bin_blocks should be larger than num_gfmc_collect_steps.")

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            """Unzip the checkpoint file for each process and load them."""
            filename = f"{mpi_rank}_{restart_chk}"
            with zipfile.ZipFile(restart_chk, "r") as zipf:
                data = zipf.read(filename)
                lrdmc = pickle.loads(data)
        else:
            with open(hamiltonian_chk, "rb") as f:
                hamiltonian_data = pickle.load(f)
                gfmc = GFMC(
                    hamiltonian_data=hamiltonian_data,
                    num_walkers=number_of_walkers,
                    num_mcmc_per_measurement=num_mcmc_per_measurement,
                    num_gfmc_collect_steps=num_gfmc_collect_steps,
                    mcmc_seed=mcmc_seed,
                    E_scf=E_scf,
                    gamma=gamma,
                    alat=alat,
                    non_local_move=non_local_move,
                )
                lrdmc = QMC(gfmc)
        lrdmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)
        E_mean, E_std = lrdmc.get_E(
            num_mcmc_warmup_steps=num_gfmc_warmup_steps,
            num_mcmc_bin_blocks=num_gfmc_bin_blocks,
        )
        logger.info("Final output(s):")
        logger.info(f"  Total Energy: E = {E_mean:.5f} +- {E_std:5f} Ha.")
        logger.info("")
        logger.info(f"Dump restart checkpoint file(s) to {restart_chk}.")
        logger.info("")

        # Save the checkpoint file for each process and zip them."""
        filename = f".{mpi_rank}_{restart_chk}"
        with open(filename, "wb") as f:
            pickle.dump(lrdmc, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Wait all MPI processes
        mpi_comm.Barrier()

        # Zip them.
        if mpi_rank == 0:
            filename_list = [f".{rank}_{restart_chk}" for rank in range(mpi_size)]
            with zipfile.ZipFile(restart_chk, "w", zipfile.ZIP_DEFLATED) as zipf:
                for filename in filename_list:
                    zipf.write(filename, arcname=filename.lstrip("."))
                    os.remove(filename)

        logger.info("")

    print_footer()


if __name__ == "__main__":
    cli()
