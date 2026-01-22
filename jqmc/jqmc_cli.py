"""command-line module."""

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

# python modules
import gzip
import os
import pickle
import sys
import zipfile
from logging import FileHandler, Formatter, StreamHandler, getLogger

import jax
import toml

# MPI
from mpi4py import MPI
from uncertainties import ufloat

from .hamiltonians import Hamiltonian_data

# jQMC
from .header_footer import _print_footer, _print_header
from .jqmc_gfmc import GFMC_n, GFMC_t
from .jqmc_mcmc import MCMC
from .jqmc_miscs import cli_parameters

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# set logger
logger = getLogger("jqmc").getChild(__name__)


def _cli():
    """Main function."""
    if len(sys.argv) == 1:
        raise ValueError("Please specify input toml file.")
    elif len(sys.argv) > 2:
        raise ValueError("More than one input toml files are not acceptable.")
    else:
        toml_file = sys.argv[1]
        if not os.path.isfile(toml_file):
            raise FileNotFoundError(f"toml_file = {toml_file} does not exist.")
        else:
            with open(toml_file, "r") as f:
                dict_toml = toml.load(f)

    # MPI related
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # set verbosity
    try:
        verbosity = dict_toml["control"]["verbosity"]
    except KeyError:
        verbosity = cli_parameters["control"]["verbosity"]

    # set logger level
    if verbosity == "devel":
        logger_level = "DEVEL"
    elif verbosity == "high":
        logger_level = "DEBUG"
    elif verbosity == "low":
        logger_level = "INFO"
    elif verbosity == "mpi-devel":
        logger_level = "MPI-DEVEL"
    elif verbosity == "mpi-high":
        logger_level = "MPI-DEBUG"
    elif verbosity == "mpi-low":
        logger_level = "MPI-INFO"
    else:
        logger_level = "INFO"

    log = getLogger("jqmc")
    if log.hasHandlers():
        log.handlers.clear()

    if logger_level == "INFO":
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
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
    elif logger_level == "DEBUG":
        if mpi_rank == 0:
            log.setLevel("DEBUG")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("DEBUG")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
        else:
            log.setLevel("ERROR")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("ERROR")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
    elif logger_level == "DEVEL":
        if mpi_rank == 0:
            log.setLevel("DEVEL")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("DEVEL")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
        else:
            log.setLevel("ERROR")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("ERROR")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
    elif logger_level == "MPI-INFO":
        log.setLevel("INFO")
        file_handler = FileHandler(f"log_jqmc_mpirank_{mpi_rank}.log", mode="w")
        file_handler.setLevel("INFO")
        formatter = Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    elif logger_level == "MPI-DEBUG":
        log.setLevel("DEBUG")
        file_handler = FileHandler(f"log_jqmc_mpirank_{mpi_rank}.log", mode="w")
        file_handler.setLevel("DEBUG")
        formatter = Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    elif logger_level == "MPI-DEVEL":
        log.setLevel("DEVEL")
        file_handler = FileHandler(f"log_jqmc_mpirank_{mpi_rank}.log", mode="w")
        file_handler.setLevel("DEVEL")
        formatter = Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    else:
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
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
    # print header
    _print_header()

    # jax-MPI related
    # Avoid JAX distributed init warnings/hangs due to proxy env vars.
    for _proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(_proxy_var, None)
    try:
        jax.distributed.initialize(cluster_detection_method="mpi4py")
        logger.info("JAX distributed initialization is successful.")
        logger.info(f"JAX backend = {jax.default_backend()}.")
        logger.info("")
        jax_distributed_is_initialized = True
    except Exception as e:
        logger.info("Running on CPUs or single GPU. JAX distributed initialization is skipped.")
        logger.debug(f"Distributed initialization Exception: {e}")
        logger.info("")
        jax_distributed_is_initialized = False

    if jax_distributed_is_initialized:
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
    hamiltonian_h5 = parameters["control"]["hamiltonian_h5"]

    # MCMC
    if job_type == "mcmc":
        logger.info("***Variational Monte Carlo***")

        # vmc section
        section = "mcmc"
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
        num_mcmc_steps = parameters[section]["num_mcmc_steps"]
        num_mcmc_per_measurement = parameters[section]["num_mcmc_per_measurement"]
        num_mcmc_warmup_steps = parameters[section]["num_mcmc_warmup_steps"]
        num_mcmc_bin_blocks = parameters[section]["num_mcmc_bin_blocks"]
        Dt = parameters[section]["Dt"]
        epsilon_AS = parameters[section]["epsilon_AS"]
        atomic_force = parameters[section]["atomic_force"]
        parameter_derivatives = parameters[section]["parameter_derivatives"]

        # check num_mcmc_steps, num_mcmc_warmup_steps, num_mcmc_bin_blocks
        if not restart:
            if num_mcmc_steps < num_mcmc_warmup_steps:
                raise ValueError("num_mcmc_steps should be larger than num_mcmc_warmup_steps")
            if num_mcmc_steps - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
                raise ValueError("(num_mcmc_steps - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            """Unzip the checkpoint file for each process and load them."""
            with zipfile.ZipFile(restart_chk, "r") as zf:
                arcname = f"{mpi_rank}.pkl.gz"
                with zf.open(arcname) as zipped_gz_fobj:
                    with gzip.open(zipped_gz_fobj, "rb") as gz:
                        mcmc = pickle.load(gz)
        else:
            hamiltonian_data = Hamiltonian_data.load_from_hdf5(hamiltonian_h5)

            mcmc = MCMC(
                hamiltonian_data=hamiltonian_data,
                Dt=Dt,
                mcmc_seed=mcmc_seed,
                num_walkers=number_of_walkers,
                num_mcmc_per_measurement=num_mcmc_per_measurement,
                epsilon_AS=epsilon_AS,
                comput_position_deriv=atomic_force,
                comput_param_deriv=parameter_derivatives,
            )
        mcmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)
        E_mean, E_std, Var_mean, Var_std = mcmc.get_E(
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            num_mcmc_bin_blocks=num_mcmc_bin_blocks,
        )
        if mcmc.comput_position_deriv:
            f_mean, f_std = mcmc.get_aF(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            )
        logger.info("Final output(s):")
        logger.info(f"  Total Energy: E = {E_mean:.5f} +- {E_std:5f} Ha.")
        logger.info(f"  Variance: Var = {Var_mean:.5f} +- {Var_std:5f} Ha^2.")
        if mcmc.comput_position_deriv:
            logger.info("  Atomic Forces:")
            sep = 16 * 3
            logger.info("  " + "-" * sep)
            logger.info("  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)")
            logger.info("  " + "-" * sep)
            for i in range(len(mcmc.hamiltonian_data.structure_data.atomic_labels)):
                atomic_label = str(mcmc.hamiltonian_data.structure_data.atomic_labels[i])
                row_values = [f"{ufloat(f_mean[i, j], f_std[i, j]):+2uS}" for j in range(3)]
                row_str = "  " + atomic_label.ljust(8) + " ".join(val.ljust(12) for val in row_values)
                logger.info(row_str)
            logger.info("  " + "-" * sep)
        if mcmc.comput_param_deriv:
            logger.info("  Parameter Derivatives are computed.")
        logger.info("")
        logger.info(f"Dump restart checkpoint file(s) to {restart_chk}.")
        logger.info("")

        # Save the checkpoint file for each process and zip them."""
        tmp_gz_filename = f".{mpi_rank}.pkl.gz"

        with gzip.open(tmp_gz_filename, "wb") as gz:
            pickle.dump(mcmc, gz, protocol=pickle.HIGHEST_PROTOCOL)

        mpi_comm.Barrier()

        if mpi_rank == 0:
            if os.path.exists(restart_chk):
                os.remove(restart_chk)

            with zipfile.ZipFile(restart_chk, "w", zipfile.ZIP_DEFLATED) as zipf:
                for mpi_rank in range(mpi_size):
                    gz_name = f".{mpi_rank}.pkl.gz"
                    arcname = gz_name.lstrip(".")
                    zipf.write(gz_name, arcname=arcname)
                    os.remove(gz_name)

        mpi_comm.Barrier()

    # VMC
    if job_type == "vmc":
        logger.info("***WF optimization with Variational Monte Carlo***")

        # vmcopt section
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

        logger.info("")

        # parameters
        num_mcmc_steps = parameters[section]["num_mcmc_steps"]
        num_mcmc_per_measurement = parameters[section]["num_mcmc_per_measurement"]
        num_mcmc_warmup_steps = parameters[section]["num_mcmc_warmup_steps"]
        num_mcmc_bin_blocks = parameters[section]["num_mcmc_bin_blocks"]
        Dt = parameters[section]["Dt"]
        epsilon_AS = parameters[section]["epsilon_AS"]
        num_opt_steps = parameters[section]["num_opt_steps"]
        wf_dump_freq = parameters[section]["wf_dump_freq"]
        opt_J1_param = parameters[section]["opt_J1_param"]
        opt_J2_param = parameters[section]["opt_J2_param"]
        opt_J3_param = parameters[section]["opt_J3_param"]
        opt_JNN_param = parameters[section]["opt_JNN_param"]
        opt_lambda_param = parameters[section]["opt_lambda_param"]
        num_param_opt = parameters[section]["num_param_opt"]
        optimizer_kwargs = parameters[section]["optimizer_kwargs"]
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        elif not isinstance(optimizer_kwargs, dict):
            raise TypeError("optimizer_kwargs must be a dictionary when provided in the VMC section.")
        else:
            optimizer_kwargs = dict(optimizer_kwargs)

        vmc_section = dict_toml.get(section, {}) if isinstance(dict_toml.get(section, {}), dict) else {}
        optimizer_method = vmc_section.get("optimizer")
        if optimizer_method is not None and not isinstance(optimizer_method, str):
            raise TypeError("The 'optimizer' key must be a string when provided in the VMC section.")
        if optimizer_method is not None:
            logger.warning("The 'optimizer' key is deprecated. Please move the value under optimizer_kwargs.method instead.")
            optimizer_kwargs.setdefault("method", optimizer_method)

        optimizer_kwargs.setdefault("method", "sr")
        if not isinstance(optimizer_kwargs["method"], str):
            raise TypeError("optimizer_kwargs['method'] must be a string when provided in the VMC section.")

        # check num_mcmc_steps, num_mcmc_warmup_steps, num_mcmc_bin_blocks
        if num_mcmc_steps < num_mcmc_warmup_steps:
            raise ValueError("num_mcmc_steps should be larger than num_mcmc_warmup_steps")
        if num_mcmc_steps - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
            raise ValueError("(num_mcmc_steps - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            """Unzip the checkpoint file for each process and load them."""
            with zipfile.ZipFile(restart_chk, "r") as zf:
                arcname = f"{mpi_rank}.pkl.gz"
                with zf.open(arcname) as zipped_gz_fobj:
                    with gzip.open(zipped_gz_fobj, "rb") as gz:
                        mcmc = pickle.load(gz)
        else:
            hamiltonian_data = Hamiltonian_data.load_from_hdf5(hamiltonian_h5)

            mcmc = MCMC(
                hamiltonian_data=hamiltonian_data,
                Dt=Dt,
                mcmc_seed=mcmc_seed,
                num_walkers=number_of_walkers,
                num_mcmc_per_measurement=num_mcmc_per_measurement,
                epsilon_AS=epsilon_AS,
                comput_position_deriv=False,
                comput_param_deriv=True,
            )
        mcmc.run_optimize(
            num_mcmc_steps=num_mcmc_steps,
            num_opt_steps=num_opt_steps,
            wf_dump_freq=wf_dump_freq,
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            opt_J1_param=opt_J1_param,
            opt_J2_param=opt_J2_param,
            opt_J3_param=opt_J3_param,
            opt_JNN_param=opt_JNN_param,
            opt_lambda_param=opt_lambda_param,
            num_param_opt=num_param_opt,
            max_time=max_time,
            optimizer_kwargs=optimizer_kwargs,
        )
        logger.info("")

        logger.info(f"Dump restart checkpoint file(s) to {restart_chk}.")

        logger.info("")

        # Save the checkpoint file for each process and zip them."""
        tmp_gz_filename = f".{mpi_rank}.pkl.gz"

        with gzip.open(tmp_gz_filename, "wb") as gz:
            pickle.dump(mcmc, gz, protocol=pickle.HIGHEST_PROTOCOL)

        mpi_comm.Barrier()

        if mpi_rank == 0:
            if os.path.exists(restart_chk):
                os.remove(restart_chk)

            with zipfile.ZipFile(restart_chk, "w", zipfile.ZIP_DEFLATED) as zipf:
                for mpi_rank in range(mpi_size):
                    gz_name = f".{mpi_rank}.pkl.gz"
                    arcname = gz_name.lstrip(".")
                    zipf.write(gz_name, arcname=arcname)
                    os.remove(gz_name)

        mpi_comm.Barrier()

    # LRDMC!
    if job_type == "lrdmc":
        logger.info("***Lattice Regularized diffusion Monte Carlo***")

        # lrdmc section
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
        num_mcmc_steps = parameters[section]["num_mcmc_steps"]
        num_mcmc_per_measurement = parameters[section]["num_mcmc_per_measurement"]
        alat = parameters[section]["alat"]
        non_local_move = parameters[section]["non_local_move"]
        num_gfmc_warmup_steps = parameters[section]["num_gfmc_warmup_steps"]
        num_gfmc_bin_blocks = parameters[section]["num_gfmc_bin_blocks"]
        num_gfmc_collect_steps = parameters[section]["num_gfmc_collect_steps"]
        E_scf = parameters[section]["E_scf"]
        atomic_force = parameters[section]["atomic_force"]

        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if not restart:
            if num_mcmc_steps < num_gfmc_warmup_steps:
                raise ValueError("num_mcmc_steps should be larger than num_gfmc_warmup_steps")
            if num_mcmc_steps - num_gfmc_warmup_steps < num_gfmc_bin_blocks:
                raise ValueError("(num_mcmc_steps - num_gfmc_warmup_steps) should be larger than num_gfmc_bin_blocks.")

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            """Unzip the checkpoint file for each process and load them."""
            with zipfile.ZipFile(restart_chk, "r") as zf:
                arcname = f"{mpi_rank}.pkl.gz"
                with zf.open(arcname) as zipped_gz_fobj:
                    with gzip.open(zipped_gz_fobj, "rb") as gz:
                        lrdmc = pickle.load(gz)
        else:
            hamiltonian_data = Hamiltonian_data.load_from_hdf5(hamiltonian_h5)

            lrdmc = GFMC_n(
                hamiltonian_data=hamiltonian_data,
                num_walkers=number_of_walkers,
                num_mcmc_per_measurement=num_mcmc_per_measurement,
                num_gfmc_collect_steps=num_gfmc_collect_steps,
                mcmc_seed=mcmc_seed,
                E_scf=E_scf,
                alat=alat,
                non_local_move=non_local_move,
                comput_position_deriv=atomic_force,
            )
        lrdmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)
        E_mean, E_std, Var_mean, Var_std = lrdmc.get_E(
            num_mcmc_warmup_steps=num_gfmc_warmup_steps,
            num_mcmc_bin_blocks=num_gfmc_bin_blocks,
        )
        if lrdmc.comput_position_deriv:
            f_mean, f_std = lrdmc.get_aF(
                num_mcmc_warmup_steps=num_gfmc_warmup_steps,
                num_mcmc_bin_blocks=num_gfmc_bin_blocks,
            )
        logger.info("Final output(s):")
        logger.info(f"  Total Energy: E = {E_mean:.5f} +- {E_std:5f} Ha.")
        logger.info(f"  Variance: Var = {Var_mean:.5f} +- {Var_std:5f} Ha^2.")
        if lrdmc.comput_position_deriv:
            logger.info("  Atomic Forces:")
            sep = 16 * 3
            logger.info("  " + "-" * sep)
            logger.info("  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)")
            logger.info("  " + "-" * sep)
            for i in range(len(lrdmc.hamiltonian_data.structure_data.atomic_labels)):
                atomic_label = str(lrdmc.hamiltonian_data.structure_data.atomic_labels[i])
                row_values = [f"{ufloat(f_mean[i, j], f_std[i, j]):+2uS}" for j in range(3)]
                row_str = "  " + atomic_label.ljust(8) + "".join(val.ljust(12) for val in row_values)
                logger.info(row_str)
            logger.info("  " + "-" * sep)
        logger.info("")
        logger.info(f"Dump restart checkpoint file(s) to {restart_chk}.")
        logger.info("")

        # Save the checkpoint file for each process and zip them."""
        tmp_gz_filename = f".{mpi_rank}.pkl.gz"

        with gzip.open(tmp_gz_filename, "wb") as gz:
            pickle.dump(lrdmc, gz, protocol=pickle.HIGHEST_PROTOCOL)

        mpi_comm.Barrier()

        if mpi_rank == 0:
            if os.path.exists(restart_chk):
                os.remove(restart_chk)

            with zipfile.ZipFile(restart_chk, "w", zipfile.ZIP_DEFLATED) as zipf:
                for mpi_rank in range(mpi_size):
                    gz_name = f".{mpi_rank}.pkl.gz"
                    arcname = gz_name.lstrip(".")
                    zipf.write(gz_name, arcname=arcname)
                    os.remove(gz_name)

        mpi_comm.Barrier()

    # LRDMC with fixed time!
    if job_type == "lrdmc-tau":
        logger.info("***Lattice Regularized diffusion Monte Carlo with a fixed projection time***")

        # lrdmc-tau section
        section = "lrdmc-tau"
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
        num_mcmc_steps = parameters[section]["num_mcmc_steps"]
        tau = parameters[section]["tau"]
        alat = parameters[section]["alat"]
        non_local_move = parameters[section]["non_local_move"]
        num_gfmc_warmup_steps = parameters[section]["num_gfmc_warmup_steps"]
        num_gfmc_bin_blocks = parameters[section]["num_gfmc_bin_blocks"]
        num_gfmc_collect_steps = parameters[section]["num_gfmc_collect_steps"]

        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if not restart:
            if num_mcmc_steps < num_gfmc_warmup_steps:
                raise ValueError("num_mcmc_steps should be larger than num_gfmc_warmup_steps")
            if num_mcmc_steps - num_gfmc_warmup_steps < num_gfmc_bin_blocks:
                raise ValueError("(num_mcmc_steps - num_gfmc_warmup_steps) should be larger than num_gfmc_bin_blocks.")

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            """Unzip the checkpoint file for each process and load them."""
            with zipfile.ZipFile(restart_chk, "r") as zf:
                arcname = f"{mpi_rank}.pkl.gz"
                with zf.open(arcname) as zipped_gz_fobj:
                    with gzip.open(zipped_gz_fobj, "rb") as gz:
                        lrdmc = pickle.load(gz)
        else:
            hamiltonian_data = Hamiltonian_data.load_from_hdf5(hamiltonian_h5)

            lrdmc = GFMC_t(
                hamiltonian_data=hamiltonian_data,
                num_walkers=number_of_walkers,
                tau=tau,
                num_gfmc_collect_steps=num_gfmc_collect_steps,
                mcmc_seed=mcmc_seed,
                alat=alat,
                non_local_move=non_local_move,
            )
        lrdmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)
        E_mean, E_std, Var_mean, Var_std = lrdmc.get_E(
            num_mcmc_warmup_steps=num_gfmc_warmup_steps,
            num_mcmc_bin_blocks=num_gfmc_bin_blocks,
        )
        logger.info("Final output(s):")
        logger.info(f"  Total Energy: E = {E_mean:.5f} +- {E_std:5f} Ha.")
        logger.info(f"  Variance: Var = {Var_mean:.5f} +- {Var_std:5f} Ha^2.")
        logger.info("")
        logger.info(f"Dump restart checkpoint file(s) to {restart_chk}.")
        logger.info("")

        # Save the checkpoint file for each process and zip them."""
        tmp_gz_filename = f".{mpi_rank}.pkl.gz"

        with gzip.open(tmp_gz_filename, "wb") as gz:
            pickle.dump(lrdmc, gz, protocol=pickle.HIGHEST_PROTOCOL)

        mpi_comm.Barrier()

        if mpi_rank == 0:
            if os.path.exists(restart_chk):
                os.remove(restart_chk)

            with zipfile.ZipFile(restart_chk, "w", zipfile.ZIP_DEFLATED) as zipf:
                for mpi_rank in range(mpi_size):
                    gz_name = f".{mpi_rank}.pkl.gz"
                    arcname = gz_name.lstrip(".")
                    zipf.write(gz_name, arcname=arcname)
                    os.remove(gz_name)

        mpi_comm.Barrier()

    _print_footer()


if __name__ == "__main__":
    _cli()
