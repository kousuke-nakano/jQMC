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
from logging import Formatter, StreamHandler, getLogger

import jax
import toml

# MPI
from mpi4py import MPI

from .lrdmc_vectorized import GFMC_multiple_walkers

# jQMC module
from .miscs.header_footer import print_footer, print_header
from .vmc_vectorized import VMC_multiple_walkers

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


def main():
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

    # print recognized XLA devices
    logger.info("*** XLA devices recognized by JAX***")
    logger.info(jax.devices())
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

    # job_type
    try:
        job_type = dict_toml["control"]["job_type"]
    except KeyError as e:
        logger.error("job_type should be specified.")
        raise ValueError from e
    # mcmc_seed
    try:
        mcmc_seed = dict_toml["control"]["mcmc_seed"]
    except KeyError:
        mcmc_seed = 34456
        logger.info(f"The default value of mcmc_seed = {mcmc_seed}.")
    # number_of_walkers
    try:
        number_of_walkers = dict_toml["control"]["number_of_walkers"]
    except KeyError:
        number_of_walkers = 40
        logger.info(f"The default value of number_of_walkers = {number_of_walkers}.")
    # max_time
    try:
        max_time = dict_toml["control"]["max_time"]
    except KeyError:
        max_time = 86400
        logger.info(f"The default value of max_time = {max_time}.")
    logger.info(f"max_time = {max_time} sec.")
    # restart
    try:
        restart = dict_toml["control"]["restart"]
    except KeyError:
        restart = False
        logger.info(f"The default value of restart = {restart}.")
    if restart:
        restart_chk = dict_toml["control"]["restart_chk"]
        logger.info(f"restart = {restart}, restart_chk = {restart_chk}.")
    else:
        try:
            restart_chk = dict_toml["control"]["restart_chk"]
        except KeyError:
            restart_chk = "restart.chk"
            logger.info(f"The default value of restart_chk = {restart_chk}.")
        hamiltonian_chk = dict_toml["control"]["hamiltonian_chk"]
        logger.info(f"restart = {restart}, hamiltonian_chk = {hamiltonian_chk}, restart_chk = {restart_chk}.")

    logger.info("")

    # VMC!
    if job_type == "vmc":
        logger.info("***Variational Monte Carlo***")
        # num_mcmc_steps
        try:
            num_mcmc_steps = dict_toml["vmc"]["num_mcmc_steps"]
        except KeyError as e:
            logger.error("num_mcmc_steps should be specified.")
            raise KeyError from e
        # num_mcmc_warmup_steps
        try:
            num_mcmc_warmup_steps = dict_toml["vmc"]["num_mcmc_warmup_steps"]
        except KeyError:
            num_mcmc_warmup_steps = 0
            logger.warning(f"The default value of num_mcmc_warmup_steps = {num_mcmc_warmup_steps}.")
        # num_mcmc_bin_blocks
        try:
            num_mcmc_bin_blocks = dict_toml["vmc"]["num_mcmc_bin_blocks"]
        except KeyError:
            num_mcmc_bin_blocks = 1
            logger.warning(f"The default value of num_mcmc_bin_blocks = {num_mcmc_bin_blocks}.")
        # check num_mcmc_steps, num_mcmc_warmup_steps, num_mcmc_bin_blocks
        if num_mcmc_steps < num_mcmc_warmup_steps:
            raise ValueError("num_mcmc_steps should be larger than num_mcmc_warmup_steps")
        if num_mcmc_steps - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
            raise ValueError("(num_mcmc_steps - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            if mpi_rank == 0:
                with open(restart_chk, "rb") as f:
                    chk_dyad_list = pickle.load(f)
                vmc = [chk for _, chk in chk_dyad_list]
            else:
                vmc = None
            vmc = mpi_comm.scatter(vmc, root=0)

        else:
            with open(hamiltonian_chk, "rb") as f:
                hamiltonian_data = pickle.load(f)

                vmc = VMC_multiple_walkers(
                    hamiltonian_data=hamiltonian_data,
                    mcmc_seed=mcmc_seed,
                    num_walkers=number_of_walkers,
                    comput_position_deriv=False,
                    comput_jas_param_deriv=False,
                )
        vmc.run_single_shot(num_mcmc_steps=num_mcmc_steps, max_time=max_time)
        e_L_mean, e_L_std = vmc.get_e_L(
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            num_mcmc_bin_blocks=num_mcmc_bin_blocks,
        )

        logger.info("Final output(s):")
        logger.info(f"  Total Energy: E = {e_L_mean:.5f} +- {e_L_std:5f} Ha.")
        logger.info("")

        logger.info(f"Dump restart checkpoint file(s) to {restart_chk}.")
        chk_dyad_list = [(mpi_rank, vmc)]
        chk_dyad_list = mpi_comm.reduce(chk_dyad_list, op=MPI.SUM, root=0)
        if mpi_rank == 0:
            with open(restart_chk, "wb") as f:
                pickle.dump(chk_dyad_list, f)
        logger.info("")

    # VMCopt!
    if job_type == "vmcopt":
        logger.info("***WF optimization with Variational Monte Carlo***")
        # num_mcmc_steps
        try:
            num_mcmc_steps = dict_toml["vmcopt"]["num_mcmc_steps"]
        except KeyError as e:
            logger.error("num_mcmc_steps should be specified.")
            raise KeyError from e
        # num_opt_steps
        try:
            num_opt_steps = dict_toml["vmcopt"]["num_opt_steps"]
        except KeyError as e:
            logger.error("num_opt_steps should be specified.")
            raise KeyError from e
        # delta
        try:
            delta = dict_toml["vmcopt"]["delta"]
        except KeyError:
            delta = 0.01
            logger.warning(f"The default value of delta = {delta}.")
        # epsilon
        try:
            epsilon = dict_toml["vmcopt"]["epsilon"]
        except KeyError:
            epsilon = 0.001
            logger.warning(f"The default value of epsilon = {epsilon}.")
        # num_mcmc_warmup_steps
        try:
            num_mcmc_warmup_steps = dict_toml["vmcopt"]["num_mcmc_warmup_steps"]
        except KeyError:
            num_mcmc_warmup_steps = 0
            logger.warning(f"The default value of num_mcmc_warmup_steps = {num_mcmc_warmup_steps}.")
        # wf_dump_freq
        try:
            wf_dump_freq = dict_toml["vmcopt"]["wf_dump_freq"]
        except KeyError:
            wf_dump_freq = 1
            logger.warning(f"The default value of wf_dump_freq = {wf_dump_freq}.")
        # num_mcmc_bin_blocks
        try:
            num_mcmc_bin_blocks = dict_toml["vmcopt"]["num_mcmc_bin_blocks"]
        except KeyError:
            num_mcmc_bin_blocks = 1
            logger.warning(f"The default value of num_mcmc_bin_blocks = {num_mcmc_bin_blocks}.")
        # check num_mcmc_steps, num_mcmc_warmup_steps, num_mcmc_bin_blocks
        if num_mcmc_steps < num_mcmc_warmup_steps:
            raise ValueError("num_mcmc_steps should be larger than num_mcmc_warmup_steps")
        if num_mcmc_steps - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
            raise ValueError("(num_mcmc_steps - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            if mpi_rank == 0:
                with open(restart_chk, "rb") as f:
                    chk_dyad_list = pickle.load(f)
                vmc = [chk for _, chk in chk_dyad_list]
            else:
                vmc = None
            vmc = mpi_comm.scatter(vmc, root=0)

        else:
            with open(hamiltonian_chk, "rb") as f:
                hamiltonian_data = pickle.load(f)

                vmc = VMC_multiple_walkers(
                    hamiltonian_data=hamiltonian_data,
                    mcmc_seed=mcmc_seed,
                    num_walkers=number_of_walkers,
                    comput_position_deriv=False,
                    comput_jas_param_deriv=True,
                )
        vmc.run_optimize(
            num_mcmc_steps=num_mcmc_steps,
            num_opt_steps=num_opt_steps,
            delta=delta,
            var_epsilon=epsilon,
            wf_dump_freq=wf_dump_freq,
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            max_time=max_time,
        )
        logger.info("")

        logger.info(f"Dump restart checkpoint file(s) to {restart_chk}.")
        chk_dyad_list = [(mpi_rank, vmc)]
        chk_dyad_list = mpi_comm.reduce(chk_dyad_list, op=MPI.SUM, root=0)
        if mpi_rank == 0:
            with open(restart_chk, "wb") as f:
                pickle.dump(chk_dyad_list, f)
        logger.info("")

    # LRDMC!
    if job_type == "lrdmc":
        logger.info("***Lattice Regularized diffusion Monte Carlo***")
        # num_branching
        try:
            num_branching = dict_toml["lrdmc"]["num_branching"]
        except KeyError as e:
            logger.error("num_branching should be specified.")
            raise ValueError from e
        # tau
        try:
            tau = dict_toml["lrdmc"]["tau"]
        except KeyError:
            tau = 0.01
            logger.warning(f"The default value of tau = {tau}.")
        # alat
        try:
            alat = dict_toml["lrdmc"]["alat"]
        except KeyError as e:
            logger.error("num_branching should be specified.")
            raise ValueError from e
        # non_local_move
        try:
            non_local_move = dict_toml["lrdmc"]["non_local_move"]
        except KeyError:
            non_local_move = "tmove"
            logger.warning(f"The default value of non_local_move = {non_local_move}.")
        # num_gmfc_warmup_steps:
        try:
            num_gfmc_warmup_steps = dict_toml["lrdmc"]["num_gfmc_warmup_steps"]
        except KeyError:
            num_gfmc_warmup_steps = 0
            logger.warning(f"The default value of num_gfmc_warmup_steps = {num_gfmc_warmup_steps}.")
        # num_gmfc_bin_blocks
        try:
            num_gfmc_bin_blocks = dict_toml["lrdmc"]["num_gfmc_bin_blocks"]
        except KeyError:
            num_gfmc_bin_blocks = 1
            logger.warning(f"The default value of num_gfmc_bin_blocks = {num_gfmc_bin_blocks}.")
        # num_gfmc_bin_collect
        try:
            num_gfmc_bin_collect = dict_toml["lrdmc"]["num_gfmc_bin_collect"]
        except KeyError:
            num_gfmc_bin_collect = 0
            logger.warning(f"The default value of num_gfmc_bin_collect = {num_gfmc_bin_collect}.")
        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if num_branching < num_gfmc_warmup_steps:
            raise ValueError("num_branching should be larger than num_gfmc_warmup_steps")
        if num_branching - num_gfmc_warmup_steps < num_gfmc_bin_blocks:
            raise ValueError("(num_branching - num_gfmc_warmup_steps) should be larger than num_gfmc_bin_blocks.")
        if num_gfmc_bin_blocks < num_gfmc_bin_collect:
            raise ValueError("num_gfmc_bin_blocks should be larger than num_gfmc_bin_collect.")

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            if mpi_rank == 0:
                with open(restart_chk, "rb") as f:
                    chk_dyad_list = pickle.load(f)
                gfmc = [chk for _, chk in chk_dyad_list]
            else:
                gfmc = None
            gfmc = mpi_comm.scatter(gfmc, root=0)

        else:
            with open(hamiltonian_chk, "rb") as f:
                hamiltonian_data = pickle.load(f)

                gfmc = GFMC_multiple_walkers(
                    hamiltonian_data=hamiltonian_data,
                    mcmc_seed=mcmc_seed,
                    num_walkers=number_of_walkers,
                    tau=tau,
                    alat=alat,
                    non_local_move=non_local_move,
                )
        gfmc.run(num_branching=num_branching, max_time=max_time)
        e_L_mean, e_L_std = gfmc.get_e_L(
            num_gfmc_warmup_steps=num_gfmc_warmup_steps,
            num_gfmc_bin_blocks=num_gfmc_bin_blocks,
            num_gfmc_bin_collect=num_gfmc_bin_collect,
        )
        logger.info("Final output(s):")
        logger.info(f"  Total Energy: E = {e_L_mean:.5f} +- {e_L_std:5f} Ha.")
        logger.info("")
        logger.info(f"Dump restart checkpoint file(s) to {restart_chk}.")
        chk_dyad_list = [(mpi_rank, gfmc)]
        chk_dyad_list = mpi_comm.reduce(chk_dyad_list, op=MPI.SUM, root=0)
        if mpi_rank == 0:
            with open(restart_chk, "wb") as f:
                pickle.dump(chk_dyad_list, f)
        logger.info("")

    print_footer()


if __name__ == "__main__":
    main()
    main()
    main()
