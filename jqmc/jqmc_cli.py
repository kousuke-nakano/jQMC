"""LRDMC module."""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# This file is part of phonopy.
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

import toml

# MPI
from mpi4py import MPI

from .lrdmc import GFMC

# jQMC module
from .miscs.header_footer import print_footer, print_header
from .vmc import VMC

# MPI related
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

rank = comm.Get_rank()

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


def main():
    logger_level = "MPI-INFO"

    log = getLogger("jqmc")

    if logger_level == "MPI-INFO":
        if rank == 0:
            log.setLevel("INFO")
            stream_handler = StreamHandler()
            stream_handler.setLevel("INFO")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
    else:
        log.setLevel(logger_level)
        stream_handler = StreamHandler()
        stream_handler.setLevel(logger_level)
        handler_format = Formatter(f"MPI-rank={rank}: %(name)s - %(levelname)s - %(lineno)d - %(message)s")
        stream_handler.setFormatter(handler_format)
        log.addHandler(stream_handler)

    print_header()

    if len(sys.argv) == 1:
        raise ValueError("Please specify input toml file.")
    elif len(sys.argv) > 2:
        raise ValueError("More than one input toml files are not acceptable.")
    else:
        toml_file = sys.argv[1]
        if not os.path.isfile(toml_file):
            raise FileNotFoundError(f"toml_file = {toml_file} does not exsit.")
        else:
            dict_toml = toml.load(open(toml_file))

    logger.info(f"Input file = {toml_file}")
    logger.info("")
    logger.info("Input parameters are")
    for section, dict_item in dict_toml.items():
        logger.info(f"**section:{section}**")
        for key, item in dict_item.items():
            logger.info(f"  {key}={item}")
        logger.info("")

    job_type = dict_toml["control"]["job_type"]
    mcmc_seed = dict_toml["control"]["mcmc_seed"]
    restart = dict_toml["control"]["restart"]
    if restart:
        restart_chk = dict_toml["control"]["restart_chk"]
        logger.info(f"restart = {restart}, restart_chk = {restart_chk}")
    else:
        hamiltonian_chk = dict_toml["control"]["hamiltonian_chk"]
        logger.info(f"restart = {restart}, hamiltonian_chk = {hamiltonian_chk}")

    # VMC!
    if job_type == "vmc":
        num_mcmc_steps = dict_toml["vmc"]["num_mcmc_steps"]
        num_mcmc_warmup_steps = dict_toml["vmc"]["num_mcmc_warmup_steps"]
        num_mcmc_bin_blocks = dict_toml["vmc"]["num_mcmc_bin_blocks"]

        with open(hamiltonian_chk, "rb") as f:
            hamiltonian_data = pickle.load(f)

            vmc = VMC(
                hamiltonian_data=hamiltonian_data,
                mcmc_seed=mcmc_seed,
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
                comput_position_deriv=False,
                comput_jas_param_deriv=False,
            )
            vmc.run_single_shot(num_mcmc_steps=num_mcmc_steps)
            e_L_mean, e_L_std = vmc.get_e_L()
            logger.info("Final output(s):")
            logger.info(f"  Total Energy: E = {e_L_mean:.5f} +- {e_L_std:5f} Ha.")
            logger.info("")

    # LRDMC!
    if job_type == "lrdmc":
        logger.info("***Lattice Regularized diffusion Monte Carlo***")
        num_branching = dict_toml["lrdmc"]["num_branching"]
        tau = dict_toml["lrdmc"]["tau"]
        alat = dict_toml["lrdmc"]["alat"]
        non_local_move = dict_toml["lrdmc"]["non_local_move"]
        num_gfmc_warmup_steps = dict_toml["lrdmc"]["num_gfmc_warmup_steps"]
        num_gfmc_bin_blocks = dict_toml["lrdmc"]["num_gfmc_bin_blocks"]
        num_gfmc_bin_collect = dict_toml["lrdmc"]["num_gfmc_bin_collect"]

        with open(hamiltonian_chk, "rb") as f:
            hamiltonian_data = pickle.load(f)

            gfmc = GFMC(
                hamiltonian_data=hamiltonian_data, mcmc_seed=mcmc_seed, tau=tau, alat=alat, non_local_move=non_local_move
            )
            gfmc.run(num_branching=num_branching)
            e_L_mean, e_L_std = gfmc.get_e_L(
                num_gfmc_warmup_steps=num_gfmc_warmup_steps,
                num_gfmc_bin_blocks=num_gfmc_bin_blocks,
                num_gfmc_bin_collect=num_gfmc_bin_collect,
            )
            logger.info("Final output(s):")
            logger.info(f"  Total Energy: E = {e_L_mean:.5f} +- {e_L_std:5f} Ha.")
            logger.info("")

    print_footer()


if __name__ == "__main__":
    main()
