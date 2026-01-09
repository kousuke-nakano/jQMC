"""jQMC miscs."""

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

cli_parameters = {
    "control": {
        "job_type": None,
        "mcmc_seed": 34456,
        "number_of_walkers": 4,
        "max_time": 86400,
        "restart": False,
        "restart_chk": "restart.chk",
        "hamiltonian_h5": "hamiltonian_data.h5",
        "verbosity": "low",
    },
    "control_comments": {
        "job_type": 'Specify the job type. "mcmc", "vmc", "lrdmc", or "lrdmc-tau".',
        "mcmc_seed": "Random seed for MCMC",
        "number_of_walkers": "Number of walkers per MPI process",
        "max_time": "Maximum time in sec.",
        "restart": "Restart calculation. A checkpoint file should be specified.",
        "restart_chk": "Restart checkpoint file. If restart is True, this file is used.",
        "hamiltonian_h5": "Hamiltonian checkpoint file. If restart is False, this file is used.",
        "verbosity": 'Verbosity level. "low", "high", "devel", "mpi-low", "mpi-high", "mpi-devel"',
    },
    "mcmc": {
        "num_mcmc_steps": None,
        "num_mcmc_per_measurement": 40,
        "num_mcmc_warmup_steps": 0,
        "num_mcmc_bin_blocks": 1,
        "Dt": 2.0,
        "epsilon_AS": 0.0,
        "atomic_force": False,
        "parameter_derivatives": False,
    },
    "mcmc_comments": {
        "num_mcmc_steps": "Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.",
        "num_mcmc_per_measurement": "Number of MCMC updates per measurement. Every local energy and other observeables are measured every this steps.",
        "num_mcmc_warmup_steps": "Number of observable measurement steps for warmup (i.e., discarged).",
        "num_mcmc_bin_blocks": "Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.",
        "Dt": "Step size for the MCMC update (bohr).",
        "epsilon_AS": "the epsilon parameter used in the Attacalite-Sandro regulatization method.",
        "atomic_force": "If true, compute atomic forces.",
        "parameter_derivatives": "If true, compute parameter derivatives.",
    },
    "vmc": {
        "num_mcmc_steps": None,
        "num_mcmc_per_measurement": 40,
        "num_mcmc_warmup_steps": 0,
        "num_mcmc_bin_blocks": 1,
        "Dt": 2.0,
        "epsilon_AS": 0.0,
        "num_opt_steps": None,
        "wf_dump_freq": 1,
        "opt_J1_param": False,
        "opt_J2_param": True,
        "opt_J3_param": True,
        "opt_JNN_param": True,
        "opt_lambda_param": False,
        "num_param_opt": 0,
        "optimizer_kwargs": {
            "method": "sr",
            "delta": 0.01,
            "epsilon": 0.001,
            "cg_flag": True,
            "cg_max_iter": 10000,
            "cg_tol": 1e-4,
        },
    },
    "vmc_comments": {
        "num_mcmc_steps": "Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.",
        "num_mcmc_per_measurement": "Number of MCMC updates per measurement. Every local energy and other observeables are measured every this steps.",
        "num_mcmc_warmup_steps": "Number of observable measurement steps for warmup (i.e., discarged).",
        "num_mcmc_bin_blocks": "Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.",
        "Dt": "Step size for the MCMC update (bohr).",
        "epsilon_AS": "the epsilon parameter used in the Attacalite-Sandro regulatization method.",
        "num_opt_steps": "Number of optimization steps.",
        "wf_dump_freq": "Frequency of wavefunction (i.e. hamiltonian_data) dump.",
        "opt_J1_param": "Optimize the J1 parameter",
        "opt_J2_param": "Optimize the J2 parameter.",
        "opt_J3_param": "Optimize the J3 parameters.",
        "opt_JNN_param": "Optimize the neural-network Jastrow parameters.",
        "opt_lambda_param": "Optimize the lambda parameters in the geminal part.",
        "num_param_opt": "the number of parameters to optimize in the descending order of |f|/|std f|. If it is set 0, all parameters are optimized.",
        "optimizer_kwargs": (
            "Optimizer configuration. Set 'method' to 'sr' (default) for stochastic reconfiguration or to any "
            "optax optimizer name (e.g., 'adam'). For SR, keep the 'delta' (prefactor in c_i <- c_i + delta * "
            "S^{-1} f), 'epsilon' (regularization strength added to S), and optional 'cg_flag'/'cg_max_iter'/'cg_tol' "
            "entries controlling the conjugate-gradient solver. For optax optimizers, the remaining keys are passed "
            "directly to optax (e.g., {method = 'adam', learning_rate = 1e-3})."
        ),
    },
    "lrdmc": {
        "num_mcmc_steps": None,
        "num_mcmc_per_measurement": 40,
        "alat": 0.30,
        "non_local_move": "tmove",
        "num_gfmc_warmup_steps": 0,
        "num_gfmc_bin_blocks": 1,
        "num_gfmc_collect_steps": 0,
        "E_scf": 0.0,
        "atomic_force": False,
    },
    "lrdmc_comments": {
        "num_mcmc_steps": "Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.",
        "num_mcmc_per_measurement": "Number of GFMC projections per measurement. Every local energy and other observeables are measured every this projection.",
        "alat": "The lattice discretization parameter (i.e. grid size) used for discretized the Hamiltonian and potential. The lattice spacing is alat * a0, where a0 is the Bohr radius.",
        "non_local_move": "The treatment of the non-local term in the Effective core potential. tmove (T-move) and dltmove (Determinant locality approximation with T-move) are available.",
        "num_gfmc_warmup_steps": "Number of observable measurement steps for warmup (i.e., discarged).",
        "num_gfmc_bin_blocks": "Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_gfmc_bin_blocks, not num_gfmc_bin_blocks * mpi_size * number_of_walkers.",
        "num_gfmc_collect_steps": "Number of measurement (before binning) for collecting the weights.",
        "E_scf": "The initial guess of the total energy. This is used to compute the initial energy shift in the GFMC.",
        "atomic_force": "If true, compute atomic forces.",
    },
    "lrdmc-tau": {
        "num_mcmc_steps": None,
        "tau": 0.10,
        "alat": 0.30,
        "non_local_move": "tmove",
        "num_gfmc_warmup_steps": 0,
        "num_gfmc_bin_blocks": 1,
        "num_gfmc_collect_steps": 0,
    },
    "lrdmc-tau_comments": {
        "num_mcmc_steps": "Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.",
        "tau": "the imaginary time step size between projections (bohr).",
        "alat": "The lattice discretization parameter (i.e. grid size) used for discretized the Hamiltonian and potential. The lattice spacing is alat * a0, where a0 is the Bohr radius.",
        "non_local_move": "The treatment of the non-local term in the Effective core potential. tmove (T-move) and dltmove (Determinant locality approximation with T-move) are available.",
        "num_gfmc_warmup_steps": "Number of observable measurement steps for warmup (i.e., discarged).",
        "num_gfmc_bin_blocks": "Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_gfmc_bin_blocks, not num_gfmc_bin_blocks * mpi_size * number_of_walkers.",
        "num_gfmc_collect_steps": "Number of measurement (before binning) for collecting the weights.",
    },
}
