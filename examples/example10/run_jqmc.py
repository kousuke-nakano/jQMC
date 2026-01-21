import logging
import os
import sys

import jax
import numpy as np

from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_two_body_data,
)
from jqmc.jqmc_gfmc import GFMC_n
from jqmc.jqmc_mcmc import MCMC
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("jqmc")
logger.setLevel(logging.INFO)

# JAX configuration
jax.config.update("jax_enable_x64", True)

# =============================================================================
# 0. Settings
# =============================================================================
trexio_file = "water_ccecp_ccpvtz.h5"
hamiltonian_file = "hamiltonian_data.h5"

if not os.path.exists(trexio_file):
    logger.error(f"{trexio_file} not found. Please run run_pyscf.py first.")
    sys.exit(1)

# =============================================================================
# 1. Convert TREXIO to Hamiltonian
# =============================================================================
logger.info(f"Converting {trexio_file} to {hamiltonian_file}...")

(structure_data, aos_data, mos_data, _, geminal_data, coulomb_potential_data) = read_trexio_file(trexio_file, store_tuple=True)

# Jastrow 1-body
if coulomb_potential_data.ecp_flag:
    core_electrons = coulomb_potential_data.z_cores
else:
    core_electrons = [0] * structure_data.natom

jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
    jastrow_1b_param=1.0, structure_data=structure_data, core_electrons=core_electrons
)

# Jastrow 2-body
jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

# Jastrow 3-body (None for this example)
jastrow_threebody_data = None

# Jastrow NN (None for this example)
nn_jastrow_data = None

jastrow_data = Jastrow_data(
    jastrow_one_body_data=jastrow_onebody_data,
    jastrow_two_body_data=jastrow_twobody_data,
    jastrow_three_body_data=jastrow_threebody_data,
    jastrow_nn_data=nn_jastrow_data,
)

wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)

hamiltonian_data = Hamiltonian_data(
    structure_data=structure_data,
    coulomb_potential_data=coulomb_potential_data,
    wavefunction_data=wavefunction_data,
)

hamiltonian_data.save_to_hdf5(hamiltonian_file)
logger.info(f"Saved Hamiltonian data to {hamiltonian_file}")


# =============================================================================
# 2. VMC Optimization
# =============================================================================
logger.info("Starting VMC Optimization...")

mcmc_opt = MCMC(
    hamiltonian_data=hamiltonian_data,
    Dt=0.01,
    mcmc_seed=42,
    num_walkers=2000,
    num_mcmc_per_measurement=1,
    epsilon_AS=0.0,
    comput_position_deriv=False,
    comput_param_deriv=True,
)

mcmc_opt.run_optimize(
    num_mcmc_steps=100,  # Steps per optimization step
    num_opt_steps=20,
    wf_dump_freq=10,
    num_mcmc_warmup_steps=10,
    num_mcmc_bin_blocks=5,
    opt_J1_param=True,
    opt_J2_param=True,
    opt_J3_param=False,
    opt_JNN_param=False,
    opt_lambda_param=False,
    num_param_opt=1,
    max_time=3600,
    optimizer_kwargs={"method": "sr", "lr": 0.01},
)

optimized_hamiltonian = mcmc_opt.hamiltonian_data


# =============================================================================
# 3. VMC Sampling
# =============================================================================
logger.info("Starting VMC Sampling...")

mcmc_prod = MCMC(
    hamiltonian_data=optimized_hamiltonian,
    Dt=0.01,
    mcmc_seed=123,
    num_walkers=2000,
    num_mcmc_per_measurement=1,
    epsilon_AS=0.0,
    comput_position_deriv=False,
    comput_param_deriv=False,
)

num_vmc_steps = 500
mcmc_prod.run(num_mcmc_steps=num_vmc_steps, max_time=3600)

vmc_E_mean, vmc_E_std, _, _ = mcmc_prod.get_E(
    num_mcmc_warmup_steps=int(num_vmc_steps * 0.1),
    num_mcmc_bin_blocks=10,
)

logger.info(f"VMC Energy: {vmc_E_mean:.5f} +/- {vmc_E_std:.5f} Ha")


# =============================================================================
# 4. LRDMC with multiple alats
# =============================================================================
alats = [0.5, 0.4, 0.3]
lrdmc_results = []

for alat in alats:
    logger.info(f"Starting LRDMC with alat={alat}...")

    lrdmc = GFMC_n(
        hamiltonian_data=optimized_hamiltonian,
        num_walkers=2000,
        num_mcmc_per_measurement=1,
        num_gfmc_collect_steps=1,
        mcmc_seed=456,
        E_scf=vmc_E_mean,  # Approximate energy for branching
        alat=alat,
        non_local_move=True,
        comput_position_deriv=False,
    )

    num_lrdmc_steps = 500
    lrdmc.run(num_mcmc_steps=num_lrdmc_steps, max_time=3600)

    lrdmc_E_mean, lrdmc_E_std, _, _ = lrdmc.get_E(
        num_mcmc_warmup_steps=int(num_lrdmc_steps * 0.1),
        num_mcmc_bin_blocks=10,
    )

    logger.info(f"LRDMC Energy (alat={alat}): {lrdmc_E_mean:.5f} +/- {lrdmc_E_std:.5f} Ha")
    lrdmc_results.append((alat, lrdmc_E_mean, lrdmc_E_std))


# =============================================================================
# 5. Extrapolation
# =============================================================================
# Simple linear regression E(a) = E_0 + k * a^2 (usually LRDMC error scales as a^2)

x = np.array([r[0] ** 2 for r in lrdmc_results])
y = np.array([r[1] for r in lrdmc_results])
w = np.array([1.0 / r[2] ** 2 for r in lrdmc_results])  # weights

# Weighted linear regression
# y = c0 + c1 * x
A = np.vstack([x, np.ones(len(x))]).T
W = np.diag(w)
c1, c0 = np.linalg.inv(A.T @ W @ A) @ (A.T @ W @ y)

logger.info("-" * 40)
logger.info("Extrapolation Results (E vs a^2):")
for a, E, err in lrdmc_results:
    logger.info(f"alat={a:.2f}, a^2={a**2:.4f}, E={E:.5f} +/- {err:.5f}")

logger.info(f"Extrapolated Energy (a->0): {c0:.5f} Ha")
logger.info("-" * 40)
