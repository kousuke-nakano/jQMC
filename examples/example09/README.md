# example09

Total energy of water molecule. One can learn how to obtain the VMC and DMC (in the extrapolated limit) energies of the Water dimer, starting from scratch (i.e., DFT calculation), with cartesian GTOs.

## Generate a trial WF

The first step of ab-initio QMC is to generate a trial WF by a mean-field theory such as DFT/HF. `jQMC` interfaces with other DFT/HF software packages via `TREX-IO`.

One of the easiest ways to produce it is using `pySCF` as a converter to the `TREX-IO` format is implemented.

The following is a script to run a HF calculation for the water molecule using `pyscf-forge` and dump it as a `TREX-IO` file.

> [!NOTE]
> This `TREX-IO` converter is being develped in the `pySCF-forge` [repository](https://github.com/pyscf/pyscf-forge) and not yet merged to the main repository of `pySCF`. Please use `pySCF-forge`.

<!-- include: 01DFT/run_pyscf.py -->
```python
from pyscf import gto, scf
from pyscf.tools import trexio

filename = "water_ccecp_ccpvtz.h5"

mol = gto.Mole()
mol.verbose = 5
mol.atom = """
               O    5.00000000   7.14707700   7.65097100
               H    4.06806600   6.94297500   7.56376100
               H    5.38023700   6.89696300   6.80798400
               """
mol.basis = "ccecp-ccpvtz"
mol.unit = "A"
mol.ecp = "ccecp"
mol.charge = 0
mol.spin = 0
mol.symmetry = False
mol.cart = True
mol.output = "water.out"
mol.build()

mf = scf.HF(mol)
mf.max_cycle = 200
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
```

Launch it on a terminal. You may get `E = -16.9450309201805 Ha` [Hartree-Forck].

```bash
% python run_pyscf.py
```

Next step is to convert the `TREXIO` file to the `jqmc` format using `jqmc-tool`

```bash
% jqmc-tool trexio convert-to water_ccecp_ccpvtz.h5 -j2 1.0 -j3 none -j-nn-type schnet -jp hidden_dim=16 -jp num_layers=3 -jp num_rbf=8
> Hamiltonian data is saved in hamiltonian_data.h5.
```

The generated `hamiltonian_data.chk` is a wavefunction file with the `jqmc` format. `-j2` specifies the initial value of the two-body Jastrow parameter and `-j3` specifies the basis set (`ao`:atomic orbital or `mo`:molecular orbital) for the three-body Jastrow part (here is none), and `-j-nn-type` specifies the type of neural-network Jastrow factor.

### Neural Network Jastrow (SchNet-type)

The `schnet` option for `-j-nn-type` enables a neural-network-based many-body Jastrow factor. This implementation is heavily inspired by the PauliNet architecture (specifically the Jastrow factor part described in [Hermann et al., Nature Chemistry 12, 891–897 (2020)](https://www.nature.com/articles/s41557-020-0544-y)). It uses a graph neural network to capture complex electron-electron and electron-nucleus correlations. For more implementation details, please refer to the `NNJastrow` class and `Jastrow_NN_data` dataclass in `jqmc/jastrow_factor.py`.

### Hyperparameters

The hyperparameters specified via `-jp` (e.g., `-jp hidden_dim=16`) control the capacity and computational cost of the neural network:

*   `hidden_dim` (default: 64): The size of the feature vectors (embeddings) for electrons and nuclei. Larger values allow the network to learn more complex representations but increase computational cost.
*   `num_layers` (default: 3): The number of message-passing interaction blocks. More layers allow information to propagate further across the molecular graph (i.e., capturing higher-order correlations), but make the network deeper and more expensive to evaluate.
*   `num_rbf` (default: 32): The number of radial basis functions used to expand inter-particle distances. A larger number provides higher resolution for spatial features.
*   `cutoff` (default: 5.0): The cutoff distance (in Bohr) for the radial basis functions. Interactions beyond this distance are smoothly decayed to zero.

## Optimize a trial WF (VMC)
The next step is to optimize variational parameters included in the generated wavefunction. More in details, here, we optimize the two-body Jastrow parameter and the matrix elements of the three-body Jastrow parameter.

You can generate a template file for a VMCopt calculation using `jqmc-tool`. Please directly edit `vmc.toml` if you want to change a parameter.

```bash
% jqmc-tool vmc generate-input -g
> Input file is generated: vmc.toml
```

<!-- include: 03vmc_JNNSD/vmc.toml -->
```toml
[control]
job_type = "vmc" # Specify the job type. "mcmc", "vmc", "lrdmc", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 42000 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low", "high", "devel", "mpi-low", "mpi-high", "mpi-devel"

[vmc]
num_mcmc_steps = 100 # Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates per measurement. Every local energy and other observeables are measured every this steps.
num_mcmc_warmup_steps = 0 # Number of observable measurement steps for warmup (i.e., discarged).
num_mcmc_bin_blocks = 1 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
num_opt_steps = 500 # Number of optimization steps.
wf_dump_freq = 50 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
opt_J1_param = false
opt_J2_param = true
opt_J3_param = false
opt_JNN_param = true
opt_lambda_param = false
num_param_opt = 0 # the number of parameters to optimize in the descending order of |f|/|std f|. If it is set 0, all parameters are optimized.
optimizer_kwargs = { method = "adam" } # Optimizer backend used for parameter updates. "sr" keeps the stochastic reconfiguration scheme; any other value should be the name of an optax optimizer (e.g., "adam").
```

Please lunch the job.

```bash
% jqmc vmc.toml > out_vmc 2> out_vmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc vmc.toml > out_vmc 2> out_vmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc vmc.toml > out_vmc 2> out_vmc.e # w/ MPI on GPU, depending the queueing system.
```

You can see the outcome using `jqmc-tool`.

```bash
% jqmc-tool vmc analyze-output out_vmc

------------------------------------------------------
Iter     E (Ha)     Max f (Ha)   Max of signal to noise of f
------------------------------------------------------
   1  -16.681(17)  +1.040(19)    82.811
   2  -16.619(16)  +0.901(19)    78.463
   3  -16.639(15)  +0.915(18)    79.656
   4  -16.694(15)  +0.901(18)    80.446
   5  -16.724(15)  +0.860(17)    76.196
   6  -16.753(15)  +0.839(17)    83.615
   7  -16.735(15)  +0.828(18)    73.593
   ...
------------------------------------------------------
```

The important criteria are `Max f` and `Max of signal to noise of f`. `Max f` should be zero within the error bar. A practical criterion for the `signal to noise` is < 4~5 because it means that all the residual forces are zero in the statistical sense.

You can also plot them and make a figure.

```bash
% jqmc-tool vmc analyze-output out_vmc -p -s vmc.jpg
```

![VMC optimization](03vmc_JNNSD/vmc.jpg)

If the optimization is not converged. You can restart the optimization.

```toml:vmc.toml
[control]
...
restart = true
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
...
```

```bash
% jqmc vmc.toml > out_vmc_cont 2> out_vmc_cont.e
```

You can see and plot the outcome using `jqmc-tool`.

```bash
% jqmc-tool vmc analyze-output out_vmc out_vmc_cont
```

## Compute Energy and Forces (MCMC)
The next step is MCMC calculation. You can generate a template file for a MCMC calculation using `jqmc-tool`. Please directly edit `mcmc.toml` if you want to change a parameter.

```bash
% jqmc-tool mcmc generate-input -g
> Input file is generated: mcmc.toml
```

<!-- include: 04mcmc_JNNSD/mcmc.toml -->
```toml
[control]
job_type = "mcmc" # Specify the job type. "mcmc", "vmc", or "lrdmc"
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 300 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"
[mcmc]
num_mcmc_steps = 90000 # Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates per measurement. Every local energy and other observeables are measured every this steps.
num_mcmc_warmup_steps = 0 # Number of observable measurement steps for warmup (i.e., discarged).
num_mcmc_bin_blocks = 5 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
atomic_force = true
```

The final step is to run the `jqmc` job w/ or w/o MPI on a CPU or GPU machine (via a job queueing system such as PBS).

```bash
% jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on GPU, depending the queueing system.
```

You may get `E = -16.97202 +- 0.000288 Ha` and `Var(E) = 1.99127 +- 0.000901 Ha^2` [VMC w/ NN Jastrow factors]. It depends on your hyperparameter choice.
