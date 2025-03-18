# example01

Total energy of water molecule. One can learn how to obtain the VMC and DMC (in the extrapolated limit) energies of the Water dimer, starting from scratch (i.e., DFT calculation), with cartesian GTOs.

## Generate a trial WF

The first step of ab-initio QMC is to generate a trial WF by a mean-field theory such as DFT/HF. `jQMC` interfaces with other DFT/HF software packages via `TREXIO`.

One of the easiest ways to produce it is using `pySCF` as a converter to the `TREXIO` format is implemented. The following is a script to run a HF calculation of the water molecule and dump it as a `TREXIO` file.

```python:run_pyscf.py
from pyscf import gto, scf
from pyscf.tools import trexio

filename = 'water_ccecp_ccpvtz.h5'

mol = gto.Mole()
mol.verbose  = 5
mol.atom     = '''
               O    5.00000000   7.14707700   7.65097100
               H    4.06806600   6.94297500   7.56376100
               H    5.38023700   6.89696300   6.80798400
               '''
mol.basis    = 'ccecp-ccpvtz'
mol.unit     = 'A'
mol.ecp      = 'ccecp'
mol.charge   = 0
mol.spin     = 0
mol.symmetry = False
mol.cart = True
mol.output   = 'water.out'
mol.build()

mf = scf.HF(mol)
mf.max_cycle=200
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)

```

Launch it on a terminal. You may get `E = -16.9450309201805 Ha` [Hartree-Forck].

```bash
% python run_pyscf.py
```

Next step is to convert the `TREXIO` file to the `jqmc` format using `jqmc-tool`

```bash
% jqmc-tool trexio convert-to water_ccecp_ccpvtz.h5 -j2 1.0 -j3 mo
> Hamiltonian data is saved in hamiltonian_data.chk.
```

The generated `hamiltonian_data.chk` is a wavefunction file with the `jqmc` format. `-j2` specifies the initial value of the two-body Jastrow parameter and `-j3` specifies the basis set (`ao`:atomic orbital or `mo`:molecular orbital) for the three-body Jastrow part.

## Optimize a trial WF (VMCopt)
The next step is to optimize variational parameters included in the generated wavefunction. More in details, here, we optimize the two-body Jastrow parameter and the matrix elements of the three-body Jastrow parameter.

You can generate a template file for a VMCopt calculation using `jqmc-tool`. Please directly edit `vmcopt.toml` if you want to change a parameter.

```bash
% jqmc-tool vmcopt generate-input -g
> Input file is generated: vmcopt.toml
```

```toml:vmcopt.toml
[control]
job_type = "vmcopt" # Specify the job type. "vmc", "vmcopt", "lrdmc", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_chk = "hamiltonian_data.chk" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"

[vmcopt]
num_mcmc_steps = 500 # Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates per measurement. Every local energy and other observeables are measured every this steps.
num_mcmc_warmup_steps = 0 # Number of observable measurement steps for warmup (i.e., discarged).
num_mcmc_bin_blocks = 5 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
num_opt_steps = 300 # Number of optimization steps.
wf_dump_freq = 1 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
delta = 0.01 # Step size for the Stochastic reconfiguration (i.e., the natural gradient) optimization.
epsilon = 0.001 # Regularization parameter, a positive number added to the diagnoal elements of the Fisher-Information matrix, used during the Stochastic reconfiguration to improve the numerical stability.
opt_J1_param = false
opt_J2_param = true
opt_J3_param = true
opt_lambda_param = false
num_param_opt = 0 # the number of parameters to optimize in the descending order of |f|/|std f|. If None, all parameters are optimized.
```

Please lunch the job.

```bash
% jqmc vmcopt.toml > out_vmcopt 2> out_vmcopt.e # w/o MPI on CPU
% mpirun -np 4 jqmc vmcopt.toml > out_vmcopt 2> out_vmcopt.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc vmcopt.toml > out_vmcopt 2> out_vmcopt.e # w/ MPI on GPU, depending the queueing system.
```

You can see and plot the outcome using `jqmc-tool`.

```bash
% jqmc-tool vmcopt analyze-output out_vmcopt

------------------------------------------------------
Iter     E (Ha)     Max f (Ha)   Signal to Noise
------------------------------------------------------
   1  -16.5752(98)  +1.126(12)   114.338
   2  -16.6269(94)  +1.074(12)   111.055
   3  -16.6717(89)  +1.049(11)   107.720
   4  -16.6947(89)  +0.990(11)    96.205
   5  -16.7395(84)  +0.933(11)    98.656
   6  -16.7417(88)  +0.896(10)    98.756
   7  -16.8048(83)  +0.874(10)    89.545
   8  -16.8248(79)  +0.805(10)    79.823
   9  -16.8250(86)  +0.775(10)    80.669
  10  -16.8525(81)  +0.729(10)    76.278
  11  -16.8705(84)  +0.696(10)    71.823
  12  -16.8812(77)  +0.6400(90)    74.887
------------------------------------------------------

```

The important criteria are `f_max` and max of `signal-to-noise` of `f`. `f_max` should be zero within the error bar. A practical criterion for `signal-to-noise` is < 4~5 because it means that all the residual forces are zero in the statistical sense.

If the optimization is not converged. You can restart the optimization.

```toml:vmc.toml
[control]
...
restart = true
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
...
```

```bash
% jqmc vmcopt.toml > out_vmcopt_cont 2> out_vmcopt_cont.e
```

You can see and plot the outcome using `jqmc-tool`.

```bash
% jqmc-tool vmcopt analyze-output out_vmcopt out_vmcopt_cont
```

## Compute Energy (VMC)
The next step is VMC calculation. You can generate a template file for a VMC calculation using `jqmc-tool`. Please directly edit `vmc.toml` if you want to change a parameter.

```bash
% jqmc-tool vmc generate-input -g
> Input file is generated: vmc.toml
```

```toml:vmc.toml
[control]
job_type = "vmc" # Specify the job type. "vmc", "vmcopt", or "lrdmc"
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 300 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_chk = "hamiltonian_data.chk" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"
[vmc]
num_mcmc_steps = 90000 # Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates per measurement. Every local energy and other observeables are measured every this steps.
num_mcmc_warmup_steps = 0 # Number of observable measurement steps for warmup (i.e., discarged).
num_mcmc_bin_blocks = 5 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
```

The final step is to run the `jqmc` job w/ or w/o MPI on a CPU or GPU machine (via a job queueing system such as PBS).

```bash
% jqmc vmc.toml > out_vmc 2> out_vmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc vmc.toml > out_vmc 2> out_vmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc vmc.toml > out_vmc 2> out_vmc.e # w/ MPI on GPU, depending the queueing system.
```

You may get `E = -xxxxx +- xxx` [VMC w/ Jastrow factors]

## Compute Energy (LRDMC)
The final step is LRDMC calculation. You can generate a template file for a LRDMC calculation using `jqmc-tool`. Please directly edit `lrdmc.toml` if you want to change a parameter.

```bash
% jqmc-tool lrdmc generate-input -g lrdmc_a_0.30.toml
> Input file is generated: lrdmc_a_0.30.toml
```

```toml:lrdmc.toml
[control]
...
```

LRDMC energy is biased with the discretized lattice space ($a$) by $O(a^2)$. It means that, to get an unbiased energy, one should compute LRDMC energies with several lattice parameters ($a$) extrapolate them into $a \rightarrow 0$.

The final step is to run the `jqmc` jobs with several $a$.

```bash
% jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e
```

You may get `E = -xxxxx +- xxx` [VMC w/ Jastrow factors]
