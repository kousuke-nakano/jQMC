# example05

Potential energy surface of hydrogen molecule with cartesian GTOs. All electron calculations. Comparison atomic forces with the derivative of the PES.

## Generate a trial WF

The first step of ab-initio QMC is to generate a trial WF by a mean-field theory such as DFT/HF. `jQMC` interfaces with other DFT/HF software packages via `TREXIO`.

One of the easiest ways to produce it is using `pySCF` as a converter to the `TREXIO` format is implemented. The following is a script to run a HF calculation of the water molecule and dump it as a `TREXIO` file.

```python:run_pyscf.py
from pyscf import gto, scf
from pyscf.tools import trexio

R = 0.74 # angstrom
filename = f'H2_R_{R}.h5'

mol = gto.Mole()
mol.verbose  = 5
mol.atom     = f'''
  H    0.000000000   0.000000000  {-R/2}
  H    0.000000000   0.000000000  {+R/2}
               '''
mol.basis    = 'ccpvtz'
mol.unit     = 'A'
mol.ecp      = None
mol.charge   = 0
mol.spin     = 0
mol.symmetry = False
mol.cart = True
mol.output   = f'H2_R_{R}.out'
mol.build()

mf = scf.KS(mol).density_fit()
mf.max_cycle=200
mf.xc = 'LDA_X,LDA_C_PZ'
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)

```

Launch it on a terminal. You may get `E = -1.13700890749411 Ha` [DFT-LDA-PZ].

```bash
% python run_pyscf.py
```

Next step is to convert the `TREXIO` file to the `jqmc` format using `jqmc-tool`

```bash
% jqmc-tool trexio convert-to H2_R_0.74.h5 -j1 1.0 -j2 1.0 -j3 mo
> Hamiltonian data is saved in hamiltonian_data.chk.
```

The generated `hamiltonian_data.chk` is a wavefunction file with the `jqmc` format. `-j2` specifies the initial value of the one-body Jastrow parameter, `-j2` specifies the initial value of the two-body Jastrow parameter, and `-j3` specifies the basis set (`ao`:atomic orbital or `mo`:molecular orbital) for the three-body Jastrow part.

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
num_mcmc_steps = 300 # Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates per measurement. Every local energy and other observeables are measured every this steps.
num_mcmc_warmup_steps = 0 # Number of observable measurement steps for warmup (i.e., discarged).
num_mcmc_bin_blocks = 1 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 1.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
num_opt_steps = 100 # Number of optimization steps.
wf_dump_freq = 20 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
delta = 0.050 # Step size for the Stochastic reconfiguration (i.e., the natural gradient) optimization.
epsilon = 0.001 # Regularization parameter, a positive number added to the diagnoal elements of the Fisher-Information matrix, used during the Stochastic reconfiguration to improve the numerical stability.
opt_J1_param = true
opt_J2_param = true
opt_J3_param = true
opt_lambda_param = false
num_param_opt = 0 # the number of parameters to optimize in the descending order of |f|/|std f|. If it is set 0, all parameters are optimized.
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

------------------------------------------------------------
Iter     E (Ha)     Max f (Ha)   Max signal to noise of f
------------------------------------------------------------
   1  -0.9591(84)  +0.4370(70)    65.126
   2  -1.0080(32)  +1.0180(80)   183.299
   3  -1.0534(24)  +0.8800(80)   124.577
   4  -1.0883(26)  +0.7810(60)   126.926
   5  -1.1001(22)  +0.6730(70)   101.442
   6  -1.1122(23)  +0.5990(60)   104.694
   7  -1.1205(25)  +0.5280(60)    89.668
   8  -1.1299(22)  +0.4800(60)    79.839
   9  -1.1336(19)  +0.4500(50)    93.379
  10  -1.1384(20)  +0.4020(50)    81.551
  11  -1.1426(20)  +0.3630(50)    76.426
  12  -1.1481(21)  +0.3190(50)    63.541
  13  -1.1505(19)  +0.2820(40)    66.548
  14  -1.1570(18)  +0.2390(40)    60.177
  15  -1.1588(15)  +0.2190(40)    60.633
  16  -1.1585(17)  +0.2140(30)    64.229
  17  -1.1595(14)  +0.1800(30)    57.780
  18  -1.1672(14)  +0.1620(30)    49.647
  19  -1.1648(13)  +0.1580(30)    56.120
  20  -1.1645(14)  +0.1490(30)    56.617
  21  -1.1646(13)  +0.1320(30)    45.242
  22  -1.1666(12)  +0.1180(30)    44.301
  23  -1.1678(13)  +0.1060(30)    38.721
  24  -1.1663(10)  +0.1000(20)    45.989
  25  -1.1652(10)  +0.0950(30)    38.079
  ...
  90  -1.17014(86)  +0.0060(20)     3.136
  91  -1.16973(96)  +0.0120(20)     6.641
  92  -1.16976(94)  +0.0060(20)     2.916
  93  -1.17070(86)  +0.0120(20)     6.649
  94  -1.17152(92)  +0.0110(20)     5.675
  95  -1.16896(86)  +0.0070(20)     3.924
  96  -1.17063(85)  +0.0040(20)     2.648
  97  -1.16826(95)  +0.0050(20)     2.783
  98  -1.16788(79)  +0.0110(20)     8.350
  99  -1.16882(96)  +0.0050(20)     2.976
 100  -1.17031(82)  +0.0050(20)     2.758
------------------------------------------------------------

```

The important criteria are `Max f` and `max signal to noise of f`. `f_max` should be zero within the error bar. A practical criterion for `signal to noise` is < 4~5 because it means that all the residual forces are zero in the statistical sense.


## Compute Energy and Atomic forces (VMC)
The next step is VMC calculation. You can generate a template file for a VMC calculation using `jqmc-tool`. Please directly edit `vmc.toml` if you want to change a parameter.

```bash
% jqmc-tool vmc generate-input -g
> Input file is generated: vmc.toml
```

```toml:vmc.toml
[control]
job_type = "vmc" # Specify the job type. "vmc", "vmcopt", "lrdmc", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_chk = "hamiltonian_data.chk" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"

[vmc]
num_mcmc_steps = 1000 # Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 30 # Number of MCMC updates per measurement. Every local energy and other observeables are measured every this steps.
num_mcmc_warmup_steps = 10 # Number of observable measurement steps for warmup (i.e., discarged).
num_mcmc_bin_blocks = 50 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 1.2 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
atomic_force = true
```

Run the `jqmc` job w/ or w/o MPI on a CPU or GPU machine (via a job queueing system such as PBS).

```bash
% jqmc vmc.toml > out_vmc 2> out_vmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc vmc.toml > out_vmc 2> out_vmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc vmc.toml > out_vmc 2> out_vmc.e # w/ MPI on GPU, depending the queueing system.
```

You may get `E = -1.17034 +- 0.000522` and  `Var(E) = 0.03412 +- 0.000707 Ha^2`.

and `F =`

```bash
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       -0.00018(31)+0.00009(31)-0.0530(14)
  H       +0.00018(31)-0.00009(31)+0.0530(14)
  ------------------------------------------------
```

## PES of the Hydrogen dimer
xxx
