# example05

Potential energy surface of hydrogen molecule with cartesian GTOs. All electron calculations. Comparison of atomic forces with the derivative of the PES. The atomic forces are computed by fully exploiting algorithmic differentiation (AD) as implemented in **JAX**. The pioneering application of AD in ab initio QMC was first introduced by S. Sorella and L. Capriotti in 2010 [^2010SORjcp].

## Generate a trial WF

The first step of ab-initio QMC is to generate a trial WF by a mean-field theory such as DFT/HF. `jQMC` interfaces with other DFT/HF software packages via `TREXIO`.

One of the easiest ways to produce it is using `pySCF` as a converter to the `TREXIO` format is implemented. The following is a script to run a HF calculation of the water molecule and dump it as a `TREXIO` file.

```bash
% cd 01trialWF_DFT
```

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
% cd 02convert_DFT_to_JSD
% jqmc-tool trexio convert-to H2_R_0.74.h5 -j1 1.0 -j2 1.0 -j3 mo
> Hamiltonian data is saved in hamiltonian_data.h5.
```

The generated `hamiltonian_data.h5` is a wavefunction file with the `jqmc` format. `-j2` specifies the initial value of the one-body Jastrow parameter, `-j2` specifies the initial value of the two-body Jastrow parameter, and `-j3` specifies the basis set (`ao`:atomic orbital or `mo`:molecular orbital) for the three-body Jastrow part.

## Optimize a trial WF (MCMC)
The next step is to optimize variational parameters included in the generated wavefunction. More in details, here, we optimize the two-body Jastrow parameter and the matrix elements of the three-body Jastrow parameter.

You can generate a template file for a MCMC calculation using `jqmc-tool`. Please directly edit `vmc.toml` if you want to change a parameter.

```bash
% cd 03vmc_JSD
% jqmc-tool vmc generate-input -g
> Input file is generated: vmc.toml
```

```toml:vmc.toml
[control]
job_type = "vmc" # Specify the job type. "mcmc", "vmc", "lrdmc", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"

[vmc]
num_mcmc_steps = 300 # Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates per measurement. Every local energy and other observeables are measured every this steps.
num_mcmc_warmup_steps = 0 # Number of observable measurement steps for warmup (i.e., discarged).
num_mcmc_bin_blocks = 1 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 1.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
num_opt_steps = 100 # Number of optimization steps.
wf_dump_freq = 20 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
optimizer_kwargs = { method = "sr", delta = 0.050, epsilon = 0.001 } # SR optimizer configuration (method plus step/regularization).
opt_J1_param = true
opt_J2_param = true
opt_J3_param = true
opt_lambda_param = false
num_param_opt = 0 # the number of parameters to optimize in the descending order of |f|/|std f|. If it is set 0, all parameters are optimized.
```

Please lunch the job.

```bash
% jqmc vmc.toml > out_vmc 2> out_vmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc vmc.toml > out_vmc 2> out_vmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc vmc.toml > out_vmc 2> out_vmc.e # w/ MPI on GPU, depending the queueing system.
```

You can see and plot the outcome using `jqmc-tool`.

```bash
% jqmc-tool vmc analyze-output out_vmc

------------------------------------------------------------
Iter     E (Ha)     Max f (Ha)   Max signal to noise of f
------------------------------------------------------------
   1  -0.9543(42)  +0.4320(40)   129.023
   2  -0.8488(21)  +1.6470(70)   311.388
   3  -0.9400(19)  +1.3950(60)   240.943
   4  -0.9955(17)  +1.1830(60)   206.101
   5  -1.0330(16)  +1.0330(50)   203.363
   6  -1.0612(15)  +0.8890(50)   195.200
   7  -1.0826(14)  +0.7780(40)   196.997
   8  -1.1026(13)  +0.6830(40)   186.626
   9  -1.1141(12)  +0.5990(30)   190.358
  10  -1.1238(12)  +0.5240(30)   186.276
 ...
 191  -1.16993(48)  +0.0000(10)     0.430
 192  -1.17020(45)  -0.0000(10)     0.448
 193  -1.16920(47)  +0.0010(10)     1.656
 194  -1.16956(46)  +0.0000(10)     0.389
 195  -1.17023(47)  +0.0000(00)     0.774
 196  -1.16927(45)  +0.0020(10)     2.858
 197  -1.17070(47)  -0.0010(10)     1.263
 198  -1.16959(47)  -0.0020(10)     2.805
 199  -1.16965(46)  -0.0000(10)     0.874
 200  -1.16922(47)  -0.0000(10)     0.550
------------------------------------------------------------

```

The important criteria are `Max f` and `max signal to noise of f`. `f_max` should be zero within the error bar. A practical criterion for `signal to noise` is < 4~5 because it means that all the residual forces are zero in the statistical sense.


## Compute Energy and Atomic forces (MCMC)
The next step is MCMC calculation. You can generate a template file for a MCMC calculation using `jqmc-tool`. Please directly edit `mcmc.toml` if you want to change a parameter.

```bash
% cd 04mcmc_JSD
% jqmc-tool mcmc generate-input -g
> Input file is generated: mcmc.toml
```

```toml:mcmc.toml
[control]
job_type = "mcmc" # Specify the job type. "mcmc", "vmc", "lrdmc", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"

[mcmc]
num_mcmc_steps = 10000 # Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates per measurement. Every local energy and other observeables are measured every this steps.
num_mcmc_warmup_steps = 10 # Number of observable measurement steps for warmup (i.e., discarged).
num_mcmc_bin_blocks = 5 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 1.2 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
atomic_force = true
```

Run the `jqmc` job w/ or w/o MPI on a CPU or GPU machine (via a job queueing system such as PBS).

```bash
% jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on GPU, depending the queueing system.
```

You may get `E = -1.16986 +- 0.000079 Ha` and  `Var(E) = 0.03025 +- 0.000071 Ha^2`.

and `F =`

```bash
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       -9(9)e-05    +6(9)e-05    +0.00311(22)
  H       +9(9)e-05    -6(9)e-05    -0.00311(22)
  ------------------------------------------------
```

## Compute Energy and Atomic forces (LRDMC)
The next step is LRDMC calculation. You can generate a template file for a LRDMC calculation using `jqmc-tool`. Please directly edit `lrdmc.toml` if you want to change a parameter.

```bash
% cd 05lrdmc_JSD
% jqmc-tool lrdmc generate-input -g
> Input file is generated: lrdmc.toml
```

```toml:lrdmc.toml
[control]
job_type = "lrdmc" # Specify the job type. "mcmc", "vmc", "lrdmc", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"

[lrdmc]
num_mcmc_steps = 10000 # Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 30 # Number of GFMC projections per measurement. Every local energy and other observeables are measured every this projection.
alat = 0.10 # The lattice discretization parameter (i.e. grid size) used for discretized the Hamiltonian and potential. The lattice spacing is alat * a0, where a0 is the Bohr radius.
non_local_move = "tmove" # The treatment of the non-local term in the Effective core potential. tmove (T-move) and dltmove (Determinant locality approximation with T-move) are available.
num_gfmc_warmup_steps = 10 # Number of observable measurement steps for warmup (i.e., discarged).
num_gfmc_bin_blocks = 10 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_gfmc_bin_blocks, not num_gfmc_bin_blocks * mpi_size * number_of_walkers.
num_gfmc_collect_steps = 5 # Number of measurement (before binning) for collecting the weights.
E_scf = -1.0 # The initial guess of the total energy. This is used to compute the initial energy shift in the GFMC.
atomic_force = true
```

Run the `jqmc` job w/ or w/o MPI on a CPU or GPU machine (via a job queueing system such as PBS).

```bash
% jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e # w/ MPI on GPU, depending the queueing system.
```

You may get `E = -1.17485 +- 0.000248 Ha` and  `Var(E) = 0.02985 +- 0.000165 Ha^2`.

and `F =`

```bash
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       -0.0009(6)  +0.0006(7)  -0.0066(8)
  H       +0.0009(6)  -0.0006(7)  +0.0066(8)
  ------------------------------------------------
```

> [!NOTE]
> The LRDMC forces are intringically biased because the so-called Reynolds approximation[^1989REYijqc] is employed. See benchmark papers[^2021NAKjcp] [^2022TIHjcp].

## Conversion of WF: from JSD to JAGP

The next step is to convert the optimized JSD ansatz to JAGP one.

```bash
% cd 06convert_JSD_to_JAGP
% cp ../04mcmc_JSD/hamiltonian_data.h5 ./hamiltonian_data_JSD.h5
% jqmc-tool hamiltonian conv-wf --convert-to jagp hamiltonian_data_JSD.h5
> Convert SD to AGP.
> Hamiltonian data is saved in hamiltonian_data_conv.h5.
% mv hamiltonian_data_conv.h5 hamiltonian_data_JAGP.h5
```

## Optimize a trial WF (MCMC)
The next step is to optimize variational parameters included in the generated wavefunction. Here, we optimize all the variational parameters.

```bash
% cd 07vmc_JAGP
% cp ../06convert_JSD_to_JAGP/hamiltonian_data_JAGP.h5 ./hamiltonian_data.h5
% jqmc-tool vmc generate-input -g
> Input file is generated: vmc.toml
```

```vmc.toml
[control]
job_type = "vmc" # Specify the job type. "mcmc", "vmc", "lrdmc", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"

[vmc]
num_mcmc_steps = 300 # Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates per measurement. Every local energy and other observeables are measured every this steps.
num_mcmc_warmup_steps = 0 # Number of observable measurement steps for warmup (i.e., discarged).
num_mcmc_bin_blocks = 1 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 1.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
num_opt_steps = 200 # Number of optimization steps.
wf_dump_freq = 10 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
optimizer_kwargs = { method = "sr", delta = 0.05, epsilon = 0.001 } # SR optimizer configuration (method plus step/regularization).
opt_J1_param = true
opt_J2_param = true
opt_J3_param = true
opt_lambda_param = true
num_param_opt = 0 # the number of parameters to optimize in the descending order of |f|/|std f|. If None, all parameters are optimized.
```

Please lunch the job.

```bash
% jqmc vmc.toml > out_vmc 2> out_vmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc vmc.toml > out_vmc 2> out_vmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc vmc.toml > out_vmc 2> out_vmc.e # w/ MPI on GPU, depending the queueing system.
```

You can see and plot the outcome using `jqmc-tool`.

```bash
% jqmc-tool vmc analyze-output out_vmc

------------------------------------------------------
Iter     E (Ha)     Max f (Ha)   Max of signal to noise of f
------------------------------------------------------
   1  -1.17025(46)  +0.0690(10)    79.768
   2  -1.17148(31)  +0.0570(10)    68.719
   3  -1.17300(21)  +0.0340(10)    70.244
   4  -1.17354(17)  -0.0300(10)    72.861
   5  -1.17376(15)  -0.0260(10)    66.032
   6  -1.17371(14)  -0.0220(10)    62.463
   7  -1.17405(14)  -0.0190(10)    57.303
   8  -1.17381(14)  -0.0150(10)    52.436
   9  -1.17401(13)  -0.0130(10)    43.747
  10  -1.17420(14)  -0.0110(10)    40.793
  11  -1.17404(14)  -0.0100(10)    35.368
  12  -1.17415(13)  -0.0080(10)    29.669
  13  -1.17403(13)  -0.0060(10)    27.673
  14  -1.17408(14)  -0.0050(10)    24.524
  15  -1.17414(15)  -0.0050(10)    21.827
  16  -1.17426(14)  -0.0050(10)    20.321
  17  -1.17405(14)  -0.0040(10)    15.438
  18  -1.17401(13)  -0.0050(10)    15.205
  19  -1.17419(14)  -0.0020(10)    12.648
  20  -1.17404(14)  -0.0030(10)    10.937
  21  -1.17403(14)  -0.0020(10)    10.505
  22  -1.17409(14)  -0.0020(10)    10.847
  23  -1.17413(15)  -0.0020(10)     7.368
  24  -1.17424(14)  -0.0020(10)     7.243
  25  -1.17414(14)  -0.0020(10)     6.574
  26  -1.17409(13)  +0.0010(10)     5.080
  27  -1.17407(15)  -0.0020(10)     4.100
  28  -1.17414(14)  -0.0010(10)     4.578
  29  -1.17368(14)  -0.0002(00)     4.743
  30  -1.17407(15)  +0.0010(10)     3.801
```

One should gain energy with repect to the JSD one.

## Compute Energy and Atomic forces (MCMC)
The following steps are the same as those with JSD.

```bash
% cd 08mcmc_JAGP
% jqmc-tool mcmc generate-input -g
> Input file is generated: mcmc.toml
```

```toml:mcmc.toml
[control]
job_type = "mcmc" # Specify the job type. "mcmc", "vmc", "lrdmc", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"

[mcmc]
num_mcmc_steps = 10000 # Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates per measurement. Every local energy and other observeables are measured every this steps.
num_mcmc_warmup_steps = 10 # Number of observable measurement steps for warmup (i.e., discarged).
num_mcmc_bin_blocks = 5 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 1.2 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
atomic_force = true
```

Run the `jqmc` job w/ or w/o MPI on a CPU or GPU machine (via a job queueing system such as PBS).

```bash
% jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on GPU, depending the queueing system.
```

You may get `E = -1.17543 +- 0.001343 Ha` and  `Var(E) = 0.00327 +- 0.000475 Ha^2`.

and `F =`

```bash
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       +0.00015(5)  -3(5)e-05    -0.00042(20)
  H       -0.00015(5)  +3(5)e-05    +0.00042(20)
  ------------------------------------------------
```

## Compute Energy and Atomic forces (LRDMC)
The next step is LRDMC calculation. You can generate a template file for a LRDMC calculation using `jqmc-tool`. Please directly edit `lrdmc.toml` if you want to change a parameter.

```bash
% cd 09lrdmc_JAGP
% jqmc-tool lrdmc generate-input -g
> Input file is generated: lrdmc.toml
```

```toml:lrdmc.toml
[control]
job_type = "lrdmc" # Specify the job type. "mcmc", "mcmc", "lrdmc", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"

[lrdmc]
num_mcmc_steps = 10000 # Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 30 # Number of GFMC projections per measurement. Every local energy and other observeables are measured every this projection.
alat = 0.10 # The lattice discretization parameter (i.e. grid size) used for discretized the Hamiltonian and potential. The lattice spacing is alat * a0, where a0 is the Bohr radius.
non_local_move = "tmove" # The treatment of the non-local term in the Effective core potential. tmove (T-move) and dltmove (Determinant locality approximation with T-move) are available.
num_gfmc_warmup_steps = 10 # Number of observable measurement steps for warmup (i.e., discarged).
num_gfmc_bin_blocks = 10 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_gfmc_bin_blocks, not num_gfmc_bin_blocks * mpi_size * number_of_walkers.
num_gfmc_collect_steps = 5 # Number of measurement (before binning) for collecting the weights.
E_scf = -1.0 # The initial guess of the total energy. This is used to compute the initial energy shift in the GFMC.
atomic_force = true
```

Run the `jqmc` job w/ or w/o MPI on a CPU or GPU machine (via a job queueing system such as PBS).

```bash
% jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e # w/ MPI on GPU, depending the queueing system.
```

You may get `E = -1.17442 +- 0.000069 Ha` and  `Var(E) = 0.00287 +- 0.000010 Ha^2`.

and `F =`

```bash
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       -0.0007(6)  +0.0009(10) -0.0058(10)
  H       +0.0007(6)  -0.0009(10) +0.0058(10)
  ------------------------------------------------
```

## Summary: PES of the Hydrogen dimer
One should repeat the above calculations with many $R$ to compute the PES. Here, we plot the PESs, thier derivatives, and forces obtained with JSD and JAGP anstaz.


![H2 MCMC PES](H2/H2_PES_mcmc.jpg)

We notice that the forces shown in the left panel (JSD) remain finite even near the equilibrium position of the PES, though it is negligible for the simple molecule, H$_2$. This discrepancy originates from the so-called self-consistency error[^2021NAKjcp][^2022TIHjcp]. In the JSD calculation, only the Jastrow factor was optimized, while the determinant part was kept fixed to the DFT solution. Although the DFT orbitals satisfy the stationary condition of the Kohn–Sham equations, they are not stationary with respect to the MCMC energy. As a result, if the variational parameters in the determinant part are not optimized, an additional contribution should be considered in the force evaluation, which is not captured by standard MCMC and DMC formalisms.

There are two ways to eliminate this bias. One is to optimize all variational parameters, including those in the determinant part[^2024SLOjctc]. Indeed, the forces shown in the left panel obtained using the JAGP ansatz exhibit no such inconsistency: the derivative of the PES agrees with the calculated force values. The other approach is to explicitly correct for the missing contribution[^2024NAKprb]. If one is intereted the latter approach, please contact a `jQMC` develper.

[^2010SORjcp]: S. Sorella and L. Capriotti, J. Chem. Phys. **133** 234111 (2010) [https://doi.org/10.1063/1.3516208](https://doi.org/10.1063/1.3516208)

[^2021NAKjcp]: K. Nakano et al. J. Chem. Phys. **154**, 204111 (2021), [https://doi.org/10.1063/5.0076302](https://doi.org/10.1063/5.0076302)
[^2022TIHjcp]: J. Tiihonen et al. J. Chem. Phys. **156**, 034101 (2022), [https://doi.org/10.1063/5.0052266](https://doi.org/10.1063/5.0052266)
[^1989REYijqc]: P.J. Reynolds et al. Int. J. Quantum Chem. **29**, 589 (1986). [https://doi.org/10.1063/5.0052266](https://doi.org/10.1063/5.0052266)
[^2024SLOjctc]: E. Slootman et al. J. Chem. Theory Comput. **20**, 6020-6027 (2024), [https://doi.org/10.1021/acs.jctc.4c00498](https://doi.org/10.1021/acs.jctc.4c00498)
[^2024NAKprb]: K. Nakano et al. Phys. Rev. B **109**, 205151 (2024), [https://doi.org/10.1103/PhysRevB.109.205151](https://doi.org/10.1103/PhysRevB.109.205151)
