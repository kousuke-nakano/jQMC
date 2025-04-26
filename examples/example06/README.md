# example06

Binding energy of the water-water dimer with the JSD and JAGP anstaz. One can learn how to obtain the VMC and DMC (in the extrapolated limit) energies of the Water dimer, starting from scratch (i.e., DFT calculation), with cartesian GTOs, and one can see how the binding energy is improved by optimizing the nodal surface. The following calculation is for the water-water dimer. The water molecule fragments should also be computed to get the binding energy. In this example, the water-water dimer geometry is taken from the S22 dataset[^2006JURpccp].

[2006JURpccp]: P. Jurecka, et al. Phys Chem Chem Phys, 8, 1985-1993 (2006) [https://doi.org/10.1039/B600027D](https://doi.org/10.1039/B600027D)

## Generate a trial WF

The first step of ab-initio QMC is to generate a trial WF by a mean-field theory such as DFT/HF. `jQMC` interfaces with other DFT/HF software packages via `TREXIO`.

One of the easiest ways to produce it is using `pySCF` as a converter to the `TREXIO` format is implemented. The following is a script to run a HF calculation of the water molecule and dump it as a `TREXIO` file.

> [!NOTE]
> This `TREXIO` converter is being develped in the `pySCF-forge` [repository](https://github.com/pyscf/pyscf-forge) and not yet merged to the main repository of `pySCF`. Please use `pySCF-forge`.

```python:run_pyscf.py
from pyscf import gto, scf
from pyscf.tools import trexio

filename = f'water_dimer.h5'

mol = gto.Mole()
mol.verbose  = 5
mol.atom     = f'''
	    O  -1.551007  -0.114520   0.000000
	    H  -1.934259   0.762503   0.000000
	    H  -0.599677   0.040712   0.000000
	    O   1.350625   0.111469   0.000000
	    H   1.680398  -0.373741  -0.758561
	    H   1.680398  -0.373741   0.758561
'''
mol.basis    = 'ccecp-aug-ccpvtz'
mol.unit     = 'A'
mol.ecp      = 'ccecp'
mol.charge   = 0
mol.spin     = 0
mol.symmetry = False
mol.cart = True
mol.output   = f'water_dimer.out'
mol.build()

mf = scf.KS(mol).density_fit()
mf.max_cycle=200
mf.xc = 'LDA_X,LDA_C_PZ'
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)

```

Launch it on a terminal. You may get `E = -34.3124355699676 Ha` [Hartree-Forck].

```bash
% python run_pyscf.py
```

Next step is to convert the `TREXIO` file to the `jqmc` format using `jqmc-tool`

```bash
% jqmc-tool trexio convert-to water_dimer.h5 -j2 1.0 -j3 ao-medium
> Hamiltonian data is saved in hamiltonian_data.chk.
```

The generated `hamiltonian_data.chk` is a wavefunction file with the `jqmc` format. `-j2` specifies the initial value of the two-body Jastrow parameter and `-j3` specifies the basis set (`ao-xxx`:atomic orbital or `mo`:molecular orbital) for the three-body Jastrow part.

[!NOTE]
> The `-j3` option in `jqmc-tool` controls how the atomic orbital (AO) basis is partitioned by Gaussian exponent strength for each nucleus. When you specify `ao-small`, `jqmc-tool` sorts each atom’s contracted AOs by the exponent of the primitive shell with the largest coefficient, divides them into **three** equal groups, and retains only the central group (i.e., 1/3). Specifying `ao-medium` divides into **four** groups and keeps the **two** central groups (i.e., 2/4), while `ao-large` divides into **five** groups and keeps the **three** central groups (i.e., 3/5). Using `ao` or `ao-full` disables any partitioning and includes all AOs from the original dataset. Using `mo` includes all MOs. This mechanism lets you tailor the trade-off between computational cost and accuracy by focusing on the most representative subset of basis functions.

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
number_of_walkers = 1 # Number of walkers per MPI process
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
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
num_opt_steps = 200 # Number of optimization steps.
wf_dump_freq = 20 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
delta = 0.01 # Step size for the Stochastic reconfiguration (i.e., the natural gradient) optimization.
epsilon = 0.001 # Regularization parameter, a positive number added to the diagnoal elements of the Fisher-Information matrix, used during the Stochastic reconfiguration to improve the numerical stability.
opt_J1_param = false
opt_J2_param = true
opt_J3_param = true
opt_lambda_param = false
num_param_opt = 0 # the number of parameters to optimize in the descending order of |f|/|std f|. If it is set 0, all parameters are optimized.
cg_flag = true
cg_max_iter = 100000 # Maximum number of iterations for the conjugate gradient method.
cg_tol = 0.0001 # Tolerance for the conjugate gradient method.
```

Please lunch the job.

```bash
% jqmc vmcopt.toml > out_vmcopt 2> out_vmcopt.e # w/o MPI on CPU
% mpirun -np 4 jqmc vmcopt.toml > out_vmcopt 2> out_vmcopt.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc vmcopt.toml > out_vmcopt 2> out_vmcopt.e # w/ MPI on GPU, depending the queueing system.
```

You can see the outcome using `jqmc-tool`.

```bash
% jqmc-tool vmcopt analyze-output out_vmcopt

------------------------------------------------------
Iter     E (Ha)     Max f (Ha)   Max of signal to noise of f
------------------------------------------------------
   1  -16.5743(97)  +1.132(12)   110.335
   2  -16.5921(96)  +1.097(12)   109.386
   3  -16.6117(95)  +1.084(12)   104.849
   4  -16.6399(93)  +1.059(12)   104.245
   5  -16.6678(91)  +1.029(12)   102.269
   6  -16.6819(90)  +1.009(12)   100.122
   7  -16.7028(90)  +0.993(12)    97.718
   8  -16.6974(87)  +0.963(12)    96.040
   9  -16.7200(87)  +0.948(11)    94.616
  10  -16.7511(87)  +0.914(11)    91.563
  11  -16.7602(85)  +0.895(11)    90.790
  12  -16.7714(85)  +0.878(11)    88.758
  13  -16.7867(85)  +0.848(10)    87.979
  14  -16.7940(86)  +0.835(11)    83.253
  15  -16.8065(83)  +0.787(10)    82.875
  16  -16.8112(83)  +0.777(10)    81.196
  17  -16.8284(82)  +0.741(10)    80.058
  18  -16.8327(83)  +0.743(10)    76.214
------------------------------------------------------
```

The important criteria are `Max f` and `Max of signal to noise of f`. `Max f` should be zero within the error bar. A practical criterion for the `signal to noise` is < 4~5 because it means that all the residual forces are zero in the statistical sense.

You can also plot them and make a figure.

```bash
% jqmc-tool vmcopt analyze-output out_vmcopt -p -s vmcopt.jpg
```

![VMC optimization](03vmcopt_JSD/vmcopt.jpg)

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

You may get `E = -xx.xxxx +- x.xxxxx Ha` [VMC w/ Jastrow factors]

> [!NOTE]
> We are going to discuss the sub kcal/mol accuracy in the binding energy. So, we need to decrease the error bars of the mononer and dimer calculations up to $\sim$ 0.10 mHa and $\sim$ 0.15 mHa.

## Compute Energy (LRDMC)
The final step is LRDMC calculation. You can generate a template file for a LRDMC calculation using `jqmc-tool`. Please directly edit `lrdmc.toml` if you want to change a parameter.

```bash
% cd alat_0.30
% jqmc-tool lrdmc generate-input -g lrdmc.toml
> Input file is generated: lrdmc.toml
```

```toml:lrdmc.toml
[control]
  job_type = 'lrdmc'
  mcmc_seed = 34467
  number_of_walkers = 300
  max_time = 10400
  restart = false
  restart_chk = 'lrdmc.rchk'
  hamiltonian_chk = '../hamiltonian_data.chk'
  verbosity = 'low'

[lrdmc]
  num_mcmc_steps  = 40000
  num_mcmc_per_measurement = 30
  alat = 0.30
  non_local_move = "dltmove"
  num_gfmc_warmup_steps = 50
  num_gfmc_bin_blocks = 50
  num_gfmc_collect_steps = 20
  E_scf = -17.00
```

LRDMC energy is biased with the discretized lattice space ($a$) by $O(a^2)$. It means that, to get an unbiased energy, one should compute LRDMC energies with several lattice parameters ($a$) extrapolate them into $a \rightarrow 0$.

The final step is to run the `jqmc` jobs with several $a$, e.g.

```bash
% cd alat_0.30
% jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e
```

You may get:

| a (bohr)   | E (Ha)                  |  Var (Ha^2)             |
|------------|-------------------------|-------------------------|
| 0.10       | -17.23667 ± 0.000277    | 1.61602 +- 0.000643     |
| 0.15       | -17.23821 ± 0.000286    | 1.61417 +- 0.000773     |
| 0.20       | -17.24097 ± 0.000325    | 1.69783 +- 0.079714     |
| 0.25       | -17.24402 ± 0.000270    | 1.63235 +- 0.006160     |
| 0.30       | -17.24786 ± 0.000269    | 1.78517 +- 0.066418     |

You can extrapolate them into $a \rightarrow 0$ by `jqmc-tool`

```bash
% jqmc-tool lrdmc extrapolate-energy alat_0.10/lrdmc.rchk alat_0.15/lrdmc.rchk alat_0.20/lrdmc.rchk alat_0.25/lrdmc.rchk alat_0.30/lrdmc.rchk -s lrdmc_ext.jpg
> ------------------------------------------------------------------------
> Read restart checkpoint files from ['alat_0.10/lrdmc.rchk', 'alat_0.15/lrdmc.rchk', 'alat_0.20/lrdmc.rchk', 'alat_0.25/lrdmc.rchk', 'alat_0.30/lrdmc.rchk'].
>   Total number of binned samples = 5
>   For a = 0.1 bohr: E = -17.236661112856858 +- 0.00032635704517869677 Ha.
>   Total number of binned samples = 5
>   For a = 0.15 bohr: E = -17.2382052864809 +- 0.00029723715520135464 Ha.
>   Total number of binned samples = 5
>   For a = 0.2 bohr: E = -17.240993162088692 +- 0.00025740878490131835 Ha.
>   Total number of binned samples = 5
>   For a = 0.25 bohr: E = -17.24401036198691 +- 0.0002365677591168457 Ha.
>   Total number of binned samples = 5
>   For a = 0.3 bohr: E = -17.247804041851044 +- 0.00032247173445041217 Ha.
> ------------------------------------------------------------------------
> Extrapolation of the energy with respect to a^2.
>   Polynomial order = 2.
>   For a -> 0 bohr: E = -17.235093943871842 +- 0.00045277462865289897 Ha.
> ------------------------------------------------------------------------
> Graph is saved in lrdmc_ext.jpg.
> ------------------------------------------------------------------------
> Extrapolation is finished.

```

You may get `E = -xx.xxxx +- x.xxxxx Ha` [LRDMC w/ Jastrow factor in a -> 0].

> [!NOTE]
> We are going to discuss the sub kcal/mol accuracy in the binding energy. So, we need to decrease the error bars of the mononer and dimer calculations up to $\sim$ 0.10 mHa and $\sim$ 0.15 mHa.

Your binding enery is....

| Ansatz     | Method                  | Binding energy          |  ref      |
|------------|-------------------------|-------------------------|-----------|
| JDFT       | VMC                     | 1.61602 +- 0.000643     | this work |

## Conversion of WF: from JSD to JAGP

The next step is to convert the optimized JSD ansatz to JAGP one.

```bash
% cd 06convert_JSD_to_JAGP
% cp ../04vmc_JSD/hamiltonian_data.chk ./hamiltonian_data_JSD.chk
% jqmc-tool hamiltonian conv-wf --convert-to jagp hamiltonian_data_JSD.chk
> Convert SD to AGP.
> Hamiltonian data is saved in hamiltonian_data_conv.chk.
% mv hamiltonian_data_conv.chk hamiltonian_data_JAGP.chk
```


## Optimize a trial WF (VMCopt)
The next step is to optimize variational parameters included in the generated wavefunction. More in details, here, we optimize the two-body Jastrow parameter and the matrix elements of the three-body Jastrow parameter and the AGP matrix elements!

You can generate a template file for a VMCopt calculation using `jqmc-tool`. Please directly edit `vmcopt.toml` if you want to change a parameter.

```bash
% cd 07vmcopt_JAGP
% cp ../06convert_JSD_to_JAGP/hamiltonian_data_JAGP.chk ./hamiltonian_data.chk
% jqmc-tool vmcopt generate-input -g
> Input file is generated: vmcopt.toml
```

```toml:vmcopt.toml
[control]
job_type = "vmcopt" # Specify the job type. "vmc", "vmcopt", "lrdmc", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 1 # Number of walkers per MPI process
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
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
num_opt_steps = 200 # Number of optimization steps.
wf_dump_freq = 20 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
delta = 0.01 # Step size for the Stochastic reconfiguration (i.e., the natural gradient) optimization.
epsilon = 0.001 # Regularization parameter, a positive number added to the diagnoal elements of the Fisher-Information matrix, used during the Stochastic reconfiguration to improve the numerical stability.
opt_J1_param = false
opt_J2_param = true
opt_J3_param = true
opt_lambda_param = true
num_param_opt = 0 # the number of parameters to optimize in the descending order of |f|/|std f|. If it is set 0, all parameters are optimized.
cg_flag = true
cg_max_iter = 100000 # Maximum number of iterations for the conjugate gradient method.
cg_tol = 0.0001 # Tolerance for the conjugate gradient method.
```

Please lunch the job.

```bash
% jqmc vmcopt.toml > out_vmcopt 2> out_vmcopt.e # w/o MPI on CPU
% mpirun -np 4 jqmc vmcopt.toml > out_vmcopt 2> out_vmcopt.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc vmcopt.toml > out_vmcopt 2> out_vmcopt.e # w/ MPI on GPU, depending the queueing system.
```

You can see the outcome using `jqmc-tool`.

```bash
% jqmc-tool vmcopt analyze-output out_vmcopt

------------------------------------------------------
Iter     E (Ha)     Max f (Ha)   Max of signal to noise of f
------------------------------------------------------
   1  -16.5743(97)  +1.132(12)   110.335
   2  -16.5921(96)  +1.097(12)   109.386
   3  -16.6117(95)  +1.084(12)   104.849
   4  -16.6399(93)  +1.059(12)   104.245
   5  -16.6678(91)  +1.029(12)   102.269
   6  -16.6819(90)  +1.009(12)   100.122
   7  -16.7028(90)  +0.993(12)    97.718
   8  -16.6974(87)  +0.963(12)    96.040
   9  -16.7200(87)  +0.948(11)    94.616
  10  -16.7511(87)  +0.914(11)    91.563
  11  -16.7602(85)  +0.895(11)    90.790
  12  -16.7714(85)  +0.878(11)    88.758
  13  -16.7867(85)  +0.848(10)    87.979
  14  -16.7940(86)  +0.835(11)    83.253
  15  -16.8065(83)  +0.787(10)    82.875
  16  -16.8112(83)  +0.777(10)    81.196
  17  -16.8284(82)  +0.741(10)    80.058
  18  -16.8327(83)  +0.743(10)    76.214
------------------------------------------------------
```

The important criteria are `Max f` and `Max of signal to noise of f`. `Max f` should be zero within the error bar. A practical criterion for the `signal to noise` is < 4~5 because it means that all the residual forces are zero in the statistical sense.

You can also plot them and make a figure.

```bash
% jqmc-tool vmcopt analyze-output out_vmcopt -p -s vmcopt.jpg
```

![VMC optimization](03vmcopt_JSD/vmcopt.jpg)

You should gain the energy with respect the the JSD value; otherwise, the optimization went wrong.