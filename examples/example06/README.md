# example06

Binding energy of the water-water dimer with the JSD and JAGP anstaz. One can learn how to obtain the VMC and DMC (in the extrapolated limit) energies of the Water dimer, starting from scratch (i.e., DFT calculation), with cartesian GTOs, and one can see how the binding energy is improved by optimizing the nodal surface. The following calculation is for the water-water dimer. The water molecule fragments should also be computed to get the binding energy. In this example, the water-water dimer geometry is taken from the S22 dataset[^2006JURpccp].

[^2006JURpccp]: P. Jurecka, et al. Phys Chem Chem Phys, 8, 1985-1993 (2006) [https://doi.org/10.1039/B600027D](https://doi.org/10.1039/B600027D)

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
> Hamiltonian data is saved in hamiltonian_data.h5.
```

The generated `hamiltonian_data.h5` is a wavefunction file with the `jqmc` format. `-j2` specifies the initial value of the two-body Jastrow parameter and `-j3` specifies the basis set (`ao-xxx`:atomic orbital or `mo`:molecular orbital) for the three-body Jastrow part.

[!NOTE]
> The `-j3` option in `jqmc-tool` controls how the atomic orbital (AO) basis is partitioned by Gaussian exponent strength for each nucleus. When you specify `ao-small`, `jqmc-tool` sorts each atom’s contracted AOs by the exponent of the primitive shell with the largest coefficient, divides them into **three** equal groups, and retains only the central group (i.e., 1/3). Specifying `ao-medium` divides into **four** groups and keeps the **two** central groups (i.e., 2/4), while `ao-large` divides into **five** groups and keeps the **three** central groups (i.e., 3/5). Using `ao` or `ao-full` disables any partitioning and includes all AOs from the original dataset. Using `mo` includes all MOs. This mechanism lets you tailor the trade-off between computational cost and accuracy by focusing on the most representative subset of basis functions.

## Optimize a trial WF (VMC)
The next step is to optimize variational parameters included in the generated wavefunction. More in details, here, we optimize the two-body Jastrow parameter and the matrix elements of the three-body Jastrow parameter.

You can generate a template file for a VMC calculation using `jqmc-tool`. Please directly edit `mcmcopt.toml` if you want to change a parameter.

```bash
% jqmc-tool vmc generate-input -g
> Input file is generated: vmc.toml
```

```toml:vmc.toml
[control]
job_type = "vmc" # Specify the job type. "mcmc", "vmc", "lrdmc", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 1 # Number of walkers per MPI process
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
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
num_opt_steps = 200 # Number of optimization steps.
wf_dump_freq = 20 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
optimizer_kwargs = { method = "sr", delta = 0.01, epsilon = 0.001, cg_flag = true, cg_max_iter = 100000, cg_tol = 0.0001 } # SR optimizer configuration (method, step/regularization, and CG settings).
opt_J1_param = false
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

You can see the outcome using `jqmc-tool`.

```bash
% jqmc-tool vmc analyze-output out_vmc

------------------------------------------------------
Iter     E (Ha)     Max f (Ha)   Max of signal to noise of f
------------------------------------------------------
   1  -34.4508(14)  -0.262(19)    15.415
   2  -34.4517(14)  -0.245(25)    18.438
   3  -34.4513(13)  -0.221(11)    19.472
   4  -34.4524(13)  -0.238(34)    18.540
   5  -34.4520(14)   +0.30(26)    15.594
   6  -34.4547(13)  -0.218(14)    15.248
   7  -34.4555(13)  -0.186(22)    13.766
   8  -34.4592(13)  -0.146(13)    15.021
   9  -34.4566(13)  -0.156(21)    15.199
  10  -34.4569(13)   +0.57(60)    13.896
  ...
  91  -34.4647(13)   -0.14(12)     4.680
  92  -34.4657(13)   +0.17(18)     3.916
  93  -34.4653(12)   +0.16(12)     4.260
  94  -34.4651(12)   -0.15(13)     4.895
  95  -34.4670(12)   -0.13(14)     4.310
  96  -34.4641(13)   -0.74(73)     4.135
  97  -34.4666(12)   -0.12(10)     6.130
  98  -34.4670(12)  -0.071(77)     4.565
  99  -34.4637(13)  -0.121(76)     4.880
 100  -34.4661(12)   -0.14(12)     5.124
------------------------------------------------------
```

The important criteria are `Max f` and `Max of signal to noise of f`. `Max f` should be zero within the error bar. A practical criterion for the `signal to noise` is < 4~5 because it means that all the residual forces are zero in the statistical sense.

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

You can also plot them and save it.

```bash
% jqmc-tool vmc analyze-output out_vmc -p -s vmc_JSD.jpg
```

![VMC JSD optimization](03_S22_water_dimer/03vmc_JSD/vmc_JSD.jpg)

## Compute Energy (MCMC)
The next step is MCMC calculation. You can generate a template file for a MCMC calculation using `jqmc-tool`. Please directly edit `mcmc.toml` if you want to change a parameter.

```bash
% cd 04mcmc_JSD
% cp ../03vmc_JSD/hamiltonian_data_opt_step_200.h5 ./hamiltonian_data.h5
% jqmc-tool mcmc generate-input -g
> Input file is generated: mcmc.toml
```

```toml:mcmc.toml
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
```

The final step is to run the `jqmc` job w/ or w/o MPI on a CPU or GPU machine (via a job queueing system such as PBS).

```bash
% jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on GPU, depending the queueing system.
```

You may get `E = -34.45005 +- 0.000506 Ha` [MCMC]

> [!NOTE]
> We are going to discuss the sub kcal/mol accuracy in the binding energy. So, we need to decrease the error bars of the mononer and dimer calculations up to $\sim$ 0.10 mHa and $\sim$ 0.15 mHa.

## Compute Energy (LRDMC)
The final step is LRDMC calculation. You can generate a template file for a LRDMC calculation using `jqmc-tool`. Please directly edit `lrdmc.toml` if you want to change a parameter.

```bash
% cd 05lrdmc_JSD
% cp ../04mcmc_JSD/hamiltonian_data.h5 ./hamiltonian_data.h5
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
  hamiltonian_h5 = '../hamiltonian_data.h5'
  verbosity = 'low'

[lrdmc]
  num_mcmc_steps  = 40000
  num_mcmc_per_measurement = 30
  alat = 0.20
  non_local_move = "dltmove"
  num_gfmc_warmup_steps = 50
  num_gfmc_bin_blocks = 50
  num_gfmc_collect_steps = 20
  E_scf = -34.00
```

LRDMC energy is biased with the discretized lattice space ($a$) by $O(a^2)$. It means that, to get an unbiased energy, one should compute LRDMC energies with several lattice parameters ($a$) extrapolate them into $a \rightarrow 0$. However, in this benchmark, we simply choose $a = 0.20$ Bohr because the error canceltion might work for the binding energy calculation. The final step is to run the `jqmc` jobs with the given $a$, e.g.

```bash
% jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e
```

You may get `E = -34.49139 +- 0.000651 Ha` [LRDMC with a = 0.2].

> [!NOTE]
> We are going to discuss the sub kcal/mol accuracy in the binding energy. So, we need to decrease the error bars of the mononer and dimer calculations up to $\sim$ 0.10 mHa and $\sim$ 0.15 mHa.

Your total energies of the water-water dimer are:

| Ansatz     | Method                  | Total energy (Ha)       |  ref      |
|------------|-------------------------|-------------------------|-----------|
| JSD        | VMC                     | -34.45005 +- 0.000506   | this work |
| JSD        | LRDMC ($a = 0.2$)       | -34.49139 +- 0.000651   | this work |


Your binding energies are:

| Ansatz     | Method                  | Binding energy (kcal/mol)  |  ref                 |
|------------|-------------------------|----------------------------|----------------------|
| JSD        | VMC                     | -5.1 +- 0.4                | this work            |
| JSD        | LRDMC ($a = 0.2$)       | -5.1 +- 0.5                | this work            |
| JSD        | VMC                     | -4.61 +- 0.05              | Zen et al.[^2015ZEN] |
| JSD        | LRDMC ($a = 0.2$)       | -4.94 +- 0.07              | Zen et al.[^2015ZEN] |

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


## Optimize a trial WF (VMC)
The next step is to optimize variational parameters included in the generated wavefunction. More in details, here, we optimize the two-body Jastrow parameter and the matrix elements of the three-body Jastrow parameter and the AGP matrix elements!

You can generate a template file for a VMC calculation using `jqmc-tool`. Please directly edit `vmc.toml` if you want to change a parameter.

```bash
% cd 07vmc_JAGP
% cp ../06convert_JSD_to_JAGP/hamiltonian_data_JAGP.h5 ./hamiltonian_data.h5
% jqmc-tool vmc generate-input -g
> Input file is generated: vmc.toml
```

```toml:vmc.toml
[control]
job_type = "vmc" # Specify the job type. "mcmc", "vmc", "lrdmc", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 1 # Number of walkers per MPI process
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
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.05 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
num_opt_steps = 200 # Number of optimization steps.
wf_dump_freq = 20 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
optimizer_kwargs = { method = "sr", delta = 0.01, epsilon = 0.001, cg_flag = true, cg_max_iter = 100000, cg_tol = 0.0001 } # SR optimizer configuration (method, step/regularization, and CG settings).
opt_J1_param = false
opt_J2_param = true
opt_J3_param = true
opt_lambda_param = true
num_param_opt = 0 # the number of parameters to optimize in the descending order of |f|/|std f|. If it is set 0, all parameters are optimized.
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
   1  -34.4508(14)  -0.262(19)    15.415
   2  -34.4517(14)  -0.245(25)    18.438
   3  -34.4513(13)  -0.221(11)    19.472
   4  -34.4524(13)  -0.238(34)    18.540
   5  -34.4520(14)   +0.30(26)    15.594
   6  -34.4547(13)  -0.218(14)    15.248
   7  -34.4555(13)  -0.186(22)    13.766
   8  -34.4592(13)  -0.146(13)    15.021
   9  -34.4566(13)  -0.156(21)    15.199
  10  -34.4569(13)   +0.57(60)    13.896
  ...
  90  -34.4643(12)   -0.16(16)     4.384
  91  -34.4647(13)   -0.14(12)     4.680
  92  -34.4657(13)   +0.17(18)     3.916
  93  -34.4653(12)   +0.16(12)     4.260
  94  -34.4651(12)   -0.15(13)     4.895
  95  -34.4670(12)   -0.13(14)     4.310
  96  -34.4641(13)   -0.74(73)     4.135
  97  -34.4666(12)   -0.12(10)     6.130
  98  -34.4670(12)  -0.071(77)     4.565
  99  -34.4637(13)  -0.121(76)     4.880
 100  -34.4661(12)   -0.14(12)     5.124
------------------------------------------------------
```

The important criteria are `Max f` and `Max of signal to noise of f`. Again, a practical criterion for the `signal to noise` is < 4~5 because it means that all the residual forces are zero in the statistical sense. However, `Max f` is different from the Jastrow optimization above. Despite the signal-to-noise ratio approaching below 4, unlike the Jastrow factor optimization, the `Max f` remains with a large error bar rather than driving it toward zero. The determinant part modifies the nodal surface of the wave function, and its parameter derivatives are known to *diverge* near those nodes. As a result, when one Monte Carlo samples the energy derivative $F \equiv -\cfrac{\partial E}{\partial c_{\rm det}}$ divergences appear, leading to the so-called infinite-variance problem. To address this, techniques such as reweighting[^2008ATT][^2015UMR] and regularization[^2020PAT] have been developed. jQMC implements the reweighting scheme invented by Attaccalite and Sorella[^2008ATT]. However, as Pathak and Wagner have shown, when the wave function (i.e., the nodal surface) becomes sufficiently complex, even these reweighting or regularization procedures cannot completely remove all divergent contributions[^2020PAT]. Because the variance of the derivatives exhibits the so-called *fat tail* behavior, it is inherently difficult to eliminate every divergence encountered during Monte Carlo sampling. Nevertheless, Pathak and Wagner also report that these remaining divergences are effectively masked in optimizations of the wave function in practice[^2020PAT], so they do not pose a serious issue in applications. Therefore, once the `signal-to-noise` ratio has converged to a satisfactory level (< 4), one may regard the optimization as effectively converged.

[^2008ATT]: C. Attaccalite and S. Sorella, *Phys. Rev. Lett.* **100**, 114501 (2008).

[^2015UMR]: C. J. Umrigar, *J. Chem. Phys.* **143**, 164105 (2015).

[^2020PAT]: S. Pathak and L. K. Wagner, *AIP Advances* **10**, 085213 (2020).

You can also plot them and save it.

```bash
% jqmc-tool vmc analyze-output out_vmc -p -s vmc_JAGP.jpg
```

![VMC JSD optimization](03_S22_water_dimer/07vmc_JAGP/vmc_JAGP.jpg)

## Compute Energy (MCMC)
The next step is MCMC calculation. You can generate a template file for a MCMC calculation using `jqmc-tool`. Please directly edit `mcmc.toml` if you want to change a parameter.

```bash
% cd 08mcmc_JAGP
% cp ../07vmc_JAGP/hamiltonian_data_opt_step_200.h5 ./hamiltonian_data.h5
% jqmc-tool mcmc generate-input -g
> Input file is generated: mcmc.toml
```

```toml:mcmc.toml
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
```

The final step is to run the `jqmc` job w/ or w/o MPI on a CPU or GPU machine (via a job queueing system such as PBS).

```bash
% jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on GPU, depending the queueing system.
```

You may get `E = -34.46554 +- 0.000476 Ha` [MCMC]

You should gain the energy with respect the the JSD value; otherwise, the optimization went wrong.

## Compute Energy (LRDMC)
The final step is LRDMC calculation. You can generate a template file for a LRDMC calculation using `jqmc-tool`. Please directly edit `lrdmc.toml` if you want to change a parameter.

```bash
% cd 09lrdmc_JAGP
% cp ../08mcmc_JAGP/hamiltonian_data.h5 ./hamiltonian_data.h5
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
  hamiltonian_h5 = '../hamiltonian_data.h5'
  verbosity = 'low'

[lrdmc]
  num_mcmc_steps  = 40000
  num_mcmc_per_measurement = 30
  alat = 0.20
  non_local_move = "dltmove"
  num_gfmc_warmup_steps = 50
  num_gfmc_bin_blocks = 50
  num_gfmc_collect_steps = 20
  E_scf = -17.00
```

LRDMC energy is biased with the discretized lattice space ($a$) by $O(a^2)$. It means that, to get an unbiased energy, one should compute LRDMC energies with several lattice parameters ($a$) extrapolate them into $a \rightarrow 0$. However, in this benchmark, we simply choose $a = 0.20$ Bohr because the error canceltion might work for the binding energy calculation. The final step is to run the `jqmc` jobs with the given $a$, e.g.

```bash
% jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e
```

You may get `E = -34.49444 +- 0.000529 Ha` [LRDMC with a = 0.2]

You should gain the energy with respect the the JSD value; otherwise, the optimization went wrong.

> [!NOTE]
> We are going to discuss the sub kcal/mol accuracy in the binding energy. So, we need to decrease the error bars of the mononer and dimer calculations up to $\sim$ 0.10 mHa and $\sim$ 0.15 mHa.

Your total energies of the water-water dimer are:

| Ansatz     | Method                  | Total energy (Ha)       |  ref      |
|------------|-------------------------|-------------------------|-----------|
| JSD        | VMC                     | -34.45005 +- 0.000506   | this work |
| JAGPs      | VMC                     | -34.46554 +- 0.000476   | this work |
| JSD        | LRDMC ($a = 0.2$)       | -34.49139 +- 0.000651   | this work |
| JAGPs      | LRDMC ($a = 0.2$)       | -34.49444 +- 0.000529   | this work |


Your binding energies are:

| Ansatz     | Method                  | Binding energy (kcal/mol)  |  ref                 |
|------------|-------------------------|----------------------------|----------------------|
| JSD        | VMC                     | -5.1 +- 0.4                | this work            |
| JSD        | VMC                     | -4.61 +- 0.05              | Zen et al.[^2015ZEN] |
| JAGPs      | VMC                     | -3.9 +- 0.4                | this work            |
| JAGPs      | VMC                     | -4.17 +- 0.1               | Zen et al.[^2015ZEN] |
| JSD        | LRDMC ($a = 0.2$)       | -5.1 +- 0.5                | this work            |
| JSD        | LRDMC ($a = 0.2$)       | -4.94 +- 0.07              | Zen et al.[^2015ZEN] |
| JAGPs      | LRDMC ($a = 0.2$)       | -4.9 +- 0.4                | this work            |
| JAGPs      | LRDMC ($a = 0.2$)       | -4.88 +- 0.06              | Zen et al.[^2015ZEN] |

[^2015ZEN]: A. Zen et al. J. Chem. Phys. 142, 144111 (2015) [https://doi.org/10.1063/1.4917171](https://doi.org/10.1063/1.4917171)
