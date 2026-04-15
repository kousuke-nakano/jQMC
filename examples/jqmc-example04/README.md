# jqmc-example04:

Binding energy of the water-water dimer with the JSD and JAGP ansatz. One can learn how to obtain the VMC and LRDMC energies of the water dimer, starting from scratch (i.e., DFT calculation by `pySCF`), with cartesian GTOs, and one can see how the binding energy is improved by optimizing the nodal surface (i.e., JAGP). The water-water dimer geometry is taken from the S22 dataset[^2006JURpccp].

[^2006JURpccp]: P. Jurecka, et al. Phys Chem Chem Phys, 8, 1985-1993 (2006) [https://doi.org/10.1039/B600027D](https://doi.org/10.1039/B600027D)

## Setup

This example involves three separate calculations: two water monomers and the dimer. Each follows the same workflow (DFT --> VMC --> MCMC --> LRDMC). We demonstrate the full workflow for the **dimer** below; repeat the same steps for each monomer by changing the geometry.

Create a working directory and move into it.

```bash
% mkdir water_dimer_qmc && cd water_dimer_qmc
```

The directory structure will look like:

```
water_dimer_qmc/
├── 01_S22_water_monomer_1/   # monomer 1
│   └── 01DFT/
├── 02_S22_water_monomer_2/   # monomer 2
│   └── 01DFT/
└── 03_S22_water_dimer/       # dimer
    ├── 01DFT/
    ├── 02vmc_JSD/
    ├── 03mcmc_JSD/
    ├── 04lrdmc_JSD/
    ├── 05vmc_JAGP/
    ├── 06mcmc_JAGP/
    └── 07lrdmc_JAGP/
```

## Generate trial WFs (DFT)

The first step of ab-initio QMC is to generate a trial WF by a mean-field theory such as DFT/HF. `jQMC` interfaces with other DFT/HF software packages via `TREX-IO`.

> [!NOTE]
> This `TREX-IO` converter is being develped in the `pySCF-forge` [repository](https://github.com/pyscf/pyscf-forge) and not yet merged to the main repository of `pySCF`. Please use `pySCF-forge`.

### Water dimer

<!-- include: 03_S22_water_dimer/01DFT/run_pyscf.py -->
```python
from pyscf import gto, scf
from pyscf.tools import trexio

filename = f"water_dimer.h5"

mol = gto.Mole()
mol.verbose = 5
mol.atom = f"""
	    O  -1.551007  -0.114520   0.000000
	    H  -1.934259   0.762503   0.000000
	    H  -0.599677   0.040712   0.000000
	    O   1.350625   0.111469   0.000000
	    H   1.680398  -0.373741  -0.758561
	    H   1.680398  -0.373741   0.758561
"""
mol.basis = "ccecp-aug-ccpvtz"
mol.unit = "A"
mol.ecp = "ccecp"
mol.charge = 0
mol.spin = 0
mol.symmetry = False
mol.cart = True
mol.output = f"water_dimer.out"
mol.build()

mf = scf.KS(mol).density_fit()
mf.max_cycle = 200
mf.xc = "LDA_X,LDA_C_PZ"
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
```

Launch it on a terminal. You may get `E = -34.3124355699676 Ha` [LDA].

```bash
% mkdir -p 03_S22_water_dimer/01DFT && cd 03_S22_water_dimer/01DFT
% python run_pyscf.py
% cd ../..
```

### Water monomer 1

<!-- 01_S22_water_monomer_1/01DFT/run_pyscf.py -->
```python
from pyscf import gto, scf
from pyscf.tools import trexio

filename = f'water_monomer_1.h5'

mol = gto.Mole()
mol.verbose  = 5
mol.atom     = f'''
	    O  -1.551007  -0.114520   0.000000
	    H  -1.934259   0.762503   0.000000
	    H  -0.599677   0.040712   0.000000
'''
mol.basis    = 'ccecp-aug-ccpvtz'
mol.unit     = 'A'
mol.ecp      = 'ccecp'
mol.charge   = 0
mol.spin     = 0
mol.symmetry = False
mol.cart = True
mol.output   = f'water_monomer_1.out'
mol.build()

mf = scf.KS(mol).density_fit()
mf.max_cycle=200
mf.xc = 'LDA_X,LDA_C_PZ'
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
```

```bash
% mkdir -p 01_S22_water_monomer_1/01DFT && cd 01_S22_water_monomer_1/01DFT
% python run_pyscf.py
% cd ../..
```

### Water monomer 2

<!-- 02_S22_water_monomer_2/01DFT/run_pyscf.py -->
```python
from pyscf import gto, scf
from pyscf.tools import trexio

filename = f'water_monomer_2.h5'

mol = gto.Mole()
mol.verbose  = 5
mol.atom     = f'''
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
mol.output   = f'water_monomer_2.out'
mol.build()

mf = scf.KS(mol).density_fit()
mf.max_cycle=200
mf.xc = 'LDA_X,LDA_C_PZ'
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
```

```bash
% mkdir -p 02_S22_water_monomer_2/01DFT && cd 02_S22_water_monomer_2/01DFT
% python run_pyscf.py
% cd ../..
```

> [!NOTE]
> One can start from any HF/DFT code that can dump `TREX-IO` file. See the [TREX-IO website](https://github.com/TREX-CoE/trexio) for the detail.

---

In the following sections, we demonstrate the full workflow for the **water dimer** (`03_S22_water_dimer`). The same steps apply to both monomers -- simply adjust the TREXIO filename and working directory accordingly.

```bash
% cd 03_S22_water_dimer
```

## Optimize a trial WF (VMC) -- JSD
The next step is to optimize variational parameters included in the generated wavefunction. Here, we optimize the two-body Jastrow parameter and the matrix elements of the three-body Jastrow parameter.

Create a directory for the VMC optimization and move into it. Then generate a template file using `jqmc-tool`. Please directly edit `vmc.toml` if you want to change a parameter.

```bash
% mkdir 02vmc_JSD && cd 02vmc_JSD
% cp ../01DFT/water_dimer.h5 .
% jqmc-tool trexio convert-to water_dimer.h5 -j2 1.0 -j3 ao-medium
> Hamiltonian data is saved in hamiltonian_data.h5.
% jqmc-tool vmc generate-input -g
> Input file is generated: vmc.toml
```

The generated `hamiltonian_data.h5` is a wavefunction file with the `jqmc` format. `-j2` specifies the initial value of the two-body Jastrow parameter and `-j3` specifies the basis set (`ao-xxx`:atomic orbital or `mo`:molecular orbital) for the three-body Jastrow part.

> [!NOTE]
> The `-j3` option in `jqmc-tool` controls how the atomic orbital (AO) basis is partitioned by Gaussian exponent strength for each nucleus. `ao-small` selects a compact subset, `ao-medium` a moderate one, and `ao-large` a larger one. Using `ao` or `ao-full` disables any partitioning and includes all AOs. Using `mo` includes all MOs. Please refer to the command-line reference for the available options.

<!-- include: 03_S22_water_dimer/02vmc_JSD/vmc.toml -->
```toml
[control]
job_type = "vmc" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 1 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
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
optimizer_kwargs = { method = "sr", delta = 0.15, epsilon = 0.001, cg_flag = true, cg_max_iter = 10000, cg_tol = 1e-6, use_lm = true } # SR optimizer configuration (method plus step/regularization).
opt_J1_param = false
opt_J2_param = true
opt_J3_param = true
opt_JNN_param = false
opt_lambda_param = false
opt_with_projected_MOs = false
opt_J3_basis_exp = false
opt_J3_basis_coeff = false
opt_lambda_basis_exp = false
opt_lambda_basis_coeff = false
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

> [!TIP]
> If the optimization does not converge well, try the following:
> - Adjust the `delta` parameter in `optimizer_kwargs`. A smaller `delta` (e.g., `0.05`) makes the optimization more conservative but stable, while a larger one (e.g., `0.30`) is more aggressive but may cause instabilities.
> - Set `adaptive_learning_rate = false` in `optimizer_kwargs` to disable the adaptive learning rate and use a fixed step size instead. This can sometimes improve convergence for difficult cases.

You can also plot them and make a figure.

```bash
% jqmc-tool vmc analyze-output out_vmc -p -s vmc_JSD.jpg
```

![VMC JSD optimization](03_S22_water_dimer/02vmc_JSD/vmc_JSD.jpg)

If the optimization is not converged. You can restart the optimization.

```toml:vmc.toml
[control]
...
restart = true
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
...
```

```bash
% jqmc vmc.toml > out_vmc_cont 2> out_vmc_cont.e
```

You can see and plot the outcome using `jqmc-tool`.

```bash
% jqmc-tool vmc analyze-output out_vmc out_vmc_cont
```

## Compute Energy (MCMC) -- JSD
The next step is MCMC calculation. Create a directory for the MCMC calculation and move into it. Then generate a template file using `jqmc-tool`. Please directly edit `mcmc.toml` if you want to change a parameter.

```bash
% cd ..
% mkdir 03mcmc_JSD && cd 03mcmc_JSD
% cp ../02vmc_JSD/hamiltonian_data_opt_step_200.h5 ./hamiltonian_data.h5  # use the optimized WF
% jqmc-tool mcmc generate-input -g
> Input file is generated: mcmc.toml
```

<!-- include: 03_S22_water_dimer/03mcmc_JSD/mcmc.toml -->
```toml
[control]
job_type = "mcmc" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau"
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 300 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
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

Run the `jqmc` job w/ or w/o MPI on a CPU or GPU machine (via a job queueing system such as PBS).

```bash
% jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on GPU, depending the queueing system.
```

You may get `E = -34.45005 +- 0.000506 Ha` [MCMC]

> [!NOTE]
> We are going to discuss the sub kcal/mol accuracy in the binding energy. So, we need to decrease the error bars of the monomer and dimer calculations down to $\sim$ 0.10 mHa and $\sim$ 0.15 mHa, respectively.

## Compute Energy (LRDMC) -- JSD
The next step is LRDMC calculation. Create a directory for the LRDMC calculation and move into it. Then generate a template file using `jqmc-tool`. Please directly edit `lrdmc.toml` if you want to change a parameter.

```bash
% cd ..
% mkdir -p 04lrdmc_JSD/alat_0.20 && cd 04lrdmc_JSD
% cp ../03mcmc_JSD/hamiltonian_data.h5 .
% cd alat_0.20
% jqmc-tool lrdmc generate-input -g
> Input file is generated: lrdmc.toml
```

<!-- include: 03_S22_water_dimer/04lrdmc_JSD/lrdmc.toml -->
```toml
[control]
  job_type = 'lrdmc-bra'
  mcmc_seed = 34467
  number_of_walkers = 300
  max_time = 10400
  restart = false
  restart_chk = 'restart.h5'
  hamiltonian_h5 = '../hamiltonian_data.h5'
  verbosity = 'low'

[lrdmc-bra]
  num_mcmc_steps  = 40000
  num_mcmc_per_measurement = 30
  alat = 0.20
  non_local_move = "dltmove"
  num_gfmc_warmup_steps = 50
  num_gfmc_bin_blocks = 50
  num_gfmc_collect_steps = 20
  E_scf = -34.00
```

LRDMC energy is biased with the discretized lattice space ($a$) by $O(a^2)$. To get an unbiased energy, one should compute LRDMC energies with several lattice parameters ($a$) and extrapolate them into $a \rightarrow 0$. However, in this benchmark, we simply choose $a = 0.20$ Bohr because the error cancellation might work for the binding energy calculation.

```bash
% jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e
```

You may get `E = -34.49139 +- 0.000651 Ha` [LRDMC with a = 0.2].

> [!NOTE]
> We are going to discuss the sub kcal/mol accuracy in the binding energy. So, we need to decrease the error bars of the monomer and dimer calculations down to $\sim$ 0.10 mHa and $\sim$ 0.15 mHa, respectively.

Your total energies of the water-water dimer (JSD) are:

| Ansatz     | Method                  | Total energy (Ha)       |  ref      |
|------------|-------------------------|-------------------------|-----------|
| JSD        | VMC                     | -34.45005 +- 0.000506   | this work |
| JSD        | LRDMC ($a = 0.2$)       | -34.49139 +- 0.000651   | this work |


## Optimize a trial WF (VMC) -- JAGP
The next step is to convert the optimized JSD ansatz to JAGP and optimize it. The JAGP (Jastrow Antisymmetrized Geminal Power) ansatz goes beyond JSD by allowing the determinantal part to be variationally optimized, which can improve the nodal surface.

Create a directory for the VMC optimization, convert the JSD wavefunction to JAGP, and generate a template file.

```bash
% cd ..
% mkdir 05vmc_JAGP && cd 05vmc_JAGP
% cp ../03mcmc_JSD/hamiltonian_data.h5 ./hamiltonian_data_JSD.h5
% jqmc-tool hamiltonian conv-wf --convert-to jagp hamiltonian_data_JSD.h5
> Convert SD to AGP.
> Hamiltonian data is saved in hamiltonian_data_conv.h5.
% mv hamiltonian_data_conv.h5 hamiltonian_data.h5
% jqmc-tool vmc generate-input -g
> Input file is generated: vmc.toml
```

<!-- include: 03_S22_water_dimer/05vmc_JAGP/vmc.toml -->
```toml
[control]
job_type = "vmc" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 1 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
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
optimizer_kwargs = { method = "sr", delta = 0.15, epsilon = 0.001, cg_flag = true, cg_max_iter = 10000, cg_tol = 1e-6, use_lm = true } # SR optimizer configuration (method plus step/regularization).
opt_J1_param = false
opt_J2_param = true
opt_J3_param = true
opt_JNN_param = false
opt_lambda_param = true
opt_with_projected_MOs = false
opt_J3_basis_exp = false
opt_J3_basis_coeff = false
opt_lambda_basis_exp = false
opt_lambda_basis_coeff = false
```

> [!IMPORTANT]
> Note the key differences from the JSD optimization:
> - `opt_lambda_param = true` -- enables optimization of the AGP matrix elements (determinantal part).
> - `epsilon_AS = 0.05` -- enables the Attaccalite-Sorella regularization, which is important for stable optimization of the determinantal part.

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

The important criteria are `Max f` and `Max of signal to noise of f`. Again, a practical criterion for the `signal to noise` is < 4~5 because it means that all the residual forces are zero in the statistical sense. However, `Max f` behaves differently from the Jastrow-only optimization above. Despite the signal-to-noise ratio approaching below 4, unlike the Jastrow factor optimization, the `Max f` remains with a large error bar rather than driving it toward zero. The determinant part modifies the nodal surface of the wave function, and its parameter derivatives are known to *diverge* near those nodes. As a result, when one Monte Carlo samples the energy derivative $F \equiv -\cfrac{\partial E}{\partial c_{\rm det}}$ divergences appear, leading to the so-called infinite-variance problem. To address this, techniques such as reweighting[^2008ATT][^2015UMR] and regularization[^2020PAT] have been developed. jQMC implements the reweighting scheme invented by Attaccalite and Sorella[^2008ATT]. However, as Pathak and Wagner have shown, when the wave function (i.e., the nodal surface) becomes sufficiently complex, even these reweighting or regularization procedures cannot completely remove all divergent contributions[^2020PAT]. Because the variance of the derivatives exhibits the so-called *fat tail* behavior, it is inherently difficult to eliminate every divergence encountered during Monte Carlo sampling. Nevertheless, Pathak and Wagner also report that these remaining divergences are effectively masked in optimizations of the wave function in practice[^2020PAT], so they do not pose a serious issue in applications. Therefore, once the `signal-to-noise` ratio has converged to a satisfactory level (< 4), one may regard the optimization as effectively converged.

[^2008ATT]: C. Attaccalite and S. Sorella, *Phys. Rev. Lett.* **100**, 114501 (2008).

[^2015UMR]: C. J. Umrigar, *J. Chem. Phys.* **143**, 164105 (2015).

[^2020PAT]: S. Pathak and L. K. Wagner, *AIP Advances* **10**, 085213 (2020).

You can also plot them and make a figure.

```bash
% jqmc-tool vmc analyze-output out_vmc -p -s vmc_JAGP.jpg
```

![VMC JAGP optimization](03_S22_water_dimer/05vmc_JAGP/vmc_JAGP.jpg)

If the optimization is not converged. You can restart the optimization.

```toml:vmc.toml
[control]
...
restart = true
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
...
```

```bash
% jqmc vmc.toml > out_vmc_cont 2> out_vmc_cont.e
```

You can see and plot the outcome using `jqmc-tool`.

```bash
% jqmc-tool vmc analyze-output out_vmc out_vmc_cont
```

## Compute Energy (MCMC) -- JAGP
The next step is MCMC calculation. Create a directory for the MCMC calculation and move into it.

```bash
% cd ..
% mkdir 06mcmc_JAGP && cd 06mcmc_JAGP
% cp ../05vmc_JAGP/hamiltonian_data_opt_step_200.h5 ./hamiltonian_data.h5  # use the optimized WF
% jqmc-tool mcmc generate-input -g
> Input file is generated: mcmc.toml
```

<!-- include: 03_S22_water_dimer/06mcmc_JAGP/mcmc.toml -->
```toml
[control]
job_type = "mcmc" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau"
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 300 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
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

Run the `jqmc` job.

```bash
% jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on GPU, depending the queueing system.
```

You may get `E = -34.46554 +- 0.000476 Ha` [MCMC]

You should gain the energy with respect the JSD value; otherwise, the optimization went wrong.

## Compute Energy (LRDMC) -- JAGP
The final step is LRDMC calculation. Create a directory for the LRDMC calculation and move into it.

```bash
% cd ..
% mkdir -p 07lrdmc_JAGP/alat_0.20 && cd 07lrdmc_JAGP
% cp ../06mcmc_JAGP/hamiltonian_data.h5 .
% cd alat_0.20
% jqmc-tool lrdmc generate-input -g
> Input file is generated: lrdmc.toml
```

<!-- include: 03_S22_water_dimer/07lrdmc_JAGP/lrdmc.toml -->
```toml
[control]
  job_type = 'lrdmc-bra'
  mcmc_seed = 34467
  number_of_walkers = 300
  max_time = 10400
  restart = false
  restart_chk = 'restart.h5'
  hamiltonian_h5 = '../hamiltonian_data.h5'
  verbosity = 'low'

[lrdmc-bra]
  num_mcmc_steps  = 40000
  num_mcmc_per_measurement = 30
  alat = 0.20
  non_local_move = "dltmove"
  num_gfmc_warmup_steps = 50
  num_gfmc_bin_blocks = 50
  num_gfmc_collect_steps = 20
  E_scf = -34.00
```

```bash
% jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e
```

You may get `E = -34.49444 +- 0.000529 Ha` [LRDMC with a = 0.2]

You should gain the energy with respect the JSD value; otherwise, the optimization went wrong.

> [!NOTE]
> We are going to discuss the sub kcal/mol accuracy in the binding energy. So, we need to decrease the error bars of the monomer and dimer calculations down to $\sim$ 0.10 mHa and $\sim$ 0.15 mHa, respectively.

## Results

### Total energies

Your total energies of the water-water dimer are:

| Ansatz     | Method                  | Total energy (Ha)       |  ref      |
|------------|-------------------------|-------------------------|-----------|
| JSD        | VMC                     | -34.45005 +- 0.000506   | this work |
| JAGPs      | VMC                     | -34.46554 +- 0.000476   | this work |
| JSD        | LRDMC ($a = 0.2$)       | -34.49139 +- 0.000651   | this work |
| JAGPs      | LRDMC ($a = 0.2$)       | -34.49444 +- 0.000529   | this work |


### Binding energies

Your binding energies ($E_{\rm bind} = E_{\rm dimer} - E_{\rm monomer1} - E_{\rm monomer2}$) are:

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
