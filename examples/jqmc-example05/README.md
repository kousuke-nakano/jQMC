# jqmc-example05:

Energy and atomic force of hydrogen molecule ($R = 0.74\;\text{\AA}$) with cartesian GTOs. All electron calculations. Comparison of JSD and JAGP ansatz. The atomic forces are computed by fully exploiting algorithmic differentiation (AD) as implemented in **JAX**. The pioneering application of AD in ab initio QMC was first introduced by S. Sorella and L. Capriotti in 2010 [^2010SORjcp].

## Generate a trial WF

The first step of ab-initio QMC is to generate a trial WF by a mean-field theory such as DFT/HF. `jQMC` interfaces with other DFT/HF software packages via `TREXIO`.

One of the easiest ways to produce it is using `pySCF` as a converter to the `TREXIO` format is implemented. The following is a script to run a DFT-LDA calculation of the hydrogen molecule at $R = 0.74\;\text{\AA}$ and dump it as a `TREXIO` file.

```bash
% cd 01DFT
```

```python:run_pyscf.py
from pyscf import gto, scf
from pyscf.tools import trexio

R = 0.74  # angstrom
filename = f"H2_R_{R:.2f}.h5"

mol = gto.Mole()
mol.verbose = 5
mol.atom = f"""
H    0.000000000   0.000000000  {-R / 2}
H    0.000000000   0.000000000  {+R / 2}
"""
mol.basis = "ccpvtz"
mol.unit = "A"
mol.ecp = None
mol.charge = 0
mol.spin = 0
mol.symmetry = False
mol.cart = True
mol.output = f"H2_R_{R:.2f}.out"
mol.build()

mf = scf.KS(mol).density_fit()
mf.max_cycle = 200
mf.xc = "LDA_X,LDA_C_PZ"
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
```

Launch it on a terminal. You may get `E = -1.13700890749411 Ha` [DFT-LDA-PZ].

```bash
% python run_pyscf.py
% cd ..
```

## Convert TREXIO to jQMC format and optimize a trial WF: JSD (VMC)

Next, convert the `TREXIO` file to the `jqmc` format using `jqmc-tool`, and then optimize the variational parameters in the Jastrow factor (J1, J2, and J3).

```bash
% cd 02vmc_JSD
% cp ../01DFT/H2_R_0.74.h5 .
% jqmc-tool trexio convert-to H2_R_0.74.h5 -j1 1.0 -j2 1.0 -j3 mo
> Hamiltonian data is saved in hamiltonian_data.h5.
```

The generated `hamiltonian_data.h5` is a wavefunction file with the `jqmc` format. `-j1` specifies the initial value of the one-body Jastrow parameter, `-j2` specifies the initial value of the two-body Jastrow parameter, and `-j3` specifies the basis set (`ao`:atomic orbital or `mo`:molecular orbital) for the three-body Jastrow part.

You can generate a template file for a VMC optimization using `jqmc-tool`. Please directly edit `vmc.toml` if you want to change a parameter.

```bash
% jqmc-tool vmc generate-input -g
> Input file is generated: vmc.toml
```

<!-- include: 02vmc_JSD/vmc.toml -->
```toml
[control]
job_type = "vmcopt" # Specify the job type. "vmc", "vmcopt", "lrdmc-bra", or "lrdmc-tau".
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
num_opt_steps = 300 # Number of optimization steps.
wf_dump_freq = 10 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
opt_J1_param = true
opt_J2_param = true
opt_J3_param = true
opt_lambda_param = false
opt_J3_basis_exp = false
opt_J3_basis_coeff = false
opt_lambda_basis_exp = false
opt_lambda_basis_coeff = false
```

Please launch the job.

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

The important criteria are `Max f` and `Max signal to noise of f`. `Max f` should be zero within the error bar. A practical criterion for `signal to noise` is < 4~5 because it means that all the residual forces are zero in the statistical sense.

```bash
% cd ..
```

## Compute Energy and Atomic forces: JSD (MCMC)

Using the optimized wavefunction, compute the energy and atomic forces via MCMC. Copy the optimized `hamiltonian_data` from the previous step and generate a template file using `jqmc-tool`. Please directly edit `mcmc.toml` if you want to change a parameter.

```bash
% cd 03mcmc_JSD
% cp ../02vmc_JSD/hamiltonian_data_opt_step_200.h5 ./hamiltonian_data.h5
% jqmc-tool mcmc generate-input -g
> Input file is generated: mcmc.toml
```

<!-- include: 03mcmc_JSD/mcmc.toml -->
```toml
[control]
job_type = "vmc" # Specify the job type. "vmc", "vmcopt", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_chk = "hamiltonian_data.chk" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"

[vmc]
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

You may get `E = -1.16986 +- 0.000079 Ha` and `Var(E) = 0.03025 +- 0.000071 Ha^2`.

```
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       -9(9)e-05    +6(9)e-05    +0.00311(22)
  H       +9(9)e-05    -6(9)e-05    -0.00311(22)
  ------------------------------------------------
```

> [!NOTE]
> If one were to optimize only the Jastrow factor while keeping the determinant part fixed to the DFT solution (i.e., `opt_with_projected_MOs = false`), the atomic forces would contain a self-consistency bias[^2021NAKjcp][^2022TIHjcp] because the DFT orbitals are not stationary with respect to the MCMC energy. In that case, a finite $F_z$ would appear even at the equilibrium geometry. By setting `opt_with_projected_MOs = true` as in this example, the MO coefficients are also optimized and this bias is eliminated.

```bash
% cd ..
```

## Compute Energy and Atomic forces: JSD (LRDMC)

Using the same optimized wavefunction, compute the energy and atomic forces via LRDMC. Please directly edit `lrdmc.toml` if you want to change a parameter.

```bash
% cd 04lrdmc_JSD
% cp ../02vmc_JSD/hamiltonian_data_opt_step_200.h5 ./hamiltonian_data.h5
% jqmc-tool lrdmc generate-input -g
> Input file is generated: lrdmc.toml
```

<!-- include: 04lrdmc_JSD/lrdmc.toml -->
```toml
[control]
job_type = "lrdmc-bra" # Specify the job type. "vmc", "vmcopt", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_chk = "hamiltonian_data.chk" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"

[lrdmc-bra]
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

You may get `E = -1.17485 +- 0.000248 Ha` and `Var(E) = 0.02985 +- 0.000165 Ha^2`.

```
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       -0.0009(6)  +0.0006(7)  -0.0066(8)
  H       +0.0009(6)  -0.0006(7)  +0.0066(8)
  ------------------------------------------------
```

> [!NOTE]
> The LRDMC forces are intrinsically biased because the so-called Reynolds approximation[^1989REYijqc] is employed. See benchmark papers[^2021NAKjcp] [^2022TIHjcp].

```bash
% cd ..
```

## Convert JSD to JAGP and optimize: JAGP (VMC)

Convert the optimized JSD ansatz to the JAGP ansatz using `jqmc-tool`, and then optimize all variational parameters including the geminal (lambda) parameters.

```bash
% cd 05vmc_JAGP
% cp ../02vmc_JSD/hamiltonian_data_opt_step_200.h5 ./hamiltonian_data_JSD.h5
% jqmc-tool hamiltonian conv-wf --convert-to jagp hamiltonian_data_JSD.h5
> Convert SD to AGP.
> Hamiltonian data is saved in hamiltonian_data_conv.h5.
% mv hamiltonian_data_conv.h5 hamiltonian_data.h5
```

Generate a template file for VMC optimization. Please directly edit `vmc.toml` if you want to change a parameter.

```bash
% jqmc-tool vmc generate-input -g
> Input file is generated: vmc.toml
```

<!-- include: 05vmc_JAGP/vmc.toml -->
```toml
[control]
job_type = "vmcopt" # Specify the job type. "vmc", "vmcopt", "lrdmc-bra", or "lrdmc-tau".
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
num_mcmc_bin_blocks = 1 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 1.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
num_opt_steps = 400 # Number of optimization steps.
wf_dump_freq = 10 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
opt_J1_param = true
opt_J2_param = true
opt_J3_param = true
opt_lambda_param = true
opt_J3_basis_exp = false
opt_J3_basis_coeff = false
opt_lambda_basis_exp = false
opt_lambda_basis_coeff = false
```

Please launch the job.

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
 ...
  26  -1.17409(13)  +0.0010(10)     5.080
  27  -1.17407(15)  -0.0020(10)     4.100
  28  -1.17414(14)  -0.0010(10)     4.578
  29  -1.17368(14)  -0.0002(00)     4.743
  30  -1.17407(15)  +0.0010(10)     3.801
 ...
------------------------------------------------------
```

One should gain energy with respect to the JSD ansatz. Note that `opt_lambda_param = true` is set to optimize the geminal parameters in the JAGP ansatz.

```bash
% cd ..
```

## Compute Energy and Atomic forces: JAGP (MCMC)

Using the optimized JAGP wavefunction, compute the energy and atomic forces via MCMC.

```bash
% cd 06mcmc_JAGP
% cp ../05vmc_JAGP/hamiltonian_data_opt_step_200.h5 ./hamiltonian_data.h5
% jqmc-tool mcmc generate-input -g
> Input file is generated: mcmc.toml
```

<!-- include: 06mcmc_JAGP/mcmc.toml -->
```toml
[control]
job_type = "vmc" # Specify the job type. "vmc", "vmcopt", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_chk = "hamiltonian_data.chk" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"

[vmc]
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

You may get `E = -1.17543 +- 0.001343 Ha` and `Var(E) = 0.00327 +- 0.000475 Ha^2`.

```
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       +0.00015(5)  -3(5)e-05    -0.00042(20)
  H       -0.00015(5)  +3(5)e-05    +0.00042(20)
  ------------------------------------------------
```

```bash
% cd ..
```

## Compute Energy and Atomic forces: JAGP (LRDMC)

Using the same optimized JAGP wavefunction, compute the energy and atomic forces via LRDMC. Please directly edit `lrdmc.toml` if you want to change a parameter.

```bash
% cd 07lrdmc_JAGP
% cp ../05vmc_JAGP/hamiltonian_data_opt_step_200.h5 ./hamiltonian_data.h5
% jqmc-tool lrdmc generate-input -g
> Input file is generated: lrdmc.toml
```

<!-- include: 07lrdmc_JAGP/lrdmc.toml -->
```toml
[control]
job_type = "lrdmc-bra" # Specify the job type. "vmc", "vmcopt", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_chk = "hamiltonian_data.chk" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"

[lrdmc-bra]
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

You may get `E = -1.17442 +- 0.000069 Ha` and `Var(E) = 0.00287 +- 0.000010 Ha^2`.

```
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       -0.0007(6)  +0.0009(10) -0.0058(10)
  H       +0.0007(6)  -0.0009(10) +0.0058(10)
  ------------------------------------------------
```

> [!NOTE]
> The LRDMC forces are intrinsically biased because the so-called Reynolds approximation[^1989REYijqc] is employed. See benchmark papers[^2021NAKjcp] [^2022TIHjcp].

```bash
% cd ..
```

## Summary

Results for H$_2$ at $R = 0.74\;\text{\AA}$ with cc-pVTZ basis set (all electron):

| Ansatz | Method | E (Ha) | Fz (Ha/bohr) |
|--------|--------|--------|---------------|
| JSD    | MCMC   | -1.16986(8)  | +0.00311(22) |
| JSD    | LRDMC  | -1.17485(25) | -0.0066(8)   |
| JAGP   | MCMC   | -1.17543(134)| -0.00042(20) |
| JAGP   | LRDMC  | -1.17442(7)  | -0.0058(10)  |

In both the JSD and JAGP calculations, the MO coefficients are optimized (`opt_with_projected_MOs = true` for JSD, `opt_lambda_param = true` for JAGP), so the self-consistency bias[^2021NAKjcp][^2022TIHjcp] is eliminated. At the equilibrium geometry ($R = 0.74\;\text{\AA}$), the force should be nearly zero, and indeed both the JSD and JAGP MCMC results show $F_z$ consistent with zero within the error bar.

If one were to keep the determinant part fixed to the DFT solution (i.e., `opt_with_projected_MOs = false`), the DFT orbitals would not be stationary with respect to the MCMC energy, and a finite self-consistency bias would appear in the atomic forces[^2024SLOjctc]. An alternative way to correct for this bias without full orbital optimization is described in Ref. [^2024NAKprb]. If one is interested in the latter approach, please contact a `jQMC` developer.

[^2010SORjcp]: S. Sorella and L. Capriotti, J. Chem. Phys. **133** 234111 (2010) [https://doi.org/10.1063/1.3516208](https://doi.org/10.1063/1.3516208)
[^2021NAKjcp]: K. Nakano et al. J. Chem. Phys. **154**, 204111 (2021), [https://doi.org/10.1063/5.0076302](https://doi.org/10.1063/5.0076302)
[^2022TIHjcp]: J. Tiihonen et al. J. Chem. Phys. **156**, 034101 (2022), [https://doi.org/10.1063/5.0052266](https://doi.org/10.1063/5.0052266)
[^1989REYijqc]: P.J. Reynolds et al. Int. J. Quantum Chem. **29**, 589 (1986). [https://doi.org/10.1063/5.0052266](https://doi.org/10.1063/5.0052266)
[^2024SLOjctc]: E. Slootman et al. J. Chem. Theory Comput. **20**, 6020-6027 (2024), [https://doi.org/10.1021/acs.jctc.4c00498](https://doi.org/10.1021/acs.jctc.4c00498)
[^2024NAKprb]: K. Nakano et al. Phys. Rev. B **109**, 205151 (2024), [https://doi.org/10.1103/PhysRevB.109.205151](https://doi.org/10.1103/PhysRevB.109.205151)
