# example01

Total energy of water molecule. One can learn how to obtain the VMC and DMC (in the extrapolated limit) energies of the Water dimer, starting from scratch (i.e., DFT calculation), with cartesian GTOs.

## Generate a trial WF

The first step of ab-initio QMC is to generate a trial WF by a mean-field theory such as DFT/HF. `jQMC` interfaces with other DFT/HF software packages via `TREX-IO`.

One of the easiest ways to produce it is using `pySCF` as a converter to the `TREX-IO` format is implemented. Another way is using `CP2K`, where a converter to the `TREX-IO` format is implemented.

The following is a script to run a HF calculation for the water molecule using `pyscf-forge` and dump it as a `TREX-IO` file.

> [!NOTE]
> This `TREX-IO` converter is being develped in the `pySCF-forge` [repository](https://github.com/pyscf/pyscf-forge) and not yet merged to the main repository of `pySCF`. Please use `pySCF-forge`.

<!-- include: 01DFT/01pyscf-forge/run_pyscf.py -->
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

The following is a script to run a LDA calculation for the water molecule using `cp2k` and dump it as a `TREXIO` file.

<!-- include: 01DFT/02cp2k/water.xyz -->
```bash
3

O    5.00000000   7.14707700   7.65097100
H    4.06806600   6.94297500   7.56376100
H    5.38023700   6.89696300   6.80798400
```

<!-- include: 01DFT/02cp2k/water_ccecp_ccpvtz.inp -->
```bash
&global
  project      water_ccecp_ccpvtz
  print_level  medium
  run_type     energy
&end

&force_eval
  method quickstep

  &subsys
    &cell
      abc  15.0 15.0 15.0
      periodic none
    &end

    &topology
      coord_file_format  xyz
      coord_file_name    water.xyz
    &end

    &kind H
      basis_set  ccecp-cc-pVTZ
      potential  ecp ccECP
    &end

    &kind O
      basis_set  ccecp-cc-pVTZ
      potential  ecp ccECP
    &end
&end

  &dft
    multiplicity  1
    uks           false
    charge        0

    basis_set_file_name  ./basis.cp2k
    potential_file_name  ./ecp.cp2k

    &qs
      method gpw
      eps_default  1.0e-15
    &end

    &xc
      &xc_functional
        &lda_x
        &end
        &lda_c_pz
        &end
      &end
    &end

    &poisson
      psolver   wavelet
      periodic  none
    &end

    &mgrid
    cutoff 1000
    rel_cutoff 50
    &end

    &scf
     scf_guess  atomic
     eps_scf    1.0e-7
     max_scf    5
     eps_diis   0.1
     &ot on
        minimizer cg
        linesearch adapt
        preconditioner full_single_inverse
        max_scf_diis 50
     &end ot
     &outer_scf
        eps_scf    1.0e-7
        max_scf    3
     &end outer_scf
      &print
        &restart on
        &end
        &restart_history off
        &end
      &end
    &end

    &print
      &trexio
      &end
      &mo
        energies      true
        occnums       false
        cartesian     false
        coefficients  true
        &each
          qs_scf  0
        &end
      &end mo
      &overlap_condition on
        diagonalization   .true.
      &end overlap_condition
    &end print
  &end dft
&end
```

<!-- include: 01DFT/02cp2k/basis.cp2k -->
```bash
# ccecp-cc-pVTZ
# SOURCE: https://pseudopotentiallibrary.org/recipes/H/ccECP/H.cc-pVTZ.nwchem
# SOURCE: https://pseudopotentiallibrary.org/recipes/O/ccECP/O.cc-pVTZ.nwchem
####
H ccecp-cc-pVTZ
6
1  0  0  8  1
  23.843185  0.00411490
  10.212443  0.01046440
  4.374164  0.02801110
  1.873529  0.07588620
  0.802465  0.18210620
  0.343709  0.34852140
  0.147217  0.37823130
  0.063055  0.11642410
1  0  0  1  1
  0.091791  1.00000000
1  0  0  1  1
  0.287637  1.00000000
1  1  1  1  1
  0.393954  1.00000000
1  1  1  1  1
  1.462694  1.00000000
1  2  2  1  1
  1.065841  1.0000000
####
O ccecp-cc-pVTZ
9
2  0  0  9  1
  54.775216 -0.0012444
  25.616801 0.0107330
  11.980245 0.0018889
  6.992317 -0.1742537
  2.620277 0.0017622
  1.225429 0.3161846
  0.577797 0.4512023
  0.268022 0.3121534
  0.125346 0.0511167
2  0  0  1  1
  1.686633 1.0000000
2  0  0  1  1
  0.237997 1.0000000
2  1  1  9  1
  22.217266 0.0104866
  10.74755 0.0366435
  5.315785 0.0803674
  2.660761 0.1627010
  1.331816 0.2377791
  0.678626 0.2811422
  0.333673 0.2643189
  0.167017 0.1466014
  0.083598 0.0458145
2  1  1  1  1
  0.600621 1.0000000
2  1  1  1  1
  0.184696 1.0000000
3  2  2  1  1
  2.404278 1.0000000
3  2  2  1  1
  0.669340 1.0000000
4  3  3  1  1
  1.423104 1.0000000
```

<!-- include: 01DFT/02cp2k/ecp.cp2k -->
```bash
# ccecp-cc-pVTZ
# SOURCE: https://pseudopotentiallibrary.org/recipes/H/ccECP/H.ccECP.nwchem
# SOURCE: https://pseudopotentiallibrary.org/recipes/O/ccECP/O.ccECP.nwchem
ccECP
H nelec 0
H ul
1 21.24359508259891 1.00000000000000
3 21.24359508259891 21.24359508259891
2 21.77696655044365 -10.85192405303825
H S
2 1.000000000000000 0.00000000000000
O nelec 2
O ul
1 12.30997 6.000000
3 14.76962 73.85984
2 13.71419 -47.87600
O S
2 13.65512 85.86406
####
END ccECP
```

Launch it on a terminal. You may get `E = -17.146760756813901 Ha` [LDA].

```bash
% cp2k.ssmp -i water_ccecp_ccpvtz.inp -o water_ccecp_ccpvtz.out
% mv water_ccecp_ccpvtz-TREXIO.h5 water_ccecp_ccpvtz.h5
```

> [!NOTE]
> One can start from any HF/DFT code that can dump `TREX-IO` file. See the [TREX-IO website](https://github.com/TREX-CoE/trexio) for the detail.

Next step is to convert the `TREXIO` file to the `jqmc` format using `jqmc-tool`

```bash
% jqmc-tool trexio convert-to water_ccecp_ccpvtz.h5 -j2 1.0 -j3 mo
> Hamiltonian data is saved in hamiltonian_data.h5.
```

The generated `hamiltonian_data.chk` is a wavefunction file with the `jqmc` format. `-j2` specifies the initial value of the two-body Jastrow parameter and `-j3` specifies the basis set (`ao`:atomic orbital or `mo`:molecular orbital) for the three-body Jastrow part.

You can see the content of `hamiltonian_data.h5` by `jqmc-tool`

```bash
% jqmc-tool hamiltonian show-info hamiltonian_data.h5

Structure_data
  PBC flag = False
  --------------------------------------------------
  element, label, Z, x, y, z in cartesian (Bohr)
  --------------------------------------------------
  O, O, 8.0, 9.44863062, 13.50601812, 14.45823978
  H, H, 1.0, 7.68753060, 13.12032124, 14.29343676
  H, H, 1.0, 10.16717442, 13.03337116, 12.86522522
  --------------------------------------------------
Coulomb_potential_data
  ecp_flag = True
...

```

You can also see all the variables stored in `hamiltonian_data.h5` by `h5dump` if it is installed on your machine.

```bash
% h5dump hamiltonian_data.h5

HDF5 "hamiltonian_data.h5" {
GROUP "/" {
   ATTRIBUTE "_class_name" {
      DATATYPE  H5T_STRING {
         STRSIZE H5T_VARIABLE;
         STRPAD H5T_STR_NULLTERM;
         CSET H5T_CSET_UTF8;
         CTYPE H5T_C_S1;
      }
      DATASPACE  SCALAR
      DATA {
      (0): "Hamiltonian_data"
      }
   }
...

```

## Optimize a trial WF (VMC)
The next step is to optimize variational parameters included in the generated wavefunction. More in details, here, we optimize the two-body Jastrow parameter and the matrix elements of the three-body Jastrow parameter.

You can generate a template file for a VMCopt calculation using `jqmc-tool`. Please directly edit `vmc.toml` if you want to change a parameter.

```bash
% jqmc-tool vmc generate-input -g
> Input file is generated: vmc.toml
```

<!-- include: 03vmc_JSD/vmc.toml -->
```toml
[control]
job_type = "vmc" # Specify the job type. "mcmc", "vmc", "lrdmc", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.chk" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_chk = "hamiltonian_data.chk" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low" or "high"

[vmc]
num_mcmc_steps = 500 # Number of observable measurement steps per MPI and Walker. Every local energy and other observeables are measured num_mcmc_steps times in total. The total number of measurements is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates per measurement. Every local energy and other observeables are measured every this steps.
num_mcmc_warmup_steps = 0 # Number of observable measurement steps for warmup (i.e., discarged).
num_mcmc_bin_blocks = 5 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # the epsilon parameter used in the Attacalite-Sandro regulatization method.
num_opt_steps = 300 # Number of optimization steps.
wf_dump_freq = 1 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
optimizer_kwargs = { method = "sr", delta = 0.01, epsilon = 0.001 } # SR optimizer configuration (method plus step/regularization).
opt_J1_param = false
opt_J2_param = true
opt_J3_param = true
opt_lambda_param = false
num_param_opt = 0 # the number of parameters to optimize in the descending order of |f|/|std f|. If None, all parameters are optimized.
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
% jqmc-tool vmc analyze-output out_vmc -p -s vmc.jpg
```

![VMC optimization](03vmc_JSD/vmc.jpg)

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

## Compute Energy (MCMC)
The next step is MCMC calculation. You can generate a template file for a MCMC calculation using `jqmc-tool`. Please directly edit `mcmc.toml` if you want to change a parameter.

```bash
% jqmc-tool mcmc generate-input -g
> Input file is generated: mcmc.toml
```

<!-- include: 04mcmc_JSD/mcmc.toml -->
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
```

The final step is to run the `jqmc` job w/ or w/o MPI on a CPU or GPU machine (via a job queueing system such as PBS).

```bash
% jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/o MPI on CPU
% mpirun -np 4 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on CPU
% mpiexec -n 4 -map-by ppr:4:node jqmc mcmc.toml > out_mcmc 2> out_mcmc.e # w/ MPI on GPU, depending the queueing system.
```

You may get `E = -16.97202 +- 0.000288 Ha` and `Var(E) = 1.99127 +- 0.000901 Ha^2` [VMC w/ Jastrow factors]

## Compute Energy (LRDMC)
The final step is LRDMC calculation. You can generate a template file for a LRDMC calculation using `jqmc-tool`. Please directly edit `lrdmc.toml` if you want to change a parameter.

```bash
% cd alat_0.30
% jqmc-tool lrdmc generate-input -g lrdmc.toml
> Input file is generated: lrdmc.toml
```

<!-- include: 05lrdmc_JSD/lrdmc.toml -->
```toml
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

You may get `E = -17.235094 +- 0.00045 Ha` [LRDMC a -> 0]. This is the final result of this tutorial.

![VMC optimization](05lrdmc_JSD/lrdmc_ext.jpg)
