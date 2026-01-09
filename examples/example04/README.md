# MCMC validation

`jQMC` interfaces with other QC software packages via `TREXIO`. One of the easiest ways to produce it is using `pySCF` as a converter to the `TREXIO` format is implemented. The following is a script to run a HF calculation of the water molecule and dump it as a `TREXIO` file.

```python:run_pyscf.py
from pyscf import gto, scf
from pyscf.tools import trexio

filename = 'water_ccecp_ccpvqz_cart.h5'

mol = gto.Mole()
mol.verbose  = 5
mol.atom     = '''
               O    5.00000000   7.14707700   7.65097100
               H    4.06806600   6.94297500   7.56376100
               H    5.38023700   6.89696300   6.80798400
               '''
mol.basis    = 'ccecp-ccpvqz'
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
% jqmc-tool trexio convert-to water_ccecp_ccpvqz_cart.h5
> Hamiltonian data is saved in hamiltonian_data.h5.
```

The generated `hamiltonian_data.h5` is a wavefunction file with the `jqmc` format. No Jastrow factors are added here.

Then, you can generate a template file for a MCMC calculation using `jqmc-tool`. Please directly edit `mcmc.toml` if you want to change a parameter.

```bash
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

You may get `E = -16.94478 +- 0.000203` [MCMC wo/ Jastrow factors]

These two energies should be consistent with the MCMC error bar as far as the MCMC implemenation is correct.

Here is the summary of the Validation tests.

| System  | Spin     |  Type    |   basis        |  ECP    |GTOs           |  HF (Ha)      | MCMC (Ha)     |
|---------|----------|----------|----------------|---------|---------------|---------------|---------------|
| H2O     | 0        | RHF      | ccecp-ccpvqz   |  ccECP  | Cartesian     | -16.94503     | -16.94487(28) |
| H2O     | 0        | RHF      | ccecp-ccpvqz   |  ccECP  | Spherical     | -16.94490     | -16.94482(28) |
| Ar      | 0        | RHF      | ccecp-ccpv5z   |  ccECP  | Cartesian     | -20.77966     | -20.77960(22) |
| Ar      | 0        | RHF      | ccecp-ccpv5z   |  ccECP  | Spherical     | -20.77966     | -20.77960(22) |
| N       | 3        | ROHF     | ccecp-ccpvqz   |  ccECP  | Cartesian     |  -9.63387     |  -9.63371(28) |
| N       | 3        | ROHF     | ccecp-ccpvqz   |  ccECP  | Spherical     |  -9.63387     |  -9.63350(28) |
| N       | 3        | UHF      | ccecp-ccpvqz   |  ccECP  | Cartesian     |  -9.63859     |  -9.63815(27) |
| N       | 3        | UHF      | ccecp-ccpvqz   |  ccECP  | Spherical     |  -9.63856     |  -9.63835(28) |
| O2      | 2        | ROHF     | ccecp-ccpvqz   |  ccECP  | Cartesian     | -31.42286     | -31.42254(19) |
| O2      | 2        | ROHF     | ccecp-ccpvqz   |  ccECP  | Spherical     | -31.42194     | -31.42177(18) |
| O2      | 2        | UHF      | ccecp-ccpvqz   |  ccECP  | Cartesian     | -31.44677     | -31.44668(18) |
| O2      | 2        | UHF      | ccecp-ccpvqz   |  ccECP  | Spherical     | -31.44579     | -31.44589(18) |
