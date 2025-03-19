# Water molecule

The first step of ab-initio QMC is to generate a trial WF by HF. `jQMC` interfaces with other QC software packages via `TREXIO`. One of the easiest ways to produce it is using `pySCF` as a converter to the `TREXIO` format is implemented. The following is a script to run a HF calculation of the water molecule and dump it as a `TREXIO` file.

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
> Hamiltonian data is saved in hamiltonian_data.chk.
```

The generated `hamiltonian_data.chk` is a wavefunction file with the `jqmc` format. No Jastrow factors are added here.

Then, you can generate a template file for a VMC calculation using `jqmc-tool`. Please directly edit `vmc.toml` if you want to change a parameter.

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

You may get `E = -16.94478 +- 0.000203` [VMC wo/ Jastrow factors]

These two energies should be consistent with the VMC error bar as far as the VMC implemenation is correct.

```
E = -16.94503 Ha (Hartree Fock by PySCF)
E = -16.94478(20) Ha (VMC wo/ Jastrow by jQMC)
```
