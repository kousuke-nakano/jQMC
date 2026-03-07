# Examples

A collection of examples of jQMC codes.

## jQMC examples

### jqmc-example01

Total energy of water molecule with the Jastrow-Slater determinant (JSD) ansatz. One can learn how to obtain the VMC and LRDMC (in the extrapolated limit) energies of the water molecule, starting from scratch (i.e., DFT calculation by `pySCF` or `CP2K`), with cartesian GTOs. In this example, users can learn how to perform energy calculations using VMC and LRDMC methods with the simplest JSD, starting from either `PySCF` or `CP2K`. Additional applications can be explored by appropriately modifying the input files provided for `PySCF` or `CP2K` in this example.


### jqmc-example02

Total energy of water molecule with the Neural-Network-Jastrow-Slater determinant (JNNSD) ansatz. One can learn how to obtain the VMC energy of the water molecule, starting from scratch (i.e., DFT calculation by `pySCF`), with cartesian GTOs.


### jqmc-example03

Projected-MO optimization workflow for the water molecule, using the same `PySCF` water setup as `example01`/`example02` and running VMC with `opt_with_projected_MOs = true` and `opt_lambda_param = true`. This example demonstrates how to optimize the determinantal coefficients (lambda matrix) together with Jastrow factors in a restricted MO subspace.


### jqmc-example04

Binding energy of the water-water dimer with the Jastrow Antisymmetrized Geminal Power (JAGP) ansatz. This example demonstrates how to perform energy calculations using the JAGP ansatz, which goes beyond the JSD ansatz to incorporate more sophisticated electron correlation. Both VMC and LRDMC methods are covered, starting from either `PySCF` or `CP2K`. Further applications can be carried out by adapting the input files provided in this example.


### jqmc-example05
In this example, users will learn how to compute atomic forces using a hydrogen molecule as a test system.


### jqmc-example06

How to call modules of `jQMC` from a `python` script.


### jqmc-example07

End-to-end Python API workflow (no CLI): generate a TREXIO file with PySCF, build `Hamiltonian_data` in Python, run VMC optimization and sampling, then scan LRDMC over multiple lattice constants and extrapolate $a\to 0$.


### jqmc-example08

MPI-enabled Python API workflow: similar to example10 but using `mpirun` on `run_jqmc_mpi.py` with `mpi4py` including VMC optimization, VMC sampling, and LRDMC lattice scans with extrapolation.


## jQMC-workflow examples

### jqmc-workflow-example01

Potential energy surface of hydrogen molecule with cartesian GTOs. All electron calculations. Comparison atomic forces with the derivative of the PES.


### jqmc-workflow-example02

Benchmarking of MPI/Vectorization on GPU machines, i.e., how to choose the most efficient number of walkers and number of MPI processes.


### jqmc-workflow-example03

Energy and Force calculations of the water molecule.
