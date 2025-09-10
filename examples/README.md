# Examples

A collection of examples of jQMC codes.

## example01

Total energy of water molecule with the Jastrow-Slater determinant (JSD) ansatz. One can learn how to obtain the VMC and LRDMC (in the extrapolated limit) energies of the water molecule, starting from scratch (i.e., DFT calculation by `pySCF` or `CP2K`), with cartesian GTOs. In this example, users can learn how to perform energy calculations using VMC and LRDMC methods with the simplest JSD, starting from either `PySCF` or `CP2K`. Additional applications can be explored by appropriately modifying the input files provided for `PySCF` or `CP2K` in this example.

## example02

Benchmarking of MPI/Vectorization on CPU and GPU machines, i.e., how to choose the most efficient number of walkers and number of MPI processes.

## example03

Binding energy of water-methane dimer, one of the validation tests for ab initio DMC implementation.

## example04

MCMC and HF energies of water molecule, for validation of the MCMC implementation of jQMC, with spherical or cartesian GTOs.

## example05

Potential energy surface of hydrogen molecule with cartesian GTOs. All electron calculations. Comparison atomic forces with the derivative of the PES. In this example, users will learn how to compute atomic forces using a hydrogen molecule as a test system. Note, however, that atomic force calculations in QMC are still under active methodological development and should be considered advanced. We recommend consulting with experts before using such results in research studies.

## example06

Binding energy of the water-water dimer with the Jastrow Antisymmetrized Geminal Power (JAGP) ansatz. This example demonstrates how to perform energy calculations using the JAGP ansatz, which goes beyond the JSD ansatz to incorporate more sophisticated electron correlation. Both VMC and LRDMC methods are covered, starting from either `PySCF` or `CP2K`. Further applications can be carried out by adapting the input files provided in this example.

## example07

Weak scalings of LRDMC using the water molecule.

## example08

How to call modules of `jQMC` from a `python` script.
