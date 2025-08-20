(examples_link)=

# Examples

Example files for VMC, VMCopt, and LRDMC calculations by **jQMC** are found at
https://github.com/kousuke-nakano/jQMC/tree/main/examples.

The `examples/` directory contains several examples. For first-time users of jQMC, we recommend starting with `example01`. In this example, one can learn how to obtain the VMC and LRDMC (in the extrapolated limit) energies of the water molecule from scratch—that is, starting from a DFT calculation using either `PySCF` or `CP2K`, with cartesian GTOs. This example demonstrates how to perform energy calculations using VMC and LRDMC methods with the simplest Jastrow-Slater determinant (JSD) ansatz. Additional applications can be explored by appropriately modifying the input files provided for `PySCF` or `CP2K` in this example.

The next recommended example is `example06`. This example shows how to perform energy calculations using the Jastrow Antisymmetrized Geminal Power (JAGP) ansatz, which goes beyond the JSD ansatz by capturing more sophisticated electron correlation. Both VMC and LRDMC methods are demonstrated, starting from either `PySCF` or `CP2K`. Further applications can be explored by adapting the input files included in this example.

For users interested in more advanced calculations-particularly atomic force evaluations-we suggest examining `example05`. In this example, users will learn how to compute atomic forces using a hydrogen molecule as a test system. Note, however, that atomic force calculations within QMC are still under active methodological development and should be regarded as advanced. We recommend consulting with experts before employing such results in research studies.
