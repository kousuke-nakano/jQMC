# jQMC Vectorization Benchmark Results on GPUs

This directory contains vectroziationi benchmark results. The benchmarks compare the throughputs of Markov-Chain Monte Carlo (MCMC) and Lattice Regularized Diffusion Monte Carlo (LRDMC) calculations on GPU with respect to the number of walkers assigned to one GPU. The attached graphs are summaries of the throughputs of both VMC and LRDMC calculations with respect to the number of walkers per GPU.

---

## Benchmark Setup

The benchmark calculations were performed using four molecular systems:

| Molecule         | Number of electrons | Basis Set           |   ECP          |
|------------------|---------------------|---------------------|----------------|
| Water            | 8                   | `ccecp_ccpvtz`      |  ccECP         |
| Water dimer      | 16                  | `ccecp_ccpvtz`      |  ccECP         |
| Benzene          | 30                  | `ccecp_augccpvtz`   |  ccECP         |
| Benzene dimer    | 60                  | `ccecp_augccpvtz`   |  ccECP         |

**Additional details:**

- **Pseudopotential:** The [ccECP pseudopotential](https://pseudopotentiallibrary.org) was employed for all calculations.
- **Trial Wavefunctions:** Generated using [pySCF](https://pyscf.org) with Gaussian basis functions (Cartesian).
- **Hardware Configuration:** Benchmarks were measured on the supercomputer [Miyabi](https://www.cc.u-tokyo.ac.jp/en/supercomputer/miyabi/system.php) at the University of Tokyo in Japan. One node is equipped with a NVIDIA Grace CPU with an NVIDIA H100 (Hopper) GPU. One node (i.e., 1 CPU + 1 GPU) is used for this benchmark test.

---

## Benchmark Results

### MCMC Benchmark

The following graph plots GPU throughput for MCMC calculations:

![MCMC Benchmark](jqmc_MCMC_vectorization_benchmark.jpg)

### LRDMC Benchmark

The following graph plots GPU throughput for LRDMC calculations:

![LRDMC Benchmark](jqmc_LRDMC_vectorization_benchmark.jpg)

---

## Reproducing the Benchmarks

Please have a look at the files included in this directory.
