# jQMC weak scaling results on GPUs

This directory contains weak-scaling results for **jQMC**. The weak scaling is measured for Lattice Regularized Diffusion Monte Carlo (LRDMC) calculations on GPUs. The weak-scaling were measured on the supercomputer [Leonardo](https://www.hpc.cineca.it/systems/hardware/leonardo/) at CINECA in Italy and on the supercomputer [Miyabi](https://www.cc.u-tokyo.ac.jp/en/supercomputer/miyabi/system.php) at the University of Tokyo in Japan.

---

## Target System

The weak-scaling calculations were performed using the benzene molecule:

| Molecule         | Number of electrons | Basis Set           |   ECP          |
|------------------|---------------------|---------------------|----------------|
| Benzene          | 30                  | `ccecp-aug-ccpvtz`  |  ccECP         |

- **Pseudopotential:** The [ccECP pseudopotential](https://pseudopotentiallibrary.org) was employed for all calculations.
- **Trial Wavefunctions:** Generated using [pySCF](https://pyscf.org) with Gaussian basis functions (Cartesian).

## Hardware configurations

Here is the Hardware configurations used in these benchmark tests.

| Category                | Component            | Leonardo                              | Miyabi                           |
| ----------------------- | -------------------- | ------------------------------------- | -------------------------------- |
| **Cluster name**        |                      | Leonardo                              | Miyabi                           |
| **Vendor & Model**      |                      | BullSequana XH2000, Atos              | PRIMERGY CX2550 M7, Fujitsu      |
| **Operator & Location** |                      | CINECA in Italy                       | The University of Tokyo in Japan |
| **CPU**                 | Processor name       | Intel Xeon Platinum 8358              | NVIDIA Grace CPU C1              |
|                         | Number of processors | 1 CPU                                 | 1 CPU                            |
|                         | Number of cores      | 32 cores                              | 72 cores                         |
|                         | Frequency            | 2.6 GHz                               | 3.0 GHz                          |
|                         | Memory               | 512 GB                                | 120 GB                           |
|                         | Memory bandwidth     | –                                     | 512 GB/s                         |
| **GPU**                 | Processor name       | NVIDIA custom A100                    | NVIDIA H100                      |
|                         | Number of processors | 4                                     | 1                                |
|                         | Memory               | 64 GB                                 | 96 GB                            |
|                         | Memory bandwidth     | 461 GB/s                              | 4.02 TB/s                        |
|                         | CPU–GPU connection   | NVLink 3.0 (200 GB/s)                 | NVLink C2C (450 GB/s)            |
|                         | Interconnect         | InfiniBand 2×dual-port HDR (400 Gbps) | InfiniBand NDR (200 Gbps)        |
| **Total nodes**         |                      | 3456                                  | 1096                             |
| **Total GPUs**          |                      | 13 824 GPUs                           | 1096 GPUs                        |

## Results

Although JAX provides native support for multi-processing via its distributed runtime, jQMC currently enables multi-GPU execution through explicit MPI parallelization using `mpi4py` and `mpi4jax`.

A key factor in evaluating the efficiency of multi-GPU computations is scalability—specifically, weak-scaling behavior. This metric quantifies how effectively additional GPUs contribute to performance gains under different workload scenarios.

Figure 1 presents the results of the weak-scaling test for the conventional (`lrdmc-tau`) and load-balanced (`lrdmc`) LRDMC algorithms. These benchmarks provide a quantitative assessment of the parallel efficiency of jQMC across multiple GPUs and serve as critical indicators of its suitability for large-scale QMC simulations. As clearly shown in the figure, the conventional algorithm exhibits a steep decline in computational efficiency as the number of walkers (i.e., the degree of GPU parallelization) increases. This is because the conventional algorithm determines the length of each projection step using random numbers, resulting in significant load imbalance among parallel walkers. Since all walkers must wait until the longest projection operation is completed, many walkers remain idle for extended periods. This behavior leads to an increased likelihood of encountering “slow” walkers with long projection times as the number of walkers grows, resulting in a linear degradation of weak-scaling efficiency.

In contrast, the load-balanced LRDMC algorithm implemented in jQMC ensures, by design, that the computational workload is uniformly distributed among walkers. Consequently, the benchmark results demonstrate that jQMC maintains stable weak scaling even as the number of walkers (GPUs) increases. Unless there is a specific reason not to, we recommend using `lrdmc` for LRDMC calculations.

![Comparison of the weak-scaling benchmark between the textbook and load-balancing LRDMC algorithms, measured on Miyabi using the benzene molecule ($N_e = 30$).](jqmc_tau_nbra_comparison_benzene_on_gpu_leonardo.jpg)

Figure 1: Comparison of the weak-scaling benchmark between the conventional and load-balanced LRDMC algorithms, measured on Miyabi using the benzene molecule ($N_e = 30$).

---

(!!TO BE REPLACED with the updated results!!) Figure 2 presents the results of a weak-scaling test for the LRDMC algorithm. These benchmarks provide a quantitative assessment of the parallel efficiency of jQMC across multiple GPUs and serve as critical indicators of its suitability for large-scale QMC simulations.

As shown in Figure 2 (b), both the Miyabi and Leonardo systems maintain high parallel efficiency—close to the ideal value of 1—even up to 1024 GPUs (102 400 walkers). In both cases, increasing the number of GPUs to 1024 results in only about a \~2 % reduction in computation speed, demonstrating that jQMC achieves exceptionally efficient parallel scaling.

Notice that the slightly lower weak-scaling performance observed on Miyabi compared to Leonardo is due to differences in hardware architecture: Miyabi is configured with one GPU per node, whereas Leonardo uses four GPUs per node. As a result, for the same number of GPUs, Miyabi requires more inter-node MPI communication, which contributes to the minor decline in scaling efficiency.

Additionally, Figure 2 (a) shows that the actual wall-clock execution time on Miyabi is shorter than on Leonardo—about 1.7× faster. This reflects the higher performance of the GPUs installed on Miyabi.

![Weak-scaling benchmark measured on Miyabi and Leonardo using the benzene molecule ($N_e = 30$). (a) The elapsed times of the LRDMC runs with respect to the number of GPUs. (b) The normalized times of LRDMC runs with respect to the number of GPUs. These benchmark tests were measured on Leonardo and Miyabi supercomputers.](jqmc_weak_scaling_benzene_on_gpu.jpg)

Figure 2: (!!TO BE REPLACED with the updated results!!) Weak-scaling benchmark measured on Miyabi and Leonardo using the benzene molecule ($N_e = 30$). (a) The elapsed times of the LRDMC runs with respect to the number of GPUs. (b) The normalized times of LRDMC runs with respect to the number of GPUs. These benchmark tests were measured on Leonardo and Miyabi supercomputers.

## Reproducing the Benchmarks

Please have a look at the files included in this directory.
