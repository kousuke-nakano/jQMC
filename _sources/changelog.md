(changelog)=

# Change Log

## Mar-10-2026: v0.2.0a1

This is a major update with drastic performance improvements, new features, and a new workflow automation package (`jqmc-workflow`).

### Performance

*   **Drastic speedups**: MCMC, VMC, and LRDMC are all significantly faster than the previous version thanks to pervasive use of fast-update algorithms throughout the code.
*   **LU -> SVD replacement**: Replaced LU factorizations with SVD across determinant, geminal, `GFMC_n`, and `GFMC_t` modules, greatly improving numerical stability for ill-conditioned matrices.
*   **GEMM optimization**: Converted matrix-vector operations to matrix-matrix (GEMM) operations in Coulomb potential, determinant, and Jastrow factor modules for better GPU utilization.
*   **Cartesian / Spherical AO conversion**: Implemented Cartesian AO <-> Spherical AO conversion. Cartesian GTOs are substantially faster than spherical GTOs on GPUs, so users can now exploit this for better throughput.
*   **ECP fast computation**: Implemented `compute_ecp_coulomb_potential_fast` for efficient pseudopotential evaluation.
*   **`vmap` + `jit` fix**: `vmap`-ed functions are now explicitly wrapped with `jit`, as `vmap` does not automatically JIT-compile the mapped function.
*   **Removed `mpi4jax` dependency for CG**: Conjugate gradient (CG) solver now uses pure MPI on CPUs, eliminating the `mpi4jax` dependency.

### Optimization

*   **Adaptive learning rate for Stochastic Reconfiguration**: Implemented a linear-method-inspired automatic learning-rate adjustment scheme, leading to dramatically faster optimization convergence.
*   **Molecular orbital optimization**: Added MO optimization for JSD wavefunctions via the projection method with Attacalite-Sorella regularization.
*   **Geminal AO -> MO projection**: Implemented AO overlap matrix computation and geminal AO -> MO projection for constrained optimization.

### New features

*   **LRDMC force calculations**: Implemented LRDMC atomic forces with the Pathak–Wagner regularization.
*   **Jastrow functions**: Added `jastrow_1b_type` (`'exp'` / `'pade'`) and `jastrow_2b_type` (`'pade'` / `'exp'`) fields to `Jastrow_one_body_data` and `Jastrow_two_body_data`, enabling runtime selection of the one-body and two-body Jastrow functional forms.
    *   Exponential form: $u(r) = \frac{1}{2b}(1 - e^{-br})$
    *   Padé form: $u(r) = \frac{r}{2(1 + br)}$

### Bug fixes

*   Fixed force calculations producing NaN values; added NaN checks in all tests.
*   Fixed MCMC memory overflow caused by storing `r_up_history` / `r_dn_history`.
*   Fixed wavefunction without Jastrow not working for MCMC.
*   Fixed missing NN-Jastrow derivatives in `_GFMC_n_debug`.

### Infrastructure

*   **Restart file format change**: Switched restart files from pickle-based `*.chk` to HDF5-based `*.h5`. **Note:** backward compatibility with old `*.chk` files is *not* maintained.
*   **`jqmc_workflow` package**: Introduced the `jqmc_workflow` automation package for orchestrating multi-stage QMC pipelines (WF conversion → VMC optimization → MCMC / LRDMC production) with automatic step estimation, checkpointing, and remote job management.
*   **Removed `SWCT_data`**: Cleaned up legacy `SWCT_data` class as part of codebase refactoring.
*   **More comprehensive tests**: Substantially expanded the test suite to cover the new features and improve overall reliability.
*   **Expanded examples**: Reorganized and enriched the `examples/` directory with 11 end-to-end tutorials (`jqmc-example01`–`jqmc-example08`, `jqmc-workflow-example01`–`jqmc-workflow-example03`) covering single-point VMC/LRDMC, force calculations, GPU walker-scaling benchmarks, interaction-energy workflows, and PES scans with automated `jqmc_workflow` pipelines.


## Feb-5-2026: v0.1.0

- Release of the first stable version of **jQMC**.

### Known Limitations

*   Periodic Boundary Condition (PBC) calculations are being implemented for the next major release.

## Jan-23-2026: v0.1.0a3

- Release of the third alpha version of **jQMC**.

### Key Features

*   **Analytical derivatives**:
    *   Implemented analytical gradients and Laplacians for atomic and molecular orbitals in both spherical and Cartesian GTO bases.
    *   JAX autograd is now used primarily for validating the analytical gradients.
    *   Logarithmic derivatives of the wavefunction and derivatives of atomic force calculations still use JAX autograd.
*   **Testing precision**:
    *   Tightened and systematized decimal controls in tests, improving overall reliability.
*   **Fast updates**:
    *   Expanded fast-update implementations to more functions, yielding significant speedups in both MCMC and GFMC modules.

## Jan-14-2026: v0.1.0a1

- Release of the second alpha version of **jQMC**.

### Key Features

*   **Neural Network Jastrow**:
    *   Introduced `NNJastrow`, a PauliNet-inspired neural network architecture for many-body Jastrow factors, enabling more accurate wavefunction ansatz.
*   **Optimization Control**:
    *   Implemented proper gradient masking mechanisms (e.g., `with_param_grad_mask`). This allows for selectively freezing or optimizing specific parameter blocks (One-body, Two-body, Three-body, NN, and Geminal coefficients) during the VMC optimizations.

### Enhancements & Fixes

*   **I/O**: Changed the storage format for `hamiltonian_data` from pickled binary files to HDF5 (`.h5`) for better portability and compatibility.
*   **Documentation**: Updated `README.md`, docstrings, and API references to reflect recent changes and fix Sphinx warnings.
*   **CI/CD**: Updated pre-commit configurations and GitHub workflow triggers.
*   **Code Quality**: Refactored code based on suggestions and improved type hinting.

## Aug-20-2025: v0.1.0a0

- Release of the first alpha version of **jQMC**.

We are pleased to announce the first alpha release of **jQMC**, a Python-based Quantum Monte Carlo package built on **JAX**.

### Key Features

*   **JAX-based Core**: Fully utilizes JAX's Just-In-Time (JIT) compilation and automatic vectorization (`vmap`) for high-performance simulations on GPUs and TPUs.
*   **Algorithms**:
    *   **Variational Monte Carlo (VMC)**: Supports wavefunction optimization via Stochastic Reconfiguration (SR) and Natural Gradient methods.
    *   **Lattice Regularized Diffusion Monte Carlo (LRDMC)**: A stable and efficient projection method for ground state calculations.
*   **Wavefunctions**:
    *   **Ansatz**: Supports Jastrow-Slater Determinant (JSD) and Jastrow-Antisymmetrized Geminal Power (JAGP).
    *   **Jastrow Factors**: Includes One-body, Two-body, Three/Four-body terms.
    *   **Determinant Types**: Single Determinant (SD), Antisymmetrized Geminal Power (AGP), and Number-constrained AGP (AGPn).
*   **I/O & Interoperability**:
    *   **TREX-IO Support**: Interfaces with the [TREX-IO](https://github.com/TREX-CoE/trexio) library (HDF5 backend) for standardized input of molecular structure and basis sets (Cartesian & Spherical GTOs).
*   **Parallelization**:
    *   **MPI Support**: Implements `mpi4py` for efficient parallelization across multiple nodes.
*   **Documentation**:
    *   Comprehensive technical notes on Wavefunctions, VMC, LRDMC, and JAX implementation details.
    *   Examples demonstrating usage for various systems (H2, N2, Water, etc.).

### Known Limitations (Alpha)

*   Periodic Boundary Conditions (PBC) are currently in development.
*   Atomic force calculations with spherical harmonics are computationally intensive on current JAX versions.
*   Complex wavefunctions are not yet supported.
