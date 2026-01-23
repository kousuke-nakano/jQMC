(changelog)=

# Change Log

## Jan-23-2026: v0.1.0a2

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
