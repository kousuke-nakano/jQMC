(changelog)=

# Change Log

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
