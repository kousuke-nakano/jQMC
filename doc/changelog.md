(changelog)=

# Change Log

## Apr-24-2026: v0.2.1a2

Minor update focusing on workflow improvements, bug fixes, and new benchmark infrastructure.

### New features

* **Kernel benchmark suite**: Added benchmark modules and tests for profiling kernel performance.
* **`cleanup_patterns` option**: Added a `cleanup_patterns` configuration option to `jqmc_workflow` for automatic post-run file cleanup, with support for recursive matching in subdirectories.

### Bug fixes

* **MPI deadlock in `max_time` / `stop_flag`**: Fixed a deadlock that could occur during `max_time` and `stop_flag` checks in MPI runs.

## Apr-16-2026: v0.2.1a1

This release focuses on a major update of the VMC optimizer (Linear Method), extended AO basis optimization, memory/performance improvements of the `jqmc` kernel package, and substantial hardening of the `jqmc_workflow` automation package.

### New features

*   **Linear Method (LM) optimizer**: Implemented the Linear Method optimizer that solves the generalized eigenvalue problem $\bar{H} v = E \bar{S} v$ for optimal parameter updates, providing a powerful alternative to the naive Stochastic Reconfiguration (SR). The optimization is robust and fast.
    *   LM is now integrated into the `method="sr"` code path, controlled by the `use_lm` flag and `lm_subspace_dim` parameter (inspired by TurboRVB's `ncg=1` design).
    *   **Unified optimizer hierarchy** under `method="sr"`:
        *   `use_lm=false`: plain SR
        *   `use_lm=true, lm_subspace_dim=0`: adaptive SR (aSR) with gamma scaling
        *   `use_lm=true, lm_subspace_dim>0`: LM with SR collective variable + top-$p$ S/N ratio parameters
        *   `use_lm=true, lm_subspace_dim=-1`: LM with SR collective variable + all parameters
    *   SR collective variable ($g = S^{-1}f$) is used as the first LM basis vector for stability.
    *   `dgelscut`-based preconditioning with iterative eigenvalue conditioning on the correlation matrix (condition number $\leq 1/\epsilon$), inspired by TurboRVB's implementation.
    *   S-orthonormalization ($P = U \Lambda^{-1/2}$) converts the generalized eigenvalue problem to standard form.
    *   Symmetrization of $H = K + B$ before eigenvalue solve to suppress finite-sample noise.
    *   Eigenvector selection by $\max |v_0|^2$ criterion.
    *   Separate `epsilon` (SR regularization) and `lm_cond` (LM dgelscut threshold) parameters.
    *   Fallback mechanisms: plain SR fallback ($\gamma=0.1$) when aSR finds no positive root; plain SR fallback ($0.1 \cdot g_\mathrm{sr}$) when LM does not predict energy improvement ($E_\mathrm{LM} > E_0 + 3\sigma$).
*   **Extended AO basis optimization**: Implemented `opt_J3_basis_coeff`, `opt_J3_basis_exp`, `opt_lambda_basis_coeff`, and `opt_lambda_basis_exp` options for optimizing three-body Jastrow and geminal AO basis exponents and coefficients in VMC.
    *   **Shell-shared constraint**: Same-atom, same-shell primitives share exponents/coefficients via `symmetrize_metric` (size-preserving shell averaging), consistent with `j3_matrix`/`lambda_matrix` symmetrization.
    *   **Dual symmetrization strategy**: $O_k$ derivatives are symmetrized at source in `get_dln_WF` for accurate $f$ and $S$, and post-hoc symmetrization is applied after `apply_block_update` to prevent floating-point drift over hundreds of optimization steps.
    *   Improved AO basis exponent selection using a log-spaced median window with widened margin (`/2.5`).
*   **`use_swct` parameter**: Added `use_swct` flag to MCMC, GFMC_t, and GFMC_n classes to control Space Warp Coordinate Transformation (SWCT) on/off for atomic force calculations. Default is `True` for MCMC and `False` for GFMC (LRDMC). When disabled, zero arrays are used for `omega`/`grad_omega`, and force formulas reduce to bare Hellmann-Feynman and Pulay forces.
*   **S/N ratio filter**: Applied S/N ratio filtering before SR matrix construction to reduce the effective matrix dimension, improving both speed and numerical stability. `O_matrix_local` is sliced to selected parameters before building $S$, so all SR computations operate in the reduced space.
*   **Shape assertions**: Added rigorous shape assertions (using `mcmc_counter`, `num_walkers`, `n_atoms`) to `get_E`, `get_aF`, `get_gF`, `get_aH` in MCMC, GFMC_t, GFMC_n, and their debug counterparts.

### Performance & memory

*   **Buffer-based MPI reduce in SR**: Eliminated `list()` round-trips and switched to buffer-based MPI reduce in the SR optimizer for lower overhead.
*   **Pre-compute collective observable**: Compute $O_\mathrm{SR} = \delta O \cdot g$ while the full $O$-matrix is still in memory during the SR solve, avoiding a redundant `get_dln_WF` call in the LM path.
*   **Avoid redundant JIT compilation**: Refactored `run_optimize` in `jqmc_mcmc.py` to skip redundant computation via early `continue`, reducing unnecessary recompilation.
*   **`jax.clear_caches()` after optimization loop**: Added cache clearing after the optimization loop as an OOM workaround.
*   **Store `np.array` instead of `list(np.array)`**: Refactored internal data storage to use `np.array` directly, reducing memory fragmentation.
*   **Wrapped properties with `np.asarray()`**: Prevent accidental storage of JAX arrays in checkpoint data to avoid OOM.
*   **Avoid redundant energy/force post-processing**: Skip unnecessary re-computation of energy and force post-processing.
*   **Better memory management**: Improved memory handling in `jqmc_gfmc.py` and `jqmc_mcmc.py`.


### Bug fixes

*   **GFMC_n / GFMC_t spin-polarized MPI bug**: Fixed a critical bug for systems where `n_up != n_dn` and `n_dn >= 1` with MPI >= 2 processes.
*   **GFMC_t projection averaging**: Fixed incorrect averaging of the number of projections across MPI ranks in GFMC_t.
*   **SR with `num_params >= num_samples`**: Fixed MPI bug when the number of optimizable parameters exceeds the number of samples.
*   **MPI `Allreduce` for scalars**: Replaced `Allreduce` with `allreduce` for scalar `int` and `float` values in `jqmc_mcmc.py` and `jqmc_gfmc.py`, as `Allreduce` for scalars exhibits implementation-dependent behavior.
*   **Optimizer step estimation**: Fixed `estimate_required_steps` — removed incorrect `ceil` rounding and `max` clamp that ignored `walker_ratio`; added `min_steps` parameter.
*   **SR stability near convergence**: Improved stability of SR with adaptive learning rate in the vicinity of convergence.
*   **Pytree inconsistency**: Fixed a JAX pytree structural mismatch.
*   **S/N ratio diagnostics**: Fixed averaging (last S/N ratio → averaged S/N ratios) and trivial output bugs.


### Workflow (`jqmc_workflow`)

*   **Major refactoring**: Comprehensive overhaul of all workflow modules (`vmc_workflow.py`, `mcmc_workflow.py`, `lrdmc_workflow.py`, `workflow.py`) with improved robustness, cleaner code structure, and new `_phase.py` module for phase management.
*   **SSH / file-descriptor leak fixes**: Fixed SSH connection hangs and leaks; consolidated `Machine` objects to prevent resource exhaustion.
*   **Continuation behavior**: Changed and improved the behavior of workflow continuations with `SHA256`-based input fingerprinting for reliable restart detection.
*   **Step count accumulation**: Fixed a bug in accumulated step counts for VMC and LRDMC workflows.
*   **VMC convergence check**: Implemented a new VMC energy-slope-based convergence check.
*   **New VMC workflow parameters**: Introduced additional configurable parameters for `vmc_workflow.py`.
*   **Output parser fixes**: Fixed parsers for workflow output processing.
*   **`FileFrom` handling**: Fixed and polished `FileFrom` file-transfer logic.
*   **Job ID checks**: Updated and improved job ID check logic for remote execution.
*   **Error estimation in workflows**: Fixed error estimation methods used by workflows.


### Breaking changes

*   **Removed `num_param_opt` and `opt_filter_min_SN_ratio`**: These parameters have been removed from `run_optimize()`, CLI, workflow, and TOML config. SR and optax now always optimize all parameters; parameter selection is handled internally by the LM subspace mechanism.
*   **Replaced `adaptive_learning_rate` with `use_lm`**: The `adaptive_learning_rate` flag is replaced by the `use_lm` flag, which controls the unified LM/aSR optimizer hierarchy.
*   **Removed `method="lm"` as separate code path**: The Linear Method is now accessed via `method="sr"` with `use_lm=true`.
*   **New optimizer parameter names**: `lm_cond` (default 0.001) replaces the previous LM-specific delta/epsilon naming.


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
