# Interfaces with TREX-IO

(trexio_tags)=

## Conventions employed in jQMC and TREX-IO

jQMC utilizes the [TREX-IO](https://github.com/TREX-CoE/trexio) library for handling input and output of quantum chemical data. This interface ensures standardization and portability of wavefunction parameters and system coordinate data. The current implementation supports reading from TREXIO files using the HDF5 backend.

### The `read_trexio_file` Function

The core function for data ingestion is `read_trexio_file`, located in `jqmc/trexio_wrapper.py`. This function parses a TREXIO file and populates internal jQMC data structures.

**Signature:**

```python
def read_trexio_file(
    trexio_file: str,
    store_tuple: bool = False
) -> tuple[Structure_data, AOs_data, MOs_data, MOs_data, Geminal_data, Coulomb_potential_data]:
```

**Arguments:**

*   `trexio_file` (str): The file path to the TREXIO file (typically `.h5`).
*   `store_tuple` (bool, default=`False`): If `True`, stores list variables as tuples. This is primarily for internal testing (e.g., with JAX JIT compilation compatibility checks) but is slower for production runs.

**Returns:**

Ordering of the returned tuple:

1.  `Structure_data`: Molecular geometry and atomic information.
2.  `AOs_data` (`AOs_cart_data` or `AOs_sphe_data`): Atomic Orbital basis set information.
3.  `MOs_data` (Up): Molecular Orbitals for spin-up electrons.
4.  `MOs_data` (Down): Molecular Orbitals for spin-down electrons.
5.  `Geminal_data`: Data relevant for Antisymmetrized Geminal Power (AGP) wavefunctions.
6.  `Coulomb_potential_data`: Information on pseudopotentials (ECPs) if present.

---

### Implementation Details

#### 1. System Structure

*   **Boundary Conditions**: The wrapper currently checks `trexio.read_pbc_periodic`.
    *   **Open Boundary Conditions (Molecules)**: Fully supported.
    *   **Periodic Boundary Conditions (Crystals)**: Detected but currently raises `NotImplementedError`.
*   **Atomic Information**: Reads nuclear labels (`H`, `C`, etc.) and coordinates. Labels are converted to atomic numbers (Z) using an internal mapping supporting elements up to Radon (Z=86).

#### 2. Electron Configuration

*   Reads `electron_up_num` and `electron_dn_num`.
*   Determines if the system is **spin-polarized**:
    *   `spin_polarized = True` if $N_{\uparrow} \neq N_{\downarrow}$.
    *   `spin_polarized = False` if $N_{\uparrow} = N_{\downarrow}$.

#### 3. Atomic Orbitals (AOs)

The wrapper distinguishes between Cartesian and Spherical Harmonic basis sets via `trexio.read_ao_cartesian`.

**Cartesian Gaussians:**
*   Polynomial orders ($n_x, n_y, n_z$) for each orbital are reconstructed based on the angular momentum $L$.
*   In the case of polynomials, the canonical (or alphabetical) ordering is used (e.g., $xx, xy, xz, yy, yz, zz$ for $d$-orbitals).
*   Basis function normalization and coefficients are computed and stored in `AOs_cart_data`.

**Spherical Harmonics:**
*   If `ao_cartesian` is false, `AOs_sphe_data` is initialized.
*   The Magnetic quantum numbers ($m$) follow the TREXIO canonical order: $0, 1, -1, 2, -2, \dots$ for a given $L$.

#### 4. Molecular Orbitals (MOs)

*   **Real vs. Complex**: Checks `trexio.has_mo_coefficient_im`. Currently, **only real wavefunctions are supported**; complex coefficients raise `NotImplementedError`.
*   **Spin Handling**:
    *   **Restricted (RHF/ROHF)**: Used when `spin_dependent` is false. Up and Down orbitals share the same spatial coefficients.
    *   **Unrestricted (UHF)**: Used when `spin_dependent` is true and system is polarized. Up and Down coefficients are read separately.
*   **Occupation Threshold**: Orbitals are only loaded if their occupation number is $\ge 10^{-6}$. This filters out virtual orbitals with effectively zero occupation from the active space calculation.

#### 5. Pseudopotentials (ECPs)

*   The function checks for the existence of ECP data using `trexio.has_ecp_num`.
*   If present, `Coulomb_potential_data` is populated with:
    *   Core charges ($Z_{\text{core}}$)
    *   Angular momentum limits
    *   Exponents, coefficients, and powers ($n$) for the Gaussian expansion of the potential.
