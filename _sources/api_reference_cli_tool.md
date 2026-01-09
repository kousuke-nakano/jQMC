(api_reference_cli_tool_link)=

# API reference for the command-line tool (jqmc-tool)

`jqmc-tool` provides ancillary CLI utilities for TREXIO conversion, Hamiltonian inspection/conversion, and pre/post-processing of VMC/MCMC/LRDMC workflows. Typical invocation:

```bash
jqmc-tool <group> <command> [options]
# or, module-style
python -m jqmc.jqmc_tool <group> <command> [options]
```

Groups: `trexio`, `hamiltonian`, `vmc`, `mcmc`, `lrdmc`.

---

## `trexio` group

### `show-info`
Show concise metadata stored in a TREXIO file.

```
jqmc-tool trexio show-info <filename>
```
- `filename` (arg): TREXIO file.

### `show-detail`
Dump full objects (structure, AOs, MOs, geminal, Coulomb).

```
jqmc-tool trexio show-detail <filename>
```
- `filename` (arg): TREXIO file.

### `convert-to`
Convert TREXIO to `hamiltonian_data.h5` with optional Jastrow setup.

```
jqmc-tool trexio convert-to <trexio_file> [options]
```
- `-o, --output` (str, default `hamiltonian_data.h5`): Output filename.
- `-j1, --jastrow-1b-parameter` (float|None): J1 parameter. If set and ECP is present, core electrons are taken from ECP; otherwise zeroed.
- `-j2, --jastrow-2b-parameter` (float|None): J2 parameter.
- `-j3, --jastrow-3b-basis-set-type` (`ao|ao-full|ao-small|ao-medium|ao-large|mo|none`, default `none`): Choose J3 basis; `ao-*` trims AOs by key exponent grouping, `mo` uses MOs, `none` disables J3.
- `-j-nn-type, --jastrow-nn-type` (str|None): Add NN Jastrow (e.g., `schnet`).
- `-jp, --jastrow-nn-param` (repeatable `key=value`): Hyperparameters forwarded to `Jastrow_NN_data.init_from_structure`; supported keys (type, default): `hidden_dim (int, 64)`, `num_layers (int, 3)`, `num_rbf (int, 16)`, `cutoff (float, 5.0)`.

Outputs an HDF5 `Hamiltonian_data` with Jastrow/Geminal embedded.

---

## `hamiltonian` group

### `show-info`
Print metadata from an existing Hamiltonian HDF5/checkpoint.

```
jqmc-tool hamiltonian show-info <hamiltonian_data>
```

### `to-xyz`
Export nuclear geometry to XYZ (Bohr -> Å).

```
jqmc-tool hamiltonian to-xyz <hamiltonian_data> [-o struct.xyz]
```

### `conv-wf`
Convert wavefunction ansatz inside a Hamiltonian file.

```
jqmc-tool hamiltonian conv-wf <hamiltonian_data> -c {jsd|jagp} [-o output.h5]
```
- `-c, --convert-to`: `jagp` converts SD→AGP (AOs); `jsd` is not implemented.

---

## `vmc` group (pre/post utilities)

### `fix` (legacy)
Rewrite `vmc.chk` archives to per-rank gzip members; creates `bak_<chk>` backup.

```
jqmc-tool vmc fix <restart_chk>
```

### `generate-input`
Emit a VMC `toml` template (sets `control.job_type = "vmc"`).

```
jqmc-tool vmc generate-input -g [-f vmc.toml] [--without-comment]
```

### `analyze-output`
Parse VMC optimization logs (energies, |f|, signal-to-noise) and optionally plot.

```
jqmc-tool vmc analyze-output <log...> [-p] [--save-graph file]
```

---

## `mcmc` group (pre/post utilities)

### `fix` (legacy)
Rewrite `mcmc.chk` archives to per-rank gzip members; creates `bak_<chk>` backup.

```
jqmc-tool mcmc fix <restart_chk>
```

### `compute-energy`
Jackknife estimator of VMC energy from an MCMC restart archive.

```
jqmc-tool mcmc compute-energy <restart_chk> [-b N] [-w W]
```
- `-b, --num_mcmc_bin_blocks` (int, default 1): Binning blocks per MPI × walker; total blocks = `b * mpi_size * walkers`. Must be ≥ `MCMC_MIN_BIN_BLOCKS`.
- `-w, --num_mcmc_warmup_steps` (int, default 0): Discarded warmup measurements; must be ≥ `MCMC_MIN_WARMUP_STEPS`.

### `generate-input`
Emit an MCMC `toml` template (sets `control.job_type = "mcmc"`).

```
jqmc-tool mcmc generate-input -g [-f mcmc.toml] [--without-comment]
```

---

## `lrdmc` group (pre/post utilities)

### `fix` (legacy)
Rewrite `lrdmc.chk` archives to per-rank gzip members; creates `bak_<chk>` backup.

```
jqmc-tool lrdmc fix <restart_chk>
```

### `compute-energy`
Jackknife estimator of LRDMC energy from an LRDMC restart archive.

```
jqmc-tool lrdmc compute-energy <restart_chk> [-b N] [-w W] [-c C]
```
- `-b, --num_gfmc_bin_blocks` (int, default 5): Binning blocks per MPI × walker (note: total blocks = `b`, not multiplied by ranks × walkers). Must be ≥ `GFMC_MIN_BIN_BLOCKS`.
- `-w, --num_gfmc_warmup_steps` (int, default 0): Discarded warmup steps; must be ≥ `GFMC_MIN_WARMUP_STEPS`.
- `-c, --num_gfmc_collect_steps` (int, default 5): Pre-binning measurements used to collect weights; must be ≥ `GFMC_MIN_COLLECT_STEPS`.

### `extrapolate-energy`
Fit energy vs $a^2$ from multiple LRDMC restart archives and extrapolate $a\to 0$.

```
jqmc-tool lrdmc extrapolate-energy <restart_chk...> [-p order] [-b N] [-w W] [-c C] [-g] [--save-graph file]
```
- `-p, --polynomial-order` (int, default 2): Fit $E(a^2) = E_0 + a^2 E_2 + a^4 E_4 + \dots$.
- Other options as in `compute-energy`.

### `generate-input`
Emit an LRDMC `toml` template (sets `control.job_type = "lrdmc"`).

```
jqmc-tool lrdmc generate-input -g [-f lrdmc.toml] [--without-comment]
```

---

## Notes on variational parameters (when NN Jastrow is enabled)
- All trainable weights/biases of the NN Jastrow (message, receiver, readout networks) plus spin and species embeddings are treated as variational parameters within VMC.
- NN hyperparameters are provided via `-j-nn-type` and `-jp key=value` on `trexio convert-to`; unsupported keys are passed through and validated by `Jastrow_NN_data.init_from_structure`.
