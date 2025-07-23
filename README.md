# jQMC

![jqmc_logo](logo/logo_yoko2.jpg)

**jQMC** is an ab initio quantum Monte Carlo (QMC) simulation package developed entirely from scratch using `Python` and `JAX`. Originally designed for molecular systems—with future extensions planned for periodic systems—**jQMC** implements two well-established QMC algorithms: Variational Monte Carlo (VMC) and a robust and efficient variant of Diffusion Monte Carlo algorithm known as Lattice Regularized Diffusion Monte Carlo (LRDMC). By leveraging `JAX` just-in-time (`jit`) compilation and vectorized mapping (`vmap`) functionalities, `jQMC` achieves high-performance computations **especially on GPUs and TPUs** while remaining portable across CPUs, GPUs, and TPUs. See [here](http://jax.readthedocs.io/) for the details of `JAX`.

![license](https://img.shields.io/github/license/kousuke-nakano/jQMC)
![release](https://img.shields.io/github/release/kousuke-nakano/jQMC/all.svg)
![fork](https://img.shields.io/github/forks/kousuke-nakano/jQMC?style=social)
![stars](https://img.shields.io/github/stars/kousuke-nakano/jQMC?style=social)
![workflows](https://github.com/kousuke-nakano/jQMC/actions/workflows/jqmc-run-pytest.yml/badge.svg)
![codecov](https://codecov.io/github/kousuke-nakano/jQMC/graph/badge.svg?token=H0Z7M86C1E)

What sets **jQMC** apart:

- It employs a resonating valence bond (RVB)-type wave function, such as the Jastrow Antisymmetrized Geminal (JAGP) wavefunction, which captures correlation effects beyond the conventional Jastrow-Slater wave function used in many other QMC codes.
- It features a state-of-the-art optimization algorithm, stochastic reconfiguration, that enables stable optimization of both the amplitudes and nodal surfaces of many-body wave functions at the variational level.
- It implements the LRDMC method, providing a numerically stable approach to diffusion Monte Carlo calculations.
- The use of adjoint algorithmic differentiation in `JAX` allows for efficient differentiation of many-body wave functions, facilitating the computation of atomic forces analytically.
- Written in `Python`, **jQMC** is designed to be user-friendly for executing simulations and easily extensible for developers implementing and testing new QMC methods.
- By leveraging `JAX` just-in-time (`jit`) compilation and vectorized mapping (`vmap`) functionalities, the code achieves high-performance computations **especially on GPUs and TPUs** while remaining portable across CPUs, GPUs, and TPUs.
- MPI support enables the execution of large-scale computations on HPC facilities.
- To minimize bugs, the code is written in a loosely coupled manner and includes comprehensive unit tests and regression tests (managed by `pytest`).

This combination of features makes **jQMC** a versatile and powerful tool for both users and developers in the field of quantum Monte Carlo simulations.

## Known issues
- On CPUs, `jQMC` is ~10 times slower than other QMC codes implemented by a compiled language, such as C++, Fortran. Further improvements from the algorith and implementation viewpoints are needed. On GPUs, `jQMC` should be compatible with other QMC codes, but further benchmark tests are needed to confirm this.
- Atomic force calculations with **solid (sperical) harmonics GTOs** are much slower than energy and energy-optimization calculations due to the very slow compilations of dlnPsi/dR and de_L/dR. This is because `grad`, `jvp`, and `vjp` are slow for these terms for some reason. A more detailed analysis will be needed. Please use **cartesian GTOs** to do those calculations
- Periodic boundary condition calculations are not supoorted yet. It will be implemented in the future as `JAX` supports complex128. Work in progress.

## Developer(s)
Kosuke Nakano (National Institute for Materials Science, NIMS, Japan)


## How to install jQMC

First please git clone this repo.

```bash
% git clone https://github.com/kousuke-nakano/jQMC
```

**jQMC** can be installed via pip

```bash
% cd jQMC
% pip install .
```

> [!NOTE]
> `jQMC` is not yet distributed from `PyPI`. So, %pip install jqmc does not work at present.



## Examples
Examples are in `examples` directory.


## Supporting HF/DFT packages
`jQMC` can prepare a trial wavefunction from a `TREX-IO` file. Below is the list of HF/DFT packages that adopt `TREX-IO` for writing wave functions:

- [Quantum Package](https://github.com/QuantumPackage/qp2)
- [PySCF](https://github.com/pyscf/pyscf)
- [FHI-aims](https://fhi-aims.org/)
- [CP2K](https://github.com/cp2k/cp2k)
- [Dirac](https://www.diracprogram.org)
- [pymolpro](https://molpro.github.io/pymolpro)

See the [TREX-IO website](https://github.com/TREX-CoE/trexio) for the detail.


## Documentation

**jQMC** user documentation is written using python sphinx. The source files are
stored in `doc` directory. Please see how to write the documentation at
`doc/README.md`.

## Contribution

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Develpment branch

The development of jQMC is managed on the `devel` branch of github jQMC repository.

- Github issues is the place to discuss about jQMC issues.
- Github pull request is the place to request merging source code.

## Formatting

Formatting rules are written in `pyproject.toml`.

## pre-commit

Pre-commit (https://pre-commit.com/) is mainly used for applying the formatting
rules automatically. Therefore, it is strongly encouraged to use it at or before
git-commit. Pre-commit is set-up and used in the following way:

- Installed by `pip install pre-commit`, `conda install pre_commit` or see
  https://pre-commit.com/#install.
- pre-commit hook is installed by `pre-commit install`.
- pre-commit hook is run by `pre-commit run --all-files`.

Unless running pre-commit, pre-commit.ci may push the fix at PR by github
action. In this case, the fix should be merged by the contributor's repository.

## VSCode setting
- Not strictly, but VSCode's `settings.json` may be written like below

  ```json
  "ruff.lint.args": [
      "--config=${workspaceFolder}/pyproject.toml",
  ],
  "[python]": {
      "editor.defaultFormatter": "charliermarsh.ruff",
      "editor.codeActionsOnSave": {
          "source.organizeImports": "explicit"
      }
  },
  ```

## How to run tests

Tests are written using pytest. To run tests, pytest has to be installed.
The tests can be run by

```bash
% pytest -s -v  # with jax-jit
% pytest -s -v --disable-jit  # without jax jit
```
