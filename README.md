# jQMC

![jqmc_logo](logo/logo_yoko2.jpg)

**jQMC** is an ab initio quantum Monte Carlo (QMC) simulation package developed entirely from scratch using `Python` and `JAX`. Originally designed for molecular systems --with future extensions planned for periodic systems-- **jQMC** implements two well-established QMC algorithms: Variational Monte Carlo (VMC) and a robust and efficient variant of Diffusion Monte Carlo algorithm known as Lattice Regularized Diffusion Monte Carlo (LRDMC). By leveraging `JAX` just-in-time (`jit`) compilation and vectorized mapping (`vmap`) functionalities, `jQMC` achieves high-performance computations **especially on GPUs** while remaining portable across CPUs, GPUs, and TPUs. See [here](http://jax.readthedocs.io/) for the details of `JAX`. The **jQMC** users and developers manual is available from [GitHub Pages](https://kousuke-nakano.github.io/jQMC/).

![license](https://img.shields.io/github/license/kousuke-nakano/jQMC)
![tag](https://img.shields.io/github/v/tag/kousuke-nakano/jQMC)
![fork](https://img.shields.io/github/forks/kousuke-nakano/jQMC?style=social)
![stars](https://img.shields.io/github/stars/kousuke-nakano/jQMC?style=social)
![full-pytest](https://github.com/kousuke-nakano/jQMC/actions/workflows/jqmc-run-full-pytest.yml/badge.svg)
![codecov](https://codecov.io/github/kousuke-nakano/jQMC/graph/badge.svg?token=H0Z7M86C1E)
![DL](https://img.shields.io/pypi/dm/jqmc)
![python_version](https://img.shields.io/pypi/pyversions/jqmc)
![pypi_version](https://badge.fury.io/py/jqmc.svg)

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
- On CPUs, **`jQMC` is significantly slower than other QMC packages written in compiled languages (e.g., C++ or Fortran) although all the implemented functions are `jit`-compiled and `vmap`-vectorized by `JAX`. Further improvements are needed on both the algorithmic and implementation fronts. As this is an initial release, there remain many bottlenecks and hot spots to address.** Please use GPUs with a large number of walkers to achieve comparable speed.
- Atomic force calculations with **solid (sperical) harmonics GTOs** are much slower than energy and energy-optimization calculations due to the very slow compilations of dlnPsi/dR and de_L/dR. This is because `grad`, `jvp`, and `vjp` are slow for these terms for some reason. A more detailed analysis will be needed. Please use **cartesian GTOs** to do those calculations
- Periodic boundary condition calculations are not supoorted yet. It will be implemented in the future as `JAX` supports `complex128`. Work in progress.


## Ongoing works
- Improving the computational performance of VMC and LRDMC. The current implementation remains slow due to several reasons (especially for LRDMC); however, substantial speedups should be possible with code optimization.
- Implementing periodic boundary conditions (PBC), at least at the Gamma point (i.e., using a real-space wave function).


## Developer(s)
Kosuke Nakano (National Institute for Materials Science (NIMS), Japan)


## How to install jQMC

**The release version of jQMC** can be installed from `PyPI` via pip.

```bash
% pip install jqmc
```

**The latest version of jQMC** can be installed via pip from the cloned GitHub repository.

```bash
% git clone https://github.com/kousuke-nakano/jQMC
% cd jQMC
% pip install .
```


## Examples
Examples are in `examples` directory.


## Supporting HF/DFT packages
`jQMC` can prepare a trial (guiding) wavefunction from a `TREX-IO` file. Below is the list of HF/DFT packages that adopt `TREX-IO` for writing wave functions:

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


## Branches

 - `main`: main branch.
 - `devel*`: development branches.
 - `rc`: the latest stable version ready for deployment of the package.
 - `rc-gh-pages`: the latest stable version ready for deployment of the documentation.

Every time a change is pushed to the `main` or `devel*` branch, the `GitHub` workflow launches the implemented unit and integration tests (`jqmc-run-short-pytest.yml` and `jqmc-run-full-pytest.yml` for the `main` and `devel*` branches, respectively).


## How to deploy the package

Once the `main` branch is merged into the `rc` branch, the `GitHub` workflow launches the implemented unit and integration tests (`jqmc-run-full-pytest.yml`) and test a deployment using `test-PyPI`. Then, once a tag is attached to (the latest) commit in the `rc` branch, the `GitHub` workflow checks the tag format (PEP 440 with the starting v, e.g., v0.1.0b4, v0.1.1, v1.0) and deploy the package to `PyPI`.

## How to deploy the documentation

Once the `main` branch is merged into the `rc-gh-pages` branch, the `GitHub` workflow launches the implemented documentaion building process (`jqmc-deploy-gh-pages.yml`) and deploy the compiled documentaiton to [GitHub Pages](https://kousuke-nakano.github.io/jQMC/).

## Contribution

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.


## Formatting

Formatting rules are written in `pyproject.toml`.


## Pre-commit

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


## Citation of jQMC

If you used `jQMC` in your reseach project, please cite the following articles. This indeed helps the `jQMC` project to continue:

- "jQMC: JAX-based ab initio Quantum Monte Carlo package",

  Kousuke Nakano and Michele Casula, in preparation (2025)

  ```
  @article{jqmc,
    author  = {Nakano, Kousuke and Casula, Michele},
    title   = {jQMC: JAX-based ab initio Quantum Monte Carlo package},
    journal = {in preparation},
    %volume  = {},
    %number  = {},
    %pages   = {},
    year    = {2025},
    %doi     = {}
  }
  ```

- "Load-Balanced Diffusion Monte Carlo Method with Lattice Regularization",

  [K. Nakano, S. Sorella, and M. Casula, J. Chem. Phys. 163, 194117 (2025)](https://doi.org/10.1063/5.0296986)

  ```
  @article{10.1063/5.0296986,
      author = {Nakano, Kousuke and Sorella, Sandro and Casula, Michele},
      title = {Load-balanced diffusion Monte Carlo method with lattice regularization},
      journal = {J. Chem. Phys.},
      volume = {163},
      number = {19},
      pages = {194117},
      year = {2025},
      month = {11},
      doi = {10.1063/5.0296986}
  }
  ```
