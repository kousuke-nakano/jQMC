# Overview

**jQMC** is an ab initio quantum Monte Carlo (QMC) simulation package developed entirely from scratch using Python and JAX. Originally designed for molecular systems—with future extensions planned for periodic systems—**jQMC** implements two well-established QMC algorithms: Variational Monte Carlo (VMC) and a robust and efficient variant of Diffusion Monte Carlo known as Lattice Regularized Diffusion Monte Carlo (LRDMC). By leveraging JAX just-in-time (jit) compilation and vectorized mapping (vmap) functionalities, jQMC achieves high-performance computations especially on GPUs while remaining portable across CPUs and GPUs.

![license](https://img.shields.io/github/license/kousuke-nakano/jQMC)
![tag](https://img.shields.io/github/v/tag/kousuke-nakano/jQMC)
![fork](https://img.shields.io/github/forks/kousuke-nakano/jQMC?style=social)
![stars](https://img.shields.io/github/stars/kousuke-nakano/jQMC?style=social)
![full-pytest](https://github.com/kousuke-nakano/jQMC/actions/workflows/jqmc-run-full-pytest.yml/badge.svg)
![codecov](https://codecov.io/github/kousuke-nakano/jQMC/graph/badge.svg?token=H0Z7M86C1E)
![DL](https://img.shields.io/pypi/dm/jqmc)
![python_version](https://img.shields.io/pypi/pyversions/jqmc)
![pypi_version](https://badge.fury.io/py/jqmc.svg)
