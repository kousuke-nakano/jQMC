# Overview

**jQMC** is an ab initio quantum Monte Carlo (QMC) simulation package developed entirely from scratch using Python and JAX. Originally designed for molecular systems--with future extensions planned for periodic systems--**jQMC** implements two well-established QMC algorithms: Variational Monte Carlo (VMC) and a robust and efficient variant of Diffusion Monte Carlo known as Lattice Regularized Diffusion Monte Carlo (LRDMC). By leveraging JAX just-in-time (jit) compilation and vectorized mapping (vmap) functionalities, jQMC achieves high-performance computations especially on GPUs while remaining portable across CPUs and GPUs.

![license](https://img.shields.io/github/license/jqmc-project/jQMC)
![tag](https://img.shields.io/github/v/tag/jqmc-project/jQMC)
![fork](https://img.shields.io/github/forks/jqmc-project/jQMC?style=social)
![stars](https://img.shields.io/github/stars/jqmc-project/jQMC?style=social)
![full-pytest](https://github.com/jqmc-project/jQMC/actions/workflows/jqmc-run-full-pytest.yml/badge.svg)
![codecov](https://codecov.io/github/jqmc-project/jQMC/graph/badge.svg)
![DL](https://img.shields.io/pypi/dm/jqmc)
![python_version](https://img.shields.io/pypi/pyversions/jqmc)
![pypi_version](https://badge.fury.io/py/jqmc.svg)
