# References

## Method used in jQMC

(reference_stochastic_reconfiguration)=

### Stochastic Reconfiguration

In jQMC, wavefunctions are optimized using the so-called Stochastic Reconfiguration method, published in.

- [S. Sorella, Phys. Rev. Lett. 80, 4558 (1998)](https://doi.org/10.1103/PhysRevLett.80.4558)
- [S. Sorella, M. Casula, and D. Rocca, J. Chem. Phys. 127, 014105 (2007)](https://doi.org/10.1063/1.2746035).


(reference_lrdmc)=

### Diffusion Monte Carlo Method with Lattice Regularization (LRDMC)

Two LRDMC algorithms are implemented in jQMC. The first is the original LRDMC algorithm, referred to as `lrdmc-tau` in `jQMC`, which is described in detail below:

- [M. Casula, C. Filippi, and S. Sorella, Phys. Rev. Lett. 95, 100201 (2005)](https://doi.org/10.1103/PhysRevLett.95.100201)

The second, referred to as `lrdmc` in `jQMC`, is a slightly modified version of the original algorithm designed to maintain better load balancing across walkers. Details are provided below:

- [K. Nakano, S. Sorella, and M. Casula, J. Chem. Phys. 163, 194117 (2025)](https://doi.org/10.1063/5.0296986)

Although the latter algorithm (`lrdmc`) offers superior parallel efficiency, both implementations are algorithmically equivalent and have been verified to yield consistent results.
