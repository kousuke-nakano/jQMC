# Lattice Regularized Diffusion Monte Carlo (LRDMC)

(lrdmc_tags)=

## Overview

Lattice regularized diffusion Monte Carlo (LRDMC), initially proposed by Casula[^1], is a projection technique that systematically improves a variational Ansatz. It is based on Green's function Monte Carlo (GFMC)[^2][^3][^4], filtering the ground-state wavefunction from a given trial wavefunction.

## Practical points

1. No time-step error: unlike standard DMC, LRDMC does not rely on a Suzuki–Trotter decomposition. Instead, a systematic bias comes from the finite lattice spacing $a$[^1]. To obtain an unbiased fixed-node (FN) energy, extrapolate results to $a \to 0$ using several lattice spacings; the $a \to 0$ extrapolation is typically smooth and well captured by low-order polynomial fits.
2. Consistency with DMC: after removing the controllable $a \to 0$ extrapolation, FN energies from LRDMC agree with standard DMC calculations[^5].
3. Multiple mesh sizes: LRDMC can introduce two mesh sizes ($a$ and $a'$) so that regions near nuclei and valence regions are diffused appropriately[^1][^6]. This will be introduced into `jQMC` in future.
4. Variational principle with ECPs: LRDMC retains the variational principle even in the presence of effective core potentials, analogous to the T-move treatment in standard DMC[^1][^7][^8].

`jQMC` implements an LRDMC algorithm that maintains parallel efficiency for many walkers and many nodes via load-balancing across walkers[^9].

For further algorithmic details, please see the textbook “Quantum Monte Carlo Approaches for Correlated Systems”[^10].

[^1]: M. Casula, C. Filippi, and S. Sorella, Phys. Rev. Lett. 95, 100201 (2005). DOI: [10.1103/PhysRevLett.95.100201](https://doi.org/10.1103/PhysRevLett.95.100201)
[^2]: D. F. Ten Haaf, H. J. Van Bemmel, J. M. Van Leeuwen, W. Van Saarloos, and D. M. Ceperley, Phys. Rev. B 51, 13039 (1995). DOI: [10.1103/PhysRevB.51.13039](https://doi.org/10.1103/PhysRevB.51.13039)
[^3]: M. Calandra Buonaura and S. Sorella, Phys. Rev. B 57, 11446 (1998). DOI: [10.1103/PhysRevB.57.11446](https://doi.org/10.1103/PhysRevB.57.11446)
[^4]: S. Sorella and L. Capriotti, Phys. Rev. B 61, 2599 (2000). DOI: [10.1103/PhysRevB.61.2599](https://doi.org/10.1103/PhysRevB.61.2599)
[^5]: F. Della Pia, et al., J. Chem. Phys. 163, 104110 (2025). DOI: [10.1063/5.0272974](https://doi.org/10.1063/5.0272974)
[^6]: K. Nakano, R. Maezono, and S. Sorella, Phys. Rev. B 101, 155106 (2020). DOI: [10.1103/PhysRevB.101.155106](https://doi.org/10.1103/PhysRevB.101.155106)
[^7]: K. Nakano, S. Sorella, D. Alfè, A. Zen, J. Chem. Theory Comput. 20, 4591-4604 (2024). DOI: [10.1021/acs.jctc.4c00139](https://doi.org/10.1021/acs.jctc.4c00139)
[^8]: M. Casula, Phys. Rev. B 74, 161102 (2006). DOI: [10.1103/PhysRevB.74.161102](https://doi.org/10.1103/PhysRevB.74.161102)
[^9]: K. Nakano, S. Sorella, M. Casula, J. Chem. Phys. 163, 194117 (2025). DOI: [10.1063/5.0296986](https://doi.org/10.1063/5.0296986)
[^10]: “Quantum Monte Carlo Approaches for Correlated Systems”, Cambridge University Press (2017). DOI: [10.1017/9781316417041](https://doi.org/10.1017/9781316417041)
