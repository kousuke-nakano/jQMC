# Binding energy of water-methane dimer
Here, we show one of the validation tests for our **ab initio** DMC implementation. See the [paper](https://doi.org/10.48550/arXiv.2501.12950) for details. For the benchmark test, the correlation-consistent effective core potentials (ccECPs) and the corresponding triple-zeta basis set (ccECP-cc-pVTZ) were used. The orbitals used with the Slater–Jastrow ansatz were obtained from DFT calculations using the Perdew–Zunger parameterization of the local-density approximation. The Jastrow factors are taken from the corresponding TurboRVB WFs.

In this benchmark calculation, we employed the locality approximation via the T-move algorithm with DLA (DTM). From the figure and table below, we first demonstrate that the “textbook” algorithm implemented in jQMC and our load-balancing algorithm yield identical results in the limit $a \to 0$. Furthermore, those values agree to very high precision with the validation test proposed in the above paper, as performed by other codes (CASINO, QMCPACK, TurboRVB).

![LRDMC validation](jqmc_validation_water_methane.jpg)

| Package               | Methane (Ha) | Water (Ha)   | Methane–Water (Ha) | Binding energy (meV) |
| --------------------- | ------------ | ------------ | ------------------ | -------------------- |
| CASINO                | -8.07856(1)  | -17.23473(2) | -25.31432(3)       | -26.8(1.0)           |
| QMCPACK               | -8.07858(2)  | -17.23482(3) | -25.31443(7)       | -29.0(1.1)           |
| TurboRVB              | -8.07860(1)  | -17.23479(1) | -25.31445(1)       | -28.1(0.3)           |
| jQMC (textbook)       | XXX          | XXX          | XXX                | XXX                  |
| jQMC (load-balancing) | XXX          | XXX          | XXX                | XXX                  |


