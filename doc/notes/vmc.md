(sec_opt_wf)=
# Variational Monte Carlo (VMC)

(vmc_tags)=

## Variational Monte Carlo

The expectation value of the energy for a trial wave function $\Psi$ can be written as:

```{math}
:label: eq-vmc-energy
\langle E \rangle
= \frac{\displaystyle \int \mathrm{d}\mathbf{x}\,\Psi^2(\mathbf{x})\,\displaystyle\frac{\hat{\mathcal{H}}\Psi(\mathbf{x})}{\Psi(\mathbf{x})}}
       {\displaystyle \int \mathrm{d}\mathbf{x}\,\Psi^2(\mathbf{x})}
```

This is equivalently expressed as an average over the local energy $e_L(\mathbf{x})$ with probability density $\pi(\mathbf{x})$:

```{math}
\langle E \rangle = \int \mathrm{d}\mathbf{x}\;e_L(\mathbf{x})\,\pi(\mathbf{x})
```

Here:

* $\mathbf{x}=(\mathbf{r}_1\sigma_1,\mathbf{r}_2\sigma_2,\dots,\mathbf{r}_N\sigma_N)$ denotes all electron coordinates and spins.
* The **local energy** is defined by:

  ```{math}
  e_L(\mathbf{x}) = \frac{\hat{\mathcal{H}}\Psi(\mathbf{x})}{\Psi(\mathbf{x})}
  ```
* The **sampling probability** is:

  ```{math}
  \pi(\mathbf{x}) = \frac{\Psi^2(\mathbf{x})}{\displaystyle \int \mathrm{d}\mathbf{x}'\,\Psi^2(\mathbf{x}')}
  ```

To evaluate this multidimensional integral stochastically, one generates samples $\{\mathbf{x}_i\}$ according to $\pi(\mathbf{x})$ via Markov Chain Monte Carlo (MCMC) and computes the Monte Carlo estimator:

```{math}
:label: eq-eval-observable
E_{\mathrm{MCMC}} = \bigl\langle e_L(\mathbf{x})\bigr\rangle_{\pi} \approx \frac{1}{M} \sum_{i=1}^M e_L(\mathbf{x}_i)
```

The associated statistical error is:

```{math}
\sqrt{\frac{\mathrm{Var}[e_L(\mathbf{x}_i)]}{\tilde M}}
```

where:

* $\mathrm{Var}[e_L(\mathbf{x}_i)]$ is the variance of the local energies.
* $\tilde M = M / \tau_\mathrm{corr}$, with $\tau_\mathrm{corr}$ the autocorrelation time.

If $\Psi$ is an exact eigenfunction of $\hat{\mathcal{H}}$ with eigenvalue $E_0$, then $e_L(\mathbf{x})=E_0$ for all $\mathbf{x}$, giving zero variance and $E_{\mathrm{MCMC}} = E_0$ exactly (the **zero-variance property**).

By the **variational theorem**, the estimator provides an upper bound to the true ground-state energy $E_0$.  Introducing variational parameters $\boldsymbol{\alpha}=(\alpha_1,\dots,\alpha_p)$ into the trial wave function $\Psi(\mathbf{x},\boldsymbol{\alpha})$, one defines:

```{math}
E_{\mathrm{VMC}}(\boldsymbol{\alpha}) = \int \mathrm{d}\mathbf{x}\;e_L(\mathbf{x},\boldsymbol{\alpha})\,\pi(\mathbf{x},\boldsymbol{\alpha}) \ge E_0
```

This constitutes the Variational Monte Carlo (VMC) framework.

The optimization of $\boldsymbol{\alpha}$ is challenging due to a complex energy landscape with statistical noise.  jQMC leverages JAX automatic differentiation to compute energy derivatives and employs the **stochastic reconfiguration** method\~\cite{1998SOR,2007SOR} for efficient parameter updates.

### MCMC Sampling and Metropolis–Hastings

jQMC uses a generalized Metropolis–Hastings algorithm to sample $\pi(\mathbf{x})$.  A proposed move from $\vec{x}$ to $\vec{x}'$ is accepted with probability:

```{math}
P(\vec{x}\to\vec{x}') = \min\Bigl[1, \frac{|\Psi_T(\vec{x}')|^2}{|\Psi_T(\vec{x})|^2} \frac{T(\vec{x}'\to\vec{x})}{T(\vec{x}\to\vec{x}')},\Bigr]
```

where $\Psi_T$ is the trial wave function and $T$ are transition kernels.  In the accelerated scheme, a local move for electron $l$ is proposed with a position-dependent variance:

```{math}
f(\vec{r}_l) = \frac{1}{Z_{I(l)}^2|\vec{r}_l-\vec{R}_{I(l)}|} \frac{1 + Z_{I(l)}^2|\vec{r}_l-\vec{R}_{I(l)}|}{1 + |\vec{r}_l-\vec{R}_{I(l)}|}
```

and

```{math}
r'_{l,\gamma} = r_{l,\gamma} + g_l, \quad g_l\sim\mathcal{N}(0,\,f(\vec{r}_l)\Delta t).
```

The detailed-balance is ensured by Gaussian kernels:

```{math}
T(\vec{x}\to\vec{x}') =
\frac{1}{\sqrt{2\pi (f(\vec{r}_l)\Delta t)^2}} \exp\Bigl[-\frac{|\vec{r}'_l-\vec{r}_l|^2}{2 (f(\vec{r}_l)\Delta t)^2}\Bigr]
```

and similarly for $T(\vec{x}'\to\vec{x})$, giving the ratio:

```{math}
\frac{T(\vec{x}'\to\vec{x})}{T(\vec{x}\to\vec{x}')} = \frac{f(\vec{r}_l)}{f(\vec{r}'_l)} \exp\Bigl[-|\vec{r}_l-\vec{r}'_l|^2 (\frac{1}{2(f(\vec{r}'_l)\Delta t)^2} - \frac{1}{2(f(\vec{r}_l)\Delta t)^2})\Bigr].
```

### Reweighting and AS Regularization

jQMC implements the Attaccalite–Sorella (AS) reweighting to handle divergences near nodal surfaces\~\cite{2008ATT}.  One samples a guiding distribution $\Pi_G$ defined by:

```{math}
\Psi_G(\mathbf{x}) = \frac{R^\varepsilon(\mathbf{x})}{R(\mathbf{x})}\Psi_T(\mathbf{x})
```

with

```{math}
R(\mathbf{x}) = (S \sum_{i,j} |G^{-1}_{ij}|^2)^{-\theta_R},
```

where $G$ is the geminal matrix, $S=\min_i\sum_j|G_{ij}|^2$, and $\theta_R=3/8$.  Using the Frobenius norm and SVD:

```{math}
\|G^{-1}\|_F^2 = \sum_{k=1}^n \sigma_k^{-2}
```

avoids matrix inversion.  The regularization

```{math}
R^\varepsilon(\mathbf{x}) = \max[R(\mathbf{x}),\varepsilon]
```

ensures non-vanishing guiding weights.  Observables with weight $\mathcal{W}=(\Psi_T/\Psi_G)^2$ are estimated as:

```{math}
\frac{\langle O(\mathbf{x})\,\mathcal{W}(\mathbf{x})\rangle_{\Pi_G}}{\langle \mathcal{W}(\mathbf{x})\rangle_{\Pi_G}}.
```

The parameter $\varepsilon$ is chosen so that the average weight $\langle\mathcal{W}\rangle\approx0.8$.
