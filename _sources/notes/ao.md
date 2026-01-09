(sec:atomic_orbitals)=
# Atomic orbitals (AOs) and Basis sets

## Atomic orbitals for the pairing function and the Jastrow factor

One of the most common choices for atomic orbitals in QMC is atom‑centered Gaussian‑type orbitals (GTOs). A primitive GTO $\psi_{l,m,\alpha}(\mathbf{r})$ can be constructed using either solid harmonics or Cartesian polynomial basis functions.

### Gaussian‑Type Orbitals with Solid‑Harmonic Basis

Primitive orbitals with regular solid harmonics are given by:

```{math}
\psi_{l,m,\alpha}(\mathbf{r}) = \mathcal{N}^{\rm solid}_{l,m,\alpha}\;\mathcal{R}_{\alpha}(\mathbf{r})\;\mathcal{S}_{l,m,\alpha}(\mathbf{r})
```

where the radial part is

```{math}
\mathcal{R}_{\alpha}(\mathbf{r}) = e^{-Z_\alpha\,|\mathbf{r}-\mathbf{R}_\alpha|^2}
```

and the solid harmonics are

```{math}
\mathcal{S}_{l,m,\alpha}(\mathbf{r}) = \sqrt{\frac{2l+1}{4\pi}}\;|\mathbf{r}-\mathbf{R}_\alpha|^l\;\mathcal{Y}_{l,m}(\theta_\alpha,\phi_\alpha).
```

Here $\mathcal{Y}_{l,m}(\theta,\phi)$ are real spherical harmonics, and the Racah normalization ensures

```{math}
\int_0^\pi\!\int_0^{2\pi} \sin\theta\,\mathcal{S}_{l,m,\alpha}^*(\mathbf{r})\,\mathcal{S}_{l,m,\alpha}(\mathbf{r})\,\mathrm{d}\phi\,\mathrm{d}\theta = \frac{4\pi}{2l+1}\;r^{2l}.
```

The normalization constant is

```{math}
\mathcal{N}^{\rm solid}_{l,m,\alpha} = \sqrt{\frac{(2Z_\alpha/\pi)^{3/2}\,(4Z_\alpha)^l}{(2l-1)!!}}.
```

One may also write GTOs with spherical harmonics normalization

```{math}
\psi_{l,m,\alpha}(\mathbf{r}) = \mathcal{N}^{\rm sphe}_{l,m,\alpha}\;|\mathbf{r}-\mathbf{R}_\alpha|^l\;\mathcal{R}_{\alpha}(\mathbf{r})\;\mathcal{Y}_{l,m}(\theta_\alpha,\phi_\alpha)
```

with

```{math}
\mathcal{N}^{\rm sphe}_{l,m,\alpha} = \sqrt{\frac{2^{2l+3}(l+1)!(2Z_\alpha)^{l+3/2}}{(2l+2)!\,\sqrt{\pi}}}.
```

These two normalizations satisfy

```{math}
\frac{\mathcal{N}^{\rm solid}_{l,m,\alpha}}{\mathcal{N}^{\rm sphe}_{l,m,\alpha}} = \sqrt{\frac{2l+1}{4\pi}}.
```

### Gaussian‑Type Orbitals with Cartesian Basis

Primitive orbitals in the Cartesian basis are

```{math}
\psi_{n_x,n_y,n_z,\alpha}(\mathbf{r}) = \mathcal{N}^{\rm cart}_{n_x,n_y,n_z,\alpha}\;e^{-Z_\alpha\,|\mathbf{r}-\mathbf{R}_\alpha|^2}\;x^{n_x}y^{n_y}z^{n_z}.
```

The normalization is

```{math}
\mathcal{N}^{\rm cart}_{n_x,n_y,n_z,\alpha} = \sqrt{\frac{(2Z_\alpha/\pi)^{3/2}\,(4Z_\alpha)^{n_x+n_y+n_z}}{(2n_x-1)!!\,(2n_y-1)!!\,(2n_z-1)!!}} = \sqrt{\frac{(2Z_\alpha/\pi)^{3/2}\,(8Z_\alpha)^{n_x+n_y+n_z}n_x!n_y!n_z!}{(2n_x)!(2n_y)!(2n_z)!}}.
```

We define the total angular momentum as $l=n_x+n_y+n_z$.  A basis of order $l$ includes all monomials of that degree (e.g., $l=2$ gives $d_{xx},d_{xy},\dots,d_{zz}$).  Note that normalization constants differ among Cartesian orbitals with the same $l$ (except for $s$ and $p$ shells).  For example:

```{math}
\mathcal{N}^{\rm cart}_{2,0,0,\alpha} = \sqrt{\frac{(2Z_\alpha/\pi)^{3/2}(4Z_\alpha)^2}{3\cdot1}},
```

```{math}
\mathcal{N}^{\rm cart}_{1,1,0,\alpha} = \sqrt{\frac{(2Z_\alpha/\pi)^{3/2}(4Z_\alpha)^2}{1\cdot1}}.
```

### Practical Tip

In jQMC (JAX), Cartesian GTOs are computationally faster than spherical ones when using `jit` and `grad`, since they avoid branching logic by varying only polynomial exponents rather than basis‑function forms.


## Real Spherical and Solid Harmonics

The **real spherical harmonics** $\mathcal{Y}_{l,m}(\theta,\phi)$ are built from the complex spherical harmonics $Y_{l,m}(\theta,\phi)$ with the Condon–Shortley phase:

```{math}
:label: eq-real-spherical
\mathcal{Y}_{l,m}(\theta,\phi) =
\begin{cases}
\displaystyle
\frac{1}{\sqrt{2}}\bigl(Y_{l,-|m|}(\theta,\phi) + (-1)^m\,Y_{l,|m|}(\theta,\phi)\bigr), & m>0,\\[1em]
Y_{l,0}(\theta,\phi), & m=0,\\[0.75em]
\displaystyle
\frac{i}{\sqrt{2}}\bigl(Y_{l,-|m|}(\theta,\phi) - (-1)^m\,Y_{l,|m|}(\theta,\phi)\bigr), & m<0.
\end{cases}
```

Because the complex spherical harmonics satisfy

```{math}
:label: eq-ortho-complex
\int_{0}^{\pi}\!\!\int_{0}^{2\pi}
Y_{l',m'}^*(\theta,\phi)\,Y_{l,m}(\theta,\phi)\,
\sin\theta\,d\phi\,d\theta
= \delta_{l,l'}\,\delta_{m,m'},
```

the real ones are also orthonormal:

```{math}
:label: eq-ortho-real
\int_{0}^{\pi}\!\!\int_{0}^{2\pi}
\mathcal{Y}_{l',m'}(\theta,\phi)\,\mathcal{Y}_{l,m}(\theta,\phi)\,
\sin\theta\,d\phi\,d\theta
= \delta_{l,l'}\,\delta_{m,m'}.
```

The spherical harmonics have singularities at the origin $(x,y,z)=(0,0,0)$. In practice one uses the **regular solid harmonics** centered at $\mathbf{R}_\alpha$:

```{math}
:label: eq-solid-harmonic-def
\mathcal{S}_{l,m,\alpha}(\mathbf{r})
= \sqrt{\frac{2l+1}{4\pi}}\;
|\mathbf{R}_\alpha-\mathbf{r}|^l\;
\mathcal{Y}_{l,m}(\theta_\alpha,\phi_\alpha).
```

### Polynomial Form of Solid Harmonics

One efficient way to compute $\mathcal{S}_{l,m}$ is via polynomials in $(x,y,z)$. Define:

```{math}
:label: eq-Cm
C_m(x,y) =
\begin{cases}
\displaystyle
\sum_{p=0}^{|m|}\binom{|m|}{p}x^p\,y^{|m|-p}\,\cos\!\bigl(\tfrac\pi2(|m|-p)\bigr),
& m\ge0,\\[0.75em]
\displaystyle
\sum_{p=0}^{|m|}\binom{|m|}{p}x^p\,y^{|m|-p}\,\sin\!\bigl(\tfrac\pi2(|m|-p)\bigr),
& m<0,
\end{cases}
```

```{math}
:label: eq-gamma-lmk
\gamma_{l,m,k}(z)
= (-1)^k\,2^{-l}
\,\binom{l}{k}\,\binom{2l-2k}{l}\,
\frac{(l-2k)!}{(l-2k-|m|)!},
```

```{math}
:label: eq-Gamma-lm
\Gamma_{l,m}(z)
= \sum_{k=0}^{\lfloor (l-|m|)/2\rfloor}
\gamma_{l,m,k}(z)\;
|\mathbf{r}|^{2k}\;z^{\,l-2k-|m|}.
```

Then

```{math}
:label: eq-solid-harmonic-poly
\mathcal{S}_{l,m,\alpha}(\mathbf{r})
= \sqrt{\frac{(2-\delta_{m,0})(l-|m|)!}{(l+|m|)!}}\;
C_m(x-\!X_\alpha,\;y-\!Y_\alpha)\;
\Gamma_{l,m}(z-\!Z_\alpha).
```

### Normalization of the Radial Part

For a Gaussian radial factor
$\psi_{l,m,\alpha}(\mathbf{r})=\mathcal{N}_{l,\alpha}\,r^l\,e^{-Z_\alpha r^2}\,\mathcal{Y}_{l,m}(\theta,\phi)$,
the overall normalization requires

```{math}
:label: eq-normalization-integral
\int_{\mathbb{R}^3}
\bigl|\psi_{l,m,\alpha}(\mathbf{r})\bigr|^2\,d\mathbf{r}
=1.
```

Separating radial and angular parts:

```{math}
:label: eq-radial-integral
\mathcal{N}_{l,\alpha}^2
\int_{0}^{\infty}r^{2l+2}\,e^{-2Z_\alpha r^2}\,dr
=1
\;\Longrightarrow\;
\mathcal{N}_{l,\alpha}^2
\frac{(2l+2)!\,\sqrt{\pi}}{2^{2l+3}(l+1)!\,(2Z_\alpha)^{l+\frac{3}{2}}}
=1,
```

```{math}
:label: eq-angular-integral
\int_{0}^{\pi}\!\!\int_{0}^{2\pi}
\bigl|\mathcal{Y}_{l,m}(\theta,\phi)\bigr|^2\,
\sin\theta\,d\phi\,d\theta
=1.
```

Hence the **radial normalization constant** is

```{math}
:label: eq-normalization-factor
\mathcal{N}_{l,\alpha}
= \sqrt{
\frac{2^{2l+3}(l+1)!\,(2Z_\alpha)^{\,l+\tfrac{3}{2}}}
     {(2l+2)!\,\sqrt{\pi}}
}.
```

This factor is *identical* for all $(l,m)$ in the same shell.

### Practical Implementation Note

In JAX-based codes, **Cartesian GTOs** (polynomials in $x,y,z$) are often faster than spherical ones, because they avoid branching logic over $(l,m)$.

---

## Tables of Real Solid Harmonics

Below are the explicit real solid harmonics up to $l=6$.  Let $r=\sqrt{x^2+y^2+z^2}$.

### $l=0$ ($s$ orbital)

```{math}
:label: eq-l0
Y_{0,0} = \frac{1}{2}\,\sqrt{\frac{1}{\pi}}.
```

### $l=1$ ($p$ orbitals)

```{math}
:label: eq-l1
\begin{aligned}
Y_{1,-1} &= i\sqrt{\tfrac12}\bigl(Y_1^{-1}+Y_1^1\bigr)
= \sqrt{\tfrac{3}{4\pi}}\;\frac{y}{r},\\
Y_{1,0}  &= \;\;Y_1^0
= \sqrt{\tfrac{3}{4\pi}}\;\frac{z}{r},\\
Y_{1,1}  &= \sqrt{\tfrac12}\bigl(Y_1^{-1}-Y_1^1\bigr)
= \sqrt{\tfrac{3}{4\pi}}\;\frac{x}{r}.
\end{aligned}
```

### $l=2$ ($d$ orbitals)

```{math}
:label: eq-l2
\begin{aligned}
Y_{2,-2} &= i\sqrt{\tfrac12}\bigl(Y_2^{-2}-Y_2^2\bigr)
= \tfrac12\sqrt{\tfrac{15}{\pi}}\;\frac{xy}{r^2},\\
Y_{2,-1} &= i\sqrt{\tfrac12}\bigl(Y_2^{-1}+Y_2^1\bigr)
= \tfrac12\sqrt{\tfrac{15}{\pi}}\;\frac{yz}{r^2},\\
Y_{2,0}  &= \;\;Y_2^0
= \tfrac14\sqrt{\tfrac{5}{\pi}}\;\frac{3z^2-r^2}{r^2},\\
Y_{2,1}  &= \sqrt{\tfrac12}\bigl(Y_2^{-1}-Y_2^1\bigr)
= \tfrac12\sqrt{\tfrac{15}{\pi}}\;\frac{xz}{r^2},\\
Y_{2,2}  &= \sqrt{\tfrac12}\bigl(Y_2^{-2}+Y_2^2\bigr)
= \tfrac14\sqrt{\tfrac{15}{\pi}}\;\frac{x^2-y^2}{r^2}.
\end{aligned}
```

### $l=3$ ($f$ orbitals)

```{math}
:label: eq-l3
\begin{aligned}
Y_{3,-3} &= i\sqrt{\tfrac12}\bigl(Y_3^{-3}+Y_3^3\bigr)
= \tfrac14\sqrt{\tfrac{35}{2\pi}}\;\frac{y(3x^2-y^2)}{r^3},\\
Y_{3,-2} &= i\sqrt{\tfrac12}\bigl(Y_3^{-2}-Y_3^2\bigr)
= \tfrac12\sqrt{\tfrac{105}{\pi}}\;\frac{xyz}{r^3},\\
Y_{3,-1} &= i\sqrt{\tfrac12}\bigl(Y_3^{-1}+Y_3^1\bigr)
= \tfrac14\sqrt{\tfrac{21}{2\pi}}\;\frac{y(5z^2-r^2)}{r^3},\\
Y_{3,0}  &= \;\;Y_3^0
= \tfrac14\sqrt{\tfrac{7}{\pi}}\;\frac{5z^3-3zr^2}{r^3},\\
Y_{3,1}  &= \sqrt{\tfrac12}\bigl(Y_3^{-1}-Y_3^1\bigr)
= \tfrac14\sqrt{\tfrac{21}{2\pi}}\;\frac{x(5z^2-r^2)}{r^3},\\
Y_{3,2}  &= \sqrt{\tfrac12}\bigl(Y_3^{-2}+Y_3^2\bigr)
= \tfrac14\sqrt{\tfrac{105}{\pi}}\;\frac{(x^2-y^2)z}{r^3},\\
Y_{3,3}  &= \sqrt{\tfrac12}\bigl(Y_3^{-3}-Y_3^3\bigr)
= \tfrac14\sqrt{\tfrac{35}{2\pi}}\;\frac{x(x^2-3y^2)}{r^3}.
\end{aligned}
```

### $l=4$ ($g$ orbitals)

```{math}
:label: eq-l4
\begin{aligned}
Y_{4,-4} &= i\sqrt{\tfrac12}\bigl(Y_4^{-4}-Y_4^4\bigr)
= \tfrac34\sqrt{\tfrac{35}{\pi}}\;\frac{xy(x^2-y^2)}{r^4},\\
Y_{4,-3} &= i\sqrt{\tfrac12}\bigl(Y_4^{-3}+Y_4^3\bigr)
= \tfrac34\sqrt{\tfrac{35}{2\pi}}\;\frac{y(3x^2-y^2)z}{r^4},\\
Y_{4,-2} &= i\sqrt{\tfrac12}\bigl(Y_4^{-2}-Y_4^2\bigr)
= \tfrac34\sqrt{\tfrac{5}{\pi}}\;\frac{xy(7z^2-r^2)}{r^4},\\
Y_{4,-1} &= i\sqrt{\tfrac12}\bigl(Y_4^{-1}+Y_4^1\bigr)
= \tfrac34\sqrt{\tfrac{5}{2\pi}}\;\frac{y(7z^3-3zr^2)}{r^4},\\
Y_{4,0}  &= \;\;Y_4^0
= \tfrac{3}{16}\sqrt{\tfrac{1}{\pi}}\;\frac{35z^4-30z^2r^2+3r^4}{r^4},\\
Y_{4,1}  &= \sqrt{\tfrac12}\bigl(Y_4^{-1}-Y_4^1\bigr)
= \tfrac34\sqrt{\tfrac{5}{2\pi}}\;\frac{x(7z^3-3zr^2)}{r^4},\\
Y_{4,2}  &= \sqrt{\tfrac12}\bigl(Y_4^{-2}+Y_4^2\bigr)
= \tfrac38\sqrt{\tfrac{5}{\pi}}\;\frac{(x^2-y^2)(7z^2-r^2)}{r^4},\\
Y_{4,3}  &= \sqrt{\tfrac12}\bigl(Y_4^{-3}-Y_4^3\bigr)
= \tfrac34\sqrt{\tfrac{35}{2\pi}}\;\frac{x(x^2-3y^2)z}{r^4},\\
Y_{4,4}  &= \sqrt{\tfrac12}\bigl(Y_4^{-4}+Y_4^4\bigr)
= \tfrac{3}{16}\sqrt{\tfrac{35}{\pi}}\;\frac{x^2(x^2-3y^2)-y^2(3x^2-y^2)}{r^4}.
\end{aligned}
```

### $l=5$ ($h$ orbitals)

```{math}
:label: eq-l5
\begin{aligned}
Y_{5,-5} &= i\sqrt{\tfrac12}\bigl(Y_5^{-5}+Y_5^5\bigr)
= \tfrac{3}{16}\sqrt{\tfrac{77}{2\pi}}\;\frac{5x^4y-10x^2y^3+y^5}{r^5},\\
Y_{5,-4} &= i\sqrt{\tfrac12}\bigl(Y_5^{-4}-Y_5^4\bigr)
= \tfrac{3}{16}\sqrt{\tfrac{385}{\pi}}\;\frac{4xyz(x^2-y^2)}{r^5},\\
Y_{5,-3} &= i\sqrt{\tfrac12}\bigl(Y_5^{-3}+Y_5^3\bigr)
= \tfrac{1}{16}\sqrt{\tfrac{385}{2\pi}}\;\frac{-(y^3-3x^2y)(9z^2-r^2)}{r^5},\\
Y_{5,-2} &= i\sqrt{\tfrac12}\bigl(Y_5^{-2}-Y_5^2\bigr)
= \tfrac{1}{8}\sqrt{\tfrac{1155}{\pi}}\;\frac{2xy(3z^3-zr^2)}{r^5},\\
Y_{5,-1} &= i\sqrt{\tfrac12}\bigl(Y_5^{-1}+Y_5^1\bigr)
= \tfrac{1}{16}\sqrt{\tfrac{165}{\pi}}\;\frac{y(21z^4-14z^2r^2+r^4)}{r^5},\\
Y_{5,0}  &= \;\;Y_5^0
= \tfrac{1}{16}\sqrt{\tfrac{11}{\pi}}\;\frac{63z^5-70z^3r^2+15zr^4}{r^5},\\
Y_{5,1}  &= \sqrt{\tfrac12}\bigl(Y_5^{-1}-Y_5^1\bigr)
= \tfrac{1}{16}\sqrt{\tfrac{165}{\pi}}\;\frac{x(21z^4-14z^2r^2+r^4)}{r^5},\\
Y_{5,2}  &= \sqrt{\tfrac12}\bigl(Y_5^{-2}+Y_5^2\bigr)
= \tfrac{1}{8}\sqrt{\tfrac{1155}{\pi}}\;\frac{(x^2-y^2)(3z^3-zr^2)}{r^5},\\
Y_{5,3}  &= \sqrt{\tfrac12}\bigl(Y_5^{-3}-Y_5^3\bigr)
= \tfrac{1}{16}\sqrt{\tfrac{385}{2\pi}}\;\frac{(x^3-3xy^2)(9z^2-r^2)}{r^5},\\
Y_{5,4}  &= \sqrt{\tfrac12}\bigl(Y_5^{-4}+Y_5^4\bigr)
= \tfrac{3}{16}\sqrt{\tfrac{385}{\pi}}\;\frac{x^2z(x^2-3y^2)-y^2z(3x^2-y^2)}{r^5},\\
Y_{5,5}  &= \sqrt{\tfrac12}\bigl(Y_5^{-5}-Y_5^5\bigr)
= \tfrac{3}{16}\sqrt{\tfrac{77}{2\pi}}\;\frac{x^5-10x^3y^2+5xy^4}{r^5}.
\end{aligned}
```

### $l=6$ ($i$ orbitals)

```{math}
:label: eq-l6
\begin{aligned}
Y_{6,-6} &= i\sqrt{\tfrac12}\bigl(Y_6^{-6}-Y_6^6\bigr)
= \tfrac{1}{64}\sqrt{\tfrac{6006}{\pi}}\;\frac{6x^5y-20x^3y^3+6xy^5}{r^6},\\
Y_{6,-5} &= i\sqrt{\tfrac12}\bigl(Y_6^{-5}+Y_6^5\bigr)
= \tfrac{3}{32}\sqrt{\tfrac{2002}{\pi}}\;\frac{z(5x^4y-10x^2y^3+y^5)}{r^6},\\
Y_{6,-4} &= i\sqrt{\tfrac12}\bigl(Y_6^{-4}-Y_6^4\bigr)
= \tfrac{3}{32}\sqrt{\tfrac{91}{\pi}}\;\frac{4xy(11z^2-r^2)(x^2-y^2)}{r^6},\\
Y_{6,-3} &= i\sqrt{\tfrac12}\bigl(Y_6^{-3}+Y_6^3\bigr)
= \tfrac{1}{32}\sqrt{\tfrac{2730}{\pi}}\;\frac{-(11z^3-3zr^2)(y^3-3x^2y)}{r^6},\\
Y_{6,-2} &= i\sqrt{\tfrac12}\bigl(Y_6^{-2}-Y_6^2\bigr)
= \tfrac{1}{64}\sqrt{\tfrac{2730}{\pi}}\;\frac{2xy(33z^4-18z^2r^2+r^4)}{r^6},\\
Y_{6,-1} &= i\sqrt{\tfrac12}\bigl(Y_6^{-1}+Y_6^1\bigr)
= \tfrac{1}{16}\sqrt{\tfrac{273}{\pi}}\;\frac{y(33z^5-30z^3r^2+5zr^4)}{r^6},\\
Y_{6,0}  &= \;\;Y_6^0
= \tfrac{1}{32}\sqrt{\tfrac{13}{\pi}}\;\frac{231z^6-315z^4r^2+105z^2r^4-5r^6}{r^6},\\
Y_{6,1}  &= \sqrt{\tfrac12}\bigl(Y_6^{-1}-Y_6^1\bigr)
= \tfrac{1}{16}\sqrt{\tfrac{273}{\pi}}\;\frac{x(33z^5-30z^3r^2+5zr^4)}{r^6},\\
Y_{6,2}  &= \sqrt{\tfrac12}\bigl(Y_6^{-2}+Y_6^2\bigr)
= \tfrac{1}{64}\sqrt{\tfrac{2730}{\pi}}\;\frac{(x^2-y^2)(33z^4-18z^2r^2+r^4)}{r^6},\\
Y_{6,3}  &= \sqrt{\tfrac12}\bigl(Y_6^{-3}-Y_6^3\bigr)
= \tfrac{1}{32}\sqrt{\tfrac{2730}{\pi}}\;\frac{(11z^3-3zr^2)(x^3-3xy^2)}{r^6},\\
Y_{6,4}  &= \sqrt{\tfrac12}\bigl(Y_6^{-4}+Y_6^4\bigr)
= \tfrac{3}{32}\sqrt{\tfrac{91}{\pi}}\;\frac{(11z^2-r^2)\bigl[x^2(x^2-3y^2)+y^2(y^2-3x^2)\bigr]}{r^6},\\
Y_{6,5}  &= \sqrt{\tfrac12}\bigl(Y_6^{-5}-Y_6^5\bigr)
= \tfrac{3}{32}\sqrt{\tfrac{2002}{\pi}}\;\frac{z(x^5-10x^3y^2+5xy^4)}{r^6},\\
Y_{6,6}  &= \sqrt{\tfrac12}\bigl(Y_6^{-6}+Y_6^6\bigr)
= \tfrac{1}{64}\sqrt{\tfrac{6006}{\pi}}\;\frac{x^6-15x^4y^2+15x^2y^4-y^6}{r^6}.
\end{aligned}
```

*$l\ge7$ harmonics are rarely needed in practice.*
