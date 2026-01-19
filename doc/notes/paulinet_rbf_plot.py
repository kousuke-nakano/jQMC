import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 1.5

num_rbf = 11
cutoff = 5.0

# Simulate the logic in jastrow_factor.py
# q = jnp.linspace(0.0, 1.0, self.num_rbf + 2, dtype=distances.dtype)[1:-1]
q = np.linspace(0.0, 1.0, num_rbf + 2)[1:-1]

# mu = self.cutoff * q**2
mu = cutoff * q**2

# sigma = (1.0 / 7.0) * (1.0 + self.cutoff * q)
sigma = (1.0 / 7.0) * (1.0 + cutoff * q)

print("q:", q)
print("mu:", mu)
print("sigma:", sigma)

r = np.linspace(0, 7.0, 500)

plt.figure(figsize=(5, 5), facecolor="white", dpi=600)

# r needs to be broadcast against scalar parameters for each k
# features = (d**2) * jnp.exp(-d - ((d - mu) ** 2) / (sigma**2 + 1e-12))

for k in range(num_rbf):
    mu_k = mu[k]
    sigma_k = sigma[k]

    # Calculate e_k(r)
    term_gaussian = np.exp(-((r - mu_k) ** 2) / (sigma_k**2 + 1e-12))
    term_exp_r = np.exp(-r)
    prefactor = r**2

    y = prefactor * term_exp_r * term_gaussian

    plt.plot(r, y, label=rf"k={k + 1}, $\mu$={mu_k:.2f}")

plt.title(f"PauliNet RBF features vector (K={num_rbf}, $r_c$={cutoff})")
plt.xlabel("r (a.u.)")
plt.ylabel("$e_k(r)$")
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("paulinet_rbf_plot.pdf")
plt.savefig("paulinet_rbf_plot.png", dpi=300)
print("Plot saved to paulinet_rbf_plot.pdf and .png")
