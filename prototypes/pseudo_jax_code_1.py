# import numpy as np
import jax.numpy as jnp
from jax import grad


class Laplacian:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def compute(self, r):
        a = self.a
        b = self.b

        # return a * np.sin(b) * r
        return a * jnp.sin(b) * r


def compute_laplacian(a, b, r):
    return a * jnp.sin(b) * r


class Coulomb:
    def __init__(self, b, c):
        self.b = b
        self.c = c

    def compute(self, r):
        b = self.b
        c = self.c

        # return b * np.cos(c) * r
        return b * jnp.cos(c) * r


def compute_coulomb(b, c, r):
    return b * jnp.cos(c) * r


class Hamiltonian:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def compute_local_energy(self, r):
        laplacian = Laplacian(a=self.a, b=self.b)
        coulomb = Coulomb(b=self.b, c=self.c)

        e_L = laplacian.compute(r) + coulomb.compute(r)

        return e_L


def compute_hamiltonian_local_energy(a, b, c, r):
    e_L = compute_laplacian(a, b, r) + compute_coulomb(b, c, r)

    return e_L


def f_compute_local_energy(a, b, c, r):
    hamilt = Hamiltonian(a=a, b=b, c=c)
    e_L = hamilt.compute_local_energy(r)
    return e_L


def plain_method(a, b, c, r):
    return a * jnp.sin(b) * r + b * jnp.cos(c) * r


# main operations
a = 1.0
b = 2.0
c = 3.0
r = 10.0

# of course, it works, but it is not efficient at all.
e_L = f_compute_local_energy(a=a, b=b, c=c, r=r)
de_L = grad(f_compute_local_energy, argnums=(0, 1, 2))(a, b, c, r)
print(f"e_L = {e_L}")
print(f"de_L = {de_L}")

# for validation
e_L = plain_method(a=a, b=b, c=c, r=r)
de_L = grad(plain_method, argnums=(0, 1, 2))(a, b, c, r)
print(f"e_L = {e_L}")
print(f"de_L = {de_L}")

# it does not work, because hamilt.compute_local_energy is not a function of a,b,c.
hamilt = Hamiltonian(a=a, b=b, c=c)
e_L = hamilt.compute_local_energy(r)
de_L = grad(hamilt.compute_local_energy)(r)
print(f"e_L = {e_L}")
print(f"de_L = {de_L}")

# We should define a function of a,b,c.... indeed, OOP is not recommended for jax.
e_L = compute_hamiltonian_local_energy(a, b, c, r)
de_L = grad(compute_hamiltonian_local_energy, argnums=(0, 1, 2))(a, b, c, r)
print(f"e_L = {e_L}")
print(f"de_L = {de_L}")
