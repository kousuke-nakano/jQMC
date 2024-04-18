# import dataclasses
import numpy as np
import numpy.typing as npt

# flax, jax
import jax.numpy as jnp
from jax import lax
from flax import struct
from jax import grad, jit


############################################
# AOs data and AOs method -> AO module
############################################


# @dataclasses.dataclass(frozen=True)
# @chex.dataclass
@struct.dataclass
class AOs_data:
    t: str = struct.field(pytree_node=False)
    d: float


@jit
def aos_compute(aos_data: AOs_data, r):
    t = aos_data.t
    d = aos_data.d
    print(t)
    return d * jnp.tan(d) * r


###############################################################
# Laplacian data and Laplacian method -> Laplacian module
###############################################################


# @dataclasses.dataclass(frozen=True)
# @chex.dataclass
@struct.dataclass
class Laplacian_data:
    a: float
    b: float
    aos_data: AOs_data


@jit
def laplacian_compute(laplacian_data: Laplacian_data, r):
    a = laplacian_data.a
    b = laplacian_data.b
    aos_data = laplacian_data.aos_data
    return (
        a * jnp.sin(b) * r
        - aos_compute(aos_data, r)
        + jnp.sum(np.array([10.0 for _ in range(10)]))
    )


###############################################################
# Coulomb data and Coulomb method -> Coulomb module
###############################################################


# @dataclasses.dataclass(frozen=True)
# @chex.dataclass
@struct.dataclass
class Coulomb_data:
    b: float
    c: float
    A: npt.NDArray[np.float64]
    B: npt.NDArray[np.float64]


@jit
def coulomb_compute(coulomb_data: Coulomb_data, A_old, r_old, flag, r):
    b = coulomb_data.b
    c = coulomb_data.c
    A = coulomb_data.A
    B = coulomb_data.B

    A_dup = np.zeros(A.shape)

    """ incompatible with jax-jit
    if flag:
        return 2.0 * A
    else:
        return 3.0 * A
    """

    def true_flag_fun(A):
        return 2.0 * A

    def false_flag_fun(A):
        return 3.0 * A

    A_dup = lax.cond(flag, true_flag_fun, false_flag_fun, A)

    """ incompatible with jax-jit
    if b < 2.0:
        return b * jnp.cos(c) * r + jnp.exp(-b * c**2) + jnp.trace(jnp.dot(A, B))
    else:
        return b * jnp.cos(c) * r + jnp.exp(-b * c**2) + jnp.trace(jnp.dot(A, B))
    """

    def true_fun(b, c, A, B):
        return (
            b * jnp.cos(c) * r
            + jnp.exp(-b * c**2)
            + jnp.trace(jnp.dot(A, B))
            + jnp.trace(jnp.dot(r_old * A_old, B))
        )

    def false_fun(b, c, A, B):
        return (
            b * jnp.cos(c) * r
            + jnp.exp(-b * c**2)
            + jnp.trace(jnp.dot(A, B))
            + jnp.trace(jnp.dot(r_old * A_old, B))
        )

    return lax.cond(b < 2.0, true_fun, false_fun, b, c, A_dup, B)


#################################################################
# Most upper class
# Hamiltonian data and Hamiltonian method -> Hamiltonian module
#################################################################


# @dataclasses.dataclass(frozen=True)
# @chex.dataclass(frozen=True)
@struct.dataclass
class Hamiltonian_data:
    laplacian_data: Laplacian_data
    coulomb_data: Coulomb_data


@jit
def compute_local_energy(hamiltonian_data: Hamiltonian_data, A_old, r_old, flag, r):

    laplacian_data = hamiltonian_data.laplacian_data
    coulomb_data = hamiltonian_data.coulomb_data
    e_L = (
        laplacian_compute(laplacian_data, r)
        * coulomb_compute(coulomb_data, A_old, r_old, flag, r) ** 2
    )

    return e_L


def validation(a, b, c, d, A, B, A_old, r_old, flag, r):
    laplacian = (
        a * jnp.sin(b) * r
        - d * jnp.tan(d) * r
        + jnp.sum(np.array([10.0 for _ in range(10)]))
    )

    A_dup = np.zeros(A.shape)
    if flag:
        A_dup = 2.0 * A
    else:
        A_dup = 3.0 * A
    if b < 2.0:
        coulomb = (
            b * jnp.cos(c) * r
            + jnp.exp(-b * c**2)
            + jnp.trace(jnp.dot(A_dup, B))
            + jnp.trace(jnp.dot(r_old * A_old, B))
        )
    else:
        coulomb = (
            b * jnp.cos(c) * r
            + jnp.exp(-b * c**2)
            + jnp.trace(jnp.dot(A_dup, B))
            + jnp.trace(jnp.dot(r_old * A_old, B))
        )
    return laplacian * coulomb**2


# main operations
a = 1.0
b = 2.0
c = 3.0
d = 4.0
r = 10.0

dim = 2
g = np.random.uniform(-1, 1, (dim, dim))
A = g.dot(g.T)
g = np.random.uniform(-1, 1, (dim, dim))
B = g.dot(g.T)
A_old = A
r_old = 3
flag = True

# jax-based e_L and its grad computations
coulomb_data = Coulomb_data(b=b, c=c, A=A, B=B)
aos_data = AOs_data(d=d, t="test")
laplacian_data = Laplacian_data(a=a, b=b, aos_data=aos_data)
hamiltonian_data = Hamiltonian_data(
    laplacian_data=laplacian_data, coulomb_data=coulomb_data
)
e_L = compute_local_energy(hamiltonian_data, A_old, r_old, flag, r)
de_L_param, de_L_r = grad(compute_local_energy, argnums=(0, 4))(
    hamiltonian_data, A_old, r_old, flag, r
)
print(f"e_L = {e_L}")
print(f"de_L_param = {de_L_param}")
print(f"de_L_param(b) = {de_L_param.laplacian_data.b +de_L_param.coulomb_data.b}")
print(f"de_L_r = {de_L_r}")

# for validation
e_L = validation(a=a, b=b, c=c, d=d, A=A, B=B, A_old=A_old, r_old=r_old, flag=flag, r=r)
de_L = grad(validation, argnums=(0, 1, 2, 3, 4, 5, 9))(
    a, b, c, d, A, B, A_old, r_old, flag, r
)
print(f"e_L = {e_L}")
print(f"de_L = {de_L}")
