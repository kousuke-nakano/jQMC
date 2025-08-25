# import dataclasses
# flax, jax
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import grad, jit, lax

# JAX float64
jax.config.update("jax_enable_x64", True)

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

    @property
    def b_p(self):
        return self.b


@jit
def laplacian_compute(laplacian_data: Laplacian_data, r):
    a = laplacian_data.a
    b = laplacian_data.b
    aos_data = laplacian_data.aos_data
    return a * jnp.sin(b) * r - aos_compute(aos_data, r) + jnp.sum(np.array([10.0 for _ in range(10)]))


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

    @property
    def b_p(self):
        return self.b


@jit
def coulomb_compute(coulomb_data: Coulomb_data, A_old, r_old, flag, r):
    b = coulomb_data.b_p
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

    # this lax statement is compatible with git
    def true_fun(b, c, A, B):
        return b * jnp.cos(c) * r + jnp.exp(-b * c**2) + jnp.trace(jnp.dot(A, B)) + jnp.trace(jnp.dot(r_old * A_old, B))

    def false_fun(b, c, A, B):
        return b * jnp.cos(c) * r + jnp.exp(-b * c**2) + jnp.trace(jnp.dot(A, B)) + jnp.trace(jnp.dot(r_old * A_old, B))

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
    e_L = laplacian_compute(laplacian_data, r) * coulomb_compute(coulomb_data, A_old, r_old, flag, r) ** 2

    return e_L


# the same calculation as above, while it is realized with a single function
def validation_jnp(a, b, c, d, A, B, A_old, r_old, flag, r):
    laplacian = a * jnp.sin(b) * r - d * jnp.tan(d) * r + jnp.sum(jnp.array([10.0 for _ in range(10)]))

    A_dup = jnp.zeros(A.shape)
    if flag:
        A_dup = 2.0 * A
    else:
        A_dup = 3.0 * A
    if b < 2.0:
        coulomb = b * jnp.cos(c) * r + jnp.exp(-b * c**2) + jnp.trace(jnp.dot(A_dup, B)) + jnp.trace(jnp.dot(r_old * A_old, B))
    else:
        coulomb = b * jnp.cos(c) * r + jnp.exp(-b * c**2) + jnp.trace(jnp.dot(A_dup, B)) + jnp.trace(jnp.dot(r_old * A_old, B))
    return laplacian * coulomb**2


def validation_np(a, b, c, d, A, B, A_old, r_old, flag, r):
    laplacian = a * np.sin(b) * r - d * np.tan(d) * r + np.sum(np.array([10.0 for _ in range(10)]))

    A_dup = np.zeros(A.shape)
    if flag:
        A_dup = 2.0 * A
    else:
        A_dup = 3.0 * A
    if b < 2.0:
        coulomb = b * np.cos(c) * r + np.exp(-b * c**2) + np.trace(np.dot(A_dup, B)) + np.trace(np.dot(r_old * A_old, B))
    else:
        coulomb = b * np.cos(c) * r + np.exp(-b * c**2) + np.trace(np.dot(A_dup, B)) + np.trace(np.dot(r_old * A_old, B))
    return laplacian * coulomb**2


# main operations
a = 1.0
b = 5.0
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

print("Computing e_L with JAX-JIT and it derivatives using JAX-auto-grad with dataclass inheretance")
# jax-based e_L and its auto-grad computations
aos_data = AOs_data(d=d, t="test")
coulomb_data = Coulomb_data(b=b, c=c, A=A, B=B)
laplacian_data = Laplacian_data(a=a, b=b, aos_data=aos_data)
hamiltonian_data = Hamiltonian_data(laplacian_data=laplacian_data, coulomb_data=coulomb_data)
e_L = compute_local_energy(hamiltonian_data, A_old, r_old, flag, r)
de_L_param, de_L_r = grad(compute_local_energy, argnums=(0, 4))(hamiltonian_data, A_old, r_old, flag, r)
print(f"  e_L = {e_L}")
print(f"  de_L_db = {de_L_param.laplacian_data.b + de_L_param.coulomb_data.b}")
print(f"  de_L_dr = {de_L_r}")

print(
    "Computing e_L with JAX-JIT and it derivatives without JAX auto-grad (i.e., numerical derivatives) with dataclass inheretance"
)
diff_b = 5.0e-8
aos_data = AOs_data(d=d, t="test")
coulomb_data = Coulomb_data(b=b + diff_b, c=c, A=A, B=B)
laplacian_data = Laplacian_data(a=a, b=b + diff_b, aos_data=aos_data)
hamiltonian_data = Hamiltonian_data(laplacian_data=laplacian_data, coulomb_data=coulomb_data)
de_L_p_b = compute_local_energy(hamiltonian_data, A_old, r_old, flag, r)

aos_data = AOs_data(d=d, t="test")
coulomb_data = Coulomb_data(b=b - diff_b, c=c, A=A, B=B)
laplacian_data = Laplacian_data(a=a, b=b - diff_b, aos_data=aos_data)
hamiltonian_data = Hamiltonian_data(laplacian_data=laplacian_data, coulomb_data=coulomb_data)
de_L_m_b = compute_local_energy(hamiltonian_data, A_old, r_old, flag, r)

de_L_db = (de_L_p_b - de_L_m_b) / (2 * diff_b)

diff_r = 5.0e-8
aos_data = AOs_data(d=d, t="test")
coulomb_data = Coulomb_data(b=b, c=c, A=A, B=B)
laplacian_data = Laplacian_data(a=a, b=b, aos_data=aos_data)
hamiltonian_data = Hamiltonian_data(laplacian_data=laplacian_data, coulomb_data=coulomb_data)
de_L_p_r = compute_local_energy(hamiltonian_data, A_old, r_old, flag, r + diff_r)
de_L_m_r = compute_local_energy(hamiltonian_data, A_old, r_old, flag, r - diff_r)
de_L_dr = (de_L_p_r - de_L_m_r) / (2 * diff_r)

print(f"  e_L = {e_L}")
print(f"  de_L_db = {de_L_db}")
print(f"  de_L_dr = {de_L_dr}")

print("Computing e_L without JAX-numpy and its derivatives using JAX-auto-grad with a single function")
e_L = validation_jnp(a=a, b=b, c=c, d=d, A=A, B=B, A_old=A_old, r_old=r_old, flag=flag, r=r)
de_L = grad(validation_jnp, argnums=(0, 1, 2, 3, 4, 5, 9))(a, b, c, d, A, B, A_old, r_old, flag, r)
print(f"  e_L = {e_L}")
print(f"  de_L_db = {de_L[1]}")
print(f"  de_L_dr = {de_L[-1]}")

print(
    "Computing e_L with JAX-numpy and its derivatives without JAX auto-grad (i.e., numerical derivatives) with a single function"
)
e_L = validation_jnp(a=a, b=b, c=c, d=d, A=A, B=B, A_old=A_old, r_old=r_old, flag=flag, r=r)
diff_b = 5.0e-8
de_L_p_b = validation_jnp(a=a, b=b + diff_b, c=c, d=d, A=A, B=B, A_old=A_old, r_old=r_old, flag=flag, r=r)
de_L_m_b = validation_jnp(a=a, b=b - diff_b, c=c, d=d, A=A, B=B, A_old=A_old, r_old=r_old, flag=flag, r=r)

de_L_db = (de_L_p_b - de_L_m_b) / (2 * diff_b)

diff_r = 5.0e-8
de_L_p_r = validation_jnp(a=a, b=b, c=c, d=d, A=A, B=B, A_old=A_old, r_old=r_old, flag=flag, r=r + diff_r)
de_L_m_r = validation_jnp(a=a, b=b, c=c, d=d, A=A, B=B, A_old=A_old, r_old=r_old, flag=flag, r=r - diff_r)
de_L_dr = (de_L_p_r - de_L_m_r) / (2 * diff_r)

print(f"  e_L = {e_L}")
print(f"  de_L_db = {de_L_db}")
print(f"  de_L_dr = {de_L_dr}")

print(
    "Computing e_L with native numpy and its derivatives without JAX auto-grad (i.e., numerical derivatives) with a single function"
)
e_L = validation_np(a=a, b=b, c=c, d=d, A=A, B=B, A_old=A_old, r_old=r_old, flag=flag, r=r)
diff_b = 5.0e-8
de_L_p_b = validation_np(a=a, b=b + diff_b, c=c, d=d, A=A, B=B, A_old=A_old, r_old=r_old, flag=flag, r=r)
de_L_m_b = validation_np(a=a, b=b - diff_b, c=c, d=d, A=A, B=B, A_old=A_old, r_old=r_old, flag=flag, r=r)

de_L_db = (de_L_p_b - de_L_m_b) / (2 * diff_b)

diff_r = 5.0e-4
de_L_p_r = validation_np(a=a, b=b, c=c, d=d, A=A, B=B, A_old=A_old, r_old=r_old, flag=flag, r=r + diff_r)
de_L_m_r = validation_np(a=a, b=b, c=c, d=d, A=A, B=B, A_old=A_old, r_old=r_old, flag=flag, r=r - diff_r)
de_L_dr = (de_L_p_r - de_L_m_r) / (2 * diff_r)

print(f"  e_L = {e_L}")
print(f"  de_L_db = {de_L_db}")
print(f"  de_L_dr = {de_L_dr}")
