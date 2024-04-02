import dataclasses
import numpy as np
import numpy.typing as npt

# flax, jax
import jax.numpy as jnp
from flax import struct
from jax import grad, jit


############################################
# AOs data and AOs method -> AO module
############################################

#@dataclasses.dataclass(frozen=True)
#@chex.dataclass
@struct.dataclass
class AOs_data:
    t: str = struct.field(pytree_node=False)
    d: float

@jit
def aos_compute(aos_data:AOs_data, r):
    t = aos_data.t
    d = aos_data.d
    print(t)
    return d * jnp.tan(d) * r

###############################################################
# Laplacian data and Laplacian method -> Laplacian module
###############################################################

#@dataclasses.dataclass(frozen=True)
#@chex.dataclass
@struct.dataclass
class Laplacian_data:
    a: float
    b: float
    aos_data: AOs_data

@jit
def laplacian_compute(laplacian_data:Laplacian_data, r):
    a = laplacian_data.a
    b = laplacian_data.b
    aos_data = laplacian_data.aos_data
    return a * jnp.sin(b) * r  - aos_compute(aos_data, r)

###############################################################
# Coulomb data and Coulomb method -> Coulomb module
###############################################################

#@dataclasses.dataclass(frozen=True)
#@chex.dataclass
@struct.dataclass
class Coulomb_data:
    b: float
    c: float
    A: npt.NDArray[np.float64]
    B: npt.NDArray[np.float64]

@jit
def coulomb_compute(coulomb_data:Coulomb_data, r):
    b = coulomb_data.b
    c = coulomb_data.c
    A = coulomb_data.A
    B = coulomb_data.B
    return b * jnp.cos(c) * r + jnp.exp(-b*c**2) + jnp.trace(jnp.dot(A,B))

#################################################################
# Most upper class
# Hamiltonian data and Hamiltonian method -> Hamiltonian module
#################################################################

#@dataclasses.dataclass(frozen=True)
#@chex.dataclass(frozen=True)
@struct.dataclass
class Hamiltonian_data:
    laplacian_data: Laplacian_data
    coulomb_data: Coulomb_data

@jit
def compute_local_energy(hamiltonian_data:Hamiltonian_data, r):
    
    laplacian_data = hamiltonian_data.laplacian_data
    coulomb_data = hamiltonian_data.coulomb_data
    e_L = laplacian_compute(laplacian_data, r) * coulomb_compute(coulomb_data, r)**2

    return e_L

def validation(a,b,c,d,A,B,r):
    laplacian = (a * jnp.sin(b) * r - d * jnp.tan(d) * r)
    coulomb = (b * jnp.cos(c) * r + jnp.exp(-b*c**2) + jnp.trace(jnp.dot(A,B)))
    return laplacian * coulomb**2

# main operations
a=1.0; b=2.0; c=3.0; d=4.0; r=10.0

dim=2
g = np.random.uniform(-1, 1, (dim, dim))
A=g.dot(g.T)
g = np.random.uniform(-1, 1, (dim, dim))
B=g.dot(g.T)

# jax-based e_L and its grad computations
coulomb_data=Coulomb_data(b=b,c=c,A=A,B=B)
aos_data=AOs_data(d=d, t='test')
laplacian_data=Laplacian_data(a=a,b=b,aos_data=aos_data)
hamiltonian_data=Hamiltonian_data(laplacian_data=laplacian_data,coulomb_data=coulomb_data)
e_L=compute_local_energy(hamiltonian_data, r)
de_L_param, de_L_r=grad(compute_local_energy, argnums=(0,1))(hamiltonian_data, r)
print(f'e_L = {e_L}')
print(f"de_L_param = {de_L_param}")
print(f"de_L_param(b) = {de_L_param.laplacian_data.b +de_L_param.coulomb_data.b}")
print(f"de_L_r = {de_L_r}")

# for validation
e_L= validation(a=a,b=b,c=c,d=d,A=A,B=B,r=r)
de_L=grad(validation, argnums=(0,1,2,3,4,5,6))(a,b,c,d,A,B,r)
print(f'e_L = {e_L}')
print(f'de_L = {de_L}')
