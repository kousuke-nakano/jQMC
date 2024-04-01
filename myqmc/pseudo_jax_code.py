import dataclasses
import numpy as np
import jax.numpy as jnp
from jax import grad


@dataclasses.dataclass
class Data:
    a: float
    b: float
    c: float

class Laplacian:
    def __init__(self, data:Data):
        self.data=data
    
    def compute(self, r):
        a = self.data.a
        b = self.data.b
        
        #return a * np.sin(b)
        return a * jnp.sin(b)

class Coulomb:
    def __init__(self, data:Data):
        self.data=data
    
    def compute(self, r):
        b = self.data.b
        c = self.data.c
        
        #return b * np.cos(c)
        return b * jnp.cos(c)

def coulomb_compute_jax(b,c):
        
class Hamiltonian:
    def __init__(self, data:Data):
        self.data=data

    def compute_local_energy(self, r):
        laplacian=Laplacian(data=self.data)
        coulomb=Coulomb(data=self.data)
        
        e_L = laplacian.compute(r) + coulomb.compute(r)
    
        return e_L

    def compute_d_local_energy(self, r):
        return grad(self.compute_local_energy)(r)

def plain_method(a,b,c,r):
    return a * jnp.sin(b) + b * jnp.cos(c)

# main operations
a=1.0; b=2.0; c=3.0; r=10.0
data=Data(a=a, b=b, c=c)
hamilt=Hamiltonian(data=data)
e_L=hamilt.compute_local_energy(r)
de_L=hamilt.compute_d_local_energy(r)
print(f'e_L = {e_L}')
print(f'de_L = {de_L}')

e_L= plain_method(a=a,b=b,c=c,r=r)
de_L=grad(plain_method, argnums=(0,1,2))(a,b,c,r)
print(f'e_L = {e_L}')
print(f'de_L = {de_L}')
