import time, os, jax, numpy as np, jax.numpy as jnp
from jax import grad, jit, vmap

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")  # insures we use the CPU


def func(r_cart_i, r_cart_j):
    return jnp.linalg.norm(r_cart_i - r_cart_j)


vmap_func = vmap(func, in_axes=(None, 0))
vmapvmap_func = vmap(vmap(func, in_axes=(None, 0)), in_axes=(0, None))

# test MOs
num_r_up_cart_samples = 5
num_r_dn_cart_samples = 2

r_cart_min, r_cart_max = -1.0, 1.0

r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_up_cart_samples, 3) + r_cart_min
r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_dn_cart_samples, 3) + r_cart_min

print(r_up_carts)
print(r_dn_carts)
print("-vmap-")
print(vmap_func(r_up_carts, r_dn_carts))
print(np.linalg.norm(r_up_carts - r_dn_carts[0]))
print(np.linalg.norm(r_up_carts - r_dn_carts[1]))
print("-vmapvmap-")
print(vmapvmap_func(r_up_carts, r_dn_carts))
print(np.linalg.norm(r_up_carts[0] - r_dn_carts[0]))

# print(jnp.linalg.norm(r_up_carts - r_dn_carts[0]))
# print(func(r_up_carts, r_dn_carts[0]))
