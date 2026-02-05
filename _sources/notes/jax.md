# JAX compatible modules

(jax_compatible_modules_tags)=

## Automatic Differentiations

All functions implemented in `jqmc` are written to be JAX compatible: they operate on `jax.numpy` arrays, avoid hidden side effects, and therefore work with `jax.grad`, `jax.hessian`, `jax.jit`, `jax.vmap`, and `jax.pmap`. This means you can take first- and second-order derivatives of any observable or loss you build from `jqmc` modules with respect to parameters (e.g., Jastrow coefficients), atomic coordinates, or other continuous inputs.

Functions that accept dataclass inputs (for example, `compute_local_energy(Hamiltonian_data, ...)`) are also differentiable. `jQMC` dataclasses use `flax.struct.dataclass`, and differentiable fields are registered as `pytree_node`, so JAX sees them as PyTrees and can propagate derivatives through them.

With the JAX automatic differentiation, `jQMC` can compute derivative quantities used in wavefunction optimization and force evaluation, such as log-derivatives of the wavefunction and derivatives of the local energy with respect to atomic positions (forces), by applying `jax.grad` to the corresponding scalar-valued functions in `jQMC`.
