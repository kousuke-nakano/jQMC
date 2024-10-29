import jax

with jax.profiler.trace("/tmp/tensorboard", create_perfetto_link=True):
    key = jax.random.key(0)
    x = jax.random.normal(key, (5000, 5000))
    y = x @ x
    y.block_until_ready()
