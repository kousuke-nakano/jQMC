# conftest.py
import jax
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--enable-jit", action="store_true", default=False, help="Enable jax.jit for tests"
    )


@pytest.fixture(autouse=True)
def enable_jit(request):
    if request.config.getoption("--enable-jit"):
        # Enable jax.jit
        jax.config.update("jax_disable_jit", False)
    else:
        # Disable jax.jit by default
        jax.config.update("jax_disable_jit", True)
    yield
    # Reset to default after tests
    jax.config.update("jax_disable_jit", False)


# Custom marker for conditional skip
def pytest_configure(config):
    config.addinivalue_line("markers", "skip_if_enable_jit: skip test if --enable-jit is set")
