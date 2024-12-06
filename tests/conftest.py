# conftest.py
import jax
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--disable-jit", action="store_true", default=False, help="Disable jax.jit for pytests"
    )


@pytest.fixture(autouse=True)
def enable_jit(request):
    if request.config.getoption("--disable-jit"):
        # Enable jax.jit
        jax.config.update("jax_disable_jit", True)
    else:
        # Disable jax.jit by default
        jax.config.update("jax_disable_jit", False)
    yield
    # Reset to default after tests
    jax.config.update("jax_disable_jit", False)


# Custom marker for conditional skip
def pytest_configure(config):
    config.addinivalue_line("markers", "activate_if_disable_jit: activate test if --disable-jit is set")
