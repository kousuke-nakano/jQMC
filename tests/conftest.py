# conftest.py
import jax
import pytest


def pytest_addoption(parser):
    """Add options for pytests."""
    parser.addoption("--disable-jit", action="store_true", default=False, help="Disable jax.jit for pytests")
    parser.addoption("--enable-heavy", action="store_true", default=False, help="Enable heavy calculations for pytests")


@pytest.fixture(autouse=True)
def enable_jit(request):
    """Fixture to enable/disable jax.jit for pytests."""
    if request.config.getoption("--disable-jit"):
        # Enable jax.jit
        jax.config.update("jax_disable_jit", True)
    else:
        # Disable jax.jit by default
        jax.config.update("jax_disable_jit", False)
    yield
    # Reset to default after tests
    jax.config.update("jax_disable_jit", False)


def pytest_itemcollected(item):
    """Show reason for obsolete tests."""
    obsolete_marker = item.get_closest_marker("obsolete")
    if obsolete_marker:
        reason = obsolete_marker.kwargs.get("reasons", "")
        item._nodeid += f" [OBSOLETE: {reason}]"


# Custom marker for conditional skip
def pytest_configure(config):
    """Pytest configuration."""
    config.addinivalue_line("markers", "activate_if_disable_jit: activate test if --disable-jit is set")
    config.addinivalue_line("markers", "activate_if_enable_heavy: activate test if --enable-heavy is set")
    config.addinivalue_line("markers", "obsolete: tests that are obsolete and should be removed in the future")
