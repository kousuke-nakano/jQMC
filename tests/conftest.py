# conftest.py
import jax
import pytest


def pytest_addoption(parser):
    """Add options for pytests."""
    parser.addoption("--disable-jit", action="store_true", default=False, help="Disable jax.jit for pytests")
    parser.addoption("--skip-heavy", action="store_true", default=False, help="Skip heavy calculations for pytests")


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
    config.addinivalue_line("markers", "activate_if_skip_heavy: skip test if --skip-heavy is set")
    config.addinivalue_line("markers", "obsolete: tests that are obsolete and should be removed in the future")


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with activate_if_skip_heavy when --skip-heavy is set."""
    if not config.getoption("--skip-heavy"):
        return
    skip_marker = pytest.mark.skip(reason="skipped by --skip-heavy")
    for item in items:
        if item.get_closest_marker("activate_if_skip_heavy"):
            item.add_marker(skip_marker)
