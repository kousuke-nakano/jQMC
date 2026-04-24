# conftest.py
import jax
import pytest


def pytest_addoption(parser):
    """Add options for pytests."""
    parser.addoption("--disable-jit", action="store_true", default=False, help="Disable jax.jit for pytests")
    parser.addoption("--skip-heavy", action="store_true", default=False, help="Skip heavy calculations for pytests")
    parser.addoption(
        "--precision-mode",
        default="full",
        choices=["full", "mixed"],
        help="Precision mode for tests (default: full)",
    )


@pytest.fixture(autouse=True)
def enable_jit(request):
    """Fixture to enable/disable jax.jit for pytests."""
    if request.config.getoption("--disable-jit"):
        # Disable jax.jit (jax_disable_jit=True means JIT is disabled)
        jax.config.update("jax_disable_jit", True)
    else:
        # Enable jax.jit (default)
        jax.config.update("jax_disable_jit", False)
    yield
    # Reset to default after tests
    jax.config.update("jax_disable_jit", False)


@pytest.fixture(autouse=True)
def configure_precision(request):
    """Configure precision mode before each test."""
    from jqmc._precision import configure

    mode = request.config.getoption("--precision-mode")
    configure({"mode": mode})


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
    config.addinivalue_line(
        "markers",
        "numerical_diff: test compares analytic or autodiff results "
        "against finite-difference derivatives or numerical quadrature. "
        "Skipped automatically when --precision-mode=mixed because "
        "float32 round-off dominates the FD / quadrature error.",
    )
    config.addinivalue_line(
        "markers",
        "external_reference: test compares against an external reference "
        "(e.g. TurboRVB). Validated only in --precision-mode=full; "
        "skipped in mixed mode.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on CLI options (--skip-heavy, --precision-mode)."""
    if config.getoption("--skip-heavy"):
        skip_marker = pytest.mark.skip(reason="skipped by --skip-heavy")
        for item in items:
            if item.get_closest_marker("activate_if_skip_heavy"):
                item.add_marker(skip_marker)

    if config.getoption("--precision-mode") == "mixed":
        skip_fd = pytest.mark.skip(
            reason="FD / numerical-quadrature comparison is invalid under mixed precision (float32 round-off dominates)."
        )
        skip_extref = pytest.mark.skip(reason="External-reference comparison validated only in mode=full.")
        for item in items:
            if "numerical_diff" in item.keywords:
                item.add_marker(skip_fd)
            if "external_reference" in item.keywords:
                item.add_marker(skip_extref)
