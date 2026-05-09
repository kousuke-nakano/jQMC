# conftest.py
import jax
import pytest


@pytest.fixture(scope="session", autouse=True)
def _jax_distributed_init():
    """Initialize ``jax.distributed`` once per pytest session under multi-rank MPI.

    The device branch of ``run_optimize`` (``use_device_collectives=True``)
    relies on ``jax.lax.psum`` / ``jax.lax.all_gather`` to aggregate across
    MPI ranks. Without ``jax.distributed.initialize`` each rank's JAX sees
    only its own local devices (``jax.process_count() == 1``) and the
    collectives degenerate to no-ops, silently producing wrong results.

    Mirrors the production CLI setup in ``jqmc_cli.py``: strip HTTP proxy
    environment variables before calling ``initialize`` so the JAX gRPC
    coordination service doesn't try to route the local-host connection
    through a proxy (which is the typical cause of "hangs forever" on
    macOS / corporate networks).
    """
    import os

    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:
        # Same proxy-strip workaround as jqmc_cli.py.
        for _proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(_proxy_var, None)
        try:
            jax.distributed.initialize(cluster_detection_method="mpi4py")
        except Exception:
            # already initialized in the same process, or backend cannot start
            pass
    yield


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
    configure(mode)


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
