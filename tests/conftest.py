import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_torch: skip test if torch is not installed"
    )


def pytest_collection_modifyitems(config, items):
    try:
        import torch  # noqa: F401

        torch_available = True
    except ImportError:
        torch_available = False

    skip_torch = pytest.mark.skip(reason="torch not installed")
    for item in items:
        if "requires_torch" in item.keywords and not torch_available:
            item.add_marker(skip_torch)
