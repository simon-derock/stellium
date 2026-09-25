# Smoke test to verify package import and environment integrity
from src import __version__


# Test package version string
def test_package_version() -> None:
    assert __version__ == "0.1.0"


# Test mathematical and environment sanity
def test_environment_sanity() -> None:
    assert 1 + 1 == 2
