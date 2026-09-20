"""Offline tests run normally; lock-dependent checks require the owner's lock."""
import os
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption("--upstream", action="store_true",
                     help="expect the runtime overlay revision to differ from the pin")


@pytest.fixture(scope="session")
def lock():
    path = ROOT / "uv.lock"
    if not path.is_file():
        if os.environ.get("CI"):
            pytest.fail("uv.lock is required in CI; generate and commit it with uv")
        pytest.skip("owner must generate uv.lock outside the offline sandbox")
    return tomllib.loads(path.read_text())


@pytest.fixture
def isolated_wrapper(monkeypatch, tmp_path):
    from geomancer import app

    for key in list(os.environ):
        if key.startswith("MANYRUNS_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    app._cleanup()
    yield app
    # New directory/caller variables were written by the wrapper, not monkeypatch.
    for key in list(os.environ):
        if key.startswith("MANYRUNS_"):
            monkeypatch.delenv(key)
    app._cleanup()
