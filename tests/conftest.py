import pytest
import os
import tempfile
from contextlib import contextmanager

try:
    # Force a non-interactive backend before any test imports pyplot, so
    # building a figure never depends on a real GUI toolkit (e.g. Tk) being
    # present -- these tests only ever inspect a figure's contents, never
    # display one, and a GUI backend is a real, environment-dependent
    # source of flakiness (e.g. a broken/missing Tcl/Tk install) that a
    # headless test run has no use for regardless.
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass

@contextmanager
def temp_path(*, suffix=""):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd) # Windows runner compatibility
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

@pytest.fixture
def make_temp_path():
    """
    Usage:
    def test_for_something(make_temp_path):
        ...

        with make_temp_path(suffix=".json") as p:
            <process with temppath p>
        
        ...
    """
    return temp_path