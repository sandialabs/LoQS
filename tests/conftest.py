import pytest
import os
import tempfile
from contextlib import contextmanager

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