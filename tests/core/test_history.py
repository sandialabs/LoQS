"""Tester for loqs.core.history"""

import os
from tempfile import NamedTemporaryFile
import pytest

from loqs.core.frame import Frame
from loqs.core.history import History

class TestHistory:

    def test_init(self):
        data = {"a": 1, "b": 2}

        f = Frame(data, "test")
        h = History([f, f, f])
        for frame in h:
            assert frame._data == data
            assert frame.log == "test"
        
        h2 = History([data,]*3)
        for frame in h2:
            assert frame._data == data
            assert frame.log == "N/A"

        h3 = History.cast(h)
        for frame in h3:
            assert frame._data == data
            assert frame.log == "test"

        h4 = History.cast([data,]*3)
        for frame in h4:
            assert frame._data == data
            assert frame.log == "N/A"

        # Test failure raises error
        with pytest.raises(ValueError):
            History("abc") # type: ignore
        with pytest.raises(ValueError):
            History([1, 2, 3]) # type: ignore
    
    def test_expiring_propagating_keys(self):
        h = History([{'a': 1, "b": 2}, {'c': 3}, {'d': 4, 'a': 5}, {'b': 6}],
                    expiring_keys=["b"], propagating_keys=["a"])
        
        # Every frame after frame 2 should have an a
        # Frame 1 "b" should expire as last frame enters
        assert h[0]["a"] == 1
        with pytest.warns(UserWarning):
            assert h[0]["b"] == 2
        assert h[1]._data == {'c': 3, "a": 1}
        assert h[2]._data == {'d': 4, "a": 5}
        assert h[3]._data == {'b': 6, "a": 5}

        assert h._expiring_key_locs["b"] == 3 # Which frame keeps up-to-date b

        # A key can also be propagating and expiring, like state by default
        h2 = History([{'a': 1, "state": 2}, {'c': 3}, {'d': 4}])

        assert h2[0]["a"] == 1
        with pytest.warns(UserWarning):
            assert h2[0]["state"] == 2
        assert h[1]["c"] == 3
        with pytest.warns(UserWarning):
            assert h2[1]["state"] == 2
        assert h[2]["d"] == 4
        assert h2[2]["state"] == 2 # No warning, this one is up to date
        assert h2._expiring_key_locs["state"] == 2
    
    @pytest.mark.skipif(os.getenv("RUNNER_OS", "N/A") == "Windows", reason="Permission issues on Windows GitHub runner")
    def test_serialization(self):
        data = {"a": 1, "b": 2}

        f = Frame(data, "test 1")
        h = History([f, f.update(new_log="test 2"), f.update(new_log="test 3")])

        with NamedTemporaryFile("w+", dir='.', suffix='.json') as tempf:
            h.write(tempf.name)

            h2 = Frame.read(tempf.name)
        
        for i, frame in enumerate(h2):
            assert frame._data == data
            assert frame.log == f"test {i+1}"