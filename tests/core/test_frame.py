"""Tester for loqs.core.frame"""

import os
from tempfile import NamedTemporaryFile
import pytest

from loqs.core.frame import Frame

class TestFrame:

    def test_init(self):
        data = {"a": 1, "b": 2}

        f0 = Frame()
        assert f0._data == {}
        assert f0.log == "N/A"

        f1 = Frame(data, "test")
        assert f1._data == data
        assert f1.log == "test"

        f2 = Frame(f1)
        assert f2._data == data
        assert f2.log == "test"

        f3 = Frame(f1, "test 2")
        assert f3._data == data
        assert f3.log == "test 2"

        f4 = Frame.cast(f1)
        assert f4._data == data
        assert f4.log == "test"

        f5 = Frame.cast(data)
        assert f5._data == data
        assert f5.log == "N/A"

        # Test failure raises error
        with pytest.raises(ValueError):
            Frame("abc") # type: ignore
        with pytest.raises(ValueError):
            Frame([1, 2, 3]) # type: ignore
    
    def test_expire(self):
        f = Frame({"a": 1, "b": 2})
        f.expire("b")

        assert f["a"] == 1
        with pytest.warns(UserWarning):
            assert f["b"] == 2

    def test_update(self):
        f = Frame({"a": 1, "b": 2}, "test")

        f2 = f.update({'c': 3})
        assert f2._data == {"a": 1, "b": 2, "c": 3}
        assert f2.log == "test"

        f3 = f.update({'a': 3}, "test 2")
        assert f3._data == {"a": 3, "b": 2}
        assert f3.log == "test 2"

        f.expire('b')
        f4 = f.update({'c': 3})
        assert f4._data == {"a": 1, "b": 2, "c": 3}
        assert f4.log == "test"
        assert f4._expired_keys == ["b"]
    
    @pytest.mark.skipif(os.getenv("RUNNER_OS", "N/A") == "Windows", reason="Permission issues on Windows GitHub runner")
    def test_serialization(self):
        f1 = Frame({"a": 1, "b": 2})
        f1.expire("b")
        
        # Test recursive functionality
        f2 = Frame({"c": 3, "other": f1}, "test")
        f2.no_serialize("c")

        with NamedTemporaryFile("w+", dir='.', suffix='.json') as tempf:
            f2.write(tempf.name)

            f3 = Frame.read(tempf.name)
        
        assert f3["c"] == "NOT SERIALIZED"
        # Real data should come back, even for expired keys
        assert f3["other"]._data == {'a': 1, 'b': 2}
        assert f3["other"]._expired_keys == ["b"]
        assert f3.log == "test"