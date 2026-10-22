"""Tester for loqs.core.frame"""

import pytest

from loqs.core.frame import Frame


class TestFrame:

    def test_init(self):
        data = {"a": 1, "b": 2}

        f0 = Frame()
        assert f0._data == {}

        f1 = Frame(data, "test")
        assert f1._data == data
        assert f1.log == "test"

        f2 = Frame(f1)
        assert f2._data == data
        assert f2.log == "test"

        f3 = Frame(f1, "test 2")
        assert f3._data == data
        assert f3.log == "test 2"

        f5 = Frame(data)
        assert f5._data == data

        # Test failure raises error
        with pytest.raises(ValueError):
            Frame("abc")  # type: ignore

    def test_init_from_existing_frame_preserves_expiry_and_no_serialize(self):
        f1 = Frame({"a": 1, "b": 2, "c": 3})
        f1.expire("a")
        f1.no_serialize("b")

        f2 = Frame(f1)
        assert f2._expired_keys == ["a"]
        assert f2._no_serialize_keys == ["b"]

    def test_expire(self):
        f = Frame({"a": 1, "b": 2})
        f.expire("b")

        assert f["a"] == 1
        with pytest.warns(UserWarning):
            assert f["b"] == 2

    def test_update(self):
        f = Frame({"a": 1, "b": 2}, "test")

        f2 = f.update({"c": 3})
        assert f2._data == {"a": 1, "b": 2, "c": 3}
        assert f2.log == "test"

        f3 = f.update({"a": 3}, "test 2")
        assert f3._data == {"a": 3, "b": 2}
        assert f3.log == "test 2"

        f.expire("b")
        f4 = f.update({"c": 3})
        assert f4._data == {"a": 1, "b": 2, "c": 3}
        assert f4.log == "test"
        assert f4._expired_keys == ["b"]

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_serialization(self, format, make_temp_path):
        f1 = Frame({"a": 1, "b": 2})
        f1.expire("b")

        f2 = Frame(
            {
                "state": "initial",
                "qubits": ["Q0", "Q1"],
                "count": 42,
                "c": 3,
                "other": f1,
            },
            log="test",
        )
        f2.no_serialize("c")

        with make_temp_path(suffix=f".{format}") as tmp_path:
            f2.write(tmp_path)
            f3 = Frame.read(tmp_path)

        assert isinstance(f3, Frame)
        assert f3.log == "test"
        assert f3["state"] == "initial"
        assert f3["qubits"] == ["Q0", "Q1"]
        assert f3["count"] == 42
        assert f3["c"] == "NOT SERIALIZED"
        # Real data should come back, even for expired keys
        other = f3["other"]
        assert isinstance(other, Frame)
        assert other._data == {"a": 1, "b": 2}
        assert other._expired_keys == ["b"]
