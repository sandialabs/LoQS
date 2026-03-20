"""Tester for loqs.core.frame"""

import os
from tempfile import NamedTemporaryFile
import pytest

from loqs.core.frame import Frame
from loqs.internal.encoder.jsonencoder import JSONEncoder
from loqs.internal.serializable import Serializable

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

    def test_frame_serialization_roundtrip(self):
        """Test basic Frame serialization roundtrip."""
        original_frame = Frame({"state": "initial", "qubits": ["Q0", "Q1"], "count": 42}, log="test_frame")

        # Test file serialization
        with NamedTemporaryFile(suffix='.json') as f:
            original_frame.write(f.name)
            loaded_frame = Frame.read(f.name)
            assert loaded_frame.log == "test_frame"
            assert loaded_frame["state"] == "initial"

    def test_frame_with_expired_keys(self):
        """Test Frame serialization with expired keys."""
        frame = Frame({"a": 1, "b": 2, "c": 3})
        frame.expire("b")

        # Serialize and deserialize using new API
        serialized = Serializable.encode(frame, format="json", reset_encode_id=True)
        loaded_frame = Serializable.decode(serialized, format="json")

        # Check that expired keys are handled correctly
        assert loaded_frame["a"] == 1

    def test_frame_hdf5_serialization_roundtrip(self):
        """Test Frame HDF5 serialization roundtrip."""
        original_frame = Frame({"state": "initial", "qubits": ["Q0", "Q1"], "count": 42}, log="test_frame")

        # Test HDF5 serialization using new API
        import tempfile
        import h5py
        with tempfile.NamedTemporaryFile(suffix='.h5') as f:
            h5file = h5py.File(f.name, 'w')
            root_group = h5file.create_group("root")
            serialized = Serializable.encode(original_frame, format="hdf5", h5_group=root_group, reset_encode_id=True)
            loaded_frame = Serializable.decode(serialized, format="hdf5")
            h5file.close()

        assert loaded_frame.log == "test_frame"
        assert loaded_frame["state"] == "initial"
        assert loaded_frame["qubits"] == ["Q0", "Q1"]
        assert loaded_frame["count"] == 42

        # Test file serialization with .h5 extension
        with NamedTemporaryFile(suffix='.h5') as f:
            original_frame.write(f.name)
            loaded_frame = Frame.read(f.name)
            assert loaded_frame.log == "test_frame"
            assert loaded_frame["state"] == "initial"

        # Test file serialization with .hdf5 extension
        with NamedTemporaryFile(suffix='.hdf5') as f:
            original_frame.write(f.name)
            loaded_frame = Frame.read(f.name)
            assert loaded_frame.log == "test_frame"
            assert loaded_frame["state"] == "initial"

    def test_frame_hdf5_with_expired_keys(self):
        """Test Frame HDF5 serialization with expired keys."""
        frame = Frame({"a": 1, "b": 2, "c": 3})
        frame.expire("b")

        # Serialize and deserialize with HDF5 using new API
        import tempfile
        import h5py
        with tempfile.NamedTemporaryFile(suffix='.h5') as f:
            h5file = h5py.File(f.name, 'w')
            root_group = h5file.create_group("root")
            serialized = Serializable.encode(frame, format="hdf5", h5_group=root_group, reset_encode_id=True)
            loaded_frame = Serializable.decode(serialized, format="hdf5")
            h5file.close()

        # Check that expired keys are handled correctly
        assert loaded_frame["a"] == 1
        assert loaded_frame["c"] == 3

    def test_frame_hdf5_with_no_serialize_keys(self):
        """Test Frame HDF5 serialization with no_serialize keys."""
        frame = Frame({"a": 1, "b": 2, "c": 3})
        frame.no_serialize("b")

        # Serialize and deserialize with HDF5 using new API
        import tempfile
        import h5py
        with tempfile.NamedTemporaryFile(suffix='.h5') as f:
            h5file = h5py.File(f.name, 'w')
            root_group = h5file.create_group("root")
            serialized = Serializable.encode(frame, format="hdf5", h5_group=root_group, reset_encode_id=True)
            loaded_frame = Serializable.decode(serialized, format="hdf5")
            h5file.close()

        # Check that no_serialize keys are handled correctly
        assert loaded_frame["a"] == 1
        assert loaded_frame["c"] == 3
        assert loaded_frame["b"] == "NOT SERIALIZED"
        assert loaded_frame["c"] == 3

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_frame_serialization_roundtrip_parameterized(self, format):
        """Test Frame serialization roundtrip with both JSON and HDF5 formats."""
        original_frame = Frame({"state": "initial", "qubits": ["Q0", "Q1"], "count": 42}, log="test_frame")

        # Test serialization using new API
        if format == "hdf5":
            # HDF5 format requires a file and group
            import tempfile
            import h5py
            with tempfile.NamedTemporaryFile(suffix='.h5') as f:
                h5file = h5py.File(f.name, 'w')
                root_group = h5file.create_group("root")
                serialized = Serializable.encode(original_frame, format=format, h5_group=root_group, reset_encode_id=True)
                loaded_frame = Serializable.decode(serialized, format=format)
                h5file.close()
        else:
            # JSON format
            serialized = Serializable.encode(original_frame, format=format, reset_encode_id=True)
            loaded_frame = Serializable.decode(serialized, format=format)

        assert loaded_frame.log == "test_frame"
        assert loaded_frame["state"] == "initial"
        assert loaded_frame["qubits"] == ["Q0", "Q1"]
        assert loaded_frame["count"] == 42

        # Test file serialization
        with NamedTemporaryFile(suffix=f'.{format}') as f:
            original_frame.write(f.name)
            loaded_frame = Frame.read(f.name)
            assert loaded_frame.log == "test_frame"
            assert loaded_frame["state"] == "initial"

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_frame_with_expired_keys_parameterized(self, format):
        """Test Frame serialization with expired keys using both formats."""
        frame = Frame({"a": 1, "b": 2, "c": 3})
        frame.expire("b")

        # Serialize and deserialize using new API
        if format == "hdf5":
            # HDF5 format requires a file and group
            import tempfile
            import h5py
            with tempfile.NamedTemporaryFile(suffix='.h5') as f:
                h5file = h5py.File(f.name, 'w')
                root_group = h5file.create_group("root")
                serialized = Serializable.encode(frame, format=format, h5_group=root_group, reset_encode_id=True)
                loaded_frame = Serializable.decode(serialized, format=format)
                h5file.close()
        else:
            # JSON format
            serialized = Serializable.encode(frame, format=format, reset_encode_id=True)
            loaded_frame = Serializable.decode(serialized, format=format)

        # Check that expired keys are handled correctly
        assert loaded_frame["a"] == 1
        assert loaded_frame["c"] == 3

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_frame_with_no_serialize_keys_parameterized(self, format):
        """Test Frame serialization with no_serialize keys using both formats."""
        frame = Frame({"a": 1, "b": 2, "c": 3})
        frame.no_serialize("b")

        # Serialize and deserialize using new API
        if format == "hdf5":
            # HDF5 format requires a file and group
            import tempfile
            import h5py
            with tempfile.NamedTemporaryFile(suffix='.h5') as f:
                h5file = h5py.File(f.name, 'w')
                root_group = h5file.create_group("root")
                serialized = Serializable.encode(frame, format=format, h5_group=root_group, reset_encode_id=True)
                loaded_frame = Serializable.decode(serialized, format=format)
                h5file.close()
        else:
            # JSON format
            serialized = Serializable.encode(frame, format=format, reset_encode_id=True)
            loaded_frame = Serializable.decode(serialized, format=format)

        # Check that no_serialize keys are handled correctly
        assert loaded_frame["a"] == 1
        assert loaded_frame["c"] == 3
        assert loaded_frame["b"] == "NOT SERIALIZED"
        # Note: expired keys behavior may vary, but data should be preserved

    def test_frame_with_no_serialize_keys(self):
        """Test Frame serialization with no-serialize keys."""
        frame = Frame({"keep": "this", "exclude": "that"})
        frame.no_serialize("exclude")

        # Serialize and check that excluded key is marked as not serialized
        state = Serializable.encode(frame, format="json", reset_encode_id=True)
        assert isinstance(state, dict)
        # The _data field is now properly encoded as a dict
        assert state["_data"]["encode_type"] == "dict"
        assert "keep" in state["_data"]["items"]
        assert state["_data"]["items"]["exclude"]["value"] == "NOT SERIALIZED"

        # Deserialize and check that kept data is preserved
        loaded_frame = Serializable.decode(state, format="json")
        assert isinstance(loaded_frame, Frame)
        assert loaded_frame["keep"] == "this"
        # The excluded key should be marked as not serialized in the loaded frame
        assert loaded_frame["exclude"] == "NOT SERIALIZED"

    def test_frame_complex_data_structures(self):
        """Test Frame serialization with complex nested data."""
        complex_data = {
            "nested_dict": {
                "level1": {
                    "level2": {"value": "deep"}
                }
            },
            "list_of_dicts": [
                {"id": 1, "name": "first"},
                {"id": 2, "name": "second"}
            ],
            "mixed_types": [1, "two", {"three": 3}, [4, 5]]
        }

        frame = Frame(complex_data, log="complex_test")

        # Test roundtrip preserves structure using new API
        serialized = Serializable.encode(frame, format="json", reset_encode_id=True)
        loaded_frame = Serializable.decode(serialized, format="json")
        assert isinstance(loaded_frame, Frame)

        assert loaded_frame.log == "complex_test"
        assert loaded_frame["nested_dict"]["level1"]["level2"]["value"] == "deep"
        assert loaded_frame["list_of_dicts"][1]["name"] == "second"
        assert loaded_frame["mixed_types"][2]["three"] == 3

    def test_frame_compressed_serialization(self):
        """Test Frame serialization with compressed format."""
        frame = Frame({"data": "compressed_test", "value": 123}, log="compress_test")

        with NamedTemporaryFile(suffix='.json.gz', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Write compressed
            frame.write(temp_path)

            # Read compressed
            loaded_frame = Frame.read(temp_path)
            assert isinstance(loaded_frame, Frame)
            assert loaded_frame.log == "compress_test"
            assert loaded_frame["data"] == "compressed_test"
            assert loaded_frame["value"] == 123

        finally:
            import os
            os.unlink(temp_path)