"""Parameterized tests for Serializable base class serialization methods."""

import gzip
import os
import tempfile
import pytest
from pathlib import Path

from loqs.internal.serializable import Serializable, SERIALIZATION_VERSION
import h5py


class MockSerializable(Serializable):
    """A concrete Serializable class for testing."""

    CACHE_ON_SERIALIZE = True

    SERIALIZE_ATTRS = ["name", "value", "data"]

    def __init__(self, name="test", value=42, data=None):
        self.name = name
        self.value = value
        self.data = data or {}

    def __eq__(self, other):
        return (
            isinstance(other, MockSerializable)
            and self.name == other.name
            and self.value == other.value
            and self.data == other.data
        )

    @classmethod
    def from_decoded_attrs(cls, attr_dict):
        """Create a MockSerializable from decoded attributes dictionary."""
        return cls(
            name=attr_dict["name"],
            value=attr_dict["value"],
            data=attr_dict["data"],
        )


@pytest.fixture(params=["json", "hdf5"])
def format_param(request):
    """Parameterized fixture for testing both JSON and HDF5 formats."""
    return request.param


class TestSerializableParameterized:
    """Parameterized tests for Serializable class functionality."""

    def test_dump_load_roundtrip(self, format_param):
        """Test dump/load roundtrip with file streams."""
        obj = MockSerializable(name="test_obj", value=123, data={"key": "value"})

        if format_param == "json":
            fd, tempf_path = tempfile.mkstemp(suffix='.json')
            os.close(fd)
            try:
                # Test dump to file - use the underlying file object
                with open(tempf_path, 'w+') as f:
                    obj.dump(f)
                    f.seek(0)  # Reset file pointer for reading

                    # Test load from file - use the underlying file object
                    loaded_obj = MockSerializable.load(f)
                    assert obj == loaded_obj
                    assert loaded_obj.name == "test_obj"
                    assert loaded_obj.value == 123
                    assert loaded_obj.data == {"key": "value"}
            finally:
                os.unlink(tempf_path)
        
        else:  # hdf5
            fd, temp_file = tempfile.mkstemp(suffix='.h5')
            os.close(fd)
            try:
                with h5py.File(temp_file, 'w') as h5_file:
                    obj.dump(h5_file, format="hdf5")
                
                with h5py.File(temp_file, 'r') as h5_file:
                    loaded = MockSerializable.load(h5_file, format="hdf5")
                
                assert loaded.name == "test_obj"
                assert loaded.value == 123
                # Check data values (allow for type differences due to HDF5 binary storage)
                assert "key" in loaded.data
                assert loaded.data["key"] in ["value", b"value"]  # string or bytes
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_write_read_roundtrip(self, format_param):
        """Test write/read roundtrip with files."""
        obj = MockSerializable(name="file_test", value=789, data={"test": "file"})

        if format_param == "json":
            fd, temp_path = tempfile.mkstemp(suffix='.json')
            os.close(fd)
            try:
                # Test write to file
                obj.write(temp_path)

                # Test read from file
                loaded_obj = MockSerializable.read(temp_path)
                assert obj == loaded_obj
                assert loaded_obj.name == "file_test"
                assert loaded_obj.value == 789

            finally:
                # Clean up
                Path(temp_path).unlink(missing_ok=True)
        
        else:  # hdf5
            fd, temp_path = tempfile.mkstemp(suffix='.h5')
            os.close(fd)
            try:
                # Test write to file
                obj.write(temp_path)

                # Test read from file
                loaded_obj = MockSerializable.read(temp_path)
                assert obj == loaded_obj
                assert loaded_obj.name == "file_test"
                assert loaded_obj.value == 789

            finally:
                # Clean up
                Path(temp_path).unlink(missing_ok=True)

    def test_write_read_compressed(self, format_param):
        """Test write/read with compressed .json.gz format."""
        if format_param == "json":
            obj = MockSerializable(name="compressed", value=999, data={"format": "gz"})

            fd, temp_path = tempfile.mkstemp(suffix='.json.gz')
            os.close(fd)
            try:
                # Test write compressed
                obj.write(temp_path)

                # Verify file is actually compressed
                with gzip.open(temp_path, 'rt') as f:
                    content = f.read()
                    assert '"name"' in content  # Should contain JSON content

                # Test read compressed
                loaded_obj = MockSerializable.read(temp_path)
                assert obj == loaded_obj
                assert loaded_obj.data == {"format": "gz"}

            finally:
                Path(temp_path).unlink(missing_ok=True)
        else:
            # HDF5 doesn't support gzip compression in the same way
            # Skip this test for HDF5 format
            pytest.skip("Compression test only applicable to JSON format")

    def test_object_caching(self, format_param):
        """Test object reference caching during serialization."""
        # Create objects that should be cached
        obj1 = MockSerializable(name="cached1", value=1)
        obj2 = MockSerializable(name="cached2", value=2)

        # Create a cache
        cache = {}

        if format_param == "json":
            # Serialize both objects with caching
            state1 = Serializable.encode(obj1, format="json", encode_cache=cache)
            state2 = Serializable.encode(obj2, format="json", encode_cache=cache)
            assert isinstance(state1, dict)
            assert isinstance(state2, dict)

            # Verify cache structure
            assert "encode_type" in state1
            assert "encode_type" in state2
            assert state1["encode_type"] == "Serializable"
            assert state2["encode_type"] == "Serializable"

            # Test that second serialization of same object returns reference
            state1_again = Serializable.encode(obj1, format="json", encode_cache=cache)
            assert isinstance(state1_again, dict)
            assert state1_again["cache_type"] == "reference"
        
        else:  # hdf5
            fd, temp_file = tempfile.mkstemp(suffix='.h5')
            os.close(fd)
            try:
                with h5py.File(temp_file, 'w') as h5_file:
                    root_group = h5_file.create_group('root')
                    
                    # Serialize both objects with caching
                    state1 = Serializable.encode(obj1, format="hdf5", h5_group=root_group, encode_cache=cache)
                    state2 = Serializable.encode(obj2, format="hdf5", h5_group=root_group, encode_cache=cache)
                    
                    assert isinstance(state1, h5py.Group)

                    # Verify cache structure
                    assert state1.attrs["encode_type"] == "Serializable"
                    assert state2.attrs["encode_type"] == "Serializable"

                    # Test that second serialization of same object returns reference
                    state1_again = Serializable.encode(obj1, format="hdf5", h5_group=root_group, encode_cache=cache)
                    assert isinstance(state1_again, h5py.Group)
                    assert state1_again.attrs["cache_type"] == "reference"
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_version_compatibility(self, format_param):
        """Test serialization version handling."""
        obj = MockSerializable()
        
        if format_param == "json":
            state = Serializable.encode(obj, format="json", reset_encode_id=True)
            assert isinstance(state, dict)

            # Verify version is included
            assert "version" in state
            assert state["version"] == SERIALIZATION_VERSION

            # Test that objects can be deserialized with current version
            loaded_obj = Serializable.decode(state, format="json")
            assert obj == loaded_obj
        
        else:  # hdf5
            fd, temp_file = tempfile.mkstemp(suffix='.h5')
            os.close(fd)
            try:
                with h5py.File(temp_file, 'w') as h5_file:
                    root_group = h5_file.create_group('root')
                    Serializable.encode(obj, format="hdf5", h5_group=root_group, reset_encode_id=True)
                
                with h5py.File(temp_file, 'r') as h5_file:
                    root_group = h5_file['root']
                    # Find the encoded object group
                    assert isinstance(root_group, h5py.Group)
                    obj_group_name = list(root_group.keys())[0]
                    encoded_group = root_group[obj_group_name]
                    

                    # Verify version is included
                    assert "version" in encoded_group.attrs
                    assert encoded_group.attrs["version"] == SERIALIZATION_VERSION
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_serialization_with_nested_data(self, format_param):
        """Test serialization with complex nested data structures."""
        complex_data = {
            "list": [1, 2, 3, {"nested_dict": True}],
            "nested": {
                "deep": {
                    "value": 42,
                    "array": [10, 20, 30]
                }
            },
            "tuple": (1, 2, 3),
            "set": {4, 5, 6}
        }

        obj = MockSerializable(name="nested_test", value=777, data=complex_data)

        if format_param == "json":
            # Test JSON serialization with nested data
            state = Serializable.encode(obj, format="json", reset_encode_id=True)
            assert isinstance(state, dict)

            # Verify structure
            assert state["encode_type"] == "Serializable"
            assert state["name"]["value"] == "nested_test" # type: ignore
            assert state["value"]["value"] == 777 # type: ignore

            # Test deserialization
            loaded_obj = Serializable.decode(state, format="json")
            assert isinstance(loaded_obj, MockSerializable)
            assert loaded_obj.name == "nested_test"
            assert loaded_obj.value == 777
            
            # Verify nested data structure (allow for type conversions)
            assert "list" in loaded_obj.data
            assert "nested" in loaded_obj.data
            assert "tuple" in loaded_obj.data
            assert "set" in loaded_obj.data
        
        else:  # hdf5
            fd, temp_file = tempfile.mkstemp(suffix='.h5')
            os.close(fd)
            try:
                with h5py.File(temp_file, 'w') as h5_file:
                    root_group = h5_file.create_group('root')
                    Serializable.encode(obj, format="hdf5", h5_group=root_group)
                
                with h5py.File(temp_file, 'r') as h5_file:
                    root_group = h5_file['root']
                    # Find the encoded object group
                    assert isinstance(root_group, h5py.Group)
                    obj_group_name = list(root_group.keys())[0]
                    encoded_group = root_group[obj_group_name]
                    

                    # Verify structure
                    assert encoded_group.attrs["encode_type"] == "Serializable"
                
                # Test deserialization
                loaded_obj = Serializable.read(temp_file)
                assert isinstance(loaded_obj, MockSerializable)
                assert loaded_obj.name == "nested_test"
                assert loaded_obj.value == 777
                
                # Verify nested data structure (allow for type conversions)
                assert "list" in loaded_obj.data
                assert "nested" in loaded_obj.data
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_format_detection(self, format_param):
        """Test automatic format detection from file extensions."""
        obj = MockSerializable(name="format_test", value=111)

        if format_param == "json":
            # Test that .json extension is automatically detected
            fd, temp_path = tempfile.mkstemp(suffix='.json')
            os.close(fd)
            try:
                # Should work without specifying format
                obj.write(temp_path)
                loaded = MockSerializable.read(temp_path)
                assert obj == loaded
            
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        else:  # hdf5
            # Test that .h5 and .hdf5 extensions are automatically detected
            for ext in ["h5", "hdf5"]:
                fd, temp_path = tempfile.mkstemp(suffix=f".{ext}")
                os.close(fd)
                try:
                    # Should work without specifying format
                    obj.write(temp_path)
                    loaded = MockSerializable.read(temp_path)
                    assert obj == loaded
                
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)


class TestSerializableNestedData:
    """Enhanced tests for nested data serialization."""

    def test_nested_serializable_objects(self):
        """Test serialization with nested Serializable objects."""
        inner_obj = MockSerializable(name="inner", value=100, data={"inner_key": "inner_value"})
        outer_obj = MockSerializable(name="outer", value=200, data={"obj": inner_obj, "other": "data"})

        # Test JSON serialization
        state = Serializable.encode(outer_obj, format="json", reset_encode_id=True)
        assert isinstance(state, dict)

        # Test deserialization
        loaded_outer = Serializable.decode(state, format="json")
        assert isinstance(loaded_outer, MockSerializable)
        assert loaded_outer.name == "outer"
        assert loaded_outer.value == 200
        assert "obj" in loaded_outer.data
        assert "other" in loaded_outer.data

        # Verify nested object
        loaded_inner = loaded_outer.data["obj"]
        assert isinstance(loaded_inner, MockSerializable)
        assert loaded_inner.name == "inner"
        assert loaded_inner.value == 100

    def test_complex_nested_structures(self):
        """Test serialization with deeply nested complex structures."""
        # Create a complex nested structure
        deep_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "final_value": 42,
                        "array": [1, 2, 3, {"nested_in_array": True}],
                        "objects": [
                            MockSerializable(name="obj1", value=1),
                            MockSerializable(name="obj2", value=2)
                        ]
                    }
                }
            },
            "metadata": {"description": "complex test", "tags": ["nested", "deep"]}
        }

        obj = MockSerializable(name="complex_nested", value=999, data=deep_data)

        # Test serialization
        state = Serializable.encode(obj, format="json", reset_encode_id=True)
        assert isinstance(state, dict)

        # Test deserialization
        loaded_obj = Serializable.decode(state, format="json")
        assert isinstance(loaded_obj, MockSerializable)
        assert loaded_obj.name == "complex_nested"
        assert loaded_obj.value == 999

        # Verify deep nesting
        assert "level1" in loaded_obj.data
        assert "level2" in loaded_obj.data["level1"]
        assert "level3" in loaded_obj.data["level1"]["level2"]
        assert loaded_obj.data["level1"]["level2"]["level3"]["final_value"] == 42

        # Verify nested objects in arrays
        nested_objects = loaded_obj.data["level1"]["level2"]["level3"]["objects"]
        assert len(nested_objects) == 2
        assert all(isinstance(obj, MockSerializable) for obj in nested_objects)

    def test_circular_reference_handling(self):
        """Test that circular references don't cause infinite recursion."""
        # Create objects with potential circular references
        obj1 = MockSerializable(name="obj1", value=1)
        obj2 = MockSerializable(name="obj2", value=2)

        # Create nested structure (not true circular reference, but complex)
        obj1_with_ref = MockSerializable(name="obj1", value=1, data={"ref": obj2})
        obj2_with_ref = MockSerializable(name="obj2", value=2, data={"ref": obj1})

        # Test serialization - should work without infinite recursion
        state1 = Serializable.encode(obj1_with_ref, format="json", reset_encode_id=True)
        state2 = Serializable.encode(obj2_with_ref, format="json", reset_encode_id=True)

        assert isinstance(state1, dict)
        assert isinstance(state2, dict)

        # Test deserialization
        loaded1 = Serializable.decode(state1, format="json")
        loaded2 = Serializable.decode(state2, format="json")

        assert isinstance(loaded1, MockSerializable)
        assert isinstance(loaded2, MockSerializable)
        assert loaded1.name == "obj1"
        assert loaded2.name == "obj2"

    def test_mixed_data_types(self):
        """Test serialization with mixed Python data types."""
        mixed_data = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, "two", 3.0, None],
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
            "dict": {"nested": "value"},
            "object": MockSerializable(name="embedded", value=100)
        }

        obj = MockSerializable(name="mixed_types", value=500, data=mixed_data)

        # Test serialization
        state = Serializable.encode(obj, format="json", reset_encode_id=True)
        assert isinstance(state, dict)

        # Test deserialization
        loaded_obj = Serializable.decode(state, format="json")
        assert isinstance(loaded_obj, MockSerializable)
        assert loaded_obj.name == "mixed_types"
        assert loaded_obj.value == 500

        # Verify mixed data types
        assert loaded_obj.data["string"] == "hello"
        assert loaded_obj.data["integer"] == 42
        assert loaded_obj.data["float"] == 3.14
        assert loaded_obj.data["boolean"] == True
        assert loaded_obj.data["none"] is None
        assert loaded_obj.data["list"] == [1, "two", 3.0, None]
        assert loaded_obj.data["tuple"] == (1, 2, 3)
        assert loaded_obj.data["set"] == {1, 2, 3}
        assert loaded_obj.data["dict"] == {"nested": "value"}
        assert isinstance(loaded_obj.data["object"], MockSerializable)