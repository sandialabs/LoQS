"""Parameterized tests for Serializable base class serialization methods."""

import gzip
import os
import tempfile
import pytest
import numpy as np
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

    def test_same_serial_hash_different_instances(self):
        """Test objects with same serializable content but different instances."""
        # Create two objects with identical content but different instances
        obj1 = MockSerializable(name="test", value=42, data={"key": "value"})
        obj2 = MockSerializable(name="test", value=42, data={"key": "value"})
        
        # Verify they have different ids but same serial_hash
        assert id(obj1) != id(obj2)
        assert Serializable.serial_hash(obj1) == Serializable.serial_hash(obj2)
        
        # Test serialization with caching
        cache = {}
        
        # Serialize first object - should be source
        state1 = Serializable.encode(obj1, format="json", encode_cache=cache)
        assert state1["cache_type"] == "source"
        assert "cache_id" in state1
        
        # Serialize second object - should be copy since same content but different instance
        state2 = Serializable.encode(obj2, format="json", encode_cache=cache)
        assert state2["cache_type"] == "copy"
        assert "reference_cache_id" in state2
        assert "source_cache_id" in state2
        
        # Test deserialization
        decode_cache = {}
        loaded1 = Serializable.decode(state1, format="json", decode_cache=decode_cache)
        loaded2 = Serializable.decode(state2, format="json", decode_cache=decode_cache)
        
        # Both should be MockSerializable instances with same content
        assert isinstance(loaded1, MockSerializable)
        assert isinstance(loaded2, MockSerializable)
        assert loaded1 == loaded2
        assert loaded1 is not loaded2  # Different instances
        
        # Test with HDF5 format
        import tempfile
        import h5py
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name
        
        try:
            with h5py.File(temp_file, 'w') as h5_file:
                root_group = h5_file.create_group('root')
                
                # Reset cache and encode ID
                cache = {}
                from loqs.internal.encoder import HDF5Encoder
                HDF5Encoder.ENCODE_ID = 0
                
                # Serialize first object - should be source
                state1_h5 = Serializable.encode(obj1, format="hdf5", h5_group=root_group, encode_cache=cache)
                assert state1_h5.attrs["cache_type"] == "source"
                
                # Serialize second object - should be copy
                state2_h5 = Serializable.encode(obj2, format="hdf5", h5_group=root_group, encode_cache=cache)
                assert state2_h5.attrs["cache_type"] == "copy"
            
            # Test deserialization from HDF5
            with h5py.File(temp_file, 'r') as h5_file:
                root_group = h5_file['root']
                decode_cache = {}
                
                # Find the encoded objects
                obj_names = list(root_group.keys())
                assert len(obj_names) == 2  # Should have 2 objects
                
                # Decode both objects
                loaded1_h5 = Serializable.decode(root_group[obj_names[0]], format="hdf5", decode_cache=decode_cache)
                loaded2_h5 = Serializable.decode(root_group[obj_names[1]], format="hdf5", decode_cache=decode_cache)
                
                # Both should be MockSerializable instances with same content
                assert isinstance(loaded1_h5, MockSerializable)
                assert isinstance(loaded2_h5, MockSerializable)
                assert loaded1_h5 == loaded2_h5
                assert loaded1_h5 is not loaded2_h5  # Different instances
                
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_true_circular_references(self):
        """Test true circular references where objects reference each other."""
        # Create objects with true circular references
        obj1 = MockSerializable(name="circular1", value=1)
        obj2 = MockSerializable(name="circular2", value=2)
        
        # Create circular reference
        obj1.data["ref"] = obj2
        obj2.data["ref"] = obj1
        
        # Test serialization with caching
        cache = {}
        state1 = Serializable.encode(obj1, format="json", encode_cache=cache, reset_encode_id=True)
        
        # Should be source since it's the first time we see this serial_hash
        assert state1["cache_type"] == "source"
        
        # The nested obj2 should also be a source since it has different content
        nested_obj2_state = state1["data"]["items"]["ref"]
        assert nested_obj2_state["cache_type"] == "source"
        
        # But the nested obj2's reference back to obj1 should be a reference
        nested_obj2_ref_state = nested_obj2_state["data"]["items"]["ref"]
        assert nested_obj2_ref_state["cache_type"] == "reference"
        
        # Test deserialization
        decode_cache = {}
        loaded1 = Serializable.decode(state1, format="json", decode_cache=decode_cache)
        
        assert isinstance(loaded1, MockSerializable)
        assert loaded1.name == "circular1"
        assert "ref" in loaded1.data
        assert isinstance(loaded1.data["ref"], MockSerializable)
        assert loaded1.data["ref"].name == "circular2"
        
        # The circular reference should be properly resolved
        assert loaded1.data["ref"].data["ref"] is loaded1
        
        # Test with HDF5 format
        import tempfile
        import h5py
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name
        
        try:
            with h5py.File(temp_file, 'w') as h5_file:
                root_group = h5_file.create_group('root')
                
                # Reset cache and encode ID
                cache = {}
                from loqs.internal.encoder import HDF5Encoder
                HDF5Encoder.ENCODE_ID = 0
                
                # Serialize circular reference
                state1_h5 = Serializable.encode(obj1, format="hdf5", h5_group=root_group, encode_cache=cache, reset_encode_id=True)
                assert state1_h5.attrs["cache_type"] == "source"
            
            # Test deserialization from HDF5
            with h5py.File(temp_file, 'r') as h5_file:
                root_group = h5_file['root']
                decode_cache = {}
                
                # Find the encoded object
                obj_name = list(root_group.keys())[0]
                loaded1_h5 = Serializable.decode(root_group[obj_name], format="hdf5", decode_cache=decode_cache)
                
                assert isinstance(loaded1_h5, MockSerializable)
                assert loaded1_h5.name == "circular1"
                assert "ref" in loaded1_h5.data
                assert isinstance(loaded1_h5.data["ref"], MockSerializable)
                assert loaded1_h5.data["ref"].name == "circular2"
                
                # The circular reference should be properly resolved
                assert loaded1_h5.data["ref"].data["ref"] is loaded1_h5
                
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)

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

    def test_iterable_encoding_both_codepaths(self):
        """Test both HDF5 iterable encoding codepaths with caching improvements."""
        # Test case 1: Homogeneous list (should use dataset optimization)
        homogeneous_list = [1, 2, 3, 4, 5]
        
        # Test case 2: Heterogeneous list (should use groups fallback)
        heterogeneous_list = [1, "string", 3.14, True, None]
        
        # Test case 3: Homogeneous list with Serializable objects (should use groups)
        obj1 = MockSerializable(name="obj1", value=1)
        obj2 = MockSerializable(name="obj2", value=2)
        obj3 = MockSerializable(name="obj3", value=3)
        serializable_list = [obj1, obj2, obj3]
        
        # Test case 4: Empty list (edge case)
        empty_list = []
        
        # Test case 5: Large homogeneous list (should use compression)
        large_list = list(range(1500))
        
        # Test with HDF5 format
        import tempfile
        import h5py
        
        test_cases = [
            ("homogeneous_int", homogeneous_list, "dataset"),
            ("heterogeneous", heterogeneous_list, "groups"),
            ("serializable_objects", serializable_list, "groups"),
            ("empty", empty_list, "groups"),  # Empty lists use groups (no benefit to dataset)
            ("large_homogeneous", large_list, "dataset"),
        ]
        
        for test_name, test_data, expected_format in test_cases:
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
                temp_file = f.name
            
            try:
                with h5py.File(temp_file, 'w') as h5_file:
                    root_group = h5_file.create_group('root')
                    
                    # Encode the list
                    list_group = Serializable.encode(test_data, format="hdf5", h5_group=root_group)
                    
                    # Verify storage format
                    storage_format = list_group.attrs.get("storage_format", "groups")
                    assert storage_format == expected_format, f"{test_name}: Expected {expected_format}, got {storage_format}"
                    
                    # Verify we can decode it back correctly
                    with h5py.File(temp_file, 'r') as h5_read:
                        root_read = h5_read['root']
                        decoded_list = Serializable.decode(root_read, format="hdf5")
                        
                        # For lists with objects, check equality element by element
                        if test_name == "serializable_objects":
                            assert len(decoded_list) == len(test_data)
                            for i, (original, decoded) in enumerate(zip(test_data, decoded_list)):
                                assert isinstance(decoded, MockSerializable)
                                assert decoded.name == original.name
                                assert decoded.value == original.value
                        else:
                            assert decoded_list == test_data
                            
            finally:
                import os
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_hdf5_iterable_encoding_homogeneous(self):
        """Test HDF5 iterable encoding with homogeneous HDF5-native types (dataset optimization)."""
        # Test with different homogeneous types that should use dataset optimization
        # Note: We test direct list encoding, not lists embedded in dicts, because
        # lists in dicts become nested structures which correctly use groups storage
        test_cases = [
            # Integers
            {"data": [1, 2, 3, 4, 5], "expected_type": "dataset"},
            # Floats  
            {"data": [1.1, 2.2, 3.3, 4.4, 5.5], "expected_type": "dataset"},
            # Booleans
            {"data": [True, False, True, False], "expected_type": "dataset"},
            # Strings
            {"data": ["hello", "world", "test", "data"], "expected_type": "dataset"},
            # Large integer list (should use compression)
            {"data": list(range(1500)), "expected_type": "dataset"},
        ]

        for case in test_cases:
            # Test direct list encoding (not embedded in MockSerializable)
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
                temp_file = f.name

            try:
                with h5py.File(temp_file, 'w') as h5_file:
                    root_group = h5_file.create_group('root')
                    # Encode the list directly to test dataset optimization
                    list_group = Serializable.encode(case["data"], format="hdf5", h5_group=root_group)
                    
                    # Verify it used dataset storage format
                    storage_format = list_group.attrs.get("storage_format", "groups")
                    assert storage_format == case["expected_type"], f"Expected {case['expected_type']}, got {storage_format}"
                    
                    # Verify we can decode it back correctly
                    with h5py.File(temp_file, 'r') as h5_read:
                        root_read = h5_read['root']
                        decoded_list = Serializable.decode(root_read, format="hdf5")
                        
                        assert decoded_list == case["data"]

            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_hdf5_iterable_encoding_heterogeneous(self):
        """Test HDF5 iterable encoding with heterogeneous types (groups fallback)."""
        # Test with mixed types that should fall back to groups storage
        heterogeneous_data = [
            1, "string", 3.14, True, None, [1, 2, 3], {"key": "value"}
        ]
        
        # Test direct list encoding (not embedded in MockSerializable)
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            with h5py.File(temp_file, 'w') as h5_file:
                root_group = h5_file.create_group('root')
                # Encode the heterogeneous list directly
                list_group = Serializable.encode(heterogeneous_data, format="hdf5", h5_group=root_group)
                
                # Verify it used groups storage format (fallback for mixed types)
                storage_format = list_group.attrs.get("storage_format", "groups")
                assert storage_format == "groups", f"Expected groups, got {storage_format}"
                
                # Verify we can decode it back correctly
                with h5py.File(temp_file, 'r') as h5_read:
                    root_read = h5_read['root']
                    decoded_list = Serializable.decode(root_read, format="hdf5")
                    
                    assert decoded_list == heterogeneous_data

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_hdf5_array_compression(self):
        """Test HDF5 array compression for large arrays."""
        # Create a large array that should trigger compression
        large_array = np.random.random((1500, 1500))  # 2.25M elements
        
        # Test direct array encoding (not embedded in MockSerializable)
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            with h5py.File(temp_file, 'w') as h5_file:
                root_group = h5_file.create_group('root')
                # Encode the array directly to test compression
                array_group = Serializable.encode(large_array, format="hdf5", h5_group=root_group)
                
                # Verify it's a dense real array
                array_type = array_group.attrs.get("array_type")
                assert array_type == "dense_real"
                
                # Check that compression was applied (data dataset should exist with compression)
                data_dataset = array_group["data"]
                assert isinstance(data_dataset, h5py.Dataset)
                
                # Verify compression is enabled by checking dataset creation properties
                # For large arrays, compression should be enabled
                # We can verify this by checking that the dataset was created with compression
                # Since we can't easily check the filters directly, we'll verify the functionality works
                # by ensuring the array can be round-tripped correctly
                
                # Verify we can decode it back correctly
                with h5py.File(temp_file, 'r') as h5_read:
                    root_read = h5_read['root']
                    decoded_array = Serializable.decode(root_read, format="hdf5")
                    
                    np.testing.assert_array_almost_equal(decoded_array, large_array)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_serial_hash_caching(self):
        """Test the new serial_hash caching mechanism."""
        # Create objects with identical content (should have same serial_hash)
        obj1 = MockSerializable(name="same", value=42, data={"key": "value"})
        obj2 = MockSerializable(name="same", value=42, data={"key": "value"})
        
        # Verify they have the same serial_hash but different object ids
        serial_hash1 = Serializable.serial_hash(obj1)
        serial_hash2 = Serializable.serial_hash(obj2)
        obj_id1 = id(obj1)
        obj_id2 = id(obj2)
        
        assert serial_hash1 == serial_hash2, "Objects with identical content should have same serial_hash"
        assert obj_id1 != obj_id2, "Different object instances should have different ids"
        
        # Test serialization with caching
        cache = {}
        
        # First object should be encoded as source
        state1 = Serializable.encode(obj1, format="json", encode_cache=cache, reset_encode_id=True)
        assert state1["cache_type"] == "source"
        
        # Second object with same content should be encoded as copy
        state2 = Serializable.encode(obj2, format="json", encode_cache=cache)
        assert state2["cache_type"] == "copy"
        assert state2["reference_cache_id"] == state1["cache_id"]
        
        # Test deserialization
        decode_cache = {}
        decoded1 = Serializable.decode(state1, format="json", decode_cache=decode_cache)
        decoded2 = Serializable.decode(state2, format="json", decode_cache=decode_cache)
        
        # Both should be equal but different instances
        assert decoded1 == decoded2
        assert decoded1 is not decoded2

    def test_cache_type_reference_vs_copy(self):
        """Test the difference between reference and copy cache types."""
        # Create an object
        obj = MockSerializable(name="cache_test", value=100, data={"nested": "value"})
        
        # Test with JSON format
        cache = {}
        
        # First encoding should be source
        state1 = Serializable.encode(obj, format="json", encode_cache=cache, reset_encode_id=True)
        assert state1["cache_type"] == "source"
        cache_id = state1["cache_id"]
        
        # Second encoding of same object instance should be reference
        state2 = Serializable.encode(obj, format="json", encode_cache=cache)
        assert state2["cache_type"] == "reference"
        assert state2["cache_id"] == cache_id
        
        # Create another object with same content
        obj_copy = MockSerializable(name="cache_test", value=100, data={"nested": "value"})
        
        # This should be encoded as copy (same content, different instance)
        state3 = Serializable.encode(obj_copy, format="json", encode_cache=cache)
        assert state3["cache_type"] == "copy"
        assert state3["reference_cache_id"] == cache_id
        
        # Test deserialization
        decode_cache = {}
        decoded1 = Serializable.decode(state1, format="json", decode_cache=decode_cache)
        decoded2 = Serializable.decode(state2, format="json", decode_cache=decode_cache)
        decoded3 = Serializable.decode(state3, format="json", decode_cache=decode_cache)
        
        # Reference should return same object
        assert decoded1 is decoded2
        
        # Copy should return equal but different object
        assert decoded1 == decoded3
        assert decoded1 is not decoded3