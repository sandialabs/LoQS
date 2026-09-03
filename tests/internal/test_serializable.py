"""Parameterized tests for Serializable base class serialization methods."""

import pytest
import numpy as np
import h5py
from unittest.mock import MagicMock

from loqs.core import Frame
from loqs.internal.serializable import (
    Serializable,
    SERIALIZATION_VERSION,
    ResolvingDecodeCache,
    DeferredRef,
)
from loqs.types import NDArray



class MockSerializable(Serializable):
    """A concrete Serializable class for testing."""

    _CACHE_ON_SERIALIZE = True

    _SERIALIZE_ATTRS = ["name", "value", "data"]

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
    def _from_decoded_attrs(cls, attr_dict):
        """Create a MockSerializable from decoded attributes dictionary."""
        return cls(
            name=attr_dict["name"],
            value=attr_dict["value"],
            data=attr_dict["data"],
        )


@pytest.fixture(params=["json", "json.gz", "hdf5"])
def format_param(request):
    """Parameterized fixture for testing both JSON and HDF5 formats."""
    return request.param


class TestSerializableParameterized:
    """Parameterized tests for Serializable class functionality."""

    def test_dump_load_roundtrip(self, format_param, make_temp_path):
        """Test dump/load roundtrip with file streams."""
        obj = MockSerializable(name="test_obj", value=123, data={"key": "value"})

        if format_param == "json":
            with make_temp_path(suffix='.json') as tempf_path:
                # Test dump to file - use the underlying file object
                with open(tempf_path, 'w+') as f:
                    obj.dump(f)
                    f.seek(0)
                    loaded_obj = MockSerializable.load(f)

        else:  # hdf5
            with make_temp_path(suffix='.h5') as temp_file:
                with h5py.File(temp_file, 'w') as h5_file:
                    obj.dump(h5_file, format="hdf5")
                
                with h5py.File(temp_file, 'r') as h5_file:
                    loaded_obj = MockSerializable.load(h5_file, format="hdf5")
        
        assert isinstance(loaded_obj, MockSerializable)
        assert loaded_obj.name == "test_obj"
        assert loaded_obj.value == 123
        assert loaded_obj.data["key"] == "value"

    def test_write_read_roundtrip(self, format_param, make_temp_path):
        """Test write/read roundtrip with files."""
        obj = MockSerializable(name="file_test", value=789, data={"test": "file"})

        with make_temp_path(suffix=f'.{format_param}') as temp_path:
            # Test write to file
            obj.write(temp_path)

            # Test read from file
            loaded_obj = MockSerializable.read(temp_path)
            assert isinstance(loaded_obj, MockSerializable)
            assert obj == loaded_obj
            assert loaded_obj.name == "file_test"
            assert loaded_obj.value == 789

    def test_object_caching(self, format_param, make_temp_path):
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
            with make_temp_path(suffix='.h5') as temp_file:
                with h5py.File(temp_file, 'w') as h5_file:
                    root_group = h5_file.create_group('root')
                    
                    # Serialize both objects with caching
                    state1 = Serializable.encode(obj1, format="hdf5", h5_group=root_group, encode_cache=cache)
                    state2 = Serializable.encode(obj2, format="hdf5", h5_group=root_group, encode_cache=cache)
                    
                    assert isinstance(state1, h5py.Group)
                    assert isinstance(state2, h5py.Group)

                    # Verify cache structure
                    assert state1.attrs["encode_type"] == "Serializable"
                    assert state2.attrs["encode_type"] == "Serializable"

                    # Test that second serialization of same object returns reference
                    state1_again = Serializable.encode(obj1, format="hdf5", h5_group=root_group, encode_cache=cache)
                    assert isinstance(state1_again, h5py.Group)
                    assert state1_again.attrs["cache_type"] == "reference"

    def test_version_compatibility(self, format_param, make_temp_path):
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
            with make_temp_path(suffix='.h5') as temp_file:
                with h5py.File(temp_file, 'w') as h5_file:
                    obj.dump(h5_file, format="hdf5")

                with h5py.File(temp_file, 'r') as h5_file:
                    root_group = h5_file['root']
                    assert isinstance(root_group, h5py.Group)

                    # Version is stamped once on the file's root group,
                    # not repeated at every node (see HDF5Encoder's
                    # `_HDF5_DECODE_VERSION`).
                    assert "version" in root_group.attrs
                    assert root_group.attrs["version"] == SERIALIZATION_VERSION

                # Test that objects can be deserialized with current version
                loaded_obj = MockSerializable.read(temp_file)
                assert obj == loaded_obj

    def test_serialization_with_nested_data(self, format_param, make_temp_path):
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
            # Non-string primitives are stored bare; strings stay wrapped
            # (see JSONEncoder.encode_primitive for why).
            assert state["name"]["value"] == "nested_test" # type: ignore
            assert state["value"] == 777 # type: ignore

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
            with make_temp_path(suffix='.h5') as temp_file:
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

    def test_format_detection(self, format_param, make_temp_path):
        """Test automatic format detection from file extensions."""
        obj = MockSerializable(name="format_test", value=111)

        if format_param == "json":
            # Test that .json extension is automatically detected
            with make_temp_path(suffix='.json') as temp_path:
                # Should work without specifying format
                obj.write(temp_path)
                loaded = MockSerializable.read(temp_path)
                assert obj == loaded        
        else:  # hdf5
            # Test that .h5 and .hdf5 extensions are automatically detected
            for ext in ["h5", "hdf5"]:
                with make_temp_path(suffix=f".{ext}") as temp_path:
                    # Should work without specifying format
                    obj.write(temp_path)
                    loaded = MockSerializable.read(temp_path)
                    assert obj == loaded


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

    def test_same__serial_hash_different_instances(self, make_temp_path):
        """Test objects with same serializable content but different instances."""
        # Create two objects with identical content but different instances
        obj1 = MockSerializable(name="test", value=42, data={"key": "value"})
        obj2 = MockSerializable(name="test", value=42, data={"key": "value"})
        
        # Verify they have different ids but same _serial_hash
        assert id(obj1) != id(obj2)
        assert Serializable._serial_hash(obj1) == Serializable._serial_hash(obj2)
        
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

        
        with make_temp_path(suffix='.h5') as temp_file:
            with h5py.File(temp_file, 'w') as h5_file:
                root_group = h5_file.create_group('root')
                
                # Reset cache and encode ID
                cache = {}
                from loqs.internal.encoder import HDF5Encoder
                HDF5Encoder.ENCODE_ID = 0
                
                # Serialize first object - should be source
                state1_h5 = Serializable.encode(obj1, format="hdf5", h5_group=root_group, encode_cache=cache)
                assert isinstance(state1_h5, h5py.Group)
                assert state1_h5.attrs["cache_type"] == "source"
                
                # Serialize second object - should be copy
                state2_h5 = Serializable.encode(obj2, format="hdf5", h5_group=root_group, encode_cache=cache)
                assert isinstance(state2_h5, h5py.Group)
                assert state2_h5.attrs["cache_type"] == "copy"
            
            # Test deserialization from HDF5
            with h5py.File(temp_file, 'r') as h5_file:
                root_group = h5_file['root']
                assert isinstance(root_group, h5py.Group)
                decode_cache = {}
                
                # Find the encoded objects
                obj_names = list(root_group.keys())
                assert len(obj_names) == 2  # Should have 2 objects
                
                # Decode both objects
                obj_group1 = root_group[obj_names[0]]
                obj_group2 = root_group[obj_names[1]]
                assert isinstance(obj_group1, h5py.Group)
                assert isinstance(obj_group2, h5py.Group)
                loaded1_h5 = Serializable.decode(obj_group1, format="hdf5", decode_cache=decode_cache)
                loaded2_h5 = Serializable.decode(obj_group2, format="hdf5", decode_cache=decode_cache)
                
                # Both should be MockSerializable instances with same content
                assert isinstance(loaded1_h5, MockSerializable)
                assert isinstance(loaded2_h5, MockSerializable)
                assert loaded1_h5 == loaded2_h5
                assert loaded1_h5 is not loaded2_h5  # Different instances

    def test_true_circular_references(self, make_temp_path):
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
        assert isinstance(state1, dict)
        
        # Should be source since it's the first time we see this _serial_hash
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
        
        with make_temp_path(suffix='.h5') as temp_file:
            with h5py.File(temp_file, 'w') as h5_file:
                root_group = h5_file.create_group('root')
                
                # Reset cache and encode ID
                cache = {}
                from loqs.internal.encoder import HDF5Encoder
                HDF5Encoder.ENCODE_ID = 0
                
                # Serialize circular reference
                state1_h5 = Serializable.encode(obj1, format="hdf5", h5_group=root_group, encode_cache=cache, reset_encode_id=True)
                assert isinstance(state1_h5, h5py.Group)
                assert state1_h5.attrs["cache_type"] == "source"
            
            # Test deserialization from HDF5
            with h5py.File(temp_file, 'r') as h5_file:
                root_group = h5_file['root']
                assert isinstance(root_group, h5py.Group)
                decode_cache = {}
                
                # Find the encoded object
                obj_name = list(root_group.keys())[0]
                obj_group = root_group[obj_name]
                assert isinstance(obj_group, h5py.Group)
                loaded1_h5 = Serializable.decode(obj_group, format="hdf5", decode_cache=decode_cache)
                
                assert isinstance(loaded1_h5, MockSerializable)
                assert loaded1_h5.name == "circular1"
                assert "ref" in loaded1_h5.data
                assert isinstance(loaded1_h5.data["ref"], MockSerializable)
                assert loaded1_h5.data["ref"].name == "circular2"
                
                # The circular reference should be properly resolved
                assert loaded1_h5.data["ref"].data["ref"] is loaded1_h5

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

    def test_iterable_encoding_both_codepaths(self, make_temp_path):
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
        
        test_cases = [
            ("homogeneous_int", homogeneous_list, "dataset"),
            ("heterogeneous", heterogeneous_list, "groups"),
            ("serializable_objects", serializable_list, "groups"),
            ("empty", empty_list, "groups"),  # Empty lists use groups (no benefit to dataset)
            ("large_homogeneous", large_list, "dataset"),
        ]
        
        for test_name, test_data, expected_format in test_cases:
            with make_temp_path(suffix='.h5') as temp_file:
                with h5py.File(temp_file, 'w') as h5_file:
                    root_group = h5_file.create_group('root')
                    
                    # Encode the list
                    list_group = Serializable.encode(test_data, format="hdf5", h5_group=root_group)
                    assert isinstance(list_group, h5py.Group)
                    
                    # Verify storage format
                    storage_format = list_group.attrs.get("storage_format", "groups")
                    assert storage_format == expected_format, f"{test_name}: Expected {expected_format}, got {storage_format}"
                    
                    # Verify we can decode it back correctly
                    with h5py.File(temp_file, 'r') as h5_read:
                        root_read = h5_read['root']
                        assert isinstance(root_read, h5py.Group)
                        decoded_list = Serializable.decode(root_read, format="hdf5")
                        assert isinstance(decoded_list, list)
                        
                        # For lists with objects, check equality element by element
                        if test_name == "serializable_objects":
                            assert len(decoded_list) == len(test_data)
                            for i, (original, decoded) in enumerate(zip(test_data, decoded_list)):
                                assert isinstance(decoded, MockSerializable)
                                assert decoded.name == original.name
                                assert decoded.value == original.value
                        else:
                            assert decoded_list == test_data

    def test_hdf5_iterable_encoding_homogeneous(self, make_temp_path):
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
            with make_temp_path(suffix='.h5') as temp_file:
                with h5py.File(temp_file, 'w') as h5_file:
                    root_group = h5_file.create_group('root')
                    # Encode the list directly to test dataset optimization
                    list_group = Serializable.encode(case["data"], format="hdf5", h5_group=root_group)
                    assert isinstance(list_group, h5py.Group)
                    
                    # Verify it used dataset storage format
                    storage_format = list_group.attrs.get("storage_format", "groups")
                    assert storage_format == case["expected_type"], f"Expected {case['expected_type']}, got {storage_format}"
                    
                    # Verify we can decode it back correctly
                    with h5py.File(temp_file, 'r') as h5_read:
                        root_read = h5_read['root']
                        assert isinstance(root_read, h5py.Group)
                        decoded_list = Serializable.decode(root_read, format="hdf5")
                        
                        assert decoded_list == case["data"]

    def test_hdf5_iterable_encoding_heterogeneous(self, make_temp_path):
        """Test HDF5 iterable encoding with heterogeneous types (groups fallback)."""
        # Test with mixed types that should fall back to groups storage
        heterogeneous_data = [
            1, "string", 3.14, True, None, [1, 2, 3], {"key": "value"}
        ]
        
        # Test direct list encoding (not embedded in MockSerializable)
        with make_temp_path(suffix='.h5') as temp_file:
            with h5py.File(temp_file, 'w') as h5_file:
                root_group = h5_file.create_group('root')
                # Encode the heterogeneous list directly
                list_group = Serializable.encode(heterogeneous_data, format="hdf5", h5_group=root_group)
                assert isinstance(list_group, h5py.Group)
                
                # Verify it used groups storage format (fallback for mixed types)
                storage_format = list_group.attrs.get("storage_format", "groups")
                assert storage_format == "groups", f"Expected groups, got {storage_format}"
                
                # Verify we can decode it back correctly
                with h5py.File(temp_file, 'r') as h5_read:
                    root_read = h5_read['root']
                    assert isinstance(root_read, h5py.Group)
                    decoded_list = Serializable.decode(root_read, format="hdf5")
                    
                    assert decoded_list == heterogeneous_data

    def test_hdf5_array_compression(self, make_temp_path):
        """Test HDF5 array compression for large arrays."""
        # Create a large array that should trigger compression
        large_array = np.random.random((1500, 1500))  # 2.25M elements
        
        # Test direct array encoding (not embedded in MockSerializable)
        with make_temp_path(suffix='.h5') as temp_file:
            with h5py.File(temp_file, 'w') as h5_file:
                root_group = h5_file.create_group('root')
                # Encode the array directly to test compression
                array_group = Serializable.encode(large_array, format="hdf5", h5_group=root_group)
                assert isinstance(array_group, h5py.Group)
                
                # Verify it's a dense real array
                array_type = array_group.attrs.get("array_type")
                assert array_type == "dense_real"
                
                # Check that compression was applied (data dataset should exist with compression)
                data_dataset = array_group["data"]
                assert isinstance(data_dataset, h5py.Dataset)
                assert data_dataset.compression
                
                # Verify we can decode it back correctly
                with h5py.File(temp_file, 'r') as h5_read:
                    root_read = h5_read['root']
                    assert isinstance(root_read, h5py.Group)
                    decoded_array = Serializable.decode(root_read, format="hdf5")
                    assert isinstance(decoded_array, NDArray)
                    
                    np.testing.assert_array_almost_equal(decoded_array, large_array)

    def test_cache_type_reference_vs_copy(self, format_param, make_temp_path):
        """Test the difference between reference and copy cache types."""
        # Create an object
        obj = MockSerializable(name="cache_test", value=100, data={"nested": "value"})

        # Equivalent when encoded, but diff id
        dup_obj = MockSerializable(name="cache_test", value=100, data={"nested": "value"})

        frame = Frame({
            "obj1": obj,
            "obj2": obj, # Same instance, should be a reference to obj
            "obj3": dup_obj, # Same content, diff instance, should be a copy to obj
            "obj4": dup_obj, # Same instance as dup_obj, should be a reference to dup_obj
        })
        
        with make_temp_path(suffix=f".{format_param}") as temp_file:
            frame.write(temp_file)
            decoded = Frame.read(temp_file)
        
        assert isinstance(decoded, Frame)

        # obj2 is obj1
        assert decoded["obj2"] is decoded["obj1"]

        # obj3 should have same hash, but diff id
        assert Serializable._serial_hash(decoded["obj3"]) == Serializable._serial_hash(decoded["obj1"])
        assert decoded["obj3"] is not decoded["obj1"]

        # obj4 is obj3
        assert decoded["obj4"] is decoded["obj3"]

    def test_history_copy_reconstruction_does_not_alias_frame_list(
        self, format_param, make_temp_path
    ):
        """A regression test for the "copy" cache-type reconstruction
        using History's own cheap wrap-constructor (sharing Frame
        identity) instead of a full `copy.deepcopy()`. Two independently-
        decoded Histories built from identical content deliberately share
        their underlying Frame objects by identity -- confirmed here as
        the expected, intentional behavior -- but must never share the
        same underlying frame list itself, or mutating one (`.append()`
        genuinely mutates `History` in place) would silently corrupt the
        other."""
        from loqs.core.history import History

        shared_frame = Frame({"x": 1})
        history_a = History([shared_frame])
        history_b = History([shared_frame])  # same content, different identity

        container = Frame({"history_a": history_a, "history_b": history_b})

        with make_temp_path(suffix=f".{format_param}") as temp_file:
            container.write(temp_file)
            decoded = Frame.read(temp_file)

        decoded_a = decoded["history_a"]
        decoded_b = decoded["history_b"]

        assert decoded_a is not decoded_b
        assert len(decoded_a) == len(decoded_b) == 1
        # The whole point of the optimization: the underlying Frame is
        # shared by identity, not deep-copied.
        assert decoded_a[0] is decoded_b[0]

        # But the outer History containers must stay independent.
        decoded_a.append(Frame({"y": 2}))
        assert len(decoded_a) == 2
        assert len(decoded_b) == 1

    def test_instructionstack_copy_reconstruction_does_not_alias_list(
        self, format_param, make_temp_path
    ):
        """The same regression test as above, for `InstructionStack`'s
        own cheap wrap-constructor. `InstructionStack` itself never
        mutates in place (every "mutation" method returns a new stack),
        so the sharper risk here is simpler: two independently-decoded
        stacks built from identical content must never resolve to the
        *same* object."""
        from loqs.core.instructions.instructionstack import InstructionStack

        stack_a = InstructionStack([("Increment", "L0")])
        stack_b = InstructionStack([("Increment", "L0")])

        container = Frame({"stack_a": stack_a, "stack_b": stack_b})

        with make_temp_path(suffix=f".{format_param}") as temp_file:
            container.write(temp_file)
            decoded = Frame.read(temp_file)

        decoded_a = decoded["stack_a"]
        decoded_b = decoded["stack_b"]

        assert decoded_a is not decoded_b
        assert len(decoded_a) == len(decoded_b) == 1
        # Sharing the underlying InstructionLabel by identity is fine --
        # InstructionStack's own mutation methods never modify one in
        # place, always returning a new stack.
        assert decoded_a[0] is decoded_b[0]

        appended = decoded_a.append_instruction(("Increment", "L1"))
        assert len(appended) == 2
        assert len(decoded_a) == 1
        assert len(decoded_b) == 1


def _load_module_from_source(tmp_path, filename, source):
    """Write `source` to a real file under `tmp_path` and import it as a
    fresh module, so `inspect.getsource`/`getsourcefile` behave exactly
    as they would for a real, on-disk module (unlike a function defined
    directly in a test method, whose "source file" is this test file
    itself, cluttered with unrelated imports)."""
    import importlib.util
    import sys

    path = tmp_path / filename
    path.write_text(source)
    module_name = f"_test_get_function_str_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestGetFunctionStr:
    """Regression tests for Serializable._get_function_str's libcst-based
    import extraction, which replaced an earlier line-based text scan."""

    def test_basic_single_and_multiline_imports(self, tmp_path):
        module = _load_module_from_source(
            tmp_path,
            "mod_basic.py",
            "from collections import OrderedDict\n"
            "from collections.abc import (\n"
            "    Mapping,\n"
            "    Sequence,\n"
            ")\n"
            "\n"
            "def apply_fn(x):\n"
            "    OrderedDict()\n"
            "    return isinstance(x, Mapping)\n",
        )
        result = Serializable._get_function_str(module.apply_fn)
        assert "from collections import OrderedDict" in result
        # Sequence isn't itself referenced, but the whole statement it
        # shares with Mapping (which is) is still pulled in as a unit --
        # matching the granularity of the import-location rewriting
        # elsewhere in this codebase, not a per-name split.
        assert "Mapping" in result
        assert "Sequence" in result

    def test_comment_and_docstring_mentions_are_not_false_positives(
        self, tmp_path
    ):
        module = _load_module_from_source(
            tmp_path,
            "mod_falsepos.py",
            "import unittest\n"
            "\n"
            "def apply_fn(x):\n"
            '    """Talks about unittest but never uses it."""\n'
            "    # unittest appears here too\n"
            "    return x\n",
        )
        result = Serializable._get_function_str(module.apply_fn)
        assert "import unittest" not in result

    def test_conditionally_guarded_import_is_still_found_and_valid(
        self, tmp_path
    ):
        """A name only reachable via `if TYPE_CHECKING:`/`try/except
        ImportError:` is still a real module-scope binding by Python's
        own scoping rules, but its original indented text isn't valid
        syntax once lifted out of its enclosing block on its own --
        regression test for a real bug found while building this."""
        import ast

        module = _load_module_from_source(
            tmp_path,
            "mod_guarded.py",
            "from typing import TYPE_CHECKING\n"
            "\n"
            "if TYPE_CHECKING:\n"
            "    from collections import OrderedDict\n"
            "else:\n"
            "    try:\n"
            "        from collections import OrderedDict\n"
            "    except ImportError:\n"
            "        OrderedDict = dict\n"
            "\n"
            "def apply_fn(x):\n"
            "    return OrderedDict(x)\n",
        )
        result = Serializable._get_function_str(module.apply_fn)
        ast.parse(result)  # must be valid, standalone Python on its own
        assert "OrderedDict" in result

    def test_multiple_references_to_same_import_are_not_duplicated(
        self, tmp_path
    ):
        module = _load_module_from_source(
            tmp_path,
            "mod_dedup.py",
            "import os\n"
            "\n"
            "def apply_fn(x):\n"
            "    return os.path.join(str(os.getcwd()), x)\n",
        )
        result = Serializable._get_function_str(module.apply_fn)
        assert result.count("import os") == 1

    def test_nested_function_finds_module_level_imports(self, tmp_path):
        """A function defined deep inside a class method (matching real
        usage, e.g. a closure in a test) should still resolve
        module-level imports correctly regardless of its own
        indentation."""
        import ast

        module = _load_module_from_source(
            tmp_path,
            "mod_nested.py",
            "from collections import OrderedDict\n"
            "\n"
            "class Builder:\n"
            "    def make(self):\n"
            "        def apply_fn(x):\n"
            "            return OrderedDict(x)\n"
            "        return apply_fn\n",
        )
        apply_fn = module.Builder().make()
        result = Serializable._get_function_str(apply_fn)
        ast.parse(result)
        assert "from collections import OrderedDict" in result

    def test_public_serialize_function_matches_private_helper(
        self, tmp_path
    ):
        module = _load_module_from_source(
            tmp_path,
            "mod_public.py",
            "from collections import OrderedDict\n"
            "\n"
            "def apply_fn(x):\n"
            "    return OrderedDict(x)\n",
        )
        assert Serializable.serialize_function(
            module.apply_fn
        ) == Serializable._get_function_str(module.apply_fn)

    def test_import_extraction_failure_falls_back_to_bare_source(
        self, tmp_path, monkeypatch
    ):
        """If everything up through inspect.getsource succeeds but import
        extraction itself fails for any reason, the bare function
        definition is returned instead of raising -- per the Notes in
        _get_function_str's own docstring."""
        module = _load_module_from_source(
            tmp_path,
            "mod_extractfail.py",
            "from collections import OrderedDict\n"
            "\n"
            "def apply_fn(x):\n"
            "    return OrderedDict(x)\n",
        )

        def _raise(*args, **kwargs):
            raise RuntimeError("simulated import-extraction failure")

        monkeypatch.setattr(
            Serializable, "_imports_needed_by", staticmethod(_raise)
        )
        result = Serializable._get_function_str(module.apply_fn)
        assert "def apply_fn" in result
        assert "import" not in result

    def test_cache_invalidated_when_file_changes(self, tmp_path):
        """_import_usage_index_for_file caches per (path, mtime) -- a
        file edited after its first use must not serve stale imports."""
        import importlib.util
        import os
        import sys

        path = tmp_path / "mod_cache.py"
        path.write_text(
            "from collections import OrderedDict\n"
            "\n"
            "def apply_fn(x):\n"
            "    return OrderedDict(x)\n"
        )
        spec = importlib.util.spec_from_file_location(
            "_test_cache_mod_v1", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["_test_cache_mod_v1"] = module
        spec.loader.exec_module(module)

        result1 = Serializable._get_function_str(module.apply_fn)
        assert "OrderedDict" in result1

        # Rewrite with a different import, forcing a distinct mtime
        # regardless of the filesystem's timestamp resolution.
        mtime = os.stat(path).st_mtime
        path.write_text(
            "from collections import defaultdict\n"
            "\n"
            "def apply_fn(x):\n"
            "    return defaultdict(x)\n"
        )
        os.utime(path, (mtime + 5, mtime + 5))

        spec2 = importlib.util.spec_from_file_location(
            "_test_cache_mod_v2", path
        )
        module2 = importlib.util.module_from_spec(spec2)
        sys.modules["_test_cache_mod_v2"] = module2
        spec2.loader.exec_module(module2)

        result2 = Serializable._get_function_str(module2.apply_fn)
        assert "defaultdict" in result2
        assert "OrderedDict" not in result2


class TestResolvingDecodeCache:
    """Tests for ResolvingDecodeCache with reference-before-source ordering."""

    def test_json_reference_before_source_resolution(self, make_temp_path):
        """Test JSON format with reference decoded before source.

        Constructs a JSON structure where a reference to an object is placed
        and decoded before the source entry is encountered, verifying that
        ResolvingDecodeCache.resolve() can find and decode the source on demand.
        """
        # Manually construct encoded structure with reference appearing first
        # This represents a real case where traversal order doesn't match encoding order
        source_encoded = {
            "encode_type": "Serializable",
            "module": "test_serializable",
            "class": "MockSerializable",
            "version": 2,
            "cache_type": "source",
            "cache_id": 100,
            "name": "shared",
            "value": 42,
            "data": {"key": "val"},
        }

        reference_encoded = {
            "encode_type": "Serializable",
            "module": "test_serializable",
            "class": "MockSerializable",
            "version": 2,
            "cache_type": "reference",
            "cache_id": 100,
        }

        # Create root with reference first (forcing resolve on demand)
        root = {
            "outer": reference_encoded,  # Reference comes first
            "inner": source_encoded,     # Source comes second
        }

        # Create cache and decode
        cache = ResolvingDecodeCache(root=root, format="json")
        decoded_ref = Serializable.decode(
            reference_encoded, format="json", decode_cache=cache
        )

        # Should have resolved the source on demand
        assert isinstance(decoded_ref, MockSerializable)
        assert decoded_ref.name == "shared"
        assert decoded_ref.value == 42
        assert decoded_ref.data == {"key": "val"}

        # Verify it's in cache now
        assert 100 in cache
        assert cache[100] is decoded_ref

    def test_hdf5_reference_before_source_resolution(self, make_temp_path):
        """Test HDF5 format with a "copy" entry decoded before its source.

        Two distinct instances sharing one _serial_hash are encoded as
        siblings (the first becomes cache_type="source", the second
        cache_type="copy"). The "copy" entry is then decoded directly, in
        isolation, with a fresh decode_cache that has never seen the source
        -- forcing ResolvingDecodeCache to locate and decode the source on
        demand rather than relying on prior top-down traversal order.
        """

        with make_temp_path(suffix=".h5") as temp_path:
            source_obj = MockSerializable(name="shared", value=42, data={"x": 1})
            copy_obj = MockSerializable(name="shared", value=42, data={"x": 1})

            encode_cache: dict = {}
            with h5py.File(temp_path, "w") as f:
                container = f.create_group("container", track_order=True)
                first_group = Serializable.encode(
                    source_obj,
                    format="hdf5",
                    encode_cache=encode_cache,
                    h5_group=container,
                )
                second_group = Serializable.encode(
                    copy_obj,
                    format="hdf5",
                    encode_cache=encode_cache,
                    h5_group=container,
                )
                assert first_group.attrs["cache_type"] == "source"
                assert second_group.attrs["cache_type"] == "copy"
                second_name = second_group.name

            # Decode only the "copy" entry, in isolation, with a fresh cache --
            # the source is only ever found via `ResolvingDecodeCache`'s own scan.
            with h5py.File(temp_path, "r") as f:
                container = f["container"]
                second_group = f[second_name]
                cache = ResolvingDecodeCache(root=container, format="hdf5")
                decoded_copy = Serializable.decode(
                    second_group, format="hdf5", decode_cache=cache
                )

                assert isinstance(decoded_copy, MockSerializable)
                assert decoded_copy.name == "shared"
                assert decoded_copy.value == 42

    def test_resolve_finds_copy_node_not_just_source(self):
        """A `cache_type="copy"` node mints its own resolvable
        `source_cache_id` (for a later reference to the copy itself, not
        the original object it copies from) -- `.resolve()` must match a
        "copy" node by `source_cache_id`, not only a plain "source" node
        by `cache_id`.

        Two distinct `MockSerializable` instances with identical field
        values share one `_serial_hash`: the first becomes `cache_type="source"`,
        the second (a different instance, same content) becomes
        `cache_type="copy"`. A third reference to that same second instance
        is decoded directly, in isolation, forcing `.resolve()` to locate
        the "copy" node -- there is no plain "source" node for its cache_id
        anywhere in the structure.
        """
        first = MockSerializable(name="shared", value=1, data={})
        second = MockSerializable(name="shared", value=1, data={})

        encode_cache: dict = {}
        first_encoded = Serializable.encode(
            first, format="json", encode_cache=encode_cache
        )
        second_encoded = Serializable.encode(
            second, format="json", encode_cache=encode_cache
        )
        third_encoded = Serializable.encode(
            second, format="json", encode_cache=encode_cache
        )

        assert second_encoded["cache_type"] == "copy"
        assert third_encoded["cache_type"] == "reference"
        assert third_encoded["cache_id"] == second_encoded["source_cache_id"]

        root = {
            "first": first_encoded,
            "second": second_encoded,
            "third": third_encoded,
        }
        cache = ResolvingDecodeCache(root=root, format="json")
        decoded_third = Serializable.decode(
            third_encoded, format="json", decode_cache=cache
        )

        assert isinstance(decoded_third, MockSerializable)
        assert decoded_third.name == "shared"
        assert decoded_third.value == 1

    def test_missing_source_raises_error(self, make_temp_path):
        """Test that missing source raises RuntimeError."""
        # Create reference to non-existent source
        reference_encoded = {
            "encode_type": "Serializable",
            "module": "test_serializable",
            "class": "MockSerializable",
            "version": 2,
            "cache_type": "reference",
            "cache_id": 999,  # Non-existent source
        }

        root = {"ref": reference_encoded}
        cache = ResolvingDecodeCache(root=root, format="json")

        # Should raise RuntimeError about missing source
        with pytest.raises(RuntimeError, match="source object not available"):
            Serializable.decode(
                reference_encoded, format="json", decode_cache=cache
            )

    def test_collapsed_blob_source_resolution(self, make_temp_path):
        """Test that a `cache_type="source"` entry nested inside a real HDF5
        `"$collapsed"` blob dataset (not a real `h5py.Group`) can be resolved
        on demand by `ResolvingDecodeCache`, exercising
        `_find_source_in_collapsed_blob` specifically rather than the plain
        real-group scan.

        A shared, array-free `MockSerializable` referenced twice from one
        small outer object collapses per `_contains_no_array`
        (`loqs/internal/encoder/hdf5encoder.py`), so both its `"source"` and
        `"reference"` cache entries land inside one `"$collapsed"` dataset
        rather than real HDF5 groups. `.resolve()` is called directly against
        a fresh, empty cache (never decoding anything first), so the only way
        it can find the source is by actually scanning into that blob.
        """
        from loqs.internal.encoder.hdf5encoder import _COLLAPSED_BLOB_NAME
        import json as jsonlib

        with make_temp_path(suffix=".h5") as temp_path:
            shared = MockSerializable(name="shared", value=7, data={"a": 1})
            outer = MockSerializable(
                name="outer", value=1, data={"x": shared, "y": shared}
            )

            with h5py.File(temp_path, "w") as f:
                outer.dump(f, format="hdf5")

            with h5py.File(temp_path, "r") as f:
                outer_group = f["root"][next(iter(f["root"].keys()))]

                # Confirm the test setup actually produced a collapsed blob
                # (proves the scenario below, not incidental).
                assert _COLLAPSED_BLOB_NAME in outer_group

                # Read the shared object's real cache_id directly off the
                # blob's own JSON content, rather than hardcoding it.
                raw = outer_group[_COLLAPSED_BLOB_NAME][()].tobytes().decode(
                    "utf-8"
                )
                blob = jsonlib.loads(raw)
                items = blob["data"]["items"]
                source_entries = [
                    v
                    for v in items.values()
                    if isinstance(v, dict) and v.get("cache_type") == "source"
                ]
                assert len(source_entries) == 1
                shared_cache_id = source_entries[0]["cache_id"]

                # Fresh, empty cache -- nothing decoded through it yet.
                cache = ResolvingDecodeCache(root=f["root"], format="hdf5")

                call_count = [0]
                original = cache._find_source_in_collapsed_blob

                def spy(*args, **kwargs):
                    call_count[0] += 1
                    return original(*args, **kwargs)

                cache._find_source_in_collapsed_blob = spy

                resolved = cache.resolve(shared_cache_id)

                assert call_count[0] > 0, (
                    "_find_source_in_collapsed_blob was not called; the "
                    "collapsed-blob scanning code path was not exercised"
                )
                assert isinstance(resolved, MockSerializable)
                assert resolved.name == "shared"
                assert resolved.value == 7
                assert resolved.data == {"a": 1}

    def test_programresults_decode_cache_repointing(self, make_temp_path):
        """Regression: repeated lazy per-shot loads via `get_shot_history`
        (`_load_shot_from_checkpoint` -> `_load_shot_from_single_file`) reuse
        one persistent `_checkpoint_decode_cache` across separate calls, each
        of which opens and closes its own fresh `h5py.File` handle -- this is
        the exact scenario the original bug was found in (a shared object
        referenced across two lazily-loaded shots).

        Two shots share one cacheable `Frame` (`_CACHE_ON_SERIALIZE`), so
        checkpointing produces one `"source"` + one `"reference"`/`"copy"`
        entry across them. Loading shot 1 (whichever shot's own checkpoint
        entry holds the reference, not the source) via a fresh
        `ProgramResults` with `lazy_loading=True` (the default) must resolve
        the shared `Frame` on demand through the re-pointed cache, not raise.
        """
        from loqs.core import ProgramResults
        from loqs.core.history import History
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"

            writer = ProgramResults(lazy_loading=False)
            shared_frame = Frame({"shared_data": "common_value"})

            history0 = History()
            history0.append(shared_frame)
            history0.append(Frame({"shot": 0}))
            history1 = History()
            history1.append(shared_frame)
            history1.append(Frame({"shot": 1}))

            writer.add_shot(0, history0)
            writer.add_shot(1, history1)
            writer.checkpoint(checkpoint_dir=checkpoint_dir)
            assert (checkpoint_dir / "results.h5").exists()

            # Lazy consumer: each get_shot_history call below opens and
            # closes its own fresh h5py.File via _load_shot_from_single_file,
            # reusing this instance's one persistent _checkpoint_decode_cache.
            reader = ProgramResults(lazy_loading=True)
            reader._checkpoint_dir = checkpoint_dir

            shot0 = reader.get_shot_history(0)
            shot1 = reader.get_shot_history(1)

            assert shot0 is not None and shot1 is not None
            assert shot0[0]["shared_data"] == "common_value"
            assert shot1[0]["shared_data"] == "common_value"
            assert shot0[1]["shot"] == 0
            assert shot1[1]["shot"] == 1

            # The persistent per-instance cache must actually have been used
            # (and re-pointed across the two separate file opens above),
            # not bypassed.
            assert isinstance(reader._checkpoint_decode_cache, ResolvingDecodeCache)

    def test_strict_validators_with_qeccodepatch(self, make_temp_path):
        """Regression: QECCodePatch strict validators don't leak DeferredRef.

        QECCodePatch._from_decoded_attrs has strict type checking that raises
        ValueError if 'code' is not a real QECCode instance. This test verifies
        that ResolvingDecodeCache resolves the code before passing it to the
        validator, avoiding a DeferredRef reaching the constructor.

        The test constructs a reference-before-source scenario where a shared
        QECCode is referenced by two QECCodePatches, with the first reference
        reaching the constructor before the code's source is decoded.
        """
        from loqs.core.qeccode import QECCode
        from loqs.core.recordables.qeccodepatch import QECCodePatch
        from loqs.core.instructions import Instruction

        # Build a minimal QECCode fixture matching test_qeccode.py pattern
        def apply_fn(state, qubits):
            return Frame({"state": state + 1, "qubits": qubits})

        def map_qubits_fn(qubit_mapping, qubits, **kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs["qubits"] = [qubit_mapping[q] for q in qubits]
            return new_kwargs

        ins_data = {"qubits": ["Q0", "Q1"]}
        ins = Instruction(apply_fn, ins_data, map_qubits_fn, name="test")
        code = QECCode({"test_ins": ins}, ["Q0", "Q1"], ["Q0"], "Test code")

        # Create a shared code and two patches that reference it
        patch1 = QECCodePatch(code=code, qubits=["D0", "A0"], pauli_frame="II")
        patch2 = QECCodePatch(code=code, qubits=["D1", "A1"], pauli_frame="II")

        # Encode both patches to JSON (shared code becomes source+reference)
        cache = {}
        patch1_encoded = Serializable.encode(
            patch1, format="json", encode_cache=cache, reset_encode_id=True
        )
        patch2_encoded = Serializable.encode(
            patch2, format="json", encode_cache=cache
        )

        # Construct a root where patch2's reference appears first
        # (forcing decode to resolve the code on-demand)
        root = {"patch2": patch2_encoded, "patch1": patch1_encoded}

        # Decode patch2 first through ResolvingDecodeCache
        # (its code reference should trigger resolution of the code source)
        decode_cache = ResolvingDecodeCache(root=root, format="json")
        decoded_patch2 = Serializable.decode(
            patch2_encoded, format="json", decode_cache=decode_cache
        )

        # Verify: patch2 was decoded successfully without DeferredRef leaking
        assert isinstance(decoded_patch2, QECCodePatch)
        assert not isinstance(decoded_patch2.code, DeferredRef)
        assert isinstance(decoded_patch2.code, QECCode)
        assert decoded_patch2.code.name == "Test code"
        assert decoded_patch2.qubits == ["D1", "A1"]

    def test_strict_validators_with_instructionlabel_and_instruction(self, make_temp_path):
        """Regression: InstructionLabel doesn't leak unresolved Instruction refs.

        InstructionLabel is not itself a Serializable, but can contain
        Serializable Instruction objects. This test verifies that when an
        Instruction is shared and referenced before its source is decoded,
        the ResolvingDecodeCache correctly resolves it before it reaches
        InstructionLabel's own constructor (which has runtime type checks).
        """
        from loqs.core.instructions import Instruction, InstructionLabel

        # Build a minimal Instruction
        def apply_fn():
            return Frame({"result": "test"})

        ins = Instruction(apply_fn, name="TestInstruction")

        # Create two InstructionLabels that reference the same Instruction
        label1 = InstructionLabel(ins, patch_label="L0")
        label2 = InstructionLabel(ins, patch_label="L1")

        # Encode both labels to JSON (shared instruction becomes source+reference)
        cache = {}
        label1_encoded = Serializable.encode(
            label1, format="json", encode_cache=cache, reset_encode_id=True
        )
        label2_encoded = Serializable.encode(
            label2, format="json", encode_cache=cache
        )

        # Construct a root where label2's reference appears first
        root = {"label2": label2_encoded, "label1": label1_encoded}

        # Decode label2 first through ResolvingDecodeCache
        decode_cache = ResolvingDecodeCache(root=root, format="json")
        decoded_label2 = Serializable.decode(
            label2_encoded, format="json", decode_cache=decode_cache
        )

        # Verify: label2 was decoded as a dict (InstructionLabel is a dict subclass),
        # and its instruction is not a DeferredRef
        assert isinstance(decoded_label2, dict)
        assert decoded_label2["patch_label"] == "L1"
        # The instruction should be a real Instruction, not a DeferredRef
        instruction = decoded_label2["instruction"]
        assert isinstance(instruction, Instruction)
        assert not isinstance(instruction, DeferredRef)
        assert instruction.name == "TestInstruction"
