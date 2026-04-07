"""Complete parameterized tests for JSONEncoder and HDF5Encoder classes."""

import inspect
import json
import tempfile
import os
import numpy as np
import scipy.sparse as sps
import pytest
import h5py

from loqs.internal.encoder.jsonencoder import JSONEncoder
from loqs.internal.encoder.hdf5encoder import HDF5Encoder
from loqs.internal.serializable import Serializable, SERIALIZATION_VERSION
from loqs.types import Int, Float


class MockSerializable(Serializable):
    """A simple Serializable class for testing encoders."""

    CACHE_ON_SERIALIZE = True
    SERIALIZE_ATTRS = ["name", "value"]

    def __init__(self, name="test", value=42):
        self.name = name
        self.value = value

    def __eq__(self, other):
        return (
            isinstance(other, MockSerializable)
            and self.name == other.name
            and self.value == other.value
        )

    @classmethod
    def from_decoded_attrs(cls, attr_dict):
        return cls(name=attr_dict["name"], value=attr_dict["value"])


@pytest.fixture(params=["json", "hdf5"])
def encoder_format(request):
    """Parameterized fixture for testing both JSON and HDF5 encoders."""
    return request.param


class TestEncoderParameterized:
    """Parameterized tests for both JSON and HDF5 encoders."""

    def test_encode_uncached_obj(self, encoder_format):
        """Test encoding of uncached Serializable objects."""
        obj = MockSerializable(name="test_obj", value=123)
        
        if encoder_format == "json":
            encoded = JSONEncoder.encode_uncached_obj(obj)
            assert isinstance(encoded, dict)
            assert encoded["encode_type"] == "Serializable"
            assert encoded["module"] == "test_encoders"
            assert encoded["class"] == "MockSerializable"
            assert "version" in encoded
            assert "name" in encoded
            assert "value" in encoded
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.ENCODE_ID = 1234
                    encoded_group = HDF5Encoder.encode_uncached_obj(obj, h5_group=root_group)
                    assert isinstance(encoded_group, h5py.Group)
                    assert encoded_group.name == "/root/Serializable_1234"
                    assert encoded_group.attrs["encode_type"] == "Serializable"
                    assert encoded_group.attrs["module"] == "test_encoders"
                    assert encoded_group.attrs["class"] == "MockSerializable"
                    assert "version" in encoded_group.attrs
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
    
    def test_decode_uncached_obj(self, encoder_format):
        """Test decoding of uncached Serializable objects."""
        obj = MockSerializable(name="decode_test", value=456)
        
        if encoder_format == "json":
            encoded = JSONEncoder.encode_uncached_obj(obj)
            decoded = JSONEncoder.decode_uncached_obj(encoded)
            assert isinstance(decoded, MockSerializable)
            assert decoded.name == "decode_test"
            assert decoded.value == 456
            assert decoded == obj
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_uncached_obj(obj, h5_group=root_group)
                
                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    assert isinstance(root_group, h5py.Group)
                    obj_group_name = list(root_group.keys())[0]
                    encoded_group = root_group[obj_group_name]
                    decoded = HDF5Encoder.decode_uncached_obj(encoded_group) # type: ignore
                    assert isinstance(decoded, MockSerializable)
                    assert decoded.name == "decode_test"
                    assert decoded.value == 456
                    assert decoded == obj
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_encode_cached_obj(self, encoder_format):
        """Test encoding of cached objects."""
        cache_id = 123
        
        if encoder_format == "json":
            encoded = JSONEncoder.encode_cached_obj(cache_id)
            assert isinstance(encoded, dict)
            assert encoded["cache_type"] == "reference"
            assert encoded["cache_id"] == cache_id
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    encoded_group = HDF5Encoder.encode_cached_obj(cache_id, h5_group=root_group)
                    assert isinstance(encoded_group, h5py.Group)
                    assert encoded_group.attrs["encode_type"] == "Serializable"
                    assert encoded_group.attrs["cache_type"] == "reference"
                    assert encoded_group.attrs["cache_id"] == cache_id
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_decode_cached_obj(self, encoder_format):
        """Test decoding of cached objects."""
        cache_id = 456
        obj = MockSerializable(name="cached", value=789)
        
        if encoder_format == "json":
            decode_cache: dict[int, Serializable] = {cache_id: obj}
            encoded = JSONEncoder.encode_cached_obj(cache_id)
            decoded = JSONEncoder.decode_cached_obj(encoded, decode_cache)
            assert decoded is obj
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                decode_cache = {cache_id: obj}
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_cached_obj(cache_id, h5_group=root_group)
                
                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    ref_group_name = list(root_group.keys())[0]
                    encoded_group = root_group[ref_group_name]
                    decoded = HDF5Encoder.decode_cached_obj(encoded_group, decode_cache)
                    assert decoded is obj
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_encode_iterable(self, encoder_format):
        """Test encoding of iterables (list, tuple, set)."""
        test_list = [1, 2, 3]

        # TODO: Test tuple and set versions as well
        
        if encoder_format == "json":
            encoded_list = JSONEncoder.encode_iterable(test_list)
            assert encoded_list["encode_type"] == "iterable"
            assert encoded_list["iterable_type"] == "list"
            assert len(encoded_list["items"]) == 3
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    encoded_group = HDF5Encoder.encode_iterable(test_list, h5_group=root_group)
                    assert isinstance(encoded_group, h5py.Group)
                    assert "iterable" in root_group
                    assert encoded_group.attrs["iterable_type"] == "list"
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_decode_iterable(self, encoder_format):
        """Test decoding of iterables."""
        test_list = [1, 2, 3]

        # TODO: Test tuple and set versions as well

        if encoder_format == "json":
            encoded_list = {
                "encode_type": "iterable",
            "version": SERIALIZATION_VERSION,
            "iterable_type": "list",
            "items": [
                    {"encode_type": "primitive", "value": 1, "version": SERIALIZATION_VERSION},
                    {"encode_type": "primitive", "value": 2, "version": SERIALIZATION_VERSION},
                    {"encode_type": "primitive", "value": 3, "version": SERIALIZATION_VERSION}
                ]
            }
            decoded_list = JSONEncoder.decode_iterable(encoded_list)
            assert isinstance(decoded_list, list)
            assert decoded_list == [1, 2, 3]

        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name

            try:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_iterable(test_list, h5_group=root_group)

                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    decoded = HDF5Encoder.decode_iterable(root_group) # type: ignore
                    assert isinstance(decoded, list)
                    assert decoded == [1, 2, 3]

            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_encode_iterable_hdf5_native_optimization(self):
        """Test that HDF5 encoder uses dataset optimization for native types."""
        # Only test HDF5 format for this optimization
        encoder_format = "hdf5"
        
        # Test with different native types
        test_cases = [
            ([1, 2, 3], "integers"),
            ([1.1, 2.2, 3.3], "floats"),
            ([True, False, True], "booleans"),
            (["hello", "world", "test"], "strings"),
            ([b"bytes1", b"bytes2", b"bytes3"], "bytes"),
        ]
        
        for test_list, description in test_cases:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name

            try:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    encoded_group = HDF5Encoder.encode_iterable(test_list, h5_group=root_group)
                    
                    # Check that it used dataset format
                    list_group = encoded_group
                    assert "storage_format" in list_group.attrs
                    assert list_group.attrs["storage_format"] == "dataset"
                    assert "data" in list_group
                    data_dataset = list_group["data"]
                    assert isinstance(data_dataset, h5py.Dataset)

                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    decoded = HDF5Encoder.decode_iterable(root_group) # type: ignore
                    assert isinstance(decoded, list)
                    assert decoded == test_list

            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_encode_iterable_mixed_types_fallback(self):
        """Test that HDF5 encoder falls back to groups for mixed types."""
        # Only test HDF5 format for this optimization
        encoder_format = "hdf5"
        
        # Test with mixed types that should fall back to groups
        test_list = [1, "string", 3.14, MockSerializable(name="test", value=42)]
        
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_file = f.name

        try:
            with h5py.File(temp_file, "w") as h5_file:
                root_group = h5_file.create_group("root")
                encoded_group = HDF5Encoder.encode_iterable(test_list, h5_group=root_group)
                
                # Check that it used groups format (fallback)
                list_group = encoded_group
                assert "storage_format" in list_group.attrs
                assert list_group.attrs["storage_format"] == "groups"

            with h5py.File(temp_file, "r") as h5_file:
                root_group = h5_file["root"]
                decoded = HDF5Encoder.decode_iterable(root_group) # type: ignore
                assert isinstance(decoded, list)
                assert len(decoded) == len(test_list)
                assert decoded[0] == 1
                assert decoded[1] == "string"
                assert decoded[2] == 3.14
                assert isinstance(decoded[3], MockSerializable)
                assert decoded[3].name == "test"
                assert decoded[3].value == 42

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_encode_iterable_large_dataset_compression(self):
        """Test that large datasets use compression."""
        # Only test HDF5 format for this optimization
        encoder_format = "hdf5"
        
        # Create a large list that should trigger compression
        large_list = list(range(1500))  # More than 1000 elements
        
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_file = f.name

        try:
            with h5py.File(temp_file, "w") as h5_file:
                root_group = h5_file.create_group("root")
                encoded_group = HDF5Encoder.encode_iterable(large_list, h5_group=root_group)
                
                # Check that it used dataset format with compression
                list_group = encoded_group
                assert "storage_format" in list_group.attrs
                assert list_group.attrs["storage_format"] == "dataset"
                assert "data" in list_group
                data_dataset = list_group["data"]
                assert isinstance(data_dataset, h5py.Dataset)
                # Check compression filter is applied
                assert data_dataset.compression is not None

            with h5py.File(temp_file, "r") as h5_file:
                root_group = h5_file["root"]
                decoded = HDF5Encoder.decode_iterable(root_group) # type: ignore
                assert isinstance(decoded, list)
                assert decoded == large_list

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_encode_dict(self, encoder_format):
        """Test encoding of dictionaries."""
        test_dict = {"key1": "value1", "key2": 42}
        
        if encoder_format == "json":
            encoded = JSONEncoder.encode_dict(test_dict)
            assert encoded["encode_type"] == "dict"
            assert "items" in encoded
            assert "key1" in encoded["items"]
            assert "key2" in encoded["items"]
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    encoded_group = HDF5Encoder.encode_dict(test_dict, h5_group=root_group)
                    assert isinstance(encoded_group, h5py.Group)
                    assert "keys" in encoded_group
                    assert "values" in encoded_group
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_decode_dict(self, encoder_format):
        """Test decoding of dictionaries."""
        test_dict = {"key1": "value1", "key2": 42}
        
        if encoder_format == "json":
            encoded = {
                "encode_type": "dict",
            "version": SERIALIZATION_VERSION,
            "items": {
                    "key1": {"encode_type": "primitive", "value": "value1", "version": SERIALIZATION_VERSION},
                    "key2": {"encode_type": "primitive", "value": 42, "version": SERIALIZATION_VERSION}
                }
            }
            decoded = JSONEncoder.decode_dict(encoded)
            assert isinstance(decoded, dict)
            assert decoded["key1"] == "value1"
            assert decoded["key2"] == 42
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_dict(test_dict, h5_group=root_group)
                
                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    dict_group = root_group["dict"]
                    decoded = HDF5Encoder.decode_dict(root_group)
                    assert isinstance(decoded, dict)
                    assert decoded["key1"] == "value1"
                    assert decoded["key2"] == 42
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_encode_class(self, encoder_format):
        """Test encoding of classes."""
        if encoder_format == "json":
            encoded = JSONEncoder.encode_class(MockSerializable)
            assert encoded["encode_type"] == "class"
            assert encoded["module"] == "test_encoders"
            assert encoded["class"] == "MockSerializable"
            assert "version" in encoded
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    encoded_group = HDF5Encoder.encode_class(MockSerializable, h5_group=root_group)
                    assert isinstance(encoded_group, h5py.Group)
                    assert encoded_group.attrs["module"] == "test_encoders"
                    assert encoded_group.attrs["class"] == "MockSerializable"
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_decode_class(self, encoder_format):
        """Test decoding of classes."""
        if encoder_format == "json":
            encoded = {
                "encode_type": "class",
                "module": "test_encoders",
                "class": "MockSerializable",
                "version": 1
            }
            decoded = JSONEncoder.decode_class(encoded)
            assert decoded is MockSerializable
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_class(MockSerializable, h5_group=root_group)
                
                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    decoded = HDF5Encoder.decode_class(root_group)
                    assert decoded is MockSerializable
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_encode_function(self, encoder_format):
        """Test encoding of functions."""
        def test_func(x):
            return x * 2
        
        if encoder_format == "json":
            encoded = JSONEncoder.encode_function(test_func)
            assert encoded["encode_type"] == "function"
            assert "source" in encoded
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    encoded_group = HDF5Encoder.encode_function(test_func, h5_group=root_group)
                    assert isinstance(encoded_group, h5py.Group)
                    assert "source" in encoded_group
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_decode_function(self, encoder_format):
        """Test decoding of functions."""
        def test_func(x):
            return x * 2
        
        if encoder_format == "json":
            source_code = "def test_func(x):\n    return x * 2"
            encoded = {
                "encode_type": "function",
                "version": SERIALIZATION_VERSION,
                "source": source_code
            }
            decoded = JSONEncoder.decode_function(encoded)
            # get_function_str returns source code as string
            # Cannot test function execution from source code
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_function(test_func, h5_group=root_group)
                
                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    decoded = HDF5Encoder.decode_function(root_group)
                    assert decoded(42) == 2*42
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_encode_primitive(self, encoder_format):
        """Test encoding of primitive types."""
        if encoder_format == "json":
            # Test JSON primitive encoding
            # Test int
            encoded_int = JSONEncoder.encode_primitive(42)
            assert encoded_int["encode_type"] == "primitive"
            assert encoded_int["value"] == 42
            
            # Test float
            encoded_float = JSONEncoder.encode_primitive(3.14)
            assert encoded_float["encode_type"] == "primitive"
            assert encoded_float["value"] == 3.14
            
            # Test string
            encoded_str = JSONEncoder.encode_primitive("hello")
            assert encoded_str["encode_type"] == "primitive"
            assert encoded_str["value"] == "hello"
            
            # Test boolean
            encoded_bool = JSONEncoder.encode_primitive(True)
            assert encoded_bool["encode_type"] == "primitive"
            assert encoded_bool["value"] == True
            
            # Test None
            encoded_none = JSONEncoder.encode_primitive(None)
            assert encoded_none["encode_type"] == "primitive"
            assert encoded_none["value"] is None
            
            # Test special characters
            encoded_special = JSONEncoder.encode_primitive("Café & naïve")
            assert encoded_special["encode_type"] == "primitive"
            assert encoded_special["value"] == "Café & naïve"
            
            # Test emoji
            encoded_emoji = JSONEncoder.encode_primitive("🎉🎊🎈")
            assert encoded_emoji["encode_type"] == "primitive"
            assert encoded_emoji["value"] == "🎉🎊🎈"
        
        else:  # hdf5
            # Test HDF5 primitive encoding
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")

                    # TODO: Correctness tests
                    
                    # Test int
                    encoded_group = HDF5Encoder.encode_primitive(42, h5_group=root_group)
                    assert isinstance(encoded_group, h5py.Group)
                    
                    # Test float
                    encoded_group = HDF5Encoder.encode_primitive(3.14, h5_group=root_group)
                    assert isinstance(encoded_group, h5py.Group)
                    
                    # Test string
                    encoded_group = HDF5Encoder.encode_primitive("hello", h5_group=root_group)
                    assert isinstance(encoded_group, h5py.Group)
                    
                    # Test boolean
                    encoded_group = HDF5Encoder.encode_primitive(True, h5_group=root_group)
                    assert isinstance(encoded_group, h5py.Group)
                    
                    # Test None
                    encoded_group = HDF5Encoder.encode_primitive(None, h5_group=root_group)
                    assert isinstance(encoded_group, h5py.Group)
                    
                    # Test special characters
                    encoded_group = HDF5Encoder.encode_primitive("Café & naïve", h5_group=root_group)
                    assert isinstance(encoded_group, h5py.Group)
                    
                    # Test emoji
                    encoded_group = HDF5Encoder.encode_primitive("🎉🎊🎈", h5_group=root_group)
                    assert isinstance(encoded_group, h5py.Group)
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_decode_primitive(self, encoder_format):
        """Test decoding of primitive types."""
        if encoder_format == "json":
            # Test JSON primitive decoding
            # Test int
            encoded_int = {"encode_type": "primitive", "value": 42, "version": SERIALIZATION_VERSION}
            decoded_int = JSONEncoder.decode_primitive(encoded_int)
            assert decoded_int == 42
            
            # Test float
            encoded_float = {"encode_type": "primitive", "value": 3.14, "version": SERIALIZATION_VERSION}
            decoded_float = JSONEncoder.decode_primitive(encoded_float)
            assert decoded_float == 3.14
            
            # Test string
            encoded_str = {"encode_type": "primitive", "value": "hello", "version": SERIALIZATION_VERSION}
            decoded_str = JSONEncoder.decode_primitive(encoded_str)
            assert decoded_str == "hello"
            
            # Test boolean
            encoded_bool = {"encode_type": "primitive", "value": True, "version": SERIALIZATION_VERSION}
            decoded_bool = JSONEncoder.decode_primitive(encoded_bool)
            assert decoded_bool == True
            
            # Test None
            encoded_none = {"encode_type": "primitive", "value": None, "version": SERIALIZATION_VERSION}
            decoded_none = JSONEncoder.decode_primitive(encoded_none)
            assert decoded_none is None
            
            # Test special characters
            encoded_special = {"encode_type": "primitive", "value": "Café & naïve", "version": SERIALIZATION_VERSION}
            decoded_special = JSONEncoder.decode_primitive(encoded_special)
            assert decoded_special == "Café & naïve"
            
            # Test emoji
            encoded_emoji = {"encode_type": "primitive", "value": "🎉🎊🎈", "version": SERIALIZATION_VERSION}
            decoded_emoji = JSONEncoder.decode_primitive(encoded_emoji)
            assert decoded_emoji == "🎉🎊🎈"
        
        else:  # hdf5
            # Test HDF5 primitive decoding
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                # Test int
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_primitive(42, h5_group=root_group)
                
                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    # HDF5 stores primitives as attributes directly, not in a subgroup
                    decoded = HDF5Encoder.decode_primitive(root_group)
                    assert decoded == 42
                
                # Test float
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_primitive(3.14, h5_group=root_group)
                
                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    # HDF5 stores primitives as attributes directly, not in a subgroup
                    decoded = HDF5Encoder.decode_primitive(root_group)
                    assert decoded == 3.14
                
                # Test string
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_primitive("hello", h5_group=root_group)
                
                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    # HDF5 stores primitives as attributes directly, not in a subgroup
                    decoded = HDF5Encoder.decode_primitive(root_group)
                    assert decoded == "hello"
                
                # Test boolean
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_primitive(True, h5_group=root_group)
                
                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    # HDF5 stores primitives as attributes directly, not in a subgroup
                    decoded = HDF5Encoder.decode_primitive(root_group)
                    assert decoded == True
                
                # Test special characters
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_primitive("Café & naïve", h5_group=root_group)
                
                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    # HDF5 stores primitives as attributes directly, not in a subgroup
                    decoded = HDF5Encoder.decode_primitive(root_group)
                    assert decoded == "Café & naïve"
                
                # Test emoji
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_primitive("🎉🎊🎈", h5_group=root_group)
                
                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    # HDF5 stores primitives as attributes directly, not in a subgroup
                    decoded = HDF5Encoder.decode_primitive(root_group)
                    assert decoded == "🎉🎊🎈"
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)


class TestEncoderIntegration:
    """Integration tests for both encoders."""

    def test_encoder_consistency(self):
        """Test that both encoders produce equivalent results for simple objects."""
        obj = MockSerializable(name="consistency_test", value=123)
        
        # Test JSON encoder
        json_encoded = JSONEncoder.encode_uncached_obj(obj)
        json_decoded = JSONEncoder.decode_uncached_obj(json_encoded)
        
        # Test HDF5 encoder
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_file = f.name
        
        try:
            with h5py.File(temp_file, "w") as h5_file:
                root_group = h5_file.create_group("root")
                hdf5_encoded_group = HDF5Encoder.encode_uncached_obj(obj, h5_group=root_group)
            
            with h5py.File(temp_file, "r") as h5_file:
                root_group = h5_file["root"]
                obj_group_name = list(root_group.keys())[0]
                encoded_group = root_group[obj_group_name]
                hdf5_decoded = HDF5Encoder.decode_uncached_obj(encoded_group)
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Both should decode to equivalent objects
        assert json_decoded == hdf5_decoded
        assert json_decoded == obj
        assert hdf5_decoded == obj

    def test_complex_object_roundtrip(self):
        """Test roundtrip serialization of complex objects with both encoders."""
        complex_obj = MockSerializable(
            name="complex_test",
            value=999
        )
        
        # Test JSON roundtrip
        json_encoded = JSONEncoder.encode_uncached_obj(complex_obj)
        json_decoded = JSONEncoder.decode_uncached_obj(json_encoded)
        
        assert json_decoded == complex_obj
        assert json_decoded.name == "complex_test"
        assert json_decoded.value == 999
        
        # Test HDF5 roundtrip
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_file = f.name
        
        try:
            with h5py.File(temp_file, "w") as h5_file:
                root_group = h5_file.create_group("root")
                HDF5Encoder.encode_uncached_obj(complex_obj, h5_group=root_group)
            
            with h5py.File(temp_file, "r") as h5_file:
                root_group = h5_file["root"]
                obj_group_name = list(root_group.keys())[0]
                encoded_group = root_group[obj_group_name]
                hdf5_decoded = HDF5Encoder.decode_uncached_obj(encoded_group)
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        assert hdf5_decoded == complex_obj
        assert hdf5_decoded.name == "complex_test"
        assert hdf5_decoded.value == 999


class TestComprehensiveArrayEncoding:
    """Comprehensive tests for array encoding covering all matrix types."""

    def test_encode_array_all_types(self, encoder_format):
        """Test encoding of all array types: dense_real, dense_complex, sparse_csr."""
        
        # Test cases for all matrix types
        test_cases = [
            ("dense_real_2d", np.array([[1.0, 2.0], [3.0, 4.0]])),
            ("dense_real_1d", np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
            ("dense_real_3d", np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])),
            ("dense_complex", np.array([[1+2j, 3+4j], [5+6j, 7+8j]])),
            ("sparse_csr", sps.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]])),
            ("sparse_csc", sps.csc_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])),
            ("sparse_coo", sps.coo_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]])),
        ]

        if encoder_format == "json":
            for name, arr in test_cases:
                encoded = JSONEncoder.encode_array(arr)
                assert encoded["encode_type"] == "array"
                assert encoded["shape"] == arr.shape
                
                if sps.issparse(arr):
                    assert encoded["sparse_matrix_type"] == "csr"
                else:
                    assert "data" in encoded
                
                print(f"✓ JSON {name}: {arr.shape}")
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                with h5py.File(temp_file, "w") as h5_file:
                    for i, (name, arr) in enumerate(test_cases):
                        group = h5_file.create_group(f"test_{i}")
                        encoded = HDF5Encoder.encode_array(arr, h5_group=group)
                        assert isinstance(encoded, h5py.Group)
                        
                        if sps.issparse(arr):
                            assert encoded.attrs["array_type"] == "sparse_csr"
                        else:
                            assert encoded.attrs["array_type"] in ["dense_real", "dense_complex"]
                        
                        print(f"✓ HDF5 {name}: {arr.shape}")
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_decode_array_all_types(self, encoder_format):
        """Test decoding of all array types."""
        
        # Test dense real array
        dense_real = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        if encoder_format == "json":
            # Dense real
            encoded = {
                "encode_type": "array",
                "data": [1.0, 2.0, 3.0, 4.0],
                "shape": (2, 2),
                "dtype": "float64",
                "version": SERIALIZATION_VERSION
            }
            decoded = JSONEncoder.decode_array(encoded)
            assert isinstance(decoded, np.ndarray)
            assert np.allclose(decoded, dense_real)
            print("✓ JSON dense_real decoding")
            
            # Dense complex
            dense_complex = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
            encoded_complex = {
                "encode_type": "array",
                "data": [(1+2j), (3+4j), (5+6j), (7+8j)],
                "shape": (2, 2),
                "dtype": "complex128",
                "version": SERIALIZATION_VERSION
            }
            decoded_complex = JSONEncoder.decode_array(encoded_complex)
            assert isinstance(decoded_complex, np.ndarray)
            assert np.allclose(decoded_complex, dense_complex)
            print("✓ JSON dense_complex decoding")
            
            # Sparse CSR
            sparse_csr = sps.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
            encoded_sparse = {
                "encode_type": "array",
                "sparse_matrix_type": "csr",
                "data": [1, 2, 3, 4, 5],
                "indices": [0, 2, 1, 0, 2],
                "indptr": [0, 2, 3, 5],
                "dtype": "int64",
                "shape": (3, 3),
                "version": SERIALIZATION_VERSION
            }
            decoded_sparse = JSONEncoder.decode_array(encoded_sparse)
            assert sps.issparse(decoded_sparse)
            assert np.allclose(decoded_sparse.toarray(), sparse_csr.toarray())
            print("✓ JSON sparse_csr decoding")
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                # Dense real
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_array(dense_real, h5_group=root_group)
                
                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    decoded = HDF5Encoder.decode_array(root_group)
                    assert isinstance(decoded, np.ndarray)
                    assert np.allclose(decoded, dense_real)
                print("✓ HDF5 dense_real decoding")
                
                # Sparse CSR
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_array(sps.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]]), h5_group=root_group)
                
                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    decoded = HDF5Encoder.decode_array(root_group)
                    assert sps.issparse(decoded)
                    assert np.allclose(decoded.toarray(), sps.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]]).toarray())
                print("✓ HDF5 sparse_csr decoding")
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_array_roundtrip_all_types(self, encoder_format):
        """Test encode->decode roundtrip for all array types."""
        
        test_arrays = [
            ("dense_real", np.array([[1.5, 2.5], [3.5, 4.5]])),
            ("dense_complex", np.array([[1+1j, 2+2j], [3+3j, 4+4j]])),
            ("sparse_csr", sps.csr_matrix([[0, 1, 0], [2, 0, 3], [0, 4, 0]]))
        ]

        if encoder_format == "json":
            for name, arr in test_arrays:
                # Encode
                encoded = JSONEncoder.encode_array(arr)
                
                # Decode
                decoded = JSONEncoder.decode_array(encoded)
                
                # Verify
                if sps.issparse(arr):
                    assert sps.issparse(decoded)
                    assert np.allclose(decoded.toarray(), arr.toarray())
                else:
                    assert isinstance(decoded, np.ndarray)
                    assert np.allclose(decoded, arr)
                
                print(f"✓ JSON {name} roundtrip")
        
        else:  # hdf5
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_file = f.name
            
            try:
                for name, arr in test_arrays:
                    # Encode
                    with h5py.File(temp_file, "w") as h5_file:
                        root_group = h5_file.create_group("root")
                        HDF5Encoder.encode_array(arr, h5_group=root_group)
                    
                    # Decode
                    with h5py.File(temp_file, "r") as h5_file:
                        root_group = h5_file["root"]
                        decoded = HDF5Encoder.decode_array(root_group)
                    
                    # Verify
                    if sps.issparse(arr):
                        assert sps.issparse(decoded)
                        assert np.allclose(decoded.toarray(), arr.toarray())
                    else:
                        assert isinstance(decoded, np.ndarray)
                        assert np.allclose(decoded, arr)
                    
                    print(f"✓ HDF5 {name} roundtrip")
            
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)