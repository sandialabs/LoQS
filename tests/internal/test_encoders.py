"""Tests for the JSONEncoder and HDF5Encoder classes."""

import numpy as np
import scipy.sparse as sps
import pytest
import h5py

from loqs.internal.encoder.jsonencoder import JSONEncoder
from loqs.internal.encoder.hdf5encoder import (
    HDF5Encoder,
    _COLLAPSED_BLOB_NAME,
    _GZIP_MIN_BYTES,
    _encode_collapsed_children,
)
from loqs.internal.serializable import Serializable
from loqs.types import NDArray


class MockSerializable(Serializable):
    """A simple Serializable class for testing encoders."""

    _CACHE_ON_SERIALIZE = True
    _SERIALIZE_ATTRS = ["name", "value"]

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
    def _from_decoded_attrs(cls, attr_dict):
        return cls(name=attr_dict["name"], value=attr_dict["value"])


@pytest.fixture(params=["json", "hdf5"])
def encoder_format(request):
    """Parameterized fixture for testing both JSON and HDF5 encoders."""
    return request.param


class TestEncoderParameterized:
    """Parameterized tests for both JSON and HDF5 encoders."""

    def test_encoded_layout(self, encoder_format, make_temp_path):
        obj = MockSerializable(name="test_obj", value=123)
        cache_id = 123

        if encoder_format == "json":
            encoded = JSONEncoder.encode_uncached_obj(obj)
            assert encoded["encode_type"] == "Serializable"
            assert encoded["module"] == "test_encoders"
            assert encoded["class"] == "MockSerializable"
            assert "version" in encoded

            cached = JSONEncoder.encode_cached_obj(cache_id)
            assert cached["cache_type"] == "reference"
            assert cached["cache_id"] == cache_id
        else:
            with make_temp_path(suffix=".h5") as temp_file:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    group = HDF5Encoder.encode_uncached_obj(
                        obj, h5_group=root_group
                    )
                    assert group.attrs["encode_type"] == "Serializable"
                    assert group.attrs["module"] == "test_encoders"
                    assert group.attrs["class"] == "MockSerializable"
                    # Only the file's root group carries a version.
                    assert "version" not in group.attrs

                    cached = HDF5Encoder.encode_cached_obj(
                        cache_id, h5_group=root_group
                    )
                    assert cached.attrs["cache_type"] == "reference"
                    assert cached.attrs["cache_id"] == cache_id

    def test_class_round_trip(self, encoder_format, make_temp_path):
        if encoder_format == "json":
            encoded = JSONEncoder.encode_class(MockSerializable)
            decoded = JSONEncoder.decode_class(encoded)
        else:
            with make_temp_path(suffix=".h5") as temp_file:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_class(
                        MockSerializable, h5_group=root_group
                    )

                with h5py.File(temp_file, "r") as h5_file:
                    decoded = HDF5Encoder.decode_class(h5_file["root"])  # type: ignore

        assert decoded is MockSerializable

    @pytest.mark.parametrize(
        "value", [42, 3.14, "hello", True, None, "Café & naïve", "🎉🎊🎈"]
    )
    def test_primitive_round_trip(self, encoder_format, make_temp_path, value):
        if encoder_format == "json":
            decoded = JSONEncoder.decode_primitive(
                JSONEncoder.encode_primitive(value)  # type: ignore
            )
        else:
            with make_temp_path(suffix=".h5") as temp_file:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")
                    HDF5Encoder.encode_primitive(value, h5_group=root_group)

                with h5py.File(temp_file, "r") as h5_file:
                    decoded = HDF5Encoder.decode_primitive(h5_file["root"])  # type: ignore

        assert decoded == value
        if value is True:
            assert type(decoded) is bool


class TestComprehensiveArrayEncoding:
    """Round trips for dense, complex and sparse arrays."""

    def test_array_roundtrip_all_types(self, encoder_format, make_temp_path):
        test_arrays = [
            np.array([[1.5, 2.5], [3.5, 4.5]]),
            np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]),
            sps.csr_matrix([[0, 1, 0], [2, 0, 3], [0, 4, 0]]),
        ]

        if encoder_format == "json":
            for arr in test_arrays:
                decoded = JSONEncoder.decode_array(
                    JSONEncoder.encode_array(arr)
                )

                if sps.issparse(arr):
                    assert sps.issparse(decoded)
                    assert np.allclose(decoded.toarray(), arr.toarray())
                else:
                    assert isinstance(decoded, NDArray)
                    assert np.allclose(decoded, arr)

        else:
            with make_temp_path(suffix=".h5") as temp_file:
                for arr in test_arrays:
                    with h5py.File(temp_file, "w") as h5_file:
                        root_group = h5_file.create_group("root")
                        HDF5Encoder.encode_array(arr, h5_group=root_group)

                    with h5py.File(temp_file, "r") as h5_file:
                        root_group = h5_file["root"]
                        assert isinstance(root_group, h5py.Group)
                        decoded = HDF5Encoder.decode_array(root_group)

                    if sps.issparse(arr):
                        assert sps.issparse(decoded)
                        assert np.allclose(decoded.toarray(), arr.toarray())
                    else:
                        assert isinstance(decoded, np.ndarray)
                        assert np.allclose(decoded, arr)


_MIXED_LIST = [1, "string", 3.14, MockSerializable(name="test", value=42)]


class TestHDFIterableOptimizations:
    @pytest.mark.parametrize(
        "test_list,storage_format,types",
        [
            pytest.param([1, 2, 3], "dataset", None, id="ints"),
            pytest.param([1.1, 2.2, 3.3], "dataset", None, id="floats"),
            pytest.param(
                [True, False, True], "dataset", [bool] * 3, id="bools"
            ),
            pytest.param(
                ["hello", "world", "test"], "dataset", None, id="strs"
            ),
            pytest.param(
                [b"bytes1", b"bytes2", b"bytes3"], "dataset", None, id="bytes"
            ),
            pytest.param(
                list(range(1500)), "dataset", None, id="large_compressed"
            ),
            pytest.param(
                [0, True, False, 1],
                "groups",
                [int, bool, bool, int],
                id="int_bool_mix",
            ),
            pytest.param(_MIXED_LIST, "groups", None, id="mixed"),
            pytest.param((1, 2, 3), "dataset", None, id="tuple"),
            pytest.param({1, 2, 3}, "dataset", None, id="set"),
        ],
    )
    def test_hdf5_iterable_round_trip(
        self, make_temp_path, test_list, storage_format, types
    ):
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                root_group = h5_file.create_group("root")
                encoded_group = HDF5Encoder.encode_iterable(
                    test_list, h5_group=root_group
                )
                assert encoded_group.attrs["storage_format"] == storage_format
                if storage_format == "dataset":
                    data_dataset = encoded_group["data"]
                    assert isinstance(data_dataset, h5py.Dataset)
                    if len(test_list) > 1000:
                        assert data_dataset.compression is not None

            with h5py.File(temp_file, "r") as h5_file:
                decoded = HDF5Encoder.decode_iterable(h5_file["root"])  # type: ignore

        assert type(decoded) is type(test_list)
        assert decoded == test_list
        if types is not None:
            # `True == 1`, so check the exact types too.
            assert [type(x) for x in decoded] == types


class MockSerializableWithArray(Serializable):
    """A Serializable class with one array-free attr and one array attr, for testing HDF5's array-free-subtree collapse."""

    _CACHE_ON_SERIALIZE = True
    _SERIALIZE_ATTRS = ["name", "data"]

    def __init__(self, name="test", data=None):
        self.name = name
        self.data = data if data is not None else np.array([1, 2, 3])

    def __eq__(self, other):
        return (
            isinstance(other, MockSerializableWithArray)
            and self.name == other.name
            and np.array_equal(self.data, other.data)
        )

    @classmethod
    def _from_decoded_attrs(cls, attr_dict):
        return cls(name=attr_dict["name"], data=attr_dict["data"])


class TestHDF5Collapse:
    """Regression tests for HDF5's array-free-subtree collapse: a
    Serializable's attrs, a dict's values, or an iterable's elements with
    no array anywhere below them are batched into one shared blob dataset
    instead of getting their own HDF5 group each."""

    @pytest.mark.parametrize("case", ["array_free", "with_array"])
    def test_collapse_partition(self, make_temp_path, case):
        if case == "array_free":
            obj = MockSerializable(name="collapsed", value=99)
        else:
            obj = MockSerializableWithArray(
                name="mixed", data=np.array([1.0, 2.0, 3.0])
            )

        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                root_group = h5_file.create_group("root")
                HDF5Encoder.encode_uncached_obj(obj, h5_group=root_group)

            with h5py.File(temp_file, "r") as h5_file:
                root_group = h5_file["root"]
                obj_group = root_group[list(root_group.keys())[0]]
                assert isinstance(
                    obj_group[_COLLAPSED_BLOB_NAME], h5py.Dataset
                )
                if case == "array_free":
                    # Both attrs are batched into one shared blob.
                    assert list(obj_group.keys()) == [_COLLAPSED_BLOB_NAME]
                else:
                    # The array keeps its own group; "name" joins the blob.
                    assert set(obj_group.keys()) == {
                        "data",
                        _COLLAPSED_BLOB_NAME,
                    }
                    assert isinstance(obj_group["data"], h5py.Group)

                decoded = HDF5Encoder.decode_uncached_obj(obj_group)  # type: ignore
                assert decoded == obj

    @pytest.mark.parametrize("case", ["copy", "reference"])
    def test_collapsed_cache_hits(self, make_temp_path, case):
        """A cache hit on an array-bearing source is a cheap stub, so it
        folds into the surrounding collapse blob; only the source gets a
        real group."""
        shared = MockSerializableWithArray(
            name="shared", data=np.array([1.0, 2.0])
        )
        if case == "copy":
            same_content_copy = MockSerializableWithArray(
                name="shared", data=np.array([1.0, 2.0])
            )
            elements = [shared, same_content_copy, "an array-free sibling"]
        else:
            elements = [shared, "an array-free sibling", shared]

        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                root_group = h5_file.create_group("root")
                HDF5Encoder.encode_iterable(
                    elements, h5_group=root_group, encode_cache={}
                )

            with h5py.File(temp_file, "r") as h5_file:
                root_group = h5_file["root"]
                list_group = root_group["iterable"]
                assert set(list_group.keys()) == {"0", _COLLAPSED_BLOB_NAME}

                decoded = HDF5Encoder.decode_iterable(
                    root_group, decode_cache={}  # type: ignore
                )
                assert decoded == elements
                if case == "reference":
                    assert decoded[0] is decoded[2]

    def test_collapsed_blob_respects_gzip_threshold(self, make_temp_path):
        """A collapsed blob is gzip-compressed once it reaches
        `_GZIP_MIN_BYTES`."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                large_group = h5_file.create_group("large")
                _encode_collapsed_children(
                    large_group,
                    {"x": "a" * _GZIP_MIN_BYTES},
                    encode_cache=None,
                    ignore_no_serialize_flags=False,
                )

            with h5py.File(temp_file, "r") as h5_file:
                assert (
                    h5_file["large"][_COLLAPSED_BLOB_NAME].compression
                    == "gzip"
                )
