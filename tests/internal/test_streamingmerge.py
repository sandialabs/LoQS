"""Tests for streaming merge primitives for HDF5-backed dict attributes."""

import pytest
import h5py
from loqs.internal.streamingmerge import (
    merge_dict_attr,
    iter_dict_attr_entries,
    get_dict_attr_value,
    get_dict_attr_group,
)
from loqs.internal.serializable import Serializable


class MockSerializable(Serializable):
    """A simple Serializable class for testing."""

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


class TestMergeDictAttr:
    """Tests for merge_dict_attr function."""

    @pytest.mark.parametrize(
        "key_ds, value_ds, first, second",
        [
            pytest.param(
                True,
                False,
                [(5, "five"), (3, "three")],
                [(1, "one"), (4, "four"), (2, "two")],
                id="dataset_keys_groups_values",
            ),
            pytest.param(
                True,
                True,
                [(1, 10.5), (2, 20.5)],
                [(3, 30.5)],
                id="dataset_keys_dataset_values",
            ),
            pytest.param(
                False,
                True,
                [("a", 1), ("b", 2)],
                [("c", 3)],
                id="groups_keys_dataset_values",
            ),
            pytest.param(
                False,
                False,
                [
                    (MockSerializable("x", 1), MockSerializable("y", 2)),
                    (MockSerializable("p", 3), MockSerializable("q", 4)),
                ],
                [(MockSerializable("k", 5), MockSerializable("v", 6))],
                id="groups_keys_groups_values_serializable",
            ),
            pytest.param(
                False,
                False,
                [(1, "string"), (2, 3.14)],
                [(3, [1, 2, 3])],
                id="groups_mixed_values",
            ),
        ],
    )
    def test_merge_round_trip(
        self, make_temp_path, key_ds, value_ds, first, second
    ):
        """Create then append recovers every entry, in order."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                merge_dict_attr(
                    h5_file,
                    "test_dict",
                    first,
                    key_use_dataset=key_ds,
                    value_use_dataset=value_ds,
                )
                merge_dict_attr(h5_file, "test_dict", second, encode_cache={})

                result = list(iter_dict_attr_entries(h5_file, "test_dict"))
                assert result == first + second

    def test_mismatched_type_raises_error(self, make_temp_path):
        """A float key onto an int dataset is refused."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                merge_dict_attr(
                    h5_file,
                    "test_dict",
                    [(1, "a"), (2, "b")],
                    key_use_dataset=True,
                    value_use_dataset=False,
                )

                with pytest.raises(TypeError):
                    merge_dict_attr(
                        h5_file,
                        "test_dict",
                        [(3.5, "c")],
                        encode_cache={},
                    )

    def test_append_to_empty_dict(self, make_temp_path):
        """Test appending to an initially empty dict."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                merge_dict_attr(h5_file, "dict", [])

                merge_dict_attr(
                    h5_file,
                    "dict",
                    [(1, "a"), (2, "b")],
                    key_use_dataset=True,
                    value_use_dataset=False,
                )

                result = list(iter_dict_attr_entries(h5_file, "dict"))
                assert result == [(1, "a"), (2, "b")]

    @pytest.mark.parametrize("phase", ["fresh", "append"])
    def test_refused_value_leaves_dict_intact(self, make_temp_path, phase):
        """A refused value leaves the key and value datasets the same length."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                flags = {}
                if phase == "fresh":
                    entries = [(1, 10), (2, "bad")]
                    expected = 1
                    flags = {
                        "key_use_dataset": True,
                        "value_use_dataset": True,
                    }
                else:
                    merge_dict_attr(
                        h5_file,
                        "dict",
                        [(1, 10), (2, 20)],
                        key_use_dataset=True,
                        value_use_dataset=True,
                    )
                    entries = [(3, "bad")]
                    expected = 2

                with pytest.raises(TypeError):
                    merge_dict_attr(h5_file, "dict", entries, **flags)

            with h5py.File(temp_file, "r") as h5_file:
                dict_group = h5_file["dict"]["dict"]
                keys_dataset = dict_group["keys"]["iterable"]["data"]
                values_dataset = dict_group["values"]["iterable"]["data"]
                assert len(keys_dataset) == len(values_dataset) == expected

    def test_dataset_format_with_non_native_type_raises(self, make_temp_path):
        """Requesting dataset format for a non-native key type fails."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                with pytest.raises(TypeError):
                    merge_dict_attr(
                        h5_file,
                        "dict",
                        [(MockSerializable("obj", 1), "value")],
                        key_use_dataset=True,
                    )


class TestIterDictAttrEntriesStartIndex:
    """Tests for start_index parameter in iter_dict_attr_entries."""

    @pytest.mark.parametrize("fmt", ["groups", "dataset"])
    def test_start_index_skips_entries(self, make_temp_path, fmt):
        """start_index skips early entries and yields the rest."""
        if fmt == "groups":
            entries = [
                (i, MockSerializable(f"obj_{i}", i * 10)) for i in range(5)
            ]
            value_ds = False
        else:
            entries = [(i, i * 100) for i in range(5)]
            value_ds = True

        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                merge_dict_attr(
                    h5_file,
                    "test_dict",
                    entries,
                    key_use_dataset=True,
                    value_use_dataset=value_ds,
                )

            with h5py.File(temp_file, "r") as h5_file:
                result = list(
                    iter_dict_attr_entries(h5_file, "test_dict", start_index=2)
                )

        assert [k for k, _ in result] == [2, 3, 4]
        assert [v for _, v in result] == [v for _, v in entries[2:]]


class TestGetDictAttrValue:
    """Tests for get_dict_attr_value function."""

    @pytest.mark.parametrize("fmt", ["groups", "dataset"])
    def test_get_dict_attr_value(self, make_temp_path, fmt):
        """The value stored under a key is returned."""
        if fmt == "groups":
            entries = [
                (1, MockSerializable("obj_1", 100)),
                (2, MockSerializable("obj_2", 200)),
                (3, MockSerializable("obj_3", 300)),
            ]
            value_ds = False
        else:
            entries = [(1, 100.5), (2, 200.5), (3, 300.5)]
            value_ds = True

        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                merge_dict_attr(
                    h5_file,
                    "test_dict",
                    entries,
                    key_use_dataset=True,
                    value_use_dataset=value_ds,
                )

            with h5py.File(temp_file, "r") as h5_file:
                assert get_dict_attr_value(h5_file, "test_dict", 2) == (
                    entries[1][1]
                )
                if fmt == "groups":
                    with pytest.raises(KeyError):
                        get_dict_attr_value(h5_file, "test_dict", 999)


class TestGetDictAttrGroup:
    """Tests for get_dict_attr_group function."""

    def test_get_dict_attr_group(self, make_temp_path):
        """A groups-format key gives a Group; a dataset-format one is refused."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                merge_dict_attr(
                    h5_file,
                    "groups_dict",
                    [(1, MockSerializable("obj_1", 100))],
                    key_use_dataset=True,
                    value_use_dataset=False,
                )
                merge_dict_attr(
                    h5_file,
                    "dataset_dict",
                    [(1, 100), (2, 200)],
                    key_use_dataset=True,
                    value_use_dataset=True,
                )

            with h5py.File(temp_file, "r") as h5_file:
                group = get_dict_attr_group(h5_file, "groups_dict", 1)
                assert isinstance(group, h5py.Group)
                with pytest.raises(TypeError):
                    get_dict_attr_group(h5_file, "dataset_dict", 1)


class TestDecodeCacheForwardingKeysResolution:
    """Tests for decode_cache forwarding on key-side dict reads.

    Regression tests for the bug where keys-side reads of dict attributes
    were hardcoded with decode_cache=None, preventing correct reference
    resolution when Serializable objects are used as keys.
    """

    def test_shared_serializable_key_resolved_via_decode_cache(
        self, make_temp_path
    ):
        """A key and a later value sharing one object decode as one instance."""
        with make_temp_path(suffix=".h5") as temp_file:
            shared_key_obj = MockSerializable("shared_key", 1000)
            other_key_obj = MockSerializable("other_key", 2000)
            val1_obj = MockSerializable("val1", 3000)

            encode_cache = {}
            with h5py.File(temp_file, "w") as h5_file:
                merge_dict_attr(
                    h5_file,
                    "shared_dict",
                    [(shared_key_obj, val1_obj)],
                    encode_cache=encode_cache,
                    key_use_dataset=False,
                    value_use_dataset=False,
                )
                merge_dict_attr(
                    h5_file,
                    "shared_dict",
                    [(other_key_obj, shared_key_obj)],
                    encode_cache=encode_cache,
                    key_use_dataset=False,
                    value_use_dataset=False,
                )

            decode_cache = {}
            with h5py.File(temp_file, "r") as h5_file:
                entries = list(
                    iter_dict_attr_entries(
                        h5_file, "shared_dict", decode_cache=decode_cache
                    )
                )

            assert entries[0][0] is entries[1][1]
