"""Tests for streaming merge primitives for HDF5-backed dict attributes."""

import pytest
import h5py
import numpy as np
from loqs.internal.streamingmerge import (
    get_dict_attr_keys,
    merge_dict_attr,
    iter_dict_attr_entries,
    get_dict_attr_value,
    get_dict_attr_group,
)
from loqs.core.frame import Frame
from loqs.core.history import History
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

    @pytest.mark.parametrize(
        "first, key_ds, value_ds, refused, exc, good, expected",
        [
            pytest.param(
                [(1, "a"), (2, "b")],
                True,
                False,
                [(3.5, "c")],
                TypeError,
                [(4, "e")],
                [(1, "a"), (2, "b"), (4, "e")],
                id="append_dataset_key",
            ),
            pytest.param(
                None,
                True,
                False,
                [(MockSerializable("obj", 1), "a")],
                TypeError,
                [(4, "e")],
                [(4, "e")],
                id="fresh_first_entry",
            ),
            pytest.param(
                [(1, 10), (2, 20)],
                True,
                True,
                [(3, 2**70)],
                OverflowError,
                [(4, 40)],
                [(1, 10), (2, 20), (4, 40)],
                id="dataset_value_overflow",
            ),
            pytest.param(
                None,
                True,
                True,
                [(1, 10), (2, "bad")],
                TypeError,
                [(4, 40)],
                [(1, 10), (4, 40)],
                id="fresh_later_entry",
            ),
        ],
    )
    def test_refused_entry_leaves_keys_and_values_paired(
        self,
        make_temp_path,
        first,
        key_ds,
        value_ds,
        refused,
        exc,
        good,
        expected,
    ):
        """A refused entry leaves no orphan, so later entries keep their own
        values."""
        flags = {"key_use_dataset": key_ds, "value_use_dataset": value_ds}
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                if first is not None:
                    merge_dict_attr(h5_file, "dict", first, **flags)
                with pytest.raises(exc):
                    merge_dict_attr(h5_file, "dict", refused, **flags)
                merge_dict_attr(h5_file, "dict", good, **flags)

                result = list(iter_dict_attr_entries(h5_file, "dict"))
                assert result == expected

    @pytest.mark.parametrize(
        "first",
        [pytest.param([(1, 0)], id="append"), pytest.param([], id="fresh")],
    )
    def test_refused_groups_entry_leaves_no_cache_entries(
        self, make_temp_path, first
    ):
        """A retried entry reads back, as its refused try left no cache
        entries behind."""
        encode_cache: dict = {}
        h1 = History([Frame({"v": 1})])
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                if first:
                    merge_dict_attr(
                        h5_file,
                        "dict",
                        [(k, History([Frame({"v": v})])) for k, v in first],
                        encode_cache=encode_cache,
                    )
                with pytest.raises(ValueError):
                    merge_dict_attr(
                        h5_file,
                        "dict",
                        [(object(), h1)],
                        encode_cache=encode_cache,
                    )
                merge_dict_attr(
                    h5_file, "dict", [(4, h1)], encode_cache=encode_cache
                )

                result = [
                    (k, v[0]["v"])
                    for k, v in iter_dict_attr_entries(h5_file, "dict")
                ]
                assert result == first + [(4, 1)]

    @pytest.mark.parametrize(
        "first, fits, too_wide, wrong_type",
        [
            pytest.param("abcd", "αβ", "ααα", b"ab", id="str"),
            pytest.param(b"abcd", b"\xff\x00y", b"\xff" * 6, "ab", id="bytes"),
        ],
    )
    def test_string_side_append_measures_utf8_bytes(
        self, make_temp_path, first, fits, too_wide, wrong_type
    ):
        """A width check counts encoded bytes, and a side keeps its type."""
        flags = {"key_use_dataset": True, "value_use_dataset": True}
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                merge_dict_attr(h5_file, "dict", [(1, first)], **flags)
                merge_dict_attr(h5_file, "dict", [(2, fits)])
                with pytest.raises(ValueError):
                    merge_dict_attr(h5_file, "dict", [(3, too_wide)])
                with pytest.raises(TypeError):
                    merge_dict_attr(h5_file, "dict", [(3, wrong_type)])

                result = list(iter_dict_attr_entries(h5_file, "dict"))
        assert result == [(1, first), (2, fits)]
        assert [type(v) for _, v in result] == [type(first)] * 2

    def test_append_to_encoded_string_keys_keeps_their_type(
        self, make_temp_path
    ):
        """Appending to a dict the encoder wrote leaves its str keys str."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                parent = h5_file.create_group("parent")
                Serializable.encode(
                    {"a": np.arange(2), "b": np.arange(3)},
                    format="hdf5",
                    h5_group=parent.create_group("d"),
                )
                merge_dict_attr(parent, "d", [("γ", np.arange(1))])

                keys = list(get_dict_attr_keys(parent, "d"))
        assert keys == ["a", "b", "γ"]
        assert all(type(k) is str for k in keys)


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
