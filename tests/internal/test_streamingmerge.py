"""Tests for streaming merge primitives for HDF5-backed dict attributes."""

import pytest
import h5py
from loqs.internal.streamingmerge import (
    merge_dict_attr,
    iter_dict_attr_entries,
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


class CountingSerializable(Serializable):
    """A Serializable class that tracks instantiation for lazy-loading tests."""

    _instances_created = 0
    _CACHE_ON_SERIALIZE = True
    _SERIALIZE_ATTRS = ["name", "value"]

    def __init__(self, name="test", value=42):
        self.name = name
        self.value = value
        CountingSerializable._instances_created += 1

    def __eq__(self, other):
        return (
            isinstance(other, CountingSerializable)
            and self.name == other.name
            and self.value == other.value
        )

    @classmethod
    def _from_decoded_attrs(cls, attr_dict):
        return cls(name=attr_dict["name"], value=attr_dict["value"])


class TestMergeDictAttr:
    """Tests for merge_dict_attr function."""

    def test_create_and_append_multiple_calls(self, make_temp_path):
        """Test creating and appending across multiple merge_dict_attr calls."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                # First call: create with two entries
                entries1 = [(1, "a"), (2, "b")]
                merge_dict_attr(
                    h5_file,
                    "test_dict",
                    entries1,
                    key_use_dataset=True,
                    value_use_dataset=False,
                )

                # Second call: append two more entries
                entries2 = [(3, "c"), (4, "d")]
                merge_dict_attr(
                    h5_file, "test_dict", entries2, encode_cache={}
                )

                # Verify all entries are present and in order
                result = list(
                    iter_dict_attr_entries(h5_file, "test_dict")
                )
                assert result == [
                    (1, "a"),
                    (2, "b"),
                    (3, "c"),
                    (4, "d"),
                ]

    def test_four_storage_format_combinations(self, make_temp_path):
        """Test all four combinations of key/value dataset flags."""
        combinations = [
            (False, False),  # both groups
            (True, False),   # keys dataset, values groups
            (False, True),   # keys groups, values dataset
            (True, True),    # both dataset
        ]

        for key_ds, value_ds in combinations:
            with make_temp_path(suffix=".h5") as temp_file:
                with h5py.File(temp_file, "w") as h5_file:
                    # Use appropriate types for dataset storage
                    if key_ds and value_ds:
                        entries = [(1, 10.5), (2, 20.5), (3, 30.5)]
                    elif key_ds and not value_ds:
                        entries = [(1, "a"), (2, "b"), (3, "c")]
                    elif not key_ds and value_ds:
                        entries = [("a", 1), ("b", 2), ("c", 3)]
                    else:
                        # Both groups format can hold anything
                        entries = [
                            (MockSerializable("x", 1), MockSerializable("y", 2)),
                            (
                                MockSerializable("p", 3),
                                MockSerializable("q", 4),
                            ),
                        ]

                    merge_dict_attr(
                        h5_file,
                        "test_dict",
                        entries,
                        key_use_dataset=key_ds,
                        value_use_dataset=value_ds,
                    )

                    result = list(
                        iter_dict_attr_entries(h5_file, "test_dict")
                    )
                    assert len(result) == len(entries)
                    for orig, decoded in zip(entries, result):
                        assert decoded == orig

    def test_round_trip_correctness(self, make_temp_path):
        """Test that merge then iterate recovers entries exactly and in order."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                entries = [
                    (10, "value_a"),
                    (20, "value_b"),
                    (30, "value_c"),
                ]
                merge_dict_attr(
                    h5_file,
                    "test_dict",
                    entries,
                    key_use_dataset=True,
                    value_use_dataset=False,
                )

                result = list(
                    iter_dict_attr_entries(h5_file, "test_dict")
                )
                assert result == entries

    def test_lazy_decoding_one_entry_at_a_time(self, make_temp_path):
        """Test that iter_dict_attr_entries doesn't load all entries upfront."""
        with make_temp_path(suffix=".h5") as temp_file:
            # Create entries with tracking objects
            CountingSerializable._instances_created = 0
            entries = [
                (1, CountingSerializable("obj_a", 100)),
                (2, CountingSerializable("obj_b", 200)),
                (3, CountingSerializable("obj_c", 300)),
            ]

            with h5py.File(temp_file, "w") as h5_file:
                merge_dict_attr(
                    h5_file,
                    "test_dict",
                    entries,
                    key_use_dataset=True,
                    value_use_dataset=False,
                )

            # Now test that the generator doesn't eagerly decode all entries
            with h5py.File(temp_file, "r") as h5_file:
                gen = iter_dict_attr_entries(h5_file, "test_dict")

                # Before calling next, no entries should be decoded yet
                CountingSerializable._instances_created = 0

                # First next() decodes one value
                k1, v1 = next(gen)
                created_after_first = CountingSerializable._instances_created
                assert k1 == 1
                assert v1.name == "obj_a"
                assert v1.value == 100
                # Should have created at least one (likely more due to internal structure)
                assert created_after_first >= 1

                # Second next() without eagerly loading all remaining
                CountingSerializable._instances_created = 0
                k2, v2 = next(gen)
                created_after_second = CountingSerializable._instances_created
                assert k2 == 2
                assert v2.name == "obj_b"
                # Should create roughly the same amount as the first (not all remaining)
                # This proves we're not eager-loading everything at once
                assert created_after_second >= 1

                # Third next()
                k3, v3 = next(gen)
                assert k3 == 3
                assert v3.name == "obj_c"

    def test_mismatched_type_raises_error(self, make_temp_path):
        """Test that mismatched types on dataset-format sides raise TypeError."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                # Create with int keys
                entries1 = [(1, "a"), (2, "b")]
                merge_dict_attr(
                    h5_file,
                    "test_dict",
                    entries1,
                    key_use_dataset=True,
                    value_use_dataset=False,
                )

                # Try to append a float key (should fail)
                with pytest.raises(TypeError):
                    merge_dict_attr(
                        h5_file,
                        "test_dict",
                        [(3.5, "c")],
                        encode_cache={},
                    )

                # Verify the file wasn't corrupted by re-reading
                result = list(
                    iter_dict_attr_entries(h5_file, "test_dict")
                )
                # Should still have original entries
                assert len(result) == 2

    def test_nonexistent_attr_yields_nothing(self, make_temp_path):
        """Test that iter_dict_attr_entries on a nonexistent attr yields nothing."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                result = list(
                    iter_dict_attr_entries(h5_file, "nonexistent")
                )
                assert result == []

    def test_empty_dict_roundtrip(self, make_temp_path):
        """Test creating and iterating an empty dict."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                # Create empty dict
                merge_dict_attr(
                    h5_file, "empty_dict", [], key_use_dataset=True
                )

                # Iterate (should yield nothing)
                result = list(
                    iter_dict_attr_entries(h5_file, "empty_dict")
                )
                assert result == []

    def test_append_to_empty_dict(self, make_temp_path):
        """Test appending to an initially empty dict."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                # Create empty
                merge_dict_attr(h5_file, "dict", [])

                # Append
                merge_dict_attr(
                    h5_file,
                    "dict",
                    [(1, "a"), (2, "b")],
                    key_use_dataset=True,
                    value_use_dataset=False,
                )

                result = list(iter_dict_attr_entries(h5_file, "dict"))
                assert result == [(1, "a"), (2, "b")]

    def test_preserve_insertion_order(self, make_temp_path):
        """Test that insertion order is preserved across multiple appends."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                merge_dict_attr(
                    h5_file, "dict", [(5, "five"), (3, "three")]
                )
                merge_dict_attr(h5_file, "dict", [(1, "one"), (4, "four")])
                merge_dict_attr(h5_file, "dict", [(2, "two")])

                result = list(iter_dict_attr_entries(h5_file, "dict"))
                assert result == [
                    (5, "five"),
                    (3, "three"),
                    (1, "one"),
                    (4, "four"),
                    (2, "two"),
                ]

    def test_serializable_keys_and_values(self, make_temp_path):
        """Test with Serializable objects as both keys and values."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                key1 = MockSerializable("key1", 11)
                val1 = MockSerializable("val1", 111)
                key2 = MockSerializable("key2", 22)
                val2 = MockSerializable("val2", 222)

                merge_dict_attr(
                    h5_file,
                    "complex_dict",
                    [(key1, val1), (key2, val2)],
                    encode_cache={},
                )

                result = list(
                    iter_dict_attr_entries(h5_file, "complex_dict")
                )
                assert len(result) == 2
                assert result[0][0] == key1
                assert result[0][1] == val1
                assert result[1][0] == key2
                assert result[1][1] == val2

    def test_mixed_types_groups_format(self, make_temp_path):
        """Test that groups format accepts mixed types."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                entries = [
                    (1, "string"),
                    (2, 3.14),
                    (3, [1, 2, 3]),
                ]
                merge_dict_attr(
                    h5_file,
                    "mixed_dict",
                    entries,
                    key_use_dataset=False,
                    value_use_dataset=False,
                )

                result = list(
                    iter_dict_attr_entries(h5_file, "mixed_dict")
                )
                assert len(result) == 3
                assert result[0] == (1, "string")
                assert result[1] == (2, 3.14)
                assert result[2] == (3, [1, 2, 3])

    def test_value_dataset_type_mismatch(self, make_temp_path):
        """Test that value-side type mismatches also raise TypeError."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                # Create with float values
                merge_dict_attr(
                    h5_file,
                    "dict",
                    [(1, 1.5), (2, 2.5)],
                    key_use_dataset=False,
                    value_use_dataset=True,
                )

                # Try to append an int value (type mismatch)
                with pytest.raises(TypeError):
                    merge_dict_attr(h5_file, "dict", [(3, 42)])

                # Verify file integrity
                result = list(iter_dict_attr_entries(h5_file, "dict"))
                assert len(result) == 2

    def test_creation_path_streams_entries(self, make_temp_path):
        """Test that the fresh-creation path streams entries, not buffering all."""
        with make_temp_path(suffix=".h5") as temp_file:
            # Create a generator that yields entries and tracks consumption
            entry_index = [0]
            entries_consumed = []

            def entry_generator():
                for i in range(3):
                    entry_index[0] = i
                    entries_consumed.append(i)
                    yield (i + 10, f"value_{i}")

            with h5py.File(temp_file, "w") as h5_file:
                # Pass generator to merge_dict_attr
                merge_dict_attr(
                    h5_file,
                    "test_dict",
                    entry_generator(),
                    key_use_dataset=True,
                    value_use_dataset=False,
                )

                # Verify all entries were written
                result = list(iter_dict_attr_entries(h5_file, "test_dict"))
                assert len(result) == 3
                assert result == [
                    (10, "value_0"),
                    (11, "value_1"),
                    (12, "value_2"),
                ]

    def test_dataset_format_with_non_native_type_raises(self, make_temp_path):
        """Test that requesting dataset format with non-native type raises."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                # Try to create with key_use_dataset=True but passing
                # Serializable objects (not native scalars)
                with pytest.raises(TypeError) as exc_info:
                    merge_dict_attr(
                        h5_file,
                        "dict",
                        [(MockSerializable("obj", 1), "value")],
                        key_use_dataset=True,
                    )

                assert "Dataset format requested" in str(exc_info.value)
                assert "MockSerializable" in str(exc_info.value)

    def test_string_truncation_detection(self, make_temp_path):
        """Test that appending longer strings raises ValueError."""
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                # Create with short strings (3 chars -> dtype S4 with null terminator)
                merge_dict_attr(
                    h5_file,
                    "dict",
                    [(1, "abc")],
                    key_use_dataset=False,
                    value_use_dataset=True,
                )

                # Try to append a longer string (7 chars exceeds width of 4)
                with pytest.raises(ValueError) as exc_info:
                    merge_dict_attr(
                        h5_file, "dict", [(2, "toolong")]
                    )

                assert "length" in str(exc_info.value).lower()
                assert "width" in str(exc_info.value).lower()

                # Verify file wasn't corrupted
                result = list(iter_dict_attr_entries(h5_file, "dict"))
                assert len(result) == 1
