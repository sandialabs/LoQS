#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Streaming append and iteration primitives for HDF5-backed dict attributes.

This module provides functions to incrementally append entries to, and lazily
iterate over, HDF5-encoded dictionary structures without materializing the entire
dictionary in memory -- critical for checkpoint consolidation and similar
streaming operations over large, multi-entry dictionaries.
"""

from __future__ import annotations

import h5py
import numpy as np
from typing import Any, Iterable, Iterator

from loqs.internal.encoder.hdf5encoder import HDF5Encoder
from loqs.internal.serializable import Serializable


def merge_dict_attr(
    parent_group: h5py.Group,
    attr_name: str,
    entries: Iterable[tuple[Any, Any]],
    encode_cache: dict | None = None,
    key_use_dataset: bool = False,
    value_use_dataset: bool = False,
) -> None:
    """Append entries into a dict-shaped HDF5 attribute, one at a time.

    Creates or extends `parent_group[attr_name]` as a dict-shaped HDF5
    structure (matching `HDF5Encoder.encode_dict`'s layout), appending
    new (key, value) pairs without materializing more than one pair in
    memory at a time -- even on the very first call that creates the
    attribute.

    If the attribute doesn't exist yet, it is created fresh with the
    storage format determined by `key_use_dataset` and `value_use_dataset`.
    If it already exists, those flags are ignored; the existing storage
    format (recorded on each side's `"iterable"` subgroup) is used instead.

    For dataset-format sides (homogeneous native scalar types), appending
    reuses `HDF5Encoder._encode_iterable_dataset` directly. If a new entry's
    key or value doesn't match the dataset's established native type, a
    `TypeError` is raised and no partial write completes.

    For dataset-format string/bytes sides: the dataset's fixed-width dtype
    is determined from the first entry's length. Attempting to append a
    later string/bytes value longer than that initial width will raise
    `ValueError` rather than silently truncating.

    For groups-format sides (arbitrary or Serializable-typed entries), a new
    indexed subgroup is created and the entry is encoded into it via
    `Serializable.encode`.

    Parameters
    ----------
    parent_group : h5py.Group
        The HDF5 group to hold the dict attribute.
    attr_name : str
        Name of the dict attribute to create or extend.
    entries : Iterable[tuple[Any, Any]]
        An iterable of (key, value) pairs to append.
    encode_cache : dict | None, optional
        Cache for encoding operations (passed to `Serializable.encode`).
        Enables reference tracking across multiple append calls.
    key_use_dataset : bool, optional
        Whether to use dataset storage format for keys (only consulted if
        creating the attribute fresh). Default is False (groups format).
        If True and the first key entry is not a native scalar type, raises
        `TypeError`.
    value_use_dataset : bool, optional
        Whether to use dataset storage format for values (only consulted if
        creating the attribute fresh). Default is False (groups format).
        If True and the first value entry is not a native scalar type, raises
        `TypeError`.

    Raises
    ------
    TypeError
        If a key or value is appended to a dataset-format side and doesn't
        match the established native type of that dataset. Also raised if
        `key_use_dataset=True` or `value_use_dataset=True` is requested on
        fresh creation but the first entry's type doesn't support dataset
        storage (e.g., a Serializable object when dataset was explicitly
        requested).
    ValueError
        If a string or bytes value is appended to a dataset-format string/
        bytes column and the value is longer than the fixed width of the
        dataset (determined from the first entry).
    """
    if attr_name not in parent_group:
        # Create the dict structure fresh with streaming appends
        _stream_into_new_dict_attr(
            parent_group,
            attr_name,
            entries,
            encode_cache,
            key_use_dataset,
            value_use_dataset,
        )
    else:
        # Extend the existing dict structure
        _stream_into_existing_dict_attr(
            parent_group, attr_name, entries, encode_cache
        )


def _resolve_dict_target_group(
    parent_group: h5py.Group, attr_name: str
) -> h5py.Group:
    """Find the group containing attr_name, navigating single-child root wrappers if needed."""
    if attr_name in parent_group:
        return parent_group
    current = parent_group
    while len(current.keys()) == 1:
        first_child = current[next(iter(current.keys()))]
        if isinstance(first_child, h5py.Group):
            if attr_name in first_child:
                return first_child
            current = first_child
        else:
            break
    return parent_group


def iter_dict_attr_entries(
    parent_group: h5py.Group,
    attr_name: str,
    decode_cache: dict | None = None,
    start_index: int = 0,
) -> Iterator[tuple[Any, Any]]:
    """Lazily iterate over (key, value) pairs from a dict-shaped HDF5 attribute.

    Yields pairs one at a time from `parent_group[attr_name]`'s dict structure
    (matching `HDF5Encoder.encode_dict`'s layout), without materializing the
    entire dict in memory.

    For dataset-format sides, the entire side is read in one shot (native
    scalars are cheap). For groups-format sides, each entry is decoded
    individually via `Serializable.decode`, so memory is bounded to one
    entry at a time (critical for large Serializable objects like History).

    If the attribute doesn't exist, yields nothing (empty iterator).

    Parameters
    ----------
    parent_group : h5py.Group
        The HDF5 group holding the dict attribute.
    attr_name : str
        Name of the dict attribute to iterate over.
    decode_cache : dict | None, optional
        Cache for decoding operations (passed to `Serializable.decode`).
        Enables reference tracking across multiple entries.
    start_index : int, optional
        Starting position for iteration (0-based). For groups-format sides,
        entries below start_index are skipped without decoding. For dataset-
        format sides, entries are sliced from start_index to the end.
        Default is 0 (start from the beginning).

    Yields
    ------
    tuple[Any, Any]
        (key, value) pairs in insertion order, starting from start_index.
    """
    parent_group = _resolve_dict_target_group(parent_group, attr_name)
    if attr_name not in parent_group:
        return

    dict_group = parent_group[attr_name]
    if "dict" not in dict_group:
        return

    dict_subgroup = dict_group["dict"]
    if "keys" not in dict_subgroup or "values" not in dict_subgroup:
        return

    keys_group = dict_subgroup["keys"]
    values_group = dict_subgroup["values"]

    if "iterable" not in keys_group or "iterable" not in values_group:
        return

    keys_iterable_group = keys_group["iterable"]
    values_iterable_group = values_group["iterable"]

    # Determine storage formats for each side
    keys_storage_format = keys_iterable_group.attrs.get(
        "storage_format", "groups"
    )
    values_storage_format = values_iterable_group.attrs.get(
        "storage_format", "groups"
    )

    # Read the keys side (cheap, even if large)
    keys = _read_iterable_side(
        keys_iterable_group, keys_storage_format, decode_cache=None
    )

    # Iterate values, decoding one at a time for groups format
    if values_storage_format == "dataset":
        # Read all values at once (cheap native scalars)
        values = _read_iterable_side(
            values_iterable_group, values_storage_format, decode_cache=None
        )
        # Slice from start_index onward
        for k, v in zip(keys[start_index:], values[start_index:]):
            yield k, v
    else:
        # Decode one value at a time for groups format, skipping indices below start_index
        for i, key in enumerate(keys):
            if i < start_index:
                continue
            value = _decode_group_entry(values_iterable_group, i, decode_cache)
            yield key, value


def get_dict_attr_keys(parent_group: h5py.Group, attr_name: str) -> list:
    """Return just the keys of a dict-shaped HDF5 attribute, without decoding
    any values -- cheaper than `iter_dict_attr_entries` when only the set of
    present keys is needed (e.g. checking which shot indices exist), since
    that function always decodes each groups-format value eagerly before
    yielding its paired key. Returns `[]` if the attribute doesn't exist.
    """
    parent_group = _resolve_dict_target_group(parent_group, attr_name)
    if attr_name not in parent_group:
        return []
    dict_group = parent_group[attr_name]
    if "dict" not in dict_group:
        return []
    dict_subgroup = dict_group["dict"]
    if "keys" not in dict_subgroup or "iterable" not in dict_subgroup["keys"]:
        return []
    keys_iterable_group = dict_subgroup["keys"]["iterable"]
    storage_format = keys_iterable_group.attrs.get("storage_format", "groups")
    return _read_iterable_side(
        keys_iterable_group, storage_format, decode_cache=None
    )


def _stream_into_new_dict_attr(
    parent_group: h5py.Group,
    attr_name: str,
    entries: Iterable[tuple[Any, Any]],
    encode_cache: dict | None,
    key_use_dataset: bool,
    value_use_dataset: bool,
) -> None:
    """Create and stream entries into a fresh dict attribute.

    Streams entries one at a time without materializing the entire iterable.
    For dataset-format sides, the dataset is created on the first entry.
    """
    dict_group = parent_group.create_group(attr_name)
    dict_subgroup = dict_group.create_group("dict")

    keys_group = dict_subgroup.create_group("keys")
    values_group = dict_subgroup.create_group("values")

    keys_iterable_group = keys_group.create_group("iterable", track_order=True)
    values_iterable_group = values_group.create_group(
        "iterable", track_order=True
    )
    # A dict's keys/values are always encoded as a "list" regardless of the
    # original container type; `HDF5Encoder.decode_iterable` requires this
    # attr, so a structure built here stays decodable via the normal
    # recursive `Serializable.decode` path, not just `iter_dict_attr_entries`.
    keys_iterable_group.attrs["iterable_type"] = "list"
    values_iterable_group.attrs["iterable_type"] = "list"

    # Track whether datasets have been created for dataset-format sides
    keys_dataset_created = False
    values_dataset_created = False

    # Stream entries one at a time
    for key, value in entries:
        # Append key
        keys_dataset_created = _stream_entry_into_iterable_side(
            keys_iterable_group,
            key,
            key_use_dataset,
            keys_dataset_created,
            encode_cache,
        )

        # Append value
        values_dataset_created = _stream_entry_into_iterable_side(
            values_iterable_group,
            value,
            value_use_dataset,
            values_dataset_created,
            encode_cache,
        )

    # If no entries were provided, set default storage format
    if not keys_dataset_created:
        keys_iterable_group.attrs["storage_format"] = "groups"
    if not values_dataset_created:
        values_iterable_group.attrs["storage_format"] = "groups"


def _stream_into_existing_dict_attr(
    parent_group: h5py.Group,
    attr_name: str,
    entries: Iterable[tuple[Any, Any]],
    encode_cache: dict | None,
) -> None:
    """Stream entries into an existing dict attribute, preserving formats."""
    dict_group = parent_group[attr_name]
    dict_subgroup = dict_group["dict"]

    keys_group = dict_subgroup["keys"]
    values_group = dict_subgroup["values"]

    keys_iterable_group = keys_group["iterable"]
    values_iterable_group = values_group["iterable"]

    keys_storage_format = keys_iterable_group.attrs.get(
        "storage_format", "groups"
    )
    values_storage_format = values_iterable_group.attrs.get(
        "storage_format", "groups"
    )

    # Append each entry one at a time
    for key, value in entries:
        _append_to_iterable_side(
            keys_iterable_group, key, keys_storage_format, encode_cache
        )
        _append_to_iterable_side(
            values_iterable_group, value, values_storage_format, encode_cache
        )


def _stream_entry_into_iterable_side(
    iterable_group: h5py.Group,
    item: Any,
    use_dataset: bool,
    dataset_created: bool,
    encode_cache: dict | None,
) -> bool:
    """Stream a single entry into a freshly-created iterable subgroup.

    Handles both dataset creation (on the first entry) and appending.
    Returns True if the dataset was created/is active, False if groups
    format was used.

    Parameters
    ----------
    iterable_group : h5py.Group
        The iterable subgroup to append to.
    item : Any
        The item to append.
    use_dataset : bool
        Whether to use dataset format (only consulted on first entry if not
        yet created; raises TypeError if True but item isn't a native scalar).
    dataset_created : bool
        Whether a dataset has already been created for this side. Only
        relevant when called multiple times within the same fresh creation.
    encode_cache : dict | None
        Cache for encoding operations.

    Returns
    -------
    bool
        True if item was written to dataset format, False if groups format.

    Raises
    ------
    TypeError
        If use_dataset=True but item is not a native scalar type.
    """
    hdf5_native_types = (int, float, bool, str, bytes)

    if use_dataset and not dataset_created:
        # First entry for a requested dataset-format side
        item_type = type(item)

        if item_type not in hdf5_native_types:
            raise TypeError(
                f"Dataset format requested for side with first entry type "
                f"{item_type.__name__}, which is not a native HDF5 scalar "
                f"type (int, float, bool, str, or bytes)"
            )

        # Create the dataset from this first entry
        iterable_group.attrs["storage_format"] = "dataset"
        HDF5Encoder._encode_iterable_dataset(
            iterable_group, [item], extendable_dataset=True
        )
        return True
    elif dataset_created:
        # Already in dataset format, append to it
        _append_to_iterable_side(iterable_group, item, "dataset", encode_cache)
        return True
    else:
        # Use groups format; mark it on the first entry
        next_index = len(iterable_group)
        if next_index == 0:
            iterable_group.attrs["storage_format"] = "groups"
        item_group = iterable_group.create_group(
            str(next_index), track_order=True
        )
        Serializable.encode(
            item, format="hdf5", h5_group=item_group, encode_cache=encode_cache
        )
        return False


def _check_string_truncation(item: Any, dtype: np.dtype) -> None:
    """Check if a string/bytes item would be truncated by the dtype width."""
    dtype_kind = dtype.kind
    item_type = type(item)

    if dtype_kind == "U":
        # Unicode: itemsize in bytes, each char is 4 bytes
        fixed_width = dtype.itemsize // 4
    else:
        # Byte string: itemsize in bytes
        fixed_width = dtype.itemsize

    item_length = len(item)
    if item_length > fixed_width:
        raise ValueError(
            f"Cannot append {item_type.__name__} of length "
            f"{item_length} to dataset with fixed width {fixed_width}"
        )


def _validate_string_bytes_type(
    item: Any, dtype: np.dtype, iterable_group: h5py.Group
) -> None:
    """Validate string/bytes item type for a dataset."""
    item_type = type(item)
    dtype_kind = dtype.kind

    if dtype_kind == "U":  # Unicode string
        if not isinstance(item, str):
            raise TypeError(
                f"Cannot append {item_type.__name__} to dataset "
                f"of string dtype {dtype}"
            )
    elif dtype_kind == "S":  # Byte string
        # Check if originally a string (HDF5 stores strings as bytes)
        original_type = iterable_group.attrs.get("original_type")
        if original_type == "bytes":
            if not isinstance(item, bytes):
                raise TypeError(
                    f"Cannot append {item_type.__name__} to dataset "
                    f"of bytes dtype {dtype}"
                )
        else:
            # Originally string, stored as bytes
            if not isinstance(item, (str, bytes)):
                raise TypeError(
                    f"Cannot append {item_type.__name__} to dataset "
                    f"of string dtype {dtype}"
                )


def _validate_item_for_dataset(
    item: Any, dtype: np.dtype, iterable_group: h5py.Group
) -> None:
    """Validate that an item can be appended to a dataset of given dtype.

    Raises TypeError or ValueError if the item is incompatible.
    """
    item_type = type(item)
    dtype_kind = dtype.kind

    # Check type compatibility based on dtype kind
    if dtype_kind in ("i", "u"):  # integer types (signed, unsigned)
        if not isinstance(item, (int, np.integer)):
            raise TypeError(
                f"Cannot append {item_type.__name__} to dataset "
                f"of integer dtype {dtype}"
            )
    elif dtype_kind == "f":  # floating point types
        if not isinstance(item, (float, np.floating)):
            raise TypeError(
                f"Cannot append {item_type.__name__} to dataset "
                f"of float dtype {dtype}"
            )
    elif dtype_kind == "b":  # boolean types
        if not isinstance(item, (bool, np.bool_)):
            raise TypeError(
                f"Cannot append {item_type.__name__} to dataset "
                f"of bool dtype {dtype}"
            )
    elif dtype_kind in ("U", "S", "O"):  # string, bytes, or object types
        if dtype_kind != "O":  # Not object, validate string/bytes
            _validate_string_bytes_type(item, dtype, iterable_group)
        # For object dtype, allow anything
    else:
        # For other types, try conversion
        try:
            np.array([item], dtype=dtype)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Cannot append {item_type.__name__} to dataset "
                f"of dtype {dtype}"
            ) from e

    # For string/bytes, check truncation risk
    if dtype_kind in ("U", "S"):
        _check_string_truncation(item, dtype)


def _append_to_iterable_side(
    iterable_group: h5py.Group,
    item: Any,
    storage_format: str,
    encode_cache: dict | None,
) -> None:
    """Append a single item to an iterable subgroup."""
    if storage_format == "dataset":
        # Append to the dataset
        dataset = iterable_group["data"]
        current_length = len(dataset)
        dtype = dataset.dtype

        # Validate item type and check for truncation
        _validate_item_for_dataset(item, dtype, iterable_group)

        # If dataset was created non-chunked (e.g. by generic encoder), convert to extendable
        if dataset.chunks is None:
            existing_data = dataset[()]
            del iterable_group["data"]
            HDF5Encoder._encode_iterable_dataset(
                iterable_group, list(existing_data), extendable_dataset=True
            )
            dataset = iterable_group["data"]
            current_length = len(dataset)

        # Resize and append
        new_size = current_length + 1
        dataset.resize((new_size,))

        # Convert item to numpy array for assignment
        if isinstance(item, (str, bytes)):
            # Handle string/bytes specially
            if isinstance(item, bytes):
                item_to_write = item.decode("utf-8", errors="replace")
            else:
                item_to_write = item
            dataset[current_length] = item_to_write
        else:
            dataset[current_length] = item
    else:
        # Append to groups format
        next_index = len(iterable_group)
        item_group = iterable_group.create_group(
            str(next_index), track_order=True
        )
        Serializable.encode(
            item,
            format="hdf5",
            h5_group=item_group,
            encode_cache=encode_cache,
        )


def _read_iterable_side(
    iterable_group: h5py.Group,
    storage_format: str,
    decode_cache: dict | None,
) -> list[Any]:
    """Read an entire iterable side (for cheap reads like native scalars)."""
    if storage_format == "dataset":
        dataset = iterable_group["data"]
        data = dataset[()]

        # Convert numpy array back to Python types
        if data.dtype.kind in ["i", "u"]:  # integer types
            return [int(x) for x in data.flat]
        elif data.dtype.kind == "f":  # float types
            return [float(x) for x in data.flat]
        elif data.dtype.kind == "b":  # boolean types
            return [bool(x) for x in data.flat]
        elif data.dtype.kind in ["U", "S"]:  # string types
            # Check if original type was bytes
            if iterable_group.attrs.get("original_type") == "bytes":
                return [
                    x.encode("utf-8") if isinstance(x, str) else x
                    for x in data.flat
                ]
            else:
                return [
                    str(x, "utf-8") if isinstance(x, bytes) else str(x)
                    for x in data.flat
                ]
        else:
            return list(data.flat)
    else:
        # Read from groups format
        result = []
        for i in range(len(iterable_group)):
            entry = _decode_group_entry(iterable_group, i, decode_cache)
            result.append(entry)
        return result


def _decode_group_entry(
    iterable_group: h5py.Group,
    index: int,
    decode_cache: dict | None,
) -> Any:
    """Decode a single entry from a groups-format iterable."""
    item_group = iterable_group[str(index)]
    return Serializable.decode(
        item_group, format="hdf5", decode_cache=decode_cache
    )


def _find_group_index_for_key(
    parent_group: h5py.Group,
    attr_name: str,
    key: Any,
) -> int:
    """Find the physical index of an entry by its key in a dict attribute.

    Reads a dict attribute's keys side (either a dataset of native scalars,
    or for groups-format, the subgroup names) and returns the insertion index
    (physical position) whose key equals `key`. Uses the cheap side of the dict
    structure, never decoding the values.

    Parameters
    ----------
    parent_group : h5py.Group
        The HDF5 group holding the dict attribute.
    attr_name : str
        Name of the dict attribute to search.
    key : Any
        The key to find.

    Returns
    -------
    int
        The physical/insertion index of the matching entry.

    Raises
    ------
    KeyError
        If `attr_name` doesn't exist on `parent_group`, or if no entry's key
        matches the provided key.
    """
    parent_group = _resolve_dict_target_group(parent_group, attr_name)
    if attr_name not in parent_group:
        raise KeyError(f"Attribute {attr_name} not found in group")

    dict_group = parent_group[attr_name]
    if "dict" not in dict_group:
        raise KeyError(f"Attribute {attr_name} is not a valid dict structure")

    dict_subgroup = dict_group["dict"]
    if "keys" not in dict_subgroup:
        raise KeyError(f"Attribute {attr_name} has no keys subgroup")

    keys_group = dict_subgroup["keys"]
    if "iterable" not in keys_group:
        raise KeyError(f"Attribute {attr_name} has no keys iterable")

    keys_iterable_group = keys_group["iterable"]
    keys_storage_format = keys_iterable_group.attrs.get(
        "storage_format", "groups"
    )

    # Read all keys (cheap side)
    keys = _read_iterable_side(
        keys_iterable_group, keys_storage_format, decode_cache=None
    )

    # Find the index of the matching key
    for i, k in enumerate(keys):
        if k == key:
            return i

    raise KeyError(f"Key {key} not found in {attr_name}")


def get_dict_attr_value(
    parent_group: h5py.Group,
    attr_name: str,
    key: Any,
    decode_cache: dict | None = None,
) -> Any:
    """Get a single value from a dict attribute without materializing others.

    Uses `_find_group_index_for_key` to locate the entry, then decodes and
    returns only that one value, without touching any other entries.

    For dataset-format values, reads only the indexed entry from the dataset.
    For groups-format values, decodes only the indexed subgroup.

    Parameters
    ----------
    parent_group : h5py.Group
        The HDF5 group holding the dict attribute.
    attr_name : str
        Name of the dict attribute.
    key : Any
        The key to retrieve.
    decode_cache : dict | None, optional
        Cache for decoding operations (passed to `Serializable.decode`).
        Default is None.

    Returns
    -------
    Any
        The decoded value associated with the key.

    Raises
    ------
    KeyError
        If `attr_name` doesn't exist or if the key is not found.
    """
    parent_group = _resolve_dict_target_group(parent_group, attr_name)
    index = _find_group_index_for_key(parent_group, attr_name, key)

    dict_group = parent_group[attr_name]
    dict_subgroup = dict_group["dict"]
    values_group = dict_subgroup["values"]
    values_iterable_group = values_group["iterable"]

    values_storage_format = values_iterable_group.attrs.get(
        "storage_format", "groups"
    )

    if values_storage_format == "dataset":
        # Read a single entry from the dataset
        dataset = values_iterable_group["data"]
        value = dataset[index]
        # Convert numpy scalar back to Python type
        if dataset.dtype.kind in ["i", "u"]:
            return int(value)
        elif dataset.dtype.kind == "f":
            return float(value)
        elif dataset.dtype.kind == "b":
            return bool(value)
        elif dataset.dtype.kind in ["U", "S"]:
            if values_iterable_group.attrs.get("original_type") == "bytes":
                return (
                    value.encode("utf-8") if isinstance(value, str) else value
                )
            else:
                return (
                    str(value, "utf-8")
                    if isinstance(value, bytes)
                    else str(value)
                )
        else:
            return value
    else:
        # Decode a single group entry
        return _decode_group_entry(values_iterable_group, index, decode_cache)


def get_dict_attr_group(
    parent_group: h5py.Group,
    attr_name: str,
    key: Any,
) -> h5py.Group:
    """Get the raw HDF5 Group for a dict entry (groups-format values only).

    Uses `_find_group_index_for_key` to locate the entry, then returns the
    raw (undecoded) `h5py.Group` for that entry's value subgroup, without
    calling `Serializable.decode` on it.

    Parameters
    ----------
    parent_group : h5py.Group
        The HDF5 group holding the dict attribute.
    attr_name : str
        Name of the dict attribute.
    key : Any
        The key to retrieve.

    Returns
    -------
    h5py.Group
        The raw HDF5 Group for the value's indexed subgroup.

    Raises
    ------
    KeyError
        If `attr_name` doesn't exist or if the key is not found.
    TypeError
        If the values side for `attr_name` is dataset-format (there is no
        group to return in that case).
    """
    parent_group = _resolve_dict_target_group(parent_group, attr_name)
    index = _find_group_index_for_key(parent_group, attr_name, key)

    dict_group = parent_group[attr_name]
    dict_subgroup = dict_group["dict"]
    values_group = dict_subgroup["values"]
    values_iterable_group = values_group["iterable"]

    values_storage_format = values_iterable_group.attrs.get(
        "storage_format", "groups"
    )

    if values_storage_format == "dataset":
        raise TypeError(
            f"Cannot get group for dict attribute '{attr_name}' with "
            f"dataset-format values; only groups-format entries have "
            f"raw HDF5 Groups to return"
        )

    # Return the raw group for the indexed entry
    return values_iterable_group[str(index)]
