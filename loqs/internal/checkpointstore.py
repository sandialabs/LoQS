#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Durable checkpoint storage for streamed, int-keyed dict attributes.

A store holds entries keyed by an integer (a shot or item index). Each entry
carries one value per streamed attribute. Entries live in a canonical HDF5
file and, until consolidated, in atomically written per-worker payload
files. `CheckpointStore` is the durable implementation and
`MemoryCheckpointStore` is an in-memory twin with the same interface.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ContextManager

import h5py
import numpy as np

from loqs.internal.encoder.hdf5encoder import HDF5Encoder, _get_version
from loqs.internal.serializable import ResolvingDecodeCache, Serializable
from loqs.internal.streamingmerge import (
    _decode_group_entry,
    decode_object_attrs,
    _read_iterable_side,
    _reset_empty_groups_format_dict_attr,
    _resolve_checkpoint_object_group,
    merge_dict_attr,
    remove_transient_file,
)
from loqs.internal.swmrledger import is_swmr_ledger_file

COMMIT_MARKER_ATTR = "checkpoint_store_committed"
"""Name of the file-root attribute holding the committed lengths."""

NEXT_CACHE_ID_ATTR = "checkpoint_store_next_cache_id"
"""Name of the int file-root attribute of a canonical file holding the next
free encode cache ID. Every cache ID in the file is below it."""


@dataclass(frozen=True)
class StreamedAttr:
    """One streamed dict attribute of a store layout.

    `name` is the attribute's name in payloads and in `put_batch` entries.
    `use_dataset` is the values side's storage format when the attribute is
    first created. `canonical_name` is the name in the canonical file, when
    it differs from `name`.
    """

    name: str
    use_dataset: bool
    canonical_name: str | None = None

    @property
    def canonical(self) -> str:
        """The attribute's name in the canonical file."""
        return self.canonical_name or self.name


@dataclass(frozen=True)
class StoreLayout:
    """File names and streamed attributes of one kind of store.

    `attrs[0]` is the primary attribute, whose keys say whether an entry
    exists.

    `shared_attr` names the canonical file's non-streamed attribute whose
    cached objects payloads may reference, or is None when payloads are
    self-contained.
    """

    canonical_filename: str
    ledger_tag: str
    payload_tag: str
    attrs: tuple[StreamedAttr, ...]
    shared_attr: str | None = None

    @property
    def primary(self) -> StreamedAttr:
        """The attribute whose keys define which entries exist."""
        return self.attrs[0]

    def attr(self, name: str) -> StreamedAttr:
        """The attribute called `name`; `ValueError` if there is none."""
        for streamed in self.attrs:
            if streamed.name == name:
                return streamed
        raise ValueError(
            f"Unknown streamed attribute {name!r}; known attributes are "
            f"{[a.name for a in self.attrs]}"
        )


SHOT_LAYOUT = StoreLayout(
    "results.h5",
    "checkpoint",
    "shots",
    (
        StreamedAttr("shot_histories", False),
        StreamedAttr("shot_wall_clock_times", True),
    ),
    "parent_program",
)
"""Shot-level layout, as `ProgramResults` checkpoints today. Shot payloads
reference the parent program stored in the canonical file."""

ITEM_LAYOUT = StoreLayout(
    "runner.h5",
    "runner",
    "item",
    (
        StreamedAttr("results", False, "_reduced_results"),
        StreamedAttr("item_wall_clock_times", True),
        StreamedAttr("shot_wall_clock_times", False),
        StreamedAttr("_program_results", False),
    ),
)
"""Item-level layout, as `MultiProgramRunner` checkpoints today."""


@dataclass(frozen=True)
class BatchReceipt:
    """What a `put_batch` wrote: the sorted keys and the file, if any."""

    keys: tuple[int, ...]
    path: str | None


def _validate_batch(
    layout: StoreLayout, entries: Mapping[int, Mapping[str, Any]]
) -> None:
    """Raise `ValueError` for an entry that lacks the primary attribute or
    holds an unknown one."""
    primary = layout.primary.name
    for key, values in entries.items():
        for name in values:
            layout.attr(name)
        if primary not in values:
            raise ValueError(
                f"Entry {key!r} is missing the primary attribute {primary!r}"
            )


class BaseCheckpointStore(ABC):
    """The interface shared by the durable and in-memory stores.

    `attr` defaults to the primary attribute and is a payload-side name.
    """

    layout: StoreLayout

    @abstractmethod
    def put_batch(
        self,
        entries: Mapping[int, Mapping[str, Any]],
        *,
        worker_id: str | None = None,
        encode_cache: dict | None = None,
    ) -> BatchReceipt:
        """Store a batch of entries and return a receipt."""

    @abstractmethod
    def keys(self, attr: str | None = None, *, strict: bool = True) -> list:
        """The sorted keys stored for `attr`."""

    @abstractmethod
    def get(
        self, key: int, attr: str | None = None, *, strict: bool = True
    ) -> Any:
        """The value of `attr` at `key`; `KeyError` if absent everywhere."""

    @abstractmethod
    def iter_items(
        self,
        attr: str | None = None,
        *,
        keys: Iterable[int] | None = None,
        strict: bool = True,
    ) -> Iterator[tuple[int, Any]]:
        """Yield `(key, value)` pairs, each key once, in ascending order.

        The durable store keeps the current file open between yields, so
        finish or close the iterator before writing to the same store.
        """

    def missing(
        self, expected: Iterable[int], attr: str | None = None
    ) -> list[int]:
        """The sorted expected keys that are absent. Reads strictly."""
        present = set(self.keys(attr, strict=True))
        return sorted({int(k) for k in expected} - present)

    @abstractmethod
    def consolidate(
        self, *, encode_cache: dict | None = None
    ) -> tuple[int, ...]:
        """Merge every payload into the canonical file."""

    @abstractmethod
    def write_snapshot(
        self, obj: Serializable, *, keep_entries: bool = False
    ) -> None:
        """Atomically write the canonical file from `obj`.

        With `keep_entries`, the committed entries of the existing file
        replace `obj`'s own streamed entries."""

    @abstractmethod
    def nested(
        self,
        key: int,
        attr: str = "_program_results",
        layout: StoreLayout = SHOT_LAYOUT,
    ) -> BaseCheckpointStore:
        """A read-only view of the object stored at `key` in `attr`."""


def _temp_path(path: Path) -> Path:
    """The name `atomic_h5_file` writes `path` under before renaming."""
    return path.with_name(path.name + ".tmp")


@contextlib.contextmanager
def atomic_h5_file(path: str | os.PathLike) -> Iterator[h5py.File]:
    """Write an HDF5 file atomically.

    Opens `<path>.tmp` in mode `"w"` and yields it. On a normal exit the
    file is closed and renamed onto `path`. On any `BaseException` the file
    is closed, the temp file is removed and the exception propagates.

    Windows can't replace a file that a handle holds open, so a caller
    holding a handle on `path` closes it inside the block.
    """
    final = Path(path)
    temp = _temp_path(final)
    f: h5py.File | None = None
    try:
        f = open_h5(temp, "w")
        yield f
        f.close()
        os.replace(temp, final)
    except BaseException:
        if f is not None:
            with contextlib.suppress(Exception):
                f.close()
        with contextlib.suppress(OSError):
            temp.unlink()
        raise


def _iterable_groups(
    container: h5py.Group, canonical_name: str
) -> tuple[h5py.Group, h5py.Group] | None:
    """The keys-side and values-side `iterable` groups of a dict attr, or
    None if the attr is absent.

    An attr whose structure is only partly built, as a write cut short
    leaves it, counts as absent: nothing in it can have been committed.
    """
    if canonical_name not in container:
        return None
    try:
        sides = container[canonical_name]["dict"]
        pair = sides["keys"]["iterable"], sides["values"]["iterable"]
    except KeyError:
        return None
    for side in pair:
        if side.attrs.get("storage_format", "groups") == "dataset" and (
            "data" not in side
        ):
            return None
    return pair


def side_length(iterable_group: h5py.Group) -> int:
    """The number of entries a dict side physically holds."""
    if iterable_group.attrs.get("storage_format", "groups") == "dataset":
        return int(iterable_group["data"].shape[0])
    return len(iterable_group)


def _make_side_growable(iterable_group: h5py.Group) -> None:
    """Rewrite a fixed-size dataset side as an extendable one."""
    if iterable_group.attrs.get("storage_format", "groups") != "dataset":
        return
    dataset = iterable_group["data"]
    if dataset.chunks is not None or dataset.shape[0] == 0:
        return
    data = dataset[()]
    if (
        data.dtype.kind == "S"
        and iterable_group.attrs.get("original_type") != "bytes"
    ):
        items: list[Any] = [x.decode("utf-8") for x in data.flat]
    elif data.dtype.kind in ("S", "U"):
        items = list(data.flat)
    else:
        items = [x.item() if isinstance(x, np.generic) else x for x in data]
    del iterable_group["data"]
    HDF5Encoder._encode_iterable_dataset(
        iterable_group, items, extendable_dataset=True
    )


class CheckpointReadError(RuntimeError):
    """A store file could not be read.

    `path` names the file. The underlying error is chained as the cause.
    """

    def __init__(self, path: str | os.PathLike, reason: str) -> None:
        self.path = os.fspath(path)
        self.reason = reason
        super().__init__(f"{self.path}: {reason}")

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.path, self.reason))


_OPEN_ATTEMPTS = 5


def open_h5(path: str | os.PathLike, mode: str) -> h5py.File:
    """Open an HDF5 file, retrying only on `BlockingIOError`.

    Lock contention is retried up to 5 attempts in all, waiting
    `delay + random.uniform(0, delay)` with `delay = 0.01 * 2**attempt`.
    The last `BlockingIOError` is re-raised. Any other error propagates at
    once.
    """
    for attempt in range(_OPEN_ATTEMPTS):
        try:
            return h5py.File(path, mode)
        except BlockingIOError:
            if attempt == _OPEN_ATTEMPTS - 1:
                raise
            delay = 0.01 * 2**attempt
            time.sleep(delay + random.uniform(0, delay))
    raise AssertionError("unreachable")  # pragma: no cover


def _dataset_value(values_group: h5py.Group, pos: int) -> Any:
    """One native value from a dataset-format side, as a Python object."""
    dataset = values_group["data"]
    value = dataset[pos]
    kind = dataset.dtype.kind
    if kind in ("i", "u"):
        return int(value)
    if kind == "f":
        return float(value)
    if kind == "b":
        return bool(value)
    if kind in ("U", "S"):
        if values_group.attrs.get("original_type") == "bytes":
            return value.encode("utf-8") if isinstance(value, str) else value
        return str(value, "utf-8") if isinstance(value, bytes) else str(value)
    return value


@dataclass
class _Opened:
    """An open store file, ready to read.

    `container` holds the streamed attrs. `limits` maps canonical names to
    committed lengths, or is None when every entry counts.
    """

    container: h5py.Group
    limits: dict[str, int] | None
    cache: Any
    path: str
    canonical_names: bool
    require_primary: bool

    def _name(self, streamed: StreamedAttr) -> str:
        return streamed.canonical if self.canonical_names else streamed.name

    def sides(
        self, streamed: StreamedAttr
    ) -> tuple[h5py.Group, h5py.Group] | None:
        return _iterable_groups(self.container, self._name(streamed))

    def count(self, streamed: StreamedAttr) -> int:
        """The number of entries of `streamed` that count as stored."""
        sides = self.sides(streamed)
        if sides is None:
            return 0
        if self.limits is None:
            return side_length(sides[0])
        return self.limits.get(streamed.canonical, 0)

    def validate(self, layout: StoreLayout) -> None:
        """Raise if the file is not a consistent store file."""
        if self.require_primary and self.sides(layout.primary) is None:
            raise ValueError(
                f"missing the primary attribute {layout.primary.name!r}"
            )
        for streamed in layout.attrs:
            sides = self.sides(streamed)
            committed = (
                0
                if self.limits is None
                else self.limits.get(streamed.canonical, 0)
            )
            if sides is None:
                if committed:
                    raise ValueError(
                        f"attribute {streamed.canonical!r} is absent but "
                        f"{committed} entries are committed"
                    )
                continue
            n_keys, n_values = (side_length(s) for s in sides)
            if self.limits is None:
                if n_keys != n_values:
                    raise ValueError(
                        f"attribute {self._name(streamed)!r} has {n_keys} "
                        f"keys but {n_values} values"
                    )
            elif min(n_keys, n_values) < committed:
                raise ValueError(
                    f"attribute {streamed.canonical!r} holds "
                    f"{min(n_keys, n_values)} entries, fewer than the "
                    f"{committed} committed"
                )

    def keys(self, streamed: StreamedAttr) -> list[int]:
        """The stored keys of `streamed`, in position order."""
        sides = self.sides(streamed)
        if sides is None:
            return []
        keys_group = sides[0]
        fmt = keys_group.attrs.get("storage_format", "groups")
        if self.limits is None:
            n = side_length(keys_group)
        else:
            n = self.limits.get(streamed.canonical, 0)
        all_keys = _read_iterable_side(keys_group, fmt, None)
        return [int(k) for k in all_keys[:n]]

    def read(self, streamed: StreamedAttr, pos: int) -> Any:
        """Decode the value at physical position `pos`."""
        sides = self.sides(streamed)
        assert sides is not None
        values_group = sides[1]
        if values_group.attrs.get("storage_format", "groups") == "dataset":
            return _dataset_value(values_group, pos)
        try:
            return _decode_group_entry(values_group, pos, self.cache)
        except BaseException:
            # A cut-short decode leaves `DeferredRef` placeholders; reset.
            # The cache's fallback still supplies the shared objects.
            self.cache.clear()
            raise


@dataclass
class _Source:
    """One place entries live: a path and a way to open it."""

    path: Path
    opener: Callable[[], ContextManager[_Opened]]


class _ReadableStore(BaseCheckpointStore):
    """Reads over a list of sources under the collision rule: for each
    attr, the first source holding a key wins, and within one source the
    first position does."""

    def _sources(self) -> list[_Source]:
        raise NotImplementedError

    def _streamed(self, attr: str | None) -> StreamedAttr:
        if attr is None:
            return self.layout.primary
        return self.layout.attr(attr)

    @contextlib.contextmanager
    def _opened(
        self,
        source: _Source,
        *,
        wrap_body: bool = True,
        validate: bool = True,
    ) -> Iterator[_Opened]:
        """Open and validate a source; any failure becomes a
        `CheckpointReadError` naming it. With `wrap_body=False`, errors
        raised by the caller's own `with` body propagate unchanged."""
        in_body = False
        try:
            with source.opener() as opened:
                if validate:
                    opened.validate(self.layout)
                in_body = True
                yield opened
        except CheckpointReadError:
            raise
        except Exception as exc:
            if in_body and not wrap_body:
                raise
            raise CheckpointReadError(
                source.path, f"{type(exc).__name__}: {exc}"
            ) from exc

    def _key_index(
        self, streamed: StreamedAttr, strict: bool
    ) -> dict[int, tuple[_Source, int]]:
        """Map each stored key to the source and position that win."""
        index: dict[int, tuple[_Source, int]] = {}
        for source in self._sources():
            try:
                with self._opened(source) as opened:
                    keys = opened.keys(streamed)
            except CheckpointReadError:
                if strict:
                    raise
                continue
            for pos, key in enumerate(keys):
                index.setdefault(key, (source, pos))
        return index

    def keys(self, attr: str | None = None, *, strict: bool = True) -> list:
        return sorted(self._key_index(self._streamed(attr), strict))

    def get(
        self, key: int, attr: str | None = None, *, strict: bool = True
    ) -> Any:
        streamed = self._streamed(attr)
        index = self._key_index(streamed, strict)
        if key not in index:
            raise KeyError(key)
        source, pos = index[key]
        try:
            with self._opened(source) as opened:
                return opened.read(streamed, pos)
        except CheckpointReadError as exc:
            if strict:
                raise
            raise KeyError(key) from exc

    def iter_items(
        self,
        attr: str | None = None,
        *,
        keys: Iterable[int] | None = None,
        strict: bool = True,
    ) -> Iterator[tuple[int, Any]]:
        streamed = self._streamed(attr)
        index = self._key_index(streamed, strict)
        if keys is None:
            wanted = sorted(index)
        else:
            wanted = sorted({int(k) for k in keys}.intersection(index))
        # Only one decoded value is alive at a time, and a source stays
        # open while consecutive keys come from it.
        with contextlib.ExitStack() as stack:
            current: _Source | None = None
            opened: _Opened | None = None
            for key in wanted:
                source, pos = index[key]
                try:
                    if source is not current:
                        stack.close()
                        current = None
                        opened = stack.enter_context(self._opened(source))
                        current = source
                    assert opened is not None
                    value = opened.read(streamed, pos)
                except CheckpointReadError:
                    if strict:
                        raise
                    continue
                except Exception as exc:
                    if strict:
                        raise CheckpointReadError(
                            source.path, f"{type(exc).__name__}: {exc}"
                        ) from exc
                    continue
                yield key, value


def _working_copy(encode_cache: dict | None) -> dict:
    """A copy of an encode cache that encoding can't change in place.

    Encoding appends to the per-hash lists, so the lists are copied too.
    """
    if encode_cache is None:
        return {}
    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in encode_cache.items()
    }


_PAYLOAD_SEED: tuple[tuple[str, tuple[int, int, int]], tuple[dict, int]] | None
_PAYLOAD_SEED = None
"""The process's one payload seed: `((canonical path, file identity),
(encode seed, next-ID record))`. Each batch builds a new store, so only a
module-level slot lets a worker's cache outlive a batch."""


def _invert_shared(shared: Mapping[int, Any]) -> dict:
    """An encode cache from decoded shared objects. Entries carry no object
    id, since a decoded object's id goes stale once its cache is freed. The
    lowest ID per hash wins, which is the source's, never a copy's."""
    seed: dict = {}
    for cache_id in sorted(shared):
        seed.setdefault(
            Serializable._serial_hash(shared[cache_id]), [(None, cache_id)]
        )
    return seed


def _parse_marker(f: h5py.File) -> dict[str, int]:
    """The committed lengths in a canonical file's marker; `ValueError` if
    the marker is missing."""
    marker = f.attrs.get(COMMIT_MARKER_ATTR)
    if marker is None:
        raise ValueError("the canonical file has no commit marker")
    if isinstance(marker, bytes):
        marker = marker.decode("utf-8")
    return {k: int(v) for k, v in json.loads(marker).items()}


def raise_if_uncommitted(f: h5py.File) -> None:
    """Raise `CheckpointReadError` if a canonical file holds entries past
    its commit marker. A file with no marker passes."""
    if COMMIT_MARKER_ATTR not in f.attrs:
        return
    container = _resolve_checkpoint_object_group(f)
    for name, committed in _parse_marker(f).items():
        sides = _iterable_groups(container, name)
        if sides is None:
            if name not in container:
                continue
            problem = "is only partly built"
        else:
            longest = max(side_length(side) for side in sides)
            if longest <= committed:
                continue
            problem = (
                f"holds {longest} entries, more than {committed} committed"
            )
        raise CheckpointReadError(
            f.filename,
            f"attribute {name!r} {problem}: the file holds entries an "
            "interrupted write left past its commit marker, which aren't "
            "part of the checkpoint. Resume the run (or use "
            "`load_checkpoint` for a shot checkpoint) to read only the "
            "committed entries.",
        )


def _trim_attr(
    container: h5py.Group,
    streamed: StreamedAttr,
    committed: int,
    path: str,
) -> None:
    """Cut both sides of a streamed attr back to `committed` entries.

    With nothing committed the whole attr is deleted, so the next append
    recreates it in the layout's format. Sides shorter than `committed`
    are corruption and raise `CheckpointReadError`.
    """
    name = streamed.canonical
    sides = _iterable_groups(container, name)
    if sides is None:
        if committed:
            raise CheckpointReadError(
                path,
                f"attribute {name!r} is absent but {committed} entries "
                "are committed",
            )
        if name in container:
            del container[name]  # a partly built attr from a cut write
        return
    lengths = [side_length(side) for side in sides]
    if min(lengths) < committed:
        raise CheckpointReadError(
            path,
            f"attribute {name!r} holds {min(lengths)} entries, fewer than "
            f"the {committed} committed",
        )
    if committed == 0:
        del container[name]
        return
    for side, length in zip(sides, lengths):
        if length == committed:
            continue
        if side.attrs.get("storage_format", "groups") == "dataset":
            side["data"].resize((committed,))
        else:
            for i in range(committed, length):
                del side[str(i)]


def _read_next_id(f: h5py.File, path: str | os.PathLike) -> int:
    """The next-ID record of a canonical file; `CheckpointReadError` if it
    is missing or not an int."""
    value = f.attrs.get(NEXT_CACHE_ID_ATTR)
    if value is None:
        raise CheckpointReadError(
            path, f"the canonical file has no {NEXT_CACHE_ID_ATTR!r} record"
        )
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise CheckpointReadError(
            path, f"{NEXT_CACHE_ID_ATTR!r} is not an int: {value!r}"
        )
    return int(value)


@dataclass
class _CanonicalWriter:
    """The canonical file, open for appending after its tail was trimmed.

    `committed` holds the keys already committed for each attr (by
    payload-side name). Nothing counts as committed until `commit` has
    rewritten the marker.
    """

    f: h5py.File
    container: h5py.Group
    layout: StoreLayout
    limits: dict[str, int]
    committed: dict[str, set[int]]
    next_id: int

    def new_keys(self, streamed: StreamedAttr, keys: Iterable[int]) -> list:
        """The sorted keys of `keys` not yet committed for `streamed`."""
        done = self.committed[streamed.name]
        return sorted({int(k) for k in keys} - done)

    def append(
        self,
        streamed: StreamedAttr,
        pairs: Iterable[tuple[int, Any]],
        encode_cache: dict,
    ) -> None:
        """Append `(key, value)` pairs, uncommitted until `commit`.

        The encoder's ID counter is raised to the file's next free ID
        first, so a counter reset between appends is harmless.
        """
        HDF5Encoder.ENCODE_ID = max(HDF5Encoder.ENCODE_ID, self.next_id)
        merge_dict_attr(
            self.container,
            streamed.canonical,
            pairs,
            encode_cache=encode_cache,
            key_use_dataset=True,
            value_use_dataset=streamed.use_dataset,
        )
        self.next_id = max(self.next_id, HDF5Encoder.ENCODE_ID)

    def commit(self, added: Mapping[str, Iterable[int]]) -> None:
        """Write the next-ID record, then overwrite the marker in place
        with the physical lengths.

        The record goes first: one above the committed IDs only skips
        some, while one below them would let a later write reuse them.
        """
        lengths = dict(self.limits)
        for streamed in self.layout.attrs:
            sides = _iterable_groups(self.container, streamed.canonical)
            if sides is None:
                lengths[streamed.canonical] = 0
                continue
            n_keys, n_values = (side_length(s) for s in sides)
            if n_keys != n_values:
                raise RuntimeError(
                    f"attribute {streamed.canonical!r} has {n_keys} keys "
                    f"but {n_values} values"
                )
            lengths[streamed.canonical] = n_keys
        self.f.attrs.modify(NEXT_CACHE_ID_ATTR, self.next_id)
        self.f.attrs.modify(COMMIT_MARKER_ATTR, json.dumps(lengths))
        self.limits = lengths
        for name, keys in added.items():
            self.committed[name].update(keys)


class CheckpointStore(_ReadableStore):
    """A durable store in one directory: a canonical HDF5 file plus
    atomically written payload files and SWMR ledgers.

    The store owns every checkpoint file name. Its canonical file carries a
    commit marker (`COMMIT_MARKER_ATTR` on the file root): a JSON object
    that maps each streamed attribute's canonical name to its committed
    length.

    With a layout that has a shared attribute, a payload references the
    objects of that attribute stored in the canonical file, and every read
    decodes the payload with them.
    """

    def __init__(
        self,
        directory: str | os.PathLike,
        layout: StoreLayout,
        *,
        canonical_filename: str | None = None,
    ) -> None:
        if canonical_filename is not None:
            layout = replace(layout, canonical_filename=canonical_filename)
        self.directory = Path(directory)
        self.layout = layout
        self._init_caches()

    def _init_caches(self) -> None:
        self._canonical_decode_cache: Any = None
        self._canonical_identity: tuple[int, int, int] | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {"directory": self.directory, "layout": self.layout}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.directory = state["directory"]
        self.layout = state["layout"]
        self._init_caches()

    # -- naming -----------------------------------------------------------

    @property
    def canonical_path(self) -> Path:
        """The canonical file's path."""
        return self.directory / self.layout.canonical_filename

    def ledger_path(self, worker_id: str) -> Path:
        """The SWMR ledger file of a worker."""
        return self.directory / (
            f"worker_{worker_id}_{self.layout.ledger_tag}.h5"
        )

    def payload_path(self, worker_id: str, first_key: int) -> Path:
        """The payload file of a worker's batch starting at `first_key`."""
        return self.directory / (
            f"worker_{worker_id}_{self.layout.payload_tag}_{first_key}"
            "_payload.h5"
        )

    def _name_patterns(self) -> tuple[re.Pattern[str], re.Pattern[str]]:
        ledger = re.compile(
            rf"worker_(?P<id>.+)_{re.escape(self.layout.ledger_tag)}\.h5"
        )
        payload = re.compile(
            rf"worker_(?P<id>.+)_{re.escape(self.layout.payload_tag)}"
            r"_-?\d+_payload\.h5"
        )
        return ledger, payload

    def _existing(self, pattern: re.Pattern[str]) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(
            (
                p
                for p in self.directory.iterdir()
                if pattern.fullmatch(p.name) and p.is_file()
            ),
            key=lambda p: p.name,
        )

    def ledger_paths(self) -> list[Path]:
        """Existing ledger files, sorted by name, `.tmp` files excluded."""
        return self._existing(self._name_patterns()[0])

    def payload_paths(self) -> list[Path]:
        """Existing payload files, sorted by name, `.tmp` files excluded."""
        return self._existing(self._name_patterns()[1])

    def worker_id_of(self, path: str | os.PathLike) -> str:
        """The worker ID in a ledger or payload file name.

        Raises `ValueError` on any other name. Worker IDs may contain
        underscores, so the match is anchored on the name's suffix.
        """
        name = os.path.basename(os.fspath(path))
        for pattern in self._name_patterns():
            match = pattern.fullmatch(name)
            if match:
                return match.group("id")
        raise ValueError(f"{name!r} is not a ledger or payload file name")

    def has_content(self) -> bool:
        """Whether the directory exists and holds any entry other than the
        canonical file's temp file.

        That temp file is what a kill during the first snapshot leaves, and
        the next snapshot overwrites it, so it doesn't count as content.
        """
        if not self.directory.is_dir():
            return False
        ignored = _temp_path(self.canonical_path)
        return any(p != ignored for p in self.directory.iterdir())

    # -- writing ----------------------------------------------------------

    def write_snapshot(
        self, obj: Serializable, *, keep_entries: bool = False
    ) -> None:
        """Atomically write the canonical file from `obj`.

        Inside the temp file and before the rename, empty groups-format
        streamed attributes are removed, every fixed-size dataset side is
        made growable and the commit marker is written from the physical
        lengths.

        With `keep_entries`, the existing canonical file must exist and be
        readable. Its committed entries of every streamed attribute are
        copied into the new file as stored, never decoded, in place of
        whatever `obj` held. Any tail beyond the old marker is dropped, and
        the new cache IDs continue above every ID the old file holds.
        Either way the next free cache ID is recorded in the new file
        (`NEXT_CACHE_ID_ATTR`).
        """
        with contextlib.ExitStack() as stack:
            old: tuple[h5py.Group, dict[str, int], int] | None = None
            if keep_entries:
                old = self._open_for_rewrite(stack)
            self.directory.mkdir(parents=True, exist_ok=True)
            self._write_snapshot_file(obj, old, stack)
        self._canonical_decode_cache = None
        self._canonical_identity = None

    def _open_for_rewrite(
        self, stack: contextlib.ExitStack
    ) -> tuple[h5py.Group, dict[str, int], int]:
        """Open the canonical file read-only on `stack` and validate it;
        returns its object group, committed lengths and next-ID record."""
        path = self.canonical_path
        if not path.is_file():
            raise FileNotFoundError(
                f"The canonical file {path} does not exist; "
                "write_snapshot creates it"
            )
        try:
            f = stack.enter_context(open_h5(path, "r"))
            limits = _parse_marker(f)
            container = _resolve_checkpoint_object_group(f)
            _Opened(container, limits, None, str(path), True, False).validate(
                self.layout
            )
            next_id = _read_next_id(f, path)
        except CheckpointReadError:
            raise
        except Exception as exc:
            raise CheckpointReadError(
                path, f"{type(exc).__name__}: {exc}"
            ) from exc
        return container, limits, next_id

    def _carry_entries(
        self,
        container: h5py.Group,
        old: tuple[h5py.Group, dict[str, int], int],
    ) -> None:
        """Replace every streamed attr of `container` with a copy of the
        old file's committed entries."""
        old_container, limits, _ = old
        for streamed in self.layout.attrs:
            name = streamed.canonical
            if name in container:
                del container[name]
            committed = limits.get(name, 0)
            if committed <= 0:
                continue
            # Called on the destination, which is the group written to.
            container.copy(old_container[name], container, name=name)
            _trim_attr(
                container, streamed, committed, str(self.canonical_path)
            )
            sides = _iterable_groups(container, name)
            assert sides is not None
            for side in sides:
                _make_side_growable(side)

    def _write_snapshot_file(
        self,
        obj: Serializable,
        old: tuple[h5py.Group, dict[str, int], int] | None,
        stack: contextlib.ExitStack,
    ) -> None:
        with atomic_h5_file(self.canonical_path) as f:
            if old is None:
                obj.dump(f, "hdf5")
            else:
                # Continue above every ID the carried entries may hold.
                HDF5Encoder.ENCODE_ID = max(HDF5Encoder.ENCODE_ID, old[2])
                obj.dump(f, "hdf5", reset_encode_id=False)
            for streamed in self.layout.attrs:
                _reset_empty_groups_format_dict_attr(f, streamed.canonical)
            container = _resolve_checkpoint_object_group(f)
            if old is not None:
                self._carry_entries(container, old)
            lengths: dict[str, int] = {}
            for streamed in self.layout.attrs:
                sides = _iterable_groups(container, streamed.canonical)
                if sides is None:
                    lengths[streamed.canonical] = 0
                    continue
                for side in sides:
                    _make_side_growable(side)
                lengths[streamed.canonical] = side_length(sides[0])
            f.attrs[NEXT_CACHE_ID_ATTR] = HDF5Encoder.ENCODE_ID
            f.attrs[COMMIT_MARKER_ATTR] = json.dumps(lengths)
            # Windows can't replace a file a handle holds open, so the old
            # file closes before the rename. `old` is stale from here on.
            stack.close()

    def put_batch(
        self,
        entries: Mapping[int, Mapping[str, Any]],
        *,
        worker_id: str | None = None,
        encode_cache: dict | None = None,
    ) -> BatchReceipt:
        """Store a batch of entries.

        With a `worker_id` the batch is one atomically written payload
        file. Without one it is a canonical append: entries are added to
        the canonical file in place and committed by rewriting its marker
        last. `encode_cache` is only for canonical appends and changes only
        when the batch commits.

        For a layout with a shared attribute, a payload references the
        objects of that attribute stored in the canonical file, instead of
        embedding them, so the canonical file must exist
        (`FileNotFoundError` otherwise). The process keeps one encode
        cache seeded from it across batches, and the payload's IDs start
        above the file's next-ID record.
        """
        if worker_id is not None and encode_cache is not None:
            raise ValueError(
                "encode_cache applies only to canonical appends, not to "
                "payload batches"
            )
        _validate_batch(self.layout, entries)
        if not entries:
            return BatchReceipt((), None)
        keys = tuple(sorted(entries))
        if worker_id is None:
            self._put_canonical(entries, keys, encode_cache)
            return BatchReceipt(keys, str(self.canonical_path))
        path = self.payload_path(worker_id, keys[0])
        if self.layout.shared_attr is None:
            # A fresh cache per payload keeps it self-contained.
            cache: dict = {}
        else:
            seed, floor = self._payload_seed()
            HDF5Encoder.ENCODE_ID = max(HDF5Encoder.ENCODE_ID, floor)
            cache = _working_copy(seed)
        self.directory.mkdir(parents=True, exist_ok=True)
        with atomic_h5_file(path) as f:
            for streamed in self.layout.attrs:
                pairs = [
                    (key, entries[key][streamed.name])
                    for key in keys
                    if streamed.name in entries[key]
                ]
                if not pairs:
                    continue
                merge_dict_attr(
                    f,
                    streamed.name,
                    pairs,
                    encode_cache=cache,
                    key_use_dataset=True,
                    value_use_dataset=streamed.use_dataset,
                )
        return BatchReceipt(keys, str(path))

    # -- reading ----------------------------------------------------------

    def read_attrs(
        self,
        names: Iterable[str],
        *,
        decode_cache: Any = None,
    ) -> dict[str, Any]:
        """Decode the named non-streamed attributes of the committed
        canonical file, without decoding any streamed entry.

        Raises `CheckpointReadError` naming the file if the file, marker or
        a name is missing or decoding fails, and `ValueError` for a
        streamed attribute. A given `decode_cache` is re-rooted at the file.
        """
        wanted = list(names)
        streamed_names = {s.canonical for s in self.layout.attrs}
        for name in wanted:
            if name in streamed_names:
                raise ValueError(
                    f"{name!r} is a streamed attribute; read it through "
                    "keys and get"
                )
        with self._committed_object(decode_cache) as (group, cache):
            result = decode_object_attrs(group, wanted, cache)
        absent = [name for name in wanted if name not in result]
        if absent:
            raise CheckpointReadError(
                self.canonical_path, f"attributes {absent} are missing"
            )
        return result

    def read_config(self) -> Serializable:
        """The committed canonical file's object, rebuilt with its stored
        class and without its streamed attributes.

        Raises `CheckpointReadError` naming the file if the file or marker
        is missing or decoding fails. The canonical decode cache is left
        alone.
        """
        streamed_names = {s.canonical for s in self.layout.attrs}
        with self._committed_object() as (group, cache):
            cls = Serializable._import_class(
                group.attrs["module"],
                group.attrs["class"],
                _get_version(group),
            )
            names = [
                n for n in cls._SERIALIZE_ATTRS if n not in streamed_names
            ]
            attrs = decode_object_attrs(group, names, cache)
            return cls._from_decoded_attrs(attrs)

    @contextlib.contextmanager
    def _committed_object(
        self, decode_cache: Any = None
    ) -> Iterator[tuple[h5py.Group, Any]]:
        """Open the committed canonical file read-only and yield its object
        group and a decode cache; any failure, in the body too, becomes a
        `CheckpointReadError` naming the file."""
        path = self.canonical_path
        try:
            f = open_h5(path, "r")
            try:
                _parse_marker(f)
                if decode_cache is None:
                    decode_cache = ResolvingDecodeCache(root=f, format="hdf5")
                else:
                    decode_cache._root = f
                yield _resolve_checkpoint_object_group(f), decode_cache
            finally:
                f.close()
        except CheckpointReadError:
            raise
        except Exception as exc:
            raise CheckpointReadError(
                path, f"{type(exc).__name__}: {exc}"
            ) from exc

    def shared_encode_cache(self) -> dict:
        """A new encode cache seeded from the canonical file's shared
        attribute, for encoding objects that reference it.

        Empty for a layout without a shared attribute. Raises
        `CheckpointReadError` if the attribute can't be read.
        """
        if self.layout.shared_attr is None:
            return {}
        return _invert_shared(self._shared_objects())

    def _shared_objects(self) -> dict[int, Any]:
        """The decoded shared attribute's cached objects by cache ID; empty
        with no canonical file."""
        attr = self.layout.shared_attr
        if attr is None or not self.canonical_path.is_file():
            return {}
        cache = ResolvingDecodeCache(root=None, format="hdf5")
        self.read_attrs((attr,), decode_cache=cache)
        return dict(cache)

    def _shared_loader(self) -> Callable[[], Mapping[int, Any]] | None:
        """A loader of the shared objects that decodes them at most once,
        for one store call; None for a layout without a shared attribute."""
        if self.layout.shared_attr is None:
            return None
        loaded: dict[int, Any] | None = None

        def load() -> Mapping[int, Any]:
            nonlocal loaded
            if loaded is None:
                loaded = self._shared_objects()
            return loaded

        return load

    def _payload_seed(self) -> tuple[dict, int]:
        """The encode seed and ID floor for a payload, from the process's
        memo of the canonical file, rebuilt when the file is replaced.

        Raises `FileNotFoundError` with no canonical file and
        `CheckpointReadError` if the next-ID record or the shared
        attribute can't be read.
        """
        global _PAYLOAD_SEED
        path = self.canonical_path
        if not path.is_file():
            raise FileNotFoundError(
                f"The canonical file {path} does not exist; "
                "write_snapshot creates it"
            )
        key = str(path.resolve())
        memo = _PAYLOAD_SEED
        if memo is not None and memo[0] == (
            key,
            self._identity(os.stat(path)),
        ):
            return memo[1]
        attr = self.layout.shared_attr
        assert attr is not None
        with self._committed_object() as (group, cache):
            f = group.file
            identity = self._identity(os.fstat(f.id.get_vfd_handle()))
            floor = _read_next_id(f, path)
            result = decode_object_attrs(group, [attr], cache)
        if attr not in result:
            raise CheckpointReadError(path, f"attributes {[attr]} are missing")
        value = (_invert_shared(cache), floor)
        _PAYLOAD_SEED = ((key, identity), value)
        return value

    def _sources(self) -> list[_Source]:
        sources: list[_Source] = []
        if self.canonical_path.is_file():
            sources.append(_Source(self.canonical_path, self._open_canonical))
        shared = self._shared_loader()
        for path in self.payload_paths():
            sources.append(
                _Source(
                    path,
                    functools.partial(self._open_payload, path, shared),
                )
            )
        return sources

    @contextlib.contextmanager
    def _open_canonical(self) -> Iterator[_Opened]:
        path = self.canonical_path
        f = open_h5(path, "r")
        try:
            # Stat the open file, not the path, which a replacement could
            # already point at a different file.
            identity = self._identity(os.fstat(f.id.get_vfd_handle()))
            if (
                self._canonical_decode_cache is None
                or identity != self._canonical_identity
            ):
                self._canonical_decode_cache = ResolvingDecodeCache(
                    root=f, format="hdf5"
                )
                self._canonical_identity = identity
            cache = self._canonical_decode_cache
            cache._root = f
            yield _Opened(
                _resolve_checkpoint_object_group(f),
                _parse_marker(f),
                cache,
                str(path),
                canonical_names=True,
                require_primary=False,
            )
        finally:
            f.close()

    @contextlib.contextmanager
    def _open_payload(
        self,
        path: Path,
        shared: Callable[[], Mapping[int, Any]] | None = None,
    ) -> Iterator[_Opened]:
        f = open_h5(path, "r")
        try:
            yield _Opened(
                f,
                None,
                ResolvingDecodeCache(root=f, format="hdf5", fallback=shared),
                str(path),
                canonical_names=False,
                require_primary=True,
            )
        finally:
            f.close()

    # -- canonical appends and consolidation -------------------------------

    @staticmethod
    def _identity(stat: os.stat_result) -> tuple[int, int, int]:
        return stat.st_dev, stat.st_ino, stat.st_mtime_ns

    @contextlib.contextmanager
    def _canonical_writer(self) -> Iterator[_CanonicalWriter]:
        """Open the canonical file for appending, with any tail beyond the
        committed lengths trimmed first.

        After a normal exit the store records the file's new identity, so
        its own appends keep the canonical decode cache.
        """
        path = self.canonical_path
        if not path.is_file():
            raise FileNotFoundError(
                f"The canonical file {path} does not exist; "
                "write_snapshot creates it"
            )
        # Checked before the open, which may itself touch the file.
        if self._canonical_identity != self._identity(os.stat(path)):
            self._canonical_decode_cache = None
        f = open_h5(path, "a")
        try:
            try:
                limits = _parse_marker(f)
            except Exception as exc:
                raise CheckpointReadError(
                    path, f"{type(exc).__name__}: {exc}"
                ) from exc
            next_id = _read_next_id(f, path)
            container = _resolve_checkpoint_object_group(f)
            committed: dict[str, set[int]] = {}
            for streamed in self.layout.attrs:
                _trim_attr(
                    container,
                    streamed,
                    limits.get(streamed.canonical, 0),
                    str(path),
                )
                sides = _iterable_groups(container, streamed.canonical)
                keys: list[int] = []
                if sides is not None:
                    fmt = sides[0].attrs.get("storage_format", "groups")
                    keys = [
                        int(k)
                        for k in _read_iterable_side(sides[0], fmt, None)
                    ]
                committed[streamed.name] = set(keys)
            yield _CanonicalWriter(
                f, container, self.layout, limits, committed, next_id
            )
        finally:
            f.close()
        self._canonical_identity = self._identity(os.stat(path))

    def _put_canonical(
        self,
        entries: Mapping[int, Mapping[str, Any]],
        keys: tuple[int, ...],
        encode_cache: dict | None,
    ) -> None:
        # The caller's cache changes only once the batch has committed.
        work = _working_copy(encode_cache)
        with self._canonical_writer() as writer:
            added: dict[str, list[int]] = {}
            for streamed in self.layout.attrs:
                pairs = [
                    (key, entries[key][streamed.name])
                    for key in writer.new_keys(
                        streamed,
                        (k for k in keys if streamed.name in entries[k]),
                    )
                ]
                if pairs:
                    writer.append(streamed, pairs, work)
                    added[streamed.name] = [key for key, _ in pairs]
            writer.commit(added)
        if encode_cache is not None:
            encode_cache.update(work)

    def consolidate(
        self, *, encode_cache: dict | None = None
    ) -> tuple[int, ...]:
        """Merge every payload into the canonical file, then remove the
        merged payloads, the SWMR ledgers and orphaned temp files.

        Run it only when no writer is live. An unreadable payload raises
        `CheckpointReadError` and nothing is deleted. Returns the sorted
        keys that got at least one attr appended. A shot payload is
        decoded with the parent program stored in the canonical file, once
        per call.
        """
        payloads = self.payload_paths()
        exists = self.canonical_path.is_file()
        if payloads and not exists:
            raise FileNotFoundError(
                f"The canonical file {self.canonical_path} does not exist; "
                "write_snapshot creates it"
            )
        appended: set[int] = set()
        if exists:
            work = _working_copy(encode_cache)
            shared = self._shared_loader()
            with self._canonical_writer() as writer:
                for path in payloads:
                    appended.update(
                        self._merge_payload(writer, path, work, shared)
                    )
            if encode_cache is not None:
                encode_cache.update(work)
        self._remove_leftovers(payloads)
        return tuple(sorted(appended))

    def _merge_payload(
        self,
        writer: _CanonicalWriter,
        path: Path,
        work: dict,
        shared: Callable[[], Mapping[int, Any]] | None,
    ) -> set[int]:
        """Append one payload's uncommitted entries and commit them."""
        source = _Source(
            path, functools.partial(self._open_payload, path, shared)
        )
        added: dict[str, list[int]] = {}
        with self._opened(source, wrap_body=False) as opened:
            for streamed in self.layout.attrs:
                try:
                    position: dict[int, int] = {}
                    for pos, key in enumerate(opened.keys(streamed)):
                        position.setdefault(key, pos)
                except Exception as exc:
                    raise CheckpointReadError(
                        path, f"{type(exc).__name__}: {exc}"
                    ) from exc
                new = writer.new_keys(streamed, position)
                if not new:
                    continue
                writer.append(
                    streamed,
                    self._payload_pairs(opened, streamed, new, position),
                    work,
                )
                added[streamed.name] = new
            writer.commit(added)
        return {key for keys in added.values() for key in keys}

    @staticmethod
    def _payload_pairs(
        opened: _Opened,
        streamed: StreamedAttr,
        keys: list[int],
        position: Mapping[int, int],
    ) -> Iterator[tuple[int, Any]]:
        """One decoded `(key, value)` pair at a time; a decode failure is
        a `CheckpointReadError`, never confused with a write failure."""
        for key in keys:
            try:
                value = opened.read(streamed, position[key])
            except CheckpointReadError:
                raise
            except Exception as exc:
                raise CheckpointReadError(
                    opened.path, f"{type(exc).__name__}: {exc}"
                ) from exc
            yield key, value

    def _remove_leftovers(self, merged: Iterable[Path]) -> None:
        """Delete merged payloads, confirmed ledgers and orphaned temp
        files. A failed delete warns and keeps the file."""
        for path in merged:
            remove_transient_file(path)
        for path in self.ledger_paths():
            if is_swmr_ledger_file(path):
                remove_transient_file(path)
        if not self.directory.is_dir():
            return
        orphans = {_temp_path(self.canonical_path).name}
        for path in sorted(self.directory.iterdir()):
            is_worker_tmp = path.name.startswith(
                "worker_"
            ) and path.name.endswith(".h5.tmp")
            if (is_worker_tmp or path.name in orphans) and path.is_file():
                remove_transient_file(path)

    def nested(
        self,
        key: int,
        attr: str = "_program_results",
        layout: StoreLayout = SHOT_LAYOUT,
    ) -> _NestedStore:
        """A read-only store over the `ProgramResults` held at `key` in
        `attr`, located afresh on every read."""
        return _NestedStore(self, key, self._streamed(attr), layout)

    def payload_view(self, path: str | os.PathLike) -> _PayloadView:
        """A read-only store over the one payload file at `path`.

        The file is read as the store reads payloads. Raises `ValueError`
        if `path` is not one of this store's payload file names."""
        name = os.path.basename(os.fspath(path))
        if not self._name_patterns()[1].fullmatch(name):
            raise ValueError(f"{name!r} is not a payload file name")
        return _PayloadView(self, Path(path))


class _ReadOnlyStore(_ReadableStore):
    """A read-only view: every write, and further views, are refused."""

    def put_batch(self, *args: Any, **kwargs: Any) -> BatchReceipt:
        raise TypeError("a read-only view cannot be written")

    def consolidate(self, *args: Any, **kwargs: Any) -> tuple[int, ...]:
        raise TypeError("a read-only view cannot be consolidated")

    def write_snapshot(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("a read-only view cannot be written")

    def nested(self, *args: Any, **kwargs: Any) -> BaseCheckpointStore:
        raise TypeError("a read-only view cannot be nested")

    def payload_view(self, *args: Any, **kwargs: Any) -> _PayloadView:
        raise TypeError("a read-only view has no payload views")


class _PayloadView(_ReadOnlyStore):
    """A read-only view of one payload file of a `CheckpointStore`."""

    def __init__(self, parent: CheckpointStore, path: Path) -> None:
        self._parent = parent
        self._path = path
        self.layout = parent.layout

    def _sources(self) -> list[_Source]:
        return [
            _Source(
                self._path,
                functools.partial(
                    self._parent._open_payload,
                    self._path,
                    self._parent._shared_loader(),
                ),
            )
        ]


class _NestedStore(_ReadOnlyStore):
    """A read-only view of a `ProgramResults`-like object stored as one
    entry of an enclosing store. Every entry counts, because the enclosing
    entry is committed. It holds no open handle between calls."""

    def __init__(
        self,
        parent: CheckpointStore,
        key: int,
        parent_attr: StreamedAttr,
        layout: StoreLayout,
    ) -> None:
        self._parent = parent
        self._key = key
        self._parent_attr = parent_attr
        self.layout = layout
        self._positions: dict[str, dict[int, int]] = {}

    def _sources(self) -> list[_Source]:
        return [_Source(Path("<nested>"), self._open_nested)]

    @contextlib.contextmanager
    def _open_nested(self) -> Iterator[_Opened]:
        # The first source holding the key wins, as in a key index; later
        # sources are never opened.
        parent = self._parent
        with contextlib.ExitStack() as stack:
            pos = -1
            for source in parent._sources():
                outer = stack.enter_context(
                    parent._opened(source, validate=False)
                )
                try:
                    keys = outer.keys(self._parent_attr)
                except Exception as exc:
                    raise CheckpointReadError(
                        source.path, f"{type(exc).__name__}: {exc}"
                    ) from exc
                if self._key in keys:
                    pos = keys.index(self._key)
                    break
                stack.close()
            if pos < 0:
                raise KeyError(self._key)
            sides = outer.sides(self._parent_attr)
            assert sides is not None
            group = _resolve_checkpoint_object_group(sides[1][str(pos)])
            if group.attrs.get("cache_type") in ("copy", "reference"):
                raise CheckpointReadError(
                    outer.path,
                    f"entry {self._key} of "
                    f"{self._parent_attr.canonical!r} is a cache stub, "
                    "which a nested view can't read",
                )
            yield _Opened(
                group,
                None,
                outer.cache,
                outer.path,
                canonical_names=True,
                require_primary=False,
            )

    def get(
        self, key: int, attr: str | None = None, *, strict: bool = True
    ) -> Any:
        # One open of the enclosing file serves the lookup and the read.
        # Nothing is validated on open: a cut-short entry fails on its read.
        streamed = self._streamed(attr)
        try:
            with self._opened(self._sources()[0], validate=False) as opened:
                # A committed entry never changes, so a position found
                # once stays valid; a miss re-reads the keys.
                positions = self._positions.setdefault(streamed.name, {})
                if key not in positions:
                    positions.clear()
                    for pos, k in enumerate(opened.keys(streamed)):
                        positions.setdefault(k, pos)
                if key in positions:
                    return opened.read(streamed, positions[key])
        except CheckpointReadError as exc:
            if strict:
                raise
            raise KeyError(key) from exc
        raise KeyError(key)

    def read_attrs(
        self, names: Iterable[str], *, decode_cache: Any = None
    ) -> dict[str, Any]:
        """Decode the named non-streamed attributes of the nested object,
        with the enclosing file's decode cache.

        Streamed attributes are never touched: naming one raises
        `ValueError`. A missing enclosing entry, a missing name or a decode
        failure raises `CheckpointReadError` naming the enclosing file.
        """
        wanted = list(names)
        streamed_names = {s.canonical for s in self.layout.attrs}
        for name in wanted:
            if name in streamed_names:
                raise ValueError(
                    f"{name!r} is a streamed attribute; read it through "
                    "keys and get"
                )
        try:
            with self._open_nested() as opened:
                path = opened.path
                result = decode_object_attrs(
                    opened.container, wanted, opened.cache
                )
        except KeyError as exc:
            raise CheckpointReadError(
                self._parent.canonical_path,
                f"entry {self._key} of {self._parent_attr.canonical!r} "
                "is missing",
            ) from exc
        absent = [name for name in wanted if name not in result]
        if absent:
            raise CheckpointReadError(path, f"attributes {absent} are missing")
        return result


class MemoryCheckpointStore(BaseCheckpointStore):
    """An in-memory twin of `CheckpointStore`, holding values by reference.

    It keeps the first value stored for a key and attribute. `consolidate`
    and `write_snapshot` do nothing.
    """

    def __init__(self, layout: StoreLayout) -> None:
        self.layout = layout
        self._values: dict[str, dict[int, Any]] = {
            a.name: {} for a in layout.attrs
        }

    def _attr_name(self, attr: str | None) -> str:
        return self.layout.primary.name if attr is None else attr

    def _attr_values(self, attr: str | None) -> dict[int, Any]:
        name = self._attr_name(attr)
        self.layout.attr(name)
        return self._values[name]

    def put_batch(
        self,
        entries: Mapping[int, Mapping[str, Any]],
        *,
        worker_id: str | None = None,
        encode_cache: dict | None = None,
    ) -> BatchReceipt:
        if worker_id is not None and encode_cache is not None:
            raise ValueError(
                "encode_cache applies only to canonical appends, not to "
                "payload batches"
            )
        _validate_batch(self.layout, entries)
        for key in sorted(entries):
            for name, value in entries[key].items():
                self._values[name].setdefault(key, value)
        return BatchReceipt(tuple(sorted(entries)), None)

    def keys(self, attr: str | None = None, *, strict: bool = True) -> list:
        return sorted(self._attr_values(attr))

    def get(
        self, key: int, attr: str | None = None, *, strict: bool = True
    ) -> Any:
        values = self._attr_values(attr)
        if key not in values:
            raise KeyError(key)
        return values[key]

    def iter_items(
        self,
        attr: str | None = None,
        *,
        keys: Iterable[int] | None = None,
        strict: bool = True,
    ) -> Iterator[tuple[int, Any]]:
        values = self._attr_values(attr)
        wanted = sorted(values) if keys is None else sorted(set(keys))
        for key in wanted:
            if key in values:
                yield key, values[key]

    def consolidate(
        self, *, encode_cache: dict | None = None
    ) -> tuple[int, ...]:
        return ()

    def write_snapshot(
        self, obj: Serializable, *, keep_entries: bool = False
    ) -> None:
        return None

    def nested(
        self,
        key: int,
        attr: str = "_program_results",
        layout: StoreLayout = SHOT_LAYOUT,
    ) -> BaseCheckpointStore:
        raise TypeError("MemoryCheckpointStore has no nested view")
