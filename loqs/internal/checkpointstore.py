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

from loqs.internal.encoder.hdf5encoder import HDF5Encoder
from loqs.internal.serializable import ResolvingDecodeCache, Serializable
from loqs.internal.streamingmerge import (
    _decode_group_entry,
    _read_iterable_side,
    _reset_empty_groups_format_dict_attr,
    _resolve_checkpoint_object_group,
    merge_dict_attr,
    remove_transient_file,
)
from loqs.internal.swmrledger import is_swmr_ledger_file

COMMIT_MARKER_ATTR = "checkpoint_store_committed"
"""Name of the file-root attribute holding the committed lengths."""


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
    """

    canonical_filename: str
    ledger_tag: str
    payload_tag: str
    attrs: tuple[StreamedAttr, ...]

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
)
"""Shot-level layout, as `ProgramResults` checkpoints today."""

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
    def write_snapshot(self, obj: Serializable) -> None:
        """Atomically write the canonical file from `obj`."""

    @abstractmethod
    def nested(
        self,
        key: int,
        attr: str = "_program_results",
        layout: StoreLayout = SHOT_LAYOUT,
    ) -> BaseCheckpointStore:
        """A read-only view of the object stored at `key` in `attr`."""


@contextlib.contextmanager
def atomic_h5_file(path: str | os.PathLike) -> Iterator[h5py.File]:
    """Write an HDF5 file atomically.

    Opens `<path>.tmp` in mode `"w"` and yields it. On a normal exit the
    file is closed and renamed onto `path`. On any `BaseException` the file
    is closed, the temp file is removed and the exception propagates.
    """
    final = Path(path)
    temp = final.with_name(final.name + ".tmp")
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
    None if the attr is absent."""
    if canonical_name not in container:
        return None
    sides = container[canonical_name]["dict"]
    return sides["keys"]["iterable"], sides["values"]["iterable"]


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
        all_keys = _read_iterable_side(keys_group, fmt, None)
        return [int(k) for k in all_keys[: self.count(streamed)]]

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
            # A decode cut short leaves `DeferredRef` placeholders behind,
            # which later references would resolve to.
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
        self, source: _Source, *, wrap_body: bool = True
    ) -> Iterator[_Opened]:
        """Open and validate a source; any failure becomes a
        `CheckpointReadError` naming it. With `wrap_body=False`, errors
        raised by the caller's own `with` body propagate unchanged."""
        in_body = False
        try:
            with source.opener() as opened:
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


def _parse_marker(f: h5py.File) -> dict[str, int]:
    """The committed lengths in a canonical file's marker; `ValueError` if
    the marker is missing."""
    marker = f.attrs.get(COMMIT_MARKER_ATTR)
    if marker is None:
        raise ValueError("the canonical file has no commit marker")
    if isinstance(marker, bytes):
        marker = marker.decode("utf-8")
    return {k: int(v) for k, v in json.loads(marker).items()}


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
        """Append `(key, value)` pairs, uncommitted until `commit`."""
        merge_dict_attr(
            self.container,
            streamed.canonical,
            pairs,
            encode_cache=encode_cache,
            key_use_dataset=True,
            value_use_dataset=streamed.use_dataset,
        )

    def commit(self, added: Mapping[str, Iterable[int]]) -> None:
        """Overwrite the marker, in place, with the physical lengths."""
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

    # -- writing ----------------------------------------------------------

    def write_snapshot(self, obj: Serializable) -> None:
        """Atomically write the canonical file from `obj`.

        Inside the temp file and before the rename, empty groups-format
        streamed attributes are removed, every fixed-size dataset side is
        made growable and the commit marker is written from the physical
        lengths.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        with atomic_h5_file(self.canonical_path) as f:
            obj.dump(f, "hdf5")
            for streamed in self.layout.attrs:
                _reset_empty_groups_format_dict_attr(f, streamed.canonical)
            container = _resolve_checkpoint_object_group(f)
            lengths: dict[str, int] = {}
            for streamed in self.layout.attrs:
                sides = _iterable_groups(container, streamed.canonical)
                if sides is None:
                    lengths[streamed.canonical] = 0
                    continue
                for side in sides:
                    _make_side_growable(side)
                lengths[streamed.canonical] = side_length(sides[0])
            f.attrs[COMMIT_MARKER_ATTR] = json.dumps(lengths)
        self._canonical_decode_cache = None
        self._canonical_identity = None

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
        self.directory.mkdir(parents=True, exist_ok=True)
        # One fresh cache per payload keeps it self-contained.
        cache: dict = {}
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

    def _sources(self) -> list[_Source]:
        sources: list[_Source] = []
        if self.canonical_path.is_file():
            sources.append(_Source(self.canonical_path, self._open_canonical))
        for path in self.payload_paths():
            sources.append(
                _Source(path, functools.partial(self._open_payload, path))
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
    def _open_payload(self, path: Path) -> Iterator[_Opened]:
        f = open_h5(path, "r")
        try:
            yield _Opened(
                f,
                None,
                ResolvingDecodeCache(root=f, format="hdf5"),
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
                f, container, self.layout, limits, committed
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
        keys that got at least one attr appended.
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
            with self._canonical_writer() as writer:
                for path in payloads:
                    appended.update(self._merge_payload(writer, path, work))
            if encode_cache is not None:
                encode_cache.update(work)
        self._remove_leftovers(payloads)
        return tuple(sorted(appended))

    def _merge_payload(
        self, writer: _CanonicalWriter, path: Path, work: dict
    ) -> set[int]:
        """Append one payload's uncommitted entries and commit them."""
        source = _Source(path, functools.partial(self._open_payload, path))
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
        orphans = {self.canonical_path.name + ".tmp"}
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
    ) -> BaseCheckpointStore:
        """A read-only store over the `ProgramResults` held at `key` in
        `attr`, located afresh on every read."""
        return _NestedStore(self, key, self._streamed(attr), layout)


class _NestedStore(_ReadableStore):
    """A read-only view of a `ProgramResults`-like object stored as one
    entry of an enclosing store. Every entry counts, because the enclosing
    entry is committed. It holds no open handle between calls."""

    def __init__(
        self,
        parent: _ReadableStore,
        key: int,
        parent_attr: StreamedAttr,
        layout: StoreLayout,
    ) -> None:
        self._parent = parent
        self._key = key
        self._parent_attr = parent_attr
        self.layout = layout

    def _sources(self) -> list[_Source]:
        return [_Source(Path("<nested>"), self._open_nested)]

    @contextlib.contextmanager
    def _open_nested(self) -> Iterator[_Opened]:
        parent = self._parent
        index = parent._key_index(self._parent_attr, True)
        if self._key not in index:
            raise KeyError(self._key)
        source, pos = index[self._key]
        with parent._opened(source) as outer:
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

    def put_batch(self, *args: Any, **kwargs: Any) -> BatchReceipt:
        raise TypeError("a nested view is read-only")

    def consolidate(self, *args: Any, **kwargs: Any) -> tuple[int, ...]:
        raise TypeError("a nested view is read-only")

    def write_snapshot(self, obj: Serializable) -> None:
        raise TypeError("a nested view is read-only")

    def nested(self, *args: Any, **kwargs: Any) -> BaseCheckpointStore:
        raise TypeError("a nested view cannot be nested further")


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

    def write_snapshot(self, obj: Serializable) -> None:
        return None

    def nested(
        self,
        key: int,
        attr: str = "_program_results",
        layout: StoreLayout = SHOT_LAYOUT,
    ) -> BaseCheckpointStore:
        raise TypeError("MemoryCheckpointStore has no nested view")
