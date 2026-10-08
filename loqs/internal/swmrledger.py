#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Single-writer/multi-reader (SWMR) HDF5 progress ledger primitives.

Provides a small set of datasets, kept in a dedicated `/swmr_ledger`
subgroup of an HDF5 file, that one writer process can update in place
while any number of reader processes observe its progress live via
HDF5's SWMR mode -- without either side needing a file lock. A caller
picks which of the six available fields it needs (an item-level ledger
wants all six; a shot-level ledger typically omits the in-flight-item
fields), so this module never assumes a fixed schema.

SWMR relies on the file system preserving POSIX write ordering, so
ledgers need a local disk or a parallel file system such as Lustre or
GPFS, not an NFS or SMB share. Ledger opens turn HDF5 file locking off:
SWMR's guarantees don't depend on it, and the locks are what made a
writer's open fail while a reader was polling the ledger.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import h5py
import numpy as np

DEFAULT_GROUP_NAME = "swmr_ledger"
DEFAULT_CHUNK_SIZE = 1024

# Writer handles shared by nested `held_swmr_writer` holders in this
# process, keyed by resolved ledger path: [file, ledger group, hold count].
_HELD_WRITERS: dict[Path, list] = {}
_HELD_WRITERS_LOCK = threading.Lock()

# Datasets that grow to accommodate an index beyond their current capacity
# (loky distributes work dynamically, so a resumed run's assigned indices
# aren't known upfront), keyed by (dtype, fill value).
_GROWABLE_FIELD_SPECS: dict[str, tuple[type, Any]] = {
    "done": (bool, False),
    "wall_clock_times": (np.float64, np.nan),
}

# Fixed-size, shape-(1,) scalar-holder datasets, keyed by (dtype, fill value).
_SCALAR_FIELD_SPECS: dict[str, tuple[type, Any]] = {
    "current_item_index": (np.int64, 0),
    "item_shots_done": (np.int64, 0),
    "item_shots_total": (np.int64, 0),
    "last_heartbeat": (np.float64, np.nan),
}


@dataclass(frozen=True)
class SwmrLedgerSnapshot:
    """A point-in-time read of a SWMR progress ledger's fields.

    Returned by [](api:read_swmr_ledger_status). Only fields actually
    present in the ledger group being read are populated; every other
    field stays `None`, since a ledger may be item-level (all six
    fields) or shot-level (a smaller subset).
    """

    done: np.ndarray | None = None
    wall_clock_times: np.ndarray | None = None
    current_item_index: int | None = None
    item_shots_done: int | None = None
    item_shots_total: int | None = None
    last_heartbeat: float | None = None


def init_swmr_ledger(
    target: h5py.File | h5py.Group,
    fields: Sequence[str],
    capacity: int = 0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    group_name: str = DEFAULT_GROUP_NAME,
) -> h5py.Group:
    """Create the requested ledger datasets under `target`'s `group_name`
    subgroup, if not already present.

    Must be called before SWMR mode is enabled on the file, since dataset
    creation is a structural change HDF5 doesn't allow once SWMR write
    mode is active.

    Parameters
    ----------
    target : h5py.File | h5py.Group
        The group (or file root) to hold the ledger subgroup.
    fields : Sequence[str]
        Which ledger fields to create -- any of `"done"`,
        `"wall_clock_times"`, `"current_item_index"`, `"item_shots_done"`,
        `"item_shots_total"`, `"last_heartbeat"`.
    capacity : int, optional
        Initial length for the growable (`done`/`wall_clock_times`)
        datasets. Default is 0 (grown lazily on first write).
    chunk_size : int, optional
        Chunk length for the growable datasets. Default is
        `DEFAULT_CHUNK_SIZE`.
    group_name : str, optional
        Name of the ledger subgroup. Default is `DEFAULT_GROUP_NAME`.

    Returns
    -------
    h5py.Group
        The ledger subgroup, containing the requested datasets.

    Raises
    ------
    ValueError
        If `fields` contains a name that isn't a recognized ledger field.
    """
    ledger_group = target.require_group(group_name)
    for field in fields:
        if field in ledger_group:
            continue
        if field in _GROWABLE_FIELD_SPECS:
            dtype, fill_value = _GROWABLE_FIELD_SPECS[field]
            ledger_group.create_dataset(
                field,
                shape=(capacity,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(chunk_size,),
                fillvalue=fill_value,
            )
        elif field in _SCALAR_FIELD_SPECS:
            dtype, fill_value = _SCALAR_FIELD_SPECS[field]
            ledger_group.create_dataset(
                field,
                shape=(1,),
                dtype=dtype,
                chunks=(1,),
                fillvalue=fill_value,
            )
        else:
            raise ValueError(
                f"Unknown SWMR ledger field {field!r}; expected one of "
                f"{sorted({**_GROWABLE_FIELD_SPECS, **_SCALAR_FIELD_SPECS})}"
            )
    return ledger_group


def open_swmr_writer(
    path: Path,
    fields: Sequence[str],
    capacity: int = 0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    group_name: str = DEFAULT_GROUP_NAME,
) -> tuple[h5py.File, h5py.Group]:
    """Open (creating if necessary) `path` as the single SWMR writer,
    initializing the ledger group and enabling SWMR write mode.

    `libver="latest"` is required for SWMR to be available at all; the
    ledger's datasets are created via `init_swmr_ledger` before the file
    is opened in SWMR write mode, since dataset creation is not permitted
    once it's active.

    A new ledger is built under `path` plus a `.tmp` suffix, closed, and
    renamed onto `path` (an open file can't be renamed on Windows), so a
    poller never sees a half-built ledger. If anything fails before the
    rename, the temporary file is closed and removed; a failure in the
    reopen that follows leaves a valid, closed ledger at `path`.

    Both a new and an existing ledger end in a direct SWMR-write open of
    `path` with HDF5 file locking off, so a concurrent SWMR reader can't
    make the open fail. An existing ledger must already hold every
    requested field; `capacity` and `chunk_size` only apply to a new one.

    Parameters
    ----------
    path : Path
        Path to the ledger's HDF5 file.
    fields : Sequence[str]
        Which ledger fields to create -- see `init_swmr_ledger`.
    capacity : int, optional
        Initial length for the growable datasets. Default is 0.
    chunk_size : int, optional
        Chunk length for the growable datasets. Default is
        `DEFAULT_CHUNK_SIZE`.
    group_name : str, optional
        Name of the ledger subgroup. Default is `DEFAULT_GROUP_NAME`.

    Returns
    -------
    tuple[h5py.File, h5py.Group]
        The open file (in SWMR write mode) and its ledger subgroup.

    Raises
    ------
    ValueError
        If `fields` names an unknown field, or an existing ledger lacks
        one of `fields`.
    """
    path = Path(path)
    if path.exists():
        return _reopen_swmr_writer(path, fields, group_name)

    tmp_path = path.with_name(path.name + ".tmp")
    f: h5py.File | None = None
    try:
        f = h5py.File(tmp_path, "w", libver="latest")
        init_swmr_ledger(
            f,
            fields,
            capacity=capacity,
            chunk_size=chunk_size,
            group_name=group_name,
        )
        f.close()
        f = None
        os.replace(tmp_path, path)
    except BaseException:
        if f is not None:
            f.close()
        tmp_path.unlink(missing_ok=True)
        raise
    return _reopen_swmr_writer(path, fields, group_name)


def _reopen_swmr_writer(
    path: Path,
    fields: Sequence[str],
    group_name: str,
) -> tuple[h5py.File, h5py.Group]:
    """Open the existing ledger at `path` directly in SWMR write mode, with
    HDF5 file locking off, and check that it holds every requested field.

    The close degree stays at h5py's default, since HDF5 refuses a second
    open of a file in the same process whose close degree differs.

    Raises
    ------
    ValueError
        If the ledger group or any of `fields` is missing; datasets can't
        be created once SWMR write mode is active.
    """
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
    fapl.set_file_locking(False, True)
    fid = h5py.h5f.open(
        os.fsencode(path),
        h5py.h5f.ACC_RDWR | h5py.h5f.ACC_SWMR_WRITE,
        fapl=fapl,
    )
    f = h5py.File(fid)
    group = f.get(group_name)
    if not isinstance(group, h5py.Group):
        missing = [group_name]
    else:
        missing = [field for field in fields if field not in group]
    if missing:
        f.close()
        raise ValueError(
            f"SWMR ledger {str(path)!r} is missing {missing}, which can't be "
            "created under SWMR write mode"
        )
    return f, group


@contextmanager
def held_swmr_writer(
    path: Path,
    fields: Sequence[str],
) -> Iterator[h5py.Group]:
    """Hold `path` open as this process's SWMR writer for the `with` block,
    yielding its ledger group.

    Holders of the same ledger (keyed by resolved path) share one handle,
    opened via `open_swmr_writer` by the first holder and closed when the
    outermost holder exits. An enclosing `with` therefore keeps one handle
    across many inner uses (for example a loop of `checkpoint()` calls),
    while a lone use closes its handle before returning. Thread-safe.
    A reader of the same ledger open in this process when the handle is
    opened makes the open fail, so thread-based executors must not poll
    ledgers that their own threads write.

    Parameters
    ----------
    path : Path
        Path to the ledger's HDF5 file, created if absent.
    fields : Sequence[str]
        Which ledger fields it holds -- see `open_swmr_writer`.

    Yields
    ------
    h5py.Group
        The ledger subgroup, in SWMR write mode.
    """
    key = Path(path).resolve()
    with _HELD_WRITERS_LOCK:
        entry = _HELD_WRITERS.get(key)
        if entry is None:
            f, group = open_swmr_writer(key, fields)
            entry = [f, group, 0]
            _HELD_WRITERS[key] = entry
        entry[2] += 1
    try:
        yield entry[1]
    finally:
        with _HELD_WRITERS_LOCK:
            entry[2] -= 1
            if entry[2] == 0:
                del _HELD_WRITERS[key]
                entry[0].close()


def open_swmr_reader(
    path: Path,
    group_name: str = DEFAULT_GROUP_NAME,
) -> tuple[h5py.File, h5py.Group]:
    """Open an existing ledger file for SWMR read access.

    Parameters
    ----------
    path : Path
        Path to the ledger's HDF5 file. Must already exist, with SWMR
        write mode already enabled by its writer.
    group_name : str, optional
        Name of the ledger subgroup. Default is `DEFAULT_GROUP_NAME`.

    Returns
    -------
    tuple[h5py.File, h5py.Group]
        The open file (in SWMR read mode) and its ledger subgroup.
    """
    f = h5py.File(path, "r", libver="latest", swmr=True, locking=False)
    return f, f[group_name]


def is_swmr_ledger_file(path: Path) -> bool:
    """Whether `path` is a SWMR ledger file (holding a `swmr_ledger` group)
    rather than a legacy checkpoint file with its data at its own root.

    Opens like `open_swmr_reader` (SWMR read, file locking off), so the
    check succeeds while the ledger's writer still holds it open. Any open
    or read error counts as "not a ledger", so the caller's own handling
    of that file still applies.
    """
    try:
        with h5py.File(
            path, "r", libver="latest", swmr=True, locking=False
        ) as f:
            return DEFAULT_GROUP_NAME in f
    except (BlockingIOError, OSError):
        return False


def read_swmr_ledger_done_union(
    directory: Path,
    pattern: str,
    group_name: str = DEFAULT_GROUP_NAME,
) -> set[int]:
    """Return the union of `done` indices across the ledger files in
    `directory` whose names match the glob `pattern`.

    Each file is opened here, in SWMR read mode, and closed again before
    the next one. A file that fails to open, or has no ledger group with a
    `done` dataset (a ledger still being created, or a non-ledger file
    sharing the name pattern), is skipped.

    Parameters
    ----------
    directory : Path
        Directory to scan. A missing directory yields an empty set.
    pattern : str
        Glob pattern, relative to `directory`, naming the ledger files.
    group_name : str, optional
        Name of the ledger subgroup. Default is `DEFAULT_GROUP_NAME`.

    Returns
    -------
    set[int]
        Every index marked done in at least one matching ledger.
    """
    done: set[int] = set()
    for path in sorted(Path(directory).glob(pattern)):
        try:
            with h5py.File(
                path, "r", libver="latest", swmr=True, locking=False
            ) as f:
                group = f.get(group_name)
                if not isinstance(group, h5py.Group) or "done" not in group:
                    continue
                done_dataset = group["done"]
                done_dataset.refresh()
                done.update(np.flatnonzero(done_dataset[()]).tolist())
        except (OSError, KeyError):
            continue
    return done


def mark_ledger_item_done(
    ledger_group: h5py.Group,
    index: int,
    wall_clock_time: float,
    auto_flush: bool = True,
) -> None:
    """Record `index` as done in the `done`/`wall_clock_times` datasets,
    auto-extending both (geometric growth) if `index` is beyond their
    current capacity.

    Extension is required, not optional: loky distributes work
    dynamically, so a resumed run processes a non-contiguous set of
    indices whose total count isn't knowable upfront.

    Parameters
    ----------
    ledger_group : h5py.Group
        The ledger subgroup, as returned by `open_swmr_writer`.
    index : int
        The item/shot index to mark done.
    wall_clock_time : float
        The wall-clock duration to record for `index`.
    auto_flush : bool, optional
        Whether to flush the affected datasets immediately, making the
        write visible to a SWMR reader's next `.refresh()`. Default is
        True.
    """
    done_dataset = ledger_group["done"]
    times_dataset = ledger_group["wall_clock_times"]

    capacity = done_dataset.shape[0]
    if index >= capacity:
        new_capacity = max(capacity * 2, index + 1)
        done_dataset.resize((new_capacity,))
        times_dataset.resize((new_capacity,))

    done_dataset[index] = True
    times_dataset[index] = wall_clock_time

    if auto_flush:
        done_dataset.flush()
        times_dataset.flush()


def update_ledger_in_flight(
    ledger_group: h5py.Group,
    item_index: int,
    shots_done: int = 0,
    shots_total: int = 0,
    auto_flush: bool = True,
) -> None:
    """Update the in-flight-item fields (`current_item_index`,
    `item_shots_done`, `item_shots_total`).

    When `item_index` differs from the currently stored index,
    `item_shots_done` is reset to 0 and flushed *before* `current_item_index`
    is advanced and flushed -- so a reader can never observe a new item
    index paired with a stale, misleadingly-high shot count carried over
    from the previous item.

    Parameters
    ----------
    ledger_group : h5py.Group
        The ledger subgroup, as returned by `open_swmr_writer`.
    item_index : int
        The item index now in flight.
    shots_done : int, optional
        Shots completed so far for `item_index`. Default is 0.
    shots_total : int, optional
        Total shots planned for `item_index`. Default is 0.
    auto_flush : bool, optional
        Whether to flush the affected datasets immediately, making the
        write visible to a SWMR reader's next `.refresh()`. Default is
        True.
    """
    index_dataset = ledger_group["current_item_index"]
    shots_done_dataset = ledger_group["item_shots_done"]
    shots_total_dataset = ledger_group["item_shots_total"]

    if int(index_dataset[0]) != item_index:
        shots_done_dataset[0] = 0
        if auto_flush:
            shots_done_dataset.flush()

        index_dataset[0] = item_index
        if auto_flush:
            index_dataset.flush()

    shots_done_dataset[0] = shots_done
    shots_total_dataset[0] = shots_total
    if auto_flush:
        shots_done_dataset.flush()
        shots_total_dataset.flush()


def update_ledger_heartbeat(
    ledger_group: h5py.Group,
    timestamp: float | None = None,
    auto_flush: bool = True,
) -> None:
    """Write the current heartbeat time to the `last_heartbeat` dataset.

    Parameters
    ----------
    ledger_group : h5py.Group
        The ledger subgroup, as returned by `open_swmr_writer`.
    timestamp : float | None, optional
        Heartbeat time to record. If None (default), uses `time.time()`.
    auto_flush : bool, optional
        Whether to flush the affected dataset immediately, making the
        write visible to a SWMR reader's next `.refresh()`. Default is
        True.
    """
    heartbeat_dataset = ledger_group["last_heartbeat"]
    heartbeat_dataset[0] = timestamp if timestamp is not None else time.time()
    if auto_flush:
        heartbeat_dataset.flush()


def refresh_swmr_ledger(ledger_group: h5py.Group) -> None:
    """Refresh every dataset present in `ledger_group`, pulling in the
    latest writer-flushed chunks under SWMR read mode.

    Parameters
    ----------
    ledger_group : h5py.Group
        The ledger subgroup, as returned by `open_swmr_reader`.
    """
    for name in ledger_group:
        dataset = ledger_group[name]
        if isinstance(dataset, h5py.Dataset):
            dataset.refresh()


def read_swmr_ledger_status(
    ledger_group: h5py.Group,
    refresh: bool = True,
) -> SwmrLedgerSnapshot:
    """Build a `SwmrLedgerSnapshot` from `ledger_group`'s current values.

    Only fields actually present in `ledger_group` are populated on the
    returned snapshot, since a ledger may be item-level or shot-level.

    Parameters
    ----------
    ledger_group : h5py.Group
        The ledger subgroup, as returned by `open_swmr_writer` or
        `open_swmr_reader`.
    refresh : bool, optional
        Whether to call `refresh_swmr_ledger` first, so the snapshot
        reflects the writer's latest flushed state. Default is True.

    Returns
    -------
    SwmrLedgerSnapshot
        The current ledger state, with absent fields left as `None`.
    """
    if refresh:
        refresh_swmr_ledger(ledger_group)

    fields = {}
    if "done" in ledger_group:
        fields["done"] = ledger_group["done"][()]
    if "wall_clock_times" in ledger_group:
        fields["wall_clock_times"] = ledger_group["wall_clock_times"][()]
    if "current_item_index" in ledger_group:
        fields["current_item_index"] = int(
            ledger_group["current_item_index"][0]
        )
    if "item_shots_done" in ledger_group:
        fields["item_shots_done"] = int(ledger_group["item_shots_done"][0])
    if "item_shots_total" in ledger_group:
        fields["item_shots_total"] = int(ledger_group["item_shots_total"][0])
    if "last_heartbeat" in ledger_group:
        fields["last_heartbeat"] = float(ledger_group["last_heartbeat"][0])

    return SwmrLedgerSnapshot(**fields)
