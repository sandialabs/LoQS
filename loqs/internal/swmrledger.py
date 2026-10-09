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

SWMR relies on the file system preserving POSIX write ordering. Without
it (an NFS or SMB share), live progress reads can stall or show wrong
counts; results never depend on ledger contents, so they stay correct.
Use a local disk or a parallel file system such as Lustre or GPFS for
reliable live progress. Ledger opens turn HDF5 file locking off:
SWMR's guarantees don't depend on it, and the locks are what made a
writer's open fail while a reader was polling the ledger.

HDF5 drops its own lock once a SWMR-write open completes, so a writer
also holds an exclusive `flock` on its ledger until it closes the file
(best effort: an open never fails because the lock couldn't be taken).
A killed writer leaves HDF5's "open for write" flag set, which fails
every open but an SWMR read. When a writer open fails, this process
holds no handle on the file and no live writer holds the `flock`, the
ledger is rebuilt without the flag (copied under a `.tmp` name and
renamed into place) and reopened, so an explicit `worker_id` can be
reused after a crash. Two gaps remain: where `flock` isn't coherent
across nodes (Lustre `localflock`, NFS), two live writers on different
nodes sharing one explicit worker id aren't detected; and on Windows
there is no recovery.
"""

from __future__ import annotations

import os
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import h5py
import numpy as np

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]

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

    The returned file holds an exclusive `flock` on `path` until its
    `close()`. If the open fails because a killed writer left its "open
    for write" flag set, the ledger is rebuilt without the flag and
    reopened, unless this process holds a handle on it or a live writer
    holds the lock; then the original `OSError` is raised.

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
    OSError
        If an existing ledger can't be opened and isn't recoverable.
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


def _try_lock_exclusive(path: Path) -> int | None:
    """Take a non-blocking exclusive `flock` on `path`, returning the file
    descriptor that holds it, or `None` if the lock can't be taken (held
    elsewhere, unsupported, open failed, or no `fcntl`).

    `flock` locks belong to the open file description and the OS drops
    them when the process dies. Record locks (`lockf`) would instead drop
    whenever any descriptor of the file closes, as a SWMR reader's does.
    """
    if fcntl is None:
        return None
    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDONLY)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except OSError:
        if fd is not None:
            os.close(fd)
        return None


class _LockedWriterFile(h5py.File):
    """An `h5py.File` that also owns the writer lock's file descriptor.

    `close()` releases the lock. The release runs once, so an explicit
    close and the garbage-collection backstop can't close a descriptor
    number the OS has since reused.
    """

    def __init__(self, fid: Any, lock_fd: int | None) -> None:
        super().__init__(fid)
        self._lock_release = (
            weakref.finalize(self, os.close, lock_fd)
            if lock_fd is not None
            else None
        )

    def close(self) -> None:
        try:
            super().close()
        finally:
            release = getattr(self, "_lock_release", None)
            if release is not None:
                release()


def _open_swmr_write_fid(path: Path) -> Any:
    """Open `path` in SWMR write mode with HDF5 file locking off."""
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
    fapl.set_file_locking(False, True)
    return h5py.h5f.open(
        os.fsencode(path),
        h5py.h5f.ACC_RDWR | h5py.h5f.ACC_SWMR_WRITE,
        fapl=fapl,
    )


def _process_holds_file(path: Path) -> bool:
    """Whether this process has an HDF5 file handle open on `path`."""
    for fid in h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE):
        try:
            name = os.fsdecode(h5py.h5f.get_name(fid))
            if os.path.samefile(name, path):
                return True
        except (OSError, ValueError):
            continue
    return False


def _rebuild_ledger(path: Path, tmp_path: Path) -> None:
    """Copy `path`'s root members and attributes into a new file at
    `tmp_path`, which has no stale writer flag."""
    with h5py.File(
        path, "r", libver="latest", swmr=True, locking=False
    ) as src:
        with h5py.File(tmp_path, "w", libver="latest") as dst:
            for name in src:
                src.copy(name, dst, name=name)
            for key, value in src.attrs.items():
                dst.attrs[key] = value


def _recover_stale_ledger(
    path: Path, original: OSError
) -> tuple[Any, int | None]:
    """Recover a ledger whose SWMR-write open failed with `original`, and
    return the retried open's file id and writer lock descriptor.

    A killed writer leaves HDF5's "open for write" flag set, which fails
    every open but an SWMR read. If this process holds no handle on `path`
    and no live writer holds its `flock`, the ledger is rebuilt under a
    `.tmp` name and renamed onto `path`. Otherwise, or if any step fails,
    `original` is raised (chained from the failure, if any).
    """
    if _process_holds_file(path):
        raise original
    old_lock = _try_lock_exclusive(path)
    if old_lock is None:
        raise original

    tmp_path = path.with_name(path.name + ".tmp")
    new_lock: int | None = None
    failure: Exception | None = None
    try:
        try:
            _rebuild_ledger(path, tmp_path)
            new_lock = _try_lock_exclusive(tmp_path)
            os.replace(tmp_path, path)
        except Exception as err:
            failure = err
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            if new_lock is not None:
                os.close(new_lock)
            raise
    finally:
        os.close(old_lock)
    if failure is not None:
        tmp_path.unlink(missing_ok=True)
        if new_lock is not None:
            os.close(new_lock)
        raise original from failure

    try:
        return _open_swmr_write_fid(path), new_lock
    except OSError as err:
        if new_lock is not None:
            os.close(new_lock)
        raise err from original


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
    try:
        fid = _open_swmr_write_fid(path)
        lock_fd = _try_lock_exclusive(path)
    except OSError as err:
        fid, lock_fd = _recover_stale_ledger(path, err)
    f = _LockedWriterFile(fid, lock_fd)
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

    Raises
    ------
    KeyError
        If the file has no ledger subgroup. The file is closed first.
    """
    f = h5py.File(path, "r", libver="latest", swmr=True, locking=False)
    try:
        return f, f[group_name]
    except BaseException:
        # e.g. KeyError for a file without the ledger group: don't leak
        # the handle.
        f.close()
        raise


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
    paths: Iterable[str | os.PathLike],
    group_name: str = DEFAULT_GROUP_NAME,
) -> set[int]:
    """Return the union of `done` indices across the ledger files at `paths`.

    Each file is opened in SWMR read mode and closed before the next. A
    file that fails to open or has no ledger group with a `done` dataset
    is skipped.

    Parameters
    ----------
    paths : iterable of path-like
        The ledger files to read; a path that doesn't exist is skipped.
    group_name : str, optional
        Name of the ledger subgroup. Default is `DEFAULT_GROUP_NAME`.

    Returns
    -------
    set[int]
        Every index marked done in at least one matching ledger.
    """
    done: set[int] = set()
    for path in paths:
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
