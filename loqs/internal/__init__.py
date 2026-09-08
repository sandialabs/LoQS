#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Utility classes and functions for LoQS."""

import os
import socket
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None  # type: ignore

from .serializable import (
    Serializable,
    SERIALIZATION_VERSION,
    IncorrectDecodableTypeError,
    MisformedDecodableError,
)

# Must be after Serializable
from .displayable import Displayable


def worker_id() -> str:
    """Return this process's `hostname_pid` worker identity string, used to
    key per-writer checkpoint files across LoQS's parallel dispatch
    mechanisms."""
    return f"{socket.gethostname()}_{os.getpid()}"


def pin_worker_threads() -> None:
    """Pin this process's numerical-library thread pools to one thread.

    The primary, always-correct layer of the thread-oversubscription
    discipline every chunk-processing worker entry point must apply as
    its first action, regardless of which executor backend runs it:
    environment variables (`OMP_NUM_THREADS`, etc.) only help if set
    before the relevant library first initializes its own thread pool,
    which isn't guaranteed for a worker process that already imported
    `numpy`/`pygsti`-adjacent code before reaching this call. Meant to be
    called directly inside a plain, module-level worker function -- not
    built via a decorator, since a decorator would return a closure that
    plain `pickle` (needed for `mpi4py.futures.MPIPoolExecutor`) can't
    resolve by dotted import path.
    """
    if threadpool_limits is not None:
        threadpool_limits(1)
    else:
        warnings.warn(
            "threadpoolctl is not installed, so worker thread pools "
            "cannot be limited to avoid oversubscription. Install "
            "loqs[parallel] or loqs[mpi]."
        )


def _retry_hdf5_write(
    worker_file_path: Path,
    write_fn: Callable[[h5py.File], None],
    max_retries: int = 5,
) -> None:
    """Open `worker_file_path` in append mode and call `write_fn(f)`, retrying with
    exponential backoff on transient HDF5 locking errors (`BlockingIOError`/`OSError`).
    """
    for attempt in range(max_retries):
        try:
            with h5py.File(worker_file_path, "a") as f:
                write_fn(f)
            break
        except (BlockingIOError, OSError):
            if attempt < max_retries - 1:
                time.sleep(0.01 * (2**attempt))
            else:
                raise


def _retry_hdf5_read(
    filename: Path,
    read_fn: Callable[[h5py.File], Any],
    max_retries: int = 5,
) -> Any:
    """Open `filename` read-only and call `read_fn(f)`, retrying with the
    same exponential backoff as `_retry_hdf5_write` on transient HDF5
    locking errors (`BlockingIOError`/`OSError`). Kept separate from that
    helper since it can't use its append-mode-only open.
    """
    for attempt in range(max_retries):
        try:
            with h5py.File(filename, "r") as f:
                return read_fn(f)
        except (BlockingIOError, OSError):
            if attempt < max_retries - 1:
                time.sleep(0.01 * (2**attempt))
            else:
                raise
