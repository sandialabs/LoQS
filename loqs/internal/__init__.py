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
import uuid
import warnings

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

_worker_id_cache: tuple[int, str] | None = None


def worker_id() -> str:
    """Return this process's cached `hostname_pid_suffix` worker identity
    string, used to key per-writer checkpoint files across LoQS's parallel
    dispatch mechanisms.

    The suffix is a short UUID4-derived string, appended so that no two
    workers ever resolve to the same identity even if the OS reuses a PID
    across process launches on the same host. The result is cached
    per-process, so multiple call sites within the same worker process
    always agree -- but the cache is only reused after re-checking
    `os.getpid()` against the cached PID, so a value cached before a
    `fork()` is never silently inherited by the child process.
    """
    global _worker_id_cache
    pid = os.getpid()
    if _worker_id_cache is not None and _worker_id_cache[0] == pid:
        return _worker_id_cache[1]
    value = f"{socket.gethostname()}_{pid}_{uuid.uuid4().hex[:8]}"
    _worker_id_cache = (pid, value)
    return value


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
