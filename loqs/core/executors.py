#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Structural executor protocols shared by shot-level and program-level parallelism.

Two distinct executor shapes exist in this codebase, matching two different
dispatch mechanisms:

- [](api:SubmitExecutor): anything exposing a `concurrent.futures`-style
  `.submit()` method returning a `Future`. Satisfied by
  `loky.get_reusable_executor()` and `mpi4py.futures.MPIPoolExecutor` (both
  real `concurrent.futures.Executor` subclasses) without a hard dependency
  on either package. Used for shot-level parallelism
  ([](api:QuantumProgram.run)'s `shot_executor` parameter) and, via
  `loqs.tools.paralleltools.ParallelStrategy`, for program-level
  parallelism dispatched one `.submit()` call per chunk.
- [](api:MapArrayExecutor): anything exposing a `submitit`-style
  `.map_array()` bulk-submission method. Satisfied by `submitit.Executor`,
  which submits a single `sbatch` call covering an entire batch of chunks
  rather than one job per chunk. Used exclusively for program-level
  parallelism via `loqs.tools.paralleltools.ParallelStrategy`; never for
  shot-level parallelism. Note that a `submitit.Executor` actually
  satisfies *both* protocols (it has a `.submit()` method too), so code
  that cares about `map_array`'s bulk-submission efficiency must check for
  [](api:MapArrayExecutor) first.

Both are `runtime_checkable`, so `isinstance(x, SubmitExecutor)`/
`isinstance(x, MapArrayExecutor)` work directly for dispatch-mechanism
detection, without hardcoding which concrete package produced `x`.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SubmitExecutor(Protocol):
    """Structural type for any `concurrent.futures`-style executor.

    Only a `.submit()` method returning a `Future` is required, so a
    `loky.get_reusable_executor()` instance or an
    `mpi4py.futures.MPIPoolExecutor` both satisfy this without this module
    depending on either package.
    """

    def submit(self, fn, /, *args, **kwargs) -> Future: ...


@runtime_checkable
class MapArrayExecutor(Protocol):
    """Structural type for a `submitit`-style bulk-submission executor.

    Only a `.map_array()` method is required, so a `submitit.Executor`
    satisfies this without this module depending on `submitit` directly.
    """

    def map_array(self, fn, *iterables) -> list[Any]: ...
