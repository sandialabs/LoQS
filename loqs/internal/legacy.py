#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Generic machinery for keeping old class names constructible after a move or removal."""

from __future__ import annotations

import re
import sys
import types
import warnings
from typing import Any, Callable

from loqs.internal import Displayable


def install_legacy_module(dotted_name: str, exports: dict[str, object]) -> None:
    """Make `dotted_name` importable, backed by no real file on disk.

    Registers a synthetic module in `sys.modules` and as an attribute of
    its parent package, so both `import <dotted_name>` and
    `from <dotted_name> import <name>` succeed exactly as if a real module
    existed at that historical location. Relies on parent packages always
    being imported before Python attempts to resolve a dotted child module,
    so calling this once inside the parent's own `__init__.py` is enough to
    guarantee it runs before any import of the historical path is attempted.
    """
    assert dotted_name not in sys.modules, f"{dotted_name} already exists"
    module = types.ModuleType(dotted_name)
    module.__dict__.update(exports)
    sys.modules[dotted_name] = module
    parent_name, _, child_name = dotted_name.rpartition(".")
    if parent_name:
        setattr(sys.modules[parent_name], child_name, module)


def make_legacy_construction_shim(
    name: str,
    build: Callable[..., Any] | None = None,
    message: str | None = None,
) -> type:
    """Build a class whose direct construction is a deprecated legacy path.

    If `build` is given, constructing this class warns (`DeprecationWarning`)
    and returns `build(*args, **kwargs)` instead of an instance of this
    class -- for an old class kept working by redirecting to its modern
    replacement. If `build` is `None`, construction instead raises
    `TypeError(message)` -- for an old class that must not be constructed
    at all. `name` is used only to build a default message when `message`
    isn't given.
    """

    def __new__(cls, *args, **kwargs):
        if build is None:
            raise TypeError(
                message
                or f"{name} is deprecated and can no longer be constructed."
            )
        warnings.warn(
            message
            or f"{name} is deprecated; constructing its replacement on your behalf.",
            DeprecationWarning,
            stacklevel=2,
        )
        return build(*args, **kwargs)

    return type(name, (Displayable,), {"__new__": __new__})


_LEGACY_CONSTRUCTION_PATTERNS: dict[str, re.Pattern] = {
    # NOTE: a straight class-location rename (like PatchDict -> PatchLayout)
    # is deliberately *not* listed here -- confirmed directly that
    # Serializable._update_imports already rewrites both the import line
    # and every other occurrence of the old name throughout the frozen
    # source's body (its "third pass"), so old frozen source constructing
    # PatchDict() is transparently and losslessly rewritten to construct
    # PatchLayout() directly before it's ever re-executed; there is no
    # runtime risk left to gate for that case, and no shim is ever
    # actually reached. This mechanism only needs to cover cases where the
    # class *name* is unchanged but its constructor's calling convention
    # is not -- text-rewriting can't fix that, only a construction-time
    # shim (checked at the moment the old call is actually made) can.
    #
    # Old-style positional InstructionLabel construction had at least one
    # positional argument after the instruction itself (patch_label,
    # inst_args, inst_kwargs); the modern form only ever takes keyword
    # arguments past the first. A second bare positional argument (not
    # starting with a keyword= name) is the telltale sign.
    "InstructionLabel (old positional form)": re.compile(
        r"InstructionLabel\([^()]*?,\s*(?!patch_labels?\s*=)[^()=,\s]"
    ),
}


def detect_legacy_construction(source: str) -> list[str]:
    """Cheap regex scan for known legacy-construction patterns in source text.

    Returns the name of each pattern found (empty list if none). This is a
    lighter, decode-time-only cousin of the full source-migration tool's
    detection engine: pattern matching only, not a real parse, so it can
    miss unusual formatting (aliased imports, etc.) -- it exists to catch
    the common case cheaply during decode, not to be exhaustive; the
    migration tool itself uses a real `ast`-based scan. Only lists
    patterns with no other fix available (see the module-level comment on
    `_LEGACY_CONSTRUCTION_PATTERNS`) -- a straight class rename needs no
    entry here at all, since `_update_imports` already fixes those
    transparently.
    """
    return [
        name
        for name, pattern in _LEGACY_CONSTRUCTION_PATTERNS.items()
        if pattern.search(source)
    ]
