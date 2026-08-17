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


def install_legacy_module(
    dotted_name: str, exports: dict[str, object]
) -> None:
    """Make `dotted_name` importable, backed by no real file on disk.

    Registers a synthetic module in `sys.modules` and as an attribute of
    its parent package, so both `import <dotted_name>` and
    `from <dotted_name> import <name>` succeed as if a real module existed
    there. Call once from the parent package's own `__init__.py`, so it
    runs before anything tries to import the historical path.
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
    # Only calling-convention changes need an entry -- a straight class
    # rename doesn't, since Serializable._update_imports already rewrites
    # every occurrence of the old name in frozen source. Detected via a
    # second bare positional argument (not a keyword= or **kwargs unpack)
    # after the instruction itself, the telltale sign of the old
    # positional InstructionLabel form.
    "InstructionLabel (old positional form)": re.compile(
        r"InstructionLabel\([^()]*?,\s*(?!\w+\s*=(?!=)|\*\*)[^()=,\s]"
    ),
}


def detect_legacy_construction(source: str) -> list[str]:
    """Cheap regex scan for known legacy-construction patterns in source text.

    Returns the name of each pattern found (empty if none). Pattern
    matching only, not a real parse -- trades exhaustiveness for a fast
    decode-time check (a real `ast`-based scan lives in the separate
    migration tool). Only covers patterns with no other fix available; a
    straight class rename needs no entry, since `_update_imports` already
    fixes those.
    """
    return [
        name
        for name, pattern in _LEGACY_CONSTRUCTION_PATTERNS.items()
        if pattern.search(source)
    ]
