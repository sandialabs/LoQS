#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Generic machinery for deprecating old classes and functions after a move, rename, or removal."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import sys
import types
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
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


def install_legacy_module_aliases_for_relocations(
    table: Mapping[tuple[str, str], tuple[str, str] | None],
) -> None:
    """Auto-register an `install_legacy_module` alias for every table entry
    that's a *pure module relocation* -- the class itself is unchanged,
    only which module it lives in moved. Safe to do unconditionally,
    unlike a real rename (see `loqs.tools.migrate.renames`'s module
    docstring for why those need a human, not an automatic forward): the
    old path ends up pointing at the exact same class object, so there's
    no constructor-compatibility question to get wrong.

    `table` is expected to be a `(old_module, old_name) -> (new_module,
    new_name) | None` mapping in the shape of
    `Serializable._get_cumulative_changes(0)`. An entry is skipped (not
    auto-aliased) if: its target is `None` (deleted outright, nothing to
    forward to); the class's own name also changed (a real rename, not a
    pure relocation -- e.g. every `*CastableTypes -> *Like` type-alias
    entry, and any class actually renamed rather than just moved, are
    both excluded this way); nothing actually moved; the old module path
    is still real for some other reason (never clobbers an existing
    module); or the new location can't actually be imported right now
    (e.g. it's behind an optional third-party backend dependency that
    isn't installed -- not this function's problem to solve). Multiple
    classes that used to share the same old module (e.g. `SyndromeLabel`
    and `PauliFrame` both moved out of `loqs.core.syndrome`) are grouped
    into a single `install_legacy_module` call for that module, rather
    than one call per class -- `install_legacy_module` can only register
    a given dotted name once.
    """
    exports_by_old_module: dict[str, dict[str, object]] = {}
    for (old_module, old_name), new_loc in table.items():
        if new_loc is None:
            continue
        new_module, new_name = new_loc
        if new_name != old_name or new_module == old_module:
            continue
        try:
            obj = getattr(importlib.import_module(new_module), new_name)
        except ImportError:
            continue
        exports_by_old_module.setdefault(old_module, {})[new_name] = obj

    for old_module, exports in exports_by_old_module.items():
        try:
            if importlib.util.find_spec(old_module) is not None:
                continue
        except (ModuleNotFoundError, ValueError):
            pass  # a missing parent package/broken existing entry -- proceed
        if old_module in sys.modules:
            continue  # already real for some other reason -- don't clobber it
        install_legacy_module(old_module, exports)


_RENAMED_STRING_LITERALS: dict[str, str] = {
    "Iz": "Imrz",
}
"""Bare string literals (not importable names, so no shim can catch them)
renamed in v1.2, mapping old name to new. Checked by `legacy_name_hint`."""


def legacy_name_hint(name: str) -> str:
    """A short, appendable hint (e.g. `" ('Iz' was renamed to 'Imrz' in
    v1.2)"`) if `name` matches a bare string literal renamed in v1.2, or
    an empty string otherwise. Meant to be tacked onto an existing
    "not found"-style error message at a lookup site that can't otherwise
    tell a genuine typo from an old, since-renamed name.
    """
    new_name = _RENAMED_STRING_LITERALS.get(name)
    if new_name is None:
        return ""
    return f" ({name!r} was renamed to {new_name!r} in v1.2)"


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
                or (
                    f"{name} is deprecated and can no longer be constructed. "
                    "If this call is in your own source code, `loqs-migrate "
                    "source <path>` may be able to update it for you."
                )
            )
        warnings.warn(
            message
            or f"{name} is deprecated; constructing its replacement on your behalf.",
            DeprecationWarning,
            stacklevel=2,
        )
        return build(*args, **kwargs)

    return type(name, (Displayable,), {"__new__": __new__})


@dataclass
class DeprecationInfo:
    """Structured info describing why a function is deprecated, stored on
    its wrapped version by [](api:deprecated) for later introspection --
    e.g. by a future `loqs-migrate` rename-table entry once the function
    is actually removed."""

    replacement: str
    note: str | None


_DEFAULT_DEPRECATION_NOTE = (
    "Will possibly be removed in a future release. Create a GitHub issue "
    "if new functionality does not cover your usecase."
)


def deprecated(
    replacement: str,
    *,
    note: str | None = _DEFAULT_DEPRECATION_NOTE,
    stacklevel: int = 2,
) -> Callable[[Callable], Callable]:
    """Mark a function as deprecated.

    Wraps the function so calling it warns (`DeprecationWarning`) with a
    consistently-formatted message naming `replacement`, then delegates to
    the original function unchanged. Unlike
    [](api:make_legacy_construction_shim), which replaces a class's
    construction outright, this leaves the function's own behavior
    untouched and only adds the warning.
    """

    def decorator(func: Callable) -> Callable:
        message = f"{func.__name__} is deprecated; use {replacement} instead."
        if note:
            message += f" {note}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)

        wrapper.__deprecated__ = DeprecationInfo(replacement=replacement, note=note)
        return wrapper

    return decorator
