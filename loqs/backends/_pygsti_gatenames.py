#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Shared helpers for working around pyGSTi's strict string-form label
parser (`pygsti.circuits.circuitparser.parse_label`), which is stricter
than what it's willing to write out: it splits a label's name at its
first uppercase letter after the initial character, so a mixed-case gate
name like `"GiMCM"` doesn't survive a string round-trip unchanged
(`"GiMCM"` comes back as just `"Gi"`). Used by both
[](api:PyGSTiPhysicalCircuit) (which serializes a circuit via its own
string form) and [](api:PyGSTiNoiseModel) (whose underlying `pygsti`
`Model` serializes its own operation labels the same way via
`Model.dumps()`/`Model.loads()`), so both backends can hit the same
collision independently.
"""

from __future__ import annotations

from collections.abc import Iterable


def pygsti_safe_gatename(name: str) -> str:
    """The version of `name` that survives pyGSTi's string-form label
    parser round-trip (see module docstring). Lowercasing everything
    after the first character is always safe here, since the first
    character is only ever checked against pyGSTi's own fixed gate-name
    prefix (`"G"`).
    """
    return name if len(name) <= 1 else name[0] + name[1:].lower()


def gatename_pygsti_safe_renames(names: Iterable[str]) -> dict[str, str]:
    """`{original_name: pygsti_safe_name}` for every distinct name in
    `names` that doesn't already survive pyGSTi's string-form label
    parser round-trip (see [](api:pygsti_safe_gatename)).

    Raises `ValueError` on a collision, i.e. two distinct names that
    share the same safe form (e.g. `"GiMCM"` and `"Gimcm"`) -- aliasing
    both to the same name here would create the collision this rewrite
    exists to avoid, rather than prevent it.
    """
    renames: dict[str, str] = {}
    safe_to_original: dict[str, str] = {}
    for name in names:
        safe_name = pygsti_safe_gatename(name)
        existing = safe_to_original.setdefault(safe_name, name)
        if existing != name:
            raise ValueError(
                f"Gate names {existing!r} and {name!r} both round-trip "
                f"through pyGSTi's string parser as {safe_name!r} -- "
                "rename one of them to serialize this circuit/model."
            )
        if safe_name != name:
            renames[name] = safe_name
    return renames
