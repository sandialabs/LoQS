#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Per-file configuration for resolving a source file's instruction registry.

Resolving *which* [](api:Instruction) a legacy `InstructionLabel` call
refers to (needed for [](api:loqs.tools.migrate.labels)) requires either
running the code or statically loading the same instruction registry the
code would use at runtime. Codepacks build their whole `instructions` dict
eagerly in plain Python, so the registry is usually reachable just by
importing the relevant codepack module and calling its instruction-set-
building function (typically named `create_qec_code`) -- no simulation
required. Since the set of files needing this is small and self-owned (not
arbitrary third-party code), an explicit mapping is simpler and more
reliable than trying to infer it automatically.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import json
from pathlib import Path

from loqs.core.instructions.instruction import Instruction


def load_instruction_registry(
    dotted_path: str, /, **kwargs: object
) -> dict[str, Instruction]:
    """Import and call a `module:function` reference, returning its
    result's `.instructions` dict.

    Parameters
    ----------
    dotted_path:
        A `"module.submodule:function_name"` reference to a callable that
        returns an object with an `.instructions: dict[str, Instruction]`
        attribute (e.g. a codepack's own `create_qec_code`, which returns
        a [](api:QECCode)).
    **kwargs:
        Forwarded to the resolved callable (e.g. `layout="surf10"`).
    """
    module_name, _, func_name = dotted_path.partition(":")
    if not func_name:
        raise ValueError(
            f"{dotted_path!r} is not a 'module:function' reference "
            "(missing ':')."
        )
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    result = func(**kwargs)
    return dict(result.instructions)


class MigrationConfig:
    """A `{file_path: dotted_path}` mapping (see
    [](api:load_instruction_registry)), plus a cache of already-loaded
    registries (importing/building a codepack is not free, and the same
    codepack is often shared by several files).

    File paths are matched as given -- typically relative to the
    directory `loqs-migrate` is invoked from, matching how the mapping
    file itself would name them (see [](api:MigrationConfig.from_json)).
    """

    def __init__(
        self, file_registries: Mapping[str, str] | None = None
    ) -> None:
        self._file_registries = dict(file_registries or {})
        self._cache: dict[str, dict[str, Instruction]] = {}

    @classmethod
    def from_json(cls, path: str | Path) -> "MigrationConfig":
        """Load a config from a JSON file mapping file paths to
        `"module:function"` references, e.g.:

        ```json
        {
            "tests/codepacks/test_codepack_surf17_surgery.py":
                "loqs.codepacks.codepack_surf17_tomita2014:create_qec_code"
        }
        ```
        """
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def instructions_for(
        self, file_path: str | Path
    ) -> dict[str, Instruction]:
        """The instruction registry configured for `file_path`, or an
        empty dict if none is configured (every candidate in that file
        will then be flagged for manual review instead of rewritten)."""
        key = str(file_path)
        dotted_path = self._file_registries.get(key)
        if dotted_path is None:
            return {}
        if dotted_path not in self._cache:
            self._cache[dotted_path] = load_instruction_registry(dotted_path)
        return self._cache[dotted_path]
