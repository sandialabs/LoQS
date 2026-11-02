#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Generated codepacks on `BaseCodepack`: the lock and provenance.

A shipped codepack sets `spec` and `artifact` and is locked: construction
loads the generated module and refuses a stale, edited or missing one. A
subclass with `locked = False` derives the tables at construction instead.
This module doesn't import Stim, except inside an unlocked construction.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Mapping
from typing import Any

from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.codepacks.base import BaseCodepack
from loqs.codepacks.generator import gates
from loqs.codepacks.generator.render import render, render_two_patch
from loqs.codepacks.generator.schema import (
    CodeSpec,
    CustomApplyFn,
    Gate,
    Src,
    spec_hash,
)
from loqs.core import Instruction, QECCode


class StaleArtifactError(RuntimeError):
    """A locked codepack's generated artifact is missing, stale or edited."""


def tables_hash(generated: Mapping[str, Any]) -> str:
    """The SHA-256 of generated tables' JSON, tuples read as lists."""
    blob = json.dumps(generated, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def load_artifact(spec: CodeSpec, module: str, owner: str) -> dict:
    """The locked tables of `module`, or a `StaleArtifactError` naming both
    ways out: regenerate, or subclass with `locked = False`."""
    ways = (
        f"Regenerate it with `python -m loqs.codepacks.generator "
        f"<module>:{owner}`, or derive at construction with a one-line "
        f"subclass: `class My{owner}({owner}): locked = False`."
    )
    sys.modules.pop(module, None)
    try:
        mod = importlib.import_module(module)
    except ImportError:
        raise StaleArtifactError(
            f"{owner}: no generated artifact {module!r}. {ways}"
        ) from None
    if mod.INPUT_HASH != spec_hash(spec):
        raise StaleArtifactError(
            f"{owner}: artifact {module!r} is stale (the spec changed). "
            f"{ways}"
        )
    if tables_hash(mod.GENERATED) != mod.TABLES_HASH:
        raise StaleArtifactError(
            f"{owner}: artifact {module!r} was edited by hand. {ways}"
        )
    generated: dict = mod.GENERATED
    return generated


class GeneratedCodepack(BaseCodepack):
    """A codepack rendered from a spec and its generated tables.

    Class attributes: `spec` (a `CodeSpec`), `artifact` (the dotted name
    of the generated module) and `locked` (default `True`).

    Options, all keyword-only:

    - `circuit_backend`: the physical circuit class of every circuit.
    - `include_idles`: pad every circuit with idle gates by duration.
    - `gate_durations`: duration of each gate name; default from the gate
      table's duration classes.
    - `idle_gates`: idle gate name of each duration; same default.
    - `ft_state_prep_max_repeats`: bound on every repeat-until-success
      prep's attempts. It does nothing for a spec with none.

    A locked class loads and checks its artifact, refusing a stale, edited
    or missing one. To derive the tables at construction instead (needs
    Stim), subclass in one line::

        class MySteane(SteaneCodepack):
            locked = False
    """

    spec: CodeSpec
    artifact: str = ""
    locked: bool = True

    def __init__(
        self,
        *,
        circuit_backend: Any = PyGSTiPhysicalCircuit,
        include_idles: bool = False,
        gate_durations: Mapping[str, int] | None = None,
        idle_gates: Mapping[int, str] | None = None,
        ft_state_prep_max_repeats: int = 100,
    ) -> None:
        self.circuit_backend = circuit_backend
        self.include_idles = include_idles
        self.gate_durations = dict(
            gate_durations
            if gate_durations is not None
            else {
                n: gates.DEFAULT_DURATIONS[e.duration]
                for n, e in gates.GATES.items()
            }
        )
        self.idle_gates = dict(
            idle_gates
            if idle_gates is not None
            else {
                gates.DEFAULT_DURATIONS[c]: n
                for c, n in gates.DEFAULT_IDLES.items()
            }
        )
        self.ft_state_prep_max_repeats = ft_state_prep_max_repeats
        self._idle_gate_names = tuple(self.idle_gates.values())
        if self.locked:
            self.generated = load_artifact(
                self.spec, self.artifact, type(self).__name__
            )
            self._lock = f"locked ({self.artifact})"
        else:
            from loqs.codepacks.generator.generate import generate

            self.generated = generate(self.spec)
            self._lock = "derived at construction (unlocked)"

    def _options(self) -> dict[str, Any]:
        return dict(
            circuit_backend=self.circuit_backend,
            include_idles=self.include_idles,
            gate_durations=self.gate_durations,
            idle_gates=self.idle_gates,
        )

    def create_qec_code(self) -> QECCode:
        """Render the code from the spec and its tables."""
        return render(
            self.spec,
            self.generated,
            ft_state_prep_max_repeats=self.ft_state_prep_max_repeats,
            **self._options(),
        )

    def build_two_patch(self, key: str, geometry: Any) -> Instruction:
        """Render the spec's two-patch gate `key` on `geometry`."""
        return render_two_patch(
            self.spec, self.generated, key, geometry, **self._options()
        )

    def provenance(self) -> list[dict]:
        """One row per spec element and per derived table: key, part,
        kind, citation and lock status."""
        s, rows = self.spec, []

        def row(key: str, part: str, src: Src, kind: str = "") -> None:
            cite = src.cite
            if src.reason:
                cite = f"{cite} -- {src.reason}" if cite else src.reason
            rows.append(
                {
                    "key": key,
                    "part": part,
                    "kind": kind or src.kind,
                    "citation": cite,
                    "lock": self._lock,
                }
            )

        def derived(key: str, part: str, cite: str) -> None:
            rows.append(
                {
                    "key": key,
                    "part": part,
                    "kind": "derived",
                    "citation": cite,
                    "lock": self._lock,
                }
            )

        f = s.facts
        row("(facts)", "stabilizers, logicals", f.src)
        if f.conventions_src:
            row("(facts)", "tracking conventions", f.conventions_src)
        for it in s.items:
            row(it.key, type(it).__name__, it.src)
            if isinstance(it, Gate):
                trivial = self.generated["tables"]["gates"][it.key]["trivial"]
                what = (
                    "per-qubit frame"
                    if trivial
                    else "frame, syndrome and per-qubit maps"
                )
                derived(
                    it.update_key or it.key,
                    f"bookkeeping ({what})",
                    f"generator from the facts and the circuit; declared "
                    f"{it.logical} checked",
                )
            if isinstance(it, CustomApplyFn):
                for k, ref in it.derived:
                    derived(it.key, f"data {k!r}", ref)
        for t in s.tables:
            row(t.name, f"lookup table ({t.kind})", t.src)
        for g in s.two_patch:
            row(g.key, "two-patch circuit", g.src)
            derived(
                g.key,
                "bookkeeping (joint, split per patch)",
                f"declared {g.logical} checked",
            )
        return rows
