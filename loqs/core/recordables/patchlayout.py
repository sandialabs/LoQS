#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, MutableMapping
from typing import ClassVar, TypeAlias, TypeVar

from loqs.core.recordables.qeccodepatch import QECCodePatch
from loqs.internal import Displayable

T = TypeVar("T", bound="PatchLayout")

PatchLayoutLike: TypeAlias = "PatchLayout | Mapping[str, QECCodePatch] | None"
"""Objects that can be cast to a [](api:PatchLayout)."""


class PatchRelation(Displayable):
    """Runtime data about a relationship between a set of patches, at some
    point during a program's execution.

    Analogous to [](api:QECCodePatch)'s runtime half
    (`.pauli_frame`/`.data`), never to [](api:QECCode) -- there is no
    template/instance split here and no `Instruction` lookup on this
    class. When a multi-patch operation genuinely needs a physical
    circuit, it is built directly by an ordinary codepack function,
    independent of any `PatchRelation`: a circuit must be fixed at
    `Instruction`-build time, before any `QuantumProgram`/`PatchLayout`
    (and therefore any relation) exists, so a relation could never
    supply one anyway.

    `data` is fully unstructured, with no reserved keys -- exactly like
    [](api:QECCodePatch.data). `patch_labels` names the participants by
    role (e.g. `{"ctrl": "L0", "tgt": "L1"}`), the same convention
    [](api:InstructionLabel)'s `patch_labels` already uses.

    Examples
    --------
    >>> from loqs.core.recordables import PatchRelation
    >>> rel = PatchRelation({"a": "L0", "b": "L1"}, data={"seam_qubits": ["S0", "S1", "S2"]})
    >>> rel.patch_labels
    {'a': 'L0', 'b': 'L1'}
    >>> rel.data["seam_qubits"]
    ['S0', 'S1', 'S2']
    """

    _CACHE_ON_SERIALIZE: ClassVar[bool] = True

    _SERIALIZE_ATTRS = ["patch_labels", "data"]

    def __init__(
        self,
        patch_labels: Mapping[str, str],
        data: dict | None = None,
    ) -> None:
        """
        Parameters
        ----------
        patch_labels:
            See [](api:patch_labels).

        data:
            Tracked relation-specific data dictionary. If provided, it
            will be deep-copied to ensure independence of the new
            relation's state.
        """
        self.patch_labels = dict(patch_labels)
        """Role -> patch label naming this relation's participants."""

        import copy

        self.data = copy.deepcopy(data) if data is not None else {}
        """Extra relation-specific data to be tracked, fully unstructured."""

    def __str__(self) -> str:
        return f"PatchRelation({self.patch_labels})"

    def copy(self) -> "PatchRelation":
        """Return a copy of this [](api:PatchRelation), deep-copying `data`.

        Returns
        -------
        PatchRelation
            The copied [](api:PatchRelation).
        """
        return PatchRelation(dict(self.patch_labels), data=self.data)

    @classmethod
    def _from_decoded_attrs(cls, attr_dict) -> "PatchRelation":
        """Create a PatchRelation from decoded attributes dictionary."""
        obj = cls(attr_dict["patch_labels"])
        obj.data = attr_dict.get("data", {})
        return obj


class PatchLayout(MutableMapping[str, QECCodePatch], Displayable):
    """A collection of [](api:QECCodePatch) objects, plus relational data.

    This is a dict-like object where the keys are patch labels (literally,
    as any `patch_label` usage in an [](api:Instruction) apply function
    refers to these keys) and the values are [](api:QECCodePatch) objects
    -- a strict superset of the old `PatchDict`'s API. It also tracks
    [](api:PatchRelation) objects, keyed by the sorted tuple of their
    participants' patch labels (order-independent lookup; `frozenset`
    keys were considered but are not supported by LoQS's serialization
    framework, so a canonical sorted `tuple` is used instead).

    Unlike many other `LoQS` objects, this is a mutable object to make it easy
    to manipulate patches. Users should be careful to first use [](api:copy)
    to avoid messing up previous [](api:Frame) objects (or use
    [](api:Frame.expire) properly).

    Examples
    --------
    >>> from loqs.core.recordables.patchlayout import PatchLayout
    >>> layout = PatchLayout()
    >>> len(layout)
    0
    """

    _CACHE_ON_SERIALIZE: ClassVar[bool] = True

    _SERIALIZE_ATTRS = ["patches", "relations"]

    patches: dict[str, QECCodePatch]
    """Underlying dict of patch labels and [](api:QECCodePatch) objects."""

    relations: dict[tuple[str, ...], PatchRelation]
    """[](api:PatchRelation) objects, keyed by the sorted tuple of their
    participants' patch labels. Prefer [](api:get_relation)/
    [](api:set_relation) over touching this directly.
    """
    # FUTURE WORK: relations are looked up purely by participant label
    # set, with no secondary "kind" tag -- multiple simultaneous,
    # independent relations over the same patch pair are not supported.
    # No concrete need for this was found in the surf17 multi-patch
    # codepacks; extend the key to tuple[tuple[str, ...], str] with an
    # explicit kind if that changes.

    def __init__(self, patches: PatchLayoutLike = None) -> None:
        """
        Parameters
        ----------
        patches:
            See [](api:patches). Defaults to `None`, which uses
            an empty `dict`. If a [](api:PatchLayout), its `relations`
            are copied too (via [](api:PatchRelation.copy)).
        """
        if patches is None:
            patches = {}

        if isinstance(patches, PatchLayout):
            self.patches = dict(patches.patches)
            self.relations = {
                k: v.copy() for k, v in patches.relations.items()
            }
        else:
            assert all([isinstance(k, str) for k in patches.keys()])
            assert all([isinstance(v, QECCodePatch) for v in patches.values()])
            self.patches = {k: v for k, v in patches.items()}
            self.relations = {}

    def __getitem__(self, key: str) -> QECCodePatch:
        return self.patches[key]

    def __len__(self) -> int:
        return len(self.patches)

    def __iter__(self) -> Iterator[str]:
        return iter(self.patches)

    def __setitem__(self, key: str, value: QECCodePatch) -> None:
        self.patches[key] = value

    def __delitem__(self, key: str) -> None:
        del self.patches[key]
        # Auto-drop any relation referencing the now-removed patch: unlike
        # QECCodePatch.data (deleted atomically along with its own patch),
        # a PatchRelation lives in a separate structure keyed by label, so
        # nothing else would ever clean up a reference to a removed patch.
        stale_keys = [rk for rk in self.relations if key in rk]
        for rk in stale_keys:
            del self.relations[rk]

    def __str__(self) -> str:
        str_dict = {k: str(v) for k, v in self.patches.items()}
        return f"PatchLayout({str_dict})"

    @property
    def all_qubit_labels(self) -> list[str | int]:
        """All qubits managed by patches in this [](api:PatchLayout)."""
        qubits: list[str | int] = []
        for patch in self.patches.values():
            qubits.extend(patch.qubits)
        return qubits

    def copy(self) -> PatchLayout:
        """Return a copy of this [](api:PatchLayout).

        Returns
        -------
        PatchLayout
            The copied [](api:PatchLayout), with its own copies of
            [](api:relations) too.
        """
        return PatchLayout(self)

    @staticmethod
    def _relation_key(labels) -> tuple[str, ...]:
        return tuple(sorted(labels))

    def get_relation(self, *labels: str) -> PatchRelation | None:
        """Look up a [](api:PatchRelation) by its participants' labels.

        Order of `labels` does not matter. Warns (does not raise) if any
        label isn't a current key of this [](api:PatchLayout) -- with
        patch removal auto-dropping stale relations, reaching this should
        only happen from a typo or a reference to an already-removed
        patch, not a relation that simply doesn't exist yet.

        Parameters
        ----------
        *labels:
            The participant patch labels naming the relation.

        Returns
        -------
        PatchRelation | None
            The matching relation, or `None` if none has been registered
            for this exact participant set.

        Examples
        --------
        >>> from loqs.core import QECCode
        >>> from loqs.core.recordables.patchlayout import PatchLayout, PatchRelation
        >>> code = QECCode(instructions={}, template_qubits=["q0"], template_data_qubits=["q0"])
        >>> layout = PatchLayout({"L0": code.create_patch(["Q0"]), "L1": code.create_patch(["Q1"])})
        >>> layout.get_relation("L0", "L1") is None
        True
        >>> layout.set_relation(PatchRelation({"a": "L0", "b": "L1"}, data={"m": 1}))
        >>> layout.get_relation("L1", "L0").data["m"]
        1
        """
        missing = [lbl for lbl in labels if lbl not in self.patches]
        if missing:
            warnings.warn(
                f"get_relation called with label(s) not in this "
                f"PatchLayout's patches: {missing}"
            )
        return self.relations.get(self._relation_key(labels))

    def set_relation(self, relation: PatchRelation) -> None:
        """Register a [](api:PatchRelation), keyed by its own participants.

        Overwrites any existing relation for the same participant set.

        Parameters
        ----------
        relation:
            The [](api:PatchRelation) to register.
        """
        key = self._relation_key(relation.patch_labels.values())
        self.relations[key] = relation

    @classmethod
    def _from_decoded_attrs(cls, attr_dict) -> "PatchLayout":
        """Create a PatchLayout from decoded attributes dictionary."""
        obj = cls(attr_dict["patches"])
        raw_relations = attr_dict.get("relations", {}) or {}
        obj.relations = {
            (tuple(k) if not isinstance(k, tuple) else k): v
            for k, v in raw_relations.items()
        }
        return obj
