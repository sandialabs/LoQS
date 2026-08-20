#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""A build-time convenience for constructing multi-patch instructions.

[](api:PatchGeometry) bundles the patch labels, qubit lists, seam qubit
lists, and layout name a multi-patch instruction builder needs, so a
caller types them once instead of re-typing them at every builder call
(and independently again for the corresponding `"Init Patch"` stack
entries). Reuses the `Mapping[str, str]`-style role-naming convention
[](api:InstructionLabel.patch_labels) already established. Currently
used by [](api:codepack_surf17_multipatch)/[](api:codepack_surf17_surgery),
the only codepacks with multi-patch instructions today, but nothing about
it is specific to those codepacks. It is a plain, static, build-time
object -- never part of a running program's state, and unrelated to
[](api:PatchLayout)/[](api:PatchRelation), which exist only once a
`QuantumProgram` is already running.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class PatchGeometry:
    """Role -> `(patch label, qubits)` assignment, plus named seam(s).

    Reuses the `Mapping[str, str]`-style role-naming convention already
    established by [](api:InstructionLabel.patch_labels) and
    [](api:PatchRelation.patch_labels). The two-patch case (most builders)
    uses roles `"a"`/`"b"` and a single unnamed seam; builders needing more
    than two patches or more than one seam (e.g. a 3-patch lattice-surgery
    CNOT) use whatever role names make sense and name each seam.

    Examples
    --------
    >>> from loqs.core import PatchGeometry
    >>> geometry = PatchGeometry(
    ...     patches={"a": ("L0", ["D0", "D1"]), "b": ("L1", ["D2", "D3"])},
    ...     seam=["S0"],
    ...     layout="surf10",
    ... )
    >>> geometry.label("a")
    'L0'
    >>> geometry.seam
    ['S0']
    """

    def __init__(
        self,
        patches: Mapping[str, tuple[str, Sequence[str]]],
        layout: str,
        seam: Sequence[str] | None = None,
        seams: Mapping[str, Sequence[str]] | None = None,
    ):
        """
        Parameters
        ----------
        patches:
            Role name -> `(patch_label, qubits)`. `qubits` is each patch's
            full qubit list (template order, data + ancillas).

        layout:
            One of `"surf17"`, `"surf13"`, `"surf10"` (all patches must
            share one layout).

        seam:
            The seam qubits, for the common case of exactly one seam.
            Omit both `seam` and `seams` for an operation with no seam at
            all (e.g. a transversal CNOT or an ancilla-mediated joint
            parity measurement). Mutually exclusive with `seams`.

        seams:
            Name -> seam qubits, for geometries with more than one seam
            (e.g. `{"zz": [...], "xx": [...]}` for a 3-patch surgery CNOT).
            Mutually exclusive with `seam`. Internally, `seam=...` is
            equivalent to `seams={"main": ...}` -- a `PatchGeometry`
            always stores its seams the same way regardless of which
            constructor argument was used, so [](api:PatchGeometry.seam)
            and [](api:PatchGeometry.subset) work uniformly either way.
        """
        assert (
            seam is None or seams is None
        ), "Pass at most one of `seam`/`seams`, not both"
        resolved_seams: dict[str, list[str]]
        if seam is not None:
            resolved_seams = {"main": list(seam)}
        elif seams is not None:
            resolved_seams = {name: list(qs) for name, qs in seams.items()}
        else:
            resolved_seams = {}

        self.patches: dict[str, tuple[str, list[str]]] = {
            role: (label, list(qubits))
            for role, (label, qubits) in patches.items()
        }
        """Role -> `(patch_label, qubits)`."""

        self.layout = layout
        """One of `"surf17"`, `"surf13"`, `"surf10"`."""

        self.seams = resolved_seams
        """Seam name -> seam qubits (always a dict, even for one seam)."""

        # Every patch's and every seam's qubits must be pairwise disjoint --
        # the most plausible real mistake here is copy-pasting the wrong
        # role/patch's qubit list (e.g. control vs. target differing only
        # by a suffix), which would otherwise silently corrupt a program.
        seen: dict[str, str] = {}
        for group_name, qubits in [
            (role, qubits) for role, (_, qubits) in self.patches.items()
        ] + list(self.seams.items()):
            for q in qubits:
                assert q not in seen, (
                    f"PatchGeometry qubit '{q}' appears in both "
                    f"'{seen[q]}' and '{group_name}' -- all patch/seam "
                    "qubit lists must be disjoint"
                )
                seen[q] = group_name

    @property
    def patch_labels(self) -> dict[str, str]:
        """Role -> patch label, dropping the qubit lists."""
        return {role: label for role, (label, _) in self.patches.items()}

    def label(self, role: str) -> str:
        """The patch label assigned to `role`."""
        return self.patches[role][0]

    def qubits(self, role: str) -> list[str]:
        """The qubit list assigned to `role`."""
        return self.patches[role][1]

    @property
    def seam(self) -> list[str]:
        """The seam qubits, when this geometry has exactly one seam.

        Works uniformly whether this `PatchGeometry` was built with the
        bare `seam=` argument or produced by [](api:PatchGeometry.subset)
        (whose output always has exactly one seam) -- callers never need
        to know which.
        """
        if len(self.seams) != 1:
            raise ValueError(
                "PatchGeometry.seam requires exactly one seam; this "
                f"geometry has {list(self.seams)}. Use .seams[<name>] "
                "instead."
            )
        return next(iter(self.seams.values()))

    def subset(self, roles: Sequence[str], seam: str) -> PatchGeometry:
        """A new, two-role `PatchGeometry` for a subset of this one's roles.

        `roles` (exactly 2) are remapped to canonical roles `"a"`/`"b"` in
        the order given, and `seam` names which of this geometry's seams
        the result carries. This is what a 3-patch, multi-seam geometry
        (e.g. a surgery CNOT's `ctrl`/`tgt`/`anc` roles and `zz`/`xx`
        seams) uses to derive the two-role geometry each underlying
        pairwise merge/split builder actually needs.

        Parameters
        ----------
        roles:
            Exactly 2 of this geometry's role names, in `(a, b)` order.

        seam:
            Name of one of this geometry's seams.

        Returns
        -------
        PatchGeometry
            Roles `"a"`/`"b"`, one seam, same layout.
        """
        assert len(roles) == 2, "subset() produces a two-role PatchGeometry"
        role_a, role_b = roles
        return PatchGeometry(
            patches={"a": self.patches[role_a], "b": self.patches[role_b]},
            layout=self.layout,
            seams={seam: self.seams[seam]},
        )

    def init_patch_entries(self, patch_type_tag: str) -> list[dict]:
        """`"Init Patch <patch_type_tag>"` stack entries for every role.

        Parameters
        ----------
        patch_type_tag:
            The key a `QuantumProgram`'s `patch_types` mapping registers
            the desired [](api:QECCode) under (e.g. `"SURF"`), used to
            resolve the `"Init Patch <tag>"` stack shorthand.

        Returns
        -------
        list[dict]
            One `{"instruction": ..., "new_patch_label": ..., "qubits": ...}`
            entry per role, in role-insertion order.
        """
        return [
            {
                "instruction": f"Init Patch {patch_type_tag}",
                "new_patch_label": label,
                "qubits": list(qubits),
            }
            for label, qubits in self.patches.values()
        ]

    def __repr__(self) -> str:
        return (
            f"PatchGeometry(patches={self.patch_labels}, "
            f"seams={list(self.seams)}, layout={self.layout!r})"
        )
