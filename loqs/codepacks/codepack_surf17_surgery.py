#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Lattice-surgery machinery for the Surface-17/13/10 codepack.

This module implements merge/split lattice surgery between two d=3 rotated
surface-code patches built from [](api:codepack_surf17_tomita2014), on top of
the multi-patch foundation in [](api:codepack_surf17_multipatch).

Geometry
--------
Each patch is a 3x3 data grid (row-major D0..D8) with the Tomita-Svore check
layout. The codepack's logical representatives are `Z_L = Z0 Z4 Z8` and
`X_L = X2 X4 X6`, with the boundary-string equivalences (by stabilizer
multiplication):

- `Z_L == Z0 Z1 Z2` (top row) `== Z6 Z7 Z8` (bottom row)
- `X_L == X0 X3 X6` (left column) `== X2 X5 X8` (right column)

A **ZZ merge** therefore stacks the patches **vertically** (patch A on top,
a 3-qubit seam row, patch B below): the four new Z checks that span the seam
multiply out to `Z_L(A) (x) Z_L(B)`. Seam qubits are prepared in `|+>` and
split-measured in the X basis. Two X checks grow across the seam (A's
bottom-boundary check and B's top-boundary check).

An **XX merge** places the patches **side by side** (A left, seam column,
B right) with everything X<->Z dual: seams in `|0>`, Z-basis split, four new
X checks multiplying to `X_L(A) (x) X_L(B)`, two grown Z checks.

Merged code (either orientation): 21 data qubits, 20 independent checks,
1 logical qubit.

Supports below are given as `(patch, index)` tuples where `patch` is `"A"`,
`"B"` (data-qubit template index 0-8) or `"S"` (seam index 0-2). Tiles are
4-slot `[NW, NE, SW, SE]` lists (with None for boundary checks) matching the
codepack's syndrome-extraction template slot order `[a, b, c, d]`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from loqs.backends.circuit.basecircuit import BasePhysicalCircuit
from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.codepacks.codepack_surf17_tomita2014 import (
    DEFAULT_GATE_DURATIONS as _DEFAULT_GATE_DURATIONS,
    DEFAULT_IDLE_GATES as _DEFAULT_IDLE_GATES,
    LAYOUT_SE_SPECS as _LAYOUT_SE_SPECS,
    X_TILE_DATA as _X_TILE_DATA,
    X_TILE_ROWS as _X_TILE_ROWS,
    Z_TILE_DATA as _Z_TILE_DATA,
    Z_TILE_ROWS as _Z_TILE_ROWS,
    _LAYOUT_PARALLELISM,
    build_se_templates as _se_templates,
    layout_qubits,
)
from loqs.core import Instruction, History
from loqs.core.frame import Frame
from loqs.core.instructions import builders
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchlayout import PatchLayout, PatchRelation
from loqs.core.recordables.pauliframe import PauliFrame

# Base single-patch check matrices (mirrors codepack_surf17_tomita2014;
# consistency is asserted by the unit tests).
BASE_H_X = np.array(
    [
        [1, 1, 0, 1, 1, 0, 0, 0, 0],  # SX0
        [0, 1, 1, 0, 0, 0, 0, 0, 0],  # SX1 (top boundary)
        [0, 0, 0, 0, 1, 1, 0, 1, 1],  # SX2
        [0, 0, 0, 0, 0, 0, 1, 1, 0],  # SX3 (bottom boundary)
    ]
)

BASE_H_Z = np.array(
    [
        [1, 0, 0, 1, 0, 0, 0, 0, 0],  # SZ0 (left boundary)
        [0, 1, 1, 0, 1, 1, 0, 0, 0],  # SZ1
        [0, 0, 0, 1, 1, 0, 1, 1, 0],  # SZ2
        [0, 0, 0, 0, 0, 1, 0, 0, 1],  # SZ3 (right boundary)
    ]
)

# Merged-code data-qubit index convention (both orientations):
# 0-8 = A.D0-A.D8, 9-11 = S0-S2, 12-20 = B.D0-B.D8.
MERGED_NUM_DATA = 21
_A_OFFSET = 0
_S_OFFSET = 9
_B_OFFSET = 12


def merged_index(support_elem: tuple[str, int]) -> int:
    """Map an ('A'|'S'|'B', idx) support element to its merged-code column."""
    patch, idx = support_elem
    if patch == "A":
        return _A_OFFSET + idx
    if patch == "S":
        return _S_OFFSET + idx
    if patch == "B":
        return _B_OFFSET + idx
    raise ValueError(f"Unknown patch tag: {patch}")


SEAM_GEOMETRY_ZZ = {
    "kind": "ZZ",
    "orientation": "vertical",  # A rows 0-2, seam row 3, B rows 4-6
    # Seam qubits are prepared in |+> and split-measured in the X basis
    "seam_prep_basis": "X",
    "new_check_type": "Z",
    # The four new Z checks spanning the seam. Their product telescopes to
    # Z(A.D6,A.D7,A.D8) (x) Z(B.D0,B.D1,B.D2) = Z_L(A) (x) Z_L(B).
    "new_checks": [
        {
            "support": [("A", 7), ("A", 8), ("S", 1), ("S", 2)],
            "tile": [("A", 7), ("A", 8), ("S", 1), ("S", 2)],
        },
        {
            "support": [("S", 0), ("S", 1), ("B", 0), ("B", 1)],
            "tile": [("S", 0), ("S", 1), ("B", 0), ("B", 1)],
        },
        {
            "support": [("A", 6), ("S", 0)],  # left boundary, weight 2
            "tile": [None, ("A", 6), None, ("S", 0)],
        },
        {
            "support": [("S", 2), ("B", 2)],  # right boundary, weight 2
            "tile": [("S", 2), None, ("B", 2), None],
        },
    ],
    # Boundary checks of A and B that grow across the seam during the merge.
    # `check_row` indexes the patch's own H (X-type here). After the split,
    # the re-formed old check equals the grown value XOR the two seam
    # outcomes ("seam_pair").
    "grown_check_type": "X",
    "grown_checks": {
        "A": {
            "check_row": 3,  # SX3 = {D6, D7}
            "old_support": [("A", 6), ("A", 7)],
            "seam_pair": [("S", 0), ("S", 1)],
            "support": [("A", 6), ("A", 7), ("S", 0), ("S", 1)],
            "tile": [("A", 6), ("A", 7), ("S", 0), ("S", 1)],
        },
        "B": {
            "check_row": 1,  # SX1 = {D1, D2}
            "old_support": [("B", 1), ("B", 2)],
            "seam_pair": [("S", 1), ("S", 2)],
            "support": [("S", 1), ("S", 2), ("B", 1), ("B", 2)],
            "tile": [("S", 1), ("S", 2), ("B", 1), ("B", 2)],
        },
    },
    # Telescoped joint-parity data support and the Pauli-frame bit type that
    # flips the measured parity (X bits flip Z-type parities).
    "parity_support": [
        ("A", 6),
        ("A", 7),
        ("A", 8),
        ("B", 0),
        ("B", 1),
        ("B", 2),
    ],
    "parity_frame_bit_type": "X",
    # The telescoped operator is Z(A bottom row) (x) Z(B top row); converting
    # to the canonical readout representatives Z0 Z4 Z8 multiplies in
    # SZ0*SZ2 on A and SZ1*SZ3 on B. Those stabilizers' recorded values
    # (random for non-Z-basis preps) must be XORed into the raw seam-check
    # parity to report the canonical joint parity.
    "telescope_reference_checks": {"A": [0, 2], "B": [1, 3]},
    # Split byproduct: the through-seam merged X_L runs along the left
    # column (A.D0, A.D3, A.D6, S0, B.D0, B.D3, B.D6), so the left-column
    # X_L(A) (x) X_L(B) product picks up the seam outcome on S0. Converting
    # to the codepack's canonical readout representative X2 X4 X6 on patch B
    # multiplies in SX0(B) and SX1(B) -- and SX1(B) is B's grown check,
    # whose value changes by s1 (+) s2 at the split. The canonical-product
    # flip is therefore s0 (+) s1 (+) s2, fixed up by a conditional logical
    # Z on patch B (frame Z on Z_L support D0, D4, D8).
    "byproduct_seam_indices": [0, 1, 2],
    "byproduct_patch": "B",
    "byproduct_logical": "Z",
    "byproduct_frame_support": [0, 4, 8],
    # Merged logical representatives (for tests / decoding):
    # Z_L(merged) = Z_L(A); X_L(merged) = left column through the seam.
    "merged_Z_L": [("A", 0), ("A", 4), ("A", 8)],
    "merged_X_L": [
        ("A", 0),
        ("A", 3),
        ("A", 6),
        ("S", 0),
        ("B", 0),
        ("B", 3),
        ("B", 6),
    ],
}

SEAM_GEOMETRY_XX = {
    "kind": "XX",
    "orientation": "horizontal",  # A cols 0-2, seam col 3, B cols 4-6
    # Seam qubits are prepared in |0> and split-measured in the Z basis
    "seam_prep_basis": "Z",
    "new_check_type": "X",
    # The four new X checks spanning the seam. Their product telescopes to
    # X(A.D2,A.D5,A.D8) (x) X(B.D0,B.D3,B.D6) = X_L(A) (x) X_L(B).
    # Seam column runs S0 (top) to S2 (bottom).
    "new_checks": [
        {
            "support": [("A", 2), ("A", 5), ("S", 0), ("S", 1)],
            "tile": [("A", 2), ("S", 0), ("A", 5), ("S", 1)],
        },
        {
            "support": [("S", 1), ("S", 2), ("B", 3), ("B", 6)],
            "tile": [("S", 1), ("B", 3), ("S", 2), ("B", 6)],
        },
        {
            "support": [("S", 0), ("B", 0)],  # top boundary, weight 2
            "tile": [None, None, ("S", 0), ("B", 0)],
        },
        {
            "support": [("A", 8), ("S", 2)],  # bottom boundary, weight 2
            "tile": [("A", 8), ("S", 2), None, None],
        },
    ],
    "grown_check_type": "Z",
    "grown_checks": {
        "A": {
            "check_row": 3,  # SZ3 = {D5, D8}
            "old_support": [("A", 5), ("A", 8)],
            "seam_pair": [("S", 1), ("S", 2)],
            "support": [("A", 5), ("A", 8), ("S", 1), ("S", 2)],
            "tile": [("A", 5), ("S", 1), ("A", 8), ("S", 2)],
        },
        "B": {
            "check_row": 0,  # SZ0 = {D0, D3}
            "old_support": [("B", 0), ("B", 3)],
            "seam_pair": [("S", 0), ("S", 1)],
            "support": [("S", 0), ("S", 1), ("B", 0), ("B", 3)],
            "tile": [("S", 0), ("B", 0), ("S", 1), ("B", 3)],
        },
    },
    "parity_support": [
        ("A", 2),
        ("A", 5),
        ("A", 8),
        ("B", 0),
        ("B", 3),
        ("B", 6),
    ],
    "parity_frame_bit_type": "Z",
    # The telescoped operator is X(A right col) (x) X(B left col); converting
    # to the canonical readout representatives X2 X4 X6 multiplies in
    # SX2*SX3 on A and SX0*SX1 on B (see the ZZ dual above).
    "telescope_reference_checks": {"A": [2, 3], "B": [0, 1]},
    # Split byproduct: the through-seam merged Z_L runs along the top row
    # (A.D0, A.D1, A.D2, S0, B.D0, B.D1, B.D2), so the top-row
    # Z_L(A) (x) Z_L(B) product picks up the seam outcome on S0. Converting
    # to the canonical readout representative Z0 Z4 Z8 on patch A multiplies
    # in SZ1(A) and SZ3(A) -- and SZ3(A) is A's grown check, whose value
    # changes by s1 (+) s2 at the split. The canonical-product flip is
    # therefore s0 (+) s1 (+) s2, fixed up by a conditional logical X on
    # patch B (frame X on X_L support D2, D4, D6).
    "byproduct_seam_indices": [0, 1, 2],
    "byproduct_patch": "B",
    "byproduct_logical": "X",
    "byproduct_frame_support": [2, 4, 6],
    "merged_X_L": [("A", 2), ("A", 4), ("A", 6)],
    "merged_Z_L": [
        ("A", 0),
        ("A", 1),
        ("A", 2),
        ("S", 0),
        ("B", 0),
        ("B", 1),
        ("B", 2),
    ],
}

SEAM_GEOMETRIES = {"ZZ": SEAM_GEOMETRY_ZZ, "XX": SEAM_GEOMETRY_XX}


def _support_row(support: Sequence[tuple[str, int]]) -> np.ndarray:
    row = np.zeros(MERGED_NUM_DATA, dtype=int)
    for elem in support:
        row[merged_index(elem)] = 1
    return row


def build_merged_check_matrices(
    kind: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build the merged-code check matrices for a ZZ or XX merge.

    Column convention: 0-8 = A.D0-D8, 9-11 = S0-S2, 12-20 = B.D0-D8.
    Row order groups patch checks first (A rows 0-3, B rows 4-7, with the
    grown boundary checks substituted in place at their patch row) followed
    by the four new seam checks (for the seam-check type only).

    Parameters
    ----------
    kind:
        `"ZZ"` (vertical merge, 8 X checks / 12 Z checks) or `"XX"`
        (horizontal merge, 12 X checks / 8 Z checks).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str], list[str]]
        `(H_X_merged, H_Z_merged, x_check_labels, z_check_labels)`.
    """
    geometry = SEAM_GEOMETRIES[kind]

    def patch_rows(H: np.ndarray, patch: str, grown: dict | None):
        rows = []
        labels = []
        check_type = "X" if H is BASE_H_X else "Z"
        offset = _A_OFFSET if patch == "A" else _B_OFFSET
        for i in range(H.shape[0]):
            if grown is not None and i == grown["check_row"]:
                rows.append(_support_row(grown["support"]))
                labels.append(f"{patch}.S{check_type}{i}_grown")
            else:
                row = np.zeros(MERGED_NUM_DATA, dtype=int)
                row[offset : offset + 9] = H[i]
                rows.append(row)
                labels.append(f"{patch}.S{check_type}{i}")
        return rows, labels

    grown_type = geometry["grown_check_type"]
    new_type = geometry["new_check_type"]
    grown_A = geometry["grown_checks"]["A"]
    grown_B = geometry["grown_checks"]["B"]

    matrices = {}
    labels = {}
    for check_type, H in (("X", BASE_H_X), ("Z", BASE_H_Z)):
        rows_A, labels_A = patch_rows(
            H, "A", grown_A if check_type == grown_type else None
        )
        rows_B, labels_B = patch_rows(
            H, "B", grown_B if check_type == grown_type else None
        )
        rows = rows_A + rows_B
        row_labels = labels_A + labels_B
        if check_type == new_type:
            for i, check in enumerate(geometry["new_checks"]):
                rows.append(_support_row(check["support"]))
                row_labels.append(f"SEAM_{new_type}{i}")
        matrices[check_type] = np.array(rows, dtype=int)
        labels[check_type] = row_labels

    return matrices["X"], matrices["Z"], labels["X"], labels["Z"]


# ---------------------------------------------------------------------------
# Merged syndrome-extraction circuits
# ---------------------------------------------------------------------------

TEMPLATE_QUBITS = {
    layout: layout_qubits(layout) for layout in ("surf17", "surf13", "surf10")
}
"""Template qubit labels per layout -- sourced from
codepack_surf17_tomita2014.layout_qubits, the single canonical definition."""

# Patch syndrome-extraction tiles (_X_TILE_DATA/_X_TILE_ROWS/etc.), the
# per-layout ancilla/mode specs (_LAYOUT_SE_SPECS), the shared 7-layer
# templates (_se_templates), and the default gate-duration/idle-name
# tables (_DEFAULT_GATE_DURATIONS/_DEFAULT_IDLE_GATES) are all imported
# from codepack_surf17_tomita2014 above -- this module only adds the
# grown-check substitution and the seam-specific machinery on top.


def _build_patch_se_block(
    layout: str,
    actual_map: dict,
    grown_type: str,
    grown_row: int,
    grown_tile_actual: list,
    all_labels: list,
    circuit_backend: type[BasePhysicalCircuit],
    counter: dict,
):
    """One patch's syndrome-extraction block with the grown-tile substitution.

    Returns `(circuit, labels_X, labels_Z)` where the label lists give, in
    H-row order, the `(ancilla_label, outcome_idx)` pair locating each
    check's outcome within the full merged-SE frame. `counter` tracks
    per-qubit measurement occurrences across blocks and is updated in place.
    """
    X_template, Z_template = _se_templates(circuit_backend)
    spec = _LAYOUT_SE_SPECS[layout]

    def make_tiles(tile_data, tile_rows, aux_names, check_type):
        tiles = []
        for exec_idx in range(4):
            row = tile_rows[exec_idx]
            aux = actual_map[aux_names[exec_idx]]
            if check_type == grown_type and row == grown_row:
                data_slots = list(grown_tile_actual)
            else:
                data_slots = [
                    actual_map[t] if t is not None else None
                    for t in tile_data[exec_idx]
                ]
            tiles.append(data_slots + [aux])
        return tiles

    x_tiles = make_tiles(_X_TILE_DATA, _X_TILE_ROWS, spec["X_aux"], "X")
    z_tiles = make_tiles(_Z_TILE_DATA, _Z_TILE_ROWS, spec["Z_aux"], "Z")

    mode = spec["mode"]
    if mode == "parallel":
        cx = circuit_backend.from_circuit_tiling(
            X_template, all_labels, x_tiles, merge_offsets=0
        )
        cz = circuit_backend.from_circuit_tiling(
            Z_template, all_labels, z_tiles, merge_offsets=0
        )
        block = cx.merge(cz, 0)
    elif mode == "blocks":
        cx = circuit_backend.from_circuit_tiling(
            X_template, all_labels, x_tiles, merge_offsets=0
        )
        cz = circuit_backend.from_circuit_tiling(
            Z_template, all_labels, z_tiles, merge_offsets=0
        )
        block = cx.append(cz)
    else:  # serial
        block = None
        for template, tiles in ((X_template, x_tiles), (Z_template, z_tiles)):
            for tile in tiles:
                c = circuit_backend.from_circuit_tiling(
                    template, all_labels, [tile], merge_offsets=0
                )
                block = c if block is None else block.append(c)

    # Assign measurement occurrences in temporal order (X tiles then Z tiles;
    # in "parallel" mode the X and Z ancilla sets are disjoint so the
    # relative order within the block is immaterial).
    labels_X: list = [None] * 4
    labels_Z: list = [None] * 4
    for tile_rows, aux_names, out in (
        (_X_TILE_ROWS, spec["X_aux"], labels_X),
        (_Z_TILE_ROWS, spec["Z_aux"], labels_Z),
    ):
        for exec_idx in range(4):
            aux = actual_map[aux_names[exec_idx]]
            occ = counter.get(aux, 0)
            counter[aux] = occ + 1
            out[tile_rows[exec_idx]] = (aux, occ)
    return block, labels_X, labels_Z


def _build_seam_block(
    geometry: dict,
    resolve,
    qubits_a: Sequence,
    all_labels: list,
    circuit_backend: type[BasePhysicalCircuit],
    counter: dict,
):
    """The four new seam checks, extracted with borrowed ancillas.

    Ancillas are borrowed from patch A (they are free after A's own block;
    every `Imrz` is measure-and-reset). When at least as many distinct
    ancillas are available as there are new checks (surf17/surf13), all
    four checks are tiled into a single 7-layer block: their seam-qubit
    supports pairwise overlap, but always at a different relative layer
    for each check that shares them (verified directly against the
    checks' tile role assignments), so merging is conflict-free -- see
    the CNOT-order comment on `_se_templates` for the analogous argument.
    Surf10 has only one ancilla to borrow, so the checks must stay serial
    there (a single ancilla cannot run two checks at once). Returns
    `(circuit, labels_seam)`.
    """
    X_template, Z_template = _se_templates(circuit_backend)
    template = (
        Z_template if geometry["new_check_type"] == "Z" else X_template
    )
    borrow = list(qubits_a[9:])
    new_checks = geometry["new_checks"]
    labels_seam = []

    def tile_for(i, check):
        aux = borrow[i % len(borrow)]
        tile = [
            resolve(e) if e is not None else None for e in check["tile"]
        ] + [aux]
        occ = counter.get(aux, 0)
        counter[aux] = occ + 1
        labels_seam.append((aux, occ))
        return tile

    if len(borrow) >= len(new_checks):
        tiles = [tile_for(i, check) for i, check in enumerate(new_checks)]
        block = circuit_backend.from_circuit_tiling(
            template, all_labels, tiles, merge_offsets=0
        )
    else:
        block = None
        for i, check in enumerate(new_checks):
            c = circuit_backend.from_circuit_tiling(
                template, all_labels, [tile_for(i, check)], merge_offsets=0
            )
            block = c if block is None else block.append(c)
    return block, labels_seam


def _build_reference_se_circuit(
    kind: str,
    qubits_a: Sequence,
    qubits_b: Sequence,
    seam_qubits: Sequence,
    idle_layout: str,
    circuit_backend: type[BasePhysicalCircuit],
):
    """A fully self-padded `se_circuit` for `idle_layout`, reusing the real
    data/seam qubit names but fresh scratch ancilla names sized for
    `idle_layout`'s own ancilla count.

    Used as a [](api:BasePhysicalCircuit.transplant_idle_schedule_inplace)
    reference: the per-patch blocks are structurally identical to
    codepack_surf17_tomita2014's own Syndrome Extraction circuit for
    existing data qubits (the grown-check substitution only fills
    previously-blank template slots with seam qubits, never changing an
    existing qubit's own role/timing -- verified directly), and the seam
    checks' geometry is layout-independent the same way (only the ancilla
    borrowing scheme differs per layout). Scratch ancillas never appear in
    the transplant's own qubit list, so their exact names don't matter --
    only that patch A's and patch B's scratch pools are disjoint from each
    other (they get merged together) and large enough for `idle_layout`.
    """
    geometry = SEAM_GEOMETRIES[kind]
    ref_template_qubits = layout_qubits(idle_layout)
    num_scratch_anc = len(ref_template_qubits) - 9
    scratch_a = [f"__idle_ref_anc_a{i}" for i in range(num_scratch_anc)]
    scratch_b = [f"__idle_ref_anc_b{i}" for i in range(num_scratch_anc)]

    data_a = list(qubits_a[:9])
    data_b = list(qubits_b[:9])
    map_a = dict(zip(ref_template_qubits, data_a + scratch_a))
    map_b = dict(zip(ref_template_qubits, data_b + scratch_b))

    def resolve(elem):
        patch, i = elem
        if patch == "A":
            return data_a[i]
        if patch == "S":
            return seam_qubits[i]
        return data_b[i]

    all_labels = data_a + scratch_a + data_b + scratch_b + list(seam_qubits)
    grown_type = geometry["grown_check_type"]
    grown_a = geometry["grown_checks"]["A"]
    grown_b = geometry["grown_checks"]["B"]
    counter: dict = {}

    block_a, *_ = _build_patch_se_block(
        idle_layout,
        map_a,
        grown_type,
        grown_a["check_row"],
        [resolve(e) if e is not None else None for e in grown_a["tile"]],
        all_labels,
        circuit_backend,
        counter,
    )
    block_b, *_ = _build_patch_se_block(
        idle_layout,
        map_b,
        grown_type,
        grown_b["check_row"],
        [resolve(e) if e is not None else None for e in grown_b["tile"]],
        all_labels,
        circuit_backend,
        counter,
    )
    # The reference seam block borrows from patch A's reference ancilla
    # pool, matching the real structure (seam checks borrow from qubits_a).
    seam_block, _ = _build_seam_block(
        geometry, resolve, data_a + scratch_a, all_labels, circuit_backend, counter
    )
    se_circuit_ref = block_a.merge(block_b, 0).append(seam_block)
    se_circuit_ref.pad_single_qubit_idles_by_duration_inplace(
        _DEFAULT_IDLE_GATES, _DEFAULT_GATE_DURATIONS
    )
    return se_circuit_ref


def _surgery_metadata(
    kind: str,
    patch_a_label: str,
    patch_b_label: str,
    qubits_a: Sequence,
    qubits_b: Sequence,
    seam_qubits: Sequence,
    layout: str,
    circuit_backend: type[BasePhysicalCircuit],
    idle_layout: str | None = None,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
) -> dict:
    """Circuits, outcome labels, and bookkeeping data shared by merge/split.

    `idle_layout`, if given, self-pads (`idle_layout == layout`) or borrows
    an equally-or-more-parallel layout's idle schedule (matching
    codepack_surf17_tomita2014.create_qec_code's convention), except that
    "surf10" cannot supply its own idle schedule either way -- its single
    shared ancilla forces heavy serialization in both the per-patch blocks
    and the seam-check extraction, so self-padding it would give a far
    larger idle budget than surf17/surf13, the same inflation problem the
    base codepack's idle_layout avoids.
    """
    geometry = SEAM_GEOMETRIES[kind]
    template_qubits = TEMPLATE_QUBITS[layout]
    assert len(qubits_a) == len(template_qubits), (
        f"qubits_a must be the full {layout} patch qubit list "
        f"({len(template_qubits)} labels, template order D0..D8 + ancillas)"
    )
    assert len(qubits_b) == len(template_qubits)
    assert len(seam_qubits) == 3, "A d=3 seam has exactly 3 qubits"

    map_a = dict(zip(template_qubits, qubits_a))
    map_b = dict(zip(template_qubits, qubits_b))

    def resolve(elem):
        patch, i = elem
        if patch == "A":
            return qubits_a[i]
        if patch == "S":
            return seam_qubits[i]
        return qubits_b[i]

    all_labels = list(qubits_a) + list(qubits_b) + list(seam_qubits)
    counter: dict = {}
    grown_type = geometry["grown_check_type"]
    grown_a = geometry["grown_checks"]["A"]
    grown_b = geometry["grown_checks"]["B"]

    block_a, labels_X_a, labels_Z_a = _build_patch_se_block(
        layout,
        map_a,
        grown_type,
        grown_a["check_row"],
        [resolve(e) if e is not None else None for e in grown_a["tile"]],
        all_labels,
        circuit_backend,
        counter,
    )
    block_b, labels_X_b, labels_Z_b = _build_patch_se_block(
        layout,
        map_b,
        grown_type,
        grown_b["check_row"],
        [resolve(e) if e is not None else None for e in grown_b["tile"]],
        all_labels,
        circuit_backend,
        counter,
    )
    seam_block, labels_seam = _build_seam_block(
        geometry, resolve, qubits_a, all_labels, circuit_backend, counter
    )
    # block_a and block_b act on entirely disjoint qubits (each patch's own
    # data + ancilla labels are distinct, even where template positions
    # match, e.g. "A9" resolves differently per patch) -- verified
    # conflict-free for every layout, so they can run in parallel rather
    # than sequentially.
    se_circuit = block_a.merge(block_b, 0).append(seam_block)

    if idle_layout is not None:
        if idle_layout not in _LAYOUT_PARALLELISM:
            raise ValueError(f"Unknown idle_layout: {idle_layout}")
        if _LAYOUT_PARALLELISM[idle_layout] > _LAYOUT_PARALLELISM[layout]:
            raise ValueError(
                f"idle_layout={idle_layout!r} is more serialized than layout={layout!r}; "
                "an idle schedule can only be borrowed from an equally or more parallel "
                "layout (e.g. layout='surf10' may use idle_layout='surf17', not the reverse)."
            )
        if idle_layout == layout == "surf10":
            raise ValueError(
                "surf10 has no well-defined idle schedule of its own for lattice "
                "surgery (its single shared ancilla forces heavy serialization in "
                "both the per-patch blocks and the seam-check extraction); "
                "specify idle_layout='surf17' or 'surf13' instead."
            )
        if idle_layout == layout:
            se_circuit.pad_single_qubit_idles_by_duration_inplace(
                idle_gates or _DEFAULT_IDLE_GATES,
                gate_durations or _DEFAULT_GATE_DURATIONS,
            )
        else:
            reference_se = _build_reference_se_circuit(
                kind, qubits_a, qubits_b, seam_qubits, idle_layout, circuit_backend
            )
            transplant_qubits = list(qubits_a[:9]) + list(qubits_b[:9]) + list(seam_qubits)
            se_circuit.transplant_idle_schedule_inplace(
                reference_se, transplant_qubits, list((idle_gates or _DEFAULT_IDLE_GATES).values())
            )

    # Seam prep / split-measurement circuits (defined on the seam qubits
    # only so error-injection tools enumerate a minimal set of locations)
    seam_list = list(seam_qubits)
    if geometry["seam_prep_basis"] == "X":
        prep_layers = [
            [("Imrz", s) for s in seam_list],
            [("Gh", s) for s in seam_list],
        ]
        split_layers = [
            [("Gh", s) for s in seam_list],
            [("Imrz", s) for s in seam_list],
        ]
    else:
        prep_layers = [[("Imrz", s) for s in seam_list]]
        split_layers = [[("Imrz", s) for s in seam_list]]
    prep_circuit = circuit_backend(prep_layers, qubit_labels=seam_list)
    split_circuit = circuit_backend(split_layers, qubit_labels=seam_list)

    byproduct_patch_label = (
        patch_b_label
        if geometry["byproduct_patch"] == "B"
        else patch_a_label
    )
    byproduct_qubits = (
        qubits_b if geometry["byproduct_patch"] == "B" else qubits_a
    )

    return {
        "kind": kind,
        "geometry": geometry,
        "all_labels": all_labels,
        "prep_circuit": prep_circuit,
        "se_circuit": se_circuit,
        "split_circuit": split_circuit,
        "labels_X_a": labels_X_a,
        "labels_Z_a": labels_Z_a,
        "labels_X_b": labels_X_b,
        "labels_Z_b": labels_Z_b,
        "labels_seam": labels_seam,
        "seam_history_key": (
            f"seam_check_history_{kind.lower()}"
            f"_{patch_a_label}_{patch_b_label}"
        ),
        "parity_frame_key": (
            f"surgery_parity_zz_{patch_a_label}_{patch_b_label}"
            if kind == "ZZ"
            else f"surgery_parity_xx_{patch_a_label}_{patch_b_label}"
        ),
        "repair_key": (
            f"split_byproduct_repair_{kind.lower()}_{patch_a_label}_{patch_b_label}"
        ),
        "support_a": [
            resolve(e) for e in geometry["parity_support"] if e[0] == "A"
        ],
        "support_b": [
            resolve(e) for e in geometry["parity_support"] if e[0] == "B"
        ],
        "parity_frame_bit_type": geometry["parity_frame_bit_type"],
        # FT-mode decode data: merged check matrix of the seam-check type
        # (row order A 0-3, B 0-3, seam 0-3, matching the window assembly
        # in the split bookkeeping) and the telescoped support columns.
        "new_check_type": geometry["new_check_type"],
        "merged_H_new": (
            build_merged_check_matrices(kind)[
                1 if geometry["new_check_type"] == "Z" else 0
            ]
        ),
        "support_cols": [
            merged_index(e) for e in geometry["parity_support"]
        ],
        # (patch_label, check_type, H row): boundary stabilizers whose
        # recorded values convert the telescoped seam-check product to the
        # canonical joint parity.
        "telescope_reference": [
            (patch_a_label, geometry["new_check_type"], row)
            for row in geometry["telescope_reference_checks"]["A"]
        ]
        + [
            (patch_b_label, geometry["new_check_type"], row)
            for row in geometry["telescope_reference_checks"]["B"]
        ],
        # (patch_label, check_type, H row, seam indices absorbed): after the
        # split, the re-formed boundary check equals the grown value XOR the
        # two seam outcomes, so the last recorded history entry is flipped
        # to keep the space-time diff syndrome continuous.
        "grown_flips": [
            (
                patch_a_label,
                grown_type,
                grown_a["check_row"],
                [e[1] for e in grown_a["seam_pair"]],
            ),
            (
                patch_b_label,
                grown_type,
                grown_b["check_row"],
                [e[1] for e in grown_b["seam_pair"]],
            ),
        ],
        "byproduct": {
            "seam_indices": geometry["byproduct_seam_indices"],
            "patch_label": byproduct_patch_label,
            "pauli": geometry["byproduct_logical"],
            "support": [
                byproduct_qubits[i]
                for i in geometry["byproduct_frame_support"]
            ],
        },
        "patch_a_label": patch_a_label,
        "patch_b_label": patch_b_label,
        "seam_qubits": seam_list,
    }


# ---------------------------------------------------------------------------
# FT merged-window decoding
# ---------------------------------------------------------------------------


def pymatching_merged_window_decode(
    H: np.ndarray,
    syndrome_window: Sequence[Sequence[int]],
    prev_round: Sequence[int],
    fresh_rows: Sequence[int],
    fresh_start: int = 0,
    noisy_prev: bool = False,
    ref_rows: Sequence[int] = (),
    continuing_rows: dict[int, int] | None = None,
    settled_rows: dict[int, int] | None = None,
) -> np.ndarray:
    """Decode a syndrome window on the 3D space-time graph of an arbitrary H.

    Generalizes the codepack's global-measurement graph construction (node
    `t * num_checks + row`; weight-2 columns become spatial edges, weight-1
    columns boundary edges, fault id = column; temporal edges weight 0.9)
    to any check matrix, so it can run on the 21-qubit merged code.

    Parameters
    ----------
    H:
        Binary check matrix (num_checks x num_data). Columns must have
        weight 1 or 2 (true for surface-code CSS sectors).

    syndrome_window:
        Raw check outcomes, one length-num_checks row per round.

    prev_round:
        Baseline outcomes for the round preceding the window; the round-0
        detector of a continuing check diffs against this.

    fresh_rows:
        Rows of checks first measured inside the window (the new seam
        checks). Their first measured round is a pure time boundary: no
        detector, no node. Consequences encoded in the graph: a fresh
        check's first-round reference absorbs one endpoint of any data
        error it watches in that round (so a column shared with a
        continuing check becomes a boundary edge on the continuing node),
        and a measurement error on it in that round is a boundary edge
        at its next-round node.

    fresh_start:
        Window layer at which the fresh rows are first measured. Before
        this layer fresh rows do not exist: their detectors are 0 and
        they participate in no edges (data columns they watch fall back
        to the continuing rows only, reproducing separate-patch physics
        for a window that extends back before the merge).

    noisy_prev:
        Set when `prev_round` holds raw measured values (e.g. the first
        QEC round after a prep that leaves this check type random) rather
        than trusted values. A measurement flip in that baseline round
        leaves a single defect at layer 0; escape edges (measurement
        weight) let the matcher explain it without a spurious data
        correction.

    ref_rows:
        Rows whose `prev_round` value the CALLER consumes as a reference
        (e.g. the telescoped-to-canonical conversion checks). Their
        layer-0 escape edges carry fault ids `num_data + 1 + k` (k the
        position in `ref_rows`): when the matcher routes a layer-0 defect
        on such a row to the time boundary it is asserting the reference
        value itself was mismeasured, and the caller must flip its read.
        This keeps the observable consistent for BOTH interpretations of
        the ambiguous single-defect signature (baseline measurement flip
        vs. a data error just after the baseline that the matcher chose
        to absorb).

    continuing_rows:
        Rows -> the window layer at which each one's own pre-window
        history stops being reachable (e.g. a shallower-history second
        patch that only starts contributing partway through a window
        sized to a deeper block's reach). Unlike `fresh_rows` (checks
        with no prior existence at all), a continuing row DOES have real
        history before its start layer; `prev_round` there is trusted
        the same (untrusted, `noisy_prev`-style) way as at layer 0, just
        time-shifted. No node/edges before the start layer.

    settled_rows:
        Same shape as `continuing_rows`, but for a baseline that is
        CONFIRMED rather than noisy (e.g. a row an earlier merge's own
        repair already examined and took full responsibility for). No
        escape edge: a new defect after it must be explained by a real
        correction, not cheaply blamed on the settled baseline -- using
        `continuing_rows` here would let the matcher absorb a genuine
        new defect for free instead.

    Notes
    -----
    The correction is referenced to the LAST window round (the joint
    parity is read there): every spatial edge carries its column as a
    fault id since a data error at any window time flips the last round's
    seam-check product. Only FRESH rows get a future-boundary edge at the
    last round: a final-round seam measurement error is undetectable inside
    the window yet flips the recorded product, so that edge carries the
    virtual fault id `num_data`. Continuing rows get no future boundary —
    their final-round measurement errors are caught by the patch QEC that
    resumes after the split, and giving them a cheap escape here misprices
    staggered mid-window defects (patch check and seam check firing one
    round apart) whose correct matching must flip the parity.

    Returns
    -------
    np.ndarray
        Length `num_data + 1` binary indicator: data-error corrections
        plus the virtual observable flip at index `num_data`.
    """
    import pymatching

    H = np.asarray(H)
    num_checks, num_data = H.shape
    num_rounds = len(syndrome_window)
    fresh = set(fresh_rows)
    continuing = dict(continuing_rows or {})
    settled = dict(settled_rows or {})
    assert not (fresh & continuing.keys()), (
        "a row cannot be both a fresh row and a continuing row"
    )
    assert not (fresh & settled.keys()), (
        "a row cannot be both a fresh row and a settled row"
    )
    assert not (continuing.keys() & settled.keys()), (
        "a row cannot be both a continuing row and a settled row"
    )
    # Rows that have no node before their own start layer, for the loops
    # below that don't need to distinguish WHY (continuing vs settled).
    late_start = {**continuing, **settled}

    matching = pymatching.Matching()
    matching.ensure_num_fault_ids(num_data + 1 + len(ref_rows))
    ref_fault_id = {
        row: num_data + 1 + k for k, row in enumerate(ref_rows)
    }
    for t in range(num_rounds):
        for j in range(num_data):
            rows = [i for i in range(num_checks) if H[i, j] == 1]
            if t <= fresh_start:
                rows = [i for i in rows if i not in fresh]
            rows = [
                i for i in rows if not (i in late_start and t < late_start[i])
            ]
            if len(rows) == 2:
                matching.add_edge(
                    t * num_checks + rows[0],
                    t * num_checks + rows[1],
                    fault_ids={j},
                    weight=1.0,
                    merge_strategy="smallest-weight",
                )
            elif len(rows) == 1:
                matching.add_boundary_edge(
                    t * num_checks + rows[0],
                    fault_ids={j},
                    weight=1.0,
                    merge_strategy="smallest-weight",
                )
    for t in range(num_rounds - 1):
        for i in range(num_checks):
            if i in fresh and t < fresh_start:
                continue  # fresh check does not exist yet: no node
            if i in late_start and t < late_start[i]:
                continue  # continuing/settled row does not exist yet: no node
            if t == fresh_start and i in fresh:
                # First-round measurement error on a fresh check: only
                # the next round's diff fires (its first round is its
                # reference). Weight 1.0 (not the 0.9 temporal weight):
                # paired with the future boundary it must not undercut
                # the weight-1.9 temporal+spatial path of a staggered
                # seam-qubit data error (caught by the serial seam
                # checks one round apart), whose seam-column fault id
                # keeps m correct.
                matching.add_boundary_edge(
                    (t + 1) * num_checks + i,
                    weight=1.0,
                    merge_strategy="smallest-weight",
                )
            else:
                # A continuing/settled row's own start round falls here
                # too: just the ordinary temporal edge onward from it.
                matching.add_edge(
                    t * num_checks + i,
                    (t + 1) * num_checks + i,
                    weight=0.9,
                    merge_strategy="smallest-weight",
                )
    if noisy_prev:
        for i in range(num_checks):
            if i in fresh or i in settled:
                # Fresh rows have no baseline-mismeasurement node; settled
                # rows have a CONFIRMED baseline, not a noisy one -- no
                # escape (see `settled_rows` above).
                continue
            s = continuing.get(i, 0)
            matching.add_boundary_edge(
                s * num_checks + i,
                fault_ids=(
                    {ref_fault_id[i]} if i in ref_fault_id else set()
                ),
                weight=0.9,
                merge_strategy="smallest-weight",
            )
    for i in fresh:
        matching.add_boundary_edge(
            (num_rounds - 1) * num_checks + i,
            fault_ids={num_data},
            weight=1.0,
            merge_strategy="smallest-weight",
        )

    detectors = []
    for t in range(num_rounds):
        for i in range(num_checks):
            if i in fresh and t <= fresh_start:
                detectors.append(0)
            elif i in late_start and t < late_start[i]:
                detectors.append(0)
            elif i in late_start and t == late_start[i]:
                detectors.append(syndrome_window[t][i] ^ prev_round[i])
            elif t == 0:
                detectors.append(syndrome_window[0][i] ^ prev_round[i])
            else:
                detectors.append(
                    syndrome_window[t][i] ^ syndrome_window[t - 1][i]
                )
    return matching.decode(np.array(detectors, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Merge / split bookkeeping
# ---------------------------------------------------------------------------

_PAULI_FROM_BITS = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}


def _multiply_pauli_into_frame(
    frame: PauliFrame, qubits: Sequence[str | int], pauli: str
) -> PauliFrame:
    """Multiply a Pauli string (X or Z on the given qubits) into a frame."""
    new_frame = frame.copy()
    for q in qubits:
        x_bit = new_frame.get_bit("X", q)
        z_bit = new_frame.get_bit("Z", q)
        if pauli == "Z":
            z_bit ^= 1
        elif pauli == "X":
            x_bit ^= 1
        else:
            raise ValueError(f"Unsupported byproduct Pauli: {pauli}")
        new_frame.pauli_frame[new_frame.qubit_labels.index(q)] = (
            _PAULI_FROM_BITS[(x_bit, z_bit)]
        )
    return new_frame


def _majority(bits: Sequence[int]) -> int:
    """Majority vote (ties resolve to 1; noiseless histories are constant)."""
    return int(2 * sum(bits) >= len(bits))


def _merge_bookkeeping_apply_fn(
    patches: PatchLayout,
    history: History,
    merge_outcomes,
    patch_a_label: str,
    patch_b_label: str,
    labels_X_a: list,
    labels_Z_a: list,
    labels_X_b: list,
    labels_Z_b: list,
    labels_seam: list,
    seam_history_key: str,
    num_merge_rounds: int,
    grown_rows: list,
    byproduct: dict,
    repair_key: str,
) -> Frame:
    """Accumulate merged-SE outcomes into per-patch and seam histories.

    Per-patch histories continue through the merge window with the grown
    boundary check recorded in place of the old one (a deterministic
    continuation, since the seam qubits are prepared in the grown check's
    eigenbasis), tracked on each patch's own `.data`. The four new seam
    checks go to `seam_history_key`, a `Frame`-global key namespaced by the
    (patch_a_label, patch_b_label) pair (overwritten fresh each merge, so
    consecutive surgeries do not mix). Unlike the per-patch histories, the
    seam checks have no single owning patch, so they remain on the global
    Frame for now.

    Also registers a [](api:PatchRelation) between the two patches holding
    `grown_rows`/`byproduct`/`repair_key`, so a later
    [](api:_split_byproduct_repair_apply_fn) can read them back instead of
    independently recomputing `_surgery_metadata`.
    """
    if isinstance(merge_outcomes, MeasurementOutcomes):
        merge_outcomes = [merge_outcomes]
    assert len(merge_outcomes) == num_merge_rounds, (
        f"Expected {num_merge_rounds} merged-SE frames, "
        f"got {len(merge_outcomes)}"
    )

    def row(labels, t):
        return [merge_outcomes[t][q][i] for (q, i) in labels]

    new_patches = patches.copy()
    for lbl, labels_X, labels_Z in (
        (patch_a_label, labels_X_a, labels_Z_a),
        (patch_b_label, labels_X_b, labels_Z_b),
    ):
        new_patch = patches[lbl].copy()
        hist_X = list(new_patch.data.get("syndrome_history_X", []))
        hist_Z = list(new_patch.data.get("syndrome_history_Z", []))
        for t in range(num_merge_rounds):
            hist_X.append(row(labels_X, t))
            hist_Z.append(row(labels_Z, t))
        new_patch.data["syndrome_history_X"] = hist_X
        new_patch.data["syndrome_history_Z"] = hist_Z
        new_patches[lbl] = new_patch

    new_patches.set_relation(
        PatchRelation(
            {"a": patch_a_label, "b": patch_b_label},
            data={
                "grown_rows": grown_rows,
                "byproduct": byproduct,
                "repair_key": repair_key,
            },
        )
    )

    out = {
        "patches": new_patches,
        seam_history_key: [
            row(labels_seam, t) for t in range(num_merge_rounds)
        ],
    }
    history.propagating_keys.add(seam_history_key)
    return Frame(out)


def _merge_bookkeeping_map_qubits_fn(
    qubit_mapping,
    labels_X_a,
    labels_Z_a,
    labels_X_b,
    labels_Z_b,
    labels_seam,
    byproduct,
    **kwargs,
):
    new_kwargs = kwargs.copy()
    for key, labels in (
        ("labels_X_a", labels_X_a),
        ("labels_Z_a", labels_Z_a),
        ("labels_X_b", labels_X_b),
        ("labels_Z_b", labels_Z_b),
        ("labels_seam", labels_seam),
    ):
        new_kwargs[key] = [
            (qubit_mapping.get(q, q), i) for (q, i) in labels
        ]
    new_byproduct = dict(byproduct)
    new_byproduct["support"] = [
        qubit_mapping.get(q, q) for q in byproduct["support"]
    ]
    new_kwargs["byproduct"] = new_byproduct
    return new_kwargs


def _split_bookkeeping_apply_fn(
    patches: PatchLayout,
    history: History,
    split_outcomes: MeasurementOutcomes,
    patch_a_label: str,
    patch_b_label: str,
    seam_qubits: list,
    seam_history_key: str,
    parity_frame_key: str,
    support_a: list,
    support_b: list,
    parity_frame_bit_type: str,
    telescope_reference: list,
    new_check_type: str,
    merged_H_new: np.ndarray,
    support_cols: list,
    grown_flips: list,
    byproduct: dict,
    mode: str,
) -> Frame:
    """Extract the joint parity and restore the two separate patches.

    - Joint parity: majority vote over merge rounds of the product of the
      four seam-check outcomes (the telescoped `Z_L Z_L` / `X_L X_L`
      operator), XORed with both patches' pending Pauli-frame bits on the
      telescoped support and with the recorded values of the boundary
      stabilizers that convert the telescoped representative to the
      canonical one (`telescope_reference`; those values are random for
      preps that do not stabilize the seam-check type).
    - Post-split boundary flip: the re-formed boundary check's value equals
      the grown value XOR its two seam outcomes, so EVERY stored history
      entry of that check row is XORed with the seam pair. Decoding is
      fully deferred, so this leaves the diff syndrome defect-free at all
      interior transitions; the only residue lands in the round-0 absolute
      layer. The residue is exported to the affected patch's own tracked
      data (`patch.data["syndrome_round0_offset_X"]` /
      `["syndrome_round0_offset_Z"]`, same convention as
      `syndrome_history_X/Z`) so a terminating measure in the grown-check
      basis can either drop round 0 (`reference_round_mode_X="guarded_diff"`
      (ZZ) / `reference_round_mode_Z="guarded_diff"` (XX)) or - when the
      prep makes that check type deterministic in round 0, as in the mzz
      Bell prep - keep round 0 as a detector layer (`"raw"`, the default)
      and subtract the offset, closing the prep blind window.
    - Split byproduct: the XOR of the three seam outcomes conditionally
      multiplies a logical Pauli into one patch's frame, preserving the
      anticommuting logical correlation across the surgery for the
      codepack's canonical readout representatives (see the geometry
      constants for the derivation).
    """
    assert mode in ("simple", "ft")
    seam_outcomes = [split_outcomes[q][0] for q in seam_qubits]

    try:
        last = history[-1]
        seam_hist = list(last.get(seam_history_key, []) or [])
    except (IndexError, AttributeError):
        seam_hist = []
    assert seam_hist, (
        "Split bookkeeping found no seam-check history; a merge instruction "
        "must precede the split in the same shot"
    )

    num_rounds = len(seam_hist)
    if mode == "simple":
        # Majority vote of the telescoped seam-check product over the
        # window; reference values read from the LAST recorded entry, so
        # mid-window data errors on the reference checks self-cancel.
        round_parities = []
        for round_vals in seam_hist:
            p = 0
            for b in round_vals:
                p ^= b
            round_parities.append(p)
        m_raw = _majority(round_parities)
        reference_correction = 0
        for lbl, check_type, row in telescope_reference:
            hist = (
                patches[lbl].data.get(f"syndrome_history_{check_type}", [])
                or []
            )
            assert hist, (
                "Split bookkeeping needs pre-existing syndrome history "
                "to reference the telescoped-to-canonical conversion; "
                "run QEC on both patches before the surgery"
            )
            reference_correction ^= hist[-1][row]
    else:  # ft
        # Merged-window matching over the FULL history since prep: the
        # window runs from QEC round 1 through the last merge round (A
        # rows, B rows, seam rows; seam rows zero-padded and edge-free
        # before the merge), diffing against round 0 as a noisy reference.
        # This is the SAME reference the terminating per-patch decode
        # uses, so any single fault is either visible to both decoders
        # (both correct it) or baked into both references (both absorb it
        # into the state) - never split between them, which is what kept
        # breaking the m/readout consistency when the window started at
        # the merge. m is read from the LAST window round and corrected
        # by the matched errors crossing the telescoped support; any
        # syndrome-consistent correction has the same support parity
        # because the telescoped operator is a product of merged-code
        # stabilizers. Telescope reference values come from ROUND 0 (the
        # shared reference layer): errors after round 0 belong to the
        # matching, whose support-crossing corrections convert m
        # consistently.
        #
        # Each of the 8 rows reaches back to round 0 independently,
        # unless an earlier merge's own repair already claimed it (see
        # `byproduct_settled_{new_check_type}` in
        # [](api:_split_byproduct_repair_apply_fn)), in which case it
        # anchors no further back than that repair's last round.
        hist_a = (
            patches[patch_a_label].data.get(
                f"syndrome_history_{new_check_type}", []
            )
            or []
        )
        hist_b = (
            patches[patch_b_label].data.get(
                f"syndrome_history_{new_check_type}", []
            )
            or []
        )
        assert len(hist_a) > num_rounds and len(hist_b) > num_rounds, (
            "FT surgery decoding needs pre-merge QEC history on both "
            "patches to anchor the merge window"
        )
        settled_a = (
            patches[patch_a_label].data.get(
                f"byproduct_settled_{new_check_type}", {}
            )
            or {}
        )
        settled_b = (
            patches[patch_b_label].data.get(
                f"byproduct_settled_{new_check_type}", {}
            )
            or {}
        )
        hists = [hist_a] * 4 + [hist_b] * 4
        settled_maps = [settled_a] * 4 + [settled_b] * 4
        local_rows = [0, 1, 2, 3] * 2
        is_settled = [local_rows[i] in settled_maps[i] for i in range(8)]
        # Per-row (0-3 = A, 4-7 = B) reach: round 0 by default, or an
        # earlier repair's last examined round if this row is settled.
        reach_start = [
            settled_maps[i].get(local_rows[i], 0) for i in range(8)
        ]
        k_shared_rows = [
            len(hists[i]) - num_rounds - reach_start[i] for i in range(8)
        ]
        k_shared = max(k_shared_rows)
        fresh_start = k_shared - 1
        num_window = k_shared - 1 + num_rounds
        # Window round at which each row's own contribution starts (0
        # for whichever row(s) set `k_shared`, i.e. reach the deepest).
        row_offset = [k_shared - ks for ks in k_shared_rows]

        def row_value(i: int, t: int) -> int:
            if t < row_offset[i]:
                return 0
            idx = reach_start[i] + 1 + (t - row_offset[i])
            return hists[i][idx][local_rows[i]]

        window = [
            [row_value(i, t) for i in range(8)]
            + (
                list(seam_hist[t - fresh_start])
                if t >= fresh_start
                else [0, 0, 0, 0]
            )
            for t in range(num_window)
        ]
        prev_round = [
            hists[i][reach_start[i]][local_rows[i]] for i in range(8)
        ] + [0, 0, 0, 0]
        # Merged-row indices of the telescope-reference checks (A rows
        # 0-3, B rows 4-7): their layer-0 escapes carry fault ids so the
        # reference read below stays consistent with the matching.
        ref_rows = [
            (row if lbl == patch_a_label else 4 + row)
            for lbl, check_type, row in telescope_reference
        ]
        num_data = merged_H_new.shape[1]
        # A shorter-patch-history offset is a NOISY baseline
        # (`continuing_rows`); a settled-row offset is CONFIRMED
        # (`settled_rows`) -- see [](api:pymatching_merged_window_decode).
        continuing_rows = {
            i: row_offset[i]
            for i in range(8)
            if row_offset[i] > 0 and not is_settled[i]
        }
        settled_rows_arg = {
            i: row_offset[i]
            for i in range(8)
            if row_offset[i] > 0 and is_settled[i]
        }
        correction = pymatching_merged_window_decode(
            merged_H_new,
            window,
            prev_round,
            fresh_rows=range(8, 12),
            fresh_start=fresh_start,
            noisy_prev=True,
            ref_rows=ref_rows,
            continuing_rows=continuing_rows,
            settled_rows=settled_rows_arg,
        )
        m_raw = 0
        for b in seam_hist[-1]:
            m_raw ^= b
        for col in support_cols:
            m_raw ^= int(correction[col])
        # Virtual fault: an undetectable final-round seam measurement
        # error flipped the recorded product, not the state.
        m_raw ^= int(correction[num_data])
        # Telescope references at each patch's own anchor round: errors
        # after it belong to the matching, and a layer-0 escape on a
        # reference row means the matcher judged the anchor value itself
        # mismeasured - flip the read.
        reference_correction = 0
        for k, (lbl, check_type, row) in enumerate(telescope_reference):
            merged_row = row if lbl == patch_a_label else 4 + row
            reference_correction ^= prev_round[merged_row]
            reference_correction ^= int(correction[num_data + 1 + k])

    frame_correction = 0
    for lbl, support in (
        (patch_a_label, support_a),
        (patch_b_label, support_b),
    ):
        pauli_frame = patches[lbl].pauli_frame
        for q in support:
            frame_correction ^= pauli_frame.get_bit(
                parity_frame_bit_type, q
            )

    out = {
        parity_frame_key: m_raw ^ frame_correction ^ reference_correction,
        seam_history_key: [],  # consumed
    }

    # Both the post-split boundary-check flips below and the conditional
    # logical byproduct mutate individual patches' `.data`/Pauli frame, so
    # lazily copy each touched patch exactly once into `new_patches`.
    new_patches = patches.copy()
    copied_labels: set = set()

    def get_mutable_patch(lbl):
        if lbl not in copied_labels:
            new_patches[lbl] = patches[lbl].copy()
            copied_labels.add(lbl)
        return new_patches[lbl]

    # Post-split boundary-check flips: rewriting the full stored history
    # (decoding is deferred, nothing has consumed it yet) keeps every
    # interior diff transition defect-free; the residue lands only in the
    # round-0 absolute layer. The residue is a KNOWN classical bit, so it
    # is also exported as a round-0 offset, tracked on the patch's own
    # data (same convention as syndrome_history_X/Z above): a terminating
    # decode that keeps round 0 as a detector layer (possible when the
    # prep basis makes this check type deterministic, e.g. the X sector
    # after |+> prep) subtracts it and loses no prep-fault coverage.
    for lbl, check_type, check_row, seam_pair in grown_flips:
        patch = get_mutable_patch(lbl)
        key = f"syndrome_history_{check_type}"
        hist = list(patch.data.get(key, []) or [])
        flip = 0
        for si in seam_pair:
            flip ^= seam_outcomes[si]
        if flip:
            for t in range(len(hist)):
                row = list(hist[t])
                row[check_row] ^= 1
                hist[t] = row
        patch.data[key] = hist
        offset_key = f"syndrome_round0_offset_{check_type}"
        offset = list(patch.data.get(offset_key, []) or [0] * len(hist[0]))
        offset[check_row] ^= flip
        patch.data[offset_key] = offset

    # Conditional logical byproduct
    byproduct_bit = 0
    for si in byproduct["seam_indices"]:
        byproduct_bit ^= seam_outcomes[si]
    if byproduct_bit == 1:
        patch = get_mutable_patch(byproduct["patch_label"])
        new_frame = _multiply_pauli_into_frame(
            patch.pauli_frame, byproduct["support"], byproduct["pauli"]
        )
        new_patches[byproduct["patch_label"]] = patch.copy(
            pauli_frame=new_frame
        )
    out["patches"] = new_patches

    return Frame(out)


def _split_bookkeeping_map_qubits_fn(
    qubit_mapping, seam_qubits, support_a, support_b, byproduct, **kwargs
):
    new_kwargs = kwargs.copy()
    new_kwargs["seam_qubits"] = [
        qubit_mapping.get(q, q) for q in seam_qubits
    ]
    new_kwargs["support_a"] = [qubit_mapping.get(q, q) for q in support_a]
    new_kwargs["support_b"] = [qubit_mapping.get(q, q) for q in support_b]
    new_byproduct = dict(byproduct)
    new_byproduct["support"] = [
        qubit_mapping.get(q, q) for q in byproduct["support"]
    ]
    new_kwargs["byproduct"] = new_byproduct
    return new_kwargs


def _byproduct_repair_row_decode(
    hist: list,
    check_type: str,
    row: int,
    anchor: int,
    num_post_split_rounds: int,
) -> int:
    """Discriminate a genuine seam-qubit fault from an ordinary bulk fault
    on `row` (a grown-check row) using a real matching decode, instead of
    [](api:_split_byproduct_repair_apply_fn)'s raw persistence check.

    `row`'s reformed support always has one column private to it (weight
    1 in the patch's own check matrix, e.g. `D2` for `SX1 = {D1, D2}`)
    and one shared with an adjacent row (weight 2, `D1` here). A genuine
    seam fault and a bulk fault on the PRIVATE column look identical --
    intentionally not discriminated, since `_split_byproduct_repair_apply_fn`
    already treats that ambiguity as self-correcting. Only a bulk fault
    on the SHARED column (correlated with the adjacent row) is real and
    needs excluding, which the raw single-row heuristic never checked.

    Decodes `[hist[-anchor-1], hist[-anchor], hist[-num_post_split_rounds:]]`,
    skipping the merge-round entries (the split's history rewrite already
    makes every round read as if the reformed check were used
    throughout, so comparing snapshots across the gap is exact).
    Including the anchor as its own window layer -- not a `noisy_prev`
    escape -- matters: a one-round anchor glitch then costs two ordinary
    temporal edges (reverting), while a true persistent defect (never
    reverting) is cheapest via a single data correction; a `noisy_prev`
    escape there would let the matcher blame ANY persistent defect on
    the anchor for free, defeating the discrimination.
    """
    H = BASE_H_X if check_type == "X" else BASE_H_Z
    col_weights = H.sum(axis=0)
    private_cols = [
        j for j in range(H.shape[1]) if H[row, j] == 1 and col_weights[j] == 1
    ]
    window = [hist[-anchor]] + hist[-num_post_split_rounds:]
    prev_round = hist[-anchor - 1]
    correction = pymatching_merged_window_decode(
        H, window, prev_round, fresh_rows=(), noisy_prev=False
    )
    return int(any(correction[j] for j in private_cols))


def _split_byproduct_repair_apply_fn(
    patches: PatchLayout,
    history: History,
    patch_a_label: str,
    patch_b_label: str,
    num_merge_rounds: int,
    num_post_split_rounds: int,
    fire_rule: str = "both",
    defect_decode_mode: Literal["heuristic", "matching"] = "heuristic",
) -> Frame:
    """Repair the split byproduct against a middle-seam fault.

    The split byproduct (XOR of the destructive seam outcomes -> one
    conditional logical Pauli frame) is read from raw measurement data.
    An effective seam-basis-conjugate error on a seam qubit anywhere in
    the merge window makes that record wrong relative to the projected
    state. For the two OUTER seam qubits this is self-correcting: the
    fault also leaves exactly one defect in the adjacent patch's grown
    boundary-check row, and that patch's termination decoder
    boundary-matches it with a correction that crosses the logical
    readout representative, cancelling the wrong byproduct in the
    two-patch parity. The MIDDLE seam qubit sits in BOTH grown checks,
    so both patches boundary-match independently: three net flips, and
    the logical correlation breaks under a single fault.

    This bookkeeping step (run after at least one post-split QEC round
    on both patches) reads each patch's grown-check-row defect between
    the last pre-merge round and the post-split rounds (the uniform
    grown-row history rewrite at split cancels in this XOR) and applies
    one extra byproduct Pauli when the signature indicates a net-odd
    flip count. Which signature that is depends on which termination
    decoders supply compensating flips - `fire_rule`:

    - "both" (mzz Bell prep; XX merge of the surgery CNOT): BOTH
      patches are eventually decoded in the byproduct-sensitive sector,
      so single one-sided (outer-seam) faults self-correct and only the
      two-sided middle-seam signature `d_A AND d_B` is net-odd. No
      other single fault fakes it: post-split the patches are disjoint
      circuits, so any single data/measurement fault flips at most one
      patch's grown-row endpoints.
    - "b_only" (ZZ merge of the surgery CNOT): only patch A (the
      control) gets a byproduct-sensitive termination decode - patch B
      is the ancilla, destructively measured in the conjugate basis, so
      its grown-check defect never earns a compensating flip. The
      middle-seam class (d_A AND d_B) is then already net-even, and the
      uncompensated class is the B-side outer seam qubit:
      `d_B AND NOT d_A`. A one-sided signature is easier for an
      unrelated single fault to fake than a two-sided one, so d_B is
      hardened: the defect must be present in EVERY post-split round
      (rejects a final-round measurement error of the grown check) and
      the anchor round must agree with the round before it (rejects a
      measurement error in the last pre-merge round, which would offset
      the whole comparison).

    `defect_decode_mode` selects how each row's `defect[lbl]` bit is
    computed: `"heuristic"` (default) is the raw check above, kept for
    backward compatibility; `"matching"` replaces it with a real per-row
    decode ([](api:_byproduct_repair_row_decode)) that fixes a real
    false-fire the heuristic has -- an ordinary bulk fault on the row's
    shared-column neighbor looks identical to a genuine seam fault, but
    the heuristic never checks that neighbor. Only used for
    `fire_rule == "b_only"`; `"both"` already can't be faked by a
    single fault and needs no such refinement.

    Also marks each row in `grown_rows` "settled" through its last
    examined round (`patch.data["byproduct_settled_{check_type}"][row]`),
    regardless of `fire`, since this function has now taken full
    responsibility for that row -- a later merge reaching back through
    the same row's history (see [](api:_split_bookkeeping_apply_fn))
    must not also explain the same defect via an ordinary correction.
    """
    relation = patches.get_relation(patch_a_label, patch_b_label)
    assert relation is not None, (
        f"No PatchRelation registered for ({patch_a_label}, {patch_b_label}) "
        "-- a matching build_merge_instruction must run first in the same shot"
    )
    grown_rows = relation.data["grown_rows"]
    byproduct = relation.data["byproduct"]
    repair_key = relation.data["repair_key"]

    anchor = num_merge_rounds + num_post_split_rounds + 1
    min_rounds = anchor if fire_rule == "both" else anchor + 1
    defect = {}
    glitch = {}
    settled_at = {}
    for lbl, check_type, row in grown_rows:
        hist = patches[lbl].data.get(f"syndrome_history_{check_type}", []) or []
        assert len(hist) >= min_rounds, (
            "Split byproduct repair needs pre-merge QEC history and at "
            "least one post-split QEC round on both patches "
            f"(have {len(hist)} rounds, need >= {min_rounds})"
        )
        if fire_rule == "both":
            defect[lbl] = hist[-1][row] ^ hist[-anchor][row]
        elif defect_decode_mode == "matching":
            defect[lbl] = _byproduct_repair_row_decode(
                hist, check_type, row, anchor, num_post_split_rounds
            )
        else:
            base = hist[-anchor][row]
            defect[lbl] = int(
                all(
                    hist[-j][row] ^ base
                    for j in range(1, num_post_split_rounds + 1)
                )
            )
            glitch[lbl] = hist[-anchor][row] ^ hist[-anchor - 1][row]
        settled_at[(lbl, check_type, row)] = len(hist) - 1
    if fire_rule == "both":
        fire = defect[patch_a_label] & defect[patch_b_label]
    elif fire_rule == "b_only":
        if defect_decode_mode == "matching":
            # The decode already resolves the anchor-glitch ambiguity
            # via temporal edges, so no separate glitch check is needed.
            fire = defect[patch_b_label] & (1 - defect[patch_a_label])
        else:
            fire = (
                defect[patch_b_label]
                & (1 - defect[patch_a_label])
                & (1 - glitch[patch_b_label])
            )
    else:
        raise ValueError(f"Unknown fire_rule '{fire_rule}'")

    new_patches = patches.copy()
    copied_labels: set = set()

    def get_mutable_patch(lbl):
        if lbl not in copied_labels:
            new_patches[lbl] = patches[lbl].copy()
            copied_labels.add(lbl)
        return new_patches[lbl]

    for (lbl, check_type, row), idx in settled_at.items():
        patch = get_mutable_patch(lbl)
        key = f"byproduct_settled_{check_type}"
        settled = dict(patch.data.get(key, {}) or {})
        settled[row] = idx
        patch.data[key] = settled

    if fire:
        patch = get_mutable_patch(byproduct["patch_label"])
        new_patches[byproduct["patch_label"]] = patch.copy(
            pauli_frame=_multiply_pauli_into_frame(
                patch.pauli_frame, byproduct["support"], byproduct["pauli"]
            )
        )
    return Frame({repair_key: fire, "patches": new_patches})


# ---------------------------------------------------------------------------
# Public instruction builders
# ---------------------------------------------------------------------------


def _merge_instruction_parts(
    meta: dict, num_merge_rounds: int, name: str
) -> tuple[Instruction, Instruction, Instruction]:
    """(seam prep, one merged-SE round, merge bookkeeping) instructions."""
    prep_inst = builders.build_physical_circuit_instruction(
        meta["prep_circuit"], name=f"{name} (seam prep)"
    )
    se_inst = builders.build_physical_circuit_instruction(
        meta["se_circuit"], name=f"{name} (merged syndrome extraction)"
    )
    bookkeeping_inst = Instruction(
        _merge_bookkeeping_apply_fn,
        data={
            "patch_a_label": meta["patch_a_label"],
            "patch_b_label": meta["patch_b_label"],
            "labels_X_a": meta["labels_X_a"],
            "labels_Z_a": meta["labels_Z_a"],
            "labels_X_b": meta["labels_X_b"],
            "labels_Z_b": meta["labels_Z_b"],
            "labels_seam": meta["labels_seam"],
            "seam_history_key": meta["seam_history_key"],
            "num_merge_rounds": num_merge_rounds,
            "grown_rows": [
                (lbl, check_type, row)
                for lbl, check_type, row, _ in meta["grown_flips"]
            ],
            "byproduct": meta["byproduct"],
            "repair_key": meta["repair_key"],
        },
        map_qubits_fn=_merge_bookkeeping_map_qubits_fn,
        param_priorities={
            "merge_outcomes": [f"history[-{num_merge_rounds}:]"]
        },
        param_aliases={"merge_outcomes": "measurement_outcomes"},
        name=f"{name} (bookkeeping)",
    )
    return prep_inst, se_inst, bookkeeping_inst


def _merge_instruction_from_metadata(
    meta: dict, num_merge_rounds: int, name: str
) -> Instruction:
    prep_inst, se_inst, bookkeeping_inst = _merge_instruction_parts(
        meta, num_merge_rounds, name
    )
    return builders.build_composite_instruction(
        [prep_inst] + [se_inst] * num_merge_rounds + [bookkeeping_inst],
        name=name,
    )


def _split_instruction_parts(
    meta: dict, mode: str, name: str
) -> tuple[Instruction, Instruction]:
    """(seam measurement, split bookkeeping) instructions."""
    split_circ_inst = builders.build_physical_circuit_instruction(
        meta["split_circuit"], name=f"{name} (seam measurement)"
    )
    bookkeeping_inst = Instruction(
        _split_bookkeeping_apply_fn,
        data={
            "patch_a_label": meta["patch_a_label"],
            "patch_b_label": meta["patch_b_label"],
            "seam_qubits": meta["seam_qubits"],
            "seam_history_key": meta["seam_history_key"],
            "parity_frame_key": meta["parity_frame_key"],
            "support_a": meta["support_a"],
            "support_b": meta["support_b"],
            "parity_frame_bit_type": meta["parity_frame_bit_type"],
            "telescope_reference": meta["telescope_reference"],
            "new_check_type": meta["new_check_type"],
            "merged_H_new": meta["merged_H_new"],
            "support_cols": meta["support_cols"],
            "grown_flips": meta["grown_flips"],
            "byproduct": meta["byproduct"],
            "mode": mode,
        },
        map_qubits_fn=_split_bookkeeping_map_qubits_fn,
        param_priorities={"split_outcomes": ["history[-1]"]},
        param_aliases={"split_outcomes": "measurement_outcomes"},
        name=f"{name} (bookkeeping)",
    )
    return split_circ_inst, bookkeeping_inst


def _split_instruction_from_metadata(
    meta: dict, mode: str, name: str
) -> Instruction:
    split_circ_inst, bookkeeping_inst = _split_instruction_parts(
        meta, mode, name
    )
    return builders.build_composite_instruction(
        [split_circ_inst, bookkeeping_inst],
        name=name,
    )


def build_merge_instruction(
    kind: str,
    patch_a_label: str,
    patch_b_label: str,
    qubits_a: Sequence,
    qubits_b: Sequence,
    seam_qubits: Sequence,
    layout: str,
    num_merge_rounds: int = 3,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    idle_layout: str | None = None,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str | None = None,
) -> Instruction:
    """Merge two patches through a seam for `num_merge_rounds` SE rounds.

    Composite of seam preparation (|+> for ZZ, |0> for XX), the merged
    syndrome-extraction circuit repeated `num_merge_rounds` times, and a
    bookkeeping step that extends the per-patch namespaced syndrome
    histories (with the grown boundary checks substituted in place),
    records the four new seam checks, and registers a
    [](api:PatchRelation) between the two patches (`grown_rows`/
    `byproduct`/`repair_key`) for a later
    [](api:build_split_byproduct_repair_instruction) to read. Must be
    followed by the matching [](api:build_split_instruction) in the same
    shot.

    Parameters
    ----------
    kind:
        `"ZZ"` (vertical merge measuring Z_L Z_L) or `"XX"` (horizontal).

    patch_a_label, patch_b_label:
        Patch labels. For a ZZ merge A is the top patch; for XX, the left.

    qubits_a, qubits_b:
        Full patch qubit lists (template order D0..D8 + ancillas).

    seam_qubits:
        The 3 seam qubits (left-to-right for ZZ, top-to-bottom for XX).

    layout:
        One of "surf17", "surf13", "surf10" (both patches must match).

    num_merge_rounds:
        Number of merged syndrome-extraction rounds (default 3 = d).

    circuit_backend:
        The circuit backend. Default is PyGSTiPhysicalCircuit.

    idle_layout, gate_durations, idle_gates:
        See [](api:_surgery_metadata); idle padding is off by default.

    name:
        Name for logging purposes.

    Returns
    -------
    Instruction
        The composite merge instruction.
    """
    meta = _surgery_metadata(
        kind,
        patch_a_label,
        patch_b_label,
        qubits_a,
        qubits_b,
        seam_qubits,
        layout,
        circuit_backend,
        idle_layout=idle_layout,
        gate_durations=gate_durations,
        idle_gates=idle_gates,
    )
    return _merge_instruction_from_metadata(
        meta, num_merge_rounds, name or f"Lattice-Surgery {kind} Merge"
    )


def build_split_instruction(
    kind: str,
    patch_a_label: str,
    patch_b_label: str,
    qubits_a: Sequence,
    qubits_b: Sequence,
    seam_qubits: Sequence,
    layout: str,
    mode: str = "simple",
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    idle_layout: str | None = None,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str | None = None,
) -> Instruction:
    """Split a merged patch pair and extract the joint logical parity.

    Composite of the seam measurement (X basis for ZZ, Z basis for XX) and a
    bookkeeping step that stores the joint parity under the frame key
    `surgery_parity_zz_{patch_a_label}_{patch_b_label}` /
    `surgery_parity_xx_{patch_a_label}_{patch_b_label}`, applies the post-split
    boundary-check flips to the per-patch histories, and injects the
    conditional logical byproduct into one patch's Pauli frame. Parameters
    must match the preceding [](api:build_merge_instruction).

    `mode="simple"` majority-votes the seam-check product with per-patch
    decoding untouched; `mode="ft"` additionally decodes the merge window
    on the merged matching graph.

    The split rewrites the grown-check rows of both patches' stored
    syndrome histories (see [](api:_split_bookkeeping_apply_fn)), leaving a
    residue only in the round-0 absolute detector layer, exported to the
    affected patch's own tracked data (`patch.data["syndrome_round0_offset_X"]`
    / `["syndrome_round0_offset_Z"]`). Subsequent FT logical measurements in
    the grown-check basis must either pass
    `reference_round_mode_X="guarded_diff"` after a ZZ surgery
    (`reference_round_mode_Z="guarded_diff"` after XX) or - when the prep
    makes that check type deterministic in round 0 - keep round 0 as a
    detector layer (`"raw"`, the default), in which case the terminating
    decode subtracts the offset; measurements in the other basis are
    unaffected.

    `idle_layout`, `gate_durations`, `idle_gates`: see
    [](api:_surgery_metadata); must match the preceding
    [](api:build_merge_instruction) call's values.
    """
    meta = _surgery_metadata(
        kind,
        patch_a_label,
        patch_b_label,
        qubits_a,
        qubits_b,
        seam_qubits,
        layout,
        circuit_backend,
        idle_layout=idle_layout,
        gate_durations=gate_durations,
        idle_gates=idle_gates,
    )
    return _split_instruction_from_metadata(
        meta, mode, name or f"Lattice-Surgery {kind} Split"
    )


def build_surgery_parity_instruction(
    kind: str,
    patch_a_label: str,
    patch_b_label: str,
    qubits_a: Sequence,
    qubits_b: Sequence,
    seam_qubits: Sequence,
    layout: str,
    mode: str = "simple",
    num_merge_rounds: int = 3,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    idle_layout: str | None = None,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str | None = None,
) -> Instruction:
    """Full merge+split lattice-surgery joint parity measurement.

    Measures `Z_L(A) Z_L(B)` (kind="ZZ") or `X_L(A) X_L(B)` (kind="XX")
    non-destructively via `num_merge_rounds` rounds of merged syndrome
    extraction; the result is stored under the frame key
    `surgery_parity_zz_{patch_a_label}_{patch_b_label}` /
    `surgery_parity_xx_{patch_a_label}_{patch_b_label}`. See
    [](api:build_merge_instruction) and [](api:build_split_instruction)
    for parameters and bookkeeping details.
    """
    base_name = name or f"Lattice-Surgery {kind} Parity ({mode})"
    meta = _surgery_metadata(
        kind,
        patch_a_label,
        patch_b_label,
        qubits_a,
        qubits_b,
        seam_qubits,
        layout,
        circuit_backend,
        idle_layout=idle_layout,
        gate_durations=gate_durations,
        idle_gates=idle_gates,
    )
    merge_inst = _merge_instruction_from_metadata(
        meta, num_merge_rounds, f"{base_name} merge"
    )
    split_inst = _split_instruction_from_metadata(
        meta, mode, f"{base_name} split"
    )
    return builders.build_composite_instruction(
        [merge_inst, split_inst],
        name=base_name,
    )


def build_surgery_parity_instruction_sequence(
    kind: str,
    patch_a_label: str,
    patch_b_label: str,
    qubits_a: Sequence,
    qubits_b: Sequence,
    seam_qubits: Sequence,
    layout: str,
    mode: str = "ft",
    num_merge_rounds: int = 3,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    idle_layout: str | None = None,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str | None = None,
) -> list[Instruction]:
    """The surgery parity measurement as a flat instruction list.

    Semantically identical to [](api:build_surgery_parity_instruction) but
    returns `[seam prep, merged SE (x num_merge_rounds), merge bookkeeping,
    seam measurement, split bookkeeping]` as separate stack-ready
    instructions, so error-injection tools (which attach `error_injections`
    kwargs to individual stack entries) can target a single merged-SE round
    or the seam circuits directly.

    `idle_layout`, `gate_durations`, `idle_gates`: see
    [](api:_surgery_metadata).
    """
    base_name = name or f"Lattice-Surgery {kind} Parity ({mode})"
    meta = _surgery_metadata(
        kind,
        patch_a_label,
        patch_b_label,
        qubits_a,
        qubits_b,
        seam_qubits,
        layout,
        circuit_backend,
        idle_layout=idle_layout,
        gate_durations=gate_durations,
        idle_gates=idle_gates,
    )
    prep_inst, se_inst, merge_book = _merge_instruction_parts(
        meta, num_merge_rounds, f"{base_name} merge"
    )
    split_circ_inst, split_book = _split_instruction_parts(
        meta, mode, f"{base_name} split"
    )
    return (
        [prep_inst]
        + [se_inst] * num_merge_rounds
        + [merge_book, split_circ_inst, split_book]
    )


def get_surgery_stage_circuits(
    kind: str,
    patch_a_label: str,
    patch_b_label: str,
    qubits_a: Sequence,
    qubits_b: Sequence,
    seam_qubits: Sequence,
    layout: str,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    idle_layout: str | None = None,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
) -> dict:
    """The three physical circuits of one merge/split surgery, for drawing.

    Returns `{"seam_prep": ..., "merged_se": ..., "seam_measure": ...}`:
    the seam preparation circuit (|+> for ZZ, |0> for XX), ONE round of the
    merged syndrome-extraction circuit (repeated `num_merge_rounds` times at
    runtime), and the destructive seam measurement circuit (X basis for ZZ,
    Z basis for XX). These are the same circuit objects the merge/split
    instructions execute; the classical bookkeeping steps (history
    extension, decode, byproduct frames) have no circuit representation.

    `idle_layout`, `gate_durations`, `idle_gates`: see
    [](api:_surgery_metadata).
    """
    meta = _surgery_metadata(
        kind,
        patch_a_label,
        patch_b_label,
        qubits_a,
        qubits_b,
        seam_qubits,
        layout,
        circuit_backend,
        idle_layout=idle_layout,
        gate_durations=gate_durations,
        idle_gates=idle_gates,
    )
    return {
        "seam_prep": meta["prep_circuit"],
        "merged_se": meta["se_circuit"],
        "seam_measure": meta["split_circuit"],
    }


# ---------------------------------------------------------------------------
# Surgery CNOT (Horsman et al. 2012: M_ZZ then M_XX through a |+> ancilla)
# ---------------------------------------------------------------------------


def _cnot_corrections_apply_fn(
    patches: PatchLayout,
    zz_parity_history,
    xx_parity_history,
    anc_outcome,
    ctrl_patch_label: str,
    tgt_patch_label: str,
) -> Frame:
    """Conditional Pauli-frame corrections closing the surgery CNOT.

    `Z_L(ctrl)^m_xx` and `X_L(tgt)^(m_zz ^ m_anc)`, with m_zz / m_xx the
    most recent surgery parities in the history and m_anc the ancilla
    patch's destructive logical Z outcome (the immediately preceding
    instruction). Support qubits are read from [](api:QECCodePatch.data_qubits)
    at apply time rather than being supplied at build time.
    """
    m_zz = [m for m in zz_parity_history if m is not None][-1]
    m_xx = [m for m in xx_parity_history if m is not None][-1]
    m_anc = anc_outcome

    z_on_ctrl = m_xx
    x_on_tgt = m_zz ^ m_anc

    new_patches = patches.copy()
    if z_on_ctrl:
        patch = new_patches[ctrl_patch_label]
        z_support_ctrl = [patch.data_qubits[i] for i in (0, 4, 8)]
        new_patches[ctrl_patch_label] = patch.copy(
            pauli_frame=_multiply_pauli_into_frame(
                patch.pauli_frame, z_support_ctrl, "Z"
            )
        )
    if x_on_tgt:
        patch = new_patches[tgt_patch_label]
        x_support_tgt = [patch.data_qubits[i] for i in (2, 4, 6)]
        new_patches[tgt_patch_label] = patch.copy(
            pauli_frame=_multiply_pauli_into_frame(
                patch.pauli_frame, x_support_tgt, "X"
            )
        )
    return Frame(
        {
            "patches": new_patches,
            f"surgery_cnot_corrections_{ctrl_patch_label}_{tgt_patch_label}": (
                z_on_ctrl,
                x_on_tgt,
            ),
        }
    )


def build_surgery_cnot_corrections_instruction(
    ctrl_patch_label: str,
    tgt_patch_label: str,
    anc_patch_label: str,
    name: str = "Surgery CNOT corrections",
) -> Instruction:
    """The conditional-correction step of the surgery CNOT.

    Must be placed immediately after the ancilla patch's destructive
    `FT Logical Z Measure` (its `logical_measurement` value is read from
    `history[-1]`); the `surgery_parity_zz_{ctrl_patch_label}_{anc_patch_label}`
    / `surgery_parity_xx_{anc_patch_label}_{tgt_patch_label}` values are
    taken as the most recent occurrences anywhere in the history (matching
    the M_ZZ(ctrl, anc) / M_XX(anc, tgt) pairing in
    [](api:build_surgery_cnot_sequence)). Applies `Z_L(ctrl)^m_xx` (frame Z
    on the Z_L support D0, D4, D8) and `X_L(tgt)^(m_zz ^ m_anc)` (frame X
    on the X_L support D2, D4, D6), and records the applied pair under
    `surgery_cnot_corrections_{ctrl_patch_label}_{tgt_patch_label}`.
    """
    return Instruction(
        _cnot_corrections_apply_fn,
        data={
            "ctrl_patch_label": ctrl_patch_label,
            "tgt_patch_label": tgt_patch_label,
        },
        param_priorities={
            "zz_parity_history": ["history[all]"],
            "xx_parity_history": ["history[all]"],
            "anc_outcome": ["history[-1]"],
        },
        param_aliases={
            "zz_parity_history": f"surgery_parity_zz_{ctrl_patch_label}_{anc_patch_label}",
            "xx_parity_history": f"surgery_parity_xx_{anc_patch_label}_{tgt_patch_label}",
            "anc_outcome": "logical_measurement",
        },
        name=name,
    )


def build_surgery_cnot_sequence(
    ctrl_patch_label: str,
    tgt_patch_label: str,
    anc_patch_label: str,
    qubits_ctrl: Sequence,
    qubits_tgt: Sequence,
    qubits_anc: Sequence,
    seam_qubits_zz: Sequence,
    seam_qubits_xx: Sequence,
    layout: str,
    mode: str = "ft",
    num_merge_rounds: int = 3,
    num_post_split_rounds: int = 3,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    idle_layout: str | None = None,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str | None = None,
) -> list:
    """Stack-entry sequence for a logical CNOT via lattice surgery.

    Implements Horsman et al. 2012: the ancilla patch is prepared in |+>,
    `M_ZZ(ctrl, anc)` runs through a vertical seam (ctrl on top),
    `M_XX(anc, tgt)` through a horizontal seam (anc on the left), the
    ancilla is destructively measured in the Z basis, and the outcome-
    conditioned logical Paulis `Z_L(ctrl)^m_xx`, `X_L(tgt)^(m_zz^m_anc)`
    are injected into the Pauli frames.

    In ft mode each merge is followed by its split-byproduct repair (see
    [](api:build_split_byproduct_repair_instruction)) after the post-split
    QEC blocks of its two patches: the ZZ repair before the XX merge (the
    ancilla's conditional Z_L frame must be settled before its X_L is
    consumed), and the XX repair - which needs post-split syndrome rounds
    on the TARGET as well - after an extra `("QEC", tgt)` entry inserted
    for that purpose. `num_post_split_rounds` must equal the number of
    syndrome rounds one `QEC` stack entry appends (the program's
    `num_qec_rounds`).

    Returns a list of stack entries (mix of patch-scoped names and bare
    instructions) to splice into a program stack with `*`. The three
    patches must already be initialized (`Init Patch`), ctrl/tgt prepped
    and QEC'd by the caller; the ancilla is prepped here. The ancilla's
    destructive measure uses `reference_round_mode_Z="guarded_diff"`
    because the XX split rewrites its Z-check history (grown-check basis
    change).

    Post-CNOT measurement flags for the caller: the ZZ merge grows an X
    check on ctrl (use `reference_round_mode_X="guarded_diff"` for an FT
    X measure of ctrl) and the XX merge grows a Z check on tgt (use
    `reference_round_mode_Z="guarded_diff"` for an FT Z measure of tgt).
    Independently of the surgery, the usual prep-basis rule still
    applies: a patch prepped in the conjugate basis of the measurement
    has a random round-0 syndrome layer, so e.g. an FT Z measure of a
    |+>-prepped ctrl also needs `reference_round_mode_Z="guarded_diff"`
    (without it the decoder matches the round-0 layer as real defects and
    applies a random logical correction, which destroys e.g. Bell
    correlations).

    Two FT holes formerly open here are now fixed: an ancilla data fault
    during the M_ZZ merged-SE rounds could miscorrect `X_L(anc)`'s
    contribution to `m_xx` (fixed in [](api:_split_bookkeeping_apply_fn)),
    and the ZZ repair's `"b_only"` fire rule could false-fire on an
    ordinary bulk fault (fixed via `defect_decode_mode="matching"`, see
    [](api:_split_byproduct_repair_apply_fn)).
    """
    base_name = name or f"Lattice-Surgery CNOT ({mode})"
    zz = build_surgery_parity_instruction(
        "ZZ",
        ctrl_patch_label,
        anc_patch_label,
        qubits_ctrl,
        qubits_anc,
        seam_qubits_zz,
        layout,
        mode=mode,
        num_merge_rounds=num_merge_rounds,
        circuit_backend=circuit_backend,
        idle_layout=idle_layout,
        gate_durations=gate_durations,
        idle_gates=idle_gates,
        name=f"{base_name} M_ZZ",
    )
    xx = build_surgery_parity_instruction(
        "XX",
        anc_patch_label,
        tgt_patch_label,
        qubits_anc,
        qubits_tgt,
        seam_qubits_xx,
        layout,
        mode=mode,
        num_merge_rounds=num_merge_rounds,
        circuit_backend=circuit_backend,
        idle_layout=idle_layout,
        gate_durations=gate_durations,
        idle_gates=idle_gates,
        name=f"{base_name} M_XX",
    )
    corrections = build_surgery_cnot_corrections_instruction(
        ctrl_patch_label,
        tgt_patch_label,
        anc_patch_label,
        name=f"{base_name} corrections",
    )
    entries = [
        ("Plus Prep", anc_patch_label),
        ("QEC", anc_patch_label),
        (zz, None),
        ("QEC", ctrl_patch_label),
        ("QEC", anc_patch_label),
    ]
    if mode == "ft":
        repair_zz = build_split_byproduct_repair_instruction(
            ctrl_patch_label,
            anc_patch_label,
            num_merge_rounds=num_merge_rounds,
            num_post_split_rounds=num_post_split_rounds,
            fire_rule="b_only",
            defect_decode_mode="matching",
            name=f"{base_name} M_ZZ byproduct repair",
        )
        entries.append((repair_zz, None))
    entries += [
        ("QEC", tgt_patch_label),
        (xx, None),
        ("QEC", anc_patch_label),
    ]
    if mode == "ft":
        repair_xx = build_split_byproduct_repair_instruction(
            anc_patch_label,
            tgt_patch_label,
            num_merge_rounds=num_merge_rounds,
            num_post_split_rounds=num_post_split_rounds,
            name=f"{base_name} M_XX byproduct repair",
        )
        entries += [
            ("QEC", tgt_patch_label),
            (repair_xx, None),
        ]
    entries += [
        {
            "instruction": "FT Logical Z Measure",
            "patch_label": anc_patch_label,
            "reference_round_mode_Z": "guarded_diff",
        },
        (corrections, None),
    ]
    return entries


# ---------------------------------------------------------------------------
# M_ZZ Bell preparation (|+>|+> -> M_ZZ -> conditional X_L; no ancilla patch)
# ---------------------------------------------------------------------------


def _mzz_bell_corrections_apply_fn(
    patches: PatchLayout,
    zz_parity_history,
    patch_a_label: str,
    tgt_patch_label: str,
) -> Frame:
    """Conditional Pauli-frame correction closing the M_ZZ Bell prep.

    `X_L(tgt)^m_zz`, with m_zz the most recent surgery ZZ parity in the
    history: projecting |+>_L |+>_L onto the m_zz eigenspace of Z_L Z_L
    leaves (|00> + |11>)/sqrt(2) for m_zz = 0 and (|01> + |10>)/sqrt(2)
    for m_zz = 1, so a frame X_L on either patch restores the Bell state.
    The support qubits are read from [](api:QECCodePatch.data_qubits) at
    apply time rather than being supplied at build time.
    """
    m_zz = [m for m in zz_parity_history if m is not None][-1]

    new_patches = patches.copy()
    if m_zz:
        patch = new_patches[tgt_patch_label]
        x_support_tgt = [patch.data_qubits[i] for i in (2, 4, 6)]
        new_patches[tgt_patch_label] = patch.copy(
            pauli_frame=_multiply_pauli_into_frame(
                patch.pauli_frame, x_support_tgt, "X"
            )
        )
    return Frame(
        {
            "patches": new_patches,
            f"mzz_bell_correction_{patch_a_label}_{tgt_patch_label}": m_zz,
        }
    )


def build_split_byproduct_repair_instruction(
    patch_a_label: str,
    patch_b_label: str,
    num_merge_rounds: int = 3,
    num_post_split_rounds: int = 3,
    fire_rule: str = "both",
    defect_decode_mode: Literal["heuristic", "matching"] = "heuristic",
    name: str | None = None,
) -> Instruction:
    """Post-split repair of the surgery byproduct (seam-fault FT hole).

    Place AFTER at least one QEC block on both patches following the
    merge/split of the same geometry (the QEC block must add
    `num_post_split_rounds` syndrome rounds per patch). Detects the
    grown-check defect signature of a single effective seam-conjugate
    error on the seam qubit whose byproduct flip is NOT compensated by
    the downstream termination decoders, and applies one extra
    byproduct logical Pauli frame, restoring single-fault tolerance of
    the logical correlation in the byproduct-sensitive basis. Records
    the applied bit under the `repair_key` registered by the matching
    [](api:build_merge_instruction) call's [](api:PatchRelation) (see
    below) -- no `kind`/`qubits_a`/`qubits_b`/`seam_qubits`/`layout` here:
    this instruction has no physical circuit of its own, and reads its
    bookkeeping (`grown_rows`/`byproduct`/`repair_key`) from that relation
    instead of independently recomputing `_surgery_metadata`.

    `fire_rule` selects the uncompensated signature for the context:
    "both" when both patches get a byproduct-sensitive termination
    decode (mzz Bell prep; the surgery CNOT's XX merge - the ancilla's
    Z decode feeds m_anc into the X_L(tgt) correction), "b_only" when
    only patch A does (the surgery CNOT's ZZ merge, whose patch B is
    the ancilla, terminated in the conjugate basis). `defect_decode_mode`
    selects the fire-decision mechanism ("heuristic" default, "matching"
    a real per-row decode fixing a "b_only" false-fire -- see
    [](api:_split_byproduct_repair_apply_fn)).

    Without this step a single fault on the middle seam qubit (or a
    hook error reaching it) flips the decoded logical XOR: the raw
    byproduct record is wrong AND both patches' termination decoders
    independently boundary-match the resulting grown-check defects -
    three net flips (see [](api:_split_byproduct_repair_apply_fn)).
    """
    return Instruction(
        _split_byproduct_repair_apply_fn,
        data={
            "patch_a_label": patch_a_label,
            "patch_b_label": patch_b_label,
            "num_merge_rounds": num_merge_rounds,
            "num_post_split_rounds": num_post_split_rounds,
            "fire_rule": fire_rule,
            "defect_decode_mode": defect_decode_mode,
        },
        name=name or "Lattice-Surgery byproduct repair",
    )


def build_mzz_bell_corrections_instruction(
    patch_a_label: str,
    tgt_patch_label: str,
    name: str = "M_ZZ Bell prep corrections",
) -> Instruction:
    """The conditional-correction step of the M_ZZ Bell preparation.

    Applies `X_L(tgt)^m_zz` (frame X on the X_L support D2, D4, D6), with
    m_zz the most recent `surgery_parity_zz_{patch_a_label}_{tgt_patch_label}`
    value anywhere in the history, and records the applied bit under
    `mzz_bell_correction_{patch_a_label}_{tgt_patch_label}`.
    """
    return Instruction(
        _mzz_bell_corrections_apply_fn,
        data={
            "patch_a_label": patch_a_label,
            "tgt_patch_label": tgt_patch_label,
        },
        param_priorities={
            "zz_parity_history": ["history[all]"],
        },
        param_aliases={
            "zz_parity_history": f"surgery_parity_zz_{patch_a_label}_{tgt_patch_label}",
        },
        name=name,
    )


def build_mzz_bell_prep_sequence(
    patch_a_label: str,
    patch_b_label: str,
    qubits_a: Sequence,
    qubits_b: Sequence,
    seam_qubits: Sequence,
    layout: str,
    mode: str = "ft",
    num_merge_rounds: int = 3,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    idle_layout: str | None = None,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str | None = None,
) -> list:
    """Stack-entry sequence for Bell prep by a direct M_ZZ merge.

    Measures `Z_L(A) Z_L(B)` on the |+>_L (x) |+>_L product state with a
    single merge/split through one 3-qubit seam (no ancilla patch), then
    injects the outcome-conditioned frame correction `X_L(B)^m_zz`,
    leaving the Bell state (|00> + |11>)/sqrt(2). The caller preps both
    patches in |+>_L and runs QEC before and after the returned entries.

    Measurement flags for the caller (cf. build_surgery_cnot_sequence):
    both patches are |+>-prepped so their round-0 Z syndrome layer is
    random (use `reference_round_mode_Z="guarded_diff"` for FT Z
    measures), but the round-0 X layer is DETERMINISTIC — use
    `reference_round_mode_X="raw"` (the default) for FT X measures so
    faults in the prep window stay detectable; the grown-X-check history
    rewrite at split is absorbed by the exported
    round-0 offset key, not by dropping round 0.

    `idle_layout`, `gate_durations`, `idle_gates`: see
    [](api:_surgery_metadata).
    """
    base_name = name or "M_ZZ Bell prep"
    zz = build_surgery_parity_instruction(
        "ZZ",
        patch_a_label,
        patch_b_label,
        qubits_a,
        qubits_b,
        seam_qubits,
        layout,
        mode=mode,
        num_merge_rounds=num_merge_rounds,
        circuit_backend=circuit_backend,
        idle_layout=idle_layout,
        gate_durations=gate_durations,
        idle_gates=idle_gates,
        name=f"{base_name} M_ZZ",
    )
    corrections = build_mzz_bell_corrections_instruction(
        patch_a_label,
        patch_b_label,
        name=f"{base_name} corrections",
    )
    return [
        (zz, None),
        (corrections, None),
    ]
