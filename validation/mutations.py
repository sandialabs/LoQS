"""Reintroduced historical Steane bugs, for the harness's self-tests.

Each mutation edits the built code (`code_hook`) or an engine step
(`op_hook`) and names the cases that must fail the targeted checks on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from loqs.backends import ListPhysicalCircuit
from loqs.codepacks import codepack_7_1_3_multipatch as multipatch
from loqs.codepacks import codepack_surf17_multipatch as surf_multipatch
from loqs.core import Instruction
from loqs.core.frame import Frame
from loqs.core.instructions import builders

P = "steane.default.tracking."
T = "steane.default.truth.truth."


@dataclass(frozen=True)
class Mutation:
    """A mutation and, per case id, the checks that must fail on it."""

    cases: dict[str, tuple[str, ...]]
    codepack: str = "steane"
    code_hook: Callable[..., Any] | None = None
    op_hook: Callable[..., Any] | None = None


def _s_bare_circuit(code, config):
    """`S` and `Sdag` run only their physical circuit: no frame update."""
    ins = code.instructions
    ins["S"], ins["Sdag"] = ins["S Circuit"], ins["Sdag Circuit"]


def _h_no_syndrome_swap(code, config):
    """`H`'s frame update swaps the logical frame but not the syndrome."""
    ins = code.instructions

    def apply_fn(patches, patch_label):
        patch = patches[patch_label]
        frame = list(patch.data.get("logical_pauli_frame", [0, 0]))[::-1]
        new_patch = patch.copy()
        new_patch.data["logical_pauli_frame"] = frame
        patches[patch_label] = new_patch
        return Frame({"patches": patches})

    ins["H Frame Update"] = Instruction(apply_fn, name="H frame only")
    ins["H"] = builders.build_composite_instruction(
        [ins["H Circuit"], ins["H Frame Update"]], name="Logical H"
    )


def _x_measure_uses_h(code, config):
    """`FT Logical X Measure` is built with `H`, which also swaps the
    stored syndrome, in place of the bare `H Circuit`."""
    ins = code.instructions
    ins["FT Logical X Measure"] = builders.build_composite_instruction(
        [
            ins["H"],
            ins["Raw Z Data Measure"],
            ins["FT X logical parity calculation"],
        ],
        name="FT logical X measurement",
    )


def _s_sdag_swapped(code, config):
    """The `S` and `Sdag` instructions are exchanged."""
    ins = code.instructions
    ins["S"], ins["Sdag"] = ins["Sdag"], ins["S"]


def _x_wrong_support(code, config):
    """Logical `X` acts on the first three data qubits, not the last."""
    circuit = config.backends[0].circuit_backend(
        [[("Gxpi", q) for q in code.template_qubits[:3]]],
        qubit_labels=code.template_qubits,
    )
    code.instructions["X"] = builders.build_physical_circuit_instruction(
        circuit, pauli_frame_update="X", name="Logical X"
    )


def _cx_frame_only(op, entries, geometry):
    """The transversal CX's bookkeeping skips the syndrome conjugation."""
    if op.key != "transversal-cx":
        return entries
    ctrl, tgt = geometry.label("ctrl"), geometry.label("tgt")

    def apply_fn(patches, ctrl_patch_label, tgt_patch_label):
        pc, pt = patches[ctrl_patch_label], patches[tgt_patch_label]
        fc, ft = multipatch.conjugate_cx_logical_pauli_frames(
            pc.data.get("logical_pauli_frame", [0, 0]),
            pt.data.get("logical_pauli_frame", [0, 0]),
        )
        new_c, new_t = pc.copy(), pt.copy()
        new_c.data["logical_pauli_frame"] = fc
        new_t.data["logical_pauli_frame"] = ft
        new_patches = patches.copy()
        new_patches[ctrl_patch_label] = new_c
        new_patches[tgt_patch_label] = new_t
        return Frame({"patches": new_patches})

    circuit = multipatch.build_transversal_cx_circuit_instruction(
        geometry, circuit_backend=ListPhysicalCircuit
    )
    book = Instruction(
        apply_fn,
        data={"ctrl_patch_label": ctrl, "tgt_patch_label": tgt},
        name="CX frame only",
    )
    return [(circuit, None), (book, None)]


def _surface_cnot_xor_dropped(op, entries, geometry):
    """The transversal CNOT's bookkeeping conjugates the Pauli frames but
    leaves the syndrome histories unXORed."""
    if op.key != "transversal-cnot":
        return entries
    ctrl, tgt = geometry.label("ctrl"), geometry.label("tgt")

    def apply_fn(patches, ctrl_patch_label, tgt_patch_label):
        pc, pt = patches[ctrl_patch_label], patches[tgt_patch_label]
        fc, ft = surf_multipatch.pairwise_cnot_pauli_frames(
            pc.pauli_frame,
            pt.pauli_frame,
            pc.data_qubits,
            pt.data_qubits,
        )
        new_patches = patches.copy()
        new_patches[ctrl_patch_label] = pc.copy(pauli_frame=fc)
        new_patches[tgt_patch_label] = pt.copy(pauli_frame=ft)
        return Frame({"patches": new_patches})

    circuit = surf_multipatch.build_transversal_cnot_circuit_instruction(
        geometry.qubits("ctrl")[:9],
        geometry.qubits("tgt")[:9],
        circuit_backend=ListPhysicalCircuit,
    )
    book = Instruction(
        apply_fn,
        data={"ctrl_patch_label": ctrl, "tgt_patch_label": tgt},
        name="CNOT frame only",
    )
    return [(circuit, None), (book, None)]


_WORD = P + "word.FT_Zero_Prep-H-W{}-Adaptive_QEC-FT_Logical_Z_Measure"
_FT = "steane.default.ft.ft.check-S1-S5-S6-FT_Zero_Prep+H-"
S = "surface.surf10.tracking.mp-track.transversal-cnot-"
_MP = P + "mp-track.transversal-cx-FT_Zero_Prep-FT_Zero_Prep-Adaptive_QEC-"

MUTATIONS: dict[str, Mutation] = {
    "s-bare-circuit": Mutation(
        {
            _WORD.format("S"): ("outcome", "O3"),
            _WORD.format("Sdag"): ("outcome", "O3"),
            _FT + "FT_Logical_Z_Measure-WS-clean": ("outcome",),
        },
        code_hook=_s_bare_circuit,
    ),
    "h-no-syndrome-swap": Mutation(
        {
            P + "gate.FT_Zero_Prep-none-H-eqgq-Adaptive_QEC-"
            "FT_Logical_X_Measure": ("O2",),
            P + "gate.FT_Zero_Prep-H-H-eqgq-Adaptive_QEC-"
            "FT_Logical_Z_Measure": ("O2",),
        },
        code_hook=_h_no_syndrome_swap,
    ),
    "cx-frame-only": Mutation(
        {
            _MP + "pre": ("O2", "O3"),
            P + "mp-track.transversal-cx-FT_Zero_Prep+H-FT_Zero_Prep+H-"
            "Adaptive_QEC-twice": ("O2",),
        },
        op_hook=_cx_frame_only,
    ),
    "x-measure-uses-h": Mutation(
        {
            P
            + "word.FT_Zero_Prep-none-WS-Adaptive_QEC-FT_Logical_X_Measure": (
                "outcome",
            ),
            P + "word.Non-FT_Zero_Prep-none-WS-Adaptive_QEC-"
            "FT_Logical_X_Measure": ("outcome",),
        },
        code_hook=_x_measure_uses_h,
    ),
    "s-sdag-swapped": Mutation(
        {
            T + "FT_Zero_Prep-H+S-FT_Logical_Z_Measure-noqec": ("O3",),
            T + "FT_Zero_Prep-H+Sdag-FT_Logical_Z_Measure-noqec": ("O3",),
        },
        code_hook=_s_sdag_swapped,
    ),
    "x-wrong-support": Mutation(
        {
            T + "FT_Zero_Prep-X-FT_Logical_Z_Measure-noqec": ("outcome",),
            T + "Non-FT_Zero_Prep-X-FT_Logical_Z_Measure-noqec": ("outcome",),
        },
        code_hook=_x_wrong_support,
    ),
    "surface-cnot-xor-dropped": Mutation(
        {
            S + "Zero_Prep-Zero_Prep-QEC-pre": ("O2",),
            S + "Plus_Prep-Plus_Prep-QEC-pre": ("O2",),
        },
        codepack="surface",
        op_hook=_surface_cnot_xor_dropped,
    ),
}
