"""Tester for the [[7,1,3]] codepack's transversal logical CX."""

import pytest

pygsti = pytest.importorskip("pygsti")
stim = pytest.importorskip("stim")

from loqs.backends import (
    DictNoiseModel,
    ListPhysicalCircuit,
    PyGSTiPhysicalCircuit,
    STIMQuantumState,
    StimCircuitGateRep,
)
from loqs.core import InstructionStack, PatchGeometry, QuantumProgram
from loqs.core.recordables.patchlayout import PatchLayout
from loqs.core.instructions import Instruction
from loqs.core.instructions import builders
from loqs.codepacks import SteaneCodepack
from loqs.codepacks import codepack_7_1_3_multipatch as multipatch

NUM_SHOTS = 10


def steane_cx(geometry):
    """The class's transversal CX on `geometry`."""
    return SteaneCodepack().build_two_patch("Transversal CX", geometry)


def steane_qubits(suffix: str = "") -> list[str]:
    """A [[7,1,3]] patch's (auxiliary + data) qubit labels, with a suffix."""
    base = ["A0", "A1", "A2"] + [f"D{i}" for i in range(7)]
    return [f"{q}{suffix}" for q in base]


def two_patch_cx_stack(prep_ctrl, prep_tgt, meas):
    """Stack: preps, transversal CX L0->L1, FT measures (no QEC rounds)."""
    q0 = steane_qubits("_0")
    q1 = steane_qubits("_1")
    all_q = q0 + q1
    geometry = PatchGeometry(
        patches={"ctrl": ("L0", q0), "tgt": ("L1", q1)}, layout="7_1_3"
    )
    cx = steane_cx(geometry)
    stack = [
        {
            "instruction": "Init State",
            "state": len(all_q),
            "qubit_labels": all_q,
        },
        *geometry.init_patch_entries("Steane"),
        (prep_ctrl, "L0"),
        (prep_tgt, "L1"),
        (cx, None),
        (meas, "L0"),
        (meas, "L1"),
    ]
    return stack, all_q


def make_program(stack, all_qubits, code=None):
    """Build a STIM-backed QuantumProgram over the given stack.

    Parameters
    ----------
    code:
        A pre-built QECCode to use in place of a fresh
        `SteaneCodepack().create_qec_code()`, e.g. one with an
        `"Injected Data Error"` instruction added for fault-injection
        tests.
    """
    if code is None:
        code = SteaneCodepack().create_qec_code()
    model = SteaneCodepack().create_ideal_model(
        all_qubits,
        gaterep=StimCircuitGateRep,
        model_backend=DictNoiseModel,
    )
    return QuantumProgram(
        InstructionStack(stack),
        default_noise_model=model,
        state_type=STIMQuantumState,
        patch_types={"Steane": code},
        name="Transversal CX",
    )


def code_with_injected_error(error_gate: str, data_qubit: str):
    """A Steane QECCode with an `"Injected Data Error"` instruction.

    The instruction applies a single-qubit `error_gate` (e.g. `"Gxpi"`) to
    `data_qubit` (a template label, e.g. `"D6"`), to be invoked as
    `("Injected Data Error", <patch_label>)` in a program stack.
    """
    code = SteaneCodepack().create_qec_code()
    error_circ = PyGSTiPhysicalCircuit(
        [[(error_gate, data_qubit)]], qubit_labels=steane_qubits()
    )
    code.instructions["Injected Data Error"] = (
        builders.build_physical_circuit_instruction(
            error_circ, name="Injected Data Error"
        )
    )
    return code


def run_cx_bookkeeping(frame_ctrl, frame_tgt, synd_ctrl, synd_tgt):
    """Run the class's CX bookkeeping on two patches holding the given
    logical frames and syndromes; return both patches' new
    `(logical_pauli_frame, latest_syndrome)`."""
    qubits = steane_qubits
    geometry = PatchGeometry(
        patches={
            "ctrl": ("L0", qubits("_0")),
            "tgt": ("L1", qubits("_1")),
        },
        layout="7_1_3",
    )
    codepack = SteaneCodepack(circuit_backend=ListPhysicalCircuit)
    code = codepack.create_qec_code()
    layout = PatchLayout(
        {
            "L0": code.create_patch(qubits("_0")),
            "L1": code.create_patch(qubits("_1")),
        }
    )
    for label, frame, synd in (
        ("L0", frame_ctrl, synd_ctrl),
        ("L1", frame_tgt, synd_tgt),
    ):
        layout[label].data = {
            "logical_pauli_frame": list(frame),
            "latest_syndrome": list(synd),
        }
    cx = codepack.build_two_patch("Transversal CX", geometry)
    book = cx.data["instructions"][1]
    new = book.apply_fn(patches=layout, **book.data)["patches"]
    return tuple(
        (
            new[label].data["logical_pauli_frame"],
            new[label].data["latest_syndrome"],
        )
        for label in ("L0", "L1")
    )


ZERO_SYND = [0, 0, 0, 0, 0, 0]


class TestConjugateCxLogicalPauliFrames:
    """The CX bookkeeping's logical-frame truth table (frame = [Z_bit, X_bit])."""

    TRUTH_TABLE = [
        # (frame_ctrl, frame_tgt, expected_ctrl, expected_tgt)
        ([0, 0], [0, 0], [0, 0], [0, 0]),
        ([0, 1], [0, 0], [0, 1], [0, 1]),  # ctrl X copies to tgt
        ([1, 0], [0, 0], [1, 0], [0, 0]),  # ctrl Z untouched
        ([0, 0], [1, 0], [1, 0], [1, 0]),  # tgt Z copies to ctrl
        ([0, 0], [0, 1], [0, 0], [0, 1]),  # tgt X untouched
        ([1, 1], [1, 1], [0, 1], [1, 0]),
    ]

    @pytest.mark.parametrize("fc,ft,exp_c,exp_t", TRUTH_TABLE)
    def test_truth_table(self, fc, ft, exp_c, exp_t):
        (new_c, _), (new_t, _) = run_cx_bookkeeping(
            fc, ft, ZERO_SYND, ZERO_SYND
        )
        assert new_c == exp_c
        assert new_t == exp_t


class TestConjugateCxLatestSyndromes:
    """The CX bookkeeping's syndrome truth table (syndrome = [S1..S6])."""

    TRUTH_TABLE = [
        # (syndrome_ctrl, syndrome_tgt, expected_ctrl, expected_tgt)
        (
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ),
        # ctrl X-type (S1) alone: tgt is all-zero, so nothing changes.
        (
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ),
        # tgt X-type (S1): ctrl's X-type picks it up; tgt's own is untouched.
        (
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ),
        # ctrl Z-type (S4): untouched on ctrl, copies onto tgt's Z-type.
        (
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ),
        # tgt Z-type (S4) alone: ctrl is all-zero, so nothing changes.
        (
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ),
        # Both fully set: X-type slices cancel on ctrl, Z-type cancel on tgt.
        (
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ),
    ]

    @pytest.mark.parametrize("sc,st,exp_c,exp_t", TRUTH_TABLE)
    def test_truth_table(self, sc, st, exp_c, exp_t):
        (_, new_c), (_, new_t) = run_cx_bookkeeping([0, 0], [0, 0], sc, st)
        assert new_c == exp_c
        assert new_t == exp_t


class TestTransversalCx:
    """Truth table and Bell-correlation checks for the transversal logical CX."""

    def test_equal_after_hdf5_round_trip(self, tmp_path):
        """The CX equals its decoded HDF5 file, and not a swapped CX."""
        q0 = steane_qubits("_0")
        q1 = steane_qubits("_1")

        def make_cx(first, second):
            geometry = PatchGeometry(
                patches={"ctrl": ("L0", first), "tgt": ("L1", second)},
                layout="7_1_3",
            )
            return steane_cx(geometry)

        cx = make_cx(q0, q1)
        path = tmp_path / "cx.h5"
        cx.write(path)
        loaded = Instruction.read(path)

        assert cx == loaded
        assert cx != make_cx(q1, q0)

    def test_x_on_control(self):
        """X_L|0>|0> -> CX -> |1>|1>: X_L copies to the target patch."""
        stack, all_q = two_patch_cx_stack(
            "Non-FT Zero Prep", "Non-FT Zero Prep", "FT Logical Z Measure"
        )
        # Insert an X on the control patch right after its own prep.
        stack.insert(4, ("X", "L0"))
        program = make_program(stack, all_q)
        results = program.run(num_shots=NUM_SHOTS, verbose=False)
        per_shot = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(l0 == 1 and l1 == 1 for l0, l1 in per_shot)


class TestSyndromeCouplingAfterCx:
    """Regression coverage for the `conjugate_cx_latest_syndromes` fix.

    Unlike `TestTransversalCx`, these stacks run a QEC round on the
    control patch *before* the CX (to establish a genuinely nonzero
    `latest_syndrome` there) and a QEC round on the target patch *after*
    the CX (to exercise the coupling the fix corrects for).
    """

    def _two_patch_stack_with_qec(self, error_gate, error_qubit):
        q0 = steane_qubits("_0")
        q1 = steane_qubits("_1")
        all_q = q0 + q1
        geometry = PatchGeometry(
            patches={"ctrl": ("L0", q0), "tgt": ("L1", q1)}, layout="7_1_3"
        )
        cx = steane_cx(geometry)
        stack = [
            {
                "instruction": "Init State",
                "state": len(all_q),
                "qubit_labels": all_q,
            },
            *geometry.init_patch_entries("Steane"),
            ("Non-FT Zero Prep", "L0"),
            ("Non-FT Zero Prep", "L1"),
            ("Injected Data Error", "L0"),
            ("Adaptive QEC", "L0"),
            (cx, None),
            ("Adaptive QEC", "L1"),
            ("FT Logical Z Measure", "L0"),
            ("FT Logical Z Measure", "L1"),
        ]
        code = code_with_injected_error(error_gate, error_qubit)
        return stack, all_q, code

    def test_qec_round_after_cx_uses_coupled_syndrome(self):
        """Coupling-induced syndrome shift must not be mistaken for a new error.

        An X-type error on the control's D6 is fully corrected (virtually)
        by a QEC round *before* the CX, leaving a nonzero
        `latest_syndrome[3:]` (Z-type) baseline on the control -- the
        control's data qubits are still physically perturbed, only
        virtually tracked via the Pauli frame. The transversal CX
        physically couples this into the target's own Z-type checks (per
        the standard CSS map). Without `conjugate_cx_latest_syndromes`,
        the target's subsequent QEC round would misread that coupling as
        a brand-new error on the target and wrongly flip its frame; with
        the fix, both patches measure the correct (unperturbed) logical Z
        outcome.
        """
        stack, all_q, code = self._two_patch_stack_with_qec("Gxpi", "D6")
        program = make_program(stack, all_q, code=code)
        results = program.run(num_shots=NUM_SHOTS, verbose=False)
        per_shot = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(l0 == 0 and l1 == 0 for l0, l1 in per_shot)


class TestDeprecatedMultipatchWrappers:
    """The old CX builders warn and build the class's CX under `name`."""

    def _geometry(self):
        return PatchGeometry(
            patches={
                "ctrl": ("L0", steane_qubits("_0")),
                "tgt": ("L1", steane_qubits("_1")),
            },
            layout="7_1_3",
        )

    def test_composite_and_parts_are_renamed(self):
        geometry = self._geometry()
        with pytest.warns(DeprecationWarning):
            old = multipatch.build_transversal_cx_instruction(
                geometry, circuit_backend=ListPhysicalCircuit, name="My CX"
            )
        new = SteaneCodepack(
            circuit_backend=ListPhysicalCircuit
        ).build_two_patch("Transversal CX", geometry)
        assert old.name == "My CX"
        names = [i.name for i in old.data["instructions"]]
        assert names == ["My CX (physical circuit)", "My CX (bookkeeping)"]
        for o, n in zip(old.data["instructions"], new.data["instructions"]):
            assert o.data.keys() == n.data.keys()
        circuit = old.data["instructions"][0].data["circuit"]
        assert type(circuit) is ListPhysicalCircuit

    def test_half_builders(self):
        geometry = self._geometry()
        with pytest.warns(DeprecationWarning):
            circuit_inst = multipatch.build_transversal_cx_circuit_instruction(
                geometry, name="Half"
            )
        with pytest.warns(DeprecationWarning):
            book = multipatch.build_cx_bookkeeping_instruction(
                "La", "Lb", name="Book"
            )
        assert circuit_inst.name == "Half"
        assert book.name == "Book"
        assert book.data["patch_labels"] == ["La", "Lb"]
