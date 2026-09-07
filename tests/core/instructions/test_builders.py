"""Tester for loqs.core.instructions.builders"""

import pytest

from loqs.backends import NumpyStatevectorQuantumState as SVState
from loqs.core import QECCode, QuantumProgram
from loqs.core.frame import Frame
from loqs.core.instructions import builders
from loqs.core.instructions.instruction import Instruction
from loqs.core.instructions.instructionstack import InstructionStack
from loqs.core.recordables import PatchLayout, PatchRelation


def _leaf_apply_needing_model(model, patch_label=None):
    """A minimal apply_fn that just records whatever `model` it received."""
    return Frame({"seen_model": model})


class _Widget:
    """Dummy class with required, defaulted, and None-defaulted params."""

    def __init__(self, size, mode="round", finish=None):
        self.size = size
        self.mode = mode
        self.finish = finish


class _Broken:
    """Dummy class with a required param no source can provide."""

    def __init__(self, size, mystery):
        self.size = size
        self.mystery = mystery


class TestObjectBuilderInstruction:

    def _build_program(self, obj_class, label):
        inst = builders.build_object_builder_instruction(
            "thing", obj_class, name="Init Thing"
        )
        return QuantumProgram(
            [label],
            global_instructions={"Init Thing": inst},
            name="object builder test",
        )

    def test_uncollectable_defaulted_params_use_defaults(self):
        # Regression test: constructor params that cannot be collected from
        # any source (label/instruction/patch_data/program/history) must
        # fall back to their constructor defaults. This used to crash with
        # an IndexError from the history[-1] lookup on the empty initial
        # history (or a RuntimeError once history was non-empty)
        program = self._build_program(_Widget, {"instruction": "Init Thing", "size": 3})
        results = program.run()
        widget = results.shot_histories[0][-1]["thing"]
        assert widget.size == 3
        assert widget.mode == "round"
        assert widget.finish is None

    def test_label_kwargs_override_defaults(self):
        program = self._build_program(
            _Widget, {"instruction": "Init Thing", "size": 3, "mode": "square"}
        )
        results = program.run()
        widget = results.shot_histories[0][-1]["thing"]
        assert widget.size == 3
        assert widget.mode == "square"

    def test_uncollectable_required_param_raises(self):
        # A required constructor param that no source provides must still
        # fail loudly (via the object builder's construction error), not
        # silently produce a broken object
        program = self._build_program(_Broken, {"instruction": "Init Thing", "size": 3})
        with pytest.raises(ValueError, match="Failed to create object"):
            program.run()

    def test_init_state_collects_svstate_options(self):
        # SVState's kraus_sampling/contraction are ordinary
        # positional-or-keyword params with defaults: absent from the label
        # they must default, present in the label kwargs they must apply
        stack = [{"instruction": "Init State", "state": 1, "qubit_labels": ["Q0"]}]
        program = QuantumProgram(
            stack, state_type=SVState, name="init state defaults"
        )
        state = program.run().shot_histories[0][-1]["state"]
        assert state.kraus_sampling == "lazy"
        assert state.contraction == "matmul"

        stack = [
            {
                "instruction": "Init State",
                "state": 1,
                "qubit_labels": ["Q0"],
                "kraus_sampling": "choice",
                "contraction": "einsum",
            }
        ]
        program = QuantumProgram(
            stack, state_type=SVState, name="init state overrides"
        )
        state = program.run().shot_histories[0][-1]["state"]
        assert state.kraus_sampling == "choice"
        assert state.contraction == "einsum"


class TestCompositeInstruction:
    """Regression tests for issue #57: a composite instruction must
    forward a per-call kwarg override (e.g. "model") to nested
    instructions that need it, not silently drop it in favor of
    whatever the program/instruction's own default resolution provides.
    """

    def _leaf(self):
        return Instruction(apply_fn=_leaf_apply_needing_model, data={}, name="leaf")

    def test_param_priorities_include_nested_instructions_keys(self):
        composite = builders.build_composite_instruction(
            [self._leaf()], name="composite"
        )
        assert "model" in composite.param_priorities

    def test_label_kwarg_reaches_nested_instruction(self):
        composite = builders.build_composite_instruction(
            [self._leaf()], name="composite"
        )
        stack = InstructionStack([])
        frame = composite.apply(
            stack=stack,
            instructions=composite.data["instructions"],
            patch_label="L0",
            model="OVERRIDE",
        )
        nested_label = frame["stack"][0]
        assert nested_label["patch_label"] == "L0"
        assert nested_label["model"] == "OVERRIDE"

    def test_end_to_end_label_override_wins_over_program_default(self):
        # A plain object stands in for a real noise model here: only its
        # identity matters for this test, and passing a str would instead
        # be (mis)interpreted by QuantumProgram as a file to read from.
        default_model, override_model = object(), object()
        composite = builders.build_composite_instruction(
            [self._leaf()], name="H"
        )
        stack = [{"instruction": "H", "model": override_model}]
        program = QuantumProgram(
            stack,
            global_instructions={"H": composite},
            default_noise_model=default_model,
            name="composite override test",
        )
        history = program.run().shot_histories[0]
        assert history[-1]["seen_model"] is override_model

    def test_end_to_end_falls_back_to_program_default_without_override(self):
        # Regression check for param_error_behavior="continue": pulling
        # "model" up onto the composite must not break the no-override
        # case, which should still fall through to the program default.
        default_model = object()
        composite = builders.build_composite_instruction(
            [self._leaf()], name="H"
        )
        stack = ["H"]
        program = QuantumProgram(
            stack,
            global_instructions={"H": composite},
            default_noise_model=default_model,
            name="composite default test",
        )
        history = program.run().shot_histories[0]
        assert history[-1]["seen_model"] is default_model


class TestPatchBuilderAndRemoverInstructions:

    def _code(self):
        return QECCode(instructions={}, template_qubits=["q0"], template_data_qubits=["q0"])

    def test_builder_creates_first_patch_from_none(self):
        inst = builders.build_patch_builder_instruction(self._code())
        f = inst.apply(new_patch_label="L0", qubits=["Q0"], qec_code=self._code(), patches=None)
        patches = f["patches"]
        assert isinstance(patches, PatchLayout)
        assert patches["L0"].qubits == ["Q0"]

    def test_builder_rejects_overlapping_qubits(self):
        code = self._code()
        inst = builders.build_patch_builder_instruction(code)
        patches = PatchLayout({"L0": code.create_patch(["Q0"])})
        with pytest.raises(AssertionError):
            inst.apply(new_patch_label="L1", qubits=["Q0"], qec_code=code, patches=patches)

    def test_builder_rejects_duplicate_label(self):
        code = self._code()
        inst = builders.build_patch_builder_instruction(code)
        patches = PatchLayout({"L0": code.create_patch(["Q0"])})
        with pytest.raises(AssertionError):
            inst.apply(new_patch_label="L0", qubits=["Q1"], qec_code=code, patches=patches)

    def test_remover_removes_patch(self):
        code = self._code()
        inst = builders.build_patch_remover_instruction()
        patches = PatchLayout({"L0": code.create_patch(["Q0"])})
        f = inst.apply(del_patch_label="L0", patches=patches)
        assert "L0" not in f["patches"]

    def test_remover_rejects_missing_label(self):
        inst = builders.build_patch_remover_instruction()
        with pytest.raises(AssertionError):
            inst.apply(del_patch_label="L0", patches=PatchLayout())

    def test_remover_auto_drops_referencing_relations(self):
        code = self._code()
        patches = PatchLayout(
            {"L0": code.create_patch(["Q0"]), "L1": code.create_patch(["Q1"])}
        )
        patches.set_relation(PatchRelation({"a": "L0", "b": "L1"}))
        assert patches.get_relation("L0", "L1") is not None

        inst = builders.build_patch_remover_instruction()
        f = inst.apply(del_patch_label="L0", patches=patches)
        assert f["patches"].relations == {}


class TestPhysicalCircuitInstructionSTIM:
    """Regression test for a `not is_backend_available("stim_state")`
    inversion bug in `build_physical_circuit_instruction.<locals>.apply_fn`
    that silently prevented `applied_stim_circuit_str` from ever being
    recorded on the output `Frame` for a real `STIMQuantumState`.
    """

    def test_applied_stim_circuit_str_is_populated(self):
        pytest.importorskip("stim")
        from loqs.backends import STIMQuantumState, DictNoiseModel
        from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit
        from loqs.backends.reps import StimCircuitGateRep

        circuit = STIMPhysicalCircuit("H 0\nTICK", ["Q0"])
        model = DictNoiseModel({"H": "H 0"}, {}, gatereps=[StimCircuitGateRep])
        state = STIMQuantumState(1, ["Q0"])

        inst = builders.build_physical_circuit_instruction(
            circuit=circuit, name="PhysCirc STIM"
        )
        frame = inst.apply(
            model=model,
            circuit=circuit,
            state=state,
            inplace=True,
            error_injections=None,
            pauli_frame_update=None,
            patch_label="L0",
            patches=PatchLayout(),
        )

        applied_str = frame["applied_stim_circuit_str"]
        assert applied_str is not None
        assert applied_str == str(frame["state"].latest_applied_circuit)
        assert "H 0" in applied_str

    def test_error_injection_on_non_highest_qubit_does_not_raise(self):
        """Regression test for a bug where injecting an error onto any
        qubit other than the circuit's highest-indexed one raised
        `ValueError`, since STIM infers `num_qubits` from the highest
        qubit index explicitly referenced in the injected error's own
        circuit string, which then mismatched `qubit_labels`.
        """
        pytest.importorskip("stim")
        from loqs.backends import STIMQuantumState, DictNoiseModel
        from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit
        from loqs.backends.reps import (
            StimCircuitGateRep,
            ZBasisProjectionInstrumentRep,
        )

        qubits = ["Q0", "Q1", "Q2"]
        circuit = STIMPhysicalCircuit("M 0 1 2\nTICK", qubits)
        inst_dict = {"M": ZBasisProjectionInstrumentRep(None, True, ("Q0",))}
        model = DictNoiseModel(
            {"X": "X 0"},
            inst_dict,
            gatereps=[StimCircuitGateRep],
            instreps=[ZBasisProjectionInstrumentRep],
        )
        state = STIMQuantumState(3, qubits)

        inst = builders.build_physical_circuit_instruction(
            circuit=circuit, name="PhysCirc STIM"
        )
        # Inject an "X" error on qubit index 0, which is not the highest
        # qubit index (2) referenced elsewhere in the circuit.
        frame = inst.apply(
            model=model,
            circuit=circuit,
            state=state,
            inplace=True,
            error_injections=[(0, "X", 0)],
            pauli_frame_update=None,
            patch_label="L0",
            patches=PatchLayout(),
        )

        outcomes = frame["measurement_outcomes"]
        assert outcomes["Q0"] == [1]
        assert outcomes["Q1"] == [0]
        assert outcomes["Q2"] == [0]
