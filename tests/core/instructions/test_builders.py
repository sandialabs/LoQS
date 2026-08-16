"""Tester for loqs.core.instructions.builders"""

import pytest

from loqs.backends import NumpyStatevectorQuantumState as SVState
from loqs.core import QuantumProgram
from loqs.core.frame import Frame
from loqs.core.instructions import builders
from loqs.core.instructions.instruction import Instruction
from loqs.core.instructions.instructionstack import InstructionStack


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
