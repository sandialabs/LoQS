"""Tester for loqs.core.instructions.builders"""

import pytest

from loqs.backends import NumpyStatevectorQuantumState as SVState
from loqs.core import QuantumProgram
from loqs.core.instructions import builders


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
        program = self._build_program(_Widget, ("Init Thing", None, (3,)))
        results = program.run()
        widget = results.shot_histories[0][-1]["thing"]
        assert widget.size == 3
        assert widget.mode == "round"
        assert widget.finish is None

    def test_label_kwargs_override_defaults(self):
        program = self._build_program(
            _Widget, ("Init Thing", None, (3,), {"mode": "square"})
        )
        results = program.run()
        widget = results.shot_histories[0][-1]["thing"]
        assert widget.size == 3
        assert widget.mode == "square"

    def test_uncollectable_required_param_raises(self):
        # A required constructor param that no source provides must still
        # fail loudly (via the object builder's construction error), not
        # silently produce a broken object
        program = self._build_program(_Broken, ("Init Thing", None, (3,)))
        with pytest.raises(ValueError, match="Failed to create object"):
            program.run()

    def test_init_state_collects_svstate_options(self):
        # SVState's kraus_sampling/contraction are ordinary
        # positional-or-keyword params with defaults: absent from the label
        # they must default, present in the label kwargs they must apply
        stack = [("Init State", None, (1,), {"qubit_labels": ["Q0"]})]
        program = QuantumProgram(
            stack, state_type=SVState, name="init state defaults"
        )
        state = program.run().shot_histories[0][-1]["state"]
        assert state.kraus_sampling == "lazy"
        assert state.contraction == "matmul"

        stack = [
            (
                "Init State",
                None,
                (1,),
                {
                    "qubit_labels": ["Q0"],
                    "kraus_sampling": "choice",
                    "contraction": "einsum",
                },
            )
        ]
        program = QuantumProgram(
            stack, state_type=SVState, name="init state overrides"
        )
        state = program.run().shot_histories[0][-1]["state"]
        assert state.kraus_sampling == "choice"
        assert state.contraction == "einsum"
