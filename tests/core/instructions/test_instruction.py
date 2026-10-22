"""Tester for loqs.core.instructions.instruction"""

import warnings

import numpy as np
import pytest

from loqs.core import InstructionStack
from loqs.core.frame import Frame
from loqs.core.instructions import Instruction, builders
from loqs.core.instructions.instruction import DEFAULT_PRIORITIES


class TestInstruction:

    def test_init_apply_map(self):
        def apply_fn(state, qubits):
            return Frame({"state": state + 1, "qubits": qubits})

        def map_qubits_fn(qubit_mapping, qubits, **kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs["qubits"] = [qubit_mapping[q] for q in qubits]
            return new_kwargs

        data = {"qubits": ["Q0", "Q1"]}

        ins = Instruction(apply_fn, data, map_qubits_fn, name="test")

        # Check apply works as expected
        result = ins.apply(state=0, qubits=ins.data["qubits"])
        assert result._data == {
            "state": 1,
            "qubits": ["Q0", "Q1"],
            "instruction": ins,
        }

        # Check map works as expected
        ins2 = ins.map_qubits({"Q1": "A3", "Q0": "D2"})
        result2 = ins2.apply(state=0, qubits=ins2.data["qubits"])
        assert result2._data == {
            "state": 1,
            "qubits": ["D2", "A3"],
            "instruction": ins2,
        }

        # Also check we can set param priorities and aliases
        # Full testing of these should happen in QuantumProgram though
        ins3 = Instruction(
            apply_fn,
            data,
            map_qubits_fn,
            param_aliases={"state": "state_name_in_program"},
            param_error_behavior="raise",
            param_priorities={"qubits": ["instruction"]},
            name="test",
        )
        assert ins3.param_priorities == {
            "state": DEFAULT_PRIORITIES,
            "qubits": ["instruction"],
        }
        assert ins3.param_alias("state") == "state_name_in_program"

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_serialization_round_trip(self, format, make_temp_path):
        def apply_fn(state, qubits):
            return Frame(
                {
                    "state": state + 1,
                    "qubits": qubits,
                    "result": f"state_{state}_qubits_{'_'.join(qubits)}",
                }
            )

        def map_qubits_fn(qubit_mapping, qubits, **kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs["qubits"] = [qubit_mapping.get(q, q) for q in qubits]
            return new_kwargs

        data = {
            "qubits": ["Q0", "Q1"],
            "complex": {"nested": "data"},
            "items": [1, 2, {"inner": "dict"}],
        }
        ins = Instruction(apply_fn, data, map_qubits_fn, name="round_trip")

        with make_temp_path(suffix=f".{format}") as tmp_path:
            ins.write(tmp_path)
            loaded = Instruction.read(tmp_path)

        assert isinstance(loaded, Instruction)
        assert loaded.name == "round_trip"
        assert loaded.data == data

        original_result = ins.apply(state=5, qubits=["Q0", "Q1"])
        loaded_result = loaded.apply(state=5, qubits=["Q0", "Q1"])
        assert original_result._data["result"] == loaded_result._data["result"]
        assert original_result._data["state"] == loaded_result._data["state"]

        mapped = loaded.map_qubits({"Q0": "X", "Q1": "Y"})
        mapped_result = mapped.apply(state=5, qubits=mapped.data["qubits"])
        assert mapped_result._data["qubits"] == ["X", "Y"]
        assert mapped_result._data["result"] == "state_5_qubits_X_Y"

        # The loaded instruction has no source to re-read, so this rewrite
        # relies on the serialized function text it kept
        with make_temp_path(suffix=f".{format}") as tmp_path:
            loaded.write(tmp_path)
            reloaded = Frame.read(tmp_path)

        assert isinstance(reloaded, Instruction)
        reloaded_result = reloaded.apply(state=5, qubits=["Q0", "Q1"])
        assert (
            reloaded_result._data["result"] == original_result._data["result"]
        )

    def test_equality_and_hash(self):
        """Equal instructions hash equal, which the encode cache relies on."""

        def other_apply(state):
            return Frame({"state": state + 2})

        a = Instruction(_module_level_apply, name="x")
        b = Instruction(_module_level_apply, name="x")
        c = Instruction(_module_level_apply, name="y")
        d = Instruction(other_apply, name="x")

        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1
        assert a != c
        assert a != d
        # Comparison against non-instructions is not an error
        assert a != "x"
        assert (a, "L0") != ("x", "L0")

    def test_eq_compares_array_data(self, make_temp_path):
        """Array-valued data compares by content, through an HDF5 decode."""
        data = {"H_X": np.array([[1, 0, 1], [0, 1, 1]]), "labels": ["a", "b"]}
        ins = Instruction(_module_level_apply, data, name="arrays")

        with make_temp_path(suffix=".h5") as tmp_path:
            ins.write(tmp_path)
            loaded = Instruction.read(tmp_path)

        assert ins == loaded

        changed = Instruction(
            _module_level_apply,
            {"H_X": np.array([[1, 0, 1], [0, 1, 0]]), "labels": ["a", "b"]},
            name="arrays",
        )
        assert ins != changed

        reshaped = Instruction(
            _module_level_apply,
            {"H_X": np.array([1, 0, 1, 0, 1, 1]), "labels": ["a", "b"]},
            name="arrays",
        )
        assert ins != reshaped

    def test_tuple_held_instructions_survive_encode_cache(
        self, make_temp_path
    ):
        """Instructions that share name and type (so hash equal) but differ
        in data, held in `(instruction, label)` tuples, keep their own data
        through the encode cache."""
        a = Instruction(
            _module_level_apply, {"value": 1}, name="same", type="T"
        )
        b = Instruction(
            _module_level_apply, {"value": 2}, name="same", type="T"
        )
        comp_a = builders.build_composite_instruction([(a, "L0")])
        comp_b = builders.build_composite_instruction([(b, "L0")])
        stack = InstructionStack([(comp_a, "L0"), (comp_b, "L0")])

        with make_temp_path(suffix=".h5") as tmp_path:
            stack.write(tmp_path)
            loaded = InstructionStack.read(tmp_path)

        values = []
        for label in loaded:
            sub = label["instruction"].data["instructions"][0]
            sub_inst = sub[0] if isinstance(sub, tuple) else sub
            values.append(sub_inst.data["value"])
        assert values == [1, 2]


def _module_level_apply(state=0):
    return Frame({"state": state + 1})


_EXEC_SOURCE = (
    "def apply_fn(state, qubits):\n"
    "    return {'state': state, 'qubits': qubits}\n"
)


def test_source_unavailable_warns():
    env = {}
    exec(_EXEC_SOURCE, env)

    with pytest.warns(UserWarning):
        Instruction(env["apply_fn"], {}, name="test")

    # A module-level function, or one given its source explicitly (as the
    # decode path does), has nothing to warn about
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Instruction(_module_level_apply, {}, name="test")
        Instruction(
            env["apply_fn"],
            {},
            name="test",
            serialized_apply_fn="def apply_fn():\n    pass\n",
        )
    assert len(caught) == 0
