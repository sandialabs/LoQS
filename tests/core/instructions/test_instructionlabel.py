"""Tester for loqs.core.instructions.instructionlabel"""

import pytest

from loqs.core.instructions import Instruction, InstructionLabel


class TestInstructionLabel:

    @classmethod
    def setup_class(cls):
        def apply_fn():
            pass
        cls.ins = Instruction(apply_fn, name="test")  # type: ignore

    def test_init_rejects_bad_instruction_type(self):
        with pytest.raises(TypeError, match="instruction must be an Instruction or str"):
            InstructionLabel(3)  # type: ignore

    def test_init_bare_label(self):
        ilbl = InstructionLabel("Label")
        assert ilbl["instruction"] == "Label"
        assert ilbl.get("patch_label") is None
        assert ilbl.get("patch_labels") is None

    def test_init_with_patch_label(self):
        ilbl = InstructionLabel("Label", patch_label="L0")
        assert ilbl["instruction"] == "Label"
        assert ilbl["patch_label"] == "L0"

    def test_init_with_instruction_object(self):
        ilbl = InstructionLabel(self.ins, patch_label="L0")
        assert ilbl["instruction"] is self.ins
        assert ilbl["patch_label"] == "L0"

    def test_init_with_arbitrary_kwargs(self):
        # No hardcoded positional slot needed to reach extra kwargs -- any
        # apply_fn parameter name is just an ordinary dict key.
        ilbl = InstructionLabel(
            "FT Logical X Measure Classical Decoder",
            patch_label="L0",
            flagged_check="XZIIZ",
            flagged_check_order=[4, 0, 1],
        )
        assert ilbl["flagged_check"] == "XZIIZ"
        assert ilbl["flagged_check_order"] == [4, 0, 1]

    def test_init_with_patch_labels_multipatch(self):
        ilbl = InstructionLabel(
            "CNOT Bookkeeping", patch_labels={"ctrl": "L0", "tgt": "L1"}
        )
        assert ilbl.get("patch_label") is None
        assert ilbl["patch_labels"] == {"ctrl": "L0", "tgt": "L1"}

    def test_from_raw_bare_str_and_instruction(self):
        assert InstructionLabel.from_raw("Label") == InstructionLabel("Label")
        assert InstructionLabel.from_raw(self.ins) == InstructionLabel(self.ins)

    def test_from_raw_one_tuple(self):
        assert InstructionLabel.from_raw(("Label",)) == InstructionLabel("Label")

    def test_from_raw_two_tuple(self):
        assert InstructionLabel.from_raw(("Label", "L0")) == InstructionLabel(
            "Label", patch_label="L0"
        )

    def test_from_raw_dict(self):
        d = {"instruction": "Label", "patch_label": "L0", "flagged_check": "XZIIZ"}
        assert InstructionLabel.from_raw(d) == InstructionLabel(
            "Label", patch_label="L0", flagged_check="XZIIZ"
        )

    def test_from_raw_already_built_label_returns_same_object(self):
        ilbl = InstructionLabel("Label", patch_label="L0")
        assert InstructionLabel.from_raw(ilbl) is ilbl

    def test_from_raw_rejects_long_tuples(self):
        # The old fixed-position tuple format is no longer supported --
        # use the dict form instead.
        with pytest.raises(TypeError, match="Tuples longer than 2 elements"):
            InstructionLabel.from_raw(("Label", "L0", (), {"flagged_check": "XZIIZ"}))

    def test_from_raw_rejects_unrecognized_type(self):
        with pytest.raises(TypeError, match="Cannot cast"):
            InstructionLabel.from_raw(3)  # type: ignore

    def test_equality_is_structural(self):
        # dict.__eq__ gives real, content-based equality for free -- no
        # hand-written __eq__ needed, and no more relying on object
        # identity (the previous class had no __eq__ at all).
        a = InstructionLabel("Label", patch_label="L0")
        b = InstructionLabel("Label", patch_label="L0")
        c = InstructionLabel("Label", patch_label="L1")
        assert a == b
        assert a is not b
        assert a != c

    def test_dict_style_access(self):
        # InstructionLabel really is just a dict -- ordinary dict
        # operations (mutation, iteration, **-spreading) all just work.
        ilbl = InstructionLabel("Label", patch_label="L0")
        ilbl["error_injections"] = [(0, "Gxpi", 3)]
        assert ilbl["error_injections"] == [(0, "Gxpi", 3)]
        assert set(ilbl.keys()) == {"instruction", "patch_label", "error_injections"}
