"""Tester for loqs.core.instructions.instructionlabel"""

import pytest

from loqs.core.instructions import Instruction, InstructionLabel


class TestInstructionLabel:

    @classmethod
    def setup_class(cls):
        def apply_fn():
            pass

        cls.ins = Instruction(apply_fn, name="test")  # type: ignore

    def test_init(self):
        ilbl = InstructionLabel("Label")
        assert ilbl["instruction"] == "Label"
        assert ilbl.get("patch_label") is None
        assert ilbl.get("patch_labels") is None

        ilbl = InstructionLabel("Label", patch_label="L0")
        assert ilbl["instruction"] == "Label"
        assert ilbl["patch_label"] == "L0"

        ilbl = InstructionLabel(self.ins, patch_label="L0")
        assert ilbl["instruction"] is self.ins
        assert ilbl["patch_label"] == "L0"

        # Any apply_fn parameter name is an ordinary dict key
        ilbl = InstructionLabel(
            "FT Logical X Measure Classical Decoder",
            patch_label="L0",
            flagged_check="XZIIZ",
            flagged_check_order=[4, 0, 1],
        )
        assert ilbl["flagged_check"] == "XZIIZ"
        assert ilbl["flagged_check_order"] == [4, 0, 1]

        ilbl = InstructionLabel(
            "CNOT Bookkeeping", patch_labels={"ctrl": "L0", "tgt": "L1"}
        )
        assert ilbl.get("patch_label") is None
        assert ilbl["patch_labels"] == {"ctrl": "L0", "tgt": "L1"}

    def test_from_raw(self):
        assert InstructionLabel.from_raw("Label") == InstructionLabel("Label")
        assert InstructionLabel.from_raw(self.ins) == InstructionLabel(
            self.ins
        )
        assert InstructionLabel.from_raw(("Label",)) == InstructionLabel(
            "Label"
        )
        assert InstructionLabel.from_raw(("Label", "L0")) == InstructionLabel(
            "Label", patch_label="L0"
        )
        raw = {
            "instruction": "Label",
            "patch_label": "L0",
            "flagged_check": "XZIIZ",
        }
        assert InstructionLabel.from_raw(raw) == InstructionLabel(
            "Label", patch_label="L0", flagged_check="XZIIZ"
        )
        built = InstructionLabel("Label", patch_label="L0")
        assert InstructionLabel.from_raw(built) is built

    def test_from_raw_rejects_long_tuples(self):
        with pytest.raises(TypeError):
            InstructionLabel.from_raw(
                ("Label", "L0", (), {"flagged_check": "XZIIZ"})
            )
