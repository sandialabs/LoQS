"""Tester for loqs.core.instructions.instructionstack"""

import pytest

from loqs.core.instructions import Instruction, InstructionLabel, InstructionStack


class TestInstructionStack:

    @classmethod
    def setup_class(cls):
        def apply_fn():
            pass
        ins = Instruction(apply_fn, name="test")  # type: ignore

        cls.ilbl1 = ("Label", "L0")
        cls.ilbl2 = {
            "instruction": ins,
            "patch_label": "L1",
            "flagged_check": "XZIIZ",
        }

    def _check(self, stack, lbls):
        for el, lbl in zip(stack, lbls):
            assert el.get("patch_label") == lbl

    def test_init(self):
        s = InstructionStack([self.ilbl1, self.ilbl2])
        self._check(s, ["L0", "L1"])

        s1 = InstructionStack([self.ilbl1, self.ilbl2])
        self._check(s1, ["L0", "L1"])

        s2 = InstructionStack(s)
        self._check(s2, ["L0", "L1"])

    def test_init_from_bare_labels(self):
        """A list of multiple bare instruction labels is a sequence of
        separate items, not one InstructionLabel's own raw form."""
        s = InstructionStack(["LabelA", "LabelB"])
        assert len(s) == 2
        assert s[0]["instruction"] == "LabelA"
        assert s[1]["instruction"] == "LabelB"

        # A bare tuple, in contrast, is one InstructionLabel's own raw form.
        s2 = InstructionStack(("Label", "L0"))
        assert len(s2) == 1
        assert s2[0]["instruction"] == "Label"
        assert s2[0]["patch_label"] == "L0"

    def test_init_with_multipatch_label(self):
        s = InstructionStack(
            [{"instruction": "CNOT Bookkeeping", "patch_labels": {"ctrl": "L0", "tgt": "L1"}}]
        )
        assert len(s) == 1
        assert s[0].get("patch_label") is None
        assert s[0]["patch_labels"] == {"ctrl": "L0", "tgt": "L1"}

    def test_list_operations(self):
        s = InstructionStack([self.ilbl1, self.ilbl2])
        self._check(s, ["L0", "L1"])

        s2 = s.append_instruction(("test", "L2"))
        self._check(s2, ["L0", "L1", "L2"])

        s3 = s.insert_instruction(0, ("test", "L2"))
        self._check(s3, ["L2", "L0", "L1"])

        s4 = s.delete_instruction(0)
        self._check(s4, ["L1"])

        ilbl, s5 = s.pop_instruction()
        assert ilbl["patch_label"] == "L0"
        self._check(s5, ["L1"])

    def test_serialization(self, make_temp_path):
        s = InstructionStack([self.ilbl1, self.ilbl2])

        # Create and write, but keep file closed before re-opening
        with make_temp_path(suffix=".json") as tmp_path:
            s.write(tmp_path)

            # Now safe to open/read/remove on Windows
            s2 = InstructionStack.read(tmp_path)
            self._check(s2, ["L0", "L1"])

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_serialization_round_trip(self, format, make_temp_path):
        # Every decoded entry comes back as a real InstructionLabel, since
        # InstructionStack.__init__ re-normalizes elements on construction.
        stack = InstructionStack([self.ilbl1, self.ilbl2, self.ilbl1])

        with make_temp_path(suffix=f".{format}") as tmp_path:
            stack.write(tmp_path)
            loaded_stack = InstructionStack.read(tmp_path)

        assert len(loaded_stack) == len(stack)
        for original, loaded in zip(stack, loaded_stack):
            assert isinstance(loaded, InstructionLabel)
            assert loaded.get("patch_label") == original.get("patch_label")
        assert loaded_stack[1]["flagged_check"] == "XZIIZ"
