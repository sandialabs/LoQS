"""Tester for loqs.core.instructions.instructionlabel"""

import os
import tempfile
import json

import pytest

from loqs.core.instructions import Instruction, InstructionLabel


class TestInstructionLabel:

    @classmethod
    def setup_class(cls):
        def apply_fn():
            pass
        cls.ins = Instruction(apply_fn, name="test") # type: ignore
        cls.args = [cls.ins]
        cls.kwargs = {"ins": cls.ins}

    def _check(self, ilbl, ins, il, pl, a, k):
        assert ilbl.instruction is None or ilbl.instruction.name == ins.name
        assert ilbl.inst_label == il
        assert ilbl.patch_label == pl
        if len(a):
            assert ilbl.inst_args[0].name == a[0].name
        else:
            assert ilbl.inst_args == a
        if len(k):
            assert ilbl.inst_kwargs['ins'].name == k['ins'].name
        else:
            assert ilbl.inst_kwargs == k

    def test_init(self):
        ilbl = InstructionLabel("Label")
        self._check(ilbl, None, "Label", None, (), {})

        ilbl2 = InstructionLabel.cast("Label")
        self._check(ilbl2, None, "Label", None, (), {})

        ilbl3 = InstructionLabel.cast(("Label",))
        self._check(ilbl3, None, "Label", None, (), {})

        # With patch label
        ilbl4 = InstructionLabel("Label", "L0")
        self._check(ilbl4, None, "Label", "L0", (), {})

        ilbl5 = InstructionLabel.cast(("Label", "L0"))
        self._check(ilbl5, None, "Label", "L0", (), {})

        # With args and kwargs
        ilbl6 = InstructionLabel("Label", "L0", self.args, self.kwargs)
        self._check(ilbl6, None, "Label", "L0", self.args, self.kwargs)

        ilbl7 = InstructionLabel.cast(("Label", "L0", self.args, self.kwargs))
        self._check(ilbl7, None, "Label", "L0", self.args, self.kwargs)

        # With instruction instead of label
        ilbl8 = InstructionLabel(self.ins, "L0", self.args, self.kwargs)
        self._check(ilbl8, self.ins, None, "L0", self.args, self.kwargs)

        ilbl9 = InstructionLabel.cast((self.ins, "L0", self.args, self.kwargs))
        self._check(ilbl9, self.ins, None, "L0", self.args, self.kwargs)

        # and the solo instruction casts
        ilbl10 = InstructionLabel.cast(self.ins)
        self._check(ilbl10, self.ins, None, None, (), {})

        ilbl11 = InstructionLabel.cast((self.ins,))
        self._check(ilbl11, self.ins, None, None, (), {})

    
    def test_serialization(self):
        # Test string version
        ilbl = InstructionLabel("Label", "L0", self.args, self.kwargs)

        fd, tmp_path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            ilbl.write(tmp_path)
            ilbl2 = InstructionLabel.read(tmp_path)
            self._check(ilbl2, None, "Label", "L0", self.args, self.kwargs)
        finally:
            os.unlink(tmp_path)

        # And instruction version
        ilbl3 = InstructionLabel(self.ins, "L0", self.args, self.kwargs)

        fd, tmp_path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            ilbl3.write(tmp_path)
            ilbl4 = InstructionLabel.read(tmp_path)
            self._check(ilbl4, self.ins, None, "L0", self.args, self.kwargs)
        finally:
            os.unlink(tmp_path)

    def test_instruction_label_serialization_comprehensive(self):
        """Comprehensive test of InstructionLabel serialization methods."""
        # Test with instruction and all parameters
        label = InstructionLabel(self.ins, "L1", self.args, self.kwargs)

        # Test string serialization
        fd, tmp_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            label.write(tmp_path)
            loaded_label = InstructionLabel.read(tmp_path)
            self._check(loaded_label, self.ins, None, "L1", self.args, self.kwargs)
        finally:
            os.unlink(tmp_path)

        # Test file serialization
        fd, tmp_path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            label.write(tmp_path)
            loaded_label = InstructionLabel.read(tmp_path)
            self._check(loaded_label, self.ins, None, "L1", self.args, self.kwargs)
        finally:
            os.unlink(tmp_path)

        # Test compressed format
        fd, temp_path = tempfile.mkstemp(suffix='.json.gz')
        os.close(fd)
        try:
            label.write(temp_path)
            loaded_label = InstructionLabel.read(temp_path)
            self._check(loaded_label, self.ins, None, "L1", self.args, self.kwargs)
        finally:
            os.unlink(temp_path)

    def test_instruction_label_without_instruction(self):
        """Test InstructionLabel serialization without instruction."""
        # Test label without instruction
        label = InstructionLabel(self.ins, "L0")

        fd, tmp_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            label.write(tmp_path)
            loaded_label = InstructionLabel.read(tmp_path)
            self._check(loaded_label, self.ins, None, "L0", (), {})
        finally:
            os.unlink(tmp_path)

    def test_instruction_label_equality_after_serialization(self):
        """Test that InstructionLabel equality is preserved after serialization."""
        original = InstructionLabel(self.ins, "L2", self.args)

        # Serialize and deserialize
        fd, tmp_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            original.write(tmp_path)
            deserialized = InstructionLabel.read(tmp_path)
        finally:
            os.unlink(tmp_path)

        # Should be equal (content-wise after serial_hash removal)
        assert original.patch_label == deserialized.patch_label
        assert original.instruction.name == deserialized.instruction.name
        # Check lengths and types of args/kwargs (not exact equality due to different object instances)
        assert len(original.inst_args) == len(deserialized.inst_args)
        assert len(original.inst_kwargs) == len(deserialized.inst_kwargs)

        # Different label should not be equal
        different = InstructionLabel(self.ins, "L3", self.args)
        assert original != different
            