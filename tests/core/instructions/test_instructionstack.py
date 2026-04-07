"""Tester for loqs.core.instructions.instructionlabel"""

import os
from tempfile import NamedTemporaryFile
import pytest

import pytest

from loqs.core.instructions import Instruction, InstructionStack


class TestInstructionStack:

    @classmethod
    def setup_class(cls):
        def apply_fn():
            pass
        ins = Instruction(apply_fn, name="test") # type: ignore
        args = [ins.copy()]
        kwargs = {"ins": ins.copy()}

        cls.ilbl1 = ('Label', 'L0')
        cls.ilbl2 = (ins, 'L1', args, kwargs)
    
    def _check(self, stack, lbls):
        for el, lbl in zip(stack, lbls):
            assert el.patch_label == lbl

    def test_init(self):
        s = InstructionStack([self.ilbl1, self.ilbl2]) # type: ignore
        self._check(s, ["L0", "L1"])

        s1 = InstructionStack.cast([self.ilbl1, self.ilbl2])
        self._check(s1, ["L0", "L1"])

        s2 = InstructionStack.cast(s)
        self._check(s2, ["L0", "L1"])
    
    def test_list_operations(self):
        s = InstructionStack([self.ilbl1, self.ilbl2]) # type: ignore
        self._check(s, ["L0", "L1"])

        s2 = s.append_instruction(('test', 'L2')) # type: ignore
        self._check(s2, ["L0", "L1", "L2"])

        s3 = s.insert_instruction(0, ('test', 'L2')) # type: ignore
        self._check(s3, ["L2", "L0", "L1"])

        s4 = s.delete_instruction(0)
        self._check(s4, ["L1"])

        ilbl, s5 = s.pop_instruction()
        assert ilbl.patch_label == "L0"
        self._check(s5, ["L1"])
    
    @pytest.mark.skipif(os.getenv("RUNNER_OS", "N/A") == "Windows", reason="Permission issues on Windows GitHub runner")
    def test_serialization(self):
        s = InstructionStack([self.ilbl1, self.ilbl2]) # type: ignore

        with NamedTemporaryFile("w+", dir='.', suffix='.json') as tempf:
            s.write(tempf.name)

            s2 = InstructionStack.read(tempf.name)
            self._check(s2, ["L0", "L1"])

    def test_instruction_stack_serialization_comprehensive(self):
        """Comprehensive test of InstructionStack serialization methods."""
        # Create a more complex instruction stack
        stack = InstructionStack([self.ilbl1, self.ilbl2, self.ilbl1]) # type: ignore

        # Test string serialization
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            stack.write(tempf.name)
            loaded_stack = InstructionStack.read(tempf.name)
        self._check(loaded_stack, ["L0", "L1", "L0"])

        # Test file serialization
        with NamedTemporaryFile(suffix='.json') as f:
            stack.write(f.name)
            loaded_stack = InstructionStack.read(f.name)
            self._check(loaded_stack, ["L0", "L1", "L0"])

        # Test compressed format
        with NamedTemporaryFile(suffix='.json.gz', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            stack.write(temp_path)
            loaded_stack = InstructionStack.read(temp_path)
            self._check(loaded_stack, ["L0", "L1", "L0"])
        finally:
            import os
            os.unlink(temp_path)

    def test_instruction_stack_equality_after_serialization(self):
        """Test that InstructionStack equality is preserved after serialization."""
        original = InstructionStack([self.ilbl1, self.ilbl2]) # type: ignore

        # Serialize and deserialize
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            original.write(tempf.name)
            deserialized = InstructionStack.read(tempf.name)

        # Should be equal (content-wise after serial_hash removal)
        assert len(original) == len(deserialized)
        for i in range(len(original)):
            assert original[i].patch_label == deserialized[i].patch_label
            # Handle cases where instruction might be None
            if original[i].instruction is not None and deserialized[i].instruction is not None:
                assert original[i].instruction.name == deserialized[i].instruction.name

        # Check that individual elements are preserved
        for i in range(len(original)):
            assert original[i].patch_label == deserialized[i].patch_label

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_instruction_stack_serialization_comprehensive_parameterized(self, format):
        """Comprehensive test of InstructionStack serialization methods with both formats."""
        # Create a more complex instruction stack
        stack = InstructionStack([self.ilbl1, self.ilbl2, self.ilbl1]) # type: ignore

        # Test string serialization
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            stack.write(tempf.name)
            loaded_stack = InstructionStack.read(tempf.name)
        self._check(loaded_stack, ["L0", "L1", "L0"])

        # Test file serialization
        with NamedTemporaryFile(suffix=f'.{format}') as f:
            stack.write(f.name)
            loaded_stack = InstructionStack.read(f.name)
            self._check(loaded_stack, ["L0", "L1", "L0"])

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_instruction_stack_equality_after_serialization_parameterized(self, format):
        """Test that InstructionStack equality is preserved after serialization with both formats."""
        original = InstructionStack([self.ilbl1, self.ilbl2]) # type: ignore

        # Serialize and deserialize
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            original.write(tempf.name)
            deserialized = InstructionStack.read(tempf.name)

        # Should be equal (content-wise after serial_hash removal)
        assert len(original) == len(deserialized)
        for i in range(len(original)):
            assert original[i].patch_label == deserialized[i].patch_label
            # Handle cases where instruction might be None
            if original[i].instruction is not None and deserialized[i].instruction is not None:
                assert original[i].instruction.name == deserialized[i].instruction.name

        # Check that individual elements are preserved
        for i in range(len(original)):
            assert original[i].patch_label == deserialized[i].patch_label
