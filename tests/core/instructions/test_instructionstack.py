"""Tester for loqs.core.instructions.instructionlabel"""

from tempfile import NamedTemporaryFile

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
    
    def test_serialization(self):
        s = InstructionStack([self.ilbl1, self.ilbl2]) # type: ignore

        with NamedTemporaryFile("w+", suffix='.json') as tempf:
            s.write(tempf.name)

            s2 = InstructionStack.read(tempf.name)
            self._check(s2, ["L0", "L1"])
