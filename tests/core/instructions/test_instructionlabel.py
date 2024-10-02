"""Tester for loqs.core.instructions.instructionlabel"""

from tempfile import NamedTemporaryFile

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

        with NamedTemporaryFile("w+", suffix='.json') as tempf:
            ilbl.write(tempf.name)

            ilbl2 = InstructionLabel.read(tempf.name)
            self._check(ilbl2, None, "Label", "L0", self.args, self.kwargs)

        # And instruction version
        ilbl3 = InstructionLabel(self.ins, "L0", self.args, self.kwargs)

        with NamedTemporaryFile("w+", suffix='.json') as tempf:
            ilbl3.write(tempf.name)

            ilbl4 = InstructionLabel.read(tempf.name)
            self._check(ilbl4, self.ins, None, "L0", self.args, self.kwargs)
            