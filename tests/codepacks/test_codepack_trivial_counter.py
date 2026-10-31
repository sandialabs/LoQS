"""Tester for loqs.codepacks.trivial_counter"""

import pytest

from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.codepacks import BaseCodepack, codepack_trivial_counter
from loqs.codepacks.trivial_counter import TrivialCounterCodepack
from loqs.core import Frame


class TestTrivialCounterCodepack:

    @classmethod
    def setup_class(cls):
        cls.codepack = TrivialCounterCodepack()
        cls.code = cls.codepack.create_qec_code()

    def test_create_qec_code(self):
        assert isinstance(self.codepack, BaseCodepack)
        code = self.code
        assert code.name == "Trivial Counter Code"
        assert set(code.instructions) == {"Increment", "Init Counter", "Sleep"}
        assert code.instructions["Increment"].data == {"increment_by": 1}
        assert code.template_qubits == ["Q0"]
        assert code.template_data_qubits == ["Q0"]

    def test_increment_instruction(self):
        increment = self.code.instructions["Increment"]
        assert increment.apply(counter=0, increment_by=1)["counter"] == 1
        assert increment.apply(counter=5, increment_by=3)["counter"] == 8

    def test_init_counter_instruction(self):
        init = self.code.instructions["Init Counter"]
        assert init.apply(initial_value=42)["counter"] == 42

    def test_sleep_instruction(self):
        sleep = self.code.instructions["Sleep"]
        frame = sleep.apply(duration=0.0, counter=3)
        assert isinstance(frame, Frame)
        assert frame["counter"] == 3
        # With no counter given, the instruction's default (None) is used
        assert sleep.apply(duration=0.0)["counter"] is None


def test_deprecated_wrappers():
    with pytest.warns(DeprecationWarning):
        code = codepack_trivial_counter.create_qec_code()
    assert set(code.instructions) == set(
        TrivialCounterCodepack().create_qec_code().instructions
    )

    with pytest.warns(DeprecationWarning):
        model = codepack_trivial_counter.create_ideal_model(["Q0"])
    assert isinstance(model, DictNoiseModel)
    assert model.gate_dict == {}
    assert model.inst_dict == {}
