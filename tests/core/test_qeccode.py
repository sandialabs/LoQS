"""Tester for loqs.core.qeccode"""

import pytest

from loqs.core.frame import Frame
from loqs.core.instructions import Instruction
from loqs.core.qeccode import QECCode, QECCodePatch


class TestQECCodeAndPatch:

    @classmethod
    def setup_class(cls):
        def apply_fn(state, qubits):
            return Frame({"state": state + 1, "qubits": qubits})

        def map_qubits_fn(qubit_mapping, qubits, **kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs["qubits"] = [qubit_mapping[q] for q in qubits]
            return new_kwargs

        data = {"qubits": ["Q0", "Q1"]}
        cls.ins = Instruction(apply_fn, data, map_qubits_fn, name="test")

    def test_code_patch(self):
        # Not much to test, it is just a container
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")

        patch = code.create_patch(["D0", "A0"])

        # Instruction from patch should be mapped to qubits
        ins2 = patch["ins"]
        result = ins2.apply(state=0, qubits=ins2.data["qubits"])
        assert result._data == {
            "state": 1,
            "qubits": ["D0", "A0"],
            "instruction": ins2,
        }

    def test_data_qubits_multiple_out_of_order(self):
        # template_data_qubits in a different order than template_qubits,
        # to confirm data_qubits maps by position, not just takes a prefix.
        code = QECCode(
            {"ins": self.ins}, ["Q0", "Q1", "Q2"], ["Q1", "Q0"], "Test code"
        )
        patch = code.create_patch(["D0", "D1", "A0"])
        assert patch.data_qubits == ["D1", "D0"]

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_serialization(self, format, make_temp_path):
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")
        patch = code.create_patch(["D0", "A0"])

        # Patch should serialize code, so just do that
        with make_temp_path(suffix=f".{format}") as tmp_path:
            patch.write(tmp_path)
            patch2 = QECCodePatch.read(tmp_path)

        assert isinstance(patch2, QECCodePatch)
        assert patch2.code.name == "Test code"
        assert patch2.code.template_qubits == ["Q0", "Q1"]
        assert patch2.code.template_data_qubits == ["Q0"]
        assert patch2.qubits == ["D0", "A0"]

        ins2 = patch2["ins"]
        assert isinstance(ins2, Instruction)
        assert ins2.data["qubits"] == ["D0", "A0"]
        result = ins2.apply(state=0, qubits=ins2.data["qubits"])
        assert result._data == {
            "state": 1,
            "qubits": ["D0", "A0"],
            "instruction": ins2,
        }
