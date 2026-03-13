"""Tester for loqs.core.qeccode"""

import os
from tempfile import NamedTemporaryFile

import pytest

from loqs.core.frame import Frame
from loqs.core.instructions import Instruction
from loqs.core.qeccode import QECCode

class TestQECCodeAndPatch:

    @classmethod
    def setup_class(cls):
        def apply_fn(state, qubits):
            return Frame({"state": state+1, 'qubits': qubits})
        def map_qubits_fn(qubit_mapping, qubits, **kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs["qubits"] = [qubit_mapping[q] for q in qubits]
            return new_kwargs
        data= {"qubits": ["Q0", "Q1"]}
        cls.ins = Instruction(apply_fn, data, map_qubits_fn, name="test")

    def test_code_patch(self):
        # Not much to test, it is just a container
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")

        patch = code.create_patch(["D0", "A0"])
        
        # Instruction from patch should be mapped to qubits
        ins2 = patch["ins"]
        result = ins2.apply(state=0, qubits=ins2.data["qubits"])
        assert result._data == {
            'state': 1,
            'qubits': ["D0", "A0"],
            'instruction': ins2,
            #'collected_params': {'state': 0, 'qubits': ins2.data["qubits"]}
        }
        assert result.log == "test result"
    
    @pytest.mark.skipif(os.getenv("RUNNER_OS", "N/A") == "Windows", reason="Permission issues on Windows GitHub runner")
    def test_serialization(self):
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")
        patch = code.create_patch(["D0", "A0"])

        # Patch should serialize code, so just do that
        with NamedTemporaryFile("w+", dir='.', suffix='.json') as tempf:
            patch.write(tempf.name)

            patch2 = Frame.read(tempf.name)
        
            ins2 = patch2["ins"]
            result = ins2.apply(state=0, qubits=ins2.data["qubits"])
            assert result._data == {
                'state': 1,
                'qubits': ["D0", "A0"],
                'instruction': ins2,
                #'collected_params': {'state': 0, 'qubits': ins2.data["qubits"]}
            }
            assert result.log == "test result"
            