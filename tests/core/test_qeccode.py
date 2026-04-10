"""Tester for loqs.core.qeccode"""

import pytest

from loqs.core.frame import Frame
from loqs.core.instructions import Instruction
from loqs.core.qeccode import QECCode, QECCodePatch

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
    
    def test_serialization(self, make_temp_path):
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")
        patch = code.create_patch(["D0", "A0"])

        # Patch should serialize code, so just do that
        with make_temp_path(suffix='.json') as tmp_path:
            patch.write(tmp_path)
            patch2 = QECCodePatch.read(tmp_path)
            assert isinstance(patch2, QECCodePatch)

            ins2 = patch2["ins"]
            assert isinstance(ins2, Instruction)
            result = ins2.apply(state=0, qubits=ins2.data["qubits"])
            assert result._data == {
                'state': 1,
                'qubits': ["D0", "A0"],
                'instruction': ins2,
                #'collected_params': {'state': 0, 'qubits': ins2.data["qubits"]}
            }
            assert result.log == "test result"

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_qeccode_serialization_parameterized(self, format, make_temp_path):
        """Test QECCode serialization roundtrip with both JSON and HDF5 formats."""
        # Create a QEC code with instructions
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")

        with make_temp_path(suffix=f'.{format}') as f_path:
            code.write(f_path)
            loaded_code = QECCode.read(f_path)
            assert isinstance(loaded_code, QECCode)
            # Verify structure is preserved
            assert loaded_code.name == "Test code"
            assert loaded_code.template_qubits == ["Q0", "Q1"]
            assert loaded_code.template_data_qubits == ["Q0"]

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_qeccode_patch_serialization_parameterized(self, format, make_temp_path):
        """Test QECCode patch serialization with both JSON and HDF5 formats."""
        # Create a QEC code and patch
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")
        patch = code.create_patch(["D0", "A0"])

        # Test patch serialization
        with make_temp_path(suffix=f".{format}") as tempf_path:
            patch.write(tempf_path)
            loaded_patch = QECCodePatch.read(tempf_path)
            assert isinstance(loaded_patch, QECCodePatch)

            # Verify the patch data is preserved
            assert loaded_patch.code.name == "Test code"
            assert loaded_patch.qubits == ["D0", "A0"]
            # The instruction should be mapped to the patch qubits
            loaded_ins = loaded_patch["ins"]
            assert loaded_ins.data["qubits"] == ["D0", "A0"]
