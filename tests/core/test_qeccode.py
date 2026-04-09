"""Tester for loqs.core.qeccode"""

import os
from tempfile import NamedTemporaryFile
import json
import pytest

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
    
    def test_serialization(self):
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")
        patch = code.create_patch(["D0", "A0"])

        # Patch should serialize code, so just do that
        with NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix='.json') as tmp:
            patch.write(tmp.name)
            tmp_path = tmp.name

        try:
            patch2 = Frame.read(tmp_path)
        
            ins2 = patch2["ins"]
            result = ins2.apply(state=0, qubits=ins2.data["qubits"])
            assert result._data == {
                'state': 1,
                'qubits': ["D0", "A0"],
                'instruction': ins2,
                #'collected_params': {'state': 0, 'qubits': ins2.data["qubits"]}
            }
            assert result.log == "test result"
        finally:
            os.unlink(tmp_path)

    def test_qeccode_serialization(self):
        """Test QECCode serialization roundtrip."""
        # Create a QEC code with instructions
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")

        # Test string serialization
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            code.write(tempf.name)
            loaded_code = QECCode.read(tempf.name)

        # Verify structure is preserved
        assert loaded_code.name == "Test code"
        assert loaded_code.template_qubits == ["Q0", "Q1"]
        assert loaded_code.template_data_qubits == ["Q0"]

        # Test file serialization
        with NamedTemporaryFile(suffix='.json') as f:
            code.write(f.name)
            loaded_code = QECCode.read(f.name)
            assert loaded_code.name == "Test code"

    def test_qeccode_patch_serialization(self):
        """Test QECCode patch serialization."""
        # Create a QEC code and patch
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")
        patch = code.create_patch(["D0", "A0"])

        # Test patch serialization
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            patch.write(tempf.name)
            loaded_patch = QECCode.read(tempf.name)  # Patches are loaded as QECCode objects

        # Verify the patch data is preserved
        assert loaded_patch.code.name == "Test code"
        assert loaded_patch.qubits == ["D0", "A0"]
        # The instruction should be mapped to the patch qubits
        loaded_ins = loaded_patch["ins"]
        assert loaded_ins.data["qubits"] == ["D0", "A0"]

    def test_qeccode_compressed_serialization(self):
        """Test QECCode serialization with compressed format."""
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")

        with NamedTemporaryFile(suffix='.json.gz', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Write compressed
            code.write(temp_path)

            # Read compressed
            loaded_code = QECCode.read(temp_path)
            # Verify structure is preserved
            assert loaded_code.name == "Test code"
            assert loaded_code.template_qubits == ["Q0", "Q1"]
            assert loaded_code.template_data_qubits == ["Q0"]

        finally:
            import os
            os.unlink(temp_path)

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_qeccode_serialization_parameterized(self, format):
        """Test QECCode serialization roundtrip with both JSON and HDF5 formats."""
        # Create a QEC code with instructions
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")

        # Test string serialization
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            code.write(tempf.name)
            loaded_code = QECCode.read(tempf.name)

        # Verify structure is preserved
        assert loaded_code.name == "Test code"
        assert loaded_code.template_qubits == ["Q0", "Q1"]
        assert loaded_code.template_data_qubits == ["Q0"]

        # Test file serialization
        with NamedTemporaryFile(suffix=f'.{format}') as f:
            code.write(f.name)
            loaded_code = QECCode.read(f.name)
            assert loaded_code.name == "Test code"

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_qeccode_patch_serialization_parameterized(self, format):
        """Test QECCode patch serialization with both JSON and HDF5 formats."""
        # Create a QEC code and patch
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")
        patch = code.create_patch(["D0", "A0"])

        # Test patch serialization
        with NamedTemporaryFile("w+", suffix=".json") as tempf:
            patch.write(tempf.name)
            loaded_patch = QECCode.read(tempf.name)  # Patches are loaded as QECCode objects

        # Verify the patch data is preserved
        assert loaded_patch.code.name == "Test code"
        assert loaded_patch.qubits == ["D0", "A0"]
        # The instruction should be mapped to the patch qubits
        loaded_ins = loaded_patch["ins"]
        assert loaded_ins.data["qubits"] == ["D0", "A0"]
            