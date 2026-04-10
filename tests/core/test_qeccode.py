"""Tester for loqs.core.qeccode"""

import os
import tempfile
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
        fd, tmp_path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            patch.write(tmp_path)
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
        fd, tempf_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            code.write(tempf_path)
            loaded_code = QECCode.read(tempf_path)

            # Verify structure is preserved
            assert loaded_code.name == "Test code"
            assert loaded_code.template_qubits == ["Q0", "Q1"]
            assert loaded_code.template_data_qubits == ["Q0"]
        finally:
            os.unlink(tempf_path)

        # Test file serialization
        fd, f_path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            code.write(f_path)
            loaded_code = QECCode.read(f_path)
            assert loaded_code.name == "Test code"
        finally:
            os.unlink(f_path)

    def test_qeccode_patch_serialization(self):
        """Test QECCode patch serialization."""
        # Create a QEC code and patch
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")
        patch = code.create_patch(["D0", "A0"])

        # Test patch serialization
        fd, tempf_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            patch.write(tempf_path)
            loaded_patch = QECCode.read(tempf_path)  # Patches are loaded as QECCode objects

            # Verify the patch data is preserved
            assert loaded_patch.code.name == "Test code"
            assert loaded_patch.qubits == ["D0", "A0"]
            # The instruction should be mapped to the patch qubits
            loaded_ins = loaded_patch["ins"]
            assert loaded_ins.data["qubits"] == ["D0", "A0"]
        finally:
            os.unlink(tempf_path)

    def test_qeccode_compressed_serialization(self):
        """Test QECCode serialization with compressed format."""
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")

        fd, temp_path = tempfile.mkstemp(suffix='.json.gz')
        os.close(fd)
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
            os.unlink(temp_path)

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_qeccode_serialization_parameterized(self, format):
        """Test QECCode serialization roundtrip with both JSON and HDF5 formats."""
        # Create a QEC code with instructions
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")

        # Test string serialization
        fd, tempf_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            code.write(tempf_path)
            loaded_code = QECCode.read(tempf_path)

            # Verify structure is preserved
            assert loaded_code.name == "Test code"
            assert loaded_code.template_qubits == ["Q0", "Q1"]
            assert loaded_code.template_data_qubits == ["Q0"]
        finally:
            os.unlink(tempf_path)

        # Test file serialization
        fd, f_path = tempfile.mkstemp(suffix=f'.{format}')
        os.close(fd)
        try:
            code.write(f_path)
            loaded_code = QECCode.read(f_path)
            assert loaded_code.name == "Test code"
        finally:
            os.unlink(f_path)

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_qeccode_patch_serialization_parameterized(self, format):
        """Test QECCode patch serialization with both JSON and HDF5 formats."""
        # Create a QEC code and patch
        code = QECCode({"ins": self.ins}, ["Q0", "Q1"], ["Q0"], "Test code")
        patch = code.create_patch(["D0", "A0"])

        # Test patch serialization
        fd, tempf_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            patch.write(tempf_path)
            loaded_patch = QECCode.read(tempf_path)  # Patches are loaded as QECCode objects

            # Verify the patch data is preserved
            assert loaded_patch.code.name == "Test code"
            assert loaded_patch.qubits == ["D0", "A0"]
            # The instruction should be mapped to the patch qubits
            loaded_ins = loaded_patch["ins"]
            assert loaded_ins.data["qubits"] == ["D0", "A0"]
        finally:
            os.unlink(tempf_path)
            