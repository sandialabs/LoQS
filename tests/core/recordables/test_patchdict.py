"""Tester for loqs.core.recordables.patchdict"""

from tempfile import NamedTemporaryFile
import pytest

from loqs.core.recordables import PatchDict
from loqs.core.qeccode import QECCode, QECCodePatch


class TestMeasurementOutcomes:

    def test_init_all_qubits(self):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        patch1 = code.create_patch(["D0", "A0"])
        patch2 = code.create_patch(["D1", "A1"])

        patches = PatchDict({"L0": patch1})
        assert patches.all_qubit_labels == ["D0", "A0"]

        patches["L1"] = patch2
        assert patches.all_qubit_labels == ["D0", "A0", "D1", "A1"]

        with pytest.raises(AssertionError):
            PatchDict({"key": "not a patch"}) # type: ignore
    
    def test_serialization(self):
        code = QECCode({}, ["Q0", "Q1"], ["Q0"])
        patch1 = code.create_patch(["D0", "A0"])
        patch2 = code.create_patch(["D1", "A1"])
        patches = PatchDict({"L0": patch1, "L1": patch2})
        
        with NamedTemporaryFile("w+", suffix='.json') as tempf:
            patches.write(tempf.name)

            patches2 = PatchDict.read(tempf.name)
            assert patches2.all_qubit_labels == ["D0", "A0", "D1", "A1"]
