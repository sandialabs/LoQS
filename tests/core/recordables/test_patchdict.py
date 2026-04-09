"""Tester for loqs.core.recordables.patchdict"""

import os
from tempfile import NamedTemporaryFile
import json
import pytest

from loqs.core.recordables import PatchDict
from loqs.core.qeccode import QECCode


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
        
        with NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix='.json') as tmp:
            patches.write(tmp.name)
            tmp_path = tmp.name

        try:
            patches2 = PatchDict.read(tmp_path)
            assert patches2.all_qubit_labels == ["D0", "A0", "D1", "A1"]
        finally:
            os.unlink(tmp_path)
