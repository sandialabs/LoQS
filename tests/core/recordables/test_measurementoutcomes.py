"""Tester for loqs.core.recordables.measurementoutcomes"""

from tempfile import NamedTemporaryFile
import pytest

from loqs.core.recordables import MeasurementOutcomes
from loqs.core.syndrome import PauliFrame


class TestMeasurementOutcomes:

    def _check(self, mo, outcomes):
        assert set(mo.keys()) == set(outcomes.keys())
        for k,v in mo.items():
            assert v == outcomes[k]

    def test_init(self):
        outcomes = {"Q0": [0, 1], "Q1": 1}
        expected = {"Q0": [0, 1], "Q1": [1]}
        m = MeasurementOutcomes(outcomes)
        self._check(m, expected)

        m2 = MeasurementOutcomes(m)
        self._check(m2, expected)

        m3 = MeasurementOutcomes.cast(m)
        assert m3 is m

        m4 = MeasurementOutcomes.cast(outcomes)
        self._check(m4, expected)

        with pytest.raises(TypeError):
            MeasurementOutcomes([0, 1, 0]) # type: ignore
        
    def test_inferred(self):
        qubits = ["Q0", "Q1","Q2","Q3"]
        outcomes = {k: [0, 1] for k in qubits}
        m = MeasurementOutcomes(outcomes)
        pf = PauliFrame(qubits, "IXYZ")

        # X basis, Y and Z errors cause bitflips
        X_expected = {"Q0": [0, 1], "Q1": [0,1], "Q2": [1,0], "Q3": [1,0]}
        m2 = m.get_inferred_outcomes("X", pf)
        self._check(m2, X_expected)

        # Z basis, X and Y errors cause bitflips
        Z_expected = {"Q0": [0, 1], "Q1": [1,0], "Q2": [1,0], "Q3": [0,1]}
        m3 = m.get_inferred_outcomes("Z", pf)
        self._check(m3, Z_expected)
    
    def test_serialization(self):
        outcomes = {"Q0": [0, 1], "Q1": 1}
        expected = {"Q0": [0, 1], "Q1": [1]}
        m = MeasurementOutcomes(outcomes)

        with NamedTemporaryFile("w+", suffix='.json') as tempf:
            m.write(tempf.name)

            m2 = MeasurementOutcomes.read(tempf.name)
            self._check(m2, expected)
