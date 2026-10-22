"""Tester for loqs.core.recordables.measurementoutcomes"""

import pytest

from loqs.core.recordables import MeasurementOutcomes
from loqs.core.recordables.pauliframe import PauliFrame


class TestMeasurementOutcomes:

    def _check(self, mo, outcomes):
        assert set(mo.keys()) == set(outcomes.keys())
        for k, v in mo.items():
            assert v == outcomes[k]

    def test_init(self):
        outcomes = {"Q0": [0, 1], "Q1": 1}
        expected = {"Q0": [0, 1], "Q1": [1]}
        m = MeasurementOutcomes(outcomes)  # type: ignore
        self._check(m, expected)

        m2 = MeasurementOutcomes(m)
        self._check(m2, expected)

        with pytest.raises(TypeError):
            MeasurementOutcomes([0, 1, 0])  # type: ignore

    def test_inferred(self):
        qubits = ["Q0", "Q1", "Q2", "Q3"]
        outcomes = {k: [0, 1] for k in qubits}
        m = MeasurementOutcomes(outcomes)  # type: ignore
        pf = PauliFrame(qubits, "IXYZ")

        # X basis, Y and Z errors cause bitflips
        X_expected = {"Q0": [0, 1], "Q1": [0, 1], "Q2": [1, 0], "Q3": [1, 0]}
        m2 = m.get_inferred_outcomes(pf, "X")
        self._check(m2, X_expected)

        # Z basis, X and Y errors cause bitflips
        Z_expected = {"Q0": [0, 1], "Q1": [1, 0], "Q2": [1, 0], "Q3": [0, 1]}
        m3 = m.get_inferred_outcomes(pf, "Z")
        self._check(m3, Z_expected)

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_serialization(self, format, make_temp_path):
        outcomes = {
            "Q0": [0, 1, 0, 1],
            "Q1": [1],
            "aux_0": [1],
            "aux_1": [0, 1],
        }
        m = MeasurementOutcomes(outcomes)  # type: ignore

        with make_temp_path(suffix=f".{format}") as tmp_path:
            m.write(tmp_path)
            m2 = MeasurementOutcomes.read(tmp_path)

        assert isinstance(m2, MeasurementOutcomes)
        self._check(m2, outcomes)
