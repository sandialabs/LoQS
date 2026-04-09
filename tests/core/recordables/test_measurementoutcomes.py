"""Tester for loqs.core.recordables.measurementoutcomes"""

import os
from tempfile import NamedTemporaryFile
import json
import pytest

from loqs.core.recordables import MeasurementOutcomes
from loqs.core.recordables.pauliframe import PauliFrame


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
        m2 = m.get_inferred_outcomes(pf, "X")
        self._check(m2, X_expected)

        # Z basis, X and Y errors cause bitflips
        Z_expected = {"Q0": [0, 1], "Q1": [1,0], "Q2": [1,0], "Q3": [0,1]}
        m3 = m.get_inferred_outcomes(pf, "Z")
        self._check(m3, Z_expected)
    
    def test_serialization(self):
        outcomes = {"Q0": [0, 1], "Q1": 1}
        expected = {"Q0": [0, 1], "Q1": [1]}
        m = MeasurementOutcomes(outcomes)

        with NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix='.json') as tmp:
            m.write(tmp.name)
            tmp_path = tmp.name

        try:
            m2 = MeasurementOutcomes.read(tmp_path)
            self._check(m2, expected)
        finally:
            os.unlink(tmp_path)

    def test_hdf5_serialization(self):
        """Test MeasurementOutcomes HDF5 serialization."""
        outcomes = {"Q0": [0, 1], "Q1": 1}
        expected = {"Q0": [0, 1], "Q1": [1]}
        m = MeasurementOutcomes(outcomes)

        # Test file serialization with HDF5
        import h5py
        with NamedTemporaryFile(suffix='.h5') as tempf:
            with h5py.File(tempf.name, 'w') as h5f:
                m.dump(h5f)
            with h5py.File(tempf.name, 'r') as h5f:
                m2 = MeasurementOutcomes.load(h5f)
            self._check(m2, expected)

        # Test file serialization with .h5 extension
        with NamedTemporaryFile(suffix='.h5') as tempf:
            m.write(tempf.name)
            m2 = MeasurementOutcomes.read(tempf.name)
            self._check(m2, expected)

        # Test file serialization with .hdf5 extension
        with NamedTemporaryFile(suffix='.hdf5') as tempf:
            m.write(tempf.name)
            m2 = MeasurementOutcomes.read(tempf.name)
            self._check(m2, expected)

    def test_hdf5_serialization_complex(self):
        """Test MeasurementOutcomes HDF5 serialization with complex data."""
        outcomes = {
            "Q0": [0, 1, 0, 1],
            "Q1": [1, 0, 1, 0],
            "Q2": [0, 0, 1, 1],
            "aux_0": [1],
            "aux_1": [0, 1]
        }
        m = MeasurementOutcomes(outcomes)

        # Test HDF5 serialization
        import h5py
        with NamedTemporaryFile(suffix='.h5') as tempf:
            with h5py.File(tempf.name, 'w') as h5f:
                m.dump(h5f)
            with h5py.File(tempf.name, 'r') as h5f:
                m2 = MeasurementOutcomes.load(h5f)

        # Check all outcomes are preserved
        for qubit in outcomes:
            assert m2[qubit] == outcomes[qubit]

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_measurementoutcomes_serialization_parameterized(self, format):
        """Test MeasurementOutcomes serialization with both JSON and HDF5 formats."""
        outcomes = {"Q0": [0, 1], "Q1": 1}
        expected = {"Q0": [0, 1], "Q1": [1]}
        m = MeasurementOutcomes(outcomes)

        # Test bytes serialization
        if format == "json":
            with NamedTemporaryFile(mode="w+", suffix=".json") as tempf:
                m.dump(tempf.file)
                tempf.seek(0)
                m2 = MeasurementOutcomes.load(tempf.file)
        else:  # hdf5
            import h5py
            with NamedTemporaryFile(suffix=".h5") as tempf:
                with h5py.File(tempf.name, "w") as h5f:
                    m.dump(h5f)
                with h5py.File(tempf.name, "r") as h5f:
                    m2 = MeasurementOutcomes.load(h5f)
        self._check(m2, expected)

        # Test file serialization
        with NamedTemporaryFile(suffix=f'.{format}') as tempf:
            m.write(tempf.name)
            m2 = MeasurementOutcomes.read(tempf.name)
            self._check(m2, expected)

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_measurementoutcomes_serialization_complex_parameterized(self, format):
        """Test MeasurementOutcomes serialization with complex data using both formats."""
        outcomes = {
            "Q0": [0, 1, 0, 1],
            "Q1": [1, 0, 1, 0],
            "Q2": [0, 0, 1, 1],
            "aux_0": [1],
            "aux_1": [0, 1]
        }
        m = MeasurementOutcomes(outcomes)

        # Test serialization
        if format == "json":
            with NamedTemporaryFile(mode="w+", suffix=".json") as tempf:
                m.dump(tempf.file)
                tempf.seek(0)
                m2 = MeasurementOutcomes.load(tempf.file)
        else:  # hdf5
            import h5py
            with NamedTemporaryFile(suffix=".h5") as tempf:
                with h5py.File(tempf.name, "w") as h5f:
                    m.dump(h5f)
                with h5py.File(tempf.name, "r") as h5f:
                    m2 = MeasurementOutcomes.load(h5f)

        # Check all outcomes are preserved
        for qubit in outcomes:
            assert m2[qubit] == outcomes[qubit]
