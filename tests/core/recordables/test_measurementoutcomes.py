"""Tester for loqs.core.recordables.measurementoutcomes"""

import pytest
import h5py

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
        m = MeasurementOutcomes(outcomes) # type: ignore
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
        m = MeasurementOutcomes(outcomes) # type: ignore
        pf = PauliFrame(qubits, "IXYZ")

        # X basis, Y and Z errors cause bitflips
        X_expected = {"Q0": [0, 1], "Q1": [0,1], "Q2": [1,0], "Q3": [1,0]}
        m2 = m.get_inferred_outcomes(pf, "X")
        self._check(m2, X_expected)

        # Z basis, X and Y errors cause bitflips
        Z_expected = {"Q0": [0, 1], "Q1": [1,0], "Q2": [1,0], "Q3": [0,1]}
        m3 = m.get_inferred_outcomes(pf, "Z")
        self._check(m3, Z_expected)
    
    def test_serialization(self, make_temp_path):
        outcomes = {"Q0": [0, 1], "Q1": 1}
        expected = {"Q0": [0, 1], "Q1": [1]}
        m = MeasurementOutcomes(outcomes) # type: ignore

        with make_temp_path(suffix='.json') as tmp_path:
            m.write(tmp_path)
            m2 = MeasurementOutcomes.read(tmp_path)
            self._check(m2, expected)

    def test_hdf5_serialization(self, make_temp_path):
        """Test MeasurementOutcomes HDF5 serialization."""
        outcomes = {"Q0": [0, 1], "Q1": 1}
        expected = {"Q0": [0, 1], "Q1": [1]}
        m = MeasurementOutcomes(outcomes) # type: ignore

        # Test file serialization with HDF5
        with make_temp_path(suffix='.h5') as tempf_path:
            with h5py.File(tempf_path, 'w') as h5f:
                m.dump(h5f)
            with h5py.File(tempf_path, 'r') as h5f:
                m2 = MeasurementOutcomes.load(h5f)
            self._check(m2, expected)

        with make_temp_path(suffix='.h5') as tempf_path:
            m.write(tempf_path)
            m2 = MeasurementOutcomes.read(tempf_path)
            self._check(m2, expected)

        with make_temp_path(suffix='.hdf5') as tempf_path:
            m.write(tempf_path)
            m2 = MeasurementOutcomes.read(tempf_path)
            self._check(m2, expected)

    def test_hdf5_serialization_complex(self, make_temp_path):
        """Test MeasurementOutcomes HDF5 serialization with complex data."""
        outcomes = {
            "Q0": [0, 1, 0, 1],
            "Q1": [1, 0, 1, 0],
            "Q2": [0, 0, 1, 1],
            "aux_0": [1],
            "aux_1": [0, 1]
        }
        m = MeasurementOutcomes(outcomes) # type: ignore

        # Test HDF5 serialization
        with make_temp_path(suffix='.h5') as tempf_path:
            with h5py.File(tempf_path, 'w') as h5f:
                m.dump(h5f)
            with h5py.File(tempf_path, 'r') as h5f:
                m2 = MeasurementOutcomes.load(h5f)
            
            assert isinstance(m2, MeasurementOutcomes)

            # Check all outcomes are preserved
            for qubit in outcomes:
                assert m2[qubit] == outcomes[qubit]

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_measurementoutcomes_serialization_parameterized(self, format, make_temp_path):
        """Test MeasurementOutcomes serialization with both JSON and HDF5 formats."""
        outcomes = {"Q0": [0, 1], "Q1": 1}
        expected = {"Q0": [0, 1], "Q1": [1]}
        m = MeasurementOutcomes(outcomes) # type: ignore

        # Test bytes serialization
        if format == "json":
            with make_temp_path(suffix=".json") as tempf_path:
                with open(tempf_path, "w+") as tempf:
                    m.dump(tempf)
                    tempf.seek(0)
                    m2 = MeasurementOutcomes.load(tempf)

        else:  # hdf5
            with make_temp_path(suffix=".h5") as tempf_path:
                with h5py.File(tempf_path, "w") as h5f:
                    m.dump(h5f)
                with h5py.File(tempf_path, "r") as h5f:
                    m2 = MeasurementOutcomes.load(h5f)

        self._check(m2, expected)

        # Test file serialization
        with make_temp_path(suffix=f'.{format}') as tempf_path:
            m.write(tempf_path)
            m2 = MeasurementOutcomes.read(tempf_path)
            self._check(m2, expected)

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_measurementoutcomes_serialization_complex_parameterized(self, format, make_temp_path):
        """Test MeasurementOutcomes serialization with complex data using both formats."""
        outcomes = {
            "Q0": [0, 1, 0, 1],
            "Q1": [1, 0, 1, 0],
            "Q2": [0, 0, 1, 1],
            "aux_0": [1],
            "aux_1": [0, 1]
        }
        m = MeasurementOutcomes(outcomes) # type: ignore

        # Test serialization
        if format == "json":
            with make_temp_path(suffix=".json") as tempf_path:
                with open(tempf_path, "w+") as tempf:
                    m.dump(tempf)
                    tempf.seek(0)
                    m2 = MeasurementOutcomes.load(tempf)

        else:  # hdf5
            with make_temp_path(suffix=".h5") as tempf_path:
                with h5py.File(tempf_path, "w") as h5f:
                    m.dump(h5f)
                with h5py.File(tempf_path, "r") as h5f:
                    m2 = MeasurementOutcomes.load(h5f)
        
        assert isinstance(m2, MeasurementOutcomes)
        for qubit in outcomes:
            assert m2[qubit] == outcomes[qubit]
