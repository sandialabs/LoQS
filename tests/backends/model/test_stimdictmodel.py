"""Tester for loqs.backends.model.stimdictmodel"""

import mock
import pytest
import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

try:
    import stim
    from loqs.backends.circuit import STIMPhysicalCircuit
    from loqs.backends.model.stimdictmodel import STIMDictNoiseModel
    from loqs.backends.reps import GateRep, InstrumentRep, RepTuple
    NO_STIM = False
except ImportError:
    NO_STIM = True


@pytest.mark.skipif(
    NO_STIM,
    reason="Skipping STIM backend tests due to failed import"
)
class TestSTIMDictNoiseModel:

    def test_init_basic(self):
        """Test basic initialization of STIMDictNoiseModel."""
        # Create empty gate and instrument dicts
        gate_dict = {}
        inst_dict = {}

        # Test initialization with empty dicts
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        assert model.name == "STIM gate dict"
        assert hasattr(model, 'gate_dict')
        assert hasattr(model, 'inst_dict')
        assert isinstance(model.gate_dict, dict)
        assert isinstance(model.inst_dict, dict)

    def test_init_with_gate_dict(self):
        """Test initialization with gate dictionary."""
        # Create a simple gate dict
        gate_dict = {
            "X": RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
            "Y": RepTuple("Y 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {}

        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Check that gate dict was properly initialized
        assert len(model.gate_dict) == 2
        assert "X" in model.gate_dict
        assert "Y" in model.gate_dict

    def test_init_with_instrument_dict(self):
        """Test initialization with instrument dictionary."""
        gate_dict = {}
        inst_dict = {
            "M": RepTuple((None, True), ("Q0",), InstrumentRep.ZBASIS_PROJECTION),
        }

        model = STIMDictNoiseModel(
            (gate_dict, inst_dict),
            instreps=[InstrumentRep.STIM_CIRCUIT_STR, InstrumentRep.ZBASIS_PROJECTION]
        )

        # Check that instrument dict was properly initialized
        assert len(model.inst_dict) == 1
        assert "M" in model.inst_dict

    def test_init_with_qubit_specific_labels(self):
        """Test initialization with qubit-specific labels."""
        gate_dict = {
            ("X", ("Q0",)): RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
            ("Y", ("Q1",)): RepTuple("Y 1", ("Q1",), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {}

        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Check that qubit-specific labels work
        assert len(model.gate_dict) == 2
        assert ("X", ("Q0",)) in model.gate_dict
        assert ("Y", ("Q1",)) in model.gate_dict

    def test_init_from_existing_model(self):
        """Test initialization from existing STIMDictNoiseModel."""
        # Create initial model
        gate_dict = {"X": RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR)}
        inst_dict = {}
        model1 = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create copy from existing model
        model2 = STIMDictNoiseModel(model1)

        # Should have same content
        assert len(model2.gate_dict) == len(model1.gate_dict)
        assert len(model2.inst_dict) == len(model1.inst_dict)

    def test_init_invalid_input(self):
        """Test initialization with invalid input."""
        # Test with invalid input type
        with pytest.raises(TypeError):
            STIMDictNoiseModel("invalid_input")

    def test_gaterep_conversion(self):
        """Test gate representation conversion."""
        # Test with string gate representations
        gate_dict = {
            "X": "X 0",  # String representation
            "Y": "Y 0",  # String representation
        }
        inst_dict = {}

        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Should be converted to RepTuples
        assert isinstance(model.gate_dict["X"], RepTuple)
        assert isinstance(model.gate_dict["Y"], RepTuple)
        assert model.gate_dict["X"].reptype == GateRep.STIM_CIRCUIT_STR

    def test_instrument_rep_conversion(self):
        """Test instrument representation conversion."""
        gate_dict = {}
        inst_dict = {
            "M": (None, True),  # Tuple representation for projection
        }

        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Should be converted to RepTuple
        assert isinstance(model.inst_dict["M"], RepTuple)
        assert model.inst_dict["M"].reptype == InstrumentRep.ZBASIS_PROJECTION

    def test_probabilistic_operations(self):
        """Test probabilistic operation handling."""
        gate_dict = {
            "NOISY_X": [("X 0", 0.9), ("Y 0", 0.1)],  # Probabilistic operation
        }
        inst_dict = {}

        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Should be converted to appropriate RepTuple
        assert isinstance(model.gate_dict["NOISY_X"], RepTuple)
        assert model.gate_dict["NOISY_X"].reptype == GateRep.PROBABILISTIC_STIM_OPERATIONS

    def test_pre_post_operations(self):
        """Test pre/post operation instrument handling."""
        gate_dict = {
            "IDLE": RepTuple("", ("Q0",), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {
            "NOISY_M": [
                0, True,
                RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
                RepTuple("Y 0", ("Q0",), GateRep.STIM_CIRCUIT_STR)
            ],
        }

        model = STIMDictNoiseModel(
            (gate_dict, inst_dict),
            instreps=[InstrumentRep.ZBASIS_PRE_POST_OPERATIONS]
        )

        # Should be converted to appropriate RepTuple
        assert isinstance(model.inst_dict["NOISY_M"], RepTuple)
        assert model.inst_dict["NOISY_M"].reptype == InstrumentRep.ZBASIS_PRE_POST_OPERATIONS

    def test_case_normalization(self):
        """Test that gate names are normalized to uppercase."""
        gate_dict = {
            "x": "x 0",  # lowercase
            "h": "h 0",  # lowercase
        }
        inst_dict = {}

        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Should be normalized to uppercase
        assert "X" in model.gate_dict
        assert "H" in model.gate_dict
        assert "x" not in model.gate_dict
        assert "h" not in model.gate_dict

    def test_get_reps_basic_circuit(self):
        """Test get_reps method with basic STIM circuit."""
        # Create a simple model with X gate
        gate_dict = {
            "X": RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create a simple STIM circuit
        circuit = STIMPhysicalCircuit("X 0", ["Q0"])

        # Get representations
        reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

        # Should return appropriate RepTuples
        assert len(reps) == 1
        assert isinstance(reps[0], RepTuple)

    def test_get_reps_complex_circuit(self):
        """Test get_reps method with more complex circuit."""
        # Create model with multiple gates
        gate_dict = {
            "X": RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
            "H": RepTuple("H 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
            "CNOT": RepTuple("CNOT 0 1", ("Q0", "Q1"), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create a multi-gate circuit
        circuit = STIMPhysicalCircuit("H 0\nCNOT 0 1\nX 0", ["Q0", "Q1"])

        # Get representations
        reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

        # Should return multiple RepTuples
        assert len(reps) > 1
        for rep in reps:
            assert isinstance(rep, RepTuple)

    def test_get_reps_with_instruments(self):
        """Test get_reps method with instruments."""
        gate_dict = {
            "X": RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {
            "M": RepTuple((None, True), ("Q0",), InstrumentRep.ZBASIS_PROJECTION),
        }
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create circuit with measurement
        circuit = STIMPhysicalCircuit("X 0\nM 0", ["Q0"])

        # Get representations
        reps = model.get_reps(
            circuit,
            [GateRep.STIM_CIRCUIT_STR],
            [InstrumentRep.ZBASIS_PROJECTION]
        )

        # Should include both gate and instrument reps
        assert len(reps) == 2

    def test_get_reps_unsupported_command(self):
        """Test get_reps with unsupported STIM commands."""
        gate_dict = {}
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create circuit with unsupported command
        circuit = STIMPhysicalCircuit("DEPOLARIZE1 0.1 0", ["Q0"])

        # This should still work but may produce warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

            # Should still return some reps (possibly as dummy/comment reps)
            assert len(reps) >= 1

    def test_warnings_for_noise_channels(self):
        """Test that warnings are issued for noise channels."""
        gate_dict = {}
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create circuit with noise channel
        circuit = STIMPhysicalCircuit("X_ERROR(0.1) 0", ["Q0"])

        # This should produce a warning about noise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

            # Should have issued a warning
            assert len(w) > 0
            assert "Noise channel" in str(w[0].message)

    def test_warnings_for_measure_noise(self):
        """Test that warnings are issued for measurement noise."""
        gate_dict = {}
        inst_dict = {
            "M": RepTuple((None, True), ("Q0",), InstrumentRep.ZBASIS_PROJECTION),
        }
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create circuit with noisy measurement
        circuit = STIMPhysicalCircuit("M(0.1) 0", ["Q0"])

        # This should produce a warning about measurement noise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

            # Should have issued a warning
            assert len(w) > 0
            assert "Measure noise" in str(w[0].message)

    def test_qubit_label_mapping(self):
        """Test qubit label mapping in get_reps."""
        gate_dict = {
            "X": RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create circuit with different qubit labels
        circuit = STIMPhysicalCircuit("X 0", ["A0"])

        # Get representations
        reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

        # Should map qubit labels correctly
        assert len(reps) == 1
        assert isinstance(reps[0], RepTuple)

    def test_common_command_combining(self):
        """Test combining of common commands in get_reps."""
        gate_dict = {
            "X": RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create circuit with multiple X gates that could be combined
        circuit = STIMPhysicalCircuit("X 0\nX 1", ["Q0", "Q1"])

        # Get representations
        reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

        # Should handle multiple instances correctly
        assert len(reps) >= 1

    def test_gaterep_validation(self):
        """Test validation of gatereps parameter."""
        gate_dict = {
            "X": RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {}

        # This should work with valid gatereps
        model = STIMDictNoiseModel(
            (gate_dict, inst_dict),
            gatereps=[GateRep.STIM_CIRCUIT_STR]
        )

        assert model._gatereps == [GateRep.STIM_CIRCUIT_STR]

    def test_instrep_validation(self):
        """Test validation of instreps parameter."""
        gate_dict = {}
        inst_dict = {
            "M": RepTuple((None, True), ("Q0",), InstrumentRep.ZBASIS_PROJECTION),
        }

        # This should work with valid instreps
        model = STIMDictNoiseModel(
            (gate_dict, inst_dict),
            instreps=[InstrumentRep.ZBASIS_PROJECTION]
        )

        assert model._instreps == [InstrumentRep.ZBASIS_PROJECTION]

    def test_gaterep_array_cast_warning(self):
        """Test warning for unused gaterep_array_cast_rep parameter."""
        gate_dict = {}
        inst_dict = {}

        # This should produce a warning about unused parameter
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = STIMDictNoiseModel(
                (gate_dict, inst_dict),
                gaterep_array_cast_rep=GateRep.QSIM_SUPEROPERATOR
            )

            # Should have issued a warning
            assert len(w) > 0
            assert "gaterep_array_cast_rep" in str(w[0].message)

    def test_empty_circuit(self):
        """Test get_reps with empty circuit."""
        gate_dict = {}
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create empty circuit
        circuit = STIMPhysicalCircuit("", ["Q0"])

        # Get representations
        reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

        # Should handle empty circuit gracefully
        assert isinstance(reps, list)

    def test_circuit_with_comments(self):
        """Test get_reps with circuit containing comments."""
        gate_dict = {}
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create circuit with comment lines
        circuit = STIMPhysicalCircuit("# This is a comment\nX 0", ["Q0"])

        # Get representations
        reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

        # Should handle comments appropriately
        assert len(reps) >= 1


class TestSTIMDictNoiseModelFailedImport:
    """Test behavior when STIM is not available."""

    def test_failed_import(self):
        """Test that STIMDictNoiseModel handles missing STIM gracefully."""
        with mock.patch.dict('sys.modules', {
                'stim': None,
            }):
            # This should raise ImportError when STIM is not available
            with pytest.raises(ImportError):
                import importlib
                import sys

                mod = sys.modules.get('loqs.backends.model.stimdictmodel')
                if mod:
                    importlib.reload(mod)
