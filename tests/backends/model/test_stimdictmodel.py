"""Tester for loqs.backends.model.stimdictmodel"""

import pytest
import warnings

try:
    import stim
    from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit
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
        """Test initialization with gate dictionary.

        NOTE: This test has some overlap with test_gaterep_conversion.
        """
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
        """Test initialization with instrument dictionary.

        NOTE: This test has some overlap with test_instrument_rep_conversion.
        """
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

    def test_gaterep_conversion(self):
        """Test gate representation conversion.

        NOTE: This test has some overlap with test_init_with_gate_dict.
        """
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
        """Test instrument representation conversion.

        NOTE: This test has some overlap with test_init_with_instrument_dict.
        """
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
        """Test get_reps method with basic STIM circuit.

        NOTE: This is the simplest case, more complex cases are covered by other tests.
        """
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
        model = STIMDictNoiseModel(
            (gate_dict, inst_dict),
            instreps=[InstrumentRep.STIM_CIRCUIT_STR, InstrumentRep.ZBASIS_PROJECTION]
        )

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
            # Should still return some reps (possibly as dummy/comment reps)
            assert len(reps) >= 1
        return

    def test_warnings_for_measure_noise(self):
        """Test that warnings are issued for measurement noise."""
        gate_dict = {}
        inst_dict = {
            "M": RepTuple((None, True), ("Q0",), InstrumentRep.ZBASIS_PROJECTION),
        }
        model = STIMDictNoiseModel(
            (gate_dict, inst_dict),
            instreps=[InstrumentRep.STIM_CIRCUIT_STR, InstrumentRep.ZBASIS_PROJECTION]
        )

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
        """Test validation of gatereps parameter.

        NOTE: This follows the same pattern as test_instrep_validation.
        """
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
        """Test validation of instreps parameter.

        NOTE: This follows the same pattern as test_gaterep_validation.
        """
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
            _ = STIMDictNoiseModel(
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
        gate_dict = {
            "X": RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create circuit with comment lines
        circuit = STIMPhysicalCircuit("# This is a comment\nX 0", ["Q0"])

        # Get representations
        reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

        # Should handle comments appropriately
        assert len(reps) >= 1

    def test_tuple_key_aliasing(self):
        """Test command aliasing with tuple keys."""
        # Use 'CNOT' which should be aliased to 'CX' according to stim_command_aliases
        gate_dict = {
            ("CNOT", ("Q0", "Q1",)): RepTuple("CNOT 0 1", ("Q0", "Q1"), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Should have both original and aliased keys after aliasing
        has_original = ("CNOT", ("Q0", "Q1")) in model.gate_dict
        has_aliased = ("CX", ("Q0", "Q1")) in model.gate_dict
        assert has_original or has_aliased
        assert len(model.gate_dict) >= 1

    def test_string_instrument_representation(self):
        """Test instrument initialization with string representation."""
        gate_dict = {}
        inst_dict = {
            "M": "M 0",  # String representation for instrument
        }
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Should be converted to RepTuple with STIM_CIRCUIT_STR type
        assert "M" in model.inst_dict
        assert isinstance(model.inst_dict["M"], RepTuple)
        assert model.inst_dict["M"].reptype == InstrumentRep.STIM_CIRCUIT_STR

    def test_complex_command_combining(self):
        """Test combining multiple instances of same command."""
        gate_dict = {
            "X": RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create circuit with multiple X gates on different qubits
        circuit = STIMPhysicalCircuit("X 0\nX 1\nX 2", ["Q0", "Q1", "Q2"])

        # Get representations
        reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

        # Should combine multiple X commands
        assert len(reps) >= 1
        # Check that we have combined commands by looking for multiple qubits in rep
        combined_reps = [r for r in reps if len(r.qubits) > 1]
        assert len(combined_reps) > 0

    def test_non_common_command_handling(self):
        """Test handling of commands that should not be combined."""
        gate_dict = {
            "X": RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
            "Y": RepTuple("Y 0", ("Q1",), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create circuit with different commands that shouldn't be combined
        circuit = STIMPhysicalCircuit("X 0\nY 1", ["Q0", "Q1"])

        # Get representations
        reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

        # Should have individual commands, not combined
        assert len(reps) >= 2
        command_types = [type(r.rep) for r in reps]
        # Should have separate RepTuples for X and Y
        assert any("X" in str(r.rep) for r in reps)
        assert any("Y" in str(r.rep) for r in reps)

    def test_multiple_same_command_combining(self):
        """Test combining when same command appears multiple times sequentially."""
        # Create a gate dict with a specific qubit label to trigger the combining logic
        gate_dict = {
            "X": RepTuple("X", ("Q0",), GateRep.STIM_CIRCUIT_STR),  # Note: no qubit index in rep
        }
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create circuit with multiple X gates that should trigger the 'command in common' path
        circuit = STIMPhysicalCircuit("X 0\nX 1\nX 2", ["Q0", "Q1", "Q2"])

        # Get representations
        reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

        # Should have combined the X commands
        assert len(reps) >= 1
        # Look for a combined rep with multiple qubits
        combined_reps = [r for r in reps if len(r.qubits) > 1]
        assert len(combined_reps) > 0, "Should have combined multiple X commands"

    def test_individual_command_no_combining(self):
        """Test case where commands are processed individually without combining."""
        # Use qubit-specific gate definitions to prevent combining
        gate_dict = {
            ("X", ("Q0",)): RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
            ("X", ("Q1",)): RepTuple("X 1", ("Q1",), GateRep.STIM_CIRCUIT_STR),
            ("X", ("Q2",)): RepTuple("X 2", ("Q2",), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create circuit with X gates on different qubits
        circuit = STIMPhysicalCircuit("X 0\nX 1\nX 2", ["Q0", "Q1", "Q2"])

        # Get representations
        reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

        # Should have individual reps since each has specific qubit definitions
        assert len(reps) >= 3
        # Each rep should be for individual qubits (not combined)
        individual_reps = [r for r in reps if len(r.qubits) == 1]
        assert len(individual_reps) >= 3

    def test_existing_command_in_common_dict(self):
        """Test the specific case where command already exists in common dict."""
        # Create a gate with a template that can be extended
        gate_dict = {
            "X": RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
        }
        inst_dict = {}
        model = STIMDictNoiseModel((gate_dict, inst_dict))

        # Create circuit designed to trigger the 'command in common' condition
        # Use multiple X gates that will be processed sequentially and combined
        circuit = STIMPhysicalCircuit("X 0\nX 1\nX 2\nX 3", ["Q0", "Q1", "Q2", "Q3"])

        # Get representations - this should trigger the missing lines
        reps = model.get_reps(circuit, [GateRep.STIM_CIRCUIT_STR], [InstrumentRep.ZBASIS_PROJECTION])

        # Should have combined representations
        assert len(reps) >= 1
        # Look for evidence of the combining logic being triggered
        combined_reps = [r for r in reps if "X" in str(r.rep) and len(r.qubits) > 1]
        assert len(combined_reps) > 0, "Should have combined X commands"

        # Verify that at least one combined rep has multiple qubits mentioned
        for rep in combined_reps:
            if len(rep.qubits) > 1:
                # This indicates the combining logic was triggered
                break
        else:
            assert False, "No combined reps with multiple qubits found"
