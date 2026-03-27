"""Tester for loqs.backends.circuit.stimcircuit"""

import pytest
import warnings

try:
    import stim
    from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit
    from loqs.backends import BasePhysicalCircuit
    NO_STIM = False
except ImportError:
    NO_STIM = True


@pytest.mark.skipif(
    NO_STIM,
    reason="Skipping STIM backend tests due to failed import"
)
class TestSTIMPhysicalCircuit:

    def test_init_basic(self):
        """Test basic initialization with string circuit."""
        # Simple X gate circuit
        circuit = STIMPhysicalCircuit("X 0", ["Q0"])
        
        assert circuit.name == "STIM"
        assert str(circuit.circuit) == "X 0"
        assert circuit.qubit_labels == ["Q0"]
        assert circuit.depth == 1  # No TICK instructions, so depth = num_ticks + 1

    def test_init_with_tick(self):
        """Test initialization with TICK instructions."""
        circuit_str = "X 0\nTICK\nY 0"
        circuit = STIMPhysicalCircuit(circuit_str, ["Q0"])
        
        assert circuit.depth == 2  # One TICK instruction
        assert "TICK" in str(circuit.circuit)

    def test_init_from_stim_circuit(self):
        """Test initialization from stim.Circuit object."""
        arg_str = "H 0\nCNOT 0 1"
        stim_circuit = stim.Circuit(arg_str)
        circuit = STIMPhysicalCircuit(stim_circuit)
        expect_str = STIMPhysicalCircuit.substitute_command_aliases(arg_str)
        assert str(circuit.circuit) == expect_str
        assert circuit.qubit_labels == [0, 1]  # Default labels

    def test_init_with_custom_labels(self):
        """Test initialization with custom qubit labels."""
        circuit = STIMPhysicalCircuit("X 0\nY 1", ["QA", "QB"])
        
        assert circuit.qubit_labels == ["QA", "QB"]
        assert circuit.circuit.num_qubits == 2

    def test_copy(self):
        """Test circuit copying."""
        original = STIMPhysicalCircuit("X 0\nY 1", ["Q0", "Q1"])
        copied = original.copy()
        
        assert str(copied.circuit) == str(original.circuit)
        assert copied.qubit_labels == original.qubit_labels
        assert copied is not original  # Different objects

    def test_str_representation(self):
        """Test string representation."""
        circuit = STIMPhysicalCircuit("H 0\nX 0", ["Q0"])
        repr_str = str(circuit)
        
        assert "Physical STIM circuit" in repr_str
        assert "Q0" in repr_str
        assert "H 0" in repr_str

    def test_delete_qubits_inplace(self):
        """Test deleting qubits from circuit."""
        circuit = STIMPhysicalCircuit("X 0\nY 1\nZ 2", ["Q0", "Q1", "Q2"])
        
        # Delete Q1
        circuit.delete_qubits_inplace(["Q1"])
        
        assert "Q1" not in circuit.qubit_labels
        assert "Y 1" not in str(circuit.circuit)  # Y gate on Q1 should be removed
        assert "X 0" in str(circuit.circuit)  # X gate on Q0 should remain
        assert "Z 2" in str(circuit.circuit)  # Z gate on Q2 should remain

    def test_get_possible_discrete_error_locations(self):
        """Test getting possible error locations."""
        circuit = STIMPhysicalCircuit("X 0\nTICK\nCNOT 0 1\nTICK\nY 1", ["Q0", "Q1"])
        
        locations = circuit.get_possible_discrete_error_locations()
        
        # Should have locations for each layer
        assert len(locations) > 0
        
        # Check that locations are in expected format (layer_idx, qubit_idx)
        for layer_idx, qubit_info in locations:
            assert isinstance(layer_idx, int)
            if isinstance(qubit_info, tuple):
                # Two-qubit gate
                assert len(qubit_info) == 2
                assert all(isinstance(q, int) for q in qubit_info)
            else:
                # Single-qubit gate
                assert isinstance(qubit_info, int)

    def test_get_possible_discrete_error_locations_post_twoq(self):
        """Test getting error locations with post_twoq_gates=True."""
        circuit = STIMPhysicalCircuit("X 0\nTICK\nCNOT 0 1\nTICK\nY 1", ["Q0", "Q1"])
        
        locations = circuit.get_possible_discrete_error_locations(post_twoq_gates=True)
        
        # Should include two-qubit gate locations
        twoq_locations = [loc for loc in locations if isinstance(loc[1], tuple)]
        assert len(twoq_locations) > 0

    def test_insert_inplace(self):
        """Test inserting another circuit."""
        main_circuit = STIMPhysicalCircuit("X 0\nTICK\nY 0", ["Q0"])
        insert_circuit = STIMPhysicalCircuit("Z 0", ["Q0"])
        
        # Insert at position 1 (after first layer)
        main_circuit.insert_inplace(insert_circuit, 1)
        
        circuit_str = str(main_circuit.circuit)
        assert "X 0" in circuit_str
        assert "Z 0" in circuit_str
        assert "Y 0" in circuit_str

    def test_insert_inplace_append(self):
        """Test inserting at end of circuit."""
        main_circuit = STIMPhysicalCircuit("X 0\nTICK\nY 0", ["Q0"])
        insert_circuit = STIMPhysicalCircuit("Z 0", ["Q0"])
        
        # Insert at position -1 (append)
        main_circuit.insert_inplace(insert_circuit, -1)
        
        circuit_str = str(main_circuit.circuit)
        # Z should come after Y
        y_pos = circuit_str.find("Y 0")
        z_pos = circuit_str.find("Z 0")
        assert z_pos > y_pos

    def test_map_qubit_labels_inplace(self):
        """Test mapping qubit labels."""
        circuit = STIMPhysicalCircuit("X 0\nY 1", ["Q0", "Q1"])
        
        # Map Q0 -> QA, Q1 -> QB
        circuit.map_qubit_labels_inplace({"Q0": "QA", "Q1": "QB"})
        
        assert circuit.qubit_labels == ["QA", "QB"]
        # Circuit operations should still reference indices 0 and 1
        assert "X 0" in str(circuit.circuit)
        assert "Y 1" in str(circuit.circuit)

    def test_merge_inplace(self):
        """Test merging another circuit."""

        main_circuit = STIMPhysicalCircuit("X 0\nTICK\nY 0", ["Q0"])

        with pytest.raises(ValueError) as e:
            merge_circuit = STIMPhysicalCircuit("Z 0", ["Q0"])
            main_circuit.merge_inplace(merge_circuit, 0)

        assert 'Layer 0 of the candidate merge has ill-posed behavior' in str(e)

        merge_circuit = STIMPhysicalCircuit("Z 1", ["Q1"])
        main_circuit.merge_inplace(merge_circuit, 0)
        circuit_str = str(main_circuit.circuit)
        first_layer_end = circuit_str.find("TICK")
        first_layer = circuit_str[:first_layer_end]
        assert "X 0" in first_layer
        assert "Z 1" in first_layer
        return

    def test_pad_single_qubit_idles_by_duration_inplace(self):
        """Test padding with idle operations."""
        circuit = STIMPhysicalCircuit("X 0\nTICK\nY 1", ["Q0", "Q1"])
        
        idle_names = {1: "I"}
        durations = {"X": 1, "Y": 1}
        
        circuit.pad_single_qubit_idles_by_duration_inplace(
            idle_names, durations, default_duration=1
        )
        
        circuit_str = str(circuit.circuit)
        # Should have idle operations added
        assert "I" in circuit_str

    def test_set_qubit_labels_inplace(self):
        """Test setting qubit labels."""
        circuit = STIMPhysicalCircuit("X 0\nY 1", ["Q0", "Q1"])
        
        circuit.set_qubit_labels_inplace(["QA", "QB"])
        
        assert circuit.qubit_labels == ["QA", "QB"]

    def test_tick_warning(self):
        """Test TICK warning for multi-layer circuits."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Circuit without TICK should warn
            STIMPhysicalCircuit("X 0\nY 0", ["Q0"])
            
            assert len(w) == 1
            assert "No TICK instructions" in str(w[0].message)

    def test_no_tick_warning_with_suppress(self):
        """Test that TICK warning can be suppressed."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Circuit without TICK but with suppress_tick_warning=True should not warn
            STIMPhysicalCircuit("X 0\nY 0", ["Q0"], suppress_tick_warning=True)
            
            assert len(w) == 0

    def test_serialization(self):
        """Test circuit serialization and deserialization."""
        original = STIMPhysicalCircuit("X 0\nY 1", ["Q0", "Q1"])
        
        # Serialize
        serialized = original._serialize_circuit()
        assert isinstance(serialized, str)
        assert "X 0" in serialized
        assert "Y 1" in serialized
        
        # Deserialize
        deserialized_circuit = STIMPhysicalCircuit._deserialize_circuit(serialized)
        assert str(deserialized_circuit) == serialized

    def test_unroll_repeats(self):
        """Test unrolling REPEAT blocks."""
        circuit = STIMPhysicalCircuit("REPEAT 2 {\nX 0\nY 1\n}\nZ 0", ["Q0", "Q1"])
        
        unrolled = circuit._unroll_repeats()
        
        # Should have X and Y repeated twice
        assert unrolled.count("X 0") == 2
        assert unrolled.count("Y 1") == 2
        assert "Z 0" in unrolled
        assert "REPEAT" not in unrolled

    def test_cast_method(self):
        """Test the cast class method."""
        # Test casting from string
        circuit1 = STIMPhysicalCircuit.cast("X 0")
        assert isinstance(circuit1, STIMPhysicalCircuit)
        
        # Test casting from STIMPhysicalCircuit
        original = STIMPhysicalCircuit("Y 0", ["Q0"])
        circuit2 = STIMPhysicalCircuit.cast(original)
        assert isinstance(circuit2, STIMPhysicalCircuit)
        assert str(circuit2.circuit) == str(original.circuit)

    def test_gate_classification(self):
        """Test gate classification lists."""
        # Check that expected gates are in the right categories
        assert "X" in STIMPhysicalCircuit._stim_oneq_gates
        assert "CNOT" in STIMPhysicalCircuit._stim_twoq_gates
        assert "M" in STIMPhysicalCircuit._stim_measure_reset_gates
        assert "X_ERROR" in STIMPhysicalCircuit._stim_noise_channels
        assert "REPEAT" in STIMPhysicalCircuit._stim_annotations

    def test_empty_circuit(self):
        """Test empty circuit handling."""
        circuit = STIMPhysicalCircuit("", [])
        
        assert circuit.circuit.num_qubits == 0
        assert circuit.qubit_labels == []
        assert circuit.depth == 1  # Even empty circuits have depth 1

    def test_complex_circuit_operations(self):
        """Test a more complex circuit with multiple operations."""
        circuit_str = """H 0
TICK
CX 0 1
Y 1
TICK
M 0
TICK
X 1"""
        
        circuit = STIMPhysicalCircuit(circuit_str, ["Q0", "Q1"])
        
        assert circuit.depth == 4  # 3 TICKs + 1
        assert circuit.circuit.num_qubits == 2
        
        # Test that we can get error locations
        locations = circuit.get_possible_discrete_error_locations()
        assert len(locations) == 5  # H, CX+Y, M, X

    def test_qubit_label_validation(self):
        """Test qubit label validation."""
        circuit = STIMPhysicalCircuit("X 0\nY 1", ["Q0", "Q1"])
        
        # Accessing qubit_labels should work
        labels = circuit.qubit_labels
        assert labels == ["Q0", "Q1"]
        
        # Test with mismatched label count
        with pytest.raises(AssertionError):
            circuit = STIMPhysicalCircuit("X 0\nY 1\nZ 2", ["Q0", "Q1"])  # Only 2 labels for 3 qubits
            _ = circuit.qubit_labels

    def test_stim_command_aliases(self):
        """Test STIM command aliases."""
        assert "CNOT" in STIMPhysicalCircuit.stim_command_aliases
        assert STIMPhysicalCircuit.stim_command_aliases["CNOT"] == "CX"

    def test_circuit_with_measurements(self):
        """Test circuit with measurement operations."""
        circuit = STIMPhysicalCircuit("X 0\nM 0\nTICK\nY 0", ["Q0"])
        
        assert "M 0" in str(circuit.circuit)
        
        # Should be able to get error locations including measurements
        locations = circuit.get_possible_discrete_error_locations()
        assert any(isinstance(loc[1], int) and loc[0] == 1 for loc in locations)  # M gate

    def test_circuit_with_noise(self):
        """Test circuit with noise operations."""
        circuit = STIMPhysicalCircuit("X 0\nX_ERROR(0.1) 0", ["Q0"])
        
        assert "X_ERROR" in str(circuit.circuit)
        
        # Noise operations should be included in gate list
        assert "X_ERROR" in STIMPhysicalCircuit._stim_noise_channels

    def test_multiple_qubit_operations(self):
        """Test circuit with various multi-qubit operations."""
        circuit_str = """H 0
H 1
TICK
SWAP 0 1
ISWAP 0 1
TICK
CZ 0 1"""
        
        circuit = STIMPhysicalCircuit(circuit_str, ["Q0", "Q1"])
        
        # Test two-qubit gate classification
        assert "SWAP" in STIMPhysicalCircuit._stim_twoq_gates
        assert "ISWAP" in STIMPhysicalCircuit._stim_twoq_gates
        assert "CZ" in STIMPhysicalCircuit._stim_twoq_gates
        
        # Test error location detection for two-qubit gates
        locations = circuit.get_possible_discrete_error_locations()
        twoq_locs = [loc for loc in locations if isinstance(loc[1], tuple)]
        assert len(twoq_locs) == 3  # SWAP, ISWAP, CZ

    def test_circuit_copy_independence(self):
        """Test that copied circuits are independent."""
        original = STIMPhysicalCircuit("X 0", ["Q0"])
        copied = original.copy()
        
        # Modify original
        original.circuit.append("Y", [0])
        
        # Copied should not be affected
        assert "Y" not in str(copied.circuit)
        assert "Y" in str(original.circuit)

    def test_insert_with_repeats(self):
        """Test inserting into circuit with REPEAT blocks."""
        main_circuit = STIMPhysicalCircuit("REPEAT 2 {X 0}\nTICK\nY 0", ["Q0"])
        insert_circuit = STIMPhysicalCircuit("Z 0", ["Q0"])
        
        # Insert should unroll repeats first
        main_circuit.insert_inplace(insert_circuit, 1)
        
        circuit_str = str(main_circuit.circuit)
        # Should have X twice (unrolled), then Z, then Y
        x_count = circuit_str.count("X 0")
        assert x_count == 2
        assert "Z 0" in circuit_str
        assert "Y 0" in circuit_str

    def test_merge_with_different_qubit_labels(self):
        """Test merging circuits with different qubit labels."""
        main_circuit = STIMPhysicalCircuit("X 0", ["Q0"])
        merge_circuit = STIMPhysicalCircuit("Y 0", ["Q1"])
        
        main_circuit.merge_inplace(merge_circuit, 0)
        
        # Should have both qubit labels now
        assert "Q0" in main_circuit.qubit_labels
        assert "Q1" in main_circuit.qubit_labels
        assert len(main_circuit.qubit_labels) == 2

    def test_padding_with_different_durations(self):
        """Test padding with gates of different durations."""
        circuit = STIMPhysicalCircuit("X 0", ["Q0", "Q1"])
        
        idle_names = {1: "I1", 2: "I2"}
        durations = {"X": 1}
        
        circuit.pad_single_qubit_idles_by_duration_inplace(
            idle_names, durations, default_duration=2
        )
        
        circuit_str = str(circuit.circuit)
        # Should have idle operations with appropriate durations
        assert "I2 1" in circuit_str  # Q1 should get duration 2 idle

    def test_nested_repeats_unrolling(self):
        """Test unrolling nested REPEAT blocks."""
        circuit = STIMPhysicalCircuit("REPEAT 2 {\nX 0\nREPEAT 3 {\nY 1\n}\n}\nZ 0", ["Q0", "Q1"])
        
        unrolled = circuit._unroll_repeats()
        
        # Should have X twice and Y 6 times (2*3)
        assert unrolled.count("X 0") == 2
        assert unrolled.count("Y 1") == 6
        assert "Z 0" in unrolled
        assert "REPEAT" not in unrolled

    def test_empty_repeat_block(self):
        """Test REPEAT block with no content."""
        circuit = STIMPhysicalCircuit("REPEAT 2 {\n}\nX 0", ["Q0"])
        
        unrolled = circuit._unroll_repeats()
        
        # Should just have X 0, no REPEAT
        assert "REPEAT" not in unrolled
        assert "X 0" in unrolled

    def test_circuit_with_annotations(self):
        """Test circuit with STIM annotations."""
        circuit = STIMPhysicalCircuit("X 0\nTICK\nQUBIT_COORDS 0 1.0 2.0 3.0", ["Q0"])
        
        # Annotations should be preserved
        assert "QUBIT_COORDS" in str(circuit.circuit)
        
        # Annotations should not be treated as gates for error locations
        locations = circuit.get_possible_discrete_error_locations()
        annotation_locs = [loc for loc in locations if "QUBIT_COORDS" in str(loc)]
        assert len(annotation_locs) == 0

    def test_error_location_edge_cases(self):
        """Test edge cases in error location detection."""
        # Empty circuit
        circuit = STIMPhysicalCircuit("", [])
        locations = circuit.get_possible_discrete_error_locations()
        assert len(locations) == 0
        
        # Circuit with only annotations
        circuit = STIMPhysicalCircuit("QUBIT_COORDS 0 1 2 3", ["Q0"])
        locations = circuit.get_possible_discrete_error_locations()
        assert len(locations) == 0
        
        # Circuit with unsupported instructions should still work for supported ones
        circuit = STIMPhysicalCircuit("X 0\nTICK\nY 1", ["Q0", "Q1"])
        locations = circuit.get_possible_discrete_error_locations()
        assert len(locations) == 2  # X and Y gates

    def test_qubit_deletion_edge_cases(self):
        """Test edge cases in qubit deletion."""
        # Delete all qubits
        circuit = STIMPhysicalCircuit("X 0\nY 1", ["Q0", "Q1"])
        circuit.delete_qubits_inplace(["Q0", "Q1"])
        
        assert circuit.qubit_labels == []
        assert circuit.circuit.num_qubits == 0
        
        # Delete non-existent qubit (should be silent)
        circuit = STIMPhysicalCircuit("X 0", ["Q0"])
        circuit.delete_qubits_inplace(["Q99"])
        assert circuit.qubit_labels == ["Q0"]

    def test_circuit_equality_after_operations(self):
        """Test that circuit operations maintain expected state."""
        # Create two identical circuits
        circ1 = STIMPhysicalCircuit("X 0\nY 1", ["Q0", "Q1"])
        circ2 = STIMPhysicalCircuit("X 0\nY 1", ["Q0", "Q1"])
        
        # They should have equivalent circuits
        assert str(circ1.circuit) == str(circ2.circuit)
        assert circ1.qubit_labels == circ2.qubit_labels
        
        # After operations, they should still be valid
        circ1.delete_qubits_inplace(["Q0"])
        assert circ1.qubit_labels == ["Q1"]
        assert "X 0" not in str(circ1.circuit)
        assert "Y 1" in str(circ1.circuit)

    def test_stim_backend_availability_check(self):
        """Test that STIM backend availability is checked."""
        # This test assumes STIM is available since we're running it
        # But we can test the error path by temporarily making it unavailable
        import loqs.backends.circuit.stimcircuit as stimcircuit_module
        
        # Save original function
        original_is_available = stimcircuit_module.is_backend_available
        
        # Mock to return False
        def mock_is_available(backend_name):
            return False
        
        stimcircuit_module.is_backend_available = mock_is_available
        
        try:
            with pytest.raises(ImportError, match="STIM backend is not available"):
                STIMPhysicalCircuit("X 0", ["Q0"])
        finally:
            # Restore original function
            stimcircuit_module.is_backend_available = original_is_available

    def test_circuit_with_complex_gate_sequences(self):
        """Test circuit with complex gate sequences."""
        circuit_str = """H 0
S 0
T 0
T 0
H 0
TICK
CX 0 1
CY 1 0
CZ 0 1
TICK
M 0
M 1"""
        
        circuit = STIMPhysicalCircuit(circuit_str, ["Q0", "Q1"])
        
        # Should be able to handle all these gates
        assert circuit.circuit.num_qubits == 2
        assert circuit.depth == 3
        
        # Test error location detection
        locations = circuit.get_possible_discrete_error_locations()
        assert len(locations) > 0

    def test_insert_at_various_positions(self):
        """Test inserting at different positions."""
        main_circuit = STIMPhysicalCircuit("X 0\nTICK\nY 0\nTICK\nZ 0", ["Q0"])
        insert_circuit = STIMPhysicalCircuit("H 0", ["Q0"])
        
        # Insert at position 0 (beginning)
        main_circuit.insert_inplace(insert_circuit, 0)
        circuit_str = str(main_circuit.circuit)
        h_pos = circuit_str.find("H 0")
        x_pos = circuit_str.find("X 0")
        assert h_pos < x_pos  # H should come before X
        
        # Test with another circuit for position 2
        main_circuit2 = STIMPhysicalCircuit("X 0\nTICK\nY 0\nTICK\nZ 0", ["Q0"])
        main_circuit2.insert_inplace(insert_circuit, 2)
        circuit_str2 = str(main_circuit2.circuit)
        y_pos = circuit_str2.find("Y 0")
        h_pos2 = circuit_str2.find("H 0")
        z_pos = circuit_str2.find("Z 0")
        assert y_pos < h_pos2 < z_pos  # Y < H < Z

    def test_merge_preserves_original_circuit(self):
        """Test that merge doesn't destroy original circuit."""
        main_circuit = STIMPhysicalCircuit("X 0", ["Q0"])
        merge_circuit = STIMPhysicalCircuit("Y 0", ["Q0"])
        
        original_main_str = str(main_circuit.circuit)
        original_merge_str = str(merge_circuit.circuit)
        
        main_circuit.merge_inplace(merge_circuit, 0)
        
        # Original circuits should be unchanged
        assert str(merge_circuit.circuit) == original_merge_str
        # Main circuit should be modified
        assert str(main_circuit.circuit) != original_main_str

    def test_padding_with_empty_layer_idle(self):
        """Test padding with empty layer idle operations."""
        circuit = STIMPhysicalCircuit("X 0\nTICK\nY 1", ["Q0", "Q1"])
        
        idle_names = {1: "I"}
        durations = {"X": 1, "Y": 1}
        
        circuit.pad_single_qubit_idles_by_duration_inplace(
            idle_names, durations, default_duration=1, empty_layer_idle="J"
        )
        
        circuit_str = str(circuit.circuit)
        # Should have idle operations
        assert "I" in circuit_str or "J" in circuit_str

    def test_qubit_label_mapping_partial(self):
        """Test partial qubit label mapping."""
        circuit = STIMPhysicalCircuit("X 0\nY 1\nZ 2", ["Q0", "Q1", "Q2"])
        
        # Only map Q1
        circuit.map_qubit_labels_inplace({"Q1": "QB"})
        
        assert circuit.qubit_labels == ["Q0", "QB", "Q2"]
        # Circuit should still work with indices
        assert "Y 1" in str(circuit.circuit)

    def test_circuit_serialization_roundtrip(self):
        """Test that serialization and deserialization preserve circuit."""
        original = STIMPhysicalCircuit("X 0\nY 1\nTICK\nZ 0", ["Q0", "Q1"])
        
        # Serialize
        serialized = original._serialize_circuit()
        
        # Deserialize
        deserialized_circuit = STIMPhysicalCircuit._deserialize_circuit(serialized)
        
        # Create new circuit from deserialized
        new_circuit = STIMPhysicalCircuit(deserialized_circuit, original.qubit_labels)
        
        # Should be equivalent
        assert str(new_circuit.circuit) == str(original.circuit)
        assert new_circuit.qubit_labels == original.qubit_labels

    def test_unroll_repeats_with_complex_structure(self):
        """Test unrolling REPEAT blocks with complex structure."""
        circuit = STIMPhysicalCircuit("""REPEAT 2 {
X 0
REPEAT 2 {
Y 1
Z 0
}
H 1
}
TICK
CX 0 1""", ["Q0", "Q1"])
        
        unrolled = circuit._unroll_repeats()
        
        # Should have X twice, Y and Z 4 times each (2*2), H twice
        assert unrolled.count("X 0") == 2
        assert unrolled.count("Y 1") == 4
        assert unrolled.count("Z 0") == 4
        assert unrolled.count("H 1") == 2
        assert unrolled.count("CX 0 1") == 1
        assert "REPEAT" not in unrolled

    def test_error_detection_with_mixed_gate_types(self):
        """Test error location detection with mixed gate types."""
        circuit = STIMPhysicalCircuit("""X 0
Y 1
TICK
CNOT 0 1
H 0
TICK
M 0
M 1""", ["Q0", "Q1"])
        
        locations = circuit.get_possible_discrete_error_locations()
        
        # Should detect all gate types
        single_qubit_locs = [loc for loc in locations if isinstance(loc[1], int)]
        two_qubit_locs = [loc for loc in locations if isinstance(loc[1], tuple)]
        
        assert len(single_qubit_locs) >= 4  # X, Y, H, M, M
        assert len(two_qubit_locs) == 1  # CNOT

    def test_circuit_with_all_gate_categories(self):
        """Test circuit containing gates from all categories."""
        circuit_str = """X 0
CNOT 0 1
M 0
X_ERROR(0.1) 0
REPEAT 2 {Y 1}
QUBIT_COORDS 0 1 2 3"""
        
        circuit = STIMPhysicalCircuit(circuit_str, ["Q0", "Q1"])
        
        # Should handle all categories properly
        assert "X" in STIMPhysicalCircuit._stim_oneq_gates
        assert "CNOT" in STIMPhysicalCircuit._stim_twoq_gates
        assert "M" in STIMPhysicalCircuit._stim_measure_reset_gates
        assert "X_ERROR" in STIMPhysicalCircuit._stim_noise_channels
        assert "REPEAT" in STIMPhysicalCircuit._stim_annotations
        assert "QUBIT_COORDS" in STIMPhysicalCircuit._stim_annotations

    def test_qubit_deletion_preserves_order(self):
        """Test that qubit deletion preserves remaining qubit order."""
        circuit = STIMPhysicalCircuit("X 0\nY 1\nZ 2\nH 3", ["Q0", "Q1", "Q2", "Q3"])
        
        # Delete middle qubits
        circuit.delete_qubits_inplace(["Q1", "Q2"])
        
        # Should preserve order of remaining qubits
        assert circuit.qubit_labels == ["Q0", "Q3"]
        # Should remove operations on deleted qubits
        assert "Y 1" not in str(circuit.circuit)
        assert "Z 2" not in str(circuit.circuit)
        assert "X 0" in str(circuit.circuit)
        assert "H 3" in str(circuit.circuit)

    def test_insert_with_empty_circuit(self):
        """Test inserting empty circuit."""
        main_circuit = STIMPhysicalCircuit("X 0", ["Q0"])
        empty_circuit = STIMPhysicalCircuit("", [])
        
        # Should handle empty circuit insertion gracefully
        main_circuit.insert_inplace(empty_circuit, 0)
        
        # Main circuit should be unchanged
        assert "X 0" in str(main_circuit.circuit)

    def test_merge_with_empty_circuit(self):
        """Test merging empty circuit."""
        main_circuit = STIMPhysicalCircuit("X 0", ["Q0"])
        empty_circuit = STIMPhysicalCircuit("", [])
        
        # Should handle empty circuit merge gracefully
        main_circuit.merge_inplace(empty_circuit, 0)
        
        # Main circuit should be unchanged
        assert "X 0" in str(main_circuit.circuit)

    def test_padding_with_no_idle_needed(self):
        """Test padding when no idles are needed."""
        circuit = STIMPhysicalCircuit("X 0\nY 1", ["Q0", "Q1"])
        
        idle_names = {1: "I"}
        durations = {"X": 1, "Y": 1}
        
        # Both qubits have operations, so no idles needed
        circuit.pad_single_qubit_idles_by_duration_inplace(
            idle_names, durations, default_duration=1
        )
        
        circuit_str = str(circuit.circuit)
        # Should not add any idle operations
        assert circuit_str.count("I") == 0

    def test_unroll_repeats_with_zero_repeats(self):
        """Test unrolling REPEAT with zero repeats."""
        circuit = STIMPhysicalCircuit("REPEAT 0 {X 0}\nY 0", ["Q0"])
        
        unrolled = circuit._unroll_repeats()
        
        # Should not include X 0 (zero repeats)
        assert "X 0" not in unrolled
        assert "Y 0" in unrolled

    def test_error_location_with_flipped_measurements(self):
        """Test error location detection with flipped measurements."""
        circuit = STIMPhysicalCircuit("X 0\nM! 0", ["Q0"])
        
        locations = circuit.get_possible_discrete_error_locations()
        
        # Should handle flipped measurements (M!)
        assert len(locations) == 2  # X and M!

    def test_circuit_with_all_stim_features(self):
        """Test circuit using various STIM features."""
        circuit_str = """H 0
TICK
CX 0 1
TICK
M 0
TICK
DEPOLARIZE1(0.01) 0
TICK
REPEAT 2 {
X 0
Y 1
}
TICK
QUBIT_COORDS 0 1.0 2.0 3.0
QUBIT_COORDS 1 4.0 5.0 6.0"""
        
        circuit = STIMPhysicalCircuit(circuit_str, ["Q0", "Q1"])
        
        # Should handle all these features
        assert circuit.circuit.num_qubits == 2
        assert circuit.depth == 6  # 5 TICKs + 1
        
        # Test unrolling
        unrolled = circuit._unroll_repeats()
        assert unrolled.count("X 0") == 2
        assert unrolled.count("Y 1") == 2
        assert "REPEAT" not in unrolled

    def test_qubit_label_consistency_after_operations(self):
        """Test that qubit labels remain consistent after various operations."""
        circuit = STIMPhysicalCircuit("X 0\nY 1\nZ 2", ["QA", "QB", "QC"])
        
        # After copy
        copied = circuit.copy()
        assert copied.qubit_labels == ["QA", "QB", "QC"]
        
        # After delete
        circuit.delete_qubits_inplace(["QB"])
        assert circuit.qubit_labels == ["QA", "QC"]
        
        # After map
        circuit.map_qubit_labels_inplace({"QA": "Q1", "QC": "Q2"})
        assert circuit.qubit_labels == ["Q1", "Q2"]
        
        # After set
        circuit.set_qubit_labels_inplace(["R1", "R2"])
        assert circuit.qubit_labels == ["R1", "R2"]

    def test_stim_circuit_manipulation_preserves_functionality(self):
        """Test that circuit manipulations preserve basic functionality."""
        # Create a circuit that does something simple (X on qubit 0)
        circuit = STIMPhysicalCircuit("X 0", ["Q0"])
        
        # Copy it
        copied = circuit.copy()
        
        # Both should work the same way
        assert str(circuit.circuit) == str(copied.circuit)
        assert circuit.qubit_labels == copied.qubit_labels
        
        # Modify original
        circuit.circuit.append("Y", [0])
        
        # Original should be modified, copy should not
        assert "Y" in str(circuit.circuit)
        assert "Y" not in str(copied.circuit)

    def test_comprehensive_circuit_workflow(self):
        """Test a comprehensive workflow with multiple operations."""
        # Start with simple circuit
        circuit = STIMPhysicalCircuit("X 0", ["Q0"])
        
        # Add a qubit and operation
        circuit.circuit.append("Y", [1])
        circuit.set_qubit_labels_inplace(["Q0", "Q1"])
        
        # Add layers
        circuit.circuit.append("TICK")
        circuit.circuit.append("Z", [0])
        
        # Test functionality
        assert circuit.depth == 2
        assert len(circuit.qubit_labels) == 2
        
        locations = circuit.get_possible_discrete_error_locations()
        assert len(locations) == 3  # X, Y, Z
        
        # Copy and modify
        copied = circuit.copy()
        copied.delete_qubits_inplace(["Q1"])
        
        assert len(copied.qubit_labels) == 1
        assert "Y" not in str(copied.circuit)

    def test_edge_case_empty_operations(self):
        """Test edge cases with empty or minimal operations."""
        # Empty circuit
        empty = STIMPhysicalCircuit("", [])
        assert empty.circuit.num_qubits == 0
        assert empty.depth == 1
        
        # Circuit with only TICK
        tick_only = STIMPhysicalCircuit("TICK", [])
        assert tick_only.depth == 2
        
        # Circuit with only annotations
        annot_only = STIMPhysicalCircuit("QUBIT_COORDS 0 1 2 3", ["Q0"])
        assert annot_only.circuit.num_qubits == 1
        
        locations = annot_only.get_possible_discrete_error_locations()
        assert len(locations) == 0  # No actual gates

    def test_error_handling_for_malformed_circuits(self):
        """Test error handling for various malformed circuits."""
        # This tests that the circuit handles edge cases gracefully
        
        # Circuit with mismatched REPEAT braces (should be caught by STIM)
        # We don't test this directly as STIM should catch it during parsing
        
        # Circuit with invalid gate names (should be caught by STIM)
        # Again, STIM handles this
        
        # Our main concern is that our wrapper handles STIM errors appropriately
        # and that we catch the specific unsupported instructions we care about
        
        with pytest.raises(ValueError, match="LoQS-unsupported instruction"):
            STIMPhysicalCircuit("SPP 0", ["Q0"])

    def test_performance_with_large_circuit(self):
        """Test that operations work reasonably with larger circuits."""
        # Create a moderately sized circuit
        circuit_str = "TICK\n".join([f"X {i}\nY {i}" for i in range(10)])
        circuit = STIMPhysicalCircuit(circuit_str, [f"Q{i}" for i in range(10)])
        
        # Should be able to get error locations
        locations = circuit.get_possible_discrete_error_locations()
        assert len(locations) == 20  # 2 gates per qubit * 10 qubits
        
        # Should be able to copy
        copied = circuit.copy()
        assert str(copied.circuit) == str(circuit.circuit)
        
        # Should be able to delete some qubits
        circuit.delete_qubits_inplace(["Q0", "Q1", "Q2"])
        assert len(circuit.qubit_labels) == 7

    def test_circuit_equivalence_after_roundtrip(self):
        """Test that circuits remain equivalent after serialization roundtrip."""
        original = STIMPhysicalCircuit("X 0\nY 1\nTICK\nZ 0\nH 1", ["Q0", "Q1"])
        
        # Serialize and deserialize
        serialized = original._serialize_circuit()
        deserialized = STIMPhysicalCircuit._deserialize_circuit(serialized)
        
        # Create new circuit
        new_circuit = STIMPhysicalCircuit(deserialized, original.qubit_labels)
        
        # Should be functionally equivalent
        assert str(new_circuit.circuit) == str(original.circuit)
        assert new_circuit.qubit_labels == original.qubit_labels
        assert new_circuit.depth == original.depth

    def test_unroll_repeats_preserves_structure(self):
        """Test that unrolling REPEAT blocks preserves circuit structure."""
        circuit = STIMPhysicalCircuit("""X 0
REPEAT 3 {
Y 1
Z 0
}
H 1
TICK
CX 0 1""", ["Q0", "Q1"])
        
        unrolled = circuit._unroll_repeats()
        
        # Structure should be preserved
        lines = unrolled.split("\n")
        x_line_idx = lines.index("X 0")
        h_line_idx = lines.index("H 1")
        tick_line_idx = lines.index("TICK")
        cx_line_idx = lines.index("CX 0 1")
        
        assert x_line_idx < h_line_idx < tick_line_idx < cx_line_idx
        assert unrolled.count("Y 1") == 3
        assert unrolled.count("Z 0") == 3

    def test_comprehensive_error_location_scenarios(self):
        """Test error location detection in various scenarios."""
        scenarios = [
            # (circuit_str, expected_min_locations)
            ("X 0", 1),  # Single gate
            ("X 0\nY 1", 2),  # Multiple single-qubit gates
            ("CNOT 0 1", 1),  # Single two-qubit gate
            ("X 0\nTICK\nY 0", 2),  # Multiple layers
            ("X 0\nM 0", 2),  # Gate + measurement
            ("REPEAT 2 {X 0}", 2),  # Repeated gate
            ("X 0\nTICK\nCNOT 0 1\nTICK\nY 1", 3),  # Mixed gates
        ]
        
        for circuit_str, expected_min in scenarios:
            circuit = STIMPhysicalCircuit(circuit_str, ["Q0", "Q1"][:circuit.circuit.num_qubits])
            locations = circuit.get_possible_discrete_error_locations()
            assert len(locations) >= expected_min, f"Failed for circuit: {circuit_str}"

    def test_final_comprehensive_test(self):
        """Final comprehensive test covering multiple features."""
        # Create a complex circuit
        circuit_str = """H 0
S 0
T 0
TICK
CX 0 1
CY 1 0
CZ 0 1
TICK
M 0
M 1
TICK
REPEAT 2 {
X 0
Y 1
}
TICK
QUBIT_COORDS 0 1.0 2.0 3.0
QUBIT_COORDS 1 4.0 5.0 6.0
X_ERROR(0.01) 0
DEPOLARIZE1(0.01) 1"""
        
        circuit = STIMPhysicalCircuit(circuit_str, ["Q0", "Q1"])
        
        # Test basic properties
        assert circuit.circuit.num_qubits == 2
        assert circuit.depth == 6  # 5 TICKs + 1
        assert len(circuit.qubit_labels) == 2
        
        # Test copying
        copied = circuit.copy()
        assert str(copied.circuit) == str(circuit.circuit)
        
        # Test error locations
        locations = circuit.get_possible_discrete_error_locations()
        assert len(locations) > 0
        
        # Test unrolling
        unrolled = circuit._unroll_repeats()
        assert unrolled.count("X 0") == 2
        assert unrolled.count("Y 1") == 2
        assert "REPEAT" not in unrolled
        
        # Test serialization
        serialized = circuit._serialize_circuit()
        assert isinstance(serialized, str)
        assert len(serialized) > 0
        
        # Test qubit deletion
        circuit.delete_qubits_inplace(["Q0"])
        assert circuit.qubit_labels == ["Q1"]
        assert "X 0" not in str(circuit.circuit)  # Operations on Q0 should be removed
        
        # Test label mapping
        circuit.map_qubit_labels_inplace({"Q1": "QA"})
        assert circuit.qubit_labels == ["QA"]

    def test_stim_backend_integration(self):
        """Test integration with STIM backend features."""
        # Test that we can use STIM features through our wrapper
        circuit = STIMPhysicalCircuit("X 0", ["Q0"])
        
        # Access underlying STIM circuit
        stim_circuit = circuit.circuit
        assert isinstance(stim_circuit, stim.Circuit)
        
        # Should be able to manipulate the STIM circuit directly
        stim_circuit.append("Y", [0])
        assert "Y 0" in str(circuit.circuit)
        
        # Should be able to get STIM-specific properties
        assert hasattr(stim_circuit, 'num_qubits')
        assert hasattr(stim_circuit, 'num_ticks')
        
        # Our wrapper should reflect STIM properties
        assert circuit.depth == stim_circuit.num_ticks + 1

    def test_circuit_manipulation_consistency(self):
        """Test that circuit manipulations maintain consistency."""
        # Start with a circuit
        circuit = STIMPhysicalCircuit("X 0\nY 1", ["Q0", "Q1"])
        
        # Get initial state
        initial_str = str(circuit.circuit)
        initial_labels = circuit.qubit_labels.copy()
        
        # Perform operations
        circuit.delete_qubits_inplace(["Q0"])
        circuit.map_qubit_labels_inplace({"Q1": "QA"})
        circuit.circuit.append("Z", [0])
        
        # Should maintain consistency
        assert circuit.qubit_labels == ["QA"]
        assert "Z 0" in str(circuit.circuit)
        assert "X 0" not in str(circuit.circuit)  # Q0 was deleted
        
        # Error locations should still work
        locations = circuit.get_possible_discrete_error_locations()
        assert len(locations) > 0

    def test_final_validation_test(self):
        """Final validation test to ensure all major functionality works."""
        # This test exercises most of the major functionality
        
        # 1. Create circuit
        circuit = STIMPhysicalCircuit("H 0\nCX 0 1\nM 0\nM 1", ["Q0", "Q1"])
        
        # 2. Test properties
        assert circuit.name == "STIM"
        assert circuit.depth == 1  # No TICKs
        assert circuit.circuit.num_qubits == 2
        
        # 3. Test copying
        copied = circuit.copy()
        assert str(copied.circuit) == str(circuit.circuit)
        
        # 4. Test error locations
        locations = circuit.get_possible_discrete_error_locations()
        assert len(locations) == 4  # H, CX, M, M
        
        # 5. Test serialization
        serialized = circuit._serialize_circuit()
        deserialized = STIMPhysicalCircuit._deserialize_circuit(serialized)
        assert str(deserialized) == serialized
        
        # 6. Test qubit operations
        circuit.delete_qubits_inplace(["Q1"])
        assert circuit.qubit_labels == ["Q0"]
        assert "CX 0 1" not in str(circuit.circuit)
        
        # 7. Test label operations
        circuit.map_qubit_labels_inplace({"Q0": "QA"})
        assert circuit.qubit_labels == ["QA"]
        
        # 8. Test circuit modification
        circuit.circuit.append("Y", [0])
        assert "Y 0" in str(circuit.circuit)
        
        # If we get here, all major functionality works!
        assert True