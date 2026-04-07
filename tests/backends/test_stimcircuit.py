"""Tester for loqs.backends.circuit.stimcircuit"""

import pytest

try:
    import stim

    NO_STIM = False
except ImportError:
    NO_STIM = True

from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit


@pytest.mark.skipif(
    NO_STIM,
    reason="Skipping stim backend tests due to failed import"
)
class TestSTIMPhysicalCircuit:

    def test_init(self):
        # Test sparse STIM circuit (non-contiguous indices)
        sparse_circuit_str = "H 0\nTICK\nCX 0 5"
        
        # Test with None qubit_labels - should extract used indices [0, 5]
        circ1 = STIMPhysicalCircuit(sparse_circuit_str)
        assert circ1.qubit_labels == [0, 5]
        assert circ1.circuit.num_qubits == 2  # Compact indices
        
        # Test with explicit qubit_labels - the string contains integer indices
        custom_labels = ['Q0', 'Q5']
        circ2 = STIMPhysicalCircuit("H 0\nTICK\nCX 0 1", custom_labels)
        assert circ2.qubit_labels == custom_labels
        assert circ2.circuit.num_qubits == 2
        
        # Test with stim.Circuit object
        stim_circ = stim.Circuit(sparse_circuit_str)
        circ3 = STIMPhysicalCircuit(stim_circ)
        assert circ3.qubit_labels == [0, 5]
        assert circ3.circuit.num_qubits == 2
        
        # Test with stim.Circuit and custom labels
        circ4 = STIMPhysicalCircuit(stim_circ, custom_labels)
        assert circ4.qubit_labels == custom_labels
        assert circ4.circuit.num_qubits == 2
        
        # Test copying from another STIMPhysicalCircuit
        circ5 = STIMPhysicalCircuit(circ1)
        assert circ5.qubit_labels == circ1.qubit_labels
        assert str(circ5.circuit) == str(circ1.circuit)
        
        # Test copying with different qubit_labels
        new_labels = ['A', 'B']
        circ6 = STIMPhysicalCircuit(circ1, new_labels)
        assert circ6.qubit_labels == new_labels
        assert circ6.circuit.num_qubits == 2
        
        # Test ValueError cases
        with pytest.raises(ValueError):
            STIMPhysicalCircuit(circ1, ['A'])  # Wrong length
            
        with pytest.raises(ValueError):
            # String with custom labels that don't match circuit references
            STIMPhysicalCircuit("H Q0\nTICK\nCX Q0 Q1", ['Q0'])
        
        with pytest.raises(ValueError):
            # stim.Circuit with insufficient labels
            stim_circ_3q = stim.Circuit("H 0\nCX 0 1\nCX 1 2")
            STIMPhysicalCircuit(stim_circ_3q, ['Q0', 'Q1'])

    def test_delete_qubits(self):
        # Create a circuit with 3 qubits
        circ_str = "H 0\nCX 0 1\nTICK\nX 2"
        circ = STIMPhysicalCircuit(circ_str)
        
        # Delete qubit 1
        circ.delete_qubits_inplace([1])
        assert circ.qubit_labels == [0, 2]
        assert circ.circuit.num_qubits == 2
        
        # Check that the circuit string no longer contains operations on qubit 1
        circ_str_after = str(circ.circuit)
        assert ' 1 ' not in circ_str_after and ' 1\n' not in circ_str_after
        
        # Delete another qubit
        circ.delete_qubits_inplace([0])
        assert circ.qubit_labels == [2]
        assert circ.circuit.num_qubits == 1

    def test_merge(self):
        # Create two circuits with overlapping and new qubits
        circ1_str = "H Q0\nTICK\nX Q1"
        circ2_str = "Y Q1\nTICK\nZ Q2"  # circ2 shares Q1 and adds Q2
        
        circ1 = STIMPhysicalCircuit(circ1_str, ['Q0', 'Q1'])
        circ2 = STIMPhysicalCircuit(circ2_str, ['Q1', 'Q2'])
        
        # Merge circ2 into circ1 starting at layer 0
        circ1.merge_inplace(circ2, 0)
        
        # Should have all qubit labels from both circuits
        assert set(circ1.qubit_labels) == {'Q0', 'Q1', 'Q2'}
        assert circ1.circuit.num_qubits == 3
        
        # Check that operations are correctly mapped
        circ_str = str(circ1.circuit)
        assert 'H 0' in circ_str  # Q0 -> STIM idx 0 (from circ1)
        assert 'X 1' in circ_str  # Q1 -> STIM idx 1 (from circ1)
        assert 'Y 1' in circ_str  # Q1 -> STIM idx 1 (from circ2, remapped)
        assert 'Z 2' in circ_str  # Q2 -> STIM idx 2 (from circ2, remapped)

    def test_get_possible_discrete_error_locations(self):
        # Create a simple circuit
        circ_str = "H 0\nTICK\nCX 0 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])
        
        # Get error locations
        locations = circ.get_possible_discrete_error_locations()
        
        # Should return LoQS labels, not STIM indices
        for layer_idx, qubit_info in locations:
            if isinstance(qubit_info, tuple):
                # Two-qubit gate
                assert qubit_info[0] in ['Q0', 'Q1']
                assert qubit_info[1] in ['Q0', 'Q1']
            else:
                # Single-qubit gate
                assert qubit_info in ['Q0', 'Q1']
        
        # Test post_twoq_gates mode
        locations_2q = circ.get_possible_discrete_error_locations(post_twoq_gates=True)
        for layer_idx, qubit_info in locations_2q:
            assert isinstance(qubit_info, tuple)
            assert qubit_info[0] in ['Q0', 'Q1']
            assert qubit_info[1] in ['Q0', 'Q1']

    def test_map_qubit_labels(self):
        # Create a circuit
        circ_str = "H 0\nTICK\nCX 0 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])
        
        # Map qubit labels
        mapping = {'Q0': 'A', 'Q1': 'B'}
        circ.map_qubit_labels_inplace(mapping)
        
        assert circ.qubit_labels == ['A', 'B']
        
        # The internal STIM circuit should remain unchanged (still uses compact indices)
        assert circ.circuit.num_qubits == 2
        circ_str_after = str(circ.circuit)
        assert 'H 0' in circ_str_after
        assert 'CX 0 1' in circ_str_after

    def test_pad_idles(self):
        # Create a simple circuit with 2 qubits
        circ_str = "H 0\nTICK\nX 0\nTICK\nH 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])
        
        # Pad with idles
        durations = {'H': 1, 'X': 1}
        idle_names = {1: 'I'}
        
        circ.pad_single_qubit_idles_by_duration_inplace(
            idle_names, durations, default_duration=1
        )
        
        # Should have added idle operations where needed
        circ_str_after = str(circ.circuit)
        # Check that we have the expected structure
        assert 'H 0' in circ_str_after
        assert 'X 0' in circ_str_after
        assert 'H 1' in circ_str_after
        
        # Test simple pad_single_qubit_idles (without durations)
        # Use a circuit that actually uses both qubits to maintain the invariant
        circ2 = STIMPhysicalCircuit("H 0\nI 1\nTICK\nX 0\nI 1", ['Q0', 'Q1'])
        circ2.pad_single_qubit_idles_inplace("I")
        
        circ2_str = str(circ2.circuit)
        assert 'H 0' in circ2_str
        assert 'X 0' in circ2_str

    def test_copy(self):
        # Create a circuit
        circ_str = "H 0\nTICK\nCX 0 1"
        circ1 = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])
        
        # Copy it
        circ2 = circ1.copy()
        
        # Should be identical
        assert circ2.qubit_labels == circ1.qubit_labels
        assert str(circ2.circuit) == str(circ1.circuit)
        
        # Modifying one shouldn't affect the other
        circ2.map_qubit_labels_inplace({'Q0': 'A', 'Q1': 'B'})
        assert circ1.qubit_labels == ['Q0', 'Q1']
    
    def test_set_qubit_labels(self):
        # Create a circuit
        circ_str = "H 0\nTICK\nCX 0 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])
        
        # Set new qubit labels
        new_labels = ['A', 'B']
        circ.set_qubit_labels_inplace(new_labels)
        
        assert circ.qubit_labels == new_labels
        # Internal STIM circuit should be unchanged
        assert circ.circuit.num_qubits == 2
        circ_str_after = str(circ.circuit)
        assert 'H 0' in circ_str_after
        assert 'CX 0 1' in circ_str_after
        
        # Test non-inplace version
        circ2 = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])
        circ3 = circ2.set_qubit_labels(['X', 'Y'])
        
        assert circ2.qubit_labels == ['Q0', 'Q1']  # Original unchanged
        assert circ3.qubit_labels == ['X', 'Y']    # New circuit has new labels

    def test_sparse_circuit_compactness(self):
        # Test that sparse circuits maintain compact indices
        sparse_circuit_str = "H 0\nTICK\nCX 0 10\nTICK\nM 10"
        circ = STIMPhysicalCircuit(sparse_circuit_str)
        
        # Should have exactly the used indices as labels
        assert circ.qubit_labels == [0, 10]
        assert circ.circuit.num_qubits == 2
        
        # Internal circuit should use compact indices 0, 1
        circ_str = str(circ.circuit)
        assert "H 0" in circ_str
        assert "CX 0 1" in circ_str
        assert "M 1" in circ_str
        assert "H 10" not in circ_str  # Original sparse index should be gone
        
        # Test deletion from sparse circuit
        circ.delete_qubits_inplace([10])
        assert circ.circuit.num_qubits == 1
        assert circ.qubit_labels == [0]
        
        # After merging with another sparse circuit
        other_str = "X 10\nTICK\nY 15"
        other_circ = STIMPhysicalCircuit(other_str)
        circ.merge_inplace(other_circ, 0)
        
        # Should have compact indices for all qubits
        assert circ.circuit.num_qubits == 3
        assert set(circ.qubit_labels) == {0, 10, 15}

    def test_qubit_labels_property(self):
        # Test that the assertion holds
        circ_str = "H 0\nTICK\nCX 0 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])
        
        # This should not raise an assertion error
        labels = circ.qubit_labels
        assert labels == ['Q0', 'Q1']
        assert len(labels) == circ.circuit.num_qubits

    def test_measurement_with_custom_labels(self):
        # Test measurement operations with custom labels
        circ_str = "H 0\nTICK\nM 0\nTICK\nMR 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])
        
        assert circ.qubit_labels == ['Q0', 'Q1']
        assert circ.circuit.num_qubits == 2
        
        circ_str_after = str(circ.circuit)
        assert "H 0" in circ_str_after
        assert "M 0" in circ_str_after
        assert "MR 1" in circ_str_after

    def test_repeat_blocks(self):
        # Test that repeat blocks work correctly
        circ_str = "REPEAT 2 {\n    H 0\n    CX 0 1\n    TICK\n}"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])
        
        assert circ.qubit_labels == ['Q0', 'Q1']
        assert circ.circuit.num_qubits == 2
        
        # After unrolling, should have correct operations
        unrolled = circ._unroll_repeats()
        assert unrolled.count("H 0") == 2
        assert unrolled.count("CX 0 1") == 2

    def test_invalid_qubit_labels(self):
        # Test various invalid qubit label scenarios
        
        # Too few labels for circuit (using label names in string)
        circ_str = "H Q0\nCX Q0 Q1"
        with pytest.raises(ValueError):
            STIMPhysicalCircuit(circ_str, ['Q0'])
        
        # Mismatched labels when copying from STIMPhysicalCircuit
        circ1 = STIMPhysicalCircuit("H 0\nCX 0 1", ['Q0', 'Q1'])
        with pytest.raises(ValueError):
            STIMPhysicalCircuit(circ1, ['Q0'])  # Wrong number
        
        # Unknown label in circuit string
        with pytest.raises(ValueError):
            STIMPhysicalCircuit("H Unknown", ['Q0', 'Q1'])

    def test_empty_circuit(self):
        # Test empty circuit
        circ = STIMPhysicalCircuit("", [])
        assert circ.qubit_labels == []
        assert circ.circuit.num_qubits == 0
        
        # Test circuit with only TICK
        circ2 = STIMPhysicalCircuit("TICK", [])
        assert circ2.qubit_labels == []
        assert circ2.circuit.num_qubits == 0

    def test_annotation_instructions(self):
        # Test that annotation instructions work
        # Note: DETECTOR requires specific syntax, so we'll test with a simpler case
        circ_str = "H 0\nTICK"
        circ = STIMPhysicalCircuit(circ_str)
        
        assert circ.qubit_labels == [0]
        assert circ.circuit.num_qubits == 1
        
        # Basic operations should work
        circ_str_after = str(circ.circuit)
        assert "H 0" in circ_str_after
        assert "TICK" in circ_str_after

    def test_helper_functions(self):
        # Test the helper functions directly
        import stim
        from loqs.backends.circuit.stimcircuit import _get_used_stim_indices, _reindex_stim_circuit
        
        # Test _get_used_stim_indices
        circ = stim.Circuit("H 0\nCX 0 5\nM 3")
        used_indices = _get_used_stim_indices(circ)
        assert used_indices == [0, 3, 5]
        
        # Test _reindex_stim_circuit
        index_map = {0: 0, 3: 1, 5: 2}
        reindexed_circ = _reindex_stim_circuit(circ, index_map)
        circ_str = str(reindexed_circ)
        assert "H 0" in circ_str
        assert "CX 0 2" in circ_str
        assert "M 1" in circ_str
        assert reindexed_circ.num_qubits == 3
        
        # Test with measurement targets including inversion
        circ_with_inv = stim.Circuit("H 0\nM !1\nCX 0 2")
        used_indices_inv = _get_used_stim_indices(circ_with_inv)
        assert used_indices_inv == [0, 1, 2]
        
        index_map_inv = {0: 0, 1: 1, 2: 2}
        reindexed_inv = _reindex_stim_circuit(circ_with_inv, index_map_inv)
        circ_inv_str = str(reindexed_inv)
        assert "H 0" in circ_inv_str
        assert "M !1" in circ_inv_str
        assert "CX 0 2" in circ_inv_str

    def test_edge_cases(self):
        # Test various edge cases
        
        # Single qubit circuit
        circ1 = STIMPhysicalCircuit("H 0", ['Q0'])
        assert circ1.qubit_labels == ['Q0']
        assert circ1.circuit.num_qubits == 1
        
        # Circuit with only measurements
        circ2 = STIMPhysicalCircuit("M Q0\nTICK\nM Q1", ['Q0', 'Q1'])
        assert circ2.qubit_labels == ['Q0', 'Q1']
        assert circ2.circuit.num_qubits == 2
        
        # Circuit with mixed operations
        circ3 = STIMPhysicalCircuit("H Q0\nCX Q0 Q1\nM Q0\nTICK\nX Q1", ['Q0', 'Q1'])
        assert circ3.qubit_labels == ['Q0', 'Q1']
        assert circ3.circuit.num_qubits == 2
        
        # Test error locations with mixed operations
        error_locs = circ3.get_possible_discrete_error_locations()
        assert len(error_locs) > 0
        for layer_idx, qubit_label in error_locs:
            assert qubit_label in ['Q0', 'Q1']
            assert isinstance(qubit_label, str)
        
        # Test post_twoq_gates mode
        error_locs_2q = circ3.get_possible_discrete_error_locations(post_twoq_gates=True)
        for layer_idx, qubit_tuple in error_locs_2q:
            assert isinstance(qubit_tuple, tuple)
            assert len(qubit_tuple) == 2
            assert qubit_tuple[0] in ['Q0', 'Q1']
            assert qubit_tuple[1] in ['Q0', 'Q1']

    def test_stim_circuit_object_input(self):
        # Test initialization with stim.Circuit object
        import stim
        
        # Test with no qubit_labels
        stim_circ = stim.Circuit("H 0\nCX 0 5")
        circ1 = STIMPhysicalCircuit(stim_circ)
        assert circ1.qubit_labels == [0, 5]
        assert circ1.circuit.num_qubits == 2
        
        # Test with qubit_labels
        stim_circ2 = stim.Circuit("H 0\nCX 0 1")
        circ2 = STIMPhysicalCircuit(stim_circ2, ['A', 'B'])
        assert circ2.qubit_labels == ['A', 'B']
        assert circ2.circuit.num_qubits == 2
        
        # Test error case: not enough labels
        with pytest.raises(ValueError):
            STIMPhysicalCircuit(stim_circ2, ['A'])

    def test_complex_merge_scenario(self):
        # Test a more complex merge scenario
        circ1 = STIMPhysicalCircuit("H Q0\nTICK\nX Q1\nTICK\nM Q0", ['Q0', 'Q1'])
        circ2 = STIMPhysicalCircuit("Y Q2\nTICK\nZ Q3", ['Q2', 'Q3'])
        
        # Merge at layer 2 (after second TICK) - no collision
        circ1.merge_inplace(circ2, 2)
        
        # Should have all four labels
        assert set(circ1.qubit_labels) == {'Q0', 'Q1', 'Q2', 'Q3'}
        assert circ1.circuit.num_qubits == 4
        
        # Check that the circuit structure is correct
        circ_str = str(circ1.circuit)
        assert 'H 0' in circ_str
        assert 'X 1' in circ_str
        assert 'Y 2' in circ_str
        assert 'Z 3' in circ_str
        assert 'M 0' in circ_str
    
    def test_insert_and_append(self):
        # Test insert_inplace method
        circ1 = STIMPhysicalCircuit("H 0\nTICK\nX 0", ['Q0'])
        circ2 = STIMPhysicalCircuit("Y 0\nTICK\nZ 0", ['Q0'])
        
        # Insert circ2 at layer 1 (after first TICK)
        circ1.insert_inplace(circ2, 1)
        
        circ_str = str(circ1.circuit)
        assert circ_str.count('H 0') == 1
        assert circ_str.count('Y 0') == 1
        assert circ_str.count('X 0') == 1
        assert circ_str.count('Z 0') == 1
        
        # Test append_inplace method
        circ3 = STIMPhysicalCircuit("H 0\nTICK", ['Q0'])
        circ4 = STIMPhysicalCircuit("X 0\nTICK", ['Q0'])
        
        circ3.append_inplace(circ4)
        
        circ3_str = str(circ3.circuit)
        assert 'H 0' in circ3_str
        assert 'X 0' in circ3_str
        assert circ3_str.count('TICK') == 2
        
        # Test append method (non-inplace)
        circ5 = STIMPhysicalCircuit("H 0\nTICK", ['Q0'])
        circ6 = circ5.append(circ4)
        
        # Original should be unchanged
        assert str(circ5.circuit) == "H 0\nTICK"
        # New circuit should have both
        circ6_str = str(circ6.circuit)
        assert 'H 0' in circ6_str
        assert 'X 0' in circ6_str
    
    def test_serialization_methods(self):
        # Test _serialize_circuit and _deserialize_circuit methods
        circ_str = "H 0\nTICK\nCX 0 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])
        
        # Test serialization
        serialized = circ._serialize_circuit()
        assert isinstance(serialized, str)
        assert 'H 0' in serialized
        assert 'CX 0 1' in serialized
        
        # Test deserialization
        deserialized_circ = STIMPhysicalCircuit._deserialize_circuit(serialized, ['Q0', 'Q1'])
        assert str(deserialized_circ) == serialized
        
        # Test that serialization preserves the circuit
        circ2 = STIMPhysicalCircuit(deserialized_circ, ['Q0', 'Q1'])
        assert circ2.qubit_labels == circ.qubit_labels
        assert str(circ2.circuit) == str(circ.circuit)
    
    def test_command_aliases(self):
        # Test substitute_command_aliases method
        circ_str = "CNOT 0 1\nTICK\nH 0"
        
        # Apply alias substitution
        aliased_str = STIMPhysicalCircuit.substitute_command_aliases(circ_str)
        
        # CNOT should be replaced with CX
        assert 'CX 0 1' in aliased_str
        assert 'CNOT' not in aliased_str
        assert 'H 0' in aliased_str
        
        # Test with a circuit that has aliases
        circ = STIMPhysicalCircuit("CNOT 0 1\nTICK", ['Q0', 'Q1'])
        circ_str_after = str(circ.circuit)
        # The alias should be preserved in the internal circuit
        assert 'CNOT 0 1' in circ_str_after or 'CX 0 1' in circ_str_after
    
    def test_comprehensive_init_cases(self):
        # Test various initialization scenarios to cover more __init__ paths
        
        # Test with BasePhysicalCircuit (should raise NotImplementedError)
        # from loqs.backends import ListPhysicalCircuit
        # list_circ = ListPhysicalCircuit([('H', 'Q0')], ['Q0'])
        #
        # try:
        #     STIMPhysicalCircuit(list_circ)
        #     assert False, "Should have raised NotImplementedError"
        # except NotImplementedError:
        #     pass  # Expected
        
        # Test with unsupported STIM instructions
        try:
            STIMPhysicalCircuit("MPP 0 1\nTICK")
            assert False, "Should have raised ValueError for unsupported instruction"
        except ValueError as e:
            assert "MPP" in str(e)
        
        # Test warning suppression
        circ_no_warn = STIMPhysicalCircuit("H 0", ['Q0'], suppress_tick_warning=True)
        assert circ_no_warn.qubit_labels == ['Q0']
    
    def test_insert_edge_cases(self):
        # Test insert at various positions
        circ1 = STIMPhysicalCircuit("H 0\nTICK\nX 0\nTICK\nY 0", ['Q0'])
        circ2 = STIMPhysicalCircuit("Z 0\nTICK", ['Q0'])
        
        # Insert at beginning (idx=0)
        circ1.insert_inplace(circ2, 0)
        circ_str = str(circ1.circuit)
        assert circ_str.startswith('Z 0')
        
        # Test insert at end (use depth instead of -1)
        circ3 = STIMPhysicalCircuit("H 0\nTICK", ['Q0'])
        circ4 = STIMPhysicalCircuit("X 0\nTICK", ['Q0'])
        circ3.insert_inplace(circ4, circ3.depth)  # Insert at end
        circ3_str = str(circ3.circuit)
        assert 'H 0' in circ3_str
        assert 'X 0' in circ3_str
        assert circ3_str.count('TICK') == 2
    
    def test_merge_edge_cases(self):
        # Test a case that should cause collision
        circ1 = STIMPhysicalCircuit("H 0\nTICK\nX 0", ['Q0'])
        circ2 = STIMPhysicalCircuit("Y 0\nTICK\nZ 0", ['Q0'])
        
        # Try to merge at layer 0 where both circuits have operations on Q0
        try:
            circ1.merge_inplace(circ2, 0)
            assert False, "Should have raised ValueError for collision"
        except ValueError as e:
            assert "ill-posed" in str(e).lower()
    
    def test_pad_edge_cases(self):
        # Test padding with empty layers - use circuits that actually use both qubits
        circ = STIMPhysicalCircuit("H 0\nI 1\nTICK\nH 0\nI 1", ['Q0', 'Q1'])
        
        durations = {'H': 1, 'I': 1}
        idle_names = {1: 'I'}
        
        # This should work since both qubits are already used
        circ.pad_single_qubit_idles_by_duration_inplace(
            idle_names, durations, default_duration=1, empty_layer_idle='I'
        )
        
        circ_str = str(circ.circuit)
        assert 'H 0' in circ_str
        assert 'I 1' in circ_str
    
    def test_comprehensive_helper_coverage(self):
        # Test helper functions more comprehensively
        import stim
        from loqs.backends.circuit.stimcircuit import _get_used_stim_indices, _reindex_stim_circuit
        
        # Test _get_used_stim_indices with various gate types
        circ = stim.Circuit("H 0\nCX 0 1\nM 2\nR 3\nTICK")
        used_indices = _get_used_stim_indices(circ)
        assert used_indices == [0, 1, 2, 3]
        
        # Test _reindex_stim_circuit with complex mapping
        index_map = {0: 1, 1: 0, 2: 2, 3: 3}
        reindexed = _reindex_stim_circuit(circ, index_map)
        circ_str = str(reindexed)
        assert "H 1" in circ_str
        assert "CX 1 0" in circ_str
        assert "M 2" in circ_str
        assert "R 3" in circ_str
        
        # Test with empty circuit
        empty_circ = stim.Circuit()
        empty_used = _get_used_stim_indices(empty_circ)
        assert empty_used == []
        
        empty_reindexed = _reindex_stim_circuit(empty_circ, {})
        assert str(empty_reindexed) == ""
    
    def test_init_edge_cases_comprehensive(self):
        # Test more edge cases in __init__
        
        # Test with stim.Circuit that has no qubits
        import stim
        empty_stim_circ = stim.Circuit()
        circ1 = STIMPhysicalCircuit(empty_stim_circ, [])
        assert circ1.qubit_labels == []
        assert circ1.circuit.num_qubits == 0
        
        # Test with stim.Circuit that has annotations only (no qubit targets)
        annot_circ = stim.Circuit("TICK")
        circ2 = STIMPhysicalCircuit(annot_circ, [])
        assert circ2.qubit_labels == []
        
        # Test copy constructor with different label types
        circ3 = STIMPhysicalCircuit("H 0\nTICK", [0])
        circ4 = STIMPhysicalCircuit(circ3, ['Q0'])
        assert circ4.qubit_labels == ['Q0']
        assert circ4.circuit.num_qubits == 1
    
    def test_method_properties(self):
        # Test various method properties and edge cases
        
        # Test depth property
        circ = STIMPhysicalCircuit("H 0\nTICK\nX 0\nTICK", ['Q0'])
        assert circ.depth == 3  # 2 TICKs create 3 layers
        
        # Test __str__ method
        circ_str = str(circ)
        assert "Physical STIM circuit" in circ_str
        assert "H 0" in circ_str
        
        # Test __repr__ method
        circ_repr = repr(circ)
        assert "Physical STIM circuit" in circ_repr
        
        # Test circuit property
        stim_circ = circ.circuit
        assert isinstance(stim_circ, stim.Circuit)
        assert stim_circ.num_qubits == 1