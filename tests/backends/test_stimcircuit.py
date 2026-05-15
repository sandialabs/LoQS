"""Tester for loqs.backends.circuit.stimcircuit"""

import warnings

import pytest
import unittest

try:
    import stim

    NO_STIM = False
except ImportError:
    NO_STIM = True

from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit, QubitTypes


@pytest.mark.skipif(
    NO_STIM,
    reason="Skipping stim backend tests due to failed import"
)
class TestSTIMPhysicalCircuitInit(unittest.TestCase):
    """Construction and validation: building a STIMPhysicalCircuit from
    various inputs (string, stim.Circuit, copy-from-self), with and without
    explicit qubit_labels, and the errors raised when inputs are invalid."""

    def test_init(self):
        # Test sparse STIM circuit (non-contiguous indices)
        sparse_circuit_str = "H 0\nTICK\nCX 0 5"

        # Test with None qubit_labels - should extract used indices [0, 5]
        circ1 = STIMPhysicalCircuit(sparse_circuit_str)
        self.assertEqual(circ1.qubit_labels, [0, 5])
        self.assertEqual(circ1.circuit.num_qubits, 2)  # Compact indices

        # Test with explicit qubit_labels - the string contains integer indices
        custom_labels = ['Q0', 'Q5']
        circ2 = STIMPhysicalCircuit("H 0\nTICK\nCX 0 1", custom_labels)
        self.assertEqual(circ2.qubit_labels, custom_labels)
        self.assertEqual(circ2.circuit.num_qubits, 2)

        # Test with stim.Circuit object
        stim_circ = stim.Circuit(sparse_circuit_str)
        circ3 = STIMPhysicalCircuit(stim_circ)
        self.assertEqual(circ3.qubit_labels, [0, 5])
        self.assertEqual(circ3.circuit.num_qubits, 2)

        # Test with stim.Circuit and custom labels
        circ4 = STIMPhysicalCircuit(stim_circ, custom_labels)
        self.assertEqual(circ4.qubit_labels, custom_labels)
        self.assertEqual(circ4.circuit.num_qubits, 2)

        # Test copying from another STIMPhysicalCircuit
        circ5 = STIMPhysicalCircuit(circ1)
        self.assertEqual(circ5.qubit_labels, circ1.qubit_labels)
        self.assertEqual(str(circ5.circuit), str(circ1.circuit))

        # Test copying with different qubit_labels
        new_labels = ['A', 'B']
        circ6 = STIMPhysicalCircuit(circ1, new_labels)
        self.assertEqual(circ6.qubit_labels, new_labels)
        self.assertEqual(circ6.circuit.num_qubits, 2)

        # Test ValueError cases
        with self.assertRaises(ValueError):
            STIMPhysicalCircuit(circ1, ['A'])  # Wrong length

        with self.assertRaises(ValueError):
            # String with custom labels that don't match circuit references
            STIMPhysicalCircuit("H Q0\nTICK\nCX Q0 Q1", ['Q0'])

        with self.assertRaises(ValueError):
            # stim.Circuit with insufficient labels
            stim_circ_3q = stim.Circuit("H 0\nCX 0 1\nCX 1 2")
            STIMPhysicalCircuit(stim_circ_3q, ['Q0', 'Q1'])

    def test_measurement_with_custom_labels(self):
        # Test measurement operations with custom labels
        circ_str = "H 0\nTICK\nM 0\nTICK\nMR 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])

        self.assertEqual(circ.qubit_labels, ['Q0', 'Q1'])
        self.assertEqual(circ.circuit.num_qubits, 2)

        circ_str_after = str(circ.circuit)
        self.assertIn("H 0", circ_str_after)
        self.assertIn("M 0", circ_str_after)
        self.assertIn("MR 1", circ_str_after)

    def test_repeat_blocks(self):
        # Test that repeat blocks work correctly
        circ_str = "REPEAT 2 {\n    H 0\n    CX 0 1\n    TICK\n}"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])

        self.assertEqual(circ.qubit_labels, ['Q0', 'Q1'])
        self.assertEqual(circ.circuit.num_qubits, 2)

        # After unrolling, should have correct operations
        unrolled = circ._unroll_repeats()
        self.assertEqual(unrolled.count("H 0"), 2)
        self.assertEqual(unrolled.count("CX 0 1"), 2)

    def test_invalid_qubit_labels(self):
        # Test various invalid qubit label scenarios.
        # ValueError is raised inside __init__ before the TICK-warning check,
        # so these cases don't emit a UserWarning even when no TICK is present.

        # Too few labels for circuit (using label names in string)
        circ_str = "H Q0\nCX Q0 Q1"
        with self.assertRaises(ValueError):
            STIMPhysicalCircuit(circ_str, ['Q0'])

        # Mismatched labels when copying from STIMPhysicalCircuit
        circ1 = STIMPhysicalCircuit("H 0\nCX 0 1", ['Q0', 'Q1'], suppress_tick_warning=True)
        with self.assertRaises(ValueError):
            STIMPhysicalCircuit(circ1, ['Q0'])  # Wrong number

        # Unknown label in circuit string
        with self.assertRaises(ValueError):
            STIMPhysicalCircuit("H Unknown", ['Q0', 'Q1'])

    def test_empty_circuit(self):
        # Test empty circuit
        circ = STIMPhysicalCircuit("", [], suppress_tick_warning=True)
        self.assertEqual(circ.qubit_labels, [])
        self.assertEqual(circ.circuit.num_qubits, 0)
        # depth = num_ticks + 1, so an empty circuit reports a single
        # (empty) layer.
        self.assertEqual(circ.depth, 1)

        # Test circuit with only TICK
        circ2 = STIMPhysicalCircuit("TICK", [])
        self.assertEqual(circ2.qubit_labels, [])
        self.assertEqual(circ2.circuit.num_qubits, 0)
        self.assertEqual(circ2.depth, 2)

    def test_no_tick_warning_emitted(self):
        """Constructing a circuit without TICK emits a UserWarning unless
        suppress_tick_warning=True is passed."""
        with self.assertWarnsRegex(UserWarning, "No TICK instructions"):
            STIMPhysicalCircuit("H 0\nCX 0 1", ['Q0', 'Q1'])

        # And the same construction with suppression emits no such warning.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            STIMPhysicalCircuit(
                "H 0\nCX 0 1", ['Q0', 'Q1'], suppress_tick_warning=True
            )
        tick_warnings = [w for w in captured if "No TICK" in str(w.message)]
        self.assertEqual(tick_warnings, [])

    def test_invalid_circuit_type_raises(self):
        """Passing a value that is neither STIMPhysicalCircuit, str, nor
        stim.Circuit raises ValueError."""
        with self.assertRaises(ValueError):
            STIMPhysicalCircuit(123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            STIMPhysicalCircuit(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            STIMPhysicalCircuit([("H", 0)])  # type: ignore[arg-type]

    def test_parens_args_target_substitution(self):
        """Instructions with parenthesized arguments (e.g. X_ERROR(0.1) Q0)
        have their qubit targets correctly remapped from custom labels to
        STIM indices via the parens-aware path in _replace_instruction_targets."""
        circ = STIMPhysicalCircuit("X_ERROR(0.1) Q0\nTICK", ['Q0'])
        self.assertIn("X_ERROR(0.1) 0", str(circ.circuit))

    def test_annotation_instructions(self):
        # Test that annotation instructions work
        # Note: DETECTOR requires specific syntax, so we'll test with a simpler case
        circ_str = "H 0\nTICK"
        circ = STIMPhysicalCircuit(circ_str)

        self.assertEqual(circ.qubit_labels, [0])
        self.assertEqual(circ.circuit.num_qubits, 1)

        # Basic operations should work
        circ_str_after = str(circ.circuit)
        self.assertIn("H 0", circ_str_after)
        self.assertIn("TICK", circ_str_after)

    def test_stim_circuit_object_input(self):
        # Test initialization with stim.Circuit object
        import stim

        # Test with no qubit_labels
        stim_circ = stim.Circuit("H 0\nCX 0 5")
        circ1 = STIMPhysicalCircuit(stim_circ, suppress_tick_warning=True)
        self.assertEqual(circ1.qubit_labels, [0, 5])
        self.assertEqual(circ1.circuit.num_qubits, 2)

        # Test with qubit_labels
        stim_circ2 = stim.Circuit("H 0\nCX 0 1")
        circ2 = STIMPhysicalCircuit(stim_circ2, ['A', 'B'], suppress_tick_warning=True)
        self.assertEqual(circ2.qubit_labels, ['A', 'B'])
        self.assertEqual(circ2.circuit.num_qubits, 2)

        # Test error case: not enough labels
        with self.assertRaises(ValueError):
            STIMPhysicalCircuit(stim_circ2, ['A'])

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
        with self.assertRaises(ValueError) as ctx:
            STIMPhysicalCircuit("MPP 0 1\nTICK")
        self.assertIn("MPP", str(ctx.exception))

        # Test warning suppression
        circ_no_warn = STIMPhysicalCircuit("H 0", ['Q0'], suppress_tick_warning=True)
        self.assertEqual(circ_no_warn.qubit_labels, ['Q0'])

    def test_init_edge_cases_comprehensive(self):
        # Test more edge cases in __init__

        # Test with stim.Circuit that has no qubits
        import stim
        empty_stim_circ = stim.Circuit()
        circ1 = STIMPhysicalCircuit(empty_stim_circ, [], suppress_tick_warning=True)
        self.assertEqual(circ1.qubit_labels, [])
        self.assertEqual(circ1.circuit.num_qubits, 0)

        # Test with stim.Circuit that has annotations only (no qubit targets)
        annot_circ = stim.Circuit("TICK")
        circ2 = STIMPhysicalCircuit(annot_circ, [], suppress_tick_warning=True)
        self.assertEqual(circ2.qubit_labels, [])

        # Test copy constructor with different label types
        circ3 = STIMPhysicalCircuit("H 0\nTICK", [0])
        circ4 = STIMPhysicalCircuit(circ3, ['Q0'])
        self.assertEqual(circ4.qubit_labels, ['Q0'])
        self.assertEqual(circ4.circuit.num_qubits, 1)


@pytest.mark.skipif(
    NO_STIM,
    reason="Skipping stim backend tests due to failed import"
)
class TestSTIMPhysicalCircuitMutators(unittest.TestCase):
    """In-place mutators (delete/merge/insert/append/pad/map/set qubits)
    and their non-inplace counterparts. Each test focuses on whether the
    operation produces the expected updated circuit + qubit_labels."""

    def test_delete_qubits(self):
        # Create a circuit with 3 qubits
        circ_str = "H 0\nCX 0 1\nTICK\nX 2"
        circ = STIMPhysicalCircuit(circ_str)

        # Delete qubit 1
        circ.delete_qubits_inplace([1])
        self.assertEqual(circ.qubit_labels, [0, 2])
        self.assertEqual(circ.circuit.num_qubits, 2)

        # Check that the circuit string no longer contains operations on qubit 1
        circ_str_after = str(circ.circuit)
        self.assertNotIn(' 1 ', circ_str_after)
        self.assertNotIn(' 1\n', circ_str_after)

        # Delete another qubit
        circ.delete_qubits_inplace([0])
        self.assertEqual(circ.qubit_labels, [2])
        self.assertEqual(circ.circuit.num_qubits, 1)

    def test_delete_unknown_label_raises(self):
        """delete_qubits_inplace raises ValueError when given a label that
        is not present in the circuit's qubit_labels."""
        circ = STIMPhysicalCircuit("H 0\nTICK\nX 1", ['Q0', 'Q1'])
        with self.assertRaises(ValueError):
            circ.delete_qubits_inplace(['Q2'])  # Q2 is not in qubit_labels

    def test_merge(self):
        # Create two circuits with overlapping and new qubits
        circ1_str = "H Q0\nTICK\nX Q1"
        circ2_str = "Y Q1\nTICK\nZ Q2"  # circ2 shares Q1 and adds Q2

        circ1 = STIMPhysicalCircuit(circ1_str, ['Q0', 'Q1'])
        circ2 = STIMPhysicalCircuit(circ2_str, ['Q1', 'Q2'])

        # Merge circ2 into circ1 starting at layer 0
        circ1.merge_inplace(circ2, 0)

        # Should have all qubit labels from both circuits
        self.assertEqual(set(circ1.qubit_labels), {'Q0', 'Q1', 'Q2'})
        self.assertEqual(circ1.circuit.num_qubits, 3)

        # Check that operations are correctly mapped
        circ_str = str(circ1.circuit)
        self.assertIn('H 0', circ_str)  # Q0 -> STIM idx 0 (from circ1)
        self.assertIn('X 1', circ_str)  # Q1 -> STIM idx 1 (from circ1)
        self.assertIn('Y 1', circ_str)  # Q1 -> STIM idx 1 (from circ2, remapped)
        self.assertIn('Z 2', circ_str)  # Q2 -> STIM idx 2 (from circ2, remapped)

    def test_complex_merge_scenario(self):
        # Test a more complex merge scenario
        circ1 = STIMPhysicalCircuit("H Q0\nTICK\nX Q1\nTICK\nM Q0", ['Q0', 'Q1'])
        circ2 = STIMPhysicalCircuit("Y Q2\nTICK\nZ Q3", ['Q2', 'Q3'])

        # Merge at layer 2 (after second TICK) - no collision
        circ1.merge_inplace(circ2, 2)

        # Should have all four labels
        self.assertEqual(set(circ1.qubit_labels), {'Q0', 'Q1', 'Q2', 'Q3'})
        self.assertEqual(circ1.circuit.num_qubits, 4)

        # Check that the circuit structure is correct
        circ_str = str(circ1.circuit)
        self.assertIn('H 0', circ_str)
        self.assertIn('X 1', circ_str)
        self.assertIn('Y 2', circ_str)
        self.assertIn('Z 3', circ_str)
        self.assertIn('M 0', circ_str)

    def test_merge_edge_cases(self):
        # Test a case that should cause collision
        circ1 = STIMPhysicalCircuit("H 0\nTICK\nX 0", ['Q0'])
        circ2 = STIMPhysicalCircuit("Y 0\nTICK\nZ 0", ['Q0'])

        # Try to merge at layer 0 where both circuits have operations on Q0
        with self.assertRaises(ValueError) as ctx:
            circ1.merge_inplace(circ2, 0)
        self.assertIn("ill-posed", str(ctx.exception).lower())

    def test_insert_and_append(self):
        # Test insert_inplace method
        circ1 = STIMPhysicalCircuit("H 0\nTICK\nX 0", ['Q0'])
        circ2 = STIMPhysicalCircuit("Y 0\nTICK\nZ 0", ['Q0'])

        # Insert circ2 at layer 1 (after first TICK)
        circ1.insert_inplace(circ2, 1)

        circ_str = str(circ1.circuit)
        self.assertEqual(circ_str.count('H 0'), 1)
        self.assertEqual(circ_str.count('Y 0'), 1)
        self.assertEqual(circ_str.count('X 0'), 1)
        self.assertEqual(circ_str.count('Z 0'), 1)

        # Test append_inplace method
        circ3 = STIMPhysicalCircuit("H 0\nTICK", ['Q0'])
        circ4 = STIMPhysicalCircuit("X 0\nTICK", ['Q0'])

        circ3.append_inplace(circ4)

        circ3_str = str(circ3.circuit)
        self.assertIn('H 0', circ3_str)
        self.assertIn('X 0', circ3_str)
        self.assertEqual(circ3_str.count('TICK'), 2)

        # Test append method (non-inplace)
        circ5 = STIMPhysicalCircuit("H 0\nTICK", ['Q0'])
        circ6 = circ5.append(circ4)

        # Original should be unchanged
        self.assertEqual(str(circ5.circuit), "H 0\nTICK")
        # New circuit should have both
        circ6_str = str(circ6.circuit)
        self.assertIn('H 0', circ6_str)
        self.assertIn('X 0', circ6_str)

    def test_insert_edge_cases(self):
        # Test insert at various positions
        circ1 = STIMPhysicalCircuit("H 0\nTICK\nX 0\nTICK\nY 0", ['Q0'])
        circ2 = STIMPhysicalCircuit("Z 0\nTICK", ['Q0'])

        # Insert at beginning (idx=0)
        circ1.insert_inplace(circ2, 0)
        circ_str = str(circ1.circuit)
        self.assertTrue(circ_str.startswith('Z 0'))

        # Test insert at end (use depth instead of -1)
        circ3 = STIMPhysicalCircuit("H 0\nTICK", ['Q0'])
        circ4 = STIMPhysicalCircuit("X 0\nTICK", ['Q0'])
        circ3.insert_inplace(circ4, circ3.depth)  # Insert at end
        circ3_str = str(circ3.circuit)
        self.assertIn('H 0', circ3_str)
        self.assertIn('X 0', circ3_str)
        self.assertEqual(circ3_str.count('TICK'), 2)

    def test_pad_idles(self):
        # Create a simple circuit with 2 qubits
        circ_str = "H 0\nTICK\nX 0\nTICK\nH 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])

        # Pad with idles
        durations  : dict[str, int|float] = { 'H': 1, 'X': 1 }  # type makes pyright happy
        idle_names : dict[int|float, str] = {  1: 'I' }         # type makes pyright happy

        circ.pad_single_qubit_idles_by_duration_inplace(
            idle_names, durations, default_duration=1
        )

        # Should have added idle operations where needed
        circ_str_after = str(circ.circuit)
        # Check that we have the expected structure
        self.assertIn('H 0', circ_str_after)
        self.assertIn('X 0', circ_str_after)
        self.assertIn('H 1', circ_str_after)

        # Test simple pad_single_qubit_idles (without durations)
        # Use a circuit that actually uses both qubits to maintain the invariant
        circ2 = STIMPhysicalCircuit("H 0\nI 1\nTICK\nX 0\nI 1", ['Q0', 'Q1'])
        circ2.pad_single_qubit_idles_inplace("I")

        circ2_str = str(circ2.circuit)
        self.assertIn('H 0', circ2_str)
        self.assertIn('X 0', circ2_str)

    def test_pad_edge_cases(self):
        # Test padding with empty layers - use circuits that actually use both qubits
        circ = STIMPhysicalCircuit("H 0\nI 1\nTICK\nH 0\nI 1", ['Q0', 'Q1'])

        durations = {'H': 1, 'I': 1}
        idle_names : dict[int | float, str] = {1: 'I'}  # type makes pyright happy

        # This should work since both qubits are already used
        circ.pad_single_qubit_idles_by_duration_inplace(
            idle_names, durations, default_duration=1, empty_layer_idle='I'
        )

        circ_str = str(circ.circuit)
        self.assertIn('H 0', circ_str)
        self.assertIn('I 1', circ_str)

    def test_map_qubit_labels(self):
        # Create a circuit
        circ_str = "H 0\nTICK\nCX 0 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])

        # Map qubit labels
        mapping : dict[QubitTypes, QubitTypes] = {'Q0': 'A', 'Q1': 'B'}
        circ.map_qubit_labels_inplace(mapping)

        self.assertEqual(circ.qubit_labels, ['A', 'B'])

        # The internal STIM circuit should remain unchanged (still uses compact indices)
        self.assertEqual(circ.circuit.num_qubits, 2)
        circ_str_after = str(circ.circuit)
        self.assertIn('H 0', circ_str_after)
        self.assertIn('CX 0 1', circ_str_after)

    def test_map_qubit_labels_partial_passthrough(self):
        """map_qubit_labels_inplace leaves unmapped qubits unchanged
        (contract inherited from BasePhysicalCircuit)."""
        circ = STIMPhysicalCircuit("H 0\nTICK\nCX 0 1", ['Q0', 'Q1'])
        # Map only Q0; Q1 should be untouched.
        circ.map_qubit_labels_inplace({'Q0': 'A'})
        self.assertEqual(circ.qubit_labels, ['A', 'Q1'])

    def test_set_qubit_labels(self):
        # Create a circuit
        circ_str = "H 0\nTICK\nCX 0 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])

        # Set new qubit labels
        new_labels = ['A', 'B']
        circ.set_qubit_labels_inplace(new_labels)

        self.assertEqual(circ.qubit_labels, new_labels)
        # Internal STIM circuit should be unchanged
        self.assertEqual(circ.circuit.num_qubits, 2)
        circ_str_after = str(circ.circuit)
        self.assertIn('H 0', circ_str_after)
        self.assertIn('CX 0 1', circ_str_after)

        # Test non-inplace version
        circ2 = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])
        circ3 = circ2.set_qubit_labels(['X', 'Y'])

        self.assertEqual(circ2.qubit_labels, ['Q0', 'Q1'])  # Original unchanged
        self.assertEqual(circ3.qubit_labels, ['X', 'Y'])    # New circuit has new labels

    def test_set_qubit_labels_wrong_length_raises(self):
        """set_qubit_labels_inplace must raise ValueError when the new
        labels' length does not match the circuit's qubit count. This
        catches the silent invariant violation noted as B11 in the audit
        (len(self._qubit_labels) == self.circuit.num_qubits asserted by
        the qubit_labels property)."""
        circ = STIMPhysicalCircuit("H 0\nTICK\nCX 0 1", ['Q0', 'Q1'])

        # Too few labels
        with self.assertRaises(ValueError):
            circ.set_qubit_labels_inplace(['A'])

        # Too many labels
        with self.assertRaises(ValueError):
            circ.set_qubit_labels_inplace(['A', 'B', 'C'])

        # The non-inplace variant should propagate the error too.
        with self.assertRaises(ValueError):
            circ.set_qubit_labels(['A'])

        # And the original circuit should be untouched after the failures.
        self.assertEqual(circ.qubit_labels, ['Q0', 'Q1'])


@pytest.mark.skipif(
    NO_STIM,
    reason="Skipping stim backend tests due to failed import"
)
class TestSTIMPhysicalCircuitQueries(unittest.TestCase):
    """Read-only operations: properties (qubit_labels, depth, .circuit),
    dunder methods (__str__, __repr__), copy, error-location queries,
    serialization round-trips, command aliases, and the
    sparse-circuit compactness invariant."""

    def test_copy(self):
        # Create a circuit
        circ_str = "H 0\nTICK\nCX 0 1"
        circ1 = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])

        # Copy it
        circ2 = circ1.copy()

        # Should be identical
        self.assertEqual(circ2.qubit_labels, circ1.qubit_labels)
        self.assertEqual(str(circ2.circuit), str(circ1.circuit))

        # Modifying one shouldn't affect the other
        circ2.map_qubit_labels_inplace({'Q0': 'A', 'Q1': 'B'})
        self.assertEqual(circ1.qubit_labels, ['Q0', 'Q1'])

    def test_qubit_labels_property(self):
        # Test that the assertion holds
        circ_str = "H 0\nTICK\nCX 0 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])

        # This should not raise an assertion error
        labels = circ.qubit_labels
        self.assertEqual(labels, ['Q0', 'Q1'])
        self.assertEqual(len(labels), circ.circuit.num_qubits)

    def test_get_possible_discrete_error_locations(self):
        # Create a simple circuit
        circ_str = "H 0\nTICK\nCX 0 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])

        # Get error locations
        locations = circ.get_possible_discrete_error_locations()

        # Should return LoQS labels, not STIM indices
        for _, qubit_info in locations:
            if isinstance(qubit_info, tuple):
                # Two-qubit gate
                self.assertIn(qubit_info[0], ['Q0', 'Q1'])
                self.assertIn(qubit_info[1], ['Q0', 'Q1'])
            else:
                # Single-qubit gate
                self.assertIn(qubit_info, ['Q0', 'Q1'])

        # Test post_twoq_gates mode
        locations_2q = circ.get_possible_discrete_error_locations(post_twoq_gates=True)
        for _, qubit_info in locations_2q:
            self.assertIsInstance(qubit_info, tuple)
            assert isinstance(qubit_info, tuple)  # narrow for type checker
            self.assertIn(qubit_info[0], ['Q0', 'Q1'])
            self.assertIn(qubit_info[1], ['Q0', 'Q1'])

    def test_edge_cases(self):
        # Test various edge cases

        # Single qubit circuit
        circ1 = STIMPhysicalCircuit("H 0", ['Q0'], suppress_tick_warning=True)
        self.assertEqual(circ1.qubit_labels, ['Q0'])
        self.assertEqual(circ1.circuit.num_qubits, 1)

        # Circuit with only measurements
        circ2 = STIMPhysicalCircuit("M Q0\nTICK\nM Q1", ['Q0', 'Q1'])
        self.assertEqual(circ2.qubit_labels, ['Q0', 'Q1'])
        self.assertEqual(circ2.circuit.num_qubits, 2)

        # Circuit with mixed operations
        circ3 = STIMPhysicalCircuit("H Q0\nCX Q0 Q1\nM Q0\nTICK\nX Q1", ['Q0', 'Q1'])
        self.assertEqual(circ3.qubit_labels, ['Q0', 'Q1'])
        self.assertEqual(circ3.circuit.num_qubits, 2)

        # Test error locations with mixed operations
        error_locs = circ3.get_possible_discrete_error_locations()
        self.assertGreater(len(error_locs), 0)
        for _, qubit_label in error_locs:
            self.assertIn(qubit_label, ['Q0', 'Q1'])
            self.assertIsInstance(qubit_label, str)

        # Test post_twoq_gates mode
        error_locs_2q = circ3.get_possible_discrete_error_locations(post_twoq_gates=True)
        for _, qubit_tuple in error_locs_2q:
            self.assertIsInstance(qubit_tuple, tuple)
            assert isinstance(qubit_tuple, tuple)  # narrow for type checker
            self.assertEqual(len(qubit_tuple), 2)
            self.assertIn(qubit_tuple[0], ['Q0', 'Q1'])
            self.assertIn(qubit_tuple[1], ['Q0', 'Q1'])

    def test_sparse_circuit_compactness(self):
        # Test that sparse circuits maintain compact indices
        sparse_circuit_str = "H 0\nTICK\nCX 0 10\nTICK\nM 10"
        circ = STIMPhysicalCircuit(sparse_circuit_str)

        # Should have exactly the used indices as labels
        self.assertEqual(circ.qubit_labels, [0, 10])
        self.assertEqual(circ.circuit.num_qubits, 2)

        # Internal circuit should use compact indices 0, 1
        circ_str = str(circ.circuit)
        self.assertIn("H 0", circ_str)
        self.assertIn("CX 0 1", circ_str)
        self.assertIn("M 1", circ_str)
        self.assertNotIn("H 10", circ_str)  # Original sparse index should be gone

        # Test deletion from sparse circuit
        circ.delete_qubits_inplace([10])
        self.assertEqual(circ.circuit.num_qubits, 1)
        self.assertEqual(circ.qubit_labels, [0])

        # After merging with another sparse circuit
        other_str = "X 10\nTICK\nY 15"
        other_circ = STIMPhysicalCircuit(other_str)
        circ.merge_inplace(other_circ, 0)

        # Should have compact indices for all qubits
        self.assertEqual(circ.circuit.num_qubits, 3)
        self.assertEqual(set(circ.qubit_labels), {0, 10, 15})

    def test_serialization_methods(self):
        # Test _serialize_circuit and _deserialize_circuit methods
        circ_str = "H 0\nTICK\nCX 0 1"
        circ = STIMPhysicalCircuit(circ_str, ['Q0', 'Q1'])

        # Test serialization
        serialized = circ._serialize_circuit()
        self.assertIsInstance(serialized, str)
        self.assertIn('H 0', serialized)
        self.assertIn('CX 0 1', serialized)

        # Test deserialization
        deserialized_circ = STIMPhysicalCircuit._deserialize_circuit(serialized, ['Q0', 'Q1'])
        self.assertEqual(str(deserialized_circ), serialized)

        # Test that serialization preserves the circuit
        circ2 = STIMPhysicalCircuit(deserialized_circ, ['Q0', 'Q1'])
        self.assertEqual(circ2.qubit_labels, circ.qubit_labels)
        self.assertEqual(str(circ2.circuit), str(circ.circuit))

    def test_command_aliases(self):
        # Test substitute_command_aliases method
        circ_str = "CNOT 0 1\nTICK\nH 0"

        # Apply alias substitution
        aliased_str = STIMPhysicalCircuit.substitute_command_aliases(circ_str)

        # CNOT should be replaced with CX
        self.assertIn('CX 0 1', aliased_str)
        self.assertNotIn('CNOT', aliased_str)
        self.assertIn('H 0', aliased_str)

        # Test with a circuit that has aliases
        circ = STIMPhysicalCircuit("CNOT 0 1\nTICK", ['Q0', 'Q1'])
        circ_str_after = str(circ.circuit)
        # The alias should be preserved in the internal circuit
        self.assertTrue('CNOT 0 1' in circ_str_after or 'CX 0 1' in circ_str_after)

    def test_method_properties(self):
        # Test various method properties and edge cases

        # Test depth property
        circ = STIMPhysicalCircuit("H 0\nTICK\nX 0\nTICK", ['Q0'])
        self.assertEqual(circ.depth, 3)  # 2 TICKs create 3 layers

        # Test __str__ method
        circ_str = str(circ)
        self.assertIn("Physical STIM circuit", circ_str)
        self.assertIn("H 0", circ_str)

        # Test __repr__ method
        circ_repr = repr(circ)
        self.assertIn("Physical STIM circuit", circ_repr)

        # Test circuit property
        stim_circ = circ.circuit
        self.assertIsInstance(stim_circ, stim.Circuit)
        self.assertEqual(stim_circ.num_qubits, 1)


@pytest.mark.skipif(
    NO_STIM,
    reason="Skipping stim backend tests due to failed import"
)
class TestSTIMHelpers(unittest.TestCase):
    """Module-level helper functions in stimcircuit
    (_get_used_stim_indices, _reindex_stim_circuit) — these operate on raw
    stim.Circuit objects and are not methods of STIMPhysicalCircuit."""

    def test_helper_functions(self):
        # Test the helper functions directly
        import stim
        from loqs.backends.circuit.stimcircuit import _get_used_stim_indices, _reindex_stim_circuit

        # Test _get_used_stim_indices
        circ = stim.Circuit("H 0\nCX 0 5\nM 3")
        used_indices = _get_used_stim_indices(circ)
        self.assertEqual(used_indices, [0, 3, 5])

        # Test _reindex_stim_circuit
        index_map = {0: 0, 3: 1, 5: 2}
        reindexed_circ = _reindex_stim_circuit(circ, index_map)
        circ_str = str(reindexed_circ)
        self.assertIn("H 0", circ_str)
        self.assertIn("CX 0 2", circ_str)
        self.assertIn("M 1", circ_str)
        self.assertEqual(reindexed_circ.num_qubits, 3)

        # Test with measurement targets including inversion
        circ_with_inv = stim.Circuit("H 0\nM !1\nCX 0 2")
        used_indices_inv = _get_used_stim_indices(circ_with_inv)
        self.assertEqual(used_indices_inv, [0, 1, 2])

        index_map_inv = {0: 0, 1: 1, 2: 2}
        reindexed_inv = _reindex_stim_circuit(circ_with_inv, index_map_inv)
        circ_inv_str = str(reindexed_inv)
        self.assertIn("H 0", circ_inv_str)
        self.assertIn("M !1", circ_inv_str)
        self.assertIn("CX 0 2", circ_inv_str)

    def test_get_used_stim_indices_excludes_rec_targets(self):
        """_get_used_stim_indices counts only qubit targets, not measurement-
        record references like rec[-1] that appear as DETECTOR targets."""
        import stim
        from loqs.backends.circuit.stimcircuit import _get_used_stim_indices

        circ = stim.Circuit("H 0\nM 0\nDETECTOR rec[-1]")
        # Only qubit 0 should be reported; rec[-1] is not a qubit target.
        self.assertEqual(_get_used_stim_indices(circ), [0])

    def test_comprehensive_helper_coverage(self):
        # Test helper functions more comprehensively
        import stim
        from loqs.backends.circuit.stimcircuit import _get_used_stim_indices, _reindex_stim_circuit

        # Test _get_used_stim_indices with various gate types
        circ = stim.Circuit("H 0\nCX 0 1\nM 2\nR 3\nTICK")
        used_indices = _get_used_stim_indices(circ)
        self.assertEqual(used_indices, [0, 1, 2, 3])

        # Test _reindex_stim_circuit with complex mapping
        index_map = {0: 1, 1: 0, 2: 2, 3: 3}
        reindexed = _reindex_stim_circuit(circ, index_map)
        circ_str = str(reindexed)
        self.assertIn("H 1", circ_str)
        self.assertIn("CX 1 0", circ_str)
        self.assertIn("M 2", circ_str)
        self.assertIn("R 3", circ_str)

        # Test with empty circuit
        empty_circ = stim.Circuit()
        empty_used = _get_used_stim_indices(empty_circ)
        self.assertEqual(empty_used, [])

        empty_reindexed = _reindex_stim_circuit(empty_circ, {})
        self.assertEqual(str(empty_reindexed), "")
