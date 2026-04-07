"""Tester for loqs.codepacks.codepack_trivial_counter"""

import pytest

from loqs.core import Frame, Instruction, QuantumProgram
from loqs.codepacks import codepack_trivial_counter as trivial_codepack


class TestTrivialCounterCodepack:
    
    @classmethod
    def setup_class(cls):
        # Create the trivial codepack
        cls.trivial_code = trivial_codepack.create_qec_code()
        
    def test_create_qec_code(self):
        """Test that we can create the trivial QEC code."""
        code = self.trivial_code
        
        # Check that it's a QECCode instance
        assert hasattr(code, 'instructions')
        assert hasattr(code, 'template_qubits')
        assert hasattr(code, 'template_data_qubits')
        assert hasattr(code, 'name')
        
        # Check the name
        assert code.name == "Trivial Counter Code"
        
        # Check that we have the expected instructions
        expected_instructions = ["Increment", "Init Counter"]
        for instr_name in expected_instructions:
            assert instr_name in code.instructions
            assert isinstance(code.instructions[instr_name], Instruction)
        
        # Check that the Increment instruction has the correct data
        increment_instr = code.instructions["Increment"]
        assert "increment_by" in increment_instr.data
        assert increment_instr.data["increment_by"] == 1
        
        # Check qubits
        assert len(code.template_qubits) == 1
        assert code.template_qubits[0] == "Q0"
        assert code.template_data_qubits == ["Q0"]
    
    def test_increment_instruction(self):
        """Test the increment instruction."""
        increment_instr = self.trivial_code.instructions["Increment"]
        
        # Test with default increment_by value (1) - must provide increment_by parameter
        # Test with initial counter 0
        result_frame = increment_instr.apply(counter=0, increment_by=1)
        assert isinstance(result_frame, Frame)
        assert result_frame["counter"] == 1
        
        # Test with counter 5
        result_frame = increment_instr.apply(counter=5, increment_by=1)
        assert result_frame["counter"] == 6
        
        # Test with negative counter
        result_frame = increment_instr.apply(counter=-3, increment_by=1)
        assert result_frame["counter"] == -2
        
        # Test with custom increment_by value
        result_frame = increment_instr.apply(counter=0, increment_by=2)
        assert result_frame["counter"] == 2
        
        result_frame = increment_instr.apply(counter=5, increment_by=3)
        assert result_frame["counter"] == 8
        
        result_frame = increment_instr.apply(counter=-3, increment_by=5)
        assert result_frame["counter"] == 2
    
    def test_init_counter_instruction(self):
        """Test the init state instruction."""
        init_counter_instr = self.trivial_code.instructions["Init Counter"]
        
        # Test default initialization (should be 0)
        result_frame = init_counter_instr.apply(initial_value=0)
        assert isinstance(result_frame, Frame)
        assert result_frame["counter"] == 0
        
        # Test custom initialization
        result_frame = init_counter_instr.apply(initial_value=42)
        assert result_frame["counter"] == 42
        
        result_frame = init_counter_instr.apply(initial_value=-10)
        assert result_frame["counter"] == -10
    
    def test_ideal_model(self):
        """Test that the ideal model returns an empty DictNoiseModel."""
        qubits = ["Q0"]
        ideal_model = trivial_codepack.create_ideal_model(qubits)
        
        # Should return an empty DictNoiseModel for this trivial case
        from loqs.backends.model.dictmodel import DictNoiseModel
        assert isinstance(ideal_model, DictNoiseModel)
        # Check that the model has empty gate and instrument dicts
        assert ideal_model.gate_dict == {}
        assert ideal_model.inst_dict == {}
    
    def test_instruction_sequence(self):
        """Test a sequence of instructions to verify the counter behavior."""
        # Get the instructions
        init_instr = self.trivial_code.instructions["Init Counter"]
        increment_instr = self.trivial_code.instructions["Increment"]
        
        # Initialize to 0
        frame = init_instr.apply(initial_value=0)
        assert frame["counter"] == 0
        
        # Increment once
        frame = increment_instr.apply(counter=frame["counter"], increment_by=1)
        assert frame["counter"] == 1
        
        # Increment again
        frame = increment_instr.apply(counter=frame["counter"], increment_by=1)
        assert frame["counter"] == 2
        
        # Continue with the current counter
        current_counter = frame["counter"]
        
        # Continue incrementing with default increment_by (1)
        for i in range(3):
            frame = increment_instr.apply(counter=current_counter, increment_by=1)
            current_counter = frame["counter"]
        
        assert current_counter == 5
        
        # Test with custom increment_by value
        frame = increment_instr.apply(counter=current_counter, increment_by=2)
        current_counter = frame["counter"]
        assert current_counter == 7  # 5 + 2
        
        frame = increment_instr.apply(counter=current_counter, increment_by=3)
        current_counter = frame["counter"]
        assert current_counter == 10  # 7 + 3


def test_trivial_counter_integration():
    """Test basic integration of the trivial counter in a simple program."""
    # This is a more comprehensive test that simulates how the codepack
    # might be used in a real scenario
    
    code = trivial_codepack.create_qec_code()
    
    # Simulate a simple program that initializes and increments state
    instructions = code.instructions
    
    # Step 1: Initialize state to 0
    frame = instructions["Init Counter"].apply(initial_value=0)
    current_counter = frame["counter"]

    # Step 2: Increment 3 times
    for _ in range(3):
        frame = instructions["Increment"].apply(counter=current_counter, increment_by=1)
        current_counter = frame["counter"]

    # Should be 3 now
    assert current_counter == 3

    # Step 3: Increment 2 more times
    for _ in range(2):
        frame = instructions["Increment"].apply(counter=current_counter, increment_by=1)
        current_counter = frame["counter"]

    # Should be 5 now
    assert current_counter == 5
    
    print("Trivial counter integration test passed!")


if __name__ == "__main__":
    # Run the tests
    test_class = TestTrivialCounterCodepack()
    test_class.setup_class()
    
    test_class.test_create_qec_code()
    test_class.test_ideal_model()
    test_class.test_init_counter_instruction()
    test_class.test_increment_instruction()
    test_class.test_instruction_sequence()
    
    test_trivial_counter_integration()
    
    print("All trivial counter tests passed!")