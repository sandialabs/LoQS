"""Tester for loqs.core.instructions.instruction"""

from loqs.core.frame import Frame
from loqs.core.instructions import Instruction
from loqs.core.instructions.instruction import DEFAULT_PRIORITIES, KwargDict

class TestInstruction:

    def test_init_apply_map(self):
        def apply_fn(state, qubits):
            return Frame({"state": state+1, 'qubits': qubits})
        def map_qubits_fn(qubit_mapping, qubits, **kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs["qubits"] = [qubit_mapping[q] for q in qubits]
            return new_kwargs
    
        data= {"qubits": ["Q0", "Q1"]}

        ins = Instruction(apply_fn, data, map_qubits_fn, name="test")
        
        # Check apply works as expected
        result = ins.apply(state=0, qubits=ins.data["qubits"])
        assert result._data == {
            'state': 1,
            'qubits': ["Q0", "Q1"],
            'instruction': ins,
            #'collected_params': {'state': 0, 'qubits': ins.data["qubits"]}
        }
        assert result.log == "test result"

        # Check map works as expected
        ins2 = ins.map_qubits({"Q1": "A3", 'Q0': "D2"})
        result2 = ins2.apply(state=0, qubits=ins2.data["qubits"])
        assert result2._data == {
            'state': 1,
            'qubits': ["D2", "A3"],
            'instruction': ins2,
            #'collected_params': {'state': 0, 'qubits': ins2.data["qubits"]}
        }
        assert result2.log == "test result"

        # Also check we can set param priorities and aliases
        # Full testing of these should happen in QuantumProgram though
        ins3 = Instruction(apply_fn, data, map_qubits_fn,
            param_aliases={"state": "state_name_in_program"},
            param_error_behavior="raise",
            param_priorities={"qubits": ["instruction"]},
            name="test")
        assert ins3.param_priorities == {
            "state": DEFAULT_PRIORITIES,
            "qubits": ["instruction"]
        }
        assert ins3.param_alias("state") == "state_name_in_program"

    
    def test_serialization(self, make_temp_path):
        def apply_fn(state, qubits):
            return Frame({"state": state+1, 'qubits': qubits})
        def map_qubits_fn(qubit_mapping, qubits, **kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs["qubits"] = [qubit_mapping[q] for q in qubits]
            return new_kwargs

        data= {"qubits": ["Q0", "Q1"]}

        ins = Instruction(apply_fn, data, map_qubits_fn, name="test")

        with make_temp_path(suffix='.json') as tmp_path:
            ins.write(tmp_path)
            ins2 = Instruction.read(tmp_path)
            assert isinstance(ins2, Instruction)

            result = ins2.apply(state=0, qubits=ins2.data["qubits"])
            assert result._data == {
                'state': 1,
                'qubits': ["Q0", "Q1"],
                'instruction': ins2,
                #'collected_params': {'state': 0, 'qubits': ins2.data["qubits"]}
            }
            assert result.log == "test result"

        # This is because we cache the serialization the first time,
        # so even though the second time we don't have access to the source code, it works
        with make_temp_path(suffix='.json') as tmp_path:
            ins2.write(tmp_path)
            ins3 = Frame.read(tmp_path)
            assert isinstance(ins3, Instruction)

            result2 = ins3.apply(state=0, qubits=ins3.data["qubits"])
            assert result2._data == {
                'state': 1,
                'qubits': ["Q0", "Q1"],
                'instruction': ins3,
                #'collected_params': {'state': 0, 'qubits': ins3.data["qubits"]}
            }
            assert result2.log == "test result"

    def test_instruction_serialization_comprehensive(self, make_temp_path):
        """Comprehensive test of Instruction serialization methods."""
        def apply_fn(state, qubits):
            return Frame({"state": state+1, 'qubits': qubits})
        def map_qubits_fn(qubit_mapping, qubits, **kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs["qubits"] = [qubit_mapping[q] for q in qubits]
            return new_kwargs

        data = {"qubits": ["Q0", "Q1"], "complex": {"nested": "data"}}
        ins = Instruction(apply_fn, data, map_qubits_fn, name="comprehensive_test")

        # Test dumps/loads roundtrip
        with make_temp_path(suffix=".json") as tmp_path:
            ins.write(tmp_path)
            loaded_ins = Instruction.read(tmp_path)
            assert isinstance(loaded_ins, Instruction)
            assert loaded_ins.name == "comprehensive_test"
            assert loaded_ins.data == data

        with make_temp_path(suffix='.json') as temp_path:
            ins.write(temp_path)
            loaded_ins = Instruction.read(temp_path)
            assert isinstance(loaded_ins, Instruction)
            assert loaded_ins.name == "comprehensive_test"
            assert loaded_ins.data == data

        with make_temp_path(suffix='.json.gz') as temp_path:
            ins.write(temp_path)
            loaded_ins = Instruction.read(temp_path)
            assert isinstance(loaded_ins, Instruction)
            assert loaded_ins.name == "comprehensive_test"
            assert loaded_ins.data == data

    def test_instruction_with_complex_data(self, make_temp_path):
        """Test Instruction serialization with complex nested data."""
        def apply_fn(state, qubits):
            return Frame({"state": state})

        complex_data = {
            "nested": {
                "deep": {"value": "test"},
                "list": [1, 2, {"inner": "dict"}]
            },
            "simple": "data"
        }

        ins = Instruction(apply_fn, complex_data, name="complex_data_test")

        # Test roundtrip preserves structure
        with make_temp_path(suffix=".json") as tmp_path:
            ins.write(tmp_path)
            loaded_ins = Instruction.read(tmp_path)
            assert isinstance(loaded_ins, Instruction)
            assert loaded_ins.data == complex_data

    def test_instruction_function_serialization(self, make_temp_path):
        """Test that apply and map_qubits functions are serialized correctly."""
        def apply_fn(state, qubits):
            return Frame({"result": f"state_{state}_qubits_{'_'.join(qubits)}"})

        def map_qubits_fn(qubit_mapping, qubits, **kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs["qubits"] = [qubit_mapping.get(q, q) for q in qubits]
            return new_kwargs

        data = {"qubits": ["A", "B"]}
        ins = Instruction(apply_fn, data, map_qubits_fn, name="function_test")

        # Test serialization and deserialization
        with make_temp_path(suffix=".json") as tmp_path:
            ins.write(tmp_path)
            loaded_ins = Instruction.read(tmp_path)
            assert isinstance(loaded_ins, Instruction)

        original_result = ins.apply(state=5, qubits=["A", "B"])
        loaded_result = loaded_ins.apply(state=5, qubits=["A", "B"])

        assert original_result._data["result"] == loaded_result._data["result"]

        # Test map_qubits functionality
        mapped_ins = loaded_ins.map_qubits({"A": "X", "B": "Y"})
        mapped_result = mapped_ins.apply(state=5, qubits=["X", "Y"])
        assert "X_Y" in mapped_result._data["result"]
            