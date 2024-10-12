"""Tester for loqs.core.instructions.instruction"""

from tempfile import NamedTemporaryFile

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
            'parameters': {'state': 0, 'qubits': ins.data["qubits"]}
        }
        assert result.log == "test result"

        # Check map works as expected
        ins2 = ins.map_qubits({"Q1": "A3", 'Q0': "D2"})
        result2 = ins2.apply(state=0, qubits=ins2.data["qubits"])
        assert result2._data == {
            'state': 1,
            'qubits': ["D2", "A3"],
            'instruction': ins2,
            'parameters': {'state': 0, 'qubits': ins2.data["qubits"]}
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
            "state_name_in_program": DEFAULT_PRIORITIES,
            "qubits": ["instruction"]
        }

    
    def test_serialization(self):
        def apply_fn(state, qubits):
            return Frame({"state": state+1, 'qubits': qubits})
        def map_qubits_fn(qubit_mapping, qubits, **kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs["qubits"] = [qubit_mapping[q] for q in qubits]
            return new_kwargs
    
        data= {"qubits": ["Q0", "Q1"]}

        ins = Instruction(apply_fn, data, map_qubits_fn, name="test")

        with NamedTemporaryFile("w+", suffix='.json') as tempf:
            ins.write(tempf.name)

            ins2 = Frame.read(tempf.name)
        
            result = ins2.apply(state=0, qubits=ins2.data["qubits"])
            assert result._data == {
                'state': 1,
                'qubits': ["Q0", "Q1"],
                'instruction': ins2,
                'parameters': {'state': 0, 'qubits': ins2.data["qubits"]}
            }
            assert result.log == "test result"

        # We should be able to do it a second time also
        # This is because we cache the serialization the first time,
        # so even though the second time we don't have access to the source code, it works
        with NamedTemporaryFile("w+", suffix='.json') as tempf:
            ins2.write(tempf.name)

            ins3 = Frame.read(tempf.name)
        
            result2 = ins3.apply(state=0, qubits=ins3.data["qubits"])
            assert result2._data == {
                'state': 1,
                'qubits': ["Q0", "Q1"],
                'instruction': ins3,
                'parameters': {'state': 0, 'qubits': ins3.data["qubits"]}
            }
            assert result2.log == "test result"
            