"""Creation of syndrome circuits from stabilizer plaquette templates.
"""

from typing import Iterable, Mapping, Optional, TypeAlias, Union

from pygsti.circuits import Circuit

class StabilizerPlaquetteFactory():
    def __init__(self, template_circuit_dict: Mapping[str, Circuit])-> None:
        self.template_circuit_dict = template_circuit_dict
    
    def get_circuit(self, template_key: str, qubit_labels: Iterable[Optional[str]],
                    omit_gates: Optional[Union[Iterable[str], str]] = None) -> Circuit:
        circuit = self.template_circuit_dict[template_key].copy(editable=True)

        # Either map the qubit or remove it if providing None
        # The latter feature is useful to get lower weight checks with the same schedule
        mapping = {}
        for template_qubit, qubit in zip(circuit.line_labels, qubit_labels):
            if qubit is None:
                circuit.delete_lines(template_qubit, delete_straddlers=True)
            else:
                mapping[template_qubit] = qubit
        
        circuit.map_state_space_labels_inplace(mapping)
        
        # Optionally omit some gates, e.g. a measurement if interested
        # in the syndrome preparation rather than extraction circuit
        if omit_gates is not None:
            if isinstance(omit_gates, str):
                omit_gates = [omit_gates]
            
            for gate in omit_gates:
                circuit.replace_gatename_with_idle_inplace(gate)
        
        circuit.done_editing()
        return circuit

    
ListStrMap: TypeAlias = Mapping[str, Iterable[str]]
class SyndromeCircuit():
    def __init__(self, factory: StabilizerPlaquetteFactory,
                 stabilizers: Union[ListStrMap, Iterable[ListStrMap]]) -> None:
        self.factory = factory
        if isinstance(stabilizers, (list, tuple)):
            self.stabilizer_stages = stabilizers
        else:
            # We only got one stage, make it a list for consistency
            self.stabilizer_stages = [stabilizers]
        
        for stage in self.stabilizer_stages:
            assert all([c in factory.template_circuit_dict.keys() for c in stage.keys()]), \
                "Not all provided checks can be created by the provided factory"
    
    def get_circuit(self, qubit_labels: Iterable[str],
                    omit_gates: Optional[Union[Iterable[str], str]] = None,
                    compress_within_stages: bool = False,
                    compress_between_stages: bool = False,
                    delete_idle_layers: bool = True) -> Circuit:
        if qubit_labels is None:
            qubit_labels = 'auto'
        
        circuit = Circuit([], line_labels=qubit_labels, editable=True)
        for stage in self.stabilizer_stages:
            stage_layers = []
            for i, (key, qubit_list) in enumerate(stage.items()):
                for j, qubits in enumerate(qubit_list):
                    check_circuit = self.factory.get_circuit(key, qubits, omit_gates=omit_gates)
                    if i == 0 and j == 0:
                        # Just take layers directly
                        stage_layers.extend([list(check_circuit.layer(j)) for j in range(check_circuit.num_layers)])
                    else:
                        # Concatenate layers
                        for k in range(check_circuit.num_layers):
                            if k > len(stage_layers) - 1:
                                stage_layers.append([])
                            
                            stage_layers[k] += list(check_circuit.layer(k))

            # TODO: I expect this to fail on a collision
            # Figure out what that error is and put a nice error message
            stage_circuit = Circuit(stage_layers, line_labels=qubit_labels)
            
            if compress_within_stages:
                stage_circuit = stage_circuit.parallelize()

            circuit.append_circuit_inplace(stage_circuit)

        if compress_between_stages:
            circuit = circuit.parallelize()
        if delete_idle_layers:
            if circuit._static:
                circuit = circuit.copy(editable=True)
            circuit.delete_idle_layers_inplace()
        circuit.done_editing()

        return circuit

    def map_stabilizer_keys(self, key_mapping: Mapping[str, str]) -> 'SyndromeCircuit':
        new_stabilizer_stages = []
        for stage in self.stabilizer_stages:
            new_stage = {key_mapping.get(k, k): v for k,v in stage.items()}
            new_stabilizer_stages.append(new_stage)
        
        return SyndromeCircuit(self.factory, new_stabilizer_stages)