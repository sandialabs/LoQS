""":class:`PyGSTiCircuitBackend` definition.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional, Type, TypeAlias, Union

from pygsti.baseobjs import Label as PyGSTiLabel
from pygsti.circuits import Circuit as PyGSTiCircuit

from loqs.backends import BaseCircuitBackend


class PyGSTiCircuitBackend(BaseCircuitBackend):
    """Circuit backend for handling pygsti.circuits.Circuit."""

    @property
    def name(self) -> str:
        return "pyGSTi"

    @property
    def CircuitType(self) -> Type:
        """PyGSTi backend circuit type (pygsti.circuits.Circuit)"""
        return PyGSTiCircuit

    @property
    def CircuitCastable(self) -> TypeAlias:
        """The types this backend is able to cast into a raw circuit object.

        The :class:`pygsti.circuits.Circuit`, a string representation of a circuit,
        and a list of operations/layers) are directly passable into
        :meth:`pygsti.circuits.Circuit.cast`.
        The tuple of list of operations/layers and list of qubits is a way to set
        the qubit labels after casting.
        """
        return Union[
            self.CircuitType,
            str,
            Iterable[self.OperationTypes],
            tuple[Iterable[self.OperationTypes], Iterable[self.QubitTypes]],
        ]

    @property
    def QubitTypes(self) -> TypeAlias:
        """PyGSTi backend qubit label types (strs and ints)"""
        return Union[str, int]

    @property
    def LayerTypes(self) -> TypeAlias:
        """Helper type alias for things allowed to be in circuit layer

        For PyGSTi, this would be:
        - strings, e.g. "Gcnot"
        - tuples or lists of strings with either one or several qubits,
            e.g. ('Gxpi2', 0) or ['Gcnot', ("Q0", "Q1")]
        - actual Labels (which is what all the above cast to anyway)
        - or lists and tuples of the above (in which case it is a whole layer)
        """
        return Union[
            str,  # e.g., gate names
            tuple[str, self.QubitTypes],  # e.g., gate name and one qubit
            list[str, self.QubitTypes],  # e.g., gate name and one qubit
            tuple[
                str, Iterable[self.QubitTypes]
            ],  # e.g., gate name and tuple of qubits
            list[
                str, Iterable[self.QubitTypes]
            ],  # e.g., gate name and tuple of qubits
            PyGSTiLabel,  # or an actual pyGSTi label
        ]

    @property
    def OperationTypes(self) -> TypeAlias:
        """PyGSTi backend operations type (one or several gates/layers)"""
        return Union[self.LayerTypes, Iterable[self.LayerTypes]]

    def cast_circuit(
        self,
        obj: PyGSTiCircuitBackend.CircuitCastable,
    ) -> CircuitType:
        def try_cast(circ_obj):
            try:
                circuit = PyGSTiCircuit.cast(circ_obj)
            except Exception as e:
                raise ValueError(
                    f"Failed to cast {obj} as a pyGSTi Circuit"
                ) from e
            return circuit

        if (
            isinstance(obj, tuple)
            and len(obj) == 2
            and isinstance(obj[0], Iterable[self.OperationTypes])
            and isinstance(obj[1], Iterable[self.QubitTypes])
        ):
            # Then we were passed a circuit and qubit labels
            circuit = try_cast(obj[0])
            circuit.line_labels = obj[1]
        else:
            # This is something Circuit.cast should be able to handle
            circuit = try_cast(obj)

        return circuit

    def copy_circuit(
        self, circuit: CircuitType, finalized: bool = False
    ) -> CircuitType:
        return circuit.copy(editable=not finalized)

    def delete_qubits(
        self, circuit: CircuitType, qubits_to_delete: Iterable[QubitTypes]
    ) -> CircuitType:
        return super().delete_qubits(circuit, qubits_to_delete)

    def delete_qubits_inplace(
        self,
        circuit: CircuitType,
        qubits_to_delete: Iterable[QubitTypes],
    ) -> None:
        circuit.delete_lines(qubits_to_delete, delete_straddlers=True)

    def finalize_circuit_inplace(self, circuit: CircuitType) -> None:
        circuit.done_editing()

    def get_qubit_labels(self, circuit: CircuitType) -> Iterable[QubitTypes]:
        return circuit.line_labels

    def map_qubit_labels(
        self,
        circuit: CircuitType,
        qubit_mapping: Mapping[QubitTypes, QubitTypes],
    ) -> CircuitType:
        return super().map_qubit_labels(circuit, qubit_mapping)

    def map_qubit_labels_inplace(
        self,
        circuit: CircuitType,
        qubit_mapping: Mapping[QubitTypes, QubitTypes],
    ) -> None:
        complete_mapping = {
            q: qubit_mapping.get(q, q) for q in circuit.line_labels
        }
        circuit.map_state_space_labels_inplace(complete_mapping)

    def set_qubit_labels(
        self, circuit: CircuitType, qubit_labels: Iterable[QubitTypes]
    ) -> CircuitType:
        return super().set_qubit_labels(circuit, qubit_labels)

    def set_qubit_labels_inplace(
        self,
        circuit: CircuitType,
        qubit_labels: Iterable[QubitTypes],
    ) -> None:
        circuit.line_labels = qubit_labels

    def process_circuit(
        self,
        circuit: CircuitType,
        qubit_labels: Optional[Iterable[QubitTypes]] = None,
        omit_gates: Optional[Iterable[OperationTypes]] = None,
        delete_idle_layers: bool = False,
    ) -> CircuitType:
        processed_circuit = circuit.copy(editable=True)

        if qubit_labels is not None:
            processed_circuit.line_labels = qubit_labels

        if omit_gates is None:
            omit_gates = {}
        else:
            omit_gates = {k: [] for k in omit_gates}
        processed_circuit.change_gate_library(
            omit_gates, depth_compression=False, allow_unchanged_gates=True
        )

        if delete_idle_layers:
            processed_circuit.delete_idle_layers_inplace()

        return processed_circuit
