"""TODO
"""

from __future__ import annotations

from collections.abc import Sequence, Mapping
from typing import ClassVar, TypeAlias

from loqs.backends.circuit import BasePhysicalCircuit


## Type aliases for static type checking
CastableTypes: TypeAlias = "BasePhysicalCircuit | Sequence[OperationTypes]"
"""Types we can cast to a built-in circuit.
"""

QubitTypes: TypeAlias = str | int
"""Qubit types for builtins"""

LabelType: TypeAlias = tuple[str, QubitTypes | Sequence[QubitTypes]]
"""Helper type alias for things allowed to be a label"""

OperationTypes: TypeAlias = LabelType | Sequence[LabelType]
"""Type alias for things allowed to be in circuit layer
"""


class ListPhysicalCircuit(BasePhysicalCircuit):
    """Circuit backend using built-in Python objects."""

    CastableTypes: ClassVar[TypeAlias] = CastableTypes

    _circuit: list[list[tuple[str, list[QubitTypes]]]]
    """List of layers (which are lists of 2-tuples with (name, qubits))
    """

    _qubit_labels: list[QubitTypes]
    """List of qubit labels"""

    def __init__(
        self,
        circuit: CastableTypes,
        qubit_labels: Sequence[QubitTypes] | None = None,
    ) -> None:
        from loqs.backends.circuit import PyGSTiPhysicalCircuit

        if isinstance(circuit, PyGSTiPhysicalCircuit):
            try:
                circuit = PyGSTiPhysicalCircuit.cast(circuit)

                layers = []
                for layer in circuit.circuit.layertup:  # type: ignore
                    new_layer = []
                    for comp in layer.components:  # type: ignore
                        new_layer.append((comp.name, comp.qubits))
                    layers.append(new_layer)
                self._circuit = layers

                if qubit_labels is None:
                    qubit_labels = circuit.circuit.line_labels
            except ImportError as e:
                raise ValueError("Could not ") from e
        else:
            assert isinstance(circuit, Sequence), "Expecting a list of layers"
            seen_qubits = set()

            def process_label(label) -> tuple[str, list[QubitTypes]]:
                new_label = (
                    (label[0], [label[1]])
                    if not isinstance(label[1], Sequence)
                    else label
                )

                assert len(new_label) == 2, "Labels must be 2-tuples"
                assert isinstance(
                    new_label[0], str
                ), "Label name must be a str"
                assert all(
                    [
                        isinstance(qubit_label, (str, int))
                        for qubit_label in new_label[1]
                    ]
                )

                for qubit_label in new_label[1]:
                    seen_qubits.add(qubit_label)

                return new_label

            for layer in circuit:
                if isinstance(layer, tuple):
                    # If we have a single tuple, promote it to a length-1 layer
                    self._circuit.append([process_label(layer)])
                else:
                    self._circuit.append([process_label(lbl) for lbl in layer])

            if qubit_labels is None:
                qubit_labels = list(seen_qubits)

        super().__init__(circuit, qubit_labels)

    name: ClassVar[str] = "Built-in list"

    @property
    def circuit(self) -> list[list[tuple[str, list[QubitTypes]]]]:
        return self._circuit

    @property
    def depth(self) -> int:
        return len(self.circuit)

    @property
    def qubit_labels(self) -> list[QubitTypes]:
        return self._qubit_labels

    def copy(self) -> ListPhysicalCircuit:
        return ListPhysicalCircuit(self._circuit)

    def delete_qubits_inplace(
        self, qubits_to_delete: Sequence[QubitTypes]
    ) -> None:
        new_layers = []
        for layer in self._circuit:
            new_layer = []
            for label in layer:
                if not any([q in qubits_to_delete for q in label[1]]):
                    new_layer.append(label)
            new_layers.append(new_layer)
        self._circuit = new_layers

        qubits_to_keep = []
        for q in self._qubit_labels:
            if q not in qubits_to_delete:
                qubits_to_keep.append(q)
        self._qubit_labels = qubits_to_keep

    def insert_inplace(self, circuit: BasePhysicalCircuit, idx: int) -> None:
        other_circuit = ListPhysicalCircuit.cast(circuit)
        self._circuit = (
            self._circuit[:idx] + other_circuit._circuit + self._circuit[idx:]
        )

    def map_qubit_labels_inplace(
        self, qubit_mapping: Mapping[QubitTypes, QubitTypes]
    ) -> None:
        # Pass through any unspecified qubits
        complete_mapping = {
            q: qubit_mapping.get(q, q) for q in self.qubit_labels
        }

        new_layers = []
        for layer in self._circuit:
            new_layer = []
            for label in layer:
                new_label = (label[0], [complete_mapping[q] for q in label[1]])
                new_layer.append(new_label)
            new_layers.append(new_layer)
        self._circuit = new_layers

        self._qubit_labels = [complete_mapping[q] for q in self.qubit_labels]

    def merge_inplace(self, circuit: BasePhysicalCircuit, idx: int) -> None:
        other_circuit = ListPhysicalCircuit.cast(circuit)

        # Ensure circuit is long enough for merge
        end = idx + other_circuit.depth
        for lidx in range(self.depth, end):
            self._circuit.append([])

        # Perform merge
        for lidx in range(idx, end):
            self._circuit[lidx].extend(other_circuit._circuit[lidx])

        # Also add any new qubit labels
        for other_qubit in other_circuit.qubit_labels:
            if other_qubit not in self.qubit_labels:
                self._qubit_labels.append(other_qubit)

    def set_qubit_labels_inplace(
        self, qubit_labels: Sequence[QubitTypes]
    ) -> None:
        """Set the qubit labels of an underlying circuit.

        This only adds or deletes qubits from the circuit,
        but does not modify the qubit labels of operations.
        For a complete change of qubit labels, see
        :meth:`map_qubit_labels_inplace` instead.

        Parameters
        ----------
        qubit_labels:
            Qubit labels to assign to circuit.
        """
        self._qubit_labels = list(qubit_labels)
