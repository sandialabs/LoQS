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

    _circuit: list[list[tuple[str, tuple[QubitTypes, ...]]]]
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

        self._circuit = []
        if isinstance(circuit, ListPhysicalCircuit):
            self._circuit = circuit.circuit.copy()
            self._qubit_labels = circuit.qubit_labels
        elif isinstance(circuit, PyGSTiPhysicalCircuit):
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
                raise ValueError("Could not cast pyGSTi circuit") from e
        elif isinstance(circuit, Sequence):
            seen_qubits = set()

            def process_label(label) -> tuple[str, tuple[QubitTypes, ...]]:
                if len(label) == 2 and isinstance(label[1], (list, tuple)):
                    new_label = (label[0], tuple(label[1]))
                elif len(label) == 2:
                    new_label = (label[0], (label[1],))
                else:
                    new_label = (label[0], tuple(label[1:]))

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
        else:
            raise ValueError("Expected BasePhysicalCircuit or list of layers")

        super().__init__(circuit, qubit_labels)

    name: ClassVar[str] = "Built-in list"

    @property
    def circuit(self) -> list[list[tuple[str, tuple[QubitTypes, ...]]]]:
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
                new_label = (
                    label[0],
                    tuple([complete_mapping[q] for q in label[1]]),
                )
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
            self._circuit[lidx].extend(other_circuit._circuit[lidx - idx])

        # Also add any new qubit labels
        for other_qubit in other_circuit.qubit_labels:
            if other_qubit not in self.qubit_labels:
                self._qubit_labels.append(other_qubit)

    def pad_single_qubit_idles_by_duration_inplace(
        self,
        idle_names: Mapping[int | float, str],
        durations: Mapping[str, int | float],
        default_duration: int | float | None = None,
        empty_layer_idle: str | None = None,
    ) -> None:
        for lidx in range(self.depth):
            # Check with qubits are not idling and compute duration
            seen_qubits = set()
            layer_duration = None
            for comp in self._circuit[lidx]:
                duration = durations.get(comp[0], default_duration)
                if duration is None:
                    raise KeyError(
                        f"No duration for {comp[0]} or default specified"
                    )
                if layer_duration is None:
                    layer_duration = duration
                else:
                    layer_duration = max(layer_duration, duration)

                for qubit in comp[1]:
                    seen_qubits.add(qubit)

            # Get idling operation (or skip for empty layers with no idles)
            if layer_duration is None:
                if empty_layer_idle is None:
                    continue
                layer_idle = empty_layer_idle
            else:
                layer_idle = idle_names[layer_duration]

            # Insert idling operations
            missing_qubits = set(self._qubit_labels) - seen_qubits
            for qubit in missing_qubits:
                self._circuit[lidx].append((layer_idle, (qubit,)))

    def set_qubit_labels_inplace(
        self, qubit_labels: Sequence[QubitTypes]
    ) -> None:
        self._qubit_labels = list(qubit_labels)

    @classmethod
    def _deserialize_circuit(
        cls,
        serial_circuit: str | list | dict,
        qubit_labels: Sequence | None = None,
    ) -> list[list[tuple[str, tuple[QubitTypes, ...]]]]:
        """Helper function to deserialize a circuit.

        Derived classes should implement this for
        deserialization to work.
        """
        # For list circuit, it is already serializable
        # qubit_labels not needed
        assert isinstance(serial_circuit, list)
        return serial_circuit

    def _serialize_circuit(self) -> str | list | dict:
        """Helper function to serialize a circuit.

        Derived classes should implement this for
        serialization to work.
        """
        # For list circuit, it is already serializable
        return self.circuit
