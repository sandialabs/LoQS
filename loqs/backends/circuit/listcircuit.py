#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################



from __future__ import annotations

from collections.abc import Sequence, Mapping
from typing import ClassVar, TypeAlias, Any

from loqs.backends.circuit import BasePhysicalCircuit


## Type aliases for static type checking
QubitTypes: TypeAlias = str | int
"""Qubit types for builtins"""

LabelType: TypeAlias = tuple[str, QubitTypes | Sequence[QubitTypes]]
"""Helper type alias for things allowed to be a label"""

OperationTypes: TypeAlias = LabelType | Sequence[LabelType]
"""Type alias for things allowed to be in circuit layer
"""

ListCircuitCastableTypes: TypeAlias = (
    BasePhysicalCircuit | Sequence[OperationTypes]
)
"""Types we can cast to a built-in circuit.
"""


class ListPhysicalCircuit(BasePhysicalCircuit):
    """Circuit backend using built-in Python objects."""

    _circuit: list[list[tuple[str, tuple[QubitTypes, ...]]]]
    """List of layers (which are lists of 2-tuples with (name, qubits))
    """

    _qubit_labels: list[QubitTypes]
    """List of qubit labels"""

    def __init__(
        self,
        circuit: ListCircuitCastableTypes,
        qubit_labels: Sequence[QubitTypes] | None = None,
    ) -> None:
        from loqs.backends import is_backend_available

        try:
            from loqs.backends import PyGSTiPhysicalCircuit
        except ImportError:
            PyGSTiPhysicalCircuit = Any

        self._circuit = []
        if isinstance(circuit, ListPhysicalCircuit):
            self._circuit = circuit.circuit.copy()
            self._qubit_labels = circuit.qubit_labels
        elif is_backend_available("pygsti_circuit") and isinstance(
            circuit, PyGSTiPhysicalCircuit
        ):
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
                    qubit_labels = circuit.circuit.line_labels  # type: ignore
            except ImportError as e:
                raise ValueError("Could not cast pyGSTi circuit") from e
        elif isinstance(circuit, Sequence):
            seen_qubits = set()

            def process_label(label) -> tuple[str, tuple[QubitTypes, ...]]:
                """Process a circuit label into a standardized format.

                This helper function normalizes various label formats into a consistent
                2-tuple format of (operation_name, qubit_tuple).

                Parameters
                ----------
                label : tuple
                    The input label to process. Can be in various formats:
                    - (name, [qubits]) or (name, (qubits,)): already properly formatted
                    - (name, qubit): single qubit that needs tuple wrapping
                    - (name, qubit1, qubit2, ...): multiple qubits that need tuple wrapping

                Returns
                -------
                tuple[str, tuple[QubitTypes, ...]]
                    A standardized 2-tuple containing:
                    - The operation name (str)
                    - A tuple of qubit labels

                Raises
                ------
                AssertionError
                    If the label cannot be processed into a valid 2-tuple format
                    or if the operation name is not a string.

                Notes
                -----
                This function also tracks seen qubits in the parent function's
                `seen_qubits` set for qubit label management.

                REVIEW_NO_DOCSTRING
                """
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
        """The underlying circuit as a list of layers.

        Returns
        -------
        circuit : list[list[tuple[str, tuple[QubitTypes, ...]]]]
            A list of layers, where each layer is a list of tuples.
            Each tuple contains a gate name (str) and a tuple of qubit labels.

        REVIEW_NO_DOCSTRING
        """
        return self._circuit

    @property
    def depth(self) -> int:
        """The depth of the circuit.

        Returns the number of layers in the circuit.

        Returns
        -------
        int
            The number of layers in the circuit.

        REVIEW_NO_DOCSTRING
        """
        return len(self.circuit)

    @property
    def qubit_labels(self) -> list[QubitTypes]:
        """Get the list of qubit labels in the circuit.

        Returns
        -------
        list[QubitTypes]
            A list containing the labels of all qubits in the circuit.
            QubitTypes can be either str or int.

        REVIEW_NO_DOCSTRING
        """
        return self._qubit_labels

    def copy(self) -> ListPhysicalCircuit:
        """Create a copy of this circuit.

        Returns a new ListPhysicalCircuit instance with the same circuit structure
        and qubit labels as this one.

        Returns
        -------
        ListPhysicalCircuit
            A copy of this circuit.

        REVIEW_NO_DOCSTRING
        """
        return ListPhysicalCircuit(self._circuit)

    def delete_qubits_inplace(
        self, qubits_to_delete: Sequence[QubitTypes]
    ) -> None:
        """Delete qubits from the circuit in place.

        Removes all operations involving the specified qubits and updates the qubit labels.

        Parameters
        ----------
        qubits_to_delete : Sequence[QubitTypes]
            Sequence of qubit labels to delete from the circuit.

        REVIEW_NO_DOCSTRING
        """
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

    def get_possible_discrete_error_locations(
        self, post_twoq_gates: bool = False
    ) -> list[tuple[int, int | tuple[int, ...]]]:
        """Get possible locations for discrete error injection in the circuit.

        This method identifies potential locations where discrete errors could be
        injected into the circuit. The behavior depends on the `post_twoq_gates`
        parameter.

        Parameters
        ----------
        post_twoq_gates : bool, optional
            If True, only consider locations after two-qubit gates and return qubit
            indices as tuples. If False (default), consider all gates and return
            individual qubit indices.

        Returns
        -------
        list[tuple[int, int | tuple[int, ...]]]
            A list of tuples where each tuple contains:
            - The layer index (int)
            - Either a single qubit index (int) or a tuple of qubit indices
              (tuple[int, ...]) depending on the `post_twoq_gates` parameter

        Notes
        -----
        When `post_twoq_gates` is True, the layer indices are incremented by 1 to
        represent positions after the gates. When False, layer indices represent
        the actual layer positions.

        REVIEW_NO_DOCSTRING
        """
        circuit_locations: list[tuple[int, int | tuple[int, ...]]] = []
        for lidx in range(len(self._circuit)):
            for comp in self._circuit[lidx]:
                if post_twoq_gates:
                    if len(comp[1]) == 2:
                        circuit_locations.append(
                            (
                                lidx + 1,
                                tuple(
                                    [
                                        self.qubit_labels.index(q)
                                        for q in comp[1]
                                    ]
                                ),
                            )
                        )
                else:
                    circuit_locations.extend(
                        [(lidx, self.qubit_labels.index(q)) for q in comp[1]]
                    )
        return circuit_locations

    def insert_inplace(self, circuit: BasePhysicalCircuit, idx: int) -> None:
        """Insert another circuit into this circuit at a specified position.

        Parameters
        ----------
        circuit : BasePhysicalCircuit
            The circuit to insert into this circuit.
        idx : int
            The index at which to insert the circuit.

        REVIEW_NO_DOCSTRING
        """
        other_circuit = ListPhysicalCircuit.cast(circuit)
        self._circuit = (
            self._circuit[:idx] + other_circuit._circuit + self._circuit[idx:]
        )

    def map_qubit_labels_inplace(
        self, qubit_mapping: Mapping[QubitTypes, QubitTypes]
    ) -> None:
        """Substitute qubit labels in the underlying circuit objects.

        Parameters
        ----------
        qubit_mapping : Mapping[QubitTypes, QubitTypes]
            Mapping from old qubit labels to new qubit labels.
            If a qubit label is not provided, it remains unchanged.

        REVIEW_NO_DOCSTRING
        """
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
        """Merge another circuit into this circuit at a specified position.

        Parameters
        ----------
        circuit : BasePhysicalCircuit
            The circuit to merge into this circuit.
        idx : int
            The index at which to start merging the circuit.

        Notes
        -----
        This method extends the current circuit if necessary to accommodate the
        merged circuit. It also adds any new qubit labels from the merged circuit
        that are not already present in this circuit.

        REVIEW_NO_DOCSTRING
        """
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
        """Replace empty spaces in layers with duration-specific idles.

        This computes the max duration of all other operations in
        the layer, and then inserts the appropriate idle.

        Parameters
        ----------
        idle_names : Mapping[int | float, str]
            A mapping from layer duration to idle operation name.

        durations : Mapping[str, int | float]
            A mapping from operation names to durations.

        default_duration : int | float | None, optional
            Default duration to use if not provided in `durations`.
            Defaults to None, which will cause a KeyError to be thrown.

        empty_layer_idle : str | None, optional
            Label to use for qubits in a completely empty label.
            Defaults to None, which inserts no idles.

        REVIEW_NO_DOCSTRING
        """
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
        """Set the qubit labels for the circuit in place.

        This method replaces the current qubit labels with the provided sequence
        of qubit labels. The new labels will be used for all subsequent operations
        on the circuit.

        Parameters
        ----------
        qubit_labels : Sequence[QubitTypes]
            Sequence of new qubit labels to set for the circuit.
            QubitTypes can be either str or int.

        Notes
        -----
        This operation modifies the circuit in place and does not return a new circuit.
        The qubit labels are converted to a list internally.

        REVIEW_NO_DOCSTRING
        """
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
