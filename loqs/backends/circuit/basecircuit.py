""":class:`BaseCircuitBackend` definition.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence, Mapping
from typing import ClassVar, TypeAlias, TypeVar

from loqs.internal.castable import Castable


# Generic type variable to stand-in for derived class below
T = TypeVar("T", bound="BasePhysicalCircuit")


class BasePhysicalCircuit(Castable):
    """Base class for an object that can holds a physical quantum circuit."""

    # Class attributes
    name: ClassVar[str]
    """Name of circuit backend"""

    CastableTypes: ClassVar[TypeAlias]

    ## Dunder methods
    @abstractmethod
    def __init__(
        self,
        circuit: object,
        qubit_labels: Sequence | None = None,
    ) -> None:
        """Initialize a PhysicalCircuit.

        Parameters
        ----------
        circuit:
            The underlying circuit object

        qubit_labels:
            Qubit labels to use for the circuit
        """
        if qubit_labels is not None:
            self.set_qubit_labels_inplace(qubit_labels)

    def __str__(self) -> str:
        return f"Physical {self.name} circuit:\n{str(self.circuit)}"

    def __repr__(self) -> str:
        return f"Physical {self.name} circuit:\n{repr(self.circuit)}"

    # Instance properties
    @property
    @abstractmethod
    def circuit(self) -> object:
        """Getter for underlying circuit object"""
        pass

    @property
    @abstractmethod
    def depth(self) -> int:
        """The length/depth of the circuit (number of layers)."""
        pass

    @property
    @abstractmethod
    def qubit_labels(self) -> Sequence:
        """Get the qubit labels of an underlying circuit.

        Returns
        -------
            Qubit labels
        """
        pass

    # Instance methods
    def append(self: T, circuit: BasePhysicalCircuit) -> T:
        """Append another circuit to a copy of this circuit.

        Parameters
        ----------
        Other Parameters:
            Refer to :meth:`append_inplace`

        Returns
        -------
        modified_circuit:
            A modified copy of the circuit.
        """
        modified_circuit = self.copy()
        modified_circuit.append_inplace(circuit)
        return modified_circuit

    def append_inplace(self, circuit: BasePhysicalCircuit) -> None:
        """Append another circuit in-place to this circuit.

        Parameters
        ----------
        circuit:
            Circuit to append
        """
        self.insert_inplace(circuit, self.depth)

    @abstractmethod
    def copy(self: T) -> T:
        """Copy a circuit object.

        Returns
        -------
            Copied circuit
        """
        pass

    def delete_qubits(self: T, qubits_to_delete: Sequence) -> T:
        """Delete qubit lines in a copy of the provided circuit.

        Operations involving the deleted qubits are also removed.

        Parameters
        ----------
        Other Parameters:
            Refer to :meth:`delete_qubits_inplace`

        Returns
        -------
        modified_circuit:
            A modified copy of the circuit.
        """
        modified_circuit = self.copy()
        modified_circuit.delete_qubits_inplace(qubits_to_delete)
        return modified_circuit

    @abstractmethod
    def delete_qubits_inplace(self, qubits_to_delete: Sequence) -> None:
        """Delete qubit lines in-place in a provided circuit.

        Parameters
        ----------
        qubits_to_delete:
            List of qubit lines to remove.
        """
        # TODO: Check qubits available
        pass

    def insert(self: T, circuit: BasePhysicalCircuit, idx: int) -> T:
        """Insert another circuit to a copy of this circuit.

        Parameters
        ----------
        Other Parameters:
            Refer to :meth:`insert_inplace`

        Returns
        -------
        modified_circuit:
            A modified copy of the circuit.
        """
        modified_circuit = self.copy()
        modified_circuit.insert_inplace(circuit, idx)
        return modified_circuit

    @abstractmethod
    def insert_inplace(self, circuit: BasePhysicalCircuit, idx: int) -> None:
        """Insert another circuit to this circuit.

        Parameters
        ----------
        circuit:
            Circuit to append

        idx:
            Starting index to begin insert. If -1, append to the end.
        """
        pass

    def map_qubit_labels(self: T, qubit_mapping: Mapping) -> T:
        """Substitute qubit labels in underlying circuit objects.

        Parameters
        ----------
        Other Parameters:
            Refer to :meth:`map_qubit_labels_inplace`

        Returns
        -------
        modified_circuit:
            A copy of the circuit with mapped qubits.
        """
        # Providing a default implementation for circuits
        # that can be modified in-place
        modified_circuit = self.copy()
        modified_circuit.map_qubit_labels_inplace(qubit_mapping)
        return modified_circuit

    @abstractmethod
    def map_qubit_labels_inplace(self, qubit_mapping: Mapping) -> None:
        """Substitute qubit labels in underlying circuit objects.

        Parameters
        ----------
        qubit_mapping:
            Mapping from old qubit labels to new qubit labels.
            If a qubit label is not provided, it remains unchanged.
        """
        pass

    def merge(self: T, circuit: BasePhysicalCircuit, idx: int) -> T:
        """Merge another circuit to a copy of this circuit.

        While :meth:`insert` adds new layers, :meth:`merge`
        will try to add operations to existing layers.

        Parameters
        ----------
        Other Parameters:
            Refer to :meth:`merge_inplace`

        Returns
        -------
        modified_circuit:
            A modified copy of the circuit.
        """
        modified_circuit = self.copy()
        modified_circuit.merge_inplace(circuit, idx)
        return modified_circuit

    @abstractmethod
    def merge_inplace(self, circuit: BasePhysicalCircuit, idx: int) -> None:
        """Merge another circuit to this circuit.

        While :meth:`insert_inplace` adds new layers,
        :meth:`merge_inplace` will try to add operations to
        existing layers.

        Parameters
        ----------
        circuit:
            Circuit to merge

        idx:
            Layer index to start merge
        """
        pass

    def set_qubit_labels(self: T, qubit_labels: Sequence) -> T:
        """Set the qubit labels of an underlying circuit.

        Parameters
        ----------
        Other Parameters:
            Refer to :meth:`set_qubit_labels_inplace`

        Returns
        -------
        modified_circuit:
            A modified copy of the circuit.
        """
        # Providing a default implementation for circuits
        # that can be modified in-place
        modified_circuit = self.copy()
        modified_circuit.set_qubit_labels_inplace(qubit_labels)
        return modified_circuit

    @abstractmethod
    def set_qubit_labels_inplace(self, qubit_labels: Sequence) -> None:
        """Set the qubit labels of an underlying circuit.

        Note that depending on the backend, this may only adjust
        the qubit labels and not the operations in the circuit.
        For a complete change of qubit labels, see
        :meth:`map_qubit_labels_inplace` instead.

        Parameters
        ----------
        qubit_labels:
            Qubit labels to assign to circuit.
        """
        # TODO: Type check
        pass

    @abstractmethod
    def process_circuit(
        self: T,
        qubit_mapping: Mapping | None = None,
        omit_gates: Sequence | None = None,
        delete_idle_layers: bool = False,
    ) -> T:
        """Helper function to provide consistent circuit processing.

        Parameters
        ----------
        qubit_mapping:
            Mapping from old qubit labels to new qubit labels.
            If a qubit label is not provided, it remains unchanged.

        omit_gates:
            If provided, an operation (or list of operations) to replace with
            idles in the final circuit.

        delete_idle_layers:
            If True, drop any layers with no operations.
            Defaults to False, maintaining idle layers which may be used for
            scheduling later in circuit composition pipeline.

        Returns
        -------
        processed_circuit:
            The processed circuit
        """
        pass
