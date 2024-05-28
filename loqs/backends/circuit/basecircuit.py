""":class:`BaseCircuitBackend` definition.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Mapping

from loqs.internal.castable import Castable
from loqs.internal.classproperty import abstractroclassproperty


class BasePhysicalCircuit(Castable):
    """Base class for an object that can holds a physical quantum circuit."""

    _circuit: CircuitType
    """The underlying quantum circuit"""

    ## Dunder methods
    @abstractmethod
    def __init__(
        self,
        circuit: CastableTypes,
        qubit_labels: Iterable[QubitTypes] | None = None,
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

    # Class properties
    @abstractroclassproperty
    def name(self) -> str:
        """Name of circuit backend"""
        return ""

    @abstractroclassproperty
    def CastableTypes(self) -> type:
        """Types that this backend can cast to an underlying circuit object."""
        return type(None)

    @abstractroclassproperty
    def CircuitType(self) -> type:
        """The type of underlying circuit objects handled by this backend."""
        return type(None)

    @abstractroclassproperty
    def QubitTypes(self) -> type:
        """Possible types for a circuit's qubit labels.

        In general, these will be the only types we accept for arguments
        that ask for qubit labels.
        """
        return type(None)

    @abstractroclassproperty
    def OperationTypes(self) -> type:
        """Possible types for a circuit's operations.

        In general, these will be the only types we accept for arguments
        that ask for operations.
        """
        pass

    # Instance properties
    @property
    def circuit(self) -> CircuitType:
        """Getter for underlying circuit object"""
        return self._circuit

    @property
    @abstractmethod
    def qubit_labels(self) -> Iterable[QubitTypes]:
        """Get the qubit labels of an underlying circuit.

        Returns
        -------
            Qubit labels
        """
        pass

    # Instance methods
    def append(self, circuit: BasePhysicalCircuit) -> BasePhysicalCircuit:
        """Append another circuit to a copy of this circuit.

        Parameters
        ----------
        Other Parameters:
            Refer to :meth:`append_qubits_inplace`

        Returns
        -------
        modified_circuit:
            A modified copy of the circuit.
        """
        modified_circuit = self.copy()
        modified_circuit.append_inplace(circuit)
        return modified_circuit

    @abstractmethod
    def append_inplace(self, circuit: BasePhysicalCircuit) -> None:
        """Append another circuit in-place to this circuit.

        Parameters
        ----------
        circuit:
            Circuit to append
        """
        pass

    @abstractmethod
    def copy(self) -> BasePhysicalCircuit:
        """Copy a circuit object.

        Returns
        -------
            Copied circuit
        """
        return BasePhysicalCircuit(self.circuit)

    def delete_qubits(
        self, qubits_to_delete: Iterable[QubitTypes]
    ) -> BasePhysicalCircuit:
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
    def delete_qubits_inplace(
        self, qubits_to_delete: Iterable[QubitTypes]
    ) -> None:
        """Delete qubit lines in-place in a provided circuit.

        Parameters
        ----------
        qubits_to_delete:
            List of qubit lines to remove.
        """
        # TODO: Check qubits available
        pass

    def map_qubit_labels(
        self,
        qubit_mapping: Mapping[QubitTypes, QubitTypes],
    ) -> BasePhysicalCircuit:
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
    def map_qubit_labels_inplace(
        self, qubit_mapping: Mapping[QubitTypes, QubitTypes]
    ) -> None:
        """Substitute qubit labels in underlying circuit objects.

        Parameters
        ----------
        qubit_mapping:
            Mapping from old qubit labels to new qubit labels.
            If a qubit label is not provided, it remains unchanged.
        """
        # TODO: Type check
        pass

    def set_qubit_labels(
        self, qubit_labels: Iterable[QubitTypes]
    ) -> BasePhysicalCircuit:
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
    def set_qubit_labels_inplace(
        self, qubit_labels: Iterable[QubitTypes]
    ) -> None:
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
        self,
        qubit_mapping: Mapping[QubitTypes, QubitTypes] | None = None,
        omit_gates: Iterable[OperationTypes] | None = None,
        delete_idle_layers: bool = False,
    ) -> BasePhysicalCircuit:
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
