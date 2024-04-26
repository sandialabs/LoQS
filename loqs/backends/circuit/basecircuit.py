""":class:`BaseCircuitBackend` definition.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import Optional, Type, TypeAlias

from loqs.utils.castable import IsCastable
from loqs.utils.classproperty import abstractroclassproperty


class BasePhysicalCircuit(IsCastable):
    """Base class for an object that can holds a physical quantum circuit."""

    _circuit: CircuitType
    """The underlying quantum circuit"""

    ## Dunder methods
    @abstractmethod
    def __init__(
        self,
        circuit: Castable,
        qubit_labels: Optional[Iterable[QubitTypes]] = None,
    ) -> None:
        """Initialize a PhysicalCircuit.

        Parameters
        ----------
        circuit:
            The underlying circuit object

        qubit_labels:
            Qubit labels to use for the circuit
        """
        pass

    def __str__(self) -> str:
        return f"Physical {self.name} circuit:\n{str(self.circuit)}"

    def __repr__(self) -> str:
        return f"Physical {self.name} circuit:\n{repr(self.circuit)}"

    # Class properties
    @abstractroclassproperty
    def name(self) -> str:
        """Name of circuit backend"""
        pass

    @abstractroclassproperty
    def Castable(self) -> TypeAlias:
        """Types that this backend can cast to an underlying circuit object."""
        pass

    @abstractroclassproperty
    def CircuitType(self) -> Type:
        """The type of underlying circuit objects handled by this backend."""
        pass

    @abstractroclassproperty
    def QubitTypes(self) -> TypeAlias:
        """Possible types for a circuit's qubit labels.

        In general, these will be the only types we accept for arguments
        that ask for qubit labels.
        """
        pass

    @abstractroclassproperty
    def OperationTypes(self) -> TypeAlias:
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
    def finalized(self) -> bool:
        """Whether the underlying circuit is finalized, i.e. not editable"""
        pass

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
    @abstractmethod
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
        modified_circuit = self.copy(finalized=False)
        modified_circuit.append_inplace(circuit)
        if self.finalized:
            modified_circuit.finalize_inplace()
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
    def copy(self, finalized: bool = False) -> BasePhysicalCircuit:
        """Copy a circuit object.

        Parameters
        ----------
        finalized:
            If True, the returned circuit will be in a non-editable "finalized"
            state (if the backend supports this). If False, the returned
            circuit will be in an editable state (if the backend support this),
            and :meth:`finalize_circuit_inplace` can be used to get the
            finalized version of the circuit later. Defaults to False.

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
        modified_circuit = self.copy(finalized=False)
        modified_circuit.delete_qubits_inplace(qubits_to_delete)
        if self.finalized:
            modified_circuit.finalize_inplace()
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
        pass

    @abstractmethod
    def finalize_inplace(self) -> None:
        """Indicate a circuit is in a finalized state.

        For backends that support both editable and non-editable modes for
        circuits (e.g. for convenience vs performance reasons), this will take
        an editable circuit to a non-editable "finalized" circuit.
        Editable circuits would then be available by using :meth:`copy_circuit`
        with the `finalized=False` flag.

        For backends that do not support multiple circuit modes, this should
        be a no-op.
        """
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
        modified_circuit = self.copy(finalized=False)
        modified_circuit.map_qubit_labels_inplace(qubit_mapping)
        if self.finalized:
            modified_circuit.finalize_inplace()
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
        modified_circuit = self.copy(finalized=False)
        modified_circuit.set_qubit_labels_inplace(qubit_labels)
        if self.finalized:
            modified_circuit.finalize_inplace()
        return modified_circuit

    @abstractmethod
    def set_qubit_labels_inplace(
        self, qubit_labels: Iterable[QubitTypes]
    ) -> None:
        """Set the qubit labels of an underlying circuit.

        Parameters
        ----------
        qubit_labels:
            Qubit labels to assign to circuit.
        """
        assert len(qubit_labels) == len(self.qubit_labels), (
            f"Only provided {len(qubit_labels)} labels for ",
            f"{len(self.qubit_labels)} qubits",
        )
        pass

    @abstractmethod
    def process_circuit(
        self,
        qubit_labels: Optional[Iterable[QubitTypes]] = None,
        omit_gates: Optional[Iterable[OperationTypes]] = None,
        delete_idle_layers: bool = False,
    ) -> BasePhysicalCircuit:
        """Helper function to provide consistent circuit processing.

        Parameters
        ----------
        qubit_labels: list of str, optional
            Qubit labels to use for the returned circuit. If not provided,
            the default qubit labels of the object are used.

        omit_gates: str or list of str, optional
            If provided, an operation (or list of operations) to replace with
            idles in the final circuit.

        delete_idle_layers: bool, optional
            If True, drop any layers with no operations.
            Defaults to False, maintaining idle layers which may be used for
            scheduling later in circuit composition pipeline.

        Returns
        -------
        processed_circuit:
            The processed circuit
        """
        pass
