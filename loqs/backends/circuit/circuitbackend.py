""":class:`CircuitBackend` definition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Optional, Type, TypeAlias


class CircuitBackend(ABC):
    """Base class for an object that can create physical quantum circuits."""

    @abstractmethod
    @property
    def name(self) -> str:
        """Name of circuit backend"""
        pass

    @abstractmethod
    @property
    def CircuitType(self) -> Type:
        """The type of underlying circuit objects handled by this backend."""
        pass

    @property
    def CircuitCastable(self) -> TypeAlias:
        """The types this backend is able to cast into a raw circuit object."""
        return self.CircuitType

    @abstractmethod
    @property
    def QubitTypes(self) -> TypeAlias:
        """Possible types for a circuit's qubit labels.

        In general, these will be the only types we accept for arguments
        that ask for qubit labels.
        """
        pass

    @abstractmethod
    @property
    def OperationTypes(self) -> TypeAlias:
        """Possible types for a circuit's operations.

        In general, these will be the only types we accept for arguments
        that ask for operations.
        """
        pass

    def cast_circuit(self, obj: CircuitCastable) -> CircuitType:
        """Helper function to create the raw circuit

        Parameters
        ----------
        obj: CircuitBackend.CircuitCastable
            Object to cast to a circuit

        Returns
        -------
        CircuitBackend.CircuitType
            The cast circuit with specified qubit labels, if given
        """
        if isinstance(obj, self.CircuitType):
            return obj
        else:
            raise NotImplementedError(
                "Derived classes must implement "
                + "casting to a circuit types other the base circuit type."
            )

    @abstractmethod
    def copy_circuit(
        self, circuit: CircuitType, finalized: bool = False
    ) -> CircuitType:
        """Copy a circuit object.

        Parameters
        ----------
        circuit:
            Circuit to copy

        finalized:
            If True, the returned circuit will be in a non-editable "finalized"
            state (if the backend supports this). If False, the returned
            circuit will be in an editable state (if the backend support this),
            and :meth:`finalize_circuit_inplace` can be used to get the
            finalized version of the circuit later. Defaults to False.

        Returns
        -------
        copied_circuit:
            Copied circuit
        """
        pass

    def delete_qubits(
        self,
        circuit: CircuitType,
        qubits_to_delete: Iterable[QubitTypes],
    ) -> CircuitType:
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
        # Providing a default implementation for circuits
        # that can be modified in-place
        modified_circuit = self.copy_circuit(circuit)
        self.delete_qubits_inplace(modified_circuit, qubits_to_delete)
        return modified_circuit

    @abstractmethod
    def delete_qubits_inplace(
        self,
        circuit: CircuitType,
        qubits_to_delete: Iterable[QubitTypes],
    ) -> None:
        """Delete qubit lines in-place in a provided circuit.

        Parameters
        ----------
        circuit:
            Circuit to modify.

        qubits_to_delete:
            List of qubit lines to remove.
        """
        pass

    @abstractmethod
    def finalize_circuit_inplace(self, circuit: CircuitType) -> None:
        """Indicate a circuit is in a finalized state.

        For backends that support both editable and non-editable modes for
        circuits (e.g. for convenience vs performance reasons), this will take
        an editable circuit to a non-editable "finalized" circuit.
        Editable circuits would then be available by using :meth:`copy_circuit`
        with the `finalized=False` flag.

        For backends that do not support multiple circuit modes, this should
        be a no-op.

        Parameters
        ----------
        circuit:
            Circuit to finalize.
        """
        pass

    @abstractmethod
    def get_qubit_labels(self, circuit: CircuitType) -> Iterable[QubitTypes]:
        """Get the qubit labels of an underlying circuit.

        Parameters
        ----------
        circuit: CircuitBackend.CircuitType
            Circuit to extract qubit labels from

        Returns
        -------
        Iterable[CircuitBackend.QubitTypes]
            Qubit labels
        """
        pass

    def map_qubit_labels(
        self,
        circuit: CircuitType,
        qubit_mapping: Mapping[QubitTypes, QubitTypes],
    ) -> CircuitType:
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
        modified_circuit = self.copy_circuit(circuit)
        self.map_qubit_labels_inplace(modified_circuit, qubit_mapping)
        return modified_circuit

    @abstractmethod
    def map_qubit_labels_inplace(
        self,
        circuit: CircuitType,
        qubit_mapping: Mapping[QubitTypes, QubitTypes],
    ) -> None:
        """Substitute qubit labels in underlying circuit objects.

        Parameters
        ----------
        circuit:
            Circuit to modify.

        qubit_mapping:
            Mapping from old qubit labels to new qubit labels.
            If a qubit label is not provided, it remains unchanged.
        """
        pass

    def set_qubit_labels(
        self, circuit: CircuitType, qubit_labels: Iterable[QubitTypes]
    ) -> CircuitType:
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
        modified_circuit = self.copy_circuit(circuit)
        self.set_qubit_labels_inplace(modified_circuit, qubit_labels)
        return modified_circuit

    @abstractmethod
    def set_qubit_labels_inplace(
        self, circuit: CircuitType, qubit_labels: Iterable[QubitTypes]
    ) -> None:
        """Set the qubit labels of an underlying circuit.

        Parameters
        ----------
        circuit: CircuitBackend.CircuitType
            Circuit to modify.

        qubit_labels:
            Qubit labels to assign to circuit.
        """
        pass

    @abstractmethod
    def process_circuit(
        self,
        circuit: CircuitType,
        qubit_labels: Optional[Iterable[QubitTypes]] = None,
        omit_gates: Optional[Iterable[OperationTypes]] = None,
        delete_idle_layers: bool = False,
    ) -> CircuitType:
        """Helper function to provide consistent circuit processing.

        Parameters
        ----------
        circuit: CircuitBackend.CircuitType
            Circuit to process

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
