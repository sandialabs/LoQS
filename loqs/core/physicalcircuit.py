""":class:`PhysicalCircuitContainer` and :class:`PhysicalCircuit` definitions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Optional, Union


from loqs.backends import (
    CircuitBackend,
    CircuitBackendCastable,
    cast_circuit_backend,
)
from loqs.utils.sigtools import compose_funcs_by_first_arg


class PhysicalCircuitContainer(ABC):
    """An abstract interface class for containers of bare circuit objects.

    This class has some pretty nonstandard intialization logic intended to
    provide transparency to the underlying :class:`CircuitBackend` while making
    it simple to add new backends or types of circuit containers.

    Derived classes just need to implement :meth:`get_bare_circuit` and
    :meth:`get_bare_circuit_iter`. The :meth:`compose_funcs_by_first_arg`
    utility function will then be used to create the remaining member functions
    such that the first arguments will be those required to retrieve a bare
    circuit from the container, and the remaining arguments will be whatever
    is expected by the backend.

    For example, a container with a list or dict of circuits may have an index
    or key arg prepended to the backend functions, while a more complicated
    factory container could generate a bare circuit on the fly to be then
    further modified by the backend.
    Examples can be found in :class:`PhysicalCircuit` below, or in
    :class:`CircuitTemplateFactory` or :class:`TemplatedCircuit`.
    """

    circuit_backend: CircuitBackend
    """Underlying circuit backend for handling bare circuit objects."""

    def __init__(self, backend: CircuitBackendCastable) -> None:  # noqa: C901
        """Initialize a PhysicalCircuitContainer.

        This includes dynamically generating the passthrough functions to the
        circuit backend based on the type of container, i.e. implementations of
        :meth:`get_bare_circuit` and :meth:`get_bare_circuit_iter`.

        Parameters
        ----------
        backend:
            The underlying circuit backend
        """
        self.circuit_backend = cast_circuit_backend(backend=backend)

        ### Create convenience passthrough functions to backend
        # In-place functions
        try:
            self.delete_qubits_inplace()
        except (TypeError, NotImplementedError):
            compose_funcs_by_first_arg(
                [
                    "self.get_bare_circuit",
                    "self.circuit_backend.delete_qubits_inplace",
                ],
                self_obj=self,
                bind_name="delete_qubits_inplace",
            )
        try:
            self.finalize_circuit_inplace()
        except (TypeError, NotImplementedError):
            compose_funcs_by_first_arg(
                [
                    "self.get_bare_circuit",
                    "self.circuit_backend.finalize_qubits_inplace",
                ],
                self_obj=self,
                bind_name="finalize_circuit_inplace",
            )
        try:
            self.get_qubit_labels()
        except (TypeError, NotImplementedError):
            compose_funcs_by_first_arg(
                [
                    "self.get_bare_circuit",
                    "self.circuit_backend.get_qubit_labels",
                ],
                self_obj=self,
                bind_name="get_qubit_labels",
            )
        try:
            self.map_qubit_labels_inplace()
        except (TypeError, NotImplementedError):
            compose_funcs_by_first_arg(
                [
                    "self.get_bare_circuit",
                    "self.circuit_backend.map_qubit_labels_inplace",
                ],
                self_obj=self,
                bind_name="map_qubit_labels_inplace",
            )
        try:
            self.set_qubit_labels_inplace()
        except (TypeError, NotImplementedError):
            compose_funcs_by_first_arg(
                [
                    "self.get_bare_circuit",
                    "self.circuit_backend.set_qubit_labels_inplace",
                ],
                self_obj=self,
                bind_name="set_qubit_labels_inplace",
            )

        # Functions that return a copy of the circuit
        # We will return a PhysicalCircuit instead of the bare circuit
        try:
            self.copy_circuit()
        except (TypeError, NotImplementedError):
            compose_funcs_by_first_arg(
                [
                    "self.get_bare_circuit",
                    "self.circuit_backend.copy_circuit",
                    "self.wrap_circuit",
                ],
                self_obj=self,
                bind_name="copy_circuit",
            )
        try:
            self.delete_qubits()
        except (TypeError, NotImplementedError):
            compose_funcs_by_first_arg(
                [
                    "self.get_bare_circuit",
                    "self.circuit_backend.delete_qubits",
                    "self.wrap_circuit",
                ],
                self_obj=self,
                bind_name="delete_qubits",
            )
        try:
            self.map_qubit_labels()
        except (TypeError, NotImplementedError):
            compose_funcs_by_first_arg(
                [
                    "self.get_bare_circuit",
                    "self.circuit_backend.map_qubit_labels",
                    "self.wrap_circuit",
                ],
                self_obj=self,
                bind_name="map_qubit_labels",
            )
        try:
            self.set_qubit_labels()
        except (TypeError, NotImplementedError):
            compose_funcs_by_first_arg(
                [
                    "self.get_bare_circuit",
                    "self.circuit_backend.set_qubit_labels",
                    "self.wrap_circuit",
                ],
                self_obj=self,
                bind_name="set_qubit_labels",
            )
        try:
            self.get_processed_circuit()
        except (TypeError, NotImplementedError):
            compose_funcs_by_first_arg(
                [
                    "self.get_bare_circuit",
                    "self.circuit_backend.process_circuit",
                    "self.wrap_circuit",
                ],
                self_obj=self,
                bind_name="get_processed_circuit",
            )

    @property
    def CircuitType(self):
        return self.circuit_backend.CircuitType

    @property
    def CircuitCastable(self):
        return self.circuit_backend.CircuitCastable

    @property
    def QubitTypes(self):
        return self.circuit_backend.QubitTypes

    @property
    def OperationTypes(self):
        return self.circuit_backend.OperationTypes

    @abstractmethod
    def get_bare_circuit(self) -> CircuitType:
        """"""
        pass

    @abstractmethod
    def get_bare_circuit_iter(self) -> Iterator[CircuitType]:
        pass

    def wrap_circuit(self, circuit: CircuitType) -> PhysicalCircuit:
        return PhysicalCircuit(circuit, self.circuit_backend)

    # Convenience passthrough functions
    # These are here to help define the interface, but do not need to be
    # implemented as they will be generated by composition from the
    # corresponding backend functions (if they are still unimplemented)
    def copy_circuit(self):
        raise NotImplementedError()

    def delete_qubits(self):
        raise NotImplementedError()

    def delete_qubits_inplace(self):
        raise NotImplementedError()

    def finalize_circuit_inplace(self):
        raise NotImplementedError()

    # Note the name change from process_circuit
    def get_processed_circuit(self):
        raise NotImplementedError()

    def get_qubit_labels(self):
        raise NotImplementedError()

    def map_qubit_labels(self):
        raise NotImplementedError()

    def map_qubit_labels_inplace(self):
        raise NotImplementedError()

    def set_qubit_labels(self):
        raise NotImplementedError()

    def set_qubit_labels_inplace(self):
        raise NotImplementedError()

    # Convenience functions for modifying all bare circuits in this object
    def delete_all_qubits_inplace(self, *args, **kwargs) -> None:
        """Delete qubit lines in-place for all circuits.

        See :meth:`delete_qubits_inplace` for more details.
        """
        for circ in self.get_bare_circuit_iter():
            self.delete_qubits_inplace(circ, *args, **kwargs)

    def finalize_all_circuits_inplace(self, *args, **kwargs) -> None:
        """Indicate all circuits are in a finalized state.

        See :meth:`finalize_circuit_inplace` for more details.
        """
        for circ in self.get_bare_circuit_iter():
            self.finalize_circuit_inplace(circ, *args, **kwargs)

    def map_all_qubit_labels_inplace(self, *args, **kwargs) -> None:
        """Substitute qubit labels in-place for all circuits.

        See :meth:`map_qubit_labels_inplace` for more details.
        """
        for circ in self.get_bare_circuit_iter():
            self.map_qubit_labels_inplace(circ, *args, **kwargs)

    def set_all_qubit_labels_inplace(self, *args, **kwargs) -> None:
        """Set qubit labels in-place for all circuits.

        See :meth:`set_qubit_labels_inplace` for more details.
        """
        for circ in self.get_bare_circuit_iter():
            self.set_qubit_labels_inplace(circ, *args, **kwargs)


class PhysicalCircuit(PhysicalCircuitContainer):
    """Base wrapped circuit class.

    This is essentially a :class:`PhysicalCircuitContainer` for a single
    circuit.
    """

    def __init__(
        self,
        circuit: Union[PhysicalCircuit, PhysicalCircuit.CircuitCastable],
        qubit_labels: Optional[Iterable[PhysicalCircuit.QubitTypes]] = None,
        backend: Optional[CircuitBackendCastable] = None,
    ) -> None:
        """Initialize a PhysicalCircuit.

        Parameters
        ----------
        circuit:
            The underlying (potentially bare) circuit to store

        qubit_labels:
            Explicit qubit labels to use

        backend:
            The underlying circuit backend
        """
        super().__init__(backend)

        if isinstance(circuit, PhysicalCircuit):
            self._circuit = circuit._circuit
        else:
            self._circuit = self.circuit_backend.cast_circuit(circuit)

        if qubit_labels is not None:
            self.set_qubit_labels_inplace(self._circuit, qubit_labels)

    def get_bare_circuit(self) -> PhysicalCircuit.CircuitType:
        return self._circuit

    def get_bare_circuit_iter(self) -> Iterator[PhysicalCircuit.CircuitType]:
        yield self._circuit

    def copy(self) -> PhysicalCircuit:
        return self.copy_circuit()

    def __str__(self) -> str:
        return (
            f"Physical circuit with {self.circuit_backend.name} backend:\n"
            + str(self._circuit)
        )

    def __repr__(self) -> str:
        return (
            f"Physical circuit with {self.circuit_backend.name} backend:\n"
            + repr(self._circuit)
        )
