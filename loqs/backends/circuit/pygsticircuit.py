""":class:`PyGSTiPhysicalCircuit` definition.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional, Type, TypeAlias, Union, Unpack

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.utils.classproperty import roclassproperty


class PyGSTiPhysicalCircuit(BasePhysicalCircuit):
    """Circuit backend for handling ``pygsti.circuits.Circuit`` objects."""

    def __init__(
        self,
        circuit: Castable,
        qubit_labels: Optional[Iterable[QubitTypes]] = None,
        finalized: bool = True,
    ) -> None:
        try:
            from pygsti.circuits import Circuit
        except ImportError as e:
            raise ImportError(
                "Failed import, cannot use pyGSTi as backend"
            ) from e

        if isinstance(circuit, PyGSTiPhysicalCircuit):
            self._circuit = circuit.circuit
        elif isinstance(circuit, Circuit):
            self._circuit = circuit
        else:
            try:
                self._circuit = Circuit.cast(circuit)
            except Exception as e:
                raise ValueError("Failed to cast to pyGSTi circuit") from e

        if self.finalized and finalized is False:
            self._circuit = self.circuit.copy(editable=True)

        super().__init__(circuit, qubit_labels, finalized)

    @roclassproperty
    def name(self) -> str:
        return "pyGSTi"

    @roclassproperty
    def Castable(self) -> TypeAlias:
        """Types we can cast to a pyGSTi circuit.

        These include another PyGSTiPhysicalCircuit, a bare pyGSTi Circuit,
        or things that a pyGSTi Circuit can cast (a subset of these include
        a string and a list of operations/layers).
        """
        try:
            from pygsti.circuits import Circuit
        except ImportError as e:
            raise ImportError(
                "Failed import, cannot use pyGSTi as backend"
            ) from e

        return Union[
            PyGSTiPhysicalCircuit,
            Circuit,
            str,
            Iterable[self.OperationTypes],
        ]

    @roclassproperty
    def CircuitType(self) -> Type:
        """PyGSTi backend circuit type (pygsti.circuits.Circuit)"""
        try:
            from pygsti.circuits import Circuit
        except ImportError as e:
            raise ImportError(
                "Failed import, cannot use pyGSTi as backend"
            ) from e
        return Circuit

    @roclassproperty
    def QubitTypes(self) -> TypeAlias:
        """PyGSTi backend qubit label types (strs and ints)"""
        return Union[str, int]

    @roclassproperty
    def LayerTypes(self) -> TypeAlias:
        """Helper type alias for things allowed to be in circuit layer

        For PyGSTi, this would be:
        - strings, e.g. "Gcnot"
        - tuples or lists of strings with either one or several qubits,
        e.g. ('Gxpi2', 0) or ['Gcnot', ("Q0", "Q1")]
        - actual Labels (which is what all the above cast to anyway)
        - or lists and tuples of the above (in which case it is a whole layer)
        """
        try:
            from pygsti.baseobjs import Label
        except ImportError as e:
            raise ImportError(
                "Failed import, cannot use pyGSTi as backend"
            ) from e

        return Union[
            str,  # e.g., gate names
            Iterable[str, self.QubitTypes],  # e.g., gate name and one qubit
            Iterable[
                str, Unpack[Iterable[self.QubitTypes]]
            ],  # e.g., gate name and tuple of qubits
            Label,  # or an actual pyGSTi label
        ]

    @roclassproperty
    def OperationTypes(self) -> TypeAlias:
        """PyGSTi backend operations type (one or several gates/layers)"""
        return Union[self.LayerTypes, Iterable[self.LayerTypes]]

    @property
    def finalized(self) -> bool:
        return self.circuit._static

    @property
    def qubit_labels(self) -> Iterable[QubitTypes]:
        return self.circuit.line_labels

    def append_inplace(self, circuit: CircuitType) -> None:
        super().append_inplace(circuit)
        try:
            self.circuit.append_circuit_inplace(circuit.circuit)
        except AssertionError as e:
            raise AssertionError(
                "Underlying circuit is static.",
                "Try using .copy(finalized=False) first",
            ) from e

    def copy(self, finalized: bool = True) -> PyGSTiPhysicalCircuit:
        return PyGSTiPhysicalCircuit(self.circuit, finalized=finalized)

    def delete_qubits_inplace(
        self, qubits_to_delete: Iterable[QubitTypes]
    ) -> None:
        super().delete_qubits_inplace(qubits_to_delete)
        try:
            self.circuit.delete_lines(qubits_to_delete, delete_straddlers=True)
        except AssertionError as e:
            raise AssertionError(
                "Underlying circuit is static.",
                "Try using .copy(finalized=False) first",
            ) from e

    def finalize_inplace(self) -> None:
        self.circuit.done_editing()

    def map_qubit_labels_inplace(
        self, qubit_mapping: Mapping[QubitTypes, QubitTypes]
    ) -> None:
        complete_mapping = {
            q: qubit_mapping.get(q, q) for q in self.circuit.line_labels
        }
        super().map_qubit_labels_inplace(qubit_mapping)
        try:
            self.circuit.map_state_space_labels_inplace(complete_mapping)
        except AssertionError as e:
            raise AssertionError(
                "Underlying circuit is static.",
                "Try using .copy(finalized=False) first",
            ) from e

    def set_qubit_labels_inplace(
        self, qubit_labels: Iterable[QubitTypes]
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
        super().set_qubit_labels_inplace(qubit_labels)
        self.circuit.line_labels = qubit_labels

    def process_circuit(
        self,
        qubit_mapping: Optional[Mapping[QubitTypes, QubitTypes]] = None,
        omit_gates: Optional[Iterable[OperationTypes]] = None,
        delete_idle_layers: bool = False,
        finalized: bool = True,
    ) -> PyGSTiPhysicalCircuit:
        processed = self.copy(finalized=False)

        if qubit_mapping is not None:
            processed.map_qubit_labels_inplace(qubit_mapping)

        if omit_gates is not None:
            for og in omit_gates:
                processed.circuit.replace_gatename_with_idle_inplace(og)

        if delete_idle_layers:
            processed.circuit.delete_idle_layers_inplace()

        if finalized:
            processed.finalize_inplace()

        return processed
