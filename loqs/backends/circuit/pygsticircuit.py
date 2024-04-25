""":class:`PyGSTiPhysicalCircuit` definition.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional, Type, TypeAlias, Union
import warnings

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.utils.classproperty import roclassproperty


class PyGSTiPhysicalCircuit(BasePhysicalCircuit):
    """Circuit backend for handling ``pygsti.circuits.Circuit`` objects."""

    def __init__(
        self,
        circuit: Castable,
        qubit_labels: Optional[Iterable[QubitTypes]] = None,
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

        if qubit_labels is not None:
            self.circuit.line_labels = qubit_labels

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

            return Union[
                PyGSTiPhysicalCircuit,
                Circuit,
                str,
                Iterable[self.OperationTypes],
            ]
        except ImportError:
            warnings.warn(
                UserWarning(
                    "Calling PyGSTiPhysicalCircuit property, but "
                    "pygsti is not installed and backend is unavailable"
                )
            )
            return None

    @roclassproperty
    def CircuitType(self) -> Type:
        """PyGSTi backend circuit type (pygsti.circuits.Circuit)"""
        try:
            from pygsti.circuits import Circuit

            return Circuit
        except ImportError:
            warnings.warn(
                UserWarning(
                    "Calling PyGSTiPhysicalCircuit property, but "
                    "pygsti is not installed and backend is unavailable"
                )
            )
            return None

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
            from pygsti.circuits import Label

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
                Label,  # or an actual pyGSTi label
            ]
        except ImportError:
            warnings.warn(
                UserWarning(
                    "Calling PyGSTiPhysicalCircuit property, but "
                    "pygsti is not installed and backend is unavailable"
                )
            )
            return None

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

    def append(self, circuit: CircuitType) -> CircuitType:
        return super().append(circuit)

    def append_inplace(self, circuit: CircuitType) -> None:
        self.circuit.append_circuit_inplace(circuit.circuit)

    def copy(self, finalized: bool = False) -> PyGSTiPhysicalCircuit:
        return PyGSTiPhysicalCircuit(self.circuit.copy(editable=not finalized))

    def delete_qubits(
        self, qubits_to_delete: Iterable[QubitTypes]
    ) -> PyGSTiPhysicalCircuit:
        return super().delete_qubits(qubits_to_delete)

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

    def map_qubit_labels(
        self,
        qubit_mapping: Mapping[QubitTypes, QubitTypes],
    ) -> PyGSTiPhysicalCircuit:
        return super().map_qubit_labels(qubit_mapping)

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

    def set_qubit_labels(
        self, circuit: CircuitType, qubit_labels: Iterable[QubitTypes]
    ) -> CircuitType:
        return super().set_qubit_labels(circuit, qubit_labels)

    def set_qubit_labels_inplace(
        self, qubit_labels: Iterable[QubitTypes]
    ) -> None:
        super().set_qubit_labels_inplace(qubit_labels)
        self.circuit.line_labels = qubit_labels

    def process_circuit(
        self,
        qubit_labels: Optional[Iterable[QubitTypes]] = None,
        omit_gates: Optional[Iterable[OperationTypes]] = None,
        delete_idle_layers: bool = False,
    ) -> PyGSTiPhysicalCircuit:
        processed = self.copy(finalized=False)

        if qubit_labels is not None:
            processed.circuit.line_labels = qubit_labels

        try:
            if omit_gates is None:
                omit_gates = {}
            else:
                omit_gates = {k: [] for k in omit_gates}
            processed.circuit.change_gate_library(
                omit_gates, depth_compression=False, allow_unchanged_gates=True
            )

            if delete_idle_layers:
                processed.circuit.delete_idle_layers_inplace()
        except AssertionError as e:
            raise AssertionError(
                "Underlying circuit is static.",
                "Try using .copy(finalized=False) first",
            ) from e

        if self.finalized:
            processed.finalize_inplace()

        return processed
