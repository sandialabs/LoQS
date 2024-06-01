""":class:`PyGSTiPhysicalCircuit` definition.
"""

from __future__ import annotations

from collections.abc import Sequence, Mapping
from typing import ClassVar, TypeAlias

from loqs.backends.circuit import BasePhysicalCircuit


try:
    from pygsti.circuits import Circuit as _Circuit
    from pygsti.baseobjs import Label as _Label
except ImportError as e:
    raise ImportError("Failed import, cannot use pyGSTi as backend") from e

## Type aliases for static type checking
CastableTypes: TypeAlias = (
    "PyGSTiPhysicalCircuit | _Circuit | str | Sequence[OperationTypes]"
)
"""Types we can cast to a pyGSTi circuit.

These include another PyGSTiPhysicalCircuit, a bare pyGSTi Circuit,
or things that a pyGSTi Circuit can cast (a subset of these include
a string and a list of operations/layers).
"""

QubitTypes: TypeAlias = str | int
"""PyGSTi backend qubit label types (strs and ints)"""

LayerTypes: TypeAlias = str | Sequence[str | QubitTypes] | _Label
"""Helper type alias for things allowed to be in circuit layer

For PyGSTi, this would be:
- strings, e.g. "Gcnot"
- tuples or lists of strings with either one or several qubits,
e.g. ('Gxpi2', 0) or ['Gcnot', ("Q0", "Q1")]
- actual Labels (which is what all the above cast to anyway)
- or lists and tuples of the above (in which case it is a whole layer)
"""

OperationTypes: TypeAlias = LayerTypes | Sequence[LayerTypes]
"""PyGSTi backend operations type (one or several gates/layers)"""


class PyGSTiPhysicalCircuit(BasePhysicalCircuit):
    """Circuit backend for handling ``pygsti.circuits.Circuit`` objects."""

    CastableTypes: ClassVar[TypeAlias] = CastableTypes

    def __init__(
        self,
        circuit: CastableTypes,
        qubit_labels: Sequence[QubitTypes] | None = None,
    ) -> None:
        if isinstance(circuit, PyGSTiPhysicalCircuit):
            self._circuit = circuit.circuit
        else:
            try:
                self._circuit = _Circuit.cast(circuit)
            except Exception as e:
                raise ValueError("Failed to cast to pyGSTi circuit") from e

        # Keep our version of the circuit mutable
        # We ignore a warning from pyGSTi (should be str|bool, but is just str)
        self._circuit = self._circuit.copy(editable=True)  # type: ignore

        super().__init__(circuit, qubit_labels)

    name: ClassVar[str] = "pyGSTi"

    @property
    def circuit(self) -> _Circuit:
        return self._circuit

    @property
    def depth(self) -> int:
        return self._circuit.depth

    @property
    def qubit_labels(self) -> Sequence[QubitTypes]:
        return self.circuit.line_labels

    def copy(self) -> PyGSTiPhysicalCircuit:
        return PyGSTiPhysicalCircuit(self.circuit, self.qubit_labels)

    def delete_qubits_inplace(
        self, qubits_to_delete: Sequence[QubitTypes]
    ) -> None:
        self.circuit.delete_lines(qubits_to_delete, delete_straddlers=True)

    def insert_inplace(self, circuit: BasePhysicalCircuit, idx: int) -> None:
        other_circuit: _Circuit = PyGSTiPhysicalCircuit.cast(circuit).circuit
        self.circuit.insert_circuit_inplace(other_circuit, idx)

    def map_qubit_labels_inplace(
        self, qubit_mapping: Mapping[QubitTypes, QubitTypes]
    ) -> None:
        # Pass through any unspecified qubits
        complete_mapping = {
            q: qubit_mapping.get(q, q) for q in self.circuit.line_labels
        }

        self.circuit.map_state_space_labels_inplace(complete_mapping)

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
        self.circuit.line_labels = qubit_labels

    def process_circuit(
        self,
        qubit_mapping: Mapping[QubitTypes, QubitTypes] | None = None,
        omit_gates: Sequence[OperationTypes] | None = None,
        delete_idle_layers: bool = False,
    ) -> PyGSTiPhysicalCircuit:
        processed = self.copy()

        if qubit_mapping is not None:
            processed.map_qubit_labels_inplace(qubit_mapping)

        if omit_gates is not None:
            for og in omit_gates:
                processed.circuit.replace_gatename_with_idle_inplace(og)

        if delete_idle_layers:
            processed.circuit.delete_idle_layers_inplace()

        return processed
