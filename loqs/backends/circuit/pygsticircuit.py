""":class:`.PyGSTiPhysicalCircuit` definition.
"""

from __future__ import annotations

from collections.abc import Sequence, Mapping
from typing import ClassVar, TypeAlias, TypeVar

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.circuit.listcircuit import ListPhysicalCircuit

try:
    from pygsti.circuits import Circuit as _Circuit
    from pygsti.baseobjs import Label as _Label
except ImportError as e:
    raise ImportError("Failed import, cannot use pyGSTi as backend") from e

T = TypeVar("T", bound="PyGSTiPhysicalCircuit")

## Type aliases for static type checking
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

PyGSTiCircuitCastableTypes: TypeAlias = (
    BasePhysicalCircuit | _Circuit | str | Sequence[OperationTypes]
)
"""Types we can cast to a pyGSTi circuit.

These include another PyGSTiPhysicalCircuit, a bare pyGSTi Circuit,
or things that a pyGSTi Circuit can cast (a subset of these include
a string and a list of operations/layers).
"""


class PyGSTiPhysicalCircuit(BasePhysicalCircuit):
    """Circuit backend for handling ``pygsti.circuits.Circuit`` objects."""

    def __init__(
        self,
        circuit: PyGSTiCircuitCastableTypes,
        qubit_labels: Sequence[QubitTypes] | None = None,
    ) -> None:
        if isinstance(circuit, PyGSTiPhysicalCircuit):
            self._circuit = circuit.circuit
        elif isinstance(circuit, ListPhysicalCircuit):
            try:
                expanded_circuit = [
                    [(t[0], *t[1]) for t in layer] for layer in circuit.circuit
                ]
                self._circuit = _Circuit.cast(expanded_circuit)
            except Exception as e:
                raise ValueError(
                    "Failed to cast list circuit to pyGSTi circuit"
                ) from e
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

    def get_possible_discrete_error_locations(self) -> list[tuple[int, int]]:
        circuit_locations = []
        for lidx in range(self._circuit.depth):
            for comp in self._circuit._layer_components(lidx):
                for q in comp.qubits:
                    circuit_locations.append(
                        (lidx, self.qubit_labels.index(q))
                    )
        return circuit_locations

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

    def merge_inplace(self, circuit: BasePhysicalCircuit, idx: int) -> None:
        other_circuit: _Circuit = PyGSTiPhysicalCircuit.cast(circuit).circuit
        end = idx + other_circuit.depth

        # Ensure circuit is long enough for merge
        for lidx in range(self._circuit.depth, end):
            self._circuit.insert_layer_inplace([], lidx)

        # Perform merge
        for lidx in range(idx, end):
            comps = self._circuit._layer_components(
                lidx
            ) + other_circuit._layer_components(lidx - idx)
            self._circuit.set_labels(comps, lidx)

    def pad_single_qubit_idles_by_duration_inplace(
        self,
        idle_names: Mapping[int | float, str],
        durations: Mapping[str, int | float],
        default_duration: int | float | None = None,
        empty_layer_idle: str | None = None,
    ) -> None:
        for lidx in range(self._circuit.depth):
            comps = self._circuit._layer_components(lidx)

            # Check which qubits are not idling
            seen_qubits = set()
            layer_duration = None
            for comp in comps:
                if comp.qubits is None:
                    # This has no qubit labels, assume it is a whole layer instruction
                    seen_qubits = set(self._circuit.line_labels)
                    continue

                duration = durations.get(comp.name, default_duration)
                if duration is None:
                    raise KeyError(
                        f"No duration for {comp.name} or default specified"
                    )
                if layer_duration is None:
                    layer_duration = duration
                else:
                    layer_duration = max(layer_duration, duration)

                for qubit in comp.qubits:
                    seen_qubits.add(qubit)

            # Get idling operation (or skip for empty layers with no idles)
            if layer_duration is None and empty_layer_idle is None:
                continue
            elif layer_duration is None:
                layer_idle = empty_layer_idle
            else:
                layer_idle = idle_names[layer_duration]

            # Insert idling operations
            missing_qubits = set(self._circuit.line_labels) - seen_qubits
            for qubit in missing_qubits:
                comps.append(_Label(layer_idle, (qubit,)))

            # Substitute new padded layer
            self._circuit.set_labels(comps, lidx)

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

    @classmethod
    def _deserialize_circuit(
        cls,
        serial_circuit: str | list | dict,
        qubit_labels: Sequence | None = None,
    ) -> _Circuit:
        """Helper function to deserialize a circuit.

        Derived classes should implement this for
        deserialization to work.
        """
        # For pyGSTi circuit, we can load from string rep
        # (minus leading "Circuit(" and trailing ")" )
        assert isinstance(serial_circuit, str)
        cstr = serial_circuit[8:-1]
        line_labels = cstr.split("@")[1][1:-1].split(",")

        if qubit_labels is None:
            qubit_labels = line_labels
        else:
            assert set(line_labels) == set(qubit_labels)

        # However, qubit labels must be ints or start with "Q"
        # First, lets map our string to have Q labels
        # Do this through temp in case some of them already have Q labels
        # (common convention)
        old_to_temp = {lbl: f"TEMP{i}" for i, lbl in enumerate(qubit_labels)}
        temp_to_new = {f"TEMP{i}": f"Q{i}" for i in range(len(qubit_labels))}

        for k, v in old_to_temp.items():
            cstr = cstr.replace(k, v)
        for k, v in temp_to_new.items():
            cstr = cstr.replace(k, v)

        # Now let the parser at it
        try:
            circ = _Circuit(cstr, editable=True)
        except ValueError as e:
            raise ValueError(
                f"Failed to parse circuit string {serial_circuit}"
            ) from e

        # And convert state space labels back
        circ.map_state_space_labels_inplace(
            {v: k for k, v in temp_to_new.items()}
        )
        circ.map_state_space_labels_inplace(
            {v: k for k, v in old_to_temp.items()}
        )
        circ.line_labels = qubit_labels

        return circ

    def _serialize_circuit(self) -> str | list | dict:
        """Helper function to serialize a circuit.

        Derived classes should implement this for
        serialization to work.
        """
        # For pyGSTi circuit, we use the string rep
        return repr(self.circuit)
