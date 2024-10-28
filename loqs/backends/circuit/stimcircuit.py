""":class:`.STIMPhysicalCircuit` definition.
"""

from __future__ import annotations

from collections.abc import Sequence, Mapping
from typing import ClassVar, TypeAlias
import warnings

from loqs.backends.circuit import BasePhysicalCircuit

try:
    from stim import Circuit as _Circuit
except ImportError as e:
    raise ImportError("Failed import, cannot use STIM as backend") from e

## Type aliases for static type checking
QubitTypes: TypeAlias = str | int
"""Qubit types for builtins"""

STIMCircuitCastableTypes: TypeAlias = BasePhysicalCircuit | str
"""Types we can cast to a STIM circuit.
"""


class STIMPhysicalCircuit(BasePhysicalCircuit):
    """Circuit backend using STIM."""

    _circuit: _Circuit
    """STIM circuit
    """

    _qubit_labels: list[QubitTypes]
    """List of qubit labels"""

    _stim_annotations: ClassVar[list[str]] = [
        "REPEAT" "DETECTOR",
        "MPAD",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "TICK",
    ]
    """STIM control or annotations.

    These instructions are handled differently
    (or often ignored) by many circuit manipulation
    functions.
    """

    def __init__(
        self,
        circuit: STIMCircuitCastableTypes,
        qubit_labels: Sequence[QubitTypes] | None = None,
    ) -> None:
        if isinstance(circuit, STIMPhysicalCircuit):
            self._circuit = circuit.circuit.copy()
            self._qubit_labels = circuit.qubit_labels
        elif isinstance(circuit, str):
            self._circuit = _Circuit(circuit)
            self._qubit_labels = list(range(self.circuit.num_qubits))
        elif isinstance(circuit, BasePhysicalCircuit):
            raise NotImplementedError(
                "Have not implemented this conversion yet"
            )
        else:
            raise ValueError("Expected BasePhysicalCircuit or list of layers")

        unsupported = ("MPP", "SPP", "SPP_DAG")
        if any([u in str(self.circuit) for u in unsupported]):
            raise ValueError(
                f"STIM circuit contains a LoQS-unsupported instruction {unsupported}"
            )

        if "TICK" not in str(self.circuit):
            warnings.warn(
                "No TICK instructions, layer-based functionality will not work as intended."
            )

        super().__init__(circuit, qubit_labels)

    name: ClassVar[str] = "STIM"

    @property
    def circuit(self) -> _Circuit:
        return self._circuit

    @property
    def depth(self) -> int:
        return self.circuit.num_ticks + 1

    @property
    def qubit_labels(self) -> list[QubitTypes]:
        assert len(self._qubit_labels) > self.circuit.num_qubits
        return self._qubit_labels

    def copy(self) -> STIMPhysicalCircuit:
        return STIMPhysicalCircuit(str(self._circuit), self.qubit_labels)

    def delete_qubits_inplace(
        self, qubits_to_delete: Sequence[QubitTypes]
    ) -> None:
        qubit_idxs_to_delete = [
            str(self.qubit_labels.index(q)) for q in qubits_to_delete
        ]

        new_lines = []
        for line in str(self.circuit).split("\n"):
            entries = line.split()
            if len(entries) == 0 or entries[0] in self._stim_annotations:
                # Empty line or not a gate, don't do qubit idx check
                pass
            elif any([qidx in qubit_idxs_to_delete for qidx in entries[1:]]):
                # This has one of our qubits to delete, don't add it!
                continue

            # Otherwise, this line can be safely added
            new_lines.append(line)

        self._circuit = _Circuit("\n".join(new_lines))

        qubits_to_keep = []
        for q in self._qubit_labels:
            if q not in qubits_to_delete:
                qubits_to_keep.append(q)
        self._qubit_labels = qubits_to_keep

    def get_possible_discrete_error_locations(self) -> list[tuple[int, int]]:
        circuit_locations = []
        unrolled_str = self._unroll_repeats()
        for lidx, lstr in enumerate(unrolled_str.split("TICK\n")):
            for line in lstr.split("\n"):
                entries = line.split()
                if len(entries) == 0 or entries[0] in self._stim_annotations:
                    # Empty line or not a gate, skip to next line
                    continue

                # Otherwise, each qubit index is a location
                for qidx in entries[1]:
                    # Normally we would look up the qubit index,
                    # but for stim, the circuit uses indices already
                    circuit_locations.append((lidx, qidx))
        return circuit_locations

    def insert_inplace(self, circuit: BasePhysicalCircuit, idx: int) -> None:
        """Insert another circuit to this circuit.

        Note that for STIM circuits, this will first unroll repeat blocks
        in the current circuit to ensure insertion at the correct location.
        Repeats in the inserted circuit will be maintained.

        Parameters
        ----------
        circuit:
            Circuit to insert

        idx:
            Starting index to begin insert. If -1, append to the end.
        """
        other_circuit = STIMPhysicalCircuit.cast(circuit)

        unrolled = self._unroll_repeats()
        layers = unrolled.split("TICK\n")
        pre_str = "TICK\n".join(layers[:idx]) + "TICK\n"
        post_str = "TICK\n".join(layers[idx:])

        self._circuit = _Circuit(pre_str + str(other_circuit) + post_str)

    def map_qubit_labels_inplace(
        self, qubit_mapping: Mapping[QubitTypes, QubitTypes]
    ) -> None:
        # Pass through any unspecified qubits
        complete_mapping = {
            q: qubit_mapping.get(q, q) for q in self.qubit_labels
        }

        # For STIM, we don't need to adjust internal circuit at all,
        # since it only store qubit indices. Just update the labels!

        self._qubit_labels = [complete_mapping[q] for q in self.qubit_labels]

    def merge_inplace(self, circuit: BasePhysicalCircuit, idx: int) -> None:
        """Merge another circuit to this circuit.

        While :meth:`.insert_inplace` adds new layers,
        :meth:`.merge_inplace` will try to add operations to
        existing layers.

        Note that for STIM circuits, this will first unroll repeat blocks
        in the both circuits to ensure merging of correct layers.

        Parameters
        ----------
        circuit:
            Circuit to merge

        idx:
            Layer index to start merge
        """
        other_circuit = STIMPhysicalCircuit.cast(circuit)

        layers = self._unroll_repeats().split("TICK\n")
        other_layers = other_circuit._unroll_repeats().split("TICK\n")

        # Ensure circuit is long enough for merge
        end = idx + other_circuit.depth
        for lidx in range(self.depth, end):
            layers.append("")

        # Perform merge
        for lidx in range(idx, end):
            layers[lidx] += other_layers[lidx - idx]

        self._circuit = _Circuit("TICK\n".join(layers))

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
        # We don't need to unroll for this, works fine with repeat blocks
        new_circ_str = ""
        for lstr in str(self.circuit).split("TICK\n"):
            # Check with qubits are not idling and compute duration
            seen_qubits = set()
            layer_duration = None
            for line in lstr.split("\n"):
                entries = line.split()
                if len(entries) == 0 or entries[0] in self._stim_annotations:
                    continue

                duration = durations.get(entries[0], default_duration)
                if duration is None:
                    raise KeyError(
                        f"No duration for {entries[0]} or default specified"
                    )
                if layer_duration is None:
                    layer_duration = duration
                else:
                    layer_duration = max(layer_duration, duration)

                for qubit in entries[1:]:
                    seen_qubits.add(self.qubit_labels[int(qubit)])

            # Get idling operation (or skip for empty layers with no idles)
            if layer_duration is None:
                if empty_layer_idle is None:
                    continue
                layer_idle = empty_layer_idle
            else:
                layer_idle = idle_names[layer_duration]

            # Add existing layer
            new_circ_str += lstr

            # Insert idling operations
            missing_qubits = set(self._qubit_labels) - seen_qubits
            for qubit in missing_qubits:
                new_circ_str += (
                    "\n" + f"{layer_idle} {self.qubit_labels.index(qubit)}"
                )

            # Finish layer
            new_circ_str += "\nTICK\n"

        self._circuit = _Circuit(new_circ_str)

    def set_qubit_labels_inplace(
        self, qubit_labels: Sequence[QubitTypes]
    ) -> None:
        self._qubit_labels = list(qubit_labels)

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
        # For STIM circuit, it is already deserializable from str
        # qubit_labels not needed
        assert isinstance(serial_circuit, str)
        return _Circuit(serial_circuit)

    def _serialize_circuit(self) -> str | list | dict:
        """Helper function to serialize a circuit.

        Derived classes should implement this for
        serialization to work.
        """
        # For SIM circuit, string version is already serializable
        return str(self.circuit)

    def _unroll_repeats(self) -> str:
        circuit_str = str(self.circuit)
        unrolled_lines = circuit_str.split("\n")

        def find_first_repeat_start(lines):
            for i, line in enumerate(lines):
                entries = line.split()
                if entries[0] == "REPEAT":
                    return i
            return None

        def find_last_repeat_end(lines):
            for i, line in enumerate(lines[::-1]):
                entries = line.split()
                if entries[0] == "}":
                    return len(lines) - i - 1
            return None

        start = find_first_repeat_start(unrolled_lines)
        while start is not None:
            end = find_last_repeat_end(unrolled_lines)
            assert end is not None, "Misformed REPEAT (no closing })"

            try:
                num_repeats = int(unrolled_lines[start].split()[1])
            except ValueError as e:
                raise ValueError("Could not cast number of repeats") from e

            # Unroll lines, skipping the REPEAT <num> { and } lines
            new_unrolled_lines = unrolled_lines[:start]
            for _ in range(num_repeats):
                new_unrolled_lines.extend(unrolled_lines[start + 1 : end])
            new_unrolled_lines.extend(unrolled_lines[end + 1 :])

            # Set for next cycle
            unrolled_lines = new_unrolled_lines
            start = find_first_repeat_start(unrolled_lines)

        return "\n".join(unrolled_lines)
