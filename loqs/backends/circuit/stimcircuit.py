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
        "REPEAT",
        "DETECTOR",
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

    _stim_oneq_gates: ClassVar[list[str]] = [
        "I",
        "X",
        "Y",
        "Z",
        "C_XYZ",
        "C_ZYX",
        "H",
        "H_XY",
        "H_XZ",
        "H_YZ",
        "S",
        "SQRT_X",
        "SQRT_X_DAG",
        "SQRT_Y",
        "SQRT_Y_DAG",
        "SQRT_Z",
        "SQRT_Z_DAG",
        "S_DAG",
    ]
    """STIM 1Q gates.
    """

    _stim_twoq_gates: ClassVar[list[str]] = [
        "CNOT",
        "CX",
        "CXSWAP",
        "CY",
        "CZ",
        "CZSWAP",
        "ISWAP",
        "ISWAP_DAG",
        "SQRT_XX",
        "SQRT_XX_DAG",
        "SQRT_YY",
        "SQRT_YY_DAG",
        "SQRT_ZZ",
        "SQRT_ZZ_DAG",
        "SWAP",
        "SWAPCX",
        "SWAPCZ",
        "XCX",
        "XCY",
        "XCZ",
        "YCX",
        "YCY",
        "YCZ",
        "ZCX",
        "ZCY",
        "ZCZ",
    ]
    """STIM 2Q gates.

    Number of qubits is an unreliable way to keep track
    of 2Q gates, since multiple 2Q gates can be defined
    in one line as pairs of indices. So we keep the 2Q
    gate names to check for them.
    """

    _stim_measure_reset_gates: ClassVar[list[str]] = [
        "M",
        "MR",
        "MRX",
        "MRY",
        "MRZ",
        "MX",
        "MY",
        "MZ",
        "R",
        "RX",
        "RY",
        "RZ",
        "MXX",
        "MYY",
        "MZZ",
    ]
    """STIM measure and reset gates.

    These may be treated differently as arguments may
    include ! before the qubit index to denote that the
    measurement should be flipped before being recorded.
    """

    _stim_noise_channels: ClassVar[list[str]] = [
        "CORRELATED_ERROR",
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "E",
        "ELSE_CORRELATED_ERROR",
        "HERALDED_ERASE",
        "HERALDED_PAULI_CHANNEL_1",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
    ]
    """STIM noise channels.

    These should probably not be part of a circuit
    prior to it going through a :class:`.BaseNoiseModel`,
    but currently they will just pass through.
    """

    _stim_gates: ClassVar[list[str]] = (
        _stim_oneq_gates + _stim_twoq_gates + _stim_measure_reset_gates
    )
    """STIM 1Q, 2Q, and measurement gates.

    These are the STIM instructions that will be treated
    as possible keys into a :class:`.STIMDictNoiseModel`.
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
        elif isinstance(circuit, _Circuit):
            self._circuit = circuit
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
                "No TICK instructions, layer-based functionality will not work as intended if this is more than one layer."
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
        assert len(self._qubit_labels) >= self.circuit.num_qubits
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
            if len(entries) == 0 or entries[0] not in self._stim_gates:
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

    def get_possible_discrete_error_locations(
        self, post_twoq_gates: bool = False
    ) -> list[tuple[int, int | tuple[int, ...]]]:
        circuit_locations: list[tuple[int, int | tuple[int, ...]]] = []
        unrolled_str = self._unroll_repeats()
        for lidx, lstr in enumerate(unrolled_str.split("TICK\n")):
            for line in lstr.split("\n"):
                entries = line.split()
                if len(entries) == 0 or entries[0] not in self._stim_gates:
                    # Empty line or not a gate, skip to next line
                    continue

                # Normally we would look up the qubit index,
                # but for stim, the circuit uses indices already
                if post_twoq_gates:
                    if entries[0] in self._stim_twoq_gates:
                        # Handle the case where multiple 2Q gates are defined on one line
                        for i in range(1, len(entries[1:]), 2):
                            circuit_locations.append(
                                (
                                    lidx + 1,
                                    (int(entries[i]), int(entries[i + 1])),
                                )
                            )
                else:
                    circuit_locations.extend(
                        [(lidx, int(q)) for q in entries[1:]]
                    )

                # Otherwise, each qubit index is a location
                for qidx in entries[1]:
                    circuit_locations.append((lidx, int(qidx)))
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
                if len(entries) == 0 or entries[0] not in self._stim_gates:
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
            for mq in missing_qubits:
                idx = self.qubit_labels.index(mq)
                new_circ_str += f"\n{layer_idle} {idx}"

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
