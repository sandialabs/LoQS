#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################



from __future__ import annotations

from collections.abc import Sequence, Mapping
import textwrap
from typing import ClassVar, TypeAlias, TYPE_CHECKING, Any
import warnings

from loqs.backends import BasePhysicalCircuit, is_backend_available

# Conditional imports for STIM
if TYPE_CHECKING:
    # Type checking imports - these won't be executed at runtime
    from stim import Circuit as _Circuit
else:
    # Runtime imports - these will be attempted only when needed
    try:
        from stim import Circuit as _Circuit
    except ImportError:
        _Circuit = Any  # type: ignore

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
    prior to it going through a [](api:BaseNoiseModel),
    but currently they will just pass through.
    """

    _stim_gates: ClassVar[list[str]] = (
        _stim_oneq_gates + _stim_twoq_gates + _stim_measure_reset_gates
    )
    """STIM 1Q, 2Q, and measurement gates.

    These are the STIM instructions that will be treated
    as possible keys into a [](api:STIMDictNoiseModel).
    """

    def __init__(
        self,
        circuit: STIMCircuitCastableTypes,
        qubit_labels: Sequence[QubitTypes] | None = None,
        suppress_tick_warning: bool = False,
    ) -> None:
        if not is_backend_available("stim_circuit"):
            raise ImportError(
                "STIM backend is not available. "
                "Please install stim: pip install loqs[stim]"
            )
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

        if not suppress_tick_warning and "TICK" not in str(self.circuit):
            warnings.warn(
                "No TICK instructions, layer-based functionality will not work as intended if this is more than one layer."
            )

        super().__init__(circuit, qubit_labels)

    name: ClassVar[str] = "STIM"

    def __str__(self) -> str:
        s = f"Physical {self.name} circuit ({self.qubit_labels}):\n"
        s += textwrap.indent(str(self.circuit), "  ")
        return s

    @property
    def circuit(self) -> _Circuit:
        """Get the underlying STIM circuit object.

        Returns
        -------
        _Circuit
            The underlying stim.Circuit object.

        REVIEW_NO_DOCSTRING
        """
        return self._circuit

    @property
    def depth(self) -> int:
        """Get the depth of the circuit.

        The depth is calculated as the number of ticks plus one.

        Returns
        -------
        int
            The depth of the circuit.

        REVIEW_NO_DOCSTRING
        """
        return self.circuit.num_ticks + 1

    @property
    def qubit_labels(self) -> list[QubitTypes]:
        """Get the list of qubit labels for this circuit.

        Returns
        -------
        list[QubitTypes]
            List of qubit labels, where each label corresponds to a qubit in the circuit.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
        assert len(self._qubit_labels) >= self.circuit.num_qubits
        return self._qubit_labels

    def copy(self) -> STIMPhysicalCircuit:
        """Create a copy of this circuit.

        Returns
        -------
        STIMPhysicalCircuit
            A new circuit object with the same circuit and qubit labels.

        REVIEW_NO_DOCSTRING
        """
        return STIMPhysicalCircuit(str(self._circuit), self.qubit_labels)

    def delete_qubits_inplace(
        self, qubits_to_delete: Sequence[QubitTypes]
    ) -> None:
        """Delete qubits from the circuit in-place.

        Parameters
        ----------
        qubits_to_delete : Sequence[QubitTypes]
            Sequence of qubit labels to delete from the circuit.

        REVIEW_NO_DOCSTRING
        """
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
        """Get possible discrete error locations in the circuit.

        This method identifies locations in the circuit where discrete errors
        could potentially occur. It can optionally focus on locations after
        two-qubit gates.

        Parameters
        ----------
        post_twoq_gates : bool, optional
            If True, only return locations after two-qubit gates. Default is False.

        Returns
        -------
        list[tuple[int, int | tuple[int, ...]]]
            List of circuit locations where discrete errors could occur.
            Each location is represented as a tuple of (layer_index, qubit_index)
            or (layer_index, (qubit1_index, qubit2_index)) for two-qubit gates.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
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
        pre_str = "\nTICK\n".join(layers[:idx]) + "\n"
        post_str = "\n" + "\nTICK\n".join(layers[idx:])

        self._circuit = _Circuit(
            pre_str + str(other_circuit.circuit) + post_str
        )

    def map_qubit_labels_inplace(
        self, qubit_mapping: Mapping[QubitTypes, QubitTypes]
    ) -> None:
        """Map qubit labels in-place according to a provided mapping.

        This method updates the qubit labels in the circuit based on the provided
        mapping dictionary. Qubits not specified in the mapping will retain their
        original labels.

        Parameters
        ----------
        qubit_mapping : Mapping[QubitTypes, QubitTypes]
            Dictionary mapping current qubit labels to new qubit labels.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
        # Pass through any unspecified qubits
        complete_mapping = {
            q: qubit_mapping.get(q, q) for q in self.qubit_labels
        }

        # For STIM, we don't need to adjust internal circuit at all,
        # since it only store qubit indices. Just update the labels!

        self._qubit_labels = [complete_mapping[q] for q in self.qubit_labels]

    def merge_inplace(self, circuit: BasePhysicalCircuit, idx: int) -> None:
        """Merge another circuit to this circuit.

        While [insert_inplace](api:STIMPhysicalCircuit.insert_inplace) adds new layers,
        [merge_inplace](api:STIMPhysicalCircuit.merge_inplace) will try to add operations to
        existing layers.

        Note that for STIM circuits, this will first unroll repeat blocks
        in the both circuits to ensure merging of correct layers.

        Parameters
        ----------
        circuit : BasePhysicalCircuit
            Circuit to merge

        idx : int
            Layer index to start merge

        REVIEW_SPHINX_REFERENCE
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
        """Pad single qubit idles by duration in-place.

        This method adds idle operations to qubits that are not being used in a layer,
        based on the duration of operations in that layer. This ensures that all qubits
        have operations that span the same duration, which can be important for accurate
        simulation and timing.

        Parameters
        ----------
        idle_names : Mapping[int | float, str]
            Mapping from durations to idle operation names. The idle operation
            corresponding to the layer's duration will be used for padding.

        durations : Mapping[str, int | float]
            Mapping from operation names to their durations. Used to determine
            the duration of each layer.

        default_duration : int | float | None, optional
            Default duration to use if an operation's duration is not specified
            in the durations mapping. If None and an operation's duration is not
            specified, a KeyError will be raised.

        empty_layer_idle : str | None, optional
            Idle operation to use for empty layers (layers with no operations).
            If None, empty layers will not be padded.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
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
        """Set the qubit labels for this circuit in-place.

        Parameters
        ----------
        qubit_labels : Sequence[QubitTypes]
            Sequence of new qubit labels to set for the circuit.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
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
        # For STIM circuit, string version is already serializable
        return str(self.circuit)

    def _unroll_repeats(self) -> str:
        circuit_str = str(self.circuit)
        unrolled_lines = circuit_str.split("\n")

        def find_first_repeat_start(lines):
            """Find the first REPEAT statement in a list of circuit lines.

            This function searches through a list of circuit lines and returns the index
            of the first line that contains a REPEAT statement.

            Parameters
            ----------
            lines : list[str]
                List of circuit lines to search through.

            Returns
            -------
            int or None
                Index of the first REPEAT statement, or None if no REPEAT statement is found.

            Notes
            -----
            REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
            previously had no documentation. Please review and update as needed.
            """
            for i, line in enumerate(lines):
                entries = line.split()
                if entries[0] == "REPEAT":
                    return i
            return None

        def find_last_repeat_end(lines):
            """Find the last REPEAT end statement in a list of circuit lines.

            This function searches through a list of circuit lines in reverse order and returns the index
            of the last line that contains a closing brace '}' for a REPEAT block.

            Parameters
            ----------
            lines : list[str]
                List of circuit lines to search through.

            Returns
            -------
            int or None
                Index of the last REPEAT end statement, or None if no REPEAT end statement is found.

            Notes
            -----
            REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
            previously had no documentation. Please review and update as needed.
            """
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
