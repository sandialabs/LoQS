#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

""":class:`.STIMPhysicalCircuit` definition.
"""

from __future__ import annotations

from collections.abc import Sequence, Mapping
import textwrap
from typing import ClassVar, TypeAlias, TYPE_CHECKING, Any
import warnings
import re

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

STIMCircuitCastableTypes: TypeAlias = BasePhysicalCircuit | str | _Circuit
"""Types we can cast to a STIM circuit."""

r"""
STIM circuit grammar
--------------------
<CIRCUIT> ::= <LINE>*
<LINE> ::= <INDENT> (<INSTRUCTION> | <BLOCK_START> | <BLOCK_END>)? <COMMENT>? '\n'
<BLOCK_START> ::= <INSTRUCTION> /[ \t]*/ '{'
<BLOCK_END> ::= '}' 
<INSTRUCTION> ::= <NAME> <TAG>? <PARENS_ARGUMENTS>? <TARGETS>
<NAME> ::= /[a-zA-Z][a-zA-Z0-9_]*/ 
<TAG> ::= '[' /[^\r\]\n]/* ']'
<PARENS_ARGUMENTS> ::= '(' <ARGUMENTS> ')' 
<ARGUMENTS> ::= /[ \t]*/ <ARG> /[ \t]*/ (',' <ARGUMENTS>)?
<ARG> ::= <double> 
<TARGETS> ::= /[ \t]+/ <TARG> <TARGETS>?
<TARG> ::= <QUBIT_TARGET> | <MEASUREMENT_RECORD_TARGET> | <SWEEP_BIT_TARGET> | <PAULI_TARGET> | <COMBINER_TARGET> 
<QUBIT_TARGET> ::= '!'? <uint>
<MEASUREMENT_RECORD_TARGET> ::= "rec[-" <uint> "]"
<SWEEP_BIT_TARGET> ::= "sweep[" <uint> "]"
<PAULI_TARGET> ::= '!'? /[XYZ]/ <uint>
<COMBINER_TARGET> ::= '*'
<INDENT> ::= /[ \t]*/
<COMMENT> ::= '#' /[^\n]*/
"""


def _get_used_stim_indices(circuit: _Circuit) -> list[int]:
    """Return sorted list of qubit indices that appear as qubit targets."""
    used_indices = set()
    for instruction in circuit:
        # Skip REPEAT blocks as they don't have qubit targets
        if instruction.name == "REPEAT":
            continue
        for target in instruction.targets_copy():
            if target.is_qubit_target:
                used_indices.add(target.value)
    return sorted(used_indices)


def _reindex_stim_circuit(circuit: _Circuit, index_map: dict[int, int]) -> _Circuit:
    """Return a new STIM circuit with qubit targets remapped according to index_map."""
    # Build the circuit string and parse it - this is more reliable than trying to
    # reconstruct instructions manually with the STIM API
    circuit_lines = []
    
    for instruction in circuit:
        if instruction.name == "" or instruction.name.startswith("#"):
            # Skip comments and annotations for now
            continue
            
        # Start with instruction name
        line_parts = [instruction.name]
        
        # Add gate arguments if any
        gate_args = instruction.gate_args_copy()
        if gate_args:
            line_parts.extend(str(arg) for arg in gate_args)
        
        # Process targets
        for target in instruction.targets_copy():
            if target.is_qubit_target:
                # Remap qubit target
                new_idx = index_map[target.value]
                if target.is_inverted_result_target:
                    line_parts.append(f"!{new_idx}")
                else:
                    line_parts.append(str(new_idx))
            else:
                # Pass through non-qubit targets unchanged
                if target.is_inverted_result_target:
                    line_parts.append(f"!{target.value}")
                elif target.is_measurement_record_target:
                    line_parts.append(f"rec[{target.value}]")
                elif target.is_combiner_target:
                    line_parts.append("*")
                elif target.is_relative_target:
                    line_parts.append(f"+{target.value}")
                else:
                    # Fallback: use the string representation
                    line_parts.append(str(target.value))
        
        circuit_lines.append(" ".join(line_parts))
    
    # Create new circuit from the rebuilt string
    return _Circuit("\n".join(circuit_lines))


def _separate_stimcircuit_instruction(ell: str) -> tuple[str, str]:
    """
    Assume ell is a <LINE> in the following sense.
    
        <LINE> ::= <INDENT> (<INSTRUCTION> | <BLOCK_START> | <BLOCK_END>)? <COMMENT>? '\n'
            <INDENT>      ::=  /[ \t]*/
            <COMMENT>     ::=  '#' /[^\n]*/
            <BLOCK_START> ::=  <INSTRUCTION> /[ \t]*/ '{'
            <BLOCK_END>   ::=  '}' 

    Return a pair of strings (p1, p2), where p1 = ((<INSTRUCTION> + ' ' ) | '') and 
    ell.replace('\t','    ').lstrip(' ') == p1 + p2, up to whitepsace dfferences.
    """
    ell = ell.replace('\t', '    ')
    ell = ell.lstrip(' ')

    if len(ell) == 0:
        return '', ''
    
    if '#' in ell:
        # No syntax constraints after the first '#'. The presence or absence
        # of <INSTRUCTION> is determined by the substring preceding '#'.
        ells = ell.split('#', maxsplit=1)
        p1, p2 = _separate_stimcircuit_instruction(ells[0])
        p2 = p2 + '#' + ells[1]
        return p1, p2
    
    # No comments past this point.
    if '}' in ell:
        # We match <BLOCK_END>; such lines cannot contain instructions.
        return '', ell
    elif '{' in ell:
        # We match <BLOCK_START> ::= <INSTRUCTION> /[ \t]*/ '{'
        ells = ell.split('{', maxsplit=1)
        len_before = len(ells[0])
        p1 = ells[0].rstrip(' ')
        len_after  = len(p1)
        whitespace = ' ' * (len_before - len_after)
        p2 = whitespace + '{' + ells[1]
        return p1, p2
    else:
        # We match <INSTRUCTION> directly
        return ell, ''


def _replace_instruction_targets(inst: str, targets_map: dict[str, str]) -> str:
    parts = inst.split(')', maxsplit=1)
    if len(parts) == 2:
        pre, post = parts
        post = _replace_instruction_targets(post, targets_map)
        return pre + ')' + post
    
    parts = inst.split(']', maxsplit=1)
    if len(parts) == 2:
        pre, post = parts
        post = _replace_instruction_targets(post, targets_map)
        return pre + ']' + post
    
    parts = inst.split(' ')
    for i in range(1, len(parts)):
        pi = parts[i]
        prefix = '' if (not pi.startswith('!')) else '!'
        pi = pi.lstrip('!')
        if pi in targets_map:
            pi = str(targets_map[pi])
        pi = prefix + pi
        parts[i] = pi
    inst = ' '.join(parts)
    return inst


def _as_stim_circuit(circuit: str, qubit_labels) -> _Circuit:
    lines = []
    label_map = {str(lbl): str(i) for i,lbl in enumerate(qubit_labels)}
    for line in circuit.split('\n'):
        p1, p2 = _separate_stimcircuit_instruction(line)
        if len(p1) == 0:
            lines.append(line)
        else:
            p1_mapped = _replace_instruction_targets(p1, label_map)
            line_mapped = p1_mapped + p2
            lines.append(line_mapped)
    circuit_str = '\n'.join(lines)
    c = _Circuit(circuit_str)
    return c


def _check_label_count(actual, claimed):
    if len(actual) != len(claimed):
        msg  = f"Circuit uses {len(actual)} unique qubit labels "
        msg += f"but only {len(claimed)} labels provided"
        raise ValueError(msg)


class STIMPhysicalCircuit(BasePhysicalCircuit):
    """Circuit backend using STIM."""

    _circuit: _Circuit
    """STIM circuit
    """

    _qubit_labels: list[QubitTypes]
    """list of qubit labels"""

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

    stim_command_aliases : ClassVar[dict[str,str]] = {
        'CNOT': 'CX'
    }

    def __init__(
        self,
        circuit: STIMCircuitCastableTypes,
        qubit_labels: Sequence[QubitTypes] | None = None,
        suppress_tick_warning: bool = False,
    ) -> None:
        if not is_backend_available("stim_circuit"):
            msg  = "STIM backend is not available.\n"
            msg += "Please install stim: pip install loqs[stim]"
            raise ImportError(msg)
        
        if not isinstance(circuit, (STIMPhysicalCircuit, str, _Circuit)):
            raise ValueError()

        if qubit_labels is None:
            if isinstance(circuit, STIMPhysicalCircuit):
                qubit_labels = circuit.qubit_labels
            elif isinstance(circuit, _Circuit):
                qubit_labels = _get_used_stim_indices(circuit)
            else:  # we're a plain str
                qubit_labels = _get_used_stim_indices(_Circuit(circuit))

        if isinstance(circuit, STIMPhysicalCircuit):
            _check_label_count(circuit.qubit_labels, qubit_labels)
            self._circuit = circuit.circuit.copy()
            self._qubit_labels = list(qubit_labels)
            
        elif isinstance(circuit, _Circuit):
            used_indices = _get_used_stim_indices(circuit)
            _check_label_count(used_indices, qubit_labels)
            index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}
            self._circuit = _reindex_stim_circuit(circuit, index_map)
            self._qubit_labels = list(qubit_labels)

        else: # we're a plain str
            self._circuit = _as_stim_circuit(circuit, qubit_labels)
            self._qubit_labels = list(qubit_labels)

        unsupported = ("MPP", "SPP", "SPP_DAG")
        if any([u in str(self.circuit) for u in unsupported]):
            msg = f"STIM circuit contains a LoQS-unsupported instruction {unsupported}"
            raise ValueError(msg)

        if not suppress_tick_warning and "TICK" not in str(self.circuit):
            warnings.warn(
                "No TICK instructions, layer-based functionality will not work as intended if this is more than one layer."
            )

        super().__init__(circuit, qubit_labels)
        return

    name: ClassVar[str] = "STIM"

    def __str__(self) -> str:
        s = f"Physical {self.name} circuit ({self._qubit_labels}):\n"
        s += textwrap.indent(str(self.circuit), "  ")
        return s

    @property
    def circuit(self) -> _Circuit:
        return self._circuit

    @property
    def depth(self) -> int:
        return self.circuit.num_ticks + 1

    @property
    def qubit_labels(self) -> list[QubitTypes]:
        assert len(self._qubit_labels) == self.circuit.num_qubits
        # ^ After our fix, we maintain the invariant that the STIM circuit
        #   uses exactly len(self._qubit_labels) qubits with compact indices.
        return self._qubit_labels

    def copy(self) -> STIMPhysicalCircuit:
        return STIMPhysicalCircuit(str(self._circuit), self._qubit_labels)

    def delete_qubits_inplace(
        self, qubits_to_delete: Sequence[QubitTypes]
    ) -> None:
        # Convert qubit labels to STIM indices
        qubit_idxs_to_delete = [
            self._qubit_labels.index(q) for q in qubits_to_delete
        ]

        new_lines = []
        for line in str(self.circuit).split("\n"):
            entries = line.split()
            if len(entries) == 0 or entries[0] not in self._stim_gates:
                # Empty line or not a gate, don't do qubit idx check
                pass
            elif any([str(qidx) in entries[1:] for qidx in qubit_idxs_to_delete]):
                # This has one of our qubits to delete, don't add it!
                continue

            # Otherwise, this line can be safely added
            new_lines.append(line)

        # Create temporary circuit from filtered lines
        temp_circuit = _Circuit("\n".join(new_lines))
        
        # Update qubit labels by removing deleted ones
        qubits_to_keep = []
        for q in self._qubit_labels:
            if q not in qubits_to_delete:
                qubits_to_keep.append(q)
        
        # Build index map for reindexing: old_stim_idx -> new_stim_idx
        index_map = {}
        new_idx = 0
        for old_idx in range(len(self._qubit_labels)):
            if old_idx not in qubit_idxs_to_delete:
                index_map[old_idx] = new_idx
                new_idx += 1
        
        # Reindex the circuit to maintain compact indices
        self._circuit = _reindex_stim_circuit(temp_circuit, index_map)
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

                # Convert STIM indices to LoQS labels
                if post_twoq_gates:
                    if entries[0] in self._stim_twoq_gates:
                        # Handle the case where multiple 2Q gates are defined on one line
                        for i in range(1, len(entries[1:]), 2):
                            stim_idx1 = int(entries[i])
                            stim_idx2 = int(entries[i + 1])
                            circuit_locations.append(
                                (
                                    lidx + 1,
                                    (self._qubit_labels[stim_idx1], self._qubit_labels[stim_idx2]),
                                )
                            )
                else:
                    circuit_locations.extend(
                        [(lidx, self._qubit_labels[int(q)]) for q in entries[1:]]
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
        # Pass through any unspecified qubits
        complete_mapping = {
            q: qubit_mapping.get(q, q) for q in self._qubit_labels
        }

        # For STIM, we don't need to adjust internal circuit at all,
        # since it only store qubit indices. Just update the labels!

        self._qubit_labels = [complete_mapping[q] for q in self._qubit_labels]

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

        # Build index map for the other circuit
        index_map = {}
        new_qubit_labels = list(self._qubit_labels)
        
        # Map shared qubit labels to their existing STIM indices
        for stim_idx, label in enumerate(self._qubit_labels):
            if label in other_circuit.qubit_labels:
                other_stim_idx = other_circuit.qubit_labels.index(label)
                index_map[other_stim_idx] = stim_idx
        
        # Add new qubit labels and map them to new STIM indices
        for other_stim_idx, other_label in enumerate(other_circuit.qubit_labels):
            if other_label not in self._qubit_labels:
                new_stim_idx = len(new_qubit_labels)
                index_map[other_stim_idx] = new_stim_idx
                new_qubit_labels.append(other_label)
        
        # Reindex the other circuit to use our STIM indices
        reindexed_other_circuit = _reindex_stim_circuit(
            other_circuit.circuit.copy(), index_map
        )

        layers = self._unroll_repeats().split("\nTICK\n")
        other_layers = str(reindexed_other_circuit).split("\nTICK\n")

        # Ensure circuit is long enough for merge
        end = idx + other_circuit.depth
        for lidx in range(self.depth, end):
            layers.append("")

        # Perform merge
        for lidx in range(idx, end):
            
            incoming = other_layers[lidx - idx]
            current  = layers[lidx]
            
            targets_incoming = re.findall( r'\d+', incoming )
            targets_current  = re.findall( r'\d+', current  )

            collision = set(targets_current).intersection(targets_incoming)
            
            if collision := set(targets_current).intersection(targets_incoming):
                msg  = f"Cannot merge\n{self}\nwith\n{circuit}.\n"
                msg += f"Layer {lidx} of the candidate merge has ill-posed behavior\n"
                msg += f"for target qubit(s) {collision}."
                raise ValueError(msg)

            layers[lidx] = current + '\n' + incoming

        # Check for multiple constructions applied to the same qubit.

        arg = "\nTICK\n".join(layers)
        self._circuit = _Circuit(arg)

        # Update qubit labels
        self._qubit_labels = new_qubit_labels
        return

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
                    seen_qubits.add(self._qubit_labels[int(qubit)])

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
                idx = self._qubit_labels.index(mq)
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
        # For STIM circuit, string version is already serializable
        return str(self.circuit)

    def _unroll_repeats(self) -> str:
        circuit_str = str(self.circuit)
        unrolled_lines = circuit_str.split("\n")

        def find_first_repeat_start(lines):
            for i, line in enumerate(lines):
                entries = line.split()
                if entries and entries[0] == "REPEAT":
                    return i
            return None

        def find_last_repeat_end(lines):
            for i, line in enumerate(lines[::-1]):
                entries = line.split()
                if entries and entries[0] == "}":
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

    @staticmethod
    def substitute_command_aliases(s: str) -> str:
        for k, v in STIMPhysicalCircuit.stim_command_aliases.items():
            s = s.replace(k, v)
        return s
