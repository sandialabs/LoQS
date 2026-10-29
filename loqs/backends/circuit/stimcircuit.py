#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
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
    from stim import CircuitRepeatBlock as _CircuitRepeatBlock
else:
    # Runtime imports - these will be attempted only when needed
    try:
        from stim import Circuit as _Circuit
        from stim import CircuitRepeatBlock as _CircuitRepeatBlock
    except ImportError:
        _Circuit = Any  # type: ignore
        _CircuitRepeatBlock = Any  # type: ignore

## Type aliases for static type checking
QubitTypes: TypeAlias = str | int
"""Qubit types for builtins"""

STIMCircuitLike: TypeAlias = BasePhysicalCircuit | str | _Circuit
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


def _is_pauli_target(target) -> bool:
    """Whether `target` is a Pauli-basis-annotated qubit target (e.g. the
    `X0`/`Y1`/`Z2` targets of an `MPP` instruction) -- these reference a
    qubit index just like a plain qubit target, but `is_qubit_target` is
    `False` for them.
    """
    return target.is_x_target or target.is_y_target or target.is_z_target


def _get_used_stim_indices(circuit: _Circuit) -> list[int]:
    """Return sorted list of qubit indices that appear as qubit targets.

    Recurses into ``REPEAT`` block bodies, since qubits referenced only
    inside a repeat block are still genuinely used by the circuit.
    """
    used_indices = set()
    for instruction in circuit:
        if isinstance(instruction, _CircuitRepeatBlock):
            used_indices.update(
                _get_used_stim_indices(instruction.body_copy())
            )
            continue
        for target in instruction.targets_copy():
            if target.is_qubit_target or _is_pauli_target(target):
                used_indices.add(target.value)
    return sorted(used_indices)


def _reindex_stim_circuit_lines(
    circuit: _Circuit, index_map: dict[int, int]
) -> list[str]:
    """Return the list of remapped circuit-text lines for `circuit`
    (recursing into any `REPEAT` block bodies), without parsing them back
    into a `stim.Circuit` -- used by `_reindex_stim_circuit` itself, and
    recursively for nested `REPEAT` blocks.
    """
    circuit_lines = []

    for instruction in circuit:
        if isinstance(instruction, _CircuitRepeatBlock):
            body_lines = _reindex_stim_circuit_lines(
                instruction.body_copy(), index_map
            )
            circuit_lines.append(f"REPEAT {instruction.repeat_count} {{")
            circuit_lines.extend(body_lines)
            circuit_lines.append("}")
            continue

        if instruction.name == "" or instruction.name.startswith("#"):
            # Skip comments and annotations for now
            continue

        # Start with instruction name, attaching any gate arguments
        # directly in parentheses (e.g. "X_ERROR(0.1)"), matching STIM's
        # actual syntax -- these are not separate space-separated tokens.
        gate_args = instruction.gate_args_copy()
        if gate_args:
            args_str = ", ".join(str(arg) for arg in gate_args)
            line_parts = [f"{instruction.name}({args_str})"]
        else:
            line_parts = [instruction.name]

        # Process targets
        for target in instruction.targets_copy():
            if target.is_qubit_target or _is_pauli_target(target):
                # Remap qubit target (optionally Pauli-basis-annotated,
                # e.g. the X0/Y1/Z2 targets of an MPP instruction)
                new_idx = index_map[target.value]
                prefix = (
                    "X"
                    if target.is_x_target
                    else (
                        "Y"
                        if target.is_y_target
                        else "Z" if target.is_z_target else ""
                    )
                )
                inv = "!" if target.is_inverted_result_target else ""
                line_parts.append(f"{inv}{prefix}{new_idx}")
            else:
                # Pass through non-qubit targets unchanged
                if target.is_inverted_result_target:
                    line_parts.append(f"!{target.value}")
                elif target.is_measurement_record_target:
                    line_parts.append(f"rec[{target.value}]")
                elif target.is_combiner:
                    line_parts.append("*")
                elif target.is_sweep_bit_target:
                    line_parts.append(f"sweep[{target.value}]")
                else:
                    # Fallback: use the string representation
                    line_parts.append(str(target.value))

        circuit_lines.append(" ".join(line_parts))

    return circuit_lines


def _reindex_stim_circuit(
    circuit: _Circuit, index_map: dict[int, int]
) -> _Circuit:
    """Return a new STIM circuit with qubit targets remapped according to index_map."""
    # Build the circuit string and parse it - this is more reliable than trying to
    # reconstruct instructions manually with the STIM API
    circuit_lines = _reindex_stim_circuit_lines(circuit, index_map)

    # Create new circuit from the rebuilt string
    return _Circuit("\n".join(circuit_lines))


def _split_stim_layers(
    text: str,
) -> tuple[list[tuple[list[str], str]], bool]:
    """Split circuit text into layers of ``(lines, closing TICK line)``.

    A ``TICK`` line is ``TICK`` or ``TICK[...]`` at any indentation, and
    its tag is kept. A final layer with no closing line gets ``"TICK"``.
    Returns the layers and whether the text ended with a ``TICK`` line
    (blank lines after it don't count as a layer).
    """
    layers: list[tuple[list[str], str]] = []
    current: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "TICK" or stripped.startswith("TICK["):
            layers.append((current, stripped))
            current = []
        elif stripped:
            current.append(line)
    terminated = bool(layers) and not current
    if current:
        layers.append((current, "TICK"))
    return layers, terminated


def _join_stim_layers(
    layers: list[tuple[list[str], str]], terminated: bool
) -> str:
    """Write layers back as text; the last layer's ``TICK`` line is written
    only if ``terminated``."""
    out: list[str] = []
    for i, (lines, tick) in enumerate(layers):
        out.extend(lines)
        if terminated or i < len(layers) - 1:
            out.append(tick)
    return "\n".join(out)


def _separate_stimcircuit_instruction(ell: str) -> tuple[str, str]:
    """Split a single STIM-circuit line into its instruction and remainder.

    Assume ``ell`` is a <LINE> in the following sense.

        <LINE> ::= <INDENT> (<INSTRUCTION> | <BLOCK_START> | <BLOCK_END>)? <COMMENT>? '\\n'
            <INDENT>      ::=  /[ \\t]*/
            <COMMENT>     ::=  '#' /[^\\n]*/
            <BLOCK_START> ::=  <INSTRUCTION> /[ \\t]*/ '{'
            <BLOCK_END>   ::=  '}'

    Returns a pair of strings ``(p1, p2)`` where ``p1`` is the
    ``<INSTRUCTION>`` portion of the line (no trailing whitespace) and
    ``p2`` is the remainder (any trailing whitespace, ``{`` or ``}``
    block delimiter, and/or ``<COMMENT>``). If the line contains no
    instruction, ``p1`` is the empty string and ``p2`` carries the whole
    line (modulo the tab-expansion / leading-whitespace stripping that
    this function performs).

    Concatenation reproduces the de-tabbed, left-stripped line: i.e.
    ``ell.replace('\\t', '    ').lstrip(' ') == p1 + p2`` up to a
    possible single run of spaces shifted between p1's tail and p2's
    head in the ``<BLOCK_START>`` case.
    """
    ell = ell.replace("\t", "    ")
    ell = ell.lstrip(" ")

    if len(ell) == 0:
        return "", ""

    if "#" in ell:
        # No syntax constraints after the first '#'. The presence or absence
        # of <INSTRUCTION> is determined by the substring preceding '#'.
        ells = ell.split("#", maxsplit=1)
        p1, p2 = _separate_stimcircuit_instruction(ells[0])
        p2 = p2 + "#" + ells[1]
        return p1, p2

    # No comments past this point.
    if "}" in ell:
        # We match <BLOCK_END>; such lines cannot contain instructions.
        return "", ell
    elif "{" in ell:
        # We match <BLOCK_START> ::= <INSTRUCTION> /[ \t]*/ '{'
        ells = ell.split("{", maxsplit=1)
        len_before = len(ells[0])
        p1 = ells[0].rstrip(" ")
        len_after = len(p1)
        whitespace = " " * (len_before - len_after)
        p2 = whitespace + "{" + ells[1]
        return p1, p2
    else:
        # We match <INSTRUCTION> directly
        return ell, ""


def _replace_instruction_targets(
    inst: str, targets_map: dict[str, str]
) -> str:
    parts = inst.split(")", maxsplit=1)
    if len(parts) == 2:
        pre, post = parts
        post = _replace_instruction_targets(post, targets_map)
        return pre + ")" + post

    parts = inst.split("]", maxsplit=1)
    if len(parts) == 2:
        pre, post = parts
        post = _replace_instruction_targets(post, targets_map)
        return pre + "]" + post

    parts = inst.split(" ")
    for i in range(1, len(parts)):
        pi = parts[i]
        prefix = "" if (not pi.startswith("!")) else "!"
        pi = pi.lstrip("!")
        if pi in targets_map:
            pi = str(targets_map[pi])
        pi = prefix + pi
        parts[i] = pi
    inst = " ".join(parts)
    return inst


def _as_stim_circuit(circuit: str, qubit_labels) -> _Circuit:
    lines = []
    label_map = {str(lbl): str(i) for i, lbl in enumerate(qubit_labels)}
    for line in circuit.split("\n"):
        p1, p2 = _separate_stimcircuit_instruction(line)
        if len(p1) == 0:
            lines.append(line)
        else:
            p1_mapped = _replace_instruction_targets(p1, label_map)
            line_mapped = p1_mapped + p2
            lines.append(line_mapped)
    circuit_str = "\n".join(lines)
    c = _Circuit(circuit_str)
    return c


class STIMPhysicalCircuit(BasePhysicalCircuit):
    """Circuit backend using STIM.

    Qubit ``i`` of the underlying ``stim.Circuit`` is ``qubit_labels[i]``.
    Labels past the circuit's highest qubit index are idle qubits: they
    are kept in ``qubit_labels`` but no instruction acts on them.

    When built from a ``stim.Circuit``, the used indices are compacted
    onto the labels if there are as many labels as used indices;
    otherwise the circuit's indices are kept as they are.

    Layers are separated by ``TICK`` lines (tagged ones too), with or
    without a final ``TICK``. The layer-editing methods (``insert``,
    ``merge``, ``pad_single_qubit_idles_by_duration`` and
    ``transplant_idle_schedule``) unroll every ``REPEAT`` block of each
    circuit they read, so the result has them expanded.
    """

    _circuit: _Circuit
    """STIM circuit
    """

    _qubit_labels: list[QubitTypes]
    """list of qubit labels"""

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
    as possible keys into a [](api:DictNoiseModel).
    """

    stim_command_aliases: ClassVar[dict[str, str]] = {"CNOT": "CX"}

    def __init__(
        self,
        circuit: STIMCircuitLike,
        qubit_labels: Sequence[QubitTypes] | None = None,
        suppress_tick_warning: bool = False,
    ) -> None:
        if not is_backend_available("stim_circuit"):
            msg = "STIM backend is not available.\n"
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
            self._circuit = circuit.circuit.copy()
            self._qubit_labels = list(qubit_labels)

        elif isinstance(circuit, _Circuit):
            used_indices = _get_used_stim_indices(circuit)
            if len(used_indices) == len(qubit_labels):
                index_map = {
                    old_idx: new_idx
                    for new_idx, old_idx in enumerate(used_indices)
                }
                self._circuit = _reindex_stim_circuit(circuit, index_map)
            else:
                self._circuit = circuit.copy()
            self._qubit_labels = list(qubit_labels)

        else:  # we're a plain str
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
        return self._qubit_labels

    def copy(self) -> STIMPhysicalCircuit:
        return STIMPhysicalCircuit(self)

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
            elif any(
                [str(qidx) in entries[1:] for qidx in qubit_idxs_to_delete]
            ):
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
        layers, _ = _split_stim_layers(self._unroll_repeats())
        for lidx, (lines, _tick) in enumerate(layers):
            for line in lines:
                entries = line.split()
                if len(entries) == 0 or entries[0] not in self._stim_gates:
                    # Empty line or not a gate, skip to next line
                    continue

                # Report plain STIM qubit indices, matching the other
                # backends and this method's own declared return type.
                if post_twoq_gates:
                    if entries[0] in self._stim_twoq_gates:
                        # Handle the case where multiple 2Q gates are defined on one line
                        for i in range(1, len(entries[1:]), 2):
                            # Strip inverted-target prefix; it does not
                            # change which physical qubit is targeted.
                            stim_idx1 = int(entries[i].lstrip("!"))
                            stim_idx2 = int(entries[i + 1].lstrip("!"))
                            circuit_locations.append(
                                (
                                    lidx + 1,
                                    (
                                        stim_idx1,
                                        stim_idx2,
                                    ),
                                )
                            )
                else:
                    circuit_locations.extend(
                        [(lidx, int(q.lstrip("!"))) for q in entries[1:]]
                    )
        return circuit_locations

    def insert_inplace(self, circuit: BasePhysicalCircuit, idx: int) -> None:
        """Insert another circuit to this circuit.

        Both circuits are unrolled: ``REPEAT`` blocks in either come out
        expanded in the result. A final ``TICK`` is kept or omitted as in
        this circuit.

        Parameters
        ----------
        circuit:
            Circuit to insert

        idx:
            Layer index at which the other circuit's layers are placed,
            with Python list slicing semantics (so -1 inserts before the
            last layer).
        """
        other_circuit = STIMPhysicalCircuit(circuit)

        layers, terminated = _split_stim_layers(self._unroll_repeats())
        other_layers, _ = _split_stim_layers(other_circuit._unroll_repeats())

        layers = layers[:idx] + other_layers + layers[idx:]
        self._circuit = _Circuit(_join_stim_layers(layers, terminated))

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

        While [](api:STIMPhysicalCircuit.insert_inplace) adds new layers,
        [](api:STIMPhysicalCircuit.merge_inplace) will try to add operations to
        existing layers.

        Both circuits are unrolled: ``REPEAT`` blocks in either come out
        expanded in the result. A final ``TICK`` is kept or omitted as in
        this circuit.

        Parameters
        ----------
        circuit : BasePhysicalCircuit
            Circuit to merge

        idx : int
            Layer index to start merge
        """
        other_circuit = STIMPhysicalCircuit(circuit)

        # Build index map for the other circuit
        index_map = {}
        new_qubit_labels = list(self._qubit_labels)

        # Map shared qubit labels to their existing STIM indices
        for stim_idx, label in enumerate(self._qubit_labels):
            if label in other_circuit.qubit_labels:
                other_stim_idx = other_circuit.qubit_labels.index(label)
                index_map[other_stim_idx] = stim_idx

        # Add new qubit labels and map them to new STIM indices
        for other_stim_idx, other_label in enumerate(
            other_circuit.qubit_labels
        ):
            if other_label not in self._qubit_labels:
                new_stim_idx = len(new_qubit_labels)
                index_map[other_stim_idx] = new_stim_idx
                new_qubit_labels.append(other_label)

        # Reindex the other circuit (unrolled) to use our STIM indices
        reindexed_other_circuit = _reindex_stim_circuit(
            _Circuit(other_circuit._unroll_repeats()), index_map
        )

        layers, terminated = _split_stim_layers(self._unroll_repeats())
        other_layers, _ = _split_stim_layers(str(reindexed_other_circuit))

        # Ensure circuit is long enough for merge
        end = idx + len(other_layers)
        while len(layers) < end:
            layers.append(([], "TICK"))

        # Perform merge
        for lidx in range(idx, end):
            incoming, _tick = other_layers[lidx - idx]
            current, tick = layers[lidx]

            collision = set(
                _get_used_stim_indices(_Circuit("\n".join(current)))
            ).intersection(
                _get_used_stim_indices(_Circuit("\n".join(incoming)))
            )
            if collision:
                msg = f"Cannot merge\n{self}\nwith\n{circuit}.\n"
                msg += f"Layer {lidx} of the candidate merge has ill-posed behavior\n"
                msg += f"for target qubit(s) {collision}."
                raise ValueError(msg)

            layers[lidx] = (current + incoming, tick)

        self._circuit = _Circuit(_join_stim_layers(layers, terminated))

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
        """Pad idle qubits in every layer with a single-qubit idle.

        ``REPEAT`` blocks are unrolled: they come out expanded in the
        result. Every layer is kept; an empty one gets idles only with
        ``empty_layer_idle``. Idles are appended in qubit label order.
        See [](api:BasePhysicalCircuit.pad_single_qubit_idles_by_duration_inplace)
        for the parameters.
        """
        layers, terminated = _split_stim_layers(self._unroll_repeats())
        new_layers = []
        for lines, tick in layers:
            # Check with qubits are not idling and compute duration
            seen_qubits = set()
            layer_duration = None
            for line in lines:
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
                    seen_qubits.add(int(qubit))

            # Get idling operation (empty layers get none without one)
            new_lines = list(lines)
            if layer_duration is not None or empty_layer_idle is not None:
                layer_idle = (
                    empty_layer_idle
                    if layer_duration is None
                    else idle_names[layer_duration]
                )
                new_lines.extend(
                    f"{layer_idle} {idx}"
                    for idx in range(len(self._qubit_labels))
                    if idx not in seen_qubits
                )
            new_layers.append((new_lines, tick))

        self._circuit = _Circuit(_join_stim_layers(new_layers, terminated))

    def transplant_idle_schedule_inplace(
        self,
        reference: BasePhysicalCircuit,
        qubits: Sequence[QubitTypes],
        idle_names: Sequence[str],
    ) -> None:
        """Copy the reference's idle schedule onto this circuit's layers.

        ``REPEAT`` blocks in both circuits are unrolled: they come out
        expanded in the result. Idles are appended in ``qubits`` order.
        See [](api:BasePhysicalCircuit.transplant_idle_schedule_inplace)
        for the parameters.
        """
        # Convert the reference to this class, avoiding a redundant copy if
        # it's already the right type (reference is only ever read below).
        if not isinstance(reference, STIMPhysicalCircuit):
            reference = STIMPhysicalCircuit(reference)
        idle_set = set(idle_names)
        qubit_set = set(qubits)

        def layer_events(
            circ: "STIMPhysicalCircuit", lines: list[str]
        ) -> dict[QubitTypes, tuple[str, str]]:
            """qubit -> (kind, gate_name) for every qubit of interest with a
            recognized instruction in this layer's lines."""
            events: dict[QubitTypes, tuple[str, str]] = {}
            for line in lines:
                entries = line.split()
                if len(entries) == 0 or entries[0] not in circ._stim_gates:
                    continue
                kind = "idle" if entries[0] in idle_set else "real"
                for tok in entries[1:]:
                    try:
                        idx = int(tok)
                    except ValueError:
                        continue
                    q = circ._qubit_labels[idx]
                    if q in qubit_set:
                        events[q] = (kind, entries[0])
            return events

        # Reference sequence: for each qubit, its ordered (kind, gate_name)
        # across every layer it appears in.
        true_seqs: dict[QubitTypes, list[tuple[str, str]]] = {
            q: [] for q in qubit_set
        }
        ref_layers, _ = _split_stim_layers(reference._unroll_repeats())
        for ref_lines, _tick in ref_layers:
            for q, ref_event in layer_events(reference, ref_lines).items():
                true_seqs[q].append(ref_event)

        ptrs = {q: 0 for q in qubit_set}
        layers, terminated = _split_stim_layers(self._unroll_repeats())
        new_layers = []
        for lines, tick in layers:
            events = layer_events(self, lines)
            new_lines = list(lines)

            for q in dict.fromkeys(qubits):
                true_seq = true_seqs[q]
                ptr = ptrs[q]
                event = events.get(q)
                if event is not None and event[0] == "real":
                    if ptr >= len(true_seq) or true_seq[ptr][0] != "real":
                        raise ValueError(
                            f"Qubit {q!r}: real operation in this layer does not match "
                            f"the reference sequence at position {ptr}"
                        )
                    ptrs[q] = ptr + 1
                    continue
                if (
                    event is None
                    and ptr < len(true_seq)
                    and true_seq[ptr][0] == "idle"
                ):
                    idx = self._qubit_labels.index(q)
                    new_lines.append(f"{true_seq[ptr][1]} {idx}")
                    ptrs[q] = ptr + 1

            new_layers.append((new_lines, tick))

        self._circuit = _Circuit(_join_stim_layers(new_layers, terminated))

        for q in qubit_set:
            if ptrs[q] != len(true_seqs[q]):
                raise ValueError(
                    f"Qubit {q!r}: only matched {ptrs[q]}/{len(true_seqs[q])} reference "
                    "operations -- this circuit does not have enough layers/real "
                    "operations to receive the full reference idle schedule"
                )

    def set_qubit_labels_inplace(
        self, qubit_labels: Sequence[QubitTypes]
    ) -> None:
        """Re-label the circuit's qubits in place.

        Index ``i`` of the underlying circuit acts on ``qubit_labels[i]``.
        Labels past the circuit's highest qubit index are idle qubits.
        When a ``stim.Circuit`` is passed to the constructor, its used
        indices are compacted onto the labels if the counts are equal,
        and otherwise kept as they are (positional).

        Parameters
        ----------
        qubit_labels:
            New labels for the circuit's qubits. There must be at least
            :attr:`.circuit.num_qubits` of them.

        Raises
        ------
        ValueError
            If there are fewer labels than the circuit has qubits.
        """
        if len(qubit_labels) < self.circuit.num_qubits:
            msg = (
                f"Cannot set {len(qubit_labels)} qubit labels on a STIM "
                f"circuit with {self.circuit.num_qubits} qubits; at least "
                f"as many labels as qubits are required. (Use "
                f"delete_qubits_inplace to change the qubit count.)"
            )
            raise ValueError(msg)
        self._qubit_labels = list(qubit_labels)

    @classmethod
    def _deserialize_circuit(
        cls,
        serial_circuit: str | list | dict,
        qubit_labels: Sequence | None = None,
    ) -> _Circuit:
        """Deserialize a serialized STIM circuit back into a ``stim.Circuit``.

        Parameters
        ----------
        serial_circuit:
            The serialized form, as produced by :meth:`._serialize_circuit`.
            Must be a ``str``; the ``list``/``dict`` types in the parent
            signature are not supported by this backend.

        qubit_labels:
            Accepted for API compatibility with
            :meth:`.BasePhysicalCircuit.from_decoded_attrs` but unused —
            STIM circuits round-trip through ``str(circuit)`` directly.
        """
        assert isinstance(serial_circuit, str)
        return _Circuit(serial_circuit)

    def _serialize_circuit(self) -> str | list | dict:
        # For STIM circuit, string version is already serializable
        return str(self.circuit)

    def _unroll_repeats(self) -> str:
        """Return the circuit string with all ``REPEAT`` blocks fully expanded.

        Each ``REPEAT`` line is paired with its matching ``}`` by walking
        forward with a brace-depth counter, so nested ``REPEAT`` blocks
        and multiple sibling ``REPEAT`` blocks are handled correctly.

        Raises
        ------
        ValueError
            If a ``REPEAT`` line is followed by a non-integer repeat count.

        AssertionError
            If a ``REPEAT`` block is missing its closing ``}``.
        """
        circuit_str = str(self.circuit)
        unrolled_lines = circuit_str.split("\n")

        def find_first_repeat_start(lines):
            for i, line in enumerate(lines):
                entries = line.split()
                if entries and entries[0] == "REPEAT":
                    return i
            return None

        def find_matching_end(lines, start):
            """``lines[start]`` is a ``REPEAT`` opener; walk forward
            tracking brace depth and return the index of the matching
            ``}``. Returns ``None`` if no matching ``}`` is found."""
            depth = 1
            for i in range(start + 1, len(lines)):
                entries = lines[i].split()
                if not entries:
                    continue
                if entries[0] == "REPEAT":
                    depth += 1
                elif entries[0] == "}":
                    depth -= 1
                    if depth == 0:
                        return i
            return None

        start = find_first_repeat_start(unrolled_lines)
        while start is not None:
            end = find_matching_end(unrolled_lines, start)
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
