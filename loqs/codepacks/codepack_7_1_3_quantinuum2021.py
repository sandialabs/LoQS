#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""A LoQS QEC codepack for the [[7,1,3]] color code.

This implementation is based on the 2021 implementation from
Quantinuum in :cite:`ryananderson_realizing_2021`.

We require three auxiliary qubits for stabilizer checks.
Thus, we will have 10 qubits total: 7 data and 3 auxiliary.

.. bibliography::
    :filter: docname in docnames
"""

from collections.abc import Sequence
import copy
import itertools
from typing import Mapping
import numpy as np

from loqs.backends.circuit.basecircuit import BasePhysicalCircuit
from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.backends.model.basemodel import (
    BaseNoiseModel,
    GateRep,
    InstrumentRep,
)
from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.backends.model.pygstimodel import PyGSTiNoiseModel
from loqs.backends.reps import RepTuple
from loqs.core import Instruction, QECCode
from loqs.core.frame import Frame
from loqs.core.instructions import builders
from loqs.core.instructions.instruction import KwargDict
from loqs.core.instructions.instructionstack import InstructionStack
from loqs.core.recordables import QECCodePatch
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchdict import PatchDict
import loqs.tools.pygstitools as pt


def create_qec_code(
    ft_state_prep_max_repeats: int = 100,
    include_idles: bool = False,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
):
    """Create a QECCode implementing the [[7,1,3]] code.

    Parameters
    ----------
    ft_state_prep_max_repeats:
        The number of max repeats to include in the repeat-until-success
        fault-tolerant state prep instruction.

    include_idles:
        Whether to include (``True``) or not (``False``, default) idle gates
        in physical circuits.

    gate_durations:
        Mapping from gate names to durations. Defaults to ``None``, which uses
        dummy values 1, 2, 3 for 1Q gates, 2Q gates, and mid-circuit
        measurements, respectively.
        See ``durations`` from
        :meth:`.BasePhysicalCircuit.pad_single_qubit_idles_by_duration_inplace`
        for more details.

    idle_gates:
        Mapping from gate duration to idle gate names. Defaults to ``None``,
        which maps the dummy values from ``gate_durations`` to ``"Gi1Q"``,
        ``"Gi2Q"``, and ``"GiMCM"``, respectively.
        See ``idle_names`` from
        :meth:`.BasePhysicalCircuit.pad_single_qubit_idles_by_duration_inplace`
        for more details.

    circuit_backend:
        The circuit backend to use when generating physical circuits.

    Returns
    -------
        A :class:`.QECCode` implementing the [[5,1,3]] code.
    """

    # Template qubits for defining one patch
    qubits = ["A0", "A1", "A2"] + [f"D{i}" for i in range(7)]
    data_qubits = qubits[2:]

    instructions: dict[str, Instruction] = {}

    # For padding by idles with duration
    # TODO
    if gate_durations is None:
        gate_durations = {
            k: 1
            for k in [
                "Gi",
                "Gi1Q",
                "Gxpi",
                "Gypi",
                "Gzpi",
                "Gzpi2",
                "Gzmpi2",
                "Gh",
            ]
        }
        gate_durations["Gcnot"] = 2
        gate_durations["Gi2Q"] = 2
        gate_durations["Iz"] = 3
        gate_durations["GiMCM"] = 3
    if idle_gates is None:
        idle_gates = {1: "Gi1Q", 2: "Gi2Q", 3: "GiMCM"}

    ## PREP
    # Non-FT |0> state prep
    # Encoding circuit box of Fig 10 of 10.1103/PhysRevX.11.041058
    # without auxiliary qubit check
    nonft_state_prep_circ = circuit_backend(
        [
            [
                ("Gh", "D0"),
                ("Gh", "D4"),
                ("Gh", "D6"),
            ],
            [
                ("Gcnot", "D0", "D1"),
                ("Gcnot", "D4", "D5"),
                ("Gcnot", "D6", "D3"),
            ],
            [
                ("Gcnot", "D0", "D3"),
                ("Gcnot", "D4", "D2"),
                ("Gcnot", "D6", "D5"),
            ],
            [("Gcnot", "D4", "D1"), ("Gcnot", "D3", "D2")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        nonft_state_prep_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Non-FT Zero Prep"] = (
        builders.build_physical_circuit_instruction(
            nonft_state_prep_circ,
            name="Non-FT zero state prep",
        )
    )

    ### Try-until-success FT |0> state prep
    # Qubit reset in case of failure
    reset = builders.build_physical_circuit_instruction(
        circuit_backend(
            [[("Iz", q) for q in qubits[3:]]], qubit_labels=qubits
        ),
        name="Reset to all 0 state",
    )

    # Auxiliary qubit check from Encoding circuit box of Fig 10 of 10.1103/PhysRevX.11.041058
    ft_state_prep_checks_circ = circuit_backend(
        [
            [("Gcnot", "D1", "A0")],
            [("Gcnot", "D3", "A0")],
            [("Gcnot", "D5", "A0")],
            [("Iz", "A0")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        ft_state_prep_checks_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    ft_state_prep_circ = nonft_state_prep_circ.append(
        ft_state_prep_checks_circ
    )
    instructions["Non-FT Zero Prep + Checks"] = (
        builders.build_physical_circuit_instruction(
            ft_state_prep_circ,
            name="Non-FT Zero Prep + Checks",
        )
    )

    # On success, we expect 0 outcome on the flag qubit from the check circuit
    rus_success_expected = MeasurementOutcomes({"A0": [0]})

    instructions["FT Zero Prep"] = (
        builders.build_repeat_until_success_instruction(
            [reset, instructions["Non-FT Zero Prep + Checks"]],
            rus_key="FT Zero Prep",
            test_frame_key="measurement_outcomes",
            expected=rus_success_expected,
            max_repeats=ft_state_prep_max_repeats,
            name="Repeat-until-success FT Zero Prep",
        )
    )

    ## GATES
    # Logical Clifford gates (transversal)
    # (or rather, a common subset of them)
    # Table from Fig 1 of 10.1103/PhysRevX.11.041058
    clifford_gates = {
        "X": "Gxpi",
        "Y": "Gypi",
        "Z": "Gzpi",
        "H": "Gh",
        "S": "Gzmpi2",  # Logical S is really all Sdagger...
        "Sdag": "Gzpi2",  # ...and vice versa
        "I": "Gi",
    }
    # Paulis only act on last three data qubits (bottom row)
    edge_qubits = qubits[-3:]
    for n, gn in clifford_gates.items():
        # Bottom row for paulis, all data for others
        active_qubits = edge_qubits if n in "XYZ" else qubits[3:]
        logical_circ = circuit_backend(
            [[(gn, q) for q in active_qubits]],
            qubit_labels=qubits,
        )
        if include_idles:
            logical_circ.pad_single_qubit_idles_by_duration_inplace(
                idle_gates, gate_durations
            )
        instructions[n] = builders.build_physical_circuit_instruction(
            logical_circ,
            pauli_frame_update=n,
            name=f"Logical {n}",
        )

    ## QEC
    # This is "First flagged parallel circuit" from Figure 10 of 10.1103/PhysRevX.11.041058
    flagged_QEC_part1_circ = circuit_backend(
        [
            [("Gh", "A0")],
            [
                ("Gcnot", "A0", "D3"),
                ("Gcnot", "D2", "A2"),
                ("Gcnot", "D5", "A1"),
            ],
            [("Gcnot", "A0", "A1")],
            [
                ("Gcnot", "A0", "D0"),
                ("Gcnot", "D3", "A2"),
                ("Gcnot", "D4", "A1"),
            ],
            [
                ("Gcnot", "A0", "D1"),
                ("Gcnot", "D6", "A2"),
                ("Gcnot", "D2", "A1"),
            ],
            [("Gcnot", "A0", "A2")],
            [
                ("Gcnot", "A0", "D2"),
                ("Gcnot", "D5", "A2"),
                ("Gcnot", "D1", "A1"),
            ],
            [("Gh", "A0")],
            [("Iz", "A0"), ("Iz", "A1"), ("Iz", "A2")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        flagged_QEC_part1_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Flagged Parallel S1-S5-S6 Check"] = (
        builders.build_physical_circuit_instruction(
            flagged_QEC_part1_circ,
            name="Flagged S1-S5-S6 Check",
        )
    )

    # This is "Second flagged parallel circuit" from Figure 10 of 10.1103/PhysRevX.11.041058
    flagged_QEC_part2_circ = circuit_backend(
        [
            [("Gh", "A1"), ("Gh", "A2")],
            [
                ("Gcnot", "D3", "A0"),
                ("Gcnot", "A2", "D2"),
                ("Gcnot", "A1", "D5"),
            ],
            [("Gcnot", "A1", "A0")],
            [
                ("Gcnot", "D0", "A0"),
                ("Gcnot", "A2", "D3"),
                ("Gcnot", "A1", "D4"),
            ],
            [
                ("Gcnot", "D1", "A0"),
                ("Gcnot", "A2", "D6"),
                ("Gcnot", "A1", "D2"),
            ],
            [("Gcnot", "A2", "A0")],
            [
                ("Gcnot", "D2", "A0"),
                ("Gcnot", "A2", "D5"),
                ("Gcnot", "A1", "D1"),
            ],
            [("Gh", "A1"), ("Gh", "A2")],
            [("Iz", "A0"), ("Iz", "A1"), ("Iz", "A2")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        flagged_QEC_part2_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Flagged Parallel S2-S3-S4 Check"] = (
        builders.build_physical_circuit_instruction(
            flagged_QEC_part2_circ,
            name="Flagged S2-S3-S4 Check",
        )
    )

    # These are the 6 unflagged stabilizer checks from Figure 10 of 10.1103/PhysRevX.11.041058
    # Scheduling less clear, we provide individual and merged Z/X circuits
    # We do this by defining a template circuit and mapping to the various plaquettes
    Z_temp_circ = circuit_backend(
        [[("Gcnot", q, "aux")] for q in "abcd"] + [[("Iz", "aux")]]
    )
    X_temp_circ = circuit_backend(
        [[("Gh", "aux")]]
        + [[("Gcnot", "aux", q)] for q in "abcd"]
        + [[("Gh", "aux")], [("Iz", "aux")]]
    )
    mappings = {
        0: ["D0", "D1", "D2", "D3"],
        1: ["D1", "D2", "D4", "D5"],
        2: ["D2", "D3", "D5", "D6"],
    }
    parallel_X_circ = circuit_backend([], qubit_labels=qubits)
    parallel_Z_circ = circuit_backend([], qubit_labels=qubits)
    for i, m in mappings.items():
        # Map circuits
        mapping = {q1: q2 for q1, q2 in zip("abcd", m)}
        mapping["aux"] = f"A{i}"
        X_circ = X_temp_circ.map_qubit_labels(mapping)
        Z_circ = Z_temp_circ.map_qubit_labels(mapping)

        # Standalone checks
        if include_idles:
            X_circ_padded = X_circ.pad_single_qubit_idles_by_duration(
                idle_gates, gate_durations
            )
            Z_circ_padded = Z_circ.pad_single_qubit_idles_by_duration(
                idle_gates, gate_durations
            )
        else:
            X_circ_padded = X_circ
            Z_circ_padded = Z_circ
        instructions[f"Unflagged S{i + 1} Check"] = (
            builders.build_physical_circuit_instruction(
                X_circ_padded, name=f"Unflagged S{i+1} Check"
            )
        )
        instructions[f"Unflagged S{i + 4} Check"] = (
            builders.build_physical_circuit_instruction(
                Z_circ_padded, name=f"Unflagged S{i+1} Check"
            )
        )

        # Merge (unpadded) into parallel check
        parallel_X_circ.merge_inplace(X_circ, 0)
        parallel_Z_circ.merge_inplace(Z_circ, 0)
    # Parallel checks
    if include_idles:
        parallel_X_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
        parallel_Z_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Unflagged Parallel S1-S2-S3 Check"] = (
        builders.build_physical_circuit_instruction(
            parallel_X_circ, name="Unflagged Parallel S1-S2-S3 Check"
        )
    )
    instructions["Unflagged Parallel S4-S5-S6 Check"] = (
        builders.build_physical_circuit_instruction(
            parallel_Z_circ, name="Unflagged S4-S5-S6 Check"
        )
    )

    _create_adaptive_qec_instructions(instructions, qubits)

    # Convenience start function for QEC
    instructions["Adaptive QEC"] = builders.build_composite_instruction(
        [
            instructions["Flagged Parallel S1-S5-S6 Check"],
            instructions["Flagged S1-S5-S6 Feed-Forward"],
        ],
        name="Start of Adaptive QEC",
    )

    ## MEASURE
    # Full data qubit measurements
    raw_Z_meas_circ = circuit_backend(
        [[("Iz", q) for q in qubits[3:]]], qubit_labels=qubits
    )
    raw_X_meas_circ = circuit_backend(
        [
            [("Gh", q) for q in qubits[3:]],
            [("Iz", q) for q in qubits[3:]],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        raw_Z_meas_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
        raw_X_meas_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Raw Z Data Measure"] = (
        builders.build_physical_circuit_instruction(
            raw_Z_meas_circ,
            name="Raw logical Z-basis measurement",
        )
    )
    instructions["Raw X Data Measure"] = (
        builders.build_physical_circuit_instruction(
            raw_X_meas_circ,
            name="Raw logical X-basis measurement",
        )
    )

    # We can also compute the logical measurement based on the raw logical output
    # This is described in Section II.A.4 of 10.1103/PhysRevX.11.041058, summarized here:
    # 1. Compute the non-FT logical outcome as the product of outcomes of bottom edge of plaquette
    # 2. Classically compute the syndrome (we will only get the syndrome for the opposite of the measure basis)
    # 3. Decode the syndrome and determine whether to flip the raw logical output
    #    (multiple ways to do this, we'll just use the syndrome to update the Pauli frame and infer again.
    #    Could also look at weight of correction on edge qubits, odd weight = flip, even weight = don't flip.)
    def logical_meas_apply_fn(
        patch_label: str,
        patches: PatchDict,
        data_qubits: list[str],
        measurement_basis: str,
        measurement_outcomes: MeasurementOutcomes,
    ) -> Frame:
        # Get the logical pauli frame
        logical_pauli_frame = patches[patch_label].data.get(
            "logical_pauli_frame", [0, 0]
        )

        # Compute uncorrected output
        raw_bitstring = [measurement_outcomes[q][0] for q in data_qubits]
        uncorrected_outcome = sum(raw_bitstring) % 2

        plaq_idxs = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
        classical_syndrome = []
        for plaq in plaq_idxs:
            plaq_outcomes = [
                measurement_outcomes[data_qubits[q]][0] for q in plaq
            ]
            parity = sum(plaq_outcomes) % 2
            classical_syndrome.append(parity)

        # Final data qubit decoding. This is Table 1 of 10.1103/PhysRevX.11.041058, as implemented
        # by the algorithm in Fig 21
        def data_decode(sd, pf_idx):
            if sd in [[0, 1, 0], [0, 1, 1], [0, 0, 1]]:
                logical_pauli_frame[pf_idx] ^= 1

        # We are correcting the opposite basis as our measurement, because
        # that is the thing that does not commute/we are sensitive to
        pf_idx = 1 if measurement_basis == "X" else 0
        data_decode(classical_syndrome, pf_idx)

        # Flip if needed based on logical pauli frame
        logical_outcome = uncorrected_outcome ^ logical_pauli_frame[pf_idx]
        return Frame(
            {
                "patch_label": patch_label,
                "logical_measurement": logical_outcome,
                "uncorrected_measurement": uncorrected_outcome,
                "classical_syndrome": classical_syndrome,
                "final_logical_pauli_frame": logical_pauli_frame,
            }
        )

    def logical_meas_map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        data_qubits: list[str],
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["data_qubits"] = [qubit_mapping[q] for q in data_qubits]
        return new_kwargs

    Z_logical_meas = Instruction(
        logical_meas_apply_fn,
        data={"data_qubits": qubits[3:], "measurement_basis": "Z"},
        map_qubits_fn=logical_meas_map_qubits_fn,
        name="FT Z logical parity calculation",
    )

    instructions["FT Logical Z Measure"] = (
        builders.build_composite_instruction(
            [instructions["Raw Z Data Measure"], Z_logical_meas],
            name="FT logical Z measurement",
        )
    )

    X_logical_meas = Instruction(
        logical_meas_apply_fn,
        data={"data_qubits": qubits[3:], "measurement_basis": "X"},
        map_qubits_fn=logical_meas_map_qubits_fn,
        name="FT X logical parity calculation",
    )

    instructions["FT Logical X Measure"] = (
        builders.build_composite_instruction(
            [
                instructions["H"],
                instructions["Raw Z Data Measure"],
                X_logical_meas,
            ],
            name="FT logical X measurement",
        )
    )

    code = QECCode(
        instructions,
        qubits,
        data_qubits,
        "Quantinuum 2021 [[7,1,3]] color code",
    )
    return code


def _create_adaptive_qec_instructions(instructions, qubits):
    # Now that we have all the circuits, we can make our feed-forward QEC instructions
    # Overall, this will encapsulate the dashed QEC cycle box in Fig 10 of 10.1103/PhysRevX.11.041058
    # NOTE: Quantinuum's version of QEC does not have a per-qubit Pauli frame, but rather
    # a "logical" Pauli frame just for the patch. We will store this "logical" Pauli frame
    # in the extra `data` dict in the QECCodePatch and use it rather than the default Pauli frame object
    # TODO: We will almost certainly want the qubit-specific version too.
    # Need to test if the automated workflow that we used for 5Q code will work here too
    # Quantinuum's procedure is detailed in Section II.A.3 of 10.1103/PhysRevX.11.041058, and is summarized here
    # (corresponding to the two apply functions and their if/else branches):
    # 1a. Measure parallel flagged S1-S5-S6.
    #   IF: No syndrome diff to last round, proceed to 1b.
    #   ELSE: Proceed to 2.
    # 1b. Measure parallel flagged S2-S3-S4.
    #   IF: No syndrome diff to last round, no error. Terminate.
    #   ELSE: proceed to 2.
    # 2. Measure all unflagged syndromes. Save these as "ground truth" for next round.
    #   a. Computed unflagged syndrome diffs to last round.
    #      Send these to Table 1 LUT to get the data qubit error correction.
    #      Update the "logical" Pauli frame accordingly.
    #   b. Send all 6 sets of syndrome diffs (flagged to last, and unflagged to last)
    #      to Table 2 LUT to get the hook error correction.
    #      Update the "logical" Pauli frame accordingly.

    # The differences between steps 1 and 2 are minor enough that we can handle
    # it with a single function that switches behavior slightly on `first_check`
    def QEC_flagged_feedforward_apply_fn(
        # For patch to get our logical Pauli frame and last syndromes
        patch_label: str,
        patches: PatchDict,
        # To check flag outcomes
        measurement_outcomes: MeasurementOutcomes,
        # For flag/aux qubit labels in the measurement outcome
        flag_qubits: list[str],
        # Whether this is the first check or not
        first_check: bool,
        # For feed-forward processing
        stack: InstructionStack,
    ) -> Frame:
        # Get last syndromes (or default to trivial)
        patch = patches[patch_label]
        # S_previous in paper
        last_syndromes = patch.data.get("latest_syndrome", [0] * 6)
        # \Delta S^f in paper. Here because we may have populated half of it in first step
        flag_syndrome_diff = patch.data.get("flagged_syndrome_diff", [0] * 6)

        # Compute flagged syndrome diffs for these three checks
        # S^f in paper
        # Specifically S^f_{1,5,6} if first_check else S^f_{2,3,4}
        flag_syndromes = [measurement_outcomes[fq][0] for fq in flag_qubits]

        syndrome_idxs = [0, 4, 5] if first_check else [1, 2, 3]
        for i, j in enumerate(syndrome_idxs):
            flag_syndrome_diff[j] = flag_syndromes[i] ^ last_syndromes[j]

        # Save in patch information for if needed in decoding later
        new_patch = QECCodePatch(patch.code, patch.qubits, patch.pauli_frame)
        new_patch.data = copy.deepcopy(patch.data)
        new_patch.data["flagged_syndrome_diff"] = flag_syndrome_diff

        if any([sd == 1 for sd in flag_syndrome_diff]):
            # Non-trivial syndrome, go to unflagged
            next_instructions = [
                ("Unflagged Parallel S1-S2-S3 Check", patch_label),
                ("Unflagged Parallel S4-S5-S6 Check", patch_label),
                ("QEC Decoder", patch_label),  # FORWARD REFERENCE
            ]
            new_stack = stack.insert_instructions(0, next_instructions)
        elif first_check:
            # This is our first check with no errors, proceed to part 2
            next_instructions = [
                ("Flagged Parallel S2-S3-S4 Check", patch_label),
                (
                    "Flagged S2-S3-S4 Feed-Forward",
                    patch_label,
                ),  # FORWARD REFERENCE
            ]
            new_stack = stack.insert_instructions(0, next_instructions)
        else:
            # this is our second check with no errors, we proceed with no correction
            new_stack = stack

            # Delete flag syndrome
            del new_patch.data["flagged_syndrome_diff"]

        patches[patch_label] = new_patch

        return Frame(
            {
                "stack": new_stack,
                "patches": patches,
                "flag_syndrome_diff": flag_syndrome_diff,
            }
        )

    def QEC_flagged_feedforward_map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        flag_qubits: list[str | int],
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["flag_qubits"] = [qubit_mapping[q] for q in flag_qubits]
        return new_kwargs

    instructions["Flagged S1-S5-S6 Feed-Forward"] = Instruction(
        QEC_flagged_feedforward_apply_fn,
        data={"flag_qubits": qubits[:3], "first_check": True},
        map_qubits_fn=QEC_flagged_feedforward_map_qubits_fn,
        name="Flagged S1-S5-S6 Feed-Forward",
    )

    instructions["Flagged S2-S3-S4 Feed-Forward"] = Instruction(
        QEC_flagged_feedforward_apply_fn,
        data={"flag_qubits": qubits[:3], "first_check": False},
        map_qubits_fn=QEC_flagged_feedforward_map_qubits_fn,
        name="Flagged S2-S3-S4 Feed-Forward",
    )

    # Unflagged decoder
    def QEC_decoder_apply_fn(
        # For patch to get our logical Pauli frame and last syndromes
        patch_label: str,
        patches: PatchDict,
        # To check X/Z outcomes
        X_outcomes: MeasurementOutcomes,
        Z_outcomes: MeasurementOutcomes,
        # For flag/aux qubit labels in the measurement outcome
        X_qubits: list[str],
        Z_qubits: list[str],
    ) -> Frame:
        # Get last syndromes (or default to trivial)
        patch = patches[patch_label]
        # S_previous in paper
        last_syndromes = patch.data.get("latest_syndrome", [0] * 6)
        # Whether to apply logical X or Z corrections, i.e. Correction col in Table I/II
        logical_pauli_frame = patch.data.get("logical_pauli_frame", [0, 0])
        # \Delta S^f in paper
        flagged_syndrome_diff = patch.data.get(
            "flagged_syndrome_diff", [None] * 6
        )

        # Get syndrome bits
        # This will be S in paper
        unflagged_syndrome = [X_outcomes[fq][0] for fq in X_qubits]
        unflagged_syndrome.extend([Z_outcomes[fq][0] for fq in Z_qubits])

        # Compute syndrome diffs
        # This is \Delta S in paper
        unflagged_syndrome_diff = [
            int(i) for i in np.bitwise_xor(last_syndromes, unflagged_syndrome)
        ]

        # Data qubit decoding. This is Table 1 of 10.1103/PhysRevX.11.041058, as implemented
        # by the algorithm in Fig 21
        def data_decode(sd, pf_idx):
            if sd in [[0, 1, 0], [0, 1, 1], [0, 0, 1]]:
                logical_pauli_frame[pf_idx] ^= 1

        data_decode(unflagged_syndrome_diff[:3], 0)
        data_decode(unflagged_syndrome_diff[3:], 1)

        # Hook error decoding. This is Table 2 of 10.1103/PhysRevX.11.041058, as implemented
        # by a modified version of the algorithm in Fig 22
        def hook_decode(sd, fsd, pf_idx):
            if (fsd, sd) in [
                ([1, 0, 0], [0, 1, 0]),
                ([1, 0, 0], [0, 0, 1]),
                ([0, 1, 1], [0, 0, 1]),
            ]:
                logical_pauli_frame[pf_idx] ^= 1

        hook_decode(unflagged_syndrome_diff[:3], flagged_syndrome_diff[:3], 0)
        hook_decode(unflagged_syndrome_diff[3:], flagged_syndrome_diff[3:], 1)

        # Update the patch data
        new_patch = QECCodePatch(patch.code, patch.qubits, patch.pauli_frame)
        new_patch.data = copy.deepcopy(patch.data)
        new_patch.data["last_syndrome"] = (
            unflagged_syndrome  # S becomes the new S_previous
        )
        new_patch.data["logical_pauli_frame"] = logical_pauli_frame
        del new_patch.data["flagged_syndrome_diff"]

        patches[patch_label] = new_patch

        return Frame(
            {
                "unflagged_syndrome": unflagged_syndrome,
                "unflagged_syndrome_diff": unflagged_syndrome_diff,
                "flagged_syndrome_diff": flagged_syndrome_diff,
                "new_logical_pauli_frame": logical_pauli_frame,
                "patches": patches,
            }
        )

    def QEC_decoder_map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        X_qubits: list[str | int],
        Z_qubits: list[str | int],
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["X_qubits"] = [qubit_mapping[q] for q in X_qubits]
        new_kwargs["Z_qubits"] = [qubit_mapping[q] for q in Z_qubits]
        return new_kwargs

    instructions["QEC Decoder"] = Instruction(
        QEC_decoder_apply_fn,
        data={"X_qubits": qubits[:3], "Z_qubits": qubits[:3]},
        param_priorities={
            "X_outcomes": ["history[-2]"],
            "Z_outcomes": ["history[-1]"],
        },
        param_aliases={
            "X_outcomes": "measurement_outcomes",
            "Z_outcomes": "measurement_outcomes",
        },
        map_qubits_fn=QEC_decoder_map_qubits_fn,
        name="QEC Decoder",
    )


def create_ideal_model(  # noqa: C901
    qubits: Sequence[str],
    model_backend: type[BaseNoiseModel] = PyGSTiNoiseModel,
    gaterep: GateRep = GateRep.QSIM_SUPEROPERATOR,
    instrep: InstrumentRep = InstrumentRep.ZBASIS_PROJECTION,
):
    """Create an ideal (i.e. noiseless) model for the [[7,1,3]] code.

    This model will contain all the instructions needed to run the
    physical circuits in the :class:`QECCode` returned by :meth:`create_qec_code()`.


    Parameters
    ----------
    qubits:
        List of qubit labels to use. It should be have 10 entries,
        and the first three qubits should be the auxiliary qubits.

    model_backend:
        The model backend to use when generating operations.
        Currently, only :class:`PyGSTiNoiseModel` is allowed.

    Returns
    -------
        A noiseless model for the `QECCode` returned by
        :meth:`create_qec_code`
    """
    # assert len(qubits) == 10, "Must provide exactly 10 qubit labels"
    # model_qubits = [f"Q{i}" for i in range(10)]
    model_qubits = [f"Q{i}" for i in range(len(qubits))]

    gate_names = [
        "Gxpi",
        "Gypi",
        "Gzpi",
        "Gzpi2",
        "Gzmpi2",
        "Gh",
        "Gcnot",
        "Gi",
        "Gi1Q",
        "Gi2Q",
        "GiMCM",
    ]

    nonstd_unitaries = {
        "Gi1Q": np.eye(2),
        "Gi2Q": np.eye(2),
        "GiMCM": np.eye(2),
    }

    if model_backend == PyGSTiNoiseModel:
        try:
            import pygsti
        except ImportError:
            raise ImportError(
                "pyGSTi not found, cannot construct pyGSTi noise model"
            )

        # TODO: Instrument not specified here
        pspec = pygsti.processors.QubitProcessorSpec(
            len(model_qubits),
            gate_names=gate_names,
            qubit_labels=model_qubits,
            nonstd_gate_unitaries=nonstd_unitaries,
            availability={k: "all-permutations" for k in gate_names},
        )

        ideal_model_pygsti = pygsti.models.create_crosstalk_free_model(pspec)

        model = PyGSTiNoiseModel(ideal_model_pygsti, qubits)
    elif model_backend == DictNoiseModel:
        gate_dict = {}
        if gaterep == GateRep.STIM_CIRCUIT_STR:
            name_to_stim_ops = {
                "Gxpi": ["X"],
                "Gypi": ["Y"],
                "Gzpi": ["Z"],
                "Gzpi2": ["SQRT_Z"],
                "Gzmpi2": ["SQRT_Z_DAG"],
                "Gh": ["H"],
                "Gcnot": ["CX"],
                "Gi": ["I"],
                "Gi1Q": ["I"],
                "Gi2Q": ["I"],
                "GiMCM": ["I"],
            }

            for gate in gate_names:
                num_qubits = 2 if gate in ["Gcnot", "Gcphase"] else 1

                # For stim strings, all the representations are "local"
                stim_str = ""
                for stim_op in name_to_stim_ops[gate]:
                    stim_str += stim_op
                    for i in range(num_qubits):
                        stim_str += f" {i}"
                    stim_str += "\n"

                qubit_perms = itertools.permutations(qubits, r=num_qubits)
                for qs in qubit_perms:
                    gate_dict[(gate, qs)] = stim_str
        else:
            # Currently we use pyGSTi to look up definitions for dense reps
            # TODO: Remove if needed
            try:
                import pygsti
            except ImportError:
                raise ImportError(
                    "pyGSTi not found, cannot construct dict noise model"
                )

            std_unitaries = (
                pygsti.tools.internalgates.standard_gatename_unitaries()
            )

            for gate in gate_names:
                U = std_unitaries.get(gate, None)
                if U is None:
                    U = nonstd_unitaries[gate]

                num_qubits = int(np.log2(U.shape[0]))
                qubit_perms = itertools.permutations(qubits, r=num_qubits)
                for qs in qubit_perms:
                    if gaterep == GateRep.UNITARY:
                        gate_dict[(gate, qs)] = RepTuple(
                            U, qs, GateRep.UNITARY
                        )
                    elif gaterep == GateRep.PTM:
                        gate_dict[(gate, qs)] = (
                            pygsti.tools.unitary_to_pauligate(U)
                        )
                    elif gaterep == GateRep.QSIM_SUPEROPERATOR:
                        gate_dict[(gate, qs)] = pt.unitary_to_qsim_ptm(U)
                    else:
                        raise NotImplementedError(
                            "Conversion to this rep is not implemented yet."
                        )

        # Setting the value as (0, True) here means it will reset to 0 state
        # and it will record the outcomes
        inst_dict = {("Iz", (q,)): (0, True) for q in qubits}

        return DictNoiseModel(
            (gate_dict, inst_dict), gatereps=[gaterep], instreps=[instrep]
        )

    elif issubclass(model_backend, BaseNoiseModel):
        raise NotImplementedError(
            "Cannot generate ideal model for this backend"
        )
    else:
        raise ValueError("Must pass a noise model class")

    assert gaterep in model.output_gate_reps
    assert instrep in model.output_instrument_reps

    return model
