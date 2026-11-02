#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Steane's custom apply functions, named in `spec.py` by `module:function`.

They cover what the generator can't derive (feed-forward, decoder, parity).
Each takes its tables through `Instruction.data` and writes into a copy of
the incoming `PatchLayout`. Transcribed from the old module's apply
functions (`codepack_7_1_3_quantinuum2021.py` at ce1ee55, S:527-610,
S:715-812 and S:856-954).
"""

from __future__ import annotations

import numpy as np

from loqs.core.frame import Frame
from loqs.core.instructions.instructionstack import InstructionStack
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchlayout import PatchLayout


def flagged_feedforward(
    patch_label: str,
    patches: PatchLayout,
    measurement_outcomes: MeasurementOutcomes,
    flag_qubits: list[str],
    syndrome_index: dict[str, int],
    first_check: bool,
    stack: InstructionStack,
) -> Frame:
    """PRX Sec II.A.3 steps 1a and 1b: diff the flagged readings against
    the stored syndrome; on a change run the unflagged checks and decode."""
    patch = patches[patch_label]
    last_syndromes = patch.data.get("latest_syndrome", [0] * 6)
    flag_syndrome_diff = list(patch.data.get("flagged_syndrome_diff", [0] * 6))
    for fq in flag_qubits:
        j = syndrome_index[fq]
        flag_syndrome_diff[j] = measurement_outcomes[fq][0] ^ last_syndromes[j]

    new_patch = patch.copy()
    new_patch.data["flagged_syndrome_diff"] = flag_syndrome_diff

    if any(sd == 1 for sd in flag_syndrome_diff):
        new_stack = stack.insert_instructions(
            0,
            [
                ("Unflagged Parallel S1-S2-S3 Check", patch_label),
                ("Unflagged Parallel S4-S5-S6 Check", patch_label),
                ("QEC Decoder", patch_label),
            ],
        )
    elif first_check:
        new_stack = stack.insert_instructions(
            0,
            [
                ("Flagged Parallel S2-S3-S4 Check", patch_label),
                ("Flagged S2-S3-S4 Feed-Forward", patch_label),
            ],
        )
    else:
        new_stack = stack
        new_patch.data.pop("flagged_syndrome_diff", None)

    new_patches = patches.copy()
    new_patches[patch_label] = new_patch
    return Frame(
        {
            "stack": new_stack,
            "patches": new_patches,
            "flag_syndrome_diff": flag_syndrome_diff,
        }
    )


def qec_decoder(
    patch_label: str,
    patches: PatchLayout,
    X_outcomes: MeasurementOutcomes,
    Z_outcomes: MeasurementOutcomes,
    X_qubits: list[str],
    Z_qubits: list[str],
    data_lut: list[list[int]],
    hook_lut: list[list[list[int]]],
) -> Frame:
    """PRX Sec II.A.3 step 2: Table 1 on the unflagged diff, Table 2 on
    the (flagged diff, unflagged diff) pair, per check type."""
    patch = patches[patch_label]
    last_syndromes = patch.data.get("latest_syndrome", [0] * 6)
    logical_pauli_frame = list(patch.data.get("logical_pauli_frame", [0, 0]))
    flagged_syndrome_diff = patch.data.get("flagged_syndrome_diff", [None] * 6)

    unflagged_syndrome = [X_outcomes[q][0] for q in X_qubits]
    unflagged_syndrome.extend([Z_outcomes[q][0] for q in Z_qubits])
    unflagged_syndrome_diff = [
        int(i) for i in np.bitwise_xor(last_syndromes, unflagged_syndrome)
    ]

    for pf_idx, sl in ((0, slice(0, 3)), (1, slice(3, 6))):
        sd = unflagged_syndrome_diff[sl]
        fsd = list(flagged_syndrome_diff[sl])
        if sd in data_lut:
            logical_pauli_frame[pf_idx] ^= 1
        if [fsd, sd] in hook_lut:
            logical_pauli_frame[pf_idx] ^= 1

    new_patch = patch.copy()
    new_patch.data["latest_syndrome"] = unflagged_syndrome
    new_patch.data["logical_pauli_frame"] = logical_pauli_frame
    new_patch.data.pop("flagged_syndrome_diff", None)
    new_patches = patches.copy()
    new_patches[patch_label] = new_patch
    return Frame(
        {
            "unflagged_syndrome": unflagged_syndrome,
            "unflagged_syndrome_diff": unflagged_syndrome_diff,
            "flagged_syndrome_diff": flagged_syndrome_diff,
            "new_logical_pauli_frame": logical_pauli_frame,
            "patches": new_patches,
        }
    )


def logical_parity(
    patch_label: str,
    patches: PatchLayout,
    data_qubits: list[str],
    measurement_basis: str,
    measurement_outcomes: MeasurementOutcomes,
    checks: list[list[int]],
    readout: list[int],
    data_lut: list[list[int]],
) -> Frame:
    """PRX Sec II.A.4: raw parity of the logical's support, the checks of
    the measured type rebuilt from the data, Table 1 on their diff against
    the stored syndrome, then the frame."""
    logical_pauli_frame = list(
        patches[patch_label].data.get("logical_pauli_frame", [0, 0])
    )
    last_syndrome = patches[patch_label].data.get("latest_syndrome", [0] * 6)
    raw = [measurement_outcomes[data_qubits[i]][0] for i in readout]
    uncorrected_outcome = sum(raw) % 2
    classical_syndrome = [
        sum(measurement_outcomes[data_qubits[i]][0] for i in c) % 2
        for c in checks
    ]
    if measurement_basis == "X":
        pf_idx, last = 0, last_syndrome[:3]
    else:
        pf_idx, last = 1, last_syndrome[3:]
    diff = [c ^ s for c, s in zip(classical_syndrome, last)]
    if diff in data_lut:
        logical_pauli_frame[pf_idx] ^= 1
    return Frame(
        {
            "patch_label": patch_label,
            "logical_measurement": uncorrected_outcome
            ^ logical_pauli_frame[pf_idx],
            "uncorrected_measurement": uncorrected_outcome,
            "classical_syndrome": classical_syndrome,
            "classical_syndrome_diff": diff,
            "final_logical_pauli_frame": logical_pauli_frame,
        }
    )
