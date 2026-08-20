#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Generate `SERIALIZATION_VERSION` 2 serialization fixtures for the
current `GateRep`/`InstrumentRep` class hierarchy.

The v2 counterpart of `generate_reps_fixtures.py` -- see that file's
module docstring for the full rationale (in short: freezing real,
current-code output as a byte-for-byte regression oracle rather than
relying only on hand-constructed dicts). Unlike that script, this one
uses the current API directly and can actually be run against a normal
checkout; it exists so `reps_v2.json`/`reps_v2.h5` are ready in advance
as the reference bytes whenever the next serialization version bump
needs its own legacy-decode regression test, the same role
`reps_v1.json`/`reps_v1.h5` played for this one.

Run with: `python tests/backends/fixtures/generate_reps_fixtures_v2.py`

Only re-run this if you intend to deliberately change what these
fixtures represent -- otherwise these files should remain untouched so
they keep representing a known, previously-committed serialized format.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from loqs.backends.reps import (
    KrausGateRep,
    OutcomeOperationDictInstrumentRep,
    PTMGateRep,
    ProbabilisticStimGateRep,
    QSimSuperopGateRep,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.internal.serializable import Serializable

FIXTURES_DIR = Path(__file__).parent

QUBITS_1Q = ("Q0",)

# A real (non-trivial) unitary: pi/2 rotation about X.
_THETA = np.pi / 2
_UNITARY_1Q = np.array(
    [
        [np.cos(_THETA / 2), -1j * np.sin(_THETA / 2)],
        [-1j * np.sin(_THETA / 2), np.cos(_THETA / 2)],
    ]
)

# A real 4x4 Pauli-transfer matrix (identity channel, PTM basis).
_PTM_1Q = np.eye(4)

# A real 4x4 "QuantumSim-basis" superoperator (identity channel).
_QSIM_SUPEROP_1Q = np.eye(4)

# TP-preserving amplitude-damping Kraus operators (gamma=0.1).
_GAMMA = 0.1
_K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - _GAMMA)]])
_K1 = np.array([[0.0, np.sqrt(_GAMMA)], [0.0, 0.0]])


def build_flat_rep_fixtures() -> dict[str, object]:
    """One instance per concrete `GateRep`/`InstrumentRep` class (10
    total), including the two `InstrumentRep`s whose payload contains
    nested gate reps (`ZBasisPrePostInstrumentRep`,
    `OutcomeOperationDictInstrumentRep`) -- mirroring
    `generate_reps_fixtures.py`'s v1 coverage member-for-member.
    """
    preop = UnitaryGateRep(_UNITARY_1Q, QUBITS_1Q)
    postop = UnitaryGateRep(_K0, QUBITS_1Q)  # any unitary-shaped array is fine here

    outcome_0 = PTMGateRep(_PTM_1Q, QUBITS_1Q)
    outcome_1 = QSimSuperopGateRep(_QSIM_SUPEROP_1Q, QUBITS_1Q)

    return {
        # GateRep subclasses
        "GATEREP_UNITARY": UnitaryGateRep(_UNITARY_1Q, QUBITS_1Q),
        "GATEREP_PTM": PTMGateRep(_PTM_1Q, QUBITS_1Q),
        "GATEREP_QSIM_SUPEROPERATOR": QSimSuperopGateRep(
            _QSIM_SUPEROP_1Q, QUBITS_1Q
        ),
        "GATEREP_STIM_CIRCUIT_STR": StimCircuitGateRep("X 0", QUBITS_1Q),
        "GATEREP_PROBABILISTIC_STIM_OPERATIONS": ProbabilisticStimGateRep(
            (("X 0", 0.5), ("Y 0", 0.5)), QUBITS_1Q
        ),
        "GATEREP_KRAUS_OPERATORS": KrausGateRep(
            ((_K0, None), (_K1, None)), QUBITS_1Q
        ),
        # InstrumentRep subclasses
        "INSTRUMENTREP_ZBASIS_PROJECTION": ZBasisProjectionInstrumentRep(
            None, True, QUBITS_1Q
        ),
        "INSTRUMENTREP_ZBASIS_PRE_POST_OPERATIONS": ZBasisPrePostInstrumentRep(
            0, True, preop, postop, QUBITS_1Q
        ),
        "INSTRUMENTREP_ZBASIS_OUTCOME_OPERATION_DICT": (
            OutcomeOperationDictInstrumentRep(
                {0: outcome_0, 1: outcome_1}, True, QUBITS_1Q
            )
        ),
        "INSTRUMENTREP_STIM_CIRCUIT_STR": StimCircuitInstrumentRep(
            "M 0", QUBITS_1Q
        ),
    }


def write_json(obj, path: Path) -> None:
    import json

    encoded = Serializable.encode(obj, format="json", reset_encode_id=True)
    with open(path, "w") as f:
        json.dump(encoded, f, indent=2)


def write_hdf5(obj, path: Path) -> None:
    with h5py.File(path, "w") as f:
        root = f.create_group("root")
        Serializable.encode(
            obj, format="hdf5", reset_encode_id=True, h5_group=root
        )


def main() -> None:
    fixtures = build_flat_rep_fixtures()
    write_json(fixtures, FIXTURES_DIR / "reps_v2.json")
    write_hdf5(fixtures, FIXTURES_DIR / "reps_v2.h5")
    print(f"Wrote {len(fixtures)} rep fixtures to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
