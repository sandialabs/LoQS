#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Generate serialization fixtures for `RepTuple`.

This script is NOT a test; it is a one-off/regenerable utility that captures
what the current `RepTuple`/`GateRep`/`InstrumentRep` serialization format
actually looks like, for every gate/instrument representation member, in
both JSON and HDF5. These fixtures exist so that any future change to how
these classes are represented and serialized can be validated against real
bytes produced by the current code, not just against hand-constructed dicts.

Run with: `python tests/backends/fixtures/generate_reps_fixtures.py`

Only re-run this if you intend to deliberately change what these fixtures
represent (e.g. if a bug fix to current behavior should be reflected in
them) -- otherwise these files should remain untouched so they keep
representing a known, previously-committed serialized format.

Historical note: this script uses the pre-v1.2 `RepTuple`/`GateRep`/
`InstrumentRep` enum-based API, which no longer even *imports* against
current code (`RepTuple` was removed entirely in v1.2; excluded from
pytest's collection in `pytest.ini` for the same reason). It is kept only
as a historical record of exactly how `reps_v1.json`/`reps_v1.h5` (the
frozen fixtures it originally produced) were generated -- those fixtures
are still live, as the byte-for-byte regression oracle for the legacy
decode redirect in `loqs.backends.reps.base.OperationRep._from_decoded_attrs`
(see `tests/backends/reps/test_legacy.py`). To actually run this script
(e.g. to regenerate the fixtures from scratch, or just to confirm it
still produces the same bytes), check out commit `01b00cc` ("Minor
wording fix for #59"), the last commit on `main` before the pre-v1.2 API
was removed. See `generate_reps_fixtures_v2.py` for the equivalent
script against the *current* API, producing `reps_v2.json`/`reps_v2.h5`
as the analogous reference for whenever the next serialization version
bump happens.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from loqs.backends.reps import GateRep, InstrumentRep, RepTuple
from loqs.internal.serializable import Serializable

FIXTURES_DIR = Path(__file__).parent

QUBITS_1Q = ("Q0",)
QUBITS_2Q = ("Q0", "Q1")

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


def build_flat_rep_fixtures() -> dict[str, RepTuple]:
    """One `RepTuple` per `GateRep`/`InstrumentRep` member (10 total),
    including the two `InstrumentRep` members whose payload contains nested
    `RepTuple`s (`ZBASIS_PRE_POST_OPERATIONS`, `ZBASIS_OUTCOME_OPERATION_DICT`).
    """
    preop = RepTuple(_UNITARY_1Q, QUBITS_1Q, GateRep.UNITARY)
    postop = RepTuple(_K0, QUBITS_1Q, GateRep.UNITARY)  # any unitary-shaped array is fine here

    outcome_0 = RepTuple(_PTM_1Q, QUBITS_1Q, GateRep.PTM)
    outcome_1 = RepTuple(_QSIM_SUPEROP_1Q, QUBITS_1Q, GateRep.QSIM_SUPEROPERATOR)

    return {
        # GateRep members
        "GATEREP_UNITARY": RepTuple(_UNITARY_1Q, QUBITS_1Q, GateRep.UNITARY),
        "GATEREP_PTM": RepTuple(_PTM_1Q, QUBITS_1Q, GateRep.PTM),
        "GATEREP_QSIM_SUPEROPERATOR": RepTuple(
            _QSIM_SUPEROP_1Q, QUBITS_1Q, GateRep.QSIM_SUPEROPERATOR
        ),
        "GATEREP_STIM_CIRCUIT_STR": RepTuple(
            "X 0", QUBITS_1Q, GateRep.STIM_CIRCUIT_STR
        ),
        "GATEREP_PROBABILISTIC_STIM_OPERATIONS": RepTuple(
            (("X 0", 0.5), ("Y 0", 0.5)),
            QUBITS_1Q,
            GateRep.PROBABILISTIC_STIM_OPERATIONS,
        ),
        "GATEREP_KRAUS_OPERATORS": RepTuple(
            ((_K0, None), (_K1, None)), QUBITS_1Q, GateRep.KRAUS_OPERATORS
        ),
        # InstrumentRep members
        "INSTRUMENTREP_ZBASIS_PROJECTION": RepTuple(
            (None, True), QUBITS_1Q, InstrumentRep.ZBASIS_PROJECTION
        ),
        "INSTRUMENTREP_ZBASIS_PRE_POST_OPERATIONS": RepTuple(
            (0, True, preop, postop),
            QUBITS_1Q,
            InstrumentRep.ZBASIS_PRE_POST_OPERATIONS,
        ),
        "INSTRUMENTREP_ZBASIS_OUTCOME_OPERATION_DICT": RepTuple(
            ({0: outcome_0, 1: outcome_1}, True),
            QUBITS_1Q,
            InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT,
        ),
        "INSTRUMENTREP_STIM_CIRCUIT_STR": RepTuple(
            "M 0", QUBITS_1Q, InstrumentRep.STIM_CIRCUIT_STR
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
    write_json(fixtures, FIXTURES_DIR / "reps_v1.json")
    write_hdf5(fixtures, FIXTURES_DIR / "reps_v1.h5")
    print(f"Wrote {len(fixtures)} RepTuple fixtures to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
