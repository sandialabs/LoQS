#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Generate serialization fixtures for `DictNoiseModel` and
`STIMDictNoiseModel`.

Companion to `tests/backends/fixtures/generate_reps_fixtures.py` -- see that
file's module docstring for the rationale. This script captures a
non-trivial (multi-rep-type) `DictNoiseModel` and `STIMDictNoiseModel`, in
both JSON and HDF5, as they are actually serialized by the current code.

Run with: `python tests/backends/model/fixtures/generate_model_fixtures.py`

Historical note: this script uses the pre-v1.2 `RepTuple`/`GateRep`/
`InstrumentRep` enum-based API and constructs a real `STIMDictNoiseModel`,
neither of which even *import* against current code (both removed
entirely in v1.2; excluded from pytest's collection in `pytest.ini` for
the same reason). It is kept only as a historical record of exactly how
`dictmodel_v1.{json,h5}`/`stimdictmodel_v1.{json,h5}` (the frozen fixtures
it originally produced) were generated -- `stimdictmodel_v1.{json,h5}` is
still live, as the byte-for-byte regression oracle for
`STIMDictNoiseModel`'s decode redirect to `DictNoiseModel` (see
`tests/backends/model/test_dictmodel.py`). To actually run this script,
check out commit `01b00cc` ("Minor wording fix for #59"), the last
commit on `main` before the pre-v1.2 API was removed. See
`generate_model_fixtures_v2.py` for the equivalent script against the
*current* API, producing `dictmodel_v2.{json,h5}`/
`dictmodel_stim_v2.{json,h5}` as the analogous reference for whenever the
next serialization version bump happens.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit
from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.backends.model.stimdictmodel import STIMDictNoiseModel
from loqs.backends.reps import GateRep, InstrumentRep, RepTuple
from loqs.internal.serializable import Serializable

FIXTURES_DIR = Path(__file__).parent

# TP-preserving (per ConcreteGateReps.sequence_is_krausop_rep's convention,
# sum_i K_i K_i^dagger = I) amplitude-damping Kraus operators.
_GAMMA = 0.1
_K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - _GAMMA)]])
_K1 = np.array([[0.0, 0.0], [np.sqrt(_GAMMA), 0.0]])


def build_dictmodel_fixture() -> DictNoiseModel:
    """A `DictNoiseModel` mixing several `GateRep`/`InstrumentRep` types
    (`QSIM_SUPEROPERATOR`, `KRAUS_OPERATORS`, `ZBASIS_PROJECTION`), to
    exercise more than a single rep type in one fixture."""
    gate_dict = {
        ("X", ("Q0",)): np.eye(4),  # cast to GateRep.QSIM_SUPEROPERATOR (default)
        ("KRAUS", ("Q0",)): ((_K0, None), (_K1, None)),  # -> GateRep.KRAUS_OPERATORS
    }
    inst_dict = {
        ("M", ("Q0",)): (None, True),  # -> InstrumentRep.ZBASIS_PROJECTION
    }
    return DictNoiseModel(
        (gate_dict, inst_dict),
        gatereps=[GateRep.QSIM_SUPEROPERATOR, GateRep.KRAUS_OPERATORS],
        instreps=[InstrumentRep.ZBASIS_PROJECTION],
    )


def build_stimdictmodel_fixture() -> STIMDictNoiseModel:
    """A `STIMDictNoiseModel` with a real `STIMPhysicalCircuit`-compatible
    gate/instrument dict, mirroring the shapes used throughout
    `test_stimdictmodel.py`."""
    gate_dict = {
        "X": RepTuple("X 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
        "H": RepTuple("H 0", ("Q0",), GateRep.STIM_CIRCUIT_STR),
        "CNOT": RepTuple("CNOT 0 1", ("Q0", "Q1"), GateRep.STIM_CIRCUIT_STR),
    }
    inst_dict = {
        "M": RepTuple((None, True), ("Q0",), InstrumentRep.ZBASIS_PROJECTION),
    }
    return STIMDictNoiseModel(
        (gate_dict, inst_dict),
        instreps=[InstrumentRep.ZBASIS_PROJECTION],
    )


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
    dictmodel = build_dictmodel_fixture()
    write_json(dictmodel, FIXTURES_DIR / "dictmodel_v1.json")
    write_hdf5(dictmodel, FIXTURES_DIR / "dictmodel_v1.h5")

    stimdictmodel = build_stimdictmodel_fixture()
    write_json(stimdictmodel, FIXTURES_DIR / "stimdictmodel_v1.json")
    write_hdf5(stimdictmodel, FIXTURES_DIR / "stimdictmodel_v1.h5")

    print(f"Wrote DictNoiseModel/STIMDictNoiseModel fixtures to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
