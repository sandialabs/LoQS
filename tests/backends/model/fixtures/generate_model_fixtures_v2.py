#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Generate `SERIALIZATION_VERSION` 2 serialization fixtures for
`DictNoiseModel`.

The v2 counterpart of `generate_model_fixtures.py` -- see that file's
module docstring, and `generate_reps_fixtures_v2.py`'s, for the full
rationale. Two fixtures are produced, mirroring the v1 pair's coverage:
a plain multi-rep-type model (`dictmodel_v2.{json,h5}`), and a
STIM-flavored one (`dictmodel_stim_v2.{json,h5}`) built directly as a
`DictNoiseModel` with STIM-appropriate `gatereps`/`instreps` -- there is
no separate `STIMDictNoiseModel` class to construct anymore, since
`DictNoiseModel` itself now handles STIM circuits natively.

Run with: `python tests/backends/model/fixtures/generate_model_fixtures_v2.py`

Only re-run this if you intend to deliberately change what these
fixtures represent -- otherwise these files should remain untouched so
they keep representing a known, previously-committed serialized format.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.backends.reps import (
    KrausGateRep,
    QSimSuperopGateRep,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.internal.serializable import Serializable

FIXTURES_DIR = Path(__file__).parent

# TP-preserving (sum_i K_i K_i^dagger = I) amplitude-damping Kraus operators.
_GAMMA = 0.1
_K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - _GAMMA)]])
_K1 = np.array([[0.0, np.sqrt(_GAMMA)], [0.0, 0.0]])


def build_dictmodel_fixture() -> DictNoiseModel:
    """A `DictNoiseModel` mixing several `GateRep`/`InstrumentRep`
    subclasses (`QSimSuperopGateRep`, `KrausGateRep`,
    `ZBasisProjectionInstrumentRep`), mirroring
    `generate_model_fixtures.py`'s v1 `build_dictmodel_fixture`."""
    gate_dict = {
        ("X", ("Q0",)): QSimSuperopGateRep(np.eye(4), ("Q0",)),
        ("KRAUS", ("Q0",)): KrausGateRep(((_K0, None), (_K1, None)), ("Q0",)),
    }
    inst_dict = {
        ("M", ("Q0",)): ZBasisProjectionInstrumentRep(None, True, ("Q0",)),
    }
    return DictNoiseModel(
        gate_dict,
        inst_dict,
        gatereps=[QSimSuperopGateRep, KrausGateRep],
        instreps=[ZBasisProjectionInstrumentRep],
    )


def build_dictmodel_stim_fixture() -> DictNoiseModel:
    """A `DictNoiseModel` with a real `STIMPhysicalCircuit`-compatible
    gate/instrument dict, mirroring `generate_model_fixtures.py`'s v1
    `build_stimdictmodel_fixture` -- but as a plain `DictNoiseModel`
    directly, since there is no longer a separate `STIMDictNoiseModel`
    class."""
    gate_dict = {
        "X": StimCircuitGateRep("X 0", ("Q0",)),
        "H": StimCircuitGateRep("H 0", ("Q0",)),
        "CNOT": StimCircuitGateRep("CNOT 0 1", ("Q0", "Q1")),
    }
    inst_dict = {
        "M": ZBasisProjectionInstrumentRep(None, True, ("Q0",)),
    }
    return DictNoiseModel(
        gate_dict,
        inst_dict,
        gatereps=[StimCircuitGateRep],
        instreps=[ZBasisProjectionInstrumentRep, StimCircuitInstrumentRep],
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
    write_json(dictmodel, FIXTURES_DIR / "dictmodel_v2.json")
    write_hdf5(dictmodel, FIXTURES_DIR / "dictmodel_v2.h5")

    stim_dictmodel = build_dictmodel_stim_fixture()
    write_json(stim_dictmodel, FIXTURES_DIR / "dictmodel_stim_v2.json")
    write_hdf5(stim_dictmodel, FIXTURES_DIR / "dictmodel_stim_v2.h5")

    print(f"Wrote DictNoiseModel fixtures to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
