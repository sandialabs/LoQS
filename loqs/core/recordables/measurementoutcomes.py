#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

""":class:`.MeasurementOutcomes` definition.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Literal, TypeAlias, TypeVar

from loqs.backends.state.basestate import OutcomeDict
from loqs.core.syndrome import PauliFrame
from loqs.internal import MapCastable, Displayable


T = TypeVar("T", bound="MeasurementOutcomes")

MeasurementOutcomesCastableTypes: TypeAlias = (
    "MeasurementOutcomes | Mapping[str | int, int | Sequence[int]]"
)
"Things that can be cast to :class:`.MeasurementOutcomes`."


class MeasurementOutcomes(
    Mapping[str | int, list[int]], MapCastable, Displayable
):
    """Measurement outcomes from physical circuit instructions.

    This is a dict-like object with qubit label keys and lists of 0/1
    outcome values. These can represent both raw measurement outcomes
    or "inferred" outcomes where a :class:`.PauliFrame` has been applied
    (see :attr:`.get_inferred_outcomes`).
    """

    outcomes: OutcomeDict
    """Dict with qubit label keys and list of 0/1 outcome values.

    Can be multiple outcomes if the qubit was measured multiple times,
    e.g. auxiliary qubit reuse in a single circuit.
    """

    def __init__(self, outcomes: MeasurementOutcomesCastableTypes) -> None:
        """
        Parameters
        ----------
        outcomes:
            See :attr:`.outcomes`. No default since this is intended to be
            immutable, i.e. data is given once now and then not changed.
        """
        if isinstance(outcomes, MeasurementOutcomes):
            self.outcomes = outcomes.outcomes
        elif isinstance(outcomes, Mapping):
            self.outcomes = {}
            for k, v in outcomes.items():
                self.outcomes[k] = [v] if isinstance(v, int) else list(v)
        else:
            raise TypeError(
                "Must pass dict of qubit keys and outcome/list of outcome values"
            )

    def __getitem__(self, key: str | int) -> list[int]:
        return self.outcomes[key]

    def __len__(self) -> int:
        return len(self.outcomes)

    def __iter__(self) -> Iterator[str | int]:
        return iter(self.outcomes)

    def __str__(self) -> str:
        return f"MeasurementOutcomes({self.outcomes})"

    def __hash__(self) -> int:
        return self.hash(self.outcomes)

    def map_qubits_inplace(
        self, qubit_mapping: Mapping[str | int, str | int]
    ) -> None:
        """Map the qubit label keys (in-place).

        Parameters
        ----------
        qubit_mapping:
            Qubit mapping from current keys to new values.
        """
        self.outcomes = {
            qubit_mapping.get(q, q): v for q, v in self.outcomes.items()
        }

    def map_qubits(
        self, qubit_mapping: Mapping[str | int, str | int]
    ) -> MeasurementOutcomes:
        """Return a copy with mapped qubit label keys.

        Parameters
        ----------
        qubit_mapping:
            Qubit mapping from current keys to new values.

        Returns
        -------
        MeasurementOutcomes
            Outcomes with mapped qubit labels
        """
        new_outcomes = MeasurementOutcomes(self)
        new_outcomes.map_qubits_inplace(qubit_mapping)
        return new_outcomes

    def get_inferred_outcomes(
        self,
        pauli_frame: PauliFrame | None = None,
        basis: Literal["Z"] | Literal["X"] = "Z",
    ) -> MeasurementOutcomes:
        """Apply a :class:`.PauliFrame` to get inferred outcomes.

        Parameters
        ----------
        pauli_frame:
            The :class:`.PauliFrame` to apply. Defaults to ``None``,
            in which case this just returns a copy.

        basis:
            Which measurement basis to use when applying the ``pauli_frame``.
            Must be one of ``["X", "Z"]``, and defaults to ``"Z"``.
        """
        if pauli_frame is None:
            return MeasurementOutcomes(self.outcomes.copy())

        assert basis in "XZ"
        bitflip_basis = "Z" if basis == "X" else "X"

        inferred_outcomes = {}
        for qubit, outs in self.outcomes.items():
            bitflip = pauli_frame.get_bit(bitflip_basis, qubit)
            inferred_outcomes[qubit] = [(o + bitflip) % 2 for o in outs]
        return MeasurementOutcomes(inferred_outcomes)

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        outcomes = state["outcomes"]
        return cls(outcomes)

    def _to_serialization(
        self, hash_to_serial_id_cache=None, ignore_no_serialize_flags=False
    ) -> dict:
        state = super()._to_serialization()
        state.update({"outcomes": self.outcomes})
        return state
