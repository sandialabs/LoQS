""":class:`MeasurementOutcomes` definition.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Literal, TypeAlias, TypeVar

from loqs.backends.state.basestate import OutcomeDict
from loqs.core.syndrome import PauliFrame
from loqs.internal import Castable, Serializable


T = TypeVar("T", bound="MeasurementOutcomes")

MeasurementOutcomesCastableTypes: TypeAlias = (
    "MeasurementOutcomes | Mapping[str, int | Sequence[int]]"
)


class MeasurementOutcomes(Mapping[str, list[int]], Castable, Serializable):
    """TODO"""

    outcomes: OutcomeDict
    """Dict with qubit label keys and list of 0/1 outcome values.

    Can be multiple outcomes if the qubit
    was measured multiple times, e.g.
    auxiliary qubit reuse.
    """

    def __init__(self, outcomes: MeasurementOutcomesCastableTypes) -> None:
        """Initialize a :class:`MockState`.

        Parameters
        ----------
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

    def __getitem__(self, key: str) -> list[int]:
        return self.outcomes[key]

    def __len__(self) -> int:
        return len(self.outcomes)

    def __iter__(self) -> Iterator[str]:
        return iter(self.outcomes)

    def __str__(self) -> str:
        return f"MeasurementOutcomes({self.outcomes})"

    def __hash__(self) -> int:
        return self.hash(self.outcomes)

    @classmethod
    def cast(cls: type[T], obj: object) -> T:
        """Cast to the derived class.

        For Frame objects, a dict is an allowed first argument,
        so we add a check for expected constructor kwarg names.

        Parameters
        ----------
        obj:
            A castable object that is either:
            - Already the derived class type, in which case `obj`
            is returned
            - A kwarg dict that is passed into the derived class
            constructor
            - The first argument of the derived class constructor

        Returns
        -------
            An object with type T (matching the derived class)
        """
        if isinstance(obj, cls):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, dict) and ("outcomes" in obj):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)

        # Otherwise, assume this is the first __init__ argument
        return cls(obj)  # type: ignore

    def get_inferred_outcomes(
        self,
        basis: Literal["Z"] | Literal["X"],
        pauli_frame: PauliFrame | None = None,
    ) -> MeasurementOutcomes:
        """TODO"""
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
    def _from_serialization(cls: type[T], state: Mapping) -> T:
        outcomes = state["outcomes"]
        return cls(outcomes)

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        state = super()._to_serialization()
        state.update({"outcomes": self.outcomes})
        return state
