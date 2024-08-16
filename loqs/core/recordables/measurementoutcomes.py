""":class:`MeasurementOutcomes` definition.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import TypeAlias

from loqs.backends.state.basestate import OutcomeDict
from loqs.internal.castable import Castable


MeasurementOutcomesCastableTypes: TypeAlias = (
    "MeasurementOutcomes | Mapping[str, int | Sequence[int]]"
)


class MeasurementOutcomes(Castable, Mapping[str, list[int]]):
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
            self.outcomes = self.outcomes
        else:
            self.outcomes = {}
            for k, v in outcomes.items():
                self.outcomes[k] = [v] if isinstance(v, int) else list(v)

    def __getitem__(self, key: str) -> list[int]:
        return self.outcomes[key]

    def __len__(self) -> int:
        return len(self.outcomes)

    def __iter__(self) -> Iterator[str]:
        return iter(self.outcomes)

    def __str__(self) -> str:
        return f"MeasurementOutcomes({self.outcomes})"
