""":class:`MeasurementOutcomes` definition.
"""

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from loqs.core import Recordable


# TODO: increase this
MeasurementOutcomesCastableTypes: TypeAlias = (
    "MeasurementOutcomes | Mapping[str, int | Sequence[int]]"
)


class MeasurementOutcomes(Recordable):
    """TODO"""

    outcomes: Mapping[str, list[int]]
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
