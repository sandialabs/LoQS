"""TODO
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TypeAlias

from loqs.backends.state import BaseQuantumState
from loqs.core import HistoryStack, HistoryFrame
from loqs.core.history import HistoryStackCastableTypes
from loqs.core.instructions import QuantumClassicalLogicalOperation
from loqs.core.instructions.logicaloperation import LogicalOperationCastable
from loqs.core.recordables import MeasurementOutcomes, Syndrome
from loqs.internal import PauliStr


SyndromeLabelTypes: TypeAlias = str | tuple[str, int]
"""TODO
"""

SyndromeLabelsTypes: TypeAlias = (
    SyndromeLabelTypes
    | Sequence[SyndromeLabelTypes]
    | Mapping[PauliStr, SyndromeLabelTypes]
)
"""TODO
"""


class SyndromeExtraction(QuantumClassicalLogicalOperation):
    """TODO"""

    def __init__(
        self,
        physical_circuit: LogicalOperationCastable,
        syndrome_measurements: SyndromeLabelsTypes,
    ) -> None:
        """TODO

        Parameters
        ----------
        """
        self.physical_circuit = physical_circuit

        if isinstance(syndrome_measurements, Mapping):
            self.stabilizers = list(syndrome_measurements.keys())
            syndrome_measurements = list(syndrome_measurements.values())
        if not isinstance(syndrome_measurements, Sequence):
            syndrome_measurements = [syndrome_measurements]

        counter: dict[str, int] = defaultdict(int)
        self.syndrome_labels = []
        for syndrome_label in syndrome_measurements:
            if isinstance(syndrome_label, tuple):
                self.syndrome_labels.append(syndrome_label)
            elif isinstance(syndrome_label, str):
                # We are only given qubit name
                # Assume the n-th subsequence measurement on this qubit
                self.syndrome_labels.append(
                    (syndrome_label, counter[syndrome_label])
                )

                # and increment counter
                counter[syndrome_label] += 1

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return {
            "state": BaseQuantumState,
            "measurement_outcomes": MeasurementOutcomes,
            "syndrome": Syndrome,
        }

    def apply_unsafe(self, input: HistoryStackCastableTypes) -> HistoryFrame:
        """Map the input :class:`MockState` forward.

        This

        Parameters
        ----------
        input:
            The input frame/history information

        Returns
        -------
        output_frame:
            The new output frame
        """
        input = HistoryStack.cast(input)

        last_frame: HistoryFrame = input[-1]

        state = last_frame["state"]

        # TODO: apply operations to new state

        # TODO: Extract syndrome from measurement outcomes

        new_data = {
            "state": state,
            "measurement_outcomes": None,  # TODO
            "syndrome": None,  # TODO
            "instruction": self,
        }

        output_frame = last_frame.update(
            new_data=new_data, new_log=f"{self.name} result"
        )
        return output_frame
