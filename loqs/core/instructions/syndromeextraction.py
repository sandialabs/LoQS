"""TODO
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TypeAlias

from loqs.backends.state import BaseQuantumState
from loqs.core import HistoryStack, HistoryFrame
from loqs.core.history import HistoryStackCastableTypes
from loqs.core.instruction import InstructionParentTypes
from loqs.core.instructions import QuantumClassicalLogicalOperation
from loqs.core.instructions.logicaloperation import (
    LogicalOperationCastableTypes,
)
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
        physical_circuit: LogicalOperationCastableTypes,
        syndrome_measurements: SyndromeLabelsTypes,
        name: str = "(Unnamed syndrome extraction)",
        parent: InstructionParentTypes = None,
        fault_tolerant: bool | None = None,
    ) -> None:
        """TODO

        Parameters
        ----------
        """
        super().__init__(physical_circuit, name, parent, fault_tolerant)

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

    # This is one of the cases where we do not have a nice single argument cast
    # Override the cast method to accept a tuple of the first two args
    @classmethod
    def cast(cls, obj: object) -> SyndromeExtraction:
        """Cast to a :class:`SyndromeExtraction` object.

        Unlike most castable objects, :class:`SyndromeExtraction`
        requires 2 inputs. This version of cast additionally allows
        a tuple/list variant for the two arguments and disallows
        a single object being passed in.

        Parameters
        ----------
        obj:
            A castable object that is either:
            - Already a :class:`SyndromeExtraction` object,
            in which case `obj` is returned
            - A kwarg dict that is passed into the constructor
            - A sequence of the first two arguments of the
            :class:`SyndromeExtraction` constructor

        Returns
        -------
            A :class:`SyndromeExtraction` object
        """
        if isinstance(obj, cls):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, dict):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)
        elif isinstance(obj, Sequence) and len(obj) == 2:
            # Assume this is a tuple/list of first two args
            return cls(obj[0], obj[1])

        # Else we can't handle this
        raise ValueError(
            "SyndromeExtraction requires two arguments to cast. "
            + "Use a 2-tuple or kwarg dict when casting."
        )

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

    def map_qubits(
        self, qubit_mapping: Mapping[str, str]
    ) -> SyndromeExtraction:
        mapped_circ = self.physical_circuit.map_qubit_labels(qubit_mapping)
        mapped_syndrome_labels = [
            (qubit_mapping[sl[0]], sl[1]) for sl in self.syndrome_labels
        ]
        return SyndromeExtraction(
            mapped_circ, mapped_syndrome_labels, self.name, self.parent
        )
