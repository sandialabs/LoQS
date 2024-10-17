""":class:`.RepTuple` and :class:`.RepEnum` definitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TypeVar

from loqs.internal import Castable, Displayable


class RepEnum(Enum):
    """Base class for all operation representation enums."""


class GateRep(RepEnum):
    """Representations for gate objects."""

    UNITARY = 1
    """Unitary matrices.

    The expected rep type is an array with
    shape (2^n, 2^n) where n is the number
    of qubits.
    """

    PTM = 2
    """Pauli-transfer matrices

    A process matrix in the Pauli-product basis.
    The expected rep type is an array with
    shape (4^n, 4^n) where n is the number
    of qubits.
    """

    QSIM_SUPEROPERATOR = 3
    """QuantumSim-basis superoperator

    Process matrices in QuantumSim's non-standard basis.
    The expected rep type is an array with
    shape (4^n, 4^n) where n is the number
    of qubits.
    """
    # TODO: Kraus? Some other Clifford/stabilizer/symplectic stuff?


class InstrumentRep(RepEnum):
    """Representations for instrument objects."""

    ZBASIS_PROJECTION = 1
    """Z-basis projection.

    Essentially a perfect mid-circuit measurement.
    The expected rep is None.
    """

    ZBASIS_PRE_POST_OPERATIONS = 2
    """Perfect Z-basis projection with noisy pre-/post-operations.

    For when a mid-circuit measurement can be modeled by a perfect
    Z-basis projection sandwiched by two noisy operations.
    The expected rep is a 2-tuple of RepTuples.
    TODO: Make them labels that can be looked up?
    """

    ZBASIS_OUTCOME_OPERATION_DICT = 3
    """Dict with MCM outcome labels and CP map operation keys.

    For when a mid-circuit measurement can be modeled by a
    ``pyGSTi``-like quantum instrument.
    The expected rep is a dict with tuple of outcome keys
    and RepTuple values.
    TODO: Make values labels that can be looked up?
    """


T = TypeVar("T", bound="RepTuple")


class RepTuple(Castable, Displayable):
    rep: object
    """Underlying representation object."""

    qubits: tuple[str | int, ...]
    """Qubit labels that :attr:`.rep` should be applied to."""

    reptype: RepEnum
    """Enum entry indicating how :attr:`.rep` should be interpreted."""

    def __init__(
        self,
        rep: object,
        qubits: str | int | Sequence[str | int],
        reptype: RepEnum,
    ):
        self.rep = rep
        if isinstance(qubits, (str, int)):
            self.qubits = (qubits,)
        else:
            self.qubits = tuple(qubits)
        self.reptype = reptype
        assert isinstance(self.reptype, RepEnum)

    # Make it tuple-like
    def __getitem__(self, i: int) -> object:
        if i == 0:
            return self.rep
        elif i == 1:
            return self.qubits
        elif i == 2:
            return self.reptype
        else:
            return KeyError("RepTuple only has 3 entries")

    @classmethod
    def cast(cls: type[RepTuple], obj: object) -> RepTuple:
        """Cast this object to a :class:`RepTuple`.

        This is specialized because lists/tuples with up to 3 entries
        should be unpacked into the three arguments.
        """
        if isinstance(obj, cls):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, dict):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)
        elif isinstance(obj, (tuple, list)):
            assert len(obj) < 4
            return cls(*obj)

        raise ValueError(f"Cannot cast {obj} to a RepTuple")

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        rep = cls.deserialize(state["rep"])
        qubits = state["qubits"]
        reptype = state["reptype"]
        return cls(rep, qubits, reptype)

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "rep": self.serialize(self.rep),
                "qubits": self.qubits,
                "reptype": self.reptype,
            }
        )
        return state
