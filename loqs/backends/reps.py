""":class:`.RepTuple` and :class:`.RepEnum` definitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TypeVar

from loqs.internal import Castable, Displayable


T = TypeVar("T", bound="RepEnum")
U = TypeVar("U", bound="RepTuple")


class RepEnum(Displayable, Enum):
    """Base class for all operation representation enums."""

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.value))

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        value = state["value"]
        return cls(value)

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        state = super()._to_serialization()
        state.update({"value": self.value})
        return state


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

    STIM_CIRCUIT_STR = 4
    """STIM circuit string

    The expected rep type is STIM circuit string placeholder
    qubit labels. The string can include bott gates (e.g. ``"H"``,
    ``"CX"``) and noise specifications (e.g. ``"X_ERROR(<rate>)"``,
    ``"DEPOLARIZE1(<rate>)"``). Qubit labels are placeholders
    indexing into paired :attr:`.RepTuple.qubits`.
    """

    PROBABILISTIC_STIM_OPERATIONS = 5
    """A weighted set of STIM circuit strings.

    By default, STIM can only do Pauli noise channels. However,
    some error channels can be "unraveled" into a probabilistic
    choice from Pauli channels. For example, amplitude damping
    can be performed as a probabilistic reset.

    The expected rep type is a list of 2-tuples, where the first
    entry is circuit string to apply if chosen and the second entry
    is the probability of sampling that operation.
    Probabilities should be positive and add to 1.
    """

    KRAUS_OPERATORS = 6
    r"""A list of Kraus operators.

    The Kraus operators for a CP channel :math:`\Lambda` are
    defined as :math:`K_i` s.t. the

    .. math::

        \Lambda(\rho) = \sum_i K_i \rho K_i^\dagger

    The Kraus operators do not have to be unitary, Hermitian,
    or invertible, but the map is also TP if they obey

    .. math::

        \sum_i K_i^\dagger K_i = I


    This representation is convenient for all sorts of
    "unraveling" techniques. Critically, it is also possible
    to unravel non-unital channels such as amplitude damping.
    In that case, one must sample from the probability
    distribution given by

    .. math::

        P_i = \mathrm{Tr}\left[\rho K_i K_i^\dagger]

    After sampling which Kraus operator to apply, the final state
    is then

    .. math::

        \rho \rightarrow K_i \rho K_i^\dagger / P_i

    Note the renormalization by probability here!

    This unraveling of non-unital channels can even be done with a
    :class:`.STIMQuantumState`, enabling fast stabilizer simulation
    with amplitude damping.

    The expected rep type is a list of arrays with shape (2^n, 2^n)
    where n is the number of qubits.
    """


class InstrumentRep(RepEnum):
    """Representations for instrument objects."""

    ZBASIS_PROJECTION = 1
    """Z-basis projection.

    Essentially a perfect mid-circuit measurement,
    followed by optional reset.
    The expected rep is 2-tuple where the first entry
    is None for no reset or 0 or 1 for reset to the
    corresponding state, and the second entry is a bool
    which indicates whether the outcome should be
    recorded, e.g. (0, False) would look like a pure reset.
    """

    ZBASIS_PRE_POST_OPERATIONS = 2
    """Perfect Z-basis projection with noisy pre-/post-operations.

    For when a mid-circuit measurement can be modeled by a perfect
    Z-basis projection sandwiched by two noisy operations.
    The expected rep is a 4-tuple where the first two elements are
    as per :attr:`.InstrumentRep.ZBASIS_PROJECTION`, and then two
    :class:`.RepTuple` objects with a :class:`.GateRep` ``reptype``.
    """

    ZBASIS_OUTCOME_OPERATION_DICT = 3
    """Dict with MCM outcome labels and CP map operation keys.

    For when a mid-circuit measurement can be modeled by a
    ``pyGSTi``-like quantum instrument.
    The expected rep is a 2-tuple where the first entry is a
    dict with tuple of outcome keys and :class:`.RepTuple` objects
    with a :class:`.GateRep` ``reptype`` for values, and the second
    entry is a bool which indicates whether the outcome should be recorded,
    e.g. ({...}, False) would look like a noisy reset.
    """


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

    def __len__(self) -> int:
        return 3

    def __hash__(self) -> int:
        return hash(
            (
                self.hash(self.rep),
                self.hash(self.qubits),
                self.hash(self.reptype),
            )
        )

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
        cls: type[U], state: Mapping, serial_id_to_obj_cache=None
    ) -> U:
        rep = cls.deserialize(state["rep"])
        qubits = state["qubits"]
        reptype = cls.deserialize(state["reptype"])
        assert isinstance(reptype, RepEnum)
        return cls(rep, qubits, reptype)

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "rep": self.serialize(self.rep),
                "qubits": self.qubits,
                "reptype": self.serialize(self.reptype),
            }
        )
        return state
