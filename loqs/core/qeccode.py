""":class:`.QECCode` and :class:`.QECCodePatch` definitions.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import ClassVar, TypeVar

from loqs.core.instructions import Instruction
from loqs.core.syndrome import PauliFrame, PauliFrameCastableTypes
from loqs.internal import Displayable


T = TypeVar("T", bound="QECCode")
U = TypeVar("U", bound="QECCodePatch")


class QECCode(Displayable):
    """A set of :class:`.Instruction` objects that implement a QEC code.

    All qubit-specific quantities are defined with respect to a set of
    template qubits that can then be replaced with real qubit labels
    at runtime.
    """

    CACHE_ON_SERIALIZE: ClassVar[bool] = True

    def __init__(
        self,
        instructions: Mapping[str, Instruction],
        template_qubits: Sequence[str | int],
        template_data_qubits: Sequence[str | int],
        name: str = "(Unnamed QEC code)",
    ):
        """
        Parameters
        ----------
        instructions:
            See :attr:`.instructions`.

        template_qubits:
            See :attr:`.template_qubits`.

        template_data_qubits:
            See :attr:`.template_data_qubits`.

        name:
            See :attr:`.name`.
        """
        self.instructions = dict(instructions)
        """A mapping from name keys to :class:`.Instruction` values."""

        self.template_qubits = list(template_qubits)
        """All template qubits used in :attr:`instructions`."""

        self.template_data_qubits = list(template_data_qubits)
        """The entries of :attr:`template_qubits` corresponding to data qubits."""

        assert all(
            [tdq in self.template_qubits for tdq in self.template_data_qubits]
        ), "Data qubits must a subset of all template qubits"

        self.name = name
        """Name for logging"""

    def __str__(self) -> str:
        return f"QECCode {self.name}"

    def __hash__(self) -> int:
        return hash(
            (
                self.hash(self.instructions),
                tuple(self.template_qubits),
                tuple(self.template_data_qubits),
                self.name,
            )
        )

    def create_patch(
        self,
        qubits: Sequence[str | int],
        pauli_frame: PauliFrameCastableTypes | None = None,
    ) -> QECCodePatch:
        """Create a :class:`.QECCodePatch` based on this :class:`QECCode`.

        Parameters
        ----------
        qubits:
            Qubit labels to replace :attr:`.template_qubits`.

        pauli_frame:
            An initial :class:`.PauliFrame` to assign to the patch.
            Defaults to ``None``, which assigns the trivial Pauli frame
            of all ``"I"`` entries.

        Returns
        -------
        QECCodePatch
            The constructed :class:`.QECCodePatch`
        """
        if pauli_frame is None:
            # Map template data qubits to real qubits
            data_qubits = [
                qubits[self.template_qubits.index(tdq)]
                for tdq in self.template_data_qubits
            ]
            # Initialize an empty PauliFrame on our data qubits
            pauli_frame = data_qubits
        return QECCodePatch(self, qubits, pauli_frame)

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        instructions = cls.deserialize(
            state["instructions"], serial_id_to_obj_cache
        )
        assert isinstance(instructions, dict)
        template_qubits = state["template_qubits"]
        template_data_qubits = state["template_data_qubits"]
        name = state["name"]
        return cls(
            instructions,
            template_qubits,
            template_data_qubits,
            name=name,
        )

    def _to_serialization(
        self, hash_to_serial_id_cache=None, ignore_no_serialize_flags=False
    ) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "instructions": self.serialize(
                    self.instructions,
                    hash_to_serial_id_cache,
                    ignore_no_serialize_flags,
                ),
                "template_qubits": self.template_qubits,
                "template_data_qubits": self.template_data_qubits,
                "name": self.name,
            }
        )
        return state


class QECCodePatch(Mapping[str, Instruction], Displayable):
    """An instantiation of a :class:`.QECCode` on a set of qubits.

    This object acts like a ``dict``, where instruction names are the
    keys and the appropriate :class:`.Instruction` (mapped to the patch
    qubits) is returned.
    It also stores the :class:`.PauliFrame` for the data qubits, as this
    is the natural place for it.
    """

    CACHE_ON_SERIALIZE: ClassVar[bool] = True

    def __init__(
        self,
        code: QECCode,
        qubits: Sequence[str | int],
        pauli_frame: PauliFrameCastableTypes,
    ):
        """
        Parameters
        ----------
        code:
            See :attr:`.code`.

        qubits:
            See :attr:`.qubits`.

        pauli_frame:
            See :attr:`.pauli_frame`.
        """
        assert len(qubits) == len(code.template_qubits), (
            f"Patch must have {len(code.template_qubits)} qubits "
            + f"to match code {code}, not {len(qubits)}"
        )

        self.code = code
        """The :class:`.QECCode` being used on this patch of qubits."""

        self.qubits = qubits
        """The qubits this patch acts on."""

        self.pauli_frame = PauliFrame.cast(pauli_frame)
        """The Pauli frame tracking errors on these qubits."""

        self.data = {}
        """Extra patch-specific data to be tracked."""

    def __getitem__(self, key: str) -> Instruction:
        try:
            template_op = self.code.instructions[key]
        except KeyError:
            raise KeyError(
                f"Operation {key} not available in code {self.code}"
            )

        mapping = {
            k: v for k, v in zip(self.code.template_qubits, self.qubits)
        }
        return template_op.map_qubits(mapping)

    def __len__(self) -> int:
        return len(self.code.instructions)

    def __iter__(self) -> Iterator[str]:
        return iter(self.code.instructions)

    def __str__(self) -> str:
        s = f"QECCodePatch for {self.code.name} on qubits "
        s += f"[{self.qubits[0]},...,{self.qubits[-1]}]" + "\n"
        s += f"  Current frame: {self.pauli_frame.pauli_frame}"
        return s

    def __hash__(self) -> int:
        return hash(
            (
                hash(self.code),
                tuple(self.qubits),
                hash(self.pauli_frame),
            )
        )

    @classmethod
    def _from_serialization(
        cls: type[U], state: Mapping, serial_id_to_obj_cache=None
    ) -> U:
        code = cls.deserialize(state["code"], serial_id_to_obj_cache)
        assert isinstance(code, QECCode)
        qubits = state["qubits"]
        pauli_frame = cls.deserialize(
            state["pauli_frame"], serial_id_to_obj_cache
        )
        assert isinstance(pauli_frame, PauliFrame)
        return cls(code, qubits, pauli_frame)

    def _to_serialization(
        self, hash_to_serial_id_cache=None, ignore_no_serialize_flags=False
    ) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "code": self.serialize(
                    self.code,
                    hash_to_serial_id_cache,
                    ignore_no_serialize_flags,
                ),
                "qubits": self.qubits,
                "pauli_frame": self.serialize(
                    self.pauli_frame,
                    hash_to_serial_id_cache,
                    ignore_no_serialize_flags,
                ),
            }
        )
        return state
