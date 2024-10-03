"""TODO
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TypeVar

from loqs.core.instructions import Instruction
from loqs.core.syndrome import PauliFrame, PauliFrameCastableTypes
from loqs.internal import Serializable


T = TypeVar("T", bound="QECCode")
U = TypeVar("U", bound="QECCodePatch")


class QECCode(Serializable):
    """TODO"""

    def __init__(
        self,
        instructions: Mapping[str, Instruction],
        template_qubits: Sequence[str],
        template_data_qubits: Sequence[str],
        name: str = "(Unnamed QEC code)",
    ):
        """TODO"""
        self.instructions = dict(instructions)
        self.template_qubits = list(template_qubits)
        self.template_data_qubits = list(template_data_qubits)
        assert all(
            [tdq in self.template_qubits for tdq in self.template_data_qubits]
        ), "Data qubits must a subset of all template qubits"
        self.name = name

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
        qubits: Sequence[str],
        pauli_frame: PauliFrameCastableTypes | None = None,
    ):
        """TODO"""
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
    def _from_serialization(cls: type[T], state: Mapping) -> T:
        instructions = cls.deserialize(state["instructions"])
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

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        state = super()._to_serialization()

        # QECCode is also a thing we want to cache a lot
        if (
            hash_to_serial_id_cache is not None
            and hash(self) in hash_to_serial_id_cache
        ):
            state.update(
                {
                    "type": "cached_object_reference",
                    "cache_id": hash_to_serial_id_cache[hash(self)],
                }
            )
            return state

        state.update(
            {
                "instructions": self.serialize(
                    self.instructions, hash_to_serial_id_cache
                ),
                "template_qubits": self.template_qubits,
                "template_data_qubits": self.template_data_qubits,
                "name": self.name,
            }
        )

        # If we have a cache, add this to the cache
        if hash_to_serial_id_cache is not None:
            hash_to_serial_id_cache[hash(self)] = len(hash_to_serial_id_cache)
            state.update(
                {
                    "type": "cached_object_source",
                    "cache_id": hash_to_serial_id_cache[hash(self)],
                }
            )

        return state


class QECCodePatch(Mapping[str, Instruction], Serializable):
    """TODO"""

    def __init__(
        self,
        code: QECCode,
        qubits: Sequence[str],
        pauli_frame: PauliFrameCastableTypes,
    ):
        """TODO"""
        assert len(qubits) == len(code.template_qubits), (
            f"Patch must have {len(code.template_qubits)} qubits "
            + f"to match code {code}, not {len(qubits)}"
        )

        self.code = code
        self.qubits = qubits
        self.pauli_frame = PauliFrame.cast(pauli_frame)

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
    def _from_serialization(cls: type[U], state: Mapping) -> U:
        code = cls.deserialize(state["code"])
        assert isinstance(code, QECCode)
        qubits = state["qubits"]
        pauli_frame = cls.deserialize(state["pauli_frame"])
        assert isinstance(pauli_frame, PauliFrame)
        return cls(code, qubits, pauli_frame)

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        state = super()._to_serialization()

        # Probably less impactful than the QECCode itself,
        # but could see some use in caching a QECCodePatch also
        if (
            hash_to_serial_id_cache is not None
            and hash(self) in hash_to_serial_id_cache
        ):
            state.update(
                {
                    "type": "cached_object_reference",
                    "cache_id": hash_to_serial_id_cache[hash(self)],
                }
            )
            return state

        state.update(
            {
                "code": self.serialize(self.code, hash_to_serial_id_cache),
                "qubits": self.qubits,
                "pauli_frame": self.serialize(
                    self.pauli_frame, hash_to_serial_id_cache
                ),
            }
        )

        # If we have a cache, add this to the cache
        if hash_to_serial_id_cache is not None:
            hash_to_serial_id_cache[hash(self)] = len(hash_to_serial_id_cache)
            state.update(
                {
                    "type": "cached_object_source",
                    "cache_id": hash_to_serial_id_cache[hash(self)],
                }
            )

        return state
