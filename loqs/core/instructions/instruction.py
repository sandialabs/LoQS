"""TODO
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import inspect as ins
import textwrap
from typing import (
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)
import warnings

from loqs.core import Frame
from loqs.core.frame import FrameCastableTypes
from loqs.internal.serializable import Serializable


T = TypeVar("T", bound="Instruction")
P = ParamSpec("P")

KwargDict: TypeAlias = dict[str, object]


@runtime_checkable
class ApplyCallable(Protocol[P]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Frame: ...  # noqa


class MapQubitsCallable(Protocol[P]):
    def __call__(  # noqa
        self,
        qubit_mapping: Mapping[str, str],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> KwargDict: ...


def default_map_qubits(qubit_mapping: Mapping[str, str], **kwargs):
    # Assume nothing needs to be mapped in kwargs
    return kwargs


DEFAULT_PRIORITIES = ["label", "instruction", "program", "history[-1]"]
"""Default parameter priority order.
"""


class Instruction(Serializable):
    """TODO"""

    def __init__(
        self,
        apply_fn: ApplyCallable,
        data: Mapping[str, object] | None = None,
        map_qubits_fn: MapQubitsCallable = default_map_qubits,
        param_priorities: Mapping[str, Sequence[str]] | None = None,
        param_error_behavior: Literal["continue", "warn", "raise"] = "warn",
        param_aliases: Mapping[str, str] | None = None,
        name: str = "(Unnamed instruction)",
    ) -> None:
        """TODO

        If apply_fn uses variadic args/kwargs, you must
        supply param_priorities.
        """
        self.apply_fn = apply_fn
        self.map_qubits_fn = map_qubits_fn

        if data is None:
            data = {}
        self.data = deepcopy(dict(data))

        # Introspect to ensure we set priorities for every arg needed
        if param_priorities is None:
            param_priorities = {}
        assert param_error_behavior in ["continue", "warn", "raise"]
        self.param_error_behavior = param_error_behavior

        self._param_priorities = {}
        sig = ins.signature(self.apply_fn)
        for key, param in sig.parameters.items():
            if param.kind != param.POSITIONAL_OR_KEYWORD:
                if self.param_error_behavior == "warn":
                    warnings.warn(f"Skipping param priority for {key}")
                elif self.param_error_behavior == "raise":
                    raise NotImplementedError(
                        f"Cannot handle param priority for {key}"
                    )
                continue

            self._param_priorities[key] = param_priorities.get(
                key, DEFAULT_PRIORITIES
            )

        # Go through and add any missing keys also
        for key, priorities in param_priorities.items():
            if key not in self._param_priorities:
                self._param_priorities[key] = priorities

        if param_aliases is None:
            param_aliases = {}
        self._param_aliases = dict(param_aliases)
        self._rev_param_aliases = {
            v: k for k, v in self._param_aliases.items()
        }

        self.name = name

    def __str__(self) -> str:
        s = f"Instruction {self.name}\n"
        sig = ins.signature(self.apply_fn)
        # All Instruction signatures end in Frame
        # Drop the return annotation
        sig._return_annotation = sig.empty  # type: ignore
        s += f"  Apply arguments: {sig}\n"
        s += "  Data:\n"
        for k, v in self.data.items():
            s += textwrap.indent(f"{k}: {v}", "    ")
            if not s.endswith("\n"):
                s += "\n"
        s += "  Non-default parameter priorities:"
        have_non_default = False
        for k, v in self.param_priorities.items():
            if v == DEFAULT_PRIORITIES:
                continue
            if not have_non_default:
                s += "\n"
            have_non_default = True
            s += textwrap.indent(f"{k}: {v}", "    ")
            if not s.endswith("\n"):
                s += "\n"
        if not have_non_default:
            s += " None (i.e. all defaults)\n"
        s += "  Parameter aliases:"
        if len(self._param_aliases):
            s += "\n"
            for k, v in self._param_aliases.items():
                s += textwrap.indent(f"{k}: {v}", "    ")
                if not s.endswith("\n"):
                    s += "\n"
        else:
            s += " None\n"
        return s

    @property
    def param_priorities(self) -> dict[str, Sequence[str]]:
        """TODO"""
        # Map priorities using aliases
        aliased_priorities = {
            self._param_aliases.get(k, k): v
            for k, v in self._param_priorities.items()
        }
        return aliased_priorities

    def apply(self, **kwargs) -> Frame:
        """TODO"""
        # Adjust aliases
        aliased_kwargs = {
            self._rev_param_aliases.get(k, k): v for k, v in kwargs.items()
        }

        applied_frame = self.apply_fn(**aliased_kwargs)

        output_frame = applied_frame.update(
            {
                "instruction": self,
                # "instruction_kwargs": aliased_kwargs,
            },
            f"{self.name} result",
        )
        return output_frame

    def copy(self) -> Instruction:
        return Instruction(
            apply_fn=self.apply_fn,
            data=deepcopy(self.data),
            map_qubits_fn=self.map_qubits_fn,
            param_priorities=self._param_priorities,
            param_error_behavior=self.param_error_behavior,  # type: ignore
            param_aliases=self._param_aliases,
            name=self.name,
        )

    def map_qubits(
        self: Instruction, qubit_mapping: Mapping[str, str]
    ) -> Instruction:
        """TODO"""
        new_instruction = self.copy()
        # Map qubits on all data
        new_kwargs = self.map_qubits_fn(qubit_mapping, **self.data)
        assert all(
            [k in new_kwargs for k in self.data]
        ), "map_qubits_fn did not output all expected keys"
        new_instruction.data = new_kwargs
        return new_instruction

    @classmethod
    def _from_serialization(cls: type[T], state: Mapping) -> T:
        apply_fn = None
        map_qubit_fn = None
        data = cls.deserialize(state["data"])
        assert isinstance(data, dict)
        param_error_behavior = state["param_error_behavior"]
        name = state["name"]

        obj = cls(apply_fn, data, map_qubit_fn, param_error_behavior, name)
        obj._param_priorities = state["_param_priorities"]
        obj._param_aliases = state["_param_aliases"]
        obj._rev_param_aliases = {v: k for k, v in obj._param_aliases.items()}

        return obj

    def _to_serialization(self) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "apply_fn": textwrap.dedent(ins.getsource(self.apply_fn)),
                "map_qubits_fn": textwrap.dedent(
                    ins.getsource(self.map_qubits_fn)
                ),
                "data": self.serialize(self.data),
                "param_error_behavior": self.param_error_behavior,
                "_param_priorities": self._param_priorities,
                "_param_aliases": self._param_aliases,
                "name": self.name,
            }
        )
        return state
