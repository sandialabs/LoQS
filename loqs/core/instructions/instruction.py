"""TODO
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import inspect as ins
import textwrap
from typing import Literal, ParamSpec, Protocol, TypeAlias, runtime_checkable
import warnings

from loqs.core import Frame


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


class Instruction:
    """TODO"""

    def __init__(
        self,
        apply_fn: ApplyCallable,
        dry_run_apply_fn: ApplyCallable | Sequence[str],
        map_qubits_fn: MapQubitsCallable = default_map_qubits,
        data: Mapping[str, object] | None = None,
        param_priorities: Mapping[str, Sequence[str]] | None = None,
        param_error_behavior: Literal["continue", "warn", "raise"] = "warn",
        param_aliases: Mapping[str, str] | None = None,
        name: str = "(Unnamed instruction)",
        parent: (
            object | None
        ) = None,  # TODO: Let be Instruction or InstructionStack
        fault_tolerant: bool | None = None,
    ) -> None:
        """TODO

        If apply_fn uses variadic args/kwargs, you must
        supply param_priorities.
        """
        self.apply_fn = apply_fn

        if isinstance(dry_run_apply_fn, Sequence):
            frame_keys = list(dry_run_apply_fn)
            assert all([isinstance(k, str) for k in frame_keys])

            def default_dry_run_apply_fn(**kwargs):
                return Frame({k: "DRY_RUN" for k in frame_keys})

            dry_run_apply_fn = default_dry_run_apply_fn
        self.dry_run_apply_fn = dry_run_apply_fn

        self.map_qubits_fn = map_qubits_fn

        if data is None:
            data = {}
        self.data = dict(deepcopy(data))

        # Introspect to ensure we set priorities for every arg needed
        if param_priorities is None:
            param_priorities = {}
        self._param_priorities = dict(param_priorities)

        assert param_error_behavior in ["continue", "warn", "raise"]
        sig = ins.signature(self.apply_fn)
        for key, param in sig.parameters.items():
            if param.kind != param.POSITIONAL_OR_KEYWORD:
                if param_error_behavior == "warn":
                    warnings.warn(f"Skipping param priority for {key}")
                elif param_error_behavior == "raise":
                    raise NotImplementedError(
                        f"Cannot handle param priority for {key}"
                    )
                continue

            if key in self._param_priorities:
                # We have already set this
                continue

            # We have been provided no info, but this is a required parameter
            # Use the default parameter priority
            self._param_priorities[key] = DEFAULT_PRIORITIES

        if param_aliases is None:
            param_aliases = {}
        self._param_aliases = dict(param_aliases)

        self.name = name
        self.parent = parent
        self.fault_tolerant = fault_tolerant

    def __str__(self) -> str:
        s = f"Instruction {self.name}\n"
        s += f"  Apply signature:{ins.signature(self.apply_fn)}\n"
        s += "  Data:\n"
        for k, v in self.data.items():
            s += textwrap.indent(f"{k}: {v}", "    ")
            if not s.endswith("\n"):
                s += "\n"
        s += f"  Fault-tolerant: {self.fault_tolerant}\n"
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

    def apply(self, dry_run: bool = False, **kwargs) -> Frame:
        """TODO"""
        if dry_run:
            applied_frame = self.dry_run_apply_fn(**kwargs)
        else:
            applied_frame = self.apply_fn(**kwargs)

        output_frame = applied_frame.update(
            {
                "instruction": self,
                "instruction_kwargs": kwargs,
            },
            f"{self.name} result",
        )
        return output_frame

    def copy(self) -> Instruction:
        return Instruction(
            self.apply_fn,
            self.dry_run_apply_fn,
            self.map_qubits_fn,
            deepcopy(self.data),
            self._param_priorities,
            "warn",  # Should not warn unless something weird happens
            self._param_aliases,
            self.name,
            self.parent,
            self.fault_tolerant,
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
