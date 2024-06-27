"""TODO"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import textwrap
from typing import Literal, TypeAlias

from loqs.internal.castable import Castable

QueryIndicesTypes: TypeAlias = int | slice | Sequence[int] | Literal["all"]

IndexParamSources: TypeAlias = (
    Literal["history"] | Literal["default"] | Literal["label"]
)

InputParamInSpecType: TypeAlias = tuple[
    IndexParamSources, str | None, QueryIndicesTypes | None, str | None
]

InputSpecCastableTypes: TypeAlias = (
    "InputSpec | Sequence[InputParamInSpecType | None]"
)


class InputParam:
    """TODO"""

    def __init__(
        self,
        position: int,
        sources: IndexParamSources | Sequence[IndexParamSources],
        key: str | None = None,
        hist_idxs: QueryIndicesTypes | None = None,
        hist_key: str | None = None,
    ) -> None:
        """TODO"""
        if isinstance(sources, str):
            sources = [sources]
        assert all(
            [s in ["history", "default", "label"] for s in sources]
        ), "Sources must be specified from ['history', 'default', 'label']."

        self.position = position
        self.key = key
        self.sources = sources

        self.hist_key = None
        self.hist_idxs = None
        if "history" in sources:
            assert (
                hist_idxs is not None
            ), "If history is a source, hist_idxs must be provided"

            if hist_idxs == "all":
                hist_idxs = slice(None)
            assert isinstance(
                hist_idxs, (int, Sequence, slice)
            ), "hist_idxs must be int, slice, Sequence, or 'all'"

            if hist_key is None:
                assert self.key is not None, (
                    "If history is a source and no default/kwarg key,"
                    + " hist_key must be provided"
                )
                hist_key = self.key

            self.hist_key = hist_key
            self.hist_idxs = hist_idxs

    def __str__(self):
        s = f"InputParam({self.position},{self.sources},{self.key}"
        s += f",{self.hist_idxs},{self.hist_key})\n"
        return s


class InputSpec(Sequence[InputParam], Castable):
    """TODO"""

    _input_spec: Sequence[InputParam]
    """TODO"""

    def __init__(self, input_spec: InputSpecCastableTypes):
        if isinstance(input_spec, InputSpec):
            self._input_spec = input_spec._input_spec
        elif isinstance(input_spec, Sequence):
            none_allowed = True
            self._input_spec = []
            for param in input_spec:
                if param is None:
                    assert none_allowed, (
                        "None can only be used for sequential params "
                        + "at the start of the InputSpec input list"
                    )
                    # Shorthand for label-only positional arg
                    input_param = InputParam(len(self._input_spec), ["label"])
                else:
                    none_allowed = False

                    if isinstance(param, InputParam):
                        input_param = param
                    else:
                        input_param = InputParam(len(self._input_spec), *param)

                self._input_spec.append(input_param)

    def __getitem__(self, i: int):
        return self._input_spec[i]

    def __iter__(self) -> Iterator[InputParam]:
        return iter(self._input_spec)

    def __len__(self) -> int:
        return len(self._input_spec)

    def __str__(self):
        s = f"InputSpec with {len(self)} items:\n"
        for ip in self._input_spec:
            sip = str(ip)
            sip = textwrap.indent(sip, "  ")
            s += sip
        return s

    @property
    def keys(self) -> list[str]:
        keys = []
        for param in self._input_spec:
            if param.key is not None:
                keys.append(param.key)
        return keys
