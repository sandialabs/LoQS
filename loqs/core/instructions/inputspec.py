"""TODO"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import textwrap
from typing import Any, Literal, TypeAlias

from loqs.internal.castable import Castable

QueryIndicesTypes: TypeAlias = int | slice | Sequence[int] | Literal["all"]

IndexParamSources: TypeAlias = (
    Literal["history"] | Literal["default"] | Literal["label"]
)

InputParamCastableTypes: TypeAlias = (
    "InputParam | Mapping[str, Any] | tuple[str, IndexParamSources, QueryIndicesTypes | None, str | None]"
)


class InputParam:
    """TODO"""

    def __init__(
        self,
        kwargs_key: str,
        sources: IndexParamSources | Sequence[IndexParamSources],
        hist_idxs: QueryIndicesTypes | None = None,
        hist_key: str | None = None,
    ) -> None:
        """TODO"""
        if isinstance(sources, str):
            sources = [sources]
        assert all(
            [s in ["history", "default", "label"] for s in sources]
        ), "Sources must be specified from ['history', 'default', 'label']."

        self.kwargs_key = kwargs_key
        self.sources = sources

        if "history" in sources:
            assert hist_idxs is not None
            if isinstance(hist_idxs, int):
                hist_idxs = [hist_idxs]
            elif hist_idxs == "all":
                hist_idxs = slice(None)
            else:
                assert isinstance(
                    hist_idxs, (Sequence, slice)
                ), "hist_idxs must be int, slice, Sequence, or 'all'"

            if hist_key is None:
                hist_key = self.kwargs_key

            self.hist_key = hist_key
            self.hist_idxs = hist_idxs

    def __str__(self):
        s = "InputParam:\n"
        s += f"  kwargs_key: {self.kwargs_key}"
        s += f"  sources: {self.sources}"
        s += f"  hist_idxs: {self.hist_idxs}"
        s += f"  hist_key: {self.hist_key}"
        return s

    # This is one of the cases where we do not have a nice single argument cast
    # Override the cast method to accept a tuple of the first several
    @classmethod
    def cast(cls, obj: InputParamCastableTypes) -> InputParam:
        """Cast to a :class:`InputParam` object.

        Unlike most castable objects, :class:`InputParam`
        requires at least 2 inputs. This version of cast additionally allows
        a tuple/list variant for the arguments and disallows
        a single object being passed in.

        Parameters
        ----------
        obj:
            A castable object that is either:
            - Already a :class:`InputParam` object,
            in which case `obj` is returned
            - A kwarg dict that is passed into the constructor
            - A sequence of the first two to four arguments of the
            :class:`InputParam` constructor

        Returns
        -------
            A :class:`ObjectBuilder` object
        """
        if isinstance(obj, cls):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, dict):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)
        elif isinstance(obj, tuple):
            # Assume this is a tuple/list of first several args
            return cls(*obj)

        # Else we can't handle this
        raise ValueError(
            "InputParam requires multiple arguments to cast. "
            + "Use a tuple or kwarg dict when casting."
        )


InputSpecCastableTypes: TypeAlias = (
    "InputSpec | Sequence[InputParamCastableTypes]"
)


class InputSpec(Sequence[InputParam], Castable):
    """TODO"""

    _input_spec: Sequence[InputParam]
    """TODO"""

    def __init__(self, input_spec: Sequence[InputParamCastableTypes]):
        if isinstance(input_spec, InputSpec):
            self._input_spec = input_spec._input_spec
        elif isinstance(input_spec, Sequence):
            self._input_spec = [InputParam.cast(ip) for ip in input_spec]

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
    def kwarg_keys(self):
        return [param.kwargs_key for param in self._input_spec]
