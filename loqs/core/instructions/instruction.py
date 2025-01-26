""":class:`.Instruction` definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import inspect as ins
import textwrap
from typing import (
    ClassVar,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
)
import warnings

from loqs.core import Frame
from loqs.internal import Displayable


T = TypeVar("T", bound="Instruction")
P = ParamSpec("P")

KwargDict: TypeAlias = dict[str, object]
"""A type alias for kwarg dicts (str keys, object values)."""


class ApplyCallable(Protocol[P]):
    """The protocol a user-defined apply function must follow.

    Specifically, it must return a :class:`.Frame`.
    """

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Frame: ...  # noqa


class MapQubitsCallable(Protocol[P]):
    """The protocol a user-defined map qubits function must follow.

    Specifically, it must take a qubit_mapping ``dict[str,str]`` as the
    the first argument, and return the mapped :attr:`.KwargDict`.
    """

    def __call__(  # noqa
        self,
        qubit_mapping: Mapping[str | int, str | int],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> KwargDict: ...


def default_map_qubits(qubit_mapping: Mapping[str | int, str | int], **kwargs):
    """A default map qubit function that does not change kwargs."""
    # Assume nothing needs to be mapped in kwargs
    return kwargs


DEFAULT_PRIORITIES = ["label", "instruction", "program", "history[-1]"]
"""Default parameter priority order."""


class Instruction(Displayable):
    """An object that moves the state of the simulation forward.

    This is the possibly the most important LoQS object.
    It was designed to be maximally flexible: it can take in any
    data it needs from the current state of the simulation,
    perform any transformation on that data, and output any
    information to be used by a downstream :class:`.Instruction`.

    NOTE: The :class:`Instruction` is flexible and powerful; however,
    with that flexibility comes complexity, and we are aware
    it may not be immediately clear how to use these. Interested users are
    encouraged to look at the Object Quickstart > Instructions and
    Tutorials > Building a Complex Instruction for more,
    or at :mod:`.builders` for concrete examples.

    At its core, an :class:`.Instruction` is defined by five
    pieces of user-defined information:

    - An apply function that takes in simulation information and
      outputs a new :class:`.Frame`
    - Data that is needed for the apply function but will
      not be provided by another source
    - A map qubits function that can change any
      :class:`.Instruction` data that has qubit labels in it
      (needed to make the apply function qubit/patch agnostic)
    - A set of parameter priorities for apply function input
      collection
    - A set of parameter aliases between apply function kwargs
      and what to look for during input collection

    The :class:`.Instruction` is then used in the following ways:

    - A :class:`QECCode` will define these with respect to a
      template set of qubits
    - The :class:`QECCodePatch` will use :meth:`.Instruction.map_qubits`
      to swap out the template qubits with the real ones (using
      the user-defined map qubits function)
    - The :class:`QuantumProgram` will use the data and parameter
      priorities/aliases to collect the right simulation information,
      and then call :meth:`.Instruction.apply` to generate the next
      :class:`.Frame` (using the user-defined apply function)

    NOTE: The :class:`Instruction` is annoying to serialize because it
    contains user-defined code. The way LoQS handles this is by
    storing the function definitions as strings for serialization,
    and re-executing them during deserialization. This has several
    important caveats:

    1. THIS HAS OBVIOUS SECURITY IMPLICATIONS. DO NOT DESERIALIZE
       INSTRUMENT-CONTAINING LOQS OBJECTS THAT YOU DO NOT TRUST.
       The good news is that because the function is stored in plain text,
       you can verify whether it is doing anything malicious.
    2. The serialized versions are computed at construction time
       and require access to the source code. They are then saved
       so that they persist through deserialization - otherwise,
       you could not re-serialize after deserialization because
       you would not have access to the source code of the executed
       function.
    3. As a side effect of the string versions of the functions
       being used for serialization, these are also the objects
       used when hashing and (potentially importantly) when doing
       equality testing. Two :class:`Instruction` objects can have
       functionally equivalently :attr:`.apply_fn` and :attr:`.map_qubits_fn`,
       but they will not test as equal if the string representations differ
       in any way. Similarly, two :class:`Instruction` objects that
       have very different functions would test as equal if one had
       serialized versions that were set to match with the other.
    4. Importantly for Jupyter users, Caveat 2 means that you may run
       into issues when your apply function is only defined in a
       notebook cell. There are two solutions to this: you can provide
       the plain text versions during object construction, or you
       can keep your function definitions in a separate script.
       The latter is preferred, but both should work.
    """

    CACHE_ON_SERIALIZE: ClassVar[bool] = True

    def __init__(
        self,
        apply_fn: ApplyCallable,
        data: Mapping[str, object] | None = None,
        map_qubits_fn: MapQubitsCallable = default_map_qubits,
        param_priorities: Mapping[str, Sequence[str]] | None = None,
        param_error_behavior: Literal["continue", "warn", "raise"] = "warn",
        param_aliases: Mapping[str, str] | None = None,
        serialized_apply_fn: str | None = None,
        serialized_map_qubits_fn: str | None = None,
        name: str = "(Unnamed instruction)",
        type: str = "User-defined",
    ) -> None:
        """
        Parameters
        ----------
        apply_fn:
            See :attr:`.apply_fn`

        data:
            See :attr:`.data`. Defaults to ``None``, which uses an empty ``dict``.

        map_qubits_fn:
            See :attr:`.map_qubits_fn`. Defaults to :meth:`.default_map_qubits`.

        param_priorities:
            A mapping of :attr:`.apply_fn` parameter names to lists of priorities
            to using during parameter collection with
            :meth:`.QuantumProgram._collect_kwarg`. Defaults to ``None``,
            which sets every parameter's priority to :attr:`.DEFAULT_PARAMETERS`.
            For an example, see :meth:`.builders.build_lookup_decoder_instruction`.

        param_error_behavior:
            See :attr:`.param_error_behavior`. Defaults to ``"warn"``.

        param_aliases:
            A mapping from `.apply_fn` parameter names to names to use during
            parameter collection with :meth:`.QuantumProgram._collect_kwarg`.
            For an example, see :meth:`.builders.build_lookup_decoder_instruction`.

        serialized_apply_fn:
            A serialized version of :attr:`.apply_fn`. Defaults to ``None``,
            which sets this by calling :meth:`.serialize` on :attr:`.apply_fn`.
            Not intended to be set by the user, see caveats above.

        serialized_map_qubits_fn:
            A serialized version of :attr:`.map_qubits_fn`. Defaults to ``None``,
            which sets this by calling :meth:`.serialize` on :attr:`.map_qubits_fn`.
            Not intended to be set by the user, see caveats above.

        name:
            See :attr:`.name`.

        type:
            See :attr:`.type`.
        """
        self.apply_fn = apply_fn
        """A user-defined function called in :meth:`.apply`.

        It must conform to the :attr:`.ApplyCallable` protocol.
        """

        self.map_qubits_fn = map_qubits_fn
        """A user-defined function called in :meth:`.map_qubits`.

        It must conform to the :attr:`.MapQubitsCallable` protocol.
        """

        # Let's serialize the functions now, when we know we have access to source code
        self._serialized_apply_fn = serialized_apply_fn
        if serialized_apply_fn is None:
            self._serialized_apply_fn = self._serialize_function(apply_fn)
        self._serialized_map_qubits_fn = serialized_map_qubits_fn
        if serialized_map_qubits_fn is None:
            self._serialized_map_qubits_fn = self._serialize_function(
                map_qubits_fn
            )

        if data is None:
            data = {}
        self.data = deepcopy(dict(data))
        """Data to keep with this :class:`.Instruction`.

        NOTE: There is currently a limitation that this data
        cannot store functions due to serialization issues.
        """

        # Introspect to ensure we set priorities for every arg needed
        if param_priorities is None:
            param_priorities = {}
        assert param_error_behavior in ["continue", "warn", "raise"]
        self.param_error_behavior = param_error_behavior
        """Error behaviour when processing :attr:`.apply_fn` parameters.

        Must be one of ``["continue", "warn", "raise"]``.
        """

        self._param_priorities = {}
        sig = ins.signature(self.apply_fn)
        for key, param in sig.parameters.items():
            if param.kind != param.POSITIONAL_OR_KEYWORD:
                if self.param_error_behavior == "warn" and key != "kwargs":
                    warnings.warn(f"Skipping param priority for {key}")
                elif self.param_error_behavior == "raise" and key != "kwargs":
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

        self.name = name
        """Name for logging"""

        self.type = type
        """Type for logging"""

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

    def __hash__(self) -> int:
        return hash(
            (
                self._serialized_apply_fn,
                self._serialized_map_qubits_fn,
                self.hash(self.data),
                self.hash(self._param_priorities),
                self.hash(self._param_aliases),
                self.name,
            )
        )

    @property
    def param_priorities(self) -> dict[str, Sequence[str]]:
        """The unaliased parameter priorities."""
        return self._param_priorities

    def param_alias(self, key: str) -> str:
        return self._param_aliases.get(key, key)

    def apply(self, **kwargs) -> Frame:
        """Apply this :class:`.Instruction` to get a new :class:`.Frame`.

        Parameters
        ----------
        **kwargs:
            Parameters to pass on to :attr:`.apply_fn`.

        Returns
        -------
        Frame
            The output :class:`.Frame` of :attr:`.apply_fn`, with this
            :class:`Instruction` and the input parameters appended for
            informational/debugging purposes
        """
        # Pull out only kwargs we need
        apply_kwargs = {k: kwargs[k] for k in self.param_priorities}

        applied_frame = self.apply_fn(**apply_kwargs)

        # TODO: Collected_params is a nice debugging feature here
        # It fails if the History is passed in though, so commenting out for now
        output_frame = applied_frame.update(
            {"instruction": self},  # "collected_params": apply_kwargs},
            new_log=f"{self.name} result",
        )

        return output_frame

    def copy(self) -> Instruction:
        """Return a copy of this :class:`.Instruction`."""
        return Instruction(
            apply_fn=self.apply_fn,
            data=deepcopy(self.data),
            map_qubits_fn=self.map_qubits_fn,
            param_priorities=self._param_priorities,
            param_error_behavior=self.param_error_behavior,  # type: ignore
            param_aliases=self._param_aliases,
            serialized_apply_fn=self._serialized_apply_fn,
            serialized_map_qubits_fn=self._serialized_map_qubits_fn,
            name=self.name,
            type=self.type,
        )

    def map_qubits(
        self, qubit_mapping: Mapping[str | int, str | int]
    ) -> Instruction:
        """Get a copy with mapped qubits.

        Parameters
        ----------
        qubit_mapping:
            The qubit mapping to apply, with old labels as keys
            and new labels as values

        Returns
        -------
        Instruction
            A copy of the :class:`Instruction` with mapped qubits
        """
        new_instruction = self.copy()
        # Map qubits on all data
        new_kwargs = self.map_qubits_fn(qubit_mapping, **self.data)
        assert all(
            [k in new_kwargs for k in self.data]
        ), "map_qubits_fn did not output all expected keys"
        new_instruction.data = new_kwargs
        return new_instruction

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        serialized_apply_fn = state["_serialized_apply_fn"]
        serialized_map_qubits_fn = state["_serialized_map_qubits_fn"]
        apply_fn = cls._deserialize_function(
            serialized_apply_fn,
        )
        map_qubits_fn = cls._deserialize_function(serialized_map_qubits_fn)
        data = cls.deserialize(state["data"], serial_id_to_obj_cache)
        assert isinstance(data, dict)
        param_error_behavior = state["param_error_behavior"]
        name = state["name"]
        type = state["type"]

        obj = cls(
            apply_fn,
            data,
            map_qubits_fn,
            param_error_behavior=param_error_behavior,
            serialized_apply_fn=serialized_apply_fn,
            serialized_map_qubits_fn=serialized_map_qubits_fn,
            name=name,
            type=type,
        )
        obj._param_priorities = state["_param_priorities"]
        obj._param_aliases = state["_param_aliases"]

        return obj

    def _to_serialization(
        self, hash_to_serial_id_cache=None, ignore_no_serialize_flags=False
    ) -> dict:
        state = super()._to_serialization()
        # Ordering here is to be nicer during display()
        state.update(
            {
                "name": self.name,
                "type": self.type,
                "data": self.serialize(
                    self.data,
                    hash_to_serial_id_cache,
                    ignore_no_serialize_flags,
                ),
                "param_error_behavior": self.param_error_behavior,
                "_param_priorities": self._param_priorities,
                "_param_aliases": self._param_aliases,
                "_serialized_apply_fn": self._serialized_apply_fn,
                "_serialized_map_qubits_fn": self._serialized_map_qubits_fn,
            }
        )
        return state
