#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
from loqs.internal.serializable import (
    MIGRATE_LEGACY_FNS,
    SERIALIZATION_VERSION,
    Serializable,
)

T = TypeVar("T", bound="Instruction")
P = ParamSpec("P")

KwargDict: TypeAlias = dict[str, object]
"""A type alias for kwarg dicts (str keys, object values)."""


class ApplyCallable(Protocol[P]):
    """The protocol a user-defined apply function must follow.

    Specifically, it must return a [](api:Frame).
    """

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Frame: ...  # noqa


class MapQubitsCallable(Protocol[P]):
    """The protocol a user-defined map qubits function must follow.

    Specifically, it must take a qubit_mapping `dict[str,str]` as the
    the first argument, and return the mapped [](api:KwargDict).
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


DEFAULT_PRIORITIES = [
    "label",
    "instruction",
    "patch_data",
    "program",
    "history[-1]",
]
"""Default parameter priority order."""


class Instruction(Displayable):
    """An object that moves the state of the simulation forward.

    This is the possibly the most important `LoQS` object.
    It was designed to be maximally flexible: it can take in any
    data it needs from the current state of the simulation,
    perform any transformation on that data, and output any
    information to be used by a downstream [](api:Instruction).

    NOTE: The [](api:Instruction) is flexible and powerful; however,
    with that flexibility comes complexity, and we are aware
    it may not be immediately clear how to use these. Interested users are
    encouraged to look at the Object Quickstart > Instructions and
    Tutorials > Building a Complex Instruction for more,
    or at [](api:builders) for concrete examples.

    At its core, an [](api:Instruction) is defined by five
    pieces of user-defined information:

    - An apply function that takes in simulation information and
      outputs a new [](api:Frame)
    - Data that is needed for the apply function but will
      not be provided by another source
    - A map qubits function that can change any
      [](api:Instruction) data that has qubit labels in it
      (needed to make the apply function qubit/patch agnostic)
    - A set of parameter priorities for apply function input
      collection
    - A set of parameter aliases between apply function kwargs
      and what to look for during input collection

    The [](api:Instruction) is then used in the following ways:

    - A [](api:QECCode) will define these with respect to a
      template set of qubits
    - The [](api:QECCodePatch) will use [](api:Instruction.map_qubits)
      to swap out the template qubits with the real ones (using
      the user-defined map qubits function)
    - The [](api:QuantumProgram) will use the data and parameter
      priorities/aliases to collect the right simulation information,
      and then call [](api:Instruction.apply) to generate the next
      [](api:Frame) (using the user-defined apply function)

    NOTE: The [](api:Instruction) is annoying to serialize because it
    contains user-defined code. The way `LoQS` handles this is by
    storing the function definitions as strings for serialization,
    and re-executing them during deserialization. This has several
    important caveats:

    1. THIS HAS OBVIOUS SECURITY IMPLICATIONS. DO NOT DESERIALIZE
       INSTRUCTION-CONTAINING `LoQS` OBJECTS THAT YOU DO NOT TRUST.
       The good news is that because the function is stored in plain text,
       you can verify whether it is doing anything malicious.
    2. The serialized versions are computed lazily, the first time
       they're actually needed (e.g. at encode time), and cached from
       then on. This requires access to the source code at that point,
       not necessarily at construction time -- see Caveat 4 for the
       biggest practical risk this poses. Once computed, the cached
       version persists through deserialization, so re-serializing
       after deserialization still works even without access to the
       original source code.
    3. As a side effect of the string versions of the functions
       being used for serialization, these are also the objects
       used when hashing and (potentially importantly) when doing
       equality testing. Two [](api:Instruction) objects can have
       functionally equivalently [](api:apply_fn) and [](api:map_qubits_fn),
       but they will not test as equal if the string representations differ
       in any way. Similarly, two [](api:Instruction) objects that
       have very different functions would test as equal if one had
       serialized versions that were set to match with the other.
    4. Importantly for Jupyter users: a notebook cell's function is only
       inspectable while its defining kernel session is alive (via
       IPython's own linecache patch, not a real file on disk), so if
       this Instruction isn't serialized until sometime after a kernel
       restart, Caveat 2's lazy computation fails then instead. The
       constructor warns immediately if this looks likely. The safest
       fix is to keep function definitions in a separate script;
       alternatively, pass an already-serialized version explicitly
       (via [](api:Serializable.serialize_function)) at construction
       time, while the live function is still available.

    Examples
    --------
    >>> from loqs.core import Instruction, Frame
    >>> def my_apply(state):
    ...     return Frame({"state": state})
    >>> inst = Instruction(name="MyInstruction", apply_fn=my_apply, serialized_apply_fn="my_apply")
    >>> inst.name
    'MyInstruction'
    """

    _CACHE_ON_SERIALIZE: ClassVar[bool] = True

    _SERIALIZE_ATTRS = [
        "name",
        "type",
        "data",
        "param_error_behavior",
        "_param_priorities",
        "_param_aliases",
        "_serialized_apply_fn",
        "_serialized_map_qubits_fn",
    ]

    data: dict
    """Data to keep with this [](api:Instruction).

    NOTE: There is currently a limitation that this data
    cannot store functions due to serialization issues.
    """

    apply_fn: ApplyCallable
    """A user-defined function called in [](api:Instruction.apply).

    It must conform to the [](api:ApplyCallable) protocol.
    """

    map_qubits_fn: MapQubitsCallable | None
    """A user-defined function called in [](api:Instruction.map_qubits).

    It must conform to the [MapQubitsCallable](api:MapQubitsCallable] protocol.
    """

    param_error_behavior: Literal["continue", "warn", "raise"]
    """Error behaviour when processing [](api:Instruction.apply_fn) parameters.
    """

    name: str
    """Name for logging"""

    type: str
    """Type for logging"""

    @staticmethod
    def _warn_if_source_unavailable(func: Callable, param_name: str) -> None:
        """Warn immediately if `func`'s source looks unlikely to still be
        there when actually needed later (serialization is deferred to
        first real use, e.g. encode time, rather than done eagerly here).

        Covers both a callable `inspect.getsource` flatly can't handle
        right now, and the classic Jupyter notebook case: a cell's
        function is inspectable *now* only via IPython's linecache patch
        for a backing file that doesn't really exist on disk, and stops
        working after a kernel restart.
        """
        import inspect
        import os

        try:
            inspect.getsource(func)
            srcfile = inspect.getsourcefile(func)
            unavailable = srcfile is not None and not os.path.exists(
                srcfile
            )
        except (OSError, TypeError):
            unavailable = True

        if unavailable:
            warnings.warn(
                f"Source for '{param_name}' may not be available later "
                "(e.g. a Jupyter cell, lambda, or exec'd function). Pass "
                f"serialized_{param_name}="
                f"Serializable.serialize_function({param_name}) now to "
                "avoid a failure at encode time. Ignore if you do not "
                "plan to serialize/encode this Instruction.",
                stacklevel=3,
            )

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
            See [](api:apply_fn)

        data:
            See [](api:data). Defaults to `None`, which uses an empty `dict`.

        map_qubits_fn:
            See [](api:map_qubits_fn). Defaults to [](api:default_map_qubits).

        param_priorities:
            A mapping of [](api:apply_fn) parameter names to lists of priorities
            to using during parameter collection with
            [](api:QuantumProgram._collect_kwarg). Defaults to `None`,
            which sets every parameter's priority to [](api:DEFAULT_PARAMETERS).
            For an example, see [](api:builders.build_lookup_decoder_instruction).

        param_error_behavior:
            See [](api:param_error_behavior). Defaults to `"warn"`.

        param_aliases:
            A mapping from `.apply_fn` parameter names to names to use during
            parameter collection with [](api:QuantumProgram._collect_kwarg).
            For an example, see [](api:builders.build_lookup_decoder_instruction).

        serialized_apply_fn:
            A serialized version of [](api:apply_fn). Defaults to `None`,
            which sets this by calling [](api:serialize) on [](api:apply_fn).
            Not intended to be set by the user, see caveats above.

        serialized_map_qubits_fn:
            A serialized version of [](api:map_qubits_fn). Defaults to `None`,
            which sets this by calling [](api:serialize) on [](api:map_qubits_fn).
            Not intended to be set by the user, see caveats above.

        name:
            See [](api:name).

        type:
            See [](api:type).
        """
        self.apply_fn = apply_fn

        self.map_qubits_fn = map_qubits_fn

        # Deferred to first actual use (see the _serialized_apply_fn/
        # _serialized_map_qubits_fn properties below) rather than computed
        # eagerly here -- apply_fn/map_qubits_fn are never reassigned after
        # construction, so this is always safe to defer, and most
        # instructions constructed during a simulation are never actually
        # serialized at all. Still checked (cheaply) right now, though: if
        # source access is already failing, it's much more useful to warn
        # immediately, while the live callable and a chance to work around
        # it are both still at hand, than to find out much later at
        # encode time.
        self._serialized_apply_fn_cache = serialized_apply_fn
        if serialized_apply_fn is None:
            self._warn_if_source_unavailable(apply_fn, "apply_fn")
        self._serialized_map_qubits_fn_cache = serialized_map_qubits_fn
        if serialized_map_qubits_fn is None:
            self._warn_if_source_unavailable(
                map_qubits_fn, "map_qubits_fn"
            )

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

        self.type = type

    @property
    def _serialized_apply_fn(self) -> str | ApplyCallable:
        """Serialized `apply_fn`, computed via
        `Serializable._get_function_str` on first access and cached from
        then on -- see the deferred-computation note in `__init__`."""
        if self._serialized_apply_fn_cache is None:
            self._serialized_apply_fn_cache = (
                Serializable._get_function_str(self.apply_fn)
            )
        return self._serialized_apply_fn_cache

    @property
    def _serialized_map_qubits_fn(self) -> str | MapQubitsCallable:
        """Serialized `map_qubits_fn` -- see `_serialized_apply_fn`."""
        if self._serialized_map_qubits_fn_cache is None:
            self._serialized_map_qubits_fn_cache = (
                Serializable._get_function_str(self.map_qubits_fn)
            )
        return self._serialized_map_qubits_fn_cache

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
        """The unaliased parameter priorities."""
        return self._param_priorities

    def param_alias(self, key: str) -> str:
        """Get the parameter alias for a given key.

        Parameters
        ----------
        key : str
            The parameter key to look up.

        Returns
        -------
        str
            The aliased parameter name, or the original key if no alias exists.
        """
        return self._param_aliases.get(key, key)

    def apply(self, **kwargs) -> Frame:
        """Apply this [](api:Instruction) to get a new [](api:Frame).

        Parameters
        ----------
        **kwargs:
            Parameters to pass on to [](api:Instruction.apply_fn).

        Returns
        -------
        Frame
            The output [](api:Frame) of [](api:Instruction.apply_fn), with this
            [](api:Instruction) and the input parameters appended for
            informational/debugging purposes
        """
        # Pull out only kwargs we need; params omitted at collection time
        # (see param_error_behavior "continue") stay omitted so the
        # apply_fn's own defaults apply
        apply_kwargs = {
            k: kwargs[k] for k in self.param_priorities if k in kwargs
        }

        applied_frame = self.apply_fn(**apply_kwargs)

        # TODO: Collected_params is a nice debugging feature here
        # It fails if the History is passed in though, so commenting out for now
        output_frame = applied_frame.update(
            {"instruction": self},  # "collected_params": apply_kwargs},
            new_log=f"{self.name} result",
        )

        return output_frame

    def copy(self) -> Instruction:
        """Return a copy of this [](api:Instruction)."""
        return Instruction(
            apply_fn=self.apply_fn,
            data=deepcopy(self.data),
            map_qubits_fn=self.map_qubits_fn,
            param_priorities=self._param_priorities,
            param_error_behavior=self.param_error_behavior,  # type: ignore
            param_aliases=self._param_aliases,
            # The raw (possibly not-yet-computed) cache, not the
            # property, so copying an instruction that's never been
            # serialized (e.g. via map_qubits, a hot path when mapping
            # template instructions onto a real patch) doesn't force
            # that computation just because it was copied.
            serialized_apply_fn=self._serialized_apply_fn_cache,
            serialized_map_qubits_fn=self._serialized_map_qubits_fn_cache,
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
            A copy of the [](api:Instruction) with mapped qubits
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
    def _from_decoded_attrs(cls, attr_dict) -> "Instruction":
        """Create an Instruction from decoded attributes dictionary."""
        # Deserialize functions
        serialized_apply_fn = attr_dict["_serialized_apply_fn"]
        serialized_map_qubits_fn = attr_dict["_serialized_map_qubits_fn"]
        version = attr_dict["version"]

        # Prepare source once (imports, then a confident rewrite of any
        # resolvable old-format InstructionLabel(...) call) and gate
        # execution on whatever's left unresolved, rather than a separate
        # detect-then-rewrite-then-detect-again pass over the same
        # source. Only a string needs preparing -- a version-0 file's
        # source may already be a live callable by this point
        # (decode_function's own version-0 heuristic), leaving nothing
        # to prepare.
        unresolved = []
        if version < SERIALIZATION_VERSION:
            if isinstance(serialized_apply_fn, str):
                serialized_apply_fn, review = (
                    Serializable._prepare_function_source(
                        serialized_apply_fn, version
                    )
                )
                unresolved += review
            if isinstance(serialized_map_qubits_fn, str):
                serialized_map_qubits_fn, review = (
                    Serializable._prepare_function_source(
                        serialized_map_qubits_fn, version
                    )
                )
                unresolved += review

        # Gate re-executing any remaining old, now-incompatible calling
        # convention behind an explicit opt-in rather than doing it
        # silently.
        if unresolved and not MIGRATE_LEGACY_FNS.get():
            details = "; ".join(str(item) for item in unresolved)
            raise RuntimeError(
                f"Instruction {attr_dict.get('name')!r}'s frozen source "
                f"(serialized at version {version}) appears to construct "
                f"an unresolvable legacy InstructionLabel(...): {details}. "
                "Pass migrate_legacy_fns=True to QuantumProgram.read/"
                "Serializable.load to run it as-is, an "
                "instruction_registry to resolve it automatically if "
                "possible, or migrate the source with loqs-migrate "
                "first."
            )

        apply_fn = Serializable._exec_function_str(
            serialized_apply_fn, attr_dict["_serialized_apply_fn"]
        )
        map_qubits_fn = Serializable._exec_function_str(
            serialized_map_qubits_fn, attr_dict["_serialized_map_qubits_fn"]
        )

        # Create instruction
        instruction_type = attr_dict["type"]
        obj = cls(
            apply_fn,
            attr_dict["data"],
            map_qubits_fn,
            param_error_behavior=attr_dict["param_error_behavior"],
            serialized_apply_fn=serialized_apply_fn,
            serialized_map_qubits_fn=serialized_map_qubits_fn,
            name=attr_dict["name"],
            type=instruction_type,
        )
        obj._param_priorities = attr_dict["_param_priorities"]
        obj._param_aliases = attr_dict["_param_aliases"]

        return obj
