#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""[](api:Serializable) definition."""

from __future__ import annotations

from dataclasses import dataclass
import contextvars
import functools
import gzip
import importlib
import re
import h5py
import numpy as np
from pathlib import Path
from io import TextIOBase
from typing import (
    IO,
    Any,
    Callable,
    ClassVar,
    Literal,
    Mapping,
    Type,
    TypeAlias,
    TypeVar,
)

from loqs.types import Bool, Complex, Float, Int, NDArray, SPSArray


class IncorrectDecodableTypeError(Exception):
    """Exception raised when an BaseEncoder function cannot handle an object.

    This is a recoverable error (to a point), signaling that a different
    [](api:BaseEncoder) function should be tried.
    """

    pass


class MisformedDecodableError(Exception):
    """Exception raised when an object is properly identified but misformed.

    This is not a recoverable error. The serialized object is misformed
    and cannot be loaded.
    """

    pass


class DecodableVersionError(Exception):
    """Exception raised when decoding a object with an unsupported version."""

    def __init__(
        self, msg="Version is not supported for decoding", *args, **kwargs
    ):
        super().__init__(msg, *args, **kwargs)


IMPORT_LOCATION_CHANGES_BY_VERSION: dict[
    int, dict[tuple[str, str], tuple[str, str] | None]
] = {
    # None means the name was deleted outright, not relocated -- decoding
    # then fails clearly. A version with no changes has no entry at all.
    1: {
        ("loqs.core.syndrome", "SyndromeLabel"): (
            "loqs.core.syndromelabel",
            "SyndromeLabel",
        ),
        # Module move only -- the name itself wasn't renamed to
        # SyndromeLabelLike until version 2 (see that entry below).
        ("loqs.core.syndrome", "SyndromeLabelCastableTypes"): (
            "loqs.core.syndromelabel",
            "SyndromeLabelCastableTypes",
        ),
        ("loqs.core.syndrome", "PauliFrame"): (
            "loqs.core.recordables.pauliframe",
            "PauliFrame",
        ),
    },
    2: {
        # `*CastableTypes` -> `*Like` renames: every one keeps the same
        # module, only the class name changes.
        ("loqs.core.instructions.instructionlabel", "InstructionLabelCastableTypes"): (
            "loqs.core.instructions.instructionlabel",
            "InstructionLabelLike",
        ),
        ("loqs.core.instructions.instructionstack", "InstructionStackCastableTypes"): (
            "loqs.core.instructions.instructionstack",
            "InstructionStackLike",
        ),
        ("loqs.core.syndromelabel", "SyndromeLabelCastableTypes"): (
            "loqs.core.syndromelabel",
            "SyndromeLabelLike",
        ),
        ("loqs.core.recordables.pauliframe", "PauliFrameCastableTypes"): (
            "loqs.core.recordables.pauliframe",
            "PauliFrameLike",
        ),
        # PatchDictCastableTypes -> PatchLayoutLike, not PatchDictLike:
        # loqs.core.recordables.patchdict's legacy shim exports
        # `PatchDict` only, with no `PatchDictLike` attribute at all.
        ("loqs.core.recordables.patchdict", "PatchDictCastableTypes"): (
            "loqs.core.recordables.patchlayout",
            "PatchLayoutLike",
        ),
        (
            "loqs.core.recordables.measurementoutcomes",
            "MeasurementOutcomesCastableTypes",
        ): (
            "loqs.core.recordables.measurementoutcomes",
            "MeasurementOutcomesLike",
        ),
        ("loqs.core.frame", "FrameCastableTypes"): (
            "loqs.core.frame",
            "FrameLike",
        ),
        ("loqs.core.history", "HistoryCastableTypes"): (
            "loqs.core.history",
            "HistoryLike",
        ),
        ("loqs.backends.model.pygstimodel", "PyGSTiModelCastableTypes"): (
            "loqs.backends.model.pygstimodel",
            "PyGSTiModelLike",
        ),
        ("loqs.backends.state.npsvstate", "NumpyStatevectorCastableTypes"): (
            "loqs.backends.state.npsvstate",
            "NumpyStatevectorLike",
        ),
        ("loqs.backends.state.qsimstate", "QSimStateCastableTypes"): (
            "loqs.backends.state.qsimstate",
            "QSimStateLike",
        ),
        ("loqs.backends.state.stimstate", "STIMStateCastableTypes"): (
            "loqs.backends.state.stimstate",
            "STIMStateLike",
        ),
        ("loqs.backends.circuit.pygsticircuit", "PyGSTiCircuitCastableTypes"): (
            "loqs.backends.circuit.pygsticircuit",
            "PyGSTiCircuitLike",
        ),
        ("loqs.backends.circuit.listcircuit", "ListCircuitCastableTypes"): (
            "loqs.backends.circuit.listcircuit",
            "ListCircuitLike",
        ),
        ("loqs.backends.circuit.stimcircuit", "STIMCircuitCastableTypes"): (
            "loqs.backends.circuit.stimcircuit",
            "STIMCircuitLike",
        ),
        # Deleted outright, not renamed: DictNoiseModel.__init__ dropped
        # this castable parameter rather than replacing it with a
        # same-shaped *Like type.
        ("loqs.backends.model.dictmodel", "DictModelCastableTypes"): None,
        # Castable/SeqCastable/MapCastable mixins, deleted outright; both
        # historical module paths (utils -> internal) are listed.
        ("loqs.internal.castable", "Castable"): None,
        ("loqs.internal.castable", "SeqCastable"): None,
        ("loqs.internal.castable", "MapCastable"): None,
        ("loqs.utils.castable", "Castable"): None,
        ("loqs.utils.castable", "SeqCastable"): None,
        ("loqs.utils.castable", "MapCastable"): None,
        # PatchDict -> PatchLayout: a strict superset (adds `relations`,
        # defaulting to {}), so decode redirects with no shim class needed
        # (see recordables/__init__.py for the construction-time shim).
        ("loqs.core.recordables.patchdict", "PatchDict"): (
            "loqs.core.recordables.patchlayout",
            "PatchLayout",
        ),
        # RepTuple: deleted outright, with no construction shim -- its old
        # (rep, qubits, reptype) constructor doesn't map onto any single
        # modern class (reptype dispatches across ~10 differently-shaped
        # concrete GateRep/InstrumentRep classes), and the far more common
        # failure mode (an old GateRep/InstrumentRep enum member passed as
        # reptype) already breaks on that attribute access before a
        # RepTuple shim could ever run anyway. Decode redirects to the
        # modern class instead.
        ("loqs.backends.reps", "RepTuple"): (
            "loqs.backends.reps.base",
            "OperationRep",
        ),
        # STIMDictNoiseModel: deleted outright, but does have a live
        # construction shim (loqs.backends.model.__init__) translating its
        # old (gate_dict, inst_dict) positional-tuple call shape onto
        # DictNoiseModel's own two separate positional arguments -- unlike
        # RepTuple above, loqs-migrate still can't blindly rewrite this
        # call (the shapes genuinely differ), but a live call keeps
        # working via the shim regardless.
        ("loqs.backends.model.stimdictmodel", "STIMDictNoiseModel"): (
            "loqs.backends.model.dictmodel",
            "DictNoiseModel",
        ),
    },
}  # (module, class) mapping from OLD to NEW locations for each version change

SERIALIZATION_VERSION = 2
"""Serialization versions.

0: First version. JSON encoding only, per-shot checkpointing only.
1: HDF5 encoding now available. Backwards compatible to version 0.
2: `*CastableTypes` -> `*Like` renames and several class removals/
   relocations; see IMPORT_LOCATION_CHANGES_BY_VERSION.
"""

# Module-level ContextVars below stand in for a "global-ish" flag set once
# at a top-level call and read deep inside the generic recursive decode
# dispatch, without threading a parameter through every layer in between.
#
# TODO: `ignore_no_serialize_flags` (encode-side; threaded explicitly
# through BaseEncoder/JSONEncoder/HDF5Encoder and every `_get_encoding_attr`
# override) looks like a good candidate for the same treatment -- every
# call site just relays it unchanged. Watch for `Serializable._serial_hash`,
# which deliberately wants `False` regardless of any ambient encode.

MIGRATE_LEGACY_FNS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "migrate_legacy_fns", default=False
)
"""Whether decoding may run a known legacy-construction pattern found in
an old `Instruction`'s frozen source, rather than raising a clear error.
Set for the duration of a `Serializable.read`/`.load` call, read once in
`Instruction._from_decoded_attrs`.
"""


@dataclass
class DeferredRef:
    """Helper class to keep track of deferred references."""

    cache_id: int


# Encoding types
EncodableArrays: TypeAlias = NDArray | SPSArray
EncodableIterables: TypeAlias = list | tuple | set
EncodablePrimitives: TypeAlias = (
    Int | Float | Bool | str | bytes | Complex | None
)
Encodable: TypeAlias = (
    "Serializable | EncodableIterables | dict | EncodableArrays | type | Callable | EncodablePrimitives"
)
Encoded: TypeAlias = dict | h5py.Group
EncodeFormats: TypeAlias = Literal["json", "json.gz", "hdf5", "h5"] | None
EncodeCache: TypeAlias = dict[int, list[tuple[int, int]]] | None
DecodeCache: TypeAlias = dict[int, "Serializable | DeferredRef"] | None


# Generic type variable to stand-in for derived class below
T = TypeVar("T", bound="Serializable")


@functools.lru_cache(maxsize=32)
def _import_usage_index_for_file(srcfile: str, mtime: float):
    """For every module-level import statement in a file, precompute a
    clean, standalone rendering of it plus a sorted list of every line
    elsewhere in the file that references a name it binds. Cached per
    `(srcfile, mtime)` so a file with many functions (e.g. a codepack
    with dozens of instructions) only needs its syntax tree walked once,
    not once per function checked; an edited-and-reloaded file is never
    served stale.

    Each import is rendered fresh from its own CST node rather than
    sliced from the original file text: a conditionally-guarded import
    (e.g. inside `if TYPE_CHECKING:` or a `try/except ImportError:`
    fallback) is a real module-scope binding by Python's own scoping
    rules, but its indented text isn't valid syntax once lifted out of
    its enclosing block on its own. This does lose the guard itself --
    a name only reachable via a `try/except ImportError:` fallback is
    reproduced as a plain, unconditional import, which could raise in an
    environment missing that optional dependency.
    """
    import libcst as cst
    from libcst.metadata import ImportAssignment, MetadataWrapper, PositionProvider, ScopeProvider

    with open(srcfile, "r") as f:
        file_src = f.read()
    wrapper = MetadataWrapper(cst.parse_module(file_src))

    # import node identity -> (its own start line, clean rendered source)
    import_info: dict[int, tuple[int, str]] = {}
    # import node identity -> sorted line numbers where it's referenced
    usage_lines: dict[int, list[int]] = {}

    class _Indexer(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider)

        def visit_Name(self, node: cst.Name) -> None:
            try:
                scope = self.get_metadata(ScopeProvider, node)
                assignments = scope[node.value]
            except KeyError:
                # No scope metadata at all (e.g. a keyword argument's own
                # name, not a real reference) or no resolution found
                # anywhere (e.g. a builtin with nothing bound).
                return
            for assignment in assignments:
                if not isinstance(assignment, ImportAssignment):
                    continue
                import_node = assignment.node
                key = id(import_node)
                if key not in import_info:
                    import_pos = self.get_metadata(
                        PositionProvider, import_node
                    )
                    rendered = cst.Module(
                        body=[cst.SimpleStatementLine(body=[import_node])]
                    ).code
                    import_info[key] = (import_pos.start.line, rendered)
                pos = self.get_metadata(PositionProvider, node)
                usage_lines.setdefault(key, []).append(pos.start.line)

    wrapper.visit(_Indexer())
    for lines in usage_lines.values():
        lines.sort()

    return import_info, usage_lines


class Serializable:
    """
    The base class for all serializable objects in LoQS.

    This class provides a unified serialization framework that supports both JSON
    and HDF5 formats. Derived classes must implement the abstract methods to
    define their serialization behavior.

    Key Features:

    - Support for both JSON and HDF5 serialization formats
    - Automatic object caching and reference tracking
    - Recursive serialization of complex nested structures
    - Format-agnostic API for easy switching between formats

    Derived classes should implement:

    - `_from_decoded_attrs()`: Create object from decoded attributes

    Example:
        >>> # Define a simple serializable class
        >>> class SimpleClass(Serializable):
        ...     _CACHE_ON_SERIALIZE = True
        ...     _SERIALIZE_ATTRS = ["name", "value"]
        ...
        ...     def __init__(self, name, value):
        ...         self.name = name
        ...         self.value = value
        ...
        ...
        ...     @classmethod
        ...     def _from_decoded_attrs(cls, attr_dict):
        ...         return cls(**attr_dict)

        >>> # Create and serialize an object
        >>> obj = SimpleClass("test", 42)
        >>> encoded = Serializable.encode(obj, format="json", reset_encode_id=True)
        >>> isinstance(encoded, dict)  # Should return True
        True
        >>> encoded["encode_type"]  # Should be 'Serializable'
        'Serializable'
    """

    @staticmethod
    def _serial_hash(obj: Any, _visited: set | None = None) -> int:
        """
        Generate a unique serial ID for an object based on its serializable content.

        This method recursively computes a hash of an object's serializable attributes,
        allowing objects with identical content to share the same serial ID even if
        they are different instances.

        Parameters
        ----------
        obj : Any
            The object to compute a serial ID for.
        _visited : set, optional
            Internal parameter to track visited objects and prevent circular references.

        Returns
        -------
        int
            A hash representing the object's serializable content.
        """
        if _visited is None:
            _visited = set()

        # Handle circular references by tracking object IDs
        obj_id = id(obj)
        if obj_id in _visited:
            # For circular references, use the object ID as a fallback
            # This ensures we don't get infinite recursion
            return hash(f"circular_ref_{obj_id}")

        _visited.add(obj_id)

        try:
            if isinstance(obj, Serializable):
                # For Serializable objects, hash the tuple of serial IDs of their _SERIALIZE_ATTRS
                attr_ids = []
                for attr in obj._SERIALIZE_ATTRS:
                    attr_value = obj._get_encoding_attr(attr)
                    attr_ids.append(
                        Serializable._serial_hash(attr_value, _visited)
                    )
                return hash(tuple(attr_ids))
            elif isinstance(obj, list):
                # For lists, hash the tuple of serial IDs of each element
                return hash(
                    tuple(
                        Serializable._serial_hash(item, _visited)
                        for item in obj
                    )
                )
            elif isinstance(obj, dict):
                # For dicts, hash the tuple of serial IDs of keys and values
                keys_id = Serializable._serial_hash(list(obj.keys()), _visited)
                values_id = Serializable._serial_hash(
                    list(obj.values()), _visited
                )
                return hash((keys_id, values_id))
            elif isinstance(obj, np.ndarray):
                # For numpy arrays, hash the shape and flattened data
                shape_id = Serializable._serial_hash(obj.shape, _visited)
                data_id = Serializable._serial_hash(
                    obj.flatten().tolist(), _visited
                )
                return hash((shape_id, data_id))
            else:
                # Base case: hash the object itself
                try:
                    return hash(obj)
                except TypeError:
                    # For unhashable objects, use their string representation
                    return hash(str(obj))
        finally:
            _visited.remove(obj_id)

    # Class attributes
    _CACHE_ON_SERIALIZE: ClassVar[bool] = False
    """Flag to indicate whether this class should be cached.

    Every Serializable object _can_ be cached, but caching does
    introduce some overhead. For cases where the serialized object
    is small or not frequently references, we can save time for very
    little filesize by not caching (the default behavior).

    Some large objects that are heavily referenced *should* use caching,
    however. Some examples: Instruction, InstructionStack, QECCode,
    QECCodePatch, any backend objects, etc.
    """

    _SERIALIZE_ATTRS: ClassVar[list[str]] = []
    """Attributes to serialize.

    If encoding requires a different access pattern
    than getattr(), derived classes should
    implement [](api:Serializable._get_encoding_attr).
    """

    _NO_COLLAPSE_ATTRS: ClassVar[frozenset[str]] = frozenset()
    """`_SERIALIZE_ATTRS` names that must always keep their own real HDF5
    group (and, one level further in, their own individually-addressable
    per-element groups if their value is a dict/list), never folded into
    HDF5's array-free-subtree collapse blob even when their content happens
    to have no array anywhere in it.

    Exists for attrs some other code depends on being able to navigate to
    and incrementally append into via raw HDF5 structure, bypassing the
    normal recursive decode entirely for speed -- collapsing would silently
    break that navigation whenever the attr's content happens to be
    array-free (e.g. `ProgramResults.shot_histories` for an all-classical
    program with no quantum state at all). Content nested more than one
    level below a listed attr is unaffected and still collapses normally.
    """

    _SERIALIZE_ATTRS_MAP: ClassVar[dict[str, str]] = {}
    """Attribute map to use in [](api:Serializable._from_decoded_attrs).

    Useful when internal (e.g. _<attr>) attributes are
    serialized, but they are named differently (e.g. <attr>)
    in class constructors. If decoding requires more complex
    state management than the class constructor, derived
    classes should implement [](api:Serializable._from_decoded_attrs).
    """

    ## ABSTRACT METHODS
    # Implement these in derived classes

    def _get_encoding_attr(
        self, attr: str, ignore_no_serialize_flags: bool = False
    ) -> Any:
        """
        Extract the attributes needed for encoding to a dictionary.

        By default, this assumes all requested attributes are available
        via getattr.
        This should be implemented in all [](api:Serializable)-derived classes
        that required objects for encoding where this is not true,
        e.g. state backends. This is also true for the Frame object,
        which may modify the underlying data depending
        on the `ignore_no_serialization` flag passed down.

        Parameters
        ----------
        attr:
            "Attribute" to retrieve

        Returns
        -------
        Any
            The "attribute" to be encoded in [](api:BaseEncoder.encode_uncached_obj).
        """
        return getattr(self, attr)

    @classmethod
    def _from_decoded_attrs(cls: Type[T], attr_dict: Mapping[str, Any]) -> T:
        """
        Create an object from decoded attributes dictionary.

        By default, this assumes that attributes are either directly named
        as constructor arguments, or at least are one of the arguments and
        thus can be remapped to the proper kwarg via _SERIALIZE_ATTRS_MAP.
        This should be implemented by all Serializable subclasses that for
        which the default behavior or mapping via _SERIALIZE_ATTRS_MAP is not
        sufficient to map decoded attributes to constructor arguments.

        Parameters
        ----------
        attr_dict : Mapping[str, Any]
            Dictionary of attribute names to their deserialized values.

        Returns
        -------
        object
            The reconstructed object.
        """
        # Filter out serialization metadata fields
        metadata_fields = {
            "encode_type",
            "module",
            "class",
            "version",
            "cache_type",
            "cache_id",
        }
        filtered_dict = {
            cls._SERIALIZE_ATTRS_MAP.get(k, k): v
            for k, v in attr_dict.items()
            if k not in metadata_fields
        }
        return cls(**filtered_dict)

    ## PUBLIC CLASS METHODS
    # Primarily for deserialization

    @classmethod
    def load(
        cls,
        f: IO[str] | TextIOBase | h5py.File,
        format: EncodeFormats = None,
        use_caching: bool = True,
        decode_cache: DecodeCache = None,
        migrate_legacy_fns: bool = False,
    ) -> Encodable:
        """
        Load an object of this type, or a subclass of this type, from an input stream.

        This method deserializes objects from both JSON and HDF5 formats,
        automatically handling object caching and reference resolution.

        Parameters
        ----------
        f : file-like or h5py.File
            An open input stream or HDF5 file to read from.

        format : {'json', 'json.gz', 'hdf5', 'h5'}, optional
            The format of the input stream data. If None, auto-detect from file type.
            - 'json': JSON text format
            - 'json.gz': Gzip-compressed JSON format
            - 'hdf5' or 'h5': HDF5 binary format

        migrate_legacy_fns : bool, optional
            If a decoded `Instruction`'s frozen source (from a file older
            than the current `SERIALIZATION_VERSION`) contains a known
            legacy-construction pattern (e.g. an old-style positional
            `InstructionLabel(...)` call), decoding normally raises a
            clear error rather than silently running it through a
            construction-time compatibility shim. Pass `True` to allow it
            to run as-is instead. Default is `False`.

        Returns
        -------
        Serializable
            The deserialized object of the appropriate class.
        """
        # Auto-detect format if not specified
        if format is None:
            if isinstance(f, h5py.File):
                format = "hdf5"
            elif isinstance(f, TextIOBase):
                format = "json"

        assert format is not None

        decode_cache = None
        if use_caching:
            decode_cache = decode_cache if decode_cache is not None else {}

        migrate_token = MIGRATE_LEGACY_FNS.set(migrate_legacy_fns)
        try:
            if format in ["json", "json.gz"]:
                # Check if it's a file-like object that supports text I/O
                assert isinstance(f, TextIOBase)

                import json

                state = json.load(f)
                assert isinstance(state, dict)

                decoded = Serializable.decode(
                    state, "json", decode_cache=decode_cache
                )
            elif format in ["hdf5", "h5"]:
                assert isinstance(f, h5py.File)

                root_group = f["root"]
                assert isinstance(root_group, h5py.Group)

                decoded = Serializable.decode(
                    root_group, "hdf5", decode_cache=decode_cache
                )
            else:
                raise ValueError(f"Invalid `format` value for load: {format}")
        finally:
            MIGRATE_LEGACY_FNS.reset(migrate_token)

        # At this point, at least outer object should not be a deferred reference
        assert not isinstance(decoded, DeferredRef)

        return decoded

    @classmethod
    def read(
        cls: Type[T],
        path: str | Path,
        format: EncodeFormats = None,
        use_caching: bool = True,
        decode_cache: DecodeCache = None,
        migrate_legacy_fns: bool = False,
    ) -> Encodable:
        """Read and deserialize an object from a file.

        Convenience method that combines file opening with deserialization.
        Automatically detects the serialization format from the file extension
        and delegates to the appropriate loading mechanism.

        Parameters
        ----------
        path : str or Path
            Path to the file containing the serialized object.
        format : EncodeFormats, optional
            The serialization format. If None, automatically detected from file extension.
            Supported extensions: .json, .json.gz, .h5, .hdf5.
        use_caching : bool, optional
            Whether to use object caching during deserialization. Default is True.
        decode_cache : DecodeCache, optional
            Existing decode cache to use for reference resolution.
        migrate_legacy_fns : bool, optional
            See [](api:Serializable.load). Default is `False`.

        Returns
        -------
        Encodable
            The deserialized object.

        Raises
        ------
        ValueError
            If the format cannot be determined from the file extension.
        """
        if format is None:
            if str(path).endswith(".json"):
                format = "json"
            elif str(path).endswith(".json.gz"):
                format = "json.gz"
            elif str(path).endswith(".h5") or str(path).endswith(".hdf5"):
                format = "hdf5"
            else:
                raise ValueError(
                    "Cannot determine format from extension of filename: %s"
                    % str(path)
                )

        if format == "json":
            f = open(str(path), "r")
        elif format == "json.gz":
            f = gzip.open(str(path), "rt")
            format = "json"
        elif format in ["hdf5", "h5"]:
            f = h5py.File(str(path), "r")
        else:
            raise ValueError("Cannot write format")

        loaded = cls.load(
            f,
            format,
            use_caching=use_caching,
            decode_cache=decode_cache,
            migrate_legacy_fns=migrate_legacy_fns,
        )

        f.close()

        return loaded

    ## PUBLIC INSTANCE FUNCTIONS
    # Primarily for serializing

    def dump(
        self,
        f: IO[str] | TextIOBase | h5py.File,
        format: EncodeFormats = None,
        use_caching: bool = True,
        encode_cache: EncodeCache = None,
        json_format_kwargs: Mapping | None = None,
    ) -> None:
        """
        Serializes and writes this object to a given output stream.

        This method provides the core serialization functionality that supports
        both JSON and HDF5 formats through a unified interface.

        Parameters
        ----------
        f : file-like or h5py.File
            A writable output stream or HDF5 file.

        format : {'json', 'hdf5', 'h5'}, optional
            The format to write. If None, auto-detect from file type.
            - 'json': JSON text format
            - 'hdf5' or 'h5': HDF5 binary format

        json_format_kwargs : dict, optional
            Additional arguments specific to the JSON format.
            For example, the JSON format accepts `indent` as an argument
            because `json.dump` does.

        Returns
        -------
        None
        """
        # Auto-detect format if not specified
        if format is None:
            if isinstance(f, h5py.File):
                format = "hdf5"
            elif isinstance(f, TextIOBase):
                format = "json"

        assert format is not None

        encode_cache = None
        if use_caching:
            encode_cache = encode_cache if encode_cache is not None else {}

        if format in ["json", "json.gz"]:
            # Check if it's a file-like object that supports text I/O
            assert isinstance(f, TextIOBase)

            if json_format_kwargs is None:
                json_format_kwargs = {}
            json_format_kwargs = dict(json_format_kwargs)

            # Compact by default; pass json_format_kwargs={"indent": 4} to opt
            # into human-readable pretty-printing instead.

            if "sort_keys" in json_format_kwargs:
                # Sorting keys will potentially break caching on deserialization,
                # so let's catch that here
                raise ValueError(
                    "Cannot use the 'sort_key' formatting option for caching reasons."
                )

            encoded = Serializable.encode(
                self, "json", encode_cache=encode_cache, reset_encode_id=True
            )

            import json

            json.dump(encoded, f, **json_format_kwargs)
        elif format in ["hdf5", "h5"]:
            assert isinstance(f, h5py.File)

            # track_order=True: see HDF5Encoder's own _create_group helper for why
            # every HDF5 group this codebase creates tracks link creation order.
            root_group = f.create_group("root", track_order=True)
            # The only "version" attribute written anywhere in the file --
            # see HDF5Encoder's `_HDF5_DECODE_VERSION` for why every other
            # node no longer repeats it.
            root_group.attrs["version"] = SERIALIZATION_VERSION
            Serializable.encode(
                self,
                "hdf5",
                encode_cache=encode_cache,
                reset_encode_id=True,
                h5_group=root_group,
            )
        else:
            raise ValueError(f"Invalid `format` value for dump: {format}")

    def write(
        self,
        path: str | Path,
        format: EncodeFormats = None,
        use_caching: bool = True,
        encode_cache: EncodeCache = None,
        json_format_kwargs: Mapping | None = None,
    ) -> None:
        """
        Writes this object to a file.

        Parameters
        ----------
        path : str or Path
            The name of the file that is written.

        format_kwargs : dict, optional
            Additional arguments specific to the format being used.
            For example, the JSON format accepts `indent` as an argument
            because `json.dump` does.

        Returns
        -------
        None
        """
        if format is None:
            if str(path).endswith(".json"):
                format = "json"
            elif str(path).endswith(".json.gz"):
                format = "json.gz"
            elif str(path).endswith(".h5") or str(path).endswith(".hdf5"):
                format = "hdf5"
            else:
                raise ValueError(
                    "Cannot determine format from extension of filename: %s"
                    % str(path)
                )

        if format == "json":
            f = open(str(path), "w")
        elif format == "json.gz":
            f = gzip.open(str(path), "wt")
        elif format in ["hdf5", "h5"]:
            f = h5py.File(str(path), "w")
        else:
            raise ValueError("Cannot write format")

        self.dump(
            f,
            format,
            use_caching=use_caching,
            encode_cache=encode_cache,
            json_format_kwargs=json_format_kwargs,
        )

        f.close()

    ## INTERNAL FUNCTIONS

    @staticmethod
    def encode(
        obj: Encodable,
        format: EncodeFormats = "hdf5",
        encode_cache: EncodeCache = None,
        ignore_no_serialize_flags: bool = False,
        reset_encode_id: bool = False,
        h5_group: h5py.Group | None = None,
    ):
        """
        Recursively encode an object to the specified format.

        This method handles the recursive serialization logic for both JSON and HDF5 formats.
        It serves as the entry point for the serialization process, automatically dispatching
        to the appropriate encoder based on the format parameter.

        Parameters
        ----------
        obj : Encodable
            The object to encode. Can be a Serializable object, primitive type,
            collection (dict, list, tuple, set), or numpy array.

        format : {'json', 'hdf5', 'h5'}, default: 'hdf5'
            The target serialization format.
            - 'json': Encode to JSON-compatible dictionary structure
            - 'hdf5' or 'h5': Encode to HDF5 group structure

        encode_cache : dict, optional
            Dictionary mapping object hashes to serialization IDs for caching.
            Enables object reference tracking and prevents duplicate serialization.

        ignore_no_serialize_flags : bool, optional
            Whether to ignore serialization flags and force serialization.

        reset_encode_id : bool, optional
            Whether to reset the global encode ID counter. Useful for starting
            a new serialization session.

        h5_group : h5py.Group, optional
            Required for HDF5 format. The HDF5 group to write the object to.

        Returns
        -------
        Encoded
            The encoded object in the appropriate format:
            - For JSON: dict with encode_type structure
            - For HDF5: h5py.Group with appropriate attributes

        Examples
        --------
        Basic encoding examples:

        >>> from tests.internal.test_serializable import MockSerializable
        >>> obj = MockSerializable(name="test", value=42)
        >>>
        >>> # JSON encoding produces a dictionary
        >>> encoded_json = Serializable.encode(obj, format="json", reset_encode_id=True)
        >>> isinstance(encoded_json, dict)  # Should return True
        True
        >>> encoded_json["encode_type"]  # Should be 'Serializable'
        'Serializable'
        """
        from loqs.internal.encoder import JSONEncoder, HDF5Encoder

        if format in ["json", "json.gz"]:
            encode_uncached_obj = functools.partial(
                JSONEncoder.encode_uncached_obj,
                encode_cache=encode_cache,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
            )
            encode_cached_obj = JSONEncoder.encode_cached_obj
            encode_iterable = functools.partial(
                JSONEncoder.encode_iterable,
                encode_cache=encode_cache,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
            )
            encode_dict = functools.partial(
                JSONEncoder.encode_dict,
                encode_cache=encode_cache,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
            )
            encode_array = JSONEncoder.encode_array
            encode_primitive = JSONEncoder.encode_primitive
            encode_class = JSONEncoder.encode_class
            encode_function = JSONEncoder.encode_function

            if reset_encode_id:
                JSONEncoder.ENCODE_ID = 0
        elif format in ["hdf5", "h5"]:
            assert (
                h5_group is not None
            ), "Cannot encode in HDF5 format without passing in h5_group"
            encode_uncached_obj = functools.partial(
                HDF5Encoder.encode_uncached_obj,
                encode_cache=encode_cache,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
                h5_group=h5_group,
            )
            encode_cached_obj = functools.partial(
                HDF5Encoder.encode_cached_obj, h5_group=h5_group
            )
            encode_iterable = functools.partial(
                HDF5Encoder.encode_iterable,
                encode_cache=encode_cache,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
                h5_group=h5_group,
            )
            encode_dict = functools.partial(
                HDF5Encoder.encode_dict,
                encode_cache=encode_cache,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
                h5_group=h5_group,
            )
            encode_array = functools.partial(
                HDF5Encoder.encode_array, h5_group=h5_group
            )
            encode_primitive = functools.partial(
                HDF5Encoder.encode_primitive, h5_group=h5_group
            )
            encode_class = functools.partial(
                HDF5Encoder.encode_class, h5_group=h5_group
            )
            encode_function = functools.partial(
                HDF5Encoder.encode_function, h5_group=h5_group
            )

            if reset_encode_id:
                HDF5Encoder.ENCODE_ID = 0
        else:
            raise ValueError("Invalid format for encoding")

        # Handle Serializable objects
        if isinstance(obj, Serializable):
            return Serializable._encode_Serializable(
                obj,
                format,
                encode_cache,
                encode_cached_obj,
                encode_uncached_obj,
            )

        # Handle dictionaries
        elif isinstance(obj, dict):
            return encode_dict(obj)

        # Handle NumPy arrays and SciPy sparse matrices
        elif isinstance(obj, EncodableArrays):
            return encode_array(obj)

        # Handle lists, tuples, sets
        elif isinstance(obj, EncodableIterables):
            return encode_iterable(obj)

        # Handle classes/types
        elif isinstance(obj, type):
            return encode_class(obj)

        # Handle callable functions
        elif callable(obj):
            return encode_function(obj)

        # Otherwise, assume we are a built-in serializable object
        elif isinstance(obj, EncodablePrimitives):
            return encode_primitive(obj)

        raise ValueError("Unknown type to encode")

    @staticmethod
    def decode(  # noqa: C901
        encoded: Encoded,
        format: EncodeFormats = "hdf5",
        decode_cache: DecodeCache = None,
    ) -> Encodable | DeferredRef:
        """
        Recursively decode a serialized object following the same pattern as encode.

        This method handles the recursive deserialization logic for both JSON and HDF5 formats.
        It automatically resolves object references, reconstructs complex nested structures,
        and handles all supported data types.

        Parameters
        ----------
        encoded : dict or h5py.Group
            The encoded object (either JSON dict or HDF5 group).
            - For JSON: Dictionary with 'encode_type' field
            - For HDF5: h5py.Group with appropriate attributes

        format : {'json', 'hdf5', 'h5'}, default: 'hdf5'
            The format of the encoded data.
            - 'json': Decode from JSON dictionary structure
            - 'hdf5' or 'h5': Decode from HDF5 group structure

        decode_cache : dict, optional
            Dictionary mapping serialization IDs to object instances for caching.
            Enables proper handling of object references and prevents duplicate
            deserialization.

        Returns
        -------
        Encodable
            The deserialized object. Can be a Serializable object, primitive type,
            collection (dict, list, tuple, set), or numpy array.
        """
        assert format is not None
        from loqs.internal.encoder import JSONEncoder, HDF5Encoder

        if decode_cache is None:
            decode_cache = {}

        # Determine format based on encoded type
        if format in ["json", "json.gz"]:
            # JSON format
            decode_cached_obj = functools.partial(
                JSONEncoder.decode_cached_obj, decode_cache=decode_cache
            )
            decode_uncached_obj = functools.partial(
                JSONEncoder.decode_uncached_obj, decode_cache=decode_cache
            )
            decode_iterable = functools.partial(
                JSONEncoder.decode_iterable, decode_cache=decode_cache
            )
            decode_dict = functools.partial(
                JSONEncoder.decode_dict, decode_cache=decode_cache
            )
            decode_array = functools.partial(JSONEncoder.decode_array)
            decode_primitive = functools.partial(JSONEncoder.decode_primitive)
            decode_class = functools.partial(JSONEncoder.decode_class)
            decode_function = functools.partial(JSONEncoder.decode_function)
        elif format in ["hdf5", "h5"]:
            # HDF5 format
            decode_cached_obj = functools.partial(
                HDF5Encoder.decode_cached_obj, decode_cache=decode_cache
            )
            decode_uncached_obj = functools.partial(
                HDF5Encoder.decode_uncached_obj, decode_cache=decode_cache
            )
            decode_iterable = functools.partial(
                HDF5Encoder.decode_iterable, decode_cache=decode_cache
            )
            decode_dict = functools.partial(
                HDF5Encoder.decode_dict, decode_cache=decode_cache
            )
            decode_array = functools.partial(HDF5Encoder.decode_array)
            decode_primitive = functools.partial(HDF5Encoder.decode_primitive)
            decode_class = functools.partial(HDF5Encoder.decode_class)
            decode_function = functools.partial(HDF5Encoder.decode_function)

            # For HDF5, check if root group
            try:
                return HDF5Encoder.decode_root_group(encoded, decode_cache)
            except IncorrectDecodableTypeError:
                pass
        else:
            raise ValueError("Invalid format for decoding")

        # Handle dicts
        try:
            return decode_dict(encoded)
        except IncorrectDecodableTypeError:
            pass

        # Handle matrix data
        try:
            return decode_array(encoded)
        except IncorrectDecodableTypeError:
            pass

        # Handle cached object references
        try:
            return decode_cached_obj(encoded)
        except IncorrectDecodableTypeError:
            pass

        # Handle class type
        try:
            return decode_class(encoded)
        except IncorrectDecodableTypeError:
            pass

        # Handle Serializable
        try:
            result = decode_uncached_obj(encoded)
            # Post-process to replace any placeholders with actual objects
            if decode_cache is not None:
                result = Serializable._replace_placeholders(
                    result, decode_cache
                )
            return result
        except IncorrectDecodableTypeError:
            pass

        # Handle lists/sets/tuples
        try:
            return decode_iterable(encoded)
        except IncorrectDecodableTypeError:
            pass

        # Handle function
        try:
            return decode_function(encoded)
        except IncorrectDecodableTypeError:
            pass

        try:
            return decode_primitive(encoded)
        except IncorrectDecodableTypeError:
            pass

        raise IncorrectDecodableTypeError("Unknown type to decode")

    @staticmethod
    def _prepare_function_source(
        src: str, version: int
    ) -> tuple[str, list]:
        """Run every source-level backwards-compatibility rewrite on a
        frozen function's source, in order: import location/rename
        updates, old-format `InstructionLabel(...)` construction
        rewriting, then any other one-off fixes.

        Returns the rewritten source alongside a list of
        `ManualReviewItem`s for anything the `InstructionLabel` rewrite
        found but couldn't confidently resolve -- callers decide what, if
        anything, to do with those (e.g. `Instruction._from_decoded_attrs`'s
        `MIGRATE_LEGACY_FNS` gate; `_eval_function_str`'s other callers
        ignore them).
        """
        updated_src = Serializable._update_imports(src, version)
        updated_src, manual_review = (
            Serializable._update_legacy_constructions(updated_src, version)
        )
        updated_src = Serializable._function_compatibility(
            updated_src, version
        )
        return updated_src, manual_review

    @staticmethod
    def _exec_function_str(
        updated_src: str, original_src: str | None = None
    ) -> Callable:
        """Exec already-prepared function source (see
        `_prepare_function_source`) and pull the defined function back out
        by name.

        `original_src` locates the function's own name, via its last
        `def ...(` line -- kept separate from `updated_src` since a
        rewrite pass could in principle touch everything but that line;
        defaults to `updated_src` itself. Also accepts an already-callable
        `updated_src` unchanged, for callers that skip preparation
        entirely when their input was never a string to begin with.
        """
        if callable(updated_src):
            return updated_src
        if original_src is None:
            original_src = updated_src

        # Evaluate function
        env: dict[str, Any] = {}
        exec(updated_src, env)

        # We need to find the function name
        # Search for last def, then first paren after it
        # Trim "def " and that should be the function name
        fn_defs = re.findall(r"^def .*\(", original_src, re.MULTILINE)
        last_fn_def = fn_defs[-1]
        key = last_fn_def[4:-1]

        # Pull the function out of the executed environment
        return env[key]

    @staticmethod
    def _eval_function_str(
        src: str, version: int = SERIALIZATION_VERSION
    ) -> Callable:
        """Evaluate a function from its source code string.

        Reconstructs a callable function from its source code, handling
        backwards compatibility issues and import updates for different
        serialization versions.

        Parameters
        ----------
        src : str
            The source code string containing the function definition.
        version : int, optional
            The serialization version of the function source. Used to apply
            appropriate backwards compatibility transformations.

        Returns
        -------
        callable
            The reconstructed function object.

        Raises
        ------
        Exception
            If the function source code cannot be evaluated or if the function
            name cannot be extracted.
        """
        # Backwards compatibility, it may have been evaluated already
        if callable(src):
            return src

        updated_src, _ = Serializable._prepare_function_source(src, version)
        return Serializable._exec_function_str(updated_src, src)

    @staticmethod
    def serialize_function(func: Callable) -> str:
        """Public entry point for pre-computing a function's serialized
        source, for a caller who needs to supply it explicitly (e.g. as
        `Instruction`'s `serialized_apply_fn=`/`serialized_map_qubits_fn=`)
        rather than relying on it being computed automatically later --
        most importantly, for a function whose source might not stay
        inspectable until then, such as one defined directly in a Jupyter
        notebook cell (see `Instruction`'s own docstring for why that
        matters)."""
        return Serializable._get_function_str(func)

    @staticmethod
    def _get_function_str(func):
        """Extract the source code and necessary imports for a function.

        Retrieves the complete source code of a function including its definition
        and any required import statements. This is used for serialization of
        callable functions.

        Parameters
        ----------
        func : callable
            The function to extract source code from.

        Returns
        -------
        str
            The complete source code string including function definition and
            necessary imports.

        Notes
        -----
        If the source file cannot be accessed or if import extraction fails,
        only the function definition is returned.
        """
        import inspect
        import textwrap

        # Get source code
        src = textwrap.dedent(inspect.getsource(func))

        # Also try to get imports
        srcfile = inspect.getsourcefile(func)
        if srcfile is None:
            # We'll fail to get imports, just return source
            return src

        try:
            imports = Serializable._imports_needed_by(func, srcfile)
        except Exception:
            # Best-effort improvement on top of the bare function body --
            # fall back silently rather than let a source snippet that
            # isn't really a standalone def (e.g. a lambda) break
            # serialization outright.
            return src

        return imports + src

    @staticmethod
    def _imports_needed_by(func, srcfile: str) -> str:
        """Every module-level import statement in `srcfile` that `func`'s
        own source actually references, found via libcst's real scope and
        reference resolution rather than text matching.

        Resolving real references (instead of checking whether an
        imported name's text merely occurs somewhere in the function's
        source) avoids two concrete failure modes of a text-based check:
        a false positive when the name only appears inside a comment,
        docstring, or unrelated identifier, and a false negative when a
        genuine usage is missed because the import spans multiple lines
        in a shape the text scan doesn't expect. `func`'s own line range
        (from `inspect`) is checked against `srcfile`'s precomputed
        per-import usage index (see `_import_usage_index_for_file`)
        without needing to separately locate `func`'s specific node by
        type or name -- which would need special-casing for a lambda, a
        decorated function, or any other shape `inspect` can report a
        line range for.
        """
        import bisect
        import inspect
        import os

        lines, start_line = inspect.getsourcelines(func)
        end_line = start_line + len(lines) - 1

        import_info, usage_lines = _import_usage_index_for_file(
            srcfile, os.stat(srcfile).st_mtime
        )

        needed = []
        for key, (import_line, rendered) in import_info.items():
            lines_used = usage_lines[key]
            i = bisect.bisect_left(lines_used, start_line)
            if i < len(lines_used) and lines_used[i] <= end_line:
                needed.append((import_line, rendered))
        needed.sort()

        return "".join(rendered for _, rendered in needed)

    @staticmethod
    def _import_class(module_name, class_name, version) -> Type:
        """Returns the class specified by the given state dictionary"""
        location_changes = (
            {}
            if version == SERIALIZATION_VERSION
            else Serializable._get_cumulative_changes(version)
        )

        if (module_name, class_name) in location_changes:
            new_location = location_changes[module_name, class_name]
            if new_location is None:
                raise ImportError(
                    f"{module_name}.{class_name} was removed in a later"
                    " serialization version and cannot be decoded."
                )
            module_name, class_name = new_location
        try:
            m = importlib.import_module(module_name)
            c = getattr(
                m, class_name
            )  # will raise AttributeError if class cannot be found
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(
                (
                    "Class or module not found when instantiating a Serializable"
                    f" {module_name}.{class_name} object!  If this class has"
                    " moved, consider adding (module, classname) mapping to"
                    " the loqs.internal.serializable.class_location_changes dict"
                )
            ) from e

        return c

    @staticmethod
    def _encode_Serializable(
        obj,
        format: str,
        encode_cache: EncodeCache,
        encode_cached_obj: Callable,
        encode_uncached_obj: Callable,
    ) -> Encoded:
        from loqs.internal.encoder import JSONEncoder, HDF5Encoder

        # Get serial ID for this object
        _serial_hash = Serializable._serial_hash(obj)
        object_id = id(obj)

        # Short-circuit on no cache behavior
        if encode_cache is None or not obj._CACHE_ON_SERIALIZE:
            return encode_uncached_obj(obj)

        # First check if this specific object instance is already being processed
        # This handles circular references within the same object graph
        if _serial_hash in encode_cache:
            cached_entries = encode_cache[_serial_hash]
            for entry in cached_entries:
                if entry[0] == object_id:
                    # This object is already being processed (circular reference)
                    # Create a reference to avoid infinite recursion
                    cache_id = entry[1]
                    return encode_cached_obj(
                        cache_id,
                        cache_type="reference",
                        reference_cache_id=cache_id,
                    )

        # Proceed with caching, look up id
        cache_id = (
            JSONEncoder.ENCODE_ID
            if format == "json"
            else HDF5Encoder.ENCODE_ID
        )

        # Increment encoder ID
        if format == "json":
            JSONEncoder.ENCODE_ID += 1
        else:
            HDF5Encoder.ENCODE_ID += 1

        # Check if _serial_hash exists in cache (different instances with same content)
        if _serial_hash in encode_cache:
            # Same serial content but different instance, create a copy
            # First entry in list is the source object
            source_cache_id = encode_cache[_serial_hash][0][1]

            # Add to cache (all other entries in list are copy objects)
            encode_cache[_serial_hash].append((object_id, cache_id))

            return encode_cached_obj(
                cache_id,
                cache_type="copy",
                reference_cache_id=source_cache_id,
                source_cache_id=cache_id,
            )

        # Otherwise, cache-miss so we create a new source
        encode_cache[_serial_hash] = [(object_id, cache_id)]

        # Encode as source
        result = encode_uncached_obj(obj)

        # Add cache info to result
        if format == "json":
            result.update({"cache_type": "source", "cache_id": cache_id})
        else:
            result.attrs["cache_type"] = "source"
            result.attrs["cache_id"] = cache_id

        return result

    @staticmethod
    def _update_imports(function_str, initial_version=None, loc_change=None):
        """
        Update Python import statements based on a dictionary of location changes.

        Args:
            function_str: String containing Python import statements
            initial_version: Version of function_str
            loc_change: Dictionary mapping (old_module, old_class) to (new_module, new_class)

        Returns:
            String with updated import statements, each on its own line
        """
        from loqs.tools.migrate.renames import rewrite_renames

        # Either provide initial version or the location change dict
        if loc_change is None:
            assert (
                initial_version is not None
            ), "Provide either initial_version (recommended) or loc_change (for testing)"
            if initial_version < SERIALIZATION_VERSION:
                loc_change = Serializable._get_cumulative_changes(
                    initial_version
                )
            else:
                assert (
                    initial_version == SERIALIZATION_VERSION
                ), f"Cannot handle serialization versions higher than {SERIALIZATION_VERSION}"
                loc_change = {}

        if not loc_change:
            return function_str

        return rewrite_renames(function_str, renames=loc_change).source

    @staticmethod
    def _update_legacy_constructions(
        function_str: str, version: int
    ) -> tuple[str, list]:
        """Rewrite any resolvable old-format `InstructionLabel(...)`
        construction in a frozen function's source, sharing
        `loqs.tools.migrate.labels`'s own detect/resolve/rewrite engine --
        a sibling pass to `_update_imports`, not a generalization of it,
        since renames are a pure text substitution while this is a real
        (if narrow) source rewrite.

        Returns `(function_str, [])` unchanged once `version >=
        SERIALIZATION_VERSION`. Otherwise returns the rewritten source
        alongside a list of `ManualReviewItem`s for anything found but not
        confidently rewritten -- the caller decides what, if anything, to
        do with those (e.g. `Instruction._from_decoded_attrs`'s
        `MIGRATE_LEGACY_FNS` gate).
        """
        if version >= SERIALIZATION_VERSION:
            return function_str, []
        # Deferred: loqs.tools ultimately imports Serializable itself
        # (e.g. via Instruction), so this can't be a top-level import.
        from loqs.tools.migrate.labels import migrate_instruction_labels

        try:
            result = migrate_instruction_labels(function_str)
        except Exception:
            # A frozen function's source isn't guaranteed to be a single,
            # standalone-parseable module (e.g. indentation relative to
            # its original enclosing scope) -- fall through unchanged
            # rather than let a best-effort rewrite break decoding.
            return function_str, []
        return result.source, result.manual_review

    @staticmethod
    def _replace_placeholders(obj, decode_cache):
        """Recursively replace circular reference objects with actual objects from decode_cache."""
        if obj is None:
            return obj

        # Check if this is a circular reference placeholder
        if isinstance(obj, DeferredRef):
            # This is a circular reference, replace it with the actual object
            actual_cache_id = obj.cache_id
            if actual_cache_id in decode_cache:
                actual_obj = decode_cache[actual_cache_id]
                # If the actual object is still a circular reference, keep it as is
                # (this can happen during the replacement process)
                if not isinstance(actual_obj, DeferredRef):
                    return actual_obj
            return obj

        # Handle Serializable objects
        if isinstance(obj, Serializable):
            # Recursively process attributes
            for attr in obj._SERIALIZE_ATTRS:
                if hasattr(obj, attr):
                    attr_value = getattr(obj, attr)
                    new_attr_value = Serializable._replace_placeholders(
                        attr_value, decode_cache
                    )
                    # Handle numpy array comparison
                    if hasattr(attr_value, "__array__") and hasattr(
                        new_attr_value, "__array__"
                    ):
                        # For numpy arrays, check if they are different arrays
                        if not np.array_equal(attr_value, new_attr_value):
                            setattr(obj, attr, new_attr_value)
                    elif new_attr_value != attr_value:
                        setattr(obj, attr, new_attr_value)
            return obj

        # Handle dictionaries
        elif isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                new_v = Serializable._replace_placeholders(v, decode_cache)
                new_dict[k] = new_v
            return new_dict

        # Handle lists, tuples, sets
        elif isinstance(obj, (list, tuple, set)):
            new_items = []
            for item in obj:
                new_item = Serializable._replace_placeholders(
                    item, decode_cache
                )
                new_items.append(new_item)

            if isinstance(obj, tuple):
                return tuple(new_items)
            elif isinstance(obj, set):
                return set(new_items)
            else:
                return new_items

        # Handle other types (primitives, arrays, etc.)
        else:
            return obj

    @staticmethod
    def _get_cumulative_changes(initial_version):
        assert initial_version < SERIALIZATION_VERSION

        # `.get(..., {})`: a version with no import-location changes has no
        # entry at all here, not an empty one -- must not raise KeyError.
        complete_location_changes = IMPORT_LOCATION_CHANGES_BY_VERSION.get(
            initial_version + 1, {}
        ).copy()

        # Compose multi-hop renames across every later version (A -> B,
        # B -> C becomes A -> C).
        for version in range(initial_version + 2, SERIALIZATION_VERSION + 1):
            for new_k, new_v in IMPORT_LOCATION_CHANGES_BY_VERSION.get(
                version, {}
            ).items():
                updated_map = False

                # If new_k corresponds to a value in the current location changes,
                # it needs to be remapped. i.e. we need to handle A -> B, B-> C = A->C
                for k, v in complete_location_changes.items():
                    if v == new_k:
                        complete_location_changes[k] = new_v
                        updated_map = True

                if not updated_map:
                    # We don't collide with any existing mappings, add it in
                    complete_location_changes[new_k] = new_v

        return complete_location_changes

    @staticmethod
    def _function_compatibility(src, version):
        """Other known backwards-compatibility fixes"""
        if version == 0:
            # Physical circuit instructions used _stim_available, which is now is_backend_available("stim")
            if "_stim_available" in src:
                src = (
                    "from loqs.backends import is_backend_available\n"
                    + src.replace(
                        "_stim_available", 'is_backend_available("stim")'
                    )
                )

        return src
