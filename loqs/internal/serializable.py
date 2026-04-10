#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

""":class:`.Serializable` definition.
"""

from __future__ import annotations

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
    Union,
)

from loqs.types import Bool, Complex, Float, Int, NDArray, SPSArray


class IncorrectDecodableTypeError(Exception):
    """Exception raised when an BaseEncoder function cannot handle an object.

    This is a recoverable error (to a point), signaling that a different
    :class:`.BaseEncoder` function should be tried.
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
    int, dict[tuple[str, str], tuple[str, str]]
] = {
    1: {
        ("loqs.core.syndrome", "SyndromeLabel"): (
            "loqs.core.syndromelabel",
            "SyndromeLabel",
        ),
        ("loqs.core.syndrome", "SyndromeLabelCastableTypes"): (
            "loqs.core.syndromelabel",
            "SyndromeLabelCastableTypes",
        ),
        ("loqs.core.syndrome", "PauliFrame"): (
            "loqs.core.recordables.pauliframe",
            "PauliFrame",
        ),
    }
}  # (module, class) mapping from OLD to NEW locations for each version change

SERIALIZATION_VERSION = 1
"""Serialization versions.

0: First version. JSON encoding only, per-shot checkpointing only.
1: HDF5 encoding now available. Backwards compatible to version 0.
"""

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
DecodeCache: TypeAlias = dict[int, "Serializable"] | None


# Generic type variable to stand-in for derived class below
T = TypeVar("T", bound="Serializable")


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
    - `from_decoded_attrs()`: Create object from decoded attributes

    Example:
        >>> # Define a simple serializable class
        >>> class SimpleClass(Serializable):
        ...     CACHE_ON_SERIALIZE = True
        ...     SERIALIZE_ATTRS = ["name", "value"]
        ...
        ...     def __init__(self, name, value):
        ...         self.name = name
        ...         self.value = value
        ...
        ...
        ...     @classmethod
        ...     def from_decoded_attrs(cls, attr_dict):
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
    def serial_hash(obj: Any, _visited: set | None = None) -> int:
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
                # For Serializable objects, hash the tuple of serial IDs of their SERIALIZE_ATTRS
                attr_ids = []
                for attr in obj.SERIALIZE_ATTRS:
                    attr_value = obj.get_encoding_attr(attr)
                    attr_ids.append(
                        Serializable.serial_hash(attr_value, _visited)
                    )
                return hash(tuple(attr_ids))
            elif isinstance(obj, list):
                # For lists, hash the tuple of serial IDs of each element
                return hash(
                    tuple(
                        Serializable.serial_hash(item, _visited)
                        for item in obj
                    )
                )
            elif isinstance(obj, dict):
                # For dicts, hash the tuple of serial IDs of keys and values
                keys_id = Serializable.serial_hash(list(obj.keys()), _visited)
                values_id = Serializable.serial_hash(
                    list(obj.values()), _visited
                )
                return hash((keys_id, values_id))
            elif isinstance(obj, np.ndarray):
                # For numpy arrays, hash the shape and flattened data
                shape_id = Serializable.serial_hash(obj.shape, _visited)
                data_id = Serializable.serial_hash(
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
    CACHE_ON_SERIALIZE: ClassVar[bool] = False
    """Flag to indicate whether this class should be cached.

    Every Serializable object _can_ be cached, but caching does
    introduce some overhead. For cases where the serialized object
    is small or not frequently references, we can save time for very
    little filesize by not caching (the default behavior).

    Some large objects that are heavily referenced *should* use caching,
    however. Some examples: Instruction, InstructionStack, QECCode,
    QECCodePatch, any backend objects, etc.
    """

    SERIALIZE_ATTRS: ClassVar[list[str]] = []
    """Attributes to serialize.

    If encoding requires a different access pattern
    than :meth:`getattr()`, derived classes should
    implement :meth:`.get_encoding_attrs`.
    """

    SERIALIZE_ATTRS_MAP: ClassVar[dict[str, str]] = {}
    """Attribute map to use in :meth:`.from_decoded_attrs()`.

    Useful when internal (e.g. _<attr>) attributes are
    serialized, but they are named differently (e.g. <attr>)
    in class constructors. If decoding requires more complex
    state management than the class constructor, derived
    classes should implement :meth:`.from_decoded_attrs`.
    """

    ## ABSTRACT METHODS
    # Implement these in derived classes

    def get_encoding_attr(
        self, attr: str, ignore_no_serialize_flags: bool = False
    ) -> Any:
        """
        Extract the attributes needed for encoding to a dictionary.

        By default, this assumes all requested attributes are available
        via getattr.
        This should be implemented in all Serializable-derived classes
        that required objects for encoding where this is not true,
        e.g. state backends. This is also true for the Frame object,
        which may modify the :attr:`.Frame.data` attribute depending
        on the ``ignore_no_serialization`` flag passed down.

        Parameters
        ----------
        attr:
            "Attribute" to retrieve

        Returns
        -------
        Any
            The "attribute" to be encoded in :meth:`.BaseEncoder.encode_uncached_obj()`.
        """
        return getattr(self, attr)

    @classmethod
    def from_decoded_attrs(cls: Type[T], attr_dict: Mapping[str, Any]) -> T:
        """
        Create an object from decoded attributes dictionary.

        By default, this assumes that attributes are either directly named
        as constructor arguments, or at least are one of the arguments and
        thus can be remapped to the proper kwarg via SERIALIZE_ATTRS_MAP.
        This should be implemented by all Serializable subclasses that for
        which the default behavior or mapping via SERIALIZE_ATTRS_MAP is not
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
            cls.SERIALIZE_ATTRS_MAP.get(k, k): v
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

        return decoded

    @classmethod
    def read(
        cls: Type[T],
        path: str | Path,
        format: EncodeFormats = None,
        use_caching: bool = True,
        decode_cache: DecodeCache = None,
    ) -> Encodable:
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
            f, format, use_caching=use_caching, decode_cache=decode_cache
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

            # Sanity check format kwargs
            if (
                "indent" not in json_format_kwargs
            ):  # default indent=4 JSON argument
                json_format_kwargs = (
                    json_format_kwargs.copy()
                )  # don't update caller's dict!
                json_format_kwargs["indent"] = 4

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

            root_group = f.create_group("root")
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
    def encode(  # noqa: C901
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

        if format == "json":
            encode_uncached_obj = functools.partial(
                JSONEncoder.encode_uncached_obj,
                encode_cache=encode_cache,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
            )
            encode_cached_obj = functools.partial(
                JSONEncoder.encode_cached_obj, h5_group=None
            )
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
            encode_array = functools.partial(
                JSONEncoder.encode_array, h5_group=None
            )
            encode_primitive = functools.partial(
                JSONEncoder.encode_primitive, h5_group=None
            )
            encode_class = functools.partial(
                JSONEncoder.encode_class, h5_group=None
            )
            encode_function = functools.partial(
                JSONEncoder.encode_function, h5_group=None
            )

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
            # Get serial ID for this object
            serial_hash = Serializable.serial_hash(obj)
            object_id = id(obj)

            # Check if this object is already in cache
            if encode_cache is not None and obj.CACHE_ON_SERIALIZE:
                # First check if this specific object instance is already being processed
                # This handles circular references within the same object graph
                if serial_hash in encode_cache:
                    cached_entries = encode_cache[serial_hash]
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

                # Check if serial_hash exists in cache (different instances with same content)
                if serial_hash in encode_cache:
                    # Same serial content but different instance, create a copy
                    source_cache_id = encode_cache[serial_hash][0][
                        1
                    ]  # First entry is the source
                    new_cache_id = (
                        JSONEncoder.ENCODE_ID
                        if format == "json"
                        else HDF5Encoder.ENCODE_ID
                    )

                    # Add to cache
                    encode_cache[serial_hash].append((object_id, new_cache_id))

                    # Increment encoder ID
                    if format == "json":
                        JSONEncoder.ENCODE_ID += 1
                    else:
                        HDF5Encoder.ENCODE_ID += 1

                    return encode_cached_obj(
                        new_cache_id,
                        cache_type="copy",
                        reference_cache_id=source_cache_id,
                        source_cache_id=new_cache_id,
                    )
                else:
                    # New serial content, create a source
                    cache_id = (
                        JSONEncoder.ENCODE_ID
                        if format == "json"
                        else HDF5Encoder.ENCODE_ID
                    )
                    encode_cache[serial_hash] = [(object_id, cache_id)]

                    # Increment encoder ID
                    if format == "json":
                        JSONEncoder.ENCODE_ID += 1
                    else:
                        HDF5Encoder.ENCODE_ID += 1

                    # Encode as source
                    result = encode_uncached_obj(obj)
                    # Add cache info to result
                    if format == "json":
                        result.update(
                            {"cache_type": "source", "cache_id": cache_id}
                        )
                    else:
                        result.attrs["cache_type"] = "source"
                        result.attrs["cache_id"] = cache_id
                    return result
            else:
                # No caching, just encode normally
                return encode_uncached_obj(obj)

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
    ) -> Encodable:
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
    def eval_function_str(
        src: str, version: int = SERIALIZATION_VERSION
    ) -> Callable:
        # Backwards compatibility, it may have been evaluated already
        if callable(src):
            return src

        # Before executing source, update imports for backwards compatibility
        updated_src = Serializable._update_imports(src, version)
        # And other known fixes
        updated_src = Serializable._function_compatibility(
            updated_src, version
        )

        # Evaluate function
        env: dict[str, Any] = {}
        exec(updated_src, env)

        # We need to find the function name
        # Search for last def, then first paren after it
        # Trim "def " and that should be the function name
        fn_defs = re.findall(r"^def .*\(", src, re.MULTILINE)
        last_fn_def = fn_defs[-1]
        key = last_fn_def[4:-1]

        # Pull the function out of the executed environment
        return env[key]

    @staticmethod
    def get_function_str(func):
        import inspect
        import textwrap

        # Get source code
        src = textwrap.dedent(inspect.getsource(func))

        # Also try to get imports
        srcfile = inspect.getsourcefile(func)
        if srcfile is None:
            # We'll fail to get imports, just return source
            return src

        # Get all import lines
        with open(srcfile, "r") as f:
            import_lines = []
            multiline = ""
            for line in f.readlines():
                if len(multiline):
                    multiline += line
                    if ")" in line:
                        import_lines.append(textwrap.dedent(multiline))
                        multiline = ""
                elif "import " in line:
                    if "(" in line and ")" not in line:
                        multiline = line
                    else:
                        import_lines.append(textwrap.dedent(line))

        # Get all things that are imported
        needed_import_lines = []
        for line in import_lines:
            if " as " in line:
                entry_str = line.split(" as ")[1]
            else:
                entry_str = line.split("import ")[1]
            # Remove parentheses from multiline imports
            entries = [
                e.replace("(", "").replace(")", "")
                for e in entry_str.split(",")
            ]

            # Get rid of newline and whitespace for better searching
            entries = [e.strip() for e in entries if len(e.strip())]

            # If the imported thing is in our source code, we need this import
            if any([e in src for e in entries]):
                needed_import_lines.append(line)

        imports = "".join(needed_import_lines)
        return imports + src

    @staticmethod
    def import_class(module_name, class_name, version) -> Type:
        """Returns the class specified by the given state dictionary"""
        location_changes = (
            {}
            if version == SERIALIZATION_VERSION
            else Serializable._get_cumulative_changes(version)
        )

        if (module_name, class_name) in location_changes:
            module_name, class_name = location_changes[module_name, class_name]
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
    def _update_imports(  # noqa: C901
        function_str, initial_version=None, loc_change=None
    ):
        """
        Update Python import statements based on a dictionary of location changes.

        Args:
            function_str: String containing Python import statements
            initial_version: Version of function_str
            loc_change: Dictionary mapping (old_module, old_class) to (new_module, new_class)

        Returns:
            String with updated import statements, each on its own line
        """
        lines = function_str.split("\n")
        result_lines = []

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

        # First pass: join multi-line imports
        processed_lines = []
        current_import = None

        for line in lines:
            stripped_line = line.strip()

            # Keep empty lines and comments as-is
            if not stripped_line or stripped_line.startswith("#"):
                processed_lines.append(line)
                continue

            # Check if this is the start of a multi-line import
            if stripped_line.startswith("from") and stripped_line.endswith(
                "("
            ):
                current_import = stripped_line
                continue

            # Check if this is a continuation of a multi-line import
            if current_import is not None:
                current_import += " " + stripped_line
                if stripped_line.endswith(")"):
                    # Remove parentheses and extra white space, collapsing this into single line
                    current_import = current_import.replace("(", "").replace(
                        ")", ""
                    )
                    current_import = " ".join(current_import.split())
                    if current_import.endswith(","):
                        current_import = current_import[:-1]
                    processed_lines.append(current_import)
                    current_import = None
                continue

            # Single line
            processed_lines.append(line)

        # Second pass: process imports
        # TODO: This will probably fail if a higher-level module is used in import
        # For example, from mod1 import cls1 instead of from mod1.mod2 import cls1,
        # if the key in loc_changes is (mod1.mod2, cls)
        # Should be able to check substrings in that case, but hasn't been necessary yet
        # TODO: If any of the backends move out of loqs.backends, we may need special code here
        updated_imported_names = {}
        for line in processed_lines:
            stripped_line = line.strip()

            # Keep empty lines and comments as-is
            if not stripped_line or stripped_line.startswith("#"):
                result_lines.append(line)
                continue

            # Check if this is an import statement
            match = re.match(r"from\s+([\w.]+)\s+import\s+(.+)", stripped_line)
            if not match:
                # Not an import line, keep as-is
                result_lines.append(line)
                continue

            module = match.group(1)
            imports_str = match.group(2)

            # Split imports by comma and handle each one
            import_items = [item.strip() for item in imports_str.split(",")]

            # Check if any of the imports in this line need to be updated
            needs_update = any(
                (module, item.split(" as ")[0].strip()) in loc_change
                for item in import_items
            )

            if needs_update:
                # Process each import individually
                for item in import_items:
                    item = item.strip()
                    if not item:
                        continue

                    # Check if this is an aliased import (cls as alias)
                    alias_match = re.match(r"(\w+)\s+as\s+(\w+)", item)
                    if alias_match:
                        original_name = alias_match.group(1)
                        alias = alias_match.group(2)
                    else:
                        original_name = item
                        alias = None

                    # Check if this import needs to be updated
                    key = (module, original_name)
                    if key in loc_change:
                        new_module, new_name = loc_change[key]
                        if alias:
                            result_lines.append(
                                f"from {new_module} import {new_name} as {alias}"
                            )
                        else:
                            result_lines.append(
                                f"from {new_module} import {new_name}"
                            )
                            # We may need to replace this name throughout the rest of the program
                            if new_name != original_name:
                                updated_imported_names[original_name] = (
                                    new_name
                                )
                    else:
                        # Keep the original import
                        if alias:
                            result_lines.append(
                                f"from {module} import {original_name} as {alias}"
                            )
                        else:
                            result_lines.append(
                                f"from {module} import {original_name}"
                            )
            else:
                # No updates needed, keep the original line
                result_lines.append(line)

        # Third pass: replace renamed modules throughout code
        final_lines = []
        for line in result_lines:
            updated_line = line
            for orig_name, new_name in updated_imported_names.items():
                # Don't remap an already mapped name
                if new_name not in line:
                    updated_line = updated_line.replace(orig_name, new_name)
            final_lines.append(updated_line)

        final_result = "\n".join(final_lines)

        return final_result

    @staticmethod
    def _replace_placeholders(obj, decode_cache):
        """Recursively replace circular reference objects with actual objects from decode_cache."""
        if obj is None:
            return obj

        # Check if this is a circular reference placeholder
        if hasattr(obj, "cache_id"):
            # This is a circular reference, replace it with the actual object
            actual_cache_id = obj.cache_id
            if actual_cache_id in decode_cache:
                actual_obj = decode_cache[actual_cache_id]
                # If the actual object is still a circular reference, keep it as is
                # (this can happen during the replacement process)
                if not hasattr(actual_obj, "cache_id"):
                    return actual_obj
            return obj

        # Handle Serializable objects
        if isinstance(obj, Serializable):
            # Recursively process attributes
            for attr in obj.SERIALIZE_ATTRS:
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

        # Get cumulative changes in import locations
        complete_location_changes = IMPORT_LOCATION_CHANGES_BY_VERSION[
            initial_version + 1
        ].copy()

        version = initial_version + 1
        while version < SERIALIZATION_VERSION:
            for new_k, new_v in IMPORT_LOCATION_CHANGES_BY_VERSION[
                version
            ].items():
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
