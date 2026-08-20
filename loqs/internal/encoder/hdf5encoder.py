#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

import contextvars
import json
from typing import ClassVar

import h5py
import numpy as np
import scipy.sparse as sps

from loqs.internal.serializable import (
    DecodeCache,
    DeferredRef,
    Encodable,
    EncodableArrays,
    EncodableIterables,
    EncodablePrimitives,
    Encoded,
    IncorrectDecodableTypeError,
)
from loqs.types import NDArray, SPSArray
from loqs.internal import Serializable, SERIALIZATION_VERSION
from loqs.internal.encoder import BaseEncoder
from loqs.internal.encoder.baseencoder import copy_cached_reference
from loqs.internal.versioning import VersionedDecoder

# Per-shape version dispatch (see VersionedDecoder), each validating its
# own version's expected structure. No HDF5 shape has varied between
# versions 1 and 2 yet, so every registry below aliases 2 to 1.

# ---- hdf5_serializable ----
_decode_hdf5_serializable_version = VersionedDecoder("hdf5_serializable")


@_decode_hdf5_serializable_version.register(1)
def _decode_hdf5_serializable_version_v1(encoded):
    assert "module" in encoded.attrs
    assert "class" in encoded.attrs


_decode_hdf5_serializable_version.alias(2, same_as=1)


# ---- hdf5_cached_obj ----
_decode_hdf5_cached_obj_version = VersionedDecoder("hdf5_cached_obj")


@_decode_hdf5_cached_obj_version.register(1)
def _decode_hdf5_cached_obj_version_v1(encoded):
    cache_type = encoded.attrs["cache_type"]
    if cache_type == "reference":
        assert "cache_id" in encoded.attrs
    elif cache_type == "copy":
        assert "reference_cache_id" in encoded.attrs
        assert "source_cache_id" in encoded.attrs


_decode_hdf5_cached_obj_version.alias(2, same_as=1)


# ---- hdf5_iterable ----
_decode_hdf5_iterable_version = VersionedDecoder("hdf5_iterable")


@_decode_hdf5_iterable_version.register(1)
def _decode_hdf5_iterable_version_v1(list_group):
    assert isinstance(list_group, h5py.Group)
    assert list_group.attrs.get("iterable_type", "") in ["list", "tuple", "set"]


_decode_hdf5_iterable_version.alias(2, same_as=1)


# ---- hdf5_dict ----
_decode_hdf5_dict_version = VersionedDecoder("hdf5_dict")


@_decode_hdf5_dict_version.register(1)
def _decode_hdf5_dict_version_v1(dict_group):
    assert isinstance(dict_group, h5py.Group)
    assert "keys" in dict_group
    assert "values" in dict_group


_decode_hdf5_dict_version.alias(2, same_as=1)


# ---- hdf5_array ----
_decode_hdf5_array_version = VersionedDecoder("hdf5_array")


@_decode_hdf5_array_version.register(1)
def _decode_hdf5_array_version_v1(array_group):
    assert isinstance(array_group, h5py.Group)
    assert "shape" in array_group.attrs
    assert "dtype" in array_group.attrs
    array_type = array_group.attrs.get("array_type", None)
    assert array_type in ["sparse_csr", "dense_complex", "dense_real"]


_decode_hdf5_array_version.alias(2, same_as=1)


# ---- hdf5_class ----
_decode_hdf5_class_version = VersionedDecoder("hdf5_class")


@_decode_hdf5_class_version.register(1)
def _decode_hdf5_class_version_v1(class_group):
    assert isinstance(class_group, h5py.Group)
    assert "module" in class_group.attrs
    assert "class" in class_group.attrs


_decode_hdf5_class_version.alias(2, same_as=1)


# ---- hdf5_function ----
_decode_hdf5_function_version = VersionedDecoder("hdf5_function")


@_decode_hdf5_function_version.register(1)
def _decode_hdf5_function_version_v1(function_group):
    """Validate and extract the raw source string from `function_group`."""
    assert isinstance(function_group, h5py.Group)
    assert "source" in function_group
    source_dataset = function_group["source"]
    assert isinstance(source_dataset, h5py.Dataset)
    source = source_dataset[()]
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    else:
        source = str(source)
    assert isinstance(source, str)
    return source


_decode_hdf5_function_version.alias(2, same_as=1)


# ---- hdf5_primitive ----
_decode_hdf5_primitive_version = VersionedDecoder("hdf5_primitive")


@_decode_hdf5_primitive_version.register(1)
def _decode_hdf5_primitive_version_v1(encoded):
    assert "value" in encoded.attrs


_decode_hdf5_primitive_version.alias(2, same_as=1)


_HDF5_DECODE_VERSION: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "hdf5_decode_version", default=None
)
"""The serialization version of the file currently being decoded.

Every node's shape has been identical across every `SERIALIZATION_VERSION`
so far, so repeating the same version number as an HDF5 attribute at every
one of the thousands of nodes a typical file contains is pure overhead,
non-trivial against files with thousands of small objects. `decode_root_group`
reads the version once, from the file's own
root group, and sets this for the duration of decoding that file, instead
of threading an extra parameter through the whole recursive decode dispatch
chain (the same pattern `Serializable`'s own `MIGRATE_LEGACY_FNS` uses).
"""


def _get_version(encoded) -> int:
    """Resolve the serialization version to validate `encoded` against.

    A node's own local "version" attribute, if present, is trusted first --
    it's real ground truth for a file written before this per-node
    stamping was consolidated onto just the file's root. Otherwise, falls
    back to the ambient value `decode_root_group` set once for the whole
    file being decoded, or, failing that (e.g. a `Serializable`-encoded
    HDF5 group built directly rather than via `Serializable.dump`, so no
    root group ever set the ambient value either), to the current
    `SERIALIZATION_VERSION` -- the only sane assumption left once neither
    of the real version markers is available.
    """
    if "version" in encoded.attrs:
        return encoded.attrs["version"]
    version = _HDF5_DECODE_VERSION.get()
    if version is not None:
        return version
    return SERIALIZATION_VERSION


def _create_group(h5_group, name):
    """Create a subgroup with link-creation-order tracking enabled.

    Switches HDF5 from its legacy symbol-table group format to its more
    compact one, roughly halving per-group overhead for the small groups this
    encoder creates constantly, and fixes a real decode-ordering bug: without
    it, `h5py.Group.keys()` returns children alphabetically rather than in
    the insertion order the cache/reference-resolution mechanism relies on.
    """
    return h5_group.create_group(name, track_order=True)


# ---- Array-free-subtree collapse: a `Serializable`'s attrs, a dict's
# values, or an iterable's elements that have no array anywhere below them
# get batched into one compact JSON blob dataset instead of one HDF5 group
# per node. ----

_COLLAPSED_BLOB_NAME = "$collapsed"
"""Reserved sibling name for a collapsed blob, chosen to never collide with a
real `_SERIALIZE_ATTRS` name (a valid Python identifier) or a stringified
iterable index (a bare integer)."""

_GZIP_MIN_BYTES = 128
"""Below this raw blob size, gzip's own container/filter overhead costs more
than it saves: compression is reliably worse under ~90 bytes, a noisy
break-even band up to ~130, and reliably better above that. 128 sits safely
inside the "reliably better" zone with a small margin."""

_decode_hdf5_collapsed_version = VersionedDecoder("hdf5_collapsed")
"""Versions the collapsed blob's own on-disk *wrapper* shape (currently
always a gzip-or-not JSON byte dataset), independent of `SERIALIZATION_VERSION`
itself -- collapsing is a per-subtree, content-dependent encode-time choice,
not a version-gated shape, so this registry exists purely so a future
change to the wrapper shape has somewhere to register a new version without
retrofitting one later. The blob's own *contents* need no separate registry
-- they're just JSON, decoded via `JSONEncoder`'s own existing
version-aware decode path."""


@_decode_hdf5_collapsed_version.register(SERIALIZATION_VERSION)
def _decode_hdf5_collapsed_version_current(encoded):
    assert isinstance(encoded, h5py.Dataset)


def _contains_no_array(value, encode_cache, ignore_no_serialize_flags, _visited=None):
    """Whether encoding `value` right now would touch no real array anywhere in its own expansion.

    A value that would resolve to an already-registered cache
    "reference"/"copy" counts as array-free regardless of its own content --
    it's already a cheap stub either way, mirroring `_encode_Serializable`'s
    own cache-hit check. This check is read-only: it never registers
    anything in `encode_cache` itself, since the real encode call that
    follows a "yes, collapse this" answer does that.

    `_visited` guards against infinite recursion on a genuine circular
    reference (an object reachable from its own attrs); a value already
    being examined further up the same call chain is conservatively treated
    as "may contain an array" rather than walked again -- this only ever
    costs a slightly less compact encoding for that one node, never
    correctness, since falling back to an ordinary HDF5 group is always safe.
    """
    if _visited is None:
        _visited = frozenset()

    if isinstance(value, EncodableArrays):
        return False

    if isinstance(value, Serializable):
        obj_id = id(value)
        if obj_id in _visited:
            return False
        if (
            encode_cache is not None
            and value._CACHE_ON_SERIALIZE
            and Serializable._serial_hash(value) in encode_cache
        ):
            return True
        _visited = _visited | {obj_id}
        return all(
            _contains_no_array(
                value._get_encoding_attr(
                    attr, ignore_no_serialize_flags=ignore_no_serialize_flags
                ),
                encode_cache,
                ignore_no_serialize_flags,
                _visited,
            )
            for attr in value._SERIALIZE_ATTRS
        )

    if isinstance(value, dict):
        return all(
            _contains_no_array(v, encode_cache, ignore_no_serialize_flags, _visited)
            for v in (*value.keys(), *value.values())
        )

    if isinstance(value, (list, tuple, set)):
        return all(
            _contains_no_array(v, encode_cache, ignore_no_serialize_flags, _visited)
            for v in value
        )

    return True


def _encode_collapsed_children(
    h5_group, children, encode_cache, ignore_no_serialize_flags
):
    """Write `children` (name -> already-collapse-eligible value) as one combined JSON blob dataset."""
    from loqs.internal.encoder import JSONEncoder

    blob = {}
    for name, value in children.items():
        # Borrow HDF5's own ENCODE_ID counter for this JSON sub-encode, then
        # hand the advanced counter back -- JSONEncoder and HDF5Encoder
        # otherwise track separate counters, which could hand out a cache ID
        # that collides with an unrelated object elsewhere in this same file.
        JSONEncoder.ENCODE_ID = HDF5Encoder.ENCODE_ID
        blob[name] = Serializable.encode(
            value,
            format="json",
            encode_cache=encode_cache,
            ignore_no_serialize_flags=ignore_no_serialize_flags,
        )
        HDF5Encoder.ENCODE_ID = JSONEncoder.ENCODE_ID

    blob_bytes = json.dumps(blob, separators=(",", ":")).encode("utf-8")
    compression = "gzip" if len(blob_bytes) >= _GZIP_MIN_BYTES else None
    blob_dataset = h5_group.create_dataset(
        _COLLAPSED_BLOB_NAME,
        data=np.frombuffer(blob_bytes, dtype=np.uint8),
        compression=compression,
    )
    blob_dataset.attrs["encode_type"] = "collapsed"


def _decode_collapsed_children(blob_dataset, decode_cache):
    """Decode a combined blob dataset back into its `{name: decoded_value}` children."""
    with HDF5Encoder.assert_decode(fatal=False):
        assert isinstance(blob_dataset, h5py.Dataset)
        assert blob_dataset.attrs.get("encode_type", "") == "collapsed"

    version = _get_version(blob_dataset)
    with HDF5Encoder.assert_decode(fatal=True):
        _decode_hdf5_collapsed_version(version, blob_dataset)

    raw = blob_dataset[()].tobytes().decode("utf-8")
    blob = json.loads(raw)
    return {
        name: Serializable.decode(value, format="json", decode_cache=decode_cache)
        for name, value in blob.items()
    }


_FORCE_REAL_GROUP: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "hdf5_force_real_group", default=False
)
"""Set for the duration of encoding one specific child's value, forcing
*that* child's own direct children to skip collapse consideration too --
propagates a `Serializable._NO_COLLAPSE_ATTRS` exemption exactly one level
deeper (e.g. from `ProgramResults.shot_histories` into the dict's own
keys/values element lists), then is consumed (reset to the default) by the
very next `_partition_and_encode_children` call, so collapse resumes
normally any deeper than that."""


def _partition_and_encode_children(
    h5_group,
    children,
    encode_cache,
    ignore_no_serialize_flags,
    no_collapse_names=frozenset(),
):
    """Partition `children` (name -> value) between real HDF5 groups and one shared collapsed blob, writing both directly into `h5_group`.

    `children` covers both a `Serializable`'s own `_SERIALIZE_ATTRS` (named
    by attribute) and an iterable's elements (named by stringified index,
    including the element list backing a dict's keys/values). Any child
    with no array anywhere below it -- or that would just be a cheap cache
    reference/copy of an already-encoded source regardless of its own
    content -- is batched into one combined JSON blob dataset instead of
    its own HDF5 group. A child containing a real, not-yet-cached array, or
    named in `no_collapse_names` (see `Serializable._NO_COLLAPSE_ATTRS`),
    gets an ordinary named group, with this same partition applied one
    level further down inside it via the normal recursive
    `Serializable.encode` dispatch.
    """
    force_all = _FORCE_REAL_GROUP.get()
    reset_token = _FORCE_REAL_GROUP.set(False)
    try:
        collapsible = {}
        for name, value in children.items():
            must_be_real_group = force_all or name in no_collapse_names
            if not must_be_real_group and _contains_no_array(
                value, encode_cache, ignore_no_serialize_flags
            ):
                collapsible[name] = value
                continue

            child_group = _create_group(h5_group, name)
            propagate_token = (
                _FORCE_REAL_GROUP.set(True)
                if name in no_collapse_names
                else None
            )
            try:
                Serializable.encode(
                    value,
                    format="hdf5",
                    encode_cache=encode_cache,
                    ignore_no_serialize_flags=ignore_no_serialize_flags,
                    h5_group=child_group,
                )
            finally:
                if propagate_token is not None:
                    _FORCE_REAL_GROUP.reset(propagate_token)

        if collapsible:
            _encode_collapsed_children(
                h5_group, collapsible, encode_cache, ignore_no_serialize_flags
            )
    finally:
        _FORCE_REAL_GROUP.reset(reset_token)


def _decode_partitioned_children(h5_group, decode_cache):
    """Yield `(name, decoded_value)` for every child of `h5_group`, expanding the collapsed blob (if present) into its own individual entries.

    Backward compatible by construction: a file predating this feature
    simply has no `_COLLAPSED_BLOB_NAME` child, so every child is decoded
    exactly as it always was.
    """
    for key in h5_group.keys():
        child = h5_group[key]
        if key == _COLLAPSED_BLOB_NAME:
            yield from _decode_collapsed_children(child, decode_cache).items()
        else:
            with HDF5Encoder.assert_decode(fatal=True):
                assert isinstance(child, h5py.Group)
            yield key, Serializable.decode(
                child, format="hdf5", decode_cache=decode_cache
            )


class HDF5Encoder(BaseEncoder):

    ENCODE_ID: ClassVar[int] = 0

    @staticmethod
    def decode_root_group(
        encoded: Encoded,
        decode_cache: DecodeCache = None,
    ) -> Encodable:
        """Decode the root HDF5 group containing a serialized object.

        This same "unwrap a single anonymous subgroup" shape is also what
        every individual `_SERIALIZE_ATTRS` wrapper group looks like
        (`Serializable.encode` never puts its own attributes directly on
        the `h5_group` it's handed), so this method is actually invoked
        recursively throughout a file's whole tree, not only once at the
        true file root -- only the true root ever carries a "version"
        attribute, which is how it's told apart below.

        Parameters
        ----------
        encoded : Encoded
            An HDF5 group containing a serialized object, or a wrapper
            around one. Should contain exactly one subgroup, with no
            attributes of its own except possibly a single "version"
            attribute (present only on the file's true root group).
        decode_cache : DecodeCache, optional
            Cache used to track decoded objects and resolve references during
            deserialization.

        Returns
        -------
        Encodable
            The decoded object contained in the HDF5 file.

        Raises
        ------
        IncorrectDecodableTypeError
            If the encoded structure is not a valid HDF5 serialization format.
        """
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert len(encoded.keys()) == 1
            assert set(encoded.attrs.keys()) <= {"version"}

        # Get the first (and only) subgroup
        subgroup_name = list(encoded.keys())[0]
        subgroup = encoded[subgroup_name]
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(subgroup, h5py.Group)

        version = encoded.attrs.get("version", None)
        token = (
            _HDF5_DECODE_VERSION.set(version) if version is not None else None
        )
        try:
            # Try to decode the subgroup
            # It could either error out or give IncorrectDecodableTypeError
            # Both are acceptable
            return Serializable.decode(
                subgroup, format="hdf5", decode_cache=decode_cache
            )
        finally:
            if token is not None:
                _HDF5_DECODE_VERSION.reset(token)

    @staticmethod
    def encode_uncached_obj(
        to_encode,
        encode_cache=None,
        ignore_no_serialize_flags=False,
        h5_group=None,
    ):
        assert isinstance(to_encode, Serializable)
        assert isinstance(h5_group, h5py.Group)

        obj_group = _create_group(
            h5_group, f"Serializable_{HDF5Encoder.ENCODE_ID}"
        )
        obj_group.attrs["encode_type"] = "Serializable"
        obj_group.attrs["module"] = to_encode.__class__.__module__
        obj_group.attrs["class"] = to_encode.__class__.__name__

        # Use _SERIALIZE_ATTRS pattern for encoding. Attrs with no array
        # anywhere below them are batched into one shared collapsed blob
        # instead of getting their own HDF5 group each -- see
        # _partition_and_encode_children.
        attrs = {
            serial_attr: to_encode._get_encoding_attr(
                serial_attr,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
            )
            for serial_attr in to_encode._SERIALIZE_ATTRS
        }
        _partition_and_encode_children(
            obj_group,
            attrs,
            encode_cache,
            ignore_no_serialize_flags,
            no_collapse_names=to_encode._NO_COLLAPSE_ATTRS,
        )

        HDF5Encoder.ENCODE_ID += 1

        return obj_group

    @staticmethod
    def decode_uncached_obj(encoded, decode_cache=None):
        """Decode a Serializable object from HDF5 format.

        Deserializes a Serializable object that was not previously cached.
        This method handles the core deserialization logic for Serializable objects,
        including attribute reconstruction and circular reference handling.

        Parameters
        ----------
        encoded : h5py.Group
            The HDF5 group containing the encoded Serializable object.
            Should have 'encode_type' attribute set to 'Serializable' and
            contain subgroups for each serialized attribute.
        decode_cache : dict, optional
            Dictionary mapping cache IDs to decoded objects. Used to handle
            circular references and object caching during deserialization.

        Returns
        -------
        Serializable
            The decoded Serializable object of the appropriate class.

        Raises
        ------
        IncorrectDecodableTypeError
            If the encoded object is not a valid Serializable object.
        DecodableVersionError
            If the serialization version is not supported.
        ImportError
            If the class cannot be imported from the specified module.
        """
        # Check if right type
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert encoded.attrs.get("encode_type", "") == "Serializable"

        # Check if properly formed
        version = _get_version(encoded)
        with HDF5Encoder.assert_decode(fatal=True):
            _decode_hdf5_serializable_version(version, encoded)

        # Get the class
        cls = Serializable._import_class(
            encoded.attrs["module"], encoded.attrs["class"], version
        )

        # Handle circular references by adding a placeholder to decode_cache early
        cache_id = None
        if (
            encoded.attrs.get("cache_type", None) == "source"
            and decode_cache is not None
        ):
            try:
                cache_id = int(encoded.attrs["cache_id"])  # type: ignore
                decode_cache[cache_id] = DeferredRef(cache_id)
            except (KeyError, TypeError):
                pass  # Not a source object, no need for early caching

        # Create the attribute dictionary for deserialization. Expands the
        # collapsed blob (if present) back into its own individual named
        # entries alongside any attrs that kept their own real HDF5 group.
        attr_dict = dict(_decode_partitioned_children(encoded, decode_cache))

        # If our class is an Instruction, we also need to pass in version
        # so that imports can be updated properly on apply_fn/map_qubits_fn creation
        from loqs.core import Instruction

        if cls == Instruction:
            attr_dict["version"] = version

        # Create the object using its _from_decoded_attrs method
        decoded = cls._from_decoded_attrs(attr_dict)

        # Replace the placeholder with the actual object
        if (
            decode_cache is not None
            and cache_id is not None
            and cache_id in decode_cache
        ):
            decode_cache[cache_id] = decoded  # type: ignore

        return decoded

    @staticmethod
    def encode_cached_obj(
        cache_id,
        h5_group=None,
        cache_type="reference",
        reference_cache_id=None,
        source_cache_id=None,
    ):
        """Encode a cached object reference in HDF5 format.

        This method creates a reference to an object that has already been serialized,
        avoiding duplicate storage of identical objects. Used for implementing object
        caching during serialization to improve efficiency and handle circular references.

        Parameters
        ----------
        cache_id : int
            The cache ID for this reference.
        h5_group : h5py.Group
            The HDF5 group to write the cached object reference to.
        cache_type : str, optional
            Type of cache reference, either 'reference' (multiple references to same object)
            or 'copy' (copy of an existing object). Default is 'reference'.
        reference_cache_id : int, optional
            For copy-type caching, the cache ID of the reference object.
        source_cache_id : int, optional
            For copy-type caching, the cache ID to assign to the copied object.

        Returns
        -------
        h5py.Group
            The HDF5 group containing the encoded cached object reference.
        """
        assert isinstance(h5_group, h5py.Group)

        obj_group = _create_group(
            h5_group, f"Serializable_{HDF5Encoder.ENCODE_ID}"
        )
        HDF5Encoder.ENCODE_ID += 1
        obj_group.attrs["encode_type"] = "Serializable"
        obj_group.attrs["cache_type"] = cache_type

        if cache_type == "reference":
            obj_group.attrs["cache_id"] = cache_id
        elif cache_type == "copy":
            obj_group.attrs["reference_cache_id"] = reference_cache_id
            obj_group.attrs["source_cache_id"] = source_cache_id

        return obj_group

    @staticmethod
    def decode_cached_obj(encoded, decode_cache=None):
        """Decode a cached object reference from HDF5 format.

        This method handles the deserialization of object references that were
        cached during encoding to avoid duplicate serialization of identical objects.
        It supports both reference-type caching (where multiple references point to
        the same object) and copy-type caching (where objects with identical content
        are stored once and copied).

        Parameters
        ----------
        encoded : h5py.Group
            The HDF5 group containing the encoded cached object reference.
            Should have 'encode_type' attribute set to 'Serializable' and
            'cache_type' attribute indicating the type of cache reference.
        decode_cache : dict, optional
            Dictionary mapping cache IDs to decoded objects. Used to resolve
            object references and handle circular references.

        Returns
        -------
        Serializable | DeferredRef
            The decoded object. If the referenced object is not yet available
            in the cache, returns a DeferredRef placeholder that will be
            resolved later.

        Raises
        ------
        IncorrectDecodableTypeError
            If the encoded object is not a valid cached object reference.
        DecodableVersionError
            If the serialization version is not supported.
        RuntimeError
            If object references cannot be resolved due to missing source objects.
        """
        # Check if right type
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert encoded.attrs.get("encode_type", "") == "Serializable"
            # Only proceed if this actually has cache_type attribute
            if "cache_type" not in encoded.attrs:
                raise IncorrectDecodableTypeError("Not a cached object")
            cache_type = encoded.attrs["cache_type"]
            assert cache_type in ["reference", "copy"]

            assert decode_cache is not None

        # Check if properly formed
        version = _get_version(encoded)
        with HDF5Encoder.assert_decode(fatal=True):
            _decode_hdf5_cached_obj_version(version, encoded)

        try:
            if cache_type == "reference":
                cache_id = int(encoded.attrs["cache_id"])  # type: ignore
                cached_obj = decode_cache[cache_id]
                return cached_obj

            # Get the reference object and create a copy
            reference_cache_id = int(encoded.attrs["reference_cache_id"])  # type: ignore
            source_cache_id = int(encoded.attrs["source_cache_id"])  # type: ignore

            # Check if reference object is available
            if reference_cache_id not in decode_cache:
                # Reference object not available yet, create a placeholder
                copied_obj = DeferredRef(reference_cache_id)
            else:
                reference_obj = decode_cache[reference_cache_id]
                copied_obj = copy_cached_reference(reference_obj)

            # Add the copy to cache
            decode_cache[source_cache_id] = copied_obj
            return copied_obj

        except (KeyError, TypeError):
            raise RuntimeError(
                "Object reference found but source object not available."
            )

    @staticmethod
    def encode_iterable(
        to_encode,
        encode_cache=None,
        ignore_no_serialize_flags=False,
        h5_group=None,
    ):
        assert isinstance(h5_group, h5py.Group)
        assert isinstance(to_encode, EncodableIterables)

        if isinstance(to_encode, list):
            name = "list"
        elif isinstance(to_encode, set):
            name = "set"
        elif isinstance(to_encode, tuple):
            name = "tuple"
        else:
            raise ValueError(
                f"Type {type(to_encode)} not handled by encode_iterable"
            )

        list_group = _create_group(h5_group, "iterable")
        list_group.attrs["iterable_type"] = name

        # Short circuit empty list
        if len(to_encode) == 0:
            list_group.attrs["storage_format"] = "groups"
            return list_group

        # Cast to list so we can handle sets
        to_encode_list = list(to_encode)

        # Check if all elements are HDF5-native types that can be stored directly as datasets
        # Not exactly EncodablePrimitives because of Nones
        hdf5_native_types = (int, float, bool, str, bytes)
        first_element = to_encode_list[0]
        first_type = type(first_element)
        # NOTE: exact type comparison, not isinstance -- `bool` is a
        # subclass of `int`, so isinstance would treat e.g. `(0, True)` as
        # a homogeneous int tuple and silently coerce `True` to `1`.
        if first_type in hdf5_native_types and all(
            type(e) is first_type for e in to_encode
        ):
            # Use HDF5 dataset for optimized storage
            list_group.attrs["storage_format"] = "dataset"
            # By default, these are fixed-size
            # Users can replace them with extendable ones as needed
            # by overriding the dataset with _encode_iterable_dataset(..., ..., True)
            HDF5Encoder._encode_iterable_dataset(
                list_group, to_encode_list, False
            )
        else:
            # Mixed native types or non-native types - fall back to groups.
            # Elements with no array anywhere below them are batched into
            # one shared collapsed blob instead of their own HDF5 group
            # each -- see _partition_and_encode_children (this also covers
            # a dict's own keys/values lists, which route through here).
            list_group.attrs["storage_format"] = "groups"
            items = {str(i): e for i, e in enumerate(to_encode_list)}
            _partition_and_encode_children(
                list_group, items, encode_cache, ignore_no_serialize_flags
            )

        return list_group

    @staticmethod
    def _encode_iterable_dataset(
        list_group, to_encode_list, extendable_dataset
    ):
        first_element = to_encode_list[0]
        if isinstance(first_element, (str, bytes)):
            # Find maximum str/bytes length to determine dtype
            max_len = (
                max(len(b) for b in to_encode_list) if to_encode_list else 0
            )
            dtype = f"S{max_len + 1}"  # +1 for null terminator

            if isinstance(first_element, bytes):
                data = np.array(
                    [
                        b.decode("utf-8", errors="replace")
                        for b in to_encode_list
                    ],
                    dtype=dtype,
                )
                list_group.attrs["original_type"] = "bytes"
            else:
                data = np.array(to_encode_list, dtype=dtype)
        else:
            # For numeric types, determine appropriate dtype
            data = np.array(to_encode_list)
            dtype = data.dtype

        # If either one of these are triggered, chunking is also silently used
        # We let HDF5 guess for chunk size (i.e. we don't provide a size)
        shape = data.shape
        maxshape = (None, *shape[1:]) if extendable_dataset else None
        compression = "gzip" if len(to_encode_list) > 1000 else None

        list_group.create_dataset(
            "data",
            data=data,
            dtype=dtype,
            compression=compression,
            maxshape=maxshape,
        )

    @staticmethod
    def decode_iterable(encoded, decode_cache=None):
        """Decode an iterable (list, tuple, set) from HDF5 format.

        Deserializes iterable objects that were serialized using encode_iterable.
        Supports both the original format (individual groups for each element)
        and the optimized format (HDF5 datasets for homogeneous native types).

        Parameters
        ----------
        encoded : h5py.Group
            The HDF5 group containing the encoded iterable.
            Should have an 'iterable' subgroup with appropriate structure.
        decode_cache : dict, optional
            Dictionary mapping cache IDs to decoded objects for reference resolution.

        Returns
        -------
        list | tuple | set
            The decoded iterable object of the appropriate type.

        Raises
        ------
        IncorrectDecodableTypeError
            If the encoded object is not a valid iterable.
        DecodableVersionError
            If the serialization version is not supported.
        """
        # Check if right type
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert "iterable" in encoded

        list_group = encoded["iterable"]

        # Check if properly formed
        version = _get_version(list_group)
        with HDF5Encoder.assert_decode(fatal=True):
            _decode_hdf5_iterable_version(version, list_group)

        # Determine storage format (default to "groups" for backwards compatibility)
        storage_format = list_group.attrs.get("storage_format", "groups")

        if storage_format == "dataset":
            # New optimized format using HDF5 datasets
            with HDF5Encoder.assert_decode(fatal=True):
                assert "data" in list_group
                data_dataset = list_group["data"]
                assert isinstance(data_dataset, h5py.Dataset)

            # Read data from dataset
            data = data_dataset[()]

            # Convert numpy array back to appropriate Python types
            if data.dtype.kind in ["i", "u"]:  # integer types
                value = [int(x) for x in data.flat]
            elif data.dtype.kind == "f":  # float types
                value = [float(x) for x in data.flat]
            elif data.dtype.kind == "b":  # boolean types
                value = [bool(x) for x in data.flat]
            elif data.dtype.kind in ["U", "S"]:  # string types
                # Check if original type was bytes
                if list_group.attrs.get("original_type") == "bytes":
                    value = [
                        x.encode("utf-8") if isinstance(x, str) else x
                        for x in data.flat
                    ]
                else:
                    value = [
                        str(x, "utf-8") if isinstance(x, bytes) else str(x)
                        for x in data.flat
                    ]
            elif data.dtype.kind == "O":  # object types (could be mixed)
                # Handle object arrays which might contain strings, bytes, etc.
                value = []
                for x in data.flat:
                    if isinstance(x, bytes):
                        value.append(x)  # Keep as bytes
                    elif isinstance(x, str):
                        value.append(x)
                    else:
                        value.append(x)
            else:
                # Fallback: convert to list
                value = list(data.flat)
        else:
            # Original format using individual groups, plus (possibly) one
            # shared collapsed blob covering some subset of the elements --
            # expand it back into its own individually-indexed entries
            # before reassembling the list in the right order (collapsing
            # can leave fewer top-level HDF5 keys than actual elements).
            decoded_by_index = dict(
                _decode_partitioned_children(list_group, decode_cache)
            )
            with HDF5Encoder.assert_decode(fatal=True):
                assert set(decoded_by_index.keys()) == {
                    str(i) for i in range(len(decoded_by_index))
                }
            value = [
                decoded_by_index[str(i)] for i in range(len(decoded_by_index))
            ]

        # Cast if needed
        if "iterable_type" in list_group.attrs:
            if list_group.attrs["iterable_type"] == "tuple":
                return tuple(value)
            elif list_group.attrs["iterable_type"] == "set":
                return set(value)

        # Otherwise return list
        return value

    @staticmethod
    def encode_dict(
        to_encode,
        encode_cache=None,
        ignore_no_serialize_flags=False,
        h5_group=None,
    ):
        assert isinstance(to_encode, dict)
        assert isinstance(h5_group, h5py.Group)

        dict_group = _create_group(h5_group, "dict")

        # Store keys and values in order to preserve dict insertion order
        key_group = _create_group(dict_group, "keys")
        Serializable.encode(
            list(to_encode.keys()),
            format="hdf5",
            encode_cache=encode_cache,
            ignore_no_serialize_flags=ignore_no_serialize_flags,
            h5_group=key_group,
        )

        val_group = _create_group(dict_group, "values")
        Serializable.encode(
            list(to_encode.values()),
            format="hdf5",
            encode_cache=encode_cache,
            ignore_no_serialize_flags=ignore_no_serialize_flags,
            h5_group=val_group,
        )

        return dict_group

    @staticmethod
    def decode_dict(encoded, decode_cache=None):
        """Decode a dictionary from HDF5 format.

        Deserializes dictionary objects that were serialized using encode_dict.
        Preserves the original dictionary structure and insertion order by
        separately serializing keys and values.

        Parameters
        ----------
        encoded : h5py.Group
            The HDF5 group containing the encoded dictionary.
            Should have a 'dict' subgroup with 'keys' and 'values' subgroups.
        decode_cache : dict, optional
            Dictionary mapping cache IDs to decoded objects for reference resolution.

        Returns
        -------
        dict
            The decoded dictionary object.

        Raises
        ------
        IncorrectDecodableTypeError
            If the encoded object is not a valid dictionary.
        DecodableVersionError
            If the serialization version is not supported.
        """
        # Check if right type
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert "dict" in encoded

        dict_group = encoded["dict"]

        # Check if properly formed
        version = _get_version(dict_group)
        with HDF5Encoder.assert_decode(fatal=True):
            _decode_hdf5_dict_version(version, dict_group)

        key_group = dict_group["keys"]
        with HDF5Encoder.assert_decode(fatal=True):
            assert isinstance(key_group, h5py.Group)
        keys = Serializable.decode(
            key_group, format="hdf5", decode_cache=decode_cache
        )
        with HDF5Encoder.assert_decode(fatal=True):
            assert isinstance(keys, list)

        val_group = dict_group["values"]
        with HDF5Encoder.assert_decode(fatal=True):
            assert isinstance(val_group, h5py.Group)
        vals = Serializable.decode(
            val_group, format="hdf5", decode_cache=decode_cache
        )
        with HDF5Encoder.assert_decode(fatal=True):
            assert isinstance(vals, list)

        return {k: v for k, v in zip(keys, vals)}

    @staticmethod
    def encode_array(to_encode, h5_group=None):
        """Encode NumPy arrays and SciPy sparse matrices to HDF5 format.

        Serializes array data with support for both dense and sparse matrices.
        Uses optimized storage strategies including compression and chunking for
        large arrays, and handles complex numbers by separating real and imaginary parts.

        Parameters
        ----------
        to_encode : EncodableArrays
            The array to encode. Can be a NumPy array (NDArray) or SciPy sparse matrix (SPSArray).
        h5_group : h5py.Group
            The HDF5 group to write the array data to.

        Returns
        -------
        h5py.Group
            The HDF5 group containing the encoded array data.

        Raises
        ------
        ValueError
            If the array type is not supported.
        """
        assert isinstance(to_encode, EncodableArrays)
        assert isinstance(h5_group, h5py.Group)

        matrix_group = _create_group(h5_group, "array")
        matrix_group.attrs["shape"] = to_encode.shape
        matrix_group.attrs["dtype"] = str(to_encode.dtype)  # type: ignore

        if isinstance(to_encode, SPSArray):
            # For dense arrays, store as HDF5 dataset
            csr_mx = sps.csr_matrix(
                to_encode
            )  # convert to CSR and save in this format

            matrix_group.attrs["array_type"] = "sparse_csr"

            # Store sparse matrix components as separate datasets
            # Use compression and chunking for large sparse arrays
            data_size = len(csr_mx.data)
            indices_size = len(csr_mx.indices)
            indptr_size = len(csr_mx.indptr)

            if data_size > 1000:
                matrix_group.create_dataset(
                    "data", data=csr_mx.data, compression="gzip"
                )
            else:
                matrix_group.create_dataset("data", data=csr_mx.data)

            if indices_size > 1000:
                matrix_group.create_dataset(
                    "indices", data=csr_mx.indices, compression="gzip"
                )
            else:
                matrix_group.create_dataset("indices", data=csr_mx.indices)

            if indptr_size > 1000:
                matrix_group.create_dataset(
                    "indptr", data=csr_mx.indptr, compression="gzip"
                )
            else:
                matrix_group.create_dataset("indptr", data=csr_mx.indptr)
        elif isinstance(to_encode, NDArray):
            # Determine if array is large enough for compression and chunking
            total_elements = to_encode.size
            use_compression = total_elements > 1000

            if np.iscomplexobj(to_encode):
                # Handle complex numbers by storing real and imaginary parts separately
                matrix_group.attrs["array_type"] = "dense_complex"

                # Apply compression and chunking for large arrays
                if use_compression:
                    # Calculate reasonable chunk size - aim for ~100KB chunks
                    element_size = to_encode.dtype.itemsize
                    target_chunk_elements = max(1000, 100000 // element_size)
                    chunk_shape = tuple(
                        min(dim, target_chunk_elements)
                        for dim in to_encode.shape
                    )

                    matrix_group.create_dataset(
                        "real",
                        data=np.real(to_encode),
                        compression="gzip",
                        chunks=chunk_shape,
                    )
                    matrix_group.create_dataset(
                        "imag",
                        data=np.imag(to_encode),
                        compression="gzip",
                        chunks=chunk_shape,
                    )
                else:
                    matrix_group.create_dataset(
                        "real", data=np.real(to_encode)
                    )
                    matrix_group.create_dataset(
                        "imag", data=np.imag(to_encode)
                    )
            else:
                # For real-valued arrays, store directly as dataset
                matrix_group.attrs["array_type"] = "dense_real"

                # Apply compression and chunking for large arrays
                if use_compression:
                    # Calculate reasonable chunk size - aim for ~100KB chunks
                    element_size = to_encode.dtype.itemsize
                    target_chunk_elements = max(1000, 100000 // element_size)
                    chunk_shape = tuple(
                        min(dim, target_chunk_elements)
                        for dim in to_encode.shape
                    )

                    matrix_group.create_dataset(
                        "data",
                        data=to_encode,
                        compression="gzip",
                        chunks=chunk_shape,
                    )
                else:
                    matrix_group.create_dataset("data", data=to_encode)
        else:
            raise ValueError(
                f"Type {type(to_encode)} not handled by encode_array"
            )

        return matrix_group

    @staticmethod
    def decode_array(encoded):
        """Deserialize matrices."""
        # Check if right type
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert "array" in encoded

        array_group = encoded["array"]

        # Check if properly formed
        version = _get_version(array_group)
        with HDF5Encoder.assert_decode(fatal=True):
            _decode_hdf5_array_version(version, array_group)

        array_type = array_group.attrs.get("array_type", None)
        if array_type == "sparse_csr":
            with HDF5Encoder.assert_decode(fatal=True):
                data = array_group["data"]
                assert isinstance(data, h5py.Dataset)
                indices = array_group["indices"]
                assert isinstance(indices, h5py.Dataset)
                indptr = array_group["indptr"]
                assert isinstance(indptr, h5py.Dataset)

            return sps.csr_matrix(
                (data[()], indices[()], indptr[()]),
                shape=array_group.attrs["shape"],
                dtype=array_group.attrs["dtype"],
            )
        elif array_type == "dense_complex":
            # Reconstruct complex array
            with HDF5Encoder.assert_decode(fatal=True):
                real = array_group["real"]
                assert isinstance(real, h5py.Dataset)
                imag = array_group["imag"]
                assert isinstance(imag, h5py.Dataset)

            decoded = real[()] + 1j * imag[()]
            decoded = decoded.reshape(array_group.attrs["shape"])
            decoded = decoded.astype(array_group.attrs["dtype"])
            return decoded
        else:
            # Dense real
            with HDF5Encoder.assert_decode(fatal=True):
                data = array_group["data"]
                assert isinstance(data, h5py.Dataset)

            decoded = data[()]
            decoded = decoded.reshape(array_group.attrs["shape"])
            decoded = decoded.astype(array_group.attrs["dtype"])
            return decoded

    @staticmethod
    def encode_class(to_encode, h5_group=None):
        """Serialize a class/type."""
        assert isinstance(h5_group, h5py.Group)

        class_group = _create_group(h5_group, "class")
        class_group.attrs["module"] = to_encode.__module__
        class_group.attrs["class"] = to_encode.__name__
        return class_group

    @staticmethod
    def decode_class(encoded) -> type:
        """Decode a class/type from HDF5 format.

        Deserializes a class reference that was serialized using encode_class.
        This allows for proper reconstruction of class types during deserialization.

        Parameters
        ----------
        encoded : h5py.Group
            The HDF5 group containing the encoded class information.
            Should have a 'class' subgroup with 'module' and 'class' attributes.

        Returns
        -------
        type
            The decoded class/type object.

        Raises
        ------
        IncorrectDecodableTypeError
            If the encoded object is not a valid class reference.
        DecodableVersionError
            If the serialization version is not supported.
        ImportError
            If the class cannot be imported from the specified module.
        """
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert "class" in encoded

        class_group = encoded["class"]

        version = _get_version(class_group)
        with HDF5Encoder.assert_decode(fatal=True):
            _decode_hdf5_class_version(version, class_group)

        # Get the class
        return Serializable._import_class(
            class_group.attrs["module"],
            class_group.attrs["class"],
            version,
        )

    @staticmethod
    def encode_function(to_encode, h5_group=None):
        """Serialize a callable function."""
        assert callable(to_encode)
        assert isinstance(h5_group, h5py.Group)

        full_src = Serializable._get_function_str(to_encode)

        function_group = _create_group(h5_group, "function")
        function_group.create_dataset("source", data=full_src)
        return function_group

    @staticmethod
    def decode_function(encoded):
        """Decode a callable function from HDF5 format.

        Deserializes function objects that were serialized using encode_function.
        Reconstructs the function by evaluating its source code in an appropriate
        environment with necessary imports.

        Parameters
        ----------
        encoded : h5py.Group
            The HDF5 group containing the encoded function.
            Should have a 'function' subgroup with 'source' dataset containing
            the function's source code.

        Returns
        -------
        callable
            The decoded function object.

        Raises
        ------
        IncorrectDecodableTypeError
            If the encoded object is not a valid function.
        DecodableVersionError
            If the serialization version is not supported.
        """
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert "function" in encoded

        function_group = encoded["function"]

        version = _get_version(function_group)
        with HDF5Encoder.assert_decode(fatal=True):
            source = _decode_hdf5_function_version(version, function_group)

        return Serializable._eval_function_str(source, version)

    @staticmethod
    def encode_primitive(to_encode, h5_group=None):
        """Encode a primitive value in HDF5 format.

        Serializes primitive Python types (int, float, bool, complex, None, str, bytes)
        to HDF5 format. Handles type preservation and special cases like None values.

        Parameters
        ----------
        to_encode : EncodablePrimitives
            The primitive value to encode. Can be int, float, bool, complex, None, str, or bytes.
        h5_group : h5py.Group
            The HDF5 group to write the primitive value to.

        Returns
        -------
        h5py.Group
            The HDF5 group containing the encoded primitive value.
        """
        assert isinstance(to_encode, EncodablePrimitives)
        assert isinstance(h5_group, h5py.Group)

        h5_group.attrs["encode_type"] = "primitive"

        if isinstance(to_encode, bool):
            # Checked before `int`, since `bool` is a subclass of `int`.
            h5_group.attrs["cast_to"] = "bool"
        elif isinstance(to_encode, int):
            h5_group.attrs["cast_to"] = "int"
        elif isinstance(to_encode, float):
            h5_group.attrs["cast_to"] = "float"
        elif isinstance(to_encode, complex):
            h5_group.attrs["cast_to"] = "complex"
        elif to_encode is None:
            h5_group.attrs["is_none"] = True
            to_encode = 0

        h5_group.attrs["value"] = to_encode

        return h5_group

    @staticmethod
    def decode_primitive(encoded):
        """Decode a primitive value from HDF5 format.

        Deserializes primitive Python types that were serialized using encode_primitive.
        Handles type reconstruction and special cases like None values and bytes encoding.

        Parameters
        ----------
        encoded : h5py.Group
            The HDF5 group containing the encoded primitive value.
            Should have 'encode_type' attribute set to 'primitive'.

        Returns
        -------
        EncodablePrimitives
            The decoded primitive value (int, float, bool, complex, None, str, or bytes).

        Raises
        ------
        IncorrectDecodableTypeError
            If the encoded object is not a valid primitive.
        DecodableVersionError
            If the serialization version is not supported.
        ValueError
            If the decoded primitive type is unexpected.
        """
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert encoded.attrs.get("encode_type", "") == "primitive"

        version = _get_version(encoded)
        with HDF5Encoder.assert_decode(fatal=True):
            _decode_hdf5_primitive_version(version, encoded)

        if encoded.attrs.get("is_none", False):
            return None

        value = encoded.attrs["value"]

        # Handle bytes to string conversion for HDF5 stored strings
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, keep as bytes
                pass
        if not isinstance(value, EncodablePrimitives):
            raise ValueError(
                f"Unexpected decoded primitive type {type(value)}"
            )

        # Handle any requested casting out of numpy types
        cast_to = encoded.attrs.get("cast_to", None)
        if cast_to is not None:
            cast_type = __builtins__.get(cast_to, None)
            if cast_type is not None:
                return cast_type(value)

        return value
