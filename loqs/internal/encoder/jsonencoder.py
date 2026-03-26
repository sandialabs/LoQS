#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

import h5py
import numpy as np
import scipy.sparse as sps

from loqs.internal.serializable import (
    DecodableVersionError,
    EncodablePrimitives,
    MisformedDecodableError,
)
from loqs.types import Float, Int
from loqs.internal import Serializable, SERIALIZATION_VERSION
from loqs.internal.encoder import BaseEncoder


class JSONEncoder(BaseEncoder):

    @staticmethod
    def encode_uncached_obj(
        to_encode,
        encode_cache=None,
        ignore_no_serialize_flags=False,
        h5_group=None,
    ):
        assert isinstance(to_encode, Serializable)

        state = {
            "encode_type": "Serializable",
            "module": to_encode.__class__.__module__,
            "class": to_encode.__class__.__name__,
            "version": SERIALIZATION_VERSION,
        }

        # Encode attributes using the standard JSON encode method
        for serial_attr in to_encode.SERIALIZE_ATTRS:
            attr_value = to_encode.get_encoding_attr(
                serial_attr,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
            )

            state[serial_attr] = Serializable.encode(
                attr_value,
                format="json",
                encode_cache=encode_cache,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
            )

        # Add to cache if class marked as should be cached
        if encode_cache is not None and to_encode.CACHE_ON_SERIALIZE:
            object_id = id(to_encode)
            encode_cache[object_id] = JSONEncoder.ENCODE_ID
            state.update(
                {
                    "cache_type": "source",
                    "cache_id": JSONEncoder.ENCODE_ID,
                }
            )
            JSONEncoder.ENCODE_ID += 1

        return state

    @staticmethod
    def decode_uncached_obj(encoded, decode_cache=None) -> Serializable | dict:  # type: ignore
        # Check if right type
        with JSONEncoder.assert_decode(fatal=False):
            assert isinstance(encoded, dict)
            # Version 0 compatibility, array dict doesn't have version in it
            assert not encoded.get("type", "") == "matrix"
            version = encoded.get("version", -1)
            if version == 0:
                # Do not have encode_type, check for module/class but no as_type
                assert "module" in encoded
                assert "class" in encoded
                assert not encoded.get("as_type", False)
            elif version == 1:
                # Can check for encode_type
                assert encoded.get("encode_type", "") == "Serializable"
            else:
                raise DecodableVersionError()

        # Check if properly formed
        with JSONEncoder.assert_decode(fatal=True):
            assert "module" in encoded
            assert "class" in encoded
            # Cache type and id are not strictly required

        # pygsti also serialized objects that look exactly like this
        # For version 0 backwards compatibility, we cannot differentiate
        # except for module name being in pygsti
        if encoded["module"].startswith("pygsti"):
            # in this case, we want to return the raw dict as successful,
            # assuming that it is being deserialized elsewhere,
            # e.g. "model" in PyGSTiNoiseModel decoding
            return encoded

        # Get the class
        cls = Serializable.import_class(
            encoded["module"], encoded["class"], version
        )

        # Create the attribute dictionary for deserialization
        attr_dict = {}
        for k, v in encoded.items():
            # Skip metadata keys
            if version == 0:
                metadata_keys = [
                    "module",
                    "class",
                    "version",
                    "type",
                    "cache_id",
                ]

                # Other compatibility: QSimQuantumState internal data
                # should be matrix but might be encoded as a list
                # We need matrix handling to cast "U" type back to complex
                if k == "_qsim_dm_data" and isinstance(v, list):
                    v = {"type": "matrix", "data": v}
            else:
                metadata_keys = [
                    "module",
                    "class",
                    "version",
                    "encode_type",
                    "cache_type",
                    "cache_id",
                ]

            if k in metadata_keys:
                continue

            attr_dict[k] = Serializable.decode(
                v, format="json", decode_cache=decode_cache
            )

        # If our class is an Instruction, we also need to pass in version
        # so that imports can be updated properly on apply_fn/map_qubits_fn creation
        from loqs.core import Instruction

        if cls == Instruction:
            attr_dict["version"] = version

            # If we are version 0, we actually need to add type back in as well
            if version == 0:
                attr_dict["type"] = encoded["type"]

        # Create the object using its from_decoded_attrs method
        decoded = cls.from_decoded_attrs(attr_dict)

        # Save new object in cache if it is a source
        if (
            version == 0 and encoded.get("type", "") == "cached_object_source"
        ) or encoded.get("cache_type", "") == "source":
            try:
                cache_id = encoded["cache_id"]
                decode_cache[cache_id] = decoded  # type: ignore
            except (KeyError, TypeError):
                raise RuntimeError("Failed to look up cache source")

        return decoded

    @staticmethod
    def encode_cached_obj(cache_id, h5_group=None):
        return {
            "encode_type": "Serializable",
            "version": SERIALIZATION_VERSION,
            "cache_type": "reference",
            "cache_id": cache_id,
        }

    @staticmethod
    def decode_cached_obj(encoded, decode_cache=None):
        # Check if right type
        with JSONEncoder.assert_decode(fatal=False):
            assert isinstance(encoded, dict)
            assert "version" in encoded
            version = encoded["version"]
            if version == 0:
                # Do not have encode_type, check for module/class but no as_type
                assert "module" in encoded
                assert "class" in encoded
                assert not encoded.get("as_type", False)
                assert encoded.get("type", "") == "cached_object_reference"
            elif version == 1:
                # Can check for encode_type
                assert encoded.get("encode_type", "") == "Serializable"
                assert encoded.get("cache_type", "") == "reference"
            else:
                raise DecodableVersionError()

        # Check if properly formed
        with JSONEncoder.assert_decode(fatal=True):
            assert "cache_id" in encoded

        try:
            assert decode_cache is not None
            return decode_cache[encoded["cache_id"]]
        except AssertionError:
            raise RuntimeError("Object reference found but no cache provided.")
        except KeyError:
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

        encoded_items = []
        for item in to_encode:
            encoded_item = Serializable.encode(
                item,
                format="json",
                encode_cache=encode_cache,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
                h5_group=None,
            )
            encoded_items.append(encoded_item)

        return {
            "encode_type": "iterable",
            "version": SERIALIZATION_VERSION,
            "iterable_type": name,
            "items": encoded_items,
        }

    @staticmethod
    def decode_iterable(encoded, decode_cache=None):
        # Check if right type
        with JSONEncoder.assert_decode(fatal=False):
            if isinstance(encoded, (list, tuple)):
                # We must assume that we are reading version 0 iterable
                version = 0
            else:
                assert isinstance(encoded, dict)
                version = encoded.get("version", -1)
                if version == 1:
                    assert encoded.get("encode_type", "") == "iterable"
                else:
                    raise DecodableVersionError()

        # Check if properly formed
        if version == 1:
            with JSONEncoder.assert_decode(fatal=True):
                iter_type = encoded.get("iterable_type", "")
                assert iter_type in ["list", "tuple", "set"]
                assert isinstance(encoded.get("items", None), list)
                items_to_decode = encoded["items"]
        else:
            # Version 0, bare list and no casting information
            items_to_decode = encoded
            iter_type = "list"

        # Decode all items
        items = []
        for item in items_to_decode:  # type: ignore
            decoded_item = Serializable.decode(
                item, format="json", decode_cache=decode_cache
            )
            items.append(decoded_item)

        # Return the appropriate type
        if iter_type == "tuple":
            return tuple(items)
        elif iter_type == "set":
            return set(items)
        else:
            return items

    @staticmethod
    def encode_dict(
        to_encode,
        encode_cache=None,
        ignore_no_serialize_flags=False,
        h5_group=None,
    ):
        """
        Encode a dictionary (JSON version).

        Parameters
        ----------
        d : dict
            The dictionary to encode.
        h5_group : Any
            The storage group/object to write to (ignored for JSON).
        encode_cache : dict
            Dictionary mapping object hashes to serialization IDs.
        ignore_no_serialize_flags : bool
            Whether to ignore serialization flags.

        Returns
        -------
        dict
            The JSON dictionary with the encoded dictionary.
        """
        encoded_dict = {
            "encode_type": "dict",
            "version": SERIALIZATION_VERSION,
            "items": {},
        }
        for k, v in to_encode.items():
            # Handle tuple keys by converting to strings
            if isinstance(k, tuple):
                encoded_k = str(k)
            else:
                encoded_k = k
            encoded_dict["items"][encoded_k] = Serializable.encode(
                v,
                format="json",
                encode_cache=encode_cache,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
            )
        return encoded_dict

    @staticmethod
    def decode_dict(encoded, decode_cache=None):
        # Check if right type
        with JSONEncoder.assert_decode(fatal=False):
            assert isinstance(encoded, dict)

            version = encoded.get("version", -1)
            if version in [-1, 0]:
                # Don't be Serializable or class
                assert "module" not in encoded
                assert "class" not in encoded
                # Don't be array
                assert encoded.get("type", "") != "matrix"

                # Else, version 0 assumes we are all good to proceed
                # Unfortuantely version 0 also did not version dicts,
                # so version 0 will act as our catch-all for dicts
            elif version == 1:
                assert encoded.get("encode_type", "") == "dict"
            else:
                DecodableVersionError()

        # Check if properly formed
        if version == 1:
            with JSONEncoder.assert_decode(fatal=True):
                assert isinstance(encoded.get("items", None), dict)
                items_to_decode = encoded["items"]
                assert isinstance(items_to_decode, dict)
        else:
            items_to_decode = encoded

        decoded_dict = {}
        for k, v in items_to_decode.items():

            # Handle tuple keys
            if isinstance(k, str) and k.startswith("(") and k.endswith(")"):
                import ast

                try:
                    decoded_k = ast.literal_eval(k)
                except (ValueError, SyntaxError):
                    decoded_k = k
            else:
                decoded_k = k

            decoded_v = Serializable.decode(
                v, format="json", decode_cache=decode_cache
            )
            decoded_dict[decoded_k] = decoded_v

        return decoded_dict

    @staticmethod
    def encode_array(to_encode, h5_group=None):

        def _serialize_matrix_component(arr):
            """Inline helper for serializing matrix components."""
            if arr is None:
                return None
            if np.iscomplexobj(arr):
                return [str(x) for x in arr.flatten().tolist()]
            elif arr.dtype == Int:
                return [int(x) for x in arr.flatten().tolist()]
            else:
                return arr.flatten().tolist()

        if sps.issparse(to_encode):
            csr_mx = sps.csr_matrix(to_encode)
            array_data = {
                "sparse_matrix_type": "csr",
                "data": _serialize_matrix_component(csr_mx.data),
                "indices": _serialize_matrix_component(csr_mx.indices),
                "indptr": _serialize_matrix_component(csr_mx.indptr),
                "dtype": str(csr_mx.dtype),
                "shape": csr_mx.shape,
            }
        elif isinstance(to_encode, np.ndarray):
            array_data = {
                "data": _serialize_matrix_component(to_encode),
                "shape": to_encode.shape,
                "dtype": str(to_encode.dtype),
            }
        else:
            raise ValueError(
                f"Type {type(to_encode)} not handled by encode_array"
            )

        array_data["encode_type"] = "array"
        array_data["version"] = SERIALIZATION_VERSION

        return array_data

    @staticmethod
    def decode_array(encoded):
        """
        Decode matrix data (JSON version).

        Parameters
        ----------
        json_dict : dict
            The JSON dictionary containing the matrix.

        Returns
        -------
        Any
            The decoded matrix.
        """
        # Check if right type
        with JSONEncoder.assert_decode(fatal=False):
            assert isinstance(encoded, dict)
            # module and class being in here means its a Serializable
            assert "module" not in encoded
            assert "class" not in encoded
            version = encoded.get("version", -1)
            if version == -1 and encoded.get("type", "") == "matrix":
                # Version 0 assumes any dict with type: matrix is an array type
                version = 0
            elif version == 1:
                assert encoded.get("encode_type", "") == "array"
            else:
                raise DecodableVersionError()

        # Arrays are one of the things that changed most version 0->1
        if version == 0:

            def _deserialize_mx_v0(mx):
                if isinstance(mx, dict):  # then a sparse mx
                    assert mx["sparse_matrix_type"] == "csr"
                    data = _deserialize_mx_v0(mx["data"])
                    indices = _deserialize_mx_v0(mx["indices"])
                    indptr = _deserialize_mx_v0(mx["indptr"])
                    decoded = sps.csr_matrix(
                        (data, indices, indptr), shape=mx["shape"]
                    )
                else:
                    basemx = np.array(mx)
                    if (
                        basemx.dtype.kind == "U"
                    ):  # character type array => complex numbers as strings
                        decoded = np.array([complex(x) for x in basemx.flat])
                        decoded = decoded.reshape(basemx.shape)
                    else:
                        decoded = basemx
                return decoded

            return _deserialize_mx_v0(encoded["data"])

        # Otherwise, continue with v1 code
        with JSONEncoder.assert_decode(fatal=True):
            assert "data" in encoded
            assert "shape" in encoded
            assert "dtype" in encoded

        def _deserialize_matrix(mx, dtype, shape=None) -> np.ndarray:
            decoded = np.array(mx)
            if decoded.dtype.kind == "U":
                # character type array => complex numbers as strings
                decoded = np.array([complex(x) for x in decoded.flat])
            if shape is not None:
                decoded = decoded.reshape(shape)
            return decoded.astype(dtype)

        shape = encoded["shape"]
        dtype = np.dtype(getattr(np, encoded["dtype"]))  # type: ignore

        if isinstance(encoded, dict) and "sparse_matrix_type" in encoded:
            if encoded["sparse_matrix_type"] == "csr":
                data = _deserialize_matrix(encoded["data"], dtype)
                indices = _deserialize_matrix(encoded["indices"], dtype)
                indptr = _deserialize_matrix(encoded["indptr"], dtype)
                decoded = sps.csr_matrix((data, indices, indptr), shape=shape)
            else:
                raise MisformedDecodableError("Invalid sparse_matrix_type")
        else:
            decoded = _deserialize_matrix(encoded["data"], dtype, shape)

        return decoded

    @staticmethod
    def encode_class(to_encode, h5_group=None):
        """
        Encode a class/type (JSON version).

        Parameters
        ----------
        to_encode : type
            The class/type to encode.
        h5_group : Any
            The storage group/object to write to (ignored for JSON).

        Returns
        -------
        dict
            The JSON dictionary with the encoded class.
        """
        assert isinstance(to_encode, type)
        return {
            "encode_type": "class",
            "module": to_encode.__module__,
            "class": to_encode.__name__,
            "version": SERIALIZATION_VERSION,
        }

    @staticmethod
    def decode_class(encoded):
        """
        Decode a class/type (JSON version).

        Parameters
        ----------
        json_dict : dict
            The JSON dictionary containing the class.

        Returns
        -------
        type
            The decoded class.
        """
        # Check if right type
        with JSONEncoder.assert_decode(fatal=False):
            assert isinstance(encoded, dict)
            version = encoded.get("version", -1)
            if version == 0:
                assert "module" in encoded
                assert "class" in encoded
                assert encoded.get("as_type", False)
            elif version == 1:
                assert encoded.get("encode_type", "") == "class"
            else:
                raise DecodableVersionError()

        # Check if properly formed
        with JSONEncoder.assert_decode(fatal=True):
            assert "module" in encoded
            assert "class" in encoded

        return Serializable.import_class(
            encoded["module"], encoded["class"], version
        )

    @staticmethod
    def encode_function(to_encode, h5_group=None):
        """
        Encode a callable function (JSON version).

        Parameters
        ----------
        func : callable
            The function to encode.
        h5_group : Any
            The storage group/object to write to (ignored for JSON).

        Returns
        -------
        dict
            The JSON dictionary with the encoded function.
        """
        assert callable(to_encode)
        return {
            "encode_type": "function",
            "version": SERIALIZATION_VERSION,
            "source": Serializable.get_function_str(to_encode),
        }

    @staticmethod
    def decode_function(encoded, h5_group=None):
        """
        Decode a callable function (JSON version).

        Parameters
        ----------
        encoded : dict
            The JSON dictionary containing the function.

        Returns
        -------
        callable
            The decoded function.
        """
        # Check if right type
        with JSONEncoder.assert_decode(fatal=False):
            # Version 0 assumes any string with def is a function
            if isinstance(encoded, str) and "def " in encoded:
                version = 0
            else:
                assert isinstance(encoded, dict)
                version = encoded.get("version", -1)
                if version == 1:
                    assert encoded.get("encode_type", "") == "function"
                else:
                    raise DecodableVersionError()

        # Check if properly formed
        if version == 0:
            source = encoded
        else:
            with JSONEncoder.assert_decode(fatal=True):
                assert "source" in encoded
                source = encoded.get("source", None)  # type: ignore

        assert isinstance(source, str)
        return Serializable.eval_function_str(source, version)

    @staticmethod
    def encode_primitive(to_encode, h5_group=None):
        """
        Encode a primitive value (JSON version).

        Parameters
        ----------
        to_encode : Any
            The primitive value to encode.
        h5_group : Any
            The storage group/object to write to (ignored for JSON).

        Returns
        -------
        Any
            The primitive value directly (for JSON).
        """
        if isinstance(to_encode, Int):
            to_encode = int(to_encode)
        elif isinstance(to_encode, Float):
            to_encode = float(to_encode)

        return {
            "encode_type": "primitive",
            "version": SERIALIZATION_VERSION,
            "value": to_encode,
        }

    @staticmethod
    def decode_primitive(encoded):
        """
        Decode a primitive value (JSON version).

        Parameters
        ----------
        json_dict : dict
            The JSON dictionary containing the primitive.

        Returns
        -------
        Any
            The decoded primitive value.
        """
        # Check if right type
        with JSONEncoder.assert_decode(fatal=False):
            if not isinstance(encoded, (dict, h5py.Group)):
                # Must assume this is a version 0 primitive
                # Return immediately
                return encoded

            assert isinstance(encoded, dict)
            version = encoded.get("version", -1)
            if version == 1:
                assert encoded.get("encode_type", None) == "primitive"
            else:
                raise DecodableVersionError()

        # Check if properly formed
        with JSONEncoder.assert_decode(fatal=True):
            assert "version" in encoded
            assert "value" in encoded
            value = encoded["value"]
            assert isinstance(value, EncodablePrimitives)

        return value
