#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

from typing import ClassVar

import h5py
import numpy as np
import scipy.sparse as sps

from loqs.internal.serializable import (
    DecodableVersionError,
    DecodeCache,
    Encodable,
    EncodableArrays,
    EncodableIterables,
    EncodablePrimitives,
    Encoded,
)
from loqs.types import NDArray, SPSArray
from loqs.internal import Serializable, SERIALIZATION_VERSION
from loqs.internal.encoder import BaseEncoder


class HDF5Encoder(BaseEncoder):

    ENCODE_ID: ClassVar[int] = 0

    @staticmethod
    def decode_root_group(
        encoded: Encoded,
        decode_cache: DecodeCache = None,
    ) -> Encodable:
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert len(encoded.keys()) == 1
            assert not encoded.attrs

        # Get the first (and only) subgroup
        subgroup_name = list(encoded.keys())[0]
        subgroup = encoded[subgroup_name]
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(subgroup, h5py.Group)

        # Try to decode the subgroup
        # It could either error out or give IncorrectDecodableTypeError
        # Both are acceptable
        return Serializable.decode(
            subgroup, format="hdf5", decode_cache=decode_cache
        )

    @staticmethod
    def encode_uncached_obj(
        to_encode,
        encode_cache=None,
        ignore_no_serialize_flags=False,
        h5_group=None,
    ):
        assert isinstance(to_encode, Serializable)
        assert isinstance(h5_group, h5py.Group)

        obj_group = h5_group.create_group(
            f"Serializable_{HDF5Encoder.ENCODE_ID}"
        )
        obj_group.attrs["encode_type"] = "Serializable"
        obj_group.attrs["module"] = to_encode.__class__.__module__
        obj_group.attrs["class"] = to_encode.__class__.__name__
        obj_group.attrs["version"] = SERIALIZATION_VERSION

        # Use SERIALIZE_ATTRS pattern for encoding
        for serial_attr in to_encode.SERIALIZE_ATTRS:
            attr_value = to_encode.get_encoding_attr(
                serial_attr,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
            )
            attr_group = obj_group.create_group(serial_attr)

            Serializable.encode(
                attr_value,
                format="hdf5",
                encode_cache=encode_cache,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
                h5_group=attr_group,
            )

        # Add to cache if class marked as should be cached
        if encode_cache is not None and to_encode.CACHE_ON_SERIALIZE:
            object_id = id(to_encode)
            encode_cache[object_id] = HDF5Encoder.ENCODE_ID
            obj_group.attrs["cache_type"] = "source"
            obj_group.attrs["cache_id"] = HDF5Encoder.ENCODE_ID

        HDF5Encoder.ENCODE_ID += 1

        return obj_group

    @staticmethod
    def decode_uncached_obj(encoded, decode_cache=None):
        # Check if right type
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert encoded.attrs.get("encode_type", "") == "Serializable"

        # Check if properly formed
        with HDF5Encoder.assert_decode(fatal=True):
            assert "module" in encoded.attrs
            assert "class" in encoded.attrs
            version = encoded.attrs.get("version", -1)
            if version != 1:
                raise DecodableVersionError()

        # Get the class
        cls = Serializable.import_class(
            encoded.attrs["module"], encoded.attrs["class"], version
        )

        # Create the attribute dictionary for deserialization
        attr_dict = {}
        for key in encoded.keys():
            obj_group = encoded[key]
            assert isinstance(obj_group, h5py.Group)

            attr_dict[key] = Serializable.decode(
                obj_group, format="hdf5", decode_cache=decode_cache
            )

        # If our class is an Instruction, we also need to pass in version
        # so that imports can be updated properly on apply_fn/map_qubits_fn creation
        from loqs.core import Instruction

        if cls == Instruction:
            attr_dict["version"] = version

        # Create the object using its from_decoded_attrs method
        decoded = cls.from_decoded_attrs(attr_dict)

        # Save new object in cache if it is a source
        if encoded.attrs.get("cache_type", None) == "source":
            try:
                cache_id = encoded.attrs["cache_id"]
                decode_cache[cache_id] = decoded  # type: ignore
            except (KeyError, TypeError):
                raise RuntimeError("Failed to look up cache source")

        return decoded

    @staticmethod
    def encode_cached_obj(cache_id, h5_group=None):
        assert isinstance(cache_id, int)
        assert isinstance(h5_group, h5py.Group)

        obj_group = h5_group.create_group(
            f"Serializable_{HDF5Encoder.ENCODE_ID}"
        )
        HDF5Encoder.ENCODE_ID += 1
        obj_group.attrs["encode_type"] = "Serializable"
        obj_group.attrs["version"] = SERIALIZATION_VERSION
        obj_group.attrs["cache_type"] = "reference"
        obj_group.attrs["cache_id"] = cache_id
        return obj_group

    @staticmethod
    def decode_cached_obj(encoded, decode_cache=None):
        # Check if right type
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert encoded.attrs.get("encode_type", "") == "Serializable"
            assert encoded.attrs.get("cache_type", "") == "reference"

        # Check if properly formed
        with HDF5Encoder.assert_decode(fatal=True):
            assert "cache_id" in encoded.attrs
            version = encoded.attrs.get("version", -1)
            if version != 1:
                raise DecodableVersionError()

        try:
            cache_id = encoded.attrs["cache_id"]
            return decode_cache[cache_id]  # type: ignore
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

        # TODO: This can be made more efficient as dataset with correct dtype/casting
        list_group = h5_group.create_group("iterable")
        list_group.attrs["iterable_type"] = name
        list_group.attrs["version"] = SERIALIZATION_VERSION
        for i, e in enumerate(to_encode):
            item_group = list_group.create_group(str(i))
            Serializable.encode(
                e,
                format="hdf5",
                encode_cache=encode_cache,
                ignore_no_serialize_flags=ignore_no_serialize_flags,
                h5_group=item_group,
            )
        return list_group

    @staticmethod
    def decode_iterable(encoded, decode_cache=None):
        # Check if right type
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert "iterable" in encoded

        list_group = encoded["iterable"]

        # Check if properly formed
        with HDF5Encoder.assert_decode(fatal=True):
            assert isinstance(list_group, h5py.Group)
            assert list_group.attrs.get("iterable_type", "") in [
                "list",
                "tuple",
                "set",
            ]
            version = list_group.attrs.get("version", -1)
            if version != 1:
                raise DecodableVersionError()

        value = []
        for i in range(len(list_group.keys())):
            with HDF5Encoder.assert_decode(fatal=True):
                assert str(i) in list_group

            item_group = list_group[str(i)]
            with HDF5Encoder.assert_decode(fatal=True):
                assert isinstance(item_group, h5py.Group)

            value.append(
                Serializable.decode(
                    item_group, format="hdf5", decode_cache=decode_cache
                )
            )

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

        dict_group = h5_group.create_group("dict")
        dict_group.attrs["version"] = SERIALIZATION_VERSION

        # Store keys and values in order to preserve dict insertion order
        key_group = dict_group.create_group("keys")
        Serializable.encode(
            list(to_encode.keys()),
            format="hdf5",
            encode_cache=encode_cache,
            ignore_no_serialize_flags=ignore_no_serialize_flags,
            h5_group=key_group,
        )

        val_group = dict_group.create_group("values")
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
        # Check if right type
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert "dict" in encoded

        dict_group = encoded["dict"]

        # Check if properly formed
        with HDF5Encoder.assert_decode(fatal=True):
            assert isinstance(dict_group, h5py.Group)
            assert "keys" in dict_group
            assert "values" in dict_group
            version = dict_group.attrs.get("version", -1)
            if version != 1:
                raise DecodableVersionError()

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
        """Serialize NumPy arrays and SciPy sparse matrices."""
        assert isinstance(to_encode, EncodableArrays)
        assert isinstance(h5_group, h5py.Group)

        # TODO: This could be made more efficient,
        # especially with compression and chunks

        matrix_group = h5_group.create_group("array")
        matrix_group.attrs["version"] = SERIALIZATION_VERSION
        matrix_group.attrs["shape"] = to_encode.shape
        matrix_group.attrs["dtype"] = str(to_encode.dtype)  # type: ignore

        if isinstance(to_encode, SPSArray):
            # For dense arrays, store as HDF5 dataset
            csr_mx = sps.csr_matrix(
                to_encode
            )  # convert to CSR and save in this format

            matrix_group.attrs["array_type"] = "sparse_csr"

            # Store sparse matrix components as separate datasets
            matrix_group.create_dataset("data", data=csr_mx.data)
            matrix_group.create_dataset("indices", data=csr_mx.indices)
            matrix_group.create_dataset("indptr", data=csr_mx.indptr)
        elif isinstance(to_encode, NDArray):
            if np.iscomplexobj(to_encode):
                # Handle complex numbers by storing real and imaginary parts separately
                matrix_group.attrs["array_type"] = "dense_complex"

                matrix_group.create_dataset("real", data=np.real(to_encode))
                matrix_group.create_dataset("imag", data=np.imag(to_encode))
            else:
                # For real-valued arrays, store directly as dataset
                matrix_group.attrs["array_type"] = "dense_real"
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
        with HDF5Encoder.assert_decode(fatal=True):
            assert isinstance(array_group, h5py.Group)
            assert "shape" in array_group.attrs
            assert "dtype" in array_group.attrs
            version = array_group.attrs.get("version", -1)
            if version != 1:
                raise DecodableVersionError()
            array_type = array_group.attrs.get("array_type", None)
            assert array_type in ["sparse_csr", "dense_complex", "dense_real"]

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

        class_group = h5_group.create_group("class")
        class_group.attrs["module"] = to_encode.__module__
        class_group.attrs["class"] = to_encode.__name__
        class_group.attrs["version"] = SERIALIZATION_VERSION
        return class_group

    @staticmethod
    def decode_class(encoded) -> type:
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert "class" in encoded

        class_group = encoded["class"]

        with HDF5Encoder.assert_decode(fatal=True):
            assert isinstance(class_group, h5py.Group)
            assert "module" in class_group.attrs
            assert "class" in class_group.attrs
            version = class_group.attrs.get("version", -1)
            if version != 1:
                raise DecodableVersionError()

        # Get the class
        return Serializable.import_class(
            class_group.attrs["module"],
            class_group.attrs["class"],
            class_group.attrs["version"],
        )

    @staticmethod
    def encode_function(to_encode, h5_group=None):
        """Serialize a callable function."""
        assert callable(to_encode)
        assert isinstance(h5_group, h5py.Group)

        full_src = Serializable.get_function_str(to_encode)

        function_group = h5_group.create_group("function")
        function_group.attrs["version"] = SERIALIZATION_VERSION
        function_group.create_dataset("source", data=full_src)
        return function_group

    @staticmethod
    def decode_function(encoded):
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert "function" in encoded

        function_group = encoded["function"]

        with HDF5Encoder.assert_decode(fatal=True):
            assert isinstance(function_group, h5py.Group)
            version = function_group.attrs.get("version", -1)
            if version != 1:
                raise DecodableVersionError()
            assert "source" in function_group
            source_dataset = function_group["source"]
            assert isinstance(source_dataset, h5py.Dataset)
            source = source_dataset[()]
            if isinstance(source, bytes):
                source = source.decode("utf-8")
            else:
                source = str(source)
            assert isinstance(source, str)

        return Serializable.eval_function_str(source, version)

    @staticmethod
    def encode_primitive(to_encode, h5_group=None):
        assert isinstance(to_encode, EncodablePrimitives)
        assert isinstance(h5_group, h5py.Group)

        h5_group.attrs["encode_type"] = "primitive"
        h5_group.attrs["version"] = SERIALIZATION_VERSION

        if isinstance(to_encode, int):
            h5_group.attrs["cast_to"] = "int"
        elif isinstance(to_encode, float):
            h5_group.attrs["cast_to"] = "float"
        elif isinstance(to_encode, bool):
            h5_group.attrs["cast_to"] = "bool"
        elif isinstance(to_encode, complex):
            h5_group.attrs["cast_to"] = "complex"
        elif to_encode is None:
            h5_group.attrs["is_none"] = True
            to_encode = 0

        h5_group.attrs["value"] = to_encode

        return h5_group

    @staticmethod
    def decode_primitive(encoded):
        with HDF5Encoder.assert_decode(fatal=False):
            assert isinstance(encoded, h5py.Group)
            assert encoded.attrs.get("encode_type", "") == "primitive"

        with HDF5Encoder.assert_decode(fatal=True):
            assert "value" in encoded.attrs
            version = encoded.attrs.get("version", -1)
            if version != 1:
                raise DecodableVersionError()

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
