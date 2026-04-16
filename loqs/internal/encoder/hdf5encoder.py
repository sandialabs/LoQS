#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

from typing import ClassVar

import copy
import h5py
import numpy as np
import scipy.sparse as sps

from loqs.internal.serializable import (
    DecodableVersionError,
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


class HDF5Encoder(BaseEncoder):

    ENCODE_ID: ClassVar[int] = 0

    @staticmethod
    def decode_root_group(
        encoded: Encoded,
        decode_cache: DecodeCache = None,
    ) -> Encodable:
        """Decode the root HDF5 group containing a serialized object.

        This method handles the top-level deserialization of HDF5 files by extracting
        the main content group and delegating the actual deserialization to the
        Serializable.decode method. REVIEW_NO_DOCSTRING

        Parameters
        ----------
        encoded : Encoded
            The root HDF5 group containing the serialized object.
            Should contain exactly one subgroup with no attributes.
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

        obj_group = h5_group.create_group(
            f"Serializable_{HDF5Encoder.ENCODE_ID}"
        )
        HDF5Encoder.ENCODE_ID += 1
        obj_group.attrs["encode_type"] = "Serializable"
        obj_group.attrs["version"] = SERIALIZATION_VERSION
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
        with HDF5Encoder.assert_decode(fatal=True):
            version = encoded.attrs.get("version", -1)
            if version != 1:
                raise DecodableVersionError()

            cache_type = encoded.attrs["cache_type"]

            if cache_type == "reference":
                assert "cache_id" in encoded.attrs
            elif cache_type == "copy":
                assert "reference_cache_id" in encoded.attrs
                assert "source_cache_id" in encoded.attrs

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
                copied_obj = copy.deepcopy(reference_obj)

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

        list_group = h5_group.create_group("iterable")
        list_group.attrs["iterable_type"] = name
        list_group.attrs["version"] = SERIALIZATION_VERSION

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
        if first_type in hdf5_native_types and all(
            isinstance(e, first_type) for e in to_encode
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
            # Mixed native types or non-native types - fall back to groups
            list_group.attrs["storage_format"] = "groups"
            for i, e in enumerate(to_encode_list):
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
            # Original format using individual groups
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
