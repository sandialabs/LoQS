#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Callable, ClassVar
import h5py

from loqs.internal.serializable import (
    DeferredRef,
    Serializable,
    Encoded,
    EncodeCache,
    DecodeCache,
    EncodableArrays,
    EncodableIterables,
    EncodablePrimitives,
    IncorrectDecodableTypeError,
    MisformedDecodableError,
)


class BaseEncoder(ABC):
    """
    Abstract base class for serialization implementations.
    """

    ENCODE_ID: ClassVar[int] = 0
    """Internal counter to ensure unique encoding ids.

    This is set back to 0 if `reset_encode_id=True` in
    :meth:`.Serializable.encode`. Internally, this is
    only happens in :meth:`.Serializable.dump`.
    Nothing bad happens if it is not reset, but the cache ids
    will not start at 0 as is "standard".
    """

    @classmethod
    @contextmanager
    def assert_decode(
        cls,
        fatal: bool = True,
    ):
        """Context manager for handling decode assertion errors.

        This context manager wraps decode operations and handles AssertionError
        exceptions by converting them to appropriate error types based on the
        fatal parameter. REVIEW_NO_DOCSTRING

        Parameters
        ----------
        fatal : bool, optional
            If True, raises MisformedDecodableError for assertion failures.
            If False, raises IncorrectDecodableTypeError. Default is True.

        Yields
        ------
        None
            Yields control to the wrapped code block.

        Raises
        ------
        MisformedDecodableError
            When fatal=True and an AssertionError occurs during decoding.
        IncorrectDecodableTypeError
            When fatal=False and an AssertionError occurs during decoding.
        """
        try:
            yield
        except AssertionError as e:
            if fatal:
                # This is a breaking error
                raise MisformedDecodableError(
                    f"Misformed object passed to {cls.__name__}"
                ) from e
            # Otherwise, signal bad type which is caught in Serializable.decode
            raise IncorrectDecodableTypeError(
                f"Object is not decodable by {cls.__name__}"
            ) from e

    ## Abstract methods
    @staticmethod
    @abstractmethod
    def encode_uncached_obj(
        to_encode: Serializable,
        encode_cache: EncodeCache = None,
        ignore_no_serialize_flags: bool = False,
    ):
        """Encode an object that has not been previously cached.

        This abstract method should be implemented by concrete encoder classes
        to handle the encoding of objects that are being encountered for the first
        time during the encoding process. REVIEW_NO_DOCSTRING

        Parameters
        ----------
        to_encode : Serializable
            The object to be encoded.
        encode_cache : EncodeCache, optional
            Cache used to track encoded objects and avoid duplication.
        ignore_no_serialize_flags : bool, optional
            If True, ignores any no-serialize flags on objects. Default is False.

        Returns
        -------
        Encoded
            The encoded representation of the object.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def decode_uncached_obj(
        encoded: Encoded, decode_cache: DecodeCache = None
    ) -> Serializable | DeferredRef:
        """Decode an object that was not previously cached during encoding.

        This abstract method should be implemented by concrete encoder classes
        to handle the decoding of objects that are being encountered for the first
        time during the decoding process. REVIEW_NO_DOCSTRING

        Parameters
        ----------
        encoded : Encoded
            The encoded representation of an uncached object.
        decode_cache : DecodeCache, optional
            Cache used to track decoded objects and resolve references.

        Returns
        -------
        Serializable | DeferredRef
            The decoded object or a deferred reference to be resolved later.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def encode_cached_obj(cache_id: int) -> Encoded:
        """Encode a cached object reference.

        This abstract method should be implemented by concrete encoder classes
        to handle the encoding of objects that have been previously cached
        during the encoding process. REVIEW_NO_DOCSTRING

        Parameters
        ----------
        cache_id : int
            The unique identifier for the cached object.

        Returns
        -------
        Encoded
            The encoded representation of the cached object reference.

        Raises
        ------
        RuntimeError
            This is an abstract method that should raise RuntimeError
            when not implemented by subclasses.
        """
        raise RuntimeError()

    @staticmethod
    @abstractmethod
    def decode_cached_obj(
        encoded: Encoded, decode_cache: DecodeCache = None
    ) -> Serializable | DeferredRef:
        """Decode a cached object reference.

        This abstract method should be implemented by concrete encoder classes
        to handle the decoding of objects that have been previously cached
        during the encoding process. REVIEW_NO_DOCSTRING

        Parameters
        ----------
        encoded : Encoded
            The encoded representation of a cached object reference.
        decode_cache : DecodeCache, optional
            Cache used to track decoded objects and resolve references.

        Returns
        -------
        Serializable | DeferredRef
            The decoded object or a deferred reference to be resolved later.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def encode_iterable(
        to_encode: EncodableIterables,
        encode_cache: EncodeCache = None,
        ignore_no_serialize_flags: bool = False,
    ) -> Encoded:
        """Encode an iterable into a serializable format.

        This abstract method should be implemented by concrete encoder classes
        to handle the encoding of iterable data structures (lists, tuples, etc.).
        REVIEW_NO_DOCSTRING

        Parameters
        ----------
        to_encode : EncodableIterables
            The iterable to be encoded.
        encode_cache : EncodeCache, optional
            Cache used to track encoded objects and avoid duplication.
        ignore_no_serialize_flags : bool, optional
            If True, ignores any no-serialize flags on objects. Default is False.

        Returns
        -------
        Encoded
            The encoded representation of the iterable.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def decode_iterable(
        encoded: Encoded, decode_cache: DecodeCache = None
    ) -> EncodableIterables:
        """Decode an encoded iterable back to its original form.

        This abstract method should be implemented by concrete encoder classes
        to handle the decoding of iterable data structures from the encoded format.
        REVIEW_NO_DOCSTRING

        Parameters
        ----------
        encoded : Encoded
            The encoded iterable data to be decoded.
        decode_cache : DecodeCache, optional
            Cache used to track decoded objects and resolve references.

        Returns
        -------
        EncodableIterables
            The decoded iterable in its original form.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def encode_dict(
        to_encode: dict,
        encode_cache: EncodeCache = None,
        ignore_no_serialize_flags: bool = False,
    ) -> Encoded:
        """Encode a dictionary into a serializable format.

        This abstract method should be implemented by concrete encoder classes
        to handle the encoding of dictionary data structures. REVIEW_NO_DOCSTRING

        Parameters
        ----------
        to_encode : dict
            The dictionary to be encoded.
        encode_cache : EncodeCache, optional
            Cache used to track encoded objects and avoid duplication.
        ignore_no_serialize_flags : bool, optional
            If True, ignores any no-serialize flags on objects. Default is False.

        Returns
        -------
        Encoded
            The encoded representation of the dictionary.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def decode_dict(
        encoded: Encoded, decode_cache: DecodeCache = None
    ) -> dict:
        """Decode an encoded dictionary back to its original form.

        This abstract method should be implemented by concrete encoder classes
        to handle the decoding of dictionary data from the encoded format.
        REVIEW_NO_DOCSTRING

        Parameters
        ----------
        encoded : Encoded
            The encoded dictionary data to be decoded.
        decode_cache : DecodeCache, optional
            Cache used to track decoded objects and resolve references.

        Returns
        -------
        dict
            The decoded dictionary in its original form.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def encode_array(to_encode: EncodableArrays):
        """Encode an array into a serializable format.

        This abstract method should be implemented by concrete encoder classes
        to handle the encoding of array data structures. REVIEW_NO_DOCSTRING

        Parameters
        ----------
        to_encode : EncodableArrays
            The array to be encoded.

        Returns
        -------
        Encoded
            The encoded representation of the array.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def decode_array(encoded: Encoded) -> EncodableArrays:
        """Decode an encoded array back to its original form.

        This abstract method should be implemented by concrete encoder classes
        to handle the decoding of array data from the encoded format.
        REVIEW_NO_DOCSTRING

        Parameters
        ----------
        encoded : Encoded
            The encoded array data to be decoded.

        Returns
        -------
        EncodableArrays
            The decoded array in its original form.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def encode_class(to_encode: type):
        """Encode a class type into a serializable format.

        This abstract method should be implemented by concrete encoder classes
        to handle the encoding of class type information. REVIEW_NO_DOCSTRING

        Parameters
        ----------
        to_encode : type
            The class type to be encoded.

        Returns
        -------
        Encoded
            The encoded representation of the class type.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def decode_class(encoded: Encoded) -> type:
        """Decode an encoded class reference back to its class object.

        This abstract method should be implemented by concrete encoder classes
        to handle the decoding of class type information. REVIEW_NO_DOCSTRING

        Parameters
        ----------
        encoded : Encoded
            The encoded representation of a class reference.

        Returns
        -------
        type
            The decoded class object.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def encode_function(to_encode: Callable) -> Encoded:
        """Encode a function into a serializable format.

        This abstract method should be implemented by concrete encoder classes
        to handle the encoding of function objects. REVIEW_NO_DOCSTRING

        Parameters
        ----------
        to_encode : Callable
            The function to be encoded.

        Returns
        -------
        Encoded
            The encoded representation of the function.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        pass

    @staticmethod
    @abstractmethod
    def decode_function(encoded: Encoded) -> Callable:
        """Decode an encoded function reference back to its callable form.

        This abstract method should be implemented by concrete encoder classes
        to handle the decoding of function objects. REVIEW_NO_DOCSTRING

        Parameters
        ----------
        encoded : Encoded
            The encoded representation of a function.

        Returns
        -------
        Callable
            The decoded function object.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        pass

    @staticmethod
    @abstractmethod
    def encode_primitive(to_encode: EncodablePrimitives):
        """Encode a primitive value into a serializable format.

        This abstract method should be implemented by concrete encoder classes
        to handle the encoding of primitive data types (int, float, str, bool, etc.).
        REVIEW_NO_DOCSTRING

        Parameters
        ----------
        to_encode : EncodablePrimitives
            The primitive value to be encoded.

        Returns
        -------
        Encoded
            The encoded representation of the primitive value.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def decode_primitive(encoded: Encoded) -> EncodablePrimitives:
        """Decode an encoded primitive value back to its original form.

        This abstract method should be implemented by concrete encoder classes
        to handle the decoding of primitive data types (int, float, str, bool, etc.)
        from the encoded format. REVIEW_NO_DOCSTRING

        Parameters
        ----------
        encoded : Encoded
            The encoded primitive value to be decoded.

        Returns
        -------
        EncodablePrimitives
            The decoded primitive value in its original form.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()
