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

    This is set back to 0 if ``reset_encode_id=True`` in
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
        h5_group: h5py.Group | None = None,
    ):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def decode_uncached_obj(
        encoded: Encoded, decode_cache: DecodeCache = None
    ) -> Serializable | DeferredRef:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def encode_cached_obj(
        cache_id: int, h5_group: h5py.Group | None = None
    ) -> Encoded:
        raise RuntimeError()

    @staticmethod
    @abstractmethod
    def decode_cached_obj(
        encoded: Encoded, decode_cache: DecodeCache = None
    ) -> Serializable | DeferredRef:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def encode_iterable(
        to_encode: EncodableIterables,
        encode_cache: EncodeCache = None,
        ignore_no_serialize_flags: bool = False,
        h5_group: h5py.Group | None = None,
    ) -> Encoded:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def decode_iterable(
        encoded: Encoded, decode_cache: DecodeCache = None
    ) -> EncodableIterables:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def encode_dict(
        to_encode: dict,
        encode_cache: EncodeCache = None,
        ignore_no_serialize_flags: bool = False,
        h5_group: h5py.Group | None = None,
    ) -> Encoded:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def decode_dict(
        encoded: Encoded, decode_cache: DecodeCache = None
    ) -> dict:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def encode_array(
        to_encode: EncodableArrays, h5_group: h5py.Group | None = None
    ):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def decode_array(encoded: Encoded) -> EncodableArrays:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def encode_class(to_encode: type, h5_group: h5py.Group | None = None):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def decode_class(encoded: Encoded) -> type:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def encode_function(
        to_encode: Callable, h5_group: h5py.Group | None = None
    ) -> Encoded:
        pass

    @staticmethod
    @abstractmethod
    def decode_function(encoded: Encoded) -> Callable:
        pass

    @staticmethod
    @abstractmethod
    def encode_primitive(
        to_encode: EncodablePrimitives, h5_group: h5py.Group | None = None
    ):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def decode_primitive(encoded: Encoded) -> EncodablePrimitives:
        raise NotImplementedError()
