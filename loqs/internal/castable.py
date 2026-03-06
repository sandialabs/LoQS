#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

""":class:`.Castable` definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar


# Generic type variable to stand-in for derived class below
T = TypeVar("T", bound="Castable")
U = TypeVar("U", bound="SeqCastable")
V = TypeVar("V", bound="MapCastable")


class Castable:
    """Utility class for objects that are "castable".

    By default, a :class:`.Castable` object is one that can be
    initialized from a single argument, i.e. the object
    can be passed as the first argument to the constructor and any
    remaining arguments have sensible defaults.
    They also have a :meth:`.cast` function, which either does nothing
    if the object matches the correct type, or initializes it by calling
    the constructor by passing the object as the first parameter.
    """

    def __init__(self, obj: object) -> None:
        """Construct a :class:`Castable` object.

        Parameters
        ----------
        obj:
            Castable objects should take at least one positional argument
        """
        pass

    @classmethod
    def cast(cls: type[T], obj: object) -> T:
        """Cast to the derived class.

        This is the base implementation. Derived classes should
        reimplement this if additional casting logic is desired.

        Parameters
        ----------
        obj:
            A castable object that is either:
            - Already the derived class type, in which case `obj`
              is returned
            - A kwarg dict that is passed into the derived class
              constructor
            - An args list that is passed into the derived class
              constructor
            - The first argument of the derived class constructor

        Returns
        -------
            An object with type T (matching the derived class)
        """
        if isinstance(obj, cls):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, Mapping):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)
        elif isinstance(obj, Sequence) and not isinstance(obj, str):
            # Assume this is a args list, pass in the args
            return cls(*obj)

        # Otherwise, assume this is the first __init__ argument
        return cls(obj)


class SeqCastable(Castable):
    """:class:`.Castable` object whose first argument is a ``list`` or ``tuple``."""

    @classmethod
    def cast(cls: type[U], obj: object) -> U:
        """Cast to the derived class.

        The difference from :meth:`.Castable.cast` is that we are
        expecting the first arg to be a ``Sequence``, so we skip
        the args list logic.

        Parameters
        ----------
        obj:
            A castable object that is either:
            - Already the derived class type, in which case `obj`
              is returned
            - A kwarg dict that is passed into the derived class
              constructor
            - The first argument of the derived class constructor

        Returns
        -------
            An object with type U (matching the derived class)
        """
        if isinstance(obj, cls):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, Mapping):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)

        # Otherwise, assume this is the first __init__ argument
        return cls(obj)


class MapCastable(Castable):
    """:class:`.Castable` object whose first argument is a ``dict``."""

    @classmethod
    def cast(cls: type[V], obj: object) -> V:
        """Cast to the derived class.

        The difference from :meth:`.Castable.cast` is that we are
        expecting the first arg to be a ``Mapping``, so we skip
        the kwargs dict logic.

        Parameters
        ----------
        obj:
            A castable object that is either:
            - Already the derived class type, in which case `obj`
              is returned
            - A kwarg dict that is passed into the derived class
              constructor
            - The first argument of the derived class constructor

        Returns
        -------
            An object with type U (matching the derived class)
        """
        if isinstance(obj, cls):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, Sequence) and not isinstance(obj, str):
            # Assume this is a args list, pass in the args
            return cls(*obj)

        # Otherwise, assume this is the first __init__ argument
        return cls(obj)
