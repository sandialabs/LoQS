"""A utility read-only class property class.

This file is heavily based on
https://stackoverflow.com/a/5191224,
but modified to have no setter capabilities.

This is used extensively in :class:`IsCastable` objects,
as well as throughout the backend code.
The goal is provide a property-like thing on the class,
so you can check types without needing an instance.
Naively, one would expect a decorator chain like
@classmethod @property to do this, and it did...
from Python 3.9 to 3.11. But it was buggy; see
https://github.com/python/cpython/issues/89519
for more details.

It is still possible to code up a simple read-only
version of a class property though, which is provided
here. Additionally, there are lots of places in the code
where we want to define an interface for read-only
class properties -- i.e. have abstract read-only class
properties -- so we also define a metaclass that combines
this with :class:`abc.ABCMeta`.

Examples
--------

.. testsetup:: *

    from loqs.utils.classproperty import *
    import warnings

Consider the following example that uses an :class:`roclassproperty`.

.. doctest::

    >>> class Bad:
    ...     @roclassproperty
    ...     def a(self):
    ...         return "A"
    >>> Bad.a
    'A'

And this also works on an instance of the class.

.. doctest::

    >>> bad = Bad()
    >>> bad.a
    'A'

Trying to set the property on the instance works as expected,
in that it is disallowed and raises an :class:`AttributeError`.

.. doctest::

    >>> bad.a = "b"
    Traceback (most recent call last):
    AttributeError: Cannot set read-only class properties

But what happens when the user tries to set :attr:`Bad.a`?
We would like this to fail, but in actuality the user can just
directly override the class member.

.. doctest::

    >>> Bad.a = "b"
    >>> Bad.a
    'b'

To fix this, we simply need to inherit from
:class:`HasROClassProperties`, which implements a check to avoid
this sort of overwrite.

.. doctest::

    >>> class Good(HasROClassProperties):
    ...     @readonlyclassproperty
    ...     def a(self):
    ...         return "A"
    >>> Good.a
    'A'
    >>> Good.a = "b"
    Traceback (most recent call last):
    AttributeError: Cannot set read-only class properties


Now, let's consider how we want to define an abstract base class
with read-only class properties that the derived classes should
implement. This can be done using :meth:`@abstractroclassproperty`.

.. doctest::

    >>> class Base(HasAbstractROClassProperties):
    ...     @abstractroclassproperty
    ...     def a(self):
    ...         pass

Unlike :meth:`@abstractmethod`, there's no initializiation here where
we can check that this has been defined. The best we can do is throw
a warning if the user accesses this abstrac property.
For sake of using :mod:`doctest`, we will upgrade the expected
:class:`UserWarning` to an error to make it easier to catch.

.. doctest::

    >>> warnings.simplefilter('error', UserWarning)
    >>> Base.a
    UserWarning: Abstract read-only class property called from \
<class '__main__.Base'>. Ensure that all derived classes implement this \
class property. Be aware that downstream code depending on this property \
may break in unexpected ways.

But we can construct a derived class and implement the class property.

.. doctest::

    >>> class Derived(Base):
    ...     @roclassproperty
    ...     def a(self):
    ...         return "A"
    >>> Derived.a
    'A'
"""

from abc import ABCMeta
import warnings


class _ReadOnlyClassPropertyDescriptor:
    """Wrapper class for read-only class properties."""

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self.fget.__get__(obj, cls)()

    def __set__(self, obj, value):
        raise AttributeError("Cannot set read-only class properties")

    def setter(self, func):
        raise AttributeError("Cannot set read-only class properties")


def roclassproperty(property):
    """A decorator for read-only class properties.

    See :mod:`loqs.utils.classproperty` for examples.
    """
    if not isinstance(property, (classmethod, staticmethod)):
        property = classmethod(property)

    return _ReadOnlyClassPropertyDescriptor(property)


class _HasROClassPropertiesMeta(type):
    """A metaclass that stops read-only class property overwrites.

    Users should usually inherit from :class:`HasROClassProperties`
    instead of using this metaclass.
    """

    def __setattr__(self, key, value):
        obj = self.__dict__.get(key, None)
        if obj and type(obj) is _ReadOnlyClassPropertyDescriptor:
            raise AttributeError("Cannot set read-only class properties")

        return super().__setattr__(key, value)


class HasROClassProperties(metaclass=_HasROClassPropertiesMeta):
    """A class that has read-only class properties.

    See :mod:`loqs.utils.classproperty` for examples.
    """

    pass


class _AbstractReadOnlyClassPropertyDescriptor(
    _ReadOnlyClassPropertyDescriptor
):
    """Wrapper class for abstract read-only class properties."""

    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        # If this is an error, autosummary breaks because a getattr fails...
        # Probably a warning is sufficient to help explain any subsequent errors
        # if this actually happens?
        warnings.warn(
            UserWarning(
                f"Abstract read-only class property called from {cls}. "
                "Ensure that all derived classes implement this class property. "
                "Be aware that downstream code depending on this property may break "
                "in unexpected ways."
            )
        )
        return None


def abstractroclassproperty(property):
    """A decorator for abstract read-only class properties.

    See :mod:`loqs.utils.classproperty` for examples.
    """
    if not isinstance(property, (classmethod, staticmethod)):
        property = classmethod(property)

    return _AbstractReadOnlyClassPropertyDescriptor(property)


class _HasAbstractROClassPropertiesMeta(ABCMeta, _HasROClassPropertiesMeta):
    """A combined metaclass for ABCMeta and _HasROClassPropertiesMeta.

    Users should inherit from :class:`HasAbstractROClassProperties`
    instead of using this metaclass.
    """

    pass


class HasAbstractROClassProperties(
    metaclass=_HasAbstractROClassPropertiesMeta
):
    """A combined class for ABC and HasROClassProperties.

    Use this if you need the features of both, or have (abstract) read-only
    class properties.

    See :mod:`loqs.utils.classproperty` for examples.
    """

    pass
