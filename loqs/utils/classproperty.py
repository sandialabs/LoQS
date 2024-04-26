"""Utilities for read-only class properties.

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
where we want to define an interface class using
:mod:`abc.ABC` and also have class properties -- i.e.
backends -- so we also define a metaclass that combines
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
    ...     @roclassproperty
    ...     def a(self):
    ...         return "A"
    >>> Good.a
    'A'
    >>> Good.a = "b"
    Traceback (most recent call last):
    AttributeError: Cannot set read-only class properties


Now, let's consider how we want to define an abstract base class
with read-only class properties that the derived classes should
implement. This can be done using :obj:`@abstractroclassproperty`.

If only the class properties are abstract, you can actually
get away with only inheriting from :class:`HasROClassProperties`,
because the abstract class property does not make use of
:mod:`abc.ABC` at all -- it simply throws a warning if the
class property is accessed.

For sake of using :mod:`doctest`, we will upgrade the expected
:class:`UserWarning` to an error to make it easier to catch in
this example.

.. doctest::

    >>> class Base(HasROClassProperties):
    ...     @abstractroclassproperty
    ...     def a(self):
    ...         pass
    >>> warnings.simplefilter('error', UserWarning)
    >>> Base.a
    UserWarning: Abstract read-only class property called from \
<class '__main__.Base'>. Ensure that all derived classes implement this \
class property. Be aware that downstream code depending on this property \
may break in unexpected ways.

But we can construct a derived class and implement the class property!

.. doctest::

    >>> class Derived(Base):
    ...     @roclassproperty
    ...     def a(self):
    ...         return "A"
    >>> Derived.a
    'A'


If your base class also needs to inherit from :class:`abc.ABC`,
then you should use :class:`ABCWithROClassProperties` instead.
Because both of these classes depend on metaclasses, you need to use
a metaclass that inherits both of them (which is all that
:class:`ABCWithROClassProperties` is). If you don't, you get:

.. doctest::

    >>> class MetaclassConflict(ABC, HasROClassProperties):
    ...     pass
    Traceback (most recent call last)
    TypeError: metaclass conflict: the metaclass of a derived class \
must be a (non-strict) subclass of the metaclasses of all its bases

If we instead use :class:`ABCWithROClassProperties`, everything
works as intended:

.. doctest::

    >>> class ABCBase(ABCWithROClassProperties):
    ...     @roclassproperty
    ...     def a(self):
    ...         return "A"
    ...     @abstractmethod
    ...     def b(self):
    ...         pass
    >>> class ABCDerived(ABCBase):
    ...     def b(self):
    ...         return "B"
    >>> ABCDerived.a
    'A'
    >>> derived = ABCDerived()
    >>> derived.b
    'B'

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
        if obj and issubclass(type(obj), _ReadOnlyClassPropertyDescriptor):
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


class _ABCWithROClassPropertiesMeta(ABCMeta, _HasROClassPropertiesMeta):
    """A combined metaclass for ABCMeta and _HasROClassPropertiesMeta.

    Users should inherit from :class:`HasAbstractROClassProperties`
    instead of using this metaclass.
    """

    pass


class ABCWithROClassProperties(metaclass=_ABCWithROClassPropertiesMeta):
    """A combined class for ABC and HasROClassProperties.

    Use this if you need the features of both, as this prevents
    metaclass collisions.

    See :mod:`loqs.utils.classproperty` for examples.
    """

    pass
