"""A utility read-only class property class.

This is used extensively in :class:`IsCastable` objects,
as well as throughout the backend code.
The goal is provide a property-like thing on the class,
so you can check types without needing an instance.
Naively, one would expect a decorator chain like
@classmethod @property to do this, and it did...
from Python 3.9 to 3.11. But it was buggy; see:
https://github.com/python/cpython/issues/89519

It is still possible to code up a simple read-only
version of a class property though, which is provided
here.
I'll note that I think @abstractmethod chaining with
properties is also a little borked, so I recommend
not using the decorator and instead just raising
NotImplementedError. Slightly less static checking,
but will still signal to developers that an
implementation is needed.
"""


# Inspired by https://stackoverflow.com/a/13624858
class roclassproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)

    def __set__(self, owner_self, owner_cls):
        raise AttributeError("Read-only class properties cannot be set")

    def __delete__(self, owner_self, owner_cls):
        raise AttributeError("Read-only class properties cannot be deleted")
