"""Mentions Castable in prose, not code -- must not be flagged here."""
from loqs.internal.castable import Castable


class Foo(Castable):
    pass
