"""Tests for the selective history reads in `loqs.internal.historyread`."""

import pytest

from loqs.internal.historyread import Unnavigable, _LazyData
from loqs.internal.serializable import SERIALIZATION_VERSION


def test_lazy_value_decode_failure_falls_back():
    """A value whose decode raises `KeyError` (here a `Frame` node missing
    its attributes) asks for the full decode instead of reading as a key
    the frame doesn't hold."""
    node = {
        "encode_type": "Serializable",
        "module": "loqs.core.frame",
        "class": "Frame",
        "version": SERIALIZATION_VERSION,
    }
    data = _LazyData({"bad": node}, {})
    with pytest.raises(Unnavigable):
        data["bad"]
