"""Tests for the shared rep-conversion helper in basemodel."""

import pytest

from loqs.backends.model.basemodel import first_successful_rep
from loqs.backends.reps import RepConstructionError


def test_first_successful_rep():
    calls = []

    def make_source():
        calls.append("source")
        return "src"

    def build(source, candidate):
        calls.append(candidate)
        if candidate == "A":
            raise RepConstructionError("no A")
        return (source, candidate)

    # fallback: A fails, B is built from one source and stored
    cache: dict = {}
    rep = first_successful_rep(cache, "k", ["A", "B"], make_source, build)
    assert rep == ("src", "B")
    assert cache == {("k", "B"): rep}
    assert calls == ["source", "A", "B"]

    # cached: a hit builds nothing
    calls.clear()
    hit = first_successful_rep(cache, "k", ["A", "B"], make_source, build)
    assert hit is rep
    assert calls == []

    # refused: every candidate failing raises
    with pytest.raises(RepConstructionError):
        first_successful_rep(None, "k", ["A"], make_source, build)
