"""Selective reads of an encoded `History`, for `History.collect_data` only.

A skeleton `History` decodes each frame's `log` and `_expired_keys` and
decodes a `_data` value only when it is asked for. It is not a faithful
`History`: it serves one `collect_data` call and lives only inside one store
call. A layout it can't navigate, or a value that references an object it
hasn't decoded, raises `Unnavigable` and the caller does a full decode.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

import h5py

from loqs.internal.encoder.hdf5encoder import (
    _COLLAPSED_BLOB_NAME,
    _get_version,
)
from loqs.internal.serializable import SERIALIZATION_VERSION, Serializable


class Unnavigable(Exception):
    """The selective read can't answer; decode the whole entry instead."""


class _Overlay(dict):
    """A per-entry decode cache layered over a store's cache.

    Reads see the overlay, then the base; writes stay in the overlay.
    `resolve` never scans a file, so a reference to an object that isn't
    held raises `Unnavigable`.
    """

    def __init__(self, base: Any):
        super().__init__()
        self._base = base

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, key) or key in self._base

    def __getitem__(self, key: Any) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return self._base[key]

    def resolve(self, cache_id: int) -> Any:
        if cache_id in self:
            return self[cache_id]
        raise Unnavigable(f"reference to object {cache_id} not decoded")


def _version(node: Any) -> int | None:
    return (
        node.get("version") if isinstance(node, dict) else _get_version(node)
    )


def _unwrap(node: Any) -> Any:
    """Descend through single-child wrapper groups to the encoded node."""
    if isinstance(node, dict):
        return node
    while "encode_type" not in node.attrs:
        names = list(node.keys())
        if len(names) != 1 or not isinstance(node[names[0]], h5py.Group):
            raise Unnavigable("unexpected wrapper group")
        node = node[names[0]]
    return node


def _object(node: Any, cls: type) -> Any:
    """The encoded node of exactly a `cls` source object, else `Unnavigable`."""
    node = _unwrap(node)
    attrs = node if isinstance(node, dict) else node.attrs
    if (
        attrs.get("encode_type") != "Serializable"
        or attrs.get("module") != cls.__module__
        or attrs.get("class") != cls.__name__
        or attrs.get("cache_type", "source") != "source"
        or _version(node) != SERIALIZATION_VERSION
    ):
        raise Unnavigable(f"not a plain {cls.__name__} node")
    return node


def _children(node: Any) -> Any:
    """Named children of an object node or a groups-format iterable group."""
    if isinstance(node, dict):
        return node
    out: dict = {}
    for name in node.keys():
        child = node[name]
        if name == _COLLAPSED_BLOB_NAME:
            out.update(json.loads(child[()].tobytes()))
        else:
            out[name] = child
    return out


def _elements(node: Any) -> list:
    """The element nodes of an encoded list."""
    if isinstance(node, dict):
        if (
            node.get("encode_type") != "iterable"
            or _version(node) != SERIALIZATION_VERSION
        ):
            raise Unnavigable("not an encoded list")
        return node["items"]
    group = node.get("iterable")
    if (
        group is None
        or group.attrs.get("storage_format", "groups") != "groups"
        or _version(group) != SERIALIZATION_VERSION
    ):
        raise Unnavigable("not a groups-format list")
    children = _children(group)
    return [children[str(i)] for i in range(len(children))]


def _items(node: Any, cache: Any) -> dict:
    """`{key: value node}` of an encoded dict."""
    if isinstance(node, dict):
        if (
            node.get("encode_type") != "dict"
            or _version(node) != SERIALIZATION_VERSION
        ):
            raise Unnavigable("not an encoded dict")
        items = node["items"]
        # The JSON decoder would `literal_eval` such a key.
        if any(k.startswith("(") and k.endswith(")") for k in items):
            raise Unnavigable("tuple-shaped key")
        return items
    group = node.get("dict")
    if group is None or _version(group) != SERIALIZATION_VERSION:
        raise Unnavigable("not an encoded dict group")
    keys: Any = Serializable.decode(
        group["keys"], format="hdf5", decode_cache=cache
    )
    return dict(zip(keys, _elements(group["values"])))


def _decode(node: Any, cache: Any) -> Any:
    """Decode one node; a JSON blob holds bare primitives, so is `"json"`."""
    fmt: Literal["hdf5", "json"] = (
        "hdf5" if isinstance(node, (h5py.Group, h5py.Dataset)) else "json"
    )
    return Serializable.decode(node, format=fmt, decode_cache=cache)


class _LazyData(dict):
    """A frame's `_data` that decodes each value on first access."""

    def __init__(self, encoded: dict, cache: Any):
        super().__init__()
        self._encoded = encoded
        self._cache = cache

    def __missing__(self, key: Any) -> Any:
        node = self._encoded[key]  # KeyError: the frame doesn't hold it
        try:
            value = _decode(node, self._cache)
        except Unnavigable:
            raise
        except Exception as exc:
            # A failed decode must not read as an absent key.
            raise Unnavigable(f"{type(exc).__name__}: {exc}") from exc
        self[key] = value
        return value


def lazy_history(
    entry: Any, cache: Any, locate: Callable[[int], Any] | None = None
) -> Any:
    """A skeleton `History` of the encoded `entry`.

    `locate(cache_id)` gives the entry node of the History a `copy` or
    `reference` stub points at. Raises `Unnavigable` for any other shape.
    """
    from loqs.core.frame import Frame
    from loqs.core.history import History

    top = _unwrap(entry)
    attrs = top if isinstance(top, dict) else top.attrs
    cache_type = attrs.get("cache_type")
    if cache_type in ("copy", "reference"):
        if locate is None:
            raise Unnavigable("stub without a locator")
        key = "reference_cache_id" if cache_type == "copy" else "cache_id"
        return lazy_history(locate(int(attrs[key])), cache)

    hist = _object(entry, History)
    frames = []
    for element in _elements(_children(hist)["_history"]):
        fields = _children(_object(element, Frame))
        frame = Frame(log=_decode(fields["log"], cache))
        frame._expired_keys = _decode(fields["_expired_keys"], cache)
        frame._data = _LazyData(_items(fields["_data"], cache), cache)
        frames.append(frame)
    history = History()
    history._history = frames
    return history
