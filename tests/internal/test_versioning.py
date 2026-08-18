"""Tests for the VersionedDecoder registry mechanism."""

import pytest

import loqs.internal.versioning as versioning_module
from loqs.internal.serializable import DecodableVersionError, SERIALIZATION_VERSION
from loqs.internal.versioning import _ALL_VERSIONED_DECODERS, VersionedDecoder


class TestVersionedDecoder:
    """Unit tests for the registry class itself, independent of any real encoder."""

    @pytest.fixture(autouse=True)
    def isolate_all_versioned_decoders(self, monkeypatch):
        """Every VersionedDecoder self-registers into the shared,
        module-level `_ALL_VERSIONED_DECODERS` list -- redirect it to a
        throwaway list for these tests so the ephemeral registries created
        here don't pollute the real coverage check below."""
        monkeypatch.setattr(versioning_module, "_ALL_VERSIONED_DECODERS", [])

    def test_register_and_call(self):
        registry = VersionedDecoder("test")

        @registry.register(1)
        def _(x):
            return x + 1

        assert registry(1, 5) == 6

    def test_call_with_unregistered_version_raises(self):
        registry = VersionedDecoder("test")

        @registry.register(1)
        def _(x):
            return x

        with pytest.raises(DecodableVersionError):
            registry(2, "anything")

    def test_double_registration_asserts(self):
        registry = VersionedDecoder("test")

        @registry.register(1)
        def _(x):
            return x

        with pytest.raises(AssertionError):

            @registry.register(1)
            def _(x):
                return x

    def test_alias_reuses_same_function(self):
        registry = VersionedDecoder("test")

        @registry.register(1)
        def _(x):
            return x * 2

        registry.alias(2, same_as=1)
        assert registry(1, 10) == registry(2, 10) == 20

    def test_alias_to_unregistered_version_asserts(self):
        registry = VersionedDecoder("test")
        with pytest.raises(AssertionError):
            registry.alias(2, same_as=1)


class TestAllVersionedDecodersCoverage:
    """Every real VersionedDecoder must have a decoder registered for every
    version from its own shape's earliest version through the current
    SERIALIZATION_VERSION, with no gaps -- a mechanical replacement for a
    manual grep-based checklist. Not every registry starts at version 0
    (HDF5's shapes only exist from version 1 onward)."""

    def test_all_versioned_decoders_cover_every_version(self):
        # Import the encoder modules so their registries exist -- pytest
        # collection alone doesn't guarantee this ordering.
        import loqs.internal.encoder.hdf5encoder  # noqa: F401
        import loqs.internal.encoder.jsonencoder  # noqa: F401

        missing = []
        for registry in _ALL_VERSIONED_DECODERS:
            if not registry._decoders:
                continue
            earliest = min(registry._decoders)
            for v in range(earliest, SERIALIZATION_VERSION + 1):
                if v not in registry._decoders:
                    missing.append((registry._name, v))
        assert missing == [], f"Missing versioned decoders: {missing}"


class TestVersionBoundHandling:
    """Confirm the registry-based bound check (replacing the old hardcoded
    `!= 1`/`== 1` checks) actually rejects a too-new version, and that a
    real round-trip at the current SERIALIZATION_VERSION works."""

    def test_current_version_round_trips(self):
        from loqs.internal.serializable import Serializable

        assert SERIALIZATION_VERSION == 2
        encoded = Serializable.encode(42, format="json")
        assert encoded["version"] == 2
        assert Serializable.decode(encoded, format="json") == 42

    def test_unregistered_future_version_rejected(self):
        from loqs.internal.encoder.jsonencoder import JSONEncoder

        too_new = {"encode_type": "primitive", "version": 3, "value": 42}
        with pytest.raises(DecodableVersionError):
            JSONEncoder.decode_primitive(too_new)
