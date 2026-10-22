"""Tests for the VersionedDecoder registry mechanism."""

import pytest

from loqs.internal.serializable import (
    DecodableVersionError,
    SERIALIZATION_VERSION,
)
from loqs.internal.versioning import _ALL_VERSIONED_DECODERS


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
        assert missing == []


class TestVersionBoundHandling:
    """Confirm the registry-based bound check actually rejects a too-new
    version."""

    def test_unregistered_future_version_rejected(self):
        from loqs.internal.encoder.jsonencoder import JSONEncoder

        too_new = {"encode_type": "primitive", "version": 3, "value": 42}
        with pytest.raises(DecodableVersionError):
            JSONEncoder.decode_primitive(too_new)
