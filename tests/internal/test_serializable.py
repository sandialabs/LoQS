"""Parameterized tests for Serializable base class serialization methods."""

import pytest
import numpy as np
import h5py

from loqs.core import Frame
from loqs.internal.serializable import (
    Serializable,
    SERIALIZATION_VERSION,
    ResolvingDecodeCache,
    DeferredRef,
    _rebuild_from_encoding_attrs,
)
from loqs.types import NDArray


class MockSerializable(Serializable):
    """A concrete Serializable class for testing."""

    _CACHE_ON_SERIALIZE = True

    _SERIALIZE_ATTRS = ["name", "value", "data"]

    def __init__(self, name="test", value=42, data=None):
        self.name = name
        self.value = value
        self.data = data or {}

    def __eq__(self, other):
        return (
            isinstance(other, MockSerializable)
            and self.name == other.name
            and self.value == other.value
            and self.data == other.data
        )

    @classmethod
    def _from_decoded_attrs(cls, attr_dict):
        """Create a MockSerializable from decoded attributes dictionary."""
        return cls(
            name=attr_dict["name"],
            value=attr_dict["value"],
            data=attr_dict["data"],
        )


class TestPickleBridge:
    def test_unflagged_class_pickles_by_default(self):
        import pickle

        class Plain(Serializable):
            _SERIALIZE_ATTRS = ["a"]

            def __init__(self, a):
                self.a = a
                self.extra = "kept"

        # Pickle needs an importable class, so register it on this module.
        Plain.__qualname__ = Plain.__name__ = "_PickleBridgePlain"
        globals()["_PickleBridgePlain"] = Plain
        try:
            obj = Plain(3)
            assert (
                obj.__reduce_ex__(pickle.HIGHEST_PROTOCOL)[0]
                is not _rebuild_from_encoding_attrs
            )
            obj2 = pickle.loads(pickle.dumps(obj))
            assert obj2.a == 3
            assert obj2.extra == "kept"
        finally:
            del globals()["_PickleBridgePlain"]


@pytest.fixture(params=["json", "hdf5"])
def format_param(request):
    """Parameterized fixture for testing both JSON and HDF5 formats."""
    return request.param


class TestSerializableParameterized:
    """Parameterized tests for Serializable class functionality."""

    def test_dump_load_roundtrip(self, format_param, make_temp_path):
        """Test dump/load roundtrip with file streams."""
        obj = MockSerializable(
            name="test_obj", value=123, data={"key": "value"}
        )

        if format_param == "json":
            with make_temp_path(suffix=".json") as tempf_path:
                # Test dump to file - use the underlying file object
                with open(tempf_path, "w+") as f:
                    obj.dump(f)
                    f.seek(0)
                    loaded_obj = MockSerializable.load(f)

        else:  # hdf5
            with make_temp_path(suffix=".h5") as temp_file:
                with h5py.File(temp_file, "w") as h5_file:
                    obj.dump(h5_file, format="hdf5")

                with h5py.File(temp_file, "r") as h5_file:
                    loaded_obj = MockSerializable.load(h5_file, format="hdf5")

        assert isinstance(loaded_obj, MockSerializable)
        assert loaded_obj.name == "test_obj"
        assert loaded_obj.value == 123
        assert loaded_obj.data["key"] == "value"

    @pytest.mark.parametrize("ext", [".json", ".json.gz", ".h5", ".hdf5"])
    def test_format_detection(self, ext, make_temp_path):
        """The format comes from the file suffix, with no format given."""
        obj = MockSerializable(name="format_test", value=111)

        with make_temp_path(suffix=ext) as temp_path:
            obj.write(temp_path)
            loaded = MockSerializable.read(temp_path)
            assert obj == loaded

    def test_version_compatibility(self, format_param, make_temp_path):
        """Test serialization version handling."""
        obj = MockSerializable()

        if format_param == "json":
            state = Serializable.encode(
                obj, format="json", reset_encode_id=True
            )
            assert isinstance(state, dict)

            # Verify version is included
            assert "version" in state
            assert state["version"] == SERIALIZATION_VERSION

            # Test that objects can be deserialized with current version
            loaded_obj = Serializable.decode(state, format="json")
            assert obj == loaded_obj

        else:  # hdf5
            with make_temp_path(suffix=".h5") as temp_file:
                with h5py.File(temp_file, "w") as h5_file:
                    obj.dump(h5_file, format="hdf5")

                with h5py.File(temp_file, "r") as h5_file:
                    root_group = h5_file["root"]
                    assert isinstance(root_group, h5py.Group)

                    # Version is stamped once on the file's root group,
                    # not repeated at every node (see HDF5Encoder's
                    # `_HDF5_DECODE_VERSION`).
                    assert "version" in root_group.attrs
                    assert root_group.attrs["version"] == SERIALIZATION_VERSION

                # Test that objects can be deserialized with current version
                loaded_obj = MockSerializable.read(temp_file)
                assert obj == loaded_obj


class TestSerializableNestedData:
    """Enhanced tests for nested data serialization."""

    @pytest.mark.parametrize("ext", [".json", ".h5"])
    def test_nested_roundtrip(self, ext, make_temp_path):
        """Nested objects and containers survive a file round trip."""
        inner = MockSerializable(
            name="inner", value=100, data={"inner_key": "inner_value"}
        )
        data = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, "two", 3.0, None, {"nested_in_list": True}],
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
            "dict": {"deep": {"value": 42, "array": [10, 20, 30]}},
            "object": inner,
            "objects": [
                MockSerializable(name="obj1", value=1),
                MockSerializable(name="obj2", value=2),
            ],
        }
        obj = MockSerializable(name="nested", value=500, data=data)

        with make_temp_path(suffix=ext) as temp_path:
            obj.write(temp_path)
            loaded = MockSerializable.read(temp_path)

        assert loaded == obj

    def test_equal_sets_share_a_serial_hash_in_any_order(self):
        # 1 and 9 share a slot in a small set, so these two iterate in
        # different orders, as a set rebuilt on decode can.
        a, b = {1, 9}, {9, 1}
        assert list(a) != list(b)
        obj1 = MockSerializable(name="s", value=1, data={"keys": a})
        obj2 = MockSerializable(name="s", value=1, data={"keys": b})
        assert Serializable._serial_hash(obj1) == Serializable._serial_hash(
            obj2
        )

    @pytest.mark.parametrize("ext", [".json", ".h5"])
    def test_circular_reference_roundtrip(self, ext, make_temp_path):
        """A real cycle decodes to itself."""
        obj1 = MockSerializable(name="circular1", value=1)
        obj2 = MockSerializable(name="circular2", value=2)
        obj1.data["ref"] = obj2
        obj2.data["ref"] = obj1

        with make_temp_path(suffix=ext) as temp_path:
            obj1.write(temp_path)
            loaded = MockSerializable.read(temp_path)

        assert loaded.name == "circular1"
        assert loaded.data["ref"].name == "circular2"
        assert loaded.data["ref"].data["ref"] is loaded

    def test_iterable_encoding_both_codepaths(self, make_temp_path):
        """Test both HDF5 iterable encoding codepaths with caching improvements."""
        # Test case 1: Heterogeneous list (should use groups fallback)
        heterogeneous_list = [1, "string", 3.14, True, None]

        # Test case 2: Homogeneous list with Serializable objects (should use groups)
        obj1 = MockSerializable(name="obj1", value=1)
        obj2 = MockSerializable(name="obj2", value=2)
        obj3 = MockSerializable(name="obj3", value=3)
        serializable_list = [obj1, obj2, obj3]

        # Test case 3: Empty list (edge case)
        empty_list = []

        test_cases = [
            ("heterogeneous", heterogeneous_list, "groups"),
            ("serializable_objects", serializable_list, "groups"),
            (
                "empty",
                empty_list,
                "groups",
            ),  # Empty lists use groups (no benefit to dataset)
        ]

        for test_name, test_data, expected_format in test_cases:
            with make_temp_path(suffix=".h5") as temp_file:
                with h5py.File(temp_file, "w") as h5_file:
                    root_group = h5_file.create_group("root")

                    # Encode the list
                    list_group = Serializable.encode(
                        test_data, format="hdf5", h5_group=root_group
                    )
                    assert isinstance(list_group, h5py.Group)

                    # Verify storage format
                    storage_format = list_group.attrs.get(
                        "storage_format", "groups"
                    )
                    assert storage_format == expected_format

                    # Verify we can decode it back correctly
                    with h5py.File(temp_file, "r") as h5_read:
                        root_read = h5_read["root"]
                        assert isinstance(root_read, h5py.Group)
                        decoded_list = Serializable.decode(
                            root_read, format="hdf5"
                        )
                        assert isinstance(decoded_list, list)

                        # For lists with objects, check equality element by element
                        if test_name == "serializable_objects":
                            assert len(decoded_list) == len(test_data)
                            for i, (original, decoded) in enumerate(
                                zip(test_data, decoded_list)
                            ):
                                assert isinstance(decoded, MockSerializable)
                                assert decoded.name == original.name
                                assert decoded.value == original.value
                        else:
                            assert decoded_list == test_data

    def test_hdf5_array_compression(self, make_temp_path):
        """Test HDF5 array compression for large arrays."""
        # Create an array just over the compression threshold
        large_array = np.random.random((40, 40))  # 1,600 elements

        # Test direct array encoding (not embedded in MockSerializable)
        with make_temp_path(suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as h5_file:
                root_group = h5_file.create_group("root")
                # Encode the array directly to test compression
                array_group = Serializable.encode(
                    large_array, format="hdf5", h5_group=root_group
                )
                assert isinstance(array_group, h5py.Group)

                # Verify it's a dense real array
                array_type = array_group.attrs.get("array_type")
                assert array_type == "dense_real"

                # Check that compression was applied (data dataset should exist with compression)
                data_dataset = array_group["data"]
                assert isinstance(data_dataset, h5py.Dataset)
                assert data_dataset.compression

                # Verify we can decode it back correctly
                with h5py.File(temp_file, "r") as h5_read:
                    root_read = h5_read["root"]
                    assert isinstance(root_read, h5py.Group)
                    decoded_array = Serializable.decode(
                        root_read, format="hdf5"
                    )
                    assert isinstance(decoded_array, NDArray)

                    np.testing.assert_array_almost_equal(
                        decoded_array, large_array
                    )

    def test_cache_type_reference_vs_copy(self, format_param, make_temp_path):
        """Test the difference between reference and copy cache types."""
        # Create an object
        obj = MockSerializable(
            name="cache_test", value=100, data={"nested": "value"}
        )

        # Equivalent when encoded, but diff id
        dup_obj = MockSerializable(
            name="cache_test", value=100, data={"nested": "value"}
        )

        frame = Frame(
            {
                "obj1": obj,
                "obj2": obj,  # Same instance, should be a reference to obj
                "obj3": dup_obj,  # Same content, diff instance, should be a copy to obj
                "obj4": dup_obj,  # Same instance as dup_obj, should be a reference to dup_obj
            }
        )

        with make_temp_path(suffix=f".{format_param}") as temp_file:
            frame.write(temp_file)
            decoded = Frame.read(temp_file)

        assert isinstance(decoded, Frame)

        # obj2 is obj1
        assert decoded["obj2"] is decoded["obj1"]

        # obj3 should have same hash, but diff id
        assert Serializable._serial_hash(
            decoded["obj3"]
        ) == Serializable._serial_hash(decoded["obj1"])
        assert decoded["obj3"] is not decoded["obj1"]

        # obj4 is obj3
        assert decoded["obj4"] is decoded["obj3"]

    def test_copy_reconstruction_does_not_alias(
        self, format_param, make_temp_path
    ):
        """A regression test for the "copy" cache-type reconstruction
        using the cheap wrap-constructors of `History` and
        `InstructionStack` (sharing element identity) instead of a full
        `copy.deepcopy()`. Two independently decoded containers built
        from identical content share their elements by identity, but
        never the same underlying list: `History.append` mutates in
        place, so sharing it would silently corrupt the other."""
        from loqs.core.history import History
        from loqs.core.instructions.instructionstack import InstructionStack

        shared_frame = Frame({"x": 1})
        history_a = History([shared_frame])
        history_b = History([shared_frame])  # same content, different identity
        stack_a = InstructionStack([("Increment", "L0")])
        stack_b = InstructionStack([("Increment", "L0")])

        container = Frame(
            {
                "history_a": history_a,
                "history_b": history_b,
                "stack_a": stack_a,
                "stack_b": stack_b,
            }
        )

        with make_temp_path(suffix=f".{format_param}") as temp_file:
            container.write(temp_file)
            decoded = Frame.read(temp_file)

        decoded_ha = decoded["history_a"]
        decoded_hb = decoded["history_b"]
        decoded_sa = decoded["stack_a"]
        decoded_sb = decoded["stack_b"]

        for first, second in [
            (decoded_ha, decoded_hb),
            (decoded_sa, decoded_sb),
        ]:
            assert first is not second
            assert len(first) == len(second) == 1
            assert first[0] is second[0]

        decoded_ha.append(Frame({"y": 2}))
        assert len(decoded_ha) == 2
        assert len(decoded_hb) == 1


def _load_module_from_source(tmp_path, filename, source):
    """Write `source` to a real file under `tmp_path` and import it as a
    fresh module, so `inspect.getsource`/`getsourcefile` behave exactly
    as they would for a real, on-disk module (unlike a function defined
    directly in a test method, whose "source file" is this test file
    itself, cluttered with unrelated imports)."""
    import importlib.util
    import sys

    path = tmp_path / filename
    path.write_text(source)
    module_name = f"_test_get_function_str_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestGetFunctionStr:
    """Regression tests for Serializable._get_function_str's libcst-based
    import extraction, which replaced an earlier line-based text scan."""

    def test_import_extraction(self, tmp_path):
        module = _load_module_from_source(
            tmp_path,
            "mod_imports.py",
            "import os\n"
            "import unittest\n"
            "from collections import OrderedDict\n"
            "from collections.abc import (\n"
            "    Mapping,\n"
            "    Sequence,\n"
            ")\n"
            "\n"
            "def apply_fn(x):\n"
            '    """Talks about unittest but never uses it."""\n'
            "    # unittest appears here too\n"
            "    OrderedDict()\n"
            "    os.getcwd()\n"
            "    return isinstance(x, Mapping) and os.path.join(x)\n",
        )
        result = Serializable._get_function_str(module.apply_fn)
        assert "from collections import OrderedDict" in result
        assert "Mapping" in result
        assert "import unittest" not in result
        assert result.count("import os") == 1

    def test_conditionally_guarded_import_is_still_found_and_valid(
        self, tmp_path
    ):
        """A name only reachable via `if TYPE_CHECKING:`/`try/except
        ImportError:` is still a real module-scope binding by Python's
        own scoping rules, but its original indented text isn't valid
        syntax once lifted out of its enclosing block on its own --
        regression test for a real bug found while building this."""
        import ast

        module = _load_module_from_source(
            tmp_path,
            "mod_guarded.py",
            "from typing import TYPE_CHECKING\n"
            "\n"
            "if TYPE_CHECKING:\n"
            "    from collections import OrderedDict\n"
            "else:\n"
            "    try:\n"
            "        from collections import OrderedDict\n"
            "    except ImportError:\n"
            "        OrderedDict = dict\n"
            "\n"
            "def apply_fn(x):\n"
            "    return OrderedDict(x)\n",
        )
        result = Serializable._get_function_str(module.apply_fn)
        ast.parse(result)  # must be valid, standalone Python on its own
        assert "OrderedDict" in result

    def test_nested_function_finds_module_level_imports(self, tmp_path):
        """A function defined deep inside a class method (matching real
        usage, e.g. a closure in a test) should still resolve
        module-level imports correctly regardless of its own
        indentation."""
        import ast

        module = _load_module_from_source(
            tmp_path,
            "mod_nested.py",
            "from collections import OrderedDict\n"
            "\n"
            "class Builder:\n"
            "    def make(self):\n"
            "        def apply_fn(x):\n"
            "            return OrderedDict(x)\n"
            "        return apply_fn\n",
        )
        apply_fn = module.Builder().make()
        result = Serializable._get_function_str(apply_fn)
        ast.parse(result)
        assert "from collections import OrderedDict" in result

    def test_public_serialize_function_matches_private_helper(self, tmp_path):
        module = _load_module_from_source(
            tmp_path,
            "mod_public.py",
            "from collections import OrderedDict\n"
            "\n"
            "def apply_fn(x):\n"
            "    return OrderedDict(x)\n",
        )
        assert Serializable.serialize_function(
            module.apply_fn
        ) == Serializable._get_function_str(module.apply_fn)

    def test_cache_invalidated_when_file_changes(self, tmp_path):
        """_import_usage_index_for_file caches per (path, mtime) -- a
        file edited after its first use must not serve stale imports."""
        import importlib.util
        import os
        import sys

        path = tmp_path / "mod_cache.py"
        path.write_text(
            "from collections import OrderedDict\n"
            "\n"
            "def apply_fn(x):\n"
            "    return OrderedDict(x)\n"
        )
        spec = importlib.util.spec_from_file_location(
            "_test_cache_mod_v1", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["_test_cache_mod_v1"] = module
        spec.loader.exec_module(module)

        result1 = Serializable._get_function_str(module.apply_fn)
        assert "OrderedDict" in result1

        # Rewrite with a different import, forcing a distinct mtime
        # regardless of the filesystem's timestamp resolution.
        mtime = os.stat(path).st_mtime
        path.write_text(
            "from collections import defaultdict\n"
            "\n"
            "def apply_fn(x):\n"
            "    return defaultdict(x)\n"
        )
        os.utime(path, (mtime + 5, mtime + 5))

        spec2 = importlib.util.spec_from_file_location(
            "_test_cache_mod_v2", path
        )
        module2 = importlib.util.module_from_spec(spec2)
        sys.modules["_test_cache_mod_v2"] = module2
        spec2.loader.exec_module(module2)

        result2 = Serializable._get_function_str(module2.apply_fn)
        assert "defaultdict" in result2
        assert "OrderedDict" not in result2


class TestFunctionDecoding:
    """Stored functions decode in their defining module's globals and
    keep their source."""

    _NESTED = (
        "def helper(x):\n"
        "    return x + 1\n"
        "\n"
        "def build():\n"
        "    def apply_fn(x):\n"
        "        return helper(x) * 2\n"
        "    return apply_fn\n"
    )

    def test_nested_function_sees_module_globals(self, tmp_path):
        module = _load_module_from_source(
            tmp_path, "mod_decode_nested.py", self._NESTED
        )
        src = Serializable._get_function_str(module.build())
        decoded = Serializable._eval_function_str(src)
        assert decoded(3) == 8
        assert decoded.__qualname__ == "build.<locals>.apply_fn"
        # The module's own namespace is untouched
        assert not hasattr(module, "apply_fn")

    def test_rebuilt_function_reencodes(self, tmp_path):
        module = _load_module_from_source(
            tmp_path, "mod_decode_reencode.py", self._NESTED
        )
        src = Serializable._get_function_str(module.build())
        decoded = Serializable._eval_function_str(src)
        assert Serializable._get_function_str(decoded) == src

    def test_unimportable_module_falls_back_to_source(self):
        src = (
            "# loqs-function: _loqs_no_such_module_xyz:build.<locals>.f\n"
            "def f(x):\n"
            "    return x + 5\n"
        )
        decoded = Serializable._eval_function_str(src)
        assert decoded(1) == 6
        assert decoded.__qualname__ == "build.<locals>.f"


class TestResolvingDecodeCache:
    """Tests for ResolvingDecodeCache with reference-before-source ordering."""

    def test_hdf5_reference_before_source_resolution(self, make_temp_path):
        """Test HDF5 format with a "copy" entry decoded before its source.

        Two distinct instances sharing one _serial_hash are encoded as
        siblings (the first becomes cache_type="source", the second
        cache_type="copy"). The "copy" entry is then decoded directly, in
        isolation, with a fresh decode_cache that has never seen the source
        -- forcing ResolvingDecodeCache to locate and decode the source on
        demand rather than relying on prior top-down traversal order.
        """

        with make_temp_path(suffix=".h5") as temp_path:
            source_obj = MockSerializable(
                name="shared", value=42, data={"x": 1}
            )
            copy_obj = MockSerializable(name="shared", value=42, data={"x": 1})

            encode_cache: dict = {}
            with h5py.File(temp_path, "w") as f:
                container = f.create_group("container", track_order=True)
                first_group = Serializable.encode(
                    source_obj,
                    format="hdf5",
                    encode_cache=encode_cache,
                    h5_group=container,
                )
                second_group = Serializable.encode(
                    copy_obj,
                    format="hdf5",
                    encode_cache=encode_cache,
                    h5_group=container,
                )
                assert first_group.attrs["cache_type"] == "source"
                assert second_group.attrs["cache_type"] == "copy"
                second_name = second_group.name

            # Decode only the "copy" entry, in isolation, with a fresh cache --
            # the source is only ever found via `ResolvingDecodeCache`'s own scan.
            with h5py.File(temp_path, "r") as f:
                container = f["container"]
                second_group = f[second_name]
                cache = ResolvingDecodeCache(root=container, format="hdf5")
                decoded_copy = Serializable.decode(
                    second_group, format="hdf5", decode_cache=cache
                )

                assert isinstance(decoded_copy, MockSerializable)
                assert decoded_copy.name == "shared"
                assert decoded_copy.value == 42

    def test_resolve_finds_copy_node_not_just_source(self):
        """A `cache_type="copy"` node mints its own resolvable
        `source_cache_id` (for a later reference to the copy itself, not
        the original object it copies from) -- `.resolve()` must match a
        "copy" node by `source_cache_id`, not only a plain "source" node
        by `cache_id`.

        Two distinct `MockSerializable` instances with identical field
        values share one `_serial_hash`: the first becomes `cache_type="source"`,
        the second (a different instance, same content) becomes
        `cache_type="copy"`. A third reference to that same second instance
        is decoded directly, in isolation, forcing `.resolve()` to locate
        the "copy" node -- there is no plain "source" node for its cache_id
        anywhere in the structure.
        """
        first = MockSerializable(name="shared", value=1, data={})
        second = MockSerializable(name="shared", value=1, data={})

        encode_cache: dict = {}
        first_encoded = Serializable.encode(
            first, format="json", encode_cache=encode_cache
        )
        second_encoded = Serializable.encode(
            second, format="json", encode_cache=encode_cache
        )
        third_encoded = Serializable.encode(
            second, format="json", encode_cache=encode_cache
        )

        assert second_encoded["cache_type"] == "copy"
        assert third_encoded["cache_type"] == "reference"
        assert third_encoded["cache_id"] == second_encoded["source_cache_id"]

        root = {
            "first": first_encoded,
            "second": second_encoded,
            "third": third_encoded,
        }
        cache = ResolvingDecodeCache(root=root, format="json")
        decoded_third = Serializable.decode(
            third_encoded, format="json", decode_cache=cache
        )

        assert isinstance(decoded_third, MockSerializable)
        assert decoded_third.name == "shared"
        assert decoded_third.value == 1

    def test_missing_source_raises_error(self):
        """Test that missing source raises RuntimeError."""
        # Create reference to non-existent source
        reference_encoded = {
            "encode_type": "Serializable",
            "module": "test_serializable",
            "class": "MockSerializable",
            "version": 2,
            "cache_type": "reference",
            "cache_id": 999,  # Non-existent source
        }

        root = {"ref": reference_encoded}
        cache = ResolvingDecodeCache(root=root, format="json")

        with pytest.raises(RuntimeError):
            Serializable.decode(
                reference_encoded, format="json", decode_cache=cache
            )

    def test_collapsed_blob_source_resolution(self, make_temp_path):
        """Test that a `cache_type="source"` entry nested inside a real HDF5
        `"$collapsed"` blob dataset (not a real `h5py.Group`) can be resolved
        on demand by `ResolvingDecodeCache`, exercising
        `_find_source_in_collapsed_blob` specifically rather than the plain
        real-group scan.

        A shared, array-free `MockSerializable` referenced twice from one
        small outer object collapses per `_contains_no_array`
        (`loqs/internal/encoder/hdf5encoder.py`), so both its `"source"` and
        `"reference"` cache entries land inside one `"$collapsed"` dataset
        rather than real HDF5 groups. `.resolve()` is called directly against
        a fresh, empty cache (never decoding anything first), so the only way
        it can find the source is by actually scanning into that blob.
        """
        from loqs.internal.encoder.hdf5encoder import _COLLAPSED_BLOB_NAME
        import json as jsonlib

        with make_temp_path(suffix=".h5") as temp_path:
            shared = MockSerializable(name="shared", value=7, data={"a": 1})
            outer = MockSerializable(
                name="outer", value=1, data={"x": shared, "y": shared}
            )

            with h5py.File(temp_path, "w") as f:
                outer.dump(f, format="hdf5")

            with h5py.File(temp_path, "r") as f:
                outer_group = f["root"][next(iter(f["root"].keys()))]

                # Confirm the test setup actually produced a collapsed blob
                # (proves the scenario below, not incidental).
                assert _COLLAPSED_BLOB_NAME in outer_group

                # Read the shared object's real cache_id directly off the
                # blob's own JSON content, rather than hardcoding it.
                raw = (
                    outer_group[_COLLAPSED_BLOB_NAME][()]
                    .tobytes()
                    .decode("utf-8")
                )
                blob = jsonlib.loads(raw)
                items = blob["data"]["items"]
                source_entries = [
                    v
                    for v in items.values()
                    if isinstance(v, dict) and v.get("cache_type") == "source"
                ]
                assert len(source_entries) == 1
                shared_cache_id = source_entries[0]["cache_id"]

                # Fresh, empty cache -- nothing decoded through it yet.
                cache = ResolvingDecodeCache(root=f["root"], format="hdf5")

                resolved = cache.resolve(shared_cache_id)

                assert isinstance(resolved, MockSerializable)
                assert resolved.name == "shared"
                assert resolved.value == 7
                assert resolved.data == {"a": 1}

    def test_programresults_decode_cache_repointing(self):
        """Regression: lazy `get_shot_history` calls, each opening a fresh
        `h5py.File`, must reuse the store's decode cache re-pointed at it.

        Every shot's first frame holds one shared, cache-eligible
        `MockSerializable`, so the file holds one "source" and later
        "reference"/"copy" entries. Reading the shots lazily, out of
        order, must resolve it through the re-pointed cache; a cache
        still pointing at an earlier file fails to resolve it.
        """
        from loqs.core import ProgramResults
        from loqs.core.history import History
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"

            writer = ProgramResults(lazy_loading=False)
            shared = MockSerializable(name="shared", value=7)

            for shot in range(3):
                history = History()
                history.append(Frame({"shared": shared}))
                history.append(Frame({"shot": shot}))
                writer.add_shot(shot, history)
            writer.checkpoint(checkpoint_dir=checkpoint_dir)
            writer.consolidate_checkpoints(checkpoint_dir=checkpoint_dir)
            assert (checkpoint_dir / "results.h5").exists()

            # Lazy consumer: each call opens its own h5py.File but reuses
            # the store's one canonical decode cache.
            reader = ProgramResults(lazy_loading=True)
            reader._checkpoint_dir = checkpoint_dir

            for shot in (2, 1, 0):
                history = reader.get_shot_history(shot)
                assert history is not None
                assert history[0]["shared"].value == 7
                assert history[1]["shot"] == shot

    def test_strict_validators_with_qeccodepatch(self):
        """Regression: QECCodePatch strict validators don't leak DeferredRef.

        QECCodePatch._from_decoded_attrs has strict type checking that raises
        ValueError if 'code' is not a real QECCode instance. This test verifies
        that ResolvingDecodeCache resolves the code before passing it to the
        validator, avoiding a DeferredRef reaching the constructor.

        The test constructs a reference-before-source scenario where a shared
        QECCode is referenced by two QECCodePatches, with the first reference
        reaching the constructor before the code's source is decoded.
        """
        from loqs.core.qeccode import QECCode
        from loqs.core.recordables.qeccodepatch import QECCodePatch
        from loqs.core.instructions import Instruction

        # Build a minimal QECCode fixture matching test_qeccode.py pattern
        def apply_fn(state, qubits):
            return Frame({"state": state + 1, "qubits": qubits})

        def map_qubits_fn(qubit_mapping, qubits, **kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs["qubits"] = [qubit_mapping[q] for q in qubits]
            return new_kwargs

        ins_data = {"qubits": ["Q0", "Q1"]}
        ins = Instruction(apply_fn, ins_data, map_qubits_fn, name="test")
        code = QECCode({"test_ins": ins}, ["Q0", "Q1"], ["Q0"], "Test code")

        # Create a shared code and two patches that reference it
        patch1 = QECCodePatch(code=code, qubits=["D0", "A0"], pauli_frame="II")
        patch2 = QECCodePatch(code=code, qubits=["D1", "A1"], pauli_frame="II")

        # Encode both patches to JSON (shared code becomes source+reference)
        cache = {}
        patch1_encoded = Serializable.encode(
            patch1, format="json", encode_cache=cache, reset_encode_id=True
        )
        patch2_encoded = Serializable.encode(
            patch2, format="json", encode_cache=cache
        )

        # Construct a root where patch2's reference appears first
        # (forcing decode to resolve the code on-demand)
        root = {"patch2": patch2_encoded, "patch1": patch1_encoded}

        # Decode patch2 first through ResolvingDecodeCache
        # (its code reference should trigger resolution of the code source)
        decode_cache = ResolvingDecodeCache(root=root, format="json")
        decoded_patch2 = Serializable.decode(
            patch2_encoded, format="json", decode_cache=decode_cache
        )

        # Verify: patch2 was decoded successfully without DeferredRef leaking
        assert isinstance(decoded_patch2, QECCodePatch)
        assert not isinstance(decoded_patch2.code, DeferredRef)
        assert isinstance(decoded_patch2.code, QECCode)
        assert decoded_patch2.code.name == "Test code"
        assert decoded_patch2.qubits == ["D1", "A1"]

    def test_strict_validators_with_instructionlabel_and_instruction(self):
        """Regression: InstructionLabel doesn't leak unresolved Instruction refs.

        InstructionLabel is not itself a Serializable, but can contain
        Serializable Instruction objects. This test verifies that when an
        Instruction is shared and referenced before its source is decoded,
        the ResolvingDecodeCache correctly resolves it before it reaches
        InstructionLabel's own constructor (which has runtime type checks).
        """
        from loqs.core.instructions import Instruction, InstructionLabel

        # Build a minimal Instruction
        def apply_fn():
            return Frame({"result": "test"})

        ins = Instruction(apply_fn, name="TestInstruction")

        # Create two InstructionLabels that reference the same Instruction
        label1 = InstructionLabel(ins, patch_label="L0")
        label2 = InstructionLabel(ins, patch_label="L1")

        # Encode both labels to JSON (shared instruction becomes source+reference)
        cache = {}
        label1_encoded = Serializable.encode(
            label1, format="json", encode_cache=cache, reset_encode_id=True
        )
        label2_encoded = Serializable.encode(
            label2, format="json", encode_cache=cache
        )

        # Construct a root where label2's reference appears first
        root = {"label2": label2_encoded, "label1": label1_encoded}

        # Decode label2 first through ResolvingDecodeCache
        decode_cache = ResolvingDecodeCache(root=root, format="json")
        decoded_label2 = Serializable.decode(
            label2_encoded, format="json", decode_cache=decode_cache
        )

        # Verify: label2 was decoded as a dict (InstructionLabel is a dict subclass),
        # and its instruction is not a DeferredRef
        assert isinstance(decoded_label2, dict)
        assert decoded_label2["patch_label"] == "L1"
        # The instruction should be a real Instruction, not a DeferredRef
        instruction = decoded_label2["instruction"]
        assert isinstance(instruction, Instruction)
        assert not isinstance(instruction, DeferredRef)
        assert instruction.name == "TestInstruction"


def _handle_is_closed(f):
    if isinstance(f, h5py.File):
        return not f.id.valid
    return f.closed


class TestFileHandles:
    """`read` and `write` must close their file when `load`/`dump` raises."""

    @pytest.mark.parametrize("ext", ["json", "hdf5"])
    def test_read_closes_file_on_error(self, tmp_path, monkeypatch, ext):
        path = tmp_path / f"obj.{ext}"
        MockSerializable(name="a", value=1).write(path)

        seen = []

        def failing_load(cls, f, *args, **kwargs):
            seen.append(f)
            raise RuntimeError("load failed")

        monkeypatch.setattr(Serializable, "load", classmethod(failing_load))
        # Keep excinfo (and so the traceback frames) alive during the check
        with pytest.raises(RuntimeError) as excinfo:
            MockSerializable.read(path)

        assert len(seen) == 1
        assert _handle_is_closed(seen[0])
        assert excinfo.value is not None

    @pytest.mark.parametrize("ext", ["json", "hdf5"])
    def test_write_closes_file_on_error(self, tmp_path, monkeypatch, ext):
        path = tmp_path / f"obj.{ext}"
        seen = []

        def failing_dump(self, f, *args, **kwargs):
            seen.append(f)
            raise RuntimeError("dump failed")

        monkeypatch.setattr(Serializable, "dump", failing_dump)
        with pytest.raises(RuntimeError) as excinfo:
            MockSerializable(name="a", value=1).write(path)

        assert len(seen) == 1
        assert _handle_is_closed(seen[0])
        assert excinfo.value is not None


class _FreshAttrs(MockSerializable):
    """Returns a new `MockSerializable` from `_get_encoding_attr` each call."""

    _SERIALIZE_ATTRS = ["name", "value", "data", "first", "second"]

    def _get_encoding_attr(self, attr, ignore_no_serialize_flags=False):
        if attr == "first":
            return MockSerializable(name="fresh", value=1)
        if attr == "second":
            return MockSerializable(name="fresh", value=2)
        return super()._get_encoding_attr(attr, ignore_no_serialize_flags)

    @classmethod
    def _from_decoded_attrs(cls, attr_dict):
        obj = cls(
            name=attr_dict["name"],
            value=attr_dict["value"],
            data=attr_dict["data"],
        )
        obj.decoded_pair = (attr_dict["first"], attr_dict["second"])
        return obj


class TestEncodeHashing:
    """The per-encode hash memo and the raw-bytes array hash."""

    def test_fresh_encoding_attr_objects_keep_their_content(self):
        encoded = Serializable.encode(
            _FreshAttrs(),
            format="json",
            encode_cache={},
            reset_encode_id=True,
        )
        decoded = Serializable.decode(encoded, format="json")
        first, second = decoded.decoded_pair
        assert (first.value, second.value) == (1, 2)

    @pytest.mark.parametrize("how", ["completed", "raised"])
    def test_object_changed_between_encodes_is_rewritten(self, how):
        obj = MockSerializable(name="x", value=1)
        cache: dict = {}
        if how == "completed":
            Serializable.encode(
                obj, format="json", encode_cache=cache, reset_encode_id=True
            )
        else:
            with pytest.raises(ValueError):
                Serializable.encode(
                    [obj, object()],
                    format="json",
                    encode_cache=cache,
                    reset_encode_id=True,
                )
        obj.value = 2
        encoded = Serializable.encode(obj, format="json", encode_cache=cache)
        assert Serializable.decode(encoded, format="json").value == 2

    def test_equal_valued_arrays_keep_their_dtypes_and_shapes(
        self, make_temp_path
    ):
        arrays = {
            "a": np.array([True, False]),
            "b": np.array([1, 0], dtype=np.uint8),
            "c": np.array([[1], [0]], dtype=np.uint8),
        }
        frame = Frame(
            {k: MockSerializable(data={"arr": v}) for k, v in arrays.items()}
        )
        with make_temp_path(suffix=".h5") as path:
            frame.write(path)
            decoded = Frame.read(path)
        for k, arr in arrays.items():
            out = decoded[k].data["arr"]
            assert (out.dtype, out.shape) == (arr.dtype, arr.shape)

    def test_serial_hash_calls_scale_linearly(self, monkeypatch):
        original = Serializable._serial_hash
        calls = []

        def counting(obj, _visited=None):
            calls.append(1)
            return original(obj, _visited)

        monkeypatch.setattr(
            Serializable, "_serial_hash", staticmethod(counting)
        )

        def count(n, m):
            arr = np.arange(m, dtype=float)
            obj = MockSerializable(name="end", value=0, data={"arr": arr})
            for i in range(n):
                obj = MockSerializable(
                    name="link", value=i, data={"next": obj, "arr": arr}
                )
            calls.clear()
            Serializable.encode(
                obj, format="json", encode_cache={}, reset_encode_id=True
            )
            return len(calls)

        assert count(20, 8) < 2.5 * count(10, 8)
        assert count(10, 16) == count(10, 8)
