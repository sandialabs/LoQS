"""Tester for loqs.internal.jsonencoding"""

import json
import pytest
from loqs.internal.jsonencoding import JSONEncoderWithErrors, dump_or_dumps_with_error_handling, JSONEncodingFailure
import tempfile
import os


class UnserializableObject:
    # dummy class used to trigger different errors.
    pass


class TestJSONEncoderWithErrors:
    """Test the JSONEncoderWithErrors class."""

    def test_jsonable_success(self):
        """Test basic JSON serialization with the custom encoder."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        result = json.dumps(data, cls=JSONEncoderWithErrors)
        parsed = json.loads(result)
        assert parsed == data

    def test_jsonable_fail(self):
        """Test error handling when serialization fails."""
        encoder = JSONEncoderWithErrors()
        obj = UnserializableObject()
        with pytest.raises(TypeError) as exc_info:
            encoder.encode(obj)
        assert "Error serializing object" in str(exc_info.value)
        return

    def test_wrapper_failure_1(self):
        """Test that the encoder tracks and reports the current key when errors occur."""    
        data = { "level1": { "level2": { "problem": UnserializableObject() } } }
        with pytest.warns(JSONEncodingFailure, match="Key: problem"):
            dump_or_dumps_with_error_handling(data)
        return
    
    def test_wrapper_failure_2(self):
        """Test that key tracking works with nested dictionaries."""
        data = { "outer": { "inner": { "deep": (1, 'a', UnserializableObject()) } } }
        with pytest.warns(JSONEncodingFailure, match="Key: deep"):
            dump_or_dumps_with_error_handling(data)
        return
    
    def test_wrapper_dumps_success(self):
        """Test successful serialization with dump_or_dumps_with_error_handling."""
        data = {"test": "data", "number": 123}
        result = dump_or_dumps_with_error_handling(data)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == data

    def test_wrapper_dump_success(self):
        """Test successful file dump with dump_or_dumps_with_error_handling."""
        data = {"test": "data", "number": 123}
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            temp_path = f.name
            dump_or_dumps_with_error_handling(data, f)
            f.flush()
        with open(temp_path, 'r') as f:
            content = f.read()
            parsed = json.loads(content)
            assert parsed == data
        os.unlink(temp_path)
        return

    def test_data_mixed(self):
        """Test with various data types that should work."""
        data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, "two", {"three": 3}],
            "nested": {
                "level1": {
                    "level2": "value"
                }
            }
        }
        result = dump_or_dumps_with_error_handling(data)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == data

    def test_data_empty(self):
        """Test with empty data structures."""
        result1 = dump_or_dumps_with_error_handling({})
        assert result1 == "{}"
        result2 = dump_or_dumps_with_error_handling([])
        assert result2 == "[]"
        result3 = dump_or_dumps_with_error_handling(None)
        assert result3 == "null"

    def test_data_numerical(self):
        """Test with special float values."""
        data = {
            "zero": 0.0,
            "negative": -42.5,
            "scientific": 1.23e-10,
            "large": 1.23e10
        }

        result : str = dump_or_dumps_with_error_handling(data)  # type: ignore
        parsed = json.loads(result)
        # Should preserve the values (within JSON precision)
        assert parsed["zero"] == 0.0
        assert parsed["negative"] == -42.5
        assert abs(parsed["scientific"] - 1.23e-10) < 1e-15
        assert abs(parsed["large"] - 1.23e10) < 1e-5

    def test_data_unicode(self):
        """Test with unicode data."""
        data = {"unicode": "Hello 世界 🌍", "emoji": "🚀🔥"}
        result : str = dump_or_dumps_with_error_handling(data)  # type: ignore
        parsed = json.loads(result)
        assert parsed == data
