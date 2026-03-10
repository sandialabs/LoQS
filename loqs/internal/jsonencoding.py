"""JSON tools for serialization.
"""

import json
from typing import Any, Callable
from warnings import warn


class JSONEncodingFailure(UserWarning):
    pass


class JSONEncoderWithErrors(json.JSONEncoder):
    """JSON encoder with more helpful errors.

    This should print out what key/value are causing
    JSON encoding issues.
    """

    def __init__(self, *, skipkeys: bool = False, ensure_ascii: bool = True, check_circular: bool = True, allow_nan: bool = True, sort_keys: bool = False, indent: int | str | None = None, separators: tuple[str, str] | None = None, default: Callable[..., Any] | None = None) -> None:
        super().__init__(skipkeys=skipkeys, ensure_ascii=ensure_ascii, check_circular=check_circular, allow_nan=allow_nan, sort_keys=sort_keys, indent=indent, separators=separators, default=default)
        self.key : Any = None

    def default(self, o):
        try:
            return json.JSONEncoder.default(self, o)
        except TypeError as e:
            e.args = (f"Error serializing object: {o}, Key: {self.key}",)
            raise e


def dump_or_dumps_with_error_handling(data, f=None):
    encoder = JSONEncoderWithErrors()

    def encode_with_key_tracking(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                encoder.key = key
                encode_with_key_tracking(value)
        else:
            encoder.encode(obj)

    try:
        encode_with_key_tracking(data)
        if f is not None:
            json.dump(data, f, cls=JSONEncoderWithErrors)
        else:
            return json.dumps(data, cls=JSONEncoderWithErrors)
    except TypeError as e:
        warn(str(e), JSONEncodingFailure)

    return
