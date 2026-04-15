---
title: Frame
marimo-version: 0.23.1
---

```python {marimo}
import marimo as mo
```

# Frames

A `Frame` is essentially an (almost) immutable dictionary with strings for keys and arbitrary objects for values.
This allows for users to store any type of information that they want in a `Frame` and have the rest of `LoQS` machinery operate on it.

## Frame Basics

As we have already seen in the [History tutorial](/markdown/history), a `Frame` can be created from any `collections.abc.Mapping` object, most often a `dict`.

```python {marimo}
from loqs.core import Frame

frame = Frame({"int": 0, "data": "Dummy data"}, log="Log string for frame")
print(frame)
```

The data in a `Frame` cannot be changed, but an updated copy of the `Frame` can be created with the `update()` function.

```python {marimo}
_new_frame = frame.update({'int': 1})
print(_new_frame)
```

By default, the log string is not changed during an update, but it can also be set if desired.

```python {marimo}
_new_frame = frame.update({'int': 1}, new_log='With updated log')
print(_new_frame)
```

### Expired Keys

Sometimes objects are updated in-place and we want to indicate that this key is no longer valid in an existing `Frame`.
We can use the `expire()` function for this.

```{note}
This will rarely need to be called by the user;
instead, set `expiring_keys` in the relevant `History` constructor as described in the [History tutorial](history-expiring-keys).
```

```python {marimo}
frame.expire("data")
print(frame)
```

## What's next?

See the [API Reference](/_autosummary/loqs.core.frame.Frame) for more in-depth documentation of `Frame` objects.

Next, we will cover some of the built-in `LoQS` objects that are intended to be stored in a `Frame`.