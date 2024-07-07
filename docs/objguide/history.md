---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# History

At its core, a `History` is simply an list of `Frame` objects.

It is designed such that existing `Frame` objects cannot be overwritten, and a new `Frame` can only be added using the `append` operation.

## Basic Operation

You can initialize a `History` from a list of `Frame` objects, including an empty list.

```{note}
We will cover the `Frame` in the next section, but for now we pass in some dummy data/log string purely for demonstrative purposes.
```

```{code-cell} ipython3
from loqs.core import History, Frame

test_frames = [Frame({"index": i}, f"Title {i}") for i in range(5)]

history = History(test_frames)
```

Printing a `History` will also print all the underlying `Frame` objects.

```{code-cell} ipython3
print(history)
```

The core editing operation for a `History` is `append`, which adds a new `Frame`.
This operation returns a new `History`, maintaining the objects' immutability.

```{code-cell} ipython3
new_frame = Frame({"index": 5}, "New frame")
history.append(new_frame)
print(history)
```

The `History` uses `collections.abc.Sequence` as a base, rather than `collections.abc.MutableSequence`.
As a result, trying to set/override an existing `Frame` will error. 

```{code-cell} ipython3
:tags: [raises-exception]

# TypeError expected!
history[0] = new_frame
```

## Expiring and Propagating Keys

We rarely care about a single `Frame` by itself;
instead, we care about the larger context of the simulation so far.
This means that sometimes the `History` needs to adjust either the existing `Frame` objects or an incoming `Frame` to ensure the state of the simulation is recorded faithfully.
In particular, there are two mechanisms for this: expiring keys and propagating keys.

### Expiring Keys

As we cover in the next section, the data for each `Frame` is intended to be immutable and therefore updates are mostly done via copies.
However, some simulation objects are too unwieldy to constantly copy (most notably, the quantum state).
These objects are usually modified in-place; however, this means that all existing `Frame` objects with a reference to that object now need to changed,

Expiring keys are the solution to this problem.
During construction, the user can tell the `History` which keys should expire (i.e. correspond to objects that are updated in-place).
When a new `Frame` is added, its keys are checked against the set of expiring keys.
If the key exists in the new `Frame`, the `History` goes back and expires that key in previous `Frame` objects.

To showcase this, let's build our previous `History` example iteratively, but set `index` to be expiring. This means only the latest `index` should be available.

```{code-cell} ipython3
expiring_history = History(None, expiring_keys=["index"])

for tf in test_frames:
    expiring_history.append(tf)

print(expiring_history)
```

### Propagating Keys

The counterpart to expiring keys is propagating keys.
In this case, we want to ensure that a particular set of keys is always propagated forward and available in the latest frame.
This typically includes things like the quantum state and code patch metadata.

Similar to expiring keys, propagating keys can be set by the user during `History` construction.
When a new `Frame` is added, its keys are checked against the propagating keys.
If a key is not available, the `Frame` is updated with the previous frame's value for that key.

To showcase this, let's build our previous `History` example iteratively, but this time add a `'data'` key that we want to propagate.
We'll set this on frames 0 and 3 so we can see cases where the key both exists and doesn't exist. 

```{code-cell} ipython3
# This Frame functionality is covered in the next section
test_frames[0] = test_frames[0].update({"data": "Dummy data 1"})
test_frames[3] = test_frames[3].update({"data": "Dummy data 2"})

propagating_history = History([], propagating_keys=["data"])
for tf in test_frames:
    propagating_history.append(tf)

print(propagating_history)
```

## What's next?

See the [API Reference](/devguide/_autosummary/loqs.core.history.History) for more in-depth documentation of `History` objects.

Next, we will cover the building blocks of a`History`: the `Frame`.
