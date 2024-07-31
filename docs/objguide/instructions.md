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

# Instructions

The `Instruction` is arguably the core (and most complex) object in `LoQS`.
This is because the goal is provide the user with ultimate flexibility in manipulating the state of the simulation, while simultaneously encapsulating the effects of arbitrary user-defined functions so that it plays nicely with the rest of the `LoQS` framework.

The role of the `Instruction` is to logical qubit simulation as the unitary or superoperator matrix is to physical qubit simulation -- it takes in the current simulation state and outputs the new simulation state
More concretely, an `Instruction` is primarily defined by a single user-defined function called `apply_fn` that takes any state it needs as arguments and outputs a single new `Frame`. However, there are several other things that may need to be defined in order for an `Instruction` to work properly, which will be covered here.

## The Apply Functions

As mentioned above, the `apply_fn` is at the core of an `Instruction`.
It takes in any state it requires as arguments and outputs a new `Frame`.
In the context of a `QuantumProgram` execution, the `QuantumProgram` will collect all the required arguments, call `Instruction.apply()` (which wraps `apply_fn`), and then appends the output `Frame` to the current `History`.
Note that `apply_fn` does not take `self` or `cls` as the first argument - it is best to think of this as a static function that needs all stateful information passed in explicitly.

### Basic Example: An Adder

```{note}
Please note that the examples shown here are designed to show basic `Instruction` functionality and are missing important context from the `QuantumProgram`.
Users are not intended to call `Instruction.apply()` directly like this.
```

For the remainder of this section, we will be using an adder as an example `Instruction`.
This will take an integer `data_val`(which would normally be provided from the last frame) and increment it by `add_val`.

```{code-cell} ipython3
from loqs.core import Frame, Instruction

def my_apply_fn1(data_val: int, add_val: int) -> Frame:
    new_val = data_val + add_val
    new_frame = Frame({"data_val": new_val})
    return new_frame

# Define our Instruction based on apply_fn
adder1 = Instruction(my_apply_fn1, name="Adder test 1")

# Apply it!
frame1 = adder1.apply(data_val=2, add_val=1)
print(frame1)
```

### Dry Runs

As mentioned in [the example workflow](/gettingstarted/workflow), it is possible to execute an `Instruction` in "dry run" mode.
This is toggled by the `dry_run` kwarg to `Instruction.apply()`, so we can show that here as well.

The behavior of an `Instruction` during a dry run is determined by the `dry_run_apply_fn`.
By default, if not provided, it falls back to the `apply_fn`.
This is typically good behavior for an `Instruction` that only modifies metadata or does not perform difficult computation.

This fallback is being used by our `adder1` instruction, so performing a dry run has the same effect as a real run.

```{code-cell} ipython3
# Apply it!
dry_run_frame1 = adder1.apply(dry_run=True, data_val=2, add_val=1)
print(dry_run_frame1)
```

However, let us pretend that our addition was actually some expensive operation. In this case, we have a few options for `dry_run_apply_fn`:

- We can define a function that has the same signature as `apply_fn`.
- We can pass in a default frame to be returned.
- We can pass a list of strings, which are assumed to be frame keys and given the placeholder value of "DRY_RUN".

Of these options, the last is probably the easiest in most cases and what we will showcase here.
More complex operations may need to use one of the first two options instead.

```{code-cell} ipython3
# Define our Instruction based on apply_fn and a placeholder for dry run
adder2 = Instruction(
    apply_fn=my_apply_fn1,
    # dry_run_apply_fn here is converted to a function that returns a dummy Frame with these keys
    dry_run_apply_fn=["data_val"],
    name="Adder test 2")

# Now the dry run has a placeholder instead
frame2 = adder2.apply(dry_run=True, data_val=2, add_val=1)
print(frame2)
```

## Instruction Data

Sometimes it make sense for the `Instruction` to store some data that is used.
This can be stored in the `Instruction.data` attribute and set by the constructor kwarg of the same name.

In the case of our example, maybe we want to set the `add_val` to be part of the instruction data.

```{code-cell} ipython3
# Define our Instruction with add_val as part of the data
adder3 = Instruction(
    apply_fn=my_apply_fn1,
    dry_run_apply_fn=["data_val"],
    data={"add_val": 1},
    name="Adder test 3")

# Normally the QuantumProgram would be responsible for extracting the value from the data
# Here, we do it manually and describe the argument collection process more later
frame3 = adder3.apply(data_val=2, add_val=adder3.data["add_val"])
print(frame3)
```

### Mapping Qubits in Instruction Data

If the `Instruction.data` attribute contains entries that have qubit labels, then we also need to define the `map_qubits_fn` function (e.g. `Instruction` objects with physical circuits, qubit permutations, feed-forward operations depending on qubit labels, etc.).
By default, the `map_qubits_fn` is a passthrough assuming no qubit labels need to be updated.

The default has been good for our adder `Instruction` thus far, but we can make it not the case to show what happens when qubits need to be mapped.
Instead of returning a bare `int`, let's return a dict with qubit label keys and store what that label is in `data`.

```{code-cell} ipython3
from collections.abc import Mapping


def my_apply_fn2(data_val: int, add_val: int, qubit_label: str) -> Frame:
    new_val = data_val + add_val
    new_frame = Frame({"data_val": {qubit_label: new_val}})
    return new_frame

def my_map_qubit_fn(qubit_mapping: Mapping[str, str], qubit_label: str, **kwargs):
    new_data = kwargs.copy() # All other kwargs can pass through without mapping
    new_data["qubit_label"] = qubit_mapping[qubit_label] # Map the qubit_label
    return new_data

# Define our Instruction with qubit_label as part of the data and a map_qubits_fn
adder4 = Instruction(
    apply_fn=my_apply_fn2,
    dry_run_apply_fn=["data_val"],
    data={"add_val": 1, "qubit_label": "Q0"},
    map_qubits_fn=my_map_qubit_fn,
    name="Adder test 4")

# Similar to last example, manually extract from data for standalone example
frame4 = adder4.apply(
    data_val=2,
    add_val=adder4.data["add_val"],
    qubit_label=adder4.data["qubit_label"])
print(frame4)
```

Now that we have defined the `map_qubits_fn`, we should be able to use the `Instrument.map_qubits()` function to get a new `Instruction` with the new qubit labels.

```{code-cell} ipython3
mapped_adder4 = adder4.map_qubits({"Q0": "Q1"})

mapped_frame4 = mapped_adder4.apply(
    data_val=2,
    add_val=mapped_adder4.data["add_val"],
    qubit_label=mapped_adder4.data["qubit_label"])
# Notice Q1 in the data_val output and Instruction data for qubit_label
print(mapped_frame4)
```

## Parameters

Throughout the previous examples, we have ignored how the `Instruction` gets fed the correct arguments.
The exact details of this are described in the [argument collection section of the QuantumProgram tutorial](quantumprogram-parameter-collection),
but it relies on two parameter-related attributes in the `Instruction`: priority and aliases.

(instruction-parameter-priorities)=
### Parameter Priorities

Arguments can generally come from four sources during `QuantumProgram` execution:

1. The `InstructionLabel` (covered in [later](instruction-labels))
2. The `Instruction.data` mentioned above
3. The `QuantumProgram` itself
4. The `History` from the program execution so far

Each argument to `apply_fn` can be assigned a "parameter priority" to let the `QuantumProgram` know what order to check these sources in.
This takes form as the `param_priority` kwarg in the `Instruction` constructor, which takes argument name as the key and list of priorities where the options are:

- `"label"` corresponding to the `InstructionLabel`
- `"instruction"` corresponding to `Instruction.data`
- `"program"` corresponding to the `QuantumProgram`
- `"history[<idx>]"` corresponding to the `History` where `<idx>` is the indices for [a `collect_data` call](history-collecting-data)

The default priority order is defined [here](/devguide/_autosummary/loqs.core.instructions.instruction.DEFAULT_PRIORITIES).

```{note}
`param_priorities` is automatically set by the `Instruction` construction. Users will only need to adjust this when they want to change the priority or if the `apply_fn` uses variadic kwargs (i.e. `**kwargs`), in which case the automatic functionality will not work properly.
```

### Parameter Priority Error Behavior

A related constructor kwarg is `param_error_behavior`, which determines what happens when the constructor fails to automatically set the priority.
This is set to `'warn'` by default, which throws a `UserWarning`.
It is possible to `'raise'` instead, which currently throws a `NotImplementedError` (it is a TODO to extend this to other parameter types and therefore this may eventually be implemented).
It is also possible to set it to `'continue'`, in which case the constructor silently continues.
If your `apply_fn` contains variadic kwargs and you have set `param_priorities` manually, then `'continue'` could be a good option so that the constructor does not warn about `**kwargs`.

### Parameter Aliases

The sharp-eyed may have noticed that `param_priorities` assumes that the argument name and frame/data key are the same.
This may not always be true; in this case, we can use `param_aliases` to map from `apply_fn` argument name to the frame/data key we expect.

```{note}
If using parameter aliases, then the keys for `param_priorities` should be the argument names and not frame/data keys.
The `Instruction.param_priorities` property does the aliasing when needed.
```

### Parameter Example

We can make a version of our adder with custom priorities and aliases to showcase these features.
Note that full use of this machinery requires context from a `QuantumProgram`, so we cannot show the full functionality here.

```{code-cell} ipython3
# Define this with argument names that don't match frame/data keys
def my_apply_fn3(dval: int, aval: int, qubit_label: str) -> Frame:
    new_val = dval + aval
    # The Frame does need to use the correct frame key
    new_frame = Frame({"data_val": {qubit_label: new_val}})
    return new_frame

# Let's define some custom priorities
# Maybe qubit_label should be explicitly read from data first and not at all from the History
# Anything not defined will use the defaults
param_priorities = {"qubit_label": ["instruction", "label", "program"]}

param_aliases = {"dval": "data_val", "aval": "add_val"}

# Define our Instruction with custom priorities and aliases
adder5 = Instruction(
    apply_fn=my_apply_fn3,
    dry_run_apply_fn=["data_val"],
    data={"add_val": 1, "qubit_label": "Q0"},
    map_qubits_fn=my_map_qubit_fn,
    param_priorities=param_priorities,
    param_aliases=param_aliases,
    name="Adder test 5")

# We can check what the QuantumProgram will see as its priorities
# Note that we have data_val and add_val here, not dval and aval
print(adder5.param_priorities)
```

```{code-cell} ipython3
# Similar to last example, manually extract from data for standalone example
# Note that we are still using the aliased values here
frame5 = adder5.apply(
    data_val=2,
    add_val=adder5.data["add_val"],
    qubit_label=adder4.data["qubit_label"])
print(frame5)
```

## What's next?

While this covered many of the main points in creating an `Instruction`, it may also be useful to see a more complex example in the ["Building a Complex Instruction" tutorial](/tutorials/buildinstruction).

See the [API Reference](/devguide/_autosummary/loqs.core.instructions.instruction.Instruction) for more in-depth documentation of `Instruction` objects.

Next, we will cover how to tell the `QuantumProgram` which instructions to run during execution using the `InstructionLabel` and `InstructionStack`.