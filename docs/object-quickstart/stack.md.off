---
title: Stack
marimo-version: 0.23.1
---

```python {marimo}
import marimo as mo
```

# Instruction Labels and Stack

To continue the logical-to-physical analogy, if the `Instruction` is a unitary or superoperator matrix, then the `InstructionLabel` is the gate label consisting of the gate name and target qubits, and the `InstructionStack` is the full circuit description.

As described in the [Overview section](/markdown/overview), there is one major difference between the two that makes standard physical circuit constructions a suboptimal choice for describing logical "circuits":
the logical circuit can (and often does) change in time!
So rather than a static object, we utilize a "stack" where new instructions can be pushed on as needed.

(instruction-labels)=
## InstructionLabels

The `InstructionLabel` object is fairly straightforward: it takes either an `Instruction` or instruction label, a patch label, and then any user-defined args and kwargs to be passed on to the `Instruction`.
The latter inputs are just ways to pass in extra information that may not be available from other sources, so are not that interesting;
however, one may notice that you can pass in an "arbitrary" instruction label here, which is both convenient *and* mysterious.

The short version is that any instruction label passed in here will automatically "resolved" into an `Instruction` at runtime by the `QuantumProgram` (for more information, see the [this part of the `QuantumProgram` section](quantumprogram-instruction-resolution)).
This is both convenient for the user (it allows for entries of an `InstructionStack` to have human-readable strings instead of pointing to random Python objects) and useful for passing in data that is only known at runtime (e.g. physical circuits can be mapped onto the proper qubits based on the state of code patches).

An `InstructionLabel` can be initialized as a standalone object;
however, it is more commonly declared as a `tuple` and then the `InstructionStack` can cast them all to `InstructionLabel` objects as needed.
We will see an example of this below.

## The InstructionStack

As previously mentioned, the `InstructionStack` defines which operations are going to be run by the `QuantumProgram`.
The `InstructionStack` is intended to be largely immutable and derives from `collections.abc.Sequence`.
It has functions such as `append_instruction()`, `delete_instruction()`, `insert_instruction()`, and `pop_instruction()`, but these functions return a modified copy of the `InstructionStack`.
It is implemented this way so that previous stacks (which are commonly stored in `Frame` objects) are not invalidated, making it easier to debug any adaptive instructions.

## Basic Example

We have already implicitly seen the `InstructionStack` at work in the [workflow example](/markdown/workflow), but we can highlight some more of its features here.

```python {marimo}
from loqs.core import InstructionStack

# Here we take the stack from the workflow example
stack_list = [
    ("Init Dummy State", None, ["arg1", "arg2"], {"kwarg1": "val1"}),
    ("Non-FT Minus Prep", "L0"),
    ("Non-FT Logical X Measure", "L0")
]

stack1 = InstructionStack(stack_list)
print(stack1)
```

The `QuantumProgram` will pop instructions off the `InstructionStack` to execute them.

```python {marimo}
inst_label, stack2 = stack1.pop_instruction()
print(inst_label)
print(stack2)
```

Adaptive instructions may prepend new instructions to the `InstructionStack`.

```python {marimo}
stack3 = stack2.insert_instruction(0, ("New Instruction Label", "L0"))
print(stack3)
```

## What's next?

See the API Reference for [`InstructionLabel`](/_autosummary/loqs.core.instructions.instructionlabel.InstructionLabel) or [`InstructionStack`](/_autosummary/loqs.core.instructions.instructionstack.InstructionStack) for more in-depth documentation of respective objects.

Next, we will cover the `loqs.core.instructions.builder` module and see how several commonly used types of `Instructions` can be quickly built.