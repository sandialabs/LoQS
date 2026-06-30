---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# First Steps with LoQS

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sandialabs/LoQS/docs-updates?filepath=docs/notebooks/shortcourse-firststeps.ipynb)

A rough analogy of pyGSTi $\rightarrow$ LoQS concepts:


| Concept |  pyGSTi | LoQS (physical level) | LoQS (logical level) |
| --- | --- | --- | --- |
| Series of operations to apply | Circuit | `<backend>`PhysicalCircuit | InstructionStack |
| Labels for operations | Labels | N/A | InstructionLabel |
| Custom operations | OperationFactories | N/A | Instruction |
| "Standard" operations | `*`Op (e.g., FullCPTPOp) | N/A | Helper functions to build "standard" Instructions |
| Mapping from labels to operations | `*`Model (e.g. ExplicitOpModel) |  `<backend>`NoiseModel | QECCode/QECCodePatch |
| The "state" | `*`State (e.g. CPTPState) | `<backend>`QuantumState | Frame (quantum **and** classical) |
| Flag for how to store state | Evotype | N/A | GateRep/InstrumentRep |
| Record of states after every operator | N/A | N/A | History (list of Frames) |
| Thing with the `.run()` | Protocols (e.g. GST) | N/A | QuantumProgram |
| Output of `.run()` | ModelEstimateResults | N/A | ProgramResults |
| Quickstart for operation mapping object| Modelpack | N/A | Codepack |

With this in mind and the documentation available, let's dive straight in!

+++

## Grab a codepack

Where most pyGSTi examples start with a modelpack, we'll start with a codepack. We currently have two choices: the 5-qubit code and the Steane (or [[7,1,3]] color code).

I'm going to use the Steane code here, but feel free to go through and try this with the 5Q code later for extra credit.

```python
from loqs.codepacks import codepack_7_1_3_quantinuum2021 as codepack_Steane
```

```python
# We can print the module's docstring to get more information on the codepack
# This is the same as what is in the documentation (with less nice formatting)
print(codepack_Steane.__doc__)
```

Codepacks offer two things: a QECCode and an ideal physical noise model. Let's extract those here.

In order to define our physical noise model, we must make a few choices: qubit labels, model backend, and gate/instrument/measurement representations.

```python
# Let's check what our available backends are
from loqs.backends import get_available_backends
get_available_backends()

## MAKE SURE YOU HAVE pygsti_circuit, pygsti_model, AND qsim_state AVAILABLE
# If you don't, reinstall with loqs[pygsti,quantumsim] (or `-e ."[pygsti,quantumsim]"`) at least and check again
```

```python
# Let's define qubits for a single Quantinuum-style Steane patch: 7 data qubits, 3 aux qubits
qubits = ["A0", "A1", "A2"] + [f"D{i}" for i in range(7)]

# Let's use a pyGSTi noise model
# LoQS Best Practice: Always import backends from loqs.backends, and not a further submodule
# This is because loqs.backends handles missing backends by doing lazy imports
from loqs.backends import PyGSTiNoiseModel

ideal_model = codepack_Steane.create_ideal_model(qubits=qubits, model_backend=PyGSTiNoiseModel)
```

The QECCode creation function also has several options. For the purposes of this tutorial, we'll just explicitly set the circuit backend as a reminder that it's there.

```python
from loqs.backends import PyGSTiPhysicalCircuit

steane_code = codepack_Steane.create_qec_code(circuit_backend=PyGSTiPhysicalCircuit)
```

Before we get too far, let's see what Instructions are available to us:

```python
#steane_code.instructions
# Or displayed more compactly:
list(steane_code.instructions.keys())
```

We see some |0> state preps, a few logical operations, a bunch of SE/QEC instructions, and then some measurements.

## Setting up our first program

Let's start simple: Let's do a state preservation experiment!

In order to do this, we need to create an initial History and set up an InstructionStack.

```python
from loqs.core import Frame, History, InstructionStack, PatchDict
from loqs.backends import NumpyStatevectorQuantumState, QSimQuantumState

# Our starting frame just needs two things:
# the physical quantum system, and
init_state = NumpyStatevectorQuantumState(10, qubit_labels=qubits)
# letting LoQS know we want this to be a Steane code patch
init_patches = PatchDict({
    "L0": steane_code.create_patch(qubits=qubits)
})

# And now we bundle it up
init_frame = Frame({"state": init_state, "patches": init_patches})
init_history = History(init_frame)

# And finally, let's create our stack for a |0> state preservation with one round of QEC
# The labels here are (key in steane_code.instructions, key in the Frame's patches)
# There are two more possible arguments, but we will skip them for now
stack = InstructionStack([
    ("FT Zero Prep", "L0"),
    ("Adaptive QEC", "L0"),
    ("FT Logical Z Measure", "L0")
])
```

And finally, we are ready to run some shots!

```python
from loqs.core import QuantumProgram

program1 = QuantumProgram(
    instruction_stack=stack,
    initial_history=init_history,
    default_noise_model=ideal_model,
    name="Zero state preservation"
)

results1 = program1.run(num_shots=100)
```

Great, but how do we look at the results?
We could print the full History for each shot, but the results object has some convenient collation tools.

There are some caveats: To use the collation tools effectively, you have to have some idea of what is in the Frames that you want to check.
In our case, all LoQS codepacks currently include logical_measurement in measurement Instructions.

```python
print(results1.shot_histories[0])
```

```python
results1.collect_shot_data(
    key="logical_measurement", # key in the Frame,
    indices=-1, # Check the last frame of each shot,
    return_counter=True # For convenience, just tell us how many of each output there were
)
```

## Exercise 1

#### Can you do this for the |1>, |+>, and |-> states?

You may be interested in the QuantumProgram.from_quantum_program function, which lets you copy a program but override some of the parameters.
For example:

```
program2 = QuantumProgram.from_quantum_program(program1, instruction_stack=new_stack)
```

This can be used to override almost anything in the program: noise model, base RNG, etc. Any fields not passed in are not overwritten (but deepcopied from the source program so that we don't have any copy/reference shenanigans).

<details>
<summary>Hint 1</summary>

Check what other instructions are available in the codepack that might let you access the other states.

</details>

<details>
<summary>Hint 2</summary>

Make sure you are measuring in the right basis!

</details>

---

<details>
<summary>Solution</summary>

We just need several different stacks:

For |1>
```
stack = InstructionStack([
    ("FT Zero Prep", "L0"),
    ("X", "L0"),
    ("Adaptive QEC", "L0"),
    ("FT Logical Z Measure", "L0")
])
```

For |+>
```
stack = InstructionStack([
    ("FT Zero Prep", "L0"),
    ("H", "L0"),
    ("Adaptive QEC", "L0"),
    ("FT Logical X Measure", "L0")
])
```

For |->
```
stack = InstructionStack([
    ("FT Zero Prep", "L0"),
    ("X", "L0")
    ("H", "L0"),
    ("Adaptive QEC", "L0"),
    ("FT Logical X Measure", "L0")
])
```

These are not unique solutions (e.g. HZ also works for |->), but it all boils down to changing the prep and meas bases.
Note that if you have a basis mismatch, you should get something close to 50/50.

</details>

## Exercise 2

#### Create a version of the state preservation test for the 5Q code instead of the Steane code.

Be aware that not all codepacks have the same instructions! As a hint, the 5Q codepack has |-> and X as the basic prep/meas, and Flagged QEC is the top-level QEC instruction.

---
<details>
<summary>Solution</summary>

We need to make a few changes: 

First, import and use the 5Q codepack instead of the Steane:

```
from loqs.codepacks import codepack_5_1_3_quantinuum2022 as codepack_5Q

# We only need 7 qubits: 5 data qubits, 2 aux qubits
qubits = ["A0", "A1"] + [f"D{i}" for i in range(5)]

ideal_model = codepack_5Q.create_ideal_model(qubits=qubits)

code_5Q = codepack_5Q.create_qec_code()
```

We also need to make sure to tell LoQS to use a 5Q patch instead of a Steane patch:
```
init_patches = PatchDict({
    "L0": code_5Q.create_patch(qubits=qubits)
})
```

And finally, we need to change the stack to instructions that the 5Q code has.
Rather than 0 and Z basis as the available operations, we have |-> and X basis instead.

```
stack = InstructionStack([
    ("FT Minus Prep", "L0"),
    ("Flagged QEC", "L0"),
    ("FT Logical X Measure", "L0")
])
```

and the other bases are analogous to Exercise 1, with the exception of not having a native Logical Z Measure, i.e. use H + Logical X Measure instead. The resulting circuits look a little silly when we think of the ideal noise model case.

</details>

+++
