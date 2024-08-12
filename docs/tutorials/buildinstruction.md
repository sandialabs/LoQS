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

# Building a Complex Instruction

While the [instruction builders](/objguide/builders) can create many of the `Instruction` objects we will need to implement QEC codes, they are not exhaustive.
This is particularly true for feed-forward operations that process classical data and have branching logic.
In this tutorial, we will cover how one can take a multi-stage feed-forward operation from a flowchart figure in a paper into something that `LoQS` can perform.

```{note}
If you have not read through the [`Instruction` object guide](/objguide/instructions), it is highly recommended you do so before reading this tutorial.
We will assume general familiarity with almost all `Instrument` functionality.
```

## Adaptive Measure Out in the [[5,1,3]] Code

We will show how to take the adaptive measure out scheme from Figure 13 of {cite}`buildinstruction-ryananderson_implementing_2022`, which is reproduced here for convenience.

TODO: Picture

We will create an `Instruction` and store it in an `instructions` dictionary that is intended to be the input to a `QECCode` constructor.
Branches in the flowchart will turn into `if/else` statements in the code that include stack updates that will reference the other `Instruction` objects as needed.

We will use the `PyGSTiPhysicalCircuit` as our [circuit backend](circuit-backends).

```{code-cell} ipython3
from collections.abc import Sequence
from typing import Mapping

from loqs.backends import propagate_state
from loqs.backends.circuit.basecircuit import BasePhysicalCircuit
from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.backends.model.basemodel import BaseNoiseModel
from loqs.backends.model.pygstimodel import PyGSTiNoiseModel
from loqs.backends.state.basestate import BaseQuantumState
from loqs.core import Instruction, QECCode
from loqs.core.frame import Frame
from loqs.core.instructions import builders
from loqs.core.instructions.instruction import KwargDict
from loqs.core.instructions.instructionlabel import InstructionLabel
from loqs.core.instructions.instructionstack import InstructionStack
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
```

```{code-cell} ipython3
# We will need to define a set to template qubits for physical circuits
template_qubits = ["A0", "A1"] + [f"D{i}" for i in range(5)]

# We will also define an dict to save the instructions into
instructions = {}
```

### Part I

Our part I `Instruction` is going to:
1. Propagate the state forward
    - Requires a circuit, model, state, and in-place flag
    - Generates a new state and measurement outcomes
2. Prepend the next `InstructionLabel` based on the measured flag qubit
    - Requires the stack and patch label
    - Generate a new stack

```{code-cell} ipython3
def partI_apply_fn(
    circuit: BasePhysicalCircuit,
    model: BaseNoiseModel,
    state: BaseQuantumState,
    inplace: bool,
    stack: InstructionStack,
    patch_label: str,
) -> Frame:
    # Run circuit
    new_state, outcomes = propagate_state(circuit, model, state, inplace)

    # Do classical feed forward
    flag_qubit = circuit.qubit_labels[1]
    F1 = outcomes[flag_qubit][0]
    if F1 == 0:
        # We go to part II (forward reference, must match key later)
        next_instruction = "Adaptive Measure Part II"
    else:
        # We go to decoding circuit  (forward reference, must match key later)
        next_instruction = "Non-FT Minus Unprep"

    # We need to make sure and feed the patch label forward
    new_label = InstructionLabel(next_instruction, patch_label)
    new_stack = stack.insert_instruction(0, new_label)

    # Return new frame
    frame_data = {
        "stack": new_stack,
        "state": new_state,
        "measurement_outcomes": MeasurementOutcomes(outcomes),
    }
    return Frame(frame_data)
```

For dry runs, we obviously won't have measurement outcomes to switch on. In that case, let's pretend that we did not flag so that we avoid early termination.

```{code-cell} ipython3
def partI_dry_run_apply_fn(stack: InstructionStack, patch_label: str, **kwargs) -> Frame:
    # Shortcut apply to go straight to part II feed forward
    new_label = InstructionLabel("Adaptive Measure Part II", patch_label)
    new_stack = stack.insert_instruction(0, new_label)

    frame_data = {
        "stack": new_stack,
        "state": "DRY_RUN",
        "measurement_outcomes": "DRY_RUN"
    }
    return Frame(frame_data)
```

Now we consider what data we want to store with this `Instruction`. 
Similar to other physical circuit instructions, we will store the physical circuit as well as any flags we pass in (just `inplace` in this case).

```{code-cell} ipython3
measI_circ = PyGSTiPhysicalCircuit(
    [
        ("Gh", "A0"),
        ("Gcphase", "A0", "D4"),
        ("Gcnot", "A0", "A1"),
        ("Gcnot", "A0", "D0"),
        ("Gcnot", "A0", "A1"),
        ("Gcphase", "A0", "D1"),
        ("Gh", "A0"),
        [("Iz", "A0"), ("Iz", "A1")],
    ],
    qubit_labels=template_qubits
)

partI_data = {
    "circuit": measI_circ,
    "inplace": True,
}
```

Since we are keeping a physical circuit in the `data`, we need to ensure that our `map_qubits_fn` maps this appropriately.
Looking ahead, it turns out that this `map_qubits_fn` will be sufficient for all of the `Instruction` implementations, so we'll name it without a suffix.

```{code-cell} ipython3
def map_qubits_fn(
    qubit_mapping: Mapping[str, str],
    circuit: BasePhysicalCircuit,
    **kwargs,
) -> KwargDict:
    new_kwargs = kwargs.copy()
    new_kwargs["circuit"] = circuit.map_qubit_labels(qubit_mapping)
    return new_kwargs
```

Finally we have all the components to define the entire instruction!

```{code-cell} ipython3
instructions["Adaptive Measure Part I"] = Instruction(
    partI_apply_fn,
    partI_dry_run_apply_fn,
    partI_data,
    map_qubits_fn,
    name="Part I of adaptive logical measurement",
)
```

### Part II

We can follow a similar pattern part II with one major difference: in this case, we also need the previous outcomes for our classical conditional logic.

Here, we choose to use a parameter alias to showcase that functionality.

```{code-cell} ipython3
def partII_apply_fn(
    circuit: BasePhysicalCircuit,
    model: BaseNoiseModel,
    state: BaseQuantumState,
    inplace: bool,
    previous_outcome: MeasurementOutcomes,
    stack: InstructionStack,
    patch_label: str,
) -> Frame:
    # Run circuit
    new_state, outcomes = propagate_state(circuit, model, state, inplace)

    # Pull measurements/flags
    meas_qubit = circuit.qubit_labels[0]
    flag_qubit = circuit.qubit_labels[1]
    F2 = outcomes[flag_qubit][0]
    M1 = previous_outcome[meas_qubit][0]
    M2 = outcomes[meas_qubit][0]

    # Do classical feed forward
    if F2 != 0:
        # We go to termination (forward reference, must match key later)
        next_instruction = "Adaptive Measure Termination"
    elif M1 == M2:
        # We go to part III (forward reference, must match key later)
        next_instruction = "Adaptive Measure Part III"
    else:
        # We go to decoding circuit (forward reference, must match key later)
            next_instruction = "Non-FT Minus Unprep"

    # We need to make sure and feed the patch label forward
    new_label = InstructionLabel(next_instruction, patch_label)
    new_stack = stack.insert_instruction(0, new_label)

    # Return new frame
    frame_data = {
        "stack": new_stack,
        "state": new_state,
        "measurement_outcomes": MeasurementOutcomes(outcomes),
    }
    return Frame(frame_data)
```

```{code-cell} ipython3
def partII_dry_run_apply_fn(stack: InstructionStack, patch_label: str, **kwargs) -> Frame:
    # Shortcut apply to go straight to part III feed forward
    new_label = InstructionLabel("Adaptive Measure Part III", patch_label)
    new_stack = stack.insert_instruction(0, new_label)

    frame_data = {
        "stack": new_stack,
        "state": "DRY_RUN",
        "measurement_outcomes": "DRY_RUN"
    }
    return Frame(frame_data)
```

```{code-cell} ipython3
measII_circ = PyGSTiPhysicalCircuit(
    [
        ("Gh", "A0"),
        ("Gcphase", "A0", "D0"),
        ("Gcnot", "A0", "A1"),
        ("Gcnot", "A0", "D1"),
        ("Gcnot", "A0", "A1"),
        ("Gcphase", "A0", "D2"),
        ("Gh", "A0"),
        [("Iz", "A0"), ("Iz", "A1")],
    ],
    qubit_labels=template_qubits,
)

partII_data = {
    "circuit": measII_circ,
    "inplace": True
}
```

```{code-cell} ipython3
# This step is new for Part II!
paramII_aliases = {"previous_outcome": "measurement_outcomes"}
```

```{code-cell} ipython3
# Remember that this key must match what Part I put for instruction label
instructions["Adaptive Measure Part II"] = Instruction(
    partII_apply_fn,
    partII_dry_run_apply_fn,
    partII_data,
    map_qubits_fn,
    param_aliases=paramII_aliases,
    name="Part II of adaptive logical measurement",
)
```

### Part III

We again follow a similar pattern, except that this time a further modification is required:
we need the past *two* measurement outcomes to do our conditional logic.
We will achieve this by also modifying our parameter priorities.

```{code-cell} ipython3
def partIII_apply_fn(
    circuit: BasePhysicalCircuit,
    model: BaseNoiseModel,
    state: BaseQuantumState,
    inplace: bool,
    previous_outcomes: list[MeasurementOutcomes],
    stack: InstructionStack,
    patch_label: str,
) -> Frame:
    # Run circuit
    new_state, outcomes = propagate_state(circuit, model, state, inplace)

    assert len(previous_outcomes) == 2

    # Pull measurements/flags
    meas_qubit = circuit.qubit_labels[0]
    flag_qubit = circuit.qubit_labels[1]
    F3 = outcomes[flag_qubit][0]
    M1 = previous_outcomes[-2][meas_qubit][0]
    M2 = previous_outcomes[-1][meas_qubit][0]
    M3 = outcomes[meas_qubit][0]

    # Do feed forward
    if F3 == 0 and M1 == M2 and M1 != M3:
        # Go to decoding circuit (forward reference, must match key later)
        next_instruction = "Non-FT Minus Unprep"
    else:
        # Otherwise, we terminate (forward reference, must match key later)
        next_instruction = "Adaptive Measure Termination"

    # We need to make sure and feed the patch label forward
    new_label = InstructionLabel(next_instruction, patch_label)
    new_stack = stack.insert_instruction(0, new_label)

    # Return new frame
    frame_data = {
        "stack": new_stack,
        "state": new_state,
        "measurement_outcomes": MeasurementOutcomes(outcomes),
    }
    return Frame(frame_data)
```

```{code-cell} ipython3
def partIII_dry_run_apply_fn(stack: InstructionStack, patch_label: str, **kwargs) -> Frame:
    # Shortcut apply to go straight to termination instruction
    new_label = InstructionLabel("Adaptive Measure Termination", patch_label)
    new_stack = stack.insert_instruction(0, new_label)

    frame_data = {
        "stack": new_stack,
        "state": "DRY_RUN",
        "measurement_outcomes": "DRY_RUN"
    }
    return Frame(frame_data)
```

```{code-cell} ipython3
measIII_circ = PyGSTiPhysicalCircuit(
    [
        ("Gh", "A0"),
        ("Gcphase", "A0", "D2"),
        ("Gcnot", "A0", "A1"),
        ("Gcnot", "A0", "D3"),
        ("Gcnot", "A0", "A1"),
        ("Gcphase", "A0", "D4"),
        ("Gh", "A0"),
        [("Iz", "A0"), ("Iz", "A1")],
    ],
    qubit_labels=template_qubits,
)

partIII_data = {
    "circuit": measIII_circ,
    "inplace": True,
}
```

```{code-cell} ipython3
paramIII_aliases = {"previous_outcomes": "measurement_outcomes"}

# This part is new for Part III!
paramIII_priorities = {"previous_outcomes": ["history[-2,-1]"]}
```

```{code-cell} ipython3
# Make sure our key matches the forward reference in part II
instructions["Adaptive Measure Part III"] = Instruction(
    partIII_apply_fn,
    partIII_dry_run_apply_fn,
    partIII_data,
    map_qubits_fn,
    param_priorities=paramIII_priorities,
    param_aliases=paramIII_aliases,
    name="Part III of adaptive logical measurement",
)
```

### State Unprep

The "decoder" circuit, or the $\ket{-}$ state unprep circuit, is simply a physical circuit instruction.
In this case, we can just use the builder directly.

```{code-cell} ipython3
state_unprep_circ = PyGSTiPhysicalCircuit(
    [
        [("Gcphase", "D0", "D4")],
        [("Gcphase", "D1", "D2"), ("Gcphase", "D3", "D4")],
        [("Gcphase", "D0", "D1"), ("Gcphase", "D2", "D3")],
        [
            ("Gh", "D0"),
            ("Gh", "D1"),
            ("Gh", "D2"),
            ("Gh", "D3"),
            ("Gh", "D4"),
        ],
        [
            ("Iz", "D0"),
            ("Iz", "D1"),
            ("Iz", "D2"),
            ("Iz", "D3"),
            ("Iz", "D4"),
        ],
    ],
    qubit_labels=template_qubits,
)

# Make sure our key matches the forward reference in previous parts
instructions["Non-FT Minus Unprep"] = (
    builders.build_physical_circuit_instruction(
        state_unprep_circ,
        name="Non-FT minus unprep circuit",
        reset_mcms=False,
    )
)
```

### Termination

Finally, we have our termination. In this case, we are just returning the previous outcome as the measurement.

```{code-cell} ipython3
def term_apply_fn(measurement_outcomes: MeasurementOutcomes, meas_qubit: str) -> Frame:
    return Frame({"logical_measurement": measurement_outcomes[meas_qubit][0]})
```

For our dry run, it is sufficient to use the convenience behavior of specifying frame keys to fill in with `"DRY_RUN"`.

```{code-cell} ipython3
term_dry_run = ["logical_measurement"]
```

The caveat is that which qubit of the measured outcomes to return is dependent on the template qubits, so we need to store this as data.

```{code-cell} ipython3
term_data = {"meas_qubit": "A0"}

def term_map_qubits_fn(
    qubit_mapping: Mapping[str, str], meas_qubit: str, **kwargs
) -> KwargDict:
    return {"meas_qubit": qubit_mapping[meas_qubit]}
```

We can keep the default parameter prioirities and have assigned no aliases, so can go straight to `Instruction` definition.

```{code-cell} ipython3
# Make sure this key matches forward references from previous parts
instructions["Adaptive Measure Termination"] = Instruction(
    term_apply_fn,
    term_dry_run,
    term_data,
    term_map_qubits_fn,
    name="Termination for adaptive logical measurement",
)
```

And voila! We have defined a complicated multistage feed-forward operation by relatively straightforward functions and a little bit of metadata management to ensure all the data is forwarded around properly.

## What's next?

The natural next step after defining your own instructions is to store them in a `QECCode`, which is what our next tutorial covers.

## References

```{bibliography}
:labelprefix: buildinstruction
:keyprefix: buildinstruction-
:filter: docname in docnames
:style: unsrt
```