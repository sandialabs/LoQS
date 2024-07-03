# Instructions

```{warning}
This chapter is currently a plan/outline, and may not represent the exact implementation once completed. 
```

In this section, we'll look at how we can move the state of the simulation forward through `Instructions`. As with `Record`-type objects, this can vary greatly depending on the architecture and quantum error correcting code, so we provide a way for users to specify what state is needed for input, what state is expected on output, and how to propagate the necessary state forward.

## Overview

The expected structure of the four `Instruction`-type objects is as follows:

* All information on how an instruction operations is stored in `Instruction` objects.
* In analogy to `RecordSpecs`, these are specified using (potentially user-defined) `InstructionSpec` objects.
* A `CompositeInstruction` holds a list of `Instructions` to be executed.
* An `InstructionStack` is a mutable list of `Instructions`. This will be a necessary input to the `QuantumProgram`.
    * Not sure if this makes `CompositeInstruction` obselete... TBD

## The `InstructionSpec`

Short for "instruction specification", this contains an input `RecordSpec` and output `RecordSpec` describing the state information coming in/going out of the parent `Instruction`.
It also provides a convenience function called `check_record` which verifies that an incoming `Record` can be acted upon by this type of `InstructionSpec`.

## The `Instruction`

This takes an `InstructionSpec`, a `name` for logging, a `parent` for provenance, and finally has a member function called `apply()`.
The `apply()` function is where the actual technical detail of the `Instruction` get applied to the incoming `Record` and the new state of the simulation is output in another `Record`.
Note that an `Instruction` object can also take a `RecordHistory`, in which case the "standard" `RecordSpec` of the history must match the input `RecordSpec`.
This could be useful for decoding with global syndrome data, for example.

## The `CompositeInstruction`

Currently this is basically just a list of Instructions.
The `apply()` is not implemented yet, but it will essentially be like a `StackUpdater` instruction. More on that later.

## The `InstructionStack`

This is more or less the logical "circuit".
It's a list of `Instruction` objects to be applied, and serves as the main input to the `QuantumProgram`.
The actually executed set of `Instruction` objects may differ from the input as some `Instruction` objects can modify the expected stack.
