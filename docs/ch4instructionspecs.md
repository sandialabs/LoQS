# Instruction Specifications

```{warning}
This chapter is currently a plan/outline, and may not represent the exact implementation once completed. 
```

In this chapter, we cover what `InstructionSpec` objects we provide in `LoQS`.
These are all kept in `loqs.core.instructionspecs`, and include the following:

1. A `LogicalGate`: Probably the most obvious. This will require a `QuantumState`, output a `QuantumState`, and probably store a physical circuit and model. Maybe the state and model will be stored by the `QuantumProgram` instead, TBD.
1. A `Decoder`: This will take `Outcomes` (either from a `Record` or `RecordHistory`) and the current `StabilizerFrame` and output a new `StabilizerFrame`?
1. A `StackUpdater`: This is basically our classical conditional logic. It can look at the past state, the current `InstructionStack`, and return a new `InstructionStack` that has been modified. This is how things like try-until-success, extra rounds of syndrome extraction, conditional operations, or a `CompositeInstruction` should work.

```{note}
What am I missing from this list?
I guess some form of `Measurement`?
```