# Instruction Builders

The downside of flexibility is complexity, and that is certainly true of constructing an `Instruction` from scratch.
Luckily, there are several kinds of `Instruction` types that are used ubiquitously throughout QEC implementations that have been provided
in the `loqs.core.instructions.builders` module.

This section will cover these convenience functions in alphabetical order.
Note that this is not necessarily the easiest way to understand these builders.
Someone looking for a more pedagogical/simple-to-complex ordering may consider starting with the physical circuit, moving on to the patch instructions and lookup decoder, and finally the composite instruction, repeat-until-success, and object builders.

```{note}
Demonstrating these functions is difficult to do without talking about [circuit backends](circuit-backends) or [code patches](qec-code-patches) first.

Rather than have constant forward references, we will only talk about the builders at a high-level here.
To see the builders in action, check out the [Building a Codepack tutorial](/tutorials/buildcodepack) after this.
```

All builder functions take `name`, `parent`, and `fault_tolerant` as flags to pass on to the `Instruction` constructor.
We will ignore talking about those here and focus on the other unique parameters for each builder.

## Composite Instructions

We often want to bundle instructions together, either to build complex instructions from smaller building blocks or just to make writing out the `InstructionStack` simpler.
This can be achieved with the `build_composite_instruction()` function -- check the [API documentation](/devguide/_autosummary/loqs.core.instructions.builders.build_composite_instruction) for the full docstring.

### Inputs

Building a composite instruction requires two inputs:

- A list of `Instruction` objects
- Parameter priorities for the composite `Instruction`

The latter is needed because we do not know what the `apply()` functions for the underlying instructions need until runtime; therefore, we define the composite apply function as `apply_fn(**kwargs)`.
As mentioned in [the previous section](instruction-parameter-priorities), `param_priorities` is a mandatory argument when variadic kwargs are used.

Luckily, we do not need to define *all* the parameter priorities here; instead, we only need to define any parameters that any underlying `Instruction` needs 

### Apply 


### Reserved Keys

## Lookup Decoders

```{warning}
This section is incomplete. Finish this when the lookup decoder API stabilizes.
```

## Object Builder

TODO

## Patch Builder

## Patch Remover

## Patch Permuter

## Physical Circuits

This is the `Instruction` that most people expect: apply a physical circuit onto a physical quantum state to perform forward simulation.
In this case, we want our physical circuits to implement some logical operation.



Check the [API documentation](/devguide/_autosummary/loqs.core.instructions.builders.build_physical_circuit_instruction) for the full docstring.

## Repeat-until-success Instructions

## What's next?

As mentioned above, consider seeing many of the builders in action in the ["Building a Codepack" tutorial](/tutorials/buildcodepack).

Before jumping to that tutorial, it may be worth looking at our next section where we explore the `QECCode` and `QECCodePatch` objects (which directly precedes the section on codepacks themselves).