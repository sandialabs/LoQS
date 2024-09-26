# (Advanced) Instruction Builders

```{warning}
Although this is located in the Object Quickstart section, this should probably be considered an advanced topic.
It is not required to understand *how* the builders work in order to use them, although understanding them may avoid misuse/errors in some heretofore uncaught edge cases.

This section could still be helpful to power users who are trying to write complex `Instruction` objects themselves and want to see how almost every part of the `Instruction` API has been used in practice.

If this power user description does not apply to you and/or the wall of text below seems intimidating, you have two options:

1. For the more-technically inclined, there is frankly more text in these explanations than lines of code. If you are very comfortable with the [Instruction guide](/objguide/instructions), it may be useful to read the source code directly by following the links to the API reference and then come back to this page if/when the inline comments leave you puzzled.
2. For the more-application inclined, a showcase of how to use many of these builders can be found in the [Building a Codepack tutorial](/tutorials/buildqeccode). Feel free to assume the black box works, skip all of this, and go do some actual science.
```

The downside of flexibility is complexity, and that is certainly true of constructing an `Instruction` from scratch.
Luckily, there are several kinds of `Instruction` types that are used ubiquitously throughout QEC implementations that have been provided
in the `loqs.core.instructions.builders` module.

This section will cover these convenience functions in alphabetical order.
Note that this is not necessarily the easiest way to understand these builders.
Someone looking for a more pedagogical/simple-to-complex ordering may consider starting with: the physical circuit instruction (which is not the simplest but probably the most familiar to most users); then the patch instructions (which are the simplest); moving on to the lookup decoder instruction; and finally the object builder, composite, and repeat-until-success instructions.

```{note}
Demonstrating these functions is difficult to do without talking about [circuit backends](circuit-backends) or [code patches](qec-code-patches) first.

Rather than have constant forward references, we will only talk about the builders at a high-level here and leave demonstrations to the [Building a QEC Code tutorial](/tutorials/buildqeccode).
```

Even with a high-level overview, there can still be a lot going on in each builder.
We will start with the inputs to the function to provide context for the rest of the discussion.
All builder functions take `name` as flags to pass on to the `Instruction` constructor.
We will ignore talking about those here and focus on the other unique parameters for each builder.
Then we will use subsections mirroring the [Instruction guide](/objguide/instructions) to present the information: the apply functions, instruction data and data qubit mapping, and parameter priority/alias specifications.

## Composite Instructions

We often want to bundle instructions together, either to build complex instructions from smaller building blocks or just to make writing out the `InstructionStack` simpler.
This can be achieved with the `build_composite_instruction()` function, which is essentially a stack update that prepends a given list of `Instructions`.

### Inputs

Building a composite instruction requires two inputs:

- A list of `Instruction` objects
- Parameter priorities for the composite `Instruction`

Check the [API documentation](/devguide/_autosummary/loqs.core.instructions.builders.build_composite_instruction) for the full docstring.

### Apply

Unfortunately, the `apply_fn` for the composite instruction is one of the least specified. THis is because because we do not know what the `apply_fn` functions for the underlying instructions need until runtime; therefore, we have to define the composite apply function as `apply_fn(**kwargs)`.

Since this is just a stack update, the output `Frame` contains the new stack with the `"stack"` frame key.

### Data and Qubit Mapping

The only data we need to store is the list of `Instruction` objects to insert into the stack.

The `map_qubits_fn` simply calls `map_qubits()` on each `Instruction`.

### Parameters

As mentioned in [the previous section](instruction-parameter-priorities), `param_priorities` is a mandatory argument when variadic kwargs are used.

Luckily, we do not need to define *all* the parameter priorities here;
instead, we only need to define any parameters that any underlying `Instruction` needs from a `"label"` source (i.e. patch labels, label args/kwargs).
These parameters will then be added to the generated `InstructionLabel` objects.
All other parameters can use the other sources at runtime like normal.

In addition to any user-defined parameters, the composite instruction `apply_fn` also requires `"patch_label"` (typically taken from the `InstructionLabel` source), `"stack"` (typically taken from the `QuantumProgram` source), and `"instructions"` (typically taken from the `Instruction.data` source).
These are automatically added by the builder after processing the user's supplied parameter priorities.



## Lookup Decoders

```{warning}
This section is incomplete. Finish this when the lookup decoder API stabilizes.
```



## Object Builder

The object builder `Instruction` type is almost a meta-instruction.
It allows users to tell the `QuantumProgram` to build `LoQS` objects by essentially using an object's constructor as the `apply_fn`.
This is particularly useful to initialize objects such as quantum states at the start of program execution.

### Inputs

The object builder takes two arguments:

- A frame key that determines what key in the output `Frame` the new object is saved to
- The object class that we want to initialize

Check the [API documentation](/devguide/_autosummary/loqs.core.instructions.builders.build_object_builder_instruction) for the full docstring.

### Apply

Like the composite instruction, we don't know what arguments we need until runtime so we define the apply function as `apply_fn(**kwargs)`.
Unlike the composite instruction, we are just directly passing all the kwargs into the constructor to build the relevant object.

The created object is then stored in the output `Frame` with the given frame key.

### Data and Qubit Mapping

We store the frame key and object class in data.

Neither depends on any qubit information, so we can use the default passthrough `map_qubits_fn`.

### Parameters

Like the composite instruction, we have to explicitly include `param_priorities` because we have variadic kwargs.
Unlike composite instruction, it is significantly easier to automatically do this since we only have one function to handle (the object constructor) and we resolve it immediately.
As such, the user does not need to specify this manually -- we simply set all constructor kwargs to have the default priority.
As a side note, it's almost surely the case that the constructor kwargs will come from the `InstructionLabel` source since the user is trying to initialize a specific object;
however, there's (currently) no (obvious) harm in keeping the other sources in the priority list.



## Patch Builder

The patch builder is like the object builder in that it is "meta"-instruction designed to create a `QECCodePatch` object and add it to the main `PatchDict`.

### Inputs

The patch builder take only a single argument:

- The `QECCode` that will be used to generate the `QECCodePatch`

Check the [API documentation](/devguide/_autosummary/loqs.core.instructions.builders.build_patch_builder_instruction) for the full docstring.

### Apply

The `apply_fn` is straightforward in this case. We take the following arguments:

- `patch_label`: A label that will become the key into the `PatchDict`
- `qubits`: A list of qubits to use in the `QECCode.create_patch()` function
- `qec_code`: A `QECCode` to call `create_patch()` from
- `patches`: A `PatchDict` to insert the new patch into

It returns a `Frame` with an updated `PatchDict` assigned to the `"patches"` key.

### Data and Qubit Mapping

We store the passed-in `qec_code` in the data.
Additionally, we provide a default empty `PatchDict` in case there is no `"patches"` key in the History (i.e. we are trying to add the first patch).

Since both the `QECCode` and an empty `PatchDict` are qubit-independent, we can use the default passthrough `map_qubits_fn`.

### Parameters

The default `param_priorities` are mostly fine, with the exception of `"patches"`.
In this case, we would like to try and pull from `History` first before falling back to the default empty `PatchDict` stored in the `Instruction.data`.
Because `"patches"` is by default a propagating key, we assume we can just check the last frame for it.



## Patch Remover

The patch remover -- like the patch builder -- is another "meta"-instruction. This one is even simpler and is designed to remove a `QECCodePatch` object from the main `PatchDict`.

### Inputs

The patch builder takes no additional arguments.

Check the [API documentation](/devguide/_autosummary/loqs.core.instructions.builders.build_patch_remover_instruction) for the full docstring.

### Apply

The `apply_fn` is straightforward in this case. We take the following arguments:

- `patch_label`: The key to remove from the `PatchDict`
- `patches`: A `PatchDict` to remove the new patch frome

It returns a `Frame` with an updated `PatchDict` assigned to the `"patches"` key.

### Data and Qubit Mapping

We store no data, and thus we can use the default passthrough `map_qubits_fn`.

### Parameters

The default `param_priorities` are fine across the board, so no action is needed.



## Patch Permuter

Like the patch builder and remover, the patch permuter is a "meta"-instruction that only affects a `QECCodePatch` object in the main `PatchDict`.
Unlike them, however, the patch permuter often signifies some logical operation -- we are effectively permuting the qubit labels in the patch rather than performing physical swaps, but this *could* have been carried out by a physical circuit of swaps.

### Inputs

The patch permuter takes one additional argument:

- `mapping`: The qubit mapping with initial labels as keys and final labels as values

Check the [API documentation](/devguide/_autosummary/loqs.core.instructions.builders.build_patch_permute_instruction) for the full docstring.

### Apply

The `apply_fn` is straightforward in this case. We take the following arguments:

- `patch_label`: The key to modify int the `PatchDict`
- `mapping`: The qubit mapping containing the permutation (initial as keys, final as values)
- `patches`: A `PatchDict` containing the `QECCodePatch` to modify

It returns a `Frame` with an updated `PatchDict` assigned to the `"patches"` key.

### Data and Qubit Mapping

We store the `mapping` in `data`.

This is obviously qubit-dependent, so we define a `map_qubits_fn` that maps both the initial and final qubit labels in `mapping`.

### Parameters

The default `param_priorities` are fine across the board, so no action is needed.



## Physical Circuits

This is the `Instruction` that most people expect: apply a physical circuit onto a physical quantum state to perform forward simulation.
In this case, we want our physical circuits to implement some logical operation.

### Inputs

The physical circuit instruction takes four additional argument:

- `circuit`: The physical circuit to be applied
- `include_outcomes`: A flag that determines if `measurement_outcomes` are added to the output `Frame`
- `inplace`: A flag that determines if state propagation is done in-place
- `reset_mcms`: A flag that determines if qubits are reset after mid-circuit measurements

Check the [API documentation](/devguide/_autosummary/loqs.core.instructions.builders.build_physical_circuit_instruction) for the full docstring.

### Apply

The `apply_fn` is straightforward in this case. We take the following arguments:

- `model`: The noise model used to turn the physical circuit actionable operators
- `circuit`: The physical circuit to be applied
- `state`: The quantum state to be propagated forward
- `include_outcomes`: A flag that determines if `measurement_outcomes` are added to the output `Frame`
- `inplace`: A flag that determines if state propagation is done in-place
- `reset_mcms`: A flag that determines if qubits are reset after mid-circuit measurements

It performs the state propagation and then returns a `Frame` with the updated state assigned to the `"state"` key.
If `include_outcomes=True`, then the `Frame` will also contain a `MeasurementOutcomes` object assigned to the `"measurement_outcomes"` key.

### Data and Qubit Mapping

We store the `circuit` and all three flags in `data`.

The `circuit` is qubit-dependent, so we define a `map_qubits_fn` that uses the `BaseCircuitBackend.map_qubit_labels()` function to map the circuit as needed.

### Parameters

The default `param_priorities` are fine across the board, so no action is needed.



## Repeat-until-success Instructions

The repeat-until-success (RUS) instruction is a specific type of a feed-forward operation: it places itself back onto the stack until a certain `"measurement_outcome"` is observed.
The stack manipulation aspect makes it like the composite instruction, but it is not *purely* a stack operation.
The self-referential nature also makes it unique and require some post-processing.

### Inputs

The RUS instruction takes thre additional argument:

- `instruction`: The `Instruction` to repeat
- `success_fn`: A function that determines whether the `Instruction` succeeded based on the `"measurement_outcomes"`
- `max_repeats`: A max number of recursions allowed (to prevent infinite loops)

Check the [API documentation](/devguide/_autosummary/loqs.core.instructions.builders.build_repeat_until_success_instruction) for the full docstring.

### Apply

The feed-forward nature makes this one of the more complex `apply_fn` implementations.
Like the composite instruction, we don't know what arguments we need until runtime so we define the apply function as `apply_fn(**kwargs)`.

Some of these `kwargs` are needed by the RUS `Instruction` itself:

- `instruction`: The `Instruction` to apply/repeat if failed
- `self`: The RUS `Instruction` that should be added back to the stack on a failure
- `success_fn`: The function to use for testing `instruction` success
- `max_repeats`: A max number of recursions allowed (to prevent infinite loops)
- `repeat_count`: The current recursion level of the RUS
- `stack`: The current `InstructionStack`

The rest of the `kwargs` are passed on to `Instruction.apply()` to generate the new `Frame`.
The `"measurement_outcomes"` are pulled out of the `Frame` and passed to `success_fn` with two possible outcomes:

- On a success, we simply return the output `Frame`.
- On a failure, we increment `repeat_count`, check against `max_repeats`, add the RUS `Instruction` back onto the `stack`, and add the updated stack to the output `Frame` before returning

### Data and Qubit Mapping

We store the `instruction`, `success_fn`, `max_repeats`, and (critically) the `repeat_count`.

The `instruction` could be qubit-dependent, so we define a `map_qubits_fn` that uses the `Instruction.map_qubitss()` function to map it as needed.

### Parameters

The default `param_priorities` are fine across the board, so no action is needed.

Note that the `repeat_count` will start using the default value in `Instruction.data`, but afterwards will be taken from the `InstructionLabel` data.



## What's next?

As mentioned above, consider seeing many of the builders in action in the ["Building a QEC Code" tutorial](/tutorials/buildqeccode).

Before jumping to that tutorial, it may be worth looking at our next section where we explore the `QECCode` and `QECCodePatch` objects (which directly precedes the section on codepacks themselves).