---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
---

# Overview

The idea of a physical circuit to describe the instructions for a physical quantum processor is ubiquitous and fairly straightforward.
Nearly all operations are quantum ones, and thus can be represented by their qubit support (i.e. which horizontal lines they touch) and time of execution (i.e. which vertical "slice" they fall into).
Measurements are generally local operations which can optionally dump their output to a classical line and/or condition a possible future operation;
often, double lines are used to indicate these classical channels, but they are a small auxiliary part to the otherwise quantum circuit.

While many of these conventions carry up to describing the instructions for a logical quantum processor, a simple circuit with qubit lines and time slices of quantum operations is not really sufficient to capture the complexity of a logical circuit.
In short, the amount of classical information and processing is not insignificant compared to the quantum portion.
Examples of this include, but are not limited to: keeping track of stabilizer frames rather than applying recovery operations directly, doing repeat-until-success operations for things like logical state preparation, running additional rounds of syndrome extraction if errors are detected, and quantum operations that involve physical qubits outside of logical qubits (e.g., split/merge operations in lattice surgery).
Additionally, the type of classical information and processing needed changes based on the physical hardware, architecture, and type of quantum error correction code being used.

## High-Level Design Philosophy for Core Objects

The core objects in `LoQS` have been designed with these differences in mind.
Detailed documentation and usage examples for all of these classes can be found directly in the [API Reference](/reference). At a high level:

- Instead of a "circuit" object, there is instead an [InstructionStack](/reference/loqs/core/instructions/instructionstack/InstructionStack/) that denotes the order of generic [Instruction](/reference/loqs/core/instructions/instruction/Instruction/) objects to be applied.
    * Crucially, the [InstructionStack](/reference/loqs/core/instructions/instructionstack/InstructionStack/) *can be updated during the simulation* with new [Instruction](/reference/loqs/core/instructions/instruction/Instruction/) objects to account for feed-forward operations, etc.
- Each [Instruction](/reference/loqs/core/instructions/instruction/Instruction/) outputs a [Frame](/reference/loqs/core/frame/Frame/) object, which is a snapshot of the "state" of the simulation. The [Frame](/reference/loqs/core/frame/Frame/) objects are then collected into a [History](/reference/loqs/core/history/History/) that can be used to pass around the entire record of the simulation up to that point.
    * Like physical circuit simulation, the [History](/reference/loqs/core/history/History/) will contain the underlying physical quantum state.
    * Unlike physical circuit simulation, each [Frame](/reference/loqs/core/frame/Frame/) can *also* include additional information (e.g. measurement outcomes, stabilizer frames, a new stack to use, etc.)
    * There is *no restriction* on what information can be stored in a [Frame](/reference/loqs/core/frame/Frame/). This means that users can output their own data from an [Instruction](/reference/loqs/core/instructions/instruction/Instruction/), and `LoQS` will seamlessly move this information forward for future [Instruction](/reference/loqs/core/instructions/instruction/Instruction/) objects.
- [QECCode](/reference/loqs/core/qeccode/QECCode/) objects contain all the [Instruction](/reference/loqs/core/instructions/instruction/Instruction/) objects needed to implement a specific QEC code.
    * One can also define a [QECCodePatch](/reference/loqs/core/recordables/qeccodepatch/QECCodePatch/), which allows entries in the [InstructionStack](/reference/loqs/core/instructions/instructionstack/InstructionStack/) to quickly run an [Instruction](/reference/loqs/core/instructions/instruction/Instruction/) from the [QECCode](/reference/loqs/core/qeccode/QECCode/) on a "patch" while abstracting away the mapping of the patch onto physical qubits.
- Finally, the [QuantumProgram](/reference/loqs/core/quantumprogram/QuantumProgram/) object contains everything needed in order to run a logical qubit simulation: an [InstructionStack](/reference/loqs/core/instructions/instructionstack/InstructionStack/), any initial [History](/reference/loqs/core/history/History/) needed, a noise model, and all available [Instruction](/reference/loqs/core/instructions/instruction/Instruction/) objects (either "global" or from a [QECCode](/reference/loqs/core/qeccode/QECCode/)).
- The physical circuit simulation is abstracted away through the use of three types of [simulation backends](/reference/loqs/backends/): circuit, noise model, and state.
    * The goal is to be able to define a [QuantumProgram](/reference/loqs/core/quantumprogram/#loqs.core.quantumprogram.QuantumProgram) in a backend-agnostic way. This is currently not quite true, but very few direct backend package calls are required and it is a continued aspiration.
