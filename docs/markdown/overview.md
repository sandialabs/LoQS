
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
Later tutorials will include many more details on the individual objects, but at a high level:

- Instead of a "circuit" object, there is instead an `InstructionStack` that denotes the order of generic `Instructions` to be applied.
    * Crucially, the `InstructionStack` *can be updated during the simulation* with new `Instructions` to account for feed-forward operations, etc.
- Each `Instruction` outputs a `Frame` object, which is a snapshot of the "state" of the simulation. The `Frame` objects are then collected into a `History` that can be used to pass around the entire record of the simulation up to that point.
    * Like physical circuit simulation, the `History` will contain the underlying physical quantum state.
    * Unlike physical circuit simulation, each `Frame` can *also* include additional information (e.g. measurement outcomes, stabilizer frames, a new stack to use, etc.)
    * There is *no restriction* on what information can be stored in a `Frame`. This means that users can output their own data from an `Instruction`, and `LoQS` will seemlessly move this information forward for future `Instruction` objects.
- `QECCode` objects contain all the `Instruction` objects needed to implement a specific QEC code
    * One can also define a `QECCodePatch`, which allows entries in the `InstructionStack` to quickly run an `Instruction` from the `QECCode` on a "patch" while abstracting away the mapping of the patch onto physical qubits
- Finally, the `QuantumProgram` object contains everything needed in order to run a logical qubit simulation: an `InstructionStack`, any initial `History` needed, a noise model, and all available `Instructions` (either "global" or from a `QECCode`).
- The physical circuit simulation is abstracted away through the use of three types of backends: circuit, noise model, and state.
    * The goal is to be able to define a `QuantumProgram` in a backend-agnostic way. This is currently not quite true, but very few direct backend package calls are required and it is a continued aspiration.
