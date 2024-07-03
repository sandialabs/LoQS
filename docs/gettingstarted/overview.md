# Specifying and Simulating Logical Circuits

The idea of a physical circuit to describe the instructions for a physical quantum processor is ubiquitous and fairly straightforward.
Nearly all operations are quantum ones, and thus can be represented by their qubit support (i.e. which horizontal lines they touch) and time of execution (i.e. which vertical "slice" they fall into).
Measurements are generally local operations which can optionally dump their output to a classical line and/or condition a possible future operation;
often, double lines are used to indicate these classical channels, but they are a small auxiliary part to the otherwise quantum circuit.

While many of these conventions carry up to describing the instructions for a logical quantum processor, a simple circuit with qubit lines and time slices of quantum operations is not really sufficient to capture the complexity of a logical circuit.
In short, the amount of classical information and processing is not insignificant compared to the quantum portion.
Examples of this include, but are not limited to: keeping track of stabilizer frames rather than applying recovery operations directly, doing repeat-until-success operations for things like logical state preparation, running additional rounds of syndrome extraction if errors are detected, and quantum operations that involve physical qubits outside of logical qubits (e.g., split/merge operations in lattice surgery).
Additionally, the type of classical information and processing needed changes based on the physical hardware, architecture, and type of quantum error correction code being used.

In this chapter, we will look at the `QuantumProgram` object, which is how logical circuits are specified in `LoQS`. We will show how both quantum and classical `Instructions` can be specified, how the state information can be specified using `Records`, and how the simulation of a `QuantumProgram` works.