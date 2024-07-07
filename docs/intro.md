# Welcome to the Logical Qubit Simulator (LoQS)

The **Lo**gical **Q**ubit **S**imulator (`LoQS`) is intended to make it easy to implement and simulate logical qubits with a variety of quantum error correction (QEC) codes on many different hardware platforms and architectures.
While LoQS is not unique in this purpose, our goal is to provide the ability to simulate logical qubits with expressive noise models;
while many logical qubit simulators focus on Clifford or Pauli stochastic noise and stabilizer simulators such as `Stim` and `PECOS`, `LoQS` is built on  top of [`pyGSTi`](https://github.com/sandialabs/pyGSTi) and can handle noise models with arbitrary process matrices (at the cost of simulation speed).

With this in mind, the goal of `LoQS` is to be a platform for investigating more comprehensive noise models that Pauli stochastic noise.
This primary goal is supported by two subgoals:

1. Do so as efficiently as possible (with the realization that `LoQS` will never be as performant as stabilizer simulators due to the necessity of propagating more information forward)
2. Allow for a flexible API to make defining new QEC operations as simple as possible

In the rest of this documentation, we will describe how these subgoals have influenced the design philosophies of `LoQS` and showcase how to perform logical qubit simulations with `LoQS`.

```{tableofcontents}
```
