# Welcome to the Logical Qubit Simulator (LoQS)

The __Lo__gical __Q__ubit __S__imulator (`LoQS`) is intended to make it easy to implement and simulate logical qubits with a variety of quantum error correction (QEC) codes on many different hardware platforms and architectures.
While LoQS is not unique in this purpose, our goal is to provide the ability to simulate logical qubits with expressive noise models;
while many logical qubit simulators focus on Clifford or Pauli stochastic noise and stabilizer simulators such as `Stim` and `PECOS`, `LoQS` is built on  top of [`pyGSTi`](https://github.com/sandialabs/pyGSTi) and can handle noise models with arbitrary process matrices (at the cost of simulation speed).

In the rest of this documentation, we will lay out some of the design philosophies of `LoQS` and showcase how to perform logical qubit simulations with `LoQS`.

```{tableofcontents}
```
