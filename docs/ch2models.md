# `NoiseModel` Backends

```{warning}
This chapter is currently a plan/outline, and may not represent the exact implementation once completed. 
```

All model backends are stored in `loqs.backends.model` and implement a `NoiseModel`-type object.
Generally speaking, these objects wrap an underlying object that can take a circuit and return a representation that can act on the desired state object.

## Current `OpRep` options

We support three types of representations currently:

1. Unitaries
1. Pauli transfer matrices (PTMs)
1. QuantumSim-basis PTMs

```{note}
There are several other options we probably want to implement, including but not limited to:
1. Generic process matrices in any basis?
1. Kraus representations
1. Some stabilizer/symplectic stuff for Clifford-only simulation?
We should only implement as needed though.
```

## Current `NoiseModel` backends

We currently only have one model backend.

1. `pyGSTi`

```{note}
Again, we may want to consider some other common packages with simple noise models, e.g. Qiskit, etc.
```