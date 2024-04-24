# `QuantumState` Backends

```{warning}
This chapter is currently a plan/outline, and may not represent the exact implementation once completed. 
```

All state backends are stored in `loqs.backends.state` and implement a `QuantumState`-type object.
Generally speaking, these objects wrap an underlying object that describes the state of a physical quantum device.
It should know how to apply one of the representations out of a `NoiseModel` to its' underlying state.

## Current `QuantumState` Backends:

We currently only have one circuit backend.

1. `qsim` for QuantumSim==0.2.0

```{important}
This will be the key place for us to add new simulation capabilities.
While the other backends are more or less bookkeeping, this one does the actual simulation.

For proof of concept, we can add a pyGSTi backend but we should also consider what we really want.
STIM for Clifford would also be one I would be interested to add.
```