# `PhysicalCircuit` Backends

```{warning}
This chapter is currently a plan/outline, and may not represent the exact implementation once completed. 
```

All circuit backends are stored in `loqs.backends.circuit` and implement a `PhysicalCircuit`-type object.
Generally speaking, these objects wrap an underlying object that describes a physical circuit and can do some simple circuit manipulation, including:

* Concatenating circuits
* Changing qubit labels
* Removing qubits
* And a generic "processing" function required by `TemplatedCircuits` (more on that later)

## Current `PhysicalCircuit` Backends

We currently only have one circuit backend.

1. `pyGSTi`

```{note}
One other potential one we may want is Qiskit.
Some users may have QASM that they want to convert easily, and may not want to be restricted to pyGSTi parsing.
This is pretty low-priority though.
```