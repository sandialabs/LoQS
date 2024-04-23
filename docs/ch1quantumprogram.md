# The `QuantumProgram`

```{warning}
This chapter is currently a plan/outline, and may not represent the exact implementation once completed. 
```

As mentioned in the previous section, the `QuantumProgram` is the core object for describing and executing a logical circuit.

Currently, the `QuantumProgram` takes:

1. A `QECCode`, which will hold definitions for logical operations. Essentially, the `QECCode` is to logical operations as a `pyGSTi.models.OpModel` is to PTM representations of the noisy physical operations.
    1. Does this hold the noise model for the physical device? My gut feeling is no, but then that information needs to go somewhere else. Maybe just a standalone kwarg or passed into whatever executes the `QuantumProgram`
1. An input `InstructionStack`, which is what operations the user wants to execute. The actual stack executed may differ from this based on what occurs during the simulation (i.e. rounds of syndrome extraction added, etc), but this is the input deck.
1. An optional `RecordHistory`, which could be used to pass in state information. Maybe this `QuantumProgram` is being executed after another one without reset. Maybe this could be used for checkpointing/warmstarts. Uses TBD, but I'm pretty sure I want it coming in.



