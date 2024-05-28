

# Trajectories and Frames

```{warning}
This chapter is currently a plan/outline, and may not represent the exact implementation once completed. 
```

In this section, we'll look at how "state" information is stored in a `QuantumProgram`.
Since this is potentially different for every hardware platform, architecture, and quantum error correcting code, the design philosophy in `LoQS` is to provide a way for users to quickly generate specifications for what state information needs to be stored.

## Overview

Unfortunately, the word "state" quickly becomes overloaded when talking about simulating quantum devices.
To avoid this, all stateful information related to a simulated run of a logical circuit is held in "Trajectory"-type objects.
There are two "Trajectory"-type objects:

1. `TrajectoryFrame`: This is a snapshot of the relevant state information at a single point in the simulated logical circuit, often directly after an operation has been applied.
2. `Trajectory`: This is a stack-like list of `TrajectoryFrame` objects that together describe the entire simulatied logical circuit. This would be the output of an entire `QuantumProgram`.

## The `RecordSpec`

Short for "record specification", this is a `dict`-like object that links `str` keys to object types.
Essentially, this is how users describe what a possible `Record` looks like.
The `RecordSpec` itself does not store the data, but it will help define the interface for `Record` entries, ensuring:

1. All expected data is stored in a `Record`
1. Unexpected data does *not* get stored in a `Record`
1. The input `Record` contains all the necessary information for a given `Instruction`
1. A given `Instruction` returns an output `Record` with all expected information

The `RecordSpec` has the following validation functions:

* `check_class` validates whether a given class is a subclass of the type value in the `RecordSpec` for a given key
* `check_instance` validates whether a given object instance is an instance of the type value in the `RecordSpec` for a given key
* `create_record` takes input data and returns the resulting `Record` object

## The `Record`

This is a `dict`-like object that maps `str` keys to `IsRecordable` objects, as specified by a `RecordSpec`.
This is intended to be relatively immutable, enabling "snapshots" of the state along the simulation to enable all sorts of analysis after the fact.
It may be the case that some objects (such as the quantum state) are too large to actually save a snapshot of - TBD on how to handle this.

## The `RecordHistory`

This is essentially a list of `Record` objects.
TODO on the analysis this can do, but it at least stores the "standard" `RecordSpec` shared by all of its `Record` objects and the "nonstandard" `RecordSpec` for things only some `Records` store.

+++
