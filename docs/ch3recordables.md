# Recordables

```{warning}
This chapter is currently a plan/outline, and may not represent the exact implementation once completed. 
```

In this chapter, we cover what `LoQS` objects we provide that you can store in a `Record`.
The short, less useful answer is that anything that inherits from `loqs.utils.IsRecordable` can be stored!
But the longer answer includes some more details about what kinds of objects these can be. Some of them will be defined in `loqs.core.recordables`, but a few other core items are also recordable:

1. A `QuantumState`: Of course, simulation will involve propagating the underlying physical quantum state forward, so these must be able to be stored in a `Record`!
1. An `Instruction`: For provenance reasons, the best way to keep track of what `Instruction` generated a `Record`... is just to save the `Instruction` as well!
1. Measurement `Outcomes`: We definitely need these, e.g. syndrome measurements for decoding
1. A `StabilizerFrame`: We'll also need this if we want to keep track of the frame instead of applying corrections

Also, remember that the user can define their own recordables. Simply make a class that inherits from `IsRecordable`, and store whatever information you need!

```{note}
What am I missing from this list?
```