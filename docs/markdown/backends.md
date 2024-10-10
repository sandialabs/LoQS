
# Simulation Backends

Traditionally, `LoQS` has been built on top of `pyGSTi` and `quantumsim==0.2.0`;
however, as we expand the QEC implementations and types of noise that we are simulating, it may be useful to not reinvent the wheel and use one of the many performant codes for physical qubit simulation out there.

With this goal in mind, we have abstracted the details of the physical simulation away into three types of backends: circuits, models, and states.
These are designed to be interchangeable (when possible, i.e. describing equivalent mathematical objects) such that new backends can be added and new functionality can be enabled without need to change the descriptions of the `QECCode` implementations or `QuantumProgram` realiziations.

(circuit-backends)=
## Circuit Backends

In this context, a circuit is essentially a list of layers that are lists of gate names and target qubits.
Qubit labels are assumed to be either a `str` or `int`.

The circuit backend interface is enforced by the abstract `BasePhysicalCircuit` class, which has the following functions defined:

- Property getters for:
    - The underlying circuit
    - Circuit depth
    - Qubit labels
- Append circuit (in-place and copy)
- Delete qubits (in-place and copy)
- Insert circuit (in-place and copy)
- Map qubit labels (in-place and copy)
- Merge, i.e. layer combination (in-place and copy)
- Set qubit labels (in-place and copy)

The packages currently available as circuit backends:

- `pygsti` via `PyGSTiPhysicalCircuit`

See the [API Reference](/_autosummary/loqs.backends.circuit) for more in-depth documentation of circuit backends.

(model-backends)=
## Model Backends

In this context, a noise model is a mapping from a circuit label (i.e. gate name and target qubits) into some operator representation that can be applied to a state.
We need some way to enforce that a model's output can act on a state.
The `GateRep` enum defines what types of representations can be used for gates, and the allowed options can be checked in the [API Reference](/_autosummary/loqs.backends.model.basemodel.GateRep).
Similarly, the `InstrumentRep` enum defines what types of representations can be used for quantum instruments (often used to describe midcircuit measurements).
The allowed options can also be checked in the [API Reference](/_autosummary/loqs.backends.model.basemodel.InstrumentRep).

The model backend interface is enforced by the abstract `BaseNoiseModel` class, which has the following functions defined:

- Property getters for:
    - Allowed input circuit types
    - Output representation types
- Representation getter that converts a circuit into a list of gate/instrument representations (as a tuple of rep, target qubits, and rep type)

The packages currently available as noise models:

- `pygsti.models.ExplicitOpModel` and `pygsti.models.ImplicitOpModel` via `PyGSTiNoiseModel`

See the [API Reference](/_autosummary/loqs.backends.model) for more in-depth documentation of model backends.

(state-backends)=
## State Backends

In this context, a quantum state represents the state of a group of physical qubits.

The state backend interface is enforced by the abstract `BaseQuantumState` class, which has the following functions defined:

- Property getters for:
    - The underlying state object
    - Allowed input gate/instrument representations
- Copy (i.e. deepcopy)
- Representation application that takes the list of gate/instrument representation and propagates the state forward in time (in-place and copy)

The packages currently available as quantum states:

- `quantumsim.sparsedm` (0.2.0 only) via `QSimQuantumState`

See the [API Reference](/_autosummary/loqs.backends.state) for more in-depth documentation of state backends.

## What's next?

Congratulations!
We have covered all the core objects in `LoQS`.

Next, we will start putting all these pieces together through a series of tutorials.
