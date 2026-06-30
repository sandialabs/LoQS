---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Adding (Global) Model Noise

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sandialabs/LoQS/docs-updates?filepath=docs/notebooks/shortcourse-generalnoise.ipynb)

Ok, we ran some programs but they are not very interesting. Let's use some more interesting noise models!

## Stochastic Errors
Now, let's add some stochastic noise to see how things change up! There are several ways to do this, but they all basically fall down into pyGSTi land since we are tweaking the physical noise model and are using the pyGSTi model backend.

We follow a very similar pattern from the previous notebook for setting up our program:

```{code-cell} ipython3
from loqs.codepacks import codepack_7_1_3_quantinuum2021 as codepack_Steane
from loqs.core import Frame, History, InstructionStack, PatchDict, QuantumProgram
from loqs.backends import NumpyStatevectorQuantumState, PyGSTiNoiseModel, QSimQuantumState

# Let's define qubits for a single Quantinuum-style Steane patch: 7 data qubits, 3 aux qubits
qubits = ["A0", "A1", "A2"] + [f"D{i}" for i in range(7)]

steane_code = codepack_Steane.create_qec_code()

init_state = NumpyStatevectorQuantumState(10, qubit_labels=qubits)
init_patches = PatchDict({"L0": steane_code.create_patch(qubits=qubits)})
init_frame = Frame({"state": init_state, "patches": init_patches})
init_history = History(init_frame)

stack = InstructionStack([
    ("FT Zero Prep", "L0"),
    ("Adaptive QEC", "L0"),
    ("FT Logical Z Measure", "L0")
])
```

However, instead of calling the codepack's ideal model creation function, let's pattern match off that function but add some noise like we do in pyGSTi:

```{code-cell} ipython3
# PyGSTi models expect qubit labels to start with Q, so we define these here
# LoQS will take care of aliasing between the two sets of qubit labels
model_qubits = [f"Q{i}" for i in range(len(qubits))]

# These are (most) of the gates needed to run the codepack's circuit
gate_names = [
    "Gxpi",
    "Gypi",
    "Gzpi",
    "Gzpi2",
    "Gzmpi2",
    "Gh",
    "Gcnot",
    "Gi",
]

import pygsti

pspec = pygsti.processors.QubitProcessorSpec(
    len(model_qubits),
    gate_names=gate_names,
    qubit_labels=model_qubits,
    availability={k: "all-permutations" for k in gate_names},
)

stochastic_model_pygsti = pygsti.models.create_crosstalk_free_model(
    pspec,
    depolarization_strengths={ # Some dummy depolarizations
        "Gxpi": 0.001,
        "Gypi": 0.003,
        "Gh": 0.002,
        "Gcnot": 0.01
    },
    depolarization_parameterization="lindblad",
)

# Wrap it in the LoQS backend class
stochastic_model = PyGSTiNoiseModel(stochastic_model_pygsti, qubits)
```

And now we run the program as before, but use our new model:

```{code-cell} ipython3
sto_program = QuantumProgram(
    instruction_stack=stack,
    initial_history=init_history,
    default_noise_model=stochastic_model,
    name="Zero state preservation"
)

sto_results = sto_program.run(num_shots=100)
```

```{code-cell} ipython3
sto_results.collect_shot_data("logical_measurement", -1, return_counter=True)
```

Two things to note:

First, we no longer get a perfect result, as expected.

Second, you may have noticed that this program took longer to run. This is because we have to do Kraus unraveling for the stochastic noise, so we are pausing forward simulation to do RNG frequently.

The more operations with stochastic noise you have, the longer the simulation will take (assuming the gates are commonly used in the circuits).
Additionally, any non-unital Kraus operators will further slow the simulation down since it is state-dependent and must be computed on the fly.

+++

## Exercise 1

#### Play around with the depolarizing strengths and see what happens to the outcomes. Recommended to do some sweeps and plot the survival probability vs noise strength.

Some things to consider: What happens if you scale everything up and down? What about changing the fraction of 1Q to 2Q noise? What happens if you change from depolarizing noise to general (or a specific) Pauli stochastic noise instead?

Note: I'm not expecting anything particularly interesting here, but just some practice.

+++

## Coherent Noise

Now let's try some coherent noise! We can use the same pspec as above, just change our create_crosstalk_free_model call:

```{code-cell} ipython3
coherent_model_pygsti = pygsti.models.create_crosstalk_free_model(
    pspec,
    lindblad_error_coeffs={ # Some overrotations
        "Gxpi": {("H", "X"): 0.01},
        "Gypi": {("H", "Y"): 0.03},
        "Gcnot": {("H", "ZX"): 0.05, ("H", "ZI"): 0.05, ("H", "IX"): 0.05}
    },
)

# Wrap it in the LoQS backend class
coherent_model = PyGSTiNoiseModel(coherent_model_pygsti, qubits)
```

```{code-cell} ipython3
# Let's copy and run
co_program = QuantumProgram.from_quantum_program(sto_program, default_noise_model=coherent_model)
co_results = co_program.run(100)
```

```{code-cell} ipython3
# And check our results
co_results.collect_shot_data("logical_measurement", -1, return_counter=True)
```

## Exercise 2

#### Create a model that has both stochastic and coherent noise and do some parameter sweeps. Anything interesting happen as you switch off noise between stochastic and coherent?

Note: I'm not expecting anything crazy to pop out here either, especially at low shot counts. But it's worth the exercise regardless.

+++
