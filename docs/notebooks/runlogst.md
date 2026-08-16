---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Running Logical GST (LoGST)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sandialabs/LoQS/{{ binder_branch }}?filepath=docs/notebooks/runlogst.ipynb)

## Step 1: Generate a GST design

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XZ as modelpack
```

```{code-cell} ipython3
gst_design = modelpack.create_gst_experiment_design(max_max_length=16, qubit_labels=["Q0"])
gst_model = modelpack.target_model(qubit_labels=["Q0"]) # 1 physical qubit model
```

## Step 2: Convert each `Circuit` into a `QuantumProgram`

```{code-cell} ipython3
from loqs.backends.circuit import PyGSTiPhysicalCircuit
from loqs.backends.state import QSimQuantumState
from loqs.codepacks import codepack_5_1_3_quantinuum2022 as codepack
from loqs.tools import pygstitools as pt
```

```{code-cell} ipython3
qubits = ["A0", "A1"] + [f"D{i+2}" for i in range(5)]

ideal_model = codepack.create_ideal_model(qubits) # 7 physical qubit model for simulating 1 logical qubit
code_5q = codepack.create_qec_code(circuit_backend=PyGSTiPhysicalCircuit)

program_kwargs = {
    "default_noise_model": ideal_model,
    "state_type": QSimQuantumState,
    "patch_types":  {"5Q": code_5q},
}
```

```{code-cell} ipython3
physical_to_logical = {
    "rho0": [
        {"instruction": "Init State", "state": len(qubits), "qubit_labels": qubits},
        {"instruction": "Init Patch 5Q", "new_patch_label": "L0", "qubits": qubits},
        ("Non-FT Minus Prep", "L0"), # in logical minus
        ('H', "L0"), # in logical 1
        ('X', "L0"), # in logical 0
    ],
    ("Gxpi2", "Q0"): [
        ("X", "L0") # Not technically correct, but just an example
    ],
    ("Gzpi2", "Q0"): [
        ("Z", "L0") # Not technically correct, but just an example
    ],
    "Mdefault": [
        ("Non-FT Logical Z Measure", "L0")
    ]
}
```

```{code-cell} ipython3
programs = pt.convert_edesign_to_programs(gst_design, gst_model, physical_to_logical, **program_kwargs)
```

```{code-cell} ipython3
len(gst_design.all_circuits_needing_data) == len(programs)
```

## Step 3: Run the programs

This takes the place of a `pygsti.data.simulate_data`

```{code-cell} ipython3
for program in programs:
    program.run(num_shots=10)
```

## Step 4: Convert this to a pyGSTi dataset

```{code-cell} ipython3
ds = pt.convert_run_programs_to_dataset(gst_design, programs)
```

## Step 5: Normal GST pipeline

## Step 6: Profit $$$
