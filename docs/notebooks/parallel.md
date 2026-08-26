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

# Parallel Execution

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sandialabs/LoQS/{{ binder_branch }}?filepath=docs/notebooks/parallel.ipynb)

LoQS can parallelize shots within a single [](api:QuantumProgram) across
multiple worker processes. This requires the `parallel` optional
dependency group (`pip install -e ".[parallel]"`), which provides
[`loky`](https://loky.readthedocs.io/).

## Step 1: Build a program

```{code-cell} ipython3
from loqs.codepacks import codepack_trivial_counter as trivial_codepack
from loqs.core import QuantumProgram

trivial_code = trivial_codepack.create_qec_code()
ideal_model = trivial_codepack.create_ideal_model(["Q0"])

stack = [
    {"instruction": "Init Patch Trivial", "new_patch_label": "L0", "qubits": ["Q0"]},
    {"instruction": "Init Counter", "patch_label": "L0", "initial_value": 0},
    {"instruction": "Increment", "patch_label": "L0", "increment_by": 1},
]
program = QuantumProgram(
    stack,
    default_noise_model=ideal_model,
    patch_types={"Trivial": trivial_code},
    default_base_seed=0,
)
```

## Step 2: Run shots across a `loky` executor

Pass a `loky.get_reusable_executor()` instance as `shot_executor` to
[](api:QuantumProgram.run). Any object exposing a
`concurrent.futures`-style `.submit()` method works the same way,
including an `mpi4py.futures.MPIPoolExecutor` for multi-node runs. Each
worker pins its own BLAS/OpenMP thread pools to one thread before
running, so oversubscription is avoided regardless of how many workers
`shot_executor` spawns.

```{code-cell} ipython3
import loky

executor = loky.get_reusable_executor(max_workers=4)
results = program.run(num_shots=20, shot_executor=executor, verbose=False)
len(results.shot_histories)
```

Omitting `shot_executor` (the default) runs shots serially in the calling
process, exactly as before.
