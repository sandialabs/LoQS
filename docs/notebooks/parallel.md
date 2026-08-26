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

LoQS can parallelize both shots within a single
[QuantumProgram](api:QuantumProgram) and whole programs across a batch (e.g.
many error-injected variants, or many circuits in a GST experiment design).
This requires the `parallel` optional dependency group (`pip install -e
".[parallel]"`), which provides [`loky`](https://loky.readthedocs.io/) for
single-node parallelism and
[`submitit`](https://github.com/facebookincubator/submitit) for SLURM-array
fan-out; the separate `mpi` extra (`pip install -e ".[mpi]"`) adds `mpi4py` for
multi-node parallelism via an existing MPI installation. The diagrams below
also need `matplotlib`, via the `visualization` extra (`pip install -e
".[visualization]"`).

## Two axes of parallelism

There are two independent things that can be parallelized, and either,
both, or neither can be active at once:

- **Shots**, within one [QuantumProgram](api:QuantumProgram). Controlled directly
  by [QuantumProgram.run](api:QuantumProgram.run)'s `shot_executor` parameter --
  shots of that one program are dispatched across the executor's workers instead
  of running one at a time in the calling process.
- **Programs**, across a whole batch. Every program-level call site --
  [simulate_dataset_for_edesign](api:simulate_dataset_for_edesign) (one
  `QuantumProgram` per edesign circuit),
  [run_discrete_error_injected_programs](api:run_discrete_error_injected_programs)
  (one per error-injected variant), and
  [NoiseSweepRunner.run](api:NoiseSweepRunner.run) (one per sweep point) --
  accepts a `parallel` argument (a [ParallelStrategy](api:ParallelStrategy))
  instead of a single executor. The batch is split into chunks, and each chunk (a
  sub-list of programs) is built/run/collected as a unit by one worker.

These compose: a [ParallelStrategy](api:ParallelStrategy) can dispatch chunks
of programs to *outer* workers (e.g. one per node), each of which then runs its
own chunk's shots through an *inner* shot-level executor (e.g. one per core) --
programs distributed across nodes, shots distributed across cores within a
node, exactly the hybrid shape a real HPC allocation usually wants. A single
flat `shot_executor` (no chunking) and a single flat `program_executor` (no
shot nesting) are just the two degenerate cases of this same design.

Three concrete backends fill these roles, but not interchangeably -- each
satisfies a different structural protocol (see
[SubmitExecutor](api:SubmitExecutor)/[MapArrayExecutor](api:MapArrayExecutor)),
which determines where it can be plugged in:

| Backend | Protocol | Can be `shot_executor`? | Can be `program_executor`? | Multi-node? |
| --- | --- | --- | --- | --- |
| `loky.get_reusable_executor()` | `SubmitExecutor` | Yes | Yes | No (single node) |
| `mpi4py.futures.MPIPoolExecutor` | `SubmitExecutor` | Yes | Yes | Yes |
| `submitit.AutoExecutor` | `MapArrayExecutor` (and `SubmitExecutor`) | No | Yes | Yes (SLURM) |

`submitit`'s whole value is its bulk `map_array` submission (one `sbatch` call
covering an entire batch of chunks);
[QuantumProgram.run](api:QuantumProgram.run)'s `shot_executor` only ever
dispatches one `.submit()` call per shot, so a `submitit.Executor` is never
used as `shot_executor` in practice even though it technically satisfies
`SubmitExecutor` too.

## `ParallelStrategy` in detail

[ParallelStrategy](api:ParallelStrategy) (`loqs.tools.paralleltools`) is a
small `dataclass`, reused identically by every program-level call site:

- `program_executor`: dispatches chunks, auto-selecting one `.submit()` call
  per chunk vs. a bulk `.map_array()` call depending on which protocol it
  satisfies.
- `n_program_chunks`: how many round-robin chunks to split the batch into.
- `shot_executor`: nested shot-level parallelism. A live executor can't
  normally cross the process boundary each dispatched chunk crosses once
  `program_executor` is also set, but a recognized backend (currently
  `loky`) is auto-converted into a picklable factory built from its own
  parameters -- no factory function needs to be written by hand. An
  unrecognized backend still requires an explicit zero-argument factory
  callable instead.

`ParallelStrategy.describe(items, num_shots)` prints a quick summary of both
axes -- a short backend tag per axis (e.g. `loky(max_workers=2)`), plus
whichever counts can actually be computed from what's known. The program
axis can report a real chunk count, programs per chunk, and chunks per
worker (`n_program_chunks` is an explicit, independently-chosen setting);
the shot axis has no analogous chunking setting -- shots are dispatched one
at a time -- so it only reports the total and the resulting average per
worker.

[ParallelStrategy.plot](api:ParallelStrategy.plot) draws the same
information as a diagram: the node is one box, program-axis workers
(`PW0`, `PW1`, ...) are lanes within it separated by dashed dividers,
since they really do run concurrently; but a worker's own chunk(s), and
the programs within one chunk, are *not* concurrent with each other (one
worker process runs one chunk at a time, and one chunk's programs run one
at a time inside it), so those are drawn instead as a left-to-right
sequence connected by curved arrows. Each program's own box directly
contains a stack of shot boxes (one per shot worker, or a single "Serial"
one when `shot_executor` is `None`); dashed `SW0`/`SW1`/... lines span the
whole chunk's width at the gaps between shot-box rows, since it really is
the same resolved shot executor, reused sequentially by every program in
that chunk. Workers that would otherwise be exact duplicates of each
other (the common case under round-robin dispatch) collapse to a single
`"PW1 = PW0"`-style label rather than being redrawn, and a worker left
with no chunks at all (more workers requested than there are chunks to
hand out) is drawn hatched and grayed out rather than silently omitted,
since that's itself useful information. The examples below show both
`describe()` and the diagram before each run. `plot()` always builds its
own new figure (it doesn't accept a caller-supplied `ax` to draw into),
since embedding one in a caller-managed subplot grid reads as more
confusing than helpful once a real multi-node layout is in the picture.
See the [ParallelStrategy](api:ParallelStrategy) API reference for the
full field-by-field details and validation rules.

## Build a program

```{code-cell} ipython3
from loqs.codepacks import codepack_trivial_counter as trivial_codepack
from loqs.core import QuantumProgram

trivial_code = trivial_codepack.create_qec_code()
ideal_model = trivial_codepack.create_ideal_model(["Q0"])

# A trivial program: initialize a patch and a counter, then increment the
# counter once. Every shot should report a final counter value of 1.
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

## Single-node parallelism (`loky`)

`loky` is the right default choice for a single machine: it needs no
external services, and it falls back to `cloudpickle` whenever plain
`pickle` fails, which real LoQS `Instruction` objects need (many are
built as closures, which plain `pickle` cannot serialize at all).

### Shot-level

Pass a `loky.get_reusable_executor()` instance as `shot_executor` to
[QuantumProgram.run](api:QuantumProgram.run) directly -- no `ParallelStrategy`
needed here, since there's only one axis in play.

```{code-cell} ipython3
import loky

# 4 worker processes share this one program's 20 shots.
executor = loky.get_reusable_executor(max_workers=4)
results = program.run(num_shots=20, shot_executor=executor, verbose=False)
len(results.shot_histories)
```

Omitting `shot_executor` (the default) runs shots serially in the calling
process, exactly as before.

### Program-level

Build several programs (here, just repeated copies of the same one for
illustration) and pass a [ParallelStrategy](api:ParallelStrategy) instead of a
single executor. The batch is split into `n_program_chunks` round-robin chunks
(so cost drift across the batch, e.g. later GST circuits typically being
deeper, doesn't concentrate onto a few workers), and each chunk is processed as
a unit by one worker.

```{code-cell} ipython3
from loqs.tools import fttools
from loqs.tools.paralleltools import ParallelStrategy

# 6 copies of the same program, split into 2 chunks of 3 -- each chunk is
# tested as a unit by one of the 4 loky workers.
programs = [program] * 6
strategy = ParallelStrategy(
    program_executor=loky.get_reusable_executor(max_workers=4),
    n_program_chunks=2,
)
print(strategy.describe(programs, num_shots=1))
strategy.plot(programs)

failed = fttools.run_discrete_error_injected_programs(
    programs,
    collect_shot_data_args=[("counter", -1)],
    expected_outcomes=[1],
    num_shots=1,
    parallel=strategy,
)
len(failed)
```

### Hybrid (both axes)

`ParallelStrategy.shot_executor` nests shot-level parallelism inside
program-level parallelism. A plain live `loky` executor works directly here,
just like the shot-level example above -- `ParallelStrategy` auto-converts it
into a picklable factory built from its own parameters (see `ParallelStrategy`
in detail above), so no factory function needs to be written by hand.

```{code-cell} ipython3
strategy = ParallelStrategy(
    program_executor=loky.get_reusable_executor(max_workers=2),  # outer: programs
    n_program_chunks=2,
    shot_executor=loky.get_reusable_executor(max_workers=2),  # inner: shots
)
print(strategy.describe(programs, num_shots=20))
strategy.plot(programs)

failed = fttools.run_discrete_error_injected_programs(
    programs,
    collect_shot_data_args=[("counter", -1)],
    expected_outcomes=[1],
    num_shots=20,
    parallel=strategy,
)
len(failed)
```

#### Comparing worker shapes

The same 4 total workers can be split between the two axes in different
ways -- 4 program workers x 1 shot worker, 2x2 (as above), or 1x4 -- all
dispatching the exact same computation, just with a different shape. The
diagrams below compare all three; only the 2x2 split is actually run below,
since all three would give identical results.

`ParallelStrategy(program_executor=..., shot_executor=...)` normally takes a
live executor directly for `shot_executor` (see above) -- but
`loky.get_reusable_executor()` is a **process-wide singleton**, not one pool
per `max_workers` value: calling it again with a *different* `max_workers`
resizes the same underlying pool in place, silently changing what any
earlier reference to it now points to. Building program and shot executors
of genuinely different sizes from two separate `get_reusable_executor()`
calls in the same process (as the three shapes below need) hits this
directly; passing an explicit [ExecutorSpec](api:ExecutorSpec) for the shot
axis instead sidesteps it entirely, since it doesn't touch the live pool at
all until it's actually called (inside a worker subprocess, where no such
collision can occur). This is unlikely to come up in ordinary use --
building several *different-sized* strategies side by side like this is
mainly a showcase/comparison need, not something a real workflow (which
just builds and uses one strategy per run) normally does.

Each shape gets its own figure below -- `plot()` always builds a new one
rather than accepting a caller-supplied `ax` to draw into (see above).

```{code-cell} ipython3
from loqs.tools.paralleltools import ExecutorSpec

for program_workers, shot_workers in [(4, 1), (2, 2), (1, 4)]:
    shape_strategy = ParallelStrategy(
        program_executor=loky.get_reusable_executor(
            max_workers=program_workers
        ),
        n_program_chunks=program_workers,
        shot_executor=ExecutorSpec("loky", {"max_workers": shot_workers}),
    )
    print(shape_strategy.describe(programs, num_shots=20))
    shape_strategy.plot(programs)
```

## Multi-node and hybrid parallelism

The examples above all ran on a single machine. The sections below combine a
multi-node backend (`submitit` or `mpi4py`) with the single-node `loky` axis
from above, for the full node/core hybrid shape a real HPC allocation usually
wants.

### Multi-node programs via `submitit` (for SLURM/HPC)

`submitit` targets a scheduler-based multi-node shape:
`submitit.AutoExecutor`'s `map_array` submits a single `sbatch` job array
covering an entire batch of chunks at once (one `sbatch` call, not one per
chunk), and each array task runs as its own independent SLURM allocation.
`submitit.Executor` only ever plugs in as `program_executor` (see the backend
table above) -- never as `shot_executor` -- since its value is specifically
that bulk `map_array` submission.

`cluster="local"` below runs the exact same code path through ordinary
subprocesses rather than real `sbatch` calls, so it works without a real
scheduler; swap in real `slurm_*` parameters (via
`executor.update_parameters(...)`) to target an actual cluster.

`submitit` itself only supports Linux/macOS: it unconditionally registers a
`SIGCONT` handler for every job it runs, a POSIX-only signal that doesn't
exist in Windows's `signal` module at all (a real, unconditional upstream
limitation, unsurprising given `submitit` targets SLURM, a Linux-only
scheduler). The cells below still build and `describe()`/`plot()` a
`submitit`-backed [ParallelStrategy](api:ParallelStrategy) on every platform,
but skip actually dispatching it on Windows.

```{code-cell} ipython3
import sys

import submitit

# cluster="local" runs each chunk as an ordinary subprocess instead of a
# real sbatch job, so this works without an actual SLURM allocation.
strategy = ParallelStrategy(
    program_executor=submitit.AutoExecutor(
        folder="submitit-logs", cluster="local"
    ),
    n_program_chunks=2,
)
print(strategy.describe(programs, num_shots=1))
# submitit.Executor's real parallelism is scheduler-determined (unlike
# loky's fixed max_workers), so plot() can't introspect a worker count --
# program_workers is given explicitly here, matching one concurrent
# allocation per chunk.
strategy.plot(programs, program_workers=2)

if sys.platform == "win32":
    print("submitit does not support Windows -- skipping dispatch.")
else:
    failed = fttools.run_discrete_error_injected_programs(
        programs,
        collect_shot_data_args=[("counter", -1)],
        expected_outcomes=[1],
        num_shots=1,
        parallel=strategy,
    )
    print(f"{len(failed)} failed")
```

The real hybrid shape a cluster allocation usually wants combines `submitit`
with the single-node `loky` axis above: `submitit` fans each chunk out to its
own node-level SLURM allocation, and each of those chunks then runs its own
shots across a `loky` pool local to that allocation -- programs distributed
across nodes, shots distributed across cores within a node, via the same
auto-converted `shot_executor` mechanism used in the one-node hybrid above.

```{code-cell} ipython3
# Same idea as the one-node hybrid above, but the outer (program) axis is
# now a SLURM array via submitit instead of a local loky pool.
strategy = ParallelStrategy(
    program_executor=submitit.AutoExecutor(
        folder="submitit-logs", cluster="local"
    ),
    n_program_chunks=2,
    shot_executor=loky.get_reusable_executor(max_workers=2),
)
print(strategy.describe(programs, num_shots=20))
strategy.plot(programs, program_workers=2)

if sys.platform == "win32":
    print("submitit does not support Windows -- skipping dispatch.")
else:
    failed = fttools.run_discrete_error_injected_programs(
        programs,
        collect_shot_data_args=[("counter", -1)],
        expected_outcomes=[1],
        num_shots=20,
        parallel=strategy,
    )
    print(f"{len(failed)} failed")
```

On a real cluster, `submitit.Executor`'s own `slurm_*` parameters (e.g.
`cpus_per_task`) should be sized to match the inner `loky` pool's own
`max_workers`, so each array task actually gets the cores its own nested
executor tries to use.

### Multi-node parallelism via `MPI`

`mpi4py.futures.MPIPoolExecutor` is a real `concurrent.futures.Executor`
subclass, so it's a drop-in `SubmitExecutor` -- it can be used exactly where a
`loky` executor was used above, as either `shot_executor` or
`program_executor`, with no LoQS-side code needing to know which backend is
actually running. A shared pool of MPI ranks (potentially spanning many nodes)
pulls submitted tasks from a queue as they finish, giving genuine dynamic load
balancing rather than a fixed upfront assignment.

This requires the `mpi` extra (`pip install -e ".[mpi]"`) plus a working MPI
installation (e.g. OpenMPI/MPICH) already present in the environment -- `pip
install` alone does not provide one, and neither is available in this Binder/CI
environment, so none of the examples below are actually run here.

Shot-level, parallelizing across MPI ranks instead of `loky` workers:

```python
from mpi4py.futures import MPIPoolExecutor

# Same call as the loky shot-level example above, just backed by MPI
# ranks (potentially spanning many nodes) instead of local processes.
with MPIPoolExecutor() as executor:
    results = program.run(num_shots=20, shot_executor=executor)
```

Program-level, chunking the batch across MPI ranks the same way `loky`
did above:

```python
with MPIPoolExecutor() as executor:
    strategy = ParallelStrategy(program_executor=executor, n_program_chunks=3)
    print(strategy.describe(programs, num_shots=1))
    # program_workers is given explicitly since it depends on how many
    # ranks mpiexec -n N actually launched with, not something knowable
    # from the executor object alone ahead of time.
    strategy.plot(programs, program_workers=3)

    failed = fttools.run_discrete_error_injected_programs(
        programs,
        collect_shot_data_args=[("counter", -1)],
        expected_outcomes=[1],
        num_shots=1,
        parallel=strategy,
    )
```

`MPIPoolExecutor` is typically launched as one atomic multi-node allocation
(`mpiexec -n N`, one reservation covering every rank up front) -- a real
environment dependency beyond what `loky` needs, but normally already provided
via the module system on real HPC systems.

For completion, MPI can nest with `loky` the same way `submitit` did above:
`program_executor` chunks the batch across MPI ranks (potentially spanning many
nodes), and each rank's chunk then runs its own shots across a local `loky`
pool, auto-converted from a plain live executor the same way as above.

```python
with MPIPoolExecutor() as executor:
    # Outer axis: MPI ranks, potentially spanning many nodes. Inner axis:
    # a local loky pool per rank's own chunk of programs.
    strategy = ParallelStrategy(
        program_executor=executor,
        n_program_chunks=3,
        shot_executor=loky.get_reusable_executor(max_workers=2),
    )
    print(strategy.describe(programs, num_shots=20))
    strategy.plot(programs, program_workers=3)

    failed = fttools.run_discrete_error_injected_programs(
        programs,
        collect_shot_data_args=[("counter", -1)],
        expected_outcomes=[1],
        num_shots=20,
        parallel=strategy,
    )
```

## Thread-oversubscription safety

Every worker (shot-level or program-level, on any backend above) pins its
own BLAS/OpenMP thread pools to one thread via `threadpoolctl` before
running, regardless of how many workers are active at either axis. This
avoids oversubscription -- N worker processes each independently trying
to use every available core for BLAS operations, which would otherwise
make a parallel run slower than serial rather than faster, and matters
more, not less, once both axes are combined (as in the hybrid examples
above).
