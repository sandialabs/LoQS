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
  [EdesignRunner](api:EdesignRunner) (one `QuantumProgram` per edesign circuit,
  configured via its own `parallel_strategy` field),
  [FaultInjectionRunner](api:FaultInjectionRunner)
  (one per error-injected variant), and
  [NoiseSweepRunner.run](api:NoiseSweepRunner.run) (one per sweep point) --
  accepts a [ParallelStrategy](api:ParallelStrategy) instead of a single
  executor. The batch is split into chunks, and each chunk (a sub-list of
  programs) is built/run/collected as a unit by one worker.

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

[ParallelStrategy.describe](api:ParallelStrategy.describe) prints a quick,
per-axis text summary of a configuration (backend, chunk/worker counts) before
running anything. [ParallelStrategy.plot](api:ParallelStrategy.plot) draws
the same configuration as a worker/chunk/shot diagram instead. The examples
below show both before each run -- see their API reference entries (linked
above) for the exact fields/diagram elements each one produces.

See the [ParallelStrategy](api:ParallelStrategy) API reference for the full
field-by-field details and validation rules.

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

programs = [program] * 6
strategy = ParallelStrategy(
    program_executor=loky.get_reusable_executor(max_workers=2),
    n_program_chunks=2,
)
print(strategy.describe(programs, num_shots=1))
strategy.plot(programs)

failed = fttools.FaultInjectionRunner(
    programs,
    collect_shot_data_args=[("counter", -1)],
    expected_outcomes=[1],
    num_shots=1,
    parallel_strategy=strategy,
).run()
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

failed = fttools.FaultInjectionRunner(
    programs,
    collect_shot_data_args=[("counter", -1)],
    expected_outcomes=[1],
    num_shots=20,
    parallel_strategy=strategy,
).run()
len(failed)
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

if sys.platform == "win32":
    print("submitit does not support Windows -- skipping dispatch.")
else:
    failed = fttools.FaultInjectionRunner(
        programs,
        collect_shot_data_args=[("counter", -1)],
        expected_outcomes=[1],
        num_shots=1,
        parallel_strategy=strategy,
    ).run()
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
    failed = fttools.FaultInjectionRunner(
        programs,
        collect_shot_data_args=[("counter", -1)],
        expected_outcomes=[1],
        num_shots=20,
        parallel_strategy=strategy,
    ).run()
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

    failed = fttools.FaultInjectionRunner(
        programs,
        collect_shot_data_args=[("counter", -1)],
        expected_outcomes=[1],
        num_shots=1,
        parallel_strategy=strategy,
    ).run()
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

    failed = fttools.FaultInjectionRunner(
        programs,
        collect_shot_data_args=[("counter", -1)],
        expected_outcomes=[1],
        num_shots=20,
        parallel_strategy=strategy,
    ).run()
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

## Checkpointing under parallelism

The [Basic Workflow](workflow.md) tutorial introduces
[QuantumProgram.run](api:QuantumProgram.run)'s `checkpoint_batch_size`/
`checkpoint_dir` for the serial case. With a `shot_executor`,
`checkpoint_batch_size` also sets how many shots are dispatched to one
worker per call: each worker computes its whole batch, checkpoints all of
it to its own private file (keyed by that worker's own hostname/PID, so
concurrent workers never open the same file), and only then returns. Once
every dispatched batch is confirmed done, `run()` runs one race-free,
driver-side pass that streams every worker's file into the same canonical
`results.h5` a serial run would have written directly -- callers never
need to touch the per-worker files themselves.

```{code-cell} ipython3
import tempfile

from loqs.core import ProgramResults

with tempfile.TemporaryDirectory() as checkpoint_dir:
    results = program.run(
        num_shots=20,
        shot_executor=executor,
        checkpoint=True,
        checkpoint_batch_size=5,
        checkpoint_dir=checkpoint_dir,
        verbose=False,
    )
    print(f"in-memory shots right after run(): {len(results.shot_histories)}")

    # The consolidated checkpoint is an ordinary, single-writer checkpoint,
    # readable the same way regardless of how many workers wrote it.
    recovered = ProgramResults()
    recovered.load_checkpoint(checkpoint_dir=checkpoint_dir)
    print(f"shots recovered from disk: {len(recovered.shot_histories)}")
```

Since each worker computes its whole batch before writing anything, a
crash loses at most one worker's own in-flight batch (up to
`checkpoint_batch_size` shots from that one worker, not the whole run) --
set it to `1` to checkpoint every shot as soon as it's computed, trading
batching efficiency for the finest possible durability.

The `0` above isn't a bug: `lazy_loading` (default `True`) evicts a
shot from the returned [ProgramResults](api:ProgramResults)'s own in-memory
`shot_histories` as soon as it's durably checkpointed, so `run()`'s return
value holds only the yet-unwritten tail once checkpointing is active --
pass `lazy_loading=False` to keep every shot in memory regardless,
or reload the full set from disk via `load_checkpoint()` as above.

### Resuming after a crash

A checkpointing-enabled `run()` also supports genuine resume, regardless of whether the original call used a `shot_executor`: calling `run()` again with `checkpoint=True, resume=True` against the same `checkpoint_dir` only computes whatever wasn't already durably checkpointed, rather than starting over from shot 0. `run()` never returns leaving stray `worker_*_checkpoint.h5` files behind either -- a race-free consolidation pass into `results.h5` runs whenever any exist, even if the resuming call itself dispatches serially, so a parallel run that crashed mid-batch can always be finished off with a plain serial call.

```{code-cell} ipython3
with tempfile.TemporaryDirectory() as resume_checkpoint_dir:
    partial = program.run(
        num_shots=10,
        shot_executor=executor,
        checkpoint=True,
        checkpoint_batch_size=5,
        checkpoint_dir=resume_checkpoint_dir,
        lazy_loading=False,
        verbose=False,
    )
    print(f"shots after first call: {len(partial.shot_histories)}")

    # Finishing off with more shots than originally planned: num_shots
    # differs from the original call, so force_resume=True is needed to
    # bypass the mismatch check -- the 10 already-checkpointed shots are
    # kept as-is, and only the remaining 10 are actually computed.
    finished = program.run(
        num_shots=20,
        shot_executor=executor,
        checkpoint=True,
        resume=True,
        force_resume=True,
        checkpoint_batch_size=5,
        checkpoint_dir=resume_checkpoint_dir,
        lazy_loading=False,
        verbose=False,
    )
    print(f"shots after resuming call: {len(finished.shot_histories)}")
```

If a resuming call's own `num_shots`/`max_frame_limit`/RNG seed happen to exactly match the original call, no mismatch is detected and `force_resume` isn't needed at all.

This mechanism is scoped to `QuantumProgram`/`ProgramResults`'s own
shot-level checkpointing. The program-level call sites ([EdesignRunner](api:EdesignRunner),
[FaultInjectionRunner](api:FaultInjectionRunner), and
[NoiseSweepRunner.run](api:NoiseSweepRunner.run)) use a separate, unified item-level
checkpoint mechanism with per-worker HDF5 checkpoint files, crash recovery, and per-item completion tracking.

## Performance profiling

`profile_strategies` measures real wall-clock time -- and, whenever `psutil`
is installed, peak memory and CPU utilization -- for a caller-chosen set of
[ParallelStrategy](api:ParallelStrategy) configurations against a
caller-chosen working example. Deciding between candidate configurations (how
many outer/inner workers, which backend, how to split workers between the two
axes) is a real, workload-dependent question -- this measures it directly on
real hardware rather than guessing.

The trivial toy program used elsewhere in this tutorial does too little real
work per shot to show a meaningful difference between configurations. The
cells below build a small, real workload instead: single-Pauli error
injection into the [[5,1,3]] code's `FT Minus Prep` stage, the same one used
in the [Testing Fault Tolerance](fttests.md) tutorial.

```{code-cell} ipython3
from loqs.backends import QSimQuantumState
from loqs.codepacks import codepack_5_1_3_quantinuum2022 as codepack_5_1_3

code_5q = codepack_5_1_3.create_qec_code()
ft_qubits = ["A0", "A1"] + [f"D{i + 2}" for i in range(5)]
ft_ideal_model = codepack_5_1_3.create_ideal_model(ft_qubits)

stack_ft = [
    {"instruction": "Init State", "state": len(ft_qubits), "qubit_labels": ft_qubits},
    {"instruction": "Init Patch 5Q", "new_patch_label": "L0", "qubits": ft_qubits},
    ("FT Minus Prep", "L0"),
    ("Flagged QEC", "L0"),
    ("FT Logical X Measure", "L0"),
]
ft_program = QuantumProgram(
    stack_ft,
    default_noise_model=ft_ideal_model,
    state_type=QSimQuantumState,
    patch_types={"5Q": code_5q},
    name="FT Prep -, Flagged QEC, FT measure X",
)

# 8 of the 171 possible single-Pauli-error variants of "FT Minus Prep",
# enough to give a program-level pool real, evenly-splittable work.
ft_programs = fttools.build_discrete_error_injection_programs(
    base_program=ft_program,
    instruction_to_analyze=code_5q.instructions["Non-FT Minus Prep + Checks"],
    stack_idx_to_modify=2,
    error_circuit_labels=["Gxpi", "Gypi", "Gzpi"],
)[:8]
len(ft_programs)
```

### Comparing worker shapes

The same total workers can be split between the two axes in different ways --
all program workers, all shot workers, or some balance of both -- dispatching
the exact same computation with a different shape. `describe()`/`plot()`
sanity-check a shape before running anything, without needing to actually
dispatch it:

`ParallelStrategy(program_executor=..., shot_executor=...)` normally takes a
live executor directly for either axis (see above) -- but
`loky.get_reusable_executor()` is a **process-wide singleton**, not one pool
per `max_workers` value: calling it again with a *different* `max_workers`
resizes the same underlying pool in place, silently changing what any
earlier reference to it now points to. Building executors of genuinely
different sizes side by side (as the shapes below do) hits this directly;
passing an explicit [ExecutorSpec](api:ExecutorSpec) instead sidesteps it
entirely, since it doesn't touch a live pool at all until it's actually
called (inside a worker subprocess, where no such collision can occur).

```{code-cell} ipython3
from loqs.tools.paralleltools import ExecutorSpec

for program_workers, shot_workers in [(4, 1), (2, 2), (1, 4)]:
    shape_strategy = ParallelStrategy(
        program_executor=ExecutorSpec("loky", {"max_workers": program_workers}),
        n_program_chunks=program_workers,
        shot_executor=ExecutorSpec("loky", {"max_workers": shot_workers}),
    )
    print(shape_strategy.describe(ft_programs, num_shots=20))
    shape_strategy.plot(ft_programs)
```

Whether splitting workers onto the shot axis actually helps is workload
dependent, not automatic: [QuantumProgram.run](api:QuantumProgram.run)
dispatches one `.submit()` call per shot, unbatched, so each shot pays a real
serialization/IPC cost to cross into a worker process. For a workload whose
per-shot compute is itself fast (like this one), that per-shot cost can
outweigh the benefit -- the program axis, which amortizes one dispatch over
an entire chunk of programs, is usually the more reliable one to reach for
first. The measurement below sticks to program-axis splits for this reason.

```{code-cell} ipython3
import os

from loqs.tools.paralleltools import (
    format_profile_table,
    plot_profile_results,
    profile_strategies,
)


def profile_work(strategy):
    return fttools.FaultInjectionRunner(
        ft_programs,
        collect_shot_data_args=[("logical_measurement", -1)],
        expected_outcomes=[1],
        num_shots=20,
        parallel_strategy=strategy,
    ).run()


strategies = {
    "serial": ParallelStrategy(),
    "program-2x": ParallelStrategy(
        program_executor=ExecutorSpec("loky", {"max_workers": 2}),
        n_program_chunks=2,
    ),
    "program-4x": ParallelStrategy(
        program_executor=ExecutorSpec("loky", {"max_workers": 4}),
        n_program_chunks=4,
    ),
}

# Timing-sensitive on a small/shared CI runner (which sets CI=true) --
# skip the live measurement there and just show how it's called.
if os.environ.get("CI"):
    print("Skipping the live profiling sweep in CI -- run interactively instead.")
    results = None
else:
    results = profile_strategies(
        profile_work, strategies, repeats=2, sample_interval=0.05, warmup=True
    )
    print(format_profile_table(results))
```

`warmup=True` runs one throwaway pass through each strategy's dispatch path
before the timed `repeats` loop, so real worker-process startup/import cost
doesn't land on whichever repeat happens to run first. Since `"serial"` is
fully serial, every other row also reports a `speedup` relative to it. On
real hardware, splitting this same workload across worker counts like this
typically shows real, if sub-linear, speedup (e.g. roughly 1.6x-1.9x at 2
workers, 2.1x-2.8x at 4 workers, relative to the serial baseline) -- workers
beyond the first pay real, if smaller, marginal overhead of their own, so
returns diminish rather than scaling linearly.

Every result's raw per-chunk data (peak memory, mean CPU%, sample count) is
kept, not just the summary numbers above -- `plot_profile_results` draws a
wall-time bar chart plus box plots of that per-chunk resource data directly.

```{code-cell} ipython3
if results is not None:
    plot_profile_results(results);
```

### Profiling a `submitit` sweep on a real SLURM allocation

Comparing several `submitit`-backed configurations the way `profile_strategies`
does above would normally mean each candidate strategy's own dispatch
independently submits (and queues for) its own `sbatch` job -- exactly the
"stuck behind someone else's job" risk a real profiling sweep wants to avoid.
`reuse_slurm_allocation=True` sidesteps this: it installs a small wrapper
script that intercepts every `sbatch` call `submitit` makes during the sweep,
running it directly against whatever real SLURM allocation the current
process is already inside (obtained separately first, e.g. via `salloc` or an
enclosing `sbatch` job) instead of requesting a new one each time. Requires
already running inside a real allocation (`SLURM_JOB_ID` set); not runnable in
this Binder/CI environment, so not executed here.

```python
# Obtain one allocation sized for the largest candidate strategy first, e.g.
#   salloc --nodes=4 --ntasks-per-node=1 --cpus-per-task=8
# then, from inside that shell:
results = profile_strategies(
    profile_work, strategies, reuse_slurm_allocation=True
)
```
