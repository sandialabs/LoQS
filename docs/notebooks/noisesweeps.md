---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Continuous Noise-Strength Sweeps

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sandialabs/LoQS/{{ binder_branch }}?filepath=docs/notebooks/noisesweeps.ipynb)

The [Adding (Global) Model Noise](shortcourse-generalnoise.md) short course notebook poses an
exercise: "sweep the depolarizing strength and plot the survival probability." Doing this by hand
means writing your own loop over noise-strength values, building a fresh `QuantumProgram` for each
one, keeping track of RNG seeds so shots don't silently reuse randomness across sweep points, and
tallying up pass/fail counts into a failure rate.

`loqs.tools.noisesweeptools` does all of that bookkeeping for you. Unlike `loqs.tools.fttools`
(which exhaustively injects every single discrete Pauli fault to answer "is this circuit FT at
all"), `noisesweeptools` answers a different, complementary question: "how does the logical
failure rate scale as a *continuous* physical noise parameter grows?" This notebook walks through
the whole feature, using the same Steane `[[7,1,3]]` code and depolarizing-noise setup as the
short course.

```{code-cell} ipython3
from loqs.codepacks import codepack_7_1_3_quantinuum2021 as codepack_Steane
from loqs.core import Frame, History, InstructionStack, PatchDict
from loqs.backends import NumpyStatevectorQuantumState, PyGSTiNoiseModel
from loqs.tools.noisesweeptools import (
    NoiseSweepRunner,
    NoiseSweepResult,
    compare_noise_sweeps,
    plot_noise_sweep,
)
```

## Setup: a fixed instruction stack, a swept noise model

We build the same Steane patch and initial state as in the short course, and two instruction
stacks: one that prepares $\ket{0}_L$, runs a round of QEC, and measures in the $Z$ basis, and a
second that additionally applies a transversal $H$ to prepare/measure in the $X$ basis instead.
Both stacks are *fixed* -- they don't depend on the noise strength -- so we can build them once and
reuse them across every sweep point.

```{code-cell} ipython3
qubits = ["A0", "A1", "A2"] + [f"D{i}" for i in range(7)]
steane_code = codepack_Steane.create_qec_code()

init_state = NumpyStatevectorQuantumState(10, qubit_labels=qubits)
init_patches = PatchDict({"L0": steane_code.create_patch(qubits=qubits)})
init_frame = Frame({"state": init_state, "patches": init_patches})
init_history = History(init_frame)

stack_z = InstructionStack([
    ("FT Zero Prep", "L0"),
    ("Adaptive QEC", "L0"),
    ("FT Logical Z Measure", "L0"),
])

stack_x = InstructionStack([
    ("FT Zero Prep", "L0"),
    ("H", "L0"),
    ("Adaptive QEC", "L0"),
    ("FT Logical X Measure", "L0"),
])
```

The noise *model*, on the other hand, is exactly what we want to vary from point to point. We write
it as a plain function of one argument (the noise strength) that returns a `BaseNoiseModel`, mirroring
the depolarizing model construction from the short course.

```{note}
This function must be **fully self-contained**: `NoiseSweepRunner` only ever captures *this
function's own source code*, not any variables from the enclosing scope it might reference. If we
built `pspec`/`gate_names` outside the function and just referenced them from inside it, the
function would work fine right now, but would fail with a `NameError` after being written to disk
and read back in a later session. Constructing everything the function needs *inside* it (aside
from ordinary `import` statements) avoids this entirely.
```

```{code-cell} ipython3
def build_noise_model(depol_strength):
    import pygsti
    from loqs.backends import PyGSTiNoiseModel

    qubits = ["A0", "A1", "A2"] + [f"D{i}" for i in range(7)]
    model_qubits = [f"Q{i}" for i in range(len(qubits))]
    gate_names = ["Gxpi", "Gypi", "Gzpi", "Gzpi2", "Gzmpi2", "Gh", "Gcnot", "Gi"]

    pspec = pygsti.processors.QubitProcessorSpec(
        len(model_qubits),
        gate_names=gate_names,
        qubit_labels=model_qubits,
        availability={k: "all-permutations" for k in gate_names},
    )
    model_pygsti = pygsti.models.create_crosstalk_free_model(
        pspec,
        depolarization_strengths={name: depol_strength for name in gate_names if name != "Gi"},
        depolarization_parameterization="lindblad",
    )
    return PyGSTiNoiseModel(model_pygsti, qubits)
```

```{note}
Functions defined directly in a notebook cell (like `build_noise_model` above) are also worth an
extra step: `NoiseSweepRunner` normally captures a callable's source automatically via
`inspect.getsource`, but LoQS's helper for that also needs to re-open the file the function came
from to check for extra needed imports -- and a notebook cell isn't backed by a real file on disk,
so that second step fails. `Instruction` has this same rough edge (see e.g. the `serialized_apply_fn`
workaround in the [FT Tests](fttests.md) notebook). The fix is the same either way: capture the
source with plain `inspect.getsource` ourselves (which *does* work directly against the notebook
kernel's cell cache) and hand it to `NoiseSweepRunner` via `serialized_callables`, bypassing
automatic detection entirely. If you instead define your noise model/stack-building functions in a
regular `.py` file and import them, none of this is necessary.
```

```{code-cell} ipython3
import inspect

noise_model_source = inspect.getsource(build_noise_model)
```

## Building a `NoiseSweepRunner`

`NoiseSweepRunner` takes the full range of noise-strength values to sweep over, plus every
`QuantumProgram` constructor argument (except `default_base_seed`, which the runner controls
itself) -- each one given as either a fixed value (used unchanged at every point, like
`instruction_stack` and `initial_history` here) or a callable of one noise-strength value (like
`default_noise_model` here).

```{code-cell} ipython3
strengths = [0.001, 0.02, 0.05, 0.12]

runner = NoiseSweepRunner(
    strengths,
    instruction_stack=stack_z,
    initial_history=init_history,
    default_noise_model=build_noise_model,
    serialized_callables={"default_noise_model": noise_model_source},
    seed_stride=200,  # comfortably larger than num_shots below, so per-point seed ranges never overlap
    name="Steane Z-basis QEC sweep",
)
```

Passing `default_base_seed` here would raise a `TypeError` -- seeding is entirely the runner's
job, so that every sweep is deterministic and reproducible regardless of what the noise model or
instruction stack do internally. Point `i`'s `QuantumProgram` always gets seed
`base_seed + i * seed_stride`.

```{code-cell} ipython3
# The QuantumProgram for one sweep point can be inspected directly, if you want to sanity-check
# what will actually get run before committing to a full sweep.
example_program = runner.build_program(2)  # the strength=0.05 point
print(example_program.name, example_program.default_base_seed)
print(example_program.default_noise_model)
```

## Running the sweep

`run` builds and executes one `QuantumProgram` per strength, extracts a `(failure_rate, stderr)`
pair from each using the same per-shot pass/fail convention as `fttools.test_program_output`
(`collect_shot_data_args`/`expected_outcomes`), and returns a single `NoiseSweepResult` covering
the whole sweep.

```{code-cell} ipython3
result_z = runner.run(
    num_shots=40,
    collect_shot_data_args=[("logical_measurement", -1)],
    expected_outcomes=[0],
)

result_z.failure_rates
```

```{code-cell} ipython3
result_z.stderrs
```

```{code-cell} ipython3
# is_complete is False for a sweep that was interrupted partway through (see "Resuming an
# interrupted sweep" below) -- here, since we ran it start-to-finish, it's simply True.
result_z.is_complete
```

## Plotting

`plot_noise_sweep` reads directly from `failure_rates`/`stderrs` and produces a log-log plot.
`reference_slope=2` overlays a dashed guide line with slope 2, the naive expectation for a
distance-3 code's $p^2$ scaling.

```{code-cell} ipython3
plot_noise_sweep(result_z, reference_slope=2)
```

## Comparing multiple sweeps

Let's run the same sweep again, but for the $X$-basis stack, and compare the two. `compare_noise_sweeps`
validates that a set of named results share the same `strengths`/`num_shots` (so they're safe to
plot together) and hands them back unchanged; `plot_noise_sweep` accepts that same mapping directly
to draw one series per entry.

```{code-cell} ipython3
runner_x = NoiseSweepRunner(
    strengths,
    instruction_stack=stack_x,
    initial_history=init_history,
    default_noise_model=build_noise_model,
    serialized_callables={"default_noise_model": noise_model_source},
    seed_stride=200,
    name="Steane X-basis QEC sweep",
)

result_x = runner_x.run(
    num_shots=40,
    collect_shot_data_args=[("logical_measurement", -1)],
    expected_outcomes=[0],
)

results = compare_noise_sweeps({"Z basis": result_z, "X basis": result_x})
plot_noise_sweep(results, reference_slope=2)
```

`compare_noise_sweeps` raises a `ValueError` if the named results don't share the same
`strengths`/`num_shots` (they'd never be comparable on the same axes). It also has a `strict`
parameter for a different, unrelated situation covered below -- comparing sweeps where one hasn't
finished yet.

## Saving and reloading a `NoiseSweepResult`

A `NoiseSweepResult` is `Displayable`/`Serializable`, so it can be written and read back like any
other LoQS object, without needing to keep the original `NoiseSweepRunner` (or re-run any shots)
around.

```{code-cell} ipython3
result_z.write("steane_zsweep_result.json")

reloaded_result = NoiseSweepResult.read("steane_zsweep_result.json")
reloaded_result.failure_rates == result_z.failure_rates
```

## Saving and reloading a `NoiseSweepRunner`

The runner itself is serializable too, including the `default_noise_model` callable we gave it --
`NoiseSweepRunner` keeps its source code (the `serialized_callables` override we supplied above)
separate from the ordinary, directly-serializable `QuantumProgram` arguments, and reconstructs a
live, callable function again on read.

```{code-cell} ipython3
runner.write("steane_zsweep_runner.json")

reloaded_runner = NoiseSweepRunner.read("steane_zsweep_runner.json")
reloaded_runner.build_program(0).default_noise_model
```

### What if there's no source available at all?

`serialized_callables` isn't just a workaround for notebook cells -- it's required for *any*
callable with no retrievable source at all, such as one built dynamically with `exec`. Without it,
`NoiseSweepRunner` raises immediately, at construction time, rather than letting the problem
surface later as a confusing failure on `.write()`:

```{code-cell} ipython3
env = {}
exec("def rebuilt_noise_model(strength):\n    return build_noise_model(strength)\n", env)

try:
    NoiseSweepRunner(
        [0.001],
        instruction_stack=stack_z,
        default_noise_model=env["rebuilt_noise_model"],
        seed_stride=10,
    )
except OSError as e:
    print(f"Failed as expected: {e}")
```

Here there's no `inspect.getsource` fallback available either (there's genuinely no source on
disk anywhere), so we have to write out the source by hand instead:

```{code-cell} ipython3
runner_with_override = NoiseSweepRunner(
    [0.001],
    instruction_stack=stack_z,
    default_noise_model=env["rebuilt_noise_model"],
    serialized_callables={
        "default_noise_model": "def rebuilt_noise_model(strength):\n    return build_noise_model(strength)\n"
    },
    seed_stride=10,
)
```

## Keeping raw shot data with `keep_program_results`

By default, each sweep point's raw `ProgramResults` (the full per-shot histories) is discarded as
soon as its failure rate has been extracted -- only the summary `NoiseSweepResult` is kept. If you
want to dig into the raw shot data later (e.g. to try a different pass/fail criterion without
re-running anything), pass `keep_program_results=True` along with `program_results_dir`, a path
stem that gets a `_sweep_<index>` suffix inserted for each point.

```{code-cell} ipython3
result_kept = runner.run(
    num_shots=20,
    collect_shot_data_args=[("logical_measurement", -1)],
    expected_outcomes=[0],
    keep_program_results=True,
    program_results_dir="steane_zsweep_shots.json",
)

result_kept.program_results_paths
```

```{code-cell} ipython3
from collections import Counter

# Re-load the raw ProgramResults for the strength=0.12 point (index 3, the noisiest one we
# swept) and look at the distribution of raw outcomes, rather than just the pass/fail rate.
raw_results = result_kept.load_program_results(3)
Counter(raw_results.collect_shot_data("logical_measurement", -1))
```

## Resuming an interrupted sweep

Large sweeps (many strengths, many shots each) can take a while, and it would be a shame to lose
all progress if the process is interrupted partway through. Passing `resume=True` together with
`result_path` writes the in-progress `NoiseSweepResult` out after *every* completed point, not just
at the end. If you call `run` again with the same `result_path`, already-completed points are
recognized and skipped entirely -- at most one point's worth of shots is ever repeated, no matter
how large the sweep is.

```{code-cell} ipython3
resumable_runner = NoiseSweepRunner(
    strengths,
    instruction_stack=stack_z,
    initial_history=init_history,
    default_noise_model=build_noise_model,
    serialized_callables={"default_noise_model": noise_model_source},
    seed_stride=200,
    name="Resumable sweep",
)

result_first_pass = resumable_runner.run(
    num_shots=20,
    collect_shot_data_args=[("logical_measurement", -1)],
    expected_outcomes=[0],
    resume=True,
    result_path="steane_zsweep_progress.json",
)
result_first_pass.failure_rates
```

Calling `run` again with the exact same arguments finds every point already recorded as complete in
`steane_zsweep_progress.json`, so it returns instantly without simulating anything:

```{code-cell} ipython3
result_second_pass = resumable_runner.run(
    num_shots=20,
    collect_shot_data_args=[("logical_measurement", -1)],
    expected_outcomes=[0],
    resume=True,
    result_path="steane_zsweep_progress.json",
)
result_second_pass.failure_rates == result_first_pass.failure_rates
```

If a sweep is genuinely interrupted mid-point (say, a crash while running the 3rd of 4 strengths),
resuming skips the first two points and re-runs the third from scratch -- shot-level resume
*within* a single point isn't supported yet (that needs a small change to `QuantumProgram.run`
itself, tracked separately), but this still bounds the worst-case wasted work to a single sweep
point, however large the overall sweep is.

## Comparing an in-progress sweep

Since a sweep can now legitimately be checkpointed mid-flight, `NoiseSweepResult` has an
`is_complete` property, and `compare_noise_sweeps` will still compare/plot an incomplete result by
default -- it just emits a warning rather than an error, since you may well want to check progress
on a sweep that's still running. Passing `strict=True` upgrades that to a hard error instead, for
situations (e.g. generating a final report) where every series needs to be finished.

```{code-cell} ipython3
import warnings

partial_result = NoiseSweepResult(
    strengths=strengths,
    failure_rates=result_z.failure_rates[:2],  # pretend only the first two points have finished
    stderrs=result_z.stderrs[:2],
    num_shots=result_z.num_shots,
)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    compare_noise_sweeps({"Z basis": result_z, "partial run": partial_result})
print(caught[0].message)
```

```{code-cell} ipython3
try:
    compare_noise_sweeps({"Z basis": result_z, "partial run": partial_result}, strict=True)
except ValueError as e:
    print(f"Failed as expected: {e}")
```

## Next steps

This covered the full `noisesweeptools` API: building a `NoiseSweepRunner` from fixed and
per-point-callable `QuantumProgram` arguments, running a sweep, plotting and comparing results,
persisting both `NoiseSweepRunner` and `NoiseSweepResult` objects, keeping raw shot data around for
later re-analysis, and resuming an interrupted sweep. For exhaustive discrete fault-injection
testing (rather than continuous noise-strength sweeps), see `loqs.tools.fttools` and the
[FT Tests](fttests.md) notebook instead.

```{code-cell} ipython3

```
