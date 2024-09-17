"""TODO"""

from collections.abc import Sequence

from dask.distributed import Client

from loqs.core import QuantumProgram


def run_program_list(
    programs: Sequence[QuantumProgram],
    dask_client: Client,
    shots_per_program: int,
    shots_per_program_per_batch: int,
    default_base_seed: int | None = None,
    dry_run: bool = False,
    max_frame_limit: int = 100,
) -> None:
    """TODO"""
    histories_list = []
    task_list = []
    for program in programs:
        # Store old shots to minimize data needed to copy
        histories_list.append(program.shot_histories)
        program.shot_histories = []

        # Scatter to all workers
        program = dask_client.scatter(program)

        tasks = []
        for i in range(shots_per_program):
            # For RNG seeding, increment the base seed +1 for every shot (if seeded)
            seed_for_shot = None
            if default_base_seed is not None:
                seed_for_shot = default_base_seed + len(histories_list[-1]) + i

            tasks.append((program, dry_run, max_frame_limit, seed_for_shot))

        task_list.append(tasks)

    # To help load balancing, we want to batch over programs for the same shots
    # We could in theory let Dask load balance everything, but we know a good heuristic
    # and we can avoid scheduler overhead this way
    batched_task_list = [
        [
            tasks_per_program[i : i + shots_per_program_per_batch]
            for tasks_per_program in task_list
        ]
        for i in range(0, shots_per_program, shots_per_program_per_batch)
    ]

    # Not pure because RNG for shots underneath
    futures = dask_client.map(
        _run_shot_program_batch, batched_task_list, pure=False
    )

    # Retrive results (blocks until all tasks are finished)
    batched_results = dask_client.gather(futures)

    results_per_program = [[]] * len(programs)
    for batch_results in batched_results:  # type: ignore
        for i, batch_results_per_program in enumerate(batch_results):
            results_per_program[i].extend(batch_results_per_program)

    for i, program in enumerate(programs):
        program.shot_histories = histories_list[i] + results_per_program[i]


# Helper function to run a chunk of shots at once
def _run_shot_program_batch(tasks: Sequence[Sequence[tuple]]):
    return [
        [QuantumProgram._run_shot(*task) for task in tasks_per_program]
        for tasks_per_program in tasks
    ]
