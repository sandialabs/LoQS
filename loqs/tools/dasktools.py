"""TODO"""

from collections.abc import Sequence

try:
    from dask.distributed import Client
except ImportError:
    Client = None

from loqs.core import QuantumProgram


# TODO: This currently runs super slow, don't use?
def run_program_list(
    programs: Sequence[QuantumProgram],
    dask_client: Client,  # type: ignore
    shots_per_program: int,
    shots_per_program_per_batch: int,
    max_frame_limit: int = 100,
) -> None:
    """TODO"""
    histories_list = []
    tasks_by_program = []
    for program in programs:
        # Store old shots to minimize data needed to copy
        histories_list.append(program.shot_histories)
        program.shot_histories = []

        tasks = []
        for i in range(shots_per_program):
            # For RNG seeding, increment the base seed +1 for every shot (if seeded)
            seed_for_shot = None
            if program.default_base_seed is not None:
                seed_for_shot = (
                    program.default_base_seed + len(histories_list[-1]) + i
                )

            tasks.append((max_frame_limit, seed_for_shot))

        tasks_by_program.append(tasks)

    # Scatter all programs
    dask_client.scatter(programs, broadcast=True)

    # Helper function to run a chunk of shots at once
    # This should use the already remote programs and avoid more copying
    def _run_shot_program_batch(tasks: Sequence[Sequence[tuple]]):
        results_by_program = []
        for program, tasks_per_program in zip(programs, tasks):
            full_tasks = [(program, *task) for task in tasks_per_program]
            results = QuantumProgram._run_shot_batch(full_tasks)
            results_by_program.append(results)
        return results_by_program

    # To help load balancing, we want to batch over programs for the same shots
    # We could in theory let Dask load balance everything, but we know a good heuristic
    # and we can avoid scheduler overhead this way
    batched_task_list = [
        [
            tasks_per_program[i : i + shots_per_program_per_batch]
            for tasks_per_program in tasks_by_program
        ]
        for i in range(0, shots_per_program, shots_per_program_per_batch)
    ]

    # Not pure because RNG for shots underneath
    futures = dask_client.map(
        _run_shot_program_batch, batched_task_list, pure=False
    )

    # Retrive results (blocks until all tasks are finished)
    batched_results = dask_client.gather(futures)

    results_per_program = []
    for i, batch_results in enumerate(batched_results):  # type: ignore
        for j, batch_results_per_program in enumerate(batch_results):
            if i == 0:
                results_per_program.append(batch_results_per_program)
            else:
                results_per_program[j].extend(batch_results_per_program)

    for i, program in enumerate(programs):
        program.shot_histories = histories_list[i] + results_per_program[i]
