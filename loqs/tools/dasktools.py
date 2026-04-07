#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Collection of functions for using Dask.
"""

from collections.abc import Sequence
import json
import os
from tempfile import NamedTemporaryFile

try:
    from dask.distributed import Client, progress
except ImportError:
    Client = None  # type: ignore

from loqs.core import QuantumProgram


# TODO: Batch and shuffle for coarse-grained load balancing?
def run_program_list(
    programs: Sequence[QuantumProgram],
    dask_client: Client,  # type: ignore
    num_shots_per_program: int | Sequence[int],
    max_frame_limit: int = 100,
) -> None:
    """Run a set of programs in parallel.

    Parameters
    ----------
    programs:
        The programs to run

    dask_client:
        The Dask client to use when submitting shots

    num_shots_per_program:
        The number of shots to execute per program.
        See :meth:`.QuantumProgram.run`.

    max_frame_limit:
        See :meth:`.QuantumProgram.run`
    """
    if isinstance(num_shots_per_program, int):
        num_shots_per_program = [
            num_shots_per_program,
        ] * len(programs)
    else:
        num_shots_per_program = list(num_shots_per_program)
    assert len(num_shots_per_program) == len(programs)

    histories_list = []
    program_filenames = []
    all_tasks = []
    for program, num_shots in zip(programs, num_shots_per_program):
        # Store old num_shots to minimize data needed to copy
        histories_list.append(program.shot_histories)
        program.shot_histories = []

        # Write program to file so that other processes can load
        with NamedTemporaryFile("w+", suffix=".json", delete=False) as tempf:
            p_dict = program.to_serialization(ignore_no_serialize_flags=True)
            json.dump(p_dict, tempf)
            program_filenames.append(tempf.name)

        for j in range(num_shots):
            # For RNG seeding, increment the base seed +1 for every shot (if seeded)
            seed_for_shot = None
            if program.default_base_seed is not None:
                seed_for_shot = (
                    program.default_base_seed + len(histories_list[-1]) + j
                )

            all_tasks.append(
                (program_filenames[-1], max_frame_limit, seed_for_shot)
            )

    loaded_programs = {}

    def _run_task(task: tuple):
        if task[0] not in loaded_programs:
            loaded_programs[task[0]] = QuantumProgram.read(task[0])
        program = loaded_programs[task[0]]
        return QuantumProgram._run_shot(program, *task[1:])

    # Not pure because RNG for num_shots underneath
    futures = dask_client.map(_run_task, all_tasks, pure=False)

    # Retrive results (blocks until all tasks are finished)
    results = dask_client.gather(futures)

    # Add results back into programs
    offset = 0
    for i, (program, num_shots) in enumerate(
        zip(programs, num_shots_per_program)
    ):
        program.shot_histories = (
            histories_list[i] + results[offset : offset + num_shots]
        )
        offset += num_shots

    # Remove temporary files
    for fname in program_filenames:
        os.unlink(fname)
