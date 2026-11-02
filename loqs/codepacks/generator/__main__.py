#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""`python -m loqs.codepacks.generator <module>:<Class> [...]` writes each
named codepack class's generated artifact and prints its path."""

import importlib
import sys

from loqs.codepacks.generator.generate import write_artifact


def main(argv: list[str]) -> int:
    if not argv:
        print(
            "usage: python -m loqs.codepacks.generator <module>:<Class> "
            "[...]",
            file=sys.stderr,
        )
        return 2
    for target in argv:
        module, _, name = target.partition(":")
        cls = getattr(importlib.import_module(module), name)
        print(write_artifact(cls))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
