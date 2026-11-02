#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""`python -m loqs.codepacks.generator <module>:<Class> [...]` writes each
named codepack class's generated artifact and prints its path. A class
that ships with LoQS also gets its document in the checkout's `docs/`."""

import importlib
import sys
from pathlib import Path

import loqs
from loqs.codepacks.generator.codepack import shipped_codepacks
from loqs.codepacks.generator.document import write_document
from loqs.codepacks.generator.generate import generate, write_artifact


def main(argv: list[str]) -> int:
    if not argv:
        print(
            "usage: python -m loqs.codepacks.generator <module>:<Class> "
            "[...]",
            file=sys.stderr,
        )
        return 2
    docs = Path(loqs.__file__).parents[1] / "docs"
    shipped = shipped_codepacks()
    classes = []
    for target in argv:
        module, _, name = target.partition(":")
        cls = getattr(importlib.import_module(module), name)
        if cls in shipped and not (docs.parent / "mkdocs.yml").exists():
            raise FileNotFoundError(
                f"{cls.__name__} ships with LoQS and has a document in "
                f"{docs}, but {docs.parent / 'mkdocs.yml'} doesn't exist: "
                "run this from a LoQS checkout"
            )
        classes.append(cls)
    for cls in classes:
        print(write_artifact(cls))
        if cls in shipped:
            print(write_document(cls, generate(cls.spec), docs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
