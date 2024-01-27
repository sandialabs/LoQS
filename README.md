# LoQS (Public Release)

This repository is intended to be a sanitized version of the Logical Qubit Simulator (LoQS)
for eventual public release. Note that this repo is currently on CEE-GitLab only out of an
abundance of caution while porting things from LoQS. It is eventually intended to be public
on GitHub in the SandiaLabs organization.

Remaining items before ready for release:

- [ ] Port over necessary simulation functions from `loqs-code`
- [ ] Refactor all high-level objects (syndrome data, decoder, etc)
- [ ] Add subsystem surface code and 9-qubit Shor as additional codepacks
- [ ] Add some semblance of testing, documentation, and tutorials
- [ ] Deal with QuantumSim licensing. It is GPL, and we almost certainly don't want that in LoQS

Note that this is not a full "LoQS 2.0" wishlish. A full refactor would also include major
backend changes, which we are avoiding here. However, this should move us to at least a LoQS 1.0
release when we go public, i.e. this should be a reasonable interface when we are done.

## User Guide
### Installation

The following installation instructions can be used on M1/M2 Macs using Anaconda/Miniconda to create a local virtual environment.

```
conda create -p ./venv python=3.11 cvxopt       # Note that we install cvxopt here for the M1/M2 Macs.
                                                # Other platforms can skip that and install via pip
conda activate ./venv
pip install -r requirements.txt
pip install -e src/quantumsim
```

## Using LoQS

TODO. I plan to have a series of examples/tutorials that we point users to.


## Developer Guide
### Code Formatting and Linting

This project uses `black` for autoformatting code and `flake8` for linting.
These will be automatically run as part of the CI/CD platform, but developers
can also set this up to be done locally by performing the following steps:

```
pip install pre-commit
pre-commit install
```

If you have staged files for a commit, you can then do `pre-commit run`
to run the script that will be ran during the commit process.
Alternatively, you can run `pre-commit run -a` to just format and lint
the entire codebase.