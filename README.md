# LoQS (Public Release)

This repository is intended to be a sanitized version of the Logical Qubit Simulator (LoQS) for eventual public release.
Note that this repo is currently on CEE-GitLab only out of an abundance of caution while porting things from LoQS.
It is eventually intended to be public on GitHub in the SandiaLabs organization.

Remaining items before ready for release:

- [] Port over necessary simulation functions from `loqs-code`
- [] Refactor all high-level objects (syndrome data, decoder, etc)
- [] Add subsystem surface code and 9-qubit Shor as additional codepacks
- [] Add some semblance of testing and tutorials
- [] Deal with QuantumSim licensing. It is GPL, and we almost certainly don't want to distribute it/have to GPL LoQS.

Note that this is not a full "LoQS 2.0" wishlish. A full refactor would also include major backend changes, which we are avoiding here. However, this should move us to at least a LoQS 1.0 release when we go public, i.e. this should be a reasonable interface when we are done.

## Installation Instructions

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
