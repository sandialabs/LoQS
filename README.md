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
conda create -p ./venv python=3.11
conda activate ./venv
pip install -e src/quantumsim
pip install -e .
```

There are various optional requirements that are available, such as `test`, `docs`, `examples`, and `dev`.
If you want to install all optional dependencies, you can use `all`.

To use these, simply modify the last line of the installation instructions. For example:

```
pip install -e ".[all]"
```

(where the quotes are only needed if using zsh instead of bash).

### Using LoQS

TODO. I plan to have a series of examples/tutorials that we point users to.

## Developer Guide
### Editable pyGSTi

For developers who may want an editable version of `pyGSTi`, you can run:

```
pip install -e git+https://github.com/sandialabs/pyGSTi.git@v0.9.12#egg=pyGSTi
```

to get the 0.9.12 release of pyGSTi, which will be located in `src`.
Alternatively, you can use any other tag or commit hash instead of `v0.9.12`
if you are working off of a feature branch.

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

### Documentation

This project uses `JupyterBook` for its documentation, which should be
installed if using the `[docs]` optional dependencies.
I have also noticed that `PYTHONPATH` needs to be set to the root directory
in order for `sphinx.ext.autosummary` to find the package properly.

For a developer with a local install, this should build the docs:

```
pip install -e .'[docs]'
PYTHONPATH=. jupyter-book build docs
```

Open `docs/_build/html/index.html` and view your docs!

Something close to the following should work for `Sphinx`-only builds:

```
jupyter-book config sphinx docs # Generate the Sphinx conf.py file
PYTHONPATH=. sphinx-build docs docs/_build/html -b html
```