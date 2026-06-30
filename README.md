# LoQS v1.0

[![tests](https://github.com/sandialabs/LoQS/actions/workflows/loqs.yml/badge.svg)](https://github.com/sandialabs/LoQS/actions/workflows/loqs.yml)
[![coverage](https://coveralls.io/repos/github/sandialabs/LoQS/badge.svg?branch=main)](https://coveralls.io/github/sandialabs/LoQS?branch=main)

The *Lo*gical *Q*ubit *S*imulator (LoQS) is designed to simulate a few logical qubits with arbitrary noise models and arbitrary quantum *and* classical operations.

## Installation

The following installation instructions can be used on M1/M2 Macs using Anaconda/Miniconda to create a local virtual environment.

```
conda create -n loqs-env python=3.14
conda activate loqs-env
pip install -e .
```

There are various optional requirements that are available, including:

- `dask`: Enables usage of Dask for parallelizing over shots.
- `dev`: Allows the use of `black` and `flake8` prior to committing
(see Code Formatting and Linting below).
- `docs`: Allows building of the documentation (see Documentation below).
- `quantumsim`: Enables the QuantumSim (state) backend.
- `pygsti`: Enables the PyGSTi (circuit, model, state) backend.
- `stim`: Enables the STIM (state) backend.
- `test`: Allows testing (see Testing below)
- `visualization`: Enables some of the visualization tools in `loqs.tools`. Note that
  `pdflatex` is also required for full visualization support.

There are several helper "categories" for optional dependencies, including:

- `backends`: Packages needed to enable *all* backends
- `nobackends`: The complement of `backends`, i.e. all developer packages with no backends
(useful for testing)
- `all`: All optional dependencies

To use these, simply modify the last line of the installation instructions. For example:

```
pip install -e ".[all]"
```

(where the quotes are only needed if using zsh instead of bash).

For developers who may want an editable version of `pyGSTi`, you can run:

```
pip install -e git+https://github.com/sandialabs/pyGSTi.git@v0.9.14#egg=pyGSTi
```

to get the 0.9.14 release of pyGSTi, which will be located in `src`.
Alternatively, you can use any other tag or commit hash instead of `v0.9.14`
if you are working off of a feature branch.

## Documentation

This project uses MkDocs and Jupytext-compatible Markdown notebooks under `docs/notebooks/` for its documentation and interactive tutorials. In order to build or preview the documentation locally, do at least an installation of `loqs[docs]`.

### Interactive Cloud Notebooks (Binder)

The tutorials and examples are configured to be run interactively in the cloud using **Binder**! You can open any of the tutorial or example notebooks directly in your browser by clicking the "Launch Binder" badges on the generated documentation pages.

Alternatively, you can launch the interactive Binder environment for the entire repository here:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sandialabs/LoQS/docs-updates)

### Local Building & Preview

To build and serve the documentation locally, run:

```
python docs/serve.py
```

This will launch a local server typically accessible at `http://127.0.0.1:8000/`.

More details on the documentation structure and Jupytext workflow are available in [docs/DOCS_README](docs/DOCS_README).
