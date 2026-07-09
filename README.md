# LoQS v1.1

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
- `pymatching`: Enables the PyMatching (minimum-weight perfect matching) decoder for certain codepacks.
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

### Local Building & Preview

To build and serve the documentation locally, run:

```
python docs/serve.py
```

This will launch a local server typically accessible at `http://127.0.0.1:8000/`.

More details on the documentation structure and Jupytext workflow are available in [docs/DOCS_README](docs/DOCS_README).

### Interactive Cloud Notebooks (Binder)

The tutorials and examples are configured to be run interactively in the cloud using **Binder**! You can open any of the tutorial or example notebooks directly in your browser by clicking the "Launch Binder" badges on the generated documentation pages.

Note that the first launch may take up to 5-10 minutes to build the Binder environment.

### Contributing Interactive Notebooks

The easiest way to modify and add interactive notebooks is to use Jupytext.

1. Navigate to `docs/notebooks`, where all interactive notebooks are kept as Markdown for easy source control.
1. Generate the corresponding Jupyter notebook via `jupytext --to ipynb <target_to_edit>.md`
1. Edit the notebooks, e.g. `jupyter lab` to start a server, edit, and save.
1. Sync them back to Markdown via `jupytext --to myst <target_to_edit>.md`
1. Commit your changes!