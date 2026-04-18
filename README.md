# LoQS v1.0

[![tests](https://github.com/sandialabs/LoQS/actions/workflows/loqs.yml/badge.svg)](https://github.com/sandialabs/LoQS/actions/workflows/loqs.yml)
[![coverage](https://coveralls.io/repos/github/sandialabs/LoQS/badge.svg?branch=main)](https://coveralls.io/github/sandialabs/LoQS?branch=main)

The *Lo*gical *Q*ubit *S*imulator (LoQS) is designed to simulate a few logical qubits with arbitrary noise models and arbitrary quantum *and* classical operations.

## Installation

The following installation instructions can be used on M1/M2 Macs using Anaconda/Miniconda to create a local virtual environment.

```
conda create -p ./venv python=3.11
conda activate ./venv
pip install -e .
```

By default, this will not install any of the backends.
In order to install PyGSTi and QuantumSim (i.e. previous LoQS backends),
you can alter the last line to 

```
pip install -e ".[pygsti,quantumsim]"
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
pip install -e git+https://github.com/sandialabs/pyGSTi.git@v0.9.12#egg=pyGSTi
```

to get the 0.9.12 release of pyGSTi, which will be located in `src`.
Alternatively, you can use any other tag or commit hash instead of `v0.9.12`
if you are working off of a feature branch.

## Documentation

This project uses Marimo notebooks and MkDocs for documentation.
In order to use these features, do at least a installation of `loqs[docs]`.

To interactively edit and run the notebooks locally, run: `marimo edit docs`

To build and serve the documentation, run: `python docs/serve.py`

Both commands will launch local servers that can be navigated to in your web browser of choice.
Marimo will auto-open the browser, while `serve.py` will simply tell you the URL
(typically `localhost:8000`).

More details are available in (docs/DOCS_README.md)[docs/DOCS_README.md].