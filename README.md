# LoQS v1.0

Static badges based on this [CI run](https://github.com/sandialabs/pyGSTi/actions/runs/23029812177/job/66885596222).

![v1.0](https://img.shields.io/badge/v1.0-passing-brightgreen)
![coverage](https://img.shields.io/badge/coverage-67%25-red)

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
- `docs`: Allows building of the JupyterBook documentation (see Documentation below).
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

### Visualization

LoQS now has some capability to turn circuit diagrams into LaTeX via the quantikz package.
This requires `pdflatex`, commonly from the a TeX installation, as well as `loqs[visualization]`.

## Documentation

This project uses JupyterBook for documentation.
Assuming the `docs` requirements have been installed, the documentation can be generated via:

```
jupyterbook build docs
```

and then viewed by opening `docs/_build/html/index.html` in a browser.

### Jupytext Notebooks

For users who want executable versions of the MyST Markdown can use Jupytext to turn them into IPython/Jupyter notebooks.
```
jupytext --sync docs/markdown/*
```

will synchronize all the Markdown files in `docs/markdown` with the Jupyter notebooks in `docs/notebook`. Only the 
Markdown is committed and used for generating the JupyterBook, but the notebooks can be handy to test execution
in an interactive way.