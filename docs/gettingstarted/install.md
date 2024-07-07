# Installing LoQS

The following installation instructions can be used on M1/M2 Macs using Anaconda/Miniconda to create a local virtual environment.

By default, `LoQS` will install without any backends present.
In order to have feature parity with previous `LoQS`, users can install with the `pygsti,quantumsim` optional dependencies.
Also note that `LoQS` requires Python 3.10 or higher.

```
conda create -p ./venv python=3.11
conda activate ./venv
pip install -e ".[pygsti,quantumsim]"
```

There are various optional requirements that are available, including:

- `dev`: Allows the use of `black` and `flake8` prior to committing
(see Code Formatting and Linting below).
- `docs`: Allows building of the JupyterBook documentation (see Documentation below).
- `quantumsim`: Enables the QuantumSim backend.
- `pygsti`: Enables the PyGSTi backend.
- `test`: Allows testing (see Testing below)

There are several helper "categories" for optional dependencies, including:

- `backends`: Packages needed to enable *all* backends
- `nobackends`: The complement of `backends`, i.e. all developer packages with no backends
(useful for testing)
- `all`: All optional dependencies


`LoQS` should now be installed and ready to go!

Next, we will show how to create and run a basic `QuantumProgram` in `LoQS`.