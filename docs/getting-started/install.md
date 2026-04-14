# Installing LoQS

By default, `LoQS` will install without any backends present.
In order to have feature parity with previous `LoQS`, users can install with the `pygsti,quantumsim` optional dependencies.
Also note that `LoQS` requires Python 3.10 or higher.

1. Create a virtual environment using your favorite tool, e.g. `conda`, `uv`, `virtualenv`, etc.
1. Install LoQS!
    1. If installing from PyPi: `pip install loqs`
    1. If installing from source:
        1. `git clone https://github.com/sandialabs/LoQS.git`
        1. `cd LoQS`
        1. `pip install .` (include `-e` for an editable install)

There are various optional requirements that are available, including:

- `dask`: Allows the use of Dask for parallelization
- `dev`: Allows the use of `black` and `flake8` prior to committing
- `docs`: Allows building of the documentation
- `quantumsim`: Enables the QuantumSim state backend.
- `pygsti`: Enables the PyGSTi circuit/model backends.
- `stim`: Enables the STIM circuit/state backends.
- `test`: Allows testing (see Testing below)
- `visualization`: Enables visualization (primarily for circuit generation currently)

There are several helper "categories" for optional dependencies, including:

- `backends`: Packages needed to enable *all* backends
- `nobackends`: The complement of `backends`, i.e. all developer packages with no backends
(useful for testing)
- `all`: All optional dependencies

`LoQS` should now be installed and ready to go!

Next, we will show how to create and run a basic `QuantumProgram` in `LoQS`.