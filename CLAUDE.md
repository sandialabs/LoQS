# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LoQS (Logical Qubit Simulator) is a quantum computing simulator focused on logical qubit operations with support for quantum error correction codes. The project is designed to handle complex classical information processing alongside quantum operations.

## Codebase Structure

### Main Directories

- `loqs/`: Main source code
  - `core/`: Core simulation components (instructions, frames, history, etc.)
  - `backends/`: Backend implementations (circuit, model, state)
  - `codepacks/`: Quantum error correction code implementations
  - `tools/`: Utility functions and tools
  - `internal/`: Internal utilities

- `tests/`: Test suite
- `docs/`: Documentation (JupyterBook format)

The following directories are local, i.e. not committed to the repo
- `src/`: Optional for external dependencies (like pyGSTi)
- `build/`: Build artifacts
- `htmlcov/`: Test coverage reports

### Key Architectural Concepts

1. **InstructionStack**: Replaces traditional circuit objects, can be dynamically updated during simulation
2. **Instructions**: Individual operations that output Frame objects
3. **Frame**: Snapshot of simulation state containing quantum state + additional classical information
4. **History**: Collection of Frames representing the simulation record
5. **QECCode**: Quantum Error Correction Code implementations
6. **QuantumProgram**: Main container for running logical qubit simulations
7. **Backends**: Abstract physical circuit simulation (circuit, noise model, state)

## Development Workflow

### Installation

```bash
# Basic installation
pip install -e .

# With all dependencies
pip install -e ".[all]"

# With specific backends
pip install -e ".[pygsti,quantumsim,stim]"
```

### Testing

```bash
# Run all tests using tox
tox run

# Run specific test environment
tox run -e py311  # Python 3.11 tests
tox run -e lint   # Linting
tox run -e type   # Type checking
tox run -e report # Coverage report

# Run pytest directly
pytest tests          # Run unit tests
pytest loqs           # Run doctests
pytest tests/core     # Run specific test module

# Run single test file
pytest tests/core/test_instructions.py

# Run single test function
pytest tests/core/test_instructions.py::test_specific_function
```

### Code Quality

```bash
# Format code with black
black loqs

# Lint with flake8
flake8 loqs

# Type checking with mypy
mypy loqs

# Setup pre-commit hooks
pre-commit install
pre-commit run  # Run on staged files
```

### Documentation

```bash
# Build documentation
jupyter-book build docs

# Clean and rebuild docs
jb clean docs
jb build docs

# Sync Jupytext notebooks
jupytext --sync docs/markdown/*
```

## Common Development Tasks

### Adding a New QEC Code

1. Create new code implementation in `loqs/codepacks/`
2. Implement required `QECCode` interface methods
3. Add tests in `tests/codepacks/`
4. Update documentation in `docs/markdown/codepacks.md`

### Adding a New Backend

1. Create backend implementation in appropriate `loqs/backends/{circuit,model,state}/` directory
2. Implement required backend interface
3. Add backend-specific tests
4. Update `pyproject.toml` with new optional dependency

### Adding a New Instruction

1. Create instruction class in `loqs/core/instructions/`
2. Implement `Instruction` interface (must output `Frame` objects)
3. Add tests in `tests/core/instructions/`
4. Register instruction in appropriate instruction sets

## Important Files

- `pyproject.toml`: Project configuration and dependencies
- `tox.ini`: Testing configuration
- `README.md`: Installation and setup instructions
- `docs/markdown/overview.md`: High-level architecture overview
- `docs/markdown/devguide.md`: Development guide

## Backend Information

The project supports multiple backends:
- **PyGSTi**: Circuit, model, and state backends
- **QuantumSim**: State backend
- **STIM**: State backend
- **Basic Python objects**: Circuit and model backends
- **NumPy**: State backend

Backend-specific code should be isolated in the `loqs/backends/` directory to maintain backend-agnostic design where possible.

## Testing Notes

- Tests are organized by module structure (core, backends, codepacks, tools)
- Doctests are run on the source code using `pytest loqs`
- Coverage is tracked and reported via `pytest-cov`
- Test environments are managed by `tox` for consistency

## Documentation Notes

- Documentation uses JupyterBook with MyST Markdown
- Some markdown files are Jupytext notebooks that can be converted to/from .ipynb format
- Documentation is auto-built in CI/CD pipeline
- `PYTHONPATH` may need to be set to root directory for Sphinx autodoc to work properly