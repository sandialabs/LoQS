# Ralph Context

We are migrating from a Sphinx-based documentation system to a custom MkDocs one.
Details for the new system are written in `docs/DOCS_README`.
Read this for context.

Our task now is to audit every docstring in the codebase.
Follow the task list, and use the below descriptions to help
you generate docstrings.

Do NOT apply any scripts to the entire codebase. Handle every task independently.
Do NOT alter any lines of code that are not docstrings.
If you change anything that is not a docstring, revert and try again.

## Legend

This is the legend of docstring issues in the task list.

- `no_docstring`: Public function has no docstring
- `sphinx_reference`: Docstring contains Sphinx-like reference (e.g., :class:)
- `non_numpy_format`: Docstring does not conform to NumPy format

### Fixing `no_docstring`

Take your best guess at generating a Numpy-formatted docstring for this function.
Include a REVIEW_NO_DOCSTRING tag in the function description.

### Fixing `sphinx_reference`

The docs/DOCS_README details the new crosslinking system.
In short, replace things like :class:`ClassName` with `(ClassName)[api:ClassName]`, etc.
Use progressively more qualified names in the `[api:]` links as needed to disambiguate,
but keep the shortest name possible.
Include a REVIEW_SPHINX_REFERENCE tag in the function description.

### Fixing `non_numpy_format`

Parse the current documentation and try to regenerate a Numpy-formatted docstring for this function.
Include a REVIEW_NUMPY_FORMAT tag in the function description.