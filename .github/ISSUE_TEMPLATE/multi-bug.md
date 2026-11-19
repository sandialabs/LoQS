---
name: Multiple related bugs
about: Report several related bugs in one issue, with a checklist and one section per bug.
title: ''
type: Bug
assignees: ''

---

<!--
Use this template to group related bugs, e.g. several small bugs found in
the same area or during the same piece of work. For a single bug, feature
or task, use the "Single issue" template instead. Please search the
existing issues first.

A good title names the part of LoQS it affects, then a short summary,
e.g. "Serialization: ..." or "Tools: ...".

Link related issues or PRs inline where they matter, e.g.
Repo style preference to avoid GitHub expanding out to titles:
[#123](https://github.com/sandialabs/LoQS/issues/123)
(use /pull/123 for a PR).
-->

## Summary

<!-- The shared context, such as how these bugs were found and how they're related. -->

<!--
One checklist line per bug, then one section per bug below.
New sections can be added later via comments.
-->

- [ ] Bug 1: [short label]
- [ ] Bug 2: [short label]

---
### Bug 1: [one-line symptom]

#### Problem
<!-- What goes wrong or is missing, and what you expected instead. -->

#### MWE
<!--
Code that fails today and should pass once it's fixed, with the error or
wrong output as a comment. Or drag and drop a notebook (.ipynb) or script
here to attach it.
-->

```python
import loqs
...
# ValueError: ...
```

<!--
#### Fix

A proposed fix, if you have one, with any real caveats.
-->

---
### Bug 2: [one-line symptom]

#### Problem

#### MWE

<!-- #### Fix --->

---
## Environment

- LoQS version or commit: <!-- e.g. 1.3.1, or 84053ba -->
- Python version: <!-- e.g. 3.12 -->
- OS: <!-- e.g. macOS 15, Ubuntu 24.04 -->

<!--
Before you submit: this issue's type is set to Bug. If these are features or
tasks instead, change the Type below. If you can, also set the Priority and
Effort fields below for the issue as a whole (Low if unsure); otherwise a
maintainer will set them.
-->
