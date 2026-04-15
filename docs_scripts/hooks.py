from __future__ import annotations

import re

MARK_RE = re.compile(
    r"<!--\s*API_METHOD\s+owner=([^\s]+)\s+member=([^\s]+)\s*-->",
    re.IGNORECASE,
)

# We operate only inside a single API_METHOD block region.

DOC_CLASS_OPEN_RE = re.compile(r'<div class="doc doc-object doc-class"[^>]*>', re.IGNORECASE)
CONTENTS_FIRST_OPEN_RE = re.compile(r'<div class="doc doc-contents first"[^>]*>', re.IGNORECASE)
CHILDREN_OPEN_RE = re.compile(r'<div class="doc doc-children"[^>]*>', re.IGNORECASE)

# Leading <p> blocks we want to strip (bases and docstring live here)
LEADING_P_RE = re.compile(r'^\s*<p\b[^>]*>.*?</p>\s*', re.IGNORECASE | re.DOTALL)


def _find(m: re.Pattern, s: str, start: int = 0) -> re.Match | None:
    return m.search(s, start)


def _strip_intro_from_block(block_html: str) -> str:
    """
    Given the HTML covering one API_METHOD marker + its rendered mkdocstrings block,
    remove all <p>...</p> elements inside:
        <div class="doc doc-contents first">
            ... <p>bases</p>
            ... <p>class docstring</p>
            <div class="doc doc-children"> ... keep ...
    """
    m_doc = _find(DOC_CLASS_OPEN_RE, block_html, 0)
    if not m_doc:
        return block_html

    m_contents = _find(CONTENTS_FIRST_OPEN_RE, block_html, m_doc.end())
    if not m_contents:
        return block_html

    m_children = _find(CHILDREN_OPEN_RE, block_html, m_contents.end())
    if not m_children:
        return block_html

    # Partition:
    #   [0 : contents_end] + [contents_end : children_start] + [children_start : end]
    prefix = block_html[: m_contents.end()]
    mid = block_html[m_contents.end() : m_children.start()]
    suffix = block_html[m_children.start() :]

    # Strip all leading <p> blocks from mid (bases/docstring/etc)
    while True:
        m_p = LEADING_P_RE.match(mid)
        if not m_p:
            break
        mid = mid[m_p.end() :]

    return prefix + mid + suffix


def on_post_page(output: str, page, config) -> str:
    marks = list(MARK_RE.finditer(output))
    if not marks:
        return output

    out_parts: list[str] = []
    last = 0

    for idx, m in enumerate(marks):
        start = m.start()
        end = marks[idx + 1].start() if idx + 1 < len(marks) else len(output)

        out_parts.append(output[last:start])
        block = output[start:end]
        out_parts.append(_strip_intro_from_block(block))
        last = end

    out_parts.append(output[last:])
    return "".join(out_parts)