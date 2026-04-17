#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MkDocs post-processing hooks for the LoQS API reference site.

This module is used only by the API-reference build (``mkdocs-api-ref.yml``).
It runs after mkdocstrings has rendered docstrings to HTML and performs several
cleanup and rewrite passes that are difficult or impossible to do reliably in
Markdown space.

Responsibilities
----------------
- Clean mkdocstrings-rendered HTML blocks:

  - Remove stray ``<a id="..."></a>`` anchors that can appear between the outer
    mkdocstrings container and the ``doc-contents first`` region.
  - Strip leading introductory paragraphs and leading doctest/code highlight
    blocks that would otherwise be duplicated across many per-member renders.

  Cleanup is applied only to blocks preceded by one of the generator markers:

  - ``<!-- API_METHOD owner=<cls_ident> member=<name> -->``
  - ``<!-- API_MODULE_MEMBERS owner=<mod_ident> -->``

- Prune the right-hand "On this page" TOC:

  - Remove anchors listed in ``<!-- API_TOC_REMOVE ... -->``. These are emitted
    by the reference-page generator to suppress redundant base-class entries.

- Rewrite API cross-references emitted in docstrings:

  - Convert ``href="api:Target"`` links (produced by mkdocstrings Markdown
    rendering) into concrete URLs using the generated API inventory
    (``docs/_api_inventory.json``). The API site is mounted under
    ``/reference`` in the merged documentation build, so rewritten links are
    prefixed accordingly.

- Rewrite citations in rendered docstrings:

  - Convert Pandoc-style citations like ``[@key]`` (and ``[@k1; @k2]``) into
    hyperlinks targeting the global bibliography page (e.g.
    ``/reference/bib/#fn:key``).

Notes
-----
- These hooks operate on *HTML* (``on_post_page``) because mkdocstrings renders
  docstrings late in the pipeline; earlier hooks like ``on_page_markdown`` do
  not see the fully expanded docstring content.
- The inventory file is generated during the API build by
  ``docs_scripts/gen_ref_pages.py`` and written to ``docs/_api_inventory.json``
  so it can be loaded during the same build.
"""

from __future__ import annotations

import re
from pathlib import Path

from docs_scripts.api_inventory import ApiInventory

# ----------------------------------------------------------------------
#  Block markers
# ----------------------------------------------------------------------
MARK_RE = re.compile(
    r"<!--\s*API_METHOD\s+owner=([^\s]+)\s+member=([^\s]+)\s*-->",
    re.IGNORECASE,
)

MODULE_MEMBERS_MARK_RE = re.compile(
    r"<!--\s*API_MODULE_MEMBERS\s+owner=([^\s]+)\s*-->",
    re.IGNORECASE,
)

TOC_REMOVE_RE = re.compile(r"<!--\s*API_TOC_REMOVE\s+([^>]+?)\s*-->", re.IGNORECASE)

# ----------------------------------------------------------------------
#  DIV boundaries created by mkdocstrings
# ----------------------------------------------------------------------
DOC_CLASS_OPEN_RE = re.compile(r'<div class="doc doc-object doc-class"[^>]*>', re.IGNORECASE)
DOC_MODULE_OPEN_RE = re.compile(r'<div class="doc doc-object doc-module"[^>]*>', re.IGNORECASE)

CONTENTS_FIRST_OPEN_RE = re.compile(r'<div class="doc doc-contents first"[^>]*>', re.IGNORECASE)
CHILDREN_OPEN_RE = re.compile(r'<div class="doc doc-children"[^>]*>', re.IGNORECASE)

# ----------------------------------------------------------------------
#  Unwanted markup
# ----------------------------------------------------------------------
LEADING_P_RE = re.compile(r"^\s*<p\b[^>]*>.*?</p>\s*", re.IGNORECASE | re.DOTALL)
CLASS_TOC_ANCHOR_RE = re.compile(r'<a\s+id="[^"]*"\s*></a>\s*', re.IGNORECASE)
LEADING_HIGHLIGHT_RE = re.compile(
    r'^\s*<div class="highlight"[^>]*>.*?</div>\s*',
    re.IGNORECASE | re.DOTALL,
)
LEADING_ADMONITION_RE = re.compile(
    r'^\s*<div class="admonition\b[^"]*"[^>]*>.*?</div>\s*',
    re.IGNORECASE | re.DOTALL,
)

# ----------------------------------------------------------------------
#  TOC <li> entry matcher (Material theme)
# ----------------------------------------------------------------------
TOC_LI_RE = re.compile(
    r'<li class="md-nav__item">\s*'
    r'<a href="#(?P<anchor>[^"]+)" class="md-nav__link">.*?</a>\s*'
    r"</li>",
    re.DOTALL | re.IGNORECASE,
)

API_A_TAG_RE = re.compile(
    r'<a(?P<attrs>[^>]*?)\s+href=(?P<q>["\'])api:(?P<target>[^"\'>\s]+)(?P=q)(?P<attrs2>[^>]*)>(?P<body>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

# Pandoc-style citations in rendered HTML text (from docstrings), e.g. [@key] or [@k1; @k2]
CITE_BRACKET_RE = re.compile(r"\[@(?P<keys>[^\]]+)\]")

def _find(pat: re.Pattern, s: str, start: int = 0) -> re.Match | None:
    return pat.search(s, start)


def _strip_intro_from_block(block: str) -> str:
    """
    Clean a single marked block (class or module):

    - Remove stray <a id="..."></a> that can sit between outer div and contents-first div
    - Strip leading <p> blocks inside the "doc-contents first" region (module/class doc blurb)
    - Strip leading <p> blocks at start of the "doc-children" region (repeated base/doc blurb)
    """
    m_doc = _find(DOC_CLASS_OPEN_RE, block, 0) or _find(DOC_MODULE_OPEN_RE, block, 0)
    if not m_doc:
        return block
    m_contents = _find(CONTENTS_FIRST_OPEN_RE, block, m_doc.end())
    if not m_contents:
        return block
    m_children = _find(CHILDREN_OPEN_RE, block, m_contents.end())
    if not m_children:
        return block

    # Drop stray <a id="..."></a> between doc-object open and contents-first open.
    pre = block[m_doc.end() : m_contents.start()]
    pre = CLASS_TOC_ANCHOR_RE.sub("", pre)

    # Up through the end of the opening contents-first tag
    prefix = block[: m_doc.end()] + pre + block[m_contents.start() : m_contents.end()]

    # Strip leading <p> blocks immediately inside contents-first, up to doc-children
    mid = block[m_contents.end() : m_children.start()]
    while True:
        m_adm = LEADING_ADMONITION_RE.match(mid)
        if m_adm:
            mid = mid[m_adm.end() :]
            continue

        m_h = LEADING_HIGHLIGHT_RE.match(mid)
        if m_h:
            mid = mid[m_h.end() :]
            continue

        m_p = LEADING_P_RE.match(mid)
        if m_p:
            mid = mid[m_p.end() :]
            continue

        break

    # Now strip leading <p> blocks at start of doc-children region
    after_children = block[m_children.start() :]
    while True:
        m_adm = LEADING_ADMONITION_RE.match(after_children)
        if m_adm:
            after_children = after_children[m_adm.end() :]
            continue

        m_h = LEADING_HIGHLIGHT_RE.match(after_children)
        if m_h:
            after_children = after_children[m_h.end() :]
            continue

        m_p = LEADING_P_RE.match(after_children)
        if m_p:
            after_children = after_children[m_p.end() :]
            continue

        break

    return prefix + mid + after_children


def _strip_specific_toc_entries(html: str, anchors_to_remove: set[str]) -> str:
    if not anchors_to_remove:
        return html

    def repl(m: re.Match) -> str:
        anchor = m.group("anchor")
        return "" if anchor in anchors_to_remove else m.group(0)

    return TOC_LI_RE.sub(repl, html)


def _clean_marked_blocks(output: str, mark_re: re.Pattern) -> str:
    marks = list(mark_re.finditer(output))
    if not marks:
        return output

    parts: list[str] = []
    last = 0
    for i, mk in enumerate(marks):
        start = mk.start()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(output)

        parts.append(output[last:start])
        block = output[start:end]
        parts.append(_strip_intro_from_block(block))
        last = end

    parts.append(output[last:])
    return "".join(parts)


def on_post_page(output: str, page, config) -> str:
    # 1) Gather TOC anchors to remove for this page; remove marker from output.
    anchors_to_remove: set[str] = set()
    m = TOC_REMOVE_RE.search(output)
    if m:
        anchors_to_remove = set(m.group(1).split())
        output = TOC_REMOVE_RE.sub("", output)

    # 2) Clean marked blocks
    output = _clean_marked_blocks(output, MARK_RE)
    output = _clean_marked_blocks(output, MODULE_MEMBERS_MARK_RE)

    # 3) Remove derived/base class TOC entries by exact anchor match
    output = _strip_specific_toc_entries(output, anchors_to_remove)

    # 4) Rewrite rendered HTML links: href="api:Target" -> href="...resolved..."
    cfg_dir = Path(config["config_file_path"]).resolve().parent
    inv_path = cfg_dir / "docs" / "_api_inventory.json"
    if not inv_path.exists():
        raise RuntimeError(f"API inventory not found at {inv_path} (expected during API build).")

    inv = ApiInventory.load(inv_path)
    src = getattr(page.file, "src_path", "") if hasattr(page, "file") else ""

    def _repl_api_a(m: re.Match) -> str:
        target = m.group("target")
        try:
            rel = inv.resolve(target)
        except KeyError as e:
            raise RuntimeError(f"{src}: {e}") from None

        url = ("/reference" + rel) if rel.startswith("/") else ("/reference/" + rel)

        body = m.group("body") or ""
        if body.strip() == "":
            # Fill empty link text with a sensible default
            try:
                fqn = inv.resolve_fqn(target)
            except Exception:
                fqn = target
            name = fqn.split(".")[-1]
            k = (inv.kinds.get(fqn) or "").lower()
            if k in {"function", "method"} and not name.endswith("()"):
                name = name + "()"
            body = f"<code>{name}</code>"

        return f'<a{m.group("attrs")} href="{url}"{m.group("attrs2")}>{body}</a>'

    output = API_A_TAG_RE.sub(_repl_api_a, output)

    # 5) Rewrite Pandoc-style citations in docstring HTML into links to the global bibliography page.
    #    Example: [@tomita_lowdistance_2014] -> [<a href="/reference/bib/#fn:tomita_lowdistance_2014">tomita_lowdistance_2014</a>]
    def _repl_cite(m: re.Match) -> str:
        keys_raw = m.group("keys")
        # split on ';' (pandoc allows [@a; @b]) and normalize
        keys = []
        for part in keys_raw.split(";"):
            part = part.strip()
            if part.startswith("@"):
                part = part[1:]
            if not part:
                continue
            keys.append(part)

        if not keys:
            return m.group(0)

        links = []
        for k in keys:
            href = f"/reference/bib/#fn:{k}"
            links.append(f'<a class="citation" href="{href}">{k}</a>')

        # keep bracketed formatting
        return "[" + "; ".join(links) + "]"

    output = CITE_BRACKET_RE.sub(_repl_cite, output)

    return output