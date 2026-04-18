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
  - Italicize inherited-method entries marked by
    ``<!-- API_INHERITED_HEADING <anchor> -->``.

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

from docs_scripts.api_inventory import ApiInventory, external_api_url

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

# Markers for inherited stub headings emitted by gen_ref_pages.py
INHERITED_MARK_RE = re.compile(r"<!--\s*API_INHERITED_HEADING\s+([^\s]+)\s*-->")

CONSTRUCTOR_HEADING_RE = re.compile(
    r"<!--\s*API_CONSTRUCTOR_HEADING\s+([^\s]+)\s+([^\s]+)\s*-->",
    re.IGNORECASE,
)

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

RIGHT_TOC_OPEN = '<nav class="md-nav md-nav--secondary" aria-label="Table of contents">'

TOC_LINK_TEXT_RE = re.compile(
    r'(<a[^>]*href="#(?P<anchor>[^"]+)"[^>]*>\s*<span class="md-ellipsis">)\s*.*?\s*(</span>\s*</a>)',
    re.IGNORECASE | re.DOTALL,
)

API_A_TAG_RE = re.compile(
    r'<a(?P<pre>[^>]*?)\s+href=(?P<q>["\'])api:(?P<target>[^"\'>\s]+)(?P=q)(?P<post>[^>]*)>(?P<body>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

# Pandoc-style citations in rendered HTML text (from docstrings), e.g. [@key] or [@k1; @k2]
CITE_BRACKET_RE = re.compile(r"\[@(?P<keys>[^\]]+)\]")


def _find(pat: re.Pattern, s: str, start: int = 0) -> re.Match | None:
    return pat.search(s, start)


def _strip_intro_from_block(block: str) -> str:
    """
    Clean a single marked block (class or module):

    - Remove stray <a id="..."></a> tags that can sit between the outer
      mkdocstrings container and the "doc-contents first" region.
    - Strip leading paragraphs, admonitions, and highlight blocks inside the
      "doc-contents first" region.
    - Strip leading paragraphs, admonitions, and highlight blocks at the start
      of the "doc-children" region.
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

    # Drop stray <a id="..."></a> tags between doc-object open and contents-first open.
    pre = block[m_doc.end() : m_contents.start()]
    pre = CLASS_TOC_ANCHOR_RE.sub("", pre)

    # Prefix through end of opening "contents-first" tag.
    prefix = block[: m_doc.end()] + pre + block[m_contents.start() : m_contents.end()]

    # Region between contents-first open and children open: strip leading intro material.
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

    # Children region: keep the <div class="doc doc-children"...> open tag,
    # then strip leading material immediately inside it.
    child_open = block[m_children.start() : m_children.end()]
    child_body = block[m_children.end() :]

    while True:
        m_adm = LEADING_ADMONITION_RE.match(child_body)
        if m_adm:
            child_body = child_body[m_adm.end() :]
            continue

        m_h = LEADING_HIGHLIGHT_RE.match(child_body)
        if m_h:
            child_body = child_body[m_h.end() :]
            continue

        m_p = LEADING_P_RE.match(child_body)
        if m_p:
            child_body = child_body[m_p.end() :]
            continue

        break

    return prefix + mid + child_open + child_body


def _strip_specific_toc_entries(html: str, anchors_to_remove: set[str]) -> str:
    """
    Remove exact-anchor entries from the right-hand TOC HTML fragment.
    """
    if not anchors_to_remove:
        return html

    def repl(m: re.Match) -> str:
        anchor = m.group("anchor")
        return "" if anchor in anchors_to_remove else m.group(0)

    return TOC_LI_RE.sub(repl, html)


def _clean_marked_blocks(output: str, mark_re: re.Pattern) -> str:
    """
    Apply `_strip_intro_from_block` to each region starting at a matching marker
    and ending at the next such marker or end of page output.
    """
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

def _rewrite_constructor_headings(html: str) -> str:
    """
    Rewrite constructor signature names from ``__init__`` to ``ClassName`` while
    preserving the mkdocstrings-generated heading and anchor.

    The generator emits:
      <!-- API_CONSTRUCTOR_HEADING <anchor_id> <ClassName> -->

    immediately before the mkdocstrings block for a declared constructor.
    """
    out = html

    while True:
        m = CONSTRUCTOR_HEADING_RE.search(out)
        if not m:
            break

        anchor_id = m.group(1)
        cls_name = m.group(2)

        sig_pat = re.compile(
            rf'(<h2 id="{re.escape(anchor_id)}" class="doc doc-heading">.*?</h2>\s*'
            rf'<div class="doc-signature highlight"><pre><span></span><code><span class="nf">)'
            rf'__init__'
            rf'(</span>)',
            re.IGNORECASE | re.DOTALL,
        )

        out = out.replace(m.group(0), "", 1)
        out = sig_pat.sub(
            lambda m2: m2.group(1) + cls_name + m2.group(2),
            out,
            count=1,
        )

    return out


def _italicize_inherited_in_right_toc(html: str, inherited_anchors: set[str]) -> str:
    """
    Wrap labels for inherited-method entries in the right-hand TOC only, so CSS
    can style them distinctly.
    """
    if not inherited_anchors:
        return html

    m = re.search(
        r'(<div class="md-sidebar md-sidebar--secondary"[^>]*>.*?</div>\s*</div>)',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return html

    frag = m.group(1)

    def repl(m2: re.Match) -> str:
        anchor = m2.group("anchor")
        if anchor not in inherited_anchors:
            return m2.group(0)

        label = anchor.rsplit(".", 1)[-1]
        return (
            m2.group(1)
            + f'<span class="api-inherited-toc">{label}</span>'
            + m2.group(3)
        )

    frag2 = TOC_LINK_TEXT_RE.sub(repl, frag)
    return html[: m.start(1)] + frag2 + html[m.end(1) :]


def on_post_page(output: str, page, config) -> str:
    # 1) Gather TOC anchors to remove for this page, then remove the markers.
    anchors_to_remove: set[str] = set()
    for m in TOC_REMOVE_RE.finditer(output):
        anchors_to_remove |= set((m.group(1) or "").split())
    output = TOC_REMOVE_RE.sub("", output)

    # 2) Clean marked mkdocstrings blocks.
    output = _clean_marked_blocks(output, MARK_RE)
    output = _clean_marked_blocks(output, MODULE_MEMBERS_MARK_RE)

    # Rewrite declared constructor headings from __init__ to ClassName().
    output = _rewrite_constructor_headings(output)

    # 3) Remove derived/base class TOC entries by exact anchor match.
    output = _strip_specific_toc_entries(output, anchors_to_remove)

    # 4) Rewrite rendered HTML links: href="api:Target" -> href="...resolved..."
    inv_path = Path(config["docs_dir"]) / "_api_inventory.json"
    if not inv_path.exists():
        raise RuntimeError(f"API inventory not found at {inv_path} (expected during API build).")

    inv = ApiInventory.load(inv_path)
    src = getattr(page.file, "src_path", "") if hasattr(page, "file") else ""

    def _repl_api_a(m: re.Match) -> str:
        target = m.group("target").strip()

        # Resolve URL.
        try:
            rel = inv.resolve(target)
        except KeyError:
            rel = None

        if rel is not None:
            url = ("/reference" + rel) if rel.startswith("/") else ("/reference/" + rel)
        else:
            # External targets: use known mapping when available; otherwise keep
            # the original api: href and mark it for styling.
            url = external_api_url(target)

        # Normalize link body:
        # - fill if empty
        # - wrap plain text in <code>...</code>
        # - append () for methods/functions when known
        body = (m.group("body") or "").strip()

        if not body:
            body = target.split(".")[-1]

        # Only wrap if it appears to be plain text (no nested tags).
        if "<" not in body and ">" not in body:
            name = body

            try:
                fqn = inv.resolve_fqn(target)
                kind = (inv.kinds.get(fqn) or "").lower()
                if kind in {"function", "method"} and not name.endswith("()"):
                    name = name + "()"
            except Exception:
                pass

            body = f"<code>{name}</code>"

        if url is None:
            return (
                f'<a{m.group("pre")} href="{target}" class="api-unresolved-external"{m.group("post")}>'
                f"{body}</a>"
            )

        return f'<a{m.group("pre")} href="{url}"{m.group("post")}>{body}</a>'

    output = API_A_TAG_RE.sub(_repl_api_a, output)

    # 5) Italicize inherited methods in the right-hand TOC.
    inherited_anchors = set(INHERITED_MARK_RE.findall(output))
    output = INHERITED_MARK_RE.sub("", output)
    output = _italicize_inherited_in_right_toc(output, inherited_anchors)

    # 6) Rewrite Pandoc-style citations in rendered docstring HTML into links
    #    to the global bibliography page.
    def _repl_cite(m: re.Match) -> str:
        keys_raw = m.group("keys")
        keys: list[str] = []

        # Pandoc allows [@a; @b]; split on ';' and normalize.
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

        # Keep bracketed formatting.
        return "[" + "; ".join(links) + "]"

    output = CITE_BRACKET_RE.sub(_repl_cite, output)

    return output