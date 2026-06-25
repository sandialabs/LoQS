#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-processing hooks for the generated LoQS API reference site.

Goals
-----
- Clean up mkdocstrings-rendered HTML in marked generated blocks.
- Rewrite rendered `api:` links and selected type spans using the shared inventory.
- Rewrite inherited-doc renders so the visible heading/TOC/anchor all point to the
  derived-class identity even though the doc body was rendered from a base method.
- Prune redundant right-hand TOC entries and style inherited-only stubs distinctly.
- Rewrite constructor labels and bibliography citations after docstring rendering.

Design
------
The API site now has one primary doc-rendering pathway:
- normal docs: mkdocstrings render -> HTML postprocessing
- inherited documented overrides: also mkdocstrings render -> HTML postprocessing

That keeps link/citation/signature handling much closer between inherited and
non-inherited method docs, and centralizes the identity remapping in this file.
"""

from __future__ import annotations

import html as _html
import inspect
import re
from pathlib import Path

from docs_scripts.api_inventory import ApiInventory, resolve_api_target_url


MARK_RE = re.compile(
    r"<!--\s*API_METHOD\s+owner=([^\s]+)\s+member=([^\s]+)\s*-->",
    re.IGNORECASE,
)

MODULE_MEMBERS_MARK_RE = re.compile(
    r"<!--\s*API_MODULE_MEMBERS\s+owner=([^\s]+)\s*-->",
    re.IGNORECASE,
)

TOC_REMOVE_RE = re.compile(r"<!--\s*API_TOC_REMOVE\s+([^>]+?)\s*-->", re.IGNORECASE)
INHERITED_MARK_RE = re.compile(r"<!--\s*API_INHERITED_HEADING\s+([^\s]+)\s*-->")

INHERITED_RENDER_MARK_RE = re.compile(
    r'<!--\s*API_INHERITED_RENDER\s+derived="(?P<derived>[^"]+)"\s+base="(?P<base>[^"]+)"\s+owner="(?P<owner>[^"]+)"\s*-->',
    re.IGNORECASE,
)

DOC_CLASS_OPEN_RE = re.compile(r'<div class="doc doc-object doc-class"[^>]*>', re.IGNORECASE)
DOC_MODULE_OPEN_RE = re.compile(r'<div class="doc doc-object doc-module"[^>]*>', re.IGNORECASE)
DOC_FUNCTION_OPEN_RE = re.compile(r'<div class="doc doc-object doc-function"[^>]*>', re.IGNORECASE)

CONTENTS_FIRST_OPEN_RE = re.compile(r'<div class="doc doc-contents first"[^>]*>', re.IGNORECASE)
CHILDREN_OPEN_RE = re.compile(r'<div class="doc doc-children"[^>]*>', re.IGNORECASE)

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

TOC_LI_RE = re.compile(
    r'<li class="md-nav__item">\s*'
    r'<a href="#(?P<anchor>[^"]+)" class="md-nav__link">.*?</a>\s*'
    r"</li>",
    re.DOTALL | re.IGNORECASE,
)

TOC_LINK_TEXT_RE = re.compile(
    r'(<a[^>]*href="#(?P<anchor>[^"]+)"[^>]*>\s*<span class="md-ellipsis">)\s*.*?\s*(</span>\s*</a>)',
    re.IGNORECASE | re.DOTALL,
)

CONSTRUCTOR_HEADING_RE = re.compile(
    r'(<h2 id="(?P<anchor>[^"]+__init__)" class="doc doc-heading">.*?<span class="doc doc-object-name doc-function-name">)__init__(</span>)',
    re.IGNORECASE | re.DOTALL,
)

CONSTRUCTOR_SIG_RE = re.compile(
    r'(<h2 id="(?P<anchor>[^"]+__init__)" class="doc doc-heading">.*?</h2>\s*'
    r'<div class="doc-signature highlight"><pre><span></span><code><span class="nf">)__init__(</span>)',
    re.IGNORECASE | re.DOTALL,
)

API_A_TAG_RE = re.compile(
    r'<a(?P<pre>[^>]*?)\s+href=(?P<q>["\'])api:(?P<target>[^"\'>\s]+)(?P=q)(?P<post>[^>]*)>(?P<body>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

TYPED_SPAN_RE = re.compile(
    r'<span\s+title="(?P<target>loqs(?:\.[A-Za-z_][A-Za-z0-9_]*)+)"\s*>(?P<label>[^<]+)</span>',
    re.IGNORECASE,
)

CITE_BRACKET_RE = re.compile(r"\[@(?P<keys>[^\]]+)\]")


def _find(pat: re.Pattern, s: str, start: int = 0) -> re.Match | None:
    return pat.search(s, start)


def _load_inventory(config) -> ApiInventory:
    inv_path = Path(config["docs_dir"]) / "_api_inventory.json"
    if not inv_path.exists():
        raise RuntimeError(f"API inventory not found at {inv_path} (expected during API build).")
    return ApiInventory.load(inv_path)


def _strip_intro_from_block(block: str) -> str:
    m_doc = _find(DOC_CLASS_OPEN_RE, block, 0) or _find(DOC_MODULE_OPEN_RE, block, 0) or _find(DOC_FUNCTION_OPEN_RE, block, 0)
    if not m_doc:
        return block

    m_contents = _find(CONTENTS_FIRST_OPEN_RE, block, m_doc.end())
    if not m_contents:
        return block

    m_children = _find(CHILDREN_OPEN_RE, block, m_contents.end())
    if not m_children:
        return block

    pre = block[m_doc.end() : m_contents.start()]
    pre = CLASS_TOC_ANCHOR_RE.sub("", pre)

    prefix = block[: m_doc.end()] + pre + block[m_contents.start() : m_contents.end()]

    mid = block[m_contents.end() : m_children.start()]
    while True:
        for pat in (LEADING_ADMONITION_RE, LEADING_HIGHLIGHT_RE, LEADING_P_RE):
            m = pat.match(mid)
            if m:
                mid = mid[m.end() :]
                break
        else:
            break

    child_open = block[m_children.start() : m_children.end()]
    child_body = block[m_children.end() :]

    while True:
        for pat in (LEADING_ADMONITION_RE, LEADING_HIGHLIGHT_RE, LEADING_P_RE):
            m = pat.match(child_body)
            if m:
                child_body = child_body[m.end() :]
                break
        else:
            break

    return prefix + mid + child_open + child_body


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
        parts.append(_strip_intro_from_block(output[start:end]))
        last = end

    parts.append(output[last:])
    return "".join(parts)


def _strip_specific_toc_entries(html: str, anchors_to_remove: set[str]) -> str:
    if not anchors_to_remove:
        return html

    def repl(m: re.Match) -> str:
        return "" if m.group("anchor") in anchors_to_remove else m.group(0)

    return TOC_LI_RE.sub(repl, html)


def _strip_dead_fnref_links(html: str) -> str:
    """
    Remove local bibliography backlink anchors like `href="#fnref:..."` when the
    target anchor does not exist on the same page.

    These are typically plugin-generated backlink artifacts on bibliography pages
    and cause noisy MkDocs warnings.
    """
    existing_ids = set(re.findall(r'id="([^"]+)"', html, flags=re.IGNORECASE))

    def repl(m: re.Match) -> str:
        target = m.group("target")
        body = m.group("body")
        if target in existing_ids:
            return m.group(0)
        return body

    return re.sub(
        r'<a[^>]*href="#(?P<target>fnref:[^"]+)"[^>]*>(?P<body>.*?)</a>',
        repl,
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _italicize_inherited_in_right_toc(html: str, inherited_anchors: set[str]) -> str:
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
        return m2.group(1) + f'<span class="api-inherited-toc">{label}</span>' + m2.group(3)

    frag2 = TOC_LINK_TEXT_RE.sub(repl, frag)
    return html[: m.start(1)] + frag2 + html[m.end(1) :]


def _constructor_class_name_from_anchor(anchor: str) -> str:
    parts = anchor.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return "__init__"


def _rewrite_constructor_labels(html: str) -> str:
    def repl_heading(m: re.Match) -> str:
        cls_name = _constructor_class_name_from_anchor(m.group("anchor"))
        return m.group(1) + cls_name + m.group(3)

    out = CONSTRUCTOR_HEADING_RE.sub(repl_heading, html)

    def repl_sig(m: re.Match) -> str:
        cls_name = _constructor_class_name_from_anchor(m.group("anchor"))
        return m.group(1) + cls_name + m.group(3)

    out = CONSTRUCTOR_SIG_RE.sub(repl_sig, out)

    out = re.sub(
        r'(<a[^>]*href="#(?P<anchor>[^"]+__init__)"[^>]*>\s*<span class="md-ellipsis">)\s*__init__\s*(</span>)',
        lambda m: m.group(1) + f"{_constructor_class_name_from_anchor(m.group('anchor'))}" + m.group(3),
        out,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return out


def _rewrite_rendered_api_links(output: str, inv: ApiInventory, *, src: str) -> str:
    def repl(m: re.Match) -> str:
        target = m.group("target").strip()
        url = resolve_api_target_url(inv, target, src=src, prefix="/reference", allow_external=True)

        body = (m.group("body") or "").strip()
        if not body:
            base = target.split(".")[-1]
            if base == "__init__" and "." in target:
                body = target.split(".")[-2]
            else:
                body = base

        # Detect whether this <a> is already inside an outer <code>...</code> wrapper.
        before = output[max(0, m.start() - 64):m.start()]
        after = output[m.end():min(len(output), m.end() + 64)]
        inside_outer_code = bool(re.search(r"<code>[^<>]*?$", before, flags=re.IGNORECASE)) and bool(
            re.search(r"^[^<>]*?</code>", after, flags=re.IGNORECASE)
        )

        if "<" not in body and ">" not in body:
            name = body
            try:
                fqn = inv.resolve_fqn(target)
                kind = (inv.kinds.get(fqn) or "").lower()
                if kind in {"function", "method"} and not name.endswith("()"):
                    name = name + "()"
            except Exception:
                pass

            if not inside_outer_code:
                body = f"<code>{name}</code>"
            else:
                body = name

        if url is None:
            return (
                f'<a{m.group("pre")} href="api:{target}" class="api-unresolved-external"{m.group("post")}>'
                f"{body}</a>"
            )

        return f'<a{m.group("pre")} href="{url}"{m.group("post")}>{body}</a>'

    return API_A_TAG_RE.sub(repl, output)


def _rewrite_typed_spans(output: str, inv: ApiInventory) -> str:
    def repl(m: re.Match) -> str:
        target = m.group("target").strip()
        label = (m.group("label") or "").strip() or target.split(".")[-1]
        try:
            url = inv.resolve_mounted_url(target, prefix="/reference")
        except KeyError:
            return m.group(0)
        return f'<a href="{url}">{label}</a>'

    return TYPED_SPAN_RE.sub(repl, output)


def _rewrite_citations(output: str) -> str:
    def repl(m: re.Match) -> str:
        keys_raw = m.group("keys")
        keys: list[str] = []

        for part in keys_raw.split(";"):
            part = part.strip()
            if part.startswith("@"):
                part = part[1:]
            if part:
                keys.append(part)

        if not keys:
            return m.group(0)

        links = [f'<a class="citation" href="/reference/bib/#fn:{k}">{k}</a>' for k in keys]
        return "[" + "; ".join(links) + "]"

    return CITE_BRACKET_RE.sub(repl, output)


def _extract_balanced_div(html: str, start: int) -> tuple[str, int] | None:
    """
    Return `(block_html, end_index)` for the balanced <div>...</div> block whose
    opening tag starts at `start`, or None if no balanced block is found.
    """
    m = re.match(r"<div\b[^>]*>", html[start:], flags=re.IGNORECASE)
    if not m:
        return None

    i = start
    end = start + m.end()
    depth = 1
    pos = end

    tag_re = re.compile(r"</?div\b[^>]*>", flags=re.IGNORECASE)
    while depth > 0:
        m2 = tag_re.search(html, pos)
        if not m2:
            return None

        tag = m2.group(0)
        if tag.startswith("</"):
            depth -= 1
        else:
            depth += 1
        pos = m2.end()

    return html[start:pos], pos


def _rewrite_inherited_return_types(
    html: str,
    *,
    derived: str,
    owner: str,
    base_signature_html: str = "",
) -> str:
    """
    In inherited method doc bodies, rewrite Returns-table type cells that point to
    the base owner class so they instead point to the derived method's actual return
    annotation, but only when the rendered base signature return annotation is a
    TypeVar whose bound equals the base owner class.

    This uses the rendered base signature HTML as the primary source for the base
    return-annotation name, which is more robust than relying only on runtime
    annotation objects after signature/doc rendering.
    """
    try:
        parts = derived.split(".")
        if len(parts) < 3:
            return html

        mod_ident = ".".join(parts[:-2])
        cls_name = parts[-2]
        meth_name = parts[-1]

        owner_parts = owner.split(".")
        if len(owner_parts) < 2:
            return html
        owner_mod_ident = ".".join(owner_parts[:-1])
        owner_cls_name = owner_parts[-1]

        mod = __import__(mod_ident, fromlist=[cls_name])
        cls_obj = getattr(mod, cls_name, None)
        if cls_obj is None:
            return html

        owner_mod = __import__(owner_mod_ident, fromlist=[owner_cls_name])
        owner_cls_obj = getattr(owner_mod, owner_cls_name, None)
        if owner_cls_obj is None:
            return html

        # Extract rendered base signature return name, e.g. "T"
        arrow_idx = base_signature_html.find("-&gt;")
        if arrow_idx < 0:
            arrow_idx = base_signature_html.find("&gt;")
        if arrow_idx < 0:
            return html

        after_arrow = base_signature_html[arrow_idx:]

        m_ret = re.search(
            r'<span[^>]*>(?P<ret>[A-Za-z_][A-Za-z0-9_]*)</span>',
            after_arrow,
            flags=re.IGNORECASE,
        )
        if not m_ret:
            return html

        base_ret_name = m_ret.group("ret").strip()
        if not base_ret_name:
            return html

        # Resolve the base signature return name as a TypeVar on the owner module.        
        tv_obj = getattr(owner_mod, base_ret_name, None)
        if tv_obj is None:
            return html

        # The runtime TypeVar object may not retain a usable __bound__ here.
        # We therefore use a narrower semantic guard:
        #   - the rendered base signature returns a module-level TypeVar name
        #   - the rendered Returns-table cell points to the exact base owner class
        # That combination is the self-type case we want to rewrite.

        raw = cls_obj.__dict__.get(meth_name)
        if isinstance(raw, staticmethod):
            derived_meth = raw.__func__
        elif isinstance(raw, classmethod):
            derived_meth = raw.__func__
        else:
            derived_meth = getattr(cls_obj, meth_name, None)

        if derived_meth is None:
            return html

        try:
            derived_sig = inspect.signature(derived_meth)
        except (TypeError, ValueError):
            return html

        ann = derived_sig.return_annotation
        if ann is inspect.Signature.empty or ann is None:
            return html

        derived_target: str | None = None
        derived_label: str | None = None

        if isinstance(ann, str):
            ann_text = ann.strip()
            if ann_text and ann_text != "None":
                obj = getattr(mod, ann_text, None)
                if obj is not None:
                    obj_mod = getattr(obj, "__module__", "") or ""
                    obj_qual = getattr(obj, "__qualname__", "") or getattr(obj, "__name__", "")
                    if obj_mod.startswith("loqs") and obj_qual:
                        derived_target = f"{obj_mod}.{obj_qual}"
                        derived_label = obj_qual.split(".")[-1]
                elif ann_text.startswith("loqs."):
                    derived_target = ann_text
                    derived_label = ann_text.split(".")[-1]
            if not derived_target or not derived_label:
                return html
        else:
            mod_name = getattr(ann, "__module__", "") or ""
            qual_name = getattr(ann, "__qualname__", "") or getattr(ann, "__name__", "")
            if not qual_name:
                return html
            if mod_name == "builtins" and qual_name == "NoneType":
                return html
            if not mod_name.startswith("loqs"):
                return html

            derived_target = f"{mod_name}.{qual_name}"
            derived_label = qual_name.split(".")[-1]

        owner_href_suffix = f"#{owner}"

        pattern = re.compile(
            rf'(<tr class="doc-section-item">.*?<td>\s*<code>\s*)'
            rf'<a(?P<pre>[^>]*?)href="[^"]*{re.escape(owner_href_suffix)}"(?P<post>[^>]*)>'
            rf'(?P<label>.*?)'
            rf'</a>'
            rf'(\s*</code>\s*</td>)',
            flags=re.IGNORECASE | re.DOTALL,
        )

        def repl(m: re.Match) -> str:
            return (
                f'{m.group(1)}'
                f'<a href="api:{derived_target}">{derived_label}</a>'
                f'{m.group(5)}'
            )

        return pattern.sub(repl, html)

    except Exception:
        return html


def _rewrite_inherited_render_blocks(output: str) -> str:
    """
    For each generated inherited-doc render marker, strip the enclosing base-class
    framing and the base method heading/signature from the subsequent mkdocstrings
    render, keeping only the inherited method doc contents.

    Important assumption:
    - Generated class pages always use `<hr>` as the delimiter between method
      sections. This function relies on that structure and treats the next `<hr>`
      after an inherited-render marker as the hard boundary for that inherited
      render region.

    The derived method block remains the canonical heading/signature/anchor on the page.
    """
    marks = list(INHERITED_RENDER_MARK_RE.finditer(output))
    if not marks:
        return output

    parts: list[str] = []
    last = 0

    for mk in marks:
        start = mk.start()
        parts.append(output[last:start])

        # IMPORTANT: the generator always emits <hr> between functions, and this
        # hook relies on that invariant to bound the inherited-render region.
        hr = re.search(r"<hr\s*/?>", output[start:], flags=re.IGNORECASE)
        end = start + hr.start() if hr else len(output)

        block = output[start:end]
        derived = mk.group("derived")
        base = mk.group("base")
        owner = mk.group("owner")

        block = INHERITED_RENDER_MARK_RE.sub("", block, count=1)

        # Remove explicit anchors that would create duplicate autorefs targets.
        block = re.sub(rf'<a\s+id="{re.escape(owner)}"\s*></a>\s*', "", block, flags=re.IGNORECASE)
        block = re.sub(rf'<a\s+id="{re.escape(base)}"\s*></a>\s*', "", block, flags=re.IGNORECASE)

        # Also remove any rendered base-class heading anchor that mkdocstrings may have
        # emitted for the inherited owner class. This is the main source of duplicate
        # primary URLs reported by mkdocs-autorefs on derived pages.
        block = re.sub(
            rf'<h[1-6]\s+id="{re.escape(owner)}"[^>]*>.*?</h[1-6]>\s*',
            "",
            block,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Safer cleanup: remove any rendered base-method heading from the whole inherited
        # render region before subtree extraction, not just from the extracted function block.
        # This prevents duplicate method anchors from surviving if the HTML structure shifts.
        block = re.sub(
            rf'<h[1-6]\s+id="{re.escape(base)}"[^>]*>.*?</h[1-6]>\s*',
            "",
            block,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Find the first inherited function doc-object block.
        m_func = re.search(
            r'<div class="doc doc-object doc-function">',
            block,
            flags=re.IGNORECASE,
        )
        if not m_func:
            last = end
            continue

        extracted = _extract_balanced_div(block, m_func.start())
        if not extracted:
            last = end
            continue

        func_block, _ = extracted

        m_sig = re.search(
            r'<div class="doc-signature highlight"><pre>.*?</pre></div>',
            func_block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        base_signature_html = m_sig.group(0) if m_sig else ""

        # Remove the base method heading.
        func_block = re.sub(
            rf'<h[1-6]\s+id="{re.escape(base)}"\s+class="doc doc-heading">.*?</h[1-6]>\s*',
            "",
            func_block,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Remove the base method signature block.
        func_block = re.sub(
            r'<div class="doc-signature highlight"><pre>.*?</pre></div>\s*',
            "",
            func_block,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Extract the inner doc-contents div from the function block using balanced parsing.
        m_contents = re.search(
            r'<div class="doc doc-contents ">',
            func_block,
            flags=re.IGNORECASE,
        )
        if not m_contents:
            last = end
            continue

        extracted_contents = _extract_balanced_div(func_block, m_contents.start())
        if not extracted_contents:
            last = end
            continue

        contents_block, _ = extracted_contents

        # Strip the outer <div class="doc doc-contents "> ... </div>
        body = re.sub(
            r'^<div class="doc doc-contents ">\s*|\s*</div>\s*$',
            "",
            contents_block,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        body = _rewrite_inherited_return_types(
            body,
            derived=derived,
            owner=owner,
            base_signature_html=base_signature_html,
        )

        parts.append(body)
        last = end

    parts.append(output[last:])
    return "".join(parts)


def _extract_toc_remove_markers(output: str) -> tuple[str, set[str]]:
    anchors_to_remove: set[str] = set()
    for m in TOC_REMOVE_RE.finditer(output):
        anchors_to_remove |= set((m.group(1) or "").split())
    return TOC_REMOVE_RE.sub("", output), anchors_to_remove


def _extract_inherited_toc_remove_anchors(output: str) -> set[str]:
    """
    Collect base method anchors from inherited-render markers so the right-hand TOC
    does not show duplicate entries for the inherited base-method render.

    These markers are left in the HTML until after TOC pruning so we can reuse the
    normal anchor-removal pathway.
    """
    anchors: set[str] = set()
    for m in INHERITED_RENDER_MARK_RE.finditer(output):
        base = (m.group("base") or "").strip()
        if base:
            anchors.add(base)
    return anchors

def _rewrite_table_doc_parbreaks(html: str) -> str:
    """
    Replace generated table-doc paragraph break sentinels with HTML paragraph
    spacing after Markdown/HTML rendering has already happened.
    """
    return html.replace("@@API_DOC_PBREAK@@", "<br><br>")


def on_post_page(output: str, page, config) -> str:
    inv = _load_inventory(config)
    src = getattr(page.file, "src_path", "") if hasattr(page, "file") else ""

    output, anchors_to_remove = _extract_toc_remove_markers(output)
    anchors_to_remove |= _extract_inherited_toc_remove_anchors(output)

    output = _clean_marked_blocks(output, MARK_RE)
    output = _clean_marked_blocks(output, MODULE_MEMBERS_MARK_RE)

    output = _rewrite_inherited_render_blocks(output)
    output = _strip_specific_toc_entries(output, anchors_to_remove)

    output = _rewrite_rendered_api_links(output, inv, src=src)
    output = _rewrite_typed_spans(output, inv)

    inherited_anchors = set(INHERITED_MARK_RE.findall(output))
    output = INHERITED_MARK_RE.sub("", output)
    output = _italicize_inherited_in_right_toc(output, inherited_anchors)

    output = _rewrite_constructor_labels(output)
    output = _rewrite_citations(output)
    output = _strip_dead_fnref_links(output)
    output = _rewrite_table_doc_parbreaks(output)

    return output