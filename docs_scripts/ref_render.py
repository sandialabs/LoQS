from __future__ import annotations

"""
Markdown rendering helpers for generated API reference pages.

Goals
-----
- Emit predictable MkDocs/MkDocStrings-compatible page structures for modules and classes.
- Keep page-writing logic separate from source/introspection logic.
- Reuse shared rendering helpers for mkdocstrings blocks and member tables.
- Ensure generated content emits consistent `api:` links so later hook passes can
  resolve/rewrite them uniformly.

Key capabilities
----------------
- Module landing pages with intro, class list, attribute table, and function blocks.
- Class pages with intro, attribute table, declared methods, and inherited method stubs.
- Inherited documented overrides rendered through mkdocstrings on the base owner while
  preserving derived-class headings/anchors using generator markers that `api_hooks.py`
  rewrites after render.
"""

import re
from pathlib import Path

import mkdocs_gen_files

from docs_scripts.api_inventory import external_api_url
from docs_scripts.ref_introspect import (
    K_METHOD,
    K_PROPERTY,
    K_TYPE_VARIABLE,
    K_VARIABLE,
    classvar_inner,
)


def render_inline_md(
    text: str,
    link_names: set[str] | None = None,
    *,
    local_targets: dict[str, str] | None = None,
    html_code: bool = True,
    prose: bool = False,
) -> str:
    """
    Render inline documentation content for generated table cells.

    Modes
    -----
    prose=False:
        Type/value-expression mode. Autolinks recognized identifiers and emits a
        single outer `<code>...</code>` block for HTML output.

    prose=True:
        Prose/doc-cell mode. Does not autolink bare identifiers. Instead it:
        - collapses ordinary newlines to spaces
        - preserves blank lines as `<br><br>`
        - converts Markdown api-links `[text](api:Target)` to raw HTML anchors
        - converts inline code spans `` `x` `` to `<code>x</code>`
        - leaves math delimiters untouched as literal text

    Internal LoQS identifiers are linked using:
    - exact local fully-qualified targets when available via `local_targets`
    - otherwise short progressive `api:` targets for locally known names
    - otherwise short progressive `api:` targets for `loqs.*` tokens

    Known external fully-qualified identifiers are linked via `external_api_url()`.
    """
    s = (text or "").strip()
    if not s:
        return ""

    local_targets = local_targets or {}
    link_names = link_names or set()

    if prose:
        doc = s.replace("|", "\\|").strip()
        if not doc:
            return ""

        paras = [" ".join(line.strip() for line in p.splitlines() if line.strip()) for p in doc.split("\n\n")]
        doc = "<br><br>".join(p for p in paras if p)

        doc = re.sub(
            r"`([^`]+)`",
            lambda m: f"<code>{m.group(1)}</code>",
            doc,
        )

        doc = re.sub(
            r"\[(?P<text>[^\]]+)\]\(\s*api:(?P<target>[^)\s]+)\s*\)",
            lambda m: f'<a href="api:{m.group("target")}">{m.group("text")}</a>',
            doc,
        )

        return doc

    def _html_link(target: str, label: str) -> str:
        return f'<a href="api:{target}">{label}</a>'

    def _html_external(url: str, label: str) -> str:
        return f'<a href="{url}">{label}</a>'

    # Special-case TypeVar(...) so we only link the bound target and do not let
    # the generic token pass incorrectly link the TypeVar name itself.
    if s.startswith("TypeVar(") and "bound=" in s:
        m = re.search(r"bound=(?P<quote>['\"]?)(?P<bound>[A-Za-z_][A-Za-z0-9_\.]*)(?P=quote)", s)
        if m:
            quote = m.group("quote") or ""
            bound = m.group("bound")
            bound_name = bound.split(".")[-1]

            target: str | None = None
            url: str | None = None

            if bound_name in local_targets:
                target = local_targets[bound_name]
            elif bound_name in link_names:
                target = bound_name
            elif bound.startswith("loqs."):
                target = bound
            else:
                url = external_api_url(bound)

            if html_code:
                if target is not None:
                    replacement = f"bound={quote}{_html_link(target, bound_name)}{quote}"
                elif url is not None:
                    replacement = f"bound={quote}{_html_external(url, bound_name)}{quote}"
                else:
                    replacement = m.group(0)

                out = s[: m.start()] + replacement + s[m.end() :]
                return f"<code>{out}</code>"

            if target is not None:
                replacement = f"bound={quote}[`{bound_name}`](api:{target}){quote}"
            elif url is not None:
                replacement = f"bound={quote}[`{bound_name}`]({url}){quote}"
            else:
                replacement = m.group(0)

            out = s[: m.start()] + replacement + s[m.end() :]
            return out

    names = sorted(link_names, key=len, reverse=True)
    if names:
        token_re = re.compile(
            r"\bloqs(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b|"
            r"\b(?:pygsti|stim)(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b|"
            + "|".join(rf"\b{re.escape(n)}\b" for n in names)
        )
    else:
        token_re = re.compile(
            r"\bloqs(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b"
            r"|\b(?:pygsti|stim)(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b"
        )

    parts: list[str] = []
    last = 0
    found_link = False

    for m in token_re.finditer(s):
        prefix = s[last:m.start()]
        if prefix:
            parts.append(prefix)

        token = m.group(0)
        if token.startswith("loqs."):
            target = token.split(".")[-1]
            label = target
            if html_code:
                parts.append(_html_link(target, label))
            else:
                parts.append(f"[`{label}`](api:{target})")
            found_link = True
        else:
            url = external_api_url(token)
            if url is not None:
                label = token.split(".")[-1]
                if html_code:
                    parts.append(_html_external(url, label))
                else:
                    parts.append(f"[`{label}`]({url})")
                found_link = True
            elif token in local_targets:
                target = local_targets[token]
                label = token
                if html_code:
                    parts.append(_html_link(target, label))
                else:
                    parts.append(f"[`{label}`](api:{target})")
                found_link = True
            else:
                target = token
                label = token
                if html_code:
                    parts.append(_html_link(target, label))
                else:
                    parts.append(f"[`{label}`](api:{target})")
                found_link = True

        last = m.end()

    suffix = s[last:]
    if suffix:
        parts.append(suffix)

    if parts:
        out = "".join(parts)
        if html_code:
            return f"<code>{out}</code>"
        return out if found_link else f"`{s}`"

    return f"<code>{s}</code>" if html_code else f"`{s}`"


def write_mkdocstrings_block(
    f,
    ident: str,
    *,
    members: list[str] | bool,
    inherited_members: bool = False,
    heading_level: int | None = None,
) -> None:
    """
    Emit a generic mkdocstrings block.
    """
    f.write(f"::: {ident}\n")
    f.write("    options:\n")
    if heading_level is not None:
        f.write(f"      heading_level: {heading_level}\n")

    if members is False:
        f.write("      members: false\n")
    else:
        f.write("      members:\n")
        for m in members:
            f.write(f"        - {m}\n")

    f.write(f"      inherited_members: {'true' if inherited_members else 'false'}\n")
    f.write("\n")


def write_class_members_table(
    f,
    rows: list[dict],
    *,
    derived_ident: str,
    class_anchor_prefix: str,
    inv_objects: dict[str, str],
    inv_kinds: dict[str, str],
    page_url: str,
    link_names: set[str] | None = None,
    module_local_targets: dict[str, str] | None = None,
) -> None:
    if not rows:
        return

    local_targets = dict(module_local_targets or {})
    local_targets.update(
        {
            r["name"]: f"{class_anchor_prefix}.{r['name']}"
            for r in rows
            if isinstance(r.get("name"), str) and r.get("name")
        }
    )

    f.write("| Name | Type | Value | Doc |\n")
    f.write("|---|---|---|---|\n")
    for r in rows:
        nm = r["name"]
        anchor_id = f"{class_anchor_prefix}.{nm}"
        inv_objects[anchor_id] = f"{page_url}#{anchor_id}"

        name_cell = f'<a id="{anchor_id}"></a>`{nm}`'
        if (r.get("owner") or "") != derived_ident:
            name_cell += "<br><em>(inherited)</em>"

        typ_raw = (r.get("type") or "")
        typ = render_inline_md(
            classvar_inner(typ_raw).replace("\n", " ").replace("|", "\\|"),
            link_names=link_names,
            local_targets=local_targets,
        )

        val = r.get("value")
        if isinstance(val, str) and "property" in val:
            inv_kinds[anchor_id] = K_PROPERTY
        else:
            inv_kinds[anchor_id] = K_VARIABLE

        if val is None or str(val).strip() == "":
            val_s = "*unset*"
        else:
            val_s_raw = str(val).replace("\n", " ").strip()
            val_s = val_s_raw if "*" in val_s_raw else render_inline_md(
                val_s_raw.replace("|", "\\|"),
                link_names=link_names,
                local_targets=local_targets,
            )

        doc = render_inline_md(r.get("doc") or "", prose=True)

        f.write(f"| {name_cell} | {typ} | {val_s} | {doc} |\n")
    f.write("\n")


def write_class_intro(f, cls_ident: str) -> None:
    """
    Emit the class intro block, explicitly including `__init__`.
    """
    write_mkdocstrings_block(f, cls_ident, members=["__init__"], inherited_members=False)


def write_declared_method_block(
    f,
    *,
    cls_ident: str,
    member_name: str,
) -> None:
    """
    Emit a declared method block rendered from the derived class object.
    """
    f.write(f"<!-- API_METHOD owner={cls_ident} member={member_name} -->\n\n")
    write_mkdocstrings_block(f, cls_ident, members=[member_name], inherited_members=False)


def write_inherited_doc_render_block(
    f,
    *,
    derived_cls_ident: str,
    derived_member_name: str,
    base_owner_ident: str,
    base_member_name: str,
) -> None:
    """
    Emit a block that renders a base method via mkdocstrings, with enough marker
    information for api_hooks.py to strip the base class/method framing and keep
    only the inherited doc contents.
    """
    derived_anchor = f"{derived_cls_ident}.{derived_member_name}"
    base_anchor = f"{base_owner_ident}.{base_member_name}"

    f.write(
        f'<!-- API_INHERITED_RENDER derived="{derived_anchor}" base="{base_anchor}" owner="{base_owner_ident}" -->\n\n'
    )
    write_mkdocstrings_block(f, base_owner_ident, members=[base_member_name], inherited_members=False)


def write_inherited_method_stub(
    f,
    *,
    derived_cls_ident: str,
    method_name: str,
    method_kind: str,
    base_ident: str,
) -> None:
    """
    Emit a lightweight stub for inherited methods not declared on the derived class.
    """
    anchor_id = f"{derived_cls_ident}.{method_name}"

    kind_label = {
        "static": "static ",
        "class": "class ",
        "instance": "",
    }.get(method_kind, "")

    f.write(f"<!-- API_INHERITED_HEADING {anchor_id} -->\n")

    if method_name == "__init__":
        class_display = derived_cls_ident.rsplit(".", 1)[-1]
        f.write(f"## {class_display}() {{: #{anchor_id} }}\n\n")
        if base_ident.startswith("loqs."):
            base_class = base_ident.split(".")[-1]
            f.write(f"Inherited constructor from [`{base_class}()`](api:{base_ident}.{method_name}).\n\n")
        else:
            f.write(f"Inherited constructor from [](api:{base_ident}).\n\n")
        return

    f.write(f"## {method_name} {{: #{anchor_id} }}\n\n")

    if base_ident.startswith("loqs."):
        base_link_text = f"{base_ident.split('.')[-1]}.{method_name}"
        f.write(f"Inherited {kind_label}method from [{base_link_text}](api:{base_ident}.{method_name}).\n\n")
    else:
        f.write(f"Inherited {kind_label}method from [](api:{base_ident}).\n\n")


def write_module_functions_block(f, mod_ident: str, funcs: list[str]) -> None:
    if not funcs:
        return

    for i, fn in enumerate(funcs):
        f.write(f"<!-- API_MODULE_MEMBERS owner={mod_ident} -->\n\n")
        write_mkdocstrings_block(f, mod_ident, members=[fn], inherited_members=False, heading_level=3)
        if i != len(funcs) - 1:
            f.write("---\n\n")


def write_module_members_table(
    f,
    mod_ident: str,
    page_url: str,
    rows: list[dict],
    inv_objects: dict[str, str],
    inv_kinds: dict[str, str],
    link_names: set[str] | None = None,
) -> None:
    if not rows:
        return

    local_targets = {
        r["name"]: f"{mod_ident}.{r['name']}"
        for r in rows
        if isinstance(r.get("name"), str) and r.get("name")
    }

    f.write("| Name | Type | Value | Doc |\n")
    f.write("|---|---|---|---|\n")
    for r in rows:
        nm = r["name"]
        anchor_id = f"{mod_ident}.{nm}"
        inv_objects[anchor_id] = f"{page_url}#{anchor_id}"
        inv_kinds[anchor_id] = r.get("kind") or K_VARIABLE

        name_cell = f'<a id="{anchor_id}"></a>`{nm}`'
        typ = render_inline_md(
            (r.get("type") or "").replace("\n", " ").replace("|", "\\|"),
            link_names=link_names,
            local_targets=local_targets,
        )

        val = r.get("value")
        typevar_bound = (r.get("typevar_bound") or "").strip()
        if r.get("kind") == K_TYPE_VARIABLE and typevar_bound:
            bound_name = typevar_bound.split(".")[-1]
            if bound_name in local_targets:
                target = local_targets[bound_name]
            else:
                target = bound_name

            val_s = (
                f"<code>TypeVar('{nm}', bound='</code>"
                f'<a href="api:{target}"><code>{bound_name}</code></a>'
                f"<code>')</code>"
            )
        elif val is None or str(val).strip() == "":
            val_s = "*unset*"
        else:
            val_s_raw = str(val).replace("\n", " ").strip()
            val_s = render_inline_md(
                val_s_raw.replace("|", "\\|"),
                link_names=link_names,
                local_targets=local_targets,
            )

        doc = render_inline_md(r.get("doc") or "", prose=True)
        
        f.write(f"| {name_cell} | {typ} | {val_s} | {doc} |\n")
    f.write("\n")


def write_module_page(
    path: Path,
    title: str,
    mod_ident: str,
    page_url: str,
    *,
    rows: list[dict],
    funcs: list[str],
    classes: list[str],
    inv_objects: dict[str, str],
    inv_kinds: dict[str, str],
    link_names: set[str] | None = None,
) -> None:
    with mkdocs_gen_files.open(path, "w") as f:
        f.write(f"# `{mod_ident}`\n\n")

        f.write(f"<!-- API_TOC_REMOVE {mod_ident} -->\n\n")

        write_mkdocstrings_block(f, mod_ident, members=False, inherited_members=False)

        if classes:
            f.write("## Classes\n\n")
            for cls in classes:
                f.write(f"- [`{cls}`]({cls}.md)\n")
            f.write("\n\n\n")

        if rows:
            f.write("## Attributes\n\n")
            write_module_members_table(f, mod_ident, page_url, rows, inv_objects, inv_kinds, link_names)
            f.write("\n\n\n")

        if funcs:
            f.write("## Functions\n\n")
            write_module_functions_block(f, mod_ident, funcs)
            f.write("\n\n\n")


def write_class_page(
    path: Path,
    title: str,
    cls_ident: str,
    page_url: str,
    *,
    cls_obj: type | None,
    var_rows: list[dict],
    inherited_method_stubs: dict[str, tuple[str, str]],
    methods: list[str],
    owner_override: dict[str, str],
    toc_remove_anchors: list[str],
    inv_objects: dict[str, str],
    inv_kinds: dict[str, str],
    link_names: set[str] | None = None,
) -> None:
    with mkdocs_gen_files.open(path, "w") as f:
        f.write(f"# `{title}`\n\n")

        if toc_remove_anchors:
            f.write(f"<!-- API_TOC_REMOVE {' '.join(toc_remove_anchors)} -->\n\n")

        # Special case __init__
        write_class_intro(f, cls_ident)

        init_owner = owner_override.get("__init__", cls_ident)
        if cls_obj is not None and "__init__" in methods and init_owner != cls_ident:
            write_inherited_doc_render_block(
                f,
                derived_cls_ident=cls_ident,
                derived_member_name="__init__",
                base_owner_ident=init_owner,
                base_member_name="__init__",
            )

        if var_rows:
            f.write("\n---\n\n")
            f.write("## Attributes\n")
            write_class_members_table(
                f,
                var_rows,
                derived_ident=cls_ident,
                class_anchor_prefix=cls_ident,
                inv_objects=inv_objects,
                inv_kinds=inv_kinds,
                page_url=page_url,
                link_names=link_names,
                module_local_targets={
                    name: f"{cls_ident.rsplit('.', 1)[0]}.{name}"
                    for name in (link_names or set())
                },
            )

        other_methods = [m for m in methods if m != "__init__"]

        if other_methods or inherited_method_stubs:
            f.write("\n---\n\n")

            inherited_method_stubs = inherited_method_stubs or {}
            declared_set = set(other_methods)
            all_names = sorted(set(other_methods) | set(inherited_method_stubs), key=lambda name: name.lower())

            for m in all_names:
                if m in declared_set:
                    owner = owner_override.get(m, cls_ident)

                    f.write(f"<!-- API_METHOD owner={cls_ident} member={m} -->\n\n")
                    write_mkdocstrings_block(f, cls_ident, members=[m], inherited_members=False)

                    if cls_obj is not None and owner != cls_ident:
                        write_inherited_doc_render_block(
                            f,
                            derived_cls_ident=cls_ident,
                            derived_member_name=m,
                            base_owner_ident=owner,
                            base_member_name=m,
                        )
                else:
                    kind, base_ident = inherited_method_stubs[m]
                    write_inherited_method_stub(
                        f,
                        derived_cls_ident=cls_ident,
                        method_name=m,
                        method_kind=kind,
                        base_ident=base_ident,
                    )

                inv_objects[f"{cls_ident}.{m}"] = f"{page_url}#{cls_ident}.{m}"
                inv_kinds[f"{cls_ident}.{m}"] = K_METHOD
                f.write("\n---\n\n")