# docs_scripts/gen_ref_pages.py
from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

import mkdocs_gen_files

from docs_scripts.api_inventory import build_suffix_index

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "loqs"

INVENTORY_PATH = "api_inventory.json"  # generated into API site output root


def _is_public_method(name: str) -> bool:
    return not name.startswith("_") and not name.startswith("__")


def _is_public_var(name: str) -> bool:
    return not name.startswith("_") and not name.startswith("__")


def _is_public_property(name: str) -> bool:
    return _is_public_method(name)


_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _var_sort_key(row: dict) -> tuple[int, str]:
    name = (row.get("name") or "")
    return (0 if _ALL_CAPS_RE.fullmatch(name) else 1, name.lower())


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _doc_hint_from_next_stmt(body: list[ast.stmt], i: int) -> str:
    if i + 1 >= len(body):
        return ""
    nxt = body[i + 1]
    if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) and isinstance(nxt.value.value, str):
        return nxt.value.value.strip().splitlines()[0]
    return ""


def _is_typevar_call(value: ast.AST | None) -> bool:
    if not isinstance(value, ast.Call):
        return False
    fn = value.func
    if isinstance(fn, ast.Name):
        return fn.id == "TypeVar"
    if isinstance(fn, ast.Attribute):
        return fn.attr == "TypeVar"
    return False


def _is_typealias_ann(annotation: ast.AST | None) -> bool:
    ann = _unparse(annotation).strip()
    return bool(ann) and ann.split(".")[-1] == "TypeAlias"


def _qualname_to_ident(obj: Any) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def _has_doc(obj: Any) -> bool:
    return bool(inspect.getdoc(obj))


def _method_owner_for_docs(cls: type, name: str) -> type:
    if name not in getattr(cls, "__dict__", {}):
        return cls
    obj = cls.__dict__.get(name)
    if obj is None:
        return cls

    # For properties, check fget docstring (and fset/fdel if desired later)
    if isinstance(obj, property):
        if obj.fget is not None and _has_doc(obj.fget):
            return cls
    else:
        if _has_doc(obj):
            return cls

    for base in cls.__mro__[1:]:
        if name in getattr(base, "__dict__", {}):
            base_obj = base.__dict__.get(name)
            if base_obj is None:
                continue
            if isinstance(base_obj, property):
                if base_obj.fget is not None and _has_doc(base_obj.fget):
                    return base
            else:
                if _has_doc(base_obj):
                    return base
    return cls


def module_public_api(py_file: Path) -> tuple[list[str], list[str], list[dict]]:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return [], [], []

    classes: list[str] = []
    funcs: list[str] = []
    rows: list[dict] = []

    body = tree.body
    for i, node in enumerate(body):
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            classes.append(node.name)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_method(node.name):
            funcs.append(node.name)

        elif isinstance(node, ast.Assign):
            doc = _doc_hint_from_next_stmt(body, i)
            value_s = _unparse(node.value).strip() if node.value is not None else ""
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and _is_public_var(tgt.id):
                    is_tv = _is_typevar_call(node.value)
                    kind = "type variable" if is_tv else "variable"
                    rows.append(
                        {"name": tgt.id, "kind": kind, "type": "TypeVar" if is_tv else "", "value": value_s, "doc": doc}
                    )

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and _is_public_var(node.target.id):
                doc = _doc_hint_from_next_stmt(body, i)
                ann_s = _unparse(node.annotation).strip()
                value_s = _unparse(node.value).strip() if node.value is not None else ""

                is_alias = _is_typealias_ann(node.annotation)
                kind = "type alias" if is_alias else "variable"

                rows.append(
                    {
                        "name": node.target.id,
                        "kind": kind,
                        "type": "TypeAlias" if is_alias else ann_s,
                        "value": value_s,
                        "doc": doc,
                    }
                )

    classes.sort(key=str.lower)
    funcs.sort(key=str.lower)

    def score(r: dict) -> tuple[int, int, int]:
        return (
            1 if (r.get("value") or "").strip() else 0,
            1 if (r.get("type") or "").strip() else 0,
            1 if (r.get("doc") or "").strip() else 0,
        )

    by_name: dict[str, dict] = {}
    for r in rows:
        prev = by_name.get(r["name"])
        if prev is None or score(r) > score(prev):
            by_name[r["name"]] = r
    rows = sorted(by_name.values(), key=lambda d: d["name"].lower())

    return classes, funcs, rows


def _classvar_inner(type_s: str) -> str:
    s = (type_s or "").strip()
    if not s or "ClassVar[" not in s:
        return s

    start = s.find("ClassVar[")
    if start < 0:
        return s

    i = start + len("ClassVar[")
    depth = 1
    inner_chars: list[str] = []
    while i < len(s):
        ch = s[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                break
        inner_chars.append(ch)
        i += 1

    inner = "".join(inner_chars).strip()
    return inner or s


def _class_var_info_map_from_ast(py_file: Path, class_name: str, *, owner_ident: str) -> dict[str, dict]:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return {}

    cls: ast.ClassDef | None = None
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            cls = n
            break
    if cls is None:
        return {}

    out: dict[str, dict] = {}
    body = cls.body

    for i, node in enumerate(body):
        if isinstance(node, ast.Assign):
            doc = _doc_hint_from_next_stmt(body, i)
            value_s = _unparse(node.value).strip() if node.value is not None else ""
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and _is_public_var(tgt.id):
                    out[tgt.id] = {"name": tgt.id, "type": "", "owner": owner_ident, "value": value_s, "doc": doc}

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and _is_public_var(node.target.id):
                doc = _doc_hint_from_next_stmt(body, i)
                ann_s = _unparse(node.annotation).strip()
                value_s = _unparse(node.value).strip() if node.value is not None else ""
                out[node.target.id] = {
                    "name": node.target.id,
                    "type": ann_s,
                    "owner": owner_ident,
                    "value": value_s,
                    "doc": doc,
                }

    return out


def class_var_rows_with_mro(derived_py_file: Path, cls_obj: type) -> list[dict]:
    derived_ident = _qualname_to_ident(cls_obj)
    derived_map = _class_var_info_map_from_ast(derived_py_file, cls_obj.__name__, owner_ident=derived_ident)

    inherited_map: dict[str, dict] = {}

    for base in cls_obj.__mro__[1:]:
        if base is object:
            continue
        if getattr(base, "__module__", "") == "builtins":
            continue

        try:
            src = inspect.getsourcefile(base)
        except TypeError:
            continue
        except Exception:
            continue
        if not src:
            continue

        base_file = Path(src)
        if not base_file.exists():
            continue

        base_ident = _qualname_to_ident(base)
        base_map = _class_var_info_map_from_ast(base_file, base.__name__, owner_ident=base_ident)
        if not base_map:
            continue

        for name, drow in derived_map.items():
            brow = base_map.get(name)
            if not brow:
                continue
            if not (drow.get("doc") or "").strip() and (brow.get("doc") or "").strip():
                drow["doc"] = brow["doc"]
            if not (drow.get("type") or "").strip() and (brow.get("type") or "").strip():
                drow["type"] = brow["type"]
            if not (drow.get("value") or "").strip() and (brow.get("value") or "").strip():
                drow["value"] = brow["value"]

        for name, brow in base_map.items():
            if name in derived_map:
                continue
            inherited_map.setdefault(name, brow)

    merged: dict[str, dict] = {}
    merged.update(inherited_map)
    merged.update(derived_map)

    return sorted(merged.values(), key=_var_sort_key)


def property_rows_from_introspection(cls_obj: type, *, owner_ident: str) -> list[dict]:
    """
    Build "member variable" rows for @property descriptors.

    Table mapping:
      - Type column: inferred return type annotation of the property's fget (if any)
      - Value column: *property* / *read-only property* (+ *(abstract)* if applicable)
    """
    rows: list[dict] = []
    for name, val in getattr(cls_obj, "__dict__", {}).items():
        if not _is_public_property(name):
            continue
        if not isinstance(val, property):
            continue

        # Type column: return annotation on fget if present
        typ = ""
        fget = val.fget
        if fget is not None:
            try:
                ann = inspect.signature(fget).return_annotation
            except (TypeError, ValueError):
                ann = inspect.Signature.empty

            if ann is not inspect.Signature.empty:
                # Prefer a readable type string
                if isinstance(ann, str):
                    typ = ann
                else:
                    typ = getattr(ann, "__name__", None) or str(ann)

        # Value column: property kind + abstract marker
        kind = "*read-only property*" if val.fset is None else "*property*"
        is_abstract = bool(getattr(fget, "__isabstractmethod__", False)) if fget is not None else False
        val_s = kind + (" *(abstract)*" if is_abstract else "")

        # Doc column: first line of fget docstring (if any)
        doc = ""
        if fget is not None:
            d = inspect.getdoc(fget) or ""
            d = d.strip()
            if d:
                doc = d.splitlines()[0]

        rows.append(
            {
                "name": name,
                "type": typ,
                "owner": owner_ident,
                "value": val_s,
                "doc": doc,
            }
        )

    return sorted(rows, key=_var_sort_key)


def class_members_from_introspection(
    class_name: str,
    mod_ident: str,
) -> tuple[
    type | None,
    list[str],  # instance methods
    list[str],  # class methods
    list[str],  # static methods
    dict[str, str],  # owner_override
    list[str],  # toc_remove_anchors
]:
    try:
        mod = importlib.import_module(mod_ident)
    except Exception:
        return None, [], [], {}, []

    cls = getattr(mod, class_name, None)
    if cls is None or not isinstance(cls, type):
        return None, [], [], {}, []

    toc_remove_anchors: list[str] = [_qualname_to_ident(cls)]
    for base in cls.__mro__[1:]:
        if base is object:
            continue
        toc_remove_anchors.append(_qualname_to_ident(base))

    inst: list[str] = []
    clsm: list[str] = []
    stat: list[str] = []
    owner_override: dict[str, str] = {}

    for name, val in getattr(cls, "__dict__", {}).items():
        if not _is_public_method(name):
            continue

        if isinstance(val, staticmethod):
            stat.append(name)
        elif isinstance(val, classmethod):
            clsm.append(name)
        elif inspect.isfunction(val):
            inst.append(name)
        else:
            # NOTE: properties intentionally excluded from "Methods"
            continue

        doc_owner = _method_owner_for_docs(cls, name)
        owner_override[name] = _qualname_to_ident(doc_owner)

    inst.sort(key=str.lower)
    clsm.sort(key=str.lower)
    stat.sort(key=str.lower)

    return cls, inst, clsm, stat, owner_override, toc_remove_anchors


def write_class_members_table(
    f,
    rows: list[dict],
    *,
    derived_ident: str,
    class_anchor_prefix: str,
    inv_objects: dict[str, str],
    page_url: str,
) -> None:
    if not rows:
        return

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
        typ = _classvar_inner(typ_raw).replace("\n", " ").replace("|", "\\|")

        val = r.get("value")
        if val is None or str(val).strip() == "":
            val_s = "*unset*"
        else:
            val_s = str(val).replace("\n", " ").replace("|", "\\|")

            # If the value already contains Markdown (e.g., *property*), don't wrap in code ticks.
            if "*" not in val_s and "_" not in val_s:
                val_s = f"`{val_s}`"

        doc = (r.get("doc") or "").replace("\n", " ").replace("|", "\\|")

        f.write(f"| {name_cell} | {typ} | {val_s} | {doc} |\n")
    f.write("\n")


def write_class_intro(f, cls_ident: str) -> None:
    f.write(f"::: {cls_ident}\n")
    f.write("    options:\n")
    f.write("      members: false\n")
    f.write("      inherited_members: false\n")
    f.write("\n")


def write_class_member_block(f, owner_ident: str, member_name: str) -> None:
    f.write(f"<!-- API_METHOD owner={owner_ident} member={member_name} -->\n\n")
    f.write(f"::: {owner_ident}\n")
    f.write("    options:\n")
    f.write("      members:\n")
    f.write(f"        - {member_name}\n")
    f.write("      inherited_members: false\n")


def write_module_functions_block(f, mod_ident: str, funcs: list[str]) -> None:
    f.write(f"<!-- API_MODULE_MEMBERS owner={mod_ident} -->\n\n")
    f.write(f"::: {mod_ident}\n")
    f.write("    options:\n")
    f.write("      members:\n")
    for fn in funcs:
        f.write(f"        - {fn}\n")
    f.write("      inherited_members: false\n")
    f.write("\n")


def write_module_members_table(
    f,
    mod_ident: str,
    page_url: str,
    rows: list[dict],
    inv_objects: dict[str, str],
) -> None:
    if not rows:
        return

    f.write("| Name | Type | Value | Doc |\n")
    f.write("|---|---|---|---|\n")
    for r in rows:
        nm = r["name"]
        anchor_id = f"{mod_ident}.{nm}"
        inv_objects[anchor_id] = f"{page_url}#{anchor_id}"

        name_cell = f'<a id="{anchor_id}"></a>`{nm}`'
        typ = (r.get("type") or "").replace("\n", " ").replace("|", "\\|")

        val = r.get("value")
        if val is None or str(val).strip() == "":
            val_s = "*unset*"
        else:
            val_s = str(val).replace("\n", " ")#.replace("|", "\\|")
            val_s = f"`{val_s}`"

        doc = (r.get("doc") or "").replace("\n", " ").replace("|", "\\|")
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
) -> None:
    with mkdocs_gen_files.open(path, "w") as f:
        f.write(f"# `{title}`\n\n")
        
        # Always render the module/package docstring
        f.write(f"::: {mod_ident}\n")
        f.write("    options:\n")
        f.write("      members: false\n")
        f.write("      inherited_members: false\n\n")

        if classes:
            f.write("## Classes\n\n")
            for cls in classes:
                f.write(f"- [`{cls}`]({cls}/)\n")
            f.write("\n\n\n")

        if rows:
            f.write("## Members\n\n")
            write_module_members_table(f, mod_ident, page_url, rows, inv_objects)
            f.write("\n\n\n")

        if funcs:
            f.write("## Methods\n\n")
            write_module_functions_block(f, mod_ident, funcs)
            f.write("\n\n\n")



def write_class_page(
    path: Path,
    title: str,
    cls_ident: str,
    page_url: str,
    *,
    var_rows: list[dict],
    instance_methods: list[str],
    class_methods: list[str],
    static_methods: list[str],
    owner_override: dict[str, str],
    toc_remove_anchors: list[str],
    inv_objects: dict[str, str],
) -> None:
    with mkdocs_gen_files.open(path, "w") as f:
        f.write(f"# `{title}`\n\n")

        if toc_remove_anchors:
            f.write(f"<!-- API_TOC_REMOVE {' '.join(toc_remove_anchors)} -->\n\n")

        write_class_intro(f, cls_ident)

        if var_rows:
            f.write("## Members\n")
            write_class_members_table(
                f,
                var_rows,
                derived_ident=cls_ident,
                class_anchor_prefix=cls_ident,
                inv_objects=inv_objects,
                page_url=page_url,
            )

        if static_methods or class_methods or instance_methods:
            f.write("## Methods\n")
            for m in static_methods:
                owner = owner_override.get(m, cls_ident)
                write_class_member_block(f, owner, m)
            for m in class_methods:
                owner = owner_override.get(m, cls_ident)
                write_class_member_block(f, owner, m)
            for m in instance_methods:
                owner = owner_override.get(m, cls_ident)
                write_class_member_block(f, owner, m)


def main() -> None:
    nav = mkdocs_gen_files.Nav()
    inv_objects: dict[str, str] = {}

    # Global references page for citations in docstrings
    nav[("Bibliography",)] = "bib.md"
    with mkdocs_gen_files.open("bib.md", "w") as f:
        f.write("# Bibliography\n\n")
        f.write("\\full_bibliography\n")

    nav[("loqs",)] = "loqs/index.md"
    with mkdocs_gen_files.open("loqs/index.md", "w") as f:
        f.write("# `loqs`\n\n")
        f.write("Package reference. Use the sidebar to browse.\n")

    def api_page_url(mod_parts: tuple[str, ...]) -> str:
        return "/" + "/".join(mod_parts) + "/"

    for py in sorted(PKG_DIR.rglob("*.py")):
        rel = py.relative_to(PKG_DIR)
        parts = rel.with_suffix("").parts

        classes, funcs, rows = module_public_api(py)

        if py.name == "__init__.py":
            mod_parts = ("loqs",) + parts[:-1]
            mod_ident = "loqs" + ("" if len(mod_parts) == 1 else "." + ".".join(mod_parts[1:]))
        else:
            mod_parts = ("loqs",) + parts
            mod_ident = "loqs." + ".".join(mod_parts[1:])

        page = Path(*mod_parts) / "index.md"
        nav_key = mod_parts
        label = mod_parts[-1]

        mod_page_url = api_page_url(mod_parts)

        inv_objects[mod_ident] = f"{mod_page_url}#{mod_ident}"
        for fn in funcs:
            inv_objects[f"{mod_ident}.{fn}"] = f"{mod_page_url}#{mod_ident}.{fn}"
        for cls_name in classes:
            inv_objects[f"{mod_ident}.{cls_name}"] = f"{mod_page_url}#{mod_ident}.{cls_name}"

        nav[nav_key] = page.as_posix()
        write_module_page(
            page,
            title=label,
            mod_ident=mod_ident,
            page_url=mod_page_url,
            rows=rows,
            funcs=funcs,
            classes=classes,
            inv_objects=inv_objects,
        )

        for cls_name in classes:
            cls_ident = f"{mod_ident}.{cls_name}"
            cls_page = Path(*mod_parts) / f"{cls_name}.md"
            nav[(*nav_key, cls_name)] = cls_page.as_posix()

            (
                cls_obj,
                inst,
                clsm,
                stat,
                owner_override,
                toc_remove_anchors,
            ) = class_members_from_introspection(cls_name, mod_ident)

            cls_page_url = mod_page_url + f"{cls_name}/"

            # Inventory entries for methods should point at the class page
            for m in stat + clsm + inst:
                inv_objects[f"{cls_ident}.{m}"] = f"{cls_page_url}#{cls_ident}.{m}"

            if cls_obj is not None:
                var_rows = class_var_rows_with_mro(py, cls_obj)

                # Add @property descriptors to the Members table (as "member variables")
                prop_rows = property_rows_from_introspection(cls_obj, owner_ident=cls_ident)

                # Merge (prefer var_rows entries if name collides)
                by_name = {r["name"]: r for r in prop_rows}
                for r in var_rows:
                    by_name[r["name"]] = r
                var_rows = sorted(by_name.values(), key=_var_sort_key)
            else:
                derived_map = _class_var_info_map_from_ast(py, cls_name, owner_ident=cls_ident)
                var_rows = sorted(derived_map.values(), key=_var_sort_key)

            write_class_page(
                cls_page,
                title=cls_name,
                cls_ident=cls_ident,
                page_url=cls_page_url,
                var_rows=var_rows,
                instance_methods=inst,
                class_methods=clsm,
                static_methods=stat,
                owner_override=owner_override,
                toc_remove_anchors=toc_remove_anchors,
                inv_objects=inv_objects,
            )

    with mkdocs_gen_files.open("index.md", "w") as f:
        f.write("# API Reference\n\n")
        f.write("Use the sidebar to browse.\n")

    with mkdocs_gen_files.open("SUMMARY.md", "w") as f:
        f.write("* [Home](/)\n")
        f.write("* [API Reference](index.md)\n")
        for line in nav.build_literate_nav():
            f.write("  " + line)

    suffix_index = build_suffix_index(inv_objects, package="loqs")
    with mkdocs_gen_files.open(INVENTORY_PATH, "w") as f:
        json.dump({"objects": inv_objects, "suffix_index": suffix_index}, f, indent=2, sort_keys=True)

    # ALSO write to disk so api_hooks.py can load it during the same build
    # serve.py will clean it
    disk_path = REPO_ROOT / "docs" / f"_{INVENTORY_PATH}"
    disk_path.write_text(
        json.dumps({"objects": inv_objects, "suffix_index": suffix_index}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


main()