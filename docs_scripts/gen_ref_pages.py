from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import mkdocs_gen_files

from docs_scripts.api_inventory import build_suffix_index

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "loqs"

INVENTORY_PATH = "api_inventory.json"  # generated into API site output root


def _is_public_method(name: str) -> bool:
    return not name.startswith("_") and not name.startswith("__")


def _is_documented_class_method(name: str) -> bool:
    """
    Methods documented on class pages.

    Public methods are included, and ``__init__`` is included explicitly so
    constructors appear in the generated API like AutoAPI-style class docs.
    """
    return name == "__init__" or _is_public_method(name)


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


def _has_own_doc(obj: Any) -> bool:
    """
    Return whether an object defines its own docstring directly, without
    inheriting one through ``inspect.getdoc`` MRO fallback.
    """
    return bool((getattr(obj, "__doc__", None) or "").strip())


def _method_owner_for_docs(cls: type, name: str) -> type:
    if name not in getattr(cls, "__dict__", {}):
        return cls

    obj = cls.__dict__.get(name)
    if obj is None:
        return cls

    # For locally overridden members, only treat an object as documented if it
    # defines its own docstring directly. Using inspect.getdoc() here would
    # incorrectly inherit base-class docstrings for undocumented overrides.
    if isinstance(obj, property):
        if obj.fget is not None and _has_own_doc(obj.fget):
            return cls
    else:
        if _has_own_doc(obj):
            return cls

    for base in cls.__mro__[1:]:
        if name not in getattr(base, "__dict__", {}):
            continue

        base_obj = base.__dict__.get(name)
        if base_obj is None:
            continue

        if isinstance(base_obj, property):
            if base_obj.fget is not None and inspect.getdoc(base_obj.fget):
                return base
        else:
            if inspect.getdoc(base_obj):
                return base

    return cls


def _collect_import_aliases(tree: ast.AST) -> dict[str, str]:
    """
    Collect local import aliases from a module AST.

    Includes imports nested under if/try blocks (e.g. TYPE_CHECKING patterns),
    since we walk the full tree.
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out[a.asname] = a.name
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            mod = node.module
            for a in node.names:
                if a.asname:
                    out[a.asname] = f"{mod}.{a.name}"
                else:
                    out[a.name] = f"{mod}.{a.name}"
    return out


def _expand_type_aliases(type_s: str, aliases: dict[str, str]) -> str:
    """
    Expand imported names/aliases inside an annotation string, then normalize
    verbose module prefixes for display.

    Replaces whole identifier tokens only, prefers longer keys first, and
    strips common prefixes like ``typing.`` and ``collections.abc.``.
    """
    s = (type_s or "").strip()
    if not s:
        return s

    if aliases:
        keys = sorted(aliases.keys(), key=len, reverse=True)
        for k in keys:
            v = aliases[k]
            s = re.sub(rf"\b{re.escape(k)}\b", v, s)

    s = re.sub(r"\btyping\.", "", s)
    s = re.sub(r"\bcollections\.abc\.", "", s)
    return s


def _type_to_md(
    type_s: str,
    link_names: set[str] | None = None,
) -> str:
    """
    Render a type/value string for Markdown tables.

    Imported identifiers are first expanded via the local alias map. Fully-qualified
    ``loqs.`` identifiers are replaced with short ``api:`` links whose target and
    display text are both just the final symbol name. Additionally, short local names
    listed in ``link_names`` are linked directly, which is useful for module-level
    type aliases and variables that already have anchors on the current page.

    Non-linked fragments are wrapped in backticks so mixed types like
    ``list[loqs.foo.Bar]`` render as code plus links.
    """
    s = (type_s or "").strip()
    if not s:
        return ""
    
    # Short circuit for TypeVar, we want to link bound class
    m_typevar = re.match(
        r"^(?P<prefix>TypeVar\('(?P<name>[^']+)'\s*,\s*bound=')(?P<bound>[A-Za-z_][A-Za-z0-9_\.]*)(?P<suffix>'\))$",
        s,
    )
    if m_typevar:
        bound = m_typevar.group("bound")
        if bound.startswith("loqs."):
            target = bound.split(".")[-1]
            label = target
        else:
            target = bound
            label = bound
        return (
            f"<code>{m_typevar.group('prefix')}</code>"
            f'<a href="api:{target}"><code>{label}</code></a>'
            f"<code>{m_typevar.group('suffix')}</code>"
        )

    names = sorted(link_names or set(), key=len, reverse=True)
    if names:
        token_re = re.compile(
            r"\bloqs(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b|"
            + "|".join(rf"\b{re.escape(n)}\b" for n in names)
        )
    else:
        token_re = re.compile(r"\bloqs(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")

    parts: list[str] = []
    last = 0

    for m in token_re.finditer(s):
        prefix = s[last:m.start()]
        if prefix:
            parts.append(f"<code>{prefix}</code>")

        token = m.group(0)
        if token.startswith("loqs."):
            target = token.split(".")[-1]
            label = target
        else:
            target = token
            label = token

        parts.append(f'<a href="api:{target}"><code>{label}</code></a>')
        last = m.end()

    suffix = s[last:]
    if suffix:
        parts.append(f"<code>{suffix}</code>")

    if parts:
        return "".join(parts)

    return f"<code>{s}</code>"


def module_public_api(py_file: Path) -> tuple[list[str], list[str], list[dict]]:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
        aliases = _collect_import_aliases(tree)
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
                ann_s = _expand_type_aliases(_unparse(node.annotation).strip(), aliases)
                value_s = _expand_type_aliases(_unparse(node.value).strip() if node.value is not None else "", aliases)

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
        aliases = _collect_import_aliases(tree)
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
                ann_s = _expand_type_aliases(_unparse(node.annotation).strip(), aliases)
                value_s = _expand_type_aliases(_unparse(node.value).strip() if node.value is not None else "", aliases)
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

        def _score_row(r: dict) -> tuple[int, int, int]:
            return (
                1 if (r.get("value") or "").strip() else 0,
                1 if (r.get("type") or "").strip() else 0,
                1 if (r.get("doc") or "").strip() else 0,
            )

        for name, brow in base_map.items():
            if name in derived_map:
                continue

            prev = inherited_map.get(name)
            if prev is None or _score_row(brow) > _score_row(prev):
                inherited_map[name] = brow

    merged: dict[str, dict] = {}
    merged.update(inherited_map)
    merged.update(derived_map)

    return sorted(merged.values(), key=_var_sort_key)


def property_rows_from_introspection(cls_obj: type, *, owner_ident: str, aliases: dict[str, str]) -> list[dict]:
    """
    Build "member variable" rows for @property descriptors.

    Table mapping:
      - Type column: inferred return type annotation of the property's fget
        (falling back through the MRO when the derived property omits it)
      - Value column: *property* / *read-only property* (+ *(abstract)* if applicable)
      - Doc column: first line of the property's getter docstring, with base-class
        fallback when the derived property omits it
    """
    rows: list[dict] = []
    for name, val in getattr(cls_obj, "__dict__", {}).items():
        if not _is_public_property(name):
            continue
        if not isinstance(val, property):
            continue

        typ = ""
        fget = val.fget
        type_owner = cls_obj
        if fget is not None:
            try:
                ann = inspect.signature(fget).return_annotation
            except (TypeError, ValueError):
                ann = inspect.Signature.empty

            if ann is not inspect.Signature.empty:
                if isinstance(ann, str):
                    typ = ann
                else:
                    typ = getattr(ann, "__name__", None) or str(ann)
            else:
                for base in cls_obj.__mro__[1:]:
                    if base is object:
                        continue
                    base_prop = getattr(base, "__dict__", {}).get(name)
                    if not isinstance(base_prop, property) or base_prop.fget is None:
                        continue
                    try:
                        base_ann = inspect.signature(base_prop.fget).return_annotation
                    except (TypeError, ValueError):
                        base_ann = inspect.Signature.empty
                    if base_ann is inspect.Signature.empty:
                        continue
                    if isinstance(base_ann, str):
                        typ = base_ann
                    else:
                        typ = getattr(base_ann, "__name__", None) or str(base_ann)
                    type_owner = base
                    break
        typ = _expand_type_aliases(typ, aliases)

        kind = "*read-only property*" if val.fset is None else "*property*"
        is_abstract = bool(getattr(fget, "__isabstractmethod__", False)) if fget is not None else False
        val_s = kind + (" *(abstract)*" if is_abstract else "")

        doc = ""
        doc_owner = cls_obj
        if fget is not None:
            d = inspect.getdoc(fget) or ""
            d = d.strip()
            if d:
                doc = d.splitlines()[0]
            else:
                for base in cls_obj.__mro__[1:]:
                    if base is object:
                        continue
                    base_prop = getattr(base, "__dict__", {}).get(name)
                    if not isinstance(base_prop, property) or base_prop.fget is None:
                        continue
                    bd = inspect.getdoc(base_prop.fget) or ""
                    bd = bd.strip()
                    if bd:
                        doc = bd.splitlines()[0]
                        doc_owner = base
                        break

        row_owner = owner_ident
        if doc_owner is not cls_obj and type_owner is doc_owner:
            row_owner = _qualname_to_ident(doc_owner)

        rows.append(
            {
                "name": name,
                "type": typ,
                "owner": row_owner,
                "value": val_s,
                "doc": doc,
            }
        )

    return sorted(rows, key=_var_sort_key)


def inherited_only_methods(cls_obj: type, *, declared: set[str]) -> dict[str, tuple[str, str]]:
    """
    Return mapping: member_name -> (kind, base_ident) for documented methods
    present via inheritance but not declared on cls_obj.__dict__.

    kind in {"static", "class", "instance"}
    """
    out: dict[str, tuple[str, str]] = {}

    for name in dir(cls_obj):
        if not _is_documented_class_method(name):
            continue
        if name in declared:
            continue

        for base in cls_obj.__mro__[1:]:
            if base is object:
                continue
            if getattr(base, "__module__", "") == "builtins":
                continue
            if name not in getattr(base, "__dict__", {}):
                continue

            base_val = base.__dict__.get(name)

            if isinstance(base_val, property):
                break

            if isinstance(base_val, staticmethod):
                kind = "static"
            elif isinstance(base_val, classmethod):
                kind = "class"
            elif inspect.isfunction(base_val):
                kind = "instance"
            else:
                break

            out[name] = (kind, _qualname_to_ident(base))
            break

    def _sort_key(item: tuple[str, tuple[str, str]]) -> tuple[int, str]:
        name = item[0]
        return (0 if name == "__init__" else 1, name.lower())

    return dict(sorted(out.items(), key=_sort_key))


def class_members_from_introspection(
    class_name: str,
    mod_ident: str,
) -> tuple[
    type | None,
    list[str],          # declared documented methods
    dict[str, str],     # doc owner override
    list[str],          # toc_remove_anchors
]:
    try:
        mod = importlib.import_module(mod_ident)
    except Exception:
        return None, [], {}, []

    cls = getattr(mod, class_name, None)
    if cls is None or not isinstance(cls, type):
        return None, [], {}, []

    toc_remove_anchors: list[str] = [_qualname_to_ident(cls)]
    for base in cls.__mro__[1:]:
        if base is object:
            continue
        if getattr(base, "__module__", "") == "builtins":
            continue
        toc_remove_anchors.append(_qualname_to_ident(base))

    methods: list[str] = []
    owner_override: dict[str, str] = {}

    for name, val in getattr(cls, "__dict__", {}).items():
        if not _is_documented_class_method(name):
            continue

        if isinstance(val, (staticmethod, classmethod)) or inspect.isfunction(val):
            methods.append(name)
        else:
            continue

        doc_owner = _method_owner_for_docs(cls, name)
        if getattr(doc_owner, "__module__", "") == "builtins":
            doc_owner = cls
        owner_override[name] = _qualname_to_ident(doc_owner)

    def _method_sort_key(name: str) -> tuple[int, str]:
        return (0 if name == "__init__" else 1, name.lower())

    methods.sort(key=_method_sort_key)

    return cls, methods, owner_override, toc_remove_anchors


# ----------------------------------------------------------------------
#  Lightweight NumPy-doc fallback rendering for undocumented overrides
# ----------------------------------------------------------------------

_SECTION_NAMES = {
    "parameters",
    "returns",
    "yields",
    "raises",
    "notes",
    "examples",
    "see also",
    "references",
    "warnings",
    "attributes",
    "methods",
}


def _is_numpy_section_header(lines: list[str], i: int) -> bool:
    if i + 1 >= len(lines):
        return False
    title = lines[i].strip()
    underline = lines[i + 1].strip()
    return bool(title) and title.lower() in _SECTION_NAMES and len(underline) >= 3 and set(underline) == {"-"}


def _split_numpy_doc_sections(doc: str) -> tuple[list[str], dict[str, list[str]]]:
    """
    Split a NumPy-style docstring into a leading prose block and named sections.

    Returns ``(lead_lines, sections)``, where ``sections`` maps lowercased
    section names to their raw lines.
    """
    text = textwrap.dedent(doc or "").strip("\n")
    if not text:
        return [], {}

    lines = text.splitlines()
    lead: list[str] = []
    sections: dict[str, list[str]] = {}

    i = 0
    while i < len(lines) and not _is_numpy_section_header(lines, i):
        lead.append(lines[i])
        i += 1

    while i < len(lines):
        if not _is_numpy_section_header(lines, i):
            i += 1
            continue

        name = lines[i].strip().lower()
        i += 2

        body: list[str] = []
        while i < len(lines) and not _is_numpy_section_header(lines, i):
            body.append(lines[i])
            i += 1
        sections[name] = body

    return lead, sections


def _paragraphs_from_lines(lines: list[str]) -> list[str]:
    """
    Convert raw docstring lines into Markdown paragraphs separated by blank lines.
    """
    out: list[str] = []
    buf: list[str] = []
    for line in lines:
        if line.strip():
            buf.append(line.strip())
        else:
            if buf:
                out.append(" ".join(buf).strip())
                buf = []
    if buf:
        out.append(" ".join(buf).strip())
    return [p for p in out if p]


def _parse_numpy_parameters(lines: list[str]) -> list[dict[str, str]]:
    """
    Parse a NumPy-style Parameters section into rows with keys:
      - name
      - type
      - desc
    """
    rows: list[dict[str, str]] = []
    i = 0
    n = len(lines)

    while i < n:
        if not lines[i].strip():
            i += 1
            continue

        if ":" in lines[i]:
            name_part, type_part = lines[i].split(":", 1)
            name = name_part.strip()
            typ = type_part.strip()
            i += 1

            desc_lines: list[str] = []
            while i < n:
                line = lines[i]
                if not line.strip():
                    desc_lines.append("")
                    i += 1
                    continue
                if not line.startswith(" ") and ":" in line:
                    break
                desc_lines.append(line.strip())
                i += 1

            desc = " ".join(x for x in desc_lines if x).strip()
            rows.append({"name": name, "type": typ, "desc": desc})
        else:
            i += 1

    return rows


def _signature_parameter_map(func: Any, aliases: dict[str, str] | None = None) -> dict[str, dict[str, str]]:
    """
    Build a parameter metadata map from a callable signature.

    Returned mapping keys are parameter names and values contain:
      - ``type``: display type string (possibly empty)
      - ``default``: ``*required*`` or a rendered default value
    """
    aliases = aliases or {}
    out: dict[str, dict[str, str]] = {}

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return out

    for name, param in sig.parameters.items():
        if name in {"self", "cls"}:
            continue

        if param.annotation is inspect.Signature.empty:
            ann = ""
        elif isinstance(param.annotation, str):
            ann = _expand_type_aliases(param.annotation, aliases)
        else:
            ann = _expand_type_aliases(str(param.annotation), aliases)

        default = "*required*" if param.default is inspect.Signature.empty else f"<code>{param.default!r}</code>"
        out[name] = {"type": ann, "default": default}

    return out


def _parse_numpy_returns(lines: list[str]) -> list[dict[str, str]]:
    """
    Parse a NumPy-style Returns section into rows with keys:
      - type
      - desc
    """
    rows: list[dict[str, str]] = []
    i = 0
    n = len(lines)

    while i < n:
        if not lines[i].strip():
            i += 1
            continue

        typ = lines[i].strip()
        i += 1

        desc_lines: list[str] = []
        while i < n:
            line = lines[i]
            if not line.strip():
                desc_lines.append("")
                i += 1
                continue
            if not line.startswith(" "):
                break
            desc_lines.append(line.strip())
            i += 1

        desc = " ".join(x for x in desc_lines if x).strip()
        rows.append({"type": typ, "desc": desc})

    return rows


def _signature_return_type(func: Any, aliases: dict[str, str] | None = None) -> str:
    """
    Return a display string for a callable return annotation, or ``""`` if none.
    """
    aliases = aliases or {}
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return ""

    ann = sig.return_annotation
    if ann is inspect.Signature.empty:
        return ""

    text = _expand_type_aliases(str(ann), aliases)
    return "" if text == "None" else text


def _callable_for_method_name(cls_obj: type, name: str) -> Any | None:
    obj = getattr(cls_obj, "__dict__", {}).get(name)
    if obj is None:
        return None
    if isinstance(obj, staticmethod):
        return obj.__func__
    if isinstance(obj, classmethod):
        return obj.__func__
    return obj


def _emit_inherited_doc_summary(
    f,
    *,
    fallback_owner_ident: str,
    member_name: str,
    doc_text: str,
    derived_callable: Any | None = None,
) -> None:
    """
    Emit a lightweight inherited-doc summary for an undocumented override.

    Supported sections:
      - lead/body prose
      - Parameters
      - Returns
      - signature-derived parameter types/defaults
      - signature-derived return type when omitted in the docstring

    Any additional recognized NumPy-style sections trigger a note directing the
    reader to the base method for the full documentation.
    """
    lead_lines, sections = _split_numpy_doc_sections(doc_text)

    base_class = fallback_owner_ident.rsplit(".", 1)[-1]
    if member_name == "__init__":
        base_link = f"[`{base_class}()`](api:{fallback_owner_ident}.{member_name})"
    else:
        base_link = f"[`{base_class}.{member_name}()`](api:{fallback_owner_ident}.{member_name})"

    lead_paragraphs = _paragraphs_from_lines(lead_lines)
    for p in lead_paragraphs:
        f.write(p + "\n\n")

    sig_params = _signature_parameter_map(derived_callable) if derived_callable is not None else {}
    params = _parse_numpy_parameters(sections.get("parameters", []))
    if sig_params:
        seen = {row["name"] for row in params}
        for row in params:
            meta = sig_params.get(row["name"], {})
            if not row.get("type"):
                row["type"] = meta.get("type", "")
            row["default"] = meta.get("default", "*required*")
        for name, meta in sig_params.items():
            if name not in seen:
                params.append(
                    {
                        "name": name,
                        "type": meta.get("type", ""),
                        "default": meta.get("default", "*required*"),
                        "desc": "",
                    }
                )

    returns = _parse_numpy_returns(sections.get("returns", []))
    sig_return = _signature_return_type(derived_callable) if derived_callable is not None else ""
    if returns:
        for row in returns:
            if not row.get("type"):
                row["type"] = sig_return
    elif sig_return:
        returns = [{"type": sig_return, "desc": ""}]

    if params:
        f.write("**Parameters**\n\n")
        f.write("| Name | Type | Description | Default |\n")
        f.write("|---|---|---|---|\n")
        for row in params:
            name = row["name"].replace("|", "\\|")
            typ = _type_to_md(_expand_type_aliases(row["type"], {}))
            desc = row["desc"].replace("|", "\\|")
            default = row.get("default", "*required*").replace("|", "\\|")
            f.write(f"| `{name}` | `{typ}` | {desc} | {default} |\n")
        f.write("\n")

    if returns:
        f.write("**Returns**\n\n")
        f.write("| Type | Description |\n")
        f.write("|---|---|\n")
        for row in returns:
            typ = _expand_type_aliases(row["type"], {}).replace("|", "\\|")
            desc = row["desc"].replace("|", "\\|")
            f.write(f"| {typ} | {desc} |\n")
        f.write("\n")

    supported = {"parameters", "returns"}
    extra_sections = [s for s in sections.keys() if s not in supported]
    if extra_sections:
        f.write("!!! note\n\n")
        f.write(f"    Original documentation in {base_link} has more content not rendered here.\n\n")


def _base_method_doc_owner_and_text(owner_ident: str, cls_obj: type, name: str) -> tuple[str, str]:
    """
    Return ``(owner_ident, doc_text)`` for the inherited documentation selected
    for an undocumented override, or ``('', '')`` if none is available.
    """
    if not owner_ident or owner_ident == _qualname_to_ident(cls_obj):
        return "", ""

    for base in cls_obj.__mro__[1:]:
        if _qualname_to_ident(base) != owner_ident:
            continue

        obj = getattr(base, "__dict__", {}).get(name)
        if obj is None or isinstance(obj, property):
            return "", ""

        if isinstance(obj, staticmethod):
            target = obj.__func__
        elif isinstance(obj, classmethod):
            target = obj.__func__
        else:
            target = obj

        doc_text = inspect.getdoc(target) or ""
        if not doc_text.strip():
            return "", ""

        return owner_ident, doc_text

    return "", ""


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
        typ = _type_to_md(_classvar_inner(typ_raw).replace("\n", " ").replace("|", "\\|"), link_names=link_names)

        val = r.get("value")
        if isinstance(val, str) and "property" in val:
            inv_kinds[anchor_id] = "property"
        else:
            inv_kinds[anchor_id] = "variable"
        if val is None or str(val).strip() == "":
            val_s = "*unset*"
        else:
            val_s_raw = str(val).replace("\n", " ").strip()
            if "*" in val_s_raw:
                val_s = val_s_raw
            else:
                val_s = _type_to_md(val_s_raw.replace("|", "\\|"), link_names=link_names)

        doc = (r.get("doc") or "").replace("\n", " ").replace("|", "\\|")

        f.write(f"| {name_cell} | {typ} | {val_s} | {doc} |\n")
    f.write("\n")


def write_class_intro(f, cls_ident: str) -> None:
    """
    Emit the class intro block.

    The intro render includes the class docstring and explicitly includes
    ``__init__`` so the constructor appears first on the page and in the TOC.
    """
    f.write(f"::: {cls_ident}\n")
    f.write("    options:\n")
    f.write("      members:\n")
    f.write("        - __init__\n")
    f.write("      inherited_members: false\n")
    f.write("\n")


def write_class_member_block(f, owner_ident: str, member_name: str) -> None:
    """
    Emit a declared method block for non-constructor methods.

    Methods are always rendered from the derived class object so anchors remain
    unique to the derived page.
    """
    f.write(f"<!-- API_METHOD owner={owner_ident} member={member_name} -->\n\n")
    f.write(f"::: {owner_ident}\n")
    f.write("    options:\n")
    f.write("      members:\n")
    f.write(f"        - {member_name}\n")
    f.write("      inherited_members: false\n\n")


def write_inherited_method_stub(
    f,
    *,
    derived_cls_ident: str,
    method_name: str,
    method_kind: str,
    base_ident: str,
) -> None:
    """
    Emit a lightweight stub for inherited methods that are not declared on the derived class.

    This creates a heading/anchor on the derived class page and links to the method's
    defining class documentation. The link uses the api: scheme and is resolved later
    (internal inventory for loqs.*, external_api_url() fallback for third-party/stdlib).
    """
    anchor_id = f"{derived_cls_ident}.{method_name}"

    kind_label = {
        "static": "static ",
        "class": "class ",
        "instance": " ",
    }.get(method_kind, " ")

    f.write(f"<!-- API_INHERITED_HEADING {anchor_id} -->\n")

    if method_name == "__init__":
        class_display = derived_cls_ident.rsplit(".", 1)[-1]
        f.write(f'## {class_display}() {{: #{anchor_id} }}\n\n')
        if base_ident.startswith("loqs."):
            base_class = base_ident.split(".")[-1]
            f.write(f'Inherited constructor from [`{base_class}()`](api:{base_ident}.{method_name}).\n\n')
        else:
            f.write(f'Inherited constructor from [](api:{base_ident}).\n\n')
        return

    f.write(f'## {method_name} {{: #{anchor_id} }}\n\n')

    if base_ident.startswith("loqs."):
        base_link_text = f"{base_ident.split('.')[-1]}.{method_name}"
        f.write(f'Inherited {kind_label}method from [{base_link_text}](api:{base_ident}.{method_name}).\n\n')
    else:
        f.write(f'Inherited {kind_label}method from [](api:{base_ident}).\n\n')


def write_module_functions_block(f, mod_ident: str, funcs: list[str]) -> None:
    if not funcs:
        return

    for i, fn in enumerate(funcs):
        f.write(f"<!-- API_MODULE_MEMBERS owner={mod_ident} -->\n\n")
        f.write(f"::: {mod_ident}\n")
        f.write("    options:\n")
        f.write("      heading_level: 3\n")
        f.write("      members:\n")
        f.write(f"        - {fn}\n")
        f.write("      inherited_members: false\n\n")

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

    f.write("| Name | Type | Value | Doc |\n")
    f.write("|---|---|---|---|\n")
    for r in rows:
        nm = r["name"]
        anchor_id = f"{mod_ident}.{nm}"
        inv_objects[anchor_id] = f"{page_url}#{anchor_id}"
        inv_kinds[anchor_id] = (r.get("kind") or "variable").replace(" ", "_")

        name_cell = f'<a id="{anchor_id}"></a>`{nm}`'
        typ = _type_to_md((r.get("type") or "").replace("\n", " ").replace("|", "\\|"), link_names=link_names)

        val = r.get("value")
        if val is None or str(val).strip() == "":
            val_s = "*unset*"
        else:
            val_s_raw = str(val).replace("\n", " ").strip()
            val_s = _type_to_md(val_s_raw.replace("|", "\\|"), link_names=link_names)

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
    inv_kinds: dict[str, str],
    link_names: set[str] | None = None,
) -> None:
    with mkdocs_gen_files.open(path, "w") as f:
        f.write(f"# `{mod_ident}`\n\n")

        f.write(f"<!-- API_TOC_REMOVE {mod_ident} -->\n\n")

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

        write_class_intro(f, cls_ident)

        # If the constructor itself is undocumented but a base constructor has
        # docs, append the lightweight inherited-doc summary directly after the
        # intro block so it sits with the rendered constructor.
        if cls_obj is not None and "__init__" in methods:
            owner = owner_override.get("__init__", cls_ident)
            if owner != cls_ident:
                fallback_owner_ident, fallback_doc_text = _base_method_doc_owner_and_text(owner, cls_obj, "__init__")
                derived_callable = _callable_for_method_name(cls_obj, "__init__")
                if fallback_owner_ident and fallback_doc_text:
                    _emit_inherited_doc_summary(
                        f,
                        fallback_owner_ident=fallback_owner_ident,
                        member_name="__init__",
                        doc_text=fallback_doc_text,
                        derived_callable=derived_callable,
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
            )

        other_methods = [m for m in methods if m != "__init__"]

        if other_methods or inherited_method_stubs:
            f.write("\n---\n\n")

            inherited_method_stubs = inherited_method_stubs or {}
            declared_set = set(other_methods)

            def _method_sort_key(name: str) -> tuple[int, str]:
                return name.lower(),

            all_names = sorted(set(other_methods) | set(inherited_method_stubs), key=lambda name: name.lower())

            for m in all_names:
                if m in declared_set:
                    write_class_member_block(f, cls_ident, m)

                    owner = owner_override.get(m, cls_ident)
                    if cls_obj is not None and owner != cls_ident:
                        fallback_owner_ident, fallback_doc_text = _base_method_doc_owner_and_text(owner, cls_obj, m)
                        derived_callable = _callable_for_method_name(cls_obj, m)
                        if fallback_owner_ident and fallback_doc_text:
                            _emit_inherited_doc_summary(
                                f,
                                fallback_owner_ident=fallback_owner_ident,
                                member_name=m,
                                doc_text=fallback_doc_text,
                                derived_callable=derived_callable,
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

                f.write("\n---\n\n")


def main() -> None:
    nav = mkdocs_gen_files.Nav()
    inv_objects: dict[str, str] = {}
    inv_kinds: dict[str, str] = {}

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
        inv_kinds[mod_ident] = "module"
        for fn in funcs:
            inv_objects[f"{mod_ident}.{fn}"] = f"{mod_page_url}#{mod_ident}.{fn}"
            inv_kinds[f"{mod_ident}.{fn}"] = "function"
        for cls_name in classes:
            inv_kinds[f"{mod_ident}.{cls_name}"] = "class"
        mod_link_names = {r["name"] for r in rows if isinstance(r.get("name"), str)}

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
            inv_kinds=inv_kinds,
            link_names=mod_link_names
        )

        for cls_name in classes:
            cls_ident = f"{mod_ident}.{cls_name}"
            cls_page = Path(*mod_parts) / f"{cls_name}.md"
            nav[(*nav_key, cls_name)] = cls_page.as_posix()

            (
                cls_obj,
                methods,
                owner_override,
                toc_remove_anchors,
            ) = class_members_from_introspection(cls_name, mod_ident)

            declared = set(getattr(cls_obj, "__dict__", {}).keys()) if cls_obj is not None else set()
            inherited_missing: dict[str, tuple[str, str]] = {}
            if cls_obj is not None:
                inherited_missing = inherited_only_methods(cls_obj, declared=declared)

            cls_page_url = mod_page_url + f"{cls_name}/"
            inv_objects[cls_ident] = f"{cls_page_url}#{cls_ident}"

            for m in methods:
                inv_objects[f"{cls_ident}.{m}"] = f"{cls_page_url}#{cls_ident}.{m}"
                inv_kinds[f"{cls_ident}.{m}"] = "method"
            for m in inherited_missing.keys():
                inv_objects[f"{cls_ident}.{m}"] = f"{cls_page_url}#{cls_ident}.{m}"
                inv_kinds[f"{cls_ident}.{m}"] = "method"

            if cls_obj is not None:
                var_rows = class_var_rows_with_mro(py, cls_obj)

                try:
                    tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
                    aliases = _collect_import_aliases(tree)
                except SyntaxError:
                    aliases = {}

                mod_link_names = {r["name"] for r in rows if isinstance(r.get("name"), str)}

                prop_rows = property_rows_from_introspection(cls_obj, owner_ident=cls_ident, aliases=aliases)

                by_name = {r["name"]: r.copy() for r in prop_rows}
                for r in var_rows:
                    prev = by_name.get(r["name"])
                    if prev is not None:
                        merged = r.copy()
                        if not (merged.get("doc") or "").strip() and (prev.get("doc") or "").strip():
                            merged["doc"] = prev["doc"]
                        if not (merged.get("type") or "").strip() and (prev.get("type") or "").strip():
                            merged["type"] = prev["type"]
                        if not (merged.get("value") or "").strip() and (prev.get("value") or "").strip():
                            merged["value"] = prev["value"]
                        by_name[r["name"]] = merged
                    else:
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
                cls_obj=cls_obj,
                var_rows=var_rows,
                inherited_method_stubs=inherited_missing,
                methods=methods,
                owner_override=owner_override,
                toc_remove_anchors=toc_remove_anchors,
                inv_objects=inv_objects,
                inv_kinds=inv_kinds,
                link_names=mod_link_names,
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
        json.dump({"objects": inv_objects, "suffix_index": suffix_index, "kinds": inv_kinds}, f, indent=2, sort_keys=True)

    disk_path = REPO_ROOT / "docs" / f"_{INVENTORY_PATH}"
    disk_path.write_text(
        json.dumps({"objects": inv_objects, "suffix_index": suffix_index, "kinds": inv_kinds}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


main()