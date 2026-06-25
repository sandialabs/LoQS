from __future__ import annotations

"""
Collection and normalization helpers for generated API reference pages.

Goals
-----
- Gather public API structure from source files and runtime introspection.
- Treat `__init__.py` package surfaces as real API surfaces.
- Collect module-level classes/functions/variables/type aliases/type variables.
- Collect class member variables and properties, including inheritance-aware fallbacks.
- Classify documented declared methods and inherited-only methods.
- Track which base owner should provide documentation when a derived override has
  no own docstring.

This module intentionally does not write Markdown pages. It returns normalized
metadata that `ref_render.py` and `gen_ref_pages.py` can use to build pages and
inventories in a single, predictable way.
"""

import ast
import importlib
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


K_MODULE = "module"
K_CLASS = "class"
K_FUNCTION = "function"
K_METHOD = "method"
K_PROPERTY = "property"
K_VARIABLE = "variable"
K_TYPE_ALIAS = "type_alias"
K_TYPE_VARIABLE = "type_variable"


def _is_public_method(name: str) -> bool:
    return not name.startswith("_") and not name.startswith("__")


def _is_documented_class_method(name: str) -> bool:
    """
    Methods documented on class pages.

    Public methods are included, and `__init__` is included explicitly so
    constructors appear in the generated API like AutoAPI-style class docs.
    """
    return name == "__init__" or _is_public_method(name)


def _is_public_var(name: str) -> bool:
    return not name.startswith("_") and not name.startswith("__")


def _is_public_property(name: str) -> bool:
    return _is_public_method(name)


_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def var_sort_key(row: dict) -> tuple[int, int | str, str]:
    name = (row.get("name") or "")
    sort_value = (row.get("sort_value") or "").strip()

    if sort_value:
        try:
            return (0, int(sort_value), name.lower())
        except Exception:
            pass

    return (1 if _ALL_CAPS_RE.fullmatch(name) else 2, name.lower(), name.lower())


def unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def doc_hint_from_next_stmt(body: list[ast.stmt], i: int) -> str:
    if i + 1 >= len(body):
        return ""
    nxt = body[i + 1]
    if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) and isinstance(nxt.value.value, str):
        return nxt.value.value.strip().splitlines()[0]
    return ""


def is_typevar_call(value: ast.AST | None) -> bool:
    if not isinstance(value, ast.Call):
        return False
    fn = value.func
    if isinstance(fn, ast.Name):
        return fn.id == "TypeVar"
    if isinstance(fn, ast.Attribute):
        return fn.attr == "TypeVar"
    return False


def is_typealias_ann(annotation: ast.AST | None) -> bool:
    ann = unparse(annotation).strip()
    return bool(ann) and ann.split(".")[-1] == "TypeAlias"


def qualname_to_ident(obj: Any) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def has_own_doc(obj: Any) -> bool:
    """
    Return whether an object defines its own docstring directly, without
    inheriting one through `inspect.getdoc` MRO fallback.
    """
    return bool((getattr(obj, "__doc__", None) or "").strip())


def method_owner_for_docs(cls: type, name: str) -> type:
    if name not in getattr(cls, "__dict__", {}):
        return cls

    obj = cls.__dict__.get(name)
    if obj is None:
        return cls

    if isinstance(obj, property):
        if obj.fget is not None and has_own_doc(obj.fget):
            return cls
    else:
        if has_own_doc(obj):
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


def collect_import_aliases(tree: ast.AST) -> dict[str, str]:
    """
    Collect local import aliases from a module AST.

    Includes imports nested under if/try blocks (e.g. TYPE_CHECKING patterns),
    since the full tree is walked.
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


def expand_type_aliases(type_s: str, aliases: dict[str, str]) -> str:
    """
    Expand imported names/aliases inside an annotation string, then normalize
    verbose module prefixes for display.
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


def module_public_api(py_file: Path) -> tuple[list[str], list[str], list[dict]]:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
        aliases = collect_import_aliases(tree)
    except SyntaxError:
        return [], [], []

    classes: list[str] = []
    funcs: list[str] = []
    rows: list[dict] = []

    def _full_doc_from_next_stmt(body: list[ast.stmt], i: int) -> str:
        if i + 1 >= len(body):
            return ""
        nxt = body[i + 1]
        if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) and isinstance(nxt.value.value, str):
            return nxt.value.value.strip()
        return ""

    body = tree.body
    for i, node in enumerate(body):
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            classes.append(node.name)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_method(node.name):
            funcs.append(node.name)

        elif isinstance(node, ast.Assign):
            doc = _full_doc_from_next_stmt(body, i)
            value_s = unparse(node.value).strip() if node.value is not None else ""

            typevar_bound = ""
            if is_typevar_call(node.value) and isinstance(node.value, ast.Call):
                for kw in node.value.keywords:
                    if kw.arg == "bound":
                        typevar_bound = expand_type_aliases(unparse(kw.value).strip(), aliases).strip("'\"")
                        break

            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and _is_public_var(tgt.id):
                    is_tv = is_typevar_call(node.value)
                    kind = K_TYPE_VARIABLE if is_tv else K_VARIABLE
                    rows.append(
                        {
                            "name": tgt.id,
                            "kind": kind,
                            "type": "TypeVar" if is_tv else "",
                            "value": value_s,
                            "doc": doc,
                            "typevar_bound": typevar_bound if is_tv else "",
                            "sort_value": value_s,
                        }
                    )

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and _is_public_var(node.target.id):
                doc = _full_doc_from_next_stmt(body, i)
                ann_s = expand_type_aliases(unparse(node.annotation).strip(), aliases)
                value_s = expand_type_aliases(unparse(node.value).strip() if node.value is not None else "", aliases)

                is_alias = is_typealias_ann(node.annotation)
                kind = K_TYPE_ALIAS if is_alias else K_VARIABLE

                rows.append(
                    {
                        "name": node.target.id,
                        "kind": kind,
                        "type": "TypeAlias" if is_alias else ann_s,
                        "value": value_s,
                        "doc": doc,
                        "typevar_bound": "",
                        "sort_value": value_s,
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
    order: list[str] = []
    for r in rows:
        name = r["name"]
        prev = by_name.get(name)
        if prev is None:
            by_name[name] = r
            order.append(name)
        elif score(r) > score(prev):
            by_name[name] = r

    rows = [by_name[name] for name in order]

    return classes, funcs, rows


def classvar_inner(type_s: str) -> str:
    s = (type_s or "").strip()
    if not s or "ClassVar[" not in s:
        return s

    start = s.find("ClassVar[")
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


def class_var_info_map_from_ast(py_file: Path, class_name: str, *, owner_ident: str) -> dict[str, dict]:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
        aliases = collect_import_aliases(tree)
    except SyntaxError:
        return {}

    def _full_doc_from_next_stmt(body: list[ast.stmt], i: int) -> str:
        if i + 1 >= len(body):
            return ""
        nxt = body[i + 1]
        if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) and isinstance(nxt.value.value, str):
            return nxt.value.value.strip()
        return ""

    cls: ast.ClassDef | None = None
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            cls = n
            break
    if cls is None:
        return {}

    out: dict[str, dict] = {}
    body = cls.body

    is_enum_like = any(
        (isinstance(base, ast.Name) and base.id.endswith("Enum"))
        or (isinstance(base, ast.Attribute) and base.attr.endswith("Enum"))
        for base in cls.bases
    )

    for i, node in enumerate(body):
        if isinstance(node, ast.Assign):
            doc = _full_doc_from_next_stmt(body, i)
            value_s = unparse(node.value).strip() if node.value is not None else ""
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and _is_public_var(tgt.id):
                    out[tgt.id] = {
                        "name": tgt.id,
                        "type": "",
                        "owner": owner_ident,
                        "value": value_s,
                        "doc": doc,
                        "sort_value": value_s if is_enum_like else "",
                    }

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and _is_public_var(node.target.id):
                doc = _full_doc_from_next_stmt(body, i)
                ann_s = expand_type_aliases(unparse(node.annotation).strip(), aliases)
                value_s = expand_type_aliases(unparse(node.value).strip() if node.value is not None else "", aliases)
                out[node.target.id] = {
                    "name": node.target.id,
                    "type": ann_s,
                    "owner": owner_ident,
                    "value": value_s,
                    "doc": doc,
                    "sort_value": value_s if is_enum_like else "",
                }

    return out


def class_var_rows_with_mro(derived_py_file: Path, cls_obj: type) -> list[dict]:
    derived_ident = qualname_to_ident(cls_obj)
    derived_map = class_var_info_map_from_ast(derived_py_file, cls_obj.__name__, owner_ident=derived_ident)

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

        base_ident = qualname_to_ident(base)
        base_map = class_var_info_map_from_ast(base_file, base.__name__, owner_ident=base_ident)
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

        def score_row(r: dict) -> tuple[int, int, int]:
            return (
                1 if (r.get("value") or "").strip() else 0,
                1 if (r.get("type") or "").strip() else 0,
                1 if (r.get("doc") or "").strip() else 0,
            )

        for name, brow in base_map.items():
            if name in derived_map:
                continue

            prev = inherited_map.get(name)
            if prev is None or score_row(brow) > score_row(prev):
                inherited_map[name] = brow

    merged: dict[str, dict] = {}
    merged.update(inherited_map)
    merged.update(derived_map)

    return sorted(merged.values(), key=var_sort_key)


def property_rows_from_introspection(cls_obj: type, *, owner_ident: str, aliases: dict[str, str]) -> list[dict]:
    """
    Build member-variable rows for @property descriptors.
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
        typ = expand_type_aliases(typ, aliases)

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
            row_owner = qualname_to_ident(doc_owner)

        rows.append(
            {
                "name": name,
                "type": typ,
                "owner": row_owner,
                "value": val_s,
                "doc": doc,
            }
        )

    return sorted(rows, key=var_sort_key)


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

            out[name] = (kind, qualname_to_ident(base))
            break

    return dict(sorted(out.items(), key=lambda item: (0 if item[0] == "__init__" else 1, item[0].lower())))


@dataclass(frozen=True)
class ClassDocPlan:
    """
    Documentation plan for one generated class page.
    """

    cls_obj: type | None
    methods: list[str]
    owner_override: dict[str, str]
    toc_remove_anchors: list[str]
    inherited_missing: dict[str, tuple[str, str]]


def class_doc_plan(class_name: str, mod_ident: str) -> ClassDocPlan:
    try:
        mod = importlib.import_module(mod_ident)
    except Exception:
        return ClassDocPlan(None, [], {}, [], {})

    cls = getattr(mod, class_name, None)
    if cls is None or not isinstance(cls, type):
        return ClassDocPlan(None, [], {}, [], {})

    toc_remove_anchors: list[str] = [qualname_to_ident(cls)]
    for base in cls.__mro__[1:]:
        if base is object:
            continue
        if getattr(base, "__module__", "") == "builtins":
            continue
        toc_remove_anchors.append(qualname_to_ident(base))

    methods: list[str] = []
    owner_override: dict[str, str] = {}

    for name, val in getattr(cls, "__dict__", {}).items():
        if not _is_documented_class_method(name):
            continue

        if isinstance(val, (staticmethod, classmethod)) or inspect.isfunction(val):
            methods.append(name)
        else:
            continue

        doc_owner = method_owner_for_docs(cls, name)
        if getattr(doc_owner, "__module__", "") == "builtins":
            doc_owner = cls
        owner_override[name] = qualname_to_ident(doc_owner)

    methods.sort(key=lambda name: (0 if name == "__init__" else 1, name.lower()))

    declared = set(getattr(cls, "__dict__", {}).keys())
    inherited_missing = inherited_only_methods(cls, declared=declared)

    return ClassDocPlan(
        cls_obj=cls,
        methods=methods,
        owner_override=owner_override,
        toc_remove_anchors=toc_remove_anchors,
        inherited_missing=inherited_missing,
    )