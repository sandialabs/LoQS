from __future__ import annotations

from pathlib import Path
import ast
import mkdocs_gen_files

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "loqs"


def _is_public_method(name: str) -> bool:
    return not name.startswith("_") and not name.startswith("__")


def _is_public_var(name: str) -> bool:
    return not name.startswith("_") and not name.startswith("__")


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


def _decorator_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for d in fn.decorator_list:
        s = _unparse(d).strip()
        if s:
            names.add(s.split(".")[-1])
    return names


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
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and _is_public_var(tgt.id):
                    kind = "type variable" if _is_typevar_call(node.value) else "variable"
                    rows.append({"name": tgt.id, "kind": kind, "doc": doc})

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and _is_public_var(node.target.id):
                doc = _doc_hint_from_next_stmt(body, i)
                kind = "type alias" if _is_typealias_ann(node.annotation) else "variable"
                rows.append({"name": node.target.id, "kind": kind, "doc": doc})

    classes.sort(key=str.lower)
    funcs.sort(key=str.lower)

    by_name: dict[str, dict] = {}
    for r in rows:
        by_name[r["name"]] = r
    rows = sorted(by_name.values(), key=lambda d: d["name"].lower())

    return classes, funcs, rows


def class_api(py_file: Path, class_name: str) -> tuple[list[dict], list[str], list[str], list[str]]:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return [], [], [], []

    cls: ast.ClassDef | None = None
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            cls = n
            break
    if cls is None:
        return [], [], [], []

    var_rows: list[dict] = []
    inst: list[str] = []
    clsm: list[str] = []
    stat: list[str] = []

    body = cls.body
    for i, node in enumerate(body):
        if isinstance(node, ast.Assign):
            doc = _doc_hint_from_next_stmt(body, i)
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and _is_public_var(tgt.id):
                    kind = "type variable" if _is_typevar_call(node.value) else "variable"
                    var_rows.append({"name": tgt.id, "kind": kind, "doc": doc})

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and _is_public_var(node.target.id):
                doc = _doc_hint_from_next_stmt(body, i)
                kind = "type alias" if _is_typealias_ann(node.annotation) else "variable"
                var_rows.append({"name": node.target.id, "kind": kind, "doc": doc})

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if not _is_public_method(name):
                continue
            decs = _decorator_names(node)
            if "staticmethod" in decs:
                stat.append(name)
            elif "classmethod" in decs:
                clsm.append(name)
            else:
                inst.append(name)

    by_name: dict[str, dict] = {}
    for r in var_rows:
        by_name[r["name"]] = r
    var_rows = sorted(by_name.values(), key=lambda d: d["name"].lower())

    inst.sort(key=str.lower)
    clsm.sort(key=str.lower)
    stat.sort(key=str.lower)
    return var_rows, inst, clsm, stat


def write_members_table(f, rows: list[dict]) -> None:
    if not rows:
        return
    f.write("| Name | Type | Doc |\n")
    f.write("|---|---|---|\n")
    for r in rows:
        name = f"`{r['name']}`"
        kind = r["kind"]
        doc = (r.get("doc") or "").replace("\n", " ").replace("|", "\\|")
        f.write(f"| {name} | {kind} | {doc} |\n")
    f.write("\n")


def write_obj(f, ident: str, *, members: bool) -> None:
    f.write(f"::: {ident}\n")
    f.write("    options:\n")
    f.write(f"      members: {str(members).lower()}\n")
    f.write("\n")


def write_class_member_block(f, cls_ident: str, member_name: str) -> None:
    f.write(f"<!-- API_METHOD owner={cls_ident} member={member_name} -->\n\n")
    f.write(f"::: {cls_ident}\n")
    f.write("    options:\n")
    f.write("      members:\n")
    f.write(f"        - {member_name}\n")
    f.write("      inherited_members: false\n")


def write_module_page(path: Path, title: str, mod_ident: str, *, rows: list[dict], funcs: list[str], classes: list[str]) -> None:
    with mkdocs_gen_files.open(path, "w") as f:
        f.write(f"# `{title}`\n\n")

        if rows:
            f.write("## Members\n\n")
            write_members_table(f, rows)

        if funcs:
            f.write("## Functions\n\n")
            for fn in funcs:
                write_obj(f, f"{mod_ident}.{fn}", members=True)

        if classes:
            f.write("## Classes\n\n")
            for cls in classes:
                f.write(f"- [`{cls}`]({cls}/)\n")


def write_class_page(
    path: Path,
    title: str,
    cls_ident: str,
    *,
    var_rows: list[dict],
    instance_methods: list[str],
    class_methods: list[str],
    static_methods: list[str],
) -> None:
    with mkdocs_gen_files.open(path, "w") as f:
        f.write(f"# `{title}`\n\n")

        if var_rows:
            f.write("## Members\n\n")
            write_members_table(f, var_rows)

        # These headings remain (TOC-friendly); no per-method H3s
        if instance_methods:
            f.write("## Instance methods\n\n")
            for m in instance_methods:
                write_class_member_block(f, cls_ident, m)

        if class_methods:
            f.write("## Class methods\n\n")
            for m in class_methods:
                write_class_member_block(f, cls_ident, m)

        if static_methods:
            f.write("## Static methods\n\n")
            for m in static_methods:
                write_class_member_block(f, cls_ident, m)


def main() -> None:
    nav = mkdocs_gen_files.Nav()

    nav[("loqs",)] = "loqs/index.md"
    with mkdocs_gen_files.open("loqs/index.md", "w") as f:
        f.write("# `loqs`\n\n")
        f.write("Package reference. Use the sidebar to browse.\n")

    for py in sorted(PKG_DIR.rglob("*.py")):
        rel = py.relative_to(PKG_DIR)
        parts = rel.with_suffix("").parts
        is_pkg = py.name == "__init__.py"

        if is_pkg:
            mod_parts = ("loqs",) + parts[:-1]
            mod_ident = "loqs" + ("" if len(mod_parts) == 1 else "." + ".".join(mod_parts[1:]))
            page = Path(*mod_parts) / "index.md"
            nav_key = mod_parts
            label = mod_parts[-1]

            classes: list[str] = []
            funcs: list[str] = []
            rows: list[dict] = []
        else:
            mod_parts = ("loqs",) + parts
            mod_ident = "loqs." + ".".join(mod_parts[1:])
            page = Path(*mod_parts) / "index.md"
            nav_key = mod_parts
            label = mod_parts[-1]

            classes, funcs, rows = module_public_api(py)

        nav[nav_key] = page.as_posix()
        write_module_page(page, title=label, mod_ident=mod_ident, rows=rows, funcs=funcs, classes=classes)

        for cls in classes:
            cls_ident = f"{mod_ident}.{cls}"
            cls_page = Path(*mod_parts) / f"{cls}.md"
            nav[(*nav_key, cls)] = cls_page.as_posix()

            var_rows, inst, clsm, stat = class_api(py, cls)
            write_class_page(
                cls_page,
                title=cls,
                cls_ident=cls_ident,
                var_rows=var_rows,
                instance_methods=inst,
                class_methods=clsm,
                static_methods=stat,
            )

    with mkdocs_gen_files.open("index.md", "w") as f:
        f.write("# API Reference\n\n")
        f.write("Use the sidebar to browse.\n")

    with mkdocs_gen_files.open("SUMMARY.md", "w") as f:
        f.write("* [API Reference](index.md)\n")
        for line in nav.build_literate_nav():
            f.write("  " + line)


main()