from __future__ import annotations

"""
Build-time generator for the LoQS API reference site.

Goals
-----
- Discover package modules under `loqs/`.
- Generate module/class Markdown pages for the dedicated API-reference site.
- Build a complete API inventory used by both the API site and the main docs site.
- Keep generation orchestration small by delegating introspection and rendering to
  focused helper modules.

This file is intentionally the coordinator layer:
- source discovery
- page/url layout
- inventory assembly
- generated navigation
- inventory serialization to both site output and `docs/_api_inventory.json`
"""

import ast
import json
from pathlib import Path

import mkdocs_gen_files

from docs_scripts.api_inventory import build_suffix_index
from docs_scripts.ref_introspect import (
    K_CLASS,
    K_FUNCTION,
    K_METHOD,
    K_MODULE,
    class_doc_plan,
    class_var_info_map_from_ast,
    class_var_rows_with_mro,
    collect_import_aliases,
    module_public_api,
    property_rows_from_introspection,
    var_sort_key,
)

from docs_scripts.ref_render import write_class_page, write_module_page


REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "loqs"
INVENTORY_PATH = "api_inventory.json"


def api_page_url(mod_parts: tuple[str, ...]) -> str:
    return "/" + "/".join(mod_parts) + "/"


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
        inv_kinds[mod_ident] = K_MODULE

        for fn in funcs:
            inv_objects[f"{mod_ident}.{fn}"] = f"{mod_page_url}#{mod_ident}.{fn}"
            inv_kinds[f"{mod_ident}.{fn}"] = K_FUNCTION

        for cls_name in classes:
            inv_kinds[f"{mod_ident}.{cls_name}"] = K_CLASS

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
            link_names=mod_link_names,
        )

        for cls_name in classes:
            cls_ident = f"{mod_ident}.{cls_name}"
            cls_page = Path(*mod_parts) / f"{cls_name}.md"
            nav[(*nav_key, cls_name)] = cls_page.as_posix()

            plan = class_doc_plan(cls_name, mod_ident)
            cls_page_url = mod_page_url + f"{cls_name}/"
            inv_objects[cls_ident] = f"{cls_page_url}#{cls_ident}"
            inv_kinds[cls_ident] = K_CLASS

            if plan.cls_obj is not None:
                var_rows = class_var_rows_with_mro(py, plan.cls_obj)

                try:
                    tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
                    aliases = collect_import_aliases(tree)
                except SyntaxError:
                    aliases = {}

                prop_rows = property_rows_from_introspection(plan.cls_obj, owner_ident=cls_ident, aliases=aliases)

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
                var_rows = sorted(by_name.values(), key=var_sort_key)
            else:
                derived_map = class_var_info_map_from_ast(py, cls_name, owner_ident=cls_ident)
                var_rows = sorted(derived_map.values(), key=var_sort_key)

            write_class_page(
                cls_page,
                title=cls_name,
                cls_ident=cls_ident,
                page_url=cls_page_url,
                cls_obj=plan.cls_obj,
                var_rows=var_rows,
                inherited_method_stubs=plan.inherited_missing,
                methods=plan.methods,
                owner_override=plan.owner_override,
                toc_remove_anchors=plan.toc_remove_anchors,
                inv_objects=inv_objects,
                inv_kinds=inv_kinds,
                link_names=mod_link_names,
            )

            for m in plan.methods:
                inv_objects[f"{cls_ident}.{m}"] = f"{cls_page_url}#{cls_ident}.{m}"
                inv_kinds[f"{cls_ident}.{m}"] = K_METHOD

            for m in plan.inherited_missing.keys():
                inv_objects[f"{cls_ident}.{m}"] = f"{cls_page_url}#{cls_ident}.{m}"
                inv_kinds[f"{cls_ident}.{m}"] = K_METHOD

    with mkdocs_gen_files.open("index.md", "w") as f:
        f.write("# API Reference\n\n")
        f.write("Use the sidebar to browse.\n")

    import os
    from urllib.parse import urlparse
    canonical_url = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
    home_url = "/"
    if canonical_url:
        path = urlparse(canonical_url).path.rstrip("/")
        if path:
            home_url = f"{path}/"

    with mkdocs_gen_files.open("SUMMARY.md", "w") as f:
        f.write(f"* [Home]({home_url})\n")
        f.write("* [API Reference](index.md)\n")
        for line in nav.build_literate_nav():
            f.write("  " + line)

    suffix_index = build_suffix_index(inv_objects, package="loqs")
    inv_json = {"objects": inv_objects, "suffix_index": suffix_index, "kinds": inv_kinds}

    with mkdocs_gen_files.open(INVENTORY_PATH, "w") as f:
        json.dump(inv_json, f, indent=2, sort_keys=True)

    disk_path = REPO_ROOT / "docs" / f"_{INVENTORY_PATH}"
    disk_path.write_text(json.dumps(inv_json, indent=2, sort_keys=True), encoding="utf-8")


if __name__ in ("__main__", "<run_path>"):  # "<run_path>" is what mkdocs-gen-files' runpy.run_path(...) uses
    main()