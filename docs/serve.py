#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path


def pick_port(host: str, start: int = 8000, max_tries: int = 200) -> int:
    """Find a free TCP port on *host* starting at *start*."""
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"Could not find a free port on {host} in range [{start}, {start + max_tries})")


def run_cmd(cmd: list[str], cwd: Path) -> None:
    """
    Execute *cmd* in *cwd* with DISABLE_MKDOCS_2_WARNING=true set.
    Abort on non-zero return.

    Suppress known benign MkDocs warnings and re-color only the leading WARNING
    token in yellow for readable console output.
    """
    env = os.environ.copy()
    env["DISABLE_MKDOCS_2_WARNING"] = "true"
    env["NO_MKDOCS_2_WARNING"] = "true"
    env.setdefault("PYTHONUNBUFFERED", "1")

    suppress_res = [
        re.compile(
            r"contains a link '#fnref:[^']+', but there is no such anchor on this page",
            re.IGNORECASE,
        ),
        re.compile(
            r"mkdocs_autorefs:\s+Multiple primary URLs found for",
            re.IGNORECASE,
        ),
    ]

    yellow = "\033[33m"
    reset = "\033[0m"

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        if any(rx.search(line) for rx in suppress_res):
            continue

        line_out = re.sub(
            r"^(\s*)(WARNING)(\s*-?\s*)",
            rf"\1{yellow}\2{reset}\3",
            line.rstrip("\n"),
            count=1,
        )
        print(line_out)

    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def copytree_into(src: Path, dst: Path) -> None:
    """Recursively copy *src* into *dst* (overwrites files)."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for fn in files:
            s = Path(root) / fn
            d = dst / rel / fn
            shutil.copy2(s, d)


def _rm_rf(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def looks_like_jupytext_markdown(md: Path) -> bool:
    """
    Heuristic: treat a Markdown file as a Jupytext notebook if it starts with
    YAML front matter containing both 'jupyter:' and 'kernelspec:'.
    """
    try:
        text = md.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False

    if not text.startswith("---\n"):
        return False

    end = text.find("\n---\n", 4)
    if end == -1:
        return False

    front_matter = text[4:end]
    return "jupyter:" in front_matter and "kernelspec:" in front_matter


def convert_markdown_notebooks_to_ipynb(src_dir: Path, dst_dir: Path, cwd: Path) -> tuple[int, int]:
    """
    Convert Markdown notebooks under *src_dir* to .ipynb files under *dst_dir*,
    preserving relative paths.

    Only converts files that look like Jupytext notebook Markdown.

    Returns:
        (converted_count, skipped_count)
    """
    if not src_dir.exists():
        return 0, 0

    converted = 0
    skipped = 0

    for md in sorted(src_dir.rglob("*.md")):
        if not looks_like_jupytext_markdown(md):
            skipped += 1
            continue

        rel = md.relative_to(src_dir).with_suffix(".ipynb")
        out = dst_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        run_cmd(
            [sys.executable, "-m", "jupytext", "--to", "ipynb", str(md), "-o", str(out)],
            cwd=cwd,
        )
        converted += 1

    return converted, skipped


def _serve(
    site_merged: Path,
    host: str,
    port: int,
    port_start: int,
    port_tries: int,
    ref_mount: str,
    lite_mount: str,
    lite_enabled: bool,
) -> None:
    port = port or pick_port(host, start=port_start, max_tries=port_tries)

    os.chdir(site_merged)
    httpd = http.server.ThreadingHTTPServer((host, port), http.server.SimpleHTTPRequestHandler)

    print(f"Merged site: http://{host}:{port}/")
    print(f"API ref:     http://{host}:{port}/{ref_mount}/")
    if lite_enabled:
        print(f"JupyterLite: http://{host}:{port}/{lite_mount}/")
    print("Ctrl+C to stop (re-run script to rebuild).")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build main docs + API reference + optional JupyterLite, merge into one site, optionally serve it."
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 = find port starting at 8000")
    ap.add_argument("--port-start", type=int, default=8000)
    ap.add_argument("--port-tries", type=int, default=200)
    ap.add_argument("--docs-config", default="mkdocs.yml")
    ap.add_argument("--ref-config", default="mkdocs-api-ref.yml")
    ap.add_argument("--ref-mount", default="reference", help="mount point for reference site")

    ap.add_argument(
        "--build-only",
        action="store_true",
        help="build+merge only (no server); defaults output to ./site",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output directory for merged site (default: temp dir; or ./site with --build-only)",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="delete output directory before building (recommended with --build-only)",
    )

    # JupyterLite options
    ap.add_argument("--no-lite", action="store_true", help="skip JupyterLite build")
    ap.add_argument("--lite-config", default="jupyterlite/jupyter_lite_config.json")
    ap.add_argument("--lite-mount", default="lite", help="mount point for JupyterLite site")
    ap.add_argument(
        "--notebooks-dir",
        default="notebooks",
        help="directory containing canonical Jupytext Markdown notebooks",
    )
    ap.add_argument(
        "--lite-files-dir",
        default="jupyterlite/files",
        help="directory where generated .ipynb notebooks are staged for JupyterLite",
    )
    ap.add_argument(
        "--no-lite-notebooks",
        action="store_true",
        help="do not generate .ipynb notebooks via Jupytext before JupyterLite build",
    )
    ap.add_argument(
        "--keep-lite-files",
        action="store_true",
        help="do not clean the staged JupyterLite notebook files directory before regeneration",
    )

    args = ap.parse_args()

    docs_root = Path(__file__).resolve().parent
    project_root = docs_root.parent

    docs_cfg = (project_root / args.docs_config).resolve()
    ref_cfg = (project_root / args.ref_config).resolve()

    lite_cfg = (docs_root / args.lite_config).resolve()
    notebooks_dir = (docs_root / args.notebooks_dir).resolve()
    lite_files_dir = (docs_root / args.lite_files_dir).resolve()
    
    # The inventory file is written by gen_ref_pages.py to disk at this location
    # so BOTH builds can resolve api: links with progressive qualification.
    inv_disk = docs_root / "_api_inventory.json"

    # Output selection:
    # - serve mode: default to temp dir unless --out provided
    # - build-only: default to docs_root/site unless --out provided
    if args.out is None:
        out_dir = (docs_root / "site") if args.build_only else None
    else:
        out_dir = (docs_root / args.out).resolve()

    def build_into(base: Path) -> Path:
        site_docs = base / "site-docs"
        site_ref = base / "site-ref"
        site_lite = base / "site-lite"
        site_merged = base / "site-merged"

        _rm_rf(site_docs)
        _rm_rf(site_ref)
        _rm_rf(site_lite)
        _rm_rf(site_merged)

        site_docs.mkdir(parents=True, exist_ok=True)
        site_ref.mkdir(parents=True, exist_ok=True)
        site_lite.mkdir(parents=True, exist_ok=True)
        site_merged.mkdir(parents=True, exist_ok=True)

        try:
            # Build API first. gen_ref_pages.py is responsible for writing docs/_api_inventory.json.
            run_cmd(
                [sys.executable, "-m", "mkdocs", "build", "-f", str(ref_cfg), "-d", str(site_ref)],
                cwd=project_root,
            )

            if not inv_disk.exists():
                raise SystemExit(
                    f"API inventory not found at {inv_disk}.\n"
                    "Expected gen_ref_pages.py to write it during the API build."
                )

            # Build main docs next (will use the same inventory file).
            run_cmd(
                [sys.executable, "-m", "mkdocs", "build", "-f", str(docs_cfg), "-d", str(site_docs)],
                cwd=project_root,
            )

            # Build JupyterLite last.
            if not args.no_lite:
                if not args.no_lite_notebooks:
                    if lite_files_dir.exists() and not args.keep_lite_files:
                        _rm_rf(lite_files_dir)
                    lite_files_dir.mkdir(parents=True, exist_ok=True)

                    converted, skipped = convert_markdown_notebooks_to_ipynb(
                        src_dir=notebooks_dir,
                        dst_dir=lite_files_dir,
                        cwd=project_root,
                    )
                    print(
                        f"Generated {converted} notebook(s) for JupyterLite in {lite_files_dir}"
                        f" (skipped {skipped} non-notebook markdown file(s))"
                    )

                if not lite_cfg.exists():
                    raise SystemExit(
                        f"JupyterLite config not found at {lite_cfg}.\n"
                        "Create it or use --no-lite to skip the JupyterLite build."
                    )

                run_cmd(
                    [
                        sys.executable,
                        "-m",
                        "jupyter",
                        "lite",
                        "build",
                        "--config",
                        str(lite_cfg),
                        "--output-dir",
                        str(site_lite),
                    ],
                    cwd=docs_root / "jupyterlite",
                )

        finally:
            # Always clean up the inventory file (do not leave generated artifacts in repo)
            if inv_disk.exists():
                inv_disk.unlink()

        # Merge output trees
        copytree_into(site_docs, site_merged)
        copytree_into(site_ref, site_merged / args.ref_mount)
        if not args.no_lite:
            copytree_into(site_lite, site_merged / args.lite_mount)

        return site_merged

    if args.build_only:
        assert out_dir is not None
        if args.clean:
            _rm_rf(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        merged = build_into(out_dir)

        # Flatten merged output into out_dir root
        final_site = out_dir
        for child in list(final_site.iterdir()):
            if child.name in {"site-docs", "site-ref", "site-lite", "site-merged"}:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        copytree_into(merged, final_site)

        print(f"Wrote merged site to: {final_site}")
        print(f"API ref mounted at:   {final_site / args.ref_mount}")
        if not args.no_lite:
            print(f"JupyterLite mounted at: {final_site / args.lite_mount}")
        return

    # Serve mode: build into temp dir unless --out is provided
    if out_dir is None:
        with tempfile.TemporaryDirectory(prefix="mkdocs-merged-") as td:
            site_merged = build_into(Path(td))
            _serve(
                site_merged,
                args.host,
                args.port,
                args.port_start,
                args.port_tries,
                args.ref_mount,
                args.lite_mount,
                lite_enabled=not args.no_lite,
            )
    else:
        if args.clean:
            _rm_rf(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        site_merged = build_into(out_dir)
        _serve(
            site_merged,
            args.host,
            args.port,
            args.port_start,
            args.port_tries,
            args.ref_mount,
            args.lite_mount,
            lite_enabled=not args.no_lite,
        )


if __name__ == "__main__":
    main()