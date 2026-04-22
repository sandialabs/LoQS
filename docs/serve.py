#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import os
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
    """
    env = os.environ.copy()
    env["DISABLE_MKDOCS_2_WARNING"] = "true"
    env["NO_MKDOCS_2_WARNING"] = "true"
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def copytree_into(src: Path, dst: Path) -> None:
    """Recursively copy *src* into *dst* (overwrites files)."""
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


def _serve(site_merged: Path, host: str, port: int, port_start: int, port_tries: int, ref_mount: str) -> None:
    port = port or pick_port(host, start=port_start, max_tries=port_tries)

    os.chdir(site_merged)
    httpd = http.server.ThreadingHTTPServer((host, port), http.server.SimpleHTTPRequestHandler)

    print(f"Merged site: http://{host}:{port}/")
    print(f"API ref:     http://{host}:{port}/{ref_mount}/")
    print("Ctrl+C to stop (re-run script to rebuild).")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build main docs + API reference, merge into one site, optionally serve it."
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

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    docs_cfg = (repo_root / args.docs_config).resolve()
    ref_cfg = (repo_root / args.ref_config).resolve()

    # The inventory file is written by gen_ref_pages.py to disk at this location
    # so BOTH builds can resolve api: links with progressive qualification.
    inv_disk = repo_root / "docs" / "_api_inventory.json"

    # Output selection:
    # - serve mode: default to temp dir unless --out provided
    # - build-only: default to repo_root/site unless --out provided
    if args.out is None:
        out_dir = (repo_root / "site") if args.build_only else None
    else:
        out_dir = (repo_root / args.out).resolve()

    def build_into(base: Path) -> Path:
        site_docs = base / "site-docs"
        site_ref = base / "site-ref"
        site_merged = base / "site-merged"

        _rm_rf(site_docs)
        _rm_rf(site_ref)
        _rm_rf(site_merged)

        site_docs.mkdir(parents=True, exist_ok=True)
        site_ref.mkdir(parents=True, exist_ok=True)
        site_merged.mkdir(parents=True, exist_ok=True)

        try:
            # Build API first. gen_ref_pages.py is responsible for writing docs/_api_inventory.json.
            run_cmd(
                [sys.executable, "-m", "mkdocs", "build", "-f", str(ref_cfg), "-d", str(site_ref)],
                cwd=repo_root,
            )

            if not inv_disk.exists():
                raise SystemExit(
                    f"API inventory not found at {inv_disk}.\n"
                    "Expected gen_ref_pages.py to write it during the API build."
                )

            # Build main docs next (will use the same inventory file).
            run_cmd(
                [sys.executable, "-m", "mkdocs", "build", "-f", str(docs_cfg), "-d", str(site_docs)],
                cwd=repo_root,
            )

        finally:
            # Always clean up the inventory file (do not leave generated artifacts in repo)
            if inv_disk.exists():
                inv_disk.unlink()

        # Merge output trees
        copytree_into(site_docs, site_merged)
        copytree_into(site_ref, site_merged / args.ref_mount)

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
            if child.name in {"site-docs", "site-ref", "site-merged"}:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        copytree_into(merged, final_site)

        print(f"Wrote merged site to: {final_site}")
        print(f"API ref mounted at:   {final_site / args.ref_mount}")
        return

    # Serve mode: build into temp dir unless --out is provided
    if out_dir is None:
        with tempfile.TemporaryDirectory(prefix="mkdocs-merged-") as td:
            site_merged = build_into(Path(td))
            _serve(site_merged, args.host, args.port, args.port_start, args.port_tries, args.ref_mount)
    else:
        if args.clean:
            _rm_rf(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        site_merged = build_into(out_dir)
        _serve(site_merged, args.host, args.port, args.port_start, args.port_tries, args.ref_mount)


if __name__ == "__main__":
    main()