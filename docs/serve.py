#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile


def pick_port(host: str, start: int = 8000, max_tries: int = 200) -> int:
    """
    Try start, then start+1, ... until a free port is found.
    """
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
    p = subprocess.run(cmd, cwd=str(cwd))
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def copytree_into(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for fn in files:
            s = Path(root) / fn
            d = dst / rel / fn
            shutil.copy2(s, d)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build main docs + API reference, merge into one site, then serve it."
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 = find port starting at 8000")
    ap.add_argument("--port-start", type=int, default=8000)
    ap.add_argument("--port-tries", type=int, default=200)
    ap.add_argument("--docs-config", default="mkdocs.yml")
    ap.add_argument("--ref-config", default="mkdocs-api-ref.yml")
    ap.add_argument("--ref-mount", default="reference", help="mount point for reference site")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    port = args.port or pick_port(args.host, start=args.port_start, max_tries=args.port_tries)

    docs_cfg = (repo_root / args.docs_config).resolve()
    ref_cfg = (repo_root / args.ref_config).resolve()

    with tempfile.TemporaryDirectory(prefix="mkdocs-merged-") as td:
        base = Path(td)
        site_docs = base / "site-docs"
        site_ref = base / "site-ref"
        site_merged = base / "site-merged"

        run_cmd([sys.executable, "-m", "mkdocs", "build", "-f", str(docs_cfg), "-d", str(site_docs)], cwd=repo_root)
        run_cmd([sys.executable, "-m", "mkdocs", "build", "-f", str(ref_cfg), "-d", str(site_ref)], cwd=repo_root)

        copytree_into(site_docs, site_merged)
        copytree_into(site_ref, site_merged / args.ref_mount)

        os.chdir(site_merged)
        httpd = http.server.ThreadingHTTPServer((args.host, port), http.server.SimpleHTTPRequestHandler)

        print(f"Merged site: http://{args.host}:{port}/")
        print(f"API ref:     http://{args.host}:{port}/{args.ref_mount}/")
        print("Ctrl+C to stop (re-run script to rebuild).")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()