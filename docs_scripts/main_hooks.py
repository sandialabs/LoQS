from __future__ import annotations

"""
Main-site MkDocs hooks for the narrative/Marimo documentation build.

Goals
-----
- Rewrite author-facing `api:` links in Markdown pages into mounted `/reference/...`
  URLs using the generated API inventory.
- Fail the build for ambiguous or unresolved internal targets so broken cross-links
  never silently ship.
- Keep the main-docs link semantics aligned with the API-reference site by delegating
  to the shared inventory/link-rewrite helpers.
"""

from pathlib import Path

from docs_scripts.api_inventory import ApiInventory, rewrite_api_links


def on_page_markdown(markdown: str, page, config, files) -> str:
    """
    Rewrite `[text](api:Target)` into `/reference/...` URLs and resolve Binder branch placeholders.

    - Rewrites author-facing `api:` links to `/reference/...` URLs using the generated API inventory.
    - Hard build failure on unresolved or ambiguous targets.
    - Dynamically resolves the currently checked-out Git branch name and replaces `{{ binder_branch }}` in every page's markdown.
    """
    import subprocess
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            cwd=config["config_dir"],
        ).strip()
        if branch == "HEAD":
            branch = "main"
    except Exception:
        branch = "main"

    markdown = markdown.replace("{{ binder_branch }}", branch)

    inv_path = Path(config["docs_dir"]) / "_api_inventory.json"
    if not inv_path.exists():
        raise RuntimeError(
            f"API inventory not found at {inv_path}. "
            "Run docs via serve.py so the API inventory is generated and injected."
        )

    inv = ApiInventory.load(inv_path)
    src = getattr(page.file, "src_path", "") if hasattr(page, "file") else ""
    return rewrite_api_links(markdown, inv, url_prefix="/reference", page_src=src)