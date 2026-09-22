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


def _convert_code_cell_fences(markdown: str) -> str:
    """
    Convert Jupytext code-cell fences to plain Python fences and strip cell options.

    Rewrites lines matching ```{code-cell} (with optional kernel name) to ```python,
    and removes any consecutive MyST-style cell-option lines (:key: value) that appear
    immediately after such a fence. Leaves closing ``` and other content unchanged.
    """
    lines = markdown.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is a code-cell opening fence
        if line.strip().startswith("```{code-cell}"):
            # Replace with plain Python fence
            result.append("```python")
            i += 1

            # Strip consecutive MyST cell-option lines
            while i < len(lines):
                next_line = lines[i]
                # Cell options are lines starting with : and ending before closing ```
                if next_line.strip().startswith(
                    ":"
                ) and not next_line.strip().startswith("```"):
                    # This is a cell-option line, skip it
                    i += 1
                else:
                    # Not a cell-option line, stop stripping
                    break
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def _convert_note_admonitions(markdown: str) -> str:
    """
    Convert MyST-style note fences to MkDocs Material admonition syntax.

    Rewrites lines matching ```{note} to !!! note with properly indented content.
    The opening fence is replaced with !!! note, all non-blank lines in the block
    get 4 extra spaces of leading indentation, and the closing ``` fence is removed.
    """
    lines = markdown.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is a note opening fence
        if line.strip() == "```{note}":
            # Replace with admonition syntax
            result.append("!!! note")
            i += 1

            # Process content until closing fence
            while i < len(lines):
                content_line = lines[i]
                # Check for closing fence
                if content_line.strip() == "```":
                    # Consume closing fence without emitting it
                    i += 1
                    break
                else:
                    # Indent non-blank lines with 4 spaces
                    if content_line.strip():
                        result.append("    " + content_line)
                    else:
                        # Emit blank lines as truly empty (no padding)
                        result.append("")
                    i += 1
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def get_rtd_prefix() -> str:
    import os
    from urllib.parse import urlparse

    canonical_url = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
    if canonical_url:
        path = urlparse(canonical_url).path.rstrip("/")
        if path:
            return path
    return ""


def on_nav(nav, config, files):
    rtd_prefix = get_rtd_prefix()
    if not rtd_prefix:
        return nav

    def walk_items(items):
        for item in items:
            if hasattr(item, "url") and item.url == "/reference":
                item.url = f"{rtd_prefix}/reference/"
            if hasattr(item, "children") and item.children:
                walk_items(item.children)

    walk_items(nav.items)
    return nav


def on_page_markdown(markdown: str, page, config, files) -> str:
    """
    Rewrite `[text](api:Target)` into `/reference/...` URLs, resolve Binder branch placeholders, and normalize code-cell fences.

    - Converts Jupytext ````{code-cell}` opening fences to plain ````python` and strips trailing cell-option lines.
    - Converts MyST-style ````{note}` fences to MkDocs Material `!!!` admonition syntax.
    - Rewrites author-facing `api:` links to `/reference/...` URLs using the generated API inventory.
    - Hard build failure on unresolved or ambiguous targets.
    - Dynamically resolves the currently checked-out Git branch name and replaces `{{ binder_branch }}` in every page's markdown.
    """
    markdown = _convert_code_cell_fences(markdown)
    markdown = _convert_note_admonitions(markdown)

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
    rtd_prefix = get_rtd_prefix()
    url_prefix = f"{rtd_prefix}/reference" if rtd_prefix else "/reference"
    return rewrite_api_links(
        markdown, inv, url_prefix=url_prefix, page_src=src
    )
