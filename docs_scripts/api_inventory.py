from __future__ import annotations

"""
Shared API-inventory and cross-link utilities for the two-stage documentation build.

Goals
-----
- Provide a canonical representation of the generated API inventory.
- Resolve progressive-qualification `api:` targets to fully-qualified objects.
- Rewrite Markdown `api:` links for both the main docs and the API-reference docs.
- Centralize URL mounting/prefix behavior so hooks and generators use the same logic.
- Provide best-effort external documentation mappings for selected third-party and
  standard-library APIs.

Key capabilities
----------------
- Progressive target resolution:
  - `api:Serializable`
  - `api:internal.serializable.Serializable`
  - `api:loqs.internal.serializable.Serializable`
- Link rewriting for:
  - inline Markdown links: `[Text](api:Target)`
  - reference-style links: `[Text][api:Target]`
- URL helpers for mounting the API site under `/reference`.
- External mappings for pyGSTi, Stim, and selected Python stdlib modules.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path


# Inline Markdown links: [text](api:Target) or [](api:Target)
_API_LINK_RE = re.compile(r"\[(?P<text>[^\]]*)\]\(\s*api:(?P<target>[^)\s]+)\s*\)")

# Reference-style: [text][api:Target] or [][api:Target]
_API_REF_RE = re.compile(r"\[(?P<text>[^\]]*)\]\[\s*api:(?P<target>[^\]\s]+)\s*\]")


def normalize_target(t: str) -> str:
    """
    Normalize author input:
      - strip trailing "()" for methods/functions
      - strip trailing "." (just in case)
    """
    t = t.strip()
    if t.endswith("()"):
        t = t[:-2]
    while t.endswith("."):
        t = t[:-1]
    return t


def mount_url(rel: str, prefix: str = "") -> str:
    """
    Mount an inventory-relative URL under a prefix.

    Examples:
      mount_url("/loqs/internal/", "/reference") -> "/reference/loqs/internal/"
      mount_url("/loqs/internal/", "")           -> "/loqs/internal/"
    """
    if not prefix:
        return rel
    if rel.startswith("/"):
        return prefix + rel
    return prefix.rstrip("/") + "/" + rel.lstrip("/")


def external_api_url(target: str) -> str | None:
    """
    Map non-loqs api: targets to an external documentation URL.

    Return None if no mapping is known.
    """
    t = normalize_target(target)

    # pyGSTi: pygsti.<module>.<Name> -> RTD autoapi anchor
    if t.startswith("pygsti."):
        parts = t.split(".")
        if len(parts) >= 3:
            cls = parts[-1]
            mod = ".".join(parts[:-1])
            mod_path = "/".join(parts[:-1])
            return f"https://pygsti.readthedocs.io/en/latest/autoapi/{mod_path}/index.html#{mod}.{cls}"

    # Stim Python API reference
    if t.startswith("stim."):
        return f"https://github.com/quantumlib/Stim/wiki/Stim-v1.13-Python-API-Reference#{t}"

    # Python stdlib (best-effort): module page + qualified anchor
    stdlib_prefixes = (
        "collections.",
        "collections.abc.",
        "typing.",
        "pathlib.",
        "dataclasses.",
        "abc.",
        "enum.",
    )
    if t.startswith(stdlib_prefixes) and "." in t:
        mod = t.rsplit(".", 1)[0]
        return f"https://docs.python.org/3/library/{mod}.html#{t}"

    return None


@dataclass(frozen=True)
class ApiInventory:
    """
    API inventory used by both docs stages.

    objects:
      map from fully qualified anchor id -> URL (relative to API site root)

    suffix_index:
      map from suffix string -> list of matching fully qualified anchor ids

    kinds:
      map from fully qualified anchor id -> kind string, e.g.
      module/class/function/method/property/variable/type_alias/type_variable
    """

    objects: dict[str, str]
    suffix_index: dict[str, list[str]]
    kinds: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "ApiInventory":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            objects=data["objects"],
            suffix_index=data["suffix_index"],
            kinds=data.get("kinds", {}),
        )

    def resolve_fqn(self, target: str) -> str:
        """
        Resolve progressive qualification and return the fully-qualified inventory key.
        """
        t = normalize_target(target)

        # Exact FQN
        if t.startswith("loqs."):
            if t not in self.objects:
                raise KeyError(f"Unresolved api target (no such API object): {t}")
            return t

        # Exact suffix match
        hits = self.suffix_index.get(t)
        if hits:
            if len(hits) == 1:
                return hits[0]
            opts = "\n  - ".join(hits)
            raise KeyError(
                f"Ambiguous api target: {t}\n"
                f"Matches multiple API objects:\n  - {opts}\n"
                f"Disambiguate by adding more qualification."
            )

        # Package-relative exact
        fqn2 = "loqs." + t
        if fqn2 in self.objects:
            return fqn2

        raise KeyError(
            f"Unresolved api target: {t}\n"
            "Try qualifying it further (e.g. api:internal.serializable.Serializable) "
            "or using a full FQN (api:loqs....)."
        )

    def resolve(self, target: str) -> str:
        """
        Resolve progressive qualification and return the inventory-relative URL.
        """
        fqn = self.resolve_fqn(target)
        return self.objects[fqn]

    def kind_of(self, target: str, *, default: str = "") -> str:
        """
        Return the kind for a target if known (empty string if unknown).

        Accepts the same target forms as `resolve`.
        """
        try:
            fqn = self.resolve_fqn(target)
        except KeyError:
            return default
        return (self.kinds.get(fqn) or default)

    def resolve_mounted_url(self, target: str, *, prefix: str = "") -> str:
        """
        Resolve a target and mount it under the requested prefix.
        """
        return mount_url(self.resolve(target), prefix=prefix)


def resolve_api_target_url(
    inv: ApiInventory,
    target: str,
    *,
    src: str = "",
    prefix: str = "",
    allow_external: bool = True,
) -> str | None:
    """
    Resolve a target to a mounted internal URL or an external URL.

    Behavior:
    - Internal `loqs.*` targets must resolve or raise.
    - Non-loqs targets first try inventory resolution, then optional external mapping.
    - Returns None for unresolved non-loqs targets when no external mapping exists.
    """
    try:
        return inv.resolve_mounted_url(target, prefix=prefix)
    except KeyError as e:
        t = normalize_target(target)
        if t.startswith("loqs."):
            raise RuntimeError(f"{src}: {e}") from None
        if not allow_external:
            return None
        return external_api_url(t)


def build_suffix_index(objects: dict[str, str], *, package: str = "loqs") -> dict[str, list[str]]:
    """
    Build suffix_index mapping from progressive suffixes to matching FQNs.

    For each FQN like:
      loqs.internal.serializable.Serializable.encode

    Add suffixes:
      internal.serializable.Serializable.encode
      serializable.Serializable.encode
      Serializable.encode
      encode

    Note: full FQN resolution is handled by `objects` directly.
    """
    out: dict[str, list[str]] = {}

    for fqn in objects.keys():
        if not fqn.startswith(package + "."):
            continue
        tail = fqn[len(package) + 1 :]
        parts = tail.split(".")
        for i in range(len(parts)):
            suff = ".".join(parts[i:])
            out.setdefault(suff, []).append(fqn)

    for k in list(out.keys()):
        out[k] = sorted(set(out[k]))
    return out


def rewrite_api_links(markdown: str, inv: ApiInventory, *, url_prefix: str, page_src: str = "") -> str:
    """
    Rewrite api: links in Markdown into real URLs.

    url_prefix:
      - main docs: "/reference"
      - API docs:  ""
    """

    def resolve_url(target: str) -> str:
        try:
            return inv.resolve_mounted_url(target, prefix=url_prefix)
        except KeyError as e:
            raise RuntimeError(f"{page_src}: {e}") from None

    def ref_to_inline(m: re.Match) -> str:
        text = m.group("text") or ""
        target = m.group("target")
        return f"[{text}](api:{target})"

    out = _API_REF_RE.sub(ref_to_inline, markdown)

    def repl_inline(m: re.Match) -> str:
        target = m.group("target")
        raw_text = (m.group("text") or "").strip()

        url = resolve_url(target)

        fqn = inv.resolve_fqn(target)
        kind = (inv.kinds.get(fqn) or "").lower()

        if not raw_text:
            # Use resolved object name as a guaranteed non-empty display.
            # Special-case constructors so empty-text links display `ClassName()`
            # rather than `__init__()`.
            base = fqn.split(".")[-1]
            if base == "__init__" and "." in fqn:
                display = fqn.split(".")[-2]
            else:
                display = base
        else:
            display = raw_text
            if display.startswith("`") and display.endswith("`") and len(display) >= 2:
                display = display[1:-1].strip()

        display = display.strip() or fqn.split(".")[-1]

        if kind in {"function", "method"} and not display.endswith("()"):
            display = display + "()"

        if not (display.startswith("`") and display.endswith("`")):
            display = f"`{display}`"

        return f"[{display}]({url})"

    out = _API_LINK_RE.sub(repl_inline, out)
    return out