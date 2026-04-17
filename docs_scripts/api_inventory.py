from __future__ import annotations

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


@dataclass(frozen=True)
class ApiInventory:
    """
    objects: map from fully qualified anchor id -> URL (relative to API site root)
    suffix_index: map from suffix string -> list of fully qualified anchor ids
    kinds: map from fully qualified anchor id -> kind string
           (e.g. module/class/function/method/property/variable/type_alias/type_variable)
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
        Resolve progressive qualification and return the URL.
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
        tail = fqn[len(package) + 1 :]  # remove "loqs."
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
      - main docs: "/reference" (so inventory URLs become /reference/loqs/...)
      - API docs:  ""           (so inventory URLs stay /loqs/...)
    """

    def resolve_url(target: str) -> str:
        try:
            rel = inv.resolve(target)
        except KeyError as e:
            raise RuntimeError(f"{page_src}: {e}") from None
        # Ensure prefix concatenation is clean
        if url_prefix and rel.startswith("/"):
            return url_prefix + rel
        return url_prefix + rel

    # 1) Convert reference-style to inline-style so the same logic handles both.
    def ref_to_inline(m: re.Match) -> str:
        text = m.group("text") or ""
        target = m.group("target")
        return f"[{text}](api:{target})"

    out = _API_REF_RE.sub(ref_to_inline, markdown)

    # Inline: [text](api:Target) -> [text](URL)
    def repl_inline(m: re.Match) -> str:
        target = m.group("target")
        raw_text = (m.group("text") or "").strip()

        url = resolve_url(target)

        # Resolve to canonical inventory key so we can classify kind accurately
        fqn = inv.resolve_fqn(target)
        kind = (inv.kinds.get(fqn) or "").lower()

        # Choose display text
        if not raw_text:
            # Use resolved object name as a guaranteed non-empty display
            base = fqn.split(".")[-1]
            display = base
        else:
            display = raw_text
            # Strip one layer of backticks if present; we'll reapply consistently below
            if display.startswith("`") and display.endswith("`") and len(display) >= 2:
                display = display[1:-1].strip()

        display = display.strip()
        if not display:
            # Absolute fallback (should never happen)
            display = fqn.split(".")[-1]

        # Append () for callables (even when text is qualified), unless already present
        if kind in {"function", "method"}:
            if not display.endswith("()"):
                display = display + "()"

        # Backtick everything (consistently)
        display = display.strip()
        if not (display.startswith("`") and display.endswith("`")):
            display = f"`{display}`"

        return f"[{display}]({url})"

    out = _API_LINK_RE.sub(repl_inline, out)
    return out