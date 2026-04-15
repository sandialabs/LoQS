from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Inline Markdown links: [text](api:Target)
_API_LINK_RE = re.compile(r"\]\(api:(?P<target>[^)\s]+)\)")

# Reference-style: [text][api:Target]  -> rewritten to [text](URL)
_API_REF_RE = re.compile(r"\]\[api:(?P<target>[^\]\s]+)\]")


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
    """
    objects: dict[str, str]
    suffix_index: dict[str, list[str]]

    @classmethod
    def load(cls, path: Path) -> "ApiInventory":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(objects=data["objects"], suffix_index=data["suffix_index"])

    def resolve(self, target: str) -> str:
        """
        Resolve progressive qualification:

        - If target starts with "loqs.": must match exactly in objects
        - Else:
            1) if suffix_index has exact key, use if unique
            2) else try objects["loqs."+target]
            3) else fail
        """
        t = normalize_target(target)

        # Exact FQN
        if t.startswith("loqs."):
            url = self.objects.get(t)
            if not url:
                raise KeyError(f"Unresolved api target (no such API object): {t}")
            return url

        # Exact suffix match
        hits = self.suffix_index.get(t)
        if hits:
            if len(hits) == 1:
                return self.objects[hits[0]]
            opts = "\n  - ".join(hits)
            raise KeyError(
                f"Ambiguous api target: {t}\n"
                f"Matches multiple API objects:\n  - {opts}\n"
                f"Disambiguate by adding more qualification."
            )

        # Package-relative exact
        fqn2 = "loqs." + t
        url2 = self.objects.get(fqn2)
        if url2:
            return url2

        raise KeyError(
            f"Unresolved api target: {t}\n"
            "Try qualifying it further (e.g. api:internal.serializable.Serializable) "
            "or using a full FQN (api:loqs....)."
        )


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

    # Reference-style: [text][api:Target] -> [text](URL)
    def repl_ref(m: re.Match) -> str:
        url = resolve_url(m.group("target"))
        return f"]({url})"

    # Inline: [text](api:Target) -> [text](URL)
    def repl_inline(m: re.Match) -> str:
        url = resolve_url(m.group("target"))
        return f"]({url})"

    out = _API_REF_RE.sub(repl_ref, markdown)
    out = _API_LINK_RE.sub(repl_inline, out)
    return out