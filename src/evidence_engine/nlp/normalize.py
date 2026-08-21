"""Canonicalization + text normalization (pipeline stage `normalize`)."""

from __future__ import annotations

import hashlib
import html
import math
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "referrer",
}

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def canonical_url(url: str) -> str:
    """Strip tracking params, www, fragments; sort remaining params."""
    url = (url or "").strip()
    if not url:
        return ""
    parts = urlsplit(url)
    scheme = (parts.scheme or "https").lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    params = sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k not in TRACKING_PARAMS
    )
    return urlunsplit((scheme, host, path, urlencode(params), ""))


def domain_of(url: str) -> str:
    return (urlsplit(canonical_url(url)).hostname or "").lower()


def norm_text(text: str) -> str:
    """Collapse whitespace, unescape entities."""
    return _WHITESPACE.sub(" ", html.unescape(text or "")).strip()


def near_dup_key(text: str) -> str:
    """Lowercase/punctuation-stripped key for near-duplicate detection."""
    lowered = _PUNCT.sub(" ", norm_text(text).lower())
    return _WHITESPACE.sub(" ", lowered).strip()


def content_hash(url: str, title: str, snippet: str) -> str:
    """Exact-duplicate hash over canonical url + normalized text."""
    basis = f"{canonical_url(url)}|{norm_text(title)}|{norm_text(snippet)}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def sha256_hex(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def cosine(a: list[float], b: list[float]) -> float:
    """Pure-python cosine similarity (1024-dim scale is fine in-process)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
