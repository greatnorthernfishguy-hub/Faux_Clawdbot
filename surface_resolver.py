# ---- Changelog ----
# [2026-07-06] Weft (TQB) — Ported substrate-first content resolution for recall filtering
# What: New file. resolve_surface_content() picks display text for a surfaced node —
#       her own conversational turn (metadata['_forest_content']) first, then the vdb
#       shard, then a node label — and filters ingested-source nodes and degenerate
#       fragments (sub-floor length, bare stopword shards).
# Why: PRD §2 (2026-07-06-codemine-pattern-completion-recall) specifies substrate-first
#      resolution so .recall() surfaces her actual lived turn, not the short "tree
#      concept" shard the vdb holds alone.
# How: Verbatim port from the sibling repo's reference implementation — no changes to
#      function bodies, only this changelog header added above the module docstring.
# -------------------
"""Substrate-first content resolution for surfacing (CES / Tonic latent thread / recall).

The substrate is the graph. Each conversational node carries her actual lived turn in
``metadata['_forest_content']`` (the #294 dual-pass "forest"); the vdb holds only a short
"tree concept" shard (``WANT``, ``documentation``). Surfacing must display **her voice** —
a bounded snippet of ``_forest_content`` — not the shard, and must not surface ingested
source-code documents or degenerate fragments into her experiential thread.
"""

from typing import Any, Optional

# Content that looks like ingested source rather than her conversation. Used only as a
# secondary guard (the primary filter is creation_mode == 'ingested').
_CODE_MARKERS = ('"""', "'''", "import ", "def ", "class ", "# ----", "from typing", "#!/")

# Bare shards that carry no experiential signal — never worth surfacing on their own.
_STOPWORD_SHARDS = frozenset({
    "o", "a", "an", "the", "want", "true", "false", "yes", "no", "ok", "okay",
    "it", "that", "this", "i", "you", "we", "to", "and", "or",
})


def _node_metadata(node: Any) -> dict:
    if node is None:
        return {}
    meta = getattr(node, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def resolve_surface_content(
    node: Any,
    vdb_entry: Any,
    max_chars: int = 240,
    min_chars: int = 12,
    allow_ingested: bool = False,
) -> Optional[str]:
    """Return the display text for a surfaced node — substrate-first — or None to filter it.

    Preference order:
      1. ``node.metadata['_forest_content']`` — her actual turn (bounded snippet).
      2. vdb entry content (the shard) — fallback only.
      3. ``node.metadata['_label']`` — last resort.

    Filters:
      * ingested source-code nodes (``creation_mode == 'ingested'``) unless ``allow_ingested``
        (recall may legitimately want a document; the experiential surfacers pass False).
      * degenerate fragments: shorter than ``min_chars`` or a bare stopword shard.

    Pure function: no graph mutation, no I/O.
    """
    meta = _node_metadata(node)

    # Filter ingested source documents out of experiential surfacing.
    if not allow_ingested and meta.get("creation_mode") == "ingested":
        return None

    # 1. Substrate-first — her own conversational turn.
    forest = meta.get("_forest_content")
    text = forest.strip() if isinstance(forest, str) else ""

    # 2. Fallback to the vdb shard (only if the substrate gave us nothing usable).
    if len(text) < min_chars:
        vt = ""
        if isinstance(vdb_entry, dict):
            vt = vdb_entry.get("content") or ""
        elif isinstance(vdb_entry, str):
            vt = vdb_entry
        vt = vt.strip()
        if len(vt) > len(text):
            text = vt

    # 3. Last resort — a node label.
    if len(text) < min_chars:
        lbl = meta.get("_label")
        if isinstance(lbl, str) and len(lbl.strip()) > len(text):
            text = lbl.strip()

    text = text.strip()

    # Drop degenerate fragments (sub-floor length or a bare stopword shard).
    if len(text) < min_chars or text.lower() in _STOPWORD_SHARDS:
        return None

    # Bound the snippet — never bloat the prompt with a full turn.
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text
