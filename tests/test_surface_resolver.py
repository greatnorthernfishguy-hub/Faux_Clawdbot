# ---- Changelog ----
# [2026-07-06] Weft (TQB) — Initial test suite for surface_resolver.resolve_surface_content()
# What: New file. Establishes tests/ as the repo's first test directory.
# Why: PRD §2 (2026-07-06-codemine-pattern-completion-recall) sub-step 1 requires
#      coverage of resolve_surface_content()'s preference order and filters before
#      it is wired into any .recall() call site.
# How: Pure-function tests against a tiny stub node (a plain class with a .metadata
#      dict, per resolve_surface_content()'s documented contract) — no NeuroGraphMemory
#      or graph needed since this module has zero substrate dependencies.
# -------------------

from surface_resolver import resolve_surface_content


class _StubNode:
    """Minimal stand-in for a neuro_foundation.Node — only `.metadata` is read."""

    def __init__(self, metadata=None):
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# 1. Forest-content preference
# ---------------------------------------------------------------------------

def test_prefers_forest_content_over_vdb_and_label():
    node = _StubNode({
        "_forest_content": "This is her actual lived turn, long enough to pass the floor.",
        "_label": "short label",
    })
    vdb_entry = {"content": "WANT documentation shard, also long enough to pass the floor."}

    result = resolve_surface_content(node, vdb_entry)

    assert result == "This is her actual lived turn, long enough to pass the floor."


# ---------------------------------------------------------------------------
# 2. Vdb-shard fallback
# ---------------------------------------------------------------------------

def test_falls_back_to_vdb_shard_when_forest_content_missing():
    node = _StubNode({"_label": "a label that is long enough to pass the floor too"})
    vdb_entry = {"content": "the vdb shard content, long enough to pass the floor"}

    result = resolve_surface_content(node, vdb_entry)

    assert result == "the vdb shard content, long enough to pass the floor"


def test_falls_back_to_vdb_shard_when_forest_content_too_short():
    # _forest_content present but sub-floor (shorter than min_chars=12 default)
    node = _StubNode({"_forest_content": "short"})
    vdb_entry = {"content": "the vdb shard content, long enough to pass the floor"}

    result = resolve_surface_content(node, vdb_entry)

    assert result == "the vdb shard content, long enough to pass the floor"


def test_vdb_entry_as_plain_string_also_works():
    node = _StubNode({})
    vdb_entry = "a plain string vdb shard, long enough to pass the floor"

    result = resolve_surface_content(node, vdb_entry)

    assert result == "a plain string vdb shard, long enough to pass the floor"


# ---------------------------------------------------------------------------
# 3. Label-last-resort
# ---------------------------------------------------------------------------

def test_falls_back_to_label_when_forest_and_vdb_both_missing():
    node = _StubNode({"_label": "a node label that is long enough to pass the floor"})
    vdb_entry = {}

    result = resolve_surface_content(node, vdb_entry)

    assert result == "a node label that is long enough to pass the floor"


def test_falls_back_to_label_when_forest_and_vdb_both_too_short():
    node = _StubNode({
        "_forest_content": "short",
        "_label": "a node label that is long enough to pass the floor",
    })
    vdb_entry = {"content": "brief"}

    result = resolve_surface_content(node, vdb_entry)

    assert result == "a node label that is long enough to pass the floor"


# ---------------------------------------------------------------------------
# 4. Degenerate-fragment rejection — too short, and bare stopword shard
# ---------------------------------------------------------------------------

def test_rejects_fragment_shorter_than_min_chars():
    node = _StubNode({"_forest_content": "too short"})
    vdb_entry = {"content": "brief"}

    result = resolve_surface_content(node, vdb_entry)

    assert result is None


def test_rejects_bare_stopword_shard_even_when_min_chars_is_lowered():
    # "the" passes a lowered min_chars floor on length alone, but must still
    # be rejected because it is a bare stopword shard with no experiential signal.
    node = _StubNode({"_forest_content": "the"})
    vdb_entry = {}

    result = resolve_surface_content(node, vdb_entry, min_chars=1)

    assert result is None


def test_stopword_check_is_case_insensitive():
    node = _StubNode({"_forest_content": "THE"})
    vdb_entry = {}

    result = resolve_surface_content(node, vdb_entry, min_chars=1)

    assert result is None


def test_rejects_when_nothing_usable_anywhere():
    node = _StubNode({})
    vdb_entry = None

    result = resolve_surface_content(node, vdb_entry)

    assert result is None


# ---------------------------------------------------------------------------
# 5. max_chars truncation with the "…" suffix
# ---------------------------------------------------------------------------

def test_truncates_to_max_chars_with_ellipsis_suffix():
    node = _StubNode({"_forest_content": "A" * 100})
    vdb_entry = {}

    result = resolve_surface_content(node, vdb_entry, max_chars=20)

    assert result == ("A" * 20) + "…"
    assert len(result) == 21


def test_does_not_truncate_when_under_max_chars():
    text = "a short turn that stays under the default max_chars ceiling"
    node = _StubNode({"_forest_content": text})
    vdb_entry = {}

    result = resolve_surface_content(node, vdb_entry)

    assert result == text
    assert "…" not in result


# ---------------------------------------------------------------------------
# 6. creation_mode == "ingested" filtering — both allowed and disallowed
# ---------------------------------------------------------------------------

def test_ingested_node_filtered_by_default():
    node = _StubNode({
        "creation_mode": "ingested",
        "_forest_content": "some ingested source content, long enough to pass the floor",
    })
    vdb_entry = {}

    result = resolve_surface_content(node, vdb_entry)

    assert result is None


def test_ingested_node_allowed_through_when_allow_ingested_true():
    node = _StubNode({
        "creation_mode": "ingested",
        "_forest_content": "some ingested source content, long enough to pass the floor",
    })
    vdb_entry = {}

    result = resolve_surface_content(node, vdb_entry, allow_ingested=True)

    assert result == "some ingested source content, long enough to pass the floor"


# ---------------------------------------------------------------------------
# Misc: node=None is handled gracefully (falls straight to vdb/label)
# ---------------------------------------------------------------------------

def test_node_none_falls_back_to_vdb_entry():
    result = resolve_surface_content(None, {"content": "vdb-only content, long enough to pass the floor"})

    assert result == "vdb-only content, long enough to pass the floor"
