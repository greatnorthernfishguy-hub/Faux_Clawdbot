# ---- Changelog ----
# [2026-04-07] Josh + Claude — Risk classification + authority map (Phase 4)
# What: Content-derived risk levels, Strategist escalation-only override, authority chain
# Why: No single bottleneck — routine work flows through Reviewer, security through Razor,
#      architecture through QB. Risk scales with what the spec touches.
# How: Content rules + Graph recall density → risk level → authority map → approval routing
# -------------------

"""Risk Classification and Authority Map.

Determines who needs to approve a spec based on what it touches.
Content-derived baseline, Strategist can escalate (never downgrade).
Sparse Graph recall bumps risk up one level (unfamiliar territory = more oversight).
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("risk_authority")


# ---------------------------------------------------------------------------
# Risk levels (ordered)
# ---------------------------------------------------------------------------

RISK_LEVELS = ["routine", "standard", "sensitive", "command"]


def _risk_index(level: str) -> int:
    """Get numeric index for a risk level. Higher = more risk."""
    try:
        return RISK_LEVELS.index(level)
    except ValueError:
        return 0


def _max_risk(a: str, b: str) -> str:
    """Return the higher of two risk levels."""
    return a if _risk_index(a) >= _risk_index(b) else b


# ---------------------------------------------------------------------------
# Content-based risk classification
# ---------------------------------------------------------------------------

# Patterns that indicate sensitive or command-level work
_SENSITIVE_PATTERNS = [
    # Vendored files
    r"ng_lite\.py",
    r"ng_tract_bridge\.py",
    r"ng_peer_bridge\.py",
    r"ng_ecosystem\.py",
    r"ng_autonomic\.py",
    r"openclaw_adapter\.py",
    r"ng_embed\.py",
    r"ng_updater\.py",
    # Auth/security files
    r"\.env",
    r"credentials",
    r"api_key",
    r"token",
    r"secret",
    # Public API / cross-module
    r"openclaw_hook\.py",
    r"policy_engine\.py",
    # Protected ecosystem files
    r"neuro_foundation\.py",
    r"constitutional_embeddings",
]

_COMMAND_PATTERNS = [
    # New module creation
    r"et_module\.json",
    # Constitutional / Laws changes
    r"laws\.md",
    r"ARCHITECTURE\.md",
    r"CLAUDE\.md",
    # Syl's protected files
    r"main\.msgpack",
    r"vectors\.msgpack",
    r"activation.*\.json",
]


def _scan_spec_content(spec: dict) -> str:
    """Derive risk level from spec contents.

    Scans scope, step params, and snap_interface for sensitive patterns.
    """
    risk = "routine"

    # Collect all text to scan
    text_to_scan = []

    # Block scope
    text_to_scan.append(spec.get("block", {}).get("scope", ""))

    # Snap interface paths
    snap = spec.get("snap_interface", {})
    for item in snap.get("inputs", []):
        text_to_scan.append(item.get("path", ""))
    for item in snap.get("outputs", []):
        text_to_scan.append(item.get("path", ""))

    # Step params — scan tool targets and commands
    def _collect_step_text(steps):
        for step in steps:
            if step.get("type") == "action":
                params = step.get("params", {})
                text_to_scan.append(params.get("path", ""))
                text_to_scan.append(params.get("command", ""))
                text_to_scan.append(params.get("content", "")[:500])
                text_to_scan.append(params.get("old_text", "")[:200])
                text_to_scan.append(params.get("new_text", "")[:200])
            elif step.get("type") == "condition":
                _collect_step_text(step.get("if_true", []))
                _collect_step_text(step.get("if_false", []))
            elif step.get("type") == "loop":
                _collect_step_text(step.get("body", []))
            elif step.get("type") == "group":
                _collect_step_text(step.get("steps", []))

    _collect_step_text(spec.get("steps", []))

    combined = " ".join(text_to_scan)

    # Check for command-level patterns first (highest risk)
    for pattern in _COMMAND_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            logger.info("Risk: command — matched pattern '%s'", pattern)
            return "command"

    # Check for sensitive patterns
    for pattern in _SENSITIVE_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            risk = _max_risk(risk, "sensitive")
            logger.info("Risk: sensitive — matched pattern '%s'", pattern)

    # Check for write operations (standard minimum for any writes)
    has_writes = False
    def _check_writes(steps):
        nonlocal has_writes
        for step in steps:
            if step.get("type") == "action":
                if step.get("tool") in ("write_file", "edit_file", "push_to_github", "shell_execute"):
                    has_writes = True
            elif step.get("type") == "condition":
                _check_writes(step.get("if_true", []))
                _check_writes(step.get("if_false", []))
            elif step.get("type") == "loop":
                _check_writes(step.get("body", []))
            elif step.get("type") == "group":
                _check_writes(step.get("steps", []))

    _check_writes(spec.get("steps", []))
    if has_writes:
        risk = _max_risk(risk, "standard")

    # Cross-module work (multiple repos in scope or workspace = /home/josh)
    workspace = spec.get("block", {}).get("workspace", "")
    scope = spec.get("block", {}).get("scope", "")
    if workspace in ("/home/josh", "~") or "ecosystem-wide" in scope.lower() or "cross-module" in scope.lower():
        risk = _max_risk(risk, "standard")

    return risk


def _graph_recall_density(graph_context: Optional[list]) -> str:
    """Check Graph recall density. Sparse recall = bump risk up one level.

    The logic: if the Graph has little relevant experience for this type
    of work, we're in unfamiliar territory and need more oversight.
    """
    if graph_context is None:
        return "bump"  # No context at all = unfamiliar

    if len(graph_context) == 0:
        return "bump"  # Graph returned nothing relevant

    # Check average similarity — low similarity means weak matches
    similarities = [r.get("similarity", 0) for r in graph_context]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0

    if avg_similarity < 0.3:
        return "bump"  # Weak matches = unfamiliar territory

    if len(graph_context) < 3 and avg_similarity < 0.5:
        return "bump"  # Few matches, mediocre relevance

    return "no_bump"  # Sufficient recall density


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_risk(
    spec: dict,
    graph_context: Optional[list] = None,
    strategist_override: Optional[str] = None,
) -> str:
    """Classify a spec's risk level.

    Args:
        spec: The WorkBlockSpec to classify
        graph_context: Graph recall results for the spec's domain
        strategist_override: Strategist can escalate (never downgrade)

    Returns:
        One of: "routine", "standard", "sensitive", "command"
    """
    # Content-derived baseline
    risk = _scan_spec_content(spec)
    logger.info("Content-derived risk: %s", risk)

    # Graph recall density bump
    density = _graph_recall_density(graph_context)
    if density == "bump" and risk != "command":
        original = risk
        risk = RISK_LEVELS[min(_risk_index(risk) + 1, len(RISK_LEVELS) - 1)]
        logger.info("Graph recall sparse — bumped %s → %s", original, risk)

    # Strategist override (escalation only, never downgrade)
    if strategist_override:
        override = strategist_override.lower()
        if override in RISK_LEVELS:
            if _risk_index(override) > _risk_index(risk):
                logger.info("Strategist escalated %s → %s", risk, override)
                risk = override
            elif _risk_index(override) < _risk_index(risk):
                logger.warning("Strategist tried to downgrade %s → %s — denied", risk, override)

    return risk


# ---------------------------------------------------------------------------
# Authority map
# ---------------------------------------------------------------------------

# Who can approve at each risk level, and how many retries they get
AUTHORITY_MAP = {
    "routine": {
        "approver": "reviewer",
        "retries": 3,
        "escalates_to": "strategist",
        "co_signer": None,
    },
    "standard": {
        "approver": "reviewer",
        "retries": 2,
        "escalates_to": "strategist",
        "co_signer": None,
    },
    "sensitive": {
        "approver": "reviewer",
        "retries": 2,
        "escalates_to": "qb",
        "co_signer": "razor",  # Security co-sign required
    },
    "command": {
        "approver": "qb",
        "retries": 1,
        "escalates_to": "josh",  # Human in the loop
        "co_signer": "strategist",  # Architecture co-sign
    },
}


def get_authority(risk_level: str) -> dict:
    """Get the authority configuration for a risk level.

    Returns:
        {approver, retries, escalates_to, co_signer}
    """
    return AUTHORITY_MAP.get(risk_level, AUTHORITY_MAP["standard"])


def can_approve(role: str, risk_level: str) -> bool:
    """Check if a role has authority to approve at this risk level."""
    auth = get_authority(risk_level)
    if role == auth["approver"]:
        return True
    if role == "qb":
        return True  # QB can approve anything
    return False


def needs_co_signer(risk_level: str) -> Optional[str]:
    """Check if this risk level needs a co-signer. Returns role name or None."""
    return get_authority(risk_level).get("co_signer")


def get_retry_budget(risk_level: str) -> int:
    """Get the number of retries allowed at this risk level before escalation."""
    return get_authority(risk_level).get("retries", 2)


def get_escalation_target(risk_level: str) -> str:
    """Get who to escalate to when retries are exhausted."""
    return get_authority(risk_level).get("escalates_to", "josh")
