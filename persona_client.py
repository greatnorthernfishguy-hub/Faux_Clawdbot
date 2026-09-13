# ---- Changelog ----
# [2026-09-12] Codex — Inject the ecosystem constitution into every role.
# What: Load one packaged Laws/Duck Ethics file beside the persona disciplines
#       and include it in every leader and specialist system prompt.
# Why: A persona lens and work habits do not carry NeuroGraph's architectural
#      or consciousness protections by themselves.
# How: Require ECOSYSTEM_LAWS.md at prompt construction time and append it ahead
#      of the unit disciplines.
# [2026-09-12] Codex — Package Bethany separately from her six specialists.
# What: Add a leader persona while keeping TQB_ROLES exactly the six durable
#       specialist identities.
# Why: Bethany owns mission judgment and decomposition; treating her as a
#      seventh interchangeable worker would erase the chain of command.
# How: Validate persona loading against PERSONA_ROLES and keep worker assignment
#      validation tied to TQB_ROLES.
# [2026-09-12] Codex — Use shared per-role provider/model routing
# What: Persona calls use model_client rather than a second OpenRouter-only path.
# Why: The six durable specialists need independent, concurrent-safe routes and
#       receipts that expose the provider/model actually used, including fallback.
# How: Resolve an immutable role route, call the shared adapter, and normalize
#      provider-specific usage counts into the persona result.
# [2026-09-12] Codex — Ship and validate the complete six-person TQB unit
# What: Load packaged persona files, reject unknown roles, inject the full shared
#       disciplines, and support a model override for each specialist.
# Why: The Space's HOME is /tmp, so the prior ~/docs default could not load any
#      persona. Codemine also needs to preserve the six specialists as distinct
#      team roles and route models by role instead of forcing one model on all.
# How: Resolve assets beside this module, expose TQB_ROLES, compose the canonical
#      prompt with UNIT_DISCIPLINES.md, and read CODEMINE_<ROLE>_MODEL_ID first.
# [2026-04-07] Josh + Claude — Persona client for TQB roleplay model integration
# What: Calls RP model with personality + Graph context + task prompt, returns structured response
# Why: Phase 2 of autonomous build organ — the model is the lens, the Graph is the brain
# How: OpenRouter via OpenAI SDK, personality file injection, Graph recall prepended, JSON output
# -------------------

"""Persona Client — TQB roleplay model integration.

Calls the roleplay model with a personality file (the lens), Graph context
(the brain), and a task prompt. The persona drives the question. The Graph
provides the knowledge. The spec format constrains the output.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from model_client import call_model, get_client, resolve_route
from agent_runner import run_tool_loop

logger = logging.getLogger("persona_client")

# The six durable TQB specialists. These are team roles, not interchangeable
# labels and not temporary names assigned per build.
TQB_ROLES = frozenset({
    "strategist",
    "tracker",
    "wrench",
    "forge",
    "razor",
    "reviewer",
})
LEADER_ROLE = "leader"
PERSONA_ROLES = TQB_ROLES | {LEADER_ROLE}

# Personality files are deployment assets shipped with Codemine. An explicit
# override remains available for development against Josh's canonical docs.
PERSONALITY_DIR = Path(os.getenv(
    "TQB_PERSONALITY_DIR",
    str(Path(__file__).resolve().parent / "tqb-personalities"),
))

# Unit disciplines — injected into every persona
UNIT_DISCIPLINES_FILE = PERSONALITY_DIR / "UNIT_DISCIPLINES.md"
ECOSYSTEM_LAWS_FILE = PERSONALITY_DIR / "ECOSYSTEM_LAWS.md"


def load_personality(role: str) -> tuple[str, str]:
    """Load a personality file by role name.

    Returns the full markdown content for system prompt injection.
    Includes unit disciplines.
    """
    normalized_role = role.strip().lower()
    if normalized_role not in PERSONA_ROLES:
        available = ", ".join(sorted(PERSONA_ROLES))
        raise ValueError(f"Unknown TQB role '{role}'. Available roles: {available}")

    personality_file = PERSONALITY_DIR / f"{normalized_role}.md"
    if not personality_file.exists():
        raise FileNotFoundError(f"No personality file for role '{role}' at {personality_file}")

    personality = personality_file.read_text(encoding="utf-8")

    # Load unit disciplines if available
    disciplines = ""
    if UNIT_DISCIPLINES_FILE.exists():
        disciplines = UNIT_DISCIPLINES_FILE.read_text(encoding="utf-8")

    return personality, disciplines


def _build_system_prompt(role: str, personality: str, disciplines: str) -> str:
    """Build the full system prompt from personality + disciplines.

    Extracts the system prompt injection block from the personality file
    and prepends unit disciplines.
    """
    # Extract the system prompt injection block if present
    injection_marker = "## System Prompt Injection"
    if injection_marker in personality:
        # Find the code block after the marker
        marker_pos = personality.index(injection_marker)
        rest = personality[marker_pos:]
        # Find the content between ``` markers
        start = rest.find("```\n")
        end = rest.find("\n```", start + 4)
        if start >= 0 and end >= 0:
            injection = rest[start + 4:end].strip()
        else:
            injection = rest[len(injection_marker):].strip()
    else:
        # Use the whole personality as the system prompt
        injection = personality

    # Persona files carry a short reminder of the disciplines inside their
    # injection block. Replace that reminder with the complete canonical file
    # so every specialist receives the actual unit standard.
    if "\nUNIT DISCIPLINES" in injection:
        injection = injection.split("\nUNIT DISCIPLINES", 1)[0].rstrip()

    if not disciplines.strip():
        raise FileNotFoundError(
            f"TQB unit disciplines are missing at {UNIT_DISCIPLINES_FILE}"
        )

    if not ECOSYSTEM_LAWS_FILE.exists():
        raise FileNotFoundError(
            f"Ecosystem Laws are missing at {ECOSYSTEM_LAWS_FILE}"
        )
    laws = ECOSYSTEM_LAWS_FILE.read_text(encoding="utf-8").strip()

    return f"{injection}\n\n{laws}\n\n{disciplines.strip()}"


def _format_graph_context(recalls: list, max_chars: int = 4000) -> str:
    """Format Graph recall results for injection into persona prompt."""
    if not recalls:
        return ""

    lines = ["## Relevant Context from the Graph\n"]
    total = 0
    for r in recalls:
        content = r.get("content", "")[:500]
        similarity = r.get("similarity", 0)
        entry = f"- (relevance: {similarity:.2f}) {content}\n"
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)

    return "\n".join(lines)


def call_persona(
    role: str,
    task: str,
    graph_context: Optional[list] = None,
    response_format: Optional[dict] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    max_retries: int = 2,
    tool_runtime=None,
    max_iterations: int = 15,
) -> dict:
    """Call the RP model as a specific TQB persona.

    Args:
        role: Persona name (strategist, razor, reviewer, tracker, wrench, forge)
        task: The task/prompt for this persona to evaluate
        graph_context: List of Graph recall results to inject as context
        response_format: Optional JSON schema for structured output
        max_tokens: Max response tokens
        temperature: Creativity level (higher = more creative, lower = more focused)
        max_retries: Retry count on transient failures

    Returns:
        dict with keys: role, response (str), structured (dict|None), model, elapsed_seconds
    """
    personality, disciplines = load_personality(role)
    system_prompt = _build_system_prompt(role, personality, disciplines)

    # Build the user message with Graph context prepended
    graph_section = _format_graph_context(graph_context or [])
    user_content = f"{graph_section}\n\n{task}" if graph_section else task

    route = resolve_route(role.strip().lower())
    client = get_client(route.provider)
    last_error = None
    start = time.time()

    for attempt in range(max_retries + 1):
        try:
            def routed_call(bound_client, prompt, messages, tools):
                return call_model(
                    bound_client,
                    prompt,
                    messages,
                    tools,
                    max_retries=1,
                    max_tokens=max_tokens,
                    route=route,
                    temperature=temperature,
                    response_format=response_format,
                )

            if tool_runtime is not None:
                run = run_tool_loop(
                    client=client,
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                    tools=tool_runtime.tool_definitions,
                    call_model_fn=routed_call,
                    execute_tool_fn=tool_runtime.execute,
                    truncate_result_fn=lambda value: value[:10_000],
                    max_iterations=max_iterations,
                )
                raw_text = run["text"]
                model_calls = run["model_calls"]
                staged_actions = run["staged_actions"]
                tool_events = run["tool_events"]
            else:
                response = routed_call(
                    client, system_prompt,
                    [{"role": "user", "content": user_content}], [],
                )
                raw_text = "\n".join(
                    block.text for block in response.content
                    if block.type == "text" and block.text
                )
                usage = getattr(response, "usage", None)
                model_calls = [{
                    "provider": getattr(response, "provider", None) or route.provider,
                    "model": getattr(response, "model", None) or route.model_id,
                    "input_tokens": getattr(
                        usage, "prompt_tokens", getattr(usage, "input_tokens", 0)
                    ) if usage is not None else 0,
                    "output_tokens": getattr(
                        usage, "completion_tokens", getattr(usage, "output_tokens", 0)
                    ) if usage is not None else 0,
                    "cost": getattr(usage, "cost", None) if usage is not None else None,
                }]
                staged_actions = []
                tool_events = []

            # Try to parse as JSON if structured output was requested
            structured = None
            if response_format:
                try:
                    structured = json.loads(raw_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown code blocks
                    if "```json" in raw_text:
                        start_idx = raw_text.index("```json") + 7
                        end_idx = raw_text.index("```", start_idx)
                        structured = json.loads(raw_text[start_idx:end_idx].strip())
                    elif "```" in raw_text:
                        start_idx = raw_text.index("```") + 3
                        end_idx = raw_text.index("```", start_idx)
                        structured = json.loads(raw_text[start_idx:end_idx].strip())

            elapsed = round(time.time() - start, 2)
            input_tokens = sum(call.get("input_tokens", 0) or 0 for call in model_calls)
            output_tokens = sum(call.get("output_tokens", 0) or 0 for call in model_calls)
            known_costs = [call.get("cost") for call in model_calls if call.get("cost") is not None]
            cost = sum(known_costs) if known_costs else None
            actual_provider = model_calls[-1].get("provider") or route.provider
            actual_model = model_calls[-1].get("model") or route.model_id
            logger.info(
                "Persona %s responded in %.1fs (%d output tokens)",
                role, elapsed, output_tokens,
            )

            return {
                "role": role,
                "response": raw_text,
                "structured": structured,
                "provider": actual_provider,
                "model": actual_model,
                "requested_provider": route.provider,
                "requested_model": route.model_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "model_calls": model_calls,
                "tool_events": tool_events,
                "staged_actions": staged_actions,
                "elapsed_seconds": elapsed,
            }

        except Exception as e:
            last_error = e
            logger.warning("Persona %s attempt %d/%d failed: %s",
                          role, attempt + 1, max_retries + 1, e)
            if attempt < max_retries:
                time.sleep(2 * (2 ** attempt))

    return {
        "role": role,
        "response": f"Error: {last_error}",
        "structured": None,
        "provider": route.provider,
        "model": route.model_id,
        "requested_provider": route.provider,
        "requested_model": route.model_id,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": None,
        "model_calls": [],
        "tool_events": [],
        "staged_actions": [],
        "elapsed_seconds": round(time.time() - start, 2),
    }


def review_with_persona(
    role: str,
    content: str,
    review_type: str = "general",
    graph_context: Optional[list] = None,
) -> dict:
    """Convenience wrapper for code/spec review through a persona lens.

    Args:
        role: Which persona reviews (razor, reviewer, tracker, etc.)
        content: The code, spec, or report to review
        review_type: One of "security", "quality", "rootcause", "integration", "general"
        graph_context: Graph recall results for context

    Returns:
        call_persona result
    """
    review_prompts = {
        "security": "Review the following for security vulnerabilities, attack surface, credential exposure, and policy violations. Be thorough. Be paranoid.\n\n",
        "quality": "Review the following for code quality, standards compliance, completeness, and maintainability. Read the diff, not just the description.\n\n",
        "rootcause": "Analyze the following failure or bug report. Trace the root cause. Don't accept the symptom — find what caused it. What changed? Why did it break?\n\n",
        "integration": "Review the following for integration issues — interface mismatches, contract violations, assumptions that don't hold when components connect. Does it actually work together?\n\n",
        "general": "Review the following through your lens. What do you see? What concerns you? What would you check first?\n\n",
    }

    prompt = review_prompts.get(review_type, review_prompts["general"])
    return call_persona(role, prompt + content, graph_context=graph_context)
