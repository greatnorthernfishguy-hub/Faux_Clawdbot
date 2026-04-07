# ---- Changelog ----
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

logger = logging.getLogger("persona_client")

# Personality files directory
PERSONALITY_DIR = Path(os.getenv(
    "TQB_PERSONALITY_DIR",
    os.path.expanduser("~/docs/queen-bitch/tqb-personalities")
))

# Unit disciplines — injected into every persona
UNIT_DISCIPLINES_FILE = PERSONALITY_DIR / "UNIT_DISCIPLINES.md"


def _get_rp_client():
    """Create OpenRouter client for the RP model."""
    from openai import OpenAI
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


def _get_rp_model_id() -> str:
    """Return the RP model ID from env."""
    return os.getenv("CODEMINE_RP_MODEL_ID", "nousresearch/hermes-3-llama-3.1-70b")


def load_personality(role: str) -> str:
    """Load a personality file by role name.

    Returns the full markdown content for system prompt injection.
    Includes unit disciplines.
    """
    personality_file = PERSONALITY_DIR / f"{role.lower()}.md"
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

    return injection


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

    client = _get_rp_client()
    model_id = _get_rp_model_id()
    last_error = None
    start = time.time()

    for attempt in range(max_retries + 1):
        try:
            kwargs = {
                "model": model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)
            raw_text = response.choices[0].message.content or ""

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
            logger.info("Persona %s responded in %.1fs (%d tokens)", role, elapsed,
                       response.usage.completion_tokens if response.usage else 0)

            return {
                "role": role,
                "response": raw_text,
                "structured": structured,
                "model": model_id,
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
        "model": model_id,
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
