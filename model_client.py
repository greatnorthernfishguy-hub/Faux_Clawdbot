# ---- Changelog ----
# [2026-03-29] Switchblade (TQB / Block E) — Anthropic model client
# What: Claude API client with retry logic, replacing HuggingFace InferenceClient
# Why: PRD Block E — swap from Kimi K2.5 (HF) to Claude (Anthropic SDK)
# How: Anthropic SDK, exponential backoff on transient errors, env-configurable model
# -------------------

import os
import time
import logging

from anthropic import Anthropic, APIStatusError, APITimeoutError, APIConnectionError

logger = logging.getLogger(__name__)


def get_client() -> Anthropic:
    """Create and return an Anthropic client instance."""
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_model_id() -> str:
    """Return the model ID, configurable via CLAUDE_MODEL_ID env var."""
    return os.getenv("CLAUDE_MODEL_ID", "claude-sonnet-4-6")


def call_model(
    client: Anthropic,
    system_prompt: str,
    messages: list,
    tools: list,
    max_retries: int = 3,
    max_tokens: int = 8192,
):
    """Call the Claude API with native tool use and retry logic.

    Args:
        client: Anthropic client instance.
        system_prompt: System prompt string.
        messages: Conversation message list (role/content dicts).
        tools: Claude-native tool definitions list from tool_definitions.py.
        max_retries: Number of retries on transient errors (5xx, timeout).
        max_tokens: Maximum tokens in the response.

    Returns:
        The Anthropic Message response object.

    Raises:
        APIStatusError: On non-transient API errors (4xx).
        APITimeoutError: After all retries exhausted on timeout.
        APIConnectionError: After all retries exhausted on connection failure.
    """
    model_id = get_model_id()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tools if tools else [],
            )
            return response

        except APITimeoutError as e:
            last_error = e
            logger.warning(
                "API timeout on attempt %d/%d: %s", attempt + 1, max_retries, e
            )
        except APIConnectionError as e:
            last_error = e
            logger.warning(
                "API connection error on attempt %d/%d: %s",
                attempt + 1, max_retries, e,
            )
        except APIStatusError as e:
            if e.status_code >= 500:
                last_error = e
                logger.warning(
                    "API %d error on attempt %d/%d: %s",
                    e.status_code, attempt + 1, max_retries, e,
                )
            else:
                # 4xx errors are not transient — raise immediately
                raise

        if attempt < max_retries - 1:
            backoff = 2 * (2 ** attempt)
            logger.info("Retrying in %d seconds...", backoff)
            time.sleep(backoff)

    # All retries exhausted
    raise last_error
