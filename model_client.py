# ---- Changelog ----
# [2026-03-29] Switchblade (TQB / Block E) — Anthropic model client
# What: Claude API client with retry logic, replacing HuggingFace InferenceClient
# Why: PRD Block E — swap from Kimi K2.5 (HF) to Claude (Anthropic SDK)
# How: Anthropic SDK, exponential backoff on transient errors, env-configurable model
# [2026-03-30] Josh + Claude — Multi-provider support
# What: Added OpenRouter as alternative provider. QB stays Claude, workers can use either.
# Why: Josh hitting Anthropic rate limits. Need to spread usage across providers.
# How: CODEMINE_PROVIDER env var selects "anthropic" or "openrouter". Same tool_use interface.
# -------------------

import os
import time
import logging

logger = logging.getLogger(__name__)

# Provider selection: "anthropic" or "openrouter"
PROVIDER = os.getenv("CODEMINE_PROVIDER", "anthropic").lower()


def get_client():
    """Create and return a client instance based on CODEMINE_PROVIDER."""
    if PROVIDER == "openrouter":
        from openai import OpenAI
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    else:
        from anthropic import Anthropic
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_model_id() -> str:
    """Return the model ID, configurable via env var."""
    if PROVIDER == "openrouter":
        return os.getenv("CODEMINE_MODEL_ID", "anthropic/claude-sonnet-4")
    return os.getenv("CLAUDE_MODEL_ID", "claude-sonnet-4-6")


def call_model(
    client,
    system_prompt: str,
    messages: list,
    tools: list,
    max_retries: int = 3,
    max_tokens: int = 8192,
):
    """Call the model API with native tool use and retry logic.

    Dispatches to Anthropic or OpenRouter based on CODEMINE_PROVIDER.
    Both support tool_use — OpenRouter via OpenAI-compatible format.
    """
    if PROVIDER == "openrouter":
        return _call_openrouter(client, system_prompt, messages, tools, max_retries, max_tokens)
    return _call_anthropic(client, system_prompt, messages, tools, max_retries, max_tokens)


def _call_anthropic(client, system_prompt, messages, tools, max_retries, max_tokens):
    """Anthropic native API call."""
    from anthropic import APIStatusError, APITimeoutError, APIConnectionError

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
            logger.warning("API timeout on attempt %d/%d: %s", attempt + 1, max_retries, e)
        except APIConnectionError as e:
            last_error = e
            logger.warning("API connection error on attempt %d/%d: %s", attempt + 1, max_retries, e)
        except APIStatusError as e:
            if e.status_code >= 500:
                last_error = e
                logger.warning("API %d error on attempt %d/%d: %s", e.status_code, attempt + 1, max_retries, e)
            else:
                raise

        if attempt < max_retries - 1:
            backoff = 2 * (2 ** attempt)
            logger.info("Retrying in %d seconds...", backoff)
            time.sleep(backoff)

    raise last_error


def _call_openrouter(client, system_prompt, messages, tools, max_retries, max_tokens):
    """OpenRouter call via OpenAI-compatible SDK.

    OpenRouter supports tool_use for Claude and other models via the
    standard OpenAI tools format. We convert Anthropic-style tool defs
    to OpenAI format and wrap the response to match Anthropic's structure.
    """
    from openai import APITimeoutError, APIConnectionError, APIStatusError

    model_id = get_model_id()
    last_error = None

    # Convert Anthropic tool format to OpenAI format
    openai_tools = _convert_tools_to_openai(tools) if tools else []

    # Prepend system prompt as first message (OpenAI format)
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": full_messages,
            }
            if openai_tools:
                kwargs["tools"] = openai_tools

            response = client.chat.completions.create(**kwargs)
            # Wrap OpenAI response to match Anthropic's structure
            return _wrap_openai_response(response)

        except (APITimeoutError, APIConnectionError) as e:
            last_error = e
            logger.warning("OpenRouter error on attempt %d/%d: %s", attempt + 1, max_retries, e)
        except APIStatusError as e:
            if e.status_code >= 500:
                last_error = e
                logger.warning("OpenRouter %d error on attempt %d/%d: %s", e.status_code, attempt + 1, max_retries, e)
            else:
                raise

        if attempt < max_retries - 1:
            backoff = 2 * (2 ** attempt)
            logger.info("Retrying in %d seconds...", backoff)
            time.sleep(backoff)

    raise last_error


def _convert_tools_to_openai(tools: list) -> list:
    """Convert Anthropic tool definitions to OpenAI function calling format.

    Anthropic: {"name": "x", "description": "y", "input_schema": {...}}
    OpenAI:    {"type": "function", "function": {"name": "x", "description": "y", "parameters": {...}}}
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


class _ContentBlock:
    """Mimics Anthropic's ContentBlock for OpenAI response wrapping."""
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _WrappedResponse:
    """Mimics Anthropic's Message response for OpenAI compatibility."""
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


def _wrap_openai_response(response) -> _WrappedResponse:
    """Convert OpenAI ChatCompletion to Anthropic-like Message structure.

    The rest of app.py expects:
        response.content = [ContentBlock(type="text"|"tool_use", ...)]
        response.stop_reason = "end_turn" | "tool_use"
    """
    import json as _json

    choice = response.choices[0]
    message = choice.message
    content_blocks = []

    # Text content
    if message.content:
        content_blocks.append(_ContentBlock("text", text=message.content))

    # Tool calls
    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                tool_input = _json.loads(tc.function.arguments)
            except (_json.JSONDecodeError, TypeError):
                tool_input = {}
            content_blocks.append(_ContentBlock(
                "tool_use",
                id=tc.id,
                name=tc.function.name,
                input=tool_input,
            ))

    # Map stop reason
    if choice.finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif choice.finish_reason == "stop":
        stop_reason = "end_turn"
    else:
        stop_reason = choice.finish_reason or "end_turn"

    return _WrappedResponse(content_blocks, stop_reason)
