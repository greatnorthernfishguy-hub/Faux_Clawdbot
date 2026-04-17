# ---- Changelog ----
# [2026-04-16] Claude (Sonnet 4.6) — Add HuggingFace Inference API as primary provider
# What: "huggingface" provider added; auto-fallback to OpenRouter on 402 (credits exhausted)
# Why: Leverage HF more; OpenRouter stays as backup. Explicit user request.
# How: Same OpenAI-compat path as OpenRouter. _call_huggingface() catches 402 and retries
#      via _call_openrouter(). HF_MODEL_ID env var for HF model name (format differs from OR).
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

# Provider selection: "anthropic", "openrouter", or "huggingface"
PROVIDER = os.getenv("CODEMINE_PROVIDER", "anthropic").lower()


def get_client():
    """Create and return a client instance based on CODEMINE_PROVIDER."""
    if PROVIDER == "openrouter":
        from openai import OpenAI
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    elif PROVIDER == "huggingface":
        from openai import OpenAI
        return OpenAI(
            base_url="https://api-inference.huggingface.co/v1",
            api_key=os.getenv("HF_TOKEN"),
        )
    else:
        from anthropic import Anthropic
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_model_id() -> str:
    """Return the model ID, configurable via env var.

    HF model IDs use HuggingFace hub format (e.g. Qwen/Qwen3-Coder-480B-A35B-Instruct).
    OpenRouter uses its own slug format (e.g. qwen/qwen3-coder).
    Use HF_MODEL_ID and CODEMINE_MODEL_ID to override each independently.
    """
    if PROVIDER == "huggingface":
        return os.getenv("HF_MODEL_ID", "Qwen/Qwen3-Coder-480B-A35B-Instruct")
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
    if PROVIDER == "huggingface":
        return _call_huggingface(client, system_prompt, messages, tools, max_retries, max_tokens)
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


def _convert_messages_to_openai(messages: list) -> list:
    """Convert Anthropic-style messages to OpenAI chat format.

    Anthropic uses content block arrays for tool_use (assistant) and
    tool_result (user) messages. OpenAI uses tool_calls on the assistant
    message and separate role="tool" messages for results.
    """
    import json as _json
    converted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Simple string content — pass through
        if isinstance(content, str):
            if content:
                converted.append({"role": role, "content": content})
            continue

        # List content — Anthropic content blocks
        if not isinstance(content, list):
            continue

        if role == "assistant":
            # Extract text and tool_use blocks
            text_parts = []
            tool_calls = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text" and block.get("text"):
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": _json.dumps(block.get("input", {})),
                            },
                        })
            assistant_msg = {"role": "assistant"}
            assistant_msg["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            converted.append(assistant_msg)

        elif role == "user":
            # Could be tool_result blocks or mixed content
            tool_results = []
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_result":
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": str(block.get("content", "")),
                        })
                    elif block.get("type") == "text" and block.get("text"):
                        text_parts.append(block["text"])
                elif isinstance(block, str) and block:
                    text_parts.append(block)
            if text_parts:
                converted.append({"role": "user", "content": "\n".join(text_parts)})
            converted.extend(tool_results)

    return converted


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

    # Convert Anthropic-style messages to OpenAI format and prepend system prompt
    full_messages = [{"role": "system", "content": system_prompt}] + _convert_messages_to_openai(messages)

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


def _call_huggingface(client, system_prompt, messages, tools, max_retries, max_tokens):
    """HuggingFace Inference API call via OpenAI-compatible SDK.

    Same interface as OpenRouter. On 402 (credits exhausted or model unavailable),
    automatically falls back to OpenRouter so runs don't silently die.
    """
    from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError

    model_id = get_model_id()
    openai_tools = _convert_tools_to_openai(tools) if tools else []
    full_messages = [{"role": "system", "content": system_prompt}] + _convert_messages_to_openai(messages)
    last_error = None

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
            return _wrap_openai_response(response)

        except APIStatusError as e:
            if e.status_code == 402:
                logger.warning(
                    "HF Inference API 402 (credits/quota). Falling back to OpenRouter."
                )
                or_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
                return _call_openrouter(or_client, system_prompt, messages, tools, max_retries, max_tokens)
            elif e.status_code >= 500:
                last_error = e
                logger.warning("HF %d error on attempt %d/%d: %s", e.status_code, attempt + 1, max_retries, e)
            else:
                raise

        except (APITimeoutError, APIConnectionError) as e:
            last_error = e
            logger.warning("HF connection error on attempt %d/%d: %s", attempt + 1, max_retries, e)

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
