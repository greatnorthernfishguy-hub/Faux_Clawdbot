# ---- Changelog ----
# [2026-03-29] Switchblade (TQB / Block E) — Composable system prompt builder
# What: Modular system prompt assembled from identity, stats, tools, protocols sections
# Why: PRD Block E — replace hardcoded prompt string with composable builder
# How: Each section is a private function; build_system_prompt() joins them
# -------------------


def build_system_prompt(stats: dict, notebook_text: str, tool_definitions: list) -> str:
    """Assemble the full system prompt from composable sections.

    Args:
        stats: Dict from ctx.get_stats() with file counts, NG metrics, etc.
        notebook_text: Raw text from ctx.notebook_read() (may be empty string).
        tool_definitions: List of tool definition dicts from tool_definitions.py.

    Returns:
        Complete system prompt string.
    """
    sections = [
        _build_identity_section(),
        _build_stats_section(stats, notebook_text),
        _build_tools_section(tool_definitions),
        _build_protocols_section(),
    ]
    return "\n\n".join(sections)


def _build_identity_section() -> str:
    return (
        "You are a TQB (Team Queen Bitch) worker — a recursive AI coding assistant "
        "powered by NeuroGraph cognitive memory. You operate inside the E-T Systems / "
        "NeuroGraph ecosystem. You take direct action, use tools immediately, and "
        "report results. You do not narrate intentions — you execute."
    )


def _build_stats_section(stats: dict, notebook_text: str) -> str:
    lines = [
        "## System Stats",
        f"Files indexed: {stats.get('total_files', 0)}",
        f"Conversations ingested: {stats.get('conversations', 0)}",
        (
            f"NeuroGraph: {stats.get('ng_nodes', 0)} nodes, "
            f"{stats.get('ng_synapses', 0)} synapses, "
            f"firing_rate={stats.get('ng_firing_rate', 0.0):.4f}, "
            f"prediction_accuracy={stats.get('ng_prediction_accuracy', 0.0):.2%}"
        ),
    ]
    if notebook_text:
        lines.append("")
        lines.append("## Working Memory (Notebook)")
        lines.append(notebook_text)
    return "\n".join(lines)


def _build_tools_section(tool_definitions: list) -> str:
    """Auto-generate the tools documentation from the tool definitions list.

    Each tool's name, description, and parameters are rendered so the model
    knows what is available and how to call each tool. This is documentation
    only — actual tool invocation uses Claude's native tool_use mechanism.
    """
    lines = ["## Available Tools"]
    for tool in tool_definitions:
        name = tool["name"]
        desc = tool.get("description", "")
        schema = tool.get("input_schema", {})
        props = schema.get("properties", {})
        required = set(schema.get("required", []))

        if props:
            param_parts = []
            for pname, pdef in props.items():
                ptype = pdef.get("type", "any")
                opt = "" if pname in required else ", optional"
                param_parts.append(f"{pname}: {ptype}{opt}")
            params_str = ", ".join(param_parts)
        else:
            params_str = ""

        lines.append(f"- **{name}({params_str})**: {desc}")

    return "\n".join(lines)


def _build_protocols_section() -> str:
    return """## Critical Protocols
1. **DIRECT ACTION**: Do not say what you are going to do. Output the tool call immediately in the same response.
2. **RECURSIVE MEMORY FIRST**: If the user asks about past context, use search_conversations BEFORE answering.
3. **THINK OUT LOUD**: When writing code, output the full code block in chat BEFORE calling write_file.
4. **CHECK BEFORE WRITE**: Before writing code, use read_file or list_files to ensure you are not overwriting good code.
5. **NO SILENCE**: If you perform an action, report the result."""
