# ---- Changelog ----
# [2026-04-16] Claude (Sonnet 4.6) — Add WorkBlockSpec generation section
#   What: _build_spec_generation_section() added; included in build_system_prompt().
#   Why:  Teaching QB to generate its own specs. Section gives the JSON schema,
#         authoring rules, and a mandatory pre-spec verification workflow.
#         Gate stays human: QB presents JSON, Josh/CC reviews before execution.
#   How:  New section injected after protocols. Contains schema, 5-step workflow,
#         and 7 critical authoring rules distilled from BLK-NG-123/133/137/159.
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
        _build_spec_generation_section(),
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


def _build_spec_generation_section() -> str:
    return """## WorkBlockSpec Generation

When asked to generate a spec for a code change, follow this workflow exactly.
Do not skip steps. Do not draft the spec before completing Steps 1 and 2.

### Step 1 — Verify current code state
Use read_file on every target file in the task description.
- Confirm the described problem still exists in the current code
- Note actual line numbers — punchlist descriptions drift as files change
- Use shell_execute with `grep -c 'pattern' file` to count exact occurrences
- If the problem no longer exists, report that and stop. Do not write a spec for a closed item.

### Step 2 — Check callers
For every function or method being modified, grep the full repo for callers.
The scope of a change includes every call site. A spec that fixes the function
but leaves callers using the old API is broken. Find them all before drafting.

### Step 3 — Draft the spec
Structure:

```json
{
  "spec_version": "1.0.0",
  "block": {
    "id": "BLK-XX-NNN",
    "name": "...",
    "agent": "qwen3-coder",
    "scope": "RepoName",
    "acceptance_criteria": ["..."],
    "depends_on": []
  },
  "constraints": {
    "never": ["..."],
    "anti_drift": ["..."],
    "tool_allowlist": ["shell_execute", "edit_file", "read_file"],
    "shell_allowlist": ["git", "echo", "python3"],
    "max_iterations": 30,
    "timeout_seconds": 600
  },
  "steps": [
    {
      "id": "check_gh_token",
      "type": "action",
      "tool": "shell_execute",
      "params": { "command": "echo \\"GITHUB_TOKEN length: ${#GITHUB_TOKEN}\\"" },
      "validation": { "checks": [
        { "operator": "contains", "value": "GITHUB_TOKEN length: " },
        { "operator": "not_contains", "value": "length: 0" }
      ]},
      "on_failure": "abort"
    },
    {
      "id": "check_already_patched",
      "type": "condition",
      "description": "Skip if already applied (idempotency)",
      "check": { "operator": "file_contains", "value": "path/to/file.py::unique_sentinel_from_new_code" },
      "if_true": [{ "id": "skip", "type": "action", "tool": "shell_execute",
                    "params": { "command": "echo \\"already patched, skipping\\"" },
                    "validation": { "checks": [{ "operator": "contains", "value": "skipping" }] },
                    "on_failure": "continue" }],
      "if_false": [
        {
          "id": "patch_the_file",
          "type": "action",
          "tool": "edit_file",
          "params": { "path": "...", "old_text": "...", "new_text": "..." },
          "validation": { "checks": [{ "operator": "result_is_not_error" }] },
          "on_failure": "abort"
        }
      ]
    }
  ]
}
```

Step types: `action` | `condition` | `gate` | `loop` | `group`
Validation operators: `contains`, `not_contains`, `file_exists`, `file_contains`, `result_is_not_error`, `result_is_string`

### Step 4 — Self-review before presenting
Check every item before outputting the spec:

- [ ] **Rule 1**: Every edit_file old_text was confirmed unique via grep -c (count = 1)
- [ ] **Rule 2**: No old_text appears as a substring inside its new_text — this is the idempotency trap. If new_text preserves part of old_text (e.g. a changelog that keeps the previous entry), the anchor must start at a line that diverges from new_text at the first character. Example: old_text `"# ---- Changelog ----\\n# [2026-03-24]"` is NOT a substring of new_text `"# ---- Changelog ----\\n# [2026-04-16]..."` because they diverge after the header.
- [ ] **Rule 7**: All shell_execute validation checks use `contains`, never `equals`. The executor wraps output as `STDOUT:\\n<output>\\nSTDERR:\\n<err>` — `equals` always fails.
- [ ] **Rule 8**: `echo` is not on the default allowlist. Add it to shell_allowlist whenever used.
- [ ] **Rule 6**: shell_allowlist entries are bare binary names only: `"git"`, `"echo"`, `"rm"`. Never full commands.
- [ ] **Callers**: The spec updates every call site found in Step 2, not just the function definition.
- [ ] **Idempotency**: There is a condition step at the top of each patch block checking a sentinel that only exists after a successful apply.

### Step 5 — Present for approval
Output the complete spec JSON. State clearly what the spec will change and why.
**Do not execute the spec yourself. Wait for human approval before it runs.**"""
