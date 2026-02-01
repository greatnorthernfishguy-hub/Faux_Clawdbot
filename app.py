"""
Clawdbot Phase 1 Orchestrator
[CHANGELOG 2026-01-31 - Gemini]
ADDED: HITL Gate for Windows-style "Step-through" approvals.
ADDED: Shadow Branch Failsafe logic via RecursiveContextManager.
ADDED: Vector-Native substrate mandate for E-T Systems.
"""

import gradio as gr
from huggingface_hub import InferenceClient
from recursive_context import RecursiveContextManager
import os, re, json

# --- STATE MANAGEMENT ---
repo_path = os.getenv("REPO_PATH", "/workspace/e-t-systems")
ctx = RecursiveContextManager(repo_path)

class ProposalManager:
    def __init__(self):
        self.pending = []

    def add(self, tool, args):
        # Format for CheckboxGroup
        label = f"{tool}: {args.get('path', args.get('command', 'unknown'))}"
        self.pending.append({"label": label, "tool": tool, "args": args})
        return label

    def get_labels(self):
        return [p["label"] for p in self.pending]

proposals = ProposalManager()

def execute_tool_orchestrated(tool_name, args):
    """Orchestrates tool execution with HITL interrupts."""
    if tool_name in ["write_file", "shell_execute"]:
        # First write in a session triggers shadow branch
        if not proposals.pending:
            ctx.create_shadow_branch()
        
        label = proposals.add(tool_name, args)
        return f"⏳ PROPOSAL STAGED: {label}. Please review in the 'Build Approval' tab."

    # Immediate execution for read-only tools
    mapping = {"search_code": ctx.search_code, "read_file": ctx.read_file}
    return mapping[tool_name](**args) if tool_name in mapping else "Unknown tool."

# --- UI COMPONENTS ---
with gr.Blocks(title="Clawdbot Orchestrator") as demo:
    gr.Markdown("# 🦞 Clawdbot: E-T Systems Orchestrator")
    
    with gr.Tabs() as tabs:
        with gr.Tab("Vibe Chat", id="chat_tab"):
            chatbot = gr.Chatbot(type="messages")
            msg = gr.Textbox(placeholder="Describe the build task...")
            
        with gr.Tab("Build Approval Gate", id="build_tab"):
            gr.Markdown("### 🛠️ Pending Build Proposals")
            gate_list = gr.CheckboxGroup(label="Select actions to execute", choices=[])
            
            with gr.Row():
                btn_exec = gr.Button("✅ Execute Selected", variant="primary")
                btn_all = gr.Button("🚀 Accept All & Build")
                btn_clear = gr.Button("❌ Reject All", variant="stop")
            
            status_out = gr.Markdown("No pending builds.")

    # --- UI LOGIC ---
    def process_selected(selected):
        results = []
        for label in selected:
            for p in proposals.pending:
                if p["label"] == label:
                    res = execute_tool_direct(p["tool"], p["args"])
                    results.append(res)
        proposals.pending = [p for p in proposals.pending if p["label"] not in selected]
        return gr.update(choices=proposals.get_labels()), f"Executed: {len(results)} actions."

    def execute_tool_direct(name, args):
        if name == "write_file": return ctx.write_file(**args)
        if name == "shell_execute": return ctx.shell_execute(**args)

    btn_exec.click(process_selected, inputs=[gate_list], outputs=[gate_list, status_out])
    # Additional event logic would be linked here for 'Accept All' and 'Reject'

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
