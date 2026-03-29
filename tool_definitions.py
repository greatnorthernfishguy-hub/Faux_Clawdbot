# ---- Changelog ----
# [2026-03-29] Switchblade (TQB / Block E) — Claude-native tool definitions
# What: Centralized tool definitions with JSON Schema parameters for Anthropic API
# Why: PRD Block E — modularize tools so adding a new tool = one list entry
# How: Each tool is a dict with name, description, input_schema (JSON Schema)
# -------------------

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read file content from the workspace. Returns the text contents of the specified file, optionally limited to a line range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (1-based, optional)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Ending line number (1-based, optional)"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Create or update a file in the workspace. REQUIRES a changelog header entry for any code file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to create or update"
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "list_files",
        "description": "Explore directory tree structure. Returns a listing of files and directories at the specified path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the directory (defaults to root)"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth to traverse (default: 3)"
                }
            },
            "required": []
        }
    },
    {
        "name": "search_code",
        "description": "Semantic search across the codebase using NeuroGraph-powered embeddings. Returns the most relevant code snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "n": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_conversations",
        "description": "Search persistent conversation memory using NeuroGraph recall. Finds past exchanges relevant to a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "n": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_testament",
        "description": "Search docs and plans using NeuroGraph recall. Finds documentation snippets relevant to a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "n": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "ingest_workspace",
        "description": "Index or re-index the entire workspace into NeuroGraph memory. Use this to refresh the semantic index after major changes.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "shell_execute",
        "description": "Run a shell command in the workspace environment. Returns stdout and stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "push_to_github",
        "description": "Commit and push the current workspace state to GitHub.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Commit message (default: 'Manual Backup')"
                }
            },
            "required": []
        }
    },
    {
        "name": "pull_from_github",
        "description": "Hard reset the workspace state from a GitHub branch.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {
                    "type": "string",
                    "description": "Branch name to pull from (default: 'main')"
                }
            },
            "required": []
        }
    },
    {
        "name": "create_shadow_branch",
        "description": "Create a backup shadow branch of the current repository state before making risky changes.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "notebook_read",
        "description": "Read the contents of your working memory notebook.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "notebook_add",
        "description": "Add a note to working memory (max 50 notes).",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Note content to add"
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "notebook_delete",
        "description": "Delete a note from working memory by index.",
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Index of the note to delete (0-based)"
                }
            },
            "required": ["index"]
        }
    },
    {
        "name": "map_repository_structure",
        "description": "Analyze and map the full code structure of the repository, including files and function definitions.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_stats",
        "description": "Get current workspace statistics including file counts, conversation counts, and NeuroGraph metrics.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
]
