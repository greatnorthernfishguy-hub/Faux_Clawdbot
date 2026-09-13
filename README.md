---
title: Working in a Codemine
emoji: 🦞
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Working in a Codemine

Codemine is the Hugging Face-hosted coding organ directed by Bethany Anne
Reynolds, Queen Bitch. Bethany holds the mission and delegates bounded work to
six durable Team Queen Bitch specialists:

| Specialist | Primary lens |
|---|---|
| Strategist | Decomposition, dependencies, architecture, and specs |
| Tracker | Root cause, regressions, and evidence chains |
| Wrench | Unconventional solutions and neglected edge cases |
| Forge | Implementation, assembly, and integration |
| Razor | Security, policy, and attack surface |
| Reviewer | Quality, completeness, and routine approval |

Any specialist can perform implementation work when a MissionPlan gives that
block explicit write paths. A role supplies a lens; it does not grant authority.

Bethany and all six specialists use **one shared Codemine NeuroGraph**. The
personas are stable viewpoints over one learned substrate, not separate agents
with private Graphs. The Graph carries experience and recall across model
changes. Recalled material is memory and must be checked against current files.

> **Modernization hold:** The live Space must remain stopped or in preservation
> mode until its existing third-hemisphere NeuroGraph has been copied off the
> Space and independently verified. Follow [PRESERVATION.md](PRESERVATION.md).
> The modernization branch is not authorization to wake, restore, migrate, or
> normally start the Space.

## Current architecture

Codemine has three user-facing paths:

- **Chat** uses the shared model/tool loop for interactive investigation.
- **Spec** retains the legacy deterministic WorkBlockSpec executor for existing
  work orders.
- **Mission** uses the new plan-then-execute flow. Bethany inspects the actual
  repository, creates a mechanically validated MissionPlan, and persists it for
  review. A separate Execute action runs approved blocks.

Mission execution applies these boundaries:

1. The source checkout may be dirty; Codemine records its status and starts from
   its committed `HEAD` without modifying it.
2. Every block receives its own Git worktree and branch.
3. Blocks in the same dependency level run concurrently only when the plan's
   write sets are mechanically disjoint.
4. Model tools can write only beneath that block's declared relative paths.
5. Models select validation IDs from the environment-owned validation catalog; they cannot
   invent executable commands.
6. The coordinator reruns the selected validations itself.
7. Reviewer reads the actual diff. Sensitive work also requires Razor; command
   work requires the explicit human gate plus Strategist and Bethany review.
8. Approved block commits are cherry-picked into a separate integration
   worktree.
9. The mission stops at `ready_for_external_review`. It does not merge the
   source branch, push GitHub, deploy, or restart a service.

Each proposal and run has an atomic JSON receipt under
`CODEMINE_MISSION_ROOT`. Receipts include the plan, dependency levels, requested
and actual provider/model routes, token and cost data when the provider returns
it, tool events, validation output, reviews, commits, and preserved worktree
paths.

## Model routing

Provider and model choices are environment-owned and resolved independently for
every call. The image currently defaults Bethany, Strategist, Wrench, Razor,
and Reviewer to `moonshotai/kimi-k3`; Tracker and Forge default to
`z-ai/glm-5.3-flash`. These are routing defaults, not identity. Any role can be
moved to Anthropic, OpenRouter, or Hugging Face without changing its persona or
shared NeuroGraph.

For a role named `TRACKER`, for example:

```text
CODEMINE_TRACKER_PROVIDER=openrouter
CODEMINE_TRACKER_MODEL_ID=z-ai/glm-5.3-flash
```

The corresponding variables exist for `LEADER`, `STRATEGIST`, `TRACKER`,
`WRENCH`, `FORGE`, `RAZOR`, and `REVIEWER`. Shared fallbacks remain available
through the existing provider configuration. Every result records the route
requested and the provider/model actually used.

## Runtime configuration

| Variable | Purpose | Image default |
|---|---|---|
| `CODEMINE_BOOT_MODE` | `preserve` runs preservation only; `normal` permits application startup | `normal` |
| `NEUROGRAPH_WORKSPACE_DIR` | One shared Codemine Graph workspace | `/data/neurograph_worker` |
| `CODEMINE_PRESERVATION_DIR` | Same-volume preservation archives | `/data/codemine-preservation` |
| `CODEMINE_PERSISTENT_ROOT` | Required persistent mount | `/data` |
| `CODEMINE_MISSION_ROOT` | Durable proposals, receipts, and recovery patches | `/data/codemine-missions` |
| `CODEMINE_WORKTREE_ROOT` | Scratch Git worktrees used during a live run | `/tmp/codemine-worktrees` |
| `CODEMINE_DEFAULT_WORKSPACE` | Default mission target repository | `/workspace/e-t-systems` |
| `CODEMINE_WORKSPACE_ROOTS` | Colon-separated roots that Mission Control may target | `/workspace` |
| `CODEMINE_VALIDATIONS_JSON` | Coordinator-owned named validation commands | packaged JSON value |
| `CODEMINE_MAX_PARALLEL` | Maximum workers in one safe dependency level | `3` |
| `CODEMINE_VALIDATION_TIMEOUT` | Seconds allowed for one named validation | `300` |
| `TQB_PERSONALITY_DIR` | Packaged leader and specialist prompts | `/app/tqb-personalities` |

Secrets belong in Space secrets. At least one configured model provider needs
its corresponding credential, such as `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`,
or `HF_TOKEN`. The target repository clone is configured with `REPO_URL`.

## Persistent substrate

The application uses the full NeuroGraph implementation in this repository and
its dual-pass embedding path. It does not use ChromaDB and does not create a
Graph for each worker. Tool results and persona outputs are deposited as raw
experience before downstream classification or summarization.

The existing HF graph predates this modernization. Its checkpoint and vector
sidecar must be preserved and inspected before normal startup. Compatibility of
its Tonic and embedding path is a separate restore decision; do not substitute
heuristic execution for embeddings.

## Development checks

Run the local test suite without starting a model or NeuroGraph:

```bash
python3 -m pytest -q
python3 -m py_compile mission_plan.py mission_generator.py mission_dispatcher.py mission_service.py
```

The dispatcher tests use disposable Git repositories and fake model clients.
They verify dirty-source preservation, scoped writes, command gating,
validation, Reviewer iteration, isolated commits, integration, durable receipts,
and duplicate-execution refusal.

## Known boundaries

- The modernized path is prepared locally and has not yet been deployed to the
  live Space.
- Final integration requires an independent high-level/model-family review and,
  for NeuroGraph ecosystem work, a Law Enforcer audit.
- The superseded linear orchestrator and generator are retained only under
  `Defunct-Historical`; they are absent from the runtime image.
- The initial validation catalog intentionally contains only named checks that
  the coordinator owns. Add project-specific checks deliberately rather than
  allowing a planning model to author commands.
- Normal boot remains blocked by the preservation and compatibility gates.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the gated delivery sequence.
