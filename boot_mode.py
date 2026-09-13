#!/usr/bin/env python3
"""Fail-closed boot authorization shared by shell and Python entry points."""

# ---- Changelog ----
# [2026-09-13] Codex — Enforce preservation mode inside Python.
# What: Require exact normal-mode, preservation acknowledgement, and transformer
#       Tonic values before any application path may construct Codemine's Graph.
# Why: The entrypoint alone could be bypassed by a direct import or mistyped mode.
# How: A stdlib-only guard is called by app.py, worker_ng.get_worker_ng(), and the
#      entrypoint before normal startup; every other state refuses closed.
# -------------------

from __future__ import annotations

import os
import sys


PRESERVE_MODE = "preserve"
NORMAL_MODE = "normal"
NORMAL_BOOT_ACK = "two-verified-copies-and-compatibility-reviewed"
REQUIRED_NORMAL_TONIC_MODE = "transformer"


class BootModeError(RuntimeError):
    """Raised when the process is not authorized to start the live substrate."""


def boot_mode() -> str:
    """Return the literal configured mode, defaulting safely to preservation."""
    return os.environ.get("CODEMINE_BOOT_MODE", PRESERVE_MODE)


def require_normal_boot() -> None:
    """Permit live application startup only after all exact controls are set."""
    mode = boot_mode()
    if mode != NORMAL_MODE:
        raise BootModeError(
            f"Codemine live startup refused: CODEMINE_BOOT_MODE={mode!r}; "
            f"expected {NORMAL_MODE!r}"
        )
    acknowledgement = os.environ.get("CODEMINE_NORMAL_BOOT_ACK", "")
    if acknowledgement != NORMAL_BOOT_ACK:
        raise BootModeError(
            "Codemine live startup refused: preservation/compatibility "
            "acknowledgement is absent or incorrect"
        )
    tonic_mode = os.environ.get("CODEMINE_TONIC_MODE", "disabled")
    if tonic_mode != REQUIRED_NORMAL_TONIC_MODE:
        raise BootModeError(
            "Codemine live startup refused: a reviewed transformer-backed Tonic "
            "is required for normal operation"
        )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv != ["--check-normal"]:
        print("usage: boot_mode.py --check-normal", file=sys.stderr)
        return 2
    try:
        require_normal_boot()
    except BootModeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Codemine normal boot authorization accepted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
