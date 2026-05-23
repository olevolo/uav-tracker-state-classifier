"""Backward-compatible entry point: delegates to run_baseline --tracker sglatrack.

All original CLI flags are forwarded unchanged.  The only transformation is
that ``--checkpoint`` is mapped to ``--weights_path`` (the generic runner uses
the latter), and ``--config`` is silently dropped (it was unused in the original).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the tools/ directory is on sys.path so ``import run_baseline`` works
# whether this script is invoked directly or via the test harness.
_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import run_baseline  # noqa: E402


def _translate_argv(argv: list[str]) -> list[str]:
    """Map legacy ``run_sglatrack_baseline`` flags to ``run_baseline`` flags."""
    out: list[str] = ["--tracker", "sglatrack"]
    skip_next = False
    i = 0
    while i < len(argv):
        tok = argv[i]
        if skip_next:
            skip_next = False
            i += 1
            continue
        if tok == "--checkpoint":
            # rename to --weights_path
            if i + 1 < len(argv):
                out.extend(["--weights_path", argv[i + 1]])
                i += 2
                continue
        elif tok == "--config":
            # drop silently (was unused)
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                i += 2
                continue
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


if __name__ == "__main__":
    translated = _translate_argv(sys.argv[1:])
    sys.exit(run_baseline.main(translated))
