"""Backward-compatible entry point: delegates to run_with_csc --tracker sglatrack.

Original flags mapped:
  --tracker_checkpoint  → --weights_path
  --tracker_config      → dropped (unused in run_with_csc)
  --csc_config          → dropped (CSC config is embedded in the checkpoint)
  all other flags       → forwarded unchanged
"""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import run_with_csc  # noqa: E402


def _translate_argv(argv: list[str]) -> list[str]:
    out: list[str] = ["--tracker", "sglatrack"]
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--tracker_checkpoint":
            if i + 1 < len(argv):
                out.extend(["--weights_path", argv[i + 1]])
                i += 2
                continue
        elif tok in ("--tracker_config", "--csc_config"):
            # drop the flag and its value
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
    sys.exit(run_with_csc.main(translated))
