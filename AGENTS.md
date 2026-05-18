# AGENTS — Governance & Git Workflow

> Terse cross-agent reference. Authoritative source for scope, roles, and git rules is `PLAN.md` §5 and §7a plus each agent's full brief in `agents/*.md`. This file only summarizes — when in doubt, follow PLAN.md.

---

## Roles

Three cooperating Claude agents build the UAV Entropy-Guided Tracker. Each owns a disjoint slice of the repo. Crossing into another agent's paths requires a handoff.

### Architect / Coordinator — `agents/architect.md`
- **Owns:** `PLAN.md`, `AGENTS.md`, `docs/adr/**`, `docs/paper/**`, `docs/research/**`, `docs/status/**`, `configs/experiments/**`, every `*/base.py`, `src/uav_tracker/types.py`, PR/issue templates, `CODEOWNERS` (joint with DevOps).
- **Decides:** module boundaries, Protocol shapes, plugin contract, experiment design, methodological-fidelity verdicts, resolution of `PLAN.md` §16 open questions via ADRs.
- **Bright lines — never:** write production code under `src/uav_tracker/{trackers,detectors,signals,schedulers,metrics,viz}/` beyond base stubs; touch CI workflows, Dockerfiles, or lockfiles; gate merges on paper numerical targets; self-merge a PR.

### Engineer — `agents/engineer.md`
- **Owns:** all non-`base.py` modules under `src/uav_tracker/` (trackers, detectors, signals, schedulers, runner, metrics, evaluation, viz, kalman, `cli.py`, `__init__.py`), `tests/**`, `notebooks/**`, eval scripts under `scripts/run_*.py`, `results/**`.
- **Decides:** algorithm implementation choices within Protocol contracts, test design for invariants and behavior, numerical tuning within the fidelity contract (PLAN §10).
- **Bright lines — never:** edit Protocols or `types.py` (open an ADR request with Architect); author CI workflows or Docker files; commit dataset/weight binaries; self-merge.

### DevOps — `agents/devops.md`
- **Owns:** `pyproject.toml`, `uv.lock`, `Makefile`, `.pre-commit-config.yaml`, `.envrc.example`, `.gitignore`, `CHANGELOG.md`, `LICENSE`, `CODEOWNERS` (joint), `.github/workflows/**`, `.github/settings.yml`, `.github/ISSUE_TEMPLATE/**`, PR template skeleton, `infra/**`, `scripts/download_*.py`, `scripts/provision_*.py`, Dockerfiles.
- **Decides:** build/dep/lock/image pipelines, CI matrix, runner topology, branch-protection rules, release cadence, edge provisioning.
- **Bright lines — never:** edit algorithmic code under `src/uav_tracker/`; change Protocols; relax branch protection or bypass required reviews; skip signed-commits or pre-commit hooks; self-merge.

---

## Governance

### Handoff
- Architect authors an ADR (MADR 3.0) or PLAN update whenever a decision crosses an agent boundary.
- Engineer asks Architect to revise a Protocol before implementing deviations from it.
- DevOps consults Architect before adding/removing CI gates that affect the methodological-fidelity contract.

### Acceptance gates (per phase)
1. Phase demo command runs cleanly end-to-end (see `PLAN.md` §11).
2. All CI status checks green (`ci`, `smoke-eval`, `plugin-contract`, `security-review` on DevOps paths).
3. Architect posts acceptance-review comment on the phase-tracking issue citing demo output + test evidence.
4. All ADR open questions on the phase's critical path are closed (`PLAN.md` §16).
5. `docs/status/phase-N.md` recorded.

---

## Git Workflow

Trunk-based; short-lived feature branches; squash-merge into `master`. Full detail in `PLAN.md` §7a.

### Reviewer matrix (PLAN §7a.5)
| PR author | Required reviewer | Optional reviewer |
|---|---|---|
| Engineer | Architect | DevOps (if infra surface touched) |
| DevOps | Architect | Engineer (if DX/runtime affected) |
| Architect — docs-only (`PLAN`, `AGENTS`, `docs/**`, `agents/**`) | Engineer or DevOps | — |
| Architect — `*/base.py` or `types.py` | Engineer | DevOps (if registry impacted) |

An agent never merges their own PR — even a typo fix — and never bypasses review.

### Branch naming (PLAN §7a.1)
`<agent-prefix>/<scope>-<slug>`. Agent prefixes: `architect/`, `engineer/`, `devops/`. Scope prefixes: `adr`, `plan`, `doc`, `phase`, `eng`, `infra`, `ci`, `release`, `hotfix`, `research`. Examples: `architect/adr-0004-registry`, `engineer/eng-0003-kcf`, `devops/ci-smoke-eval`.

### Merge rules (PLAN §7a.4)
- Squash-merge only; linear `master` history.
- `master` requires: ≥1 cross-agent approval, all required status checks green, up-to-date with `master` (merge queue when available), signed commits, no direct push, no force-push, no admin-bypass.
- Don't force-push a PR branch once review comments have arrived (reviewers lose their place).

### Phase close-outs (PLAN §11 preamble)
At the end of each phase, a `chore(phase-N): close-out — <summary>` **commit** on `master` adds a dated `### Phase N — YYYY-MM-DD — <name>` block to `CHANGELOG.md` under `## [Unreleased]`, pasting the exit-demo output and listing what shipped + any ADRs. No feature branches, no PR, no git tag — the commit + CHANGELOG block is the record. (Shifts to a `chore(phase-N): close-out` PR once a remote + branch protection are live — see PLAN §7a.)

### Never-do list (PLAN §7a.10)
- Push directly to `master` or any protected release branch.
- Force-push `master`.
- Merge your own PR (trivial content included).
- Bypass CI, hooks, or required reviews.
- Disable branch protection to expedite a merge.
- Use `--no-verify`, `--no-gpg-sign`, or analogous flags.

---

## Contact / Escalation

- Cross-agent disagreement > 24 h → Architect posts decision comment referencing an ADR (PLAN §7a.6).
- Red `master` → merging agent reverts within 30 min (PLAN §7a.9). Reverts are single-reviewer fast-track.
- Hotfix required → DevOps branches from latest release tag, labels PR `hotfix`, Architect may single-approve (PLAN §7a.8). Post-mortem in `docs/incidents/YYYY-MM-DD-<slug>.md` within 48 h.

For the full charter (workflows, deliverables, checklist, bright lines) read `agents/architect.md`, `agents/engineer.md`, `agents/devops.md`.
