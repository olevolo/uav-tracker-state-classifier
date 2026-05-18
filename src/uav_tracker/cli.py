"""Typer-based CLI for the UAV entropy-guided tracker.

Commands (PLAN §11 phase demos):
    * ``doctor`` — environment sanity check (Phase 0 exit demo).
    * ``list-plugins`` — enumerate the four plugin registries (Phase 2).
    * ``evaluate`` — run one experiment on one dataset (Phase 1+).
    * ``ablate``   — signal/scheduler sweep (Phase 5).
    * ``demo``     — render MP4 demo with overlays (Phase 8).

The CLI is deliberately tolerant to missing deps: ``doctor`` must still run
usefully on a bare Python install so the user can see *what's* missing.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="uav-tracker",
    help="UAV Entropy-Guided Tracker CLI.",
    no_args_is_help=True,
    add_completion=False,
)

_console = Console()


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


_REQUIRED_ENV_ROOTS = ("UAV_DATA_ROOT", "UAV_WEIGHTS_ROOT", "UAV_RESULTS_ROOT")


def _check_python() -> tuple[bool, str]:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    return ok, f"Python {major}.{minor}.{sys.version_info.micro}"


def _check_import(modname: str, version_attr: str = "__version__") -> tuple[bool, str]:
    """Try to import ``modname`` and read a version attribute.

    Used for optional heavy deps (numpy, cv2, torch). Never raises.
    """
    try:
        mod = importlib.import_module(modname)
    except Exception as exc:  # pragma: no cover - depends on host env
        return False, f"{modname}: not importable ({exc.__class__.__name__})"
    version = getattr(mod, version_attr, "unknown")
    return True, f"{modname} {version}"


def _check_env_root(var: str) -> tuple[bool, str]:
    raw = os.environ.get(var)
    if not raw:
        return False, f"${var} not set"
    path = Path(raw).expanduser()
    if not path.exists():
        return False, f"${var}={path} does not exist"
    if not os.access(path, os.W_OK):
        return False, f"${var}={path} not writable"
    return True, f"${var}={path} writable"


def _check_ffmpeg() -> tuple[bool, str]:
    exe = shutil.which("ffmpeg")
    if exe is None:
        return False, "ffmpeg not on PATH"
    return True, f"ffmpeg at {exe}"


@app.command()
def doctor() -> None:
    """Run environment diagnostics (Phase 0 exit demo).

    Exits 0 iff all checks pass; otherwise exits 1 and prints failures.
    Never crashes on missing optional deps — warns instead.
    """
    checks: list[tuple[str, tuple[bool, str]]] = [
        ("python", _check_python()),
        ("numpy", _check_import("numpy")),
        ("opencv", _check_import("cv2")),
        ("torch", _check_import("torch")),
        ("ffmpeg", _check_ffmpeg()),
    ]
    for var in _REQUIRED_ENV_ROOTS:
        checks.append((var, _check_env_root(var)))

    table = Table(title="uav-tracker doctor", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    any_failed = False
    for name, (ok, detail) in checks:
        marker = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        if not ok:
            any_failed = True
        table.add_row(name, marker, detail)

    _console.print(table)

    if any_failed:
        _console.print("[red]Doctor found failures. See table above.[/red]")
        raise typer.Exit(code=1)
    _console.print("[green]All checks passed.[/green]")


# ---------------------------------------------------------------------------
# list-plugins
# ---------------------------------------------------------------------------


@app.command("list-plugins")
def list_plugins() -> None:
    """Enumerate the four plugin registries (Phase 2 exit demo).

    Importing ``uav_tracker`` triggers ``_register_plugins`` which in turn
    imports each plugin submodule; their ``@<REGISTRY>.register`` decorators
    fire. We then just read ``.names()`` off each registry.
    """
    # Importing the package triggers plugin registration side effects.
    uav_tracker = importlib.import_module("uav_tracker")

    registries = [
        ("trackers", uav_tracker.TRACKERS),
        ("detectors", uav_tracker.DETECTORS),
        ("signals", uav_tracker.SIGNALS),
        ("schedulers", uav_tracker.SCHEDULERS),
    ]

    table = Table(title="Registered plugins", show_header=True, header_style="bold")
    table.add_column("Kind")
    table.add_column("Names")
    for kind, reg in registries:
        try:
            names = reg.names()
        except Exception:
            names = []
        table.add_row(kind, ", ".join(names) if names else "[dim](none)[/dim]")
    _console.print(table)


# ---------------------------------------------------------------------------
# evaluate (Phase 1+)
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    config: Optional[Path] = typer.Option(None, "--config", help="Hydra experiment config."),
    tracker: Optional[str] = typer.Option(None, "--tracker", help="Registered tracker name."),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Registered dataset name."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only run N sequences."),
    seed: int = typer.Option(42, "--seed", help="Global RNG seed."),
) -> None:
    """Run the OPE evaluation pipeline (Phase 1+).

    Resolves ``--tracker`` and ``--dataset`` by name from the plugin
    registries, constructs an ``OPERunner``, runs the evaluation, and
    prints a Rich table of per-sequence results (Sequence, AUC, Pr@20,
    FPS) plus a summary row.

    ``--config`` accepts a YAML experiment config.  Explicit CLI flags
    override config file values when both are provided.
    """
    # ------------------------------------------------------------------
    # Config file loading (omegaconf)
    # ------------------------------------------------------------------
    cfg: dict[str, Any] = {}
    if config is not None:
        try:
            from omegaconf import OmegaConf  # type: ignore[import]

            raw = OmegaConf.load(str(config))
            cfg = dict(OmegaConf.to_container(raw, resolve=True))  # type: ignore[arg-type]
        except Exception as exc:
            _console.print(f"[red]Failed to load config {config}: {exc}[/red]")
            raise typer.Exit(code=2)

    # Resolve effective values: CLI flags override config
    def _cfg_get(key: str, default: Any = None) -> Any:
        return cfg.get(key, default)

    effective_tracker: str | None = tracker or (
        _cfg_get("tracker", {}).get("name") if isinstance(_cfg_get("tracker"), dict) else _cfg_get("tracker")
    )
    effective_dataset: str | None = dataset or (
        _cfg_get("dataset", {}).get("name") if isinstance(_cfg_get("dataset"), dict) else _cfg_get("dataset")
    )
    effective_limit: int | None = limit if limit is not None else _cfg_get("limit")
    effective_seed: int = seed if seed != 42 else int(_cfg_get("seed", 42))

    # Tracker kwargs from config (args section)
    tracker_kwargs: dict[str, Any] = {}
    tracker_section = _cfg_get("tracker")
    if isinstance(tracker_section, dict):
        tracker_kwargs = dict(tracker_section.get("args", {}) or {})

    # ------------------------------------------------------------------
    # Hybrid runner detection: config has top-level 'trackers' dict key
    # ------------------------------------------------------------------
    trackers_section = _cfg_get("trackers")
    detectors_section = _cfg_get("detectors")
    signals_section = _cfg_get("signals")
    scheduler_section = _cfg_get("scheduler")
    is_hybrid = (
        isinstance(trackers_section, dict)
        and signals_section is not None
        and scheduler_section is not None
    )

    # Trigger plugin registration.
    import uav_tracker  # noqa: F401 — side-effect: _register_plugins()
    from uav_tracker.registry import TRACKERS, DATASETS, SIGNALS, SCHEDULERS
    from uav_tracker.evaluation.ope import OPERunner

    if is_hybrid:
        _run_hybrid_evaluate(
            cfg=cfg,
            trackers_section=trackers_section,
            detectors_section=detectors_section,
            signals_section=signals_section,
            scheduler_section=scheduler_section,
            effective_dataset=effective_dataset,
            effective_limit=effective_limit,
            effective_seed=effective_seed,
        )
        return

    # ------------------------------------------------------------------
    # Single-tracker path (Phase 1 / 2)
    # ------------------------------------------------------------------
    if effective_tracker is None:
        _console.print("[red]--tracker is required (or set tracker.name in config).[/red]")
        raise typer.Exit(code=1)
    if effective_dataset is None:
        _console.print("[red]--dataset is required (or set dataset.name in config).[/red]")
        raise typer.Exit(code=1)

    try:
        tracker_obj = TRACKERS.build(effective_tracker, **tracker_kwargs)
    except KeyError as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except TypeError as exc:
        _console.print(f"[red]Tracker construction failed: {exc}[/red]")
        raise typer.Exit(code=1)

    try:
        dataset_obj = (
            DATASETS.build(effective_dataset, seed=effective_seed)
            if effective_dataset == "synthetic"
            else DATASETS.build(effective_dataset)
        )
    except KeyError as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except TypeError:
        try:
            dataset_obj = DATASETS.build(effective_dataset)
        except KeyError as exc:
            _console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)

    runner = OPERunner(seed=effective_seed)
    result = runner.run(tracker=tracker_obj, dataset=dataset_obj, limit=effective_limit)

    table = Table(
        title=f"OPE: {effective_tracker} on {effective_dataset}",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Sequence")
    table.add_column("AUC", justify="right")
    table.add_column("Pr@20", justify="right")
    table.add_column("FPS", justify="right")

    for sr in result.per_sequence:
        table.add_row(
            sr.name,
            f"{sr.auc:.3f}",
            f"{sr.precision_at_20:.3f}",
            f"{sr.fps:.1f}",
        )

    # Summary row.
    table.add_section()
    table.add_row(
        "[bold]MEAN[/bold]",
        f"[bold]{result.auc:.3f}[/bold]",
        f"[bold]{result.precision_at_20:.3f}[/bold]",
        f"[bold]{result.fps:.1f}[/bold]",
    )

    _console.print(table)


def _run_hybrid_evaluate(
    cfg: dict[str, Any],
    trackers_section: dict[str, Any],
    signals_section: list[dict[str, Any]],
    scheduler_section: dict[str, Any],
    effective_dataset: str | None,
    effective_limit: int | None,
    effective_seed: int,
    detectors_section: dict[str, Any] | None = None,
) -> None:
    """Build a HybridRunner from config and run OPE."""
    from uav_tracker.registry import TRACKERS, DATASETS, SIGNALS, SCHEDULERS, DETECTORS
    from uav_tracker.runner import HybridRunner
    from uav_tracker.evaluation.ope import OPERunner

    if effective_dataset is None:
        _console.print("[red]--dataset is required (or set dataset.name in config).[/red]")
        raise typer.Exit(code=1)

    # Build tier trackers.  Keys in YAML may be ints or strings.
    tier_trackers: dict[int, Any] = {}
    for tier_key, tracker_cfg in trackers_section.items():
        tier = int(tier_key)
        t_name = tracker_cfg.get("name")
        t_args = dict(tracker_cfg.get("args") or {})
        try:
            tier_trackers[tier] = TRACKERS.build(t_name, **t_args)
        except KeyError as exc:
            _console.print(f"[red]Unknown tracker {t_name!r}: {exc}[/red]")
            raise typer.Exit(code=1)
        except TypeError as exc:
            _console.print(f"[red]Tracker {t_name!r} construction failed: {exc}[/red]")
            raise typer.Exit(code=1)

    # Build tier detectors (Phase 6: detectors: section).
    tier_detectors: dict[int, Any] = {}
    if detectors_section:
        for tier_key, det_cfg in detectors_section.items():
            tier = int(tier_key)
            d_name = det_cfg.get("name")
            d_args = dict(det_cfg.get("args") or {})
            try:
                tier_detectors[tier] = DETECTORS.build(d_name, **d_args)
            except KeyError as exc:
                _console.print(f"[red]Unknown detector {d_name!r}: {exc}[/red]")
                _console.print("[yellow]Hint: ensure Engineer A's yolo plugin is importable.[/yellow]")
                raise typer.Exit(code=1)
            except TypeError as exc:
                _console.print(f"[red]Detector {d_name!r} construction failed: {exc}[/red]")
                raise typer.Exit(code=1)

    # Build signals.
    sig_list = []
    for sig_cfg in (signals_section if isinstance(signals_section, list) else [signals_section]):
        s_name = sig_cfg.get("name")
        s_args = dict(sig_cfg.get("args") or {})
        try:
            sig_list.append(SIGNALS.build(s_name, **s_args))
        except KeyError as exc:
            _console.print(f"[red]Unknown signal {s_name!r}: {exc}[/red]")
            raise typer.Exit(code=1)

    # Build scheduler.
    sched_name = scheduler_section.get("name")
    sched_args = dict(scheduler_section.get("args") or {})
    try:
        scheduler_obj = SCHEDULERS.build(sched_name, **sched_args)
    except KeyError as exc:
        _console.print(f"[red]Unknown scheduler {sched_name!r}: {exc}[/red]")
        raise typer.Exit(code=1)
    except TypeError as exc:
        _console.print(f"[red]Scheduler {sched_name!r} construction failed: {exc}[/red]")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Phase 14 v2 ML optional sections (additive; missing = no-op)
    # ------------------------------------------------------------------
    scene_classifier_obj: Any = None
    scene_classifier_section = cfg.get("scene_classifier")
    if scene_classifier_section:
        from uav_tracker.registry import SCENE_CLASSIFIERS
        sc_name = scene_classifier_section.get("name")
        sc_args = {
            k: v for k, v in scene_classifier_section.items() if k != "name"
        }
        try:
            scene_classifier_obj = SCENE_CLASSIFIERS.build(sc_name, **sc_args)
            _console.print(f"[dim]scene_classifier: {sc_name}[/dim]")
        except Exception as exc:
            _console.print(
                f"[yellow]scene_classifier {sc_name!r} failed to build: {exc} — skipping[/yellow]"
            )

    appearance_memory_obj: Any = None
    appearance_memory_section = cfg.get("appearance_memory")
    if appearance_memory_section:
        from uav_tracker.registry import APPEARANCE_MEMORIES
        am_name = appearance_memory_section.get("name")
        am_args = {
            k: v for k, v in appearance_memory_section.items() if k != "name"
        }
        try:
            appearance_memory_obj = APPEARANCE_MEMORIES.build(am_name, **am_args)
            _console.print(f"[dim]appearance_memory: {am_name}[/dim]")
        except Exception as exc:
            _console.print(
                f"[yellow]appearance_memory {am_name!r} failed to build: {exc} — skipping[/yellow]"
            )

    motion_predictor_obj: Any = None
    motion_predictor_section = cfg.get("motion_predictor")
    if motion_predictor_section:
        from uav_tracker.registry import MOTION_PREDICTORS
        mp_name = motion_predictor_section.get("name")
        mp_args = {
            k: v for k, v in motion_predictor_section.items() if k != "name"
        }
        try:
            motion_predictor_obj = MOTION_PREDICTORS.build(mp_name, **mp_args)
            _console.print(f"[dim]motion_predictor: {mp_name}[/dim]")
        except Exception as exc:
            _console.print(
                f"[yellow]motion_predictor {mp_name!r} failed to build: {exc} — skipping[/yellow]"
            )

    warmer_obj: Any = None
    warmer_section = cfg.get("warmer")
    if warmer_section:
        from uav_tracker.registry import ML_WARMERS
        w_name = warmer_section.get("name")
        w_args = {
            k: v for k, v in warmer_section.items() if k != "name"
        }
        try:
            warmer_obj = ML_WARMERS.build(w_name, **w_args)
            _console.print(f"[dim]warmer: {w_name}[/dim]")
        except Exception as exc:
            _console.print(
                f"[yellow]warmer {w_name!r} failed to build: {exc} — skipping[/yellow]"
            )

    hybrid_runner = HybridRunner(
        trackers=tier_trackers,
        signals=sig_list,
        scheduler=scheduler_obj,
        detectors=tier_detectors if tier_detectors else None,
        seed=effective_seed,
        warmer=warmer_obj,
        scene_classifier=scene_classifier_obj,
        appearance_memory=appearance_memory_obj,
        motion_predictor=motion_predictor_obj,
    )

    # Build dataset.
    try:
        dataset_obj = (
            DATASETS.build(effective_dataset, seed=effective_seed)
            if effective_dataset == "synthetic"
            else DATASETS.build(effective_dataset)
        )
    except KeyError as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except TypeError:
        try:
            dataset_obj = DATASETS.build(effective_dataset)
        except KeyError as exc:
            _console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)

    ope = OPERunner(seed=effective_seed)
    result = ope.run(tracker=hybrid_runner, dataset=dataset_obj, limit=effective_limit)
    total_recoveries = getattr(hybrid_runner, "recoveries", 0)

    # Determine if any sequence reported hybrid tier data.
    has_hybrid = any("time_in_tier" in sr.aux for sr in result.per_sequence)
    has_detectors = bool(tier_detectors)

    table = Table(
        title=f"OPE (hybrid): {sched_name} on {effective_dataset}",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Sequence")
    table.add_column("AUC", justify="right")
    table.add_column("Pr@20", justify="right")
    table.add_column("FPS", justify="right")
    if has_hybrid:
        table.add_column("tier1 %", justify="right")
    if has_detectors:
        table.add_column("recoveries", justify="right")

    for sr in result.per_sequence:
        tier_pct = ""
        if has_hybrid:
            tit = sr.aux.get("time_in_tier", {})
            t1 = tit.get(1, 0)
            total = sum(tit.values())
            tier_pct = f"{100.0 * t1 / total:.1f}" if total > 0 else "0.0"
        row = [sr.name, f"{sr.auc:.3f}", f"{sr.precision_at_20:.3f}", f"{sr.fps:.1f}"]
        if has_hybrid:
            row.append(tier_pct)
        if has_detectors:
            row.append(str(sr.aux.get("recoveries", 0)))
        table.add_row(*row)

    # Aggregate tier1 % across sequences.
    table.add_section()
    if has_hybrid:
        total_t1 = sum(
            sr.aux.get("time_in_tier", {}).get(1, 0) for sr in result.per_sequence
        )
        total_all = sum(
            sum(sr.aux.get("time_in_tier", {}).values()) for sr in result.per_sequence
        )
        mean_t1_pct = f"[bold]{100.0 * total_t1 / total_all:.1f}[/bold]" if total_all > 0 else "[bold]0.0[/bold]"
        summary_row = [
            "[bold]MEAN[/bold]",
            f"[bold]{result.auc:.3f}[/bold]",
            f"[bold]{result.precision_at_20:.3f}[/bold]",
            f"[bold]{result.fps:.1f}[/bold]",
            mean_t1_pct,
        ]
        if has_detectors:
            summary_row.append(f"[bold]{total_recoveries}[/bold]")
        table.add_row(*summary_row)
    else:
        summary_row = [
            "[bold]MEAN[/bold]",
            f"[bold]{result.auc:.3f}[/bold]",
            f"[bold]{result.precision_at_20:.3f}[/bold]",
            f"[bold]{result.fps:.1f}[/bold]",
        ]
        if has_detectors:
            summary_row.append(f"[bold]{total_recoveries}[/bold]")
        table.add_row(*summary_row)

    _console.print(table)
    if has_detectors:
        _console.print(f"[cyan]Total detector recoveries: {total_recoveries}[/cyan]")


# ---------------------------------------------------------------------------
# ablate (Phase 5)
# ---------------------------------------------------------------------------


def _load_sweep_config(sweep_path: Path) -> dict[str, Any]:
    """Load ablation sweep YAML (plain PyYAML — no Hydra dependency)."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        _console.print("[red]PyYAML is required for ablate. Install with: pip install pyyaml[/red]")
        raise typer.Exit(code=2)
    with open(sweep_path) as fh:
        return dict(yaml.safe_load(fh))


def _run_ablation_variant(
    variant: dict[str, Any],
    base: dict[str, Any],
    sweep_type: str,
    effective_limit: int | None,
    effective_seed: int,
) -> dict[str, Any]:
    """Run a single ablation variant and return a result dict."""
    from uav_tracker.registry import TRACKERS, DATASETS, SIGNALS, SCHEDULERS
    from uav_tracker.runner import HybridRunner
    from uav_tracker.evaluation.ope import OPERunner

    variant_name = variant.get("name", "unnamed")

    # Build trackers from base config.
    trackers_cfg = base.get("trackers", {})
    tier_trackers: dict[int, Any] = {}
    for tier_key, tracker_cfg in trackers_cfg.items():
        tier = int(tier_key)
        t_name = tracker_cfg.get("name")
        t_args = dict(tracker_cfg.get("args") or {})
        tier_trackers[tier] = TRACKERS.build(t_name, **t_args)

    # Build signal(s).
    if sweep_type == "signals":
        sig_cfg = variant.get("signal", {})
        s_name = sig_cfg.get("name")
        s_args = dict(sig_cfg.get("args") or {})
        sig_obj = SIGNALS.build(s_name, **s_args)
        signals_list = [sig_obj]
        # Update scheduler's signal_name for this variant.
        sched_cfg = dict(base.get("scheduler", {}))
        sched_args = dict(sched_cfg.get("args") or {})
        sched_args["signal_name"] = variant.get("scheduler_signal_name", s_name)
        sched_name = sched_cfg.get("name", "hysteresis_binary")
        scheduler_obj = SCHEDULERS.build(sched_name, **sched_args)
    else:  # sweep_type == "schedulers"
        base_sig_cfg = base.get("signal", {})
        s_name = base_sig_cfg.get("name")
        s_args = dict(base_sig_cfg.get("args") or {})
        sig_obj = SIGNALS.build(s_name, **s_args)
        signals_list = [sig_obj]
        sched_cfg = variant.get("scheduler", {})
        sched_name = sched_cfg.get("name")
        sched_args = dict(sched_cfg.get("args") or {})
        scheduler_obj = SCHEDULERS.build(sched_name, **sched_args)

    hybrid_runner = HybridRunner(
        trackers=tier_trackers,
        signals=signals_list,
        scheduler=scheduler_obj,
        seed=effective_seed,
    )

    # Build dataset.
    dataset_cfg = base.get("dataset", {})
    ds_name = dataset_cfg.get("name", "synthetic")
    if ds_name == "synthetic":
        dataset_obj = DATASETS.build(ds_name, seed=effective_seed)
    else:
        dataset_obj = DATASETS.build(ds_name)

    ope = OPERunner(seed=effective_seed)
    result = ope.run(tracker=hybrid_runner, dataset=dataset_obj, limit=effective_limit)

    # Count switches across sequences.
    total_switches = sum(
        sr.aux.get("time_in_tier", {}).get(1, 0) for sr in result.per_sequence
    )

    return {
        "variant": variant_name,
        "auc": result.auc,
        "pr20": result.precision_at_20,
        "fps": result.fps,
        "n_seq": len(result.per_sequence),
    }


@app.command()
def ablate(
    sweep: Optional[Path] = typer.Option(None, "--sweep", help="Ablation sweep YAML."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only run N sequences."),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Run an ablation sweep over signals/schedulers (Phase 5).

    Reads a sweep YAML that specifies a base hybrid config and a list of
    variants. For each variant, builds the HybridRunner, runs OPE, and
    prints a Rich comparison table. Per-variant CSVs are written to
    ``$UAV_RESULTS_ROOT/ablation_<timestamp>/`` (or ``results/ablation_<ts>/``).
    """
    if sweep is None:
        _console.print("[red]--sweep is required.[/red]")
        raise typer.Exit(code=1)
    if not sweep.exists():
        _console.print(f"[red]Sweep config not found: {sweep}[/red]")
        raise typer.Exit(code=2)

    import datetime
    import csv

    # Trigger plugin registration.
    import uav_tracker  # noqa: F401

    cfg = _load_sweep_config(sweep)
    sweep_type = cfg.get("sweep_type", "signals")
    base = cfg.get("base", {})
    variants = cfg.get("variants", [])
    effective_seed = int(cfg.get("seed", seed))

    if not variants:
        _console.print("[red]No variants found in sweep config.[/red]")
        raise typer.Exit(code=1)

    # Resolve output directory.
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    from uav_tracker.paths import outputs_root
    out_dir = outputs_root() / "ablation" / f"ablation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _console.print(
        f"[bold]Phase 5 ablation sweep[/bold]: {sweep_type} | "
        f"{len(variants)} variants | output → {out_dir}"
    )

    # Run each variant.
    rows: list[dict[str, Any]] = []
    for variant in variants:
        vname = variant.get("name", "unnamed")
        _console.print(f"  Running variant [cyan]{vname}[/cyan] …")
        try:
            row = _run_ablation_variant(
                variant=variant,
                base=base,
                sweep_type=sweep_type,
                effective_limit=limit,
                effective_seed=effective_seed,
            )
        except Exception as exc:
            _console.print(f"    [red]FAILED: {exc}[/red]")
            row = {"variant": vname, "auc": float("nan"), "pr20": float("nan"), "fps": 0.0, "n_seq": 0}

        rows.append(row)

        # Write per-variant CSV.
        csv_path = out_dir / f"{vname}.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["variant", "auc", "pr20", "fps", "n_seq"])
            writer.writeheader()
            writer.writerow(row)

    # Print comparison table.
    table = Table(
        title=f"Ablation: {sweep_type} ({Path(sweep).stem})",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Variant")
    table.add_column("AUC", justify="right")
    table.add_column("Pr@20", justify="right")
    table.add_column("FPS", justify="right")
    table.add_column("N seqs", justify="right")

    for row in rows:
        auc_str = f"{row['auc']:.3f}" if row['auc'] == row['auc'] else "ERROR"
        pr20_str = f"{row['pr20']:.3f}" if row['pr20'] == row['pr20'] else "ERROR"
        table.add_row(
            row["variant"],
            auc_str,
            pr20_str,
            f"{row['fps']:.1f}",
            str(row["n_seq"]),
        )

    _console.print(table)
    _console.print(f"[green]Results written to {out_dir}[/green]")


# ---------------------------------------------------------------------------
# restart-eval (Phase 7)
# ---------------------------------------------------------------------------


@app.command("restart-eval")
def restart_eval(
    tracker: Optional[str] = typer.Option(None, "--tracker", help="Registered tracker name."),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Registered dataset name."),
    threshold: float = typer.Option(0.5, "--threshold", help="IoU failure threshold."),
    restart_gap: int = typer.Option(5, "--restart-gap", help="Frames to skip after restart."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only run N sequences."),
    seed: int = typer.Option(42, "--seed", help="Global RNG seed."),
) -> None:
    """Run restart-based OPE (PLAN §10, OTB restart protocol) — Phase 7.

    Mirrors ``evaluate`` but uses ``RestartOPE`` which re-initialises the
    tracker after a failure (IoU < ``--threshold``) rather than continuing
    blindly.  Produces a per-sequence table of success_rate and n_restarts
    plus summary totals.

    Requires ``--tracker`` and ``--dataset``.
    """
    if tracker is None:
        _console.print("[red]--tracker is required.[/red]")
        raise typer.Exit(code=1)
    if dataset is None:
        _console.print("[red]--dataset is required.[/red]")
        raise typer.Exit(code=1)

    # Trigger plugin registration.
    import uav_tracker  # noqa: F401
    from uav_tracker.registry import TRACKERS, DATASETS
    from uav_tracker.metrics.restart_ope import RestartOPE

    try:
        tracker_obj = TRACKERS.build(tracker)
    except KeyError as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except TypeError as exc:
        _console.print(f"[red]Tracker construction failed: {exc}[/red]")
        raise typer.Exit(code=1)

    try:
        dataset_obj = (
            DATASETS.build(dataset, seed=seed)
            if dataset == "synthetic"
            else DATASETS.build(dataset)
        )
    except KeyError as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except TypeError:
        try:
            dataset_obj = DATASETS.build(dataset)
        except KeyError as exc:
            _console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)

    runner = RestartOPE(threshold=threshold, restart_gap=restart_gap)
    result = runner.run(tracker=tracker_obj, dataset=dataset_obj, limit=limit)

    table = Table(
        title=f"Restart-OPE: {tracker} on {dataset} (thr={threshold}, gap={restart_gap})",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Sequence")
    table.add_column("SuccessRate", justify="right")
    table.add_column("N_restarts", justify="right")
    table.add_column("N_frames", justify="right")

    for sr in result.per_sequence:
        table.add_row(
            sr.name,
            f"{sr.success_rate:.3f}",
            str(sr.n_restarts),
            str(sr.n_frames),
        )

    table.add_section()
    table.add_row(
        "[bold]MEAN / TOTAL[/bold]",
        f"[bold]{result.mean_success_rate:.3f}[/bold]",
        f"[bold]{result.total_restarts}[/bold]",
        "",
    )

    _console.print(table)
    _console.print(
        f"[cyan]Total restarts: {result.total_restarts} | "
        f"Mean success rate: {result.mean_success_rate:.3f}[/cyan]"
    )


# ---------------------------------------------------------------------------
# figures (Phase 8)
# ---------------------------------------------------------------------------


@app.command()
def figures(
    result_dir: Path = typer.Option(..., "--result-dir", help="Directory of benchmark result CSVs."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Directory where PNG figures are written (default: <repo>/outputs/figures)."),
) -> None:
    """Generate paper figures from benchmark result CSVs (Phase 8).

    For each CSV in ``--result-dir``:
      - If it contains tier / signal columns, renders an entropy-timeline PNG.
      - Always renders success-curve and precision-curve PNGs using the
        per-sequence AUC / Pr@20 data.

    Output filenames mirror the CSV stem under ``--out-dir``.
    """
    import csv as _csv

    if not result_dir.exists():
        _console.print(f"[red]--result-dir does not exist: {result_dir}[/red]")
        raise typer.Exit(code=1)

    if out_dir is None:
        from uav_tracker.paths import outputs_root
        out_dir = outputs_root() / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    from uav_tracker.viz import (
        plot_entropy_timeline,
        plot_success_curve,
        plot_precision_curve,
    )
    from uav_tracker.evaluation.ope import OPEResult, SequenceResult

    csv_files = sorted(result_dir.glob("*.csv"))
    if not csv_files:
        _console.print(f"[yellow]No CSV files found in {result_dir}[/yellow]")
        raise typer.Exit(code=0)

    for csv_path in csv_files:
        stem = csv_path.stem
        _console.print(f"Processing [cyan]{csv_path.name}[/cyan] …")

        # Read CSV rows.
        try:
            with open(csv_path, newline="") as fh:
                reader = _csv.DictReader(fh)
                rows = [row for row in reader if not row.get("sequence", "").startswith("#")]
        except Exception as exc:
            _console.print(f"  [red]Failed to read {csv_path}: {exc}[/red]")
            continue

        if not rows:
            _console.print(f"  [yellow]Empty CSV, skipping.[/yellow]")
            continue

        # ------------------------------------------------------------------
        # Detect entropy-timeline columns and plot if present.
        # ------------------------------------------------------------------
        has_entropy = any("H_bar" in r or "entropy" in r or "signal" in r.get("sequence", "") for r in rows)
        first_keys = set(rows[0].keys())
        has_tier_col = "tier" in first_keys or "H_bar" in first_keys or "entropy" in first_keys

        if has_tier_col:
            try:
                import numpy as np

                n = len(rows)
                H_bar = np.zeros(n)
                tier_seq = np.zeros(n, dtype=int)

                for i, row in enumerate(rows):
                    H_bar[i] = float(row.get("H_bar", row.get("entropy", 0.0)) or 0.0)
                    tier_seq[i] = int(float(row.get("tier", 0) or 0))

                # Infer E_hi / E_lo from data range.
                E_hi = float(np.percentile(H_bar, 75)) if len(H_bar) > 0 else 0.65
                E_lo = float(np.percentile(H_bar, 25)) if len(H_bar) > 0 else 0.50

                # Switch events: frames where tier changes.
                switch_events = [i for i in range(1, n) if tier_seq[i] != tier_seq[i - 1]]

                et_path = out_dir / f"{stem}_entropy_timeline.png"
                plot_entropy_timeline(
                    H_bar=H_bar,
                    tier_sequence=tier_seq,
                    E_hi=E_hi,
                    E_lo=E_lo,
                    switch_events=switch_events,
                    out_path=et_path,
                )
                _console.print(f"  Entropy timeline → [green]{et_path}[/green]")
            except Exception as exc:
                _console.print(f"  [yellow]Entropy timeline skipped: {exc}[/yellow]")

        # ------------------------------------------------------------------
        # Build OPEResult and plot success + precision curves.
        # ------------------------------------------------------------------
        try:
            seq_results: list[SequenceResult] = []
            for row in rows:
                seq_name = row.get("sequence", row.get("name", "seq"))
                try:
                    auc_val = float(row.get("auc", 0.0) or 0.0)
                    pr20_val = float(row.get("pr20", row.get("precision_at_20", 0.0)) or 0.0)
                    fps_val = float(row.get("fps", 0.0) or 0.0)
                except (ValueError, TypeError):
                    auc_val, pr20_val, fps_val = 0.0, 0.0, 0.0
                seq_results.append(
                    SequenceResult(
                        name=seq_name,
                        auc=auc_val,
                        precision_at_20=pr20_val,
                        fps=fps_val,
                        n_frames=0,
                    )
                )

            if seq_results:
                import numpy as np

                mean_auc = float(np.mean([r.auc for r in seq_results]))
                mean_pr20 = float(np.mean([r.precision_at_20 for r in seq_results]))
                mean_fps = float(np.mean([r.fps for r in seq_results]))
                ope_result = OPEResult(
                    auc=mean_auc,
                    precision_at_20=mean_pr20,
                    fps=mean_fps,
                    per_sequence=seq_results,
                )
                tracker_label = stem

                sc_path = out_dir / f"{stem}_success_curve.png"
                plot_success_curve([ope_result], out_path=sc_path, labels=[tracker_label])
                _console.print(f"  Success curve    → [green]{sc_path}[/green]")

                pc_path = out_dir / f"{stem}_precision_curve.png"
                plot_precision_curve([ope_result], out_path=pc_path, labels=[tracker_label])
                _console.print(f"  Precision curve  → [green]{pc_path}[/green]")
        except Exception as exc:
            _console.print(f"  [yellow]OPE curves skipped: {exc}[/yellow]")

    _console.print(f"[green]Figures written to {out_dir}[/green]")


# ---------------------------------------------------------------------------
# demo (Phase 8)
# ---------------------------------------------------------------------------


@app.command()
def demo(
    sequence: Optional[str] = typer.Option(None, "--sequence", help="Sequence name."),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Registered dataset name."),
    config: Optional[Path] = typer.Option(None, "--config", help="Experiment YAML config."),
    tracker: Optional[str] = typer.Option(None, "--tracker", help="Registered tracker name."),
    out: Optional[Path] = typer.Option(None, "--out", help="Output MP4 path."),
    fps: int = typer.Option(30, "--fps", help="Output video frame-rate."),
) -> None:
    """Render a demo MP4 with bbox overlays and mode badges (Phase 8).

    Loads the named sequence from ``--dataset``, runs the tracker/hybrid
    per-frame, annotates each frame via ``viz.draw_frame_overlay``, and
    writes the result to ``--out`` via ``viz.write_mp4``.

    Use ``--config`` OR ``--tracker``/``--dataset`` (CLI flags take priority
    over config values when both are provided).

    Note: requires Engineer B's overlay/video modules
    (``uav_tracker.viz.overlay`` + ``uav_tracker.viz.video``).
    """
    # ------------------------------------------------------------------
    # Config / CLI resolution
    # ------------------------------------------------------------------
    cfg: dict[str, Any] = {}
    if config is not None:
        try:
            from omegaconf import OmegaConf  # type: ignore[import]

            raw = OmegaConf.load(str(config))
            cfg = dict(OmegaConf.to_container(raw, resolve=True))  # type: ignore[arg-type]
        except Exception as exc:
            _console.print(f"[red]Failed to load config {config}: {exc}[/red]")
            raise typer.Exit(code=2)

    def _cfg_get(key: str, default: Any = None) -> Any:
        return cfg.get(key, default)

    effective_tracker: str | None = tracker or (
        _cfg_get("tracker", {}).get("name")
        if isinstance(_cfg_get("tracker"), dict)
        else _cfg_get("tracker")
    )
    effective_dataset: str | None = dataset or (
        _cfg_get("dataset", {}).get("name")
        if isinstance(_cfg_get("dataset"), dict)
        else _cfg_get("dataset")
    )
    effective_sequence: str | None = sequence or _cfg_get("sequence")
    effective_out: Path | None = out or (Path(_cfg_get("out")) if _cfg_get("out") else None)

    if effective_dataset is None:
        _console.print("[red]--dataset is required.[/red]")
        raise typer.Exit(code=1)
    if effective_sequence is None:
        _console.print("[red]--sequence is required.[/red]")
        raise typer.Exit(code=1)
    if effective_out is None:
        import datetime as _datetime
        from uav_tracker.paths import demos_root
        ts = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        effective_out = demos_root() / f"{effective_sequence}_{effective_tracker or 'tracker'}_{ts}.mp4"
        effective_out.parent.mkdir(parents=True, exist_ok=True)
        _console.print(f"[dim]--out not given; defaulting to {effective_out}[/dim]")
    if effective_tracker is None:
        _console.print("[red]--tracker (or tracker.name in config) is required.[/red]")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Lazy imports (overlay/video may be absent if Engineer B isn't done)
    # ------------------------------------------------------------------
    import uav_tracker  # noqa: F401 — side-effect: _register_plugins()
    from uav_tracker.registry import TRACKERS, DATASETS
    import uav_tracker.viz as _viz

    draw_fn = getattr(_viz, "draw_frame_overlay", None)
    write_fn = getattr(_viz, "write_mp4", None)

    if draw_fn is None or write_fn is None:
        _console.print(
            "[red]viz.draw_frame_overlay / viz.write_mp4 are not available.[/red]\n"
            "[yellow]Engineer B's overlay/video modules have not been installed yet.[/yellow]"
        )
        raise typer.Exit(code=3)

    # ------------------------------------------------------------------
    # Build tracker
    # ------------------------------------------------------------------
    tracker_kwargs: dict[str, Any] = {}
    tracker_section = _cfg_get("tracker")
    if isinstance(tracker_section, dict):
        tracker_kwargs = dict(tracker_section.get("args") or {})

    try:
        tracker_obj = TRACKERS.build(effective_tracker, **tracker_kwargs)
    except KeyError as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except TypeError as exc:
        _console.print(f"[red]Tracker construction failed: {exc}[/red]")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Load dataset and find the named sequence
    # ------------------------------------------------------------------
    try:
        dataset_obj = DATASETS.build(effective_dataset)
    except KeyError as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    target_seq = None
    for seq in dataset_obj:
        if seq.name == effective_sequence:
            target_seq = seq
            break

    if target_seq is None:
        _console.print(
            f"[red]Sequence {effective_sequence!r} not found in dataset {effective_dataset!r}.[/red]"
        )
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Per-frame tracking loop
    # ------------------------------------------------------------------
    frames_list = list(target_seq.frames)
    gt_bboxes = target_seq.ground_truth

    if len(frames_list) < 2:
        _console.print("[red]Sequence has fewer than 2 frames.[/red]")
        raise typer.Exit(code=1)

    tracker_obj.init(frames_list[0], gt_bboxes[0])

    annotated_frames: list[Any] = []
    for frame in frames_list[1:]:
        state = tracker_obj.update(frame)
        bbox = state.bbox
        tier = getattr(state, "tier", 0)
        signals = getattr(state, "signals", {})
        annotated = draw_fn(frame, bbox, tier, signals, fps)
        annotated_frames.append(annotated)

    # ------------------------------------------------------------------
    # Write MP4
    # ------------------------------------------------------------------
    effective_out.parent.mkdir(parents=True, exist_ok=True)
    write_fn(annotated_frames, effective_out, fps=fps)

    _console.print(f"[green]Demo written to {effective_out}[/green]")


def main() -> None:  # pragma: no cover - thin entrypoint
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
