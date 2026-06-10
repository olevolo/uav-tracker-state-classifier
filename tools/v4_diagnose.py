#!/usr/bin/env python
"""CSC-v4 module A11 (tool 1) — offline separability / calibration diagnostics.

Quick, GT-aware, OFFLINE check of whether the v4 telemetry features actually
*separate* the new failure SUBTYPES the v4 labels introduce — BEFORE training a
model on them. It answers two questions for a passive run dir:

  (1) Per-feature **rank-AUROC** for the two subtype splits that motivate v4:
        FC split : true-FC  vs  false-FC   (is a confident wrong peak distinguishable
                   from a confident *correct* peak / a non-FC frame?)
        LA split : true-loss vs false-LA   (when the target is truly gone vs when CSC
                   over-fires LA on a still-tracked target — the documented LA wall)
      AUROC is the Mann-Whitney rank statistic (== probability a random positive
      scores above a random negative); >0.5 means the feature is informative, 0.5
      is chance, and we report it oriented so AUROC>0.5 always = "higher value
      indicates the positive subtype" (sign is auto-flipped + reported).

  (2) **Calibrator coverage**: for each response feature, fit an A1
      V4FeatureCalibrator on this run's telemetry and report finite-sample count,
      median/IQR, and the percentile-mapped coverage of the observed values (what
      fraction land in the well-supported [0.02, 0.98] CDF band vs saturating at
      the 0/1 tails). Flags features that are degenerate (near-constant -> can't
      calibrate / won't transfer), mirroring the A1 negative-transfer motivation.

GT subtype proxies (this tool, OFFLINE)
---------------------------------------
The v4 label module (A4 `csc_lib/csc/v4/labeling_v4.py`) is not built yet, so the
positive/negative *subtype* assignments here are computed locally from GT + flags
+ telemetry as cheap proxies (marked ``# APPROX`` / ``# INTEGRATION:``). Once A4
lands, ``--use_v4_labels`` can swap these for the real ``build_v4_labels`` output;
the AUROC / coverage machinery is unchanged. We import the v4 enums for the
canonical subtype meanings so the proxy stays interface-faithful.

CLI
---
  python tools/v4_diagnose.py --passive_dir <run> --dataset uav123
  python tools/v4_diagnose.py --selftest          # synthetic, NO dataset

Read-only; offline single-object-tracking benchmark diagnostics only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---- sys.path: mirror tools/la_smoke.py header EXACTLY (live tracker shadows) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    sys.path.insert(0, str(_p))

# v4 shared enums (single source of truth for subtype meanings).
from csc_lib.csc.v4.v4types import FCSubtype, LASubtype  # noqa: E402

# A1 calibrators (coverage stats). Import is cheap (numpy/json only).
from csc_lib.csc.v4.features_v4 import (  # noqa: E402
    RESPONSE_FEATURES,
    V4FeatureCalibrator,
)


# ----------------------------------------------------------------------
# Core math (pure numpy; tested by --selftest with NO dataset).
# ----------------------------------------------------------------------
def rank_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney rank-AUROC of ``scores`` against binary ``labels`` (1=pos).

    AUROC = P(score(pos) > score(neg)), tie-corrected via average ranks. NaN
    scores are dropped (paired with their label). Returns ``nan`` if either
    class is empty after dropping NaNs.

    This is the exact rank statistic (no thresholds, no sklearn): rank all
    finite scores ascending (ties -> mean rank), then
    ``AUROC = (sum_pos_ranks - n_pos*(n_pos+1)/2) / (n_pos*n_neg)``.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    if scores.shape != labels.shape:
        raise ValueError(f"rank_auroc shape mismatch: {scores.shape} vs {labels.shape}")
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # average ranks within tied groups
    s_sorted = scores[order]
    i = 0
    n = s_sorted.size
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0  # mean of ranks (1-based)
        i = j
    sum_pos = float(ranks[labels].sum())
    auroc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auroc)


def oriented_auroc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    """AUROC oriented so the returned value is >= 0.5.

    Returns ``(auroc, sign)`` where ``sign == +1`` means "higher score => positive"
    and ``sign == -1`` means the feature was flipped (lower score => positive).
    NaN stays NaN with sign ``+1``.
    """
    a = rank_auroc(scores, labels)
    if not np.isfinite(a):
        return a, 1
    if a < 0.5:
        return 1.0 - a, -1
    return a, 1


def coverage_stats(
    values: np.ndarray, cal: V4FeatureCalibrator, lo: float = 0.02, hi: float = 0.98
) -> dict:
    """Calibrator coverage for one feature's observed values.

    Reports finite-sample count, median/IQR (from the fitted calibrator), the
    fraction of observed values whose percentile lands in the well-supported
    ``[lo, hi]`` CDF band (vs saturating at the 0/1 tails), and a degeneracy flag
    (IQR ~ 0 -> near-constant feature, can't normalize / won't transfer).
    """
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return {"n": 0, "median": float("nan"), "iqr": float("nan"),
                "band_frac": float("nan"), "tail_frac": float("nan"), "degenerate": True}
    pcts = np.fromiter((cal.percentile(float(v)) for v in vals), dtype=np.float64, count=n)
    in_band = float(((pcts >= lo) & (pcts <= hi)).mean())
    at_tail = float(((pcts < lo) | (pcts > hi)).mean())
    iqr = float(cal._iqr)  # noqa: SLF001  (read-only diagnostic; A1 exposes no getter)
    return {"n": n, "median": float(cal._median), "iqr": iqr,  # noqa: SLF001
            "band_frac": in_band, "tail_frac": at_tail,
            "degenerate": bool(iqr < 1e-9)}


# ----------------------------------------------------------------------
# GT subtype PROXIES (offline; replaced by A4 build_v4_labels when available).
#
# These are deliberately simple, GT-grounded definitions of the two subtype
# *splits* the AUROC report needs. They are NOT the v4 training labels (A4 owns
# those); they exist so this tool can quantify separability today.
# ----------------------------------------------------------------------
def _fc_split_label(iou: float, pred_state: int, occ: bool, oov: bool) -> Optional[int]:
    """FC split: 1 = true-FC, 0 = confident-correct/non-FC, None = skip.

    # APPROX: true-FC ~ tracker confidently localised the WRONG place. We proxy
    # "confidently localised" by the runtime CSC having predicted CC/FC (states
    # 0 or 3) — i.e. NOT lost-aware — and judge correctness by GT IoU. Pure
    # occlusion / out-of-view frames are excluded (not an FC). This matches the
    # FCSubtype intent (DISTRACTOR/BACKGROUND are both confident-wrong) without
    # needing candidate identity, which A4 will add.
    """
    if occ or oov:
        return None
    confident = pred_state in (int(0), int(3))  # CC or FC predicted = "confident"
    if not confident:
        return None
    if iou < 0.2:
        return 1  # confident but wrong  -> true-FC
    if iou >= 0.5:
        return 0  # confident and correct -> negative
    return None    # ambiguous mid-IoU -> skip


def _la_split_label(iou: float, pred_state: int, occ: bool, oov: bool) -> Optional[int]:
    """LA split: 1 = true-loss, 0 = false-LA, None = skip.

    # APPROX: among frames the runtime CSC flagged LA (state 2), GT tells us
    # whether the target was actually lost (true-loss, iou<0.2 or occluded/oov)
    # or still well-tracked (false-LA, iou>=0.5 -> CSC over-fired). This is the
    # documented LA-precision wall slice (project_la_gate_is_feasible).
    """
    if pred_state != int(2):  # only frames CSC called LA
        return None
    if occ or oov or iou < 0.2:
        return 1  # truly lost
    if iou >= 0.5:
        return 0  # false alarm (target fine)
    return None


def _proxy_subtype_names() -> tuple[str, str]:
    """Human-readable names tying the proxy splits to the v4 enums."""
    fc = f"true-FC({FCSubtype.DISTRACTOR.name}/{FCSubtype.BACKGROUND.name}) vs confident-correct"
    la = f"true-loss vs {LASubtype.FALSE.name}-LA"
    return fc, la


# ----------------------------------------------------------------------
# Dataset-backed loaders (reuse la_smoke for index/IoU; read states/labels).
# ----------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return out


def _states_by_frame(states_path: Path) -> dict[int, int]:
    """{frame_idx -> derived_state} from a states/<seq>.jsonl file."""
    out: dict[int, int] = {}
    for d in _read_jsonl(states_path):
        t = int(d.get("frame_idx", -1))
        v = d.get("derived_state")
        if t >= 0 and v is not None:
            out[t] = int(v)
    return out


def _telemetry_by_frame(tel_path: Path) -> dict[int, dict]:
    """{frame_idx -> telemetry row} from a telemetry/<seq>.jsonl file."""
    out: dict[int, dict] = {}
    for d in _read_jsonl(tel_path):
        t = d.get("frame_idx")
        if t is not None and not d.get("init", False):
            out[int(t)] = d
    return out


def _labels_by_seq_frame(labels_dir: Path, dataset: str, split: str) -> dict[str, dict[int, dict]]:
    """{seq -> {frame_idx -> GT label row}} from labels_v3 (per-seq preferred)."""
    base = labels_dir / dataset / split
    out: dict[str, dict[int, dict]] = {}
    psd = base / "labels_per_sequence"
    if psd.is_dir():
        for f in sorted(psd.glob("*.jsonl")):
            fr = {int(d["frame_idx"]): d for d in _read_jsonl(f) if "frame_idx" in d}
            out[f.stem] = fr
        return out
    flat = base / "labels.jsonl"  # fallback: one big file keyed by 'sequence'
    for d in _read_jsonl(flat):
        nm = d.get("sequence")
        if nm is None or "frame_idx" not in d:
            continue
        out.setdefault(nm, {})[int(d["frame_idx"])] = d
    return out


# The diagnostic features we score for separability (telemetry field names). We
# include the A1 response features + the appearance/identity + APCE/PSR signals
# that the LA-feasibility note found discriminative.
DIAG_FEATURES: tuple[str, ...] = RESPONSE_FEATURES + (
    "last_cosine_sim",
    "initial_template_sim",
    "appearance_drift",
    "apce",
    "psr",
    "confidence",
)


def collect_diagnostics(passive_dir: Path, dataset: str, split: str) -> dict:
    """Walk a passive run dir + GT and build per-feature score/label columns.

    Returns a dict with, per split ('fc','la'), ``{feature -> (scores, labels)}``
    aligned arrays, plus per-feature raw value columns for calibrator coverage.
    """
    import la_smoke as ls  # local import (heavy: pulls dataset registry)

    states_dir = passive_dir / "states"
    tel_dir = passive_dir / "telemetry"
    labels_dir = passive_dir / "labels_v3"
    idx = ls.build_index(dataset, split=split)
    labels = _labels_by_seq_frame(labels_dir, dataset, split)

    # accumulate columns
    feat_vals: dict[str, list[float]] = {f: [] for f in DIAG_FEATURES}
    fc_scores: dict[str, list[float]] = {f: [] for f in DIAG_FEATURES}
    fc_lab: list[int] = []
    la_scores: dict[str, list[float]] = {f: [] for f in DIAG_FEATURES}
    la_lab: list[int] = []
    n_frames = n_fc_pos = n_la_pos = 0

    for name, seq in idx.items():
        st = _states_by_frame(states_dir / f"{name}.jsonl")
        tel = _telemetry_by_frame(tel_dir / f"{name}.jsonl")
        lab = labels.get(name, {})
        if not st or not lab:
            continue
        for t, pred_state in st.items():
            lrow = lab.get(t)
            trow = tel.get(t)
            if lrow is None or trow is None:
                continue
            iou = lrow.get("iou")
            if iou is None:
                continue
            iou = float(iou)
            aux = lrow.get("aux", {}) or {}
            occ = bool(aux.get("occlusion", False)) or bool(lrow.get("absent", False))
            oov = bool(aux.get("out_of_view", False))
            n_frames += 1
            # collect raw feature values (for coverage) once per frame
            for f in DIAG_FEATURES:
                v = trow.get(f)
                feat_vals[f].append(float(v) if isinstance(v, (int, float)) else float("nan"))
            # FC split
            fc_y = _fc_split_label(iou, pred_state, occ, oov)
            if fc_y is not None:
                fc_lab.append(fc_y)
                n_fc_pos += fc_y
                for f in DIAG_FEATURES:
                    v = trow.get(f)
                    fc_scores[f].append(float(v) if isinstance(v, (int, float)) else float("nan"))
            # LA split
            la_y = _la_split_label(iou, pred_state, occ, oov)
            if la_y is not None:
                la_lab.append(la_y)
                n_la_pos += la_y
                for f in DIAG_FEATURES:
                    v = trow.get(f)
                    la_scores[f].append(float(v) if isinstance(v, (int, float)) else float("nan"))

    return {
        "feat_vals": {f: np.asarray(v, float) for f, v in feat_vals.items()},
        "fc": {f: (np.asarray(fc_scores[f], float), np.asarray(fc_lab)) for f in DIAG_FEATURES},
        "la": {f: (np.asarray(la_scores[f], float), np.asarray(la_lab)) for f in DIAG_FEATURES},
        "counts": {"frames": n_frames, "fc_pos": n_fc_pos, "fc_n": len(fc_lab),
                   "la_pos": n_la_pos, "la_n": len(la_lab)},
    }


def _report_auroc_block(title: str, split_cols: dict, subtype_name: str) -> None:
    rows = []
    for f, (sc, lab) in split_cols.items():
        if lab.size == 0:
            rows.append((float("nan"), f, 1, 0, 0))
            continue
        a, sign = oriented_auroc(sc, lab)
        rows.append((a, f, sign, int(lab.sum()), int(lab.size)))
    # sort by AUROC desc (NaN last)
    rows.sort(key=lambda r: (-(r[0] if np.isfinite(r[0]) else -1.0)))
    print(f"\n=== {title} ===")
    print(f"split positive = {subtype_name}")
    print(f"{'feature':<24}{'AUROC':>8}{'dir':>5}{'n_pos':>7}{'n_tot':>7}")
    for a, f, sign, npos, ntot in rows:
        astr = "  nan" if not np.isfinite(a) else f"{a:6.3f}"
        d = "+" if sign > 0 else "-"
        print(f"{f:<24}{astr:>8}{d:>5}{npos:>7}{ntot:>7}")


def _report_coverage_block(feat_vals: dict) -> None:
    print(f"\n=== A1 calibrator coverage (response features) ===")
    print(f"{'feature':<24}{'n_fin':>7}{'median':>9}{'IQR':>9}{'band%':>7}{'tail%':>7}  flag")
    for f in RESPONSE_FEATURES:
        vals = feat_vals.get(f, np.asarray([], float))
        finite = vals[np.isfinite(vals)]
        try:
            cal = V4FeatureCalibrator(f).fit(finite)
            cov = coverage_stats(finite, cal)
            flag = "DEGENERATE" if cov["degenerate"] else ("low-cover" if cov["band_frac"] < 0.5 else "ok")
            print(f"{f:<24}{cov['n']:>7}{cov['median']:>9.3f}{cov['iqr']:>9.3f}"
                  f"{100*cov['band_frac']:>6.1f}%{100*cov['tail_frac']:>6.1f}%  {flag}")
        except ValueError as exc:
            print(f"{f:<24}{finite.size:>7}{'—':>9}{'—':>9}{'—':>7}{'—':>7}  UNFIT ({str(exc).split(':')[-1].strip()[:30]})")


# ----------------------------------------------------------------------
# Self-test: synthetic numpy arrays, NO dataset. Asserts the AUROC + coverage
# math against known constructions.
# ----------------------------------------------------------------------
def selftest() -> None:
    rng = np.random.default_rng(0)

    # --- rank_auroc exact cases ---
    # perfectly separable, positives strictly above negatives -> AUROC 1.0
    sc = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    lb = np.array([0, 0, 0, 1, 1, 1])
    assert abs(rank_auroc(sc, lb) - 1.0) < 1e-12, rank_auroc(sc, lb)
    # reversed -> 0.0, and oriented flips it to 1.0 with sign -1
    a, sign = oriented_auroc(sc, 1 - lb)
    assert abs(a - 1.0) < 1e-12 and sign == -1, (a, sign)
    # all-tied -> 0.5
    assert abs(rank_auroc(np.zeros(6), lb) - 0.5) < 1e-12
    # single discordant pair: pos={2}, neg={1,3}; scores 1,2,3
    #   pairs (pos>neg): (2>1) yes, (2>3) no  -> AUROC = 1/2
    sc2 = np.array([1.0, 2.0, 3.0]); lb2 = np.array([0, 1, 0])
    assert abs(rank_auroc(sc2, lb2) - 0.5) < 1e-12, rank_auroc(sc2, lb2)
    # tie-correction: pos and neg share a value -> half credit
    #   scores [1,1,2], labels [0,1,0]: pos rank avg=1.5 -> AUROC=(1.5-1)/(1*2)=0.25
    sc3 = np.array([1.0, 1.0, 2.0]); lb3 = np.array([0, 1, 0])
    assert abs(rank_auroc(sc3, lb3) - 0.25) < 1e-12, rank_auroc(sc3, lb3)
    # empty class -> nan
    assert np.isnan(rank_auroc(sc, np.zeros(6, int)))
    # NaN scores dropped: drop the one bad positive, rest perfectly separable
    sc4 = np.array([0.0, 0.1, np.nan, 0.9, 1.0]); lb4 = np.array([0, 0, 1, 1, 1])
    assert abs(rank_auroc(sc4, lb4) - 1.0) < 1e-12, rank_auroc(sc4, lb4)
    # shape-mismatch guard
    try:
        rank_auroc(np.zeros(3), np.zeros(4)); raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # --- a realistic informative feature: pos drawn higher than neg -> AUROC>0.8 ---
    pos = rng.normal(1.0, 1.0, 500)
    neg = rng.normal(-1.0, 1.0, 500)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(500, int), np.zeros(500, int)])
    a = rank_auroc(scores, labels)
    assert 0.8 < a < 0.95, f"separated-normal AUROC out of range: {a}"
    # an uninformative feature -> ~0.5
    a0 = rank_auroc(rng.normal(0, 1, 1000), labels)
    assert 0.42 < a0 < 0.58, f"random feature AUROC not ~0.5: {a0}"

    # --- coverage_stats on a calibrated normal feature ---
    vals = rng.normal(0.0, 1.0, 2000)
    cal = V4FeatureCalibrator("toy").fit(vals)
    cov = coverage_stats(vals, cal)
    assert cov["n"] == 2000 and not cov["degenerate"]
    # most of a normal sample lands in the [0.02,0.98] CDF band on its own fit
    assert cov["band_frac"] > 0.9, f"band_frac too low: {cov['band_frac']}"
    assert abs(cov["band_frac"] + cov["tail_frac"] - 1.0) < 1e-9
    # a near-constant feature -> calibrator fit fails (IQR collapse) => degenerate
    const = np.full(2000, 3.0) + rng.normal(0, 1e-12, 2000)
    cal_c = V4FeatureCalibrator("const").fit(const)
    cov_c = coverage_stats(const, cal_c)
    assert cov_c["degenerate"], "near-constant feature should be flagged degenerate"

    # --- proxy subtype label rules ---
    # confident (CC) + wrong IoU + not occluded -> true-FC (1)
    assert _fc_split_label(0.05, 0, False, False) == 1
    # confident (FC) + correct IoU -> negative (0)
    assert _fc_split_label(0.8, 3, False, False) == 0
    # occluded -> skipped (None) regardless
    assert _fc_split_label(0.05, 0, True, False) is None
    # not confident (LA predicted) -> skipped for FC split
    assert _fc_split_label(0.05, 2, False, False) is None
    # mid IoU -> ambiguous -> None
    assert _fc_split_label(0.35, 0, False, False) is None
    # LA split: CSC said LA + truly lost -> 1
    assert _la_split_label(0.05, 2, False, False) == 1
    # LA split: CSC said LA + target fine -> false-LA (0)
    assert _la_split_label(0.9, 2, False, False) == 0
    # LA split: CSC said LA + occluded -> true-loss (1)
    assert _la_split_label(0.6, 2, True, False) == 1
    # not an LA frame -> None
    assert _la_split_label(0.9, 0, False, False) is None

    # --- end-to-end: synthetic split cols feed the report without a dataset ---
    fc_cols = {
        "good_feat": (scores, labels),     # informative
        "noise_feat": (rng.normal(0, 1, 1000), labels),  # ~chance
    }
    _report_auroc_block("SELFTEST FC split (synthetic)", fc_cols, "synthetic-true-FC")
    _report_coverage_block({f: rng.normal(0, 1, 1000) for f in RESPONSE_FEATURES})

    fc_name, la_name = _proxy_subtype_names()
    assert "FC" in fc_name and "LA" in la_name
    print("\nOK v4_diagnose selftest: rank_auroc + oriented_auroc + coverage_stats + proxy rules verified")


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="CSC-v4 offline separability/calibration diagnostics.")
    ap.add_argument("--passive_dir", default=None,
                    help="passive run dir (with states/, telemetry/, labels_v3/). "
                         "Default: la_smoke.PASSIVE.")
    ap.add_argument("--dataset", default="uav123")
    ap.add_argument("--split", default="test")
    ap.add_argument("--selftest", action="store_true",
                    help="run synthetic-array math self-test (NO dataset) and exit.")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import la_smoke as ls  # noqa: F401  (also resolves PASSIVE default)
    passive = Path(args.passive_dir) if args.passive_dir else Path(ls.PASSIVE)
    print(f"loading {args.dataset}/{args.split} + scoring v4 separability for {passive} ...",
          file=sys.stderr)

    diag = collect_diagnostics(passive, args.dataset, args.split)
    c = diag["counts"]
    fc_name, la_name = _proxy_subtype_names()
    print(f"\n================ v4 diagnose: {passive} ================")
    print(f"frames scored: {c['frames']}   "
          f"FC-split n={c['fc_n']} (pos={c['fc_pos']})   "
          f"LA-split n={c['la_n']} (pos={c['la_pos']})")
    print("NOTE: subtype splits are OFFLINE GT proxies (A4 labeling_v4 not yet wired).")

    _report_auroc_block("FC-subtype separability (rank-AUROC)", diag["fc"], fc_name)
    _report_auroc_block("LA-subtype separability (rank-AUROC)", diag["la"], la_name)
    _report_coverage_block(diag["feat_vals"])
    print("\n(AUROC > ~0.70 = a usable single-feature signal for that split; 'dir' shows "
          "whether higher [+] or lower [-] values indicate the positive subtype.)")


if __name__ == "__main__":
    main()
