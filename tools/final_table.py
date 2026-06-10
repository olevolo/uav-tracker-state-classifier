#!/usr/bin/env python3
"""Final §4 comparison table: passive vs combo_mb (motion_bridge+FC+risk_gate)
across ALL/EASY/MID/HARD-22 for all 3 trackers on UAV123.

Also shows best LA/FC standalone for HARD-22.
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    sys.path.insert(0, str(_p))
sys.argv = [sys.argv[0]]
import csc_uav_tracking  # noqa
import la_smoke as ls

EXCL = {'bird1_1'}

# ── paths ────────────────────────────────────────────────────────────────────
PASSIVE = {
    'sgla': ROOT / "outputs/eval5_clamp/csc/sglatrack/uav123/test/passive",
    'av':   ROOT / "outputs/baselines/avtrack/uav123/test",
    'or':   ROOT / "outputs/baselines/ortrack/uav123/test",
}
# combo_mb = motion_bridge + FC hold_lastgood + control_risk_gate
COMBO_MB = {
    'sgla': ROOT / "outputs/eval7_gated/csc/sglatrack/uav123/test/full_combo",
    'av':   ROOT / "outputs/eval13_avtrack/csc/avtrack/uav123/test/combo_mb",
    'or':   ROOT / "outputs/eval13_ortrack/csc/ortrack/uav123/test/combo_mb",
}
# best standalone LA (sgla_redetect HARD, fc_only AV HARD)
BEST_LA = {
    'sgla': ROOT / "outputs/eval10_sgla/csc/sglatrack/uav123/test/la_only",
    'av':   ROOT / "outputs/eval10_avtrack/csc/avtrack/uav123/test/fc_only",   # fc wins for AV
    'or':   ROOT / "outputs/eval10_ortrack/csc/ortrack/uav123/test/la_only",
}

# ── helpers ──────────────────────────────────────────────────────────────────
def build_gf():
    idx = ls.build_index("uav123")
    gf = {}
    for n, seq in idx.items():
        ious, _ = ls.seq_iou(seq, PASSIVE['sgla'] / "predictions")
        if ious is not None:
            fin = np.isfinite(ious)
            gf[n] = float((ious[fin] < 0.2).mean())
    return gf

def seg_mean(seqs, idx, d: pathlib.Path):
    preds = d / "predictions"
    if not preds.exists():
        return float("nan"), 0
    vals = []
    for n in seqs:
        seq = idx.get(n)
        if not seq:
            continue
        ious, _ = ls.seq_iou(seq, preds)
        if ious is not None:
            fin = np.isfinite(ious)
            if fin.any():
                vals.append(float(ious[fin].mean()))
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)

def get_fcr(d: pathlib.Path, seqs: set):
    sd = d / "states"
    if not sd.exists():
        return float("nan")
    tf = fc = 0
    for f in sd.glob("*.jsonl"):
        if f.stem not in seqs:
            continue
        try:
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            rows = [r for r in rows if not r.get("init")]
            tf += len(rows)
            fc += sum(1 for r in rows if r.get("false_confirmed_flag") or r.get("derived_state") == 3)
        except Exception:
            pass
    return fc / tf * 100 if tf else float("nan")

def fmt(v): return f"{v:>7.4f}" if v == v else "   --- "
def fmd(v, b): return f"{v-b:>+8.4f}" if v == v and b == b else "    --- "
def fmf(v): return f"{v:>5.2f}%" if v == v else "  --- "


def main():
    idx = ls.build_index("uav123")
    gf  = build_gf()

    B = {
        "ALL":     set(idx.keys()),
        "EASY":    {n for n, g in gf.items() if g < 0.10},
        "MID":     {n for n, g in gf.items() if 0.10 <= g < 0.30},
        "HARD-22": {n for n, g in gf.items() if g >= 0.30 and n not in EXCL},
    }

    print("=" * 100)
    print("  UAV123 — §4 control table  (best per-tracker: SGLA=combined+mb, AV=csc_head+mb, OR=csc_head+mb)")
    print(f"  {'':27}  {'pass':>7}  {'combo_mb':>9}{'Δ':>9}  {'best_la/fc':>9}{'Δ':>9}  FCR_p  FCR_mb  n")
    print("  " + "-" * 97)

    for trk, name in [("sgla", "SGLATrack"), ("av", "AVTrack"), ("or", "ORTrack")]:
        p_dir = PASSIVE[trk]
        m_dir = COMBO_MB[trk]
        l_dir = BEST_LA[trk]
        print()
        for bk_name, seqs in B.items():
            if trk == "or" and bk_name in ("ALL", "EASY", "MID"):
                if not m_dir.exists():
                    print(f"  {name:<10} {bk_name:<16}  (waiting for OR combo_mb full run)")
                    continue
            b, nb  = seg_mean(seqs, idx, p_dir)
            m, nm  = seg_mean(seqs, idx, m_dir)
            la, nl = seg_mean(seqs, idx, l_dir)
            fp  = fmf(get_fcr(p_dir, seqs))
            fm  = fmf(get_fcr(m_dir, seqs))
            label = f"{name:<10} {bk_name:<16}"
            print(f"  {label}  {fmt(b)}  {fmt(m)}{fmd(m,b)}  {fmt(la)}{fmd(la,b)}  {fp}  {fm}  [{nb}]")

    print("\n  Notes:")
    print("  combo_mb = motion_bridge + FC hold_lastgood + --control_risk_gate")
    print("  best_la/fc: SGLA=sgla_redetect(eval10), AV=fc_only(eval10), OR=la_only/csc_head(eval10)")


if __name__ == "__main__":
    main()
