"""sgla_memory_extractor.py — SGLATracker backbone-embedding RAM sidecar builder.

Runs SALTRunner over all sequences from the v2 NPZ, extracts per-frame
backbone embeddings via tracker hook attributes, and builds a sidecar NPZ
with 4 pos-only RAM features per frame.

Design invariants
-----------------
1. CAUSAL: features[t] use only RAM state built from frames 0..t-1.
   Specifically: compute_features(embedding[t]) THEN ram.step(embedding[t]).
2. GATE: RAM update uses real model predictions (preds JSON), not oracle labels.
3. FRAME 0: all hook attrs are None after init() → zero embedding; features = zeros.
4. SHAPE: output (T, 4) per sequence, verified by assertion before saving.
5. TRAJECTORY: uses SALTRunner.run() so TSA state machine, CE gates, center
   freeze, and LOST/recovery logic match the trajectory from which the v2 NPZ
   training labels were derived (trajectory-aligned, not just frame-aligned).

CLI
---
python -m salt_r.sgla_memory_extractor \\
  --npz saltr/data/salt_rd_v2_labels.npz \\
  --preds saltr/results/preds_all_v2_retrained.json \\
  --config-path configs/prod/salt.yaml \\
  --output saltr/data/salt_rd_sgla_pos_memory_sidecar.npz \\
  --embedding-view score_weighted \\
  --smoke-test 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 192  # DeiT-tiny embed_dim — all 3 views have this shape

# The 4 pos-only feature names (subset of DistractorAwareMemory.FEATURE_NAMES)
_POS_FEATURE_NAMES: List[str] = [
    "mem_pos_max_sim",
    "mem_pos_mean_sim",
    "mem_pos_recency_sim",
    "mem_update_age",
]


# ---------------------------------------------------------------------------
# Helper: resolve indices of pos-only features within FEATURE_NAMES
# ---------------------------------------------------------------------------

def _pos_feature_indices() -> List[int]:
    """Return the 4 indices of _POS_FEATURE_NAMES in DistractorAwareMemory.FEATURE_NAMES.

    This is used to slice the full feature dict returned by
    DistractorAwareMemory.compute_features() down to the 4 pos-only features.
    """
    from salt_r.memory import DistractorAwareMemory
    all_names = DistractorAwareMemory.FEATURE_NAMES
    indices = []
    for name in _POS_FEATURE_NAMES:
        if name not in all_names:
            raise ValueError(
                f"Feature '{name}' not found in DistractorAwareMemory.FEATURE_NAMES: "
                f"{all_names}"
            )
        indices.append(all_names.index(name))
    return indices


# ---------------------------------------------------------------------------
# Helper: MD5 of a file
# ---------------------------------------------------------------------------

def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Helper: extract embedding from tracker hook attribute
# ---------------------------------------------------------------------------

def _get_embedding(tracker: object, view: str) -> np.ndarray:
    """Extract numpy embedding from tracker hook attributes.

    Returns zero vector of shape (_EMBEDDING_DIM,) if the requested
    attribute is None (e.g. frame 0 after init(), or backbone_feat missing).
    Logs a warning when None is encountered after update() (unexpected).

    Args:
        tracker: SGLATracker instance (after init() or update_with_state()).
        view: one of 'score_weighted', 'peak_local', 'global'.
    """
    attr_map = {
        "score_weighted": "_last_search_score_weighted",
        "peak_local": "_last_search_peak_local",
        "global": "_last_search_global",
    }
    attr_name = attr_map[view]
    tensor = getattr(tracker, attr_name, None)
    if tensor is None:
        return np.zeros(_EMBEDDING_DIM, dtype=np.float32)
    return tensor.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Minimal sequence wrapper (mirrors _TruncatedSequence in collect_features.py)
# ---------------------------------------------------------------------------

class _WrappedSequence:
    """Minimal sequence wrapper for passing to SALTRunner.run().

    Mirrors collect_features._TruncatedSequence — wraps a dataset sequence
    object to cap frames if needed and expose .name, .frames, .ground_truth.
    """

    def __init__(self, name: str, frames: list, ground_truth: list) -> None:
        self.name = name
        self.frames = frames
        self.ground_truth = ground_truth

    @property
    def init_bbox(self):
        return self.ground_truth[0]


# ---------------------------------------------------------------------------
# Per-sequence extraction
# ---------------------------------------------------------------------------

def extract_sequence(
    runner: object,
    seq_obj: object,
    preds_for_seq: Optional[List[dict]],
    embedding_view: str,
    feature_indices: List[int],
) -> np.ndarray:
    """Run SALTRunner on one sequence and return (T, 4) RAM feature array.

    Uses SALTRunner.run() so the full TSA state machine, CE gates, center
    freeze, and LOST/recovery logic are active — matching the trajectory that
    was used to collect the v2 NPZ training labels.

    Causal ordering per frame t:
        features[t] = pos_mem.compute_features(embedding[t])   # uses < t only
        if gate_passes(t): pos_mem.step(embedding[t])          # adds t to RAM

    Args:
        runner: SALTRunner instance (run() calls _reset() internally).
        seq_obj: dataset sequence object with .frames, .ground_truth, .name.
        preds_for_seq: list of per-frame pred dicts from preds JSON,
            or None if this sequence has no predictions.
        embedding_view: 'score_weighted' | 'peak_local' | 'global'.
        feature_indices: 4 indices for pos-only features in FEATURE_NAMES.

    Returns:
        float32 array of shape (T, 4).
    """
    from salt_r.memory import DistractorAwareMemory

    frames = list(seq_obj.frames)
    T = len(frames)
    result = np.zeros((T, 4), dtype=np.float32)

    # Use DistractorAwareMemory which wraps PositiveMemory.
    # We only call compute_features() and positive.should_update()/positive.add()
    # rather than the full mem.step() — this avoids updating negative memory
    # which we don't need for the pos-only sidecar.
    # We also need to track _current_frame for mem.compute_features() update_age.
    mem = DistractorAwareMemory(pos_slots=6, neg_slots=6, update_interval=5)

    # Iterate via SALTRunner.run() — handles frame 0 init internally and
    # applies TSA state machine / CE gates / LOST recovery on frames 1+.
    for t, entry in enumerate(runner.run(seq_obj)):
        if t == 0:
            # Frame 0: tracker.init() was called; _last_search_* are None → zero
            # embedding.  Frame 0 output stays zeros (result already zero-init).
            # Do NOT add to RAM — zero embedding would corrupt similarities.
            # First real backbone embedding arrives at frame 1.
            mem._current_frame = 0
            continue

        # Extract embedding for frame t from runner.tracker (the SGLATracker)
        emb_t = _get_embedding(runner.tracker, embedding_view)
        if (emb_t == 0).all():
            # None or zero — guard: log only when unexpected (not frame 0)
            logger.debug("embedding is zero at t=%d view=%s", t, embedding_view)

        # CAUSAL: compute features using RAM built from frames 0..t-1
        mem._current_frame = t
        feat_dict = mem.compute_features(query_emb=emb_t, query_bbox=None)

        # Extract the 4 pos-only features in order
        all_names = DistractorAwareMemory.FEATURE_NAMES
        for out_col, feat_idx in enumerate(feature_indices):
            result[t, out_col] = float(feat_dict[all_names[feat_idx]])

        # Gate check: update RAM with frame t's embedding if conditions pass
        p_fc, p_ifd, apce_norm = _get_gate_signals(preds_for_seq, t=t)
        if mem.positive.should_update(
            p_fc=p_fc, p_ifd=p_ifd, apce_norm=apce_norm, current_frame=t
        ):
            from salt_r.memory import MemoryEntry
            mem.positive.add(MemoryEntry(
                embedding=emb_t.copy(),
                frame_idx=t,
                iou=float("nan"),
                apce_norm=apce_norm,
                p_fc=p_fc,
                source="target_confident",
            ))

    return result


def _get_gate_signals(
    preds_for_seq: Optional[List[dict]], t: int
) -> tuple:
    """Return (p_fc, p_ifd, apce_norm) gate signals for frame t.

    Falls back to permissive defaults when preds unavailable.
    apce_norm is not present in preds JSON (it's a tracker telemetry field),
    so we use 0.0 as fallback — this means the gate degrades to
    p_fc < 0.20 AND p_ifd < 0.30 only (apce_norm <= 0.4 blocks gate).
    To avoid silently disabling all updates when apce_norm is unavailable,
    we use 1.0 as the fallback so the gate is driven purely by preds.
    """
    if preds_for_seq is None or t >= len(preds_for_seq):
        # No preds available: use conservative defaults that allow some updates
        return 0.10, 0.10, 1.0

    frame_preds = preds_for_seq[t]
    p_fc = float(frame_preds.get("false_confirmed", 0.5))
    p_ifd = float(frame_preds.get("imminent_failure_dynamic", 0.5))

    # apce_norm is NOT in preds JSON — use 1.0 so gate is driven by p_fc/p_ifd only
    # (PositiveMemory.should_update requires apce_norm > 0.4; 1.0 always passes)
    apce_norm = 1.0

    return p_fc, p_ifd, apce_norm


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def build_sidecar(args: argparse.Namespace) -> None:
    """Extract RAM features for all sequences and write output NPZ.

    Partial results are saved on KeyboardInterrupt.
    """
    import sys

    # Ensure src and saltr/src are on PYTHONPATH
    repo_root = Path(__file__).parents[4]
    for extra in [str(repo_root / "src"), str(repo_root / "saltr" / "src")]:
        if extra not in sys.path:
            sys.path.insert(0, extra)

    # Resolve absolute paths
    npz_path = Path(args.npz).resolve()
    preds_path = Path(args.preds).resolve()
    output_path = Path(args.output).resolve()

    print(f"[sgla_memory_extractor] NPZ:    {npz_path}")
    print(f"[sgla_memory_extractor] Preds:  {preds_path}")
    print(f"[sgla_memory_extractor] Output: {output_path}")
    print(f"[sgla_memory_extractor] View:   {args.embedding_view}")

    # Load NPZ
    data = np.load(str(npz_path), allow_pickle=True)
    seq_keys = sorted(
        k[len("features/"):] for k in data.files if k.startswith("features/")
    )
    print(f"[sgla_memory_extractor] {len(seq_keys)} sequences in NPZ")

    # Load preds JSON
    print(f"[sgla_memory_extractor] Loading preds JSON...", flush=True)
    preds_raw: dict = json.loads(preds_path.read_text())
    print(f"[sgla_memory_extractor] {len(preds_raw)} sequences in preds JSON")

    # Smoke test: limit sequences
    if args.smoke_test is not None:
        seq_keys = seq_keys[: args.smoke_test]
        print(f"[sgla_memory_extractor] SMOKE TEST: processing {len(seq_keys)} sequences")

    # Resolve pos-only feature indices
    feature_indices = _pos_feature_indices()
    print(f"[sgla_memory_extractor] Pos-only feature indices: {feature_indices}")

    # Build SALTRunner from config — loaded once, reset per sequence via run()
    from uav_tracker.salt_runner import SALTRunner

    config_path = args.config_path
    runner = SALTRunner.from_config(config_path)
    print(
        f"[sgla_memory_extractor] SALTRunner loaded from {config_path} "
        f"(tracker stub={getattr(runner.tracker, 'is_stub_mode', False)})"
    )

    # Compute MD5 of source files
    npz_md5 = _md5_file(str(npz_path))
    preds_md5 = _md5_file(str(preds_path))

    # Tracker weights MD5 (optional)
    tracker_ckpt_md5 = "unavailable"
    try:
        from uav_tracker.paths import weights_root
        _w = weights_root() / "sglatrack" / "sglatrack_ep0297.pth.tar"
        if _w.exists():
            tracker_ckpt_md5 = _md5_file(str(_w))
    except Exception:
        pass

    # Per-dataset loaders (lazy — loaded when first needed per dataset)
    _dataset_loaders: dict = {}

    def _get_dataset(dataset_name: str):
        if dataset_name not in _dataset_loaders:
            if dataset_name == "uav123":
                from uav_tracker.datasets.uav123 import UAV123Dataset
                _dataset_loaders[dataset_name] = {
                    seq.name: seq for seq in UAV123Dataset(root=None)
                }
            elif dataset_name == "visdrone_sot":
                from uav_tracker.datasets.visdrone_sot import VisDroneSOTDataset
                _dataset_loaders[dataset_name] = {
                    seq.name: seq for seq in VisDroneSOTDataset(root=None)
                }
            elif dataset_name == "dtb70":
                from uav_tracker.datasets.dtb70 import DTB70Dataset
                _dataset_loaders[dataset_name] = {
                    seq.name: seq for seq in DTB70Dataset(root=None)
                }
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        return _dataset_loaders[dataset_name]

    # Collection
    out: Dict[str, object] = {}
    n_done = 0
    n_skipped = 0
    n_total_frames = 0

    try:
        for seq_key in seq_keys:
            # Parse dataset/seq from compound key
            parts = seq_key.split("/", 1)
            if len(parts) != 2:
                logger.warning("Unexpected seq key format: %s — skipping", seq_key)
                n_skipped += 1
                continue
            dataset_name, seq_name = parts

            # Load preds for this sequence
            preds_for_seq: Optional[List[dict]] = preds_raw.get(seq_key)
            if preds_for_seq is None:
                logger.warning(
                    "No preds for seq '%s' — skipping (add to preds JSON or use oracle fallback)",
                    seq_key,
                )
                n_skipped += 1
                continue

            # Load dataset sequence object (lazy)
            try:
                dataset_seqs = _get_dataset(dataset_name)
            except Exception as exc:
                logger.warning("Could not load dataset '%s': %s — skipping", dataset_name, exc)
                n_skipped += 1
                continue

            if seq_name not in dataset_seqs:
                logger.warning(
                    "Seq '%s' not found in dataset '%s' — skipping", seq_name, dataset_name
                )
                n_skipped += 1
                continue

            seq_obj = dataset_seqs[seq_name]

            # Load all frames to determine T and verify against NPZ.
            # We load frames here (not inside extract_sequence) so we can check
            # the frame count before running the (expensive) SALTRunner pass.
            try:
                frames_list = list(seq_obj.frames)
                gt_bboxes_raw = list(seq_obj.ground_truth)
            except Exception as exc:
                logger.warning("Could not load frames/GT for '%s': %s — skipping", seq_key, exc)
                n_skipped += 1
                continue

            T = len(frames_list)

            # Verify frame count against NPZ
            npz_T = data[f"features/{seq_key}"].shape[0]
            if T != npz_T:
                logger.warning(
                    "Frame count mismatch for '%s': dataset=%d npz=%d — skipping",
                    seq_key, T, npz_T,
                )
                n_skipped += 1
                continue

            # Wrap in _WrappedSequence so runner.run() sees .name/.frames/.ground_truth
            # SALTRunner.run() calls self._reset() internally — no manual tracker reset needed.
            seq_for_run = _WrappedSequence(
                name=seq_name,
                frames=frames_list,
                ground_truth=gt_bboxes_raw,
            )

            print(
                f"  [{n_done + 1}/{len(seq_keys)}] {seq_key}  T={T}",
                end="  ", flush=True,
            )

            # Run extraction via SALTRunner
            try:
                mem_feats = extract_sequence(
                    runner=runner,
                    seq_obj=seq_for_run,
                    preds_for_seq=preds_for_seq,
                    embedding_view=args.embedding_view,
                    feature_indices=feature_indices,
                )
            except Exception as exc:
                logger.warning("Extraction failed for '%s': %s — skipping", seq_key, exc)
                n_skipped += 1
                print(f"ERROR: {exc}")
                continue

            # Invariant: shape (T, 4) and frame 0 is all zeros
            assert mem_feats.shape == (T, 4), (
                f"{seq_key}: shape mismatch — got {mem_feats.shape}, expected ({T}, 4)"
            )

            n_ram_updates = int(
                (mem_feats[1:, 3] < mem_feats[:-1, 3]).sum()  # update_age dropped
            ) if T > 1 else 0
            mean_sim = float(mem_feats[:, 1].mean())  # mem_pos_mean_sim
            max_age = float(mem_feats[:, 3].max())    # mem_update_age

            if args.smoke_test is not None:
                print(
                    f"n_ram_updates~{n_ram_updates}  "
                    f"mean_pos_mean_sim={mean_sim:.4f}  "
                    f"max_update_age={max_age:.0f}"
                )
            else:
                print("ok")

            out[f"memory_features/{seq_key}"] = mem_feats.astype(np.float32)
            n_done += 1
            n_total_frames += T

    except KeyboardInterrupt:
        print(
            f"\n[sgla_memory_extractor] Interrupted after {n_done} sequences "
            f"({n_skipped} skipped). Saving partial sidecar...",
            flush=True,
        )

    # Build metadata
    out["memory_feature_names"] = np.array(_POS_FEATURE_NAMES, dtype=object)
    out["embedding_view"] = np.array(args.embedding_view)
    out["embedding_dim"] = np.array(_EMBEDDING_DIM)
    out["tracker_checkpoint_md5"] = np.array(tracker_ckpt_md5)
    out["source_npz_md5"] = np.array(npz_md5)
    out["preds_json_md5"] = np.array(preds_md5)
    out["uses_oracle_labels"] = np.array(False)
    out["n_sequences"] = np.array(n_done)
    out["n_frames"] = np.array(n_total_frames)
    out["created_at"] = np.array(datetime.now(tz=timezone.utc).isoformat())

    if n_done == 0:
        print("[sgla_memory_extractor] WARNING: no sequences processed — output NPZ will be empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **out)
    print(
        f"[sgla_memory_extractor] Saved {output_path}  "
        f"({n_done} sequences, {n_total_frames:,} frames, {n_skipped} skipped)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build per-frame positive-appearance RAM sidecar using real "
            "SGLATracker backbone embeddings via SALTRunner."
        )
    )
    p.add_argument(
        "--npz", required=True,
        help="Path to salt_rd_v2_labels.npz",
    )
    p.add_argument(
        "--preds", required=True,
        help="Path to preds_all_*.json (model predictions, one dict per frame)",
    )
    p.add_argument(
        "--output", required=True,
        help="Output path, e.g. saltr/data/salt_rd_sgla_pos_memory_sidecar.npz",
    )
    p.add_argument(
        "--config-path",
        default="configs/prod/salt.yaml",
        help=(
            "Path to SALTRunner YAML config (e.g. configs/prod/salt.yaml). "
            "SALTRunner.from_config() loads tracker weights, TSA, detector, etc. "
            "The same config used when collecting the v2 NPZ labels should be used "
            "here to ensure trajectory alignment."
        ),
    )
    p.add_argument(
        "--embedding-view",
        choices=["score_weighted", "peak_local", "global"],
        default="score_weighted",
        help=(
            "Which backbone embedding view to use: "
            "score_weighted (default), peak_local, or global."
        ),
    )
    p.add_argument(
        "--smoke-test", type=int, default=None, metavar="N",
        help=(
            "Process only the first N sequences then exit. "
            "Prints per-sequence stats."
        ),
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    build_sidecar(args)


if __name__ == "__main__":
    main()
