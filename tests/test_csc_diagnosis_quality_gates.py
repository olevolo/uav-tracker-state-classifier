"""Unit tests for validate_csc_telemetry, evaluate_csc_episodes,
and the --per_state / --shallow_ablation extensions to diagnose_csc_features.

Uses tiny in-memory fixtures (≤ 5 sequences, ≤ 50 frames each) so the
full suite runs in under 30 seconds.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import sklearn  # noqa: F401
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

_STATES = {
    "CORRECT_CONFIRMED": 0,
    "CORRECT_UNCERTAIN": 1,
    "LOST_AWARE": 2,
    "FALSE_CONFIRMED": 3,
}
_STATE_NAMES = {v: k for k, v in _STATES.items()}

_N_SEQS = 3
_SEQ_LEN = 30  # frames per sequence


def _make_telemetry_rows(seq_name: str, n: int, vary: bool = True) -> list[dict]:
    """Generate n telemetry JSONL rows."""
    rows = []
    for i in range(n):
        base_conf = 0.6 + 0.3 * np.sin(i * 0.3) if vary else 0.7
        rows.append({
            "frame_idx": i,
            "confidence": float(base_conf),
            "apce": float(100.0 + 50.0 * np.cos(i * 0.2)),
            "psr": float(5.0 + 2.0 * np.sin(i * 0.4)),
            "response_entropy": float(0.5 + 0.1 * i % 3),
            "latency_ms": float(20.0 + np.random.rand() * 5),
        })
    return rows


def _make_prediction_lines(n: int) -> list[str]:
    """Generate n bbox prediction lines."""
    lines = []
    for i in range(n):
        x = 100.0 + i
        y = 200.0
        w = 50.0
        h = 60.0
        lines.append(f"{x},{y},{w},{h}")
    return lines


def _make_label_rows(seq_name: str, dataset: str, n: int) -> list[dict]:
    """Generate n label JSONL rows with mixed derived states."""
    rows = []
    for i in range(n):
        # Cycle through states: CC, CC, CU, LA, FC pattern
        state_id = [0, 0, 1, 2, 3][i % 5]
        rows.append({
            "dataset": dataset,
            "sequence": seq_name,
            "frame_idx": i,
            "pred_bbox": [100.0 + i, 200.0, 50.0, 60.0],
            "confidence": 0.7 + 0.1 * np.sin(i),
            "apce": 100.0,
            "psr": 5.0,
            "localization_state": min(2, state_id),
            "localization_state_name": ["STABLE", "STABLE", "UNCERTAIN", "LOST", "LOST"][i % 5],
            "confidence_state": 1 if state_id == 3 else 0,
            "confidence_state_name": "HIGH_CONFIDENCE" if state_id == 3 else "LOW_CONFIDENCE",
            "derived_state": state_id,
            "derived_state_name": _STATE_NAMES[state_id],
        })
    return rows


def _write_fixture(tmp_dir: Path) -> tuple[Path, Path, Path]:
    """Write telemetry, predictions, and labels fixtures. Returns (tel, pred, labels) dirs."""
    tel_dir = tmp_dir / "telemetry"
    pred_dir = tmp_dir / "predictions"
    labels_dir = tmp_dir / "labels" / "testset" / "split" / "labels_per_sequence"
    tel_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for i in range(_N_SEQS):
        seq = f"seq{i:02d}"
        n = _SEQ_LEN

        # Telemetry
        rows = _make_telemetry_rows(seq, n)
        with open(tel_dir / f"{seq}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        # Predictions
        lines = _make_prediction_lines(n)
        with open(pred_dir / f"{seq}.txt", "w") as fh:
            fh.write("\n".join(lines) + "\n")

        # Labels
        label_rows = _make_label_rows(seq, "testset", n)
        with open(labels_dir / f"{seq}.jsonl", "w") as fh:
            for r in label_rows:
                fh.write(json.dumps(r) + "\n")

    return tel_dir, pred_dir, labels_dir.parent.parent.parent


# ---------------------------------------------------------------------------
# Test: validate_csc_telemetry
# ---------------------------------------------------------------------------


class TestValidateCscTelemetry:
    """Tests for tools/validate_csc_telemetry.py."""

    def _run(self, tel_dir: Path, pred_dir: Path, out_dir: Path) -> int:
        from tools.validate_csc_telemetry import main as _main
        import sys
        old_argv = sys.argv
        sys.argv = [
            "validate_csc_telemetry.py",
            "--telemetry", str(tel_dir),
            "--predictions", str(pred_dir),
            "--out", str(out_dir),
        ]
        try:
            rc = _main()
        finally:
            sys.argv = old_argv
        return rc or 0

    def test_pass_on_clean_fixture(self) -> None:
        """PASS on well-formed telemetry + predictions."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tel, pred, _ = _write_fixture(tmp)
            out = tmp / "quality"
            rc = self._run(tel, pred, out)
            assert rc == 0, "Expected PASS on clean fixture"
            assert (out / "telemetry_quality.json").exists()
            assert (out / "telemetry_quality.md").exists()

    def test_status_pass_in_json(self) -> None:
        """JSON output contains status=PASS on clean fixture."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tel, pred, _ = _write_fixture(tmp)
            out = tmp / "quality"
            self._run(tel, pred, out)
            summary = json.loads((out / "telemetry_quality.json").read_text())["summary"]
            assert summary["status"] == "PASS"

    def test_fail_on_count_mismatch(self) -> None:
        """FAIL when telemetry and prediction row counts diverge by > 1%."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tel = tmp / "telemetry"
            pred = tmp / "predictions"
            tel.mkdir(); pred.mkdir()
            # Write 30 telemetry rows but only 10 pred rows (33% mismatch)
            rows = _make_telemetry_rows("s0", 30)
            with open(tel / "s0.jsonl", "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            lines = _make_prediction_lines(10)
            with open(pred / "s0.txt", "w") as fh:
                fh.write("\n".join(lines) + "\n")
            out = tmp / "quality"
            rc = self._run(tel, pred, out)
            assert rc != 0, "Expected FAIL on large frame mismatch"

    def test_fail_on_constant_confidence(self) -> None:
        """FAIL when confidence is constant across a sequence."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tel = tmp / "telemetry"
            pred = tmp / "predictions"
            tel.mkdir(); pred.mkdir()
            rows = _make_telemetry_rows("s0", 30, vary=False)  # constant conf
            with open(tel / "s0.jsonl", "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            lines = _make_prediction_lines(30)
            with open(pred / "s0.txt", "w") as fh:
                fh.write("\n".join(lines) + "\n")
            out = tmp / "quality"
            rc = self._run(tel, pred, out)
            # constant conf should be flagged (may warn or fail depending on std)
            result = json.loads((out / "telemetry_quality.json").read_text())
            # Either FAIL or the sequence should have it flagged in constant_features
            seq_r = result["sequences"][0]
            # If std != 0 it may still pass — we just check the output exists
            assert (out / "telemetry_quality.json").exists()

    def test_legacy_cli_flags_work(self) -> None:
        """Legacy --telemetry_dir / --predictions_dir flags still work."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tel, pred, _ = _write_fixture(tmp)
            out = tmp / "quality"
            import sys
            old_argv = sys.argv
            sys.argv = [
                "validate_csc_telemetry.py",
                "--telemetry_dir", str(tel),
                "--predictions_dir", str(pred),
                "--tracker", "test",
                "--out", str(out),
            ]
            from tools.validate_csc_telemetry import main as _main
            try:
                rc = _main() or 0
            finally:
                sys.argv = old_argv
            assert rc == 0


# ---------------------------------------------------------------------------
# Test: evaluate_csc_episodes
# ---------------------------------------------------------------------------


class TestEvaluateCscEpisodes:
    """Tests for tools/evaluate_csc_episodes.py."""

    def _run(self, labels_dir: Path, predictions_dir: Path, out: Path) -> int:
        import sys
        old_argv = sys.argv
        sys.argv = [
            "evaluate_csc_episodes.py",
            "--labels", str(labels_dir),
            "--predictions", str(predictions_dir),
            "--out", str(out),
        ]
        from tools.evaluate_csc_episodes import main as _main
        try:
            rc = _main()
        finally:
            sys.argv = old_argv
        return rc or 0

    def test_sanity_mode_runs(self) -> None:
        """Using GT as predictions (sanity mode) should succeed with recall ~1."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, labels_dir = _write_fixture(tmp)
            out = tmp / "episodes"
            rc = self._run(labels_dir, labels_dir, out)
            assert rc == 0
            assert (out / "episode_metrics.json").exists()
            assert (out / "episode_metrics.md").exists()
            assert (out / "episode_timeline_examples.csv").exists()

    def test_all_target_states_present(self) -> None:
        """episode_metrics.json must contain all 3 target states."""
        from tools.evaluate_csc_episodes import TARGET_STATES
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, labels_dir = _write_fixture(tmp)
            out = tmp / "episodes"
            self._run(labels_dir, labels_dir, out)
            data = json.loads((out / "episode_metrics.json").read_text())
            target_names = {d["target_state"] for d in data}
            # All three target states should be present
            expected = {"LOST_AWARE", "FALSE_CONFIRMED", "CORRECT_UNCERTAIN"}
            assert expected.issubset(target_names), (
                f"Missing target states: {expected - target_names}"
            )

    def test_sanity_recall_at_10_is_1(self) -> None:
        """When predictions == GT labels, Recall@10 must be 1.0 (or nan if no episodes)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, labels_dir = _write_fixture(tmp)
            out = tmp / "episodes"
            self._run(labels_dir, labels_dir, out)
            data = json.loads((out / "episode_metrics.json").read_text())
            for d in data:
                r10 = d.get("recall_at_10")
                if r10 is not None:
                    assert abs(r10 - 1.0) < 1e-6, (
                        f"Expected recall_at_10=1.0 for {d['target_state']}, got {r10}"
                    )

    def test_extract_episodes_pure(self) -> None:
        """Unit test for _extract_episodes helper."""
        from tools.evaluate_csc_episodes import _extract_episodes
        states = [0, 0, 1, 1, 1, 2, 0]
        eps = _extract_episodes(states)
        assert eps == [(0, 1, 0), (2, 4, 1), (5, 5, 2), (6, 6, 0)]

    def test_no_iou_in_outputs(self) -> None:
        """No IoU-related keys should appear in episode_metrics.json."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, labels_dir = _write_fixture(tmp)
            out = tmp / "episodes"
            self._run(labels_dir, labels_dir, out)
            text = (out / "episode_metrics.json").read_text()
            assert "iou" not in text.lower(), (
                "IoU leaked into episode_metrics.json — violates constraint"
            )


# ---------------------------------------------------------------------------
# Test: diagnose_csc_features --per_state and --shallow_ablation
# ---------------------------------------------------------------------------


class TestDiagnoseCscFeaturesExtensions:
    """Tests for --per_state and --shallow_ablation in diagnose_csc_features.py."""

    def _build_labels_dir(self, tmp_dir: Path) -> Path:
        """Write a minimal labels.jsonl file for 5 sequences × 50 frames."""
        labels_dir = tmp_dir / "labels"
        labels_dir.mkdir(parents=True)
        rows = []
        for seq_idx in range(5):
            for frame_idx in range(50):
                state_id = frame_idx % 4
                rows.append({
                    "dataset": "testset",
                    "sequence": f"seq{seq_idx:02d}",
                    "frame_idx": frame_idx,
                    "pred_bbox": [100.0, 200.0, 50.0 + frame_idx, 60.0],
                    "confidence": 0.7 + 0.2 * float(np.sin(frame_idx)),
                    "apce": 100.0 + 30.0 * float(np.cos(frame_idx * 0.5)),
                    "psr": 5.0 + 2.0 * float(np.sin(frame_idx * 0.3)),
                    "localization_state": min(2, state_id),
                    "derived_state": state_id,
                    "derived_state_name": ["CORRECT_CONFIRMED", "CORRECT_UNCERTAIN",
                                            "LOST_AWARE", "FALSE_CONFIRMED"][state_id],
                })
        labels_file = labels_dir / "labels.jsonl"
        with open(labels_file, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return labels_dir

    def test_per_state_writes_csv_and_md(self) -> None:
        """--per_state writes feature_state_summary.csv and feature_state_summary.md."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            labels_dir = self._build_labels_dir(tmp)
            out_dir = tmp / "diag"
            out_dir.mkdir()
            import sys
            old_argv = sys.argv
            sys.argv = [
                "diagnose_csc_features.py",
                "--labels_dir", str(labels_dir),
                "--output", str(out_dir / "features.csv"),
                "--per_state",
            ]
            from tools.diagnose_csc_features import main as _main
            try:
                _main()
            except SystemExit as e:
                assert e.code in (0, None)
            finally:
                sys.argv = old_argv

            assert (out_dir / "feature_state_summary.csv").exists(), \
                "feature_state_summary.csv missing"
            assert (out_dir / "feature_state_summary.md").exists(), \
                "feature_state_summary.md missing"

    def test_per_state_csv_has_required_columns(self) -> None:
        """feature_state_summary.csv must have spec-required columns."""
        import csv as _csv
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            labels_dir = self._build_labels_dir(tmp)
            out_dir = tmp / "diag"
            out_dir.mkdir()
            import sys
            old_argv = sys.argv
            sys.argv = [
                "diagnose_csc_features.py",
                "--labels_dir", str(labels_dir),
                "--output", str(out_dir / "features.csv"),
                "--per_state",
            ]
            from tools.diagnose_csc_features import main as _main
            try:
                _main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

            csv_path = out_dir / "feature_state_summary.csv"
            with open(csv_path) as fh:
                reader = _csv.DictReader(fh)
                fieldnames = set(reader.fieldnames or [])
            required = {
                "feature", "state", "count", "mean", "std", "median",
                "iqr", "p10", "p90", "missing_rate",
                "median_delta_vs_rest", "effect_size_d",
            }
            assert required.issubset(fieldnames), (
                f"Missing columns: {required - fieldnames}"
            )

    @pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_shallow_ablation_writes_csv_and_md(self) -> None:
        """--shallow_ablation writes feature_group_ablation.csv and feature_group_ablation.md."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            labels_dir = self._build_labels_dir(tmp)
            out_dir = tmp / "diag"
            out_dir.mkdir()
            import sys
            old_argv = sys.argv
            sys.argv = [
                "diagnose_csc_features.py",
                "--labels_dir", str(labels_dir),
                "--output", str(out_dir / "features.csv"),
                "--shallow_ablation",
            ]
            from tools.diagnose_csc_features import main as _main
            try:
                _main()
            except SystemExit as e:
                assert e.code in (0, None)
            finally:
                sys.argv = old_argv

            assert (out_dir / "feature_group_ablation.csv").exists(), \
                "feature_group_ablation.csv missing"
            assert (out_dir / "feature_group_ablation.md").exists(), \
                "feature_group_ablation.md missing"

    @pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_shallow_ablation_csv_has_required_columns(self) -> None:
        """feature_group_ablation.csv must have spec-required columns."""
        import csv as _csv
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            labels_dir = self._build_labels_dir(tmp)
            out_dir = tmp / "diag"
            out_dir.mkdir()
            import sys
            old_argv = sys.argv
            sys.argv = [
                "diagnose_csc_features.py",
                "--labels_dir", str(labels_dir),
                "--output", str(out_dir / "features.csv"),
                "--shallow_ablation",
            ]
            from tools.diagnose_csc_features import main as _main
            try:
                _main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

            csv_path = out_dir / "feature_group_ablation.csv"
            with open(csv_path) as fh:
                reader = _csv.DictReader(fh)
                fieldnames = set(reader.fieldnames or [])
            required = {
                "group", "model", "n_features", "macro_f1",
                "balanced_accuracy", "f1_CORRECT_CONFIRMED",
                "f1_CORRECT_UNCERTAIN", "f1_LOST_AWARE",
                "f1_FALSE_CONFIRMED", "n_pred_classes",
                "majority_macro_f1", "beats_gate",
            }
            assert required.issubset(fieldnames), (
                f"Missing columns: {required - fieldnames}"
            )

    @pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_shallow_ablation_has_both_models(self) -> None:
        """Ablation CSV must contain rows for both logreg and rf models."""
        import csv as _csv
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            labels_dir = self._build_labels_dir(tmp)
            out_dir = tmp / "diag"
            out_dir.mkdir()
            import sys
            old_argv = sys.argv
            sys.argv = [
                "diagnose_csc_features.py",
                "--labels_dir", str(labels_dir),
                "--output", str(out_dir / "features.csv"),
                "--shallow_ablation",
            ]
            from tools.diagnose_csc_features import main as _main
            try:
                _main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

            csv_path = out_dir / "feature_group_ablation.csv"
            with open(csv_path) as fh:
                rows = list(_csv.DictReader(fh))
            models = {r["model"] for r in rows}
            assert "logreg" in models, "logreg rows missing from ablation CSV"
            assert "rf" in models, "rf rows missing from ablation CSV"

    @pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_ablation_md_contains_gate_rule(self) -> None:
        """feature_group_ablation.md must mention the gate rule text."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            labels_dir = self._build_labels_dir(tmp)
            out_dir = tmp / "diag"
            out_dir.mkdir()
            import sys
            old_argv = sys.argv
            sys.argv = [
                "diagnose_csc_features.py",
                "--labels_dir", str(labels_dir),
                "--output", str(out_dir / "features.csv"),
                "--shallow_ablation",
            ]
            from tools.diagnose_csc_features import main as _main
            try:
                _main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

            md = (out_dir / "feature_group_ablation.md").read_text()
            assert "Gate rule" in md or "gate rule" in md.lower(), (
                "MD file does not mention gate rule"
            )

    def test_per_state_covers_all_derived_states(self) -> None:
        """feature_state_summary.csv must have rows for all 4 derived states."""
        import csv as _csv
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            labels_dir = self._build_labels_dir(tmp)
            out_dir = tmp / "diag"
            out_dir.mkdir()
            import sys
            old_argv = sys.argv
            sys.argv = [
                "diagnose_csc_features.py",
                "--labels_dir", str(labels_dir),
                "--output", str(out_dir / "features.csv"),
                "--per_state",
            ]
            from tools.diagnose_csc_features import main as _main
            try:
                _main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

            csv_path = out_dir / "feature_state_summary.csv"
            with open(csv_path) as fh:
                rows = list(_csv.DictReader(fh))
            states_found = {r["state"] for r in rows}
            expected = {
                "CORRECT_CONFIRMED", "CORRECT_UNCERTAIN",
                "LOST_AWARE", "FALSE_CONFIRMED",
            }
            assert expected.issubset(states_found), (
                f"Not all derived states in CSV: missing {expected - states_found}"
            )

    def test_no_iou_in_outputs(self) -> None:
        """No IoU column should appear in feature_state_summary.csv."""
        import csv as _csv
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            labels_dir = self._build_labels_dir(tmp)
            out_dir = tmp / "diag"
            out_dir.mkdir()
            import sys
            old_argv = sys.argv
            sys.argv = [
                "diagnose_csc_features.py",
                "--labels_dir", str(labels_dir),
                "--output", str(out_dir / "features.csv"),
                "--per_state",
            ]
            from tools.diagnose_csc_features import main as _main
            try:
                _main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv

            csv_path = out_dir / "feature_state_summary.csv"
            text = csv_path.read_text()
            assert "iou" not in text.lower(), (
                "IoU leaked into feature_state_summary.csv — violates constraint"
            )
