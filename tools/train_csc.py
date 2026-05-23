"""Train the CSC model (GRU / MLP / TCN) with composite heads (localization + confidence)."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from csc_lib.csc.config import CSCTrainConfig
from csc_lib.csc.dataset import build_train_val_datasets
from csc_lib.csc.features import FEATURE_DIM, FEATURE_NAMES
from csc_lib.csc.labeling.label_schema import (
    AUX_FLAGS,
    DERIVED_NAMES,
    LOCALIZATION_NAMES,
    NUM_CONFIDENCE_STATES,
    NUM_DERIVED_STATES,
    NUM_LOCALIZATION_STATES,
    ConfidenceState,
    DerivedState,
    LocalizationState,
    derive_state,
)
from csc_lib.csc.model import build_model
from csc_lib.eval.custom_metrics.scene_state_metrics import (
    confusion_matrix,
    failure_auprc,
    failure_auroc,
    macro_f1,
    per_state_prf,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_yaml(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _derive_state_array(loc: np.ndarray, conf: np.ndarray) -> np.ndarray:
    out = np.zeros_like(loc)
    for i in range(len(loc)):
        out[i] = int(derive_state(LocalizationState(int(loc[i])), ConfidenceState(int(conf[i]))))
    return out


# ---------------------------------------------------------------------------
# Focal CE Loss
# ---------------------------------------------------------------------------


class FocalCELoss:
    """Focal variant of CrossEntropyLoss.

    ``FocalCELoss(gamma=2.0, weight=w)(logits, targets)`` is equivalent to:
        p_t = softmax(logits)[target]
        loss = -(1 - p_t)^gamma * log(p_t)
    with optional class weights applied before the focal modulation.

    Implemented as a plain callable (not nn.Module) to keep the training
    loop uniform — it is called exactly like CrossEntropyLoss.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: "torch.Tensor | None" = None,
        reduction: str = "mean",
    ) -> None:
        import torch
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self._ce = torch.nn.CrossEntropyLoss(
            weight=weight,
            reduction="none",
        )

    def __call__(
        self,
        logits: "torch.Tensor",
        targets: "torch.Tensor",
    ) -> "torch.Tensor":
        import torch
        import torch.nn.functional as F

        # Standard CE per-element: (N,)
        ce = self._ce(logits, targets)

        # p_t = probability of the correct class
        probs = F.softmax(logits, dim=-1)
        # gather correct-class probability along the class axis
        p_t = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    # Make isinstance checks work (e.g. old code that checks nn.Module)
    def to(self, device: str) -> "FocalCELoss":
        if self.weight is not None:
            self.weight = self.weight.to(device)
            self._ce = type(self._ce)(weight=self.weight, reduction="none")
        return self


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CSC model (GRU/MLP/TCN, composite heads).")
    p.add_argument("--config", default="configs/csc/csc_gru.yaml")
    p.add_argument("--labels_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    return p.parse_args()


def main() -> int:
    import torch
    from torch.utils.data import DataLoader

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("train_csc")

    args = parse_args()
    raw_cfg = _load_yaml(Path(args.config))
    if args.labels_dir:
        raw_cfg["labels_dir"] = args.labels_dir
    if args.output_dir:
        raw_cfg["output_dir"] = args.output_dir
    if args.device:
        raw_cfg["device"] = args.device
    if args.seed is not None:
        raw_cfg["seed"] = args.seed
    if args.max_epochs is not None:
        raw_cfg.setdefault("optim", {})["epochs"] = args.max_epochs

    cfg = CSCTrainConfig.from_dict(raw_cfg)
    if not cfg.labels_dir or not cfg.output_dir:
        raise SystemExit("config must specify labels_dir and output_dir")
    cfg.model.feature_dim = FEATURE_DIM
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    model_kind = getattr(cfg.model, "kind", "gru")
    log.info("device=%s, feature_dim=%d, model_kind=%s", device, FEATURE_DIM, model_kind)

    # ---- Dataset ----
    use_sampler = getattr(cfg.optim, "use_balanced_sampler", True)
    if use_sampler:
        train_ds, val_ds, info, sampler = build_train_val_datasets(
            cfg.labels_dir, cfg.feature,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
            return_sampler=True,
        )
    else:
        train_ds, val_ds, info = build_train_val_datasets(
            cfg.labels_dir, cfg.feature,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
        )
        sampler = None

    log.info(
        "data: %d train seqs / %d val seqs ; %d train windows / %d val windows",
        info["n_train_sequences"], info["n_val_sequences"],
        info["n_train_windows"], info["n_val_windows"],
    )
    if info["n_train_windows"] == 0:
        raise SystemExit("no training windows generated — check labels_dir and window_size")

    (cfg.output_dir / "label_mapping.json").write_text(json.dumps(
        {
            "localization_states": LOCALIZATION_NAMES,
            "confidence_states": [s.name for s in ConfidenceState],
            "derived_states": DERIVED_NAMES,
            "aux": list(AUX_FLAGS),
            "feature_names": list(FEATURE_NAMES),
        },
        indent=2,
    ))

    if sampler is not None:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.optim.batch_size,
            sampler=sampler, num_workers=0,
        )
        log.info("using WeightedRandomSampler (balanced classes)")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.optim.batch_size,
            shuffle=True, num_workers=0,
        )
    val_loader = DataLoader(val_ds, batch_size=cfg.optim.batch_size, shuffle=False, num_workers=0)

    model = build_model(cfg.model).to(device)
    log.info("%s params=%d", model_kind.upper(), model.num_params)

    enable_forecast = bool(getattr(cfg.model, "enable_forecast_heads", False))
    if enable_forecast:
        log.info(
            "V3 forecast heads ENABLED — horizon=%d (failure/fc/lost _next_10)",
            getattr(cfg.model, "forecast_horizon", 10),
        )

    # ---- Compute pos_weight for forecast BCE losses from training set ----
    # Imbalance is severe (FC_next_10 typically 1-4%, LOST 10-30%).  Without
    # pos_weight the BCE collapses to predicting all-zeros; AUPRC craters.
    # We compute pos_weight = (n_negatives / n_positives) per head over all
    # NON-IGNORED training windows.
    forecast_pos_weights: dict[str, torch.Tensor] = {}
    if enable_forecast:
        n_pos = {"failure": 0, "fc": 0, "lost": 0}
        n_neg = {"failure": 0, "fc": 0, "lost": 0}
        for w in train_ds.windows:
            mask = (1 - w.ignore_forecast).astype(bool)
            if not mask.any():
                continue
            n_pos["failure"] += int(w.failure_next_10[mask].sum())
            n_neg["failure"] += int((1 - w.failure_next_10[mask]).sum())
            n_pos["fc"] += int(w.false_confirmed_next_10[mask].sum())
            n_neg["fc"] += int((1 - w.false_confirmed_next_10[mask]).sum())
            n_pos["lost"] += int(w.lost_aware_next_10[mask].sum())
            n_neg["lost"] += int((1 - w.lost_aware_next_10[mask]).sum())
        for k in ("failure", "fc", "lost"):
            pw = (n_neg[k] / max(1, n_pos[k])) if n_pos[k] > 0 else 1.0
            # Cap pos_weight at 100 to avoid pathologic gradients on near-zero positives
            pw = float(min(pw, 100.0))
            forecast_pos_weights[k] = torch.tensor([pw], device=device)
        log.info(
            "forecast pos_weights: failure=%.2f fc=%.2f lost=%.2f (raw n_pos=%s, n_neg=%s)",
            forecast_pos_weights["failure"].item(),
            forecast_pos_weights["fc"].item(),
            forecast_pos_weights["lost"].item(),
            n_pos, n_neg,
        )

    # ---- Loss functions ----
    # derived_loss is PRIMARY (direct supervision on the 4-class paper state).
    # loc_loss and conf_loss are AUXILIARY (strengthen the shared encoder).
    #   weights: CORRECT_CONFIRMED / CORRECT_UNCERTAIN / LOST_AWARE / FALSE_CONFIRMED
    der_weights  = torch.tensor([1.0, 1.2, 2.0, 2.5], device=device)
    loc_weights  = torch.tensor([1.0, 1.5, 2.0],       device=device)  # STABLE/UNCERTAIN/LOST
    conf_weights = torch.tensor([1.0, 1.5],             device=device)  # LOW/HIGH

    use_focal   = getattr(cfg.loss, "use_focal", False)
    focal_gamma = getattr(cfg.loss, "focal_gamma", 2.0)

    if use_focal:
        der_loss_fn  = FocalCELoss(gamma=focal_gamma, weight=der_weights)
        loc_loss_fn  = FocalCELoss(gamma=focal_gamma, weight=loc_weights)
        conf_loss_fn = FocalCELoss(gamma=focal_gamma, weight=conf_weights)
        log.info("using FocalCELoss (gamma=%.1f)", focal_gamma)
    else:
        der_loss_fn  = torch.nn.CrossEntropyLoss(weight=der_weights)
        loc_loss_fn  = torch.nn.CrossEntropyLoss(weight=loc_weights)
        conf_loss_fn = torch.nn.CrossEntropyLoss(weight=conf_weights)

    aux_loss_fn = torch.nn.BCEWithLogitsLoss()

    # V3 forecast losses — masked BCE with pos_weight per head.
    if enable_forecast:
        forecast_failure_loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=forecast_pos_weights["failure"], reduction="none"
        )
        forecast_fc_loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=forecast_pos_weights["fc"], reduction="none"
        )
        forecast_lost_loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=forecast_pos_weights["lost"], reduction="none"
        )

    if cfg.optim.optimizer.lower() == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    log_path = cfg.output_dir / "train_log.jsonl"
    log_fh = open(log_path, "w")

    best_metric = -float("inf")
    best_path = cfg.output_dir / "checkpoint_best.pth"
    last_path = cfg.output_dir / "checkpoint_last.pth"
    patience = 0

    def _save_ckpt(path: Path, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "config": cfg.to_dict(),
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "feature_names": list(FEATURE_NAMES),
                "localization_names": LOCALIZATION_NAMES,
                "derived_names": DERIVED_NAMES,
            },
            path,
        )

    def _evaluate() -> dict:
        model.eval()
        loc_p: list[np.ndarray] = []
        loc_t: list[np.ndarray] = []
        conf_p: list[np.ndarray] = []
        conf_t: list[np.ndarray] = []
        derived_p: list[np.ndarray] = []
        derived_t: list[np.ndarray] = []
        risk: list[np.ndarray] = []
        truef: list[np.ndarray] = []
        # V3 forecast accumulators
        fc_failure_p: list[np.ndarray] = []
        fc_failure_t: list[np.ndarray] = []
        fc_fc_p: list[np.ndarray] = []
        fc_fc_t: list[np.ndarray] = []
        fc_lost_p: list[np.ndarray] = []
        fc_lost_t: list[np.ndarray] = []
        fc_mask: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                x        = batch["features"].to(device)
                loc_y    = batch["localization"].to(device)
                conf_y   = batch["confidence"].to(device)
                derived_y = batch["derived"].to(device)
                out = model(x)

                # Primary: use derived head directly
                der_probs  = torch.softmax(out.derived_logits, dim=-1)
                der_pred   = der_probs.argmax(dim=-1).cpu().numpy().reshape(-1)
                # Auxiliary
                loc_pred  = out.localization_logits.argmax(dim=-1).cpu().numpy().reshape(-1)
                conf_pred = out.confidence_logits.argmax(dim=-1).cpu().numpy().reshape(-1)
                # Risk = P(LOST_AWARE) + P(FALSE_CONFIRMED) from derived head
                risk_scores = (der_probs[..., 2] + der_probs[..., 3]).flatten().cpu().numpy()

                loc_p.append(loc_pred)
                loc_t.append(loc_y.flatten().cpu().numpy())
                conf_p.append(conf_pred)
                conf_t.append(conf_y.flatten().cpu().numpy())
                derived_p.append(der_pred)
                derived_t.append(derived_y.flatten().cpu().numpy())
                risk.append(risk_scores)
                truef.append((derived_y.flatten().cpu().numpy() >= int(DerivedState.LOST_AWARE)).astype(np.int8))

                # V3 forecast metrics — only when forecast heads present
                if out.failure_next_10_logit is not None:
                    fail_y = batch["failure_next_10"].to(device)
                    fc_y = batch["false_confirmed_next_10"].to(device)
                    lost_y = batch["lost_aware_next_10"].to(device)
                    ig = batch["ignore_forecast"].to(device)
                    fail_prob = torch.sigmoid(out.failure_next_10_logit.squeeze(-1))
                    fc_prob = torch.sigmoid(out.false_confirmed_next_10_logit.squeeze(-1))
                    lost_prob = torch.sigmoid(out.lost_aware_next_10_logit.squeeze(-1))
                    valid = (1 - ig).flatten().cpu().numpy().astype(bool)
                    fc_failure_p.append(fail_prob.flatten().cpu().numpy())
                    fc_failure_t.append(fail_y.flatten().cpu().numpy())
                    fc_fc_p.append(fc_prob.flatten().cpu().numpy())
                    fc_fc_t.append(fc_y.flatten().cpu().numpy())
                    fc_lost_p.append(lost_prob.flatten().cpu().numpy())
                    fc_lost_t.append(lost_y.flatten().cpu().numpy())
                    fc_mask.append(valid)
        if not loc_p:
            return {}
        loc_p = np.concatenate(loc_p)
        loc_t = np.concatenate(loc_t)
        conf_p = np.concatenate(conf_p)
        conf_t = np.concatenate(conf_t)
        dp = np.concatenate(derived_p)
        dt = np.concatenate(derived_t)
        rk = np.concatenate(risk)
        tf = np.concatenate(truef)
        der_per = per_state_prf(dt, dp, n_states=NUM_DERIVED_STATES, state_names=DERIVED_NAMES)
        result = {
            "derived_macro_f1": macro_f1(dt, dp, n_states=NUM_DERIVED_STATES),
            "loc_macro_f1":     macro_f1(loc_t, loc_p, n_states=NUM_LOCALIZATION_STATES),
            "conf_macro_f1":    macro_f1(conf_t, conf_p, n_states=NUM_CONFIDENCE_STATES),
            "failure_auroc":    failure_auroc(tf, rk),
            "failure_auprc":    failure_auprc(tf, rk),
            "n_eval":           int(dp.size),
            "derived_per_state": der_per,
            "loc_per_state":    per_state_prf(loc_t, loc_p, n_states=NUM_LOCALIZATION_STATES, state_names=LOCALIZATION_NAMES),
        }
        if fc_failure_p:
            mask = np.concatenate(fc_mask)
            for name, ps_buf, ts_buf in [
                ("failure_next_10", fc_failure_p, fc_failure_t),
                ("false_confirmed_next_10", fc_fc_p, fc_fc_t),
                ("lost_aware_next_10", fc_lost_p, fc_lost_t),
            ]:
                ps = np.concatenate(ps_buf)[mask]
                ts = np.concatenate(ts_buf)[mask].astype(np.int8)
                if ts.size > 0 and (ts.sum() > 0) and (ts.sum() < ts.size):
                    result[f"{name}_auroc"] = failure_auroc(ts, ps)
                    result[f"{name}_auprc"] = failure_auprc(ts, ps)
                else:
                    # One class absent in val split — metrics undefined
                    result[f"{name}_auroc"] = float("nan")
                    result[f"{name}_auprc"] = float("nan")
                result[f"{name}_n"] = int(ts.size)
                result[f"{name}_pos_rate"] = float(ts.mean()) if ts.size else 0.0
        return result

    epochs = cfg.optim.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_seen = 0
        t0 = time.perf_counter()
        for batch in train_loader:
            x        = batch["features"].to(device)
            loc_y    = batch["localization"].to(device)
            conf_y   = batch["confidence"].to(device)
            derived_y = batch["derived"].to(device)
            a        = batch["aux"].to(device)
            out = model(x)
            l_derived = der_loss_fn(
                out.derived_logits.reshape(-1, NUM_DERIVED_STATES),
                derived_y.reshape(-1),
            )
            l_loc = loc_loss_fn(
                out.localization_logits.reshape(-1, NUM_LOCALIZATION_STATES),
                loc_y.reshape(-1),
            )
            l_conf = conf_loss_fn(
                out.confidence_logits.reshape(-1, NUM_CONFIDENCE_STATES),
                conf_y.reshape(-1),
            )
            l_aux = aux_loss_fn(out.aux_logits, a)
            # derived is PRIMARY (1.0), loc and conf are AUXILIARY (0.3 each)
            loss = l_derived + 0.3 * l_loc + 0.3 * l_conf + cfg.loss.aux_weight * l_aux

            # V3 forecast losses (masked BCE with pos_weight)
            if enable_forecast and out.failure_next_10_logit is not None:
                fail_y = batch["failure_next_10"].to(device).float()
                fc_y   = batch["false_confirmed_next_10"].to(device).float()
                lost_y = batch["lost_aware_next_10"].to(device).float()
                ig     = batch["ignore_forecast"].to(device).float()
                valid_mask = (1.0 - ig)  # 1 where forecast target is valid
                n_valid = valid_mask.sum().clamp_min(1.0)

                fail_logit = out.failure_next_10_logit.squeeze(-1)
                fc_logit   = out.false_confirmed_next_10_logit.squeeze(-1)
                lost_logit = out.lost_aware_next_10_logit.squeeze(-1)

                l_fail = (forecast_failure_loss_fn(fail_logit, fail_y) * valid_mask).sum() / n_valid
                l_fc_n10 = (forecast_fc_loss_fn(fc_logit, fc_y) * valid_mask).sum() / n_valid
                l_lost_n10 = (forecast_lost_loss_fn(lost_logit, lost_y) * valid_mask).sum() / n_valid

                loss = (
                    loss
                    + cfg.loss.forecast_failure_weight * l_fail
                    + cfg.loss.forecast_fc_weight * l_fc_n10
                    + cfg.loss.forecast_lost_weight * l_lost_n10
                )

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optim.step()
            bs = x.size(0)
            ep_loss += loss.item() * bs
            n_seen += bs
        train_loss = ep_loss / max(1, n_seen)
        metrics = _evaluate()
        epoch_time = time.perf_counter() - t0

        # Derive key per-state recalls from PRIMARY (derived) head
        der_per = metrics.get("derived_per_state", {})
        fc_recall   = der_per.get("FALSE_CONFIRMED", {}).get("recall", 0.0)
        lost_recall = der_per.get("LOST_AWARE", {}).get("recall", 0.0)

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_derived_f1":    metrics.get("derived_macro_f1", 0.0),
            "val_loc_f1":        metrics.get("loc_macro_f1", 0.0),
            "val_conf_f1":       metrics.get("conf_macro_f1", 0.0),
            "val_failure_auroc": metrics.get("failure_auroc", 0.0),
            "val_failure_auprc": metrics.get("failure_auprc", 0.0),
            "val_fc_recall":     fc_recall,
            "val_lost_recall":   lost_recall,
            "epoch_time_s":      epoch_time,
        }
        if enable_forecast:
            log_row.update({
                "val_failure_n10_auroc": metrics.get("failure_next_10_auroc", float("nan")),
                "val_failure_n10_auprc": metrics.get("failure_next_10_auprc", float("nan")),
                "val_fc_n10_auroc":      metrics.get("false_confirmed_next_10_auroc", float("nan")),
                "val_fc_n10_auprc":      metrics.get("false_confirmed_next_10_auprc", float("nan")),
                "val_lost_n10_auroc":    metrics.get("lost_aware_next_10_auroc", float("nan")),
                "val_lost_n10_auprc":    metrics.get("lost_aware_next_10_auprc", float("nan")),
            })
        log_fh.write(json.dumps(log_row) + "\n")
        log_fh.flush()
        log.info(
            "ep %d/%d | loss=%.4f | derivedF1=%.3f locF1=%.3f confF1=%.3f "
            "AUROC=%.3f AUPRC=%.3f FC_recall=%.3f LOST_recall=%.3f | %.1fs",
            epoch, epochs, train_loss,
            metrics.get("derived_macro_f1", 0.0),
            metrics.get("loc_macro_f1", 0.0),
            metrics.get("conf_macro_f1", 0.0),
            metrics.get("failure_auroc", 0.0),
            metrics.get("failure_auprc", 0.0),
            fc_recall, lost_recall,
            epoch_time,
        )
        if enable_forecast:
            log.info(
                "  forecast: failure_n10 AUPRC=%.3f | fc_n10 AUPRC=%.3f | lost_n10 AUPRC=%.3f",
                metrics.get("failure_next_10_auprc", float("nan")),
                metrics.get("false_confirmed_next_10_auprc", float("nan")),
                metrics.get("lost_aware_next_10_auprc", float("nan")),
            )

        # Selection: V2 uses derived_F1 + failure_AUPRC (50/50);
        # V3 uses 4-component blend: 0.4*derived_F1 + 0.3*fc_n10 + 0.2*lost_n10 + 0.1*failure_n10
        if enable_forecast:
            def _safe(x):
                return 0.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)
            sel = (
                0.4 * _safe(metrics.get("derived_macro_f1", 0.0))
                + 0.3 * _safe(metrics.get("false_confirmed_next_10_auprc", 0.0))
                + 0.2 * _safe(metrics.get("lost_aware_next_10_auprc", 0.0))
                + 0.1 * _safe(metrics.get("failure_next_10_auprc", 0.0))
            )
        else:
            sel = 0.5 * metrics.get("derived_macro_f1", 0.0) + 0.5 * metrics.get("failure_auprc", 0.0)
        _save_ckpt(last_path, epoch, metrics)
        if sel > best_metric:
            best_metric = sel
            patience = 0
            _save_ckpt(best_path, epoch, metrics)
        else:
            patience += 1
            if patience >= cfg.optim.early_stopping_patience:
                log.info("early stopping at epoch %d", epoch)
                break

    log_fh.close()

    # Final val metrics with best checkpoint
    blob = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["state_dict"])
    final = _evaluate()
    (cfg.output_dir / "val_metrics.json").write_text(json.dumps(final, indent=2))
    if enable_forecast:
        forecast_only = {
            k: final[k]
            for k in final
            if k.startswith(("failure_next_10", "false_confirmed_next_10", "lost_aware_next_10"))
        }
        (cfg.output_dir / "forecast_metrics.json").write_text(
            json.dumps(forecast_only, indent=2)
        )
    (cfg.output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    (cfg.output_dir / "split_info.json").write_text(json.dumps(info, indent=2))
    log.info("training done; best=%.4f -> %s", best_metric, cfg.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
