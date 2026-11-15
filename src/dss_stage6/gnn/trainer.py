from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dss_stage6.evaluation import RankingEvaluator

from .data import GraphExample
from .model import GraphSAGERecommender


def weighted_bce_logits(
    logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits, labels.to(dtype=logits.dtype), pos_weight=pos_weight
    )


@dataclass(frozen=True)
class TrainingResult:
    model: GraphSAGERecommender
    curves: pd.DataFrame
    validation_session_metrics: pd.DataFrame
    predictions: pd.DataFrame
    best_epoch: int
    epochs_completed: int
    runtime_seconds: float
    peak_gpu_memory_bytes: int


class GNNTrainer:
    def __init__(
        self,
        *,
        device: str,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        patience: int,
        model_seed: int,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self.device = torch.device(device)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.model_seed = int(model_seed)

    def _seed(self) -> None:
        torch.manual_seed(self.model_seed)
        np.random.seed(self.model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.model_seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    @staticmethod
    def _positive_weight(examples: list[GraphExample], device: torch.device) -> torch.Tensor:
        labels = torch.cat([example.labels for example in examples])
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        if positives == 0:
            raise ValueError("Training graph examples contain no positives")
        return torch.tensor(negatives / positives, dtype=torch.float32, device=device)

    @torch.no_grad()
    def predict(
        self, model: GraphSAGERecommender, examples: list[GraphExample]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        model.eval()
        evaluator = RankingEvaluator(10)
        prediction_frames: list[pd.DataFrame] = []
        metric_rows: list[dict[str, Any]] = []
        for original in examples:
            example = original.to(self.device)
            logits = model(example)
            scores = torch.sigmoid(logits).detach().cpu().numpy()
            score_map = dict(
                zip(original.session.exposed_job_ids, scores.tolist(), strict=True)
            )
            metric_rows.append(evaluator.evaluate_session(original.session, score_map))
            frame = original.session.candidates[
                [
                    "seed",
                    "scenario",
                    "split",
                    "delta_days",
                    "session_id",
                    "candidate_id",
                    "job_id",
                    "label",
                    "new_job",
                    "changed_profile",
                    "low_history",
                ]
            ].copy()
            frame["score"] = frame["job_id"].astype(str).map(score_map)
            prediction_frames.append(frame)
        return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)

    def fit(
        self,
        model: GraphSAGERecommender,
        train_examples: list[GraphExample],
        validation_examples: list[GraphExample],
        checkpoint_path: str | Path,
        *,
        resume: bool = False,
    ) -> TrainingResult:
        self._seed()
        started = time.monotonic()
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        start_epoch = 1
        best_epoch = 0
        best_ndcg = float("-inf")
        best_recall = float("-inf")
        best_state: dict[str, torch.Tensor] | None = None
        curves: list[dict[str, Any]] = []
        stale_epochs = 0
        if resume and checkpoint.is_file():
            payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            start_epoch = int(payload["epoch"]) + 1
            best_epoch = int(payload["best_epoch"])
            best_ndcg = float(payload["best_ndcg"])
            best_recall = float(payload["best_recall"])
            best_state = payload["best_state"]
            curves = list(payload.get("curves", []))
        pos_weight = self._positive_weight(train_examples, self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        epochs_completed = start_epoch - 1
        for epoch in range(start_epoch, self.max_epochs + 1):
            model.train()
            losses: list[float] = []
            gradient_norms: list[float] = []
            for original in train_examples:
                example = original.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(example)
                loss = weighted_bce_logits(logits, example.labels, pos_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite GNN loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                gradient_norms.append(float(gradient_norm.detach().cpu()))
            validation_metrics, _ = self.predict(model, validation_examples)
            ndcg = float(validation_metrics["ndcg_at_10"].mean())
            recall = float(validation_metrics["recall_at_10"].mean())
            curve = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "gradient_norm": float(np.mean(gradient_norms)),
                "validation_ndcg_at_10": ndcg,
                "validation_recall_at_10": recall,
            }
            curves.append(curve)
            improved = ndcg > best_ndcg or (
                np.isclose(ndcg, best_ndcg) and recall > best_recall
            )
            if improved:
                best_ndcg = ndcg
                best_recall = recall
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            epochs_completed = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_ndcg": best_ndcg,
                    "best_recall": best_recall,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_state": best_state,
                    "curves": curves,
                    "model_seed": self.model_seed,
                },
                checkpoint,
            )
            if stale_epochs >= self.patience:
                break
        if best_state is None:
            raise RuntimeError("Training completed without a best checkpoint")
        model.load_state_dict(best_state)
        final_payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        final_payload["model_state"] = best_state
        final_payload["checkpoint_role"] = "best_validation_model"
        torch.save(final_payload, checkpoint)
        validation_metrics, predictions = self.predict(model, validation_examples)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        return TrainingResult(
            model=model,
            curves=pd.DataFrame(curves),
            validation_session_metrics=validation_metrics,
            predictions=predictions,
            best_epoch=best_epoch,
            epochs_completed=epochs_completed,
            runtime_seconds=time.monotonic() - started,
            peak_gpu_memory_bytes=peak_memory,
        )
