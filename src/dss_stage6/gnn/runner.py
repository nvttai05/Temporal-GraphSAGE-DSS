from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib
import networkx
import numpy
import pandas as pd
import psutil
import scipy
import sklearn
import torch
import yaml

from dss_stage6.data import DatasetIndex
from dss_stage6.registry import RunRegistry

from .data import GNNGraphAdapter
from .model import GraphSAGERecommender
from .trainer import GNNTrainer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _device_receipt(requested: str) -> dict[str, Any]:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        nvidia = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if nvidia.returncode != 0:
            raise RuntimeError(f"CUDA torch is available but nvidia-smi failed: {nvidia.stderr}")
        return {
            "device": requested,
            "cuda_verified": True,
            "torch_cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "nvidia_smi": nvidia.stdout.strip(),
        }
    return {
        "device": "cpu",
        "cuda_verified": False,
        "torch_cuda_version": torch.version.cuda,
        "device_name": platform.processor() or "cpu",
        "nvidia_smi": "NOT_RUN_CPU_MODE",
    }


def _plot_curves(curves: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(curves["epoch"], curves["train_loss"], marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(curves["epoch"], curves["validation_ndcg_at_10"], label="NDCG@10")
    axes[1].plot(curves["epoch"], curves["validation_recall_at_10"], label="Recall@10")
    axes[1].set_title("Validation ranking")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def _cohort_metrics(session_metrics: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort, subset in {
        "all": session_metrics,
        "changed_profile": session_metrics.loc[session_metrics["changed_profile"] == 1],
        "low_history": session_metrics.loc[session_metrics["low_history"] == 1],
        "new_job": session_metrics.loc[session_metrics["has_positive_new_job"] == 1],
    }.items():
        if subset.empty:
            continue
        if cohort == "new_job":
            ndcg = subset["new_job_ndcg_at_10"].mean()
            recall = subset["new_job_recall_at_10"].mean()
        else:
            ndcg = subset["ndcg_at_10"].mean()
            recall = subset["recall_at_10"].mean()
        result[cohort] = {
            "eligible_sessions": len(subset),
            "ndcg_at_10": float(ndcg),
            "recall_at_10": float(recall),
        }
    return result


def _baseline_evidence(root: Path, commit: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stage62 = root / "artifacts" / "stage6_2" / "metrics.csv"
    if stage62.is_file():
        rows.append(
            {
                "family": "CBF/heuristic",
                "artifact": str(stage62.relative_to(root)).replace("\\", "/"),
                "evidence_status": "UNVERIFIED_MISSING_SAME_COMMIT_AND_MANIFEST_RECEIPT",
                "current_commit": commit,
                "ndcg_at_10": "",
                "recall_at_10": "",
            }
        )
    stage63 = root / "artifacts" / "stage6_3_local" / "smoke_metrics.csv"
    if stage63.is_file():
        rows.append(
            {
                "family": "XGBoost",
                "artifact": str(stage63.relative_to(root)).replace("\\", "/"),
                "evidence_status": "UNVERIFIED_LOCAL_SMOKE_NOT_SIX_CELL_VALIDATION",
                "current_commit": commit,
                "ndcg_at_10": "",
                "recall_at_10": "",
            }
        )
    return pd.DataFrame(rows)


def filter_model_variants(
    variants: list[tuple[str, bool, float]], selected_models: list[str] | tuple[str, ...]
) -> list[tuple[str, bool, float]]:
    """Keep requested model families without changing their configs or output names."""
    selected = set(selected_models)
    unknown = selected - {"static", "temporal"}
    if unknown:
        raise ValueError(f"Unknown GNN model families: {sorted(unknown)}")
    filtered = [
        variant
        for variant in variants
        if ("temporal" if variant[1] else "static") in selected
    ]
    if not filtered:
        raise ValueError("At least one GNN model family must be selected")
    return filtered


def _single_run(
    *,
    root: Path,
    index: DatasetIndex,
    config: dict[str, Any],
    output_dir: Path,
    seed: int,
    scenario: str,
    split: str,
    model_name: str,
    temporal: bool,
    half_life_days: float,
    model_seed: int,
    device: str,
    max_train_sessions: int | None,
    max_eval_sessions: int | None,
    max_epochs: int,
    resume: bool,
) -> tuple[dict[str, Any], list[Path]]:
    run_dir = output_dir / model_name / f"seed_{seed}_{scenario}"
    run_manifest_path = run_dir / "run_manifest.json"
    if resume and run_manifest_path.is_file():
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "COMPLETE":
            return manifest["metric_row"], [path for path in run_dir.rglob("*") if path.is_file()]
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    adapter = GNNGraphAdapter(index, seed, scenario)
    train_sessions = adapter.sessions(
        "train",
        # Censored sessions are never training negatives.  This remains true
        # for both the bounded smoke and the full validation path.
        eligible_only=True,
        max_sessions=max_train_sessions,
    )
    evaluation_sessions = adapter.sessions(
        split, eligible_only=True, max_sessions=max_eval_sessions
    )
    train_examples = [adapter.build(session) for session in train_sessions]
    evaluation_examples = [adapter.build(session) for session in evaluation_sessions]
    if not all(session.top10_eligible for session in train_sessions):
        raise AssertionError("Censored/non-eligible train session reached GNN loss")
    audit = pd.DataFrame(
        [example.audit for example in [*train_examples, *evaluation_examples]]
    )
    audit_path = run_dir / "graph_batch_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8", lineterminator="\n")
    if audit["future_or_current_violations"].sum() != 0:
        raise AssertionError("GNN graph audit contains leakage")
    torch.manual_seed(model_seed)
    model = GraphSAGERecommender(
        input_dim=adapter.input_dim,
        hidden_dim=config["shared_model"]["hidden_dim"],
        dropout=config["shared_model"]["dropout"],
        half_life_days=half_life_days,
        temporal=temporal,
    )
    parameter_count = model.parameter_count
    trainer = GNNTrainer(
        device=device,
        learning_rate=config["shared_model"]["learning_rate"],
        weight_decay=config["shared_model"]["weight_decay"],
        max_epochs=max_epochs,
        patience=config["shared_model"]["early_stopping_patience"],
        model_seed=model_seed,
    )
    checkpoint = run_dir / "checkpoint.pt"
    result = trainer.fit(
        model,
        train_examples,
        evaluation_examples,
        checkpoint,
        resume=resume,
    )
    curve_path = run_dir / "train_curves.csv"
    curve_plot_path = run_dir / "learning_curves.png"
    metrics_path = run_dir / "session_metrics.csv"
    prediction_path = run_dir / "predictions.csv"
    result.curves.to_csv(curve_path, index=False, encoding="utf-8", lineterminator="\n")
    _plot_curves(result.curves, curve_plot_path)
    result.validation_session_metrics.to_csv(
        metrics_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    result.predictions.to_csv(
        prediction_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    expected_pairs = {
        (session.session_id, job_id)
        for session in evaluation_sessions
        for job_id in session.exposed_job_ids
    }
    actual_pairs = set(
        result.predictions[["session_id", "job_id"]].itertuples(index=False, name=None)
    )
    duplicate_pairs = int(
        result.predictions.duplicated(["session_id", "job_id"]).sum()
    )
    if duplicate_pairs or actual_pairs != expected_pairs:
        raise AssertionError("GNN predictions do not exactly cover the exposed candidate set")

    restored = GraphSAGERecommender(
        input_dim=adapter.input_dim,
        hidden_dim=config["shared_model"]["hidden_dim"],
        dropout=config["shared_model"]["dropout"],
        half_life_days=half_life_days,
        temporal=temporal,
    ).to(device)
    checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=False)
    restored.load_state_dict(checkpoint_payload["model_state"])
    _, restored_predictions = trainer.predict(restored, evaluation_examples)
    scores_equal = bool(
        torch.equal(
            torch.tensor(result.predictions["score"].to_numpy()),
            torch.tensor(restored_predictions["score"].to_numpy()),
        )
    )
    if not scores_equal:
        raise AssertionError("Checkpoint round-trip changed prediction scores")
    acceptance_path = run_dir / "acceptance_audit.json"
    _write_json(
        acceptance_path,
        {
            "status": "PASS",
            "train_sessions_all_scoreable": all(
                session.top10_eligible for session in train_sessions
            ),
            "future_or_current_violations": int(
                audit["future_or_current_violations"].sum()
            ),
            "expected_exposed_pairs": len(expected_pairs),
            "prediction_rows": len(result.predictions),
            "duplicate_prediction_pairs": duplicate_pairs,
            "exposed_pair_coverage_exact": actual_pairs == expected_pairs,
            "checkpoint_restore_scores_exact": scores_equal,
            "shared_evaluator": "dss_stage6.evaluation.RankingEvaluator",
            "test_opened": split == "test",
        },
    )
    cohort_metrics = _cohort_metrics(result.validation_session_metrics)
    metric_row = {
        "model": model_name,
        "temporal": temporal,
        "half_life_days": half_life_days if temporal else "",
        "data_seed": seed,
        "scenario": scenario,
        "model_seed": model_seed,
        "split": split,
        "delta_days": 14,
        "eligible_sessions": len(evaluation_sessions),
        "ndcg_at_10": cohort_metrics["all"]["ndcg_at_10"],
        "recall_at_10": cohort_metrics["all"]["recall_at_10"],
        "checkpoint": str(checkpoint),
        "commit": _git_commit(root),
        "parameter_count": parameter_count,
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "runtime_seconds": result.runtime_seconds,
        "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
        "scope": "smoke" if max_train_sessions is not None else split,
    }
    manifest = {
        "status": "COMPLETE",
        "model": model_name,
        "shared_architecture": {
            "layers": 2,
            "hidden_dim": config["shared_model"]["hidden_dim"],
            "scorer": config["shared_model"]["scorer"],
            "loss": config["shared_model"]["loss"],
        },
        "only_model_difference": (
            "event_interaction_exponential_time_decay"
            if temporal
            else "uniform_event_interaction_weight"
        ),
        "content_membership_weight": 1.0,
        "cohort_metrics": cohort_metrics,
        "metric_row": metric_row,
        "artifacts": {
            path.name: _sha256(path)
            for path in (
                checkpoint,
                curve_path,
                curve_plot_path,
                metrics_path,
                prediction_path,
                audit_path,
                acceptance_path,
            )
        },
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json(run_manifest_path, manifest)
    artifacts = [
        checkpoint,
        curve_path,
        curve_plot_path,
        metrics_path,
        prediction_path,
        audit_path,
        acceptance_path,
        run_manifest_path,
    ]
    return metric_row, artifacts


def _write_comparison_outputs(
    output_dir: Path, metrics: pd.DataFrame, *, complete_validation: bool
) -> list[Path]:
    per_cell = output_dir / "per_cell_results.csv"
    comparison = output_dir / "model_comparison_validation.csv"
    costs = output_dir / "cost_comparison.csv"
    metrics.to_csv(per_cell, index=False, encoding="utf-8", lineterminator="\n")
    if complete_validation:
        summary = (
            metrics.groupby(["model", "half_life_days"], dropna=False, as_index=False)
            .agg(
                ndcg_at_10=("ndcg_at_10", "mean"),
                recall_at_10=("recall_at_10", "mean"),
                eligible_sessions=("eligible_sessions", "sum"),
                runtime_seconds=("runtime_seconds", "sum"),
                parameter_count=("parameter_count", "max"),
            )
        )
    else:
        summary = metrics.copy()
        summary["evidence_scope"] = "SMOKE_PROTOTYPE_NOT_BENCHMARK"
    summary.to_csv(comparison, index=False, encoding="utf-8", lineterminator="\n")
    metrics[
        [
            "model",
            "data_seed",
            "scenario",
            "runtime_seconds",
            "parameter_count",
            "peak_gpu_memory_bytes",
            "epochs_completed",
        ]
    ].to_csv(costs, index=False, encoding="utf-8", lineterminator="\n")
    for metric, filename, title in (
        ("ndcg_at_10", "model_comparison_ndcg10.png", "NDCG@10"),
        ("recall_at_10", "model_comparison_recall10.png", "Recall@10"),
    ):
        plot_path = output_dir / filename
        figure, axis = plt.subplots(figsize=(7, 4))
        values = summary.groupby("model")[metric].mean()
        values.plot(kind="bar", ax=axis)
        axis.set_ylabel(title)
        axis.set_title(
            f"{title} — {'six-cell validation' if complete_validation else 'smoke only'}"
        )
        figure.tight_layout()
        figure.savefig(plot_path, dpi=140)
        plt.close(figure)
    return [
        per_cell,
        comparison,
        costs,
        output_dir / "model_comparison_ndcg10.png",
        output_dir / "model_comparison_recall10.png",
    ]


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_yaml(root / "configs" / "stage6_gnn.yaml")
    protocol = _read_yaml(root / "configs" / "stage6_protocol.yaml")
    paths = _read_yaml(root / "configs" / "paths.local.yaml")
    dataset_root = (args.dataset_root or Path(paths["dataset_root"])).resolve()
    manifest_path = dataset_root / "data" / "candidate55" / "v0_3" / "dataset_manifest.json"
    environment = _device_receipt(args.device)
    environment.update(
        {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "networkx": networkx.__version__,
            "matplotlib": matplotlib.__version__,
            "psutil": psutil.__version__,
            "pyyaml": yaml.__version__,
            "platform": platform.platform(),
            "git_commit": _git_commit(root),
            "dataset_manifest_sha256": _sha256(manifest_path),
            "stage6_protocol_sha256": _sha256(root / "configs" / "stage6_protocol.yaml"),
            "experiments_sha256": _sha256(root / "configs" / "experiments.yaml"),
            "gnn_config_sha256": _sha256(root / "configs" / "stage6_gnn.yaml"),
        }
    )
    environment_path = output_dir / "environment.json"
    _write_json(environment_path, environment)
    baseline_path = output_dir / "baseline_evidence.csv"
    _baseline_evidence(root, environment["git_commit"]).to_csv(
        baseline_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    index = DatasetIndex(dataset_root, protocol)
    metrics: list[dict[str, Any]] = []
    artifacts: list[Path] = [environment_path, baseline_path]
    if args.mode == "smoke":
        cells = [(53021, "S1")]
        split = "validation"
        max_train = config["smoke"]["max_train_sessions"]
        max_eval = config["smoke"]["max_validation_sessions"]
        max_epochs = config["smoke"]["epochs"]
        requested = args.half_life_days or [config["smoke"]["half_life_days"]]
        half_lives = [float(value) for value in requested]
        variants = [("M0_static_graphsage", False, 0.0)] + [
            (f"M1_temporal_graphsage_hl{value:g}", True, value)
            for value in half_lives
        ]
    elif args.mode == "validation":
        cells = [(cell.seed, cell.scenario) for cell in index.data_cells]
        split = "validation"
        max_train = max_eval = None
        max_epochs = config["shared_model"]["max_epochs"]
        requested = args.half_life_days or config["temporal_graphsage"]["validation_half_life_days"]
        half_lives = [float(value) for value in requested]
        variants = [("M0_static_graphsage", False, 0.0)] + [
            (f"M1_temporal_graphsage_hl{value:g}", True, value) for value in half_lives
        ]
    else:
        if args.config_lock is None:
            raise ValueError("--config-lock is required for final mode")
        lock = json.loads(args.config_lock.read_text(encoding="utf-8"))
        if lock.get("status") != "LOCKED_FROM_SIX_CELL_VALIDATION":
            raise RuntimeError("Final test requires a six-cell validation config lock")
        cells = [(cell.seed, cell.scenario) for cell in index.data_cells]
        split = "test"
        max_train = max_eval = None
        max_epochs = int(lock["max_epochs"])
        half_life = float(lock["temporal_half_life_days"])
        variants = [("M0_static_graphsage", False, 0.0), (f"M1_temporal_graphsage_hl{half_life:g}", True, half_life)]
    variants = filter_model_variants(variants, args.models)
    for seed, scenario in cells:
        for model_name, temporal, half_life in variants:
            metric, run_artifacts = _single_run(
                root=root,
                index=index,
                config=config,
                output_dir=output_dir,
                seed=seed,
                scenario=scenario,
                split=split,
                model_name=model_name,
                temporal=temporal,
                half_life_days=half_life,
                model_seed=args.model_seed,
                device=environment["device"],
                max_train_sessions=max_train,
                max_eval_sessions=max_eval,
                max_epochs=max_epochs,
                resume=args.resume,
            )
            metrics.append(metric)
            artifacts.extend(run_artifacts)
    metric_frame = pd.DataFrame(metrics)
    complete_validation = args.mode == "validation" and len(cells) == 6
    artifacts.extend(
        _write_comparison_outputs(
            output_dir, metric_frame, complete_validation=complete_validation
        )
    )
    lock_eligible = complete_validation and set(args.models) == {"static", "temporal"}
    if lock_eligible:
        temporal_summary = (
            metric_frame.loc[metric_frame["temporal"]]
            .groupby("half_life_days", as_index=False)[["ndcg_at_10", "recall_at_10"]]
            .mean()
            .sort_values(
                ["ndcg_at_10", "recall_at_10", "half_life_days"],
                ascending=[False, False, True],
            )
        )
        selected_half_life = float(temporal_summary.iloc[0]["half_life_days"])
        lock_path = output_dir / "gnn_config_lock.json"
        _write_json(
            lock_path,
            {
                "status": "LOCKED_FROM_SIX_CELL_VALIDATION",
                "git_commit": environment["git_commit"],
                "dataset_manifest_sha256": environment["dataset_manifest_sha256"],
                "stage6_protocol_sha256": environment["stage6_protocol_sha256"],
                "model_seed": args.model_seed,
                "temporal_half_life_days": selected_half_life,
                "max_epochs": max_epochs,
                "selection": "macro_ndcg_at_10_then_recall_at_10_then_lower_half_life",
                "test_opened": False,
            },
        )
        artifacts.append(lock_path)
    registry_path = RunRegistry(root, output_dir / "run_registry").register(
        run_id=f"gnn_{args.mode}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        config={
            "mode": args.mode,
            "models": list(args.models),
            "requested_half_life_days": args.half_life_days,
            "gnn": config,
            "environment": environment,
        },
        seeds={"model_seed": args.model_seed, "data_seeds": [53021, 53022]},
        started_monotonic=args.started_monotonic,
        artifacts=artifacts,
        status="COMPLETE",
    )
    run_receipt_path = output_dir / "run_receipt.json"
    _write_json(
        run_receipt_path,
        {
            "status": "COMPLETE",
            "mode": args.mode,
            "models": list(args.models),
            "scope": "SMOKE_PROTOTYPE" if args.mode == "smoke" else args.mode.upper(),
            "test_opened": args.mode == "final",
            "metric_rows": len(metric_frame),
            "registry": str(registry_path),
            "runtime_seconds": time.monotonic() - args.started_monotonic,
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static/Temporal GraphSAGE Stage 6 runner")
    parser.add_argument("--mode", choices=["smoke", "validation", "final"], required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--half-life-days", type=float, nargs="*")
    parser.add_argument(
        "--models",
        choices=["static", "temporal"],
        nargs="+",
        default=["static", "temporal"],
        help="Model families to run; defaults to both.",
    )
    parser.add_argument("--config-lock", type=Path)
    args = parser.parse_args(argv)
    args.started_monotonic = time.monotonic()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
