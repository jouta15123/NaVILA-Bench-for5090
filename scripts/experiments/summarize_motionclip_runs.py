#!/usr/bin/env python3
"""Summarize MotionCLIP joint training runs into paper-friendly tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

STEP_RE = re.compile(r"^\[Step\s+(\d+)\]")
MPJPE_RE = re.compile(r"MPJPE:\s*([0-9.]+)")


@dataclass
class RunSummary:
    run_name: str
    run_prefix: str | None
    run_date: str | None
    git_commit: str | None
    seed: int | None
    sem_encoder: str | None
    label_mode: str | None
    split_ratio: float | None
    best_metric: str | None
    log_interval: int | None
    eval_interval: int | None
    scheduler: str | None
    plateau_patience: int | None
    plateau_factor: float | None
    best_step: int | None
    avg_r1: float | None
    m2t_r1: float | None
    t2m_r1: float | None
    silhouette: float | None
    mpjpe: float | None
    lambda_contrastive: float | None
    lambda_vae: float | None
    temp: float | None
    lr: float | None
    weight_decay: float | None
    freeze_steps: int | None
    full_steps: int | None


@dataclass
class AggregatedSummary:
    config_id: str
    n_runs: int
    run_prefix: str | None
    git_commit: str | None
    sem_encoder: str | None
    label_mode: str | None
    split_ratio: float | None
    best_metric: str | None
    log_interval: int | None
    eval_interval: int | None
    scheduler: str | None
    plateau_patience: int | None
    plateau_factor: float | None
    lambda_contrastive: float | None
    lambda_vae: float | None
    temp: float | None
    lr: float | None
    weight_decay: float | None
    freeze_steps: int | None
    full_steps: int | None
    best_step_mean: float | None
    best_step_std: float | None
    avg_r1_mean: float | None
    avg_r1_std: float | None
    m2t_r1_mean: float | None
    m2t_r1_std: float | None
    t2m_r1_mean: float | None
    t2m_r1_std: float | None
    silhouette_mean: float | None
    silhouette_std: float | None
    mpjpe_mean: float | None
    mpjpe_std: float | None


def _parse_runs_arg(runs_arg: str) -> list[str]:
    return [r.strip() for r in runs_arg.split(",") if r.strip()]


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _latest_log_file(logs_dir: Path) -> Path | None:
    if not logs_dir.exists():
        return None
    candidates = sorted(logs_dir.glob("train_log_*.txt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _mpjpe_by_step(log_path: Path | None) -> dict[int, float]:
    if log_path is None or not log_path.exists():
        return {}
    result: dict[int, float] = {}
    current_step: int | None = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            step_match = STEP_RE.search(line)
            if step_match:
                current_step = int(step_match.group(1))
                continue
            mpjpe_match = MPJPE_RE.search(line)
            if mpjpe_match and current_step is not None:
                result[current_step] = float(mpjpe_match.group(1))
    return result


def _best_metrics(metrics_path: Path) -> tuple[int | None, dict[str, float]]:
    best_step: int | None = None
    best_avg_r1 = float("-inf")
    best_payload: dict[str, float] = {}

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            step = int(record.get("step", 0))
            m2t_r1 = _safe_float(record.get("m2t", {}).get("R@1")) or 0.0
            t2m_r1 = _safe_float(record.get("t2m", {}).get("R@1")) or 0.0
            avg_r1 = 0.5 * (m2t_r1 + t2m_r1)

            if avg_r1 > best_avg_r1:
                best_avg_r1 = avg_r1
                best_step = step
                best_payload = {
                    "avg_r1": avg_r1,
                    "m2t_r1": m2t_r1,
                    "t2m_r1": t2m_r1,
                    "silhouette": _safe_float(record.get("silhouette")),
                }

    return best_step, best_payload


def _load_run_config(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def summarize_run(run_dir: Path) -> RunSummary | None:
    metrics_path = run_dir / "retrieval_metrics.jsonl"
    if not metrics_path.exists():
        return None

    best_step, best_payload = _best_metrics(metrics_path)
    mpjpe_map = _mpjpe_by_step(_latest_log_file(run_dir / "logs"))
    config = _load_run_config(run_dir)

    mpjpe_val = mpjpe_map.get(best_step) if best_step is not None else None

    return RunSummary(
        run_name=run_dir.name,
        run_prefix=str(config.get("run_prefix")) if config.get("run_prefix") else None,
        run_date=str(config.get("run_date")) if config.get("run_date") else None,
        git_commit=str(config.get("git_commit")) if config.get("git_commit") else None,
        seed=_safe_int(config.get("seed")),
        sem_encoder=str(config.get("sem_encoder")) if config.get("sem_encoder") else None,
        label_mode=str(config.get("label_mode")) if config.get("label_mode") else None,
        split_ratio=_safe_float(config.get("split_ratio")),
        best_metric=str(config.get("best_metric")) if config.get("best_metric") else None,
        log_interval=_safe_int(config.get("log_interval")),
        eval_interval=_safe_int(config.get("eval_interval")),
        scheduler=str(config.get("scheduler")) if config.get("scheduler") else None,
        plateau_patience=_safe_int(config.get("plateau_patience")),
        plateau_factor=_safe_float(config.get("plateau_factor")),
        best_step=best_step,
        avg_r1=_safe_float(best_payload.get("avg_r1")),
        m2t_r1=_safe_float(best_payload.get("m2t_r1")),
        t2m_r1=_safe_float(best_payload.get("t2m_r1")),
        silhouette=_safe_float(best_payload.get("silhouette")),
        mpjpe=_safe_float(mpjpe_val),
        lambda_contrastive=_safe_float(config.get("lambda_contrastive")),
        lambda_vae=_safe_float(config.get("lambda_vae")),
        temp=_safe_float(config.get("temp")),
        lr=_safe_float(config.get("lr")),
        weight_decay=_safe_float(config.get("weight_decay")),
        freeze_steps=_safe_int(config.get("freeze_steps")),
        full_steps=_safe_int(config.get("full_steps")),
    )


def _iter_run_dirs(results_dir: Path, runs: Iterable[str] | None, prefix: str) -> list[Path]:
    if runs:
        return [results_dir / r for r in runs]
    return sorted([p for p in results_dir.glob(f"{prefix}*") if p.is_dir()])


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, None
    var = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    return mean, math.sqrt(max(var, 0.0))


def _config_key(r: RunSummary) -> tuple:
    return (
        r.run_prefix,
        r.sem_encoder,
        r.label_mode,
        r.split_ratio,
        r.best_metric,
        r.log_interval,
        r.eval_interval,
        r.scheduler,
        r.plateau_patience,
        r.plateau_factor,
        r.lambda_contrastive,
        r.lambda_vae,
        r.temp,
        r.lr,
        r.weight_decay,
        r.freeze_steps,
        r.full_steps,
    )


def _stable_value(values: set[str | None]) -> str | None:
    values = {v for v in values if v}
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def aggregate_rows(rows: list[RunSummary]) -> list[AggregatedSummary]:
    grouped: dict[tuple, list[RunSummary]] = {}
    for r in rows:
        grouped.setdefault(_config_key(r), []).append(r)

    aggregates: list[AggregatedSummary] = []
    for key, members in grouped.items():
        (
            run_prefix_key,
            sem_encoder,
            label_mode,
            split_ratio,
            best_metric,
            log_interval,
            eval_interval,
            scheduler,
            plateau_patience,
            plateau_factor,
            lambda_contrastive,
            lambda_vae,
            temp,
            lr,
            weight_decay,
            freeze_steps,
            full_steps,
        ) = key

        best_step_mean, best_step_std = _mean_std([_safe_float(r.best_step) for r in members])
        avg_r1_mean, avg_r1_std = _mean_std([r.avg_r1 for r in members])
        m2t_mean, m2t_std = _mean_std([r.m2t_r1 for r in members])
        t2m_mean, t2m_std = _mean_std([r.t2m_r1 for r in members])
        sil_mean, sil_std = _mean_std([r.silhouette for r in members])
        mpjpe_mean, mpjpe_std = _mean_std([r.mpjpe for r in members])

        run_prefix = _stable_value({run_prefix_key} | {r.run_prefix for r in members})
        git_commit = _stable_value({r.git_commit for r in members})

        config_id = "|".join(
            [
                str(sem_encoder or ""),
                str(label_mode or ""),
                f"lc={lambda_contrastive}",
                f"lv={lambda_vae}",
                f"temp={temp}",
                f"lr={lr}",
                f"wd={weight_decay}",
                f"f={freeze_steps}",
                f"full={full_steps}",
            ]
        )

        aggregates.append(
            AggregatedSummary(
                config_id=config_id,
                n_runs=len(members),
                run_prefix=run_prefix,
                git_commit=git_commit,
                sem_encoder=sem_encoder,
                label_mode=label_mode,
                split_ratio=_safe_float(split_ratio),
                best_metric=best_metric,
                log_interval=_safe_int(log_interval),
                eval_interval=_safe_int(eval_interval),
                scheduler=scheduler,
                plateau_patience=_safe_int(plateau_patience),
                plateau_factor=_safe_float(plateau_factor),
                lambda_contrastive=_safe_float(lambda_contrastive),
                lambda_vae=_safe_float(lambda_vae),
                temp=_safe_float(temp),
                lr=_safe_float(lr),
                weight_decay=_safe_float(weight_decay),
                freeze_steps=_safe_int(freeze_steps),
                full_steps=_safe_int(full_steps),
                best_step_mean=best_step_mean,
                best_step_std=best_step_std,
                avg_r1_mean=avg_r1_mean,
                avg_r1_std=avg_r1_std,
                m2t_r1_mean=m2t_mean,
                m2t_r1_std=m2t_std,
                t2m_r1_mean=t2m_mean,
                t2m_r1_std=t2m_std,
                silhouette_mean=sil_mean,
                silhouette_std=sil_std,
                mpjpe_mean=mpjpe_mean,
                mpjpe_std=mpjpe_std,
            )
        )

    aggregates.sort(key=lambda r: (r.avg_r1_mean or float("-inf")), reverse=True)
    return aggregates


def write_csv(rows: list[RunSummary], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_name",
                "run_prefix",
                "git_commit",
                "seed",
                "sem_encoder",
                "label_mode",
                "best_metric",
                "log_interval",
                "eval_interval",
                "best_step",
                "avg_r@1",
                "m2t_r@1",
                "t2m_r@1",
                "silhouette",
                "mpjpe",
                "lambda_contrastive",
                "lambda_vae",
                "temp",
                "lr",
                "weight_decay",
                "freeze_steps",
                "full_steps",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.run_name,
                    r.run_prefix or "",
                    r.git_commit or "",
                    r.seed or "",
                    r.sem_encoder or "",
                    r.label_mode or "",
                    r.best_metric or "",
                    r.log_interval or "",
                    r.eval_interval or "",
                    r.best_step or "",
                    _fmt(r.avg_r1),
                    _fmt(r.m2t_r1),
                    _fmt(r.t2m_r1),
                    _fmt(r.silhouette),
                    _fmt(r.mpjpe),
                    _fmt(r.lambda_contrastive),
                    _fmt(r.lambda_vae),
                    _fmt(r.temp),
                    _fmt(r.lr, digits=6),
                    _fmt(r.weight_decay, digits=6),
                    r.freeze_steps or "",
                    r.full_steps or "",
                ]
            )


def write_markdown(rows: list[RunSummary], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| run | seed | step | avg_r@1 | m2t_r@1 | t2m_r@1 | silhouette | mpjpe | "
        "lc | lv | temp |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for r in rows:
        lines.append(
            "| {run} | {seed} | {step} | {avg} | {m2t} | {t2m} | {sil} | {mpjpe} | {lc} | {lv} | {temp} |".format(
                run=r.run_name,
                seed=r.seed or "",
                step=r.best_step or "",
                avg=_fmt(r.avg_r1, digits=3),
                m2t=_fmt(r.m2t_r1, digits=3),
                t2m=_fmt(r.t2m_r1, digits=3),
                sil=_fmt(r.silhouette, digits=3),
                mpjpe=_fmt(r.mpjpe, digits=3),
                lc=_fmt(r.lambda_contrastive, digits=3),
                lv=_fmt(r.lambda_vae, digits=3),
                temp=_fmt(r.temp, digits=3),
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_agg_csv(rows: list[AggregatedSummary], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "config_id",
                "n_runs",
                "run_prefix",
                "git_commit",
                "sem_encoder",
                "label_mode",
                "best_metric",
                "log_interval",
                "eval_interval",
                "lambda_contrastive",
                "lambda_vae",
                "temp",
                "lr",
                "weight_decay",
                "freeze_steps",
                "full_steps",
                "best_step_mean",
                "best_step_std",
                "avg_r@1_mean",
                "avg_r@1_std",
                "m2t_r@1_mean",
                "m2t_r@1_std",
                "t2m_r@1_mean",
                "t2m_r@1_std",
                "silhouette_mean",
                "silhouette_std",
                "mpjpe_mean",
                "mpjpe_std",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.config_id,
                    r.n_runs,
                    r.run_prefix or "",
                    r.git_commit or "",
                    r.sem_encoder or "",
                    r.label_mode or "",
                    r.best_metric or "",
                    r.log_interval or "",
                    r.eval_interval or "",
                    _fmt(r.lambda_contrastive),
                    _fmt(r.lambda_vae),
                    _fmt(r.temp),
                    _fmt(r.lr, digits=6),
                    _fmt(r.weight_decay, digits=6),
                    r.freeze_steps or "",
                    r.full_steps or "",
                    _fmt(r.best_step_mean, digits=1),
                    _fmt(r.best_step_std, digits=1),
                    _fmt(r.avg_r1_mean),
                    _fmt(r.avg_r1_std),
                    _fmt(r.m2t_r1_mean),
                    _fmt(r.m2t_r1_std),
                    _fmt(r.t2m_r1_mean),
                    _fmt(r.t2m_r1_std),
                    _fmt(r.silhouette_mean),
                    _fmt(r.silhouette_std),
                    _fmt(r.mpjpe_mean),
                    _fmt(r.mpjpe_std),
                ]
            )


def write_agg_markdown(rows: list[AggregatedSummary], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| config | n | avg_r@1 (mean±std) | m2t_r@1 | t2m_r@1 | mpjpe | lc | lv | temp |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for r in rows:
        avg = _fmt(r.avg_r1_mean, digits=3)
        if r.avg_r1_std is not None:
            avg = f"{avg} ± {_fmt(r.avg_r1_std, digits=3)}"
        lines.append(
            "| {config} | {n} | {avg} | {m2t} | {t2m} | {mpjpe} | {lc} | {lv} | {temp} |".format(
                config=r.config_id,
                n=r.n_runs,
                avg=avg,
                m2t=_fmt(r.m2t_r1_mean, digits=3),
                t2m=_fmt(r.t2m_r1_mean, digits=3),
                mpjpe=_fmt(r.mpjpe_mean, digits=3),
                lc=_fmt(r.lambda_contrastive, digits=3),
                lv=_fmt(r.lambda_vae, digits=3),
                temp=_fmt(r.temp, digits=3),
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MotionCLIP runs into paper tables.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="hoyo_v1_1/joint_training_results",
        help="Directory containing per-run subdirectories.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="",
        help="Comma-separated run names to include. Overrides --prefix.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="20260127_prod_supcon_v1",
        help="Run-name prefix filter when --runs is not provided.",
    )
    parser.add_argument(
        "--require-substring",
        type=str,
        default="",
        help="Only include runs whose name contains this substring (e.g., '_full').",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="hoyo_v1_1/joint_training_results/comparisons",
        help="Output directory for comparison tables.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="production_supcon_v1",
        help="Base filename for outputs (without extension).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = _parse_runs_arg(args.runs) if args.runs else None
    run_dirs = _iter_run_dirs(results_dir, runs=runs, prefix=args.prefix)
    if args.require_substring:
        run_dirs = [d for d in run_dirs if args.require_substring in d.name]

    summaries: list[RunSummary] = []
    skipped: list[str] = []
    for run_dir in run_dirs:
        summary = summarize_run(run_dir)
        if summary is None:
            skipped.append(run_dir.name)
            continue
        summaries.append(summary)

    summaries.sort(key=lambda r: (r.avg_r1 or float("-inf")), reverse=True)
    aggregates = aggregate_rows(summaries)

    out_dir = Path(args.out_dir)
    out_csv = out_dir / f"{args.name}.csv"
    out_md = out_dir / f"{args.name}.md"
    out_agg_csv = out_dir / f"{args.name}_agg.csv"
    out_agg_md = out_dir / f"{args.name}_agg.md"

    write_csv(summaries, out_csv)
    write_markdown(summaries, out_md)
    write_agg_csv(aggregates, out_agg_csv)
    write_agg_markdown(aggregates, out_agg_md)

    print(f"summarized runs: {len(summaries)}")
    if skipped:
        print(f"skipped (no metrics): {', '.join(skipped)}")
    print(f"csv: {out_csv}")
    print(f"md:  {out_md}")
    print(f"csv(agg): {out_agg_csv}")
    print(f"md(agg):  {out_agg_md}")


if __name__ == "__main__":
    main()
