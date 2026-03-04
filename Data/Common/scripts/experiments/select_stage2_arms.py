#!/usr/bin/env python3
"""Select Stage2 arms from Stage1 eval_motion results with fixed gating rules."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class ArmMetrics:
    arm: str
    source_json: str
    mean_cos_centroid: float | None
    margin_offdiag_mean: float | None
    velocity_mae: float | None
    mean_joint_error: float | None
    top1_match_count: int
    top1_total_count: int


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_json(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Path is neither file nor dir: {path}")

    direct = sorted(path.glob("eval_motion_*.json"), key=lambda p: p.stat().st_mtime)
    if direct:
        return direct[-1]
    recursive = sorted(path.rglob("eval_motion_*.json"), key=lambda p: p.stat().st_mtime)
    if recursive:
        return recursive[-1]
    raise FileNotFoundError(f"No eval_motion_*.json found in: {path}")


def _parse_arm_input(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Expected arm=path format: {text}")
    arm, raw_path = text.split("=", 1)
    arm = arm.strip()
    if not arm:
        raise ValueError(f"Arm name is empty: {text}")
    return arm, Path(raw_path.strip())


def _compute_metrics(arm: str, json_path: Path, target_speed: float) -> ArmMetrics:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"'results' is not a list in {json_path}")

    cos_vals: list[float] = []
    vel_mae_vals: list[float] = []
    joint_vals: list[float] = []
    margin_vals: list[float] = []
    top1_ok = 0
    top1_total = 0

    for row in results:
        if not isinstance(row, dict):
            continue

        cos = _to_float(row.get("mean_cos_centroid"))
        if cos is not None:
            cos_vals.append(cos)

        mean_vx = _to_float(row.get("mean_velocity_x"))
        if mean_vx is not None:
            vel_mae_vals.append(abs(mean_vx - target_speed))

        joint = _to_float(row.get("mean_joint_error"))
        if joint is not None:
            joint_vals.append(joint)

        row_label = row.get("onomatopoeia")
        sim = row.get("hoyo_similarity_centered_mean")
        if not isinstance(row_label, str) or not isinstance(sim, dict) or not sim:
            continue

        numeric_items: list[tuple[str, float]] = []
        for col_label, value in sim.items():
            value_f = _to_float(value)
            if value_f is not None:
                numeric_items.append((str(col_label), value_f))
        if not numeric_items:
            continue

        top1_total += 1
        top1_label = max(numeric_items, key=lambda x: x[1])[0]
        if top1_label == row_label:
            top1_ok += 1

        diag = sim.get(row_label)
        diag_f = _to_float(diag)
        if diag_f is None:
            continue
        for col_label, value_f in numeric_items:
            if col_label == row_label:
                continue
            margin_vals.append(diag_f - value_f)

    return ArmMetrics(
        arm=arm,
        source_json=str(json_path),
        mean_cos_centroid=_mean(cos_vals),
        margin_offdiag_mean=_mean(margin_vals),
        velocity_mae=_mean(vel_mae_vals),
        mean_joint_error=_mean(joint_vals),
        top1_match_count=top1_ok,
        top1_total_count=top1_total,
    )


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value - baseline)


def _arm_sort_key_gate(item: dict[str, Any]) -> tuple[float, float, float, float]:
    # Higher is better for first two; lower is better for error terms.
    return (
        float(item.get("cos_delta") or -1.0e9),
        float(item.get("offdiag_delta") or -1.0e9),
        -float(item.get("velocity_mae") or 1.0e9),
        -float(item.get("mean_joint_error") or 1.0e9),
    )


def _arm_sort_key_score(item: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(item.get("fallback_score") or -1.0e9),
        float(item.get("cos_delta") or -1.0e9),
        float(item.get("offdiag_delta") or -1.0e9),
        float(item.get("top1_match_count") or -1.0e9),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Stage2 arms from Stage1 eval results.")
    parser.add_argument(
        "--arm-result",
        action="append",
        required=True,
        help="arm=path_to_eval_json_or_dir. Repeat for each arm.",
    )
    parser.add_argument("--baseline", default="armA", help="Baseline arm name.")
    parser.add_argument("--target-speed", type=float, default=0.3, help="Target x velocity for MAE.")
    parser.add_argument("--cos-threshold", type=float, default=0.05)
    parser.add_argument("--offdiag-threshold", type=float, default=0.01)
    parser.add_argument("--vel-delta-max", type=float, default=0.03)
    parser.add_argument("--joint-delta-max", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, default=None, help="Write detailed selection JSON.")
    parser.add_argument("--output-selected", type=Path, default=None, help="Write selected arms CSV.")
    args = parser.parse_args()

    arm_paths: dict[str, Path] = {}
    for item in args.arm_result:
        arm, path = _parse_arm_input(item)
        arm_paths[arm] = _resolve_json(path)

    if args.baseline not in arm_paths:
        raise ValueError(f"Baseline arm '{args.baseline}' is not in --arm-result inputs.")

    metrics_map: dict[str, ArmMetrics] = {
        arm: _compute_metrics(arm, json_path, target_speed=args.target_speed)
        for arm, json_path in arm_paths.items()
    }
    baseline = metrics_map[args.baseline]

    rows: list[dict[str, Any]] = []
    for arm, m in metrics_map.items():
        cos_delta = _delta(m.mean_cos_centroid, baseline.mean_cos_centroid)
        offdiag_delta = _delta(m.margin_offdiag_mean, baseline.margin_offdiag_mean)
        vel_delta = _delta(m.velocity_mae, baseline.velocity_mae)
        joint_delta = _delta(m.mean_joint_error, baseline.mean_joint_error)
        gate_pass = (
            cos_delta is not None
            and offdiag_delta is not None
            and vel_delta is not None
            and joint_delta is not None
            and cos_delta >= args.cos_threshold
            and offdiag_delta >= args.offdiag_threshold
            and vel_delta <= args.vel_delta_max
            and joint_delta <= args.joint_delta_max
        )

        penalty_vel = 0.0
        if vel_delta is not None:
            penalty_vel = max(0.0, vel_delta - args.vel_delta_max)
        penalty_joint = 0.0
        if joint_delta is not None:
            penalty_joint = max(0.0, joint_delta - args.joint_delta_max)

        fallback_score = (
            (cos_delta or 0.0)
            + (offdiag_delta or 0.0)
            - penalty_vel
            - 0.5 * penalty_joint
        )

        row = asdict(m)
        row.update(
            {
                "cos_delta": cos_delta,
                "offdiag_delta": offdiag_delta,
                "velocity_mae_delta": vel_delta,
                "mean_joint_error_delta": joint_delta,
                "gate_pass": gate_pass,
                "fallback_score": float(fallback_score),
            }
        )
        rows.append(row)

    non_baseline = [r for r in rows if r["arm"] != args.baseline]
    passed = [r for r in non_baseline if r["gate_pass"]]

    if len(passed) >= 2:
        selected_rows = sorted(passed, key=_arm_sort_key_gate, reverse=True)[:2]
        selection_rule = "gate_pass"
    else:
        selected_rows = sorted(rows, key=_arm_sort_key_score, reverse=True)[:2]
        selection_rule = "fallback_score"

    selected_arms = [r["arm"] for r in selected_rows]

    payload = {
        "baseline_arm": args.baseline,
        "target_speed": args.target_speed,
        "thresholds": {
            "cos_delta_min": args.cos_threshold,
            "offdiag_delta_min": args.offdiag_threshold,
            "velocity_mae_delta_max": args.vel_delta_max,
            "mean_joint_error_delta_max": args.joint_delta_max,
        },
        "selection_rule": selection_rule,
        "selected_arms": selected_arms,
        "arms": rows,
    }

    print("=== Stage2 Arm Selection ===")
    print(f"baseline: {args.baseline}")
    for row in sorted(rows, key=lambda r: r["arm"]):
        print(
            f"{row['arm']}: gate={row['gate_pass']} "
            f"cos={row['mean_cos_centroid']} (d={row['cos_delta']}), "
            f"offdiag={row['margin_offdiag_mean']} (d={row['offdiag_delta']}), "
            f"vel_mae={row['velocity_mae']} (d={row['velocity_mae_delta']}), "
            f"joint={row['mean_joint_error']} (d={row['mean_joint_error_delta']}), "
            f"top1={row['top1_match_count']}/{row['top1_total_count']}, "
            f"score={row['fallback_score']}"
        )
    print(f"selection_rule: {selection_rule}")
    print(f"selected_arms: {','.join(selected_arms)}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved json: {args.output_json}")

    if args.output_selected is not None:
        args.output_selected.parent.mkdir(parents=True, exist_ok=True)
        args.output_selected.write_text(",".join(selected_arms) + "\n", encoding="utf-8")
        print(f"saved selected arms: {args.output_selected}")


if __name__ == "__main__":
    main()

