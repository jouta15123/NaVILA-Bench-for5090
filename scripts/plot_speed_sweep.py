#!/usr/bin/env python3
# Copyright (c) 2022-2024
# SPDX-License-Identifier: BSD-3-Clause
"""
Plot results from h1_speed_sweep CSV:
- mode,cmd_vx,cmd_omega,meas_vx,meas_omega
"""

import argparse
import csv
import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import math


def read_results(csv_path: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
    vx_points: List[Tuple[float, float]] = []      # (cmd_vx, meas_vx)
    om_points: List[Tuple[float, float]] = []      # (cmd_omega, meas_omega)
    vx_errs: List[Tuple[float, float]] = []        # (cmd_vx, err_vx)
    om_errs: List[Tuple[float, float]] = []        # (cmd_omega, err_omega)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = (row.get("mode") or "").strip().lower()
            try:
                cmd_vx = float(row.get("cmd_vx", "nan"))
                cmd_om = float(row.get("cmd_omega", "nan"))
                meas_vx = float(row.get("meas_vx", "nan"))
                meas_om = float(row.get("meas_omega", "nan"))
                err_vx = row.get("err_vx", "")
                err_om = row.get("err_omega", "")
                err_vx = float(err_vx) if err_vx not in ("", None) else abs((cmd_vx if mode == "vx" else 0.0) - meas_vx)
                err_om = float(err_om) if err_om not in ("", None) else abs((cmd_om if mode == "om" else 0.0) - meas_om)
            except Exception:
                continue
            if mode == "vx":
                vx_points.append((cmd_vx, meas_vx))
                vx_errs.append((cmd_vx, err_vx))
            elif mode == "om":
                om_points.append((cmd_om, meas_om))
                om_errs.append((cmd_om, err_om))
    # sort by command value for nicer plotting
    vx_points.sort(key=lambda p: p[0])
    om_points.sort(key=lambda p: p[0])
    vx_errs.sort(key=lambda p: p[0])
    om_errs.sort(key=lambda p: p[0])
    return vx_points, om_points, vx_errs, om_errs


def compute_errors(pairs: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    errors = [abs(cmd - meas) for cmd, meas in pairs]
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    mean_err = sum(errors) / len(errors)
    max_err = max(errors)
    return rmse, mean_err, max_err


def plot_tracking(vx_points, om_points, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linear velocity
    ax = axes[0]
    if vx_points:
        xs = [p[0] for p in vx_points]
        ys = [p[1] for p in vx_points]
        ax.scatter(xs, ys, color="#1f77b4", label="Measured")
        vmin, vmax = min(xs + ys), max(xs + ys)
        vmin = min(0.0, vmin)
        vmax = max(1.0, vmax)
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.0, label="Ideal y=x")
        rmse, mean_err, max_err = compute_errors(vx_points)
        ax.set_title(f"Linear (vx) tracking\nRMSE={rmse:.3f} m/s, Mean={mean_err:.3f}, Max={max_err:.3f}")
        ax.set_xlabel("Commanded vx [m/s]")
        ax.set_ylabel("Measured vx [m/s]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    else:
        ax.set_title("Linear (vx) tracking\n(no data)")
        ax.axis("off")

    # Angular velocity
    ax = axes[1]
    if om_points:
        xs = [p[0] for p in om_points]
        ys = [p[1] for p in om_points]
        ax.scatter(xs, ys, color="#d62728", label="Measured")
        vmin, vmax = min(xs + ys), max(xs + ys)
        if math.isfinite(vmin) and math.isfinite(vmax):
            ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.0, label="Ideal y=x")
        rmse, mean_err, max_err = compute_errors(om_points)
        ax.set_title(f"Angular (omega) tracking\nRMSE={rmse:.3f} rad/s, Mean={mean_err:.3f}, Max={max_err:.3f}")
        ax.set_xlabel("Commanded omega [rad/s]")
        ax.set_ylabel("Measured omega [rad/s]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    else:
        ax.set_title("Angular (omega) tracking\n(no data)")
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Wrote plot to: {out_path}")


def plot_errors(vx_errs, om_errs, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # vx absolute error
    ax = axes[0]
    if vx_errs:
        xs = [p[0] for p in vx_errs]
        es = [p[1] for p in vx_errs]
        ax.plot(xs, es, "o-", color="#1f77b4")
        rmse, mean_err, max_err = compute_errors([(x, x - e) for x, e in zip(xs, es)])  # placeholder, not used
        ax.set_title("Linear (vx) absolute error")
        ax.set_xlabel("Commanded vx [m/s]")
        ax.set_ylabel("|vx_error| [m/s]")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title("Linear (vx) absolute error\n(no data)")
        ax.axis("off")

    # omega absolute error
    ax = axes[1]
    if om_errs:
        xs = [p[0] for p in om_errs]
        es = [p[1] for p in om_errs]
        ax.plot(xs, es, "o-", color="#d62728")
        ax.set_title("Angular (omega) absolute error")
        ax.set_xlabel("Commanded omega [rad/s]")
        ax.set_ylabel("|omega_error| [rad/s]")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title("Angular (omega) absolute error\n(no data)")
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Wrote error plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot H1 speed sweep CSV results.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file (mode,cmd_vx,cmd_omega,meas_vx,meas_omega).")
    parser.add_argument("--out", type=str, default="", help="Output image path (PNG). Defaults to <csv_dir>/plot.png")
    parser.add_argument("--out_errors", type=str, default="", help="Optional error plot path (PNG). Defaults to <csv_dir>/plot_errors.png")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    out_path = args.out.strip() or os.path.join(os.path.dirname(csv_path), "plot.png")

    vx_points, om_points, vx_errs, om_errs = read_results(csv_path)
    plot_tracking(vx_points, om_points, out_path)
    # error plot
    out_err = args.out_errors.strip() or os.path.join(os.path.dirname(csv_path), "plot_errors.png")
    plot_errors(vx_errs, om_errs, out_err)


if __name__ == "__main__":
    main()


