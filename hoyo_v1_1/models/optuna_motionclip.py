"""
Optuna-based hyperparameter optimization for MotionCLIP contrastive learning.

Strategy:
    1. Run 'freeze' stage for each trial (sem_proj/logit_scale training)
    2. Continue with 'full' stage from the same run directory
    3. Optimize full-stage hyperparameters and total steps

Usage:
    /home/jouta/venvs/motionclip/bin/python hoyo_v1_1/models/optuna_motionclip.py \
        --n-trials 20 \
        --freeze-steps 1000 \
        --min-steps 2000 --max-steps 6000 --step-interval 500
"""

import argparse
import subprocess
import sys
import json
import os
import shutil
import time
from pathlib import Path

import optuna
from optuna.exceptions import DuplicatedStudyError
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "hoyo_v1_1" / "models" / "train_motionclip_joint.py"
RESULTS_DIR = REPO_ROOT / "hoyo_v1_1" / "joint_training_results"


def setup_trial_from_freeze(freeze_run: str, trial_run: str) -> bool:
    """
    Copy freeze checkpoint to trial directory so that full stage can continue from it.

    Returns True if successful, False otherwise.
    """
    freeze_dir = RESULTS_DIR / freeze_run / "checkpoints"
    trial_dir = RESULTS_DIR / trial_run / "checkpoints"

    if not freeze_dir.exists():
        print(f"ERROR: Freeze checkpoint not found: {freeze_dir}")
        return False

    # Ensure a clean checkpoint directory to avoid loading stale full-model weights
    if trial_dir.exists():
        shutil.rmtree(trial_dir)

    # Create trial checkpoint directory
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Copy freeze checkpoints (sem_proj, logit_scale, encoder) as "best" files
    # These are the files that train_motionclip_joint.py looks for when stage=full
    files_to_copy = [
        ("sem_proj_joint_best.pth", "sem_proj_joint_best.pth"),
        ("logit_scale_joint_best.pt", "logit_scale_joint_best.pt"),
        ("motionclip_encoder_joint_best.pth", "motionclip_encoder_joint_best.pth"),
    ]

    for src_name, dst_name in files_to_copy:
        src = freeze_dir / src_name
        dst = trial_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
        else:
            # Try final version if best doesn't exist
            src_final = freeze_dir / src_name.replace("_best", "_final")
            if src_final.exists():
                shutil.copy2(src_final, dst)

    return True


def build_cmd(
    stage: str,
    steps: int,
    run_name: str,
    batch_size: int,
    seed: int,
    contrastive_mode: str,
    best_metric: str,
    lambda_contrastive: float,
    lambda_vae: float,
    temp: float,
    lr: float,
    lr_encoder: float,
    lr_decoder: float,
    weight_decay: float,
    wandb_args: list[str] | None = None,
):
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--stage",
        stage,
        "--sem-encoder",
        "sarashina",
        "--label-mode",
        "fine",
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--lr-encoder",
        str(lr_encoder),
        "--lr-decoder",
        str(lr_decoder),
        "--weight-decay",
        str(weight_decay),
        "--lambda-contrastive",
        str(lambda_contrastive),
        "--lambda-vae",
        str(lambda_vae),
        "--temp",
        str(temp),
        "--contrastive-mode",
        contrastive_mode,
        "--best-metric",
        best_metric,
        "--log-interval",
        "500",
        "--eval-interval",
        "500",
        "--seed",
        str(seed),
        "--run-name",
        run_name,
    ]
    if wandb_args:
        cmd.extend(wandb_args)
    return cmd


def run_stage(
    stage: str,
    cmd,
    log_file: Path,
    timeout: int,
    env: dict,
    monitor_metrics: bool = False,
    metrics_file: Path | None = None,
    trial: optuna.Trial | None = None,
    silhouette_weight: float | None = None,
) -> None:
    process = None
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )

        start_time = time.time()
        metrics_pos = 0
        last_reported_step = -1

        while True:
            ret = process.poll()
            if ret is not None:
                break

            if time.time() - start_time > timeout:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise TimeoutError(f"{stage} stage timed out after {timeout}s")

            if monitor_metrics and metrics_file and metrics_file.exists():
                with open(metrics_file, "r", encoding="utf-8") as mf:
                    mf.seek(metrics_pos)
                    while True:
                        line = mf.readline()
                        if not line:
                            break
                        metrics_pos = mf.tell()
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        step = int(record.get("step", 0))
                        if step <= last_reported_step:
                            continue

                        m2t_r1 = record.get("m2t", {}).get("R@1", 0.0)
                        sil = record.get("silhouette", None)
                        if isinstance(sil, (int, float)):
                            sil_val = float(sil)
                        else:
                            sil_val = -1.0
                        if silhouette_weight is not None:
                            score = m2t_r1 + silhouette_weight * sil_val
                        else:
                            score = m2t_r1
                        print(
                            f"[{stage}] step {step}: "
                            f"M2T_R@1={m2t_r1:.4f}, Sil={sil_val:.4f}, Obj={score:.4f}"
                        )
                        if trial is not None:
                            trial.report(score, step=step)
                            last_reported_step = step
                            if trial.should_prune():
                                process.terminate()
                                try:
                                    process.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                raise optuna.TrialPruned()

            time.sleep(5)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"{stage} stage failed (see log: {log_file})")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def objective(trial: optuna.Trial, args) -> float:
    """
    Objective function for Optuna.
    Returns the M→T R@1 (to be maximized).
    """
    # Hyperparameters to search (full stage only)
    # Fixed values (per user request)
    lambda_contrastive = args.lambda_contrastive
    lambda_vae = args.lambda_vae

    # Search ranges
    if args.narrow_search:
        # Narrow ranges around typical good values
        lr = trial.suggest_float("lr", 3e-5, 1e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-3, log=True)
        temp = trial.suggest_float("temp", 0.05, 0.1, log=True)
    else:
        lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        temp = trial.suggest_float("temp", 0.02, 0.2, log=True)

    # Tie encoder/decoder LR to main LR for simplicity
    lr_encoder = lr
    lr_decoder = lr
    contrastive_mode = args.contrastive_mode

    # Fixed parameters
    batch_size = 32
    seed = 42

    run_name = f"{args.study_name}_trial_{trial.number}"
    out_dir = RESULTS_DIR / run_name
    log_file = out_dir / "optuna_train.log"
    metrics_file = out_dir / "retrieval_metrics.jsonl"

    # Determine full-stage steps (search or fixed)
    if args.min_steps is not None or args.max_steps is not None:
        min_steps = args.min_steps if args.min_steps is not None else args.steps
        max_steps = args.max_steps if args.max_steps is not None else args.steps
        if min_steps > max_steps:
            min_steps, max_steps = max_steps, min_steps
        full_steps = trial.suggest_int("steps", min_steps, max_steps, step=args.step_interval)
    else:
        full_steps = args.steps

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    if metrics_file.exists():
        metrics_file.unlink()

    # Build wandb args (optional)
    wandb_args = []
    if args.wandb:
        wandb_args.append("--wandb")
        if args.wandb_project:
            wandb_args.extend(["--wandb-project", args.wandb_project])
        if args.wandb_entity:
            wandb_args.extend(["--wandb-entity", args.wandb_entity])
        if args.wandb_group:
            wandb_args.extend(["--wandb-group", args.wandb_group])
    if not wandb_args:
        wandb_args = None

    # Build command per stage
    freeze_cmd = build_cmd(
        stage="freeze",
        steps=args.freeze_steps,
        run_name=run_name,
        batch_size=batch_size,
        seed=seed,
        contrastive_mode=contrastive_mode,
        best_metric=args.best_metric,
        lambda_contrastive=lambda_contrastive,
        lambda_vae=lambda_vae,
        temp=temp,
        lr=lr,
        lr_encoder=lr_encoder,
        lr_decoder=lr_decoder,
        weight_decay=weight_decay,
        wandb_args=wandb_args,
    )
    full_cmd = build_cmd(
        stage="full",
        steps=full_steps,
        run_name=run_name,
        batch_size=batch_size,
        seed=seed,
        contrastive_mode=contrastive_mode,
        best_metric=args.best_metric,
        lambda_contrastive=lambda_contrastive,
        lambda_vae=lambda_vae,
        temp=temp,
        lr=lr,
        lr_encoder=lr_encoder,
        lr_decoder=lr_decoder,
        weight_decay=weight_decay,
        wandb_args=wandb_args,
    )

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(
        f"Params: lambda_c={lambda_contrastive:.3f}, lambda_v={lambda_vae:.3f}, "
        f"lr={lr:.2e}, wd={weight_decay:.2e}, temp={temp:.4f}, "
        f"mode={contrastive_mode}, steps(full)={full_steps}"
    )
    print(f"{'='*60}\n")

    # Set GPU for this trial
    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Run training with output to file (streamed)
    try:
        if args.run_freeze:
            if args.freeze_run:
                print("Note: --freeze-run is ignored when --run-freeze is enabled.")
            run_stage(
                stage="freeze",
                cmd=freeze_cmd,
                log_file=out_dir / "optuna_train_freeze.log",
                timeout=args.timeout,
                env=env,
                monitor_metrics=False,
            )
        else:
            # Setup: copy freeze checkpoint to trial directory
            if args.freeze_run:
                print(f"Setting up trial from freeze checkpoint: {args.freeze_run}")
                if not setup_trial_from_freeze(args.freeze_run, run_name):
                    print("Failed to setup from freeze checkpoint")
                    return 0.0

        if metrics_file.exists():
            metrics_file.unlink()

        run_stage(
            stage="full",
            cmd=full_cmd,
            log_file=out_dir / "optuna_train_full.log",
            timeout=args.timeout,
            env=env,
            monitor_metrics=True,
            metrics_file=metrics_file,
            trial=trial,
            silhouette_weight=args.silhouette_weight,
        )

        if not metrics_file.exists():
            print(f"Metrics file not found: {metrics_file}")
            return 0.0

        with open(metrics_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                return 0.0
            last_metrics = json.loads(lines[-1])

        # Use Avg(M→T R@1, T→M R@1) + silhouette as the objective (higher is better)
        m2t_r1 = last_metrics.get("m2t", {}).get("R@1", 0.0)
        m2t_r3 = last_metrics.get("m2t", {}).get("R@3", 0.0)
        t2m_r1 = last_metrics.get("t2m", {}).get("R@1", 0.0)
        silhouette = last_metrics.get("silhouette", -1.0)
        if isinstance(silhouette, (int, float)):
            silhouette_val = float(silhouette)
        else:
            silhouette_val = -1.0
        avg_r1 = 0.5 * (m2t_r1 + t2m_r1)
        objective_value = avg_r1 + args.silhouette_weight * silhouette_val

        print(
            f"\nTrial {trial.number} Result: M2T R@1={m2t_r1:.3f}, "
            f"T2M R@1={t2m_r1:.3f}, AvgR@1={avg_r1:.3f}, "
            f"R@3={m2t_r3:.3f}, Sil={silhouette_val:.3f}, "
            f"Obj={objective_value:.3f}"
        )

        return objective_value

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} error: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--study-name", type=str, default="motionclip_hpo_v2", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per trial (seconds)")
    parser.add_argument("--steps", type=int, default=3000, help="Default full-stage steps per trial")
    parser.add_argument("--min-steps", type=int, default=None, help="Min full-stage steps for search")
    parser.add_argument("--max-steps", type=int, default=None, help="Max full-stage steps for search")
    parser.add_argument("--step-interval", type=int, default=500, help="Step interval for search")
    parser.add_argument("--freeze-steps", type=int, default=1000, help="Freeze-stage steps per trial")
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="Skip freeze stage and use --freeze-run to initialize full stage",
    )
    parser.add_argument(
        "--freeze-run",
        type=str,
        default=None,
        help="Name of freeze run to use as initialization (only when --run-freeze is disabled)",
    )
    parser.add_argument(
        "--contrastive-mode",
        type=str,
        choices=["supcon", "clip_ce"],
        default="supcon",
        help="Contrastive loss type (fixed across trials).",
    )
    parser.add_argument(
        "--best-metric",
        type=str,
        choices=["acc@1", "m2t_r@1", "t2m_r@1", "avg_r@1"],
        default="avg_r@1",
        help="Metric used to select the best checkpoint during training.",
    )
    parser.add_argument(
        "--lambda-contrastive",
        type=float,
        default=0.3,
        help="Fixed contrastive loss weight (lc).",
    )
    parser.add_argument(
        "--lambda-vae",
        type=float,
        default=1.0,
        help="Fixed VAE loss weight.",
    )
    parser.add_argument(
        "--silhouette-weight",
        type=float,
        default=0.1,
        help="Weight for silhouette in objective: objective = m2t_r1 + w * silhouette",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (user or team)")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group name")
    parser.add_argument(
        "--narrow-search",
        action="store_true",
        help="Use narrow search ranges (lr=3e-5-1e-4, wd=1e-4-5e-3, temp=0.05-0.1)",
    )
    args = parser.parse_args()
    args.run_freeze = not args.no_freeze

    # Verify freeze checkpoint exists (only if not running freeze stage)
    if not args.run_freeze and args.freeze_run:
        freeze_ckpt = RESULTS_DIR / args.freeze_run / "checkpoints"
        if not freeze_ckpt.exists():
            print(f"ERROR: Freeze checkpoint not found: {freeze_ckpt}")
            print("Run freeze stage first, or specify --freeze-run with a valid run name")
            sys.exit(1)
        print(f"Using freeze checkpoint: {args.freeze_run}")

    # Create study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1000)

    if args.storage:
        storage = args.storage
    else:
        db_path = REPO_ROOT / "hoyo_v1_1" / "optuna_studies.db"
        storage = f"sqlite:///{db_path}"

    try:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=args.resume,
        )
    except DuplicatedStudyError:
        print(
            f"Study '{args.study_name}' already exists; loading existing study. "
            "Use --study-name to create a new study or --resume to be explicit."
        )
        study = optuna.load_study(study_name=args.study_name, storage=storage)

    print(f"\n{'='*60}")
    print(f"Optuna HPO for MotionCLIP")
    print(f"{'='*60}")
    print(f"Study: {args.study_name}")
    if args.run_freeze:
        print(f"Freeze stage: enabled ({args.freeze_steps} steps)")
    else:
        print(f"Freeze init: {args.freeze_run}")
    print(f"N trials: {args.n_trials}")
    if args.min_steps is not None or args.max_steps is not None:
        min_steps = args.min_steps if args.min_steps is not None else args.steps
        max_steps = args.max_steps if args.max_steps is not None else args.steps
        print(f"Full steps search: {min_steps}..{max_steps} (interval={args.step_interval})")
    else:
        print(f"Full steps/trial: {args.steps}")
    print(f"Timeout: {args.timeout}s")
    print(f"{'='*60}\n")

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    # Results
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best Avg R@1: {study.best_value:.4f}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params
    best_params_path = RESULTS_DIR / "optuna_best_params.json"
    with open(best_params_path, "w") as f:
        json.dump({
            "study_name": args.study_name,
            "freeze_run": args.freeze_run,
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
        }, f, indent=2)
    print(f"\nSaved to: {best_params_path}")

    # Top 5
    print("\nTop 5 trials:")
    df = study.trials_dataframe()
    if len(df) > 0:
        df = df.sort_values("value", ascending=False).head(5)
        cols = ["number", "value"]
        param_cols = [c for c in df.columns if c.startswith("params_")][:4]
        print(df[cols + param_cols].to_string())


if __name__ == "__main__":
    main()
