import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from data import load_data
from evaluate import evaluate
from integrator import rollout
from model import (ACTIVATIONS, DEPTHS, WIDTHS, AccelerationNet,
                   architecture_configs, make_scales)


DEFAULT_SETTINGS = {
    "optimizer": "adam",
    "lr": 1e-3,
    "steps_per_stage": 300,
    "rollout_lengths": (16, 32, 64),
    "batch_size": 32,
    "eval_every": 100,
    "grad_clip": 1.0,
    "seed": 42,
}

OPTIMIZERS = ("adam", "sgd")

# Compare two optimizers and two learning rates per optimizer. Slower
# updates get more steps; longer final rollouts check accumulated error.
TUNING_PLANS = (
    {"optimizer": "adam", "lr": 1e-3, "steps_per_stage": 300,
     "rollout_lengths": (16, 32, 64)},
    {"optimizer": "adam", "lr": 3e-4, "steps_per_stage": 500,
     "rollout_lengths": (16, 32, 128)},
    {"optimizer": "sgd", "lr": 1e-2, "steps_per_stage": 300,
     "rollout_lengths": (16, 32, 64)},
    {"optimizer": "sgd", "lr": 3e-3, "steps_per_stage": 500,
     "rollout_lengths": (16, 32, 128)},
)


def train_model(data, architecture, settings=None, output_dir="runs"):
    """Train one model, save its best checkpoint, and return its summary.

    Gradients use ONLY u_train and y_train. Test RMSE selects checkpoints,
    as this homework requests; it is never part of the gradient loss.
    Lengths count samples, so a length L requires L-1 RK4 steps.
    Saved checkpoints include architecture, scales, settings, and metrics.
    If no finite evaluated checkpoint exists, best_step is None and no
    best.pt is written. Use a separate output_dir for each trial.
    """
    unknown = set(settings or {}) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    lengths = tuple(settings["rollout_lengths"])
    n_train = len(data["y_train"])
    for name in ("steps_per_stage", "batch_size", "eval_every"):
        if not isinstance(settings[name], int) or settings[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    for name in ("lr", "grad_clip"):
        if not math.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    if not lengths or any(not isinstance(length, int) or not 2 <= length <= n_train
                          for length in lengths):
        raise ValueError("Each rollout length must be between 2 and the training size.")
    if settings["optimizer"] not in OPTIMIZERS:
        raise ValueError("Choose optimizer 'adam' or 'sgd'.")

    torch.manual_seed(settings["seed"])
    # A separate generator gives architectures the same sequence of windows
    # when training settings match, despite different weight initialization.
    generator = torch.Generator().manual_seed(settings["seed"] + 1)
    scales = make_scales(data)
    model = AccelerationNet(scales, **architecture)
    y = torch.as_tensor(data["y_train"], dtype=torch.float64)
    u = torch.as_tensor(data["u_train"], dtype=torch.float64)
    dt = float(data["dt"])
    # Central differences inside training; one-sided differences at its edges.
    # Velocity initializes a window, but is not a supervised target.
    velocity = torch.as_tensor(np.gradient(data["y_train"], dt), dtype=torch.float64)
    if settings["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=settings["lr"], momentum=0.9)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    if checkpoint_path.exists():
        raise FileExistsError("Use a new trial directory to preserve its existing best.pt.")
    best_metrics, best_step = None, None
    step, status, last_loss = 0, "completed", None

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as log:
        writer = csv.DictWriter(log, fieldnames=(
            "step", "rollout_length", "batch_loss", "train_rmse", "test_rmse", "finite"))
        writer.writeheader()

        def record(length):
            nonlocal best_metrics, best_step
            metrics = evaluate(model, data)
            row = {"step": step, "rollout_length": length, "batch_loss": last_loss,
                   **{key: metrics[key] for key in ("train_rmse", "test_rmse", "finite")}}
            writer.writerow(row)
            log.flush()
            print(f"step {step:5d} | L={length:3d} | "
                  f"train RMSE {metrics['train_rmse']:.6f} V | "
                  f"test RMSE {metrics['test_rmse']:.6f} V", flush=True)
            if math.isfinite(metrics["test_rmse"]) and (
                    best_metrics is None or metrics["test_rmse"] < best_metrics["test_rmse"]):
                best_metrics = {key: metrics[key] for key in (
                    "train_rmse", "test_rmse", "zero_test_rmse")}
                best_step = step
                # Replace only after the new checkpoint is completely written.
                temporary_path = checkpoint_path.with_suffix(".tmp")
                torch.save({"model_state": model.state_dict(), "architecture": dict(architecture),
                            "scales": scales, "settings": settings, "best_step": best_step,
                            "metrics": best_metrics, "optimizer_config": optimizer.defaults,
                            "data_info": {"dt": dt, "train_samples": n_train,
                                          "total_samples": len(data["y"]),
                                          "u_mean": data.get("u_mean"),
                                          "y_mean": data.get("y_mean")}}, temporary_path)
                temporary_path.replace(checkpoint_path)

        for length in lengths:
            offsets = torch.arange(length)[:, None]
            for stage_step in range(1, settings["steps_per_stage"] + 1):
                starts = torch.randint(n_train - length + 1, (settings["batch_size"],),
                                       generator=generator)
                indices = offsets + starts[None, :]
                target = y[indices]
                initial_state = torch.stack((target[0], velocity[starts]), dim=-1)
                optimizer.zero_grad(set_to_none=True)
                states = rollout(model, initial_state, u[indices], dt)
                # Exclude the supplied initial output, whose error is always zero.
                loss = ((states[1:, :, 0] - target[1:]) / scales["y"]).square().mean()
                if not torch.isfinite(loss):
                    status = "stopped_nonfinite_loss"
                    break
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=settings["grad_clip"])
                if not torch.isfinite(grad_norm):
                    status = "stopped_nonfinite_gradient"
                    break
                optimizer.step()
                step += 1
                last_loss = loss.item()
                if step % settings["eval_every"] == 0 or stage_step == settings["steps_per_stage"]:
                    record(length)
            if status != "completed":
                print(status, "Keeping any previously saved finite checkpoint.", flush=True)
                break

    if best_metrics is None:
        status += ";no_finite_checkpoint"
        best_metrics = {"train_rmse": float("inf"), "test_rmse": float("inf"),
                        "zero_test_rmse": data["zero_test_rmse"]}
    summary = {**architecture, **settings, "rollout_lengths": "/".join(map(str, lengths)),
               "steps_completed": step, "best_step": best_step, "status": status,
               "parameters": sum(p.numel() for p in model.parameters()), **best_metrics}
    return summary


def run_experiments(data, architectures, plans, output_dir):
    """Run the requested trials and save their table and overall best model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "results.csv").exists():
        raise FileExistsError("Use a new experiment directory to preserve results.csv.")
    rows, best_score, writer = [], float("inf"), None
    total = len(architectures) * len(plans)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as table:
        for architecture in architectures:
            name = f"{architecture['activation']}_d{architecture['depth']}_w{architecture['width']}"
            for trial, plan in enumerate(plans, start=1):
                trial_dir = output_dir / name / f"trial_{trial:02d}"
                print(f"\nTrial {len(rows)+1}/{total}: {name}, {plan}", flush=True)
                summary = train_model(data, architecture, plan, trial_dir)
                row = {**summary, "checkpoint": str(trial_dir / "best.pt")
                       if summary["best_step"] is not None else ""}
                if writer is None:
                    writer = csv.DictWriter(table, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                table.flush()
                rows.append(row)
                if summary["test_rmse"] < best_score:
                    best_score = summary["test_rmse"]
                    shutil.copyfile(trial_dir / "best.pt", output_dir / "best.pt")
    if math.isfinite(best_score):
        print(f"\nBest test RMSE: {best_score:.6f} V")
        print("Best checkpoint:", output_dir / "best.pt")
    else:
        print("\nNo finite trained checkpoint was found. Inspect the trial histories.")
    print("Experiment table:", output_dir / "results.csv")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="MAT path; default is beside data.py")
    parser.add_argument("--activation", choices=ACTIVATIONS, default="tanh")
    parser.add_argument("--depth", type=int, choices=DEPTHS, default=1)
    parser.add_argument("--width", type=int, choices=WIDTHS, default=16)
    parser.add_argument("--all-architectures", action="store_true",
                        help="Train the four candidate architectures from model.py")
    parser.add_argument("--tune", action="store_true", help="Use the four preset training plans")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--steps-per-stage", type=int, default=None)
    parser.add_argument("--rollout-lengths", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SETTINGS["batch_size"])
    parser.add_argument("--eval-every", type=int, default=DEFAULT_SETTINGS["eval_every"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SETTINGS["seed"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    custom = {key: value for key, value in {
        "optimizer": args.optimizer, "lr": args.lr,
        "steps_per_stage": args.steps_per_stage,
        "rollout_lengths": args.rollout_lengths}.items() if value is not None}
    if args.tune and custom:
        parser.error("Use either --tune or custom optimizer/rate/steps/length settings.")
    common = {"batch_size": args.batch_size, "eval_every": args.eval_every, "seed": args.seed}
    plans = [{**plan, **common} for plan in TUNING_PLANS] if args.tune else [{**custom, **common}]
    architectures = architecture_configs() if args.all_architectures else [
        {"activation": args.activation, "depth": args.depth, "width": args.width}]
    output_dir = args.output_dir or str(Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S"))
    # Tiny dense layers usually benefit from avoiding CPU thread overhead.
    torch.set_num_threads(1)
    run_experiments(load_data(args.data), architectures, plans, output_dir)


if __name__ == "__main__":
    main()
