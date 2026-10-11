import numpy as np
import torch

from integrator import rollout


def rmse(measured, predicted):
    """Return RMSE in the signals' units; nonfinite errors receive infinity."""
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if measured.shape != predicted.shape or measured.size == 0:
        raise ValueError("RMSE needs nonempty arrays with matching shapes.")
    with np.errstate(over="ignore", invalid="ignore"):
        score = float(np.sqrt(np.mean((predicted - measured)**2)))
    return score if np.isfinite(score) else float("inf")


def evaluate(model, data):
    """Return full predicted states, train/test RMSE, and the zero baseline.

    data must come from data.load_data(). Tensor precision and device follow
    the model's parameters. Our Linear/Tanh/ReLU networks give the same
    predictions in training and evaluation modes. The split is 3072 training
    samples and 1024 test samples. RMSE values are in volts, without normalization.

    Initial velocity is estimated by the first forward difference. This
    is an approximation, so it must be held fixed across model comparisons.
    If any predicted state is nonfinite, both model scores are infinity
    and finite=False. Predictions are retained to diagnose that failure.
    """
    y = np.asarray(data["y"], dtype=np.float64)
    u_values = np.asarray(data["u"], dtype=np.float64)
    dt = float(data["dt"])
    split = len(data["y_train"])

    reference = next(model.parameters())
    u = torch.as_tensor(u_values, dtype=reference.dtype, device=reference.device)
    v0 = (y[1] - y[0]) / dt
    initial_state = torch.tensor([y[0], v0], dtype=reference.dtype,
                                 device=reference.device)

    # Evaluation needs no gradient graph. The weights are never updated.
    with torch.no_grad():
        states = rollout(model, initial_state, u, dt)

    states = states.detach().cpu().numpy()
    y_pred = states[:, 0]
    finite = bool(np.isfinite(states).all())
    return {
        "states": states,  # Columns are predicted y and v.
        "y_pred": y_pred,
        "train_rmse": rmse(y[:split], y_pred[:split]) if finite else float("inf"),
        "test_rmse": rmse(y[split:], y_pred[split:]) if finite else float("inf"),
        "zero_test_rmse": data["zero_test_rmse"],
        "finite": finite,
    }


def plot_evaluation(data, result, show=True):
    """Plot measured versus predicted test output from the full free run.

    Returns the figure. Use show=False, then fig.savefig(...), to save it
    without opening a plot window. This function performs no new simulation.
    """
    import matplotlib.pyplot as plt

    if not result["finite"]:
        raise ValueError("Cannot plot a rollout containing nonfinite states.")
    split = len(data["y_train"])
    t_test = np.asarray(data["t"])[split:]
    y_test = np.asarray(data["y"])[split:]
    predicted_test = result["y_pred"][split:]

    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(t_test, y_test, color="tab:blue", linewidth=1, label="Measured")
    ax.plot(t_test, predicted_test, color="tab:orange", linewidth=1,
            linestyle="--", label="Predicted free run")
    ax.set_title(f"Silverbox test segment (RMSE = {result['test_rmse']:.5f} V)")
    ax.set_xlabel("Time from excerpt start (s)")
    ax.set_ylabel("Centered output voltage (V)")
    ax.legend()
    ax.grid(alpha=0.25)
    if show:
        plt.show()
    return fig


def plot_diagnostics(data, result, zoom_seconds=0.15, show=True):
    """Plot a test peak in detail and the error across the whole test segment.

    Both panels use the existing uninterrupted prediction; the zoom does
    not restart the simulation. The largest measured test peak selects
    the zoom window so we can inspect behavior at a strong excitation.
    """
    import matplotlib.pyplot as plt

    if not result["finite"]:
        raise ValueError("Cannot plot a rollout containing nonfinite states.")
    if not np.isfinite(zoom_seconds) or zoom_seconds <= 0:
        raise ValueError("zoom_seconds must be positive and finite.")
    split = len(data["y_train"])
    t = np.asarray(data["t"])[split:]
    measured = np.asarray(data["y"])[split:]
    predicted = result["y_pred"][split:]
    peak_time = t[np.argmax(np.abs(measured))]
    left = max(t[0], min(peak_time - zoom_seconds / 2, t[-1] - zoom_seconds))
    zoom = (t >= left) & (t <= left + zoom_seconds)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True)
    axes[0].plot(t[zoom], measured[zoom], color="tab:blue", linewidth=1.6,
                 label="Measured")
    axes[0].plot(t[zoom], predicted[zoom], color="tab:orange", linewidth=1.6,
                 linestyle="--", label="Predicted free run")
    axes[0].set_title("Zoom around the largest measured test peak")
    axes[0].set_ylabel("Centered output voltage (V)")
    axes[0].legend()

    axes[1].plot(t, predicted - measured, color="tab:purple", linewidth=1,
                 label="Predicted - measured")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Prediction error across all 1024 test samples")
    axes[1].set_ylabel("Error (V)")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Time from excerpt start (s)")
        ax.grid(alpha=0.2)
    if show:
        plt.show()
    return fig


def main():
    """Reload a trained checkpoint and create the homework's test plot."""
    import argparse
    from pathlib import Path

    from data import load_data
    from model import AccelerationNet

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a saved best.pt")
    parser.add_argument("--data", default=None, help="MAT path; default is beside data.py")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show", action="store_true", help="Also open the plot windows")
    args = parser.parse_args()

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    # Some checkpoints store the centering means as NumPy float64 scalars.
    # Allow those numeric metadata types while using the weights-only loader.
    allowed = [np._core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dtype = next(iter(saved["model_state"].values())).dtype
    model = AccelerationNet(saved["scales"], **saved["architecture"], dtype=dtype)
    model.load_state_dict(saved["model_state"])
    data = load_data(args.data)
    result = evaluate(model, data)
    if not result["finite"]:
        raise RuntimeError("This checkpoint produced a nonfinite free run.")

    print("Architecture:", saved["architecture"])
    print("Training settings:", saved["settings"])
    print("Best training step:", saved["best_step"])
    print(f"Train free-run RMSE: {result['train_rmse']:.6f} V")
    print(f"Test free-run RMSE: {result['test_rmse']:.6f} V")
    print(f"Zero-prediction test RMSE: {result['zero_test_rmse']:.6f} V")

    output_dir = args.output_dir or args.checkpoint.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = (
        (plot_evaluation(data, result, show=False), "Silverbox_Test_Free_Run.png"),
        (plot_diagnostics(data, result, show=False), "Silverbox_Test_Diagnostics.png"),
    )
    for fig, name in plots:
        path = output_dir / name
        fig.savefig(path, dpi=180)
        print("Saved:", path)
    if args.show:
        plt.show()
    else:
        for fig, _ in plots:
            plt.close(fig)


if __name__ == "__main__":
    main()
