from pathlib import Path

import numpy as np
from scipy.io import loadmat


FS = 610.35
DT = 1.0 / FS
START = 42650
N_SAMPLES = 4096
N_TRAIN = 3072


def load_data(file_path=None):
    """Return centered signals, chronological splits, and time information.

    V1 is the applied input voltage u; V2 is the measured output voltage y.
    Arrays are one-dimensional, and t is in seconds from the excerpt's start.
    The means are calculated over all 4096 samples, as the homework requires.
    No shuffling, smoothing, or additional scaling is applied.
    """
    if file_path is None:
        file_path = Path(__file__).with_name("SNLS80mV.mat")

    mat = loadmat(file_path)
    if "V1" not in mat or "V2" not in mat:
        raise ValueError("The dataset must contain V1 (input) and V2 (output).")

    # The MAT file stores each signal as a row; flatten it into a 1D array.
    u_full = np.asarray(mat["V1"], dtype=np.float64).reshape(-1)
    y_full = np.asarray(mat["V2"], dtype=np.float64).reshape(-1)
    if u_full.size != y_full.size or u_full.size < START + N_SAMPLES:
        raise ValueError("Signals must have equal lengths and at least 46746 samples.")

    # Python excludes the stop index: 46746 - 42650 = 4096 samples.
    u = u_full[START : START + N_SAMPLES]
    y = y_full[START : START + N_SAMPLES]
    if not np.isfinite(u).all() or not np.isfinite(y).all():
        raise ValueError("The required excerpt contains missing or infinite values.")

    # Center the entire excerpt BEFORE making the train/test split.
    u_mean = float(u.mean())
    y_mean = float(y.mean())
    u = u - u_mean
    y = y - y_mean
    t = np.arange(N_SAMPLES) * DT
    zero_test_rmse = float(np.sqrt(np.mean(y[N_TRAIN:] ** 2)))

    # Preserve the full arrays for the uninterrupted free-run evaluation.
    return {
        "u": u,
        "y": y,
        "t": t,
        "u_train": u[:N_TRAIN],
        "y_train": y[:N_TRAIN],
        "t_train": t[:N_TRAIN],
        "u_test": u[N_TRAIN:],
        "y_test": y[N_TRAIN:],
        "t_test": t[N_TRAIN:],
        "fs": FS,
        "dt": DT,
        "u_mean": u_mean,
        "y_mean": y_mean,
        "zero_test_rmse": zero_test_rmse,
    }


def plot_data(data, zoom_seconds=0.2, show=True):
    """Plot centered signals and compare their train/test distributions.

    Row 1: the full excerpt, with training and test segments in different colors.
    Row 2: a short interval showing the individual oscillations.
    Row 3: histograms using the same bins and density normalization for both
    segments, so the different sample counts do not distort the comparison.

    Returns the Matplotlib figure. Use show=False to save it without opening
    a plot window. The input data is not modified.
    """
    import matplotlib.pyplot as plt

    if zoom_seconds <= 0:
        raise ValueError("zoom_seconds must be positive.")

    train_color = "tab:blue"
    test_color = "tab:orange"
    t = data["t"]
    zoom = t <= t[0] + zoom_seconds
    split_time = data["t_test"][0]

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle("Silverbox: centered signals and voltage distributions", fontsize=15)

    for column, (name, title) in enumerate((("u", "Input u"), ("y", "Output y"))):
        signal = data[name]
        train = data[f"{name}_train"]
        test = data[f"{name}_test"]

        # Show how the signals evolve and where the test segment begins.
        ax = axes[0, column]
        ax.plot(data["t_train"], train, color=train_color, linewidth=0.8, label="Training")
        ax.plot(data["t_test"], test, color=test_color, linewidth=0.8, label="Test")
        ax.axvline(split_time, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{title}: full excerpt")
        ax.set_xlabel("Time from excerpt start (s)")
        ax.set_ylabel("Voltage (V)")
        ax.legend()
        ax.grid(alpha=0.25)

        # Zoom in because the full excerpt contains hundreds of oscillations.
        ax = axes[1, column]
        ax.plot(t[zoom], signal[zoom], color=train_color, linewidth=1.2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{title}: first {zoom_seconds:g} s")
        ax.set_xlabel("Time from excerpt start (s)")
        ax.set_ylabel("Voltage (V)")
        ax.grid(alpha=0.25)

        # Shared bins make the training and test distributions comparable.
        ax = axes[2, column]
        bins = np.histogram_bin_edges(signal, bins=40)
        ax.hist(train, bins=bins, density=True, alpha=0.5, color=train_color, label="Training")
        ax.hist(test, bins=bins, density=True, alpha=0.5, color=test_color, label="Test")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{title}: voltage distribution")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Probability density (1/V)")
        ax.legend()
        ax.grid(alpha=0.25)

    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    data = load_data()
    print(f"Excerpt: {len(data['u'])} samples")
    print(f"Training: {len(data['u_train'])} samples")
    print(f"Test: {len(data['u_test'])} samples")
    print(f"Sampling interval: {data['dt']:.10f} seconds")
    print(f"Zero-prediction test RMSE: {data['zero_test_rmse']:.8f} V")
    plot_data(data)
