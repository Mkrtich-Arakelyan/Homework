import math

import numpy as np
import torch
from torch import nn


ACTIVATIONS = ("tanh", "relu")
DEPTHS = (1, 2, 3, 4)
WIDTHS = (8, 16, 24, 32)


def architecture_configs():
    """Return four candidates comparing baseline, depth, width, and activation.

    Depth counts hidden layers; all hidden layers have the same width.
    The original ranges above remain available for individual experiments.
    """
    return [
        {"activation": "tanh", "depth": 1, "width": 16},  # Baseline.
        {"activation": "tanh", "depth": 2, "width": 16},  # More depth.
        {"activation": "tanh", "depth": 2, "width": 32},  # More width.
        {"activation": "relu", "depth": 2, "width": 16},  # Other activation.
    ]


def make_scales(data, reference_hz=70.0):
    """Choose fixed scales using training signals and a reference frequency.

    For an oscillation at angular frequency omega, typical velocity and
    acceleration magnitudes are omega*y and omega**2*y. The homework's
    approximately 70 Hz resonance motivates this scale, but the network
    is free to learn different dynamics. No numerical differentiation is
    needed, and no test samples are used to calculate these scales.
    """
    if not math.isfinite(reference_hz) or reference_hz <= 0:
        raise ValueError("reference_hz must be positive and finite.")

    rms = {}
    for name in ("y", "u"):
        signal = np.asarray(data[f"{name}_train"], dtype=np.float64)
        if signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all():
            raise ValueError(f"{name}_train must be a finite, nonempty 1D signal.")
        rms[name] = float(np.sqrt(np.mean(signal**2)))
        if rms[name] <= 0:
            raise ValueError(f"{name}_train must have a positive RMS magnitude.")

    omega = 2 * math.pi * reference_hz
    return {"y": rms["y"], "v": omega * rms["y"],
            "u": rms["u"], "a": omega**2 * rms["y"]}


class AccelerationNet(nn.Module):
    """A dense network mapping physical (y, v, u) to physical acceleration.

    The default is a starting candidate, not a selected best architecture.
    Inputs must be PyTorch tensors with matching shapes, using the model's
    dtype and device. Scalar inputs give a scalar output; batches keep shape.
    Scales are fixed buffers, so they travel and are saved with the model.
    """

    def __init__(self, scales, depth=1, width=16, activation="tanh",
                 dtype=torch.float64, output_init_gain=0.01):
        super().__init__()
        activation = activation.lower()
        if depth not in DEPTHS or width not in WIDTHS or activation not in ACTIVATIONS:
            raise ValueError("Use depth 1-4, width 8/16/24/32, and tanh/relu.")
        if not math.isfinite(output_init_gain) or output_init_gain <= 0:
            raise ValueError("output_init_gain must be positive and finite.")
        values = [float(scales[name]) for name in ("y", "v", "u", "a")]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("All four scales must be positive and finite.")

        self.depth = depth
        self.width = width
        self.activation = activation
        self.register_buffer("input_scales", torch.tensor(values[:3], dtype=dtype))
        self.register_buffer("acceleration_scale", torch.tensor(values[3], dtype=dtype))

        activation_class = nn.Tanh if activation == "tanh" else nn.ReLU
        layers = []
        in_features = 3
        for _ in range(depth):
            layer = nn.Linear(in_features, width, dtype=dtype)
            if activation == "tanh":
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            else:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            layers.extend((layer, activation_class()))
            in_features = width

        # No output activation: acceleration has either sign and no fixed bound.
        output_layer = nn.Linear(width, 1, dtype=dtype)
        # Small nonzero weights limit initial forcing and allow hidden gradients.
        nn.init.xavier_uniform_(output_layer.weight, gain=output_init_gain)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, y, v, u):
        """Evaluate a = s_a * network([y/s_y, v/s_v, u/s_u])."""
        features = torch.stack((y, v, u), dim=-1)
        normalized_inputs = features / self.input_scales
        normalized_acceleration = self.network(normalized_inputs).squeeze(-1)
        return self.acceleration_scale * normalized_acceleration
