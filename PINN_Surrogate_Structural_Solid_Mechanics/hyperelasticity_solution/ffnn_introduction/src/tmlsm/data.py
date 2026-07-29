"""Data generation for task 1."""

import numpy as np


def bathtub():
    """Generate data for a bathtub function."""
    x = np.linspace(1, 10, 450)
    y = np.concatenate(
        [
            np.square(x[0:150] - 4) + 1,
            1 + 0.1 * np.sin(np.linspace(0, 3.14, 90)),
            np.ones(60),
            np.square(x[300:450] - 7) + 1,
        ]
    )

    x = x / 10.0
    y = y / 10.0

    x_cal = np.concatenate([x[0:240], x[330:420]])
    y_cal = np.concatenate([y[0:240], y[330:420]])

    return x, y, x_cal, y_cal
