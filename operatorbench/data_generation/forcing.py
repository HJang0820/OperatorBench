import numpy as np


def constant_source(N, value=1.0):
    """Return an (N, N) float64 array representing f(x, y) = value."""

    if (
        isinstance(N, (bool, np.bool_))
        or not isinstance(N, (int, np.integer))
        or N < 3
    ):
        raise ValueError(
            "N must be an integer greater than or equal to 3."
        )

    source_value = np.asarray(value, dtype=np.float64)

    if source_value.ndim != 0:
        raise ValueError("value must be a scalar.")

    if not np.isfinite(source_value).item():
        raise ValueError("value must be finite.")

    return np.full(
        (N, N),
        source_value.item(),
        dtype=np.float64,
    )

def sine_source(N, amplitude=1.0):
    """Return amplitude * sin(pi*x) * sin(2*pi*y) on a unit-square grid."""

    if (
        isinstance(N, (bool, np.bool_))
        or not isinstance(N, (int, np.integer))
        or N < 3
    ):
        raise ValueError(
            "N must be an integer greater than or equal to 3."
        )

    amplitude_value = np.asarray(amplitude, dtype=np.float64)

    if amplitude_value.ndim != 0:
        raise ValueError("amplitude must be a scalar.")

    if not np.isfinite(amplitude_value).item():
        raise ValueError("amplitude must be finite.")

    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")

    pattern = (
        np.sin(np.pi * X)
        * np.sin(2.0 * np.pi * Y)
    )

    return amplitude_value.item() * pattern