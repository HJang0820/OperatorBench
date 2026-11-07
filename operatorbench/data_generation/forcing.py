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


def sample_sine_sources(
    n_samples, N, seed=None, *, n_modes=4, amplitude=1.0, smoothness=2.0
):
    """Sample smooth source fields on an ij-indexed unit-square grid.

    f_i(x,y) = amplitude * sum_pq xi_i,pq * (p^2+q^2)^(-smoothness/2)
               * sin(pi*p*x) * sin(pi*q*y), with independent xi ~ N(0,1).
    Returns a float64 array with shape (n_samples, N, N).
    """
    for name, value, minimum in (
        ("n_samples", n_samples, 1), ("N", N, 3), ("n_modes", n_modes, 1)
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}.")

    if n_modes > N - 2:
        raise ValueError("n_modes must be <= N - 2 on this grid.")

    parameters = {}
    for name, value in (("amplitude", amplitude), ("smoothness", smoothness)):
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 0 or not np.isfinite(value).item() or value.item() < 0:
            raise ValueError(f"{name} must be a finite nonnegative scalar.")
        parameters[name] = value.item()

    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, N)
    k = np.arange(1, n_modes + 1, dtype=np.float64)
    Phi = np.sin(np.pi * x[:, None] * k[None, :])
    k_squared = k[:, None] ** 2 + k[None, :] ** 2
    std = k_squared ** (-0.5 * parameters["smoothness"])

    F = np.empty((n_samples, N, N), dtype=np.float64)
    for i in range(n_samples):
        xi = rng.standard_normal((n_modes, n_modes))
        coefficients = parameters["amplitude"] * xi * std
        F[i] = Phi @ coefficients @ Phi.T

    return F
