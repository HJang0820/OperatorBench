import time

import numpy as np

from operatorbench.data_generation.fields import sample_fno_darcy_coefficients
from operatorbench.data_generation.darcy_fd import solve_darcy_fd
from operatorbench.data_generation.forcing import constant_source


def build_dataset(n_samples, N, seed, split_name, *, f_grid=None):
    """Generate binary-coefficient Darcy samples with one shared source field."""
    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, (int, np.integer))
        or n_samples < 1
    ):
        raise ValueError("n_samples must be a positive integer.")

    if (
        isinstance(N, (bool, np.bool_))
        or not isinstance(N, (int, np.integer))
        or N < 3
    ):
        raise ValueError("N must be an integer greater than or equal to 3.")

    # Use f = 1 unless a full-grid source array is provided.
    if f_grid is None:
        f_grid = constant_source(N, value=1.0)

    f_grid = np.asarray(f_grid, dtype=np.float64)

    if f_grid.shape != (N, N):
        raise ValueError(f"f_grid must have shape {(N, N)}.")

    if not np.isfinite(f_grid).all():
        raise ValueError("f_grid must contain only finite values.")

    # Preserve the coefficient-generation logic from the notebook.
    A = sample_fno_darcy_coefficients(
        n_samples=n_samples,
        N=N,
        seed=seed,
    ).astype(np.float64)

    # Assign the same source field to every sample.
    F = np.repeat(f_grid[None, :, :], n_samples, axis=0)

    U = np.zeros(A.shape, dtype=np.float64)
    t0 = time.time()

    for i in range(n_samples):
        U[i] = solve_darcy_fd(A[i], f_value=F[i])

        if i == 0 or (i + 1) % 50 == 0 or i + 1 == n_samples:
            print(
                f"{split_name:>5s} {i + 1:4d}/{n_samples} solved | "
                f"elapsed {time.time() - t0:.1f} sec"
            )

    # Preserve the notebook's float64 computation / float32 storage policy.
    return (
        A.astype(np.float32),
        F.astype(np.float32),
        U.astype(np.float32),
    )