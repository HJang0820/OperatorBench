import time

import numpy as np

from operatorbench.data_generation.fields import sample_fno_darcy_coefficients
from operatorbench.data_generation.darcy_fd import solve_darcy_fd
from operatorbench.data_generation.forcing import constant_source


def build_dataset(
    n_samples, N, seed, split_name, *,
    f_grid=None, a_fixed=None, u_reference=None,
):
    """Build Darcy arrays with optional fixed coefficients and per-sample sources.

    f_grid: None (f=1), (N,N), or (n_samples,N,N).
    a_fixed: None (sample binary fields) or a positive (N,N) fixed field.
    u_reference: None (solve by FD) or supplied (n_samples,N,N) reference fields.
    Supplied references may come from analytic formulas or external solvers;
    the caller must pair them with the same coefficients and source fields.
    seed controls coefficient sampling only; supplied source arrays are unchanged.
    """
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

    # Accept either a shared source field or one source field per sample.
    if f_grid is None:
        f_grid = constant_source(N, value=1.0)

    f_grid = np.asarray(f_grid, dtype=np.float64)

    if not np.isfinite(f_grid).all():
        raise ValueError("f_grid must contain only finite values.")

    if f_grid.shape == (N, N):
        F = np.repeat(f_grid[None, :, :], n_samples, axis=0)
    elif f_grid.shape == (n_samples, N, N):
        F = f_grid.copy()
    else:
        raise ValueError(
            f"f_grid must have shape {(N, N)} or {(n_samples, N, N)}."
        )

    # Keep the original binary-field sampling unless a fixed field is supplied.
    if a_fixed is None:
        A = sample_fno_darcy_coefficients(
            n_samples=n_samples,
            N=N,
            seed=seed,
        ).astype(np.float64)
    else:
        a_fixed = np.asarray(a_fixed, dtype=np.float64)
        if a_fixed.shape != (N, N):
            raise ValueError(f"a_fixed must have shape {(N, N)}.")
        if not np.isfinite(a_fixed).all() or np.any(a_fixed <= 0):
            raise ValueError("a_fixed must contain finite positive values.")
        A = np.repeat(a_fixed[None, :, :], n_samples, axis=0)

    if u_reference is None:
        U = np.zeros(A.shape, dtype=np.float64)
        t0 = time.time()
        for i in range(n_samples):
            U[i] = solve_darcy_fd(A[i], f_value=F[i])
            if i == 0 or (i + 1) % 50 == 0 or i + 1 == n_samples:
                print(
                    f"{split_name:>5s} {i + 1:4d}/{n_samples} solved | "
                    f"elapsed {time.time() - t0:.1f} sec"
                )
    else:
        U = np.asarray(u_reference, dtype=np.float64)
        if U.shape != (n_samples, N, N):
            raise ValueError(f"u_reference must have shape {(n_samples, N, N)}.")
        if not np.isfinite(U).all():
            raise ValueError("u_reference must contain only finite values.")
        if np.any(U[:, [0, -1], :] != 0.0) or np.any(U[:, :, [0, -1]] != 0.0):
            raise ValueError("u_reference must satisfy zero Dirichlet boundaries.")
        U = U.copy()
        print(f"{split_name:>5s}: using {n_samples} supplied reference fields.")

    # Preserve the notebook's float64 computation / float32 storage policy.
    return (
        A.astype(np.float32),
        F.astype(np.float32),
        U.astype(np.float32),
    )
