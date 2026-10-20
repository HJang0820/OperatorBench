import numpy as np


def _check_field(array, name):
    """Validate a raw or transformed batch of grid fields."""
    array = np.asarray(array)

    if array.ndim != 3 or any(size == 0 for size in array.shape):
        raise ValueError(
            f"{name}: expected a nonempty (samples, Nx, Ny) array."
        )

    if array.dtype != np.float32:
        raise ValueError(
            f"{name}: expected float32, got {array.dtype}."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name}: contains NaN or infinite values."
        )

    return array


def fit_normalization(A_train, U_train):
    """Compute shared normalization statistics using training data only."""
    A_train = _check_field(A_train, "A_train")
    U_train = _check_field(U_train, "U_train")

    if A_train.shape != U_train.shape:
        raise ValueError(
            "A_train and U_train must have matching shapes."
        )

    # Preserve the explicitly float32 formulas in the DeepONet notebook.
    a_mean = np.float32(A_train.mean())
    a_std = np.float32(A_train.std() + 1e-12)
    u_scale = np.float32(np.sqrt(np.mean(U_train**2)))

    if not np.isfinite([a_mean, a_std, u_scale]).all():
        raise ValueError(
            "Normalization statistics must be finite."
        )

    if a_std <= 0.0 or u_scale <= 0.0:
        raise ValueError(
            "a_std and u_scale must be positive."
        )

    return {
        "a_mean": a_mean,
        "a_std": a_std,
        "u_scale": u_scale,
    }


def normalize_coefficient(A, stats):
    """Apply training coefficient statistics without changing A in place."""
    A = _check_field(A, "A")

    result = (
        (A - stats["a_mean"]) / stats["a_std"]
    ).astype(np.float32)

    return _check_field(result, "normalized A")


def scale_solution(U, stats):
    """Divide solution fields by the training solution RMS."""
    U = _check_field(U, "U")

    result = (U / stats["u_scale"]).astype(np.float32)

    return _check_field(result, "scaled U")


def unscale_solution(U_scaled, stats):
    """Restore solution fields to their original physical scale."""
    U_scaled = _check_field(U_scaled, "U_scaled")

    result = (
        U_scaled * stats["u_scale"]
    ).astype(np.float32)

    return _check_field(result, "restored U")