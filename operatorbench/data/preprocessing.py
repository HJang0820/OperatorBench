import numpy as np


def _check_field(array, name):
    """Validate a raw or transformed batch of grid fields."""
    array = np.asarray(array)

    if array.ndim != 3 or any(size == 0 for size in array.shape):
        raise ValueError(
            f"{name}: expected a nonempty (samples, Nx, Ny) array."
        )
    if array.dtype != np.float32:
        raise ValueError(f"{name}: expected float32, got {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contains NaN or infinite values.")
    return array


def select_input_field(data, task, split):
    """Select the raw input batch for one Darcy task and data split."""
    field_by_task = {"a_to_u": "A", "f_to_u": "F"}
    if task not in field_by_task:
        raise ValueError(f"Unsupported task: {task!r}.")
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unsupported split: {split!r}.")

    key = f"{field_by_task[task]}_{split}"
    return _check_field(data[key], key)


def fit_input_normalization(X_train, U_train):
    """Fit global scalar statistics from training input and solution fields."""
    X_train = _check_field(X_train, "X_train")
    U_train = _check_field(U_train, "U_train")
    if X_train.shape != U_train.shape:
        raise ValueError("X_train and U_train must have matching shapes.")

    # Preserve the explicitly float32 formulas in the original notebook.
    input_mean = np.float32(X_train.mean())
    input_std = np.float32(X_train.std() + 1e-12)
    u_scale = np.float32(np.sqrt(np.mean(U_train**2)))

    if not np.isfinite([input_mean, input_std, u_scale]).all():
        raise ValueError("Normalization statistics must be finite.")
    if input_std <= 0.0 or u_scale <= 0.0:
        raise ValueError("input_std and u_scale must be positive.")

    return {
        "input_mean": input_mean,
        "input_std": input_std,
        "u_scale": u_scale,
    }


def fit_task_normalization(data, task):
    """Select the task input and fit statistics using only the train split."""
    X_train = select_input_field(data, task, "train")
    return fit_input_normalization(X_train, data["U_train"])


def normalize_input(X, stats):
    """Apply fixed training input statistics without modifying X in place."""
    X = _check_field(X, "X")
    result = (
        (X - stats["input_mean"]) / stats["input_std"]
    ).astype(np.float32)
    return _check_field(result, "normalized X")


def fit_normalization(A_train, U_train):
    """Preserve the original a-to-u API and checkpoint statistic keys."""
    stats = fit_input_normalization(A_train, U_train)
    return {
        "a_mean": stats["input_mean"],
        "a_std": stats["input_std"],
        "u_scale": stats["u_scale"],
    }


def normalize_coefficient(A, stats):
    """Apply the original a-to-u statistics using their existing keys."""
    return normalize_input(
        A,
        {"input_mean": stats["a_mean"], "input_std": stats["a_std"]},
    )


def scale_solution(U, stats):
    """Divide solution fields by the training solution RMS."""
    U = _check_field(U, "U")
    result = (U / stats["u_scale"]).astype(np.float32)
    return _check_field(result, "scaled U")


def unscale_solution(U_scaled, stats):
    """Restore solution fields to their original physical scale."""
    U_scaled = _check_field(U_scaled, "U_scaled")
    result = (U_scaled * stats["u_scale"]).astype(np.float32)
    return _check_field(result, "restored U")
