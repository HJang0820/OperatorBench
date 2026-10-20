import numpy as np


def _check_fields(array, name):
    array = np.asarray(array)

    if array.ndim != 3 or any(size == 0 for size in array.shape):
        raise ValueError(f"{name}: expected a nonempty (B, Nx, Ny) array.")

    if array.dtype != np.float32:
        raise ValueError(f"{name}: expected float32.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contains NaN or infinite values.")

    return array


def _coordinate_grid(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    for name, axis in (("x", x), ("y", y)):
        if axis.ndim != 1 or axis.size < 2:
            raise ValueError(
                f"{name}: expected a one-dimensional coordinate array."
            )

        if not np.isfinite(axis).all():
            raise ValueError(f"{name}: coordinates must be finite.")

        if axis[0] != 0.0 or axis[-1] != 1.0:
            raise ValueError(f"{name}: expected coordinates spanning [0, 1].")

        if not np.all(np.diff(axis) > 0):
            raise ValueError(
                f"{name}: coordinates must be strictly increasing."
            )

    return np.meshgrid(x, y, indexing="ij")


def make_deeponet_branch(A_normalized):
    """Flatten already-normalized coefficients; do not normalize again."""
    A_normalized = _check_fields(A_normalized, "A_normalized")
    B = A_normalized.shape[0]

    return A_normalized.reshape(B, -1, order="C").copy()


def make_deeponet_target(U_scaled):
    """Flatten already-scaled solution fields."""
    U_scaled = _check_fields(U_scaled, "U_scaled")
    B = U_scaled.shape[0]

    return U_scaled.reshape(B, -1, order="C").copy()


def make_deeponet_coords(x, y):
    """Return physical coordinates and the notebook's trunk coordinates."""
    X, Y = _coordinate_grid(x, y)

    coords01 = np.stack([X, Y], axis=-1).reshape(-1, 2, order="C")
    coords_trunk = (2.0 * coords01 - 1.0).astype(np.float32)

    return coords01, coords_trunk


def make_fno_input(A_normalized, x, y):
    """Build channel-last [a_normalized, x, y] inputs."""
    A_normalized = _check_fields(A_normalized, "A_normalized")
    X, Y = _coordinate_grid(x, y)

    if A_normalized.shape[1:] != X.shape:
        raise ValueError("Coefficient grid and coordinate grid must match.")

    X_batch = np.broadcast_to(X[None, :, :], A_normalized.shape)
    Y_batch = np.broadcast_to(Y[None, :, :], A_normalized.shape)

    return np.stack([A_normalized, X_batch, Y_batch], axis=-1)


def make_fno_target(U_scaled):
    """Add the scalar output-channel axis to already-scaled solutions."""
    U_scaled = _check_fields(U_scaled, "U_scaled")

    return U_scaled[..., None].copy()