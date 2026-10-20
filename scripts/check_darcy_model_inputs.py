from pathlib import Path

import numpy as np

from operatorbench.data.datasets import load_darcy_dataset
from operatorbench.data.preprocessing import (
    fit_normalization,
    normalize_coefficient,
    scale_solution,
)
from operatorbench.data.model_inputs import (
    make_deeponet_branch,
    make_deeponet_target,
    make_deeponet_coords,
    make_fno_input,
    make_fno_target,
)


def check_model_inputs(data):
    # Fit normalization statistics using training data only.
    stats = fit_normalization(data["A_train"], data["U_train"])

    x, y = data["x"], data["y"]
    Nx, Ny = len(x), len(y)
    M = Nx * Ny

    coords01, trunk = make_deeponet_coords(x, y)

    assert coords01.shape == trunk.shape == (M, 2)
    assert coords01.dtype == trunk.dtype == np.float32

    # Reproduce the original notebooks' uniform float32 coordinates.
    grid_x = np.linspace(0.0, 1.0, Nx, dtype=np.float32)
    grid_y = np.linspace(0.0, 1.0, Ny, dtype=np.float32)
    X, Y = np.meshgrid(grid_x, grid_y, indexing="ij")

    expected_coords = np.stack([X, Y], axis=-1).reshape(-1, 2)

    np.testing.assert_array_equal(coords01, expected_coords)
    np.testing.assert_array_equal(trunk, 2.0 * expected_coords - 1.0)

    for split in ("train", "val", "test"):
        A_normalized = normalize_coefficient(data[f"A_{split}"], stats)
        U_scaled = scale_solution(data[f"U_{split}"], stats)
        B = A_normalized.shape[0]

        branch = make_deeponet_branch(A_normalized)
        don_target = make_deeponet_target(U_scaled)
        fno_input = make_fno_input(A_normalized, x, y)
        fno_target = make_fno_target(U_scaled)

        # Check shapes.
        assert branch.shape == don_target.shape == (B, M)
        assert fno_input.shape == (B, Nx, Ny, 3)
        assert fno_target.shape == (B, Nx, Ny, 1)

        # Check dtype and finite values.
        for array in (branch, don_target, fno_input, fno_target):
            assert array.dtype == np.float32
            assert np.isfinite(array).all()

        # Check that flattening preserves spatial ordering.
        np.testing.assert_array_equal(
            branch.reshape(B, Nx, Ny), A_normalized
        )
        np.testing.assert_array_equal(
            don_target.reshape(B, Nx, Ny), U_scaled
        )

        # Check each FNO input channel.
        np.testing.assert_array_equal(fno_input[..., 0], A_normalized)
        np.testing.assert_array_equal(
            fno_input[..., 1], np.broadcast_to(X, (B, Nx, Ny))
        )
        np.testing.assert_array_equal(
            fno_input[..., 2], np.broadcast_to(Y, (B, Nx, Ny))
        )
        np.testing.assert_array_equal(fno_target[..., 0], U_scaled)

        # Both models must receive the same coefficient values and targets.
        np.testing.assert_array_equal(
            branch, fno_input[..., 0].reshape(B, M)
        )
        np.testing.assert_array_equal(
            don_target, fno_target[..., 0].reshape(B, M)
        )

        # Check one off-diagonal spatial index.
        i, j = 1, Ny - 2
        k = i * Ny + j

        np.testing.assert_array_equal(
            coords01[k], np.array([x[i], y[j]], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            branch[:, k], A_normalized[:, i, j]
        )
        np.testing.assert_array_equal(
            don_target[:, k], U_scaled[:, i, j]
        )

        print(
            f"{split}: branch={branch.shape}, trunk={trunk.shape}, "
            f"DeepONet target={don_target.shape}"
        )
        print(
            f"{split}: FNO input={fno_input.shape}, "
            f"FNO target={fno_target.shape}"
        )

    # An asymmetric test array helps detect transpose/order mistakes.
    marker = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    mx = np.array([0.0, 1.0], dtype=np.float64)
    my = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    marker_branch = make_deeponet_branch(marker)
    marker_coords, _ = make_deeponet_coords(mx, my)
    marker_fno = make_fno_input(marker, mx, my)

    np.testing.assert_array_equal(
        marker_branch,
        np.array([[0, 1, 2, 3, 4, 5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        marker_coords,
        np.array(
            [
                [0, 0],
                [0, 0.5],
                [0, 1],
                [1, 0],
                [1, 0.5],
                [1, 1],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        marker_fno[0, 1, 0], [3.0, 1.0, 0.0]
    )

    print("Coordinate ordering and cross-model consistency checks passed.")
    print("Darcy model-input checks passed.")


def main():
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = project_root / "data" / "darcy" / (
        "darcy_binary_f1_v1_full_N85"
        "_train1000_val200_test200_seeds2022-2024-2025.npz"
    )

    data, metadata = load_darcy_dataset(dataset_path)

    assert metadata["mode"] == "full"
    assert metadata["task"] == "a_to_u"
    assert metadata["source"] == {"kind": "constant", "value": 1.0}

    check_model_inputs(data)


if __name__ == "__main__":
    main()