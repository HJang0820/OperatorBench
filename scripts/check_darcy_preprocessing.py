from pathlib import Path

import numpy as np

from operatorbench.data.datasets import load_darcy_dataset
from operatorbench.data.preprocessing import (
    fit_normalization,
    normalize_coefficient,
    scale_solution,
    unscale_solution,
)


def check_preprocessing(data):
    A_train = data["A_train"]
    U_train = data["U_train"]

    # Fit once, outside the train/val/test loop.
    stats = fit_normalization(A_train, U_train)
    stats_before = stats.copy()

    # Original notebook formulas for comparison.
    expected_stats = {
        "a_mean": np.float32(A_train.mean()),
        "a_std": np.float32(A_train.std() + 1e-12),
        "u_scale": np.float32(np.sqrt(np.mean(U_train**2))),
    }

    for key in expected_stats:
        assert isinstance(stats[key], np.float32), key

        np.testing.assert_array_equal(
            stats[key],
            expected_stats[key],
        )

        print(f"{key}: {float(stats[key]):.8e}")

    for split in ("train", "val", "test"):
        A = data[f"A_{split}"]
        U = data[f"U_{split}"]

        A_before = A.copy()
        U_before = U.copy()

        A_normalized = normalize_coefficient(A, stats)
        U_scaled = scale_solution(U, stats)

        assert A_normalized.shape == A.shape
        assert U_scaled.shape == U.shape
        assert A_normalized.dtype == np.float32
        assert U_scaled.dtype == np.float32

        # Check against the original notebook formulas.
        expected_A = (
            (A - expected_stats["a_mean"]) / expected_stats["a_std"]
        ).astype(np.float32)

        expected_U = (
            U / expected_stats["u_scale"]
        ).astype(np.float32)

        np.testing.assert_array_equal(A_normalized, expected_A)
        np.testing.assert_array_equal(U_scaled, expected_U)

        # Ensure the raw arrays have not been modified.
        np.testing.assert_array_equal(A, A_before)
        np.testing.assert_array_equal(U, U_before)

        # Check inverse transformations.
        A_restored = (
            A_normalized * stats["a_std"] + stats["a_mean"]
        )

        U_restored = unscale_solution(U_scaled, stats)

        np.testing.assert_allclose(
            A_restored,
            A,
            rtol=1e-6,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            U_restored,
            U,
            rtol=1e-6,
            atol=1e-9,
        )

        # These distribution checks apply only to the training data.
        if split == "train":
            np.testing.assert_allclose(
                A_normalized.mean(),
                0.0,
                rtol=0.0,
                atol=1e-5,
            )

            np.testing.assert_allclose(
                A_normalized.std(),
                1.0,
                rtol=0.0,
                atol=1e-5,
            )

            np.testing.assert_allclose(
                np.sqrt(np.mean(U_scaled**2)),
                1.0,
                rtol=0.0,
                atol=1e-5,
            )

        print(
            f"{split}: formula, shape, raw-data, "
            "and inverse checks passed."
        )

    # Synthetic probes: transforms must use the supplied training statistics.
    A_probe = np.full_like(
        data["A_val"][:1],
        stats["a_mean"] + 2.0 * stats["a_std"],
    )

    U_probe = np.full_like(
        data["U_val"][:1],
        3.0 * stats["u_scale"],
    )

    np.testing.assert_allclose(
        normalize_coefficient(A_probe, stats),
        2.0,
        rtol=0.0,
        atol=2e-6,
    )

    np.testing.assert_allclose(
        scale_solution(U_probe, stats),
        3.0,
        rtol=0.0,
        atol=2e-6,
    )

    assert stats == stats_before

    print("Fixed-statistics application checks passed.")
    print("Darcy preprocessing checks passed.")


def main():
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = project_root / "data" / "darcy" / (
        "darcy_binary_f1_v1_full_N85"
        "_train1000_val200_test200_seeds2022-2024-2025.npz"
    )

    data, metadata = load_darcy_dataset(dataset_path)

    assert metadata["mode"] == "full"
    assert metadata["task"] == "a_to_u"

    check_preprocessing(data)


if __name__ == "__main__":
    main()