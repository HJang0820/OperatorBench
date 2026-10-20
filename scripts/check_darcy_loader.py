import json
from pathlib import Path

import numpy as np

from operatorbench.data.datasets import load_darcy_dataset


def main():
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = project_root / "data" / "darcy" / (
        "darcy_binary_f1_v1_full_N85"
        "_train1000_val200_test200_seeds2022-2024-2025.npz"
    )

    data, metadata = load_darcy_dataset(dataset_path)

    # Confirm that this check uses the intended full dataset.
    assert metadata["mode"] == "full"
    assert metadata["task"] == "a_to_u"
    assert metadata["N"] == 85

    expected_counts = {
        "train": 1000,
        "val": 200,
        "test": 200,
    }

    for split, count in expected_counts.items():
        assert metadata["splits"][split]["n_samples"] == count

        for field in ("A", "F", "U"):
            key = f"{field}_{split}"

            assert data[key].shape == (count, 85, 85)

            print(
                f"{key}: "
                f"shape={data[key].shape}, "
                f"dtype={data[key].dtype}"
            )

    # Compare the loader output against a direct read of the saved file.
    with np.load(dataset_path, allow_pickle=False) as cached:
        for key, array in data.items():
            np.testing.assert_array_equal(array, cached[key])
            assert array.dtype == cached[key].dtype, key

        direct_metadata = json.loads(cached["metadata_json"].item())
        assert metadata == direct_metadata

    # Preview aligned samples; this is not a training batch sampler.
    batch_size = 20

    a_batch = data["A_train"][:batch_size]
    f_batch = data["F_train"][:batch_size]
    u_batch = data["U_train"][:batch_size]

    assert (
        a_batch.shape
        == f_batch.shape
        == u_batch.shape
        == (20, 85, 85)
    )

    print(
        "Raw batch preview:",
        a_batch.shape,
        f_batch.shape,
        u_batch.shape,
    )

    print("Direct-file equality checks passed.")
    print("Darcy dataset loader checks passed.")


if __name__ == "__main__":
    main()