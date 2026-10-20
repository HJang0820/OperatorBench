import json
from pathlib import Path

import numpy as np


def load_darcy_dataset(path):
    """Load raw Darcy split arrays and metadata without generating data."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    splits = ("train", "val", "test")
    array_keys = ["x", "y"]

    for split in splits:
        for field in ("A", "F", "U"):
            array_keys.append(f"{field}_{split}")

    with np.load(path, allow_pickle=False) as cached:
        expected_keys = set(array_keys) | {"metadata_json"}

        if set(cached.files) != expected_keys:
            raise ValueError(
                "Unexpected dataset keys; use a split dataset file."
            )

        metadata = json.loads(cached["metadata_json"].item())

        if not isinstance(metadata, dict):
            raise ValueError("Dataset metadata must be a JSON object.")

        if (
            metadata.get("format") != "darcy_split_npz"
            or metadata.get("schema_version") != 1
        ):
            raise ValueError("Unsupported dataset format or schema version.")

        if (
            metadata.get("domain") != [[0.0, 1.0], [0.0, 1.0]]
            or metadata.get("indexing") != "ij"
        ):
            raise ValueError("Expected unit-square data with ij indexing.")

        data = {key: cached[key] for key in array_keys}

    N = metadata.get("N")

    if type(N) is not int or N < 3:
        raise ValueError(
            "Metadata N must be an integer greater than or equal to 3."
        )

    split_config = metadata.get("splits")

    if (
        not isinstance(split_config, dict)
        or set(split_config) != set(splits)
    ):
        raise ValueError(
            "Metadata must describe train, val, and test splits."
        )

    for split in splits:
        settings = split_config[split]

        if not isinstance(settings, dict):
            raise ValueError(f"Invalid metadata for split: {split}")

        n_samples = settings.get("n_samples")

        if type(n_samples) is not int or n_samples < 1:
            raise ValueError(f"Invalid sample count for split: {split}")

        for field in ("A", "F", "U"):
            key = f"{field}_{split}"
            array = data[key]

            if array.shape != (n_samples, N, N):
                raise ValueError(
                    f"{key}: unexpected shape {array.shape}."
                )

            if array.dtype != np.float32:
                raise ValueError(
                    f"{key}: expected float32, got {array.dtype}."
                )

            if not np.isfinite(array).all():
                raise ValueError(
                    f"{key}: contains NaN or infinite values."
                )

    expected_grid = np.linspace(0.0, 1.0, N)

    for axis in ("x", "y"):
        coordinate = data[axis]

        if coordinate.dtype != np.float64:
            raise ValueError(
                f"{axis}: expected float64 coordinates."
            )

        if not np.array_equal(coordinate, expected_grid):
            raise ValueError(
                f"{axis}: expected a uniform unit-interval grid."
            )

    return data, metadata