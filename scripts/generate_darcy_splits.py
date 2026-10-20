import hashlib
import json
from pathlib import Path

import numpy as np

from operatorbench.data_generation.darcy_dataset import build_dataset
from operatorbench.data_generation.forcing import constant_source


SMALL_RUN = False # false도 가능


def check_split_data(data, split_config, N):
    """Check the current binary-coefficient, constant-source dataset."""
    coordinate = np.linspace(0.0, 1.0, N)
    np.testing.assert_array_equal(data["x"], coordinate)
    np.testing.assert_array_equal(data["y"], coordinate)

    seen = {}

    for split, settings in split_config.items():
        expected_shape = (settings["n_samples"], N, N)

        for field in ("A", "F", "U"):
            key = f"{field}_{split}"
            array = data[key]

            assert array.shape == expected_shape, (key, array.shape)
            assert array.dtype == np.float32, (key, array.dtype)
            assert np.isfinite(array).all(), key

        A = data[f"A_{split}"]
        F = data[f"F_{split}"]
        U = data[f"U_{split}"]

        assert np.isin(A, [3.0, 12.0]).all(), split
        assert np.all(F == 1.0), split

        assert np.all(U[:, 0, :] == 0.0), split
        assert np.all(U[:, -1, :] == 0.0), split
        assert np.all(U[:, :, 0] == 0.0), split
        assert np.all(U[:, :, -1] == 0.0), split

        for i, a in enumerate(A):
            digest = hashlib.sha256(a.tobytes(order="C")).digest()

            if digest in seen:
                raise ValueError(
                    f"Repeated coefficient field: {split}[{i}] "
                    f"matches {seen[digest]}."
                )

            seen[digest] = (split, i)


def main():
    N = 85
    mode = "small" if SMALL_RUN else "full"

    n_train, n_val, n_test = (
        (10, 2, 2) if SMALL_RUN else (1000, 200, 200)
    )

    split_config = {
        "train": {"n_samples": n_train, "seed": 2022},
        "val": {"n_samples": n_val, "seed": 2024},
        "test": {"n_samples": n_test, "seed": 2025},
    }

    seeds = [settings["seed"] for settings in split_config.values()]

    if len(set(seeds)) != 3:
        raise ValueError("Use a different generation seed for each split.")

    metadata = {
        "format": "darcy_split_npz",
        "schema_version": 1,
        "mode": mode,
        "task": "a_to_u",
        "N": N,
        "splits": split_config,
        "domain": [[0.0, 1.0], [0.0, 1.0]],
        "indexing": "ij",
        "coefficient": {
            "generator": "sample_fno_darcy_coefficients",
            "n_modes": N,
            "values": [3.0, 12.0],
        },
        "source": {"kind": "constant", "value": 1.0},
        "boundary": "homogeneous_dirichlet",
        "solver": {
            "method": "five_point_fd",
            "face_average": "arithmetic",
            "compute_dtype": "float64",
        },
        "field_dtype": "float32",
    }

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "darcy"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_tag = "-".join(str(seed) for seed in seeds)

    output_path = output_dir / (
        f"darcy_binary_f1_v1_{mode}_N{N}"
        f"_train{n_train}_val{n_val}_test{n_test}"
        f"_seeds{seed_tag}.npz"
    )

    expected_keys = {"x", "y"}

    for split in split_config:
        for field in ("A", "F", "U"):
            expected_keys.add(f"{field}_{split}")

    generated = None

    if output_path.exists():
        print(f"Using existing dataset without regenerating: {output_path}")

    else:
        generated = {
            "x": np.linspace(0.0, 1.0, N),
            "y": np.linspace(0.0, 1.0, N),
        }

        f_grid = constant_source(N, value=1.0)

        for split, settings in split_config.items():
            A, F, U = build_dataset(
                n_samples=settings["n_samples"],
                N=N,
                seed=settings["seed"],
                split_name=split,
                f_grid=f_grid,
            )

            generated[f"A_{split}"] = A
            generated[f"F_{split}"] = F
            generated[f"U_{split}"] = U

        check_split_data(generated, split_config, N)

        with output_path.open("xb") as file:
            np.savez_compressed(
                file,
                **generated,
                metadata_json=json.dumps(metadata, sort_keys=True),
            )

        print(f"Saved dataset: {output_path}")

    with np.load(output_path, allow_pickle=False) as cached:
        assert set(cached.files) == expected_keys | {"metadata_json"}

        restored_metadata = json.loads(cached["metadata_json"].item())

        if restored_metadata != metadata:
            raise ValueError(
                "Saved metadata does not match the requested settings."
            )

        data = {key: cached[key] for key in expected_keys}

    check_split_data(data, split_config, N)

    if generated is not None:
        for key in expected_keys:
            np.testing.assert_array_equal(data[key], generated[key])
            assert data[key].dtype == generated[key].dtype, key

        print("Save/load equality checks passed.")

    for split in split_config:
        print(
            f"{split}: "
            f"A={data[f'A_{split}'].shape}, "
            f"F={data[f'F_{split}'].shape}, "
            f"U={data[f'U_{split}'].shape}"
        )

    print("No duplicate coefficient fields found in or between splits.")
    print("Train/validation/test dataset checks passed.")


if __name__ == "__main__":
    main()