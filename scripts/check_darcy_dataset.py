import json
from pathlib import Path

import numpy as np

from operatorbench.data_generation.darcy_dataset import build_dataset
from operatorbench.data_generation.darcy_fd import solve_darcy_fd
from operatorbench.data_generation.forcing import constant_source


def main():
    # 1. Settings for a small dataset check.
    n_samples = 10
    N = 85
    seed = 2022

    # 2. Generate the dataset.
    f_grid = constant_source(N, value=1.0)

    A, F, U = build_dataset(
        n_samples=n_samples,
        N=N,
        seed=seed,
        split_name="smoke",
        f_grid=f_grid,
    )

    # 3. Check shapes, dtypes, and finite values.
    expected_shape = (n_samples, N, N)

    for name, array in {"A": A, "F": F, "U": U}.items():
        assert array.shape == expected_shape, (name, array.shape)
        assert array.dtype == np.float32, (name, array.dtype)
        assert np.isfinite(array).all(), name

    assert np.isin(A, [3.0, 12.0]).all()
    assert np.all(F == 1.0)

    # Check homogeneous Dirichlet boundary values.
    assert np.all(U[:, 0, :] == 0.0)
    assert np.all(U[:, -1, :] == 0.0)
    assert np.all(U[:, :, 0] == 0.0)
    assert np.all(U[:, :, -1] == 0.0)

    # Check agreement with the scalar-source call for the first sample.
    u_scalar = solve_darcy_fd(A[0], f_value=1.0)

    np.testing.assert_array_equal(
        U[0],
        u_scalar.astype(np.float32),
    )

    print("Dataset shape, dtype, boundary, and scalar-source checks passed.")

    # 4. Coordinates and generation metadata.
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)

    metadata = {
        "schema_version": 1,
        "purpose": "smoke_test",
        "task": "a_to_u",
        "n_samples": n_samples,
        "N": N,
        "seed": seed,
        "domain": [[0.0, 1.0], [0.0, 1.0]],
        "indexing": "ij",
        "coefficient": {
            "generator": "sample_fno_darcy_coefficients",
            "n_modes": N,
            "values": [3.0, 12.0],
        },
        "source": {
            "kind": "constant",
            "value": 1.0,
        },
        "boundary": "homogeneous_dirichlet",
        "solver": {
            "method": "five_point_fd",
            "face_average": "arithmetic",
            "compute_dtype": "float64",
        },
        "field_dtype": "float32",
    }

    arrays = {"A": A, "F": F, "U": U, "x": x, "y": y}

    # 5. Save to the repository's ignored data directory.
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"darcy_binary_f1_N{N}_n{n_samples}_seed{seed}.npz"
    )

    if output_path.exists():
        print(f"Existing file found; checking without overwriting: {output_path}")
    else:
        with output_path.open("xb") as file:
            np.savez_compressed(
                file,
                **arrays,
                metadata_json=json.dumps(metadata, sort_keys=True),
            )
        print(f"Saved dataset: {output_path}")

    # 6. Load the file and compare every array and its metadata.
    with np.load(output_path, allow_pickle=False) as cached:
        expected_keys = set(arrays) | {"metadata_json"}
        assert set(cached.files) == expected_keys

        for name, expected in arrays.items():
            restored = cached[name]

            np.testing.assert_array_equal(restored, expected)
            assert restored.dtype == expected.dtype, name

        restored_metadata = json.loads(cached["metadata_json"].item())
        assert restored_metadata == metadata

        print("Reloaded A shape:", cached["A"].shape)
        print("Reloaded F shape:", cached["F"].shape)
        print("Reloaded U shape:", cached["U"].shape)

    print("Dataset generation and save/load checks passed.")


if __name__ == "__main__":
    main()