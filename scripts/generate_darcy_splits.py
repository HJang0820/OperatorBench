import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from operatorbench.data.datasets import load_darcy_dataset
from operatorbench.data_generation.darcy_dataset import build_dataset
from operatorbench.data_generation.forcing import (
    constant_source, sample_sine_sources,
)


def reference_kind(settings):
    """Use the original numerical solver unless a reference method is specified."""
    return settings.get("reference", {"kind": "five_point_fd"})["kind"]


def validate_settings(settings):
    """Validate the currently supported data-generation configurations."""
    required = {"dataset", "task", "mode", "N", "splits", "coefficient", "source"}
    if not required <= set(settings) or set(settings) - required - {"reference"}:
        raise ValueError("Unexpected or missing data configuration keys.")
    reference = settings.get("reference", {"kind": "five_point_fd"})
    if (
        not isinstance(reference, dict) or set(reference) != {"kind"}
        or reference["kind"] != "five_point_fd"
    ):
        raise ValueError("The public Gaussian-sine example uses reference.kind=five_point_fd.")
    if not isinstance(settings["dataset"], str) or not settings["dataset"].endswith(".npz"):
        raise ValueError("dataset must be an NPZ path.")
    if settings["task"] not in ("a_to_u", "f_to_u"):
        raise ValueError("task must be a_to_u or f_to_u.")
    if settings["mode"] not in ("small", "full"):
        raise ValueError("mode must be small or full; counts are specified in splits.")
    N = settings["N"]
    if type(N) is not int or N < 3:
        raise ValueError("N must be an integer >= 3.")
    if set(settings["splits"]) != {"train", "val", "test"}:
        raise ValueError("splits must contain train, val, and test.")
    seeds = []
    for split in ("train", "val", "test"):
        spec = settings["splits"][split]
        if set(spec) != {"n_samples", "seed"}:
            raise ValueError(f"Invalid split settings: {split}")
        if type(spec["n_samples"]) is not int or spec["n_samples"] < 1:
            raise ValueError(f"Invalid sample count: {split}")
        if type(spec["seed"]) is not int or spec["seed"] < 0:
            raise ValueError(f"Invalid seed: {split}")
        seeds.append(spec["seed"])
    if len(set(seeds)) != 3:
        raise ValueError("Use different generation seeds for all three splits.")

    coefficient, source = settings["coefficient"], settings["source"]
    if settings["task"] == "a_to_u":
        if coefficient != {"kind": "binary"}:
            raise ValueError("a_to_u currently uses binary coefficients with values 3 and 12.")
        if set(source) != {"kind", "value"} or source["kind"] != "constant":
            raise ValueError("a_to_u requires a constant source.")
        numeric = {"source.value": (source["value"], True)}
    else:
        if set(coefficient) != {"kind", "value"} or coefficient["kind"] != "constant":
            raise ValueError("f_to_u currently requires one positive constant coefficient.")
        if source.get("kind") == "random_sine":
            if set(source) != {"kind", "n_modes", "amplitude", "smoothness"}:
                raise ValueError("Unexpected or missing random-sine settings.")
            numeric = {
                "coefficient.value": (coefficient["value"], True),
                "source.amplitude": (source["amplitude"], True),
                "source.smoothness": (source["smoothness"], False),
            }
        else:
            raise ValueError("f_to_u source.kind must be random_sine.")
        K = source["n_modes"]
        if type(K) is not int or not 1 <= K <= N - 2:
            raise ValueError("n_modes must be an integer in [1, N-2].")
    for name, (value, positive) in numeric.items():
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if value < 0 or (positive and value == 0):
            raise ValueError(f"Invalid value for {name}.")


def make_metadata(settings):
    """Keep the existing split-file schema and original a_to_u metadata."""
    coefficient = settings["coefficient"]
    if settings["task"] == "a_to_u":
        coefficient = {
            "generator": "sample_fno_darcy_coefficients",
            "n_modes": settings["N"], "values": [3.0, 12.0],
        }
    solver = {
        "method": "five_point_fd", "face_average": "arithmetic",
        "compute_dtype": "float64",
    }
    return {
        "format": "darcy_split_npz", "schema_version": 1,
        "mode": settings["mode"], "task": settings["task"],
        "N": settings["N"], "splits": settings["splits"],
        "domain": [[0.0, 1.0], [0.0, 1.0]], "indexing": "ij",
        "coefficient": coefficient, "source": settings["source"],
        "boundary": "homogeneous_dirichlet",
        "solver": solver,
        "field_dtype": "float32",
    }


def generate_split_data(settings):
    """Generate all splits using the existing source generator and solver."""
    N = settings["N"]
    source = settings["source"]
    data = {"x": np.linspace(0.0, 1.0, N), "y": np.linspace(0.0, 1.0, N)}
    a_fixed = None
    if settings["task"] == "f_to_u":
        a_fixed = np.full((N, N), settings["coefficient"]["value"], dtype=np.float64)

    for split in ("train", "val", "test"):
        spec = settings["splits"][split]
        u_reference = None
        if settings["task"] == "a_to_u":
            f_grid = constant_source(N, value=source["value"])
            coefficient_seed = spec["seed"]
        elif source["kind"] == "random_sine":
            f_grid = sample_sine_sources(
                spec["n_samples"], N, seed=spec["seed"],
                n_modes=source["n_modes"], amplitude=source["amplitude"],
                smoothness=source["smoothness"],
            )
            coefficient_seed = None

        A, F, U = build_dataset(
            n_samples=spec["n_samples"], N=N, seed=coefficient_seed,
            split_name=split, f_grid=f_grid, a_fixed=a_fixed,
            u_reference=u_reference,
        )
        for name, array in (("A", A), ("F", F), ("U", U)):
            data[f"{name}_{split}"] = array
    return data


def check_split_data(data, settings):
    """Check arrays and detect exact duplicate operator inputs across all splits."""
    N = settings["N"]
    coordinate = np.linspace(0.0, 1.0, N)
    np.testing.assert_array_equal(data["x"], coordinate)
    np.testing.assert_array_equal(data["y"], coordinate)
    seen = {}
    for split in ("train", "val", "test"):
        shape = (settings["splits"][split]["n_samples"], N, N)
        for name in ("A", "F", "U"):
            array = data[f"{name}_{split}"]
            if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Invalid array: {name}_{split}")
        A, F, U = (data[f"{name}_{split}"] for name in ("A", "F", "U"))
        if settings["task"] == "a_to_u":
            if not np.isin(A, [3.0, 12.0]).all() or not np.all(F == np.float32(settings["source"]["value"])):
                raise ValueError(f"Invalid binary-coefficient/constant-source data: {split}")
            inputs = A
        else:
            if not np.all(A == np.float32(settings["coefficient"]["value"])):
                raise ValueError(f"The fixed coefficient changed: {split}")
            inputs = F
        if np.any(U[:, [0, -1], :] != 0) or np.any(U[:, :, [0, -1]] != 0):
            raise ValueError(f"Nonzero Dirichlet boundary: {split}")
        if np.any(np.linalg.norm(U.astype(np.float64).reshape(shape[0], -1), axis=1) == 0):
            raise ValueError(f"Zero reference norm; relative L2 is undefined: {split}")
        for i, field in enumerate(inputs):
            digest = hashlib.sha256(field.tobytes(order="C")).digest()
            if digest in seen:
                raise ValueError(f"Repeated operator input: {split}[{i}] matches {seen[digest]}.")
            seen[digest] = (split, i)


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate or load configured Darcy dataset splits.")
    parser.add_argument("--config", type=Path, default=project_root / "configs" / "darcy_data_a_to_u.json")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        settings = json.load(file)
    validate_settings(settings)
    metadata = make_metadata(settings)
    print(f"Task: {settings['task']} | source: {settings['source']['kind']}")
    print(f"Reference method: {metadata['solver']['method']}")
    output_path = (project_root / settings["dataset"]).resolve()
    generated = None

    if output_path.exists():
        print(f"Using existing dataset without regenerating: {output_path}")
    else:
        generated = generate_split_data(settings)
        check_split_data(generated, settings)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("xb") as file:
            np.savez_compressed(file, **generated, metadata_json=json.dumps(metadata, sort_keys=True))
        print(f"Saved dataset: {output_path}")

    data, restored_metadata = load_darcy_dataset(output_path)
    if restored_metadata != metadata:
        raise ValueError("Saved metadata does not match this configuration. Choose a new dataset path.")
    check_split_data(data, settings)
    if generated is not None:
        for key, array in generated.items():
            np.testing.assert_array_equal(data[key], array)
        print("Save/load equality checks passed.")
    for split in ("train", "val", "test"):
        print(f"{split}: A={data[f'A_{split}'].shape}, F={data[f'F_{split}'].shape}, U={data[f'U_{split}'].shape}")
    print("No exact duplicate operator inputs found in or between splits.")
    print("Train/validation/test dataset checks passed.")


if __name__ == "__main__":
    main()
