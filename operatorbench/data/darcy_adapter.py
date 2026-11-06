"""Adapt existing Darcy split arrays without solving or normalizing again."""

import copy
import json
from pathlib import Path

import numpy as np

from .schema import SCHEMA_VERSION, SPLITS, validate_operator_dataset


def from_darcy_arrays(data, metadata, dataset_id):
    """Convert A/F/U split arrays to common fields on their original grid."""
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("Provide an explicit dataset ID.")
    task = metadata.get("task")
    if task not in {"a_to_u", "f_to_u"}:
        raise ValueError("Darcy adapter supports a_to_u and f_to_u.")
    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0):
        raise ValueError("Expected increasing one-dimensional coordinate arrays.")
    grid_shape = (int(x.size), int(y.size))
    splits = {}
    for split in SPLITS:
        count = data[f"A_{split}"].shape[0]
        splits[split] = np.asarray([f"{dataset_id}:{split}:{i}" for i in range(count)])
        for letter in ("A", "F", "U"):
            if data[f"{letter}_{split}"].shape != (count, *grid_shape):
                raise ValueError(f"Invalid {letter}_{split} shape.")
        if np.any(data[f"A_{split}"] <= 0):
            raise ValueError("Darcy coefficients must be positive.")
        u = data[f"U_{split}"]
        if np.any(u[:, [0, -1], :] != 0) or np.any(u[:, :, [0, -1]] != 0):
            raise ValueError("Reference fields must satisfy the declared zero Dirichlet boundary.")
    fields = {}
    for letter, name, quantity in [("A", "a", "permeability"), ("F", "f", "source"), ("U", "u", "solution")]:
        values = np.concatenate([data[f"{letter}_{s}"] for s in SPLITS], axis=0)
        fields[name] = {"values": values.reshape(values.shape[0], 1, int(np.prod(grid_shape)), 1),
                        "time": None, "components": [name], "quantity": quantity, "unit": None}
    space = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
    inputs, fixed = (["a"], ["f"]) if task == "a_to_u" else (["f"], ["a"])
    boundary = metadata.get("boundary")
    if boundary != "homogeneous_dirichlet":
        raise ValueError("This Darcy adapter currently supports zero Dirichlet data only.")
    result = {
        "schema_version": SCHEMA_VERSION,
        "problem": {"pde": "darcy", "equation": "-div(a grad u) = f", "domain": metadata["domain"]},
        "sample_ids": np.concatenate(list(splits.values())),
        "fields": fields,
        "coordinates": {"space": space, "grid_shape": grid_shape, "indexing": "ij", "flatten_order": "C"},
        "task": {"name": task, "inputs": inputs, "targets": ["u"], "fixed_fields": fixed},
        "conditions": {"boundary": {"kind": "dirichlet", "value": 0.0},
                       "initial_fields": [], "parameters": {}},
        "splits": splits,
        "provenance": {"reference": copy.deepcopy(metadata.get("solver", {})),
                       "source_metadata": copy.deepcopy(metadata),
                       "sample_identity": "Rows of the supplied source dataset, not cross-resolution physical-case identities.",
                       "normalization": "none; stored values retain their physical scale"},
    }
    return validate_operator_dataset(result)


def load_darcy_as_operator_dataset(path, *, dataset_id, legacy_config=None):
    """Use embedded metadata, or an explicit config for the six-array notebook cache."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as cached:
        data = {k: cached[k] for k in cached.files if k != "metadata_json"}
        if "metadata_json" in cached.files:
            metadata = json.loads(cached["metadata_json"].item())
            if metadata.get("format") != "darcy_split_npz" or metadata.get("schema_version") != 1:
                raise ValueError("Unsupported Darcy source format/version.")
        else:
            if legacy_config is None:
                raise ValueError("Legacy arrays need explicit coordinates/source/solver metadata.")
            metadata = copy.deepcopy(legacy_config)
            metadata["metadata_origin"] = "explicit legacy config; not embedded in original cache"
            if metadata.get("task") != "a_to_u":
                raise ValueError("Six-array legacy input is supported only for a_to_u.")
            nx, ny = data["A_train"].shape[1:]
            domain = metadata["domain"]
            data["x"] = np.linspace(*domain[0], nx)
            data["y"] = np.linspace(*domain[1], ny)
            f_value = metadata["source"]["value"]
            for split in SPLITS:
                data[f"F_{split}"] = np.full_like(data[f"A_{split}"], f_value)
    result = from_darcy_arrays(data, metadata, dataset_id)
    result["provenance"]["source_file"] = str(path.resolve())
    return result
