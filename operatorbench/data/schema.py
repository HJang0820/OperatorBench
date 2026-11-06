"""Common physical-field schema v0.1.

Field axes are (sample, time, point, component). This version uses one
shared, fixed Cartesian grid per dataset and explicit arrays for every sample.
Static fields have time=None and one time slot, not a physical observation at t=0.
This module does not solve PDEs, normalize data, or choose a neural network.
"""

import copy
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "0.1"
SPLITS = ("train", "val", "test")
AXES = ("sample", "time", "point", "component")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_operator_dataset(dataset):
    """Reject inconsistent physical fields, task roles, coordinates, and splits."""
    required = {"schema_version", "problem", "sample_ids", "fields", "coordinates",
                "task", "conditions", "splits", "provenance"}
    _require(isinstance(dataset, dict) and required <= dataset.keys(),
             "Missing required dataset sections.")
    _require(dataset["schema_version"] == SCHEMA_VERSION, "Unsupported schema version.")
    problem = dataset["problem"]
    _require(isinstance(problem, dict) and bool(problem.get("pde")), "Missing PDE name.")
    ids = np.asarray(dataset["sample_ids"])
    _require(ids.ndim == 1 and ids.size > 0 and ids.dtype.kind in "US", "Invalid sample IDs.")
    id_set = set(ids.tolist())
    _require(len(id_set) == ids.size, "Duplicate sample IDs.")

    grid = dataset["coordinates"]
    _require(isinstance(grid, dict), "Invalid coordinates section.")
    shape = tuple(grid.get("grid_shape", ()))
    _require(bool(shape) and all(type(n) is int and n > 0 for n in shape), "Invalid grid shape.")
    points = np.asarray(grid.get("space"))
    _require(points.shape == (int(np.prod(shape)), len(shape)), "Coordinate/grid shape mismatch.")
    _require(points.dtype.kind == "f" and np.isfinite(points).all(), "Invalid spatial coordinates.")
    _require(grid.get("indexing") == "ij" and grid.get("flatten_order") == "C",
             "Expected ij indexing and C-order flattening.")
    domain = np.asarray(problem.get("domain"), dtype=float)
    _require(domain.shape == (len(shape), 2) and np.isfinite(domain).all()
             and np.all(domain[:, 1] > domain[:, 0]), "Invalid domain.")
    _require(np.all(points >= domain[:, 0] - 1e-12)
             and np.all(points <= domain[:, 1] + 1e-12), "Coordinates outside the domain.")

    fields = dataset["fields"]
    _require(isinstance(fields, dict) and bool(fields), "No physical fields.")
    for name, field in fields.items():
        _require(isinstance(name, str) and name.isidentifier(), "Invalid field name.")
        _require(isinstance(field, dict), f"{name}: invalid field record.")
        values = np.asarray(field.get("values"))
        _require(values.ndim == 4 and values.shape[0] == ids.size
                 and values.shape[2] == points.shape[0]
                 and values.shape[1] > 0 and values.shape[3] > 0,
                 f"{name}: expected (sample, time, point, component).")
        _require(values.dtype.kind == "f" and np.isfinite(values).all(), f"{name}: invalid field values.")
        _require(bool(field.get("quantity")), f"{name}: missing physical quantity.")
        _require("unit" in field and (field["unit"] is None or isinstance(field["unit"], str)),
                 f"{name}: provide a unit or explicit None when unspecified.")
        components = field.get("components", [])
        _require(len(components) == values.shape[3]
                 and all(isinstance(c, str) for c in components), f"{name}: invalid components.")
        _require("time" in field, f"{name}: missing time semantics.")
        times = field["time"]
        if times is None:
            _require(values.shape[1] == 1, f"{name}: static fields must have one time slot.")
        else:
            times = np.asarray(times)
            _require(times.shape == (values.shape[1],) and times.dtype.kind == "f"
                     and np.isfinite(times).all() and np.all(np.diff(times) > 0),
                     f"{name}: invalid time coordinates.")

    task = dataset["task"]
    groups = []
    for role in ("inputs", "targets", "fixed_fields"):
        names = task.get(role)
        _require(isinstance(names, list) and len(names) == len(set(names))
                 and all(n in fields for n in names), f"Invalid task role: {role}.")
        groups.append(set(names))
    _require(bool(groups[0]) and bool(groups[1]), "Inputs and targets must not be empty.")
    _require(not any(groups[i] & groups[j] for i in range(3) for j in range(i)),
             "Input, target, and fixed-field roles must be disjoint.")
    for name in task["fixed_fields"]:
        values = fields[name]["values"]
        _require(np.all(values == values[:1]), f"{name}: declared fixed but varies across samples.")

    conditions = dataset["conditions"]
    _require(isinstance(conditions, dict) and isinstance(conditions.get("parameters"), dict),
             "Invalid conditions/parameters.")
    boundary = conditions.get("boundary")
    _require(isinstance(boundary, dict) and bool(boundary.get("kind")), "Missing boundary definition.")
    if "value_field" in boundary:
        _require(boundary["value_field"] in fields, "Unknown boundary value field.")
    for name in conditions.get("initial_fields", []):
        _require(name in fields and fields[name]["time"] is not None
                 and fields[name]["values"].shape[1] == 1, "Invalid initial-condition field.")

    split_ids = dataset["splits"]
    _require(set(split_ids) == set(SPLITS), "Expected train/val/test splits.")
    seen = set()
    for split in SPLITS:
        selected = np.asarray(split_ids[split])
        _require(selected.ndim == 1 and selected.dtype.kind in "US" and selected.size > 0,
                 f"Invalid split: {split}.")
        current = set(selected.tolist())
        _require(len(current) == selected.size and not (seen & current), "Duplicate/overlapping split IDs.")
        _require(current <= id_set, "Split contains unknown sample IDs.")
        seen.update(current)
    _require(seen == id_set, "Splits do not cover all samples.")
    provenance = dataset["provenance"]
    _require(isinstance(provenance, dict) and isinstance(provenance.get("reference"), dict)
             and bool(provenance["reference"].get("method")), "Missing reference-generation method.")
    return dataset


def restore_grid_field(dataset, name):
    """Recover (sample, time, spatial axes..., component) without normalization."""
    values = dataset["fields"][name]["values"]
    return values.reshape(values.shape[:2] + tuple(dataset["coordinates"]["grid_shape"])
                          + (values.shape[-1],), order="C")


def select_split(dataset, name):
    """Select the same physical sample IDs for every field."""
    lookup = {sample: i for i, sample in enumerate(dataset["sample_ids"].tolist())}
    indices = np.asarray([lookup[x] for x in dataset["splits"][name].tolist()])
    return {field: record["values"][indices] for field, record in dataset["fields"].items()}


def save_operator_dataset(dataset, path):
    """Write a separate common-schema NPZ; never overwrite the legacy source."""
    validate_operator_dataset(dataset)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    metadata = {k: copy.deepcopy(v) for k, v in dataset.items()
                if k not in {"fields", "coordinates", "sample_ids", "splits"}}
    metadata["format"] = "operatorbench_npz"
    metadata["coordinates"] = {k: v for k, v in dataset["coordinates"].items() if k != "space"}
    metadata["fields"] = {}
    arrays = {"space": dataset["coordinates"]["space"], "sample_ids": dataset["sample_ids"]}
    for i, (name, field) in enumerate(dataset["fields"].items()):
        record = {k: copy.deepcopy(v) for k, v in field.items() if k not in {"values", "time"}}
        record["values_key"] = f"field_{i}_values"
        record["time_key"] = None if field["time"] is None else f"field_{i}_time"
        arrays[record["values_key"]] = field["values"]
        if record["time_key"] is not None:
            arrays[record["time_key"]] = field["time"]
        metadata["fields"][name] = record
    for split in SPLITS:
        arrays[f"split_{split}"] = dataset["splits"][split]
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return path


def load_operator_dataset(path):
    """Read physical data only; no generator or PDE solver is called."""
    with np.load(Path(path), allow_pickle=False) as cached:
        metadata = json.loads(cached["metadata_json"].item())
        _require(metadata.pop("format", None) == "operatorbench_npz", "Invalid common dataset format.")
        dataset = metadata
        dataset["sample_ids"] = cached["sample_ids"]
        dataset["coordinates"]["space"] = cached["space"]
        dataset["coordinates"]["grid_shape"] = tuple(dataset["coordinates"]["grid_shape"])
        dataset["splits"] = {s: cached[f"split_{s}"] for s in SPLITS}
        fields = {}
        for name, record in dataset["fields"].items():
            values_key = record.pop("values_key")
            time_key = record.pop("time_key")
            fields[name] = {**record, "values": cached[values_key],
                            "time": None if time_key is None else cached[time_key]}
        dataset["fields"] = fields
    return validate_operator_dataset(dataset)
