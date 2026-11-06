"""Check an existing Darcy file and save/reload a separate schema v0.1 file."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from operatorbench.data.darcy_adapter import load_darcy_as_operator_dataset
from operatorbench.data.schema import load_operator_dataset, save_operator_dataset, select_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--legacy-config")
    parser.add_argument("--report")
    args = parser.parse_args()
    source = Path(args.dataset)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    legacy = None if args.legacy_config is None else json.loads(Path(args.legacy_config).read_text(encoding="utf-8"))
    dataset = load_darcy_as_operator_dataset(source, dataset_id=args.dataset_id, legacy_config=legacy)
    save_operator_dataset(dataset, args.output)
    loaded = load_operator_dataset(args.output)
    with np.load(source, allow_pickle=False) as original:
        for split in ("train", "val", "test"):
            restored = select_split(loaded, split)
            for raw, name in (("A", "a"), ("F", "f"), ("U", "u")):
                if f"{raw}_{split}" in original.files:
                    expected = original[f"{raw}_{split}"]
                    np.testing.assert_array_equal(restored[name].reshape(expected.shape), expected)
    assert before == hashlib.sha256(source.read_bytes()).hexdigest(), "Source file was modified."
    report = {
        "task": loaded["task"],
        "grid_shape": list(loaded["coordinates"]["grid_shape"]),
        "field_shapes": {k: list(v["values"].shape) for k, v in loaded["fields"].items()},
        "split_counts": {k: int(v.size) for k, v in loaded["splits"].items()},
        "source_sha256": before,
        "source_unchanged": True,
        "roundtrip_arrays_exact": True,
        "metadata_origin": loaded["provenance"]["source_metadata"].get("metadata_origin", "embedded"),
        "source_file": str(source.resolve()),
        "common_file": str(Path(args.output).resolve()),
        "scope": "Data/schema checks only; no new PDE solve, training, or Burgers experiment."
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Common schema conversion and save/load checks passed.")


if __name__ == "__main__":
    main()
